# Copyright 2021 (author: Meng Wu)

from dataset import RFCXDataset
import torchaudio
import torch
from torch.utils.data import DataLoader
import os
from model import ResnetMishRFCX, ResnetRFCX, ResNeStMishRFCX, EnsembleModel, EfficientNetB0
from train import load_ckpt
import math
from skimage.transform import resize
import argparse


class RFCXDatasetEval(RFCXDataset):
    def __init__(self, eval_folder, feat_type="spectrogram", chunk_size=1000):
        self.eval_folder = eval_folder
        self.file_list = self.read_folder()
        self.feat_type = feat_type # spectrogram / mfcc / fbank
        self.feat_config = self.KaldiFeatConfig()
        self.chunk_size = chunk_size
        self.except_length = chunk_size * self.feat_config["frame_shift"] / 1000 # frame_shift is ms
        self.use_resnet = True
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """
            Return:
                recording_id: int
                feats: Tensor with shape [T, C]
        """
        wav_path = self.file_list[idx]
        recording_id = wav_path.strip().split("/")[-1].split(".")[0]
        
        wav, sr = torchaudio.load(wav_path)
        
        num_segment = math.ceil(wav.shape[1] / sr / (self.chunk_size*0.01/2)) # overlap half
        begin = 0
        feat_list = []
        for i in range(num_segment):
            wav_end = begin + int(sr*self.chunk_size*0.01)
            if wav_end <= wav.shape[1]:
                cut_wav = wav[0][begin: wav_end].view(1, -1)

                begin += int(sr*self.chunk_size*0.01/2) # overlap half

                if self.feat_type == "mfcc":
                    feats = self.KaldiMfcc(cut_wav)
            
                elif self.feat_type == "fbank":
                    feats = self.KaldiFbank(cut_wav)
            
                else:
                    feats = self.KaldiSpectrogram(cut_wav)

                """Doing  add-delta first, then resize and normailze to 0-1 for resnet"""
                if self.use_resnet:
                    feats = self.AddDeltaFeatStack(feats)
                    feats = feats.permute(0, 2, 1)
                    resnet_feat_list = []
                    for j in range(3):
                        temp = torch.from_numpy(resize(feats[j], (224, 400)))
                        temp = temp - torch.min(temp)
                        temp = temp / torch.max(temp)
                        resnet_feat_list.append(temp)
                        del temp

                    feats = torch.stack(resnet_feat_list)
                    del resnet_feat_list

                feat_list.append(feats)

        feats = torch.stack(feat_list)
        # should be [N, 3, 224, 400]
        # print(feats.shape)

        return recording_id, feats

    def read_folder(self):
        recording = os.listdir(self.eval_folder)
        for idx, key in enumerate(recording):
            recording[idx] = os.path.join(self.eval_folder, key)
        
        return recording


class RFCXEvalCollateFunc:
    def __init__(self):
        pass
        
    def __call__(self, batch):
        uttid_list = []
        feat_list = []
        output_len_list = []
        for b in batch:
            uttid, feat = b
            
            N, _, _, _ = feat.shape
            
            uttid_list.append(uttid)
            
            feat_list.append(feat)
            
            output_len_list.append(N)
        
        padded_feat = torch.cat(feat_list)
        
        return uttid_list, padded_feat, output_len_list


def get_rfcx_eval_dataloader(dataset: torch.utils.data.Dataset, batch_size=1, shuffle=False, num_workers=0):

    collate_fn = RFCXEvalCollateFunc()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


def SubmitRFCS(args):
    device = "cpu"
    batch_size = 16

    if args.from_anti_model:
        outdim = 24*2
    else:
        outdim = 24

    if args.ensemble:
        model_list = []
        for ckpt in args.ckpt_path, args.ensembleB, args.ensembleC, args.ensembleD, args.ensembleE:
            if args.model_type == "ResnetRFCX":
                model = ResnetRFCX(1024, outdim)
            elif args.model_type == "ResnetMishRFCX":
                model = ResnetMishRFCX(1024, outdim)
            elif args.model_type == "ResNeStMishRFCX":
                model = ResNeStMishRFCX(1024, outdim)
            elif args.model_type == "EfficientNetB0":
                model = EfficientNetB0(outdim)
            else:
                raise NameError
        
            load_ckpt(ckpt, model)
            model.to(device)
            model_list.append(model)
        
        model = EnsembleModel(model_list[0], model_list[1], model_list[2], model_list[3], model_list[4])

    else:
        if args.model_type == "ResnetRFCX":
            model = ResnetRFCX(1024, outdim)
        elif args.model_type == "ResnetMishRFCX":
            model = ResnetMishRFCX(1024, outdim)
        elif args.model_type == "ResNeStMishRFCX":
            model = ResNeStMishRFCX(1024, outdim)
        elif args.model_type == "EfficientNetB0":
            model = EfficientNetB0(outdim)
        else:
            raise NameError

        load_ckpt(args.ckpt_path, model)
    
    model.to(device)
    model.eval()
    
    dataset = RFCXDatasetEval(args.eval_folder, feat_type=args.feature_type, chunk_size=1000)
    dataloader = get_rfcx_eval_dataloader(dataset, batch_size=batch_size, shuffle=False)

    total_file = len(dataset)
    submit_dct = {}
    submit_avg_dct = {}

    for batch_idx, batch in enumerate(dataloader):
        uttid_list, feats, feat_len_list = batch

        feats = feats.to(device)
        
        with torch.no_grad():
            # output is [N, num_class]
            output = model(feats)
        
        num = len(uttid_list)
        first = 0
        
        for i in range(num):
            uttid = uttid_list[i]
            feat_len = feat_len_list[i]
            
            result = output[first:first + feat_len, :] #.split(1, 0)
            if args.ensemble:
                result = result[:, :24]
            else:
                result = torch.sigmoid(result[:, :24])
            
            if result.shape[0] == 1:
                pred_id = result
                pred_id_avg = result
            else:
                pred_id = torch.max(result, dim=0)[0]
                pred_id_avg = result.mean(dim=0)

            first += feat_len
            submit_dct[uttid] = [ str(i) for i in pred_id.tolist() ]
            submit_avg_dct[uttid] = [ str(i) for i in pred_id_avg.tolist() ]

        print("processing {}/{} files".format(len(submit_dct.keys()), total_file))
                
    with open("RFCX_MAX_submit.csv", "w") as f:
        f.writelines("recording_id,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23" + "\n")
        for key in sorted(submit_dct.keys()):
            f.writelines(str(key) + "," + ",".join(submit_dct[key]) + "\n")
    
    with open("RFCX_AVG_submit.csv", "w") as f:
        f.writelines("recording_id,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23" + "\n")
        for key in sorted(submit_avg_dct.keys()):
            f.writelines(str(key) + "," + ",".join(submit_avg_dct[key]) + "\n")


def test_eval_dataloader():
    eval_folder = "/home/haka/meng/Rainforest/rfcx-species-audio-detection/test_split/test/"
    eval_dataset = RFCXDatasetEval(eval_folder, chunk_size=1000)
    eval_dataloader = get_rfcx_eval_dataloader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    i = 0
    for batch in eval_dataloader:
        uttid_list, feats, utt_len_list = batch
        i += 1
        print(uttid_list, feats.shape, utt_len_list)

        if i > 10:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate model")
    parser.add_argument("eval_folder", help="eval dir")
    parser.add_argument("ckpt_path", help="ckpt model path")
    parser.add_argument("--feature_type", help="spectrogram/fbank/mfcc", default="fbank")
    parser.add_argument("--model_type", help="ResnetMishRFCX/ResnetRFCX/ResNeStMishRFCX/EfficientNetB0", default="ResNeStMishRFCX")
    parser.add_argument("--from_anti_model", help="model is anti-model", default=False, action="store_true")
    parser.add_argument("--ensemble", help="do ensemble evaluation", default=False, action="store_true")
    parser.add_argument("--ensembleB", help="ckpt B", default=None)
    parser.add_argument("--ensembleC", help="ckpt C", default=None)
    parser.add_argument("--ensembleD", help="ckpt D", default=None)
    parser.add_argument("--ensembleE", help="ckpt E", default=None)
    args = parser.parse_args()
    SubmitRFCS(args)
