from dataset import RainforestDataset
import torchaudio
import torch
from torch.utils.data import DataLoader
import os
from model import RainforestModel
from train import load_ckpt


class RainforsetDatasetEval(RainforestDataset):
    def __init__(self, eval_folder, feat_type="spectrogram", cmvn=False, mask=False, chunk=150, **args):
        self.eval_folder = eval_folder
        self.file_list = self.read_folder()
        self.feat_type = feat_type # spectrogram / mfcc / fbank
        self.feat_config = self.KaldiFeatConfig(**args)
        self.cmvn = cmvn
        self.mask = mask
        self.chunk = chunk
    
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
        
        if self.feat_type == "mfcc":
            feats = self.KaldiMfcc(wav)
            feats = self.AddDeltaFeat(feats)
        elif self.feat_type == "fbank":
            feats = self.KaldiFbank(wav)
            feats = self.AddDeltaFeat(feats)
        else:
            feats = self.KaldiSpectrogram(wav)
        
        if self.cmvn:
            feats =self.Cmvn(feats)
        
        L, C = feats.shape
        # print(feats.shape)
        num_chunk = L // self.chunk
        feats = feats[:self.chunk * num_chunk, :]
        feats = feats.reshape(-1, self.chunk, C)

        assert feats.shape[0] == num_chunk

        return recording_id, feats

    def read_folder(self):
        recording = os.listdir(self.eval_folder)
        for idx, key in enumerate(recording):
            recording[idx] = os.path.join(self.eval_folder, key)
        
        return recording


class RainforestEvalCollateFunc:
    def __init__(self):
        pass
        
    def __call__(self, batch):
        uttid_list = []
        feat_list = []
        output_len_list = []
        for b in batch:
            uttid, feat = b
            T, _, _ = feat.shape
            
            uttid_list.append(uttid)
            
            feat_list.append(feat)
            
            output_len_list.append(T)
        
        padded_feat = torch.cat(feat_list)
        
        return uttid_list, padded_feat, output_len_list


def get_rainforest_eval_dataloader(dataset: torch.utils.data.Dataset, batch_size=1, shuffle=False, num_workers=0):

    collate_fn = RainforestEvalCollateFunc()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


def SubmitRFCS(eval_folder, model_path):
    device = "cpu"
    model = RainforestModel(feat_dim=120, hidden_dim=256, output_dim=24, num_lstm_layers=3)

    load_ckpt(model_path, model)
    model.to(device)
    model.eval()

    dataset = RainforsetDatasetEval(eval_folder, sample_frequency=48000.0, cmvn=True, feat_type="fbank",  num_mel_bins=40, chunk=100)
    dataloader = get_rainforest_eval_dataloader(dataset, batch_size=1, shuffle=False)

    submit_dct = {}

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
            pred_id = result.sum(0)
            pred_id = torch.nn.functional.softmax(pred_id, dim=-1)

            first += feat_len
            submit_dct[uttid] = [ str(i) for i in pred_id.tolist() ]
                
    with open("RFCX_submit.csv", "w") as f:
        f.writelines("recording_id,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23" + "\n")
        for key in sorted(submit_dct.keys()):
            f.writelines(str(key) + "," + ",".join(submit_dct[key]) + "\n")


def test_eval_dataloader():
    eval_folder = "/home/haka/meng/Rainforest/rfcx-species-audio-detection/test"
    eval_dataset = RainforsetDatasetEval(eval_folder, sample_frequency=48000.0, cmvn=True, feat_type="fbank",  num_mel_bins=40, chunk=100)
    eval_dataloader = get_rainforest_eval_dataloader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    i = 0
    for batch in eval_dataloader:
        uttid_list, feats, utt_len_list = batch
        i += 1
        print(uttid_list, feats.shape, utt_len_list)

        if i > 10:
            break


if __name__ == "__main__":
    import sys
    # test_eval_dataloader()
    SubmitRFCS(sys.argv[1], sys.argv[2])