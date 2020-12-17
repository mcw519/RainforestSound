import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import torchaudio
import sox
import random
from skimage.transform import resize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


"""Defined dataset class"""
class RFCXDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_pd=None, feat_type="fbank", num_classess=24, chunk_size=1000, data_dir=None):
        self.data_dir = data_dir
        self.manifest_pd = manifest_pd
        self.feat_type = feat_type
        self.num_classess = num_classess
        self.chunk_size = chunk_size
        self.feat_config = self.KaldiFeatConfig()
        self.except_length = chunk_size * self.feat_config["frame_shift"] / 1000 # frame_shift is ms
        self.use_resnet = True
        self.audio_aug = True

    def __len__(self):
        return len(self.manifest_pd)
    
    def KaldiFeatConfig(self, **kaldi_feats_config):
        """
            Feature config for torchaudio
        """
        kaldi_feats_config.setdefault("channel", 0) # mono
        kaldi_feats_config.setdefault("sample_frequency", 48000 )
        kaldi_feats_config.setdefault("dither", 0.0)
        kaldi_feats_config.setdefault("frame_length", 25.0)
        kaldi_feats_config.setdefault("frame_shift", 10.0)
        kaldi_feats_config.setdefault("preemphasis_coefficient", 0.97)
        kaldi_feats_config.setdefault("snip_edges", False)

        if self.feat_type == "fbank":
            kaldi_feats_config.setdefault("num_mel_bins", 128) # number of mel-banks
        
        if self.feat_type == "mfcc":
            kaldi_feats_config.setdefault("num_mel_bins", 128) # number of mel-banks
            kaldi_feats_config.setdefault("num_ceps", 128) # number of MFCC dimension

        return kaldi_feats_config

    def KaldiSpectrogram(self, wav: torch.Tensor):
        """
            Return Tensor shape is [T, num_fft]
        """
        spectrogram = torchaudio.compliance.kaldi.spectrogram(wav, **self.feat_config)

        return spectrogram
    
    def KaldiFbank(self, wav: torch.Tensor):
        """
            Return Tensor shape is [T, num_mel_bins]
        """
        fbank = torchaudio.compliance.kaldi.fbank(wav, **self.feat_config)

        return fbank
    
    def KaldiMfcc(self, wav: torch.Tensor):
        """
            Return Tensor shape is [T, num_ceps]
        """
        mfcc = torchaudio.compliance.kaldi.mfcc(wav, **self.feat_config)

        return mfcc
    
    def cut_wav(self, wav: torch.Tensor, sr: int, t_min: float, t_max: float, padding: int=0):
        """
            Wav is a Tensor with shape [1, xxx], cutting method worked on samples.
            Args:
                padding: number of samples retain in segment begin/end.
            Return:
                Tensor with shape [1, xxx]
        """
        t_start = max(0, int(t_min * sr - padding))
        if t_start == 0:
            t_end = int(t_max*sr + 2*padding - t_min*sr)
        else:
            t_end = int(t_max*sr + padding)
        cut_wav = wav[0][t_start:t_end+1]

        return cut_wav.view(1, -1)

    def audio_augment(self, wav):
        """
            pysox woking on numpy
            wav is Tensor
        """
        wav = wav[0].numpy()
        tfm = sox.Transformer()
        tfm.set_output_format(rate=self.feat_config["sample_frequency"])
        
        # speed up/ slow down
        speed = random.uniform(0.9, 1.1)
        tfm.speed(speed)

        # volume up/down
        vol = random.uniform(0.125, 2)
        tfm.vol(vol)

        wav = np.array(tfm.build_array(input_array=wav, sample_rate_in=int(self.feat_config["sample_frequency"])))

        wav = torch.from_numpy(wav)
        wav = wav.view(1, -1)

        return wav, speed

    def AddDeltaFeatStack(self, feat: torch.Tensor):
        """
            Adding delta and delta-delta feature in each frame, by stack [3, T, C]
        """
        delta1 = torchaudio.functional.compute_deltas(feat)
        # Normalize to 0...1 - this is what goes into neural net
        delta1 = delta1 - torch.min(delta1)
        delta1 = delta1 / torch.max(delta1)

        delta2 = torchaudio.functional.compute_deltas(delta1)
        # Normalize to 0...1 - this is what goes into neural net
        delta2 = delta2 - torch.min(delta2)
        delta2 = delta2 / torch.max(delta2)

        return torch.stack([feat, delta1, delta2])

    def __getitem__(self, idx: int):
        """
            Return:
                uttid_list
                feats_list
                target_list
        """
        
        recording_id = self.manifest_pd.recording_id.get(idx)
        uttid_list = [recording_id]
        label = self.manifest_pd.species_id.get(idx)
        target = torch.zeros(self.num_classess, dtype=torch.float)
        target[label] = 1.
        target_list = [target]
        t_min = self.manifest_pd.t_min.get(idx)
        t_max = self.manifest_pd.t_max.get(idx)
        f_min = self.manifest_pd.f_min.get(idx)
        f_max = self.manifest_pd.f_max.get(idx)

        """read audio from torchaudio and cut audio in self.chunk size"""
        wav, sr = torchaudio.load("{}/train/{}.flac".format(self.data_dir, recording_id))
        aug_wav, aug_speed = self.audio_augment(wav=wav)

        audio_list = []
        # general audio
        if (t_max - t_min) > self.except_length:
            cut_audio = self.cut_wav(wav, sr, t_min, t_max)
        else:
            padded_s = self.except_length - (t_max - t_min)
            padded_sample = padded_s * sr / 2
            cut_audio = self.cut_wav(wav, sr, t_min, t_max, padding=padded_sample)
        
        audio_list.append(cut_audio)

        """Augment audio"""
        if self.audio_aug:
            t_max = t_max / aug_speed
            t_min = t_min / aug_speed
            if (t_max - t_min) > self.except_length:
                aug_cut_audio = self.cut_wav(aug_wav, sr, t_min, t_max)
            else:
                padded_s = self.except_length - (t_max - t_min)
                padded_sample = padded_s * sr / 2
                aug_cut_audio = self.cut_wav(aug_wav, sr, t_min, t_max, padding=padded_sample)
            
            audio_list.append(aug_cut_audio)
            target_list.append(target)
            uttid_list.append(recording_id+"_aug")
        
        """Extract feature"""
        feats_list = []
        for audio in audio_list:
            if self.feat_type == "mfcc":
                feats = self.KaldiMfcc(audio)
        
            elif self.feat_type == "fbank":
                feats = self.KaldiFbank(audio)
        
            else:
                feats = self.KaldiSpectrogram(audio)

            """Doing  add-delta first, then resize and normailze to 0-1 for resnet"""
            if self.use_resnet:
                feats = self.AddDeltaFeatStack(feats)
                feats = feats.permute(0, 2, 1)
                resnet_feat_list = []
                for i in range(3):
                    temp = torch.from_numpy(resize(feats[i], (224, 400)))
                    temp = temp - torch.min(temp)
                    temp = temp / torch.max(temp)
                    resnet_feat_list.append(temp)
                    del temp

                feats = torch.stack(resnet_feat_list)
                del resnet_feat_list
            feats_list.append(feats)
        
        return uttid_list, feats_list, target_list


class RFCXCollateFunc:
    def __init__(self):
        pass

    def __call__(self, batch):
        col_uttid_list = []
        col_feat_list = []
        col_target_list = []
        for b in batch:
            """
                feature in feat_list with shape [T, C] or [3, 224, 400] for Resnet setting
                target in target_list with shape [C]
            """
            uttid_list, feat_list, target_list = b
                      
            col_uttid_list += uttid_list
            
            col_feat_list += feat_list
            
            col_target_list += target_list

            del feat_list, target_list, b
            
        # batch feats
        padded_feat = pad_sequence(col_feat_list, batch_first=True)
        padded_target = torch.stack(col_target_list).view(-1, 24)

        del col_feat_list, col_target_list
        assert len(col_uttid_list) == padded_feat.shape[0] == padded_target.shape[0]
        
        return col_uttid_list, padded_feat, padded_target


def get_rfcx_dataloader(dataset: torch.utils.data.Dataset, batch_size=1, shuffle=True, num_workers=0):
    
    collate_fn = RFCXCollateFunc()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


def test_dataset():
    import matplotlib.pyplot as plt
    rfcx_dir = "/home/haka/meng/RFCX/rfcx"
    train_tp_pd = pd.read_csv(os.path.join(rfcx_dir, "train_tp.csv"))
    train_fp_pd = pd.read_csv(os.path.join(rfcx_dir, "train_fp.csv"))
    test_pd = pd.read_csv(os.path.join(rfcx_dir, "sample_submission.csv"))

    dataset = RFCXDataset(manifest_pd=train_tp_pd, feat_type="spectrogram", data_dir=rfcx_dir)
    uttid_list, feat_list, target_list = dataset[0]
    
    for i in target_list:
        print(i.shape)

    fig, ax = plt.subplots(2, 3)
    for i in range(len(feat_list)):
        for j in range(3):
            ax[i][j].imshow(feat_list[i][j], origin="lower")
            ax[i][j].set_title("{}_{}".format(uttid_list[i], j))
    
    fig.tight_layout()
    plt.show()


def test_dataloader():
    rfcx_dir = "/home/haka/meng/RFCX/rfcx"
    train_tp_pd = pd.read_csv(os.path.join(rfcx_dir, "train_tp.csv"))
    train_fp_pd = pd.read_csv(os.path.join(rfcx_dir, "train_fp.csv"))
    test_pd = pd.read_csv(os.path.join(rfcx_dir, "sample_submission.csv"))

    dataset = RFCXDataset(manifest_pd=train_tp_pd, feat_type="fbank", data_dir=rfcx_dir)
    dataloader = get_rfcx_dataloader(dataset=dataset, batch_size=2, shuffle=False, num_workers=0)

    i = 0
    for batch in dataloader:
        uttid_list, feats, targets = batch
        i += 1
        print(uttid_list, feats.shape, targets.shape)

        if i > 5:
            break

if __name__ == "__main__":
    # test_dataset()
    test_dataloader()
    
