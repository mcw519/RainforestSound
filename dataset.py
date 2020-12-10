import pandas as pd
import torch
from torch.utils import data
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math


class RainforestDatasetLoad(torch.utils.data.Dataset):
    """
        This Dataset using pre-dumped feature, could save memory.
        Before using it, please check dump_feature.py.
    """
    def __init__(self, feats_path):
        
        items = torch.load(feats_path)
        self.items = list(items.values())
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx: int):
        return self.items[idx]


class RainforestDataset(torch.utils.data.Dataset):
    """General RFCX Dataset
        - Feature
            About each feature setting please refer torchaudio.compliance.kaldi.
        - Audio processing
            1. Only used the human annotated species start/end duration
            2. Chunk dependence I/O like Speech Recognition
    """
    def __init__(self, manifests, audio_dir, num_classess, feat_type="spectrogram", cmvn=False, mask=False, chunk=150, **args):
        self.manifests = pd.read_csv(manifests) # train_tp.csv / train_fp.csv
        self.data_dir = audio_dir # audio folder
        self.num_classes = num_classess # 24
        self.feat_type = feat_type # spectrogram / mfcc / fbank
        self.feat_config = self.KaldiFeatConfig(**args)
        self.cmvn = cmvn
        self.mask = mask
        self.chunk = chunk
    
    def __len__(self):
        return len(self.manifests)
    
    def __getitem__(self, idx: int):
        """
            Return:
                recording_id: int
                feats: Tensor with shape [T, C]
                y(label): Tensor with shape [num_classes]
        """
        recording_id = self.manifests.recording_id.get(idx)
        label = self.manifests.species_id.get(idx)
        t_min = self.manifests.t_min.get(idx)
        t_max = self.manifests.t_max.get(idx)
        f_min = self.manifests.f_min.get(idx)
        f_max = self.manifests.f_max.get(idx)

        wav, sr = torchaudio.load("{}/{}.flac".format(self.data_dir, recording_id))

        y = self.label_to_onehot(label)

        if (t_max - t_min) > self.chunk * 0.01:
            cut_audio = self.cut_wav(wav, sr, t_min, t_max)
        else:
            padded_s = self.chunk * 0.01 - (t_max - t_min)
            padded_sample = padded_s * sr
            cut_audio = self.cut_wav(wav, sr, t_min, t_max, padding=padded_sample)
            
        if self.feat_type == "mfcc":
            feats = self.KaldiMfcc(cut_audio)
            feats = self.AddDeltaFeat(feats)
        elif self.feat_type == "fbank":
            feats = self.KaldiFbank(cut_audio)
            feats = self.AddDeltaFeat(feats)
        else:
            feats = self.KaldiSpectrogram(cut_audio)
        
        if self.cmvn:
            feats =self.Cmvn(feats)
        
        if self.mask and self.feat_type == "spectrogram":
            feats = self.MaskSpectrogram(feats, f_min, f_max)
        
        L, C = feats.shape
        num_chunk = L // self.chunk
        feats = feats[:self.chunk * num_chunk, :]
        feats = feats.reshape(-1, self.chunk, C)

        assert feats.shape[0] == num_chunk

        return recording_id, feats, y

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
    
    def label_to_onehot(self, label: int):
        """
            convert to one-hot tensor
        """
        y = torch.zeros(self.num_classes, dtype=int)
        y[label] = 1

        return y
    
    def KaldiFeatConfig(self, **kaldi_feats_config):
        """
            Feature config for torchaudio
        """
        kaldi_feats_config.setdefault("channel", 0) # mono
        kaldi_feats_config.setdefault("sample_frequency", 16000.0 )
        kaldi_feats_config.setdefault("dither", 0.0)
        kaldi_feats_config.setdefault("frame_length", 25.0)
        kaldi_feats_config.setdefault("frame_shift", 10.0)
        kaldi_feats_config.setdefault("preemphasis_coefficient", 0.97)
        kaldi_feats_config.setdefault("snip_edges", False)

        if self.feat_type == "fbank":
            kaldi_feats_config.setdefault("num_mel_bins", 23) # number of mel-banks
        
        if self.feat_type == "mfcc":
            kaldi_feats_config.setdefault("num_mel_bins", 23) # number of mel-banks
            kaldi_feats_config.setdefault("num_ceps", 13) # number of MFCC dimension

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

    def Cmvn(self, feat: torch.Tensor):
        """
            Cepstral Mean and Variance Normalization under sentence level.
            Return:
                Tensor with shape [T, C]
        """
        eps = 1e-10

        if self.feat_type == "spectrogram":
            """
                where [:, 0] is signal_log_energy
            """
            energy_feat = feat[:, 0].view(-1, 1)
            feat = feat[:, 1:]
            mean = feat.mean(0, keepdim=True)
            std = feat.std(0, keepdim=True)
            cmvn_feat = (feat - mean)/(std + eps)
            cmvn_feat = torch.cat((energy_feat, cmvn_feat), 1)

        else:
            """
                fbank/mfcc do not use arg: use_energy
            """
            mean = feat.mean(0, keepdim=True)
            std = feat.std(0, keepdim=True)
            cmvn_feat = (feat - mean)/(std + eps)

        return cmvn_feat
    
    def AddDeltaFeat(self, feat: torch.Tensor):
        """
            Adding delta and delta-delta feature in each frame
        """
        delta1 = torchaudio.functional.compute_deltas(feat)
        delta2 = torchaudio.functional.compute_deltas(delta1)

        return torch.cat([feat, delta1, delta2], dim=-1)

    def MaskSpectrogram(self, feat: torch.Tensor, f_min: float, f_max: float):
        if self.feat_type == "spectrogram":
            mask = torch.zeros_like(feat)
            fft_points = mask.shape[1]
            fft_resolution = self.feat_config["sample_frequency"] / fft_points
            min_bin = round(f_min / fft_resolution)
            max_bin = round(f_max / fft_resolution)
            mask[:, min_bin:max_bin] = 1.
            feat = feat * mask

        return feat


class RainforestCollateFunc:
    def __init__(self):
        pass

    def __call__(self, batch):
        uttid_list = []
        feat_list = []
        target_list = []
        output_len_list = []
        for b in batch:
            uttid, feat, target = b
            
            T, _, _ = feat.shape
            
            uttid_list.append(uttid)
            
            feat_list.append(feat)
            # extend target shape equal to feat's numbers of chunk
            target_list.append(torch.stack([target for _ in range(T)]))
            # record each recording_id's chunk length, for decoding
            output_len_list.append(T)
        
        padded_feat = torch.cat(feat_list)
        padded_target = torch.cat(target_list)

        return uttid_list, padded_feat, padded_target, output_len_list


def get_rainforest_dataloader(dataset: torch.utils.data.Dataset, batch_size=1, shuffle=True, num_workers=0):

    collate_fn = RainforestCollateFunc()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


def test_dataloader():
    manifest = "train_tp.csv"
    audio_dir = "rfcx-species-audio-detection/train"
    dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=True, feat_type="fbank",  num_mel_bins=40, chunk=100)
    dataloader = get_rainforest_dataloader(dataset, batch_size=1)

    i = 0
    for batch in dataloader:
        uttid_list, feats, target, utt_len_list = batch
        i += 1
        print(uttid_list, feats.shape, target.shape, utt_len_list)

        if i > 10:
            break


def test_dataloader_load():
    feat_path = "feats/train_tp.feats"
    dataset = RainforestDatasetLoad(feats_path=feat_path)
    dataloader = get_rainforest_dataloader(dataset, batch_size=1)

    i = 0
    for batch in dataloader:
        uttid_list, feats, target, feat_len_list = batch
        i += 1
        print(uttid_list, feats.shape, target.shape, feat_len_list)

        if i > 5:
            break

if __name__ == "__main__":
    test_dataloader()
    test_dataloader_load()