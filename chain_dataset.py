from numpy.lib import utils
import pandas as pd
import torch
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math


def _add_model_left_right_context(x: torch.Tensor, left_context: int, right_context: int):
    """
        Refer from Kaldi.
        TDNN model needs right-left context information
    """
    padded = x
    if left_context > 0:
        first_frame = x[0, :]
        left_padding = [first_frame] * left_context
        left_padding = torch.stack(left_padding)
        padded = torch.cat([left_padding, x])
    
    if right_context > 0:
        last_frame = x[-1, :]
        right_padding = [last_frame] * right_context
        right_padding = torch.stack(right_padding)
        padded = torch.cat([padded, right_padding])
    
    return padded


class RainforestDataset(torch.utils.data.Dataset):
    def __init__(self, manifests, audio_dir, num_classess, feat_type="spectrogram", cmvn=False, mask=False, **args):
        self.manifests = pd.read_csv(manifests) # train_tp.csv / train_fp.csv
        self.data_dir = audio_dir # audio folder
        self.num_classes = num_classess # 24
        self.feat_type = feat_type # spectrogram / mfcc / fbank
        self.feat_config = self.KaldiFeatConfig(**args)
        self.cmvn = cmvn
        self.mask = mask
    
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
        # print("ori wav length is:", wav.shape)

        y = self.label_to_onehot(label)

        if self.feat_type == "mfcc":
            feats = self.KaldiMfcc(self.cut_wav(wav, sr, t_min, t_max))
        elif self.feat_type == "fbank":
            feats = self.KaldiFbank(self.cut_wav(wav, sr, t_min, t_max))
        else:
            feats = self.KaldiSpectrogram(self.cut_wav(wav, sr, t_min, t_max))
        
        if self.cmvn:
            feats =self.Cmvn(feats)
        
        if self.mask and self.feat_type == "spectrogram":
            feats = self.MaskSpectrogram(feats, f_min, f_max)
        
        return recording_id, feats, y

    def cut_wav(self, wav: torch.Tensor, sr: int, t_min: float, t_max: float, padding: int=0):
        """
            Wav is a Tensor with shape [1, xxx], cutting method worked on samples.
            Args:
                padding: number of samples retain in segment begin/end.
            Return:
                Tensor with shape [1, xxx]
        """
        t_start = int(t_min * sr - padding)
        t_end = int(t_max * sr + padding)
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
        kaldi_feats_config.setdefault("channel", 0) # mono
        kaldi_feats_config.setdefault("sample_frequency", 16000.0 )
        kaldi_feats_config.setdefault("dither", 0.0)
        kaldi_feats_config.setdefault("frame_length", 25.0)
        kaldi_feats_config.setdefault("frame_shift", 10.0)
        kaldi_feats_config.setdefault("preemphasis_coefficient", 0.97)

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
    def __init__(self, model_left_context, model_right_context, frame_subsampling_factor=3, frames_per_chunk=51):
        """
            Refer Kaldi pybind11 recipe.
        """
        self.model_left_context = model_left_context
        self.model_right_context = model_right_context
        self.frame_subsampling_factor = frame_subsampling_factor
        self.frames_per_chunk = frames_per_chunk

    def __call__(self, batch):
        """
            Return:
        """
        uttid_list = []
        feat_list = []
        target_list = []
        output_len_list = []
        subsampled_frames_per_chunk = ( self.frames_per_chunk // self.frame_subsampling_factor )
        
        for b in batch:
            # print("b is:", b)
            uttid, feat, target = b
            uttid_list.append(uttid)
            
            # storage each utterance actually output length (subsampling effect)
            feat_len = feat.size(0)
            output_len = ( feat_len + self.frame_subsampling_factor -1 ) // self.frame_subsampling_factor
            output_len_list.append(output_len)

            # add model left and right context
            feat = _add_model_left_right_context(feat, self.model_left_context, self.model_right_context)
            # Now, feat's shape became [ T + model_left_context + model_right_context, C ]
            
            # split feat to chunk with equal size
            input_num_frames = feat.size(0) - self.model_left_context - self.model_right_context
            for i in range(0, output_len, subsampled_frames_per_chunk):
                first_output = i * self.frame_subsampling_factor
                last_output = min(input_num_frames,
                                                    first_output + (subsampled_frames_per_chunk-1) * self.frame_subsampling_factor)
                
                first_input = first_output
                last_input = last_output + self.model_left_context + self.model_right_context
                input_x = feat[first_input:last_input+1, :]
                feat_list.append(input_x)
                target_list.append(target)
                        
        padded_feat = pad_sequence(feat_list, batch_first=True)
        
        # target = pad_sequence(target_list, batch_first=True)
        target = torch.stack(target_list)
        
        # assert sum([math.ceil(l / subsampled_frames_per_chunk) for l in output_len_list]) == padded_feat.shape[0] == target.shape[0]

        return uttid_list, padded_feat, output_len_list, target


def get_rainforest_dataloader(dataset: torch.utils.data.Dataset, model_left_context, model_right_context, 
                                                            batch_size=1, frames_per_chunk=51, shuffle=True, num_workers=0):

    collate_fn = RainforestCollateFunc(model_left_context=model_left_context, model_right_context=model_right_context, 
                                                                        frame_subsampling_factor=3, frames_per_chunk=frames_per_chunk)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader



def test_show_feats():
    import matplotlib.pyplot as plt
    manifest = "rfcx-species-audio-detection" + "/train_tp.csv"
    audio_dir = "/home/haka/meng/Rainforest/rfcx-species-audio-detection/train"
    dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False)#, feat_type="fbank", num_mel_bins=128)
    dataset_mask = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False, mask=True)#, feat_type="fbank", num_mel_bins=128)
    feats = dataset[0][1]
    mask_feats = dataset_mask[0][1]
    
    plt.subplot(1, 2, 1)
    plt.imshow(feats.T, origin="lower")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask_feats.T, origin="lower")
    plt.colorbar()
    plt.show()


def test_dataloader():
    manifest = "rfcx-species-audio-detection" + "/train_tp.csv"
    audio_dir = "/home/haka/meng/Rainforest/rfcx-species-audio-detection/train"
    dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=True, feat_type="fbank",  num_mel_bins=40)
    dataloader = get_rainforest_dataloader(dataset, model_left_context=17, model_right_context=17, batch_size=3, frames_per_chunk=51)

    i = 0
    for batch in dataloader:
        uttid_list, feats, feat_len_list, target = batch
        i += 1
        print(uttid_list, feats.shape, feat_len_list, target.shape)

        if i > 1:
            break


if __name__ == "__main__":
    # test_show_feats()
    test_dataloader()