import torch
from dataset import RainforestDataset

manifest = "train_tp.csv"
audio_dir = "rfcx-species-audio-detection/train"
# dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False, feat_type="mfcc",  num_mel_bins=40, num_ceps=40, chunk=100)
dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False, feat_type="fbank",  num_mel_bins=40, chunk=100)
# dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False, chunk=100)
dct = {}
for i in range(len(dataset)):
    uttid, feat, target = dataset[i]
    # print(uttid, feat.shape, target.shape)
    dct[uttid] = [uttid, feat, target]

torch.save(dct, "feats/train_tp.feats")

manifest = "cv_tp.csv"
audio_dir = "rfcx-species-audio-detection/train"
# dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False, feat_type="mfcc",  num_mel_bins=40, num_ceps=40, chunk=100)
dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False, feat_type="fbank",  num_mel_bins=40, chunk=100)
# dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False, chunk=100)
dct = {}
for i in range(len(dataset)):
    uttid, feat, target = dataset[i]
    dct[uttid] = [uttid, feat, target]

torch.save(dct, "feats/cv_tp.feats")