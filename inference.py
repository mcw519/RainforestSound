import math
import torch
import torch.nn.functional as F
from dataset import RainforestDataset, get_rainforest_dataloader
from train import load_ckpt
from model import RainforestModel
from loss import lwlrap

def compute_accuracy(manifest, audio_dir, model_path):
    device = "cpu"
    model = RainforestModel(feat_dim=120, hidden_dim=256, output_dim=24, num_lstm_layers=3)

    load_ckpt(model_path, model)
    model.to(device)
    model.eval()

    dataset = RainforestDataset(manifests=manifest, audio_dir=audio_dir, num_classess=24,  sample_frequency=48000.0, cmvn=False, chunk=100, feat_type="mfcc",  num_mel_bins=40, num_ceps=40)
    # dataset = RainforestDataset(manifests=manifest, audio_dir=audio_dir, num_classess=24,  sample_frequency=48000.0, cmvn=False, chunk=100, feat_type="fbank",  num_mel_bins=40)
    # dataset = RainforestDataset(manifests=manifest, audio_dir=audio_dir, num_classess=24,  sample_frequency=48000.0, cmvn=False, chunk=100)
    dataloader = get_rainforest_dataloader(dataset, batch_size=1, shuffle=False)

    total = 0
    score = 0

    for batch_idx, batch in enumerate(dataloader):
        uttid_list, feats, target, feat_len_list = batch
        feats = feats.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(feats)
        
        num = len(uttid_list)
        total += num
        first = 0
            
        precision = lwlrap(target, output)
        score += precision
    
    return float(score/total)
            
        
def main(manifest, audio_dir, model):
    result = compute_accuracy(manifest, audio_dir, model)
    print(result)


if __name__ == "__main__":
    import sys
    # main(sys.argv[1], sys.argv[2], sys.argv[3])
    main("cv_tp.csv", "rfcx-species-audio-detection/train", "exp/best_model.pt")