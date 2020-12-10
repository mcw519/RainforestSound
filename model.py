import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList


class RainforestModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, num_lstm_layers):
        super().__init__()

        self.LSTM_stack = nn.LSTM(feat_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        
        self.projection = nn.ModuleList([
                                        nn.Linear(hidden_dim, 128),
                                        nn.Linear(128, output_dim)
                                        ])
        
    def forward(self, x):
        # x is [N, chunk_size, C]
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, C)
        
        for i in range(len(self.projection)):
            x = self.projection[i](x)
        
        # only used last output
        x = x[:, -1, :]
        
        return x


def test_model():
    from SID_dataset import RainforestDatasetLoad, get_rainforest_dataloader
    from loss import lwlrap

    feat_dim = 1025
    output_dim = 24
    hidden_dim=256
    num_lstm_layers=5
    model = RainforestModel(feat_dim=feat_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_lstm_layers=num_lstm_layers)
    
    feat_path = "feats/train_tp.feats"
    dataset = RainforestDatasetLoad(feat_path)
    dataloader = get_rainforest_dataloader(dataset, batch_size=1)
    i = 0
    for batch in dataloader:
        uttid_list, feats, target, feat_len_list = batch
        i += 1
        output = model(feats)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        precision = lwlrap(target, output)
        print(loss, precision)

        if i > 10:
            break

if __name__ == "__main__":
    test_model()