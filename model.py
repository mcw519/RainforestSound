import torch
import torch.nn as nn

class ResnetRFCX(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim)
            )
        # only fine-tuned fc layer
        # for para in list(self.resnet.parameters())[:-2]:
        #     para.requires_grad=False
    
    def forward(self, x):
        x = self.resnet(x)

        return x


def test_ResnetRFCX():
    import os
    import pandas as pd
    from dataset import RFCXDataset, get_rfcx_dataloader
    from metrics import LWLRAP

    model = ResnetRFCX(1024, 24)
    print(model.parameters)
    rfcx_dir = "/home/haka/meng/RFCX/rfcx"
    train_tp_pd = pd.read_csv(os.path.join(rfcx_dir, "train_tp.csv"))
    
    dataset = RFCXDataset(manifest_pd=train_tp_pd, feat_type="fbank", data_dir=rfcx_dir)
    dataloader = get_rfcx_dataloader(dataset=dataset, batch_size=2, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    i = 0
    for batch in dataloader:
        uttid_list, feats, targets = batch
        i += 1
        y = model(feats)
        loss = criterion(y, targets)
        print("LWLRAP:", LWLRAP(targets, y))
        print("Loss:", loss)

        if i > 5:
            break


if __name__ == "__main__":
    test_ResnetRFCX()
