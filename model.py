# Copyright 2021 (author: Meng Wu)

import torch
import torch.nn as nn
from torch.autograd import Function
from mish import Mish
import torchaudio
from efficientnet_pytorch import EfficientNet


class EfficientNetB0(nn.Module):
    def __init__(self, output_dim, is_training=False):
        super().__init__()
        self.is_training = is_training
        if self.is_training:
            self.specaug = SpecAugment(fill_value=0.0, freq_mask_length=20, time_mask_length=20)

        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.class_species = nn.Linear(self.efficient_net._fc.in_features, output_dim)
        self.efficient_net._fc = nn.Identity()
    
    def forward(self, x):
        if self.is_training:
            x = self.specaug(x)

        feature = self.efficient_net(x)
        species_out = self.class_species(feature)

        return species_out
        

class EnsembleModel(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelE):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE

    def forward(self, x):
        A = self.modelA(x)
        B = self.modelB(x)
        C = self.modelC(x)
        D = self.modelD(x)
        E = self.modelE(x)

        result = (A+B+C+D+E)/5
        result = torch.sigmoid(result)
        
        return result


class ResNeStMishRFCX(nn.Module):
    def __init__(self, hidden_dim, output_dim, is_training=False):
        super().__init__()
        self.is_training = is_training
        if self.is_training:
            self.specaug = SpecAugment(fill_value=0.0, freq_mask_length=20, time_mask_length=20)
        
        resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        self.feature = nn.Sequential(*(list(resnet.children())[:-1]))
        replace_relu_to_mish(self.feature)
        self.class_species = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, x):
        if self.is_training:
            x = self.specaug(x)

        feature = self.feature(x)
        species_out = self.class_species(feature)
        
        return species_out


class ResnetMishRFCX(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        self.feature = nn.Sequential(*(list(resnet.children())[:-1]))
        # replace_relu_to_mish(self.feature)
        self.class_species = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.squeeze(3)
        feature = feature.squeeze(2)
        species_out = self.class_species(feature)
        
        return species_out


def replace_relu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            # recurse
            replace_relu_to_mish(child)


class ResnetRFCX(nn.Module):
    def __init__(self, hidden_dim, output_dim, is_training=False):
        super().__init__()
        self.is_training = is_training
        if self.is_training:
            self.specaug = SpecAugment(fill_value=0.0, freq_mask_length=20, time_mask_length=20)

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
        if self.is_training:
            x = self.specaug(x)

        x = self.resnet(x)

        return x


class SpecAugment(nn.Module):
    def __init__(self, fill_value=0.0, freq_mask_length=10, time_mask_length=10):
        super().__init__()

        self.fill_value = fill_value
        self.freq_mask_length = freq_mask_length
        self.time_mask_length = time_mask_length

    def forward(self, x):
        assert len(x.shape) == 4
        
        N, _, _, _ = x.shape
        temp_list = []

        for i in range(N):
            # mask freq
            temp_feat = torchaudio.functional.mask_along_axis(x[i], mask_param=self.freq_mask_length, mask_value=self.fill_value, axis=1)
            # mask time
            temp_feat = torchaudio.functional.mask_along_axis(temp_feat, mask_param=self.time_mask_length, mask_value=self.fill_value, axis=2)
            temp_list.append(temp_feat)
        
        x = torch.stack(temp_list)

        return x


def test_models():
    from metrics import LWLRAP

    modelA = ResnetRFCX(1024, 24)
    modelB = ResnetMishRFCX(1024, 24)
    modelC = ResNeStMishRFCX(1024, 24)

    x = torch.rand(5, 3, 224, 400)
    
    yA = modelA(x)
    yB = modelB(x)
    yC = modelC(x)


def test_specaug():
    import matplotlib.pyplot as plt
    import pandas as pd
    from dataset import RFCXDataset
    import os

    # x = torch.rand(2, 3, 224, 400)
    rfcx_dir = "/home/haka/meng/RFCX/rfcx"
    train_tp_pd = pd.read_csv(os.path.join(rfcx_dir, "train_tp.csv"))
    dataset = RFCXDataset(manifest_pd=train_tp_pd, feat_type="fbank", data_dir=rfcx_dir)
    _, x1, _ = dataset[0]
    _, x2, _ = dataset[1]
    x = torch.stack([x1[0], x2[0]])
    specaug = SpecAugment(fill_value=0.0, freq_mask_length=20, time_mask_length=20)
    y = specaug(x)
    fig, ax = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            ax[i][j].imshow(y[i][j], origin="lower")
    
    plt.show()


if __name__ == "__main__":
    # test_models()
    test_specaug()