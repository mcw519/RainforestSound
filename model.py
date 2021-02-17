# Copyright 2021 (author: Meng Wu)

import torch
import torch.nn as nn
from torch.autograd import Function
from mish import Mish
import torchaudio
from efficientnet_pytorch import EfficientNet


class EMA():
    """
        Exponential moving average
    """
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class RFCXmodel(nn.Module):
    def __init__(self, output_dim, backbone="EfficientNetB0", is_training=False, activation=None):
        super().__init__()

        self.backbone = backbone
        self.is_training = is_training

        if self.is_training:
            self.specaug = SpecAugment(fill_value=0.0, freq_mask_length=20, time_mask_length=20)

        print("Using model type is {}".format(backbone))
        
        if backbone == "EfficientNetB0":
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
            self.class_species = nn.Linear(self.efficient_net._fc.in_features, output_dim)
            self.efficient_net._fc = nn.Identity()
        
        elif backbone == "EfficientNetB1":
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b1')
            self.class_species = nn.Linear(self.efficient_net._fc.in_features, output_dim)
            self.efficient_net._fc = nn.Identity()
        
        elif backbone == "EfficientNetB2":
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b2')
            self.class_species = nn.Linear(self.efficient_net._fc.in_features, output_dim)
            self.efficient_net._fc = nn.Identity()
        
        elif backbone == "ResNeSt50":
            resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
            self.feature = nn.Sequential(*(list(resnet.children())[:-1]))
            self.class_species = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, output_dim)
                )
        
        elif backbone == "ResNeSt101":
            resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
            self.feature = nn.Sequential(*(list(resnet.children())[:-1]))
            self.class_species = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, output_dim)
                )
        
        else:
            raise NameError

        if activation is not None:
            if backbone == "ResNeSt50" or backbone == "ResNeSt101":
                print("With {} active function".format(activation))
                if activation == "SELU" or activation == "selu":
                    replace_relu_to_selu(self.feature)
                    replace_relu_to_selu(self.class_species)
                
                elif activation == "MISH" or activation == "mish":
                    replace_relu_to_mish(self.feature)
                    replace_relu_to_mish(self.class_species)
        
        else:
            print("With origin active function")
    
    def forward(self, x):
        if self.is_training:
            x = self.specaug(x)

        if self.backbone == "ResNeSt50" or self.backbone == "ResNeSt101":
            feature = self.feature(x)
        
        else:
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
        
        return result


def replace_relu_to_selu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.SELU())
        else:
            # recurse
            replace_relu_to_selu(child)


def replace_relu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            # recurse
            replace_relu_to_mish(child)


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

    modelA = RFCXmodel(output_dim=24, backbone="EfficientNetB0", is_training=True, activation="mish")
    modelB = RFCXmodel(output_dim=24, backbone="EfficientNetB1", is_training=True, activation="selu")
    modelC = RFCXmodel(output_dim=24, backbone="ResNeSt50", is_training=True, activation="selu")

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
    test_models()
    test_specaug()