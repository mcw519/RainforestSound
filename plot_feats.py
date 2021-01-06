# Plot feature image and grad-CAM image

# Copyright 2021 (author: Meng Wu)

import argparse
from train import RFCXDatasetLoad, load_ckpt
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from skimage.transform import resize
from model import ResnetMishRFCX, ResnetRFCX, ResNeStMishRFCX


class ShowMachine():
    def __init__(self, feats_path):
        self.dataset = RFCXDatasetLoad(feats_path=feats_path)
    
    def plot(self, idx):
        if isinstance(idx, int):
            uttid, feats, target = self.dataset[idx]
            fig, ax = plt.subplots(1, 3)
            for j in range(3):
                ax[j].imshow(feats[j], origin="lower")
                ax[j].set_title("{}_{}".format(uttid, j))
            
            fig.tight_layout()
            plt.show()
        
        elif isinstance(idx, range):
            fig, ax = plt.subplots(len(idx), 3)
            anchor = 0
            for i in idx:
                uttid, feats, _ = self.dataset[i]
                for j in range(3):
                    ax[anchor][j].imshow(feats[j], origin="lower")
                    ax[anchor][j].set_title("{}_{}".format(uttid, j))
                
                anchor += 1
            fig.tight_layout()
            plt.show()
    
    def CAM(self, idx, model):
        if not isinstance(idx, int):
            raise TypeError

        model = CAMmodel(model)
        model.to("cpu")
        model.eval()

        uttid, feats, target = self.dataset[idx]
        feats.requires_grad=True
        feats.to("cpu")
        label = torch.argmax(target)
        

        feats = feats.unsqueeze(0)
        pred = model(feats)
        pred_label = torch.argmax(pred)
        print("uid:", uttid, "class:", str(label), "pred class:", str(pred_label))
        # get label gradient
        pred[:, label].backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(feats).detach()

        for i in range(pooled_gradients.shape[0]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        heatmap = torch.from_numpy(resize(heatmap, (224, 400)))

        img = feats.detach()
        img = img[0]
        
        fig, ax = plt.subplots(3, 3)
        for i in range(3):
            ax[i][0].imshow(img[i], origin="lower")
            ax[i][0].set_title("original image")
            ax[i][1].imshow(heatmap, origin="lower")
            ax[i][1].set_title("CAM image")
            ax[i][2].imshow(img[i], origin="lower")
            ax[i][2].imshow(heatmap, origin="lower", alpha=0.5, cmap="jet")
            ax[i][2].set_title("What is model seeing")
        fig.tight_layout()
        plt.show()


class CAMmodel(nn.Module):
    def __init__(self, model):
        super().__init__()

        conv_list = list(model.children())[:-1]
        conv_list = conv_list[0][:-1]
        self.features_conv = nn.Sequential(*conv_list)
        max_pool = list(model.children())[:-1]
        max_pool = max_pool[0][-1]
        self.max_pool = nn.Sequential(max_pool)
        self.class_species = nn.Sequential(*(list(model.children())[-1]))

        # placeholder for gradients
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # apply remaining
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.class_species(x)

        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)


def test():
    from model import ResNeStMishRFCX
    from train import load_ckpt

    model = ResNeStMishRFCX(1024, 48)
    load_ckpt("/home/haka/meng/RFCX/exp_ResNeSt/fold3/best_model.pt", model)
    plotter = ShowMachine("/home/haka/meng/RFCX/feats/train_tp.feats")
    plotter.CAM(0, model)


def main(args):
    plotter = ShowMachine(args.feats_path)
    
    if args.CAM and args.ckpt_path is not None:
        if args.from_anti_model:
            outdim = 24*2
        else:
            outdim = 24

        if args.model_type == "ResnetRFCX":
            model = ResnetRFCX(1024, outdim)
        elif args.model_type == "ResnetMishRFCX":
            model = ResnetMishRFCX(1024, outdim)
        elif args.model_type == "ResNeStMishRFCX":
            model = ResNeStMishRFCX(1024, outdim)
        else:
            raise NameError

        load_ckpt(args.ckpt_path, model)
        plotter.CAM(int(args.idx), model)
    
    else:
        plotter.plot(int(args.idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot features")
    parser.add_argument("feats_path", help="featture path")
    parser.add_argument("idx", help="index in dataset")
    parser.add_argument("--CAM", help="plot grad-cam image", default=False, action="store_true")
    parser.add_argument("--ckpt_path", help="checkpoint path", default=None)
    parser.add_argument("--model_type", help="ResnetMishRFCX/ResnetRFCX/ResNeStMishRFCX", default="ResnetMishRFCX")
    parser.add_argument("--from_anti_model", help="model is anti-model", default=False, action="store_true")
    args = parser.parse_args()
    main(args)