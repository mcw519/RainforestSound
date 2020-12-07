"""
    Refer from Kaldi pybind11 recipe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from layers import FactorizedTDNN, TDNN

class RainforestModel(nn.Module):
    def __init__(self,
                            feat_dim,
                            output_dim,
                            hidden_dim=1024,
                            bottleneck_dim=128,
                            kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
                            subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
                            frame_subsampling_factor=3):
        
        super().__init__()

        num_layers = len(kernel_size_list)

        self.ortho_constrain_count = 0

        input_dim = feat_dim

        self.input_batchnorm = nn.BatchNorm1d(num_features=input_dim, affine=False)

        self.tdnn1 = TDNN(input_dim=feat_dim, hidden_dim=hidden_dim)

        tdnnfs = []
        for i in range(num_layers):
            kernel_size = kernel_size_list[i]
            subsampling_factor = subsampling_factor_list[i]
            layer = FactorizedTDNN(dim=hidden_dim, bottleneck_dim=bottleneck_dim, kernel_size=kernel_size, subsampling_factor=subsampling_factor)
            tdnnfs.append(layer)
        
        self.tdnnfs = nn.ModuleList(tdnnfs)

        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    
    def forward(self, x, dropout=0.):
        assert x.ndim == 3

        # x is [N, T, C]
        x = x.permute(0, 2, 1)

        # x is [N, C, T]
        x = self.input_batchnorm(x)

        x = self.tdnn1(x, dropout=dropout)

        for i in range(len(self.tdnnfs)):
            x = self.tdnnfs[i](x, dropout=dropout)
        
        x = x.permute(0, 2, 1)
        
        # x is [N, T, C]
        x = self.output(x)

        return x


def test_model():
    feat_dim = 1025
    output_dim = 24
    # model = RainforestModel(feat_dim=feat_dim, output_dim=24, kernel_size_list=[1, 1, 1], subsampling_factor_list=[1, 1, 1])
    model = RainforestModel(feat_dim=feat_dim, output_dim=24)
    N = 1
    T = 150 + 27 + 27
    C = feat_dim
    x = torch.arange(N * T * C).reshape(N, T, C).float()
    target = torch.zeros(N, output_dim, dtype=int)
    target[0][1] = 1
    output = model(x)

    print(x.shape, output.shape)

    criteron = nn.CrossEntropyLoss()
    loss = criteron(output, target)
    print(loss)

if __name__ == "__main__":
    test_model()
    