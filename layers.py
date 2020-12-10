"""
    Refer from Kaldi pybind11 recipe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout

def _constrain_orthonormal_internal(M):
    
    assert M.ndim == 2
    
    num_rows = M.size(0)
    num_cols = M.size(1)

    assert num_rows <= num_cols

    P = torch.mm(M, M.t())
    P_PT = torch.mm(P, P.t())

    trace_P = torch.trace(P)
    trace_P_P = torch.trace(P_PT)

    scale = torch.sqrt(trace_P_P / trace_P)
    ratio = trace_P_P * num_rows / (trace_P * trace_P)

    assert ratio > 0.99

    update_speed = 0.25

    if ratio > 1.02:
        update_speed *= 0.5
        if ratio > 1.1:
            update_speed *= 0.5
    
    identity = torch.eye(num_rows, dtype=P.dtype, device=P.device)
    P = P - scale * scale * identity

    alpha = update_speed / (scale * scale)

    M = M - 4 * alpha * torch.mm(P, M)

    return M


class SharedDimScaleDropout(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", torch.tensor(0.))

    def forward(self, x, alpha=0.0):
        if self.training and alpha > 0:
            tied_mask_shape = list(x.shape)
            tied_mask_shape[self.dim] = 1
            repeats = [ 1 if i != self.dim else x.shape[self.dim] for i in range(len(x.shape))]

            return x * self.mask.repeat(tied_mask_shape).uniform_(1 - 2*alpha, 1 + 2*alpha).repeat(repeats)
        
        return x

class OrthonormalLinear(nn.Module):

    def __init__(self, dim, bottleneck_dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        # conv requires [N, C, T]
        self.conv = nn.Conv1d(in_channels=dim,
                                                    out_channels=bottleneck_dim,
                                                    kernel_size=kernel_size,
                                                    bias=False)
    
    def forward(self, x):
        x = self.conv(x)

        return x
    
    def constrain_orthonormal(self):
        state_dict = self.conv.state_dict()
        w = state_dict["weight"]

        # w [out_channels, in_channels, kernel_size]
        out_channels = w.size(0)
        in_channels = w.size(1)
        kernel_size = w.size(2)

        w = w.reshape(out_channels, -1)

        num_rows = w.size(0)
        num_cols = w.size(1)

        need_transpose = False
        if num_rows > num_cols:
            w = w.t()
            need_transpose = True
        
        w = _constrain_orthonormal_internal(w)

        if need_transpose:
            w = w.t()
        
        w = w.reshape(out_channels, in_channels, kernel_size)

        state_dict["weight"] = w
        self.conv.load_state_dict(state_dict)


class TDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.affine = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

        self.batchnorm = nn.BatchNorm1d(num_features=hidden_dim, affine=False)

        self.dropout = SharedDimScaleDropout(dim=2)
    
    def forward(self, x, dropout=0.):
        # x is [N, C, T]
        x = self.affine(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x, alpha=dropout)

        return x


class FactorizedTDNN(nn.Module):
    def __init__(self, dim, bottleneck_dim, kernel_size, subsampling_factor, bypass_scale=0.66):
        super().__init__()

        self.bypass_scale = bypass_scale
        self.s = subsampling_factor

        # linear requires [N, C, T]
        self.linear = OrthonormalLinear(dim=dim,
                                                                    bottleneck_dim=bottleneck_dim,
                                                                    kernel_size=kernel_size)

        self.affine = nn.Conv1d(in_channels=bottleneck_dim,
                                                    out_channels=dim,
                                                    kernel_size=1,
                                                    stride=subsampling_factor)
        
        # batchnorm requires [N, C, T]
        self.batchnorm = nn.BatchNorm1d(num_features=dim, affine=False)

        self.dropout = SharedDimScaleDropout(dim=2)
    
    def forward(self, x, dropout=0.):
        # x is [batch_size, feat_dim, seq_len] = [N, C, T]
        assert x.ndim == 3

        # x is [N, C, T]
        input_x = x

        x = self.linear(x)

        x = self.affine(x)

        x = F.relu(x)

        x = self.batchnorm(x)

        x = self.dropout(x, alpha=dropout)

        if self.linear.kernel_size > 1:
            x = self.bypass_scale * input_x[:, :, self.s:-self.s:self.s] + x
        else:
            x = self.bypass_scale * input_x[:, :, ::self.s] + x
        
        return x



def _test_FactorizedTDNN():
    import math
    N = 1
    T = 10
    C = 4

    # kernel_size == 1, subsampling_factor == 1
    model = FactorizedTDNN(dim=C, bottleneck_dim=2, kernel_size=1, subsampling_factor=1)
    x = torch.arange(N * T * C).reshape(N, C, T).float()
    y = model(x)
    print(y.shape)
    assert y.size(2) == T

    # kernel_size == 3, subsampling_factor == 1
    model = FactorizedTDNN(dim=C, bottleneck_dim=2, kernel_size=3, subsampling_factor=1)
    y = model(x)
    print(y.shape)
    assert y.size(2) == T - 2

    # kernel_size == 1, subsampling_factor == 3
    model = FactorizedTDNN(dim=C, bottleneck_dim=2, kernel_size=1, subsampling_factor=3)
    y = model(x)
    print(y.shape)
    assert y.size(2) == math.ceil(math.ceil((T - 3)) - 3)




if __name__ == "__main__":
    _test_FactorizedTDNN()