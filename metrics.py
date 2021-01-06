# Copyright 2021 (author: Meng Wu)

from sklearn.metrics import label_ranking_average_precision_score
import torch.nn as nn
import torch.nn.functional as F


def LWLRAP(truth, score):
    """
        This is compute leaderboard score not training criterion.
        y_truch, y_score has shape [1, C]
    """
    y_truth = truth.cpu().detach()
    y_score = score.cpu().detach()       
    y_truth_np = y_truth.detach().numpy()
    y_score_np = y_score.detach().numpy()

    return label_ranking_average_precision_score(y_truth_np, y_score_np)


class F1_loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
            y_pred with shape [N, 24]
            y_true with shape [N, 24], as one-hot represent
        """

        y_pred = F.softmax(y_pred, dim=1)
        tp = (y_true * y_pred).sum(dim=0)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        return 1 - f1.mean()