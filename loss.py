import torch
import torch.nn as nn
from sklearn.metrics import label_ranking_average_precision_score

def lwlrap(y_truth, y_score):
    y_truth = y_truth.cpu()
    y_score = y_score.cpu()
    if len(y_score.shape) == 3:
        y_score = y_score.sum(1)
    y_truth_np = y_truth.detach().numpy()
    y_score_np = y_score.detach().numpy()

    return label_ranking_average_precision_score(y_truth_np, y_score_np)


