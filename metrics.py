# Copyright 2021 (author: Meng Wu)

from sklearn.metrics import label_ranking_average_precision_score
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


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



def _one_sample_positive_class_precisions(scores, truth):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]

    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def lwlrap(truth, scores):
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(scores[sample_num, :], truth[sample_num, :])
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(torch.sigmoid(y_pred).cpu().detach().numpy().tolist())

    @property
    def avg(self):
        
        score_class, weight = lwlrap(np.array(self.y_true), np.array(self.y_pred))
        self.score = (score_class * weight).sum()

        return self.score