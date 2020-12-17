from sklearn.metrics import label_ranking_average_precision_score

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
