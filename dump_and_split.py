from random import shuffle
from sklearn.model_selection import StratifiedKFold
import torch
from dataset import RFCXDataset
import pandas as pd
import os

def dump_tp():
    rfcx_dir = "/home/haka/meng/RFCX/rfcx"
    train_tp_pd = pd.read_csv(os.path.join(rfcx_dir, "train_tp.csv"))
    dataset = RFCXDataset(manifest_pd=train_tp_pd, feat_type="fbank", data_dir=rfcx_dir)
    num_aug = len(dataset[0][0])

    dct = {}
    idx = 0
    for i in range(len(dataset)):
        uttid_list, feat_list, target_list = dataset[i]
        for uid in range(len(uttid_list)):
            dct[idx] = [uttid_list[uid], feat_list[uid], target_list[uid]]
            idx += 1
        
        print("Finished {}/{} data".format(idx, len(dataset)*num_aug))
    
    if not os.path.isdir("feats"):
        os.makedirs("feats")

    torch.save(dct, "feats/train_tp.feats")


def split_data():
    dataset = torch.load("feats/train_tp.feats")
    items = list(dataset.values())
    feats = []
    labels = []
    for i in range(len(items)):
        feats.append(items[i][1])
        labels.append(torch.argmax(items[i][2]))

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for fold_idx, (train_index, val_index) in enumerate(skf.split(feats, labels)):
        train_dct = {}
        dct_idx = 0
        for j in train_index:
            train_dct[dct_idx] =  [items[j][0], items[j][1], items[j][2]]
            dct_idx += 1
        
        torch.save(train_dct, "feats/train_tp_fold{}.feats".format(fold_idx))

        val_dct = {}
        dct_idx = 0
        for j in val_index:
            val_dct[dct_idx] =  [items[j][0], items[j][1], items[j][2]]
            dct_idx += 1
        
        torch.save(val_dct, "feats/val_tp_fold{}.feats".format(fold_idx))


if __name__ == "__main__":
    dump_tp()
    split_data()
