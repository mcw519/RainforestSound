# This program used to prepare feature for training

# Copyright 2021 (author: Meng Wu)

from sklearn.model_selection import StratifiedKFold
import torch
from dataset import RFCXDataset
import pandas as pd
import os
import argparse
from tqdm import tqdm

class DumpFeature():
    def __init__(self, rfcx_dir, out_dir, feature_type="fbank", kfold=5):
        self.rfcx_dir = rfcx_dir
        self.out_dir = out_dir
        self.feature_type = feature_type
        self.kfold = kfold
        
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        
    def dump_tp(self):
        train_tp_pd = pd.read_csv(os.path.join(self.rfcx_dir, "train_tp.csv"))
        dataset = RFCXDataset(manifest_pd=train_tp_pd, feat_type=self.feature_type, data_dir=self.rfcx_dir)
        num_aug = len(dataset[0][0])
        dct = {}
        idx = 0
        for i in tqdm(range(len(dataset))):
            uttid_list, feat_list, target_list = dataset[i]
            for uid in range(len(uttid_list)):
                dct[idx] = [uttid_list[uid], feat_list[uid], target_list[uid]]
                idx += 1
            
            # print("Finished {}/{} data".format(idx, len(dataset)*num_aug))

        torch.save(dct, os.path.join(self.out_dir, "train_tp.feats"))
        del dataset, train_tp_pd, dct
    
    def dump_fp(self):
        train_fp_pd = pd.read_csv(os.path.join(self.rfcx_dir, "train_fp.csv"))
        dataset = RFCXDataset(manifest_pd=train_fp_pd, feat_type=self.feature_type, data_dir=self.rfcx_dir, audio_aug=False)
        num_aug = len(dataset[0][0])
        dct = {}
        idx = 0
        for i in tqdm(range(len(dataset))):
            uttid_list, feat_list, target_list = dataset[i]
            for uid in range(len(uttid_list)):
                dct[idx] = [uttid_list[uid], feat_list[uid], target_list[uid]]
                idx += 1
            
            # print("Finished {}/{} data".format(idx, len(dataset)*num_aug))
        
        torch.save(dct, os.path.join(self.out_dir, "train_fp.feats"))
        del dataset, train_fp_pd, dct
    
    def split_data(self):
        dataset = torch.load(os.path.join(self.out_dir, "train_tp.feats"))
        items = list(dataset.values())
        feats = []
        labels = []
        for i in range(len(items)):
            feats.append(items[i][1])
            labels.append(torch.argmax(items[i][2]))

        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)
        for fold_idx, (train_index, val_index) in enumerate(skf.split(feats, labels)):
            train_dct = {}
            dct_idx = 0
            for j in train_index:
                train_dct[dct_idx] =  [items[j][0], items[j][1], items[j][2]]
                dct_idx += 1
            
            torch.save(train_dct, "{}/train_tp_fold{}.feats".format(self.out_dir, fold_idx))

            val_dct = {}
            dct_idx = 0
            for j in val_index:
                val_dct[dct_idx] =  [items[j][0], items[j][1], items[j][2]]
                dct_idx += 1
            
            torch.save(val_dct, "{}/val_tp_fold{}.feats".format(self.out_dir, fold_idx))
        
    def run(self):
        print("Start extract {} feature for TP dataset and dump in {}".format(self.feature_type, self.out_dir))
        self.dump_tp()
        print("Runing kFold on TP data")
        self.split_data()
        print("Start extract {} feature for FP dataset and dump in {}".format(self.feature_type, self.out_dir))
        self.dump_fp()
    
    @staticmethod
    def _split_data(feature_path, out_dir, kfold):
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        dataset = torch.load(feature_path)
        items = list(dataset.values())
        feats = []
        labels = []
        for i in range(len(items)):
            feats.append(items[i][1])
            labels.append(torch.argmax(items[i][2]))

        skf = StratifiedKFold(n_splits=kfold, shuffle=True)
        for fold_idx, (train_index, val_index) in enumerate(skf.split(feats, labels)):
            print("Run {}-fold splitting".format(fold_idx))
            train_dct = {}
            dct_idx = 0
            for j in train_index:
                train_dct[dct_idx] =  [items[j][0], items[j][1], items[j][2]]
                dct_idx += 1
            
            torch.save(train_dct, "{}/train_tp_fold{}.feats".format(out_dir, fold_idx))

            val_dct = {}
            dct_idx = 0
            for j in val_index:
                val_dct[dct_idx] =  [items[j][0], items[j][1], items[j][2]]
                dct_idx += 1
            
            torch.save(val_dct, "{}/val_tp_fold{}.feats".format(out_dir, fold_idx))


def main(args):
    if args.load_feats is not None:
        DumpFeature._split_data(feature_path=args.load_feats, out_dir=args.out_dir, kfold=int(args.kfold))
    else:
        extractor = DumpFeature(rfcx_dir=args.rfcx_dir, out_dir=args.out_dir, feature_type=args.feature_type, kfold=int(args.kfold))
        extractor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dump feature in disk")
    parser.add_argument("rfcx_dir", help="RFCX data dir")
    parser.add_argument("out_dir", help="feature storage dir")
    parser.add_argument("--kfold", help="split data into k-folds", default=5)
    parser.add_argument("--feature_type", help="spectrogram/fbank/mfcc", default="fbank")
    parser.add_argument("--load_feats", help="load from dumped feature", default=None)
    args = parser.parse_args()
    main(args)