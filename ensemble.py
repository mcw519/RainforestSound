# Copyright 2021 (author: Meng Wu)

import io
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

def read_file_as_dict(x):
    with io.open(x, "r") as f:
        f_list = [ i.strip().split(",") for i in f.readlines() ]
    
    # ignore column content
    f_list = f_list[1:]
    f_dct = {}
    for i in range(len(f_list)):
        key = f_list[i][0]
        content = torch.Tensor(list(map(float, f_list[i][1:])))
        content = F.softmax(content, dim=0)
        f_dct[key] = content
    
    return f_dct


def main(args):
    num = 0
    ensemble_dct = {}
    print(f"Choose data from: {args.datas}")

    if args.LB_score is not None:
        """
            Ensemble by leadboard socre's weight
        """
        weight = list(map(float, args.LB_score.strip().split(";")))
        if len(weight) != len(args.datas):
            print(len(weight), len(args.datas))
            raise IndexError

        weight = torch.Tensor(weight)
        weight = F.softmax(weight, dim=-1)
        print(f"you using weight as {weight}")
        weight = weight.view(-1, 1)

    for arg in range(len(args.datas)):
        ensemble_dct[num] = read_file_as_dict(args.datas[arg])
        num += 1

    ensemble_result = []
    key_list = sorted(ensemble_dct[0].keys())

    for key in tqdm(key_list):
        score = []
        for j in range(num):
            score.append(ensemble_dct[j][key])

        if args.LB_score is not None:
            score = torch.sum(weight * torch.stack(score), dim=0).tolist()
        else:
            score = torch.mean(torch.stack(score), dim=0).tolist()
        score = list(map(str, score))
        ensemble_result.append([key] + score)
    
    with io.open("ensemble.csv", "w") as f:
        f.writelines("recording_id,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23" + "\n")
        for i in ensemble_result:
            f.writelines(",".join(i))
            f.writelines("\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("datas", help="output dir", nargs="+")
    parser.add_argument("--LB_score", help="leadboard score for each file, ex: '0.9;0.8;0.7'", default=None)
    args = parser.parse_args()
    main(args)
