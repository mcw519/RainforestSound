import os
import io
import sys
import torch
import torch.nn.functional as F


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


def main(argv):
    num = 0
    ensemble_dct = {}
    for arg in range(1, len(argv)):
        print(argv[arg])
        ensemble_dct[num] = read_file_as_dict(argv[arg])
        num += 1

    ensemble_result = []
    key_list = sorted(ensemble_dct[0].keys())
    for key in key_list:
        print("{}/{}".format(len(ensemble_result), len(key_list)))
        score = []
        for j in range(num):
            score.append(ensemble_dct[j][key])
        score = torch.mean(torch.stack(score), dim=0).tolist()
        score = list(map(str, score))
        ensemble_result.append([key] + score)
    
    with io.open("ensemble.txt", "w") as f:
        f.writelines("recording_id,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23" + "\n")
        for i in ensemble_result:
            f.writelines(",".join(i))
            f.writelines("\n")
    

def test_read_file_as_dict():
    f_path = "/home/haka/meng/RFCX/exp_1220_mish_model_batch4/fold0/RFCX_submit.csv"
    f_dct = read_file_as_dict(f_path)

    i = 0
    for key in f_dct.keys():
        print(f_dct[key])
        i += 1
        if i > 5:
            break


if __name__ == "__main__":
    # test_read_file_as_dict()
    main(sys.argv)