"""
    RFCX training data is very unbalanced
"""

import random

with open("rfcx-species-audio-detection/train_tp.csv") as f:
    fid = [ i for i in f.readlines() ]

col_content = fid[0]
content = fid[1:]

content_dct = {}
for i in content:
    species_id = i.strip().split(",")[1]
    try:
        content_dct[species_id].append(i)
    except:
        content_dct[species_id] = [i]

f_train = open("train_tp.csv", "w")
f_train.writelines([col_content])
f_cv = open("cv_tp.csv", "w")
f_cv.writelines([col_content])

min_num = min([len(content_dct[key]) for key in content_dct.keys()])

for key in content_dct.keys():
    cv_parts = round(min_num * 0.1)
    random.shuffle(content_dct[key])
    
    content_dct[key] = content_dct[key][:min_num + 1]

    train_set = content_dct[key][cv_parts:]
    f_train.writelines(train_set)
    cv_set = content_dct[key][:cv_parts]
    f_cv.writelines(cv_set)