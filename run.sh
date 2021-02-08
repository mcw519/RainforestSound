#!/bin/bash

# Copyright 2021 (author: Meng Wu)

# data dir
rfcx_dir="rfcx"
feats_dir="feats"
exp_dir="exp_EfficientNetB0_SpecAug_mixup_anti"

# feats config
feats_type="fbank" # spectrogram/fbank/mfcc
kfold=5
start_fold=0

# model config
model_type="EfficientNetB0" # EfficientNetB0/EfficientNetB1/EfficientNetB2/ResNeSt50/ResNeSt101
activation_type="default" # default/mish/selu
antimodel="True" # train model with anti class on FP dataset 
criterion="CrossEntropyLoss" # CrossEntropyLoss/BCEWithLogitsLoss

# inference config
ensemble="True"
test_nfold="0"

DUMPFEATURE=0
TRAIN=1
INFERENCE=1


if [ $DUMPFEATURE -eq 1 ]; then
    echo "================================================="
    echo "Prepared $feats_type feature"
    echo "================================================="
    python dump_and_split.py --kfold $kfold --feature_type $feats_type $rfcx_dir $feats_dir

fi


if [ $TRAIN -eq 1 ]; then
    echo "================================================="
    echo "Training $model_type model with $criterion"
    echo "================================================="
    if [ $antimodel == "True" ]; then
        for i in $(seq $start_fold $[$kfold - 1]); do
            echo "Start training kfold$i"
            python train.py --model_type $model_type --activation $activation_type --criterion $criterion --antimodel --train_fp_feats $feats_dir/train_fp.feats \
                $exp_dir/fold$i $feats_dir/train_tp_fold$i.feats $feats_dir/val_tp_fold$i.feats
        done
    
    else
        for i in $(seq 0 $[$kfold - 1]); do
            echo "Start training kfold$i"
            python train.py --model_type $model_type --activation $activation_type --criterion $criterion \
                $exp_dir/fold$i $feats_dir/train_tp_fold$i.feats $feats_dir/val_tp_fold$i.feats
        done
    fi
fi


if [ $INFERENCE -eq 1 ]; then
    if [ $ensemble == "True" ]; then
        if [ $antimodel == "True" ]; then
            python submit_result.py --from_anti_model --feature_type $feats_type --model_type $model_type --activation $activation_type --ensemble \
                --ensembleB $exp_dir/fold1/best_model.pt \
                --ensembleC $exp_dir/fold2/best_model.pt \
                --ensembleD $exp_dir/fold3/best_model.pt \
                --ensembleE $exp_dir/fold4/best_model.pt \
                $rfcx_dir/test/ $exp_dir/fold0/best_model.pt
        else
            python submit_result.py --feature_type $feats_type --model_type $model_type --activation $activation_type --ensemble \
                --ensembleB $exp_dir/fold1/best_model.pt \
                --ensembleC $exp_dir/fold2/best_model.pt \
                --ensembleD $exp_dir/fold3/best_model.pt \
                --ensembleE $exp_dir/fold4/best_model.pt \
                $rfcx_dir/test/ $exp_dir/fold0/best_model.pt
        fi

    else
        if [ $antimodel == "True" ]; then
            python submit_result.py --feature_type $feats_type --model_type $model_type --activation $activation_type --from_anti_model $rfcx_dir/test/ $exp_dir/fold$test_nfold/best_model.pt
        else
            python submit_result.py --feature_type $feats_type --model_type $model_type --activation $activation_type $rfcx_dir/test/ $exp_dir/fold$test_nfold/best_model.pt
        fi
    fi
fi
