# Copyright 2021 (author: Meng Wu)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import RFCXmodel
import gc
from metrics import LWLRAP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import argparse
from functional import mixup_data, mixup_criterion


class RFCXDatasetLoad(torch.utils.data.Dataset):
    """
        This Dataset using pre-dumped feature, could save memory.
        Before using it, please runing dump_and_split.py.
    """
    def __init__(self, feats_path):
        items = torch.load(feats_path)
        self.items = list(items.values())
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx: int):
        uttid, feats, target = self.items[idx]
        
        return uttid, feats, target


class RFCXCollateFuncLoad:
    def __init__(self):
        pass

    def __call__(self, batch):
        col_uttid_list = []
        col_feat_list = []
        col_target_list = []
        for b in batch:
            """
                Unlike DFCXDataset, here idx only contain one data information
                feature in feat_list with shape [T, C] or [3, 224, 400] for Resnet setting
                target in target_list with shape [C]
            """
            uttid, feat, target = b
                      
            col_uttid_list.append(uttid)
            
            col_feat_list.append(feat)
            
            col_target_list.append(target)

            del feat, target, b
            
        # batch feats
        padded_feat = pad_sequence(col_feat_list, batch_first=True)
        padded_target = torch.stack(col_target_list).view(-1, 24)

        del col_feat_list, col_target_list
        assert len(col_uttid_list) == padded_feat.shape[0] == padded_feat.shape[0]
        
        return col_uttid_list, padded_feat, padded_target


def get_rfcx_load_dataloader(dataset: torch.utils.data.Dataset, batch_size=1, shuffle=True, num_workers=0):
    
    collate_fn = RFCXCollateFuncLoad()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


def load_ckpt(filename, model):
    
    ckpt = torch.load(filename, map_location="cpu")

    # keys = [ "state_dict", "epoch", "learning_rate", "loss" ]
    model.load_state_dict(ckpt["state_dict"])
    
    epoch = ckpt["epoch"]
    learning_rate = ckpt["learning_rate"]
    loss = ckpt["loss"]

    return epoch, learning_rate, loss


def save_ckpt(filename, model, epoch, learning_rate, loss, local_rank=0):
    if local_rank != 0:
        return 
    
    ckpt = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "learning_rate": learning_rate,
        "loss": loss }
    
    torch.save(ckpt, filename)


def save_ckpt_info(filename, model_path, current_epoch, learning_rate, loss, best_loss, best_epoch, local_rank=0):
    if local_rank != 0:
        return
    
    with open(filename, "w") as f:
        f.write("model_path: {}\n".format(model_path))
        f.write("epoch: {}\n".format(current_epoch))
        f.write("learning rate: {}\n".format(learning_rate))
        f.write("loss: {}\n".format(loss))
        f.write("best loss: {}\n".format(best_loss))
        f.write("best epoch: {}\n".format(best_epoch))


def train_one_epoch(dataloader, model, device, optimizer, current_epoch, criterion):
    model.train()
    model.to(device)
    total_loss = 0.
    num = 0.
    total_precision = 0.
    mixup = True

    num_repeat = 1
    for kk in range(num_repeat):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            uttid_list, feats, target = batch

            utt_num = len(uttid_list)
            
            # target is [N, 24]
            target = target.to("cpu")

            if mixup:
                inputs, targets_a, targets_b, lam = mixup_data(feats, target, device=device)
                inputs = inputs.to(device)
                labels_a = torch.argmax(targets_a, dim=-1)
                labels_a = labels_a.to(device)
                labels_b = torch.argmax(targets_b, dim=-1)
                labels_b = labels_b.to(device)
            else:
                inputs = feats.to(device)
                labels = torch.argmax(target, dim=-1)
                labels = labels.to(device)
            
            # activation is [N, num_class]
            activation = model(inputs)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                if mixup:
                    loss = mixup_criterion(criterion, activation, labels_a, labels_b, lam)
                else:
                    loss = criterion(activation, labels)
            else:
                if mixup:
                    targets_a = target.type_as(activation)
                    targets_b = target.type_as(activation)
                    loss = mixup_criterion(criterion, activation, targets_a, targets_b, lam)
                else:
                    target = target.type_as(activation)
                    loss = criterion(activation, target)

            loss.backward()

            optimizer.step()

            total_loss += loss

            precision = LWLRAP(target, activation)
            total_precision += precision

            num += 1

            if batch_idx % 50 == 0:
                print("batch {}/{} ({:.2f}%) ({}/{}), loss {:.5f}, average {:.5f}, lwlrap {:.5f}".format(batch_idx, len(dataloader),
                            float(batch_idx) / len(dataloader) * 100, kk, num_repeat, loss.item(), total_loss / num, total_precision / num))
                
            
            del batch_idx, batch, feats, target, activation, loss
            gc.collect()
            torch.cuda.empty_cache()
    
    return total_loss / num


def get_cv_loss(cv_dataloader, model, current_epoch, criterion):
    # cross validation loss
    device = "cpu"
    cv_total_loss = 0.
    cv_num = 0.
    cv_precision = 0.

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(cv_dataloader):
            uttid_list, feats, target = batch
            utt_num = len(uttid_list)

            # feats is [N, T, C] or [N, 3, 224, 400]
            feats = feats.to(device)
            
            # target is [N, 24]
            target = target.to("cpu")
            labels = torch.argmax(target, dim=-1)
            labels = labels.to(device)
            
            # activation is [N, num_class]
            activation = model(feats)
            
            target = target.type_as(activation)

            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(activation, labels)
            else:
                loss = criterion(activation, target)

            precision = LWLRAP(target, activation)

            cv_total_loss += loss
            cv_precision += precision

            cv_num += 1

            del batch_idx, batch, feats, target, activation
            gc.collect()
            torch.cuda.empty_cache()

        print("epoch {} cross-validation loss {} lwlrap {}".format(current_epoch, cv_total_loss / cv_num, cv_precision / cv_num))


def train_one_tpfp_epoch(tp_dataloader, fp_dataloader, model, device, optimizer, current_epoch, criterion):
    model.train()
    model.to(device)
    total_loss = 0.
    total_tp_loss = 0.
    total_fp_loss = 0.
    num = 0.
    total_precision = 0.
    total_tp_precision = 0.
    total_fp_precision =0.
    
    mixup = True

    len_dataloader = min(len(tp_dataloader), len(fp_dataloader))

    num_repeat = 1
    for kk in range(num_repeat):
        for batch_idx, (batch_tp, batch_fp) in enumerate(zip(tp_dataloader, fp_dataloader)):

            optimizer.zero_grad()
            
            # train on TP data
            _, feats_tp, target_tp = batch_tp
            
            # feats is [N, T, C] or [N, 3, 224, 400]
            feats_tp = feats_tp.to(device)
            
            # target is [N, 24]
            padding_zeros = torch.zeros_like(target_tp, dtype=torch.float)
            target_tp = torch.cat([target_tp, padding_zeros], dim=-1)
            # target_tp = target_tp.to(device)
            labels_tp = torch.argmax(target_tp, dim=-1)
            labels_tp = labels_tp.to(device)

            if mixup:
                inputs_tp, targets_tp_a, targets_tp_b, lam = mixup_data(feats_tp, target_tp, device=device)
                inputs_tp = inputs_tp.to(device)
                labels_tp_a = torch.argmax(targets_tp_a, dim=-1)
                labels_tp_a = labels_tp_a.to(device)
                labels_tp_b = torch.argmax(targets_tp_b, dim=-1)
                labels_tp_b = labels_tp_b.to(device)
            else:
                inputs_tp = feats_tp.to(device)
                labels_tp = torch.argmax(target_tp, dim=-1)
                labels_tp = labels_tp.to(device)
            
            # activation is [N, num_class]
            activation_tp = model(inputs_tp)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                if mixup:
                    loss_tp = mixup_criterion(criterion, activation_tp, labels_tp_a, labels_tp_b, lam)
                else:
                    loss_tp = criterion(activation_tp, labels_tp)
            else:
                if mixup:
                    targets_tp_a = targets_tp_a.type_as(activation_tp)
                    targets_tp_b = targets_tp_b.type_as(activation_tp)
                    loss_tp = mixup_criterion(criterion, activation_tp, targets_tp_a, targets_tp_b, lam)
                else:
                    target_tp = target_tp.type_as(activation_tp)
                    loss_tp = criterion(activation_tp, target_tp)

            total_tp_loss += loss_tp
            
            # train on FP data
            _, feats_fp, target_fp = batch_fp
            
            # feats is [N, T, C] or [N, 3, 224, 400]
            feats_fp = feats_fp.to(device)
            
            # target is [N, 24]
            padding_zeros = torch.zeros_like(target_fp, dtype=torch.float)
            target_fp = torch.cat([padding_zeros, target_fp], dim=-1)
            # target_fp = target_fp.to(device)
            labels_fp = torch.argmax(target_fp, dim=-1)
            labels_fp = labels_fp.to(device)
            
            if mixup:
                inputs_fp, targets_fp_a, targets_fp_b, lam = mixup_data(feats_fp, target_fp, device=device)
                inputs_fp = inputs_fp.to(device)
                labels_fp_a = torch.argmax(targets_fp_a, dim=-1)
                labels_fp_a = labels_fp_a.to(device)
                labels_fp_b = torch.argmax(targets_fp_b, dim=-1)
                labels_fp_b = labels_fp_b.to(device)
            else:
                inputs_fp = feats_fp.to(device)
                labels_fp = torch.argmax(target_fp, dim=-1)
                labels_fp = labels_fp.to(device)

            # activation is [N, num_class]
            activation_fp = model(inputs_fp)
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                if mixup:
                    loss_fp = mixup_criterion(criterion, activation_fp, labels_fp_a, labels_fp_b, lam)
                else:
                    loss_fp = criterion(activation_fp, labels_fp)
            else:
                if mixup:
                    targets_fp_a = targets_fp_a.type_as(activation_fp)
                    targets_fp_b = targets_fp_b.type_as(activation_fp)
                    loss_fp = mixup_criterion(criterion, activation_fp, targets_fp_a, targets_fp_b, lam)
                else:
                    target_fp = target_fp.type_as(activation_fp)
                    loss_fp = criterion(activation_fp, target_fp)
            
            total_fp_loss += loss_fp
            
            loss = loss_tp + loss_fp
            
            loss.backward()

            optimizer.step()

            total_loss += loss

            precision_tp = LWLRAP(target_tp, activation_tp)
            total_tp_precision += precision_tp
            
            precision_fp = LWLRAP(target_fp, activation_fp)
            total_fp_precision += precision_fp
            
            total_precision += precision_tp
            total_precision += precision_fp
            
            num += 1

            if batch_idx % 50 == 0:
                print("batch {}/{} ({:.2f}%) ({}/{}), loss {:.5f}, average {:.5f}, lwlrap {:.5f}, TP loss {:.5f} lwlrap {:.5f}, FP loss {:.5f} lwlrap {:.5f}".format(batch_idx, len_dataloader,
                            float(batch_idx) / len_dataloader * 100, kk, num_repeat, loss.item(), total_loss / num / 2, total_precision / num / 2, total_tp_loss / num, total_tp_precision / num, total_fp_loss / num, total_fp_precision / num))
            
            # del batch_idx, batch_tp, batch_fp, feats_tp, feats_fp, target_tp, target_fp, labels_tp, labels_fp, activation_tp, activation_fp, loss, loss_tp, loss_fp
            gc.collect()
            torch.cuda.empty_cache()
    
    return total_loss / num


def get_cv_tpfp_loss(cv_dataloader, model, current_epoch, criterion):
    # cross validation loss
    device = "cpu"
    cv_total_loss = 0.
    cv_num = 0.
    cv_precision = 0.

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(cv_dataloader):
            uttid_list, feats, target = batch
            _ = len(uttid_list)

            # feats is [N, T, C] or [N, 3, 224, 400]
            feats = feats.to(device)
            
            padding_zeros = torch.zeros_like(target, dtype=torch.float)
            target = torch.cat([target, padding_zeros], dim=-1)
            # target = target.to(device)
            labels = torch.argmax(target, dim=-1)
            labels = labels.to(device)
                        
            # activation is [N, C]
            activation = model(feats)

            if isinstance(criterion, nn.CrossEntropyLoss):
                loss = criterion(activation, labels)
            else:
                loss = criterion(activation, target)

            precision = LWLRAP(target, activation)

            cv_total_loss += loss
            cv_precision += precision

            cv_num += 1

            del batch_idx, batch, feats, target, labels, activation
            gc.collect()
            torch.cuda.empty_cache()

        print("epoch {} cross-validation loss {} lwlrap {}".format(current_epoch, cv_total_loss / cv_num, cv_precision / cv_num))



def main(args):

    device = torch.device("cuda", 0)

    if args.antimodel:
        out_dim = 24*2
    else:
        out_dim = 24

    if args.model_type == "ResNeSt50":
        model = RFCXmodel(out_dim, backbone=args.model_type, activation=args.activation)
    elif args.model_type == "EfficientNetB0":
        model = RFCXmodel(out_dim, backbone=args.model_type, activation=args.activation)
    elif args.model_type == "EfficientNetB1":
        model = RFCXmodel(out_dim, backbone=args.model_type, activation=args.activation)
    elif args.model_type == "EfficientNetB2":
        model = RFCXmodel(out_dim, backbone=args.model_type, activation=args.activation)
    else:
        raise NameError

    model.to(device)

    if args.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "BCEWithLogitsLoss":
        pos_weights = torch.ones(24)
        pos_weights = pos_weights * 24
        pos_weights = pos_weights.to(device)
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise NameError

    start_epoch = 0
    num_epoch = 30
    learning_rate = 0.001
    batch_size = 2
    best_loss = None
    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    
    if start_epoch != 0:
        start_epoch, current_learning_rate, best_loss = load_ckpt("{}/epoch-{}.pt".format(exp_dir, start_epoch-1), model)
        model.to(device)
        best_loss = best_loss.to(device)
        optimizer = optim.SGD(model.parameters(), lr=current_learning_rate, weight_decay=0.0001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4)
    
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4)

    feats_path = args.train_feats
    dataset = RFCXDatasetLoad(feats_path=feats_path)
    dataloader = get_rfcx_load_dataloader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    cv_feats_path = args.cv_feats
    cv_dataset = RFCXDatasetLoad(feats_path=cv_feats_path)
    cv_dataloader = get_rfcx_load_dataloader(dataset=cv_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if args.antimodel:
        fp_feats_path = args.train_fp_feats
        fp_dataset = RFCXDatasetLoad(feats_path=fp_feats_path)
        fp_dataloader = get_rfcx_load_dataloader(dataset=fp_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        fp_dataloader = None

    model.train()

    best_epoch = 0
    best_model_path = os.path.join(exp_dir, "best_model.pt")
    best_model_info = os.path.join(exp_dir, "best-epoch-info")

    for epoch in range(start_epoch, num_epoch):

        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        
        if args.antimodel:
            loss = train_one_tpfp_epoch(tp_dataloader=dataloader, fp_dataloader=fp_dataloader, model=model, device=device, optimizer=optimizer, current_epoch=epoch, criterion=criterion)
            get_cv_tpfp_loss(cv_dataloader, model, epoch, criterion=criterion)
        else:
            loss = train_one_epoch(dataloader=dataloader, model=model, device=device, optimizer=optimizer, current_epoch=epoch, criterion=criterion)
            get_cv_loss(cv_dataloader, model, epoch, criterion=criterion)

        # save best model
        if best_loss is None or best_loss > loss:
            best_loss = loss
            best_epoch = epoch
            
            save_ckpt(filename=best_model_path,
                                model=model,
                                epoch=epoch,
                                learning_rate=learning_rate,
                                loss=loss)
            
            save_ckpt_info(filename=best_model_info,
                                        model_path=best_model_path,
                                        current_epoch=epoch,
                                        learning_rate=learning_rate,
                                        loss=loss,
                                        best_loss=best_loss,
                                        best_epoch=best_epoch)
        
        model_path = os.path.join(exp_dir, "epoch-{}.pt".format(epoch))
        save_ckpt(filename=model_path,
                                model=model,
                                epoch=epoch,
                                learning_rate=learning_rate,
                                loss=loss)

        info_filename = os.path.join(exp_dir, "epoch-{}-info".format(epoch))
        save_ckpt_info(filename=info_filename,
                                        model_path=model_path,
                                        current_epoch=epoch,
                                        learning_rate=learning_rate,
                                        loss=loss,
                                        best_loss=best_loss,
                                        best_epoch=best_epoch)
        
        scheduler.step()
       

def test_load_feats():
    feats_path = "feats/train_tp.feats"
    dataset = RFCXDatasetLoad(feats_path=feats_path)
    dataloader = get_rfcx_load_dataloader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
    i = 0
    for batch in dataloader:
        uttid_list, feats, targets = batch
        i += 1
        print(uttid_list, feats.shape, targets.shape)

        if i > 5:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("exp_dir", help="output dir")
    parser.add_argument("train_feats", help="train feats path")
    parser.add_argument("cv_feats", help="cv feats path")
    parser.add_argument("--model_type", help="EfficientNetB0/EfficientNetB1/EfficientNetB2/ResNeSt50", default="ResNeSt50")
    parser.add_argument("--activation", help="mish/selu", default=None)
    parser.add_argument("--criterion", help="CrossEntropyLoss/BCEWithLogitsLoss", default="CrossEntropyLoss")
    parser.add_argument("--antimodel", help="train model with anti class", default=False, action="store_true")
    parser.add_argument("--train_fp_feats", help="train model with anti class", default=None)
    args = parser.parse_args()
    main(args)