import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, dataset
from model import RainforestModel
from dataset import get_rainforest_dataloader, RainforestDataset, RainforestDatasetLoad
import gc
from loss import lwlrap


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


def train_one_epoch(dataloader, model, device, optimizer, current_epoch):
    model.train()
    model.to(device)
    total_loss = 0.
    num = 0.
    total_precision = 0.

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    num_repeat = 5
    for kk in range(num_repeat):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            uttid_list, feats, target, feat_len_list = batch
            
            feats = feats.to(device)
            
            target = target.to(device)

            activation = model(feats)
            
            # type match
            target = target.type_as(activation)
            
            loss = criterion(activation, target)
            precision = lwlrap(target, activation)

            loss.backward()

            optimizer.step()

            total_loss += loss
            total_precision += precision

            num += 1

            if batch_idx % 100 == 0:
                print("batch {}/{} ({:.2f}%) ({}/{}), loss {:.5f}, average {:.5f}, lwlrap {:.5f}".format(batch_idx, len(dataloader),
                            float(batch_idx) / len(dataloader) * 100, kk, num_repeat, loss.item(), total_loss / num, total_precision / num))
            
            del batch_idx, batch, feats, target
            gc.collect()
            torch.cuda.empty_cache()
    
    return total_loss / num


def get_cv_loss(cv_dataloader, model, current_epoch):
    # cross validation loss
    device = "cpu"
    cv_total_loss = 0.
    cv_num = 0.
    cv_precision = 0.

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(cv_dataloader):
            uttid_list, feats, target, feat_len_list = batch
            
            # feats is [N, T, C]
            feats = feats.to(device)
            
            # target is [N, C]
            target = target.to(device)
                        
            # activation is [N, C]
            activation = model(feats)
            target = target.type_as(activation)

            loss = criterion(activation, target)
            precision = lwlrap(target, activation)

            cv_total_loss += loss
            cv_precision += precision

            cv_num += 1

            del batch_idx, batch, feats, target
            gc.collect()
            torch.cuda.empty_cache()

        print("epoch {} cross-validation loss {} lwlrap {}".format(current_epoch, cv_total_loss / cv_num, cv_precision / cv_num))


def main():
    
    device = torch.device("cuda", 0)
    model = RainforestModel(feat_dim=120, hidden_dim=256, output_dim=24, num_lstm_layers=3)
    model.to(device)

    start_epoch = 0
    num_epoch = 30
    learning_rate = 0.0001
    best_loss = None

    feat_path = "feats/train_tp.feats"
    dataset = RainforestDatasetLoad(feat_path)
    dataloader = get_rainforest_dataloader(dataset, batch_size=1, shuffle=True)

    cv_feat_path = "feats/cv_tp.feats"
    cv_dataset = RainforestDatasetLoad(cv_feat_path)
    cv_dataloader = get_rainforest_dataloader(cv_dataset, batch_size=1, shuffle=False)

    lr = learning_rate
    optimize = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    model.train()

    best_epoch = 0
    best_model_path = os.path.join("exp", "best_model.pt")
    best_model_info = os.path.join("exp", "best-epoch-info")

    for epoch in range(start_epoch, num_epoch):
        learning_rate = lr * pow(0.8, epoch)

        for param_group in optimize.param_groups:
            param_group["lr"] = learning_rate
        
        loss = train_one_epoch(dataloader=dataloader, model=model, device=device, optimizer=optimize, current_epoch=epoch)
        get_cv_loss(cv_dataloader, model, epoch)

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
        
        model_path = os.path.join("exp", "epoch-{}.pt".format(epoch))
        save_ckpt(filename=model_path,
                                model=model,
                                epoch=epoch,
                                learning_rate=learning_rate,
                                loss=loss)

        info_filename = os.path.join("exp", "epoch-{}-info".format(epoch))
        save_ckpt_info(filename=info_filename,
                                        model_path=model_path,
                                        current_epoch=epoch,
                                        learning_rate=learning_rate,
                                        loss=loss,
                                        best_loss=best_loss,
                                        best_epoch=best_epoch)
       


if __name__ == "__main__":
    main()