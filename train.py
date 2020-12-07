import enum
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, dataset
from chain_model import RainforestModel
from chain_dataset import get_rainforest_dataloader, RainforestDataset
import gc


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

    total_loss = 0.
    num = 0.

    criterion = nn.CrossEntropyLoss()

    num_repeat = 1
    for kk in range(num_repeat):
        for batch_idx, batch in enumerate(dataloader):
            uttid_list, feats, feat_len_list, target = batch
            # print(uttid_list, feat_len_list)

            # feats is [N, T, C]
            feats = feats.to(device)
            
            # target is [N, C]
            target = target.to(device)

            # activation is [N, T, C]
            activation = model(feats)

            loss = criterion(activation, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss

            num += 1

            if batch_idx % 100 == 0:
                print("batch {}/{} ({:.2f}%) ({}/{}), loss {:.5f}, average {:.5f}".format(batch_idx, len(dataloader),
                            float(batch_idx) / len(dataloader) * 100, kk, num_repeat, loss.item(), total_loss / num))
            
            del batch_idx, batch, feats, target
            gc.collect()
            torch.cuda.empty_cache()
    
    return total_loss / num


def main():
    
    device = torch.device("cuda", 0)
    model = RainforestModel(feat_dim=1025, output_dim=24)
    model.to(device)

    start_epoch = 0
    num_epoch = 30
    learning_rate = 0.0001
    best_loss = None

    manifest = "train_tp.csv"
    audio_dir = "/home/haka/meng/Rainforest/rfcx-species-audio-detection/train"
    dataset = RainforestDataset(manifest, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False)
    dataloader = get_rainforest_dataloader(dataset, model_left_context=27, model_right_context=27, frames_per_chunk=150, batch_size=1)

    manifest_cv = "cv_tp.csv"
    dataset_cv = RainforestDataset(manifest_cv, audio_dir, num_classess=24, sample_frequency=48000.0, cmvn=False)
    dataloader_cv = get_rainforest_dataloader(dataset, model_left_context=27, model_right_context=27, frames_per_chunk=150, batch_size=1, shuffle=False)

    lr = learning_rate
    optimize = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

    model.train()

    best_epoch = 0
    best_model_path = os.path.join("exp", "best_model.pt")
    best_model_info = os.path.join("exp", "best-epoch-info.pt")

    for epoch in range(start_epoch, num_epoch):
        learning_rate = lr * pow(0.8, epoch)

        for param_group in optimize.param_groups:
            param_group["lr"] = learning_rate
        
        loss = train_one_epoch(dataloader=dataloader, model=model, device=device, optimizer=optimize, current_epoch=epoch)

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