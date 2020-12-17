import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader
from model import ResnetRFCX
from dataset import get_rfcx_dataloader
import gc
from metrics import LWLRAP, WeightBCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader



class RFCXDatasetLoad(torch.utils.data.Dataset):
    """
        This Dataset using pre-dumped feature, could save memory.
        Before using it, please check dump_and_split.py.
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


def train_one_epoch(dataloader, model, device, optimizer, current_epoch):
    model.train()
    model.to(device)
    total_loss = 0.
    num = 0.
    total_precision = 0.

    pos_weights = torch.ones(24)
    pos_weights = pos_weights * 24
    pos_weights = pos_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    num_repeat = 1
    for kk in range(num_repeat):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            uttid_list, feats, target = batch
            utt_num = len(uttid_list)

            # feats is [N, T, C] or [N, 3, 224, 400]
            feats = feats.to(device)
            
            # target is [N, 24]
            target = target.to(device)
            
            # activation is [N, num_class]
            activation = model(feats)
            
            # type matching
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


def get_cv_loss(cv_dataloader, model, current_epoch):
    # cross validation loss
    device = "cpu"
    cv_total_loss = 0.
    cv_num = 0.
    cv_precision = 0.

    pos_weights = torch.ones(24)
    pos_weights = pos_weights * 24
    pos_weights = pos_weights.to(device)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(cv_dataloader):
            uttid_list, feats, target = batch
            utt_num = len(uttid_list)

            # feats is [N, T, C] or [N, 3, 224, 400]
            feats = feats.to(device)
            
            target = target.to(device)
                        
            # activation is [N, C]
            activation = model(feats)

            loss = criterion(activation, target)

            precision = LWLRAP(target, activation)

            cv_total_loss += loss
            cv_precision += precision

            cv_num += 1

            del batch_idx, batch, feats, target, activation
            gc.collect()
            torch.cuda.empty_cache()

        print("epoch {} cross-validation loss {} lwlrap {}".format(current_epoch, cv_total_loss / cv_num, cv_precision / cv_num))


def main():

    if not os.path.isdir("exp"):
        os.makedirs("exp")
    
    device = torch.device("cuda", 0)
    model = ResnetRFCX(hidden_dim=1024, output_dim=24)

    model.to(device)

    start_epoch = 0
    num_epoch = 30
    learning_rate = 0.001
    batch_size = 4
    best_loss = None


    if start_epoch != 0:
        start_epoch, current_learning_rate, best_loss = load_ckpt("exp/epoch-{}.pt".format(start_epoch), model)
        model.to(device)
        best_loss = best_loss.to(device)

    feats_path = "/home/haka/meng/RFCX/feats/train_tp_fold1.feats"
    dataset = RFCXDatasetLoad(feats_path=feats_path)
    dataloader = get_rfcx_load_dataloader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    cv_feats_path = "/home/haka/meng/RFCX/feats/val_tp_fold1.feats"
    cv_dataset = RFCXDatasetLoad(feats_path=cv_feats_path)
    cv_dataloader = get_rfcx_load_dataloader(dataset=cv_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    lr = learning_rate
    optimize = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimize, step_size=7, gamma=0.4)

    model.train()

    best_epoch = 0
    best_model_path = os.path.join("exp", "best_model.pt")
    best_model_info = os.path.join("exp", "best-epoch-info")

    for epoch in range(start_epoch, num_epoch):
        for param_group in optimize.param_groups:
            learning_rate = param_group["lr"]
        
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
    # test_load_feats()
    main()
