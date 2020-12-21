import numpy as np
from sklearn import manifold
from train import RFCXDatasetLoad, get_rfcx_load_dataloader, load_ckpt
from model import ResnetRFCX
import torch
import torch.nn as nn
import plotly.express as px


def main(feat_path, model_path):
    #Prepare the data
    device = "cpu"

    dataset = RFCXDatasetLoad(feats_path=feat_path)
    dataloader = get_rfcx_load_dataloader(dataset=dataset, batch_size=4, shuffle=False, num_workers=0)

    model = ResnetRFCX(1024, 24)
    load_ckpt(model_path, model)

    # remove fc layer
    model = nn.Sequential(*(list(model.resnet.children())[:-1]))
    model.to(device)
    model.eval()

    total_file = len(dataset)
    feature_dct = {}
    idx = 0

    for batch_idx, batch in enumerate(dataloader):

        with torch.no_grad():
            uttid_list, feats, target = batch
            utt_num = len(uttid_list)

            # feats is [N, T, C] or [N, 3, 224, 400]
            feats = feats.to(device)
            
            # target is [N, 24]
            target = target.to(device)
            
            # activation is [N, 2048]
            feature = model(feats)
            feature = feature.squeeze(3)
            feature = feature.squeeze(2)
            feature = np.array(feature.tolist())

            for i in range(utt_num):
                label = int(torch.argmax(target[i]))
                if "_aug" in uttid_list[i]:
                    augment = 1
                else:
                    augment = 0

                feature_dct[idx] = {"feats": feature[i], "label": label, "augment": augment}
                idx += 1

            print("processing {}/{} files".format(len(feature_dct.keys()), total_file))

    #t-SNE
    X = []
    Y = []
    for key in feature_dct.keys():
        feats = feature_dct[key]["feats"]
        label = "{}_{}".format(feature_dct[key]["label"], feature_dct[key]["augment"])
        X.append(feats)
        Y.append(label)

    X = np.array(X)
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

    train = {}
    train["tsneX"] = X_tsne[:, 0]
    train["tsneY"] = X_tsne[:, 1]
    train["label"] = Y

    fig = px.scatter(train, x="tsneX", y="tsneY", color="label", labels={
        "tsneX": "X",
        "tsneY": "Y",
        "label": "Species + Aug"}, opacity = 0.5)

    fig.update_yaxes(matches=None, showticklabels=False, visible=True)
    fig.update_xaxes(matches=None, showticklabels=False, visible=True)

    fig.show()


if __name__ == "__main__":
    import sys
    # Usage: python plot_tSNE.py <feats path> <ckpt path>
    main(sys.argv[1], sys.argv[2])
