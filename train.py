from dataset import *
from net import GSSCNetPublic

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

import parameter

parameter._init()

SUFFIX = '_public_backbone'


def _select_device(cuda):
    if cuda == 'cuda0':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if cuda == 'cuda1':
        return torch.device('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    if cuda == 'cuda2':
        return torch.device('cuda:2' if torch.cuda.device_count() > 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _make_save_path(model_savepath, suffix):
    root, ext = os.path.splitext(model_savepath)
    if ext == '':
        ext = '.pth'
    return root + suffix + ext


def getLog(fileName, strInfo):
    folder = os.path.dirname(fileName)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(fileName, 'a', encoding='utf-8') as f:
        f.write(strInfo + '\n')


def train(epochs, lr, cuda, train_loader, test_loader, out_features, model_savepath, log_path, save_max_acc):
    device = _select_device(cuda)

    ms_channels = parameter.get_value('ms_channels')
    radar_channels = parameter.get_value('radar_channels')
    net = GSSCNetPublic(
        out_features=out_features,
        radar_channels=radar_channels,
        ms_channels=ms_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    max_acc = 0.0
    best_epoch = 0
    sum_time = 0.0

    folder = os.path.dirname(model_savepath)
    if folder:
        os.makedirs(folder, exist_ok=True)

    getLog(log_path, parameter.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, 'Mode: public pre-acceptance backbone; cross-entropy training only')
    getLog(log_path, 'Saved model path: {}'.format(model_savepath))

    for epoch in range(epochs):
        since = time.time()
        net.train()
        last_loss = 0.0

        for hsi, sar, ms, tr_labels in train_loader:
            hsi = hsi.to(device).unsqueeze(1)
            sar = sar.to(device)
            ms = ms.to(device)
            tr_labels = tr_labels.to(device)

            optimizer.zero_grad()
            outputs = net(hsi, sar, ms)
            loss = criterion(outputs, tr_labels)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        net.eval()
        acc1 = 0.0
        test_base = parameter.get_value('test_base')

        if epoch % test_base == (test_base - 1) or epoch + 1 == epochs:
            pred_parts = []
            gt_parts = []
            with torch.no_grad():
                for hsi, sar, ms, gtlabels in test_loader:
                    hsi = hsi.to(device).unsqueeze(1)
                    sar = sar.to(device)
                    ms = ms.to(device)
                    outputs = net(hsi, sar, ms)
                    pred_parts.append(torch.argmax(outputs, dim=1).cpu().numpy())
                    gt_parts.append(gtlabels.cpu().numpy())

            if pred_parts:
                y_pred_test = np.concatenate(pred_parts)
                gty = np.concatenate(gt_parts)
                acc1 = accuracy_score(gty, y_pred_test)

                if acc1 < save_max_acc and acc1 > max_acc:
                    torch.save(net, model_savepath)
                    max_acc = acc1
                    best_epoch = epoch + 1

        time_elapsed = time.time() - since
        sum_time += time_elapsed
        rest_time = (sum_time / (epoch + 1)) * (epochs - epoch - 1)

        log = (
            '{} [Epoch: {}] [loss: {:.4f}] [acc: {:.4f}] [best: {:.4f}] '
            '[elapsed: {:.1f}s] [remaining: {:.1f}s]'
        ).format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            epoch + 1,
            last_loss,
            acc1,
            max_acc,
            time_elapsed,
            rest_time,
        )
        print(log)
        getLog(log_path, log)

    print('max_acc: {:.4f}'.format(max_acc))
    print('best_epoch: {}'.format(best_epoch))
    print('best_model:', model_savepath)
    getLog(log_path, 'max_acc: {}'.format(max_acc))
    getLog(log_path, 'best_epoch: {}'.format(best_epoch))
    getLog(log_path, 'best_model: {}'.format(model_savepath))
    getLog(log_path, '-------------------Finished Training-------------------')


def myTrain(datasetType):
    parameter.configure_dataset(datasetType)

    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    out_features = parameter.get_value('out_features')
    cuda = parameter.get_value('cuda')
    lr = parameter.get_value('lr')
    epoch_nums = parameter.get_value('epoch_nums')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    model_savepath = parameter.get_value('model_savepath')
    log_path = parameter.get_value('log_path')
    save_max_acc = parameter.get_value('save_max_acc')

    train_loader, test_loader, _, _ = getMyData(
        datasetType,
        channels,
        windowSize,
        batch_size,
        num_workers,
    )
    set_random_seed(random_seed)

    save_path = _make_save_path(model_savepath[datasetType], SUFFIX)
    train(
        epoch_nums,
        lr,
        cuda,
        train_loader,
        test_loader,
        out_features[datasetType],
        save_path,
        log_path[datasetType],
        save_max_acc[datasetType],
    )


if __name__ == '__main__':
    parameter.set_value('cuda', 'cuda0')
    datasetType = parameter.get_value('data_type')
    myTrain(datasetType)
