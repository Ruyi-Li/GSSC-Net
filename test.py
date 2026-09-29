from dataset import *

import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

import parameter

parameter._init()

SUFFIX = '_public_backbone'

DATASET_NAMES = {
    0: 'Houston2013',
    1: 'Augsburg_City',
    2: 'Beijing',
    3: 'Berlin',
}

CLASS_NAMES = {
    0: [
        'Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil',
        'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway',
        'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track',
    ],
}


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


def _make_report_path(report_path, suffix):
    root, ext = os.path.splitext(report_path)
    if ext == '':
        ext = '.txt'
    return root + suffix + ext


def _torch_load_any(model_savepath, device):
    try:
        return torch.load(model_savepath, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_savepath, map_location=device)


def _get_class_names(datasetType, num_classes):
    if datasetType in CLASS_NAMES and len(CLASS_NAMES[datasetType]) == num_classes:
        return CLASS_NAMES[datasetType]
    return ['Class {}'.format(i + 1) for i in range(num_classes)]


@torch.no_grad()
def _collect_predictions(net, test_loader, device):
    net.eval()
    pred_list = []
    gt_list = []

    for hsi, sar, ms, labels in test_loader:
        hsi = hsi.to(device).unsqueeze(1)
        sar = sar.to(device)
        ms = ms.to(device)
        labels = labels.to(device)

        outputs = net(hsi, sar, ms)
        pred_list.append(torch.argmax(outputs, dim=1).cpu())
        gt_list.append(labels.cpu())

    if not pred_list:
        raise RuntimeError('Empty test_loader.')

    return torch.cat(pred_list).numpy(), torch.cat(gt_list).numpy()


def _metrics(pred, gt, num_classes):
    labels = list(range(num_classes))
    oa = accuracy_score(gt, pred) * 100.0
    cm = confusion_matrix(gt, pred, labels=labels)
    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    recalls = np.divide(per_class_correct, np.maximum(per_class_total, 1), dtype=np.float64)
    aa = recalls.mean() * 100.0
    kappa = cohen_kappa_score(gt, pred, labels=labels) * 100.0
    return oa, aa, kappa, cm


def _write_report(save_path, datasetType, model_path, pred, gt, num_classes):
    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    labels = list(range(num_classes))
    target_names = _get_class_names(datasetType, num_classes)
    oa, aa, kappa, cm = _metrics(pred, gt, num_classes)
    cls_report = classification_report(
        gt,
        pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('-----------------------testInfo-----------------------\n')
        f.write('Report time: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        f.write('Dataset: {}\n'.format(DATASET_NAMES.get(datasetType, str(datasetType))))
        f.write('Mode: public pre-acceptance backbone\n')
        f.write('Model path: {}\n'.format(model_path))
        f.write('------------------------------------------------------\n\n')
        f.write('{:.12f} Overall accuracy (%)\n'.format(oa))
        f.write('{:.12f} Average accuracy (%)\n'.format(aa))
        f.write('{:.12f} Kappa accuracy (%)\n\n'.format(kappa))
        f.write(cls_report)
        f.write('\n')
        f.write(str(cm))
        f.write('\n')

    print('Report saved:', save_path)
    print('OA: {:.4f}%  AA: {:.4f}%  Kappa: {:.4f}%'.format(oa, aa, kappa))
    return oa, aa, kappa


def myTest(datasetType):
    parameter.configure_dataset(datasetType)

    cuda = parameter.get_value('cuda')
    device = _select_device(cuda)
    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    model_savepath_list = parameter.get_value('model_savepath')
    report_path_list = parameter.get_value('report_path')
    out_features_list = parameter.get_value('out_features')

    _, test_loader, _, _ = getMyData(
        datasetType,
        channels,
        windowSize,
        batch_size,
        num_workers,
    )
    set_random_seed(random_seed)

    model_path = _make_save_path(model_savepath_list[datasetType], SUFFIX)
    report_path = _make_report_path(report_path_list[datasetType], SUFFIX)

    if not os.path.exists(model_path):
        raise FileNotFoundError('Model file not found: {}'.format(model_path))

    print('=' * 70)
    print('Dataset:', DATASET_NAMES.get(datasetType, datasetType))
    print('Device:', device)
    print('Model path:', model_path)
    print('Report path:', report_path)
    print('=' * 70)

    net = _torch_load_any(model_path, device).to(device)
    net.eval()
    pred, gt = _collect_predictions(net, test_loader, device)
    _write_report(report_path, datasetType, model_path, pred, gt, out_features_list[datasetType])


if __name__ == '__main__':
    parameter.set_value('cuda', 'cuda0')
    datasetType = parameter.get_value('data_type')
    myTest(datasetType)
