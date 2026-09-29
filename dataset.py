import torch
import random
import numpy as np
import h5py
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import parameter
def load_mat_v73(path, var_name):
    with h5py.File(path, 'r') as f:
        data = f[var_name][:]  # 默认可能得到 (242, 1360, 886)
        data = np.transpose(data, (2, 1, 0))  # 转为 (886, 1360, 242)
        data = data.astype(np.float32)
    return data
# 设置随机数种子
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# PCA 降维
def applyPCA(data, n_components):
    h, w, b = data.shape
    pca = PCA(n_components=n_components)
    data = np.reshape(pca.fit_transform(np.reshape(data, (-1, b))), (h, w, -1))
    return data


# 从标签图中提取所有非零像素的坐标和标签
def get_pixel_coords_from_label_map(label_map):
    """返回 (rows, cols, labels) 三个数组"""
    rows, cols = np.where(label_map != 0)
    labels = label_map[rows, cols]
    return rows, cols, labels


class HXDataset(Dataset):
    """支持标签图（train/test label maps）的数据集，完全适配 PICNet，支持三模态"""

    def __init__(self, hsi, X, ms, label_map, windowSize, transform=None, train=False):
        """
        hsi: HSI 数据 (H, W, C)
        X: LiDAR/SAR 数据 (H, W) 或 (H, W, C)  ← 支持多通道
        ms: MS 数据 (H, W, C) 多光谱数据
        label_map: 标签图 (H, W)，0 表示未标注，>0 表示类别
        windowSize: patch 大小（奇数）
        transform: torchvision transform
        train: 是否训练模式（数据增强）
        """
        modes = ['symmetric', 'reflect']
        self.train = train
        self.pad = windowSize // 2
        self.windowSize = windowSize

        # Padding HSI
        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])

        # Padding LiDAR/SAR（支持多通道）
        if len(X.shape) == 2:
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif len(X.shape) == 3:
            self.X = np.pad(X, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])

        # Padding MS 数据
        if len(ms.shape) == 2:
            self.ms = np.pad(ms, ((self.pad, self.pad),
                                  (self.pad, self.pad)), mode=modes[windowSize % 2])
        elif len(ms.shape) == 3:
            self.ms = np.pad(ms, ((self.pad, self.pad),
                                  (self.pad, self.pad), (0, 0)), mode=modes[windowSize % 2])

        # 从标签图中提取所有有效像素坐标和标签（原始坐标，未 pad）
        rows, cols, labels = get_pixel_coords_from_label_map(label_map)
        self.pos = np.stack([rows, cols], axis=1)  # (N, 2)
        self.labels = labels  # (N,)

        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        label = torch.tensor(self.labels[index] - 1, dtype=torch.long)  # 转为 0-based

        # 取 patch（注意是 pad 后的坐标）
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]  # (H, W, C)
        X = self.X[h: h + self.windowSize, w: w + self.windowSize]      # (H, W) 或 (H, W, C)
        ms = self.ms[h: h + self.windowSize, w: w + self.windowSize]    # (H, W, C)

        # 转为 tensor
        hsi = self.transform(hsi).float()   # (C, H, W)
        X = self.transform(X).float()       # (C, H, W) 或 (1, H, W)
        ms = self.transform(ms).float()     # (C, H, W)

        # 处理 SAR/LiDAR 通道：确保是 (C, H, W) 格式
        # 如果是单通道 (H, W) -> (1, H, W)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)  # (H, W) -> (1, H, W)
        # 如果已经是 (C, H, W) 格式，保持不变（支持多通道）
        # 注意：此时 X.shape[0] 就是通道数

        # 数据增强（只在训练时）
        trans = [transforms.RandomHorizontalFlip(1.),
                 transforms.RandomVerticalFlip(1.)]
        if self.train:
            if random.random() < 0.5:
                i = random.randint(0, 1)
                hsi = trans[i](hsi)
                X = trans[i](X)
                ms = trans[i](ms)

        return hsi, X, ms, label

    def __len__(self):
        return self.pos.shape[0]


def getMyData(datasetType, channels, windowSize, batch_size, num_workers):
    """
    完全适配 PICNet 的数据加载函数
    返回: train_loader, test_loader, trntst_loader, all_loader
    """
    set_random_seed(42)  # 固定随机种子

    # ========== Houston 2013 ==========
    if datasetType == 0:
        print("Loading Houston 2013...")

        hsi_path = r'D:\dataset\Houston\Houston_HS.mat'
        lidar_path = r'D:\dataset\Houston\Houston_LiDAR.mat'
        ms_path = r'D:\dataset\Houston\Houston_MS.mat'
        train_label_path = r'D:\dataset\Houston\Houston_train.mat'
        test_label_path = r'D:\dataset\Houston\Houston_test.mat'

        # 加载数据
        hsi = loadmat(hsi_path)['Houston_HS']
        lidar = loadmat(lidar_path)['Houston_LiDAR']
        ms = loadmat(ms_path)['Houston_MS']
        train_label_map = loadmat(train_label_path)['Houston_train']
        test_label_map = loadmat(test_label_path)['Houston_test']

        # 自动获取雷达通道数
        if len(lidar.shape) == 2:
            radar_channels = 1
        elif len(lidar.shape) == 3:
            radar_channels = lidar.shape[2]
        else:
            radar_channels = 1
        parameter.set_value('radar_channels', radar_channels)
        print(f"Radar channels: {radar_channels}")

        # 自动获取 MS 通道数
        ms_channels = ms.shape[2] if len(ms.shape) == 3 else 1
        parameter.set_value('ms_channels', ms_channels)
        print(f"MS channels: {ms_channels}")

        # PCA 降维
        hsi = applyPCA(hsi, channels)

        # 创建数据集
        train_dataset = HXDataset(hsi, lidar, ms, train_label_map, windowSize, transform=ToTensor(), train=True)
        test_dataset = HXDataset(hsi, lidar, ms, test_label_map, windowSize, transform=ToTensor(), train=False)

        combined_label_map = np.zeros_like(train_label_map)
        combined_label_map[train_label_map != 0] = train_label_map[train_label_map != 0]
        combined_label_map[test_label_map != 0] = test_label_map[test_label_map != 0]
        trntst_dataset = HXDataset(hsi, lidar, ms, combined_label_map, windowSize, transform=ToTensor(), train=False)

        all_label_map = np.ones_like(train_label_map)
        all_dataset = HXDataset(hsi, lidar, ms, all_label_map, windowSize, transform=ToTensor(), train=False)

    # ========== Augsburg City ==========
    elif datasetType == 1:  # Augsburg City
        print("Loading Augsburg City...")

        hsi_path = r'D:\dataset\Augsburg_City\Augsburg_City_HS.mat'
        sar_path = r'D:\dataset\Augsburg_City\Augsburg_City_SAR.mat'
        ms_path = r'D:\dataset\Augsburg_City\Augsburg_City_MS.mat'
        train_label_path = r'D:\dataset\Augsburg_City\Augsburg_City_train_200.mat'
        test_label_path = r'D:\dataset\Augsburg_City\Augsburg_City_test_200.mat'

        hsi = load_mat_v73(hsi_path, 'Augsburg_City_HS')
        sar = loadmat(sar_path)['Augsburg_City_SAR']
        ms = loadmat(ms_path)['Augsburg_City_MS']
        train_label_map = loadmat(train_label_path)['Augsburg_City_train']
        test_label_map = loadmat(test_label_path)['Augsburg_City_test']

        print(f"HSI shape: {hsi.shape}")
        print(f"SAR shape: {sar.shape}")
        print(f"MS shape: {ms.shape}")

        # 自动获取雷达通道数
        if len(sar.shape) == 2:
            radar_channels = 1
        elif len(sar.shape) == 3:
            radar_channels = sar.shape[2]
        else:
            radar_channels = 1
        parameter.set_value('radar_channels', radar_channels)
        print(f"Radar channels: {radar_channels}")

        # 自动获取 MS 通道数
        ms_channels = ms.shape[2] if len(ms.shape) == 3 else 1
        parameter.set_value('ms_channels', ms_channels)
        print(f"MS channels: {ms_channels}")

        hsi = applyPCA(hsi, channels)

        # 创建数据集（SAR 保留所有通道）
        train_dataset = HXDataset(hsi, sar, ms, train_label_map, windowSize, transform=ToTensor(), train=True)
        test_dataset = HXDataset(hsi, sar, ms, test_label_map, windowSize, transform=ToTensor(), train=False)

        combined_label_map = np.zeros_like(train_label_map)
        combined_label_map[train_label_map != 0] = train_label_map[train_label_map != 0]
        combined_label_map[test_label_map != 0] = test_label_map[test_label_map != 0]
        trntst_dataset = HXDataset(hsi, sar, ms, combined_label_map, windowSize, transform=ToTensor(), train=False)

        all_label_map = np.ones_like(train_label_map)
        all_dataset = HXDataset(hsi, sar, ms, all_label_map, windowSize, transform=ToTensor(), train=False)

    # ========== Beijing ==========
    elif datasetType == 2:  # Beijing
        print("Loading Beijing...")

        hsi_path = r'D:\dataset\Beijing\Beijing_HS.mat'
        sar_path = r'D:\dataset\Beijing\Beijing_SAR.mat'
        ms_path = r'D:\dataset\Beijing\Beijing_MS.mat'
        train_label_path = r'D:\dataset\Beijing\Beijing_train.mat'
        test_label_path = r'D:\dataset\Beijing\Beijing_test.mat'

        # 加载数据
        # HSI 可能也需要 load_mat_v73（如果格式是 v7.3）
        try:
            hsi = load_mat_v73(hsi_path, 'Beijing_HS')
        except:
            hsi = loadmat(hsi_path)['Beijing_HS']

        sar = loadmat(sar_path)['Beijing_SAR']
        ms = loadmat(ms_path)['Beijing_MS']
        train_label_map = loadmat(train_label_path)['Beijing_train']
        test_label_map = loadmat(test_label_path)['Beijing_test']

        print(f"HSI shape: {hsi.shape}")
        print(f"SAR shape: {sar.shape}")
        print(f"MS shape: {ms.shape}")
        print(f"Train label shape: {train_label_map.shape}")
        print(f"Test label shape: {test_label_map.shape}")

        # 自动获取雷达通道数
        if len(sar.shape) == 2:
            radar_channels = 1
        elif len(sar.shape) == 3:
            radar_channels = sar.shape[2]
        else:
            radar_channels = 1
        parameter.set_value('radar_channels', radar_channels)
        print(f"Radar channels: {radar_channels}")

        # 自动获取 MS 通道数
        ms_channels = ms.shape[2] if len(ms.shape) == 3 else 1
        parameter.set_value('ms_channels', ms_channels)
        print(f"MS channels: {ms_channels}")

        # PCA 降维
        hsi = applyPCA(hsi, channels)

        # 创建数据集
        train_dataset = HXDataset(hsi, sar, ms, train_label_map, windowSize, transform=ToTensor(), train=True)
        test_dataset = HXDataset(hsi, sar, ms, test_label_map, windowSize, transform=ToTensor(), train=False)

        # 合并标签（用于全图可视化）
        combined_label_map = np.zeros_like(train_label_map)
        combined_label_map[train_label_map != 0] = train_label_map[train_label_map != 0]
        combined_label_map[test_label_map != 0] = test_label_map[test_label_map != 0]
        trntst_dataset = HXDataset(hsi, sar, ms, combined_label_map, windowSize, transform=ToTensor(), train=False)

        # 全图所有像素
        all_label_map = np.ones_like(train_label_map)
        all_dataset = HXDataset(hsi, sar, ms, all_label_map, windowSize, transform=ToTensor(), train=False)

    # ========== Berlin ==========
    elif datasetType == 3:  # Berlin
        print("Loading Berlin...")

        hsi_path = r'D:\dataset\Berlin\Berlin_HS.mat'
        sar_path = r'D:\dataset\Berlin\Berlin_SAR.mat'
        ms_path = r'D:\dataset\Berlin\Berlin_MS.mat'
        train_label_path = r'D:\dataset\Berlin\Berlin_train.mat'
        test_label_path = r'D:\dataset\Berlin\Berlin_test.mat'

        try:
            hsi = load_mat_v73(hsi_path, 'Berlin_HS')
        except Exception:
            hsi = loadmat(hsi_path)['Berlin_HS']

        sar = loadmat(sar_path)['Berlin_SAR']
        ms = loadmat(ms_path)['Berlin_MS']
        train_label_map = loadmat(train_label_path)['Berlin_train']
        test_label_map = loadmat(test_label_path)['Berlin_test']

        print(f"HSI shape: {hsi.shape}")
        print(f"SAR shape: {sar.shape}")
        print(f"MS shape: {ms.shape}")
        print(f"Train label shape: {train_label_map.shape}")
        print(f"Test label shape: {test_label_map.shape}")

        # 自动获取雷达通道数
        if len(sar.shape) == 2:
            radar_channels = 1
        elif len(sar.shape) == 3:
            radar_channels = sar.shape[2]
        else:
            radar_channels = 1
        parameter.set_value('radar_channels', radar_channels)
        print(f"Radar channels: {radar_channels}")

        # 自动获取 MS 通道数
        ms_channels = ms.shape[2] if len(ms.shape) == 3 else 1
        parameter.set_value('ms_channels', ms_channels)
        print(f"MS channels: {ms_channels}")

        # PCA 降维
        hsi = applyPCA(hsi, channels)

        # 创建数据集
        train_dataset = HXDataset(hsi, sar, ms, train_label_map, windowSize, transform=ToTensor(), train=True)
        test_dataset = HXDataset(hsi, sar, ms, test_label_map, windowSize, transform=ToTensor(), train=False)

        # 合并标签（用于全图可视化）
        combined_label_map = np.zeros_like(train_label_map)
        combined_label_map[train_label_map != 0] = train_label_map[train_label_map != 0]
        combined_label_map[test_label_map != 0] = test_label_map[test_label_map != 0]
        trntst_dataset = HXDataset(hsi, sar, ms, combined_label_map, windowSize, transform=ToTensor(), train=False)

        # 全图所有像素
        all_label_map = np.ones_like(train_label_map)
        all_dataset = HXDataset(hsi, sar, ms, all_label_map, windowSize, transform=ToTensor(), train=False)

    else:
        raise ValueError(f"Unsupported datasetType: {datasetType}")

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    trntst_loader = DataLoader(trntst_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    print("Data loading complete!")
    return train_loader, test_loader, trntst_loader, all_loader