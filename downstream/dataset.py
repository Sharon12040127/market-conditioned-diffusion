# downstream/dataset.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class DownstreamDataset(Dataset):
    """
    下游预测任务的Dataset。
    输入：过去seq_len天的因子序列
    标签：第seq_len+1天的截面收益率排名（用于IC计算）
    支持混入MCD生成的增强数据。
    """

    def __init__(
        self,
        parquet_path: str,
        seq_len: int = 20,
        feature_cols: list = None,
        label_col: str = "ret_5d",      # 预测目标：5日后收益率
        augment_data: np.ndarray = None, # MCD生成的增强数据 (N, T, F)
        augment_regimes: np.ndarray = None,
        augment_ratio: float = 1.0,      # 生成数据/真实数据的比例
    ):
        if feature_cols is None:
            feature_cols = [
                "ret_5d", "ret_10d", "ret_20d",
                "vol_5d", "vol_20d", "volume_chg_5d",
                "macd", "macd_signal", "macd_hist", "rsi_14"
            ]

        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.label_col    = label_col

        df = pd.read_parquet(parquet_path)

        # ── 构造真实数据窗口 ──────────────────────────────────
        real_X, real_y, real_dates, real_stocks = [], [], [], []

        for stock, grp in df.groupby(level="instrument"):
            grp = grp.sort_index(level="datetime")
            # 改后：label用下一期的ret_5d，feature里的ret_5d是当期的，要shift
            features = grp[feature_cols].values.astype(np.float32)
            # label取窗口结束后第一天的ret_5d，即向前shift一期
            labels = grp[label_col].values.astype(np.float32)
            dates    = grp.index.get_level_values("datetime")

            valid = np.all(np.isfinite(features), axis=1) & np.isfinite(labels)
            features, labels, dates = features[valid], labels[valid], dates[valid]

            n = len(features)
            for i in range(n - seq_len):
                x = features[i: i + seq_len]          # (T, F)
                y = labels[i + seq_len]                # 下一期收益率
                if not np.all(np.isfinite(x)):
                    continue
                real_X.append(x)
                real_y.append(y)
                real_dates.append(dates[i + seq_len])
                real_stocks.append(stock)

        self.X      = np.array(real_X,    dtype=np.float32)   # (N, T, F)
        self.y      = np.array(real_y,    dtype=np.float32)   # (N,)
        self.dates  = np.array(real_dates)
        self.stocks = np.array(real_stocks)

        print(f"  真实样本数: {len(self.X)}")

        # ── 混入增强数据（如果有）────────────────────────────
        if augment_data is not None:
            n_aug = int(len(self.X) * augment_ratio)
            # 改后，允许有放回采样，这样ratio再大也能用
            n_aug = int(len(self.X) * augment_ratio)   # self.X是真实数据
            idx   = np.random.choice(len(augment_data), size=n_aug, replace=True)  # replace=True
            aug_X = augment_data[idx]                          # (n_aug, T, F)

            # 改后（整段替换）
            from sklearn.neighbors import NearestNeighbors

            real_X_arr      = np.array(real_X, dtype=np.float32)   # (N_real, T, F)
            real_y_arr      = np.array(real_y, dtype=np.float32)   # (N_real,)

            # 用每个窗口的最后一帧做KNN
            real_last_frame = real_X_arr[:, -1, :]    # (N_real, F)
            aug_last_frame  = aug_X[:, -1, :]         # (N_aug, F)

            print("  KNN匹配增强数据label...")
            knn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
            knn.fit(real_last_frame)
            _, indices      = knn.kneighbors(aug_last_frame)
            aug_y           = real_y_arr[indices.squeeze()].astype(np.float32)

            # 增强数据不参与截面排名计算，date和stock用占位符
            aug_dates  = np.array(["augmented"] * len(aug_X))
            aug_stocks = np.array(["augmented"] * len(aug_X))

            self.X      = np.concatenate([self.X,      aug_X],      axis=0)
            self.y      = np.concatenate([self.y,      aug_y],      axis=0)
            self.dates  = np.concatenate([self.dates,  aug_dates],  axis=0)
            self.stocks = np.concatenate([self.stocks, aug_stocks], axis=0)

            print(f"  增强样本数: {len(aug_X)}")
            print(f"  合计样本数: {len(self.X)}")

        # shuffle（训练集用）
        perm       = np.random.permutation(len(self.X))
        self.X      = self.X[perm]
        self.y      = self.y[perm]
        self.dates  = self.dates[perm]
        self.stocks = self.stocks[perm]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class TestDataset(Dataset):
    """
    测试集Dataset，保留date和stock信息用于截面IC计算。
    不做shuffle，不混增强数据。
    """

    def __init__(self, parquet_path: str, seq_len: int = 20, feature_cols: list = None, label_col: str = "ret_5d"):
        if feature_cols is None:
            feature_cols = [
                "ret_5d", "ret_10d", "ret_20d",
                "vol_5d", "vol_20d", "volume_chg_5d",
                "macd", "macd_signal", "macd_hist", "rsi_14"
            ]

        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.label_col    = label_col

        df = pd.read_parquet(parquet_path)
        X, y, dates, stocks = [], [], [], []

        for stock, grp in df.groupby(level="instrument"):
            grp      = grp.sort_index(level="datetime")
            features = grp[feature_cols].values.astype(np.float32)
            labels   = grp[label_col].values.astype(np.float32)
            dt       = grp.index.get_level_values("datetime")

            valid    = np.all(np.isfinite(features), axis=1) & np.isfinite(labels)
            features, labels, dt = features[valid], labels[valid], dt[valid]

            n = len(features)
            for i in range(n - seq_len):
                x = features[i: i + seq_len]
                if not np.all(np.isfinite(x)):
                    continue
                X.append(x)
                y.append(labels[i + seq_len])
                dates.append(dt[i + seq_len])
                stocks.append(stock)

        self.X      = np.array(X,      dtype=np.float32)
        self.y      = np.array(y,      dtype=np.float32)
        self.dates  = np.array(dates)
        self.stocks = np.array(stocks)
        print(f"  测试样本数: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )