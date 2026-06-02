# train_diffusion.py
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json

sys.path.append("model/diffusion")
sys.path.append("model/diffusion/Diffusion-TS")
from mcd import MarketConditionedDiffusion

# ── 配置 ─────────────────────────────────────────────────────
CONFIG = {
    # 数据
    "train_path": "data/processed/csi300_train_with_regime.parquet",
    "val_path":   "data/processed/csi300_val_with_regime.parquet",
    "seq_len":    20,
    "feature_cols": ["ret_5d", "ret_10d", "ret_20d", "vol_5d", "vol_20d",
                     "volume_chg_5d", "macd", "macd_signal", "macd_hist", "rsi_14"],
    # 模型
    "feature_size":     10,
    "n_regimes":        3,
    "d_model":          64,
    "n_layer_enc":      3,
    "n_layer_dec":      6,
    "n_heads":          4,
    "timesteps":        1000,
    "sampling_timesteps": 50,
    "beta_schedule":    "cosine",
    # 训练
    "batch_size":       256,
    "lr":               1e-4,
    "epochs":           200,
    "save_every":       50,
    "checkpoint_dir":   "checkpoints",
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Dataset ──────────────────────────────────────────────────
class StockWindowDataset(Dataset):
    """
    把每只股票的时序切成滑动窗口。
    每个样本：x (T, F)，regime_id (int，取窗口最后一天的标签)
    """
    def __init__(self, parquet_path: str, seq_len: int, feature_cols: list):
        df = pd.read_parquet(parquet_path)

        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.windows      = []   # list of (features_array, regime_id)

        # 按股票分组切窗口
        for stock, grp in df.groupby(level="instrument"):
            grp = grp.sort_index(level="datetime")
            features = grp[feature_cols].values.astype(np.float32)  # (days, F)
            regimes  = grp["regime_id"].values.astype(np.int64)      # (days,)

            # 去掉含NaN的行
            valid = np.all(np.isfinite(features), axis=1)
            features = features[valid]
            regimes  = regimes[valid]

            # 滑动窗口，步长1
            n = len(features)
            for i in range(n - seq_len + 1):
                x          = features[i: i + seq_len]        # (T, F)
                regime_id  = int(regimes[i + seq_len - 1])   # 窗口最后一天的regime
                if regime_id == -1:                           # 跳过没有regime标签的
                    continue
                self.windows.append((x, regime_id))

        print(f"  数据集: {parquet_path}")
        print(f"  样本数: {len(self.windows)}")
        regime_counts = {}
        for _, r in self.windows:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        print(f"  regime分布: {regime_counts}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, regime_id = self.windows[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(regime_id, dtype=torch.long)


# ── 训练函数 ──────────────────────────────────────────────────
def train():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # 保存config
    with open(f"{cfg['checkpoint_dir']}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── 数据
    print("\n加载数据集...")
    train_ds = StockWindowDataset(cfg["train_path"], cfg["seq_len"], cfg["feature_cols"])
    val_ds   = StockWindowDataset(cfg["val_path"],   cfg["seq_len"], cfg["feature_cols"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── 模型
    print("\n初始化模型...")
    model = MarketConditionedDiffusion(
        seq_length        = cfg["seq_len"],
        feature_size      = cfg["feature_size"],
        n_regimes         = cfg["n_regimes"],
        d_model           = cfg["d_model"],
        n_layer_enc       = cfg["n_layer_enc"],
        n_layer_dec       = cfg["n_layer_dec"],
        n_heads           = cfg["n_heads"],
        timesteps         = cfg["timesteps"],
        sampling_timesteps= cfg["sampling_timesteps"],
        beta_schedule     = cfg["beta_schedule"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # ── 优化器
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    # ── 训练loop
    print(f"\n开始训练，共{cfg['epochs']}个epoch，设备: {device}\n")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg["epochs"] + 1):

        # train
        model.train()
        train_losses = []
        for x, regime_id in train_loader:
            x         = x.to(device)
            regime_id = regime_id.to(device)
            optimizer.zero_grad()
            loss = model(x, regime_id=regime_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(train_losses)

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, regime_id in val_loader:
                x         = x.to(device)
                regime_id = regime_id.to(device)
                loss = model(x, regime_id=regime_id)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses) if val_losses else float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch:3d}/{cfg['epochs']}] "
              f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
              f"lr: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg['checkpoint_dir']}/best_model.pt")

        # 定期保存
        if epoch % cfg["save_every"] == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "history":    history,
            }, f"{cfg['checkpoint_dir']}/epoch_{epoch}.pt")
            print(f"  → checkpoint saved at epoch {epoch}")

    # 保存训练曲线
    with open(f"{cfg['checkpoint_dir']}/history.json", "w") as f:
        json.dump(history, f)
    print(f"\n训练完成，最佳val_loss: {best_val_loss:.4f}")
    print(f"模型保存在: {cfg['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__":
    train()