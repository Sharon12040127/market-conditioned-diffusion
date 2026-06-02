# model/gan.py
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, noise_dim=32, seq_len=20, feature_size=10, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, seq_len * feature_size),
        )

    def forward(self, z):
        return self.net(z).view(-1, self.seq_len, self.feature_size)


class Discriminator(nn.Module):
    def __init__(self, seq_len=20, feature_size=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * feature_size, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def train_gan(data_path, feature_cols, seq_len=20, noise_dim=32,
              hidden_dim=128, epochs=100, batch_size=256, lr=2e-4,
              device="cuda", save_path="checkpoints/gan_best.pt"):
    import pandas as pd
    from pathlib import Path
    Path("checkpoints").mkdir(exist_ok=True)

    df   = pd.read_parquet(data_path)
    vals = df[feature_cols].dropna().values.astype(np.float32)

    # 标准化（GAN对尺度敏感）
    mean = vals.mean(axis=0)
    std  = vals.std(axis=0) + 1e-8
    vals = (vals - mean) / std

    windows = []
    for i in range(len(vals) - seq_len):
        w = vals[i:i + seq_len]
        if np.all(np.isfinite(w)):
            windows.append(w)
    X = torch.tensor(np.array(windows), dtype=torch.float32)
    print(f"GAN训练样本数: {len(X)}")

    loader = DataLoader(TensorDataset(X), batch_size=batch_size,
                        shuffle=True, drop_last=True)

    G = Generator(noise_dim, seq_len, len(feature_cols), hidden_dim).to(device)
    D = Discriminator(seq_len, len(feature_cols), hidden_dim).to(device)

    opt_G = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    best_g_loss = float("inf")
    for epoch in range(1, epochs + 1):
        g_losses, d_losses = [], []
        for (real,) in loader:
            real = real.to(device)
            bs   = real.size(0)

            # ── Train D
            z        = torch.randn(bs, noise_dim).to(device)
            fake     = G(z).detach()
            real_lbl = torch.ones(bs, 1).to(device)
            fake_lbl = torch.zeros(bs, 1).to(device)
            d_loss   = criterion(D(real), real_lbl) + criterion(D(fake), fake_lbl)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ── Train G
            z      = torch.randn(bs, noise_dim).to(device)
            fake   = G(z)
            g_loss = criterion(D(fake), real_lbl)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        if epoch % 20 == 0:
            print(f"  GAN epoch {epoch}/{epochs}  "
                  f"G={np.mean(g_losses):.4f}  D={np.mean(d_losses):.4f}")

        avg_g = np.mean(g_losses)
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save({
                "G": G.state_dict(), "D": D.state_dict(),
                "mean": mean, "std": std,
            }, save_path)

    print(f"GAN训练完成，已保存到{save_path}")
    return G, mean, std


def generate_gan(n_samples, feature_cols, seq_len=20, noise_dim=32,
                 hidden_dim=128, checkpoint="checkpoints/gan_best.pt",
                 device="cuda"):
    device = torch.device(device)
    ckpt   = torch.load(checkpoint, map_location=device)
    mean, std = ckpt["mean"], ckpt["std"]

    G = Generator(noise_dim, seq_len, len(feature_cols), hidden_dim).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    results = []
    bs = 256
    with torch.no_grad():
        for start in range(0, n_samples, bs):
            curr = min(bs, n_samples - start)
            z    = torch.randn(curr, noise_dim).to(device)
            gen  = G(z).cpu().numpy()
            gen  = gen * std + mean   # 反标准化
            results.append(gen)

    return np.concatenate(results, axis=0)


if __name__ == "__main__":
    FEATURE_COLS = [
        "ret_5d","ret_10d","ret_20d","vol_5d","vol_20d",
        "volume_chg_5d","macd","macd_signal","macd_hist","rsi_14"
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_gan(
        data_path="data/processed/csi300_train_with_regime.parquet",
        feature_cols=FEATURE_COLS,
        device=device,
    )