import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path


class TimeSeriesVAE(nn.Module):
    def __init__(self, seq_len=20, feature_size=10, latent_dim=64, hidden_dim=256):
        super().__init__()
        input_dim = seq_len * feature_size
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h      = self.encoder(x.view(x.size(0), -1))
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -4, 4)
        z      = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon  = self.decoder(z).view(-1, self.seq_len, self.feature_size)
        return recon, mu, logvar

    def generate(self, n, device):
        z = torch.randn(n, self.latent_dim).to(device)
        with torch.no_grad():
            return self.decoder(z).view(-1, self.seq_len, self.feature_size)


def train_vae(
    data_path="data/processed/csi500_train_with_regime.parquet",
    feature_cols=None,
    seq_len=20, latent_dim=64, hidden_dim=256,
    epochs=100, batch_size=512, lr=1e-3,
    device="cuda", save_path="checkpoints_csi500/vae_best.pt",
):
    if feature_cols is None:
        feature_cols = [
            "ret_5d","ret_10d","ret_20d","vol_5d","vol_20d",
            "volume_chg_5d","macd","macd_signal","macd_hist","rsi_14"
        ]
    Path("checkpoints_csi500").mkdir(exist_ok=True)
    device = torch.device(device)

    df   = pd.read_parquet(data_path)
    vals = df[feature_cols].dropna().values.astype(np.float32)
    mean = vals.mean(axis=0)
    std  = vals.std(axis=0) + 1e-8
    vals = (vals - mean) / std

    windows = []
    for i in range(len(vals) - seq_len):
        w = vals[i:i + seq_len]
        if np.all(np.isfinite(w)):
            windows.append(w)
    X      = torch.tensor(np.array(windows), dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size,
                        shuffle=True, drop_last=True, num_workers=2)
    print(f"VAE训练样本数: {len(X)}")

    model    = TimeSeriesVAE(seq_len, len(feature_cols), latent_dim, hidden_dim).to(device)
    opt      = Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # KL权重从0线性增加到0.0001，避免早期KL爆炸
        kl_weight = 1e-4 * min(epoch / 50, 1.0)
        model.train()
        total = 0
        for (x,) in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            recon_loss = nn.functional.mse_loss(recon, x)
            kl_loss    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss       = recon_loss + kl_weight * kl_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += recon_loss.item()   # 只看recon loss，KL不计入best判断

        avg = total / len(loader)
        if epoch % 20 == 0:
            print(f"  VAE epoch {epoch}/{epochs}  recon_loss={avg:.6f}  kl_w={kl_weight:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": model.state_dict(), "mean": mean, "std": std}, save_path)

    print(f"VAE训练完成，best recon_loss={best_loss:.6f}，已保存到{save_path}")
    return model, mean, std


def generate_vae(
    n_samples=100000,
    data_path="data/processed/csi500_train_with_regime.parquet",
    feature_cols=None,
    seq_len=20, latent_dim=64, hidden_dim=256,
    checkpoint="checkpoints_csi500/vae_best.pt",
    device="cuda",
):
    if feature_cols is None:
        feature_cols = [
            "ret_5d","ret_10d","ret_20d","vol_5d","vol_20d",
            "volume_chg_5d","macd","macd_signal","macd_hist","rsi_14"
        ]
    device = torch.device(device)
    ckpt   = torch.load(checkpoint, map_location=device)
    mean, std = ckpt["mean"], ckpt["std"]

    model = TimeSeriesVAE(seq_len, len(feature_cols), latent_dim, hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = []
    for start in range(0, n_samples, 256):
        curr = min(256, n_samples - start)
        gen  = model.generate(curr, device).cpu().numpy()
        gen  = gen * std + mean
        results.append(gen)
    return np.concatenate(results, axis=0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_vae(device=device)