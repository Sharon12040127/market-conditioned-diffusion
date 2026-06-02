# evaluate_generation.py
import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew, spearmanr
from statsmodels.stats.diagnostic import het_arch
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path

from model.vae import generate_vae
from model.gan import generate_gan
from model.gan import generate_gan


FEATURE_COLS = [
    "ret_5d","ret_10d","ret_20d","vol_5d","vol_20d",
    "volume_chg_5d","macd","macd_signal","macd_hist","rsi_14"
]
TRAIN_PATH = "data/processed/csi300_train_with_regime.parquet"
N_SAMPLES  = 50000


def financial_quality_metrics(real: np.ndarray, generated: np.ndarray, name="") -> dict:
    """
    real, generated: (N, T, F)
    """
    real_flat = real.reshape(-1, real.shape[-1])   # (N*T, F)
    gen_flat  = generated.reshape(-1, generated.shape[-1])

    results = {"method": name}

    # ── 1. 基础统计量 ────────────────────────────────────────
    results["mean_diff"]     = float(np.abs(
        real_flat.mean(0) - gen_flat.mean(0)).mean())
    results["std_diff"]      = float(np.abs(
        real_flat.std(0) - gen_flat.std(0)).mean())
    results["kurtosis_diff"] = float(np.abs(
        kurtosis(real_flat) - kurtosis(gen_flat)).mean())
    results["skew_diff"]     = float(np.abs(
        skew(real_flat) - skew(gen_flat)).mean())

    # ── 2. 自相关差异（用第一个特征ret_5d） ──────────────────
    def autocorr(x, lag):
        return np.array([
            np.corrcoef(x[:-lag, i], x[lag:, i])[0,1]
            for i in range(x.shape[1])
        ]).mean()

    for lag in [1, 5, 10]:
        results[f"autocorr_diff_lag{lag}"] = float(abs(
            autocorr(real_flat, lag) - autocorr(gen_flat, lag)))

    # ── 3. 波动聚集性（ARCH检验，p值越小说明聚集性越强） ─────
    def arch_pval(x):
        pvals = []
        for i in range(x.shape[1]):
            try:
                _, p, _, _ = het_arch(x[:, i])
                pvals.append(p)
            except Exception:
                pass
        return np.mean(pvals) if pvals else np.nan

    results["arch_pval_real"] = float(arch_pval(real_flat))
    results["arch_pval_gen"]  = float(arch_pval(gen_flat))
    results["arch_pval_diff"] = float(abs(
        results["arch_pval_real"] - results["arch_pval_gen"]))

    # ── 4. 尾部特性 ──────────────────────────────────────────
    def var_cvar(x, alpha=0.05):
        var  = np.percentile(x, alpha * 100, axis=0)
        cvar = np.array([x[x[:, i] <= var[i], i].mean()
                         for i in range(x.shape[1])])
        return var.mean(), cvar.mean()

    real_var,  real_cvar  = var_cvar(real_flat)
    gen_var,   gen_cvar   = var_cvar(gen_flat)
    results["var_diff"]  = float(abs(real_var  - gen_var))
    results["cvar_diff"] = float(abs(real_cvar - gen_cvar))

    # ── 5. MMD（RBF kernel） ─────────────────────────────────
    def mmd(x, y, n=2000):
        # 随机采样避免内存爆炸
        xi = x[np.random.choice(len(x), min(n, len(x)), replace=False)]
        yi = y[np.random.choice(len(y), min(n, len(y)), replace=False)]
        xx = rbf_kernel(xi, xi).mean()
        yy = rbf_kernel(yi, yi).mean()
        xy = rbf_kernel(xi, yi).mean()
        return float(xx + yy - 2*xy)

    results["mmd"] = mmd(real_flat, gen_flat)

    return results


def load_real_windows(path, feature_cols, seq_len=20, n=50000):
    df   = pd.read_parquet(path)
    vals = df[feature_cols].dropna().values.astype(np.float32)
    idx  = np.random.choice(len(vals) - seq_len, size=min(n, len(vals)-seq_len), replace=False)
    return np.stack([vals[i:i+seq_len] for i in idx])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载真实数据...")
    real = load_real_windows(TRAIN_PATH, FEATURE_COLS, n=N_SAMPLES)
    print(f"真实样本: {real.shape}")

    # 加载各方法生成数据
    # from train_diffusion import load_mcd_and_generate  # 你已有的MCD生成函数

    methods = {}

    print("生成VAE样本...")
    methods["VAE"] = generate_vae(
        n_samples=N_SAMPLES, data_path=TRAIN_PATH,
        feature_cols=FEATURE_COLS, device=device,
    )
    print("生成GAN样本...")
    methods["GAN"] = generate_gan(
        n_samples=N_SAMPLES, feature_cols=FEATURE_COLS, device=device,
    )

    print("生成MCD样本...")
    from downstream.train_v1 import generate_augmented_data_by_regime, distribution_aware_augmentation, CONFIG
    cfg = CONFIG.copy()
    cfg["n_generate"] = N_SAMPLES
    generated_by_regime    = generate_augmented_data_by_regime(cfg)
    methods["MCD"]         = np.concatenate(list(generated_by_regime.values()), axis=0)
    methods["MCD-Adaptive"] = distribution_aware_augmentation(cfg, generated_by_regime)

    # 无条件Diffusion
    print("生成无条件Diffusion样本...")
    from downstream.train_v1 import generate_unconditional_diffusion
    methods["Uncond-Diff"] = generate_unconditional_diffusion(cfg, n_samples=N_SAMPLES)

    # ── 计算所有指标 ─────────────────────────────────────────
    all_results = []
    for name, gen in methods.items():
        print(f"计算 {name} 的生成质量指标...")
        r = financial_quality_metrics(real, gen, name=name)
        all_results.append(r)
        print(f"  MMD={r['mmd']:.6f}  mean_diff={r['mean_diff']:.6f}  "
              f"std_diff={r['std_diff']:.6f}  arch_diff={r['arch_pval_diff']:.4f}")

    # 在main()最后加
    for name, gen in methods.items():
        flat = gen.reshape(len(gen), -1)
        # 随机取100对样本算平均距离，距离越小说明多样性越差
        idx1 = np.random.choice(len(flat), 100)
        idx2 = np.random.choice(len(flat), 100)
        diversity = np.mean(np.abs(flat[idx1] - flat[idx2]))
        print(f"{name} 多样性(avg pairwise dist): {diversity:.4f}")
    # ── 汇总表 ───────────────────────────────────────────────
    df_res = pd.DataFrame(all_results).set_index("method")
    print("\n" + "="*70)
    print("生成质量对比表")
    print("="*70)
    print(df_res.to_string())

    Path("downstream/results").mkdir(exist_ok=True)
    df_res.to_csv("downstream/results/generation_quality_v1.csv")
    print("\n✅ 已保存到 downstream/results/generation_quality_v1.csv")


if __name__ == "__main__":
    main()