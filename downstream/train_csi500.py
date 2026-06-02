# downstream/train.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("model/diffusion")
sys.path.append("model/diffusion/Diffusion-TS")
import json
import torch
import numpy as np
import pandas as pd
from torch import nn

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from torch.utils.data import TensorDataset

from model.vae import generate_vae
from model.gan import generate_gan



from downstream.dataset import DownstreamDataset, TestDataset
from downstream.model import get_model
from downstream.evaluate import evaluate_downstream, print_results
from mcd import MarketConditionedDiffusion

# ── 配置 ─────────────────────────────────────────────────────
CONFIG = {
    "train_path": "data/processed/csi500_train_with_regime.parquet",
    "val_path":   "data/processed/csi500_val_with_regime.parquet",
    "test_path":  "data/processed/csi500_test_with_regime.parquet",
    "seq_len":    20,
    "feature_cols": [
        "ret_5d", "ret_10d", "ret_20d", "vol_5d", "vol_20d",
        "volume_chg_5d", "macd", "macd_signal", "macd_hist", "rsi_14"
    ],
    "label_col":    "future_ret_1d",
    "feature_size": 10,
    # 下游模型
    "downstream_models": ["lstm", "transformer"],
    "hidden_size":  128,
    "d_model":      64,
    "n_layers":     2,
    "dropout":      0.1,
    # 训练
    "batch_size":   256,
    "lr":           1e-3,
    "epochs":       100,
    # 数据增强实验条件
    #"augment_ratios": [0.0, 0.5, 1.0],   # 0.0=无增强
    # MCD模型路径
    "mcd_checkpoint": "checkpoints_csi500/best_model.pt",
    "mcd_config": {
        "seq_length":         20,
        "feature_size":       10,
        "n_regimes":          3,
        "d_model":            64,
        "n_layer_enc":        3,
        "n_layer_dec":        6,
        "n_heads":            4,
        "timesteps":          1000,
        "sampling_timesteps": 50,
        "beta_schedule":      "cosine",
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "downstream/results_csi500",
    "n_generate": 100000,    # 每个regime生成多少样本
}


# ── 生成增强数据 ──────────────────────────────────────────────
def generate_augmented_data_by_regime(cfg: dict) -> dict:
    """返回 {regime_id: np.ndarray (N, T, F)}"""
    device = torch.device(cfg["device"])
    print("\n加载MCD模型生成增强数据...")

    model = MarketConditionedDiffusion(**cfg["mcd_config"]).to(device)
    state = torch.load(cfg["mcd_checkpoint"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    df      = pd.read_parquet(cfg["train_path"])
    df_vals = df[cfg["feature_cols"]].dropna().values.astype(np.float32)

    def sample_ref(bs):
        idx = np.random.randint(0, len(df_vals) - cfg["seq_len"], size=bs)
        return np.stack([df_vals[i:i + cfg["seq_len"]] for i in idx])

    generated_by_regime = {}
    n_per_regime = cfg["n_generate"] // cfg["mcd_config"]["n_regimes"]

    for regime_id in range(cfg["mcd_config"]["n_regimes"]):
        print(f"  生成regime={regime_id}的数据，{n_per_regime}个样本...")
        batch_size = 256
        regime_samples = []
        total_batches = (n_per_regime + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, n_per_regime, batch_size)):
            if batch_idx % 10 == 0:
                print(f"    [{batch_idx}/{total_batches}] generating...", flush=True)
            curr_bs   = min(batch_size, n_per_regime - start)
            ref_batch = torch.tensor(sample_ref(curr_bs), dtype=torch.float32).to(device)
            generated = model.generate(batch_size=curr_bs, regime_id=regime_id, ref_x=ref_batch)
            regime_samples.append(generated.cpu().numpy())

        generated_by_regime[regime_id] = np.concatenate(regime_samples, axis=0)

    return generated_by_regime

def generate_bootstrap(cfg: dict, n_samples: int) -> np.ndarray:
    """
    Bootstrap增强：对真实窗口做随机时间平移+高斯噪声。
    不需要训练任何模型，纯数据变换。
    """
    print("\n生成Bootstrap增强数据...")
    df      = pd.read_parquet(cfg["train_path"])
    df_vals = df[cfg["feature_cols"]].dropna().values.astype(np.float32)
    seq_len = cfg["seq_len"]

    # 计算每个特征的噪声强度（用标准差的5%）
    feat_std = df_vals.std(axis=0) * 0.05   # (F,)

    generated = []
    batch = 1024
    for start in range(0, n_samples, batch):
        curr_bs = min(batch, n_samples - start)

        # 随机采真实窗口
        idx  = np.random.randint(0, len(df_vals) - seq_len, size=curr_bs)
        wins = np.stack([df_vals[i:i + seq_len] for i in idx])  # (B, T, F)

        # 随机时间平移（前后各最多3步）
        shift = np.random.randint(-3, 4, size=curr_bs)
        shifted = []
        for b in range(curr_bs):
            s = shift[b]
            new_idx = max(0, min(idx[b] + s, len(df_vals) - seq_len))
            shifted.append(df_vals[new_idx:new_idx + seq_len])
        wins = np.stack(shifted)

        # 加高斯噪声
        noise = np.random.randn(*wins.shape).astype(np.float32) * feat_std
        wins  = wins + noise

        generated.append(wins)

    result = np.concatenate(generated, axis=0)
    print(f"  Bootstrap生成完成，样本数: {result.shape[0]}")
    return result


def generate_unconditional_diffusion(cfg: dict, n_samples: int) -> np.ndarray:
    """
    无条件Diffusion生成（消融baseline）：
    和MCD用同一个checkpoint，但生成时不传regime条件。
    """
    device = torch.device(cfg["device"])
    print("\n生成无条件Diffusion增强数据...")

    model = MarketConditionedDiffusion(**cfg["mcd_config"]).to(device)
    state = torch.load(cfg["mcd_checkpoint"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    df      = pd.read_parquet(cfg["train_path"])
    df_vals = df[cfg["feature_cols"]].dropna().values.astype(np.float32)

    def sample_ref(bs):
        idx = np.random.randint(0, len(df_vals) - cfg["seq_len"], size=bs)
        return np.stack([df_vals[i:i + cfg["seq_len"]] for i in idx])

    generated = []
    batch_size = 256
    total_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 10 == 0:
            print(f"  [{batch_idx}/{total_batches}] generating...")
        curr_bs   = min(batch_size, n_samples - start)
        ref_batch = torch.tensor(sample_ref(curr_bs), dtype=torch.float32).to(device)
        gen       = model.generate_unconditional(batch_size=curr_bs, ref_x=ref_batch)
        generated.append(gen.cpu().numpy())

    result = np.concatenate(generated, axis=0)
    print(f"  无条件Diffusion生成完成，样本数: {result.shape[0]}")
    return result

def distribution_aware_augmentation(cfg, generated_by_regime):
    """
    根据train/val的regime分布偏移，决定每个regime补充多少数据。
    generated_by_regime: dict {regime_id: np.ndarray (N, T, F)}
    """
    df_train = pd.read_parquet(cfg["train_path"])
    df_val   = pd.read_parquet(cfg["val_path"])
    
    train_dist = df_train.groupby("regime_id").size() / len(df_train)
    val_dist   = df_val.groupby("regime_id").size()   / len(df_val)
    
    # 对齐index
    all_regimes = range(cfg["mcd_config"]["n_regimes"])
    train_dist  = train_dist.reindex(all_regimes, fill_value=0)
    val_dist    = val_dist.reindex(all_regimes, fill_value=0)
    
    # 分布偏移：val里比train多的regime需要补充
    shift = (val_dist - train_dist).clip(lower=0)
    
    if shift.sum() == 0:
        # 没有偏移，均等采样
        shift = pd.Series([1/3]*3, index=all_regimes)
    else:
        shift = shift / shift.sum()
    
    n_augment = cfg["n_generate"]
    aug_list  = []
    
    for regime_id in all_regimes:
        n = int(shift[regime_id] * n_augment)
        if n > 0 and regime_id in generated_by_regime:
            idx = np.random.choice(
                len(generated_by_regime[regime_id]), n, replace=True
            )
            aug_list.append(generated_by_regime[regime_id][idx])
    
    if not aug_list:
        return None
    return np.concatenate(aug_list, axis=0)

# ── 训练单个下游模型 ──────────────────────────────────────────
def train_downstream(model, train_loader, val_loader, cfg, device,
                     aug_X=None):
    """
    两阶段训练：
    阶段1：如果有增强数据，用真实+增强数据预训练encoder（重建任务）
    阶段2：只用真实数据训练完整模型（预测任务）
    """
    # ── 阶段1：encoder预训练（只有aug_X时才做）─────────────────
    if aug_X is not None and len(aug_X) > 0:
        print("  [阶段1] encoder预训练（特征重建）...")

        # 重建head：encoder输出 -> 重建输入最后一帧
        hidden_dim = (cfg["hidden_size"] if hasattr(model, "lstm")
                      else cfg["d_model"])
        recon_head = nn.Linear(hidden_dim, cfg["feature_size"]).to(device)

        opt_pretrain = Adam(
            list(model.parameters()) + list(recon_head.parameters()),
            lr=cfg["lr"]
        )

        # 合并真实+增强数据（只要X，不要label）
        real_X = np.array([train_loader.dataset.X[i]
                           for i in range(len(train_loader.dataset))])
        all_X  = np.concatenate([real_X, aug_X], axis=0)
        all_X  = torch.tensor(all_X, dtype=torch.float32)
        pre_loader = DataLoader(
            TensorDataset(all_X),
            batch_size=cfg["batch_size"], shuffle=True,
            num_workers=4, pin_memory=True,
        )

        pretrain_epochs = 30
        for epoch in range(1, pretrain_epochs + 1):
            model.train(); recon_head.train()
            for (x,) in pre_loader:
                x = x.to(device)
                h = model.encode(x)                    # (B, hidden)
                pred_last = recon_head(h)              # (B, F)
                target    = x[:, -1, :]               # 重建最后一帧 (B, F)
                loss = nn.functional.mse_loss(pred_last, target)
                opt_pretrain.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_pretrain.step()
            if epoch % 10 == 0:
                print(f"    pretrain epoch {epoch}/{pretrain_epochs}  recon_loss={loss.item():.6f}")

        del recon_head, pre_loader, all_X  # 释放内存

    # ── 阶段2：只用真实数据训练预测head ──────────────────────────
    print("  [阶段2] 预测任务训练（仅真实数据）...")
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)
    criterion = nn.MSELoss()
    best_val  = float("inf")
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_preds.append(model(x.to(device)).cpu().numpy())
                all_labels.append(y.numpy())
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        from scipy.stats import pearsonr
        ic, _ = pearsonr(all_preds, all_labels)
        val_loss = -ic

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}/{cfg['epochs']}  val_IC={ic:.4f}")

    model.load_state_dict(best_state)
    return model

# ── 在测试集上推理 ────────────────────────────────────────────
def predict_on_test(model, test_ds, cfg, device):
    loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ── 主实验循环 ────────────────────────────────────────────────
def main():
    cfg    = CONFIG
    device = torch.device(cfg["device"])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)



    # # ── 生成所有增强数据（各方法）
    generated_by_regime  = generate_augmented_data_by_regime(cfg)
    augmented_mcd        = np.concatenate(list(generated_by_regime.values()), axis=0)
    augmented_mcd_adaptive = distribution_aware_augmentation(cfg, generated_by_regime)
    augmented_uncond     = generate_unconditional_diffusion(cfg, n_samples=cfg["n_generate"])
    augmented_boot       = generate_bootstrap(cfg, n_samples=cfg["n_generate"])
    augmented_vae        = generate_vae(
        n_samples=cfg["n_generate"],
        data_path=cfg["train_path"],
        feature_cols=cfg["feature_cols"],
        device=cfg["device"],
        checkpoint="checkpoints_csi500/vae_best.pt",
    )
    augmented_gan        = generate_gan(
        n_samples=cfg["n_generate"],
        feature_cols=cfg["feature_cols"],
        device=cfg["device"],
        checkpoint="checkpoints_csi500/gan_best.pt",
    )
    # ── 测试集（固定）
    print("\n加载测试集...")
    test_ds = TestDataset(
        cfg["test_path"], cfg["seq_len"],
        cfg["feature_cols"], cfg["label_col"]
    )

    # val集
    val_ds_base = DownstreamDataset(
        cfg["val_path"], cfg["seq_len"],
        cfg["feature_cols"], cfg["label_col"],
        augment_data=None,
    )
    val_loader = DataLoader(
        val_ds_base, batch_size=cfg["batch_size"], shuffle=False
    )

    # ── 实验条件定义
    conditions = {
        "no_aug":           None,
        "mcd_x1":           augmented_mcd,
        "mcd_adaptive":     augmented_mcd_adaptive,  # 新增
        "uncond_x1":        augmented_uncond,
        "bootstrap_x1":     augmented_boot,
        "vae_x1":           augmented_vae,
        "gan_x1":           augmented_gan,
    }
    # ratio统一用1.0做主实验，消融用不同ratio
    ratios = {k: 0.0 if k == "no_aug" else 1.0 for k in conditions}

    # 先加载已有结果
    out_path = f"{cfg['output_dir']}/all_results.json"
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            all_results = json.load(f)
        print(f"加载已有结果: {out_path}")
    else:
        all_results = {}

    for model_name in cfg["downstream_models"]:
        if model_name not in all_results:
            all_results[model_name] = {}

        for cond_name, aug_data in conditions.items():
            ratio = ratios[cond_name]
            print(f"\n{'─'*55}")
            print(f"下游模型: {model_name}  条件: {cond_name}")
            print(f"{'─'*55}")

            train_ds = DownstreamDataset(
                cfg["train_path"], cfg["seq_len"],
                cfg["feature_cols"], cfg["label_col"],
                augment_data=None,
                augment_ratio=ratio,
            )
            train_loader = DataLoader(
                train_ds, batch_size=cfg["batch_size"],
                shuffle=True, num_workers=4, pin_memory=True,
            )

            if model_name == "lstm":
                downstream = get_model(
                    "lstm", input_size=cfg["feature_size"],
                    hidden_size=cfg["hidden_size"],
                    num_layers=cfg["n_layers"],
                    dropout=cfg["dropout"],
                ).to(device)
            else:
                downstream = get_model(
                    "transformer", input_size=cfg["feature_size"],
                    d_model=cfg["d_model"],
                    n_heads=4,
                    n_layers=cfg["n_layers"],
                    dropout=cfg["dropout"],
                ).to(device)

            # 调用train_downstream时传aug_X
            aug_X = conditions[cond_name]   # (N, T, F) or None
            downstream = train_downstream(
                downstream, train_loader, val_loader, cfg, device,
                aug_X=aug_X,             # ← 传进去
            )
            preds   = predict_on_test(downstream, test_ds, cfg, device)
            print(f"  [DEBUG] preds std={preds.std():.6f}, mean={preds.mean():.6f}, min={preds.min():.6f}, max={preds.max():.6f}")
            results = evaluate_downstream(
                predictions=preds,
                labels=test_ds.y,
                dates=test_ds.dates,
                stocks=test_ds.stocks,
            )
            print_results(results, model_name=model_name, condition=cond_name)
            all_results[model_name][cond_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }

            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  已保存 {model_name}|{cond_name}")

    # 保存
    out_path = f"{cfg['output_dir']}/all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ 结果已保存到 {out_path}")

    print("\n" + "="*70)
    print("汇总对比表")
    print("="*70)
    print(f"{'模型':<12} {'条件':<16} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'年化':>10} {'Sharpe':>8}")
    print("-"*70)
    for mn in cfg["downstream_models"]:
        for cond, res in all_results[mn].items():
            print(
                f"{mn:<12} {cond:<16} "
                f"{res['ic_mean']:>+8.4f} "
                f"{res['icir']:>+8.4f} "
                f"{res['rank_ic_mean']:>+8.4f} "
                f"{res['annual_return']*100:>+9.2f}% "
                f"{res['sharpe']:>+8.4f}"
        )



if __name__ == "__main__":
    main()