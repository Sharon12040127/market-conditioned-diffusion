import os
# 必须在所有qlib import之前设置，强制单线程
os.environ["QLIB_JOBLIB_BACKEND"] = "sequential"

import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
import numpy as np
from pathlib import Path

# ── 初始化 ───────────────────────────────────────────────────
qlib.init(
    provider_uri="~/.qlib/qlib_data/cn_data",
    region=REG_CN,
    joblib_backend="sequential",    # 关掉并行
)

START_DATE = "2010-02-01"
END_DATE   = "2020-09-25"

UNIVERSE   = "csi300"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 拉原始OHLCV ──────────────────────────────────────────────
print("拉取原始数据中...")

if __name__ == "__main__":
    instruments = D.instruments(UNIVERSE)
    prices = D.features(
        instruments,
        ["$close", "$open", "$high", "$low", "$volume"],
        start_time=START_DATE,
        end_time=END_DATE,
        freq="day",
    )
    prices.columns = ["close", "open", "high", "low", "volume"]
    prices.index.names = ["instrument", "datetime"]
    print(f"原始数据shape: {prices.shape}")

    # ── 按股票计算特征 ────────────────────────────────────────
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"]
        volume = df["volume"]
        feat   = pd.DataFrame(index=df.index)

        # 收益率
        for n in [5, 10, 20]:
            feat[f"ret_{n}d"] = close.pct_change(n)

        # 波动率
        daily_ret = close.pct_change()
        for n in [5, 20]:
            feat[f"vol_{n}d"] = daily_ret.rolling(n).std() * np.sqrt(252)

        # 成交量变化率
        feat["volume_chg_5d"] = volume.pct_change(5)

        # MACD
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        feat["macd"]      = macd
        feat["macd_signal"] = signal
        feat["macd_hist"] = macd - signal

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        # 在return feat之前加
        # 未来1日收益率（预测目标，训练时shift已经对齐）
        feat["future_ret_1d"] = close.pct_change(1).shift(-1)
        feat["future_ret_5d"] = close.pct_change(5).shift(-5)
        return feat

    print("计算特征中，预计2-3分钟...")
    feat_list = []
    stocks = prices.index.get_level_values("instrument").unique()
    for i, stock in enumerate(stocks):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(stocks)}")
        grp = prices.loc[stock].sort_index()
        f = compute_features(grp)
        f.index = pd.MultiIndex.from_product(
            [[stock], f.index], names=["instrument", "datetime"]
        )
        feat_list.append(f)

    features = pd.concat(feat_list).sort_index().dropna()

    print(f"\n特征矩阵shape: {features.shape}")
    print(f"股票数量: {features.index.get_level_values('instrument').nunique()}")
    print(f"特征列: {features.columns.tolist()}")

    # ── 保存 ─────────────────────────────────────────────────
    features.to_parquet(OUTPUT_DIR / "csi300_features.parquet")
    print(f"\n✅ 全量数据已保存")

    splits = {
        "train": ("2010-02-01", "2017-12-31"),
        "val":   ("2018-01-01", "2018-12-31"),
        "test":  ("2019-01-01", "2020-09-25"),
    }
    dates = features.index.get_level_values("datetime")
    for split, (s, e) in splits.items():
        mask = (dates >= s) & (dates <= e)
        features[mask].to_parquet(OUTPUT_DIR / f"csi300_{split}.parquet")
        print(f"✅ {split} shape={features[mask].shape}")