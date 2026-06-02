import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

import qlib
from qlib.constant import REG_CN
from qlib.data import D


REGIME_COLORS = {
    0: "#E74C3C",  # Bear
    1: "#2ECC71",  # Bull
    2: "#3498DB",  # Sideways
}

REGIME_NAMES = {
    0: "Bear",
    1: "Bull",
    2: "Sideways",
}


class RegimeDetector:
    """
    HMM-based Market Regime Detector

    输入:
        market_return (日收益率)
        market_vol (波动率)

    输出:
        regime_id:
            0 -> Bear
            1 -> Bull
            2 -> Sideways
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 1000,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state

        self.scaler = StandardScaler()

        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-4,
            random_state=random_state,
            verbose=False,
        )

        self.regime_map = {}
        self._fitted = False

    # ==========================================================
    # Fit
    # ==========================================================
    def fit(self, market_df: pd.DataFrame):

        market_df = market_df.dropna().sort_index()

        X_raw = market_df[
            ["market_return", "market_vol"]
        ].values

        # 标准化（非常重要）
        X = self.scaler.fit_transform(X_raw)

        # HMM训练
        self.model.fit(X)

        # 预测state
        raw_labels = self.model.predict(X)

        # ------------------------------------------------------
        # 根据收益率语义映射:
        # lowest return -> Bear
        # highest return -> Bull
        # remaining     -> Sideways
        # ------------------------------------------------------
        state_stats = {}

        for s in range(self.n_states):

            mask = raw_labels == s

            if mask.sum() > 0:
                state_stats[s] = {
                    "ret": X_raw[mask, 0].mean(),
                    "vol": X_raw[mask, 1].mean(),
                    "count": mask.sum(),
                }

        bear_state = min(
            state_stats,
            key=lambda s: state_stats[s]["ret"]
        )

        bull_state = max(
            state_stats,
            key=lambda s: state_stats[s]["ret"]
        )

        sideways_state = [
            s for s in state_stats
            if s not in [bear_state, bull_state]
        ][0]

        self.remap = {
            bear_state: 0,
            bull_state: 1,
            sideways_state: 2,
        }

        labels = np.array([
            self.remap[l] for l in raw_labels
        ])

        self._market_df = market_df
        self._labels = labels

        self.regime_map = dict(
            zip(market_df.index, labels)
        )

        self._fitted = True

        self._print_stats(X_raw, labels)

        return self

    # ==========================================================
    # Predict
    # ==========================================================
    def predict(self, market_df: pd.DataFrame):

        self._check_fitted()

        market_df = market_df.dropna().sort_index()

        X_raw = market_df[
            ["market_return", "market_vol"]
        ].values

        X = self.scaler.transform(X_raw)

        raw = self.model.predict(X)

        labels = np.array([
            self.remap[r] for r in raw
        ])

        return pd.Series(
            labels,
            index=market_df.index,
            name="regime_id",
        )

    # ==========================================================
    # Align to stock data
    # ==========================================================
    def align_to_stock_data(
        self,
        stock_df: pd.DataFrame
    ):

        self._check_fitted()

        out = stock_df.copy()

        dates = out.index.get_level_values("datetime")

        out["regime_id"] = dates.map(
            lambda d: self.regime_map.get(d, -1)
        )

        missing = (out["regime_id"] == -1).sum()

        if missing > 0:
            print(
                f"⚠️ missing regime labels: {missing}"
            )

        return out

    # ==========================================================
    # Visualization
    # ==========================================================
    def plot(
        self,
        title="Market Regime Detection",
        save_path: Optional[str] = None,
    ):

        self._check_fitted()

        dates = self._market_df.index
        ret = self._market_df["market_return"].values
        labels = self._labels

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(16, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # ------------------------------------------------------
        # Top plot
        # ------------------------------------------------------
        ax = axes[0]

        ax.plot(
            dates,
            ret,
            color="black",
            linewidth=0.8,
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=0.5,
            color="gray",
        )

        ax.set_title(title)
        ax.set_ylabel("Daily Market Return")

        self._fill_background(
            ax,
            dates,
            labels,
        )

        patches = [
            mpatches.Patch(
                color=REGIME_COLORS[s],
                label=REGIME_NAMES[s],
                alpha=0.3,
            )
            for s in range(3)
        ]

        ax.legend(handles=patches)

        # ------------------------------------------------------
        # Bottom plot
        # ------------------------------------------------------
        ax2 = axes[1]

        window = 60

        for s in range(3):

            freq = pd.Series(
                (labels == s).astype(float),
                index=dates,
            )

            ax2.plot(
                dates,
                freq.rolling(window).mean(),
                label=REGIME_NAMES[s],
                color=REGIME_COLORS[s],
            )

        ax2.set_ylim(0, 1)
        ax2.set_ylabel("60d Frequency")
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight",
            )

            print(f"✅ saved figure -> {save_path}")

        plt.show()

    def _fill_background(
        self,
        ax,
        dates,
        labels,
    ):

        i = 0

        while i < len(labels):

            j = i

            while (
                j < len(labels)
                and labels[j] == labels[i]
            ):
                j += 1

            ax.axvspan(
                dates[i],
                dates[j - 1],
                color=REGIME_COLORS[labels[i]],
                alpha=0.25,
                linewidth=0,
            )

            i = j

    # ==========================================================
    # Stats
    # ==========================================================
    def _print_stats(
        self,
        X_raw,
        labels,
    ):

        print("\n── Regime Statistics ─────────────────────")

        for s in range(3):

            mask = labels == s

            if mask.sum() > 0:
                ret = X_raw[mask, 0]
                vol = X_raw[mask, 1]

                print(
                    f"{REGIME_NAMES[s]:10s}"
                    f" count={mask.sum():4d} ({mask.sum()/len(labels)*100:.1f}%)"
                    f" avg_ret={ret.mean()*100:+.3f}%"
                    f" avg_vol={vol.mean():.4f}"
                )

        print("──────────────────────────────────────────\n")

    # ==========================================================
    # Utility
    # ==========================================================
    def get_regime_series(self):

        self._check_fitted()

        return pd.Series(
            self._labels,
            index=self._market_df.index,
            name="regime_id",
        )

    def save(self, path):

        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"✅ saved -> {path}")

    @classmethod
    def load(cls, path):

        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"✅ loaded -> {path}")

        return obj

    def _check_fitted(self):

        if not self._fitted:
            raise RuntimeError(
                "Call fit() first."
            )


# ==============================================================
# Main
# ==============================================================
if __name__ == "__main__":

    os.environ["QLIB_JOBLIB_BACKEND"] = "sequential"

    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_data",
        region=REG_CN,
        joblib_backend="sequential",
    )

    print("Loading CSI500 data...")

    prices = D.features(
        D.instruments("csi500"),
        ["$close"],
        start_time="2010-01-01",
        end_time="2023-12-31",
        freq="day",
    )

    prices.columns = ["close"]

    prices.index.names = [
        "instrument",
        "datetime",
    ]

    # ==========================================================
    # Build market features
    # ==========================================================
    close_panel = prices["close"].unstack(
        level="instrument"
    )

    # 删除缺失太多的股票
    close_panel = close_panel.dropna(axis=1, thresh=len(close_panel) * 0.7)
    print(f"有效股票数量: {close_panel.shape[1]}")


    # 日收益率
    daily_ret = close_panel.pct_change(
        fill_method=None
    )

    # ----------------------------------------------------------
    # Feature 1: 日收益率（关键改动！）
    # ----------------------------------------------------------
    market_return = (
        daily_ret.mean(axis=1)
        .rename("market_return")
    )

    # ----------------------------------------------------------
    # Feature 2: EWMA volatility
    # ----------------------------------------------------------
    market_vol = (
        daily_ret.mean(axis=1)
        .ewm(span=20, adjust=False)
        .std()
        .rename("market_vol")
    )

    market_df = pd.concat(
        [market_return, market_vol],
        axis=1,
    ).dropna()

    print("\nMarket data statistics:")
    print(f"  日期范围: {market_df.index.min()} 到 {market_df.index.max()}")
    print(f"  总天数: {len(market_df)}")
    print(f"\n  收益率统计:")
    print(f"    均值: {market_df['market_return'].mean()*100:.4f}%")
    print(f"    标准差: {market_df['market_return'].std()*100:.4f}%")
    print(f"    最小值: {market_df['market_return'].min()*100:.2f}%")
    print(f"    最大值: {market_df['market_return'].max()*100:.2f}%")
    print(f"\n  波动率统计:")
    print(f"    均值: {market_df['market_vol'].mean():.4f}")
    print(f"    标准差: {market_df['market_vol'].std():.4f}")

    # ==========================================================
    # Train
    # ==========================================================
    detector = RegimeDetector(
        n_states=3,
        n_iter=1000,
        random_state=42,
    )

    detector.fit(market_df)

    # 在训练后添加
    print("\n=== 验证熊市捕捉大跌日 ===")
    top_losses = market_df.nsmallest(10, 'market_return')
    for date, row in top_losses.iterrows():
        regime = detector.regime_map.get(date, -1)
        print(f"{date.strftime('%Y-%m-%d')}: {row['market_return']*100:.2f}% -> {REGIME_NAMES[regime]}")

    # ==========================================================
    # Plot
    # ==========================================================
    Path("data/figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    detector.plot(
        save_path="data/figures/regime_detection_csi500.png"
    )

    # ==========================================================
    # Check data ranges before alignment
    # ==========================================================
    print("\n" + "="*60)
    print("数据范围检查")
    print("="*60)
    
    print(f"Market data range: {market_df.index.min()} to {market_df.index.max()}")
    
    splits = {
        "train": ("2010-02-01", "2017-12-31"),
        "val":   ("2018-01-01", "2018-12-31"),
        "test":  ("2019-01-01", "2020-09-25"),
    }

    for split, (s, e) in splits.items():
        src = f"data/processed/csi500_{split}.parquet"
        
        if Path(src).exists():
            df = pd.read_parquet(src)
            dates = df.index.get_level_values("datetime")
            print(f"\n{split.upper()} data range: {dates.min()} to {dates.max()}")
            print(f"  {split} data days: {len(dates.unique())}")
            
            # 检查与market data的重叠
            market_dates = market_df.index
            overlap = dates.unique().intersection(market_dates)
            print(f"  Overlap with market data: {len(overlap)} days")
            
            if len(overlap) == 0:
                print(f"  ⚠️ WARNING: No overlap! {split} will be empty!")
        else:
            print(f"\n⚠️ {src} not found!")

    print("="*60 + "\n")

    # ==========================================================
    # Align regime to stock data
    # ==========================================================
    for split, (s, e) in splits.items():

        src = f"data/processed/csi500_{split}.parquet"
        
        if not Path(src).exists():
            print(f"⚠️ Skipping {split}: {src} not found")
            continue
            
        df = pd.read_parquet(src)

        df_with_regime = detector.align_to_stock_data(df)

        out_path = (
            f"data/processed/"
            f"csi500_{split}_with_regime.parquet"
        )

        df_with_regime.to_parquet(out_path)

        print(f"\n✅ {split} saved -> {out_path}")
        print(f"   Shape: {df_with_regime.shape}")
        
        regime_counts = df_with_regime["regime_id"].value_counts().sort_index()
        print(f"   Regime distribution:")
        for regime_id, count in regime_counts.items():
            if regime_id == -1:
                print(f"     Unlabeled: {count}")
            else:
                print(f"     {REGIME_NAMES[regime_id]}: {count}")

    # ==========================================================
    # Save model
    # ==========================================================
    Path("data/models").mkdir(
        parents=True,
        exist_ok=True,
    )

    detector.save(
        "data/models/regime_detector_csi500.pkl"
    )
    
    print("\n✅ All done!")