# downstream/evaluate.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict


def evaluate_downstream(
    predictions: np.ndarray,   # (N,) 模型预测的收益率
    labels: np.ndarray,        # (N,) 真实收益率
    dates: np.ndarray,         # (N,) 对应日期
    stocks: np.ndarray,        # (N,) 对应股票代码
    top_pct: float = 0.1,      # Top10%持仓
    annual_factor: int = 252,  # 年化系数
) -> Dict:
    """
    计算下游任务评估指标。

    Returns
    -------
    dict包含：
        ic_mean, ic_std, icir,
        rank_ic_mean, rank_ic_std, rank_icir,
        annual_return, sharpe,
        ic_series, rank_ic_series, daily_return_series
    """
    # 过滤掉增强数据（date == "augmented"）
    mask       = dates != "augmented"
    predictions = predictions[mask]
    labels      = labels[mask]
    dates       = dates[mask]
    stocks      = stocks[mask]

    unique_dates = sorted(np.unique(dates))

    ic_list        = []
    rank_ic_list   = []
    daily_ret_list = []

    for date in unique_dates:
        date_mask = dates == date
        pred_d    = predictions[date_mask]
        label_d   = labels[date_mask]

        # 至少要有5只股票才计算
        if len(pred_d) < 5:
            continue

        # ── IC（Pearson）
        ic, _ = stats.pearsonr(pred_d, label_d)
        if not np.isnan(ic):
            ic_list.append(ic)

        # ── Rank IC（Spearman）
        rank_ic, _ = stats.spearmanr(pred_d, label_d)
        if not np.isnan(rank_ic):
            rank_ic_list.append(rank_ic)

        # ── Top10%持仓日收益率
        n_top     = max(1, int(len(pred_d) * top_pct))
        top_idx   = np.argsort(pred_d)[::-1][:n_top]
        daily_ret = label_d[top_idx].mean()
        daily_ret_list.append(daily_ret)

    ic_arr        = np.array(ic_list)
    rank_ic_arr   = np.array(rank_ic_list)
    daily_ret_arr = np.array(daily_ret_list)

    # ── 汇总指标
    ic_mean   = ic_arr.mean()
    ic_std    = ic_arr.std()
    icir      = ic_mean / (ic_std + 1e-8)

    rank_ic_mean = rank_ic_arr.mean()
    rank_ic_std  = rank_ic_arr.std()
    rank_icir    = rank_ic_mean / (rank_ic_std + 1e-8)

    # 年化收益率
    annual_return = daily_ret_arr.mean() * annual_factor

    # Sharpe（假设无风险利率=0）
    sharpe = (daily_ret_arr.mean() / (daily_ret_arr.std() + 1e-8)) * np.sqrt(annual_factor)

    return {
        # 核心指标
        "ic_mean":       round(float(ic_mean),       4),
        "ic_std":        round(float(ic_std),         4),
        "icir":          round(float(icir),           4),
        "rank_ic_mean":  round(float(rank_ic_mean),  4),
        "rank_ic_std":   round(float(rank_ic_std),   4),
        "rank_icir":     round(float(rank_icir),      4),
        "annual_return": round(float(annual_return),  4),
        "sharpe":        round(float(sharpe),          4),
        # 时间序列（画图用）
        "ic_series":           ic_arr,
        "rank_ic_series":      rank_ic_arr,
        "daily_return_series": daily_ret_arr,
        "dates":               np.array(unique_dates[:len(ic_arr)]),
    }


def print_results(results: Dict, model_name: str = "", condition: str = ""):
    """格式化打印评估结果"""
    tag = f"{model_name} | {condition}" if condition else model_name
    print(f"\n{'='*55}")
    print(f"  {tag}")
    print(f"{'='*55}")
    print(f"  IC:          {results['ic_mean']:+.4f}  (std={results['ic_std']:.4f})")
    print(f"  ICIR:        {results['icir']:+.4f}")
    print(f"  Rank IC:     {results['rank_ic_mean']:+.4f}  (std={results['rank_ic_std']:.4f})")
    print(f"  Rank ICIR:   {results['rank_icir']:+.4f}")
    print(f"  Ann. Return: {results['annual_return']*100:+.2f}%")
    print(f"  Sharpe:      {results['sharpe']:+.4f}")
    print(f"{'='*55}")