"""
Plot regime distribution shift bar chart (Fig. 2).
CSI300 (left) and CSI500 (right), grouped by regime,
three bars per group: Train / Val / Test.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── data ──────────────────────────────────────────────────────────────────────
regimes = ["Bear", "Bull", "Sideways"]
splits  = ["Train", "Val", "Test"]

csi300 = np.array([
    [19.8, 11.3, 25.1],   # Bear:    train / val / test
    [30.5, 23.5, 47.6],   # Bull
    [49.8, 65.2, 27.3],   # Sideways
])

csi500 = np.array([
    [10.5,  4.7,  2.3],
    [48.2, 38.8, 56.3],
    [41.3, 56.5, 41.4],
])

# ── style ─────────────────────────────────────────────────────────────────────
colors = ["#E07070", "#6BAF6B", "#6B9FD4"]   # Train / Val / Test
hatches = ["", "//", ".."]
bar_w   = 0.22
x       = np.arange(len(regimes))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
fig.subplots_adjust(wspace=0.35)

for ax, data, title in zip(axes, [csi300, csi500], ["CSI300", "CSI500"]):
    for j, (split, color, hatch) in enumerate(zip(splits, colors, hatches)):
        offset = (j - 1) * bar_w
        bars = ax.bar(
            x + offset, data[:, j],
            width=bar_w, label=split,
            color=color, hatch=hatch,
            edgecolor="white", linewidth=0.6
        )
        # annotate value on top of each bar
        for bar, val in zip(bars, data[:, j]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=6.5, color="#333333"
            )

    # highlight the Sideways shift with a subtle arrow annotation
    # arrow from Train bar top to Val bar top for Sideways (index 2)
    sideways_train = data[2, 0]
    sideways_val   = data[2, 1]
    ax.annotate(
        "",
        xy  =(x[2] + 0 * bar_w, sideways_val + 1.5),
        xytext=(x[2] - 1 * bar_w, sideways_train + 1.5),
        arrowprops=dict(arrowstyle="->", color="#CC3333",
                        lw=1.2, connectionstyle="arc3,rad=-0.25"),
    )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel("Proportion (%)", fontsize=8.5)
    ax.set_ylim(0, 82)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

# shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", ncol=3,
           fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, -0.06))

fig.savefig("regime_shift.pdf", bbox_inches="tight", dpi=300)
fig.savefig("regime_shift.png", bbox_inches="tight", dpi=300)
print("Saved: regime_shift.pdf / regime_shift.png")
