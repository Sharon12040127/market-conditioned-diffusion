"""
Fig. 1: MCD Framework — ICDM double-column (figure*), 7.2 in wide.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT = "data/figures"
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(1.1, 7.0)
ax.axis("off")

C = dict(
    bear="#C0392B", bull="#27AE60", side="#2980B9",
    box="#F5F5F5",  bord="#BBBBBB", arr="#444444",
    txt="#1A1A1A",  wht="#FFFFFF",  gry="#777777",
    m1="#FFFBF0",   m2="#EDF6FC",   m3="#EFF7EF",  m4="#F4F0FC",
    pool="#E8DCF5", pb="#7B5EBF",   alloc="#2980B9",
    s1b="#D5C5EE",  s1e="#8E6BBF",
    s2b="#BBA8E0",  s2e="#6B3FAF",  out="#6B3FAF",
)

def rb(x,y,w,h, fc, ec=None, lw=1.0, rad=0.10, z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=fc, edgecolor=ec or C["bord"], linewidth=lw, zorder=z))

def tx(x,y,s, fs=7.5, c=None, ha="center", va="center", bold=False, z=5):
    ax.text(x,y,s, fontsize=fs, color=c or C["txt"], ha=ha, va=va,
            fontweight="bold" if bold else "normal", zorder=z, clip_on=False)

def ar(x1,y1,x2,y2, c=None, lw=1.2, cs="arc3,rad=0.0"):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle="-|>", color=c or C["arr"],
                        lw=lw, connectionstyle=cs, mutation_scale=8),
        zorder=4, annotation_clip=False)

def ml(x,y,s):
    tx(x,y,s, fs=5.8, c="#AAAAAA", ha="left")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1   y: 5.73 – 6.68
# ─────────────────────────────────────────────────────────────────────────────
rb(0.15, 5.73, 9.70, 0.92, C["m1"], lw=1.2, rad=0.17)
ml(0.27, 6.56, "Module 1  ·  Market Regime Identification")

rb(0.30, 5.87, 1.45, 0.62, C["box"])
tx(1.025, 6.28, "Market Data", 7.0, bold=True)
tx(1.025, 6.05, "$\\bar{r}_t$, $\\sigma_t$", 6.5)

ar(1.75, 6.18, 2.10, 6.18)

rb(2.10, 5.87, 1.90, 0.62, "#DDEEFF", ec=C["side"], lw=1.2)
tx(3.05, 6.27, "Regime Identification", 7.0)
tx(3.05, 6.03, "(Gaussian HMM)", 6.3, C["gry"])

ar(4.00, 6.18, 4.28, 6.18)

for i,(lbl,col) in enumerate([("Bear","bear"),("Bull","bull"),("Sideways","side")]):
    rb(4.28+i*1.12, 5.89, 1.02, 0.58, C[col], ec=C[col], lw=0, rad=0.09)
    tx(4.79+i*1.12, 6.18, lbl, 7.0, C["wht"], bold=True)
tx(7.68, 6.18, "→ $r$", 7.0, C["gry"])

# r conditioning arrow down
ar(5.22, 5.73, 5.22, 5.52, c=C["side"], lw=1.2)
tx(6.70, 5.62, "regime label $r$ (per window)", 6.8, C["side"])

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2   y: 4.34 – 5.50
# ─────────────────────────────────────────────────────────────────────────────
rb(0.15, 4.34, 9.70, 1.07, C["m2"], lw=1.2, rad=0.17)
ml(0.27, 5.31, "Module 2  ·  Regime-Conditioned Diffusion Model")

rb(0.30, 4.50, 1.45, 0.62, C["box"])
tx(1.025, 4.91, "Real Windows", 7.0, bold=True)
tx(1.025, 4.68, "$\\mathbf{x}_i$ (training set)", 6.5)

ar(1.75, 4.81, 2.10, 4.81)

rb(2.10, 4.50, 3.70, 0.62, "#D4ECF9", ec=C["side"], lw=1.2)
tx(3.95, 4.88, "Learns $p(\\mathbf{x}\\,|\\,r)$  —  AdaLN Transformer", 7.0)
tx(3.95, 4.62, "RevIN  ·  DDPM  ·  DDIM fast sampling", 6.2, C["gry"])

ar(5.80, 4.81, 6.12, 4.81)

# Candidate Synthetic Pool  (x: 6.12 – 9.75)
rb(6.12, 4.50, 3.53, 0.62, C["pool"], ec=C["pb"], lw=1.2)
tx(7.885, 4.90, "Candidate Synthetic Pool", 7.0, C["pb"], bold=True)
for i,(col,lbl) in enumerate([(C["bear"],"Bear"),(C["bull"],"Bull"),(C["side"],"Side.")]):
    rb(6.16+i*0.75, 4.54, 0.70, 0.24, col, ec=col, lw=0, rad=0.06)
    tx(6.51+i*0.75, 4.66, lbl, 5.8, C["wht"], bold=True)

# pool → Module 3 arrow (straight down)
ar(7.885, 4.34, 7.885, 4.09, c=C["pb"], lw=1.2)
# label on LEFT side of arrow
tx(7.30, 3.97, "sample $w_r N_{\\!\\rm aug}$", 6.2, C["pb"], ha="right")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3   y: 2.87 – 4.07
# ─────────────────────────────────────────────────────────────────────────────
rb(0.15, 2.87, 9.70, 1.18, C["m3"], lw=1.2, rad=0.17)
ml(0.27, 3.94, "Module 3  ·  Distribution-Aware Augmentation Allocation")

rb(0.30, 3.03, 1.18, 0.66, C["box"])
tx(0.89, 3.37, "$p^{\\mathrm{tr}}_r$", 8.5)
tx(0.89, 3.12, "Train dist.", 5.8, C["gry"])

ar(1.48, 3.36, 1.75, 3.36)

rb(1.75, 3.03, 1.18, 0.66, C["box"])
tx(2.34, 3.37, "$p^{\\mathrm{val}}_r$", 8.5)
tx(2.34, 3.12, "Val dist.", 5.8, C["gry"])

ar(2.93, 3.36, 3.22, 3.36)

rb(3.22, 3.03, 3.60, 0.66, "#D8EED8", ec="#27AE60", lw=1.2)
tx(5.02, 3.42, r"$\delta_r{=}\max(p^{\rm val}_r{-}p^{\rm tr}_r,\,0)$  →  $w_r{=}\delta_r/\!\sum_{r'}\!\delta_{r'}$", 7.2)
tx(5.02, 3.14, "allocate budget to underrepresented regimes", 6.2, C["gry"])

ar(6.82, 3.36, 7.08, 3.36)

rb(7.08, 3.03, 2.67, 0.66, C["alloc"], ec=C["alloc"], lw=0, rad=0.11)
tx(8.415, 3.42, "Distribution-Aware", 7.0, C["wht"], bold=True)
tx(8.415, 3.14, "Synthetic Data", 6.8, C["wht"])

# ─────────────────────────────────────────────────────────────────────────────
# TWO-STAGE DOWNSTREAM TRAINING   y: 1.42 – 2.84
# ─────────────────────────────────────────────────────────────────────────────
rb(0.15, 1.42, 9.70, 1.38, C["m4"], lw=1.2, rad=0.17)
ml(0.27, 2.70, "Two-Stage Downstream Training")

# Real Data
rb(0.30, 1.59, 1.42, 0.78, C["box"])
tx(1.01, 2.03, "Real Data", 7.0, bold=True)
tx(1.01, 1.77, "$\\{(\\mathbf{x}_i,y_i)\\}$", 6.5)

rb(1.92, 1.59, 1.48, 0.78, C["alloc"], ec=C["alloc"], lw=0, rad=0.10)
tx(2.66, 2.03, "Dist.-Aware", 6.8, C["wht"])
tx(2.66, 1.77, "Synthetic", 6.8, C["wht"])

ar(1.72, 1.99, 1.90, 1.99)

# bent arrow: D-A output (Module 3, centre x=8.415) → Synthetic box top
ax.annotate("", xy=(2.66, 2.37), xytext=(8.415, 2.87),
    arrowprops=dict(arrowstyle="-|>", color=C["alloc"], lw=1.2,
                    connectionstyle="angle,angleA=0,angleB=90,rad=0.10",
                    mutation_scale=8), zorder=4)

# merge node
rb(3.60, 1.65, 0.50, 0.64, C["box"], rad=0.08)
tx(3.85, 2.03, "R+S", 7.0, C["gry"])
ar(3.40, 1.97, 3.58, 1.97)
ar(4.10, 1.97, 4.38, 1.97)

# Stage 1
rb(4.38, 1.59, 2.10, 0.78, C["s1b"], ec=C["s1e"], lw=1.2)
tx(5.43, 2.07, "Stage 1", 8.0, C["s1e"], bold=True)
tx(5.43, 1.77, "Self-supervised Pretrain\n(Real + Synthetic, no labels)", 6.5)

ar(6.48, 1.97, 6.76, 1.97)

rb(6.76, 1.59, 2.10, 0.78, C["s2b"], ec=C["s2e"], lw=1.2)
tx(7.81, 2.07, "Stage 2", 8.0, C["s2e"], bold=True)
tx(7.81, 1.77, "Supervised Fine-tune\n(Real labels only)", 6.5)

ar(8.86, 1.97, 9.13, 1.97)

# Return Prediction
rb(9.13, 1.61, 0.67, 0.72, C["out"], ec=C["out"], lw=0, rad=0.09)
tx(9.465, 2.04, "Return", 7, C["wht"])
tx(9.465, 1.78, "Pred.", 7, C["wht"])

# ─────────────────────────────────────────────────────────────────────────────
fig.savefig(f"{OUT}/framework.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUT}/framework.png", bbox_inches="tight", dpi=300)
print("Saved framework.pdf / .png")
