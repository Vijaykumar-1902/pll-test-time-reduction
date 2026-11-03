"""Figures for the PLL test-time optimization document.
All data here is SYNTHETIC and only illustrates the methods."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

OUT = "./"   # figures and stats.txt are written here
INK = "#1f2937"; MUTED = "#6b7280"; GRID = "#e5e7eb"
C1 = "#2563eb"; C2 = "#f59e0b"; C3 = "#10b981"; C4 = "#ef4444"; C5 = "#8b5cf6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})
rng = np.random.default_rng(7)

# ---------- Fig 1: illustrative test-time breakdown ----------
items = ["Jitter / phase noise", "Lock-time waits", "Trim / calibration",
         "Config. sweeps", "Spurs / loop BW", "DC & functional"]
base = np.array([38, 22, 16, 12, 7, 5], float)
opt = np.array([14, 6, 5, 6, 5, 5], float)
fig, ax = plt.subplots(figsize=(6.4, 2.9))
y = np.arange(len(items))[::-1]
ax.barh(y + 0.19, base, 0.36, color="#cbd5e1", label="Baseline flow")
ax.barh(y - 0.19, opt, 0.36, color=C1, label="AI-optimized flow")
for yi, b, o in zip(y, base, opt):
    ax.text(b + 0.6, yi + 0.19, f"{b:.0f}", va="center", fontsize=8, color=MUTED)
    ax.text(o + 0.6, yi - 0.19, f"{o:.0f}", va="center", fontsize=8, color=C1)
ax.set_yticks(y); ax.set_yticklabels(items)
ax.set_xlabel("Share of baseline PLL test time (%)")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
ax.set_title(f"Illustrative PLL test-time budget: 100% → {opt.sum():.0f}%",
             loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(OUT + "f1_breakdown.png", dpi=220); plt.close(fig)

# ---------- Synthetic data for test-elimination study ----------
N = 200000
proc = rng.normal(0, 1, N)          # hidden process shift
x_vco_code = 16 + 4 * proc + rng.normal(0, 0.8, N)
x_ro = 1.0 + 0.05 * proc + rng.normal(0, 0.01, N)
x_icp = 100 - 3 * proc + rng.normal(0, 2, N)
x_lockt = 12 + 2.5 * proc + rng.normal(0, 1.2, N)
x_vdd_droop = rng.normal(0, 1, N)
x_wr = rng.uniform(0, 1, N)          # radial wafer position
jitter = (1.6 + 0.18 * proc + 0.12 * x_wr ** 2 + 0.05 * x_vdd_droop
          + 0.09 * rng.standard_t(5, N) * 0.6)
# inject a small population of genuinely bad parts (defect-driven)
bad = rng.random(N) < 0.0015
jitter[bad] += rng.uniform(0.4, 1.2, bad.sum())
# cheap screen that partly sees defects (e.g. loop-filter leakage / lock-detect glitches)
x_leak = rng.normal(0, 1, N) + np.where(bad, rng.uniform(3.5, 7, N), 0)
X = np.c_[x_vco_code, x_ro, x_icp, x_lockt, x_vdd_droop, x_wr, x_leak]
USL = 2.2
Xtr, Xte, ytr, yte = train_test_split(X, jitter, test_size=0.5, random_state=1)
Xtr, Xcal, ytr, ycal = train_test_split(Xtr, ytr, test_size=0.4, random_state=2)
m = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.06,
                              subsample=0.8, random_state=0).fit(Xtr, ytr)
LEAK_GATE = 2.0   # outlier pre-screen: flagged parts always get the full test
res_cal = np.abs(ycal - m.predict(Xcal))[Xcal[:, 6] < LEAK_GATE]
pred = m.predict(Xte)
fail_true = yte > USL
ks, skip, esc = [], [], []
for alpha in np.logspace(np.log10(0.5), -5, 80):
    q = np.quantile(res_cal, 1 - alpha)
    s = (pred + q < USL) & (Xte[:, 6] < LEAK_GATE)
    ks.append(alpha); skip.append(s.mean() * 100)
    esc.append((s & fail_true).sum() / len(yte) * 1e6)   # DPPM
skip = np.array(skip); esc = np.array(esc)
r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
ax = axs[0]
idx = rng.choice(len(yte), 3000, replace=False)
ax.scatter(pred[idx], yte[idx], s=3, alpha=0.35, color=C1, lw=0)
lo, hi = 1.0, 3.2
ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
ax.axhline(USL, color=C4, lw=0.9); ax.axvline(USL, color=C4, lw=0.9, ls=":")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Predicted RMS jitter (ps)"); ax.set_ylabel("Measured RMS jitter (ps)")
ax.set_title(f"Prediction vs measurement (R² = {r2:.2f})", loc="left", fontsize=9)
ax.text(lo + 0.05, USL + 0.05, "USL", color=C4, fontsize=8)
ax = axs[1]
ax.plot(skip, esc, color=C1, lw=1.6)
target = 20
ok = esc <= target
best = skip[ok].max() if ok.any() else 0
ax.axhline(target, color=C4, lw=0.9, ls="--")
ax.text(0.97, 0.38, f"DPPM budget = {target}", color=C4, fontsize=8, ha="right", transform=ax.transAxes)
ax.scatter([best], [esc[ok][np.argmax(skip[ok])]], color=C4, zorder=5, s=18)
ax.annotate(f"{best:.0f}% of parts skip\nthe jitter test", (best, esc[ok][np.argmax(skip[ok])]),
            xytext=(0.05, 0.72), textcoords="axes fraction", fontsize=8, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
ax.set_yscale("symlog", linthresh=10); ax.set_xlim(0, 100); ax.set_ylim(0, 5000)
ax.set_xlabel("Parts skipping jitter test (%)"); ax.set_ylabel("Test escapes (DPPM)")
ax.set_title("Skip rate vs escape trade-off", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "f3_tradeoff.png", dpi=220); plt.close(fig)
with open(OUT + "stats.txt", "w") as f:
    f.write(f"r2={r2:.3f}\nbest_skip={best:.1f}\nfail_rate={fail_true.mean()*100:.3f}\n")

# ---------- Fig 4: adaptive lock-time ----------
lock = np.clip(rng.lognormal(np.log(14), 0.28, 20000), 4, None)
fixed = 60.0
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.hist(lock, bins=90, color=C1, alpha=0.85)
ax.axvline(fixed, color=C4, lw=1.2)
ax.text(fixed - 1, ax.get_ylim()[1] * 0.85, f"Fixed worst-case wait\n{fixed:.0f} µs",
        ha="right", color=C4, fontsize=8)
p = np.percentile(lock, 50)
ax.axvline(p + 3, color=C3, lw=1.2, ls="--")
ax.text(p + 4, ax.get_ylim()[1] * 0.55,
        f"Adaptive: poll lock-detect\n+ 3 µs margin (median {p + 3:.0f} µs)",
        color="#047857", fontsize=8)
ax.set_xlabel("Measured lock time per die (µs), synthetic population")
ax.set_ylabel("Dies"); ax.set_xlim(0, 70)
fig.tight_layout(); fig.savefig(OUT + "f4_lock.png", dpi=220); plt.close(fig)
with open(OUT + "stats.txt", "a") as f:
    f.write(f"lock_mean={lock.mean():.1f}\nlock_saving={(1-(lock.mean()+3)/fixed)*100:.0f}\n")

# ---------- Fig 5: sequential jitter test ----------
fig, ax = plt.subplots(figsize=(6.4, 2.6))
n = np.arange(200, 20001, 200)
for true, col, lab in [(1.55, C3, "Good die (σ = 1.55 ps)"),
                       (2.12, C2, "Marginal die (σ = 2.12 ps)"),
                       (2.55, C4, "Failing die (σ = 2.55 ps)")]:
    samples = rng.normal(0, true, 20000)
    est = np.array([samples[:k].std(ddof=1) for k in n])
    ci = 1.96 * est / np.sqrt(2 * (n - 1))
    ax.plot(n, est, color=col, lw=1.4, label=lab)
    ax.fill_between(n, est - ci, est + ci, color=col, alpha=0.18, lw=0)
ax.axhline(2.2, color=INK, lw=0.9, ls="--")
ax.text(19800, 2.24, "USL 2.2 ps", ha="right", fontsize=8)
ax.set_xlabel("Period samples captured"); ax.set_ylabel("RMS jitter estimate (ps)")
ax.set_ylim(1.2, 3.0)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.13))
fig.tight_layout(); fig.savefig(OUT + "f5_seq.png", dpi=220); plt.close(fig)

# ---------- Fig 2: architecture ----------
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
def box(x, y, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=col + "22", ec=col, lw=1.1))
    ax.text(x + w / 2, y + h - 3, title, ha="center", va="top", fontsize=8.5,
            weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 7.6, sub, ha="center", va="top", fontsize=7, color=MUTED,
            linespacing=1.3)
def arrow(x1, y1, x2, y2, txt=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, txt, ha="center", fontsize=6.5, color=MUTED)
box(1, 30, 20, 18, "Data sources", "STDF datalogs\nCharacterization\nWafer maps / e-test\nGenealogy", C5)
box(27, 30, 20, 18, "Feature store", "Cleaning, joins\nSpatial features\nTester / site IDs\nVersioned sets", C1)
box(53, 30, 20, 18, "Model training", "GBM / GP / Bayes opt.\nConformal calibration\nEscape simulation\nModel registry", C3)
box(79, 30, 20, 18, "Decision engine", "On / near ATE\n< 5 ms inference\nSkip / run / extend\nFail-safe default", C2)
box(27, 2, 20, 18, "Governance", "Quality sign-off\nChange control\nDPPM budget\nRollback authority", MUTED)
box(53, 2, 20, 18, "Audit & monitoring", "Random full-test\nsample (1-5%)\nDrift (PSI, KS)\nEscape tracking", C4)
box(79, 2, 20, 18, "ATE test program", "Adaptive lock wait\nTrim seed / window\nSequential jitter\nTest ordering", C2)
arrow(21.8, 39, 26.2, 39); arrow(47.8, 39, 52.2, 39); arrow(73.8, 39, 78.2, 39)
ax.text(76, 42.5, "deploy", ha="center", fontsize=6.5, color=MUTED)
arrow(89, 29.2, 89, 20.8); ax.text(90.5, 25, "per-die\ndecisions", fontsize=6.5, color=MUTED, va="center")
arrow(78.2, 11, 73.8, 11); ax.text(76, 13.5, "logs", ha="center", fontsize=6.5, color=MUTED)
arrow(52.2, 11, 47.8, 11); ax.text(50, 13.5, "alerts", ha="center", fontsize=6.5, color=MUTED)
arrow(63, 20.8, 63, 29.2); ax.text(64.5, 25, "retrain\ndata", fontsize=6.5, color=MUTED, va="center")
fig.tight_layout(); fig.savefig(OUT + "f2_arch.png", dpi=220); plt.close(fig)
print(open(OUT + "stats.txt").read())
