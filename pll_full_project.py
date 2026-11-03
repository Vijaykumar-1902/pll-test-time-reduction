"""
AI-Optimized PLL Test Time Reduction - full project v2 (SYNTHETIC data)

 1. Data understanding and cleaning (types, ranges, splits, distributions)
 2. Adaptive lock-time control (polling, margin sweep, audit sample)
 3. Model selection: linear vs random forest vs gradient boosting, with cross-validation
 4. ML lock-time prediction with conformal margin (no lock-detect)
 5. Lock-time outlier flagging
 6. Jitter test elimination (conformal guard band + defect pre-screen + fail-safe decision function)
 7. Automated tests
 8. Report -> results/report.md, plots, tables

Run from the folder holding the two CSV files:  python pll_full_project_v2.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

HERE = Path(__file__).parent
DATA = HERE
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
T_FIXED = 60.0
SEED = 0
REPORT = []

def log(s=""):
    print(s)
    REPORT.append(s)

def r2_score_(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

# ======================================================================
# 1. DATA UNDERSTANDING AND CLEANING
# ======================================================================
lt_raw = pd.read_csv(DATA / "synthetic_pll_locktime_dataset.csv")
jt_raw = pd.read_csv(DATA / "synthetic_pll_jitter_dataset.csv")

log("# AI-Optimized PLL Test Time Reduction - Results\n")
log("All data is synthetic and illustrates the methods only.\n")
log("## 1. Data understanding and cleaning\n")

def quality_table(name, df, key="die_id"):
    log(f"### {name}\n")
    log(f"Rows: {len(df)}, columns: {df.shape[1]}. Missing cells: {int(df.isna().sum().sum())}. "
        f"Duplicate rows: {int(df.duplicated().sum())}. Duplicate {key}: {int(df[key].duplicated().sum())}.\n")
    log("| column | dtype | min | median | max | missing |")
    log("|---|---|---|---|---|---|")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            log(f"| {c} | {df[c].dtype} | {df[c].min():.4g} | {df[c].median():.4g} | {df[c].max():.4g} | {int(df[c].isna().sum())} |")
        else:
            log(f"| {c} | {df[c].dtype} | - | - | - | {int(df[c].isna().sum())} |")
    log("")

quality_table("Lock-time dataset", lt_raw)
quality_table("Jitter dataset", jt_raw)

# cleaning actions (counted so a before/after summary can be shown)
n_lt0, n_jt0 = len(lt_raw), len(jt_raw)
lt = lt_raw.drop_duplicates().dropna().copy()
jt = jt_raw.drop_duplicates().dropna().copy()
# physically impossible values are removed (none expected): non-positive times, jitter or negative radius
bad_lt = (lt.lock_time_us <= 0)
bad_jt = (jt.lock_time_us <= 0) | (jt.rms_jitter_ps <= 0) | (~jt.wafer_radius_norm.between(0, 1))
lt, jt = lt[~bad_lt].copy(), jt[~bad_jt].copy()
log("### Cleaning actions (before / after)\n")
log("| dataset | rows before | rows after | removed |")
log("|---|---|---|---|")
log(f"| lock-time | {n_lt0} | {len(lt)} | {n_lt0 - len(lt)} |")
log(f"| jitter | {n_jt0} | {len(jt)} | {n_jt0 - len(jt)} |")
removed_ids = jt_raw.loc[~jt_raw.die_id.isin(jt.die_id), "die_id"].tolist()
log(f"\nRemoved from the jitter dataset (physically impossible values): die_id {removed_ids}. "
    "Duplicates and missing values: none found. Missing values would be dropped (no imputation) "
    "because a die with a missing feature must get the full test anyway.\n")

# consistency checks
assert (lt.fixed_wait_us == T_FIXED).all()
assert np.allclose(lt.adaptive_wait_us, np.minimum(lt.lock_time_us + 3, T_FIXED), atol=1e-3)
log("Consistency: `adaptive_wait_us` = min(lock_time + 3 us, 60 us) holds for every row.")
mism = int(((jt.rms_jitter_ps > jt.usl_ps).astype(int) != jt.jitter_fail).sum())
log(f"`usl_ps` constant: {jt.usl_ps.nunique() == 1}. `jitter_fail` disagrees with rms_jitter_ps > USL "
    f"on {mism} row(s): a die with jitter rounded to exactly the limit (2.2 ps). "
    "Rounding artifact, not a data error.")

# split integrity
parts = {s: set(jt[jt.split == s].die_id) for s in ["train", "calibration", "test"]}
overlap = sum(len(parts[a] & parts[b]) for a, b in [("train", "calibration"), ("train", "test"), ("calibration", "test")])
log(f"Split sizes: { {k: len(v) for k, v in parts.items()} }; die_id overlap across splits: {overlap}.")
sm = jt.groupby("split")[["lock_time_us", "rms_jitter_ps"]].mean().round(3)
log("Mean per split (should be similar if the split is fair):\n")
log("| split | mean lock_time_us | mean rms_jitter_ps |")
log("|---|---|---|")
for k, r in sm.iterrows():
    log(f"| {k} | {r.lock_time_us} | {r.rms_jitter_ps} |")
log("")

q1, q3 = lt.lock_time_us.quantile([.25, .75]); iqr = q3 - q1
n_iqr = int((lt.lock_time_us > q3 + 1.5 * iqr).sum())
log(f"Lock-time dies above Q3+1.5*IQR: {n_iqr}. Kept: slow dies are real behaviour and the 60 us "
    "timeout covers them. Removing them would hide exactly the dies the test must handle.\n")

# ======================================================================
# 2. ADAPTIVE LOCK-TIME CONTROL (polling)
# ======================================================================
def adaptive_wait(lock_us, margin_us, t_fixed=T_FIXED):
    return np.minimum(lock_us + margin_us, t_fixed)

lt["wait_3us"] = adaptive_wait(lt.lock_time_us, 3.0)
sav3 = (1 - lt.wait_3us.mean() / T_FIXED) * 100
log("## 2. Adaptive lock-time control (polling)\n")
log("Rule: `T_wait = min(t_lock + T_m, T_fixed)`, T_m = 3 us, T_fixed = 60 us.\n")
log(f"- Mean lock time: {lt.lock_time_us.mean():.2f} us")
log(f"- Mean adaptive wait: {lt.wait_3us.mean():.2f} us vs {T_FIXED:.0f} us fixed")
log(f"- **Mean saving: {sav3:.1f}%**")
log(f"- Dies reaching the timeout: {int((lt.lock_time_us + 3 >= T_FIXED).sum())}")
margins = np.arange(0, 21, 1.0)
saving_curve = [(1 - adaptive_wait(lt.lock_time_us, m).mean() / T_FIXED) * 100 for m in margins]
log(f"- Saving falls from {saving_curve[0]:.1f}% (0 us margin) to {saving_curve[-1]:.1f}% (20 us margin).")
audit = lt.sample(frac=0.02, random_state=1)
log(f"- 2% audit sample: {len(audit)} dies; adaptive wait too short for "
    f"{int((audit.lock_time_us > audit.wait_3us).sum())} "
    "(zero by construction here; on silicon compare against a full-settling reading).\n")

# ======================================================================
# 3. MODEL SELECTION (cross-validation on the training split)
# ======================================================================
tr, cal, te = (jt[jt.split == s] for s in ("train", "calibration", "test"))
lock_feats = ["vco_band_code", "ring_osc_ghz", "chargepump_ua", "vdd_droop_norm", "wafer_radius_norm"]
jit_feats = ["vco_band_code", "ring_osc_ghz", "chargepump_ua", "lock_time_us",
             "vdd_droop_norm", "wafer_radius_norm", "leak_screen_norm"]

def make_models():
    return {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_leaf=20,
                                               n_jobs=-1, random_state=SEED),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                                                       subsample=0.8, random_state=SEED),
    }

log("## 3. Model selection\n")
log("5-fold cross-validation on the training split (R2, mean +/- std), then R2 and MAE on the held-out test split. "
    "Cross-validation uses a 10,000-row subsample of the training split to keep runtime short.\n")
sub = tr.sample(10000, random_state=SEED)
kf = KFold(5, shuffle=True, random_state=SEED)
sel_rows = []
for task, feats, target in [("lock time", lock_feats, "lock_time_us"),
                            ("jitter", jit_feats, "rms_jitter_ps")]:
    for name, m in make_models().items():
        cv = cross_val_score(m, sub[feats], sub[target], cv=kf, scoring="r2")
        m.fit(tr[feats], tr[target])
        p = m.predict(te[feats])
        sel_rows.append((task, name, cv.mean(), cv.std(),
                         r2_score_(te[target].values, p), np.abs(te[target].values - p).mean()))
sel = pd.DataFrame(sel_rows, columns=["task", "model", "cv_r2_mean", "cv_r2_std", "test_r2", "test_mae"])
sel.to_csv(OUT / "model_comparison.csv", index=False)
log("| task | model | CV R2 | CV std | test R2 | test MAE |")
log("|---|---|---|---|---|---|")
for _, r in sel.iterrows():
    log(f"| {r.task} | {r.model} | {r.cv_r2_mean:.3f} | {r.cv_r2_std:.3f} | {r.test_r2:.3f} | {r.test_mae:.3f} |")
log("\nGradient boosting is chosen when it is best or within noise of the best, because it handles "
    "non-linear effects (for example the squared radius term in jitter), is fast, supports "
    "feature importance, and is easy to export for a tester. The comparison above is the evidence; "
    "if linear regression matches it on a task, prefer the simpler model there.\n")

# ======================================================================
# 4. ML LOCK-TIME PREDICTION (no lock-detect)
# ======================================================================
lock_model = make_models()["Gradient boosting"].fit(tr[lock_feats], tr.lock_time_us)
pred_lock = lock_model.predict(te[lock_feats])
r2_lock = r2_score_(te.lock_time_us.values, pred_lock)
res_cal = cal.lock_time_us - lock_model.predict(cal[lock_feats])
log("## 4. ML-predicted wait (no lock-detect)\n")
log(f"Features {lock_feats}; `lock_time_us` is the target, not a feature. Test R2 = {r2_lock:.3f}.\n")
log("| alpha | margin (us) | mean wait (us) | saving (%) | dies not yet locked (%) |")
log("|---|---|---|---|---|")
rows = []
for alpha in [0.05, 0.01, 0.001]:
    q = np.quantile(res_cal, 1 - alpha)
    wait = np.minimum(pred_lock + q, T_FIXED)
    early = (te.lock_time_us.values > wait).mean() * 100
    sv = (1 - wait.mean() / T_FIXED) * 100
    rows.append((alpha, q, wait.mean(), sv, early))
    log(f"| {alpha} | {q:.2f} | {wait.mean():.2f} | {sv:.1f} | {early:.3f} |")
ml_tbl = pd.DataFrame(rows, columns=["alpha", "margin_us", "mean_wait_us", "saving_pct", "not_locked_pct"])
ml_tbl.to_csv(OUT / "ml_margin_summary.csv", index=False)
log("\nThe ML wait is shorter than polling but can end before a die has locked. "
    "Use a small alpha and keep the audit sample.\n")

# ======================================================================
# 5. LOCK-TIME OUTLIER FLAGGING
# ======================================================================
med = np.median(res_cal)
mad = np.median(np.abs(res_cal - med)) * 1.4826
z = (te.lock_time_us.values - pred_lock - med) / mad
log("## 5. Lock-time outlier flagging\n")
log("Dies whose lock-time residual has a large robust z-score go to full test.\n")
log("| z threshold | dies flagged (%) | injected defects caught (%) |")
log("|---|---|---|")
defect_te = te.defect_injected.values == 1
for k in [3, 4, 5]:
    flag = np.abs(z) > k
    caught = flag[defect_te].mean() * 100 if defect_te.any() else float("nan")
    log(f"| {k} | {flag.mean()*100:.2f} | {caught:.1f} |")
log("\nDefects in this synthetic data were injected into jitter, not lock time, so lock time "
    "carries almost no defect signal and nothing is caught. On real silicon, abnormal lock "
    "dynamics can indicate charge-pump or loop-filter defects.\n")

# ======================================================================
# 6. JITTER TEST ELIMINATION
# ======================================================================
USL = float(jt.usl_ps.iloc[0])
LEAK_GATE = 2.0
jm = make_models()["Gradient boosting"]
jm.set_params(n_estimators=250, learning_rate=0.06)
jm.fit(tr[jit_feats], tr.rms_jitter_ps)
pj = jm.predict(te[jit_feats])
r2_j = r2_score_(te.rms_jitter_ps.values, pj)
cal_ok = cal[cal.leak_screen_norm < LEAK_GATE]
resj = np.abs(cal_ok.rms_jitter_ps - jm.predict(cal_ok[jit_feats]))
fail_te = (te.rms_jitter_ps > USL).values
screen_ok = (te.leak_screen_norm < LEAK_GATE).values

def sweep(use_prescreen):
    out = []
    for alpha in np.logspace(np.log10(0.5), -5, 80):
        q = np.quantile(resj, 1 - alpha)
        skip = (pj + q < USL)
        if use_prescreen:
            skip &= screen_ok
        out.append((alpha, skip.mean() * 100, (skip & fail_te).sum() / len(te) * 1e6))
    return pd.DataFrame(out, columns=["alpha", "skip_pct", "dppm"])

sw_on, sw_off = sweep(True), sweep(False)
def best_skip(sw, budget):
    ok = sw[sw.dppm <= budget]
    return ok.skip_pct.max() if len(ok) else 0.0

log("## 6. Jitter test elimination\n")
log(f"Model R2 = {r2_j:.3f}. Skip rule: `pred + q_alpha < USL ({USL} ps)` AND defect pre-screen "
    f"(`leak_screen_norm < {LEAK_GATE}`). q_alpha is a split-conformal residual quantile.\n")
log("| DPPM budget | skip WITH pre-screen (%) | skip WITHOUT pre-screen (%) |")
log("|---|---|---|")
for b in [20, 50, 100, 200]:
    log(f"| {b} | {best_skip(sw_on, b):.1f} | {best_skip(sw_off, b):.1f} |")
log(f"\nTest set: {len(te)} dies, {int(fail_te.sum())} true failures ({fail_te.mean()*100:.2f}%). "
    f"One escaped die = {1e6/len(te):.0f} DPPM, so small budgets are measured coarsely.\n")
sw_on.to_csv(OUT / "jitter_skip_tradeoff.csv", index=False)

# stability: bootstrap the test set. 20 DPPM is only ~2 dies, so the headline can move a lot.
alphas = np.logspace(np.log10(0.5), -5, 80)
skip_mat = np.array([(pj + np.quantile(resj, 1 - a) < USL) & screen_ok for a in alphas])  # alphas x dies
rng_b = np.random.default_rng(5)
boot = {20: [], 100: []}
for _ in range(100):
    ix = rng_b.integers(0, len(te), len(te))
    sk, fl = skip_mat[:, ix], fail_te[ix]
    dppm = (sk & fl).sum(axis=1) / len(te) * 1e6
    spct = sk.mean(axis=1) * 100
    for b in boot:
        ok = dppm <= b
        boot[b].append(spct[ok].max() if ok.any() else 0.0)
log("Stability check (100 bootstrap resamples of the test set), skip rate with pre-screen:\n")
log("| DPPM budget | median (%) | 5th-95th percentile (%) |")
log("|---|---|---|")
for b in boot:
    lo, md, hi = np.percentile(boot[b], [5, 50, 95])
    log(f"| {b} | {md:.1f} | {lo:.1f} to {hi:.1f} |")
log("\nThe range is wide because only about 440 dies fail, so a budget of 20 DPPM allows 2 escapes. "
    "Quote the skip rate as a range, not a single number.\n")

# fail-safe production decision function (what a tester hook would call)
Q_CHOSEN = float(np.quantile(resj, 1 - 0.01))
feat_ranges = {c: (tr[c].min(), tr[c].max()) for c in jit_feats}

def decide_jitter(feats: dict, enabled=True):
    """Return ('SKIP'|'RUN', reason). Any doubt -> RUN (full test)."""
    if not enabled:
        return "RUN", "disabled"
    if any(feats.get(k) is None or (isinstance(feats.get(k), float) and np.isnan(feats[k])) for k in jit_feats):
        return "RUN", "missing_feature"
    for k, (lo, hi) in feat_ranges.items():
        if not lo <= feats[k] <= hi:
            return "RUN", f"out_of_range:{k}"
    if feats["leak_screen_norm"] >= LEAK_GATE:
        return "RUN", "defect_prescreen"
    p = float(jm.predict(pd.DataFrame([feats])[jit_feats])[0])
    return ("SKIP", "confident_pass") if p + Q_CHOSEN < USL else ("RUN", "uncertain")

# ======================================================================
# 7. AUTOMATED TESTS
# ======================================================================
log("## 7. Automated tests\n")
tests = []
def check(name, cond):
    tests.append((name, bool(cond)))

check("Adaptive wait never exceeds the 60 us timeout", (lt.wait_3us <= T_FIXED + 1e-9).all())
check("Adaptive wait always >= lock time (polling never ends before lock) when not capped",
      (lt.wait_3us >= np.minimum(lt.lock_time_us, T_FIXED) - 1e-9).all())
check("A die slower than the timeout is capped at 60 us", adaptive_wait(np.array([100.0]), 3.0)[0] == T_FIXED)
check("No die_id appears in more than one split", overlap == 0)
check("Conformal coverage at alpha=0.05 is within 1 point of 95%",
      abs(((np.abs(te.rms_jitter_ps.values - pj) <= np.quantile(
          np.abs(cal.rms_jitter_ps - jm.predict(cal[jit_feats])), 0.95)).mean()) - 0.95) < 0.01)
sample = te.iloc[0][jit_feats].to_dict()
check("Missing feature forces full test", decide_jitter({**sample, "ring_osc_ghz": None})[0] == "RUN")
check("NaN feature forces full test", decide_jitter({**sample, "vco_band_code": float("nan")})[0] == "RUN")
check("Out-of-range feature forces full test", decide_jitter({**sample, "chargepump_ua": 1e6})[0] == "RUN")
check("High leak_screen (defect pre-screen) forces full test",
      decide_jitter({**sample, "leak_screen_norm": LEAK_GATE + 1})[0] == "RUN")
check("Kill switch (enabled=False) forces full test", decide_jitter(sample, enabled=False)[0] == "RUN")
# decision function agrees with the vectorised rule on 500 test dies
chk = te.sample(500, random_state=3)
agree = np.mean([
    (decide_jitter(r[jit_feats].to_dict())[0] == "SKIP") ==
    bool((jm.predict(pd.DataFrame([r[jit_feats]])[jit_feats])[0] + Q_CHOSEN < USL) and r.leak_screen_norm < LEAK_GATE)
    for _, r in chk.iterrows()])
check("Decision function matches the skip formula on 500 dies", agree == 1.0)
check("Same input gives same decision (deterministic)",
      decide_jitter(sample) == decide_jitter(sample))

log("| test | result |")
log("|---|---|")
for n, ok in tests:
    log(f"| {n} | {'PASS' if ok else 'FAIL'} |")
log(f"\n**{sum(ok for _, ok in tests)} of {len(tests)} tests passed.**\n")

# ======================================================================
# PLOTS
# ======================================================================
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
a = ax[0, 0]; a.hist(lt.lock_time_us, bins=80, color="#2563eb", alpha=.85)
a.axvline(T_FIXED, color="red"); a.axvline(lt.wait_3us.median(), color="green", ls="--")
a.set(title="Lock-time distribution (red=fixed 60 us)", xlabel="us", ylabel="dies")
a = ax[0, 1]; a.plot(margins, saving_curve, marker="o")
a.set(title="Saving vs polling margin", xlabel="margin (us)", ylabel="saving (%)")
a = ax[0, 2]; a.scatter(te.lock_time_us[:3000], pred_lock[:3000], s=3, alpha=.4)
lim = [te.lock_time_us.min(), te.lock_time_us.max()]; a.plot(lim, lim, "k--", lw=.8)
a.set(title=f"Predicted vs actual lock time (R2={r2_lock:.2f})", xlabel="actual (us)", ylabel="predicted (us)")
a = ax[1, 0]
w = 0.25; xs = np.arange(2)
for i, name in enumerate(["Linear regression", "Random forest", "Gradient boosting"]):
    a.bar(xs + (i - 1) * w, [sel[(sel.task == t) & (sel.model == name)].test_r2.iloc[0] for t in ["lock time", "jitter"]],
          w, label=name)
a.set_xticks(xs); a.set_xticklabels(["lock time", "jitter"]); a.set(title="Model comparison (test R2)", ylim=(0, 1)); a.legend(fontsize=7)
a = ax[1, 1]; a.scatter(pj[:5000], te.rms_jitter_ps.values[:5000], s=3, alpha=.35)
a.axhline(USL, color="red"); a.plot([1, 3.2], [1, 3.2], "k--", lw=.8)
a.set(title=f"Predicted vs measured jitter (R2={r2_j:.2f})", xlabel="predicted (ps)", ylabel="measured (ps)")
a = ax[1, 2]
a.plot(sw_on.skip_pct, sw_on.dppm, label="with pre-screen")
a.plot(sw_off.skip_pct, sw_off.dppm, label="without pre-screen")
a.axhline(20, color="red", ls="--", lw=.8); a.set_yscale("symlog", linthresh=10)
a.set(title="Skip rate vs escapes", xlabel="parts skipping jitter test (%)", ylabel="DPPM"); a.legend()
fig.tight_layout(); fig.savefig(OUT / "pll_results.png", dpi=140)

# ======================================================================
# 8. LIMITATIONS AND REPORT
# ======================================================================
log("## 8. Offline vs online model\n")
log("The models are **offline**: trained in batch on historical data, then frozen and applied to new dies. "
    "They are not updated per die, because a test flow must be deterministic, versioned and auditable. "
    "Retraining is scheduled (for example per lot or week) after drift checks, then re-qualified.\n")
log("## 9. Limitations\n")
log("- Data is synthetic; savings and escape rates are not product measurements.")
log("- The lock-time and jitter CSVs are separate populations; their savings are not additive.")
log("- Per-configuration margins and waveform validation need full settling waveforms across VDD, temperature and divider settings, which these datasets do not contain.")
log("- Real deployment needs shadow mode, a random audit sample, drift monitoring and quality sign-off.")
(OUT / "report.md").write_text("\n".join(REPORT), encoding="utf-8")
print(f"\nSaved report and plots to {OUT}")