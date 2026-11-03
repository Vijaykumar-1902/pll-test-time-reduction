"""Streamlit demo for the PLL test-time project (SYNTHETIC data)."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).parent
T_FIXED = 60.0
LEAK_GATE = 2.0
JIT_FEATS = ["vco_band_code", "ring_osc_ghz", "chargepump_ua", "lock_time_us",
             "vdd_droop_norm", "wafer_radius_norm", "leak_screen_norm"]

st.set_page_config(page_title="PLL Test Time Reduction", layout="wide")

def find_csv(name):
    for d in [ROOT / "data", ROOT, ROOT / "Data set"]:
        if (d / name).exists():
            return d / name
    st.error(f"Could not find {name}. Put it in the data/ folder.")
    st.stop()

@st.cache_data
def load_lock():
    return pd.read_csv(find_csv("synthetic_pll_locktime_dataset.csv"))

@st.cache_resource(show_spinner="Training the jitter model (one time, about 30 seconds)...")
def load_jitter_model():
    jt = pd.read_csv(find_csv("synthetic_pll_jitter_dataset.csv"))
    jt = jt[(jt.lock_time_us > 0) & (jt.rms_jitter_ps > 0)]          # same cleaning as the main script
    tr, cal = jt[jt.split == "train"], jt[jt.split == "calibration"]
    m = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.06,
                                  subsample=0.8, random_state=0).fit(tr[JIT_FEATS], tr.rms_jitter_ps)
    cal_ok = cal[cal.leak_screen_norm < LEAK_GATE]
    resid = np.abs(cal_ok.rms_jitter_ps - m.predict(cal_ok[JIT_FEATS]))
    ranges = {c: (float(tr[c].min()), float(tr[c].max()), float(tr[c].median())) for c in JIT_FEATS}
    return m, resid.values, ranges, float(jt.usl_ps.iloc[0])

def decide(model, resid, ranges, usl, feats, alpha):
    """Return (decision, reason, prediction, margin). Any doubt -> RUN (full test)."""
    for k, (lo, hi, _) in ranges.items():
        if feats.get(k) is None or not lo <= feats[k] <= hi:
            return "RUN", f"input out of range or missing: {k}", None, None
    if feats["leak_screen_norm"] >= LEAK_GATE:
        return "RUN", "defect pre-screen flagged this die", None, None
    q = float(np.quantile(resid, 1 - alpha))
    p = float(model.predict(pd.DataFrame([feats])[JIT_FEATS])[0])
    if p + q < usl:
        return "SKIP", "predicted jitter plus margin is below the limit", p, q
    return "RUN", "uncertain: predicted jitter plus margin reaches the limit", p, q

st.title("AI-Optimized PLL Test Time Reduction")
st.caption("Synthetic data. Savings and escape rates illustrate the methods and are not product measurements.")
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Adaptive lock time", "Jitter test decision", "Project report"])

with tab1:
    st.markdown("""
**Problem.** Every die gets the same 60 us lock wait and the same long jitter measurement, though most dies lock
in about 14 us and sit far inside the jitter limit.

**Two techniques built**
1. **Adaptive lock-time control:** wait `min(t_lock + margin, 60 us)`. The 60 us timeout stays as a fail-safe.
2. **Jitter test elimination:** a gradient-boosting model predicts RMS jitter. A die skips the test only if
   `prediction + conformal margin < limit` **and** it passes a defect pre-screen. Any doubt runs the full test.

**Headline results (offline models, held-out test data)**
| Item | Result |
|---|---|
| Lock wait saving (3 us margin) | 70.8% |
| Jitter model R2 | 0.864 |
| Jitter test skip at 20 DPPM | 33.1% (bootstrap range about 25% to 47%) |
| Automated tests | 12 of 12 passed |
""")

with tab2:
    lt = load_lock()
    margin = st.slider("Polling margin (us)", 0.0, 20.0, 3.0, 0.5)
    wait = np.minimum(lt.lock_time_us + margin, T_FIXED)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean lock time", f"{lt.lock_time_us.mean():.2f} us")
    c2.metric("Mean adaptive wait", f"{wait.mean():.2f} us", f"fixed wait is {T_FIXED:.0f} us")
    c3.metric("Mean saving", f"{(1 - wait.mean() / T_FIXED) * 100:.1f}%")
    st.write(f"Dies reaching the 60 us timeout: **{int((lt.lock_time_us + margin >= T_FIXED).sum())}**")
    counts, edges = np.histogram(lt.lock_time_us, bins=40)
    st.bar_chart(pd.DataFrame({"dies": counts}, index=np.round(edges[:-1], 1)))
    st.caption("Distribution of lock time per die (us). The fixed wait of 60 us is far to the right of every die.")

with tab3:
    model, resid, ranges, usl = load_jitter_model()
    st.write(f"Enter one die's cheap measurements. The jitter limit is **{usl} ps**.")
    alpha = st.selectbox("Allowed miss rate (alpha)", [0.05, 0.01, 0.001], index=1)
    cols = st.columns(4)
    feats = {}
    for i, k in enumerate(JIT_FEATS):
        lo, hi, md = ranges[k]
        feats[k] = cols[i % 4].number_input(k, value=round(md, 3), format="%.3f")
    if st.button("Decide"):
        d, why, p, q = decide(model, resid, ranges, usl, feats, alpha)
        (st.success if d == "SKIP" else st.warning)(f"**{d}** the jitter test: {why}")
        if p is not None:
            st.write(f"Predicted jitter: **{p:.3f} ps**, conformal margin: **{q:.3f} ps**, limit: **{usl} ps**")
    st.caption("SKIP means the full jitter measurement is not needed for this die. RUN means measure it as usual.")

with tab4:
    for base in [ROOT / "results", ROOT]:
        ip = base / "pll_results.png"
        rp = base / "report.md"
        if ip.exists() or rp.exists():
            break
    if ip.exists():
        st.image(str(ip))
    if rp.exists():
        st.markdown(rp.read_text(encoding="utf-8"))
