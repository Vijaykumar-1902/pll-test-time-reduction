# AI-Optimized PLL Test Time Reduction

Reduces PLL production test time using adaptive lock-time control and ML-based jitter test elimination.
All data is synthetic.

**Live demo:** paste your Streamlit link here

## Files
| File | Purpose |
|---|---|
| `pll_full_project.py` | Full pipeline: cleaning, model comparison, lock-time and jitter methods, 12 tests, report |
| `app.py` | Streamlit demo (lock-time calculator, jitter skip/run decision, report) |
| `synthetic_pll_*.csv` | The two provided datasets (unchanged) |
| `generate_synthetic_pll_data.py` | Provided generator script (unchanged) |
| `results/` | `report.md`, `pll_results.png` and tables from the last run |

## Run locally
```
pip install -r requirements.txt
python pll_full_project.py     # about 2 to 4 minutes, writes results/
streamlit run app.py           # opens the demo in the browser
```

## Method summary
- Adaptive lock wait: `min(t_lock + 3 us, 60 us)`, with the 60 us timeout kept as a fail-safe.
- Jitter test elimination: gradient boosting, a split-conformal margin and a defect pre-screen.
  Any missing, out-of-range or doubtful input runs the full test.
- Models are trained offline and frozen.

## Key results (synthetic data)
- Lock wait saving: 70.8%
- Jitter model R2: 0.864
- Jitter test skip at 20 DPPM: 33.1% (bootstrap range about 25% to 47%)
- Automated tests: 12 of 12 passed

## Limitations
Synthetic data. There are no lot IDs, so validation uses the provided die split. Real deployment needs
shadow mode, audit sampling, drift monitoring and quality sign-off.