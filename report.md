# AI-Optimized PLL Test Time Reduction - Results

All data is synthetic and illustrates the methods only.

## 1. Data understanding and cleaning

### Lock-time dataset

Rows: 20000, columns: 4. Missing cells: 0. Duplicate rows: 0. Duplicate die_id: 0.

| column | dtype | min | median | max | missing |
|---|---|---|---|---|---|
| die_id | int64 | 1 | 1e+04 | 2e+04 | 0 |
| lock_time_us | float64 | 4.019 | 13.99 | 38.55 | 0 |
| fixed_wait_us | float64 | 60 | 60 | 60 | 0 |
| adaptive_wait_us | float64 | 7.019 | 16.99 | 41.55 | 0 |

### Jitter dataset

Rows: 200000, columns: 14. Missing cells: 0. Duplicate rows: 0. Duplicate die_id: 0.

| column | dtype | min | median | max | missing |
|---|---|---|---|---|---|
| die_id | int64 | 1 | 1e+05 | 2e+05 | 0 |
| vco_band_code | float64 | -2.659 | 16 | 33.87 | 0 |
| ring_osc_ghz | float64 | 0.7769 | 0.9999 | 1.234 | 0 |
| chargepump_ua | float64 | 84.74 | 99.99 | 115.2 | 0 |
| lock_time_us | float64 | -0.642 | 11.99 | 24.58 | 0 |
| vdd_droop_norm | float64 | -4.313 | -0.0016 | 4.532 | 0 |
| wafer_radius_norm | float64 | 0 | 0.5008 | 1 | 0 |
| leak_screen_norm | float64 | -4.687 | -0.0009 | 8.566 | 0 |
| rms_jitter_ps | float64 | 0.5707 | 1.64 | 3.222 | 0 |
| predicted_jitter_ps | float64 | 0.8804 | 1.641 | 3.175 | 0 |
| usl_ps | float64 | 2.2 | 2.2 | 2.2 | 0 |
| jitter_fail | int64 | 0 | 0 | 1 | 0 |
| defect_injected | int64 | 0 | 0 | 1 | 0 |
| split | object | - | - | - | 0 |

### Cleaning actions (before / after)

| dataset | rows before | rows after | removed |
|---|---|---|---|
| lock-time | 20000 | 20000 | 0 |
| jitter | 200000 | 199999 | 1 |

Removed from the jitter dataset (physically impossible values): die_id [128948]. Duplicates and missing values: none found. Missing values would be dropped (no imputation) because a die with a missing feature must get the full test anyway.

Consistency: `adaptive_wait_us` = min(lock_time + 3 us, 60 us) holds for every row.
`usl_ps` constant: True. `jitter_fail` disagrees with rms_jitter_ps > USL on 1 row(s): a die with jitter rounded to exactly the limit (2.2 ps). Rounding artifact, not a data error.
Split sizes: {'train': 60000, 'calibration': 40000, 'test': 99999}; die_id overlap across splits: 0.
Mean per split (should be similar if the split is fair):

| split | mean lock_time_us | mean rms_jitter_ps |
|---|---|---|
| calibration | 12.023 | 1.642 |
| test | 11.994 | 1.641 |
| train | 11.996 | 1.642 |

Lock-time dies above Q3+1.5*IQR: 381. Kept: slow dies are real behaviour and the 60 us timeout covers them. Removing them would hide exactly the dies the test must handle.

## 2. Adaptive lock-time control (polling)

Rule: `T_wait = min(t_lock + T_m, T_fixed)`, T_m = 3 us, T_fixed = 60 us.

- Mean lock time: 14.55 us
- Mean adaptive wait: 17.55 us vs 60 us fixed
- **Mean saving: 70.8%**
- Dies reaching the timeout: 0
- Saving falls from 75.8% (0 us margin) to 42.4% (20 us margin).
- 2% audit sample: 400 dies; adaptive wait too short for 0 (zero by construction here; on silicon compare against a full-settling reading).

## 3. Model selection

5-fold cross-validation on the training split (R2, mean +/- std), then R2 and MAE on the held-out test split. Cross-validation uses a 10,000-row subsample of the training split to keep runtime short.

| task | model | CV R2 | CV std | test R2 | test MAE |
|---|---|---|---|---|---|
| lock time | Linear regression | 0.797 | 0.005 | 0.797 | 0.997 |
| lock time | Random forest | 0.790 | 0.005 | 0.795 | 1.002 |
| lock time | Gradient boosting | 0.792 | 0.005 | 0.796 | 1.000 |
| jitter | Linear regression | 0.844 | 0.015 | 0.845 | 0.057 |
| jitter | Random forest | 0.834 | 0.015 | 0.842 | 0.058 |
| jitter | Gradient boosting | 0.852 | 0.007 | 0.864 | 0.056 |

Gradient boosting is chosen when it is best or within noise of the best, because it handles non-linear effects (for example the squared radius term in jitter), is fast, supports feature importance, and is easy to export for a tester. The comparison above is the evidence; if linear regression matches it on a task, prefer the simpler model there.

## 4. ML-predicted wait (no lock-detect)

Features ['vco_band_code', 'ring_osc_ghz', 'chargepump_ua', 'vdd_droop_norm', 'wafer_radius_norm']; `lock_time_us` is the target, not a feature. Test R2 = 0.796.

| alpha | margin (us) | mean wait (us) | saving (%) | dies not yet locked (%) |
|---|---|---|---|---|
| 0.05 | 2.06 | 14.06 | 76.6 | 4.952 |
| 0.01 | 2.91 | 14.90 | 75.2 | 1.026 |
| 0.001 | 3.91 | 15.91 | 73.5 | 0.077 |

The ML wait is shorter than polling but can end before a die has locked. Use a small alpha and keep the audit sample.

## 5. Lock-time outlier flagging

Dies whose lock-time residual has a large robust z-score go to full test.

| z threshold | dies flagged (%) | injected defects caught (%) |
|---|---|---|
| 3 | 0.26 | 0.0 |
| 4 | 0.00 | 0.0 |
| 5 | 0.00 | 0.0 |

Defects in this synthetic data were injected into jitter, not lock time, so lock time carries almost no defect signal and nothing is caught. On real silicon, abnormal lock dynamics can indicate charge-pump or loop-filter defects.

## 6. Jitter test elimination

Model R2 = 0.864. Skip rule: `pred + q_alpha < USL (2.2 ps)` AND defect pre-screen (`leak_screen_norm < 2.0`). q_alpha is a split-conformal residual quantile.

| DPPM budget | skip WITH pre-screen (%) | skip WITHOUT pre-screen (%) |
|---|---|---|
| 20 | 33.1 | 18.6 |
| 50 | 47.2 | 25.2 |
| 100 | 79.0 | 48.3 |
| 200 | 89.1 | 80.9 |

Test set: 99999 dies, 440 true failures (0.44%). One escaped die = 10 DPPM, so small budgets are measured coarsely.

Stability check (100 bootstrap resamples of the test set), skip rate with pre-screen:

| DPPM budget | median (%) | 5th-95th percentile (%) |
|---|---|---|
| 20 | 33.2 | 24.6 to 47.3 |
| 100 | 78.9 | 65.2 to 86.4 |

The range is wide because only about 440 dies fail, so a budget of 20 DPPM allows 2 escapes. Quote the skip rate as a range, not a single number.

## 7. Automated tests

| test | result |
|---|---|
| Adaptive wait never exceeds the 60 us timeout | PASS |
| Adaptive wait always >= lock time (polling never ends before lock) when not capped | PASS |
| A die slower than the timeout is capped at 60 us | PASS |
| No die_id appears in more than one split | PASS |
| Conformal coverage at alpha=0.05 is within 1 point of 95% | PASS |
| Missing feature forces full test | PASS |
| NaN feature forces full test | PASS |
| Out-of-range feature forces full test | PASS |
| High leak_screen (defect pre-screen) forces full test | PASS |
| Kill switch (enabled=False) forces full test | PASS |
| Decision function matches the skip formula on 500 dies | PASS |
| Same input gives same decision (deterministic) | PASS |

**12 of 12 tests passed.**

## 8. Offline vs online model

The models are **offline**: trained in batch on historical data, then frozen and applied to new dies. They are not updated per die, because a test flow must be deterministic, versioned and auditable. Retraining is scheduled (for example per lot or week) after drift checks, then re-qualified.

## 9. Limitations

- Data is synthetic; savings and escape rates are not product measurements.
- The lock-time and jitter CSVs are separate populations; their savings are not additive.
- Per-configuration margins and waveform validation need full settling waveforms across VDD, temperature and divider settings, which these datasets do not contain.
- Real deployment needs shadow mode, a random audit sample, drift monitoring and quality sign-off.