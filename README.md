<div align="center">

# 🌊 PatchTST + Conformal Discharge Forecasting
### Hourly River Discharge Prediction with Calibrated Uncertainty Bands — Reno River, Casalecchio (Station 425)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://img.shields.io/badge/Test%20NSE-0.935-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Conformal%20Coverage-92.4%25-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model-PatchTST-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Uncertainty-Conformal%20Prediction-red?style=flat-square"/>
</p>

*Hourly discharge (Q) forecasting for the Reno river with statistically calibrated 90% uncertainty bands, built on TOPKAPI output and ready for PAB operational integration.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline](#-pipeline)
- [Current Results](#-current-results)
- [Folder Structure](#-folder-structure)
- [Getting Started](#-getting-started)
- [Operational Use](#-operational-use)
- [Output Format](#-output-format)
- [Configuration](#-configuration)
- [Notes for Maintainers](#-notes-for-maintainers)
- [Author](#-author)

---

## 🔬 Overview

This project (Phase I of the roadmap) reads TOPKAPI hydrological model output for the Reno river at Casalecchio (station 425) and produces an hourly discharge forecast with a statistically calibrated 90% uncertainty band — a central estimate (`Qfor_mid`) plus lower and upper bounds (`Qfor_low`, `Qfor_high`) that together tell PAB not just what the river will do, but how confident the model is.

**Key methods:** PatchTST (patch-based transformer for time series) + conformal prediction for distribution-free uncertainty calibration.

---

## ⚙️ Pipeline

| Step | Script | Purpose |
|------|--------|---------|
| — | `0_check_setup.py` | Verifies environment, config, folders, and data before running |
| 1 | `1_prepare_data.py` | Feature engineering, train/val/test split |
| 2 | `2_train.py` | Trains the PatchTST model |
| 3 | `3_evaluate.py` | NSE / PBIAS / plots — model quality report |
| 4 | `4_conformal.py` | Calibrates the uncertainty band |
| 5 | `5_realtime_infer.py` | Operational forecast — runs in production |

Input: TOPKAPI output (`.sbs.ts`) → Output: calibrated hourly forecast with uncertainty bands.

---

## 📊 Current Results

> ⚠️ **Note:** These numbers are from the most recent full run using the split *before* Fix A (val = 2024 Jan–Jun, test = 2024 Jul–Dec). Fix A changed the split (val = 2023 full year, test = 2024 full year) to fix poor peak-flood coverage — see `docs/results_summary.md`. These numbers have not yet been re-validated against the new split and are not directly comparable.

| Metric | Value (pre-Fix-A) |
|--------|-------------------|
| Validation NSE | 0.9484 |
| Test NSE (2024 Jul–Dec only) | 0.9347 |
| Test PBIAS | −3.00% |
| Test NSE_peak (flood events) | 0.8609 |
| Conformal coverage | 92.4% (target 90%) |
| Conformal peak coverage | 43.8% (n=96 events, known weak point) |
| Mean uncertainty band width | 38.9 m³/s |

See `docs/results_summary.md` for the full history and `docs/known_limitations.md` for what is still being improved.

---

## 📁 Folder Structure

| Path | Description |
|------|-------------|
| `.vscode/settings.json` | VS Code workspace config (venv path, cwd) |
| `requirements.txt` | `pip install -r requirements.txt` |
| `LSTM.ini` | All configuration — paths, hyperparameters |
| `0_check_setup.py` | Run first — verifies the environment |
| `1_prepare_data.py` | Step 1: build train/val/test arrays |
| `2_train.py` | Step 2: train the model |
| `3_evaluate.py` | Step 3: evaluate and plot |
| `4_conformal.py` | Step 4: calibrate uncertainty bands |
| `5_realtime_infer.py` | Step 5: operational forecast |
| `data/raw/425.sbs.ts` | ← place the TOPKAPI export here |
| `data/processed/` | Created automatically by step 1 |
| `models/` | Created automatically by step 2 |
| `outputs/plots/` | Created automatically by steps 3–4 |
| `docs/` | Project notes, results history |

---

## 🚀 Getting Started

### First-time setup (VS Code + venv)

Requires **Python 3.9 or newer** (tested on 3.12). Check with `python --version` before creating the venv.

```bash
# 1. Open the folder in VS Code
#    File > Open Folder... and select this repo

# 2. Create the virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 3. Select the interpreter in VS Code:
#    Ctrl+Shift+P > "Python: Select Interpreter" > choose .venv

# 4. Verify the setup
python 0_check_setup.py

# 5. Place the TOPKAPI export at data/raw/425.sbs.ts

# 6. Run the full chain in order
python 1_prepare_data.py
python 2_train.py
python 3_evaluate.py
python 4_conformal.py
python 5_realtime_infer.py
```

Every script locates its own folder automatically, so they find `LSTM.ini` and the `data/`, `models/`, `outputs/` folders correctly regardless of the terminal's working directory.

---

## 🏃 Operational Use

Once trained, only step 5 needs to run repeatedly (e.g. once per TOPKAPI update cycle):

```bash
python 5_realtime_infer.py --live-ts path/to/latest_topkapi_output.sbs.ts
```

This writes `outputs/forecast_latest.csv` — the file PAB should read. If `--live-ts` is omitted, it falls back to the file configured in `LSTM.ini`.

**Historical replay** (test before going live):

```bash
python 5_realtime_infer.py --live-ts data/raw/425.sbs.ts --replay
```

Produces `outputs/replay_output.csv` plus a diagnostics summary (NSE, PBIAS, coverage, false alarm rate).

---

## 📤 Output Format

`outputs/forecast_latest.csv` — the file for PAB integration:

| Column | Meaning |
|--------|---------|
| `run_timestamp` | When this forecast was generated (UTC) |
| `forecast_time` | The hour being forecast |
| `Qfor_mid` | Central discharge forecast (m³/s) |
| `Qfor_low` / `Qfor_high` | 90% uncertainty band (m³/s) |
| `QM_head1` | Raw output of head 1 (log-space), diagnostic |
| `QM_head2` | Raw output of head 2 (magnitude), diagnostic |
| `blend_weight` | How much head2 vs head1 contributed to `Qfor_mid` (0–1) |
| `head_disagreement` | Absolute difference between heads — widens band when high |
| `rise_3h` | Change in `Qfor_mid` over last 3 hours — flood/recession signal |
| `Q_TOPKAPI` | Raw TOPKAPI discharge, for comparison |
| `band_width` | `Qfor_high − Qfor_low` |
| `flood_alert` | 1 if `Qfor_high` crosses the alert threshold |
| `coverage_pct` | Nominal coverage of the band (90.0) |
| `model_version` | Git hash, for traceability |

The diagnostic columns are not required by PAB but help explain *why* a given forecast or alert fired.

---

## 🔧 Configuration

All tunable values live in `LSTM.ini` under `[patchtst]`:

| Parameter | Meaning |
|-----------|---------|
| `start_date` | Training data starts here |
| `seq_len` | Hours of lookback per forecast (currently 60 = 2.5 days) |
| `epochs`, `patience`, `huber_delta`, `loss_w1`, `loss_w2` | Training tuning (see `2_train.py` docstring) |
| `blend_threshold`, `blend_scale` | How the two output heads are combined |

---

## 📝 Notes for Maintainers

- The train/val/test split was chosen so the calibration set includes a full year with real autumn flood events — Apennine rivers produce their largest floods in September–November, and an earlier split missed that season, making the uncertainty band too narrow during the biggest floods.
- `4_conformal.py` and `5_realtime_infer.py` share identical band-calibration logic (`get_band()`) by design — if you change one, change the other, or the 90% coverage guarantee stops being accurate.
- See `docs/known_limitations.md` for the current open issue (peak coverage on the largest, rarest floods).

---

## 👤 Author

**Farhad Bayrami**
Machine Learning Engineer — PROGEA S.r.l., Bologna
📧 [farhad.bayrami@studio.unibo.it](mailto:farhad.bayrami@studio.unibo.it)
🔗 [GitHub](https://github.com/FarhadBayrami)

---

<div align="center">
  <sub>Operational hydrological forecasting · PROGEA S.r.l. · Reno River, Casalecchio di Reno</sub>
</div>