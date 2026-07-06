# Casalecchio di Reno — PatchTST discharge forecasting

Station 425, Progea / TOPKAPI integration. Phase I of the roadmap:
hourly discharge (Q) forecasting with calibrated uncertainty bands,
ready for PAB integration.

## What this does

Reads TOPKAPI hydrological model output for the Reno river at
Casalecchio (station 425) and produces an hourly discharge forecast
with a statistically calibrated 90% uncertainty band — a central
estimate (`Qfor_mid`) plus a lower and upper bound (`Qfor_low`,
`Qfor_high`) that together tell PAB not just what the river will do,
but how confident the model is.

```
TOPKAPI output (.sbs.ts)
        │
        ▼
  1_prepare_data.py    feature engineering, train/val/test split
        │
        ▼
  2_train.py            trains the PatchTST model
        │
        ▼
  3_evaluate.py          NSE / PBIAS / plots — model quality report
        │
        ▼
  4_conformal.py         calibrates the uncertainty band
        │
        ▼
  5_realtime_infer.py    operational forecast — this is what runs in production
```

## Current results

**Note:** the numbers below are from the most recent full run, which
used the split *before* Fix A (val = 2024 Jan-Jun, test = 2024 Jul-Dec
only). Fix A changed the split (val = 2023 full year, test = 2024 full
year) specifically to fix poor peak-flood coverage — see
`docs/results_summary.md` for the full reasoning. **These numbers have
not yet been re-validated against the new split.** Run the full chain
(`1` through `5`) and update this table once you have fresh numbers —
they will not be directly comparable to the ones below since the test
set itself is now different (full year vs. half year).

| Metric | Value (pre-Fix-A) |
|---|---|
| Validation NSE | 0.9484 |
| Test NSE (2024 Jul-Dec only) | 0.9347 |
| Test PBIAS | −3.00% |
| Test NSE_peak (flood events) | 0.8609 |
| Conformal coverage | 92.4% (target 90%) |
| Conformal peak coverage | 43.8% (n=96 events, known weak point) |
| Mean uncertainty band width | 38.9 m³/s |

See `docs/results_summary.md` for the full history of how these
numbers were reached, and `docs/known_limitations.md` for what is
still being improved.

## Folder structure

```
casalecchio_patchtst_conformal/
  .vscode/settings.json       VS Code workspace config (venv path, cwd)
  requirements.txt             pip install -r requirements.txt
  LSTM.ini                    all configuration — paths, hyperparameters
  0_check_setup.py             run this first — verifies the environment
  1_prepare_data.py           step 1: build train/val/test arrays
  2_train.py                  step 2: train the model
  3_evaluate.py                step 3: evaluate and plot
  4_conformal.py               step 4: calibrate uncertainty bands
  5_realtime_infer.py          step 5: operational forecast
  data/
    raw/425.sbs.ts             <- put the TOPKAPI export here
    processed/                 created automatically by step 1
  models/                      created automatically by step 2
  outputs/
    plots/                     created automatically by steps 3-4
  docs/                        project notes, results history
```

## How to run (VS Code + venv)

### First-time setup

0. Requires **Python 3.9 or newer** (tested on 3.12). Check with
   `python --version` before creating the venv — an older Python will
   fail to install `torch` from `requirements.txt`.

1. Open this folder (`casalecchio_patchtst_conformal`) in VS Code:
   `File > Open Folder...` and select it. VS Code will auto-detect
   `.vscode/settings.json` and look for a venv at `.venv` inside this
   folder.

2. Create the virtual environment. Open a terminal in VS Code
   (`` Terminal > New Terminal `` — it opens already `cd`'d into this
   folder) and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Select the interpreter: press `Ctrl+Shift+P`, type
   "Python: Select Interpreter", choose the one inside `.venv` (VS
   Code usually finds it automatically once created — look for
   `.venv\Scripts\python.exe` in the list).

4. Verify everything is wired correctly:
   ```
   python 0_check_setup.py
   ```
   This checks packages, the venv, `LSTM.ini`, the folder structure,
   and whether the raw data file is in place — before you run anything
   that could fail confusingly three scripts later.

5. Place the TOPKAPI export file at `data/raw/425.sbs.ts`. Format:
   space-delimited, one header row, columns
   `YYYY MM DD HH mm QM Q Rain Prec Evap Snow Temp Etp Soil SoilSat
   Perco Surf YSnow EnSnow SWE Deep DeepSat Inf2Surf`.

6. Run the full chain in order, from the integrated terminal:
   ```
   python 1_prepare_data.py
   python 2_train.py
   python 3_evaluate.py
   python 4_conformal.py
   python 5_realtime_infer.py
   ```

   Every script locates its own folder automatically
   (`os.path.dirname(os.path.abspath(__file__))`), so they will find
   `LSTM.ini` and the `data/`, `models/`, `outputs/` folders correctly
   even if VS Code's terminal working directory is something else —
   you do not need to `cd` into this folder manually before running,
   though `.vscode/settings.json` sets that as the default anyway.

### Day-to-day operational use

Once trained, only step 5 needs to run repeatedly (e.g. once per
TOPKAPI update cycle):

```
python 5_realtime_infer.py --live-ts path\to\latest_topkapi_output.sbs.ts
```

This writes `outputs/forecast_latest.csv` — the file PAB should read.
If `--live-ts` is omitted, it falls back to the file configured in
`LSTM.ini` (`data/raw/425.sbs.ts`).

### Testing before going live

Run a full historical replay to see how the model would have
performed in real time across the entire dataset:

```
python 5_realtime_infer.py --live-ts data/raw/425.sbs.ts --replay
```

This produces `outputs/replay_output.csv` plus a diagnostics summary
(NSE, PBIAS, coverage, false alarm rate) printed to the console.

## Output format

`outputs/forecast_latest.csv` — the file for PAB integration:

| column | meaning |
|---|---|
| `run_timestamp` | when this forecast was generated (UTC) |
| `forecast_time` | the hour being forecast |
| `Qfor_mid` | central discharge forecast (m³/s) |
| `Qfor_low` / `Qfor_high` | 90% uncertainty band (m³/s) |
| `QM_head1` | raw output of head 1 (log-space), m³/s — diagnostic |
| `QM_head2` | raw output of head 2 (magnitude), m³/s — diagnostic |
| `blend_weight` | how much head2 vs head1 contributed to `Qfor_mid` (0-1) |
| `head_disagreement` | `\|QM_head1 - QM_head2\|` — widens the band when high |
| `rise_3h` | change in `Qfor_mid` over the last 3 hours — flood/recession signal |
| `Q_TOPKAPI` | raw TOPKAPI discharge, for comparison |
| `band_width` | `Qfor_high - Qfor_low` |
| `flood_alert` | 1 if `Qfor_high` crosses the alert threshold |
| `coverage_pct` | nominal coverage of the band (90.0) |
| `model_version` | git hash, for traceability |

The diagnostic columns (`QM_head1`, `QM_head2`, `blend_weight`,
`head_disagreement`, `rise_3h`) are not required by PAB, but are
useful for understanding *why* a given forecast or alert fired — see
`docs/results_summary.md` for what `head_disagreement` and `rise_3h`
are for.

## Configuration reference

All tunable values live in `LSTM.ini` under `[patchtst]`. The most
relevant for someone reviewing this project:

- `start_date` — training data starts here
- `seq_len` — hours of lookback per forecast (currently 60 = 2.5 days)
- `epochs`, `patience`, `huber_delta`, `loss_w1`, `loss_w2` — training
  tuning, see `2_train.py` docstring for the reasoning behind each value
- `blend_threshold`, `blend_scale` — how the two model output heads
  (log-space and magnitude) are combined into one forecast

## Notes for whoever picks this up next

- The train/val/test split (see `1_prepare_data.py` docstring) was
  specifically chosen so the calibration set includes a full year
  with real autumn flood events — this matters because Apennine
  rivers produce their largest floods in September-November, and an
  earlier version of this split missed that season entirely, causing
  the uncertainty band to be too narrow during the biggest floods.
- `4_conformal.py` and `5_realtime_infer.py` share identical
  band-calibration logic (`get_band()`) by design — if you ever
  change one, change the other to match, or the stated 90% coverage
  guarantee stops being accurate.
- See `docs/known_limitations.md` for the current open issue (peak
  coverage on the very largest, rarest floods) and what would fix it.
