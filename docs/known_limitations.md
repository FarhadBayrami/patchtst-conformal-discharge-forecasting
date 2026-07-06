# Known limitations

Honest list of what is not yet solved, for transparency with the team.

## 1. Peak coverage on extreme events

As of the last validated run (before Fix A), the 90% uncertainty band
correctly bounded the true discharge 92.4% of the time overall, but
only 43.8% of the time specifically during flood events above
200 m³/s. The single largest event in the dataset (910.6 m³/s,
9 October 2024) exceeded the band at its peak.

Root cause: the calibration set used to size the band had never
measured an error of that magnitude, because it didn't include an
autumn flood season. Fix A (see `docs/results_summary.md`) directly
addresses this by changing the calibration period to include a full
year with real autumn extremes. This needs to be re-run and
re-validated — the numbers above are pre-Fix-A.

## 2. SON safety multiplier is now conditional, not removed

The code that doubles the band width for September-November when
calibration data is insufficient is still present in `4_conformal.py`
and `5_realtime_infer.py`, but now only activates if the season
genuinely lacks enough samples (checked automatically). After Fix A,
this should rarely or never trigger — but if the real 2023 data turns
out to have had an unusually quiet autumn, it may still apply. Check
the seasonal breakdown printed by `1_prepare_data.py` after each run.

## 3. Flood alert threshold is a placeholder

`FLOOD_THRESHOLD_M3S = 200.0` in `4_conformal.py` and
`5_realtime_infer.py` was chosen based on visually inspecting the
historical flow duration curve, not an official PAB warning level.
This needs to be confirmed with the supervisor / PAB team before
operational deployment.

## 4. No automated retraining

The model is trained once and then used statically for inference.
There is no mechanism yet to detect model drift (e.g. if next year's
floods look different from the training distribution) or to trigger
retraining automatically. This is in scope for Phase II, per the
original roadmap (GAN-based error correction).

## 5. CPU-only inference assumed

All scripts default to CPU (`torch.device("cuda" if available else
"cpu")` — they will use a GPU automatically if one is present, but no
testing has been done on GPU hardware. Full training (400 epochs) on
CPU can take a long time depending on dataset size; budget accordingly
or consider GPU access for retraining cycles.

## 6. No automated tests

There is no test suite. Correctness has been verified manually,
including a side-by-side check that `4_conformal.py` and
`5_realtime_infer.py` produce mathematically identical band
calculations for the same inputs (they must stay in sync for the
coverage guarantee to hold), and that all four files defining the
`PatchTST` class produce numerically identical outputs given the same
weights. If any of these scripts are modified, this should be
re-verified.

## 7. Interrupting 2_train.py loses checkpoint.pt

`best_model.pt` is saved after every epoch that improves validation
NSE, so it survives an interrupted run. `checkpoint.pt` — which
carries the architecture hyperparameters (`hparams`) that
`3_evaluate.py`, `4_conformal.py`, and `5_realtime_infer.py` all need
to reconstruct the model — is only written once, after the training
loop finishes completely (normal end or early stopping). If you
Ctrl+C out of `2_train.py` partway through (plausible, since full
training can take a long time), `checkpoint.pt` will be missing and
the next scripts will fail with a clear `FileNotFoundError` rather
than silently using stale or wrong hyperparameters — the failure is
loud, not silent, but it does mean an interrupted run cannot be
resumed from; you would need to let training run to completion, or
manually note `hparams` and reconstruct `checkpoint.pt` by hand.
