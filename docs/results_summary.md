# Results history

This documents how the model reached its current state, in order.
Useful context for anyone reviewing the project or continuing the work.

## v4 (initial PatchTST, before tuning)

Trained with default hyperparameters from the original Progea/Sorbolo
template. Early stopping was too aggressive (`patience=40`), stopping
training before the model had converged.

- Test NSE: 0.529
- Test PBIAS: −31.4%
- Conclusion: model systematically underestimated peak discharge

## v5 — training fixes

Changes: `patience` 40→80, `huber_delta` 1.0→5.0 (penalises large
errors harder), rebalanced `loss_w1`/`loss_w2` toward the magnitude
head, larger batch size.

- Val NSE: 0.9484
- Test NSE: 0.9347
- Test PBIAS: −3.00%
- Test NSE_peak: 0.8609

This was a large improvement — PBIAS dropped from −31% to −3%, and
NSE_peak (accuracy specifically on flood events) went from poor to
excellent.

## Conformal prediction — uncertainty bands

Added `4_conformal.py` to wrap the point forecast in a calibrated 90%
uncertainty band, using split conformal prediction on the validation
set residuals.

First run (val set = 2024 Jan-Jun):
- Overall coverage: 92.1% (target 90%) — good
- Peak coverage: 41.7% (n=96 events >200 m³/s) — the known weak point

### Diagnosing low peak coverage

The val set used for calibration (Jan-Jun) never contained a
September-November period, so the SON season had zero calibration
samples and fell back to a generic global quantile. Apennine rivers
produce their largest floods in autumn — the test set's worst event
(910.6 m³/s, 9 October 2024) was a type of event the calibration set
had never measured an error against.

### v2 — early-trigger bands

Added two new signals to widen the band before a flood peaked, rather
than only after the model's own prediction crossed a fixed threshold:

- Rate-of-rise trigger: widens the band when `Qfor_mid` is climbing
  fast (not just when it's already high)
- Head-disagreement signal: the two model output heads (log-space and
  magnitude) tend to diverge right before a peak — that divergence is
  used as a free, input-dependent uncertainty signal

Result: peak coverage 41.7% → 43.8%. A real but modest improvement —
once the model itself got accurate (PBIAS −3%), the two heads stopped
disagreeing with each other except right at the steepest part of a
flood, limiting how much this approach alone could help.

### v3 — symmetric band widening

Found and fixed an asymmetry bug: the lower bound (`Qfor_low`) had no
widening mechanism at all, even though the upper bound had three. This
caused the model to badly under-cover the falling limb right after a
flood peak (e.g. true value 197 m³/s, `Qfor_low` sitting at 258 m³/s —
above the truth). Fixed by mirroring the rate-trigger and head-
disagreement logic onto the lower side, scaled by the magnitude of the
observed fall rate.

## Fix A — calibration set redesign (current)

Root-caused the peak-coverage ceiling to the calibration set itself:
no amount of trigger logic can widen a band beyond the largest error
the calibration set has ever measured, and the old calibration set
(2024 H1) never measured an autumn-flood-sized error.

Changed the train/val/test split:

| | Before | After |
|---|---|---|
| Train | 2014-2023 | 2014-2022 |
| Val (calibration) | 2024 Jan-Jun | **2023 full year** |
| Test | 2024 Jul-Dec | **2024 full year** |

The val set now contains a complete year, including real
September-November data, so SON gets genuine calibration samples
instead of falling back to a manual safety multiplier. The SON
multiplier in `get_band()` was changed to only apply conditionally,
so it automatically stops firing once real SON data exists.

**Status: implemented, not yet re-validated against the real trained
model.** Run the full chain (`1` through `5`) and update this file
with the new numbers once available.
