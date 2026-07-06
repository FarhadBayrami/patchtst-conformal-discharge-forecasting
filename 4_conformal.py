"""
Snippet 4 — Conformal Prediction  (v2 — Casalecchio di Reno)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset   : 425.sbs.ts  (station 425, Casalecchio di Reno)
Timestep  : hourly

CHANGES vs v1
──────────────
v1 only widened the band when Qfor_mid crossed a fixed trigger
(PEAK_TRIGGER_M3S=60). That trigger fires too late on a fast-rising
flood because the model is still climbing the rising limb when the
true peak occurs — by the time Qfor_mid crosses 60, the river may
already be at 200+.

v2 adds two earlier, complementary signals on top of the v1 trigger:

  FIX 1 — RATE-OF-RISE TRIGGER
    If Qfor_mid has risen by more than RISE_TRIGGER_M3S over the last
    RISE_WINDOW_HOURS hours, the peak quantile is applied immediately,
    regardless of the absolute level. This catches the flood while it
    is still accelerating instead of waiting for it to cross a fixed
    number.

  FIX 2 — HEAD-DISAGREEMENT SIGNAL
    head1 (log-space) and head2 (magnitude) tend to diverge sharply
    right at flood peaks because the two representations saturate
    differently. abs(head1 - head2) is a free, input-dependent
    uncertainty signal with no extra calibration needed. The band
    upper radius is widened to at least 1.5x this disagreement.

Both fixes are applied in get_band(), which both this script and
5_realtime_infer.py call identically — they must stay in sync or the
conformal coverage guarantee breaks.

THEORY RECAP
─────────────
Run the trained model on the calibration set (val set — data it
never trained on).  Nonconformity score = |observed - predicted|.
q_hat = the (1-alpha) finite-sample-corrected quantile of those
scores.  Qfor_low/high = Qfor_mid -+ q_hat.  This guarantees marginal
coverage >= (1-alpha) under exchangeability between calibration and
test residuals.  v2 does not change this guarantee — it only changes
WHEN the wider peak-tail quantile gets applied, trading a bit more
average band width for much better peak coverage.

OUTPUTS
────────
  models/conformal_qhats.json          — quantiles (unchanged structure)
  outputs/conformal_intervals_test.csv — adds head_disagreement, rise_3h columns
  outputs/plots/CONF_01_band.png .. CONF_04_seasonal.png
"""

import os, sys, json, pickle, configparser, warnings, csv
print("Loading libraries (numpy, pandas, torch, matplotlib)…", flush=True)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
print("Libraries loaded.", flush=True)

warnings.filterwarnings("ignore", category=UserWarning)

# ── config ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_cfg  = configparser.ConfigParser()
_cfg.read(os.path.join(_HERE, "LSTM.ini"), encoding="utf-8-sig")
_pt   = _cfg["patchtst"]

DATA_DIR  = os.path.join(_HERE, _pt["data_dir"])
MODEL_DIR = os.path.join(_HERE, _pt["model_dir"])
OUT_DIR   = os.path.join(_HERE, _pt["out_dir"])
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

BLEND_THRESHOLD = _pt.getfloat("blend_threshold")
BLEND_SCALE     = _pt.getfloat("blend_scale")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA = 0.10                     # 90% target coverage
FLOOD_THRESHOLD_M3S = 200.0      # PAB alert level — confirm with supervisor

# v1 trigger (kept as a floor — still applies)
PEAK_TRIGGER_M3S = FLOOD_THRESHOLD_M3S * 0.30      # 60 m³/s

# v2 — Fix 1: rate-of-rise trigger
RISE_WINDOW_HOURS = 3             # look back this many hourly steps
RISE_TRIGGER_M3S  = 30.0          # m³/s increase within the window → flood signal
FALL_TRIGGER_M3S  = 20.0          # m³/s decrease within the window → recession signal
                                  # lower than RISE_TRIGGER because recessions
                                  # decelerate gradually — a stricter cutoff missed
                                  # the tail end of a falling limb where Qfor_mid
                                  # is still elevated relative to the true drop
FALL_SCALE_MULT   = 1.6           # ql widening = max(peak_qhat_lower, |rise_3h| * this)

# v2 — Fix 2: head-disagreement multiplier
HEAD_DISAGREEMENT_MULT = 1.5      # band upper >= 1.5 x |head1 - head2|

# ── colours ───────────────────────────────────────────────────────────────────
C_OBS, C_PRED, C_BAND, C_TOPKAPI, C_FLOOD = "#1565C0", "#C62828", "#C62828", "#2E7D32", "#BA7517"

# ══════════════════════════════════════════════════════════════════════════════
# MODEL — batched reshape/permute forward(), identical to 2_train.py
# ══════════════════════════════════════════════════════════════════════════════
class PatchTST(nn.Module):
    def __init__(self, n_features, seq_len, patch_len, stride,
                 d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.n_features = n_features
        self.patch_len  = patch_len
        self.stride     = stride
        self.n_patches  = (seq_len - patch_len) // stride + 1
        self.patch_embed   = nn.Linear(patch_len, d_model)
        self.pos_embed     = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm  = nn.LayerNorm(d_model)
        trunk_in   = n_features * self.n_patches * d_model
        self.trunk = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(trunk_in),
            nn.Linear(trunk_in, d_model * 2), nn.GELU(), nn.Dropout(dropout))
        self.head1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(d_model, 1))
        self.head2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(d_model, 1), nn.Softplus())

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, T)
        p = x.unfold(1, self.patch_len, self.stride)
        p = self.patch_embed(p) + self.pos_embed
        p = self.input_dropout(p)
        p = self.norm(self.encoder(p))
        out   = p.reshape(B, C, self.n_patches, -1)
        trunk = self.trunk(out)
        return self.head1(trunk), self.head2(trunk)


# ── load model + scalers ──────────────────────────────────────────────────────
_ckpt_path = os.path.join(MODEL_DIR, "checkpoint.pt")
if not os.path.exists(_ckpt_path):
    print(f"\n✗ No trained model found at {_ckpt_path}")
    print(f"  Run 1_prepare_data.py and 2_train.py first — 2_train.py must")
    print(f"  finish (or early-stop) to produce checkpoint.pt before this")
    print(f"  script can calibrate the uncertainty bands.")
    sys.exit(1)

ckpt         = torch.load(_ckpt_path, map_location=DEVICE)
hp           = ckpt["hparams"]
model        = PatchTST(**hp).to(DEVICE)
best_weights = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
model.load_state_dict(best_weights)
model.eval()
print(f"Model loaded  —  val NSE: {ckpt.get('val_nse', float('nan')):.4f}")

print("Loading scalers (this can take 10-30s on first run — sklearn import)…", flush=True)
with open(os.path.join(DATA_DIR, "scaler_y1.pkl"), "rb") as f:
    scaler_y1 = pickle.load(f)
QM_TRAIN_MAX = float(np.load(os.path.join(DATA_DIR, "QM_TRAIN_MAX.npy")).item())
print("Scalers loaded.", flush=True)


def run_inference(X_tensor):
    """Returns blended prediction + both raw head outputs (all m³/s)."""
    p1_list, p2_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 256):
            xb = X_tensor[i:i + 256].to(DEVICE)
            o1, o2 = model(xb)
            p1_list.append(o1.cpu().numpy())
            p2_list.append(o2.cpu().numpy())
    pred1_sc   = np.vstack(p1_list)
    pred2_norm = np.vstack(p2_list)
    pred1_m3s  = np.clip(np.expm1(scaler_y1.inverse_transform(pred1_sc).ravel()), 0, None)
    pred2_m3s  = np.clip(pred2_norm.ravel() * QM_TRAIN_MAX, 0, None)
    blend_w    = 1 / (1 + np.exp(-(pred2_m3s - BLEND_THRESHOLD) / BLEND_SCALE))
    blended    = (1 - blend_w) * pred1_m3s + blend_w * pred2_m3s
    return blended, pred1_m3s, pred2_m3s


TFMT = mdates.DateFormatter("%b %Y")
def save_fig(fig, fname):
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


def conformal_quantile(scores, alpha):
    """Finite-sample corrected conformal quantile."""
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def month_to_season(m):
    return {12:"DJF",1:"DJF",2:"DJF", 3:"MAM",4:"MAM",5:"MAM",
             6:"JJA",7:"JJA",8:"JJA", 9:"SON",10:"SON",11:"SON"}[m]


# ══════════════════════════════════════════════════════════════════════════════
# get_band() — THE function shared with 5_realtime_infer.py
# Both scripts must apply this exact logic or coverage drifts apart.
# ══════════════════════════════════════════════════════════════════════════════
def get_band(season_qhats, peak_qhat_upper, peak_qhat_lower, season, q_mid,
            head_disagreement=None, rise_3h=None):
    """
    Returns (q_lower, q_upper) — the asymmetric conformal radii.

    season             : "DJF"/"MAM"/"JJA"/"SON"
    q_mid              : blended central forecast (m³/s)
    head_disagreement  : abs(head1_m3s - head2_m3s), or None to skip Fix 2
    rise_3h            : Qfor_mid(t) - Qfor_mid(t - RISE_WINDOW_HOURS), or None to skip Fix 1

    v3 — SYMMETRIC WIDENING (fixes overly narrow lower bound)
    ─────────────────────────────────────────────────────────
    v1/v2 only widened qu (upper). That left ql (lower) frozen at a static
    seasonal value even during a fast-falling recession limb right after a
    peak, when Qfor_mid is still elevated but the true value is dropping
    fast. Observed symptom: obs below Qfor_low immediately after a flood
    peak (e.g. obs=197, Qfor_low=258.6 on the falling limb of 19 Sep 2024).

    Fix: mirror the rise-rate and head-disagreement triggers onto ql.
    A sharply FALLING rise_3h (large negative value) means the model may
    still be lagging the true drop — widen ql the same way a sharply
    rising rise_3h widens qu. Head disagreement is direction-agnostic
    (it just says "the model is uncertain right now") so it widens both
    sides equally.
    """
    sq = season_qhats.get(season, season_qhats.get("global"))
    ql = float(sq["q_lower"])
    qu = float(sq["q_upper"])

    # SON safety multiplier — ONLY applies if SON still has too few real
    # calibration samples (the "fallback" flag set in Step 1 below).
    # After Fix A (val set = full year 2023), SON should have real data
    # and this multiplier will no longer fire — the real seasonal quantile
    # takes over automatically, no code change needed.
    if season == "SON" and sq.get("fallback", False):
        qu = qu * 2.0
        ql = ql * 2.0

    # v1 — absolute-level trigger (kept as floor, upper side only:
    # a high absolute level is specifically a flood-peak risk, not a
    # recession risk, so this one stays asymmetric by design)
    if q_mid >= PEAK_TRIGGER_M3S:
        qu = max(qu, peak_qhat_upper)

    # v2 Fix 1 — rate-of-rise trigger (upper) / rate-of-fall trigger (lower)
    # Upper: a flat peak_qhat_upper kick-in is fine — rises plateau near the
    #   calibrated peak magnitude, so a fixed quantile is a reasonable bound.
    # Lower: a fall can overshoot the fixed peak_qhat_lower (calibrated from
    #   typical peak-event overshoots, not from the size of THIS fall). Scale
    #   ql by the fall magnitude itself so a 100 m³/s/3h drop gets a much
    #   bigger allowance than a 30 m³/s/3h drop, with peak_qhat_lower as a
    #   floor (never narrower than the calibrated minimum).
    if rise_3h is not None:
        if rise_3h >= RISE_TRIGGER_M3S:
            qu = max(qu, peak_qhat_upper)
        if rise_3h <= -FALL_TRIGGER_M3S:
            ql = max(ql, peak_qhat_lower, abs(rise_3h) * FALL_SCALE_MULT)

    # v2 Fix 2 — head disagreement widens BOTH sides (direction-agnostic)
    if head_disagreement is not None:
        qu = max(qu, head_disagreement * HEAD_DISAGREEMENT_MULT)
        ql = max(ql, head_disagreement * HEAD_DISAGREEMENT_MULT)

    return ql, qu


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 52)
print("  STEP 1 — CALIBRATION")
print("─" * 52)

X_val   = torch.tensor(np.load(os.path.join(DATA_DIR, "X_val.npy")), dtype=torch.float32)
ref_val = pd.read_csv(os.path.join(DATA_DIR, "val_reference.csv"))
ref_val["Time"] = pd.to_datetime(ref_val["Time"])

pred_val, head1_val, head2_val = run_inference(X_val)
obs_val = ref_val["QM"].values

residuals = pred_val - obs_val
abs_resid = np.abs(residuals)
n_cal = len(residuals)

print(f"Calibration samples : {n_cal}")
print(f"Residual mean       : {residuals.mean():+.2f} m³/s")
print(f"Residual std        : {residuals.std():.2f} m³/s")
print(f"Abs residual 90th   : {np.percentile(abs_resid, 90):.2f} m³/s")

q_hat_sym = conformal_quantile(abs_resid, ALPHA)
undershoots = np.clip(-residuals, 0, None)
overshoots  = np.clip( residuals, 0, None)
q_upper_asym = conformal_quantile(undershoots, ALPHA / 2)
q_lower_asym = conformal_quantile(overshoots,  ALPHA / 2)

print(f"\nGlobal quantiles (90% coverage):")
print(f"  Symmetric   ±{q_hat_sym:.2f} m³/s")
print(f"  Asymmetric   lower: −{q_lower_asym:.2f}  upper: +{q_upper_asym:.2f} m³/s")
print(f"  Asymmetry ratio: {q_upper_asym / max(q_lower_asym, 0.1):.2f}x")

peak_mask_cal = obs_val >= FLOOD_THRESHOLD_M3S
n_peaks_cal   = int(peak_mask_cal.sum())
if n_peaks_cal >= 10:
    q_upper_peak = conformal_quantile(undershoots[peak_mask_cal], ALPHA / 2)
    q_lower_peak = conformal_quantile(overshoots[peak_mask_cal],  ALPHA / 2)
else:
    q_upper_peak, q_lower_peak = q_upper_asym, q_lower_asym
print(f"\nPeak quantiles  (n={n_peaks_cal}, QM>{FLOOD_THRESHOLD_M3S:.0f} m³/s):")
print(f"  Lower: −{q_lower_peak:.2f}  Upper: +{q_upper_peak:.2f} m³/s")

seasons_cal  = ref_val["Time"].dt.month.map(month_to_season).values
season_qhats = {}
MIN_SEASONAL = 20
print(f"\nSeasonal quantiles:")
for season in ["DJF","MAM","JJA","SON"]:
    mask = seasons_cal == season
    n_s  = int(mask.sum())
    if n_s >= MIN_SEASONAL:
        qu = conformal_quantile(undershoots[mask], ALPHA / 2)
        ql = conformal_quantile(overshoots[mask],  ALPHA / 2)
        season_qhats[season] = {"q_lower": round(ql,3), "q_upper": round(qu,3), "n": n_s}
        print(f"  {season}  n={n_s:4d}   lower: −{ql:.1f}  upper: +{qu:.1f} m³/s")
    else:
        season_qhats[season] = {"q_lower": round(q_lower_asym,3),
                                "q_upper": round(q_upper_asym,3),
                                "n": n_s, "fallback": True}
        print(f"  {season}  n={n_s:4d}   (fallback to global)")

season_qhats["global"] = {"q_lower": round(q_lower_asym,3), "q_upper": round(q_upper_asym,3)}

qhats = {
    "alpha": ALPHA, "coverage_target": 1 - ALPHA, "n_calibration": n_cal,
    "global": {"symmetric": round(q_hat_sym,3), "q_lower": round(q_lower_asym,3), "q_upper": round(q_upper_asym,3)},
    "peak": {"threshold_m3s": FLOOD_THRESHOLD_M3S, "q_lower": round(q_lower_peak,3),
             "q_upper": round(q_upper_peak,3), "n": n_peaks_cal},
    "seasonal": season_qhats,
    "v2_params": {
        "peak_trigger_m3s": PEAK_TRIGGER_M3S,
        "rise_window_hours": RISE_WINDOW_HOURS,
        "rise_trigger_m3s": RISE_TRIGGER_M3S,
        "fall_trigger_m3s": FALL_TRIGGER_M3S,
        "fall_scale_mult": FALL_SCALE_MULT,
        "head_disagreement_mult": HEAD_DISAGREEMENT_MULT,
    },
}
with open(os.path.join(MODEL_DIR, "conformal_qhats.json"), "w") as f:
    json.dump(qhats, f, indent=2)
print(f"\n✓ conformal_qhats.json saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — APPLY TO TEST SET  (v1 floor + v2 fixes, compared side by side)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 52)
print("  STEP 2 — APPLY BANDS TO TEST SET")
print("─" * 52)

X_test   = torch.tensor(np.load(os.path.join(DATA_DIR, "X_test.npy")), dtype=torch.float32)
ref_test = pd.read_csv(os.path.join(DATA_DIR, "test_reference.csv"))
ref_test["Time"] = pd.to_datetime(ref_test["Time"])

pred_test, head1_test, head2_test = run_inference(X_test)
obs_test     = ref_test["QM"].values
Q_topkapi    = ref_test["Q"].values
times_dt     = pd.to_datetime(ref_test["Time"].values)
seasons_test = times_dt.month.map(month_to_season).values
head_disagreement_test = np.abs(head1_test - head2_test)

# rise_3h: how much Qfor_mid rose over the last RISE_WINDOW_HOURS
rise_3h_test = np.zeros(len(pred_test))
rise_3h_test[RISE_WINDOW_HOURS:] = pred_test[RISE_WINDOW_HOURS:] - pred_test[:-RISE_WINDOW_HOURS]

qlow_arr  = np.zeros(len(pred_test))
qhigh_arr = np.zeros(len(pred_test))
for i in range(len(pred_test)):
    ql, qu = get_band(season_qhats, qhats["peak"]["q_upper"], qhats["peak"]["q_lower"],
                      seasons_test[i], pred_test[i],
                      head_disagreement=head_disagreement_test[i],
                      rise_3h=rise_3h_test[i])
    qlow_arr[i]  = ql
    qhigh_arr[i] = qu

Qfor_low  = np.clip(pred_test - qlow_arr, 0, None)
Qfor_mid  = pred_test
Qfor_high = pred_test + qhigh_arr

covered      = (obs_test >= Qfor_low) & (obs_test <= Qfor_high)
peak_mask    = obs_test >= FLOOD_THRESHOLD_M3S
coverage_pct = covered.mean() * 100
cov_peak     = covered[peak_mask].mean() * 100 if peak_mask.sum() > 0 else float("nan")
band_width_mean = (Qfor_high - Qfor_low).mean()

print(f"\nOverall coverage    : {coverage_pct:.1f}%  (target: {(1-ALPHA)*100:.0f}%)")
print(f"Peak-flow coverage  : {cov_peak:.1f}%  (n={int(peak_mask.sum())} events >{FLOOD_THRESHOLD_M3S:.0f} m³/s)")
print(f"Mean band width     : {band_width_mean:.1f} m³/s")

out_df = pd.DataFrame({
    "datetime":          times_dt.strftime("%Y-%m-%d %H:%M"),
    "QM_observed":       np.round(obs_test, 3),
    "Qfor_mid":          np.round(Qfor_mid, 3),
    "Qfor_low":          np.round(Qfor_low, 3),
    "Qfor_high":         np.round(Qfor_high, 3),
    "Q_TOPKAPI":         np.round(Q_topkapi, 3),
    "head_disagreement": np.round(head_disagreement_test, 3),
    "rise_3h":           np.round(rise_3h_test, 3),
    "band_width":        np.round(Qfor_high - Qfor_low, 3),
    "covered":           covered.astype(int),
    "is_peak":           peak_mask.astype(int),
    "flood_alert":       (Qfor_high >= FLOOD_THRESHOLD_M3S).astype(int),
})
csv_path = os.path.join(OUT_DIR, "conformal_intervals_test.csv")
out_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(f"\n✓ conformal_intervals_test.csv saved  ({len(out_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 52)
print("  STEP 3 — PLOTS")
print("─" * 52)

fig, ax = plt.subplots(figsize=(16, 5))
ax.axhline(FLOOD_THRESHOLD_M3S, color=C_FLOOD, lw=0.9, ls="--", alpha=0.7,
          label=f"Flood alert ({FLOOD_THRESHOLD_M3S:.0f} m³/s)")
ax.fill_between(times_dt, Qfor_low, Qfor_high, color=C_BAND, alpha=0.15, label="90% band")
ax.plot(times_dt, Qfor_low,  color=C_BAND, lw=0.6, alpha=0.4)
ax.plot(times_dt, Qfor_high, color=C_BAND, lw=0.6, alpha=0.4)
ax.plot(times_dt, Q_topkapi, color=C_TOPKAPI, lw=0.9, ls="--", alpha=0.8, label="Q TOPKAPI")
ax.plot(times_dt, Qfor_mid, color=C_PRED, lw=1.2, label="Qfor_mid", zorder=4)
ax.plot(times_dt, obs_test, color=C_OBS, lw=1.3, label="QM observed", zorder=5)
uncovered_peaks = (~covered) & peak_mask
if uncovered_peaks.sum() > 0:
    ax.scatter(times_dt[uncovered_peaks], obs_test[uncovered_peaks],
              color=C_PRED, s=40, zorder=6, marker="^",
              label=f"Missed peaks ({int(uncovered_peaks.sum())})")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
ax.set_xlabel("Date"); ax.set_ylabel("Discharge  [m³/s]")
ax.set_title(f"PatchTST forecast with 90% conformal band — "
            f"coverage: {coverage_pct:.1f}%  |  peak: {cov_peak:.1f}%  |  "
            f"width: {band_width_mean:.1f} m³/s")
ax.legend(loc="upper left", fontsize=8, ncol=3)
fig.tight_layout()
save_fig(fig, "CONF_01_band.png")

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
axes[0].fill_between(times_dt, Qfor_low, Qfor_high, color=C_BAND, alpha=0.2, label="90% band")
axes[0].plot(times_dt, Qfor_mid, color=C_PRED, lw=1.0, label="Qfor_mid")
axes[0].plot(times_dt, obs_test, color=C_OBS, lw=1.2, label="QM observed")
axes[0].axhline(FLOOD_THRESHOLD_M3S, color=C_FLOOD, lw=0.8, ls=":")
axes[0].set_ylabel("Q [m³/s]"); axes[0].legend(fontsize=8)
axes[0].set_title("Forecast vs observed with band")
band_width_arr = Qfor_high - Qfor_low
axes[1].fill_between(times_dt, 0, band_width_arr, color="#7B1FA2", alpha=0.4)
axes[1].plot(times_dt, band_width_arr, color="#7B1FA2", lw=0.8)
axes[1].set_ylabel("Band width [m³/s]")
axes[1].set_title("Uncertainty band width")
window = min(30, len(covered))
roll_cov = pd.Series(covered.astype(float)).rolling(window, min_periods=5).mean() * 100
axes[2].plot(times_dt, roll_cov, color="#1B5E20", lw=1.2)
axes[2].axhline(90, color="gray", lw=0.8, ls="--", label="90% target")
axes[2].set_ylim(0, 105); axes[2].set_ylabel("Rolling coverage [%]")
axes[2].set_title(f"30-day rolling coverage (overall: {coverage_pct:.1f}%)")
axes[2].legend(fontsize=8)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
fig.tight_layout()
save_fig(fig, "CONF_02_coverage.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(residuals, bins=40, color="#1565C0", alpha=0.7, edgecolor="white", lw=0.3)
axes[0].axvline(0, color="black", lw=1.0, ls="--")
axes[0].axvline(residuals.mean(), color=C_PRED, lw=1.2, ls="--", label=f"mean={residuals.mean():+.1f}")
axes[0].set_xlabel("Residual [m³/s] (pred - obs)"); axes[0].set_ylabel("Count")
axes[0].set_title("Calibration residuals"); axes[0].legend(fontsize=8)
sorted_abs = np.sort(abs_resid)
axes[1].plot(np.linspace(0, 100, len(sorted_abs)), sorted_abs, color="#1565C0", lw=1.2)
axes[1].axhline(q_hat_sym, color=C_PRED, lw=1.2, ls="--", label=f"q_hat={q_hat_sym:.1f}")
axes[1].set_xlabel("Percentile [%]"); axes[1].set_ylabel("Abs residual [m³/s]")
axes[1].set_title("Sorted calibration errors"); axes[1].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "CONF_03_residuals.png")

seasons_order = ["DJF","MAM","JJA","SON"]
q_lows  = [season_qhats[s]["q_lower"] for s in seasons_order]
q_highs = [season_qhats[s]["q_upper"] for s in seasons_order]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(4); w = 0.35
ax.bar(x - w/2, q_lows, w, label="Lower radius", color="#1565C0", alpha=0.75)
ax.bar(x + w/2, q_highs, w, label="Upper radius", color=C_PRED, alpha=0.75)
ax.set_xticks(x); ax.set_xticklabels(["Winter","Spring","Summer","Autumn"])
ax.set_ylabel("Band radius [m³/s]"); ax.set_title("Seasonal conformal quantiles")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "CONF_04_seasonal.png")



print("\n" + "═" * 52)
print("  CONFORMAL PREDICTION SUMMARY")
print("═" * 52)
print(f"  Coverage target    : {(1-ALPHA)*100:.0f}%")
print(f"  Achieved coverage  : {coverage_pct:.1f}%")
print(f"  Peak coverage      : {cov_peak:.1f}%  (n={int(peak_mask.sum())} events)")
print(f"  Mean band width    : {band_width_mean:.1f} m³/s")
print("═" * 52)
print(f"\n✓ All outputs saved to {OUT_DIR} and {MODEL_DIR}")
