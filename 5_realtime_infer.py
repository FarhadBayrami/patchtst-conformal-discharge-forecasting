"""
Snippet 5 — Real-Time Inference  (v2 — Casalecchio di Reno)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset   : 425.sbs.ts  (station 425, Casalecchio di Reno)
Timestep  : hourly

CHANGES vs v1
──────────────
get_band() is now IDENTICAL to the one in 4_conformal.py:

  v1 trigger (floor)      : qu upgraded once Qfor_mid >= PEAK_TRIGGER_M3S
  v2 Fix 1 (rate-of-rise)  : qu upgraded if Qfor_mid rose by
                              RISE_TRIGGER_M3S over the last
                              RISE_WINDOW_HOURS hours — catches the
                              flood while it is still accelerating,
                              not after it crosses an absolute level
  v2 Fix 2 (head disagree) : qu >= 1.5 x |head1_m3s - head2_m3s| —
                              a free, input-dependent uncertainty
                              signal that widens automatically right
                              at flood peaks where the two heads diverge

Both scripts read v2_params from conformal_qhats.json so the trigger
constants stay in sync without manual duplication. If 4_conformal.py
is rerun with different RISE_TRIGGER_M3S etc., this script picks up
the new values automatically on its next run — no code change needed.

Because get_band() now needs a short rolling history of Qfor_mid to
compute rise_3h, this script keeps an in-memory rolling buffer during
replay mode, and reads the last RISE_WINDOW_HOURS rows of QM_lag
features for a live single-shot run (since Qfor_mid for prior hours
isn't otherwise available between runs).

WHAT IT NEEDS (must exist before first run)
────────────────────────────────────────────
  models/best_model.pt
  models/checkpoint.pt
  models/conformal_qhats.json   (run 4_conformal.py — v2 — first)
  data/processed/scaler_X.pkl
  data/processed/scaler_y1.pkl
  data/processed/QM_TRAIN_MAX.npy

WHAT IT READS
──────────────
  The .sbs.ts file TOPKAPI writes. Reads the tail so it always uses
  the most recent data. Minimum rows after preparation: SEQ_LEN + 21
  + RISE_WINDOW_HOURS, so the rate-of-rise signal has history too.

WHAT IT WRITES
───────────────
  outputs/forecast_latest.csv        PAB reads this (always overwritten)
  outputs/forecast_YYYYMMDD_HHMM.csv timestamped archive

  Columns:
    run_timestamp, forecast_time,
    Qfor_mid, Qfor_low, Qfor_high,
    QM_head1, QM_head2, blend_weight, head_disagreement, rise_3h,
    Q_TOPKAPI, band_width, flood_alert,
    coverage_pct, model_version

HOW TO RUN
───────────
  python 5_realtime_infer.py
  python 5_realtime_infer.py --live-ts path/to/latest.sbs.ts
  python 5_realtime_infer.py --live-ts data/raw/425.sbs.ts --replay
"""

import os, sys, json, pickle, argparse, configparser, csv, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PatchTST real-time inference — Casalecchio 425 (v2 bands)")
parser.add_argument("--live-ts", type=str, default=None)
parser.add_argument("--replay",  action="store_true")
parser.add_argument("--out-dir", type=str, default=None)
args = parser.parse_args()

# ── config ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_cfg  = configparser.ConfigParser()
_cfg.read(os.path.join(_HERE, "LSTM.ini"), encoding="utf-8-sig")
_pt   = _cfg["patchtst"]

RAW_TS    = args.live_ts if args.live_ts else os.path.join(_HERE, _pt["raw_ts"])
DATA_DIR  = os.path.join(_HERE, _pt["data_dir"])
MODEL_DIR = os.path.join(_HERE, _pt["model_dir"])
OUT_DIR   = args.out_dir if args.out_dir else os.path.join(_HERE, _pt["out_dir"])
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN         = _pt.getint("seq_len")
PATCH_LEN       = _pt.getint("patch_len")
STRIDE          = _pt.getint("stride")
BLEND_THRESHOLD = _pt.getfloat("blend_threshold")
BLEND_SCALE     = _pt.getfloat("blend_scale")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DROP_COLS   = ["Q", "Deep", "DeepSat", "Inf2Surf", "EnSnow", "Surf"]
LAG_HOURS   = [1, 2, 3, 7, 14, 21]
MISSING_VAL = -9999.0
FEATURE_COLS = [
    "Rain", "Prec", "Evap", "Snow", "Temp", "Etp", "Soil", "SoilSat",
    "Perco", "YSnow", "SWE",
    "QM_lag1", "QM_lag2", "QM_lag3", "QM_lag7", "QM_lag14", "QM_lag21",
]
FLOOD_THRESHOLD_M3S = 200.0   # confirm exact PAB level with supervisor


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — identical batched forward() to 2_train.py / 4_conformal.py
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


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ASSETS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading model and assets…")

_ckpt_path  = os.path.join(MODEL_DIR, "checkpoint.pt")
_scaler_path = os.path.join(DATA_DIR, "scaler_X.pkl")
_qhats_path = os.path.join(MODEL_DIR, "conformal_qhats.json")

_missing = []
if not os.path.exists(_ckpt_path):
    _missing.append(("checkpoint.pt", "2_train.py"))
if not os.path.exists(_scaler_path):
    _missing.append(("scaler_X.pkl", "1_prepare_data.py"))
if not os.path.exists(_qhats_path):
    _missing.append(("conformal_qhats.json", "4_conformal.py"))

if _missing:
    print(f"\n✗ Missing required file(s) — the pipeline hasn't been run far enough yet:")
    for fname, script in _missing:
        print(f"    {fname}  (created by {script})")
    print(f"  Run the full chain in order first: 1 -> 2 -> 3 -> 4, then this script.")
    sys.exit(1)

ckpt         = torch.load(_ckpt_path, map_location=DEVICE)
hp           = ckpt["hparams"]
model        = PatchTST(**hp).to(DEVICE)
best_weights = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
model.load_state_dict(best_weights)
model.eval()

with open(os.path.join(DATA_DIR, "scaler_X.pkl"),  "rb") as f: scaler_X  = pickle.load(f)
with open(os.path.join(DATA_DIR, "scaler_y1.pkl"), "rb") as f: scaler_y1 = pickle.load(f)
QM_TRAIN_MAX = float(np.load(os.path.join(DATA_DIR, "QM_TRAIN_MAX.npy")).item())

with open(os.path.join(MODEL_DIR, "conformal_qhats.json")) as f:
    qhats = json.load(f)
COVERAGE_PCT = (1 - qhats["alpha"]) * 100

# v2 params — read from qhats so 4_conformal.py is the single source of truth.
# Falls back to v1 defaults if an old (pre-v2) conformal_qhats.json is loaded.
v2 = qhats.get("v2_params", {})
PEAK_TRIGGER_M3S       = v2.get("peak_trigger_m3s", FLOOD_THRESHOLD_M3S * 0.30)
RISE_WINDOW_HOURS      = int(v2.get("rise_window_hours", 3))
RISE_TRIGGER_M3S       = v2.get("rise_trigger_m3s", 30.0)
FALL_TRIGGER_M3S       = v2.get("fall_trigger_m3s", 20.0)
FALL_SCALE_MULT        = v2.get("fall_scale_mult", 1.6)
HEAD_DISAGREEMENT_MULT = v2.get("head_disagreement_mult", 1.5)

if "v2_params" not in qhats:
    print("  ⚠ conformal_qhats.json has no v2_params — using v2 defaults. "
          "Rerun 4_conformal.py (v2) to calibrate properly.")

try:
    import subprocess
    _git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=_HERE, stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    _git = "unknown"

print(f"  Model    : val NSE {ckpt.get('val_nse', float('nan')):.4f}")
print(f"  Timestep : hourly  |  SEQ_LEN={SEQ_LEN}h = {SEQ_LEN/24:.1f} days lookback")
print(f"  Coverage : {COVERAGE_PCT:.0f}% conformal bands (v2: rate-of-rise + head-disagreement)")
print(f"  Device   : {DEVICE}")
print(f"  Version  : {_git}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def load_and_prepare(ts_path: str) -> pd.DataFrame:
    df = pd.read_csv(ts_path, sep=r'\s+', skipinitialspace=True)
    df.rename(columns={"YYYY": "year", "MM": "month", "DD": "day",
                       "HH": "hour", "mm": "minute"}, inplace=True)
    df["Time"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]])
    df = df.drop(columns=["year", "month", "day", "hour", "minute"])
    df = df.sort_values("Time").reset_index(drop=True)

    # drop duplicate timestamps — see 1_prepare_data.py for the full
    # explanation of why this matters (silent QM_lag misalignment otherwise)
    _n_before_dedup = len(df)
    df = df.drop_duplicates(subset="Time", keep="first").reset_index(drop=True)
    _n_dupes = _n_before_dedup - len(df)
    if _n_dupes > 0:
        print(f"⚠ WARNING: {_n_dupes} duplicate timestamp(s) found and "
             f"removed from live input file.")

    start_date = _pt.get("start_date", "2013-03-16")
    df = df[df["Time"] >= start_date].reset_index(drop=True)
    df = df[df["QM"] != MISSING_VAL].reset_index(drop=True)

    Q_series = df["Q"].copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    for lag in LAG_HOURS:
        df[f"QM_lag{lag}"] = df["QM"].shift(lag)
    df = df.dropna(subset=[f"QM_lag{lag}" for lag in LAG_HOURS]).reset_index(drop=True)
    df["Q"] = Q_series.iloc[df.index].values

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}\nAvailable: {list(df.columns)}")
    return df


def build_window(df: pd.DataFrame, end_idx: int) -> np.ndarray:
    start_idx = end_idx - SEQ_LEN + 1
    if start_idx < 0:
        raise ValueError(f"Not enough history: need {SEQ_LEN} rows, have {end_idx + 1}.")
    window    = df[FEATURE_COLS].iloc[start_idx : end_idx + 1].values
    window_sc = scaler_X.transform(window)
    return window_sc[np.newaxis, :, :].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE + BAND  (get_band() mirrors 4_conformal.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
def month_to_season(m: int) -> str:
    return {12:"DJF",1:"DJF",2:"DJF", 3:"MAM",4:"MAM",5:"MAM",
             6:"JJA",7:"JJA",8:"JJA", 9:"SON",10:"SON",11:"SON"}[m]


def get_band(season, q_mid, head_disagreement=None, rise_3h=None):
    """
    Identical logic to 4_conformal.py get_band().
    v1 floor   : q_mid >= PEAK_TRIGGER_M3S -> upgrade qu to peak quantile
    v2 Fix 1   : rise_3h >= RISE_TRIGGER_M3S -> upgrade qu (rising flood)
                 rise_3h <= -RISE_TRIGGER_M3S -> upgrade ql (falling recession)
    v2 Fix 2   : both qu and ql >= HEAD_DISAGREEMENT_MULT x head_disagreement
                 (direction-agnostic uncertainty signal widens both sides)
    v3 fix     : the falling-rate and head-disagreement terms above are what
                 fix the "lower bound too narrow on recession limb" issue —
                 see 4_conformal.py docstring for the full explanation.
    """
    sq = qhats["seasonal"].get(season, qhats["global"])
    ql = float(sq["q_lower"])
    qu = float(sq["q_upper"])

    # SON multiplier only applies while SON is still a calibration fallback
    # (see 4_conformal.py docstring). Self-disables once Fix A's full-year
    # 2023 val set gives SON real calibration samples.
    if season == "SON" and sq.get("fallback", False):
        qu = qu * 2.0
        ql = ql * 2.0

    if q_mid >= PEAK_TRIGGER_M3S:
        qu = max(qu, float(qhats["peak"]["q_upper"]))

    if rise_3h is not None:
        if rise_3h >= RISE_TRIGGER_M3S:
            qu = max(qu, float(qhats["peak"]["q_upper"]))
        if rise_3h <= -FALL_TRIGGER_M3S:
            ql = max(ql, float(qhats["peak"]["q_lower"]), abs(rise_3h) * FALL_SCALE_MULT)

    if head_disagreement is not None:
        qu = max(qu, head_disagreement * HEAD_DISAGREEMENT_MULT)
        ql = max(ql, head_disagreement * HEAD_DISAGREEMENT_MULT)

    return ql, qu


def infer_single(window_sc: np.ndarray, forecast_time: pd.Timestamp,
                 q_mid_history: list) -> dict:
    """
    q_mid_history : list of the last RISE_WINDOW_HOURS Qfor_mid values
                    (oldest first). Pass [] if unavailable (cold start —
                    rise_3h signal is skipped for the first few forecasts).
    """
    x = torch.tensor(window_sc, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        p1, p2 = model(x)

    pred1_sc  = p1.cpu().numpy()
    pred1_m3s = float(np.clip(np.expm1(scaler_y1.inverse_transform(pred1_sc).ravel()), 0, None)[0])
    pred2_norm = p2.cpu().numpy()
    pred2_m3s  = float(np.clip(pred2_norm.ravel()[0] * QM_TRAIN_MAX, 0, None))

    blend_w = float(1.0 / (1.0 + np.exp(-(pred2_m3s - BLEND_THRESHOLD) / BLEND_SCALE)))
    q_mid   = float((1.0 - blend_w) * pred1_m3s + blend_w * pred2_m3s)

    head_disagreement = abs(pred1_m3s - pred2_m3s)
    rise_3h = (q_mid - q_mid_history[0]) if len(q_mid_history) >= RISE_WINDOW_HOURS else None

    season = month_to_season(forecast_time.month)
    q_lower, q_upper = get_band(season, q_mid,
                                head_disagreement=head_disagreement,
                                rise_3h=rise_3h)
    q_low  = max(0.0, q_mid - q_lower)
    q_high = q_mid + q_upper

    return {
        "Qfor_mid": round(q_mid, 3), "Qfor_low": round(q_low, 3), "Qfor_high": round(q_high, 3),
        "QM_head1": round(pred1_m3s, 3), "QM_head2": round(pred2_m3s, 3),
        "blend_weight": round(blend_w, 4),
        "head_disagreement": round(head_disagreement, 3),
        "rise_3h": round(rise_3h, 3) if rise_3h is not None else None,
        "flood_alert": int(q_high >= FLOOD_THRESHOLD_M3S),
        "band_width": round(q_high - q_low, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
print(f"\nRun timestamp : {run_ts}")
print(f"Input file    : {RAW_TS}")
print(f"Mode          : {'REPLAY (full file)' if args.replay else 'LIVE (latest window)'}")

df = load_and_prepare(RAW_TS)
print(f"Rows prepared : {len(df)}  "
      f"({df['Time'].iloc[0].strftime('%Y-%m-%d %H:%M')} → "
      f"{df['Time'].iloc[-1].strftime('%Y-%m-%d %H:%M')})")

ROWS_NEEDED = SEQ_LEN + 21
if len(df) < ROWS_NEEDED:
    print(f"\n✗ Not enough data: need {ROWS_NEEDED} hourly rows, have {len(df)}.")
    sys.exit(1)

OUTPUT_COLS = [
    "run_timestamp", "forecast_time", "Qfor_mid", "Qfor_low", "Qfor_high",
    "QM_head1", "QM_head2", "blend_weight", "head_disagreement", "rise_3h",
    "Q_TOPKAPI", "band_width", "flood_alert", "coverage_pct", "model_version",
]


def build_row(i: int, q_mid_history: list) -> dict:
    t         = df["Time"].iloc[i]
    q_topkapi = float(df["Q"].iloc[i]) if "Q" in df.columns else float("nan")
    result    = infer_single(build_window(df, i), t, q_mid_history)
    return {
        "run_timestamp": run_ts, "forecast_time": t.strftime("%Y-%m-%d %H:%M"),
        "Qfor_mid": result["Qfor_mid"], "Qfor_low": result["Qfor_low"], "Qfor_high": result["Qfor_high"],
        "QM_head1": result["QM_head1"], "QM_head2": result["QM_head2"],
        "blend_weight": result["blend_weight"],
        "head_disagreement": result["head_disagreement"], "rise_3h": result["rise_3h"],
        "Q_TOPKAPI": round(q_topkapi, 3), "band_width": result["band_width"],
        "flood_alert": result["flood_alert"], "coverage_pct": COVERAGE_PCT, "model_version": _git,
    }


if not args.replay:
    # ── LIVE MODE ─────────────────────────────────────────────────────────────
    # cold start: rise_3h needs RISE_WINDOW_HOURS prior Qfor_mid values, which
    # we don't have between separate script runs. Compute them quickly here
    # by running inference on the RISE_WINDOW_HOURS prior windows too.
    i = len(df) - 1
    q_mid_history = []
    for j in range(i - RISE_WINDOW_HOURS, i):
        if j < SEQ_LEN - 1:
            continue
        t_j = df["Time"].iloc[j]
        r_j = infer_single(build_window(df, j), t_j, [])
        q_mid_history.append(r_j["Qfor_mid"])

    row = build_row(i, q_mid_history)

    print(f"\n{'═'*58}")
    print(f"  FORECAST  —  {row['forecast_time']}")
    print(f"{'═'*58}")
    print(f"  Qfor_mid    : {row['Qfor_mid']:10.2f}  m³/s")
    print(f"  Qfor_low    : {row['Qfor_low']:10.2f}  m³/s")
    print(f"  Qfor_high   : {row['Qfor_high']:10.2f}  m³/s")
    print(f"  Q_TOPKAPI   : {row['Q_TOPKAPI']:10.2f}  m³/s")
    print(f"  Head disagree: {row['head_disagreement']:9.2f}  m³/s")
    print(f"  Rise (3h)   : {row['rise_3h']}")
    print(f"  Band width  : {row['band_width']:10.2f}  m³/s")
    if row["flood_alert"]:
        print(f"\n  *** FLOOD ALERT *** Qfor_high {row['Qfor_high']:.0f} >= {FLOOD_THRESHOLD_M3S:.0f} m³/s")
    print(f"{'═'*58}")

    latest_path = os.path.join(OUT_DIR, "forecast_latest.csv")
    pd.DataFrame([row], columns=OUTPUT_COLS).to_csv(latest_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    ts_tag    = datetime.now().strftime("%Y%m%d_%H%M")
    arch_path = os.path.join(OUT_DIR, f"forecast_{ts_tag}.csv")
    pd.DataFrame([row], columns=OUTPUT_COLS).to_csv(arch_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\n✓ forecast_latest.csv  →  {latest_path}")
    print(f"✓ forecast_{ts_tag}.csv  →  {arch_path}")

else:
    # ── REPLAY MODE ───────────────────────────────────────────────────────────
    start_i = SEQ_LEN - 1
    total   = len(df) - start_i
    rows    = []
    q_mid_buffer = []   # rolling buffer, length capped at RISE_WINDOW_HOURS

    print(f"\nProcessing {total} hourly windows ({total/24:.0f} days)…")
    import time as _time
    t0 = _time.perf_counter()

    for i in range(start_i, len(df)):
        row = build_row(i, q_mid_buffer[-RISE_WINDOW_HOURS:])
        rows.append(row)
        q_mid_buffer.append(row["Qfor_mid"])
        if len(q_mid_buffer) > RISE_WINDOW_HOURS:
            q_mid_buffer.pop(0)

        if (i - start_i + 1) % 2000 == 0 or i == len(df) - 1:
            elapsed = _time.perf_counter() - t0
            rate    = (i - start_i + 1) / elapsed
            print(f"  {i - start_i + 1:6d}/{total}  ({rate:.0f} w/s)  "
                  f"{row['forecast_time']}  mid={row['Qfor_mid']:.1f}  alert={row['flood_alert']}")

    elapsed_total = _time.perf_counter() - t0
    print(f"\nCompleted {total} windows in {elapsed_total:.1f}s  ({total/elapsed_total:.0f} windows/s)")

    replay_df   = pd.DataFrame(rows, columns=OUTPUT_COLS)
    replay_path = os.path.join(OUT_DIR, "replay_output.csv")
    replay_df.to_csv(replay_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    if "QM" in df.columns:
        obs   = df["QM"].iloc[start_i:].values
        pred  = replay_df["Qfor_mid"].values
        low   = replay_df["Qfor_low"].values
        high  = replay_df["Qfor_high"].values

        nse   = float(1 - np.sum((pred-obs)**2) / np.sum((obs - obs.mean())**2))
        pbias = float(100 * (pred.sum() - obs.sum()) / obs.sum())
        cov   = float(((obs >= low) & (obs <= high)).mean() * 100)

        peak_mask      = obs >= FLOOD_THRESHOLD_M3S
        n_peaks        = int(peak_mask.sum())
        cov_peak       = float(((obs[peak_mask] >= low[peak_mask]) &
                                (obs[peak_mask] <= high[peak_mask])).mean() * 100) if n_peaks > 0 else float("nan")
        correct_alerts = int(((replay_df["flood_alert"].values == 1) & peak_mask).sum())
        false_alarms   = int(((replay_df["flood_alert"].values == 1) & ~peak_mask).sum())

        print(f"\n{'═'*58}")
        print(f"  REPLAY DIAGNOSTICS — Casalecchio 425 (v2 bands)")
        print(f"{'═'*58}")
        print(f"  NSE              : {nse:+.4f}")
        print(f"  PBIAS            : {pbias:+.2f}%")
        print(f"  Band coverage    : {cov:.1f}%  (target {COVERAGE_PCT:.0f}%)")
        print(f"  Peak coverage    : {cov_peak:.1f}%  (n={n_peaks} events >{FLOOD_THRESHOLD_M3S:.0f} m³/s)")
        print(f"  Correct alerts   : {correct_alerts}/{n_peaks}  ({100*correct_alerts/max(n_peaks,1):.0f}%)")
        print(f"  False alarms     : {false_alarms}")
        print(f"{'═'*58}")

        years = pd.to_datetime(replay_df["forecast_time"]).dt.year.values
        print(f"\n  Per-year NSE and coverage:")
        for yr in sorted(set(years)):
            m  = years == yr
            o  = obs[m]; p = pred[m]
            ns = float(1 - np.sum((p-o)**2) / np.sum((o-o.mean())**2))
            cv = float(((o >= low[m]) & (o <= high[m])).mean() * 100)
            n_fl = int((o >= FLOOD_THRESHOLD_M3S).sum())
            print(f"    {yr}  NSE={ns:+.3f}  coverage={cv:.1f}%  flood_hours={n_fl}")

    print(f"\n✓ replay_output.csv saved  ({len(replay_df)} rows → {replay_path})")
    print(f"✓ Output directory: {OUT_DIR}")
