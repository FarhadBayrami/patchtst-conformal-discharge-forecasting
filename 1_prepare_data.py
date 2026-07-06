"""
Snippet 1 — Data Preparation  (v6 — Casalecchio di Reno, Fix A split)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: 425.sbs.ts  (station 425, Casalecchio di Reno)
Format : space-delimited fixed-width, one header row.
         Columns: YYYY MM DD HH mm  QM  Q  Rain Prec Evap Snow
                  Temp Etp Soil SoilSat Perco Surf YSnow EnSnow
                  SWE Deep DeepSat Inf2Surf

CHANGES vs v5 — FIX A: CALIBRATION SET NOW CONTAINS REAL AUTUMN EXTREMES
──────────────────────────────────────────────────────────────────────────
PROBLEM THIS FIXES
  The conformal uncertainty band in 4_conformal.py was calibrated on the
  val set residuals. The old split used 2024 Jan-Jun as val — a period
  with NO September/October/November data at all (SON n=0 every run).
  Apennine rivers produce their largest floods in autumn (this dataset's
  worst event, 910.6 m³/s, hit on 9 October 2024). Because the calibration
  set never saw an autumn flood, it never saw an error of that scale, so
  the conformal quantile mathematically could not be wide enough to
  bound it — no amount of trigger logic in 4_conformal.py can manufacture
  calibration information that was never collected.

THE FIX
  Split changed from (train: 2014-2023, val: 2024 H1, test: 2024 H2) to:
    train : 2014-2022   (9 years, still ample for model fitting)
    val   : 2023         (FULL YEAR — includes a real SON, used for both
                           early stopping in 2_train.py AND conformal
                           calibration in 4_conformal.py)
    test  : 2024         (FULL YEAR — held out, never seen by model or
                           calibration, this is what gets reported)

  This means every season — including SON — now has real calibration
  residuals to build a quantile from. The conformal band will widen
  itself naturally to whatever margin autumn floods actually require,
  instead of relying on the 2x manual safety multiplier that previously
  patched over the missing SON data.

WHAT TO EXPECT WHEN YOU RERUN THE FULL CHAIN
  - 1_prepare_data.py (this file): val set jumps from ~4300 rows to
    ~8700 rows (full year vs half year). Check the printed n>50 count
    for 2023 — if 2023 had a notably quiet year, the calibration may
    still be conservative; check the val summary line.
  - 2_train.py: must be rerun — Y_val.npy changed shape and date range,
    so early stopping now monitors a different (better) validation set.
  - 3_evaluate.py: test set is now 2024 (whole year, not just H2) —
    expect different NSE/PBIAS numbers than your previous H2-only run;
    this is a MORE representative number, not a worse one.
  - 4_conformal.py: rerun last. SON should now show n > 20 with real
    quantiles instead of "fallback to global". The manual SON 2x
    multiplier in get_band() becomes redundant once real SON data
    exists — you can remove it once you confirm n_SON > MIN_SEASONAL.
  - 5_realtime_infer.py: no code changes needed, it reads whatever
    4_conformal.py produces.

EVERYTHING ELSE (features, scalers, lags, augmentation) is identical to v5.
"""

import os, pickle, configparser
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── config ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_cfg  = configparser.ConfigParser()
_cfg.read(os.path.join(_HERE, "LSTM.ini"), encoding="utf-8-sig")
_pt   = _cfg["patchtst"]

RAW_TS  = os.path.join(_HERE, _pt["raw_ts"])
OUT_DIR = os.path.join(_HERE, _pt["data_dir"])
os.makedirs(OUT_DIR, exist_ok=True)

# ── window config (keep in sync with 2_train.py) ──────────────────────────────
SEQ_LEN   = _pt.getint("seq_len")
PATCH_LEN = _pt.getint("patch_len")
STRIDE    = _pt.getint("stride")

# ── 1. load & parse ───────────────────────────────────────────────────────────
# .sbs.ts is space-delimited; first column header has a leading space.
# sep=r'\s+' with skipinitialspace handles both.
df = pd.read_csv(RAW_TS, sep=r'\s+', skipinitialspace=True)

df.rename(columns={"YYYY": "year", "MM": "month", "DD": "day",
                   "HH": "hour", "mm": "minute"}, inplace=True)

df["Time"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]])
df = df.drop(columns=["year", "month", "day", "hour", "minute"])
df = df.sort_values("Time").reset_index(drop=True)

# ── 1b. drop duplicate timestamps ────────────────────────────────────────────
# Real sensor/logging systems occasionally emit the same timestamp twice
# (a duplicate row, not a second real reading). If left in, this silently
# shifts every QM_lag feature downstream by one row relative to true
# elapsed time, since .shift() operates on row position, not wall-clock
# time. Keep the first occurrence and warn loudly if any are found.
_n_before_dedup = len(df)
df = df.drop_duplicates(subset="Time", keep="first").reset_index(drop=True)
_n_dupes = _n_before_dedup - len(df)
if _n_dupes > 0:
    print(f"⚠ WARNING: {_n_dupes} duplicate timestamp(s) found and removed "
         f"(kept first occurrence). This usually indicates a data export "
         f"or sensor logging issue worth investigating in the raw .sbs.ts file.")

print(f"Raw rows loaded      : {len(df)}")
print(f"Date range           : {df['Time'].iloc[0]} → {df['Time'].iloc[-1]}")

# ── 2. filter date ────────────────────────────────────────────────────────────
df = df[df["Time"] >= _pt["start_date"]].reset_index(drop=True)
print(f"After date filter    : {len(df)} rows")

# ── 3. drop missing QM rows ───────────────────────────────────────────────────
df = df[df["QM"] != -9999.0].reset_index(drop=True)
print(f"After QM clean       : {len(df)} rows")

# ── 4. keep Q (TOPKAPI) aside BEFORE dropping it from features ───────────────
df_Q = df[["Time", "Q"]].copy()

# ── 5. drop unwanted feature columns ─────────────────────────────────────────
df = df.drop(columns=["Q", "Deep", "DeepSat", "Inf2Surf", "EnSnow", "Surf"])

# ── 6. add lagged QM features ─────────────────────────────────────────────────
for lag in [1, 2, 3, 7, 14, 21]:
    df[f"QM_lag{lag}"] = df["QM"].shift(lag)
df = df.dropna().reset_index(drop=True)
df_Q = df_Q[df_Q["Time"].isin(df["Time"])].reset_index(drop=True)
print(f"After lag creation   : {len(df)} rows  (df_Q rows: {len(df_Q)})")

# ── 7. define features and targets ───────────────────────────────────────────
EXCLUDE      = {"Time", "QM"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── 8. YEAR-BASED SPLIT — FIX A ───────────────────────────────────────────────
# train : 2014-2022 (9 years)
# val   : 2023 FULL YEAR (was 2024 Jan-Jun) — now contains a real autumn
#         (SON), so conformal calibration in 4_conformal.py will see real
#         large-flood residuals instead of falling back to a 2x multiplier.
# test  : 2024 FULL YEAR (was 2024 Jul-Dec only) — more representative
#         final evaluation; includes the 9 Oct 2024 / 910.6 m³/s event
#         used to diagnose the original band-coverage problem.
# Temporal order is strictly preserved — no future leakage.
train_mask = df["Time"].dt.year <= 2022
val_mask   = df["Time"].dt.year == 2023
test_mask  = df["Time"].dt.year == 2024

df_train = df[train_mask].copy().reset_index(drop=True)
df_val   = df[val_mask].copy().reset_index(drop=True)
df_test  = df[test_mask].copy().reset_index(drop=True)

Q_train = df_Q[train_mask.values].reset_index(drop=True)
Q_val   = df_Q[val_mask.values].reset_index(drop=True)
Q_test  = df_Q[test_mask.values].reset_index(drop=True)

for name, d in [("Train", df_train), ("Val  ", df_val), ("Test ", df_test)]:
    if len(d) == 0:
        print(f"  {name}: 0 rows (no data for this split)")
        continue
    print(f"  {name}: {len(d):5d} rows  "
          f"{d['Time'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{d['Time'].iloc[-1].strftime('%Y-%m-%d')}  "
          f"QM mean={d['QM'].mean():.1f}  max={d['QM'].max():.1f}  "
          f"n>50={(d['QM']>50).sum()}  n>200={(d['QM']>200).sum()}")

# seasonal sanity check for the val set specifically — confirms Fix A worked
print(f"\nVal set seasonal breakdown (this must show all 4 seasons with n>0):")
_season_map = {12:"DJF",1:"DJF",2:"DJF", 3:"MAM",4:"MAM",5:"MAM",
               6:"JJA",7:"JJA",8:"JJA", 9:"SON",10:"SON",11:"SON"}
if len(df_val) > 0:
    _val_seasons = df_val["Time"].dt.month.map(_season_map)
    for s in ["DJF","MAM","JJA","SON"]:
        _n = (_val_seasons == s).sum()
        _max_qm = df_val.loc[_val_seasons == s, "QM"].max() if _n > 0 else float("nan")
        flag = "✓" if _n >= 20 else "⚠ still thin"
        print(f"  {s}  n={_n:4d}  max QM={_max_qm:.1f}  {flag}")

# ── 9. fit scalers on TRAIN only ─────────────────────────────────────────────
QM_TRAIN_MAX = df_train["QM"].max()
print(f"\nQM_TRAIN_MAX : {QM_TRAIN_MAX:.2f}")

scaler_X  = StandardScaler()
scaler_y1 = StandardScaler()

for d in [df_train, df_val, df_test, df]:
    d["QM_log"]  = np.log1p(d["QM"])
    d["QM_norm"] = d["QM"] / QM_TRAIN_MAX

X_tr_sc  = scaler_X.fit_transform(df_train[FEATURE_COLS])
X_va_sc  = scaler_X.transform(df_val[FEATURE_COLS])  if len(df_val)  > 0 else np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)
X_te_sc  = scaler_X.transform(df_test[FEATURE_COLS]) if len(df_test) > 0 else np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)
X_all_sc = scaler_X.transform(df[FEATURE_COLS])

y1_tr  = scaler_y1.fit_transform(df_train[["QM_log"]])
y1_va  = scaler_y1.transform(df_val[["QM_log"]])  if len(df_val)  > 0 else np.zeros((0, 1), dtype=np.float32)
y1_te  = scaler_y1.transform(df_test[["QM_log"]]) if len(df_test) > 0 else np.zeros((0, 1), dtype=np.float32)
y1_all = scaler_y1.transform(df[["QM_log"]])

y2_tr  = df_train[["QM_norm"]].values.astype(np.float32)
y2_va  = df_val[["QM_norm"]].values.astype(np.float32)  if len(df_val)  > 0 else np.zeros((0, 1), dtype=np.float32)
y2_te  = df_test[["QM_norm"]].values.astype(np.float32) if len(df_test) > 0 else np.zeros((0, 1), dtype=np.float32)
y2_all = df[["QM_norm"]].values.astype(np.float32)

Y_tr  = np.hstack([y1_tr,  y2_tr ]).astype(np.float32)
Y_va  = np.hstack([y1_va,  y2_va ]).astype(np.float32)
Y_te  = np.hstack([y1_te,  y2_te ]).astype(np.float32)
Y_all = np.hstack([y1_all, y2_all]).astype(np.float32)

# ── 10. sliding-window sequences ─────────────────────────────────────────────
def make_sequences(X, Y):
    Xs, Ys = [], []
    for i in range(len(X) - SEQ_LEN):
        Xs.append(X[i : i + SEQ_LEN])
        Ys.append(Y[i + SEQ_LEN])
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

X_tr,  Y_tr_seq  = make_sequences(X_tr_sc,  Y_tr)

# ── 10b. flood sequence augmentation ─────────────────────────────────────────
# Physically repeat flood windows so peak events appear more during training.
# threshold ~ QM > 50 m³/s; repeat=8 → floods go from ~3% to ~20% of data.
_flood_thresh = 50.0 / QM_TRAIN_MAX
_flood_mask   = Y_tr_seq[:, 1] > _flood_thresh
_n_floods     = _flood_mask.sum()
X_tr_aug = np.concatenate([X_tr] + [X_tr[_flood_mask]] * 8, axis=0)
Y_tr_aug = np.concatenate([Y_tr_seq] + [Y_tr_seq[_flood_mask]] * 8, axis=0)
_idx     = np.random.default_rng(42).permutation(len(X_tr_aug))
X_tr     = X_tr_aug[_idx]
Y_tr_seq = Y_tr_aug[_idx]
print(f"\nFlood augmentation   : {_n_floods} flood windows x8 → ",
      f"train size {len(X_tr)} (was {len(X_tr_aug) - _n_floods*8})")

X_va,  Y_va_seq  = make_sequences(X_va_sc,  Y_va)  if len(X_va_sc)  > SEQ_LEN else (np.zeros((0, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32), np.zeros((0, 2), dtype=np.float32))
X_te,  Y_te_seq  = make_sequences(X_te_sc,  Y_te)  if len(X_te_sc)  > SEQ_LEN else (np.zeros((0, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32), np.zeros((0, 2), dtype=np.float32))
X_all, Y_all_seq = make_sequences(X_all_sc, Y_all)

n_patches = (SEQ_LEN - PATCH_LEN) // STRIDE + 1
print(f"\nX_train : {X_tr.shape}   Y_train : {Y_tr_seq.shape}")
print(f"X_val   : {X_va.shape}   Y_val   : {Y_va_seq.shape}")
print(f"X_test  : {X_te.shape}   Y_test  : {Y_te_seq.shape}")
print(f"X_all   : {X_all.shape}  Y_all   : {Y_all_seq.shape}")
print(f"Patches per sample: {n_patches}")

# ── 11. save arrays ───────────────────────────────────────────────────────────
for tag, arr in [("X_train", X_tr),  ("Y_train", Y_tr_seq),
                 ("X_val",   X_va),  ("Y_val",   Y_va_seq),
                 ("X_test",  X_te),  ("Y_test",  Y_te_seq),
                 ("X_all",   X_all), ("Y_all",   Y_all_seq)]:
    np.save(os.path.join(OUT_DIR, f"{tag}.npy"), arr)

with open(os.path.join(OUT_DIR, "scaler_X.pkl"),  "wb") as f: pickle.dump(scaler_X,  f)
with open(os.path.join(OUT_DIR, "scaler_y1.pkl"), "wb") as f: pickle.dump(scaler_y1, f)
np.save(os.path.join(OUT_DIR, "QM_TRAIN_MAX.npy"), np.array([QM_TRAIN_MAX]))

# ── 12. save reference CSVs (times + QM observed + Q TOPKAPI) ────────────────
# Offset by SEQ_LEN because each sequence target corresponds to timestep SEQ_LEN.

# val_reference.csv — required by 4_conformal.py for calibration
if len(df_val) > SEQ_LEN:
    val_ref = pd.DataFrame({
        "Time" : df_val["Time"].iloc[SEQ_LEN:].values,
        "QM"   : df_val["QM"].iloc[SEQ_LEN:].values,
        "Q"    : Q_val["Q"].iloc[SEQ_LEN:].values,
    })
    val_ref.to_csv(os.path.join(OUT_DIR, "val_reference.csv"), index=False)
    print(f"\n✓ val_reference.csv   : {len(val_ref)} rows")
else:
    print("\n⚠ val split too small — val_reference.csv not saved")

if len(df_test) > SEQ_LEN:
    test_ref = pd.DataFrame({
        "Time" : df_test["Time"].iloc[SEQ_LEN:].values,
        "QM"   : df_test["QM"].iloc[SEQ_LEN:].values,
        "Q"    : Q_test["Q"].iloc[SEQ_LEN:].values,
    })
    test_ref.to_csv(os.path.join(OUT_DIR, "test_reference.csv"), index=False)
    print(f"✓ test_reference.csv  : {len(test_ref)} rows")
else:
    print("⚠ test split too small — test_reference.csv not saved")

all_ref = pd.DataFrame({
    "Time" : df["Time"].iloc[SEQ_LEN:].values,
    "QM"   : df["QM"].iloc[SEQ_LEN:].values,
    "Q"    : df_Q["Q"].iloc[SEQ_LEN:].values,
})
all_ref.to_csv(os.path.join(OUT_DIR, "all_reference.csv"), index=False)

print(f"✓ all_reference.csv   : {len(all_ref)} rows")
print(f"✓ All arrays saved to {OUT_DIR}")
