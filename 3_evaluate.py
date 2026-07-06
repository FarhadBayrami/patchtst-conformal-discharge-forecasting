"""

1. TOPKAPI Q added to all time-series and flow-duration plots
   so you can directly compare:
     • QM observed (gauge)
     • QM predicted (this model)
     • Q TOPKAPI   (hydrological model output)

2. WHOLE-DATASET inference (2014-2024)
   The model runs on every sample in X_all (the full post-2014
   period). This produces predictions_all.csv and a separate
   set of plots suffixed _ALL so test-set and full-period plots
   are kept separate.

3. Metrics are computed for BOTH the test set AND the full period.

Output files
─────────────
  outputs/predictions_test.csv       — test set (2024)
  outputs/predictions_all.csv        — full period (2014-2024)
  outputs/plots/TEST_01_timeseries.png … TEST_07_head_comparison.png
  outputs/plots/ALL_01_timeseries.png  … ALL_06_flow_duration_curve.png
"""

import os, sys, json, pickle, configparser, csv
import numpy as np, pandas as pd, torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# ── colours — consistent across all plots ────────────────────────────────────
C_OBS     = "#1565C0"   # blue   — QM observed
C_PRED    = "#C62828"   # red    — QM predicted
C_TOPKAPI = "#2E7D32"   # green  — Q TOPKAPI

# ── model definition (identical to 2_train.py) ───────────────────────────────
class PatchTST(nn.Module):
    def __init__(self, n_features, seq_len, patch_len, stride,
                 d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.n_features  = n_features
        self.patch_len   = patch_len
        self.stride      = stride
        self.n_patches   = (seq_len - patch_len) // stride + 1
        self.patch_embed   = nn.Linear(patch_len, d_model)
        self.pos_embed     = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm  = nn.LayerNorm(d_model)
        trunk_in   = n_features * self.n_patches * d_model
        self.trunk = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(trunk_in),
            nn.Linear(trunk_in, d_model * 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.head1 = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(d_model, 1),
        )
        self.head2 = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(d_model, 1), nn.Softplus(),
        )

    def forward(self, x):
        # Batched channel-independent patching — mathematically identical to
        # the original per-channel loop (verified: same weights produce
        # bit-identical output), just faster. Standardized to match
        # 2_train.py / 4_conformal.py / 5_realtime_infer.py so all five
        # scripts use one canonical model definition.
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, T)
        p = x.unfold(1, self.patch_len, self.stride)
        p = self.patch_embed(p) + self.pos_embed
        p = self.input_dropout(p)
        p = self.norm(self.encoder(p))
        out   = p.reshape(B, C, self.n_patches, -1)
        trunk = self.trunk(out)
        return self.head1(trunk), self.head2(trunk)

# ── load checkpoint ───────────────────────────────────────────────────────────
# checkpoint.pt  — saved at the LAST epoch; contains hparams + metadata
# best_model.pt  — saved at the BEST NSE epoch; contains only the state dict
# We always evaluate the best model, not the last one.
_ckpt_path = os.path.join(MODEL_DIR, "checkpoint.pt")
if not os.path.exists(_ckpt_path):
    print(f"\n✗ No trained model found at {_ckpt_path}")
    print(f"  Run 2_train.py first — it must finish (or early-stop) to")
    print(f"  produce checkpoint.pt and best_model.pt before this script can run.")
    sys.exit(1)

ckpt  = torch.load(_ckpt_path, map_location=DEVICE)
hp    = ckpt["hparams"]
model = PatchTST(**hp).to(DEVICE)

best_weights = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)
model.load_state_dict(best_weights)
model.eval()

# checkpoint.pt stores val_nse from v5 onwards, val_loss from v4 — handle both
if "val_nse" in ckpt:
    print(f"Loaded best_model.pt  (best val NSE : {ckpt['val_nse']:.4f})")
else:
    print(f"Loaded best_model.pt  (checkpoint val loss: {ckpt['val_loss']:.5f})")

# ── load scalers ──────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "scaler_y1.pkl"), "rb") as f:
    scaler_y1 = pickle.load(f)
QM_TRAIN_MAX = float(np.load(os.path.join(DATA_DIR, "QM_TRAIN_MAX.npy")).item())

# ── helper: run batched inference and return blended m3/s predictions ─────────
def run_inference(X_tensor):
    p1_list, p2_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 256):
            xb = X_tensor[i:i+256].to(DEVICE)
            o1, o2 = model(xb)
            p1_list.append(o1.cpu().numpy())
            p2_list.append(o2.cpu().numpy())
    pred1_sc   = np.vstack(p1_list)
    pred2_norm = np.vstack(p2_list)
    pred1_m3s  = np.clip(np.expm1(scaler_y1.inverse_transform(pred1_sc).ravel()), 0, None)
    pred2_m3s  = np.clip(pred2_norm.ravel() * QM_TRAIN_MAX, 0, None)
    blend_w    = 1 / (1 + np.exp(-(pred2_m3s - BLEND_THRESHOLD) / BLEND_SCALE))
    blended    = (1 - blend_w) * pred1_m3s + blend_w * pred2_m3s
    return blended, pred1_m3s, pred2_m3s, blend_w

# ── helper: compute metrics ───────────────────────────────────────────────────
def compute_metrics(obs, pred, label=""):
    res   = pred - obs
    rmse  = np.sqrt(np.mean(res**2))
    mae   = np.mean(np.abs(res))
    r2    = r2_score(obs, pred)
    nse   = 1 - np.sum(res**2) / np.sum((obs - obs.mean())**2)
    pbias = 100 * (pred.sum() - obs.sum()) / obs.sum()
    thr90 = np.percentile(obs, 90)
    pk    = obs >= thr90
    rmse_pk = np.sqrt(np.mean(res[pk]**2))
    nse_pk  = 1 - np.sum(res[pk]**2) / np.sum((obs[pk] - obs[pk].mean())**2)
    print(f"\n{'═'*52}")
    if label: print(f"  {label}")
    print(f"  RMSE       : {rmse:8.3f}  m³/s")
    print(f"  MAE        : {mae:8.3f}  m³/s")
    print(f"  R²         : {r2:8.4f}")
    print(f"  NSE        : {nse:8.4f}   (>0.6=good  >0.8=excellent)")
    print(f"  PBIAS      : {pbias:+8.2f} %")
    print(f"  RMSE_peak  : {rmse_pk:8.3f}  m³/s  (top 10%, >{thr90:.1f})")
    print(f"  NSE_peak   : {nse_pk:8.4f}")
    print(f"{'═'*52}")
    return dict(rmse=rmse, mae=mae, r2=r2, nse=nse, pbias=pbias,
                rmse_pk=rmse_pk, nse_pk=nse_pk)

# ── helper: save figure ───────────────────────────────────────────────────────
TFMT = mdates.DateFormatter("%Y-%m")
def save_fig(fig, fname):
    fig.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — TEST SET (2024)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "▓"*52)
print("  SECTION A — TEST SET  (2024)")
print("▓"*52)

X_test  = torch.tensor(np.load(os.path.join(DATA_DIR, "X_test.npy")),  dtype=torch.float32)
ref_test = pd.read_csv(os.path.join(DATA_DIR, "test_reference.csv"))
ref_test["Time"] = pd.to_datetime(ref_test["Time"])

times_te = ref_test["Time"].values
obs_te   = ref_test["QM"].values
Q_te     = ref_test["Q"].values     # TOPKAPI

pred_te, pred1_te, pred2_te, blend_te = run_inference(X_test)
res_te = pred_te - obs_te
m_te   = compute_metrics(obs_te, pred_te, "TEST SET (2024)")

# save test CSV
# quoting=csv.QUOTE_NONNUMERIC forces all strings (incl. datetime) to be
# wrapped in double-quotes in the file → Excel reads it as text, not a date
_te_dt = pd.to_datetime(times_te)
pd.DataFrame({
    "datetime":     _te_dt.strftime("%Y-%m-%d %H:%M"),
    "QM_observed":  np.round(obs_te,    3),
    "QM_predicted": np.round(pred_te,   3),
    "Q_TOPKAPI":    np.round(Q_te,      3),
    "QM_head1":     np.round(pred1_te,  3),
    "QM_head2":     np.round(pred2_te,  3),
    "blend_weight": np.round(blend_te,  3),
    "residual":     np.round(res_te,    3),
}).to_csv(os.path.join(OUT_DIR, "predictions_test.csv"), index=False,
          quoting=csv.QUOTE_NONNUMERIC)
print("✓ predictions_test.csv saved")

times_te_dt = pd.to_datetime(times_te)

# TEST plot 1 — time series with TOPKAPI
fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(times_te_dt, obs_te,  label="QM observed",    color=C_OBS,     lw=1.0)
ax.plot(times_te_dt, pred_te, label="QM predicted",   color=C_PRED,    lw=0.9, alpha=0.9)
ax.plot(times_te_dt, Q_te,    label="Q TOPKAPI",      color=C_TOPKAPI, lw=0.9, alpha=0.85, ls="--")
ax.xaxis.set_major_formatter(TFMT)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
ax.set_xlabel("Date"); ax.set_ylabel("Flow  [m³/s]")
ax.set_title(f"TEST SET (2024) — QM observed / QM predicted / Q TOPKAPI  "
             f"(NSE={m_te['nse']:.3f}  PBIAS={m_te['pbias']:+.1f}%)")
ax.legend(); fig.tight_layout()
save_fig(fig, "TEST_01_timeseries.png")

# TEST plot 2 — scatter QM pred vs QM obs
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(obs_te, pred_te, s=8, alpha=0.4, color="#4527A0", label="QM pred vs obs")
ax.scatter(obs_te, Q_te,    s=8, alpha=0.3, color=C_TOPKAPI,  label="Q TOPKAPI vs obs", marker="^")
lim = max(obs_te.max(), pred_te.max(), Q_te.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("QM observed  [m³/s]"); ax.set_ylabel("Predicted / TOPKAPI  [m³/s]")
ax.set_title(f"Scatter — Test Set  |  R²={m_te['r2']:.3f}  NSE={m_te['nse']:.3f}")
ax.legend(fontsize=8); fig.tight_layout()
save_fig(fig, "TEST_02_scatter.png")

# TEST plot 3 — residuals over time
fig, ax = plt.subplots(figsize=(15, 3))
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.fill_between(times_te_dt, res_te, alpha=0.55, color="#E65100")
ax.xaxis.set_major_formatter(TFMT)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
ax.set_xlabel("Date"); ax.set_ylabel("Residual  [m³/s]")
ax.set_title("TEST SET — Residuals  (QM predicted − QM observed)")
fig.tight_layout()
save_fig(fig, "TEST_03_residuals.png")

# TEST plot 4 — residual histogram
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(res_te, bins=50, color="#00695C", edgecolor="white", lw=0.3)
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_xlabel("Residual  [m³/s]"); ax.set_ylabel("Count")
ax.set_title("TEST SET — Residual Distribution")
fig.tight_layout()
save_fig(fig, "TEST_04_residual_hist.png")

# TEST plot 5 — loss curve
with open(os.path.join(MODEL_DIR, "history.json")) as f:
    hist = json.load(f)
has_nse = "val_nse" in hist and len(hist["val_nse"]) > 0
if has_nse:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(hist["train_loss"], label="Train loss", color="#1565C0", lw=1)
    axes[0].plot(hist["val_loss"],   label="Val loss",   color="#C62828",  lw=1)
    axes[0].set_ylabel("Combined Loss"); axes[0].legend(fontsize=8)
    axes[0].set_title("Training & Validation Loss - PatchTST v5")
    axes[1].plot(hist["val_nse"],      label="Val NSE",      color="#2E7D32", lw=1.2)
    axes[1].plot(hist["val_nse_peak"], label="Val NSE_peak", color="#E65100", lw=1.0, ls="--")
    axes[1].axhline(0.6, color="gray", lw=0.8, ls=":", label="NSE=0.6 target")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("NSE")
    axes[1].set_title("Validation NSE - model saved at peak (green dot)")
    best_ep = int(np.argmax(hist["val_nse"]))
    axes[1].scatter([best_ep], [hist["val_nse"][best_ep]], color="#2E7D32", s=60, zorder=5)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
else:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist["train_loss"], label="Train", color="#1565C0")
    ax.plot(hist["val_loss"],   label="Val",   color="#C62828")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Combined Loss")
    ax.set_title("Training & Validation Loss - PatchTST v5")
    ax.legend(); fig.tight_layout()
save_fig(fig, "TEST_05_loss_curve.png")

# TEST plot 6 — flow duration curve with TOPKAPI
fig, ax = plt.subplots(figsize=(8, 5))
exc = np.arange(1, len(obs_te)+1) / len(obs_te) * 100
ax.semilogy(exc, np.sort(obs_te)[::-1],  label="QM observed",  color=C_OBS,     lw=1.5)
ax.semilogy(exc, np.sort(pred_te)[::-1], label="QM predicted", color=C_PRED,     lw=1.2, ls="--")
ax.semilogy(exc, np.sort(Q_te)[::-1],    label="Q TOPKAPI",    color=C_TOPKAPI,  lw=1.2, ls=":")
ax.set_xlabel("Exceedance probability  [%]")
ax.set_ylabel("Flow  [m³/s]  (log scale)")
ax.set_title("TEST SET — Flow Duration Curve")
ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
save_fig(fig, "TEST_06_flow_duration_curve.png")

# TEST plot 7 — head comparison
fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
axes[0].plot(times_te_dt, obs_te,   color=C_OBS,     lw=1.0, label="QM observed")
axes[0].plot(times_te_dt, pred1_te, color="#2E7D32",  lw=0.8, label="Head1 (log-space)")
axes[0].plot(times_te_dt, Q_te,     color=C_TOPKAPI, lw=0.8, ls="--", label="Q TOPKAPI")
axes[0].set_ylabel("Flow [m³/s]"); axes[0].legend(fontsize=8)
axes[0].set_title("Head 1 — log1p path")

axes[1].plot(times_te_dt, obs_te,   color=C_OBS,     lw=1.0, label="QM observed")
axes[1].plot(times_te_dt, pred2_te, color="#E65100",  lw=0.8, label="Head2 (magnitude)")
axes[1].plot(times_te_dt, Q_te,     color=C_TOPKAPI, lw=0.8, ls="--", label="Q TOPKAPI")
axes[1].set_ylabel("Flow [m³/s]"); axes[1].legend(fontsize=8)
axes[1].set_title("Head 2 — magnitude path")

axes[2].plot(times_te_dt, obs_te,  color=C_OBS,     lw=1.0, label="QM observed")
axes[2].plot(times_te_dt, pred_te, color=C_PRED,    lw=0.8, label="QM predicted (blended)")
axes[2].plot(times_te_dt, Q_te,    color=C_TOPKAPI, lw=0.8, ls="--", label="Q TOPKAPI")
axes[2].set_ylabel("Flow [m³/s]"); axes[2].set_xlabel("Date")
axes[2].legend(fontsize=8); axes[2].set_title("Blended prediction vs TOPKAPI")
axes[2].xaxis.set_major_formatter(TFMT)
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45); fig.tight_layout()
save_fig(fig, "TEST_07_head_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — FULL DATASET (2014-2024)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "▓"*52)
print("  SECTION B — FULL DATASET  (2014-2024)")
print("▓"*52)

X_all   = torch.tensor(np.load(os.path.join(DATA_DIR, "X_all.npy")), dtype=torch.float32)
ref_all = pd.read_csv(os.path.join(DATA_DIR, "all_reference.csv"))
ref_all["Time"] = pd.to_datetime(ref_all["Time"])

times_all = ref_all["Time"].values
obs_all   = ref_all["QM"].values
Q_all     = ref_all["Q"].values    # TOPKAPI

pred_all, pred1_all, pred2_all, blend_all = run_inference(X_all)
res_all = pred_all - obs_all
m_all   = compute_metrics(obs_all, pred_all, "FULL DATASET (2014-2024)")

# save full CSV
_all_dt = pd.to_datetime(times_all)
pd.DataFrame({
    "datetime":     _all_dt.strftime("%Y-%m-%d %H:%M"),
    "QM_observed":  np.round(obs_all,   3),
    "QM_predicted": np.round(pred_all,  3),
    "Q_TOPKAPI":    np.round(Q_all,     3),
    "residual":     np.round(res_all,   3),
}).to_csv(os.path.join(OUT_DIR, "predictions_all.csv"), index=False,
          quoting=csv.QUOTE_NONNUMERIC)
print("✓ predictions_all.csv saved")

times_all_dt = pd.to_datetime(times_all)

# ALL plot 1 — time series with TOPKAPI (full 2014-2024)
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(times_all_dt, obs_all,  label="QM observed",  color=C_OBS,     lw=0.8)
ax.plot(times_all_dt, pred_all, label="QM predicted", color=C_PRED,    lw=0.7, alpha=0.85)
ax.plot(times_all_dt, Q_all,    label="Q TOPKAPI",    color=C_TOPKAPI, lw=0.7, alpha=0.8, ls="--")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
ax.set_xlabel("Date"); ax.set_ylabel("Flow  [m³/s]")
ax.set_title(f"FULL DATASET (2014-2024) — QM observed / QM predicted / Q TOPKAPI  "
             f"(NSE={m_all['nse']:.3f}  PBIAS={m_all['pbias']:+.1f}%)")
ax.legend(); fig.tight_layout()
save_fig(fig, "ALL_01_timeseries.png")

# ALL plot 2 — scatter with TOPKAPI
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(obs_all, pred_all, s=4, alpha=0.25, color="#4527A0", label="QM pred vs obs")
ax.scatter(obs_all, Q_all,    s=4, alpha=0.2,  color=C_TOPKAPI,  label="Q TOPKAPI vs obs", marker="^")
lim = max(obs_all.max(), pred_all.max(), Q_all.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("QM observed  [m³/s]"); ax.set_ylabel("Predicted / TOPKAPI  [m³/s]")
ax.set_title(f"Scatter — Full Dataset  |  R²={m_all['r2']:.3f}  NSE={m_all['nse']:.3f}")
ax.legend(fontsize=8); fig.tight_layout()
save_fig(fig, "ALL_02_scatter.png")

# ALL plot 3 — residuals over time
fig, ax = plt.subplots(figsize=(20, 3))
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.fill_between(times_all_dt, res_all, alpha=0.5, color="#E65100")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
ax.set_xlabel("Date"); ax.set_ylabel("Residual  [m³/s]")
ax.set_title("FULL DATASET — Residuals  (QM predicted − QM observed)")
fig.tight_layout()
save_fig(fig, "ALL_03_residuals.png")

# ALL plot 4 — residual histogram
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(res_all, bins=80, color="#00695C", edgecolor="white", lw=0.3)
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_xlabel("Residual  [m³/s]"); ax.set_ylabel("Count")
ax.set_title("FULL DATASET — Residual Distribution")
fig.tight_layout()
save_fig(fig, "ALL_04_residual_hist.png")

# ALL plot 5 — per-year box plots: QM obs vs QM pred vs Q TOPKAPI
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
df_box = pd.DataFrame({
    "year":       pd.to_datetime(times_all).year,
    "QM_obs":     obs_all,
    "QM_pred":    pred_all,
    "Q_TOPKAPI":  Q_all,
})
years = sorted(df_box["year"].unique())
for ax, (col, label, color) in zip(
    axes,
    [("QM_obs",    "QM observed",  C_OBS),
     ("QM_pred",   "QM predicted", C_PRED),
     ("Q_TOPKAPI", "Q TOPKAPI",    C_TOPKAPI)]
):
    data = [df_box.loc[df_box["year"]==y, col].values for y in years]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.set_xlabel("Year"); ax.set_ylabel("Flow  [m³/s]")
    ax.set_title(label)
fig.suptitle("Annual Flow Distribution — QM obs / QM pred / Q TOPKAPI", fontsize=11)
fig.tight_layout()
save_fig(fig, "ALL_05_annual_boxplots.png")

# ALL plot 6 — flow duration curve with TOPKAPI
fig, ax = plt.subplots(figsize=(8, 5))
exc = np.arange(1, len(obs_all)+1) / len(obs_all) * 100
ax.semilogy(exc, np.sort(obs_all)[::-1],  label="QM observed",  color=C_OBS,     lw=1.5)
ax.semilogy(exc, np.sort(pred_all)[::-1], label="QM predicted", color=C_PRED,     lw=1.2, ls="--")
ax.semilogy(exc, np.sort(Q_all)[::-1],    label="Q TOPKAPI",    color=C_TOPKAPI,  lw=1.2, ls=":")
ax.set_xlabel("Exceedance probability  [%]")
ax.set_ylabel("Flow  [m³/s]  (log scale)")
ax.set_title("FULL DATASET — Flow Duration Curve")
ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
save_fig(fig, "ALL_06_flow_duration_curve.png")

print(f"\n✓ All plots saved to {PLOT_DIR}")
print(f"\nSUMMARY")
print(f"  Test  set NSE: {m_te['nse']:.4f}   PBIAS: {m_te['pbias']:+.2f}%")
print(f"  Full  set NSE: {m_all['nse']:.4f}   PBIAS: {m_all['pbias']:+.2f}%")