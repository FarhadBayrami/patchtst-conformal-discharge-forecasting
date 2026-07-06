"""
 PatchTST Model & Training  (v5)
Dataset   : 425.sbs.ts  (station 425, Casalecchio di Reno)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


─────────────
1. SAVE ON NSE, NOT HUBER LOSS
  this code decodes both heads back to real m³/s during every validation
   pass and computes NSE directly. best_model.pt is saved on the
   highest val NSE seen so far.

2. PBIAS-AWARE EARLY STOPPING
   Early stopping now also monitors PBIAS. If NSE plateaus but PBIAS
   is still improving (i.e. the systematic underestimation is reducing)
   training continues. Patience counter only increments when BOTH
   NSE and PBIAS stop improving.

3. PEAK NSE TRACKED EVERY EPOCH
   NSE_peak (top-10% flow events on the val set) is computed and
   logged every epoch. This is the hardest metric and the one most
   visible to the supervisor. It is printed at every log interval
   and saved in history.json.

4. HYPERPARAMETER CHANGES (set in LSTM.ini)
   batch_size   64    larger batches, more stable gradients
   epochs       400   scheduler needs room to anneal fully
   patience     80    generous patience avoids stopping before the
                       model has genuinely converged, especially with
                       the flood-weighted sampler making each epoch's
                       loss noisier than a uniform sample would be
   huber_delta  5.0   quadratic zone now covers realistic peak errors
   loss_w1      0.3   reduce log-head dominance
   loss_w2      1.0   with delta=5 the magnitude head already gives
                            much stronger gradient; raw weight can be lower


"""

import os, json, pickle, configparser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ── config ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_cfg  = configparser.ConfigParser()
_cfg.read(os.path.join(_HERE, "LSTM.ini"), encoding="utf-8-sig")
_pt   = _cfg["patchtst"]

DATA_DIR  = os.path.join(_HERE, _pt["data_dir"])
MODEL_DIR = os.path.join(_HERE, _pt["model_dir"])
os.makedirs(MODEL_DIR, exist_ok=True)

# ── window config (must match 1_prepare_data.py) ──────────────────────────────
SEQ_LEN   = _pt.getint("seq_len")
PATCH_LEN = _pt.getint("patch_len")
STRIDE    = _pt.getint("stride")

# ── model hyper-parameters ────────────────────────────────────────────────────
D_MODEL  = _pt.getint("d_model")
N_HEADS  = _pt.getint("n_heads")
N_LAYERS = _pt.getint("n_layers")
D_FF     = _pt.getint("d_ff")
DROPOUT  = _pt.getfloat("dropout")

# ── training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE  = _pt.getint("batch_size")
EPOCHS      = _pt.getint("epochs")
LR          = _pt.getfloat("lr")
PATIENCE    = _pt.getint("patience")
HUBER_DELTA = _pt.getfloat("huber_delta")
LOSS_W1     = _pt.getfloat("loss_w1")
LOSS_W2     = _pt.getfloat("loss_w2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

# ── load data ─────────────────────────────────────────────────────────────────
def load(tag):
    return torch.tensor(np.load(os.path.join(DATA_DIR, f"{tag}.npy")),
                        dtype=torch.float32)

X_train, Y_train = load("X_train"), load("Y_train")
X_val,   Y_val   = load("X_val"),   load("Y_val")

N_FEATURES = X_train.shape[2]
N_PATCHES  = (SEQ_LEN - PATCH_LEN) // STRIDE + 1
print(f"Features: {N_FEATURES}  |  Patches/sample: {N_PATCHES}")
print(f"Train samples: {len(X_train)}  |  Val samples: {len(X_val)}")

# ── load scalers (needed for val NSE decoding) ────────────────────────────────
with open(os.path.join(DATA_DIR, "scaler_y1.pkl"), "rb") as f:
    scaler_y1 = pickle.load(f)
QM_TRAIN_MAX = float(np.load(os.path.join(DATA_DIR, "QM_TRAIN_MAX.npy"))[0])
print(f"QM_TRAIN_MAX : {QM_TRAIN_MAX:.2f}")

# ── flood-aware weighted sampler ──────────────────────────────────────────────
# Y_train[:,1] is QM_norm = QM / QM_TRAIN_MAX
qm_norm = Y_train[:, 1]
weights = torch.ones(len(Y_train))
weights[qm_norm > 0.05] = 6.0    # QM > 5% of QM_TRAIN_MAX
weights[qm_norm > 0.12] = 15.0   # QM > 12% of QM_TRAIN_MAX
# (printed QM_TRAIN_MAX above gives the exact m³/s for these thresholds —
#  e.g. with QM_TRAIN_MAX~948 m³/s these are ~47 and ~114 m³/s; this scales
#  automatically if retrained on a different dataset/catchment)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_dl = DataLoader(TensorDataset(X_train, Y_train),
                      batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
val_dl   = DataLoader(TensorDataset(X_val, Y_val),
                      batch_size=BATCH_SIZE, shuffle=False)

# ── PatchTST with dual head ───────────────────────────────────────────────────
class PatchTST(nn.Module):
    """
    Channel-independent PatchTST with two output heads.
    Architecture identical to v4 — only training loop changes.
      head1 → log1p(QM) standardised
      head2 → QM / QM_max  (magnitude-preserving, Softplus output)
    """
    def __init__(self, n_features, seq_len, patch_len, stride,
                 d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.n_features = n_features
        self.patch_len  = patch_len
        self.stride     = stride
        self.n_patches  = (seq_len - patch_len) // stride + 1

        self.patch_embed   = nn.Linear(patch_len, d_model)
        self.pos_embed     = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        trunk_in = n_features * self.n_patches * d_model
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(trunk_in),
            nn.Linear(trunk_in, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 1),
        )

        self.head2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        B, T, C = x.shape
        # Treat each channel as an independent sequence in the batch dimension
        # (B, T, C) → (B*C, T)
        x = x.permute(0, 2, 1).reshape(B * C, T)
        p = x.unfold(1, self.patch_len, self.stride)   # (B*C, N_patches, patch_len)
        p = self.patch_embed(p) + self.pos_embed        # (B*C, N_patches, d_model)
        p = self.input_dropout(p)
        p = self.norm(self.encoder(p))                  # (B*C, N_patches, d_model)
        # Reshape back to (B, C, N_patches, d_model) for the trunk
        out   = p.reshape(B, C, self.n_patches, -1)
        trunk = self.trunk(out)
        return self.head1(trunk), self.head2(trunk)


model = PatchTST(
    n_features=N_FEATURES, seq_len=SEQ_LEN, patch_len=PATCH_LEN,
    stride=STRIDE, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")

# ── loss ──────────────────────────────────────────────────────────────────────
huber = nn.HuberLoss(delta=HUBER_DELTA)

def combined_loss(pred1, pred2, y1, y2):
    l1 = huber(pred1, y1)
    l2 = huber(pred2, y2)
    return LOSS_W1 * l1 + LOSS_W2 * l2, l1.item(), l2.item()

# ── decoding helpers (scaled outputs → real m³/s) ────────────────────────────
def decode_to_qm(p1_np, p2_np):
    """
    Convert raw model outputs back to m³/s for NSE/PBIAS computation.

    p1_np : (N, 1)  log1p(QM) in standardised space  (head1 output)
    p2_np : (N, 1)  QM / QM_TRAIN_MAX                (head2 output)

    Returns (N,) array in m³/s.

    Blend logic: head2 (magnitude) is trusted more for high flows;
    head1 (log-space) is better calibrated for low flows.
    The sigmoid blend transitions smoothly between them.
    """
    q_log = np.expm1(
        scaler_y1.inverse_transform(p1_np)
    ).flatten().clip(0)                        # head1 → m³/s

    q_mag = (p2_np * QM_TRAIN_MAX).flatten().clip(0)   # head2 → m³/s

    # adaptive blend: α transitions from 0 (trust head1) at low flow
    # to 1 (trust head2) at high flow, centred at blend_threshold
    blend_threshold = _pt.getfloat("blend_threshold")   # m³/s, from ini
    blend_scale     = _pt.getfloat("blend_scale")        # steepness
    alpha = 1.0 / (1.0 + np.exp(-(q_mag - blend_threshold) / blend_scale))
    return (1.0 - alpha) * q_log + alpha * q_mag

def compute_nse(obs, pred):
    """Nash-Sutcliffe Efficiency. Both arrays in m³/s, shape (N,)."""
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))

def compute_pbias(obs, pred):
    """Percent bias. Negative = underestimation."""
    return float(100.0 * (pred.sum() - obs.sum()) / (obs.sum() + 1e-8))

# ── pre-compute val obs in m³/s (for NSE every epoch) ────────────────────────
# Y_val[:,1] = QM_norm;  Y_val[:,0] = QM_log_scaled
val_obs_qm = (Y_val[:, 1].numpy() * QM_TRAIN_MAX)   # shape (N_val,)
peak_threshold = np.percentile(val_obs_qm, 90)
peak_mask_val  = val_obs_qm > peak_threshold
print(f"Val peak threshold (90th pct): {peak_threshold:.1f} m³/s  "
      f"({peak_mask_val.sum()} samples)")

# ── optimiser & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, epochs=EPOCHS,
    steps_per_epoch=len(train_dl),
    pct_start=0.2, anneal_strategy="cos", final_div_factor=1e4,
)

# ── training loop ─────────────────────────────────────────────────────────────
history = {
    "train_loss": [], "val_loss": [],
    "train_l1":   [], "train_l2": [],
    "val_nse":    [], "val_pbias": [], "val_nse_peak": [],
}

best_val_nse     = -float("inf")
best_val_pbias   = float("inf")   # abs value — lower is better
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    # ── train ─────────────────────────────────────────────────────────────────
    model.train()
    run_loss = run_l1 = run_l2 = 0.0
    n_batches = len(train_dl)
    for batch_idx, (xb, yb) in enumerate(train_dl):
        if batch_idx % 50 == 0:
            print(f"\r  batch {batch_idx}/{n_batches}", end="", flush=True)
        xb, yb   = xb.to(DEVICE), yb.to(DEVICE)
        y1b, y2b = yb[:, 0:1], yb[:, 1:2]
        optimizer.zero_grad()
        p1, p2   = model(xb)
        loss, l1, l2 = combined_loss(p1, p2, y1b, y2b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        n = len(xb)
        run_loss += loss.item() * n
        run_l1   += l1 * n
        run_l2   += l2 * n

    print()  # newline after batch progress
    n_tr       = len(train_dl.dataset)
    train_loss = run_loss / n_tr

    # ── validate ──────────────────────────────────────────────────────────────
    model.eval()
    val_run   = 0.0
    p1_list, p2_list = [], []

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            p1, p2 = model(xb)
            v, _, _ = combined_loss(p1, p2, yb[:, 0:1], yb[:, 1:2])
            val_run += v.item() * len(xb)
            p1_list.append(p1.cpu().numpy())
            p2_list.append(p2.cpu().numpy())

    val_loss = val_run / len(val_dl.dataset)

    # decode to m³/s and compute operational metrics
    p1_all   = np.concatenate(p1_list, axis=0)   # (N_val, 1)
    p2_all   = np.concatenate(p2_list, axis=0)   # (N_val, 1)
    pred_qm  = decode_to_qm(p1_all, p2_all)      # (N_val,)

    val_nse      = compute_nse(val_obs_qm, pred_qm)
    val_pbias    = compute_pbias(val_obs_qm, pred_qm)
    val_nse_peak = compute_nse(val_obs_qm[peak_mask_val],
                               pred_qm[peak_mask_val])

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_l1"].append(run_l1 / n_tr)
    history["train_l2"].append(run_l2 / n_tr)
    history["val_nse"].append(val_nse)
    history["val_pbias"].append(val_pbias)
    history["val_nse_peak"].append(val_nse_peak)

    # ── logging ───────────────────────────────────────────────────────────────
    if epoch % 20 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"NSE={val_nse:.4f}  NSE_peak={val_nse_peak:.4f}  "
            f"PBIAS={val_pbias:+.1f}%  lr={lr_now:.2e}"
        )

    # ── save best on NSE (primary) ────────────────────────────────────────────
    if val_nse > best_val_nse:
        best_val_nse     = val_nse
        best_val_pbias   = abs(val_pbias)
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

    else:
        # also reset patience if PBIAS is still improving meaningfully
        # (model may be correcting systematic bias even if NSE plateaus)
        pbias_improved = abs(val_pbias) < best_val_pbias - 1.0   # 1% threshold
        if pbias_improved:
            best_val_pbias   = abs(val_pbias)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

# ── save checkpoint ───────────────────────────────────────────────────────────
torch.save({
    "epoch":       epoch,
    "model_state": model.state_dict(),
    "val_nse":     best_val_nse,
    "val_loss":    val_loss,
    "hparams": dict(
        n_features=N_FEATURES, seq_len=SEQ_LEN, patch_len=PATCH_LEN,
        stride=STRIDE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
    ),
}, os.path.join(MODEL_DIR, "checkpoint.pt"))

with open(os.path.join(MODEL_DIR, "history.json"), "w") as f:
    json.dump(history, f)

print(f"\n✓ Best val NSE  : {best_val_nse:.4f}")
print(f"✓ Final PBIAS   : {val_pbias:+.1f}%")
print(f"✓ Saved to      : {MODEL_DIR}")