"""
Snippet 0 — Setup Check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run this first, before anything else, to confirm the environment and
folder structure are correct. Catches the most common setup mistakes
before they turn into confusing errors three scripts later.

Usage:
  python 0_check_setup.py
"""

import os, sys, configparser

_HERE = os.path.dirname(os.path.abspath(__file__))
print(f"Checking setup in: {_HERE}\n")

ok = True

def check(label, condition, fix_hint=""):
    global ok
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        ok = False
        if fix_hint:
            print(f"         -> {fix_hint}")

# ── 0. Terminal encoding ───────────────────────────────────────────────────────
# The other scripts print unicode characters (checkmarks, arrows, m³/s, +/-).
# Modern terminals (VS Code, Windows Terminal, PowerShell 7+) handle this fine
# by default. Older standalone cmd.exe windows on some systems default to a
# legacy codepage and can raise UnicodeEncodeError on these prints.
print("Terminal:")
stdout_enc = (sys.stdout.encoding or "").lower()
check(f"stdout encoding is UTF-8 (found: {sys.stdout.encoding})",
     "utf" in stdout_enc,
     "If later scripts crash with UnicodeEncodeError, either run from "
     "VS Code's integrated terminal (recommended), or run "
     "'chcp 65001' in cmd.exe first, or set environment variable "
     "PYTHONUTF8=1 before running python.")

# ── 1. Python packages ────────────────────────────────────────────────────────
print("Python packages:")
import_names = {"scikit-learn": "sklearn"}
for pkg in ["numpy", "pandas", "torch", "scikit-learn", "matplotlib"]:
    mod_name = import_names.get(pkg, pkg)
    try:
        __import__(mod_name)
        check(pkg, True)
    except ImportError:
        check(pkg, False, "run: pip install -r requirements.txt")

# ── 1b. venv check ─────────────────────────────────────────────────────────────
print("\nPython environment:")
in_venv = (hasattr(sys, "real_prefix") or
          (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
check("running inside a virtual environment", in_venv,
     "this works outside a venv too, but a venv is recommended — "
     "see README.md 'VS Code setup' section")
print(f"         interpreter: {sys.executable}")

# ── 2. Config file ────────────────────────────────────────────────────────────
print("\nConfiguration:")
ini_path = os.path.join(_HERE, "LSTM.ini")
check("LSTM.ini exists", os.path.exists(ini_path),
     "LSTM.ini should be in the same folder as this script")

if os.path.exists(ini_path):
    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8-sig")
    has_section = "patchtst" in cfg
    check("[patchtst] section present", has_section)
    if has_section:
        pt = cfg["patchtst"]
        for key in ["raw_ts", "data_dir", "model_dir", "out_dir",
                   "seq_len", "patch_len", "stride"]:
            check(f"  key '{key}'", key in pt)

# ── 3. Folder structure ───────────────────────────────────────────────────────
print("\nFolder structure:")
for folder in ["data/raw", "data/processed", "models", "outputs", "outputs/plots"]:
    path = os.path.join(_HERE, folder)
    check(folder, os.path.isdir(path), f"mkdir -p {folder}")

# ── 4. Raw data file ──────────────────────────────────────────────────────────
print("\nInput data:")
if os.path.exists(ini_path):
    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8-sig")
    raw_ts = cfg["patchtst"].get("raw_ts", "data/raw/425.sbs.ts")
    raw_path = os.path.join(_HERE, raw_ts.replace("\\", os.sep).replace("/", os.sep))
    exists = os.path.exists(raw_path)
    check(f"raw data file ({raw_ts})", exists,
         "Place the TOPKAPI .sbs.ts export at this path before running 1_prepare_data.py")
    if exists:
        size_mb = os.path.getsize(raw_path) / 1e6
        print(f"         file size: {size_mb:.1f} MB")

# ── 5. Pipeline stage outputs (informational, not required at first run) ─────
print("\nPipeline stage status (informational):")
stage_files = [
    ("1_prepare_data.py ran",  "data/processed/X_train.npy"),
    ("2_train.py ran",         "models/best_model.pt"),
    ("3_evaluate.py ran",      "outputs/predictions_test.csv"),
    ("4_conformal.py ran",     "models/conformal_qhats.json"),
    ("5_realtime_infer.py ran","outputs/forecast_latest.csv"),
]
for label, relpath in stage_files:
    exists = os.path.exists(os.path.join(_HERE, relpath))
    marker = "done" if exists else "not yet run"
    print(f"  [{marker:>11}] {label}")

# ── summary ───────────────────────────────────────────────────────────────────
print()
if ok:
    print("Setup looks correct. You can run the pipeline:")
    print("  python 1_prepare_data.py")
    print("  python 2_train.py")
    print("  python 3_evaluate.py")
    print("  python 4_conformal.py")
    print("  python 5_realtime_infer.py")
else:
    print("Some checks failed — fix the items marked FAIL above before proceeding.")
    sys.exit(1)
