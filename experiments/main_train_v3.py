"""
PatchTST training pipeline for GiftEval datasets — random hyperparameter search.

Pipeline
--------
For each (dataset, term) in DATASETS:
  1.  Load GiftEval Dataset(term=..., to_univariate=...). The dataset already
      exposes the FIRST 90% of the original series in `training_dataset`.
  2.  Subdivide each training series temporally:
          first 80/90 ≈ 88.89%  -> actual training
          last  10/90 ≈ 11.11%  -> validation
      i.e. 80% / 10% / 10% of the *original* data.
  3.  Build sliding-window (input_window=CONTEXT_LENGTH, horizon=H) tensors
      and replicate them on each configured GPU (one copy per device).
  4.  Random search of N_TRIALS hyperparameter configurations sampled from
      a pre-defined grid (HF transformers PatchTSTConfig).
  5.  Train each trial with:
          - empirical VRAM probe to pick the largest batch size that fits;
          - LR rescaled from the grid value via the sqrt rule (Adam-safe);
          - validation every VAL_EVERY_N_EPOCHS;
          - early stopping with patience EARLY_STOPPING_PATIENCE;
          - Student-t distribution head + NLL loss (probabilistic).
      Save *only* the weights with best validation loss across all trials.
  6.  Plot the random-search summary:
          - scatter (trial index vs. val loss)
          - running "best-so-far" curve connecting the points that improved
            upon all previous trials (everything else is dropped).

Concurrency model
-----------------
Multiple GPUs (resolved from DEVICES or CUDA_VISIBLE_DEVICES) are filled in
round-robin order. Within each GPU, up to MAX_CONCURRENT_TRIALS slots run
concurrently via independent CUDA streams. Across GPUs, work runs truly in
parallel on independent hardware. The total parallel slot count is therefore
MAX_CONCURRENT_TRIALS × num_devices.

Cache layout
------------
logs/experiments/patchtst_training/<dataset>/<term>/
        best_model.pt              <-- only the best weights across all trials
        best_config.json           <-- HPs of best trial (loaded by main_test.py)
        trials/trial_<NNN>.json    <-- per-trial result (HPs + final val_loss)
        random_search_summary.png  <-- scatter + running-best curve
        random_search.csv          <-- all trials in tabular form

Prerequisites
-------------
1. pip install gift-eval python-dotenv transformers
2. Download GiftEval data and set GIFT_EVAL env var (.env at repo root).
"""

import torch
import os
import json
import math
import time
import random
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore

from dotenv import load_dotenv
load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction


# ==============================================================================
#  EXPERIMENT CONFIGURATION
# ==============================================================================


DATASETS = [
    # --- Jena Weather ---
    ("jena_weather/10T",            "short",  "JenaWeather-10T", True),
    ("jena_weather/10T",            "medium", "JenaWeather-10T", True),
    ("jena_weather/10T",            "long",   "JenaWeather-10T", True),

    ("jena_weather/H",              "short",  "JenaWeather-H", True),
    ("jena_weather/H",              "medium", "JenaWeather-H", True),
    ("jena_weather/H",              "long",   "JenaWeather-H", True),

    ("jena_weather/D",              "short",  "JenaWeather-D", True),

    # --- BizITObs ---
    ("bizitobs_application",        "short",  "BizITObsApp", True),
    ("bizitobs_application",        "medium", "BizITObsApp", True),
    ("bizitobs_application",        "long",   "BizITObsApp", True),

    ("bizitobs_service",            "short",  "BizITObsService", True),
    ("bizitobs_service",            "medium", "BizITObsService", True),
    ("bizitobs_service",            "long",   "BizITObsService", True),

    ("bizitobs_l2c/5T",             "short",  "BizITObsL2C-5T", True),
    ("bizitobs_l2c/5T",             "medium", "BizITObsL2C-5T", True),
    ("bizitobs_l2c/5T",             "long",   "BizITObsL2C-5T", True),

    ("bizitobs_l2c/H",              "short",  "BizITObsL2C-H", True),
    ("bizitobs_l2c/H",              "medium", "BizITObsL2C-H", True),
    ("bizitobs_l2c/H",              "long",   "BizITObsL2C-H", True),

    # --- Bitbrains ---
    ("bitbrains_fast_storage/5T",   "short",  "BitbrainsFS-5T", True),
    ("bitbrains_fast_storage/5T",   "medium", "BitbrainsFS-5T", True),
    ("bitbrains_fast_storage/5T",   "long",   "BitbrainsFS-5T", True),

    ("bitbrains_fast_storage/H",    "short",  "BitbrainsFS-H", True),

    ("bitbrains_rnd/5T",            "short",  "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/5T",            "medium", "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/5T",            "long",   "BitbrainsRnD-5T", True),

    ("bitbrains_rnd/H",             "short",  "BitbrainsRnD-H", True),

    # --- Restaurant ---
    ("restaurant",                  "short",  "Restaurant", False),

    # --- ETT1 ---
    ("ett1/15T",                    "short",  "ETTm1-15T", True),
    ("ett1/15T",                    "medium", "ETTm1-15T", True),
    ("ett1/15T",                    "long",   "ETTm1-15T", True),

    ("ett1/H",                      "short",  "ETTm1-H", True),
    ("ett1/H",                      "medium", "ETTm1-H", True),
    ("ett1/H",                      "long",   "ETTm1-H", True),

    ("ett1/D",                      "short",  "ETTm1-D", True),

    ("ett1/W",                      "short",  "ETTm1-W", True),

    # --- ETT2 ---
    ("ett2/15T",                    "short",  "ETTm2-15T", True),
    ("ett2/15T",                    "medium", "ETTm2-15T", True),
    ("ett2/15T",                    "long",   "ETTm2-15T", True),

    ("ett2/H",                      "short",  "ETTm2-H", True),
    ("ett2/H",                      "medium", "ETTm2-H", True),
    ("ett2/H",                      "long",   "ETTm2-H", True),

    ("ett2/D",                      "short",  "ETTm2-D", True),

    ("ett2/W",                      "short",  "ETTm2-W", True),

    # --- Loop Seattle ---
    ("LOOP_SEATTLE/5T",             "short",  "LoopSeattle-5T", False),
    ("LOOP_SEATTLE/5T",             "medium", "LoopSeattle-5T", False),
    ("LOOP_SEATTLE/5T",             "long",   "LoopSeattle-5T", False),

    ("LOOP_SEATTLE/H",              "short",  "LoopSeattle-H", False),
    ("LOOP_SEATTLE/H",              "medium", "LoopSeattle-H", False),
    ("LOOP_SEATTLE/H",              "long",   "LoopSeattle-H", False),

    ("LOOP_SEATTLE/D",              "short",  "LoopSeattle-D", False),

    # --- SZ Taxi ---
    ("SZ_TAXI/15T",                 "short",  "SZTaxi-15T", False),
    ("SZ_TAXI/15T",                 "medium", "SZTaxi-15T", False),
    ("SZ_TAXI/15T",                 "long",   "SZTaxi-15T", False),

    ("SZ_TAXI/H",                   "short",  "SZTaxi-H", False),

    # --- M_DENSE ---
    ("M_DENSE/H",                   "short",  "MDense-H", False),
    ("M_DENSE/H",                   "medium", "MDense-H", False),
    ("M_DENSE/H",                   "long",   "MDense-H", False),

    ("M_DENSE/D",                   "short",  "MDense-D", False),

    # --- Solar ---
    ("solar/10T",                   "short",  "Solar-10T", False),
    ("solar/10T",                   "medium", "Solar-10T", False),
    ("solar/10T",                   "long",   "Solar-10T", False),

    ("solar/H",                     "short",  "Solar-H", False),
    ("solar/H",                     "medium", "Solar-H", False),
    ("solar/H",                     "long",   "Solar-H", False),

    ("solar/D",                     "short",  "Solar-D", False),

    ("solar/W",                     "short",  "Solar-W", False),

    # --- Hierarchical Sales ---
    ("hierarchical_sales/D",        "short",  "HierSales-D", False),

    ("hierarchical_sales/W",        "short",  "HierSales-W", False),

    # --- M4 ---
    ("m4_yearly",                   "short",  "M4-Yearly", False),
    ("m4_quarterly",                "short",  "M4-Quarterly", False),
    ("m4_monthly",                  "short",  "M4-Monthly", False),
    ("m4_weekly",                   "short",  "M4-Weekly", False),
    ("m4_daily",                    "short",  "M4-Daily", False),
    ("m4_hourly",                   "short",  "M4-Hourly", False),

    # --- Healthcare ---
    ("hospital",                    "short",  "Hospital", False),

    ("covid_deaths",                "short",  "CovidDeaths-D", False),

    ("us_births/D",                 "short",  "USBirths-D", False),
    ("us_births/W",                 "short",  "USBirths-W", False),
    ("us_births/M",                 "short",  "USBirths-M", False),

    # --- Saugeen ---
    ("saugeenday/D",                "short",  "SaugeenDay-D", False),
    ("saugeenday/W",                "short",  "SaugeenDay-W", False),
    ("saugeenday/M",                "short",  "SaugeenDay-M", False),

    # --- Temperature Rain ---
    ("temperature_rain_with_missing", "short", "TempRainMissing-D", True),

    # --- KDD Cup 2018 ---
    ("kdd_cup_2018_with_missing/H", "short",  "KDDCup18Missing-H", True),
    ("kdd_cup_2018_with_missing/H", "medium", "KDDCup18Missing-H", True),
    ("kdd_cup_2018_with_missing/H", "long",   "KDDCup18Missing-H", True),

    ("kdd_cup_2018_with_missing/D", "short",  "KDDCup18Missing-D", True),

    # --- Car Parts ---
    ("car_parts_with_missing",      "short",  "CarPartsMissing", False),

    # --- Electricity ---
    ("electricity/15T",             "short",  "Electricity-15T", False),
    ("electricity/15T",             "medium", "Electricity-15T", False),
    ("electricity/15T",             "long",   "Electricity-15T", False),

    ("electricity/H",               "short",  "Electricity-H", False),
    ("electricity/H",               "medium", "Electricity-H", False),
    ("electricity/H",               "long",   "Electricity-H", False),

    ("electricity/D",               "short",  "Electricity-D", False),

    ("electricity/W",               "short",  "Electricity-W", False),
]


# -- Fixed training-loop hyperparameters ---------------------------------------
CONTEXT_LENGTH         = 512
N_TRIALS               = 100
MAX_EPOCHS             = 50
VAL_EVERY_N_EPOCHS     = 4
EARLY_STOPPING_PATIENCE = 4            # in "validation events" (not epochs)
WEIGHT_DECAY           = 1e-4
GRAD_CLIP              = 1.0
MAX_TRAIN_WINDOWS      = 50_000        # cap per-(dataset,term) train windows
MAX_VAL_WINDOWS        = 5_000
SEED                   = 42

# -- Auto batch-size + LR scaling ---------------------------------------------
# `learning_rate` in HP_SPACE is interpreted as the value FOR `BS_REFERENCE`.
# The actual lr fed to AdamW is rescaled by the chosen bs through LR_SCALING_RULE:
#       "sqrt"    -> lr * sqrt(bs / bs_ref)        (Krizhevsky, 2014 — Adam-safe)
#       "linear"  -> lr * (bs / bs_ref)            (Goyal et al., 2017 — SGD)
# sqrt is the standard for Adam/AdamW because Adam already divides by sqrt(v_t),
# so the effective gradient signal scales sub-linearly with batch size.
BS_REFERENCE           = 128
BS_CANDIDATES          = [1024, 512, 256, 128, 64]   # tried in descending order
LR_SCALING_RULE        = "sqrt"        # "sqrt" or "linear"

# -- Concurrent training (multi-GPU, CUDA streams) -----------------------------
# DEVICES controls which GPUs are used. Options:
#   None       -> auto-detect all visible GPUs (respects CUDA_VISIBLE_DEVICES).
#   ["cuda:0"] -> force a single GPU even if more are visible.
#   ["cuda:0", "cuda:2"] -> use the listed devices explicitly.
# CUDA_VISIBLE_DEVICES is the standard way to expose a subset of physical GPUs;
# torch then numbers them 0..N-1 in the order given by the env var.
DEVICES                  = None
MAX_CONCURRENT_TRIALS    = 2         # per GPU
VRAM_BUDGET_PER_SLOT_GB  = 2.5       # per packable slot
VRAM_BUDGET_SOLO_GB      = 10.0      # per solo trial
ENABLE_CONCURRENT_TRAINING = False    # global on/off switch

# -- Random search HP space -------------------------------------------
HP_SPACE = {
    "patch_length": [8, 16, 32],
    "patch_stride": [4, 8, 16],   # or tie it to patch_length // 2
    "d_model": [64, 128],
    "num_hidden_layers": [3, 6],
    "num_attention_heads": [4, 8],
    "dropout": [0.1, 0.2],
    "learning_rate": [1e-4, 2.5e-4, 5e-4],
    "weight_decay": [1e-4, 1e-3, 1e-2],
}

# -- Inference sampling for CRPS (used at test time, fixed for consistency) ----
NUM_PARALLEL_SAMPLES = 100

CACHE_ROOT = "logs/experiments/patchtst_training"


# ==============================================================================
#  REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_devices() -> List[str]:
    """
    Resolve the list of devices to use for training.

    Honours CUDA_VISIBLE_DEVICES transparently — torch already renumbers visible
    GPUs to 0..N-1, so this just queries `torch.cuda.device_count()` unless the
    user overrode DEVICES at the top of the file.
    """
    if not torch.cuda.is_available():
        return ["cpu"]
    if DEVICES is not None:
        return list(DEVICES)
    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"]
    return [f"cuda:{i}" for i in range(n)]


def _device_index(device: str) -> Optional[int]:
    """'cuda:1' -> 1; 'cpu' -> None."""
    if not device.startswith("cuda"):
        return None
    return int(device.split(":")[1]) if ":" in device else 0


def _sync_device(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


# ==============================================================================
#  DATA BUILDING
# ==============================================================================

def _series_train_val_split(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a single series (already the first 90% of original) into:
        train_portion : first 80/90 of its length
        val_portion   : last  10/90 of its length

    This corresponds to 80% / 10% of the ORIGINAL data.
    """
    L = len(series)
    cut = int(L * (80.0 / 90.0))
    return series[:cut], series[cut:]

def _sliding_windows(
    series: np.ndarray,
    context_length: int,
    horizon: int,
    stride: int = 1,
    max_windows: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    L = len(series)
    win = context_length + horizon

    if L < win:
        return (
            np.empty((0, context_length), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    n_possible = (L - win) // stride + 1

    if max_windows is not None and n_possible > max_windows:
        if rng is None:
            rng = np.random.RandomState(SEED)
        starts = rng.choice(n_possible, size=max_windows, replace=False) * stride
        starts = np.sort(starts)
    else:
        starts = np.arange(0, L - win + 1, stride)

    xs = np.empty((len(starts), context_length), dtype=np.float32)
    ys = np.empty((len(starts), horizon), dtype=np.float32)

    for i, s in enumerate(starts):
        xs[i] = series[s : s + context_length]
        ys[i] = series[s + context_length : s + win]

    return xs, ys

def build_train_val_tensors(
    ge_dataset: GiftEvalDataset,
    context_length: int,
    horizon: int,
    max_train_windows: int = MAX_TRAIN_WINDOWS,
    max_val_windows: int = MAX_VAL_WINDOWS,
    rng_seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    rng = np.random.RandomState(rng_seed)
    entries = list(ge_dataset.training_dataset)
    rng.shuffle(entries)

    x_tr_list, y_tr_list, x_vl_list, y_vl_list = [], [], [], []
    n_train_total = 0
    n_val_total = 0

    for entry in entries:
        tgt = entry["target"]

        if tgt.ndim > 1:
            raise ValueError(
                f"Expected univariate target but got shape {tgt.shape}. "
                "Use to_univariate=True when loading the dataset."
            )

        tgt = np.nan_to_num(
            np.asarray(tgt, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        train_portion, val_portion = _series_train_val_split(tgt)

        if n_train_total < max_train_windows:
            remaining_train = max_train_windows - n_train_total
            x_tr, y_tr = _sliding_windows(
                train_portion,
                context_length,
                horizon,
                max_windows=remaining_train,
                rng=rng,
            )

            if len(x_tr) > 0:
                x_tr_list.append(x_tr)
                y_tr_list.append(y_tr)
                n_train_total += len(x_tr)

        if n_val_total < max_val_windows:
            remaining_val = max_val_windows - n_val_total
            x_vl, y_vl = _sliding_windows(
                val_portion,
                context_length,
                horizon,
                max_windows=remaining_val,
                rng=rng,
            )

            if len(x_vl) > 0:
                x_vl_list.append(x_vl)
                y_vl_list.append(y_vl)
                n_val_total += len(x_vl)

        if n_train_total >= max_train_windows and n_val_total >= max_val_windows:
            break

    if not x_tr_list:
        raise RuntimeError("No training windows could be built from this dataset.")
    if not x_vl_list:
        raise RuntimeError("No validation windows could be built from this dataset.")

    x_train = torch.from_numpy(np.concatenate(x_tr_list, axis=0)).unsqueeze(-1)
    y_train = torch.from_numpy(np.concatenate(y_tr_list, axis=0)).unsqueeze(-1)
    x_val = torch.from_numpy(np.concatenate(x_vl_list, axis=0)).unsqueeze(-1)
    y_val = torch.from_numpy(np.concatenate(y_vl_list, axis=0)).unsqueeze(-1)

    return x_train, y_train, x_val, y_val


# ==============================================================================
#  RANDOM SEARCH
# ==============================================================================

@dataclass
class TrialConfig:
    patch_length:        int
    patch_stride:        int
    d_model:             int
    num_hidden_layers:   int
    num_attention_heads: int
    dropout:             float
    learning_rate:       float
    weight_decay:        float

    def __post_init__(self):
        if self.d_model % self.num_attention_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )


def sample_trial_configs(n_trials: int, seed: int = SEED) -> List[TrialConfig]:
    """Sample N unique random configurations from HP_SPACE."""
    rng = random.Random(seed)
    seen, configs = set(), []
    max_attempts = n_trials * 50
    attempts = 0
    while len(configs) < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in HP_SPACE.items()}
        if cfg["d_model"] % cfg["num_attention_heads"] != 0:
            continue
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(TrialConfig(**cfg))
    return configs


# ==============================================================================
#  MODEL BUILDING
# ==============================================================================

def build_patchtst(
    trial: TrialConfig,
    context_length: int,
    horizon: int,
    num_input_channels: int = 1,
) -> PatchTSTForPrediction:
    """
    Probabilistic PatchTST with Student-t output head and NLL loss.

    `patch_stride = patch_length // 2` (50% overlap) per Nie et al. (2023).
    """
    cfg = PatchTSTConfig(
        num_input_channels  = num_input_channels,
        context_length      = context_length,
        prediction_length   = horizon,
        patch_length        = trial.patch_length,
        patch_stride        = max(1, trial.patch_length // 2),
        d_model             = trial.d_model,
        num_attention_heads = trial.num_attention_heads,
        num_hidden_layers   = trial.num_hidden_layers,
        ffn_dim             = trial.d_model * 4,
        dropout             = trial.dropout,
        head_dropout        = trial.dropout,
        attention_dropout   = trial.dropout,
        loss                = "nll",
        distribution_output = "student_t",
        scaling             = "std",
        num_parallel_samples = NUM_PARALLEL_SAMPLES,
    )
    return PatchTSTForPrediction(cfg)


# ==============================================================================
#  VRAM PROBING & AUTO BATCH-SIZE
# ==============================================================================

def scale_lr(base_lr: float, bs: int, bs_ref: int = BS_REFERENCE,
             rule: str = LR_SCALING_RULE) -> float:
    """
    Rescale a reference learning rate (calibrated for bs_ref) to a new batch size.

    sqrt rule    : Krizhevsky (2014); standard for Adam/AdamW.
    linear rule  : Goyal et al. (2017); derived for SGD.
    """
    if rule == "sqrt":
        return float(base_lr * math.sqrt(bs / bs_ref))
    if rule == "linear":
        return float(base_lr * (bs / bs_ref))
    raise ValueError(f"Unknown LR scaling rule: {rule}")


def probe_trial_vram(
    trial: "TrialConfig",
    context_length: int,
    horizon: int,
    batch_size: int,
    device: str,
) -> Optional[float]:
    """
    Empirically measure peak VRAM (in GB) of one fwd+bwd+optimizer step.

    The probe is run on `device`. We assume the GPUs in the cluster are
    homogeneous; if they aren't, peak VRAM is still a good upper-bound
    estimate (the model graph is the same).

    Returns
    -------
    peak_gb : float on success, None on OOM.
    """
    if not device.startswith("cuda"):
        return 0.0
    try:
        with torch.cuda.device(torch.device(device)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_patchtst(trial, context_length, horizon).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(batch_size, context_length, 1, device=device)
            y = torch.randn(batch_size, horizon, 1, device=device)
            out = model(past_values=x, future_values=y)
            out.loss.backward()
            opt.step()
            torch.cuda.synchronize(torch.device(device))
            peak_bytes = torch.cuda.max_memory_allocated()
            del model, opt, x, y, out
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return peak_bytes / (1024 ** 3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return None


def auto_select_bs_lr(
    trial: "TrialConfig",
    context_length: int, horizon: int, probe_device: str, budget_gb: float,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Pick the largest batch size from BS_CANDIDATES whose peak VRAM fits inside
    `budget_gb` (probed on `probe_device`), then scale `trial.learning_rate`
    accordingly.

    Returns (bs, lr_scaled, peak_vram_gb) or (None, None, None) if even the
    smallest candidate doesn't fit.
    """
    for bs in BS_CANDIDATES:
        peak = probe_trial_vram(trial, context_length, horizon, bs, probe_device)
        if peak is None:
            continue                     # OOM — try smaller
        if peak <= budget_gb:
            return bs, scale_lr(trial.learning_rate, bs), peak
    return None, None, None


# ==============================================================================
#  TRAINING / VALIDATION
# ==============================================================================

@dataclass
class TrialSlot:
    """Per-trial runtime state inside a concurrent training group."""
    trial_idx: int
    cfg:       "TrialConfig"
    bs:        int
    lr:        float
    peak_vram_gb: float
    device:    str = "cuda:0"   # set by the scheduler at assignment time
    model:     Optional[Any] = None
    optimizer: Optional[Any] = None
    stream:    Optional[Any] = None
    best_val:  float = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left: int = EARLY_STOPPING_PATIENCE
    history:   Dict[str, Any] = field(default_factory=lambda:
                                      {"train_loss": [], "val_loss": [], "val_epochs": []})
    active:    bool = True
    failed:    bool = False


def _evaluate_on_gpu(model, x_val_gpu, y_val_gpu, bs: int) -> float:
    """Validation loss over x_val_gpu, y_val_gpu in fixed mini-batches."""
    model.eval()
    total_loss, total_n = 0.0, 0
    n_val = x_val_gpu.shape[0]
    with torch.no_grad():
        for start in range(0, n_val, bs):
            end = min(start + bs, n_val)
            x = x_val_gpu[start:end]
            y = y_val_gpu[start:end]
            out = model(past_values=x, future_values=y)
            cnt = x.shape[0]
            total_loss += out.loss.item() * cnt
            total_n    += cnt
    return total_loss / max(total_n, 1)


def train_trial_group(
    slots: List[TrialSlot],
    x_train_per_dev: Dict[str, torch.Tensor],
    y_train_per_dev: Dict[str, torch.Tensor],
    x_val_per_dev:   Dict[str, torch.Tensor],
    y_val_per_dev:   Dict[str, torch.Tensor],
    context_length: int, horizon: int,
) -> List[Dict[str, Any]]:
    """
    Train a list of K trials concurrently, possibly across multiple GPUs.

    Each slot owns:
      * its `device` (set by the scheduler — slots may be distributed across
        several GPUs);
      * its own CUDA stream on that device (or default stream when K==1);
      * its own model, optimizer and per-step state.

    Streams on the same GPU give kernel-level interleaving (the CUDA scheduler
    fills free SMs); streams on different GPUs run truly in parallel on
    independent hardware. The Python driver loop dispatches work to all GPUs
    in interleaved order, and CUDA executes asynchronously.

    The training tensors are sharded by device — each GPU holds its own copy.
    Read-only access from independent streams on the same device is safe.
    """
    K = len(slots)
    devices_used = sorted({s.device for s in slots})
    n_train = next(iter(x_train_per_dev.values())).shape[0]

    # -- Per-slot setup --------------------------------------------------------
    for s in slots:
        is_cuda = s.device.startswith("cuda")
        if is_cuda:
            with torch.cuda.device(torch.device(s.device)):
                s.model = build_patchtst(s.cfg, context_length, horizon).to(s.device)
                s.optimizer = torch.optim.AdamW(
                    s.model.parameters(), lr=s.lr, weight_decay=WEIGHT_DECAY,
                )
                # K=1 on a single device uses the default stream (cheaper).
                s.stream = (torch.cuda.Stream(device=torch.device(s.device))
                            if K > 1 else None)
        else:
            s.model = build_patchtst(s.cfg, context_length, horizon).to(s.device)
            s.optimizer = torch.optim.AdamW(
                s.model.parameters(), lr=s.lr, weight_decay=WEIGHT_DECAY,
            )
            s.stream = None

    for epoch in range(1, MAX_EPOCHS + 1):
        if not any(s.active and not s.failed for s in slots):
            break

        # Per-trial shuffled indices live on the slot's own device.
        epoch_perms = [torch.randperm(n_train, device=s.device) for s in slots]
        steps_per_trial = [n_train // s.bs for s in slots]
        max_steps     = max(steps_per_trial)
        run_loss_t = [torch.zeros((), device=s.device) for s in slots]
        run_n      = [0] * K

        for s in slots:
            s.model.train()

        for step in range(max_steps):
            for i, slot in enumerate(slots):
                if not slot.active or slot.failed:           continue
                if step >= steps_per_trial[i]:               continue

                idx = epoch_perms[i][step * slot.bs : (step + 1) * slot.bs]

                # Set device + stream context (no-op on CPU).
                is_cuda = slot.device.startswith("cuda")
                dev_ctx    = (torch.cuda.device(torch.device(slot.device))
                              if is_cuda else _NullCtx())
                stream_ctx = (torch.cuda.stream(slot.stream)
                              if slot.stream is not None else _NullCtx())
                try:
                    with dev_ctx, stream_ctx:
                        x = x_train_per_dev[slot.device].index_select(0, idx)
                        y = y_train_per_dev[slot.device].index_select(0, idx)
                        out = slot.model(past_values=x, future_values=y)
                        loss = out.loss
                        slot.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(slot.model.parameters(), GRAD_CLIP)
                        slot.optimizer.step()
                        # Accumulate on-device (no host sync).
                        run_loss_t[i] = run_loss_t[i] + loss.detach() * slot.bs
                        run_n[i]     += slot.bs
                except torch.cuda.OutOfMemoryError:
                    slot.failed = True
                    print(Fore.RED + f"    [trial {slot.trial_idx:03d} @ "
                          f"{slot.device}] OOM at step {step}/{epoch} — drop"
                          + Fore.RESET)
                    if is_cuda:
                        with torch.cuda.device(torch.device(slot.device)):
                            torch.cuda.empty_cache()
                except Exception as exc:
                    slot.failed = True
                    print(Fore.RED + f"    [trial {slot.trial_idx:03d} @ "
                          f"{slot.device}] step error: {exc} — drop" + Fore.RESET)

        # Single host sync per used device — only now do we touch host.
        for d in devices_used:
            _sync_device(d)

        # -- Validation & early-stopping per trial -----------------------------
        for i, slot in enumerate(slots):
            if slot.failed or not slot.active:
                continue
            train_loss = (run_loss_t[i].item() / max(run_n[i], 1)) if run_n[i] > 0 else float("nan")
            slot.history["train_loss"].append(train_loss)

            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS:
                # Validation on slot's device, against the local val copy.
                is_cuda = slot.device.startswith("cuda")
                dev_ctx = (torch.cuda.device(torch.device(slot.device))
                           if is_cuda else _NullCtx())
                with dev_ctx:
                    val_loss = _evaluate_on_gpu(
                        slot.model,
                        x_val_per_dev[slot.device],
                        y_val_per_dev[slot.device],
                        slot.bs,
                    )
                slot.history["val_loss"].append(val_loss)
                slot.history["val_epochs"].append(epoch)
                slot.model.train()
                tag = (f"[trial {slot.trial_idx:03d} @ {slot.device} "
                       f"bs={slot.bs} lr={slot.lr:.2e}]")
                print(Fore.YELLOW
                      + f"    {tag} epoch {epoch:3d}  "
                      + f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                      + Fore.RESET)
                if val_loss < slot.best_val:
                    slot.best_val = val_loss
                    slot.best_state = {k: v.detach().cpu().clone()
                                       for k, v in slot.model.state_dict().items()}
                    slot.patience_left = EARLY_STOPPING_PATIENCE
                else:
                    slot.patience_left -= 1
                    if slot.patience_left <= 0:
                        slot.active = False
                        print(Fore.MAGENTA
                              + f"    {tag} early stop at epoch {epoch}"
                              + Fore.RESET)

    # -- Gather results & free GPU state ---------------------------------------
    results = []
    for slot in slots:
        results.append({
            "trial_idx":  slot.trial_idx,
            "val_loss":   float(slot.best_val) if not slot.failed else float("nan"),
            "history":    slot.history,
            "best_state": slot.best_state,
            "failed":     slot.failed,
            "device":     slot.device,
        })
        del slot.model, slot.optimizer
        slot.model = slot.optimizer = slot.stream = None

    for d in devices_used:
        if d.startswith("cuda"):
            with torch.cuda.device(torch.device(d)):
                torch.cuda.empty_cache()
    return results


class _NullCtx:
    """No-op context manager for the K=1 (default-stream) path."""
    def __enter__(self):  return self
    def __exit__(self, *a): return False


# ==============================================================================
#  PLOTTING — random-search summary
# ==============================================================================

def plot_random_search_summary(
    trials_df: pd.DataFrame, save_path: str, dataset_display: str, term: str
):
    """
    Scatter of all trials + running-best curve (only points that improved upon
    all previous trials are connected; others are merely scattered).
    """
    trials_df = trials_df.sort_values("trial_idx").reset_index(drop=True)
    running_best = []
    best_so_far = float("inf")
    best_idx_in_df = []
    for i, row in trials_df.iterrows():
        v = row["val_loss"]
        if v < best_so_far:
            best_so_far = v
            best_idx_in_df.append(i)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(
        trials_df["trial_idx"], trials_df["val_loss"],
        s=36, alpha=0.55, color="#1f77b4",
        edgecolor="white", linewidth=0.5, label="trial",
    )
    improved = trials_df.iloc[best_idx_in_df]
    ax.plot(
        improved["trial_idx"], improved["val_loss"],
        marker="o", markersize=9, linewidth=2.2,
        color="#d62728", label="best-so-far",
    )
    best_row = trials_df.iloc[trials_df["val_loss"].idxmin()]
    ax.annotate(
        f"  best: trial {int(best_row['trial_idx'])}\n"
        f"  val_loss={best_row['val_loss']:.4f}",
        xy=(best_row["trial_idx"], best_row["val_loss"]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3b0", ec="#999", alpha=0.95),
    )
    ax.set_xlabel("Trial index", fontsize=12)
    ax.set_ylabel("Validation NLL", fontsize=12)
    ax.set_title(
        f"Random search — {dataset_display}  (term={term}, N={len(trials_df)})",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


# ==============================================================================
#  CACHE HELPERS
# ==============================================================================

def _dataset_dir(dataset_display: str, term: str) -> str:
    return os.path.join(CACHE_ROOT, dataset_display, term)


def _trial_path(dataset_display: str, term: str, trial_idx: int) -> str:
    return os.path.join(
        _dataset_dir(dataset_display, term), "trials", f"trial_{trial_idx:03d}.json"
    )


def _best_model_path(dataset_display: str, term: str) -> str:
    return os.path.join(_dataset_dir(dataset_display, term), "best_model.pt")


def _best_config_path(dataset_display: str, term: str) -> str:
    return os.path.join(_dataset_dir(dataset_display, term), "best_config.json")


def _load_trial_result(dataset_display: str, term: str, trial_idx: int) -> Optional[Dict]:
    p = _trial_path(dataset_display, term, trial_idx)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_trial_result(dataset_display: str, term: str, trial_idx: int, result: Dict):
    p = _trial_path(dataset_display, term, trial_idx)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(result, f, indent=2)


# ==============================================================================
#  MAIN
# ==============================================================================

def _process_group_results(
    slots: List[TrialSlot],
    results: List[Dict[str, Any]],
    elapsed_total: float,
    dataset_display: str, term: str, horizon: int,
    summary_rows: list,
    global_best_state: Dict[str, Any],
):
    """
    Persist per-trial JSONs and update the GLOBAL best checkpoint based on the
    results of one concurrent training group. `global_best_state` is a mutable
    dict {"val": float, "ckpt_path": ..., "cfg_path": ...} updated in place.
    """
    # Time is split evenly across slots for accounting purposes (the streams
    # ran concurrently so per-trial wall-clock isn't meaningfully separable).
    elapsed_per = elapsed_total / max(len(slots), 1)
    slot_by_idx = {s.trial_idx: s for s in slots}

    for r in results:
        slot = slot_by_idx[r["trial_idx"]]
        val_loss = r["val_loss"]

        result_record = {
            "trial_idx":      slot.trial_idx,
            **asdict(slot.cfg),
            "val_loss":       float(val_loss) if not np.isnan(val_loss) else float("nan"),
            "elapsed_seconds": round(elapsed_per, 2),
            "auto_batch_size": slot.bs,
            "auto_lr":         slot.lr,
            "peak_vram_gb":    round(slot.peak_vram_gb, 3),
            "device":          slot.device,
            "history":         r["history"],
            "failed":          r["failed"],
        }
        _save_trial_result(dataset_display, term, slot.trial_idx, result_record)
        summary_rows.append({
            "dataset_display": dataset_display, "term": term, **result_record,
        })

        # -- Update global best ------------------------------------------------
        if (r["best_state"] is not None and not np.isnan(val_loss)
                and val_loss < global_best_state["val"]):
            global_best_state["val"] = val_loss
            torch.save(r["best_state"], global_best_state["ckpt_path"])
            with open(global_best_state["cfg_path"], "w") as f:
                json.dump({
                    "trial_idx": slot.trial_idx,
                    **asdict(slot.cfg),
                    "val_loss":           float(val_loss),
                    "auto_batch_size":    slot.bs,
                    "auto_lr":            slot.lr,
                    "context_length":     CONTEXT_LENGTH,
                    "prediction_length":  horizon,
                    "num_input_channels": 1,
                }, f, indent=2)
            print(Fore.GREEN
                  + f"    NEW BEST: trial {slot.trial_idx:03d}  "
                  + f"val_loss={val_loss:.4f} -> weights saved" + Fore.RESET)


def main():
    set_seed(SEED)
    devices = resolve_devices()
    probe_device = devices[0]
    for d in devices:
        if d.startswith("cuda"):
            with torch.cuda.device(torch.device(d)):
                torch.set_float32_matmul_precision("high")
    print(Fore.CYAN + f"Devices: {devices}  (probe on {probe_device})" + Fore.RESET)
    print(Fore.CYAN
          + f"Concurrent training: {ENABLE_CONCURRENT_TRAINING}  "
          + f"(K per GPU={MAX_CONCURRENT_TRIALS}, "
          + f"total parallel slots={MAX_CONCURRENT_TRIALS * len(devices)}, "
          + f"slot budget={VRAM_BUDGET_PER_SLOT_GB} GB, "
          + f"solo budget={VRAM_BUDGET_SOLO_GB} GB)"
          + Fore.RESET)
    print(Fore.CYAN
          + f"LR scaling rule: {LR_SCALING_RULE}  (reference bs={BS_REFERENCE})"
          + Fore.RESET)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CACHE_ROOT, f"runs/{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    trial_configs = sample_trial_configs(N_TRIALS, seed=SEED)
    print(Fore.CYAN + f"Sampled {len(trial_configs)} unique trial configs" + Fore.RESET)

    summary_rows = []
    per_gpu_capacity = MAX_CONCURRENT_TRIALS if ENABLE_CONCURRENT_TRAINING else 1
    total_capacity   = per_gpu_capacity * len(devices)

    for ge_name, term, dataset_display, to_univariate in DATASETS:
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  DATASET: {ge_name}  term={term}  ({dataset_display})" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        ge_dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
        horizon = ge_dataset.prediction_length
        print(Fore.CYAN
              + f"  freq={ge_dataset.freq}  horizon={horizon}  "
              + f"target_dim={ge_dataset.target_dim}" + Fore.RESET)

        # -- Build train/val tensors and replicate on each device --------------
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = build_train_val_tensors(
            ge_dataset, CONTEXT_LENGTH, horizon,
        )
        print(Fore.CYAN
              + f"  train: x={tuple(x_train_cpu.shape)}  y={tuple(y_train_cpu.shape)}  |  "
              + f"val: x={tuple(x_val_cpu.shape)}  y={tuple(y_val_cpu.shape)}"
              + Fore.RESET)
        x_train_per_dev = {d: x_train_cpu.to(d, non_blocking=True) for d in devices}
        y_train_per_dev = {d: y_train_cpu.to(d, non_blocking=True) for d in devices}
        x_val_per_dev   = {d: x_val_cpu.to(d,   non_blocking=True) for d in devices}
        y_val_per_dev   = {d: y_val_cpu.to(d,   non_blocking=True) for d in devices}
        del x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu

        ds_dir = _dataset_dir(dataset_display, term)
        os.makedirs(os.path.join(ds_dir, "trials"), exist_ok=True)

        # -- Bootstrap "global best" from any already-saved checkpoint ---------
        global_best_state = {
            "val":       float("inf"),
            "ckpt_path": _best_model_path(dataset_display, term),
            "cfg_path":  _best_config_path(dataset_display, term),
        }
        if os.path.isfile(global_best_state["cfg_path"]):
            try:
                with open(global_best_state["cfg_path"]) as f:
                    prev = json.load(f)
                global_best_state["val"] = float(prev.get("val_loss", float("inf")))
                print(Fore.WHITE
                      + f"  Existing best on disk: val_loss={global_best_state['val']:.4f} "
                      + f"(trial {prev.get('trial_idx', '?')})" + Fore.RESET)
            except Exception:
                pass

        # -- Scheduler: per-device pending buffers -----------------------------
        pending: Dict[str, List[TrialSlot]] = {d: [] for d in devices}

        def _all_pending() -> List[TrialSlot]:
            out = []
            for d in devices:
                out.extend(pending[d])
            return out

        def _flush_all():
            """Train all currently buffered slots (across all GPUs), persist, clear."""
            slots = _all_pending()
            if not slots:
                return
            ids_by_dev = {d: [s.trial_idx for s in pending[d]] for d in devices
                          if pending[d]}
            total_vram = sum(s.peak_vram_gb for s in slots)
            print(Fore.CYAN
                  + f"\n  >> FLUSH  K_total={len(slots)}  "
                  + f"distribution={ids_by_dev}  "
                  + f"sum_peak_vram≈{total_vram:.2f} GB" + Fore.RESET)
            t0 = time.perf_counter()
            try:
                results = train_trial_group(
                    slots, x_train_per_dev, y_train_per_dev,
                    x_val_per_dev, y_val_per_dev, CONTEXT_LENGTH, horizon,
                )
            except Exception as exc:
                print(Fore.RED + f"    GROUP FAILED: {exc}" + Fore.RESET)
                results = [{
                    "trial_idx":  s.trial_idx, "val_loss":   float("nan"),
                    "history":    {}, "best_state": None,
                    "failed":     True, "device": s.device,
                } for s in slots]
            elapsed = time.perf_counter() - t0
            print(Fore.MAGENTA + f"    GROUP TIME {elapsed:.1f}s" + Fore.RESET)
            _process_group_results(
                slots, results, elapsed,
                dataset_display, term, horizon,
                summary_rows, global_best_state,
            )
            for d in devices:
                pending[d] = []

        def _assign_device() -> str:
            """Pick the GPU with the shortest pending queue (round-robin)."""
            return min(devices, key=lambda d: len(pending[d]))

        for trial_idx, trial in enumerate(trial_configs):
            cached = _load_trial_result(dataset_display, term, trial_idx)
            if cached is not None and "val_loss" in cached \
               and not np.isnan(cached["val_loss"]):
                print(Fore.WHITE + f"  CACHED  trial {trial_idx:03d}: "
                      + f"val_loss={cached['val_loss']:.4f}" + Fore.RESET)
                summary_rows.append({
                    "dataset_display": dataset_display, "term": term,
                    "trial_idx": trial_idx, **cached,
                })
                continue

            print(Fore.YELLOW
                  + f"\n  > trial {trial_idx:03d} / {len(trial_configs)-1}: "
                  + f"{asdict(trial)}" + Fore.RESET)

            # -- Probe (on the designated probe device) -----------------------
            bs, lr, peak = auto_select_bs_lr(
                trial, CONTEXT_LENGTH, horizon, probe_device,
                budget_gb=VRAM_BUDGET_PER_SLOT_GB,
            )
            if bs is not None and ENABLE_CONCURRENT_TRAINING:
                target_dev = _assign_device()
                print(Fore.CYAN
                      + f"    probe -> bs={bs}  lr={lr:.2e}  peak={peak:.2f} GB  "
                      + f"PACKABLE -> {target_dev}" + Fore.RESET)
                pending[target_dev].append(TrialSlot(
                    trial_idx=trial_idx, cfg=trial, bs=bs, lr=lr,
                    peak_vram_gb=peak, device=target_dev,
                ))
                # Flush when total reaches K*N, OR when any single GPU is full
                # (the latter handles imbalanced assignment, e.g. all-cached
                # trials skewing the round-robin).
                total_now = len(_all_pending())
                if total_now >= total_capacity \
                   or any(len(p) >= per_gpu_capacity for p in pending.values()):
                    _flush_all()
                continue

            # -- Fall back to solo budget --------------------------------------
            bs, lr, peak = auto_select_bs_lr(
                trial, CONTEXT_LENGTH, horizon, probe_device,
                budget_gb=VRAM_BUDGET_SOLO_GB,
            )
            if bs is None:
                print(Fore.RED
                      + f"    probe FAILED: even bs={BS_CANDIDATES[-1]} doesn't "
                      + f"fit in {VRAM_BUDGET_SOLO_GB} GB. Skipping trial."
                      + Fore.RESET)
                _save_trial_result(dataset_display, term, trial_idx, {
                    "trial_idx": trial_idx, **asdict(trial),
                    "val_loss": float("nan"), "elapsed_seconds": 0.0,
                    "history": {}, "failed": True,
                    "skip_reason": "vram_oom",
                })
                continue

            # Solo trials need a dedicated GPU. Flush any pending packables
            # first (they'd compete for VRAM), then run the solo trial.
            target_dev = _assign_device()
            print(Fore.CYAN
                  + f"    probe -> bs={bs}  lr={lr:.2e}  peak={peak:.2f} GB  "
                  + f"SOLO -> {target_dev}" + Fore.RESET)
            if _all_pending():
                _flush_all()
            pending[target_dev].append(TrialSlot(
                trial_idx=trial_idx, cfg=trial, bs=bs, lr=lr,
                peak_vram_gb=peak, device=target_dev,
            ))
            _flush_all()

        # -- Final flush at end of dataset -------------------------------------
        if _all_pending():
            _flush_all()

        # -- Free dataset-level GPU memory -------------------------------------
        del x_train_per_dev, y_train_per_dev, x_val_per_dev, y_val_per_dev
        for d in devices:
            if d.startswith("cuda"):
                with torch.cuda.device(torch.device(d)):
                    torch.cuda.empty_cache()

        # -- Per-dataset summary plot + CSV ------------------------------------
        df_ds = pd.DataFrame([r for r in summary_rows
                              if r["dataset_display"] == dataset_display
                              and r["term"] == term])
        keep_cols = ["trial_idx", "val_loss", "patch_length", "d_model",
                     "num_hidden_layers", "num_attention_heads",
                     "dropout", "learning_rate",
                     "auto_batch_size", "auto_lr", "peak_vram_gb",
                     "device", "elapsed_seconds"]
        df_ds = df_ds[[c for c in keep_cols if c in df_ds.columns]]
        csv_path = os.path.join(ds_dir, "random_search.csv")
        df_ds.to_csv(csv_path, index=False)
        plot_random_search_summary(
            df_ds.dropna(subset=["val_loss"]),
            os.path.join(ds_dir, "random_search_summary.png"),
            dataset_display, term,
        )
        print(Fore.GREEN + f"  Summary CSV: {csv_path}" + Fore.RESET)

    # -- Global summary --------------------------------------------------------
    global_df = pd.DataFrame(summary_rows)
    global_csv = os.path.join(run_dir, "random_search_all.csv")
    global_df.drop(columns=["history"], errors="ignore").to_csv(global_csv, index=False)
    print(Fore.GREEN + f"\n  Global summary: {global_csv}" + Fore.RESET)

    print(Fore.GREEN + "\nTraining pipeline done." + Fore.RESET)


if __name__ == "__main__":
    main()