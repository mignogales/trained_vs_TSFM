"""
PatchTST training pipeline for GiftEval datasets — random hyperparameter search.

Architecture
------------
* Process-per-GPU parallelism: one Python worker per device, draining trials
  from a shared multiprocessing.Queue. Each GPU runs independently — no GIL
  contention, no inter-GPU dispatch serialization.
* Per-GPU VRAM budgets: heterogeneous clusters declare one budget per device.
* Auto batch-size: the largest bs in BS_CANDIDATES whose probed peak VRAM fits
  the device's budget, with LR rescaled via the sqrt rule (Adam-safe).
* On-device validation accumulator: one host sync per validation pass instead
  of one per mini-batch.
* Pinned CPU→GPU upload on worker startup.

Pipeline
--------
For each (dataset, term) in DATASETS:
  1.  Load GiftEval Dataset(term=..., to_univariate=...).
  2.  Split each training series temporally: 80/90 → train, 10/90 → val.
  3.  Build sliding-window (CONTEXT_LENGTH, H) tensors.
  4.  Random search of N_TRIALS configs from HP_SPACE.
  5.  Each trial: VRAM probe → auto batch-size + LR scaling → train →
      validation every VAL_EVERY_N_EPOCHS → early stopping. Save only the
      weights with best validation loss across all trials.
  6.  Plot scatter + running best.

Cache layout
------------
logs/experiments/patchtst_training/<dataset>/<term>/
        best_model.pt
        best_config.json
        trials/trial_<NNN>.json
        random_search_summary.png
        random_search.csv
"""

import torch
import torch.multiprocessing as mp
import os
import json
import math
import time
import random
import shutil
import tempfile
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from queue import Empty
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore

from dotenv import load_dotenv
load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction

class InsufficientDataError(RuntimeError):
    """Dataset cannot yield a train/val split at the requested context+horizon."""

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
BS_REFERENCE           = 128
BS_CANDIDATES          = [2048, 1536, 1024, 512, 256, 128, 64]   # tried in descending order
LR_SCALING_RULE        = "sqrt"        # "sqrt" or "linear"

# -- Multi-GPU configuration ---------------------------------------------------
# DEVICES: which GPUs to use.
#   None        -> auto-detect all visible GPUs.
#   ["cuda:0"]  -> force a single GPU.
#   ["cuda:0", "cuda:2"] -> use explicit subset.
DEVICES = None

# VRAM budget per GPU (GB), aligned with resolved DEVICES order.
# Use None for a single global budget (VRAM_BUDGET_DEFAULT_GB) applied to all.
# Heterogeneous example: Titan V (12) / Titan Xp (12) / RTX 2080 (8)
#                       → [10.0, 10.0, 6.0]   (leave ~2 GB headroom each)
VRAM_BUDGET_GB_PER_DEVICE: Optional[List[float]] = [10.0, 6.0, 10.0]
VRAM_BUDGET_DEFAULT_GB = 10.0

# -- Random search HP space ----------------------------------------------------
HP_SPACE = {
    "patch_length": [8, 16, 32],
    "patch_stride": [4, 8, 16],
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
    """Resolve devices honoring CUDA_VISIBLE_DEVICES + the DEVICES override."""
    if not torch.cuda.is_available():
        return ["cpu"]
    if DEVICES is not None:
        return list(DEVICES)
    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"]
    return [f"cuda:{i}" for i in range(n)]


def resolve_vram_budgets(devices: List[str]) -> List[float]:
    """Resolve per-device VRAM budgets from VRAM_BUDGET_GB_PER_DEVICE."""
    if VRAM_BUDGET_GB_PER_DEVICE is None:
        return [VRAM_BUDGET_DEFAULT_GB] * len(devices)
    if len(VRAM_BUDGET_GB_PER_DEVICE) != len(devices):
        raise ValueError(
            f"VRAM_BUDGET_GB_PER_DEVICE has {len(VRAM_BUDGET_GB_PER_DEVICE)} "
            f"entries but {len(devices)} devices were resolved: {devices}."
        )
    return list(VRAM_BUDGET_GB_PER_DEVICE)


def _sync_device(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


# ==============================================================================
#  DATA BUILDING
# ==============================================================================

def _series_train_val_split(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split first-90% series into 80/90 train + 10/90 val (orig. 80% / 10%)."""
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
    max_val_windows:   int = MAX_VAL_WINDOWS,
    rng_seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build train/val tensors with a two-stage strategy:

    Stage 1 — standard 80/10 temporal split per series (the original behavior).
    Stage 2 — fallback (only triggered if stage 1 produces ZERO val windows):
              for every series with len >= context+horizon, reserve the
              trailing (context+horizon) samples as a single val window and
              build sliding train windows from all preceding samples.

    Raises
    ------
    InsufficientDataError
        If neither stage yields at least one train window AND one val window.
    """
    rng = np.random.RandomState(rng_seed)
    entries = list(ge_dataset.training_dataset)
    rng.shuffle(entries)

    # Pre-clean targets once so both stages share the same arrays.
    cleaned: List[np.ndarray] = []
    for entry in entries:
        tgt = entry["target"]
        if tgt.ndim > 1:
            raise ValueError(
                f"Expected univariate target but got shape {tgt.shape}. "
                "Use to_univariate=True when loading the dataset."
            )
        cleaned.append(np.nan_to_num(
            np.asarray(tgt, dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        ))

    win = context_length + horizon

    # ------------------------------------------------------------------ STAGE 1
    x_tr_list, y_tr_list, x_vl_list, y_vl_list = [], [], [], []
    n_train_total = 0
    n_val_total   = 0

    for tgt in cleaned:
        train_portion, val_portion = _series_train_val_split(tgt)

        if n_train_total < max_train_windows:
            remaining = max_train_windows - n_train_total
            x_tr, y_tr = _sliding_windows(
                train_portion, context_length, horizon,
                max_windows=remaining, rng=rng,
            )
            if len(x_tr) > 0:
                x_tr_list.append(x_tr); y_tr_list.append(y_tr)
                n_train_total += len(x_tr)

        if n_val_total < max_val_windows:
            remaining = max_val_windows - n_val_total
            x_vl, y_vl = _sliding_windows(
                val_portion, context_length, horizon,
                max_windows=remaining, rng=rng,
            )
            if len(x_vl) > 0:
                x_vl_list.append(x_vl); y_vl_list.append(y_vl)
                n_val_total += len(x_vl)

        if (n_train_total >= max_train_windows
                and n_val_total >= max_val_windows):
            break

    # ------------------------------------------------------------------ STAGE 2
    # Trigger the fallback only when the standard split failed to yield ANY
    # validation window. We deliberately rebuild train as well so the val
    # window's source range is excluded from the train pool (no temporal leak).
    if not x_vl_list:
        print(Fore.YELLOW
              + "  [build_train_val_tensors] standard 10% val split produced "
              + "0 val windows — falling back to trailing-window-per-series."
              + Fore.RESET)
        x_tr_list, y_tr_list, x_vl_list, y_vl_list = [], [], [], []
        n_train_total = 0
        n_val_total   = 0
        rng = np.random.RandomState(rng_seed)   # deterministic re-run

        n_series_skipped_short = 0
        for tgt in cleaned:
            L = len(tgt)
            if L < win:
                n_series_skipped_short += 1
                continue

            # Trailing single val window — last `win` samples of the series.
            val_x = tgt[L - win : L - horizon]    # context part
            val_y = tgt[L - horizon : L]          # forecast target
            if n_val_total < max_val_windows:
                x_vl_list.append(val_x[None, :].astype(np.float32))
                y_vl_list.append(val_y[None, :].astype(np.float32))
                n_val_total += 1

            # Train pool: strictly everything before the val window.
            train_portion = tgt[: L - win]
            if n_train_total < max_train_windows and len(train_portion) >= win:
                remaining = max_train_windows - n_train_total
                x_tr, y_tr = _sliding_windows(
                    train_portion, context_length, horizon,
                    max_windows=remaining, rng=rng,
                )
                if len(x_tr) > 0:
                    x_tr_list.append(x_tr); y_tr_list.append(y_tr)
                    n_train_total += len(x_tr)

            if (n_train_total >= max_train_windows
                    and n_val_total >= max_val_windows):
                break

        print(Fore.YELLOW
              + f"  [build_train_val_tensors] fallback summary: "
              + f"n_train_windows={n_train_total}  "
              + f"n_val_windows={n_val_total}  "
              + f"series_too_short={n_series_skipped_short}"
              + Fore.RESET)

    # ----------------------------------------------------------- HARD FAILURE
    if not x_tr_list or not x_vl_list:
        raise InsufficientDataError(
            f"Cannot build train/val tensors for context_length={context_length} "
            f"and horizon={horizon}: train_windows={n_train_total}, "
            f"val_windows={n_val_total} after fallback. "
            f"All series shorter than 2*(context+horizon)={2*win}."
        )

    x_train = torch.from_numpy(np.concatenate(x_tr_list, axis=0)).unsqueeze(-1)
    y_train = torch.from_numpy(np.concatenate(y_tr_list, axis=0)).unsqueeze(-1)
    x_val   = torch.from_numpy(np.concatenate(x_vl_list, axis=0)).unsqueeze(-1)
    y_val   = torch.from_numpy(np.concatenate(y_vl_list, axis=0)).unsqueeze(-1)

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
    """Probabilistic PatchTST with Student-t output head and NLL loss."""
    cfg = PatchTSTConfig(
        num_input_channels   = num_input_channels,
        context_length       = context_length,
        prediction_length    = horizon,
        patch_length         = trial.patch_length,
        patch_stride         = max(1, trial.patch_length // 2),
        d_model              = trial.d_model,
        num_attention_heads  = trial.num_attention_heads,
        num_hidden_layers    = trial.num_hidden_layers,
        ffn_dim              = trial.d_model * 4,
        dropout              = trial.dropout,
        head_dropout         = trial.dropout,
        attention_dropout    = trial.dropout,
        loss                 = "nll",
        distribution_output  = "student_t",
        scaling              = "std",
        num_parallel_samples = NUM_PARALLEL_SAMPLES,
    )
    return PatchTSTForPrediction(cfg)


# ==============================================================================
#  VRAM PROBING & AUTO BATCH-SIZE
# ==============================================================================

def scale_lr(base_lr: float, bs: int, bs_ref: int = BS_REFERENCE,
             rule: str = LR_SCALING_RULE) -> float:
    """sqrt: Adam-safe (Krizhevsky 2014). linear: SGD (Goyal 2017)."""
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
    """Empirically measure peak VRAM (GB) of one fwd+bwd+step on `device`."""
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
    """Pick the largest bs in BS_CANDIDATES that fits in budget; scale LR."""
    for bs in BS_CANDIDATES:
        peak = probe_trial_vram(trial, context_length, horizon, bs, probe_device)
        if peak is None:
            continue
        if peak <= budget_gb:
            return bs, scale_lr(trial.learning_rate, bs), peak
    return None, None, None


# ==============================================================================
#  VALIDATION
# ==============================================================================

def _evaluate_on_gpu(model, x_val_gpu, y_val_gpu, bs: int) -> float:
    """
    Validation loss with a single host sync at the end.

    The naive `total_loss += out.loss.item()` pattern stalls the pipeline once
    per mini-batch: each `.item()` waits for the forward pass to finish, copies
    the scalar to host, and only then enqueues the next iteration. Here we
    keep a running tensor on device and call `.item()` exactly once at the end.
    """
    model.eval()
    device = x_val_gpu.device
    total_loss = torch.zeros((), device=device)
    total_n = 0
    n_val = x_val_gpu.shape[0]
    with torch.no_grad():
        for start in range(0, n_val, bs):
            end = min(start + bs, n_val)
            x = x_val_gpu[start:end]
            y = y_val_gpu[start:end]
            out = model(past_values=x, future_values=y)
            cnt = x.shape[0]
            total_loss = total_loss + out.loss * cnt
            total_n += cnt
    return (total_loss / max(total_n, 1)).item()


# ==============================================================================
#  PER-WORKER TRIAL EXECUTION
# ==============================================================================

def _run_single_trial(
    trial_idx: int,
    trial: TrialConfig,
    vram_budget_gb: float,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val:   torch.Tensor, y_val:   torch.Tensor,
    context_length: int, horizon: int, device: str,
) -> Dict[str, Any]:
    """Run one trial end-to-end on this worker's device."""

    # -- VRAM probe ----------------------------------------------------------
    bs, lr, peak = auto_select_bs_lr(
        trial, context_length, horizon, device, budget_gb=vram_budget_gb,
    )
    if bs is None:
        return {
            "trial_idx": trial_idx, "device": device,
            "val_loss": float("nan"),
            "history": {"train_loss": [], "val_loss": [], "val_epochs": []},
            "best_state_path": None, "failed": True,
            "skip_reason": "vram_oom",
            "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
            "elapsed_seconds": 0.0,
            "cfg": asdict(trial),
        }

    tag = f"[{device}] trial {trial_idx:03d}"
    print(Fore.CYAN
          + f"  {tag}: bs={bs} lr={lr:.2e} peak={peak:.2f}GB  "
          + f"budget={vram_budget_gb:.1f}GB" + Fore.RESET)

    t0 = time.perf_counter()
    n_train = x_train.shape[0]
    # Drop-last for consistent gradient noise across steps.
    steps_per_epoch = n_train // bs

    # -- Build model + optimizer --------------------------------------------
    model = build_patchtst(trial, context_length, horizon).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
    )

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = EARLY_STOPPING_PATIENCE
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_epochs": []}

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            run_loss = torch.zeros((), device=device)
            n_seen = 0

            for step in range(steps_per_epoch):
                idx = perm[step * bs : (step + 1) * bs]
                x = x_train.index_select(0, idx)
                y = y_train.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                out = model(past_values=x, future_values=y)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                # On-device accumulation: avoid per-step .item() syncs.
                run_loss = run_loss + loss.detach() * bs
                n_seen += bs

            train_loss = (run_loss / max(n_seen, 1)).item()  # one sync/epoch
            history["train_loss"].append(train_loss)

            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS:
                val_loss = _evaluate_on_gpu(model, x_val, y_val, bs)
                history["val_loss"].append(val_loss)
                history["val_epochs"].append(epoch)
                print(Fore.YELLOW
                      + f"  {tag} epoch {epoch:3d}  "
                      + f"train={train_loss:.4f}  val={val_loss:.4f}"
                      + Fore.RESET)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience_left = EARLY_STOPPING_PATIENCE
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(Fore.MAGENTA
                              + f"  {tag} early stop at epoch {epoch}"
                              + Fore.RESET)
                        break
    except torch.cuda.OutOfMemoryError as exc:
        print(Fore.RED + f"  {tag} OOM during training: {exc}" + Fore.RESET)
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0

    # -- Cleanup -------------------------------------------------------------
    del optimizer, model
    torch.cuda.empty_cache()

    # -- Persist best_state to a temp file -----------------------------------
    # We deliberately do NOT pass tensors through mp.Queue. PyTorch's default
    # tensor-sharing strategy uses Unix domain sockets owned by the SENDER
    # process; once a worker exits (e.g. after consuming its poison pill),
    # those sockets are torn down and the parent's queue.get() that triggers
    # the lazy tensor rebuild raises FileNotFoundError on the dead socket.
    # Writing the checkpoint to disk and sending only the path avoids this
    # entirely — the queue payload becomes pure JSON-compatible primitives.
    best_state_path: Optional[str] = None
    if best_state is not None:
        fd, best_state_path = tempfile.mkstemp(
            prefix=f"patchtst_w_trial{trial_idx:03d}_", suffix=".pt",
        )
        os.close(fd)
        torch.save(best_state, best_state_path)
        del best_state

    return {
        "trial_idx": trial_idx, "device": device,
        "val_loss": float(best_val) if best_val != float("inf") else float("nan"),
        "history": history, "best_state_path": best_state_path, "failed": False,
        "auto_batch_size": bs, "auto_lr": lr,
        "peak_vram_gb": round(peak, 3),
        "elapsed_seconds": round(elapsed, 2),
        "cfg": asdict(trial),
    }


# ==============================================================================
#  GPU WORKER PROCESS
# ==============================================================================

def gpu_worker(
    worker_id: int,
    device: str,
    vram_budget_gb: float,
    trial_queue: "mp.Queue",
    result_queue: "mp.Queue",
    dataset_args: Tuple[str, str, bool],
    context_length: int,
):
    """
    Worker process — owns one GPU, drains trials from the queue, returns
    results.

    Each worker:
      1. Sets its CUDA device.
      2. Builds (dataset, term) train/val tensors locally and uploads them to
         its GPU once (pinned host buffer → async H2D).
      3. Loops: pull (trial_idx, TrialConfig) from queue → train → push result.
      4. Exits on `None` poison pill or queue timeout.
    """
    # Per-worker seed differentiation: same trials, different stochastic
    # shuffling and weight init across workers.
    set_seed(SEED + worker_id)
    torch.cuda.set_device(torch.device(device))
    torch.set_float32_matmul_precision("high")

    ge_name, term, to_univariate = dataset_args

    # -- Build train/val tensors locally and upload to GPU once --------------
    try:
        ge_dataset = GiftEvalDataset(
            name=ge_name, term=term, to_univariate=to_univariate,
        )
        horizon = ge_dataset.prediction_length
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = build_train_val_tensors(
            ge_dataset, context_length, horizon,
        )
        x_train = x_train_cpu.pin_memory().to(device, non_blocking=True)
        y_train = y_train_cpu.pin_memory().to(device, non_blocking=True)
        x_val   = x_val_cpu  .pin_memory().to(device, non_blocking=True)
        y_val   = y_val_cpu  .pin_memory().to(device, non_blocking=True)
        del x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu
        torch.cuda.synchronize(torch.device(device))
        print(Fore.CYAN
              + f"  [{device}] worker {worker_id} ready: "
              + f"x_train={tuple(x_train.shape)} x_val={tuple(x_val.shape)}"
              + Fore.RESET)
    except InsufficientDataError as exc:
        # Dataset-level skip: signal the parent ONCE and drain the queue.
        print(Fore.RED + f"  [{device}] worker {worker_id} SKIP DATASET: "
              + f"{exc}" + Fore.RESET)
        result_queue.put({"_dataset_skip": True, "reason": str(exc),
                          "device": device})
        while True:
            try:
                msg = trial_queue.get(timeout=5)
            except Empty:
                break
            if msg is None:
                break
            # Drop trials silently; the parent already knows to skip.
        return
    except Exception as exc:
        print(Fore.RED + f"  [{device}] worker {worker_id} setup failed: "
              + f"{exc}" + Fore.RESET)
        while True:
            try:
                msg = trial_queue.get(timeout=5)
            except Empty:
                break
            if msg is None:
                break
            trial_idx, trial = msg
            result_queue.put({
                "trial_idx": trial_idx, "device": device,
                "val_loss": float("nan"),
                "history": {}, "best_state_path": None, "failed": True,
                "skip_reason": "worker_setup_failed",
                "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
                "elapsed_seconds": 0.0,
                "cfg": asdict(trial),
            })
        return

    # -- Main consumption loop ----------------------------------------------
    while True:
        try:
            msg = trial_queue.get(timeout=600)
        except Empty:
            print(Fore.MAGENTA + f"  [{device}] worker {worker_id} queue "
                  + "timeout — exiting." + Fore.RESET)
            break
        if msg is None:
            break
        trial_idx, trial = msg
        try:
            result = _run_single_trial(
                trial_idx, trial, vram_budget_gb,
                x_train, y_train, x_val, y_val,
                context_length, horizon, device,
            )
        except Exception as exc:
            print(Fore.RED + f"  [{device}] trial {trial_idx:03d} CRASHED: "
                  + f"{type(exc).__name__}: {exc}" + Fore.RESET)
            result = {
                "trial_idx": trial_idx, "device": device,
                "val_loss": float("nan"),
                "history": {}, "best_state_path": None, "failed": True,
                "skip_reason": f"trial_exception:{type(exc).__name__}",
                "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
                "elapsed_seconds": 0.0,
                "cfg": asdict(trial),
            }
            torch.cuda.empty_cache()
        result_queue.put(result)

    # -- Cleanup -------------------------------------------------------------
    del x_train, y_train, x_val, y_val
    torch.cuda.empty_cache()
    print(Fore.CYAN + f"  [{device}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  PLOTTING — random-search summary
# ==============================================================================

def plot_random_search_summary(
    trials_df: pd.DataFrame, save_path: str, dataset_display: str, term: str
):
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

def _handle_result(
    result: Dict[str, Any],
    dataset_display: str, term: str, horizon: int,
    summary_rows: List[Dict[str, Any]],
    global_best_state: Dict[str, Any],
):
    """Persist per-trial JSON and update the global best checkpoint."""
    trial_idx = result["trial_idx"]
    val_loss = result["val_loss"]
    cfg = result.get("cfg", {}) or {}

    result_record = {
        "trial_idx":        trial_idx,
        **cfg,
        "val_loss":         float(val_loss) if not np.isnan(val_loss)
                            else float("nan"),
        "elapsed_seconds":  result.get("elapsed_seconds", 0.0),
        "auto_batch_size":  result.get("auto_batch_size"),
        "auto_lr":          result.get("auto_lr"),
        "peak_vram_gb":     result.get("peak_vram_gb"),
        "device":           result.get("device"),
        "history":          result.get("history", {}),
        "failed":           result.get("failed", False),
    }
    if result.get("skip_reason"):
        result_record["skip_reason"] = result["skip_reason"]
    _save_trial_result(dataset_display, term, trial_idx, result_record)
    summary_rows.append({
        "dataset_display": dataset_display, "term": term, **result_record,
    })

    # -- Global best update -------------------------------------------------
    best_state_path = result.get("best_state_path")
    if (best_state_path is not None
            and os.path.isfile(best_state_path)
            and not np.isnan(val_loss)
            and val_loss < global_best_state["val"]):
        global_best_state["val"] = val_loss
        # File-level copy is faster than torch.load + torch.save and avoids
        # ever materializing the tensors in the parent's address space.
        shutil.copyfile(best_state_path, global_best_state["ckpt_path"])
        with open(global_best_state["cfg_path"], "w") as f:
            json.dump({
                "trial_idx": trial_idx,
                **cfg,
                "val_loss":           float(val_loss),
                "auto_batch_size":    result_record["auto_batch_size"],
                "auto_lr":            result_record["auto_lr"],
                "context_length":     CONTEXT_LENGTH,
                "prediction_length":  horizon,
                "num_input_channels": 1,
            }, f, indent=2)
        print(Fore.GREEN
              + f"  NEW BEST: trial {trial_idx:03d}  "
              + f"val_loss={val_loss:.4f} -> weights saved" + Fore.RESET)

    # -- Always clean up the worker's temp checkpoint -----------------------
    # Whether or not it became the global best, the temp file has served its
    # purpose. Failure to remove it (e.g. file already gone) is non-fatal.
    if best_state_path is not None and os.path.exists(best_state_path):
        try:
            os.remove(best_state_path)
        except OSError:
            pass


def main():
    # IMPORTANT: 'spawn' is required for CUDA + multiprocessing. Must be set
    # before any CUDA context is initialized in the parent. We don't touch
    # CUDA in the parent (only count devices), so this is safe here.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    set_seed(SEED)
    devices       = resolve_devices()
    vram_budgets  = resolve_vram_budgets(devices)

    print(Fore.CYAN + f"Devices: {devices}" + Fore.RESET)
    print(Fore.CYAN
          + f"VRAM budgets (GB) per device: "
          + f"{dict(zip(devices, vram_budgets))}" + Fore.RESET)
    print(Fore.CYAN
          + f"LR scaling: {LR_SCALING_RULE} (bs_ref={BS_REFERENCE})  |  "
          + f"BS candidates: {BS_CANDIDATES}"
          + Fore.RESET)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CACHE_ROOT, f"runs/{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    trial_configs = sample_trial_configs(N_TRIALS, seed=SEED)
    print(Fore.CYAN + f"Sampled {len(trial_configs)} unique trial configs"
          + Fore.RESET)

    summary_rows: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")

    for ge_name, term, dataset_display, to_univariate in DATASETS:
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN
              + f"  DATASET: {ge_name}  term={term}  ({dataset_display})"
              + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        # Quick metadata probe from parent (workers will rebuild tensors).
        ge_dataset_meta = GiftEvalDataset(
            name=ge_name, term=term, to_univariate=to_univariate,
        )
        horizon = ge_dataset_meta.prediction_length
        print(Fore.CYAN
              + f"  freq={ge_dataset_meta.freq}  horizon={horizon}  "
              + f"target_dim={ge_dataset_meta.target_dim}" + Fore.RESET)
        del ge_dataset_meta

        ds_dir = _dataset_dir(dataset_display, term)
        os.makedirs(os.path.join(ds_dir, "trials"), exist_ok=True)

        # -- Bootstrap global best from disk ---------------------------------
        global_best_state = {
            "val":       float("inf"),
            "ckpt_path": _best_model_path(dataset_display, term),
            "cfg_path":  _best_config_path(dataset_display, term),
        }
        if os.path.isfile(global_best_state["cfg_path"]):
            try:
                with open(global_best_state["cfg_path"]) as f:
                    prev = json.load(f)
                global_best_state["val"] = float(
                    prev.get("val_loss", float("inf"))
                )
                print(Fore.WHITE
                      + f"  Existing best on disk: "
                      + f"val_loss={global_best_state['val']:.4f} "
                      + f"(trial {prev.get('trial_idx', '?')})" + Fore.RESET)
            except Exception:
                pass

        # -- Build pending trial list (skip cached) --------------------------
        pending_trials = []
        for trial_idx, trial in enumerate(trial_configs):
            cached = _load_trial_result(dataset_display, term, trial_idx)
            if (cached is not None and "val_loss" in cached
                    and not np.isnan(cached["val_loss"])):
                print(Fore.WHITE + f"  CACHED  trial {trial_idx:03d}: "
                      + f"val_loss={cached['val_loss']:.4f}" + Fore.RESET)
                summary_rows.append({
                    "dataset_display": dataset_display, "term": term,
                    "trial_idx": trial_idx, **cached,
                })
                continue
            pending_trials.append((trial_idx, trial))

        if not pending_trials:
            print(Fore.GREEN + "  All trials already cached for this "
                  + "(dataset, term)." + Fore.RESET)
        else:
            print(Fore.CYAN
                  + f"  Spawning {len(devices)} worker(s) for "
                  + f"{len(pending_trials)} pending trial(s)." + Fore.RESET)

            trial_queue  = ctx.Queue()
            result_queue = ctx.Queue()

            for item in pending_trials:
                trial_queue.put(item)
            # Poison pills — one per worker.
            for _ in devices:
                trial_queue.put(None)

            workers = []
            for i, (device, budget) in enumerate(zip(devices, vram_budgets)):
                p = ctx.Process(
                    target=gpu_worker,
                    args=(i, device, budget, trial_queue, result_queue,
                          (ge_name, term, to_univariate), CONTEXT_LENGTH),
                    name=f"gpu_worker_{i}_{device.replace(':', '')}",
                )
                p.start()
                workers.append(p)

            # -- Collect results ---------------------------------------------
            n_expected = len(pending_trials)
            n_received = 0
            t_start = time.perf_counter()
            while n_received < n_expected:
                try:
                    result = result_queue.get(timeout=3600)
                except Empty:
                    alive = [p for p in workers if p.is_alive()]
                    if not alive:
                        print(Fore.RED + "  All workers died with "
                              + f"{n_received}/{n_expected} results received."
                              + Fore.RESET)
                        break
                    print(Fore.YELLOW + f"  No result in 1h "
                          + f"({n_received}/{n_expected}); "
                          + f"{len(alive)} worker(s) still alive."
                          + Fore.RESET)
                    continue

                # Dataset-level skip signal from a worker.
                if isinstance(result, dict) and result.get("_dataset_skip"):
                    print(Fore.RED
                          + f"  SKIPPING DATASET {dataset_display}/{term}: "
                          + f"{result.get('reason', 'infeasible')}"
                          + Fore.RESET)
                    # Tear down workers and move on to the next dataset.
                    for p in workers:
                        if p.is_alive():
                            p.terminate()
                    for p in workers:
                        p.join(timeout=10)
                    break

                _handle_result(
                    result, dataset_display, term, horizon,
                    summary_rows, global_best_state,
                )
                n_received += 1
            else:
                # Loop completed normally (no break).
                pass

            elapsed = time.perf_counter() - t_start
            print(Fore.MAGENTA
                  + f"  Dataset wall-clock: {elapsed:.1f}s  "
                  + f"({n_received}/{n_expected} trials)" + Fore.RESET)

            # -- Join workers ------------------------------------------------
            for p in workers:
                p.join(timeout=120)
                if p.is_alive():
                    print(Fore.RED + f"  Worker {p.name} still alive — "
                          + "terminating." + Fore.RESET)
                    p.terminate()
                    p.join(timeout=10)

        # -- Per-dataset summary plot + CSV ----------------------------------
        df_ds = pd.DataFrame([r for r in summary_rows
                              if r.get("dataset_display") == dataset_display
                              and r.get("term") == term])
        if df_ds.empty:
            print(Fore.YELLOW + f"  No trials completed for "
                  + f"{dataset_display}/{term} — skipping summary." + Fore.RESET)
            continue
        # ... existing CSV + plot code unchanged ...
        keep_cols = ["trial_idx", "val_loss", "patch_length", "d_model",
                     "num_hidden_layers", "num_attention_heads",
                     "dropout", "learning_rate",
                     "auto_batch_size", "auto_lr", "peak_vram_gb",
                     "device", "elapsed_seconds"]
        df_ds = df_ds[[c for c in keep_cols if c in df_ds.columns]]
        csv_path = os.path.join(ds_dir, "random_search.csv")
        df_ds.to_csv(csv_path, index=False)
        df_for_plot = df_ds.dropna(subset=["val_loss"])
        if len(df_for_plot) > 0:
            plot_random_search_summary(
                df_for_plot,
                os.path.join(ds_dir, "random_search_summary.png"),
                dataset_display, term,
            )
        print(Fore.GREEN + f"  Summary CSV: {csv_path}" + Fore.RESET)

    # -- Global summary --------------------------------------------------------
    global_df = pd.DataFrame(summary_rows)
    global_csv = os.path.join(run_dir, "random_search_all.csv")
    global_df.drop(columns=["history"], errors="ignore").to_csv(
        global_csv, index=False
    )
    print(Fore.GREEN + f"\n  Global summary: {global_csv}" + Fore.RESET)
    print(Fore.GREEN + "\nTraining pipeline done." + Fore.RESET)


if __name__ == "__main__":
    main()