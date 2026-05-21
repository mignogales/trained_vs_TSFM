"""
PatchTST training pipeline for GiftEval datasets — two-stage random search.

Pipeline
--------
For each (dataset, term) in DATASETS:
  1.  Load GiftEval Dataset and build sliding train/val tensors.
  2.  STAGE 1 — coarse architectural sweep:
        * Sample N_TRIALS configs from HP_SPACE.
        * Train each as a point forecaster (no distribution head) with MAE loss.
        * Select TOP_M=10 trials by validation MAE.
  3.  STAGE 2 — probabilistic refinement of the top-M:
        * Rebuild each surviving config with a Student-t output head.
        * Train with NLL.
        * On the best-NLL checkpoint, evaluate on val: NLL, MAE, CRPS,
          mean pinball loss over a quantile grid, and 80% PI coverage.
  4.  SELECTION — pick the final model:
        * Primary metric: configurable (default = CRPS).
        * Hard constraint: val MAE of the NLL model must be within
          MAE_GUARDRAIL_RATIO × (best stage-1 val MAE).
        * If no candidate clears the guardrail, fall back to best primary
          metric overall and log a warning.
  5.  Persist weights *only* for the selected model.

Architecture
------------
* Process-per-GPU parallelism: one worker per device drains trials from a
  shared queue. The queue carries (stage, trial_idx, trial) tuples so the
  same worker pool handles both stages without re-spawning.
* Per-GPU VRAM budgets; auto batch-size (largest in BS_CANDIDATES that fits
  the device's budget for the relevant model build); LR rescaled via the
  sqrt rule (Adam-safe).
* Validation accumulator stays on device — one host sync per validation pass.
* Pinned CPU→GPU upload on worker startup (shared across both stages).

CRPS computation
----------------
Empirical CRPS via Hersbach's sorted-sample identity:
        CRPS ≈ (1/S) Σ_i |X_(i) − y| − (1/S²) Σ_i (2i − S − 1) · X_(i)
Avoids the O(S²) outer-difference tensor entirely.

Cache layout
------------
logs/experiments/patchtst_training/<dataset>/<term>/
        stage1_mae/
            trials/trial_<NNN>.json
            sweep_summary.png
            sweep_summary.csv
        stage2_nll/
            trials/trial_<NNN>.json
            sweep_summary.png
            metrics_summary.csv
        selection_report.json
        best_model.pt
        best_config.json
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
N_TRIALS               = 90          # Stage 1 (MAE) sample size
STAGE2_TOP_M           = 10           # Stage 2 (NLL) refinement size
MAX_EPOCHS             = 50
VAL_EVERY_N_EPOCHS     = 4
EARLY_STOPPING_PATIENCE = 4           # in "validation events" (not epochs)
WEIGHT_DECAY           = 1e-4
GRAD_CLIP              = 1.0
MAX_TRAIN_WINDOWS      = 50_000
MAX_VAL_WINDOWS        = 5_000
SEED                   = 42

# -- Stage 2 selection ---------------------------------------------------------
# Primary probabilistic metric for selection: "crps", "pinball", or "nll".
STAGE2_SELECTION_METRIC = "crps"
# Hard MAE guardrail: NLL model's val MAE must be within this multiplicative
# margin of the best stage-1 val MAE. Set to None to disable.
MAE_GUARDRAIL_RATIO    = 1.15
# Quantile grid for empirical pinball loss and coverage.
QUANTILE_GRID          = tuple(round(0.05 * k, 2) for k in range(1, 20))   # 0.05 ... 0.95
COVERAGE_LO, COVERAGE_HI = 0.10, 0.90    # 80% prediction interval

# -- Auto batch-size + LR scaling ---------------------------------------------
BS_REFERENCE           = 128
BS_CANDIDATES          = [2048, 1536, 1024, 512, 256, 128, 64]
LR_SCALING_RULE        = "sqrt"

# -- Multi-GPU configuration ---------------------------------------------------
DEVICES = None
VRAM_BUDGET_GB_PER_DEVICE: Optional[List[float]] = [10.0, 6.0, 10.0]
VRAM_BUDGET_DEFAULT_GB = 10.0

# -- Random search HP space ----------------------------------------------------
HP_SPACE = {
    "patch_length":        [8, 16, 32],
    "patch_stride":        [4, 8, 16],
    "d_model":             [64, 128],
    "num_hidden_layers":   [3, 6],
    "num_attention_heads": [4, 8],
    "dropout":             [0.1, 0.2],
    "learning_rate":       [1e-4, 2.5e-4, 5e-4],
    "weight_decay":        [1e-4, 1e-3, 1e-2],
}

# -- Inference sampling for CRPS / pinball / coverage --------------------------
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
    if not torch.cuda.is_available():
        return ["cpu"]
    if DEVICES is not None:
        return list(DEVICES)
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]


def resolve_vram_budgets(devices: List[str]) -> List[float]:
    if VRAM_BUDGET_GB_PER_DEVICE is None:
        return [VRAM_BUDGET_DEFAULT_GB] * len(devices)
    if len(VRAM_BUDGET_GB_PER_DEVICE) != len(devices):
        raise ValueError(
            f"VRAM_BUDGET_GB_PER_DEVICE has {len(VRAM_BUDGET_GB_PER_DEVICE)} "
            f"entries but {len(devices)} devices were resolved: {devices}."
        )
    return list(VRAM_BUDGET_GB_PER_DEVICE)


# ==============================================================================
#  DATA BUILDING  (unchanged)
# ==============================================================================

def _series_train_val_split(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = len(series)
    cut = int(L * (80.0 / 90.0))
    return series[:cut], series[cut:]


def _sliding_windows(
    series: np.ndarray, context_length: int, horizon: int, stride: int = 1,
    max_windows: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    L = len(series); win = context_length + horizon
    if L < win:
        return (
            np.empty((0, context_length), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )
    n_possible = (L - win) // stride + 1
    if max_windows is not None and n_possible > max_windows:
        rng = rng or np.random.RandomState(SEED)
        starts = np.sort(rng.choice(n_possible, size=max_windows, replace=False) * stride)
    else:
        starts = np.arange(0, L - win + 1, stride)
    xs = np.empty((len(starts), context_length), dtype=np.float32)
    ys = np.empty((len(starts), horizon), dtype=np.float32)
    for i, s in enumerate(starts):
        xs[i] = series[s : s + context_length]
        ys[i] = series[s + context_length : s + win]
    return xs, ys


def build_train_val_tensors(
    ge_dataset: GiftEvalDataset, context_length: int, horizon: int,
    max_train_windows: int = MAX_TRAIN_WINDOWS,
    max_val_windows:   int = MAX_VAL_WINDOWS, rng_seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(rng_seed)
    entries = list(ge_dataset.training_dataset)
    rng.shuffle(entries)

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
    n_train_total = 0; n_val_total = 0
    for tgt in cleaned:
        train_portion, val_portion = _series_train_val_split(tgt)
        if n_train_total < max_train_windows:
            remaining = max_train_windows - n_train_total
            x_tr, y_tr = _sliding_windows(train_portion, context_length, horizon,
                                          max_windows=remaining, rng=rng)
            if len(x_tr) > 0:
                x_tr_list.append(x_tr); y_tr_list.append(y_tr)
                n_train_total += len(x_tr)
        if n_val_total < max_val_windows:
            remaining = max_val_windows - n_val_total
            x_vl, y_vl = _sliding_windows(val_portion, context_length, horizon,
                                          max_windows=remaining, rng=rng)
            if len(x_vl) > 0:
                x_vl_list.append(x_vl); y_vl_list.append(y_vl)
                n_val_total += len(x_vl)
        if n_train_total >= max_train_windows and n_val_total >= max_val_windows:
            break

    # ------------------------------------------------------------------ STAGE 2
    if not x_vl_list:
        print(Fore.YELLOW
              + "  [build_train_val_tensors] standard 10% val split produced "
              + "0 val windows — falling back to trailing-window-per-series."
              + Fore.RESET)
        x_tr_list, y_tr_list, x_vl_list, y_vl_list = [], [], [], []
        n_train_total = 0; n_val_total = 0
        rng = np.random.RandomState(rng_seed)

        n_series_skipped_short = 0
        for tgt in cleaned:
            L = len(tgt)
            if L < win:
                n_series_skipped_short += 1
                continue
            val_x = tgt[L - win : L - horizon]
            val_y = tgt[L - horizon : L]
            if n_val_total < max_val_windows:
                x_vl_list.append(val_x[None, :].astype(np.float32))
                y_vl_list.append(val_y[None, :].astype(np.float32))
                n_val_total += 1
            train_portion = tgt[: L - win]
            if n_train_total < max_train_windows and len(train_portion) >= win:
                remaining = max_train_windows - n_train_total
                x_tr, y_tr = _sliding_windows(train_portion, context_length, horizon,
                                              max_windows=remaining, rng=rng)
                if len(x_tr) > 0:
                    x_tr_list.append(x_tr); y_tr_list.append(y_tr)
                    n_train_total += len(x_tr)
            if n_train_total >= max_train_windows and n_val_total >= max_val_windows:
                break

        print(Fore.YELLOW
              + f"  [build_train_val_tensors] fallback summary: "
              + f"n_train_windows={n_train_total}  "
              + f"n_val_windows={n_val_total}  "
              + f"series_too_short={n_series_skipped_short}" + Fore.RESET)

    if not x_tr_list or not x_vl_list:
        raise InsufficientDataError(
            f"Cannot build train/val tensors for context_length={context_length} "
            f"and horizon={horizon}: train_windows={n_train_total}, "
            f"val_windows={n_val_total} after fallback."
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
    rng = random.Random(seed); seen, configs = set(), []
    max_attempts = n_trials * 50; attempts = 0
    while len(configs) < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in HP_SPACE.items()}
        if cfg["d_model"] % cfg["num_attention_heads"] != 0:
            continue
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key); configs.append(TrialConfig(**cfg))
    return configs


# ==============================================================================
#  MODEL BUILDING
# ==============================================================================

def build_patchtst_point(
    trial: TrialConfig, context_length: int, horizon: int,
    num_input_channels: int = 1,
) -> PatchTSTForPrediction:
    """Stage 1: point-forecast head (no distribution). Training loss = MAE
    (computed manually from prediction_outputs)."""
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
        loss                 = "mse",   # placeholder; we override with MAE
        distribution_output  = None,
        scaling              = "std",
    )
    return PatchTSTForPrediction(cfg)


def build_patchtst_nll(
    trial: TrialConfig, context_length: int, horizon: int,
    num_input_channels: int = 1,
) -> PatchTSTForPrediction:
    """Stage 2: probabilistic PatchTST with Student-t output head."""
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
    if rule == "sqrt":
        return float(base_lr * math.sqrt(bs / bs_ref))
    if rule == "linear":
        return float(base_lr * (bs / bs_ref))
    raise ValueError(f"Unknown LR scaling rule: {rule}")


def _probe_vram(builder, trial: TrialConfig, context_length: int, horizon: int,
                batch_size: int, device: str,
                use_mae_loss: bool) -> Optional[float]:
    """Empirically measure peak VRAM (GB) of one fwd+bwd+step on `device`."""
    if not device.startswith("cuda"):
        return 0.0
    try:
        with torch.cuda.device(torch.device(device)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = builder(trial, context_length, horizon).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(batch_size, context_length, 1, device=device)
            y = torch.randn(batch_size, horizon, 1, device=device)
            out = model(past_values=x, future_values=y)
            if use_mae_loss:
                loss = (out.prediction_outputs - y).abs().mean()
            else:
                loss = out.loss
            loss.backward(); opt.step()
            torch.cuda.synchronize(torch.device(device))
            peak_bytes = torch.cuda.max_memory_allocated()
            del model, opt, x, y, out, loss
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return peak_bytes / (1024 ** 3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        return None


def auto_select_bs_lr(
    trial: TrialConfig, context_length: int, horizon: int, device: str,
    budget_gb: float, stage: str,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Pick largest bs in BS_CANDIDATES that fits the per-device budget for the
    given training stage; scale LR by the sqrt rule."""
    builder       = build_patchtst_point if stage == "mae" else build_patchtst_nll
    use_mae_loss  = (stage == "mae")
    for bs in BS_CANDIDATES:
        peak = _probe_vram(builder, trial, context_length, horizon, bs, device,
                           use_mae_loss=use_mae_loss)
        if peak is None:
            continue
        if peak <= budget_gb:
            return bs, scale_lr(trial.learning_rate, bs), peak
    return None, None, None


# ==============================================================================
#  EVALUATION HELPERS
# ==============================================================================

def _evaluate_mae(model, x_val_gpu, y_val_gpu, bs: int) -> float:
    """Validation MAE for point-forecast models. One host sync at end."""
    model.eval()
    device = x_val_gpu.device
    total_mae = torch.zeros((), device=device); total_n = 0
    n_val = x_val_gpu.shape[0]
    with torch.no_grad():
        for start in range(0, n_val, bs):
            end = min(start + bs, n_val)
            x = x_val_gpu[start:end]; y = y_val_gpu[start:end]
            out = model(past_values=x, future_values=y)
            mae = (out.prediction_outputs - y).abs().sum()
            total_mae = total_mae + mae
            total_n += x.shape[0] * y.shape[1] * y.shape[2]
    return (total_mae / max(total_n, 1)).item()


def _evaluate_nll(model, x_val_gpu, y_val_gpu, bs: int) -> float:
    """Validation NLL for probabilistic models. One host sync at end."""
    model.eval()
    device = x_val_gpu.device
    total_loss = torch.zeros((), device=device); total_n = 0
    n_val = x_val_gpu.shape[0]
    with torch.no_grad():
        for start in range(0, n_val, bs):
            end = min(start + bs, n_val)
            x = x_val_gpu[start:end]; y = y_val_gpu[start:end]
            out = model(past_values=x, future_values=y)
            cnt = x.shape[0]
            total_loss = total_loss + out.loss * cnt
            total_n += cnt
    return (total_loss / max(total_n, 1)).item()


def _evaluate_probabilistic_full(
    model, x_val_gpu, y_val_gpu, bs: int,
    quantiles: Tuple[float, ...] = QUANTILE_GRID,
    coverage_lo: float = COVERAGE_LO, coverage_hi: float = COVERAGE_HI,
) -> Dict[str, float]:
    """Compute {val_nll, val_mae, val_crps, val_pinball, val_coverage_80} in a
    single pass over the validation set.

    CRPS via Hersbach's sorted-sample identity (no O(S²) tensor):
        CRPS = (1/S) Σ |X_(i) − y| − (1/S²) Σ (2i − S − 1) X_(i)
    """
    model.eval()
    device = x_val_gpu.device
    q_grid     = torch.tensor(quantiles, device=device, dtype=torch.float32)
    q_lo_t     = torch.tensor(coverage_lo, device=device)
    q_hi_t     = torch.tensor(coverage_hi, device=device)

    total_nll      = torch.zeros((), device=device)
    total_mae_sum  = torch.zeros((), device=device)
    total_crps_sum = torch.zeros((), device=device)
    total_pin_sum  = torch.zeros((), device=device)
    total_cov_sum  = torch.zeros((), device=device)
    total_n        = 0      # number of (window) examples
    total_elems    = 0      # number of (window × horizon × channel) elements

    n_val = x_val_gpu.shape[0]
    with torch.no_grad():
        for start in range(0, n_val, bs):
            end = min(start + bs, n_val)
            x = x_val_gpu[start:end]; y = y_val_gpu[start:end]  # (B, H, C)
            B, H, C = y.shape; cnt = B

            # NLL
            out = model(past_values=x, future_values=y)
            total_nll = total_nll + out.loss * cnt

            # Sample from predictive distribution: (B, S, H, C)
            try:
                samples = model.generate(past_values=x).sequences.to(torch.float32)
            except torch.cuda.OutOfMemoryError:
                # Halve batch size internally — rare but defensive.
                torch.cuda.empty_cache()
                samples = model.generate(past_values=x[: max(1, B // 2)]).sequences
                # If we had to truncate, we cannot fairly average; skip metrics
                # for this batch but keep NLL contribution above.
                total_n += cnt
                total_elems += B * H * C
                continue

            S = samples.shape[1]

            # --- CRPS (sorted-sample form) -----------------------------------
            samples_sorted, _ = torch.sort(samples, dim=1)                # (B,S,H,C)
            i_idx = torch.arange(1, S + 1, device=device, dtype=torch.float32)
            weights = (2.0 * i_idx - S - 1.0).view(1, -1, 1, 1)            # (1,S,1,1)
            term1 = (samples - y.unsqueeze(1)).abs().mean(dim=1)            # (B,H,C)
            term2 = (weights * samples_sorted).sum(dim=1) / (S * S)         # (B,H,C)
            crps_per_elem = term1 - term2                                   # (B,H,C)
            total_crps_sum = total_crps_sum + crps_per_elem.sum()

            # --- Point forecast (sample median) → MAE -----------------------
            point_pred = samples.median(dim=1).values                       # (B,H,C)
            total_mae_sum = total_mae_sum + (point_pred - y).abs().sum()

            # --- Pinball loss over quantile grid ----------------------------
            # quantile_preds: (Q, B, H, C)
            quantile_preds = torch.quantile(samples, q_grid, dim=1)
            err = y.unsqueeze(0) - quantile_preds                           # (Q,B,H,C)
            qv = q_grid.view(-1, 1, 1, 1)
            pinball = torch.maximum(qv * err, (qv - 1.0) * err)             # (Q,B,H,C)
            # Mean over quantiles, sum over (B,H,C) — keeps the per-element norm.
            total_pin_sum = total_pin_sum + pinball.mean(dim=0).sum()

            # --- 80% PI coverage --------------------------------------------
            q_lo = torch.quantile(samples, q_lo_t, dim=1)                   # (B,H,C)
            q_hi = torch.quantile(samples, q_hi_t, dim=1)                   # (B,H,C)
            inside = ((y >= q_lo) & (y <= q_hi)).float()
            total_cov_sum = total_cov_sum + inside.sum()

            total_n     += cnt
            total_elems += B * H * C

    n_examples = max(total_n, 1)
    n_elems    = max(total_elems, 1)
    return {
        "val_nll":         (total_nll / n_examples).item(),
        "val_mae":         (total_mae_sum / n_elems).item(),
        "val_crps":        (total_crps_sum / n_elems).item(),
        "val_pinball":     (total_pin_sum / n_elems).item(),
        "val_coverage_80": (total_cov_sum / n_elems).item(),
    }


# ==============================================================================
#  STAGE 1 — MAE point-forecast trial
# ==============================================================================

def _run_single_trial_mae(
    trial_idx: int, trial: TrialConfig, vram_budget_gb: float,
    x_train, y_train, x_val, y_val,
    context_length: int, horizon: int, device: str,
) -> Dict[str, Any]:
    """Train a point-forecast PatchTST with MAE loss; return val MAE."""
    bs, lr, peak = auto_select_bs_lr(
        trial, context_length, horizon, device, vram_budget_gb, stage="mae",
    )
    if bs is None:
        return {
            "trial_idx": trial_idx, "stage": "mae", "device": device,
            "val_mae": float("nan"),
            "history": {"train_loss": [], "val_loss": [], "val_epochs": []},
            "failed": True, "skip_reason": "vram_oom",
            "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
            "elapsed_seconds": 0.0, "cfg": asdict(trial),
        }

    tag = f"[{device}] [MAE] trial {trial_idx:03d}"
    print(Fore.CYAN
          + f"  {tag}: bs={bs} lr={lr:.2e} peak={peak:.2f}GB  "
          + f"budget={vram_budget_gb:.1f}GB" + Fore.RESET)

    t0 = time.perf_counter()
    n_train = x_train.shape[0]; steps_per_epoch = n_train // bs

    model = build_patchtst_point(trial, context_length, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)

    best_val = float("inf"); patience_left = EARLY_STOPPING_PATIENCE
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_epochs": []}

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            run_loss = torch.zeros((), device=device); n_seen = 0

            for step in range(steps_per_epoch):
                idx = perm[step * bs : (step + 1) * bs]
                x = x_train.index_select(0, idx)
                y = y_train.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                out = model(past_values=x, future_values=y)
                loss = (out.prediction_outputs - y).abs().mean()   # MAE
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                run_loss = run_loss + loss.detach() * bs
                n_seen += bs

            train_loss = (run_loss / max(n_seen, 1)).item()
            history["train_loss"].append(train_loss)

            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS:
                val_mae = _evaluate_mae(model, x_val, y_val, bs)
                history["val_loss"].append(val_mae)
                history["val_epochs"].append(epoch)
                print(Fore.YELLOW
                      + f"  {tag} epoch {epoch:3d}  "
                      + f"train_mae={train_loss:.4f}  val_mae={val_mae:.4f}"
                      + Fore.RESET)
                if val_mae < best_val:
                    best_val = val_mae
                    patience_left = EARLY_STOPPING_PATIENCE
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(Fore.MAGENTA + f"  {tag} early stop at epoch {epoch}"
                              + Fore.RESET)
                        break
    except torch.cuda.OutOfMemoryError as exc:
        print(Fore.RED + f"  {tag} OOM during training: {exc}" + Fore.RESET)
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    del optimizer, model
    torch.cuda.empty_cache()

    return {
        "trial_idx": trial_idx, "stage": "mae", "device": device,
        "val_mae": float(best_val) if best_val != float("inf") else float("nan"),
        "history": history, "failed": False,
        "auto_batch_size": bs, "auto_lr": lr,
        "peak_vram_gb": round(peak, 3),
        "elapsed_seconds": round(elapsed, 2),
        "cfg": asdict(trial),
    }


# ==============================================================================
#  STAGE 2 — Student-t NLL trial with full probabilistic eval
# ==============================================================================

def _run_single_trial_nll(
    trial_idx: int, trial: TrialConfig, vram_budget_gb: float,
    x_train, y_train, x_val, y_val,
    context_length: int, horizon: int, device: str,
) -> Dict[str, Any]:
    """Train Student-t PatchTST with NLL; at the end, reload best-NLL weights
    and compute the full probabilistic metric panel on val.

    Weights are persisted to a per-trial temp file. The parent decides which
    one to keep after stage-2 selection; non-selected files are deleted there.
    """
    bs, lr, peak = auto_select_bs_lr(
        trial, context_length, horizon, device, vram_budget_gb, stage="nll",
    )
    if bs is None:
        return {
            "trial_idx": trial_idx, "stage": "nll", "device": device,
            "val_nll": float("nan"), "val_mae": float("nan"),
            "val_crps": float("nan"), "val_pinball": float("nan"),
            "val_coverage_80": float("nan"),
            "history": {"train_loss": [], "val_loss": [], "val_epochs": []},
            "best_state_path": None, "failed": True,
            "skip_reason": "vram_oom",
            "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
            "elapsed_seconds": 0.0, "cfg": asdict(trial),
        }

    tag = f"[{device}] [NLL] trial {trial_idx:03d}"
    print(Fore.CYAN
          + f"  {tag}: bs={bs} lr={lr:.2e} peak={peak:.2f}GB  "
          + f"budget={vram_budget_gb:.1f}GB" + Fore.RESET)

    t0 = time.perf_counter()
    n_train = x_train.shape[0]; steps_per_epoch = n_train // bs

    model = build_patchtst_nll(trial, context_length, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = EARLY_STOPPING_PATIENCE
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_epochs": []}

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            run_loss = torch.zeros((), device=device); n_seen = 0

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
                run_loss = run_loss + loss.detach() * bs
                n_seen += bs

            train_loss = (run_loss / max(n_seen, 1)).item()
            history["train_loss"].append(train_loss)

            if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS:
                val_loss = _evaluate_nll(model, x_val, y_val, bs)
                history["val_loss"].append(val_loss)
                history["val_epochs"].append(epoch)
                print(Fore.YELLOW
                      + f"  {tag} epoch {epoch:3d}  "
                      + f"train_nll={train_loss:.4f}  val_nll={val_loss:.4f}"
                      + Fore.RESET)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience_left = EARLY_STOPPING_PATIENCE
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(Fore.MAGENTA + f"  {tag} early stop at epoch {epoch}"
                              + Fore.RESET)
                        break
    except torch.cuda.OutOfMemoryError as exc:
        print(Fore.RED + f"  {tag} OOM during training: {exc}" + Fore.RESET)
        torch.cuda.empty_cache()

    # -- Load best-NLL state and compute the full probabilistic metric panel.
    metrics: Dict[str, float] = {
        "val_nll": float(best_val) if best_val != float("inf") else float("nan"),
        "val_mae": float("nan"), "val_crps": float("nan"),
        "val_pinball": float("nan"), "val_coverage_80": float("nan"),
    }
    best_state_path: Optional[str] = None
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
            full = _evaluate_probabilistic_full(model, x_val, y_val, bs)
            # NLL recomputed from same checkpoint — keep the per-pass value
            # (matches the value used for early stopping & checkpoint selection).
            full["val_nll"] = metrics["val_nll"]
            metrics.update(full)
            print(Fore.GREEN
                  + f"  {tag} probabilistic eval: "
                  + f"CRPS={metrics['val_crps']:.4f}  "
                  + f"pinball={metrics['val_pinball']:.4f}  "
                  + f"MAE={metrics['val_mae']:.4f}  "
                  + f"cov80={metrics['val_coverage_80']:.3f}" + Fore.RESET)
        except Exception as exc:
            print(Fore.RED + f"  {tag} probabilistic eval failed: "
                  + f"{type(exc).__name__}: {exc}" + Fore.RESET)
            torch.cuda.empty_cache()

        # Persist checkpoint to disk; pass the path through the queue (not the
        # tensors themselves — sender-owned tensor-sharing sockets die with
        # the worker and break parent rebuilds).
        fd, best_state_path = tempfile.mkstemp(
            prefix=f"patchtst_w_trial{trial_idx:03d}_nll_", suffix=".pt")
        os.close(fd)
        torch.save(best_state, best_state_path)
        del best_state

    elapsed = time.perf_counter() - t0
    del optimizer, model
    torch.cuda.empty_cache()

    return {
        "trial_idx": trial_idx, "stage": "nll", "device": device,
        **metrics,
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
    worker_id: int, device: str, vram_budget_gb: float,
    trial_queue: "mp.Queue", result_queue: "mp.Queue",
    dataset_args: Tuple[str, str, bool], context_length: int,
):
    """Worker process — owns one GPU. Dispatches by stage tag in the message.

    Message format: ("mae"|"nll", trial_idx, TrialConfig)  or  None (poison).
    """
    set_seed(SEED + worker_id)
    torch.cuda.set_device(torch.device(device))
    torch.set_float32_matmul_precision("high")

    ge_name, term, to_univariate = dataset_args

    # -- Build train/val tensors locally and upload to GPU once --------------
    try:
        ge_dataset = GiftEvalDataset(name=ge_name, term=term,
                                     to_univariate=to_univariate)
        horizon = ge_dataset.prediction_length
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = build_train_val_tensors(
            ge_dataset, context_length, horizon)
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
        print(Fore.RED + f"  [{device}] worker {worker_id} SKIP DATASET: "
              + f"{exc}" + Fore.RESET)
        result_queue.put({"_dataset_skip": True, "reason": str(exc),
                          "device": device})
        # Drain queue.
        while True:
            try: msg = trial_queue.get(timeout=5)
            except Empty: break
            if msg is None: break
        return
    except Exception as exc:
        print(Fore.RED + f"  [{device}] worker {worker_id} setup failed: "
              + f"{exc}" + Fore.RESET)
        while True:
            try: msg = trial_queue.get(timeout=5)
            except Empty: break
            if msg is None: break
            stage, trial_idx, trial = msg
            failure_payload = {
                "trial_idx": trial_idx, "stage": stage, "device": device,
                "history": {}, "failed": True,
                "skip_reason": "worker_setup_failed",
                "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
                "elapsed_seconds": 0.0, "cfg": asdict(trial),
            }
            if stage == "mae":
                failure_payload["val_mae"] = float("nan")
            else:
                failure_payload.update({
                    "val_nll": float("nan"), "val_mae": float("nan"),
                    "val_crps": float("nan"), "val_pinball": float("nan"),
                    "val_coverage_80": float("nan"),
                    "best_state_path": None,
                })
            result_queue.put(failure_payload)
        return

    # -- Main consumption loop ----------------------------------------------
    while True:
        try:
            msg = trial_queue.get(timeout=3600)
        except Empty:
            print(Fore.MAGENTA + f"  [{device}] worker {worker_id} queue "
                  + "timeout — exiting." + Fore.RESET)
            break
        if msg is None:
            break
        stage, trial_idx, trial = msg
        try:
            if stage == "mae":
                result = _run_single_trial_mae(
                    trial_idx, trial, vram_budget_gb,
                    x_train, y_train, x_val, y_val,
                    context_length, horizon, device,
                )
            elif stage == "nll":
                result = _run_single_trial_nll(
                    trial_idx, trial, vram_budget_gb,
                    x_train, y_train, x_val, y_val,
                    context_length, horizon, device,
                )
            else:
                raise ValueError(f"Unknown stage: {stage!r}")
        except Exception as exc:
            print(Fore.RED + f"  [{device}] [{stage}] trial {trial_idx:03d} "
                  + f"CRASHED: {type(exc).__name__}: {exc}" + Fore.RESET)
            result = {
                "trial_idx": trial_idx, "stage": stage, "device": device,
                "history": {}, "failed": True,
                "skip_reason": f"trial_exception:{type(exc).__name__}",
                "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
                "elapsed_seconds": 0.0, "cfg": asdict(trial),
            }
            if stage == "mae":
                result["val_mae"] = float("nan")
            else:
                result.update({
                    "val_nll": float("nan"), "val_mae": float("nan"),
                    "val_crps": float("nan"), "val_pinball": float("nan"),
                    "val_coverage_80": float("nan"),
                    "best_state_path": None,
                })
            torch.cuda.empty_cache()
        result_queue.put(result)

    del x_train, y_train, x_val, y_val
    torch.cuda.empty_cache()
    print(Fore.CYAN + f"  [{device}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  PLOTTING
# ==============================================================================

def _plot_sweep_summary(
    trials_df: pd.DataFrame, save_path: str, dataset_display: str, term: str,
    metric_col: str, metric_label: str, title_suffix: str,
):
    """Generic running-best scatter for a sweep stage."""
    trials_df = trials_df.sort_values("trial_idx").reset_index(drop=True)
    running_best = []; best_so_far = float("inf"); best_idx_in_df = []
    for i, row in trials_df.iterrows():
        v = row[metric_col]
        if v < best_so_far:
            best_so_far = v
            best_idx_in_df.append(i)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(trials_df["trial_idx"], trials_df[metric_col],
               s=36, alpha=0.55, color="#1f77b4",
               edgecolor="white", linewidth=0.5, label="trial")
    improved = trials_df.iloc[best_idx_in_df]
    ax.plot(improved["trial_idx"], improved[metric_col],
            marker="o", markersize=9, linewidth=2.2, color="#d62728",
            label="best-so-far")
    best_row = trials_df.iloc[trials_df[metric_col].idxmin()]
    ax.annotate(
        f"  best: trial {int(best_row['trial_idx'])}\n"
        f"  {metric_label}={best_row[metric_col]:.4f}",
        xy=(best_row["trial_idx"], best_row[metric_col]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3b0", ec="#999", alpha=0.95),
    )
    ax.set_xlabel("Trial index", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(
        f"{title_suffix} — {dataset_display}  (term={term}, N={len(trials_df)})",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=10)
    plt.tight_layout(); plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


# ==============================================================================
#  CACHE HELPERS
# ==============================================================================

def _ds_dir(dataset_display: str, term: str) -> str:
    return os.path.join(CACHE_ROOT, dataset_display, term)

def _stage_dir(dataset_display: str, term: str, stage: str) -> str:
    return os.path.join(_ds_dir(dataset_display, term),
                        "stage1_mae" if stage == "mae" else "stage2_nll")

def _trial_json_path(dataset_display: str, term: str, stage: str,
                     trial_idx: int) -> str:
    return os.path.join(_stage_dir(dataset_display, term, stage),
                        "trials", f"trial_{trial_idx:03d}.json")

def _best_model_path(dataset_display: str, term: str) -> str:
    return os.path.join(_ds_dir(dataset_display, term), "best_model.pt")

def _best_config_path(dataset_display: str, term: str) -> str:
    return os.path.join(_ds_dir(dataset_display, term), "best_config.json")

def _selection_report_path(dataset_display: str, term: str) -> str:
    return os.path.join(_ds_dir(dataset_display, term), "selection_report.json")

def _load_trial_result(dataset_display: str, term: str, stage: str,
                       trial_idx: int) -> Optional[Dict]:
    p = _trial_json_path(dataset_display, term, stage, trial_idx)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def _save_trial_result(dataset_display: str, term: str, stage: str,
                       trial_idx: int, result: Dict):
    p = _trial_json_path(dataset_display, term, stage, trial_idx)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # Drop non-serializable / heavy fields.
    serializable = {k: v for k, v in result.items() if k != "best_state_path"}
    with open(p, "w") as f:
        json.dump(serializable, f, indent=2)


# ==============================================================================
#  STAGE EXECUTION HELPER  (drives one stage across workers)
# ==============================================================================

def _run_stage(
    stage: str,
    pending_trials: List[Tuple[int, TrialConfig]],
    workers: List[mp.Process],
    trial_queue: "mp.Queue", result_queue: "mp.Queue",
    dataset_display: str, term: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Enqueue trials for one stage, collect their results.
    Returns (results, dataset_skipped). If dataset_skipped is True, the parent
    must tear down workers and abandon this (dataset, term)."""
    for trial_idx, trial in pending_trials:
        trial_queue.put((stage, trial_idx, trial))

    results: List[Dict[str, Any]] = []
    n_expected = len(pending_trials)
    n_received = 0
    t_start = time.perf_counter()
    dataset_skipped = False

    while n_received < n_expected:
        try:
            r = result_queue.get(timeout=3600)
        except Empty:
            alive = [p for p in workers if p.is_alive()]
            if not alive:
                print(Fore.RED + f"  All workers died with "
                      + f"{n_received}/{n_expected} stage-{stage} results."
                      + Fore.RESET)
                break
            print(Fore.YELLOW + f"  No stage-{stage} result in 1h "
                  + f"({n_received}/{n_expected}); "
                  + f"{len(alive)} worker(s) alive." + Fore.RESET)
            continue

        if isinstance(r, dict) and r.get("_dataset_skip"):
            print(Fore.RED + f"  SKIPPING DATASET {dataset_display}/{term}: "
                  + f"{r.get('reason', 'infeasible')}" + Fore.RESET)
            dataset_skipped = True
            break

        results.append(r)
        n_received += 1

    elapsed = time.perf_counter() - t_start
    print(Fore.MAGENTA
          + f"  Stage-{stage} wall-clock: {elapsed:.1f}s  "
          + f"({n_received}/{n_expected} trials)" + Fore.RESET)
    return results, dataset_skipped


# ==============================================================================
#  SELECTION
# ==============================================================================

def _select_final(stage2_results: List[Dict[str, Any]],
                  stage1_best_mae: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick the final NLL model. Returns (chosen, report).

    Decision rule:
      eligible = trials with val_mae ≤ MAE_GUARDRAIL_RATIO × stage1_best_mae
      chosen   = argmin over eligible of val_<STAGE2_SELECTION_METRIC>
      if no eligible:
          chosen = argmin overall of val_<STAGE2_SELECTION_METRIC>,
          flag guardrail_violated = True.
    """
    metric_key = f"val_{STAGE2_SELECTION_METRIC}"

    def _key(r):
        v = r.get(metric_key)
        return float("inf") if v is None or (isinstance(v, float) and math.isnan(v)) else v

    valid = [r for r in stage2_results
             if not r.get("failed", False) and r.get("best_state_path")]
    if not valid:
        return {}, {
            "guardrail_violated": True,
            "stage1_best_mae": stage1_best_mae,
            "guardrail_ratio": MAE_GUARDRAIL_RATIO,
            "selection_metric": metric_key,
            "note": "no_valid_stage2_trials",
        }

    if MAE_GUARDRAIL_RATIO is not None and not math.isnan(stage1_best_mae):
        threshold = stage1_best_mae * MAE_GUARDRAIL_RATIO
        eligible = [r for r in valid
                    if not math.isnan(r.get("val_mae", float("nan")))
                    and r["val_mae"] <= threshold]
    else:
        threshold = None
        eligible = valid

    guardrail_violated = False
    if not eligible:
        guardrail_violated = True
        print(Fore.YELLOW + f"  No stage-2 candidate satisfies MAE guardrail "
              + f"(threshold={threshold:.4f}); falling back to best "
              + f"{metric_key} overall." + Fore.RESET)
        eligible = valid

    chosen = min(eligible, key=_key)
    report = {
        "selection_metric":   metric_key,
        "selected_trial_idx": int(chosen["trial_idx"]),
        "selected_metrics": {
            "val_nll":         chosen.get("val_nll"),
            "val_mae":         chosen.get("val_mae"),
            "val_crps":        chosen.get("val_crps"),
            "val_pinball":     chosen.get("val_pinball"),
            "val_coverage_80": chosen.get("val_coverage_80"),
        },
        "stage1_best_mae":     stage1_best_mae,
        "guardrail_ratio":     MAE_GUARDRAIL_RATIO,
        "guardrail_threshold": threshold,
        "guardrail_violated":  guardrail_violated,
        "coverage_target":     COVERAGE_HI - COVERAGE_LO,
        "n_eligible":          len(eligible),
        "n_stage2_valid":      len(valid),
    }
    return chosen, report


# ==============================================================================
#  MAIN
# ==============================================================================

def _csv_keep_cols_mae():
    return ["trial_idx", "val_mae", "patch_length", "d_model",
            "num_hidden_layers", "num_attention_heads", "dropout",
            "learning_rate", "auto_batch_size", "auto_lr",
            "peak_vram_gb", "device", "elapsed_seconds"]


def _csv_keep_cols_nll():
    return ["trial_idx", "val_nll", "val_mae", "val_crps", "val_pinball",
            "val_coverage_80", "patch_length", "d_model",
            "num_hidden_layers", "num_attention_heads", "dropout",
            "learning_rate", "auto_batch_size", "auto_lr",
            "peak_vram_gb", "device", "elapsed_seconds"]


def _persist_stage_artifacts(
    stage: str, results: List[Dict[str, Any]],
    dataset_display: str, term: str, trial_configs: List[TrialConfig],
):
    """Persist per-trial JSONs, sweep CSV, and summary plot for one stage."""
    sdir = _stage_dir(dataset_display, term, stage)
    os.makedirs(os.path.join(sdir, "trials"), exist_ok=True)

    # Write per-trial JSONs (best_state_path stripped inside helper).
    for r in results:
        _save_trial_result(dataset_display, term, stage, r["trial_idx"], r)

    # Flatten cfg into top-level columns for the CSV / plot.
    rows = []
    for r in results:
        flat = {"trial_idx": r["trial_idx"]}
        flat.update(r.get("cfg", {}))
        for key in ("val_mae", "val_nll", "val_crps", "val_pinball",
                    "val_coverage_80", "auto_batch_size", "auto_lr",
                    "peak_vram_gb", "device", "elapsed_seconds"):
            if key in r:
                flat[key] = r[key]
        rows.append(flat)
    df = pd.DataFrame(rows)
    keep = _csv_keep_cols_mae() if stage == "mae" else _csv_keep_cols_nll()
    df_csv = df[[c for c in keep if c in df.columns]]
    csv_path = os.path.join(sdir, "sweep_summary.csv"
                            if stage == "mae" else "metrics_summary.csv")
    df_csv.to_csv(csv_path, index=False)

    metric_col   = "val_mae" if stage == "mae" else f"val_{STAGE2_SELECTION_METRIC}"
    metric_label = "val MAE" if stage == "mae" else f"val {STAGE2_SELECTION_METRIC.upper()}"
    title        = "Stage 1 — MAE sweep" if stage == "mae" else "Stage 2 — NLL refinement"
    df_for_plot = df_csv.dropna(subset=[metric_col]) if metric_col in df_csv else None
    if df_for_plot is not None and len(df_for_plot) > 0:
        _plot_sweep_summary(
            df_for_plot, os.path.join(sdir, "sweep_summary.png"),
            dataset_display, term, metric_col, metric_label, title,
        )
    print(Fore.GREEN + f"  Stage-{stage} summary: {csv_path}" + Fore.RESET)


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    set_seed(SEED)
    devices      = resolve_devices()
    vram_budgets = resolve_vram_budgets(devices)

    print(Fore.CYAN + f"Devices: {devices}" + Fore.RESET)
    print(Fore.CYAN + f"VRAM budgets (GB) per device: "
          + f"{dict(zip(devices, vram_budgets))}" + Fore.RESET)
    print(Fore.CYAN + f"LR scaling: {LR_SCALING_RULE} (bs_ref={BS_REFERENCE})  |  "
          + f"BS candidates: {BS_CANDIDATES}" + Fore.RESET)
    print(Fore.CYAN + f"Stage 1: N={N_TRIALS} (MAE)   "
          + f"Stage 2: top M={STAGE2_TOP_M} (NLL)   "
          + f"select by val_{STAGE2_SELECTION_METRIC}   "
          + f"MAE guardrail × {MAE_GUARDRAIL_RATIO}" + Fore.RESET)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CACHE_ROOT, f"runs/{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    trial_configs = sample_trial_configs(N_TRIALS, seed=SEED)
    print(Fore.CYAN + f"Sampled {len(trial_configs)} unique trial configs"
          + Fore.RESET)

    ctx = mp.get_context("spawn")
    global_rows: List[Dict[str, Any]] = []

    for ge_name, term, dataset_display, to_univariate in DATASETS:
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  DATASET: {ge_name}  term={term}  "
              + f"({dataset_display})" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        ge_dataset_meta = GiftEvalDataset(name=ge_name, term=term,
                                          to_univariate=to_univariate)
        horizon = ge_dataset_meta.prediction_length
        print(Fore.CYAN + f"  freq={ge_dataset_meta.freq}  horizon={horizon}  "
              + f"target_dim={ge_dataset_meta.target_dim}" + Fore.RESET)
        del ge_dataset_meta

        ds_dir = _ds_dir(dataset_display, term)
        os.makedirs(os.path.join(_stage_dir(dataset_display, term, "mae"),
                                 "trials"), exist_ok=True)
        os.makedirs(os.path.join(_stage_dir(dataset_display, term, "nll"),
                                 "trials"), exist_ok=True)

        # ------------------------------------------------------------- STAGE 1
        # Resolve which stage-1 trials still need to run (cache-aware).
        stage1_results: List[Dict[str, Any]] = []
        pending_stage1: List[Tuple[int, TrialConfig]] = []
        for trial_idx, trial in enumerate(trial_configs):
            cached = _load_trial_result(dataset_display, term, "mae", trial_idx)
            if (cached is not None and "val_mae" in cached
                    and not (isinstance(cached["val_mae"], float)
                             and math.isnan(cached["val_mae"]))):
                print(Fore.WHITE + f"  [MAE] CACHED trial {trial_idx:03d}: "
                      + f"val_mae={cached['val_mae']:.4f}" + Fore.RESET)
                stage1_results.append(cached)
                continue
            pending_stage1.append((trial_idx, trial))

        # ------------------------------------------------------------- STAGE 2
        # Top-M from stage 1 (computed AFTER stage 1 completes; we cannot
        # determine pending_stage2 until then). Stage-2 cache check happens
        # post-selection.

        # Spawn workers (shared across both stages for this dataset).
        trial_queue  = ctx.Queue()
        result_queue = ctx.Queue()
        workers: List[mp.Process] = []
        for i, (device, budget) in enumerate(zip(devices, vram_budgets)):
            p = ctx.Process(
                target=gpu_worker,
                args=(i, device, budget, trial_queue, result_queue,
                      (ge_name, term, to_univariate), CONTEXT_LENGTH),
                name=f"gpu_worker_{i}_{device.replace(':', '')}",
            )
            p.start()
            workers.append(p)

        dataset_skipped = False

        # -- Run stage 1 ----------------------------------------------------
        if pending_stage1:
            print(Fore.CYAN + f"  Running {len(pending_stage1)} pending "
                  + f"stage-1 (MAE) trials." + Fore.RESET)
            new_results, dataset_skipped = _run_stage(
                "mae", pending_stage1, workers, trial_queue, result_queue,
                dataset_display, term,
            )
            stage1_results.extend(new_results)
        else:
            print(Fore.GREEN + "  All stage-1 trials already cached."
                  + Fore.RESET)

        # If the dataset is infeasible, tear down and continue.
        if dataset_skipped:
            for p in workers:
                if p.is_alive():
                    p.terminate()
            for p in workers:
                p.join(timeout=10)
            continue

        # Persist stage-1 artifacts.
        _persist_stage_artifacts("mae", stage1_results, dataset_display, term,
                                 trial_configs)

        # Pick top-M by val_mae.
        valid_stage1 = [r for r in stage1_results
                        if not r.get("failed", False)
                        and not (isinstance(r.get("val_mae"), float)
                                 and math.isnan(r["val_mae"]))]
        valid_stage1.sort(key=lambda r: r["val_mae"])
        top_m = valid_stage1[:STAGE2_TOP_M]
        if not top_m:
            print(Fore.RED + "  No valid stage-1 trials — skipping stage 2 "
                  + "for this dataset." + Fore.RESET)
            for _ in workers: trial_queue.put(None)
            for p in workers: p.join(timeout=120)
            continue

        stage1_best_mae = top_m[0]["val_mae"]
        print(Fore.CYAN + f"  Stage 1 done. Best val_mae={stage1_best_mae:.4f}. "
              + f"Promoting top {len(top_m)} configs to stage 2." + Fore.RESET)
        for r in top_m:
            print(Fore.WHITE + f"    -> trial {int(r['trial_idx']):03d}: "
                  + f"val_mae={r['val_mae']:.4f}" + Fore.RESET)

        # -- Run stage 2 (top-M, cache-aware) -------------------------------
        stage2_results: List[Dict[str, Any]] = []
        pending_stage2: List[Tuple[int, TrialConfig]] = []
        for r in top_m:
            tidx = int(r["trial_idx"])
            cached = _load_trial_result(dataset_display, term, "nll", tidx)
            if (cached is not None and "val_nll" in cached
                    and not (isinstance(cached["val_nll"], float)
                             and math.isnan(cached["val_nll"]))):
                # Cache hit — but we still need the weight file path.
                # Cached JSON intentionally drops it; require disk presence
                # of best_model.pt + best_config.json check during selection.
                print(Fore.WHITE + f"  [NLL] CACHED trial {tidx:03d}: "
                      + f"val_nll={cached['val_nll']:.4f}" + Fore.RESET)
                # We mark these without a best_state_path so they are excluded
                # from re-selection unless paired with the existing best_model.pt.
                cached["best_state_path"] = None
                stage2_results.append(cached)
                continue
            pending_stage2.append((tidx, trial_configs[tidx]))

        if pending_stage2:
            print(Fore.CYAN + f"  Running {len(pending_stage2)} pending "
                  + f"stage-2 (NLL) trials." + Fore.RESET)
            new_results, dataset_skipped = _run_stage(
                "nll", pending_stage2, workers, trial_queue, result_queue,
                dataset_display, term,
            )
            stage2_results.extend(new_results)

        # -- Tear down workers ---------------------------------------------
        for _ in workers: trial_queue.put(None)
        for p in workers:
            p.join(timeout=120)
            if p.is_alive():
                print(Fore.RED + f"  Worker {p.name} still alive — terminating."
                      + Fore.RESET)
                p.terminate(); p.join(timeout=10)

        # -- Persist stage-2 artifacts -------------------------------------
        _persist_stage_artifacts("nll", stage2_results, dataset_display, term,
                                 trial_configs)

        # -- Selection (only among trials whose checkpoint is still on disk).
        candidates = [r for r in stage2_results if r.get("best_state_path")
                      and os.path.isfile(r["best_state_path"])]
        if not candidates:
            print(Fore.YELLOW + "  No fresh stage-2 checkpoints in this run. "
                  + "Keeping any previously persisted best_model.pt as-is."
                  + Fore.RESET)
        else:
            chosen, report = _select_final(candidates, stage1_best_mae)
            if chosen:
                best_state_path = chosen["best_state_path"]
                target_ckpt = _best_model_path(dataset_display, term)
                shutil.copyfile(best_state_path, target_ckpt)
                with open(_best_config_path(dataset_display, term), "w") as f:
                    json.dump({
                        "trial_idx":          int(chosen["trial_idx"]),
                        **chosen.get("cfg", {}),
                        "val_nll":            chosen.get("val_nll"),
                        "val_mae":            chosen.get("val_mae"),
                        "val_crps":           chosen.get("val_crps"),
                        "val_pinball":        chosen.get("val_pinball"),
                        "val_coverage_80":    chosen.get("val_coverage_80"),
                        "stage1_best_mae":    stage1_best_mae,
                        "selection_metric":   report["selection_metric"],
                        "guardrail_ratio":    report["guardrail_ratio"],
                        "guardrail_violated": report["guardrail_violated"],
                        "auto_batch_size":    chosen.get("auto_batch_size"),
                        "auto_lr":            chosen.get("auto_lr"),
                        "context_length":     CONTEXT_LENGTH,
                        "prediction_length":  horizon,
                        "num_input_channels": 1,
                    }, f, indent=2)
                with open(_selection_report_path(dataset_display, term), "w") as f:
                    json.dump(report, f, indent=2)
                print(Fore.GREEN + f"  FINAL SELECTION: trial "
                      + f"{int(chosen['trial_idx']):03d}  "
                      + f"({report['selection_metric']}="
                      + f"{chosen.get(report['selection_metric']):.4f}, "
                      + f"val_mae={chosen.get('val_mae'):.4f}, "
                      + f"guardrail_violated={report['guardrail_violated']})"
                      + Fore.RESET)
            # Always clean up the per-trial temp checkpoints (selected one
            # is already copied to its final location).
            for r in stage2_results:
                p = r.get("best_state_path")
                if p and os.path.exists(p):
                    try: os.remove(p)
                    except OSError: pass

        # -- Accumulate to global summary ----------------------------------
        for r in stage1_results:
            global_rows.append({
                "dataset_display": dataset_display, "term": term,
                "stage": "mae", **{k: v for k, v in r.items()
                                   if k not in ("history", "best_state_path", "cfg")},
                **r.get("cfg", {}),
            })
        for r in stage2_results:
            global_rows.append({
                "dataset_display": dataset_display, "term": term,
                "stage": "nll", **{k: v for k, v in r.items()
                                   if k not in ("history", "best_state_path", "cfg")},
                **r.get("cfg", {}),
            })

    # -- Global summary --------------------------------------------------------
    global_df = pd.DataFrame(global_rows)
    global_csv = os.path.join(run_dir, "search_all.csv")
    global_df.to_csv(global_csv, index=False)
    print(Fore.GREEN + f"\n  Global summary: {global_csv}" + Fore.RESET)
    print(Fore.GREEN + "\nTwo-stage training pipeline done." + Fore.RESET)


if __name__ == "__main__":
    main()