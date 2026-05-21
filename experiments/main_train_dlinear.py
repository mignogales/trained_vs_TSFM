"""
DLinear training pipeline for GiftEval datasets — single-stage random search.

Designed as a baseline counterpart to the PatchTST pipeline. The architecture
(Zeng et al., 2023, "Are Transformers Effective for Time Series Forecasting?")
is a channel-independent linear model over a moving-average trend/seasonal
decomposition. It is a *point* forecaster by construction; no probabilistic
head is attached and the evaluation panel reduces to validation MAE.

Pipeline
--------
For each (dataset, term) in DATASETS:
  1. Load GiftEval Dataset and build sliding train/val tensors.
  2. Sample N_TRIALS configs from HP_SPACE (kernel_size, learning_rate,
     weight_decay).
  3. Train each as a point forecaster with MAE loss.
  4. Select the best by validation MAE; persist weights and config.

Architecture
------------
* Process-per-GPU parallelism via shared trial queue.
* Per-device VRAM budgets; auto batch-size; sqrt-rule LR rescaling.
* Pinned CPU->GPU upload of train/val tensors at worker startup.
* Per-instance std normalization inside the model for parity with the
  PatchTST `scaling="std"` configuration.

Cache layout
------------
logs/experiments/dlinear_training/<dataset>/<term>/
    trials/trial_<NNN>.json
    sweep_summary.png
    sweep_summary.csv
    best_model.pt
    best_config.json
"""

import torch
import torch.nn as nn
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
N_TRIALS               = 20           # Saturates the ~48-cell HP grid
MAX_EPOCHS             = 50
VAL_EVERY_N_EPOCHS     = 4
EARLY_STOPPING_PATIENCE = 4           # in "validation events" (not epochs)
GRAD_CLIP              = 1.0
MAX_TRAIN_WINDOWS      = 50_000
MAX_VAL_WINDOWS        = 5_000
SEED                   = 42

# -- Auto batch-size + LR scaling ---------------------------------------------
# DLinear is ~3 orders of magnitude smaller than PatchTST; ceiling raised.
BS_REFERENCE           = 128
BS_CANDIDATES          = [16384, 8192, 4096, 2048, 1024, 512, 256, 128]
LR_SCALING_RULE        = "sqrt"

# -- Multi-GPU configuration ---------------------------------------------------
DEVICES = None
VRAM_BUDGET_GB_PER_DEVICE: Optional[List[float]] = [10.0, 6.0]
VRAM_BUDGET_DEFAULT_GB = 10.0

# -- Random search HP space ----------------------------------------------------
# Effective dimensionality of DLinear is small: moving-average kernel size for
# the trend/seasonal decomposition, learning rate, and weight decay. All
# kernel sizes are odd to allow symmetric edge-replication padding.
HP_SPACE = {
    "kernel_size":   [13, 25, 51],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "weight_decay":  [0.0, 1e-5, 1e-4, 1e-3],
}

CACHE_ROOT = "logs/experiments/dlinear_training"


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
#  DLINEAR MODEL
# ==============================================================================

class _MovingAvg(nn.Module):
    """Causal-symmetric moving average via edge-replication padding.

    Implements the trend extractor from Zeng et al. (2023). The input shape is
    (B, L, C); replication of the first/last timestep keeps the output length
    equal to L regardless of the kernel size (which must be odd).
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got {kernel_size}.")
        self.kernel_size = kernel_size
        self.pad_each_side = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        front = x[:, 0:1, :].repeat(1, self.pad_each_side, 1)
        end   = x[:, -1:, :].repeat(1, self.pad_each_side, 1)
        x_pad = torch.cat([front, x, end], dim=1)            # (B, L + 2*pad, C)
        x_bcl = x_pad.permute(0, 2, 1)                       # (B, C, L+2*pad)
        trend = self.avg(x_bcl).permute(0, 2, 1)             # (B, L, C)
        return trend


class _SeriesDecomposition(nn.Module):
    """Trend / seasonal split: trend = moving avg, seasonal = x - trend."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


@dataclass
class _DLinearOutput:
    """Lightweight stand-in for the HF model output object."""
    prediction_outputs: torch.Tensor
    loss: Optional[torch.Tensor] = None


class DLinear(nn.Module):
    """Channel-independent DLinear with per-instance std normalization.

    Two linear maps (one for the seasonal component, one for the trend) project
    the context window of length L onto the horizon H. The normalization is
    applied per (window, channel) so that comparisons against PatchTST with
    `scaling="std"` isolate architectural differences.

    Args:
        context_length: Input window length L.
        prediction_length: Forecast horizon H.
        kernel_size: Odd integer; moving-average window for trend extraction.
        num_channels: Number of input channels C.
        individual: If True, fit a separate (Seasonal, Trend) linear pair per
            channel. For univariate forecasting (C=1) this has no effect.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        kernel_size: int,
        num_channels: int = 1,
        individual: bool = False,
    ):
        super().__init__()
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.num_channels      = num_channels
        self.individual        = individual
        self.decomposition     = _SeriesDecomposition(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(context_length, prediction_length)
                for _ in range(num_channels)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(context_length, prediction_length)
                for _ in range(num_channels)
            ])
        else:
            self.linear_seasonal = nn.Linear(context_length, prediction_length)
            self.linear_trend    = nn.Linear(context_length, prediction_length)

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
    ) -> _DLinearOutput:
        # past_values: (B, L, C)
        mean = past_values.mean(dim=1, keepdim=True)
        std  = past_values.std(dim=1, keepdim=True).clamp_min(1e-5)
        x_norm = (past_values - mean) / std

        seasonal, trend = self.decomposition(x_norm)         # both (B, L, C)
        # Permute to (B, C, L) so the Linear's last-axis maps L -> H.
        seasonal_bcl = seasonal.permute(0, 2, 1)
        trend_bcl    = trend.permute(0, 2, 1)

        if self.individual:
            B = seasonal_bcl.shape[0]
            seasonal_out = torch.empty(
                B, self.num_channels, self.prediction_length,
                dtype=seasonal_bcl.dtype, device=seasonal_bcl.device,
            )
            trend_out = torch.empty_like(seasonal_out)
            for c in range(self.num_channels):
                seasonal_out[:, c, :] = self.linear_seasonal[c](seasonal_bcl[:, c, :])
                trend_out[:, c, :]    = self.linear_trend[c](trend_bcl[:, c, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal_bcl)
            trend_out    = self.linear_trend(trend_bcl)

        pred_norm = (seasonal_out + trend_out).permute(0, 2, 1)   # (B, H, C)
        prediction = pred_norm * std + mean

        loss = None
        if future_values is not None:
            loss = (prediction - future_values).abs().mean()      # MAE
        return _DLinearOutput(prediction_outputs=prediction, loss=loss)


# ==============================================================================
#  DATA BUILDING  (identical contract to the PatchTST pipeline)
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
    kernel_size:   int
    learning_rate: float
    weight_decay:  float

    def __post_init__(self):
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd; got {self.kernel_size}"
            )


def sample_trial_configs(n_trials: int, seed: int = SEED) -> List[TrialConfig]:
    rng = random.Random(seed); seen, configs = set(), []
    max_attempts = n_trials * 50; attempts = 0
    while len(configs) < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in HP_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key); configs.append(TrialConfig(**cfg))
    if len(configs) < n_trials:
        print(Fore.YELLOW
              + f"  HP space exhausted: {len(configs)}/{n_trials} unique configs."
              + Fore.RESET)
    return configs


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


def _probe_vram(trial: TrialConfig, context_length: int, horizon: int,
                batch_size: int, device: str) -> Optional[float]:
    """Empirically measure peak VRAM (GB) of one fwd+bwd+step on `device`."""
    if not device.startswith("cuda"):
        return 0.0
    try:
        with torch.cuda.device(torch.device(device)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = DLinear(
                context_length=context_length,
                prediction_length=horizon,
                kernel_size=trial.kernel_size,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(batch_size, context_length, 1, device=device)
            y = torch.randn(batch_size, horizon, 1, device=device)
            out = model(past_values=x, future_values=y)
            out.loss.backward(); opt.step()
            torch.cuda.synchronize(torch.device(device))
            peak_bytes = torch.cuda.max_memory_allocated()
            del model, opt, x, y, out
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return peak_bytes / (1024 ** 3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        return None


def auto_select_bs_lr(
    trial: TrialConfig, context_length: int, horizon: int, device: str,
    budget_gb: float,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Pick largest bs in BS_CANDIDATES that fits the per-device budget;
    scale LR by the configured rule."""
    for bs in BS_CANDIDATES:
        peak = _probe_vram(trial, context_length, horizon, bs, device)
        if peak is None:
            continue
        if peak <= budget_gb:
            return bs, scale_lr(trial.learning_rate, bs), peak
    return None, None, None


# ==============================================================================
#  EVALUATION
# ==============================================================================

def _evaluate_mae(model: nn.Module, x_val_gpu: torch.Tensor,
                  y_val_gpu: torch.Tensor, bs: int) -> float:
    """Validation MAE. One host sync at end."""
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


# ==============================================================================
#  TRIAL
# ==============================================================================

def _run_single_trial(
    trial_idx: int, trial: TrialConfig, vram_budget_gb: float,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
    context_length: int, horizon: int, device: str,
) -> Dict[str, Any]:
    """Train one DLinear configuration with MAE loss; save best-val weights to
    a per-trial temp file. Parent decides which checkpoint to keep."""
    bs, lr, peak = auto_select_bs_lr(
        trial, context_length, horizon, device, vram_budget_gb,
    )
    if bs is None:
        return {
            "trial_idx": trial_idx, "device": device,
            "val_mae": float("nan"),
            "history": {"train_loss": [], "val_loss": [], "val_epochs": []},
            "best_state_path": None, "failed": True,
            "skip_reason": "vram_oom",
            "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
            "elapsed_seconds": 0.0, "cfg": asdict(trial),
        }

    tag = f"[{device}] trial {trial_idx:03d}"
    print(Fore.CYAN
          + f"  {tag}: kernel={trial.kernel_size} bs={bs} lr={lr:.2e} "
          + f"wd={trial.weight_decay:.0e} peak={peak:.2f}GB  "
          + f"budget={vram_budget_gb:.1f}GB" + Fore.RESET)

    t0 = time.perf_counter()
    n_train = x_train.shape[0]; steps_per_epoch = n_train // bs

    model = DLinear(
        context_length=context_length,
        prediction_length=horizon,
        kernel_size=trial.kernel_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=trial.weight_decay,
    )

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
                val_mae = _evaluate_mae(model, x_val, y_val, bs)
                history["val_loss"].append(val_mae)
                history["val_epochs"].append(epoch)
                print(Fore.YELLOW
                      + f"  {tag} epoch {epoch:3d}  "
                      + f"train_mae={train_loss:.4f}  val_mae={val_mae:.4f}"
                      + Fore.RESET)
                if val_mae < best_val:
                    best_val = val_mae
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

    best_state_path: Optional[str] = None
    if best_state is not None:
        fd, best_state_path = tempfile.mkstemp(
            prefix=f"dlinear_w_trial{trial_idx:03d}_", suffix=".pt")
        os.close(fd)
        torch.save(best_state, best_state_path)
        del best_state

    elapsed = time.perf_counter() - t0
    del optimizer, model
    torch.cuda.empty_cache()

    return {
        "trial_idx": trial_idx, "device": device,
        "val_mae": float(best_val) if best_val != float("inf") else float("nan"),
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
    """Worker process — owns one GPU. Message format: (trial_idx, TrialConfig)
    or None (poison)."""
    set_seed(SEED + worker_id)
    torch.cuda.set_device(torch.device(device))
    torch.set_float32_matmul_precision("high")

    ge_name, term, to_univariate = dataset_args

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
            trial_idx, trial = msg
            result_queue.put({
                "trial_idx": trial_idx, "device": device,
                "val_mae": float("nan"),
                "history": {}, "best_state_path": None,
                "failed": True, "skip_reason": "worker_setup_failed",
                "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
                "elapsed_seconds": 0.0, "cfg": asdict(trial),
            })
        return

    while True:
        try:
            msg = trial_queue.get(timeout=3600)
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
            print(Fore.RED + f"  [{device}] trial {trial_idx:03d} "
                  + f"CRASHED: {type(exc).__name__}: {exc}" + Fore.RESET)
            result = {
                "trial_idx": trial_idx, "device": device,
                "val_mae": float("nan"),
                "history": {}, "best_state_path": None,
                "failed": True,
                "skip_reason": f"trial_exception:{type(exc).__name__}",
                "auto_batch_size": None, "auto_lr": None, "peak_vram_gb": None,
                "elapsed_seconds": 0.0, "cfg": asdict(trial),
            }
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
):
    """Running-best scatter over trials."""
    trials_df = trials_df.sort_values("trial_idx").reset_index(drop=True)
    running_best = []; best_so_far = float("inf"); best_idx_in_df = []
    for i, row in trials_df.iterrows():
        v = row["val_mae"]
        if v < best_so_far:
            best_so_far = v
            best_idx_in_df.append(i)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(trials_df["trial_idx"], trials_df["val_mae"],
               s=36, alpha=0.55, color="#1f77b4",
               edgecolor="white", linewidth=0.5, label="trial")
    improved = trials_df.iloc[best_idx_in_df]
    ax.plot(improved["trial_idx"], improved["val_mae"],
            marker="o", markersize=9, linewidth=2.2, color="#d62728",
            label="best-so-far")
    best_row = trials_df.iloc[trials_df["val_mae"].idxmin()]
    ax.annotate(
        f"  best: trial {int(best_row['trial_idx'])}\n"
        f"  val MAE={best_row['val_mae']:.4f}",
        xy=(best_row["trial_idx"], best_row["val_mae"]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3b0", ec="#999", alpha=0.95),
    )
    ax.set_xlabel("Trial index", fontsize=12)
    ax.set_ylabel("val MAE", fontsize=12)
    ax.set_title(
        f"DLinear sweep — {dataset_display}  (term={term}, N={len(trials_df)})",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=10)
    plt.tight_layout(); plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


# ==============================================================================
#  CACHE HELPERS
# ==============================================================================

def _ds_dir(dataset_display: str, term: str) -> str:
    return os.path.join(CACHE_ROOT, dataset_display, term)

def _trial_json_path(dataset_display: str, term: str, trial_idx: int) -> str:
    return os.path.join(_ds_dir(dataset_display, term),
                        "trials", f"trial_{trial_idx:03d}.json")

def _best_model_path(dataset_display: str, term: str) -> str:
    return os.path.join(_ds_dir(dataset_display, term), "best_model.pt")

def _best_config_path(dataset_display: str, term: str) -> str:
    return os.path.join(_ds_dir(dataset_display, term), "best_config.json")

def _load_trial_result(dataset_display: str, term: str,
                       trial_idx: int) -> Optional[Dict]:
    p = _trial_json_path(dataset_display, term, trial_idx)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def _save_trial_result(dataset_display: str, term: str,
                       trial_idx: int, result: Dict):
    p = _trial_json_path(dataset_display, term, trial_idx)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # Drop non-serializable / heavy fields.
    serializable = {k: v for k, v in result.items() if k != "best_state_path"}
    with open(p, "w") as f:
        json.dump(serializable, f, indent=2)


# ==============================================================================
#  STAGE EXECUTION HELPER
# ==============================================================================

def _run_sweep(
    pending_trials: List[Tuple[int, TrialConfig]],
    workers: List[mp.Process],
    trial_queue: "mp.Queue", result_queue: "mp.Queue",
    dataset_display: str, term: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Enqueue trials and collect results. Returns (results, dataset_skipped)."""
    for trial_idx, trial in pending_trials:
        trial_queue.put((trial_idx, trial))

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
                      + f"{n_received}/{n_expected} results." + Fore.RESET)
                break
            print(Fore.YELLOW + f"  No result in 1h "
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
          + f"  Sweep wall-clock: {elapsed:.1f}s  "
          + f"({n_received}/{n_expected} trials)" + Fore.RESET)
    return results, dataset_skipped


# ==============================================================================
#  MAIN
# ==============================================================================

def _csv_keep_cols():
    return ["trial_idx", "val_mae", "kernel_size", "learning_rate",
            "weight_decay", "auto_batch_size", "auto_lr",
            "peak_vram_gb", "device", "elapsed_seconds"]


def _persist_artifacts(
    results: List[Dict[str, Any]], dataset_display: str, term: str,
):
    """Persist per-trial JSONs, sweep CSV, and summary plot."""
    ds_dir = _ds_dir(dataset_display, term)
    os.makedirs(os.path.join(ds_dir, "trials"), exist_ok=True)

    for r in results:
        _save_trial_result(dataset_display, term, r["trial_idx"], r)

    rows = []
    for r in results:
        flat = {"trial_idx": r["trial_idx"]}
        flat.update(r.get("cfg", {}))
        for key in ("val_mae", "auto_batch_size", "auto_lr",
                    "peak_vram_gb", "device", "elapsed_seconds"):
            if key in r:
                flat[key] = r[key]
        rows.append(flat)
    df = pd.DataFrame(rows)
    df_csv = df[[c for c in _csv_keep_cols() if c in df.columns]]
    csv_path = os.path.join(ds_dir, "sweep_summary.csv")
    df_csv.to_csv(csv_path, index=False)

    df_for_plot = df_csv.dropna(subset=["val_mae"]) if "val_mae" in df_csv else None
    if df_for_plot is not None and len(df_for_plot) > 0:
        _plot_sweep_summary(
            df_for_plot, os.path.join(ds_dir, "sweep_summary.png"),
            dataset_display, term,
        )
    print(Fore.GREEN + f"  Sweep summary: {csv_path}" + Fore.RESET)


def _select_best(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the trial with the lowest val_mae among non-failed runs whose
    checkpoint is still on disk."""
    candidates = [
        r for r in results
        if not r.get("failed", False)
        and r.get("best_state_path")
        and os.path.isfile(r["best_state_path"])
        and not (isinstance(r.get("val_mae"), float) and math.isnan(r["val_mae"]))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["val_mae"])


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
    print(Fore.CYAN + f"N_TRIALS={N_TRIALS}   select by val_mae   "
          + f"DLinear single-stage point forecaster" + Fore.RESET)

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

        os.makedirs(os.path.join(_ds_dir(dataset_display, term),
                                 "trials"), exist_ok=True)

        # Resolve which trials still need to run (cache-aware).
        results: List[Dict[str, Any]] = []
        pending: List[Tuple[int, TrialConfig]] = []
        for trial_idx, trial in enumerate(trial_configs):
            cached = _load_trial_result(dataset_display, term, trial_idx)
            if (cached is not None and "val_mae" in cached
                    and not (isinstance(cached["val_mae"], float)
                             and math.isnan(cached["val_mae"]))):
                print(Fore.WHITE + f"  CACHED trial {trial_idx:03d}: "
                      + f"val_mae={cached['val_mae']:.4f}" + Fore.RESET)
                # Cache hit — best_state_path is absent on disk, so this
                # trial is excluded from re-selection unless re-run.
                cached["best_state_path"] = None
                results.append(cached)
                continue
            pending.append((trial_idx, trial))

        # Spawn workers.
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
        if pending:
            print(Fore.CYAN + f"  Running {len(pending)} pending trials."
                  + Fore.RESET)
            new_results, dataset_skipped = _run_sweep(
                pending, workers, trial_queue, result_queue,
                dataset_display, term,
            )
            results.extend(new_results)
        else:
            print(Fore.GREEN + "  All trials already cached." + Fore.RESET)

        # Tear down workers.
        for _ in workers: trial_queue.put(None)
        for p in workers:
            p.join(timeout=120)
            if p.is_alive():
                print(Fore.RED + f"  Worker {p.name} still alive — terminating."
                      + Fore.RESET)
                p.terminate(); p.join(timeout=10)

        if dataset_skipped:
            continue

        # Persist sweep artifacts.
        _persist_artifacts(results, dataset_display, term)

        # Select best & promote checkpoint.
        chosen = _select_best(results)
        if chosen is not None:
            target_ckpt = _best_model_path(dataset_display, term)
            shutil.copyfile(chosen["best_state_path"], target_ckpt)
            with open(_best_config_path(dataset_display, term), "w") as f:
                json.dump({
                    "trial_idx":          int(chosen["trial_idx"]),
                    **chosen.get("cfg", {}),
                    "val_mae":            chosen.get("val_mae"),
                    "auto_batch_size":    chosen.get("auto_batch_size"),
                    "auto_lr":            chosen.get("auto_lr"),
                    "context_length":     CONTEXT_LENGTH,
                    "prediction_length":  horizon,
                    "num_input_channels": 1,
                    "model":              "DLinear",
                }, f, indent=2)
            print(Fore.GREEN + f"  FINAL SELECTION: trial "
                  + f"{int(chosen['trial_idx']):03d}  "
                  + f"(val_mae={chosen['val_mae']:.4f}, "
                  + f"kernel={chosen['cfg']['kernel_size']}, "
                  + f"lr={chosen['cfg']['learning_rate']:.0e}, "
                  + f"wd={chosen['cfg']['weight_decay']:.0e})" + Fore.RESET)
        else:
            print(Fore.YELLOW + "  No fresh checkpoints in this run. "
                  + "Keeping any previously persisted best_model.pt as-is."
                  + Fore.RESET)

        # Clean up per-trial temp checkpoints.
        for r in results:
            p = r.get("best_state_path")
            if p and os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

        # Accumulate to global summary.
        for r in results:
            global_rows.append({
                "dataset_display": dataset_display, "term": term,
                **{k: v for k, v in r.items()
                   if k not in ("history", "best_state_path", "cfg")},
                **r.get("cfg", {}),
            })

    # Global summary.
    global_df = pd.DataFrame(global_rows)
    global_csv = os.path.join(run_dir, "search_all.csv")
    global_df.to_csv(global_csv, index=False)
    print(Fore.GREEN + f"\n  Global summary: {global_csv}" + Fore.RESET)
    print(Fore.GREEN + "\nDLinear training pipeline done." + Fore.RESET)


if __name__ == "__main__":
    main()