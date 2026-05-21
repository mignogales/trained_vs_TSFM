"""
Unified test pipeline: PatchTST / DLinear (trained per dataset), AutoARIMA
(fit per series at test time), and zero-shot Foundation Models on GiftEval.

Adapted from the base window-ablation inference code. The window-size sweep
is removed; two fixed input windows are used:
    FIXED_WINDOW_TRAINED -> locally-trained PatchTST / DLinear (must match
                            training context_length)
    FIXED_WINDOW_FM      -> zero-shot Foundation Models (Chronos2, TimesFM,
                            Moirai2, ...). FMs are typically pretrained for
                            longer contexts, so this is usually
                            >= FIXED_WINDOW_TRAINED.

Model families
--------------
- patchtst_trained : weights loaded from main_train.py outputs.
- dlinear_trained  : weights loaded from dlinear_training.py outputs.
- arima_per_series : AutoARIMA fit per test series in parallel via
                     statsforecast. No checkpoint. Gaussian predictive
                     density -> analytic quantiles -> CRPS via pinball.
- chronos2 / timesfm / moirai : Zero-shot Foundation Models.

Metrics (GiftEval-compatible):
  MAE, MSE, RMSE, MASE, SMAPE, MAPE, ND, NRMSE, CRPS

Cache layout:
  logs/experiments/test_fixed_window/<dataset>/<model_short>/t<term>/metrics.json
"""

import torch
import torch.nn as nn
import os
import json
import math
import time
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
from scipy.stats import norm

from dotenv import load_dotenv
load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset

try:
    from tsfm_public import PatchTSTFMForPrediction
except ImportError:
    PatchTSTFMForPrediction = None

from transformers import PatchTSTConfig, PatchTSTForPrediction

# Lazy / optional ARIMA backend ------------------------------------------------
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, SeasonalNaive
    _HAS_STATSFORECAST = True
except ImportError:
    StatsForecast = None
    AutoARIMA = None
    SeasonalNaive = None
    _HAS_STATSFORECAST = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsforecast")


# ==============================================================================
#  EXPERIMENT CONFIGURATION
# ==============================================================================

# -- Fixed window sizes -------------------------------------------------------
FIXED_WINDOW_TRAINED = 512     # PatchTST / DLinear (must match training)
FIXED_WINDOW_FM      = 1024    # Chronos2, Moirai2, TimesFM, ...

# -- ARIMA: context length cap (per-series fit cost ~ linear in N) ------------
ARIMA_MAX_CONTEXT       = 3000
ARIMA_SEASONAL_CAP      = 24
ARIMA_QUANTILE_LEVELS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ARIMA_MEDIAN_IDX        = 4
ARIMA_N_JOBS            = -1   # statsforecast cross-series parallelism
ARIMA_VERBOSE           = True
ARIMA_KWARGS = dict(
    max_p=3, max_q=3,
    max_P=1, max_Q=1,
    max_d=2, max_D=1,
    stepwise=True,
    approximation=True,
    nmodels=94,
    method="lbfgsb",
    allowdrift=True,
    allowmean=True,
)
ARIMA_INFORMATION_CRITERION = "aicc"

# z-score for the 80% PI; used to recover sigma from `AutoARIMA-hi-80`.
_Z_80 = float(norm.ppf(0.90))


def window_size_for(model_family: str) -> int:
    """Pick the appropriate test window for a given model family."""
    if model_family in ("patchtst_trained", "dlinear_trained"):
        return FIXED_WINDOW_TRAINED
    if model_family == "arima_per_series":
        return ARIMA_MAX_CONTEXT
    return FIXED_WINDOW_FM


# -- Models -------------------------------------------------------------------
FM_MODELS = [
    ("autogluon/chronos-2-small",       "chronos2",  "Chronos2-Small"),
    ("autogluon/chronos-2-synth",       "chronos2",  "Chronos2-Synth"),
    ("google/timesfm-2.5-200m-pytorch", "timesfm",   "TimesFM2.5-200M"),
    # ("Salesforce/moirai-2.0-R-small", "moirai",    "Moirai2-Small"),
]

PATCHTST_TRAINED = ("local:patchtst-trained", "patchtst_trained", "PatchTST-Trained")
DLINEAR_TRAINED  = ("local:dlinear-trained",  "dlinear_trained",  "DLinear-Trained")
ARIMA_TRAINED    = ("local:arima",            "arima_per_series", "AutoARIMA")

MODELS = [PATCHTST_TRAINED, DLINEAR_TRAINED, ARIMA_TRAINED] + FM_MODELS


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


# -- Quantile / sampling configs ----------------------------------------------
MOIRAI2_QUANTILE_LEVELS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MOIRAI2_MEDIAN_IDX       = 4
TIMESFM_QUANTILE_LEVELS  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PATCHTST_TRAINED_NUM_SAMPLES = 100

INFERENCE_BATCH_SIZE = 128

CACHE_ROOT     = "logs/experiments/test_fixed_window"
PATCHTST_ROOT  = "logs/experiments/patchtst_training"
DLINEAR_ROOT   = "logs/experiments/dlinear_training"

PLOT_METRICS = ["mae", "mse", "rmse", "mase", "smape", "crps"]


# ==============================================================================
#  FORECAST RESULT CONTAINER
# ==============================================================================

@dataclass
class ForecastResult:
    median:           torch.Tensor
    samples:          Optional[torch.Tensor] = None
    quantiles:        Optional[torch.Tensor] = None
    quantile_levels:  Optional[List[float]]  = None


# ==============================================================================
#  SEASONALITY
# ==============================================================================

def _get_seasonality(freq_str: str) -> int:
    """Naive seasonal-MASE denominator period."""
    freq_map = {
        "S": 86400, "T": 1440, "5T": 288, "10T": 144, "15T": 96,
        "30T": 48, "H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1,
        "1H": 24, "1D": 7, "1W": 52, "1M": 12,
    }
    f = freq_str.upper().replace("MIN", "T").replace("HOURLY", "H")
    if f in freq_map:
        return freq_map[f]
    for k, v in freq_map.items():
        if f.endswith(k):
            return v
    return 1


def _infer_season_length_arima(freq_str: str) -> int:
    """Seasonal period for ARIMA (m). Matches `autoarima_baseline.py`."""
    import re
    if not freq_str:
        return 1
    s = freq_str.strip().lower().split("-")[0]
    m = re.match(r"^(\d+)\s*(?:t|min)$", s)
    if m:
        n_min = int(m.group(1))
        return max(1, (24 * 60) // n_min)
    m = re.match(r"^(\d+)\s*s$", s)
    if m:
        n_sec = int(m.group(1))
        return max(1, (24 * 3600) // n_sec)
    if s in ("t", "min"):
        return 24 * 60
    if s.startswith("h"):
        return 24
    if s.startswith("b") or s.startswith("d"):
        return 7
    if s.startswith("w"):
        return 52
    if s.startswith("m"):
        return 12
    if s.startswith("q"):
        return 4
    if s.startswith("y") or s.startswith("a"):
        return 1
    return 1


def compute_naive_seasonal_mae(dataset: GiftEvalDataset) -> float:
    season = _get_seasonality(dataset.freq)
    abs_diffs = []
    for entry in dataset.training_dataset:
        target = entry["target"]
        if target.ndim > 1:
            target = target[0]
        target = np.asarray(target, dtype=np.float64)
        if len(target) <= season:
            continue
        diffs = np.abs(target[season:] - target[:-season])
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) > 0:
            abs_diffs.append(diffs)
    if not abs_diffs:
        return 1.0
    return max(float(np.mean(np.concatenate(abs_diffs))), 1e-9)


# ==============================================================================
#  CACHE HELPERS
# ==============================================================================

def _cache_dir(dataset_display, model_short, term):
    return os.path.join(CACHE_ROOT, dataset_display, model_short, f"t{term}")


def _result_cached(dataset_display, model_short, term, expected_window: int):
    p = os.path.join(_cache_dir(dataset_display, model_short, term), "metrics.json")
    if not os.path.isfile(p):
        return False
    try:
        with open(p) as f:
            metrics = json.load(f)
        for key in ("mae", "mse", "rmse"):
            v = metrics.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
        cached_ws = metrics.get("window_size")
        if cached_ws is None or int(cached_ws) != int(expected_window):
            print(Fore.RED
                  + f"  WARNING Cached window_size={cached_ws} != expected "
                  + f"{expected_window} for {dataset_display}/{model_short}/t{term} "
                  + "-- will recompute" + Fore.RESET)
            return False
    except (json.JSONDecodeError, OSError):
        return False
    return True


def _load_cached_result(dataset_display, model_short, term):
    p = os.path.join(_cache_dir(dataset_display, model_short, term), "metrics.json")
    with open(p) as f:
        return json.load(f)


def _save_result(dataset_display, model_short, term, metrics: dict):
    d = _cache_dir(dataset_display, model_short, term)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# ==============================================================================
#  GIFT-EVAL -> BATCHED TENSORS
# ==============================================================================

def gifteval_to_batches(dataset: GiftEvalDataset, window_size: int, batch_size: int):
    horizon = dataset.prediction_length
    xs, ys = [], []
    for test_input, test_label in dataset.test_data:
        target = test_input["target"]
        label  = test_label["target"]
        if target.ndim > 1:
            raise ValueError(
                f"Expected univariate test target but got shape {target.shape}. "
                "Set to_univariate=True."
            )
        if label.ndim > 1:
            raise ValueError(
                f"Expected univariate test label but got shape {label.shape}. "
                "Set to_univariate=True."
            )

        context = target
        if len(context) < window_size:
            pad = np.zeros(window_size - len(context), dtype=np.float32)
            context = np.concatenate([pad, context])
        else:
            context = context[-window_size:]
        context = np.nan_to_num(context, nan=0.0)

        label = label[:horizon]
        if len(label) < horizon:
            continue

        xs.append(torch.tensor(context, dtype=torch.float32).unsqueeze(-1))
        ys.append(torch.tensor(label,  dtype=torch.float32).unsqueeze(-1))

    if not xs:
        raise RuntimeError(f"No valid test samples for window_size={window_size}.")

    all_x = torch.stack(xs)
    all_y = torch.stack(ys)

    batches = []
    for i in range(0, len(all_x), batch_size):
        batches.append({"x": all_x[i:i + batch_size], "y": all_y[i:i + batch_size]})
    return batches, all_x, all_y


def gifteval_to_series_list(dataset: GiftEvalDataset, max_context: int):
    """Variable-length context per test series, used by AutoARIMA.

    Unlike `gifteval_to_batches`, this does NOT left-pad short contexts:
    ARIMA is happy with whatever length is available (subject to a minimum)
    and padding would corrupt the fitted ARMA structure.
    """
    horizon = dataset.prediction_length
    contexts, labels = [], []
    for test_input, test_label in dataset.test_data:
        target = test_input["target"]
        label  = test_label["target"]
        if target.ndim > 1 or label.ndim > 1:
            raise ValueError(
                "Expected univariate target/label; set to_univariate=True."
            )
        target = np.asarray(target, dtype=np.float64)
        label  = np.asarray(label,  dtype=np.float64)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        if len(target) > max_context:
            target = target[-max_context:]
        if len(label) < horizon:
            continue
        # ARIMA needs at least a handful of points to fit anything useful.
        if len(target) < 8:
            continue
        contexts.append(target.astype(np.float32))
        labels.append(label[:horizon].astype(np.float32))
    if not contexts:
        raise RuntimeError("No valid ARIMA test series.")
    return contexts, labels


# ==============================================================================
#  METRICS
# ==============================================================================

def crps_energy_score(samples: torch.Tensor, targets: torch.Tensor) -> float:
    N, S, H = samples.shape
    term1 = (samples - targets.unsqueeze(1)).abs().mean(dim=1)
    sorted_samples, _ = samples.sort(dim=1)
    indices = torch.arange(1, S + 1, device=samples.device, dtype=samples.dtype)
    weights = (2 * indices - S - 1).reshape(1, S, 1)
    term2 = (weights * sorted_samples).sum(dim=1) / (S * S)
    return (term1 - term2).mean().item()


def crps_quantile_loss(quantiles, quantile_levels, targets):
    Q = len(quantile_levels)
    tau = torch.tensor(quantile_levels, dtype=quantiles.dtype,
                       device=quantiles.device).reshape(1, Q, 1)
    y = targets.unsqueeze(1)
    errors = y - quantiles
    pinball = torch.where(errors >= 0, tau * errors, (tau - 1) * errors)
    return (2.0 / Q) * pinball.mean().item()


def compute_all_metrics(forecast_result: ForecastResult, targets: torch.Tensor,
                       naive_seasonal_mae: float = 1.0) -> dict:
    pred = forecast_result.median
    y = targets
    valid = ~torch.isnan(y)
    if not valid.all():
        y_safe = y.clone(); y_safe[~valid] = 0.0
        pred_safe = pred.clone(); pred_safe[~valid] = 0.0
    else:
        y_safe, pred_safe = y, pred

    abs_err = (pred_safe - y_safe).abs()
    sq_err  = (pred_safe - y_safe) ** 2

    n_valid = valid.sum().float()
    if n_valid == 0:
        return {k: float("nan") for k in
                ["mae","mse","rmse","mase","smape","mape","nd","nrmse","crps"]}

    mae   = abs_err[valid].mean().item()
    mse   = sq_err[valid].mean().item()
    rmse  = float(np.sqrt(mse))
    mase  = mae / naive_seasonal_mae
    denom_smape = (pred_safe.abs() + y_safe.abs()).clamp(min=1e-13)
    smape = (2.0 * abs_err / denom_smape)[valid].mean().item()
    mape  = (abs_err / y_safe.abs().clamp(min=1e-13))[valid].mean().item()
    nd    = abs_err[valid].sum().item() / y_safe.abs()[valid].sum().clamp(min=1e-13).item()
    nrmse = rmse / y_safe.abs()[valid].mean().clamp(min=1e-13).item()

    crps = float("nan")
    if forecast_result.samples is not None:
        crps = crps_energy_score(forecast_result.samples, y_safe)
    elif forecast_result.quantiles is not None and forecast_result.quantile_levels is not None:
        crps = crps_quantile_loss(
            forecast_result.quantiles, forecast_result.quantile_levels, y_safe,
        )

    return {"mae": mae, "mse": mse, "rmse": rmse, "mase": mase,
            "smape": smape, "mape": mape, "nd": nd, "nrmse": nrmse, "crps": crps}


# ==============================================================================
#  FOUNDATION MODELS
# ==============================================================================

def load_chronos2(model_id, device):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(
        model_id, device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def predict_chronos2(pipeline, batches, horizon, device):
    all_samples, all_tgts = [], []
    with torch.no_grad():
        for batch in tqdm(batches, desc="  Chronos2 predict", leave=False):
            x, y = batch["x"], batch["y"]
            context = x.permute(0, 2, 1)
            samples = pipeline.predict(inputs=context, prediction_length=horizon)
            samples = (torch.stack(samples, dim=0).squeeze(1)
                       if isinstance(samples, list) else samples)
            if samples.dim() == 4:
                samples = samples.squeeze(2)
            all_samples.append(samples.cpu().float())
            all_tgts.append(y[:, :, 0].cpu())
    all_samples = torch.cat(all_samples, 0)
    all_tgts    = torch.cat(all_tgts, 0)
    median = torch.median(all_samples, dim=1).values
    return ForecastResult(median=median, quantiles=all_samples,
                          quantile_levels=pipeline.quantiles), all_tgts


def load_moirai_module(model_id):
    from uni2ts.model.moirai2 import Moirai2Module
    return Moirai2Module.from_pretrained(model_id)


def build_moirai_forecast(module, horizon, window_size, device):
    from uni2ts.model.moirai2 import Moirai2Forecast
    return Moirai2Forecast(
        module=module, prediction_length=horizon, context_length=window_size,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(device)


def predict_moirai(model, batches, horizon, device):
    all_q, all_tgts = [], []
    with torch.no_grad():
        for batch in tqdm(batches, desc="  Moirai predict", leave=False):
            x, y = batch["x"], batch["y"]
            bs = x.shape[0]
            ctx_list = [x[i, :, 0].numpy() for i in range(bs)]
            forecast = model.predict(past_target=ctx_list)
            forecast_t = torch.tensor(forecast[:, :, :horizon], dtype=torch.float32)
            all_q.append(forecast_t.cpu())
            all_tgts.append(y[:, :, 0].cpu())
    all_q    = torch.cat(all_q, 0)
    all_tgts = torch.cat(all_tgts, 0)
    median = all_q[:, MOIRAI2_MEDIAN_IDX, :]
    return ForecastResult(median=median, quantiles=all_q,
                          quantile_levels=MOIRAI2_QUANTILE_LEVELS), all_tgts


def load_timesfm(model_id, window_size, horizon, batch_size):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(timesfm.ForecastConfig(
        max_context=window_size, max_horizon=horizon,
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, per_core_batch_size=batch_size,
        infer_is_positive=True, fix_quantile_crossing=True,
    ))
    return model


def predict_timesfm(model, batches, horizon, device):
    PATCH = 32
    all_med, all_q, all_tgts = [], [], []
    with torch.no_grad():
        for batch in tqdm(batches, desc="  TimesFM predict", leave=False):
            x, y = batch["x"], batch["y"]
            bs, ws = x.shape[0], x.shape[1]
            remainder = ws % PATCH
            pad_len = (PATCH - remainder) % PATCH
            raw = x[:, :, 0].numpy()
            if pad_len > 0:
                padding = np.zeros((bs, pad_len), dtype=np.float32)
                padded = np.concatenate([padding, raw], axis=1)
                mask_pad = np.ones((bs, pad_len), dtype=bool)
                mask_valid = np.zeros((bs, ws), dtype=bool)
                masks_np = np.concatenate([mask_pad, mask_valid], axis=1)
            else:
                padded = raw
                masks_np = np.zeros((bs, ws), dtype=bool)
            values = [padded[i] for i in range(bs)]
            masks  = [masks_np[i] for i in range(bs)]
            point_forecast, q_forecast = model.compiled_decode(horizon, values, masks)
            pf = torch.tensor(point_forecast[:, :horizon], dtype=torch.float32)
            qf = torch.tensor(q_forecast[:, :horizon, 1:], dtype=torch.float32).permute(0, 2, 1)
            all_med.append(pf.cpu()); all_q.append(qf.cpu())
            all_tgts.append(y[:, :, 0].cpu())
    all_med  = torch.cat(all_med, 0)
    all_q    = torch.cat(all_q, 0)
    all_tgts = torch.cat(all_tgts, 0)
    return ForecastResult(median=all_med, quantiles=all_q,
                          quantile_levels=TIMESFM_QUANTILE_LEVELS), all_tgts


# ==============================================================================
#  PATCHTST (locally trained)
# ==============================================================================

def load_patchtst_trained(dataset_display: str, term: str, device: str
                          ) -> PatchTSTForPrediction:
    cfg_path = os.path.join(PATCHTST_ROOT, dataset_display, term, "best_config.json")
    pt_path  = os.path.join(PATCHTST_ROOT, dataset_display, term, "best_model.pt")
    if not os.path.isfile(cfg_path) or not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"No PatchTST checkpoint for ({dataset_display}, {term}). "
            f"Run main_train.py first.  Expected: {cfg_path} + {pt_path}"
        )
    with open(cfg_path) as f:
        meta = json.load(f)

    hf_cfg = PatchTSTConfig(
        num_input_channels   = int(meta["num_input_channels"]),
        context_length       = int(meta["context_length"]),
        prediction_length    = int(meta["prediction_length"]),
        patch_length         = int(meta["patch_length"]),
        patch_stride         = max(1, int(meta["patch_length"]) // 2),
        d_model              = int(meta["d_model"]),
        num_attention_heads  = int(meta["num_attention_heads"]),
        num_hidden_layers    = int(meta["num_hidden_layers"]),
        ffn_dim              = int(meta["d_model"]) * 4,
        dropout              = float(meta["dropout"]),
        head_dropout         = float(meta["dropout"]),
        attention_dropout    = float(meta["dropout"]),
        loss                 = "nll",
        distribution_output  = "student_t",
        scaling              = "std",
        num_parallel_samples = PATCHTST_TRAINED_NUM_SAMPLES,
    )
    model = PatchTSTForPrediction(hf_cfg)
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"    [PatchTST-Trained] context={hf_cfg.context_length}  "
          f"horizon={hf_cfg.prediction_length}  trial={meta.get('trial_idx', '?')}  "
          f"val_loss={meta.get('val_loss', '?')}")
    return model


def predict_patchtst_trained(model, batches, horizon, device):
    all_samples, all_tgts = [], []
    with torch.no_grad():
        for batch in tqdm(batches, desc="  PatchTST-Trained predict", leave=False):
            x, y = batch["x"], batch["y"]
            x = x.to(device, non_blocking=True)
            out = model.generate(past_values=x)
            seq = out.sequences
            if seq.dim() == 4:
                seq = seq.squeeze(-1)
            all_samples.append(seq.cpu().float())
            all_tgts.append(y[:, :, 0].cpu())
    all_samples = torch.cat(all_samples, 0)
    all_tgts    = torch.cat(all_tgts, 0)
    median = torch.median(all_samples, dim=1).values
    return ForecastResult(median=median, samples=all_samples), all_tgts


# ==============================================================================
#  DLINEAR (locally trained)
# ==============================================================================
#
# Architecture matches dlinear_training.py exactly so checkpoints load cleanly
# via `load_state_dict(strict=True)`. DLinear is a point forecaster only --
# no probabilistic head is attached and CRPS is left as NaN.

class _DLinearMovingAvg(nn.Module):
    """Causal-symmetric moving average via edge-replication padding."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got {kernel_size}.")
        self.kernel_size = kernel_size
        self.pad_each_side = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, self.pad_each_side, 1)
        end   = x[:, -1:, :].repeat(1, self.pad_each_side, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_bcl = x_pad.permute(0, 2, 1)
        return self.avg(x_bcl).permute(0, 2, 1)


class _DLinearSeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _DLinearMovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        return x - trend, trend


@dataclass
class _DLinearOutput:
    prediction_outputs: torch.Tensor
    loss: Optional[torch.Tensor] = None


class DLinear(nn.Module):
    """Channel-independent DLinear (Zeng et al. 2023) with per-instance
    std normalization. Must mirror dlinear_training.DLinear exactly."""

    def __init__(self, context_length, prediction_length, kernel_size,
                 num_channels: int = 1, individual: bool = False):
        super().__init__()
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.num_channels      = num_channels
        self.individual        = individual
        self.decomposition     = _DLinearSeriesDecomposition(kernel_size)
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

    def forward(self, past_values: torch.Tensor,
                future_values: Optional[torch.Tensor] = None) -> _DLinearOutput:
        mean = past_values.mean(dim=1, keepdim=True)
        std  = past_values.std(dim=1, keepdim=True).clamp_min(1e-5)
        x_norm = (past_values - mean) / std

        seasonal, trend = self.decomposition(x_norm)
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

        pred_norm = (seasonal_out + trend_out).permute(0, 2, 1)
        prediction = pred_norm * std + mean

        loss = None
        if future_values is not None:
            loss = (prediction - future_values).abs().mean()
        return _DLinearOutput(prediction_outputs=prediction, loss=loss)


def load_dlinear_trained(dataset_display: str, term: str, device: str) -> DLinear:
    cfg_path = os.path.join(DLINEAR_ROOT, dataset_display, term, "best_config.json")
    pt_path  = os.path.join(DLINEAR_ROOT, dataset_display, term, "best_model.pt")
    if not os.path.isfile(cfg_path) or not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"No DLinear checkpoint for ({dataset_display}, {term}). "
            f"Run dlinear_training.py first.  Expected: {cfg_path} + {pt_path}"
        )
    with open(cfg_path) as f:
        meta = json.load(f)
    model = DLinear(
        context_length    = int(meta["context_length"]),
        prediction_length = int(meta["prediction_length"]),
        kernel_size       = int(meta["kernel_size"]),
        num_channels      = int(meta.get("num_input_channels", 1)),
    )
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"    [DLinear-Trained] context={model.context_length}  "
          f"horizon={model.prediction_length}  "
          f"kernel={meta.get('kernel_size', '?')}  "
          f"trial={meta.get('trial_idx', '?')}  "
          f"val_mae={meta.get('val_mae', '?')}")
    return model


def predict_dlinear_trained(model: DLinear, batches, horizon, device):
    """DLinear is a point forecaster -- no `samples` or `quantiles`."""
    all_pred, all_tgts = [], []
    with torch.no_grad():
        for batch in tqdm(batches, desc="  DLinear-Trained predict", leave=False):
            x, y = batch["x"], batch["y"]
            x = x.to(device, non_blocking=True)
            out = model(past_values=x)
            pred = out.prediction_outputs               # (B, H, C)
            if pred.dim() == 3:
                pred = pred[:, :, 0]                    # (B, H)
            all_pred.append(pred.cpu().float())
            all_tgts.append(y[:, :, 0].cpu())
    all_pred = torch.cat(all_pred, 0)
    all_tgts = torch.cat(all_tgts, 0)
    return ForecastResult(median=all_pred), all_tgts


# ==============================================================================
#  AUTOARIMA (fit per series at test time)
# ==============================================================================

def _arima_recover_sigma(mu: np.ndarray, hi80: np.ndarray) -> np.ndarray:
    """Invert hi80 = mu + z * sigma (Gaussian); floor for numerical safety."""
    sigma = (hi80 - mu) / _Z_80
    return np.maximum(sigma, 1e-8)


def predict_arima(ge_dataset: GiftEvalDataset, horizon: int):
    """Fit AutoARIMA per test series in parallel via statsforecast, then
    convert the Gaussian predictive density into a quantile forecast for
    CRPS scoring.

    Notes
    -----
    * `gifteval_to_series_list` returns variable-length contexts (no padding).
    * Each series gets a synthetic timestamp index sharing `ge_dataset.freq`,
      so all series can be concatenated into a single long DataFrame and
      fit in parallel by joblib (`n_jobs=-1`).
    * Predictive law is Gaussian, so quantiles at any tau are recovered
      analytically as mu + Phi^{-1}(tau) * sigma; CRPS is then the standard
      pinball-loss approximation.
    """
    if not _HAS_STATSFORECAST:
        raise RuntimeError(
            "AutoARIMA requested but `statsforecast` is not installed. "
            "Install via `pip install statsforecast`."
        )

    freq       = ge_dataset.freq
    season_raw = _infer_season_length_arima(freq)
    season     = season_raw if season_raw <= ARIMA_SEASONAL_CAP else 1
    if season != season_raw:
        print(Fore.YELLOW
              + f"    [ARIMA] season raw={season_raw} > cap="
              + f"{ARIMA_SEASONAL_CAP}; disabling seasonal ARIMA (m=1)."
              + Fore.RESET)

    contexts, labels = gifteval_to_series_list(ge_dataset, ARIMA_MAX_CONTEXT)
    print(f"    [ARIMA] series={len(contexts)}  freq='{freq}'  "
          f"season_length={season}  horizon={horizon}")

    # Build the long-format DataFrame statsforecast expects.
    train_rows: List[pd.DataFrame] = []
    uids: List[str] = []
    for idx, ctx in enumerate(contexts):
        uid = f"S{idx:06d}"
        ds  = pd.date_range("2000-01-01", periods=len(ctx), freq=freq)
        train_rows.append(pd.DataFrame({
            "unique_id": uid, "ds": ds, "y": ctx.astype(np.float32),
        }))
        uids.append(uid)
    train_df = pd.concat(train_rows, ignore_index=True)

    model    = AutoARIMA(season_length=season, ic=ARIMA_INFORMATION_CRITERION,
                         **ARIMA_KWARGS)
    fallback = SeasonalNaive(season_length=max(1, season))
    sf       = StatsForecast(models=[model], freq=freq, n_jobs=ARIMA_N_JOBS,
                             fallback_model=fallback, verbose=ARIMA_VERBOSE)

    t_fit = time.perf_counter()
    sf.fit(train_df)
    print(f"    [ARIMA] fit done in {time.perf_counter() - t_fit:.1f}s")

    t_pred = time.perf_counter()
    pred = sf.predict(h=horizon, level=[80])
    print(f"    [ARIMA] predict done in {time.perf_counter() - t_pred:.1f}s")

    # ---- align predictions back to series order ----
    N        = len(uids)
    all_mu   = np.zeros((N, horizon), dtype=np.float32)
    all_sig  = np.zeros((N, horizon), dtype=np.float32)
    for i, uid in enumerate(uids):
        p = pred[pred["unique_id"] == uid].sort_values("ds")
        mu_i   = p["AutoARIMA"].to_numpy(dtype=np.float64)[:horizon]
        hi80_i = p["AutoARIMA-hi-80"].to_numpy(dtype=np.float64)[:horizon]
        if len(mu_i) < horizon:
            # statsforecast occasionally drops the last step for ill-fit
            # series. Forward-fill with last available value.
            pad = horizon - len(mu_i)
            mu_i   = np.concatenate([mu_i, np.full(pad, mu_i[-1] if len(mu_i) else 0.0)])
            hi80_i = np.concatenate([hi80_i, np.full(pad, hi80_i[-1] if len(hi80_i) else 0.0)])
        all_mu[i]  = mu_i.astype(np.float32)
        all_sig[i] = _arima_recover_sigma(mu_i, hi80_i).astype(np.float32)

    # ---- Build quantile tensor analytically ----
    z_q = norm.ppf(np.asarray(ARIMA_QUANTILE_LEVELS))            # (Q,)
    # quantiles[n, q, h] = mu[n, h] + z_q[q] * sigma[n, h]
    quantiles_np = (all_mu[:, None, :]
                    + z_q[None, :, None] * all_sig[:, None, :])  # (N, Q, H)
    quantiles_t  = torch.tensor(quantiles_np, dtype=torch.float32)
    median       = quantiles_t[:, ARIMA_MEDIAN_IDX, :]
    targets_t    = torch.tensor(np.stack(labels, axis=0), dtype=torch.float32)

    return (
        ForecastResult(median=median, quantiles=quantiles_t,
                       quantile_levels=ARIMA_QUANTILE_LEVELS),
        targets_t,
    )


# ==============================================================================
#  COMPARISON / PLOTTING
# ==============================================================================

def build_comparison_table(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return results_df.pivot_table(
        index=["dataset_display", "term"],
        columns="model_short",
        values=metric,
        aggfunc="mean",
    )


import matplotlib as mpl

_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

_RC_PARAMS = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "cm",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8,
    "figure.dpi":        200,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def plot_dataset_grid(results_df: pd.DataFrame, metrics: List[str],
                      save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    term_order  = ["short", "medium", "long"]
    model_order = [m[2] for m in MODELS]
    metrics     = [m for m in metrics if m in results_df.columns]
    if not metrics:
        return

    n = len(metrics)
    grid_map = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2),
                5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3)}
    nrows, ncols = grid_map.get(n, (int(np.ceil(n / 3)), 3))

    with mpl.rc_context(_RC_PARAMS):
        for dataset_display, sub in results_df.groupby("dataset_display"):
            present_terms  = [t for t in term_order  if t in sub["term"].unique()]
            present_models = [m for m in model_order if m in sub["model_short"].unique()]
            if not present_terms or not present_models:
                continue

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(3.6 * ncols, 2.9 * nrows),
                constrained_layout=True,
                squeeze=False,
            )
            axes_flat = axes.flatten()

            x = np.arange(len(present_terms))
            n_models = len(present_models)
            bar_w = 0.8 / max(n_models, 1)

            for ax, metric in zip(axes_flat, metrics):
                pivot = sub.pivot_table(
                    index="term", columns="model_short",
                    values=metric, aggfunc="mean",
                ).reindex(index=present_terms, columns=present_models)

                if pivot.isna().all().all():
                    ax.set_visible(False)
                    continue

                for i, model in enumerate(present_models):
                    vals = pivot[model].to_numpy(dtype=float)
                    offset = (i - (n_models - 1) / 2) * bar_w
                    ax.bar(
                        x + offset, vals, width=bar_w,
                        color=_PALETTE[i % len(_PALETTE)],
                        edgecolor="white", linewidth=0.6,
                        label=model,
                    )

                ax.set_title(metric.upper(), fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(present_terms)
                ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
                ax.set_axisbelow(True)
                for spine in ("left", "bottom"):
                    ax.spines[spine].set_linewidth(0.6)
                    ax.spines[spine].set_color("#888888")

            for ax in axes_flat[len(metrics):]:
                ax.set_visible(False)

            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(
                handles, labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=min(n_models, 4),
                frameon=False,
            )
            fig.suptitle(
                f"{dataset_display}  "
                f"(w$_\\mathrm{{trained}}$={FIXED_WINDOW_TRAINED}, "
                f"w$_\\mathrm{{FM}}$={FIXED_WINDOW_FM})",
                fontsize=12, fontweight="bold",
            )

            safe_name = dataset_display.replace("/", "_")
            fig.savefig(
                os.path.join(save_dir, f"{safe_name}.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CACHE_ROOT, f"runs/{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    print(Fore.CYAN
          + f"Device: {device}  |  w(trained)={FIXED_WINDOW_TRAINED}  "
          + f"w(FM)={FIXED_WINDOW_FM}  w(ARIMA)={ARIMA_MAX_CONTEXT}" + Fore.RESET)

    all_results = []

    ge_dataset_cache: Dict[tuple, GiftEvalDataset] = {}
    naive_mae_cache: Dict[tuple, float] = {}

    for model_id, model_family, model_short in MODELS:
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  MODEL: {model_id}  ({model_family})" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        pipeline      = None
        moirai_module = None

        if model_family == "chronos2":
            pipeline = load_chronos2(model_id, device)
        elif model_family == "moirai":
            moirai_module = load_moirai_module(model_id)
        # timesfm, patchtst_trained, dlinear_trained: loaded per (ds, term).
        # arima_per_series: stateless, no loader.

        for ge_name, term, dataset_display, to_univariate in DATASETS:
            window_size = window_size_for(model_family)
            tag = f"{model_short} | {dataset_display} | term={term} | w={window_size}"

            # -- Cache hit ----------------------------------------------------
            if _result_cached(dataset_display, model_short, term, window_size):
                cached = _load_cached_result(dataset_display, model_short, term)
                print(Fore.WHITE + f"  CACHED  {tag}  ->  MAE={cached['mae']:.6f}"
                      + Fore.RESET)
                all_results.append({
                    "model": model_id, "model_short": model_short,
                    "model_family": model_family,
                    "dataset": ge_name, "dataset_display": dataset_display,
                    "term": term, "window_size": window_size, **cached,
                })
                continue

            print(Fore.YELLOW + f"\n  > {tag}" + Fore.RESET)

            # -- Load dataset (once per (name, term)) -------------------------
            ds_key = (ge_name, term)
            if ds_key in ge_dataset_cache:
                ge_dataset = ge_dataset_cache[ds_key]
            else:
                ge_dataset = GiftEvalDataset(
                    name=ge_name, term=term, to_univariate=to_univariate,
                )
                ge_dataset_cache[ds_key] = ge_dataset

            horizon = ge_dataset.prediction_length

            # -- Naive seasonal MAE (for MASE) --------------------------------
            if ds_key not in naive_mae_cache:
                naive_mae_cache[ds_key] = compute_naive_seasonal_mae(ge_dataset)
            naive_mae = naive_mae_cache[ds_key]

            t_start = time.perf_counter()

            # =================================================================
            #  ARIMA path: bypasses fixed-window tensor batching.
            # =================================================================
            if model_family == "arima_per_series":
                try:
                    forecast_result, targets = predict_arima(ge_dataset, horizon)
                except RuntimeError as exc:
                    print(Fore.RED + f"    SKIP: {exc}" + Fore.RESET)
                    continue
                except Exception as exc:
                    print(Fore.RED + f"    FAILED: {type(exc).__name__}: {exc}"
                          + Fore.RESET)
                    continue

            # =================================================================
            #  All other families: standard fixed-window tensor batches.
            # =================================================================
            else:
                try:
                    batches, all_inputs, _ = gifteval_to_batches(
                        ge_dataset, window_size, INFERENCE_BATCH_SIZE,
                    )
                except RuntimeError as exc:
                    print(Fore.RED + f"    SKIP: {exc}" + Fore.RESET)
                    continue
                print(f"    Samples: {all_inputs.shape[0]}  "
                      f"Batches: {len(batches)}  "
                      f"horizon={horizon}  window={window_size}")

                if model_family == "chronos2":
                    forecast_result, targets = predict_chronos2(
                        pipeline, batches, horizon, device,
                    )
                elif model_family == "moirai":
                    moirai_model = build_moirai_forecast(
                        moirai_module, horizon, window_size, device,
                    )
                    forecast_result, targets = predict_moirai(
                        moirai_model, batches, horizon, device,
                    )
                    del moirai_model
                    torch.cuda.empty_cache()
                elif model_family == "timesfm":
                    tfm_model = load_timesfm(
                        model_id, window_size, horizon, INFERENCE_BATCH_SIZE,
                    )
                    forecast_result, targets = predict_timesfm(
                        tfm_model, batches, horizon, device,
                    )
                    del tfm_model
                    torch.cuda.empty_cache()
                elif model_family == "patchtst_trained":
                    try:
                        pt_model = load_patchtst_trained(dataset_display, term, device)
                    except FileNotFoundError as exc:
                        print(Fore.RED + f"    SKIP PatchTST: {exc}" + Fore.RESET)
                        continue
                    if pt_model.config.context_length != window_size:
                        print(Fore.RED
                              + f"    SKIP PatchTST: trained context_length="
                              + f"{pt_model.config.context_length} != "
                              + f"FIXED_WINDOW_TRAINED={window_size}. "
                              + "Retrain with matching window." + Fore.RESET)
                        del pt_model
                        torch.cuda.empty_cache()
                        continue
                    forecast_result, targets = predict_patchtst_trained(
                        pt_model, batches, horizon, device,
                    )
                    del pt_model
                    torch.cuda.empty_cache()
                elif model_family == "dlinear_trained":
                    try:
                        dl_model = load_dlinear_trained(dataset_display, term, device)
                    except FileNotFoundError as exc:
                        print(Fore.RED + f"    SKIP DLinear: {exc}" + Fore.RESET)
                        continue
                    if dl_model.context_length != window_size:
                        print(Fore.RED
                              + f"    SKIP DLinear: trained context_length="
                              + f"{dl_model.context_length} != "
                              + f"FIXED_WINDOW_TRAINED={window_size}. "
                              + "Retrain with matching window." + Fore.RESET)
                        del dl_model
                        torch.cuda.empty_cache()
                        continue
                    forecast_result, targets = predict_dlinear_trained(
                        dl_model, batches, horizon, device,
                    )
                    del dl_model
                    torch.cuda.empty_cache()
                else:
                    raise ValueError(f"Unknown model family: {model_family}")

            elapsed = time.perf_counter() - t_start

            metrics = compute_all_metrics(forecast_result, targets, naive_mae)
            metrics["elapsed_seconds"] = round(elapsed, 3)
            metrics["horizon"]         = horizon
            metrics["window_size"]     = window_size

            for k, v in metrics.items():
                if isinstance(v, float):
                    print(Fore.YELLOW + f"    {k}: {v:.6f}" + Fore.RESET)
            print(Fore.MAGENTA + f"    TIME {elapsed:.1f}s" + Fore.RESET)

            _save_result(dataset_display, model_short, term, metrics)

            all_results.append({
                "model": model_id, "model_short": model_short,
                "model_family": model_family,
                "dataset": ge_name, "dataset_display": dataset_display,
                "term": term, **metrics,
            })

        del pipeline, moirai_module
        torch.cuda.empty_cache()

    # ==========================================================================
    #  AGGREGATE  &  COMPARISON
    # ==========================================================================
    results_df = pd.DataFrame(all_results)
    print(Fore.GREEN + "\n" + "=" * 78 + Fore.RESET)
    print(Fore.GREEN + "  FULL RESULTS" + Fore.RESET)
    print(Fore.GREEN + "=" * 78 + Fore.RESET)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    # Per-metric pivoted CSVs.
    for metric in PLOT_METRICS:
        if metric not in results_df.columns:
            continue
        build_comparison_table(results_df, metric).to_csv(
            os.path.join(run_dir, f"comparison_{metric}.csv")
        )

    # One figure per dataset, all metrics in a grid.
    plot_dataset_grid(
        results_df, PLOT_METRICS,
        os.path.join(run_dir, "per_dataset"),
    )

    # Single comparison_table.csv with MAE for the quick-glance view.
    main_pivot = build_comparison_table(results_df, "mae")
    main_pivot.to_csv(os.path.join(run_dir, "comparison_table.csv"))
    print(Fore.GREEN + f"\n  Comparison table (MAE):\n{main_pivot}" + Fore.RESET)
    print(Fore.GREEN + f"  Outputs in: {run_dir}" + Fore.RESET)

    print(Fore.GREEN + "\nTest pipeline done." + Fore.RESET)


if __name__ == "__main__":
    main()