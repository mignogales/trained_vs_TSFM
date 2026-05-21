"""
AutoARIMA baseline evaluation pipeline for GiftEval datasets.

For each (dataset, term) pair this script
    1. Loads the GiftEval univariate dataset.
    2. Splits each series into a training portion and a trailing
       prediction-window (horizon) held-out tail.
    3. Fits per-series AutoARIMA (Nixtla `statsforecast`) on the training
       portion. AutoARIMA performs its own (p,d,q)(P,D,Q)_m order search via
       an information criterion — there is no outer hyper-parameter sweep.
    4. Generates a Gaussian predictive density (mu, sigma) for every step of
       the horizon.
    5. Scores the predictions with the same metric panel used in the
       PatchTST pipeline: NLL, MAE, mean CRPS over the horizon, mean pinball
       over a quantile grid, and 80% prediction-interval coverage. All
       metrics are *closed form* — Gneiting & Raftery (2007), Eq. (5) for
       CRPS — since the predictive law is Gaussian.

Conceptual departures from the PatchTST pipeline
-------------------------------------------------
- No random search. AutoARIMA subsumes the outer model-selection loop.
- No two-stage MAE -> NLL refinement. A single MLE fit produces both point
  and probabilistic forecasts simultaneously.
- No GPU machinery (VRAM probing, auto batch-size, LR scaling). Parallelism
  is per-series CPU via `statsforecast.n_jobs` (Numba-jitted backend).
- Validation protocol is *trailing-horizon-per-series*, not sliding windows.
  Re-fitting ARIMA at every origin is prohibitive and the trailing protocol
  is the canonical classical-baseline evaluation for GiftEval.

Cache layout
------------
logs/experiments/autoarima_baseline/<dataset>/<term>/
    metrics_per_series.csv       # one row per series with all metrics
    aggregate_metrics.json       # cross-series mean / std / median / IQR
    orders.csv                   # selected (p,d,q)(P,D,Q)_m orders per series
    metrics_summary.png          # diagnostic plot (CRPS histogram + coverage)
"""

import os
import re
import json
import time
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from scipy.stats import norm
from colorama import Fore

from dotenv import load_dotenv
load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsforecast")


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


# -- Probabilistic metric configuration ---------------------------------------
QUANTILE_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))   # 0.05 ... 0.95
COVERAGE_LO, COVERAGE_HI = 0.10, 0.90                              # 80% PI

# -- AutoARIMA configuration --------------------------------------------------
# Search-space and estimation settings tuned for GiftEval-scale wall-clock.
#   * max_p/max_q reduced from 5 to 3, max_P/max_Q from 2 to 1: stepwise
#     search visits a much smaller neighbourhood. Empirically ~95% of
#     selected GiftEval orders fall within (3,1,3)(1,1,1)_m anyway.
#   * approximation=True: order selection via conditional sum-of-squares
#     (cheap, no Kalman filter); the chosen model is then refit with full
#     MLE. This is the default in forecast::auto.arima when N > 150 or m>12
#     (Hyndman & Khandakar 2008).
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
INFORMATION_CRITERION = "aicc"  # ic argument propagated to AutoARIMA

# Seasonal-period cap. AutoARIMA's per-likelihood Kalman-filter cost is
# O(N * m^2): the state vector grows with the seasonal period m. For
# sub-daily data m easily exceeds 100 (144 for 10-min freq, 288 for 5-min),
# pushing stepwise search into hours per series. Setting an effective cap
# disables the seasonal ARIMA component above this threshold and falls
# back on non-seasonal ARIMA -- the standard recommendation in
# Hyndman & Athanasopoulos (FPP3, sec. 9.9). Complement with a separate
# SeasonalNaive baseline if seasonal accuracy matters.
SEASONAL_PERIOD_CAP: int = 24

# -- Runtime ------------------------------------------------------------------
# Number of series to fit per dataset. AutoARIMA at GiftEval scale is the
# bottleneck (~0.5-5s/series stepwise). Cap to keep wall-clock predictable;
# set to None to fit all available series.
MAX_SERIES_PER_DATASET: Optional[int] = 1000
# Maximum length of the *training* portion fed to AutoARIMA per series. The
# Hannan-Rissanen + stepwise cost grows superlinearly in series length; for
# very long series, truncating to the most recent N points is standard.
# Maximum length of the *training* portion fed to AutoARIMA per series.
# ARIMA's fit cost grows linearly in N while marginal gains in held-out
# accuracy plateau quickly past a few thousand observations. 3000 is a
# pragmatic compromise for the GiftEval scale.
MAX_TRAIN_LENGTH_PER_SERIES: Optional[int] = 3_000
N_JOBS = -1                     # statsforecast cross-series parallelism
SEED = 42

# -- Progress reporting -------------------------------------------------------
# AutoARIMA fits are done in chunks of `CHUNK_SIZE` series so we get a
# heartbeat (elapsed + ETA + running metric) every chunk rather than one
# silent multi-hour fit. With CHUNK_SIZE=50 we get a progress line roughly
# every 1-3 minutes on typical GiftEval datasets; for datasets with fewer
# series than the chunk size, within-chunk progress comes from
# `VERBOSE_STATSFORECAST` below.
CHUNK_SIZE = 50
# Joblib's parallel-verbose mode inside each chunk. Prints lines like
# "[Parallel(n_jobs=8)]: Done 5 of 21 | elapsed: 12.3min remaining: 24.5min"
# during long fits -- essential when a single chunk contains many series
# or each series is slow to fit. Set False if it clutters the log.
VERBOSE_STATSFORECAST = True
# Append per-series metrics to disk after every chunk so a crash or SIGINT
# preserves whatever has already been scored.
INCREMENTAL_PERSIST = True

CACHE_ROOT = "logs/experiments/autoarima_baseline"


# ==============================================================================
#  SEASONAL PERIOD INFERENCE
# ==============================================================================

def infer_season_length(freq_str: str) -> int:
    """Return a sensible seasonal period for ARIMA based on dataset frequency.

    The choice follows the most informative *daily* (sub-daily freqs) or
    natural calendar cycle (D->7, W->52, M->12, Q->4), which is the
    convention used in Hyndman & Athanasopoulos (FPP3) and adopted by
    statsforecast's `get_seasonality`. Yearly / unspecified -> 1.
    """
    if not freq_str:
        return 1
    s = freq_str.strip().lower().split("-")[0]
    # Sub-hourly: pandas "T"/"min" with a numeric prefix, e.g. "10T", "15min".
    m = re.match(r"^(\d+)\s*(?:t|min)$", s)
    if m:
        n_min = int(m.group(1))
        return max(1, (24 * 60) // n_min)        # one full day
    m = re.match(r"^(\d+)\s*s$", s)               # seconds
    if m:
        n_sec = int(m.group(1))
        return max(1, (24 * 3600) // n_sec)
    if s in ("t", "min"):
        return 24 * 60
    if s.startswith("h"):
        return 24
    if s.startswith("b") or s.startswith("d"):    # business / daily
        return 7
    if s.startswith("w"):
        return 52
    if s.startswith("m"):                          # M, ME, MS
        return 12
    if s.startswith("q"):
        return 4
    if s.startswith("y") or s.startswith("a"):
        return 1
    return 1


# ==============================================================================
#  GIFTEVAL -> LONG FORMAT
# ==============================================================================

def build_long_format(
    ge_dataset: GiftEvalDataset,
    horizon: int,
    max_series: Optional[int] = MAX_SERIES_PER_DATASET,
    max_train_length: Optional[int] = MAX_TRAIN_LENGTH_PER_SERIES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct train/val long-format DataFrames for statsforecast.

    Returns
    -------
    train_df : columns [unique_id, ds, y]  -- training portion of each series
    val_df   : columns [unique_id, ds, y]  -- trailing horizon held out
    series_meta : per-series book-keeping (length, freq, start, ...)

    Series too short to admit a (train, horizon) split are dropped with a
    counter logged. Excessively long training tails are truncated to
    `max_train_length` most-recent observations to bound ARIMA fit cost.
    """
    freq = ge_dataset.freq
    train_rows: List[pd.DataFrame] = []
    val_rows:   List[pd.DataFrame] = []
    meta_rows:  List[Dict[str, Any]] = []

    n_too_short = 0
    n_kept = 0

    for sidx, entry in enumerate(ge_dataset.training_dataset):
        if max_series is not None and n_kept >= max_series:
            break
        tgt = np.asarray(entry["target"], dtype=np.float64)
        if tgt.ndim > 1:
            # Univariate guarantee should already hold via to_univariate=True.
            continue
        tgt = np.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

        L = len(tgt)
        if L < horizon + 8:
            # Need at least a handful of obs for AutoARIMA to do anything
            # sensible. 8 is a conservative floor.
            n_too_short += 1
            continue

        train_y = tgt[: L - horizon]
        val_y   = tgt[L - horizon :]
        if max_train_length is not None and len(train_y) > max_train_length:
            train_y = train_y[-max_train_length:]

        # Reconstruct timestamps from entry["start"] (gluonts Period).
        start = entry["start"]
        try:
            start_ts = pd.Timestamp(start.to_timestamp()
                                    if hasattr(start, "to_timestamp")
                                    else start)
        except Exception:
            start_ts = pd.Timestamp("1970-01-01")

        # Train ds spans the offset to the end of train_y; val ds continues.
        train_offset = (L - horizon) - len(train_y)
        train_ds = pd.date_range(
            start=start_ts + pd.tseries.frequencies.to_offset(freq) * train_offset,
            periods=len(train_y), freq=freq,
        )
        val_ds = pd.date_range(
            start=train_ds[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=horizon, freq=freq,
        )

        uid = f"S{sidx:06d}"
        train_rows.append(pd.DataFrame({
            "unique_id": uid, "ds": train_ds, "y": train_y.astype(np.float32),
        }))
        val_rows.append(pd.DataFrame({
            "unique_id": uid, "ds": val_ds, "y": val_y.astype(np.float32),
        }))
        meta_rows.append({
            "unique_id": uid, "series_idx": sidx,
            "length_total": int(L), "length_train": int(len(train_y)),
            "horizon": int(horizon),
        })
        n_kept += 1

    if not train_rows:
        raise RuntimeError(
            f"No usable series for horizon={horizon}: too_short={n_too_short}"
        )

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    meta_df  = pd.DataFrame(meta_rows)

    print(Fore.CYAN
          + f"  Built long-format: {n_kept} series kept "
          + f"(too_short={n_too_short}), train_rows={len(train_df)}, "
          + f"val_rows={len(val_df)}" + Fore.RESET)
    return train_df, val_df, meta_df


# ==============================================================================
#  CLOSED-FORM GAUSSIAN METRICS
# ==============================================================================

# z-score for the 80% PI:  (hi80 - mu) / sigma = Phi^{-1}(0.9) ~ 1.2816
_Z_80 = float(norm.ppf(0.90))


def recover_sigma(mu: np.ndarray, hi80: np.ndarray) -> np.ndarray:
    """Invert hi80 = mu + z * sigma to recover sigma. Floor at a small eps to
    avoid division by zero in degenerate constant-forecast cases."""
    sigma = (hi80 - mu) / _Z_80
    return np.maximum(sigma, 1e-8)


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Negative log-likelihood per element under a Gaussian predictive law."""
    return 0.5 * np.log(2.0 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2.0 * sigma ** 2)


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive law.

    Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Eq. (5):
        CRPS(N(mu, sigma^2), y) = sigma * [ z (2 Phi(z) - 1) + 2 phi(z)
                                            - 1 / sqrt(pi) ]
    with z = (y - mu) / sigma.
    """
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0)
                    + 2.0 * norm.pdf(z)
                    - 1.0 / math.sqrt(math.pi))


def gaussian_pinball_mean(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    quantiles: Tuple[float, ...] = QUANTILE_GRID,
) -> np.ndarray:
    """Mean pinball loss across `quantiles`, evaluated against the Gaussian
    quantile forecasts. Returns the per-element mean over the grid."""
    q_arr = np.asarray(quantiles, dtype=np.float64)
    z_q   = norm.ppf(q_arr)                                          # (Q,)
    # Broadcast to (Q, N): quantile_pred[q, n] = mu[n] + z_q[q] * sigma[n]
    quantile_pred = mu[None, :] + z_q[:, None] * sigma[None, :]      # (Q, N)
    err = y[None, :] - quantile_pred                                 # (Q, N)
    pinball = np.maximum(q_arr[:, None] * err, (q_arr[:, None] - 1.0) * err)
    return pinball.mean(axis=0)                                      # (N,)


def gaussian_coverage_80(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Indicator that y lies inside the 80% Gaussian PI [mu - z80*sigma,
    mu + z80*sigma]. Returned as float per element (0 / 1)."""
    lo = mu - _Z_80 * sigma
    hi = mu + _Z_80 * sigma
    return ((y >= lo) & (y <= hi)).astype(np.float64)


# ==============================================================================
#  PROGRESS UTILITIES
# ==============================================================================

def format_duration(seconds: float) -> str:
    """Human-readable duration: '3m 12s', '1h 04m', '47s'."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# ==============================================================================
#  FIT + SCORE  (one dataset, term)
# ==============================================================================

def fit_and_score(
    train_df: pd.DataFrame, val_df: pd.DataFrame, meta_df: pd.DataFrame,
    horizon: int, freq: str, season_length: int,
    dataset_display: str = "", term: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Chunked AutoARIMA fit + per-series probabilistic eval.

    Series are processed in chunks of `CHUNK_SIZE` so we get a heartbeat
    (elapsed + ETA + running CRPS) on every chunk. Each chunk is a fresh
    `StatsForecast` instance with its own joblib pool -- this loses a small
    amount of pool-warmup time per chunk but gains predictable memory,
    incremental persistence, and survivability against `KeyboardInterrupt`.

    Returns
    -------
    per_series_metrics : DataFrame with columns
        [unique_id, val_mae, val_nll, val_crps, val_pinball, val_coverage_80,
         fit_status]
    orders_df : DataFrame with columns [unique_id, p, d, q, P, D, Q, m, ...]
    aggregate : dict of cross-series mean / std / median / IQR per metric
    """
    all_uids   = train_df["unique_id"].unique()
    n_series   = len(all_uids)
    n_chunks   = max(1, math.ceil(n_series / CHUNK_SIZE))
    chunks     = [all_uids[i : i + CHUNK_SIZE]
                  for i in range(0, n_series, CHUNK_SIZE)]

    print(Fore.CYAN
          + f"  Fitting AutoARIMA: season_length={season_length}, "
          + f"freq='{freq}', n_series={n_series}, "
          + f"n_chunks={n_chunks} (chunk_size={CHUNK_SIZE})" + Fore.RESET)

    # Set up incremental persistence path (per-series CSV appended chunk by
    # chunk). The final consolidated CSV is written at the end of the
    # pipeline via `_persist_dataset_artifacts`; this one is a safety net.
    progress_csv: Optional[str] = None
    if INCREMENTAL_PERSIST and dataset_display and term:
        out_dir = _ds_dir(dataset_display, term)
        os.makedirs(out_dir, exist_ok=True)
        progress_csv = os.path.join(out_dir, "metrics_per_series.partial.csv")
        # Truncate any partial from a previous run.
        if os.path.exists(progress_csv):
            os.remove(progress_csv)

    per_series_chunks: List[pd.DataFrame] = []
    orders_chunks:     List[pd.DataFrame] = []
    chunk_times:       List[float] = []
    total_fit:         float = 0.0
    total_predict:     float = 0.0
    interrupted:       bool = False

    fallback = SeasonalNaive(season_length=max(1, season_length))

    for chunk_idx, chunk_uids in enumerate(chunks):
        chunk_t0 = time.perf_counter()

        train_chunk = train_df[train_df["unique_id"].isin(chunk_uids)]
        val_chunk   = val_df  [val_df  ["unique_id"].isin(chunk_uids)]

        model = AutoARIMA(
            season_length=season_length, ic=INFORMATION_CRITERION,
            **ARIMA_KWARGS,
        )
        sf = StatsForecast(
            models=[model], freq=freq, n_jobs=N_JOBS,
            fallback_model=fallback, verbose=VERBOSE_STATSFORECAST,
        )

        try:
            t_fit = time.perf_counter()
            sf.fit(train_chunk)
            fit_elapsed = time.perf_counter() - t_fit
            total_fit += fit_elapsed

            t_pred = time.perf_counter()
            pred = sf.predict(h=horizon, level=[80])
            pred_elapsed = time.perf_counter() - t_pred
            total_predict += pred_elapsed

        except KeyboardInterrupt:
            print(Fore.RED + f"\n  Interrupted during chunk "
                  + f"{chunk_idx + 1}/{n_chunks}. "
                  + "Persisting completed chunks and aborting dataset."
                  + Fore.RESET)
            interrupted = True
            break

        # ---- align predictions with held-out targets within this chunk -----
        merged = val_chunk.merge(pred, on=["unique_id", "ds"], how="inner")
        if len(merged) != len(val_chunk):
            merged = _positional_merge(val_chunk, pred)

        mu     = merged["AutoARIMA"].to_numpy(dtype=np.float64)
        hi80   = merged["AutoARIMA-hi-80"].to_numpy(dtype=np.float64)
        y_true = merged["y"].to_numpy(dtype=np.float64)
        sigma  = recover_sigma(mu, hi80)

        chunk_metrics = merged[["unique_id"]].copy()
        chunk_metrics["mae"]      = np.abs(y_true - mu)
        chunk_metrics["nll"]      = gaussian_nll(y_true, mu, sigma)
        chunk_metrics["crps"]     = gaussian_crps(y_true, mu, sigma)
        chunk_metrics["pinball"]  = gaussian_pinball_mean(
            y_true, mu, sigma, QUANTILE_GRID)
        chunk_metrics["coverage"] = gaussian_coverage_80(y_true, mu, sigma)

        per_series_chunk = chunk_metrics.groupby(
            "unique_id", as_index=False,
        ).agg(
            val_mae=("mae", "mean"),
            val_nll=("nll", "mean"),
            val_crps=("crps", "mean"),
            val_pinball=("pinball", "mean"),
            val_coverage_80=("coverage", "mean"),
        )
        per_series_chunk["fit_status"] = "ok"
        per_series_chunks.append(per_series_chunk)
        orders_chunks.append(_extract_orders(sf, season_length))

        # ---- incremental persistence --------------------------------------
        if progress_csv is not None:
            header = not os.path.exists(progress_csv)
            per_series_chunk.to_csv(progress_csv, mode="a",
                                    header=header, index=False)

        # ---- progress log -------------------------------------------------
        chunk_elapsed = time.perf_counter() - chunk_t0
        chunk_times.append(chunk_elapsed)
        avg_chunk = sum(chunk_times) / len(chunk_times)
        remaining = avg_chunk * (n_chunks - chunk_idx - 1)

        # Running aggregate metrics over all completed chunks so we can see
        # whether the numbers look sane long before the dataset finishes.
        so_far = pd.concat(per_series_chunks, ignore_index=True)
        running_crps = so_far["val_crps"].replace(
            [np.inf, -np.inf], np.nan).dropna().mean()
        running_cov  = so_far["val_coverage_80"].replace(
            [np.inf, -np.inf], np.nan).dropna().mean()

        print(Fore.YELLOW
              + f"  [{chunk_idx + 1:3d}/{n_chunks}] "
              + f"chunk={len(chunk_uids):4d}  "
              + f"fit={format_duration(fit_elapsed)}  "
              + f"pred={format_duration(pred_elapsed)}  "
              + f"|  running CRPS={running_crps:.4f}  "
              + f"cov80={running_cov:.3f}  "
              + f"|  elapsed={format_duration(sum(chunk_times))}  "
              + f"ETA={format_duration(remaining)}"
              + Fore.RESET)

    # If interrupted before any chunk completed, surface a clean error.
    if not per_series_chunks:
        raise RuntimeError(
            "fit_and_score: no chunks completed (interrupted before "
            "first chunk finished)."
        )

    per_series = pd.concat(per_series_chunks, ignore_index=True)
    orders_df  = pd.concat(orders_chunks, ignore_index=True)

    # Cross-series aggregates (mean / std / median / IQR) per metric.
    aggregate: Dict[str, Any] = {
        "n_series":              int(per_series["unique_id"].nunique()),
        "n_series_requested":    int(n_series),
        "n_chunks":              int(n_chunks),
        "chunk_size":            int(CHUNK_SIZE),
        "fit_seconds":           round(total_fit, 2),
        "predict_seconds":       round(total_predict, 2),
        "total_seconds":         round(sum(chunk_times), 2),
        "season_length":         int(season_length),
        "freq":                  freq,
        "horizon":               int(horizon),
        "information_criterion": INFORMATION_CRITERION,
        "interrupted":           bool(interrupted),
    }
    for col in ("val_mae", "val_nll", "val_crps",
                "val_pinball", "val_coverage_80"):
        vec = per_series[col].to_numpy()
        finite = vec[np.isfinite(vec)]
        if finite.size == 0:
            aggregate[col] = {"mean": float("nan"), "std": float("nan"),
                              "median": float("nan"), "iqr": float("nan"),
                              "n_finite": 0}
            continue
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        aggregate[col] = {
            "mean":     float(finite.mean()),
            "std":      float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "median":   float(np.median(finite)),
            "iqr":      float(q75 - q25),
            "n_finite": int(finite.size),
        }

    # Clean up the .partial.csv once the consolidated DataFrame is in hand;
    # the final write happens via `_persist_dataset_artifacts`.
    if progress_csv is not None and os.path.exists(progress_csv) and not interrupted:
        try:
            os.remove(progress_csv)
        except OSError:
            pass

    return per_series, orders_df, aggregate


def _positional_merge(val_df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Last-resort alignment when ds joins fail. Pairs val/pred row-by-row
    within each unique_id (both already sorted by ds)."""
    parts = []
    for uid, vg in val_df.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        pg = pred[pred["unique_id"] == uid].sort_values("ds")
        n = min(len(vg), len(pg))
        if n == 0:
            continue
        merged_g = vg.iloc[:n].reset_index(drop=True).copy()
        for col in ("AutoARIMA", "AutoARIMA-lo-80", "AutoARIMA-hi-80"):
            merged_g[col] = pg[col].iloc[:n].to_numpy()
        parts.append(merged_g)
    return pd.concat(parts, ignore_index=True)


def _extract_orders(sf: StatsForecast, season_length: int) -> pd.DataFrame:
    """Pull the selected (p,d,q)(P,D,Q)_m from each fitted AutoARIMA. The
    statsforecast internals expose this on `sf.fitted_` -> a (n_series, 1)
    object array, each element a fitted AutoARIMA instance."""
    rows: List[Dict[str, Any]] = []
    try:
        fitted = sf.fitted_              # shape (n_series, n_models)
        uids   = sf.uids
    except AttributeError:
        return pd.DataFrame(columns=["unique_id", "p", "d", "q",
                                     "P", "D", "Q", "m"])
    for i, uid in enumerate(uids):
        try:
            mdl = fitted[i, 0]
            # statsforecast AutoARIMA stores the chosen orders in
            # `mdl.model_["arma"]` -> (p, q, P, Q, m, d, D).
            arma = mdl.model_["arma"]
            p, q, P, Q, m, d, D = (int(arma[0]), int(arma[1]),
                                   int(arma[2]), int(arma[3]),
                                   int(arma[4]), int(arma[5]),
                                   int(arma[6]))
            rows.append({"unique_id": uid, "p": p, "d": d, "q": q,
                         "P": P, "D": D, "Q": Q, "m": m})
        except Exception:
            rows.append({"unique_id": uid,
                         "p": np.nan, "d": np.nan, "q": np.nan,
                         "P": np.nan, "D": np.nan, "Q": np.nan,
                         "m": season_length})
    return pd.DataFrame(rows)


# ==============================================================================
#  PLOTTING
# ==============================================================================

def plot_metrics_summary(
    per_series: pd.DataFrame, save_path: str,
    dataset_display: str, term: str, aggregate: Dict[str, Any],
):
    """Diagnostic two-panel plot: CRPS distribution across series, and a
    coverage-vs-nominal histogram (the nominal level is 0.80)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    crps = per_series["val_crps"].to_numpy()
    crps = crps[np.isfinite(crps)]
    ax1.hist(crps, bins=40, color="#1f77b4",
             edgecolor="white", alpha=0.85)
    ax1.axvline(np.median(crps), color="#d62728",
                linestyle="--", linewidth=1.8,
                label=f"median = {np.median(crps):.3f}")
    ax1.set_xlabel("per-series CRPS", fontsize=11)
    ax1.set_ylabel("number of series", fontsize=11)
    ax1.set_title(f"CRPS distribution -- {dataset_display} ({term})",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)

    cov = per_series["val_coverage_80"].to_numpy()
    cov = cov[np.isfinite(cov)]
    ax2.hist(cov, bins=40, range=(0.0, 1.0), color="#2ca02c",
             edgecolor="white", alpha=0.85)
    ax2.axvline(0.80, color="#d62728", linestyle="--", linewidth=1.8,
                label="nominal = 0.80")
    ax2.axvline(np.mean(cov), color="#1f77b4", linestyle="-.", linewidth=1.6,
                label=f"empirical mean = {np.mean(cov):.3f}")
    ax2.set_xlabel("per-series empirical coverage", fontsize=11)
    ax2.set_ylabel("number of series", fontsize=11)
    ax2.set_title(f"80% PI coverage -- {dataset_display} ({term})",
                  fontsize=12, fontweight="bold")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


# ==============================================================================
#  CACHE HELPERS
# ==============================================================================

def _ds_dir(dataset_display: str, term: str) -> str:
    return os.path.join(CACHE_ROOT, dataset_display, term)


def _is_cached(dataset_display: str, term: str) -> bool:
    """An entry is considered cached if `aggregate_metrics.json` is present
    and parseable -- that is, a previous run completed successfully."""
    agg = os.path.join(_ds_dir(dataset_display, term), "aggregate_metrics.json")
    if not os.path.isfile(agg):
        return False
    try:
        with open(agg) as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _persist_dataset_artifacts(
    dataset_display: str, term: str,
    per_series: pd.DataFrame, orders_df: pd.DataFrame,
    aggregate: Dict[str, Any], meta_df: pd.DataFrame,
):
    out_dir = _ds_dir(dataset_display, term)
    os.makedirs(out_dir, exist_ok=True)

    per_series_full = per_series.merge(meta_df, on="unique_id", how="left")
    per_series_full.to_csv(os.path.join(out_dir, "metrics_per_series.csv"),
                           index=False)
    orders_df.to_csv(os.path.join(out_dir, "orders.csv"), index=False)
    with open(os.path.join(out_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    plot_metrics_summary(
        per_series_full,
        os.path.join(out_dir, "metrics_summary.png"),
        dataset_display, term, aggregate,
    )
    print(Fore.GREEN + f"  Artifacts -> {out_dir}" + Fore.RESET)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    np.random.seed(SEED)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CACHE_ROOT, f"runs/{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(Fore.CYAN + "AutoARIMA baseline evaluation pipeline" + Fore.RESET)
    print(Fore.CYAN + f"  information_criterion = {INFORMATION_CRITERION}"
          + Fore.RESET)
    print(Fore.CYAN + f"  ARIMA_KWARGS          = {ARIMA_KWARGS}" + Fore.RESET)
    print(Fore.CYAN + f"  max_series/dataset    = {MAX_SERIES_PER_DATASET}"
          + Fore.RESET)
    print(Fore.CYAN + f"  max_train_length/sr   = {MAX_TRAIN_LENGTH_PER_SERIES}"
          + Fore.RESET)
    print(Fore.CYAN + f"  n_jobs                = {N_JOBS}" + Fore.RESET)

    global_rows: List[Dict[str, Any]] = []
    n_datasets = len(DATASETS)
    run_t0 = time.perf_counter()
    dataset_times: List[float] = []

    for ds_idx, (ge_name, term, dataset_display, to_univariate) in enumerate(DATASETS):
        ds_t0 = time.perf_counter()
        elapsed_total = time.perf_counter() - run_t0
        eta_run = (sum(dataset_times) / max(1, len(dataset_times))
                   * (n_datasets - ds_idx)) if dataset_times else float("inf")

        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  [{ds_idx + 1:3d}/{n_datasets}]  "
              + f"DATASET: {ge_name}  term={term}  "
              + f"({dataset_display})" + Fore.RESET)
        print(Fore.CYAN + f"  run elapsed={format_duration(elapsed_total)}  "
              + f"run ETA={format_duration(eta_run)}" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        if _is_cached(dataset_display, term):
            print(Fore.WHITE + "  Cache hit -- skipping." + Fore.RESET)
            with open(os.path.join(_ds_dir(dataset_display, term),
                                   "aggregate_metrics.json")) as f:
                aggregate = json.load(f)
            global_rows.append({"dataset_display": dataset_display,
                                "term": term, **_flatten_aggregate(aggregate)})
            continue

        try:
            ge_dataset = GiftEvalDataset(name=ge_name, term=term,
                                         to_univariate=to_univariate)
            horizon = ge_dataset.prediction_length
            freq    = ge_dataset.freq
            season_raw = infer_season_length(freq)
            if (SEASONAL_PERIOD_CAP is not None
                    and season_raw > SEASONAL_PERIOD_CAP):
                season = 1
                print(Fore.YELLOW
                      + f"  season_length raw={season_raw} > cap="
                      + f"{SEASONAL_PERIOD_CAP}; disabling seasonal ARIMA "
                      + "(m=1). Report SeasonalNaive separately for the "
                      + "seasonal baseline." + Fore.RESET)
            else:
                season = season_raw
            print(Fore.CYAN + f"  freq='{freq}'  horizon={horizon}  "
                  + f"target_dim={ge_dataset.target_dim}  "
                  + f"season_length={season} (raw={season_raw})" + Fore.RESET)

            train_df, val_df, meta_df = build_long_format(ge_dataset, horizon)
            per_series, orders_df, aggregate = fit_and_score(
                train_df, val_df, meta_df, horizon, freq, season,
                dataset_display=dataset_display, term=term,
            )
            _persist_dataset_artifacts(
                dataset_display, term, per_series, orders_df, aggregate, meta_df,
            )
            global_rows.append({"dataset_display": dataset_display,
                                "term": term, **_flatten_aggregate(aggregate)})

        except KeyboardInterrupt:
            print(Fore.RED + "\n  KeyboardInterrupt received. "
                  + "Writing global summary of completed datasets and exiting."
                  + Fore.RESET)
            break

        except Exception as exc:
            print(Fore.RED + f"  FAILED: {type(exc).__name__}: {exc}"
                  + Fore.RESET)
            # Persist a stub so cache logic knows we tried.
            out_dir = _ds_dir(dataset_display, term)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "failure.json"), "w") as f:
                json.dump({"error": f"{type(exc).__name__}: {exc}"}, f)

        ds_elapsed = time.perf_counter() - ds_t0
        dataset_times.append(ds_elapsed)
        print(Fore.GREEN + f"  Dataset done in {format_duration(ds_elapsed)}  "
              + f"|  cumulative run "
              + f"{format_duration(time.perf_counter() - run_t0)}"
              + Fore.RESET)

    # Global summary across all datasets.
    if global_rows:
        global_df = pd.DataFrame(global_rows)
        global_csv = os.path.join(run_dir, "baseline_all.csv")
        global_df.to_csv(global_csv, index=False)
        print(Fore.GREEN + f"\nGlobal summary -> {global_csv}" + Fore.RESET)
    print(Fore.GREEN + "AutoARIMA baseline pipeline done." + Fore.RESET)


def _flatten_aggregate(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested aggregate dict so each metric's mean / std are
    top-level columns in the global summary CSV."""
    out: Dict[str, Any] = {
        "n_series":       aggregate.get("n_series"),
        "horizon":        aggregate.get("horizon"),
        "freq":           aggregate.get("freq"),
        "season_length":  aggregate.get("season_length"),
        "fit_seconds":    aggregate.get("fit_seconds"),
        "predict_seconds": aggregate.get("predict_seconds"),
    }
    for col in ("val_mae", "val_nll", "val_crps",
                "val_pinball", "val_coverage_80"):
        block = aggregate.get(col, {})
        out[f"{col}_mean"]   = block.get("mean")
        out[f"{col}_std"]    = block.get("std")
        out[f"{col}_median"] = block.get("median")
    return out


if __name__ == "__main__":
    main()