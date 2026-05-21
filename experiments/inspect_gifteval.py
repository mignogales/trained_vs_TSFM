"""
Inspect GiftEval datasets before training PatchTST.

This script mirrors the DATASETS list from the training pipeline and generates
lightweight plots/summaries for each (dataset, term):

  outputs/gifteval_dataset_inspection/
      index.csv
      series_summary.csv
      <DatasetDisplay>/<term>/
          sample_series.png
          length_histogram.png
          missingness_histogram.png
          value_distribution.png
          summary.json

Examples
--------
Inspect all configured datasets:
    python inspect_gifteval_datasets.py

Inspect only a few datasets while developing:
    python inspect_gifteval_datasets.py --limit-datasets 5 --max-series 200

Filter by display/name substring:
    python inspect_gifteval_datasets.py --filter jena
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
load_dotenv()

# import numpy as np

np = None
pd = None
plt = None


def _ensure_runtime_deps(needs_plot: bool = False):
    """Import heavier plotting/data deps lazily so --help stays lightweight."""
    global np, pd, plt
    if np is None:
        import numpy as _np

        np = _np
    if pd is None:
        import pandas as _pd

        pd = _pd
    if needs_plot and plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt


# Keep this in sync with the training script.
DATASETS = [
    ("jena_weather/10T", "short", "JenaWeather-10T", True),
    ("jena_weather/10T", "medium", "JenaWeather-10T", True),
    ("jena_weather/10T", "long", "JenaWeather-10T", True),
    ("jena_weather/H", "short", "JenaWeather-H", True),
    ("jena_weather/H", "medium", "JenaWeather-H", True),
    ("jena_weather/H", "long", "JenaWeather-H", True),
    ("jena_weather/D", "short", "JenaWeather-D", True),
    ("bizitobs_application", "short", "BizITObsApp", True),
    ("bizitobs_application", "medium", "BizITObsApp", True),
    ("bizitobs_application", "long", "BizITObsApp", True),
    ("bizitobs_service", "short", "BizITObsService", True),
    ("bizitobs_service", "medium", "BizITObsService", True),
    ("bizitobs_service", "long", "BizITObsService", True),
    ("bizitobs_l2c/5T", "short", "BizITObsL2C-5T", True),
    ("bizitobs_l2c/5T", "medium", "BizITObsL2C-5T", True),
    ("bizitobs_l2c/5T", "long", "BizITObsL2C-5T", True),
    ("bizitobs_l2c/H", "short", "BizITObsL2C-H", True),
    ("bizitobs_l2c/H", "medium", "BizITObsL2C-H", True),
    ("bizitobs_l2c/H", "long", "BizITObsL2C-H", True),
    ("bitbrains_fast_storage/5T", "short", "BitbrainsFS-5T", True),
    ("bitbrains_fast_storage/5T", "medium", "BitbrainsFS-5T", True),
    ("bitbrains_fast_storage/5T", "long", "BitbrainsFS-5T", True),
    ("bitbrains_fast_storage/H", "short", "BitbrainsFS-H", True),
    ("bitbrains_rnd/5T", "short", "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/5T", "medium", "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/5T", "long", "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/H", "short", "BitbrainsRnD-H", True),
    ("restaurant", "short", "Restaurant", False),
    ("ett1/15T", "short", "ETTm1-15T", True),
    ("ett1/15T", "medium", "ETTm1-15T", True),
    ("ett1/15T", "long", "ETTm1-15T", True),
    ("ett1/H", "short", "ETTm1-H", True),
    ("ett1/H", "medium", "ETTm1-H", True),
    ("ett1/H", "long", "ETTm1-H", True),
    ("ett1/D", "short", "ETTm1-D", True),
    ("ett1/W", "short", "ETTm1-W", True),
    ("ett2/15T", "short", "ETTm2-15T", True),
    ("ett2/15T", "medium", "ETTm2-15T", True),
    ("ett2/15T", "long", "ETTm2-15T", True),
    ("ett2/H", "short", "ETTm2-H", True),
    ("ett2/H", "medium", "ETTm2-H", True),
    ("ett2/H", "long", "ETTm2-H", True),
    ("ett2/D", "short", "ETTm2-D", True),
    ("ett2/W", "short", "ETTm2-W", True),
    ("LOOP_SEATTLE/5T", "short", "LoopSeattle-5T", False),
    ("LOOP_SEATTLE/5T", "medium", "LoopSeattle-5T", False),
    ("LOOP_SEATTLE/5T", "long", "LoopSeattle-5T", False),
    ("LOOP_SEATTLE/H", "short", "LoopSeattle-H", False),
    ("LOOP_SEATTLE/H", "medium", "LoopSeattle-H", False),
    ("LOOP_SEATTLE/H", "long", "LoopSeattle-H", False),
    ("LOOP_SEATTLE/D", "short", "LoopSeattle-D", False),
    ("SZ_TAXI/15T", "short", "SZTaxi-15T", False),
    ("SZ_TAXI/15T", "medium", "SZTaxi-15T", False),
    ("SZ_TAXI/15T", "long", "SZTaxi-15T", False),
    ("SZ_TAXI/H", "short", "SZTaxi-H", False),
    ("M_DENSE/H", "short", "MDense-H", False),
    ("M_DENSE/H", "medium", "MDense-H", False),
    ("M_DENSE/H", "long", "MDense-H", False),
    ("M_DENSE/D", "short", "MDense-D", False),
    ("solar/10T", "short", "Solar-10T", False),
    ("solar/10T", "medium", "Solar-10T", False),
    ("solar/10T", "long", "Solar-10T", False),
    ("solar/H", "short", "Solar-H", False),
    ("solar/H", "medium", "Solar-H", False),
    ("solar/H", "long", "Solar-H", False),
    ("solar/D", "short", "Solar-D", False),
    ("solar/W", "short", "Solar-W", False),
    ("hierarchical_sales/D", "short", "HierSales-D", False),
    ("hierarchical_sales/W", "short", "HierSales-W", False),
    ("m4_yearly", "short", "M4-Yearly", False),
    ("m4_quarterly", "short", "M4-Quarterly", False),
    ("m4_monthly", "short", "M4-Monthly", False),
    ("m4_weekly", "short", "M4-Weekly", False),
    ("m4_daily", "short", "M4-Daily", False),
    ("m4_hourly", "short", "M4-Hourly", False),
    ("hospital", "short", "Hospital", False),
    ("covid_deaths", "short", "CovidDeaths-D", False),
    ("us_births/D", "short", "USBirths-D", False),
    ("us_births/W", "short", "USBirths-W", False),
    ("us_births/M", "short", "USBirths-M", False),
    ("saugeenday/D", "short", "SaugeenDay-D", False),
    ("saugeenday/W", "short", "SaugeenDay-W", False),
    ("saugeenday/M", "short", "SaugeenDay-M", False),
    ("temperature_rain_with_missing", "short", "TempRainMissing-D", True),
    ("kdd_cup_2018_with_missing/H", "short", "KDDCup18Missing-H", True),
    ("kdd_cup_2018_with_missing/H", "medium", "KDDCup18Missing-H", True),
    ("kdd_cup_2018_with_missing/H", "long", "KDDCup18Missing-H", True),
    ("kdd_cup_2018_with_missing/D", "short", "KDDCup18Missing-D", True),
    ("car_parts_with_missing", "short", "CarPartsMissing", False),
    ("electricity/15T", "short", "Electricity-15T", False),
    ("electricity/15T", "medium", "Electricity-15T", False),
    ("electricity/15T", "long", "Electricity-15T", False),
    ("electricity/H", "short", "Electricity-H", False),
    ("electricity/H", "medium", "Electricity-H", False),
    ("electricity/H", "long", "Electricity-H", False),
    ("electricity/D", "short", "Electricity-D", False),
    ("electricity/W", "short", "Electricity-W", False),
]

CONTEXT_LENGTH = 512
DEFAULT_OUT_DIR = "outputs/gifteval_dataset_inspection"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    term: str
    display: str
    to_univariate: bool

    @property
    def label(self) -> str:
        return f"{self.display}/{self.term}"


def _safe_path_part(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def _iter_limited(items: Iterable, limit: Optional[int]) -> Iterable:
    for idx, item in enumerate(items):
        if limit is not None and idx >= limit:
            break
        yield item


def _target_to_matrix(target: np.ndarray, target_dim: Optional[int]) -> np.ndarray:
    """Return target as (num_series_or_channels, time)."""
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        flat = arr.reshape(-1, arr.shape[-1])
        return flat.astype(np.float32, copy=False)

    if target_dim is not None:
        if arr.shape[0] == target_dim:
            return arr
        if arr.shape[1] == target_dim:
            return arr.T

    # Most GluonTS-style multivariate targets are (dim, time). If uncertain,
    # treat the longer axis as time.
    return arr if arr.shape[0] <= arr.shape[1] else arr.T


def _finite_values(matrix: np.ndarray) -> np.ndarray:
    values = matrix[np.isfinite(matrix)]
    return values.astype(np.float32, copy=False)


def _sample_entries(entries: Sequence, n: int, seed: int) -> List:
    if len(entries) <= n:
        return list(entries)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(entries), size=n, replace=False))
    return [entries[int(i)] for i in idx]


def _plot_sample_series(
    samples: Sequence[np.ndarray],
    save_path: Path,
    title: str,
    horizon: int,
    max_points: int,
):
    n = len(samples)
    if n == 0:
        return

    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(3.2, 2.7 * nrows)))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, series in zip(axes_arr, samples):
        y = np.asarray(series, dtype=np.float32)
        if len(y) > max_points:
            start = len(y) - max_points
            y = y[start:]
            x = np.arange(start, start + len(y))
        else:
            x = np.arange(len(y))

        ax.plot(x, y, color="#2563eb", linewidth=1.15)
        if len(x) > horizon:
            ax.axvspan(x[-horizon], x[-1], color="#f59e0b", alpha=0.14, label="last horizon")
        ax.set_title(f"length={len(series):,}", fontsize=10)
        ax.grid(True, alpha=0.25)

    for ax in axes_arr[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hist(values: Sequence[float], save_path: Path, title: str, xlabel: str, bins: int = 40):
    if not values:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(values, bins=min(bins, max(8, len(values))), color="#0f766e", alpha=0.82)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_value_distribution(values: np.ndarray, save_path: Path, title: str, max_values: int, seed: int):
    if values.size == 0:
        return
    if values.size > max_values:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(values.size, size=max_values, replace=False)]

    lo, hi = np.nanpercentile(values, [0.5, 99.5])
    clipped = values[(values >= lo) & (values <= hi)]
    if clipped.size == 0:
        clipped = values

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(clipped, bins=80, color="#7c3aed", alpha=0.80)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Value (clipped to 0.5-99.5 percentiles)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def inspect_dataset(
    spec: DatasetSpec,
    out_root: Path,
    max_series: Optional[int],
    sample_series: int,
    max_plot_points: int,
    max_dist_values: int,
    seed: int,
) -> Tuple[dict, List[dict]]:
    _ensure_runtime_deps(needs_plot=True)

    from gift_eval.data import Dataset as GiftEvalDataset

    ge_dataset = GiftEvalDataset(
        name=spec.name,
        term=spec.term,
        to_univariate=spec.to_univariate,
    )
    entries = list(_iter_limited(ge_dataset.training_dataset, max_series))

    ds_out = out_root / _safe_path_part(spec.display) / _safe_path_part(spec.term)
    ds_out.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_pool = []
    value_chunks = []

    for entry_idx, entry in enumerate(entries):
        matrix = _target_to_matrix(entry["target"], getattr(ge_dataset, "target_dim", None))
        for channel_idx, series in enumerate(matrix):
            finite = np.isfinite(series)
            missing = ~finite
            values = series[finite]
            if len(sample_pool) < sample_series:
                sample_pool.append(np.nan_to_num(series, nan=np.nanmedian(values) if values.size else 0.0))
            if values.size:
                value_chunks.append(values)

            rows.append(
                {
                    "dataset_name": spec.name,
                    "term": spec.term,
                    "display": spec.display,
                    "entry_idx": entry_idx,
                    "channel_idx": channel_idx,
                    "length": int(series.shape[0]),
                    "n_missing": int(missing.sum()),
                    "missing_frac": float(missing.mean()) if series.size else float("nan"),
                    "n_finite": int(finite.sum()),
                    "min": float(np.nanmin(values)) if values.size else float("nan"),
                    "p05": float(np.nanpercentile(values, 5)) if values.size else float("nan"),
                    "median": float(np.nanmedian(values)) if values.size else float("nan"),
                    "mean": float(np.nanmean(values)) if values.size else float("nan"),
                    "p95": float(np.nanpercentile(values, 95)) if values.size else float("nan"),
                    "max": float(np.nanmax(values)) if values.size else float("nan"),
                    "std": float(np.nanstd(values)) if values.size else float("nan"),
                    "n_train_windows_possible": max(
                        0, int(series.shape[0] * (80.0 / 90.0)) - CONTEXT_LENGTH - ge_dataset.prediction_length + 1
                    ),
                    "n_val_windows_possible": max(
                        0, series.shape[0] - int(series.shape[0] * (80.0 / 90.0)) - CONTEXT_LENGTH - ge_dataset.prediction_length + 1
                    ),
                }
            )

    df = pd.DataFrame(rows)
    lengths = df["length"].dropna().astype(int).tolist() if not df.empty else []
    missing_fracs = df["missing_frac"].dropna().tolist() if not df.empty else []
    all_values = np.concatenate(value_chunks) if value_chunks else np.array([], dtype=np.float32)

    title_prefix = f"{spec.display} ({spec.name}, term={spec.term})"
    _plot_sample_series(
        sample_pool,
        ds_out / "sample_series.png",
        f"Sample series - {title_prefix}",
        horizon=ge_dataset.prediction_length,
        max_points=max_plot_points,
    )
    _plot_hist(
        lengths,
        ds_out / "length_histogram.png",
        f"Series lengths - {title_prefix}",
        "Length",
    )
    _plot_hist(
        missing_fracs,
        ds_out / "missingness_histogram.png",
        f"Missing fraction - {title_prefix}",
        "Missing fraction",
    )
    _plot_value_distribution(
        _finite_values(all_values),
        ds_out / "value_distribution.png",
        f"Value distribution - {title_prefix}",
        max_values=max_dist_values,
        seed=seed,
    )

    summary = {
        "dataset_name": spec.name,
        "term": spec.term,
        "display": spec.display,
        "to_univariate": spec.to_univariate,
        "freq": getattr(ge_dataset, "freq", None),
        "prediction_length": int(ge_dataset.prediction_length),
        "target_dim": int(getattr(ge_dataset, "target_dim", 1)),
        "entries_loaded": len(entries),
        "series_or_channels_loaded": int(len(df)),
        "length_min": int(df["length"].min()) if not df.empty else 0,
        "length_median": float(df["length"].median()) if not df.empty else float("nan"),
        "length_max": int(df["length"].max()) if not df.empty else 0,
        "missing_frac_mean": float(df["missing_frac"].mean()) if not df.empty else float("nan"),
        "finite_value_min": float(np.nanmin(all_values)) if all_values.size else float("nan"),
        "finite_value_median": float(np.nanmedian(all_values)) if all_values.size else float("nan"),
        "finite_value_max": float(np.nanmax(all_values)) if all_values.size else float("nan"),
        "series_with_train_windows": int((df["n_train_windows_possible"] > 0).sum()) if not df.empty else 0,
        "series_with_val_windows": int((df["n_val_windows_possible"] > 0).sum()) if not df.empty else 0,
    }

    with open(ds_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and summarize GiftEval datasets.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for plots and CSV summaries.")
    parser.add_argument("--filter", default=None, help="Case-insensitive substring filter over name/display/term.")
    parser.add_argument("--limit-datasets", type=int, default=None, help="Inspect only the first N matching dataset specs.")
    parser.add_argument("--max-series", type=int, default=None, help="Limit entries loaded per dataset spec.")
    parser.add_argument("--sample-series", type=int, default=8, help="Number of sample series panels to plot.")
    parser.add_argument("--max-plot-points", type=int, default=3000, help="Max tail points shown in sample plots.")
    parser.add_argument("--max-dist-values", type=int, default=250000, help="Max values sampled for value histograms.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling values.")
    parser.add_argument("--continue-on-error", action="store_true", help="Log dataset load errors and keep going.")
    return parser.parse_args()


def filtered_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    specs = [DatasetSpec(*row) for row in DATASETS]
    if args.filter:
        needle = args.filter.lower()
        specs = [
            s for s in specs
            if needle in s.name.lower()
            or needle in s.display.lower()
            or needle in s.term.lower()
        ]
    if args.limit_datasets is not None:
        specs = specs[: args.limit_datasets]
    return specs


def _check_gift_eval_env():
    gift_eval = os.environ.get("GIFT_EVAL")
    if not gift_eval:
        raise EnvironmentError(
            "GIFT_EVAL environment variable is not set.\n"
            "Add 'GIFT_EVAL=/path/to/gift-eval' to your .env file at the repo root."
        )
    if not Path(gift_eval).exists():
        raise EnvironmentError(
            f"GIFT_EVAL path does not exist: {gift_eval}\n"
            "Check your .env file at the repo root."
        )
    print(f"GIFT_EVAL={gift_eval}")


def main():
    args = parse_args()
    _check_gift_eval_env()
    _ensure_runtime_deps(needs_plot=False)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    specs = filtered_specs(args)
    print(f"Inspecting {len(specs)} dataset specs. Output: {out_root}")

    index_rows = []
    series_rows = []
    failures = []

    for i, spec in enumerate(specs, start=1):
        print(f"[{i:03d}/{len(specs):03d}] {spec.name} term={spec.term} display={spec.display}")
        try:
            summary, rows = inspect_dataset(
                spec=spec,
                out_root=out_root,
                max_series=args.max_series,
                sample_series=args.sample_series,
                max_plot_points=args.max_plot_points,
                max_dist_values=args.max_dist_values,
                seed=args.seed,
            )
            index_rows.append(summary)
            series_rows.extend(rows)
        except Exception as exc:
            failure = {
                "dataset_name": spec.name,
                "term": spec.term,
                "display": spec.display,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"  ERROR: {failure['error_type']}: {failure['error']}")
            if not args.continue_on_error:
                raise

    pd.DataFrame(index_rows).to_csv(out_root / "index.csv", index=False)
    pd.DataFrame(series_rows).to_csv(out_root / "series_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_root / "failures.csv", index=False)

    print(f"Done. Wrote {out_root / 'index.csv'} and {out_root / 'series_summary.csv'}")
    if failures:
        print(f"Some datasets failed. See {out_root / 'failures.csv'}")


if __name__ == "__main__":
    # Avoid surprises from user-level matplotlib config in batch jobs.
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
