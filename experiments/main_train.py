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
      and wrap them as PyTorch DataLoaders.
  4.  Random search of N_TRIALS hyperparameter configurations sampled from
      a pre-defined grid (HF transformers PatchTSTConfig).
  5.  Train each trial with:
          - validation every VAL_EVERY_N_EPOCHS
          - early stopping with patience EARLY_STOPPING_PATIENCE
          - Student-t distribution head + NLL loss (probabilistic)
      Save *only* the weights with best validation loss across all trials.
  6.  Plot the random-search summary:
          - scatter (trial index vs. val loss)
          - running "best-so-far" curve connecting the points that improved
            upon all previous trials (everything else is dropped).

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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import time
import random
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
import gc

from dotenv import load_dotenv
load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction


# ==============================================================================
#  EXPERIMENT CONFIGURATION
# ==============================================================================

DATASETS = [
    ("solar/10T",                   "short",  "Solar-10T",        False),
    ("LOOP_SEATTLE/5T",             "short",  "LoopSeattle-5T",   False),
    ("electricity/15T",             "short",  "Electricity-15T",  False),
    ("bizitobs_application",        "short",  "BizITObsApp",      True),
]

# -- Fixed training-loop hyperparameters ---------------------------------------
CONTEXT_LENGTH         = 512
N_TRIALS               = 10
MAX_EPOCHS             = 10
VAL_EVERY_N_EPOCHS     = 2
EARLY_STOPPING_PATIENCE = 2           # in "validation events" (not epochs)
BATCH_SIZE             = 128
WEIGHT_DECAY           = 1e-4
GRAD_CLIP              = 1.0
MAX_TRAIN_WINDOWS      = 50_000        # cap per-(dataset,term) train windows
MAX_VAL_WINDOWS        = 5_000
NUM_DATALOADER_WORKERS = 0
SEED                   = 42

# -- Random search HP space ----------------------------------------------------
HP_SPACE = {
    "patch_length"        : [8, 16, 32],
    "d_model"             : [64, 128, 256],
    "num_hidden_layers"   : [3, 6],
    "num_attention_heads" : [4, 8],
    "dropout"             : [0.0, 0.1, 0.2],
    "learning_rate"       : [1e-4, 2.5e-4, 5e-4],
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
    d_model:             int
    num_hidden_layers:   int
    num_attention_heads: int
    dropout:             float
    learning_rate:       float

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
#  TRAINING / VALIDATION
# ==============================================================================

def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(past_values=x, future_values=y)
            bs = x.shape[0]
            total_loss += out.loss.item() * bs
            total_n    += bs
    return total_loss / max(total_n, 1)


def train_trial(
    trial: TrialConfig,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val:   torch.Tensor, y_val:   torch.Tensor,
    context_length: int, horizon: int,
    device: str,
    trial_idx: int,
) -> Tuple[float, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    """
    Train a single trial. Returns (best_val_loss, history, best_state_dict).
    """
    model = build_patchtst(trial, context_length, horizon).to(device)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_DATALOADER_WORKERS, pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_DATALOADER_WORKERS, pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial.learning_rate, weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_left = EARLY_STOPPING_PATIENCE
    history       = {"train_loss": [], "val_loss": [], "val_epochs": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss, running_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(past_values=x, future_values=y)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            bs = x.shape[0]
            running_loss += loss.item() * bs
            running_n    += bs
        train_loss = running_loss / max(running_n, 1)
        history["train_loss"].append(train_loss)

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == MAX_EPOCHS:
            val_loss = evaluate(model, val_loader, device)
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            print(
                Fore.YELLOW
                + f"    [trial {trial_idx:03d}] epoch {epoch:3d}  "
                + f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                + Fore.RESET
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = EARLY_STOPPING_PATIENCE
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(
                        Fore.MAGENTA
                        + f"    [trial {trial_idx:03d}] early stop at epoch {epoch}"
                        + Fore.RESET
                    )
                    break

    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    return best_val_loss, history, best_state


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

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    print(Fore.CYAN + f"Device: {device}" + Fore.RESET)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CACHE_ROOT, f"runs/{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    trial_configs = sample_trial_configs(N_TRIALS, seed=SEED)
    print(Fore.CYAN + f"Sampled {len(trial_configs)} unique trial configs" + Fore.RESET)

    summary_rows = []

    for ge_name, term, dataset_display, to_univariate in DATASETS:
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  DATASET: {ge_name}  term={term}  ({dataset_display})" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        ge_dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
        horizon = ge_dataset.prediction_length
        print(Fore.CYAN
              + f"  freq={ge_dataset.freq}  horizon={horizon}  "
              + f"target_dim={ge_dataset.target_dim}" + Fore.RESET)

        # -- Build train/val tensors once per (dataset, term) ------------------
        x_train, y_train, x_val, y_val = build_train_val_tensors(
            ge_dataset, CONTEXT_LENGTH, horizon,
        )
        print(Fore.CYAN
              + f"  train: x={tuple(x_train.shape)}  y={tuple(y_train.shape)}  |  "
              + f"val: x={tuple(x_val.shape)}  y={tuple(y_val.shape)}"
              + Fore.RESET)

        # -- Random search -----------------------------------------------------
        ds_dir = _dataset_dir(dataset_display, term)
        os.makedirs(os.path.join(ds_dir, "trials"), exist_ok=True)

        # Bootstrap "global best" from any already-saved best_config / best_model
        global_best_val = float("inf")
        if os.path.isfile(_best_config_path(dataset_display, term)):
            try:
                with open(_best_config_path(dataset_display, term)) as f:
                    prev = json.load(f)
                global_best_val = float(prev.get("val_loss", float("inf")))
                print(Fore.WHITE
                      + f"  Existing best on disk: val_loss={global_best_val:.4f} "
                      + f"(trial {prev.get('trial_idx', '?')})"
                      + Fore.RESET)
            except Exception:
                pass

        for trial_idx, trial in enumerate(trial_configs):
            cached = _load_trial_result(dataset_display, term, trial_idx)
            if cached is not None and "val_loss" in cached \
               and not np.isnan(cached["val_loss"]):
                print(Fore.WHITE
                      + f"  CACHED  trial {trial_idx:03d}: "
                      + f"val_loss={cached['val_loss']:.4f}" + Fore.RESET)
                summary_rows.append({
                    "dataset_display": dataset_display, "term": term,
                    "trial_idx": trial_idx, **cached,
                })
                continue

            print(Fore.YELLOW
                  + f"\n  > trial {trial_idx:03d} / {len(trial_configs)-1}: "
                  + f"{asdict(trial)}" + Fore.RESET)
            t0 = time.perf_counter()
            try:
                val_loss, history, best_state = train_trial(
                    trial, x_train, y_train, x_val, y_val,
                    CONTEXT_LENGTH, horizon, device, trial_idx,
                )
            except Exception as exc:
                print(Fore.RED + f"    trial {trial_idx:03d} FAILED: {exc}" + Fore.RESET)
                val_loss, history, best_state = float("nan"), {}, None

            elapsed = time.perf_counter() - t0

            result = {
                "trial_idx": trial_idx,
                **asdict(trial),
                "val_loss": float(val_loss) if not np.isnan(val_loss) else float("nan"),
                "elapsed_seconds": round(elapsed, 2),
                "history": history,
            }
            _save_trial_result(dataset_display, term, trial_idx, result)
            summary_rows.append({
                "dataset_display": dataset_display, "term": term,
                **result,
            })

            # -- Keep only the GLOBAL best weights across trials ---------------
            if best_state is not None and not np.isnan(val_loss) \
               and val_loss < global_best_val:
                global_best_val = val_loss
                torch.save(best_state, _best_model_path(dataset_display, term))
                with open(_best_config_path(dataset_display, term), "w") as f:
                    json.dump({
                        "trial_idx": trial_idx,
                        **asdict(trial),
                        "val_loss": float(val_loss),
                        "context_length":    CONTEXT_LENGTH,
                        "prediction_length": horizon,
                        "num_input_channels": 1,
                    }, f, indent=2)
                print(Fore.GREEN
                      + f"    NEW BEST: val_loss={val_loss:.4f} -> weights saved"
                      + Fore.RESET)
            else:
                print(Fore.MAGENTA
                      + f"    val_loss={val_loss:.4f}  (global best={global_best_val:.4f})"
                      + Fore.RESET)
            print(Fore.MAGENTA + f"    TIME {elapsed:.1f}s" + Fore.RESET)

        # -- Per-dataset summary plot + CSV ------------------------------------
        df_ds = pd.DataFrame([r for r in summary_rows
                              if r["dataset_display"] == dataset_display
                              and r["term"] == term])
        df_ds = df_ds[["trial_idx", "val_loss", "patch_length", "d_model",
                       "num_hidden_layers", "num_attention_heads",
                       "dropout", "learning_rate", "elapsed_seconds"]]
        csv_path = os.path.join(ds_dir, "random_search.csv")
        df_ds.to_csv(csv_path, index=False)
        plot_random_search_summary(
            df_ds.dropna(subset=["val_loss"]),
            os.path.join(ds_dir, "random_search_summary.png"),
            dataset_display, term,
        )
        print(Fore.GREEN + f"  Summary CSV: {csv_path}" + Fore.RESET)

        del x_train, y_train, x_val, y_val, ge_dataset
        gc.collect()
        torch.cuda.empty_cache()

    # -- Global summary --------------------------------------------------------
    global_df = pd.DataFrame(summary_rows)
    global_csv = os.path.join(run_dir, "random_search_all.csv")
    global_df.drop(columns=["history"], errors="ignore").to_csv(global_csv, index=False)
    print(Fore.GREEN + f"\n  Global summary: {global_csv}" + Fore.RESET)

    print(Fore.GREEN + "\nTraining pipeline done." + Fore.RESET)


if __name__ == "__main__":
    main()