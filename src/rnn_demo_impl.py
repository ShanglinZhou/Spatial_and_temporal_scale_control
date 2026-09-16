#!/usr/bin/env python3
"""Functional test using all human files and one representative trained RNN."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "src"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from analysis_all_common import load_human_condition_means  # noqa: E402
from model import RNN_Dale  # noqa: E402
from tasks import (  # noqa: E402
    generate_input_motorTraj_circle,
    generate_target_motorTraj_circle,
    list_eval_conditions,
)


SEED = 20260641
CHECKPOINT = ROOT / "models" / "representative_rnn_rep15.pt"
METRICS_PATH = ROOT / "outputs" / "rnn_demo" / "rnn_demo_metrics.json"
FIGURE_PATH = ROOT / "outputs" / "rnn_demo" / "rnn_demo.png"


def load_checkpoint(path: Path):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def select_conditions(hp):
    conditions = [item for item in list_eval_conditions(hp) if item.get("condition_type") == "train"]
    if len(conditions) <= 5:
        return conditions
    indices = np.linspace(0, len(conditions) - 1, 5).round().astype(int)
    return [conditions[int(index)] for index in indices]


def main() -> None:
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    participant_files = sorted((ROOT / "human_data" / "clean_data").glob("sub-*_clean.mat"))
    human_rows = load_human_condition_means()

    checkpoint = load_checkpoint(CHECKPOINT)
    hp = dict(checkpoint["hp"])
    hp["device"] = "cpu"
    hp["batch_size"] = 1
    hp["sigma"] = 0.0
    rnn = RNN_Dale(hp)
    rnn.load_state_dict(checkpoint["model_state"], strict=True)

    results = []
    trajectories = []
    with torch.no_grad():
        for condition in select_conditions(hp):
            stim, label = generate_input_motorTraj_circle(
                hp, mode="eval", eval_conditions=condition, batch_size=1
            )
            target, mask = generate_target_motorTraj_circle(hp, label, batch_size=1)
            _, output, _, loss = rnn(stim, label, target, mask)
            output_np = output.detach().cpu().numpy()[0:2, 0, :]
            target_np = np.asarray(target, dtype=float)[0:2, 0, :]
            mask_np = np.asarray(mask, dtype=float)[0:2, 0, :]
            valid = mask_np > 0
            rmse = float(np.sqrt(np.mean((output_np[valid] - target_np[valid]) ** 2)))
            results.append(
                {
                    "size_level": float(condition["size_level"]),
                    "speed_level": float(condition["speed_level"]),
                    "target_radius": float(condition["target_radius"]),
                    "target_duration": float(condition["target_duration"]),
                    "rmse": rmse,
                    "model_loss": float(loss.detach().cpu()),
                }
            )
            trajectories.append((target_np, output_np, condition))

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(1, len(trajectories), figsize=(178.0 / 25.4, 36.0 / 25.4))
    axes = np.atleast_1d(axes)
    for ax, (target, output, condition) in zip(axes, trajectories):
        ax.plot(target[0], target[1], color="0.65", lw=0.8, label="Target")
        ax.plot(output[0], output[1], color=(0.0, 0.784, 0.784), lw=0.9, label="RNN")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("R {:.2f}, T {:.2f} s".format(condition["target_radius"], condition["target_duration"]))
        ax.tick_params(direction="out", width=0.5, length=2.835, colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.27, top=0.78, wspace=0.45)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGURE_PATH), dpi=300, facecolor="white")
    plt.close(fig)

    elapsed = time.time() - started
    metrics = {
        "description": "Lightweight functional test; not a replacement for the 20-seed manuscript inference.",
        "seed": SEED,
        "participant_files": len(participant_files),
        "human_condition_rows": len(human_rows),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_best_metric": float(checkpoint["best_metric"]),
        "conditions": results,
        "mean_demo_rmse": float(np.mean([item["rmse"] for item in results])),
        "runtime_seconds": elapsed,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Loaded {} anonymized participants.".format(len(participant_files)))
    print("Computed {} participant-condition rows.".format(len(human_rows)))
    print("Mean demonstration RMSE: {:.6f}".format(metrics["mean_demo_rmse"]))
    print("Saved {}".format(METRICS_PATH))
    print("Saved {}".format(FIGURE_PATH))


if __name__ == "__main__":
    main()
