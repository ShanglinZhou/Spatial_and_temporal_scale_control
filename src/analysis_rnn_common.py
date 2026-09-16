from __future__ import annotations

import math
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analysis_all_common import EPS, _conditions_from_eval, _movement_window, _phase_velocity_from_xy


# Canonical mechanism configuration fixed on 2026-08-21 after the updated
# human analysis showed no reliable radius-to-duration motor effect.
MAIN_RNN_RANK = "full"
MAIN_RNN_READOUT = "free"
MAIN_RNN_TIME_CODE = "speedcoded"
MAIN_RNN_INPUT_OVERLAP = -0.8
MAIN_RNN_REPRESENTATIVE_REP = 15
MAIN_RNN_EVAL_GRID_NAME = "narrow0408_fulltarget_train_interp_extra_13x13"
MAIN_RNN_CONDITION_TYPES = frozenset({"train", "interpolation"})


def install_numpy_pickle_aliases() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_payload(path) -> Dict:
    import torch  # type: ignore

    install_numpy_pickle_aliases()
    return torch.load(path, map_location="cpu", weights_only=False)


def fixed_range_normalize(values: Sequence[float], reference_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    center = (float(np.nanmin(reference_values)) + float(np.nanmax(reference_values))) / 2.0
    half_range = (float(np.nanmax(reference_values)) - float(np.nanmin(reference_values))) / 2.0
    return (values - center) / max(half_range, EPS)


def lowrank_coordinates(rates_nt: np.ndarray, m_vectors: np.ndarray) -> np.ndarray:
    rates_nt = np.asarray(rates_nt, dtype=float)
    m_vectors = np.asarray(m_vectors, dtype=float)
    return rates_nt.T @ m_vectors / float(m_vectors.shape[0])


def pca_basis(points: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    mean = np.nanmean(points, axis=0)
    centered = points - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:n_components].T


def project_pca(points: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - mean) @ basis


def condition_rows(payload: Dict, include_latent: bool = True) -> List[Dict]:
    hp = payload["hp"]
    eval_block = payload["eval"]
    outputs = np.asarray(eval_block["outputs"], dtype=float)
    inputs = np.asarray(eval_block["inputs"], dtype=float)
    rates = np.asarray(eval_block.get("rates", []), dtype=float)
    m_vectors = np.asarray(payload["model"].get("m_vectors"), dtype=float)
    conditions = _conditions_from_eval(hp, eval_block, outputs.shape[0])
    dt_s = float(hp["dt"]) / 1000.0
    rows: List[Dict] = []
    for ci, cond in enumerate(conditions):
        if str(cond.get("condition_type", "unknown")) not in MAIN_RNN_CONDITION_TYPES:
            continue
        radii = []
        durations = []
        latent_means = []
        latent_speeds = []
        latent_path_lengths = []
        latent_endpoints = []
        for ti in range(outputs.shape[1]):
            out_xy = np.asarray(outputs[ci, ti, :2], dtype=float)
            u_in = np.asarray(inputs[ci, ti], dtype=float)
            t0, _ = _movement_window(u_in, hp, float(cond["target_duration"]), out_xy.shape[-1])
            n_steps = int(round(1.1 * float(cond["target_duration"]) / dt_s))
            t1 = max(t0 + 3, min(out_xy.shape[-1], t0 + max(3, n_steps)))
            radius, phase_velocity = _phase_velocity_from_xy(out_xy[:, t0:t1], dt_s)
            radii.append(radius)
            durations.append(float(2.0 * math.pi / max(phase_velocity, EPS)))
            if include_latent and rates.size and m_vectors.size:
                kappa = lowrank_coordinates(np.asarray(rates[ci, ti, :, t0:t1], dtype=float), m_vectors)
                diffs = np.diff(kappa, axis=0)
                path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
                elapsed = max((kappa.shape[0] - 1) * dt_s, EPS)
                latent_means.append(np.nanmean(kappa, axis=0))
                latent_speeds.append(path_length / elapsed)
                latent_path_lengths.append(path_length)
                latent_endpoints.append(kappa[-1])
        row = {
            "condition_index": int(ci),
            "condition_type": str(cond.get("condition_type", "unknown")),
            "size_level": float(cond.get("size_level", np.nan)),
            "speed_level": float(cond.get("speed_level", np.nan)),
            "target_radius": float(cond["target_radius"]),
            "target_duration": float(cond["target_duration"]),
            "produced_radius": float(np.nanmean(radii)),
            "produced_duration": float(np.nanmean(durations)),
        }
        if include_latent and latent_means:
            mean_kappa = np.nanmean(np.stack(latent_means), axis=0)
            endpoint_kappa = np.nanmean(np.stack(latent_endpoints), axis=0)
            row.update(
                {
                    "latent_mean": mean_kappa,
                    "latent_endpoint": endpoint_kappa,
                    "latent_speed": float(np.nanmean(latent_speeds)),
                    "latent_timescale": float(1.0 / max(np.nanmean(latent_speeds), EPS)),
                    "latent_path_length": float(np.nanmean(latent_path_lengths)),
                }
            )
        rows.append(row)
    return rows


def fit_behavior_coefficients(rows: Sequence[Dict]) -> Dict:
    target_r = [float(row["target_radius"]) for row in rows]
    target_t = [float(row["target_duration"]) for row in rows]
    r_in = fixed_range_normalize(target_r, target_r)
    t_in = fixed_range_normalize(target_t, target_t)
    r_out = fixed_range_normalize([float(row["produced_radius"]) for row in rows], target_r)
    t_out = fixed_range_normalize([float(row["produced_duration"]) for row in rows], target_t)
    valid = np.isfinite(r_in) & np.isfinite(t_in) & np.isfinite(r_out) & np.isfinite(t_out)
    design = np.column_stack([np.ones(int(valid.sum())), r_in[valid], t_in[valid]])
    beta_r, *_ = np.linalg.lstsq(design, r_out[valid], rcond=None)
    beta_t, *_ = np.linalg.lstsq(design, t_out[valid], rcond=None)
    return {
        "intercept_R": float(beta_r[0]),
        "beta_RR": float(beta_r[1]),
        "beta_RT": float(beta_r[2]),
        "intercept_T": float(beta_t[0]),
        "beta_TR": float(beta_t[1]),
        "beta_TT": float(beta_t[2]),
        "n_conditions": int(valid.sum()),
        "condition_number": float(np.linalg.cond(design)),
    }


def nearest_condition_indices(payload: Dict, pairs: Sequence[Tuple[float, float]], condition_type: str = "train") -> List[int]:
    eval_block = payload["eval"]
    conditions = _conditions_from_eval(payload["hp"], eval_block, np.asarray(eval_block["outputs"]).shape[0])
    allowed = [i for i, cond in enumerate(conditions) if str(cond.get("condition_type")) == condition_type]
    if not allowed:
        allowed = list(range(len(conditions)))
    out = []
    for radius, duration in pairs:
        best = min(
            allowed,
            key=lambda i: abs(float(conditions[i]["target_radius"]) - float(radius))
            + abs(float(conditions[i]["target_duration"]) - float(duration)),
        )
        if best not in out:
            out.append(best)
    return out


def grouped_condition_extremes(rows: Sequence[Dict]) -> Dict[str, float]:
    radii = np.asarray([float(row["target_radius"]) for row in rows if row["condition_type"] == "train"], dtype=float)
    durations = np.asarray([float(row["target_duration"]) for row in rows if row["condition_type"] == "train"], dtype=float)
    return {
        "r_min": float(np.nanmin(radii)),
        "r_mid": float(np.nanmedian(radii)),
        "r_max": float(np.nanmax(radii)),
        "t_min": float(np.nanmin(durations)),
        "t_mid": float(np.nanmedian(durations)),
        "t_max": float(np.nanmax(durations)),
    }
