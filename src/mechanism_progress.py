"""Shared-template progress estimator required by the Figure 8 model."""

from __future__ import annotations

from typing import Dict

import numpy as np


EPS = 1e-9
TEMPLATE_MAX_STEP_FRACTION = 0.04
TEMPLATE_FIT_LO_FRACTION = 0.10
TEMPLATE_FIT_HI_FRACTION = 0.90


def _shape_normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Remove translation and isotropic RMS scale without changing time."""
    points = np.asarray(points, dtype=float)
    center = np.nanmean(points, axis=0, keepdims=True)
    centered = points - center
    scale = float(np.sqrt(np.nanmean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(scale) or scale < EPS:
        return np.full_like(points, np.nan), center.reshape(-1), float("nan")
    return centered / scale, center.reshape(-1), scale


def _continuous_forward_template_path(
    squared_distance: np.ndarray,
    max_step: int,
) -> np.ndarray:
    """Globally track a monotonic open-end template path with bounded jumps.

    The first sample is anchored to template onset because templates are built
    from movement-onset-aligned trajectories. The endpoint is deliberately not
    anchored, so the estimator does not force every trajectory to finish a
    complete cycle.
    """
    cost = np.asarray(squared_distance, dtype=float)
    if cost.ndim != 2 or cost.shape[0] < 1 or cost.shape[1] < 1:
        return np.asarray([], dtype=int)
    n_time, n_template = cost.shape
    max_step = int(np.clip(max_step, 1, max(n_template - 1, 1)))
    previous = np.full(n_template, np.inf, dtype=float)
    previous[0] = float(cost[0, 0])
    back_step = np.zeros((n_time, n_template), dtype=np.int16)
    for time_index in range(1, n_time):
        candidates = np.full((max_step + 1, n_template), np.inf, dtype=float)
        candidates[0] = previous
        for step in range(1, max_step + 1):
            candidates[step, step:] = previous[:-step]
        choice = np.argmin(candidates, axis=0)
        best = candidates[choice, np.arange(n_template)]
        previous = cost[time_index] + best
        back_step[time_index] = choice.astype(np.int16)
    endpoint = int(np.nanargmin(previous))
    path = np.zeros(n_time, dtype=int)
    path[-1] = endpoint
    for time_index in range(n_time - 1, 0, -1):
        path[time_index - 1] = path[time_index] - int(back_step[time_index, path[time_index]])
    return path


def _robust_progress_fit(time_s: np.ndarray, progress: np.ndarray) -> tuple[float, float, float]:
    """Huber iteratively reweighted line fit with an ordinary-fit R2."""
    time_s = np.asarray(time_s, dtype=float)
    progress = np.asarray(progress, dtype=float)
    design = np.column_stack([np.ones(time_s.size, dtype=float), time_s])
    beta, *_ = np.linalg.lstsq(design, progress, rcond=None)
    for _ in range(20):
        residual = progress - design @ beta
        center = float(np.nanmedian(residual))
        mad = float(np.nanmedian(np.abs(residual - center)))
        scale = max(1.4826 * mad, EPS)
        cutoff = 1.345 * scale
        absolute = np.abs(residual)
        weight = np.ones_like(absolute)
        outside = absolute > cutoff
        weight[outside] = cutoff / np.maximum(absolute[outside], EPS)
        weighted_design = design * np.sqrt(weight)[:, None]
        weighted_progress = progress * np.sqrt(weight)
        updated, *_ = np.linalg.lstsq(weighted_design, weighted_progress, rcond=None)
        if np.linalg.norm(updated - beta) < 1e-10:
            beta = updated
            break
        beta = updated
    predicted = design @ beta
    denominator = float(np.sum((progress - np.mean(progress)) ** 2))
    r2 = float(1.0 - np.sum((progress - predicted) ** 2) / max(denominator, EPS))
    return float(beta[1]), float(beta[0]), r2


def _shared_template_progress_details(
    xy_points: np.ndarray,
    template_info: Dict,
    dt_s: float,
) -> Dict:
    """Scale-invariant, continuity-constrained shared-template progress."""
    xy_points = np.asarray(xy_points, dtype=float)
    template = np.asarray(template_info["template"], dtype=float)
    template_progress = np.asarray(template_info["progress"], dtype=float)
    empty = {
        "time": np.arange(max(xy_points.shape[0], 0), dtype=float) * float(dt_s),
        "nearest_template_indices": np.asarray([], dtype=int),
        "continuous_template_indices": np.asarray([], dtype=int),
        "raw_template_progress": np.asarray([], dtype=float),
        "oriented_raw_progress": np.asarray([], dtype=float),
        "monotonic_progress": np.asarray([], dtype=float),
        "fit_mask": np.asarray([], dtype=bool),
        "fit_velocity": float("nan"),
        "fit_intercept": float("nan"),
        "fit_r2": float("nan"),
        "duration": float("nan"),
        "observed_progress_max": float("nan"),
        "progress_coverage": float("nan"),
        "fit_progress_bounds": [float("nan"), float("nan")],
        "orientation": "forward_template_time",
        "trajectory_center": np.asarray([], dtype=float),
        "template_center": np.asarray([], dtype=float),
        "trajectory_scale": float("nan"),
        "template_scale": float("nan"),
        "normalized_matching_rmse": float("nan"),
        "raw_nearest_max_step_cycles": float("nan"),
        "continuous_max_step_cycles": float("nan"),
        "max_step_indices": 0,
        "endpoint_constraint": "open",
        "method": "scale_invariant_continuous_template_progress",
    }
    if (
        xy_points.ndim != 2
        or template.ndim != 2
        or xy_points.shape[0] < 3
        or template.shape[0] < 3
        or xy_points.shape[1] != template.shape[1]
        or template_progress.size != template.shape[0]
    ):
        return empty
    normalized_xy, xy_center, xy_scale = _shape_normalize(xy_points)
    normalized_template, template_center, template_scale = _shape_normalize(template)
    if not np.all(np.isfinite(normalized_xy)) or not np.all(np.isfinite(normalized_template)):
        return empty
    squared_distance = np.sum(
        (normalized_xy[:, None, :] - normalized_template[None, :, :]) ** 2,
        axis=2,
    )
    nearest_indices = np.nanargmin(squared_distance, axis=1).astype(int)
    max_step = max(1, int(np.ceil(TEMPLATE_MAX_STEP_FRACTION * (template.shape[0] - 1))))
    path_indices = _continuous_forward_template_path(squared_distance, max_step=max_step)
    if path_indices.size != xy_points.shape[0]:
        return empty
    progress = np.asarray(template_progress[path_indices], dtype=float)
    progress = progress - float(progress[0])
    progress = np.maximum.accumulate(progress)
    pmax = float(np.nanmax(progress))
    if not np.isfinite(pmax) or pmax < EPS:
        return empty
    lo = TEMPLATE_FIT_LO_FRACTION * pmax
    hi = TEMPLATE_FIT_HI_FRACTION * pmax
    fit_mask = (progress >= lo) & (progress <= hi)
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = np.ones_like(progress, dtype=bool)
    t_s = np.arange(progress.size, dtype=float) * float(dt_s)
    if np.count_nonzero(fit_mask) < 3 or np.nanstd(t_s[fit_mask]) < EPS:
        return empty
    velocity, intercept, fit_r2 = _robust_progress_fit(
        t_s[fit_mask], progress[fit_mask]
    )
    duration = float(2.0 * np.pi / max(velocity, EPS)) if np.isfinite(velocity) else float("nan")
    selected_cost = squared_distance[np.arange(path_indices.size), path_indices]
    progress_span = max(float(template_progress[-1] - template_progress[0]), EPS)
    raw_progress = np.asarray(template_progress[nearest_indices], dtype=float)
    return {
        "time": t_s,
        "nearest_template_indices": nearest_indices,
        "continuous_template_indices": path_indices,
        "raw_template_progress": raw_progress,
        "oriented_raw_progress": progress.copy(),
        "monotonic_progress": progress,
        "fit_mask": fit_mask,
        "fit_velocity": velocity,
        "fit_intercept": float(intercept),
        "fit_r2": fit_r2,
        "duration": duration,
        "observed_progress_max": pmax,
        "progress_coverage": float(pmax / progress_span),
        "fit_progress_bounds": [float(lo), float(hi)],
        "orientation": "forward_template_time",
        "trajectory_center": xy_center,
        "template_center": template_center,
        "trajectory_scale": float(xy_scale),
        "template_scale": float(template_scale),
        "normalized_matching_rmse": float(np.sqrt(np.mean(selected_cost))),
        "raw_nearest_max_step_cycles": float(
            np.max(np.abs(np.diff(raw_progress))) / progress_span
        ),
        "continuous_max_step_cycles": float(
            np.max(np.abs(np.diff(progress))) / progress_span
        ),
        "max_step_indices": int(max_step),
        "endpoint_constraint": "open",
        "method": "scale_invariant_continuous_template_progress",
    }
