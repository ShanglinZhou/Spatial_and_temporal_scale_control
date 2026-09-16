from __future__ import annotations

import argparse
import hashlib
import inspect
from typing import Dict, Sequence, Tuple
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares
import scipy

from analysis_all_common import EPS, FIG_DIR, clean_axis, finish_figure, fit_circle_xy, load_human_condition_means, write_json
from analysis_behavior_common import COLORS, build_matched_rows, fit_subject_models, label_panel, setup_style
from analysis_rnn_common import fixed_range_normalize, pca_basis, project_pca
from mechanism_progress import _shared_template_progress_details


OUTPUT_FIGURE = "Fig8_Hopf_model_3D.png"
OUTPUT_STATS = "Fig8_Hopf_model_3D_stats.json"
FIT_CACHE_VERSION = "fig8_hopf_per_rep_fit_v1"
FIT_CACHE_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "hopf_model_3d"
    / "Fig8_Hopf_model_3D_per_rep_fits_v1.json"
)

R_LEVELS = np.linspace(-1.0, 1.0, 5)
T_LEVELS = np.linspace(-1.0, 1.0, 5)
R_GRID, T_GRID = np.meshgrid(R_LEVELS, T_LEVELS, indexing="ij")
INPUT_GRID = np.column_stack([R_GRID.ravel(), T_GRID.ravel()])
HOPF_TARGET_CENTER = 1.0
HOPF_TARGET_HALF_RANGE = 0.5
MATRIX_NORMALIZATION_VERSION = "positive_target_range_v2"
PLANE_ANGLE_SIGN_MODE = "free_theta_R_theta_T"
HOPF_RECURRENT_FEATURE_MODE = "main_pca_scale_invariant_continuous_template_progress"
SHARED_TEMPLATE_N_POINTS = 101

PARAM_NAMES = [
    "alpha0",
    "amp_R",
    "amp_T",
    "omega0",
    "omega_R",
    "omega_T",
    "plane_angle0",
    "plane_angle_R",
    "plane_angle_T",
    "readout_xx",
    "readout_xy",
    "readout_xz",
    "readout_yx",
    "readout_yy",
    "readout_yz",
]

MATRIX_LOSS_WEIGHT = 1.0
ANGLE_SLOPE_ERROR_SCALE_DEG = 5.0
ANGLE_SLOPE_MARGIN_DEG = 1.0
ANGLE_SLOPE_CONSTRAINT_WEIGHT = 25.0
PLANE_SIDE_MARGIN_DEG = 5.0
PLANE_SIDE_CONSTRAINT_WEIGHT = 25.0
READOUT_SHARED_AXIS_MARGIN_DEG = 1.0
READOUT_SHARED_AXIS_CONSTRAINT_WEIGHT = 25.0
CIRCULARITY_LOSS_WEIGHT = 2.0
N_CIRCULARITY_SAMPLES = 96
HOPF_FIT_TOL = 1e-5
HOPF_REP_FIT_MAX_NFEV = 800
HOPF_REP_FALLBACK_MAX_NFEV = 1600
HOPF_REP_ACCEPTABLE_COST = 0.06
FIG8_FIT_SEED = 5


def _set_fit_seed() -> None:
    random.seed(FIG8_FIT_SEED)
    np.random.seed(FIG8_FIT_SEED)


DEFAULT_PARAMS = np.asarray(
    [1.35, 0.35, 0.35, 6.2, -0.25, -1.2, 0.65, -0.18, 0.24, 1.20, 0.0, -0.45, 0.0, 1.20, 0.0],
    dtype=float,
)
LOWER_BOUNDS = np.asarray(
    [0.4, 0.0, 0.0, 1.2, -2.5, -2.5, 0.05, -1.2, -1.2, 0.2, -1.5, -1.5, -1.5, 0.2, -1.5],
    dtype=float,
)
UPPER_BOUNDS = np.asarray(
    [2.5, 1.5, 1.5, 8.5, 0.0, 0.0, 1.35, 1.2, 1.2, 2.5, 1.5, 1.5, 1.5, 2.5, 1.5],
    dtype=float,
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_matrix(value, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (2, 2):
        return arr
    if fallback is not None:
        return np.asarray(fallback, dtype=float)
    return np.full((2, 2), np.nan, dtype=float)


def _normalization_from_rows(condition_rows):
    stim_r = np.asarray([float(row["stimx"]) for row in condition_rows], dtype=float)
    stim_t = np.asarray([float(row["stimt"]) for row in condition_rows], dtype=float)
    bounds = {
        "R": {"min": float(np.nanmin(stim_r)), "max": float(np.nanmax(stim_r))},
        "T": {"min": float(np.nanmin(stim_t)), "max": float(np.nanmax(stim_t))},
    }
    return {
        key: {
            **value,
            "center": (value["min"] + value["max"]) / 2.0,
            "half_range": (value["max"] - value["min"]) / 2.0,
        }
        for key, value in bounds.items()
    }


def _matrix_from_coefficient_row(row: Dict) -> np.ndarray:
    return np.asarray(
        [[row["beta_RR"], row["beta_RT"]], [row["beta_TR"], row["beta_TT"]]],
        dtype=float,
    )


def _human_subject_matrices() -> Tuple[np.ndarray, list]:
    rows = load_human_condition_means()
    normalization = _normalization_from_rows(rows)
    matched_rows = build_matched_rows(rows)
    coefficients, _ = fit_subject_models(matched_rows, normalization)
    matrices = np.asarray([_matrix_from_coefficient_row(row) for row in coefficients], dtype=float)
    subjects = [str(row["subject"]) for row in coefficients]
    return matrices, subjects


def _mean_human_matrix() -> np.ndarray:
    matrices, _ = _human_subject_matrices()
    if matrices.size:
        return np.nanmean(matrices, axis=0)
    path = FIG_DIR / "Fig1_behavior_spatiotemporal_control_stats.json"
    if not path.exists():
        return np.asarray([[0.8, 0.12], [0.05, 0.95]], dtype=float)
    data = _load_json(path)
    rows = (
        data.get("cross_dimensional_control", {})
        .get("motor_model", {})
        .get("subject_coefficients", [])
    )
    if not rows:
        return np.asarray([[0.8, 0.12], [0.05, 0.95]], dtype=float)
    return np.asarray(
        [
            [np.nanmean([row["beta_RR"] for row in rows]), np.nanmean([row["beta_RT"] for row in rows])],
            [np.nanmean([row["beta_TR"] for row in rows]), np.nanmean([row["beta_TT"] for row in rows])],
        ],
        dtype=float,
    )


def _load_rnn_targets() -> Dict[str, np.ndarray]:
    path = FIG_DIR / "Fig4_pca_rnn_mechanism_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Figure 4 RNN mechanism stats: {path}")
    data = _load_json(path)
    panels = data.get("panels", {})
    selected = data.get("selected_parameters", {})
    dynamics = data.get("recurrent_dynamics", {})
    behavior_panel = panels.get("H_composed_vs_real_behavior", {}) or panels.get("I_composed_vs_real_behavior", {})
    input_panel = panels.get("E_input_to_recurrent", {}) or panels.get("F_input_to_recurrent", {})
    output_panel = panels.get("F_recurrent_to_output", {}) or panels.get("G_recurrent_to_output", {})
    composed_panel = panels.get("G_composed_behavior", {}) or panels.get("H_composed_behavior", {})
    behavior = _as_matrix(
        behavior_panel.get("real_rnn_behavior_mean"),
        fallback=data.get("behavior_matrix"),
    )
    input_to_recurrent = _as_matrix(
        input_panel.get("mean_matrix"),
        fallback=panels.get("I", {}).get("mean_matrix"),
    )
    recurrent_to_output = _as_matrix(
        output_panel.get("mean_matrix"),
        fallback=panels.get("J", {}).get("mean_matrix"),
    )
    composed = _as_matrix(
        composed_panel.get("mean_matrix"),
        fallback=panels.get("K", {}).get("mean_matrix"),
    )
    rep_ids = [int(rep) for rep in data.get("across_rep_summary", {}).get("rep_ids", [])]
    input_rep_matrices = np.asarray(input_panel.get("rep_matrices", []), dtype=float)
    output_rep_matrices = np.asarray(output_panel.get("rep_matrices", []), dtype=float)
    composed_rep_matrices = np.asarray(composed_panel.get("rep_matrices", []), dtype=float)
    behavior_rep_matrices = _load_selected_rnn_behavior_reps(selected)
    angle_rep_targets = _load_angle_rep_targets()
    rep_targets = []
    n_rep_matrices = min(len(rep_ids), len(input_rep_matrices), len(output_rep_matrices))
    for idx in range(n_rep_matrices):
        rep_id = int(rep_ids[idx])
        if rep_id not in behavior_rep_matrices:
            continue
        rep_angle = angle_rep_targets.get(rep_id)
        if rep_angle is None:
            continue
        rep_targets.append(
            {
                "rep": rep_id,
                "behavior": np.asarray(behavior_rep_matrices[rep_id], dtype=float),
                "input_to_recurrent": np.asarray(input_rep_matrices[idx], dtype=float),
                "recurrent_to_output": np.asarray(output_rep_matrices[idx], dtype=float),
                "composed": np.asarray(composed_rep_matrices[idx], dtype=float) if len(composed_rep_matrices) > idx else np.full((2, 2), np.nan),
                "angle_targets": rep_angle,
            }
        )
    return {
        "source_stats_file": str(path),
        "recurrent_timing_method": selected.get("recurrent_timing_method", "unknown"),
        "recurrent_time_metric": dynamics.get("time_metric", "unknown"),
        "selected_rep": int(selected.get("rep", -1)) if "rep" in selected else None,
        "behavior": behavior,
        "input_to_recurrent": input_to_recurrent,
        "recurrent_to_output": recurrent_to_output,
        "composed": composed,
        "rep_targets": rep_targets,
    }


def _load_angle_targets() -> Dict[str, float]:
    path = FIG_DIR / "Fig6_main_PCA_radial_axis_alignment_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Figure 8 angle targets require the Figure 6 statistics file: {path}"
        )
    data = _load_json(path)
    panel_d = data.get("panels", {}).get("D", {})
    angle_summary = data.get("readout_plane_angle_summary", {})
    slope_summary = angle_summary.get("slope_summary", {})
    slope_definition = angle_summary.get("slope_definition", {})

    radius_slope = panel_d.get("Radius", {}).get("mean")
    duration_slope = panel_d.get("Duration", {}).get("mean")
    if radius_slope is None or duration_slope is None:
        raise KeyError(
            "Figure 6 angle targets are missing panels.D.Radius.mean or "
            "panels.D.Duration.mean; refusing to substitute fallback values."
        )
    radius_slope = float(radius_slope)
    duration_slope = float(duration_slope)
    if not np.isfinite(radius_slope) or not np.isfinite(duration_slope):
        raise ValueError(
            "Figure 6D angle targets must both be finite; "
            f"received radius={radius_slope}, duration={duration_slope}."
        )

    # Panel D is generated from these across-repetition summaries.  Check the
    # redundant representation so future schema changes fail loudly instead
    # of silently loading an unrelated panel.
    summary_radius = slope_summary.get("rep_mean_radius_slope", {}).get("mean")
    summary_duration = slope_summary.get("rep_mean_duration_slope", {}).get("mean")
    if summary_radius is not None and not np.isclose(radius_slope, float(summary_radius)):
        raise ValueError(
            "Figure 6D radius slope does not match readout_plane_angle_summary."
        )
    if summary_duration is not None and not np.isclose(duration_slope, float(summary_duration)):
        raise ValueError(
            "Figure 6D duration slope does not match readout_plane_angle_summary."
        )
    return {
        "source_stats_file": str(path),
        "source_panel": "D",
        "radius_slope": radius_slope,
        "duration_slope": duration_slope,
        "slope_definition": slope_definition,
    }


def _load_selected_rnn_behavior_reps(selected: Dict) -> Dict[int, np.ndarray]:
    path = FIG_DIR / "ParamSweep_rnn_behavior_summary_stats.json"
    if not path.exists():
        return {}
    data = _load_json(path)
    selected_rank = str(selected.get("rank", selected.get("rank_label", ""))).lower()
    selected_readout = str(selected.get("readout", "")).lower()
    selected_time_code = str(selected.get("time_code", "")).lower()
    selected_overlap = float(selected.get("input_overlap", np.nan))
    for config in data.get("configurations", []):
        config_rank = str(config.get("rank_label", config.get("rank", ""))).lower()
        config_readout = str(config.get("readout_mode", "")).lower()
        config_time_code = str(config.get("time_code", "")).lower()
        config_overlap = float(config.get("input_overlap", np.nan))
        rank_match = config_rank == selected_rank or (selected_rank == "full" and config_rank == "full")
        if (
            rank_match
            and config_readout == selected_readout
            and config_time_code == selected_time_code
            and np.isclose(config_overlap, selected_overlap)
        ):
            return {
                int(row["rep"]): _matrix_from_coefficient_row(row)
                for row in config.get("rep_coefficients", [])
                if "rep" in row
            }
    return {}


def _load_angle_rep_targets() -> Dict[int, Dict[str, float]]:
    path = FIG_DIR / "Fig6_main_PCA_radial_axis_alignment_stats.json"
    if not path.exists():
        return {}
    data = _load_json(path)
    angle_summary = data.get("readout_plane_angle_summary", {})
    records = angle_summary.get("slope_summary", {}).get("rep_mean_records", angle_summary.get("rep_mean_records", []))
    return {
        int(row["rep"]): {
            "radius_slope": float(row["radius_slope"]),
            "duration_slope": float(row["duration_slope"]),
        }
        for row in records
        if "rep" in row and "radius_slope" in row and "duration_slope" in row
    }


def _normalize_to_self(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return fixed_range_normalize(arr, arr)


def _normalize_to_reference(values: Sequence[float], reference_values: Sequence[float] | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    ref = arr if reference_values is None else np.asarray(reference_values, dtype=float)
    return fixed_range_normalize(arr, ref)


def _fit_matrix(
    x1: Sequence[float],
    x2: Sequence[float],
    y1: Sequence[float],
    y2: Sequence[float],
    *,
    x1_ref: Sequence[float] | None = None,
    x2_ref: Sequence[float] | None = None,
    y1_ref: Sequence[float] | None = None,
    y2_ref: Sequence[float] | None = None,
) -> np.ndarray:
    x1 = _normalize_to_reference(x1, x1_ref)
    x2 = _normalize_to_reference(x2, x2_ref)
    y1 = _normalize_to_reference(y1, y1_ref)
    y2 = _normalize_to_reference(y2, y2_ref)
    valid = np.isfinite(x1 + x2 + y1 + y2)
    design = np.column_stack([np.ones(int(valid.sum())), x1[valid], x2[valid]])
    b1, *_ = np.linalg.lstsq(design, y1[valid], rcond=None)
    b2, *_ = np.linalg.lstsq(design, y2[valid], rcond=None)
    return np.asarray([[b1[1], b1[2]], [b2[1], b2[2]]], dtype=float)


def _unpack(params: Sequence[float]) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, params)}


def _positive_target_from_level(level: Sequence[float]) -> np.ndarray:
    return HOPF_TARGET_CENTER + HOPF_TARGET_HALF_RANGE * np.asarray(level, dtype=float)


def _rotation_y(angle: np.ndarray | float) -> np.ndarray:
    a = np.asarray(angle, dtype=float)
    c = np.cos(a)
    s = np.sin(a)
    return np.asarray([[c, np.zeros_like(c), s], [np.zeros_like(c), np.ones_like(c), np.zeros_like(c)], [-s, np.zeros_like(c), c]])


def _readout_matrix(p: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [p["readout_xx"], p["readout_xy"], p["readout_xz"]],
            [p["readout_yx"], p["readout_yy"], p["readout_yz"]],
        ],
        dtype=float,
    )


def _condition_plane_basis(angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = _rotation_y(angle)
    basis_a = np.asarray(rot[:, 0], dtype=float)
    basis_b = np.asarray(rot[:, 1], dtype=float)
    normal = np.cross(basis_a, basis_b)
    normal /= np.linalg.norm(normal) + EPS
    return basis_a, basis_b, normal


def _latent_trajectory(rho: float, angle: float, n: int = N_CIRCULARITY_SAMPLES) -> np.ndarray:
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    u = np.vstack([rho * np.cos(phi), rho * np.sin(phi), np.zeros_like(phi)])
    rot = _rotation_y(angle)
    return rot @ u


def _recurrent_excursion_for_condition(rho: float, angle: float) -> float:
    latent = _latent_trajectory(rho, angle)
    radial = np.sqrt(np.sum(latent**2, axis=0))
    return float(np.nanmax(radial))


def _readout_plane_basis(readout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, singular_values, vt = np.linalg.svd(readout, full_matrices=True)
    if singular_values[1] < 1e-6:
        basis_a = np.asarray([1.0, 0.0, 0.0])
        basis_b = np.asarray([0.0, 1.0, 0.0])
    else:
        basis_a = vt[0]
        basis_b = vt[1]
    normal = np.cross(basis_a, basis_b)
    normal /= np.linalg.norm(normal) + EPS
    return basis_a, basis_b, normal


def _plane_readout_angle_deg(angle: float, readout: np.ndarray) -> float:
    _, _, condition_normal = _condition_plane_basis(angle)
    _, _, readout_normal = _readout_plane_basis(readout)
    cosine = np.clip(abs(float(np.dot(condition_normal, readout_normal))), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _readout_shared_axis_error_deg(readout: np.ndarray) -> float:
    _, _, readout_normal = _readout_plane_basis(readout)
    shared_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    sine = np.clip(abs(float(np.dot(readout_normal, shared_axis))), 0.0, 1.0)
    return float(np.degrees(np.arcsin(sine)))


def _signed_plane_readout_offset_deg(angle: np.ndarray | float, readout: np.ndarray) -> np.ndarray:
    """Signed condition-plane offset from the closest readout orientation.

    The condition planes rotate about the native y axis.  Projecting the
    readout normal into the x-z plane gives the corresponding readout
    orientation in that one-dimensional family.  Plane normals are axial, so
    offsets are wrapped to (-90, 90] degrees.
    """
    row_x = np.asarray(readout[0], dtype=float)
    row_y = np.asarray(readout[1], dtype=float)
    readout_normal = np.cross(row_x, row_y)
    readout_normal /= np.linalg.norm(readout_normal) + EPS
    if readout_normal[2] < 0.0 or (abs(readout_normal[2]) < EPS and readout_normal[0] < 0.0):
        readout_normal = -readout_normal
    readout_angle = np.arctan2(readout_normal[0], readout_normal[2])
    offset = np.asarray(angle, dtype=float) - readout_angle
    offset = (offset + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return np.degrees(offset)


def _output_metrics_for_condition(rho: float, angle: float, readout: np.ndarray) -> Tuple[float, float]:
    latent = _latent_trajectory(rho, angle)
    output = readout @ latent
    radial = np.sqrt(np.sum(output**2, axis=0))
    # These generated traces are complete, uniformly sampled, zero-centered cycles.
    # For this symmetry, the least-squares fit_circle_xy solution is exactly
    # cx = cy = 0 and radius = sqrt(mean(x**2 + y**2)); use that equivalent
    # closed form inside the optimizer to avoid thousands of small lstsq calls.
    radius = float(np.sqrt(np.mean(radial**2)))
    circularity_cv = float(np.std(radial) / (radius + EPS))
    return float(radius), circularity_cv


def _resample_xy_normalized_time(xy: np.ndarray, n_points: int = SHARED_TEMPLATE_N_POINTS) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 2:
        return np.full((int(n_points), 2), np.nan, dtype=float)
    src = np.linspace(0.0, 1.0, xy.shape[0])
    dst = np.linspace(0.0, 1.0, int(n_points))
    return np.column_stack([np.interp(dst, src, xy[:, 0]), np.interp(dst, src, xy[:, 1])])


def _build_hopf_shared_template(projected_segments: Sequence[Tuple[int, np.ndarray]], target_radius: np.ndarray, target_duration: np.ndarray) -> Dict:
    target_radius = np.asarray(target_radius, dtype=float)
    target_duration = np.asarray(target_duration, dtype=float)
    center_r = float(np.nanmedian(target_radius))
    center_t = float(np.nanmedian(target_duration))
    condition_ids = [int(ci) for ci, _ in projected_segments]
    central_ci = min(condition_ids, key=lambda ci: abs(float(target_radius[ci]) - center_r) + abs(float(target_duration[ci]) - center_t))
    condition_templates = [
        _resample_xy_normalized_time(xy, n_points=SHARED_TEMPLATE_N_POINTS)
        for _, xy in projected_segments
        if np.asarray(xy).shape[0] >= 3
    ]
    if not condition_templates:
        template = np.full((SHARED_TEMPLATE_N_POINTS, 2), np.nan, dtype=float)
    else:
        template = np.nanmean(np.asarray(condition_templates, dtype=float), axis=0)
    step = np.linalg.norm(np.diff(template, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(step)]
    total = float(arc[-1])
    progress = np.linspace(0.0, 2.0 * np.pi, template.shape[0]) if not np.isfinite(total) or total < EPS else 2.0 * np.pi * arc / total
    return {
        "condition_index": int(central_ci),
        "target_radius": float(target_radius[central_ci]),
        "target_duration": float(target_duration[central_ci]),
        "source": "condition_mean_average",
        "template": template,
        "progress": progress,
        "arc_length": total,
        "n_conditions": int(len(condition_templates)),
        "n_trials": int(len(condition_templates)),
        "condition_weighting": "equal_weight_per_condition_mean",
    }


def _phase_velocity_with_shared_template_duration(xy_points: np.ndarray, template_info: Dict, duration_s: float) -> float:
    xy_points = np.asarray(xy_points, dtype=float)
    if xy_points.shape[0] < 3 or not np.isfinite(duration_s) or duration_s <= 0:
        return float("nan")
    dt_s = float(duration_s) / float(xy_points.shape[0])
    return float(
        _shared_template_progress_details(xy_points, template_info, dt_s)[
            "fit_velocity"
        ]
    )


def _duration_from_velocity(velocity: float) -> float:
    return float(2.0 * np.pi / max(float(velocity), EPS)) if np.isfinite(velocity) else float("nan")


def _hopf_main_pca_recurrent_features(scalar: Dict[str, np.ndarray]) -> Dict:
    rho = np.asarray(scalar["rho"], dtype=float)
    plane_angle = np.asarray(scalar["plane_angle"], dtype=float)
    native_duration = np.asarray(scalar["progress_duration"], dtype=float)
    target_radius = np.asarray(scalar["target_radius"], dtype=float)
    target_duration = np.asarray(scalar["target_duration"], dtype=float)

    trajectories = [
        _latent_trajectory(float(rho_i), float(angle_i), n=N_CIRCULARITY_SAMPLES).T
        for rho_i, angle_i in zip(rho, plane_angle)
    ]
    all_points = np.vstack(trajectories)
    mean, basis = pca_basis(all_points, n_components=2)
    centered = all_points - np.nanmean(all_points, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2 / max(centered.shape[0] - 1, 1)
    explained_variance_ratio = variance / max(float(np.nansum(variance)), EPS)

    projected_segments = [(ci, project_pca(traj, mean, basis)) for ci, traj in enumerate(trajectories)]
    template_info = _build_hopf_shared_template(projected_segments, target_radius, target_duration)
    recurrent_excursion = []
    recurrent_duration = []
    for ci, xy in projected_segments:
        displacement = np.linalg.norm(xy - xy[0:1], axis=1)
        recurrent_excursion.append(float(np.nanmax(displacement)))
        velocity = _phase_velocity_with_shared_template_duration(xy, template_info, float(native_duration[ci]))
        recurrent_duration.append(_duration_from_velocity(velocity))
    return {
        "recurrent_excursion": np.asarray(recurrent_excursion, dtype=float),
        "progress_duration": np.asarray(recurrent_duration, dtype=float),
        "pca_mean": mean,
        "pca_basis": basis,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance": np.cumsum(explained_variance_ratio),
        "shared_template": template_info,
    }


def _scalar_features(params: Sequence[float], grid: np.ndarray = INPUT_GRID) -> Dict[str, np.ndarray]:
    p = _unpack(params)
    r = np.asarray(grid[:, 0], dtype=float)
    t = np.asarray(grid[:, 1], dtype=float)
    target_radius = _positive_target_from_level(r)
    target_duration = _positive_target_from_level(t)
    alpha = np.maximum(p["alpha0"] + p["amp_R"] * r + p["amp_T"] * t, 0.05)
    progress_speed = np.maximum(p["omega0"] + p["omega_R"] * r + p["omega_T"] * t, 0.25)
    rho = np.sqrt(alpha)
    progress_duration = 2.0 * np.pi / progress_speed
    plane_angle = np.clip(p["plane_angle0"] + p["plane_angle_R"] * r + p["plane_angle_T"] * t, 0.02, 1.45)
    readout = _readout_matrix(p)
    recurrent_excursion = np.asarray(
        [_recurrent_excursion_for_condition(float(rho_i), float(angle_i)) for rho_i, angle_i in zip(rho, plane_angle)],
        dtype=float,
    )
    metrics = [_output_metrics_for_condition(float(rho_i), float(angle_i), readout) for rho_i, angle_i in zip(rho, plane_angle)]
    output_radius = np.asarray([item[0] for item in metrics], dtype=float)
    output_circularity_cv = np.asarray([item[1] for item in metrics], dtype=float)
    plane_readout_angle_deg = np.asarray([_plane_readout_angle_deg(float(angle_i), readout) for angle_i in plane_angle], dtype=float)
    plane_readout_signed_offset_deg = np.asarray(
        _signed_plane_readout_offset_deg(plane_angle, readout), dtype=float
    )
    readout_shared_axis_error_deg = _readout_shared_axis_error_deg(readout)
    output_duration = progress_duration.copy()
    return {
        "target_radius_level": r,
        "target_duration_level": t,
        "target_radius": target_radius,
        "target_duration": target_duration,
        "alpha": alpha,
        "rho": rho,
        "progress_speed": progress_speed,
        "progress_duration": progress_duration,
        "plane_angle": plane_angle,
        "plane_angle_deg": np.degrees(plane_angle),
        "native_recurrent_excursion": recurrent_excursion,
        "native_progress_duration": progress_duration.copy(),
        "recurrent_excursion": recurrent_excursion,
        "plane_readout_angle_deg": plane_readout_angle_deg,
        "plane_readout_signed_offset_deg": plane_readout_signed_offset_deg,
        "readout_shared_axis_error_deg": readout_shared_axis_error_deg,
        "output_circularity_cv": output_circularity_cv,
        "output_radius": output_radius,
        "output_duration": output_duration,
    }


def _angle_slope_deg_per_normalized_input(values: np.ndarray, x: np.ndarray) -> float:
    x_norm = fixed_range_normalize(x, x)
    values_deg = np.asarray(values, dtype=float)
    valid = np.isfinite(x_norm + values_deg)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    design = np.column_stack([np.ones(int(valid.sum())), x_norm[valid]])
    beta, *_ = np.linalg.lstsq(design, values_deg[valid], rcond=None)
    return float(beta[1])


def _simulate_features(params: Sequence[float], grid: np.ndarray = INPUT_GRID) -> Dict[str, np.ndarray]:
    scalar = _scalar_features(params, grid=grid)
    recurrent_features = _hopf_main_pca_recurrent_features(scalar)
    r = scalar["target_radius_level"]
    t = scalar["target_duration_level"]
    target_radius = scalar["target_radius"]
    target_duration = scalar["target_duration"]
    recurrent_excursion = recurrent_features["recurrent_excursion"]
    progress_duration = recurrent_features["progress_duration"]
    output_radius = scalar["output_radius"]
    output_duration = scalar["output_duration"]
    input_to_recurrent = _fit_matrix(
        target_radius,
        target_duration,
        recurrent_excursion,
        progress_duration,
        x1_ref=target_radius,
        x2_ref=target_duration,
        y1_ref=recurrent_excursion,
        y2_ref=progress_duration,
    )
    recurrent_to_output = _fit_matrix(
        recurrent_excursion,
        progress_duration,
        output_radius,
        output_duration,
        x1_ref=recurrent_excursion,
        x2_ref=progress_duration,
        y1_ref=target_radius,
        y2_ref=target_duration,
    )
    behavior = _fit_matrix(
        target_radius,
        target_duration,
        output_radius,
        output_duration,
        x1_ref=target_radius,
        x2_ref=target_duration,
        y1_ref=target_radius,
        y2_ref=target_duration,
    )
    composed = recurrent_to_output @ input_to_recurrent
    radius_mask = np.isclose(t, 0.0)
    duration_mask = np.isclose(r, 0.0)
    angle_radius_slope = _angle_slope_deg_per_normalized_input(
        scalar["plane_readout_angle_deg"][radius_mask], r[radius_mask]
    )
    angle_duration_slope = _angle_slope_deg_per_normalized_input(
        scalar["plane_readout_angle_deg"][duration_mask], t[duration_mask]
    )
    scalar.update(
        {
            "recurrent_feature_mode": HOPF_RECURRENT_FEATURE_MODE,
            "native_recurrent_excursion": scalar["native_recurrent_excursion"],
            "native_progress_duration": scalar["native_progress_duration"],
            "input_to_recurrent": input_to_recurrent,
            "recurrent_to_output": recurrent_to_output,
            "behavior": behavior,
            "composed": composed,
            "angle_radius_slope": angle_radius_slope,
            "angle_duration_slope": angle_duration_slope,
            "recurrent_excursion": recurrent_excursion,
            "progress_duration": progress_duration,
            "recurrent_pca_mean": recurrent_features["pca_mean"],
            "recurrent_pca_basis": recurrent_features["pca_basis"],
            "recurrent_pca_explained_variance_ratio": recurrent_features["explained_variance_ratio"],
            "recurrent_pca_cumulative_explained_variance": recurrent_features["cumulative_explained_variance"],
            "recurrent_shared_template": recurrent_features["shared_template"],
        }
    )
    return scalar


def _angle_slope_constraint_residuals(model: Dict[str, np.ndarray]) -> np.ndarray:
    radius_violation = max(0.0, float(model["angle_radius_slope"]) + ANGLE_SLOPE_MARGIN_DEG)
    duration_violation = max(0.0, ANGLE_SLOPE_MARGIN_DEG - float(model["angle_duration_slope"]))
    return np.sqrt(ANGLE_SLOPE_CONSTRAINT_WEIGHT) * np.asarray(
        [radius_violation, duration_violation], dtype=float
    )


def _same_side_plane_constraint_residual(model: Dict[str, np.ndarray]) -> np.ndarray:
    minimum_signed_offset = float(np.nanmin(model["plane_readout_signed_offset_deg"]))
    violation = max(0.0, PLANE_SIDE_MARGIN_DEG - minimum_signed_offset)
    return np.sqrt(PLANE_SIDE_CONSTRAINT_WEIGHT) * np.asarray([violation], dtype=float)


def _readout_shared_axis_constraint_residual(model: Dict[str, np.ndarray]) -> np.ndarray:
    violation = max(0.0, float(model["readout_shared_axis_error_deg"]) - READOUT_SHARED_AXIS_MARGIN_DEG)
    return np.sqrt(READOUT_SHARED_AXIS_CONSTRAINT_WEIGHT) * np.asarray([violation], dtype=float)


def _objective(params: Sequence[float], rnn: Dict[str, np.ndarray]) -> np.ndarray:
    model = _simulate_features(params)
    residuals = []
    residuals.extend(MATRIX_LOSS_WEIGHT * (model["behavior"] - rnn["behavior"]).ravel())
    residuals.extend(MATRIX_LOSS_WEIGHT * (model["input_to_recurrent"] - rnn["input_to_recurrent"]).ravel())
    residuals.extend(MATRIX_LOSS_WEIGHT * (model["recurrent_to_output"] - rnn["recurrent_to_output"]).ravel())
    residuals.extend(_angle_slope_constraint_residuals(model))
    residuals.extend(_same_side_plane_constraint_residual(model))
    residuals.extend(_readout_shared_axis_constraint_residual(model))
    residuals.extend(CIRCULARITY_LOSS_WEIGHT * model["output_circularity_cv"] / np.sqrt(model["output_circularity_cv"].size))
    return np.asarray(residuals, dtype=float)


def _fit_parameters(rnn: Dict[str, np.ndarray], initial_params: Sequence[float] | None = None, max_nfev: int = 30000) -> Dict:
    if initial_params is None:
        x0 = DEFAULT_PARAMS.copy()
    else:
        x0 = np.asarray(initial_params, dtype=float)
        x0 = np.clip(x0, LOWER_BOUNDS + 1e-6, UPPER_BOUNDS - 1e-6)

    fit = least_squares(
        lambda params: _objective(params, rnn),
        x0,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        max_nfev=max_nfev,
        x_scale="jac",
        xtol=HOPF_FIT_TOL,
        ftol=HOPF_FIT_TOL,
        gtol=HOPF_FIT_TOL,
    )
    return {
        "params": np.asarray(fit.x, dtype=float),
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "message": str(fit.message),
        "nfev": int(fit.nfev),
        "status": int(fit.status),
        "optimality": float(fit.optimality),
    }


def _candidate_initializations(
    warm_start: Sequence[float] | None = None,
    seeded_start: Sequence[float] | None = None,
) -> list:
    candidates = []
    if seeded_start is not None:
        seeded = np.asarray(seeded_start, dtype=float).copy()
        candidates.append((f"seed-{FIG8_FIT_SEED}-start", seeded, HOPF_REP_FIT_MAX_NFEV))
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=float).copy()
        candidates.append(("warm-start", warm, HOPF_REP_FIT_MAX_NFEV))
    candidates.append(("default", DEFAULT_PARAMS, HOPF_REP_FIT_MAX_NFEV))
    alt_a = DEFAULT_PARAMS.copy()
    alt_a[[6, 7, 8]] = [0.85, -0.35, 0.35]
    alt_b = DEFAULT_PARAMS.copy()
    alt_b[[6, 7, 8]] = [0.45, -0.08, 0.55]
    alt_c = DEFAULT_PARAMS.copy()
    alt_c[[6, 7, 8]] = [0.75, 0.35, -0.35]
    candidates.extend(
        [
            ("fallback-angle-a", alt_a, HOPF_REP_FALLBACK_MAX_NFEV),
            ("fallback-angle-b", alt_b, HOPF_REP_FALLBACK_MAX_NFEV),
            ("fallback-opposite-angle-sign", alt_c, HOPF_REP_FALLBACK_MAX_NFEV),
        ]
    )
    seen = set()
    unique = []
    for name, params, max_nfev in candidates:
        key = tuple(np.round(np.clip(params, LOWER_BOUNDS, UPPER_BOUNDS), 8))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, params, max_nfev))
    return unique


def _fit_single_rep_with_candidates(
    rnn_target: Dict[str, np.ndarray],
    warm_start: Sequence[float] | None,
    seeded_start: Sequence[float],
    rep_id: int,
) -> Dict:
    best = None
    attempts = []
    for candidate_name, candidate_params, max_nfev in _candidate_initializations(warm_start, seeded_start):
        start = time.perf_counter()
        print(f"      trying {candidate_name} (max_nfev={max_nfev})...", flush=True)
        fit = _fit_parameters(rnn_target, initial_params=candidate_params, max_nfev=max_nfev)
        elapsed = time.perf_counter() - start
        attempts.append(
            {
                "candidate": candidate_name,
                "cost": fit["cost"],
                "success": fit["success"],
                "nfev": fit["nfev"],
                "time_s": elapsed,
                "status": fit["status"],
                "message": fit["message"],
            }
        )
        print(
            f"      {candidate_name}: cost={fit['cost']:.6g}, "
            f"success={fit['success']}, nfev={fit['nfev']}, time={elapsed:.1f}s",
            flush=True,
        )
        if best is None or fit["cost"] < best["cost"]:
            best = fit
        if fit["cost"] <= HOPF_REP_ACCEPTABLE_COST:
            break
    best["attempts"] = attempts
    best["selected_candidate"] = min(attempts, key=lambda item: item["cost"])["candidate"]
    if best["cost"] > HOPF_REP_ACCEPTABLE_COST:
        print(f"      Rep {rep_id}: best cost remains above {HOPF_REP_ACCEPTABLE_COST:.3f}; keeping lowest-cost fit.", flush=True)
    return best


def _assemble_rep_model(target: Dict, fit: Dict) -> Dict:
    """Rebuild all plotting summaries from one fitted parameter vector."""
    params = np.asarray(fit["params"], dtype=float)
    model = _simulate_features(params)
    return {
        "rep": int(target["rep"]),
        "fit": {**fit, "params": params},
        "params": params,
        "model_behavior": model["behavior"],
        "model_input_to_recurrent": model["input_to_recurrent"],
        "model_recurrent_to_output": model["recurrent_to_output"],
        "rnn_input_to_recurrent": np.asarray(target["input_to_recurrent"], dtype=float),
        "rnn_recurrent_to_output": np.asarray(target["recurrent_to_output"], dtype=float),
        "angle_radius_slope": float(model["angle_radius_slope"]),
        "angle_duration_slope": float(model["angle_duration_slope"]),
        "minimum_signed_plane_readout_offset_deg": float(
            np.nanmin(model["plane_readout_signed_offset_deg"])
        ),
        "readout_shared_axis_error_deg": float(model["readout_shared_axis_error_deg"]),
        "rnn_behavior": np.asarray(target["behavior"], dtype=float),
        "rnn_angle_radius_slope": float(target["angle_targets"]["radius_slope"]),
        "rnn_angle_duration_slope": float(target["angle_targets"]["duration_slope"]),
    }


def _fit_rep_models(
    rep_targets: Sequence[Dict],
    initial_params: Sequence[float] | None = None,
    cached_models: Sequence[Dict] | None = None,
    checkpoint_callback=None,
) -> list:
    _set_fit_seed()
    fit_rng = np.random.RandomState(FIG8_FIT_SEED)
    rep_models = []
    total = len(rep_targets)
    if total == 0:
        print("No RNN rep targets found for Figure 8 rotated-Hopf per-repetition fits.", flush=True)
        return rep_models
    if initial_params is None:
        print(f"Fitting the 3D rotated Hopf model for {total} RNN repetitions: each repetition starts from its seed-controlled initialization, with sequential warm-start and fixed candidates as fallbacks.", flush=True)
    else:
        print(f"Fitting the 3D rotated Hopf model for {total} RNN repetitions from the provided initialization, then warm-starting subsequent repetitions.", flush=True)
    cached_by_rep = {int(item["rep"]): item for item in (cached_models or [])}
    seeded_starts = [
        LOWER_BOUNDS + fit_rng.random_sample(len(PARAM_NAMES)) * (UPPER_BOUNDS - LOWER_BOUNDS)
        for _ in rep_targets
    ]
    start_all = time.perf_counter()
    warm_start = initial_params
    for index, (target, seeded_start) in enumerate(zip(rep_targets, seeded_starts), start=1):
        rep_id = int(target["rep"])
        if rep_id in cached_by_rep:
            cached = cached_by_rep[rep_id]
            rep_models.append(cached)
            warm_start = np.asarray(cached["params"], dtype=float)
            print(
                f"  [{index:02d}/{total:02d}] Rep {rep_id}: loaded fitted parameters from checkpoint cache.",
                flush=True,
            )
            continue
        start = time.perf_counter()
        print(f"  [{index:02d}/{total:02d}] Rep {rep_id}: fitting behavior + mechanism targets...", flush=True)
        rnn_target = {
            "behavior": np.asarray(target["behavior"], dtype=float),
            "input_to_recurrent": np.asarray(target["input_to_recurrent"], dtype=float),
            "recurrent_to_output": np.asarray(target["recurrent_to_output"], dtype=float),
        }
        fit = _fit_single_rep_with_candidates(rnn_target, warm_start, seeded_start, rep_id)
        warm_start = fit["params"]
        elapsed = time.perf_counter() - start
        print(
            f"  [{index:02d}/{total:02d}] Rep {rep_id}: done "
            f"cost={fit['cost']:.6g}, candidate={fit.get('selected_candidate', 'unknown')}, "
            f"success={fit['success']}, time={elapsed:.1f}s",
            flush=True,
        )
        rep_models.append(_assemble_rep_model(target, fit))
        if checkpoint_callback is not None:
            checkpoint_callback(rep_models, total)
    print(f"Finished Figure 8 rotated-Hopf per-repetition fits in {time.perf_counter() - start_all:.1f}s.", flush=True)
    return rep_models


def _representative_rep_model(rep_models: Sequence[Dict], preferred_rep: int | None) -> Dict:
    if not rep_models:
        raise RuntimeError("No per-repetition rotated-Hopf fits are available for Figure 8.")
    if preferred_rep is not None:
        for item in rep_models:
            if int(item["rep"]) == int(preferred_rep):
                return item
    return min(rep_models, key=lambda item: float(item["fit"]["cost"]))


def _max_angle_modulation_rep_model(rep_models: Sequence[Dict]) -> Dict:
    if not rep_models:
        raise RuntimeError("No per-repetition rotated-Hopf fits are available for Figure 8 visualization.")
    return max(
        rep_models,
        key=lambda item: float(np.hypot(item["angle_radius_slope"], item["angle_duration_slope"])),
    )


def _typical_angle_modulation_rep_model(rep_models: Sequence[Dict]) -> Dict:
    if not rep_models:
        raise RuntimeError("No per-repetition rotated-Hopf fits are available for Figure 8 visualization.")
    radius_slopes = np.asarray([item["angle_radius_slope"] for item in rep_models], dtype=float)
    duration_slopes = np.asarray([item["angle_duration_slope"] for item in rep_models], dtype=float)
    center = np.asarray([np.nanmean(radius_slopes), np.nanmean(duration_slopes)], dtype=float)
    return min(
        rep_models,
        key=lambda item: float(
            np.hypot(
                float(item["angle_radius_slope"]) - center[0],
                float(item["angle_duration_slope"]) - center[1],
            )
        ),
    )


def _simulate_trajectory(params: Sequence[float], r_level: float, t_level: float, n: int = 240) -> Dict[str, np.ndarray]:
    p = _unpack(params)
    scalar = _scalar_features(params, grid=np.asarray([[r_level, t_level]], dtype=float))
    target_radius = float(scalar["target_radius"][0])
    target_duration = float(scalar["target_duration"][0])
    rho = float(scalar["rho"][0])
    progress_duration = float(scalar["progress_duration"][0])
    progress_speed = float(scalar["progress_speed"][0])
    plane_angle = float(scalar["plane_angle"][0])
    recurrent_excursion = float(scalar["recurrent_excursion"][0])
    latent = _latent_trajectory(rho, plane_angle, n=n)
    readout = _readout_matrix(p)
    output = readout @ latent
    out_cx, out_cy, out_radius = fit_circle_xy(output[0], output[1])
    radial = np.sqrt((output[0] - out_cx) ** 2 + (output[1] - out_cy) ** 2)
    output_circularity_cv = float(np.std(radial) / (out_radius + EPS))
    return {
        "rho": rho,
        "target_radius": target_radius,
        "target_duration": target_duration,
        "progress_duration": progress_duration,
        "progress_speed": progress_speed,
        "plane_angle": plane_angle,
        "plane_angle_deg": float(np.degrees(plane_angle)),
        "recurrent_excursion": recurrent_excursion,
        "plane_readout_angle_deg": _plane_readout_angle_deg(plane_angle, readout),
        "plane_readout_signed_offset_deg": float(_signed_plane_readout_offset_deg(plane_angle, readout)),
        "latent": latent,
        "output": output,
        "output_radius_fit": float(out_radius),
        "output_circularity_cv": output_circularity_cv,
        "output_duration": progress_duration,
    }


def _matrix_payload(mat: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str]) -> Dict:
    return {"matrix": np.asarray(mat, dtype=float), "row_labels": list(row_labels), "col_labels": list(col_labels)}


def _label_panel(ax, label: str) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(-0.18, 1.08, str(label).lower(), transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="top")
    else:
        label_panel(ax, label)


def _blend(color, n: int, low: float = 0.15, high: float = 1.22) -> np.ndarray:
    base = np.asarray(color, dtype=float)
    white = np.ones(3)
    weights = np.linspace(low, high, n)
    return np.asarray([np.clip(white * (1 - w) + base * w, 0.0, 1.0) for w in weights])


def _plot_schematic(ax, params: Sequence[float]) -> Dict:
    """Show the model equations above their geometric implementation."""
    ax.axis("off")

    def draw_box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color,
        fontsize: float = 7.0,
    ) -> None:
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.010",
            transform=ax.transAxes,
            fc=color,
            ec="black",
            lw=0.8,
            alpha=0.22,
            clip_on=False,
            zorder=1,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.18,
            transform=ax.transAxes,
            zorder=2,
        )

    def draw_matrix(
        center: Tuple[float, float],
        values: np.ndarray,
        row_labels: Sequence[str],
        col_labels: Sequence[str],
        label: str,
        cell_width: float = 0.022,
        cell_height: float = 0.030,
    ) -> None:
        values = np.asarray(values, dtype=float)
        scale = max(float(np.nanmax(np.abs(values))), EPS)
        left = center[0] - 0.5 * values.shape[1] * cell_width
        bottom = center[1] - 0.5 * values.shape[0] * cell_height
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                x0 = left + col * cell_width
                y0 = bottom + (values.shape[0] - 1 - row) * cell_height
                color = cm.RdBu_r(0.5 + 0.48 * float(values[row, col]) / scale)
                ax.fill(
                    [x0, x0 + cell_width, x0 + cell_width, x0],
                    [y0, y0, y0 + cell_height, y0 + cell_height],
                    facecolor=color,
                    edgecolor="white",
                    lw=0.45,
                    transform=ax.transAxes,
                    zorder=2,
                )
        ax.plot(
            [left, left + values.shape[1] * cell_width, left + values.shape[1] * cell_width, left, left],
            [bottom, bottom, bottom + values.shape[0] * cell_height, bottom + values.shape[0] * cell_height, bottom],
            color="0.25",
            lw=0.7,
            transform=ax.transAxes,
            zorder=3,
        )
        for row, row_label in enumerate(row_labels):
            y = bottom + (values.shape[0] - row - 0.5) * cell_height
            ax.text(left - 0.006, y, row_label, ha="right", va="center", fontsize=8, transform=ax.transAxes)
        for col, col_label in enumerate(col_labels):
            x = left + (col + 0.5) * cell_width
            ax.text(x, bottom + values.shape[0] * cell_height + 0.004, col_label, ha="center", va="bottom", fontsize=8, transform=ax.transAxes)
        ax.text(center[0], bottom - 0.010, label, ha="center", va="top", fontsize=8, transform=ax.transAxes)

    # First row: the analytic four-stage model structure.
    draw_box(
        0.075, 0.77, 0.13, 0.12,
        "Input\n" + r"$(R_{\mathrm{in}},T_{\mathrm{in}})$",
        "0.78", fontsize=8.5,
    )
    draw_box(
        0.34,
        0.77,
        0.34,
        0.23,
        "Latent controls\n"
        + r"$(\rho,\omega,\theta)=f(R_{\mathrm{in}},T_{\mathrm{in}})$",
        "0.86",
        fontsize=7.3,
    )
    draw_box(
        0.66,
        0.77,
        0.25,
        0.34,
        "Rotated Hopf\n"
        + r"$u_1(t)=\rho\cos(\omega t)$"
        + "\n"
        + r"$u_2(t)=\rho\sin(\omega t)$"
        + "\n"
        + r"$u_3(t)=0$"
        + "\n"
        + r"$x(t)=Q_y(\theta)u(t)$",
        "0.90",
        fontsize=6.8,
    )
    draw_box(
        0.90,
        0.77,
        0.18,
        0.15,
        "Readout\n" + r"$y(t)=W_{\mathrm{out}}x(t)$",
        "0.78",
        fontsize=7.2,
    )
    equation_arrows = [
        ((0.145, 0.77), (0.16, 0.77)),
        ((0.51, 0.77), (0.535, 0.77)),
        ((0.785, 0.77), (0.805, 0.77)),
    ]
    for start, end in equation_arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "black", "shrinkA": 2, "shrinkB": 2},
            zorder=3,
        )

    # Second row: an intuitive visualization of the same transformations.
    ax.plot([0.04, 0.96], [0.50, 0.50], color="0.82", lw=0.8, transform=ax.transAxes, clip_on=False)
    ax.text(0.10, 0.42, "Input", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.34, 0.42, "Controls", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.64, 0.42, "Rotated dynamics", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.90, 0.42, "Readout", ha="center", va="center", transform=ax.transAxes)

    # Input control plane: a selected point jointly specifies spatial and temporal control.
    input_origin = np.asarray([0.035, 0.105])
    input_width, input_height = 0.13, 0.25
    for fraction in np.linspace(0.0, 1.0, 5):
        ax.plot(
            [input_origin[0], input_origin[0] + input_width],
            [input_origin[1] + fraction * input_height] * 2,
            color="0.88",
            lw=0.55,
            transform=ax.transAxes,
        )
        ax.plot(
            [input_origin[0] + fraction * input_width] * 2,
            [input_origin[1], input_origin[1] + input_height],
            color="0.88",
            lw=0.55,
            transform=ax.transAxes,
        )
    ax.annotate(
        "", xy=(input_origin[0] + input_width + 0.014, input_origin[1]), xytext=input_origin,
        xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.2, "color": COLORS["T"]},
    )
    ax.annotate(
        "", xy=(input_origin[0], input_origin[1] + input_height + 0.018), xytext=input_origin,
        xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.2, "color": COLORS["R"]},
    )
    r_level, t_level = 0.55, 0.35
    selected_x = input_origin[0] + 0.5 * (t_level + 1.0) * input_width
    selected_y = input_origin[1] + 0.5 * (r_level + 1.0) * input_height
    ax.plot(
        [selected_x, selected_x], [input_origin[1], selected_y], color=COLORS["T"], lw=1.0,
        ls="--", transform=ax.transAxes,
    )
    ax.plot(
        [input_origin[0], selected_x], [selected_y, selected_y], color=COLORS["R"], lw=1.0,
        ls="--", transform=ax.transAxes,
    )
    ax.scatter(
        [selected_x], [selected_y], s=34, facecolor="white", edgecolor="black", linewidth=0.9,
        transform=ax.transAxes, zorder=4,
    )
    ax.text(input_origin[0] + input_width + 0.015, input_origin[1] - 0.006, r"$T_{\mathrm{in}}$", color=COLORS["T"], ha="left", va="top", transform=ax.transAxes)
    ax.text(input_origin[0] - 0.008, input_origin[1] + input_height + 0.018, r"$R_{\mathrm{in}}$", color=COLORS["R"], ha="right", va="bottom", transform=ax.transAxes)

    p = _unpack(params)
    alpha = p["alpha0"] + p["amp_R"] * r_level + p["amp_T"] * t_level
    rho = float(np.sqrt(max(alpha, EPS)))
    progress_speed = float(p["omega0"] + p["omega_R"] * r_level + p["omega_T"] * t_level)
    plane_angle = float(p["plane_angle0"] + p["plane_angle_R"] * r_level + p["plane_angle_T"] * t_level)

    # Three latent control coordinates generated from the two-dimensional input.
    control_origin = np.asarray([0.34, 0.225])
    control_directions = [
        (np.asarray([0.080, 0.000]), COLORS["R"], r"$\rho$"),
        (np.asarray([-0.047, 0.072]), COLORS["T"], r"$\omega$"),
        (np.asarray([0.016, 0.112]), "0.28", r"$\theta$"),
    ]
    for direction, color, label in control_directions:
        end = control_origin + direction
        ax.annotate(
            "", xy=end, xytext=control_origin, xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 1.3, "color": color}, zorder=3,
        )
        ax.text(end[0], end[1], label, color=color, ha="center", va="bottom", transform=ax.transAxes)
    control_point = control_origin + np.asarray([0.020, 0.052])
    ax.plot(
        [control_origin[0], control_point[0]], [control_origin[1], control_point[1]],
        color="0.25", lw=0.8, ls="--", transform=ax.transAxes,
    )
    ax.scatter(
        [control_point[0]], [control_point[1]], s=26, facecolor="white", edgecolor="black",
        linewidth=0.8, transform=ax.transAxes, zorder=4,
    )
    ax.text(control_origin[0], control_origin[1] - 0.040, r"$(\rho,\omega,\theta)$", ha="center", va="top", transform=ax.transAxes)

    input_control_jacobian = np.asarray(
        [
            [p["amp_R"] / (2.0 * rho), p["amp_T"] / (2.0 * rho)],
            [p["omega_R"], p["omega_T"]],
            [p["plane_angle_R"], p["plane_angle_T"]],
        ],
        dtype=float,
    )
    ax.annotate(
        "", xy=(0.275, 0.255), xytext=(0.185, 0.255), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "black"},
    )
    draw_matrix(
        (0.230, 0.145), input_control_jacobian,
        [r"$\rho$", r"$\omega$", r"$\theta$"],
        [r"$R$", r"$T$"],
        r"$J_{\mathrm{in}}$",
    )
    ax.annotate(
        "", xy=(0.505, 0.255), xytext=(0.435, 0.255), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "black"},
    )

    schematic_angle_deg = 60.0
    schematic_angle = float(np.radians(schematic_angle_deg))
    phase = np.linspace(0.0, 2.0 * np.pi, 241)
    canonical_orbit = np.column_stack(
        [np.cos(phase), np.sin(phase), np.zeros_like(phase)]
    )
    latent = canonical_orbit @ _rotation_y(schematic_angle).T
    plane = _condition_plane_vertices_centered(schematic_angle, 1.15, 1.15)
    horizontal_plane = np.asarray(
        [[-1.05, -1.05, 0.0], [1.05, -1.05, 0.0], [1.05, 1.05, 0.0], [-1.05, 1.05, 0.0]],
        dtype=float,
    )
    # Isometric-like projection: the horizontal reference and the 60-degree
    # condition plane remain simultaneously visible.
    projection = np.asarray([[0.707, 0.707, 0.000], [-0.348, 0.348, 0.870]], dtype=float)
    latent_2d = latent @ projection.T
    plane_2d = plane @ projection.T
    horizontal_2d = horizontal_plane @ projection.T
    axis_2d = np.eye(3) @ projection.T
    max_extent = max(float(np.max(np.abs(np.vstack([latent_2d, plane_2d, horizontal_2d, axis_2d])))), EPS)
    osc_center = np.asarray([0.64, 0.235])
    osc_scale = 0.125 / max_extent
    latent_plot = osc_center + osc_scale * latent_2d
    plane_plot = osc_center + osc_scale * plane_2d
    horizontal_plot = osc_center + osc_scale * horizontal_2d
    ax.fill(
        horizontal_plot[:, 0], horizontal_plot[:, 1], facecolor="0.78", edgecolor="0.55",
        alpha=0.10, lw=0.7, ls="--", transform=ax.transAxes, zorder=0,
    )
    ax.plot(
        np.r_[horizontal_plot[:, 0], horizontal_plot[0, 0]],
        np.r_[horizontal_plot[:, 1], horizontal_plot[0, 1]],
        color="0.60", lw=0.7, ls="--", transform=ax.transAxes, zorder=1,
    )
    ax.fill(
        plane_plot[:, 0], plane_plot[:, 1], facecolor=COLORS["R"], edgecolor=COLORS["R"],
        alpha=0.11, lw=0.8, transform=ax.transAxes, zorder=1,
    )
    ax.plot(
        np.r_[plane_plot[:, 0], plane_plot[0, 0]],
        np.r_[plane_plot[:, 1], plane_plot[0, 1]],
        color=COLORS["R"], lw=0.8, alpha=0.75, transform=ax.transAxes, zorder=2,
    )
    ax.plot(latent_plot[:, 0], latent_plot[:, 1], color="0.18", lw=1.8, transform=ax.transAxes, zorder=3)
    axis_labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
    dynamics_axis_origin = np.asarray([0.535, 0.120])
    for direction, label in zip(axis_2d, axis_labels):
        end = dynamics_axis_origin + 0.075 * direction
        ax.annotate(
            "", xy=end, xytext=dynamics_axis_origin, xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.50"}, zorder=1,
        )
        ax.text(end[0], end[1], label, color="0.42", ha="center", va="center", transform=ax.transAxes)
    flow_arcs = [
        (5, 27, 0.008),
        (45, 67, -0.008),
        (85, 107, 0.008),
        (125, 147, -0.008),
        (165, 187, 0.008),
        (205, 227, -0.008),
    ]
    for start_index, stop_index, normal_offset in flow_arcs:
        base_curve = latent_plot[start_index : stop_index + 1]
        tangent = np.gradient(base_curve, axis=0)
        outward_normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        outward_normal /= np.maximum(
            np.linalg.norm(outward_normal, axis=1, keepdims=True), EPS
        )
        radial = base_curve - osc_center
        orientation = np.sign(np.sum(outward_normal * radial, axis=1, keepdims=True))
        orientation[orientation == 0.0] = 1.0
        outward_normal *= orientation
        flow_curve = base_curve + normal_offset * outward_normal
        ax.plot(
            flow_curve[:, 0], flow_curve[:, 1], color=COLORS["T"], lw=1.25,
            transform=ax.transAxes, zorder=5,
        )
        ax.annotate(
            "",
            xy=flow_curve[-1],
            xytext=flow_curve[-4],
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 1.25, "color": COLORS["T"]},
            zorder=6,
        )
    ax.text(0.575, 0.335, r"Flow $\omega$", color=COLORS["T"], ha="left", va="center", transform=ax.transAxes)
    label_index = 34
    ax.annotate(
        "Limit cycle",
        xy=latent_plot[label_index],
        xytext=(0.735, 0.345),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "0.25"},
        ha="left",
        va="center",
    )
    ax.text(osc_center[0], 0.073, r"$60^{\circ}$ tilt", color="0.30", ha="center", va="top", transform=ax.transAxes)

    readout_matrix = _readout_matrix(p)
    output = latent @ readout_matrix.T
    output_centered = output - np.mean(output, axis=0, keepdims=True)
    output_extent = max(float(np.max(np.abs(output_centered))), EPS)
    output_center = np.asarray([0.90, 0.235])
    output_scale = 0.092 / output_extent
    output_plot = output_center + output_scale * output_centered
    for axis_direction, color, label in (
        (np.asarray([1.0, 0.0]), "0.55", r"$y_1$"),
        (np.asarray([0.0, 1.0]), "0.55", r"$y_2$"),
    ):
        end = output_center + 0.105 * axis_direction
        ax.annotate(
            "", xy=end, xytext=output_center, xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": color}, zorder=1,
        )
        ax.text(end[0], end[1], label, color=color, ha="center", va="center", transform=ax.transAxes)
    ax.plot(output_plot[:, 0], output_plot[:, 1], color="0.18", lw=1.8, transform=ax.transAxes, zorder=3)
    radial_index = 29
    output_radial_end = output_plot[radial_index]
    ax.annotate(
        "", xy=output_radial_end, xytext=output_center, xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": COLORS["R"]}, zorder=4,
    )
    ax.text(
        *(0.54 * output_center + 0.46 * output_radial_end), r"$R_{\mathrm{out}}$",
        color=COLORS["R"], ha="center", va="bottom", transform=ax.transAxes,
    )
    output_start_index, output_stop_index = 45, 58
    ax.annotate(
        "", xy=output_plot[output_stop_index], xytext=output_plot[output_start_index], xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": COLORS["T"]}, zorder=5,
    )
    ax.text(
        output_plot[output_stop_index, 0] + 0.010, output_plot[output_stop_index, 1], r"$T_{\mathrm{out}}$",
        color=COLORS["T"], ha="left", va="center", transform=ax.transAxes,
    )

    ax.annotate(
        "", xy=(0.835, 0.255), xytext=(0.775, 0.255), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "black"},
    )
    draw_matrix(
        (0.805, 0.145), readout_matrix,
        [r"$y_1$", r"$y_2$"],
        [r"$x_1$", r"$x_2$", r"$x_3$"],
        r"$W_{\mathrm{out}}$",
        cell_width=0.020,
        cell_height=0.034,
    )

    _, _, schematic_output_radius = fit_circle_xy(output[:, 0], output[:, 1])
    ax.set_title("Rotated Hopf model")
    return {
        "model": "condition_dependent_rotated_3d_hopf_embedding",
        "displayed_parameters": None,
        "equation_row": ["input", "rho-omega-theta controls", "Hopf normal form and rotated limit cycle", "readout y"],
        "geometric_row": {
            "example_input": {"R_in": r_level, "T_in": t_level},
            "latent_controls": {
                "rho": rho,
                "progress_speed": progress_speed,
                "plane_angle_deg": float(np.degrees(plane_angle)),
            },
            "input_to_latent_control_local_jacobian": input_control_jacobian,
            "dynamics_schematic": {
                "plane_tilt_deg": schematic_angle_deg,
                "attractor": "stable Hopf limit cycle embedded and rotated in three dimensions",
                "n_displayed_flow_arrows": int(len(flow_arcs)),
            },
            "readout_matrix": readout_matrix,
            "schematic_output_radius": float(schematic_output_radius),
        },
    }


def _readout_plane_polygon(params: Sequence[float], size: float = 1.45) -> np.ndarray:
    readout = _readout_matrix(_unpack(params))
    basis_a, basis_b, _ = _readout_plane_basis(readout)
    corners = np.asarray(
        [
            -basis_a - basis_b,
            basis_a - basis_b,
            basis_a + basis_b,
            -basis_a + basis_b,
            -basis_a - basis_b,
        ],
        dtype=float,
    )
    return size * corners


def _centered_plane_vertices(common_axis: np.ndarray, side_axis: np.ndarray, common_extent: float, side_extent: float) -> np.ndarray:
    common = np.asarray(common_axis, dtype=float)
    side = np.asarray(side_axis, dtype=float)
    common = common / (np.linalg.norm(common) + EPS)
    side = side - common * float(np.dot(side, common))
    side = side / (np.linalg.norm(side) + EPS)
    return np.asarray(
        [
            -common_extent * common - side_extent * side,
            common_extent * common - side_extent * side,
            common_extent * common + side_extent * side,
            -common_extent * common + side_extent * side,
        ],
        dtype=float,
    )


def _readout_plane_vertices_centered(params: Sequence[float], common_extent: float, side_extent: float) -> Tuple[np.ndarray, Dict[str, float]]:
    readout = _readout_matrix(_unpack(params))
    _, _, readout_normal = _readout_plane_basis(readout)
    shared_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    shared_axis_in_readout = shared_axis - readout_normal * float(np.dot(shared_axis, readout_normal))
    if np.linalg.norm(shared_axis_in_readout) < EPS:
        basis_a, _, _ = _readout_plane_basis(readout)
        shared_axis_in_readout = basis_a
    side_axis = np.cross(shared_axis_in_readout, readout_normal)
    vertices = _centered_plane_vertices(shared_axis_in_readout, side_axis, common_extent, side_extent)
    shared_axis_error_deg = float(np.degrees(np.arcsin(np.clip(abs(float(np.dot(shared_axis, readout_normal))), 0.0, 1.0))))
    return vertices, {"shared_axis_to_readout_plane_error_deg": shared_axis_error_deg}


def _readout_aligned_display_transform(params: Sequence[float]) -> Tuple[np.ndarray, Dict]:
    readout = _readout_matrix(_unpack(params))
    readout_basis_a, _, readout_normal = _readout_plane_basis(readout)
    z_axis = np.asarray(readout_normal, dtype=float)
    z_axis /= np.linalg.norm(z_axis) + EPS
    pivot = int(np.argmax(np.abs(z_axis)))
    if z_axis[pivot] < 0.0:
        z_axis = -z_axis

    native_shared_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    y_axis = native_shared_axis - z_axis * float(np.dot(native_shared_axis, z_axis))
    if np.linalg.norm(y_axis) < EPS:
        y_axis = np.asarray(readout_basis_a, dtype=float)
        y_axis = y_axis - z_axis * float(np.dot(y_axis, z_axis))
    y_axis /= np.linalg.norm(y_axis) + EPS
    if float(np.dot(y_axis, native_shared_axis)) < 0.0:
        y_axis = -y_axis

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + EPS
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + EPS
    transform = np.vstack([x_axis, y_axis, z_axis])
    transformed_normal = transform @ z_axis
    return transform, {
        "coordinate_frame": "readout_aligned_orthonormal",
        "transform_native_to_display": transform,
        "display_x_axis_in_native_coordinates": x_axis,
        "display_y_axis_in_native_coordinates": y_axis,
        "display_z_axis_in_native_coordinates": z_axis,
        "readout_normal_in_display_coordinates": transformed_normal,
        "readout_plane_display_equation": "Aligned Latent Z = 0",
    }


def _condition_plane_vertices_centered(angle: float, common_extent: float, side_extent: float) -> np.ndarray:
    basis_a, basis_b, _ = _condition_plane_basis(float(angle))
    return _centered_plane_vertices(basis_b, basis_a, common_extent, side_extent)


def _plot_plane_border(ax, vertices: np.ndarray, color, lw: float = 0.9, alpha: float = 0.95, zorder: int = 3) -> None:
    vertices = np.asarray(vertices, dtype=float)
    closed = np.vstack([vertices, vertices[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, lw=lw, alpha=alpha, zorder=zorder)


def _plot_trajectories(ax, params: Sequence[float], sweep: str, output_ax=None) -> Dict:
    if sweep == "radius":
        levels = R_LEVELS
        fixed_r, fixed_t = None, 0.0
        colors_arr = _blend(COLORS["R"], len(levels))
        title = "Radius sweep\n" + r"$T_{\mathrm{in}} = 0$"
        label = r"$R_{\mathrm{in}}$"
    else:
        levels = T_LEVELS
        fixed_r, fixed_t = 0.0, None
        colors_arr = _blend(COLORS["T"], len(levels))
        title = "Duration sweep\n" + r"$R_{\mathrm{in}} = 0$"
        label = r"$T_{\mathrm{in}}$"
    payload = {"sweep": sweep, "levels": [float(x) for x in levels], "trajectories": []}
    output_inset = output_ax if output_ax is not None else ax.inset_axes([0.68, -0.03, 0.30, 0.30])
    trajectory_records = []
    common_span = []
    side_span = []
    for level, color in zip(levels, colors_arr):
        r = float(level if fixed_r is None else fixed_r)
        t = float(level if fixed_t is None else fixed_t)
        tr = _simulate_trajectory(params, r, t)
        latent = np.asarray(tr["latent"], dtype=float)
        basis_a, basis_b, _ = _condition_plane_basis(float(tr["plane_angle"]))
        side_span.append(float(np.nanmax(np.abs(basis_a @ latent))))
        common_span.append(float(np.nanmax(np.abs(basis_b @ latent))))
        trajectory_records.append((level, color, tr, latent, np.asarray(tr["output"], dtype=float)))
    plane_margin = 1.12
    common_extent = max(1.42, plane_margin * max(common_span))
    side_extent = max(1.42, plane_margin * max(side_span))
    display_transform, display_payload = _readout_aligned_display_transform(params)
    readout_vertices_native, readout_payload = _readout_plane_vertices_centered(params, common_extent, side_extent)
    readout_vertices = readout_vertices_native @ display_transform.T
    readout_vertices[:, 2] = 0.0
    ax.add_collection3d(
        Poly3DCollection(
            [readout_vertices],
            facecolors=[(0.45, 0.45, 0.45, 0.16)],
            edgecolors=[(0.25, 0.25, 0.25, 1.0)],
            linewidths=0.85,
            zorder=1,
        )
    )
    _plot_plane_border(ax, readout_vertices, "0.28", lw=0.95, alpha=1.0, zorder=3)
    shared_axis = display_transform @ np.asarray([0.0, 1.0, 0.0], dtype=float)
    ax.plot(
        [-common_extent * shared_axis[0], common_extent * shared_axis[0]],
        [-common_extent * shared_axis[1], common_extent * shared_axis[1]],
        [-common_extent * shared_axis[2], common_extent * shared_axis[2]],
        color="0.10",
        lw=1.0,
        alpha=0.80,
        zorder=5,
    )
    all_plot_points = [readout_vertices]
    for level, color, tr, latent, output in trajectory_records:
        cond_vertices_native = _condition_plane_vertices_centered(float(tr["plane_angle"]), common_extent, side_extent)
        cond_vertices = cond_vertices_native @ display_transform.T
        latent_display = display_transform @ latent
        all_plot_points.append(cond_vertices)
        all_plot_points.append(latent_display.T)
        ax.add_collection3d(
            Poly3DCollection(
                [cond_vertices],
                facecolors=[tuple(np.r_[color, 0.09])],
                edgecolors=[tuple(np.r_[color, 1.0])],
                linewidths=0.85,
                zorder=2,
            )
        )
        _plot_plane_border(ax, cond_vertices, color, lw=0.95, alpha=1.0, zorder=3)
        ax.plot(latent_display[0], latent_display[1], latent_display[2], color=color, lw=2.3, zorder=5)
        ax.scatter(
            latent_display[0, 0],
            latent_display[1, 0],
            latent_display[2, 0],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=6,
        )
        output_inset.plot(output[0], output[1], color=color, lw=1.15)
        output_inset.scatter(output[0, 0], output[1, 0], s=8, color=color, edgecolor="white", linewidth=0.2, zorder=3)
        payload["trajectories"].append(
            {
                "level": float(level),
                "target_radius": float(tr["target_radius"]),
                "target_duration": float(tr["target_duration"]),
                "rho": float(tr["rho"]),
                "recurrent_excursion": float(tr["recurrent_excursion"]),
                "progress_duration": float(tr["progress_duration"]),
                "plane_angle_deg": float(tr["plane_angle_deg"]),
                "plane_readout_angle_deg": float(tr["plane_readout_angle_deg"]),
                "plane_readout_signed_offset_deg": float(tr["plane_readout_signed_offset_deg"]),
                "output_radius_fit": float(tr["output_radius_fit"]),
                "output_circularity_cv": float(tr["output_circularity_cv"]),
                "output_duration": float(tr["output_duration"]),
            }
        )
    ax.set_title(title)
    ax.set_xlabel("Latent x")
    ax.set_ylabel("Latent y")
    ax.set_zlabel("Latent z")
    ax.view_init(elev=22, azim=58)
    all_plot_points = np.vstack(all_plot_points)
    lim = max(1.65, float(np.nanmax(np.abs(all_plot_points))) * 1.08)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_alpha(0.0)
            axis._axinfo["grid"]["linewidth"] = 0.0
        except (AttributeError, KeyError):
            pass
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    output_inset.set_title("Output", fontsize=8)
    output_inset.set_xlabel("")
    output_inset.set_ylabel("")
    output_inset.set_aspect("equal", adjustable="box")
    try:
        output_inset.set_box_aspect(1)
    except Exception:
        pass
    output_xlim = output_inset.get_xlim()
    output_ylim = output_inset.get_ylim()
    output_lim = max(abs(output_xlim[0]), abs(output_xlim[1]), abs(output_ylim[0]), abs(output_ylim[1]), 1.0)
    output_inset.set_xlim(-output_lim, output_lim)
    output_inset.set_ylim(-output_lim, output_lim)
    output_inset.tick_params(labelsize=6, length=2)
    output_inset.locator_params(axis="x", nbins=3)
    output_inset.locator_params(axis="y", nbins=3)
    for spine in output_inset.spines.values():
        spine.set_linewidth(0.6)
    inset = ax.inset_axes([0.08, 0.04, 0.42, 0.035])
    for i, color in enumerate(colors_arr):
        inset.axvspan(i / len(colors_arr), (i + 1) / len(colors_arr), color=color, lw=0)
    inset.set_xticks([0, 1])
    inset.set_xticklabels([f"{levels.min():.0f}", f"{levels.max():.0f}"], fontsize=8)
    inset.set_yticks([])
    inset.set_xlabel(label, fontsize=8, labelpad=0)
    for spine in inset.spines.values():
        spine.set_visible(False)
    payload["plane_visualization"] = {
        "readout_plane": "fitted_readout_row_space_rigidly_aligned_to_display_xy_plane",
        "condition_planes": "true_hopf_Qy_theta_planes_transformed_by_the_same_rigid_rotation",
        "shared_axis": "latent_y_axis_transformed_by_the_same_rigid_rotation",
        "readout_plane_max_abs_display_z": float(np.nanmax(np.abs(readout_vertices[:, 2]))),
        "plane_extent": {
            "common_axis_half_width": float(common_extent),
            "side_axis_half_width": float(side_extent),
            "margin_factor": float(plane_margin),
        },
        **readout_payload,
        **display_payload,
    }
    return payload


def _plot_matrix_block(ax, model: Dict[str, np.ndarray]) -> Dict:
    ax.axis("off")
    matrices = [
        ("Input-to-recurrent", model["input_to_recurrent"], [r"$R_{\mathrm{rec}}$", r"$T_{\mathrm{rec}}$"], [r"$R_{\mathrm{in}}$", r"$T_{\mathrm{in}}$"]),
        ("Recurrent-to-output", model["recurrent_to_output"], [r"$R_{\mathrm{out}}$", r"$T_{\mathrm{out}}$"], [r"$R_{\mathrm{rec}}$", r"$T_{\mathrm{rec}}$"]),
        ("Composed behavior", model["composed"], [r"$R_{\mathrm{out}}$", r"$T_{\mathrm{out}}$"], [r"$R_{\mathrm{in}}$", r"$T_{\mathrm{in}}$"]),
        ("Model behavior", model["behavior"], [r"$R_{\mathrm{out}}$", r"$T_{\mathrm{out}}$"], [r"$R_{\mathrm{in}}$", r"$T_{\mathrm{in}}$"]),
    ]
    all_values = np.concatenate([mat.ravel() for _, mat, _, _ in matrices])
    norm = colors.Normalize(vmin=float(np.nanmin(all_values)), vmax=float(np.nanmax(all_values)))
    cmap = cm.get_cmap("viridis")
    positions = [
        [0.08, 0.48, 0.34, 0.34],
        [0.58, 0.48, 0.34, 0.34],
        [0.08, 0.12, 0.34, 0.34],
        [0.58, 0.12, 0.34, 0.34],
    ]
    payload = {}
    for position, (title, mat, row_labels, col_labels) in zip(positions, matrices):
        sub = ax.inset_axes(position)
        im = sub.imshow(mat, cmap=cmap, norm=norm)
        sub.set_title(title, fontsize=9)
        sub.set_xticks(np.arange(2))
        sub.set_xticklabels(col_labels, fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
        sub.set_yticks(np.arange(2))
        sub.set_yticklabels(row_labels, fontsize=7.5)
        sub.tick_params(length=0)
        try:
            sub.set_box_aspect(1)
        except Exception:
            sub.set_aspect("equal", adjustable="box")
        for rr in range(2):
            for cc in range(2):
                sub.text(cc, rr, f"{mat[rr, cc]:.2f}", ha="center", va="center", fontsize=9, color="black")
        for spine in sub.spines.values():
            spine.set_visible(False)
        payload[title] = _matrix_payload(mat, row_labels, col_labels)
    payload["matrix_definitions"] = {
        "Composed behavior": "Recurrent-to-output @ input-to-recurrent",
        "Model behavior": "Direct fit from rotated-Hopf output metrics to target inputs",
    }
    cax = ax.inset_axes([0.18, 0.01, 0.64, 0.035])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Normalized coefficient")
    cbar.ax.locator_params(nbins=5)
    ax.set_title("Model mechanism matrices")
    return payload


def _plot_angle_trends(ax, params: Sequence[float], angle_targets: Dict[str, float]) -> Dict:
    model = _simulate_features(params)
    r = model["target_radius_level"]
    t = model["target_duration_level"]
    angle = model["plane_readout_angle_deg"]
    radius_mask = np.isclose(t, 0.0)
    duration_mask = np.isclose(r, 0.0)
    ax.plot(r[radius_mask], angle[radius_mask], color=COLORS["R"], marker="o", lw=1.4, label=r"$R_{\mathrm{in}}$ sweep")
    ax.plot(t[duration_mask], angle[duration_mask], color=COLORS["T"], marker="o", lw=1.4, label=r"$T_{\mathrm{in}}$ sweep")
    ax.axhline(0, color="0.70", lw=0.7)
    ax.set_title("Plane-angle trends")
    ax.set_xlabel("Input level")
    ax.set_ylabel("Angle (°)")
    text = (
        f"Radius slope = {model['angle_radius_slope']:.2f}\n"
        f"Duration slope = {model['angle_duration_slope']:.2f}\n"
        f"RNN targets = {angle_targets['radius_slope']:.2f}, {angle_targets['duration_slope']:.2f}"
    )
    ax.text(
        0.04,
        0.06,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.8),
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)
    clean_axis(ax)
    return {
        "radius_sweep": {"x": r[radius_mask], "angle_deg": angle[radius_mask]},
        "duration_sweep": {"x": t[duration_mask], "angle_deg": angle[duration_mask]},
        "angle_radius_slope": float(model["angle_radius_slope"]),
        "angle_duration_slope": float(model["angle_duration_slope"]),
        "target_radius_slope": float(angle_targets["radius_slope"]),
        "target_duration_slope": float(angle_targets["duration_slope"]),
    }


def _summary_stats(values: Sequence[float]) -> Dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"values": [], "n": 0, "mean": np.nan, "median": np.nan, "sem": np.nan, "iqr": [np.nan, np.nan], "range": [np.nan, np.nan]}
    return {
        "values": arr,
        "n": int(arr.size),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "sem": float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "iqr": [float(np.nanpercentile(arr, 25)), float(np.nanpercentile(arr, 75))],
        "range": [float(np.nanmin(arr)), float(np.nanmax(arr))],
    }


def _draw_distribution(ax, values: Sequence[float], pos: float, color, width: float = 0.18, zorder: int = 2) -> Dict:
    stats_payload = _summary_stats(values)
    vals = np.asarray(stats_payload["values"], dtype=float)
    if vals.size == 0:
        return stats_payload
    q1, q3 = stats_payload["iqr"]
    median = stats_payload["median"]
    ax.fill_between([pos - width / 2.0, pos + width / 2.0], [q1, q1], [q3, q3], color=color, alpha=0.22, lw=0, zorder=zorder)
    offsets = np.linspace(-width * 0.32, width * 0.32, vals.size) if vals.size > 1 else np.zeros(vals.size)
    ax.scatter(pos + offsets, vals, s=13, color=color, alpha=0.62, linewidth=0, zorder=zorder + 1)
    ax.plot([pos - width / 2.0, pos + width / 2.0], [median, median], color="black", lw=1.9, solid_capstyle="butt", zorder=zorder + 2)
    return stats_payload


def _plot_angle_slope_summary(ax, rep_models: Sequence[Dict], angle_targets: Dict[str, float]) -> Dict:
    labels = [r"$R_{\mathrm{in}}$", r"$T_{\mathrm{in}}$"]
    rnn_values = [
        [float(item["rnn_angle_radius_slope"]) for item in rep_models],
        [float(item["rnn_angle_duration_slope"]) for item in rep_models],
    ]
    hopf_values = [
        [float(item["angle_radius_slope"]) for item in rep_models],
        [float(item["angle_duration_slope"]) for item in rep_models],
    ]
    x = np.arange(len(labels), dtype=float)
    offsets = [-0.12, 0.12]
    colors_arr = ["#4C78A8", "#F58518"]
    names = ["RNN", "3D Hopf"]
    payload = {}
    for metric_idx, label in enumerate(labels):
        for rep_idx in range(min(len(rnn_values[metric_idx]), len(hopf_values[metric_idx]))):
            ax.plot(
                [x[metric_idx] + offsets[0], x[metric_idx] + offsets[1]],
                [rnn_values[metric_idx][rep_idx], hopf_values[metric_idx][rep_idx]],
                color="0.78",
                lw=0.45,
                alpha=0.55,
                zorder=1,
            )
        for name, offset, color, values in zip(names, offsets, colors_arr, [rnn_values[metric_idx], hopf_values[metric_idx]]):
            payload[f"{label}_{name}"] = _draw_distribution(ax, values, x[metric_idx] + offset, color)
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_title("Readout-plane angle slope comparison")
    ax.set_ylabel("Slope (deg per normalized input)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors_arr[0], markeredgewidth=0, label=names[0], markersize=5),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors_arr[1], markeredgewidth=0, label=names[1], markersize=5),
        ],
        frameon=False,
        fontsize=8,
        loc="best",
    )
    ax.locator_params(axis="y", nbins=5)
    clean_axis(ax)
    payload["mean_fit_targets"] = {
        "radius_slope": float(angle_targets["radius_slope"]),
        "duration_slope": float(angle_targets["duration_slope"]),
    }
    return payload


def _plot_angle_trends_across_rep(ax, rep_models: Sequence[Dict], fallback_params: Sequence[float], angle_targets: Dict[str, float]) -> Dict:
    curve_models = [_simulate_features(item["params"]) for item in rep_models]
    if not curve_models:
        curve_models = [_simulate_features(fallback_params)]

    radius_curves = []
    duration_curves = []
    radius_slopes = []
    duration_slopes = []
    for model in curve_models:
        r = np.asarray(model["target_radius_level"], dtype=float)
        t = np.asarray(model["target_duration_level"], dtype=float)
        angle = np.asarray(model["plane_readout_angle_deg"], dtype=float)
        radius_mask = np.isclose(t, 0.0)
        duration_mask = np.isclose(r, 0.0)
        radius_curves.append(angle[radius_mask])
        duration_curves.append(angle[duration_mask])
        radius_slopes.append(float(model["angle_radius_slope"]))
        duration_slopes.append(float(model["angle_duration_slope"]))
    x_radius = np.asarray(curve_models[0]["target_radius_level"], dtype=float)[np.isclose(curve_models[0]["target_duration_level"], 0.0)]
    x_duration = np.asarray(curve_models[0]["target_duration_level"], dtype=float)[np.isclose(curve_models[0]["target_radius_level"], 0.0)]
    radius_arr = np.asarray(radius_curves, dtype=float)
    duration_arr = np.asarray(duration_curves, dtype=float)

    payload = {}
    for arr, x, color, label in [
        (radius_arr, x_radius, COLORS["R"], r"$R_{\mathrm{in}}$"),
        (duration_arr, x_duration, COLORS["T"], r"$T_{\mathrm{in}}$"),
    ]:
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(max(arr.shape[0], 1)) if arr.shape[0] > 1 else np.zeros_like(mean)
        for curve in arr:
            ax.plot(x, curve, color=color, lw=0.55, alpha=0.18, zorder=1)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, mean, color=color, marker="o", ms=3.5, lw=1.45, label=label, zorder=3)
        payload[label] = {"x": x, "values": arr, "mean": mean, "sem": sem}

    ax.axhline(0, color="0.70", lw=0.7)
    ax.set_title("Plane-angle trends")
    ax.set_xlabel("Input level")
    ax.set_ylabel("Angle (°)")
    text = (
        f"n = {len(curve_models)} fits\n"
        f"R slope = {np.nanmedian(radius_slopes):.2f}\n"
        f"T slope = {np.nanmedian(duration_slopes):.2f}"
    )
    ax.text(
        0.04,
        0.06,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.8),
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)
    clean_axis(ax)
    payload["angle_radius_slope"] = _summary_stats(radius_slopes)
    payload["angle_duration_slope"] = _summary_stats(duration_slopes)
    payload["target_radius_slope"] = float(angle_targets["radius_slope"])
    payload["target_duration_slope"] = float(angle_targets["duration_slope"])
    return payload


def _plot_target_comparison(ax, human_matrices: np.ndarray, rnn_matrices: np.ndarray, model_matrices: np.ndarray) -> Dict:
    labels = [r"$\beta_{RR}$", r"$\beta_{RT}$", r"$\beta_{TR}$", r"$\beta_{TT}$"]
    matrix_sets = [np.asarray(human_matrices, dtype=float), np.asarray(rnn_matrices, dtype=float), np.asarray(model_matrices, dtype=float)]
    names = ["Human motor", "RNN", "3D Hopf"]
    colors_arr = ["0.45", "#4C78A8", "#F58518"]
    x = np.arange(len(labels), dtype=float)
    offsets = [-0.22, 0.0, 0.22]
    payload = {}
    for label_idx, label in enumerate(labels):
        for name, offset, color, matrices in zip(names, offsets, colors_arr, matrix_sets):
            values = matrices[:, label_idx // 2, label_idx % 2] if matrices.ndim == 3 and matrices.size else []
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            pos = x[label_idx] + offset
            payload[f"{label}_{name}"] = _summary_stats(values)
            if values.size == 0:
                continue
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "lw": 1.2},
                whiskerprops={"color": "black", "lw": 0.8},
                capprops={"color": "black", "lw": 0.8},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)
                patch.set_zorder(2)
            offsets_i = np.linspace(-0.035, 0.035, values.size) if values.size > 1 else np.zeros(values.size)
            ax.scatter(pos + offsets_i, values, s=4.5, color=color, edgecolor="white", linewidth=0.2, alpha=0.52, zorder=3)
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_title("Input → output")
    ax.set_ylabel("Coefficient")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgewidth=0, label=name, markersize=3)
            for name, color in zip(names, colors_arr)
        ],
        frameon=False,
        fontsize=8,
        loc="best",
    )
    ax.locator_params(axis="y", nbins=5)
    clean_axis(ax)
    return payload


def _plot_rep_matrix_comparison(
    ax,
    rnn_matrices: np.ndarray,
    model_matrices: np.ndarray,
    labels: Sequence[str],
    title: str,
) -> Dict:
    matrix_sets = [np.asarray(rnn_matrices, dtype=float), np.asarray(model_matrices, dtype=float)]
    names = ["RNN", "3D Hopf"]
    colors_arr = ["#4C78A8", "#F58518"]
    x = np.arange(len(labels), dtype=float)
    offsets = [-0.13, 0.13]
    payload = {}
    for label_idx, label in enumerate(labels):
        for name, offset, color, matrices in zip(names, offsets, colors_arr, matrix_sets):
            values = matrices[:, label_idx // 2, label_idx % 2] if matrices.ndim == 3 and matrices.size else []
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            pos = x[label_idx] + offset
            payload[f"{label}_{name}"] = _summary_stats(values)
            if values.size == 0:
                continue
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=0.23,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "lw": 1.2},
                whiskerprops={"color": "black", "lw": 0.8},
                capprops={"color": "black", "lw": 0.8},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)
                patch.set_zorder(2)
            offsets_i = np.linspace(-0.045, 0.045, values.size) if values.size > 1 else np.zeros(values.size)
            ax.scatter(pos + offsets_i, values, s=4.5, color=color, edgecolor="white", linewidth=0.2, alpha=0.52, zorder=3)
    ax.axhline(0, color="0.35", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_ylabel("Coefficient")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=60, ha="right", rotation_mode="anchor")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgewidth=0, label=name, markersize=3)
            for name, color in zip(names, colors_arr)
        ],
        frameon=False,
        fontsize=8,
        loc="best",
    )
    ax.locator_params(axis="y", nbins=5)
    clean_axis(ax)
    return payload


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _fit_code_fingerprint() -> str:
    """Hash fit-relevant functions without invalidating the cache for plot-only edits."""
    functions = [
        fixed_range_normalize,
        pca_basis,
        project_pca,
        _shared_template_progress_details,
        _unpack,
        _readout_matrix,
        _rotation_y,
        _latent_trajectory,
        _readout_plane_basis,
        _plane_readout_angle_deg,
        _readout_shared_axis_error_deg,
        _signed_plane_readout_offset_deg,
        _output_metrics_for_condition,
        _build_hopf_shared_template,
        _phase_velocity_with_shared_template_duration,
        _hopf_main_pca_recurrent_features,
        _scalar_features,
        _fit_matrix,
        _simulate_features,
        _angle_slope_constraint_residuals,
        _same_side_plane_constraint_residual,
        _readout_shared_axis_constraint_residual,
        _objective,
        _fit_parameters,
        _candidate_initializations,
    ]
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _fit_signature(rep_targets: Sequence[Dict]) -> tuple[str, Dict]:
    fit_config = {
        "cache_version": FIT_CACHE_VERSION,
        "fit_code_sha256": _fit_code_fingerprint(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "parameter_names": PARAM_NAMES,
        "default_params": DEFAULT_PARAMS,
        "lower_bounds": LOWER_BOUNDS,
        "upper_bounds": UPPER_BOUNDS,
        "input_grid": INPUT_GRID,
        "matrix_normalization_version": MATRIX_NORMALIZATION_VERSION,
        "recurrent_feature_mode": HOPF_RECURRENT_FEATURE_MODE,
        "shared_template_n_points": SHARED_TEMPLATE_N_POINTS,
        "circularity_samples": N_CIRCULARITY_SAMPLES,
        "matrix_loss_weight": MATRIX_LOSS_WEIGHT,
        "angle_slope_margin_deg": ANGLE_SLOPE_MARGIN_DEG,
        "angle_slope_constraint_weight": ANGLE_SLOPE_CONSTRAINT_WEIGHT,
        "plane_side_margin_deg": PLANE_SIDE_MARGIN_DEG,
        "plane_side_constraint_weight": PLANE_SIDE_CONSTRAINT_WEIGHT,
        "readout_shared_axis_margin_deg": READOUT_SHARED_AXIS_MARGIN_DEG,
        "readout_shared_axis_constraint_weight": READOUT_SHARED_AXIS_CONSTRAINT_WEIGHT,
        "circularity_loss_weight": CIRCULARITY_LOSS_WEIGHT,
        "fit_tolerance": HOPF_FIT_TOL,
        "primary_max_nfev": HOPF_REP_FIT_MAX_NFEV,
        "fallback_max_nfev": HOPF_REP_FALLBACK_MAX_NFEV,
        "acceptable_cost": HOPF_REP_ACCEPTABLE_COST,
        "fit_seed": FIG8_FIT_SEED,
    }
    target_payload = [
        {
            "rep": int(target["rep"]),
            "behavior": np.asarray(target["behavior"], dtype=float),
            "input_to_recurrent": np.asarray(target["input_to_recurrent"], dtype=float),
            "recurrent_to_output": np.asarray(target["recurrent_to_output"], dtype=float),
        }
        for target in rep_targets
    ]
    signature_payload = {"fit_config": fit_config, "fit_targets": target_payload}
    canonical = json.dumps(
        _json_ready(signature_payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), signature_payload


def _cache_fit_records(rep_models: Sequence[Dict]) -> list[Dict]:
    records = []
    for item in rep_models:
        fit = {key: value for key, value in item["fit"].items() if key != "params"}
        records.append(
            {
                "rep": int(item["rep"]),
                "params": np.asarray(item["params"], dtype=float),
                "fit": fit,
            }
        )
    return records


def _write_fit_cache(
    path: Path,
    signature: str,
    signature_payload: Dict,
    rep_models: Sequence[Dict],
    total_reps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "cache_version": FIT_CACHE_VERSION,
            "signature": signature,
            "signature_payload": signature_payload,
            "complete": bool(len(rep_models) == total_reps),
            "completed_reps": int(len(rep_models)),
            "total_reps": int(total_reps),
            "fits": _cache_fit_records(rep_models),
        },
    )


def _fit_signature_payload_equivalent(
    cached_payload: Dict,
    current_payload: Dict,
    atol: float = 1e-12,
) -> tuple[bool, Dict]:
    """Check scientific equivalence while allowing library-version metadata to differ."""
    cached_payload = dict(cached_payload or {})
    current_payload = _json_ready(current_payload)
    cached_config = dict(cached_payload.get("fit_config", {}))
    current_config = dict(current_payload.get("fit_config", {}))
    environment_differences = {}
    for field in ("numpy_version", "scipy_version"):
        cached_value = cached_config.pop(field, None)
        current_value = current_config.pop(field, None)
        if cached_value != current_value:
            environment_differences[field] = {
                "cached": cached_value,
                "current": current_value,
            }
    # Cache version, fit-code fingerprint, parameterization, bounds, objective
    # settings, and seed remain strict. Only environment version labels above
    # are portable when reconstructing figures from fitted parameters.
    if cached_config != current_config:
        return False, environment_differences
    cached_targets = list(cached_payload.get("fit_targets", []))
    current_targets = list(current_payload.get("fit_targets", []))
    if len(cached_targets) != len(current_targets):
        return False, environment_differences
    for cached, current in zip(cached_targets, current_targets):
        if int(cached.get("rep", -1)) != int(current.get("rep", -2)):
            return False, environment_differences
        for key in ("behavior", "input_to_recurrent", "recurrent_to_output"):
            cached_values = np.asarray(cached.get(key), dtype=float)
            current_values = np.asarray(current.get(key), dtype=float)
            if cached_values.shape != current_values.shape:
                return False, environment_differences
            if not np.allclose(cached_values, current_values, rtol=0.0, atol=atol, equal_nan=True):
                return False, environment_differences
    return True, environment_differences


def _load_fit_cache(
    path: Path,
    signature: str,
    signature_payload: Dict,
    rep_targets: Sequence[Dict],
) -> tuple[list[Dict], Dict]:
    info = {
        "path": str(path),
        "signature": signature,
        "cache_version": FIT_CACHE_VERSION,
        "valid": False,
        "complete": False,
        "completed_reps": 0,
    }
    if not path.exists():
        info["reason"] = "missing"
        return [], info
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        info["reason"] = f"unreadable: {error}"
        return [], info
    if payload.get("cache_version") != FIT_CACHE_VERSION:
        info["reason"] = "cache_version_mismatch"
        return [], info
    signature_match = "exact"
    environment_differences = {}
    if payload.get("signature") != signature:
        equivalent, environment_differences = _fit_signature_payload_equivalent(
            payload.get("signature_payload", {}), signature_payload
        )
        if equivalent:
            signature_match = (
                "portable_environment"
                if environment_differences
                else "numeric_tolerance"
            )
        else:
            info["reason"] = "fit_signature_mismatch"
            info["environment_differences"] = environment_differences
            return [], info

    target_ids = [int(target["rep"]) for target in rep_targets]
    records = list(payload.get("fits", []))
    record_ids = [int(record.get("rep", -1)) for record in records]
    if record_ids != target_ids[: len(record_ids)]:
        info["reason"] = "cached_repetitions_are_not_a_contiguous_prefix"
        return [], info
    target_by_rep = {int(target["rep"]): target for target in rep_targets}
    rep_models = []
    try:
        for record in records:
            rep_id = int(record["rep"])
            params = np.asarray(record["params"], dtype=float)
            if params.shape != (len(PARAM_NAMES),) or not np.all(np.isfinite(params)):
                raise ValueError(f"invalid parameter vector for rep {rep_id}")
            fit = dict(record["fit"])
            fit["params"] = params
            rep_models.append(_assemble_rep_model(target_by_rep[rep_id], fit))
    except Exception as error:
        info["reason"] = f"invalid_fit_record: {error}"
        return [], info

    info.update(
        {
            "valid": True,
            "complete": bool(len(rep_models) == len(rep_targets)),
            "completed_reps": int(len(rep_models)),
            "total_reps": int(len(rep_targets)),
            "reason": (
                "complete"
                if len(rep_models) == len(rep_targets) and signature_match == "exact"
                else "complete_portable_environment"
                if len(rep_models) == len(rep_targets) and signature_match == "portable_environment"
                else "complete_numeric_tolerance"
                if len(rep_models) == len(rep_targets)
                else "partial_checkpoint"
            ),
            "signature_match": signature_match,
            "environment_differences": environment_differences,
            "cached_signature": payload.get("signature"),
        }
    )
    return rep_models, info


def _load_or_fit_rep_models(
    rep_targets: Sequence[Dict],
    force_refit: bool = False,
    require_complete_cache: bool = False,
) -> tuple[list[Dict], Dict]:
    signature, signature_payload = _fit_signature(rep_targets)
    cached_models = []
    cache_info = {
        "path": str(FIT_CACHE_PATH),
        "signature": signature,
        "cache_version": FIT_CACHE_VERSION,
        "valid": False,
        "complete": False,
        "completed_reps": 0,
        "reason": "forced_refit" if force_refit else "not_checked",
    }
    if not force_refit:
        cached_models, cache_info = _load_fit_cache(
            FIT_CACHE_PATH, signature, signature_payload, rep_targets
        )
        if cache_info["complete"]:
            print(
                f"Loaded {len(cached_models)} Figure 8 rotated-Hopf fits from {FIT_CACHE_PATH}.",
                flush=True,
            )
            return cached_models, {**cache_info, "loaded_from_cache": True, "cache_written": False}
        if require_complete_cache:
            raise RuntimeError(
                f"Figure 8 --plot-only requires a complete valid fit cache; "
                f"cache status is {cache_info.get('reason', 'unknown')} at {FIT_CACHE_PATH}."
            )
        if cached_models:
            print(
                f"Resuming Figure 8 after {len(cached_models)}/{len(rep_targets)} cached repetitions.",
                flush=True,
            )
    elif require_complete_cache:
        raise ValueError("--refit and --plot-only cannot be used together.")

    def checkpoint(models: Sequence[Dict], total: int) -> None:
        _write_fit_cache(FIT_CACHE_PATH, signature, signature_payload, models, total)

    rep_models = _fit_rep_models(
        rep_targets,
        initial_params=None,
        cached_models=cached_models,
        checkpoint_callback=checkpoint,
    )
    checkpoint(rep_models, len(rep_targets))
    return rep_models, {
        "path": str(FIT_CACHE_PATH),
        "signature": signature,
        "cache_version": FIT_CACHE_VERSION,
        "valid": True,
        "complete": True,
        "completed_reps": int(len(rep_models)),
        "total_reps": int(len(rep_targets)),
        "reason": "refitted" if force_refit else "cache_completed",
        "loaded_from_cache": bool(cached_models),
        "cache_written": True,
    }


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Fit or plot the constrained Figure 8 three-dimensional rotated Hopf model with reusable per-repetition caches."
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Ignore any valid cache and refit all RNN repetitions before plotting.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fit-only",
        action="store_true",
        help="Create or complete the fit cache without regenerating the manuscript figure.",
    )
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure from a complete valid cache and fail instead of fitting if the cache is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _set_fit_seed()
    print(f"Figure 8 fit seed fixed at {FIG8_FIT_SEED}.", flush=True)
    setup_style()
    human_matrices, human_subjects = _human_subject_matrices()
    human = np.nanmean(human_matrices, axis=0)
    rnn = _load_rnn_targets()
    angle_targets = _load_angle_targets()
    rep_models, fit_cache_info = _load_or_fit_rep_models(
        rnn.get("rep_targets", []),
        force_refit=bool(args.refit),
        require_complete_cache=bool(args.plot_only),
    )
    if args.fit_only:
        print(
            f"Figure 8 fit cache ready: {fit_cache_info['completed_reps']}/"
            f"{fit_cache_info['total_reps']} repetitions at {FIT_CACHE_PATH}.",
            flush=True,
        )
        return
    representative = _typical_angle_modulation_rep_model(rep_models)
    fit = representative["fit"]
    params = representative["params"]
    model = _simulate_features(params)
    visualization = representative
    visualization_params = visualization["params"]
    print(
        f"Using rep {representative['rep']} as the reporting solution "
        f"(cost={fit['cost']:.6g}, success={fit['success']}).",
        flush=True,
    )
    print(
        f"Using angle-typical rep {visualization['rep']} for Figure 8 A-C visualization "
        f"(R slope={visualization['angle_radius_slope']:.3f}, "
        f"T slope={visualization['angle_duration_slope']:.3f}).",
        flush=True,
    )
    rnn_behavior_matrices = np.asarray([item["rnn_behavior"] for item in rep_models], dtype=float)
    hopf_behavior_matrices = np.asarray([item["model_behavior"] for item in rep_models], dtype=float)
    rnn_input_to_recurrent_matrices = np.asarray([item["rnn_input_to_recurrent"] for item in rep_models], dtype=float)
    hopf_input_to_recurrent_matrices = np.asarray([item["model_input_to_recurrent"] for item in rep_models], dtype=float)
    rnn_recurrent_to_output_matrices = np.asarray([item["rnn_recurrent_to_output"] for item in rep_models], dtype=float)
    hopf_recurrent_to_output_matrices = np.asarray([item["model_recurrent_to_output"] for item in rep_models], dtype=float)
    if rnn_behavior_matrices.size == 0:
        rnn_behavior_matrices = np.asarray([rnn["behavior"]], dtype=float)
    if hopf_behavior_matrices.size == 0:
        hopf_behavior_matrices = np.asarray([model["behavior"]], dtype=float)
    if rnn_input_to_recurrent_matrices.size == 0:
        rnn_input_to_recurrent_matrices = np.asarray([rnn["input_to_recurrent"]], dtype=float)
    if hopf_input_to_recurrent_matrices.size == 0:
        hopf_input_to_recurrent_matrices = np.asarray([model["input_to_recurrent"]], dtype=float)
    if rnn_recurrent_to_output_matrices.size == 0:
        rnn_recurrent_to_output_matrices = np.asarray([rnn["recurrent_to_output"]], dtype=float)
    if hopf_recurrent_to_output_matrices.size == 0:
        hopf_recurrent_to_output_matrices = np.asarray([model["recurrent_to_output"]], dtype=float)

    # Reserve explicit top and bottom breathing room for 3-D titles and the
    # rotated coefficient labels.  The original Illustrator-oriented canvas
    # placed both against the image boundary in a direct PNG/SVG export.
    fig = plt.figure(figsize=(19.2, 11.4), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 48, height_ratios=[1.05, 0.95],
        left=0.040, right=0.992, bottom=0.16, top=0.90,
        wspace=0.34, hspace=0.46,
    )
    ax_a = fig.add_subplot(gs[0, 0:18])
    gs_b = gs[0, 18:33].subgridspec(1, 2, width_ratios=[1.0, 0.28], wspace=0.32)
    gs_c = gs[0, 33:48].subgridspec(1, 2, width_ratios=[1.0, 0.28], wspace=0.32)
    ax_b = fig.add_subplot(gs_b[0, 0], projection="3d")
    ax_b_output = fig.add_subplot(gs_b[0, 1])
    ax_c = fig.add_subplot(gs_c[0, 0], projection="3d")
    ax_c_output = fig.add_subplot(gs_c[0, 1])
    ax_d = fig.add_subplot(gs[1, 0:12])
    ax_e = fig.add_subplot(gs[1, 12:24])
    ax_f = fig.add_subplot(gs[1, 24:36])
    ax_g = fig.add_subplot(gs[1, 36:48])
    for label, ax in zip("ABCDEFG", [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g]):
        _label_panel(ax, label)

    panel_a = _plot_schematic(ax_a, visualization_params)
    panel_b = _plot_trajectories(ax_b, visualization_params, "radius", output_ax=ax_b_output)
    panel_c = _plot_trajectories(ax_c, visualization_params, "duration", output_ax=ax_c_output)
    for panel in [panel_a, panel_b, panel_c]:
        panel["visualization_rep"] = int(visualization["rep"])
    panel_d = _plot_angle_trends_across_rep(ax_d, rep_models, params, angle_targets)
    panel_e = _plot_rep_matrix_comparison(
        ax_e,
        rnn_input_to_recurrent_matrices,
        hopf_input_to_recurrent_matrices,
        [
            r"$R_{\mathrm{in}}\rightarrow R_{\mathrm{rec}}$",
            r"$T_{\mathrm{in}}\rightarrow R_{\mathrm{rec}}$",
            r"$R_{\mathrm{in}}\rightarrow T_{\mathrm{rec}}$",
            r"$T_{\mathrm{in}}\rightarrow T_{\mathrm{rec}}$",
        ],
        "Input → recurrent",
    )
    panel_f = _plot_rep_matrix_comparison(
        ax_f,
        rnn_recurrent_to_output_matrices,
        hopf_recurrent_to_output_matrices,
        [
            r"$R_{\mathrm{rec}}\rightarrow R_{\mathrm{out}}$",
            r"$T_{\mathrm{rec}}\rightarrow R_{\mathrm{out}}$",
            r"$R_{\mathrm{rec}}\rightarrow T_{\mathrm{out}}$",
            r"$T_{\mathrm{rec}}\rightarrow T_{\mathrm{out}}$",
        ],
        "Recurrent → output",
    )
    panel_g = _plot_target_comparison(ax_g, human_matrices, rnn_behavior_matrices, hopf_behavior_matrices)

    # Keep the four lower-row axes physically square.  This explicit position
    # adjustment is compatible with Matplotlib versions that predate
    # ``Axes.set_box_aspect`` and does not alter any plotted data limits.
    figure_width, figure_height = fig.get_size_inches()
    for ax in (ax_d, ax_e, ax_f, ax_g):
        position = ax.get_position()
        side_inches = min(position.width * figure_width, position.height * figure_height)
        square_width = side_inches / figure_width
        square_height = side_inches / figure_height
        ax.set_position(
            [
                position.x0 + 0.5 * (position.width - square_width),
                position.y0 + 0.5 * (position.height - square_height),
                square_width,
                square_height,
            ]
        )

    stats_payload = {
        "description": "Constrained three-dimensional rotated Hopf model. A planar supercritical Hopf normal form supplies a stable limit cycle whose condition-dependent amplitude and angular frequency are evaluated analytically at steady state; the cycle is embedded and rotated in three-dimensional latent space before a fixed full linear readout. One model is fitted independently to each RNN repetition's behavior and mechanism matrices, with radius- and duration-dependent readout-plane angle slopes constrained only by their expected signs and a one-degree minimum magnitude. A same-side constraint keeps every fitted condition plane at least five degrees beyond the closest readout orientation, preventing unsigned-angle foldbacks. A shared-axis constraint keeps the common condition-plane rotation axis within one degree of the readout row-space so readout-plane angles remain visually interpretable, and output circularity is regularized. Exact RNN angle-slope values and human motor-stage behavior are reported only as external references. Panels A-C use the repetition whose radius/duration angle-slope pair is closest to the across-repetition mean pair, panel D summarizes the constrained readout-plane angle trends across all repetitions, panels E-F compare RNN and rotated-Hopf mechanism coefficients across repetitions, and panel G compares coefficient patterns across human motor-stage, RNN, and rotated-Hopf results without claiming numerical equality.",
        "fit_cache": fit_cache_info,
        "claim_scope": {
            "supported": "The fitted construction provides a low-dimensional sufficiency example that reproduces the dominant directional and geometric patterns under explicit matrix and geometry constraints.",
            "not_supported": "The fits are not independent predictions of the RNN mechanism, do not reproduce every weak matrix entry or the RNN angle magnitudes, and do not test transient convergence or stability outside the embedded Hopf plane.",
            "minimality": "Three dimensions provide the plane-rotation construction used here; parameter minimality is not claimed because no exhaustive lower-parameter comparison is performed.",
        },
        "behavior_comparison_scope": {
            "human": "Subject-level perceived-input to produced-output motor-stage coefficients from Figures 1 and 2.",
            "rnn": "Target-input to RNN-output coefficients.",
            "hopf": "Abstract control-input to reduced rotated-Hopf-output coefficients.",
            "claim": "Qualitative directional comparison only; the three input semantics are not treated as numerically interchangeable.",
        },
        "rnn_source": {
            "figure4_stats_file": rnn.get("source_stats_file"),
            "figure6_stats_file": angle_targets.get("source_stats_file"),
            "figure6_angle_target_panel": angle_targets.get("source_panel"),
            "figure6_angle_slope_definition": angle_targets.get("slope_definition"),
            "recurrent_timing_method": rnn.get("recurrent_timing_method"),
            "recurrent_time_metric": rnn.get("recurrent_time_metric"),
        },
        "matrix_normalization": {
            "version": MATRIX_NORMALIZATION_VERSION,
            "target_level_range": [-1.0, 1.0],
            "positive_target_mapping": {
                "radius": "R_target = target_center + target_half_range * R_level",
                "duration": "T_target = target_center + target_half_range * T_level",
                "target_center": HOPF_TARGET_CENTER,
                "target_half_range": HOPF_TARGET_HALF_RANGE,
                "target_range": [
                    HOPF_TARGET_CENTER - HOPF_TARGET_HALF_RANGE,
                    HOPF_TARGET_CENTER + HOPF_TARGET_HALF_RANGE,
                ],
            },
            "input_to_recurrent": {
                "input_radius": "positive_target_radius_range",
                "input_duration": "positive_target_duration_range",
                "recurrent_excursion": "self_range",
                "progress_duration": "self_range",
            },
            "recurrent_to_output": {
                "recurrent_excursion": "self_range",
                "progress_duration": "self_range",
                "output_radius": "positive_target_radius_range",
                "output_duration": "positive_target_duration_range",
            },
            "behavior": {
                "input_radius": "positive_target_radius_range",
                "input_duration": "positive_target_duration_range",
                "output_radius": "positive_target_radius_range",
                "output_duration": "positive_target_duration_range",
            },
        },
        "model_equations": {
            "positive_target_radius": "R_target = target_center + target_half_range * R_level",
            "positive_target_duration": "T_target = target_center + target_half_range * T_level",
            "alpha": "alpha0 + amp_R * R + amp_T * T",
            "rho": "sqrt(alpha)",
            "progress_speed": "omega0 + omega_R * R + omega_T * T",
            "progress_duration": "D_p = 2pi / progress_speed",
            "latent_base": "u(t) = [rho cos(phi), rho sin(phi), 0]",
            "plane_angle": "theta = theta0 + theta_R * R + theta_T * T",
            "embedding": "x(t) = Q_y(theta) u(t)",
            "recurrent_metrics": "For mechanism matrices, all condition trajectories x(t) are projected into one common rotated-Hopf PCA plane. E_rec = max_t ||PC12(t) - PC12(onset)||. D_p = 2pi / shared-template progress speed.",
            "fixed_readout": "y(t) = W_out x(t), with fixed full 2x3 readout matrix",
            "readout_matrix": "W_out = [[wXX, wXY, wXZ], [wYX, wYY, wYZ]]",
            "plane_readout_angle": "Acute principal angle between the condition-specific orbit plane and the row-space plane of W_out.",
            "output_metrics": "Output radius is obtained by least-squares circle fitting of the generated readout trace; output duration is 2pi / progress_speed; circularity is the standard deviation of distances from the fitted center divided by the fitted radius.",
        },
        "representative_fit": {
            "rep": int(representative["rep"]),
            "success": fit["success"],
            "cost": fit["cost"],
            "message": fit["message"],
            "selection_rule": "Choose the repetition whose radius/duration angle-slope pair has the smallest Euclidean distance to the across-repetition mean slope pair.",
        },
        "visualization_fit": {
            "rep": int(visualization["rep"]),
            "panels": ["A", "B", "C"],
            "selection_rule": "Smallest Euclidean distance from the across-repetition mean radius/duration angle-slope pair.",
            "population_mean_radius_angle_slope": float(np.nanmean([item["angle_radius_slope"] for item in rep_models])),
            "population_mean_duration_angle_slope": float(np.nanmean([item["angle_duration_slope"] for item in rep_models])),
            "distance_to_population_mean_slope_pair": float(
                np.hypot(
                    visualization["angle_radius_slope"] - np.nanmean([item["angle_radius_slope"] for item in rep_models]),
                    visualization["angle_duration_slope"] - np.nanmean([item["angle_duration_slope"] for item in rep_models]),
                )
            ),
            "radius_angle_slope": float(visualization["angle_radius_slope"]),
            "duration_angle_slope": float(visualization["angle_duration_slope"]),
            "cost": float(visualization["fit"]["cost"]),
            "parameters": _unpack(visualization_params),
            "readout_matrix": _readout_matrix(_unpack(visualization_params)),
        },
        "rep_fit_summary": {
            "n_reps": int(len(rep_models)),
            "success_count": int(sum(1 for item in rep_models if item["fit"]["success"])),
            "cost_values": np.asarray([item["fit"]["cost"] for item in rep_models], dtype=float),
            "nfev_values": np.asarray([item["fit"].get("nfev", np.nan) for item in rep_models], dtype=float),
            "selected_candidates": [str(item["fit"].get("selected_candidate", "unknown")) for item in rep_models],
            "rep_ids": [int(item["rep"]) for item in rep_models],
            "initialization": "Each repetition first uses its own seed-controlled random point within the parameter bounds. Difficult fits then try the previous repetition's solution as a warm-start followed by fixed fallback initializations.",
            "random_seed": FIG8_FIT_SEED,
            "random_seed_scope": "Python random and NumPy global RNG are reset immediately before loading targets and fitting; a dedicated NumPy RandomState generates one bounded candidate initialization per repetition.",
            "optimizer": "SciPy least_squares with an independent seed-controlled primary initialization per repetition, sequential warm-start fallback, and fixed deterministic fallbacks.",
            "plane_angle_sign_mode": PLANE_ANGLE_SIGN_MODE,
            "acceptable_cost_for_fallback": HOPF_REP_ACCEPTABLE_COST,
            "primary_max_nfev": HOPF_REP_FIT_MAX_NFEV,
            "fallback_max_nfev": HOPF_REP_FALLBACK_MAX_NFEV,
            "fit_tolerance": HOPF_FIT_TOL,
            "recurrent_feature_mode": HOPF_RECURRENT_FEATURE_MODE,
            "parameter_count": int(len(PARAM_NAMES)),
            "parameter_names": PARAM_NAMES,
        },
        "loss_config": {
            "matrix_normalization_version": MATRIX_NORMALIZATION_VERSION,
            "fit_targets": [
                "rnn_rep_behavior_matrix",
                "rnn_rep_input_to_recurrent_matrix",
                "rnn_rep_recurrent_to_output_matrix",
                "radius_angle_slope_sign_and_minimum_magnitude",
                "duration_angle_slope_sign_and_minimum_magnitude",
                "condition_planes_same_side_of_readout_orientation",
                "shared_condition_axis_in_readout_plane",
                "output_circularity_regularizer",
            ],
            "human_behavior_in_fit": False,
            "matrix_loss_weight": MATRIX_LOSS_WEIGHT,
            "angle_slope_constraint_weight": ANGLE_SLOPE_CONSTRAINT_WEIGHT,
            "angle_slope_definition": {
                "input": "Radius or duration input is fixed-range normalized to [-1, 1].",
                "angle": "Readout-plane angle remains in raw degrees; no y normalization.",
                "unit": "degrees per normalized input unit",
                "constraint": f"radius slope <= -{ANGLE_SLOPE_MARGIN_DEG:g}; duration slope >= {ANGLE_SLOPE_MARGIN_DEG:g}",
                "penalty": "Squared hinge through least-squares residuals; zero penalty once sign and minimum magnitude are satisfied.",
                "minimum_magnitude_deg": ANGLE_SLOPE_MARGIN_DEG,
                "residual_scale_deg": ANGLE_SLOPE_ERROR_SCALE_DEG,
            },
            "same_side_plane_constraint": {
                "definition": "All 25 fitted condition-plane angles must remain on the positive signed side of the closest readout orientation.",
                "constraint": f"minimum signed condition-readout offset >= {PLANE_SIDE_MARGIN_DEG:g} deg",
                "minimum_offset_deg": PLANE_SIDE_MARGIN_DEG,
                "weight": PLANE_SIDE_CONSTRAINT_WEIGHT,
                "penalty": "One squared-hinge least-squares residual based on the minimum signed offset across conditions.",
            },
            "readout_shared_axis_constraint": {
                "definition": "The native y axis shared by all condition planes must lie in the fixed readout row-space within tolerance.",
                "constraint": f"shared-axis-to-readout-plane error <= {READOUT_SHARED_AXIS_MARGIN_DEG:g} deg",
                "maximum_error_deg": READOUT_SHARED_AXIS_MARGIN_DEG,
                "weight": READOUT_SHARED_AXIS_CONSTRAINT_WEIGHT,
                "penalty": "One squared-hinge least-squares residual on the shared-axis-to-readout-plane error.",
            },
            "circularity_loss_weight": CIRCULARITY_LOSS_WEIGHT,
            "reported_but_not_fitted": ["human_motor_stage_reference_matrix", "rnn_composed_matrix", "model_composed_matrix", "exact_rnn_angle_radius_slope", "exact_rnn_angle_duration_slope"],
            "parameter_direction_constraints": {
                "amp_R": "bounded >= 0",
                "amp_T": "bounded >= 0",
                "omega_R": "bounded <= 0",
                "omega_T": "bounded <= 0",
                "plane_angle_R": f"free parameter sign; measured radius angle slope constrained to <= -{ANGLE_SLOPE_MARGIN_DEG:g} deg per normalized input",
                "plane_angle_T": f"free parameter sign; measured duration angle slope constrained to >= {ANGLE_SLOPE_MARGIN_DEG:g} deg per normalized input",
            },
        },
        "parameters": _unpack(params),
        "readout_matrix": _readout_matrix(_unpack(params)),
        "recurrent_dynamics": {
            "feature_mode": HOPF_RECURRENT_FEATURE_MODE,
            "state_space": "three_dimensional_hopf_latent_state",
            "pca_metric": "common PCA fit to all 25 rotated Hopf latent trajectories",
            "excursion_metric": "max_t ||PC12(t) - PC12(onset)|| in the common rotated-Hopf PCA plane",
            "duration_metric": "shared-template progress duration, D_p = 2pi / progress_speed_estimate",
            "recurrent_excursion": model["recurrent_excursion"],
            "progress_duration": model["progress_duration"],
            "native_recurrent_excursion": model["native_recurrent_excursion"],
            "native_progress_duration": model["native_progress_duration"],
            "pca_mean": model["recurrent_pca_mean"],
            "pca_basis": model["recurrent_pca_basis"],
            "explained_variance_ratio": model["recurrent_pca_explained_variance_ratio"],
            "cumulative_explained_variance": model["recurrent_pca_cumulative_explained_variance"],
            "shared_template": model["recurrent_shared_template"],
        },
        "output_circularity": {
            "mean_cv": float(np.nanmean(model["output_circularity_cv"])),
            "max_cv": float(np.nanmax(model["output_circularity_cv"])),
            "values": model["output_circularity_cv"],
        },
        "targets": {
            "human_motor_stage_reference_matrix": human,
            "human_motor_stage_reference_subject_matrices": human_matrices,
            "human_subjects": human_subjects,
            "rnn_behavior_matrix": rnn["behavior"],
            "rnn_behavior_rep_matrices": rnn_behavior_matrices,
            "rnn_input_to_recurrent_matrix": rnn["input_to_recurrent"],
            "rnn_input_to_recurrent_rep_matrices": rnn_input_to_recurrent_matrices,
            "rnn_recurrent_to_output_matrix": rnn["recurrent_to_output"],
            "rnn_recurrent_to_output_rep_matrices": rnn_recurrent_to_output_matrices,
            "rnn_composed_matrix": rnn["composed"],
            "hopf_behavior_rep_matrices": hopf_behavior_matrices,
            "hopf_input_to_recurrent_rep_matrices": hopf_input_to_recurrent_matrices,
            "hopf_recurrent_to_output_rep_matrices": hopf_recurrent_to_output_matrices,
            "rnn_angle_radius_slope": float(angle_targets["radius_slope"]),
            "rnn_angle_duration_slope": float(angle_targets["duration_slope"]),
        },
        "model_matrices": {
            "input_to_recurrent": _matrix_payload(model["input_to_recurrent"], [r"$R_{\mathrm{rec}}$", r"$T_{\mathrm{rec}}$"], [r"$R_{\mathrm{in}}$", r"$T_{\mathrm{in}}$"]),
            "recurrent_to_output": _matrix_payload(model["recurrent_to_output"], [r"$R_{\mathrm{out}}$", r"$T_{\mathrm{out}}$"], [r"$R_{\mathrm{rec}}$", r"$T_{\mathrm{rec}}$"]),
            "behavior": _matrix_payload(model["behavior"], [r"$R_{\mathrm{out}}$", r"$T_{\mathrm{out}}$"], [r"$R_{\mathrm{in}}$", r"$T_{\mathrm{in}}$"]),
            "composed": _matrix_payload(model["composed"], [r"$R_{\mathrm{out}}$", r"$T_{\mathrm{out}}$"], [r"$R_{\mathrm{in}}$", r"$T_{\mathrm{in}}$"]),
        },
        "angle_trends": {
            "angle_radius_slope": float(model["angle_radius_slope"]),
            "angle_duration_slope": float(model["angle_duration_slope"]),
            "minimum_signed_plane_readout_offset_deg": float(
                np.nanmin(model["plane_readout_signed_offset_deg"])
            ),
            "signed_plane_readout_offset_deg": model["plane_readout_signed_offset_deg"],
            "readout_shared_axis_error_deg": float(model["readout_shared_axis_error_deg"]),
        },
        "panels": {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d, "E": panel_e, "F": panel_f, "G": panel_g},
    }
    write_json(FIG_DIR / OUTPUT_STATS, _json_ready(stats_payload))
    path = finish_figure(fig, OUTPUT_FIGURE)
    print(f"Saved {path}")
    print(f"Saved {FIG_DIR / OUTPUT_STATS}")


if __name__ == "__main__":
    main()
