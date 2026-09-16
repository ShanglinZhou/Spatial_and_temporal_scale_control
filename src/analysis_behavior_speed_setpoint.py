"""Shared analyses for the behavioral preferred-speed figures.

The analysis is intentionally framed as a model comparison.  A preferred
speed is treated as one regularizer of the produced movement, rather than as
an assumption that must explain the full behavioral control matrix by itself.

All fitted control models use perceived radius and duration as their inputs
and produced radius and duration as their outputs.  Model comparisons use the
same standardized log-output loss and subject-specific held-out predictions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import optimize, stats

from analysis_all_common import (
    HUMAN_DATA_DIR,
    human_draw_xy_from_row,
    human_response_from_row,
    human_subject_id,
    iter_human_subject_files,
    load_human_condition_means,
    load_human_subject_rows,
)
from analysis_behavior_common import build_matched_rows, normalize


R_C, R_H, T_C, T_H = 87.5, 48.5, 1.9, 1.2
LOG_SCALE = np.array([math.log(136.0 / 39.0), math.log(3.1 / 0.7)], dtype=float)
MODEL_ORDER = ["M0", "MR", "MRT", "MV", "MRV", "M2", "FULL"]
MODEL_LABELS = {
    "M0": "Identity",
    "MR": "Radius\nprior",
    "MRT": "Radius + time\npriors",
    "MV": "Speed\nprior",
    "MRV": "Radius + speed\npriors",
    "M2": "Old effort\nmodel",
    "FULL": "Full affine\nbenchmark",
}
MODEL_COLORS = {
    "M0": (0.76, 0.76, 0.76),
    "MR": (0.70, 0.55, 0.95),
    "MRT": (0.51, 0.40, 0.72),
    "MV": (0.00, 0.784, 0.784),
    "MRV": (0.18, 0.34, 0.48),
    "M2": (0.48, 0.48, 0.48),
    "FULL": (0.15, 0.15, 0.15),
}

OFFSET_MODEL_ORDER = ["O", "R", "T", "V", "RT", "RV", "RVrho", "TV", "RTV", "FULL"]
OFFSET_MODEL_LABELS = {
    "O": "Offset only",
    "R": "R setpoint",
    "T": "T setpoint",
    "V": "V setpoint",
    "RT": "R + T",
    "RV": "R + V",
    "RVrho": "R + V + allocation",
    "TV": "T + V",
    "RTV": "R + T + V",
    "FULL": "Full affine",
}
OFFSET_MODEL_COLORS = {
    "O": (0.76, 0.76, 0.76),
    "R": (0.70, 0.55, 0.95),
    "T": (0.00, 0.784, 0.784),
    "V": (0.35, 0.35, 0.35),
    "RT": (0.40, 0.72, 0.78),
    "RV": (0.31, 0.25, 0.48),
    "RVrho": (0.57, 0.43, 0.78),
    "TV": (0.15, 0.52, 0.52),
    "RTV": (0.12, 0.23, 0.34),
    "FULL": (0.05, 0.05, 0.05),
}
RAW_SCALE = np.array([R_H, T_H], dtype=float)
LOG_FIDELITY_SCALE = np.array([math.log(136.0 / 39.0), math.log(3.1 / 0.7)], dtype=float)
LOG_SPEED_SCALE = float(np.sqrt(np.sum(LOG_FIDELITY_SCALE ** 2)))


@dataclass
class SubjectData:
    subject: str
    stim_r: np.ndarray
    stim_t: np.ndarray
    target: np.ndarray       # columns are log perceived radius, log perceived duration
    produced: np.ndarray     # columns are log produced radius, log produced duration


def load_subject_data() -> tuple[list[dict], dict[str, SubjectData]]:
    """Return matched condition means and dense subject-specific arrays."""
    rows = build_matched_rows(load_human_condition_means())
    subjects: dict[str, SubjectData] = {}
    for subject in sorted({str(row["subject"]) for row in rows}):
        selected = [row for row in rows if str(row["subject"]) == subject]
        selected.sort(key=lambda row: (float(row["stim_t"]), float(row["stim_r"])))
        subjects[subject] = SubjectData(
            subject=subject,
            stim_r=np.asarray([row["stim_r"] for row in selected], dtype=float),
            stim_t=np.asarray([row["stim_t"] for row in selected], dtype=float),
            target=np.log(np.asarray([[row["perceived_r"], row["perceived_t"]] for row in selected], dtype=float)),
            produced=np.log(np.asarray([[row["produced_r"], row["produced_t"]] for row in selected], dtype=float)),
        )
    return rows, subjects


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0]), float(beta[1])


def subject_speed_statistics(subjects: dict[str, SubjectData]) -> list[dict]:
    """Fit produced log speed = intercept + gain * target log speed."""
    output = []
    for subject, data in subjects.items():
        target_speed = data.target[:, 0] - data.target[:, 1]
        produced_speed = data.produced[:, 0] - data.produced[:, 1]
        intercept, gain = _linear_fit(target_speed, produced_speed)
        fixed_log_speed = intercept / (1.0 - gain) if abs(1.0 - gain) > 1e-6 else np.nan
        raw_intercept, raw_gain = _linear_fit(np.exp(target_speed), np.exp(produced_speed))
        pearson_result = stats.pearsonr(target_speed, produced_speed)
        output.append(
            {
                "subject": subject,
                "intercept": intercept,
                "gain": gain,
                "fixed_log_speed": fixed_log_speed,
                "fixed_speed": float(np.exp(fixed_log_speed)) if np.isfinite(fixed_log_speed) else np.nan,
                "raw_intercept": raw_intercept,
                "raw_gain": raw_gain,
                "pearson_r": float(pearson_result[0]),
            }
        )
    return output


def speed_correction_statistics(subjects: dict[str, SubjectData]) -> list[dict]:
    """Decompose the speed correction into radius and duration components.

    Delta log speed = Delta log radius - Delta log duration.  The two fitted
    slopes therefore sum to one (up to numerical precision), and quantify how
    much of the condition-dependent correction is allocated to each effector.
    """
    output = []
    for subject, data in subjects.items():
        radius_correction = data.produced[:, 0] - data.target[:, 0]
        duration_correction = -(data.produced[:, 1] - data.target[:, 1])
        speed_correction = radius_correction + duration_correction
        ir, alpha_r = _linear_fit(speed_correction, radius_correction)
        it, alpha_t = _linear_fit(speed_correction, duration_correction)
        output.append(
            {
                "subject": subject,
                "alpha_radius": alpha_r,
                "alpha_duration": alpha_t,
                "alpha_sum": alpha_r + alpha_t,
                "intercept_radius": ir,
                "intercept_duration": it,
            }
        )
    return output


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def decode_setpoint(model: str, theta: np.ndarray) -> dict[str, float]:
    """Decode identifiable gains and the spatial allocation fraction.

    The speed correction is split explicitly: rho is applied to log radius and
    1-rho to log duration.  This makes the proposed spatial implementation a
    directly testable parameter rather than a side effect of two cost weights.
    """
    values = {"gain_r": 1.0, "gain_t": 1.0, "gain_v": 1.0, "rho": np.nan, "r0": 0.0, "t0": 0.0, "s0": 0.0}
    if model == "MR":
        values.update(gain_r=_sigmoid(theta[0]), r0=float(theta[1]))
    elif model == "MRT":
        values.update(gain_r=_sigmoid(theta[0]), r0=float(theta[1]), gain_t=_sigmoid(theta[2]), t0=float(theta[3]))
    elif model == "MV":
        values.update(gain_v=_sigmoid(theta[0]), s0=float(theta[1]), rho=_sigmoid(theta[2]))
    elif model == "MRV":
        values.update(gain_r=_sigmoid(theta[0]), r0=float(theta[1]), gain_v=_sigmoid(theta[2]), s0=float(theta[3]), rho=_sigmoid(theta[4]))
    else:
        raise ValueError(f"Unknown setpoint model: {model}")
    return values


def predict_setpoint(target: np.ndarray, model: str, theta: np.ndarray) -> np.ndarray:
    """Predict with sequential radius/time and preferred-speed corrections."""
    p = decode_setpoint(model, np.asarray(theta, dtype=float))
    output = np.asarray(target, dtype=float).copy()
    if model in {"MR", "MRT", "MRV"}:
        output[:, 0] = p["r0"] + p["gain_r"] * (output[:, 0] - p["r0"])
    if model == "MRT":
        output[:, 1] = p["t0"] + p["gain_t"] * (output[:, 1] - p["t0"])
    if model in {"MV", "MRV"}:
        base_speed = output[:, 0] - output[:, 1]
        corrected_speed = p["s0"] + p["gain_v"] * (base_speed - p["s0"])
        delta = corrected_speed - base_speed
        output[:, 0] += p["rho"] * delta
        output[:, 1] -= (1.0 - p["rho"]) * delta
    return output


def solve_old_effort(radius: np.ndarray, duration: np.ndarray, lam: float, w_t: float, iterations: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Specified zero-speed quadratic effort model used in existing SuppFig8."""
    rp = np.asarray(radius, dtype=float).copy()
    tp = np.asarray(duration, dtype=float).copy()
    for _ in range(iterations):
        rp = radius / (1.0 + lam / np.maximum(tp, 1e-6) ** 2)
        tp = duration + (lam / w_t) * rp ** 2 / np.maximum(tp, 1e-6) ** 3
        tp = np.clip(tp, 1e-4, 1e4)
    return rp, tp


def predict_old_effort(target: np.ndarray, theta: np.ndarray) -> np.ndarray:
    lam, w_t = np.exp(np.asarray(theta, dtype=float))
    radius, duration = solve_old_effort(np.exp(target[:, 0]), np.exp(target[:, 1]), float(lam), float(w_t))
    return np.log(np.column_stack([np.maximum(radius, 1e-8), np.maximum(duration, 1e-8)]))


def model_parameter_count(model: str) -> int:
    return {"M0": 0, "MR": 2, "MRT": 4, "MV": 3, "MRV": 5, "M2": 2, "FULL": 6}[model]


def _initial_and_bounds(model: str, target: np.ndarray, produced: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_med = float(np.median(produced[:, 0]))
    t_med = float(np.median(produced[:, 1]))
    s_med = float(np.median(produced[:, 0] - produced[:, 1]))
    if model == "MR":
        initial = np.array([1.4, r_med]); lower = np.array([-7.0, math.log(5.0)]); upper = np.array([7.0, math.log(300.0)])
    elif model == "MRT":
        initial = np.array([1.4, r_med, 3.0, t_med]); lower = np.array([-7.0, math.log(5.0), -7.0, math.log(0.15)]); upper = np.array([7.0, math.log(300.0), 7.0, math.log(8.0)])
    elif model == "MV":
        initial = np.array([1.4, s_med, 1.4]); lower = np.array([-7.0, math.log(1.0), -7.0]); upper = np.array([7.0, math.log(500.0), 7.0])
    elif model == "MRV":
        initial = np.array([1.4, r_med, 2.0, s_med, 1.4]); lower = np.array([-7.0, math.log(5.0), -7.0, math.log(1.0), -7.0]); upper = np.array([7.0, math.log(300.0), 7.0, math.log(500.0), 7.0])
    elif model == "M2":
        initial = np.array([-4.0, 4.0]); lower = np.array([-12.0, -4.0]); upper = np.array([8.0, 12.0])
    else:
        raise ValueError(model)
    return initial, lower, upper


def _prediction(model: str, target: np.ndarray, fitted) -> np.ndarray:
    if model == "M0":
        return target.copy()
    if model == "FULL":
        design = np.column_stack([np.ones(len(target)), target])
        return design @ fitted
    if model == "M2":
        return predict_old_effort(target, fitted)
    return predict_setpoint(target, model, fitted)


def fit_model(model: str, target: np.ndarray, produced: np.ndarray, initial=None, n_starts: int = 3):
    """Fit one candidate using identical standardized log-output residuals."""
    if model == "M0":
        return np.empty(0)
    if model == "FULL":
        design = np.column_stack([np.ones(len(target)), target])
        beta, *_ = np.linalg.lstsq(design, produced, rcond=None)
        return beta

    default, lower, upper = _initial_and_bounds(model, target, produced)
    if initial is not None and np.shape(initial) == np.shape(default):
        default = np.clip(np.asarray(initial, dtype=float), lower + 1e-7, upper - 1e-7)
    rng = np.random.RandomState(20260804 + sum(ord(c) for c in model) + len(target))
    starts = [default]
    for _ in range(max(0, n_starts - 1)):
        jitter = rng.normal(0.0, 0.55, size=default.size)
        starts.append(np.clip(default + jitter, lower + 1e-6, upper - 1e-6))

    def residual(theta):
        prediction = _prediction(model, target, theta)
        if not np.all(np.isfinite(prediction)):
            return np.full(produced.size, 1e5)
        return ((prediction - produced) / LOG_SCALE[None, :]).ravel()

    best = None
    for start in starts:
        result = optimize.least_squares(residual, start, bounds=(lower, upper), max_nfev=350, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]


def predict_model(model: str, target: np.ndarray, fitted) -> np.ndarray:
    return _prediction(model, target, fitted)


def standardized_rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((predicted - observed) / LOG_SCALE[None, :]) ** 2)))


# ---------------------------------------------------------------------------
# Offset-aware physical-space model comparison
# ---------------------------------------------------------------------------
def offset_model_parameter_count(model: str) -> int:
    if model == "FULL":
        return 6
    return 1 + 2 * sum(letter in model for letter in "RTV") + int(model == "RVrho")


def _offset_initial_bounds(model: str, target_log: np.ndarray, produced_log: np.ndarray):
    target_raw = np.exp(target_log)
    produced_raw = np.exp(produced_log)
    c_t = float(np.median(produced_raw[:, 1] - target_raw[:, 1]))
    initial = [float(np.clip(c_t, -0.4, 1.2))]
    lower = [-0.55]
    upper = [1.50]
    if "R" in model:
        initial.extend([-1.0, float(np.median(produced_log[:, 0]))])
        lower.extend([-10.0, math.log(5.0)])
        upper.extend([7.0, math.log(300.0)])
    if "T" in model:
        initial.extend([-2.0, float(np.median(produced_log[:, 1]))])
        lower.extend([-10.0, math.log(0.15)])
        upper.extend([7.0, math.log(8.0)])
    if "V" in model:
        initial.extend([-1.5, float(np.median(produced_log[:, 0] - produced_log[:, 1]))])
        lower.extend([-10.0, math.log(1.0)])
        upper.extend([7.0, math.log(500.0)])
    if model == "RVrho":
        # delta=0 exactly recovers the fixed-fidelity RV model.  Varying delta
        # changes the relative spatial/temporal correction cost while keeping
        # their geometric-mean scale fixed.
        initial.append(0.0)
        lower.append(-4.0)
        upper.append(4.0)
    return np.asarray(initial), np.asarray(lower), np.asarray(upper)


def decode_offset_model(model: str, theta: np.ndarray) -> dict[str, float]:
    """Decode the shared additive-time offset and optional setpoint priors."""
    theta = np.asarray(theta, dtype=float)
    output = {
        "c_t": float(theta[0]),
        "lambda_r": 0.0,
        "lambda_t": 0.0,
        "lambda_v": 0.0,
        "r0": np.nan,
        "t0": np.nan,
        "v0": np.nan,
        "allocation_delta": 0.0,
        "rho": float(
            (1.0 / LOG_FIDELITY_SCALE[1] ** 2)
            / np.sum(1.0 / LOG_FIDELITY_SCALE ** 2)
        ),
    }
    cursor = 1
    if "R" in model:
        output["lambda_r"] = float(np.exp(theta[cursor])); output["r0"] = float(np.exp(theta[cursor + 1])); cursor += 2
    if "T" in model:
        output["lambda_t"] = float(np.exp(theta[cursor])); output["t0"] = float(np.exp(theta[cursor + 1])); cursor += 2
    if "V" in model:
        output["lambda_v"] = float(np.exp(theta[cursor])); output["v0"] = float(np.exp(theta[cursor + 1])); cursor += 2
    if model == "RVrho":
        delta = float(theta[cursor])
        fidelity_weights = np.exp(np.array([-delta, delta])) / LOG_FIDELITY_SCALE ** 2
        output["allocation_delta"] = delta
        output["rho"] = float(fidelity_weights[1] / np.sum(fidelity_weights))
    return output


def predict_offset_model(model: str, target_log: np.ndarray, fitted) -> np.ndarray:
    """Predict raw R/T after an additive time offset and optional setpoints.

    The common baseline is R_b=R* and T_b=T*+c_T.  R, T, and V setpoints are
    simultaneous quadratic priors in log coordinates, but fitting and held-out
    scoring occur in normalized physical R/T units.  Consequently an additive
    time offset cannot masquerade as a temporal setpoint.
    """
    target_log = np.asarray(target_log, dtype=float)
    target_raw = np.exp(target_log)
    if model == "FULL":
        design = np.column_stack([np.ones(len(target_raw)), target_raw])
        return design @ fitted
    theta = np.asarray(fitted, dtype=float)
    c_t = float(theta[0])
    base_raw = target_raw.copy()
    base_raw[:, 1] = np.maximum(base_raw[:, 1] + c_t, 0.05)
    base = np.log(base_raw)

    allocation_delta = float(theta[-1]) if model == "RVrho" else 0.0
    fidelity_weights = np.exp(np.array([-allocation_delta, allocation_delta])) / LOG_FIDELITY_SCALE ** 2
    fidelity = np.diag(fidelity_weights)
    q = fidelity.copy()
    prior = np.zeros(2, dtype=float)
    cursor = 1
    if "R" in model:
        lam, r0 = float(np.exp(theta[cursor])), float(theta[cursor + 1]); cursor += 2
        weight = lam / LOG_FIDELITY_SCALE[0] ** 2
        q[0, 0] += weight
        prior[0] += weight * r0
    if "T" in model:
        lam, t0 = float(np.exp(theta[cursor])), float(theta[cursor + 1]); cursor += 2
        weight = lam / LOG_FIDELITY_SCALE[1] ** 2
        q[1, 1] += weight
        prior[1] += weight * t0
    if "V" in model:
        lam, s0 = float(np.exp(theta[cursor])), float(theta[cursor + 1])
        direction = np.array([1.0, -1.0])
        weight = lam / LOG_SPEED_SCALE ** 2
        q += weight * np.outer(direction, direction)
        prior += weight * s0 * direction
    b = base @ fidelity.T + prior[None, :]
    predicted_log = np.linalg.solve(q, b.T).T
    return np.exp(predicted_log)


def fit_offset_model(model: str, target_log: np.ndarray, produced_log: np.ndarray, initial=None, n_starts: int = 4):
    """Fit one offset-aware model using common raw physical-space residuals."""
    target_raw = np.exp(target_log)
    produced_raw = np.exp(produced_log)
    if model == "FULL":
        design = np.column_stack([np.ones(len(target_raw)), target_raw])
        beta, *_ = np.linalg.lstsq(design, produced_raw, rcond=None)
        return beta
    default, lower, upper = _offset_initial_bounds(model, target_log, produced_log)
    if initial is not None and np.shape(initial) == np.shape(default):
        default = np.clip(np.asarray(initial, dtype=float), lower + 1e-7, upper - 1e-7)
    rng = np.random.RandomState(20260811 + sum(ord(char) for char in model) + len(target_log))
    starts = [default]
    for _ in range(max(0, n_starts - 1)):
        jitter = rng.normal(0.0, 0.45, size=default.size)
        jitter[0] *= 0.35
        starts.append(np.clip(default + jitter, lower + 1e-6, upper - 1e-6))

    def residual(theta):
        predicted = predict_offset_model(model, target_log, theta)
        if not np.all(np.isfinite(predicted)):
            return np.full(produced_raw.size, 1e5)
        return ((predicted - produced_raw) / RAW_SCALE[None, :]).ravel()

    best = None
    for start in starts:
        result = optimize.least_squares(residual, start, bounds=(lower, upper), max_nfev=450, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return best[1]


def raw_standardized_rmse(predicted_raw: np.ndarray, observed_log: np.ndarray) -> float:
    observed_raw = np.exp(observed_log)
    return float(np.sqrt(np.mean(((predicted_raw - observed_raw) / RAW_SCALE[None, :]) ** 2)))


def cross_validate_offset_models(subjects: dict[str, SubjectData], models: Iterable[str], scheme: str = "condition"):
    """Cross-validate offset-aware models with raw normalized R/T loss."""
    models = list(models)
    records = []
    predictions = {model: {} for model in models}
    for subject, data in subjects.items():
        folds = _fold_masks(data, scheme)
        full_fits = {model: fit_offset_model(model, data.target, data.produced, n_starts=4) for model in models}
        for model in models:
            predicted = np.full_like(data.produced, np.nan)
            fold_rho = []
            for test in folds:
                train = ~test
                fitted = fit_offset_model(model, data.target[train], data.produced[train], initial=full_fits[model], n_starts=1)
                predicted[test] = predict_offset_model(model, data.target[test], fitted)
                if model == "RVrho":
                    fold_rho.append(decode_offset_model(model, fitted)["rho"])
            predictions[model][subject] = predicted
            record = {
                    "subject": subject,
                    "scheme": scheme,
                    "model": model,
                    "rmse": raw_standardized_rmse(predicted, data.produced),
                    "rmse_R": float(np.sqrt(np.mean(((predicted[:, 0] - np.exp(data.produced[:, 0])) / RAW_SCALE[0]) ** 2))),
                    "rmse_T": float(np.sqrt(np.mean(((predicted[:, 1] - np.exp(data.produced[:, 1])) / RAW_SCALE[1]) ** 2))),
                    "n_test_predictions": int(np.isfinite(predicted).all(axis=1).sum()),
                }
            if fold_rho:
                record["fold_rho"] = [float(value) for value in fold_rho]
                record["median_training_rho"] = float(np.median(fold_rho))
            records.append(record)
    return records, predictions


def _fold_masks(data: SubjectData, scheme: str) -> list[np.ndarray]:
    n = len(data.target)
    if scheme == "condition":
        return [np.arange(n) == i for i in range(n)]
    if scheme == "radius":
        return [np.isclose(data.stim_r, level) for level in np.unique(data.stim_r)]
    if scheme == "duration":
        return [np.isclose(data.stim_t, level) for level in np.unique(data.stim_t)]
    raise ValueError(scheme)


def cross_validate_models(subjects: dict[str, SubjectData], models: Iterable[str], scheme: str = "condition") -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    """Subject-specific cross-validation with a common held-out loss."""
    models = list(models)
    records: list[dict] = []
    predictions: dict[str, dict[str, np.ndarray]] = {model: {} for model in models}
    for subject, data in subjects.items():
        folds = _fold_masks(data, scheme)
        full_fits = {model: fit_model(model, data.target, data.produced, n_starts=3) for model in models}
        for model in models:
            pred = np.full_like(data.produced, np.nan)
            for test in folds:
                train = ~test
                fitted = fit_model(model, data.target[train], data.produced[train], initial=full_fits[model], n_starts=1)
                pred[test] = predict_model(model, data.target[test], fitted)
            predictions[model][subject] = pred
            records.append(
                {
                    "subject": subject,
                    "scheme": scheme,
                    "model": model,
                    "rmse": standardized_rmse(pred, data.produced),
                    "n_test_predictions": int(np.isfinite(pred).all(axis=1).sum()),
                }
            )
    return records, predictions


def normalized_control_matrix(target: np.ndarray, output: np.ndarray) -> np.ndarray:
    target_raw = np.exp(target)
    output_raw = np.exp(output)
    design = np.column_stack(
        [
            np.ones(len(target)),
            normalize(target_raw[:, 0], R_C, R_H),
            normalize(target_raw[:, 1], T_C, T_H),
        ]
    )
    beta_r, *_ = np.linalg.lstsq(design, normalize(output_raw[:, 0], R_C, R_H), rcond=None)
    beta_t, *_ = np.linalg.lstsq(design, normalize(output_raw[:, 1], T_C, T_H), rcond=None)
    return np.array([[beta_r[1], beta_r[2]], [beta_t[1], beta_t[2]]], dtype=float)


def matrices_from_predictions(subjects: dict[str, SubjectData], predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    observed = []
    predicted = []
    for subject, data in subjects.items():
        observed.append(normalized_control_matrix(data.target, data.produced))
        predicted.append(normalized_control_matrix(data.target, predictions[subject]))
    return np.asarray(observed), np.asarray(predicted)


def bootstrap_median_ci(values: np.ndarray, seed: int = 20260804, n_boot: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.RandomState(seed)
    medians = np.median(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def wilcoxon(values: np.ndarray, comparison: float = 0.0, alternative: str = "two-sided") -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] - comparison
    try:
        # SciPy 1.3 (the paper environment) used the normal approximation by
        # default. Request it explicitly so newer SciPy releases do not switch
        # to exact enumeration and change reported P values.
        result = stats.wilcoxon(values, alternative=alternative, method="approx")
    except TypeError:
        result = stats.wilcoxon(values, alternative=alternative)
    return {"n": int(len(values)), "statistic": float(result.statistic), "p": float(result.pvalue), "alternative": alternative, "comparison": float(comparison)}


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, float((m - rank) * values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def condition_medians(subjects: dict[str, SubjectData]) -> dict[str, np.ndarray]:
    keys = sorted({(float(r), float(t)) for data in subjects.values() for r, t in zip(data.stim_r, data.stim_t)}, key=lambda x: (x[1], x[0]))
    target, produced = [], []
    for sr, st in keys:
        x, y = [], []
        for data in subjects.values():
            mask = np.isclose(data.stim_r, sr) & np.isclose(data.stim_t, st)
            x.extend(data.target[mask])
            y.extend(data.produced[mask])
        target.append(np.median(np.asarray(x), axis=0))
        produced.append(np.median(np.asarray(y), axis=0))
    return {"keys": np.asarray(keys), "target": np.asarray(target), "produced": np.asarray(produced)}


def collect_condition_kinematics(rows: list[dict], min_points: int = 8) -> list[dict]:
    """Aggregate measured path and peak speed for each subject-condition cell."""
    perceived = {
        (str(row["subject"]), float(row["stim_r"]), float(row["stim_t"])): (float(row["perceived_r"]), float(row["perceived_t"]))
        for row in rows
    }
    trials: dict[tuple, list[tuple[float, float]]] = {}
    for path in iter_human_subject_files(HUMAN_DATA_DIR):
        subject = human_subject_id(path)
        for row in load_human_subject_rows(path):
            if str(row.get("trialtype", "")).strip().lower() != "normal":
                continue
            xy = human_draw_xy_from_row(row, min_points=min_points)
            if xy is None:
                continue
            response = human_response_from_row(row)
            try:
                sr, st = float(row["stimx"]), float(row["stimt"])
                duration = float(response["respt"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(duration) or duration <= 0:
                continue
            distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            dt = duration / (len(xy) - 1)
            instantaneous = distance / dt
            trials.setdefault((subject, sr, st), []).append((float(distance.sum() / duration), float(np.percentile(instantaneous, 95))))
    output = []
    for key, values in sorted(trials.items()):
        if key not in perceived:
            continue
        radius, duration = perceived[key]
        arr = np.asarray(values, dtype=float)
        output.append(
            {
                "subject": key[0],
                "stim_r": key[1],
                "stim_t": key[2],
                "target_path_speed": float(2.0 * math.pi * radius / duration),
                "mean_path_speed": float(np.median(arr[:, 0])),
                "peak_speed": float(np.median(arr[:, 1])),
                "n_trials": int(len(arr)),
            }
        )
    return output


def kinematic_speed_slopes(records: list[dict]) -> list[dict]:
    output = []
    for subject in sorted({record["subject"] for record in records}):
        selected = [record for record in records if record["subject"] == subject]
        target = np.log([record["target_path_speed"] for record in selected])
        mean = np.log([record["mean_path_speed"] for record in selected])
        peak = np.log([record["peak_speed"] for record in selected])
        _, mean_gain = _linear_fit(target, mean)
        _, peak_gain = _linear_fit(target, peak)
        output.append({"subject": subject, "mean_path_gain": mean_gain, "peak_gain": peak_gain, "n_conditions": len(selected)})
    return output


def bootstrap_subject_fixed_point(data: SubjectData, n_boot: int = 2000, seed: int = 20260804) -> dict:
    rng = np.random.RandomState(seed + sum(ord(c) for c in data.subject))
    x = data.target[:, 0] - data.target[:, 1]
    y = data.produced[:, 0] - data.produced[:, 1]
    intercept, gain = _linear_fit(x, y)
    point = intercept / (1.0 - gain) if abs(1.0 - gain) > 1e-6 else np.nan
    estimates = []
    for _ in range(n_boot):
        index = rng.randint(0, len(x), len(x))
        a, g = _linear_fit(x[index], y[index])
        if abs(1.0 - g) > 0.03:
            estimates.append(a / (1.0 - g))
    estimates = np.asarray(estimates, dtype=float)
    ci_log = np.percentile(estimates, [2.5, 97.5])
    return {
        "subject": data.subject,
        "fixed_speed": float(np.exp(point)),
        "ci": [float(np.exp(np.clip(ci_log[0], -20.0, 20.0))), float(np.exp(np.clip(ci_log[1], -20.0, 20.0)))],
        "n_boot_valid": int(len(estimates)),
    }


def fit_group_models(subjects: dict[str, SubjectData], models: Iterable[str]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    medians = condition_medians(subjects)
    target, produced = medians["target"], medians["produced"]
    fitted = {model: fit_model(model, target, produced, n_starts=5) for model in models}
    return target, produced, fitted


def model_recovery(subjects: dict[str, SubjectData], models: Iterable[str], n_rep: int = 40, seed: int = 20260804) -> dict:
    """Parametric recovery using BIC under the empirical group residual scale."""
    models = list(models)
    target, produced, fitted = fit_group_models(subjects, models)
    reference = predict_model("MRV", target, fit_model("MRV", target, produced, n_starts=5))
    residual_sd = np.std(produced - reference, axis=0, ddof=1)
    residual_sd = np.maximum(residual_sd, np.array([0.025, 0.025]))
    rng = np.random.RandomState(seed)
    counts = np.zeros((len(models), len(models)), dtype=int)
    bic_differences = []
    for i, generating in enumerate(models):
        mean = predict_model(generating, target, fitted[generating])
        for _ in range(n_rep):
            simulated = mean + rng.normal(0.0, residual_sd, size=mean.shape)
            bic = []
            for candidate in models:
                fit = fit_model(candidate, target, simulated, n_starts=2)
                prediction = predict_model(candidate, target, fit)
                rss = float(np.sum(((prediction - simulated) / LOG_SCALE[None, :]) ** 2))
                n = simulated.size
                bic.append(n * math.log(max(rss / n, 1e-12)) + model_parameter_count(candidate) * math.log(n))
            winner = int(np.argmin(bic))
            counts[i, winner] += 1
            ordered = np.sort(bic)
            bic_differences.append(float(ordered[1] - ordered[0]))
    return {
        "models": models,
        "counts": counts.tolist(),
        "proportions": (counts / n_rep).tolist(),
        "n_rep": int(n_rep),
        "residual_sd_log": residual_sd.tolist(),
        "median_delta_bic_best_to_second": float(np.median(bic_differences)),
    }
