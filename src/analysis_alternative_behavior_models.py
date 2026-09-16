"""Legacy full analysis of specified alternative behavioral models.

The full control matrix (spatial compression beta_RR<1, temporal fidelity
beta_TT~=1, and the directional interaction asymmetry beta_RT>0 but beta_TR~=0)
is a control-level dynamical signature. This figure tests three specified
alternative models, organised BY HYPOTHESIS (not by direct/interaction), each
panel plotting the specified model's quantitative prediction (grey) against
the data (purple = space, cyan = time). Conclusions apply only to these exact
formulations, not to all noise, effort, or spontaneous-coupling accounts.

Convention: control-side inputs use PERCEIVED R/T (matching main-figure
beta_RR^M); produced / kinematic quantities are as measured.

  M1a Simple proportional-noise model         -> Panel A
      A  duration is noisier than size yet only size compresses -- the opposite
         ordering from proportional-noise compression. Produced CV is not an
         independent estimate of sensory uncertainty.
  M1b Fast adaptive-prior / serial-history model -> Panels B-D
      B empirical lag kernels for all four direct/cross history channels
      unlettered inset normal-to-normal versus probe-to-normal lag-1 history
      C pooled blocked predictive gain for lag-1 and extended histories
      D raw change in all four behavior-matrix coefficients after lag-1 removal
  M2  Specified quadratic effort model        -> Panel E (two subaxes)
      left  the two-parameter model manifold is restricted to parameter values
         compatible with the observed temporal coefficients, then compared
         with the bootstrap uncertainty of the spatial coefficients.
      right leave-one-subject-out full-matrix errors are compared directly with a
         train-mean empirical baseline using paired subject-level values.
  M3  Constant-K spontaneous two-thirds model -> Panel F (two subaxes)
      left  produced duration is tested against the R^(2/3) prediction.
      right mean speed and subject-level K modulation are tested against constant-K
         predictions.

This module is called by the root-level `plot_fig2.py` entry point.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, stats

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from analysis_all_common import (
    FIG_DIR,
    HUMAN_DATA_DIR,
    human_draw_xy_from_row,
    human_response_from_row,
    human_subject_id,
    iter_human_subject_files,
    load_human_condition_means,
    load_human_subject_rows,
    write_json,
)
from analysis_behavior_common import (
    COLORS,
    build_matched_rows,
    fit_subject_models,
    label_panel,
    normalize,
    setup_style,
)

C_R, C_T, C_MODEL = COLORS["R"], COLORS["T"], (0.45, 0.45, 0.45)
R_C, R_H, T_C, T_H = 87.5, 48.5, 1.9, 1.2
NORM = {"R": {"center": R_C, "half_range": R_H}, "T": {"center": T_C, "half_range": T_H}}
MIN_TRAJ, MIN_CELL, MIN_LEVEL = 8, 8, 5
MATRIX_VMIN, MATRIX_VMAX = -0.2, 1.2
SERIAL_LAGS = tuple(range(1, 6))
SERIAL_LAMBDAS = np.asarray([0.02, 0.04, 0.08, 0.12, 0.20, 0.30, 0.45, 0.65, 0.85, 1.00])
SERIAL_MODELS = ("lag1_all", "exp_all", "exp_drawing")
POOLED_BLOCKED_VALIDATION_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "behavior_serial_history"
    / "extended_history_blocked_cv.json"
)


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def collect_trials():
    recs = []
    for path in iter_human_subject_files(HUMAN_DATA_DIR):
        subj = human_subject_id(path)
        for row in load_human_subject_rows(path):
            if str(row.get("trialtype", "")).strip().lower() != "normal":
                continue
            xy = human_draw_xy_from_row(row, min_points=MIN_TRAJ)
            if xy is None:
                continue
            resp = human_response_from_row(row)
            pr, pt = float(resp["respx"]), float(resp["respt"])
            try:
                sr, st = float(row["stimx"]), float(row["stimt"])
            except (KeyError, ValueError):
                continue
            if not (np.isfinite(pr) and pr > 0 and pt > 0):
                continue
            n = xy.shape[0]
            dt = pt / (n - 1)
            v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
            a = np.diff(v) / dt
            jrk = np.diff(a) / dt if a.size > 1 else np.array([0.0])
            start = xy[min(3, n - 1)] - xy[0]
            recs.append(dict(
                subj=subj, stim_r=sr, stim_t=st, produced_r=pr, produced_t=pt,
                peak_speed=float(np.percentile(v, 95)),
                mean_speed=float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) / pt),
                e_kin=float(np.sum(v ** 2) * dt),
                e_jerk=float(np.sum(jrk ** 2) * dt),
                start_dir=float(math.atan2(start[1], start[0])),
            ))
    return recs


def load_pooled_blocked_serial_validation():
    if not POOLED_BLOCKED_VALIDATION_PATH.exists():
        raise FileNotFoundError(
            "Restore the versioned serial-history cache before plotting Figure 2."
        )
    return json.loads(POOLED_BLOCKED_VALIDATION_PATH.read_text(encoding="utf-8"))


def collect_chronological_trials(perceived_lookup):
    """Load the exact within-day trial stream for serial-history analyses.

    Current motor responses are retained only for valid normal drawing trials,
    whereas the target/perceived input is retained for both normal and probe
    trials.  Perceived inputs use the same subject-by-condition lookup as the
    main behavior matrix.
    """
    sequences = {}
    for path in iter_human_subject_files(HUMAN_DATA_DIR):
        subject = human_subject_id(path)
        rows = []
        for raw in load_human_subject_rows(path):
            trialtype = str(raw.get("trialtype", "")).strip().lower()
            if trialtype not in {"normal", "probe"}:
                continue
            try:
                day = int(float(raw["day"]))
                trial = int(float(raw["trialN"]))
                stim_r = float(raw["stimx"])
                stim_t = float(raw["stimt"])
            except (KeyError, TypeError, ValueError):
                continue
            perceived = perceived_lookup.get((subject, round(stim_r, 2), round(stim_t, 2)))
            if perceived is None or not np.all(np.isfinite(perceived)):
                continue
            response = human_response_from_row(raw)
            produced_r = float(response["respx"])
            produced_t = float(response["respt"])
            valid_motor = (
                trialtype == "normal"
                and np.isfinite(produced_r)
                and np.isfinite(produced_t)
                and produced_r > 0
                and produced_t > 0
            )
            rows.append(
                dict(
                    subject=subject,
                    day=day,
                    trial=trial,
                    trialtype=trialtype,
                    stim_r=stim_r,
                    stim_t=stim_t,
                    x=np.asarray(
                        [
                            normalize(perceived[0], R_C, R_H),
                            normalize(perceived[1], T_C, T_H),
                        ],
                        dtype=float,
                    ),
                    y=np.asarray(
                        [
                            normalize(produced_r, R_C, R_H),
                            normalize(produced_t, T_C, T_H),
                        ],
                        dtype=float,
                    )
                    if valid_motor
                    else np.full(2, np.nan),
                )
            )
        rows.sort(key=lambda row: (row["day"], row["trial"]))
        day_max = {}
        for row in rows:
            day_max[row["day"]] = max(day_max.get(row["day"], 0), row["trial"])
        for row in rows:
            row["progress"] = row["trial"] / max(day_max[row["day"]], 1)
        sequences[subject] = rows
    return sequences


def _legacy_wilcoxon(x, y=None, alternative="two-sided"):
    """Reproduce the SciPy 1.3 normal-approximation signed-rank test."""
    try:
        return stats.wilcoxon(x, y, alternative=alternative, method="approx")
    except TypeError:
        return stats.wilcoxon(x, y, alternative=alternative)


def _safe_wilcoxon(values, center=0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)] - float(center)
    if values.size < 2 or np.all(np.abs(values) < 1e-12):
        return float("nan")
    return float(_legacy_wilcoxon(values).pvalue)


def _day_fixed_effects(days):
    days = np.asarray(days, dtype=int)
    levels = np.unique(days)
    if levels.size <= 1:
        return np.empty((len(days), 0), dtype=float)
    return np.column_stack([(days == level).astype(float) for level in levels[1:]])


def _serial_design(current_x, prior_x, progress, days, include_day=True):
    current_x = np.asarray(current_x, dtype=float)
    prior_x = np.asarray(prior_x, dtype=float)
    progress = np.asarray(progress, dtype=float)
    columns = [np.ones(len(current_x)), current_x[:, 0], current_x[:, 1],
               prior_x[:, 0] - current_x[:, 0], prior_x[:, 1] - current_x[:, 1],
               progress, progress ** 2, progress ** 3]
    design = np.column_stack(columns)
    if include_day:
        day_terms = _day_fixed_effects(days)
        if day_terms.shape[1]:
            design = np.column_stack([design, day_terms])
    return design


def _fit_serial_samples(samples, previous_residual=None):
    if len(samples) < 40:
        return None
    current_x = np.asarray([row["current"]["x"] for row in samples], dtype=float)
    prior_x = np.asarray([row["previous"]["x"] for row in samples], dtype=float)
    y = np.asarray([row["current"]["y"] for row in samples], dtype=float)
    progress = np.asarray([row["current"]["progress"] for row in samples], dtype=float)
    days = np.asarray([row["current"]["day"] for row in samples], dtype=int)
    design = _serial_design(current_x, prior_x, progress, days, include_day=True)
    response_slice = None
    if previous_residual is not None:
        prev_e = np.asarray(
            [previous_residual[(row["previous"]["day"], row["previous"]["trial"])] for row in samples],
            dtype=float,
        )
        response_slice = slice(design.shape[1], design.shape[1] + 2)
        design = np.column_stack([design, prev_e])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coef
    out = dict(
        n=int(len(samples)),
        B_instantaneous=coef[1:3].T,
        H=coef[3:5].T,
        residual=y - prediction,
    )
    if response_slice is not None:
        out["C"] = coef[response_slice].T
    return out


def _exact_lag_samples(sequence, lag=1, previous_type=None):
    lookup = {(row["day"], row["trial"]): row for row in sequence}
    samples = []
    for current in sequence:
        if current["trialtype"] != "normal" or not np.all(np.isfinite(current["y"])):
            continue
        previous = lookup.get((current["day"], current["trial"] - int(lag)))
        if previous is None or not np.all(np.isfinite(previous["x"])):
            continue
        if previous_type is not None and previous["trialtype"] != previous_type:
            continue
        samples.append(dict(current=current, previous=previous))
    return samples


def _baseline_response_residuals(sequence):
    normal = [row for row in sequence if row["trialtype"] == "normal" and np.all(np.isfinite(row["y"]))]
    x = np.asarray([row["x"] for row in normal], dtype=float)
    y = np.asarray([row["y"] for row in normal], dtype=float)
    progress = np.asarray([row["progress"] for row in normal], dtype=float)
    days = np.asarray([row["day"] for row in normal], dtype=int)
    design = np.column_stack([np.ones(len(x)), x, progress, progress ** 2, progress ** 3,
                              _day_fixed_effects(days)])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coef
    return {(row["day"], row["trial"]): residual[i] for i, row in enumerate(normal)}


def _prior_records(sequence, lambda_value, history_kind):
    """Return current-normal records with a causal prior available before trial n.

    all: every presented target updates the prior.
    drawing: every trial decays the prior, but only drawing targets update it.
    """
    records = []
    for day in sorted({row["day"] for row in sequence}):
        prior = np.zeros(2, dtype=float)
        day_rows = [row for row in sequence if row["day"] == day]
        for row in day_rows:
            if row["trialtype"] == "normal" and np.all(np.isfinite(row["y"])):
                records.append(
                    dict(
                        subject=row["subject"], day=day, trial=row["trial"],
                        progress=row["progress"], stim_r=row["stim_r"], stim_t=row["stim_t"],
                        x=row["x"].copy(), y=row["y"].copy(), prior=prior.copy(),
                    )
                )
            if history_kind == "all":
                prior = (1.0 - lambda_value) * prior + lambda_value * row["x"]
            elif history_kind == "drawing":
                prior = (1.0 - lambda_value) * prior
                if row["trialtype"] == "normal":
                    prior = prior + lambda_value * row["x"]
            else:
                raise ValueError(f"Unknown history_kind: {history_kind}")
    return records


def _records_design(records, history=False, include_day=False):
    x = np.asarray([row["x"] for row in records], dtype=float)
    columns = [np.ones(len(records)), x[:, 0], x[:, 1]]
    if history:
        prior = np.asarray([row["prior"] for row in records], dtype=float)
        columns.extend([prior[:, 0] - x[:, 0], prior[:, 1] - x[:, 1]])
    progress = np.asarray([row["progress"] for row in records], dtype=float)
    columns.extend([progress, progress ** 2, progress ** 3])
    design = np.column_stack(columns)
    if include_day:
        day_terms = _day_fixed_effects([row["day"] for row in records])
        if day_terms.shape[1]:
            design = np.column_stack([design, day_terms])
    y = np.asarray([row["y"] for row in records], dtype=float)
    return design, y


def _fit_predict_records(train_records, test_records, history=False):
    x_train, y_train = _records_design(train_records, history=history, include_day=False)
    x_test, y_test = _records_design(test_records, history=history, include_day=False)
    coef = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    return y_test, x_test @ coef


def _nested_lambda(sequence, training_days, history_kind, lambda_grid):
    scores = []
    for lambda_value in lambda_grid:
        records = _prior_records(sequence, float(lambda_value), history_kind)
        score = 0.0
        valid_folds = 0
        for validation_day in training_days:
            inner_train = [row for row in records if row["day"] in training_days and row["day"] != validation_day]
            inner_test = [row for row in records if row["day"] == validation_day]
            if len(inner_train) < 40 or len(inner_test) < 20:
                continue
            actual, predicted = _fit_predict_records(inner_train, inner_test, history=True)
            score += float(np.sum((actual - predicted) ** 2))
            valid_folds += 1
        scores.append(score if valid_folds else np.inf)
    return float(lambda_grid[int(np.argmin(scores))])


def _cross_validate_serial_models(sequence, lambda_grid=SERIAL_LAMBDAS):
    days = sorted({row["day"] for row in sequence})
    models = {
        "lag1_all": dict(kind="all", fixed_lambda=1.0),
        "exp_all": dict(kind="all", fixed_lambda=None),
        "exp_drawing": dict(kind="drawing", fixed_lambda=None),
    }
    sse_baseline = np.zeros(2, dtype=float)
    sse_models = {name: np.zeros(2, dtype=float) for name in models}
    all_actual = []
    selected = {name: [] for name in models}
    for held_day in days:
        training_days = [day for day in days if day != held_day]
        baseline_records = _prior_records(sequence, 1.0, "all")
        baseline_train = [row for row in baseline_records if row["day"] in training_days]
        baseline_test = [row for row in baseline_records if row["day"] == held_day]
        if len(baseline_train) < 80 or len(baseline_test) < 20:
            continue
        actual, baseline_prediction = _fit_predict_records(baseline_train, baseline_test, history=False)
        sse_baseline += np.sum((actual - baseline_prediction) ** 2, axis=0)
        all_actual.append(actual)
        for name, spec in models.items():
            lambda_value = spec["fixed_lambda"]
            if lambda_value is None:
                lambda_value = _nested_lambda(sequence, training_days, spec["kind"], lambda_grid)
            selected[name].append(float(lambda_value))
            records = _prior_records(sequence, float(lambda_value), spec["kind"])
            train = [row for row in records if row["day"] in training_days]
            test = [row for row in records if row["day"] == held_day]
            model_actual, model_prediction = _fit_predict_records(train, test, history=True)
            if model_actual.shape != actual.shape:
                raise RuntimeError("Serial CV record mismatch")
            sse_models[name] += np.sum((model_actual - model_prediction) ** 2, axis=0)
    actual_all = np.vstack(all_actual)
    denominator = np.sum((actual_all - np.mean(actual_all, axis=0)) ** 2, axis=0)
    delta_r2 = {
        name: (sse_baseline - sse_models[name]) / np.maximum(denominator, 1e-12)
        for name in models
    }
    return dict(
        delta_r2=delta_r2,
        selected_lambda={name: float(np.median(values)) for name, values in selected.items()},
        fold_lambda=selected,
        sse_baseline=sse_baseline,
        sse_models=sse_models,
    )


def _fit_adjusted_matrix(sequence, lambda_value, history_kind):
    records = _prior_records(sequence, float(lambda_value), history_kind)
    design, y = _records_design(records, history=True, include_day=True)
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    B_instantaneous = coef[1:3].T
    H = coef[3:5].T
    current_x = np.asarray([row["x"] for row in records], dtype=float)
    prior = np.asarray([row["prior"] for row in records], dtype=float)
    adjusted_y = y - (prior - current_x) @ H.T

    grouped = {}
    for row, raw_y, adj_y in zip(records, y, adjusted_y):
        key = (round(row["stim_r"], 2), round(row["stim_t"], 2))
        grouped.setdefault(key, []).append((row["x"], raw_y, adj_y))
    x_condition, raw_condition, adjusted_condition = [], [], []
    for values in grouped.values():
        x_condition.append(np.mean([value[0] for value in values], axis=0))
        raw_condition.append(np.mean([value[1] for value in values], axis=0))
        adjusted_condition.append(np.mean([value[2] for value in values], axis=0))
    x_condition = np.asarray(x_condition, dtype=float)
    condition_design = np.column_stack([np.ones(len(x_condition)), x_condition])
    B_original = np.linalg.lstsq(condition_design, np.asarray(raw_condition), rcond=None)[0][1:3].T
    B_adjusted = np.linalg.lstsq(condition_design, np.asarray(adjusted_condition), rcond=None)[0][1:3].T
    return dict(
        B_original=B_original,
        B_adjusted=B_adjusted,
        delta_B=B_adjusted - B_original,
        B_instantaneous=B_instantaneous,
        H=H,
        n=int(len(records)),
    )


def compute_serial_history(perceived_lookup, subject_order):
    sequences = collect_chronological_trials(perceived_lookup)
    subject_order = [str(subject) for subject in subject_order if str(subject) in sequences]
    n_subjects, n_lags = len(subject_order), len(SERIAL_LAGS)
    lag_H = np.full((n_subjects, n_lags, 2, 2), np.nan, dtype=float)
    lag_n = np.zeros((n_subjects, n_lags), dtype=int)
    transition_H = {kind: np.full((n_subjects, 2, 2), np.nan, dtype=float) for kind in ("normal", "probe")}
    transition_n = {kind: np.zeros(n_subjects, dtype=int) for kind in ("normal", "probe")}
    response_C = np.full((n_subjects, 2, 2), np.nan, dtype=float)
    response_H = np.full((n_subjects, 2, 2), np.nan, dtype=float)
    response_n = np.zeros(n_subjects, dtype=int)
    transition_r = np.full((n_subjects, 2, 2), np.nan, dtype=float)
    sample_cache = {}

    for subject_index, subject in enumerate(subject_order):
        sequence = sequences[subject]
        for lag_index, lag in enumerate(SERIAL_LAGS):
            samples = _exact_lag_samples(sequence, lag=lag)
            sample_cache[(subject, lag)] = samples
            fit = _fit_serial_samples(samples)
            if fit is not None:
                lag_H[subject_index, lag_index] = fit["H"]
                lag_n[subject_index, lag_index] = fit["n"]
        for previous_type in ("normal", "probe"):
            samples = _exact_lag_samples(sequence, lag=1, previous_type=previous_type)
            fit = _fit_serial_samples(samples)
            if fit is not None:
                transition_H[previous_type][subject_index] = fit["H"]
                transition_n[previous_type][subject_index] = fit["n"]

        strict_nn = _exact_lag_samples(sequence, lag=1, previous_type="normal")
        previous_residual = _baseline_response_residuals(sequence)
        response_fit = _fit_serial_samples(strict_nn, previous_residual=previous_residual)
        if response_fit is not None:
            response_C[subject_index] = response_fit["C"]
            response_H[subject_index] = response_fit["H"]
            response_n[subject_index] = response_fit["n"]

        adjacent = _exact_lag_samples(sequence, lag=1)
        current_x = np.asarray([row["current"]["x"] for row in adjacent], dtype=float)
        previous_x = np.asarray([row["previous"]["x"] for row in adjacent], dtype=float)
        for output_index in range(2):
            for history_index in range(2):
                if len(adjacent) >= 20 and np.std(current_x[:, output_index]) > 0 and np.std(previous_x[:, history_index]) > 0:
                    transition_r[subject_index, output_index, history_index] = float(stats.pearsonr(
                        current_x[:, output_index], previous_x[:, history_index]
                    )[0])

    # Randomization-aware null: circularly shift previous targets within each day.
    n_permutations = 500
    rng = np.random.RandomState(1827)
    lag_null = np.full((n_permutations, n_lags, 2, 2), np.nan, dtype=float)
    for permutation in range(n_permutations):
        permuted_subject = np.full((n_subjects, n_lags, 2, 2), np.nan, dtype=float)
        for subject_index, subject in enumerate(subject_order):
            for lag_index, lag in enumerate(SERIAL_LAGS):
                samples = sample_cache[(subject, lag)]
                if len(samples) < 40:
                    continue
                permuted = []
                by_day = {}
                for row in samples:
                    by_day.setdefault(row["current"]["day"], []).append(row)
                for day_rows in by_day.values():
                    n = len(day_rows)
                    if n < 10:
                        continue
                    allowed = np.arange(5, max(6, n - 4))
                    shift = int(rng.choice(allowed)) if allowed.size else max(1, n // 2)
                    prior_rows = [row["previous"] for row in day_rows]
                    shifted = np.roll(np.arange(n), shift)
                    for row_index, row in enumerate(day_rows):
                        permuted.append(dict(current=row["current"], previous=prior_rows[int(shifted[row_index])]))
                fit = _fit_serial_samples(permuted)
                if fit is not None:
                    permuted_subject[subject_index, lag_index] = fit["H"]
        lag_null[permutation] = np.nanmedian(permuted_subject, axis=0)

    cv_rows = []
    adjusted = {name: [] for name in SERIAL_MODELS}
    for subject in subject_order:
        sequence = sequences[subject]
        cv = _cross_validate_serial_models(sequence)
        cv_rows.append(dict(subject=subject, **cv))
        specifications = {
            "lag1_all": (1.0, "all"),
            "exp_all": (cv["selected_lambda"]["exp_all"], "all"),
            "exp_drawing": (cv["selected_lambda"]["exp_drawing"], "drawing"),
        }
        for name, (lambda_value, history_kind) in specifications.items():
            adjusted[name].append(
                dict(subject=subject, lambda_=float(lambda_value), **_fit_adjusted_matrix(sequence, lambda_value, history_kind))
            )

    cv_delta_r2 = {
        name: np.asarray([row["delta_r2"][name] for row in cv_rows], dtype=float)
        for name in SERIAL_MODELS
    }
    selected_lambda = {
        name: np.asarray([row["selected_lambda"][name] for row in cv_rows], dtype=float)
        for name in SERIAL_MODELS
    }
    adjusted_original = {
        name: np.asarray([row["B_original"] for row in adjusted[name]], dtype=float)
        for name in SERIAL_MODELS
    }
    adjusted_matrix = {
        name: np.asarray([row["B_adjusted"] for row in adjusted[name]], dtype=float)
        for name in SERIAL_MODELS
    }
    adjusted_delta = {
        name: np.asarray([row["delta_B"] for row in adjusted[name]], dtype=float)
        for name in SERIAL_MODELS
    }
    full_H = {
        name: np.asarray([row["H"] for row in adjusted[name]], dtype=float)
        for name in SERIAL_MODELS
    }
    return dict(
        subjects=subject_order,
        n_subjects=n_subjects,
        lag_H=lag_H,
        lag_n=lag_n,
        lag_null=lag_null,
        n_permutations=n_permutations,
        transition_H=transition_H,
        transition_n=transition_n,
        response_C=response_C,
        response_H=response_H,
        response_n=response_n,
        transition_r=transition_r,
        cv_rows=cv_rows,
        cv_delta_r2=cv_delta_r2,
        selected_lambda=selected_lambda,
        adjusted=adjusted,
        adjusted_original=adjusted_original,
        adjusted_matrix=adjusted_matrix,
        adjusted_delta=adjusted_delta,
        full_H=full_H,
    )


# ----------------------------------------------------------------------------
# effort joint model (M2): min (R_p-R*)^2 + (T_p-T*)^2 + lam (R_p/T_p)^2
# ----------------------------------------------------------------------------
def solve_effort(R0, T0, lam, w, iters=400):
    Rp, Tp = R0.copy(), T0.copy()
    for _ in range(iters):
        Rp = R0 / (1.0 + lam / Tp ** 2)
        Tp = T0 + (lam / w) * Rp ** 2 / Tp ** 3
    return Rp, Tp


def effort_matrix(Rperc, Tperc, lam, w, iters=400):
    Rp, Tp = solve_effort(Rperc, Tperc, lam, w, iters=iters)
    X = np.column_stack([np.ones_like(Rperc), normalize(Rperc, R_C, R_H), normalize(Tperc, T_C, T_H)])
    br = np.linalg.lstsq(X, normalize(Rp, R_C, R_H), rcond=None)[0]
    bt = np.linalg.lstsq(X, normalize(Tp, T_C, T_H), rcond=None)[0]
    return br[1], br[2], bt[1], bt[2]


def effort_spatial_gain_by_duration(Rperc, Tperc, target_t, Ts, lam, w, iters=400):
    """Spatial gain predicted by the effort model at each instructed duration."""
    Rp, _ = solve_effort(np.asarray(Rperc, float), np.asarray(Tperc, float), lam, w, iters=iters)
    x = normalize(Rperc, R_C, R_H)
    y = normalize(Rp, R_C, R_H)
    target_t = np.asarray(target_t, float)
    gains = []
    for t in Ts:
        m = np.isclose(target_t, t)
        if m.sum() >= 3 and np.ptp(x[m]) > 1e-9:
            gains.append(float(np.polyfit(x[m], y[m], 1)[0]))
        else:
            gains.append(float("nan"))
    return gains


def _format_p(p):
    p = float(p)
    if not np.isfinite(p):
        return "p=n/a"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, float((m - rank) * p_values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def _matrix_error(mat, target):
    return float(np.linalg.norm(np.asarray(mat, float) - np.asarray(target, float)))


def _noise_prediction_matrix(CVR, CVT, CR, CT):
    """Variance-minimization prediction scaled to the observed mean compression."""
    cv = np.array([np.median(CVR), np.median(CVT)], float)
    comp = np.array([np.median(CR), np.median(CT)], float)
    scale = float(np.mean(comp) / np.mean(cv))
    mat = np.array([[1.0 - scale * cv[0], 0.0], [0.0, 1.0 - scale * cv[1]]])
    return mat, scale


def _twothirds_prediction_matrix(Rperc, Tperc):
    """Classic spontaneous-coupling prediction: duration follows R^(2/3)."""
    Rout = np.asarray(Rperc, float)
    Tout = T_C * (np.maximum(Rout, 1e-9) / R_C) ** (2.0 / 3.0)
    X = np.column_stack([np.ones_like(Rout), normalize(Rperc, R_C, R_H), normalize(Tperc, T_C, T_H)])
    br = np.linalg.lstsq(X, normalize(Rout, R_C, R_H), rcond=None)[0]
    bt = np.linalg.lstsq(X, normalize(Tout, T_C, T_H), rcond=None)[0]
    return np.array([[br[1], br[2]], [bt[1], bt[2]]])


def _fit_effort_matrix_full(Rperc, Tperc, target, initial=None):
    target_vec = np.asarray(target, float).reshape(-1)

    def residual(u):
        lam = float(np.exp(u[0]))
        w = float(np.exp(u[1]))
        pred = np.array(effort_matrix(Rperc, Tperc, lam, w, iters=160), float)
        if not np.all(np.isfinite(pred)):
            return np.ones_like(target_vec) * 1e6
        return pred - target_vec

    if initial is None:
        starts = []
        for log_lam in np.linspace(-6.0, 2.0, 5):
            for log_w in np.linspace(-1.0, 12.0, 6):
                starts.append([log_lam, log_w])
    else:
        center = np.log(np.maximum(np.asarray(initial, dtype=float), 1e-12))
        starts = [center, center + np.array([-0.5, 0.0]), center + np.array([0.5, 0.0])]
    best = None
    for start in starts:
        sol = optimize.least_squares(
            residual,
            start,
            bounds=([-12.0, -4.0], [5.0, 18.0]),
            max_nfev=1000,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        err = float(np.linalg.norm(sol.fun))
        if best is None or err < best["error"]:
            lam = float(np.exp(sol.x[0]))
            w = float(np.exp(sol.x[1]))
            mat = np.array(effort_matrix(Rperc, Tperc, lam, w, iters=400), float).reshape(2, 2)
            best = dict(lam=lam, w=w, matrix=mat, error=_matrix_error(mat, target), residual=mat - target)
    return best


def _boot_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, float)
    if vals.size < 2:
        v = float(vals[0]) if vals.size else float("nan")
        return v, v
    rng = np.random.RandomState(seed)
    meds = np.median(rng.choice(vals, size=(n, vals.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _family_ci(subj, sr, st, Rs, Ts, subs, xarr, yarr, min_trials=3, min_subj=5):
    """Per-condition family: cross-subject median of per-subject condition medians,
    with bootstrap 95% CI on y. Returns {t: (x, y, ylo, yhi)}."""
    fam = {}
    for t in Ts:
        xs, ys, ylo, yhi = [], [], [], []
        for a in Rs:
            sx, sy = [], []
            for s in subs:
                m = (subj == s) & (np.abs(st - t) < 1e-6) & (np.abs(sr - a) < 1e-6)
                if m.sum() >= min_trials and np.all(np.isfinite(xarr[m])) and np.all(np.isfinite(yarr[m])):
                    sx.append(np.median(xarr[m])); sy.append(np.median(yarr[m]))
            if len(sy) >= min_subj:
                xs.append(float(np.median(sx))); ys.append(float(np.median(sy)))
                lo, hi = _boot_ci(np.array(sy)); ylo.append(lo); yhi.append(hi)
        fam[t] = (np.array(xs), np.array(ys), np.array(ylo), np.array(yhi))
    return fam


def lag1_behavior_contribution(serial, n_boot=20000, seed=2031):
    """Bootstrap group-matrix percentages for all four behavioral coefficients.

    Direct terms are expressed relative to the observed compression (1-beta),
    whereas cross terms are expressed relative to the observed cross-effect
    beta.  Near-zero denominators are intentionally retained and exposed as
    unstable percentages rather than silently suppressing a coefficient.
    """
    original = np.asarray(serial["adjusted_original"]["lag1_all"], dtype=float)
    adjusted = np.asarray(serial["adjusted_matrix"]["lag1_all"], dtype=float)
    observed_mean = np.mean(original, axis=0)
    adjusted_mean = np.mean(adjusted, axis=0)

    def percentages(observed_matrix, adjusted_matrix):
        return np.asarray([
            100.0 * (adjusted_matrix[0, 0] - observed_matrix[0, 0]) / (1.0 - observed_matrix[0, 0]),
            100.0 * (observed_matrix[0, 1] - adjusted_matrix[0, 1]) / observed_matrix[0, 1],
            100.0 * (observed_matrix[1, 0] - adjusted_matrix[1, 0]) / observed_matrix[1, 0],
            100.0 * (adjusted_matrix[1, 1] - observed_matrix[1, 1]) / (1.0 - observed_matrix[1, 1]),
        ], dtype=float)

    estimate = percentages(observed_mean, adjusted_mean)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(original), size=(n_boot, len(original)))
    bootstrap = np.empty((n_boot, 4), dtype=float)
    for bootstrap_index, sample_indices in enumerate(indices):
        bootstrap[bootstrap_index] = percentages(
            np.mean(original[sample_indices], axis=0),
            np.mean(adjusted[sample_indices], axis=0),
        )
    confidence_interval = np.nanpercentile(bootstrap, [2.5, 97.5], axis=0).T
    p_raw = []
    for effect in range(4):
        finite = bootstrap[np.isfinite(bootstrap[:, effect]), effect]
        n_nonpositive = int(np.sum(finite <= 0))
        n_nonnegative = int(np.sum(finite >= 0))
        p_raw.append(min(1.0, 2.0 * (min(n_nonpositive, n_nonnegative) + 1) / (len(finite) + 1)))
    p_raw = np.asarray(p_raw, dtype=float)
    return dict(
        labels=["Spatial compression", "Time-to-space interaction",
                "Space-to-time interaction", "Temporal compression"],
        estimate=estimate,
        confidence_interval=confidence_interval,
        bootstrap=bootstrap,
        p_raw=p_raw,
        p_Holm=_holm_adjust(p_raw),
        observed_mean=observed_mean,
        adjusted_mean=adjusted_mean,
        n_boot=int(n_boot),
        definition=[
            "100*(beta_RR_adjusted-beta_RR_observed)/(1-beta_RR_observed)",
            "100*(beta_RT_observed-beta_RT_adjusted)/beta_RT_observed",
            "100*(beta_TR_observed-beta_TR_adjusted)/beta_TR_observed",
            "100*(beta_TT_adjusted-beta_TT_observed)/(1-beta_TT_observed)",
        ],
    )


# ----------------------------------------------------------------------------
# compute everything
# ----------------------------------------------------------------------------
def compute():
    recs = collect_trials()
    matched = build_matched_rows(load_human_condition_means())
    coefs, _ = fit_subject_models(matched, NORM)
    subs = [c["subject"] for c in coefs]
    bRR = np.array([c["beta_RR"] for c in coefs]); bRT = np.array([c["beta_RT"] for c in coefs])
    bTR = np.array([c["beta_TR"] for c in coefs]); bTT = np.array([c["beta_TT"] for c in coefs])
    # The formal group behavior matrix follows the main figures: fit one matrix
    # per subject, then take the element-wise, equally weighted subject mean.
    # Medians remain the descriptive center for boxplots and nonparametric
    # subject-level inference, but they are not used as a second matrix target.
    mRR, mRT, mTR, mTT = map(np.mean, (bRR, bRT, bTR, bTT))
    B_obs = np.array([[mRR, mRT], [mTR, mTT]])

    # trial arrays
    subj = np.array([r["subj"] for r in recs])
    sr = np.array([r["stim_r"] for r in recs], float); st = np.array([r["stim_t"] for r in recs], float)
    prod_r = np.array([r["produced_r"] for r in recs], float)
    peak = np.array([r["peak_speed"] for r in recs], float)
    mean_sp = np.array([r["mean_speed"] for r in recs], float)
    e_kin = np.array([r["e_kin"] for r in recs], float); e_jerk = np.array([r["e_jerk"] for r in recs], float)
    start_dir = np.array([r["start_dir"] for r in recs], float)
    Ts = sorted(set(np.round(st, 2))); Rs = sorted(set(np.round(sr, 2)))

    # perceived lookup + condition grid
    pmap = {}
    for row in matched:
        pmap[(str(row["subject"]), round(row["stim_r"], 2), round(row["stim_t"], 2))] = (row["perceived_r"], row["perceived_t"])
    perc_r = np.array([pmap.get((s, round(a, 2), round(t, 2)), (np.nan, np.nan))[0] for s, a, t in zip(subj, sr, st)])
    perc_t = np.array([pmap.get((s, round(a, 2), round(t, 2)), (np.nan, np.nan))[1] for s, a, t in zip(subj, sr, st)])
    pT_bin = [float(np.median([pt for (_, _, tt), (_, pt) in pmap.items() if abs(tt - t) < 1e-6 and np.isfinite(pt)])) for t in Ts]
    serial = compute_serial_history(pmap, subs)
    serial_pooled_blocked = load_pooled_blocked_serial_validation()

    def perceived_grid(rows):
        grid = {}
        for row in rows:
            grid.setdefault((round(row["stim_r"], 2), round(row["stim_t"], 2)), []).append(
                (row["perceived_r"], row["perceived_t"])
            )
        keys = sorted(grid)
        r_values = np.array([np.mean([v[0] for v in grid[k]]) for k in keys])
        t_values = np.array([np.mean([v[1] for v in grid[k]]) for k in keys])
        return keys, r_values, t_values

    gk, Rperc_g, Tperc_g = perceived_grid(matched)

    # ---------- Panel A: gain vs perceived T + effort model + dBIC ----------
    rows_by_st = {}
    for row in matched:
        rows_by_st.setdefault((str(row["subject"]), round(row["stim_t"], 2)), []).append(row)

    def gain_at(rows):
        x = normalize([r["perceived_r"] for r in rows], R_C, R_H)
        y = normalize([r["produced_r"] for r in rows], R_C, R_H)
        return float(np.polyfit(x, y, 1)[0]) if len(x) >= 3 and np.ptp(x) > 1e-9 else None

    gain = {t: [] for t in Ts}
    for (s, t), rows in rows_by_st.items():
        g = gain_at(rows)
        if g is not None and t in gain:
            gain[t].append(g)
    gain_med = [float(np.median(gain[t])) if gain[t] else np.nan for t in Ts]

    subj_slopes = []
    empirical_slope_by_subject = {}
    for s in subs:
        xs, ys = [], []
        for t in Ts:
            rows = rows_by_st.get((s, t), [])
            g = gain_at(rows)
            if g is not None:
                pt = np.nanmean([r["perceived_t"] for r in rows])
                if np.isfinite(pt):
                    xs.append(pt); ys.append(g)
        if len(xs) >= 4:
            slope = float(np.polyfit(xs, ys, 1)[0])
            subj_slopes.append(slope)
            empirical_slope_by_subject[str(s)] = slope
    subj_slopes = np.array(subj_slopes)
    emp_slope = float(np.median(subj_slopes))
    emp_slope_p = float(_legacy_wilcoxon(subj_slopes)[1])

    peak_ratio = float(np.median(peak[np.abs(st - min(Ts)) < 1e-6]) / np.median(peak[np.abs(st - max(Ts)) < 1e-6]))

    # ---------- M2 full-matrix fit + leave-one-subject-out predictions ----------
    effort_best = _fit_effort_matrix_full(Rperc_g, Tperc_g, B_obs)
    best_w = float(effort_best["w"])
    coef_by_subject = {str(c["subject"]): c for c in coefs}
    subject_matrices = np.asarray(
        [
            [[c["beta_RR"], c["beta_RT"]], [c["beta_TR"], c["beta_TT"]]]
            for c in coefs
        ],
        dtype=float,
    )
    rng_effort = np.random.RandomState(2026)
    bootstrap_indices = rng_effort.randint(0, len(subject_matrices), size=(5000, len(subject_matrices)))
    # Bootstrap the same estimator used for the observed matrix: the
    # element-wise subject mean.
    bootstrap_matrices = np.mean(subject_matrices[bootstrap_indices], axis=1)
    timing_ci_tr = np.percentile(bootstrap_matrices[:, 1, 0], [2.5, 97.5])
    timing_ci_tt = np.percentile(bootstrap_matrices[:, 1, 1], [2.5, 97.5])
    spatial_bootstrap = bootstrap_matrices[:, 0, :]
    spatial_bootstrap_cov = np.cov(spatial_bootstrap.T, ddof=1)

    lambda_grid = np.logspace(-4.0, 1.0, 55)
    w_grid = np.logspace(1.0, 8.0, 55)
    effort_grid_rows = []
    for lambda_value in lambda_grid:
        for w_value in w_grid:
            matrix = np.asarray(
                effort_matrix(Rperc_g, Tperc_g, lambda_value, w_value, iters=180), dtype=float
            ).reshape(2, 2)
            if np.all(np.isfinite(matrix)):
                effort_grid_rows.append(
                    [lambda_value, w_value, matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]]
                )
    effort_grid = np.asarray(effort_grid_rows, dtype=float)
    timing_feasible = (
        (effort_grid[:, 4] >= timing_ci_tr[0])
        & (effort_grid[:, 4] <= timing_ci_tr[1])
        & (effort_grid[:, 5] >= timing_ci_tt[0])
        & (effort_grid[:, 5] <= timing_ci_tt[1])
    )
    effort_feasible = effort_grid[timing_feasible]
    if effort_feasible.size:
        covariance_inverse = np.linalg.pinv(spatial_bootstrap_cov)
        displacement = effort_feasible[:, 2:4] - np.asarray([mRR, mRT])
        feasible_mahalanobis2 = np.einsum("ni,ij,nj->n", displacement, covariance_inverse, displacement)
        feasible_inside_observed_95 = int(np.sum(feasible_mahalanobis2 <= stats.chi2.ppf(0.95, df=2)))
        closest_feasible_index = int(np.argmin(feasible_mahalanobis2))
        closest_feasible = effort_feasible[closest_feasible_index]
        closest_feasible_mahalanobis2 = float(feasible_mahalanobis2[closest_feasible_index])
        closest_feasible_source = "grid"

        def constrained_model_vector(log_parameters):
            lambda_value, w_value = np.exp(log_parameters)
            return np.asarray(
                effort_matrix(Rperc_g, Tperc_g, lambda_value, w_value, iters=240), dtype=float
            )

        def spatial_distance(log_parameters):
            delta = constrained_model_vector(log_parameters)[:2] - np.asarray([mRR, mRT])
            return float(delta @ covariance_inverse @ delta)

        temporal_constraint = optimize.NonlinearConstraint(
            lambda u: constrained_model_vector(u)[2:4],
            [timing_ci_tr[0], timing_ci_tt[0]],
            [timing_ci_tr[1], timing_ci_tt[1]],
        )
        start_indices = np.argsort(feasible_mahalanobis2)[: min(6, len(effort_feasible))]
        log_bounds = optimize.Bounds(
            np.log([lambda_grid.min(), w_grid.min()]),
            np.log([lambda_grid.max(), w_grid.max()]),
        )
        for start_index in start_indices:
            start = np.log(effort_feasible[start_index, :2])
            solution = optimize.minimize(
                spatial_distance,
                start,
                method="SLSQP",
                bounds=log_bounds,
                constraints=[temporal_constraint],
                options=dict(maxiter=300, ftol=1e-10),
            )
            solution_vector = constrained_model_vector(solution.x)
            temporally_valid = (
                timing_ci_tr[0] - 1e-5 <= solution_vector[2] <= timing_ci_tr[1] + 1e-5
                and timing_ci_tt[0] - 1e-5 <= solution_vector[3] <= timing_ci_tt[1] + 1e-5
            )
            if temporally_valid:
                solution_distance = spatial_distance(solution.x)
                if solution_distance < closest_feasible_mahalanobis2:
                    closest_feasible = np.r_[np.exp(solution.x), solution_vector]
                    closest_feasible_mahalanobis2 = float(solution_distance)
                    closest_feasible_source = "continuous constrained optimization"
    else:
        feasible_mahalanobis2 = np.asarray([], dtype=float)
        feasible_inside_observed_95 = 0
        closest_feasible = np.full(6, np.nan, dtype=float)
        closest_feasible_mahalanobis2 = float("nan")
        closest_feasible_source = "none"

    cv_folds = []
    cv_model_matrices = []
    cv_observed_matrices = []
    cv_baseline_matrices = []
    cv_model_errors = []
    cv_baseline_errors = []
    cv_model_gains = []
    cv_data_gains = []
    cv_model_slopes = []
    cv_data_slopes = []
    for held_out in subs:
        held_out = str(held_out)
        train_subjects = [str(s) for s in subs if str(s) != held_out]
        train_coefs = [coef_by_subject[s] for s in train_subjects]
        train_matrix = np.array(
            [
                [np.mean([c["beta_RR"] for c in train_coefs]), np.mean([c["beta_RT"] for c in train_coefs])],
                [np.mean([c["beta_TR"] for c in train_coefs]), np.mean([c["beta_TT"] for c in train_coefs])],
            ],
            dtype=float,
        )
        train_rows = [row for row in matched if str(row["subject"]) != held_out]
        _, train_r, train_t = perceived_grid(train_rows)
        fold_fit = _fit_effort_matrix_full(
            train_r,
            train_t,
            train_matrix,
            initial=(effort_best["lam"], effort_best["w"]),
        )

        held_rows = [row for row in matched if str(row["subject"]) == held_out]
        _, held_r, held_t = perceived_grid(held_rows)
        predicted_matrix = np.asarray(
            effort_matrix(held_r, held_t, fold_fit["lam"], fold_fit["w"], iters=400), dtype=float
        ).reshape(2, 2)
        held_coef = coef_by_subject[held_out]
        observed_matrix = np.array(
            [
                [held_coef["beta_RR"], held_coef["beta_RT"]],
                [held_coef["beta_TR"], held_coef["beta_TT"]],
            ],
            dtype=float,
        )
        model_error = _matrix_error(predicted_matrix, observed_matrix)
        baseline_error = _matrix_error(train_matrix, observed_matrix)

        target_t = np.asarray([round(row["stim_t"], 2) for row in held_rows], dtype=float)
        held_r_rows = np.asarray([row["perceived_r"] for row in held_rows], dtype=float)
        held_t_rows = np.asarray([row["perceived_t"] for row in held_rows], dtype=float)
        predicted_gain = np.asarray(
            effort_spatial_gain_by_duration(
                held_r_rows,
                held_t_rows,
                target_t,
                Ts,
                fold_fit["lam"],
                fold_fit["w"],
                iters=400,
            ),
            dtype=float,
        )
        observed_gain = np.asarray([gain_at(rows_by_st.get((held_out, t), [])) for t in Ts], dtype=float)
        perceived_t_bins = np.asarray(
            [np.nanmean([r["perceived_t"] for r in rows_by_st.get((held_out, t), [])]) for t in Ts], dtype=float
        )
        slope_valid = np.isfinite(predicted_gain + observed_gain + perceived_t_bins)
        if int(slope_valid.sum()) >= 4:
            predicted_slope = float(np.polyfit(perceived_t_bins[slope_valid], predicted_gain[slope_valid], 1)[0])
            observed_slope = float(np.polyfit(perceived_t_bins[slope_valid], observed_gain[slope_valid], 1)[0])
            cv_model_slopes.append(predicted_slope)
            cv_data_slopes.append(observed_slope)
        else:
            predicted_slope = observed_slope = float("nan")

        cv_model_matrices.append(predicted_matrix)
        cv_observed_matrices.append(observed_matrix)
        cv_baseline_matrices.append(train_matrix)
        cv_model_errors.append(model_error)
        cv_baseline_errors.append(baseline_error)
        cv_model_gains.append(predicted_gain)
        cv_data_gains.append(observed_gain)
        cv_folds.append(
            dict(
                subject=held_out,
                lambda_=float(fold_fit["lam"]),
                w=float(fold_fit["w"]),
                model_matrix=predicted_matrix,
                observed_matrix=observed_matrix,
                baseline_matrix=train_matrix,
                model_error=model_error,
                baseline_error=baseline_error,
                model_gain=predicted_gain,
                observed_gain=observed_gain,
                model_slope=predicted_slope,
                observed_slope=observed_slope,
            )
        )

    cv_model_matrices = np.asarray(cv_model_matrices, dtype=float)
    cv_observed_matrices = np.asarray(cv_observed_matrices, dtype=float)
    cv_baseline_matrices = np.asarray(cv_baseline_matrices, dtype=float)
    cv_model_errors = np.asarray(cv_model_errors, dtype=float)
    cv_baseline_errors = np.asarray(cv_baseline_errors, dtype=float)
    cv_model_gains = np.asarray(cv_model_gains, dtype=float)
    cv_data_gains = np.asarray(cv_data_gains, dtype=float)
    cv_model_slopes = np.asarray(cv_model_slopes, dtype=float)
    cv_data_slopes = np.asarray(cv_data_slopes, dtype=float)
    cv_error_p = float(_legacy_wilcoxon(cv_model_errors, cv_baseline_errors)[1])
    cv_error_p_adj = cv_error_p  # Single planned headline comparison in the M2 LOSO error subpanel.
    cv_slope_p = float(_legacy_wilcoxon(cv_model_slopes, cv_data_slopes)[1])
    cv_model_mean_gain = np.nanmean(cv_model_gains, axis=1)
    cv_data_mean_gain = np.nanmean(cv_data_gains, axis=1)
    cv_gain_level_p = float(_legacy_wilcoxon(cv_model_mean_gain, cv_data_mean_gain)[1])
    matrix_labels = ["beta_RR^M", "beta_RT^M", "beta_TR^M", "beta_TT^M"]
    cv_model_matrix_residuals = (cv_model_matrices - cv_observed_matrices).reshape(-1, 4)
    cv_baseline_matrix_residuals = (cv_baseline_matrices - cv_observed_matrices).reshape(-1, 4)
    cv_matrix_bias_p = np.asarray(
        [_legacy_wilcoxon(cv_model_matrix_residuals[:, i])[1] for i in range(4)], dtype=float
    )
    cv_matrix_bias_p_holm = _holm_adjust(cv_matrix_bias_p)
    cv_matrix_abs_error_p = np.asarray(
        [
            _legacy_wilcoxon(
                np.abs(cv_model_matrix_residuals[:, i]),
                np.abs(cv_baseline_matrix_residuals[:, i]),
            )[1]
            for i in range(4)
        ],
        dtype=float,
    )
    cv_matrix_abs_error_p_holm = _holm_adjust(cv_matrix_abs_error_p)
    cv_gain_residuals = cv_model_gains - cv_data_gains
    cv_gain_residual_p = np.asarray(
        [
            _legacy_wilcoxon(cv_gain_residuals[np.isfinite(cv_gain_residuals[:, i]), i])[1]
            for i in range(len(Ts))
        ],
        dtype=float,
    )
    cv_gain_residual_p_holm = _holm_adjust(cv_gain_residual_p)
    model_gain = np.nanmedian(cv_model_gains, axis=0)
    model_gain_lo, model_gain_hi = np.nanpercentile(cv_model_gains, [25, 75], axis=0)
    model_slope = float(np.nanmedian(cv_model_slopes))

    # ---------- Panel B: effort feasible locus + measured scatter ----------
    lam_sweep = np.linspace(0.001, 6, 400)
    locus = np.array([effort_matrix(Rperc_g, Tperc_g, l, best_w)[:2] for l in lam_sweep])
    idx = int(np.argmin(np.abs(locus[:, 0] - mRR)))
    effort_bRT = float(locus[idx, 1])

    # ---------- Panel D: cross-trial effort-sign diagnostic ----------
    def zscore(a):
        a = np.asarray(a, float); s = a.std()
        return (a - a.mean()) / s if s > 1e-9 else a * 0.0

    cost_vars = {"Peak\nspeed": peak, "Kinetic\ncost": e_kin, "Jerk\ncost": e_jerk}
    cost_coefs = {k: [] for k in cost_vars}
    cost_crossfit_cells = []
    for s in subs:
        direction_rows = {0: [], 1: []}
        for a in Rs:
            for t in Ts:
                indices = np.flatnonzero(
                    (subj == s)
                    & (np.abs(sr - a) < 1e-6)
                    & (np.abs(st - t) < 1e-6)
                    & np.isfinite(perc_r)
                    & np.isfinite(perc_t)
                    & np.isfinite(prod_r)
                )
                if indices.size < MIN_CELL:
                    continue
                halves = (indices[::2], indices[1::2])
                if min(len(halves[0]), len(halves[1])) < 3:
                    continue
                for direction, (cost_idx, radius_idx) in enumerate((halves, halves[::-1])):
                    row = {
                        "perceived_r": float(np.nanmedian(perc_r[indices])),
                        "perceived_t": float(np.nanmedian(perc_t[indices])),
                        "produced_r": float(np.nanmedian(prod_r[radius_idx])),
                    }
                    for name, arr in cost_vars.items():
                        row[name] = float(np.nanmedian(arr[cost_idx]))
                    direction_rows[direction].append(row)

        subject_betas = {name: [] for name in cost_vars}
        for rows in direction_rows.values():
            if len(rows) < 8:
                continue
            y = zscore([row["produced_r"] for row in rows])
            x_r = zscore([row["perceived_r"] for row in rows])
            x_t = zscore([row["perceived_t"] for row in rows])
            for name in cost_vars:
                x_cost = zscore([row[name] for row in rows])
                design = np.column_stack([np.ones(len(rows)), x_r, x_t, x_cost])
                if np.linalg.matrix_rank(design) == design.shape[1]:
                    subject_betas[name].append(float(np.linalg.lstsq(design, y, rcond=None)[0][3]))
        if all(subject_betas[name] for name in cost_vars):
            cost_crossfit_cells.append(min(len(rows) for rows in direction_rows.values()))
            for name in cost_vars:
                cost_coefs[name].append(float(np.mean(subject_betas[name])))
    cost_stats = {k: (float(np.median(v)), float(_legacy_wilcoxon(v)[1]) if len(v) > 1 else np.nan) for k, v in cost_coefs.items()}

    # Continuous directional anisotropy of spatial gain, tested by within-condition permutation.
    def harmonic_gain_stat(x_r, x_t, y_r, theta):
        harmonics = np.column_stack([np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta)])
        base = np.column_stack([np.ones(len(y_r)), x_r, x_t, harmonics])
        full = np.column_stack([base, x_r[:, None] * harmonics])
        residual_base = y_r - base @ np.linalg.lstsq(base, y_r, rcond=None)[0]
        beta_full = np.linalg.lstsq(full, y_r, rcond=None)[0]
        residual_full = y_r - full @ beta_full
        total = max(float(np.sum((y_r - np.mean(y_r)) ** 2)), 1e-12)
        delta_r2 = float((np.sum(residual_base**2) - np.sum(residual_full**2)) / total)
        interaction = beta_full[-4:]
        amplitude = float(np.sqrt(np.sum(interaction**2)))
        return delta_r2, amplitude

    aniso_subject_data = []
    for s in subs:
        m = (
            (subj == s)
            & np.isfinite(perc_r)
            & np.isfinite(perc_t)
            & np.isfinite(prod_r)
            & np.isfinite(start_dir)
        )
        if int(m.sum()) < 80:
            continue
        x_r = normalize(perc_r[m], R_C, R_H)
        x_t = normalize(perc_t[m], T_C, T_H)
        y_r = normalize(prod_r[m], R_C, R_H)
        theta = start_dir[m]
        condition = np.column_stack([np.round(sr[m], 6), np.round(st[m], 6)])
        _, inverse = np.unique(condition, axis=0, return_inverse=True)
        groups = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
        observed_delta, amplitude = harmonic_gain_stat(x_r, x_t, y_r, theta)
        aniso_subject_data.append(
            dict(subject=str(s), x_r=x_r, x_t=x_t, y_r=y_r, theta=theta, groups=groups,
                 delta_r2=observed_delta, amplitude=amplitude)
        )

    aniso_delta_r2 = np.asarray([row["delta_r2"] for row in aniso_subject_data], dtype=float)
    aniso_amplitude = np.asarray([row["amplitude"] for row in aniso_subject_data], dtype=float)
    aniso_observed = float(np.median(aniso_delta_r2)) if aniso_delta_r2.size else float("nan")
    n_aniso_permutations = 1000
    rng_aniso = np.random.RandomState(1701)
    aniso_null = np.empty(n_aniso_permutations, dtype=float)
    for permutation in range(n_aniso_permutations):
        permuted_stats = []
        for row in aniso_subject_data:
            theta_perm = row["theta"].copy()
            for indices in row["groups"]:
                theta_perm[indices] = rng_aniso.permutation(theta_perm[indices])
            delta_r2, _ = harmonic_gain_stat(row["x_r"], row["x_t"], row["y_r"], theta_perm)
            permuted_stats.append(delta_r2)
        aniso_null[permutation] = np.median(permuted_stats)
    aniso_p = float((1 + np.sum(aniso_null >= aniso_observed)) / (n_aniso_permutations + 1))

    # ---------- Panel D: compression vs produced-noise ----------
    comp = {c["subject"]: (1 - c["beta_RR"], 1 - c["beta_TT"]) for c in coefs}
    prod_by = {}
    prod_t = np.array([r["produced_t"] for r in recs], float)
    for i in range(len(recs)):
        prod_by.setdefault((subj[i], round(sr[i], 2), round(st[i], 2)), []).append((prod_r[i], prod_t[i]))
    scv = {}
    for (s, a, t), vals in prod_by.items():
        arr = np.array(vals)
        if arr.shape[0] >= 4:
            scv.setdefault(s, []).append((arr[:, 0].std() / arr[:, 0].mean(), arr[:, 1].std() / arr[:, 1].mean()))
    CVR, CVT, CR, CT = [], [], [], []
    for s, lst in scv.items():
        if s in comp:
            arr = np.array(lst)
            CVR.append(arr[:, 0].mean()); CVT.append(arr[:, 1].mean())
            CR.append(comp[s][0]); CT.append(comp[s][1])
    CVR, CVT, CR, CT = map(np.array, (CVR, CVT, CR, CT))
    rhoR, pR_ = stats.spearmanr(CR, CVR)
    rhoT_, pT_2 = stats.spearmanr(CT, CVT)
    pearsonR, pearson_pR = stats.pearsonr(CR, CVR)
    pearsonT, pearson_pT = stats.pearsonr(CT, CVT)
    fitR = np.polyfit(CVR, CR, 1); fitT = np.polyfit(CVT, CT, 1)
    p_noise = float(_legacy_wilcoxon(CVR, CVT)[1])
    noise_matrix, noise_scale = _noise_prediction_matrix(CVR, CVT, CR, CT)

    # ---------- Panel E: 2/3 law family + K constancy ----------
    # x = produced radius (actual curvature radius; the 2/3 law is a kinematic law
    # of the executed trajectory, so it must use produced not perceived radius).
    fam = _family_ci(subj, sr, st, Rs, Ts, subs, prod_r, mean_sp)
    subj_23 = []
    for s in subs:
        tsl = []
        for t in Ts:
            xr, yr = [], []
            for a in Rs:
                m = (subj == s) & (np.abs(st - t) < 1e-6) & (np.abs(sr - a) < 1e-6)
                if m.sum() >= MIN_LEVEL:
                    xr.append(float(np.median(prod_r[m]))); yr.append(float(np.median(mean_sp[m])))
            if len(xr) >= 3:
                tsl.append(float(np.polyfit(np.log10(xr), np.log10(yr), 1)[0]))
        if tsl:
            subj_23.append(float(np.median(tsl)))
    subj_23 = np.array(subj_23)
    slope_med = float(np.median(subj_23))
    p_vs_1 = float(_legacy_wilcoxon(subj_23 - 1.0)[1]); p_vs_third = float(_legacy_wilcoxon(subj_23 - 1.0 / 3.0)[1])
    residual = float((1.0 - slope_med) / (1.0 - 1.0 / 3.0))
    # Subject-level K modulation by perceived duration, controlling produced radius.
    k_subject_slopes = []
    for s in subs:
        log_k, log_t, log_r = [], [], []
        for a in Rs:
            for t in Ts:
                m = (
                    (subj == s)
                    & (np.abs(st - t) < 1e-6)
                    & (np.abs(sr - a) < 1e-6)
                    & (mean_sp > 0)
                    & (prod_r > 0)
                    & (perc_t > 0)
                    & np.isfinite(perc_t)
                )
                if int(m.sum()) < MIN_LEVEL:
                    continue
                radius = float(np.median(prod_r[m]))
                speed = float(np.median(mean_sp[m]))
                perceived_duration = float(np.median(perc_t[m]))
                log_k.append(math.log10(speed / radius ** (1.0 / 3.0)))
                log_t.append(math.log10(perceived_duration))
                log_r.append(math.log10(radius))
        if len(log_k) >= 8:
            design = np.column_stack([np.ones(len(log_k)), log_t, log_r])
            if np.linalg.matrix_rank(design) == design.shape[1]:
                k_subject_slopes.append(float(np.linalg.lstsq(design, np.asarray(log_k), rcond=None)[0][1]))
    k_subject_slopes = np.asarray(k_subject_slopes, dtype=float)
    k_slope = float(np.median(k_subject_slopes))
    k_slope_p = float(_legacy_wilcoxon(k_subject_slopes)[1])
    k_slope_ci = _boot_ci(k_subject_slopes, seed=23)

    # ---------- Panel E: produced duration versus produced radius ----------
    famT = _family_ci(subj, sr, st, Rs, Ts, subs, prod_r, prod_t)
    subj_TR = []
    for s in subs:
        sl = []
        for t in Ts:
            xr, yr = [], []
            for a in Rs:
                m = (subj == s) & (np.abs(st - t) < 1e-6) & (np.abs(sr - a) < 1e-6)
                if m.sum() >= MIN_LEVEL and np.all(np.isfinite(prod_r[m])):
                    xr.append(float(np.median(prod_r[m]))); yr.append(float(np.median(prod_t[m])))
            if len(xr) >= 3 and np.ptp(np.log10(xr)) > 1e-9:
                sl.append(float(np.polyfit(np.log10(xr), np.log10(yr), 1)[0]))
        if sl:
            subj_TR.append(float(np.median(sl)))
    subj_TR = np.array(subj_TR)
    timeR_slope = float(np.median(subj_TR))
    timeR_p0 = float(_legacy_wilcoxon(subj_TR)[1])
    timeR_p23 = float(_legacy_wilcoxon(subj_TR - 2.0 / 3.0)[1])
    twothirds_matrix = _twothirds_prediction_matrix(Rperc_g, Tperc_g)

    return dict(
        n_subj=len(subs), n_trials=len(recs), Ts=Ts, pT_bin=pT_bin, serial=serial,
        serial_pooled_blocked=serial_pooled_blocked,
        B_obs=B_obs, noise_matrix=noise_matrix, noise_scale=noise_scale,
        effort_best=effort_best, twothirds_matrix=twothirds_matrix,
        bRR=bRR, bRT=bRT, mRR=mRR, mRT=mRT, mTR=mTR, mTT=mTT,
        gain=gain, gain_med=gain_med, model_gain=model_gain, emp_slope=emp_slope,
        emp_slope_p=emp_slope_p, model_slope=model_slope,
        model_gain_lo=model_gain_lo, model_gain_hi=model_gain_hi, peak=peak, st=st,
        peak_ratio=peak_ratio, locus=locus, effort_bRT=effort_bRT,
        effort_grid=effort_grid, effort_feasible=effort_feasible,
        timing_ci_tr=timing_ci_tr, timing_ci_tt=timing_ci_tt,
        spatial_bootstrap=spatial_bootstrap, spatial_bootstrap_cov=spatial_bootstrap_cov,
        feasible_mahalanobis2=feasible_mahalanobis2,
        feasible_inside_observed_95=feasible_inside_observed_95,
        closest_feasible=closest_feasible,
        closest_feasible_mahalanobis2=closest_feasible_mahalanobis2,
        closest_feasible_source=closest_feasible_source,
        cv_folds=cv_folds, cv_model_matrices=cv_model_matrices, cv_observed_matrices=cv_observed_matrices,
        cv_baseline_matrices=cv_baseline_matrices,
        cv_model_errors=cv_model_errors, cv_baseline_errors=cv_baseline_errors,
        cv_error_p=cv_error_p, cv_error_p_adj=cv_error_p_adj,
        matrix_labels=matrix_labels,
        cv_model_matrix_residuals=cv_model_matrix_residuals,
        cv_baseline_matrix_residuals=cv_baseline_matrix_residuals,
        cv_matrix_bias_p=cv_matrix_bias_p, cv_matrix_bias_p_holm=cv_matrix_bias_p_holm,
        cv_matrix_abs_error_p=cv_matrix_abs_error_p,
        cv_matrix_abs_error_p_holm=cv_matrix_abs_error_p_holm,
        cv_model_gains=cv_model_gains, cv_data_gains=cv_data_gains,
        cv_gain_residuals=cv_gain_residuals,
        cv_gain_residual_p=cv_gain_residual_p,
        cv_gain_residual_p_holm=cv_gain_residual_p_holm,
        cv_model_slopes=cv_model_slopes, cv_data_slopes=cv_data_slopes, cv_slope_p=cv_slope_p,
        cv_model_mean_gain=cv_model_mean_gain, cv_data_mean_gain=cv_data_mean_gain,
        cv_gain_level_p=cv_gain_level_p,
        cost_coefs=cost_coefs, cost_stats=cost_stats, cost_crossfit_cells=np.asarray(cost_crossfit_cells, dtype=int),
        aniso_subject_ids=[row["subject"] for row in aniso_subject_data],
        aniso_delta_r2=aniso_delta_r2, aniso_amplitude=aniso_amplitude,
        aniso_observed=aniso_observed, aniso_null=aniso_null,
        aniso_p=aniso_p, n_aniso_permutations=n_aniso_permutations,
        CVR=CVR, CVT=CVT, CR=CR, CT=CT, rhoR=float(rhoR), pR=float(pR_),
        rhoT=float(rhoT_), pT=float(pT_2),
        pearsonR=float(pearsonR), pearson_pR=float(pearson_pR),
        pearsonT=float(pearsonT), pearson_pT=float(pearson_pT),
        fitR=fitR, fitT=fitT, p_noise=p_noise,
        fam=fam, slope_med=slope_med, p_vs_1=p_vs_1, p_vs_third=p_vs_third,
        residual=residual, k_slope=k_slope, k_subject_slopes=k_subject_slopes,
        k_slope_p=k_slope_p, k_slope_ci=k_slope_ci,
        famT=famT, timeR_slope=timeR_slope, timeR_p0=timeR_p0, timeR_p23=timeR_p23,
    )


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
def _make_figure_legacy_unused(D):
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    # Layout: row1 = M1 noise, M2 effort-direct, M2 effort-interaction;
    # row2 = M2 execution, M3 speed, M3 time. The variable names keep their
    # content (axA=direct, axB=interaction, axC=execution, axD=noise, axE=speed,
    # axF=time); only their physical slots are remapped so reading order is
    # M1 noise -> M2 effort(direct, interaction, execution) -> M3.
    axD, axA, axB = axes[0]
    axC, axE, axF = axes[1]

    def tidy(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    Ts = D["Ts"]; pT = D["pT_bin"]

    # ===== Panel A: M1 direct =====
    tidy(axA)
    gdata = [np.array(D["gain"][t]) for t in Ts]
    bp = axA.boxplot(gdata, positions=range(len(Ts)), widths=0.5, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor=C_R, alpha=0.35, edgecolor="black", linewidth=0.8)
    for e in bp["medians"]:
        e.set(color="black", linewidth=1.2)
    for e in bp["whiskers"] + bp["caps"]:
        e.set(color="black", linewidth=0.8)
    rng = np.random.RandomState(0)
    for i, g in enumerate(gdata):
        axA.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=8, color=C_R, edgecolor="white", linewidth=0.3, zorder=3)
    axA.plot(range(len(Ts)), D["model_gain"], "--o", color=C_MODEL, markersize=4, linewidth=1.4, label="Effort model", zorder=4)
    axA.axhline(1, color="0.6", linestyle=":", linewidth=0.8)
    axA.set_xticks(range(len(Ts))); axA.set_xticklabels([f"{t:.1f}" for t in pT])
    axA.set_xlabel("Perceived Duration (s)"); axA.set_ylabel("Spatial Gain (Produced/Perceived R)")
    axA.set_title("M2 Effort — Direct Level")
    axA.legend(loc="upper left", frameon=False, handlelength=1.5)
    n_ctrl = int(np.sum(D["dBIC"] > 0))
    axA.text(0.97, 0.04, f"Data {D['emp_slope']:+.3f}/s vs effort {D['model_slope']:+.3f}/s\n"
                         f"({D['model_slope'] / D['emp_slope']:.1f}x steeper)\n"
                         f"dBIC favours constant gain: {n_ctrl}/{len(D['dBIC'])}",
             transform=axA.transAxes, va="bottom", ha="right", fontsize=8)
    axA.set_ylim(0.28, 1.40); axA.set_yticks(np.linspace(0.4, 1.2, 5))
    # inset: speed manipulation check
    axAi = axA.inset_axes([0.10, 0.04, 0.27, 0.24])
    sp = [D["peak"][np.abs(D["st"] - t) < 1e-6] for t in Ts]
    axAi.plot(range(len(Ts)), [np.median(x) for x in sp], "-o", color=C_T, markersize=3, linewidth=1.0)
    axAi.set_xticks([0, len(Ts) - 1]); axAi.set_xticklabels([f"{pT[0]:.1f}", f"{pT[-1]:.1f}"], fontsize=6)
    axAi.tick_params(labelsize=6)
    axAi.set_xlabel("Perceived T (s)", fontsize=6, labelpad=1)
    axAi.set_ylabel("Peak Speed", fontsize=6, labelpad=1)
    axAi.set_title(f"Manipulation ×{D['peak_ratio']:.1f}", fontsize=6)
    axAi.spines["top"].set_visible(False)
    axAi.spines["right"].set_visible(False)

    # ===== Panel B: M1 interaction =====
    tidy(axB)
    axB.plot(D["locus"][:, 0], D["locus"][:, 1], color=C_MODEL, lw=1.6, label="Effort locus")
    axB.scatter(D["bRR"], D["bRT"], s=15, color=C_R, alpha=0.55, edgecolor="white", linewidth=0.3, zorder=3)
    axB.scatter([D["mRR"]], [D["mRT"]], s=70, color=C_R, edgecolor="black", linewidth=0.8, zorder=5, label="Measured")
    axB.scatter([D["mRR"]], [D["effort_bRT"]], s=70, color=C_MODEL, edgecolor="black", linewidth=0.8, marker="D", zorder=5, label="Effort at measured $\\beta_{RR}$")
    axB.plot([D["mRR"], D["mRR"]], [D["mRT"], D["effort_bRT"]], color="0.5", lw=0.8, ls=":")
    axB.annotate(f"{D['effort_bRT'] / D['mRT']:.1f}x", xy=(D["mRR"], (D["mRT"] + D["effort_bRT"]) / 2),
                 xytext=(D["mRR"] + 0.03, (D["mRT"] + D["effort_bRT"]) / 2), va="center", fontsize=10)
    axB.axhline(0, color="0.7", lw=0.6)
    axB.set_xlabel(r"Spatial Direct Gain $\beta_{RR}$"); axB.set_ylabel(r"Time$\to$Space $\beta_{RT}$")
    axB.set_title("M2 Effort — Interaction Level")
    axB.axvline(1.0, color="0.8", lw=0.6, ls=":")
    axB.set_xlim(0.42, 1.28); axB.set_ylim(-0.16, 0.90)
    axB.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.4)

    # ===== Panel C: M1 execution / biomechanics =====
    tidy(axC)
    names = list(D["cost_coefs"].keys())
    pos = range(len(names))
    cdata = [np.array(D["cost_coefs"][n]) for n in names]
    bp = axC.boxplot(cdata, positions=pos, widths=0.5, patch_artist=True, showfliers=False, vert=True)
    for b in bp["boxes"]:
        b.set(facecolor=C_R, alpha=0.3, edgecolor="black", linewidth=0.8)
    for e in bp["medians"]:
        e.set(color="black", linewidth=1.2)
    for e in bp["whiskers"] + bp["caps"]:
        e.set(color="black", linewidth=0.8)
    for i, c in enumerate(cdata):
        axC.scatter(i + rng.uniform(-0.12, 0.12, len(c)), c, s=8, color=C_R, edgecolor="white", linewidth=0.3, zorder=3)
    axC.axhline(0, color="black", lw=0.8)
    axC.set_xticks(list(pos)); axC.set_xticklabels(names, fontsize=8)
    axC.set_ylabel("Partial Coef. On Produced R\n(control perceived R, T)")
    axC.set_title("M2 Effort — Execution / Biomechanics")
    txt = "Effort predicts < 0\n" + "\n".join(
        f"{n.replace(chr(10), ' ')}: {D['cost_stats'][n][0]:+.2f} (p={D['cost_stats'][n][1]:.2f})" for n in names)
    txt += f"\nWithin-cond r(speed,R)={np.median(D['within_r']):+.2f}"
    axC.text(0.03, 0.97, txt, transform=axC.transAxes, va="top", fontsize=7.5)

    # ===== Panel D: M2 noise minimization =====
    tidy(axD)
    xline = np.array([0.10, 0.36])
    axD.plot(xline, 2.2 * (xline - 0.08), color=C_MODEL, lw=1.4, ls="--", label="Min-variance (predicted)")
    axD.scatter(D["CVR"], D["CR"], s=20, color=C_R, alpha=0.6, edgecolor="white", linewidth=0.3, label="Space", zorder=3)
    axD.scatter(D["CVT"], D["CT"], s=20, color=C_T, alpha=0.6, edgecolor="white", linewidth=0.3, label="Time", zorder=3)
    # per-dimension linear fits (SDN requires a positive slope in each)
    xr = np.linspace(D["CVR"].min(), D["CVR"].max(), 10)
    axD.plot(xr, np.polyval(D["fitR"], xr), color=C_R, lw=1.8, zorder=4)
    xt = np.linspace(D["CVT"].min(), D["CVT"].max(), 10)
    axD.plot(xt, np.polyval(D["fitT"], xt), color=C_T, lw=1.8, zorder=4)
    axD.axhline(0, color="0.7", lw=0.6)
    axD.set_xlabel("Produced Noise (CV)"); axD.set_ylabel(r"Compression $1-\beta$")
    axD.set_title("M1 Noise-Driven Compression")
    axD.legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.4)
    axD.text(0.97, 0.97, f"Time noisier (CV {np.median(D['CVT']):.2f} vs {np.median(D['CVR']):.2f}, p={D['p_noise']:.0e})\n"
                         f"yet only space compresses.\nSDN needs positive slope in both:",
             transform=axD.transAxes, va="top", ha="right", fontsize=7.5)
    axD.text(0.97, 0.79, f"space fit: $\\rho$={D['rhoR']:+.2f} (p={D['pR']:.2f})",
             transform=axD.transAxes, va="top", ha="right", fontsize=8, color=C_R, fontweight="bold")
    axD.text(0.97, 0.73, f"time fit: $\\rho$={D['rhoT']:+.2f} (p={D['pT']:.2f})",
             transform=axD.transAxes, va="top", ha="right", fontsize=8, color=C_T, fontweight="bold")
    axD.set_xlim(0.10, 0.37)

    # ===== Panel E: M3 two-thirds law =====
    tidy(axE)
    all_x, all_y = [], []
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["fam"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axE.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        j = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        axE.plot(span, ly[j] + 1.0 * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
        all_x.append(lx); all_y.append(ly)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    xc, yc = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min() - 0.02, all_x.max() + 0.02])
    axE.plot(span, yc + (1.0 / 3.0) * (span - xc), "--", color=C_MODEL, linewidth=1.4, zorder=2)
    handles = [Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
               Line2D([], [], color="0.3", ls=":", lw=1.0, label="Instructed (slope 1)"),
               Line2D([], [], color=C_MODEL, ls="--", lw=1.4, label="Two-thirds (slope 1/3)")]
    axE.set_xlabel("Log10 Produced Radius"); axE.set_ylabel("Log10 Mean Speed")
    axE.set_title("M3 Speed Not Set By Geometry")
    axE.legend(handles=handles, loc="lower right", frameon=False, fontsize=8, handlelength=1.6)
    axE.text(0.03, 0.97, f"Speed–radius slope {D['slope_med']:.2f} ($\\approx$1)\n"
                         f"vs two-thirds (1/3): p={D['p_vs_third']:.0e}\n"
                         f"$\\to$ speed set by duration,\nnot geometry "
                         f"(log K–T {D['k_slope']:+.2f})\n(darker = shorter T)",
             transform=axE.transAxes, va="top", fontsize=7.5)

    # ===== Panel F: M3 time footprint — produced T vs target R (isochrony) =====
    tidy(axF)
    all_x = []
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["famT"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axF.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        # two-thirds / isochrony prediction: geometry sets duration, slope 2/3
        j = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        axF.plot(span, ly[j] + (2.0 / 3.0) * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
        all_x.append(lx)
    handles = [Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
               Line2D([], [], color="0.3", ls=":", lw=1.0, label="Two-thirds (slope 2/3)")]
    axF.set_xlabel("Log10 Perceived Radius"); axF.set_ylabel("Log10 Produced Duration")
    axF.set_title("M3 Time Is Isochronous")
    axF.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, handlelength=1.6)
    axF.text(0.97, 0.03, f"Duration–radius slope {D['timeR_slope']:.2f}\n"
                         f"vs 0 (isochrony): p={D['timeR_p0']:.2f}\n"
                         f"vs two-thirds (2/3): p={D['timeR_p23']:.0e}\n"
                         f"$\\to$ 2/3-law time coupling\noverridden ($\\beta_{{TR}}\\approx$0)",
             transform=axF.transAxes, va="bottom", ha="right", fontsize=7.5)

    for ax, lab in zip([axD, axA, axB, axC, axE, axF], "ABCDEF"):
        label_panel(ax, lab)
    fig.tight_layout()
    png = FIG_DIR / "alternative_behavior_model_legacy.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def make_figure_main_standard(D):
    setup_style()
    fig = plt.figure(figsize=(16.0, 8.8))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.55, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.40,
    )
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[0, 3])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])
    axG = fig.add_subplot(gs[1, 3])

    def tidy(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_matrix_summary(ax):
        ax.axis("off")
        ax.set_title("Alternative Models Fail At The Matrix Level", pad=16)
        ax.text(
            0.50,
            0.975,
            "Rows: produced R/T; columns: control R/T",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        matrices = [
            ("Observed", D["B_obs"], "data"),
            ("M1 Noise", D["noise_matrix"], f"err={_matrix_error(D['noise_matrix'], D['B_obs']):.2f}"),
            ("M2 Effort Best", D["effort_best"]["matrix"], f"err={D['effort_best']['error']:.2f}"),
            ("M3 Two-Thirds", D["twothirds_matrix"], f"err={_matrix_error(D['twothirds_matrix'], D['B_obs']):.2f}"),
        ]
        boxes = [(0.06, 0.59, 0.40, 0.30), (0.56, 0.59, 0.40, 0.30),
                 (0.06, 0.17, 0.40, 0.30), (0.56, 0.17, 0.40, 0.30)]
        im = None
        for (title, mat, footer), box in zip(matrices, boxes):
            iax = ax.inset_axes(box)
            im = iax.imshow(mat, cmap="viridis", vmin=MATRIX_VMIN, vmax=MATRIX_VMAX)
            iax.set_aspect("equal")
            iax.set_title(title, fontsize=9, pad=3)
            iax.set_xticks([0, 1])
            iax.set_yticks([0, 1])
            iax.set_xticklabels(["R", "T"], fontsize=8)
            iax.set_yticklabels(["R", "T"], fontsize=8)
            iax.tick_params(length=0, pad=1)
            for i in range(2):
                for j in range(2):
                    val = float(mat[i, j])
                    contrast = (val - MATRIX_VMIN) / (MATRIX_VMAX - MATRIX_VMIN)
                    txt_col = "white" if contrast < 0.45 else "black"
                    iax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_col)
            for spine in iax.spines.values():
                spine.set_linewidth(0.8)
            ax.text(box[0] + box[2] / 2, box[1] - 0.035, footer, transform=ax.transAxes,
                    ha="center", va="top", fontsize=8)
        cax = ax.inset_axes([0.18, 0.055, 0.64, 0.025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Normalized Coefficient", fontsize=8, labelpad=1)
        cb.ax.tick_params(labelsize=8, length=3)

    draw_matrix_summary(axA)

    Ts = D["Ts"]
    pT = D["pT_bin"]
    rng = np.random.RandomState(0)
    text_box = dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5)

    # Panel B: M1 noise minimization.
    tidy(axB)
    xline = np.array([0.10, 0.36])
    axB.plot(xline, 2.2 * (xline - 0.08), color=C_MODEL, lw=1.4, ls="--", label="Min-variance prediction")
    axB.scatter(D["CVR"], D["CR"], s=22, color=C_R, alpha=0.65, edgecolor="white", linewidth=0.3, label="Space", zorder=3)
    axB.scatter(D["CVT"], D["CT"], s=22, color=C_T, alpha=0.65, edgecolor="white", linewidth=0.3, label="Time", zorder=3)
    xr = np.linspace(D["CVR"].min(), D["CVR"].max(), 10)
    axB.plot(xr, np.polyval(D["fitR"], xr), color=C_R, lw=1.8, zorder=4)
    xt = np.linspace(D["CVT"].min(), D["CVT"].max(), 10)
    axB.plot(xt, np.polyval(D["fitT"], xt), color=C_T, lw=1.8, zorder=4)
    axB.axhline(0, color="0.7", lw=0.6)
    axB.set_xlabel("Produced Noise (CV)")
    axB.set_ylabel(r"Compression $1-\beta$")
    axB.set_title("M1 Noise-Driven Compression")
    axB.legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.4)
    axB.text(
        0.97,
        0.97,
        f"Time noisier (CV {np.median(D['CVT']):.2f} vs {np.median(D['CVR']):.2f}, {_format_p(D['p_noise'])})\n"
        "yet only space compresses.\n"
        f"space: rho={D['rhoR']:+.2f} ({_format_p(D['pR'])})\n"
        f"time: rho={D['rhoT']:+.2f} ({_format_p(D['pT'])})",
        transform=axB.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        bbox=text_box,
    )
    axB.set_xlim(0.10, 0.37)

    # Panel C: M2 effort direct level.
    tidy(axC)
    gdata = [np.array(D["gain"][t]) for t in Ts]
    bp = axC.boxplot(gdata, positions=range(len(Ts)), widths=0.5, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor=C_R, alpha=0.35, edgecolor="black", linewidth=0.8)
    for e in bp["medians"]:
        e.set(color="black", linewidth=1.2)
    for e in bp["whiskers"] + bp["caps"]:
        e.set(color="black", linewidth=0.8)
    for i, g in enumerate(gdata):
        axC.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=8, color=C_R, edgecolor="white", linewidth=0.3, zorder=3)
    axC.plot(range(len(Ts)), D["model_gain"], "--o", color=C_MODEL, markersize=4, linewidth=1.4, label="Effort model", zorder=4)
    axC.axhline(1, color="0.6", linestyle=":", linewidth=0.8)
    axC.set_xticks(range(len(Ts)))
    axC.set_xticklabels([f"{t:.1f}" for t in pT])
    axC.set_xlabel("Perceived Duration (s)")
    axC.set_ylabel("Spatial Gain (Produced/Perceived R)")
    axC.set_title("M2 Effort: Direct Level")
    axC.legend(loc="upper left", frameon=False, handlelength=1.5)
    n_ctrl = int(np.sum(D["dBIC"] > 0))
    axC.text(
        0.97,
        0.04,
        f"Data {D['emp_slope']:+.3f}/s ({_format_p(D['emp_slope_p'])})\n"
        f"Effort {D['model_slope']:+.3f}/s ({D['model_slope'] / D['emp_slope']:.1f}x)\n"
        f"dBIC favours constant gain: {n_ctrl}/{len(D['dBIC'])}\n"
        f"Peak speed manipulation x{D['peak_ratio']:.1f}",
        transform=axC.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        bbox=text_box,
    )
    axC.set_ylim(0.28, 1.40)
    axC.set_yticks(np.linspace(0.4, 1.2, 5))

    # Panel D: M2 effort interaction and full matrix fit.
    tidy(axD)
    best = D["effort_best"]["matrix"]
    axD.plot(D["locus"][:, 0], D["locus"][:, 1], color=C_MODEL, lw=1.6, label="Fixed-T effort locus")
    axD.scatter(D["bRR"], D["bRT"], s=15, color=C_R, alpha=0.55, edgecolor="white", linewidth=0.3, zorder=3)
    axD.scatter([D["mRR"]], [D["mRT"]], s=70, color=C_R, edgecolor="black", linewidth=0.8, zorder=5, label="Measured")
    axD.scatter([D["mRR"]], [D["effort_bRT"]], s=65, color=C_MODEL, edgecolor="black", linewidth=0.8,
                marker="D", zorder=5, label="Effort at measured RR")
    axD.scatter([best[0, 0]], [best[0, 1]], s=70, facecolor="white", edgecolor="black", linewidth=1.0,
                marker="s", zorder=6, label="Best full matrix")
    axD.plot([D["mRR"], D["mRR"]], [D["mRT"], D["effort_bRT"]], color="0.5", lw=0.8, ls=":")
    axD.annotate(f"{D['effort_bRT'] / D['mRT']:.1f}x", xy=(D["mRR"], (D["mRT"] + D["effort_bRT"]) / 2),
                 xytext=(D["mRR"] + 0.03, (D["mRT"] + D["effort_bRT"]) / 2), va="center", fontsize=10)
    axD.axhline(0, color="0.7", lw=0.6)
    axD.axvline(1.0, color="0.8", lw=0.6, ls=":")
    axD.set_xlabel(r"Spatial Direct Gain $\beta_{RR}$")
    axD.set_ylabel(r"Time To Space $\beta_{RT}$")
    axD.set_title("M2 Effort: Matrix Constraint")
    axD.set_xlim(0.42, 1.28)
    axD.set_ylim(-0.16, 0.90)
    axD.legend(loc="upper right", frameon=False, fontsize=7.5, handlelength=1.4)
    axD.text(
        0.04,
        0.04,
        f"Best full-matrix error={D['effort_best']['error']:.2f}\n"
        f"lambda={D['effort_best']['lam']:.2g}, w={D['effort_best']['w']:.2g}",
        transform=axD.transAxes,
        va="bottom",
        fontsize=7.5,
        bbox=text_box,
    )

    # Panel E: M2 execution / biomechanics.
    tidy(axE)
    names = list(D["cost_coefs"].keys())
    pos = range(len(names))
    cdata = [np.array(D["cost_coefs"][n]) for n in names]
    bp = axE.boxplot(cdata, positions=pos, widths=0.5, patch_artist=True, showfliers=False, vert=True)
    for b in bp["boxes"]:
        b.set(facecolor=C_R, alpha=0.3, edgecolor="black", linewidth=0.8)
    for e in bp["medians"]:
        e.set(color="black", linewidth=1.2)
    for e in bp["whiskers"] + bp["caps"]:
        e.set(color="black", linewidth=0.8)
    for i, c in enumerate(cdata):
        axE.scatter(i + rng.uniform(-0.12, 0.12, len(c)), c, s=8, color=C_R, edgecolor="white", linewidth=0.3, zorder=3)
    axE.axhline(0, color="black", lw=0.8)
    axE.set_xticks(list(pos))
    axE.set_xticklabels(names, fontsize=8)
    axE.set_ylabel("Partial Coef. On Produced R\n(control perceived R, T)")
    axE.set_title("M2 Effort: Execution")
    txt = "Effort predicts < 0\n" + "\n".join(
        f"{n.replace(chr(10), ' ')}: {D['cost_stats'][n][0]:+.2f} ({_format_p(D['cost_stats'][n][1])})" for n in names
    )
    txt += f"\nWithin-cond r(speed,R)={np.median(D['within_r']):+.2f}"
    axE.text(0.03, 0.97, txt, transform=axE.transAxes, va="top", fontsize=7.5, bbox=text_box)

    # Panel F: M3 speed-radius arm.
    tidy(axF)
    all_x, all_y = [], []
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["fam"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axF.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        j = int(np.argmin(lx))
        span = np.array([lx.min(), lx.max()])
        axF.plot(span, ly[j] + 1.0 * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
        all_x.append(lx)
        all_y.append(ly)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    xc, yc = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min() - 0.02, all_x.max() + 0.02])
    axF.plot(span, yc + (1.0 / 3.0) * (span - xc), "--", color=C_MODEL, linewidth=1.4, zorder=2)
    handles = [
        Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
        Line2D([], [], color="0.3", ls=":", lw=1.0, label="Instructed slope 1"),
        Line2D([], [], color=C_MODEL, ls="--", lw=1.4, label="Two-thirds slope 1/3"),
    ]
    axF.set_xlabel("Log10 Produced Radius")
    axF.set_ylabel("Log10 Mean Speed")
    axF.set_title("M3 Speed Not Set By Geometry")
    axF.legend(handles=handles, loc="lower right", frameon=False, fontsize=8, handlelength=1.6)
    axF.text(
        0.03,
        0.97,
        f"Speed-radius slope {D['slope_med']:.2f}\n"
        f"vs two-thirds: {_format_p(D['p_vs_third'])}\n"
        f"log K vs T slope {D['k_slope']:+.2f}\n"
        "(darker = shorter T)",
        transform=axF.transAxes,
        va="top",
        fontsize=7.5,
        bbox=text_box,
    )

    # Panel G: M3 duration-radius arm.
    tidy(axG)
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["famT"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axG.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        j = int(np.argmin(lx))
        span = np.array([lx.min(), lx.max()])
        axG.plot(span, ly[j] + (2.0 / 3.0) * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
    handles = [
        Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
        Line2D([], [], color="0.3", ls=":", lw=1.0, label="Two-thirds slope 2/3"),
    ]
    axG.set_xlabel("Log10 Perceived Radius")
    axG.set_ylabel("Log10 Produced Duration")
    axG.set_title("M3 Time Is Isochronous")
    axG.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, handlelength=1.6)
    axG.text(
        0.97,
        0.03,
        f"Duration-radius slope {D['timeR_slope']:.2f}\n"
        f"vs 0: {_format_p(D['timeR_p0'])}\n"
        f"vs two-thirds: {_format_p(D['timeR_p23'])}\n"
        r"$\beta_{TR}\approx0$",
        transform=axG.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.5,
        bbox=text_box,
    )

    for ax, lab in zip([axA, axB, axC, axD, axE, axF, axG], "ABCDEFG"):
        label_panel(ax, lab)
    png = FIG_DIR / "alternative_behavior_model_legacy.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def make_figure_2x4(D):
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.4))
    axA, axB, axC, axD = axes[0]
    axE, axF, axG, axH = axes[1]
    text_box = dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5)

    def tidy(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def draw_matrix(ax, mat, title, footer=None, colorbar=False):
        im = ax.imshow(mat, cmap="viridis", vmin=MATRIX_VMIN, vmax=MATRIX_VMAX)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["R", "T"])
        ax.set_yticklabels(["R", "T"])
        ax.tick_params(length=0, pad=2)
        ax.set_xlabel("Control Input")
        ax.set_ylabel("Produced Output")
        for i in range(2):
            for j in range(2):
                val = float(mat[i, j])
                contrast = (val - MATRIX_VMIN) / (MATRIX_VMAX - MATRIX_VMIN)
                txt_col = "white" if contrast < 0.45 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=txt_col)
        if footer:
            ax.text(0.50, -0.28, footer, transform=ax.transAxes, ha="center", va="top", fontsize=8)
        if colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Normalized Coefficient", fontsize=8)
            cb.ax.tick_params(labelsize=8)
        return im

    Ts = D["Ts"]
    pT = D["pT_bin"]
    rng = np.random.RandomState(0)

    # A. Define the behavior matrix to be explained.
    draw_matrix(axA, D["B_obs"], "Observed Control Matrix", colorbar=True)
    axA.text(
        0.50,
        -0.46,
        r"$\beta_{RR}<1$, $\beta_{TT}\approx1$, $\beta_{RT}>0$, $\beta_{TR}\approx0$",
        transform=axA.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    # B. M1 noise minimization.
    tidy(axB)
    xline = np.array([0.10, 0.36])
    axB.plot(xline, 2.2 * (xline - 0.08), color=C_MODEL, lw=1.4, ls="--", label="Min-variance prediction")
    axB.scatter(D["CVR"], D["CR"], s=22, color=C_R, alpha=0.65, edgecolor="white", linewidth=0.3, label="Space", zorder=3)
    axB.scatter(D["CVT"], D["CT"], s=22, color=C_T, alpha=0.65, edgecolor="white", linewidth=0.3, label="Time", zorder=3)
    xr = np.linspace(D["CVR"].min(), D["CVR"].max(), 10)
    xt = np.linspace(D["CVT"].min(), D["CVT"].max(), 10)
    axB.plot(xr, np.polyval(D["fitR"], xr), color=C_R, lw=1.8, zorder=4)
    axB.plot(xt, np.polyval(D["fitT"], xt), color=C_T, lw=1.8, zorder=4)
    axB.axhline(0, color="0.7", lw=0.6)
    axB.set_xlabel("Produced Noise (CV)")
    axB.set_ylabel(r"Compression $1-\beta$")
    axB.set_title("M1 Noise-Driven Compression")
    axB.legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.4)
    axB.text(
        0.97,
        0.97,
        f"Time noisier ({_format_p(D['p_noise'])})\n"
        "but space compresses.\n"
        f"space rho={D['rhoR']:+.2f} ({_format_p(D['pR'])})\n"
        f"time rho={D['rhoT']:+.2f} ({_format_p(D['pT'])})",
        transform=axB.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        bbox=text_box,
    )
    axB.set_xlim(0.10, 0.37)

    # C. M2 effort tradeoff.
    tidy(axC)
    best = D["effort_best"]["matrix"]
    axC.plot(D["locus"][:, 0], D["locus"][:, 1], color=C_MODEL, lw=1.6, label="Fixed-T effort locus")
    axC.scatter(D["bRR"], D["bRT"], s=15, color=C_R, alpha=0.55, edgecolor="white", linewidth=0.3, zorder=3)
    axC.scatter([D["mRR"]], [D["mRT"]], s=70, color=C_R, edgecolor="black", linewidth=0.8, zorder=5, label="Measured")
    axC.scatter([D["mRR"]], [D["effort_bRT"]], s=65, color=C_MODEL, edgecolor="black", linewidth=0.8,
                marker="D", zorder=5, label="Effort at measured RR")
    axC.scatter([best[0, 0]], [best[0, 1]], s=70, facecolor="white", edgecolor="black", linewidth=1.0,
                marker="s", zorder=6, label="Best full matrix")
    axC.plot([D["mRR"], D["mRR"]], [D["mRT"], D["effort_bRT"]], color="0.5", lw=0.8, ls=":")
    axC.annotate(f"{D['effort_bRT'] / D['mRT']:.1f}x", xy=(D["mRR"], (D["mRT"] + D["effort_bRT"]) / 2),
                 xytext=(D["mRR"] + 0.03, (D["mRT"] + D["effort_bRT"]) / 2), va="center", fontsize=10)
    axC.axhline(0, color="0.7", lw=0.6)
    axC.axvline(1.0, color="0.8", lw=0.6, ls=":")
    axC.set_xlabel(r"Spatial Direct Gain $\beta_{RR}$")
    axC.set_ylabel(r"Time To Space $\beta_{RT}$")
    axC.set_title("M2 Effort Tradeoff")
    axC.set_xlim(0.42, 1.28)
    axC.set_ylim(-0.16, 0.90)
    axC.legend(loc="upper right", frameon=False, fontsize=7.3, handlelength=1.3)
    axC.text(
        0.04,
        0.04,
        f"Best matrix error={D['effort_best']['error']:.2f}\n"
        f"best RR={best[0, 0]:.2f}, RT={best[0, 1]:.2f}",
        transform=axC.transAxes,
        va="bottom",
        fontsize=7.5,
        bbox=text_box,
    )

    # D. M2 execution / biomechanics.
    tidy(axD)
    names = list(D["cost_coefs"].keys())
    pos = range(len(names))
    cdata = [np.array(D["cost_coefs"][n]) for n in names]
    axD.set_ylim(-0.36, 0.68)
    axD.axhspan(-0.36, 0.0, facecolor=C_MODEL, alpha=0.18, edgecolor="none", zorder=0)
    bp = axD.boxplot(cdata, positions=pos, widths=0.5, patch_artist=True, showfliers=False, vert=True)
    for b in bp["boxes"]:
        b.set(facecolor=C_R, alpha=0.3, edgecolor="black", linewidth=0.8)
    for e in bp["medians"]:
        e.set(color="black", linewidth=1.2)
    for e in bp["whiskers"] + bp["caps"]:
        e.set(color="black", linewidth=0.8)
    for i, c in enumerate(cdata):
        axD.scatter(i + rng.uniform(-0.12, 0.12, len(c)), c, s=8, color=C_R, edgecolor="white", linewidth=0.3, zorder=3)
    axD.axhline(0, color="black", lw=0.8)
    axD.set_ylim(-0.36, 0.78)
    axD.set_yticks([-0.3, 0.0, 0.3, 0.6])
    axD.set_xticks(list(pos))
    axD.set_xticklabels(names, fontsize=8)
    axD.set_ylabel("Partial Coef. On Produced R\n(control perceived R, T)")
    axD.set_title("M2 Execution Check")
    txt = "Effort predicts < 0\n" + "\n".join(
        f"{n.replace(chr(10), ' ')}: {D['cost_stats'][n][0]:+.2f} ({_format_p(D['cost_stats'][n][1])})" for n in names
    )
    txt += f"\nWithin-cond r(speed,R)={np.median(D['within_r']):+.2f}"
    axD.text(0.03, 0.97, txt, transform=axD.transAxes, va="top", fontsize=7.5, bbox=text_box)

    # E. M3 duration-radius arm.
    tidy(axE)
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["famT"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axE.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        j = int(np.argmin(lx))
        span = np.array([lx.min(), lx.max()])
        axE.plot(span, ly[j] + (2.0 / 3.0) * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
    handles = [
        Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
        Line2D([], [], color="0.3", ls=":", lw=1.0, label="Two-thirds slope 2/3"),
    ]
    axE.set_xlabel("Log10 Perceived Radius")
    axE.set_ylabel("Log10 Produced Duration")
    axE.set_title("M3 Time Is Isochronous")
    axE.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, handlelength=1.6)
    axE.text(
        0.97,
        0.03,
        f"Duration-radius slope {D['timeR_slope']:.2f}\n"
        f"vs 0: {_format_p(D['timeR_p0'])}\n"
        f"vs two-thirds: {_format_p(D['timeR_p23'])}\n"
        r"$\beta_{TR}\approx0$",
        transform=axE.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.5,
        bbox=text_box,
    )

    # F. M3 speed-radius diagnostic.
    tidy(axF)
    all_x, all_y = [], []
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["fam"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axF.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        j = int(np.argmin(lx))
        span = np.array([lx.min(), lx.max()])
        axF.plot(span, ly[j] + 1.0 * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
        all_x.append(lx)
        all_y.append(ly)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    xc, yc = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min() - 0.02, all_x.max() + 0.02])
    axF.plot(span, yc + (1.0 / 3.0) * (span - xc), "--", color=C_MODEL, linewidth=1.4, zorder=2)
    handles = [
        Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
        Line2D([], [], color="0.3", ls=":", lw=1.0, label="Instructed slope 1"),
        Line2D([], [], color=C_MODEL, ls="--", lw=1.4, label="Two-thirds slope 1/3"),
    ]
    axF.set_xlabel("Log10 Produced Radius")
    axF.set_ylabel("Log10 Mean Speed")
    axF.set_title("M3 Speed Not Set By Geometry")
    axF.legend(handles=handles, loc="lower right", frameon=False, fontsize=8, handlelength=1.6)
    axF.text(
        0.03,
        0.97,
        f"Speed-radius slope {D['slope_med']:.2f}\n"
        f"vs two-thirds: {_format_p(D['p_vs_third'])}\n"
        f"log K vs T slope {D['k_slope']:+.2f}",
        transform=axF.transAxes,
        va="top",
        fontsize=7.5,
        bbox=text_box,
    )

    # G. Full-matrix summary.
    tidy(axG)
    labels = ["M1\nNoise", "M2\nEffort", "M3\n2/3 Law"]
    errors = [
        _matrix_error(D["noise_matrix"], D["B_obs"]),
        D["effort_best"]["error"],
        _matrix_error(D["twothirds_matrix"], D["B_obs"]),
    ]
    bars = axG.bar(range(3), errors, color=[(0.68, 0.68, 0.68), (0.50, 0.50, 0.50), (0.32, 0.32, 0.32)], edgecolor="none")
    for bar, err in zip(bars, errors):
        axG.text(bar.get_x() + bar.get_width() / 2, err + 0.035, f"{err:.2f}", ha="center", va="bottom", fontsize=9)
    axG.set_xticks(range(3))
    axG.set_xticklabels(labels)
    axG.set_ylabel("Full-Matrix Error")
    axG.set_title("Classical Models Cannot Match The Full Matrix")
    axG.set_ylim(0, max(errors) * 1.28)
    axG.text(
        0.03,
        0.97,
        "M2 is closest, but trades\ncompression against leakage.",
        transform=axG.transAxes,
        va="top",
        fontsize=7.5,
        bbox=text_box,
    )

    # H. Bridge to the dynamics figures.
    axH.axis("off")
    axH.set_title("Implication")
    boxes = [
        (0.10, 0.66, "Behavior\nMatrix"),
        (0.10, 0.39, "Classical\nAlternatives\nInsufficient"),
        (0.10, 0.12, "Dynamics-Control\nMechanism"),
    ]
    for x, y, label in boxes:
        axH.text(x + 0.40, y + 0.09, label, ha="center", va="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.8))
    for y_start, y_end in [(0.63, 0.56), (0.36, 0.29)]:
        axH.annotate(
            "",
            xy=(0.50, y_end),
            xytext=(0.50, y_start),
            xycoords=axH.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.1, color="black"),
        )
    axH.text(
        0.50,
        0.02,
        "Motivates recurrent dynamics\nand readout geometry analyses.",
        transform=axH.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    for ax, lab in zip([axA, axB, axC, axD, axE, axF, axG, axH], "ABCDEFGH"):
        label_panel(ax, lab)
    fig.tight_layout()
    png = FIG_DIR / "alternative_behavior_model_legacy.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def make_figure_supplemental_1x4(D):
    setup_style()
    figure_width, figure_height = 18.0, 4.8
    fig = plt.figure(figsize=(figure_width, figure_height))
    axis_bottom, axis_height = 0.16, 0.70
    square_width = axis_height * figure_height / figure_width
    narrow_width = square_width / 3.0
    main_gap, inner_gap = 0.032, 0.030
    left_margin = (1.0 - (4 * square_width + narrow_width + 3 * main_gap + inner_gap)) / 2.0
    x_a = left_margin
    x_b = x_a + square_width + main_gap
    x_b_error = x_b + square_width + inner_gap
    x_c = x_b_error + narrow_width + main_gap
    x_d = x_c + square_width + main_gap
    axA = fig.add_axes([x_a, axis_bottom, square_width, axis_height])
    axB = fig.add_axes([x_b, axis_bottom, square_width, axis_height])
    axC = fig.add_axes([x_b_error, axis_bottom, narrow_width, axis_height])
    axD = fig.add_axes([x_c, axis_bottom, square_width, axis_height])
    axE = fig.add_axes([x_d, axis_bottom, square_width, axis_height])
    text_box = dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.5)

    def tidy(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    Ts = D["Ts"]
    rng = np.random.RandomState(0)

    # A. M1 simple proportional-noise model.
    tidy(axA)
    xline = np.array([0.10, 0.36])
    axA.plot(
        xline,
        D["noise_scale"] * xline,
        color=C_MODEL,
        lw=1.4,
        ls="--",
        label="Proportional-noise prediction",
    )
    axA.scatter(D["CVR"], D["CR"], s=22, color=C_R, alpha=0.65, edgecolor="white", linewidth=0.3, label="Space", zorder=3)
    axA.scatter(D["CVT"], D["CT"], s=22, color=C_T, alpha=0.65, edgecolor="white", linewidth=0.3, label="Time", zorder=3)
    xr = np.linspace(D["CVR"].min(), D["CVR"].max(), 50)
    xt = np.linspace(D["CVT"].min(), D["CVT"].max(), 50)
    axA.plot(xr, np.polyval(D["fitR"], xr), color=C_R, lw=1.8, zorder=4)
    axA.plot(xt, np.polyval(D["fitT"], xt), color=C_T, lw=1.8, zorder=4)
    axA.axhline(0, color="0.7", lw=0.6)
    axA.set_xlabel("Produced noise (CV)")
    axA.set_ylabel(r"Compression $1-\beta$")
    axA.set_title("M1 simple proportional-noise\nmodel")
    axA.legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.4)
    axA.text(
        0.97,
        0.97,
        f"Time noisier ({_format_p(D['p_noise'])})\n"
        "but space compresses.\n"
        f"space r={D['pearsonR']:+.2f} ({_format_p(D['pearson_pR'])})\n"
        f"time r={D['pearsonT']:+.2f} ({_format_p(D['pearson_pT'])})",
        transform=axA.transAxes,
        va="top",
        ha="right",
        fontsize=7.5,
        bbox=text_box,
    )
    axA.set_xlim(0.10, 0.37)

    # B. M2 model manifold after enforcing compatibility with temporal effects.
    tidy(axB)
    feasible = D["effort_feasible"]
    if len(feasible):
        axB.scatter(
            feasible[:, 2],
            feasible[:, 3],
            s=13,
            color=C_MODEL,
            alpha=0.22,
            edgecolor="none",
            zorder=2,
            label="Timing-compatible",
        )
    eigval, eigvec = np.linalg.eigh(D["spatial_bootstrap_cov"])
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.maximum(eigval[order], 0), eigvec[:, order]
    ellipse_scale = math.sqrt(stats.chi2.ppf(0.95, df=2))
    ellipse_angle = math.degrees(math.atan2(eigvec[1, 0], eigvec[0, 0]))
    observed_ellipse = Ellipse(
        (D["mRR"], D["mRT"]),
        width=2 * ellipse_scale * math.sqrt(eigval[0]),
        height=2 * ellipse_scale * math.sqrt(eigval[1]),
        angle=ellipse_angle,
        facecolor=C_R,
        edgecolor=C_R,
        alpha=0.18,
        linewidth=1.2,
        zorder=3,
        label="Observed 95% CI",
    )
    axB.add_patch(observed_ellipse)
    axB.scatter(
        [D["mRR"]], [D["mRT"]], s=65, color=C_R, edgecolor="black", linewidth=0.8,
        zorder=5, label="Observed",
    )
    if np.all(np.isfinite(D["closest_feasible"])):
        axB.scatter(
            [D["closest_feasible"][2]], [D["closest_feasible"][3]], s=60,
            facecolor="white", edgecolor="black", linewidth=0.9, marker="D", zorder=6,
            label="Closest model",
        )
    axB.axhline(0, color="0.75", lw=0.6)
    axB.axvline(1.0, color="0.8", lw=0.6, ls=":")
    axB.set_xlabel(r"Spatial direct gain $\beta_{RR}^{M}$")
    axB.set_ylabel(r"Time to space $\beta_{RT}^{M}$")
    axB.set_title("M2 timing-constrained\nmodel manifold")
    axB.legend(loc="upper right", frameon=False, fontsize=7.1, handlelength=1.3)
    axB.text(
        0.04,
        0.04,
        fr"Timing 95% CI: $\beta_{{TR}}^{{M}}$ [{D['timing_ci_tr'][0]:+.2f}, {D['timing_ci_tr'][1]:+.2f}]" "\n"
        fr"$\beta_{{TT}}^{{M}}$ [{D['timing_ci_tt'][0]:.2f}, {D['timing_ci_tt'][1]:.2f}]" "\n"
        f"Grid inside spatial ellipse: {D['feasible_inside_observed_95']}\n"
        f"Closest $D^2$={D['closest_feasible_mahalanobis2']:.2f}",
        transform=axB.transAxes,
        va="bottom",
        fontsize=7.4,
        bbox=text_box,
    )
    manifold_x = feasible[:, 2] if len(feasible) else np.asarray([D["mRR"]])
    manifold_y = feasible[:, 3] if len(feasible) else np.asarray([D["mRT"]])
    display_x = np.r_[manifold_x, D["mRR"] - observed_ellipse.width / 2, D["mRR"] + observed_ellipse.width / 2]
    display_y = np.r_[manifold_y, D["mRT"] - observed_ellipse.height / 2, D["mRT"] + observed_ellipse.height / 2]
    x_margin = max(0.04 * np.ptp(display_x), 0.025)
    y_margin = max(0.08 * np.ptp(display_y), 0.04)
    axB.set_xlim(float(np.min(display_x) - x_margin), float(np.max(display_x) + x_margin))
    axB.set_ylim(float(min(np.min(display_y) - y_margin, -0.04)), float(np.max(display_y) + y_margin))
    axB.locator_params(axis="both", nbins=5)

    # C. M2 LOSO full-matrix error versus the empirical train-mean baseline.
    tidy(axC)
    error_data = [D["cv_model_errors"], D["cv_baseline_errors"]]
    error_pos = np.asarray([0.0, 1.0])
    for model_error, baseline_error in zip(*error_data):
        axC.plot(error_pos, [model_error, baseline_error], color="0.78", lw=0.7, alpha=0.75, zorder=1)
    bp_error = axC.boxplot(
        error_data, positions=error_pos, widths=0.28, patch_artist=True, showfliers=False,
    )
    bp_error["boxes"][0].set(facecolor=C_MODEL, alpha=0.40, edgecolor="black", linewidth=0.8)
    bp_error["boxes"][1].set(facecolor="white", edgecolor="0.35", linewidth=0.8)
    for e in bp_error["medians"]:
        e.set(color="black", linewidth=1.2)
    for e in bp_error["whiskers"] + bp_error["caps"]:
        e.set(color="black", linewidth=0.8)
    for i, values in enumerate(error_data):
        facecolor = C_MODEL if i == 0 else "white"
        edgecolor = "white" if i == 0 else "0.35"
        axC.scatter(
            error_pos[i] + rng.uniform(-0.055, 0.055, len(values)), values,
            s=10, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.4, alpha=0.85, zorder=3,
        )
    error_max = float(np.nanmax(np.concatenate(error_data)))
    axC.set_xlim(-0.38, 1.38)
    axC.set_ylim(0, error_max * 1.17)
    axC.locator_params(axis="y", nbins=5)
    axC.set_xticks(error_pos)
    axC.set_xticklabels(["Effort\nmodel", "Train-mean\nbaseline"])
    axC.set_xlabel(rf"Paired $p_{{\mathrm{{adj}}}}$ = {D['cv_error_p_adj']:.3g}")
    axC.set_ylabel("LOSO matrix error\n(Frobenius norm)")
    axC.set_title("LOSO full-matrix\nerror")

    # C-left. M3 constant-K spontaneous two-thirds model: duration-radius arm.
    tidy(axD)
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["famT"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axD.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        j = int(np.argmin(lx))
        span = np.array([lx.min(), lx.max()])
        axD.plot(span, ly[j] + (2.0 / 3.0) * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
    handles = [
        Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
        Line2D([], [], color="0.3", ls=":", lw=1.0, label="Constant-k slope 2/3"),
    ]
    axD.set_xlabel("Log10 produced radius")
    axD.set_ylabel("Log10 produced duration")
    axD.set_title("M3 constant-k two-thirds\nduration-radius prediction")
    axD.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, handlelength=1.6)
    axD.text(
        0.97,
        0.03,
        f"Duration-radius slope {D['timeR_slope']:.2f}\n"
        f"vs 0: {_format_p(D['timeR_p0'])}\n"
        f"vs constant-K 2/3: {_format_p(D['timeR_p23'])}",
        transform=axD.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.5,
        bbox=text_box,
    )

    # C-right. M3 constant-K spontaneous two-thirds model: speed-radius diagnostic.
    tidy(axE)
    all_x, all_y = [], []
    for i, t in enumerate(Ts):
        x, y, ylo, yhi = D["fam"][t]
        if len(x) == 0:
            continue
        col = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axE.errorbar(lx, ly, yerr=yerr, fmt="-o", color=col, markersize=4, linewidth=1.3,
                     capsize=2, elinewidth=0.8, zorder=3)
        j = int(np.argmin(lx))
        span = np.array([lx.min(), lx.max()])
        axE.plot(span, ly[j] + 1.0 * (span - lx[j]), ":", color=col, linewidth=1.0, zorder=2)
        all_x.append(lx)
        all_y.append(ly)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    xc, yc = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min() - 0.02, all_x.max() + 0.02])
    axE.plot(span, yc + (1.0 / 3.0) * (span - xc), "--", color=C_MODEL, linewidth=1.4, zorder=2)
    handles = [
        Line2D([], [], color="0.3", marker="o", ls="-", lw=1.3, label="Measured"),
        Line2D([], [], color="0.3", ls=":", lw=1.0, label="Instructed slope 1"),
        Line2D([], [], color=C_MODEL, ls="--", lw=1.4, label="Constant-k slope 1/3"),
    ]
    axE.set_xlabel("Log10 produced radius")
    axE.set_ylabel("Log10 mean speed")
    axE.set_title("M3 constant-k two-thirds\nspeed-radius prediction")
    axE.legend(handles=handles, loc="lower right", frameon=False, fontsize=8, handlelength=1.6)
    axE.text(
        0.03,
        0.97,
        f"Speed-radius slope {D['slope_med']:.2f}\n"
        f"vs constant-K 1/3: {_format_p(D['p_vs_third'])}\n"
        f"K-T slope {D['k_slope']:+.2f} ({_format_p(D['k_slope_p'])})",
        transform=axE.transAxes,
        va="top",
        fontsize=7.5,
        bbox=text_box,
    )

    label_panel(axA, "A")
    label_panel(axB, "B")
    axD.text(
        -0.03,
        1.08,
        "C",
        transform=axD.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="top",
    )
    png = FIG_DIR / "alternative_behavior_model_legacy.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def make_figure_serial_exploratory(D):
    """Expanded exploratory layout; panels can be pruned for the final paper."""
    setup_style()
    fig, axes = plt.subplots(3, 4, figsize=(15.8, 11.1))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.965, wspace=0.42, hspace=0.58)
    axA, axB, axC, axD, axE, axF, axG, axH, axI, axJ, axK, axL = axes.flat
    S = D["serial"]
    rng = np.random.RandomState(13)
    text_box = dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.2)

    def tidy(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.locator_params(axis="both", nbins=5)

    def draw_box(ax, values, position, color, width=0.20, filled=True, jitter=0.045, alpha=0.75):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        bp = ax.boxplot([values], positions=[position], widths=width, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        bp["boxes"][0].set(
            facecolor=color if filled else "white", alpha=0.35 if filled else 1.0,
            edgecolor=color, linewidth=0.9,
        )
        bp["medians"][0].set(color="black", linewidth=1.2)
        for artist in bp["whiskers"] + bp["caps"]:
            artist.set(color=color, linewidth=0.8)
        ax.scatter(position + rng.uniform(-jitter, jitter, len(values)), values, s=9,
                   facecolor=color if filled else "white", edgecolor=color, linewidth=0.45,
                   alpha=alpha, zorder=3)

    # A. Static proportional-noise account.
    tidy(axA)
    xline = np.array([0.10, 0.36])
    axA.plot(xline, D["noise_scale"] * xline, color=C_MODEL, lw=1.4, ls="--",
             label="Proportional-noise prediction")
    axA.scatter(D["CVR"], D["CR"], s=22, color=C_R, alpha=0.65, edgecolor="white", linewidth=0.3,
                label="Space", zorder=3)
    axA.scatter(D["CVT"], D["CT"], s=22, color=C_T, alpha=0.65, edgecolor="white", linewidth=0.3,
                label="Time", zorder=3)
    xr = np.linspace(D["CVR"].min(), D["CVR"].max(), 50)
    xt = np.linspace(D["CVT"].min(), D["CVT"].max(), 50)
    axA.plot(xr, np.polyval(D["fitR"], xr), color=C_R, lw=1.8)
    axA.plot(xt, np.polyval(D["fitT"], xt), color=C_T, lw=1.8)
    axA.axhline(0, color="0.72", lw=0.7)
    axA.set(xlabel="Produced Noise (CV)", ylabel=r"Compression $1-\beta$",
            title="M1a Static Proportional-Noise Model", xlim=(0.10, 0.37))
    axA.legend(loc="lower left", frameon=False, fontsize=7.5, handlelength=1.4)
    axA.text(0.97, 0.97,
             f"Time noisier ({_format_p(D['p_noise'])})\n"
             f"Space r={D['pearsonR']:+.2f} ({_format_p(D['pearson_pR'])})\n"
             f"Time r={D['pearsonT']:+.2f} ({_format_p(D['pearson_pT'])})",
             transform=axA.transAxes, ha="right", va="top", fontsize=7.2, bbox=text_box)

    # B-C. Empirical target-history kernels with circular-shift null envelopes.
    lag_x = np.asarray(SERIAL_LAGS, dtype=float)
    kernel_specs = [
        (axB, [(0, 0, C_R, r"$R_{n-k}\ \to\ R_n$"), (1, 1, C_T, r"$T_{n-k}\ \to\ T_n$")],
         "M1b Same-Dimension Serial Kernel"),
        (axC, [(0, 1, C_R, r"$T_{n-k}\ \to\ R_n$"), (1, 0, C_T, r"$R_{n-k}\ \to\ T_n$")],
         "M1b Cross-Dimension Serial Kernel"),
    ]
    lag1_p_holm = _holm_adjust(np.asarray([
        _safe_wilcoxon(S["lag_H"][:, 0, i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ], dtype=float))
    lag1_p_map = {
        (0, 0): lag1_p_holm[0], (0, 1): lag1_p_holm[1],
        (1, 0): lag1_p_holm[2], (1, 1): lag1_p_holm[3],
    }
    for ax, channels, title in kernel_specs:
        tidy(ax)
        for output_index, history_index, color, label in channels:
            values = S["lag_H"][:, :, output_index, history_index]
            for subject_values in values:
                ax.plot(lag_x, subject_values, color=color, alpha=0.10, lw=0.65)
            median = np.nanmedian(values, axis=0)
            low, high = np.nanpercentile(S["lag_null"][:, :, output_index, history_index], [2.5, 97.5], axis=0)
            ax.fill_between(lag_x, low, high, color="0.75", alpha=0.28, linewidth=0)
            ax.plot(lag_x, median, "-o", color=color, lw=1.8, ms=4, label=label)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(lag_x)
        ax.set_xlabel("Chronological Lag (Trials)")
        ax.set_ylabel("Target-History Coefficient")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, handlelength=1.5)
        lag1_text = []
        for output_index, history_index, _, label in channels:
            p = lag1_p_map[(output_index, history_index)]
            lag1_text.append(f"{label.replace('$', '')}: {_format_p(p)}")
        ax.text(0.98, 0.03, "Lag 1 Holm\n" + "\n".join(lag1_text), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.0, bbox=text_box)

    # D. Does a probe trial update the same prior as a drawing trial?
    tidy(axD)
    coef_labels = [r"$H_{RR}$", r"$H_{RT}$", r"$H_{TR}$", r"$H_{TT}$"]
    coef_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for index, (output_index, history_index) in enumerate(coef_indices):
        color = C_R if output_index == 0 else C_T
        draw_box(axD, S["transition_H"]["normal"][:, output_index, history_index], index - 0.13,
                 color, width=0.20, filled=True)
        draw_box(axD, S["transition_H"]["probe"][:, output_index, history_index], index + 0.13,
                 color, width=0.20, filled=False)
    axD.axhline(0, color="black", lw=0.7)
    axD.set_xticks(range(4)); axD.set_xticklabels(coef_labels)
    axD.set_ylabel("Lag-1 Target-History Coefficient")
    axD.set_title("Transition-Specific History")
    axD.legend(handles=[
        Line2D([], [], marker="s", color="0.3", markerfacecolor="0.3", ls="none", label="Normal to normal"),
        Line2D([], [], marker="s", color="0.3", markerfacecolor="white", ls="none", label="Probe to normal"),
    ], frameon=False, fontsize=7.5, loc="best")
    transition_p = []
    for output_index, history_index in coef_indices:
        transition_p.append(_safe_wilcoxon(
            S["transition_H"]["normal"][:, output_index, history_index]
            - S["transition_H"]["probe"][:, output_index, history_index]
        ))
    transition_p = _holm_adjust(np.asarray(transition_p, dtype=float))
    axD.text(0.98, 0.97,
             "Paired normal-probe, Holm\n"
             f"RR/RT: {_format_p(transition_p[0])}, {_format_p(transition_p[1])}\n"
             f"TR/TT: {_format_p(transition_p[2])}, {_format_p(transition_p[3])}",
             transform=axD.transAxes, ha="right", va="top", fontsize=6.6, bbox=text_box)

    # E. Previous response residual after controlling both current and previous targets.
    tidy(axE)
    for index, (output_index, history_index) in enumerate(coef_indices):
        color = C_R if output_index == 0 else C_T
        draw_box(axE, S["response_C"][:, output_index, history_index], index, color, width=0.42, filled=True)
    axE.axhline(0, color="black", lw=0.7)
    axE.set_xticks(range(4)); axE.set_xticklabels([r"$C_{RR}$", r"$C_{RT}$", r"$C_{TR}$", r"$C_{TT}$"])
    axE.set_ylabel("Previous-Response Coefficient")
    axE.set_title("Motor Residual Carry-Over")
    response_ps = _holm_adjust(np.asarray([_safe_wilcoxon(S["response_C"][:, i, j])
                                            for i, j in coef_indices], dtype=float))
    axE.text(0.98, 0.97, "Holm: " + ", ".join(_format_p(p) for p in response_ps), transform=axE.transAxes,
             ha="right", va="top", fontsize=6.6, bbox=text_box)

    # F. Check that the actual pseudorandom sequence did not create lag correlations.
    tidy(axF)
    for index, (current_index, previous_index) in enumerate(coef_indices):
        color = C_R if current_index == 0 else C_T
        draw_box(axF, S["transition_r"][:, current_index, previous_index], index, color, width=0.42, filled=True)
    axF.axhline(0, color="black", lw=0.7)
    axF.set_xticks(range(4)); axF.set_xticklabels([r"$R_n,R_{n-1}$", r"$R_n,T_{n-1}$",
                                                   r"$T_n,R_{n-1}$", r"$T_n,T_{n-1}$"], rotation=18)
    axF.set_ylabel("Pearson r")
    axF.set_title("Exact-Sequence Transition Balance")
    sequence_ps = _holm_adjust(np.asarray([_safe_wilcoxon(S["transition_r"][:, i, j])
                                            for i, j in coef_indices], dtype=float))
    axF.text(0.98, 0.97, "Holm: " + ", ".join(_format_p(p) for p in sequence_ps), transform=axF.transAxes,
             ha="right", va="top", fontsize=6.6, bbox=text_box)

    # G. Strict leave-one-day-out predictive contribution.
    tidy(axG)
    model_labels = ["Lag-1\nAll", "Exponential\nAll", "Exponential\nDrawing"]
    for model_index, model_name in enumerate(SERIAL_MODELS):
        draw_box(axG, S["cv_delta_r2"][model_name][:, 0], model_index - 0.16, C_R,
                 width=0.25, filled=True)
        draw_box(axG, S["cv_delta_r2"][model_name][:, 1], model_index + 0.16, C_T,
                 width=0.25, filled=True)
    axG.axhline(0, color="black", lw=0.7)
    axG.set_xticks(range(3)); axG.set_xticklabels(model_labels)
    axG.set_ylabel(r"LODO $\Delta R^2$")
    axG.set_title("Out-of-Day Predictive Value")
    axG.legend(handles=[Line2D([], [], marker="s", color=C_R, ls="none", label="Radius"),
                        Line2D([], [], marker="s", color=C_T, ls="none", label="Duration")],
               frameon=False, fontsize=7.5, loc="best")
    cv_lines = []
    for model_name in SERIAL_MODELS:
        p_r, p_t = _holm_adjust(np.asarray([
            _safe_wilcoxon(S["cv_delta_r2"][model_name][:, 0]),
            _safe_wilcoxon(S["cv_delta_r2"][model_name][:, 1]),
        ], dtype=float))
        cv_lines.append(f"{_format_p(p_r)}/{_format_p(p_t)}")
    lambda_all = np.median(S["selected_lambda"]["exp_all"])
    lambda_draw = np.median(S["selected_lambda"]["exp_drawing"])
    axG.text(0.98, 0.03,
             "Holm R/T\n" + "; ".join(cv_lines) + f"\nMedian lambda: {lambda_all:.2f}/{lambda_draw:.2f}",
             transform=axG.transAxes, ha="right", va="bottom", fontsize=6.5, bbox=text_box)

    # H. Counterfactual behavior matrix after target-history removal.
    tidy(axH)
    offsets = [-0.22, 0.0, 0.22]
    model_colors = [C_MODEL, (0.25, 0.25, 0.25), (0.70, 0.70, 0.70)]
    for model_index, model_name in enumerate(SERIAL_MODELS):
        delta = S["adjusted_delta"][model_name]
        for coef_index, (output_index, history_index) in enumerate(coef_indices):
            output_color = C_R if output_index == 0 else C_T
            filled = model_index != 2
            edge_color = output_color if model_index > 0 else C_MODEL
            draw_box(axH, delta[:, output_index, history_index], coef_index + offsets[model_index],
                     edge_color, width=0.18, filled=filled, jitter=0.032, alpha=0.62)
    axH.axhline(0, color="black", lw=0.7)
    axH.set_yscale("symlog", linthresh=0.05, linscale=0.9)
    axH.set_yticks([-3, -0.3, 0, 0.3, 1])
    axH.set_xticks(range(4)); axH.set_xticklabels([r"$\beta_{RR}$", r"$\beta_{RT}$", r"$\beta_{TR}$", r"$\beta_{TT}$"])
    axH.set_ylabel(r"History-Adjusted Change $\Delta\beta$ (Symlog)")
    axH.set_title("Contribution To Behavior Matrix")
    axH.legend(handles=[
        Line2D([], [], marker="s", color=C_MODEL, markerfacecolor=C_MODEL, ls="none", label="Lag-1 all"),
        Line2D([], [], marker="s", color="0.25", markerfacecolor="0.25", ls="none", label="Exponential all"),
        Line2D([], [], marker="s", color="0.55", markerfacecolor="white", ls="none", label="Exponential drawing"),
    ], frameon=False, fontsize=7.0, loc="best")
    exp_delta = S["adjusted_delta"]["exp_all"]
    exp_p = _holm_adjust(np.asarray([_safe_wilcoxon(exp_delta[:, i, j]) for i, j in coef_indices], dtype=float))
    axH.text(0.98, 0.97,
             "Exponential all, Holm\n"
             f"RR/RT: {_format_p(exp_p[0])}, {_format_p(exp_p[1])}\n"
             f"TR/TT: {_format_p(exp_p[2])}, {_format_p(exp_p[3])}",
             transform=axH.transAxes, ha="right", va="top", fontsize=6.5, bbox=text_box)

    # I. M2 timing-constrained model manifold.
    tidy(axI)
    feasible = D["effort_feasible"]
    if len(feasible):
        axI.scatter(feasible[:, 2], feasible[:, 3], s=10, color=C_MODEL, alpha=0.20,
                    edgecolor="none", label="Timing-compatible")
    eigval, eigvec = np.linalg.eigh(D["spatial_bootstrap_cov"])
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.maximum(eigval[order], 0), eigvec[:, order]
    ellipse_scale = math.sqrt(stats.chi2.ppf(0.95, df=2))
    ellipse_angle = math.degrees(math.atan2(eigvec[1, 0], eigvec[0, 0]))
    observed_ellipse = Ellipse((D["mRR"], D["mRT"]),
                               width=2 * ellipse_scale * math.sqrt(eigval[0]),
                               height=2 * ellipse_scale * math.sqrt(eigval[1]), angle=ellipse_angle,
                               facecolor=C_R, edgecolor=C_R, alpha=0.18, linewidth=1.1,
                               label="Observed 95% CI")
    axI.add_patch(observed_ellipse)
    axI.scatter([D["mRR"]], [D["mRT"]], s=55, color=C_R, edgecolor="black", linewidth=0.7,
                label="Observed", zorder=4)
    if np.all(np.isfinite(D["closest_feasible"])):
        axI.scatter([D["closest_feasible"][2]], [D["closest_feasible"][3]], s=50,
                    facecolor="white", edgecolor="black", marker="D", linewidth=0.8,
                    label="Closest model", zorder=5)
    axI.axhline(0, color="0.75", lw=0.6); axI.axvline(1, color="0.8", lw=0.6, ls=":")
    axI.set_xlabel(r"Spatial Direct Gain $\beta_{RR}^{M}$")
    axI.set_ylabel(r"Time To Space $\beta_{RT}^{M}$")
    axI.set_title("M2 Timing-Constrained Manifold")
    axI.legend(frameon=False, fontsize=6.8, loc="upper right")
    manifold_x = feasible[:, 2] if len(feasible) else np.asarray([D["mRR"]])
    manifold_y = feasible[:, 3] if len(feasible) else np.asarray([D["mRT"]])
    display_x = np.r_[manifold_x, D["mRR"] - observed_ellipse.width / 2, D["mRR"] + observed_ellipse.width / 2]
    display_y = np.r_[manifold_y, D["mRT"] - observed_ellipse.height / 2, D["mRT"] + observed_ellipse.height / 2]
    axI.set_xlim(np.min(display_x) - 0.03, np.max(display_x) + 0.03)
    axI.set_ylim(min(np.min(display_y) - 0.04, -0.04), np.max(display_y) + 0.04)
    axI.text(0.03, 0.03, f"Closest D2={D['closest_feasible_mahalanobis2']:.2f}",
             transform=axI.transAxes, fontsize=7.0, bbox=text_box)

    # J. M2 held-out full-matrix error.
    tidy(axJ)
    error_data = [D["cv_model_errors"], D["cv_baseline_errors"]]
    for model_error, baseline_error in zip(*error_data):
        axJ.plot([0, 1], [model_error, baseline_error], color="0.78", lw=0.65, alpha=0.75)
    draw_box(axJ, error_data[0], 0, C_MODEL, width=0.38, filled=True)
    draw_box(axJ, error_data[1], 1, C_MODEL, width=0.38, filled=False)
    axJ.set_xticks([0, 1]); axJ.set_xticklabels(["Effort\nModel", "Train-Mean\nBaseline"])
    axJ.set_ylabel("LOSO Matrix Error")
    axJ.set_title("M2 Held-Out Full-Matrix Error")
    axJ.text(0.98, 0.97, _format_p(D["cv_error_p_adj"]), transform=axJ.transAxes,
             ha="right", va="top", fontsize=7.5, bbox=text_box)

    # K-L. M3 constant-K predictions.
    Ts = D["Ts"]
    tidy(axK)
    for i, target_t in enumerate(Ts):
        x, y, ylo, yhi = D["famT"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axK.errorbar(lx, ly, yerr=yerr, fmt="-o", color=color, markersize=3.5,
                     linewidth=1.2, capsize=2, elinewidth=0.8)
        anchor = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        axK.plot(span, ly[anchor] + (2.0 / 3.0) * (span - lx[anchor]), ":", color=color, lw=1.0)
    axK.set_xlabel("Log10 Produced Radius"); axK.set_ylabel("Log10 Produced Duration")
    axK.set_title("M3 Constant-K Duration Prediction")
    axK.text(0.98, 0.03, f"Slope={D['timeR_slope']:.2f}\nVs 2/3: {_format_p(D['timeR_p23'])}",
             transform=axK.transAxes, ha="right", va="bottom", fontsize=7.2, bbox=text_box)

    tidy(axL)
    all_x, all_y = [], []
    for i, target_t in enumerate(Ts):
        x, y, ylo, yhi = D["fam"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * i / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axL.errorbar(lx, ly, yerr=yerr, fmt="-o", color=color, markersize=3.5,
                     linewidth=1.2, capsize=2, elinewidth=0.8)
        anchor = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        axL.plot(span, ly[anchor] + (span - lx[anchor]), ":", color=color, lw=1.0)
        all_x.append(lx); all_y.append(ly)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    center_x, center_y = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min(), all_x.max()])
    axL.plot(span, center_y + (span - center_x) / 3.0, "--", color=C_MODEL, lw=1.3)
    axL.set_xlabel("Log10 Produced Radius"); axL.set_ylabel("Log10 Mean Speed")
    axL.set_title("M3 Constant-K Speed Prediction")
    axL.text(0.03, 0.97, f"Slope={D['slope_med']:.2f}\nVs 1/3: {_format_p(D['p_vs_third'])}",
             transform=axL.transAxes, ha="left", va="top", fontsize=7.2, bbox=text_box)

    for ax, panel in zip(axes.flat, "ABCDEFGHIJKL"):
        label_panel(ax, panel)
    png = FIG_DIR / "alternative_behavior_model_legacy.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def make_figure_compact_serial(D):
    """Compact final-candidate layout: A-E above and grouped F-G below."""
    setup_style()
    fig = plt.figure(figsize=(18.0, 8.7))
    panel_width, panel_height = 0.185, 0.355
    panel_gap = 0.040
    top_total = 4 * panel_width + 3 * panel_gap
    bottom_total = 4 * panel_width + 3 * panel_gap
    top_left = (1.0 - top_total) / 2.0
    bottom_left = (1.0 - bottom_total) / 2.0
    top_y, bottom_y = 0.585, 0.085
    top_axes = [
        fig.add_axes([top_left + index * (panel_width + panel_gap), top_y, panel_width, panel_height])
        for index in range(4)
    ]
    bottom_axes = [
        fig.add_axes([bottom_left + index * (panel_width + panel_gap), bottom_y, panel_width, panel_height])
        for index in range(4)
    ]
    axA, axB, axD, axE = top_axes
    axF, axG, axH, axI = bottom_axes
    # Reserve a narrow linear segment below panel C for the single extreme
    # blocked-CV value.  The main segment keeps the distribution near zero
    # readable without deleting or winsorizing that observation.
    d_position = axD.get_position()
    d_low_height = 0.050
    d_gap = 0.018
    axD.set_position([
        d_position.x0, d_position.y0 + d_low_height + d_gap,
        d_position.width, d_position.height - d_low_height - d_gap,
    ])
    axD_low = fig.add_axes([
        d_position.x0, d_position.y0, d_position.width, d_low_height,
    ], sharex=axD)
    S = D["serial"]
    V = D["serial_pooled_blocked"]
    contribution = lag1_behavior_contribution(S)
    rng = np.random.RandomState(13)
    text_box = dict(facecolor="white", alpha=0.84, edgecolor="none", pad=1.2)

    def tidy(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.locator_params(axis="both", nbins=5)

    def draw_box(ax, values, position, color, width=0.22, filled=True, jitter=0.042,
                 alpha=0.72, show_points=True):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        bp = ax.boxplot([values], positions=[position], widths=width, patch_artist=True,
                        showfliers=False, manage_ticks=False)
        bp["boxes"][0].set(facecolor=color if filled else "white", alpha=0.35 if filled else 1.0,
                            edgecolor=color, linewidth=0.9)
        bp["medians"][0].set(color="black", linewidth=1.2)
        for artist in bp["whiskers"] + bp["caps"]:
            artist.set(color=color, linewidth=0.8)
        if show_points:
            ax.scatter(position + rng.uniform(-jitter, jitter, len(values)), values, s=9,
                       facecolor=color if filled else "white", edgecolor=color, linewidth=0.45,
                       alpha=alpha, zorder=3)

    # A. Static proportional-noise model.
    tidy(axA)
    xline = np.array([0.10, 0.36])
    axA.plot(xline, D["noise_scale"] * xline, color=C_MODEL, lw=1.4, ls="--",
             label="Proportional-noise prediction")
    axA.scatter(D["CVR"], D["CR"], s=22, color=C_R, alpha=0.65, edgecolor="white", linewidth=0.3,
                label="Space", zorder=3)
    axA.scatter(D["CVT"], D["CT"], s=22, color=C_T, alpha=0.65, edgecolor="white", linewidth=0.3,
                label="Time", zorder=3)
    xr = np.linspace(D["CVR"].min(), D["CVR"].max(), 50)
    xt = np.linspace(D["CVT"].min(), D["CVT"].max(), 50)
    axA.plot(xr, np.polyval(D["fitR"], xr), color=C_R, lw=1.8)
    axA.plot(xt, np.polyval(D["fitT"], xt), color=C_T, lw=1.8)
    axA.axhline(0, color="0.72", lw=0.7)
    axA.set(xlabel="Produced Noise (CV)", ylabel=r"Compression $1-\beta$",
            title="M1a Static Proportional-Noise Model", xlim=(0.10, 0.37))
    axA.legend(loc="lower left", frameon=False, handlelength=1.3)
    axA.text(0.97, 0.97,
             f"Time noisier ({_format_p(D['p_noise'])})\n"
             f"Space r={D['pearsonR']:+.2f} ({_format_p(D['pearson_pR'])})\n"
             f"Time r={D['pearsonT']:+.2f} ({_format_p(D['pearson_pT'])})",
             transform=axA.transAxes, ha="right", va="top", bbox=text_box)

    # B. Merged same- and cross-dimensional empirical lag kernels.
    tidy(axB)
    lag_x = np.asarray(SERIAL_LAGS, dtype=float)
    channel_specs = [
        (0, 0, C_R, "-", "o", r"$R_{n-k}\to R_n$"),
        (0, 1, C_R, "--", "s", r"$T_{n-k}\to R_n$"),
        (1, 1, C_T, "-", "o", r"$T_{n-k}\to T_n$"),
        (1, 0, C_T, "--", "s", r"$R_{n-k}\to T_n$"),
    ]
    null_lows, null_highs, channel_medians = [], [], []
    for output_index, history_index, color, linestyle, marker, label in channel_specs:
        values = S["lag_H"][:, :, output_index, history_index]
        median = np.nanmedian(values, axis=0)
        channel_medians.append(median)
        ci = np.asarray([_boot_ci(values[:, lag], n=4000, seed=500 + 10 * output_index + history_index + lag)
                         for lag in range(len(SERIAL_LAGS))], dtype=float)
        axB.fill_between(lag_x, ci[:, 0], ci[:, 1], color=color, alpha=0.10, linewidth=0)
        axB.plot(lag_x, median, color=color, ls=linestyle, marker=marker, ms=3.8, lw=1.6, label=label)
        null_ci = np.nanpercentile(S["lag_null"][:, :, output_index, history_index], [2.5, 97.5], axis=0)
        null_lows.append(null_ci[0]); null_highs.append(null_ci[1])
    axB.fill_between(lag_x, np.min(null_lows, axis=0), np.max(null_highs, axis=0),
                     color="0.65", alpha=0.14, linewidth=0, zorder=0, label="Shifted null")
    axB.axhline(0, color="black", lw=0.7)
    axB.set_xticks(lag_x)
    axB.set_xlabel("Chronological Lag (Trials)")
    axB.set_ylabel("Target-History Coefficient")
    axB.set_title("M1b Empirical Serial Kernel")
    lag_legend = axB.legend(frameon=False, handlelength=1.5, ncol=1, loc="upper left")
    lag_legend.set_zorder(1)
    lag_p_all = np.asarray([
        _holm_adjust(np.asarray([
            _safe_wilcoxon(S["lag_H"][:, lag_index, output_index, history_index])
            for output_index, history_index, *_ in channel_specs
        ]))
        for lag_index in range(len(SERIAL_LAGS))
    ])
    star_x_offsets = [0.0, 0.0, -0.10, 0.10]
    for channel_index, median in enumerate(channel_medians):
        for lag_index, x_value in enumerate(lag_x):
            if lag_p_all[lag_index, channel_index] < 0.05:
                offset = 0.006 if median[lag_index] >= 0 else -0.006
                axB.text(x_value + star_x_offsets[channel_index], median[lag_index] + offset,
                         "*", ha="center", va="bottom" if offset > 0 else "top",
                         fontsize=10, fontweight="bold", zorder=10)
    lag1_p = _holm_adjust(np.asarray([
        _safe_wilcoxon(S["lag_H"][:, 0, i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]))
    axB.text(0.98, 0.03,
             "Lag 1 Holm\n"
             f"RR/RT: {_format_p(lag1_p[0])}, {_format_p(lag1_p[1])}\n"
             f"TR/TT: {_format_p(lag1_p[2])}, {_format_p(lag1_p[3])}",
             transform=axB.transAxes, ha="right", va="bottom", bbox=text_box)

    # C. Inset: action-specific normal/probe transition comparison.  All four
    # channels are retained, but represented by medians and bootstrap CIs so
    # the complete comparison remains legible at inset scale.
    axC = axB.inset_axes([0.50, 0.52, 0.48, 0.44])
    axC.set_facecolor("white")
    tidy(axC)
    coefficient_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    inset_specs = [
        (0, 0, C_R, "-", "o"), (0, 1, C_R, "--", "s"),
        (1, 0, C_T, "--", "s"), (1, 1, C_T, "-", "o"),
    ]
    transition_p = []
    offsets = np.linspace(-0.12, 0.12, 4)
    for index, (output_index, history_index, color, linestyle, marker) in enumerate(inset_specs):
        medians, intervals = [], []
        for transition_index, transition in enumerate(("normal", "probe")):
            values = S["transition_H"][transition][:, output_index, history_index]
            medians.append(float(np.nanmedian(values)))
            intervals.append(_boot_ci(values, n=4000, seed=720 + 10 * index + transition_index))
        x_values = np.asarray([0.0, 1.0]) + offsets[index]
        intervals = np.asarray(intervals)
        yerr = np.vstack([np.asarray(medians) - intervals[:, 0], intervals[:, 1] - np.asarray(medians)])
        axC.errorbar(x_values, medians, yerr=yerr, color=color, ls=linestyle, marker=marker,
                     ms=3.2, lw=1.0, capsize=1.5, elinewidth=0.7)
        transition_p.append(_safe_wilcoxon(
            S["transition_H"]["normal"][:, output_index, history_index]
            - S["transition_H"]["probe"][:, output_index, history_index]
        ))
    transition_p = _holm_adjust(np.asarray(transition_p))
    for index, p_value in enumerate(transition_p):
        if p_value < 0.05:
            output_index, history_index, *_ = inset_specs[index]
            values = S["transition_H"]["probe"][:, output_index, history_index]
            median = float(np.nanmedian(values))
            axC.text(1.0 + offsets[index], median + (0.012 if median >= 0 else -0.012), "*",
                     ha="center", va="bottom" if median >= 0 else "top", fontsize=10,
                     fontweight="bold")
    axC.axhline(0, color="black", lw=0.7)
    axC.set_xticks([0, 1]); axC.set_xticklabels(["Normal", "Probe"])
    axC.set_ylabel(r"Lag-1 $H$")
    axC.set_title("")
    axC.text(0.02, 0.98, "Action Dependence", transform=axC.transAxes,
             ha="left", va="top")

    # C. Pooled three-day, within-day blocked CV (not leave-one-day-out).
    tidy(axD)
    validation_models = ["lag1", "lag1_to_5", "exp_all", "exp_drawing"]
    validation_labels = ["Lag 1", "Lags\n1-5", "Exp.\nAll", "Exp.\nDrawing"]
    validation_data = V["delta_R2_from_current"]
    for model_index, model_name in enumerate(validation_models):
        values = np.asarray(validation_data[model_name]["subject_values"], dtype=float)
        draw_box(axD, values[:, 0], model_index - 0.15, C_R, width=0.23, filled=True, jitter=0.035)
        draw_box(axD, values[:, 1], model_index + 0.15, C_T, width=0.23, filled=True, jitter=0.035)
        for output_index, color in enumerate((C_R, C_T)):
            extreme = values[:, output_index]
            extreme = extreme[extreme < -0.16]
            if len(extreme):
                x_value = model_index + (-0.15 if output_index == 0 else 0.15)
                axD_low.scatter(np.repeat(x_value, len(extreme)), extreme, s=14,
                                facecolor=color, edgecolor=color, linewidth=0.5, zorder=3)
                for value in extreme:
                    axD_low.text(x_value + 0.09, value, f"{value:.3f}", ha="left", va="center")
    axD.axhline(0, color="black", lw=0.7)
    axD.set_ylim(-0.16, 0.08)
    axD.set_yticks([-0.15, -0.10, -0.05, 0.00, 0.05])
    axD.set_xticks(range(4)); axD.tick_params(axis="x", bottom=False, labelbottom=False)
    axD.set_ylabel(r"Blocked-CV $\Delta R^2$ Vs Current-Only")
    axD.set_title("Pooled Blocked Prediction")
    axD.legend(handles=[Line2D([], [], marker="s", color=C_R, ls="none", label="Radius"),
                        Line2D([], [], marker="s", color=C_T, ls="none", label="Duration")],
               frameon=False, loc="lower left", ncol=2)
    tidy(axD_low)
    axD_low.set_ylim(-0.66, -0.61)
    axD_low.set_yticks([-0.65])
    axD_low.set_xticks(range(4)); axD_low.set_xticklabels(validation_labels)
    axD.spines["bottom"].set_visible(False)
    axD_low.spines["top"].set_visible(False)
    break_size = 0.012
    break_style = dict(color="black", clip_on=False, lw=0.8)
    axD.plot((-break_size, +break_size), (-break_size, +break_size),
             transform=axD.transAxes, **break_style)
    axD.plot((1 - break_size, 1 + break_size), (-break_size, +break_size),
             transform=axD.transAxes, **break_style)
    axD_low.plot((-break_size, +break_size), (1 - break_size, 1 + break_size),
                 transform=axD_low.transAxes, **break_style)
    axD_low.plot((1 - break_size, 1 + break_size), (1 - break_size, 1 + break_size),
                 transform=axD_low.transAxes, **break_style)
    p_lookup = dict(zip(V["inference"]["comparison_order_current"], V["inference"]["p_Holm_current"]))
    for model_index, model_name in enumerate(validation_models):
        for output_index, suffix in enumerate(("R", "T")):
            p_value = float(p_lookup[model_name + "_" + suffix])
            if p_value < 0.05:
                values = np.asarray(validation_data[model_name]["subject_values"], dtype=float)[:, output_index]
                visible = values[(values >= -0.16) & (values <= 0.08)]
                y_value = min(0.065, float(np.max(visible)) + 0.012)
                x_value = model_index + (-0.15 if output_index == 0 else 0.15)
                axD.text(x_value, y_value, _format_p(p_value), ha="center", va="bottom")

    # D. Raw subject-level changes in the complete 2x2 behavior matrix.  The
    # observed group-mean beta is drawn on the same axis so the adjustment is
    # visible relative to the actual behavioral coefficient.
    tidy(axE)
    x_positions = np.arange(4, dtype=float)
    colors = [C_R, C_R, C_T, C_T]
    delta_beta = np.column_stack([
        S["adjusted_delta"]["lag1_all"][:, output_index, input_index]
        for output_index, input_index in coefficient_indices
    ])
    for index in range(4):
        draw_box(axE, delta_beta[:, index], index, colors[index], width=0.52,
                 filled=True, show_points=False)
    axE.axhline(0, color="black", lw=0.8)
    axE.set_xticks(x_positions)
    axE.set_xticklabels([r"$\beta_{RR}$", r"$\beta_{RT}$", r"$\beta_{TR}$", r"$\beta_{TT}$"])
    axE.set_ylim(-0.10, 1.12)
    axE.set_yticks([-0.10, 0.20, 0.50, 0.80, 1.10])
    axE.set_ylabel(r"Observed $\beta$ And Lag-1 Adjustment $\Delta\beta$")
    axE.set_title("Lag-1 Adjustment Of Behavior Matrix")

    observed_beta = np.mean(S["adjusted_original"]["lag1_all"], axis=0)
    for index, (output_index, input_index) in enumerate(coefficient_indices):
        beta_observed = float(observed_beta[output_index, input_index])
        axE.scatter(index, beta_observed, marker="o", s=32, facecolor="white",
                    edgecolor="black", linewidth=0.9, zorder=5)
        axE.text(index, beta_observed + 0.035, f"{beta_observed:.3f}",
                 ha="center", va="bottom")
    axE.text(0.02, 0.98, "Open circle: observed beta\nBoxplot: adjusted delta-beta",
             transform=axE.transAxes, ha="left", va="top", bbox=text_box)

    # F. Current panel I: M2 timing-constrained manifold.
    tidy(axF)
    feasible = D["effort_feasible"]
    if len(feasible):
        axF.scatter(feasible[:, 2], feasible[:, 3], s=10, color=C_MODEL, alpha=0.20,
                    edgecolor="none", label="Timing-compatible")
    eigval, eigvec = np.linalg.eigh(D["spatial_bootstrap_cov"])
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.maximum(eigval[order], 0), eigvec[:, order]
    ellipse_scale = math.sqrt(stats.chi2.ppf(0.95, df=2))
    ellipse_angle = math.degrees(math.atan2(eigvec[1, 0], eigvec[0, 0]))
    observed_ellipse = Ellipse((D["mRR"], D["mRT"]),
                               width=2 * ellipse_scale * math.sqrt(eigval[0]),
                               height=2 * ellipse_scale * math.sqrt(eigval[1]), angle=ellipse_angle,
                               facecolor=C_R, edgecolor=C_R, alpha=0.18, linewidth=1.1,
                               label="Observed 95% CI")
    axF.add_patch(observed_ellipse)
    axF.scatter([D["mRR"]], [D["mRT"]], s=55, color=C_R, edgecolor="black", linewidth=0.7,
                label="Observed", zorder=4)
    if np.all(np.isfinite(D["closest_feasible"])):
        axF.scatter([D["closest_feasible"][2]], [D["closest_feasible"][3]], s=50,
                    facecolor="white", edgecolor="black", marker="D", linewidth=0.8,
                    label="Closest model", zorder=5)
    axF.axhline(0, color="0.75", lw=0.6); axF.axvline(1, color="0.8", lw=0.6, ls=":")
    axF.set_xlabel(r"Spatial Direct Gain $\beta_{RR}^{M}$")
    axF.set_ylabel(r"Time To Space $\beta_{RT}^{M}$")
    axF.set_title("M2 Timing-Constrained Manifold")
    axF.legend(frameon=False, loc="upper right")
    manifold_x = feasible[:, 2] if len(feasible) else np.asarray([D["mRR"]])
    manifold_y = feasible[:, 3] if len(feasible) else np.asarray([D["mRT"]])
    display_x = np.r_[manifold_x, D["mRR"] - observed_ellipse.width / 2, D["mRR"] + observed_ellipse.width / 2]
    display_y = np.r_[manifold_y, D["mRT"] - observed_ellipse.height / 2, D["mRT"] + observed_ellipse.height / 2]
    axF.set_xlim(np.min(display_x) - 0.03, np.max(display_x) + 0.03)
    axF.set_ylim(min(np.min(display_y) - 0.04, -0.04), np.max(display_y) + 0.04)
    axF.text(0.03, 0.03, f"Closest D2={D['closest_feasible_mahalanobis2']:.2f}",
             transform=axF.transAxes, bbox=text_box)

    # G. Current panel J: M2 held-out matrix error.
    tidy(axG)
    error_data = [D["cv_model_errors"], D["cv_baseline_errors"]]
    for model_error, baseline_error in zip(*error_data):
        axG.plot([0, 1], [model_error, baseline_error], color="0.78", lw=0.65, alpha=0.75)
    draw_box(axG, error_data[0], 0, C_MODEL, width=0.38, filled=True)
    draw_box(axG, error_data[1], 1, C_MODEL, width=0.38, filled=False)
    axG.set_xticks([0, 1]); axG.set_xticklabels(["Effort\nModel", "Train-Mean\nBaseline"])
    axG.set_ylabel("LOSO Matrix Error")
    axG.set_title("M2 Held-Out Full-Matrix Error")
    axG.text(0.98, 0.97, _format_p(D["cv_error_p_adj"]), transform=axG.transAxes,
             ha="right", va="top", bbox=text_box)

    # H-I. Current panels K-L: M3 constant-K predictions.
    Ts = D["Ts"]
    tidy(axH)
    for index, target_t in enumerate(Ts):
        x, y, ylo, yhi = D["famT"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * index / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axH.errorbar(lx, ly, yerr=yerr, fmt="-o", color=color, markersize=3.5,
                     linewidth=1.2, capsize=2, elinewidth=0.8)
        anchor = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        axH.plot(span, ly[anchor] + (2.0 / 3.0) * (span - lx[anchor]), ":", color=color, lw=1.0)
    axH.set_xlabel("Log10 Produced Radius"); axH.set_ylabel("Log10 Produced Duration")
    axH.set_title("M3 Constant-K Duration Prediction")
    axH.text(0.98, 0.03, f"Slope={D['timeR_slope']:.2f}\nVs 2/3: {_format_p(D['timeR_p23'])}",
             transform=axH.transAxes, ha="right", va="bottom", bbox=text_box)

    tidy(axI)
    all_x, all_y = [], []
    for index, target_t in enumerate(Ts):
        x, y, ylo, yhi = D["fam"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * index / (len(Ts) - 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        axI.errorbar(lx, ly, yerr=yerr, fmt="-o", color=color, markersize=3.5,
                     linewidth=1.2, capsize=2, elinewidth=0.8)
        anchor = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        axI.plot(span, ly[anchor] + (span - lx[anchor]), ":", color=color, lw=1.0)
        all_x.append(lx); all_y.append(ly)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    center_x, center_y = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min(), all_x.max()])
    axI.plot(span, center_y + (span - center_x) / 3.0, "--", color=C_MODEL, lw=1.3)
    axI.set_xlabel("Log10 Produced Radius"); axI.set_ylabel("Log10 Mean Speed")
    axI.set_title("M3 Constant-K Speed Prediction")
    axI.text(0.03, 0.97, f"Slope={D['slope_med']:.2f}\nVs 1/3: {_format_p(D['p_vs_third'])}",
             transform=axI.transAxes, ha="left", va="top", bbox=text_box)

    for ax, panel in zip(top_axes, ("A", "B", "C", "D")):
        label_panel(ax, panel)
    label_panel(axF, "E")
    label_panel(axH, "F")
    png = FIG_DIR / "alternative_behavior_model_legacy.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def main():
    D = compute()
    png = make_figure_compact_serial(D)
    S = D["serial"]
    pooled_validation = D["serial_pooled_blocked"]
    contribution = lag1_behavior_contribution(S)
    serial_labels = ["H_RR", "H_RT", "H_TR", "H_TT"]
    serial_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    lag_p = np.asarray(
        [[_safe_wilcoxon(S["lag_H"][:, lag_index, i, j]) for i, j in serial_indices]
         for lag_index in range(len(SERIAL_LAGS))],
        dtype=float,
    )
    lag_p_holm = np.vstack([_holm_adjust(row) for row in lag_p])
    transition_difference_p = np.asarray(
        [_safe_wilcoxon(S["transition_H"]["normal"][:, i, j]
                        - S["transition_H"]["probe"][:, i, j]) for i, j in serial_indices],
        dtype=float,
    )
    response_p = np.asarray([_safe_wilcoxon(S["response_C"][:, i, j]) for i, j in serial_indices], dtype=float)
    sequence_p = np.asarray([_safe_wilcoxon(S["transition_r"][:, i, j]) for i, j in serial_indices], dtype=float)
    cv_p = {
        name: np.asarray([_safe_wilcoxon(S["cv_delta_r2"][name][:, output]) for output in range(2)], dtype=float)
        for name in SERIAL_MODELS
    }
    adjusted_p = {
        name: np.asarray([_safe_wilcoxon(S["adjusted_delta"][name][:, i, j]) for i, j in serial_indices], dtype=float)
        for name in SERIAL_MODELS
    }
    stats_out = dict(
        description="Specified alternative models are compared with the observed control matrix: "
                    "M1a simple proportional-noise model (A); M1b empirical serial kernel, action "
                    "specificity (unlettered inset), pooled blocked prediction, and lag-1 adjustment (B-D); "
                    "M2 specified quadratic effort model (E, two coordinated subaxes); and M3 constant-K "
                    "spontaneous two-thirds model (F, two coordinated subaxes). "
                    "Conclusions apply only to these exact formulations. Control inputs use perceived R/T; "
                    "produced and kinematic quantities are measured from the executed trajectory.",
        model_scope=dict(
            M1a="Simple proportional-noise model: direct compression is proportional to produced coefficient of variation, with zero cross-terms.",
            M1b="Fast serial prior: current produced R/T depends on current perceived R/T and a causal target-history prior. Target-history, previous-response carry-over, probe transitions, actual sequence balance, held-out prediction, and history-adjusted matrices are separated.",
            M2="Specified quadratic effort model: squared spatial and temporal errors plus lambda*(R/T)^2 with timing weight w.",
            M3="Constant-K spontaneous two-thirds model: speed scales as produced radius^(1/3), duration as radius^(2/3), without instructed-duration modulation of K.",
        ),
        n_subj=D["n_subj"], n_trials=D["n_trials"],
        measured_matrix_aggregation="Element-wise arithmetic mean across equally weighted subject-level coefficient matrices.",
        measured_matrix=dict(beta_RR=round(D["mRR"], 3), beta_RT=round(D["mRT"], 3),
                             beta_TR=round(D["mTR"], 3), beta_TT=round(D["mTT"], 3)),
        matrix_level_alternatives=dict(
            observed=np.round(D["B_obs"], 4).tolist(),
            M1_simple_proportional_noise=dict(
                matrix=np.round(D["noise_matrix"], 4).tolist(),
                error=round(_matrix_error(D["noise_matrix"], D["B_obs"]), 4),
                definition="Compression is proportional to produced CV and scaled to the observed mean direct compression.",
                noise_to_compression_scale=round(float(D["noise_scale"]), 4),
            ),
            M2_specified_quadratic_effort_best=dict(
                matrix=np.round(D["effort_best"]["matrix"], 4).tolist(),
                error=round(float(D["effort_best"]["error"]), 4),
                lambda_=float(D["effort_best"]["lam"]),
                w=float(D["effort_best"]["w"]),
                residual=np.round(D["effort_best"]["residual"], 4).tolist(),
                definition="In-sample best fit over lambda and w for the full 2x2 behavior matrix.",
                leave_one_subject_out=dict(
                    n_folds=int(len(D["cv_model_errors"])),
                    model_error_median=float(np.median(D["cv_model_errors"])),
                    model_error_values=np.round(D["cv_model_errors"], 6).tolist(),
                    train_mean_baseline_error_median=float(np.median(D["cv_baseline_errors"])),
                    train_mean_baseline_error_values=np.round(D["cv_baseline_errors"], 6).tolist(),
                    paired_model_vs_baseline_p_raw=float(D["cv_error_p"]),
                    paired_model_vs_baseline_p_adjusted=float(D["cv_error_p_adj"]),
                    fold_lambda=[float(row["lambda_"]) for row in D["cv_folds"]],
                    fold_w=[float(row["w"]) for row in D["cv_folds"]],
                ),
            ),
            M3_constant_K_spontaneous_two_thirds=dict(
                matrix=np.round(D["twothirds_matrix"], 4).tolist(),
                error=round(_matrix_error(D["twothirds_matrix"], D["B_obs"]), 4),
                definition="Specified constant-K spontaneous prediction with R_out=R and T_out=T_center*(R/R_center)^(2/3).",
            ),
        ),
        M1_simple_proportional_noise=dict(spatial_compression=round(float(np.median(D["CR"])), 3),
                      temporal_compression=round(float(np.median(D["CT"])), 3),
                      spatial_CV=round(float(np.median(D["CVR"])), 3), temporal_CV=round(float(np.median(D["CVT"])), 3),
                      time_vs_space_noise_p=float(D["p_noise"]),
                      primary_linear_association=dict(
                          method="Pearson correlation paired with the displayed ordinary-least-squares line.",
                          spatial_r=round(D["pearsonR"], 3), spatial_p=float(D["pearson_pR"]),
                          temporal_r=round(D["pearsonT"], 3), temporal_p=float(D["pearson_pT"]),
                      ),
                      rank_sensitivity=dict(
                          method="Spearman rank correlation.",
                          spatial_rho=round(D["rhoR"], 3), spatial_p=float(D["pR"]),
                          temporal_rho=round(D["rhoT"], 3), temporal_p=float(D["pT"]),
                      )),
        M1b_fast_adaptive_prior=dict(
            scope="Tests a fast target-history prior and serial response carry-over; it does not exclude all Bayesian models.",
            input_definition="Subject-by-condition perceived R/T, normalized identically to the main behavior matrix.",
            output_definition="Trial-level produced R/T on normal drawing trials.",
            chronological_rules="Lags never cross day boundaries. Exact lag k requires trialN[n-k]=trialN[n]-k. Normal-to-normal and probe-to-normal transitions are reported separately.",
            subject_ids=S["subjects"],
            n_subjects=int(S["n_subjects"]),
            lag_kernel=dict(
                lags=list(SERIAL_LAGS),
                coefficient_order=serial_labels,
                subject_median=np.round(
                    np.column_stack([
                        np.nanmedian(S["lag_H"][:, :, i, j], axis=0) for i, j in serial_indices
                    ]), 8
                ).tolist(),
                subject_values={
                    label: np.round(S["lag_H"][:, :, i, j], 8).tolist()
                    for label, (i, j) in zip(serial_labels, serial_indices)
                },
                n_pairs_per_subject=np.asarray(S["lag_n"], dtype=int).tolist(),
                wilcoxon_p_raw=np.round(lag_p, 10).tolist(),
                wilcoxon_p_Holm_within_lag=np.round(lag_p_holm, 10).tolist(),
                circular_shift=dict(
                    n_permutations=int(S["n_permutations"]),
                    null_median_95_CI=np.round(
                        np.stack([
                            np.nanpercentile(S["lag_null"][:, :, i, j], [2.5, 97.5], axis=0)
                            for i, j in serial_indices
                        ]), 8
                    ).tolist(),
                    method="Previous-target sequences are circularly shifted by at least five observations within each subject and day; the group-median coefficient is recomputed.",
                ),
            ),
            transition_specific_lag1=dict(
                coefficient_order=serial_labels,
                normal_to_normal_median=np.round(
                    [np.nanmedian(S["transition_H"]["normal"][:, i, j]) for i, j in serial_indices], 8
                ).tolist(),
                probe_to_normal_median=np.round(
                    [np.nanmedian(S["transition_H"]["probe"][:, i, j]) for i, j in serial_indices], 8
                ).tolist(),
                normal_to_normal_subject_values={
                    label: np.round(S["transition_H"]["normal"][:, i, j], 8).tolist()
                    for label, (i, j) in zip(serial_labels, serial_indices)
                },
                probe_to_normal_subject_values={
                    label: np.round(S["transition_H"]["probe"][:, i, j], 8).tolist()
                    for label, (i, j) in zip(serial_labels, serial_indices)
                },
                n_pairs_normal_to_normal=np.asarray(S["transition_n"]["normal"], dtype=int).tolist(),
                n_pairs_probe_to_normal=np.asarray(S["transition_n"]["probe"], dtype=int).tolist(),
                paired_normal_vs_probe_p_raw=np.round(transition_difference_p, 10).tolist(),
                paired_normal_vs_probe_p_Holm=np.round(_holm_adjust(transition_difference_p), 10).tolist(),
            ),
            previous_response_residual=dict(
                coefficient_order=["C_RR", "C_RT", "C_TR", "C_TT"],
                median=np.round([np.nanmedian(S["response_C"][:, i, j]) for i, j in serial_indices], 8).tolist(),
                subject_values={
                    label.replace("H_", "C_"): np.round(S["response_C"][:, i, j], 8).tolist()
                    for label, (i, j) in zip(serial_labels, serial_indices)
                },
                wilcoxon_p_raw=np.round(response_p, 10).tolist(),
                wilcoxon_p_Holm=np.round(_holm_adjust(response_p), 10).tolist(),
                n_pairs_per_subject=np.asarray(S["response_n"], dtype=int).tolist(),
                method="Strict adjacent normal-to-normal trials. Previous response residuals are computed after current-target, day, and within-day trend regression; current and previous targets remain in the final model.",
            ),
            exact_sequence_balance=dict(
                coefficient_order=["corr_Rn_Rprev", "corr_Rn_Tprev", "corr_Tn_Rprev", "corr_Tn_Tprev"],
                pearson_r_subject_median=np.round(
                    [np.nanmedian(S["transition_r"][:, i, j]) for i, j in serial_indices], 8
                ).tolist(),
                pearson_r_subject_values={
                    label: np.round(S["transition_r"][:, i, j], 8).tolist()
                    for label, (i, j) in zip(
                        ["corr_Rn_Rprev", "corr_Rn_Tprev", "corr_Tn_Rprev", "corr_Tn_Tprev"], serial_indices
                    )
                },
                wilcoxon_p_raw=np.round(sequence_p, 10).tolist(),
                wilcoxon_p_Holm=np.round(_holm_adjust(sequence_p), 10).tolist(),
            ),
            leave_one_day_out_prediction=dict(
                shown_in_figure=False,
                scope="Retained as a cross-session sensitivity analysis; panel C uses pooled within-day blocked cross-validation.",
                model_order=list(SERIAL_MODELS),
                output_order=["R", "T"],
                delta_R2_subject_values={name: np.round(S["cv_delta_r2"][name], 9).tolist() for name in SERIAL_MODELS},
                delta_R2_subject_median={name: np.round(np.nanmedian(S["cv_delta_r2"][name], axis=0), 9).tolist() for name in SERIAL_MODELS},
                wilcoxon_p_raw={name: np.round(cv_p[name], 10).tolist() for name in SERIAL_MODELS},
                wilcoxon_p_Holm_within_model={name: np.round(_holm_adjust(cv_p[name]), 10).tolist() for name in SERIAL_MODELS},
                selected_lambda_subject_values={name: np.round(S["selected_lambda"][name], 6).tolist() for name in SERIAL_MODELS},
                selected_lambda_subject_median={name: float(np.median(S["selected_lambda"][name])) for name in SERIAL_MODELS},
                method="Outer leave-one-day-out prediction within subject. Exponential lambda is selected using inner validation between the two training days. Baseline and history models use identical current trials and current-input/progress predictors.",
                history_definitions=dict(
                    lag1_all="Previous chronological target only (lambda=1); normal and probe targets update history.",
                    exp_all="Exponentially weighted history; every presented target updates the prior.",
                    exp_drawing="Gap-decayed exponentially weighted history; only normal drawing targets update the prior.",
                ),
            ),
            pooled_within_day_blocked_prediction=dict(
                shown_in_figure=True,
                source_file=str(POOLED_BLOCKED_VALIDATION_PATH),
                method=pooled_validation["method"],
                delta_R2_from_current=pooled_validation["delta_R2_from_current"],
                incremental_delta_R2_beyond_lag1=pooled_validation["incremental_delta_R2_beyond_lag1"],
                inference=pooled_validation["inference"],
                simultaneous_lag1_to_5=pooled_validation["simultaneous_lag1_to_5"],
                selected_lambda=pooled_validation["selected_lambda"],
            ),
            history_adjusted_behavior_matrix=dict(
                shown_in_figure="Lag-1 raw subject-level delta-beta boxplots for all four matrix terms, with group-mean observed beta values on the same axis; adjusted beta, percentage normalizations, and exponential adjustments are retained as numerical sensitivity results.",
                coefficient_order=["beta_RR", "beta_RT", "beta_TR", "beta_TT"],
                original_subject_mean={
                    name: np.round(np.mean(S["adjusted_original"][name], axis=0), 8).tolist() for name in SERIAL_MODELS
                },
                adjusted_subject_mean={
                    name: np.round(np.mean(S["adjusted_matrix"][name], axis=0), 8).tolist() for name in SERIAL_MODELS
                },
                delta_subject_median={
                    name: np.round([np.nanmedian(S["adjusted_delta"][name][:, i, j]) for i, j in serial_indices], 8).tolist()
                    for name in SERIAL_MODELS
                },
                delta_subject_values={
                    name: {
                        label.replace("H_", "beta_"): np.round(S["adjusted_delta"][name][:, i, j], 8).tolist()
                        for label, (i, j) in zip(serial_labels, serial_indices)
                    }
                    for name in SERIAL_MODELS
                },
                delta_wilcoxon_p_raw={name: np.round(adjusted_p[name], 10).tolist() for name in SERIAL_MODELS},
                delta_wilcoxon_p_Holm={name: np.round(_holm_adjust(adjusted_p[name]), 10).tolist() for name in SERIAL_MODELS},
                serial_H_subject_mean={
                    name: np.round(np.mean(S["full_H"][name], axis=0), 8).tolist() for name in SERIAL_MODELS
                },
                method="Fit the causal history model at the cross-validated subject lambda, subtract H*(prior-current) trial by trial, re-average the 25 conditions equally, and refit the full 2x2 behavior matrix.",
                lag1_percentage_contribution=dict(
                    shown_in_figure=False,
                    labels=contribution["labels"],
                    estimate_percent=np.round(contribution["estimate"], 8).tolist(),
                    subject_bootstrap_95_CI_percent=np.round(contribution["confidence_interval"], 8).tolist(),
                    bootstrap_p_raw=np.round(contribution["p_raw"], 10).tolist(),
                    bootstrap_p_Holm=np.round(contribution["p_Holm"], 10).tolist(),
                    observed_group_mean_matrix=np.round(contribution["observed_mean"], 8).tolist(),
                    adjusted_group_mean_matrix=np.round(contribution["adjusted_mean"], 8).tolist(),
                    definitions=contribution["definition"],
                    n_bootstrap=int(contribution["n_boot"]),
                    note="All four percentages are shown. Direct terms use observed compression (1-beta) as denominator; cross terms use the observed beta. Consequently beta_TT and beta_TR explicitly reveal denominator instability when their observed effects are near zero; this is part of the result, not evidence of a large stable contribution.",
                ),
            ),
        ),
        M2_specified_quadratic_effort_timing_constrained_manifold=dict(
            method="A log-spaced lambda-by-w grid is restricted to predictions whose beta_TR and beta_TT both lie within subject-bootstrap 95% confidence intervals. The remaining beta_RR and beta_RT predictions are compared with the joint 95% bootstrap ellipse of the observed spatial coefficients.",
            lambda_grid_range=[float(np.min(D["effort_grid"][:, 0])), float(np.max(D["effort_grid"][:, 0]))],
            w_grid_range=[float(np.min(D["effort_grid"][:, 1])), float(np.max(D["effort_grid"][:, 1]))],
            total_grid_points=int(len(D["effort_grid"])),
            timing_compatible_grid_points=int(len(D["effort_feasible"])),
            beta_TR_bootstrap_95_CI=np.round(D["timing_ci_tr"], 6).tolist(),
            beta_TT_bootstrap_95_CI=np.round(D["timing_ci_tt"], 6).tolist(),
            spatial_bootstrap_covariance=np.round(D["spatial_bootstrap_cov"], 8).tolist(),
            spatial_95_ellipse_mahalanobis2_threshold=float(stats.chi2.ppf(0.95, df=2)),
            timing_compatible_points_inside_spatial_95_ellipse=int(D["feasible_inside_observed_95"]),
            closest_timing_compatible_point=dict(
                source=D["closest_feasible_source"],
                lambda_=float(D["closest_feasible"][0]),
                w=float(D["closest_feasible"][1]),
                beta_RR=float(D["closest_feasible"][2]),
                beta_RT=float(D["closest_feasible"][3]),
                beta_TR=float(D["closest_feasible"][4]),
                beta_TT=float(D["closest_feasible"][5]),
                mahalanobis2=float(D["closest_feasible_mahalanobis2"]),
                inside_spatial_95_ellipse=bool(
                    D["closest_feasible_mahalanobis2"] <= stats.chi2.ppf(0.95, df=2)
                ),
            ),
        ),
        M2_specified_quadratic_effort_matrix_LOSO=dict(
            coefficient_order=D["matrix_labels"],
            model_residual_definition="Held-out effort-model prediction minus held-out observed coefficient.",
            baseline_residual_definition="Training-subject mean coefficient minus held-out observed coefficient.",
            model_residual_median=np.round(np.median(D["cv_model_matrix_residuals"], axis=0), 6).tolist(),
            baseline_residual_median=np.round(np.median(D["cv_baseline_matrix_residuals"], axis=0), 6).tolist(),
            model_bias_wilcoxon_p_raw=np.round(D["cv_matrix_bias_p"], 9).tolist(),
            model_bias_wilcoxon_p_Holm=np.round(D["cv_matrix_bias_p_holm"], 9).tolist(),
            model_vs_baseline_absolute_error_wilcoxon_p_raw=np.round(D["cv_matrix_abs_error_p"], 9).tolist(),
            model_vs_baseline_absolute_error_wilcoxon_p_Holm=np.round(D["cv_matrix_abs_error_p_holm"], 9).tolist(),
            overall_model_error_median=float(np.median(D["cv_model_errors"])),
            overall_baseline_error_median=float(np.median(D["cv_baseline_errors"])),
            overall_paired_error_p_raw=float(D["cv_error_p"]),
            overall_paired_error_p_adjusted=float(D["cv_error_p_adj"]),
            inference="Two-sided paired Wilcoxon signed-rank. This is the single planned headline comparison in the M2 LOSO error subpanel, so p_adj equals p_raw; only p_adj is displayed outside the data region.",
        ),
        M2_specified_quadratic_effort_gain_LOSO=dict(shown_in_figure=False,
                       scope="Retained as a decomposition of the M2 mismatch; not shown because panels B-C already provide structural and held-out tests.",
                       perceived_T_bins=[round(t, 3) for t in D["pT_bin"]],
                       data_gain_slope=round(D["emp_slope"], 4), data_slope_p=float(D["emp_slope_p"]),
                       LOSO_effort_model_slope_median=round(D["model_slope"], 4),
                       LOSO_data_slope_values=np.round(D["cv_data_slopes"], 6).tolist(),
                       LOSO_model_slope_values=np.round(D["cv_model_slopes"], 6).tolist(),
                       LOSO_paired_slope_difference_p=float(D["cv_slope_p"]),
                       LOSO_data_mean_gain_values=np.round(D["cv_data_mean_gain"], 6).tolist(),
                       LOSO_model_mean_gain_values=np.round(D["cv_model_mean_gain"], 6).tolist(),
                       LOSO_paired_gain_level_difference_p=float(D["cv_gain_level_p"]),
                       LOSO_gain_residual_median_by_duration=np.round(np.nanmedian(D["cv_gain_residuals"], axis=0), 6).tolist(),
                       LOSO_gain_residual_wilcoxon_p_raw=np.round(D["cv_gain_residual_p"], 9).tolist(),
                       LOSO_gain_residual_wilcoxon_p_Holm=np.round(D["cv_gain_residual_p_holm"], 9).tolist(),
                       LOSO_gain_median=np.round(D["model_gain"], 6).tolist(),
                       LOSO_gain_IQR=[np.round(D["model_gain_lo"], 6).tolist(), np.round(D["model_gain_hi"], 6).tolist()],
                       peak_speed_ratio=round(D["peak_ratio"], 2)),
        M2_specified_quadratic_effort_sign_check=dict(
            shown_in_figure=False,
            cost_partial_coefs={k: dict(median=round(v[0], 3), p=float(v[1])) for k, v in D["cost_stats"].items()},
            n_subjects=int(len(next(iter(D["cost_coefs"].values())))),
            median_condition_cells_per_split=float(np.median(D["cost_crossfit_cells"])),
            method="Within each subject and condition, alternating trials are split into two halves. Cost from one half predicts median produced radius in the other half; directions are swapped and coefficients averaged.",
            scope="Cross-trial splitting removes direct same-trajectory coupling but does not create a causal effort manipulation.",
        ),
        anisotropy_diagnostic=dict(
            method="Continuous first- and second-harmonic interactions with perceived-radius input in the produced-radius model. Start-direction labels are permuted within subject and target condition.",
            n_subjects=int(len(D["aniso_subject_ids"])),
            subject_ids=D["aniso_subject_ids"],
            subject_delta_r2=np.round(D["aniso_delta_r2"], 7).tolist(),
            median_delta_r2=float(D["aniso_observed"]),
            median_harmonic_amplitude=float(np.median(D["aniso_amplitude"])),
            n_permutations=int(D["n_aniso_permutations"]),
            permutation_null_95_percentile=float(np.percentile(D["aniso_null"], 95)),
            permutation_p=float(D["aniso_p"]),
            scope="A non-significant permutation test means no detected directional gain modulation; it does not establish isotropy.",
        ),
        M3_constant_K_spontaneous_two_thirds=dict(
            speed_radius_slope=round(D["slope_med"], 3), speed_vs_third_p=D["p_vs_third"],
            logK_vs_perceivedT_slope_median=round(D["k_slope"], 3),
            logK_vs_perceivedT_slope_values=np.round(D["k_subject_slopes"], 6).tolist(),
            logK_vs_perceivedT_slope_bootstrap_CI=np.round(D["k_slope_ci"], 6).tolist(),
            logK_vs_perceivedT_slope_p=float(D["k_slope_p"]),
            duration_radius_slope=round(D["timeR_slope"], 3), duration_vs_isochrony0_p=round(D["timeR_p0"], 3),
            duration_vs_third_p=D["timeR_p23"],
            duration_radius_x_variable="produced_radius",
            note="The data reject the specified constant-K predictions of speed-radius slope 1/3 and duration-radius slope 2/3. "
                 "This does not exclude two-thirds-law models with condition-dependent K or explicit instructed-duration control. "
                 "The non-significant test against duration-radius slope zero is not evidence of exact equivalence to zero."),
    )
    write_json(FIG_DIR / "alternative_behavior_model_legacy_stats.json", stats_out)
    serial_md = [
        "# SuppFig8 Alternative Behavior Models: Serial-History Summary",
        "",
        f"- Subjects: {S['n_subjects']}",
        f"- Strict normal-to-normal pairs per subject: median {np.median(S['transition_n']['normal']):.0f}",
        f"- Probe-to-normal pairs per subject: median {np.median(S['transition_n']['probe']):.0f}",
        "",
        "## Lag-1 target-history coefficients",
        "",
    ]
    for label, (i, j), p in zip(serial_labels, serial_indices, lag_p[0]):
        serial_md.append(
            f"- {label}: median {np.nanmedian(S['lag_H'][:, 0, i, j]):+.4f}, {_format_p(p)}"
        )
    serial_md.extend(["", "## Transition-specific and response-history controls", ""])
    for label, (i, j), p in zip(serial_labels, serial_indices, transition_difference_p):
        serial_md.append(
            f"- {label}: normal-to-normal {np.nanmedian(S['transition_H']['normal'][:, i, j]):+.4f}; "
            f"probe-to-normal {np.nanmedian(S['transition_H']['probe'][:, i, j]):+.4f}; paired {_format_p(p)}"
        )
    for label, (i, j), p in zip(["C_RR", "C_RT", "C_TR", "C_TT"], serial_indices, response_p):
        serial_md.append(
            f"- {label}: previous-response residual coefficient {np.nanmedian(S['response_C'][:, i, j]):+.4f}, {_format_p(p)}"
        )
    serial_md.append(
        "- Exact-sequence lag-1 Pearson r medians: "
        + ", ".join(f"{np.nanmedian(S['transition_r'][:, i, j]):+.4f}" for i, j in serial_indices)
    )
    serial_md.extend(["", "## Pooled within-day blocked prediction", ""])
    for name in ("lag1", "lag1_to_5", "exp_all", "exp_drawing"):
        values = pooled_validation["delta_R2_from_current"][name]
        serial_md.append(
            f"- {name}: median delta R2 = {values['median'][0]:+.6f} (R), "
            f"{values['median'][1]:+.6f} (T)"
        )
    serial_md.extend(["", "## Lag-1 raw adjustment shown in panel D", ""])
    lag1_delta = S["adjusted_delta"]["lag1_all"]
    lag1_delta_p = _holm_adjust(np.asarray([
        _safe_wilcoxon(lag1_delta[:, i, j]) for i, j in serial_indices
    ], dtype=float))
    for label, (i, j), p_value in zip(serial_labels, serial_indices, lag1_delta_p):
        serial_md.append(
            f"- {label.replace('H_', 'beta_')}: median delta-beta "
            f"{np.nanmedian(lag1_delta[:, i, j]):+.4f}, mean delta-beta "
            f"{np.nanmean(lag1_delta[:, i, j]):+.4f}, {_format_p(p_value)}"
        )
    serial_md.append(
        f"- Observed group-mean beta: {np.array2string(contribution['observed_mean'], precision=4)}"
    )
    serial_md.append(
        f"- Adjusted group-mean beta: {np.array2string(contribution['adjusted_mean'], precision=4)}"
    )
    serial_md.extend(["", "## Percentage sensitivity analysis (not displayed in panel D)", ""])
    for label, estimate, confidence_interval, p_value in zip(
        contribution["labels"], contribution["estimate"],
        contribution["confidence_interval"], contribution["p_Holm"]
    ):
        serial_md.append(
            f"- {label}: {estimate:+.2f}% [95% CI {confidence_interval[0]:+.2f}, "
            f"{confidence_interval[1]:+.2f}], {_format_p(p_value)}"
        )
    serial_md.extend([
        "",
        "Interpretation should distinguish a statistically detectable lag coefficient from meaningful held-out prediction and from a material change in the full behavior matrix.",
        "",
    ])
    (FIG_DIR / "alternative_behavior_model_legacy_stats.md").write_text(
        "\n".join(serial_md), encoding="utf-8"
    )
    print("serial summary:")
    print("\n".join(serial_md))
    print("saved:", png)


if __name__ == "__main__":
    main()
