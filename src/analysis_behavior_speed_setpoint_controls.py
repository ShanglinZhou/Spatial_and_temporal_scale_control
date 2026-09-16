"""Shared Figure 2 analyses for offset-aware R/T/V setpoint models.

This module contains reusable numerical helpers for Figure 2 and
Supplementary Figure 4.  It is not a formal manuscript-figure entry point;
``figure2_impl.py`` imports these helpers and creates
the two formal outputs.

Running this module directly retains the broader component audit used during
model development.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis_all_common import FIG_DIR, PLOT_ONLY_MODE, clean_axis, prepare_figure_for_export, write_json
from analysis_behavior_common import COLORS, label_panel, setup_style
from analysis_behavior_speed_setpoint import (
    OFFSET_MODEL_COLORS,
    OFFSET_MODEL_ORDER,
    RAW_SCALE,
    bootstrap_median_ci,
    collect_condition_kinematics,
    condition_medians,
    cross_validate_offset_models,
    decode_offset_model,
    fit_offset_model,
    holm_adjust,
    kinematic_speed_slopes,
    load_subject_data,
    offset_model_parameter_count,
    predict_offset_model,
    subject_speed_statistics,
    wilcoxon,
)


C_R, C_T, C_V = COLORS["R"], COLORS["T"], (0.25, 0.25, 0.25)
MODELS = OFFSET_MODEL_ORDER
MECHANISTIC_MODELS = MODELS[:-1]
CACHE_DIR = Path(__file__).resolve().parent / "results" / "behavior_speed_setpoint"
MAIN_CV_CACHE = CACHE_DIR / "main_offset_raw_loco_v4.json"
STRICT_CACHE = CACHE_DIR / "supp_offset_strict_cv_v4.json"
RECOVERY_CACHE = CACHE_DIR / "supp_offset_recovery_v3.json"


def format_p(value):
    value = float(value)
    return f"p={value:.1e}" if value < 0.001 else f"p={value:.3f}"


def clean_no_locator(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", colors="black")


def boxplot(ax, groups, positions, colors, widths=0.60):
    result = ax.boxplot(
        groups,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
    )
    for patch, color in zip(result["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
    return result


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True)
    return np.percentile(np.median(samples, axis=1), [2.5, 97.5])


def load_or_run_cv(subjects):
    """Load or compute the locked condition-wise Figure 2 model comparison."""
    if MAIN_CV_CACHE.exists():
        with MAIN_CV_CACHE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        predictions = {
            model: {
                subject: np.asarray(values, dtype=float)
                for subject, values in by_subject.items()
            }
            for model, by_subject in payload["predictions"].items()
        }
        return payload["records"], predictions
    if PLOT_ONLY_MODE:
        raise RuntimeError(
            f"Plot-only mode requires the locked cross-validation cache: {MAIN_CV_CACHE}"
        )
    records, predictions = cross_validate_offset_models(
        subjects, MODELS, scheme="condition"
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with MAIN_CV_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "records": records,
                "predictions": {
                    model: {
                        subject: values.tolist()
                        for subject, values in by_subject.items()
                    }
                    for model, by_subject in predictions.items()
                },
            },
            handle,
            indent=2,
        )
    return records, predictions


def raw_duration_statistics(subjects):
    output = []
    for subject, data in subjects.items():
        target = np.exp(data.target)
        produced = np.exp(data.produced)
        design = np.column_stack([np.ones(len(target)), target])
        beta, *_ = np.linalg.lstsq(design, produced[:, 1], rcond=None)
        output.append(
            {
                "subject": subject,
                "intercept": float(beta[0]),
                "beta_TR": float(beta[1]),
                "beta_TT": float(beta[2]),
            }
        )
    return output


def offset_corrected_speed_statistics(subjects):
    output = []
    c_t_by_subject = {}
    for subject, data in subjects.items():
        fitted = fit_offset_model("O", data.target, data.produced, n_starts=4)
        c_t = decode_offset_model("O", fitted)["c_t"]
        c_t_by_subject[subject] = c_t
        target = np.exp(data.target)
        produced = np.exp(data.produced)
        x = np.log(target[:, 0] / np.maximum(target[:, 1] + c_t, 0.05))
        y = np.log(produced[:, 0] / produced[:, 1])
        design = np.column_stack([np.ones(len(x)), x])
        intercept, gain = np.linalg.lstsq(design, y, rcond=None)[0]
        fixed = (
            np.exp(intercept / (1.0 - gain))
            if abs(1.0 - gain) > 0.02
            else np.nan
        )
        output.append(
            {
                "subject": subject,
                "c_t": float(c_t),
                "intercept": float(intercept),
                "gain": float(gain),
                "fixed_speed": float(fixed),
            }
        )
    return output, c_t_by_subject


def offset_speed_condition_medians(subjects, c_t_by_subject):
    keys = sorted(
        {
            (float(radius), float(duration))
            for data in subjects.values()
            for radius, duration in zip(data.stim_r, data.stim_t)
        },
        key=lambda key: (key[1], key[0]),
    )
    target_medians = []
    produced_medians = []
    for stimulus_radius, stimulus_duration in keys:
        target_speed = []
        produced_speed = []
        for subject, data in subjects.items():
            mask = np.isclose(data.stim_r, stimulus_radius) & np.isclose(
                data.stim_t, stimulus_duration
            )
            target = np.exp(data.target[mask])
            produced = np.exp(data.produced[mask])
            target_speed.extend(
                target[:, 0] / (target[:, 1] + c_t_by_subject[subject])
            )
            produced_speed.extend(produced[:, 0] / produced[:, 1])
        target_medians.append(float(np.median(target_speed)))
        produced_medians.append(float(np.median(produced_speed)))
    return np.asarray(target_medians), np.asarray(produced_medians)


def load_or_run_strict(subjects):
    if STRICT_CACHE.exists():
        with STRICT_CACHE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if PLOT_ONLY_MODE:
        raise RuntimeError(f"Plot-only mode requires the locked strict-CV cache: {STRICT_CACHE}")
    payload = {}
    for scheme in ("radius", "duration"):
        records, _ = cross_validate_offset_models(subjects, MODELS, scheme=scheme)
        payload[scheme] = records
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with STRICT_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def model_recovery_offset(subjects, n_rep=30, seed=20260811):
    models = ["R", "T", "V", "RT", "RV", "TV", "RTV"]
    medians = condition_medians(subjects)
    target = medians["target"]
    produced_raw = np.exp(medians["produced"])
    fitted = {model: fit_offset_model(model, target, medians["produced"], n_starts=6) for model in models}
    reference = predict_offset_model("RV", target, fitted["RV"])
    residual_sd = np.maximum(np.std(produced_raw - reference, axis=0, ddof=1), np.array([2.0, 0.04]))
    rng = np.random.RandomState(seed)
    counts = np.zeros((len(models), len(models)), dtype=int)
    for i, generating in enumerate(models):
        mean = predict_offset_model(generating, target, fitted[generating])
        for _ in range(n_rep):
            simulated = np.maximum(mean + rng.normal(0.0, residual_sd, size=mean.shape), np.array([1.0, 0.05]))
            simulated_log = np.log(simulated)
            bic = []
            for candidate in models:
                fit = fit_offset_model(candidate, target, simulated_log, n_starts=2)
                prediction = predict_offset_model(candidate, target, fit)
                rss = float(np.sum(((prediction - simulated) / RAW_SCALE[None, :]) ** 2))
                n = simulated.size
                bic.append(n * math.log(max(rss / n, 1e-12)) + offset_model_parameter_count(candidate) * math.log(n))
            counts[i, int(np.argmin(bic))] += 1
    return {"models": models, "counts": counts.tolist(), "proportions": (counts / n_rep).tolist(), "n_rep": n_rep, "residual_sd_raw": residual_sd.tolist()}


def load_or_run_recovery(subjects):
    if RECOVERY_CACHE.exists():
        with RECOVERY_CACHE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if PLOT_ONLY_MODE:
        raise RuntimeError(f"Plot-only mode requires the locked recovery cache: {RECOVERY_CACHE}")
    payload = model_recovery_offset(subjects)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with RECOVERY_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def offset_corrected_gains(subjects):
    output = []
    for subject, data in subjects.items():
        fitted = fit_offset_model("O", data.target, data.produced, n_starts=4)
        c_t = decode_offset_model("O", fitted)["c_t"]
        target = np.exp(data.target); produced = np.exp(data.produced)
        x = np.log(target[:, 0] / (target[:, 1] + c_t)); y = np.log(produced[:, 0] / produced[:, 1])
        design = np.column_stack([np.ones(len(x)), x])
        intercept, gain = np.linalg.lstsq(design, y, rcond=None)[0]
        output.append({"subject": subject, "c_t": float(c_t), "gain": float(gain), "intercept": float(intercept)})
    return output


def main():
    setup_style()
    rows, subjects = load_subject_data()
    medians = condition_medians(subjects)
    strict = load_or_run_strict(subjects)
    recovery = load_or_run_recovery(subjects)
    uncorrected = subject_speed_statistics(subjects)
    corrected = offset_corrected_gains(subjects)
    corrected_map = {record["subject"]: record for record in corrected}
    kinematic_records = collect_condition_kinematics(rows)
    kinematic = kinematic_speed_slopes(kinematic_records)
    kinematic_map = {record["subject"]: record for record in kinematic}
    common = [subject for subject in subjects if subject in kinematic_map]

    # Raw subject coefficients and duration offsets by stimulus level.
    raw_coefficients = []
    duration_levels = sorted({float(value) for data in subjects.values() for value in data.stim_t})
    offset_by_level = []
    for subject, data in subjects.items():
        target = np.exp(data.target); produced = np.exp(data.produced)
        design = np.column_stack([np.ones(len(target)), target])
        beta_r, *_ = np.linalg.lstsq(design, produced[:, 0], rcond=None)
        beta_t, *_ = np.linalg.lstsq(design, produced[:, 1], rcond=None)
        raw_coefficients.append({"subject": subject, "beta_RR": float(beta_r[1]), "beta_RT": float(beta_r[2]), "beta_TR": float(beta_t[1]), "beta_TT": float(beta_t[2])})
        offset_by_level.append([float(np.median(produced[np.isclose(data.stim_t, level), 1] - target[np.isclose(data.stim_t, level), 1])) for level in duration_levels])
    offset_by_level = np.asarray(offset_by_level)
    beta_rr = np.asarray([record["beta_RR"] for record in raw_coefficients])

    # Four speed metrics.
    uncorrected_map = {record["subject"]: record for record in uncorrected}
    gain_groups = [
        np.asarray([uncorrected_map[subject]["gain"] for subject in common]),
        np.asarray([corrected_map[subject]["gain"] for subject in common]),
        np.asarray([kinematic_map[subject]["mean_path_gain"] for subject in common]),
        np.asarray([kinematic_map[subject]["peak_gain"] for subject in common]),
    ]
    gain_tests = [wilcoxon(values, 1.0, "less") for values in gain_groups]
    adjusted = holm_adjust([record["p"] for record in gain_tests])
    for record, value in zip(gain_tests, adjusted): record["p_holm"] = float(value)

    # Conditional setpoint strengths in full RTV fits and RV setpoint locations.
    rtv_parameters, rv_parameters = [], []
    normalized_setpoints = []
    for subject, data in subjects.items():
        rtv = decode_offset_model("RTV", fit_offset_model("RTV", data.target, data.produced, n_starts=6)); rtv["subject"] = subject; rtv_parameters.append(rtv)
        rv = decode_offset_model("RV", fit_offset_model("RV", data.target, data.produced, n_starts=6)); rv["subject"] = subject; rv_parameters.append(rv)
        target = np.exp(data.target)
        speed = target[:, 0] / np.maximum(target[:, 1] + rv["c_t"], 0.05)
        z_r = (np.log(rv["r0"]) - np.log(target[:, 0].min())) / (np.log(target[:, 0].max()) - np.log(target[:, 0].min()))
        z_v = (np.log(rv["v0"]) - np.log(speed.min())) / (np.log(speed.max()) - np.log(speed.min()))
        normalized_setpoints.append({"subject": subject, "R0_normalized": float(z_r), "V0_normalized": float(z_v)})
    strength_r = np.asarray([record["lambda_r"] / (1.0 + record["lambda_r"]) for record in rtv_parameters])
    strength_t = np.asarray([record["lambda_t"] / (1.0 + record["lambda_t"]) for record in rtv_parameters])
    strength_v = np.asarray([record["lambda_v"] / (1.0 + record["lambda_v"]) for record in rtv_parameters])
    r_greater_t = wilcoxon(strength_r - strength_t, 0.0, "greater")
    v_greater_t = wilcoxon(strength_v - strength_t, 0.0, "greater")

    fig = plt.figure(figsize=(12.0, 6.4))
    grid = fig.add_gridspec(2, 4, left=0.055, right=0.96, bottom=0.15, top=0.94, wspace=0.48, hspace=0.58)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(4)]
    rng = np.random.RandomState(20260811)

    # A. Raw spatial central tendency.
    ax = axes[0]
    target_r = np.exp(medians["target"][:, 0]); produced_r = np.exp(medians["produced"][:, 0])
    ax.scatter(target_r, produced_r, s=22, facecolor=(0.92, 0.92, 0.92), edgecolor="0.20", linewidth=0.6)
    limits = [min(target_r.min(), produced_r.min()) * 0.90, max(target_r.max(), produced_r.max()) * 1.08]
    xx = np.linspace(limits[0], limits[1], 120)
    ax.plot(xx, xx, color="0.60", lw=0.8, ls="--")
    slope = float(np.median(beta_rr)); intercept = float(np.median(produced_r) - slope * np.median(target_r))
    ax.plot(xx, intercept + slope * xx, color=C_R, lw=1.8)
    ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel("Perceived radius"); ax.set_ylabel("Produced radius")
    ax.set_title("Raw spatial central tendency")
    rr_less_one = wilcoxon(beta_rr, 1.0, "less")
    ax.text(0.04, 0.96, f"Median $\\beta_{{RR}}$={slope:.2f}\nvs 1: {format_p(rr_less_one['p'])}", transform=ax.transAxes, ha="left", va="top")
    clean_axis(ax)

    # B. Constant duration offset across levels.
    ax = axes[1]
    positions = np.arange(1, len(duration_levels) + 1)
    boxplot(ax, [offset_by_level[:, i] for i in range(len(duration_levels))], positions, [C_T] * len(duration_levels))
    for i in range(len(duration_levels)):
        ax.scatter(positions[i] + rng.normal(0.0, 0.04, len(subjects)), offset_by_level[:, i], s=7, color=C_T, alpha=0.65, edgecolor="none")
    friedman = stats.friedmanchisquare(*[offset_by_level[:, i] for i in range(len(duration_levels))])
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xticks(positions); ax.set_xticklabels([f"{level:.1f}" for level in duration_levels])
    ax.set_xlabel("Stimulus duration"); ax.set_ylabel(r"$T_{produced}-T_{perceived}$ (s)")
    ax.set_title("Duration offset across levels")
    ax.text(0.98, 0.96, f"Friedman {format_p(friedman.pvalue)}", transform=ax.transAxes, ha="right", va="top")
    clean_no_locator(ax)

    # C. Robust speed compression metrics.
    ax = axes[2]
    positions = np.arange(1, 5)
    colors = [(0.50, 0.50, 0.50), C_V, C_R, C_T]
    boxplot(ax, gain_groups, positions, colors)
    for position, values, color in zip(positions, gain_groups, colors):
        ax.scatter(position + rng.normal(0.0, 0.04, len(values)), values, s=8, color=color, alpha=0.72, edgecolor="none")
    ax.axhline(1, color="0.55", lw=0.8, ls="--")
    ax.set_xticks(positions); ax.set_xticklabels(["Geo.", "Corrected", "Path", "Peak"])
    ax.set_ylabel("Log-log speed gain")
    ax.set_title("Speed-metric robustness")
    ax.text(0.98, 0.96, f"All vs 1: {format_p(max(record['p_holm'] for record in gain_tests))}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    clean_no_locator(ax)

    # D. Conditional strengths in the full model.
    ax = axes[3]
    strengths = [strength_r, strength_t, strength_v]
    positions = np.arange(1, 4)
    boxplot(ax, strengths, positions, [C_R, C_T, C_V])
    for position, values, color in zip(positions, strengths, [C_R, C_T, C_V]):
        ax.scatter(position + rng.normal(0.0, 0.04, len(values)), values, s=8, color=color, alpha=0.72, edgecolor="none")
    ax.set_xticks(positions); ax.set_xticklabels(["R", "T", "V"])
    ax.set_ylabel(r"Conditional strength, $\lambda/(1+\lambda)$")
    ax.set_title("Conditional strengths")
    ax.text(0.98, 0.96, f"R > T: {format_p(r_greater_t['p'])}\nV > T: {format_p(v_greater_t['p'])}", transform=ax.transAxes, ha="right", va="top")
    clean_axis(ax)

    # E-F. Strict extrapolation.
    strict_tests = {}
    for ax, scheme, title in zip(axes[4:6], ["radius", "duration"], ["Leave-one-radius-level-out", "Leave-one-duration-level-out"]):
        by_model = {model: np.asarray([record["rmse"] for record in strict[scheme] if record["model"] == model]) for model in MODELS}
        groups = [by_model[model] for model in MECHANISTIC_MODELS]
        positions = np.arange(1, len(MECHANISTIC_MODELS) + 1)
        boxplot(ax, groups, positions, [OFFSET_MODEL_COLORS[model] for model in MECHANISTIC_MODELS], widths=0.58)
        for position, values in zip(positions, groups):
            ax.scatter(position + rng.normal(0.0, 0.035, len(values)), values, s=6, color="black", alpha=0.38, edgecolor="none")
        ax.axhline(float(np.median(by_model["FULL"])), color="black", lw=0.9, ls=":", label="Full affine median")
        display_labels = [r"$RV_\rho$" if model == "RVrho" else model for model in MECHANISTIC_MODELS]
        ax.set_xticks(positions); ax.set_xticklabels(display_labels, fontsize=7.2)
        ax.set_ylabel("Held-out raw joint RMSE"); ax.set_title(title)
        ax.legend(frameon=False, fontsize=7.3, loc="upper right")
        clean_no_locator(ax)
        rv_less_r = wilcoxon(by_model["RV"] - by_model["R"], 0.0, "less")
        rvrho_less_rv = wilcoxon(by_model["RVrho"] - by_model["RV"], 0.0, "less")
        rtv_less_rv = wilcoxon(by_model["RTV"] - by_model["RV"], 0.0, "less")
        full_less_rvrho = wilcoxon(by_model["FULL"] - by_model["RVrho"], 0.0, "less")
        strict_tests[scheme] = {"model_medians": {model: float(np.median(values)) for model, values in by_model.items()}, "RV_less_R": rv_less_r, "RVrho_less_RV": rvrho_less_rv, "RTV_less_RV": rtv_less_rv, "FULL_less_RVrho": full_less_rvrho}
        ax.text(0.98, 0.72, f"RV < R: {format_p(rv_less_r['p'])}\n" + rf"$RV_\rho$ < RV: {format_p(rvrho_less_rv['p'])}" + f"\nFull < " + r"$RV_\rho$" + f": {format_p(full_less_rvrho['p'])}", transform=ax.transAxes, ha="right", va="top", fontsize=7.1)

    # G. Setpoint locations relative to tested ranges.
    ax = axes[6]
    z_r = np.asarray([record["R0_normalized"] for record in normalized_setpoints]); z_v = np.asarray([record["V0_normalized"] for record in normalized_setpoints])
    boxplot(ax, [z_r, z_v], [1, 2], [C_R, C_V])
    ax.scatter(1 + rng.normal(0.0, 0.04, len(z_r)), z_r, s=8, color=C_R, alpha=0.72, edgecolor="none")
    ax.scatter(2 + rng.normal(0.0, 0.04, len(z_v)), z_v, s=8, color=C_V, alpha=0.72, edgecolor="none")
    ax.axhspan(0, 1, color=(0.92, 0.92, 0.92), zorder=0)
    ax.axhline(0, color="0.65", lw=0.7); ax.axhline(1, color="0.65", lw=0.7)
    ax.set_xticks([1, 2]); ax.set_xticklabels([r"$R_0$", r"$V_0$"])
    ax.set_ylabel("Normalized log setpoint location")
    ax.set_title("Setpoint heterogeneity")
    ax.text(0.04, 0.04, "0 = tested minimum\n1 = tested maximum", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5)
    clean_axis(ax)

    # H. Model recovery.
    ax = axes[7]
    matrix = np.asarray(recovery["proportions"], dtype=float); labels = recovery["models"]
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.3)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=7.3)
    ax.set_xlim(-0.5, len(labels) - 0.5); ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("Recovered model"); ax.set_ylabel("Generating model")
    ax.set_title("Offset-aware model recovery")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.2, color="white" if matrix[i, j] > 0.55 else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03); cbar.set_label("Proportion"); cbar.ax.tick_params(direction="out", labelsize=8)

    for axis, letter in zip(axes[:7], "ABCDEFG"):
        label_panel(axis, letter)
    axes[7].text(-0.30, 1.08, "H", transform=axes[7].transAxes, fontsize=12, fontweight="bold", ha="left", va="top")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / "Fig2_behavior_speed_setpoint_component_audit.png"
    prepare_figure_for_export(fig)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)

    stats_payload = {
        "analysis": "Robustness and identifiability of offset-aware R/T/V setpoint models",
        "n_subjects": len(subjects),
        "n_kinematic_condition_cells": len(kinematic_records),
        "panel_A": {"subject_raw_coefficients": raw_coefficients, "beta_RR_median": float(np.median(beta_rr)), "beta_RR_vs_one": rr_less_one},
        "panel_B": {"duration_levels": duration_levels, "subject_offsets_by_level": offset_by_level.tolist(), "friedman": {"statistic": float(friedman.statistic), "p": float(friedman.pvalue)}},
        "panel_C": {"metric_order": ["geometric", "offset_corrected_geometric", "mean_path", "peak"], "median_gains": [float(np.median(values)) for values in gain_groups], "median_gain_ci": [bootstrap_median_ci(values) for values in gain_groups], "less_than_one_holm": gain_tests},
        "panel_D": {"RTV_subject_parameters": rtv_parameters, "strength_medians": {"R": float(np.median(strength_r)), "T": float(np.median(strength_t)), "V": float(np.median(strength_v))}, "R_greater_T": r_greater_t, "V_greater_T": v_greater_t},
        "panels_E_F": {"strict_cross_validation_records": strict, "planned_tests": strict_tests},
        "panel_G": {"RV_subject_parameters": rv_parameters, "normalized_setpoint_locations": normalized_setpoints, "interpretation": "Values outside 0-1 indicate fitted fixed points outside the subject's tested input range and should not be interpreted as precisely localized preferences."},
        "panel_H": {**recovery, "interpretation": "Recovery uses empirically fitted parameter regimes. RTV simulations are recovered as RV because the fitted T component contributes too little independent structure to justify its extra parameters."},
        "figure": str(output_path),
    }
    write_json(FIG_DIR / "Fig2_behavior_speed_setpoint_component_audit_stats.json", stats_payload)
    print(f"Saved {output_path}")
    print("Median speed gains: " + ", ".join(f"{np.median(values):.3f}" for values in gain_groups))
    print("Recovery diagonal: " + ", ".join(f"{label}={matrix[i, i]:.2f}" for i, label in enumerate(labels)))


if __name__ == "__main__":
    main()
