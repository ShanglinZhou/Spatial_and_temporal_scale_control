"""Build the unified behavioral-computation Figure 2 and its supplement.

Figure 2 separates the preferred radius/dynamic-speed model family
from four independently labelled alternatives:

AM1: specified timing-constrained quadratic-effort model
AM2: static proportional-noise model
AM3: empirical serial-dependence model
AM4: constant-K/two-thirds-law model

The compact supplement contains only controls that directly support the main
figure: offset specification, speed-metric robustness, and allocation-parameter
recovery. All displayed summary points are descriptive; inference is based on
subject-level estimates.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    from lxml import etree as LET
except ModuleNotFoundError as error:
    if not (error.name == "lxml" or str(error.name).startswith("lxml.")):
        raise
    LET = None
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import NullFormatter
from scipy import optimize, stats

from analysis_all_common import FIG_DIR, PLOT_ONLY_MODE, clean_axis, prepare_figure_for_export, save_vector_figure, significance_symbol, write_json
from analysis_behavior_common import COLORS, bh_fdr, label_panel, setup_style, wilcoxon_test
from analysis_behavior_speed_setpoint import (
    LOG_FIDELITY_SCALE,
    LOG_SPEED_SCALE,
    OFFSET_MODEL_COLORS,
    OFFSET_MODEL_ORDER,
    RAW_SCALE,
    bootstrap_median_ci,
    collect_condition_kinematics,
    condition_medians,
    decode_offset_model,
    fit_offset_model,
    holm_adjust,
    kinematic_speed_slopes,
    load_subject_data,
    normalized_control_matrix,
    predict_offset_model,
    subject_speed_statistics,
    wilcoxon,
)
from analysis_behavior_speed_setpoint_controls import (
    bootstrap_ci,
    load_or_run_cv,
    load_or_run_recovery,
    load_or_run_strict,
    offset_corrected_gains,
    offset_corrected_speed_statistics,
    offset_speed_condition_medians,
    raw_duration_statistics,
)
from analysis_alternative_behavior_models import (
    SERIAL_LAGS,
    _boot_ci,
    _holm_adjust,
    _safe_wilcoxon,
    compute as compute_alternatives,
)
import analysis_alternative_behavior_models as alternative_analysis


C_R = COLORS["R"]
C_T = COLORS["T"]
C_V = (0.25, 0.25, 0.25)
C_MODEL = (0.48, 0.48, 0.48)
MODELS = OFFSET_MODEL_ORDER
DISPLAY_MODELS = ["O", "R", "T", "V", "RT", "RV", "RVrho", "TV", "RTV", "FULL"]
DISPLAY_LABELS = ["O", "R", "T", "V", "RT", "RV", r"$RV_\rho$", "TV", "RTV", "Full"]
AM_NAMES = {
    "AM1": "Specified quadratic effort",
    "AM2": "Static proportional noise",
    "AM3": "Empirical serial dependence",
    "AM4": "Constant-K two-thirds law",
}
SYMMETRIC_CACHE = Path(__file__).resolve().parent / "results" / "behavior_speed_setpoint" / "symmetric_cr_rvrho_loco_v1.json"
RHO_RECOVERY_CACHE = Path(__file__).resolve().parent / "results" / "behavior_speed_setpoint" / "rvrho_parameter_recovery_v1.json"
ALTERNATIVE_REFERENCE_CACHE = Path(__file__).resolve().parent / "results" / "behavior_model_comparison" / "alternative_model_reference_stats_v1.json"
ALTERNATIVE_PLOT_CACHE = Path(__file__).resolve().parent / "results" / "behavior_model_comparison" / "alternative_model_plot_data_v1.npy"
TEST_FIG_DIR = FIG_DIR.parent / "tests"


def format_p(value: float) -> str:
    return significance_symbol(value)


def format_pearson_p(value: float) -> str:
    """Keep Pearson-correlation P values numeric."""
    value = float(value)
    if not np.isfinite(value):
        return r"$P = \mathrm{n/a}$"
    if value < 0.001:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10.0 ** exponent)
        return rf"$P = {mantissa:.3f}\times10^{{{exponent}}}$"
    return rf"$P = {value:.3f}$"


_FIG2_FORMULA_SPECS = {}
_SVG_NS = "http://www.w3.org/2000/svg"
_MATH_COMMANDS = {
    "beta": ("β", "italic"),
    "rho": ("ρ", "italic"),
    "Delta": ("Δ", "italic"),
    "log": ("log", "normal"),
    "times": ("×", "normal"),
    "rightarrow": ("→", "normal"),
    "to": ("→", "normal"),
}


def register_formula_text(artist, gid: str):
    """Register a MathText artist for phrase-level Illustrator SVG output."""
    artist.set_gid(gid)
    _FIG2_FORMULA_SPECS[gid] = artist.get_text()
    return artist


def _append_svg_run(runs, text, style="normal", script=None):
    if not text:
        return
    text = text.replace(" ", "\xa0")
    if runs and runs[-1][1:] == (style, script):
        runs[-1] = (runs[-1][0] + text, style, script)
    else:
        runs.append((text, style, script))


def _consume_tex_group(source: str, start: int):
    if start >= len(source):
        return "", start
    if source[start] != "{":
        if source[start] == "\\":
            match = re.match(r"\\[A-Za-z]+|\\.", source[start:])
            token = match.group(0) if match else source[start]
            return token, start + len(token)
        return source[start], start + 1
    depth = 1
    index = start + 1
    while index < len(source) and depth:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
        index += 1
    if depth:
        raise ValueError(f"Unclosed MathText group: {source!r}")
    return source[start + 1 : index - 1], index


def _parse_math_runs(source: str, *, script=None, forced_style=None):
    runs = []
    index = 0
    while index < len(source):
        char = source[index]
        if char in "_^":
            group, index = _consume_tex_group(source, index + 1)
            for text, style, _ in _parse_math_runs(
                group,
                script="sub" if char == "_" else "super",
                forced_style=forced_style,
            ):
                _append_svg_run(runs, text, style, "sub" if char == "_" else "super")
            continue
        if char == "\\":
            if index + 1 < len(source) and source[index + 1] in {" ", ","}:
                _append_svg_run(runs, " ", "normal", script)
                index += 2
                continue
            match = re.match(r"\\([A-Za-z]+)", source[index:])
            if match is None:
                index += 1
                continue
            command = match.group(1)
            index += len(match.group(0))
            if command == "mathrm":
                group, index = _consume_tex_group(source, index)
                for text, _, nested_script in _parse_math_runs(
                    group, script=script, forced_style="normal"
                ):
                    _append_svg_run(runs, text, "normal", nested_script)
                continue
            text, style = _MATH_COMMANDS.get(command, (command, "normal"))
            _append_svg_run(runs, text, forced_style or style, script)
            continue
        if char in "{}":
            index += 1
            continue
        if char == "-":
            char = "−"
        style = forced_style or ("italic" if char.isalpha() else "normal")
        _append_svg_run(runs, char, style, script)
        index += 1
    return runs


def _parse_mixed_svg_runs(line: str):
    runs = []
    parts = re.split(r"(\$[^$]*\$)", line)
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            for text, style, script in _parse_math_runs(part[1:-1]):
                _append_svg_run(runs, text, style, script)
        else:
            _append_svg_run(runs, part, "normal", None)
    return runs


def _style_font_size(style: str):
    match = re.search(r"font-size:([0-9.]+)px", style or "")
    return float(match.group(1)) if match else float("nan")


def _rewrite_formula_text_element(text_element, line: str) -> None:
    tspans = list(text_element)
    if not tspans:
        raise ValueError(f"Expected MathText tspans for {line!r}")
    x_values = []
    base_candidates = []
    base_size = 0.0
    for tspan in tspans:
        size = _style_font_size(tspan.get("style", ""))
        base_size = max(base_size, size if np.isfinite(size) else 0.0)
    for tspan in tspans:
        x_values.extend(float(value) for value in (tspan.get("x") or "").split())
        size = _style_font_size(tspan.get("style", ""))
        if np.isclose(size, base_size):
            base_candidates.extend(float(value) for value in (tspan.get("y") or "").split())
    if not x_values or not base_candidates:
        raise ValueError(f"Missing MathText coordinates for {line!r}")
    x_start = min(x_values)
    baseline = float(np.median(base_candidates))
    for child in list(text_element):
        text_element.remove(child)
    text_element.text = None
    for run_index, (text, font_style, script) in enumerate(_parse_mixed_svg_runs(line)):
        tspan = LET.SubElement(text_element, f"{{{_SVG_NS}}}tspan")
        size = base_size if script is None else 0.70 * base_size
        tspan.set(
            "style",
            f"font-family:Arial;font-size:{size:g}px;font-style:{font_style};font-weight:normal;",
        )
        if run_index == 0:
            tspan.set("x", f"{x_start:g}")
            tspan.set("y", f"{baseline:g}")
        if script is not None:
            tspan.set("baseline-shift", script)
        tspan.text = text
        tspan.tail = None


def rewrite_illustrator_formula_text(svg_path: Path) -> None:
    """Keep MathText typography while removing per-glyph SVG positioning."""
    if LET is None:
        return
    parser = LET.XMLParser(remove_blank_text=False, strip_cdata=False)
    tree = LET.parse(str(svg_path), parser)
    root = tree.getroot()
    for gid, source in _FIG2_FORMULA_SPECS.items():
        groups = root.xpath(f".//*[@id='{gid}']")
        if len(groups) != 1:
            raise ValueError(f"Expected one SVG group for {gid}, found {len(groups)}")
        text_elements = groups[0].xpath(".//*[local-name()='text']")
        source_lines = source.split("\n")
        if len(text_elements) != len(source_lines):
            raise ValueError(
                f"SVG line mismatch for {gid}: {len(text_elements)} elements, "
                f"{len(source_lines)} source lines"
            )
        for text_element, line in zip(text_elements, source_lines):
            if "$" in line:
                _rewrite_formula_text_element(text_element, line)
    doctype = tree.docinfo.doctype
    tree.write(
        str(svg_path),
        encoding="utf-8",
        xml_declaration=True,
        doctype=doctype,
        pretty_print=False,
    )


def format_reported_p(value: float) -> str:
    """Format non-Pearson inference with the manuscript-wide symbols."""
    return significance_symbol(value)


def am2_inference(D):
    """Return the two correlation tests and the separate paired-noise test."""
    space_r, space_p_raw = stats.pearsonr(D["CVR"], D["CR"])
    time_r, time_p_raw = stats.pearsonr(D["CVT"], D["CT"])
    space_p_bh, time_p_bh = bh_fdr([space_p_raw, time_p_raw])
    noise_test = wilcoxon_test(D["CVT"], D["CVR"])
    return {
        "space": {"r": float(space_r), "p_raw": float(space_p_raw), "p_bh": float(space_p_bh)},
        "time": {"r": float(time_r), "p_raw": float(time_p_raw), "p_bh": float(time_p_bh)},
        "time_vs_space_noise": noise_test,
        "correlation_multiple_comparison": (
            "Benjamini-Hochberg FDR across the two planned Pearson correlations "
            "in the left AM2 subpanel."
        ),
        "noise_multiple_comparison": (
            "No adjustment: the right AM2 subpanel contains one planned paired comparison."
        ),
    }


def tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", colors="black")
    # Keep explicit categorical x ticks intact; numerical x axes in this
    # figure are explicitly set or already contain no more than five ticks.
    ax.locator_params(axis="y", nbins=5)


def make_axes_square(fig, axes) -> None:
    """Center axes in their allocated cells with equal physical width/height."""
    fig.canvas.draw()
    figure_width, figure_height = fig.get_size_inches()
    for ax in axes:
        position = ax.get_position()
        physical_width = position.width * figure_width
        physical_height = position.height * figure_height
        if physical_width > physical_height:
            new_width = physical_height / figure_width
            x0 = position.x0 + (position.width - new_width) / 2.0
            ax.set_position([x0, position.y0, new_width, position.height], which="both")
        else:
            new_height = physical_width / figure_height
            y0 = position.y0 + (position.height - new_height) / 2.0
            ax.set_position([position.x0, y0, position.width, new_height], which="both")


def draw_box(
    ax,
    values,
    position,
    color,
    *,
    width=0.55,
    filled=True,
    points=True,
    rng=None,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    result = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": color, "linewidth": 0.8},
    )
    result["boxes"][0].set_facecolor(color if filled else "white")
    result["boxes"][0].set_alpha(0.72 if filled else 1.0)
    if points:
        if rng is None:
            rng = np.random.RandomState(20260811)
        ax.scatter(
            position + rng.uniform(-0.055, 0.055, len(values)),
            values,
            s=8,
            color=color,
            alpha=0.68,
            edgecolor="none",
            zorder=3,
        )


def annotate_box_p_values(ax, groups, positions, p_values) -> None:
    """Place the manuscript-wide significance symbol above each box."""
    finite_groups = []
    for group in groups:
        values = np.asarray(group, dtype=float)
        finite_groups.append(values[np.isfinite(values)])
    pooled = np.concatenate([group for group in finite_groups if len(group)])
    data_min, data_max = float(np.min(pooled)), float(np.max(pooled))
    data_span = max(
        data_max - data_min,
        0.12 * max(abs(data_min), abs(data_max)),
        1e-6,
    )
    label_y = data_max + 0.10 * data_span
    lower, upper = ax.get_ylim()
    ax.set_ylim(
        min(lower, data_min - 0.08 * data_span),
        max(upper, label_y + 0.12 * data_span),
    )
    for position, p_value in zip(positions, p_values):
        ax.text(
            position,
            label_y,
            significance_symbol(float(p_value)),
            ha="center",
            va="bottom",
        )


def draw_comparison_bracket(ax, x_left, x_right, y, label, *, height=0.012):
    """Connect two displayed model boxes and place a comparison label above them."""
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y - height, y, y, y - height],
        color="black",
        lw=0.8,
        clip_on=False,
    )
    return ax.text(
        (x_left + x_right) / 2.0,
        y + 0.007,
        label,
        ha="center",
        va="bottom",
    )


def _rho_to_delta(rho: float) -> float:
    odds = float(rho) / max(1.0 - float(rho), 1e-9)
    scale_ratio = LOG_FIDELITY_SCALE[0] ** 2 / LOG_FIDELITY_SCALE[1] ** 2
    return float(0.5 * math.log(odds / scale_ratio))


def _predict_rvrho_cr(target_log: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """RVrho prediction with symmetric physical-space c_R and c_T offsets."""
    target_raw = np.exp(np.asarray(target_log, dtype=float))
    theta = np.asarray(theta, dtype=float)
    c_r, c_t = float(theta[0]), float(theta[1])
    base_raw = target_raw.copy()
    base_raw[:, 0] = np.maximum(base_raw[:, 0] + c_r, 1.0)
    base_raw[:, 1] = np.maximum(base_raw[:, 1] + c_t, 0.05)
    base = np.log(base_raw)

    log_lam_r, r0, log_lam_v, v0, delta = theta[2:]
    fidelity_weights = np.exp(np.array([-delta, delta])) / LOG_FIDELITY_SCALE ** 2
    fidelity = np.diag(fidelity_weights)
    q = fidelity.copy()
    prior = np.zeros(2, dtype=float)
    weight_r = float(np.exp(log_lam_r)) / LOG_FIDELITY_SCALE[0] ** 2
    q[0, 0] += weight_r
    prior[0] += weight_r * float(r0)
    direction = np.array([1.0, -1.0])
    weight_v = float(np.exp(log_lam_v)) / LOG_SPEED_SCALE ** 2
    q += weight_v * np.outer(direction, direction)
    prior += weight_v * float(v0) * direction
    predicted_log = np.linalg.solve(q, (base @ fidelity.T + prior[None, :]).T).T
    return np.exp(predicted_log)


def _fit_rvrho_cr(target_log, produced_log, initial=None, n_starts=2):
    target_log = np.asarray(target_log, dtype=float)
    produced_log = np.asarray(produced_log, dtype=float)
    target_raw = np.exp(target_log)
    produced_raw = np.exp(produced_log)
    base_fit = fit_offset_model("RVrho", target_log, produced_log, n_starts=max(1, n_starts))
    default = np.r_[
        np.clip(np.median(produced_raw[:, 0] - target_raw[:, 0]), -40.0, 40.0),
        base_fit,
    ]
    if initial is not None and np.shape(initial) == np.shape(default):
        default = np.asarray(initial, dtype=float)
    lower = np.array([-60.0, -0.55, -10.0, math.log(5.0), -10.0, math.log(1.0), -4.0])
    upper = np.array([60.0, 1.50, 7.0, math.log(300.0), 7.0, math.log(500.0), 4.0])
    default = np.clip(default, lower + 1e-7, upper - 1e-7)
    rng = np.random.RandomState(20260812 + len(target_log))
    starts = [default]
    for _ in range(max(0, n_starts - 1)):
        jitter = rng.normal(0.0, 0.25, len(default))
        jitter[0] *= 8.0
        jitter[1] *= 0.3
        starts.append(np.clip(default + jitter, lower + 1e-6, upper - 1e-6))

    def residual(theta):
        predicted = _predict_rvrho_cr(target_log, theta)
        return ((predicted - produced_raw) / RAW_SCALE[None, :]).ravel()

    best = None
    for start in starts:
        result = optimize.least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=350,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    return np.asarray(best[1], dtype=float)


def load_or_run_symmetric_offset(subjects, current_cv_records):
    if SYMMETRIC_CACHE.exists():
        return json.loads(SYMMETRIC_CACHE.read_text(encoding="utf-8"))
    if PLOT_ONLY_MODE:
        raise RuntimeError(f"Plot-only mode requires the locked symmetric-offset cache: {SYMMETRIC_CACHE}")
    current_rmse = {
        row["subject"]: float(row["rmse"])
        for row in current_cv_records
        if row["model"] == "RVrho"
    }
    records = []
    for subject, data in subjects.items():
        full = _fit_rvrho_cr(data.target, data.produced, n_starts=3)
        predicted = np.full((len(data.target), 2), np.nan, dtype=float)
        fold_cr, fold_ct = [], []
        for index in range(len(data.target)):
            test = np.arange(len(data.target)) == index
            fitted = _fit_rvrho_cr(
                data.target[~test], data.produced[~test], initial=full, n_starts=1
            )
            predicted[test] = _predict_rvrho_cr(data.target[test], fitted)
            fold_cr.append(float(fitted[0]))
            fold_ct.append(float(fitted[1]))
        observed = np.exp(data.produced)
        rmse = float(np.sqrt(np.mean(((predicted - observed) / RAW_SCALE[None, :]) ** 2)))
        records.append(
            {
                "subject": subject,
                "c_r_full": float(full[0]),
                "c_t_full": float(full[1]),
                "median_training_c_r": float(np.median(fold_cr)),
                "median_training_c_t": float(np.median(fold_ct)),
                "rmse": rmse,
                "current_rvrho_rmse": current_rmse[subject],
                "delta_rmse_symmetric_minus_current": rmse - current_rmse[subject],
            }
        )
    payload = {
        "description": "LOCO RVrho control with symmetric physical-space c_R and c_T nuisance offsets.",
        "records": records,
    }
    SYMMETRIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SYMMETRIC_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_or_run_rho_recovery(subjects):
    if RHO_RECOVERY_CACHE.exists():
        return json.loads(RHO_RECOVERY_CACHE.read_text(encoding="utf-8"))
    if PLOT_ONLY_MODE:
        raise RuntimeError(f"Plot-only mode requires the locked rho-recovery cache: {RHO_RECOVERY_CACHE}")
    rho_grid = np.array([0.20, 0.35, 0.50, 0.65, 0.80], dtype=float)
    rng = np.random.RandomState(20260813)
    records = []
    for subject, data in subjects.items():
        fitted = fit_offset_model("RVrho", data.target, data.produced, n_starts=4)
        reference = predict_offset_model("RVrho", data.target, fitted)
        residual = np.exp(data.produced) - reference
        for true_rho in rho_grid:
            generating = np.asarray(fitted, dtype=float).copy()
            generating[-1] = _rho_to_delta(float(true_rho))
            mean = predict_offset_model("RVrho", data.target, generating)
            sampled = residual[rng.randint(0, len(residual), len(residual))]
            simulated = np.maximum(mean + sampled, np.array([1.0, 0.05]))
            recovered_fit = fit_offset_model(
                "RVrho", data.target, np.log(simulated), initial=generating, n_starts=1
            )
            recovered_rho = decode_offset_model("RVrho", recovered_fit)["rho"]
            records.append(
                {
                    "subject": subject,
                    "true_rho": float(true_rho),
                    "recovered_rho": float(recovered_rho),
                }
            )
    payload = {
        "description": "Empirical-residual parameter recovery for the RVrho allocation parameter.",
        "rho_grid": rho_grid.tolist(),
        "records": records,
    }
    RHO_RECOVERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RHO_RECOVERY_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compute_setpoint_results():
    rows, subjects = load_subject_data()
    medians = condition_medians(subjects)
    duration_stats = raw_duration_statistics(subjects)
    speed_stats, c_t_by_subject = offset_corrected_speed_statistics(subjects)
    cv_records, cv_predictions = load_or_run_cv(subjects)
    strict = load_or_run_strict(subjects)
    base_recovery = load_or_run_recovery(subjects)
    symmetric = load_or_run_symmetric_offset(subjects, cv_records)
    rho_recovery = load_or_run_rho_recovery(subjects)

    duration_gain = np.asarray([row["beta_TT"] for row in duration_stats], dtype=float)
    c_t = np.asarray([row["c_t"] for row in speed_stats], dtype=float)
    speed_gain = np.asarray([row["gain"] for row in speed_stats], dtype=float)
    fixed_speed = np.asarray([row["fixed_speed"] for row in speed_stats], dtype=float)
    duration_vs_one = wilcoxon(duration_gain, 1.0, "two-sided")
    speed_less_one = wilcoxon(speed_gain, 1.0, "less")
    speed_greater_zero = wilcoxon(speed_gain, 0.0, "greater")

    cv = {
        model: np.asarray(
            [row["rmse"] for row in cv_records if row["model"] == model], dtype=float
        )
        for model in MODELS
    }
    rv_comparison_names = ["O", "R", "T", "V", "RT", "TV"]
    rv_tests = []
    for model in rv_comparison_names:
        result = wilcoxon(cv["RV"] - cv[model], 0.0, "less")
        result["comparison"] = f"RV < {model}"
        rv_tests.append(result)
    adjusted = holm_adjust([row["p"] for row in rv_tests])
    for row, value in zip(rv_tests, adjusted):
        row["p_holm"] = float(value)
    rvrho_vs_rv = wilcoxon(cv["RVrho"] - cv["RV"], 0.0, "less")
    rtv_vs_rv = wilcoxon(cv["RTV"] - cv["RV"], 0.0, "less")
    full_vs_rvrho = wilcoxon(cv["FULL"] - cv["RVrho"], 0.0, "less")
    rvrho_minus_full = cv["RVrho"] - cv["FULL"]
    rvrho_full_two_sided = wilcoxon(rvrho_minus_full, 0.0, "two-sided")
    rvrho_full_difference = {
        "definition": "RVrho RMSE minus Full affine RMSE",
        "n": int(len(rvrho_minus_full)),
        "paired_median": float(np.median(rvrho_minus_full)),
        "paired_median_bootstrap_95_CI": bootstrap_median_ci(
            rvrho_minus_full, seed=20260815
        ),
        "paired_wilcoxon_two_sided": rvrho_full_two_sided,
    }

    rho_records = [row for row in cv_records if row["model"] == "RVrho"]
    rho_training = np.asarray([row["median_training_rho"] for row in rho_records])
    rho_vs_half = wilcoxon(rho_training, 0.5, "greater")

    observed_matrices, predicted_rv, predicted_rvrho = [], [], []
    for subject, data in subjects.items():
        observed_matrices.append(normalized_control_matrix(data.target, data.produced))
        predicted_rv.append(
            normalized_control_matrix(
                data.target, np.log(np.maximum(cv_predictions["RV"][subject], 1e-8))
            )
        )
        predicted_rvrho.append(
            normalized_control_matrix(
                data.target,
                np.log(np.maximum(cv_predictions["RVrho"][subject], 1e-8)),
            )
        )
    observed_matrices = np.asarray(observed_matrices)
    predicted_rv = np.asarray(predicted_rv)
    predicted_rvrho = np.asarray(predicted_rvrho)
    observed_asymmetry = observed_matrices[:, 0, 1] - observed_matrices[:, 1, 0]
    predicted_asymmetry_rv = predicted_rv[:, 0, 1] - predicted_rv[:, 1, 0]
    predicted_asymmetry_rvrho = predicted_rvrho[:, 0, 1] - predicted_rvrho[:, 1, 0]
    rho_asymmetry = stats.pearsonr(rho_training, observed_asymmetry)

    raw_coefficients = []
    duration_levels = sorted(
        {float(value) for data in subjects.values() for value in data.stim_t}
    )
    radius_levels = sorted(
        {float(value) for data in subjects.values() for value in data.stim_r}
    )
    offset_by_duration, offset_by_radius = [], []
    for subject, data in subjects.items():
        target = np.exp(data.target)
        produced = np.exp(data.produced)
        design = np.column_stack([np.ones(len(target)), target])
        beta_r, *_ = np.linalg.lstsq(design, produced[:, 0], rcond=None)
        beta_t, *_ = np.linalg.lstsq(design, produced[:, 1], rcond=None)
        raw_coefficients.append(
            {
                "subject": subject,
                "beta_RR": float(beta_r[1]),
                "beta_RT": float(beta_r[2]),
                "beta_TR": float(beta_t[1]),
                "beta_TT": float(beta_t[2]),
            }
        )
        offset_by_duration.append(
            [
                float(
                    np.median(
                        produced[np.isclose(data.stim_t, level), 1]
                        - target[np.isclose(data.stim_t, level), 1]
                    )
                )
                for level in duration_levels
            ]
        )
        offset_by_radius.append(
            [
                float(
                    np.median(
                        produced[np.isclose(data.stim_r, level), 0]
                        - target[np.isclose(data.stim_r, level), 0]
                    )
                )
                for level in radius_levels
            ]
        )
    offset_by_duration = np.asarray(offset_by_duration)
    offset_by_radius = np.asarray(offset_by_radius)
    beta_rr = np.asarray([row["beta_RR"] for row in raw_coefficients])

    corrected = offset_corrected_gains(subjects)
    corrected_map = {row["subject"]: row for row in corrected}
    geometric = subject_speed_statistics(subjects)
    raw_speed_map = {row["subject"]: row for row in geometric}
    kinematic = kinematic_speed_slopes(collect_condition_kinematics(rows))
    kinematic_map = {row["subject"]: row for row in kinematic}
    common_subjects = [subject for subject in subjects if subject in kinematic_map]
    speed_gain_groups = [
        np.asarray([raw_speed_map[subject]["gain"] for subject in common_subjects]),
        np.asarray([corrected_map[subject]["gain"] for subject in common_subjects]),
        np.asarray([kinematic_map[subject]["mean_path_gain"] for subject in common_subjects]),
        np.asarray([kinematic_map[subject]["peak_gain"] for subject in common_subjects]),
    ]
    speed_gain_tests = [wilcoxon(values, 1.0, "less") for values in speed_gain_groups]
    speed_gain_adjusted = holm_adjust([row["p"] for row in speed_gain_tests])
    for row, value in zip(speed_gain_tests, speed_gain_adjusted):
        row["p_holm"] = float(value)

    rtv_parameters, rv_parameters, normalized_setpoints = [], [], []
    for subject, data in subjects.items():
        rtv = decode_offset_model(
            "RTV", fit_offset_model("RTV", data.target, data.produced, n_starts=6)
        )
        rtv["subject"] = subject
        rtv_parameters.append(rtv)
        rv = decode_offset_model(
            "RV", fit_offset_model("RV", data.target, data.produced, n_starts=6)
        )
        rv["subject"] = subject
        rv_parameters.append(rv)
        target = np.exp(data.target)
        speed = target[:, 0] / np.maximum(target[:, 1] + rv["c_t"], 0.05)
        normalized_setpoints.append(
            {
                "subject": subject,
                "R0_normalized": float(
                    (np.log(rv["r0"]) - np.log(target[:, 0].min()))
                    / (np.log(target[:, 0].max()) - np.log(target[:, 0].min()))
                ),
                "V0_normalized": float(
                    (np.log(rv["v0"]) - np.log(speed.min()))
                    / (np.log(speed.max()) - np.log(speed.min()))
                ),
            }
        )
    strength_r = np.asarray(
        [row["lambda_r"] / (1.0 + row["lambda_r"]) for row in rtv_parameters]
    )
    strength_t = np.asarray(
        [row["lambda_t"] / (1.0 + row["lambda_t"]) for row in rtv_parameters]
    )
    strength_v = np.asarray(
        [row["lambda_v"] / (1.0 + row["lambda_v"]) for row in rtv_parameters]
    )

    return {
        "rows": rows,
        "subjects": subjects,
        "medians": medians,
        "duration_stats": duration_stats,
        "speed_stats": speed_stats,
        "c_t_by_subject": c_t_by_subject,
        "duration_gain": duration_gain,
        "c_t": c_t,
        "speed_gain": speed_gain,
        "fixed_speed": fixed_speed,
        "duration_vs_one": duration_vs_one,
        "speed_less_one": speed_less_one,
        "speed_greater_zero": speed_greater_zero,
        "cv_records": cv_records,
        "cv": cv,
        "rv_tests": rv_tests,
        "rvrho_vs_rv": rvrho_vs_rv,
        "rtv_vs_rv": rtv_vs_rv,
        "full_vs_rvrho": full_vs_rvrho,
        "rvrho_full_difference": rvrho_full_difference,
        "rho_training": rho_training,
        "rho_vs_half": rho_vs_half,
        "rho_asymmetry": rho_asymmetry,
        "observed_matrices": observed_matrices,
        "predicted_rv": predicted_rv,
        "predicted_rvrho": predicted_rvrho,
        "observed_asymmetry": observed_asymmetry,
        "predicted_asymmetry_rv": predicted_asymmetry_rv,
        "predicted_asymmetry_rvrho": predicted_asymmetry_rvrho,
        "raw_coefficients": raw_coefficients,
        "beta_rr": beta_rr,
        "duration_levels": duration_levels,
        "radius_levels": radius_levels,
        "offset_by_duration": offset_by_duration,
        "offset_by_radius": offset_by_radius,
        "speed_gain_groups": speed_gain_groups,
        "speed_gain_tests": speed_gain_tests,
        "rtv_parameters": rtv_parameters,
        "rv_parameters": rv_parameters,
        "normalized_setpoints": normalized_setpoints,
        "strength_r": strength_r,
        "strength_t": strength_t,
        "strength_v": strength_v,
        "strict": strict,
        "base_recovery": base_recovery,
        "symmetric": symmetric,
        "rho_recovery": rho_recovery,
    }


def _serial_from_reference(reference):
    """Reconstruct plotting arrays from the previously validated serial analysis."""
    source = reference["M1b_fast_adaptive_prior"]
    kernel = source["lag_kernel"]
    labels = ["H_RR", "H_RT", "H_TR", "H_TT"]
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    n_subjects = int(source["n_subjects"])
    n_lags = len(kernel["lags"])
    lag_h = np.full((n_subjects, n_lags, 2, 2), np.nan, dtype=float)
    for label, (i, j) in zip(labels, indices):
        lag_h[:, :, i, j] = np.asarray(kernel["subject_values"][label], dtype=float)
    null_ci = np.asarray(kernel["circular_shift"]["null_median_95_CI"], dtype=float)

    transition = source["transition_specific_lag1"]
    transition_h = {"normal": np.full((n_subjects, 2, 2), np.nan),
                    "probe": np.full((n_subjects, 2, 2), np.nan)}
    for label, (i, j) in zip(labels, indices):
        transition_h["normal"][:, i, j] = np.asarray(transition["normal_to_normal_subject_values"][label])
        transition_h["probe"][:, i, j] = np.asarray(transition["probe_to_normal_subject_values"][label])

    history = source["history_adjusted_behavior_matrix"]
    beta_labels = ["beta_RR", "beta_RT", "beta_TR", "beta_TT"]
    adjusted_delta = {}
    adjusted_original = {}
    for model_name in ("lag1_all", "exp_all", "exp_drawing"):
        matrix = np.full((n_subjects, 2, 2), np.nan, dtype=float)
        for label, (i, j) in zip(beta_labels, indices):
            matrix[:, i, j] = np.asarray(history["delta_subject_values"][model_name][label], dtype=float)
        adjusted_delta[model_name] = matrix
        observed = np.asarray(history["original_subject_mean"][model_name], dtype=float)
        adjusted_original[model_name] = np.repeat(observed[None, :, :], n_subjects, axis=0)
    return {
        "lag_H": lag_h,
        "lag_null_ci": null_ci,
        "transition_H": transition_h,
        "adjusted_delta": adjusted_delta,
        "adjusted_original": adjusted_original,
    }


def load_alternative_results():
    """Load validated heavy analyses and recompute only lightweight plot summaries."""
    if ALTERNATIVE_PLOT_CACHE.exists():
        output = np.load(ALTERNATIVE_PLOT_CACHE, allow_pickle=True).item()
        if "timeR_subject_slopes" in output:
            output["serial_pooled_blocked"] = (
                alternative_analysis.load_pooled_blocked_serial_validation()
            )
            if not PLOT_ONLY_MODE:
                np.save(ALTERNATIVE_PLOT_CACHE, output, allow_pickle=True)
            return output
        if PLOT_ONLY_MODE:
            raise RuntimeError(
                f"Plot-only mode found an incompatible alternative-model plot cache: {ALTERNATIVE_PLOT_CACHE}"
            )
    elif PLOT_ONLY_MODE:
        raise RuntimeError(
            f"Plot-only mode requires the locked alternative-model plot cache: {ALTERNATIVE_PLOT_CACHE}"
        )
    if not ALTERNATIVE_REFERENCE_CACHE.exists():
        if PLOT_ONLY_MODE:
            raise RuntimeError(
                "Plot-only mode requires a locked alternative-model reference cache."
            )
        return compute_alternatives()
    reference = json.loads(ALTERNATIVE_REFERENCE_CACHE.read_text(encoding="utf-8"))

    records = alternative_analysis.collect_trials()
    matched = alternative_analysis.build_matched_rows(alternative_analysis.load_human_condition_means())
    coefficients, _ = alternative_analysis.fit_subject_models(matched, alternative_analysis.NORM)
    subjects = [row["subject"] for row in coefficients]
    subject_id = np.asarray([row["subj"] for row in records])
    stim_r = np.asarray([row["stim_r"] for row in records], dtype=float)
    stim_t = np.asarray([row["stim_t"] for row in records], dtype=float)
    produced_r = np.asarray([row["produced_r"] for row in records], dtype=float)
    produced_t = np.asarray([row["produced_t"] for row in records], dtype=float)
    mean_speed = np.asarray([row["mean_speed"] for row in records], dtype=float)
    radius_levels = sorted(set(np.round(stim_r, 2)))
    duration_levels = sorted(set(np.round(stim_t, 2)))

    # Subject-level proportional-noise points used in AM2.
    compression = {
        row["subject"]: (1.0 - row["beta_RR"], 1.0 - row["beta_TT"])
        for row in coefficients
    }
    produced_by_condition = {}
    for index in range(len(records)):
        key = (subject_id[index], round(stim_r[index], 2), round(stim_t[index], 2))
        produced_by_condition.setdefault(key, []).append((produced_r[index], produced_t[index]))
    noise_by_subject = {}
    for (subject, _, _), values in produced_by_condition.items():
        array = np.asarray(values, dtype=float)
        if len(array) >= 4:
            noise_by_subject.setdefault(subject, []).append(
                (array[:, 0].std() / array[:, 0].mean(), array[:, 1].std() / array[:, 1].mean())
            )
    cvr, cvt, cr, ct = [], [], [], []
    for subject, values in noise_by_subject.items():
        if subject in compression:
            values = np.asarray(values, dtype=float)
            cvr.append(float(values[:, 0].mean())); cvt.append(float(values[:, 1].mean()))
            cr.append(float(compression[subject][0])); ct.append(float(compression[subject][1]))
    cvr, cvt, cr, ct = map(np.asarray, (cvr, cvt, cr, ct))

    # Lightweight AM4 condition-family summaries and subject-level slopes.
    fam = alternative_analysis._family_ci(
        subject_id, stim_r, stim_t, radius_levels, duration_levels, subjects, produced_r, mean_speed
    )
    fam_t = alternative_analysis._family_ci(
        subject_id, stim_r, stim_t, radius_levels, duration_levels, subjects, produced_r, produced_t
    )
    timeR_subject_slopes = []
    for subject in subjects:
        within_duration_slopes = []
        for target_t in duration_levels:
            radius_medians, duration_medians = [], []
            for target_r in radius_levels:
                mask = (
                    (subject_id == subject)
                    & (np.abs(stim_t - target_t) < 1e-6)
                    & (np.abs(stim_r - target_r) < 1e-6)
                )
                valid = (
                    mask
                    & np.isfinite(produced_r)
                    & np.isfinite(produced_t)
                    & (produced_r > 0)
                    & (produced_t > 0)
                )
                if int(np.sum(valid)) >= alternative_analysis.MIN_LEVEL:
                    radius_medians.append(float(np.median(produced_r[valid])))
                    duration_medians.append(float(np.median(produced_t[valid])))
            if len(radius_medians) >= 3 and np.ptp(np.log10(radius_medians)) > 1e-9:
                within_duration_slopes.append(
                    float(np.polyfit(np.log10(radius_medians), np.log10(duration_medians), 1)[0])
                )
        if within_duration_slopes:
            timeR_subject_slopes.append(float(np.median(within_duration_slopes)))
    timeR_subject_slopes = np.asarray(timeR_subject_slopes, dtype=float)
    timeR_test = wilcoxon_test(timeR_subject_slopes, reference=2.0 / 3.0)

    model_scope = reference["matrix_level_alternatives"]["M2_specified_quadratic_effort_best"]
    loso = model_scope["leave_one_subject_out"]
    manifold = reference["M2_specified_quadratic_effort_timing_constrained_manifold"]
    closest = manifold["closest_timing_compatible_point"]
    noise_reference = reference["M1_simple_proportional_noise"]
    law_reference = reference["M3_constant_K_spontaneous_two_thirds"]
    observed = reference["measured_matrix"]
    serial = _serial_from_reference(reference)
    pooled_validation = alternative_analysis.load_pooled_blocked_serial_validation()
    output = {
        "n_subj": len(subjects),
        "n_trials": len(records),
        "Ts": duration_levels,
        "serial": serial,
        "serial_pooled_blocked": pooled_validation,
        "B_obs": np.asarray([[observed["beta_RR"], observed["beta_RT"]],
                             [observed["beta_TR"], observed["beta_TT"]]], dtype=float),
        "mRR": float(observed["beta_RR"]), "mRT": float(observed["beta_RT"]),
        "mTR": float(observed["beta_TR"]), "mTT": float(observed["beta_TT"]),
        "cv_model_errors": np.asarray(loso["model_error_values"], dtype=float),
        "cv_baseline_errors": np.asarray(loso["train_mean_baseline_error_values"], dtype=float),
        "cv_error_p_adj": float(loso["paired_model_vs_baseline_p_adjusted"]),
        "effort_feasible": np.empty((0, 6), dtype=float),
        "spatial_bootstrap_cov": np.asarray(manifold["spatial_bootstrap_covariance"], dtype=float),
        "closest_feasible": np.asarray([closest["lambda_"], closest["w"], closest["beta_RR"],
                                        closest["beta_RT"], closest["beta_TR"], closest["beta_TT"]], dtype=float),
        "closest_feasible_mahalanobis2": float(closest["mahalanobis2"]),
        "feasible_inside_observed_95": int(manifold["timing_compatible_points_inside_spatial_95_ellipse"]),
        "timing_compatible_grid_points": int(manifold["timing_compatible_grid_points"]),
        "CVR": cvr, "CVT": cvt, "CR": cr, "CT": ct,
        "fitR": np.polyfit(cvr, cr, 1), "fitT": np.polyfit(cvt, ct, 1),
        "noise_scale": float(reference["matrix_level_alternatives"]["M1_simple_proportional_noise"]["noise_to_compression_scale"]),
        "p_noise": float(noise_reference["time_vs_space_noise_p"]),
        "pearsonR": float(noise_reference["primary_linear_association"]["spatial_r"]),
        "pearson_pR": float(noise_reference["primary_linear_association"]["spatial_p"]),
        "pearsonT": float(noise_reference["primary_linear_association"]["temporal_r"]),
        "pearson_pT": float(noise_reference["primary_linear_association"]["temporal_p"]),
        "fam": fam, "famT": fam_t,
        "timeR_subject_slopes": timeR_subject_slopes,
        "timeR_slope": float(np.median(timeR_subject_slopes)),
        "timeR_test": timeR_test,
        "timeR_p23": float(timeR_test["p_raw"]),
        "slope_med": float(law_reference["speed_radius_slope"]),
        "p_vs_third": float(law_reference["speed_vs_third_p"]),
    }
    if not PLOT_ONLY_MODE:
        ALTERNATIVE_PLOT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(ALTERNATIVE_PLOT_CACHE, output, allow_pickle=True)
    return output


def _draw_model_schematic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    register_formula_text(ax.text(
        0.50,
        0.91,
        r"Perceived target $(R_{\mathrm{P}},T_{\mathrm{P}})$",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.3"},
    ), "fig2_formula_a_target")
    ax.annotate(
        "",
        xy=(0.50, 0.79),
        xytext=(0.50, 0.85),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "0.25"},
    )
    register_formula_text(ax.text(
        0.50,
        0.70,
        "Offset-corrected baseline\n"
        + r"$R_b=R_{\mathrm{P}},\ T_b=T_{\mathrm{P}}+c_T,\ V_b=R_b/T_b$",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "0.94", "edgecolor": "0.45"},
    ), "fig2_formula_a_baseline")
    candidates = [
        (0.17, "Radius", r"$R_0$", C_R, "fig2_formula_a_r0"),
        (0.50, "Duration", r"$T_0$", C_T, "fig2_formula_a_t0"),
        (0.83, "Speed", r"$V_0$", C_V, "fig2_formula_a_v0"),
    ]
    for x, label, setpoint, color, gid in candidates:
        ax.annotate(
            "",
            xy=(x, 0.585),
            xytext=(0.50, 0.62),
            arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "0.35"},
        )
        well_x = np.linspace(x - 0.10, x + 0.10, 100)
        well_y = 0.415 + 0.125 * ((well_x - x) / 0.10) ** 2
        ax.plot(well_x, well_y, color=color, lw=1.7, solid_capstyle="round", zorder=2)
        ax.scatter(
            [x],
            [0.415],
            s=22,
            facecolor=color,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
        ax.text(x, 0.565, label, ha="center", va="center")
        register_formula_text(
            ax.text(x, 0.465, setpoint, ha="center", va="center"),
            gid,
        )
        ax.annotate(
            "",
            xy=(0.50, 0.32),
            xytext=(x, 0.397),
            arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "0.35"},
        )
    ax.text(
        0.50,
        0.25,
        "Held-out comparison\n" + r"O   R   T   V   RT   RV   TV   RTV",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "0.3"},
    )
    ax.annotate(
        "",
        xy=(0.50, 0.145),
        xytext=(0.50, 0.175),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "0.35"},
    )
    register_formula_text(
        ax.text(
            0.50,
            0.105,
            r"$RV_\rho$: nominal speed-only allocation",
            ha="center",
            va="center",
        ),
        "fig2_formula_a_rvrho",
    )
    register_formula_text(
        ax.text(
            0.50,
            0.035,
            r"$\Delta r=\rho\Delta v,\ \Delta t=-(1-\rho)\Delta v$",
            ha="center",
            va="center",
        ),
        "fig2_formula_a_allocation",
    )
    ax.set_title("Candidate setpoint\nframework")


def make_main_figure(S, D):
    setup_style()
    _FIG2_FORMULA_SPECS.clear()
    # This figure was originally assembled further in Illustrator.  Give the
    # direct executable version enough physical height for titles, rotated
    # tick labels, and x-axis labels at the final fixed 178-mm width.
    fig = plt.figure(figsize=(22.0, 11.5))
    outer_grid = fig.add_gridspec(
        2,
        1,
        left=0.045,
        right=0.985,
        bottom=0.145,
        top=0.900,
        hspace=0.68,
    )
    top_grid = outer_grid[0, 0].subgridspec(
        1, 5, width_ratios=[1.25, 1.0, 1.15, 1.0, 1.0], wspace=0.45
    )
    bottom_grid = outer_grid[1, 0].subgridspec(
        1, 4, width_ratios=[4.0, 4.0, 6.0, 4.0], wspace=0.30
    )
    f_grid = bottom_grid[0, 0].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.40)
    g_grid = bottom_grid[0, 1].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.40)
    h_grid = bottom_grid[0, 2].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.28)
    i_grid = bottom_grid[0, 3].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.40)
    axes = {
        "A": fig.add_subplot(top_grid[0, 0]),
        "B": fig.add_subplot(top_grid[0, 1]),
        "C": fig.add_subplot(top_grid[0, 2]),
        "D": fig.add_subplot(top_grid[0, 3]),
        "E": fig.add_subplot(top_grid[0, 4]),
        "F": fig.add_subplot(f_grid[0, 0]),
        "F_box": fig.add_subplot(f_grid[0, 1]),
        "G": fig.add_subplot(g_grid[0, 0]),
        "G_box": fig.add_subplot(g_grid[0, 1]),
        "H": fig.add_subplot(h_grid[0, 0]),
        "H_box": fig.add_subplot(h_grid[0, 1]),
        "I": fig.add_subplot(i_grid[0, 0]),
        "I_box": fig.add_subplot(i_grid[0, 1]),
    }
    rng = np.random.RandomState(20260811)
    text_box = {"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 1.0}

    _draw_model_schematic(axes["A"])

    ax = axes["B"]
    speed_x, speed_y = offset_speed_condition_medians(S["subjects"], S["c_t_by_subject"])
    ax.scatter(speed_x, speed_y, s=22, facecolor="0.92", edgecolor="0.20", linewidth=0.65)
    limits = [min(speed_x.min(), speed_y.min()) * 0.82, max(speed_x.max(), speed_y.max()) * 1.18]
    xx = np.geomspace(limits[0], limits[1], 160)
    gain = float(np.median(S["speed_gain"]))
    fixed = float(np.nanmedian(S["fixed_speed"]))
    ax.plot(xx, xx, color="0.60", lw=0.9, ls="--", label="Identity")
    ax.plot(xx, fixed * (xx / fixed) ** gain, color=C_V, lw=1.8, label="Median subject gain")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ticks = [tick for tick in [20, 40, 80, 160] if limits[0] <= tick <= limits[1]]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_yticklabels([str(tick) for tick in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    register_formula_text(
        ax.set_xlabel(
            r"Baseline dynamic speed, $R_{\mathrm{P}}/(T_{\mathrm{P}}+c_T)$"
        ),
        "fig2_formula_b_xlabel",
    )
    register_formula_text(
    ax.set_ylabel(r"Produced speed, $R/T$"),
        "fig2_formula_b_ylabel",
    )
    ax.set_title("Dynamic speed\nremains compressed")
    gain_ax = ax.inset_axes([0.60, 0.06, 0.34, 0.45])
    gain_values = np.asarray(S["speed_gain"], dtype=float)
    gain_box = gain_ax.boxplot(
        [gain_values],
        vert=True,
        positions=[0.0],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": C_V, "linewidth": 0.8},
    )
    gain_box["boxes"][0].set_facecolor(C_V)
    gain_box["boxes"][0].set_alpha(0.55)
    gain_ax.scatter(
        rng.uniform(-0.10, 0.10, len(gain_values)),
        gain_values,
        s=8,
        color=C_V,
        alpha=0.70,
        edgecolor="none",
        zorder=3,
    )
    gain_ax.axhline(1.0, color="0.55", lw=0.8, ls="--")
    gain_ax.set_xlim(-0.42, 0.42)
    gain_ax.set_ylim(0.15, 1.55)
    gain_ax.set_xticks([])
    gain_ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    gain_ax.set_ylabel("Log–log gain", labelpad=1.0)
    gain_ax.text(
        0.04,
        0.96,
        f"Median={gain:.2f}\n{format_p(S['speed_less_one']['p'])}",
        transform=gain_ax.transAxes,
        ha="left",
        va="top",
    )
    gain_ax.patch.set_facecolor("white")
    gain_ax.patch.set_alpha(0.96)
    tidy(gain_ax)
    tidy(ax)

    ax = axes["C"]
    # Small gaps separate the null, single-coordinate, pairwise, and benchmark
    # families; RVrho remains adjacent to RV for the nested comparison.
    positions = np.asarray([0.0, 1.2, 2.0, 2.8, 4.0, 4.8, 5.6, 6.8, 8.0, 9.2])
    for position, model in zip(positions, DISPLAY_MODELS):
        draw_box(ax, S["cv"][model], position, OFFSET_MODEL_COLORS[model], width=0.58, rng=rng)
    ax.set_xticks(positions)
    model_ticklabels = ax.set_xticklabels(
        DISPLAY_LABELS, rotation=60, ha="right", rotation_mode="anchor"
    )
    register_formula_text(model_ticklabels[6], "fig2_formula_c_rvrho_tick")
    ax.set_ylabel("Held-out raw joint RMSE")
    ax.set_title("Held-out comparison\nfavors R + V")
    rv_tests = {row["comparison"]: row for row in S["rv_tests"]}
    model_position = dict(zip(DISPLAY_MODELS, positions))
    draw_comparison_bracket(
        ax,
        model_position["RT"],
        model_position["RV"],
        0.30,
        format_p(rv_tests["RV < RT"]["p_holm"]),
    )
    draw_comparison_bracket(
        ax,
        model_position["RV"],
        model_position["RVrho"],
        0.36,
        format_p(S["rvrho_vs_rv"]["p"]),
    )
    draw_comparison_bracket(
        ax,
        model_position["RV"],
        model_position["RTV"],
        0.42,
        format_p(S["rtv_vs_rv"]["p"]),
    )
    draw_comparison_bracket(
        ax,
        model_position["R"],
        model_position["RV"],
        0.48,
        format_p(rv_tests["RV < R"]["p_holm"]),
    )
    full_difference = S["rvrho_full_difference"]
    full_ci = full_difference["paired_median_bootstrap_95_CI"]
    full_bracket_text = draw_comparison_bracket(
        ax,
        model_position["RVrho"],
        model_position["FULL"],
        0.58,
        rf"$\Delta$RMSE={full_difference['paired_median']:.4f}, "
        + format_p(full_difference["paired_wilcoxon_two_sided"]["p"]) + "\n"
        rf"95% CI [{full_ci[0]:.4f}, {full_ci[1]:.4f}]",
    )
    register_formula_text(full_bracket_text, "fig2_formula_c_delta_rmse")
    tidy(ax)

    ax = axes["D"]
    ax.scatter(S["rho_training"], S["observed_asymmetry"], s=26, facecolor=C_R, alpha=0.72, edgecolor="0.2", linewidth=0.6)
    slope, intercept = np.polyfit(S["rho_training"], S["observed_asymmetry"], 1)
    xx = np.linspace(0.0, 1.0, 120)
    ax.plot(xx, intercept + slope * xx, color=C_R, lw=1.7)
    ax.axhline(0, color="0.65", lw=0.8, ls="--")
    ax.axvline(0.5, color="0.65", lw=0.8, ls=":")
    ax.set_xlim(-0.04, 1.04)
    register_formula_text(
        ax.set_xlabel(r"Spatial allocation, $\rho$"), "fig2_formula_d_xlabel"
    )
    register_formula_text(
        ax.set_ylabel(r"Observed $\beta_{RT}-\beta_{TR}$"),
        "fig2_formula_d_ylabel",
    )
    register_formula_text(
        ax.set_title(r"$\rho$ indexes directional" + "\nallocation"),
        "fig2_formula_d_title",
    )
    directional_text = ax.text(
        0.04,
        0.96,
        rf"Median $\rho$={np.median(S['rho_training']):.2f}" + "\n"
        rf"$\rho>0.5$: {format_p(S['rho_vs_half']['p'])}" + "\n"
        f"Pearson $r$={S['rho_asymmetry'][0]:.2f}, {format_pearson_p(S['rho_asymmetry'][1])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=text_box,
    )
    register_formula_text(directional_text, "fig2_formula_d_stats")
    tidy(ax)

    ax = axes["E"]
    observed = S["observed_matrices"][:, [0, 0, 1, 1], [0, 1, 0, 1]]
    rv = S["predicted_rv"][:, [0, 0, 1, 1], [0, 1, 0, 1]]
    rvrho = S["predicted_rvrho"][:, [0, 0, 1, 1], [0, 1, 0, 1]]
    centers = np.arange(4)
    for values, offset, marker, color, filled, label in [
        (observed, -0.16, "o", None, True, "Observed"),
        (rv, 0.0, "o", "0.50", False, "RV"),
        (rvrho, 0.16, "D", "0.10", False, r"$RV_\rho$"),
    ]:
        med = np.median(values, axis=0)
        ci = np.asarray([bootstrap_ci(values[:, i], 20260830 + i + int((offset + 0.2) * 100)) for i in range(4)])
        for i in range(4):
            point_color = [C_R, C_R, C_T, C_T][i] if color is None else color
            ax.errorbar(
                centers[i] + offset,
                med[i],
                yerr=[[med[i] - ci[i, 0]], [ci[i, 1] - med[i]]],
                fmt=marker,
                ms=4.8,
                mfc=point_color if filled else "white",
                mec=point_color,
                ecolor=point_color,
                capsize=2.2,
                lw=0.9,
            )
        ax.scatter([], [], marker=marker, facecolor="0.35" if filled else "white", edgecolor="0.35" if color is None else color, label=label)
    ax.axhline(0, color="0.70", lw=0.8)
    ax.axhline(1, color="0.70", lw=0.8, ls="--")
    ax.set_xticks(centers)
    coefficient_ticklabels = ax.set_xticklabels(
        [r"$\beta_{RR}$", r"$\beta_{RT}$", r"$\beta_{TR}$", r"$\beta_{TT}$"]
    )
    for index, ticklabel in enumerate(coefficient_ticklabels):
        register_formula_text(ticklabel, f"fig2_formula_e_beta_tick_{index}")
    ax.set_ylabel("Control coefficient")
    ax.set_ylim(-0.22, 1.23)
    register_formula_text(
        ax.set_title(r"$RV_\rho$ matrix" + "\napproximation"),
        "fig2_formula_e_title",
    )
    matrix_legend = ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=8,
        handlelength=0.7,
        handletextpad=0.1,
        columnspacing=0.25,
        borderaxespad=0.1,
    )
    register_formula_text(matrix_legend.get_texts()[2], "fig2_formula_e_legend_rvrho")
    matrix_text = ax.text(
        0.98,
        0.65,
        r"$\beta_{RT}-\beta_{TR}$" + "\n"
        f"Obs. {np.median(S['observed_asymmetry']):.3f}\n"
        f"RV {np.median(S['predicted_asymmetry_rv']):.3f}\n"
        rf"$RV_\rho$ {np.median(S['predicted_asymmetry_rvrho']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=text_box,
    )
    register_formula_text(matrix_text, "fig2_formula_e_stats")
    tidy(ax)

    # AM1: coefficient geometry plus a narrow held-out-error summary.
    _draw_effort_manifold(axes["F"], D, text_box)
    ax = axes["F_box"]
    for model_error, baseline_error in zip(D["cv_model_errors"], D["cv_baseline_errors"]):
        ax.plot([0, 1], [model_error, baseline_error], color="0.78", lw=0.65, alpha=0.75)
    draw_box(ax, D["cv_model_errors"], 0, C_MODEL, width=0.38, rng=rng)
    draw_box(ax, D["cv_baseline_errors"], 1, C_MODEL, width=0.38, filled=False, rng=rng)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Effort", "Baseline"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("LOSO matrix error")
    ax.set_title("Held-out\nerror")
    ax.text(0.98, 0.97, significance_symbol(D["cv_error_p_adj"]), transform=ax.transAxes,
            ha="right", va="top", bbox=text_box)
    tidy(ax)

    # AM2: static proportional-noise model.
    am2_stats = am2_inference(D)
    ax = axes["G"]
    xline = np.array([0.10, 0.36])
    ax.plot(
        xline,
        D["noise_scale"] * xline,
        color=C_MODEL,
        lw=1.4,
        ls="--",
        label="AM2 prediction",
    )
    ax.scatter(D["CVR"], D["CR"], s=20, color=C_R, alpha=0.65, label="Radius")
    ax.scatter(D["CVT"], D["CT"], s=20, color=C_T, alpha=0.65, label="Duration")
    ax.plot(np.linspace(D["CVR"].min(), D["CVR"].max()), np.polyval(D["fitR"], np.linspace(D["CVR"].min(), D["CVR"].max())), color=C_R, lw=1.6)
    ax.plot(np.linspace(D["CVT"].min(), D["CVT"].max()), np.polyval(D["fitT"], np.linspace(D["CVT"].min(), D["CVT"].max())), color=C_T, lw=1.6)
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xlabel("Produced noise (CV)")
    register_formula_text(
        ax.set_ylabel(r"Compression $1-\beta$"), "fig2_formula_g_ylabel"
    )
    ax.set_title("AM2\nProportional noise")
    ax.legend(frameon=False, loc="lower left")
    radius_corr_text = ax.text(
        0.98,
        0.97,
        rf"Radius: $r$={am2_stats['space']['r']:+.2f}, "
        + format_pearson_p(am2_stats["space"]["p_bh"]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=text_box,
        color=C_R,
    )
    register_formula_text(radius_corr_text, "fig2_formula_g_radius_corr")
    duration_corr_text = ax.text(
        0.98,
        0.87,
        rf"Duration: $r$={am2_stats['time']['r']:+.2f}, "
        + format_pearson_p(am2_stats["time"]["p_bh"]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=text_box,
        color=C_T,
    )
    register_formula_text(duration_corr_text, "fig2_formula_g_duration_corr")
    tidy(ax)

    ax = axes["G_box"]
    jitter = np.linspace(-0.055, 0.055, len(D["CVR"]))
    for index in range(len(D["CVR"])):
        ax.plot(
            [jitter[index], 1.0 + jitter[index]],
            [D["CVR"][index], D["CVT"][index]],
            color="0.78",
            lw=0.65,
            alpha=0.75,
            zorder=1,
        )
    draw_box(ax, D["CVR"], 0, C_R, width=0.50, points=False)
    draw_box(ax, D["CVT"], 1, C_T, width=0.50, points=False)
    ax.scatter(jitter, D["CVR"], s=10, color=C_R, alpha=0.75, edgecolor="none", zorder=3)
    ax.scatter(1.0 + jitter, D["CVT"], s=10, color=C_T, alpha=0.75, edgecolor="none", zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Radius", "Duration"], rotation=45, ha="right")
    ax.set_ylabel("CV")
    ax.set_title("Produced\nnoise", pad=4)
    axis_transform = ax.get_xaxis_transform()
    ax.plot(
        [0, 0, 1, 1],
        [1.01, 1.04, 1.04, 1.01],
        color="black",
        lw=0.8,
        transform=axis_transform,
        clip_on=False,
    )
    ax.text(
        0.5,
        1.055,
        significance_symbol(am2_stats["time_vs_space_noise"]["p_raw"]),
        ha="center",
        va="bottom",
        transform=axis_transform,
        clip_on=False,
    )
    noise_values = np.concatenate([D["CVR"], D["CVT"]])
    noise_span = max(float(np.ptp(noise_values)), 1e-6)
    ax.set_ylim(float(np.min(noise_values)) - 0.08 * noise_span, float(np.max(noise_values)) + 0.08 * noise_span)
    tidy(ax)

    # AM3: empirical serial kernel plus held-out predictive validation.
    _draw_serial_kernel(axes["H"], D, text_box)
    _draw_blocked_history_prediction(axes["H_box"], D, rng)

    # AM4: constant-K duration prediction.
    ax = axes["I"]
    for index, target_t in enumerate(D["Ts"]):
        x, y, ylo, yhi = D["famT"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * index / max(len(D["Ts"]) - 1, 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y)
        yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        ax.errorbar(lx, ly, yerr=yerr, fmt="-o", color=color, ms=3.2, lw=1.1, capsize=2)
        anchor = int(np.argmin(lx))
        span = np.array([lx.min(), lx.max()])
        ax.plot(span, ly[anchor] + (2.0 / 3.0) * (span - lx[anchor]), ":", color=color, lw=0.9)
    register_formula_text(
        ax.set_xlabel(r"$\log_{10}$ radius"), "fig2_formula_i_xlabel"
    )
    register_formula_text(
        ax.set_ylabel(r"$\log_{10}$ duration"), "fig2_formula_i_ylabel"
    )
    ax.set_title("AM4\nTwo-thirds law")
    tidy(ax)

    slope_ax = axes["I_box"]
    draw_box(
        slope_ax,
        D["timeR_subject_slopes"],
        0,
        C_T,
        width=0.48,
        rng=np.random.RandomState(20260819),
    )
    slope_ax.axhline(2.0 / 3.0, color="0.45", lw=0.8, ls="--")
    slope_ax.set_xlim(-0.55, 0.55)
    slope_values = np.asarray(D["timeR_subject_slopes"], dtype=float)
    slope_span = max(float(np.ptp(slope_values)), 0.20)
    slope_ax.set_ylim(
        min(float(np.min(slope_values)) - 0.12 * slope_span, -0.08),
        max(float(np.max(slope_values)) + 0.12 * slope_span, 0.90),
    )
    slope_ax.set_xticks([0])
    slope_ax.set_xticklabels(["Slope"])
    slope_ax.set_ylabel("")
    slope_ax.set_yticks([0.0, 2.0 / 3.0])
    slope_ticklabels = slope_ax.set_yticklabels(["0", r"$2/3$"])
    register_formula_text(slope_ticklabels[1], "fig2_formula_i_two_thirds_tick")
    slope_ax.set_title("Subject\nslopes")
    slope_ax.text(
        0.5,
        0.72,
        f"Median\n{D['timeR_slope']:.2f}",
        transform=slope_ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=text_box,
    )
    slope_ax.text(
        0.5,
        0.96,
        significance_symbol(D["timeR_test"]["p_raw"]),
        transform=slope_ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=text_box,
    )
    tidy(slope_ax)

    make_axes_square(
        fig, [axes[key] for key in ["A", "B", "C", "D", "E"]]
    )

    for letter in "ABCDEFH":
        label_panel(axes[letter], letter)
    for letter in "GI":
        axes[letter].text(-0.20, 1.16, letter.lower(), transform=axes[letter].transAxes,
                          fontsize=12, fontweight="bold", ha="left", va="top")
    output = FIG_DIR / "Fig2_behavior_computational_models.png"
    prepare_figure_for_export(fig)
    fig.savefig(output, dpi=300, facecolor="white")
    vector_output = save_vector_figure(
        fig, output, facecolor="white"
    )
    rewrite_illustrator_formula_text(vector_output)
    plt.close(fig)
    return output


def _make_setpoint_supplement_archive(S):
    """Legacy nine-panel setpoint audit; not generated by ``main``."""
    setup_style()
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.3))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.955, wspace=0.42, hspace=0.52)
    axes = axes.ravel()
    rng = np.random.RandomState(20260814)
    text_box = {"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 1.0}

    # A. Raw spatial central tendency, with the raw spatial offset as an inset.
    ax = axes[0]
    target_r = np.exp(S["medians"]["target"][:, 0])
    produced_r = np.exp(S["medians"]["produced"][:, 0])
    ax.scatter(target_r, produced_r, s=22, facecolor="0.92", edgecolor="0.20", linewidth=0.6)
    limits = [min(target_r.min(), produced_r.min()) * 0.90, max(target_r.max(), produced_r.max()) * 1.08]
    xx = np.linspace(limits[0], limits[1], 120)
    slope = float(np.median(S["beta_rr"]))
    intercept = float(np.median(produced_r) - slope * np.median(target_r))
    ax.plot(xx, xx, color="0.60", lw=0.8, ls="--")
    ax.plot(xx, intercept + slope * xx, color=C_R, lw=1.8)
    ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel("Perceived radius"); ax.set_ylabel("Produced radius")
    ax.set_title("Raw spatial central tendency")
    rr_test = wilcoxon(S["beta_rr"], 1.0, "less")
    ax.text(0.04, 0.96, rf"Median $\beta_{{RR}}$={slope:.2f}" + "\n" + f"vs 1: {format_p(rr_test['p'])}",
            transform=ax.transAxes, ha="left", va="top", bbox=text_box)
    inset = ax.inset_axes([0.55, 0.09, 0.41, 0.34])
    inset.axhline(0, color="0.65", lw=0.6)
    inset.plot(S["radius_levels"], np.median(S["offset_by_radius"], axis=0), "-o", color=C_R, ms=3.0, lw=1.1)
    inset.set_xlabel("Target R", fontsize=8); inset.set_ylabel(r"$\Delta R$", fontsize=8)
    inset.tick_params(labelsize=8, direction="out")
    inset.spines["top"].set_visible(False); inset.spines["right"].set_visible(False)
    tidy(ax)

    # B. The temporal nuisance offset is approximately constant across duration levels.
    ax = axes[1]
    positions = np.arange(len(S["duration_levels"]))
    for index in positions:
        draw_box(ax, S["offset_by_duration"][:, index], index, C_T, rng=rng)
    friedman_t = stats.friedmanchisquare(*[S["offset_by_duration"][:, i] for i in positions])
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xticks(positions); ax.set_xticklabels([f"{value:.1f}" for value in S["duration_levels"]])
    ax.set_xlabel("Perceived duration (s)"); ax.set_ylabel(r"$T_{produced}-T_{perceived}$ (s)")
    ax.set_title("Duration error across levels")
    ax.text(0.98, 0.96, f"Friedman {format_p(friedman_t.pvalue)}", transform=ax.transAxes,
            ha="right", va="top", bbox=text_box)
    tidy(ax)

    # C. Symmetric c_R/c_T nuisance-offset control.
    ax = axes[2]
    symmetric = S["symmetric"]["records"]
    c_r = np.asarray([row["c_r_full"] / RAW_SCALE[0] for row in symmetric])
    c_t = np.asarray([row["c_t_full"] / RAW_SCALE[1] for row in symmetric])
    delta_rmse = np.asarray([row["delta_rmse_symmetric_minus_current"] for row in symmetric])
    draw_box(ax, c_r, 0, C_R, rng=rng)
    draw_box(ax, c_t, 1, C_T, rng=rng)
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$c_R/48.5$", r"$c_T/1.2$"])
    ax.set_ylabel("Normalized physical offset")
    ax.set_title(r"Symmetric $c_R/c_T$ offset control")
    cr_test = wilcoxon(c_r, 0.0, "two-sided")
    ct_test = wilcoxon(c_t, 0.0, "two-sided")
    delta_test = wilcoxon(delta_rmse, 0.0, "two-sided")
    ax.text(0.98, 0.96,
            rf"$c_R$: {format_p(cr_test['p'])}" + "\n" + rf"$c_T$: {format_p(ct_test['p'])}" + "\n"
            + rf"Median $\Delta$RMSE={np.median(delta_rmse):+.3f}, {format_p(delta_test['p'])}",
            transform=ax.transAxes, ha="right", va="top", bbox=text_box)
    tidy(ax)

    # D. Speed compression is robust to operational speed definitions.
    ax = axes[3]
    speed_labels = ["Geometric", "Offset\ncorrected", "Path", "Peak"]
    speed_colors = ["0.55", C_V, C_R, C_T]
    for index, (values, color) in enumerate(zip(S["speed_gain_groups"], speed_colors)):
        draw_box(ax, values, index, color, rng=rng)
    ax.axhline(1, color="0.55", lw=0.8, ls="--")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(speed_labels)
    ax.set_ylabel("Log–log dynamic-speed gain"); ax.set_title("Speed-metric robustness")
    ax.text(0.98, 0.96,
            "All < 1:\n" + format_p(max(row["p_holm"] for row in S["speed_gain_tests"])),
            transform=ax.transAxes, ha="right", va="top", bbox=text_box)
    tidy(ax)

    # E. Conditional setpoint strengths in the full RTV model.
    ax = axes[4]
    strengths = [S["strength_r"], S["strength_t"], S["strength_v"]]
    for index, (values, color) in enumerate(zip(strengths, [C_R, C_T, C_V])):
        draw_box(ax, values, index, color, rng=rng)
    r_gt_t = wilcoxon(S["strength_r"] - S["strength_t"], 0.0, "greater")
    v_gt_t = wilcoxon(S["strength_v"] - S["strength_t"], 0.0, "greater")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Radius", "Duration", "Speed"])
    ax.set_ylabel(r"Conditional strength, $\lambda/(1+\lambda)$")
    ax.set_title("Conditional setpoint strengths")
    ax.text(0.98, 0.96, f"Radius > duration: {format_p(r_gt_t['p'])}\nSpeed > duration: {format_p(v_gt_t['p'])}",
            transform=ax.transAxes, ha="right", va="top", bbox=text_box)
    tidy(ax)

    # F. Subject-specific setpoints relative to the tested stimulus range.
    ax = axes[5]
    z_r = np.asarray([row["R0_normalized"] for row in S["normalized_setpoints"]])
    z_v = np.asarray([row["V0_normalized"] for row in S["normalized_setpoints"]])
    ax.axhspan(0, 1, color="0.93", zorder=0)
    draw_box(ax, z_r, 0, C_R, rng=rng)
    draw_box(ax, z_v, 1, C_V, rng=rng)
    ax.axhline(0, color="0.65", lw=0.7); ax.axhline(1, color="0.65", lw=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$R_0$", r"$V_0$"])
    ax.set_ylabel("Normalized log-setpoint location")
    ax.set_title("Setpoint-location identifiability")
    ax.text(0.04, 0.04, "Gray: tested range", transform=ax.transAxes, ha="left", va="bottom", bbox=text_box)
    tidy(ax)

    strict_tests = {}
    mechanistic = MODELS[:-1]
    for ax, scheme, title in zip(axes[6:8], ["radius", "duration"],
                                 ["Leave-one-radius-level-out", "Leave-one-duration-level-out"]):
        by_model = {
            model: np.asarray([row["rmse"] for row in S["strict"][scheme] if row["model"] == model])
            for model in MODELS
        }
        for index, model in enumerate(mechanistic):
            draw_box(ax, by_model[model], index, OFFSET_MODEL_COLORS[model], width=0.58, rng=rng)
        ax.axhline(float(np.median(by_model["FULL"])), color="black", lw=0.9, ls=":", label="Full median")
        labels = [r"$RV_\rho$" if model == "RVrho" else model for model in mechanistic]
        ax.set_xticks(np.arange(len(mechanistic)))
        ax.set_xticklabels(labels, fontsize=7.4, rotation=35, ha="right")
        ax.set_ylabel("Held-out raw joint RMSE"); ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")
        tests = {
            "RV_less_R": wilcoxon(by_model["RV"] - by_model["R"], 0.0, "less"),
            "RVrho_less_RV": wilcoxon(by_model["RVrho"] - by_model["RV"], 0.0, "less"),
            "FULL_less_RVrho": wilcoxon(by_model["FULL"] - by_model["RVrho"], 0.0, "less"),
        }
        strict_tests[scheme] = {"by_model": by_model, "tests": tests}
        ax.text(0.98, 0.68,
                f"RV < R: {format_p(tests['RV_less_R']['p'])}\n"
                + rf"$RV_\rho$ < RV: {format_p(tests['RVrho_less_RV']['p'])}" + "\n"
                + rf"Full < $RV_\rho$: {format_p(tests['FULL_less_RVrho']['p'])}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=text_box)
        tidy(ax)

    # I. Direct recovery of the directional allocation parameter rho.
    ax = axes[8]
    recovery_records = S["rho_recovery"]["records"]
    rho_grid = np.asarray(S["rho_recovery"]["rho_grid"], dtype=float)
    recovered_groups = []
    for true_value in rho_grid:
        values = np.asarray([row["recovered_rho"] for row in recovery_records if np.isclose(row["true_rho"], true_value)])
        recovered_groups.append(values)
        draw_box(ax, values, true_value, C_R, width=0.07, rng=rng)
    all_true = np.asarray([row["true_rho"] for row in recovery_records])
    all_recovered = np.asarray([row["recovered_rho"] for row in recovery_records])
    rho_corr = stats.pearsonr(all_true, all_recovered)
    rho_mae = float(np.median(np.abs(all_recovered - all_true)))
    ax.plot([0, 1], [0, 1], color="0.55", lw=0.9, ls="--")
    ax.set_xlim(0.08, 0.92); ax.set_ylim(0.0, 1.0)
    ax.set_xticks(rho_grid); ax.set_xlabel(r"Generating $\rho$"); ax.set_ylabel(r"Recovered $\rho$")
    ax.set_title(r"Allocation-parameter recovery")
    ax.text(0.04, 0.96, f"Pearson r={rho_corr[0]:.2f}, {format_pearson_p(rho_corr[1])}\nMedian absolute error={rho_mae:.2f}",
            transform=ax.transAxes, ha="left", va="top", bbox=text_box)
    tidy(ax)

    for letter, ax in zip("ABCDEFGHI", axes):
        label_panel(ax, letter)
    TEST_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = TEST_FIG_DIR / "Fig2_setpoint_robustness_audit.png"
    prepare_figure_for_export(fig)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output, {
        "raw_radius_vs_one": rr_test,
        "duration_offset_friedman": {"statistic": float(friedman_t.statistic), "p": float(friedman_t.pvalue)},
        "symmetric_offset": {"c_r_normalized": c_r, "c_t_normalized": c_t, "delta_rmse": delta_rmse,
                             "c_r_vs_zero": cr_test, "c_t_vs_zero": ct_test, "delta_rmse_vs_zero": delta_test},
        "speed_metric_robustness": {"labels": speed_labels, "gain_values": S["speed_gain_groups"],
                                    "less_than_one_tests": S["speed_gain_tests"]},
        "conditional_strength_tests": {"radius_greater_duration": r_gt_t, "speed_greater_duration": v_gt_t},
        "setpoint_locations": {"R0_normalized": z_r, "V0_normalized": z_v},
        "strict_tests": strict_tests,
        "rho_recovery": {"true": all_true, "recovered": all_recovered,
                         "pearson_r": float(rho_corr[0]), "pearson_p": float(rho_corr[1]),
                         "median_absolute_error": rho_mae},
    }


def make_combined_supplement(S, D):
    """Controls that directly support the behavioral-computation main figure."""
    setup_style()
    fig = plt.figure(figsize=(18.2, 5.4))
    grid = fig.add_gridspec(
        1, 4, width_ratios=[3.0, 4.0, 3.0, 3.0],
        left=0.045, right=0.992, bottom=0.16, top=0.82, wspace=0.56,
    )
    b_grid = grid[1].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.48)
    axes = {
        "A": fig.add_subplot(grid[0]),
        "B": fig.add_subplot(b_grid[0, 0]),
        "B_box": fig.add_subplot(b_grid[0, 1]),
        "C": fig.add_subplot(grid[2]),
        "D": fig.add_subplot(grid[3]),
    }
    rng = np.random.RandomState(20260814)
    text_box = {"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 1.0}

    # A. Test whether the empirical temporal offset changes across levels.
    ax = axes["A"]
    positions = np.arange(len(S["duration_levels"]))
    for index in positions:
        draw_box(ax, S["offset_by_duration"][:, index], index, C_T, rng=rng)
    friedman_t = stats.friedmanchisquare(
        *[S["offset_by_duration"][:, index] for index in positions]
    )
    duration_zero_tests = [
        wilcoxon(S["offset_by_duration"][:, index], 0.0, "greater")
        for index in positions
    ]
    duration_zero_p_holm = holm_adjust(
        [test["p"] for test in duration_zero_tests]
    )
    symmetric = S["symmetric"]["records"]
    c_r_raw = np.asarray([row["c_r_full"] for row in symmetric])
    c_r = c_r_raw / RAW_SCALE[0]
    cr_test = wilcoxon(c_r, 0.0, "two-sided")
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{value:.1f}" for value in S["duration_levels"]])
    ax.set_xlabel("Input duration (s)")
    ax.set_ylabel(r"$T_{produced}-T_{perceived}$ (s)")
    ax.set_title(
        "Duration error across levels\n"
        + "Across levels: " + significance_symbol(friedman_t.pvalue)
    )
    annotate_box_p_values(
        ax,
        [S["offset_by_duration"][:, index] for index in positions],
        positions,
        duration_zero_p_holm,
    )
    tidy(ax)

    # B, left. The analogous spatial error changes with radius rather than
    # forming a stable nonzero nuisance offset.
    ax = axes["B"]
    radius_positions = np.arange(len(S["radius_levels"]))
    for index in radius_positions:
        draw_box(ax, S["offset_by_radius"][:, index], index, C_R, rng=rng)
    friedman_r = stats.friedmanchisquare(
        *[S["offset_by_radius"][:, index] for index in radius_positions]
    )
    radius_zero_tests = [
        wilcoxon(S["offset_by_radius"][:, index], 0.0, "two-sided")
        for index in radius_positions
    ]
    radius_zero_p_holm = holm_adjust([test["p"] for test in radius_zero_tests])
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xticks(radius_positions)
    ax.set_xticklabels([f"{value:.0f}" for value in S["radius_levels"]])
    ax.set_xlabel("Input radius (a.u.)")
    ax.set_ylabel(r"$R_{produced}-R_{perceived}$ (a.u.)")
    ax.set_title(
        "Radius error across levels\n"
        + "Across levels: " + significance_symbol(friedman_r.pvalue)
    )
    annotate_box_p_values(
        ax,
        [S["offset_by_radius"][:, index] for index in radius_positions],
        radius_positions,
        radius_zero_p_holm,
    )
    tidy(ax)

    # B, right. Directly summarize the fitted spatial offset parameter.
    ax = axes["B_box"]
    draw_box(ax, c_r_raw, 0, C_R, width=0.48, rng=rng)
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([0])
    ax.set_xticklabels([r"$c_R$"])
    ax.set_ylabel(r"Fitted $c_R$ (a.u.)")
    ax.set_title(r"Fitted $c_R$")
    ax.text(
        0.50, 0.97, significance_symbol(cr_test["p"]),
        transform=ax.transAxes, ha="center", va="top", bbox=text_box,
    )
    tidy(ax)

    # C. The main speed-compression result is independent of speed definition.
    ax = axes["C"]
    speed_labels = ["Geometric", "Offset\ncorrected", "Path", "Peak"]
    speed_colors = ["0.55", C_V, C_R, C_T]
    for index, (values, color) in enumerate(zip(S["speed_gain_groups"], speed_colors)):
        draw_box(ax, values, index, color, rng=rng)
    ax.axhline(1, color="0.55", lw=0.8, ls="--")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(speed_labels)
    ax.set_ylabel("Log–log dynamic-speed gain")
    ax.set_title("Speed-metric robustness")
    annotate_box_p_values(
        ax,
        S["speed_gain_groups"],
        np.arange(4),
        [row["p_holm"] for row in S["speed_gain_tests"]],
    )
    tidy(ax)

    # D. Direct recovery of the allocation parameter used in main panel D.
    ax = axes["D"]
    recovery_records = S["rho_recovery"]["records"]
    rho_grid = np.asarray(S["rho_recovery"]["rho_grid"], dtype=float)
    for true_value in rho_grid:
        values = np.asarray([
            row["recovered_rho"] for row in recovery_records
            if np.isclose(row["true_rho"], true_value)
        ])
        draw_box(ax, values, true_value, C_R, width=0.07, rng=rng)
    all_true = np.asarray([row["true_rho"] for row in recovery_records])
    all_recovered = np.asarray([row["recovered_rho"] for row in recovery_records])
    rho_corr = stats.pearsonr(all_true, all_recovered)
    rho_mae = float(np.median(np.abs(all_recovered - all_true)))
    ax.plot([0, 1], [0, 1], color="0.55", lw=0.9, ls="--")
    ax.set_xlim(0.08, 0.92); ax.set_ylim(0.0, 1.0)
    ax.set_xticks(rho_grid)
    ax.set_xlabel(r"Generating $\rho$"); ax.set_ylabel(r"Recovered $\rho$")
    ax.set_title("Allocation-parameter recovery")
    ax.text(
        0.04, 0.96,
        rf"Descriptive $r$ = {rho_corr[0]:.2f}" + "\n" + rf"Median absolute error = {rho_mae:.2f}",
        transform=ax.transAxes, ha="left", va="top", bbox=text_box,
    )
    tidy(ax)

    for letter in "ABCD":
        label_panel(axes[letter], letter)
    output = FIG_DIR / "SuppFig4_Fig2_behavior_model_controls.png"
    prepare_figure_for_export(fig)
    fig.savefig(output, dpi=300, facecolor="white")
    save_vector_figure(fig, output, facecolor="white")
    plt.close(fig)
    return output, {
        "A_temporal_offset": {
            "duration_offset_friedman": {
                "statistic": float(friedman_t.statistic), "p": float(friedman_t.pvalue),
            },
            "duration_level_vs_zero": {
                "alternative": "greater",
                "levels": S["duration_levels"],
                "tests": duration_zero_tests,
                "p_holm": duration_zero_p_holm,
            },
        },
        "B_spatial_offset": {
            "spatial_error_friedman": {
                "statistic": float(friedman_r.statistic), "p": float(friedman_r.pvalue),
            },
            "radius_level_vs_zero": {
                "alternative": "two-sided",
                "levels": S["radius_levels"],
                "tests": radius_zero_tests,
                "p_holm": radius_zero_p_holm,
            },
            "fitted_spatial_offset": {
                "c_r_radius_units": c_r_raw,
                "c_r_normalized": c_r,
                "c_r_vs_zero": cr_test,
            },
        },
        "C_speed_metric_robustness": {
            "labels": speed_labels, "gain_values": S["speed_gain_groups"],
            "less_than_one_tests": S["speed_gain_tests"],
        },
        "D_rho_recovery": {
            "true": all_true, "recovered": all_recovered,
            "descriptive_pearson_r": float(rho_corr[0]),
            "median_absolute_error": rho_mae,
            "inference": "None; participant-by-generating-parameter recovery points are not treated as independent inferential observations.",
        },
    }


def _draw_effort_manifold(ax, D, text_box):
    feasible = D["effort_feasible"]
    if len(feasible):
        ax.scatter(feasible[:, 2], feasible[:, 3], s=10, color=C_MODEL, alpha=0.20,
                   edgecolor="none", label="Timing-compatible")
    eigval, eigvec = np.linalg.eigh(D["spatial_bootstrap_cov"])
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.maximum(eigval[order], 0), eigvec[:, order]
    scale = math.sqrt(stats.chi2.ppf(0.95, df=2))
    angle = math.degrees(math.atan2(eigvec[1, 0], eigvec[0, 0]))
    ellipse = Ellipse((D["mRR"], D["mRT"]),
                      width=2 * scale * math.sqrt(eigval[0]),
                      height=2 * scale * math.sqrt(eigval[1]), angle=angle,
                      facecolor=C_R, edgecolor=C_R, alpha=0.18, linewidth=1.1,
                      label="Observed 95% CI")
    ax.add_patch(ellipse)
    ax.scatter(D["mRR"], D["mRT"], s=55, color=C_R, edgecolor="black", linewidth=0.7,
               label="Observed", zorder=4)
    if np.all(np.isfinite(D["closest_feasible"])):
        ax.scatter(D["closest_feasible"][2], D["closest_feasible"][3], s=50,
                   facecolor="white", edgecolor="black", marker="D", linewidth=0.8,
                   label="Closest AM1", zorder=5)
    ax.axhline(0, color="0.75", lw=0.6); ax.axvline(1, color="0.8", lw=0.6, ls=":")
    register_formula_text(
        ax.set_xlabel(r"$\beta_{RR}$"),
        "fig2_formula_f_xlabel",
    )
    register_formula_text(
        ax.set_ylabel(r"$\beta_{RT}$"),
        "fig2_formula_f_ylabel",
    )
    ax.yaxis.set_label_coords(-0.14, 0.5)
    ax.set_title("AM1\nTiming-constrained")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.text(0.03, 0.03,
            f"Closest D²={D['closest_feasible_mahalanobis2']:.2f}" + "\n"
            + f"Inside 95% CI: {D.get('feasible_inside_observed_95', 0)}/{D.get('timing_compatible_grid_points', len(feasible))}",
            transform=ax.transAxes, bbox=text_box)
    manifold_x = feasible[:, 2] if len(feasible) else np.asarray([D["mRR"]])
    manifold_y = feasible[:, 3] if len(feasible) else np.asarray([D["mRT"]])
    display_x = np.r_[manifold_x, D["mRR"] - ellipse.width / 2, D["mRR"] + ellipse.width / 2,
                      D["closest_feasible"][2]]
    display_y = np.r_[manifold_y, D["mRT"] - ellipse.height / 2, D["mRT"] + ellipse.height / 2,
                      D["closest_feasible"][3]]
    ax.set_xlim(np.min(display_x) - 0.03, np.max(display_x) + 0.03)
    ax.set_ylim(min(np.min(display_y) - 0.04, -0.04), np.max(display_y) + 0.04)
    tidy(ax)


def _draw_serial_kernel(ax, D, text_box):
    """Draw the empirical lag kernel and normal/probe inset used in main panel H."""
    serial = D["serial"]
    lag_x = np.asarray(SERIAL_LAGS, dtype=float)
    channel_specs = [
        (0, 0, C_R, "-", "o", r"$R_{-k}\to R$"),
        (0, 1, C_R, "--", "s", r"$T_{-k}\to R$"),
        (1, 1, C_T, "-", "o", r"$T_{-k}\to T$"),
        (1, 0, C_T, "--", "s", r"$R_{-k}\to T$"),
    ]
    null_low, null_high = [], []
    for output_index, history_index, color, linestyle, marker, label in channel_specs:
        values = serial["lag_H"][:, :, output_index, history_index]
        median = np.nanmedian(values, axis=0)
        ci = np.asarray([
            _boot_ci(values[:, lag], n=4000, seed=800 + 10 * output_index + history_index + lag)
            for lag in range(len(SERIAL_LAGS))
        ])
        ax.fill_between(lag_x, ci[:, 0], ci[:, 1], color=color, alpha=0.10, linewidth=0)
        ax.plot(lag_x, median, color=color, ls=linestyle, marker=marker, ms=3.8, lw=1.5, label=label)
        if "lag_null_ci" in serial:
            channel_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(output_index, history_index)]
            null_ci = np.asarray(serial["lag_null_ci"][channel_index], dtype=float)
        else:
            null_ci = np.nanpercentile(
                serial["lag_null"][:, :, output_index, history_index], [2.5, 97.5], axis=0
            )
        null_low.append(null_ci[0]); null_high.append(null_ci[1])
    ax.fill_between(
        lag_x, np.min(null_low, axis=0), np.max(null_high, axis=0),
        color="0.65", alpha=0.14, linewidth=0, zorder=0, label="Shifted null",
    )
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(lag_x)
    ax.set_xlabel("Lag (trials)")
    ax.set_ylabel("History coefficient")
    ax.set_title("AM3\nSerial kernel")
    serial_legend = ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="upper left",
        ncol=1,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.15,
    )
    for index, legend_text in enumerate(serial_legend.get_texts()[:4]):
        register_formula_text(legend_text, f"fig2_formula_h_legend_{index}")

    inset = ax.inset_axes([0.68, 0.58, 0.30, 0.34])
    coefficient_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    offsets = np.linspace(-0.12, 0.12, 4)
    for index, ((output_index, history_index), color) in enumerate(
        zip(coefficient_indices, [C_R, C_R, C_T, C_T])
    ):
        medians, intervals = [], []
        for transition_index, transition in enumerate(("normal", "probe")):
            values = serial["transition_H"][transition][:, output_index, history_index]
            medians.append(float(np.nanmedian(values)))
            intervals.append(_boot_ci(values, n=4000, seed=900 + 10 * index + transition_index))
        intervals = np.asarray(intervals)
        x_values = np.asarray([0.0, 1.0]) + offsets[index]
        yerr = np.vstack([
            np.asarray(medians) - intervals[:, 0], intervals[:, 1] - np.asarray(medians)
        ])
        inset.errorbar(x_values, medians, yerr=yerr, color=color, marker="o", ms=2.8,
                       lw=0.8, capsize=1.3)
    inset.axhline(0, color="black", lw=0.6)
    inset.set_xticks([0, 1]); inset.set_xticklabels(["N", "P"], fontsize=7)
    register_formula_text(
        inset.set_title(r"Lag-1 $H$", fontsize=7, pad=1.0),
        "fig2_formula_h_inset_ylabel",
    )
    inset.tick_params(labelsize=7, direction="out")
    inset.spines["top"].set_visible(False); inset.spines["right"].set_visible(False)
    tidy(ax)


def _draw_blocked_history_prediction(ax, D, rng):
    """Draw blocked-CV gains of history models over the current-condition model."""
    validation = D["serial_pooled_blocked"]
    model_names = ["lag1", "lag1_to_5", "exp_all", "exp_drawing"]
    model_labels = ["Lag 1", "Lags 1-5", "Exp. all", "Exp. draw"]
    comparison_order = [
        f"{model_name}_{output}"
        for model_name in model_names
        for output in ("R", "T")
    ]
    if validation["inference"]["comparison_order_current"] != comparison_order:
        raise ValueError("Unexpected blocked-CV comparison order")
    for model_index, model_name in enumerate(model_names):
        values = np.asarray(
            validation["delta_R2_from_current"][model_name]["subject_values"],
            dtype=float,
        )
        for output_index, color in enumerate([C_R, C_T]):
            x_position = model_index + (-0.15 if output_index == 0 else 0.15)
            visible = values[:, output_index][values[:, output_index] >= -0.16]
            draw_box(ax, visible, x_position, color, width=0.23, rng=rng)
            for _ in values[:, output_index][values[:, output_index] < -0.16]:
                ax.scatter(
                    x_position, -0.155, marker="v", s=20, color=color, clip_on=False
                )
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylim(-0.16, 0.08)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(
        model_labels, rotation=60, ha="right", rotation_mode="anchor"
    )
    register_formula_text(
        ax.set_ylabel(r"Held-out $\Delta R^2$"), "fig2_formula_hbox_ylabel"
    )
    ax.set_title("Held-out\nprediction")
    ax.legend(
        handles=[
            Line2D([], [], marker="s", color=C_R, ls="none", label="Radius"),
            Line2D([], [], marker="s", color=C_T, ls="none", label="Duration"),
        ],
        frameon=False, loc="lower left", ncol=2,
    )
    tidy(ax)


def _make_alternative_supplement_full_archive(D):
    """Legacy full layout retained as an audit helper; it is not generated by main()."""
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.2))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.94, wspace=0.43, hspace=0.50)
    axes = axes.ravel()
    rng = np.random.RandomState(20260815)
    text_box = {"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 1.0}

    # A-B. AM1: specified timing-constrained quadratic-effort model.
    _draw_effort_manifold(axes[0], D, text_box)
    ax = axes[1]
    for model_error, baseline_error in zip(D["cv_model_errors"], D["cv_baseline_errors"]):
        ax.plot([0, 1], [model_error, baseline_error], color="0.78", lw=0.65, alpha=0.75)
    draw_box(ax, D["cv_model_errors"], 0, C_MODEL, width=0.38, rng=rng)
    draw_box(ax, D["cv_baseline_errors"], 1, C_MODEL, width=0.38, filled=False, rng=rng)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Effort\nmodel", "Train-mean\nbaseline"])
    ax.set_ylabel("LOSO matrix error"); ax.set_title("AM1: Held-out matrix error")
    ax.text(0.98, 0.96, format_p(D["cv_error_p_adj"]), transform=ax.transAxes,
            ha="right", va="top", bbox=text_box)
    tidy(ax)

    # C. AM2: static proportional noise.
    ax = axes[2]
    xline = np.array([0.10, 0.36])
    ax.plot(xline, D["noise_scale"] * xline, color=C_MODEL, lw=1.4, ls="--", label="AM2 prediction")
    ax.scatter(D["CVR"], D["CR"], s=22, color=C_R, alpha=0.65, label="Space")
    ax.scatter(D["CVT"], D["CT"], s=22, color=C_T, alpha=0.65, label="Time")
    xr = np.linspace(D["CVR"].min(), D["CVR"].max(), 50)
    xt = np.linspace(D["CVT"].min(), D["CVT"].max(), 50)
    ax.plot(xr, np.polyval(D["fitR"], xr), color=C_R, lw=1.7)
    ax.plot(xt, np.polyval(D["fitT"], xt), color=C_T, lw=1.7)
    ax.axhline(0, color="0.72", lw=0.7)
    ax.set_xlim(0.10, 0.37); ax.set_xlabel("Produced noise (CV)"); ax.set_ylabel(r"Compression $1-\beta$")
    ax.set_title("AM2: Static proportional noise"); ax.legend(frameon=False, loc="lower left")
    ax.text(0.98, 0.96,
            f"Duration noisier: {format_p(D['p_noise'])}\n"
            f"Radius r = {D['pearsonR']:+.2f}, {format_pearson_p(D['pearson_pR'])}\n"
            f"Duration r = {D['pearsonT']:+.2f}, {format_pearson_p(D['pearson_pT'])}",
            transform=ax.transAxes, ha="right", va="top", bbox=text_box)
    tidy(ax)

    # D. AM3: empirical serial kernels, with action dependence in an inset.
    ax = axes[3]
    serial = D["serial"]
    lag_x = np.asarray(SERIAL_LAGS, dtype=float)
    channel_specs = [
        (0, 0, C_R, "-", "o", r"$R_{n-k}\to R_n$"),
        (0, 1, C_R, "--", "s", r"$T_{n-k}\to R_n$"),
        (1, 1, C_T, "-", "o", r"$T_{n-k}\to T_n$"),
        (1, 0, C_T, "--", "s", r"$R_{n-k}\to T_n$"),
    ]
    null_low, null_high = [], []
    lag_p = []
    for output_index, history_index, color, linestyle, marker, label in channel_specs:
        values = serial["lag_H"][:, :, output_index, history_index]
        median = np.nanmedian(values, axis=0)
        ci = np.asarray([_boot_ci(values[:, lag], n=4000, seed=800 + 10 * output_index + history_index + lag)
                         for lag in range(len(SERIAL_LAGS))])
        ax.fill_between(lag_x, ci[:, 0], ci[:, 1], color=color, alpha=0.10, linewidth=0)
        ax.plot(lag_x, median, color=color, ls=linestyle, marker=marker, ms=3.8, lw=1.5, label=label)
        if "lag_null_ci" in serial:
            channel_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(output_index, history_index)]
            null_ci = np.asarray(serial["lag_null_ci"][channel_index], dtype=float)
        else:
            null_ci = np.nanpercentile(serial["lag_null"][:, :, output_index, history_index], [2.5, 97.5], axis=0)
        null_low.append(null_ci[0]); null_high.append(null_ci[1])
        lag_p.append([_safe_wilcoxon(values[:, lag]) for lag in range(len(SERIAL_LAGS))])
    ax.fill_between(lag_x, np.min(null_low, axis=0), np.max(null_high, axis=0), color="0.65", alpha=0.14,
                    linewidth=0, zorder=0, label="Shifted null")
    ax.axhline(0, color="black", lw=0.7); ax.set_xticks(lag_x)
    ax.set_xlabel("Chronological lag (trials)"); ax.set_ylabel("Target-history coefficient")
    ax.set_title("AM3: Empirical serial kernel"); ax.legend(frameon=False, fontsize=7.7, loc="upper left")
    inset = ax.inset_axes([0.56, 0.55, 0.42, 0.40])
    coefficient_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    offsets = np.linspace(-0.12, 0.12, 4)
    transition_p = []
    for index, ((output_index, history_index), color) in enumerate(zip(coefficient_indices, [C_R, C_R, C_T, C_T])):
        medians, intervals = [], []
        for transition_index, transition in enumerate(("normal", "probe")):
            values = serial["transition_H"][transition][:, output_index, history_index]
            medians.append(float(np.nanmedian(values)))
            intervals.append(_boot_ci(values, n=4000, seed=900 + 10 * index + transition_index))
        intervals = np.asarray(intervals); xvals = np.asarray([0.0, 1.0]) + offsets[index]
        yerr = np.vstack([np.asarray(medians) - intervals[:, 0], intervals[:, 1] - np.asarray(medians)])
        inset.errorbar(xvals, medians, yerr=yerr, color=color, marker="o", ms=2.8, lw=0.8, capsize=1.3)
        transition_p.append(_safe_wilcoxon(serial["transition_H"]["normal"][:, output_index, history_index]
                                            - serial["transition_H"]["probe"][:, output_index, history_index]))
    inset.axhline(0, color="black", lw=0.6); inset.set_xticks([0, 1]); inset.set_xticklabels(["Normal", "Probe"], fontsize=7)
    inset.set_ylabel(r"Lag-1 $H$", fontsize=7); inset.tick_params(labelsize=7, direction="out")
    inset.spines["top"].set_visible(False); inset.spines["right"].set_visible(False)
    tidy(ax)

    # E. AM3: pooled three-day, within-day blocked cross-validation.
    ax = axes[4]
    validation = D["serial_pooled_blocked"]
    validation_models = ["lag1", "lag1_to_5", "exp_all", "exp_drawing"]
    validation_labels = ["Lag 1", "Lags\n1–5", "Exp.\nall", "Exp.\ndrawing"]
    truncated_values = []
    for model_index, model_name in enumerate(validation_models):
        values = np.asarray(validation["delta_R2_from_current"][model_name]["subject_values"], dtype=float)
        for output_index, color in enumerate([C_R, C_T]):
            x_pos = model_index + (-0.15 if output_index == 0 else 0.15)
            visible = values[:, output_index][values[:, output_index] >= -0.16]
            draw_box(ax, visible, x_pos, color, width=0.23, rng=rng)
            for extreme in values[:, output_index][values[:, output_index] < -0.16]:
                ax.scatter(x_pos, -0.155, marker="v", s=20, color=color, clip_on=False)
                truncated_values.append(("Radius" if output_index == 0 else "Duration", float(extreme)))
    ax.axhline(0, color="black", lw=0.7); ax.set_ylim(-0.16, 0.08)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(validation_labels)
    ax.set_ylabel(r"Blocked-CV $\Delta R^2$ vs current-only")
    ax.set_title("AM3: Held-out prediction")
    ax.legend(handles=[Line2D([], [], marker="s", color=C_R, ls="none", label="Radius"),
                       Line2D([], [], marker="s", color=C_T, ls="none", label="Duration")],
              frameon=False, loc="lower left", ncol=2)
    if truncated_values:
        truncated_text = ", ".join(f"{label}={value:.3f}" for label, value in truncated_values)
        ax.text(0.98, 0.96, "Below axis: " + truncated_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, bbox=text_box)
    tidy(ax)

    # F. AM3: change in the complete behavior matrix after removing lag-1 history.
    ax = axes[5]
    delta = np.column_stack([serial["adjusted_delta"]["lag1_all"][:, i, j] for i, j in coefficient_indices])
    observed = np.mean(serial["adjusted_original"]["lag1_all"], axis=0)
    for index, ((i, j), color) in enumerate(zip(coefficient_indices, [C_R, C_R, C_T, C_T])):
        draw_box(ax, delta[:, index], index, color, width=0.50, points=False, rng=rng)
        value = float(observed[i, j])
        ax.scatter(index, value, s=30, facecolor="white", edgecolor="black", zorder=5)
        ax.text(index, value + 0.035, f"{value:.3f}", ha="center", va="bottom")
    ax.axhline(0, color="black", lw=0.8); ax.set_ylim(-0.10, 1.12)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels([r"$\beta_{RR}$", r"$\beta_{RT}$", r"$\beta_{TR}$", r"$\beta_{TT}$"])
    ax.set_ylabel(r"Observed $\beta$ and lag-1 $\Delta\beta$")
    ax.set_title("AM3: Matrix adjustment")
    ax.text(0.02, 0.98, "Open: observed\nBox: history adjustment", transform=ax.transAxes,
            ha="left", va="top", bbox=text_box)
    tidy(ax)

    # G-H. AM4: both implications of the constant-K two-thirds account.
    ax = axes[6]
    for index, target_t in enumerate(D["Ts"]):
        x, y, ylo, yhi = D["famT"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * index / max(len(D["Ts"]) - 1, 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y); yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        ax.errorbar(lx, ly, yerr=yerr, fmt="-o", color=color, ms=3.4, lw=1.2, capsize=2)
        anchor = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        ax.plot(span, ly[anchor] + (2.0 / 3.0) * (span - lx[anchor]), ":", color=color, lw=1.0)
    ax.set_xlabel(r"$\log_{10}$ produced radius"); ax.set_ylabel(r"$\log_{10}$ produced duration")
    ax.set_title("AM4: Duration–radius prediction")
    ax.text(0.98, 0.04, f"Slope={D['timeR_slope']:.2f}\nvs 2/3: {format_p(D['timeR_p23'])}",
            transform=ax.transAxes, ha="right", va="bottom", bbox=text_box)
    tidy(ax)

    ax = axes[7]
    all_x, all_y = [], []
    for index, target_t in enumerate(D["Ts"]):
        x, y, ylo, yhi = D["fam"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * index / max(len(D["Ts"]) - 1, 1)), 0, 1))
        lx, ly = np.log10(x), np.log10(y); yerr = np.vstack([ly - np.log10(ylo), np.log10(yhi) - ly])
        ax.errorbar(lx, ly, yerr=yerr, fmt="-o", color=color, ms=3.4, lw=1.2, capsize=2)
        anchor = int(np.argmin(lx)); span = np.array([lx.min(), lx.max()])
        ax.plot(span, ly[anchor] + (span - lx[anchor]), ":", color=color, lw=1.0)
        all_x.append(lx); all_y.append(ly)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    center_x, center_y = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min(), all_x.max()])
    ax.plot(span, center_y + (span - center_x) / 3.0, "--", color=C_MODEL, lw=1.3)
    ax.set_xlabel(r"$\log_{10}$ produced radius"); ax.set_ylabel(r"$\log_{10}$ mean speed")
    ax.set_title("AM4: Speed–radius prediction")
    ax.text(0.03, 0.96, f"Slope={D['slope_med']:.2f}\nvs 1/3: {format_p(D['p_vs_third'])}",
            transform=ax.transAxes, ha="left", va="top", bbox=text_box)
    tidy(ax)

    for letter, ax in zip("ABCDEFGH", axes):
        label_panel(ax, letter)
    TEST_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = TEST_FIG_DIR / "Fig2_alternative_behavior_models_full_audit.png"
    prepare_figure_for_export(fig)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output, {
        "am1": {"model_errors": np.asarray(D["cv_model_errors"]), "baseline_errors": np.asarray(D["cv_baseline_errors"]),
                "paired_p_holm": float(D["cv_error_p_adj"]), "closest_mahalanobis2": float(D["closest_feasible_mahalanobis2"])},
        "am2": {"space_cv": D["CVR"], "time_cv": D["CVT"], "space_compression": D["CR"], "time_compression": D["CT"],
                "time_noisier_p": float(D["p_noise"]), "space_pearson_r": float(D["pearsonR"]),
                "space_pearson_p": float(D["pearson_pR"]), "time_pearson_r": float(D["pearsonT"]),
                "time_pearson_p": float(D["pearson_pT"])},
        "am3": {"serial_lags": list(SERIAL_LAGS), "lag_coefficients": serial["lag_H"],
                "lag_shifted_null_95_CI": serial.get("lag_null_ci"), "lag_raw_p": np.asarray(lag_p).T,
                "normal_probe_p_holm": _holm_adjust(np.asarray(transition_p)),
                "blocked_cv": validation["delta_R2_from_current"], "matrix_delta": delta, "observed_matrix": observed},
        "am4": {"duration_radius_subject_slopes": D["timeR_subject_slopes"],
                "duration_radius_slope": float(D["timeR_slope"]),
                "duration_radius_vs_two_thirds_test": D["timeR_test"],
                "duration_radius_vs_two_thirds_p": float(D["timeR_p23"]),
                "speed_radius_slope_median": float(D["slope_med"]), "speed_radius_vs_one_third_p": float(D["p_vs_third"])},
    }


def _make_alternative_supplement_compact_archive(D):
    """Legacy two-panel alternative-model audit; not generated by ``main``."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.17, top=0.90, wspace=0.42)
    rng = np.random.RandomState(20260815)
    text_box = {"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 1.0}

    # A. AM3 held-out prediction, formerly panel E of the full supplement.
    ax = axes[0]
    validation = D["serial_pooled_blocked"]
    validation_models = ["lag1", "lag1_to_5", "exp_all", "exp_drawing"]
    validation_labels = ["Lag 1", "Lags\n1–5", "Exp.\nall", "Exp.\ndrawing"]
    truncated_values = []
    for model_index, model_name in enumerate(validation_models):
        values = np.asarray(validation["delta_R2_from_current"][model_name]["subject_values"], dtype=float)
        for output_index, color in enumerate([C_R, C_T]):
            x_position = model_index + (-0.15 if output_index == 0 else 0.15)
            visible = values[:, output_index][values[:, output_index] >= -0.16]
            draw_box(ax, visible, x_position, color, width=0.23, rng=rng)
            for extreme in values[:, output_index][values[:, output_index] < -0.16]:
                ax.scatter(x_position, -0.155, marker="v", s=20, color=color, clip_on=False)
                truncated_values.append(("Radius" if output_index == 0 else "Duration", float(extreme)))
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylim(-0.16, 0.08)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(validation_labels)
    ax.set_ylabel(r"Blocked-CV $\Delta R^2$ vs current-only")
    ax.set_title("AM3: Held-out prediction")
    ax.legend(
        handles=[Line2D([], [], marker="s", color=C_R, ls="none", label="Radius"),
                 Line2D([], [], marker="s", color=C_T, ls="none", label="Duration")],
        frameon=False, loc="lower left", ncol=2,
    )
    if truncated_values:
        truncated_text = ", ".join(f"{label}={value:.3f}" for label, value in truncated_values)
        ax.text(0.98, 0.96, "Below axis: " + truncated_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, bbox=text_box)
    tidy(ax)

    # B. AM4 speed-radius diagnostic, formerly panel H of the full supplement.
    ax = axes[1]
    all_x, all_y = [], []
    for index, target_t in enumerate(D["Ts"]):
        x, y, ylo, yhi = D["fam"][target_t]
        if len(x) == 0:
            continue
        color = tuple(np.clip(np.array(C_T) * (0.45 + 0.55 * index / max(len(D["Ts"]) - 1, 1)), 0, 1))
        log_x, log_y = np.log10(x), np.log10(y)
        yerr = np.vstack([log_y - np.log10(ylo), np.log10(yhi) - log_y])
        ax.errorbar(log_x, log_y, yerr=yerr, fmt="-o", color=color, ms=3.4, lw=1.2, capsize=2)
        anchor = int(np.argmin(log_x)); span = np.array([log_x.min(), log_x.max()])
        ax.plot(span, log_y[anchor] + (span - log_x[anchor]), ":", color=color, lw=1.0)
        all_x.append(log_x); all_y.append(log_y)
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    center_x, center_y = np.median(all_x), np.median(all_y)
    span = np.array([all_x.min(), all_x.max()])
    ax.plot(span, center_y + (span - center_x) / 3.0, "--", color=C_MODEL, lw=1.3)
    ax.set_xlabel(r"$\log_{10}$ produced radius"); ax.set_ylabel(r"$\log_{10}$ mean speed")
    ax.set_title("AM4: Speed–radius prediction")
    ax.text(0.03, 0.96, f"Slope={D['slope_med']:.2f}\nvs 1/3: {format_p(D['p_vs_third'])}",
            transform=ax.transAxes, ha="left", va="top", bbox=text_box)
    tidy(ax)

    label_panel(axes[0], "A"); label_panel(axes[1], "B")
    TEST_FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = TEST_FIG_DIR / "Fig2_alternative_behavior_models_compact_audit.png"
    prepare_figure_for_export(fig)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output, {
        "A_AM3_held_out_prediction": {
            "blocked_cv": validation["delta_R2_from_current"],
            "inference": validation["inference"],
        },
        "B_AM4_speed_radius": {
            "speed_radius_slope_median": float(D["slope_med"]),
            "speed_radius_vs_one_third_p": float(D["p_vs_third"]),
        },
    }


def _main_stats(S, D):
    serial = D["serial"]
    validation = D["serial_pooled_blocked"]
    serial_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    lag_p = np.asarray([
        [_safe_wilcoxon(serial["lag_H"][:, lag_index, i, j]) for i, j in serial_indices]
        for lag_index in range(len(SERIAL_LAGS))
    ])
    lag_p_holm = np.vstack([_holm_adjust(row) for row in lag_p])
    transition_p = _holm_adjust(np.asarray([
        _safe_wilcoxon(serial["transition_H"]["normal"][:, i, j]
                        - serial["transition_H"]["probe"][:, i, j])
        for i, j in serial_indices
    ]))
    return {
        "description": "Main behavioral-computation figure. The preferred radius/dynamic-speed model family is compared with four independently labelled specified alternatives; free allocation provides a modest held-out improvement over fixed-allocation RV.",
        "model_nomenclature": AM_NAMES,
        "model_scope": "AM1-AM4 denote exact tested formulations, not exhaustive mechanism classes.",
        "panels": {
            "A": {
                "claim": (
                    "Radius, duration, and dynamic speed are treated as parallel "
                    "candidate setpoint coordinates before held-out model comparison."
                ),
                "shared_baseline": "R_b=R*, T_b=T*+c_T, V_b=R_b/T_b",
                "candidate_models": ["R", "T", "V", "RT", "RV", "TV", "RTV"],
                "rvrho_allocation": "Delta r=rho Delta s; Delta t=-(1-rho) Delta s",
            },
            "B": {"speed_gain": S["speed_gain"], "median_speed_gain": float(np.median(S["speed_gain"])),
                  "less_than_one": S["speed_less_one"], "greater_than_zero": S["speed_greater_zero"]},
            "C": {
                "held_out_rmse": S["cv"],
                "display_order": DISPLAY_MODELS,
                "display_groups": [["O"], ["R", "T", "V"], ["RT", "RV", "RVrho", "TV"], ["RTV"], ["FULL"]],
                "primary_displayed_comparisons": ["RV < R", "RV < RT", "RVrho < RV", "RTV < RV"],
                "RVrho_less_RV": S["rvrho_vs_rv"],
                "RTV_less_RV": S["rtv_vs_rv"],
                "FULL_less_RVrho": S["full_vs_rvrho"],
                "RVrho_minus_FULL": S["rvrho_full_difference"],
                "RV_comparisons": S["rv_tests"],
            },
            "D": {"rho": S["rho_training"], "rho_greater_half": S["rho_vs_half"],
                  "asymmetry": S["observed_asymmetry"], "pearson_r": float(S["rho_asymmetry"][0]),
                  "pearson_p": float(S["rho_asymmetry"][1])},
            "E": {"observed": S["observed_matrices"], "RV": S["predicted_rv"], "RVrho": S["predicted_rvrho"],
                  "observed_asymmetry": S["observed_asymmetry"], "RV_asymmetry": S["predicted_asymmetry_rv"],
                  "RVrho_asymmetry": S["predicted_asymmetry_rvrho"]},
            "F_AM1": {"timing_compatible_grid_points": int(D["timing_compatible_grid_points"]),
                      "points_inside_observed_95_CI": int(D["feasible_inside_observed_95"]),
                      "closest_model": D["closest_feasible"],
                      "closest_mahalanobis2": float(D["closest_feasible_mahalanobis2"]),
                      "model_errors": D["cv_model_errors"], "baseline_errors": D["cv_baseline_errors"],
                      "paired_p_holm": float(D["cv_error_p_adj"])},
            "G_AM2": {"space_noise": D["CVR"], "time_noise": D["CVT"], "space_compression": D["CR"],
                      "time_compression": D["CT"], "noise_scale": float(D["noise_scale"]),
                      "inference": am2_inference(D)},
            "H_AM3": {"serial_lags": list(SERIAL_LAGS), "lag_coefficients": serial["lag_H"],
                      "lag_shifted_null_95_CI": serial.get("lag_null_ci"),
                      "lag_p_raw": lag_p, "lag_p_holm_within_lag": lag_p_holm,
                      "normal_probe_p_holm": transition_p,
                      "blocked_cv": validation["delta_R2_from_current"],
                      "blocked_cv_inference": validation["inference"],
                      "p_value_display": (
                          "Blocked-CV P values are omitted from the panel and reserved for the figure legend."
                      ),
                      "off_scale_display": {
                          "axis_lower_limit": -0.16,
                          "marker": "Downward triangle at the lower axis boundary",
                          "model": "exp_all",
                          "output": "Radius",
                          "value": float(np.min(np.asarray(
                              validation["delta_R2_from_current"]["exp_all"]["subject_values"],
                              dtype=float,
                          )[:, 0])),
                          "legend_note": (
                              "One radius held-out delta-R-squared value falls below the displayed axis; "
                              "the triangle marks its truncated position and the exact value is reported "
                              "in the figure legend."
                          ),
                      },
                      "interpretation": (
                          "No tested history variant produced a positive held-out gain "
                          "that survived Holm correction."
                      )},
            "I_AM4": {
                "duration_radius_subject_slopes": D["timeR_subject_slopes"],
                "duration_radius_slope_median": float(D["timeR_slope"]),
                "vs_two_thirds_test": D["timeR_test"],
                "display": (
                    "Independent narrow right-hand boxplot; dashed reference line denotes slope 2/3."
                ),
            },
        },
    }


def main():
    setup_style()
    S = compute_setpoint_results()
    D = load_alternative_results()
    main_png = make_main_figure(S, D)
    supplement_png, supplement_stats = make_combined_supplement(S, D)
    write_json(FIG_DIR / "Fig2_behavior_computational_models_stats.json", _main_stats(S, D))
    write_json(
        FIG_DIR / "SuppFig4_Fig2_behavior_model_controls_stats.json",
        {
            "description": (
                "Compact controls for Figure 2: temporal-offset and spatial-error "
                "profiles, the fitted spatial offset, speed-metric robustness, and "
                "rho recovery."
            ),
            "model_nomenclature": AM_NAMES,
            "panels": supplement_stats,
        },
    )
    print(main_png)
    print(supplement_png)


if __name__ == "__main__":
    main()
