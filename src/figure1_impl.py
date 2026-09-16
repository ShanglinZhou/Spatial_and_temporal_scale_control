from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

from analysis_all_common import (
    FIG_DIR,
    HUMAN_DATA_DIR,
    PLOT_ONLY_MODE,
    PLOT_ONLY_STAGING,
    clean_axis,
    fit_circle_xy,
    human_draw_xy_from_row,
    human_response_from_row,
    load_human_condition_means,
    load_human_subject_rows,
    human_subject_id,
    iter_human_subject_files,
    prepare_figure_for_export,
    save_vector_figure,
    significance_symbol,
    summarize_human_radius_fit_consistency,
    write_json,
)
from analysis_behavior_common import (
    COLORS,
    build_matched_rows,
    component_plus_residual_within_subject,
    fit_subject_models,
    fit_probe_models,
    label_panel,
    normalize,
    plot_coefficient_comparison,
    plot_partial,
    setup_style,
)


PERCEPTION_COLOR = "#B8B8B8"
MOTOR_COLOR = "#4D4D4D"
TIME_CMAP = plt.get_cmap("jet")
LOCKED_EXAMPLE_SUBJECT = "sub-03_clean.allData_clean"
LOCKED_EXAMPLE_TRIALS = {
    "R1T1": {"day": "2", "trial_number": "365"},
    "R1T2": {"day": "3", "trial_number": "216"},
    "R1T3": {"day": "2", "trial_number": "483"},
    "R1T4": {"day": "1", "trial_number": "6"},
    "R1T5": {"day": "3", "trial_number": "285"},
    "R2T1": {"day": "2", "trial_number": "388"},
    "R2T2": {"day": "2", "trial_number": "8"},
    "R2T3": {"day": "2", "trial_number": "295"},
    "R2T4": {"day": "1", "trial_number": "311"},
    "R2T5": {"day": "3", "trial_number": "187"},
    "R3T1": {"day": "3", "trial_number": "150"},
    "R3T2": {"day": "2", "trial_number": "68"},
    "R3T3": {"day": "2", "trial_number": "214"},
    "R3T4": {"day": "3", "trial_number": "326"},
    "R3T5": {"day": "3", "trial_number": "471"},
    "R4T1": {"day": "3", "trial_number": "388"},
    "R4T2": {"day": "2", "trial_number": "164"},
    "R4T3": {"day": "3", "trial_number": "497"},
    "R4T4": {"day": "1", "trial_number": "22"},
    "R4T5": {"day": "3", "trial_number": "247"},
    "R5T1": {"day": "2", "trial_number": "49"},
    "R5T2": {"day": "3", "trial_number": "145"},
    "R5T3": {"day": "3", "trial_number": "210"},
    "R5T4": {"day": "3", "trial_number": "321"},
    "R5T5": {"day": "3", "trial_number": "491"},
}


def _build_temp_path(target: Path) -> Path:
    """Return a same-name build path so stats Markdown routing stays intact."""
    build_dir = target.parent / "_figure_build_tmp"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir / target.name


def _cleanup_build_dir(path: Path) -> None:
    build_dir = path.parent
    if build_dir.name == "_figure_build_tmp" and not any(build_dir.iterdir()):
        build_dir.rmdir()


def write_json_atomic(path: Path, payload) -> None:
    """Write JSON and its Markdown companion before replacing synced outputs."""
    if PLOT_ONLY_MODE and not PLOT_ONLY_STAGING:
        write_json(path, payload)
        return
    temp_path = _build_temp_path(path)
    write_json(temp_path, payload)
    temp_md_path = temp_path.with_suffix(".md")
    target_md_path = path.with_suffix(".md")
    temp_path.replace(path)
    if temp_md_path.exists():
        temp_md_path.replace(target_md_path)
    _cleanup_build_dir(temp_path)


def save_figure_atomic(fig, path: Path) -> None:
    """Render completely before replacing a prior synced PNG."""
    prepare_figure_for_export(fig)
    temp_path = _build_temp_path(path)
    fig.savefig(temp_path)
    temp_path.replace(path)
    _cleanup_build_dir(temp_path)
    save_vector_figure(fig, path)


def normalization_from_rows(condition_rows):
    stim_r = np.asarray([float(row["stimx"]) for row in condition_rows], dtype=float)
    stim_t = np.asarray([float(row["stimt"]) for row in condition_rows], dtype=float)
    bounds = {
        "R": {"min": float(np.min(stim_r)), "max": float(np.max(stim_r))},
        "T": {"min": float(np.min(stim_t)), "max": float(np.max(stim_t))},
    }
    return {
        key: {
            **value,
            "center": (value["min"] + value["max"]) / 2.0,
            "half_range": (value["max"] - value["min"]) / 2.0,
        }
        for key, value in bounds.items()
    }


def load_behavior_example_traces(condition_rows, n_target_points=181):
    r_levels = sorted({float(row["stimx"]) for row in condition_rows})
    t_levels = sorted({float(row["stimt"]) for row in condition_rows})
    conditions = [
        (f"R{r_index + 1}T{t_index + 1}", target_r, target_t)
        for r_index, target_r in enumerate(r_levels)
        for t_index, target_t in enumerate(t_levels)
    ]
    trials = {label: [] for label, _, _ in conditions}
    condition_lookup = {
        (radius, duration): label
        for label, radius, duration in conditions
    }
    suffix = ".allData_clean"
    mat_stem = LOCKED_EXAMPLE_SUBJECT[:-len(suffix)] if LOCKED_EXAMPLE_SUBJECT.endswith(suffix) else LOCKED_EXAMPLE_SUBJECT
    mat_path = HUMAN_DATA_DIR / f"{mat_stem}.mat"
    if not mat_path.exists():
        raise RuntimeError(f"Locked Figure 1A subject file not found: {mat_path}.")
    for row in load_human_subject_rows(mat_path):
        if str(row.get("trialtype", "")).strip().lower() != "normal":
            continue
        key = (float(row["stimx"]), float(row["stimt"]))
        label = condition_lookup.get(key)
        if label is None:
            continue
        output_xy = human_draw_xy_from_row(row, min_points=20)
        if output_xy is None:
            continue
        response = human_response_from_row(row)
        trials[label].append(
            {
                "subject": LOCKED_EXAMPLE_SUBJECT,
                "day": str(row.get("day", "")),
                "trial_number": str(row.get("trialN", "")),
                "xy": output_xy,
                "produced_r": float(response["respx"]),
                "produced_t": float(response["respt"]),
                "produced_r_csv": float(response["respx_csv"]),
                "produced_r_fit": float(response["respx_fit"]),
                "produced_r_source": str(response["respx_source"]),
            }
        )

    examples = []
    target_phase = np.linspace(0.0, 1.0, n_target_points)
    for label, target_r, target_t in conditions:
        candidates = trials[label]
        if not candidates:
            raise RuntimeError(f"No valid normal traces found for {label}.")
        radii = np.asarray([trial["produced_r"] for trial in candidates], dtype=float)
        durations = np.asarray([trial["produced_t"] for trial in candidates], dtype=float)
        center_r = float(np.mean(radii))
        center_t = float(np.mean(durations))
        scale_r = float(np.median(np.abs(radii - np.median(radii))))
        scale_t = float(np.median(np.abs(durations - np.median(durations))))
        scale_r = scale_r if scale_r > 0 else max(float(np.std(radii)), 1.0)
        scale_t = scale_t if scale_t > 0 else max(float(np.std(durations)), 0.01)
        scores = ((radii - center_r) / scale_r) ** 2 + ((durations - center_t) / scale_t) ** 2
        selection_order = np.argsort(scores, kind="stable")
        locked_trial = LOCKED_EXAMPLE_TRIALS[label]
        locked_indices = [
            index
            for index, candidate in enumerate(candidates)
            if float(candidate["day"]) == float(locked_trial["day"])
            and float(candidate["trial_number"]) == float(locked_trial["trial_number"])
        ]
        if len(locked_indices) == 1:
            selected_index = int(locked_indices[0])
            selection_source = "locked"
        else:
            selected_index = int(selection_order[0])
            selection_source = "nearest_condition_center_replacement"
            replacement = candidates[selected_index]
            print(
                f"Replacing unavailable locked Figure 1A trial for {label}: "
                f"{locked_trial} -> day {replacement['day']}, trial {replacement['trial_number']}."
            )
        selected_rank = int(np.flatnonzero(selection_order == selected_index)[0])
        selected = candidates[selected_index]
        output_xy = selected["xy"]
        output_r = float(selected["produced_r"])
        output_t = float(selected["produced_t"])
        fit_cx, fit_cy, fit_radius = fit_circle_xy(output_xy[:, 0], output_xy[:, 1])
        output_phase = np.linspace(0.0, 1.0, output_xy.shape[0])
        theta = 2.0 * np.pi * target_phase
        target_xy = np.column_stack([target_r * np.sin(theta), target_r * (np.cos(theta) - 1.0)])
        fit_xy = np.column_stack(
            [fit_cx + fit_radius * np.cos(theta), fit_cy + fit_radius * np.sin(theta)]
        )
        examples.append(
            {
                "role": label,
                "label": label,
                "target_r": float(target_r),
                "target_t": float(target_t),
                "target_xy": target_xy,
                "target_time": target_phase * target_t,
                "output_xy": output_xy,
                "output_time": output_phase * output_t,
                "fit_xy": fit_xy,
                "fit_center": np.asarray([fit_cx, fit_cy], dtype=float),
                "fit_radius": float(fit_radius),
                "n_candidate_trials": len(candidates),
                "selection_rank_zero_based": int(selected_rank),
                "selection_source": selection_source,
                "subject_condition_mean_r": center_r,
                "subject_condition_mean_t": center_t,
                "subject": selected["subject"],
                "day": selected["day"],
                "trial_number": selected["trial_number"],
                "produced_r": output_r,
                "produced_r_csv": float(selected["produced_r_csv"]),
                "produced_r_fit": float(selected["produced_r_fit"]),
                "produced_r_source": str(selected["produced_r_source"]),
                "produced_t": output_t,
            }
        )
    return examples


def add_gradient_trace(ax, xy, time, cmap, norm, linestyle, linewidth, zorder, dash_pattern=None):
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    segment_time = (time[:-1] + time[1:]) / 2.0
    if dash_pattern is not None:
        on_count, off_count = dash_pattern
        mask = (np.arange(segments.shape[0]) % (on_count + off_count)) < on_count
        segments = segments[mask]
        segment_time = segment_time[mask]
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linestyles=linestyle,
        linewidths=linewidth,
        zorder=zorder,
    )
    collection.set_array(segment_time)
    ax.add_collection(collection)
    return collection


def set_editable_relation_title(ax, title, formula, r_value) -> None:
    """Keep prose as live SVG text while isolating the MathText formula."""
    ax.set_title("")
    plain_props = {"fontfamily": "Arial", "fontsize": 10, "color": "black"}
    math_props = {"fontsize": 10, "color": "black"}
    first_line = HPacker(
        children=[
            TextArea(f"{title}: ", textprops=plain_props),
            TextArea(formula, textprops=math_props),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    second_line = HPacker(
        children=[
            TextArea("Pooled descriptive ", textprops=plain_props),
            TextArea(r"$r$", textprops=math_props),
            TextArea(f" = {r_value:.2f}", textprops=plain_props),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    title_box = VPacker(
        children=[first_line, second_line], align="center", pad=0, sep=1
    )
    anchored_title = AnchoredOffsetbox(
        loc="lower center",
        child=title_box,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        bbox_transform=ax.transAxes,
        borderpad=0,
        pad=0,
    )
    ax.add_artist(anchored_title)


def p_value_offset_box(p_value, *, include_prefix: bool):
    """Build editable p-value text with a manually raised plain-text exponent."""
    p_value = float(p_value)
    plain_props = {"fontfamily": "Arial", "fontsize": 10, "color": "black"}
    if p_value < 0.001:
        exponent = int(np.floor(np.log10(p_value)))
        mantissa = p_value / (10.0 ** exponent)
        prefix = "P = " if include_prefix else ""
        base = TextArea(f"{prefix}{mantissa:.2f} × 10", textprops=plain_props)
        exponent_text = TextArea(
            f"{exponent:d}",
            textprops={"fontfamily": "Arial", "fontsize": 7, "color": "black"},
        )
        raised_exponent = VPacker(
            children=[exponent_text, DrawingArea(0, 4)],
            align="left",
            pad=0,
            sep=0,
        )
        return HPacker(
            children=[base, raised_exponent], align="bottom", pad=0, sep=0
        )
    prefix = "P = " if include_prefix else ""
    return TextArea(f"{prefix}{p_value:.3f}", textprops=plain_props)


def make_comparison_text_editable(ax, labels, references, stats) -> None:
    """Show non-Pearson inference as editable significance symbols."""
    tests = stats["tests"]
    comparison_test = tests[f"{labels[0]}_vs_{labels[1]}"]
    for artist in list(ax.texts):
        artist_text = artist.get_text()
        if "$P$" in artist_text or (
            artist_text in {"*", "**", "***", "****", "n.s.", "n/a"}
            and tuple(artist.get_position()) in {(0.5, 1.055), (0, 0.985), (1, 0.985)}
        ):
            artist.remove()
    top_p_box = AnchoredOffsetbox(
        loc="lower center",
        child=TextArea(
            significance_symbol(comparison_test["p_bh"]),
            textprops={"fontfamily": "Arial", "fontsize": 10, "color": "black"},
        ),
        frameon=False,
        bbox_to_anchor=(0.5, 1.055),
        bbox_transform=ax.get_xaxis_transform(),
        borderpad=0,
        pad=0,
    )
    ax.add_artist(top_p_box)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    axis_transform = ax.get_xaxis_transform()
    for position, label, reference in zip((0, 1), labels, references):
        test = tests[f"{label}_vs_{reference:g}"]
        ax.text(
            position,
            0.985,
            significance_symbol(test["p_bh"]),
            ha="center",
            va="bottom",
            fontsize=10,
            transform=axis_transform,
            clip_on=False,
        )


def draw_task_schematic(ax, examples) -> None:
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Task and behavioral readouts")
    levels = np.linspace(-1.0, 1.0, 5)
    rr, tt = np.meshgrid(levels, levels)
    ax.scatter(0.04 + 0.12 * (rr.ravel() + 1) / 2, 0.25 + 0.50 * (tt.ravel() + 1) / 2, s=13, color="0.45")
    ax.annotate("", xy=(0.18, 0.22), xytext=(0.03, 0.22), arrowprops={"arrowstyle": "->", "color": COLORS["T"], "lw": 1.0})
    ax.annotate("", xy=(0.025, 0.80), xytext=(0.025, 0.22), arrowprops={"arrowstyle": "->", "color": COLORS["R"], "lw": 1.0})
    ax.text(0.10, 0.14, "Input T (a.u.)", color=COLORS["T"], ha="center", fontsize=10)
    ax.text(0.005, 0.50, "Input R (a.u.)", color=COLORS["R"], ha="center", va="center", rotation=90, fontsize=10)
    ax.text(0.10, 0.83, "5 × 5 conditions", ha="center", fontsize=10)

    boxes = [
        (0.23, 0.57, "Probe", "Perceived R, T"),
        (0.23, 0.22, "Normal", "Produced R, T"),
    ]
    for x, y, title, output in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.22, 0.20, fill=False, lw=0.9, color="black"))
        ax.text(x + 0.11, y + 0.14, title, ha="center", va="center", fontweight="bold", fontsize=10)
        ax.text(x + 0.11, y + 0.06, output, ha="center", va="center", fontsize=10)
        ax.annotate("", xy=(x, y + 0.10), xytext=(0.18, 0.49), arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9})
    ax.text(0.24, 0.04, "Fixed stimulus-range normalization: −1 to 1", ha="center", fontsize=10)

    example_by_role = {example["role"]: example for example in examples}
    shared_time_max = max(
        max(float(example["target_time"][-1]), float(example["output_time"][-1]))
        for example in examples
    )
    time_norm = Normalize(vmin=0.0, vmax=shared_time_max)

    def spatial_limits_for(group, pad_fraction=0.07):
        all_xy = [example[key] for example in group for key in ("target_xy", "output_xy", "fit_xy")]
        x_min = min(float(np.min(xy[:, 0])) for xy in all_xy)
        x_max = max(float(np.max(xy[:, 0])) for xy in all_xy)
        y_min = min(float(np.min(xy[:, 1])) for xy in all_xy)
        y_max = max(float(np.max(xy[:, 1])) for xy in all_xy)
        span = max(x_max - x_min, y_max - y_min, 1.0)
        pad = pad_fraction * span
        return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)

    def draw_trace(inset, example, limits, show_radius=False, show_value=True):
        inset.plot(
            example["target_xy"][:, 0],
            example["target_xy"][:, 1],
            color=COLORS["R"],
            lw=0.9,
            ls="--",
            zorder=1,
        )
        inset.plot(
            example["fit_xy"][:, 0],
            example["fit_xy"][:, 1],
            color="0.15",
            lw=1.2,
            ls="-",
            zorder=2,
        )
        add_gradient_trace(
            inset,
            example["output_xy"],
            example["output_time"],
            TIME_CMAP,
            time_norm,
            "solid",
            1.7,
            3,
        )
        if show_radius:
            fit_center = np.asarray(example["fit_center"], dtype=float)
            fit_angle = np.deg2rad(32.0)
            fit_end = fit_center + example["fit_radius"] * np.asarray(
                [np.cos(fit_angle), np.sin(fit_angle)]
            )
            inset.scatter(
                [fit_center[0]], [fit_center[1]], s=12, color="0.05", zorder=5
            )
            inset.annotate(
                "",
                xy=fit_end,
                xytext=fit_center,
                arrowprops={"arrowstyle": "<->", "color": "0.05", "lw": 1.0},
                zorder=5,
            )
            fit_mid = (fit_center + fit_end) / 2.0
            inset.text(
                fit_mid[0],
                fit_mid[1],
                rf"$\hat{{R}}_P={example['fit_radius']:.1f}$",
                color="0.05",
                fontsize=10,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.3},
                zorder=6,
            )
        elif show_value:
            inset.text(
                0.04,
                0.05,
                rf"$\hat{{R}}_P={example['fit_radius']:.1f}$",
                transform=inset.transAxes,
                color="0.05",
                fontsize=10,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.3},
                zorder=6,
            )
        inset.set_xlim(limits[0], limits[1])
        inset.set_ylim(limits[2], limits[3])
        inset.set_aspect("equal", adjustable="box")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(False)

    displayed_examples = [
        example_by_role[f"R{r_index}T{t_index}"]
        for r_index in range(5, 0, -1)
        for t_index in range(1, 6)
    ]
    shared_spatial_limits = spatial_limits_for(displayed_examples)
    x_positions = [0.50, 0.595, 0.690, 0.785, 0.880]
    y_positions = [0.700, 0.575, 0.450, 0.325, 0.200]

    ax.text(
        0.73,
        0.96,
        "Example hand traces across all conditions",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
    )
    for t_index, x_position in enumerate(x_positions, start=1):
        example = example_by_role[f"R3T{t_index}"]
        ax.text(
            x_position + 0.0425,
            0.88,
            rf"$T_T={example['target_t']:.1f}$ s",
            ha="center",
            va="center",
            fontsize=10,
        )

    for r_index, y_position in zip(range(5, 0, -1), y_positions):
        row_example = example_by_role[f"R{r_index}T3"]
        ax.text(
            0.485,
            y_position + 0.0525,
            rf"$R_T={row_example['target_r']:.0f}$",
            ha="right",
            va="center",
            fontsize=10,
        )
        for t_index, x_position in enumerate(x_positions, start=1):
            example = example_by_role[f"R{r_index}T{t_index}"]
            inset = ax.inset_axes([x_position, y_position, 0.085, 0.105])
            draw_trace(
                inset,
                example,
                shared_spatial_limits,
                show_radius=False,
                show_value=False,
            )

    legend_handles = [
        Line2D([0], [0], color=COLORS["R"], lw=0.9, ls="--", label="Target"),
        Line2D([0], [0], color="0.15", lw=1.2, ls="-", label="Fitted circle"),
        Line2D([0], [0], color=TIME_CMAP(0.75), lw=1.7, ls="-", label="Hand trace"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.005),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    colorbar_ax = ax.inset_axes([0.79, 0.025, 0.17, 0.018])
    scalar_mappable = plt.cm.ScalarMappable(norm=time_norm, cmap=TIME_CMAP)
    colorbar = plt.colorbar(scalar_mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([0.0, 1.0, 2.0, 3.0])
    colorbar.set_label("Time (s)", fontsize=10)
    colorbar.ax.tick_params(labelsize=10, length=2)


def draw_interaction_schematic(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Interaction model")
    boxes = {
        "Input\nR": (0.02, 0.82),
        "Input\nT": (0.56, 0.82),
        "Perceived\nR": (0.02, 0.47),
        "Perceived\nT": (0.56, 0.47),
        "Produced\nR": (0.02, 0.12),
        "Produced\nT": (0.56, 0.12),
    }
    for text, (x, y) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), 0.42, 0.11, fill=False, lw=1.0, color="black"))
        ax.text(x + 0.21, y + 0.055, text, ha="center", va="center", fontsize=10)

    direct = [
        ((0.23, 0.82), (0.23, 0.58)),
        ((0.77, 0.82), (0.77, 0.58)),
        ((0.23, 0.47), (0.23, 0.23)),
        ((0.77, 0.47), (0.77, 0.23)),
    ]
    for start, end in direct:
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.78"},
        )
    cross = [
        ((0.77, 0.82), (0.23, 0.58), r"$\beta_{RT}^{P}$", COLORS["RT"], (0.66, 0.70)),
        ((0.23, 0.82), (0.77, 0.58), r"$\beta_{TR}^{P}$", COLORS["TR"], (0.34, 0.70)),
        ((0.77, 0.47), (0.23, 0.23), r"$\beta_{RT}^{M}$", COLORS["RT"], (0.66, 0.35)),
        ((0.23, 0.47), (0.77, 0.23), r"$\beta_{TR}^{M}$", COLORS["TR"], (0.34, 0.35)),
    ]
    for start, end, label, color, position in cross:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": color})
        ax.text(
            *position,
            label,
            color=color,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.2},
        )
    ax.text(0.50, 0.02, "Direct paths shown in gray", ha="center", fontsize=10, color="0.4")


def draw_direct_control_schematic(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Direct-control model")
    boxes = {
        "Input\nR": (0.02, 0.82),
        "Input\nT": (0.56, 0.82),
        "Perceived\nR": (0.02, 0.47),
        "Perceived\nT": (0.56, 0.47),
        "Produced\nR": (0.02, 0.12),
        "Produced\nT": (0.56, 0.12),
    }
    for text, (x, y) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), 0.42, 0.11, fill=False, lw=1.0, color="black"))
        ax.text(x + 0.21, y + 0.055, text, ha="center", va="center", fontsize=10)

    cross = [
        ((0.77, 0.82), (0.23, 0.58)),
        ((0.23, 0.82), (0.77, 0.58)),
        ((0.77, 0.47), (0.23, 0.23)),
        ((0.23, 0.47), (0.77, 0.23)),
    ]
    for start, end in cross:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.78"},
        )
    direct = [
        ((0.23, 0.82), (0.23, 0.58), r"$\beta_{RR}^{P}$", COLORS["R"], (0.13, 0.70)),
        ((0.77, 0.82), (0.77, 0.58), r"$\beta_{TT}^{P}$", COLORS["T"], (0.87, 0.70)),
        ((0.23, 0.47), (0.23, 0.23), r"$\beta_{RR}^{M}$", COLORS["R"], (0.13, 0.35)),
        ((0.77, 0.47), (0.77, 0.23), r"$\beta_{TT}^{M}$", COLORS["T"], (0.87, 0.35)),
    ]
    for start, end, label, color, position in direct:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 1.7, "color": color},
        )
        ax.text(*position, label, color=color, ha="center", va="center", fontsize=10)
    ax.text(0.50, 0.02, "Cross paths shown in gray", ha="center", fontsize=10, color="0.4")


def raw_relation(rows, trial_key, input_key, output_key, normalization, dimension):
    selected = [row for row in rows if trial_key in str(row["trialtype"]).lower()]
    norm = normalization[dimension]
    x = normalize([row[input_key] for row in selected], norm["center"], norm["half_range"])
    y = normalize([row[output_key] for row in selected], norm["center"], norm["half_range"])
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def load_human_condition_variability():
    grouped = {}
    for mat_path in iter_human_subject_files(HUMAN_DATA_DIR):
        for row in load_human_subject_rows(mat_path):
            trialtype = str(row.get("trialtype", "")).strip().lower()
            if trialtype not in {"probe", "normal"}:
                continue
            key = (
                human_subject_id(mat_path),
                trialtype,
                float(row["stimx"]),
                float(row["stimt"]),
            )
            response = human_response_from_row(row)
            if not np.isfinite(float(response["respx"]) + float(response["respt"])):
                continue
            grouped.setdefault(key, []).append((float(response["respx"]), float(response["respt"])))
    rows = []
    for (subject, trialtype, stim_r, stim_t), values in grouped.items():
        array = np.asarray(values, dtype=float)
        if array.shape[0] < 2:
            continue
        rows.append(
            {
                "subject": subject,
                "trialtype": trialtype,
                "stimx": stim_r,
                "stimt": stim_t,
                "mean_r": float(np.mean(array[:, 0])),
                "mean_t": float(np.mean(array[:, 1])),
                "sd_r": float(np.std(array[:, 0], ddof=1)),
                "sd_t": float(np.std(array[:, 1], ddof=1)),
                "n": int(array.shape[0]),
            }
        )
    return rows


def fit_variability_models(variability_rows, normalization):
    lookup = {
        (row["subject"], row["trialtype"], row["stimx"], row["stimt"]): row
        for row in variability_rows
    }
    subjects = sorted({row["subject"] for row in variability_rows})
    probe_coefficients = []
    motor_coefficients = []
    probe_arrays = {}
    motor_arrays = {}
    for subject in subjects:
        keys = sorted(
            {
                (row["stimx"], row["stimt"])
                for row in variability_rows
                if row["subject"] == subject
            }
        )
        matched = [
            (lookup[(subject, "probe", stim_r, stim_t)], lookup[(subject, "normal", stim_r, stim_t)])
            for stim_r, stim_t in keys
            if (subject, "probe", stim_r, stim_t) in lookup
            and (subject, "normal", stim_r, stim_t) in lookup
        ]
        if not matched:
            continue
        probe_rows = [pair[0] for pair in matched]
        normal_rows = [pair[1] for pair in matched]
        r_input = normalize(
            [row["stimx"] for row in probe_rows],
            normalization["R"]["center"],
            normalization["R"]["half_range"],
        )
        t_input = normalize(
            [row["stimt"] for row in probe_rows],
            normalization["T"]["center"],
            normalization["T"]["half_range"],
        )
        perceived_r = normalize(
            [row["mean_r"] for row in probe_rows],
            normalization["R"]["center"],
            normalization["R"]["half_range"],
        )
        perceived_t = normalize(
            [row["mean_t"] for row in probe_rows],
            normalization["T"]["center"],
            normalization["T"]["half_range"],
        )
        probe_sd_r = np.asarray([row["sd_r"] for row in probe_rows], dtype=float) / normalization["R"]["half_range"]
        probe_sd_t = np.asarray([row["sd_t"] for row in probe_rows], dtype=float) / normalization["T"]["half_range"]
        motor_sd_r = np.asarray([row["sd_r"] for row in normal_rows], dtype=float) / normalization["R"]["half_range"]
        motor_sd_t = np.asarray([row["sd_t"] for row in normal_rows], dtype=float) / normalization["T"]["half_range"]

        probe_design = np.column_stack([np.ones(r_input.size), r_input, t_input])
        probe_beta_r, *_ = np.linalg.lstsq(probe_design, probe_sd_r, rcond=None)
        probe_beta_t, *_ = np.linalg.lstsq(probe_design, probe_sd_t, rcond=None)
        motor_design = np.column_stack([np.ones(perceived_r.size), perceived_r, perceived_t])
        motor_beta_r, *_ = np.linalg.lstsq(motor_design, motor_sd_r, rcond=None)
        motor_beta_t, *_ = np.linalg.lstsq(motor_design, motor_sd_t, rcond=None)
        probe_coefficients.append(
            {
                "subject": subject,
                "n_conditions": int(r_input.size),
                "beta_RR_sd": float(probe_beta_r[1]),
                "beta_RT_sd": float(probe_beta_r[2]),
                "beta_TR_sd": float(probe_beta_t[1]),
                "beta_TT_sd": float(probe_beta_t[2]),
            }
        )
        motor_coefficients.append(
            {
                "subject": subject,
                "n_conditions": int(perceived_r.size),
                "beta_RR_sd": float(motor_beta_r[1]),
                "beta_RT_sd": float(motor_beta_r[2]),
                "beta_TR_sd": float(motor_beta_t[1]),
                "beta_TT_sd": float(motor_beta_t[2]),
            }
        )
        probe_arrays[subject] = {
            "R_in": r_input,
            "T_in": t_input,
            "R_sd_out": probe_sd_r,
            "T_sd_out": probe_sd_t,
        }
        motor_arrays[subject] = {
            "R_in": perceived_r,
            "T_in": perceived_t,
            "R_sd_out": motor_sd_r,
            "T_sd_out": motor_sd_t,
        }
    return probe_coefficients, motor_coefficients, probe_arrays, motor_arrays


def concatenate_arrays(subject_arrays, x_key, y_key):
    x = np.concatenate([arrays[x_key] for arrays in subject_arrays.values()])
    y = np.concatenate([arrays[y_key] for arrays in subject_arrays.values()])
    return x, y


def plot_raw_relation(ax, x, y, title, xlabel, ylabel, color):
    ax.scatter(x, y, s=10, alpha=0.20, color=color, edgecolor="none")
    slope, intercept = np.polyfit(x, y, 1)
    lo = float(np.min(x))
    hi = float(np.max(x))
    xx = np.linspace(lo, hi, 100)
    ax.plot(xx, xx, color="0.45", lw=0.8, ls="--")
    ax.plot(xx, slope * xx + intercept, color=color, lw=1.5)
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_title(f"{title}\nPooled descriptive $r$ = {r:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    return {
        "n_condition_means": int(x.size),
        "r_descriptive": r,
        "slope_descriptive": float(slope),
        "intercept_descriptive": float(intercept),
        "identity_line": True,
        "display_role": "Pooled condition-level descriptive visualization only.",
        "inference": "No inferential p value is displayed in this relationship panel.",
    }


def coefficient_map(rows):
    return {str(row["subject"]): row for row in rows}


def common_stats_header(normalization, n_subjects):
    return {
        "normalization": {
            "formula": "X_star = (X - center) / half_range; the designed stimulus range maps to [-1, 1].",
            "constants": normalization,
            "clipping": False,
        },
        "n_subjects": int(n_subjects),
        "inference": "Two-sided exact Wilcoxon signed-rank tests on subject-level coefficients. Multiple comparisons are controlled with Benjamini-Hochberg FDR within each coefficient-comparison panel (3 tests per panel), not pooled across panels; both raw p (p_raw) and BH-adjusted p (p_bh) are reported.",
        "display_policy": {
            "relationship_panels": "Pooled Pearson r and fitted lines are descriptive only; no inferential p values are displayed.",
            "coefficient_panels": "Subject-specific coefficient distributions carry the group inference and display BH-FDR-adjusted significance symbols (* p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001; n.s. otherwise).",
        },
    }


def main() -> None:
    setup_style()
    rows = load_human_condition_means()
    radius_fit_consistency = summarize_human_radius_fit_consistency()
    example_traces = load_behavior_example_traces(rows)
    normalization = normalization_from_rows(rows)
    matched_rows = build_matched_rows(rows)
    motor_coefficients, motor_arrays = fit_subject_models(matched_rows, normalization)
    probe_coefficients, probe_arrays, probe_rows = fit_probe_models(rows, normalization)
    variability_rows = load_human_condition_variability()
    variability_probe_coefficients, variability_motor_coefficients, variability_probe_arrays, variability_motor_arrays = (
        fit_variability_models(variability_rows, normalization)
    )

    probe_map = coefficient_map(probe_coefficients)
    motor_map = coefficient_map(motor_coefficients)
    subjects = sorted(set(probe_map) & set(motor_map))
    probe_rr = np.asarray([probe_map[s]["beta_RR"] for s in subjects], dtype=float)
    probe_tt = np.asarray([probe_map[s]["beta_TT"] for s in subjects], dtype=float)
    probe_rt = np.asarray([probe_map[s]["beta_RT"] for s in subjects], dtype=float)
    probe_tr = np.asarray([probe_map[s]["beta_TR"] for s in subjects], dtype=float)
    motor_rr = np.asarray([motor_map[s]["beta_RR"] for s in subjects], dtype=float)
    motor_tt = np.asarray([motor_map[s]["beta_TT"] for s in subjects], dtype=float)
    motor_rt = np.asarray([motor_map[s]["beta_RT"] for s in subjects], dtype=float)
    motor_tr = np.asarray([motor_map[s]["beta_TR"] for s in subjects], dtype=float)
    variability_probe_map = coefficient_map(variability_probe_coefficients)
    variability_motor_map = coefficient_map(variability_motor_coefficients)
    variability_subjects = sorted(set(variability_probe_map) & set(variability_motor_map))
    variability_probe_rr = np.asarray(
        [variability_probe_map[s]["beta_RR_sd"] for s in variability_subjects], dtype=float
    )
    variability_probe_tt = np.asarray(
        [variability_probe_map[s]["beta_TT_sd"] for s in variability_subjects], dtype=float
    )
    variability_motor_rr = np.asarray(
        [variability_motor_map[s]["beta_RR_sd"] for s in variability_subjects], dtype=float
    )
    variability_motor_rt = np.asarray(
        [variability_motor_map[s]["beta_RT_sd"] for s in variability_subjects], dtype=float
    )
    variability_motor_tr = np.asarray(
        [variability_motor_map[s]["beta_TR_sd"] for s in variability_subjects], dtype=float
    )
    variability_motor_tt = np.asarray(
        [variability_motor_map[s]["beta_TT_sd"] for s in variability_subjects], dtype=float
    )

    # Figure 1: Task plus matched direct- and cross-dimensional control analyses.
    # Within each stage, the two descriptive relation panels precede their
    # participant-level coefficient-comparison panel.
    # The unified A-Q figure is denser than the individual source figures.
    # Extra physical height and explicit constrained-layout padding keep panel
    # labels, multiline titles, and the two edge schematics inside the fixed
    # 178-mm publication canvas when the file is generated directly.
    fig1 = plt.figure(figsize=(15.5, 21.6), constrained_layout=True)
    fig1.set_constrained_layout_pads(
        w_pad=0.06, h_pad=0.06, wspace=0.04, hspace=0.06
    )
    grid1 = fig1.add_gridspec(3, 1, height_ratios=[1.55, 2.5, 2.5])
    content_grid = grid1[1, 0].subgridspec(
        2, 5, width_ratios=[0.78, 1.10, 1.10, 0.72, 0.68]
    )
    direct_grid = content_grid[:, 0].subgridspec(3, 1, height_ratios=[0.5, 1.0, 0.5])
    contrast_grid = content_grid[:, 4].subgridspec(3, 1, height_ratios=[0.5, 1.0, 0.5])
    axes1 = {
        "A": fig1.add_subplot(grid1[0, 0]),
        "B": fig1.add_subplot(direct_grid[1, 0]),
        "C": fig1.add_subplot(content_grid[0, 1]),
        "D": fig1.add_subplot(content_grid[0, 2]),
        "E": fig1.add_subplot(content_grid[0, 3]),
        "F": fig1.add_subplot(content_grid[1, 1]),
        "G": fig1.add_subplot(content_grid[1, 2]),
        "H": fig1.add_subplot(content_grid[1, 3]),
        "I": fig1.add_subplot(contrast_grid[1, 0]),
    }
    cross_grid = grid1[2, 0].subgridspec(
        2, 5, width_ratios=[0.78, 1.10, 1.10, 0.72, 0.68]
    )
    cross_model_grid = cross_grid[:, 0].subgridspec(
        3, 1, height_ratios=[0.5, 1.0, 0.5]
    )
    cross_direction_grid = cross_grid[:, 4].subgridspec(
        3, 1, height_ratios=[0.5, 1.0, 0.5]
    )
    axes2 = {
        "A": fig1.add_subplot(cross_model_grid[1, 0]),
        "B": fig1.add_subplot(cross_grid[0, 1]),
        "C": fig1.add_subplot(cross_grid[0, 2]),
        "D": fig1.add_subplot(cross_grid[0, 3]),
        "E": fig1.add_subplot(cross_grid[1, 1]),
        "F": fig1.add_subplot(cross_grid[1, 2]),
        "G": fig1.add_subplot(cross_grid[1, 3]),
        "H": fig1.add_subplot(cross_direction_grid[1, 0]),
    }
    cross_panel_labels = dict(zip("ABCDEFGH", "JKLMNOPQ"))
    draw_task_schematic(axes1["A"], example_traces)
    label_panel(axes1["A"], "A")
    draw_direct_control_schematic(axes1["B"])
    label_panel(axes1["B"], "B")
    perception_relation_specs = [
        ("C", "R_in", "R_out", "T_in", "Perceptual radius control", r"$\beta_{RR}^{P}$", "Input R (a.u.)", "Adjusted perceived R (a.u.)", COLORS["R"]),
        ("D", "T_in", "T_out", "R_in", "Perceptual duration control", r"$\beta_{TT}^{P}$", "Input T (a.u.)", "Adjusted perceived T (a.u.)", COLORS["T"]),
    ]
    perception_relation_stats = {}
    for spec in perception_relation_specs:
        panel, x_key, y_key, covariate_key, title, formula, xlabel, ylabel, color = spec
        x, y = component_plus_residual_within_subject(probe_arrays, x_key, y_key, covariate_key)
        perception_relation_stats[panel] = plot_partial(
            axes1[panel], x, y, title, xlabel, ylabel, color, identity=True
        )
        set_editable_relation_title(
            axes1[panel], title, formula, perception_relation_stats[panel]["r_descriptive"]
        )
        label_panel(axes1[panel], panel)

    motor_relation_specs = [
        ("F", "R_in", "R_out", "T_in", "Motor radius control", r"$\beta_{RR}^{M}$", "Perceived R (a.u.)", "Adjusted produced R (a.u.)", COLORS["R"]),
        ("G", "T_in", "T_out", "R_in", "Motor duration control", r"$\beta_{TT}^{M}$", "Perceived T (a.u.)", "Adjusted produced T (a.u.)", COLORS["T"]),
    ]
    motor_relation_stats = {}
    for spec in motor_relation_specs:
        panel, x_key, y_key, covariate_key, title, formula, xlabel, ylabel, color = spec
        x, y = component_plus_residual_within_subject(motor_arrays, x_key, y_key, covariate_key)
        motor_relation_stats[panel] = plot_partial(
            axes1[panel], x, y, title, xlabel, ylabel, color, identity=True
        )
        set_editable_relation_title(
            axes1[panel], title, formula, motor_relation_stats[panel]["r_descriptive"]
        )
        label_panel(axes1[panel], panel)

    perception_gain_labels = [r"$\beta_{RR}^{P}$", r"$\beta_{TT}^{P}$"]
    perception_gain_references = (1.0, 1.0)
    perception_stats = plot_coefficient_comparison(
        axes1["E"], probe_rr, probe_tt, perception_gain_labels,
        [COLORS["R"], COLORS["T"]], "Perceptual\ndirect gains", references=perception_gain_references,
        compact=True, compact_inline_group=True, outside_adjusted_p_only=True
    )
    make_comparison_text_editable(
        axes1["E"], perception_gain_labels, perception_gain_references, perception_stats
    )
    label_panel(axes1["E"], "E")
    perception_panel_tests = {
        "C": perception_stats["tests"][r"$\beta_{RR}^{P}$_vs_1"],
        "D": perception_stats["tests"][r"$\beta_{TT}^{P}$_vs_1"],
    }
    for panel, test in perception_panel_tests.items():
        perception_relation_stats[panel]["subject_level_test"] = test
        perception_relation_stats[panel]["subject_level_inference_displayed"] = False
        perception_relation_stats[panel]["corresponding_inference_panel"] = "E"
        perception_relation_stats[panel]["r_squared_descriptive"] = (
            perception_relation_stats[panel]["r_descriptive"] ** 2
        )
    motor_gain_labels = [r"$\beta_{RR}^{M}$", r"$\beta_{TT}^{M}$"]
    motor_gain_references = (1.0, 1.0)
    motor_direct_stats = plot_coefficient_comparison(
        axes1["H"], motor_rr, motor_tt, motor_gain_labels,
        [COLORS["R"], COLORS["T"]], "Motor\ndirect gains", references=motor_gain_references,
        compact=True, compact_inline_group=True, outside_adjusted_p_only=True
    )
    make_comparison_text_editable(
        axes1["H"], motor_gain_labels, motor_gain_references, motor_direct_stats
    )
    label_panel(axes1["H"], "H")
    motor_panel_tests = {
        "F": motor_direct_stats["tests"][r"$\beta_{RR}^{M}$_vs_1"],
        "G": motor_direct_stats["tests"][r"$\beta_{TT}^{M}$_vs_1"],
    }
    for panel, test in motor_panel_tests.items():
        motor_relation_stats[panel]["subject_level_test"] = test
        motor_relation_stats[panel]["subject_level_inference_displayed"] = False
        motor_relation_stats[panel]["corresponding_inference_panel"] = "H"
        motor_relation_stats[panel]["r_squared_descriptive"] = motor_relation_stats[panel]["r_descriptive"] ** 2
    delta_p = probe_rr - probe_tt
    delta_m = motor_rr - motor_tt
    contrast_labels = [r"$\Delta_{P}$", r"$\Delta_{M}$"]
    contrast_references = (0.0, 0.0)
    contrast_stats = plot_coefficient_comparison(
        axes1["I"], delta_p, delta_m, contrast_labels,
        [PERCEPTION_COLOR, MOTOR_COLOR], r"$\beta_{RR}-\beta_{TT}$",
        references=contrast_references,
        compact=True, compact_group_names=("P", "M"), outside_adjusted_p_only=True
    )
    make_comparison_text_editable(
        axes1["I"], contrast_labels, contrast_references, contrast_stats
    )
    label_panel(axes1["I"], "I")
    for panel in ("C", "D", "F", "G"):
        axes1[panel].set_aspect(1.0 / axes1[panel].get_data_ratio(), adjustable="box")

    # Lift labels above the two-line relation-title boxes in the compact
    # direct-export layout.
    for panel in ("C", "D", "F", "G"):
        for text_artist in axes1[panel].texts:
            if str(text_artist.get_gid() or "").startswith("panel_label_"):
                text_artist.set_position((-0.35, 1.52))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_path = FIG_DIR / "Fig1_behavior_spatiotemporal_control.png"

    example_target_r = np.asarray(
        sorted({float(example["target_r"]) for example in example_traces}), dtype=float
    )
    example_mean_produced_r = np.asarray(
        [
            np.mean(
                [
                    float(example["subject_condition_mean_r"])
                    for example in example_traces
                    if float(example["target_r"]) == target_r
                ]
            )
            for target_r in example_target_r
        ],
        dtype=float,
    )
    example_spatial_slope, example_spatial_intercept = np.polyfit(
        example_target_r, example_mean_produced_r, 1
    )
    canonical_spatial_slopes = []
    for subject in sorted({str(row["subject"]) for row in rows if row["trialtype"] == "normal"}):
        subject_rows = [
            row for row in rows
            if row["trialtype"] == "normal" and str(row["subject"]) == subject
        ]
        subject_target_r = np.asarray(
            sorted({float(row["stimx"]) for row in subject_rows}), dtype=float
        )
        subject_mean_r = np.asarray(
            [
                np.mean(
                    [
                        float(row["respx"])
                        for row in subject_rows
                        if float(row["stimx"]) == target_r
                    ]
                )
                for target_r in subject_target_r
            ],
            dtype=float,
        )
        subject_slope = float(np.polyfit(subject_target_r, subject_mean_r, 1)[0])
        if (
            subject_mean_r[0] > subject_target_r[0]
            and subject_mean_r[-1] < subject_target_r[-1]
        ):
            canonical_spatial_slopes.append(subject_slope)
    canonical_spatial_median = float(np.median(canonical_spatial_slopes))

    fig1_stats = {
        **common_stats_header(normalization, len(subjects)),
        "description": "Task, perceptual direct relations, perception-corrected motor direct relations, and their double dissociation.",
        "human_radius_metric": {
            "produced_R": "Normal-trial produced radius is recomputed from all finite samples of the drawing trajectory with the algebraic circle fit. Every retained trajectory has at least three unique, non-collinear positions, and no fallback radius is used.",
            "perceived_R": "Probe-trial perceived radius is read from the cleaned MAT data because probe trials have no drawing trajectory.",
            "produced_R_fit_vs_stored": radius_fit_consistency,
        },
        "n_condition_means_per_task": len(probe_rows),
        "direct_control_schematic": {
            "nodes": ["Input R", "Input T", "Perceived R", "Perceived T", "Produced R", "Produced T"],
            "emphasized_paths": ["beta_RR_P", "beta_TT_P", "beta_RR_M", "beta_TT_M"],
            "deemphasized_paths": ["beta_RT_P", "beta_TR_P", "beta_RT_M", "beta_TR_M"],
        },
        "behavior_example_traces": [
            {
                "role": example["role"],
                "condition": example["label"],
                "target_r": example["target_r"],
                "target_t": example["target_t"],
                "n_candidate_trials": example["n_candidate_trials"],
                "selected_subject": example["subject"],
                "selected_day": example["day"],
                "selected_trial_number": example["trial_number"],
                "selection_rank_zero_based": example["selection_rank_zero_based"],
                "produced_r": example["produced_r"],
                "produced_r_csv": example["produced_r_csv"],
                "produced_r_fit": example["produced_r_fit"],
                "produced_r_source": example["produced_r_source"],
                "produced_t": example["produced_t"],
                "fitted_circle_center": example["fit_center"].tolist(),
                "fitted_circle_radius": example["fit_radius"],
                "subject_condition_mean_r": example["subject_condition_mean_r"],
                "subject_condition_mean_t": example["subject_condition_mean_t"],
                "selection_rule": "Locked to the current real handwritten trial, originally selected as the trace closest to the example subject's condition mean in robustly scaled produced-R/produced-T space.",
            }
            for example in example_traces
        ],
        "behavior_example_design": {
            "example_subject": example_traces[0]["subject"],
            "subject_selection_rule": "All displayed traces are locked real trials from the same subject. The subject was selected because its raw spatial regression-to-the-mean slope was closest to the median among subjects that overestimated the smallest target R and underestimated the largest target R.",
            "layout": "5 x 5",
            "columns_left_to_right": ["T1", "T2", "T3", "T4", "T5"],
            "rows_top_to_bottom": ["R5", "R4", "R3", "R2", "R1"],
            "unique_real_trial_count": 25,
            "displayed_cell_count": 25,
            "spatial_axes": "All 25 displayed cells share one spatial range.",
            "spatial_regression_to_mean": {
                "target_R_values": example_target_r.tolist(),
                "mean_produced_R_by_target": example_mean_produced_r.tolist(),
                "slope": float(example_spatial_slope),
                "intercept": float(example_spatial_intercept),
                "smallest_target_bias": float(
                    example_mean_produced_r[0] - example_target_r[0]
                ),
                "largest_target_bias": float(
                    example_mean_produced_r[-1] - example_target_r[-1]
                ),
                "canonical_subject_count": int(len(canonical_spatial_slopes)),
                "canonical_subject_median_slope": canonical_spatial_median,
                "selection": "Closest slope to the canonical-subject median among subjects with positive smallest-target bias and negative largest-target bias.",
            },
            "trace_color": "Jet colormap encodes absolute elapsed time in seconds; all 25 displayed cells share one fixed color range.",
            "trace_color_range_s": [
                0.0,
                max(
                    max(float(example["target_time"][-1]), float(example["output_time"][-1]))
                    for example in example_traces
                ),
            ],
            "circle_fit": "Algebraic least-squares circle fit; produced R is the fitted-circle radius.",
        },
        "probe_model": {
            "model": "Perceived R_star ~ 1 + input R_star + input T_star; perceived T_star ~ 1 + input R_star + input T_star.",
            "cross_terms": True,
            "cross_terms_role": "Estimated as nuisance effects so the displayed direct relationships remove the other input dimension.",
            "subject_coefficients": probe_coefficients,
            "component_plus_residual_panels": perception_relation_stats,
            "direct_gain_statistics": perception_stats,
        },
        "motor_model": {
            "inputs": "Probe-estimated perceived R and T condition means.",
            "subject_direct_coefficients": [
                {"subject": row["subject"], "beta_RR": row["beta_RR"], "beta_TT": row["beta_TT"]}
                for row in motor_coefficients
            ],
            "component_plus_residual_panels": motor_relation_stats,
            "direct_gain_statistics": motor_direct_stats,
        },
        "double_dissociation": {
            "formula": "Delta_P = beta_RR_P - beta_TT_P; Delta_M = beta_RR_M - beta_TT_M.",
            "statistics": contrast_stats,
        },
    }
    # Supplemental Figure 1: raw probe and normal behavior only.
    # New A-D correspond to old SuppFig1 panels A, B, F, and G.
    supp1, supp1_axes_array = plt.subplots(
        2, 2, figsize=(7.6, 7.6), constrained_layout=True
    )
    supp1_axes = dict(zip("ABCD", supp1_axes_array.ravel()))
    supp1_stats = {}
    raw_behavior_specs = [
        ("A", "probe", "stimx", "respx", "R", "Raw probe radius perception", r"Input $R$ (a.u.)", r"Perceived $R$ (a.u.)", COLORS["R"]),
        ("B", "probe", "stimt", "respt", "T", "Raw probe duration perception", r"Input $T$ (a.u.)", r"Perceived $T$ (a.u.)", COLORS["T"]),
        ("C", "normal", "stimx", "respx", "R", "Raw normal radius behavior", r"Input $R$ (a.u.)", r"Produced $R$ (a.u.)", COLORS["R"]),
        ("D", "normal", "stimt", "respt", "T", "Raw normal duration behavior", r"Input $T$ (a.u.)", r"Produced $T$ (a.u.)", COLORS["T"]),
    ]
    for spec in raw_behavior_specs:
        panel, trial_key, input_key, output_key, dimension, title, xlabel, ylabel, color = spec
        x, y = raw_relation(rows, trial_key, input_key, output_key, normalization, dimension)
        supp1_stats[panel] = plot_raw_relation(
            supp1_axes[panel], x, y, title, xlabel, ylabel, color
        )
        label_panel(supp1_axes[panel], panel)
        supp1_axes[panel].set_aspect(
            1.0 / supp1_axes[panel].get_data_ratio(), adjustable="box"
        )

    supp1_path = FIG_DIR / "SuppFig1_Fig1_raw_normal_behavior.png"
    save_figure_atomic(supp1, supp1_path)
    plt.close(supp1)
    supp1_stats_payload = {
        **common_stats_header(normalization, len(subjects)),
        "description": "Raw probe and normal direct relations in a 2 x 2 layout. Panels A-D are the former SuppFig1 panels A, B, F, and G.",
        "panels": supp1_stats,
    }
    supp1_stats_path = FIG_DIR / "SuppFig1_Fig1_raw_normal_behavior_stats.json"
    write_json_atomic(supp1_stats_path, supp1_stats_payload)

    # Figure 1 J-Q: former Figure 2 cross-dimensional analysis, with its
    # original internal layout retained below the direct-control section.
    draw_interaction_schematic(axes2["A"])
    label_panel(axes2["A"], cross_panel_labels["A"])

    perception_interaction_specs = [
        ("B", "T_in", "R_out", "R_in", "Perceptual duration-to-radius", r"$\beta_{RT}^{P}$", "Input T (a.u.)", "Adjusted perceived R (a.u.)", COLORS["RT"]),
        ("C", "R_in", "T_out", "T_in", "Perceptual radius-to-duration", r"$\beta_{TR}^{P}$", "Input R (a.u.)", "Adjusted perceived T (a.u.)", COLORS["TR"]),
    ]
    perception_interaction_panel_stats = {}
    for spec in perception_interaction_specs:
        panel, x_key, y_key, covariate_key, title, formula, xlabel, ylabel, color = spec
        x, y = component_plus_residual_within_subject(
            probe_arrays, x_key, y_key, covariate_key
        )
        perception_interaction_panel_stats[panel] = plot_partial(
            axes2[panel], x, y, title, xlabel, ylabel, color
        )
        set_editable_relation_title(
            axes2[panel], title, formula,
            perception_interaction_panel_stats[panel]["r_descriptive"],
        )
        label_panel(axes2[panel], cross_panel_labels[panel])
    perception_cross_labels = [r"$\beta_{RT}^{P}$", r"$\beta_{TR}^{P}$"]
    perception_cross_references = (0.0, 0.0)
    perception_interaction_stats = plot_coefficient_comparison(
        axes2["D"],
        probe_rt,
        probe_tr,
        perception_cross_labels,
        [COLORS["RT"], COLORS["TR"]],
        "Perceptual\ncross gains",
        references=perception_cross_references,
        compact=True,
        compact_group_names=("RT", "TR"),
        compact_inline_group=True,
        outside_adjusted_p_only=True,
    )
    make_comparison_text_editable(
        axes2["D"], perception_cross_labels, perception_cross_references,
        perception_interaction_stats,
    )
    label_panel(axes2["D"], cross_panel_labels["D"])
    perception_interaction_tests = {
        "B": perception_interaction_stats["tests"][r"$\beta_{RT}^{P}$_vs_0"],
        "C": perception_interaction_stats["tests"][r"$\beta_{TR}^{P}$_vs_0"],
    }
    for panel in ("B", "C"):
        test = perception_interaction_tests[panel]
        perception_interaction_panel_stats[panel]["subject_level_test"] = test
        perception_interaction_panel_stats[panel]["subject_level_inference_displayed"] = False
        perception_interaction_panel_stats[panel]["corresponding_inference_panel"] = "M"
        perception_interaction_panel_stats[panel]["r_squared_descriptive"] = (
            perception_interaction_panel_stats[panel]["r_descriptive"] ** 2
        )

    motor_interaction_specs = [
        ("E", "T_in", "R_out", "R_in", "Motor duration-to-radius", r"$\beta_{RT}^{M}$", "Perceived T (a.u.)", "Adjusted produced R (a.u.)", COLORS["RT"]),
        ("F", "R_in", "T_out", "T_in", "Motor radius-to-duration", r"$\beta_{TR}^{M}$", "Perceived R (a.u.)", "Adjusted produced T (a.u.)", COLORS["TR"]),
    ]
    motor_interaction_panel_stats = {}
    for spec in motor_interaction_specs:
        panel, x_key, y_key, covariate_key, title, formula, xlabel, ylabel, color = spec
        x, y = component_plus_residual_within_subject(
            motor_arrays, x_key, y_key, covariate_key
        )
        motor_interaction_panel_stats[panel] = plot_partial(
            axes2[panel], x, y, title, xlabel, ylabel, color
        )
        set_editable_relation_title(
            axes2[panel], title, formula,
            motor_interaction_panel_stats[panel]["r_descriptive"],
        )
        label_panel(axes2[panel], cross_panel_labels[panel])
    motor_cross_labels = [r"$\beta_{RT}^{M}$", r"$\beta_{TR}^{M}$"]
    motor_cross_references = (0.0, 0.0)
    motor_interaction_stats = plot_coefficient_comparison(
        axes2["G"],
        motor_rt,
        motor_tr,
        motor_cross_labels,
        [COLORS["RT"], COLORS["TR"]],
        "Motor\ncross gains",
        references=motor_cross_references,
        compact=True,
        compact_group_names=("RT", "TR"),
        compact_inline_group=True,
        outside_adjusted_p_only=True,
    )
    make_comparison_text_editable(
        axes2["G"], motor_cross_labels, motor_cross_references,
        motor_interaction_stats,
    )
    label_panel(axes2["G"], cross_panel_labels["G"])
    motor_interaction_tests = {
        "E": motor_interaction_stats["tests"][r"$\beta_{RT}^{M}$_vs_0"],
        "F": motor_interaction_stats["tests"][r"$\beta_{TR}^{M}$_vs_0"],
    }
    for panel in ("E", "F"):
        test = motor_interaction_tests[panel]
        motor_interaction_panel_stats[panel]["subject_level_test"] = test
        motor_interaction_panel_stats[panel]["subject_level_inference_displayed"] = False
        motor_interaction_panel_stats[panel]["corresponding_inference_panel"] = "P"
        motor_interaction_panel_stats[panel]["r_squared_descriptive"] = (
            motor_interaction_panel_stats[panel]["r_descriptive"] ** 2
        )

    delta_interaction_p = probe_rt - probe_tr
    delta_interaction_m = motor_rt - motor_tr
    interaction_direction_labels = [r"$\Delta_{I}^{P}$", r"$\Delta_{I}^{M}$"]
    interaction_direction_references = (0.0, 0.0)
    interaction_direction_stats = plot_coefficient_comparison(
        axes2["H"],
        delta_interaction_p,
        delta_interaction_m,
        interaction_direction_labels,
        [PERCEPTION_COLOR, MOTOR_COLOR],
        r"$\beta_{RT}-\beta_{TR}$",
        references=interaction_direction_references,
        compact=True,
        compact_group_names=("P", "M"),
        outside_adjusted_p_only=True,
    )
    make_comparison_text_editable(
        axes2["H"], interaction_direction_labels,
        interaction_direction_references, interaction_direction_stats,
    )
    label_panel(axes2["H"], cross_panel_labels["H"])
    for panel in ("B", "C", "E", "F"):
        axes2[panel].set_aspect(1.0 / axes2[panel].get_data_ratio(), adjustable="box")

    for panel in ("B", "C", "E", "F"):
        for text_artist in axes2[panel].texts:
            if str(text_artist.get_gid() or "").startswith("panel_label_"):
                text_artist.set_position((-0.35, 1.52))

    fig2_stats = {
        **common_stats_header(normalization, len(subjects)),
        "description": "Cross-dimensional interactions at perceptual and motor stages, with a subject-level comparison of interaction direction.",
        "model": {
            "perception_stage": "Perceived R_star and T_star are each modeled from both normalized input dimensions.",
            "motor_stage": "Produced R_star and T_star are each modeled from both subject- and condition-matched perceived dimensions.",
            "emphasis": "Cross-dimensional paths are colored; direct paths are shown in gray.",
        },
        "perception_model": {
            "subject_coefficients": [
                {"subject": row["subject"], "beta_RT": row["beta_RT"], "beta_TR": row["beta_TR"]}
                for row in probe_coefficients
            ],
            "component_plus_residual_panels": perception_interaction_panel_stats,
            "interaction_gain_statistics": perception_interaction_stats,
        },
        "motor_model": {
            "inputs": "Subject- and condition-matched probe perception means.",
            "subject_coefficients": [
                {"subject": row["subject"], "beta_RT": row["beta_RT"], "beta_TR": row["beta_TR"]}
                for row in motor_coefficients
            ],
            "component_plus_residual_panels": motor_interaction_panel_stats,
            "interaction_gain_statistics": motor_interaction_stats,
        },
        "interaction_direction_comparison": {
            "formula": "Delta_I_P = beta_RT_P - beta_TR_P; Delta_I_M = beta_RT_M - beta_TR_M.",
            "statistics": interaction_direction_stats,
        },
    }
    combined_stats = {
        **common_stats_header(normalization, len(subjects)),
        "description": "Unified Figure 1: task and direct-control analyses (A-I), followed by the matched cross-dimensional analyses (J-Q).",
        "panel_mapping": {
            "direct_control": "A-I: C-D perceptual relations, E perceptual gains, F-G motor relations, H motor gains, and I the between-stage direct-gain contrast.",
            "cross_dimensional_control": "J-Q (former Figure 2 A-H)",
        },
        "direct_control": fig1_stats,
        "cross_dimensional_control": fig2_stats,
    }
    combined_stats_path = FIG_DIR / "Fig1_behavior_spatiotemporal_control_stats.json"
    write_json_atomic(combined_stats_path, combined_stats)
    save_figure_atomic(fig1, fig1_path)
    plt.close(fig1)

    # Supplemental Figure 1: full behavior-variability analysis.
    # Rows correspond to former SuppFig1 C-E, former SuppFig1 H-J, and
    # former SuppFig2 A-C, respectively.
    supp2, supp2_axes_array = plt.subplots(
        3, 3, figsize=(10.5, 11.4), constrained_layout=True
    )
    supp2_axes = dict(zip("ABCDEFGHI", supp2_axes_array.ravel()))
    variability_panels = {}

    direct_variability_specs = [
        ("A", variability_probe_arrays, "R_in", "R_sd_out", "Probe radius variability", r"Input $R$ (a.u.)", r"SD of perceived $R$ (a.u.)", COLORS["R"]),
        ("B", variability_probe_arrays, "T_in", "T_sd_out", "Probe duration variability", r"Input $T$ (a.u.)", r"SD of perceived $T$ (a.u.)", COLORS["T"]),
    ]
    probe_variability_panel_stats = {}
    for spec in direct_variability_specs:
        panel, arrays, x_key, y_key, title, xlabel, ylabel, color = spec
        x, y = concatenate_arrays(arrays, x_key, y_key)
        probe_variability_panel_stats[panel] = plot_partial(
            supp2_axes[panel], x, y, title, xlabel, ylabel, color
        )
        variability_panels[panel] = probe_variability_panel_stats[panel]
        label_panel(supp2_axes[panel], panel)

    probe_variability_gain_stats = plot_coefficient_comparison(
        supp2_axes["C"],
        variability_probe_rr,
        variability_probe_tt,
        [r"$\beta_{RR}^{P,\sigma}$", r"$\beta_{TT}^{P,\sigma}$"],
        [COLORS["R"], COLORS["T"]],
        "Perceptual\nvariability slopes",
        references=(0.0, 0.0),
        compact=True,
        outside_adjusted_p_only=True,
    )
    variability_panels["C"] = probe_variability_gain_stats
    label_panel(supp2_axes["C"], "C")
    probe_variability_tests = {
        "A": probe_variability_gain_stats["tests"][r"$\beta_{RR}^{P,\sigma}$_vs_0"],
        "B": probe_variability_gain_stats["tests"][r"$\beta_{TT}^{P,\sigma}$_vs_0"],
    }
    for panel in ("A", "B"):
        probe_variability_panel_stats[panel]["subject_level_test"] = probe_variability_tests[panel]
        probe_variability_panel_stats[panel]["subject_level_inference_displayed"] = False
        probe_variability_panel_stats[panel]["corresponding_inference_panel"] = "C"

    motor_variability_specs = [
        ("D", "R_in", "R_sd_out", "T_in", r"Motor radius variability: $\beta_{RR}^{M,\sigma}$", r"Perceived $R$ (a.u.)", r"Adjusted SD of produced $R$ (a.u.)", COLORS["R"]),
        ("E", "T_in", "T_sd_out", "R_in", r"Motor duration variability: $\beta_{TT}^{M,\sigma}$", r"Perceived $T$ (a.u.)", r"Adjusted SD of produced $T$ (a.u.)", COLORS["T"]),
    ]
    motor_variability_panel_stats = {}
    for spec in motor_variability_specs:
        panel, x_key, y_key, covariate_key, title, xlabel, ylabel, color = spec
        x, y = component_plus_residual_within_subject(
            variability_motor_arrays, x_key, y_key, covariate_key
        )
        motor_variability_panel_stats[panel] = plot_partial(
            supp2_axes[panel], x, y, title, xlabel, ylabel, color
        )
        variability_panels[panel] = motor_variability_panel_stats[panel]
        label_panel(supp2_axes[panel], panel)

    motor_variability_gain_stats = plot_coefficient_comparison(
        supp2_axes["F"],
        variability_motor_rr,
        variability_motor_tt,
        [r"$\beta_{RR}^{M,\sigma}$", r"$\beta_{TT}^{M,\sigma}$"],
        [COLORS["R"], COLORS["T"]],
        "Motor\nvariability slopes",
        references=(0.0, 0.0),
        compact=True,
        outside_adjusted_p_only=True,
    )
    variability_panels["F"] = motor_variability_gain_stats
    label_panel(supp2_axes["F"], "F")
    motor_variability_tests = {
        "D": motor_variability_gain_stats["tests"][r"$\beta_{RR}^{M,\sigma}$_vs_0"],
        "E": motor_variability_gain_stats["tests"][r"$\beta_{TT}^{M,\sigma}$_vs_0"],
    }
    for panel in ("D", "E"):
        motor_variability_panel_stats[panel]["subject_level_test"] = motor_variability_tests[panel]
        motor_variability_panel_stats[panel]["subject_level_inference_displayed"] = False
        motor_variability_panel_stats[panel]["corresponding_inference_panel"] = "F"

    motor_variability_interaction_specs = [
        (
            "G", "T_in", "R_sd_out", "R_in",
            r"Motor duration-to-radius variability: $\beta_{RT}^{M,\sigma}$",
            r"Perceived $T$ (a.u.)", r"Adjusted SD of produced $R$ (a.u.)", COLORS["RT"],
        ),
        (
            "H", "R_in", "T_sd_out", "T_in",
            r"Motor radius-to-duration variability: $\beta_{TR}^{M,\sigma}$",
            r"Perceived $R$ (a.u.)", r"Adjusted SD of produced $T$ (a.u.)", COLORS["TR"],
        ),
    ]
    motor_variability_interaction_panel_stats = {}
    for spec in motor_variability_interaction_specs:
        panel, x_key, y_key, covariate_key, title, xlabel, ylabel, color = spec
        x, y = component_plus_residual_within_subject(
            variability_motor_arrays, x_key, y_key, covariate_key
        )
        motor_variability_interaction_panel_stats[panel] = plot_partial(
            supp2_axes[panel], x, y, title, xlabel, ylabel, color
        )
        variability_panels[panel] = motor_variability_interaction_panel_stats[panel]
        label_panel(supp2_axes[panel], panel)

    motor_variability_interaction_stats = plot_coefficient_comparison(
        supp2_axes["I"],
        variability_motor_rt,
        variability_motor_tr,
        [r"$\beta_{RT}^{M,\sigma}$", r"$\beta_{TR}^{M,\sigma}$"],
        [COLORS["RT"], COLORS["TR"]],
        "Motor cross-variability\nslopes",
        references=(0.0, 0.0),
        outside_adjusted_p_only=True,
    )
    variability_panels["I"] = motor_variability_interaction_stats
    label_panel(supp2_axes["I"], "I")
    motor_variability_interaction_tests = {
        "G": motor_variability_interaction_stats["tests"][r"$\beta_{RT}^{M,\sigma}$_vs_0"],
        "H": motor_variability_interaction_stats["tests"][r"$\beta_{TR}^{M,\sigma}$_vs_0"],
    }
    for panel in ("G", "H"):
        test = motor_variability_interaction_tests[panel]
        motor_variability_interaction_panel_stats[panel]["subject_level_test"] = test
        motor_variability_interaction_panel_stats[panel]["subject_level_inference_displayed"] = False
        motor_variability_interaction_panel_stats[panel]["corresponding_inference_panel"] = "I"
        motor_variability_interaction_panel_stats[panel]["r_squared_descriptive"] = (
            motor_variability_interaction_panel_stats[panel]["r_descriptive"] ** 2
        )
    for panel in "ABCDEFGHI":
        supp2_axes[panel].set_aspect(
            1.0 / supp2_axes[panel].get_data_ratio(), adjustable="box"
        )

    supp2_path = FIG_DIR / "SuppFig3_Fig1_behavior_variability.png"
    save_figure_atomic(supp2, supp2_path)
    plt.close(supp2)
    supp2_stats = {
        **common_stats_header(normalization, len(subjects)),
        "description": "Unified 3 x 3 behavior-variability figure. A-C are former SuppFig1 C-E, D-F are former SuppFig1 H-J, and G-I are former SuppFig2 A-C.",
        "sd_definition": "Within-subject, within-task, within-condition sample SD with ddof=1, divided by the corresponding fixed stimulus half-range.",
        "panels": variability_panels,
        "probe_variability_model": {
            "subject_coefficients": variability_probe_coefficients,
            "component_panels": probe_variability_panel_stats,
            "direct_variability_statistics": probe_variability_gain_stats,
        },
        "motor_variability_model": {
            "R_output_sd": "intercept_R + beta_RR_sd * R_perceived + beta_RT_sd * T_perceived",
            "T_output_sd": "intercept_T + beta_TR_sd * R_perceived + beta_TT_sd * T_perceived",
            "inputs": "Subject- and condition-matched probe perception means.",
            "subject_coefficients": variability_motor_coefficients,
            "direct_component_panels": motor_variability_panel_stats,
            "direct_variability_statistics": motor_variability_gain_stats,
            "cross_component_panels": motor_variability_interaction_panel_stats,
            "interaction_variability_statistics": motor_variability_interaction_stats,
        },
    }
    supp2_stats_path = FIG_DIR / "SuppFig3_Fig1_behavior_variability_stats.json"
    write_json_atomic(supp2_stats_path, supp2_stats)

    print(f"Saved {fig1_path}")
    print(f"Saved {combined_stats_path}")
    print(f"Saved {supp1_path}")
    print(f"Saved {supp1_stats_path}")
    print(f"Saved {supp2_path}")
    print(f"Saved {supp2_stats_path}")


if __name__ == "__main__":
    main()
