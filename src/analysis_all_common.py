from __future__ import annotations

import json
import html
import math
import os
import re
import sys
import warnings
from types import MethodType
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.font_manager import FontProperties
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from scipy.io import loadmat

from analysis_stats_markdown import write_stats_markdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_DATA_DIR = PROJECT_ROOT / "human_data" / "clean_data"
FIG_DIR = Path(
    os.environ.get(
        "MANUSCRIPT_FIG_DIR",
        str(PROJECT_ROOT / "outputs"),
    )
).resolve()
RNN_RESULTS_DIR = PROJECT_ROOT / "src" / "results" / "motorTraj_circle"
PLOT_ONLY_MODE = (
    os.environ.get("MANUSCRIPT_PLOT_ONLY", "").strip().lower() in {"1", "true", "yes"}
    or "--plot-only" in sys.argv[1:]
)
PLOT_ONLY_STAGING = (
    os.environ.get("MANUSCRIPT_PLOT_ONLY_STAGING", "").strip().lower()
    in {"1", "true", "yes"}
)
EPS = 1e-9
MIN_HUMAN_RADIUS_FIT_POINTS = 3
FINAL_FIGURE_WIDTH_MM = 178.0
FINAL_FIGURE_FONT_PT = 6.0
PANEL_LABEL_FONT_PT = 12.0
AXIS_SPINE_LINEWIDTH_PT = 0.5
AXIS_SPINE_COLOR = "black"
AXIS_TICK_LINEWIDTH_PT = 0.5
AXIS_TICK_LENGTH_MM = 1.0
AXIS_TICK_LENGTH_PT = AXIS_TICK_LENGTH_MM * 72.0 / 25.4
DATA_SCATTER_MIN_POINTS = 5
DATA_SCATTER_MAX_ZORDER = 1.0
DATA_SCATTER_PANEL_FRACTION = 0.030
DATA_SCATTER_MIN_DIAMETER_PT = 1.6
DATA_SCATTER_MAX_DIAMETER_PT = 3.4


def _adaptive_scatter_area_pt2(
    collection: PathCollection,
    fig: plt.Figure,
    n_points: int,
) -> float:
    """Return a publication-size marker-area cap for one data collection.

    The cap is defined in physical points from the shorter dimension of the
    containing axes, then adjusted mildly for collection density. This makes
    marker size depend on the final panel rather than on an arbitrary source
    layout or a fixed percentage of the script's original marker area.
    """
    axes = collection.axes
    if axes is None:
        return DATA_SCATTER_MAX_DIAMETER_PT ** 2
    position = axes.get_position()
    figure_width, figure_height = np.asarray(fig.get_size_inches(), dtype=float)
    short_dimension_pt = 72.0 * min(
        float(position.width) * float(figure_width),
        float(position.height) * float(figure_height),
    )
    base_diameter_pt = float(
        np.clip(
            DATA_SCATTER_PANEL_FRACTION * short_dimension_pt,
            DATA_SCATTER_MIN_DIAMETER_PT,
            DATA_SCATTER_MAX_DIAMETER_PT,
        )
    )
    density_factor = float(
        np.clip((20.0 / max(float(n_points), 1.0)) ** 0.12, 0.78, 1.15)
    )
    diameter_pt = float(
        np.clip(
            base_diameter_pt * density_factor,
            DATA_SCATTER_MIN_DIAMETER_PT,
            DATA_SCATTER_MAX_DIAMETER_PT,
        )
    )
    return diameter_pt ** 2


def _prepare_adaptive_legends(fig: plt.Figure) -> None:
    """Fit legend handles and spacing to the final physical panel size."""
    figure_width, figure_height = np.asarray(fig.get_size_inches(), dtype=float)
    for legend in fig.findobj(Legend):
        if getattr(legend, "_manuscript_legend_prepared", False):
            continue
        labels = [str(text.get_text()) for text in legend.get_texts()]
        if not labels:
            legend._manuscript_legend_prepared = True
            continue

        axes = getattr(legend, "axes", None)
        if axes is None:
            short_dimension_pt = 72.0 * min(float(figure_width), float(figure_height))
        else:
            position = axes.get_position()
            short_dimension_pt = 72.0 * min(
                float(position.width) * float(figure_width),
                float(position.height) * float(figure_height),
            )
        n_items = max(len(labels), 1)
        footprint_scale = float(
            np.clip(
                (max(short_dimension_pt, 1.0) / 100.0) ** 0.30
                * (4.0 / float(n_items)) ** 0.10,
                0.70,
                1.0,
            )
        )

        legend.prop = FontProperties(family="Arial", size=FINAL_FIGURE_FONT_PT)
        legend._fontsize = FINAL_FIGURE_FONT_PT
        legend.handlelength = 1.35 * footprint_scale
        legend.handleheight = 0.70 * footprint_scale
        legend.handletextpad = 0.40 * footprint_scale
        legend.labelspacing = 0.30 * footprint_scale
        legend.columnspacing = 0.80 * footprint_scale
        legend.borderpad = 0.25 * footprint_scale
        legend.borderaxespad = 0.25 * footprint_scale
        marker_diameter_pt = float(np.clip(3.4 * footprint_scale, 2.2, 3.4))
        handle_width_pt = legend.handlelength * FINAL_FIGURE_FONT_PT
        handle_height_pt = max(
            legend.handleheight * FINAL_FIGURE_FONT_PT,
            marker_diameter_pt,
        )

        # Update the existing OffsetBox tree in place. Rebuilding it would
        # replace legend Text objects and break formula-object registration in
        # editable Illustrator SVG output.
        legend._legend_box.pad = legend.borderpad * FINAL_FIGURE_FONT_PT
        legend._legend_box.sep = legend.labelspacing * FINAL_FIGURE_FONT_PT
        legend._legend_handle_box.sep = legend.columnspacing * FINAL_FIGURE_FONT_PT
        for column_box in legend._legend_handle_box.get_children():
            column_box.sep = legend.labelspacing * FINAL_FIGURE_FONT_PT
            for item_box in column_box.get_children():
                item_box.sep = legend.handletextpad * FINAL_FIGURE_FONT_PT
                drawing_areas = [
                    child
                    for child in item_box.get_children()
                    if hasattr(child, "width") and hasattr(child, "height")
                ]
                if not drawing_areas:
                    continue
                drawing_area = drawing_areas[0]
                old_width = max(float(drawing_area.width), 1e-9)
                old_height = max(float(drawing_area.height), 1e-9)
                scale_x = handle_width_pt / old_width
                scale_y = handle_height_pt / old_height
                drawing_area.width = handle_width_pt
                drawing_area.height = handle_height_pt
                for handle in drawing_area.get_children():
                    if isinstance(handle, Line2D):
                        x_data = np.asarray(handle.get_xdata(), dtype=float)
                        y_data = np.asarray(handle.get_ydata(), dtype=float)
                        if x_data.size:
                            handle.set_xdata(x_data * scale_x)
                        if y_data.size:
                            handle.set_ydata(y_data * scale_y)
                        if str(handle.get_marker()).lower() not in {
                            "none", "", " ", "null"
                        }:
                            handle.set_markersize(
                                min(float(handle.get_markersize()), marker_diameter_pt)
                            )
                    elif isinstance(handle, LineCollection):
                        scaled_segments = []
                        for segment in handle.get_segments():
                            values = np.asarray(segment, dtype=float).copy()
                            if values.ndim == 2 and values.shape[1] >= 2:
                                values[:, 0] *= scale_x
                                values[:, 1] *= scale_y
                            scaled_segments.append(values)
                        handle.set_segments(scaled_segments)
                    elif isinstance(handle, PathCollection):
                        offsets = np.asarray(handle.get_offsets(), dtype=float).copy()
                        if offsets.ndim == 2 and offsets.shape[1] >= 2:
                            offsets[:, 0] *= scale_x
                            offsets[:, 1] *= scale_y
                            handle.set_offsets(offsets)
                        sizes = np.asarray(handle.get_sizes(), dtype=float)
                        if sizes.size:
                            largest = float(np.nanmax(sizes))
                            area_cap = marker_diameter_pt ** 2
                            if np.isfinite(largest) and largest > area_cap and largest > 0:
                                handle.set_sizes(sizes * (area_cap / largest))
                    elif isinstance(handle, Patch):
                        handle.set_linewidth(min(float(handle.get_linewidth()), 0.8))
                        if hasattr(handle, "set_width") and hasattr(handle, "get_width"):
                            handle.set_width(float(handle.get_width()) * scale_x)
                        if hasattr(handle, "set_height") and hasattr(handle, "get_height"):
                            handle.set_height(float(handle.get_height()) * scale_y)
        legend.get_frame().set_linewidth(
            min(float(legend.get_frame().get_linewidth()), 0.6)
        )
        legend._manuscript_legend_prepared = True


def _is_panel_label_text(text: Text) -> bool:
    """Identify manuscript panel labels, including legacy untagged labels."""
    gid = str(text.get_gid() or "").lower()
    if gid.startswith("panel_label_"):
        return True
    value = str(text.get_text()).strip()
    weight = text.get_fontweight()
    is_bold = str(weight).lower() in {"bold", "heavy", "black", "semibold", "demibold"}
    if isinstance(weight, (int, float)):
        is_bold = float(weight) >= 600.0
    return (
        len(value) == 1
        and value.isalpha()
        and abs(float(text.get_fontsize()) - PANEL_LABEL_FONT_PT) < 0.01
        and is_bold
    )


def _set_3d_tick_factor_for_renderer(axis, renderer) -> None:
    """Set the nominal projected 3D tick length for the active renderer."""
    import copy

    from mpl_toolkits.mplot3d import proj3d
    try:
        from mpl_toolkits.mplot3d.axis3d import get_flip_min_max
    except ImportError:
        # Matplotlib >=3.8 removed ``get_flip_min_max`` and changed both
        # ``_get_coord_info`` and the 3D tick-linewidth representation. Keep
        # its native projected tick factor, but enforce the manuscript's
        # outward direction, color, and linewidth without using removed APIs.
        tick_style = axis._axinfo["tick"]
        tick_style["inward_factor"] = 0.0
        if not np.isfinite(float(tick_style.get("outward_factor", 0.1))):
            tick_style["outward_factor"] = 0.1
        linewidth = tick_style.get("linewidth")
        if isinstance(linewidth, dict):
            tick_style["linewidth"] = {
                key: AXIS_TICK_LINEWIDTH_PT for key in linewidth
            }
        else:
            tick_style["linewidth"] = AXIS_TICK_LINEWIDTH_PT
        tick_style["color"] = AXIS_SPINE_COLOR
        return

    info = axis._axinfo
    index = info["i"]
    mins, maxs, centers, deltas, _tc, highs = axis._get_coord_info(renderer)
    minmax = np.where(highs, maxs, mins)
    juggled = info["juggled"]
    edge = minmax.copy()
    edge[juggled[0]] = get_flip_min_max(edge, juggled[0], mins, maxs)
    tickdir = info["tickdir"]
    ticksign = 1.0 if highs[tickdir] else -1.0
    tickdelta = float(deltas[tickdir])
    probe_factor = 0.01
    factors = []

    for tick in axis._update_ticks():
        if not tick.get_visible():
            continue
        base = copy.copy(edge)
        base[index] = tick.get_loc()
        base[tickdir] = edge[tickdir]
        outer = copy.copy(base)
        outer[tickdir] = edge[tickdir] + probe_factor * ticksign * tickdelta
        base_projected = proj3d.proj_transform(base[0], base[1], base[2], renderer.M)
        outer_projected = proj3d.proj_transform(outer[0], outer[1], outer[2], renderer.M)
        display_xy = axis.axes.transData.transform(
            [base_projected[:2], outer_projected[:2]]
        )
        probe_pixels = float(np.linalg.norm(display_xy[1] - display_xy[0]))
        if not np.isfinite(probe_pixels) or probe_pixels <= 0:
            continue
        target_pixels = float(renderer.points_to_pixels(AXIS_TICK_LENGTH_PT))
        factors.append(probe_factor * target_pixels / probe_pixels)

    tick_style = info["tick"]
    tick_style["inward_factor"] = 0.0
    if factors:
        tick_style["outward_factor"] = float(np.median(factors))
    tick_style["linewidth"] = AXIS_TICK_LINEWIDTH_PT
    tick_style["color"] = AXIS_SPINE_COLOR


def _draw_3d_axis_with_manuscript_ticks(axis, renderer) -> None:
    """Apply physical tick styling immediately before a 3D axis is drawn."""
    _set_3d_tick_factor_for_renderer(axis, renderer)
    axis._manuscript_original_draw(renderer)


def prepare_figure_for_export(fig: plt.Figure) -> plt.Figure:
    """Apply the manuscript-wide physical canvas and typography contract.

    The final canvas is exactly 178 mm wide. Its height retains the aspect
    ratio selected by the figure script. Ordinary text is Arial 6 pt; panel
    labels are lowercase Arial 12 pt bold. Visible coordinate spines are
    black and 0.5 pt wide. Visible coordinate ticks are black, 0.5 pt wide,
    outward-facing, and 1 mm long. Mathematical subscripts and superscripts
    remain smaller through Matplotlib's normal formula layout.
    """
    size_inches = np.asarray(fig.get_size_inches(), dtype=float)
    if size_inches.shape != (2,) or not np.all(np.isfinite(size_inches)) or np.any(size_inches <= 0):
        raise ValueError(f"Invalid figure size for export: {size_inches!r}")
    final_width_inches = FINAL_FIGURE_WIDTH_MM / 25.4
    final_height_inches = final_width_inches * float(size_inches[1] / size_inches[0])
    fig.set_size_inches(final_width_inches, final_height_inches, forward=True)

    # Standardize coordinate borders without changing which spines each
    # figure intentionally hides.
    # ``fig.axes`` omits some axes created with the legacy ``Axes.inset_axes``
    # API. Recursive discovery keeps coordinate styling consistent in inset
    # plots and inset colorbars as well as in top-level panels.
    all_axes = list(dict.fromkeys(fig.findobj(Axes)))
    for axes in all_axes:
        for spine in axes.spines.values():
            spine.set_edgecolor(AXIS_SPINE_COLOR)
            spine.set_linewidth(AXIS_SPINE_LINEWIDTH_PT)
        axes.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=AXIS_TICK_LENGTH_PT,
            width=AXIS_TICK_LINEWIDTH_PT,
            colors=AXIS_SPINE_COLOR,
        )
        # Matplotlib 3D axes draw their coordinate borders separately from
        # ``axes.spines``. Standardize those visible lines as well while
        # preserving panes whose borders were intentionally transparent.
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(axes, axis_name, None)
            if axis is None:
                continue
            axis_info = getattr(axis, "_axinfo", None)
            if isinstance(axis_info, dict) and "tick" in axis_info:
                tick_style = axis_info["tick"]
                linewidth = tick_style.get("linewidth")
                if isinstance(linewidth, dict):
                    tick_style["linewidth"] = {
                        key: AXIS_TICK_LINEWIDTH_PT for key in linewidth
                    }
                else:
                    tick_style["linewidth"] = AXIS_TICK_LINEWIDTH_PT
                tick_style["color"] = AXIS_SPINE_COLOR
                if not hasattr(axis, "_manuscript_original_draw"):
                    axis._manuscript_original_draw = axis.draw
                    axis.draw = MethodType(_draw_3d_axis_with_manuscript_ticks, axis)
            axis_line = getattr(axis, "line", None)
            if axis_line is not None and axis_line.get_visible():
                axis_line.set_color(AXIS_SPINE_COLOR)
                axis_line.set_linewidth(AXIS_SPINE_LINEWIDTH_PT)
            pane = getattr(axis, "pane", None)
            if pane is not None and pane.get_visible():
                edge_color = pane.get_edgecolor()
                edge_alpha = float(edge_color[-1]) if len(edge_color) >= 4 else 1.0
                if edge_alpha > 0:
                    pane.set_edgecolor((0.0, 0.0, 0.0, edge_alpha))
                    pane.set_linewidth(AXIS_SPINE_LINEWIDTH_PT)

    # Cap dense data markers using the final physical panel size and collection
    # density without changing one-off schematic/highlight markers. If a script
    # uses relative marker sizes as an encoding, rescale the whole size vector
    # proportionally so that encoding is retained. The private flag keeps this
    # idempotent because SVG saving prepares the same figure a second time after
    # the PNG has been written.
    for collection in fig.findobj(PathCollection):
        if getattr(collection, "_manuscript_scatter_prepared", False):
            continue
        offsets = np.ma.asarray(collection.get_offsets())
        n_points = int(offsets.shape[0]) if offsets.ndim >= 2 else 0
        if n_points >= DATA_SCATTER_MIN_POINTS:
            collection.set_zorder(
                min(float(collection.get_zorder()), DATA_SCATTER_MAX_ZORDER)
            )
            sizes = np.asarray(collection.get_sizes(), dtype=float)
            if sizes.size:
                area_cap = _adaptive_scatter_area_pt2(collection, fig, n_points)
                largest = float(np.nanmax(sizes))
                if np.isfinite(largest) and largest > area_cap and largest > 0:
                    collection.set_sizes(sizes * (area_cap / largest))
            collection._manuscript_scatter_prepared = True

    _prepare_adaptive_legends(fig)

    for text in fig.findobj(Text):
        text.set_fontfamily("Arial")
        if _is_panel_label_text(text):
            label = str(text.get_text()).strip().lower()
            text.set_text(label)
            text.set_gid(f"panel_label_{label}")
            text.set_fontsize(PANEL_LABEL_FONT_PT)
            text.set_fontweight("bold")
        else:
            text.set_fontsize(FINAL_FIGURE_FONT_PT)
    return fig


def significance_symbol(p_value: float) -> str:
    """Return the manuscript-wide non-Pearson significance label."""
    p_value = float(p_value)
    if not math.isfinite(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _safe_float(value: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _is_positive_finite(*values: float) -> bool:
    return all(math.isfinite(v) and v > 0 for v in values)


def human_draw_xy_from_row(row: Dict[str, str], min_points: int = 3) -> np.ndarray | None:
    if "draw" in row:
        xy = np.asarray(row.get("draw"), dtype=float)
        if xy.ndim != 2 or xy.shape[1] < 2:
            return None
        xy = xy[:, :2]
    else:
        values = []
        for key in sorted((key for key in row if key.startswith("draw_")), key=lambda item: int(item.split("_")[1])):
            value = row.get(key)
            if value in {None, ""}:
                continue
            x = _safe_float(value)
            if math.isfinite(x):
                values.append(x)
        n_xy = len(values) // 2
        if n_xy < int(min_points):
            return None
        xy = np.column_stack(
            [np.asarray(values[:n_xy], dtype=float), np.asarray(values[n_xy : 2 * n_xy], dtype=float)]
        )
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < int(min_points):
        return None
    xy = xy - xy[0]
    return xy


def human_radius_from_draw_row(row: Dict[str, str], min_points: int = MIN_HUMAN_RADIUS_FIT_POINTS) -> Tuple[float, int]:
    xy = human_draw_xy_from_row(row, min_points=1)
    if xy is None:
        return float("nan"), 0
    if xy.shape[0] < int(min_points):
        return float("nan"), int(xy.shape[0])
    if np.unique(xy, axis=0).shape[0] < 3 or np.linalg.matrix_rank(xy) < 2:
        return float("nan"), int(xy.shape[0])
    _, _, radius = fit_circle_xy(xy[:, 0], xy[:, 1])
    return float(radius), int(xy.shape[0])


def human_response_from_row(row: Dict[str, str]) -> Dict[str, Union[float, str, int]]:
    trialtype = str(row.get("trialtype", "")).strip().lower()
    respx_csv = _safe_float(row.get("respx", ""))
    respt = _safe_float(row.get("respt", ""))
    respx_fit, n_draw_points = human_radius_from_draw_row(row)
    use_fit = trialtype == "normal" and math.isfinite(respx_fit)
    return {
        "respx": float(respx_fit if use_fit else respx_csv),
        "respt": float(respt),
        "respx_csv": float(respx_csv),
        "respx_fit": float(respx_fit),
        "respx_source": "fit_circle_draw" if use_fit else "csv",
        "draw_n_points": int(n_draw_points),
    }


def iter_human_subject_files(data_dir: Path = HUMAN_DATA_DIR) -> Iterator[Path]:
    legacy = list(data_dir.glob("[0-9][0-9]_*_clean.mat"))
    anonymized = list(data_dir.glob("sub-[0-9][0-9]_clean.mat"))
    yield from sorted(set(legacy + anonymized))


def human_subject_id(path: Path) -> str:
    return f"{path.stem}.allData_clean"


def load_human_subject_rows(path: Path) -> List[Dict[str, Union[float, str, np.ndarray]]]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    required = {
        "analysis_numeric",
        "analysis_trialtype",
        "analysis_stimulus_order",
        "analysis_probe_order",
        "analysis_draw",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"{path.name} lacks MAT analysis exports: {', '.join(missing)}")
    numeric = np.asarray(payload["analysis_numeric"], dtype=float)
    if numeric.ndim == 1:
        numeric = numeric.reshape(1, -1)
    if numeric.shape[1] != 6:
        raise RuntimeError(f"Unexpected analysis_numeric shape in {path.name}: {numeric.shape}")
    trialtypes = np.atleast_1d(payload["analysis_trialtype"])
    stimulus_orders = np.atleast_1d(payload["analysis_stimulus_order"])
    probe_orders = np.atleast_1d(payload["analysis_probe_order"])
    draws = np.atleast_1d(payload["analysis_draw"])
    if not all(len(values) == numeric.shape[0] for values in (trialtypes, stimulus_orders, probe_orders, draws)):
        raise RuntimeError(f"MAT export arrays have inconsistent lengths in {path.name}.")

    rows: List[Dict[str, Union[float, str]]] = []
    for index, values in enumerate(numeric):
        row = {
            "day": float(values[0]),
            "trialN": float(values[1]),
            "trialtype": str(trialtypes[index]).strip().lower(),
            "stimulus_order": str(stimulus_orders[index]).strip(),
            "probe_order": str(probe_orders[index]).strip(),
            "stimt": float(values[2]),
            "stimx": float(values[3]),
            "respt": float(values[4]),
            "respx": float(values[5]),
            "draw": np.asarray(draws[index], dtype=float),
        }
        response = human_response_from_row(row)
        row.update(
            {
                "respx": float(response["respx"]),
                "respt": float(response["respt"]),
                "respx_csv": float(response["respx_csv"]),
                "respx_fit": float(response["respx_fit"]),
                "respx_source": str(response["respx_source"]),
                "draw_n_points": int(response["draw_n_points"]),
            }
        )
        if not _is_positive_finite(float(row["stimx"]), float(row["stimt"]), float(row["respx"]), float(row["respt"])):
            raise RuntimeError(f"Non-positive or non-finite retained trial in {path.name}, row {index + 1}.")
        rows.append(row)
    return rows


_load_subject_rows = load_human_subject_rows


def load_human_condition_means(data_dir: Path = HUMAN_DATA_DIR) -> List[Dict[str, Union[float, str]]]:
    grouped: Dict[Tuple[str, str, float, float], List[Tuple[float, float, float, float, int]]] = {}
    for mat_path in iter_human_subject_files(data_dir):
        subject = human_subject_id(mat_path)
        for row in load_human_subject_rows(mat_path):
            trialtype = str(row["trialtype"])
            key = (subject, trialtype, float(row["stimx"]), float(row["stimt"]))
            grouped.setdefault(key, []).append(
                (
                    float(row["respx"]),
                    float(row["respt"]),
                    float(row["respx_csv"]),
                    float(row["respx_fit"]),
                    1 if str(row["respx_source"]) == "fit_circle_draw" else 0,
                )
            )

    out: List[Dict[str, Union[float, str]]] = []
    for (subject, trialtype, stimx, stimt), values in grouped.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        out.append(
            {
                "subject": subject,
                "trialtype": trialtype,
                "stimx": stimx,
                "stimt": stimt,
                "respx": float(np.nanmean(arr[:, 0])),
                "respt": float(np.nanmean(arr[:, 1])),
                "respx_csv": float(np.nanmean(arr[:, 2])),
                "respx_fit": float(np.nanmean(arr[:, 3])) if np.any(np.isfinite(arr[:, 3])) else float("nan"),
                "n_radius_fit_trials": int(np.nansum(arr[:, 4])),
                "n": int(arr.shape[0]),
            }
        )
    return out


def summarize_human_radius_fit_consistency(data_dir: Path = HUMAN_DATA_DIR) -> Dict[str, Union[int, float, List[Dict[str, Union[str, int, float]]]]]:
    differences = []
    fallbacks = []
    for mat_path in iter_human_subject_files(data_dir):
        for row in load_human_subject_rows(mat_path):
            trialtype = str(row.get("trialtype", "")).strip().lower()
            if trialtype != "normal":
                continue
            response = human_response_from_row(row)
            stored_radius = float(response["respx_csv"])
            fit_radius = float(response["respx_fit"])
            if str(response["respx_source"]) == "fit_circle_draw":
                differences.append(fit_radius - stored_radius)
            else:
                fallbacks.append(
                    {
                        "file": mat_path.name,
                        "trial_number": str(row.get("trialN", "")),
                        "stored_radius": stored_radius,
                        "fit_radius": fit_radius if math.isfinite(fit_radius) else None,
                        "draw_n_points": int(response["draw_n_points"]),
                    }
                )
    diff = np.asarray(differences, dtype=float)
    return {
        "normal_trials_using_fit": int(diff.size),
        "normal_trials_stored_fallback": int(len(fallbacks)),
        "max_abs_fit_minus_stored": float(np.max(np.abs(diff))) if diff.size else float("nan"),
        "mean_abs_fit_minus_stored": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
        "n_abs_diff_gt_1e-9": int(np.sum(np.abs(diff) > 1e-9)) if diff.size else 0,
        "fallback_examples": fallbacks[:10],
        "fit_min_points": int(MIN_HUMAN_RADIUS_FIT_POINTS),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload) -> None:
    path = Path(path).resolve()
    serializable = _jsonable(payload)
    if PLOT_ONLY_MODE:
        try:
            path.relative_to(FIG_DIR)
            inside_figure_dir = True
        except ValueError:
            inside_figure_dir = False
        if not (PLOT_ONLY_STAGING and inside_figure_dir):
            if not path.exists():
                raise RuntimeError(
                    f"Plot-only mode forbids creating analysis/cache JSON: {path}"
                )
            expected = json.loads(path.read_text(encoding="utf-8"))
            expected_text = json.dumps(expected, sort_keys=True, separators=(",", ":"))
            actual_text = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
            if expected_text != actual_text:
                raise RuntimeError(
                    f"Plot-only payload differs from the locked JSON and will not be written: {path}"
                )
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    if path.name.endswith("_stats.json") and isinstance(serializable, dict):
        write_stats_markdown(path, serializable)


def _linear_level_map(level: float, base_levels: Sequence[float], base_values: Sequence[float], allow_extrapolate: bool = True) -> float:
    level = float(level)
    x = np.asarray(base_levels, dtype=float)
    y = np.asarray(base_values, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if level < x[0]:
        if not allow_extrapolate:
            raise ValueError(f"Level {level} below trained range")
        slope = (y[1] - y[0]) / max(EPS, x[1] - x[0])
        return float(y[0] + slope * (level - x[0]))
    if level > x[-1]:
        if not allow_extrapolate:
            raise ValueError(f"Level {level} above trained range")
        slope = (y[-1] - y[-2]) / max(EPS, x[-1] - x[-2])
        return float(y[-1] + slope * (level - x[-1]))
    return float(np.interp(level, x, y))


def target_radius_from_level(hp: Dict, level: float) -> float:
    levels = hp.get("circle_target_map_levels", [0.2, 0.4, 0.6, 0.8, 1.0])
    radii = hp.get("circle_radii", [0.45, 0.725, 1.0, 1.275, 1.55])
    return _linear_level_map(level, levels, radii)


def target_duration_from_level(hp: Dict, level: float) -> float:
    levels = hp.get("circle_target_map_levels", [0.2, 0.4, 0.6, 0.8, 1.0])
    durations = hp.get("circle_durations", [0.7, 1.3, 1.9, 2.5, 3.1])
    if hp.get("use_speed_coded_time_input", False):
        durations = list(durations)[::-1]
    return _linear_level_map(level, levels, durations)


def _trial_go_onset(u_in: np.ndarray, hp: Dict) -> int:
    cue = np.asarray(u_in[0], dtype=float)
    on = np.flatnonzero(cue > 0.5)
    if on.size == 0:
        return int(np.asarray(hp.get("stim_on", 0)).reshape(-1)[0])
    return int(on[0])


def _movement_window(u_in: np.ndarray, hp: Dict, target_duration_s: float, T_full: int) -> Tuple[int, int]:
    stim_dur = int(np.asarray(hp.get("stim_dur", 0)).reshape(-1)[0])
    t0 = _trial_go_onset(u_in, hp) + stim_dur
    n_steps = int(round(float(target_duration_s) * 1000.0 / float(hp["dt"])))
    t1 = max(t0 + 1, min(int(T_full), t0 + max(1, n_steps)))
    return int(t0), int(t1)


def fit_circle_xy(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([x_arr, y_arr, np.ones_like(x_arr)])
    b = -(x_arr * x_arr + y_arr * y_arr)
    try:
        a, bb, c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")
    cx = -0.5 * a
    cy = -0.5 * bb
    rad_sq = cx * cx + cy * cy - c
    radius = math.sqrt(rad_sq) if rad_sq > 0 else float("nan")
    return float(cx), float(cy), float(radius)


def _phase_velocity_from_xy(post_xy: np.ndarray, dt_s: float) -> Tuple[float, float]:
    cx, cy, radius = fit_circle_xy(post_xy[0], post_xy[1])
    if not np.isfinite(cx + cy + radius):
        return float("nan"), float("nan")
    angle = np.unwrap(np.arctan2(post_xy[1] - cy, post_xy[0] - cx))
    ccw = angle - angle[0]
    cw = -(angle - angle[0])
    progress = ccw if np.nanmax(ccw) > np.nanmax(cw) else cw
    progress = np.maximum.accumulate(progress)
    t_s = np.arange(progress.size, dtype=float) * dt_s
    if progress.size < 3:
        return radius, float("nan")
    pmax = float(np.nanmax(progress))
    lo, hi = 0.1 * pmax, 0.9 * pmax
    fit_mask = (progress >= lo) & (progress <= hi)
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = np.ones_like(progress, dtype=bool)
    if np.count_nonzero(fit_mask) < 3 or np.nanstd(t_s[fit_mask]) < EPS:
        return radius, float("nan")
    slope = float(np.polyfit(t_s[fit_mask], progress[fit_mask], 1)[0])
    return radius, slope


def _conditions_from_eval(hp: Dict, eval_block: Dict, n_cond: int) -> List[Dict]:
    raw = eval_block.get("conditions")
    if raw is not None and len(raw) == n_cond:
        conds = [dict(c) for c in raw]
    else:
        size_levels = eval_block.get("size_levels")
        speed_levels = eval_block.get("speed_levels")
        condition_types = eval_block.get("condition_types")
        if size_levels is None or speed_levels is None:
            raise ValueError("RNN eval file has no condition metadata.")
        conds = []
        for i in range(n_cond):
            conds.append(
                {
                    "size_level": float(size_levels[i]),
                    "speed_level": float(speed_levels[i]),
                    "condition_type": str(condition_types[i]) if condition_types is not None else "unknown",
                }
            )
    for cond in conds:
        cond["size_level"] = float(cond["size_level"])
        cond["speed_level"] = float(cond["speed_level"])
        cond.setdefault("condition_type", "unknown")
        cond["target_radius"] = float(cond.get("target_radius", target_radius_from_level(hp, cond["size_level"])))
        cond["target_duration"] = float(cond.get("target_duration", target_duration_from_level(hp, cond["speed_level"])))
        cond["target_phase_velocity"] = float(2.0 * math.pi / cond["target_duration"])
    return conds


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", colors="black")
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)


def _prepare_editable_vector_figure(fig: plt.Figure) -> None:
    """Replace raster-only artists with editable vector equivalents."""
    for artist in fig.findobj():
        getter = getattr(artist, "get_rasterized", None)
        if getter is not None and getter():
            artist.set_rasterized(False)

    # ``Axes.inset_axes`` stores nested axes as child artists rather than in
    # ``fig.axes``.  Find images recursively so inset heatmaps are vectorized
    # as well as images on top-level axes.
    for image in list(fig.findobj(AxesImage)):
        ax = image.axes
        if ax is None:
            raise RuntimeError("Editable SVG export found an AxesImage without a parent axes.")
        values = np.asanyarray(image.get_array())
        if values.ndim != 2:
            raise RuntimeError(
                "Editable SVG export supports scalar image matrices only; "
                f"received shape {values.shape}."
            )
        x0, x1, y0, y1 = image.get_extent()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_edges = np.linspace(x0, x1, values.shape[1] + 1)
        y_edges = np.linspace(y0, y1, values.shape[0] + 1)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap=image.get_cmap(),
            norm=image.norm,
            shading="flat",
            edgecolors="none",
            antialiased=False,
            rasterized=False,
            alpha=image.get_alpha(),
            transform=image.get_transform(),
            zorder=image.get_zorder(),
        )
        clip_path = image.get_clip_path()
        if clip_path is not None:
            mesh.set_clip_path(clip_path)
        image.remove()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


_SVG_TSPAN_RE = re.compile(
    r"(?P<open><tspan\b)(?P<attrs>[^>]*)(?P<close>>)(?P<body>.*?)(?P<end></tspan>)",
    re.DOTALL,
)
_SVG_TEXT_RE = re.compile(
    r"(?P<open><text\b)(?P<attrs>[^>]*)(?P<close>>)(?P<body>.*?)(?P<end></text>)",
    re.DOTALL,
)
_SVG_X_ATTRIBUTE_RE = re.compile(r'(?P<prefix>\bx=")(?P<value>[^"]*)(?P<suffix>")')
_SVG_Y_ATTRIBUTE_RE = re.compile(r'\by="(?P<value>[^"]*)"')


def _collapse_single_baseline_svg_text(svg_path: Union[str, Path]) -> int:
    """Let Illustrator import safe SVG text runs as whole text objects.

    Matplotlib MathText writes an explicit x coordinate for every glyph. Adobe
    Illustrator commonly imports those glyphs as separate text objects. Native
    font spacing is safe for a text element containing one run and for a pure
    P-value annotation. Mixed prose/formula elements keep their explicit glyph
    positions because later runs depend on those exact offsets. Runs with
    multiple y coordinates are also preserved because they encode genuine
    subscripts, superscripts, or other mathematical layout.
    """
    svg_path = Path(svg_path)
    source = svg_path.read_text(encoding="utf-8")
    collapsed = 0

    def collapse_tspan(match: re.Match) -> str:
        nonlocal collapsed
        attrs = match.group("attrs")
        x_match = _SVG_X_ATTRIBUTE_RE.search(attrs)
        if x_match is None:
            return match.group(0)
        x_values = x_match.group("value").split()
        if len(x_values) <= 1:
            return match.group(0)
        y_match = _SVG_Y_ATTRIBUTE_RE.search(attrs)
        if y_match is not None and len(y_match.group("value").split()) > 1:
            return match.group(0)
        collapsed += 1
        collapsed_attrs = _SVG_X_ATTRIBUTE_RE.sub(
            lambda x_attr: f'{x_attr.group("prefix")}{x_values[0]}{x_attr.group("suffix")}',
            attrs,
            count=1,
        )
        return (
            f'{match.group("open")}{collapsed_attrs}{match.group("close")}'
            f'{match.group("body")}{match.group("end")}'
        )

    def replace_text(match: re.Match) -> str:
        body = match.group("body")
        tspans = list(_SVG_TSPAN_RE.finditer(body))
        if not tspans:
            return match.group(0)

        visible_text = "".join(html.unescape(tspan.group("body")) for tspan in tspans)
        visible_text = re.sub(r"\s+", " ", visible_text.replace("\xa0", " ")).strip()
        pure_p_value = re.match(r"^[Pp]\s*(?:=|<|>|≤|≥)", visible_text) is not None

        # When several styled/formula runs share one <text> element, changing
        # one run's advance width can move or overlap the following runs.
        if len(tspans) != 1 and not pure_p_value:
            return match.group(0)

        optimized_body = _SVG_TSPAN_RE.sub(collapse_tspan, body)
        return (
            f'{match.group("open")}{match.group("attrs")}{match.group("close")}'
            f'{optimized_body}{match.group("end")}'
        )

    optimized = _SVG_TEXT_RE.sub(replace_text, source)
    if collapsed:
        ET.fromstring(optimized)
        svg_path.write_text(optimized, encoding="utf-8")
    return collapsed


def save_vector_figure(fig: plt.Figure, raster_path: Union[str, Path], **kwargs) -> Path:
    """Save an editable SVG companion after the raster figure is complete."""
    rewrite_formula_text = None
    tag_formula_texts = None
    simple_svg = os.environ.get("OPEN_CODE_SIMPLE_SVG", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not simple_svg:
        try:
            from illustrator_svg_formula import rewrite_formula_text, tag_formula_texts
        except ModuleNotFoundError as error:
            if not (error.name == "lxml" or str(error.name).startswith("lxml.")):
                raise
            warnings.warn(
                "Optional package 'lxml' is unavailable; saving a standard "
                "editable SVG without Illustrator-specific formula rewriting.",
                RuntimeWarning,
                stacklevel=2,
            )

    prepare_figure_for_export(fig)
    # A tight bounding box makes the SVG canvas content-dependent and breaks
    # the fixed 178-mm manuscript width. Exact sizing is controlled by figsize.
    kwargs.pop("bbox_inches", None)
    vector_path = Path(raster_path).with_suffix(".svg")
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_editable_vector_figure(fig)
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", vector_path.stem)
    formula_specs = (
        tag_formula_texts(fig, f"{safe_prefix}_formula")
        if tag_formula_texts is not None
        else {}
    )
    build_dir = vector_path.parent / "_figure_build_tmp"
    build_dir.mkdir(parents=True, exist_ok=True)
    temporary = build_dir / vector_path.name
    fig.savefig(temporary, format="svg", **kwargs)
    _collapse_single_baseline_svg_text(temporary)
    temporary.replace(vector_path)
    if formula_specs and rewrite_formula_text is not None:
        rewrite_formula_text(vector_path, formula_specs)
    if not any(build_dir.iterdir()):
        build_dir.rmdir()
    return vector_path


def finish_figure(fig: plt.Figure, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    prepare_figure_for_export(fig)
    fig.savefig(path)
    save_vector_figure(fig, path)
    plt.close(fig)
    return path
