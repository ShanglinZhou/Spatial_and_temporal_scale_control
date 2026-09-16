from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


_DETAIL_KEYS = {
    "all_fit_summaries",
    "behavior_example_traces",
    "conditions",
    "configurations",
    "human_like_solutions",
    "motor_stage_fit_selected_solutions",
    "load_errors",
    "raw_configs",
    "records",
    "rep_fits",
    "rep_results",
    "reps",
    "subject_coefficients",
    "trajectories",
}
_DETAIL_SUFFIXES = (
    "_rep_matrices",
    "_subject_coefficients",
    "_trajectories",
    "_traces",
)
_PANEL_KEY = re.compile(r"^panel[_\s-]*([A-Za-z][A-Za-z0-9]*)", re.IGNORECASE)
_FILE_PANEL_PATHS = {
    "Fig1_behavior_spatiotemporal_control_stats.json": {
        "A": [("Example Traces", ("direct_control", "behavior_example_traces"))],
        "B": [("Direct-Control Model", ("direct_control", "direct_control_schematic"))],
        "C": [("Perceptual Space Relation", ("direct_control", "probe_model", "component_plus_residual_panels", "C"))],
        "D": [("Perceptual Time Relation", ("direct_control", "probe_model", "component_plus_residual_panels", "D"))],
        "E": [("Motor Space Relation", ("direct_control", "motor_model", "component_plus_residual_panels", "E"))],
        "F": [("Motor Time Relation", ("direct_control", "motor_model", "component_plus_residual_panels", "F"))],
        "G": [("Perceptual Direct Gains", ("direct_control", "probe_model", "direct_gain_statistics"))],
        "H": [("Motor Direct Gains", ("direct_control", "motor_model", "direct_gain_statistics"))],
        "I": [("Double Dissociation", ("direct_control", "double_dissociation", "statistics"))],
        "J": [("Cross-Dimensional Model", ("cross_dimensional_control", "model"))],
        "K": [("Perceptual Time To Space", ("cross_dimensional_control", "perception_model", "component_plus_residual_panels", "B"))],
        "L": [("Perceptual Space To Time", ("cross_dimensional_control", "perception_model", "component_plus_residual_panels", "C"))],
        "M": [("Perceptual Cross Gains", ("cross_dimensional_control", "perception_model", "interaction_gain_statistics"))],
        "N": [("Motor Time To Space", ("cross_dimensional_control", "motor_model", "component_plus_residual_panels", "E"))],
        "O": [("Motor Space To Time", ("cross_dimensional_control", "motor_model", "component_plus_residual_panels", "F"))],
        "P": [("Motor Cross Gains", ("cross_dimensional_control", "motor_model", "interaction_gain_statistics"))],
        "Q": [("Interaction-Direction Contrast", ("cross_dimensional_control", "interaction_direction_comparison", "statistics"))],
    },
    "SuppFig2_Fig1_EIV_reliability_stats.json": {
        "A": [("Reliability", ("reliability",))],
        "B": [("Naive Gains", ("gains_naive",)), ("Corrected Gains", ("gains_corrected",))],
        "C": [("Delta M Dissociation", ("delta_M_dissociation",))],
        "D": [("Interaction Asymmetry", ("interaction_asymmetry",))],
    },
    "SuppFig1_Fig1_raw_normal_behavior_stats.json": {
        "A": [("Raw Probe Space Perception", ("panels", "A"))],
        "B": [("Raw Probe Time Perception", ("panels", "B"))],
        "C": [("Raw Normal Space Behavior", ("panels", "C"))],
        "D": [("Raw Normal Time Behavior", ("panels", "D"))],
    },
    "SuppFig3_Fig1_behavior_variability_stats.json": {
        "A": [("Probe Spatial Variability", ("panels", "A"))],
        "B": [("Probe Temporal Variability", ("panels", "B"))],
        "C": [("Perceptual Variability Slopes", ("panels", "C"))],
        "D": [("Motor Spatial Variability", ("panels", "D"))],
        "E": [("Motor Temporal Variability", ("panels", "E"))],
        "F": [("Motor Variability Slopes", ("panels", "F"))],
        "G": [("Motor Time To Spatial Variability", ("panels", "G"))],
        "H": [("Motor Space To Temporal Variability", ("panels", "H"))],
        "I": [("Motor Cross-Variability Slopes", ("panels", "I"))],
    },
    "Fig2_behavior_computational_models_stats.json": {
        "A": [("Radius-Speed Allocation Account", ("panels", "A"))],
        "B": [("Dynamic-Speed Compression", ("panels", "B"))],
        "C": [("Held-Out Setpoint Models", ("panels", "C"))],
        "D": [("Directional Allocation", ("panels", "D"))],
        "E": [("Control-Matrix Reconstruction", ("panels", "E"))],
        "F": [("AM1 Quadratic Effort", ("panels", "F_AM1"))],
        "G": [("AM2 Proportional Noise", ("panels", "G_AM2"))],
        "H": [("AM3 Serial Dependence", ("panels", "H_AM3"))],
        "I": [("AM4 Constant-K Two-Thirds Law", ("panels", "I_AM4"))],
    },
    "SuppFig4_Fig2_behavior_model_controls_stats.json": {
        "A": [("Temporal Offset", ("panels", "A_temporal_offset"))],
        "B": [("Spatial Offset", ("panels", "B_spatial_offset"))],
        "C": [("Speed-Metric Robustness", ("panels", "C_speed_metric_robustness"))],
        "D": [("Allocation-Parameter Recovery", ("panels", "D_rho_recovery"))],
    },
    "SuppFig8_Fig3_RNN_extrapolation_local_gain_stats.json": {
        "A": [("Space Trained Range", ("trained_range", "R"))],
        "B": [("Space Local Gain", ("space_local_gain",))],
        "C": [("Time Trained Range", ("trained_range", "T"))],
        "D": [("Time Local Gain", ("time_local_gain",))],
    },
    "SuppFig6_Fig3_RNN_parameter_effects_fullrank_stats.json": {
        "A": [("Speed-Coded Input Overlap", ("panel_statistics", "speedcoded_input_overlap"))],
        "B": [("Speed-Coded Readout", ("panel_statistics", "speedcoded_readout"))],
        "C": [("Duration-Coded Input Overlap", ("panel_statistics", "durationcoded_input_overlap"))],
        "D": [("Duration-Coded Readout", ("panel_statistics", "durationcoded_readout"))],
    },
    "SuppFig5_Fig3_RNN_sweep_duration_coded_stats.json": {
        "A": [("Architecture Configuration", ("time_code",))],
        "B": [("Duration-Coded Sweep", ("sweep_durationcoded_panel",))],
    },
    "Fig4_input_generator_rank_audit_stats.json": {
        "A": [("Excursion Components", ("panels", "excursion_components"))],
        "B": [("Progress-Slowing Components", ("panels", "progress_slowing_components"))],
    },
    "alternative_behavior_model_legacy_stats.json": {
        "A": [
            ("M1 Model", ("M1_simple_proportional_noise",)),
            ("M1 Matrix Alternative", ("matrix_level_alternatives", "M1_simple_proportional_noise")),
        ],
        "B": [
            ("M2 Timing-Constrained Manifold", ("M2_specified_quadratic_effort_timing_constrained_manifold",)),
            ("M2 Matrix LOSO", ("M2_specified_quadratic_effort_matrix_LOSO",)),
            ("M2 Matrix Alternative", ("matrix_level_alternatives", "M2_specified_quadratic_effort_best")),
        ],
        "C": [
            ("M3 Model", ("M3_constant_K_spontaneous_two_thirds",)),
            ("M3 Matrix Alternative", ("matrix_level_alternatives", "M3_constant_K_spontaneous_two_thirds")),
        ],
    },
}


def _humanize(key: Any) -> str:
    text = str(key).strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return "Value"
    if text.startswith("beta "):
        return "Beta " + text[5:]
    return text[0].upper() + text[1:]


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_))


def _format_number(value: float, key: str = "") -> str:
    number = float(value)
    if math.isnan(number):
        return "NA"
    if math.isinf(number):
        return "Inf" if number > 0 else "-Inf"
    if number == 0:
        return "0"
    absolute = abs(number)
    key_lower = key.lower()
    if "p" in key_lower and absolute < 0.001:
        return f"{number:.3e}"
    if absolute < 0.001 or absolute >= 10000:
        return f"{number:.4e}"
    if absolute >= 100:
        return f"{number:.2f}"
    if absolute >= 10:
        return f"{number:.3f}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _format_scalar(value: Any, key: str = "") -> str:
    if value is None:
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_number(float(value), key)
    return str(value)


def _small_scalar_sequence(value: Any, max_items: int = 8) -> list | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    items = list(value)
    if len(items) > max_items or not all(_is_scalar(item) for item in items):
        return None
    return items


def _numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _format_sequence(value: Sequence, key: str) -> str:
    return ", ".join(_format_scalar(item, key) for item in value)


def _is_detail_key(key: Any, value: Any) -> bool:
    lower = str(key).lower()
    if lower in _DETAIL_KEYS or lower.endswith(_DETAIL_SUFFIXES):
        return True
    if lower.endswith("_values"):
        try:
            return len(value) > 8
        except TypeError:
            return False
    return False


def _detail_description(value: Any) -> str:
    if isinstance(value, Mapping):
        return f"Mapping with {len(value)} entries"
    array = _numeric_array(value)
    if array is not None:
        shape = " x ".join(str(part) for part in array.shape)
        return f"Numeric array, shape {shape}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return f"Sequence with {len(value)} entries"
    return "Stored in JSON"


def _direct_cells(mapping: Mapping) -> dict[str, str]:
    cells = {}
    for key, value in mapping.items():
        if _is_detail_key(key, value):
            continue
        if _is_scalar(value):
            cells[str(key)] = _format_scalar(value, str(key))
            continue
        small = _small_scalar_sequence(value)
        if small is not None:
            cells[str(key)] = _format_sequence(small, str(key))
    return cells


def _is_leaf_mapping(mapping: Mapping) -> bool:
    for key, value in mapping.items():
        if _is_detail_key(key, value):
            continue
        if isinstance(value, Mapping):
            return False
    return bool(_direct_cells(mapping))


def _append_table(lines: list[str], headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        return
    lines.append("| " + " | ".join(_escape(header) for header in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_escape(cell) for cell in row) + " |")
    lines.append("")


def _append_key_value_table(lines: list[str], mapping: Mapping) -> None:
    rows = [[_humanize(key), value] for key, value in _direct_cells(mapping).items()]
    _append_table(lines, ["Measure", "Value"], rows)


def _append_mapping_table(lines: list[str], mapping: Mapping) -> set[str]:
    leaf_items = {
        str(key): value
        for key, value in mapping.items()
        if isinstance(value, Mapping) and _is_leaf_mapping(value) and not _is_detail_key(key, value)
    }
    if len(leaf_items) < 2:
        return set()
    columns = []
    for value in leaf_items.values():
        for column in _direct_cells(value):
            if column not in columns:
                columns.append(column)
    if not columns or len(columns) > 14:
        return set()
    rows = []
    for key, value in leaf_items.items():
        cells = _direct_cells(value)
        rows.append([_humanize(key)] + [cells.get(column, "") for column in columns])
    _append_table(lines, ["Measure"] + [_humanize(column) for column in columns], rows)
    return set(leaf_items)


def _matrix_payload(value: Any) -> np.ndarray | None:
    array = _numeric_array(value)
    if array is None or array.ndim != 2:
        return None
    if array.shape[0] > 10 or array.shape[1] > 10:
        return None
    return array


def _append_matrix(
    lines: list[str],
    title: str,
    matrix: np.ndarray,
    row_labels: Sequence | None = None,
    column_labels: Sequence | None = None,
    heading_level: int = 4,
) -> None:
    n_rows, n_columns = matrix.shape
    rows = list(row_labels) if row_labels is not None and len(row_labels) == n_rows else list(range(1, n_rows + 1))
    columns = (
        list(column_labels)
        if column_labels is not None and len(column_labels) == n_columns
        else list(range(1, n_columns + 1))
    )
    lines.append(f"{'#' * min(heading_level, 6)} {_humanize(title)}")
    lines.append("")
    table_rows = [
        [_format_scalar(rows[index])] + [_format_number(item, title) for item in matrix[index]]
        for index in range(n_rows)
    ]
    _append_table(lines, ["Row"] + [_format_scalar(item) for item in columns], table_rows)


def _append_mapping(
    lines: list[str],
    mapping: Mapping,
    heading_level: int,
    omitted: list[tuple[str, str]],
    path: str,
) -> None:
    local_details = []
    _append_key_value_table(lines, mapping)
    tabulated = _append_mapping_table(lines, mapping)

    matrix_keys = set()
    row_labels = mapping.get("row_labels") if isinstance(mapping.get("row_labels"), Sequence) else None
    column_labels = mapping.get("col_labels") if isinstance(mapping.get("col_labels"), Sequence) else None
    for key, value in mapping.items():
        matrix = _matrix_payload(value)
        if matrix is None:
            continue
        _append_matrix(lines, str(key), matrix, row_labels, column_labels, heading_level + 1)
        matrix_keys.add(str(key))

    for key, value in mapping.items():
        key_text = str(key)
        field_path = f"{path}.{key_text}" if path else key_text
        if key_text in matrix_keys or key_text in {"row_labels", "col_labels"}:
            continue
        if _is_detail_key(key, value):
            local_details.append((_humanize(key), _detail_description(value)))
            continue
        if isinstance(value, Mapping):
            if key_text in tabulated:
                continue
            lines.append(f"{'#' * min(heading_level, 6)} {_humanize(key)}")
            lines.append("")
            _append_mapping(lines, value, heading_level + 1, omitted, field_path)
            continue
        if _is_scalar(value) or _small_scalar_sequence(value) is not None:
            continue
        local_details.append((_humanize(key), _detail_description(value)))

    if local_details:
        _append_table(lines, ["Detailed Data", "Stored Data"], local_details)


def _resolve_path(payload: Mapping, path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _panel_entries(source_json: Path, payload: Mapping) -> tuple[list[tuple[str, Any]], set[str]]:
    configured = _FILE_PANEL_PATHS.get(source_json.name)
    if configured:
        entries = []
        consumed = set()
        for label, fields in configured.items():
            panel = {}
            for title, path in fields:
                value = _resolve_path(payload, path)
                if value is None:
                    continue
                panel[title] = value
                if path:
                    consumed.add(path[0])
            entries.append((label, panel))
        return entries, consumed

    panels = payload.get("panels")
    if isinstance(panels, Mapping):
        return [(str(label), value) for label, value in panels.items()], {"panels"}

    entries = []
    consumed = set()
    for key, value in payload.items():
        if not isinstance(value, Mapping) or "panel" not in str(key).lower():
            continue
        match = _PANEL_KEY.match(str(key))
        if match:
            entries.append((match.group(1), value))
            consumed.add(str(key))
            continue
        child_panels = []
        for child_key, child_value in value.items():
            child_match = re.match(r"^([A-Za-z])(?:[_\s-]|$)", str(child_key))
            if child_match:
                child_panels.append((child_match.group(1).upper(), child_value))
        if child_panels:
            entries.extend(child_panels)
            consumed.add(str(key))
    return entries, consumed


def _report_title(source_json: Path) -> str:
    stem = source_json.stem
    if stem.endswith("_stats"):
        stem = stem[:-6]
    supp_match = re.match(r"^SuppFig(\d+)[_-]?(.*)$", stem, re.IGNORECASE)
    if supp_match:
        suffix_raw = supp_match.group(2)
        parent_match = re.match(r"^Fig(\d+)[_-]?(.*)$", suffix_raw, re.IGNORECASE)
        if parent_match:
            suffix = _humanize(parent_match.group(2)) if parent_match.group(2) else ""
            parent = f" (Figure {parent_match.group(1)})"
        else:
            suffix = _humanize(suffix_raw) if suffix_raw else ""
            parent = ""
        return f"Supplementary Figure {supp_match.group(1)}{parent}{': ' + suffix if suffix else ''} Statistics"
    fig_match = re.match(r"^Fig(\d+)[_-]?(.*)$", stem, re.IGNORECASE)
    if fig_match:
        suffix = _humanize(fig_match.group(2)) if fig_match.group(2) else ""
        return f"Figure {fig_match.group(1)}{': ' + suffix if suffix else ''} Statistics"
    return _humanize(stem) + " Statistics"


def render_stats_markdown(source_json: Path, payload: Mapping) -> str:
    lines = [f"# {_report_title(source_json)}", ""]
    lines.append(
        f"> Auto-generated from `{source_json.name}`. Do not edit this file manually; rerun the analysis script instead."
    )
    lines.append("")

    description = payload.get("description")
    if description:
        lines.extend([_escape(description), ""])

    overview = {
        key: value
        for key, value in payload.items()
        if key != "description" and _is_scalar(value)
    }
    if overview:
        lines.extend(["## Analysis Overview", ""])
        _append_key_value_table(lines, overview)

    omitted: list[tuple[str, str]] = []
    panel_entries, panel_sources = _panel_entries(source_json, payload)
    if panel_entries:
        lines.extend(["## Panel Statistics", ""])
        seen_labels: dict[str, int] = {}
        for label, value in panel_entries:
            seen_labels[label] = seen_labels.get(label, 0) + 1
            suffix = f" ({seen_labels[label]})" if seen_labels[label] > 1 else ""
            lines.extend([f"### Panel {label}{suffix}", ""])
            if isinstance(value, Mapping):
                _append_mapping(lines, value, 4, omitted, f"panels.{label}")
            else:
                _append_key_value_table(lines, {"value": value})

    excluded = {"description", "panels", *panel_sources}
    additional = [
        (key, value)
        for key, value in payload.items()
        if key not in excluded and not _is_scalar(value)
    ]
    if additional:
        lines.extend(["## Figure-Level Statistics", ""])
        for key, value in additional:
            field_path = str(key)
            if _is_detail_key(key, value):
                omitted.append((field_path, _detail_description(value)))
                continue
            lines.extend([f"### {_humanize(key)}", ""])
            if isinstance(value, Mapping):
                _append_mapping(lines, value, 4, omitted, field_path)
                continue
            small = _small_scalar_sequence(value)
            if small is not None:
                _append_key_value_table(lines, {key: small})
            else:
                omitted.append((field_path, _detail_description(value)))

    if omitted:
        unique = []
        seen = set()
        for field, detail in omitted:
            if field in seen:
                continue
            seen.add(field)
            unique.append([field, detail])
        lines.extend(["## Detailed Data Retained In JSON", ""])
        lines.append(
            "The following high-volume fields are intentionally omitted from this readable report and remain available in the JSON source."
        )
        lines.append("")
        _append_table(lines, ["JSON Field", "Stored Data"], unique)

    return "\n".join(lines).rstrip() + "\n"


def write_stats_markdown(source_json: Path, payload: Mapping) -> Path:
    markdown_path = source_json.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_stats_markdown(source_json, payload), encoding="utf-8")
    return markdown_path
