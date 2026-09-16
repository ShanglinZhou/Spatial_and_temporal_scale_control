"""Preserve formula typography while making SVG text Illustrator-friendly."""

from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from lxml import etree as LET
from matplotlib.figure import Figure
from matplotlib.text import Text


_SVG_NS = "http://www.w3.org/2000/svg"
_MATH_COMMANDS = {
    "alpha": ("α", "italic"),
    "beta": ("β", "italic"),
    "gamma": ("γ", "italic"),
    "delta": ("δ", "italic"),
    "epsilon": ("ε", "italic"),
    "lambda": ("λ", "italic"),
    "mu": ("μ", "italic"),
    "omega": ("ω", "italic"),
    "phi": ("ϕ", "italic"),
    "pi": ("π", "italic"),
    "rho": ("ρ", "italic"),
    "sigma": ("σ", "italic"),
    "tau": ("τ", "italic"),
    "theta": ("θ", "italic"),
    "Delta": ("Δ", "italic"),
    "approx": ("≈", "normal"),
    "cdot": ("·", "normal"),
    "cdots": ("⋯", "normal"),
    "circ": ("°", "normal"),
    "cos": ("cos", "normal"),
    "leftarrow": ("←", "normal"),
    "leq": ("≤", "normal"),
    "log": ("log", "normal"),
    "max": ("max", "normal"),
    "min": ("min", "normal"),
    "arg": ("arg", "normal"),
    "sin": ("sin", "normal"),
    "sum": ("∑", "normal"),
    "times": ("×", "normal"),
    "pm": ("±", "normal"),
    "rightarrow": ("→", "normal"),
    "to": ("→", "normal"),
}

_ACCENT_COMMANDS = {
    "bar": "\u0304",
    "dot": "\u0307",
    "hat": "\u0302",
    "tilde": "\u0303",
}


FormulaRun = Tuple[str, str, Optional[str], str]


def tag_formula_texts(fig: Figure, prefix: str) -> Dict[str, Dict[str, object]]:
    """Assign SVG group IDs to all visible Matplotlib text containing MathText."""
    fig.canvas.draw()
    formula_specs: Dict[str, Dict[str, object]] = {}
    index = 0
    for artist in fig.findobj(match=Text):
        source = artist.get_text()
        if not artist.get_visible() or "$" not in source:
            continue
        requested_gid = artist.get_gid()
        gid = str(requested_gid) if requested_gid else f"{prefix}_{index:03d}"
        if gid in formula_specs:
            gid = f"{prefix}_{index:03d}"
        artist.set_gid(gid)
        formula_specs[gid] = {
            "source": source,
            "font_size": float(artist.get_fontsize()),
        }
        index += 1
    return formula_specs


def _append_svg_run(
    runs: List[FormulaRun],
    text: str,
    style: str = "normal",
    script: Optional[str] = None,
    weight: str = "normal",
) -> None:
    if not text:
        return
    text = text.replace(" ", "\xa0")
    if runs and runs[-1][1:] == (style, script, weight):
        runs[-1] = (runs[-1][0] + text, style, script, weight)
    else:
        runs.append((text, style, script, weight))


def _consume_tex_group(source: str, start: int) -> Tuple[str, int]:
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


def _parse_math_runs(
    source: str,
    *,
    script: Optional[str] = None,
    forced_style: Optional[str] = None,
    forced_weight: Optional[str] = None,
) -> List[FormulaRun]:
    runs: List[FormulaRun] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char in "_^":
            group, index = _consume_tex_group(source, index + 1)
            target_script = "sub" if char == "_" else "super"
            for text, style, _, weight in _parse_math_runs(
                group,
                script=target_script,
                forced_style=forced_style,
                forced_weight=forced_weight,
            ):
                _append_svg_run(runs, text, style, target_script, weight)
            continue
        if char == "\\":
            if index + 1 < len(source) and source[index + 1] in {" ", ","}:
                _append_svg_run(runs, " ", "normal", script, forced_weight or "normal")
                index += 2
                continue
            match = re.match(r"\\([A-Za-z]+)", source[index:])
            if match is None:
                index += 1
                continue
            command = match.group(1)
            index += len(match.group(0))
            if command in {"mathrm", "mathbf", "mathsf", "operatorname"}:
                group, index = _consume_tex_group(source, index)
                group_style = "normal"
                group_weight = "bold" if command == "mathbf" else (forced_weight or "normal")
                for text, _, nested_script, weight in _parse_math_runs(
                    group,
                    script=script,
                    forced_style=group_style,
                    forced_weight=group_weight,
                ):
                    _append_svg_run(runs, text, group_style, nested_script, weight)
                continue
            if command in _ACCENT_COMMANDS:
                group, index = _consume_tex_group(source, index)
                for text, style, nested_script, weight in _parse_math_runs(
                    group,
                    script=script,
                    forced_style=forced_style,
                    forced_weight=forced_weight,
                ):
                    _append_svg_run(runs, text, style, nested_script, weight)
                if runs:
                    text, style, nested_script, weight = runs[-1]
                    runs[-1] = (
                        text + _ACCENT_COMMANDS[command],
                        style,
                        nested_script,
                        weight,
                    )
                continue
            if command == "sqrt":
                group, index = _consume_tex_group(source, index)
                _append_svg_run(runs, "√", "normal", script, forced_weight or "normal")
                for text, style, nested_script, weight in _parse_math_runs(
                    group,
                    script=script,
                    forced_style=forced_style,
                    forced_weight=forced_weight,
                ):
                    _append_svg_run(runs, text, style, nested_script, weight)
                continue
            if command == "quad":
                _append_svg_run(runs, "  ", "normal", script, forced_weight or "normal")
                continue
            text, style = _MATH_COMMANDS.get(command, (command, "normal"))
            _append_svg_run(
                runs,
                text,
                forced_style or style,
                script,
                forced_weight or "normal",
            )
            continue
        if char in "{}":
            index += 1
            continue
        if char == "-":
            char = "−"
        style = forced_style or ("italic" if char.isalpha() else "normal")
        _append_svg_run(runs, char, style, script, forced_weight or "normal")
        index += 1
    return runs


def _parse_mixed_svg_runs(line: str) -> List[FormulaRun]:
    runs: List[FormulaRun] = []
    for part in re.split(r"(\$[^$]*\$)", line):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            for text, style, script, weight in _parse_math_runs(part[1:-1]):
                _append_svg_run(runs, text, style, script, weight)
        else:
            _append_svg_run(runs, part, "normal", None, "normal")
    return runs


def _style_font_size(style: str) -> float:
    match = re.search(r"font-size:([0-9.]+)px", style or "")
    return float(match.group(1)) if match else float("nan")


def _rewrite_formula_text_element(
    text_element,
    source_line: str,
    requested_base_size: Optional[float] = None,
) -> None:
    tspans = list(text_element)
    if not tspans:
        raise ValueError(f"Expected MathText tspans for {source_line!r}")

    rendered_base_size = max(
        (_style_font_size(tspan.get("style", "")) for tspan in tspans),
        default=0.0,
    )
    base_size = (
        float(requested_base_size)
        if requested_base_size is not None and math.isfinite(float(requested_base_size))
        else rendered_base_size
    )
    x_values: List[float] = []
    base_candidates: List[float] = []
    for tspan in tspans:
        x_values.extend(float(value) for value in (tspan.get("x") or "").split())
        size = _style_font_size(tspan.get("style", ""))
        if math.isclose(size, rendered_base_size, rel_tol=1e-9, abs_tol=1e-9):
            base_candidates.extend(float(value) for value in (tspan.get("y") or "").split())
    if not x_values or not base_candidates:
        raise ValueError(f"Missing MathText coordinates for {source_line!r}")

    x_start = min(x_values)
    baseline = float(median(base_candidates))
    for child in list(text_element):
        text_element.remove(child)
    text_element.text = None

    for run_index, (text, font_style, script, font_weight) in enumerate(
        _parse_mixed_svg_runs(source_line)
    ):
        tspan = LET.SubElement(text_element, f"{{{_SVG_NS}}}tspan")
        size = base_size if script is None else 0.70 * base_size
        tspan.set(
            "style",
            f"font-family:Arial;font-size:{size:g}px;"
            f"font-style:{font_style};font-weight:{font_weight};",
        )
        if run_index == 0:
            tspan.set("x", f"{x_start:g}")
            tspan.set("y", f"{baseline:g}")
        if script is not None:
            tspan.set("baseline-shift", script)
        tspan.text = text
        tspan.tail = None


def rewrite_formula_text(svg_path: Path, formula_specs: Dict[str, object]) -> None:
    """Rewrite registered MathText as phrase-level SVG text with styled runs."""
    parser = LET.XMLParser(remove_blank_text=False, strip_cdata=False)
    tree = LET.parse(str(svg_path), parser)
    root = tree.getroot()
    for gid, spec in formula_specs.items():
        if isinstance(spec, dict):
            source = str(spec["source"])
            requested_base_size = float(spec["font_size"])
        else:
            source = str(spec)
            requested_base_size = None
        groups = root.xpath(f".//*[@id='{gid}']")
        if not groups:
            # Some layout-only or removed artists remain discoverable on the
            # Matplotlib figure but are not emitted by the SVG backend.
            continue
        if len(groups) != 1:
            raise ValueError(f"Expected one SVG group for {gid}, found {len(groups)}")
        text_elements = groups[0].xpath(".//*[local-name()='text']")
        source_lines: Sequence[str] = source.split("\n")
        if len(text_elements) != len(source_lines):
            raise ValueError(
                f"SVG line mismatch for {gid}: {len(text_elements)} elements, "
                f"{len(source_lines)} source lines"
            )
        for text_element, source_line in zip(text_elements, source_lines):
            if "$" in source_line:
                _rewrite_formula_text_element(
                    text_element,
                    source_line,
                    requested_base_size=requested_base_size,
                )

    doctype = tree.docinfo.doctype
    tree.write(
        str(svg_path),
        encoding="utf-8",
        xml_declaration=True,
        doctype=doctype,
        pretty_print=False,
    )
