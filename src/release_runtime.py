"""Shared setup for the root-level plot scripts."""

from __future__ import annotations

import importlib
import json
import math
import numbers
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REFERENCE_STATS = ROOT / "reference_data" / "stats"


def configure(
    output_name: str,
    reference_stats: Iterable[str] = (),
    plot_only: bool = True,
    allow_output_stats: bool = False,
) -> Path:
    """Configure an isolated output directory before analysis imports occur."""
    output_dir = ROOT / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in reference_stats:
        source = REFERENCE_STATS / filename
        if not source.exists():
            raise FileNotFoundError("Missing reference input: {}".format(source))
        shutil.copy2(str(source), str(output_dir / filename))
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MANUSCRIPT_FIG_DIR"] = str(output_dir)
    os.environ["MANUSCRIPT_PLOT_ONLY"] = "1" if plot_only else "0"
    os.environ["MANUSCRIPT_PLOT_ONLY_STAGING"] = "1" if allow_output_stats else "0"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return output_dir


def run(module_name: str, argv: Optional[Sequence[str]] = None) -> None:
    """Import an internal figure implementation and call its main function."""
    module = importlib.import_module(module_name)
    if argv is None:
        module.main()
    else:
        module.main(list(argv))


def _compare_values(expected, actual, path, result, rtol, atol) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            result["structural_differences"].append("{}.{} missing from generated output".format(path, key))
        for key in sorted(actual_keys - expected_keys):
            result["structural_differences"].append("{}.{} absent from reference".format(path, key))
        for key in sorted(expected_keys & actual_keys):
            _compare_values(expected[key], actual[key], "{}.{}".format(path, key), result, rtol, atol)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            result["structural_differences"].append(
                "{} length differs: reference={}, generated={}".format(path, len(expected), len(actual))
            )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare_values(expected_item, actual_item, "{}[{}]".format(path, index), result, rtol, atol)
        return
    if (
        isinstance(expected, numbers.Real)
        and not isinstance(expected, bool)
        and isinstance(actual, numbers.Real)
        and not isinstance(actual, bool)
    ):
        expected_float = float(expected)
        actual_float = float(actual)
        result["numeric_values_compared"] += 1
        if math.isnan(expected_float) and math.isnan(actual_float):
            return
        absolute = abs(actual_float - expected_float)
        relative = absolute / max(abs(expected_float), abs(actual_float), atol)
        result["maximum_absolute_difference"] = max(result["maximum_absolute_difference"], absolute)
        result["maximum_relative_difference"] = max(result["maximum_relative_difference"], relative)
        if not math.isclose(expected_float, actual_float, rel_tol=rtol, abs_tol=atol):
            if len(result["values_outside_tolerance"]) < 50:
                result["values_outside_tolerance"].append(
                    {
                        "path": path,
                        "reference": expected_float,
                        "generated": actual_float,
                        "absolute_difference": absolute,
                    }
                )
        return
    if expected != actual and len(result["non_numeric_differences"]) < 50:
        result["non_numeric_differences"].append(
            {"path": path, "reference": expected, "generated": actual}
        )


def compare_output_stats(
    output_dir: Path,
    filenames: Iterable[str],
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Path:
    """Write a non-blocking comparison of regenerated and paper-reference JSON."""
    report = {
        "description": "Non-blocking comparison against locked paper statistics.",
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "files": [],
    }
    for filename in filenames:
        reference_path = REFERENCE_STATS / filename
        generated_path = output_dir / filename
        result = {
            "filename": filename,
            "exact_match": False,
            "within_numeric_tolerance": False,
            "numeric_values_compared": 0,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "structural_differences": [],
            "non_numeric_differences": [],
            "values_outside_tolerance": [],
        }
        if not reference_path.exists() or not generated_path.exists():
            result["structural_differences"].append("Reference or generated JSON is missing")
        else:
            expected = json.loads(reference_path.read_text(encoding="utf-8"))
            actual = json.loads(generated_path.read_text(encoding="utf-8"))
            result["exact_match"] = expected == actual
            _compare_values(expected, actual, "$", result, rtol, atol)
            result["within_numeric_tolerance"] = not (
                result["structural_differences"]
                or result["non_numeric_differences"]
                or result["values_outside_tolerance"]
            )
        report["files"].append(result)
    report["all_exact"] = all(item["exact_match"] for item in report["files"])
    report["all_within_numeric_tolerance"] = all(
        item["within_numeric_tolerance"] for item in report["files"]
    )
    destination = output_dir / "reference_comparison.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["all_exact"]:
        print("Regenerated statistics exactly match the locked paper references.")
    elif report["all_within_numeric_tolerance"]:
        warnings.warn(
            "Regenerated statistics differ only within numerical tolerance; see {}".format(destination),
            RuntimeWarning,
        )
    else:
        warnings.warn(
            "Regenerated statistics differ from the locked references. Figures were saved; "
            "inspect {} for details.".format(destination),
            RuntimeWarning,
        )
    print("Saved reference comparison to {}".format(destination))
    return destination
