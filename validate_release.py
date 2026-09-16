#!/usr/bin/env python3
"""Static QA for the lightweight research-code release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WINDOWS_ABSOLUTE = re.compile(r"\b[A-Za-z]:\\(?![nrt\"'])")
LEGACY_SUBJECT = re.compile(r"(?i)\b\d{2}_[A-Za-z]+_clean")
TEXT_SUFFIXES = {".py", ".m", ".md", ".json", ".txt", ".yml", ".yaml"}
REQUIRED = [
    "README.md",
    "LICENSE",
    "VERSION",
    "environment.yml",
    "requirements.txt",
    "MANIFEST.md",
    "reference_outputs/metrics/file_checksums.json",
    "src/model.py",
    "src/PARAM.py",
    "models/representative_rnn_rep15.pt",
    "docs/FIGURE_CODE_MAP.md",
    "plot_fig1.py",
    "plot_fig2.py",
    "plot_fig8.py",
    "plot_rnn_demo.py",
    "plot_all.py",
]

IGNORED_ROOTS = {"outputs", ".git", ".venv", "env", ".conda", ".idea", ".vscode"}
IGNORED_NAMES = {".DS_Store"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}


def is_ignored_path(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return False
    if not relative.parts:
        return False
    return bool(
        relative.parts[0] in IGNORED_ROOTS
        or "__pycache__" in relative.parts
        or path.name in IGNORED_NAMES
        or path.suffix.lower() in IGNORED_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()
    started = time.time()
    errors = []

    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append("Missing required file: {}".format(relative))

    participants = sorted((ROOT / "human_data" / "clean_data").glob("*.mat"))
    if len(participants) != 24:
        errors.append("Expected 24 anonymized MAT files; found {}".format(len(participants)))
    for path in participants:
        if not re.match(r"^sub-\d{2}_clean\.mat$", path.name):
            errors.append("Non-anonymous participant filename: {}".format(path.name))

    python_files = sorted(path for path in ROOT.rglob("*.py") if not is_ignored_path(path))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            errors.append("Python syntax error in {}: {!r}".format(path.relative_to(ROOT), exc))

    json_files = sorted(path for path in ROOT.rglob("*.json") if not is_ignored_path(path))
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append("Invalid JSON in {}: {!r}".format(path.relative_to(ROOT), exc))

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or is_ignored_path(path) or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in {"validation_report.json", "file_checksums.json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if WINDOWS_ABSOLUTE.search(text):
            errors.append("Absolute Windows path remains in {}".format(path.relative_to(ROOT)))
        if LEGACY_SUBJECT.search(text):
            errors.append("Legacy participant identifier remains in {}".format(path.relative_to(ROOT)))

    oversized = [
        path for path in ROOT.rglob("*")
        if path.is_file() and not is_ignored_path(path) and path.stat().st_size > 50 * 1024 * 1024
    ]
    for path in oversized:
        errors.append("File exceeds 50 MB: {}".format(path.relative_to(ROOT)))

    stats_json = list((ROOT / "reference_data" / "stats").glob("*_stats.json"))
    if len(stats_json) != 10:
        errors.append("Expected 10 directly used reference statistics JSON files; found {}".format(len(stats_json)))

    checksum_exclusions = {
        "reference_outputs/metrics/file_checksums.json",
        "reference_outputs/metrics/validation_report.json",
    }
    checksum_files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        if not path.is_file() or is_ignored_path(path) or relative in checksum_exclusions:
            continue
        checksum_files[relative] = path

    destination = ROOT / "reference_outputs" / "metrics" / "file_checksums.json"
    if args.write_checksums:
        checksums = {relative: sha256(path) for relative, path in checksum_files.items()}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    else:
        try:
            recorded_checksums = json.loads(destination.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append("Cannot read checksum manifest: {!r}".format(exc))
            recorded_checksums = {}
        recorded_names = set(recorded_checksums)
        current_names = set(checksum_files)
        for relative in sorted(recorded_names - current_names):
            errors.append("Checksum target is missing: {}".format(relative))
        for relative in sorted(current_names - recorded_names):
            errors.append("File is absent from checksum manifest: {}".format(relative))
        for relative in sorted(recorded_names & current_names):
            if sha256(checksum_files[relative]) != recorded_checksums[relative]:
                errors.append("Checksum mismatch: {}".format(relative))

    report = {
        "valid": not errors,
        "errors": errors,
        "python_files": len(python_files),
        "json_files": len(json_files),
        "reference_stats_json": len(stats_json),
        "participant_files": len(participants),
        "total_files": sum(1 for path in ROOT.rglob("*") if path.is_file() and not is_ignored_path(path)),
        "total_bytes": sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file() and not is_ignored_path(path)),
        "runtime_seconds": time.time() - started,
    }
    report_path = ROOT / "reference_outputs" / "metrics" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
