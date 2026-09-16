#!/usr/bin/env python3
"""Run every directly reproducible plot in this release."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "plot_fig1.py",
    "plot_fig2.py",
    "plot_fig8.py",
    "plot_rnn_demo.py",
)


def main() -> None:
    for script in SCRIPTS:
        print("\n=== Running {} ===".format(script), flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT), check=True)
    print("\nAll figures were generated under outputs/.")


if __name__ == "__main__":
    main()
