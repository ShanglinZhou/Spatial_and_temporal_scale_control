#!/usr/bin/env python3
"""Regenerate paper Figure 2 and Supplementary Figure 4."""

from src.release_runtime import compare_output_stats, configure, run


if __name__ == "__main__":
    stats_files = (
        "Fig2_behavior_computational_models_stats.json",
        "SuppFig4_Fig2_behavior_model_controls_stats.json",
    )
    output_dir = configure("fig2", plot_only=True, allow_output_stats=True)
    run("figure2_impl")
    compare_output_stats(output_dir, stats_files)
