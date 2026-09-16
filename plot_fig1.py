#!/usr/bin/env python3
"""Regenerate paper Figure 1 and Supplementary Figures 1 and 3."""

from src.release_runtime import compare_output_stats, configure, run


if __name__ == "__main__":
    stats_files = (
        "Fig1_behavior_spatiotemporal_control_stats.json",
        "SuppFig1_Fig1_raw_normal_behavior_stats.json",
        "SuppFig3_Fig1_behavior_variability_stats.json",
    )
    output_dir = configure("fig1", plot_only=True, allow_output_stats=True)
    run("figure1_impl")
    compare_output_stats(output_dir, stats_files)
