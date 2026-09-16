#!/usr/bin/env python3
"""Regenerate paper Figure 8 from the versioned 20-repetition fit cache."""

from src.release_runtime import configure, run


if __name__ == "__main__":
    configure(
        "fig8",
        reference_stats=(
            "Fig1_behavior_spatiotemporal_control_stats.json",
            "Fig4_pca_rnn_mechanism_stats.json",
            "Fig6_main_PCA_radial_axis_alignment_stats.json",
            "ParamSweep_rnn_behavior_summary_stats.json",
        ),
        plot_only=True,
        allow_output_stats=True,
    )
    run("figure8_impl", argv=("--plot-only",))
