# Reproducibility guide

## One-command reproduction

From a newly created environment and the repository root, run:

```bash
python validate_release.py
python plot_all.py
```

All generated files appear below `outputs/`. The versioned human data,
reference statistics, caches, and checkpoint are read-only inputs.

## Verification behavior

- `plot_fig1.py` and `plot_fig2.py` recalculate statistics and write a
  `reference_comparison.json` report beside each figure set. Differences are
  recorded without blocking figure export, so environments with small
  floating-point or statistical-metadata differences remain usable. Wilcoxon
  tests explicitly request the normal approximation used by the paper's
  SciPy 1.3 environment, preventing version-dependent switches to exact tests.
- `plot_fig8.py` validates and loads the complete 20-repetition fit cache. It
  permits NumPy/SciPy version labels to differ, but still strictly checks the
  fit code, parameterization, objective settings, and target matrices. It
  stops rather than beginning an unrequested long refit.
- `plot_rnn_demo.py` restores one trained RNN and evaluates five conditions.
  It is a functional test, not a replacement for population inference.

## Randomness

The RNN training base seed is `20260626`; repetition index is added to it. The
RNN demonstration uses seed `20260641`. Figure-specific resampling and model
fit seeds are fixed in the corresponding internal implementation and recorded
in emitted JSON where applicable.

## Full multi-seed computation

`src/PARAM.py`, `src/model.py`, `src/tasks.py`, and `src/utils.py` document the
custom RNN training pipeline. Rebuilding every paper RNN panel requires a large
multi-seed sweep and hundreds of gigabytes of generated products, so it is
outside this compact release workflow.
