# Anonymized cleaned human data

This directory contains 24 cleaned participant-level MATLAB files. Private
working filenames were replaced by `sub-01` through `sub-24`; no reidentifying
key is included.

Each file contains aligned arrays used by the Python analysis:

- `analysis_numeric`: day, trial number, target duration, target radius,
  response duration, and response radius;
- `analysis_trialtype`: Normal or Probe trial label;
- `analysis_stimulus_order`: stimulus-order label;
- `analysis_probe_order`: Probe response-order label;
- `analysis_draw`: sampled drawing trajectory;
- `cleaning_summary`: counts at the main cleaning stages.

The files are loaded by `src/analysis_all_common.py`. Run
`python plot_rnn_demo.py` to verify that all 24 files can be read and
summarized.

These data are provided to verify the analyses reported in the accompanying
manuscript. Users must comply with applicable ethical approval, consent, and
institutional requirements when reusing human-participant data.

