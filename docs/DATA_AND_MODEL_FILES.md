# Data and model files

## Human data

`human_data/clean_data/` contains 24 anonymized cleaned MATLAB files totaling
approximately 12 MB. They retain the trial-level responses and drawing
trajectories used by Figures 1, 2, and 8.

## Behavioral and Hopf caches

`src/results/behavior_*` contains compact, versioned inputs required by Figure
2. `src/results/hopf_model_3d/` contains the complete 20-repetition fitted
parameter cache required by Figure 8.

## Reference statistics

`reference_data/stats/` contains only the 10 paper-statistics JSON files used
by the four public plot entries. Files for omitted, non-executable plots are
not included.

## Representative RNN

`models/representative_rnn_rep15.pt` is a canonical full-rank, free-readout,
speed-coded network with input overlap -0.8. It supports the direct RNN demo;
it is not used as a substitute for the paper's 20-repetition inference.

## Omitted products

Full RNN recurrent-rate arrays can approach 1 GB per repetition. Redundant
checkpoints, full evaluation grids, perturbation arrays, and third-party
primate datasets are not redistributed in this compact release.
