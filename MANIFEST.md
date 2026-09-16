# Release manifest

Version: `0.2.0`

## Included

- Four thin, directly executable root-level plotting/demonstration entries.
- A single copy of each implementation under `src/`, including the core RNN
  training code.
- Twenty-four anonymized cleaned human-participant MAT files.
- Ten locked statistics JSON files used to verify or supply the executable
  figures.
- Compact behavioral-model and rotated-Hopf caches.
- One representative full-rank RNN checkpoint.
- Exact environment files, MIT License, documentation, and QA tools.

## Deliberately excluded

- Plot scripts that cannot run without omitted large intermediate arrays.
- Approximately 398 GB of redundant multi-seed checkpoints, recurrent-rate
  payloads, perturbation arrays, and parameter-grid products.
- Previously published third-party primate data.
- Manuscript drafts, Illustrator files, temporary figures, and Python caches.

Generated figures are ignored under `outputs/` and can be recreated with
`python plot_all.py`.
