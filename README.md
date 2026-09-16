# Code for Asymmetries in the control of spatial and temporal scaling

This is a compact, directly executable research-code release. Each public
`plot_*.py` file in the repository root produces a named paper figure or a
small verification figure. Generated files are written under `outputs/`;
versioned inputs are never overwritten.

## Installation

Tested on Windows 10/11 with Python 3.7.4, NumPy 1.16.5, SciPy 1.3.1,
pandas 0.25.1, PyTorch 1.13.1+cpu, and Matplotlib 3.1.1. The optional package
`lxml` 4.4.1 is used only for Illustrator-optimized SVG formula text. If it is
not installed, the scripts automatically save a standard editable SVG and all
PNG/statistical outputs normally. Install that optional feature with
`python -m pip install -r requirements-illustrator.txt`.

The direct plotting workflow has also been compatibility-tested with Python
3.9.23, NumPy 2.0.2, SciPy 1.13.1, pandas 2.3.3, PyTorch 2.8.0, and Matplotlib
3.9.4. The Figure 8 cache records these environment differences while still
requiring the scientific fit configuration and targets to match. Wilcoxon
tests explicitly use the paper environment's normal-approximation method so
that newer SciPy versions do not silently change the reported p-values.

```bash
conda env create -f environment.yml
conda activate spatial-temporal-scale-control
```

Environment creation typically takes 10-30 minutes. A compatible existing
Python 3.7 environment can instead use `python -m pip install -r requirements.txt`.

## Direct figure reproduction

Run these commands from the repository root:

```bash
python plot_fig1.py
python plot_fig2.py
python plot_fig8.py
python plot_rnn_demo.py
```

Or generate everything in sequence:

```bash
python plot_all.py
```

The complete sequence took about 70 seconds in the tested Windows CPU
environment.

| Command | Products | Typical CPU runtime |
|---|---|---:|
| `plot_fig1.py` | Fig. 1, Supp. Figs. 1 and 3 | 25 s |
| `plot_fig2.py` | Fig. 2 and Supp. Fig. 4 | 19 s |
| `plot_fig8.py` | Fig. 8 from 20 cached fits | 10 s |
| `plot_rnn_demo.py` | Representative RNN trajectories and metrics | 6 s |

Figures 1 and 2 recompute their numerical summaries from the anonymized data
and write `reference_comparison.json` beside the figures. Exact and
tolerance-based differences from the locked paper statistics are reported but
do not prevent figure export, because library versions can change statistical
metadata or floating-point rounding. Figure 8 requires a complete versioned
20-repetition fit cache and will not silently refit the model. Cache validity
strictly checks the fit code, parameters, objective settings, and target
matrices while allowing NumPy/SciPy version labels to differ across machines.
Small platform-dependent font or rasterization differences may occur. The
plotted values reproduce the locked manuscript analyses, while final
manuscript artwork may include manual layout and annotation adjustments.

## Repository layout

- `plot_*.py`: thin executable entry points;
- `src/`: the single copy of each analysis, plotting, RNN, and training
  implementation;
- `human_data/clean_data/`: 24 anonymized cleaned participant files;
- `reference_data/stats/`: only the 10 locked statistics files used by the
  executable entries;
- `models/`: one representative trained RNN checkpoint;
- `reference_outputs/`: small examples of expected outputs and QA records;
- `docs/`: data, runtime, and reproduction documentation.

## Scope

The release intentionally focuses on figures that can be reproduced from
compact inputs. It does not include the approximately 398 GB of intermediate
multi-seed RNN checkpoints and recurrent-rate arrays required to redraw every
RNN population figure. Core RNN architecture, task generation, checkpoint
handling, and training code remain in `src/` so the custom model is documented
and can be retrained. See `docs/FIGURE_CODE_MAP.md` for the exact boundary.

## Training a new RNN

The training entry point defaults to evaluation. Mode, task, and device are
selected without editing source files:

```powershell
$env:RNN_MODE='train'
$env:RNN_TASK='motorTraj_circle'
$env:RNN_DEVICE='cuda'
python src/PARAM.py
```

Full multi-seed training is computationally expensive and is not required for
the direct figure demonstrations above.

## Validation, data, and license

Run `python validate_release.py` to check required files, recorded checksums,
Python syntax, JSON, anonymous filenames, private paths, and file sizes.
Participant names are not present in the release and no reidentification key
is included. Public sharing of the anonymized human data remains subject to
the authors' ethics, consent, and institutional approvals.

Custom source code is released under the MIT License. External software and
data retain their original terms.

The current release version is recorded in `VERSION`. Create an immutable Git
tag with the matching version number (for example, `v0.2.0`). Published
releases can also be archived in a permanent repository such as Zenodo, with
the resulting DOI added to this README.
