# Directly executable figure map

| Public entry point | Output | Included inputs | Reproduction status |
|---|---|---|---|
| `plot_fig1.py` | Fig. 1; Supp. Figs. 1, 3 | 24 anonymized participant files | Full recalculation with locked-statistics verification; programmatic layout |
| `plot_fig2.py` | Fig. 2; Supp. Fig. 4 | Human data and compact behavioral caches | Full recalculation with locked-statistics verification; programmatic layout |
| `plot_fig8.py` | Fig. 8 | Human data, locked RNN targets, 20 cached Hopf fits | Data-panel regeneration from the validated cache; no silent refitting |
| `plot_rnn_demo.py` | Representative RNN trajectories | One checkpoint and task code | Functional model demonstration |

The release does not claim direct reconstruction of paper Figs. 3-7. Those
figures depend on large multi-seed recurrent-rate, perturbation, and geometry
arrays that are generated products rather than source code. Their central
model architecture, task, and training implementation remain under `src/`.
