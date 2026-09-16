# Runtime and hardware

The four direct entries require only a CPU. The representative RNN demo uses
approximately 2 GB RAM; the other entries require less in the tested setup.

Measured approximate runtimes in the reference Windows CPU environment:

- `plot_fig1.py`: 25 seconds;
- `plot_fig2.py`: 19 seconds;
- `plot_fig8.py`: 10 seconds with the valid fit cache;
- `plot_rnn_demo.py`: 6 seconds;
- `plot_all.py`: about 70 seconds;
- `validate_release.py`: 2 seconds.

Exact times vary by machine. CUDA is optional and only useful for new RNN
training. Full multi-seed training and evaluation are not part of the direct
figure workflow.
