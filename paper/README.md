# Paper Simulation & Key-Result Reproduction

This directory contains standalone scripts to reproduce key simulation results from the smoothing PSID paper ([OpenReview PDF](https://openreview.net/pdf?id=vFZhylvjKq)) without bundling paper-specific scripts or output artifacts into the `PSID` PyPI package (`source/PSID/`).

## Quick check (~10–20 seconds)

Runs 10 random linear systems across $N_{\text{train}} \in \{10^4, 10^5\}$:

```bash
python paper/reproduce_smoothing_sim.py
```

## Full paper scale (50 systems, $N_{\text{train}} \in \{10^4, 10^5, 10^6\}$)

```bash
python paper/reproduce_smoothing_sim.py --n-systems 50 --n-train-log10 4 5 6 --n-test 1000000
```

Outputs are written to `paper/results/` (gitignored):
- `realizations/`: saved random `LSSM` models (`system_<idx>_seed<seed>.pkl`) and zstd-compressed Parquet time series (`system_<idx>_seed<seed>_data.parquet`, reused on subsequent runs)
- `cache/`: per-system cached evaluation JSONs so runs can be extended (e.g., from 20 to 50 systems) without recomputing earlier systems
- `summary_metrics.json`: per-system and aggregated parameter errors and decoding $R^2$
- `summary_tables.md`: Markdown summary of parameter identification errors and decoding $R^2$
- `fig2_decoding_scatter.png` / `.pdf`: 3-panel Ideal vs. PSID decoding $R^2$ scatter plots for prediction ($k|k-1$), filtering ($k|k$), and `bwYRes` smoothing ($k|N$)
- `fig5_param_convergence.png` / `.pdf`: parameter identification error convergence across $N_{\text{train}}$
