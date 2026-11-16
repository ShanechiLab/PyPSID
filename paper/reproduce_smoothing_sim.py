"""
Copyright (c) 2026 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Reproduce key simulation results from the smoothing PSID paper.

Generates random linear state-space systems following the paper's benchmark
distribution, fits PSIDModel (prediction, causal filtering, and non-causal
smoothing variants including ablations), and reports both parameter
identification errors and behavioral decoding R^2 against the oracle baseline.
"""

import argparse
import json
import os
import pickle
import sys
import time

# Allow running directly from a cloned repository without `pip install -e .`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SOURCE_DIR = os.path.join(_REPO_ROOT, "source")
if os.path.isdir(_SOURCE_DIR) and _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from PSID import PSIDModel  # noqa: E402
from PSID.evaluation import evalPrediction, evalSysId  # noqa: E402
from PSID.sim_tools import (  # noqa: E402
    generateRandomLinearModel,
    getSysSettingsFromSysCode,
)

PAPER_URL = "https://openreview.net/pdf?id=vFZhylvjKq"

# System distribution specification from the smoothing PSID paper:
# ny in [5, 10], nz in [5, 10], nx in [1, 10], n1 in [1, 10], ne in [1, 10],
# log-uniform state/output noise scales [0.1, 10], and z SNR in [1, 100].
PAPER_SYS_CODE = (
    "20231220_stcfrm_nyR5_10_nzR5_10_NxR1_10_N1_R1_10_NeR1_10_"
    "xNScLR1e-1_1e+1_yNScLR1e-1_1e+1_zSNRLR1e+0_1e+2_icdf_pbeta2_1_"
)

PARAM_ERR_METRICS = [
    ("AErrNormed", "A"),
    ("CErrNormed", "Cy"),
    ("CzErrNormed", "Cz"),
    ("GErrNormed", "Sigma_xy"),
    ("KErrNormed", "K"),
    ("KfErrNormed", "Kf"),
    ("YCovErrNormed", "Sigma_y"),
    ("fltCzKfErrNormed", "Cz*Kf"),
    ("zEigErrNFN", "eig(z)"),
]

# Colors, linestyles, and legend labels matching Fig. 5 in the paper.
FIG5_STYLES = {
    "AErrNormed": ("#1f77b4", "-", r"$A$"),
    "CErrNormed": ("#2ca02c", ":", r"$C_y$"),
    "CzErrNormed": ("#d62728", "-.", r"$C_z$"),
    "GErrNormed": ("#8c564b", "--", r"$G_y$"),
    "KErrNormed": ("#e377c2", ":", r"$K$"),
    "KfErrNormed": ("#7f7f7f", "-.", r"$K_f$"),
    "YCovErrNormed": ("#bcbd22", "-", r"$\Sigma_y$"),
    "fltCzKfErrNormed": ("#17becf", "--", r"$C_z K_f$"),
    "zEigErrNFN": ("#aec7e8", ":", r"Eigs of $A$"),
}

FIG2_PANELS = [
    ("Ideal_pred", "PSID_pred", "a. One-step-ahead prediction ($k|k-1$)"),
    ("Ideal_filt", "PSID_filt", "b. Filtering ($k|k$)"),
    ("Ideal_smth", "PSID_smth_bwYRes", "c. Smoothing ($k|N$)"),
]

IDEAL_MODES = [
    ("Ideal_pred", dict(useXFilt=False, useXSmooth=False)),
    ("Ideal_filt", dict(useXFilt=True, useXSmooth=False)),
    ("Ideal_smth", dict(useXFilt=False, useXSmooth=True)),
]

# Ordered identically to Table 4 (tab:psid_ablation_decoding) in the paper.
TABLE4_ROWS = [
    (
        "One-step-ahead prediction",
        "PSID_std_pred",
        "Standard PSID, used as a Kalman predictor",
    ),
    ("", "Ideal_pred", "Ideal prediction (true model parameters)"),
    (
        "Filtering",
        "PSID_std_filt",
        "Standard PSID's parameters, used in a Kalman filter",
    ),
    ("", "PSID_shift1", "Prediction with time-shifted data"),
    ("", "PSID_filt", "**PSID with filtering (ours)**"),
    ("", "Ideal_filt", "Ideal filtering (true model parameters)"),
    (
        "Smoothing",
        "PSID_std_smth",
        "Standard PSID's parameters, used in a Kalman smoother",
    ),
    (
        "",
        "PSID_smth_bwYZ",
        "PSID with smoothing, backward pass on $z$ itself (non-residual)",
    ),
    (
        "",
        "PSID_smth_bwYRes",
        "**PSID with smoothing (ours, residual backward pass)**",
    ),
    ("", "Ideal_smth", "Ideal smoothing (true model parameters)"),
]

EXTRA_DECODING_ROWS = [
    ("PSID_pred", "PSID (updateKfKv=True), used as a Kalman predictor"),
    ("PSID_smth_bwZRes", "PSID with smoothing, residual backward pass on $z$ only"),
]

ALL_DECODING_KEYS = [key for _, key, _ in TABLE4_ROWS] + [
    key for key, _ in EXTRA_DECODING_ROWS
]


def _mean_r2(z_true, z_pred):
    """Computes mean R^2 across all behavioral output dimensions."""
    return float(np.nanmean(evalPrediction(z_true, z_pred, measure="R2")))


def _eval_ideal(true_sys, y_test, z_test):
    """Evaluates oracle prediction, filtering, and Kalman smoothing on test data."""
    scores = {}
    for label, kwargs in IDEAL_MODES:
        z_pred = true_sys.predict(y_test, **kwargs)[0]
        scores[label] = _mean_r2(z_test, z_pred)
    return scores


def _get_n1(true_sys):
    """Returns the behaviorally relevant state dimension n1 from a true LSSM."""
    z_dims = getattr(true_sys, "zDims", None)
    return len(z_dims) if isinstance(z_dims, (list, np.ndarray)) else 0


def _fit_and_eval_psid(true_sys, y_train, z_train, y_test, z_test, horizon):
    """Fits the default PSID model and its key ablations for one training set."""
    nx = true_sys.state_dim
    n1 = _get_n1(true_sys)

    # 1. Main model: prediction, causal filtering, and bwYRes smoothing share one fit.
    model = PSIDModel(
        nx=nx,
        n1=n1,
        i=horizon,
        updateKfKv=True,
        bw_on_residual=True,
        bw_on_residual_y=True,
    )
    model.fit(y_train, Z=z_train)

    param_errs = {}
    decoding = {}
    for mode, label in [
        ("prediction", "PSID_pred"),
        ("filtering", "PSID_filt"),
        ("smoothing", "PSID_smth_bwYRes"),
    ]:
        model.set_default_inference(mode)
        perf, z_pred, _, _ = evalSysId(
            model, YTest=y_test, ZTest=z_test, trueSys=true_sys
        )
        if not param_errs:
            for metric_key, _ in PARAM_ERR_METRICS:
                val = perf.get(metric_key, np.nan)
                param_errs[metric_key] = float(val) if val is not None else float("nan")
        decoding[label] = _mean_r2(z_test, z_pred)

    # 2. Standard PSID baseline (updateKfKv=False): predictor, filter, RTS smoother.
    model_std = PSIDModel(nx=nx, n1=n1, i=horizon, updateKfKv=False)
    model_std.fit(y_train, Z=z_train, enable_all_inference=False)
    std_lssm = model_std.getLSSM()
    for label, kwargs in [
        ("PSID_std_pred", dict(useXFilt=False, useXSmooth=False)),
        ("PSID_std_filt", dict(useXFilt=True, useXSmooth=False)),
        ("PSID_std_smth", dict(useXFilt=False, useXSmooth=True)),
    ]:
        decoding[label] = _mean_r2(z_test, std_lssm.predict(y_test, **kwargs)[0])

    # 3. Shift-1 baseline (0-lag causal prediction via zShift=1, no fltCzKf).
    model_shift1 = PSIDModel(nx=nx, n1=n1, i=horizon, updateKfKv=True, zShift=1)
    model_shift1.fit(y_train, Z=z_train, enable_all_inference=False)
    decoding["PSID_shift1"] = _mean_r2(z_test, model_shift1.predict(y_test)[0])

    # 4. Smoothing ablations: direct backward model (bwYZ) & Z-residual only (bwZRes).
    ablation_specs = [
        ("PSID_smth_bwYZ", False, False, True),
        ("PSID_smth_bwZRes", True, False, False),
    ]
    for label, bw_res, bw_res_y, eval_bw_gain in ablation_specs:
        ablation = PSIDModel(
            nx=nx,
            n1=n1,
            i=horizon,
            updateKfKv=True,
            bw_on_residual=bw_res,
            bw_on_residual_y=bw_res_y,
            default_inference="smoothing",
        )
        ablation.fit(y_train, Z=z_train)
        perf, z_pred, _, _ = evalSysId(
            ablation, YTest=y_test, ZTest=z_test, trueSys=true_sys
        )
        decoding[label] = _mean_r2(z_test, z_pred)
        if eval_bw_gain:
            bw_val = perf.get("fltCzKfBWErrNormed", np.nan)
            param_errs["fltCzKfBWErrNormed"] = (
                float(bw_val) if bw_val is not None else float("nan")
            )

    return param_errs, decoding


def _save_realization_parquet(data_path, y_all, x_all, z_all, u_all=None):
    """Writes simulated (y, x, z[, u]) time-series to a zstd-compressed Parquet file."""
    arrays, names = [], []
    series = [("y", y_all), ("x", x_all), ("z", z_all)]
    if u_all is not None:
        series.append(("u", u_all))
    for prefix, arr in series:
        for j in range(arr.shape[1]):
            arrays.append(pa.array(arr[:, j]))
            names.append(f"{prefix}_{j}")
    meta = {
        b"ny": str(y_all.shape[1]).encode(),
        b"nx": str(x_all.shape[1]).encode(),
        b"nz": str(z_all.shape[1]).encode(),
        b"nu": str(u_all.shape[1] if u_all is not None else 0).encode(),
    }
    table = pa.Table.from_arrays(arrays, names=names, metadata=meta)
    pq.write_table(table, data_path, compression="zstd")


def _load_realization_parquet(data_path, n_total, load_u=False):
    """Loads (y, z[, u]) up to n_total rows from a Parquet realization file if long enough."""
    pf = pq.ParquetFile(data_path)
    if pf.metadata.num_rows < n_total:
        return None
    s_meta = pf.schema_arrow.metadata or {}
    ny = int(s_meta[b"ny"])
    nz = int(s_meta[b"nz"])
    nu = int(s_meta.get(b"nu", b"0"))
    cols = [f"y_{j}" for j in range(ny)] + [f"z_{j}" for j in range(nz)]
    if load_u and nu > 0:
        cols += [f"u_{j}" for j in range(nu)]
    tbl = pq.read_table(data_path, columns=cols).slice(0, n_total)
    y_all = np.column_stack([tbl.column(f"y_{j}").to_numpy() for j in range(ny)])
    z_all = np.column_stack([tbl.column(f"z_{j}").to_numpy() for j in range(nz)])
    if load_u:
        u_all = (
            np.column_stack([tbl.column(f"u_{j}").to_numpy() for j in range(nu)])
            if nu > 0
            else None
        )
        return y_all, z_all, u_all
    return y_all, z_all


def _load_or_generate_realization(
    sys_settings, sys_idx, n_total, seed, realizations_dir, with_u=False
):
    """Loads cached random LSSM and data or generates and saves them to disk."""
    os.makedirs(realizations_dir, exist_ok=True)
    stem = f"system_{sys_idx + 1:02d}_seed{seed}"
    model_path = os.path.join(realizations_dir, f"{stem}.pkl")
    data_path = os.path.join(realizations_dir, f"{stem}_data.parquet")

    if os.path.exists(model_path) and os.path.exists(data_path):
        loaded = _load_realization_parquet(data_path, n_total, load_u=with_u)
        if loaded is not None:
            with open(model_path, "rb") as f:
                true_sys = pickle.load(f)
            if with_u:
                return true_sys, loaded[0], loaded[1], loaded[2]
            return true_sys, loaded[0], loaded[1]

    np.random.seed(seed + sys_idx)
    true_sys, sys_u, _ = generateRandomLinearModel(sys_settings)
    if with_u and sys_u is not None:
        u_all, _ = sys_u.generateRealization(n_total)
        y_all, x_all, z_all = true_sys.generateRealization(
            n_total, u=u_all, random_x0=False, return_z=True
        )
    else:
        u_all = None
        y_all, x_all, z_all = true_sys.generateRealization(
            n_total, random_x0=False, return_z=True
        )
    with open(model_path, "wb") as f:
        pickle.dump(true_sys, f)
    _save_realization_parquet(data_path, y_all, x_all, z_all, u_all=u_all)
    if with_u:
        return true_sys, y_all, z_all, u_all
    return true_sys, y_all, z_all


def _eval_cache_path(cache_dir, sys_idx, seed, n_train_max, n_test, horizon):
    """Returns the per-system cached evaluation JSON path."""
    name = (
        f"system_{sys_idx + 1:02d}_seed{seed}_nmax{n_train_max}"
        f"_ntest{n_test}_h{horizon}.json"
    )
    return os.path.join(cache_dir, name)


def _run_or_load_system(
    sys_settings,
    sys_idx,
    n_trains,
    n_train_max,
    n_test,
    horizon,
    seed,
    output_dir,
    sys_code=PAPER_SYS_CODE,
):
    """Evaluates a single system, reusing cached realizations and metrics."""
    realizations_dir = os.path.join(output_dir, "realizations", sys_code)
    cache_dir = os.path.join(output_dir, "cache", sys_code)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _eval_cache_path(
        cache_dir, sys_idx, seed, n_train_max, n_test, horizon
    )

    cached = None
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        has_all_ideal = all(
            lbl in cached.get("ideal_decoding", {}) for lbl, _ in IDEAL_MODES
        )
        has_all_n = all(
            str(n_tr) in cached.get("param_errors", {})
            and str(n_tr) in cached.get("psid_decoding", {})
            and all(
                k in cached["psid_decoding"][str(n_tr)] for k in ALL_DECODING_KEYS
            )
            for n_tr in n_trains
        )
        if has_all_ideal and has_all_n:
            return cached, True

    true_sys, y_all, z_all = _load_or_generate_realization(
        sys_settings, sys_idx, n_train_max + n_test, seed, realizations_dir
    )
    y_test = y_all[n_train_max:]
    z_test = z_all[n_train_max:]

    if cached is not None and "meta" in cached and "ideal_decoding" in cached:
        record = cached
    else:
        record = {
            "meta": {
                "nx": int(true_sys.state_dim),
                "n1": int(_get_n1(true_sys)),
                "ny": int(y_all.shape[1]),
                "nz": int(z_all.shape[1]),
            },
            "ideal_decoding": _eval_ideal(true_sys, y_test, z_test),
            "param_errors": {},
            "psid_decoding": {},
        }

    for n_tr in n_trains:
        key = str(n_tr)
        if (
            key in record["param_errors"]
            and key in record["psid_decoding"]
            and all(k in record["psid_decoding"][key] for k in ALL_DECODING_KEYS)
        ):
            continue
        param_errs, dec_scores = _fit_and_eval_psid(
            true_sys, y_all[:n_tr], z_all[:n_tr], y_test, z_test, horizon
        )
        record["param_errors"][key] = param_errs
        record["psid_decoding"][key] = dec_scores

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record, False


def run_benchmark(n_systems, n_train_log10, n_test, horizon, seed, output_dir):
    """Simulates systems and collects parameter error and decoding R^2 results."""
    sys_settings = getSysSettingsFromSysCode(PAPER_SYS_CODE)
    n_trains = [10**k for k in n_train_log10]
    n_train_max = max(n_trains)

    ideal_results = {label: [] for label, _ in IDEAL_MODES}
    psid_param_results = {n_tr: [] for n_tr in n_trains}
    psid_dec_results = {n_tr: [] for n_tr in n_trains}

    t0 = time.time()
    for sys_idx in range(n_systems):
        record, from_cache = _run_or_load_system(
            sys_settings,
            sys_idx,
            n_trains,
            n_train_max,
            n_test,
            horizon,
            seed,
            output_dir,
        )
        for label in ideal_results:
            ideal_results[label].append(record["ideal_decoding"][label])
        for n_tr in n_trains:
            key = str(n_tr)
            psid_param_results[n_tr].append(record["param_errors"][key])
            psid_dec_results[n_tr].append(record["psid_decoding"][key])

        meta = record["meta"]
        elapsed = time.time() - t0
        status = "cached" if from_cache else "done"
        print(
            f"[{sys_idx + 1}/{n_systems}] nx={meta['nx']}, "
            f"n1={meta['n1']}, ny={meta['ny']}, "
            f"nz={meta['nz']} ({status}, {elapsed:.1f}s)",
            flush=True,
        )

    return {
        "n_systems": n_systems,
        "n_train_sizes": n_trains,
        "n_test": n_test,
        "horizon": horizon,
        "seed": seed,
        "ideal_decoding": ideal_results,
        "param_errors": {str(n_tr): psid_param_results[n_tr] for n_tr in n_trains},
        "psid_decoding": {str(n_tr): psid_dec_results[n_tr] for n_tr in n_trains},
    }


def _aggregate_summary(results):
    """Computes parameter error stats (median/IQR and log10 mean/SEM) and R^2."""
    n_trains = results["n_train_sizes"]
    param_summary = {}
    all_param_keys = [k for k, _ in PARAM_ERR_METRICS] + ["fltCzKfBWErrNormed"]
    for n_tr in n_trains:
        rows = results["param_errors"][str(n_tr)]
        param_summary[str(n_tr)] = {}
        for key in all_param_keys:
            vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            pos_vals = vals[np.isfinite(vals) & (vals > 0)]
            log_vals = np.log10(pos_vals) if len(pos_vals) > 0 else np.array([])
            log_mean = float(np.mean(log_vals)) if len(log_vals) > 0 else float("nan")
            log_sem = (
                float(np.std(log_vals, ddof=1) / np.sqrt(len(log_vals)))
                if len(log_vals) > 1
                else 0.0
            )
            param_summary[str(n_tr)][key] = {
                "median": float(np.nanmedian(vals)),
                "q25": float(np.nanpercentile(vals, 25)),
                "q75": float(np.nanpercentile(vals, 75)),
                "log10_mean": log_mean,
                "log10_sem": log_sem,
            }

    decoding_summary = {}
    for n_tr in n_trains:
        rows = results["psid_decoding"][str(n_tr)]
        decoding_summary[str(n_tr)] = {}
        for method_key in ALL_DECODING_KEYS:
            if method_key.startswith("Ideal_"):
                vals = np.array(results["ideal_decoding"][method_key], dtype=float)
            else:
                vals = np.array([r[method_key] for r in rows], dtype=float)
            decoding_summary[str(n_tr)][method_key] = {
                "mean": float(np.nanmean(vals)),
                "sem": float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else 0.0,
            }

    return {"param_errors": param_summary, "decoding_r2": decoding_summary}


def _build_summary_tables(results, summary):
    """Builds header and row data for parameter-error and Table 4 decoding tables."""
    n_trains = results["n_train_sizes"]
    param_cols = PARAM_ERR_METRICS + [("fltCzKfBWErrNormed", "Cz*Kf(BW)")]
    param_headers = ["N_train"] + [lbl for _, lbl in param_cols]
    param_rows = []
    for n_tr in n_trains:
        stats = summary["param_errors"][str(n_tr)]
        row = [f"{n_tr:,}"] + [
            f"{stats[key]['median']:.4f}" for key, _ in param_cols
        ]
        param_rows.append(row)

    max_n = str(max(n_trains))
    dec_headers = ["Setting", "Model", "Mean decoding R^2", "SEM"]
    dec_rows = []
    for setting, key, label in TABLE4_ROWS:
        entry = summary["decoding_r2"][max_n][key]
        dec_rows.append(
            [setting, label, f"{entry['mean']:.4f}", f"{entry['sem']:.4f}"]
        )

    extra_rows = []
    for key, label in EXTRA_DECODING_ROWS:
        entry = summary["decoding_r2"][max_n][key]
        extra_rows.append([label, f"{entry['mean']:.4f}", f"{entry['sem']:.4f}"])

    return param_headers, param_rows, dec_headers, dec_rows, extra_rows


def _render_md_table(headers, rows, align_right_from=1):
    """Formats headers and string rows as a GitHub-flavored Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            "---:" if idx >= align_right_from else "---"
            for idx in range(len(headers))
        )
        + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _strip_md(text):
    """Removes inline Markdown/LaTeX markers for plain terminal tables."""
    return text.replace("**", "").replace("$", "")


def _print_summary(results, summary):
    """Prints formatted tables of parameter identification errors and Table 4 R^2."""
    p_headers, p_rows, _, d_rows, extra_rows = _build_summary_tables(results, summary)
    max_n = f"{max(results['n_train_sizes']):,}"

    print("\n=== Median Parameter Identification Errors (normalized) ===")
    p_widths = [
        max(len(h), *(len(r[i]) for r in p_rows)) for i, h in enumerate(p_headers)
    ]
    print(" | ".join(f"{h:>{w}s}" for h, w in zip(p_headers, p_widths)))
    print("-+-".join("-" * w for w in p_widths))
    for row in p_rows:
        print(" | ".join(f"{val:>{w}s}" for val, w in zip(row, p_widths)))

    print(
        f"\n=== Table 4 Behavioral Decoding R^2 (N_train = {max_n}, mean +/- SEM) ==="
    )
    plain_d = [(s, _strip_md(m), r2, sem) for s, m, r2, sem in d_rows]
    w_model = max(len("Model"), *(len(m) for _, m, _, _ in plain_d))
    print(f"{'Model':<{w_model}s} | {'Mean R^2':>8s} | {'SEM':>6s}")
    print("-" * (w_model + 20))
    for setting, model_lbl, mean_str, sem_str in plain_d:
        if setting:
            print(f"[{setting}]")
        print(f"{model_lbl:<{w_model}s} | {mean_str:>8s} | {sem_str:>6s}")
    if extra_rows:
        print("[Additional variants]")
        for model_lbl, mean_str, sem_str in extra_rows:
            plain_lbl = _strip_md(model_lbl)
            print(f"{plain_lbl:<{w_model}s} | {mean_str:>8s} | {sem_str:>6s}")


def _save_markdown_tables(results, summary, output_dir):
    """Writes parameter error and paper-aligned Table 4 results to Markdown."""
    p_headers, p_rows, d_headers, d_rows, extra_rows = _build_summary_tables(
        results, summary
    )
    n_trains_str = ", ".join(f"{n:,}" for n in results["n_train_sizes"])
    max_n = f"{max(results['n_train_sizes']):,}"

    md_d_rows = [
        [f"*{setting}*" if setting else "", model_lbl, mean_str, sem_str]
        for setting, model_lbl, mean_str, sem_str in d_rows
    ]
    sections = [
        "# Smoothing PSID Simulation Results",
        "",
        (
            f"- Systems: `{results['n_systems']}` | "
            f"Training sizes: `{n_trains_str}` | "
            f"Test samples: `{results['n_test']:,}` | "
            f"Horizon: `{results['horizon']}` | "
            f"Seed: `{results['seed']}`"
        ),
        "",
        "## Parameter Identification Errors (Fig. 5, median normalized error)",
        "",
        _render_md_table(p_headers, p_rows, align_right_from=0),
        "",
        (
            f"## Ablation Results for Decoding $R^2$ "
            f"(Table 4, $N_{{\\text{{train}}}} = {max_n}$, mean +/- SEM)"
        ),
        "",
        (
            f"This table is equivalent to Table 4 in [the paper]({PAPER_URL}), "
            "using the same row ordering and model descriptions."
        ),
        "",
        _render_md_table(d_headers, md_d_rows, align_right_from=2),
        "",
        "### Additional variants (not in Table 4)",
        "",
        _render_md_table(
            ["Model", "Mean decoding R^2", "SEM"], extra_rows, align_right_from=1
        ),
        "",
    ]
    md_path = os.path.join(output_dir, "summary_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    return md_path


def _save_fig2_scatter(results, output_dir):
    """Generates the 3-panel Fig 2 scatter comparing Ideal vs. PSID R^2."""
    n_trains = results["n_train_sizes"]
    max_n = max(n_trains)
    psid_rows = results["psid_decoding"][str(max_n)]
    ideal_dict = results["ideal_decoding"]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)
    all_vals = []
    for ideal_key, psid_key, _ in FIG2_PANELS:
        all_vals.extend(ideal_dict[ideal_key])
        all_vals.extend(row[psid_key] for row in psid_rows)
    finite_vals = [v for v in all_vals if np.isfinite(v)]
    lo = min(-0.02, float(np.min(finite_vals)) - 0.04) if finite_vals else -0.02
    hi = 1.04

    for ax, (ideal_key, psid_key, title) in zip(axes, FIG2_PANELS):
        x = np.asarray(ideal_dict[ideal_key], dtype=float)
        y = np.asarray([row[psid_key] for row in psid_rows], dtype=float)
        ax.plot([lo, hi], [lo, hi], "--", color="#6c757d", linewidth=1.2, zorder=1)
        ax.scatter(
            x,
            y,
            s=36,
            marker="x",
            linewidths=1.5,
            alpha=0.85,
            color="#1f77b4",
            zorder=2,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("Ideal $R^2$")
        ax.set_ylabel("PSID $R^2$")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        f"True vs. learned estimation $R^2$ ($N_{{\\mathrm{{train}}}}={max_n:,}$, "
        f"$n={results['n_systems']}$ systems)"
    )
    for ext in ("png", "pdf"):
        out_path = os.path.join(output_dir, f"fig2_decoding_scatter.{ext}")
        fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_fig5_convergence(results, summary, output_dir):
    """Generates paper-style Fig 5 log10 error convergence with shaded SEM."""
    n_trains = results["n_train_sizes"]
    x_log = np.log10(np.asarray(n_trains, dtype=float))
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

    for key, _ in PARAM_ERR_METRICS:
        stats = [summary["param_errors"][str(n)][key] for n in n_trains]
        means = np.asarray([s["log10_mean"] for s in stats], dtype=float)
        sems = np.asarray([s["log10_sem"] for s in stats], dtype=float)
        color, linestyle, label = FIG5_STYLES[key]
        ax.plot(
            x_log,
            means,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=label,
        )
        ax.fill_between(
            x_log,
            means - sems,
            means + sems,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlim(x_log[0], x_log[-1])
    ax.set_xlabel("Log10 training samples")
    ax.set_ylabel("Log10 Normed Fro Err")
    ax.set_title(
        f"System identification error convergence ($n={results['n_systems']}$ systems)"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.legend(
        bbox_to_anchor=(1.03, 0.5),
        loc="center left",
        frameon=False,
        fontsize=8.5,
    )

    for ext in ("png", "pdf"):
        out_path = os.path.join(output_dir, f"fig5_param_convergence.{ext}")
        fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_plots(results, summary, output_dir):
    """Saves Fig 2 scatter plots and parameter error convergence plot."""
    _save_fig2_scatter(results, output_dir)
    _save_fig5_convergence(results, summary, output_dir)


def parse_args(argv=None):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce key simulation results from the smoothing PSID paper."
    )
    parser.add_argument(
        "--n-systems",
        type=int,
        default=10,
        help="Number of random linear systems to simulate (paper uses 50).",
    )
    parser.add_argument(
        "--n-train-log10",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Training sizes as log10 values (paper uses 4 5 6).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000000,
        help="Number of held-out test samples per system (paper uses 1000000).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="PSID subspace horizon hyperparameter i (paper uses 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for system generation and simulation.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to write summary_metrics.json and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving summary figures.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Runs the simulation benchmark and saves numerical and visual summaries."""
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    results = run_benchmark(
        n_systems=args.n_systems,
        n_train_log10=args.n_train_log10,
        n_test=args.n_test,
        horizon=args.horizon,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    summary = _aggregate_summary(results)
    _print_summary(results, summary)

    payload = {"config": vars(args), "summary": summary, "raw": results}
    json_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics to {json_path}")

    md_path = _save_markdown_tables(results, summary, args.output_dir)
    print(f"Saved tables to {md_path}")

    if not args.no_plots:
        _save_plots(results, summary, args.output_dir)
        print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
