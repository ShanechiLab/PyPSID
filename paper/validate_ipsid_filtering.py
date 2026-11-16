"""
Copyright (c) 2026 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Validate causal filtering (k|k) for IPSID (nu > 0) on random linear models.

Reuses the simulation, Parquet caching, and reporting utilities from
`paper/reproduce_smoothing_sim.py`, storing results under
`paper/results/realizations/<sys_code>/` and `paper/results/cache/<sys_code>/`.
"""

import argparse
import json
import os
import sys
import time

_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PAPER_DIR, ".."))
_SOURCE_DIR = os.path.join(_REPO_ROOT, "source")
for _p in (_PAPER_DIR, _SOURCE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import reproduce_smoothing_sim as rss  # noqa: E402

from PSID import PSIDModel  # noqa: E402
from PSID.evaluation import evalSysId  # noqa: E402
from PSID.sim_tools import getSysSettingsFromSysCode  # noqa: E402

IPSID_SYS_CODE = (
    "20231220_stcfrm_nyR5_10_nzR5_10_nuR1_5_NxR1_10_N1_R1_10_NeR1_10_"
    "xNScLR1e-1_1e+1_yNScLR1e-1_1e+1_zSNRLR1e+0_1e+2_icdf_pbeta2_1_"
)

DECODING_KEYS = ["PSID_pred", "Ideal_pred", "PSID_filt", "Ideal_filt"]
PARAM_KEYS = ["AErrNormed", "CErrNormed", "CzErrNormed", "KErrNormed", "fltCzKfErrNormed"]


def _eval_ideal_ipsid(true_sys, y_test, z_test, u_test):
    """Evaluates oracle prediction and filtering on IPSID test data."""
    z_pred = true_sys.predict(y_test, U=u_test, useXFilt=False)[0]
    z_filt = true_sys.predict(y_test, U=u_test, useXFilt=True)[0]
    return {
        "Ideal_pred": rss._mean_r2(z_test, z_pred),
        "Ideal_filt": rss._mean_r2(z_test, z_filt),
    }


def _fit_and_eval_ipsid(
    true_sys, y_train, z_train, u_train, y_test, z_test, u_test, horizon
):
    """Fits PSIDModel with U (IPSID) and evaluates prediction & filtering."""
    nx = true_sys.state_dim
    n1 = rss._get_n1(true_sys)

    model = PSIDModel(
        nx=nx, n1=n1, i=horizon, default_inference="prediction", updateKfKv=True
    )
    model.fit(y_train, z_train, U=u_train)
    pred_r2 = rss._mean_r2(z_test, model.predict(y_test, U=u_test)[0])

    model.set_default_inference("filtering")
    filt_r2 = rss._mean_r2(z_test, model.predict(y_test, U=u_test)[0])

    raw_err = evalSysId(
        model,
        y_train,
        z_train,
        y_test,
        z_test,
        UTrain=u_train,
        UTest=u_test,
        trueSys=true_sys,
    )[0]
    param_errs = {
        k: float(raw_err[k]) for k in PARAM_KEYS if k in raw_err and np.isfinite(raw_err[k])
    }
    return param_errs, {"PSID_pred": pred_r2, "PSID_filt": filt_r2}


def _run_or_load_ipsid_system(
    sys_settings, sys_idx, n_trains, n_train_max, n_test, horizon, seed, output_dir, sys_code
):
    """Evaluates a single nu > 0 system, reusing cached realizations and metrics."""
    realizations_dir = os.path.join(output_dir, "realizations", sys_code)
    cache_dir = os.path.join(output_dir, "cache", sys_code)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = rss._eval_cache_path(
        cache_dir, sys_idx, seed, n_train_max, n_test, horizon
    )

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        has_ideal = all(k in cached.get("ideal_decoding", {}) for k in ["Ideal_pred", "Ideal_filt"])
        has_n = all(
            str(n_tr) in cached.get("ipsid_decoding", {}) for n_tr in n_trains
        )
        if has_ideal and has_n:
            return cached, True

    true_sys, y_all, z_all, u_all = rss._load_or_generate_realization(
        sys_settings, sys_idx, n_train_max + n_test, seed, realizations_dir, with_u=True
    )
    y_test, z_test, u_test = y_all[n_train_max:], z_all[n_train_max:], u_all[n_train_max:]

    record = {
        "meta": {
            "nx": int(true_sys.state_dim),
            "n1": int(rss._get_n1(true_sys)),
            "ny": int(y_all.shape[1]),
            "nz": int(z_all.shape[1]),
            "nu": int(u_all.shape[1]),
        },
        "ideal_decoding": _eval_ideal_ipsid(true_sys, y_test, z_test, u_test),
        "param_errors": {},
        "ipsid_decoding": {},
    }

    for n_tr in n_trains:
        key = str(n_tr)
        param_errs, dec_scores = _fit_and_eval_ipsid(
            true_sys,
            y_all[:n_tr],
            z_all[:n_tr],
            u_all[:n_tr],
            y_test,
            z_test,
            u_test,
            horizon,
        )
        record["param_errors"][key] = param_errs
        record["ipsid_decoding"][key] = dec_scores

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record, False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate causal filtering (k|k) for IPSID (nu > 0)."
    )
    parser.add_argument("--n-systems", type=int, default=10)
    parser.add_argument("--n-train-log10", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--n-test", type=int, default=50000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_PAPER_DIR, "results"),
    )
    args = parser.parse_args(argv)

    sys_settings = getSysSettingsFromSysCode(IPSID_SYS_CODE)
    n_trains = [10**k for k in args.n_train_log10]
    n_train_max = max(n_trains)

    records = []
    t0 = time.time()
    for sys_idx in range(args.n_systems):
        rec, from_cache = _run_or_load_ipsid_system(
            sys_settings,
            sys_idx,
            n_trains,
            n_train_max,
            args.n_test,
            args.horizon,
            args.seed,
            args.output_dir,
            IPSID_SYS_CODE,
        )
        records.append(rec)
        m = rec["meta"]
        tag = "cached" if from_cache else "done"
        last_n = str(n_trains[-1])
        d = rec["ipsid_decoding"][last_n]
        idl = rec["ideal_decoding"]
        print(
            f"[{sys_idx + 1:2d}/{args.n_systems}] ({tag}, "
            f"ny={m['ny']}, nz={m['nz']}, nu={m['nu']}, nx={m['nx']}, n1={m['n1']}) "
            f"pred={d['PSID_pred']:.3f}/{idl['Ideal_pred']:.3f}, "
            f"filt={d['PSID_filt']:.3f}/{idl['Ideal_filt']:.3f} "
            f"[{time.time() - t0:.1f}s]",
            flush=True,
        )

    headers = ["Mode"] + [f"N={n_tr:,}" for n_tr in n_trains]
    rows = []
    for key in DECODING_KEYS:
        vals = []
        for n_tr in n_trains:
            if key.startswith("Ideal_"):
                arr = [r["ideal_decoding"][key] for r in records]
            else:
                arr = [r["ipsid_decoding"][str(n_tr)][key] for r in records]
            vals.append(f"{np.mean(arr):.4f} +/- {np.std(arr) / np.sqrt(len(arr)):.4f}")
        rows.append([key] + vals)
    print("\n### IPSID Decoding R^2 (mean +/- SEM)")
    print(rss._render_md_table(headers, rows))

    err_rows = []
    for k in PARAM_KEYS:
        vals = []
        for n_tr in n_trains:
            arr = [
                r["param_errors"][str(n_tr)][k]
                for r in records
                if k in r["param_errors"][str(n_tr)]
            ]
            vals.append(f"{np.mean(arr):.4f}" if arr else "N/A")
        err_rows.append([k] + vals)
    print("\n### IPSID Parameter Errors (mean)")
    print(rss._render_md_table(headers, err_rows))


if __name__ == "__main__":
    main()
