"""Omid Sani, Shanechi Lab, University of Southern California, 2023"""

"A class for for training PSID models"

import sys, os, logging, time, multiprocessing

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import PSID

from .tools import prepare_fold_inds, applyFuncIf, transposeIf, subtractIf, catIf
from .evaluation import evalPrediction
from .ReducedRankRegressor import ReducedRankRegressor

logger = logging.getLogger(__name__)


def evalPerformanceMetrics(
    trueVals, predVals, prefix="", metrics=["CC", "R2"], add_mean=True
):
    perf = {}
    for metric in metrics:
        perf[prefix + metric] = evalPrediction(trueVals, predVals, metric)
        if add_mean:
            perf["mean" + prefix + metric] = np.nanmean(perf[prefix + metric])
    return perf


def trainAndEvalFold(Z, Y, fold, nx, n1, i, default_inference, logStr=None, **kwargs):
    ZTrain = Z[fold["train_inds"], :]
    YTrain = Y[fold["train_inds"], :]
    try:
        model = PSIDModel(nx=nx, n1=n1, i=i, default_inference=default_inference)
        if "time_first" in kwargs:
            kwargs["time_first"] = True
        model.fit(YTrain, ZTrain, enable_all_inference=False, **kwargs)
        model.verbose = False
        YTest = Y[fold["test_inds"], :]
        allZp, allYp, allXp = model.predict(YTest)

        ZTest = Z[fold["test_inds"], :]
        perf = {}
        perf.update(evalPerformanceMetrics(ZTest, allZp, prefix="z"))
        perf.update(evalPerformanceMetrics(YTest, allYp, prefix="y"))
    except Exception as e:
        print(f"Error in fitting model: {e}")
        model = None
        perf = None
    return {"model": model, "perf": perf}


def trainAndEvalCVed(Z, Y, iCV_folds, *args, **kwargs):
    fold_results = []
    for foldInd, inner_fold in enumerate(iCV_folds):
        if "logStr" in kwargs and kwargs["logStr"] is not None:
            logger.info(f"iCV Fold {1+foldInd}/{len(iCV_folds)} " + kwargs["logStr"])
        fold_res = trainAndEvalFold(Z, Y, inner_fold, *args, **kwargs)
        fold_results.append(fold_res)
    return fold_results


def trainAndEvalCVedFromArgs(args):
    return trainAndEvalCVed(**args)


class PSIDModel:
    def __init__(
        self,
        nx=None,  # Total number of states
        n1=None,  # Number of states to be extracted in the first stage (behaviorally relevant)
        i=None,  # An array of [iY, iZ]. Will be ignored is iY or iZ is not None
        iY=None,  # Neural horizon
        iZ=None,  # Behavior horizon
        hyper_param_settings=None,  # Settings for hyper-parameter selection
        default_inference="prediction",  # Can also be 'filtering' or 'smoothing',
        bw_on_residual=True,
    ):
        self.nx = nx
        self.n1 = n1
        self.i = i if (iY is None and iZ is None) else None
        if self.i is not None and not isinstance(self.i, list):
            self.i = [self.i]
        self.iY = iY
        self.iZ = iZ
        if hyper_param_settings is None:
            hyper_param_settings = {}
        self.hyper_param_settings = hyper_param_settings
        if "folds" not in self.hyper_param_settings:
            self.hyper_param_settings["folds"] = 3
        if "max_horizon" not in self.hyper_param_settings:
            self.hyper_param_settings["max_horizon"] = 50
        if "selection_criteria" not in self.hyper_param_settings:
            self.hyper_param_settings["selection_criteria"] = "meanzR2"
        if "min_cpu_for_parallel":
            self.hyper_param_settings["min_cpu_for_parallel"] = 4
        self.set_default_inference(default_inference)
        self.bw_on_residual = bw_on_residual
        self.info = {}

    def set_default_inference(self, default_inference):
        allowed = ["prediction", "filtering", "smoothing"]
        if default_inference not in allowed:
            raise (
                Exception(
                    f'Default inference of "{default_inference}" not supported. Must be one of: {allowed}'
                )
            )
        self.default_inference = default_inference

    def get_horizons(self):
        iY = self.iY
        iZ = self.iZ if self.iZ is not None else self.iY
        if iY is None and iZ is None and self.i is not None:
            iY = self.i[0]
            iZ = self.i[-1]
        return iY, iZ

    def fit(self, Y, Z=None, enable_all_inference=True, time_first=True, **kwargs):
        iY, iZ = self.get_horizons()
        if (iY is None and iZ is None) or self.n1 is None or self.nx is None:
            param_selected, param_sets, param_results, best_param_ind = (
                self._find_hyper_params(Y, Z=Z, **kwargs)
            )
            self.param_selected = param_selected
            self.param_sets = param_sets
            self.param_results = param_results
            self.best_param_ind = best_param_ind
            if "iY" in param_selected:
                self.iY = param_selected["iY"]
            if "iZ" in param_selected:
                self.iZ = param_selected["iZ"]
            if "n1" in param_selected:
                self.n1 = param_selected["n1"]
            if "nx" in param_selected:
                self.nx = param_selected["nx"]

        iY, iZ = self.get_horizons()
        tic = time.perf_counter()
        model = PSID.PSID(
            Y=Y,
            Z=Z,
            nx=self.nx,
            n1=self.n1,
            i=[iY, iZ],
            time_first=time_first,
            **kwargs,
        )
        model.zDims = np.arange(1, 1 + min([self.n1, self.nx]))

        if enable_all_inference or self.default_inference in ["filtering", "smoothing"]:
            YTF = Y if time_first else transposeIf(Y)
            ZTF = Z if time_first else transposeIf(Z)
            # Regression to find optimal Cz * Kf for K|k filtering
            Zp, Yp, Xp = model.predict(YTF)
            # fltCzKf = PSID.projOrth((Z - Zp).T, (Y - Yp).T)[1]
            RRR = ReducedRankRegressor(
                catIf(subtractIf(YTF, Yp), axis=0),
                catIf(subtractIf(ZTF, Zp), axis=0),
                self.nx,
            )
            fltCzKf = np.array(RRR.W @ RRR.A)
            if isinstance(YTF, (list, tuple)):
                Zf = [Zp[i] + (fltCzKf @ (YTF[i] - Yp[i]).T).T for i in range(len(YTF))]
            else:
                Zf = Zp + (fltCzKf @ (YTF - Yp).T).T
        else:
            fltCzKf = None

        if enable_all_inference or self.default_inference in ["smoothing"]:
            # Time-reversed model for optimal backward pass for K|N smoothing
            YBW = applyFuncIf(YTF, np.flipud)
            ZBW = (
                applyFuncIf(subtractIf(ZTF, Zf), np.flipud)
                if self.bw_on_residual
                else applyFuncIf(ZTF, np.flipud)
            )
            modelBW = PSID.PSID(
                Y=YBW,
                Z=ZBW,
                nx=self.nx,
                n1=self.n1,
                i=[iY, iZ],
                time_first=True,
                **kwargs,
            )
            modelBW.zDims = np.arange(1, 1 + min([self.n1, self.nx]))
            ZpBW, YpBW, XpBW = modelBW.predict(YBW)
            # fltCzKfBW = PSID.projOrth((ZBW - ZpBW).T, (YBW - YpBW).T)[1]
            RRRBW = ReducedRankRegressor(
                catIf(subtractIf(YBW, YpBW), axis=0),
                catIf(subtractIf(ZBW, ZpBW), axis=0),
                self.nx,
            )
            fltCzKfBW = np.array(RRRBW.W @ RRRBW.A)
            if isinstance(YTF, (list, tuple)):
                ZfBW = [
                    ZpBW[i] + (fltCzKfBW @ (YBW[i] - YpBW[i]).T).T
                    for i in range(len(YTF))
                ]
            else:
                ZfBW = ZpBW + (fltCzKfBW @ (YBW - YpBW).T).T
            # if self.bw_on_residual:
            #     Zs = Zf + np.flipud(ZfBW)
            # else:
            #     Zs = (Zf + np.flipud(ZfBW)) / 2
        else:
            modelBW, fltCzKfBW = None, None

        # perfs = {}
        # perfs.update( evalPerformanceMetrics(Z, Zp, 'pred', ['CC', 'R2']) )
        # perfs.update( evalPerformanceMetrics(Z, Zf, 'filt', ['CC', 'R2']) )
        # perfs.update( evalPerformanceMetrics(Z, Zs, 'smth', ['CC', 'R2']) )
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(Z[:, 0], label='True')
        # plt.plot(Zp[:, 0], label='Predicted (xk|k-1)')
        # plt.plot(Zf[:, 0], label='Filtered (xk|k)')
        # plt.plot(Zs[:, 0], label='Smoothed (xk|N)' if not bw_on_residual else 'Smoothed on res (xk|N)')
        # plt.legend()

        toc = time.perf_counter()
        idTime = toc - tic
        self.model = model
        self.fltCzKf = fltCzKf
        self.modelBW = modelBW
        self.fltCzKfBW = fltCzKfBW
        self.trained = True
        self.info = {"trained": True, "idTime": idTime}

    def remove_hyper_param_results(self):
        """Removes results from hyperparameter search to reduce model foot print
        These results are only useful for studying the hyperparameter search process,
        and are not needed for using the model with final selected hyperparameters.
        """
        if hasattr(self, "param_sets"):
            self.param_sets = None
        if hasattr(self, "param_results"):
            self.param_results = None

    def _find_hyper_params(self, Y, Z=None, **kwargs):
        n_samples, ny = Y.shape
        # Prepare folds for an inner cross-validation to pick the best n1 and horizons
        iCV_folds = prepare_fold_inds(self.hyper_param_settings["folds"], n_samples)

        if (
            "nx_values" in self.hyper_param_settings
            and self.hyper_param_settings["nx_values"] is not None
        ):
            nx_values = self.hyper_param_settings["nx_values"]
        else:
            nx_values = [self.nx]

        if (
            "n1_values" in self.hyper_param_settings
            and self.hyper_param_settings["n1_values"] is not None
        ):
            n1_values = self.hyper_param_settings["n1_values"]
        else:
            n1_values = [self.n1] if self.n1 is not None else None

        if (
            "iY_values" in self.hyper_param_settings
            and self.hyper_param_settings["iY_values"] is not None
        ):
            iY_values = self.hyper_param_settings["iY_values"]
        else:
            iY_min = max(2, int(np.ceil(np.min(np.array(nx_values)) / ny)))
            iY_max = max(
                iY_min,
                min(
                    self.hyper_param_settings["max_horizon"], int(n_samples / (ny + 1))
                ),
            )
            iY_values = np.unique(
                np.round(10 ** np.linspace(np.log10(iY_min), np.log10(iY_max), 3))
            ).astype(int)

        if Z is not None:
            nz = Z.shape[-1]
            if (
                "iZ_values" in self.hyper_param_settings
                and self.hyper_param_settings["iZ_values"] is not None
            ):
                iZ_values = self.hyper_param_settings["iZ_values"]
            else:
                iZ_min = max(
                    2,
                    int(
                        np.ceil(
                            np.min(
                                np.array(
                                    n1_values if n1_values is not None else nx_values
                                )
                            )
                            / nz
                        )
                    ),
                )
                iZ_max = max(
                    iZ_min,
                    min(
                        self.hyper_param_settings["max_horizon"],
                        int(n_samples / (nz + 1)),
                    ),
                )
                iZ_values = np.unique(
                    np.round(10 ** np.linspace(np.log10(iZ_min), np.log10(iZ_max), 3))
                ).astype(int)
        else:
            iZ_values = [None]

        inference_options = [self.default_inference]
        # inference_options = ['smoothing', 'filtering']
        for nx_ind, nx in enumerate(nx_values):
            if n1_values is not None:
                n1_vals = [n1 for n1 in n1_values if n1 <= nx]
            else:
                n1_vals = np.unique(
                    np.round(10 ** np.linspace(0, np.log10(nx), 10))
                ).astype(int)
            param_sets = []
            for n1 in n1_vals:
                for iY in iY_values:
                    for iZ in iZ_values:
                        if iY * ny < 2:
                            continue
                        if iY * ny < nx:
                            continue
                        if n1 != 0 and iZ * nz < 2:
                            continue
                        if n1 != 0 and iZ * nz < n1:
                            continue
                        for inference in inference_options:
                            param_sets.append(
                                {
                                    "nx": nx,
                                    "n1": n1,
                                    "iY": iY,
                                    "iZ": iZ if iZ is not None else iY,
                                    "i": [iY, iZ] if iZ is not None else iY,
                                    "default_inference": inference,
                                }
                            )

        if len(param_sets) > 1:
            param_results = []
            argsAll = []
            for param_ind, param in enumerate(param_sets):
                logStr = f"set {1+param_ind}/{len(param_sets)}: Considering hyper-parameters: {param}"
                args = {
                    "logStr": logStr,
                    "Z": Z,
                    "Y": Y,
                    "iCV_folds": iCV_folds,
                    "nx": nx,
                    "n1": param["n1"],
                    "i": param["i"],
                    "default_inference": param["default_inference"],
                }
                args.update(kwargs)
                argsAll.append(args)

            min_cpu_for_parallel = self.hyper_param_settings["min_cpu_for_parallel"]
            num_processors = multiprocessing.cpu_count()
            if num_processors < min_cpu_for_parallel:  # To run serially
                results = [trainAndEvalCVedFromArgs(args) for args in argsAll]
            else:
                # To run in parallel
                pool = multiprocessing.Pool(
                    processes=min(len(argsAll), num_processors - 1)
                )  # use all but one of the available processors
                results = pool.map(trainAndEvalCVedFromArgs, argsAll)
                pool.close()
                pool.join()

            for param, fold_results in zip(param_sets, results):
                if np.any([(fr["model"] is None) for fr in fold_results]):
                    logger.info(
                        f"Skipping param: {param} [some folds did not have a result]"
                    )
                    continue
                param_results.append({"param": param, "results": fold_results})
            if len(param_results) == 0:
                raise (
                    Exception(
                        f"Skipping nx={nx}, since no parameter set yielded a result"
                    )
                )
            # Find the best n1
            metrics = ["meanzCC", "meanyCC", "meanzR2", "meanyR2"]
            selection_criteria = self.hyper_param_settings["selection_criteria"]
            perfVals = {
                metric: np.array(
                    [[fr["perf"][metric] for fr in r["results"]] for r in param_results]
                )
                for metric in metrics
            }
            meanPerfVals = {metric: np.mean(perfVals[metric], -1) for metric in metrics}
            best_param_ind = np.argmax(meanPerfVals[selection_criteria])
            param_selected = param_results[best_param_ind]["param"]

            perfStr = ", ".join(
                [
                    f"{metric}={meanPerfVals[metric][best_param_ind]:.3g}"
                    for metric in metrics
                ]
            )
            logger.info(
                f"nx={nx} => Best parameters were: {param_selected} ({perfStr})"
            )
        else:
            param_results = None
            best_param_ind = 0
            param_selected = param_sets[best_param_ind]
            logger.info(f"nx={nx} => Using the only parameter set: {param_selected}")
        return param_selected, param_sets, param_results, best_param_ind

    def predict(self, Y, U=None, **kwargs):
        if isinstance(Y, (list, tuple)):
            for trialInd, trialY in enumerate(Y):
                trialOuts = self.predict(trialY, U=U, **kwargs)
                if trialInd == 0:
                    outs = [[o] for oi, o in enumerate(trialOuts)]
                else:
                    outs = [outs[oi] + [o] for oi, o in enumerate(trialOuts)]
            return tuple(outs)
        outs = self.model.predict(Y, U=U, **kwargs)
        if self.default_inference in ["filtering", "smoothing"]:
            if self.fltCzKf is None:
                raise (Exception("Model was not trained for filtering/smoothing"))
            Zp, Yp, Xp = outs
            Zf = Zp + (self.fltCzKf @ (Y - Yp).T).T
            ZpReturn = Zf
        if self.default_inference in ["smoothing"]:
            if self.modelBW is None:
                raise (Exception("Model was not trained for smoothing"))
            YBW = np.flipud(Y)
            ZpBW, YpBW, XpBW = self.modelBW.predict(YBW)
            ZfBW = ZpBW + (self.fltCzKfBW @ (YBW - YpBW).T).T
            if self.bw_on_residual:
                Zs = Zf + np.flipud(ZfBW)
            else:
                Zs = (Zf + np.flipud(ZfBW)) / 2
            ZpReturn = Zs
        if self.default_inference in ["filtering", "smoothing"]:
            outs = list(outs)
            outs[0] = ZpReturn
            outs = tuple(outs)
        return outs
