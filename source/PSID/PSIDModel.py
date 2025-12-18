"""Omid Sani, Shanechi Lab, University of Southern California, 2023"""

"A class for for training PSID models"

import logging
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from .PSID import PSID as PSIDFunc
from .PSID import blkhankskip
from .ReducedRankRegressor import ReducedRankRegressor
from .evaluation import evalPrediction
from .tools import applyFuncIf, catIf, prepare_fold_inds, subtractIf, transposeIf

logger = logging.getLogger(__name__)


def evalPerformanceMetrics(
    trueVals, predVals, prefix="", metrics=["CC", "R2"], add_mean=True
):
    """
    Evaluates performance metrics between true and predicted values.

    Args:
        trueVals (np.ndarray): The ground truth values.
        predVals (np.ndarray): The predicted values.
        prefix (str, optional): A prefix to add to the metric names in the output dictionary. Defaults to "".
        metrics (list, optional): A list of metrics to compute. Supported: "CC", "R2". Defaults to ["CC", "R2"].
        add_mean (bool, optional): Whether to add the mean of each metric across columns to the output dictionary. Defaults to True.

    Returns:
        dict: A dictionary containing the computed performance metrics.
    """
    perf = {}
    for metric in metrics:
        perf[prefix + metric] = evalPrediction(trueVals, predVals, metric)
        if add_mean:
            perf["mean" + prefix + metric] = np.nanmean(perf[prefix + metric])
    return perf


def trainAndEvalFold(
    Z, Y, fold, nx, n1, i, default_inference, zShift=None, logStr=None, **kwargs
):
    """
    Trains and evaluates a PSID model on a single fold of data.

    Args:
        Z (np.ndarray or list): The behavioral data.
        Y (np.ndarray or list): The neural data.
        fold (dict): A dictionary with 'train_inds' and 'test_inds' for the fold.
        nx (int): Total number of states.
        n1 (int): Number of states to be extracted in the first stage.
        i (int or list): Horizon(s) for constructing Hankel matrices.
        default_inference (str): The default inference mode ('prediction', 'filtering', 'smoothing').
        zShift (int, optional): Shift to apply to Z data. Defaults to None.
        logStr (str, optional): Logging string. Defaults to None.
        **kwargs: Additional arguments for PSIDModel and its fit method.

    Returns:
        dict: A dictionary containing the trained 'model' and performance 'perf'.
    """
    if isinstance(Z, (list, tuple)):
        ZTrain = [Z[i] for i in range(len(Z)) if i in fold["train_inds"]]
        YTrain = [Y[i] for i in range(len(Y)) if i in fold["train_inds"]]
    else:
        ZTrain = Z[fold["train_inds"], :]
        YTrain = Y[fold["train_inds"], :]
    try:
        model = PSIDModel(
            nx=nx, n1=n1, i=i, default_inference=default_inference, zShift=zShift
        )
        if "time_first" in kwargs:
            kwargs["time_first"] = True
        model.fit(YTrain, ZTrain, enable_all_inference=False, **kwargs)
        model.verbose = False
        if isinstance(Z, (list, tuple)):
            ZTest = [Z[i] for i in range(len(Z)) if i in fold["test_inds"]]
            YTest = [Y[i] for i in range(len(Y)) if i in fold["test_inds"]]
        else:
            YTest = Y[fold["test_inds"], :]
            ZTest = Z[fold["test_inds"], :]
        allZp, allYp, allXp = model.predict(YTest)

        perf = {}
        perf.update(evalPerformanceMetrics(ZTest, allZp, prefix="z"))
        perf.update(evalPerformanceMetrics(YTest, allYp, prefix="y"))
    except Exception as e:
        print(f"Error in fitting model: {e}")
        model = None
        perf = None
    return {"model": model, "perf": perf}


def trainAndEvalCVed(Z, Y, iCV_folds, *args, **kwargs):
    """
    Performs cross-validated training and evaluation of PSID models.

    Args:
        Z (np.ndarray or list): The behavioral data.
        Y (np.ndarray or list): The neural data.
        iCV_folds (list): A list of folds for cross-validation.
        *args: Positional arguments to be passed to trainAndEvalFold.
        **kwargs: Keyword arguments to be passed to trainAndEvalFold.

    Returns:
        list: A list of dictionaries, where each dictionary contains the results for a fold.
    """
    fold_results = []
    for foldInd, inner_fold in enumerate(iCV_folds):
        if "logStr" in kwargs and kwargs["logStr"] is not None:
            logger.info(f"iCV Fold {1+foldInd}/{len(iCV_folds)} " + kwargs["logStr"])
        fold_res = trainAndEvalFold(Z, Y, inner_fold, *args, **kwargs)
        fold_results.append(fold_res)
    return fold_results


def trainAndEvalCVedFromArgs(args):
    """
    Wrapper for trainAndEvalCVed to be used with multiprocessing, taking a single dictionary of arguments.

    Args:
        args (dict): A dictionary of arguments for trainAndEvalCVed.

    Returns:
        list: The output of trainAndEvalCVed.
    """
    return trainAndEvalCVed(**args)


def updateFilteringParam(model, YRes, ZRes, iZ, *args, **kwargs):
    """
    Updates the filtering parameters of a given LSSM model based on residuals.

    This function estimates the Kalman gain for filtering based on the correlation
    between the innovation (Y residuals) and the state estimation error (Z residuals).

    Args:
        model (LSSM): The LSSM model to update.
        YRes (np.ndarray): The residuals of the neural data (Y - Y_predicted).
        ZRes (np.ndarray): The residuals of the behavioral data (Z - Z_predicted).
        iZ (int): The horizon for the behavioral data to construct the Hankel matrix.
        *args: Additional arguments for updateFilteringParamGivenCzKf.
        **kwargs: Additional keyword arguments for updateFilteringParamGivenCzKf.

    Returns:
        np.ndarray: The estimated observability matrix multiplied by the Kalman filter gain (fltGzKf).
    """
    # fltCzKf = PSID.projOrth((Z - Zp).T, (Y - Yp).T)[1] # => does not enforce correct rank for CzKf
    nx = model.state_dim
    ny = YRes.shape[-1]
    nz = ZRes.shape[-1] if ZRes is not None else 0
    if nz > 0:
        # Form a Hankel matrix from ZRes
        rank = np.min([nx, ny, nz * iZ])
        # ZResTmp = 1.1+np.arange(20, dtype=ZRes.dtype)[:, np.newaxis] + 0.1*np.arange(2, dtype=ZRes.dtype)[np.newaxis, :]
        # YResTmp = 1.1+np.arange(20, dtype=ZRes.dtype)[:, np.newaxis] + 0.1*np.arange(2, dtype=ZRes.dtype)[np.newaxis, :]
        ZHank = blkhankskip(ZRes, iZ, time_first=True).T
        # if iZ == 1:  # Test
        #     np.testing.assert_array_equal(ZHank, ZRes)
        YHank = YRes[
            : ZHank.shape[0], :
        ]  # blkhankskip(YRes,  1, j=ZHank.shape[0], time_first=True).T
        RRR = ReducedRankRegressor(YHank, ZHank, rank)
        fltGzKf = np.array(RRR.W @ RRR.A)
        # Test:
        # RRR2 = ReducedRankRegressor(YRes, ZRes, np.min( [nx, ny, nz] ))
        # fltCzKf = np.array(RRR2.W @ RRR2.A)
        # np.testing.assert_allclose(fltGzKf[:nz, :], fltCzKf, rtol=1e-3)
    else:
        fltGzKf = model.Cz @ model.Kf
    updateFilteringParamGivenCzKf(model, fltGzKf, *args, **kwargs)
    return fltGzKf


def updateFilteringParamGivenCzKf(model, fltGzKf, updateCz=False, updateKfKv=False):
    """Given some observability matrix from (Cz,A) times Kf

    Args:
        model (LSSM): LSSM model object
        fltGzKf (numpy ndarray): empirically computed observability matrix for the pair Gz=(Cz,A) times Kf (or fltGzKf)
        updateCz (bool, optional): If True, will update Cz to make sure it matches what was used to compute Kf. This should not matter. Defaults to False.
        updateKfKv (bool, optional): If True, will update the Kf field in the model. Defaults to False.

    Returns:
        _type_: _description_
    """
    nx = model.state_dim
    nz = model.Cz.shape[0] if hasattr(model, "Cz") and model.Cz is not None else 0
    if (updateCz or updateKfKv) and nz > 0:
        GzKfRank = np.linalg.matrix_rank(fltGzKf)
        n = int(fltGzKf.shape[0] / nz)  # Number of blocks in Gz (z observability)
        Gz = model.Cz
        for i in range(n - 1):
            Gz = np.concatenate((Gz, Gz[(i * nz) : (i + 1) * nz, :] @ model.A))
        # Gz = [Cz; CzA; CzA^2; ...; CzA^(n-1)]
        Kf, fltGzKfErr = np.linalg.lstsq(Gz, fltGzKf, rcond=None)[
            :2
        ]  # Solution ? for Gz @ ? = fltGzKf, or Kf = np.linalg.pinv(Gz) @ fltGzKf
        if GzKfRank < np.min(fltGzKf.shape):
            logger.info(
                f"rank( obs(Cz,A)*Kf ) < full ({np.min(fltGzKf.shape)}), so using approximate least error norm solution for Kf"
            )
        Cz = Gz[:nz, :]
        if updateCz:
            model.Cz = Cz
        if updateKfKv:
            model.Kf = Kf
            model.Kv = model.K - model.A @ model.Kf
    return fltGzKf


def get_shape(Y):
    """Returns shape of array, or of a list of array, returns the shape of the first element on the list

    Args:
        Y (list or array): array or list of arrays

    Returns:
        np.array: shape of array
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        if len(Y) > 0:
            return Y[0].shape
        else:
            return None
    else:
        return Y.shape


class PSIDModel:
    """Class for identifying and using Preferential Subspace Identification (PSID) models.

    Provides methods for fitting models to data, hyperparameter selection, and prediction (filtering/smoothing).
    """

    def __init__(
        self,
        nx=None,  # Total number of states
        n1=None,  # Number of states to be extracted in the first stage (behaviorally relevant)
        i=None,  # An array of [iY, iZ]. Will be ignored if iY or iZ is not None
        iY=None,  # Neural horizon
        iZ=None,  # Behavior horizon
        hyper_param_settings=None,  # Settings for hyper-parameter selection
        default_inference="prediction",  # Can also be 'filtering' or 'smoothing',
        bw_on_residual=None,
        updateKfKv=None,
        zShift=None,  # A shift to apply as preprocessing during learning and undo during prediction. if zShift=1, the normal PSID would attempt to optimize and return z{k|k} instead of the default z{k|k-1}.
        steps_ahead=None,  # Number of steps ahead for prediction.
    ):
        """
        Initializes the PSIDModel.

        This class is a wrapper for the PSID algorithm, providing methods for
        hyperparameter tuning, fitting, and prediction.

        Args:
            nx (int, optional): Total number of states. If None, it will be tuned. Defaults to None.
            n1 (int, optional): Number of behaviorally relevant states. If None, it will be tuned. Defaults to None.
            i (list or int, optional): Horizon(s) [iY, iZ]. Used if iY and iZ are not provided. Defaults to None.
            iY (int, optional): Neural data horizon. If None, it will be tuned. Defaults to None.
            iZ (int, optional): Behavioral data horizon. If None, it will be tuned. Defaults to None.
            hyper_param_settings (dict, optional): Settings for hyperparameter tuning. Defaults to None.
            default_inference (str, optional): Default prediction mode. Can be 'prediction', 'filtering', or 'smoothing'. Defaults to "prediction".
            bw_on_residual (bool, optional): Whether to fit the backward model on residuals for smoothing. Defaults to True.
            updateKfKv (bool, optional): Whether to update Kf and Kv from estimated Kalman gain. Defaults to False.
            zShift (int, optional): A shift for behavioral data Z. Can be used for lag compensation. Defaults to None.
            steps_ahead (int, optional): Number of steps ahead for prediction. Defaults to None.
        """
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
        if bw_on_residual is None:
            bw_on_residual = True  # By default, fit backward model based on residuals
        self.bw_on_residual = bw_on_residual
        if updateKfKv is None:
            updateKfKv = False  # By default, do not attempt to update Kf and Kv steady state Kalman gains of the learned model
        self.updateKfKv = updateKfKv
        self.zShift = zShift
        self.steps_ahead = steps_ahead
        self.info = {}

    def set_default_inference(self, default_inference):
        """
        Sets the default inference mode.

        Args:
            default_inference (str): The inference mode. Must be one of 'prediction', 'filtering', 'smoothing'.

        Raises:
            Exception: If the provided inference mode is not supported.
        """
        allowed = ["prediction", "filtering", "smoothing"]
        if default_inference not in allowed:
            raise (
                Exception(
                    f'Default inference of "{default_inference}" not supported. Must be one of: {allowed}'
                )
            )
        self.default_inference = default_inference

    def get_horizons(self):
        """
        Gets the neural and behavioral horizons (iY, iZ).

        Returns:
            tuple[int, int]: A tuple containing the neural horizon (iY) and behavioral horizon (iZ).
        """
        iY = self.iY
        iZ = self.iZ if self.iZ is not None else self.iY
        if iY is None and iZ is None and self.i is not None:
            iY = self.i[0]
            iZ = self.i[-1]
        return iY, iZ

    def fit(self, Y, Z=None, enable_all_inference=None, time_first=True, **kwargs):
        """
        Fits the PSID model to the data.

        If hyperparameters (nx, n1, iY, iZ) are not specified during initialization,
        this method will first perform a hyperparameter search. Then, it fits the
        main PSID model. If `enable_all_inference` is True or `default_inference`
        is 'filtering' or 'smoothing', it also fits models required for these
        inference types.

        Args:
            Y (np.ndarray or list): Neural data.
            Z (np.ndarray or list, optional): Behavioral data. Defaults to None.
            enable_all_inference (bool, optional): If True, enables filtering and smoothing capabilities
                by fitting additional necessary models. Defaults to `self.zShift is None or self.zShift == 0`.
            time_first (bool, optional): Whether the time dimension is the first axis of Y and Z. Defaults to True.
            **kwargs: Additional arguments passed to the PSID function.
        """
        if enable_all_inference is None:
            enable_all_inference = self.zShift is None or self.zShift == 0

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

        if Z is not None and self.zShift is not None and self.zShift != 0:
            ZTrain = (
                np.roll(Z, self.zShift, axis=0)
                if not isinstance(Z, list)
                else [np.roll(zt, self.zShift, axis=0) for zt in Z]
            )
            if enable_all_inference:
                raise (Exception(f"Not supported!"))
        else:
            ZTrain = Z
        iY, iZ = self.get_horizons()
        tic = time.perf_counter()
        model = PSIDFunc(
            Y=Y,
            Z=ZTrain,
            nx=self.nx,
            n1=self.n1,
            i=[iY, iZ],
            time_first=time_first,
            **kwargs,
        )
        model.zDims = np.arange(1, 1 + min([self.n1, self.nx]))

        nz = get_shape(Z)[1] if Z is not None else 0
        if enable_all_inference or self.default_inference in ["filtering", "smoothing"]:
            YTF = Y if time_first else transposeIf(Y)
            ZTF = Z if time_first else transposeIf(Z)
            # Regression to find optimal Cz * Kf for K|k filtering
            Zp, Yp, Xp = model.predict(YTF)
            fltGzKf = updateFilteringParam(
                model,
                catIf(subtractIf(YTF, Yp), axis=0),
                catIf(subtractIf(ZTF, Zp), axis=0),
                iZ=self.nx,
                updateKfKv=self.updateKfKv,
            )
            fltCzKf = fltGzKf[:nz, :]
            if isinstance(YTF, (list, tuple)):
                Zf = [Zp[i] + (fltCzKf @ (YTF[i] - Yp[i]).T).T for i in range(len(YTF))]
            else:
                Zf = Zp + (fltCzKf @ (YTF - Yp).T).T
        else:
            fltCzKf, fltGzKf = None, None

        if enable_all_inference or self.default_inference in ["smoothing"]:
            # Time-reversed model for optimal backward pass for K|N smoothing
            YBW = applyFuncIf(YTF, np.flipud)
            ZBW = (
                applyFuncIf(subtractIf(ZTF, Zf), np.flipud)
                if self.bw_on_residual
                else applyFuncIf(ZTF, np.flipud)
            )
            modelBW = PSIDFunc(
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
            fltGzKfBW = updateFilteringParam(
                modelBW,
                catIf(subtractIf(YBW, YpBW), axis=0),
                catIf(subtractIf(ZBW, ZpBW), axis=0),
                iZ=self.nx,
                updateKfKv=self.updateKfKv,
            )
            fltCzKfBW = fltGzKfBW[:nz, :]
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
            modelBW, fltCzKfBW, fltGzKfBW = None, None, None

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
        self.fltGzKf = fltGzKf
        self.modelBW = modelBW
        self.fltCzKfBW = fltCzKfBW
        self.fltGzKfBW = fltGzKfBW
        self.trained = True
        self.info = {"trained": True, "idTime": idTime}

    def remove_hyper_param_results(self):
        """Removes results from hyperparameter search to reduce model footprint
        These results are only useful for studying the hyperparameter search process,
        and are not needed for using the model with final selected hyperparameters.
        """
        if hasattr(self, "param_sets"):
            self.param_sets = None
        if hasattr(self, "param_results"):
            self.param_results = None

    def _find_hyper_params(self, Y, Z=None, **kwargs):
        """
        Finds the best hyperparameters using cross-validation.

        The hyperparameters searched are nx, n1, iY, and iZ. The search ranges can be
        specified in `self.hyper_param_settings`. The best parameters are selected based on
        `self.hyper_param_settings['selection_criteria']`.

        Args:
            Y (np.ndarray or list): Neural data.
            Z (np.ndarray or list, optional): Behavioral data. Defaults to None.
            **kwargs: Additional arguments passed to the training function.

        Returns:
            tuple: A tuple containing:
                - dict: The selected best hyperparameters.
                - list: A list of all parameter sets that were tried.
                - list: A list of results for each parameter set.
                - int: The index of the best parameter set.
        """
        n_samples, ny = get_shape(Y)
        # Prepare folds for an inner cross-validation to pick the best n1 and horizons
        if isinstance(Y, (list, tuple)):
            n_samples_for_folds = len(Y)
        else:
            n_samples_for_folds = n_samples
        iCV_folds = prepare_fold_inds(
            self.hyper_param_settings["folds"], n_samples_for_folds
        )

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
            nz = get_shape(Z)[-1]
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
                    "zShift": self.zShift,
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

    def getLSSM(self):
        """
        Returns the learned forward linear state-space model (LSSM).

        Returns:
            LSSM: The forward LSSM model object.
        """
        model = self.model
        if hasattr(self, "fltCzKf"):
            setattr(model, "fltCzKf", self.fltCzKf)
        return model  # Only forward model

    def getBWLSSM(self):
        """
        Returns the learned backward linear state-space model (LSSM) used for smoothing.

        Returns:
            LSSM: The backward LSSM model object, or None if not trained.
        """
        modelBW = self.modelBW
        if modelBW is not None and hasattr(self, "fltCzKfBW"):
            setattr(modelBW, "fltCzKf", self.fltCzKfBW)
        return modelBW  # Only backward model

    def predict(self, Y, U=None, steady_state=True, **kwargs):
        """
        Predicts outputs from the input data.

        This method performs prediction, filtering, or smoothing based on the
        `self.default_inference` mode.

        Args:
            Y (np.ndarray or list): Input neural data.
            U (np.ndarray or list, optional): Exogenous inputs. Defaults to None.
            steady_state (bool, optional): Whether to use steady-state Kalman gains. Defaults to True.
            **kwargs: Additional arguments for the underlying model's predict method.

        Returns:
            tuple: A tuple of predicted values. The first element is the predicted Z,
                   followed by other model-specific outputs (e.g., predicted Y, states).
                   If Y is a list of trials, a tuple of lists of outputs is returned.
        """
        if isinstance(Y, (list, tuple)):
            for trialInd, trialY in enumerate(Y):
                trialOuts = self.predict(trialY, U=U, **kwargs)
                if trialInd == 0:
                    outs = [[o] for oi, o in enumerate(trialOuts)]
                else:
                    outs = [outs[oi] + [o] for oi, o in enumerate(trialOuts)]
            return tuple(outs)
        return_state_cov = not self.bw_on_residual and not steady_state
        outs = self.model.predict(Y, U=U, return_state_cov=return_state_cov, **kwargs)
        if hasattr(self, "steps_ahead") and self.steps_ahead is not None:
            steps_ahead = self.steps_ahead
        else:
            steps_ahead = [1]
        if outs[0] is not None and self.zShift is not None:
            Zf = np.roll(outs[0], -self.zShift, axis=0)
            outs = list(outs)
            outs[0] = Zf
            outs = tuple(outs)
        if self.default_inference in ["filtering", "smoothing"]:
            if steps_ahead != [1]:
                raise (
                    Exception(
                        f"Multi-step ahead with filtering/smoothing is not supported yet"
                    )
                )
            if not hasattr(self, "fltCzKf") or self.fltCzKf is None:
                raise (Exception("Model was not trained for filtering/smoothing"))
            Zp, Yp, Xp = outs[:3]
            Zf = Zp + (self.fltCzKf @ (Y - Yp).T).T
            ZpReturn = Zf
        if self.default_inference in ["smoothing"]:
            if self.modelBW is None:
                raise (Exception("Model was not trained for smoothing"))
            YBW = np.flipud(Y)
            outsBW = self.modelBW.predict(
                YBW, return_state_cov=return_state_cov, **kwargs
            )
            ZpBW, YpBW, XpBW = outsBW[:3]
            ZfBW = ZpBW + (self.fltCzKfBW @ (YBW - YpBW).T).T
            if self.bw_on_residual:
                Zs = Zf + np.flipud(ZfBW)
            else:
                if not steady_state:
                    fwCovs = outs[3:]
                    bwCovs = outsBW[3:]
                else:
                    fwCovs = self.model.Pp
                    bwCovs = self.modelBW.Pp
                Zs = (Zf + np.flipud(ZfBW)) / 2
            ZpReturn = Zs
        if self.default_inference in ["filtering", "smoothing"]:
            outs = list(outs)
            outs[0] = ZpReturn
            outs = tuple(outs)
        return outs
