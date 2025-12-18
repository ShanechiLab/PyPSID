"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tools for evaluating system identification
"""

import copy, itertools, warnings, logging

import numpy as np
from sklearn import metrics

from .tools import extractDiagonalBlocks, getBlockIndsFromBLKSArray
from .PSID import projOrth

logger = logging.getLogger(__name__)


def evalPrediction(trueValue, prediction, measure, missing_marker=None):
    """Evaluates prediction of data

    Args:
        trueValue (np.array): true values. The first dimension is taken as the sample dimension
            over which metrics are computed.
        prediction (np.array): predicted values
        measure (string): performance measure name
        missing_marker: if not None, will ignore samples with this value (default: None)

    Returns:
        perf (np.array): the value of performance measure, computed for each dimension of data
    """
    if missing_marker is not None:
        if np.isnan(missing_marker):
            isOk = np.all(~np.isnan(prediction), axis=1)
        else:
            isOk = np.all(prediction != missing_marker, axis=1)
        trueValue = copy.deepcopy(trueValue)[isOk, :]
        prediction = copy.deepcopy(prediction)[isOk, :]
    nSamples, nDims = trueValue.shape
    if nSamples == 0:
        perf = np.nan * np.ones((nDims,))
        return perf
    isFlat = (np.max(trueValue, axis=0) - np.min(trueValue, axis=0)) == 0
    if measure == "CC":
        if nSamples < 2:
            perf = np.nan * np.ones((nDims,))
            return perf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = np.corrcoef(trueValue, prediction, rowvar=False)
        perf = np.diag(R[nDims:, :nDims])
        if np.any(
            isFlat
        ):  # This is the fall-back values for flat true signals in the corrcoef code, but it may not detect flats correctly, so we help it
            perf.setflags(write=1)
            perf[isFlat] = np.nan
    elif measure == "R2":
        if nSamples < 2:
            perf = np.nan * np.ones((nDims,))
            return perf
        perf = metrics.r2_score(trueValue, prediction, multioutput="raw_values")
        perf[isFlat] = (
            0  # This is the fall-back values for flat true signals in the r2_score code, but it may not detect flats correctly, so we help it
        )
    elif measure == "MSE":
        perf = metrics.mean_squared_error(
            trueValue, prediction, multioutput="raw_values"
        )
    elif measure == "RMSE":
        MSE = evalPrediction(trueValue, prediction, "MSE")
        perf = np.sqrt(MSE)
    else:
        raise (Exception('Performance measure "{}" is not supported.'.format(measure)))

    return perf


def evaluateDecoding(
    ZTest=None,
    zPredTest=None,
    YTest=None,
    yPredTest=None,
    sId=None,
    missing_marker=None,
    YType=None,
    ZType=None,
    measures=["CC", "RMSE", "R2"],
):
    """Evaluates prediction of data

    Args:
        ZTest (np.array, optional): true values of the z data. Defaults to None.
        zPredTest (np.array, optional): predicted values of the z data. Defaults to None.
        YTest (np.array, optional): true values of the y data. Defaults to None.
        yPredTest (np.array, optional): predicted values of the y data. Defaults to None.
        sId (object, optional): learned model. Defaults to None.
        missing_marker (number, optional): the marker value for missing data. Defaults to None.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        measures (list, optional): list of performance measures to compute when possible.
            Defaults to ['CC', 'NRMSE', 'R2'].

    Returns:
        errs (dict): computed performance measures
    """
    if zPredTest is None and yPredTest is None and sId is not None:
        zPredTest, yPredTest, xPredTest = sId.predict(YTest)

    if zPredTest is not None:
        nonTAx = tuple(np.arange(1, len(zPredTest.shape)))
        zNotBlown = np.all(
            np.logical_not(np.logical_or(np.isnan(zPredTest), np.isinf(zPredTest))),
            axis=nonTAx,
        )
        zNotNaN = np.all(np.logical_not(np.isnan(ZTest)), axis=nonTAx)
        if missing_marker is not None:
            zNotMissing = np.all(ZTest != missing_marker, axis=1)
        else:
            zNotMissing = zNotBlown
        zOk = np.nonzero(
            np.logical_and(zNotNaN, np.logical_and(zNotBlown, zNotMissing))
        )[0]

    if yPredTest is not None:
        nonTAx = tuple(np.arange(1, len(yPredTest.shape)))
        yNotBlown = np.all(
            np.logical_not(np.logical_or(np.isnan(yPredTest), np.isinf(yPredTest))),
            axis=nonTAx,
        )
        yNotNaN = np.all(np.logical_not(np.isnan(YTest)), axis=nonTAx)
        if missing_marker is not None:
            yNotMissing = np.all(YTest != missing_marker, axis=1)
        else:
            yNotMissing = yNotBlown
        yOk = np.nonzero(
            np.logical_and(yNotNaN, np.logical_and(yNotBlown, yNotMissing))
        )[0]

    errs = {}

    for m in measures:
        if zPredTest is not None:
            if (
                len(zPredTest.shape) == 2
                and m not in ["AUC", "ACC", "ACCD1", "CM"]
                and (m != "PoissonLL" or ZType == "count_process")
            ):
                errs[m] = evalPrediction(ZTest[zOk, :], zPredTest[zOk, :], m)
                errs["mean" + m] = np.mean(errs[m])
            elif len(zPredTest.shape) == 3 and m in ["AUC", "ACC", "ACCD1", "CM"]:
                errs[m] = evalPrediction(ZTest[zOk, :], zPredTest[zOk, :, :], m)
                if m == "CM":
                    if zOk.size == 0:
                        errs[m] = (
                            np.ones(
                                (
                                    zPredTest.shape[2],
                                    zPredTest.shape[2],
                                    zPredTest.shape[1],
                                )
                            )
                            * np.nan
                        )
                    errs["mean" + m] = np.mean(errs[m], axis=2)
                else:
                    errs["mean" + m] = np.mean(errs[m])

        if yPredTest is not None:
            if (
                len(yPredTest.shape) == 2
                and m not in ["AUC", "ACC", "ACCD1", "CM"]
                and (m != "PoissonLL" or YType == "count_process")
            ):
                errs["y" + m] = evalPrediction(YTest[yOk, :], yPredTest[yOk, :], m)
                errs["meany" + m] = np.mean(errs["y" + m])

    return errs


def computeEigIdError(trueEigs, idEigVals, measure="NFN"):
    """Computes the error between a true and learned set of eigenvalues.
    Finds the closest pairing of learned and true values that minimizes the error

    Args:
        trueEigs (list): true eigenvalues
        idEigVals (list): sets of learned eigenvalues
        measure (str, optional): name of performance measure to compute. Defaults to 'NFN'.

    Returns:
        allEigError (list): computed error for each set of learned eigenvalue
    """

    permut = itertools.permutations(trueEigs)

    def computeErr(trueValue, prediction, measure):
        """Computes error between true and predicted values.

        Args:
            trueValue: Ground truth.
            prediction: Predicted values.
            measure: Error measure type ('NFN').

        Returns:
            float: Computed error.
        """
        if measure == "NFN":
            perf = np.sqrt(
                np.sum(np.abs(prediction - trueValue) ** 2, axis=1)
            ) / np.sqrt(np.sum(np.abs(trueValue) ** 2, axis=1))
        else:
            raise (Exception("Not supported"))
        return perf

    allEigError = np.array([])
    for i in range(len(idEigVals)):
        eigVals = idEigVals[i]
        if len(eigVals) > 0:
            errorVals = np.array([])
            for p in permut:
                errorVals = np.append(
                    errorVals,
                    computeErr(np.atleast_2d(p), np.atleast_2d(eigVals), measure),
                )

            pmi = np.argmin(errorVals)
            allEigError = np.append(allEigError, errorVals[pmi])
    return allEigError


def computeLSSMIdError(sTrue, sId_in):
    """Computes the parameter learning error and eigenvalue error for a learned LSSM

    Args:
        sTrue (LSSM): true model
        sId_in (LSSM): learned model

    Returns:
        errs (dict): computed error metrics
    """

    errs = {}
    if hasattr(sId_in, "state_dim") and sTrue.state_dim == sId_in.state_dim:
        sId = copy.deepcopy(sId_in)  # Do not modify the original object
        E = sId.makeSimilarTo(
            sTrue
        )  # TEMP CM!!! This was needed to be commented temporarily to test something for avoiding error

        def matrixErrNorm(trueX, idX):
            """Computes normalized Frobenius norm error between matrices.

            Args:
                trueX: True matrix.
                idX: Identified/Estimated matrix.

            Returns:
                float: Normalized error.
            """
            errX = idX - trueX
            errN = np.nan
            try:
                norm = np.linalg.norm(trueX, ord="fro")
                if norm != 0:
                    errN = np.linalg.norm(errX, ord="fro") / norm
            except Exception as e:
                print(e)
            return errN

        for field in dir(sTrue):
            valTrue = sTrue.__getattribute__(field)
            if not field.startswith("__") and isinstance(valTrue, np.ndarray):
                if field in dir(sId):
                    valId = sId.__getattribute__(field)
                    if isinstance(valId, np.ndarray) and valTrue.shape == valId.shape:
                        errFieldName = "{}ErrNormed".format(field)
                        try:
                            errs[errFieldName] = matrixErrNorm(
                                np.atleast_2d(valTrue), np.atleast_2d(valId)
                            )
                        except Exception as e:
                            print(e)
                            pass

    # Eigenvalue error
    nz = len(sTrue.zDims)
    if hasattr(sId_in, "zDims") and len(sId_in.zDims) > 0:
        try:
            subATrue = sTrue.A[
                np.ix_(np.array(sTrue.zDims) - 1, np.array(sTrue.zDims) - 1)
            ]
            sId_zDims = sId_in.zDims[: min([nz, len(sId_in.zDims)])]
            subAId = sId_in.A[np.ix_(np.array(sId_zDims) - 1, np.array(sId_zDims) - 1)]
            trueZEigs = np.linalg.eig(subATrue)[0]
            idZEigs = np.linalg.eig(subAId)[0]
            if len(idZEigs) < len(trueZEigs):
                idZEigs = np.hstack(
                    (idZEigs, np.zeros((len(trueZEigs) - len(idZEigs))))
                )
            if len(trueZEigs) < 9:
                errs["zEigErrNFN"] = computeEigIdError(trueZEigs, [idZEigs], "NFN")[0]
            else:
                errs["zEigErrNFN"] = np.nan  # Too slow to compute this
        except Exception as e:
            print(e)
            pass

    return errs


def evalSysId(
    sId,
    YTest=None,
    ZTest=None,
    YTrain=None,
    ZTrain=None,
    UTest=None,
    UTrain=None,
    trueSys=None,
    YType=None,
    ZType=None,
    useXFilt=False,
    missing_marker=None,
    undo_scaling=False,
):
    """Evaluates a learned model based on predictions and also in terms of model parameters if the true model is known.

    Args:
        sId (PredictorModel): a model that implements a predict method.
        YTest (np.array, optional): input test data. Defaults to None.
        ZTest (np.array, optional): output test data. Defaults to None.
        YTrain (np.array, optional): input training data. Defaults to None.
        ZTrain (np.array, optional): output training data. Defaults to None.
        UTest (np.array, optional): external input test data. Defaults to None.
        UTrain (np.array, optional): external training test data. Defaults to None.
        trueSys (LSSM, optional): true model, if known in simulations. Defaults to None.
        YType (string, optional): data type of Y. Defaults to None.
        ZType (string, optional): data type of Z. Defaults to None.
        useXFilt (bool, optional): if true, will pass to predict if the
            model supports that argument (i.e. is an LSSM). Defaults to False.
        missing_marker (numpy value, optional): indicator of missing samples in data.
            Is used in performing and undoing preprocessing. Defaults to None.
        undo_scaling (bool, optional): if true, will apply the inverse scaling
            on predictions. Defaults to False.

    Returns:
        perf (dict): computed performance measures
        zPredTest (np.array): predicted Z
        yPredTest (np.array): predicted Y
        xPredTest (np.array): latent state X
    """
    perf = {}
    zPredTest, yPredTest, xPredTest = None, None, None
    if trueSys is not None:
        try:
            if type(sId) is type(trueSys):
                sIdLSSM = copy.deepcopy(sId)
            else:
                sIdLSSM = sId.getLSSM()

            perf = computeLSSMIdError(trueSys, sIdLSSM)
        except Exception as e:
            logger.warning(e)

    if YTest is not None:
        zPredTest, yPredTest, xPredTest = sId.predict(YTest)
        perfD = {}
        steps_ahead = [1]
        if hasattr(sId, "steps_ahead") and sId.steps_ahead is not None:
            steps_ahead = sId.steps_ahead
        for saInd, step_ahead in enumerate(steps_ahead):
            perfDThis = evaluateDecoding(
                ZTest=ZTest,
                zPredTest=zPredTest[saInd],
                YTest=YTest,
                yPredTest=yPredTest[saInd],
                YType=YType,
                ZType=ZType,
                missing_marker=missing_marker,
            )
            for key, val in perfDThis.items():
                new_key = key if step_ahead == 1 else f"{key}_{step_ahead}step"
                perfD[new_key] = val

        for key, val in perfD.items():
            perf[key] = val

    return perf, zPredTest, yPredTest, xPredTest
