"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tests the PSIDModel class
"""

import unittest
import sys, os, copy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scipy import linalg
import numpy as np

from PSID.PSIDModel import PSIDModel
from PSID.PrepModel import PrepModel
from PSID.sim_tools import getSysSettingsFromSysCode, generateRandomLinearModel
from PSID.evaluation import evalPrediction, evalSysId, computeLSSMIdError

numTests = 10  # Increase this for a slower but more thorough test


class TestPSID(unittest.TestCase):
    """Unit tests for the PSIDModel class."""

    def test_PSID(self):
        """Tests the main PSID algorithm on random linear models."""
        np.random.seed(42)

        sysCode = "nyR1_10_nzR1_10_NxR1_2_N1R0_2"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        N = int(1e4)
        iY, iZ = 9, 11  # Neural/Behavior Horizons

        failInds = []
        failErrs = []
        # numTests = 10
        for ci in range(numTests):
            with self.subTest(ci=ci):
                sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                s = copy.deepcopy(sOrig)

                Y, X = s.generateRealizationWithKF(N)
                Z = s.generateObservationFromStates(X, u=None, param_names=["Cz", "Dz"])

                sId = PSIDModel(nx=s.state_dim, n1=s.zDims.size, iY=iY, iZ=iZ)
                sId.fit(Y, Z, time_first=True)
                err = evalSysId(sId, Y, Z, Y, Z, trueSys=s)[0]
                err.update(computeLSSMIdError(s, sId.model))

                params = [
                    "AErrNormed",
                    "CErrNormed",
                    "GErrNormed",
                    "YCovErrNormed",
                    "KErrNormed",
                ]
                if s.zDims.size > 0:
                    params.extend(["CzErrNormed", "zEigErrNFN"])

                # Test that identification works even with non-zero mean signals
                YMean = np.random.randn(Y.shape[-1]) * 100
                ZMean = np.random.randn(Z.shape[-1]) * 100
                Y = Y - np.mean(Y, axis=0)
                Z = Z - np.mean(Z, axis=0)
                sId3 = PSIDModel(nx=s.state_dim, n1=s.zDims.size, iY=iY, iZ=iZ)
                sId3.fit(Y + YMean, Z + ZMean, time_first=True)
                sWithMeans = copy.deepcopy(s)
                sWithMeans.YPrepModel = PrepModel(YMean, remove_mean=True)
                sWithMeans.ZPrepModel = PrepModel(ZMean, remove_mean=True)
                err3 = evalSysId(
                    sId3, Y + YMean, Z + ZMean, Y + YMean, Z + ZMean, trueSys=sWithMeans
                )[0]
                err3.update(computeLSSMIdError(sWithMeans, sId3.model))

                # Test that it also works if time_first = False
                sId4 = PSIDModel(nx=s.state_dim, n1=s.zDims.size, iY=iY, iZ=iZ)
                sId4.fit((Y + YMean).T, (Z + ZMean).T, time_first=False)
                err4 = evalSysId(
                    sId4, Y + YMean, Z + ZMean, Y + YMean, Z + ZMean, trueSys=sWithMeans
                )[0]
                err4.update(computeLSSMIdError(sWithMeans, sId4.model))

                # Test decoding
                allZp, allYp, allXp = sId3.predict(Y + YMean, U=None)
                innovCovEmp = np.cov(allYp - Y, rowvar=False)
                # np.testing.assert_allclose(innovCovEmp, s.innovCov, rtol=1e-1)

                sId3_cp = copy.deepcopy(sId3)
                sId3_cp.model.YPrepModel = None
                sId3_cp.model.ZPrepModel = None
                allZp2, allYp2, allXp2 = sId3_cp.predict(Y, U=None)

                # Test that it also works with trial based data
                sId2 = PSIDModel(nx=s.state_dim, n1=s.zDims.size, iY=iY, iZ=iZ)
                num_trials = 10
                trial_samples = list(
                    np.random.randint(
                        0.9 * N / num_trials, 1.1 * N / num_trials, num_trials - 1
                    )
                )
                trial_samples.append(N - sum(trial_samples))
                YTrials = [Y[: trial_samples[0], :]] + [
                    Y[
                        np.cumsum(trial_samples)[tInd - 1] : np.cumsum(trial_samples)[
                            tInd
                        ],
                        :,
                    ]
                    for tInd, samples in enumerate(trial_samples)
                    if tInd > 0
                ]
                ZTrials = [Z[: trial_samples[0], :]] + [
                    Z[
                        np.cumsum(trial_samples)[tInd - 1] : np.cumsum(trial_samples)[
                            tInd
                        ],
                        :,
                    ]
                    for tInd, samples in enumerate(trial_samples)
                    if tInd > 0
                ]
                sId2.fit(YTrials, ZTrials, time_first=True)
                err2 = evalSysId(sId2, Y, Z, Y, Z, trueSys=s)[0]
                err2.update(computeLSSMIdError(s, sId2.model))

                try:
                    # Verify identification errors are within tolerance
                    np.testing.assert_array_less(
                        [err[p] for p in params],
                        5e-2,
                        err_msg="Error too large for some params {}".format(params),
                    )
                    np.testing.assert_array_less(
                        [err2[p] for p in params],
                        5e-2,
                        err_msg="Error too large for some params {}".format(params),
                    )
                    np.testing.assert_array_less(
                        [err3[p] for p in params],
                        5e-2,
                        err_msg="Error too large for some params {}".format(params),
                    )
                    np.testing.assert_array_less(
                        [err4[p] for p in params],
                        5e-2,
                        err_msg="Error too large for some params {}".format(params),
                    )

                    # Verify decoding accuracy
                    np.testing.assert_allclose(allXp2, allXp, rtol=1e-3)
                    np.testing.assert_allclose(allZp2, allZp - ZMean, atol=1e-6)
                    np.testing.assert_allclose(allYp2, allYp - YMean, atol=1e-6)
                except Exception as e:
                    failInds.append(ci)
                    failErrs.append(e)

        if len(failInds) > 0:
            raise (
                Exception(
                    "{} => {}/{} random systems (indices: {}) failed: \n{}".format(
                        self.id(), len(failInds), numTests, failInds, failErrs
                    )
                )
            )
        else:
            print(
                "{} => Ok: Tested with {} random systems, all were ok!".format(
                    self.id(), numTests
                )
            )

    def test_PSIDModelFit_doest_change_inputs(self):
        """Tests that fitting the model does not modify the input data."""
        np.random.seed(42)

        N = int(1e3)

        horizon = 5

        # numTests = 10
        for ci in range(numTests):
            ny, nz, nu, nx, n1 = np.random.randint(1, 10, 5)

            YMean = 10 * np.random.randn(1, ny)
            ZMean = 10 * np.random.randn(1, nz)
            UMean = 10 * np.random.randn(1, nu)

            YStd = 10 * np.random.randn(1, ny)
            ZStd = 10 * np.random.randn(1, nz)
            UStd = 10 * np.random.randn(1, nu)

            Y = np.random.randn(N, ny) * YStd + YMean
            Z = np.random.randn(N, nz) * ZStd + ZMean
            U = np.random.randn(N, nu) * UStd + UMean

            YBU = copy.deepcopy(Y)
            ZBU = copy.deepcopy(Z)
            UBU = copy.deepcopy(U)

            sId = PSIDModel(nx=nx, n1=n1, iY=horizon)
            sId.fit(Y, Z, time_first=True)
            np.testing.assert_equal(Y, YBU)
            np.testing.assert_equal(Z, ZBU)
            np.testing.assert_equal(U, UBU)

            sId = PSIDModel(nx=nx, n1=n1, iY=horizon)
            sId.fit(Y, Z, time_first=True)
            np.testing.assert_equal(Y, YBU)
            np.testing.assert_equal(Z, ZBU)
            np.testing.assert_equal(U, UBU)

    def test_PSIDModel_filtering_and_smoothing(self):
        """Tests PSIDModel prediction, filtering, and smoothing (including bw_on_residual_y)."""
        np.random.seed(42)

        sysCode = "nyR5_10_nzR5_10_NxR2_4_N1R2_2_NeR0_0_S0_xNScLR1e-1_1e+1_yNScLR1e-1_1e+1_zSNRLR1e+1_1e+2_icdf_pbeta2_1_"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        N = int(2e4)
        nTrain = int(1e4)
        horizon = 5

        for ci in range(3):
            with self.subTest(ci=ci):
                s, _, zErrSys = generateRandomLinearModel(sysSettings)
                Y, X = s.generateRealization(N)
                Z = s.generateObservationFromStates(X, u=None, param_names=["Cz", "Dz"])
                if zErrSys is not None:
                    eps, _ = zErrSys.generateRealization(N)
                    Z += eps

                YTrain, YTest = Y[:nTrain], Y[nTrain:]
                ZTrain, ZTest = Z[:nTrain], Z[nTrain:]

                # Default smoothing enables both bw_on_residual and bw_on_residual_y
                sId = PSIDModel(
                    nx=s.state_dim, n1=s.zDims.size, i=horizon, updateKfKv=True
                )
                self.assertTrue(sId.bw_on_residual)
                self.assertTrue(sId.bw_on_residual_y)
                sId.fit(YTrain, ZTrain)

                for mode, idealKwargs in [
                    ("prediction", {}),
                    ("filtering", {"useXFilt": True}),
                    ("smoothing", {"useXSmooth": True}),
                ]:
                    sId.set_default_inference(mode)
                    zPred = sId.predict(YTest)[0]
                    zIdeal = s.predict(YTest, **idealKwargs)[0]
                    r2Model = np.mean(evalPrediction(ZTest, zPred, "R2"))
                    r2Ideal = np.mean(evalPrediction(ZTest, zIdeal, "R2"))
                    self.assertLess(abs(r2Model - r2Ideal), 1e-2)

                # Test smoothing with non-zero means and time_first=False
                sId.set_default_inference("smoothing")
                zSmthZeroMean = sId.predict(YTest)[0]

                YMean = np.random.randn(Y.shape[-1]) * 50
                ZMean = np.random.randn(Z.shape[-1]) * 50
                sIdMean = PSIDModel(
                    nx=s.state_dim,
                    n1=s.zDims.size,
                    i=horizon,
                    default_inference="smoothing",
                    updateKfKv=True,
                )
                sIdMean.fit((YTrain + YMean).T, (ZTrain + ZMean).T, time_first=False)
                zSmthWithMean = sIdMean.predict(YTest + YMean)[0]
                np.testing.assert_allclose(
                    zSmthWithMean - ZMean, zSmthZeroMean, atol=1e-5
                )

                # Test alternative bw_on_residual options and defaults
                sIdNoResY = PSIDModel(
                    nx=s.state_dim,
                    n1=s.zDims.size,
                    i=horizon,
                    default_inference="smoothing",
                    bw_on_residual_y=False,
                    updateKfKv=True,
                )
                self.assertTrue(sIdNoResY.bw_on_residual)
                self.assertFalse(sIdNoResY.bw_on_residual_y)
                sIdNoResY.fit(YTrain, ZTrain)
                zNoResY = sIdNoResY.predict(YTest)[0]
                r2NoResY = np.mean(evalPrediction(ZTest, zNoResY, "R2"))
                self.assertGreater(r2NoResY, 0.5)

                sIdNoRes = PSIDModel(
                    nx=s.state_dim,
                    n1=s.zDims.size,
                    i=horizon,
                    default_inference="smoothing",
                    bw_on_residual=False,
                    updateKfKv=True,
                )
                self.assertFalse(sIdNoRes.bw_on_residual)
                self.assertFalse(sIdNoRes.bw_on_residual_y)
                sIdNoRes.fit(YTrain, ZTrain)
                zNoRes = sIdNoRes.predict(YTest)[0]
                r2NoRes = np.mean(evalPrediction(ZTest, zNoRes, "R2"))
                self.assertGreater(r2NoRes, 0.5)

        # Test trial-based smoothing (list of arrays)
        YTrials = [YTrain[: nTrain // 2], YTrain[nTrain // 2 :]]
        ZTrials = [ZTrain[: nTrain // 2], ZTrain[nTrain // 2 :]]
        YTestTrials = [YTest[: nTrain // 2], YTest[nTrain // 2 :]]
        ZTestTrials = [ZTest[: nTrain // 2], ZTest[nTrain // 2 :]]
        sIdTrials = PSIDModel(
            nx=s.state_dim,
            n1=s.zDims.size,
            i=horizon,
            default_inference="smoothing",
            updateKfKv=True,
        )
        sIdTrials.fit(YTrials, ZTrials)
        zPredTrials = sIdTrials.predict(YTestTrials)[0]
        self.assertEqual(len(zPredTrials), len(YTestTrials))
        r2Trials = np.mean(
            evalPrediction(
                np.concatenate(ZTestTrials, axis=0),
                np.concatenate(zPredTrials, axis=0),
                "R2",
            )
        )
        r2IdealLast = np.mean(
            evalPrediction(ZTest, s.predict(YTest, useXSmooth=True)[0], "R2")
        )
        self.assertLess(abs(r2Trials - r2IdealLast), 2e-2)

    def test_evalSysId_backward_filtering_gain(self):
        """Tests forward and backward fltCzKf ground-truth evaluation in evalSysId."""
        np.random.seed(7)

        sysCode = "nyR5_6_nzR3_4_NxR2_2_N1R2_2_NeR0_0_S0_xNScLR1e-1_1e+1_yNScLR1e-1_1e+1_zSNRLR1e+1_1e+2_icdf_pbeta2_1_"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        s, _, zErrSys = generateRandomLinearModel(sysSettings)
        N = int(5e4)
        Y, X = s.generateRealization(N)
        Z = s.generateObservationFromStates(X, u=None, param_names=["Cz", "Dz"])
        if zErrSys is not None:
            eps, _ = zErrSys.generateRealization(N)
            Z += eps

        # When bw_on_residual=False, modelBW identifies the backward stochastic model directly,
        # so its learned fltCzKfBW should match {Cz}_{bw} {Kf}_{bw} + (S^b_{yz})^T (innovCov_bw)^{-1}.
        sId = PSIDModel(
            nx=s.state_dim,
            n1=s.zDims.size,
            i=5,
            default_inference="smoothing",
            bw_on_residual=False,
            updateKfKv=True,
        )
        sId.fit(Y, Z)
        sOrigCopy = copy.deepcopy(s)
        perf = evalSysId(sId, Y, Z, Y, Z, trueSys=s)[0]

        # Caller's trueSys should not be mutated with derived fltCzKf attribute
        self.assertEqual(hasattr(s, "fltCzKf"), hasattr(sOrigCopy, "fltCzKf"))
        self.assertIn("fltCzKfErrNormed", perf)
        self.assertIn("fltCzKfBWErrNormed", perf)
        self.assertLess(perf["fltCzKfErrNormed"], 5e-2)
        self.assertLess(perf["fltCzKfBWErrNormed"], 5e-2)


if __name__ == "__main__":
    unittest.main()
