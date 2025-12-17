"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tests the LSSM module
"""

import unittest
import sys, os, copy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scipy import linalg
import numpy as np

from PSID.LSSM import solveric, solve_discrete_are_iterative, genRandomGaussianNoise
from PSID.sim_tools import getSysSettingsFromSysCode, generateRandomLinearModel

numTests = 100


class TestLSSM(unittest.TestCase):
    #############################################
    def test_generateRealizationWithKF(self):
        sysCode = f"20230424_1_nu1_nxu1_ny1_nz1_Nx1_N11_1_Ne1_1_yNScL1e-0_zSNRLR1e-0_1e+0"  # with input
        # sysCode = f'20230424_1_ny1_nz1_Nx1_N11_1_Ne1_1_yNScL1e-0_zSNRLR1e-0_1e+0' # without input

        sysSettings = getSysSettingsFromSysCode(sysCode)
        sysSettings["predictor_form"] = True
        mse = []
        for i in range(20):
            print("Working on model number {}".format(i + 1))
            s, sysU, zErrSys = generateRandomLinearModel(sysSettings)
            # sOrig = copy.deepcopy(s)
            # s = SSM(lssm=sOrig)
            N = int(2e2)
            x0 = None
            nu = s.input_dim
            if nu > 0:
                U, XU = sysU.generateRealization(N=N)
            else:
                U, XU = None, None

            Y, X, Z, eZ = s.generateRealization(
                N, return_z=True, return_z_err=True, u=U, x0=x0
            )
            allXp, allYp, allXf = s.kalman(Y, U=U)
            # eY, eYShaping = genRandomGaussianNoise(N, s.innovCov)
            eY = Y - allYp
            s.R = None
            Yp, Xp, Zp, eZp = s.generateRealization(
                N, return_z=True, return_z_err=True, u=U, e=eY, x0=x0
            )
            """
            plt.figure()
            plt.plot(Y, label='Stochastic')
            plt.plot(Yp, label='Predictor')
            # plt.plot(allYp, label='KF')
            plt.title('Y')
            plt.legend()
            # plt.show()

            plt.figure()
            plt.plot(X, label='Stochastic')
            plt.plot(Xp, label='Predictor')
            # plt.plot(allXp, label='KF')
            plt.title('X')
            plt.legend()

            plt.figure()
            plt.plot(Z, label='Stochastic')
            plt.plot(Zp, label='Predictor')
            plt.title('Z')
            plt.legend()
            """
            mse.append(np.mean((Y - Yp) ** 2))
            np.testing.assert_almost_equal(Y, Yp, decimal=10)
            # np.testing.assert_equal(Y, Yp)
            # np.testing.assert_allclose(Y, Yp, rtol=1e-3)

        print("AVG MSE(Y,Yp)={}".format(np.mean(mse)))
        print("done!")

    def test_solveric(self):
        inp = {
            "A": np.array([[1.6931, -0.6328], [0.7111, 0.3927]]),
            "C": np.array([[-0.6978, -0.6595], [2.0193, 0.7715], [-1.7937, -0.8203]]),
            "G": np.array([[-0.8780, 1.5319, -0.7597], [0.1489, 0.5318, 0.3480]]),
            "L0": np.array(
                [
                    [0.0179, -0.9271, -0.1142],
                    [0.6545, -0.1699, 0.3695],
                    [1.2577, -0.4717, -0.6186],
                ]
            ),
        }
        out = (np.array([[0.6479, 0.0861], [0.6025, -0.1736]]), True)

        P, has_solution = solveric(**inp)
        np.testing.assert_equal(out[1], has_solution)
        np.testing.assert_allclose(out[0], P, rtol=1e-3)

    def test_solve_discrete_are_iterative(self):
        np.random.seed(42)

        sysCode = "nyR1_10_nzR1_10_nuR0_10_NxR1_10_N1R0_10"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        # numTests = 100
        for ci in range(numTests):
            with self.subTest(ci=ci):
                sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                s = copy.deepcopy(sOrig)
                # print('Testing system {}/{}'.format(1+ci, numTests))

                Pp1 = linalg.solve_discrete_are(s.A.T, s.C.T, s.Q, s.R, s=s.S)
                Pp2 = solve_discrete_are_iterative(s.A.T, s.C.T, s.Q, s.R, s=s.S)
                Pp3, log = solve_discrete_are_iterative(
                    s.A.T, s.C.T, s.Q, s.R, s=s.S, return_log=True
                )

                np.testing.assert_almost_equal(Pp1, Pp2)
                np.testing.assert_almost_equal(0, log[-1])

    def test_LSSM_randomize_in_predictor_form(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)
        sysSettings["predictor_form"] = True

        failInds = []
        failErrs = []
        # numTests = 100
        for ci in range(numTests):
            with self.subTest(ci=ci):
                sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                s = copy.deepcopy(sOrig)
                # print('Testing system {}/{}'.format(1+ci, numTests))

                n1 = s.zDims.size
                trBlock = s.A_KC[:n1, n1:]
                etcBlock = np.copy(s.A_KC)
                etcBlock[:n1, n1:] = 0
                try:
                    np.testing.assert_almost_equal(trBlock, np.zeros_like(trBlock))
                    np.testing.assert_allclose(
                        (s.A_KC - etcBlock) / np.linalg.norm(etcBlock),
                        np.zeros_like(etcBlock),
                        atol=1e-8,
                    )
                except Exception as e:
                    # print('Failed at sys {}/{} => error: {}'.format(1+ci, numTests, e))
                    # s = copy.deepcopy(sOrig)
                    # E = s.makeA_KCBlockDiagonal()
                    failInds.append(ci)
                    failErrs.append(e)
                    raise (Exception(e))

        if len(failInds) > 0:
            print(
                "{} => {}/{} random systems FAILED!".format(
                    self.id(), len(failInds), numTests
                )
            )
        else:
            print(
                "\n{} => Ok: Tested with {} random systems".format(self.id(), numTests)
            )

    def test_LSSM_makeA_KCBlockDiagonal(self):
        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_3_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        failInds = []
        failErrs = []
        numTests = 200
        for ci in range(numTests):
            with self.subTest(ci=ci):
                sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                s = copy.deepcopy(sOrig)
                # print('Testing system {}/{}'.format(1+ci, numTests))

                E = s.makeA_KCBlockDiagonal(ignore_error=True)

                n1 = s.zDims.size
                trBlock = s.A_KC[:n1, n1:]
                try:
                    np.testing.assert_almost_equal(trBlock, np.zeros_like(trBlock))
                except Exception as e:
                    # print('Failed at sys {}/{} => error: {}'.format(1+ci, numTests, e))
                    # s = copy.deepcopy(sOrig)
                    # E = s.makeA_KCBlockDiagonal()
                    failInds.append(ci)
                    failErrs.append(e)

                # Make sure A eigenvalues are the same as before
                eigOrig = np.linalg.eigvals(sOrig.A)
                eigNew = np.linalg.eigvals(s.A)
                np.testing.assert_allclose(np.sort(eigOrig), np.sort(eigNew))

        if len(failInds) > 0:
            e = "{} => {}/{} random systems FAILED!".format(
                self.id(), len(failInds), numTests
            )
            if len(failInds) > 10:
                raise (Exception(e))
            else:
                print(
                    f"Occasionally impossible diagonalization problem is expected: {e}"
                )
        else:
            print(
                "\n{} => Ok: Tested with {} random systems".format(self.id(), numTests)
            )

    def test_kalmanFilterAndSmoothingForS0Nu0(self):
        from pykalman.standard import _filter, _smooth

        np.random.seed(42)

        sysCode = "nyR1_5_nzR1_5_nuR0_0_NxR1_5_N1R0_5"
        sysSettings = getSysSettingsFromSysCode(sysCode)
        sysSettings["S0"] = True

        failInds = []
        failErrs = []
        # numTests = 100

        nxAll = []
        errPredAll = []
        errFilterAll = []
        errSmoothAll = []
        for ci in range(numTests):
            with self.subTest(ci=ci):
                sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                s = copy.deepcopy(sOrig)

                N = 1000
                if sOrig.input_dim > 0:
                    U, Xu = sysU.generateRealization(N, random_x0=True)
                else:
                    U = None
                Y, X = sOrig.generateRealization(N, random_x0=True, u=U)

                allXp, allYp, allXf, allXs, allPp, allPf, allPs = s.kalmanSmoother(
                    Y, U=U, steady_state=False, return_state_cov=True
                )

                if sOrig.input_dim == 0 and sysSettings["S0"]:
                    # Compare with the pykalman library
                    (
                        predicted_state_means,
                        predicted_state_covariances,
                        kalman_gains,
                        filtered_state_means,
                        filtered_state_covariances,
                    ) = _filter(
                        transition_matrices=s.A,
                        observation_matrices=s.C,
                        transition_covariance=s.Q,
                        observation_covariance=s.R,
                        transition_offsets=np.zeros(s.state_dim),
                        observation_offsets=np.zeros(s.output_dim),
                        initial_state_mean=np.zeros(s.state_dim),
                        initial_state_covariance=np.eye(s.state_dim),
                        observations=Y,
                    )

                    (
                        smoothed_state_means,
                        smoothed_state_covariances,
                        kalman_smoothing_gains,
                    ) = _smooth(
                        transition_matrices=s.A,
                        filtered_state_means=filtered_state_means,
                        filtered_state_covariances=filtered_state_covariances,
                        predicted_state_means=predicted_state_means,
                        predicted_state_covariances=predicted_state_covariances,
                    )

                    from pykalman import KalmanFilter

                    kf = KalmanFilter(
                        transition_matrices=s.A,
                        observation_matrices=s.C,
                        transition_covariance=s.Q,
                        observation_covariance=s.R,
                        initial_state_mean=np.zeros(s.state_dim),
                        initial_state_covariance=np.eye(s.state_dim),
                    )
                    filtered_state_means_2, filtered_state_covariances_2 = kf.filter(Y)
                    smoothed_state_means_2, smoothed_state_covariances_2 = kf.smooth(Y)

                    np.testing.assert_allclose(
                        filtered_state_means, filtered_state_means_2, rtol=1e-3
                    )
                    np.testing.assert_allclose(
                        smoothed_state_means, smoothed_state_means_2, rtol=1e-3
                    )

                    np.testing.assert_allclose(allXp, predicted_state_means, rtol=1e-3)
                    np.testing.assert_allclose(allXf, filtered_state_means, rtol=1e-3)
                    np.testing.assert_allclose(allXs, smoothed_state_means, rtol=1e-3)

                    np.testing.assert_allclose(
                        allPp, predicted_state_covariances, rtol=1e-3
                    )
                    np.testing.assert_allclose(
                        allPf, filtered_state_covariances, rtol=1e-3
                    )
                    np.testing.assert_allclose(
                        allPs, smoothed_state_covariances, rtol=1e-3
                    )

                    # Using the pykalman smoother with the PSID Kalman filter
                    allXp1, allYp1, allXf1, allPp1, allPf1 = s.kalman(
                        Y, U=U, steady_state=False, return_state_cov=True
                    )
                    (
                        smoothed_state_means3,
                        smoothed_state_covariances3,
                        kalman_smoothing_gains3,
                    ) = _smooth(
                        transition_matrices=s.A,
                        filtered_state_means=allXf1,
                        filtered_state_covariances=allPf1,
                        predicted_state_means=allXp1,
                        predicted_state_covariances=allPp1,
                    )

                    np.testing.assert_allclose(allXs, smoothed_state_means3, rtol=1e-3)
                    np.testing.assert_allclose(
                        allPs, smoothed_state_covariances3, rtol=1e-3
                    )
                else:
                    print(
                        "Cannot test the S non-zero case or the input cases against pykalman"
                    )

                errPred = np.atleast_2d(np.cov(allXp - X, rowvar=False))
                errFilter = np.atleast_2d(np.cov(allXf - X, rowvar=False))
                errSmooth = np.atleast_2d(np.cov(allXs - X, rowvar=False))

                nxAll.append(s.state_dim)
                errPredAll.append(errPred)
                errFilterAll.append(errFilter)
                errSmoothAll.append(errSmooth)

                # These comparisons aren't always true for finite data:
                # np.testing.assert_array_less(np.sum(np.diag(errFilter)), np.sum(np.diag(errPred)))
                # np.testing.assert_array_less(np.sum(np.diag(errSmooth)), np.sum(np.diag(errFilter)))
                # np.testing.assert_array_less(np.diag(errFilter), np.diag(errPred))
                # np.testing.assert_array_less(np.diag(errSmooth), np.diag(errFilter))

        errPredAvg = np.array([np.mean(np.diag(e)) for e in errPredAll])
        errFilterAvg = np.array([np.mean(np.diag(e)) for e in errFilterAll])
        errSmoothAvg = np.array([np.mean(np.diag(e)) for e in errSmoothAll])

        print(
            "For {}/{} random models, prediction error < filtering error!".format(
                np.sum(errPredAvg < errFilterAvg), errPredAvg.size
            )
        )
        print(
            "For {}/{} random models, filtering error < smoothing error!".format(
                np.sum(errFilterAvg < errSmoothAvg), errFilterAvg.size
            )
        )
        pass


if __name__ == "__main__":
    unittest.main()
