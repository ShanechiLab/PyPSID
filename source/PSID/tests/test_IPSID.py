"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tests the IPSID method
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from PSID.IPSID import IPSID as SubspaceIPSID
from PSID.PrepModel import PrepModel
from PSID.sim_tools import generateRandomLinearModel, getSysSettingsFromSysCode
from PSID.evaluation import evalSysId, computeLSSMIdError
from scipy import linalg

numTests = 10  # Increase this for a slower but more thorough test


class TestIPSID(unittest.TestCase):
    def test_IPSID(self):
        np.random.seed(42)

        sysCode = "nyR1_10_nzR1_10_nuR0_2_NxR1_2_N1R0_2"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        N = int(1e5)
        horizon = 10
        horizon2 = [9, 11]

        failInds = []
        failErrs = []
        # numTests = 10
        for ci in range(numTests):
            with self.subTest(ci=ci):
                sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                s = copy.deepcopy(sOrig)

                nu = s.input_dim
                if nu > 0:
                    U, XU = sysU.generateRealization(N=N)
                else:
                    U, XU = None, None
                Y, X = s.generateRealizationWithKF(N, u=U)
                Z = (s.Cz @ X.T).T
                if nu > 0 and hasattr(s, "Dz") and s.Dz is not None:
                    Z += (s.Dz @ U.T).T

                sId = SubspaceIPSID(
                    Y, Z, U, nx=s.state_dim, n1=s.zDims.size, i=horizon, time_first=True
                )
                err = evalSysId(sId, Y, Z, Y, Z, U, U, trueSys=s)[0]

                sId2 = SubspaceIPSID(
                    Y,
                    Z,
                    U,
                    nx=s.state_dim,
                    n1=s.zDims.size,
                    i=horizon2,
                    time_first=True,
                )
                err2 = evalSysId(sId2, Y, Z, Y, Z, U, U, trueSys=s)[0]

                params = [
                    "AErrNormed",
                    "CErrNormed",
                    "GErrNormed",
                    "YCovErrNormed",
                    "KErrNormed",
                ]
                if s.input_dim > 0:
                    params.extend(["BErrNormed", "DErrNormed"])
                if s.zDims.size > 0:
                    params.extend(["CzErrNormed", "zEigErrNFN"])
                    if s.input_dim > 0:
                        params.extend(["DzErrNormed"])

                # Test that identification works even with non-zero mean signals
                YMean = np.random.randn(Y.shape[-1]) * 100
                ZMean = np.random.randn(Z.shape[-1]) * 100
                Y = Y - np.mean(Y, axis=0)
                Z = Z - np.mean(Z, axis=0)
                if nu > 0:
                    UMean = np.random.randn(U.shape[-1]) * 100
                    U = U - np.mean(U, axis=0)
                    UNew = U + UMean
                    UNewT = UNew.T
                else:
                    UNew = None
                    UNewT = None
                sId3 = SubspaceIPSID(
                    Y + YMean,
                    Z + ZMean,
                    U=UNew,
                    nx=s.state_dim,
                    n1=s.zDims.size,
                    i=horizon,
                    time_first=True,
                )
                sWithMeans = copy.deepcopy(s)
                sWithMeans.YPrepModel = PrepModel(YMean, remove_mean=True)
                sWithMeans.ZPrepModel = PrepModel(ZMean, remove_mean=True)
                if nu > 0:
                    sWithMeans.UPrepModel = PrepModel(UMean, remove_mean=True)
                err3 = evalSysId(
                    sId3,
                    Y + YMean,
                    Z + ZMean,
                    YTrain=Y + YMean,
                    ZTrain=Z + ZMean,
                    UTest=UNew,
                    UTrain=UNew,
                    trueSys=sWithMeans,
                )[0]

                # Test that it also works if time_first = False
                sId4 = SubspaceIPSID(
                    (Y + YMean).T,
                    (Z + ZMean).T,
                    U=UNewT,
                    nx=s.state_dim,
                    n1=s.zDims.size,
                    i=horizon,
                    time_first=False,
                )
                err4 = evalSysId(
                    sId4,
                    Y + YMean,
                    Z + ZMean,
                    YTrain=Y + YMean,
                    ZTrain=Z + ZMean,
                    UTest=UNew,
                    UTrain=UNew,
                    trueSys=sWithMeans,
                )[0]

                # Test decoding
                allZp, allYp, allXp = sId3.predict(Y + YMean, U=UNew)
                innovCovEmp = np.cov(allYp - Y, rowvar=False)
                # np.testing.assert_allclose(innovCovEmp, s.innovCov, rtol=1e-1)

                sId3_cp = copy.deepcopy(sId3)
                sId3_cp.YPrepModel = None
                sId3_cp.ZPrepModel = None
                sId3_cp.UPrepModel = None
                allZp2, allYp2, allXp2 = sId3_cp.predict(Y, U=U)

                try:
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

                    np.testing.assert_allclose(allXp2, allXp, rtol=1e-3)
                    np.testing.assert_allclose(allZp2, allZp - ZMean, atol=1e-6)
                    np.testing.assert_allclose(allYp2, allYp - YMean, atol=1e-6)
                except Exception as e:
                    failInds.append(ci)
                    failErrs.append(e)

        if failInds:
            raise Exception(
                "Failed for indices {} with errors {}".format(failInds, failErrs)
            )

    def test_IPSID_enforce_stability(self):
        np.random.seed(42)

        sysCode = "nyR1_10_nzR1_10_nuR0_2_NxR1_4_N1R0_4"
        sysSettings = getSysSettingsFromSysCode(sysCode)

        N = int(
            1e3
        )  # Very small samples can cause learned models unstable, triggering the enforce_stability option that we want to test here
        horizon = 10
        horizon2 = [9, 11]

        # numTests = 10
        for ci in range(numTests):
            with self.subTest(ci=ci):
                # Prepare an unstable random model
                is_unstable = False
                while not is_unstable:
                    sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                    s = copy.deepcopy(sOrig)
                    # Make slightly unstable
                    newA = s.A + (
                        1.01 - np.max(np.abs(np.linalg.eig(s.A)[0]))
                    ) * np.eye(s.A.shape[0])
                    s.changeParams({"A": newA})
                    is_unstable = np.all(np.isnan(s.XCov))

                nu = s.input_dim
                if nu > 0:
                    U, XU = sysU.generateRealization(N=N)
                else:
                    U, XU = None, None
                Y, X = s.generateRealizationWithKF(N, u=U)
                Z = (s.Cz @ X.T).T
                if nu > 0 and hasattr(s, "Dz") and s.Dz is not None:
                    Z += (s.Dz @ U.T).T

                # Check both IPSID and INDM behavior
                for n1 in [0, s.zDims.size]:
                    sId = SubspaceIPSID(
                        Y,
                        Z,
                        U,
                        nx=s.state_dim,
                        n1=n1,
                        i=horizon,
                        time_first=True,
                        force_stable_stage1=False,
                        force_stable_stage2=False,
                        force_stable_if_not=False,
                    )

                    sId2 = SubspaceIPSID(
                        Y,
                        Z,
                        U,
                        nx=s.state_dim,
                        n1=n1,
                        i=horizon2,
                        time_first=True,
                        force_stable_stage1=True,
                        force_stable_stage2=True,
                        force_stable_if_not=False,
                    )

                    eigsAbs = np.abs(np.linalg.eig(sId2.A)[0])
                    if np.any(eigsAbs > 1):
                        print("Not Ok")
                    else:
                        print("Ok")
                    print("Done")
                    np.testing.assert_array_less(eigsAbs, 1)

        if failInds:
            raise Exception(
                "Failed for indices {} with errors {}".format(failInds, failErrs)
            )


if __name__ == "__main__":
    unittest.main()
