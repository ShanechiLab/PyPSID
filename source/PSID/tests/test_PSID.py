""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tests the PSID function
"""

import unittest
import sys, os, copy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scipy import linalg
import numpy as np

from PSID.PSID import PSID as SubspacePSID
from PSID.PrepModel import PrepModel
from PSID.sim_tools import getSysSettingsFromSysCode, generateRandomLinearModel
from PSID.evaluation import evalSysId

numTests = 10

class TestPSID(unittest.TestCase):
    def test_PSID(self):
        np.random.seed(42)

        sysCode = 'nyR1_10_nzR1_10_NxR1_2_N1R0_2'
        sysSettings = getSysSettingsFromSysCode(sysCode)
        
        N = int(1e6)
        # horizon = 10
        horizon = [9, 11]

        failInds = []
        failErrs = []
        # numTests = 10
        for ci in range(numTests):
            with self.subTest(ci=ci):
                sOrig, sysU, zErrSys = generateRandomLinearModel(sysSettings)
                s = copy.deepcopy(sOrig)

                Y, X = s.generateRealizationWithKF(N)
                Z = s.generateObservationFromStates(X, u=None, param_names=['Cz', 'Dz'])
                    
                sId = SubspacePSID(Y, Z, nx=s.state_dim, n1=s.zDims.size, i=horizon, time_first=True)
                sId.zDims = np.arange(1, 1+min([s.zDims.size, s.state_dim]))
                err = evalSysId(sId, Y, Z, Y, Z, trueSys=s)[0]

                params = ['AErrNormed', 'CErrNormed', 'GErrNormed', 'YCovErrNormed', 'KErrNormed']
                if s.zDims.size > 0:
                    params.extend(['CzErrNormed', 'zEigErrNFN'])
                
                # Test that identification works even with non-zero mean signals
                YMean = np.random.randn(Y.shape[-1])*100
                ZMean = np.random.randn(Z.shape[-1])*100
                Y = Y - np.mean(Y, axis=0)
                Z = Z - np.mean(Z, axis=0)
                sId3 = SubspacePSID(Y+YMean, Z+ZMean, nx=s.state_dim, n1=s.zDims.size, i=horizon, time_first=True)
                sId3.zDims = np.arange(1, 1+min([s.zDims.size, s.state_dim]))
                sWithMeans = copy.deepcopy(s)
                sWithMeans.YPrepModel = PrepModel(YMean, remove_mean=True)
                sWithMeans.ZPrepModel = PrepModel(ZMean, remove_mean=True)
                err3 = evalSysId(sId3, Y+YMean, Z+ZMean, Y+YMean, Z+ZMean, trueSys=sWithMeans)[0]
                

                # Test that it also works if time_first = False
                sId4 = SubspacePSID((Y+YMean).T, (Z+ZMean).T, nx=s.state_dim, n1=s.zDims.size, i=horizon, time_first=False)
                sId4.zDims = np.arange(1, 1+min([s.zDims.size, s.state_dim]))
                err4 = evalSysId(sId4, Y+YMean, Z+ZMean, Y+YMean, Z+ZMean, trueSys=sWithMeans)[0]

                # Test decoding
                allZp, allYp, allXp = sId3.predict(Y+YMean, U=None)
                innovCovEmp = np.cov(allYp-Y, rowvar=False)
                # np.testing.assert_allclose(innovCovEmp, s.innovCov, rtol=1e-1)

                sId3_cp = copy.deepcopy(sId3)
                sId3_cp.YPrepModel = None
                sId3_cp.ZPrepModel = None
                allZp2, allYp2, allXp2 = sId3_cp.predict(Y, U=None)
                
                try:
                    np.testing.assert_array_less([err[p] for p in params], 5e-2, err_msg='Error too large for some params {}'.format(params))
                    np.testing.assert_array_less([err3[p] for p in params], 5e-2, err_msg='Error too large for some params {}'.format(params))
                    np.testing.assert_array_less([err4[p] for p in params], 5e-2, err_msg='Error too large for some params {}'.format(params))

                    np.testing.assert_allclose(allXp2, allXp, rtol=1e-3)
                    np.testing.assert_allclose(allZp2, allZp-ZMean, atol=1e-6)
                    np.testing.assert_allclose(allYp2, allYp-YMean, atol=1e-6)
                except Exception as e:
                    failInds.append(ci)
                    failErrs.append(e)
        
        if len(failInds) > 0:
            raise(Exception('{} => {}/{} random systems (indices: {}) failed: \n{}'.format(self.id(), len(failInds), numTests, failInds, failErrs)))
        else:
            print('{} => Ok: Tested with {} random systems, all were ok!'.format(self.id(), numTests))
                

    def test_PSID_doest_change_inputs(self):
        np.random.seed(42)

        N = int(1e3)
        
        horizon = 5

        # numTests = 10
        for ci in range(numTests):
            ny, nz, nu, nx, n1 = np.random.randint(1,10,5)
            
            YMean = 10*np.random.randn(1, ny)
            ZMean = 10*np.random.randn(1, nz)
            UMean = 10*np.random.randn(1, nu)
            
            YStd = 10*np.random.randn(1, ny)
            ZStd = 10*np.random.randn(1, nz)
            UStd = 10*np.random.randn(1, nu)
            
            Y = np.random.randn(N, ny) * YStd + YMean
            Z = np.random.randn(N, nz) * ZStd + ZMean
            U = np.random.randn(N, nu) * UStd + UMean

            YBU = copy.deepcopy(Y)
            ZBU = copy.deepcopy(Z)
            UBU = copy.deepcopy(U)

            sId = SubspacePSID(Y, Z, nx=nx, n1=n1, i=horizon, time_first=True)
            np.testing.assert_equal(Y, YBU)
            np.testing.assert_equal(Z, ZBU)
            np.testing.assert_equal(U, UBU)

                

if __name__ == '__main__':
    unittest.main()