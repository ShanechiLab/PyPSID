""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tests the PrepModel object
"""

import unittest
import sys, os, copy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

class TestPrepModel(unittest.TestCase):
    def test_preprocessing(self):
        from PSID.PrepModel import PrepModel

        np.random.seed(42)

        arg_sets = [
            {'remove_mean': True , 'zscore': True },
            {'remove_mean': False, 'zscore': True },
            {'remove_mean': True , 'zscore': False},
            {'remove_mean': False, 'zscore': False},
            {'remove_mean': True , 'zscore': True , 'std_ddof': 0}
        ]

        for args in arg_sets:
            with self.subTest(ci=args):
                numTests = 100
                for ci in range(numTests):
                    n_dim = np.random.randint(1, 10)

                    trueMean = 10*np.random.randn(n_dim)
                    trueStd = 10*np.random.randn(n_dim)
                    
                    for time_first in [True, False]:
                        n_samples = np.random.randint(10, 100)
                        data = np.random.randn(n_samples, n_dim) * trueStd + trueMean
                        if not time_first:
                            data = data.T
                        dataCopy = copy.deepcopy(data)

                        sm = PrepModel()
                        sm.fit(data, time_first=time_first, **args)
                        
                        np.testing.assert_equal(data, dataCopy)
                        
                        ddof = args['std_ddof'] if 'std_ddof' in args else 1
                        newData = sm.apply(data, time_first=time_first)
                        if time_first:
                            newDataMean = np.mean(newData,axis=0)
                            newDataStd = np.std(newData,axis=0, ddof=ddof)
                        else:
                            newDataMean = np.mean(newData,axis=1)
                            newDataStd = np.std(newData,axis=1, ddof=ddof)
                        
                        if args['zscore'] or args['remove_mean']:
                            np.testing.assert_almost_equal(newDataMean, np.zeros_like(newDataMean))
                        if args['zscore']:
                            np.testing.assert_almost_equal(newDataStd, np.ones_like(newDataStd))

                        recoveredData = sm.apply_inverse(newData, time_first=time_first)

                        np.testing.assert_almost_equal(recoveredData, dataCopy)

    def test_preprocessing_for_segmented_data(self):
        from PSID.PrepModel import PrepModel

        np.random.seed(42)

        arg_sets = [
            {'remove_mean': True , 'zscore': True },
            {'remove_mean': False, 'zscore': True },
            {'remove_mean': True , 'zscore': False},
            {'remove_mean': False, 'zscore': False},
            {'remove_mean': True , 'zscore': True , 'std_ddof': 0}
        ]

        for args in arg_sets:
            with self.subTest(ci=args):
                numTests = 100
                for ci in range(numTests):
                    n_dim = np.random.randint(1, 10)

                    trueMean = 10*np.random.randn(n_dim)
                    trueStd = 10*np.random.randn(n_dim)

                    n_segments = np.random.randint(1, 10)

                    for time_first in [True, False]:
                        data = []
                        for t in range(n_segments):
                            n_samples = np.random.randint(10, 100)
                            dataThis = np.random.randn(n_samples, n_dim) * trueStd + trueMean
                            if not time_first:
                                dataThis = dataThis.T
                            data.append( dataThis )

                        dataCopy = copy.deepcopy(data)

                        sm = PrepModel()
                        sm.fit(data, time_first=time_first, **args)
                        
                        np.testing.assert_equal(data, dataCopy)
                        
                        ddof = args['std_ddof'] if 'std_ddof' in args else 1
                        newData = sm.apply(data, time_first)
                        if time_first:
                            newDataCat = np.concatenate(newData, axis=0)
                            newDataMean = np.mean(newDataCat,axis=0)
                            newDataStd = np.std(newDataCat,axis=0, ddof=ddof)
                        else:
                            newDataCat = np.concatenate(newData, axis=1)
                            newDataMean = np.mean(newDataCat,axis=1)
                            newDataStd = np.std(newDataCat,axis=1, ddof=ddof)

                        if args['zscore'] or args['remove_mean']:
                            np.testing.assert_almost_equal(newDataMean, np.zeros_like(newDataMean))
                        if args['zscore']:
                            np.testing.assert_almost_equal(newDataStd, np.ones_like(newDataStd))

                        recoveredData = sm.apply_inverse(newData, time_first)

                        for t in range(len(dataCopy)):
                            np.testing.assert_almost_equal(recoveredData[t], dataCopy[t])


if __name__ == '__main__':
    unittest.main()