""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tools for evaluating system identification
"""
import warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def evalPrediction(trueValue, prediction, measure):
    if prediction.shape[0] == 0:
        perf = np.empty(trueValue.shape[1])
        perf[:] = np.nan
        return perf

    if measure == 'CC':
        n = trueValue.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = np.corrcoef(trueValue, prediction, rowvar=False)
        perf = np.diag(R[n:, :n])
    elif measure == 'R2':
        perf = r2_score(trueValue, prediction, multioutput='raw_values')
    elif measure == 'MSE':
        perf = mean_squared_error(trueValue, prediction, multioutput='raw_values')
    elif measure == 'RMSE':
        MSE = evalPrediction(trueValue, prediction, 'MSE')
        perf = np.sqrt(MSE)
    else:
        raise(Exception('Performance measure "{}" is not supported.'.format(measure)))
    return perf
