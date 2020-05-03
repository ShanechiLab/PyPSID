""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tools for evaluating system identification
"""

import copy, itertools

import numpy as np
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error


def evalPrediction(trueValue, prediction, measure):

    if measure == 'CC':
        n = trueValue.shape[1]
        R = np.corrcoef(trueValue, prediction, rowvar=False)
        perf = np.diag(R[n:, :n])
    return perf
