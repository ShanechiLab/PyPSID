""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

An object for keeping track of data preprocessing (mean removal and zscoring)
"""
import warnings

import numpy as np

class PrepModel():
    """Describes a preprocessing model to change mean/std of a time-series and undo that
    """    
    def __init__(self, mean=None, std=None, remove_mean=None, zscore=False):
        """See the fit method.
        """
        self.mean = mean
        self.std = std
        self.remove_mean = remove_mean
        self.zscore = zscore

    def fit(self, Y, remove_mean=True, zscore=False, std_ddof=1, time_first=True):
        """Learns the preprocessing model from data
        Args:
            Y (numpy array or list of arrays): Input data. First dimension must be timeand the second 
                        dimension is the data. Can be an array of data in which case the stats will be
                        learned from the concatenation of all segments along the first dimension.
            remove_mean (bool, optional): If True, will remove the mean of data. Defaults to True.
            zscore (bool, optional): If True, will zscore the data to have unit std in all dimensions. Defaults to False.
            std_ddof (int, optional): ddof argument for computing std. Defaults to 1.
            time_first (bool, optional): If true, will assume input data has time as the first dimension.
                        Otherwise assumes time is the second dimension. In any case, model will by default 
                        treat new data as if time is the first dimension. Defaults to True.
        """        
        if zscore:
            remove_mean = True # Must also remove the mean for z-scoring
        
        if isinstance(Y, (list, dict)):
            if time_first:
                YCat = np.concatenate(Y, axis=0)
            else:
                YCat = np.concatenate(Y, axis=1)
        else:
            YCat = Y

        if not time_first:
            YCat = YCat.T

        yDim = YCat.shape[1]
        yMean = np.zeros(yDim)
        yStd = np.ones(yDim)
        if remove_mean:
            yMean = np.array(np.nanmean(YCat, axis=0))
        if zscore:
            yStd = np.array(np.nanstd(YCat, axis=0, ddof=std_ddof))
            if np.any(yStd==0):
                warnings.warn('{} dimension(s) of y (out of {}) are flat. Will skip scaling to unit variance for those dimensions.'.format(np.sum(yStd==0), yStd.size))
            if np.all(yStd==0): # No dimension can be z-scored
                zscore = False

        self.remove_mean = remove_mean
        self.zscore = zscore
        self.mean = yMean
        self.std = yStd
        self.stdDOF = std_ddof

    def get_mean(self, time_first=True):
        """Returns the mean, but transposes it if needed

        Args:
            time_first (bool, optional): If true, will return the mean a row vector, 
                        otherwise returns it as a row vector. Defaults to True.
        """        
        if time_first:
            return self.mean[np.newaxis, :]
        else:
            return self.mean[:, np.newaxis]

    def get_std(self, time_first=True):
        """Returns the std, but transposes it if needed

        Args:
            time_first (bool, optional): If true, will return the std a row vector, 
                        otherwise returns it as a row vector. Defaults to True.
        """        
        if time_first:
            return self.std[np.newaxis, :]
        else:
            return self.std[:, np.newaxis]
            
    def apply_segment(self, Y, time_first=True):
        """Applies the preprocessing on new data

        Args:
            Y (numpy array): Input data. First dimension must be time and the second 
                                dimension is the data. Can be an array of data.
            time_first (bool, optional): If False, will assume time is the second dimensions. 
                                Defaults to True.
        """
        if self.remove_mean:
            Y = Y - self.get_mean(time_first)
        if self.zscore:
            okDims = self.std>0
            if time_first:
                Y[:, okDims] = Y[:, okDims] / self.get_std(time_first)[:, okDims]
            else:
                Y[okDims, :] = Y[okDims, :] / self.get_std(time_first)[okDims, :]
        return Y

    def apply(self, Y, time_first=True):
        """Applies the preprocessing on new data

        Args:
            Y (numpy array or list of arrays): Input data. First dimension must be time and the second 
                                dimension is the data. Can be an array of data.
            time_first (bool, optional): If False, will assume time is the second dimensions. 
                                Defaults to True.
        """
        if isinstance(Y, (list, tuple)):
            return [self.apply_segment(YThis,time_first) for YThis in Y]
        else:
            return self.apply_segment(Y,time_first)

    def apply_inverse_segment(self, Y, time_first=True):
        """Applies inverse of the preprocessing on new data (i.e. undoes the preprocessing)

        Args:
            Y (numpy array): Input data. First dimension must be time and the second 
                                dimension is the data. Can be an array of data.
            time_first (bool, optional): If False, will assume time is the second dimensions. 
                                Defaults to True.
        """
        if self.zscore:
            okDims = self.std>0
            if time_first:
                Y[:, okDims] = Y[:, okDims] * self.get_std(time_first)[:, okDims]
            else:
                Y[okDims, :] = Y[okDims, :] * self.get_std(time_first)[okDims, :]
        if self.remove_mean:
            mean = self.get_mean(time_first)
            Y = Y + mean
        return Y

    def apply_inverse(self, Y, time_first=True):
        """Applies inverse of the preprocessing on new data (i.e. undoes the preprocessing)

        Args:
            Y (numpy array or list of arrays): Input data. First dimension must be time and the second 
                                dimension is the data. Can be an array of data.
            time_first (bool, optional): If False, will assume time is the second dimensions. 
                                Defaults to True.
        """
        if isinstance(Y, (list, tuple)):
            return [self.apply_inverse_segment(YThis,time_first) for YThis in Y]
        else:
            return self.apply_inverse_segment(Y,time_first)
