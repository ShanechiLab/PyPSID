"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tools for system identification
"""

import numpy as np


def extractDiagonalBlocks(A, emptySide="both", absThr=np.spacing(1)):
    """Extracts diagonal blocks from a matrix.

    Args:
        A (np.ndarray): Input matrix.
        emptySide (str, optional): Constraints on off-diagonal blocks ('both', 'upper', 'lower', 'either'). Defaults to "both".
        absThr (float, optional): Threshold for considering a value as zero. Defaults to machine epsilon.

    Returns:
        BLKS (np.ndarray): Array of block sizes.
    """
    if emptySide == "either":
        BLKSU = extractDiagonalBlocks(A, "upper", absThr)
        BLKSL = extractDiagonalBlocks(A, "lower", absThr)
        if len(BLKSU) >= len(BLKSL):
            BLKS = BLKSU
        else:
            BLKS = BLKSL
        return BLKS

    j = 0
    BLKS = np.empty(0, dtype=int)
    while j < A.shape[0]:
        if emptySide == "both" or emptySide == "upper":
            for j1 in range(j, A.shape[1]):
                if j1 == (A.shape[1] - 1) or np.all(
                    np.abs(A[j : (j1 + 1), (j1 + 1) :]) <= absThr
                ):
                    i1 = j1 - j + 1
                    break

        if emptySide == "both" or emptySide == "lower":
            for j2 in range(j, A.shape[0]):
                if j2 == (A.shape[0] - 1) or np.all(
                    np.abs(A[(j2 + 1) :, j : (j2 + 1)]) <= absThr
                ):
                    i2 = j2 - j + 1
                    break

        if emptySide == "upper":
            i2 = i1
        elif emptySide == "lower":
            i1 = i2

        i = j + int(np.max([i1, i2]))
        BLKS = np.concatenate((BLKS, [i - j]))
        j = i

    return BLKS


def getBlockIndsFromBLKSArray(BLKS):
    """Converts a block size array into block indices.

    Args:
        BLKS (np.ndarray): Array of block sizes.

    Returns:
        groups (np.ndarray): 2xN array of start and end indices for each block.
    """
    if len(BLKS) == 0:
        return np.empty(0, dtype=int)

    BLKSCUM = np.array(np.atleast_2d(np.cumsum(BLKS)).T, dtype=int)
    groups = np.concatenate(
        (
            np.concatenate((np.zeros((1, 1), dtype=int), BLKSCUM[:-1, :]), axis=0),
            BLKSCUM,
        ),
        axis=1,
    )

    return groups


def prepare_fold_inds(num_folds, N):
    """Prepares train/test indices for K-fold cross-validation.

    Args:
        num_folds (int): Number of folds.
        N (int): Total number of samples.

    Returns:
        folds (list): List of dictionaries, each containing 'test_inds' and 'train_inds'.
    """
    folds = []
    N_test = int(N / num_folds)
    for fold_ind in range(num_folds):
        test_inds = np.arange(
            N_test * fold_ind,
            N_test * (1 + fold_ind) if fold_ind < (num_folds - 1) else N,
        )
        train_inds = np.where(~np.isin(np.arange(N), test_inds))[0]
        folds.append(
            {
                "num_folds": num_folds,
                "fold": fold_ind + 1,
                "test_inds": test_inds,
                "train_inds": train_inds,
            }
        )
    return folds


def applyFuncIf(Y, func):
    """Applies a function on Y itself if Y is an array or on each element of Y if it is a list/tuple of arrays.

    Args:
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: transformed Y or list of transformed arrays.
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return [func(YThis) for YThis in Y]
    else:
        return func(Y)


def transposeIf(Y):
    """Transposes Y itself if Y is an array or each element of Y if it is a list/tuple of arrays.

    Args:
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: transposed Y or list of transposed arrays.
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return [transposeIf(YThis) for YThis in Y]
    else:
        return Y.T


def subtractIf(X, Y):
    """Subtracts Y from X if X is an array, or subtracts each element of Y from
    each corresponding element of X if they are list/tuple of arrays.

    Args:
        X (np.array or list or tuple): input data or list of input data arrays.
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: X - Y or list of X - Ys
    """
    if Y is None:
        return X
    if isinstance(X, (list, tuple)):
        return [X[i] - Y[i] for i in range(len(X))]
    else:
        return X - Y


def catIf(Y, axis=None):
    """If Y is a list of arrays, will concatenate them otherwise returns Y

    Args:
        Y (np.array or list or tuple): input data or list of input data arrays.

    Returns:
        np.array or list or tuple: transposed Y or list of transposed arrays.
    """
    if Y is None:
        return None
    elif isinstance(Y, (list, tuple)):
        return np.concatenate(Y, axis=axis)
    else:
        return Y
