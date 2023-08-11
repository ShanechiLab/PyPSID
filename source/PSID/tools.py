""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tools for system identification
"""

import numpy as np

def extractDiagonalBlocks(A, emptySide='both', absThr=np.spacing(1)):
    if emptySide == 'either':
        BLKSU = extractDiagonalBlocks(A, 'upper', absThr)
        BLKSL = extractDiagonalBlocks(A, 'lower', absThr)
        if len(BLKSU) >= len(BLKSL):
            BLKS = BLKSU
        else:
            BLKS = BLKSL
        return BLKS

    j = 0
    BLKS = np.empty(0, dtype=int)
    while j < A.shape[0]:
        if emptySide == 'both' or emptySide == 'upper':
            for j1 in range(j, A.shape[1]):
                if j1 == (A.shape[1]-1) or np.all(np.abs( A[j:(j1+1), (j1+1):] ) <= absThr):
                    i1 = j1 - j + 1
                    break
        
        if emptySide == 'both' or emptySide == 'lower':
            for j2 in range(j, A.shape[0]):
                if j2 == (A.shape[0]-1) or np.all(np.abs( A[(j2+1):, j:(j2+1)] ) <= absThr):
                    i2 = j2 - j + 1
                    break

        if emptySide == 'upper':
            i2 = i1
        elif emptySide == 'lower':
            i1 = i2
        
        i = j + int(np.max([i1, i2]))
        BLKS = np.concatenate((BLKS, [(i-j)]))
        j = i

    return BLKS

def getBlockIndsFromBLKSArray(BLKS):
    if len(BLKS) == 0:
        return np.empty(0, dtype=int)
    
    BLKSCUM = np.array(np.atleast_2d(np.cumsum(BLKS)).T, dtype=int)
    groups = np.concatenate((
        np.concatenate((np.zeros((1,1), dtype=int), BLKSCUM[:-1, :]), axis=0),
        BLKSCUM
    ), axis=1)

    return groups