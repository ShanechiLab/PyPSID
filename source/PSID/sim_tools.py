""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Tools for simulating models
"""

import os, copy, re, logging

import numpy as np

from .tools import extractDiagonalBlocks, getBlockIndsFromBLKSArray
from .LSSM import LSSM

logger = logging.getLogger(__name__)

def extractRangeParamFromSysCode(sysCode, prefix=''):
    regex = re.compile(prefix+"R([\\d\\.e+-]+)_([\\d\\.e+-]+)") #NxR1_10
    matches = re.finditer(regex, sysCode)
    paramVals = None
    pos = None
    for matchNum, match in enumerate(matches, start=1):
        p = match.groups()
        if len(p) == 2 and np.all([pt!='' for pt in p]):
            paramVals = [float(p[0]), float(p[1])] # np.arange(int(p[0]), 1+int(p[1]))
            pos = match.regs
    if paramVals is None:
        regex = re.compile(prefix+"([\\d\\.e+-]+)") #Nx4
        matches = re.finditer(regex, sysCode)
        for matchNum, match in enumerate(matches, start=1):
            p = match.groups()
            paramVals = [float(p[0])] #np.arange(int(p[0]), 1+int(p[1]))
            pos = match.regs
    return paramVals, pos

def getSysSettingsFromSysCode(sysCode):
    sysSettings = {}
    prefixFieldPairs = [
        ('Nx', 'nxVals'), ('N1', 'n1Vals'), ('nu', 'nuVals'),
        ('nxu', 'nxuVals'), ('ny', 'nyVals'), ('nz', 'nzVals'),
        ('Ne', 'nxZErrVals')
    ]
    for prefix, settingsField in prefixFieldPairs:
        rng = extractRangeParamFromSysCode(sysCode, prefix=prefix)[0]
        if rng is not None:
            if len(rng)==2 and rng[1] == rng[0]: rng.pop(-1)
            if len(rng)==1: rng.append(rng[0])
            sysSettings[settingsField] = np.arange(int(rng[0]), 1+int(rng[1]))
        else:
            sysSettings[settingsField] = None
    if sysSettings['nuVals'] is None or len(sysSettings['nuVals']) == 0:
        sysSettings['nuVals'] = np.arange(0,1)
    if sysSettings['nxuVals'] is None or len(sysSettings['nxuVals']) == 0:
        sysSettings['nxuVals'] = np.arange(0,1)

    sysSettings['xNScLR'] = extractRangeParamFromSysCode(sysCode, prefix='xNScL')[0]
    sysSettings['yNScLR'] = extractRangeParamFromSysCode(sysCode, prefix='yNScL')[0]
    sysSettings['zSNRLR'] = extractRangeParamFromSysCode(sysCode, prefix='zSNRL')[0]
    sysSettings['yZNZRLR'] = extractRangeParamFromSysCode(sysCode, prefix='yZNZRL')[0]

    for p in ['A', 'K', 'Cy', 'Cz']:
        sysSettings[p+'_args'] = {}

    return sysSettings

def generateRandomLinearModel(sysSettings):
    nx = np.random.choice(sysSettings['nxVals'])
    n1 = np.random.choice(sysSettings['n1Vals'][sysSettings['n1Vals']<=nx]) if np.any(np.array(sysSettings['n1Vals'])<=nx) else nx
    nu = np.random.choice(sysSettings['nuVals'])
    nxu = np.random.choice(sysSettings['nxuVals'])
    ny = np.random.choice(sysSettings['nyVals'])
    nz = np.random.choice(sysSettings['nzVals'])
    if 'nxZErrVals' in sysSettings and sysSettings['nxZErrVals'] is not None:
        nxZErr = np.random.choice(sysSettings['nxZErrVals'])
    else:
        nxZErr = 0

    if 'S0' not in sysSettings:
        sysSettings['S0'] = False

    if 'Dyz' not in sysSettings:
        sysSettings['Dyz'] = False

    if 'predictor_form' not in sysSettings:
        sysSettings['predictor_form'] = False

    if nu > 0:
        sysU = LSSM(state_dim=nxu, output_dim=nu, input_dim=0)   # Input model
    else:
        sysU = None

    for attempt_ind in range(10):
        try:
            s = LSSM(state_dim=nx, output_dim=ny, input_dim=nu, randomizationSettings={
                    'n1': n1, 'S0': sysSettings['S0'],
                    'predictor_form': sysSettings['predictor_form']
                })
            break
        except Exception as e:
            pass

    if nxZErr > 0:
        zErrSys = LSSM(state_dim=nxZErr, output_dim=nz, input_dim=0)
        Cz = np.random.randn(nz, nxZErr)
        zErrSys.changeParams({'Cz': Cz})
        s.zErrSys = zErrSys
    else:
        zErrSys = None
    tmp = np.random.randn(nz, nz)
    Rz = tmp @ np.transpose(tmp)

    zDims = np.array([], dtype=int)
    if n1 > 0 and n1 == nx:
        zDims = np.arange(n1, dtype=int)
    elif n1 > 0: # Determine dimensions used for z
        if not sysSettings['predictor_form']:
            BLKS = extractDiagonalBlocks(s.A, emptySide='lower', absThr=1e-14)
        else:
            absThr = np.min(np.abs(np.linalg.eig(s.A_KC)[0]))/1e5
            BLKS = extractDiagonalBlocks(s.A_KC, emptySide='lower', absThr=absThr)
        groupInds = getBlockIndsFromBLKSArray(BLKS)
        twoBLocks = np.nonzero(BLKS  > 1)[0]
        oneBlocks = np.nonzero(BLKS == 1)[0]
        
        tbCnt = np.min([twoBLocks.size, int(np.floor(n1/2))])
        if oneBlocks.size == 0 and 2*tbCnt < n1: # Odd n1 but no 1x1 blocks
            tbCnt += 1
        for si in range(0, tbCnt):
            groupInd = groupInds[twoBLocks[si]]
            blockZDims = [*range(groupInd[0], groupInd[1],)]  
            zDims = np.concatenate((zDims, blockZDims))
        if len(zDims) > n1:
            zDims = zDims[:-1]
        si = -1
        while len(zDims) < n1:
            si += 1
            groupInd = groupInds[oneBlocks[si]]
            blockZDims = [*range(groupInd[0], groupInd[1])]  
            zDims = np.concatenate((zDims, blockZDims))
        zDims = np.array(zDims, dtype=int)

        # Move z-dims to the top
        I = np.eye(nx)
        E = np.concatenate( (I[zDims, :], I[~np.isin(np.arange(nx), zDims), :]), axis=0 )
        s.applySimTransform(E)
        zDims = np.array(np.arange(zDims.size), dtype=int)

    Cz1 = np.random.randn(nz, n1)
    Cz = np.zeros((nz, nx))
    if n1 > 0:
        Cz[:, zDims] = Cz1
    Dz = np.random.randn(nz, nu)
    s.changeParams({
        'Cz': Cz, 
        'Dz': Dz, 
        'Rz': Rz, 
        'zDims': zDims+1 # plus 1 to be consistent with Matlab indices for zDims in simulated systems
    })

    if sysSettings['Dyz']:
        Dyz = np.random.randn(nz, ny)
        s.changeParams({
            'Dyz': Dyz, 
        })

    if 'xNScLR' in sysSettings and sysSettings['xNScLR'] is not None:
        rng = copy.copy(sysSettings['xNScLR'])
        rngL = list(np.log10(rng))
        if len(rngL) == 1: rngL.append(rngL[0])
        sc = 10 ** ( np.random.rand()*np.diff(rngL) + rngL[0] )
        s.changeParams({'Q': sc**2 * s.Q, 'S': sc * s.S})

    if 'yNScLR' in sysSettings and sysSettings['yNScLR'] is not None:
        # s.YCov s.innovCov
        rng = copy.copy(sysSettings['yNScLR'])
        rngL = list(np.log10(rng))
        if len(rngL) == 1: rngL.append(rngL[0])
        sc = 10 ** ( np.random.rand()*np.diff(rngL) + rngL[0] )
        s.changeParams({'R': sc**2 * s.R, 'S': sc * s.S})

    if 'yZNZRLR' in sysSettings and sysSettings['yZNZRLR'] is not None:
        rng = copy.copy(sysSettings['yZNZRLR'])
        rngL = list(np.log10(rng))
        if len(rngL) == 1: rngL.append(rngL[0])
        sc = 10 ** ( np.random.rand()*np.diff(rngL) + rngL[0] )
        zDimInds = np.array([ind for ind in np.arange(s.state_dim) if ind in (s.zDims - 1)])
        nonZDims = np.array([ind for ind in np.arange(s.state_dim) if ind not in (s.zDims - 1)])
        Cy = np.copy(s.C)
        CyXCovZDims = Cy[:, zDimInds] @ s.XCov[np.ix_(zDimInds, zDimInds)] @ Cy[:, zDimInds].T
        CyXCovNonZDims = Cy[:, nonZDims] @ s.XCov[np.ix_(nonZDims, nonZDims)] @ Cy[:, nonZDims].T
        ZNZRatio = np.diag(CyXCovZDims) / np.diag(CyXCovNonZDims)
        CyZDimsScalesPerRow = np.sqrt( sc / ZNZRatio )
        newCy = np.copy(s.C)
        newCy[:, zDimInds] = np.diag(CyZDimsScalesPerRow) @ newCy[:, zDimInds]
        s.changeParams({'C': newCy})

    if 'zSNRLR' in sysSettings and sysSettings['zSNRLR'] is not None:
        rng = copy.copy(sysSettings['zSNRLR'])
        rngL = list(np.log10(rng))
        if len(rngL) == 1: rngL.append(rngL[0])
        desiredZSNR = 10 ** ( np.random.rand()*np.diff(rngL) + rngL[0] )
        if not sysSettings['predictor_form']:
            XCov = s.XCov
        else:
            XCov = s.P2
        sigCov = s.Cz @ XCov @ s.Cz.T
        noiseCov = np.copy(s.Rz)
        if hasattr(s, 'zErrSys') and s.zErrSys is not None:
            noiseCov += s.zErrSys.Cz @ s.zErrSys.XCov @ s.zErrSys.Cz.T
        sigToNoiseStdRatio = np.sqrt(np.diag(sigCov)) / np.sqrt(np.diag(noiseCov))
        CzRowScales = desiredZSNR / sigToNoiseStdRatio
        newCz = np.diag(CzRowScales) @ s.Cz
        s.changeParams({'Cz': newCz})

    return s, sysU, zErrSys