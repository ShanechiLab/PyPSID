""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

import warnings

import numpy as np
from scipy import linalg

from . import LSSM
from . import PrepModel
from .PSID import blkhankskip, projOrth, getHSize

def removeProjOrth(A, B):
    """
    Projects A onto B and then subtracts the result from A. 
    A and B must be wide matrices with dim x samples.
    Returns:
    1) A_AHat: A minus the projection of A onto B
    """
    return A - projOrth(A, B)[0]

def projOblique(A, B, C):
    """
    Projects A onto B along C. 
    A, B and C must be wide matrices with dim x samples.
    Returns:
    1) AHat: projection of A onto B along C
    2) W: The matrix that gives AHat when it is right multiplied by B
    """
    if C is not None:
        A_C = removeProjOrth(A, C)
        B_C = removeProjOrth(B, C)
        W = projOrth(A_C, B_C)[1]
        AHat = W @ B
    else:
        AHat, W = projOrth(A, B)
    return AHat, W

def computeObsFromAC(A, C, i):
    """
    Computes the extended observability matrix for pair (A, C)
    Returns:
    1) Oy: extended observability matrix for (A, C)
    2) Oy_Minus: Oy, minus the last block row
    """
    ny = C.shape[0]
    Oy = C
    for ii in range(i):
        Oy = np.concatenate((Oy, 
            Oy[(ii-1)*ny:ii*ny, :] @ A
        ))
    Oy_Minus = Oy[:(-ny), :]
    return Oy, Oy_Minus

def recomputeObsAndStates(A, C, i, YHat, YHatMinus):
    """
    Computes observabilioty matrices Oy and Oy_Minus using A and C 
    and recompute Xk and Xk_Plus1 using the new Oy and Oy_Minus
    Returns:
    1) Xk: recomputed states 
    2) Xk_Plus1: recomputed states at next time step
    """
    Oy, Oy_Minus = computeObsFromAC(A, C, i)
    Xk = np.linalg.pinv(Oy) @ YHat      
    Xk_Plus1 = np.linalg.pinv(Oy_Minus) @ YHatMinus   
    return Xk, Xk_Plus1

def computeBD( A, C, Yii, Xk_Plus1, Xk, i, nu, Uf):
    """
    Computes matrices corresponding to the effect of external input
    Returns:
    1)B and 2)D matrices in the following state space equations
    x(k) = A * x(k) + B * u(k) + w(k)
    y(k) = Cy * x(k) + Dy * u(k) + v(k)
    """
    # Find B and D
    Oy, Oy_Minus = computeObsFromAC(A, C, i)
    
    # See VODM Book pages 125-127
    PP = np.concatenate(
        (Xk_Plus1 - A @ Xk,
        Yii       - C @ Xk )
    )

    L1 = A @ np.linalg.pinv(Oy)
    L2 = C @ np.linalg.pinv(Oy)
    
    nx = A.shape[0]
    ny = C.shape[0]
    
    ZM = np.concatenate(
        (np.zeros((nx,ny)), np.linalg.pinv( Oy_Minus ))
    , axis=1)
    
    # LHS * DB = PP  
    LHS = np.zeros( (PP.size, (nx+ny)*nu) )
    RMul = linalg.block_diag( np.eye(ny), Oy_Minus )
    
    NNAll = [] # VODM Book, (4.54), (4.57),..,(4.59)
    # Plug in the terms into NN
    for ii in range(i):
        NN = np.zeros( ((nx+ny), i*ny) )
        
        NN[      :nx , :((i-ii)*ny)] =   ZM[ :, (ii*ny): ] \
                                       - L1[ :, (ii*ny): ]
        NN[nx:(nx+ny), :((i-ii)*ny)] = - L2[ :, (ii*ny): ]
        if ii == 0:
            NN[nx:(nx+ny), :ny] = NN[nx:(nx+ny), :ny] + np.eye(ny)
        
        # Plug into LHS
        LHS = LHS + np.kron( Uf[(ii*nu):(ii*nu+nu), :].T, NN @ RMul)
        NNAll.append(NN)
    
    DBVec = np.linalg.lstsq( LHS, PP.flatten(order='F'), rcond=None )[0] 
    DB = np.reshape(DBVec, [nx+ny, nu], order='F')
    D = DB[:ny, :]
    B = DB[ny:(ny+nx), :]
    return B, D

def fitCzDzViaKFRegression(s, Y, Z, U=None, time_first=True, Cz=None, fit_Cz_via_KF=False, missing_marker=None):
    """
    Fits the behavior projection parameter Cz (and behavior feedthrough parameter Dz) by first estimating 
    the latent states with a Kalman filter and then using ordinary 
    least squares regression
    """
    if not isinstance(Y, (list, tuple)):
        if time_first:
            YTF = Y
            ZTF = Z
            UTF = U
        else:
            YTF = Y.T
            ZTF = Z.T
            if U is not None:
                UTF = U.T
            else:
                UTF = None
        xHat = s.kalman(YTF, UTF)[0]
    else:
        for yInd in range(len(Y)):
            if time_first:
                YTFThis = Y[yInd]
                ZTFThis = Z[yInd]
                if U is not None:
                    UTFThis = U[yInd]
                else:
                    UTFThis = None
            else:
                YTFThis = Y[yInd].T
                ZTFThis = Z[yInd].T
                if U is not None:
                    UTFThis = U[yInd].T
                else:
                    UTFThis = None
            xHatThis = s.kalman(YTFThis, UTFThis)[0]
            if yInd == 0:
                xHat = xHatThis
                ZTF = ZTFThis
                UTF = UTFThis
            else:
                xHat = np.concatenate( (xHat, xHatThis), axis=0)
                ZTF = np.concatenate( (ZTF, ZTFThis), axis=0)
                if UTFThis is not None:
                    UTF = np.concatenate( (UTF, UTFThis), axis=0)
    if missing_marker is not None:
        isNotMissing = np.logical_not(np.any(ZTF == missing_marker, axis=1))      
    else:
        isNotMissing = np.ones(ZTF.shape[0], dtype=bool)
    if fit_Cz_via_KF:
        if U is not None:
            CzDz = projOrth(ZTF[isNotMissing, :].T, np.concatenate(
                (xHat[isNotMissing, :].T, UTF[isNotMissing, :].T)
            ))[1]
            nx = xHat.shape[1]
            Cz = CzDz[:, :nx]
            Dz = CzDz[:, nx:]
        else:
            Cz = projOrth(ZTF[isNotMissing, :].T, xHat[isNotMissing, :].T)[1]
            Dz = None
    else:
        if U is not None:
            Dz = projOrth(ZTF[isNotMissing, :].T - Cz @ xHat[isNotMissing, :].T, UTF[isNotMissing, :].T)[1]
        else:
            Dz = None
    return Cz, Dz

def IPSID(Y, Z=None, U=None, nx=None, n1=0, i=None, WS=dict(), return_WS=False, \
                fit_Cz_via_KF=True, time_first=True, \
                remove_mean_Y=True, remove_mean_Z=True, remove_mean_U=True, 
                zscore_Y=False, zscore_Z=False, zscore_U=False, 
                missing_marker=None) -> LSSM:
    """
    IPSID: Input Preferential Subspace Identification Algorithm
    IPSID identifies a linear stochastic model for a signal y, while prioritizing
    the latent states that are predictive of another signal z, while a known external input  
    u is applied to the system. The complete model is as follows:
    [x1(k+1); x2(k+1)] = [A11 0; A21 A22] * [x1(k); x2(k)] + [B1;  B2] * u(k) + w(k)
                  y(k) =      [Cy1   Cy2] * [x1(k); x2(k)] + Dy * u(k) + v(k)
                  z(k) =      [Cz1   0  ] * [x1(k); x2(k)] + Dz * u(k) + e(k)
    x(k) = [x1(k); x2(k)] => Latent state time series
    x1(k) => Latent states related to y and z ( the pair (A11, Cz1) is observable )
    x2(k) => Latent states related to y but unrelated to z 
    u(k) => External input that was applied to the system
    Given training time series from y(k), z(k) and u(k), the dimension of x(k) 
    (i.e. nx), the dimension of x1(k) (i.e. n1), the algorithm finds 
    all model parameters and noise statistics:
        - A  : [A11 0; A21 A22]
        - B  : [B1     B2]
        - Cy : [Cy1   Cy2]
        - Cz : [Cz1     0]
        - Dy : [Dy]
        - Dz : [Dz]
        - Q  : Cov( w(k), w(k) )
        - R  : Cov( v(k), v(k) )
        - S  : Cov( w(k), v(k) )
    as well as the following model characteristics/parameters: 
        - G  : Cov( x(k+1), y(k) )
        - YCov: Cov( y(k), y(k) )
        - K: steady state stationary Kalman filter for estimating x from y
        - innovCov: covariance of innovation for the Kalman filter
        - P: covariance of Kalman predicted state error
        - xPCov: covariance of Kalman predicted state itself
        - xCov: covariance of the latent state
    
    Inputs:
        - (1) Y: Inputs signal 1 (e.g. neural signal). 
                Must be a T x ny matrix (unless time_first=False).
                It can also be a list of matrices, one for each data segment (e.g. trials):
                [y(1); y(2); y(3); ...; y(T)]
                Segments do not need to have the same number of samples.
        - (2) Z: Inputs signal 2, to be studied using y (e.g. behavior). 
                Format options are similar to Y. 
                Must be a T x nz matrix (unless time_first=False).
                It can also be a list of matrices, one for each data segment (e.g. trials):
                [z(1); z(2); z(3); ...; z(T)]
                Segments do not need to have the same number of samples.
        - (3) U: External inputs (e.g. task instructions). 
                Format options are similar to Y. 
                Must be a T x nu matrix (unless time_first=False).
                It can also be a list of matrices, one for each data segment (e.g. trials):
                [u(1); u(2); u(3); ...; u(T)]
                Segments do not need to have the same number of samples.
        - (4) nx: the total number of latent states in the stochastic model
        - (5) n1: number of latent states to extract in the first stage.
        - (6) i: the number of block-rows (i.e. future and past horizon). 
                Different values of i may have different identification performance. 
                Must be at least 2. It also determines the maximum n1 and nx 
                that can be used per:
                n1 <= nz * i
                nx <= ny * i
                So if you have a low dimensional y or z, you typically would choose larger 
                values for i, and vice versa.
                i Can also be a list, tuple, or array indicating [iY,iZ,iU], in which case
                different horizons will be used for Y, Z and U (for now only iY == iU is supported)
        - (7) WS: the WS output from a previous call using the exact 
                same data. If calling IPSID repeatedly with the same data 
                and horizon, several computationally costly steps can be 
                reused from before. Otherwise will be discarded.
        - (8) return_WS (default: False): if True, will return WS as the second output
        - (9) fit_Cz_via_KF (default: True): if True (preferred option), 
                refits Cz more accurately using a KF after all other 
                parameters are learned
        - (10) time_first (default: True): if True, will expect the time dimension 
                of the data to be the first dimension (e.g. Z is T x nz). If False, 
                will expect time to be the second dimension in all data 
                (e.g. Z is nz x T).
        - (11) remove_mean_Y: if True will remove the mean of Y. 
                    Must be True if data is not zero mean. Defaults to True.
        - (12) remove_mean_Z: if True will remove the mean of Z. 
                    Must be True if data is not zero mean. Defaults to True.
        - (13) remove_mean_U: if True will remove the mean of U. 
                    Must be True if data is not zero mean. Defaults to True.
        - (14) zscore_Y: if True will z-score Y. It is ok to set this to False,
                    but setting to True may help with stopping some dimensions of 
                    data from dominating others. Defaults to False.
        - (15) zscore_Z: if True will z-score Z. It is ok to set this to False,
                    but setting to True may help with stopping some dimensions of 
                    data from dominating others. Defaults to False.
        - (16) zscore_U: if True will z-score U. It is ok to set this to False,
                    but setting to True may help with stopping some dimensions of 
                    data from dominating others. Defaults to False.
        - (17) missing_marker (default: None): if not None, will discard samples of Z that 
                equal to missing_marker when fitting Cz. Only effective if fit_Cz_via_KF is
                True.
         
    Outputs:
        - (1) idSys: an LSSM object with the system parameters for 
                the identified system. Will have the following 
                attributes (defined above), and some more attributes 
                and methods:    
                'A', 'B', 'Cy', 'Cz', 'Dy', 'Dz', 'Q', 'R', 'S'
                'G', 'YCov', 'K', 'innovCov', 'P', 'xPCov', 'xCov' 
        - (2) WS (optional): dictionary to provide to later calls of PSID
                on the same data (see input (6) for more details)

    Usage example:
        idSys = IPSID(Y, Z, U, nx=nx, n1=n1, i=i);  # With external input
        idSysPSID = IPSID(Y, Z, nx=nx, n1=n1, i=i);     # No external input: PSID
        [idSys, WS] = IPSID(Y, Z, nx=nx, n1=n1, i=i, WS=WS);
        idSysISID = IPSID(Y, Z=None, U, nx, 0, i); # Set n1=0 and Z=None for ISID
        idSysSID = IPSID(Y, Z=None, U=None, nx, 0, i); # Set n1=0, Z=None and U=None for SID
    """
    if not isinstance(i, (list, tuple, np.ndarray)):
        i = [i]
    iAll = np.array(i)
    iY = int(iAll[0])  # Horizon for Y
    iZ = iY if iAll.size < 2 else int(iAll[1])  # Horizon for Z
    iU = iY if iAll.size < 3 else int(iAll[2])  # Horizon for U (must be the same as iY)
    iMax = np.max([iY,iZ,iU])

    YPrepModel = PrepModel.PrepModel()
    YPrepModel.fit(Y, remove_mean=remove_mean_Y, zscore=zscore_Y, time_first=time_first)
    Y = YPrepModel.apply(Y, time_first=time_first)
    
    ZPrepModel = PrepModel.PrepModel()
    if Z is not None:
        ZPrepModel.fit(Z, remove_mean=remove_mean_Z, zscore=zscore_Z, time_first=time_first)
        Z = ZPrepModel.apply(Z, time_first=time_first)
    
    UPrepModel = PrepModel.PrepModel()
    if U is not None:
        UPrepModel.fit(U, remove_mean=remove_mean_U, zscore=zscore_U, time_first=time_first)
        U = UPrepModel.apply(U, time_first=time_first)
    
    ny, ySamples, N, y1, NTot = getHSize(Y, iMax, time_first=time_first)
    if Z is not None:
        nz, zSamples, _, z1, NTot = getHSize(Z, iMax, time_first=time_first)
    else:
        nz, zSamples = 0, 0
    if U is not None:
        nu, uSamples, _, u1, NTot = getHSize(U, iMax, time_first=time_first)
    else:
        nu = 0

    if isinstance(N, list) and np.any(np.array(N) < 1):
        warnings.warn('{} of the {} data segments will be discarded because they are too short for using with a horizon of {}.'.format(np.sum(np.array(N) < 1), len(N), i), )

    if  'NTot' in WS and WS['NTot'] == NTot and \
        'N' in WS and WS['N'] == N and \
        'iY' in WS and WS['iY'] == iY and \
        'iZ' in WS and WS['iZ'] == iZ and \
        'iU' in WS and WS['iU'] == iU and \
        'ySamples' in WS and WS['ySamples'] == ySamples and \
        'zSamples' in WS and WS['zSamples'] == zSamples and \
        'Y1' in WS and WS['Y1'] == y1 and \
        (nz == 0 or ('Z1' in WS and WS['Z1'] == z1)):
        # Have WS from previous call with the same data
        pass
    else:
        WS = {
            'NTot': NTot,
            'N': N,
            'i': i, 'iY': iY, 'iZ': iZ, 'iU': iU, 
            'ySamples': ySamples,
            'Y1': y1
        }
        if nz > 0:
            WS['zSamples'] = zSamples
            WS['Z1'] = z1

    if 'Yp' not in WS or WS['Yp'] is None:
        WS['Yp'] = blkhankskip(Y, iY, N, iMax-iY, time_first=time_first)
        WS['Yf'] = blkhankskip(Y, iY, N, iMax, time_first=time_first)
        WS['Yii'] = blkhankskip(Y, 1, N, iMax, time_first=time_first)
        if nu > 0:
            WS['Up'] = blkhankskip(U, iU, N, iMax-iU, time_first=time_first)
            WS['Uf'] = blkhankskip(U, iU, N, iMax, time_first=time_first)
            WS['Uii'] = blkhankskip(U, 1, N, iMax, time_first=time_first)
        else:
            WS['Up'] = np.empty( (0, N) )
            WS['Uf'] = WS['Up']
            WS['Uii'] = WS['Up']
        if nz > 0:
            WS['Zii'] = blkhankskip(Z, 1, N, iMax, time_first=time_first)
    
    if n1 > nx:
        n1 = nx  # Max possible n1 value

    if n1 > 0 and nz > 0:
        if n1 > iZ*nz:
            raise(Exception('n1 (currently {}) must be at most iZ*nz={}*{}={}. Use a larger horizon iZ.'.format(n1,iZ,nz,iZ*nz)))
        if 'ZHatObUfRes_U' not in WS or WS['ZHatObUfRes_U'] is None:
            Zf = blkhankskip(Z, iZ, N, iMax, time_first=time_first)
            Zf_Minus = Zf[nz:, :]
            Uf_Minus = WS['Uf'][nu:, :]
            
            # IPSID Stage 1
            ####################################################
            # Oblique projection of Zf along Uf onto UpYp:
            ZHatOb = projOblique(Zf, np.concatenate( 
                (WS['Up'], WS['Yp'])
            ), WS['Uf'])[0] 
            WS['ZHatObUfRes'] = removeProjOrth(ZHatOb, WS['Uf'])

            # Orthogonal projection of Zf onto UpYpUf
            WS['ZHat'] = projOrth(Zf, np.concatenate( 
                (WS['Up'], WS['Yp'], WS['Uf'])
            ))[0] 
            
            # Orthogonal projection of Zf_Minus onto Up_plus, Yp_plus and Uf_Minus
            WS['ZHatMinus'] = projOrth(Zf_Minus, np.concatenate(
                (WS['Up'], WS['Uii'], WS['Yp'], WS['Yii'], Uf_Minus)
            ))[0]

            # Take SVD of ZHatObUfRes
            WS['ZHatObUfRes_U'], WS['ZHatObUfRes_S'], ZHat_V = linalg.svd(WS['ZHatObUfRes'], full_matrices=False, lapack_driver='gesvd')

        Sz = np.diag(WS['ZHatObUfRes_S'][:n1]) 
        Uz = WS['ZHatObUfRes_U'][:, :n1]  

        Oz = Uz @ Sz**(1/2)
        Oz_Minus = Oz[:-nz, :]

        Xk = np.linalg.pinv(Oz) @ WS['ZHat'];                   
        Xk_Plus1 = np.linalg.pinv(Oz_Minus) @ WS['ZHatMinus'];   
    else:
        n1 = 0
        Xk = np.empty([0, NTot])
        Xk_Plus1 = np.empty([0, NTot])

    n2 = nx - n1 

    # IPSID Stage 2
    # ----------------
    if n2 > 0:
        if nx > iY*ny:
            raise(Exception('nx (currently {}) must be at most iY*ny={}*{}={}. Use a larger horizon iY.'.format(nx,iY,ny,iY*ny)))
        if 'YHatObUfRes_U' not in WS or WS['YHatObUfRes_U'] is None or \
            'n1' not in WS or WS['n1'] != n1:
            WS['n1'] = n1
            
            Yf = WS['Yf']
            Yf_Minus = Yf[ny:, :]
            Uf_Minus = WS['Uf'][nu:, :]

            if n1 > 0: # Have already extracted some states, so remove the already predicted part of Yf
                # Remove the already predicted part of future y
                # Oblique projection of Yf along Uf, onto UpYp
                YHatOb1, Oy1 = projOblique(Yf, Xk, np.concatenate(
                    (WS['Up'], WS['Uf'])
                )) 
                Yf = Yf - YHatOb1 # Eq.(25)

                Oy1_Minus = Oy1[:-ny, :]
                Yf_Minus = Yf_Minus - Oy1_Minus @ Xk_Plus1
            
            # Oblique projection of Yf along Uf, onto UpYp: 
            YHatOb = projOblique(Yf, np.concatenate( 
                (WS['Up'], WS['Yp'])
            ), WS['Uf'])[0]
            WS['YHatObUfRes'] = removeProjOrth(YHatOb, WS['Uf']) 

            # Orthogonal projection of Yf onto UfUpYp
            WS['YHat'] = projOrth(Yf, np.concatenate(
                (WS['Up'], WS['Yp'], WS['Uf'])
            ))[0]
            
            # Orthogonal projection of Yf_Minus onto Up_plus,Yp_plus,Uf_Minus
            WS['YHatMinus'] = projOrth(Yf_Minus, np.concatenate(
                (WS['Up'], WS['Uii'], WS['Yp'], WS['Yii'], Uf_Minus)
            ))[0]

            # Take SVD of YHatObUfRes
            WS['YHatObUfRes_U'], WS['YHatObUfRes_S'], YHat_V = linalg.svd(WS['YHatObUfRes'], full_matrices=False, lapack_driver='gesvd') 
    
        S2 = np.diag(WS['YHatObUfRes_S'][:n2])
        U2 = WS['YHatObUfRes_U'][:, :n2]
    
        Oy = U2 @ S2**(1/2) 
        Oy_Minus = Oy[:-ny, :]

        Xk2 = np.linalg.pinv(Oy) @ WS['YHat'] 
        Xk2_Plus1 = np.linalg.pinv(Oy_Minus) @ WS['YHatMinus']

        Xk = np.concatenate((Xk, Xk2))
        Xk_Plus1 = np.concatenate((Xk_Plus1, Xk2_Plus1))

    # Parameter identification
    # ------------------------
    if n1 > 0:
        # A associated with the z-related states
        XkP1Hat, A1Tmp = projOrth( Xk_Plus1[:n1, :], np.concatenate(
            (Xk[:n1, :], WS['Uf'])
        )) 
        A = A1Tmp[:n1, :n1]
        w = Xk_Plus1[:n1, :] - XkP1Hat[:n1, :] 
    else:
        A = np.empty([0, 0])
        w = np.empty([0, N])
    
    if n2 > 0:
        # A associated with the other states (X2)
        XkP2Hat, A23Tmp = projOrth(Xk_Plus1[n1:, :], np.concatenate(
            (Xk, WS['Uf'])
        )) 
        A23 = A23Tmp[:, :nx]
        if n1 > 0:
            A10 = np.concatenate((A, np.zeros([n1,n2])), axis=1)
            A = np.concatenate((A10, A23))  
        else:
            A = A23
        w = np.concatenate((w, Xk_Plus1[n1:, :] - XkP2Hat)) 

    if nz > 0:
        ZiiHat, CzTmp = projOrth(WS['Zii'], np.concatenate(
            (Xk, WS['Uf'])
        )) 
        Cz = CzTmp[:, :nx]
        e = WS['Zii'] - ZiiHat
    else:
        Cz = np.empty([0, nx])

    YiiHat, CyTmp = projOrth(WS['Yii'], np.concatenate(
        (Xk, WS['Uf'])
    )) 
    Cy = CyTmp[:, :nx]
    v  = WS['Yii'] - YiiHat

    # Compute noise covariances
    NA = w.shape[1]
    Q = (w @ w.T)/NA # Eq.(36)
    S = (w @ v.T)/NA # Eq.(36)
    R = (v @ v.T)/NA # Eq.(36)

    Q = (Q + Q.T)/2      # Make precisely symmetric
    R = (R + R.T)/2      # Make precisely symmetric

    params = {
        'A': A,
        'C': Cy,
        'Q': Q,
        'R': R,
        'S': S
    }
    if nz > 0:
        params['Sxz'] = (w @ e.T)/NA
        params['Syz'] = (v @ e.T)/NA
        Rz = (e @ e.T)/NA
        params['Rz'] = (Rz + Rz.T)/2      # Make precisely symmetric

    s = LSSM.LSSM(params=params)
    if np.any(np.isnan(s.Pp)): # Riccati did not have a solution.
        warnings.warn('The learned model did not have a solution for the Riccati equation.')
    
    if nu > 0: # Following a procedure similar to VODM Book, pages 125-127 to find the least squares solution for the model parameters B and Dy
        RR = np.triu(
            np.linalg.qr( 
                np.concatenate((WS['Up'], WS['Uf'], WS['Yp'], WS['Yf'])).T / np.sqrt(NA) 
            )[1]
        ).T
        if iU != iY:
            raise(Exception('Only iY=iU is supported!'))
        RR = RR[:((2*nu+2*ny)*iY), :((2*nu+2*ny)*iY)]

        RUf = RR[ (nu*iY):(2*nu*iY) , :]
        RYf = RR[ ((2*nu+ny)*iY):((2*nu+2*ny)*iY) , :]
        RYf_Minus = RR[ ((2*nu+ny)*iY+ny):((2*nu+2*ny)*iY) , :]
        RYii = RR[ ((2*nu+ny)*iY):((2*nu+ny)*iY+ny) , :]

        YHat = np.concatenate(
            (RYf[:, :((2*nu+ny)*iY)], np.zeros((ny*iY, ny)))
        , axis=1)
        YHatMinus = RYf_Minus[:, :((2*nu+ny)*iY+ny)]
        Yii = RYii[:, :((2*nu+ny)*iY+ny)]
        Uf = RUf[:, :((2*nu+ny)*iY+ny)]
        
        # Recompute Oy and Oy_Minus using A and Cy and recompute Xk and Xk_Plus1 using the new Oy
        Xk, Xk_Plus1 = recomputeObsAndStates(A, Cy, iY, YHat, YHatMinus)
        B, Dy = computeBD( A, Cy, Yii, Xk_Plus1, Xk, iY, nu, Uf)
        s.changeParams({'B': B, 'D': Dy})
    
    s.Cz = Cz
    if nz > 0:
        Cz, Dz = fitCzDzViaKFRegression(s, Y, Z, U, time_first, Cz, fit_Cz_via_KF=fit_Cz_via_KF, missing_marker=missing_marker)
        if fit_Cz_via_KF:
            s.Cz = Cz
        if nu > 0:
            s.Dz = Dz

    s.YPrepModel = YPrepModel
    s.ZPrepModel = ZPrepModel
    s.UPrepModel = UPrepModel

    if not return_WS:
        return s
    else:
        return s, WS
