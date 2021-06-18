""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

An LSSM object for keeping parameters, filtering, etc
"""
import warnings

import numpy as np
from scipy import linalg

def dict_get_either(d, fieldNames, defaultVal=None):
    for f in fieldNames:
        if f in d:
            return d[f]
    return defaultVal

def genRandomGaussianNoise(N, Q, m=None):
    Q2 = np.atleast_2d(Q)
    dim = Q2.shape[0]
    if m is None:
        m = np.zeros((dim, 1))
    
    D, V = linalg.eig(Q2)
    if np.any(D < 0):
        raise("Cov matrix is not PSD!")
    QShaping = np.real(np.matmul(V, np.sqrt(np.diag(D))))
    w = np.matmul(np.random.randn(N, dim), QShaping.T)
    return w, QShaping
    
class LSSM:
    def __init__(self, params, output_dim=None, state_dim=None):
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.setParams(params)
    
    def setParams(self, params = {}):
        A = dict_get_either(params, ['A', 'a'])
        A = np.atleast_2d(A)
        
        C = dict_get_either(params, ['C', 'c'])
        C = np.atleast_2d(C)
        
        self.A = A
        self.state_dim = self.A.shape[0]
        if C.shape[1] != self.state_dim and C.shape[0] == self.state_dim:
            C = C.T
        self.C = C
        self.output_dim = self.C.shape[0]
        
        if 'q' in params or 'Q' in params:  # Stochastic form with QRS provided
            Q = dict_get_either(params, ['Q', 'q'], None)
            R = dict_get_either(params, ['R', 'r'], None)
            S = dict_get_either(params, ['S', 's'], None)
            Q = np.atleast_2d(Q)
            R = np.atleast_2d(R)

            self.Q = Q
            self.R = R
            if S is None or S.size == 0:
                S = np.zeros((self.state_dim, self.output_dim))
            S = np.atleast_2d(S)
            if S.shape[0] != self.state_dim:
                S = S.T
            self.S = S
        elif 'k' in params or 'K' in params:
            self.Q = None
            self.R = None
            self.S = None
            self.K = np.atleast_2d(dict_get_either(params, ['K', 'k'], None))
            self.innovCov = np.atleast_2d(dict_get_either(params, ['innovCov'], None))
            
        self.update_secondary_params()

        for f, v in params.items(): # Add any remaining params (e.g. Cz)
            if not hasattr(self, f) and not hasattr(self, f.upper()) and \
                    f not in set(['sig', 'L0', 'P']): 
                setattr(self, f, v)

        if hasattr(self, 'Cz') and self.Cz is not None:
            Cz = np.atleast_2d(self.Cz)
            if Cz.shape[1] != self.state_dim and Cz.shape[0] == self.state_dim:
                Cz = Cz.T
                self.Cz = Cz
        
    def update_secondary_params(self):
        if self.Q is not None and self.state_dim > 0: # Given QRS
            try:
                A_Eigs = linalg.eig(self.A)[0]
            except Exception as e:
                print('Error in eig ({})... Tying again!'.format(e))
                A_Eigs = linalg.eig(self.A)[0] # Try again!
            isStable = np.max(np.abs(A_Eigs)) < 1
            if isStable:
                self.XCov = linalg.solve_discrete_lyapunov(self.A, self.Q)
                self.G = self.A @ self.XCov @ self.C.T + self.S
                self.YCov = self.C @ self.XCov @ self.C.T + self.R
                self.YCov = (self.YCov + self.YCov.T)/2
            else:
                self.XCov = np.ones(self.A.shape); self.XCov[:] = np.nan
                self.YCov = np.ones(self.A.shape); self.XCov[:] = np.nan

            try:
                self.Pp = linalg.solve_discrete_are(self.A.T, self.C.T, self.Q, self.R, s=self.S) # Solves Katayama eq. 5.42a
                self.innovCov = self.C @ self.Pp @ self.C.T + self.R
                innovCovInv = np.linalg.pinv( self.innovCov )
                self.K = (self.A @ self.Pp @ self.C.T + self.S) @ innovCovInv
                self.Kf = self.Pp @ self.C.T @ innovCovInv
                self.Kv = self.S @ innovCovInv
                self.A_KC = self.A - self.K @ self.C
            except Exception as err:
                print('Could not solve DARE: {}'.format(err))
                self.Pp = np.empty(self.A.shape); self.Pp[:] = np.nan
                self.K = np.empty((self.A.shape[0], self.R.shape[0])); self.K[:] = np.nan
                self.Kf = np.array(self.K)
                self.Kv = np.array(self.K)
                self.innovCov = np.empty(self.R.shape); self.innovCov[:] = np.nan
                self.A_KC = np.empty(self.A.shape); self.A_KC[:] = np.nan
            
            self.P2 = self.XCov - self.Pp # (should give the solvric solution) Proof: Katayama Theorem 5.3 and A.3 in pvo book
        elif hasattr(self, 'K') and self.K is not None: # Given K
            self.XCov = None
            self.G = None
            self.YCov = None
        
            self.Pp = None
            self.Kf = None
            self.Kv = None
            self.A_KC = self.A - self.K @ self.C
            self.P2 = None
        elif self.R is not None:
            self.YCov = self.R
    
    def isStable(self):
        return np.all(np.abs(self.eigenvalues) < 1)
    
    def generateObservationFromStates(self, X, param_names=['C'], prep_model_param='YPrepModel'):
        """Can generate Y or Z observation time series given the latent state time series X

        Args:
            X (numpy array): Dimensions are time x dimesions.
            param_names (list, optional): The name of the read-out parameter. Defaults to ['C'].
            prep_model_param (str, optional): The name of the preprocessing model parameter. 
                        Defaults to 'YPrepModel'.

        Returns:
            numpy array: The observation time series. 
                If param_names=['C'] and prep_model_param='YPrepModel', will 
                    produce Y = C * X, and then applies the inverse of the 
                    Y preprocessing model.
                If param_names=['Cz'] and prep_model_param='ZPrepModel', will 
                    produce Y = Cz * X, and then applies the inverse of the 
                    Z preprocessing model.
        """        
        Y = None
        if hasattr(self, param_names[0]):
            C = getattr(self, param_names[0])
        else:
            C = None

        if C is not None and C.size > 0:
            ny = C.shape[0] if C is not None and self.C.size > 0 else D.shape[0]
            N = X.shape[0]
            Y = np.zeros((N, ny))
            if C is not None and C.size > 0:
                Y += (C @ X.T).T
            
        if prep_model_param is not None and hasattr(self, prep_model_param):
            prep_model_param_obj = getattr(self, prep_model_param)
            if prep_model_param_obj is not None:
                Y = prep_model_param_obj.apply_inverse(Y) # Apply inverse of any mean-removal/zscoring

        return Y
    
    def generateRealization(self, N, x0=None, w0=None, return_z=False):
        QRS = np.block([[self.Q,self.S], [self.S.T,self.R]])
        wv, self.QRSShaping = genRandomGaussianNoise(N, QRS)
        w = wv[:, :self.state_dim]
        v = wv[:, self.state_dim:]
        if x0 is None:
            x0 = np.zeros((self.state_dim, 1))
        if w0 is None:
            w0 = np.zeros((self.state_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        for i in range(N):
            if i == 0:
                Xt_1 = x0
                Wt_1 = w0
            else:
                Xt_1 = X[i-1, :].T
                Wt_1 = w[i-1, :].T
            X[i, :] = (self.A @ Xt_1 + Wt_1).T
            # Y[i, :] = (self.C @ X[i, :].T + v[i, :].T).T # Will make Y later
        Y = v + self.generateObservationFromStates(X, param_names=['C'], prep_model_param='YPrepModel')
        if return_z == False:
            return Y, X
        else:
            Z = self.generateObservationFromStates(X, param_names=['Cz'], prep_model_param='ZPrepModel')
            return Y, X, Z
    
    def kalman(self, Y, x0=None, P0=None, steady_state=True, return_state_cov=False):
        if self.state_dim == 0:
            allXp = np.zeros((Y.shape[0], self.state_dim))
            allXf = allXp
            allYp = np.zeros((Y.shape[0], self.output_dim))
            return allXp, allYp, allXf
        if np.any(np.isnan(self.K)) and steady_state:
            steady_state = False
            warnings.warn('Steady state Kalman gain not available. Will perform non-steady-state Kalman.')
        N = Y.shape[0]
        allXp = np.empty((N, self.state_dim))  # X(i|i-1)
        allXf = np.empty((N, self.state_dim))  # X(i|i)
        if return_state_cov:
            allPp = np.zeros((N,self.state_dim,self.state_dim))  # P(i|i-1) 
            allPf = np.zeros((N,self.state_dim,self.state_dim))  # P(i|i)
        if x0 == None:
            x0 = np.zeros((self.state_dim, 1))
        if P0 == None:
            P0 = np.eye(self.state_dim)
        Xp = x0
        Pp = P0
        for i in range(N):
            allXp[i, :] = np.transpose(Xp) # X(i|i-1)
            thisY = Y[i, :][np.newaxis, :]
            if hasattr(self, 'YPrepModel') and self.YPrepModel is not None:
                thisY = self.YPrepModel.apply(thisY, time_first=True) # Apply any mean removal/zscoring
            zi = thisY.T - self.C @ Xp # Innovation Z(i)
            
            if steady_state:
                Kf = self.Kf
                K = self.K
            else:
                ziCov = self.C @ Pp @ self.C.T + self.R
                Kf = np.linalg.lstsq(ziCov.T, (Pp @ self.C.T).T, rcond=None)[0].T  # Kf(i)

                if self.S.size > 0:
                    Kw = np.linalg.lstsq(ziCov.T, self.S.T, rcond=None)[0].T   # Kw(i)
                    K = self.A @ Kf + Kw                    # K(i)
                else:
                    K = self.A @ Kf                         # K(i)    

                P = Pp - Kf @ self.C @ Pp                   # P(i|i)

                if return_state_cov:
                    allPp[i, :, :] = Pp  # P(i|i-1)
                    allPf[i, :, :] = P    # P(i|i)
            
            if Kf is not None:  # Otherwise cannot do filtering
                X = Xp + Kf @ zi # X(i|i)
                allXf[i, :] = np.transpose(X)

            newXp = self.A @ Xp
            newXp += K @ zi

            Xp = newXp
            if not steady_state:
                Pp = self.A @ Pp @ self.A.T + self.Q - K @ ziCov @ K.T
        
        allYp = self.generateObservationFromStates(allXp, param_names=['C'], prep_model_param='YPrepModel')

        if not return_state_cov:
            return allXp, allYp, allXf
        else:
            return allXp, allYp, allXf, allPp, allPf

    def predict(self, Y, useXFilt=False, **kwargs):
        allXp, allYp, allXf = self.kalman(Y, **kwargs)[0:3]
        if useXFilt:
            allXp = allXf
        if (hasattr(self, 'Cz') and self.Cz is not None):
            allZp = self.generateObservationFromStates(allXp, param_names=['Cz'], prep_model_param='ZPrepModel')
        else:
            allZp = None
        return allZp, allYp, allXp
