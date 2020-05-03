""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

An LSSM object for keeping parameters, filtering, etc
"""

import numpy as np
from scipy import linalg

def dict_get_either(d, fieldNames, defaultVal=None):
    for f in fieldNames:
        if f in d:
            return d[f]
    return defaultVal

class LSSM:
    def __init__(self, params, output_dim=None, state_dim=None, randomizationSettings=None):
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
            if S is None:
                S = np.zeros((self.state_dim, self.output_dim))
            S = np.atleast_2d(S)
            if S.shape[0] != self.state_dim:
                S = S.T
            self.S = S
        elif 'k' in params or 'K' in params:
            self.Q = None
            self.R = None
            self.S = None
            self.K = dict_get_either(params, ['K', 'k'], None)
            self.innovCov = dict_get_either(params, ['innovCov'], None)
            
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
        if self.Q is not None: # Given QRS
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
            except:
                print('Could not solve DARE')
                self.Pp = np.empty(self.A.shape); self.Pp[:] = np.nan
                self.K = np.empty((self.A.shape[0], self.R.shape[0])); self.K[:] = np.nan
                self.Kf = np.array(self.K)
                self.Kv = np.array(self.K)
                self.innovCov = np.empty(self.R.shape); self.innovCov[:] = np.nan
                self.A_KC = np.empty(self.A.shape); self.A_KC[:] = np.nan
            
            self.P2 = self.XCov - self.Pp # (should give the solvric solution) Proof: Katayama Theorem 5.3 and A.3 in pvo book
        elif self.K is not None: # Given K
            self.XCov = None
            self.G = None
            self.YCov = None
        
            self.Pp = None
            self.Kf = None
            self.Kv = None
            self.A_KC = self.A - self.K @ self.C
            self.P2 = None
    
    def isStable(self):
        return np.all(np.abs(self.eigenvalues) < 1)
    
    def kalman(self, Y, x0=None, P0=None, steady_state=True):
        if np.any(np.isnan(self.K)) and steady_state:
            steady_state = False
            print('Steady state Kalman gain not available. Will perform non-steady-state Kalman')
        N = Y.shape[0]
        allXp = np.empty((N, self.state_dim))  # X(i|i-1)
        allX = np.empty((N, self.state_dim))
        if x0 == None:
            x0 = np.zeros((self.state_dim, 1))
        if P0 == None:
            P0 = np.ones((self.state_dim, self.state_dim))
        Xp = x0
        Pp = P0
        for i in range(N):
            allXp[i, :] = np.transpose(Xp) # X(i|i-1)
            zi = Y[i, :][:, np.newaxis] - self.C @ Xp # Innovation Z(i)
            
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
            
            if Kf is not None:  # Otherwise cannot do filtering
                X = Xp + Kf @ zi # X(i|i)
                allX[i, :] = np.transpose(X)

            newXp = self.A @ Xp
            newXp += K @ zi

            Xp = newXp
            if not steady_state:
                Pp = self.A @ Pp @ self.A.T + self.Q - K @ ziCov @ K.T
        
        allYp = np.transpose(self.C @ allXp.T)
        return allXp, allYp, allX
    
    def predict(self, Y, useXFilt=False):
        allXp, allYp, allX = self.kalman(Y)
        if useXFilt:
            allXp = allX
        if (hasattr(self, 'Cz') and self.Cz is not None):
            allZp = (self.Cz @ allXp.T).T
        else:
            allZp = None
        return allZp, allYp, allXp
