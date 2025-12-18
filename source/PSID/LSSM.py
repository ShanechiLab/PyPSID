"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

An LSSM object for keeping parameters, filtering, etc
"""

import copy
import logging
import warnings

import numpy as np
from scipy import linalg, optimize
from tqdm import tqdm


logger = logging.getLogger(__name__)


def dict_get_either(d, fieldNames, defaultVal=None):
    """Retrieves the first found key from a list of possible keys in a dictionary.

    Args:
        d (dict): The dictionary to search.
        fieldNames (list): List of keys to look for.
        defaultVal (any, optional): Value to return if no key is found. Defaults to None.

    Returns:
        any: The value corresponding to the first found key, or defaultVal.
    """
    for f in fieldNames:
        if f in d:
            return d[f]
    return defaultVal


def genRandomGaussianNoise(N, Q, m=None):
    """Generates random Gaussian noise with a given covariance.

    Args:
        N (int): Number of samples.
        Q (np.ndarray): Covariance matrix.
        m (np.ndarray, optional): Mean vector. Defaults to None (zeros).

    Returns:
        tuple: (noise samples, shaping matrix used)
    """
    Q2 = np.atleast_2d(Q)
    dim = Q2.shape[0]
    if m is None:
        m = np.zeros((dim, 1))

    D, V = linalg.eig(Q2)
    if np.any(D < 0):
        isTinyNeg = np.logical_and(D < 0, D > -1e-12)
        logger.warning(
            f"Q had {np.sum(isTinyNeg)}/{len(isTinyNeg)} tiny negative values that were within machine precision, setting them to 0"
        )
        D[isTinyNeg] = 0
    if np.any(D < 0):
        raise ("Cov matrix is not PSD!")
    QShaping = np.real(np.matmul(V, np.sqrt(np.diag(D))))
    w = np.matmul(np.random.randn(N, dim), QShaping.T)
    return w, QShaping


def solve_discrete_are_iterative(
    a, b, q, r, s=None, thr=1e-12, max_iter=10000, return_log=False
):
    """Solves the Discrete Algebraic Riccati Equation iteratively.

    Args:
        a, b, q, r, s: System matrices (A, B, Q, R, S).
        thr (float): Convergence threshold.
        max_iter (int): Maximum iterations.
        return_log (bool): Whether to return convergence log.

    Returns:
        np.ndarray: Solution matrix P.
    """
    A = a.T
    C = b.T
    Q, R, S = q, r, s

    Pp = np.eye(A.shape[0])
    PpPrev = Pp
    PpChanges = []
    for i in range(max_iter):
        ziCov = C @ Pp @ C.T + R
        Kf = np.linalg.lstsq(ziCov.T, (Pp @ C.T).T, rcond=None)[0].T  # Kf(i)

        if S is not None and S.size > 0:
            Kw = np.linalg.lstsq(ziCov.T, S.T, rcond=None)[0].T  # Kw(i)
            K = A @ Kf + Kw  # K(i)
        else:
            K = A @ Kf  # K(i)
        PpNew = A @ Pp @ A.T + Q - K @ ziCov @ K.T
        if np.isinf(np.linalg.norm(PpNew)):
            raise (
                Exception(
                    "Could not solve Riccati iteratively because the solution diverged."
                )
            )
        PpChange = np.linalg.norm(Pp - PpNew) / np.linalg.norm(Pp)
        PpChange2 = np.linalg.norm(PpPrev - PpNew) / np.linalg.norm(PpPrev)
        PpChanges.append(PpChange)
        PpPrev = Pp
        Pp = PpNew
        if PpChange < thr:
            break
        elif PpChange / 1e6 > PpChange2 and PpChange2 < thr:
            raise (
                Exception(
                    "There are two solutions to DARE that we are oscillating between."
                )
            )
    if return_log:
        return Pp, PpChanges
    else:
        return Pp


def solveric(A, G, C, L0):
    """
    P = solvric(A,G,C,L0)

    This solves the forward riccati equation with the following formulation
    P = A P A' + (G - A P C') (L0 - C P C')^{-1} (G - A P C')'

    To solve the backward riccati equation, transpose all parameters and swap the roles of G and C.
    N = A' N A + (C' - A' N G) (L0 - G' N G)^{-1} (C' - A' N G)'

    N = solvric(A.T,C.T,G.T,L0.T)

    Reference:
    The solveric matlab function accompanying the 1996 VODM subspace identification book.
    """
    nx = A.shape[0]
    L0_inv = np.linalg.inv(L0)

    # Matrices for eigenvalue decomposition
    AA = np.concatenate(
        (
            np.concatenate((A.T - C.T @ L0_inv @ G.T, np.zeros((nx, nx))), axis=1),
            np.concatenate((-G @ L0_inv @ G.T, np.eye(nx)), axis=1),
        ),
        axis=0,
    )

    BB = np.concatenate(
        (
            np.concatenate((np.eye(nx), -C.T @ L0_inv @ C), axis=1),
            np.concatenate((np.zeros((nx, nx)), A - G @ L0_inv @ C), axis=1),
        ),
        axis=0,
    )

    w, vr = linalg.eig(AA, BB, left=False, right=True)
    # the normalized right eigenvector corresponding to the eigenvalue w[i] is the column vr[:,i].
    # Expect: AA@vr - BB @ vr @ np.diag(w) = 0

    # If there's an eigenvalue on the unit circle => no solution
    has_solution = np.all(np.abs(np.abs(w) - 1) > 1e-9)

    # Sort the eigenvalues
    inds = np.argsort(np.abs(w))

    LHS = vr[nx : (2 * nx), inds[:nx]]
    RHS = vr[:nx, inds[:nx]]

    P = np.real(np.linalg.lstsq(RHS.T, LHS.T)[0].T)  # real(LHS/RHS)
    # LHS/RHS => Solve RHS.T x.T = LHS.T => x = np.linalg.lstsq(RHS.T, LHS.T)[0].T
    # Explanation based on: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    # b/a => Solve a.T x.T = b.T instead => solution of x a = b for x
    # a\b => linalg.solve(a, b) if a is square; linalg.lstsq(a, b) otherwise => solution of a x = b for x

    return P, has_solution


def solve_Faurres_realization_problem(A, C, G, L0):
    """Solves the stochastic realization problem using Faurre's algorithm.

    Args:
        A (np.ndarray): System matrix.
        C (np.ndarray): Observation matrix.
        G (np.ndarray): Next state-output covariance.
        L0 (np.ndarray): Output covariance.

    Returns:
        Q (np.ndarray): Process noise covariance.
        R (np.ndarray): Observation noise covariance.
        S (np.ndarray): Cross covariance.
        sig (np.ndarray): State covariance (solution to ARE).
    """
    [minSig, has_solution1] = solveric(
        A, G, C, L0
    )  # From Katayama page 191, Algorithm 2
    [maxSigInv, has_solution2] = solveric(
        A.T, C.T, G.T, L0.T
    )  # From Katayama page 191, Algorithm 1
    maxSig = np.linalg.inv(maxSigInv)

    # Expecting minSig <= maxSig
    if (
        not has_solution1
        or not has_solution2
        or np.any(np.linalg.eigvals(maxSig - minSig) < 0)
    ):
        raise (Exception("No solution for Faurre" "s stochastic realization problem"))

    # Any minSig <= sig <= maxSig is a solution
    #     sig = minSig; % This is one risky solution, we can use it as initial point and try to get away from both minSig and maxSig

    # Try to find some solution in the middle
    # Easy to prove that (minSig+maxSig)/2 is also a solution to Faurre's problem
    sig = (minSig + maxSig) / 2

    if np.linalg.norm(sig - sig.T, 1) > 100 * 1e-16 * np.linalg.norm(sig, 1):
        warnings.warn(
            "The solution to Faurre"
            "s realization problem is not completely symmetric (norm(sig-sig"
            ")=%.3g)... We will forefully make symmetric but be careful!".format(
                np.linalg.norm(sig - sig.T, 1)
            )
        )

    sig = (sig + sig.T) / 2  # Just to remove any numerical inaccuracies

    if np.any(np.linalg.eigvals(maxSig - sig) < 0) or np.any(
        np.linalg.eigvals(sig - minSig) < 0
    ):
        raise (Exception("Incorrect solution to Faurre" "s realization problem!"))

    Q = sig - A @ sig @ A.T
    R = L0 - C @ sig @ C.T
    S = G - A @ sig @ C.T

    R = (R + R.T) / 2  # Just to remove any numerical inaccuracies
    Q = (Q + Q.T) / 2  # Just to remove any numerical inaccuracies

    # We know QRS is exactly symmetric so it must be PSD
    # Let's check to see if QRS are close to being singular
    QRS = np.concatenate(
        (
            np.concatenate((Q, S), axis=1),
            np.concatenate((S.T, R), axis=1),
        ),
        axis=0,
    )
    QRSEigs = np.linalg.eigvals(QRS)
    if np.any(QRSEigs < 0):
        warnings.warn("QRS is close to singular so it is almost not PSD!")

    return Q, R, S, sig


def makeMatrixBlockDiagonal(A, n1=None, name="A", ignore_error=False):
    """Converts a matrix to block diagonal form based on provided block sizes.

    Args:
        A (np.ndarray): The matrix to be block-diagonalized.
        n1 (int, optional): Size of the first block. Defaults to None.
        name (str, optional): Name of the matrix for error logging. Defaults to "A".
        ignore_error (bool, optional): If True, suppresses errors. Defaults to False.

    Returns:
        E (np.ndarray): The similarity transformation matrix.
    """
    eigs = np.linalg.eigvals(A)
    if n1 is not None:
        # Important: we select n1 eigvals for the top to allow proper n1 vs n2 blocking
        rEigs = eigs[np.isreal(eigs)]
        iEigsP = eigs[np.logical_and(~np.isreal(eigs), np.imag(eigs) > 0)]
        iEigsN = np.conjugate(iEigsP)
        topEigs = np.concatenate(
            (
                iEigsP[: int(np.floor(n1 / 2))],
                iEigsN[: int(np.floor(n1 / 2))],
            )
        )
        topEigs = np.concatenate((topEigs, rEigs[: (n1 - topEigs.size)]))
        if topEigs.size != n1:
            msg = f"cannot achieve the desired block structure of an n1xn1 ({n1}x{n1}) top-left block in {name}"
            if ignore_error:
                logger.warning(msg)
            else:
                logger.error(msg)
                raise (Exception(msg))
    else:
        topEigs = eigs
    # Note: np.linalg.schur passes the real part of eigs to the sortFunc
    sortFunc = (
        lambda eig: topEigs.size > 0 and np.min(np.abs(eig - np.real(topEigs))) < 1e-6
    )  # Moves real eigs to the top left
    # Important: we transpose A to get schur for upper triangular
    J, EInv, sdim = linalg.schur(A.T, output="real", sort=sortFunc)
    E = np.linalg.inv(EInv)
    return E


class LSSM:
    """Linear State Space Model class.

    Represents a linear dynamical system of the form:
    x(k+1) = A x(k) + B u(k) + w(k)
    y(k)   = C x(k) + D u(k) + v(k)
    """

    def __init__(
        self,
        output_dim=None,
        state_dim=None,
        input_dim=0,
        params=None,
        randomizationSettings=None,
        missing_marker=None,
    ):
        """Initializes the LSSM object.

        Args:
            output_dim (int, optional): Dimension of the output (y). Defaults to None.
            state_dim (int, optional): Dimension of the state (x). Defaults to None.
            input_dim (int, optional): Dimension of the input (u). Defaults to 0.
            params (dict, optional): Dictionary containing model parameters (A, C, Q, R, S, etc.). Defaults to None.
            randomizationSettings (dict, optional): Settings for random initialization if params is None. Defaults to None.
            missing_marker (float, optional): Value indicating missing data. Defaults to None.
        """
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.missing_marker = missing_marker
        if params is None:
            self.randomize(randomizationSettings)
        else:
            self.setParams(params)
        self.verbose = False

    def setParams(self, params={}):
        """Sets the parameters of the LSSM.

        Args:
            params (dict): Dictionary of parameters (A, C, Q, R, etc.).
        """
        A = dict_get_either(params, ["A", "a"])
        A = np.atleast_2d(A)

        C = dict_get_either(params, ["C", "c"])
        C = np.atleast_2d(C)

        self.A = A
        self.state_dim = self.A.shape[0]
        if C.shape[1] != self.state_dim and C.shape[0] == self.state_dim:
            C = C.T
        self.C = C
        self.output_dim = self.C.shape[0]

        B = dict_get_either(params, ["B", "b"], None)
        D = dict_get_either(params, ["D", "d", "Dy", "dy"], None)
        if isinstance(B, float) or (isinstance(B, np.ndarray) and B.size > 0):
            B = np.atleast_2d(B)
            if B.shape[0] != self.state_dim:
                B = B.T
            self.input_dim = B.shape[1]
        elif isinstance(D, float) or (isinstance(D, np.ndarray) and D.size > 0):
            D = np.atleast_2d(D)
            if D.shape[0] != self.output_dim:
                D = D.T
            self.input_dim = D.shape[1]
        else:
            self.input_dim = 0
        if B is None or B.size == 0:
            B = np.zeros((self.state_dim, self.input_dim))
        B = np.atleast_2d(B)
        if (
            B.size > 0
            and B.shape[0] != self.state_dim
            and B.shape[1] == self.output_dim
        ):
            B = B.T
        self.B = B
        if D is None or D.size == 0:
            D = np.zeros((self.output_dim, self.input_dim))
        D = np.atleast_2d(D)
        if (
            D.size > 0
            and D.shape[0] != self.output_dim
            and D.shape[1] == self.output_dim
        ):
            D = D.T
        self.D = D

        if "q" in params or "Q" in params:  # Stochastic form with QRS provided
            Q = dict_get_either(params, ["Q", "q"], None)
            R = dict_get_either(params, ["R", "r"], None)
            S = dict_get_either(params, ["S", "s"], None)
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
        elif "k" in params or "K" in params:
            self.Q = None
            self.R = None
            self.S = None
            self.K = np.atleast_2d(dict_get_either(params, ["K", "k"], None))
            self.innovCov = np.atleast_2d(dict_get_either(params, ["innovCov"], None))

        self.update_secondary_params()

        for f, v in params.items():  # Add any remaining params (e.g. Cz)
            if f in set(["Cz", "Dz"]) or (
                not hasattr(self, f)
                and not hasattr(self, f.upper())
                and f not in set(["sig", "L0", "P"])
            ):
                setattr(self, f, v)

        if hasattr(self, "Cz") and self.Cz is not None:
            Cz = np.atleast_2d(self.Cz)
            if Cz.shape[1] != self.state_dim and Cz.shape[0] == self.state_dim:
                Cz = Cz.T
                self.Cz = Cz

        if not hasattr(self, "x0"):
            self.x0 = np.zeros((self.state_dim,))  # Initial state
        if not hasattr(self, "P0"):
            self.P0 = np.eye(self.state_dim)  # Initial state noise covariance
        if not hasattr(self, "predictWithXSmooth"):
            self.predictWithXSmooth = False  # If True, the predict method will use smoothed x unless an useXSmooth=False argument is passed.
        if not hasattr(self, "predictWithXFilt"):
            self.predictWithXFilt = False  # If True (and predictWithXSmooth = False), the predict method will use filtered x unless an useXSmooth=False argument is passed.

    def changeParams(self, params={}):
        """Updates the LSSM parameters.

        Args:
            params (dict): Dictionary of parameters to update.
        """
        curParams = self.getListOfParams()
        for f, v in curParams.items():
            if f not in params:
                params[f] = v
        self.setParams(params)

    def getListOfParams(self):
        """Returns a dictionary of all model parameters.

        Returns:
            dict: Dictionary of parameters.
        """
        params = {}
        for field in dir(self):
            val = self.__getattribute__(field)
            if not field.startswith("__") and isinstance(
                val, (np.ndarray, list, tuple, type(self))
            ):
                params[field] = val
        return params

    def randomize(self, randomizationSettings=None):
        """Randomizes the model parameters.

        Args:
            randomizationSettings (dict, optional): Settings for random initialization. Defaults to None.
        """
        if randomizationSettings is None:
            randomizationSettings = dict()

        if "n1" not in randomizationSettings:
            randomizationSettings["n1"] = self.state_dim

        if "S0" not in randomizationSettings:
            randomizationSettings["S0"] = False

        if "stable_A" not in randomizationSettings:
            randomizationSettings["stable_A"] = True

        if "stable_A_KC" not in randomizationSettings:
            randomizationSettings["stable_A_KC"] = True

        if "eigDist" not in randomizationSettings:
            randomizationSettings["eigDist"] = {}

        if "zEigDist" not in randomizationSettings:
            randomizationSettings["zEigDist"] = randomizationSettings["eigDist"]

        from .sim_tools import (
            generate_random_eigenvalues,
        )  # Import here to avoid circular import

        isOk = False
        while not isOk:
            n1 = randomizationSettings["n1"]
            if (
                "predictor_form" in randomizationSettings
                and randomizationSettings["predictor_form"]
            ):
                A_KC_eigs = generate_random_eigenvalues(
                    n1, randomizationSettings["zEigDist"]
                )
                if self.state_dim > n1:
                    eigs2 = generate_random_eigenvalues(
                        self.state_dim - n1, randomizationSettings["eigDist"]
                    )
                    A_KC_eigs = np.concatenate((A_KC_eigs, eigs2))
                A_KC, ev = linalg.cdf2rdf(A_KC_eigs, np.eye(self.state_dim))
                self.A_KC = A_KC
                self.K = np.random.randn(self.state_dim, self.output_dim)
                self.C = np.random.randn(self.output_dim, self.state_dim)

                self.A = self.A_KC + self.K @ self.C
                self.eigenvalues = np.linalg.eigvals(self.A)

                if (
                    np.any(np.abs(self.eigenvalues) > 1)
                    and randomizationSettings["stable_A"]
                ):
                    isOk = False
                    continue

                tmp = np.random.randn(self.output_dim, self.output_dim)
                self.innovCov = tmp @ np.transpose(tmp)
                P2 = linalg.solve_discrete_lyapunov(
                    self.A, self.K @ self.innovCov @ self.K.T
                )
                self.YCov = self.C @ P2 @ self.C.T + self.innovCov

                self.G = self.K @ self.innovCov + self.A @ P2 @ self.C.T
                self.P2 = P2

                P2New, has_solution = solveric(self.A, self.G, self.C, self.YCov)
                if np.linalg.norm(P2New - P2) / np.linalg.norm(P2) > 1e-6:
                    raise (
                        Exception(
                            "Riccati solution is not consistent with Lyapanov solution"
                        )
                    )

                self.Q = None
                try:
                    # Solve Faurre's realization problem to find Q, R, and S
                    Q, R, S, sig = solve_Faurres_realization_problem(
                        self.A, self.C, self.G, self.YCov
                    )
                    GNew = self.A @ sig @ self.C.T + S
                    YCovNew = self.C @ sig @ self.C.T + R
                    PpNew = linalg.solve_discrete_are(
                        self.A.T, self.C.T, Q, R, s=S
                    )  # Solves Katayama eq. 5.42a
                    innovCovNew = self.C @ PpNew @ self.C.T + R
                    innovCovInv = np.linalg.pinv(innovCovNew)
                    KNew = (self.A @ PpNew @ self.C.T + S) @ innovCovInv
                    if (
                        np.linalg.norm(GNew - self.G) / np.linalg.norm(self.G) > 1e-6
                        or np.linalg.norm(YCovNew - self.YCov)
                        / np.linalg.norm(self.YCov)
                        > 1e-6
                        or np.linalg.norm(KNew - self.K) / np.linalg.norm(self.K) > 1e-6
                    ):
                        raise (
                            Exception(
                                "Faurre's realization solution is not consistent with the original predictor form parameters"
                            )
                        )
                    self.Q, self.R, self.S = Q, R, S
                except Exception as e:
                    logger.info(
                        "Error in updating secondary parameters:\n{}\nRegerating random noises...".format(
                            e
                        )
                    )
                    isOk = False
                pass
            else:
                self.eigenvalues = generate_random_eigenvalues(
                    n1, randomizationSettings["zEigDist"]
                )
                if self.state_dim > n1:
                    eigs2 = generate_random_eigenvalues(
                        self.state_dim - n1, randomizationSettings["eigDist"]
                    )
                    self.eigenvalues = np.concatenate((self.eigenvalues, eigs2))
                self.A, ev = linalg.cdf2rdf(self.eigenvalues, np.eye(self.state_dim))
                self.C = np.random.randn(self.output_dim, self.state_dim)

                tmp = np.random.randn(
                    self.output_dim + self.state_dim, self.output_dim + self.state_dim
                )
                QRS = tmp @ np.transpose(tmp)
                self.Q = QRS[: self.state_dim, : self.state_dim]
                self.S = QRS[: self.state_dim, self.state_dim :]
                self.R = QRS[self.state_dim :, self.state_dim :]
                if randomizationSettings["S0"]:
                    self.S *= 0

                if (
                    "ySNR" in randomizationSettings
                    and randomizationSettings["ySNR"] is not None
                ):
                    try:
                        self.update_secondary_params()
                        ySNR = np.diag((self.C @ self.XCov @ self.C.T) / self.R)
                        CRowScale = np.sqrt(randomizationSettings["ySNR"] / ySNR)
                        self.C = np.diag(CRowScale) @ self.C
                    except:
                        continue
                        pass

            if self.input_dim > 0:
                self.B = np.random.randn(self.state_dim, self.input_dim)
                self.D = np.random.randn(self.output_dim, self.input_dim)

            try:
                self.update_secondary_params()
                if self.state_dim > 0:
                    A_KC_Eigs = linalg.eig(self.A_KC)[0]
                    if (
                        np.any(np.abs(A_KC_Eigs) > 1)
                        and randomizationSettings["stable_A_KC"]
                    ):
                        isOk = False
                        continue
                    A_eigs = linalg.eig(self.A)[0]
                    if np.any(np.abs(A_eigs) > 1) and randomizationSettings["stable_A"]:
                        isOk = False
                        continue
                isOk = True
            except Exception as e:
                logger.info(
                    "Error in updating secondary parameters:\n{}\nRegerating random noises...".format(
                        e
                    )
                )
                pass

    def update_secondary_params(self):
        """Updates secondary parameters (G, YCov, etc.) based on primary parameters (A, C, Q, R).

        Solves Lyapunov or Riccati equations as needed.
        """
        if self.Q is not None and self.state_dim > 0:  # Given QRS
            try:
                A_Eigs = linalg.eig(self.A)[0]
            except Exception as e:
                logger.info("Error in eig ({})... Tying again!".format(e))
                A_Eigs = linalg.eig(self.A)[0]  # Try again!
            isStable = np.max(np.abs(A_Eigs)) < 1
            if isStable:
                self.XCov = linalg.solve_discrete_lyapunov(self.A, self.Q)
                self.G = self.A @ self.XCov @ self.C.T + self.S
                self.YCov = self.C @ self.XCov @ self.C.T + self.R
                self.YCov = (self.YCov + self.YCov.T) / 2
            else:
                self.XCov = np.eye(self.A.shape[0])
                self.XCov[:] = np.nan
                self.YCov = np.eye(self.C.shape[0])
                self.YCov[:] = np.nan
                self.G = np.ones((self.A.shape[0], self.C.shape[0]))
                self.G[:] = np.nan

            try:
                try:
                    self.Pp = linalg.solve_discrete_are(
                        self.A.T, self.C.T, self.Q, self.R, s=self.S
                    )  # Solves Katayama eq. 5.42a
                except Exception as err:
                    logger.info("Could not solve DARE: {}".format(err))
                    logger.info("Attempting to solve iteratively")
                    self.Pp = solve_discrete_are_iterative(
                        self.A.T, self.C.T, self.Q, self.R, s=self.S
                    )  # Solves Katayama eq. 5.42a
                self.innovCov = self.C @ self.Pp @ self.C.T + self.R
                innovCovInv = np.linalg.pinv(self.innovCov)
                self.K = (self.A @ self.Pp @ self.C.T + self.S) @ innovCovInv
                self.Kf = self.Pp @ self.C.T @ innovCovInv
                self.Kv = self.S @ innovCovInv
                self.A_KC = self.A - self.K @ self.C
            except Exception as err:
                logger.info("Could not solve DARE: {}".format(err))
                self.Pp = np.empty(self.A.shape)
                self.Pp[:] = np.nan
                self.K = np.empty((self.A.shape[0], self.R.shape[0]))
                self.K[:] = np.nan
                self.Kf = np.array(self.K)
                self.Kv = np.array(self.K)
                self.innovCov = np.empty(self.R.shape)
                self.innovCov[:] = np.nan
                self.A_KC = np.empty(self.A.shape)
                self.A_KC[:] = np.nan

            self.P2 = (
                self.XCov - self.Pp
            )  # (should give the solvric solution) Proof: Katayama Theorem 5.3 and A.3 in VODM book
        elif hasattr(self, "K") and self.K is not None:  # Given K
            self.XCov = None
            if not hasattr(self, "G"):
                self.G = None
            if not hasattr(self, "YCov"):
                self.YCov = None
            self.Pp = None
            self.Kf = None
            self.Kv = None
            self.A_KC = self.A - self.K @ self.C
            if not hasattr(self, "P2"):
                self.P2 = None
        elif self.R is not None:
            self.YCov = self.R
        if self.input_dim > 0 or (hasattr(self, "B") and hasattr(self, "D")):
            self.B_KD = self.B - self.K @ self.D
        if self.state_dim == 0:
            self.A_KC = self.A

    def getBackwardModel(self):
        """Computes the backward-time LSSM.

        Returns:
            LSSM: The backward-time equivalent model.
        """
        # Sources:
        # - Katayama Lemma 4.12
        # - VODM page 63, backward model, Fig. 3.5
        newXCov = np.linalg.inv(self.XCov)
        newA = self.A.T
        # newC = ((self.C @ self.XCov @ self.A.T + self.S.T) @ newXCov).T # WRONG!
        newC = self.G.T
        if hasattr(self, "Cz"):
            Sxz = (
                self.Sxz
                if hasattr(self, "Sxz")
                else np.zeros((self.state_dim, self.Cz.shape[0]))
            )
            # newCz = (self.Cz @ self.XCov @ self.A.T + Sxz.T) @ newXCov  # WRONG!
            Gz = self.A @ self.XCov @ self.Cz.T + Sxz
            newCz = Gz.T  # Gz is the backward model for Cz
        newQ = newXCov - self.A.T @ newXCov @ self.A
        newS = self.C.T - self.A.T @ newXCov @ newC.T
        newR = self.YCov - newC @ newXCov @ newC.T
        bw = LSSM(params={"A": newA, "C": newC, "Q": newQ, "R": newR, "S": newS})
        if hasattr(self, "Cz"):
            bw.Cz = newCz
        if hasattr(self, "zDims"):
            bw.zDims = self.zDims
        return bw

    def isStable(self):
        """Checks if the system matrix A is stable (all eigenvalues inside unit circle).

        Returns:
            bool: True if stable, False otherwise.
        """
        return np.all(np.abs(self.eigenvalues) < 1)

    def generateObservationFromStates(
        self,
        X,
        u=None,
        param_names=["C", "D"],
        prep_model_param="YPrepModel",
        mapping_param="",
    ):
        """Can generate Y or Z observation time series given the latent state time series X and optional external input u

        Args:
            X (numpy array): Dimensions are time x dimesions.
            param_names (list, optional): The name of the read-out parameter. Defaults to ['C'].
            prep_model_param (str, optional): The name of the preprocessing model parameter.
                        Defaults to 'YPrepModel'.
            mapping_param (str, optional): The name of the mapping parameter.
                        Defaults to ''.
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
        if len(param_names) > 1 and hasattr(self, param_names[1]):
            D = getattr(self, param_names[1])
        else:
            D = None

        if C is not None and C.size > 0 or D is not None and D.size > 0:
            ny = C.shape[0] if C is not None and self.C.size > 0 else D.shape[0]
            N = X.shape[0]
            Y = np.zeros((N, ny))
            if C is not None and C.size > 0:
                Y += (C @ X.T).T
            if D is not None and D.size > 0 and u is not None:
                if hasattr(self, "UPrepModel") and self.UPrepModel is not None:
                    u = self.UPrepModel.apply(
                        u, time_first=True
                    )  # Apply any mean removal/zscoring
                Y += (D @ u.T).T

        if prep_model_param is not None and hasattr(self, prep_model_param):
            prep_model_param_obj = getattr(self, prep_model_param)
            if prep_model_param_obj is not None:
                Y = prep_model_param_obj.apply_inverse(
                    Y
                )  # Apply inverse of any mean-removal/zscoring

        if mapping_param is not None and hasattr(self, mapping_param):
            mapping_param_obj = getattr(self, mapping_param)
            if mapping_param_obj is not None and hasattr(mapping_param_obj, "map"):
                Y = mapping_param_obj.map(Y)
        return Y

    def generateObservationNoiseRealization(
        self, N, cov_param_name=None, sys_param_name=None
    ):
        """Generates a realization of observation noise.

        Args:
            N (int): Number of samples.
            cov_param_name (str, optional): Name of covariance parameter attributes.
            sys_param_name (str, optional): Name of system parameter attribute (for colored noise).

        Returns:
            np.ndarray: Generated noise.
        """
        err = None

        if cov_param_name is not None and hasattr(self, cov_param_name):
            R = getattr(self, cov_param_name)
            if R is not None and R.size > 0:
                err2 = genRandomGaussianNoise(N, R)[0]
                err = err + err2 if err is not None else err2

        if sys_param_name is not None and hasattr(self, sys_param_name):
            errSys = getattr(self, sys_param_name)
            if errSys is not None:
                err2 = errSys.generateRealization(N, return_z=True)[2]
                err = err + err2 if err is not None else err2

        return err

    def generateZRealizationFromStates(self, X=None, U=None, N=None, return_err=False):
        """Generates Z observations from state sequence.

        Args:
            X (np.ndarray, optional): State sequence.
            U (np.ndarray, optional): Input sequence.
            N (int, optional): Number of samples.
            return_err (bool, optional): Whether to return error term individually.

        Returns:
            np.ndarray: Generated Z.
        """
        if X is not None or U is not None:
            Z = self.generateObservationFromStates(
                X, u=U, param_names=["Cz", "Dz"], prep_model_param="ZPrepModel"
            )
            N = X.shape[0] if X is not None else U.shape[0]
        else:
            Z = None
        if N is not None:
            ZErr = self.generateObservationNoiseRealization(
                N, cov_param_name="Rz", sys_param_name="zErrSys"
            )
            if ZErr is not None:
                Z = Z + ZErr if Z is not None else ZErr
        if return_err is False:
            return Z
        else:
            return Z, ZErr

    def generateRealizationWithQRS(
        self,
        N,
        x0=None,
        w0=None,
        u0=None,
        u=None,
        wv=None,
        return_z=False,
        return_z_err=False,
        return_noise=False,
        blowup_threshold=np.inf,
        reset_x_on_blowup=False,
        randomize_x_on_blowup=False,
    ):
        """Generates a realization using Q, R, S covariances.

        Args:
            N (int): Number of time steps.
            x0, u0: Initial conditions.
            u (np.ndarray): Input sequence.
            return_z (bool): If True, returns Z.

        Returns:
            tuple: Generated data (Y, X, ...)
        """
        QRS = np.block([[self.Q, self.S], [self.S.T, self.R]])
        wv, self.QRSShaping = genRandomGaussianNoise(N, QRS)
        w = wv[:, : self.state_dim]
        v = wv[:, self.state_dim :]
        if x0 is None:
            if hasattr(self, "x0"):
                x0 = self.x0
            else:
                x0 = np.zeros((self.state_dim, 1))
        if len(x0.shape) == 1:
            x0 = x0[:, np.newaxis]
        if w0 is None:
            w0 = np.zeros((self.state_dim, 1))
        if self.input_dim > 0 and u0 is None:
            u0 = np.zeros((self.input_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        tqdm_disabled = not hasattr(self, "verbose") or not self.verbose
        for i in tqdm(range(N), "Generating realization", disable=tqdm_disabled):
            if i == 0:
                Xt_1 = x0
                Wt_1 = w0
                if self.input_dim > 0 and u is not None:
                    Ut_1 = u0
            else:
                Xt_1 = X[i - 1, :].T
                Wt_1 = w[i - 1, :].T
                if self.input_dim > 0 and u is not None:
                    Ut_1 = u[i - 1, :].T
            X[i, :] = (self.A @ Xt_1 + Wt_1).T
            # Y[i, :] = (self.C @ X[i, :].T + v[i, :].T).T # Will make Y later
            if u is not None:
                X[i, :] += np.squeeze((self.B @ Ut_1).T)
                # Y[i, :] += np.squeeze((self.D @ u[i, :]).T) # Will make Y later
            # Check if X[i, :] has blown up
            if (
                np.any(np.isnan(X[i, :]))
                or np.any(np.isinf(X[i, :]))
                or np.any(np.abs(X[i, :]) > blowup_threshold)
            ):
                msg = f"Xp blew up at sample {i} (mean Xp={np.mean(X[i, :]):.3g})"
                if reset_x_on_blowup:
                    X[i, :] = x0
                    msg += f", so it was reset to initial x0 (mean x0={np.mean(X[i, :]):.3g})"
                if randomize_x_on_blowup:
                    X[i, :] = np.atleast_2d(
                        np.random.multivariate_normal(
                            mean=np.zeros(self.state_dim), cov=self.XCov
                        )
                    ).T
                    msg += f", so it was reset to a random Gaussian x0 with XCov (mean x0={np.mean(X[i, :]):.3g})"
                logger.warning(msg)
        Y = v
        CxDu = self.generateObservationFromStates(
            X, u=u, param_names=["C", "D"], prep_model_param="YPrepModel"
        )
        if CxDu is not None:
            Y += CxDu
        out = Y, X
        if return_z:
            Z, ZErr = self.generateZRealizationFromStates(X=X, U=u, return_err=True)
            out += (Z,)
            if return_z_err:
                out += (ZErr,)
        if return_noise:
            out += (wv,)
        return out

    def generateRealizationWithKF(
        self,
        N,
        x0=None,
        u0=None,
        u=None,
        e=None,
        return_z=False,
        return_z_err=False,
        return_noise=False,
        blowup_threshold=np.inf,
        reset_x_on_blowup=False,
        randomize_x_on_blowup=False,
    ):
        """Generates a realization of data using the Kalman Filter.

        Args:
            N (int): Number of time steps to generate.
            x0 (np.ndarray, optional): Initial state. Defaults to None.
            u0 (np.ndarray, optional): Initial input. Defaults to None.
            u (np.ndarray, optional): Input data sequence. Defaults to None.
            e (np.ndarray, optional): Innovation noise sequence. Defaults to None.
            return_z (bool, optional): If True, returns Z observation. Defaults to False.
            return_z_err (bool, optional): If True, returns Z observation error. Defaults to False.
            return_noise (bool, optional): If True, returns noise sequence. Defaults to False.
            blowup_threshold (float, optional): Threshold for state blowup. Defaults to inf.
            reset_x_on_blowup (bool, optional): If True, resets state on blowup. Defaults to False.
            randomize_x_on_blowup (bool, optional): If True, randomizes state on blowup. Defaults to False.

        Returns:
            tuple: Generated data (Y, X, [Z, ZErr, noise])
        """
        if e is None:
            e, innovShaping = genRandomGaussianNoise(N, self.innovCov)
        if x0 is None:
            if hasattr(self, "x0"):
                x0 = self.x0
            else:
                x0 = np.zeros((self.state_dim, 1))
        if len(x0.shape) == 1:
            x0 = x0[:, np.newaxis]
        if self.input_dim > 0 and u0 is None:
            u0 = np.zeros((self.input_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        Xp = x0
        tqdm_disabled = not hasattr(self, "verbose") or not self.verbose
        for i in tqdm(range(N), "Generating realization", disable=tqdm_disabled):
            ek = e[i, :][:, np.newaxis]
            yk = self.C @ Xp + ek
            if u is not None:
                yk += self.D @ u[i, :][:, np.newaxis]
            X[i, :] = np.squeeze(Xp)
            # Y[i, :] = np.squeeze(yk) # Will make Y later
            Xp = self.A_KC @ Xp + self.K @ yk
            # Xp = self.A @ Xp + self.K @ ek
            if u is not None:
                Ut = u[i, :][:, np.newaxis]
                Xp += self.B_KD @ Ut
                # Y[i, :] += self.D @ u[i, :] # Will make Y later
            # Check if Xp has blown up
            if (
                np.any(np.isnan(Xp))
                or np.any(np.isinf(Xp))
                or np.any(np.abs(Xp) > blowup_threshold)
            ):
                msg = f"Xp blew up at sample {i} (mean Xp={np.mean(Xp):.3g})"
                if reset_x_on_blowup:
                    Xp = x0
                    msg += (
                        f", so it was reset to initial x0 (mean x0={np.mean(Xp):.3g})"
                    )
                if randomize_x_on_blowup:
                    Xp = np.atleast_2d(
                        np.random.multivariate_normal(
                            mean=np.zeros(self.state_dim), cov=self.XCov
                        )
                    ).T
                    msg += f", so it was reset to a random Gaussian x0 with XCov (mean x0={np.mean(Xp):.3g})"
                logger.warning(msg)
        Y = e + self.generateObservationFromStates(
            X, u=u, param_names=["C", "D"], prep_model_param="YPrepModel"
        )
        out = Y, X
        if return_z:
            Z, ZErr = self.generateZRealizationFromStates(X=X, U=u, return_err=True)
            out += (Z,)
            if return_z_err:
                out += (ZErr,)
        if return_noise:
            out += (e,)
        return out

    def generateRealization(self, N, random_x0=False, **kwargs):
        """Generates a realization of the system (simulation).

        Args:
            N (int): Number of time steps.
            random_x0 (bool, optional): If True, uses random initial state. Defaults to False.
            **kwargs: Additional arguments passed to generateRealizationWithQRS or generateRealizationWithKF.

        Returns:
            tuple: Generated data (Y, X, ...)
        """
        if random_x0 and "x0" not in kwargs and self.state_dim > 0:
            if not np.any(np.isnan(self.XCov)):
                kwargs["x0"] = np.atleast_2d(
                    np.random.multivariate_normal(
                        mean=np.zeros(self.state_dim), cov=self.XCov
                    )
                ).T
            else:
                logger.info(
                    "Count not generate random x0 because XCov in the model is not PSD"
                )
        if self.R is not None and "e" not in kwargs:
            return self.generateRealizationWithQRS(N, **kwargs)
        else:
            return self.generateRealizationWithKF(N, **kwargs)

    def kalman(
        self, Y, U=None, x0=None, P0=None, steady_state=True, return_state_cov=False
    ):
        """Applies the Kalman filter associated with the LSSM to some observation time-series

        Args:
            Y (np.ndarray): observation time series (time first).
            U (np.ndarray, optional): input time series (time first). Defaults to None.
            x0 (np.ndarray, optional): Initial Kalman state. Defaults to None.
            P0 (np.ndarray, optional): Initial Kalman state estimation error covariance. Defaults to None.
            steady_state (bool, optional): If True, will use steady state Kalman gain, which is much faster. Defaults to True.
            return_state_cov (bool, optional): If true, will return state error covariances. Defaults to False.

        Returns:
            allXp (np.ndarray): one-step ahead predicted states (t|t-1)
            allYp (np.ndarray): one-step ahead predicted observations (t|t-1)
            allXf (np.ndarray): filtered states (t|t)
            allPp (np.ndarray): error cov for one-step ahead predicted states (t|t-1)
            allPf (np.ndarray): error cov for filtered states (t|t)
        """
        if self.state_dim == 0:
            allXp = np.zeros((Y.shape[0], self.state_dim))
            allXf = allXp
            allYp = np.zeros((Y.shape[0], self.output_dim))
            return allXp, allYp, allXf
        if np.any(np.isnan(self.K)) and steady_state:
            steady_state = False
            warnings.warn(
                "Steady state Kalman gain not available. Will perform non-steady-state Kalman."
            )
        N, ny = Y.shape[0], Y.shape[1]
        allXp = np.nan * np.ones((N, self.state_dim))  # X(i|i-1)
        allXf = np.nan * np.ones((N, self.state_dim))  # X(i|i)
        if return_state_cov:
            allPp = np.zeros((N, self.state_dim, self.state_dim))  # P(i|i-1)
            allPf = np.zeros((N, self.state_dim, self.state_dim))  # P(i|i)
        if x0 is None:
            if hasattr(self, "x0"):
                x0 = self.x0
            else:
                x0 = np.zeros((self.state_dim, 1))
        if len(x0.shape) == 1:
            x0 = x0[:, np.newaxis]
        if P0 is None:
            if hasattr(self, "P0"):
                P0 = self.P0
            else:
                P0 = np.eye(self.state_dim)
        Xp = x0
        Pp = P0
        tqdm_disabled = not hasattr(self, "verbose") or not self.verbose
        for i in tqdm(range(N), "Estimating latent states", disable=tqdm_disabled):
            allXp[i, :] = np.transpose(Xp)  # X(i|i-1)
            thisY = Y[i, :][np.newaxis, :]
            if hasattr(self, "YPrepModel") and self.YPrepModel is not None:
                thisY = self.YPrepModel.apply(
                    thisY, time_first=True
                )  # Apply any mean removal/zscoring

            is_missing = (
                self.missing_marker is not None
                and (
                    np.any(thisY == self.missing_marker)
                    or (np.isnan(self.missing_marker) and np.any(np.isnan(thisY)))
                )
            ) or np.ma.is_masked(thisY)

            if not is_missing:
                zi = np.array(thisY.T) - self.C @ Xp  # Innovation Z(i)
            else:
                zi = np.zeros((ny, 1))  # Observation is missing

            if U is not None:
                ui = U[i, :][:, np.newaxis]
                if hasattr(self, "UPrepModel") and self.UPrepModel is not None:
                    ui = self.UPrepModel.apply(
                        ui, time_first=False
                    )  # Apply any mean removal/zscoring
                if self.D.size > 0:
                    zi -= self.D @ ui

            if steady_state:
                Kf = self.Kf
                K = self.K
            else:
                if np.linalg.norm(Pp) > 1e100:
                    warnings.warn("Kalman's Riccati recursion blew up... resetting Pp")
                    Pp = P0
                if not is_missing:
                    ziCov = self.C @ Pp @ self.C.T + self.R
                else:
                    ziCov = np.zeros((ny, ny))
                try:
                    Kf = np.linalg.lstsq(ziCov.T, (Pp @ self.C.T).T, rcond=None)[
                        0
                    ].T  # Kf(i)

                    if self.S.size > 0:
                        Kw = np.linalg.lstsq(ziCov.T, self.S.T, rcond=None)[
                            0
                        ].T  # Kw(i)
                        K = self.A @ Kf + Kw  # K(i)
                    else:
                        K = self.A @ Kf  # K(i)

                    P = Pp - Kf @ self.C @ Pp  # P(i|i)
                except RuntimeError as e:
                    logger.info(e)
                    pass

                if return_state_cov:
                    allPp[i, :, :] = Pp  # P(i|i-1)
                    allPf[i, :, :] = P  # P(i|i)

            if Kf is not None:  # Otherwise cannot do filtering
                X = Xp + Kf @ zi  # X(i|i)
                allXf[i, :] = np.transpose(X)

            newXp = self.A @ Xp
            if (
                hasattr(self, "useA_KC_plus_KC_in_KF") and self.useA_KC_plus_KC_in_KF
            ):  # TEEMP, just for unit tests of DPADModel
                newXp = (self.A_KC + self.K @ self.C) @ Xp
            newXp += K @ zi
            if U is not None and self.B.size > 0:
                newXp += self.B @ ui

            Xp = newXp
            if not steady_state:
                Pp = self.A @ Pp @ self.A.T + self.Q - K @ ziCov @ K.T

        allYp = self.generateObservationFromStates(
            allXp,
            u=U,
            param_names=["C", "D"],
            prep_model_param="YPrepModel",
            mapping_param="cMapY",
        )

        if not return_state_cov:
            return allXp, allYp, allXf
        else:
            return allXp, allYp, allXf, allPp, allPf

    def kalmanSmoother(
        self, Y, U=None, x0=None, P0=None, steady_state=False, return_state_cov=False
    ):
        """Applies the Kalman smoother associated with the LSSM to some observation time-series

        Args:
            Y (np.ndarray): observation time series (time first)
            U (np.ndarray, optional): input time series (time first). Defaults to None.
            x0 (np.ndarray, optional): Initial Kalman state. Defaults to None.
            P0 (np.ndarray, optional): Initial Kalman state estimation error covariance. Defaults to None.
            steady_state (bool, optional): If True, will use steady state Kalman gain, which is much faster. Defaults to True.
            return_state_cov (bool, optional): If true, will return state error covariances. Defaults to False.

        Returns:
            allXp (np.ndarray): one-step ahead predicted states (t|t-1)
            allYp (np.ndarray): one-step ahead predicted observations (t|t-1)
            allXf (np.ndarray): filtered states (t|t)
            allXs (np.ndarray): smoothed states (t|T)
            allPp (np.ndarray): error cov for one-step ahead predicted states (t|t-1)
            allPf (np.ndarray): error cov for filtered states (t|t)
            allPs (np.ndarray): error cov for smoothed states (t|T)
        """
        # First run the Kalman forward pass and get the covs
        allXp, allYp, allXf, allPp, allPf = self.kalman(
            Y, U, x0=x0, P0=P0, steady_state=steady_state, return_state_cov=True
        )
        allXpSS, allYpSS, allXfSS, allPpSS, allPfSS = self.kalman(
            Y, U, x0=x0, P0=P0, steady_state=True, return_state_cov=True
        )

        N = Y.shape[0]
        allXs = np.empty((N, self.state_dim))  # X(i|N)
        allXs[-1, :] = allXf[-1, :]  # X(N|N)

        if not steady_state:
            if return_state_cov:
                allPs = np.zeros((N, self.state_dim, self.state_dim))  # P(i|N)
                allPs[-1, :, :] = allPf[-1, :, :]  # P(N|N)
                allPps = np.zeros((N, self.state_dim, self.state_dim))  # P(i,i-1|N)
                # Based on G&H1996
                if N > 1:
                    Kf = self.Kf  # To do: replace this for Kf at last time step
                    allPps[-1, :, :] = (
                        (np.eye(self.state_dim) - Kf @ self.C)
                        @ self.A
                        @ allPf[-2, :, :]
                    )  # Pp(N,N-1|N)
        else:
            # Calc smoother gain at steady state(ish) [MAY BLOW UP, in which case you should NOT use steady state]
            P = self.Pp - self.Kf @ self.C @ self.Pp
            L = np.linalg.lstsq(self.Pp.T, (P * self.A.T).T, rcond=None)[
                0
            ].T  # (P * self.A.T) / self.Pp  ===> L(i) = P(i|i) * A.' * inv( P(i+1|i) )

            if return_state_cov:
                # Steady state Ps = P(i|N) is the solution to
                # Ps = P + L * ( Ps - Pp ) * L.'; % P(i|N)
                # Ps = L * Ps * L.' + ( P - L * Pp * L.' ); % P(i|N)
                # This is a discrete time lyapanov equation can be solved as
                Ps = linalg.solve_discrete_lyapunov(L, P - L @ self.Pp @ L.T)
                Pps = (
                    (self.eye(self.state_dim) - self.Kf @ self.C) @ self.A @ P
                )  # Pp(N,N-1|N)
        # Kalman smoother
        for i in reversed(range(N - 1)):  # From N-2 to 0
            if not steady_state:
                Li = np.linalg.lstsq(
                    allPp[i + 1, :, :].T, (allPf[i, :, :] @ self.A.T).T, rcond=None
                )[
                    0
                ].T  # (allPf[i,:,:] @ self.A.T) / allPp[i+1,:,:]  ===> # L(i) = P(i|i) * A.' * inv( P(i+1|i) )
                if return_state_cov:
                    allPs[i, :, :] = (
                        allPf[i, :, :]
                        + Li @ (allPs[i + 1, :, :] - allPp[i + 1, :, :]) @ Li.T
                    )  # P(i|N)
                    # Based on PRMLT
                    allPps[i, :, :] = allPs[i + 1, :, :] @ Li.T  # P(i,i-1|N)
            else:
                Li = L
            allXs[i, :] = allXf[i, :] + Li @ (
                allXs[i + 1, :] - allXp[i + 1, :]
            )  # X(i|N)

        if not return_state_cov:
            return allXp, allYp, allXf, allXs
        else:
            return allXp, allYp, allXf, allXs, allPp, allPf, allPs

    def forwardBackwardSmoother(
        self, Y, U=None, x0=None, P0=None, steady_state=True, return_state_cov=False
    ):
        """Runs the Forward-Backward smoother (Rauch-Tung-Striebel smoother).

        Args:
            Y (np.ndarray): Observation time series.
            U (np.ndarray, optional): Input time series. Defaults to None.
            x0 (np.ndarray, optional): Initial state. Defaults to None.
            P0 (np.ndarray, optional): Initial state covariance. Defaults to None.
            steady_state (bool, optional): If True, uses steady state assumptions. Defaults to True.
            return_state_cov (bool, optional): If True, returns state covariances. Defaults to False.

        Returns:
            allXp (np.ndarray): Predicted states.
            allYp (np.ndarray): Predicted observations.
            allX (np.ndarray): Filtered states.
            allXs (np.ndarray): Smoothed states.
            [allPp, allPf, allPs] (np.ndarray): Covariances (if return_state_cov is True).
        """
        # First run the Kalman forward pass and get the covs
        allXp, allYp, allX, allPp, allPf = self.kalman(
            Y, U, x0=x0, P0=P0, steady_state=steady_state, return_state_cov=True
        )

        N = Y.shape[0]

        # Based on Kitagawa 2023, using backward information filter
        allXpB = np.nan * np.ones_like(allXp)  # Xb(i | i+1)
        allXB = np.nan * np.ones_like(allXp)  # Xb(i | i)
        allXs = np.nan * np.ones((N, self.state_dim))  # X(i|N)
        if return_state_cov:
            allPpB = np.nan * np.ones_like(allPp)  # Pb(i | i+1)
            allPfB = np.nan * np.ones_like(allPp)  # Pb(i | i)
            allPs = np.zeros((N, self.state_dim, self.state_dim))  # P(i|N)
        invA = np.linalg.pinv(self.A)
        invQ = np.linalg.pinv(self.Q)
        invR = np.linalg.pinv(self.R)
        for i in reversed(range(N)):  # From N-1 to 0
            Xpi = allXp[i, :][:, np.newaxis]  # Xp(i|i-1)
            Xfi = allX[i, :][:, np.newaxis]  # Xf(i|i)
            Ppi = allPp[i, ...]  # Pf(i|i-1)
            Pfi = allPf[i, ...]  # Pf(i|i)

            if i == N - 1:
                XpBi = np.zeros((self.state_dim, 1))  # Xb(i|i+1)
                PpBi = np.zeros((self.state_dim, self.state_dim))  # Pb(i|i+1)

            Yi = Y[i, :][:, np.newaxis]
            # Backwards filter's update [Kitagawa 2023, Backward information filter]
            XfBi = XpBi + self.C.T @ invR @ Yi  # Xb(i|i) Backward update equation
            PfBi = (
                PpBi + self.C.T @ invR @ self.C
            )  # Pb(i|i) Cov for backward upated state

            allXpB[i, :] = XpBi.T
            allXB[i, :] = XfBi.T
            if return_state_cov:
                allPpB[i, ...] = PpBi
                allPfB[i, ...] = PfBi

            # Getting smoothed results [Kitagawa 2023, two-filter formula for smoothing 2]
            invPfi = np.linalg.pinv(Pfi)
            Psi = np.linalg.pinv(invPfi + PpBi)  # Ps(i|N) Cov of smoothed estimate
            Xsi = Psi @ (invPfi @ Xfi + XpBi)  # Xs(i|N)

            allXs[i, :] = Xsi.T
            if return_state_cov:
                allPs[i, ...] = Psi

            # Backwards filter's prediction [Kitagawa 2023, Backward information predictor]
            Lk = -self.A.T @ PfBi @ np.linalg.pinv(invQ + PfBi)
            XpBi = (self.A.T + Lk) @ XfBi
            PpBi = (self.A.T + Lk) @ PfBi @ self.A  # minus something is missing???

        if not return_state_cov:
            return allXp, allYp, allX, allXs
        else:
            return allXp, allYp, allX, allXs, allPp, allPf, allPs

    def propagateStates(self, allXp, step_ahead=1):
        """Propagates the system states forward in time.

        Args:
            allXp (np.ndarray): Current state estimates.
            step_ahead (int, optional): Number of steps to propagate. Defaults to 1.

        Returns:
            allXp (np.ndarray): Propagated states.
        """
        for step in range(step_ahead - 1):
            if (
                hasattr(self, "multi_step_with_A_KC") and self.multi_step_with_A_KC
            ):  # If true, forward predictions will be done with A-KC rather than the correct A (but will be useful for comparing with predictor form models)
                if (
                    hasattr(self, "useA_KC_plus_KC_in_KF")
                    and self.useA_KC_plus_KC_in_KF
                ):  # TEEMP, just for unit tests of DPADModel
                    allXp = (self.A_KC @ allXp.T).T
                else:
                    allXp = ((self.A - self.K @ self.C) @ allXp.T).T
            else:
                allXp = (self.A @ allXp.T).T
        return allXp

    def predict(
        self,
        Y,
        U=None,
        useXFilt=None,
        useXSmooth=None,
        return_state_cov=False,
        **kwargs,
    ):
        """Runs Kalman/PPF and returns state and observation predictions

        Args:
            Y (np.ndarray): observation time series (time first)
            U (np.ndarray, optional): input time series (time first). Defaults to None.
            useXFilt (bool, optional): If True, will use filtered states (t|t) for prediction of observations. Defaults to None.
            useXSmooth (bool, optional): If True, will use smoothed states (t|T) for prediction of observations. Defaults to None.

        Returns:
            allZp (np.ndarray): predicted z observations via Cz observation matrix (if it exists).
                    If there are n steps ahead specified in the model (self.steps_ahead), then the first n
                    outputs will be the predicted z observations with various steps ahead in order of self.steps_ahead.
            allYp (np.ndarray): predicted y observations via C observation matrix (if it exists)
                    If there are n steps ahead specified in the model (self.steps_ahead), then the second n
                    outputs will be the predicted y observations with various steps ahead in order of self.steps_ahead.
            allXp (np.ndarray): predicted states
                    If there are n steps ahead specified in the model (self.steps_ahead), then the third n
                    outputs will be the states with various steps ahead in order of self.steps_ahead.
        """
        if isinstance(Y, (list, tuple)):  # If segments of data are provided as a list
            for trialInd, trialY in enumerate(Y):
                trialOuts = self.predict(
                    trialY,
                    U=U if U is None else U[trialInd],
                    useXFilt=useXFilt,
                    useXSmooth=useXSmooth,
                    **kwargs,
                )
                if trialInd == 0:
                    outs = [[o] for oi, o in enumerate(trialOuts)]
                else:
                    outs = [outs[oi] + [o] for oi, o in enumerate(trialOuts)]
            return tuple(outs)
        # If only one data segment is provided
        if useXSmooth == True or (
            useXSmooth is None
            and hasattr(self, "predictWithXSmooth")
            and self.predictWithXSmooth
        ):
            kf_outputs = self.kalmanSmoother(
                Y, U=U, return_state_cov=return_state_cov, **kwargs
            )
            allXp, allYp, allXf, allXs = kf_outputs[:4]
            if np.any(np.isnan(allXs)) and "steady_state" not in kwargs:
                logger.warning(
                    f"Approximate steady state smoother blew up, using regular smoother"
                )
                kf_outputs = self.kalmanSmoother(
                    Y,
                    U=U,
                    steady_state=False,
                    return_state_cov=return_state_cov,
                    **kwargs,
                )
                allXp, allYp, allXf, allXs = kf_outputs[:4]
            if return_state_cov:
                cov_outputs = kf_outputs[4:]
            allXp = allXs
        else:
            kf_outputs = self.kalman(
                Y, U=U, return_state_cov=return_state_cov, **kwargs
            )
            allXp, allYp, allXf = kf_outputs[:3]
            if return_state_cov:
                cov_outputs = kf_outputs[3:]
            if useXFilt == True or (
                useXFilt is None
                and hasattr(self, "predictWithXFilt")
                and self.predictWithXFilt
            ):
                allXp = allXf

        allZpSteps, allYpSteps, allXpSteps = [], [], []
        if hasattr(self, "steps_ahead") and self.steps_ahead is not None:
            steps_ahead = self.steps_ahead
        else:
            steps_ahead = [1]
        for saInd, step_ahead in enumerate(steps_ahead):
            allXpThis = np.array(allXp)
            allXpThis = self.propagateStates(allXpThis, step_ahead)

            allZp = None
            if (hasattr(self, "Cz") and self.Cz is not None) or (
                hasattr(self, "Dz") and self.Dz is not None
            ):
                allZp = self.generateObservationFromStates(
                    allXpThis,
                    u=U,
                    param_names=["Cz", "Dz"] if step_ahead == 1 else ["Cz"],
                    prep_model_param="ZPrepModel",
                    mapping_param="cMap",
                )

            if (
                hasattr(self, "Dyz") and self.Dyz is not None and step_ahead == 1
            ):  # Additive direct regression from Y to Z
                directRegression = (self.Dyz @ Y.T).T
                if allZp is None:
                    allZp = directRegression
                else:
                    allZp += directRegression

            allYp = self.generateObservationFromStates(
                allXpThis,
                u=U,
                param_names=["C", "D"] if step_ahead == 1 else ["C"],
                prep_model_param="YPrepModel",
                mapping_param="cMapY",
            )

            allZpSteps.append(allZp)
            allYpSteps.append(allYp)
            allXpSteps.append(allXpThis)

        outputs = tuple(allZpSteps) + tuple(allYpSteps) + tuple(allXpSteps)
        if return_state_cov:
            outputs = outputs + cov_outputs
        return outputs

    def applySimTransform(self, E):
        """Applies a similarity transformation to the model parameters.

        Args:
            E (np.ndarray): The transformation matrix.
        """
        EInv = np.linalg.inv(E)

        ALikeFields = {"A"}
        for f in ALikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if (
                    val is not None
                    and val.shape[0] == E.shape[1]
                    and val.shape[0] == val.shape[1]
                ):
                    setattr(self, f, E @ val @ EInv)  # newA = E * A * EInv

        BLikeFields = {"B", "S", "G", "K", "Kf", "Kv"}
        for f in BLikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if val is not None and val.shape[0] == E.shape[1]:
                    setattr(self, f, E @ val)  # newB = E * B

        CLikeFields = {"C", "Cz"}
        for f in CLikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if val is not None and val.shape[1] == EInv.shape[0]:
                    setattr(self, f, val @ EInv)  # newC = C * EInv

        QLikeFields = {"Q", "P", "Pp", "P2", "XCov"}
        for f in QLikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if (
                    val is not None
                    and val.shape[0] == E.shape[1]
                    and val.shape[0] == val.shape[1]
                ):
                    setattr(self, f, E @ val @ E.T)  # newA = E * A * E'

        self.update_secondary_params()

    def makeSimilarTo(self, s2, N=10000):
        """Transforms this model to be similar (in terms of state basis) to another model s2.

        Args:
            s2 (LSSM): The target LSSM model to match.
            N (int, optional): Number of samples for generation if needed. Defaults to 10000.

        Returns:
            E (np.ndarray): The transformation matrix used.
        """
        Y, X = s2.generateRealization(N)
        xPredTrg = s2.kalman(Y)[0]
        xPredSrc = self.kalman(Y)[0]

        E = np.transpose(np.linalg.pinv(xPredSrc) @ xPredTrg)
        self.applySimTransform(E)
        return E

    def makeCanonical(self):
        """Transforms the model into a canonical form (e.g., using Schur decomposition).

        Returns:
            E (np.ndarray): The transformation matrix used.
        """
        J, EInv = linalg.schur(self.A, output="real")
        E = np.linalg.inv(EInv)
        self.applySimTransform(E)
        return E

    def makeA_KCBlockDiagonal(self, ignore_error=False):
        """Block-diagonalizes A and K*C based on provided blocks.

        Args:
            ignore_error (bool, optional): If True, ignores errors during diagonalization. Defaults to False.

        Returns:
            E (np.ndarray): The transformation matrix used.
        """
        n1 = (
            len(self.zDims)
            if hasattr(self, "zDims") and isinstance(self.zDims, (list, np.ndarray))
            else 0
        )
        E = makeMatrixBlockDiagonal(self.A_KC, n1, "A_KC", ignore_error=ignore_error)
        self.applySimTransform(E)
        return E

    def makeABlockDiagonal(self, ignore_error=False):
        """Block-diagonalizes the A matrix based on provided blocks.

        Args:
            ignore_error (bool, optional): If True, ignores errors during diagonalization. Defaults to False.

        Returns:
            E (np.ndarray): The transformation matrix used.
        """
        n1 = (
            len(self.zDims)
            if hasattr(self, "zDims") and isinstance(self.zDims, (list, np.ndarray))
            else 0
        )
        E = makeMatrixBlockDiagonal(self.A, n1, "A", ignore_error=ignore_error)
        self.applySimTransform(E)
        return E
