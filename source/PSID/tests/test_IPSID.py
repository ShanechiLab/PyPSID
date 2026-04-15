"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md

Tests IPSID-specific functionality.
"""

import os
import sys
import unittest
import warnings

import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def make_input_driven_data(num_steps=300, include_z=False, seed=0):
    from PSID.LSSM import LSSM

    rng = np.random.default_rng(seed)
    u = rng.standard_normal((num_steps, 1))
    params = {
        "A": np.array([[0.85]]),
        "B": np.array([[0.45]]),
        "C": np.array([[1.0]]),
        "D": np.array([[0.3]]),
        "Q": np.array([[0.02]]),
        "R": np.array([[0.03]]),
        "S": np.zeros((1, 1)),
    }
    model = LSSM(params=params)
    y, x = model.generateRealization(num_steps, u=u)
    z = None
    if include_z:
        z = 1.2 * x + 0.25 * u + 0.02 * rng.standard_normal((num_steps, 1))
    return y, z, u


def collect_warning_messages(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def solve_expected_bd(A, C, Yii, Xk_Plus1, Xk, i, nu, Uf, fit_Dy):
    from PSID.IPSID import computeObsFromAC

    Oy, Oy_Minus = computeObsFromAC(A, C, i)
    PP = np.concatenate((Xk_Plus1 - A @ Xk, Yii - C @ Xk))

    L1 = A @ np.linalg.pinv(Oy)
    L2 = C @ np.linalg.pinv(Oy)

    nx = A.shape[0]
    ny = C.shape[0]
    ZM = np.concatenate((np.zeros((nx, ny)), np.linalg.pinv(Oy_Minus)), axis=1)

    lhs = np.zeros((PP.size, (nx + ny) * nu))
    r_mul = linalg.block_diag(np.eye(ny), Oy_Minus)
    for ii in range(i):
        nn = np.zeros((nx + ny, i * ny))
        nn[:nx, : ((i - ii) * ny)] = ZM[:, (ii * ny) :] - L1[:, (ii * ny) :]
        nn[nx : (nx + ny), : ((i - ii) * ny)] = -L2[:, (ii * ny) :]
        if ii == 0:
            nn[nx : (nx + ny), :ny] = nn[nx : (nx + ny), :ny] + np.eye(ny)
        lhs = lhs + np.kron(Uf[(ii * nu) : (ii * nu + nu), :].T, nn @ r_mul)

    rhs = PP.flatten(order="F")
    if fit_Dy:
        db_vec = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        db = np.reshape(db_vec, [nx + ny, nu], order="F")
        return db[ny:, :], db[:ny, :]

    keep_cols = []
    block_size = nx + ny
    for input_idx in range(nu):
        block_start = input_idx * block_size
        keep_cols.extend(range(block_start + ny, block_start + block_size))
    b_vec = np.linalg.lstsq(lhs[:, keep_cols], rhs, rcond=None)[0]
    return np.reshape(b_vec, [nx, nu], order="F"), np.zeros((ny, nu))


class TestIPSID(unittest.TestCase):
    def test_computeBD_can_constrain_Dy_to_zero(self):
        from PSID.IPSID import computeBD

        rng = np.random.default_rng(0)
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        C = np.array([[1.0, 0.2], [0.3, 0.7]])
        i = 3
        nu = 2
        num_cols = 25
        Xk = rng.standard_normal((A.shape[0], num_cols))
        Xk_Plus1 = rng.standard_normal((A.shape[0], num_cols))
        Yii = rng.standard_normal((C.shape[0], num_cols))
        Uf = rng.standard_normal((i * nu, num_cols))

        B, D = computeBD(A, C, Yii, Xk_Plus1, Xk, i, nu, Uf)
        expected_B, expected_D = solve_expected_bd(
            A, C, Yii, Xk_Plus1, Xk, i, nu, Uf, fit_Dy=True
        )
        np.testing.assert_allclose(B, expected_B)
        np.testing.assert_allclose(D, expected_D)

        B_zero, D_zero = computeBD(
            A, C, Yii, Xk_Plus1, Xk, i, nu, Uf, fit_Dy=False
        )
        expected_B_zero, expected_D_zero = solve_expected_bd(
            A, C, Yii, Xk_Plus1, Xk, i, nu, Uf, fit_Dy=False
        )
        np.testing.assert_allclose(B_zero, expected_B_zero)
        np.testing.assert_allclose(D_zero, expected_D_zero)
        np.testing.assert_allclose(D_zero, np.zeros_like(D_zero))

    def test_ipsid_fit_Dy_false_sets_D_zero_for_input_model(self):
        from PSID.IPSID import IPSID

        y, _, u = make_input_driven_data(seed=1)
        id_sys, messages = collect_warning_messages(
            IPSID, y, Z=None, U=u, nx=1, n1=0, i=10, fit_Dy=False
        )
        self.assertEqual(id_sys.B.shape, (1, 1))
        np.testing.assert_allclose(id_sys.D, np.zeros_like(id_sys.D))
        self.assertFalse(any("fit_Dy=False has no effect" in msg for msg in messages))

    def test_ipsid_fit_Dy_false_warns_when_no_input_is_provided(self):
        from PSID.IPSID import IPSID

        y, _, _ = make_input_driven_data(seed=2)
        _, messages = collect_warning_messages(
            IPSID, y, Z=None, U=None, nx=1, n1=0, i=10, fit_Dy=False
        )
        self.assertTrue(
            any("fit_Dy=False has no effect" in msg for msg in messages)
        )

    def test_ipsid_fit_Dy_false_warns_that_Dz_may_still_be_learned(self):
        from PSID.IPSID import IPSID

        y, z, u = make_input_driven_data(include_z=True, seed=3)
        _, messages = collect_warning_messages(
            IPSID,
            y,
            Z=z,
            U=u,
            nx=1,
            n1=1,
            i=10,
            fit_Dy=False,
            remove_nonYrelated_fromX1=False,
        )
        self.assertTrue(any("Dz may still be learned" in msg for msg in messages))

    def test_ipsid_fit_Dy_false_does_not_warn_about_Dz_when_preprocessing_forces_zero(self):
        from PSID.IPSID import IPSID

        y, z, u = make_input_driven_data(include_z=True, seed=4)
        id_sys, messages = collect_warning_messages(
            IPSID,
            y,
            Z=z,
            U=u,
            nx=1,
            n1=1,
            i=10,
            fit_Dy=False,
            remove_nonYrelated_fromX1=True,
            n_pre=1,
            n3=0,
        )
        self.assertFalse(any("Dz may still be learned" in msg for msg in messages))
        self.assertFalse(any("additional step 2" in msg for msg in messages))
        np.testing.assert_allclose(id_sys.Dz, np.zeros_like(id_sys.Dz))

    def test_ipsid_fit_Dy_false_warns_when_additional_stage_can_introduce_Dz(self):
        from PSID.IPSID import IPSID

        y, z, u = make_input_driven_data(include_z=True, seed=5)
        _, messages = collect_warning_messages(
            IPSID,
            y,
            Z=z,
            U=u,
            nx=2,
            n1=1,
            i=10,
            fit_Dy=False,
            remove_nonYrelated_fromX1=True,
            n_pre=1,
            n3=1,
        )
        self.assertTrue(any("additional step 2" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()
