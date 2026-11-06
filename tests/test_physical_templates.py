import numpy as np
import torch

from DESIFlow.templates import physical_templates as pt


def _model(K=3):
    m = pt.Model.__new__(pt.Model)
    m.dev = torch.device("cpu"); m.K = K
    m.lam = torch.tensor(pt.LAM_OBS.astype(np.float32))
    R = np.zeros((11, pt.L), np.float32); R[5] = 1.0                                              # identity resolution
    m.Rm = torch.tensor(R); m.s = torch.ones(pt.L)
    return m


def test_camera_weights_partition_of_unity():
    W = pt.camera_weights(); s = W["b"] + W["r"] + W["z"]
    assert np.allclose(s, 1.0) and min(w.min() for w in W.values()) >= 0


def test_design_interpolates_at_rest_wavelength():
    m = _model(); x = pt.X0 + pt.DLOG * np.arange(pt.NR); lam_rest = 10 ** x
    phi = torch.tensor(np.stack([np.sin(lam_rest / 300.0), np.ones(pt.NR), lam_rest / 1e4]).astype(np.float32))
    z = torch.tensor([0.3, 0.7]); A = m.design(phi, z)
    for i, zz in enumerate([0.3, 0.7]):
        want = np.sin(pt.LAM_OBS / (1 + zz) / 300.0)
        assert np.abs(A[i, 0].numpy() - want).max() < 2e-3


def test_joint_solve_recovers_physical_and_offset():
    m = _model(); x = pt.X0 + pt.DLOG * np.arange(pt.NR); lam_rest = 10 ** x
    phi = torch.tensor(np.stack([np.sin(lam_rest / 300.0) + 2, lam_rest / 1e4, np.cos(lam_rest / 90.0)]).astype(np.float32))
    z = torch.tensor([0.4]); A = m.design(phi, z)
    off = torch.tensor(pt.camera_weights()["z"].astype(np.float32))[None]
    g = (1.5 * A[0, 0] - 0.7 * A[0, 1] + 0.3 * A[0, 2]).unsqueeze(0) + 0.2 * off
    X = torch.cat([A, off[:, None, :]], 1); om = torch.ones(1, pt.L)
    th = pt.Model.solve(X, g, om, torch.full((4,), 1e-9))
    assert np.allclose(th[0].numpy(), [1.5, -0.7, 0.3, 0.2], atol=2e-3)
