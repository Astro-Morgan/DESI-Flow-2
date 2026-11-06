import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from DESIFlow.templates.deeprock_lite import evaluate, order_orthonormal, span_mask, train_templates, whiten, wls_project


def toy(n=3000, L=200, K=3, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, L)
    T = np.stack([np.ones(L), np.sin(6 * x), np.exp(-0.5 * ((x - 0.7) / 0.03) ** 2) * 3])[:K]
    C = rng.normal(0, 1, (n, K)) * np.array([0.5, 1.0, 2.0])[:K]
    w = np.full((n, L), 4.0, np.float32)
    f = (C @ T + rng.normal(0, 0.5, (n, L))).astype(np.float32)
    return f, w, T


def test_wls_recovers_coefficients_and_span_mask_coverage():
    f, w, T = toy()
    g, om, s = whiten(f, w)
    U = torch.tensor(T * s[None], dtype=torch.float32)
    c = wls_project(U, torch.tensor(g[:200]), torch.tensor(om[:200]), ridge=0.0)
    r = (torch.tensor(g[:200]) - c @ U).numpy()
    assert abs(r.std() - 0.5 * s.mean()) < 0.05 * s.mean()                                       # leftover is the noise
    m = span_mask(500, 200, n_spans=12, len_range=(3, 12), gen=torch.Generator().manual_seed(0))
    assert 0.1 < m.float().mean() < 0.7
    return f"WLS leaves only the noise; span mask hides {100 * m.float().mean():.0f}% of the pixels"


def test_training_learns_the_signal_subspace_and_beats_the_zero_model_on_held_out_rows():
    f, w, T = toy()
    g, om, s = whiten(f, w)
    G, O = torch.tensor(g), torch.tensor(om)
    MK = dict(n_spans=12, len_range=(3, 12))
    U, hist = train_templates(G[:2500], O[:2500], K=3, steps=300, batch=256, seed=0, mask_kw=MK)
    hid, base, vis = evaluate(U, G[2500:], O[2500:], mask_kw=MK)
    assert hid.mean() < 0.5 * base.mean()
    B, _ = order_orthonormal(U, G[:2500], O[:2500])
    Tn = torch.tensor(T * s[None], dtype=torch.float32)
    P = B.T @ torch.linalg.solve(B @ B.T, B)                                                  # projector on the learned span
    resid = (Tn - Tn @ P).norm() / Tn.norm()
    assert resid < 0.1
    return f"held-out hidden chi2 {hid.mean():.2f} vs zero model {base.mean():.2f}; the true templates lie in the learned span to {100 * float(resid):.1f}%"


def test_merge_camera_terms_follows_the_ivar_weighted_overlap_rule():
    from DESIFlow.templates.camera_templates import CAMS, merge_camera_terms
    terms = {c: np.full(b - a, v) for (c, (a, b)), v in zip(CAMS.items(), (1.0, 2.0, 3.0))}
    ivars = {c: np.full(b - a, w) for (c, (a, b)), w in zip(CAMS.items(), (1.0, 3.0, 1.0))}
    m = merge_camera_terms(terms, ivars)
    assert m[100] == 1.0 and m[3500] == 2.0 and m[7000] == 3.0                                   # outside the overlaps each camera is copied
    assert abs(m[2720] - (1 * 1.0 + 3 * 2.0) / 4) < 1e-12                                          # b-r overlap: 5760-5800 A is brz index 2700-2750
    assert abs(m[5000] - (3 * 2.0 + 1 * 3.0) / 4) < 1e-12                                          # r-z overlap: 7520-7620 A is brz index 4900-5025
    return "overlap pixels are the inverse-variance weighted mean of the two cameras, the rest a copy"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
