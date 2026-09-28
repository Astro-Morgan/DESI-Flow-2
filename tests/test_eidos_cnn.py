"""
Checks for the Eidos CNN front end (src/DESIFlow/eidos/eidos.py).
Run directly (python tests/test_eidos_cnn.py) or with pytest.
"""
import sys, time, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.eidos import CNN
from DESIFlow.preprocessing.preprocessing import Preprocessor

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
PRE = Preprocessor().to(DEV)
NET = CNN().to(DEV)


def real_batch(n=None):
    from astropy.table import Table
    t = Table.read(ROOT / "data" / "data.fits")
    x = torch.stack([torch.tensor(np.asarray(t[c], np.float32), device=DEV) for c in ("FLUX", "IVAR", "MASK")], 1)
    return x if n is None else x[:n]


def test_shapes_and_token_wavelengths():
    xr, xs = PRE(real_batch(8))
    tok, s = NET(xr, xs)
    tw = NET.token_wave(PRE.new_wave)
    n_tok = math.ceil(PRE.new_len / NET.downsample)
    assert tok.shape == (8, n_tok, 256) and s.shape == (8,) and len(tw) == n_tok and torch.isfinite(tok).all()
    dv = 299792.458 * math.log(float(tw[1] / tw[0]))
    return f"tokens {tuple(tok.shape)} (downsample x{NET.downsample}, {dv:.0f} km/s per token); scale {tuple(s.shape)}"


def test_branch_receptive_fields_match_starlet_footprints():
    """Convolutional reach only: GroupNorm(1) normalizes with statistics over the whole sequence, which makes every
    output depend (weakly) on every input, so the norms are bypassed for this measurement."""
    import copy
    rows = []
    for i, br0 in enumerate(NET.branches):
        br = copy.deepcopy(br0)
        br.net = torch.nn.Sequential(*[m for m in br.net if not isinstance(m, torch.nn.GroupNorm)])
        L = 4 * br.receptive_field + 1
        x = torch.randn(1, 7, L, device=DEV, requires_grad=True)
        br(x)[0, :, L // 2].sum().backward()
        nz = torch.nonzero(x.grad.abs().sum(1)[0] > 0).flatten()
        extent = int(nz.max() - nz.min() + 1)
        j = min(i + 1, NET.n_scales)
        footprint = 4 * (2 ** j - 1) + 1
        assert extent == br.receptive_field
        rows.append(f"{'s' + str(i + 1) if i < NET.n_scales else 'coarse'}:{extent}/{footprint}")
    return "receptive field / starlet footprint (px): " + " ".join(rows)


def test_magnitude_invariance():
    x = real_batch(4)
    xb = x.clone(); xb[:, 0] *= 0.013; xb[:, 1] /= 0.013 ** 2                    # same objects, 77x fainter
    with torch.no_grad():
        ta, sa = NET(*PRE(x)); tb, sb = NET(*PRE(xb))
        pa, _, _ = NET.smooth_path(PRE(x)[1]); pb, _, _ = NET.smooth_path(PRE(xb)[1])
    rel = ((ta - tb).abs().max() / ta.abs().max()).item()
    h2 = (0.5 * (pa - pb).pow(2).sum((1, 2))).max().item()                       # Hellinger^2 between the two sqrtP
    ratio = (sb / sa).cpu().numpy()
    # float32: sqrt amplifies ~1e-7 round-off in near-zero cells to ~1e-4 relative (float64 end to end: ~5e-8)
    assert rel < 2e-3 and h2 < 1e-8 and np.allclose(ratio, 0.013, rtol=1e-4), (rel, h2, ratio)
    return (f"77x fainter copies: sqrtP differ by Hellinger^2 {h2:.1e}; tokens by rel {rel:.1e} (float32 round-off; "
            f"float64 ~5e-8); scale ratio {ratio.round(5).tolist()}")


def test_gradients_reach_every_branch():
    xr, xs = PRE(real_batch(4))
    NET.zero_grad()
    tok, _ = NET(xr, xs)
    tok.pow(2).mean().backward()
    ok = [all(p.grad is not None and p.grad.abs().sum() > 0 for p in br.parameters()) for br in NET.branches]
    assert all(ok)
    return f"all {len(ok)} branches and the mixer receive gradient"


def test_real_forward_backward_cost():
    x = real_batch()
    xr, xs = PRE(x)
    out = {}
    for B in [16, 64]:
        idx = torch.arange(B, device=DEV) % xr.shape[0]
        torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        t0 = time.time()
        tok, _ = NET(xr[idx], xs[idx]); tok.pow(2).mean().backward()
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        out[B] = (1e3 * (time.time() - t0), torch.cuda.max_memory_allocated() / 1e9 if DEV.type == "cuda" else float("nan"))
    n_par = sum(p.numel() for p in NET.parameters())
    n_br = sum(p.numel() for p in NET.branches.parameters())
    return (f"{n_par / 1e6:.2f}M parameters ({n_br / 1e3:.1f}k in branches); forward+backward: "
            + "; ".join(f"B={b}: {t:.0f} ms, peak {m:.2f} GB" for b, (t, m) in out.items()))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
