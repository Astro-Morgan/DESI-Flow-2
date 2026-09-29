"""Checks for the Eidos Perceiver (src/DESIFlow/eidos/perceiver.py). Run directly or with pytest."""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.perceiver import Perceiver, CrossAttention
from DESIFlow.eidos.cnn import CNN
from DESIFlow.preprocessing.preprocessing import Preprocessor

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
PER = Perceiver().to(DEV)
L = 813
TV = torch.linspace(0, 3e5, L, device=DEV)


def test_shapes_and_scale_token_first():
    tok, s = torch.randn(4, L, 256, device=DEV), torch.tensor([0.5, 1., 10., 40.], device=DEV)
    out = PER(tok, TV, s)
    exact = (out[:, 0, 0] - torch.log10(s)).abs().max().item()
    assert out.shape == (4, 65, 256) and exact < 1e-6 and torch.isfinite(out).all()
    n_ca = sum(isinstance(l, CrossAttention) for l in PER.layers)
    return f"out {tuple(out.shape)}; slot 0 = scale token (log10 s exact to {exact:.0e}); {n_ca} CA + {len(PER.layers) - n_ca} SA layers"


def test_relative_position_equivariance():
    """Shift every token position AND every latent base position by the same delta: outputs unchanged."""
    tok, s = torch.randn(2, L, 256, device=DEV), torch.ones(2, device=DEV)
    with torch.no_grad():
        a = PER(tok, TV, s)
        for l in PER.layers:
            if isinstance(l, CrossAttention):
                l.base += 12345.
        b = PER(tok, TV + 12345., s)
        for l in PER.layers:
            if isinstance(l, CrossAttention):
                l.base -= 12345.
    rel = ((a - b).abs().max() / a.abs().max()).item()
    assert rel < 1e-3, rel
    return f"joint shift of 12,345 km/s changes outputs by rel {rel:.1e} (float32 phase round-off)"


def test_positions_matter_and_are_learnable():
    tok, s = torch.randn(2, L, 256, device=DEV), torch.ones(2, device=DEV)
    a = PER(tok, TV, s)
    b = PER(tok.flip(1), TV, s)                                   # same tokens, reversed positions
    PER.zero_grad(); a[:, 1:].pow(2).mean().backward()
    ca = [l for l in PER.layers if isinstance(l, CrossAttention)]
    g_base = all(l.base.grad is not None and l.base.grad.abs().sum() > 0 for l in ca)
    g_delta = all(l.delta.weight.grad is not None and l.delta.weight.grad.abs().sum() > 0 for l in ca)
    diff = ((a - b)[:, 1:].abs().max() / a[:, 1:].abs().max()).item()
    assert diff > 1e-2 and g_base and g_delta
    return f"reversing token order changes latents by rel {diff:.2f}; gradients reach base positions and steering heads in all CA layers"


def test_end_to_end_real():
    from astropy.table import Table
    t = Table.read(ROOT / "data" / "data.fits")
    x = torch.stack([torch.tensor(np.asarray(t[c], np.float32), device=DEV) for c in ("FLUX", "IVAR", "MASK")], 1)[:16]
    pre, cnn = Preprocessor().to(DEV), CNN().to(DEV)
    tv = Perceiver.token_velocity(cnn.token_wave(pre.new_wave))
    torch.cuda.synchronize() if DEV.type == "cuda" else None
    t0 = time.time()
    tok, s = cnn(*pre(x)); out = PER(tok, tv, s); out[:, 1:].pow(2).mean().backward()
    torch.cuda.synchronize() if DEV.type == "cuda" else None
    n = sum(p.numel() for p in PER.parameters())
    assert out.shape == (16, 65, 256) and torch.isfinite(out).all()
    return f"preprocess->CNN->Perceiver on 16 DESI spectra: {tuple(out.shape)}, fwd+bwd {1e3 * (time.time() - t0):.0f} ms; Perceiver {n / 1e6:.2f}M params; token v span {tv[-1]:.0f} km/s"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
