"""
Checks for the Eidos wrapper (src/DESIFlow/eidos/eidos.py): it must be exactly the hand-assembled pipeline.
Run directly (python tests/test_eidos.py) or with pytest.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.eidos import Eidos
from DESIFlow.eidos.perceiver import Perceiver
from DESIFlow.eidos.reader import velocity

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
NET = Eidos().to(DEV).eval()
PRE = NET.preprocessor
NATIVE = torch.linspace(PRE.wave_min, PRE.wave_max, PRE.spec_len, dtype=torch.float64, device=DEV)


def real_batch(n):
    from astropy.table import Table
    t = Table.read(ROOT / "data" / "data.fits")
    return torch.stack([torch.tensor(np.asarray(t[c], np.float32), device=DEV) for c in ("FLUX", "IVAR", "MASK")], 1)[:n]


def test_encode_decode_shapes_and_scale_slot():
    x = real_batch(6)
    with torch.no_grad():
        z = NET.encode(x)
        y = NET(x, NATIVE)
        y_ps = NET.decode(z, NATIVE.expand(6, -1))                                               # per-sample (B, Q)
        s = NET.cnn(*NET.preprocess(x))[1]
    assert z.shape == (6, 65, 256) and y.shape == (6, PRE.spec_len) and torch.isfinite(y).all()
    assert torch.equal(y, NET.decode(z, NATIVE)) and (y - y_ps).abs().max() < 1e-4
    err = (10 ** z[:, 0, 0] - s).abs().max().item() / s.max().item()
    assert err < 1e-6, err
    return f"encode -> {tuple(z.shape)}; forward at the 7781 native pixels -> {tuple(y.shape)}; s = 10**latents[:, 0, 0] recovers the CNN scale to rel {err:.0e}"


def test_equals_hand_assembled_pipeline():
    x = real_batch(4)
    q = NATIVE[::7]
    with torch.no_grad():
        y = NET(x, q)
        xr, xs = NET.preprocessor(x)
        tok, s = NET.cnn(xr, xs)
        tv = Perceiver.token_velocity(NET.cnn.token_wave(PRE.new_wave))
        z = NET.perceiver(tok, tv, s)
        ref = NET.decoder(z[:, 1:], velocity(q, PRE.new_wave[0]))
    d = (y - ref).abs().max().item()
    assert d == 0.0, d
    return f"wrapper output identical to preprocess -> CNN -> Perceiver -> Decoder assembled by hand ({len(q)} queries, max diff {d:.0e})"


def test_gradients_reach_every_stage():
    NET.train()
    x = real_batch(4)
    NET.zero_grad()
    NET(x, NATIVE[::5]).pow(2).mean().backward()
    parts = {"cnn": NET.cnn, "perceiver": NET.perceiver, "decoder": NET.decoder}
    rows = []
    for name, m in parts.items():
        ps = list(m.parameters())
        n_ok = sum(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in ps)
        assert n_ok == len(ps), (name, n_ok, len(ps))
        rows.append(f"{name} {n_ok}/{len(ps)}")
    NET.eval()
    n = {k: sum(p.numel() for p in m.parameters()) / 1e6 for k, m in parts.items()}
    return "parameter tensors with gradient: " + ", ".join(rows) + f"; params (M): " + ", ".join(f"{k} {v:.2f}" for k, v in n.items())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
