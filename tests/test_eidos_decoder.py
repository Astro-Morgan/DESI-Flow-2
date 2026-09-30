"""
Checks for the Eidos reader and decoder (src/DESIFlow/eidos/reader.py, decoder.py).
Run directly (python tests/test_eidos_decoder.py) or with pytest.
"""
import sys, time, copy
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.decoder import Decoder
from DESIFlow.eidos.reader import Reader, ReadLayer, velocity
from DESIFlow.eidos.perceiver import Perceiver
from DESIFlow.eidos.cnn import CNN
from DESIFlow.preprocessing.preprocessing import Preprocessor

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
N_LAT, D_LAT = 64, 256
TV = torch.linspace(0, 3e5, 500, device=DEV)


def steered(model):
    """Give the zero-init steering heads nonzero weights so latent-announced positions depend on content."""
    for m in model.modules():
        if isinstance(m, ReadLayer) and m.positional:
            torch.nn.init.normal_(m.delta.weight, std=0.05)
    return model


def latents(b=4):
    return torch.randn(b, N_LAT, D_LAT, device=DEV)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_shapes_and_size_settings():
    z = latents()
    rows = []
    for name, kw in {"minimal (1 read, no MLP)": dict(n_reads=1, mlp_ratio=0), "default (2 reads, MLP x2)": {},
                     "deep (4 reads, MLP x4, 2 latent SA)": dict(n_reads=4, mlp_ratio=4, n_latent_sa=2)}.items():
        dec = Decoder(**kw).to(DEV)
        y = dec(z, TV)
        y2 = dec(z, TV.expand(4, -1))
        assert y.shape == (4, 500) and y2.shape == (4, 500) and torch.isfinite(y).all()
        rows.append(f"{name}: {n_params(dec) / 1e6:.2f}M")
    assert Decoder(d_out=3).to(DEV)(z, TV).shape == (4, 500, 3)
    try:
        Decoder().to(DEV)(torch.randn(2, 65, D_LAT, device=DEV), TV)
        raise AssertionError("65 latents (scale token still in slot 0) should be rejected")
    except ValueError:
        pass
    return "out (B, Q) for (Q,) and (B, Q) positions; d_out=3 -> (B, Q, 3); 65 latents rejected. Params: " + "; ".join(rows)


def test_queries_are_independent():
    """Querying a subset (any order, a different subset per sample) gives the same values as querying everything."""
    dec = steered(Decoder(n_latent_sa=1)).to(DEV).eval()
    z = latents(3)
    idx = torch.stack([torch.randperm(500, device=DEV)[:120] for _ in range(3)])                # (3, 120) per sample

    def compare(m, zz):
        with torch.no_grad():
            full, sub = m(zz, TV), m(zz, TV[idx])
            alone = torch.stack([m(zz[b:b + 1], TV[idx[b]])[0] for b in range(3)])
        return [((sub - r).abs().max() / full.abs().max()).item() for r in (torch.gather(full, 1, idx), alone)]

    d1, d2 = compare(dec, z)
    e1, e2 = compare(copy.deepcopy(dec).double(), z.double())
    # float32: batch size changes GEMM blocking, and the steering scale (1e4 km/s) amplifies round-off into the positions
    assert d1 < 1e-4 and d2 < 1e-3 and e1 < 1e-9 and e2 < 1e-9, (d1, d2, e1, e2)
    return (f"120 random queries per sample vs the same queries inside the full 500: rel {d1:.1e} (float32), {e1:.0e} (float64); "
            f"batched vs single-sample: rel {d2:.1e} (float32, round-off), {e2:.0e} (float64: exactly independent)")


def test_relative_position_equivariance():
    """Shift every query position AND every latent base position by the same delta: outputs unchanged; shifting only
    the queries changes them."""
    dec = steered(Decoder()).to(DEV).eval()
    z = latents(2)
    reads = [m for m in dec.modules() if isinstance(m, ReadLayer)]
    with torch.no_grad():
        a = dec(z, TV)
        only_q = dec(z, TV + 12345.)
        for m in reads:
            m.base += 12345.
        b = dec(z, TV + 12345.)
        for m in reads:
            m.base -= 12345.
    rel = ((a - b).abs().max() / a.abs().max()).item()
    moved = ((a - only_q).abs().max() / a.abs().max()).item()
    assert rel < 3e-3 and moved > 1e-2, (rel, moved)
    return f"joint shift of 12,345 km/s changes outputs by rel {rel:.1e} (float32 phase round-off); shifting only the queries changes them by rel {moved:.2f}"


def test_gradients_reach_everything():
    dec = steered(Decoder(n_latent_sa=1)).to(DEV)
    z = latents(2).requires_grad_()
    dec(z, TV).pow(2).mean().backward()
    missing = [n for n, p in dec.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
               or p.grad.abs().sum() == 0]
    reads = [m for m in dec.modules() if isinstance(m, ReadLayer)]
    assert not missing, missing
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert all(m.base.grad.abs().sum() > 0 and m.delta.weight.grad.abs().sum() > 0 for m in reads)
    return f"all {len(list(dec.parameters()))} parameter tensors (incl. base positions and steering heads of {len(reads)} reads) and the latents receive gradient"


def test_positionless_reader_for_plato():
    """Learned queries with no wavelength: no position parameters, positions ignored, content-only reading."""
    rd = Reader(d_query=96, n_heads=4, n_reads=2, positional=False).to(DEV)
    assert not any(hasattr(m, "base") for m in rd.modules())
    q = torch.randn(1, 5, 96, device=DEV).expand(3, -1, -1)                                      # 5 block queries
    z = latents(3)
    out = rd(q, z)
    assert out.shape == (3, 5, 96) and (out[:, 0] - out[:, 1]).abs().max() > 1e-3               # queries differ
    try:
        Reader(positional=True).to(DEV)(q[..., :1].expand(3, 5, 128), z)
        raise AssertionError("positional reader without positions should be rejected")
    except ValueError:
        pass
    return f"positional=False: no base/steering parameters, 5 learned queries -> {tuple(out.shape)}, distinct per query; positional reader without query_v is rejected"


def test_native_pixel_queries_real():
    """preprocess -> CNN -> Perceiver -> decoder queried at hidden NATIVE pixel wavelengths of real spectra."""
    from astropy.table import Table
    t = Table.read(ROOT / "data" / "data.fits")
    x = torch.stack([torch.tensor(np.asarray(t[c], np.float32), device=DEV) for c in ("FLUX", "IVAR", "MASK")], 1)[:16]
    pre, cnn, per, dec = Preprocessor().to(DEV), CNN().to(DEV), Perceiver().to(DEV), Decoder().to(DEV)
    native = torch.linspace(pre.wave_min, pre.wave_max, pre.spec_len, dtype=torch.float64, device=DEV)
    v = velocity(native, pre.new_wave[0])                                                        # (7781,)
    hidden = torch.stack([torch.randperm(pre.spec_len, device=DEV)[:3000] for _ in range(16)])   # (16, 3000)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    tok, s = cnn(*pre(x))
    out = per(tok, Perceiver.token_velocity(cnn.token_wave(pre.new_wave)), s)
    y = dec(out[:, 1:], v[hidden])
    y.pow(2).mean().backward()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    ms = 1e3 * (time.time() - t0)
    gpu = f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if DEV.type == "cuda" else ""
    assert y.shape == (16, 3000) and torch.isfinite(y).all()
    return (f"16 real spectra, 3000 hidden native px each (positions span {v[0]:.0f}..{v[-1]:.0f} km/s): out {tuple(y.shape)}, "
            f"whole model fwd+bwd {ms:.0f} ms{gpu}; decoder {n_params(dec) / 1e6:.2f}M params")


def test_can_render_a_real_spectrum():
    """Rendering capacity: free latents + decoder fitted to ONE bright real spectrum on its native pixels (chi2 per good
    pixel; ~1 = noise level). The line profiles must come out of the attention weights, with no patch head to help."""
    if DEV.type != "cuda":
        return "skipped (needs a GPU)"
    from astropy.table import Table
    t = Table.read(ROOT / "data" / "data.fits")
    x = torch.stack([torch.tensor(np.asarray(t[c], np.float32), device=DEV) for c in ("FLUX", "IVAR", "MASK")], 1)
    pre, cnn = Preprocessor().to(DEV), CNN().to(DEV)
    good_all = (x[:, 2] == 0) & (x[:, 1] > 0)
    snr = torch.stack([(x[i, 0] * x[i, 1].sqrt())[good_all[i]].median() for i in range(len(x))])
    x = x[snr.argmax():snr.argmax() + 1]
    native = torch.linspace(pre.wave_min, pre.wave_max, pre.spec_len, dtype=torch.float64, device=DEV)
    v = velocity(native, pre.new_wave[0])
    with torch.no_grad():
        s = cnn.smooth_path(pre(x)[1])[2]
    good = ((x[:, 2] == 0) & (x[:, 1] > 0)).float()
    f, w = x[:, 0] / s, x[:, 1] * s ** 2 * good
    z, dec, steps = torch.nn.Parameter(torch.randn(1, N_LAT, D_LAT, device=DEV)), Decoder().to(DEV), 2000
    opt = torch.optim.Adam([{"params": [z], "lr": 2e-2}, {"params": dec.parameters(), "lr": 3e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    chi = []
    for _ in range(steps):
        loss = (w * (dec(z, v) - f) ** 2).sum() / good.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        chi.append(loss.item())
    assert chi[-1] < 3.0, chi[-1]
    return f"bright spectrum (median S/N {snr.max():.0f}), {steps} steps: chi2/px {chi[0]:.0f} -> {chi[-1]:.2f} (~1 = noise level)"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
