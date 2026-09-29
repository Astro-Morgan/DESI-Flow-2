"""
Checks for the hardcoded starlet paths (src/DESIFlow/eidos/starlet.py).
Run directly (python tests/test_eidos_starlet.py) or with pytest.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.starlet import WeightedStarlet, RawStarletPath, SmoothStarletPath, normalize_raw
from DESIFlow.eidos.embeddings import ScaleToken
from DESIFlow.preprocessing.preprocessing import Preprocessor

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
J, N = 9, 13000
RNG = np.random.default_rng(0)
T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEV)


def plain_starlet(f, J):
    """Reference: unweighted a-trous B3 starlet with reflect padding."""
    h = torch.tensor([1, 4, 6, 4, 1], dtype=f.dtype, device=f.device).view(1, 1, 5) / 16
    c, out = f.unsqueeze(1), []
    for j in range(J):
        d = 2 ** j
        cn = F.conv1d(F.pad(c, (2 * d, 2 * d), mode="reflect"), h, dilation=d)
        out.append(c - cn); c = cn
    return torch.cat(out + [c], 1)


def spectrum(n=4):
    x = np.arange(N)
    cont = 1 + 0.5 * np.exp(-x / 4000) + 0.1 * np.sin(x / 700)
    lines = sum(a * np.exp(-0.5 * ((x - c) / s) ** 2) for c, s, a in [(3000, 4, 3), (3100, 4, 1), (7000, 8, 2), (9000, 60, -0.4)])
    return cont + lines + 0.05 * RNG.standard_normal((n, N))


def test_reconstruction_exact_on_good_pixels():
    f = spectrum(); ivar = RNG.uniform(0.5, 5, (4, N)); good = np.ones((4, N)); good[:, 5000:6600] = 0; good[:, RNG.integers(0, N, 200)] = 0
    ch = RawStarletPath(J).to(DEV)(torch.stack([T(f * good), T(ivar * good), T(good)], 1))
    err = (ch.sum(1) - T(f * good))[T(good) > 0].abs().max().item()
    assert err < 1e-4, err
    return f"sum of {J + 1} channels reproduces raw flux on good pixels to {err:.1e}"


def test_uniform_weights_equal_plain_starlet():
    f = T(spectrum())
    ch, _ = WeightedStarlet(J).to(DEV)(f, torch.ones_like(f))
    err = (ch - plain_starlet(f, J)).abs().max().item()
    assert err < 1e-4, err
    return f"uniform weights, no gaps: weighted == plain starlet to {err:.1e}"


def reference_hellinger2(a, b, J):
    """Independent reference for the metric (as in the metric study): plain starlet -> L1 composition over
    (channel, pixel, sign) cells -> H^2 = 1 - sum sqrt(P_a P_b)."""
    def cells(f):
        ch = plain_starlet(torch.as_tensor(f, dtype=torch.float64), J)
        p = torch.cat([ch.clamp_min(0), (-ch).clamp_min(0)], 1)
        return p / p.sum((1, 2), keepdim=True)
    return (1 - (cells(a) * cells(b)).sqrt().sum((1, 2))).numpy()


def test_sqrtP_reproduces_metric_hellinger():
    a, b = spectrum(3), spectrum(3)
    path = SmoothStarletPath(J).to(DEV)
    sa, _, _ = path(torch.stack([T(a), torch.ones(3, N, device=DEV)], 1))
    sb, _, _ = path(torch.stack([T(b), torch.ones(3, N, device=DEV)], 1))
    h2_path = 0.5 * ((sa - sb) ** 2).sum((1, 2)).cpu().numpy()
    h2_ref = reference_hellinger2(a, b, J)
    rel = np.abs(h2_path / h2_ref - 1).max()
    assert rel < 1e-3, rel
    return f"0.5*||sqrtP_a - sqrtP_b||^2 matches the compositional Hellinger^2 (J={J}) to rel {rel:.1e}  (H^2 ~ {h2_ref.mean():.2e})"


def test_normalization_and_invariances():
    f = spectrum(2); ivar = RNG.uniform(0.5, 5, (2, N)); ivar[:, 5000:6600] = 0
    path = SmoothStarletPath(J).to(DEV)
    s1, _, sc1 = path(torch.stack([T(f), T(ivar)], 1))
    s2, _, sc2 = path(torch.stack([T(7.3 * f), T(ivar * 123.0)], 1))
    norm = (s1 ** 2).sum((1, 2)).cpu().numpy()
    d = (s1 - s2).abs().max().item()
    ratio = (sc2 / sc1).cpu().numpy()
    assert np.allclose(norm, 1, atol=1e-5) and d < 1e-5 and np.allclose(ratio, 7.3, rtol=1e-5), (norm, d, ratio)
    return (f"sum sqrtP^2 = {norm.round(6).tolist()}; flux x7.3 and ivar x123 change sqrtP by {d:.1e}; "
            f"scale ratio {ratio.round(5).tolist()} (expect 7.3)")


def test_normalize_raw_is_magnitude_invariant_and_keeps_snr():
    f = spectrum(2); ivar = RNG.uniform(0.5, 5, (2, N)); good = np.ones((2, N)); good[:, 5000:6600] = 0
    sm, raw = SmoothStarletPath(J).to(DEV), RawStarletPath(J).to(DEV)
    out = []
    for k in [1.0, 0.013]:                                       # same object, 77x fainter
        xs = torch.stack([T(k * f * good), T(ivar * good / k ** 2)], 1)
        xr = torch.stack([T(k * f * good), T(ivar * good / k ** 2), T(good)], 1)
        _, _, s = sm(xs)
        xn = normalize_raw(xr, s)
        out.append((xn, raw(xn), s))
    (xa, cha, sa), (xb, chb, sb) = out
    snr = lambda x: x[:, 0] * x[:, 1].sqrt()
    d_flux = (xa[:, 0] - xb[:, 0]).abs().max().item(); d_ch = (cha - chb).abs().max().item()
    d_snr = (snr(xa) - snr(torch.stack([T(f * good), T(ivar * good)], 1))).abs().max().item()
    assert d_flux < 1e-4 and d_ch < 1e-4 and d_snr < 1e-3, (d_flux, d_ch, d_snr)
    return (f"object at flux x1 and x0.013: normalized raw flux differs by {d_flux:.1e}, raw starlet channels by {d_ch:.1e}; "
            f"S/N per pixel unchanged by normalization ({d_snr:.1e}); log10 s {torch.log10(sa).cpu().numpy().round(3).tolist()} "
            f"vs {torch.log10(sb).cpu().numpy().round(3).tolist()}")


def test_scale_token():
    tok = ScaleToken(d_token=256).to(DEV)
    s = T(10 ** np.linspace(-1, 2, 7))
    e = tok(s)
    n_params = sum(p.numel() for p in tok.parameters())
    recovered = (e[:, 0, 0] - torch.log10(s)).abs().max().item()
    marker = e[:, 0, 2 * 16 + 1]
    assert e.shape == (7, 1, 256) and n_params == 0 and recovered < 1e-6 and (marker == 1).all() and (e[:, 0, 34:] == 0).all()
    try:
        ScaleToken(d_token=20); raise AssertionError("small d_token accepted")
    except ValueError:
        pass
    return f"shape {tuple(e.shape)}; {n_params} parameters; log10 s recovered from slot 0 to {recovered:.1e}; marker slot = 1; rest zero"


def test_gap_handling():
    f = spectrum(1); ivar = np.ones((1, N)); ivar[:, 5000:6600] = 0
    ch_w, sup = WeightedStarlet(J).to(DEV)(T(f * (ivar > 0)), T(ivar), T(ivar > 0))
    ch_p = plain_starlet(T(f * (ivar > 0)), J)                                  # zero-filled gap, unweighted
    near = np.r_[4900:5000, 6600:6700]                                          # good pixels next to the gap
    ref = plain_starlet(T(f), J)[0][:, near].abs().amax(1)                     # what the gap-free spectrum has
    art_w = (ch_w[0][:, near].abs().amax(1) / ref).cpu().numpy(); art_p = (ch_p[0][:, near].abs().amax(1) / ref).cpu().numpy()
    s = sup[0].cpu().numpy()
    assert torch.isfinite(ch_w).all() and s[:, 5800].max() < 0.05 and s[:, 2000].min() > 0.99
    return ("max |detail| on good px within 100 px of a 1600-px gap, relative to the gap-free spectrum, scales s1..s5: "
            f"weighted {np.round(art_w[:5], 2).tolist()} vs zero-filled plain {np.round(art_p[:5], 1).tolist()}; "
            f"support mid-gap max {s[:, 5800].max():.3f}, far from gap min {s[:, 2000].min():.3f}")


def test_real_spectra_end_to_end():
    from astropy.table import Table
    t = Table.read(ROOT / "data" / "data.fits")
    x = torch.stack([T(np.asarray(t[c], np.float32)) for c in ("FLUX", "IVAR", "MASK")], 1)
    pre, raw_p, sm_p = Preprocessor().to(DEV), RawStarletPath(J).to(DEV), SmoothStarletPath(J).to(DEV)
    x_raw, x_smooth = pre(x)
    sp, sup, scale = sm_p(x_smooth)
    xn = normalize_raw(x_raw, scale)
    ch = raw_p(xn)
    good = xn[:, 2] > 0
    err = (ch.sum(1) - xn[:, 0])[good].abs().max().item() / xn[:, 0][good].abs().max().item()
    norm = (sp ** 2).sum((1, 2))
    med = torch.stack([xn[i, 0][good[i]].abs().median() for i in range(len(xn))])
    assert torch.isfinite(ch).all() and torch.isfinite(sp).all() and err < 1e-5 and (norm - 1).abs().max() < 1e-4
    torch.cuda.synchronize() if DEV.type == "cuda" else None
    xb = x_raw[:64].repeat(4, 1, 1); xs = x_smooth[:64].repeat(4, 1, 1)
    raw_p(xb); sm_p(xs); torch.cuda.synchronize() if DEV.type == "cuda" else None
    t0 = time.time(); raw_p(xb); sm_p(xs); torch.cuda.synchronize() if DEV.type == "cuda" else None
    return (f"100 DESI spectra: finite; raw recon rel err {err:.1e}; sum sqrtP^2 in [{norm.min():.5f}, {norm.max():.5f}]; "
            f"log10 s in [{torch.log10(scale).min():.2f}, {torch.log10(scale).max():.2f}], median |normalized flux| "
            f"in [{med.min():.2f}, {med.max():.2f}]; "
            f"both paths B=256, J={J}: {1e3 * (time.time() - t0):.1f} ms; outputs raw {tuple(ch.shape[1:])}, "
            f"sqrtP {tuple(sp.shape[1:])}, support {tuple(sup.shape[1:])}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
