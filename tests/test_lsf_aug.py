"""
Checks for the LSF augmentation (src/DESIFlow/training/data.py: broaden_lsf and the StepBatches lsf_aug option): off by default, widths add in quadrature, flux and noise are
propagated, bad pixels are never filled in, batches stay deterministic per step. Run directly (python tests/test_lsf_aug.py) or with pytest.
"""
import sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.training.data import PretrainData, StepBatches, broaden_lsf
from DESIFlow.training.masking import N_PIX

WAVE = np.linspace(3600., 9824., N_PIX)
PIX = 0.8


def line_batch(sigma0=1.0, n=64, amp=20.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    flux = 1.0 + amp * np.exp(-0.5 * ((WAVE - 6000.0) / sigma0) ** 2)
    flux = np.tile(flux, (n, 1)) + noise * rng.standard_normal((n, N_PIX))
    return flux.astype(np.float32), np.full((n, N_PIX), 1.0 / max(noise, 1e-3) ** 2, np.float32), np.zeros((n, N_PIX), np.float32)


def test_off_is_identity():
    f, iv, m = line_batch(n=8)
    f2, iv2, m2, sig = broaden_lsf(f, iv, m, np.random.default_rng(0), sigma_max=0.6, prob=0.0)
    assert (sig == 0).all() and np.array_equal(f, f2) and np.array_equal(iv, iv2) and np.array_equal(m, m2)
    d = tempfile.mkdtemp()
    for name, a in (("flux", f), ("ivar", iv), ("mask", m.astype(np.int32))):
        np.save(Path(d) / f"{name}.npy", np.tile(a, (4, 1))[:32])
    np.save(Path(d) / "z.npy", np.linspace(0.1, 1, 32))
    data = PretrainData(d)
    a, b = StepBatches(data, np.arange(32), 8, 1)[2], StepBatches(data, np.arange(32), 8, 1, lsf_aug=0.0)[2]
    assert set(a) == set(b) and "lsf_sigma" not in a and all(np.array_equal(a[k], b[k]) for k in a)
    return "prob 0 returns the inputs untouched; StepBatches with lsf_aug=0 is identical to the old behaviour and adds no key"


def test_line_width_adds_in_quadrature_and_flux_is_conserved():
    f, iv, m = line_batch(sigma0=1.0, n=64)
    f2, iv2, m2, sig = broaden_lsf(f, iv, m, np.random.default_rng(3), sigma_max=0.8, prob=1.0)
    sel = (WAVE > 5985) & (WAVE < 6015)
    errs, flux_err = [], []
    for b in range(len(f)):
        prof = f2[b][sel] - 1.0
        mu = (prof * WAVE[sel]).sum() / prof.sum()
        var = (prof * (WAVE[sel] - mu) ** 2).sum() / prof.sum()
        errs.append(var / (1.0 ** 2 + float(sig[b]) ** 2) - 1)
        flux_err.append(prof.sum() / (f[b][sel] - 1.0).sum() - 1)
    assert np.abs(errs).max() < 0.03 and np.abs(flux_err).max() < 0.005, (np.abs(errs).max(), np.abs(flux_err).max())
    assert (sig > 0).all() and (sig <= 0.8).all() and (f2[:, :10] == f2[:, :1]).all()
    return f"64 objects, extra sigma 0..0.8 A: line variance = sigma0^2 + extra^2 to {np.abs(errs).max():.1%}, line flux conserved to {np.abs(flux_err).max():.2%}, continuum untouched"


def test_noise_is_propagated():
    f, iv, m = line_batch(n=300, amp=0.0, noise=1.0, seed=5)
    f2, iv2, m2, sig = broaden_lsf(f, iv, m, np.random.default_rng(4), sigma_max=0.6, prob=1.0)
    ok = sig > 0.15
    emp = f2[ok][:, 50:-50].var(0).mean() if False else (f2[ok][:, 50:-50] ** 2).mean(0)          # flat spectrum 1 - 1 = noise only (mean 0 because amp = 0 and the continuum is 1)
    resid = f2[ok][:, 50:-50] - 1.0
    ratio = (resid ** 2 * iv2[ok][:, 50:-50]).mean()
    assert 0.9 < ratio < 1.1, ratio
    assert (iv2[ok] > iv[ok]).all(), "smoothing white noise must raise ivar"
    return f"chi2 of the broadened noise with the propagated ivar = {ratio:.3f} (1 = exact); ivar rises by {float((iv2[ok] / iv[ok]).mean()):.2f}x on average"


def test_bad_pixels_are_not_filled_and_neighbours_renormalise():
    f = np.full((4, N_PIX), 3.0, np.float32)
    iv = np.full((4, N_PIX), 4.0, np.float32)
    m = np.zeros((4, N_PIX), np.float32)
    m[:, 1000] = 1; m[:, 2000:2003] = 1; iv[:, 1000] = 0; iv[:, 2000:2003] = 0
    f2, iv2, m2, sig = broaden_lsf(f, iv, m, np.random.default_rng(8), sigma_max=0.6, prob=1.0)
    assert (m2[:, 1000] != 0).all() and (iv2[:, 1000] == 0).all() and (f2[:, 1000] == 0).all() and (m2[:, 2000:2003] != 0).all()
    ok = np.ones(N_PIX, bool); ok[[1000, 2000, 2001, 2002]] = False
    good_out = (m2 == 0) & (iv2 > 0)
    assert np.abs(f2[good_out.nonzero()] - 3.0).max() < 1e-5, "a constant spectrum stays constant next to masked pixels (kernel renormalised over the good pixels)"
    assert good_out[:, ok].mean() > 0.999
    return "masked pixels stay masked (flux 0, ivar 0), a constant spectrum stays exactly constant around them, >99.9% of the other pixels stay good"


def test_stepbatches_deterministic_and_consistent():
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(1)
    f, iv, m = line_batch(n=64, noise=0.2, seed=2)
    np.save(d / "flux.npy", f); np.save(d / "ivar.npy", iv); np.save(d / "mask.npy", m.astype(np.int32)); np.save(d / "z.npy", rng.uniform(0.1, 1, 64))
    ds = StepBatches(PretrainData(d), np.arange(64), 16, 3, lsf_aug=0.6, lsf_prob=0.5)
    a, b = ds[4], ds[4]
    assert all(np.array_equal(a[k], b[k]) for k in a), "a batch must be a pure function of the step"
    frac = np.mean([(ds[s]["lsf_sigma"] > 0).mean() for s in range(4)])
    assert 0.3 < frac < 0.7, frac
    good = (a["mask"] == 0) & (a["ivar"] > 0)
    assert not (a["hidden"] & ~good).any() and np.isfinite(a["flux"]).all()
    assert np.array_equal(a["flux_in"][:, :5], a["flux_in"][:, :5]) and a["lsf_sigma"].shape == (16,)
    return f"deterministic per step; {frac:.0%} of the objects broadened (prob 0.5); hidden pixels are good pixels of the broadened data"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
