"""
Checks for Preprocessor (raw and smoothed resampling onto the log-lambda grid).
Run directly (python tests/test_preprocessing.py) or with pytest.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from DESIFlow.preprocessing.preprocessing import Preprocessor, C_KMS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = Preprocessor().to(DEV)
L = P.spec_len
NATIVE = np.linspace(P.wave_min, P.wave_max, L)
NEW = P.new_wave.cpu().numpy()
RNG = np.random.default_rng(0)


def run(flux, ivar, mask, chunk=256):
    """Returns dict of numpy arrays: raw flux/ivar/good and smooth flux/ivar, each (B, new_len).
    Chunked: the gathered kernel windows cost ~5 MB per spectrum on the GPU."""
    flux, ivar, mask = (np.atleast_2d(a) for a in (flux, ivar, mask))
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)
    outs = [P(torch.stack([t(flux[s:s + chunk]), t(ivar[s:s + chunk]), t(mask[s:s + chunk])], dim=1))
            for s in range(0, flux.shape[0], chunk)]                                 # input (B, 3, L)
    xr = np.concatenate([o[0].cpu().numpy() for o in outs]); xs = np.concatenate([o[1].cpu().numpy() for o in outs])
    return {"f_raw": xr[:, 0], "iv_raw": xr[:, 1], "g_raw": xr[:, 2] > 0, "f_s": xs[:, 0], "iv_s": xs[:, 1]}


def _ivar_mask(n=4):
    ivar = RNG.uniform(0.2, 5.0, (n, L)) * (1 + 20 * (RNG.uniform(size=(n, L)) < 0.01))   # spiky, sky-like
    mask = np.zeros((n, L)); mask[:, 2000:2040] = 128; mask[:, RNG.integers(0, L, 60)] = 3   # a gap + scattered bits
    ivar[mask != 0] = 0
    return ivar, mask


# ---------------------------------------------------------------- smoothed output
def test_smooth_constant_reproduced():
    ivar, mask = _ivar_mask()
    r = run(np.full((4, L), 3.7), ivar, mask)
    err = np.abs(r["f_s"][r["iv_s"] > 0] - 3.7).max()
    assert err < 1e-5, err
    return f"constant reproduced to {err:.1e}"


def test_masked_garbage_does_not_leak():
    ivar, mask = _ivar_mask()
    f = np.ones((4, L)); f[mask != 0] = 1e6
    r = run(f, ivar, mask)
    e_s = np.abs(r["f_s"][r["iv_s"] > 0] - 1).max(); e_r = np.abs(r["f_raw"][r["g_raw"]] - 1).max()
    assert e_s < 1e-5 and e_r < 1e-5, (e_s, e_r)
    return f"masked 1e6 values leak: smooth {e_s:.1e}, raw {e_r:.1e}"


def _line(lam0, sig_kms=5.0, amp=1.0):
    return amp * np.exp(-0.5 * (C_KMS * np.log(NATIVE / lam0) / sig_kms) ** 2)


def test_smooth_flux_conserved_and_width_uniform_in_velocity():
    out = []
    dlog = np.log(NEW[1] / NEW[0])
    for lam0 in [4000.0, 6500.0, 9000.0]:
        f = 1.0 + _line(lam0)
        fs = run(f, np.ones(L), np.zeros(L))["f_s"][0]
        flux_native = ((f - 1) * (NATIVE[1] - NATIVE[0])).sum()
        flux_new = ((fs - 1) * NEW * dlog).sum()                                   # d(lambda) = lambda d(ln lambda)
        v = C_KMS * np.log(NEW / lam0); half = (fs - 1) > (fs - 1).max() / 2
        fwhm = v[half].max() - v[half].min() + C_KMS * dlog
        out.append((lam0, flux_new / flux_native, fwhm))
        assert abs(flux_new / flux_native - 1) < 5e-3
    return "; ".join(f"{l:.0f} A: flux ratio {r:.4f}, FWHM {w:.0f} km/s" for l, r, w in out) + f"  (kernel FWHM {2.3548 * P.sigma_v:.0f})"


def test_smooth_noise_matches_propagated_ivar():
    ivar, mask = _ivar_mask(1)
    sig = np.where(ivar > 0, 1 / np.sqrt(np.where(ivar > 0, ivar, 1)), 0)
    r = run(RNG.standard_normal((4000, L)) * sig, np.repeat(ivar, 4000, 0), np.repeat(mask, 4000, 0))
    ok = r["iv_s"][0] > 0
    ratio = r["f_s"][:, ok].var(0) * r["iv_s"][0, ok]
    assert abs(np.median(ratio) - 1) < 0.03, np.median(ratio)
    return f"MC variance x ivar_s: median {np.median(ratio):.3f}, 5-95% [{np.percentile(ratio, 5):.3f}, {np.percentile(ratio, 95):.3f}]"


# ---------------------------------------------------------------- raw output
def test_raw_is_linear_interpolation():
    f = np.cumsum(RNG.standard_normal(L)) * 0.1 + 5
    r = run(f, np.ones(L), np.zeros(L))
    err = np.abs(r["f_raw"][0] - np.interp(NEW, NATIVE, f)).max()
    assert err < 1e-4 and r["g_raw"].all(), err
    return f"uniform ivar, no mask: raw == np.interp to {err:.1e}; all good"


def test_raw_good_never_extends_into_masked_pixels():
    ivar, mask = _ivar_mask(1)
    r = run(RNG.standard_normal(L), ivar, mask)
    bad = np.where(mask[0] != 0)[0]
    near_bad = np.zeros(len(NEW), bool)
    for b in bad:                                                                  # log px strictly between b-1 and b+1
        near_bad |= (NEW > NATIVE[max(b - 1, 0)]) & (NEW < NATIVE[min(b + 1, L - 1)])
    assert not (r["g_raw"][0] & near_bad).any()
    exact = np.isin(np.round((NEW - NATIVE[0]) / (NATIVE[1] - NATIVE[0]), 6) % 1, [0.0])
    return (f"no raw-good log px within a native px of a masked px (n={near_bad.sum()}); "
            f"raw not-good fraction {1 - r['g_raw'].mean():.4f} vs native bad {np.mean(mask != 0):.4f}")


def test_raw_noise_matches_propagated_ivar():
    ivar, mask = _ivar_mask(1)
    sig = np.where(ivar > 0, 1 / np.sqrt(np.where(ivar > 0, ivar, 1)), 0)
    r = run(RNG.standard_normal((4000, L)) * sig, np.repeat(ivar, 4000, 0), np.repeat(mask, 4000, 0))
    ok = r["g_raw"][0]
    ratio = r["f_raw"][:, ok].var(0) * r["iv_raw"][0, ok]
    assert abs(np.median(ratio) - 1) < 0.03, np.median(ratio)
    return f"MC variance x raw ivar: median {np.median(ratio):.3f}, 5-95% [{np.percentile(ratio, 5):.3f}, {np.percentile(ratio, 95):.3f}]"


def test_log_px_per_native_px():
    dnat = (NATIVE[1] - NATIVE[0]) / NATIVE * C_KMS; dnew = C_KMS * np.log(NEW[1] / NEW[0])
    return (f"log px {dnew:.1f} km/s; native px {dnat[0]:.1f} (blue) -> {dnat[-1]:.1f} km/s (red): "
            f"each native px spans {dnat[0] / dnew:.2f} -> {dnat[-1] / dnew:.2f} log px (upsampling everywhere)")


def test_ivar_scale_invariance():
    ivar, mask = _ivar_mask(1)
    f = 1 + _line(5000.0, 60.0) + 0.1 * RNG.standard_normal(L)
    a, b = run(f, ivar, mask), run(f, 1e3 * ivar, mask)
    err = max(np.abs(a["f_s"] - b["f_s"]).max(), np.abs(a["f_raw"] - b["f_raw"]).max())
    assert err < 1e-5, err
    return f"ivar x1000 changes outputs by {err:.1e}"


def test_speed():
    B = 256
    x = torch.stack([torch.randn(B, L, device=DEV), torch.rand(B, L, device=DEV) + 0.1, torch.zeros(B, L, device=DEV)], dim=1)
    P(x); torch.cuda.synchronize() if DEV.type == "cuda" else None
    t0 = time.time()
    for _ in range(5):
        P(x)
    torch.cuda.synchronize() if DEV.type == "cuda" else None
    return f"B={B}: {1e3 * (time.time() - t0) / 5:.1f} ms per batch on {DEV.type}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
