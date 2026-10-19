"""
Pretraining data: row-matched {flux, ivar, mask, z}.npy on the native 7781-pixel DESI grid (3600-9824 A, 0.8 A), read through
numpy memory maps so a 60+ GB set never has to fit in RAM.

    flux.npy  (N, 7781)  float      1e-17 erg/s/cm2/A
    ivar.npy  (N, 7781)  float
    mask.npy  (N, 7781)  int        DESI pixel mask, 0 = good
    z.npy     (N,)       float      redshift (taken as trusted)
    [optional] group.npy (N,)       any integer/string ids (TARGETID, HEALPix pixel, ...): splits then never separate rows of one group

Splits are deterministic (seed) and disjoint: test, validation, train. Batches are a pure function of the step number
(epoch permutation seeded by the epoch, masks seeded by the step), so a resumed run continues exactly where it stopped.
Rows are sanitized on read: non-finite flux/ivar -> 0, negative ivar -> 0 (those pixels are then "not good").

Noise degradation (training only, `degrade_inputs`): the model INPUT of a bright object gets extra noise, the loss TARGETS stay the original
data. The ceiling on the noise inflation g (sigma_new / sigma_old) is tied to brightness, g_max = q / q_floor with q the object's median per-pixel
S/N, so faint objects (q <= q_floor) are never degraded and the brightest can be taken down to single-exposure levels; g is drawn log-uniformly
but skewed toward mild degradation (log g = log g_max * u**power). The added noise follows the receiver's own noise profile modulated by the relative
noise pattern of a random donor object of the batch, so the noise profile the encoder sees keeps changing. Input ivar = the combined ivar.
"""
from pathlib import Path
import numpy as np
import torch
from DESIFlow.training.masking import SpanMasker, N_PIX


def sanitize(flux, ivar, mask):
    """float32 copies with non-finite values removed; returns flux, ivar, mask (float32; any nonzero mask is bad)."""
    flux = np.asarray(flux, np.float32)
    ivar = np.asarray(ivar, np.float32)
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar < 0)
    flux = np.where(bad, 0.0, flux).astype(np.float32)
    ivar = np.where(bad, 0.0, ivar).astype(np.float32)
    mask = np.asarray(mask).astype(np.float32)
    mask = np.where(bad | ~np.isfinite(mask), 1.0, mask).astype(np.float32)
    return flux, ivar, mask


def object_snr(flux, ivar, mask):
    """Median per-pixel S/N (flux * sqrt(ivar), good pixels) of each row; 0 for rows without good pixels."""
    good = (mask == 0) & (ivar > 0)
    sn = flux * np.sqrt(np.maximum(ivar, 0))
    return np.array([np.median(sn[i][good[i]]) if good[i].any() else 0.0 for i in range(len(flux))])


def degrade_inputs(flux, ivar, mask, rng, q_floor=1.0, power=2.0, shape_clip=10.0):
    """Brightness-tied noise degradation of model inputs. Arrays (B, N) float32 -> (flux_in, ivar_in, g (B,), q (B,)).
    g = 1 (no change at all) for objects with q <= q_floor. See the module docstring for the rule."""
    B = len(flux)
    good = (mask == 0) & (ivar > 0)
    q = object_snr(flux, ivar, mask)
    g_max = np.maximum(q / q_floor, 1.0)
    g = np.exp(np.log(g_max) * rng.random(B) ** power)
    var = np.where(good, 1.0 / np.maximum(ivar, 1e-30), 0.0)
    mean_var = var.sum(1) / np.maximum(good.sum(1), 1)
    shape = np.where(good, var / np.maximum(mean_var[:, None], 1e-30), 1.0)         # donor noise pattern, unit mean over its good pixels
    shape = np.minimum(shape, shape_clip)
    donor = (np.arange(B) + rng.integers(1, B, size=B)) % B if B > 1 else np.zeros(1, int)
    add_var = (g ** 2 - 1.0)[:, None] * var * shape[donor]                           # variance added on top of the receiver's own
    noise = rng.standard_normal(flux.shape) * np.sqrt(add_var)
    flux_in = np.where(good, flux + noise, flux).astype(np.float32)
    ivar_in = np.where(good, 1.0 / (var + add_var + 1e-30), ivar).astype(np.float32)
    return flux_in, ivar_in, g.astype(np.float32), q.astype(np.float32)


def lsf_kernels(sig_px, K):
    """(B, 2K+1) normalised Gaussian kernels on integer pixel offsets whose VARIANCE equals sig_px**2 (B,) exactly: a Gaussian sampled at integer offsets under-broadens below ~0.7 px
    (at sigma 0.25 px its variance is ~0), so the width of the sampled Gaussian is solved for the target variance. sig_px = 0 -> a delta."""
    j = np.arange(-K, K + 1)
    grid = np.linspace(0.05, 4.0, 400)
    kg = np.exp(-0.5 * (j[None, :] / grid[:, None]) ** 2)
    var_grid = (kg * j[None, :] ** 2).sum(1) / kg.sum(1)
    s_eff = np.where(sig_px > 0, np.interp(np.asarray(sig_px) ** 2, var_grid, grid), 1e-6)
    k = np.exp(-0.5 * (j[None, :] / s_eff[:, None]) ** 2)
    return k / k.sum(1, keepdims=True)


def broaden_lsf(flux, ivar, mask, rng, sigma_max=0.6, prob=0.5, pix=0.8, min_cov=0.6):
    """LSF augmentation: re-observe the spectra as an instrument with a WIDER line spread function. Arrays (B, N) float32 -> (flux, ivar, mask, sigma (B,)).
    A fraction `prob` of the objects get an extra Gaussian of sigma ~ U(0, sigma_max) Angstrom (constant in wavelength = in native pixels, like the DESI LSF: R 2000 -> 5500 is
    ~0.76 A everywhere; Gaussian line widths add in quadrature, so it can only broaden, never sharpen). Applied to the data BEFORE masking and noise degradation, so the model INPUT and
    the loss TARGET are the same broadened spectrum (the model learns to reconstruct whatever width it sees, not to deconvolve).
    Pixel rule: kernel-weighted mean over the GOOD pixels in reach, variance sum(w^2 var)/(sum w)^2 (neighbouring outputs become correlated, which the pipeline ignores, as it
    does for the resampled grid); an output pixel stays good only if it was good AND the good kernel weight covers >= min_cov (bad pixels are never filled in)."""
    B, N = flux.shape
    sig = rng.random(B) * sigma_max * (rng.random(B) < prob)
    if not (sig > 0).any():
        return flux, ivar, mask, sig.astype(np.float32)
    K = max(int(np.ceil(4 * sigma_max / pix)), 1)
    j = np.arange(-K, K + 1)
    k = lsf_kernels(sig / pix, K)                                                    # (B, 2K+1); sigma = 0 -> a delta
    good = (mask == 0) & (ivar > 0)
    var = np.where(good, 1.0 / np.maximum(ivar, 1e-30), 0.0)

    def shift(a, d):                                                                 # out[:, i] = a[:, i + d], zero outside
        out = np.zeros_like(a)
        if d >= 0:
            out[:, :N - d] = a[:, d:]
        else:
            out[:, -d:] = a[:, :N + d]
        return out

    sw, swf, sw2v = np.zeros((B, N)), np.zeros((B, N)), np.zeros((B, N))
    for t, d in enumerate(j):
        w = k[:, t:t + 1] * shift(good.astype(np.float64), d)
        sw += w
        swf += w * shift(flux.astype(np.float64), d)
        sw2v += w * w * shift(var, d)
    keep = good & (sw >= min_cov)
    f_new = np.where(keep, swf / np.maximum(sw, 1e-30), 0.0)
    iv_new = np.where(keep, sw * sw / np.maximum(sw2v, 1e-300), 0.0)
    m_new = np.where(good & ~keep, 1.0, mask)
    on = (sig > 0)[:, None]
    return (np.where(on, f_new, flux).astype(np.float32), np.where(on, iv_new, ivar).astype(np.float32), np.where(on, m_new, mask).astype(np.float32), sig.astype(np.float32))


class PretrainData:
    def __init__(self, data_dir, group_file=None):
        d = Path(data_dir)
        self.flux = np.load(d / "flux.npy", mmap_mode="r")
        self.ivar = np.load(d / "ivar.npy", mmap_mode="r")
        self.mask = np.load(d / "mask.npy", mmap_mode="r")
        self.z = np.load(d / "z.npy").astype(np.float64).reshape(-1)
        gf = Path(group_file) if group_file else (d / "group.npy" if (d / "group.npy").exists() else None)
        self.group = np.load(gf) if gf else None
        n = len(self.z)
        for name, a in (("flux", self.flux), ("ivar", self.ivar), ("mask", self.mask)):
            if a.shape != (n, N_PIX):
                raise ValueError(f"{name}.npy has shape {a.shape}, expected ({n}, {N_PIX}) to match z.npy and the 7781-pixel native grid")
        if self.group is not None and len(self.group) != n:
            raise ValueError("group file length does not match z.npy")
        self.n = n

    def read(self, idx):
        """Sorted-row read -> sanitized float32 (flux, ivar, mask) for the given row indices (in the order given)."""
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        s = idx[order]
        f, i, m = sanitize(self.flux[s], self.ivar[s], self.mask[s])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return f[inv], i[inv], m[inv]

    def trusted(self, idx=None):
        z = self.z if idx is None else self.z[idx]
        return np.isfinite(z) & (z >= 0)

    def splits(self, val_n, test_n, seed=0):
        """(train_idx, val_idx, test_idx), sorted, disjoint. With a group file, whole groups go to val/test."""
        rng = np.random.default_rng(seed)
        if self.group is None:
            perm = rng.permutation(self.n)
            test, val, train = perm[:test_n], perm[test_n:test_n + val_n], perm[test_n + val_n:]
        else:
            _, inv = np.unique(self.group, return_inverse=True)
            ng = inv.max() + 1
            counts = np.bincount(inv, minlength=ng)
            order = rng.permutation(ng)
            csum = np.cumsum(counts[order])
            n_test_g = int(np.searchsorted(csum, test_n)) + 1                 # whole groups until test_n rows are reached
            n_val_g = int(np.searchsorted(csum - csum[n_test_g - 1], val_n)) + 1 - n_test_g
            role = np.zeros(ng, np.int8)                                  # 0 train, 1 val, 2 test
            role[order[:n_test_g]] = 2
            role[order[n_test_g:n_test_g + n_val_g]] = 1
            r = role[inv]
            train, val, test = np.flatnonzero(r == 0), np.flatnonzero(r == 1), np.flatnonzero(r == 2)
        return np.sort(train), np.sort(val), np.sort(test)


class StepBatches(torch.utils.data.Dataset):
    """dataset[step] -> the training batch for that step (dict of numpy arrays; use DataLoader(batch_size=None))."""

    def __init__(self, data, train_idx, batch, seed, masker=None, degrade=False, q_floor=1.0, degrade_power=2.0, lsf_aug=0.0, lsf_prob=0.5):
        self.data, self.train_idx, self.batch, self.seed = data, np.asarray(train_idx), batch, seed
        self.lsf_aug, self.lsf_prob = lsf_aug, lsf_prob                    # LSF augmentation (see broaden_lsf): extra sigma up to lsf_aug Angstrom for a fraction lsf_prob; 0 = off
        self.masker = masker or SpanMasker()
        self.degrade, self.q_floor, self.degrade_power = degrade, q_floor, degrade_power
        self.steps_per_epoch = len(self.train_idx) // batch
        if self.steps_per_epoch < 1:
            raise ValueError("training set smaller than one batch")
        self._perm_epoch, self._perm = -1, None

    def __len__(self):
        return 10 ** 12                                                  # indexed by step; the sampler decides the range

    def indices(self, step):
        epoch, i = divmod(int(step), self.steps_per_epoch)
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng([self.seed, 17, epoch]).permutation(len(self.train_idx))
            self._perm_epoch = epoch
        return self.train_idx[self._perm[i * self.batch:(i + 1) * self.batch]]

    def __getitem__(self, step):
        idx = self.indices(step)
        flux, ivar, mask = self.data.read(idx)
        lsf = None
        if self.lsf_aug > 0:
            flux, ivar, mask, lsf = broaden_lsf(flux, ivar, mask, np.random.default_rng([self.seed, 37, int(step)]), self.lsf_aug, self.lsf_prob)
        good = (mask == 0) & (ivar > 0)
        hidden, _ = self.masker.sample(good, np.random.default_rng([self.seed, 23, int(step)]))
        z_ok = self.data.trusted(idx)
        if self.degrade:
            flux_in, ivar_in, g, q = degrade_inputs(flux, ivar, mask, np.random.default_rng([self.seed, 29, int(step)]), self.q_floor, self.degrade_power)
        else:
            flux_in, ivar_in, g, q = flux, ivar, np.ones(len(flux), np.float32), object_snr(flux, ivar, mask).astype(np.float32)
        out = {"flux": flux, "ivar": ivar, "flux_in": flux_in, "ivar_in": ivar_in, "g": g, "q": q, "mask": mask,
               "z": np.where(z_ok, self.data.z[idx], 0.0).astype(np.float32), "z_ok": z_ok, "hidden": hidden, "idx": idx}
        if lsf is not None:
            out["lsf_sigma"] = lsf
        return out
