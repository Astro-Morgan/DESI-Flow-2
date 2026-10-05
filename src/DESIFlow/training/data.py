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

    def __init__(self, data, train_idx, batch, seed, masker=None):
        self.data, self.train_idx, self.batch, self.seed = data, np.asarray(train_idx), batch, seed
        self.masker = masker or SpanMasker()
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
        good = (mask == 0) & (ivar > 0)
        hidden, _ = self.masker.sample(good, np.random.default_rng([self.seed, 23, int(step)]))
        z_ok = self.data.trusted(idx)
        return {"flux": flux, "ivar": ivar, "mask": mask, "z": np.where(z_ok, self.data.z[idx], 0.0).astype(np.float32),
                "z_ok": z_ok, "hidden": hidden, "idx": idx}
