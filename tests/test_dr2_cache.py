import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from DESIFlow.data.dr2_cache import N_PIX, RowWriter, combine, to_cache_format


def test_combine_is_inverse_variance_mean_and_ignores_bad_and_unused():
    rng = np.random.default_rng(0)
    n = 50
    f = rng.normal(size=(3, n))
    iv = rng.uniform(0.5, 2.0, size=(3, n))
    bad = np.zeros((3, n), bool)
    bad[1, :10] = True
    iv[bad], f[bad] = 0.0, 0.0
    use = np.array([True, True, False])
    out_f, out_iv, out_bad, chi2, _ = combine(f, iv, bad, use)
    w = iv[:2]
    assert np.allclose(out_iv, w.sum(0)) and np.allclose(out_f, (w * f[:2]).sum(0) / w.sum(0))
    assert not out_bad.any() and np.isfinite(chi2)
    return f"combine: ivar-weighted mean over the 2 used rows, {int(bad.sum())} bad pixels skipped, chi2_red {chi2:.2f}"


def test_combine_flags_pixels_without_data():
    f, iv = np.ones((2, 5)), np.ones((2, 5))
    bad = np.zeros((2, 5), bool)
    bad[:, 2] = True
    out_f, out_iv, out_bad, _, _ = combine(np.where(bad, 0, f), np.where(bad, 0, iv), bad, np.ones(2, bool))
    assert out_bad.tolist() == [False, False, True, False, False] and out_iv[2] == 0 and out_f[2] == 0
    return "combine: a pixel bad in every contribution comes out bad with flux = ivar = 0"


def test_cache_format_zeroes_bad_pixels_and_uses_binary_mask():
    rng = np.random.default_rng(1)
    f, iv = rng.normal(size=(4, N_PIX)), rng.uniform(1, 2, size=(4, N_PIX))
    m = np.zeros((4, N_PIX), np.int32)
    m[0, :100], m[1, 5] = 65, 4
    iv[2, 200:300] = 0.0
    f[3, 7] = np.nan
    fl, ivo, mk, med, ng = to_cache_format(f, iv, m)
    assert fl.dtype == np.float32 and mk.dtype == np.uint8 and set(np.unique(mk)) == {0, 1}
    assert (fl[mk == 1] == 0).all() and (ivo[mk == 1] == 0).all() and mk[0, :100].all() and mk[2, 200:300].all() and mk[3, 7] == 1
    assert ng[0] == N_PIX - 100 and np.isfinite(med).all()
    return "to_cache_format: DESI bitmask, ivar == 0 and NaN flux all become mask 1 with flux = ivar = 0"


def test_rowwriter_matches_memmap_and_leaves_other_rows_zero():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.npy")
        mm = np.lib.format.open_memmap(p, mode="w+", dtype="f4", shape=(12, 6))
        mm.flush()
        del mm
        w = RowWriter(p)
        rows = np.array([1, 2, 3, 7, 11])
        data = np.arange(30, dtype=np.float32).reshape(5, 6) + 1
        w.write(rows, data)
        w.close()
        out = np.load(p)
        assert np.array_equal(out[rows], data) and out[[0, 4, 5, 6, 8, 9, 10]].sum() == 0
    return "RowWriter: contiguous runs and isolated rows land in the right rows of the npy file"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
