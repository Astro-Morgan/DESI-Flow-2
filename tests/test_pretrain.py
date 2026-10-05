"""
Checks for the pretraining pipeline (src/DESIFlow/training/{data,evaluation,plots,pretrain}.py) on a small dataset built from
data/data.fits when present (otherwise random smooth spectra). Run directly (python tests/test_pretrain.py) or with pytest.
"""
import sys, json, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.eidos import Eidos
from DESIFlow.eidos.reader import velocity
from DESIFlow.training.data import PretrainData, StepBatches
from DESIFlow.training.loss import hidden_loss
from DESIFlow.training.masking import apply_hidden, N_PIX
from DESIFlow.training import pretrain
from DESIFlow.training.pretrain import gather_hidden_queries, SMALL

TMP = Path(tempfile.mkdtemp(prefix="pretrain_test_"))


def make_dataset(path, n=100):
    path.mkdir(parents=True, exist_ok=True)
    fits_path = ROOT / "data" / "data.fits"
    if fits_path.exists():
        from astropy.io import fits
        d = fits.open(fits_path)[1].data
        flux, ivar, mask = (np.asarray(d[c], np.float32) for c in ("FLUX", "IVAR", "MASK"))
        z = np.asarray(d["desi_z"], np.float64)
    else:
        rng = np.random.default_rng(0)
        wave = np.linspace(3600, 9824, N_PIX)
        flux = np.stack([5 * (wave / 6000.) ** rng.uniform(-2, 1) + rng.normal(0, 1, N_PIX) for _ in range(n)]).astype(np.float32)
        ivar = np.ones((n, N_PIX), np.float32)
        mask = np.zeros((n, N_PIX), np.float32)
        z = rng.uniform(0.1, 3, n)
    np.save(path / "flux.npy", flux)
    np.save(path / "ivar.npy", ivar)
    np.save(path / "mask.npy", mask.astype(np.int32))
    np.save(path / "z.npy", z)
    return flux.shape[0]


DATA = TMP / "data"
N = make_dataset(DATA)


def test_splits_batches_and_groups():
    d = PretrainData(DATA)
    tr, va, te = d.splits(10, 10, seed=1)
    assert len(set(tr) | set(va) | set(te)) == N == len(tr) + len(va) + len(te)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    ds = StepBatches(d, tr, 8, seed=1)
    a, b = ds[5], ds[5]
    assert all(np.array_equal(a[k], b[k]) for k in a), "a batch must be a pure function of the step"
    assert not np.array_equal(ds[5]["idx"], ds[6]["idx"]) and len(set(ds[5]["idx"])) == 8
    good = (a["mask"] == 0) & (a["ivar"] > 0)
    assert not (a["hidden"] & ~good).any() and a["hidden"].any() and a["flux"].dtype == np.float32 and np.isfinite(a["flux"]).all()
    seen = np.concatenate([ds[s]["idx"] for s in range(ds.steps_per_epoch)])
    assert len(set(seen)) == len(seen), "no row repeats within an epoch"
    np.save(DATA / "group.npy", np.arange(N) // 4)                                     # groups of 4 rows
    dg = PretrainData(DATA)
    tr2, va2, te2 = dg.splits(10, 10, seed=1)
    g = dg.group
    assert not (set(g[tr2]) & set(g[va2])) and not (set(g[tr2]) & set(g[te2])) and not (set(g[va2]) & set(g[te2])), "a group must not straddle splits"
    assert len(va2) >= 10 and len(te2) >= 10
    (DATA / "group.npy").unlink()
    return (f"{N} rows split {len(tr)}/{len(va)}/{len(te)} disjointly; batches deterministic per step, no repeats within an epoch, "
            f"hidden pixels are good pixels; the group-aware split keeps groups whole")


def test_hidden_only_queries_equal_full_decode():
    torch.manual_seed(0)
    net = Eidos(**SMALL).eval()
    d = PretrainData(DATA)
    ds = StepBatches(d, np.arange(N), 4, seed=0)
    b = ds[0]
    flux, ivar, mask, hid = (torch.tensor(b[k]) for k in ("flux", "ivar", "mask", "hidden"))
    wave = torch.linspace(3600., 9824., N_PIX, dtype=torch.float64)
    qv = velocity(wave, net.wave0)
    with torch.no_grad():
        lat = net.encode(apply_hidden(torch.stack([flux, ivar, mask], 1), hid))
        s = 10 ** lat[:, 0, 0]
        full = hidden_loss(net.decoder(lat[:, 1:], qv), s, flux, ivar, hid)
        qi, qw = gather_hidden_queries(hid)
        part = hidden_loss(net.decoder(lat[:, 1:], qv[qi]), s, flux.gather(1, qi), ivar.gather(1, qi), qw)
    assert qi.shape[1] == int(hid.sum(1).max()) and qw.sum() == hid.sum()
    assert abs(full.item() - part.item()) < 1e-4 * abs(full.item()), (full.item(), part.item())
    return f"decoding only the hidden pixels (Q={qi.shape[1]} of {N_PIX}, padded) gives the same loss as decoding all pixels: {part.item():.4f} vs {full.item():.4f}"


def test_end_to_end_run_resume_and_outputs():
    out = TMP / "run"
    common = ["--data-dir", str(DATA), "--out", str(out), "--model", "small", "--batch", "4", "--val-n", "8", "--test-n", "8", "--workers", "0",
              "--eval-every", "4", "--recon-every", "4", "--save-every", "4", "--log-every", "2", "--warmup", "2", "--eval-batch", "8",
              "--plot-z", "0.2,0.8,1.5"] + ([] if torch.cuda.is_available() else ["--cpu"])
    s1 = pretrain.main(common + ["--steps", "8"])
    assert s1["steps_done"] == 8
    for f in ("config.json", "splits.npz", "train_log.jsonl", "eval_log.jsonl", "curves.png", "best.pt", "last.pt", "z_scatter_latest.npz",
              "recon/latest.png", "recon/step_0000008.png"):
        assert (out / f).exists(), f
    ev = [json.loads(l) for l in (out / "eval_log.jsonl").read_text().splitlines()]
    tr = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in ev] == [4, 8] and [t["step"] for t in tr] == [2, 4, 6, 8]
    for k in ("val_masked_chi2", "val_total_chi2", "val_unmasked_chi2", "z_rmse_log1p", "z_sigma_nmad", "z_cat_frac"):
        assert np.isfinite(ev[-1][k]), k
    for k in ("chi2_hidden", "chi2_total", "z_loss", "grad_norm", "cos", "along"):
        assert np.isfinite(tr[-1][k]), k
    assert all(t["along"] >= 1 - 1e-6 for t in tr), "PCGrad keeps at least the reconstruction-only progress"
    ck = torch.load(out / "best.pt", weights_only=False)
    assert ck["metric"] == min(e["val_masked_chi2"] for e in ev)
    pretrain.main(common + ["--steps", "12", "--resume"])
    tr2 = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert [t["step"] for t in tr2] == [2, 4, 6, 8, 10, 12], "resume must continue the logs from step 8"
    return (f"8 steps then --resume to 12: logs continue at step 8, best.pt holds the best validation hidden-pixel chi2 ({ck['metric']:.3f}), "
            f"figures and checkpoints written; PCGrad progress kept >= 1 in every logged window")


def test_underscore_options_are_accepted():
    assert pretrain.normalize_argv(["--recon_every", "30", "--out=/a_b/c", "--data-dir", "/x_y"]) == ["--recon-every", "30", "--out=/a_b/c", "--data-dir", "/x_y"]
    a = pretrain.build_parser().parse_args(pretrain.normalize_argv(["--data-dir", "d", "--out", "o", "--recon_every", "30", "--eval_every", "10"]))
    assert a.recon_every == 30 and a.eval_every == 10
    return "--recon_every / --eval_every are read as --recon-every / --eval-every; path values containing underscores are untouched"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
