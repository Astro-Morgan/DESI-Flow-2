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


def test_scale_excess_is_one_for_noise_and_flags_structure():
    from DESIFlow.preprocessing.preprocessing import Preprocessor
    from DESIFlow.eidos.starlet import RawStarletPath
    from DESIFlow.training.diagnose import scale_excess
    torch.manual_seed(0)
    pre, path = Preprocessor(), RawStarletPath(9)
    B = 12
    ivar = torch.full((B, N_PIX), 4.0)
    mask = torch.zeros(B, N_PIX)
    noise = torch.randn(B, N_PIX) / 2.0                                              # sigma = 1/sqrt(ivar) = 0.5
    gen = torch.Generator().manual_seed(1)
    e0 = scale_excess(pre, path, noise, ivar, mask, gen=gen).median(0).values
    assert (e0[:7] > 0.8).all() and (e0[:7] < 1.25).all(), e0                    # s8, s9 average few independent samples per spectrum: noisy
    x = torch.arange(N_PIX, dtype=torch.float32)
    bump = 1.0 * torch.exp(-0.5 * ((x - 3000) / 60.0) ** 2)                          # 60 px sigma (~48 A, ~2400 km/s), 2 sigma_noise tall
    e1 = scale_excess(pre, path, noise + bump, ivar, mask, gen=gen).median(0).values
    assert e1[:5].max() < 1.25 and e1[7:].max() > 2.0, e1                            # fine scales stay at noise, the matching coarse scales light up
    return (f"pure-noise residual: E_j in [{e0[:7].min():.2f}, {e0[:7].max():.2f}] at scales 1-7 (the two coarsest average few independent samples per spectrum); "
            f"with a broad bump added: E_1..E_5 <= {e1[:5].max():.2f}, E_8, E_9 = {e1[7]:.1f}, {e1[8]:.1f}")


def test_diagnose_runs_on_a_checkpoint():
    from DESIFlow.training import diagnose
    out = TMP / "run"
    rep = diagnose.main(["--data-dir", str(DATA), "--run-dir", str(out), "--n", "8", "--batch", "4"] + ([] if torch.cuda.is_available() else ["--cpu"]))
    assert (out / "diagnose.json").exists() and len(rep["bins"]) >= 1
    b = rep["bins"][0]
    assert len(b["E_unmasked"]) == 10 and np.isfinite(b["E_unmasked"]).all() and np.isfinite(b["E_hidden"]).all() and np.isfinite(b["chi2_unmasked"])
    ev = [json.loads(l) for l in (out / "eval_log.jsonl").read_text().splitlines()]
    assert "masked_chi2_by_snr" in ev[-1] and "unmasked_chi2_by_snr" in ev[-1]
    return f"diagnose ran on best.pt: {len(rep['bins'])} S/N bins, 10 starlet scales, json written; eval_log now carries chi2 by S/N bin"


def test_degradation_is_brightness_tied_mild_biased_and_exact():
    from DESIFlow.training.data import degrade_inputs, object_snr
    rng = np.random.default_rng(0)
    B, N = 64, 4000
    q_true = np.repeat([0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 10.0, 30.0], 8)               # per-pixel S/N of each object (8 objects per level)
    ivar = np.ones((B, N), np.float32) * 4.0
    flux = (q_true[:, None] / 2.0 * np.ones((B, N))).astype(np.float32)                 # flux * sqrt(ivar) = q
    mask = np.zeros((B, N), np.float32)
    mask[:, 100:140] = 1.0                                                              # some bad pixels
    f_in, iv_in, g, q = degrade_inputs(flux, ivar, mask, rng, q_floor=1.0, power=2.0)
    assert np.allclose(q, q_true, rtol=1e-3)
    assert (g[q_true <= 1.0] == 1.0).all(), "objects at or below q_floor must not be degraded"
    assert np.array_equal(f_in[q_true <= 1.0], flux[q_true <= 1.0]) and np.array_equal(iv_in[q_true <= 1.0], ivar[q_true <= 1.0])
    assert (g <= np.maximum(q_true, 1.0) + 1e-5).all() and (g >= 1.0).all(), "g is bounded by q / q_floor"
    assert np.array_equal(f_in[:, 100:140], flux[:, 100:140]) and np.array_equal(iv_in[:, 100:140], ivar[:, 100:140]), "bad pixels untouched"
    ok = mask == 0
    added = ((f_in - flux) ** 2 * ivar)[ok.nonzero()[0], ok.nonzero()[1]]                # (added noise / original sigma)^2, expectation g^2 - 1
    per_obj = np.array([((f_in[i] - flux[i]) ** 2 * ivar[i])[ok[i]].mean() for i in range(B)])
    big = g > 1.5
    assert np.allclose(per_obj[big], g[big] ** 2 - 1, rtol=0.35), (per_obj[big][:5], (g[big] ** 2 - 1)[:5])   # donor pattern modulates, mean follows g^2 - 1
    assert np.allclose(1.0 / iv_in[ok] - 1.0 / ivar[ok] >= -1e-6, True) and (iv_in[ok] <= ivar[ok] + 1e-6).all(), "input ivar is the combined (never larger) ivar"
    gs = np.concatenate([degrade_inputs(flux, ivar, mask, np.random.default_rng(k), 1.0, 2.0)[2][-8:] for k in range(300)])    # the S/N-30 objects
    gmax = 30.0
    assert 0.65 < np.mean(gs < np.sqrt(gmax)) < 0.78, "u**2 skew: ~71% of draws below sqrt(g_max)"
    return (f"q <= 1 untouched; g <= q/q_floor; bad pixels untouched; added noise variance = g^2 - 1 (within the donor-pattern scatter); input ivar = combined; "
            f"{100 * np.mean(gs < np.sqrt(gmax)):.0f}% of an S/N-30 object's draws are below sqrt(g_max)")


def test_z_log_loss_has_floor_and_growing_gradient():
    from DESIFlow.training.zhead import ZHead
    eps = 1000.0 / 299792.458
    z = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    d = torch.tensor([0.0, 0.04, 0.0004], dtype=torch.float64)
    pred = (torch.log1p(z) + d).requires_grad_(True)
    ZHead.log_loss(pred, z, None, eps).backward()
    g = pred.grad * 3
    assert torch.allclose(g, d / (d ** 2 + eps ** 2), rtol=1e-6), (g, d / (d ** 2 + eps ** 2))
    assert g[1] > 20 * 2 * 0.04, "far above the floor the gradient (~1/D) is far larger than MSE's 2D"
    l0 = 0.5 * np.log(eps ** 2)
    assert abs(ZHead.log_loss(torch.log1p(z), z, None, eps).item() - l0) < 1e-9, "loss at D = 0 is log(eps)"
    assert ZHead.log_loss(pred.detach(), z, torch.zeros(3, dtype=torch.bool), eps) is None
    return f"L = 1/2 log(D^2 + eps^2), eps = {eps:.2e} (1000 km/s): gradient D/(D^2+eps^2), {g[1].item() / (2 * 0.04):.0f}x MSE's at D = 4%, zero at D = 0, loss floor log(eps)"


def test_e_statistics_logged_and_run_uses_new_defaults():
    out = TMP / "run"
    ev = [json.loads(l) for l in (out / "eval_log.jsonl").read_text().splitlines()]
    tr = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(ev[-1]["E_by_scale"]) == 10 and np.isfinite(ev[-1]["E_by_scale"]).all() and len(ev[-1]["E1_by_snr"]) >= 1 and np.isfinite(ev[-1]["val_z_loss"])
    assert all(np.isfinite(t["g_mean"]) and t["g_mean"] >= 1 and 0 <= t["z_cap"] <= 1 for t in tr)
    cfg = json.load(open(out / "config.json"))["args"]
    assert cfg["huber"] == 0.0 and cfg["mask_frac_max"] == 0.75 and cfg["degrade"] is True and cfg["z_loss"] == "log" and cfg["z_grad_cap"] is True
    return (f"eval log carries E_by_scale (10 scales), E1_by_snr ({len(ev[-1]['E1_by_snr'])} bins) and val_z_loss; train log carries mean degradation g "
            f"({tr[-1]['g_mean']:.2f}) and the z-cap scale; defaults: plain chi2, masks 5-75%, degradation on, log z loss, z cap on")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
