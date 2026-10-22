"""
Checks for the metric package (src/DESIFlow/metric): the rest-frame Hellinger evaluation, the clean synthetic suites, the noise scaling to a median per-pixel S/N, the stress measures and the
fidelity curves, and the DR2 projection (completeness, projected stress, S/N cut needed), plus the figures. CPU only. Run directly (python tests/test_metric_fidelity.py) or with pytest.
"""
import sys, json, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.starlet import WeightedStarlet
from DESIFlow.metric import fidelity, dr2_projection
from DESIFlow.metric.hellinger import rest_cells, hellinger_cross, hellinger_cross_direct, hellinger_paired, RestMetric
from DESIFlow.metric.synth import CleanSky, N_PIX, wave_grid
from DESIFlow.training.data import object_snr

TMP = Path(tempfile.mkdtemp(prefix="metric_fidelity_test_"))
_CACHE = {}


def line_content(rng, n=13000, n_lines=30):
    u = np.arange(n)
    g = np.ones(n)
    for c, a in zip(rng.uniform(300, n - 300, n_lines), rng.uniform(-0.5, 3.0, n_lines)):
        g += a * np.exp(-0.5 * ((u - c) / 4.0) ** 2)
    return g


# ------------------------------------------------------------------ the metric
def test_rest_alignment():
    """The same rest-frame content observed at two shifts gives (nearly) the same cells once aligned; ignoring the shifts does not."""
    rng = np.random.default_rng(1)
    g = line_content(rng, 13000)
    fA = g                                                                                  # shift 0:  f[p] = g[p]
    fB = np.concatenate([np.ones(40), g[:12960]])                                           # shift 40: f[p] = g[p - 40] (the rest coordinate u = p - shift is what matches)
    f = torch.tensor(np.stack([fA, fB]))
    w = torch.ones_like(f)
    ws = WeightedStarlet(8).double()
    aligned, plain = rest_cells(f, w, [0, 40], ws), rest_cells(f, w, [0, 0], ws)
    d_al = hellinger_cross(*aligned, *aligned)[0, 1].item()
    d_pl = hellinger_cross(*plain, *plain)[0, 1].item()
    assert d_al < 0.25 * d_pl, (d_al, d_pl)
    assert aligned[0].shape == (2, 8, 2, 13040) and aligned[1].shape == (2, 13040)
    return f"aligned D {d_al:.3f} vs unaligned {d_pl:.3f}; cells (N, 8 scales, 2 signs, L + shift spread)"


def test_matrix_form_equals_the_definition():
    """The matrix-product evaluation (D^2 = 1 - sum sqrt(P_i P_j)) equals the per-pair renormalised definition, with different masks per object, between two different sets, and for matched rows."""
    rng = np.random.default_rng(5)
    f = torch.tensor(np.stack([line_content(rng) * (0.5 + rng.random()) for _ in range(7)]))
    w = torch.rand_like(f) + 0.5
    for i, (a, b) in enumerate([(0, 0), (1000, 2500), (7000, 7600), (200, 5000), (0, 0), (11000, 12000), (3000, 3100)]):
        w[i, a:b] = 0                                                                         # a different gap in most objects
    cells, good = rest_cells(f, w, [0, 15, 40, 3, 90, 60, 25], WeightedStarlet(8).double())
    A, B = (cells[:5], good[:5]), (cells[2:], good[2:])
    D, D0 = hellinger_cross(*A, *B), hellinger_cross_direct(*A, *B)
    assert torch.isfinite(D).all() and (D - D0).abs().max().item() < 1e-6, (D - D0).abs().max().item()          # identical pairs sit at sqrt(machine epsilon) ~ 1e-8 in the matrix form
    far = D0 > 0.01
    assert ((D - D0).abs()[far] / D0[far]).max().item() < 1e-9
    Dp = hellinger_paired(cells[:5], good[:5], cells[2:7], good[2:7])
    assert (Dp - torch.diag(D0)).abs().max().item() < 1e-6
    Ds = hellinger_cross(cells, good, cells, good)
    assert (Ds - hellinger_cross_direct(cells, good, cells, good)).abs().max().item() < 1e-6 and Ds.diagonal().abs().max().item() < 1e-6
    assert torch.allclose(Ds, Ds.T, atol=1e-6) and (Ds <= 1.0 + 1e-9).all()
    w2 = w.clone(); w2[0, :9000] = 0; w2[1, 4000:] = 0                                        # objects 0 and 1 then share only a few hundred pixels
    c2, g2 = rest_cells(f, w2, [0] * 7, WeightedStarlet(8).double())
    D2 = hellinger_cross(c2, g2, c2, g2, min_overlap=1000)
    assert torch.isinf(D2[0, 1]) and torch.isfinite(D2[0, 2])
    return f"max |matrix form - definition| {(D - D0).abs().max().item():.1e} (relative < 1e-9 for D > 0.01); paired rows agree; symmetric, <= 1; pairs sharing < min_overlap pixels are +inf"


# ------------------------------------------------------------------ the synthetic suites and the noise
def test_clean_suites():
    """Suites are finite, positive, normalised, classed by what reaches the band, deterministic per seed, and an ELG shows [OII] at its redshifted wavelength."""
    sky = CleanSky("cpu", seed=3)
    s = sky.suite(60, 0.4, 0.6)
    assert s["clean"].shape == (60, N_PIX) and torch.isfinite(s["clean"]).all() and (s["clean"] > 0).all()
    assert set(s["cls"].tolist()) <= {0, 1} and 0.4 <= float(s["z"].min()) and float(s["z"].max()) <= 0.6                    # no QSO below z 0.7
    q = CleanSky("cpu", seed=3).suite(30, 1.8, 2.6)
    assert set(q["cls"].tolist()) == {2}
    again = CleanSky("cpu", seed=3).suite(60, 0.4, 0.6)
    assert torch.equal(s["clean"], again["clean"])
    wave = wave_grid("cpu")
    hit = n = 0
    for i in torch.where(s["cls"] == 0)[0].tolist():
        lam = 3727.5 * (1 + float(s["z"][i]))
        peak = s["clean"][i][(wave - lam).abs() < 4].max(); cont = s["clean"][i][((wave - lam).abs() > 40) & ((wave - lam).abs() < 90)].median()
        n += 1; hit += int(peak > 1.15 * cont)
    assert n >= 10 and hit / n > 0.6, (hit, n)
    return f"60 clean spectra: finite, positive, classes {sorted(set(s['cls'].tolist()))} for z 0.4-0.6 (QSO-only above 1.8), deterministic; [OII] visible in {hit}/{n} ELGs"


def test_noise_scaling_hits_the_snr():
    """sigma is scaled so that the median per-pixel S/N of the clean signal is the target, with a masked donor pattern too; the realised S/N of the noisy data follows; the clean baseline has no noise."""
    s = CleanSky("cpu", seed=1).suite(24, 0.3, 0.5)
    prof, bad = fidelity.analytic_profile("cpu")
    g = torch.Generator().manual_seed(0)
    nz = fidelity.Noiser(s["clean"], prof, bad, g, use_masks=False)
    got = {}
    for target in (0.7, 4.0, 32.0):
        f, iv, m = nz.arrays(target)
        got[target] = nz.realised_snr(f, iv)
        assert abs(got[target] / target - 1) < 0.10, (target, got[target])                          # the median of flux * sqrt(ivar) of NOISY data sits a few % below the clean-signal S/N at S/N < 4
        sn = object_snr(f.numpy(), iv.numpy(), m.numpy())
        assert abs(np.median(sn) / target - 1) < 0.10 and abs(np.median(sn) / got[target] - 1) < 0.03     # the survey's object_snr is the simulation's definition
    f0, iv0, m0 = nz.arrays(fidelity.S_REF, noisy=False)
    assert torch.equal(f0, s["clean"]) and (iv0 > 0).all()
    badp = torch.zeros(1, N_PIX, dtype=torch.bool); badp[0, 500:900] = True
    nzm = fidelity.Noiser(s["clean"], prof, badp, torch.Generator().manual_seed(1), use_masks=True)
    f, iv, m = nzm.arrays(4.0)
    assert (iv[:, 500:900] == 0).all() and (m[:, 500:900] == 1).all() and (iv[:, :500] > 0).all()
    return "median per-pixel S/N of the noisy data " + ", ".join(f"{t:g} -> {v:.2f}" for t, v in got.items()) + "; masks zero the ivar; the clean baseline is noise-free"


def test_donor_profiles():
    d = TMP / "donor"; d.mkdir()
    rng = np.random.default_rng(0)
    iv = rng.uniform(0.5, 2.0, (30, N_PIX)).astype(np.float32); mk = np.zeros((30, N_PIX), np.int32); mk[:, 100:200] = 1; mk[:6, :] = 1                  # 6 fully masked donors are rejected
    np.save(d / "ivar.npy", iv); np.save(d / "mask.npy", mk)
    prof, bad = fidelity.donor_profiles(d, n=10, seed=0)
    assert prof.shape == (10, N_PIX) and bad[:, 100:200].all() and not bad[:, 300:].any()
    assert all(abs(float(np.median(p[~b.numpy()])) - 1.0) < 1e-4 for p, b in zip(prof.numpy(), bad))
    return "relative noise profiles (median 1 over the good pixels) and masks of real spectra; unusable donors skipped"


# ------------------------------------------------------------------ stress, curves, projection
def test_stress_metrics():
    rng = np.random.default_rng(0)
    n = 30
    A = rng.uniform(0.1, 0.5, (n, n)); D0 = (A + A.T) / 2; iu = np.triu_indices(n, 1)
    m = fidelity.stress_metrics(D0, D0, iu)
    assert m["stress"] == 0 and abs(m["spearman"] - 1) < 1e-12 and m["bias_rel"] == 0 and m["n_pairs"] == n * (n - 1) // 2
    m = fidelity.stress_metrics(1.1 * D0, D0, iu)
    assert abs(m["stress"] - 0.1) < 1e-12 and abs(m["bias_rel"] - 0.1) < 1e-12 and abs(m["spearman"] - 1) < 1e-12                       # a common scale error: stress and bias, the order is intact
    P = D0.copy(); P[iu] = rng.permutation(D0[iu])
    assert fidelity.stress_metrics(P, D0, iu)["spearman"] < 0.2
    Dn = D0.copy(); Dn[0, 1] = np.inf
    assert fidelity.stress_metrics(Dn, D0, iu)["n_pairs"] == n * (n - 1) // 2 - 1                                                       # non-finite pairs are left out
    return "stress-1 / bias / Spearman: identity -> 0 / 0 / 1, a 10% scale error -> 0.1 / 0.1 / 1, shuffled distances -> rank correlation ~ 0"


def test_fidelity_band_curve():
    """A small band on CPU: the stress grows and the rank correlation falls as the S/N drops, the confidence interval brackets the median, and a nearly noise-free copy reproduces the clean target."""
    rm = RestMetric("cpu")
    suite = CleanSky("cpu", seed=2).suite(24, 0.4, 0.6)
    prof, bad = fidelity.analytic_profile("cpu")
    band = fidelity.fidelity_band(rm, suite, prof, bad, (1e4, 32.0, 8.0, 2.0), reps=3, g=torch.Generator().manual_seed(0), log=lambda *a: None)
    r = {x["snr"]: x for x in band["rows"]}
    assert r[1e4]["stress"][2] < 1e-3 and r[1e4]["spearman"][2] > 0.999, r[1e4]
    st = [r[s]["stress"][2] for s in (32.0, 8.0, 2.0)]
    assert st[0] < st[1] < st[2] and r[32.0]["spearman"][2] > r[2.0]["spearman"][2] and r[2.0]["bias_rel"][2] > r[32.0]["bias_rel"][2] > 0, st
    assert all(x["stress"][0] <= x["stress"][2] <= x["stress"][4] for x in band["rows"]) and band["n_pairs"] == 24 * 23 // 2
    _CACHE["band"] = band
    return f"stress {st[0]:.3f} / {st[1]:.3f} / {st[2]:.3f} at S/N 32 / 8 / 2 (a 1e4 S/N copy: {r[1e4]['stress'][2]:.1e}); bias and rank correlation move the right way"


def make_curve_rows(snrs=(0.5, 1, 2, 4, 8, 16, 32), stress=(1.7, 1.3, 0.92, 0.6, 0.38, 0.22, 0.13)):
    return [{"snr": s, "snr_realised": s, "stress": [v * 0.95, v * 0.98, v, v * 1.02, v * 1.05], "bias_rel": [0, 0, v, 0, 0], "spearman": [0.9, 0.9, 0.95, 0.9, 0.9]} for s, v in zip(snrs, stress)]


def test_curve_object():
    rows = make_curve_rows()
    cv = fidelity.FidelityCurve(rows)
    assert abs(float(cv.stress(4.0)) - 0.6) < 1e-12 and abs(float(cv.stress(4.0 * np.sqrt(2.0))) - 0.49) < 1e-9                      # linear in log S/N between the grid points (midpoint of 0.6 and 0.38)
    assert cv.stress(1e-3) == cv.stress(0.5) and cv.stress(1e3) == cv.stress(32.0)                                                      # clamped
    assert list(cv.outside(np.array([0.1, 1.0, 100.0]))) == [True, False, True]
    assert cv.stress(4.0, "p2.5") < cv.stress(4.0) < cv.stress(4.0, "p97.5")
    s = cv.snr_for_stress(0.5)
    assert 4.0 < s < 8.0 and abs(float(cv.stress(s)) - 0.5) < 1e-6 and cv.snr_for_stress(0.05) is None and cv.snr_for_stress(5.0) == 0.5
    assert cv.snr_for_stress(0.5, "p97.5") > s                                                                                         # the pessimistic bound needs a brighter cut
    bumpy = fidelity.FidelityCurve(make_curve_rows(stress=(1.7, 1.3, 0.92, 0.5, 0.6, 0.22, 0.13)))
    assert np.all(np.diff(bumpy.q["p50"]) <= 0)                                                                                          # a noisy bump is removed so the curve can be inverted
    curves = [(0.0, 0.5, cv), (0.5, 1.0, bumpy)]
    assert fidelity.pick_curve(curves, 0.1, 0.3)[0] == (0.0, 0.5) and fidelity.pick_curve(curves, 0.6, 0.9)[0] == (0.5, 1.0) and fidelity.pick_curve(curves, 3.0, 4.0)[0] == (0.5, 1.0)
    return "log-S/N interpolation, clamping, outside flags, confidence bounds, inverse (and its pessimistic version), monotone repair, band selection"


def test_projection_table():
    cv = fidelity.FidelityCurve(make_curve_rows())
    fid_curves = [(0.0, 1.0, cv), (1.0, 2.0, cv)]
    z = np.array([0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 3.0, -1.0]); snr = np.array([1.0, 2.0, 4.0, 8.0, 0.5, 0.5, 4.0, 100.0, 4.0, 4.0]); gf = np.ones(10)
    table, needs, stats = dr2_projection.project(fid_curves, z, snr, gf, [0.0, 1.0, 2.0], cuts=(0.0, 2.0, 4.0), stress_targets=(0.5, 0.05))
    row = {(r["z_lo"], r["z_hi"], r["cut"]): r for r in table}
    r = row[(0.0, 1.0, 0.0)]
    assert r["n_bin"] == 4 and r["n_kept"] == 4 and r["completeness"] == 1.0 and abs(r["snr_median_kept"] - 3.0) < 1e-12 and abs(r["stress"] - float(cv.stress(3.0))) < 1e-12
    assert abs(r["fidelity"] - (1 - r["stress"])) < 1e-12 and r["stress_ci_lo"] < r["stress"] < r["stress_ci_hi"] and r["stress_pop_lo"] < r["stress_pop_hi"]
    r = row[(0.0, 1.0, 4.0)]
    assert r["n_kept"] == 2 and r["completeness"] == 0.5 and abs(r["snr_median_kept"] - 6.0) < 1e-12 and r["stress"] < row[(0.0, 1.0, 0.0)]["stress"]    # a harder cut: fewer objects, lower stress
    r = row[(1.0, 2.0, 0.0)]
    assert r["n_kept"] == 4 and abs(r["frac_outside"] - 0.25) < 1e-12                                                                 # the S/N 100 object is outside the simulated range
    r = row[(0.0, 2.0, 4.0)]                                                                                                           # pooled: 6 valid objects in the bins (z = 3 and -1 are outside)
    assert r["n_bin"] == 8 and r["n_kept"] == 4 and abs(r["completeness"] - 0.5) < 1e-12
    n0 = needs[0]
    assert n0["stress_target"] == 0.5 and 4.0 < n0["snr_needed"] < 8.0 and abs(n0["completeness"] - 0.25) < 1e-12 and needs[1]["snr_needed"] is None and needs[1]["completeness"] is None
    assert stats[0]["n"] == 4 and stats[0]["snr_quantiles"]["50"] == 3.0
    return "completeness / kept median S/N / projected stress with its intervals and clamped fraction, pooled rows, S/N cut needed (and 'never' for an unreachable target)"


def make_survey(path, n=80, seed=0):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.05, 1.45, n)
    s = np.exp(rng.uniform(np.log(0.5), np.log(12.0), n))
    flux = (10.0 + rng.standard_normal((n, N_PIX)) * (10.0 / s)[:, None]).astype(np.float32)
    ivar = np.tile((s / 10.0) ** 2, (N_PIX, 1)).T.astype(np.float32)
    mask = np.zeros((n, N_PIX), np.int32); mask[:, 100:140] = 1
    np.save(path / "flux.npy", flux); np.save(path / "ivar.npy", ivar); np.save(path / "mask.npy", mask); np.save(path / "z.npy", z)
    return z, s


def test_survey_snr_matches_object_snr_and_subset():
    d = TMP / "survey"
    z, s = make_survey(d, n=40)
    rows, zz, snr, gf = dr2_projection.survey_snr(d, chunk=7, log=lambda *a: None)
    ref = object_snr(np.load(d / "flux.npy"), np.load(d / "ivar.npy"), np.load(d / "mask.npy"))
    assert len(rows) == 40 and np.allclose(snr, ref) and np.allclose(zz, z) and np.allclose(gf, 1 - 40 / N_PIX, atol=1e-6)
    assert np.all(np.abs(snr / s - 1) < 0.15)
    rows2, z2, s2, _ = dr2_projection.survey_snr(d, chunk=4, max_rows=9, seed=1, log=lambda *a: None)
    assert len(rows2) == 9 and np.all(np.diff(rows2) > 0) and np.allclose(s2, ref[rows2]) and np.allclose(z2, z[rows2])
    return "chunked streaming pass == object_snr (and tracks the injected S/N); a random subset keeps row order and matches"


def test_end_to_end_and_figures():
    """Simulation (2 bands, analytic and donor noise) -> fidelity.json -> projection on a small survey -> both figures; --resume reuses finished bands."""
    out = TMP / "fid"
    common = ["--out", str(out), "--z-edges", "0.1", "0.4", "0.9", "--n-suite", "24", "--reps", "2", "--snr", "32", "8", "2", "--device", "cpu"]
    res = fidelity.main(common)
    assert len(res["bands"]) == 2 and (out / "fidelity.json").exists() and (out / "fidelity_curves.csv").exists() and (out / "fidelity_vs_snr.png").stat().st_size > 20000
    st = [r["stress"][2] for r in res["bands"][0]["rows"]]
    assert st[0] < st[1] < st[2], st
    res2 = fidelity.main(common + ["--resume"])
    assert res2["bands"][0]["rows"] == res["bands"][0]["rows"]
    donors = TMP / "survey_noise"; make_survey(donors, n=30, seed=4)
    res3 = fidelity.main(["--out", str(TMP / "fid_donor"), "--z-edges", "0.1", "0.4", "--n-suite", "16", "--reps", "2", "--snr", "16", "2", "--device", "cpu", "--noise-donors", str(donors),
                          "--n-donors", "8", "--use-masks", "--no-plots"])
    assert res3["bands"][0]["rows"][0]["stress"][2] < res3["bands"][0]["rows"][1]["stress"][2]
    survey = TMP / "survey_e2e"; make_survey(survey, n=120, seed=5)
    pj = dr2_projection.main(["--data-dir", str(survey), "--fidelity", str(out / "fidelity.json"), "--out", str(TMP / "proj"), "--cuts", "0", "2", "4", "--stress-targets", "0.3", "0.6"])
    assert (TMP / "proj" / "projection.csv").exists() and (TMP / "proj" / "snr_cut_needed.csv").exists() and (TMP / "proj" / "projection_vs_z.png").stat().st_size > 20000
    pooled = [r for r in pj["table"] if r["z_lo"] == 0.1 and r["z_hi"] == 0.9]
    comp = [r["completeness"] for r in pooled]
    assert comp == sorted(comp, reverse=True) and comp[0] > 0.95 and pooled[2]["stress"] < pooled[0]["stress"]
    from DESIFlow.metric import plots
    many = {"bands": [dict(res["bands"][i % 2], z_lo=0.3 * i, z_hi=0.3 * (i + 1)) for i in range(8)], "config": res["config"]}
    assert plots.fig_fidelity(many, TMP / "many.png").stat().st_size > 20000                    # more than five bands: continuous ramp and colour bar
    return f"2 bands x 3 S/N x 2 realisations; stress {st[0]:.3f} < {st[1]:.3f} < {st[2]:.3f}; donor noise + masks ok; projection completeness {[round(c, 2) for c in comp]}; figures written; --resume ok"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
