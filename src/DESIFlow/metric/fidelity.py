"""
Fidelity of the target metric to its CLEAN value as a function of median per-pixel S/N, on synthetic spectra (the simulation half of the DR2 projection, `metric.dr2_projection`).

Per redshift band (the metric lives on a z-banded interval graph):
  1. A suite of CLEAN physics-linked synthetic spectra (`metric.synth`: ELG / LRG / QSO with redshifts inside the band) is the baseline. The metric between all pairs of the clean suite
     -- redshift-aligned (integer log-pixel shift to the rest frame), NW-smoothed (77 km/s, ivar weights), weighted-starlet decomposed (J = 8, s1 dropped, +/- masses), compositional
     Hellinger over the pair's mutually good rest-frame pixels (window-renormalised) -- is the CLEAN TARGET distance matrix D0.
  2. The suite is degraded to a median per-pixel S/N s (Gaussian noise with a relative noise profile, scaled so that median(flux * sqrt(ivar)) over the good pixels equals s), the metric is
     recomputed among the degraded objects (all at the same S/N, independent noise), and its STRESS against the clean target is measured:
         stress = sqrt( sum_{i<j} (D_ij - D0_ij)^2 / sum_{i<j} D0_ij^2 )          (Kruskal stress-1; 0 = the clean geometry, 1 = errors as large as the distances themselves)
     together with the bias of the distances (mean(D - D0) / mean(D0)) and the Spearman rank correlation of D with D0 (does the ORDER of the distances survive).
  3. Step 2 is repeated for many independent noise realisations at each S/N: the spread gives the confidence interval of the fidelity curve; then the S/N is lowered and so on, which gives
     stress (and bias, rank correlation) versus median per-pixel S/N for the band.
`fidelity = 1 - stress` is also tabulated. The noise profile is either a built-in DESI-like analytic one or the relative noise pattern (1/sqrt(ivar), and optionally the mask) of random real
spectra (--noise-donors, e.g. the DR2 pretraining set). The curves are written to fidelity.json / fidelity_curves.csv and drawn to fidelity_vs_snr.png.

Not simulated: stars, the Lyman-alpha forest, object-dependent (Poisson) noise, covariance between neighbouring pixels beyond what the smoothing creates.

usage: python -m DESIFlow.metric.fidelity --out $SCRATCH/metric_snr/fidelity [--noise-donors $SCRATCH/V2_Real/pretraining_data]
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
from DESIFlow.metric.hellinger import RestMetric
from DESIFlow.metric.synth import CleanSky, N_PIX, wave_grid, CLASS_NAMES

Z_EDGES = (0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.5)
SNR_GRID = tuple(round(64.0 * 2.0 ** (-k / 2.0), 3) for k in range(16))                  # 64, 45, 32, 22.6, 16, ... 0.5, 0.35
QUANTILES = (2.5, 16.0, 50.0, 84.0, 97.5)
S_REF = 64.0                                                                              # S/N of the ivar given to the clean baseline (only the relative weights matter)
SKY = (5577.3, 5889.95, 6300.3, 6363.8, 6863.9, 7244.9, 7276.4, 7316.3, 7340.9, 7750.4, 7794.1, 7821.5, 7913.7, 7993.3, 8344.6, 8399.2, 8430.2, 8827.1, 8885.8, 8919.5, 9375.9, 9439.7)


# ------------------------------------------------------------------ noise profiles
def analytic_profile(dev="cpu"):
    """DESI-like relative noise sigma(lambda), median 1: it rises toward both ends of the spectrograph range and has narrow bumps at the strongest sky lines. (N_PIX,) float32."""
    lam = wave_grid(dev).double()
    p = 1.0 + 1.6 * torch.exp(-(lam - 3600.0) / 350.0) + 0.9 * torch.exp(-(9824.0 - lam) / 450.0)
    for i, l0 in enumerate(SKY):
        p = p + (0.8 + 1.2 * ((i * 7) % 5) / 4.0) * torch.exp(-0.5 * ((lam - l0) / 1.5) ** 2)
    return (p / p.median()).float().unsqueeze(0), torch.zeros(1, N_PIX, dtype=torch.bool, device=dev)


def donor_profiles(data_dir, n=300, seed=0, min_good=0.8, dev="cpu"):
    """Relative noise patterns of n random real spectra of a {flux, ivar, mask}.npy set: prof = (1/sqrt(ivar)) / median over the good pixels, clipped to [0.2, 20]; bad = masked / ivar <= 0.
    Spectra with fewer than `min_good` good pixels are not used. -> prof (n, N_PIX) float32, bad (n, N_PIX) bool."""
    d = Path(data_dir)
    ivar, mask = np.load(d / "ivar.npy", mmap_mode="r"), np.load(d / "mask.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(len(ivar), min(len(ivar), 6 * n), replace=False))
    prof, bad = [], []
    for i in cand:
        iv, m = np.nan_to_num(np.asarray(ivar[i], np.float64)), np.asarray(mask[i])
        good = (iv > 0) & (m == 0)
        if good.mean() < min_good:
            continue
        p = np.ones(N_PIX); p[good] = 1.0 / np.sqrt(iv[good]); p /= np.median(p[good])
        prof.append(np.clip(p, 0.2, 20.0)); bad.append(~good)
        if len(prof) >= n:
            break
    if not prof:
        raise ValueError("no usable noise donors")
    return torch.tensor(np.stack(prof), dtype=torch.float32, device=dev), torch.tensor(np.stack(bad), device=dev)


# ------------------------------------------------------------------ stress
def _ranks(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def stress_metrics(D, D0, iu):
    """Stress-1, relative bias and Spearman correlation of the distances D against the clean target D0 over the finite pairs of the upper triangle."""
    d, d0 = D[iu], D0[iu]
    ok = np.isfinite(d) & np.isfinite(d0)
    d, d0 = d[ok], d0[ok]
    rs, r0 = _ranks(d), _ranks(d0)
    return {"stress": float(np.sqrt(((d - d0) ** 2).sum() / (d0 ** 2).sum())), "bias_rel": float((d - d0).mean() / d0.mean()),
            "spearman": float(np.corrcoef(rs, r0)[0, 1]), "n_pairs": int(ok.sum())}


def _q(x):
    return [float(np.percentile(x, q)) for q in QUANTILES]


class Noiser:
    """Noise for one suite: every object gets a random donor profile (and mask); sigma(s) = prof * median_good(clean / prof) / s so that the median per-pixel S/N of the CLEAN signal is s."""

    def __init__(self, clean, prof, bad, g, use_masks):
        n, dev = clean.shape[0], clean.device
        k = torch.randint(0, prof.shape[0], (n,), device=dev, generator=g)
        self.prof, self.bad = prof[k], bad[k] if use_masks else torch.zeros(n, N_PIX, dtype=torch.bool, device=dev)
        self.good = ~self.bad
        self.clean, self.g = clean, g
        self.k = torch.where(self.good, clean / self.prof, torch.full_like(clean, float("nan"))).nanmedian(1).values.unsqueeze(1)       # sigma = prof * k / s

    def arrays(self, s, noisy=True):
        sig = self.prof * self.k / s
        flux = self.clean + (torch.randn(self.clean.shape, device=self.clean.device, generator=self.g) * sig if noisy else 0.0)
        z0 = torch.zeros_like(sig)
        return torch.where(self.good, flux, z0), torch.where(self.good, 1.0 / (sig * sig), z0), self.bad.float()

    def realised_snr(self, flux, ivar):
        sn = torch.where(self.good, flux * ivar.sqrt(), torch.full_like(flux, float("nan")))
        return float(sn.nanmedian(1).values.median())


def fidelity_band(rm, suite, prof, bad, snr_grid, reps, g, use_masks=False, log=print):
    """Clean target and fidelity-vs-S/N rows of one band. `suite`: dict from CleanSky.suite. Returns the band record."""
    clean, z = suite["clean"], suite["z"].double().cpu().numpy()
    nz = Noiser(clean, prof, bad, g, use_masks)
    f0, iv0, m0 = nz.arrays(S_REF, noisy=False)
    C0 = rm.cells(f0, iv0, m0, z)
    D0 = rm.cross(C0, C0).cpu().numpy()
    n = len(z)
    iu = np.triu_indices(n, 1)
    d0 = D0[iu]; fin = np.isfinite(d0)
    nn = np.where(np.isfinite(D0), D0, np.inf) + np.diag(np.full(n, np.inf))
    rows = []
    for s in snr_grid:
        out, real = [], []
        for _ in range(reps):
            f, iv, m = nz.arrays(s)
            C = rm.cells(f, iv, m, z)
            out.append(stress_metrics(rm.cross(C, C).cpu().numpy(), D0, iu)); real.append(nz.realised_snr(f, iv))
        row = {"snr": float(s), "snr_realised": float(np.median(real))}
        for key in ("stress", "bias_rel", "spearman"):
            row[key] = _q([o[key] for o in out])
        row["fidelity"] = [1.0 - v for v in row["stress"]][::-1]
        rows.append(row)
        log(f"    S/N {s:6.2f} (realised {row['snr_realised']:6.2f}): stress {row['stress'][2]:.3f}  [{row['stress'][1]:.3f}, {row['stress'][3]:.3f}]  bias {row['bias_rel'][2]:+.3f}  rank corr {row['spearman'][2]:.3f}")
    cls = suite["cls"].cpu().numpy()
    return {"n_suite": int(n), "class_counts": {nm: int((cls == i).sum()) for i, nm in enumerate(CLASS_NAMES)}, "n_pairs": int(fin.sum()), "d_clean_median": float(np.median(d0[fin])),
            "d_clean_nn_median": float(np.median(nn.min(1)[np.isfinite(nn.min(1))])), "z_median": float(np.median(z)), "rows": rows}


# ------------------------------------------------------------------ the curve object (shared with the DR2 projection)
class FidelityCurve:
    """Stress versus median per-pixel S/N of one band, with its confidence bounds, interpolated linearly in log S/N and clamped to the simulated range.
    The median curve is made non-increasing in S/N (running minimum from the faint end) so that it can be inverted."""

    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["snr_realised"])
        self.s = np.array([r["snr_realised"] for r in rows])
        self.q = {k: np.array([r["stress"][i] for r in rows]) for i, k in enumerate(("p2.5", "p16", "p50", "p84", "p97.5"))}
        for k in self.q:
            self.q[k] = np.minimum.accumulate(self.q[k])
        self.bias = np.array([r["bias_rel"][2] for r in rows]); self.rho = np.array([r["spearman"][2] for r in rows])

    def stress(self, s, which="p50"):
        return np.interp(np.log(np.clip(s, 1e-6, None)), np.log(self.s), self.q[which])

    def outside(self, s):
        s = np.asarray(s)
        return (s < self.s[0]) | (s > self.s[-1])

    def snr_for_stress(self, x, which="p50"):
        """Smallest S/N at which the stress curve (median, or another confidence bound) is <= x (None if even the brightest simulated S/N does not reach it; the faintest S/N if it is already met)."""
        y = self.q[which]
        if y[-1] > x:
            return None
        if y[0] <= x:
            return float(self.s[0])
        return float(np.exp(np.interp(-x, -y + 1e-12 * np.arange(len(y)), np.log(self.s))))


def load_curves(fid):
    """-> list of (z_lo, z_hi, FidelityCurve) from a fidelity.json dict."""
    return [(b["z_lo"], b["z_hi"], FidelityCurve(b["rows"])) for b in fid["bands"] if b.get("rows")]


def pick_curve(curves, z_lo, z_hi):
    """The curve of the band that contains the centre of [z_lo, z_hi] (else the nearest band)."""
    zc = 0.5 * (z_lo + z_hi)
    for lo, hi, c in curves:
        if lo <= zc < hi:
            return (lo, hi), c
    lo, hi, c = min(curves, key=lambda t: abs(0.5 * (t[0] + t[1]) - zc))
    return (lo, hi), c


# ------------------------------------------------------------------ driver
def write_csv(path, bands):
    cols = ["z_lo", "z_hi", "snr", "snr_realised"] + [f"{k}_{p}" for k in ("stress", "bias_rel", "spearman") for p in ("p2.5", "p16", "p50", "p84", "p97.5")] + ["fidelity_p50"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for b in bands:
            for r in b.get("rows", []):
                w.writerow([b["z_lo"], b["z_hi"], r["snr"], r["snr_realised"]] + r["stress"] + r["bias_rel"] + r["spearman"] + [1.0 - r["stress"][2]])


def run(a, log=print):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    if a.noise_donors:
        prof, bad = donor_profiles(a.noise_donors, n=a.n_donors, seed=a.seed, dev=dev)
        log(f"noise profiles: {len(prof)} real donors from {a.noise_donors}" + (" (with their masks)" if a.use_masks else " (profiles only, no masks)"))
    else:
        prof, bad = analytic_profile(dev)
        log("noise profile: built-in DESI-like analytic profile" + (" (masks need --noise-donors)" if a.use_masks else ""))
    rm = RestMetric(device=a.device)
    edges = list(a.z_edges)
    res_path = out / "fidelity.json"
    done = {}
    if a.resume and res_path.exists():
        old = json.load(open(res_path))
        done = {(b["z_lo"], b["z_hi"]): b for b in old["bands"] if b.get("rows")}
        log(f"resuming: {len(done)} bands already done")
    config = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(a).items()}
    res = {"config": config, "snr_grid": list(a.snr), "bands": []}
    t0 = time.time()
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if (lo, hi) in done:
            res["bands"].append(done[(lo, hi)]); continue
        sky = CleanSky(a.device, seed=a.seed + 1000 * bi, classes=tuple(a.classes))
        suite = sky.suite(a.n_suite, lo, hi)
        log(f"[{time.time() - t0:6.0f}s] z {lo:g}-{hi:g}: clean suite of {a.n_suite} ({', '.join(f'{k} {int((suite['cls'] == i).sum())}' for i, k in enumerate(CLASS_NAMES))}); "
            f"{len(a.snr)} S/N values x {a.reps} noise realisations")
        band = fidelity_band(rm, suite, prof, bad, a.snr, a.reps, torch.Generator(device=dev).manual_seed(a.seed + 7 + bi), use_masks=a.use_masks, log=log)
        res["bands"].append({"z_lo": float(lo), "z_hi": float(hi), **band})
        json.dump(res, open(res_path, "w"))
    json.dump(res, open(res_path, "w"))
    write_csv(out / "fidelity_curves.csv", res["bands"])
    log(f"saved {res_path} and fidelity_curves.csv  [{time.time() - t0:.0f} s]")
    if a.plots:
        try:
            from DESIFlow.metric import plots
            log(f"figure {plots.fig_fidelity(res, out / 'fidelity_vs_snr.png')}")
        except ImportError as e:
            log(f"plots skipped ({e})")
    return res


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("usage:")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--z-edges", type=float, nargs="+", default=list(Z_EDGES), help="redshift bands (one clean suite and one curve per band); a single band 0 6.5 gives one pooled curve")
    p.add_argument("--snr", type=float, nargs="+", default=list(SNR_GRID), help="median per-pixel S/N values of the degradation sweep")
    p.add_argument("--n-suite", type=int, default=250, help="clean synthetic spectra per band")
    p.add_argument("--reps", type=int, default=24, help="independent noise realisations per S/N (the confidence interval)")
    p.add_argument("--classes", type=float, nargs=3, default=[0.5, 0.2, 0.3], help="class weights ELG LRG QSO (renormalised over the classes that reach a band)")
    p.add_argument("--noise-donors", default=None, help="a {flux, ivar, mask}.npy directory whose spectra donate relative noise profiles (default: built-in analytic profile)")
    p.add_argument("--n-donors", type=int, default=300)
    p.add_argument("--use-masks", action="store_true", help="also apply the donors' masks (to the clean baseline and the degraded copies alike)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plots", dest="plots", action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
