"""
Project the simulated fidelity curve of the metric (`metric.fidelity`) onto a real survey: DR2 fidelity and completeness as a function of redshift at different S/N cuts.

For EVERY spectrum of a {flux, ivar, mask, z}.npy set the median per-pixel S/N (median of flux * sqrt(ivar) over the good pixels, `training.data.object_snr`) and the good-pixel fraction are
measured in one streaming pass (cached in snr_all.npz). The metric itself is never run on the survey: the survey enters only through its S/N statistics. Per redshift bin and S/N cut c:

  COMPLETENESS   n_kept / n_bin with kept = {S/N >= c} (and good-pixel fraction >= --min-good-frac).
  FIDELITY       the simulated stress curve of the matching redshift band evaluated at the MEDIAN S/N of the kept objects, with the band's noise-realisation confidence interval at that S/N;
                 the spread of the kept population (stress at its 16th and 84th percentile S/N) and the per-object expectation (mean of the curve over the kept objects) are tabulated too.
                 fidelity = 1 - stress. S/N outside the simulated range is clamped to its ends and counted (frac_outside).
  S/N CUT NEEDED the faintest S/N at which the median stress curve reaches each target (--stress-targets), and the completeness of the survey at that cut.

usage: python -m DESIFlow.metric.dr2_projection --data-dir $SCRATCH/V2_Real/pretraining_data --fidelity $SCRATCH/metric_snr/fidelity/fidelity.json --out $SCRATCH/metric_snr/dr2
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from DESIFlow.metric.fidelity import load_curves, pick_curve
from DESIFlow.training.data import sanitize, object_snr

CUTS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def survey_snr(data_dir, chunk=4096, max_rows=None, seed=0, log=print):
    """Streams the memory maps once. Returns rows (indices into the set), z, snr (median per-pixel S/N), gfrac (good-pixel fraction)."""
    d = Path(data_dir)
    flux, ivar, mask = (np.load(d / f"{n}.npy", mmap_mode="r") for n in ("flux", "ivar", "mask"))
    z_all = np.load(d / "z.npy").astype(np.float64).reshape(-1)
    n = len(z_all)
    rows = np.arange(n) if max_rows is None or max_rows >= n else np.sort(np.random.default_rng(seed).choice(n, max_rows, replace=False))
    contiguous = len(rows) == n
    snr, gfrac = np.zeros(len(rows)), np.zeros(len(rows))
    t0 = time.time()
    for a in range(0, len(rows), chunk):
        b = min(len(rows), a + chunk)
        sl = slice(a, b) if contiguous else rows[a:b]
        f, iv, m = sanitize(flux[sl], ivar[sl], mask[sl])
        snr[a:b] = object_snr(f, iv, m)
        gfrac[a:b] = ((m == 0) & (iv > 0)).mean(1)
        if (a // chunk) % 20 == 0:
            log(f"  S/N pass {b}/{len(rows)} rows  [{time.time() - t0:.0f} s]")
    return rows, z_all[rows], snr, gfrac


def project(curves, z, snr, gfrac, edges, cuts=CUTS, min_good_frac=0.0, stress_targets=(0.2, 0.4, 0.6)):
    """-> (rows per (bin, cut) plus pooled all-z rows, rows of the S/N cut each stress target needs, per-bin S/N statistics)."""
    valid = np.isfinite(z) & (z >= 0) & np.isfinite(snr) & (gfrac > 0)                    # usable redshift and at least one good pixel
    table, needs, stats = [], [], []
    members = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = valid & (z >= lo) & (z < hi)
        members.append(sel)
        (blo, bhi), cv = pick_curve(curves, lo, hi)
        s_bin = snr[sel]
        stats.append({"z_lo": float(lo), "z_hi": float(hi), "n": int(sel.sum()), "curve_band": [blo, bhi],
                      "snr_quantiles": {str(q): float(np.percentile(s_bin, q)) for q in (5, 10, 25, 50, 75, 90, 95)} if s_bin.size else {}})
        for c in cuts:
            keep = sel & ((snr >= c) if c > 0 else True) & (gfrac >= min_good_frac)                # cut 0 = no S/N cut (also keeps the objects whose median S/N is negative)
            sk = snr[keep]
            row = {"z_lo": float(lo), "z_hi": float(hi), "cut": float(c), "n_bin": int(sel.sum()), "n_kept": int(keep.sum()), "completeness": float(keep.sum() / max(sel.sum(), 1)),
                   "curve_band_lo": blo, "curve_band_hi": bhi}
            if sk.size:
                m = float(np.median(sk))
                row.update(snr_median_kept=m, snr_p16_kept=float(np.percentile(sk, 16)), snr_p84_kept=float(np.percentile(sk, 84)),
                           stress=float(cv.stress(m)), stress_ci_lo=float(cv.stress(m, "p2.5")), stress_ci_hi=float(cv.stress(m, "p97.5")),
                           stress_pop_lo=float(cv.stress(np.percentile(sk, 84))), stress_pop_hi=float(cv.stress(np.percentile(sk, 16))), stress_mean_obj=float(cv.stress(sk).mean()),
                           frac_outside=float(cv.outside(sk).mean()))
                row["fidelity"] = 1.0 - row["stress"]
            else:
                row.update({k: float("nan") for k in ("snr_median_kept", "snr_p16_kept", "snr_p84_kept", "stress", "stress_ci_lo", "stress_ci_hi", "stress_pop_lo", "stress_pop_hi",
                                                      "stress_mean_obj", "frac_outside", "fidelity")})
            table.append(row)
        for x in stress_targets:
            s_need = cv.snr_for_stress(x)
            s_cons = cv.snr_for_stress(x, "p97.5")
            comp = lambda s: float((sel & (snr >= s) & (gfrac >= min_good_frac)).sum() / max(sel.sum(), 1)) if s is not None else None
            needs.append({"z_lo": float(lo), "z_hi": float(hi), "stress_target": float(x), "snr_needed": s_need, "completeness": comp(s_need),
                          "snr_needed_ci_hi": s_cons, "completeness_ci_lo": comp(s_cons)})
    allsel = np.any(members, 0)
    nb = len(edges) - 1
    for ci, c in enumerate(cuts):
        keep = allsel & ((snr >= c) if c > 0 else True) & (gfrac >= min_good_frac)
        rs = [table[bi * len(cuts) + ci] for bi in range(nb)]
        w = np.array([r["n_kept"] if np.isfinite(r["stress"]) else 0 for r in rs], float)
        row = {"z_lo": float(edges[0]), "z_hi": float(edges[-1]), "cut": float(c), "n_bin": int(allsel.sum()), "n_kept": int(keep.sum()), "completeness": float(keep.sum() / max(allsel.sum(), 1)),
               "curve_band_lo": float("nan"), "curve_band_hi": float("nan"), "snr_median_kept": float(np.median(snr[keep])) if keep.any() else float("nan")}
        for k in ("snr_p16_kept", "snr_p84_kept", "stress", "stress_ci_lo", "stress_ci_hi", "stress_pop_lo", "stress_pop_hi", "stress_mean_obj", "frac_outside", "fidelity"):
            row[k] = float(np.sum([wi * r[k] for wi, r in zip(w, rs) if wi > 0]) / w.sum()) if w.sum() > 0 else float("nan")
        table.append(row)
    return table, needs, stats


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and not np.isfinite(v)) or v is None else v) for k, v in r.items()})


def run(a, log=print):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fid = json.load(open(a.fidelity))
    curves = load_curves(fid)
    edges = list(a.z_edges) if a.z_edges else sorted({b["z_lo"] for b in fid["bands"]} | {b["z_hi"] for b in fid["bands"]})
    log(f"fidelity curves: {len(curves)} bands from {a.fidelity}; projection bins {edges}")
    cache = out / "snr_all.npz"
    if cache.exists() and not a.recompute_snr:
        c = np.load(cache); rows, z, snr, gfrac = c["rows"], c["z"], c["snr"], c["gfrac"]
        log(f"S/N of {len(rows)} objects from {cache}")
    else:
        log(f"S/N pass over {a.data_dir}")
        rows, z, snr, gfrac = survey_snr(a.data_dir, max_rows=a.max_rows, seed=a.seed, log=log)
        np.savez(cache, rows=rows, z=z, snr=snr, gfrac=gfrac)
    valid = np.isfinite(z) & (z >= 0) & np.isfinite(snr) & (gfrac > 0)
    log(f"{len(z)} objects ({int((~valid).sum())} unusable: no redshift or no good pixel); median S/N {np.median(snr[valid]):.2f} (10-90%: {np.percentile(snr[valid], 10):.2f}-{np.percentile(snr[valid], 90):.2f})")
    table, needs, stats = project(curves, z, snr, gfrac, edges, cuts=a.cuts, min_good_frac=a.min_good_frac, stress_targets=a.stress_targets)
    res = {"config": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(a).items()}, "n_objects": int(len(z)), "n_valid": int(valid.sum()),
           "fidelity_config": fid.get("config"), "bins": stats, "table": table, "needs": needs}
    json.dump(_clean(res), open(out / "projection.json", "w"))
    write_csv(out / "projection.csv", _clean(table)); write_csv(out / "snr_cut_needed.csv", _clean(needs))
    log(f"saved {out / 'projection.json'}, projection.csv, snr_cut_needed.csv")
    for r in table:
        if r["cut"] in (0.0, 2.0, 4.0) and r["z_lo"] == edges[0] and r["z_hi"] == edges[-1]:
            log(f"  all z, S/N >= {r['cut']:g}: completeness {r['completeness']:.3f}, kept median S/N {r['snr_median_kept']:.2f}, projected stress {r['stress']:.3f}")
    if a.plots:
        try:
            from DESIFlow.metric import plots
            log(f"figure {plots.fig_projection(res, out / 'projection_vs_z.png')}")
        except ImportError as e:
            log(f"plots skipped ({e})")
    return res


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("usage:")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="directory with flux.npy ivar.npy mask.npy z.npy")
    p.add_argument("--fidelity", required=True, help="fidelity.json of metric.fidelity")
    p.add_argument("--out", required=True)
    p.add_argument("--z-edges", type=float, nargs="+", default=None, help="projection bins (default: the bands of the fidelity file)")
    p.add_argument("--cuts", type=float, nargs="+", default=list(CUTS), help="median per-pixel S/N cuts (0 = no cut)")
    p.add_argument("--stress-targets", type=float, nargs="+", default=[0.2, 0.4, 0.6], help="stress levels for which the needed S/N cut is reported")
    p.add_argument("--min-good-frac", type=float, default=0.0, help="completeness also requires this good-pixel fraction")
    p.add_argument("--max-rows", type=int, default=None, help="random subset of the set (testing)")
    p.add_argument("--recompute-snr", action="store_true")
    p.add_argument("--no-plots", dest="plots", action="store_false")
    p.add_argument("--seed", type=int, default=2026)
    return p


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
