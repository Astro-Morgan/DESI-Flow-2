"""Effect of the instrument-coefficient prior on the joint legacy + instrument + sky fit (clean galaxies, non-negative physical model): the old diagonal prior (sigma from the first 3000
hold-out SKY rows, backup-dominated), a diagonal zero-mean Gaussian from the MAD of all SKY rows, and the empirical heavy-tailed prior (Student-t per observing program, instrument_prior.py).
Reports the hidden-pixel gain of the instrument terms, the instrument energy share, the size of the fitted coefficients in prior units and the 4000-5000 A split of the faint galaxy of the demo."""
import argparse
import os

import numpy as np

from DESIFlow.templates import joint_fit_demo as jd


def programs(cache, cache_rows):
    idx = np.load(os.path.join(cache, "index", "tgt_rows.npy")); gs = np.load(os.path.join(cache, "index", "group_start.npy")); prog = np.load(os.path.join(cache, "index", "group_program.npy"))
    return np.array([str(prog[g]).strip() for g in np.searchsorted(gs, idx[cache_rows], side="right") - 1])


def run(root, phys_dir, cache, legacy, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fid = jd.Fiducial(root, phys_dir, legacy); F, W, Z, SNR, CR, show, pop, con, RR = jd.load_objects(phys_dir, cache); pr = programs(cache, CR)
    zt = np.load(os.path.join(root, "instrument_prior_t.npz")); sig_all = zt["sig_all"]
    P = {q: (zt[f"mu_{q}"], np.linalg.inv(zt[f"S_{q}"]), float(zt[f"nu_{q}"])) for q in ("dark", "bright", "backup") if f"mu_{q}" in zt.files}
    gauss = (np.zeros(23), np.diag(1 / sig_all ** 2), np.inf)
    arms = {"old diagonal (first 3000 rows)": lambda q: None, "diagonal Gaussian, MAD of all rows": lambda q: gauss, "Student-t per program (empirical)": lambda q: P.get(q, P["dark"])}
    print("programs of the 300 galaxies:", {q: int((pr[pop] == q).sum()) for q in set(pr[pop])})
    res = {a: dict(gain=[], share=[], z=[], hid_all=[]) for a in arms}; base_h = []
    from DESIFlow.data.target_nuisance_test import span_mask
    for i in pop:
        f, w, z = np.asarray(F[i], np.float64), np.asarray(W[i], np.float64), float(Z[i])
        if (w > 0).sum() < 4000: continue
        hid = span_mask(jd.L, np.random.default_rng(int(i))); sel = hid & (w > 0)
        h0 = jd.fit_object(fid, z, f, w, hidden=hid, parts_list=(("phys",),))[("phys",)]["model"]; base_h.append(float((w * (f - h0) ** 2)[sel].sum() / sel.sum()))
        for a, fn in arms.items():
            ip = fn(pr[i]); fh = jd.fit_object(fid, z, f, w, hidden=hid, inst_prior=ip, parts_list=(("phys", "inst", "sky"),))[("phys", "inst", "sky")]
            res[a]["hid_all"].append(float((w * (f - fh["model"]) ** 2)[sel].sum() / sel.sum()))
            ff = jd.fit_object(fid, z, f, w, inst_prior=ip, parts_list=(("phys", "inst", "sky"),))[("phys", "inst", "sky")]
            E = {c: float((w * ff["comp"][c] ** 2).sum()) for c in ("phys", "inst", "sky")}; res[a]["share"].append(E["inst"] / sum(E.values()))
            ii = ff["tag"] == "inst"; mu, Si, nu = (ip if ip is not None else (np.zeros(23), np.diag(1 / fid.prior_inst ** 2), np.inf)); r = ff["theta"][ii] - mu; res[a]["z"].append(float(np.sqrt(r @ Si @ r / 23)))
    base_h = np.array(base_h); out = {}
    for a in arms:
        dd = np.array(res[a]["hid_all"]) - base_h; out[a] = (dd.mean(), dd.std() / np.sqrt(len(dd)), np.median(res[a]["share"]) * 100, np.mean(res[a]["share"]) * 100, np.median(res[a]["z"]), (dd < 0).mean())
        print(f"{a:40s} hidden chi2/px paired vs physical-only {dd.mean():+.4f} +- {dd.std() / np.sqrt(len(dd)):.4f} ({100 * (dd < 0).mean():.0f}% improve) | instrument share median {np.median(res[a]['share']) * 100:.3f}% mean {np.mean(res[a]['share']) * 100:.2f}% | root-mean Mahalanobis per dof, median {np.median(res[a]['z']):.2f}")
    # the faint galaxy of the demo (object 5 of the showcase) and the contaminant
    print("showcase (z, S/N, program): ", [(round(float(Z[i]), 3), round(float(SNR[i]), 1), pr[i]) for i in show])
    n5 = show[5]; f, w, z = np.asarray(F[n5], np.float64), np.asarray(W[n5], np.float64), float(Z[n5]); band = (jd.LAM >= 4000) & (jd.LAM < 5000)
    for a, fn in arms.items():
        ff = jd.fit_object(fid, z, f, w, inst_prior=fn(pr[n5]), parts_list=(("phys", "inst", "sky"),))[("phys", "inst", "sky")]
        print(f"object 5 [{a}]: 4000-5000 A physical {ff['comp']['phys'][band].mean():.2f}  instrument {ff['comp']['inst'][band].mean():.2f}  chi2/px {jd.chi2(f, ff['model'], w)[0] / (w > 0).sum():.3f}")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5)); names = list(arms); cols = ["gray", "tab:blue", "tab:green"]
    ax[0].bar(range(3), [out[a][0] for a in names], yerr=[out[a][1] for a in names], color=cols); ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["old\ndiagonal", "diagonal\nMAD all rows", "Student-t\nper program"]); ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("paired change of the hidden-pixel chi2/px vs physical-only (negative = better)"); ax[0].set_title("what the instrument terms buy, by prior (300 clean galaxies)", fontsize=9)
    ax[1].boxplot([np.array(res[a]["share"]) * 100 for a in names], tick_labels=["old", "MAD all", "t per program"], showfliers=False); ax[1].set_yscale("log"); ax[1].set_ylabel("instrument share of the model energy [percent]")
    ax[2].boxplot([res[a]["z"] for a in names], tick_labels=["old", "MAD all", "t per program"], showfliers=False); ax[2].set_ylabel("rms Mahalanobis distance of the fitted coefficients per dof (in prior units)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "instrument_prior_effect.png"), dpi=120); np.savez(os.path.join(cachedir, "instrument_prior_effect.npz"), **{f"{k}_{a[:12]}": np.array(v) for a in arms for k, v in res[a].items()}, base_hidden=base_h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True); ap.add_argument("--phys", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--legacy", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    a = ap.parse_args(); run(a.root, a.phys, a.cache, a.legacy, a.fig, a.figcache)
