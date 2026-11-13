"""Figures of the joint legacy + instrument + sky fit demonstration (see joint_fit_demo.py). Every figure has a same-named .npz cache."""
import os

import numpy as np

from DESIFlow.templates import joint_fit_demo as jd

COL = {"phys": "tab:green", "inst": "tab:orange", "sky": "tab:purple"}


def _robust_ylim(a, k=1.6):
    lim = np.nanpercentile(np.abs(a[np.isfinite(a)]), 99) * k
    return -0.2 * lim, lim


def make(root, phys_dir, cache, legacy, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fid, objs, co, res, P, show = jd.run(root, phys_dir, cache, legacy, figdir, cachedir)
    ig = [n for n, o in enumerate(objs) if o["kind"] == "galaxy"]; iq = [n for n, o in enumerate(objs) if o["kind"] == "QSO"]; ic = [n for n, o in enumerate(objs) if o["kind"] == "contaminant"]
    # ---------------------------------------------------------------- per-object figures
    for n, o in enumerate(objs):
        d = co[f"o{n}"]; r = res[n]; g = d["w"] > 0; lam = jd.LAM
        fits = jd.fit_object(fid, o["z"], o["f"], o["w"], parts_list=(("phys", "inst", "sky"),))[("phys", "inst", "sky")]
        zi = np.abs(fits["theta"][fits["tag"] == "inst"]) / fid.prior_inst; zs = np.abs(fits["theta"][fits["tag"] == "sky"]) / fid.prior_sky
        fig, ax = plt.subplots(4, 1, figsize=(18, 15), sharex=True)
        sm = jd.nw(d["f"], d["w"], 6.0); ax[0].plot(lam[g], d["f"][g], color="0.75", lw=0.4, label="raw flux (masked pixels omitted)"); ax[0].plot(lam, sm, color="k", lw=1.0, label="raw flux, NW smooth (sigma 6 px)")
        ax[0].plot(lam, jd.nw(d["phys_only"], np.ones(jd.L), 1.0), color="tab:blue", lw=1.0, label=f"legacy-physical-only fit to the raw flux (chi2/px {r['chi2_per_pix']['phys']:.2f})")
        ax[0].plot(lam, jd.nw(d["joint"], np.ones(jd.L), 1.0), color="tab:red", lw=1.0, label=f"joint model physical + instrument + sky (chi2/px {r['chi2_per_pix']['joint']:.2f})"); ax[0].set_ylim(*_robust_ylim(sm, 1.8)); ax[0].legend(fontsize=8, ncol=2, loc="upper right")
        ax[0].set_ylabel("flux [1e-17 erg/s/cm2/A]")
        for c, nm in (("phys", "legacy physical templates (joint fit)"), ("inst", "instrument templates (b, r, z)"), ("sky", "sky templates")):
            ax[1].plot(lam, d[{"phys": "phys_joint", "inst": "inst", "sky": "sky"}[c]], color=COL[c], lw=0.8, label=f"{nm}: {100 * r['share_of_model_energy'][c]:.2f}% of the model energy ({r['energy_per_pix'][c]:.3g} chi2/px)")
        ax[1].set_yscale("symlog", linthresh=max(np.nanpercentile(np.abs(d["inst"]), 90), 0.05)); ax[1].set_ylabel("template model components [1e-17 erg/s/cm2/A] (symlog)"); ax[1].legend(fontsize=8, loc="upper right")
        cs = jd.nw(d["clean"], d["w"], 6.0); ax[2].plot(lam[g], d["clean"][g], color="0.75", lw=0.4, label="cleaned flux = raw - instrument - sky"); ax[2].plot(lam, cs, color="k", lw=1.0, label="cleaned flux, NW smooth (sigma 6 px)")
        ax[2].plot(lam, jd.nw(d["refit"], np.ones(jd.L), 1.0), color="tab:red", lw=1.0, label=f"physical refit to the cleaned flux (physical model changes by {100 * r['physical_model_change']:.1f}% in the chi2 norm)")
        ax[2].plot(lam, jd.nw(d["phys_only"], np.ones(jd.L), 1.0), color="tab:blue", lw=1.0, ls="--", label="physical-only fit to the raw flux"); ax[2].set_ylim(*_robust_ylim(cs, 1.8)); ax[2].legend(fontsize=8, ncol=2, loc="upper right"); ax[2].set_ylabel("flux [1e-17 erg/s/cm2/A]")
        ax[3].plot(lam, jd.nw(d["f"] - d["phys_only"], d["w"], 6.0), color="tab:blue", lw=0.9, label="raw - physical-only fit (smoothed)"); ax[3].plot(lam, jd.nw(d["f"] - d["joint"], d["w"], 6.0), color="tab:red", lw=0.9, label="raw - joint model (smoothed)")
        ax[3].plot(lam, jd.nw(d["inst"] + d["sky"], np.ones(jd.L), 6.0), color="tab:orange", lw=0.9, ls=":", label="instrument + sky fitted part (smoothed)"); ax[3].axhline(0, color="gray", lw=0.6); ax[3].legend(fontsize=8, ncol=3, loc="upper right")
        lim = np.nanpercentile(np.abs(jd.nw(d["f"] - d["phys_only"], d["w"], 6.0)), 99) * 1.4; ax[3].set_ylim(-lim, lim); ax[3].set_ylabel("residual [1e-17 erg/s/cm2/A]"); ax[3].set_xlabel("observed wavelength [Angstrom]")
        ax[0].set_title(f"{o['kind']} z = {o['z']:.3f}, median S/N {o['snr']:.1f}, Redrock chi2/px {o['rr']:.2f}; chi2/px physical-only {r['chi2_per_pix']['phys']:.2f} -> +instrument {r['chi2_per_pix']['phys+inst']:.2f}, +sky {r['chi2_per_pix']['phys+sky']:.2f}, joint {r['chi2_per_pix']['joint']:.2f}; "
                        f"median |coefficient|/prior sigma: instrument {np.median(zi):.1f}, sky {np.median(zs):.1f}", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"joint_fit_object_{n}.png"), dpi=100); plt.close(fig)
        np.savez(os.path.join(cachedir, f"joint_fit_object_{n}.npz"), wavelength_A=lam, z=o["z"], snr=o["snr"], **{k: v for k, v in d.items() if isinstance(v, np.ndarray)}, inst_zscore=zi, sky_zscore=zs)
    # ---------------------------------------------------------------- energy figure
    fig, ax = plt.subplots(2, 2, figsize=(16, 11)); lab = [f"{o['kind'][:3]} z={o['z']:.2f}\nS/N {o['snr']:.0f}\nRR {o['rr']:.1f}" for o in objs]; x = np.arange(len(objs)); sh = np.array([[r["share_of_model_energy"][c] for c in ("phys", "inst", "sky")] for r in res])
    bot = np.zeros(len(objs))
    for k, c in enumerate(("phys", "inst", "sky")):
        ax[0, 0].bar(x, 100 * sh[:, k], bottom=100 * bot, color=COL[c], label={"phys": "legacy physical", "inst": "instrument", "sky": "sky"}[c]); bot += sh[:, k]
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(lab, fontsize=7); ax[0, 0].set_ylim(80, 100); ax[0, 0].set_ylabel("share of the model energy sum(omega m^2) [percent] (axis from 80)"); ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("energy of each component of the joint fit (components are not orthogonal: cross terms ignored)", fontsize=9)
    gain = np.array([[1 - r["chi2_per_pix"][k] / r["chi2_per_pix"]["phys"] for k in ("phys+inst", "phys+sky", "joint")] for r in res])
    for k, (c, nm) in enumerate(((COL["inst"], "+ instrument"), (COL["sky"], "+ sky"), ("tab:red", "+ both"))):
        ax[0, 1].bar(x + (k - 1) * 0.27, 100 * gain[:, k], 0.27, color=c, label=nm)
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(lab, fontsize=7); ax[0, 1].set_ylabel("reduction of the chi2 of the physical-only fit [percent]"); ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("what the nuisance templates buy in chi2 (in-sample, all pixels)", fontsize=9)
    snr = np.array([p["snr"] for p in P]); psh = np.array([p["shares"] for p in P]) * 100
    ax[1, 0].scatter(snr, psh[:, 1], s=8, color=COL["inst"], label="instrument"); ax[1, 0].scatter(snr, psh[:, 2], s=8, color=COL["sky"], label="sky"); ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("median S/N of the spectrum")
    ax[1, 0].set_ylabel("share of the model energy [percent]"); ax[1, 0].legend(fontsize=8); ax[1, 0].set_title(f"{len(P)} random held-out galaxies", fontsize=9)
    zz = [np.abs(jd.fit_object(fid, o["z"], o["f"], o["w"], parts_list=(("phys", "inst", "sky"),))[("phys", "inst", "sky")]["theta"]) for o in objs]; tg = jd.fit_object(fid, objs[0]["z"], objs[0]["f"], objs[0]["w"], parts_list=(("phys", "inst", "sky"),))[("phys", "inst", "sky")]["tag"]
    ax[1, 1].boxplot([np.log10(z[tg == "inst"] / fid.prior_inst) for z in zz], tick_labels=lab, showfliers=False); ax[1, 1].axhline(0, color="gray", ls=":"); ax[1, 1].tick_params(axis="x", labelsize=7)
    ax[1, 1].set_ylabel("log10( |instrument coefficient| / prior sigma )"); ax[1, 1].set_title("instrument coefficients against their prior (0 = typical SKY-row scatter)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "joint_fit_energy.png"), dpi=120); plt.close(fig)
    np.savez(os.path.join(cachedir, "joint_fit_energy.npz"), objects=np.array(lab), shares=sh, energy_per_pix=np.array([[r["energy_per_pix"][c] for c in ("phys", "inst", "sky")] for r in res]), chi2_gain=gain, pop_snr=snr, pop_shares=psh,
             pop_per_pix=np.array([p["per_pix"] for p in P]), inst_zscore_log10=np.array([np.median(np.log10(z[tg == "inst"] / fid.prior_inst)) for z in zz]))
    # ---------------------------------------------------------------- physical change figure
    fig, ax = plt.subplots(2, 2, figsize=(16, 11)); cache_c = {}
    dm = np.array([p["dmodel"] for p in P]); ax[0, 0].scatter(snr, 100 * dm, s=8); ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("median S/N"); ax[0, 0].set_ylabel("change of the physical model, joint vs physical-only fit [percent of its norm]")
    ax[0, 0].set_title("how much the physical fit moves when instrument + sky are fitted with it (= refit to the cleaned flux)", fontsize=9); cache_c.update(pop_snr=snr, pop_dmodel=dm)
    import collections
    hk = collections.defaultdict(list)
    for p in P:
        for k, v in p["hidden"].items(): hk[k].append(v)
    base = np.array(hk["phys"]); names = [("phys+inst", "+ instrument"), ("phys+sky", "+ sky"), ("phys+inst+sky", "+ both")]
    for k, (key, nm) in enumerate(names):
        dd = np.array(hk[key]) - base; ax[0, 1].bar(k, dd.mean(), yerr=dd.std() / np.sqrt(len(dd)), color=["tab:orange", "tab:purple", "tab:red"][k]); cache_c[f"hidden_diff_{key}"] = dd
    ax[0, 1].set_xticks(range(3)); ax[0, 1].set_xticklabels([n for _, n in names]); ax[0, 1].axhline(0, color="k", lw=0.8); ax[0, 1].set_ylabel("paired change of the hidden-pixel chi2 per pixel vs physical-only (negative = better)"); ax[0, 1].set_title(f"{len(P)} galaxies, 36 hidden spans, fit on visible pixels only (mean of the unweighted per-galaxy chi2)", fontsize=9)
    sel = ig[:3]; w_ = 0.27
    for k, n in enumerate(sel):
        o = objs[n]; pr = jd.fit_object(fid, o["z"], o["f"], o["w"], parts_list=(("phys",), ("phys", "inst", "sky")))
        c0 = pr[("phys",)]["theta"]; c1 = pr[("phys", "inst", "sky")]["theta"][pr[("phys", "inst", "sky")]["tag"] == "phys"]; s = np.abs(c0).max()
        ax[1, 0].bar(np.arange(32) + (k - 1) * w_, (c1 - c0) / s * 100, w_, label=f"{o['kind']} z={o['z']:.2f}"); cache_c[f"dcoef_{n}"] = (c1 - c0) / s
    ax[1, 0].set_xlabel("legacy template number"); ax[1, 0].set_ylabel("coefficient change, joint - physical-only [percent of the largest physical-only coefficient]"); ax[1, 0].legend(fontsize=8); ax[1, 0].set_title("legacy template coefficients: joint fit vs physical-only fit", fontsize=9)
    for n, col in zip(ig[:3] + iq[:1] + ic, ("tab:blue", "tab:green", "tab:orange", "tab:red", "tab:gray")):
        o = objs[n]; zg = o["z"] + np.linspace(-0.004, 0.004, 41) * (1 + o["z"]); cl = co[f"o{n}"]["clean"]; c_raw, c_cl, c_j = [], [], []
        for z in zg:
            fr = jd.fit_object(fid, z, o["f"], o["w"], parts_list=(("phys",), ("phys", "inst", "sky")))
            c_raw.append(jd.chi2(o["f"], fr[("phys",)]["model"], o["w"])[0]); c_j.append(jd.chi2(o["f"], fr[("phys", "inst", "sky")]["model"], o["w"])[0])
            c_cl.append(jd.chi2(cl, jd.fit_object(fid, z, cl, o["w"], parts_list=(("phys",),))[("phys",)]["model"], o["w"])[0])
        dz = (zg - o["z"]) / (1 + o["z"]) * 3e5; lbl = f"{o['kind']} z={o['z']:.2f}"
        ax[1, 1].plot(dz, np.array(c_raw) - min(c_raw), color=col, lw=1.2, label=f"{lbl}: physical-only on the raw flux"); ax[1, 1].plot(dz, np.array(c_cl) - min(c_cl), color=col, lw=1.2, ls="--", label=f"{lbl}: physical on the cleaned flux")
        cache_c[f"zscan_{n}"] = np.array([dz, c_raw, c_cl, c_j]); print("zscan", n, "min at (km/s)", dz[int(np.argmin(c_raw))], dz[int(np.argmin(c_cl))], dz[int(np.argmin(c_j))])
    ax[1, 1].set_yscale("symlog", linthresh=10); ax[1, 1].set_xlabel("velocity offset from the catalogue redshift [km/s]"); ax[1, 1].set_ylabel("chi2 - chi2_min at fixed template coefficients refit per redshift (symlog)"); ax[1, 1].legend(fontsize=6); ax[1, 1].set_title("redshift scan of the physical fit before and after the subtraction", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "joint_fit_physical_change.png"), dpi=120); plt.close(fig); np.savez(os.path.join(cachedir, "joint_fit_physical_change.npz"), **cache_c)


def make_bounded(root, phys_dir, cache, legacy, figdir, cachedir, ks=(3.0, 10.0)):
    """Same joint fit with the instrument and sky coefficients BOUNDED to +-k prior sigma (box-constrained least squares, physical coefficients free): what survives
    when the nuisance templates are limited to amplitudes the SKY rows and VAC spectra actually show."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import lsq_linear
    fid, objs, co, res, P, show = jd.run(root, phys_dir, cache, legacy, figdir, cachedir)
    out, cache_c = [], {}; ig = [n for n, o in enumerate(objs) if o["kind"] == "galaxy"]; iq = [n for n, o in enumerate(objs) if o["kind"] == "QSO"]; ic = [n for n, o in enumerate(objs) if o["kind"] == "contaminant"]
    for n, o in enumerate(objs):
        X, prec, tag = fid.columns(o["z"], ("phys", "inst", "sky")); sig = np.concatenate([np.full((tag == "phys").sum(), np.inf), np.maximum(fid.prior_inst, 1e-12), np.maximum(fid.prior_sky, 1e-12)])
        sw = np.sqrt(o["w"]); A = (X * sw[None]).T; b = sw * o["f"]; npx = int((o["w"] > 0).sum()); row = dict(kind=o["kind"], z=o["z"], snr=o["snr"])
        p_only = jd.fit_object(fid, o["z"], o["f"], o["w"], parts_list=(("phys",),))[("phys",)]; chi0 = jd.chi2(o["f"], p_only["model"], o["w"])[0]
        for k in ks:
            lo = np.where(np.isinf(sig), -np.inf, -k * sig); hi = np.where(np.isinf(sig), np.inf, k * sig); sol = lsq_linear(A, b, bounds=(lo, hi), method="bvls", max_iter=500); th = sol.x
            comp = {t: th[tag == t] @ X[tag == t] for t in ("phys", "inst", "sky")}; m = sum(comp.values()); E = {t: float((o["w"] * comp[t] ** 2).sum()) for t in comp}
            row[f"k{k:g}"] = dict(chi2_per_pix=jd.chi2(o["f"], m, o["w"])[0] / npx, chi2_reduction=1 - jd.chi2(o["f"], m, o["w"])[0] / chi0, share={t: E[t] / sum(E.values()) for t in E}, inst_peak_flux=float(np.abs(comp["inst"]).max()),
                                  at_bound_inst=float((np.abs(th[tag == "inst"]) > 0.999 * k * fid.prior_inst).mean()), at_bound_sky=float((np.abs(th[tag == "sky"]) > 0.999 * k * fid.prior_sky).mean()))
            cache_c[f"o{n}_k{k:g}_inst"] = comp["inst"]; cache_c[f"o{n}_k{k:g}_sky"] = comp["sky"]; cache_c[f"o{n}_k{k:g}_phys"] = comp["phys"]
        row["free_prior"] = dict(chi2_per_pix=res[n]["chi2_per_pix"]["joint"], chi2_reduction=1 - res[n]["chi2_per_pix"]["joint"] / res[n]["chi2_per_pix"]["phys"], share=res[n]["share_of_model_energy"], inst_peak_flux=float(np.abs(co[f"o{n}"]["inst"]).max()))
        row["physical_only_chi2_per_pix"] = chi0 / npx; out.append(row)
        print(n, o["kind"], f"z={o['z']:.2f}", "phys-only", round(chi0 / npx, 2), "free-prior", round(row["free_prior"]["chi2_per_pix"], 2), "inst share", round(row["free_prior"]["share"]["inst"], 3), "| k=3:", round(row["k3"]["chi2_per_pix"], 2), "inst share", round(row["k3"]["share"]["inst"], 4), "at-bound", round(row["k3"]["at_bound_inst"], 2), "| k=10:", round(row["k10"]["chi2_per_pix"], 2), flush=True)
    json.dump(out, open(os.path.join(root, "joint_fit_bounded.json"), "w"), indent=1)
    sel = ig[::2][:3] + iq[:1] + ic; fig, ax = plt.subplots(len(sel), 2, figsize=(18, 3.6 * len(sel)), sharex=True)
    for r, n in enumerate(sel):
        o = objs[n]; d = co[f"o{n}"]; lam = jd.LAM; g = d["w"] > 0
        ax[r, 0].plot(lam, d["inst"], color="tab:orange", lw=0.8, label="instrument component, free fit (prior only)"); ax[r, 0].plot(lam, cache_c[f"o{n}_k3_inst"], color="tab:red", lw=0.8, label="instrument component, bounded to +-3 prior sigma")
        ax[r, 0].plot(lam, cache_c[f"o{n}_k3_sky"], color="tab:purple", lw=0.8, label="sky component, bounded to +-3 prior sigma"); ax[r, 0].set_ylabel(f"{o['kind']} z={o['z']:.2f}\ncomponent flux [1e-17 erg/s/cm2/A]"); ax[r, 0].legend(fontsize=7, loc="upper right"); ax[r, 0].axhline(0, color="gray", lw=0.5)
        cl = d["f"] - cache_c[f"o{n}_k3_inst"] - cache_c[f"o{n}_k3_sky"]; ax[r, 1].plot(lam[g], d["f"][g], color="0.8", lw=0.3); ax[r, 1].plot(lam, jd.nw(d["f"], d["w"], 6.0), color="k", lw=0.9, label="raw flux (NW smooth, sigma 6 px)")
        ax[r, 1].plot(lam, jd.nw(cl, d["w"], 6.0), color="tab:red", lw=0.9, label="flux minus the bounded instrument + sky"); ax[r, 1].plot(lam, jd.nw(d["clean"], d["w"], 6.0), color="tab:orange", lw=0.9, ls="--", label="flux minus the free-prior instrument + sky")
        ax[r, 1].set_ylim(*_robust_ylim(jd.nw(d["f"], d["w"], 6.0), 1.8)); ax[r, 1].legend(fontsize=7, loc="upper right"); ax[r, 1].set_ylabel("flux [1e-17 erg/s/cm2/A]")
        ax[r, 0].set_title(f"chi2/px: physical-only {out[n]['physical_only_chi2_per_pix']:.2f}, free {out[n]['free_prior']['chi2_per_pix']:.2f}, bounded +-3 sigma {out[n]['k3']['chi2_per_pix']:.2f}, +-10 sigma {out[n]['k10']['chi2_per_pix']:.2f}", fontsize=8)
    ax[-1, 0].set_xlabel("observed wavelength [Angstrom]"); ax[-1, 1].set_xlabel("observed wavelength [Angstrom]"); fig.tight_layout(); fig.savefig(os.path.join(figdir, "joint_fit_bounded.png"), dpi=100); plt.close(fig)
    np.savez(os.path.join(cachedir, "joint_fit_bounded.npz"), wavelength_A=jd.LAM, **cache_c)
