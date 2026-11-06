"""
Figures of the physical-template experiment (arms A plain, B augmented, C explicit nuisance group, D both), each with a same-named .npz.

  physical_arms_chi2        hidden-pixel chi2 per pixel on held-out real galaxies, every trained arm evaluated with and without the nuisance fit,
                            on real data and on augmented data scored against the clean spectrum; paired differences against arm A
  physical_arms_templates   first rest-frame templates of each arm and the overlap (mean squared cosine of the principal angles) of their spans
  physical_arms_fits        three held-out galaxies: data, fitted model of arms A, C, D under the nuisance fit, and the nuisance part alone

Run:  python -m DESIFlow.templates.physical_figures --run RUN_DIR --data DATA_DIR --templates TEMPLATE_DIR --fig FIG --figcache CACHE [--K 24]
"""
import argparse
import json
import os

import numpy as np

COND = [("plain_real", "plain fit\nreal data"), ("nuis_real", "with nuisance fit\nreal data"), ("plain_aug", "plain fit\naugmented data"), ("nuis_aug", "with nuisance fit\naugmented data")]
COL = {"A": "tab:gray", "B": "tab:orange", "C": "tab:blue", "D": "tab:red"}
NAME = {"A": "A: plain", "B": "B: augmentation", "C": "C: nuisance group", "D": "D: both"}


def nw(f, w, sig):
    h = int(4 * sig); x = np.arange(-h, h + 1); k = np.exp(-0.5 * (x / sig) ** 2)
    num = np.convolve(w * f, k, "same"); den = np.convolve(w, k, "same")
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def fig_chi2(run, K, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(os.path.join(run, f"results_K{K}.json"))); arms = [a for a in "ABCD" if a in res]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5)); cache = {}
    x = np.arange(len(COND)); wd = 0.2
    for j, a in enumerate(arms):
        m = [res[a][f"eval_{c}"][0] for c, _ in COND]; e = [res[a][f"eval_{c}"][1] for c, _ in COND]
        ax[0].bar(x + (j - 1.5) * wd, m, wd, yerr=e, color=COL[a], label=NAME[a]); cache[f"mean_{a}"] = np.array(m); cache[f"sem_{a}"] = np.array(e)
    ax[0].set_xticks(x); ax[0].set_xticklabels([c for _, c in COND], fontsize=9); ax[0].set_ylabel("hidden-pixel chi2 per pixel, mean over held-out galaxies")
    lo = min(res[a][f"eval_{c}"][0] for a in arms for c, _ in COND); ax[0].set_ylim(0.95 * lo, None); ax[0].legend(fontsize=8)
    ax[0].set_title("trained arm (colour) x evaluation condition; error bar = standard error over rows")
    for j, a in enumerate(arms[1:]):
        d_m, d_e = [], []
        for c, _ in COND:
            d = np.load(os.path.join(run, f"perrow_K{K}_{a}_{c}.npy")) - np.load(os.path.join(run, f"perrow_K{K}_A_{c}.npy"))
            d_m.append(d.mean()); d_e.append(d.std() / np.sqrt(len(d)))
        ax[1].bar(x + (j - 1) * 0.27, d_m, 0.27, yerr=d_e, color=COL[a], label=NAME[a]); cache[f"diff_vs_A_{a}"] = np.array(d_m); cache[f"diff_sem_{a}"] = np.array(d_e)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(x); ax[1].set_xticklabels([c for _, c in COND], fontsize=9)
    ax[1].set_ylabel("paired change of hidden-pixel chi2 per pixel vs arm A (negative = better)"); ax[1].legend(fontsize=8); ax[1].set_title("same rows, same hidden spans, same augmentation draws")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "physical_arms_chi2.png"), dpi=130); plt.close(fig)
    np.savez(os.path.join(cachedir, "physical_arms_chi2.npz"), conditions=np.array([c for c, _ in COND]), arms=np.array(arms), **cache)


def fig_templates(run, K, figdir, cachedir, nshow=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phis = {a: np.load(os.path.join(run, f"phi_K{K}_{a}.npz")) for a in "ABCD" if os.path.exists(os.path.join(run, f"phi_K{K}_{a}.npz"))}
    arms = list(phis); x0, dl, nr = float(phis[arms[0]]["x0"]), float(phis[arms[0]]["dlog"]), int(phis[arms[0]]["nr"])
    lam = 10 ** (x0 + dl * np.arange(nr)); cache = {"rest_wavelength_A": lam}
    Q = {a: np.linalg.qr(phis[a]["phi"].T)[0] for a in arms}
    fig = plt.figure(figsize=(17, 11)); gs = fig.add_gridspec(nshow + 1, 2, width_ratios=[3, 1.1])
    for k in range(nshow):
        a_ = fig.add_subplot(gs[k, 0])
        for a in arms:
            v = phis[a]["phi"][k]; v = v / np.linalg.norm(v) * np.sqrt(len(v)); a_.plot(lam, v, color=COL[a], lw=0.5, label=NAME[a], alpha=0.8); cache[f"{a}_phi{k + 1}"] = v
        a_.set_ylabel(f"template {k + 1}\n(unit rms, arbitrary sign)"); a_.set_xlim(lam[0], lam[-1])
        if k == 0: a_.legend(fontsize=7, ncol=4)
    a_.set_xlabel("rest wavelength [Angstrom]")
    ov = np.zeros((len(arms), len(arms)))
    for i, a in enumerate(arms):
        for j, b in enumerate(arms):
            ov[i, j] = (np.linalg.svd(Q[a].T @ Q[b], compute_uv=False) ** 2).mean()
    h = fig.add_subplot(gs[:3, 1]); im = h.imshow(ov, vmin=0, vmax=1, cmap="viridis"); h.set_xticks(range(len(arms))); h.set_xticklabels(arms); h.set_yticks(range(len(arms))); h.set_yticklabels(arms)
    for i in range(len(arms)):
        for j in range(len(arms)): h.text(j, i, f"{ov[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    h.set_title("span overlap: mean squared cosine\nof the principal angles", fontsize=9); cache["span_overlap"] = ov; cache["arms"] = np.array(arms)
    fig.suptitle(f"Learned physical templates (K={K}): the first {nshow} of each arm", fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(figdir, "physical_arms_templates.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(cachedir, "physical_arms_templates.npz"), **cache)


def fig_fits(run, data, tdir, K, figdir, cachedir, n=3, seed=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from DESIFlow.templates import physical_templates as pt
    dev = torch.device("cpu"); tr = pt.load_data(data, "train", dev, 20000); ev = pt.load_data(data, "eval", dev)
    m = pt.Model(tdir, data, dev, K=K); m.set_whitening(tr["w"])
    rng = np.random.default_rng(seed); sel = rng.choice(len(ev["f"]), n, replace=False)
    f, w, z = ev["f"][sel], ev["w"][sel], ev["z"][sel]
    gen = torch.Generator().manual_seed(7); hid = pt.span_mask(n, pt.L, gen=gen)
    om = w / m.s ** 2; g = f * m.s
    arms = [a for a in "ACD" if os.path.exists(os.path.join(run, f"phi_K{K}_{a}.npz"))]
    lam = pt.LAM_OBS; fig, ax = plt.subplots(2 * n, 1, figsize=(18, 4.2 * n), sharex=True); cache = {"wavelength_A": lam, "rows": sel, "z": z.numpy()}
    for i in range(n):
        h = hid[i].numpy(); cache[f"g{i}_data_nw"] = nw((f[i]).numpy(), w[i].numpy(), 4.0)
        ax[2 * i].plot(lam, cache[f"g{i}_data_nw"], color="0.6", lw=1.0, label="data (NW smooth, 4 px)")
        for a in arms:
            phi = torch.tensor(np.load(os.path.join(run, f"phi_K{K}_{a}.npz"))["phi"])
            with torch.no_grad():
                ph, nu = m.predict_parts(phi, z[i:i + 1], g[i:i + 1], (om * (~hid))[i:i + 1])
            tot = ((ph + nu)[0] / m.s).numpy(); nuis = (nu[0] / m.s).numpy()
            chi = float((om[i] * (g[i] - (ph + nu)[0]) ** 2)[hid[i]].sum() / max(int((om[i] > 0)[hid[i]].sum()), 1))
            ax[2 * i].plot(lam, nw(tot, np.ones(pt.L), 1.0), color=COL[a], lw=0.9, label=f"{NAME[a]} (+ nuisance fit): hidden chi2 {chi:.2f}")
            ax[2 * i + 1].plot(lam, nw(nuis, np.ones(pt.L), 1.0), color=COL[a], lw=0.9, label=NAME[a])
            cache[f"g{i}_{a}_total"] = tot; cache[f"g{i}_{a}_nuisance"] = nuis
        for k in range(2):
            s0 = None
            for j in range(pt.L + 1):
                if j < pt.L and h[j] and s0 is None: s0 = j
                if (j == pt.L or not h[j]) and s0 is not None: ax[2 * i + k].axvspan(lam[s0], lam[j - 1], color="0.88", lw=0, zorder=0); s0 = None
        ax[2 * i].set_ylabel(f"galaxy z={float(z[i]):.2f}\nflux [1e-17 erg/s/cm2/A]"); ax[2 * i + 1].set_ylabel("nuisance part alone\n[1e-17 erg/s/cm2/A]"); ax[2 * i].legend(fontsize=7, ncol=2)
    ax[-1].set_xlabel("observed wavelength [Angstrom]"); fig.suptitle("Held-out galaxies: fits fitted on visible pixels (grey = hidden spans), nuisance group active for every arm", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(os.path.join(figdir, "physical_arms_fits.png"), dpi=100); plt.close(fig)
    np.savez(os.path.join(cachedir, "physical_arms_fits.npz"), **cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--templates", required=True)
    ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True); ap.add_argument("--K", type=int, default=24); ap.add_argument("--which", default="chi2,templates,fits")
    a = ap.parse_args(); os.makedirs(a.fig, exist_ok=True); os.makedirs(a.figcache, exist_ok=True); w = a.which.split(",")
    if "chi2" in w: fig_chi2(a.run, a.K, a.fig, a.figcache)
    if "templates" in w: fig_templates(a.run, a.K, a.fig, a.figcache)
    if "fits" in w: fig_fits(a.run, a.data, a.templates, a.K, a.fig, a.figcache)
