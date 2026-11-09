"""
Diagnostics of the instrument templates (exploratory).

  capture   how much of each global K=20 instrument template (restricted to a camera) lies in the span of a per-camera template set, for several (K_b, K_r, K_z):
            captured fraction = |B t|^2 / |t|^2 in whitened units (B orthonormal)
  solar     the first camera-b template against a solar-type stellar template (z = 0 lines: Ca H&K, G band, Mg b, Na D) and its coefficient against the
            observing program of the SKY fibres (dark / bright / backup / other; bright and twilight skies carry the solar spectrum)

Run:  python -m DESIFlow.templates.template_diagnostics capture|solar --root ROOT_V2 --old OLD_SKY_TEMPLATES --cache DR2_CACHE --fig FIG --figcache CACHE
"""
import argparse
import json
import os

import numpy as np

CAMS = {"b": (0, 2751), "r": (2700, 5026), "z": (4900, 7781)}


def cam_basis(root, cam, K, seed=0):
    z = np.load(os.path.join(root, f"seed{seed}", "instrument_cam", cam, f"K{K:03d}.npz"))
    return z["templates_whitened"].astype(np.float64), z["pixel_scale_s"].astype(np.float64), z["templates_flux"].astype(np.float64)


def capture(root, old, figdir, cachedir, sets=((3, 3, 12), (5, 5, 11), (8, 8, 13), (12, 12, 16), (20, 20, 24))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.load(os.path.join(old, "instrument", "K020.npz")); Tg = g["templates_whitened"]; E = g["energy"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5)); cache = {"sets": np.array(sets)}
    for j, c in enumerate("brz"):
        a0, a1 = CAMS[c]; frac = np.zeros((len(sets), len(Tg)))
        for i, s in enumerate(sets):
            B, _, _ = cam_basis(root, c, s[j])
            for k in range(len(Tg)):
                t = Tg[k, a0:a1].astype(np.float64); frac[i, k] = float(((B @ t) ** 2).sum() / (t ** 2).sum())
        im = ax[j].imshow(frac, vmin=0, vmax=1, cmap="viridis", aspect="auto"); ax[j].set_yticks(range(len(sets))); ax[j].set_yticklabels([f"K_{c} = {s[j]}" for s in sets])
        ax[j].set_xticks(range(len(Tg))); ax[j].set_xticklabels([str(k + 1) for k in range(len(Tg))], fontsize=7); ax[j].set_xlabel("global K=20 template number"); ax[j].set_title(f"camera {c}: fraction of each global template inside the per-camera span")
        for i in range(len(sets)):
            for k in range(len(Tg)): ax[j].text(k, i, f"{100 * frac[i, k]:.0f}", ha="center", va="center", color="w", fontsize=6)
        cache[f"{c}_captured"] = frac
    fig.colorbar(im, ax=ax, shrink=0.8, label="captured fraction"); fig.savefig(os.path.join(figdir, "template_capture_K20.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    cache["global_energy"] = E; np.savez(os.path.join(cachedir, "template_capture_K20.npz"), **cache)
    for i, s in enumerate(sets):
        print(s, {c: round(float(np.average(cache[f"{c}_captured"][i], weights=E)), 3) for c in "brz"})


def solar(root, old, cache_dir, figdir, cachedir, Kb=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from redrock.templates import load_templates
    B, s, Tf = cam_basis(root, "b", Kb)
    lam = 3600.0 + 0.8 * np.arange(CAMS["b"][1])
    # solar-type stellar template at z = 0
    tpl = [t for t in load_templates() if t.template_type == "STAR"]
    names = [(t.sub_type) for t in tpl]; pick = [t for t in tpl if t.sub_type in ("G", "F", "K")]
    fig, ax = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw=dict(height_ratios=[1.3, 1, 1])); cache = {"wavelength_A": lam, "b1": Tf[0]}
    t1 = Tf[0]; w = np.ones_like(lam)

    def hp(x, sig=40):
        h = int(4 * sig); k = np.exp(-0.5 * (np.arange(-h, h + 1) / sig) ** 2); k /= k.sum(); return x - np.convolve(np.pad(x, h, mode="edge"), k, "valid")
    ax[0].plot(lam, hp(t1) / np.abs(hp(t1)).max(), color="k", lw=0.8, label="camera b template #1 (high-pass)")
    for t in pick:
        wl = t.wave; fl = t.flux[0] if t.flux.ndim > 1 else t.flux
        for sub_i in range(min(1, len(np.atleast_2d(t.flux)))):
            m = np.interp(lam, wl, np.atleast_2d(t.flux)[sub_i]); r = np.corrcoef(hp(t1), hp(m))[0, 1]
            ax[0].plot(lam, hp(m) / np.abs(hp(m)).max() - 1.5, lw=0.6, label=f"Redrock STAR {t.sub_type} template #1 (z=0), r = {r:+.2f}"); cache[f"star_{t.sub_type}"] = m
    for x0, nm in [(3934, "Ca K"), (3968, "Ca H"), (4305, "G band"), (5175, "Mg b"), (5893, "Na D")]:
        ax[0].axvline(x0, color="tab:red", lw=0.5, ls=":"); ax[0].text(x0, 1.15, nm, fontsize=7, color="tab:red", ha="center")
    ax[0].set_xlim(3600, 5800); ax[0].set_xlabel("observed wavelength [Angstrom]"); ax[0].set_ylabel("high-pass shape (sigma 40 px removed), unit peak"); ax[0].legend(fontsize=7, loc="lower right")
    # coefficient against the program
    idx = np.load(os.path.join(cache_dir, "index", "group_start.npy")); stop = np.load(os.path.join(cache_dir, "index", "group_stop.npy"))
    prog = np.load(os.path.join(cache_dir, "index", "group_program.npy")); surv = np.load(os.path.join(cache_dir, "index", "group_survey.npy"))
    z = np.load(os.path.join(old, "instrument", "eval_rows.npz")); rows = z["cache_rows"]
    f = np.load(os.path.join(old, "instrument", "eval_flux.npy"), mmap_mode="r")[:, :CAMS["b"][1]]; wv = np.load(os.path.join(old, "instrument", "eval_ivar.npy"), mmap_mode="r")[:, :CAMS["b"][1]]
    # cache row -> group: tgt/sky rows live in separate ranges; group index by searching the row ranges (sky rows indexed through sky_rows)
    sky_rows = np.load(os.path.join(cache_dir, "index", "sky_rows.npy")); tgt_rows = np.load(os.path.join(cache_dir, "index", "tgt_rows.npy"))
    gs = np.load(os.path.join(cache_dir, "index", "group_start.npy")); ge = np.load(os.path.join(cache_dir, "index", "group_stop.npy"))
    zrow = sky_rows[rows]                                                                       # cache sky row -> zall row -> group
    grp = np.searchsorted(gs, zrow, side="right") - 1
    pr = np.array([str(prog[g]).strip() for g in grp]); su = np.array([str(surv[g]).strip() for g in grp])
    T = Tf[:5]; coef = []
    for i in range(len(rows)):
        fr = np.asarray(f[i], np.float64); wr = np.asarray(wv[i], np.float64); A = (T * wr) @ T.T + 1e-9 * np.eye(5) * np.trace((T * wr) @ T.T) / 5
        coef.append(np.linalg.solve(A, (T * wr) @ fr))
    coef = np.array(coef); progs = [p for p in np.unique(pr) if (pr == p).sum() > 50]
    ax[1].boxplot([coef[pr == p, 0] for p in progs], tick_labels=[f"{p}\n(n={(pr == p).sum()})" for p in progs], showfliers=False); ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_ylabel("coefficient of template #1 (b, K=%d)\nper unit template, held-out SKY rows" % Kb); ax[1].set_title("template #1 coefficient by observing program (box = quartiles, no outliers)")
    for p in progs:
        cache[f"coef1_{p}"] = coef[pr == p, 0]
    ax[2].hist([np.abs(coef[pr == p, 0]) for p in progs], bins=np.logspace(-3, 1, 50), label=progs, histtype="step"); ax[2].set_xscale("log"); ax[2].set_xlabel("|coefficient of template #1|"); ax[2].set_ylabel("held-out SKY rows"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "template_b1_solar.png"), dpi=120); plt.close(fig)
    np.savez(os.path.join(cachedir, "template_b1_solar.npz"), **cache)
    for p in progs:
        print(p, int((pr == p).sum()), "median |c1|", round(float(np.median(np.abs(coef[pr == p, 0]))), 4), "mean c1", round(float(coef[pr == p, 0].mean()), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["capture", "solar"]); ap.add_argument("--root", required=True); ap.add_argument("--old", required=True); ap.add_argument("--cache", default=None)
    ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    a = ap.parse_args(); os.makedirs(a.fig, exist_ok=True); os.makedirs(a.figcache, exist_ok=True)
    capture(a.root, a.old, a.fig, a.figcache) if a.stage == "capture" else solar(a.root, a.old, a.cache, a.fig, a.figcache)
