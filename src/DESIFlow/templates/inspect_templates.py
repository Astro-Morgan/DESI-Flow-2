"""
What is in a set of instrument templates? Per-template structure measures and links to calibration-type effects.

For a saved set (K templates, orthonormal in whitened units, ordered by the energy they carry) this reports, per template:
  var     share of the energy of the fitted reconstructions (S^2 / sum S^2);
  edge    share of the (whitened) template power within 150 A of a camera edge (6 zones, 14% of the pixels);
  lines   share of the power on the strongest sky-line pixels (8% of the pixels, from the median VAC sky spectrum);
  camR2   share of the template variance explained by a piecewise quadratic per camera (b < 5780 A, r 5780-7570, z > 7570);
  step    jump across the camera junctions (mean in the 60 A right of the junction minus the 60 A left), in units of the template rms;
and, over the hold-out spectra, the correlation of each template's coefficient with the measured mean offsets of the three cameras
(weighted mean flux in 4200-5600, 6000-7400, 7800-9200 A) and with the camera differences r - b and z - r.

Run:  python -m DESIFlow.templates.inspect_templates --out OUT --fig FIG --figcache CACHE [--kind instrument --K 20]
"""
import argparse
import json
import os

import numpy as np

LAM = 3600.0 + 0.8 * np.arange(7781)
EDGE_ZONES = [(3600, 3750), (5650, 5800), (5760, 5910), (7470, 7620), (7520, 7670), (9674, 9824)]
JUNCTIONS = (5780.0, 7570.0)
WIN = {"b": (4200, 5600), "r": (6000, 7400), "z": (7800, 9200)}


def edge_mask():
    m = np.zeros(len(LAM), bool)
    for lo, hi in EDGE_ZONES:
        m |= (LAM >= lo) & (LAM < hi)
    return m


def line_mask(out):
    from scipy.ndimage import median_filter
    f = np.load(os.path.join(out, "sky", "train_flux.npy"), mmap_mode="r")
    s = np.median(np.asarray(f[:3000]), 0)
    d = s - median_filter(s, 101, mode="nearest")
    return d > np.percentile(d, 92)


def camera_basis(deg=2):
    cam = np.digitize(LAM, JUNCTIONS)
    cols = []
    for c in range(3):
        x = np.where(cam == c, (LAM - LAM[cam == c].mean()) / (np.ptp(LAM[cam == c]) / 2), 0.0)
        for d in range(deg + 1):
            cols.append(np.where(cam == c, np.polynomial.legendre.legval(x, [0] * d + [1]), 0.0))
    Q, _ = np.linalg.qr(np.stack(cols, 1))
    return Q


def metrics(Tf, W, energy, out):
    K = len(W)
    em, lm, Q = edge_mask(), line_mask(out), camera_basis()
    res = []
    for k in range(K):
        w, t = W[k], Tf[k]
        r2 = float(((Q.T @ w) ** 2).sum() / (w ** 2).sum())
        steps = []
        for j in JUNCTIONS:
            L = (LAM >= j - 60) & (LAM < j); R = (LAM >= j) & (LAM < j + 60)
            steps.append(float((t[R].mean() - t[L].mean()) / (t.std() + 1e-12)))
        res.append(dict(k=k + 1, var=float(energy[k] ** 2 / (energy ** 2).sum()), edge=float((w[em] ** 2).sum() / (w ** 2).sum()), lines=float((w[lm] ** 2).sum() / (w ** 2).sum()),
                        camR2=r2, step_br=steps[0], step_rz=steps[1]))
    return res, em, lm


def coefficients(out, kind, U, s, n=20000):
    import torch
    from DESIFlow.templates.deeprock_lite import wls_project
    d = os.path.join(out, kind)
    f, w = np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r")
    n = min(n, len(f)); Ut = torch.tensor(U)
    C, offs = [], []
    for a in range(0, n, 1000):
        fc, wc = np.asarray(f[a:a + 1000]), np.asarray(w[a:a + 1000])
        C.append(wls_project(Ut, torch.tensor(fc * s[None]), torch.tensor(wc / s[None] ** 2)).numpy())
        row = []
        for lo, hi in WIN.values():
            m = (LAM >= lo) & (LAM < hi)
            row.append((fc[:, m] * wc[:, m]).sum(1) / np.maximum(wc[:, m].sum(1), 1e-12))
        offs.append(np.stack(row, 1))
    return np.concatenate(C), np.concatenate(offs)


def main(out, figdir, cachedir, kind="instrument", K=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(out, kind, f"K{K:03d}.npz"))
    Tf, W, energy, s = z["templates_flux"].copy(), z["templates_whitened"].copy(), z["energy"], z["pixel_scale_s"]
    ref = (LAM > 4000) & (LAM < 9000)
    for k in range(K):                                                                    # sign: positive median over 4000-9000 A
        if np.median(Tf[k][ref]) < 0:
            Tf[k], W[k] = -Tf[k], -W[k]
    res, em, lm = metrics(Tf, W, energy, out)
    C, offs = coefficients(out, kind, W, s)
    names = ["b", "r", "z"]
    D = np.stack([offs[:, 1] - offs[:, 0], offs[:, 2] - offs[:, 1]], 1)
    for k in range(K):
        for j, nm in enumerate(names):
            res[k][f"corr_offset_{nm}"] = float(np.corrcoef(C[:, k], offs[:, j])[0, 1])
        res[k]["corr_r_minus_b"] = float(np.corrcoef(C[:, k], D[:, 0])[0, 1]); res[k]["corr_z_minus_r"] = float(np.corrcoef(C[:, k], D[:, 1])[0, 1])
        res[k]["coef_std"] = float(C[:, k].std()); res[k]["typical_flux_amplitude"] = float(C[:, k].std() * Tf[k][ref].std())
    json.dump(res, open(os.path.join(out, kind, f"K{K:03d}_template_metrics.json"), "w"), indent=1)
    print("  #   var%   edge%  lines%  camR2  step_br step_rz | corr(coef, offset b/r/z)   corr(r-b) corr(z-r) | typical amplitude [1e-17]")
    for r in res:
        print(f" {r['k']:2d}  {100 * r['var']:5.1f}  {100 * r['edge']:5.1f}  {100 * r['lines']:5.1f}  {r['camR2']:5.2f}  {r['step_br']:6.2f} {r['step_rz']:6.2f} |  {r['corr_offset_b']:5.2f} {r['corr_offset_r']:5.2f} {r['corr_offset_z']:5.2f}  "
              f"{r['corr_r_minus_b']:7.2f} {r['corr_z_minus_r']:7.2f} | {r['typical_flux_amplitude']:.3f}")
    ncol = 4; nrow = int(np.ceil(K / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 2.9 * nrow), sharex=True)
    for k, a in enumerate(ax.ravel()[:K]):
        t = Tf[k]
        a.plot(LAM, t, lw=0.6, color="k")
        for lo, hi in EDGE_ZONES:
            a.axvspan(lo, hi, color="tab:orange", alpha=0.10, lw=0)
        for j in JUNCTIONS:
            a.axvline(j, color="tab:blue", lw=0.6, ls=":")
        lo, hi = np.percentile(t[ref], [1, 99]); pad = 3 * max(hi - lo, 1e-9)
        a.set_ylim(lo - pad, hi + pad)
        r = res[k]
        a.set_title(f"#{k + 1}: energy {100 * r['var']:.1f}%, edge {100 * r['edge']:.0f}%, lines {100 * r['lines']:.0f}%, camera-quadratic R2 {r['camR2']:.2f}", fontsize=7.5)
        if k % ncol == 0:
            a.set_ylabel("template [1e-17 erg/s/cm2/A per unit coefficient]", fontsize=6)
        if k >= K - ncol:
            a.set_xlabel("observed wavelength [Angstrom]", fontsize=8)
    for a in ax.ravel()[K:]:
        a.axis("off")
    fig.suptitle(f"{kind} templates, K = {K} (orange: 150 A camera-edge zones; dotted: camera junctions; y-range clipped to the 4000-9000 A range)", fontsize=9)
    fig.tight_layout(); os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, f"{kind}_templates_K{K:03d}.png"), dpi=150); plt.close(fig)
    np.savez(os.path.join(cachedir, f"{kind}_templates_K{K:03d}.npz"), wavelength_A=LAM, templates_flux=Tf, templates_whitened=W, energy=energy, edge_mask=em, line_mask=lm,
             **{k: np.array([r[k] for r in res]) for k in res[0]})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ks = np.arange(1, K + 1)
    ax[0].bar(ks, [100 * r["var"] for r in res]); ax[0].set_ylabel("share of the reconstruction energy [percent]"); ax[0].set_xlabel("template number")
    ax[1].bar(ks - 0.2, [100 * r["edge"] for r in res], 0.4, label="in 150 A camera-edge zones (14% of pixels)"); ax[1].bar(ks + 0.2, [100 * r["lines"] for r in res], 0.4, label="on sky-line pixels (8% of pixels)")
    ax[1].set_ylabel("share of the template power [percent]"); ax[1].set_xlabel("template number"); ax[1].legend(fontsize=7)
    for j, (nm, col) in enumerate(zip(("r - b offset", "z - r offset"), ("tab:red", "tab:green"))):
        ax[2].bar(ks + (j - 0.5) * 0.4, [r["corr_r_minus_b" if j == 0 else "corr_z_minus_r"] for r in res], 0.4, color=col, label=nm)
    ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_ylabel("correlation of the coefficient with the measured offset"); ax[2].set_xlabel("template number"); ax[2].legend(fontsize=8)
    ax[2].set_title("hold-out spectra: camera-to-camera offset (weighted mean flux)")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{kind}_templates_K{K:03d}_summary.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True); ap.add_argument("--fig", required=True); ap.add_argument("--figcache", required=True)
    ap.add_argument("--kind", default="instrument"); ap.add_argument("--K", type=int, default=20)
    a = ap.parse_args()
    main(a.out, a.fig, a.figcache, a.kind, a.K)
