"""
Per-camera instrument templates: each camera has its own templates and its own number of templates.

Training data: the SKY rows of the cache cropped to the brz-grid range of each camera (b 0-2750, r 2700-5025, z 4900-7780: the
cache merged the cameras, so the overlap pixels are the inverse-variance mixture of two cameras), same masked-denoising chi2
(deeprock_lite), same tile split as the global sweep. Saved per camera as K###.npz with the templates on the camera's own pixels.

Overlap handling when the per-camera fits are applied to a merged (brz) spectrum follows desispec.coaddition.coadd_cameras: in the
overlap pixels the flux and ivar are an inverse-variance weighted combination of the two cameras, outside them each camera is copied.
`merge_camera_terms` applies the same rule to the additive term fitted for each camera, with the camera ivar as weights.

Run:  python -m DESIFlow.templates.camera_templates sweep --out OUT [--cams b,r,z] [--ks 1,2,...]
"""
import argparse
import json
import os
import time

import numpy as np

CAMS = {"b": (0, 2751), "r": (2700, 2700 + 2326), "z": (4900, 4900 + 2881)}
KS = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48]


def merge_camera_terms(terms, ivars, n=7781):
    """terms, ivars: dict camera -> array on that camera's brz-grid pixels (CAMS ranges). Returns the merged brz-grid term: inverse-variance
    weighted mean where cameras overlap, a copy elsewhere (the rule of coadd_cameras), 0 where no camera has weight."""
    num, den = np.zeros(n), np.zeros(n)
    for c, (a, b) in CAMS.items():
        num[a:b] += ivars[c] * terms[c]; den[a:b] += ivars[c]
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def sweep_camera(cam, out, ks=KS, steps=1500, batch=512, lr=3e-3, seed=0):
    import torch
    from DESIFlow.templates.deeprock_lite import evaluate, order_orthonormal, train_templates
    d = os.path.join(out, "instrument"); od = os.path.join(out, "instrument_cam", cam); os.makedirs(od, exist_ok=True)
    a0, a1 = CAMS[cam]; L = a1 - a0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftr, wtr = np.load(os.path.join(d, "train_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "train_ivar.npy"), mmap_mode="r")
    sw, sn = np.zeros(L), np.zeros(L)
    for a in range(0, len(wtr), 10000):
        c = np.asarray(wtr[a:a + 10000, a0:a1]); sw += c.sum(0, dtype=np.float64); sn += (c > 0).sum(0)
    s = np.sqrt(sw / np.maximum(sn, 1)); s = np.where(s > 0, s, np.median(s[s > 0])).astype(np.float32)

    def to_gpu(fm, wm):
        G = torch.empty((len(fm), L), dtype=torch.float32, device=dev); O = torch.empty_like(G)
        for a in range(0, len(fm), 10000):
            fc, wc = np.asarray(fm[a:a + 10000, a0:a1]), np.asarray(wm[a:a + 10000, a0:a1])
            G[a:a + 10000] = torch.from_numpy(fc * s[None]).to(dev); O[a:a + 10000] = torch.from_numpy(wc / s[None] ** 2).to(dev)
        return G, O
    G, O = to_gpu(ftr, wtr)
    GE, OE = to_gpu(np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r"))
    mk = dict(n_spans=max(6, int(36 * L / 7781)), len_range=(5, 250))
    spath = os.path.join(od, "summary.json")
    summary = {int(k): v for k, v in json.load(open(spath)).items()} if os.path.exists(spath) else {}
    for K in ks:
        t0 = time.time()
        U, hist = train_templates(G, O, K, steps=steps, batch=batch, lr=lr, seed=seed, mask_kw=mk)
        B, S = order_orthonormal(U, G, O)
        hid, base, vis = evaluate(B, GE, OE, mask_kw=mk)
        summary[K] = dict(hidden_chi2=float(hid.mean()), hidden_chi2_sem=float(hid.std() / np.sqrt(len(hid))), baseline_chi2=float(base.mean()), visible_chi2=float(vis.mean()), seconds=time.time() - t0)
        np.savez(os.path.join(od, f"K{K:03d}.npz"), templates_whitened=B.cpu().numpy(), templates_flux=B.cpu().numpy() / s[None], pixel_scale_s=s, energy=S.cpu().numpy(), K=K,
                 brz_index_range=np.array([a0, a1]), **summary[K])
        json.dump(summary, open(spath, "w"), indent=1)
        _log(f"camera {cam} K={K}: hold-out hidden chi2 {summary[K]['hidden_chi2']:.4f} +- {summary[K]['hidden_chi2_sem']:.4f} (zero model {summary[K]['baseline_chi2']:.4f})")
        torch.cuda.empty_cache()


def choose_k(summary):
    """The K with the lowest hold-out chi2 (each camera chooses its own)."""
    sm = {int(k): v for k, v in summary.items()}
    return min(sm, key=lambda k: sm[k]["hidden_chi2"])


def figure(out, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cache = {}
    for a, c in zip(ax, "brz"):
        S = {int(k): v for k, v in json.load(open(os.path.join(out, "instrument_cam", c, "summary.json"))).items()}
        k = np.array(sorted(S)); h = np.array([S[x]["hidden_chi2"] for x in k]); e = np.array([S[x]["hidden_chi2_sem"] for x in k]); v = np.array([S[x]["visible_chi2"] for x in k])
        a.errorbar(k, h, e, marker="o", ms=4, label="hold-out, hidden pixels (the score)"); a.plot(k, v, marker="s", ms=3, ls=":", label="hold-out, visible pixels (fitted)")
        a.axhline(S[int(k[0])]["baseline_chi2"], color="gray", ls="--", label="zero model"); a.axvline(choose_k(S), color="tab:red", lw=0.8, ls=":", label=f"best K = {choose_k(S)}")
        a.set_xscale("log"); a.set_xlabel("number of templates K"); a.set_ylabel("mean chi2 per pixel"); a.set_title(f"camera {c}"); a.legend(fontsize=7)
        cache.update({f"{c}_K": k, f"{c}_hidden_chi2": h, f"{c}_hidden_chi2_sem": e, f"{c}_visible_chi2": v, f"{c}_baseline": S[int(k[0])]["baseline_chi2"]})
    fig.tight_layout(); os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "camera_template_chi2_vs_K.png"), dpi=150); np.savez(os.path.join(cachedir, "camera_template_chi2_vs_K.npz"), **cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["sweep", "figure"]); ap.add_argument("--out", required=True); ap.add_argument("--fig", default=None); ap.add_argument("--figcache", default=None)
    ap.add_argument("--cams", default="b,r,z"); ap.add_argument("--ks", default=None); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "figure":
        figure(a.out, a.fig, a.figcache); raise SystemExit
    for c in a.cams.split(","):
        sweep_camera(c, a.out, ks=[int(x) for x in a.ks.split(",")] if a.ks else KS, steps=a.steps, seed=a.seed)
