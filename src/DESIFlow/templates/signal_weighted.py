"""
Exploratory: turn the per-camera instrument templates toward the rows that carry signal.

Most SKY-fibre rows are indistinguishable from noise; the structure we want to model lives in a minority of them (moonlit and twilight skies, bad
sky subtraction, ...). Row score e = (zero-model chi2 per good pixel of the whole row) - 1: pure noise gives e ~ 0 +- sqrt(2/N) (0.03 for 2751 px), signal gives
e > 0.1. Training modes for each camera and K:
  uniform    the baseline (rows drawn uniformly)
  weighted   rows drawn with probability proportional to max(e, 0) + 0.02
  subset     only the rows with e above the 80th percentile
Scored on the hold-out rows by the hidden-pixel chi2 (fixed masks) on all rows, on the high-score rows (e above the 80th percentile of the hold-out) and on the top 5%.

Run:  python -m DESIFlow.templates.signal_weighted --root ROOT --cams b,r,z --ks 3,5,8,12,16,24 --modes uniform,weighted,subset
Output: ROOT/signal_weighted/<cam>/<mode>/K###.npz and results.json
"""
import argparse
import json
import os
import time

import numpy as np


def run(cam, root, ks, modes, steps=1500, batch=512, lr=3e-3, seed=0, src="seed0"):
    import torch
    from DESIFlow.templates.camera_templates import CAMS
    from DESIFlow.templates.deeprock_lite import evaluate, order_orthonormal, train_templates
    d = os.path.join(root, src, "instrument"); a0, a1 = CAMS[cam]; L = a1 - a0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(os.path.join(root, src, "instrument_cam", cam, "K003.npz")); s = z["pixel_scale_s"]

    def to_gpu(name):
        fm = np.load(os.path.join(d, f"{name}_flux.npy"), mmap_mode="r"); wm = np.load(os.path.join(d, f"{name}_ivar.npy"), mmap_mode="r")
        G = torch.empty((len(fm), L), dtype=torch.float32, device=dev); O = torch.empty_like(G)
        for a in range(0, len(fm), 10000):
            G[a:a + 10000] = torch.from_numpy(np.asarray(fm[a:a + 10000, a0:a1]) * s[None]).to(dev); O[a:a + 10000] = torch.from_numpy(np.asarray(wm[a:a + 10000, a0:a1]) / s[None] ** 2).to(dev)
        return G, O
    G, O = to_gpu("train"); GE, OE = to_gpu("eval")
    score = lambda g, o: ((o * g ** 2).sum(1) / (o > 0).sum(1).clamp(min=1) - 1.0)
    e_tr, e_ev = score(G, O), score(GE, OE)
    thr80 = torch.quantile(e_tr, 0.8); sub = e_tr > thr80; hi = (e_ev > torch.quantile(e_ev, 0.8)).cpu().numpy(); top = (e_ev > torch.quantile(e_ev, 0.95)).cpu().numpy()
    probs = (e_tr.clamp(min=0) + 0.02); probs = probs / probs.sum()
    mk = dict(n_spans=max(6, int(36 * L / 7781)), len_range=(5, 250)); res = {}
    rp = os.path.join(root, "signal_weighted", cam, "results.json"); os.makedirs(os.path.dirname(rp), exist_ok=True)
    if os.path.exists(rp):
        res = json.load(open(rp))
    res["score"] = dict(e_train_quantiles=[float(x) for x in torch.quantile(e_tr, torch.tensor([0.05, 0.25, 0.5, 0.8, 0.95, 0.99], device=dev))], threshold80=float(thr80), noise_sigma=float((2.0 / L) ** 0.5))
    for mode in modes:
        os.makedirs(os.path.join(root, "signal_weighted", cam, mode), exist_ok=True); res.setdefault(mode, {})
        for K in ks:
            t0 = time.time()
            if mode == "subset":
                U, _ = train_templates(G[sub], O[sub], K, steps=steps, batch=min(batch, int(sub.sum())), lr=lr, seed=seed, mask_kw=mk)
                B, S = order_orthonormal(U, G[sub], O[sub])
            else:
                U, _ = train_templates(G, O, K, steps=steps, batch=batch, lr=lr, seed=seed, mask_kw=mk, probs=probs if mode == "weighted" else None)
                B, S = order_orthonormal(U, G, O)
            hid, base, vis = evaluate(B, GE, OE, mask_kw=mk)
            r = dict(all=float(hid.mean()), high=float(hid[hi].mean()), top5=float(hid[top].mean()), base_all=float(base.mean()), base_high=float(base[hi].mean()), base_top5=float(base[top].mean()))
            res[mode][str(K)] = r
            np.savez(os.path.join(root, "signal_weighted", cam, mode, f"K{K:03d}.npz"), templates_whitened=B.cpu().numpy(), templates_flux=B.cpu().numpy() / s[None], pixel_scale_s=s, energy=S.cpu().numpy(), **r)
            json.dump(res, open(rp, "w"), indent=1)
            print(time.strftime("%H:%M:%S"), f"{cam} {mode} K={K}: all {r['all']:.4f} (zero {r['base_all']:.4f}) | high-score rows {r['high']:.4f} ({r['base_high']:.4f}) | top 5% {r['top5']:.3f} ({r['base_top5']:.3f}) {time.time() - t0:.0f}s", flush=True)
            torch.cuda.empty_cache()


def figure(root, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 8)); cache = {}
    for j, c in enumerate("brz"):
        r = json.load(open(os.path.join(root, "signal_weighted", c, "results.json"))); K = np.array(sorted(int(k) for k in r["uniform"]))
        for i, (key, nm) in enumerate((("all", "all hold-out rows"), ("high", "high-score hold-out rows (top 20%)"))):
            for mode, col in (("uniform", "k"), ("weighted", "tab:red"), ("subset", "tab:blue")):
                y = np.array([r[mode][str(k)][key] for k in K]); ax[i, j].plot(K, y, marker="o", color=col, label=mode); cache[f"{c}_{key}_{mode}"] = y
            ax[i, j].axhline(r["uniform"][str(K[0])]["base_all" if key == "all" else "base_high"], color="gray", ls="--", lw=0.8, label="zero model")
            ax[i, j].set_xscale("log"); ax[i, j].set_xlabel("number of templates K"); ax[i, j].set_ylabel("hold-out hidden-pixel chi2 per pixel"); ax[i, j].set_title(f"camera {c}: {nm}", fontsize=9); ax[i, j].legend(fontsize=7)
        cache[f"{c}_K"] = K; cache[f"{c}_score_quantiles"] = np.array(r["score"]["e_train_quantiles"])
    fig.suptitle("Per-camera instrument templates trained on uniform / score-weighted / high-score-only rows (row score = whole-row zero-model chi2 - 1; noise-only sigma 0.03)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(figdir, "signal_weighted_training.png"), dpi=120); np.savez(os.path.join(cachedir, "signal_weighted_training.npz"), **cache)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True); ap.add_argument("--cams", default="b,r,z"); ap.add_argument("--ks", default="3,5,8,12,16,24"); ap.add_argument("--modes", default="uniform,weighted,subset"); ap.add_argument("--fig", default=None); ap.add_argument("--figcache", default=None)
    a = ap.parse_args()
    if a.fig:
        figure(a.root, a.fig, a.figcache); raise SystemExit
    for c in a.cams.split(","):
        run(c, a.root, [int(x) for x in a.ks.split(",")], a.modes.split(","))
