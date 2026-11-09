"""
Long-training K sweeps with a patience-driven annealing schedule, for five template groups:
  b, r, z   per-camera instrument templates (SKY-fibre residual rows cropped to the camera's brz-grid range)
  full      instrument templates trained across the whole 7781-pixel grid (like the K020 set)
  sky       sky templates (DR1 VAC, assumed weights)
Then the independently optimal sets are combined: the full-grid set is sliced to each camera, stacked with the camera's own set, tested for collinearity
and reduced to one orthonormal basis per camera.

Training (train_plateau): AdamW on the whitened templates, masked-denoising chi2 (36 random spans, scaled to the length of the crop); every `eval_every` steps the
chi2 of a fixed validation set (rows carved out of the training set, fixed masks) is computed; if it has not improved by more than a relative `rel_tol` for `patience`
checks the learning rate is multiplied by `factor` and the best templates so far are restored; training stops when the learning rate falls below `min_lr` or after `max_steps`;
the best-validation templates are kept. The hold-out rows used for the K choice are never seen by the schedule.

stages
  sweep    --group G --seed S : dense K sweep of one group -> ROOT/long/seed<S>/<G>/K###.npz + summary.json
  select   spline K* per group (k_selection.fit_spline / choose, paired bootstrap over hold-out rows) -> ROOT/long/k_selection.json + figure long_k_selection
  combine  slice + collinearity + orthonormal basis per camera -> ROOT/long/combined/<cam>.npz + figures long_combined_*

Run:  python -m DESIFlow.templates.long_sweep sweep|select|combine --root ROOT_V2 [--group b --seed 0] [--fig FIG --figcache CACHE]
"""
import argparse
import json
import os
import time

import numpy as np

CAMS = {"b": (0, 2751), "r": (2700, 5026), "z": (4900, 7781)}
GROUPS = {"b": ("instrument", *CAMS["b"]), "r": ("instrument", *CAMS["r"]), "z": ("instrument", *CAMS["z"]), "full": ("instrument", 0, 7781), "sky": ("sky", 0, 7781)}
K_CAM = list(range(1, 33)) + [34, 36, 40, 44, 48]
K_FULL = list(range(1, 41)) + [44, 48, 56, 64]
K_SKY = list(range(1, 33)) + list(range(34, 49, 2)) + [56, 64, 72, 80, 96, 112, 128]
KLIST = {"b": K_CAM, "r": K_CAM, "z": K_CAM, "full": K_FULL, "sky": K_SKY}


def _log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def load_group(root, group, dev, n_val):
    """whitened train (CPU), validation and eval (GPU) tensors for a group, and the pixel scale s."""
    import torch
    kind, a0, a1 = GROUPS[group]; d = os.path.join(root, "seed0", kind); L = a1 - a0
    ftr, wtr = np.load(os.path.join(d, "train_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "train_ivar.npy"), mmap_mode="r")
    sw, sn = np.zeros(L), np.zeros(L)
    for a in range(0, len(wtr), 10000):
        c = np.asarray(wtr[a:a + 10000, a0:a1]); sw += c.sum(0, dtype=np.float64); sn += (c > 0).sum(0)
    s = np.sqrt(sw / np.maximum(sn, 1)); s = np.where(s > 0, s, np.median(s[s > 0])).astype(np.float32)

    def whiten(fm, wm, device):
        G = torch.empty((len(fm), L), dtype=torch.float32, device=device); O = torch.empty_like(G)
        for a in range(0, len(fm), 5000):
            G[a:a + 5000] = torch.from_numpy(np.asarray(fm[a:a + 5000, a0:a1]) * s[None]).to(device); O[a:a + 5000] = torch.from_numpy(np.asarray(wm[a:a + 5000, a0:a1]) / s[None] ** 2).to(device)
        return G, O
    G, O = whiten(ftr, wtr, "cpu")
    perm = torch.randperm(len(G), generator=torch.Generator().manual_seed(11)); vi = perm[:n_val]; ti = perm[n_val:]
    GV, OV = G[vi].to(dev), O[vi].to(dev); G, O = G[ti].contiguous(), O[ti].contiguous()
    GE, OE = whiten(np.load(os.path.join(d, "eval_flux.npy"), mmap_mode="r"), np.load(os.path.join(d, "eval_ivar.npy"), mmap_mode="r"), dev)
    return G, O, GV, OV, GE, OE, s


def mask_kw(L):
    return dict(n_spans=max(6, int(36 * L / 7781)), len_range=(5, 250))


def train_plateau(G, O, K, dev, GV, OV, mk, batch=512, lr0=3e-3, factor=0.5, patience=4, eval_every=100, min_lr=2e-5, max_steps=10000, rel_tol=2e-4, seed=0, ridge=1e-3):
    import torch
    from DESIFlow.templates.deeprock_lite import masked_chi2, span_mask, svd_init
    L = G.shape[1]; gen = torch.Generator(device=dev).manual_seed(seed); gc = torch.Generator().manual_seed(seed)
    sub = torch.randperm(len(G), generator=torch.Generator().manual_seed(seed))[:20000]
    U = torch.nn.Parameter(svd_init(G[sub].to(dev), O[sub].to(dev), K, seed=seed))
    opt = torch.optim.AdamW([U], lr=lr0, weight_decay=0.0)
    hidV = span_mask(len(GV), L, gen=torch.Generator(device=dev).manual_seed(7), device=dev, **mk)
    best, bestU, bad, lr, hist = float("inf"), U.detach().clone(), 0, lr0, []

    def val():
        with torch.no_grad():
            return float(torch.cat([masked_chi2(U, GV[a:a + 250], OV[a:a + 250], hidV[a:a + 250], ridge)[0] for a in range(0, len(GV), 250)]).mean())
    step = 0
    for step in range(max_steps):
        idx = torch.randint(0, len(G), (batch,), generator=gc); g, o = G[idx].to(dev, non_blocking=True), O[idx].to(dev, non_blocking=True)
        hid = span_mask(batch, L, gen=gen, device=dev, **mk)
        loss = masked_chi2(U, g, o, hid, ridge)[0].mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % eval_every == 0:
            v = val(); hist.append((step + 1, lr, v))
            if v < best * (1 - rel_tol):
                best, bestU, bad = v, U.detach().clone(), 0
            else:
                bad += 1
                if bad >= patience:
                    lr *= factor; bad = 0
                    with torch.no_grad():
                        U.copy_(bestU)
                    for gr in opt.param_groups:
                        gr["lr"] = lr
                    if lr < min_lr:
                        break
    return bestU, np.array(hist), step + 1


def sweep(root, group, seed, ks=None, n_val=3000, **kw):
    import torch
    from DESIFlow.templates.deeprock_lite import evaluate, order_orthonormal
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); kind, a0, a1 = GROUPS[group]; L = a1 - a0
    n_val = 500 if group == "sky" else n_val
    G, O, GV, OV, GE, OE, s = load_group(root, group, dev, n_val); mk = mask_kw(L)
    od = os.path.join(root, "long", f"seed{seed}", group); os.makedirs(od, exist_ok=True); sp = os.path.join(od, "summary.json")
    summary = {int(k): v for k, v in json.load(open(sp)).items()} if os.path.exists(sp) else {}
    sub = torch.randperm(len(G), generator=torch.Generator().manual_seed(3))[:20000]; Gs, Os = G[sub].to(dev), O[sub].to(dev)
    for K in (ks or KLIST[group]):
        if K in summary:
            continue
        t0 = time.time(); U, hist, steps = train_plateau(G, O, K, dev, GV, OV, mk, seed=seed, **kw)
        B, S = order_orthonormal(U, Gs, Os); hid, base, vis = evaluate(B, GE, OE, mask_kw=mk)
        summary[K] = dict(hidden_chi2=float(hid.mean()), hidden_chi2_sem=float(hid.std() / np.sqrt(len(hid))), baseline_chi2=float(base.mean()), visible_chi2=float(vis.mean()), val_chi2=float(hist[:, 2].min()) if len(hist) else None, steps=steps, seconds=time.time() - t0)
        np.savez(os.path.join(od, f"K{K:03d}.npz"), templates_whitened=B.cpu().numpy(), templates_flux=B.cpu().numpy() / s[None], pixel_scale_s=s, energy=S.cpu().numpy(), K=K, brz_index_range=np.array([a0, a1]),
                 hidden_chi2_per_row_eval=hid.astype(np.float32), baseline_per_row_eval=base.astype(np.float32), lr_history=hist, **summary[K])
        json.dump(summary, open(sp, "w"), indent=1)
        _log(f"{group} seed {seed} K={K}: hold-out hidden chi2 {summary[K]['hidden_chi2']:.4f} +- {summary[K]['hidden_chi2_sem']:.4f} (zero {summary[K]['baseline_chi2']:.4f}), {steps} steps, {summary[K]['seconds']:.0f} s")
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- selection
def select(root, seeds, figdir, cachedir, nboot=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from DESIFlow.templates.k_selection import choose, fit_spline
    rng = np.random.default_rng(0); out, cache = {}, {}
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, g in zip(ax.ravel(), ["b", "r", "z", "full", "sky"]):
        S = {s: {int(k): v for k, v in json.load(open(os.path.join(root, "long", f"seed{s}", g, "summary.json"))).items()} for s in seeds}
        K = np.array(sorted(set.intersection(*[set(v) for v in S.values()])))
        chi = np.array([[S[s][k]["hidden_chi2"] for k in K] for s in seeds]); sem = np.array([[S[s][k]["hidden_chi2_sem"] for k in K] for s in seeds])
        m = chi.mean(0); sd = np.sqrt((sem ** 2).mean(0) + (chi.var(0, ddof=1) / len(seeds) if len(seeds) > 1 else 0)); log_y = g == "sky"
        Kd, yd, _ = fit_spline(K, m, sd, log_y); kstar, how = choose(Kd, yd, log_y)
        rows = np.array([np.load(os.path.join(root, "long", f"seed{seeds[0]}", g, f"K{k:03d}.npz"))["hidden_chi2_per_row_eval"] for k in K]); boots = []
        for _ in range(nboot):
            ii = rng.integers(0, rows.shape[1], rows.shape[1]); Kb, yb, _ = fit_spline(K, rows[:, ii].mean(1) + (m - rows.mean(1)), sd, log_y); boots.append(choose(Kb, yb, log_y)[0])
        lo, hi = np.percentile(boots, [16, 84]); kn = int(K[np.argmin(np.abs(K - kstar))])
        out[g] = dict(K_star=kstar, K_grid=kn, rule=how, boot_16=float(lo), boot_84=float(hi), chi2_at_kstar=float(np.interp(kstar, Kd, yd)), best_raw_K=int(K[m.argmin()]), best_raw_chi2=float(m.min()), baseline=float(S[seeds[0]][int(K[0])]["baseline_chi2"]),
                      steps_median=float(np.median([S[seeds[0]][int(k)]["steps"] for k in K])))
        for i, s in enumerate(seeds):
            a.plot(K, chi[i], ".", ms=4, alpha=0.5, label=f"seed {s}")
        a.errorbar(K, m, sd, fmt="k.", ms=4, lw=0.7, label="mean +- error"); a.plot(Kd, yd, "tab:red", lw=1.4, label="cubic B-spline in ln K"); a.axvline(kstar, color="tab:red", ls=":", label=f"K* = {kstar:.1f} ({how}; 16-84%: {lo:.0f}-{hi:.0f})")
        a.set_xscale("log"); a.set_xlabel("number of templates K"); a.set_ylabel("hold-out hidden-pixel chi2 per pixel"); a.set_title({"full": "instrument templates, full wavelength grid", "sky": "sky templates"}.get(g, f"camera {g} instrument templates"), fontsize=10)
        if log_y: a.set_yscale("log")
        else: a.set_ylim(m.min() - 0.003, m.min() + 0.05 * (out[g]["baseline"] - m.min()) + 0.015)
        a.legend(fontsize=7); cache.update({f"{g}_K": K, f"{g}_chi2": chi, f"{g}_sem": sem, f"{g}_spline_K": Kd, f"{g}_spline": yd, f"{g}_boot": np.array(boots)})
        print(g, out[g])
    ax[1, 2].axis("off"); fig.tight_layout(); os.makedirs(figdir, exist_ok=True); os.makedirs(cachedir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "long_k_selection.png"), dpi=120); np.savez(os.path.join(cachedir, "long_k_selection.npz"), **cache)
    json.dump(out, open(os.path.join(root, "long", "k_selection.json"), "w"), indent=1)


# ---------------------------------------------------------------------------------------------- combine
def combine(root, seed, figdir, cachedir, tol=1e-3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from DESIFlow.templates.deeprock_lite import evaluate, order_orthonormal
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); ks = json.load(open(os.path.join(root, "long", "k_selection.json")))
    os.makedirs(os.path.join(root, "long", "combined"), exist_ok=True); res, cache = {}, {}
    Kf = ks["full"]["K_grid"]; zf = np.load(os.path.join(root, "long", f"seed{seed}", "full", f"K{Kf:03d}.npz")); Tf = zf["templates_whitened"].astype(np.float64)
    fig, ax = plt.subplots(3, 3, figsize=(18, 13))
    for j, c in enumerate("brz"):
        a0, a1 = CAMS[c]; L = a1 - a0; Kc = ks[c]["K_grid"]; zc = np.load(os.path.join(root, "long", f"seed{seed}", c, f"K{Kc:03d}.npz")); Bc = zc["templates_whitened"].astype(np.float64); s = zc["pixel_scale_s"]
        F = Tf[:, a0:a1]; Qf = np.linalg.qr(F.T)[0]; cosang = np.linalg.svd(Bc @ Qf, compute_uv=False)                     # cosines of the principal angles between the two spans
        S = np.vstack([Bc, F / np.linalg.norm(F, axis=1, keepdims=True)]); sv = np.linalg.svd(S, compute_uv=False); U_, sv_, Vt = np.linalg.svd(S, full_matrices=False)
        keep = sv_ > tol * sv_[0]; V = Vt[keep]
        # evaluation sets
        G, O, GV, OV, GE, OE, s2 = load_group(root, c, dev, 500); mk = mask_kw(L); del G, O
        Gs = GE[:20000]; Os = OE[:20000]
        sets = {"per-camera": torch.tensor(Bc, dtype=torch.float32, device=dev), "full grid, sliced": torch.tensor(np.linalg.qr(F.T)[0].T.copy(), dtype=torch.float32, device=dev), "combined orthonormal": torch.tensor(V.copy(), dtype=torch.float32, device=dev)}
        Bo, Eo = order_orthonormal(sets["combined orthonormal"], Gs, Os); sets["combined orthonormal"] = Bo
        hs = {}
        for nm, B in sets.items():
            hs[nm] = evaluate(B, GE, OE, mask_kw=mk)[0]
        res[c] = dict(K_cam=Kc, K_full=Kf, dim_combined=int(keep.sum()), cond_stack=float(sv[0] / sv[-1]), n_small_sv=int((sv < tol * sv[0]).sum()), cos_gt_09=int((cosang > 0.9).sum()), cos_gt_099=int((cosang > 0.99).sum()), mean_cos2=float((cosang ** 2).mean()),
                      chi2={nm: float(h.mean()) for nm, h in hs.items()}, sem={nm: float(h.std() / np.sqrt(len(h))) for nm, h in hs.items()}, paired_combined_minus_cam=float((hs["combined orthonormal"] - hs["per-camera"]).mean()), paired_sem=float((hs["combined orthonormal"] - hs["per-camera"]).std() / np.sqrt(len(hs["per-camera"]))),
                      baseline=float(zc["baseline_chi2"]))
        np.savez(os.path.join(root, "long", "combined", f"{c}.npz"), templates_whitened=Bo.cpu().numpy(), templates_flux=Bo.cpu().numpy() / s[None], pixel_scale_s=s, energy=Eo.cpu().numpy(), brz_index_range=np.array([a0, a1]), **{k: v for k, v in res[c].items() if not isinstance(v, dict)})
        ax[0, j].plot(np.sort(cosang)[::-1], "o-"); ax[0, j].set_xlabel("principal-angle rank"); ax[0, j].set_ylabel("cosine between the per-camera span and the sliced full-grid span"); ax[0, j].set_title(f"camera {c}: K_cam={Kc}, K_full={Kf}", fontsize=9); ax[0, j].axhline(0.99, color="gray", ls=":")
        ax[1, j].semilogy(sv / sv[0], "o-", ms=3); ax[1, j].axhline(tol, color="tab:red", ls="--", label=f"tolerance {tol}"); ax[1, j].set_xlabel("singular value index of the stacked (unit-norm rows) set"); ax[1, j].set_ylabel("singular value / largest"); ax[1, j].legend(fontsize=8)
        nm = list(sets); m_ = [hs[n].mean() for n in nm]; e_ = [hs[n].std() / np.sqrt(len(hs[n])) for n in nm]
        ax[2, j].bar(range(3), m_, yerr=e_, color=["tab:blue", "tab:orange", "tab:green"]); ax[2, j].set_xticks(range(3)); ax[2, j].set_xticklabels([f"{n}\n({B.shape[0]} dims)" for n, B in sets.items()], fontsize=8); ax[2, j].set_ylim(min(m_) - 0.01, max(m_) + 0.01)
        ax[2, j].axhline(float(zc["baseline_chi2"]), color="gray", ls="--"); ax[2, j].set_ylabel("hold-out hidden chi2 per pixel")
        cache.update({f"{c}_cos": cosang, f"{c}_sv": sv, **{f"{c}_chi2_per_row_{n.replace(' ', '_').replace(',', '')}": h for n, h in hs.items()}})
        print(c, res[c])
    fig.suptitle("Combining the independently optimal per-camera and full-grid instrument sets: collinearity and hold-out score", fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(figdir, "long_combined_collinearity.png"), dpi=120); np.savez(os.path.join(cachedir, "long_combined_collinearity.npz"), **cache)
    json.dump(res, open(os.path.join(root, "long", "combined", "results.json"), "w"), indent=1)


def add_dims(root, seed, figdir, cachedir, ms=(0, 1, 2, 3, 4, 6, 8, 12, 19)):
    """How many of the sliced full-grid directions are worth adding to a camera's own set? The sliced full-grid templates are made orthogonal to the camera's span, ordered by
    their energy on the training data, and the first m are added; hold-out hidden chi2 against m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from DESIFlow.templates.deeprock_lite import evaluate, order_orthonormal
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); ks = json.load(open(os.path.join(root, "long", "k_selection.json"))); Kf = ks["full"]["K_grid"]
    Tf = np.load(os.path.join(root, "long", f"seed{seed}", "full", f"K{Kf:03d}.npz"))["templates_whitened"].astype(np.float64); res, cache = {}, {}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for j, c in enumerate("brz"):
        a0, a1 = CAMS[c]; L = a1 - a0; Kc = ks[c]["K_grid"]; Bc = np.load(os.path.join(root, "long", f"seed{seed}", c, f"K{Kc:03d}.npz"))["templates_whitened"].astype(np.float64)
        F = Tf[:, a0:a1]; Fp = F - (F @ Bc.T) @ Bc; sv = np.linalg.svd(Fp, compute_uv=False); Q, Rr = np.linalg.qr(Fp.T); keep = np.abs(np.diag(Rr)) > 1e-3 * np.abs(Rr).max(); W = Q[:, keep].T
        G, O, GV, OV, GE, OE, s = load_group(root, c, dev, 500); mk = mask_kw(L); del G, O
        Wt = torch.tensor(W.copy(), dtype=torch.float32, device=dev); Wo, _ = order_orthonormal(Wt, GE[:20000], OE[:20000])
        chi, sem = [], []
        for m in ms:
            m = min(m, len(Wo)); B = torch.cat([torch.tensor(Bc, dtype=torch.float32, device=dev), Wo[:m]], 0); h = evaluate(B, GE, OE, mask_kw=mk)[0]; chi.append(float(h.mean())); sem.append(float(h.std() / np.sqrt(len(h))))
            if m == 0: h0 = h
            else: sem[-1] = float((h - h0).std() / np.sqrt(len(h)))                     # paired error against m = 0
        ax[j].errorbar(ms, [chi[0]] + [chi[0] + (x - chi[0]) for x in chi[1:]], [0] + sem[1:], marker="o"); cache[f"{c}_m"] = np.array(ms); cache[f"{c}_chi2"] = np.array(chi); cache[f"{c}_paired_sem"] = np.array(sem)
        ax[j].set_xlabel("number m of added full-grid directions (orthogonal to the camera's set, ordered by energy)"); ax[j].set_ylabel("hold-out hidden chi2 per pixel (error bar = paired error against m = 0)"); ax[j].set_title(f"camera {c}: own set K={Kc}, {len(Wo)} independent full-grid directions", fontsize=9)
        res[c] = dict(K_cam=Kc, n_independent_full=int(len(Wo)), chi2=dict(zip(map(str, ms), chi)), paired_sem=dict(zip(map(str, ms), sem)))
        print(c, res[c])
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "long_combined_added_dims.png"), dpi=120); np.savez(os.path.join(cachedir, "long_combined_added_dims.npz"), **cache); json.dump(res, open(os.path.join(root, "long", "combined", "added_dims.json"), "w"), indent=1)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["sweep", "select", "combine", "adddims"]); ap.add_argument("--root", required=True); ap.add_argument("--group", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0"); ap.add_argument("--ks", default=None); ap.add_argument("--fig", default=None); ap.add_argument("--figcache", default=None)
    ap.add_argument("--max-steps", type=int, default=10000); ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "sweep":
        sweep(a.root, a.group, a.seed, ks=[int(x) for x in a.ks.split(",")] if a.ks else None, max_steps=a.max_steps, patience=a.patience)
    elif a.stage == "select":
        select(a.root, [int(x) for x in a.seeds.split(",")], a.fig, a.figcache)
    elif a.stage == "adddims":
        add_dims(a.root, a.seed, a.fig, a.figcache)
    else:
        combine(a.root, a.seed, a.fig, a.figcache)
