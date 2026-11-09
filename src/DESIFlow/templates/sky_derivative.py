"""
Exploratory: do derivative terms let the sky templates be much smaller?

Sky lines of a spectrum are shifted, narrowed or broadened relative to the training set, which a fixed set of line templates can only follow by spending
templates. The shift and width changes of the model S0 = c.T (first pass, c = WLS coefficients) are described by its derivatives: S(lambda + d) ~ S0 + d S0',
width change ~ (w / 2) S0''. Second pass: refit [templates, S0' * P_j (shift, Legendre degree 0..2 per camera: 9 terms), S0'' * P_j (width, degree 0..1 per camera:
6 terms)] with Gaussian priors on the 15 derivative coefficients (shift sigma 0.07 px, width 0.03 px^2 (the scatter of the unregularised coefficients)). The derivative columns are built from the first-pass model,
detached (the same linearisation as the multiplicative terms of physical_templates).

stages
  eval    hold-out hidden-pixel chi2 (fixed masks) of the saved sky templates (ROOT/seed0/sky/K###.npz) with and without the derivative terms
  train   templates retrained with the derivative terms in the loop (fresh, 1500 steps), scored with the derivative terms
  figure  chi2 against K for the three cases

Run:  python -m DESIFlow.templates.sky_derivative eval|train|figure --root ROOT --fig FIG --figcache CACHE [--ks ...]
"""
import argparse
import json
import os
import time

import numpy as np

SIG_SHIFT, SIG_WIDTH = 0.07, 0.03          # px, px^2: the rms of the unregularised coefficients on held-out sky spectra is 0.04-0.09 px (shift) and 0.01-0.05 (width)
KS_DEFAULT = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]


def curves(dev):
    import torch
    from DESIFlow.templates.physical_templates import legendre_curves
    Lg = legendre_curves(3)
    P1 = torch.tensor(np.stack([Lg[c][d] for c in "brz" for d in range(3)]), dtype=torch.float32, device=dev)
    P2 = torch.tensor(np.stack([Lg[c][d] for c in "brz" for d in range(2)]), dtype=torch.float32, device=dev)
    prec = torch.cat([torch.full((9,), 1.0 / SIG_SHIFT ** 2), torch.full((6,), 1.0 / SIG_WIDTH ** 2)]).to(dev)
    return P1, P2, prec


def solve(X, g, om, prec):
    import torch
    Xw = X * om[:, None, :]
    A = torch.einsum("bml,bnl->bmn", Xw, X) + torch.diag(prec)[None]
    return torch.linalg.solve(A.double(), torch.einsum("bml,bl->bm", Xw, g).double()[..., None])[..., 0].float()


def fit_predict(U, s, g, om, ridge, P=None, mode="plain"):
    """prediction of the whitened data g (weights om; hidden pixels already zeroed) with templates U (K, L); mode 'plain' or 'deriv'."""
    import torch
    K, B = U.shape[0], g.shape[0]; Ub = U[None].expand(B, -1, -1); pr = torch.full((K,), ridge, device=U.device)
    c = solve(Ub, g, om, pr)
    if mode == "plain":
        return torch.einsum("bk,kl->bl", c, U)
    P1, P2, prec = P
    with torch.no_grad():
        S0 = torch.einsum("bk,kl->bl", c, U) / s
        D1 = torch.zeros_like(S0); D1[:, 1:-1] = (S0[:, 2:] - S0[:, :-2]) / 2
        D2 = torch.zeros_like(S0); D2[:, 1:-1] = S0[:, 2:] - 2 * S0[:, 1:-1] + S0[:, :-2]
        cols = torch.cat([D1[:, None, :] * P1[None], D2[:, None, :] * P2[None]], 1) * s
    X = torch.cat([Ub, cols], 1); th = solve(X, g, om, torch.cat([pr, prec]))
    return torch.einsum("bm,bml->bl", th, X)


def load(root, dev, src="seed0"):
    import torch
    d = os.path.join(root, src, "sky")
    out = {}
    for name in ("train", "eval"):
        f = np.load(os.path.join(d, f"{name}_flux.npy")); w = np.load(os.path.join(d, f"{name}_ivar.npy")); out[name] = (torch.tensor(f, device=dev), torch.tensor(w, device=dev))
    good = (out["train"][1] > 0).float(); s = torch.sqrt((out["train"][1] * good).sum(0) / good.sum(0).clamp(min=1)); s = torch.where(s > 0, s, s[s > 0].median())
    return out, s


def evaluate(U, s, data, mode, P, ridge=1e-3, chunk=256, seed=123):
    import torch
    from DESIFlow.templates.deeprock_lite import span_mask
    f, w = data; gen = torch.Generator(device=U.device).manual_seed(seed); out = []
    with torch.no_grad():
        for a in range(0, len(f), chunk):
            g = f[a:a + chunk] * s; om = w[a:a + chunk] / s ** 2; hid = span_mask(len(g), g.shape[1], gen=gen, device=U.device)
            pred = fit_predict(U, s, g, om * (~hid), ridge, P, mode); omh = om * hid
            out.append(((omh * (g - pred) ** 2).sum(1) / (omh > 0).sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def stage_eval(root, ks):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); data, s = load(root, dev); P = curves(dev); res = {}
    for K in ks:
        U = torch.tensor(np.load(os.path.join(root, "seed0", "sky", f"K{K:03d}.npz"))["templates_whitened"], device=dev)
        a = evaluate(U, s, data["eval"], "plain", P); b = evaluate(U, s, data["eval"], "deriv", P)
        res[K] = dict(plain=float(a.mean()), plain_sem=float(a.std() / len(a) ** 0.5), deriv=float(b.mean()), deriv_sem=float(b.std() / len(b) ** 0.5), paired=float((b - a).mean()), paired_sem=float((b - a).std() / len(a) ** 0.5))
        print(f"K={K}: plain {res[K]['plain']:.4f} -> with derivative terms {res[K]['deriv']:.4f}", flush=True)
    p = os.path.join(root, "sky_derivative_eval.json"); old = json.load(open(p)) if os.path.exists(p) else {}
    old.update({str(k): v for k, v in res.items()}); json.dump(old, open(p, "w"), indent=1)


def stage_train(root, ks, steps=1500, batch=512, lr=3e-3, seed=0, warm=False, tag="", mode="deriv"):
    import torch
    from DESIFlow.templates.deeprock_lite import order_orthonormal, span_mask, svd_init
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); data, s = load(root, dev); P = curves(dev)
    f, w = data["train"]; G = f * s; O = w / s ** 2; res = {}; os.makedirs(os.path.join(root, "sky_derivative" + tag), exist_ok=True)
    for K in ks:
        t0 = time.time(); gen = torch.Generator(device=dev).manual_seed(seed)
        U = torch.nn.Parameter(torch.tensor(np.load(os.path.join(root, "seed0", "sky", f"K{K:03d}.npz"))["templates_whitened"], device=dev) if warm else svd_init(G, O, K, seed=seed)); opt = torch.optim.AdamW([U], lr=lr, weight_decay=0.0); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for it in range(steps):
            idx = torch.randint(0, len(G), (min(batch, len(G)),), generator=gen, device=dev); g, om = G[idx], O[idx]
            hid = span_mask(len(g), g.shape[1], gen=gen, device=dev); pred = fit_predict(U, s, g, om * (~hid), 1e-3, P, mode); omh = om * hid
            loss = ((omh * (g - pred) ** 2).sum(1) / (omh > 0).sum(1).clamp(min=1)).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        B_, S_ = order_orthonormal(U.detach(), G, O)
        a = evaluate(B_, s, data["eval"], "deriv", P); b = evaluate(B_, s, data["eval"], "plain", P)
        res[K] = dict(deriv_trained_deriv_eval=float(a.mean()), deriv_trained_deriv_sem=float(a.std() / len(a) ** 0.5), deriv_trained_plain_eval=float(b.mean()), seconds=time.time() - t0)
        np.savez(os.path.join(root, "sky_derivative" + tag, f"K{K:03d}.npz"), templates_whitened=B_.cpu().numpy(), templates_flux=B_.cpu().numpy() / s.cpu().numpy()[None], energy=S_.cpu().numpy(), **res[K])
        print(f"train K={K}: deriv-trained, deriv-eval {res[K]['deriv_trained_deriv_eval']:.4f}, plain eval {res[K]['deriv_trained_plain_eval']:.4f}  {res[K]['seconds']:.0f}s", flush=True)
        p = os.path.join(root, f"sky_derivative_train{tag}.json"); old = json.load(open(p)) if os.path.exists(p) else {}; old.update({str(k): v for k, v in res.items()}); json.dump(old, open(p, "w"), indent=1)


def stage_figure(root, figdir, cachedir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = {int(k): v for k, v in json.load(open(os.path.join(root, "sky_derivative_eval.json"))).items()}
    tr = {int(k): v for k, v in json.load(open(os.path.join(root, "sky_derivative_train.json"))).items()} if os.path.exists(os.path.join(root, "sky_derivative_train.json")) else {}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5)); K = np.array(sorted(ev)); cache = {"K_eval": K}
    ax[0].errorbar(K, [ev[k]["plain"] for k in K], [ev[k]["plain_sem"] for k in K], marker="o", label="plain-trained templates, plain fit")
    ax[0].errorbar(K, [ev[k]["deriv"] for k in K], [ev[k]["deriv_sem"] for k in K], marker="s", label="plain-trained templates + derivative terms at fit time")
    cache["plain"] = np.array([ev[k]["plain"] for k in K]); cache["deriv_eval_only"] = np.array([ev[k]["deriv"] for k in K])
    if tr:
        Kt = np.array(sorted(tr)); ax[0].errorbar(Kt, [tr[k]["deriv_trained_deriv_eval"] for k in Kt], [tr[k]["deriv_trained_deriv_sem"] for k in Kt], marker="^", label="templates trained with derivative terms, scored with them")
        cache["K_train"] = Kt; cache["deriv_trained"] = np.array([tr[k]["deriv_trained_deriv_eval"] for k in Kt])
    for tag, lab, mk in (("_warm", "plain-trained, then 600 steps with derivative terms in the loop (lr 3e-4), scored with them", "v"), ("_warm_plaincontrol", "control: plain-trained, then 600 more plain steps, plain fit", "x")):
        pth = os.path.join(root, f"sky_derivative_train{tag}.json")
        if os.path.exists(pth):
            w = {int(k): v for k, v in json.load(open(pth)).items()}; Kw = np.array(sorted(w)); key = "deriv_trained_deriv_eval" if tag == "_warm" else "deriv_trained_plain_eval"
            ax[0].plot(Kw, [w[k][key] for k in Kw], marker=mk, ls="--", label=lab); cache[f"K{tag}"] = Kw; cache[f"chi2{tag}"] = np.array([w[k][key] for k in Kw])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("number of sky templates K"); ax[0].set_ylabel("hold-out hidden-pixel chi2 per pixel (assumed weights)"); ax[0].legend(fontsize=8); ax[0].axhline(0.348, color="gray", ls=":", lw=0.8)
    ax[0].set_title("sky templates: effect of line shift / width terms (dotted: plain K=108 level)")
    ax[1].errorbar(K, [ev[k]["paired"] for k in K], [ev[k]["paired_sem"] for k in K], marker="o"); ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xscale("log"); ax[1].set_xlabel("number of sky templates K")
    ax[1].set_ylabel("paired change of hold-out chi2 per pixel from the derivative terms (negative = better)"); ax[1].set_title("same rows, same hidden spans")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "sky_derivative_terms.png"), dpi=130); np.savez(os.path.join(cachedir, "sky_derivative_terms.npz"), **cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["eval", "train", "figure"]); ap.add_argument("--root", required=True); ap.add_argument("--fig", default=None); ap.add_argument("--figcache", default=None)
    ap.add_argument("--ks", default=",".join(map(str, KS_DEFAULT))); ap.add_argument("--warm", action="store_true"); ap.add_argument("--mode", default="deriv"); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args(); ks = [int(x) for x in a.ks.split(",")]
    {"eval": lambda: stage_eval(a.root, ks), "train": lambda: stage_train(a.root, ks, steps=a.steps, lr=a.lr, warm=a.warm, tag=("_warm" if a.warm else "") + ("_plaincontrol" if a.mode == "plain" else ""), mode=a.mode), "figure": lambda: stage_figure(a.root, a.fig, a.figcache)}[a.stage]()
