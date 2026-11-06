"""
Physical (rest-frame) templates learned through the resolution matrix at the known redshift, with an optional nuisance group fitted jointly
in every spectrum, and an optional augmentation of the input that is scored against the clean spectrum.

Model of a spectrum in the observed frame (whitened units g = f s, omega = ivar / s^2):
    g_hat = s * ( R phi(lambda / (1 + z)) c  +  offsets  +  instrument templates  +  sky templates  +  M0 * Legendre multipliers )
phi (K, N) are the learnable rest-frame templates on a log-lambda grid (linearly interpolated at lambda_obs / (1 + z), then the 11-diagonal
resolution matrix R is applied, like Redrock's Template.eval). All coefficients of a spectrum are solved in closed form by regularised weighted
least squares with the gradient flowing to phi (DeepRock-lite). The multiplicative terms make the model bilinear; they are linearised with
M0 = the physical model of a first pass without them (two passes, M0 detached).

Nuisance group (mode "nuis"): per-camera additive offsets (3), per-camera instrument templates (b 3, r 4, z 12, trained on SKY-fibre residuals),
sky templates (DR1 VAC, K = 24; prior = 1% of the VAC coefficient scatter) and per-camera multiplicative Legendre polynomials of degree <= 2
times M0 (8 terms: the constant of camera b is dropped, b is the reference camera). Every term has a Gaussian prior (ridge) set from data or stated
assumptions (see prior_numbers). Camera terms enter the merged brz grid with a linear ramp across the two overlaps (partition of unity).
Loss: masked denoising (36 spans, ~25% hidden): fit on visible pixels, hidden-pixel chi2 per pixel against the CLEAN spectrum.
Augmentation (arms B, D): the input is multiplied by random per-camera Legendre curves, shifted by random per-camera offsets, plus a random
combination of the global K = 20 instrument templates (coefficients of a random SKY row) plus a random VAC sky spectrum at a signed amplitude.
"""
import json
import os
import time

import numpy as np
import torch

from DESIFlow.templates.camera_templates import CAMS, choose_k
from DESIFlow.templates.deeprock_lite import span_mask

L = 7781
DLOG = 3e-5
ZMIN, ZMAX = 0.02, 1.0
X0 = float(np.log10(3600.0 / (1 + ZMAX))) - 2e-3
NR = int(np.ceil((np.log10(9824.0 / (1 + ZMIN)) + 2e-3 - X0) / DLOG)) + 2
LAM_OBS = 3600.0 + 0.8 * np.arange(L)
SIG_MULT0 = {"b": None, "r": 0.015, "z": 0.015}                                                # prior sigma of the constant multiplier (b is the reference)
SIG_MULT12 = 0.01
SKY_FRAC = 0.01                                                                                 # natural amplitude of the sky injection (fraction of a VAC sky spectrum)


def camera_weights():
    l = np.arange(L, dtype=np.float64)
    wb = np.clip((2750.0 - l) / 50.0, 0, 1); wz = np.clip((l - 4900.0) / 125.0, 0, 1); wr = 1.0 - wb - wz
    return {"b": wb, "r": wr, "z": wz}


def legendre_curves(n=3):
    """{cam: (n, L)} Legendre degree 0..n-1 over the camera's own index range, times the camera weight."""
    from numpy.polynomial import legendre
    W = camera_weights(); out = {}
    for c, (a, b) in CAMS.items():
        x = np.zeros(L); x[a:b] = np.linspace(-1, 1, b - a)
        out[c] = np.stack([legendre.legval(x, [0] * k + [1]) * W[c] for k in range(n)])
    return out


def load_fixed(tdir, dev):
    """Shared additive columns (physical flux units): offsets, per-camera instrument templates, sky templates; and the multiplier shapes."""
    W = camera_weights(); cols, kinds = [], []
    for c in "brz":
        cols.append(W[c]); kinds.append(f"offset_{c}")
    for c in "brz":
        S = json.load(open(os.path.join(tdir, "instrument_cam", c, "summary.json"))); K = choose_k(S)
        t = np.load(os.path.join(tdir, "instrument_cam", c, f"K{K:03d}.npz"))["templates_flux"]
        a, b = CAMS[c]
        for k in range(K):
            v = np.zeros(L); v[a:b] = t[k]; cols.append(v * W[c]); kinds.append(f"inst_{c}")
    sk = np.load(os.path.join(tdir, "sky", "K024.npz"))["templates_flux"]
    for k in range(len(sk)):
        cols.append(sk[k]); kinds.append("sky")
    Lg = legendre_curves(3); mult, mk = [], []
    for c in "brz":
        for d in range(3):
            if c == "b" and d == 0:
                continue
            mult.append(Lg[c][d]); mk.append((c, d))
    return np.array(cols, np.float32), np.array(kinds), np.array(mult, np.float32), mk


def prior_numbers(tdir, fixed, kinds, dev):
    """sigma of every shared column in its own (flux) units: offsets and instrument templates from SKY-row coefficient scatter, sky = 1% of VAC scatter."""
    sig = np.zeros(len(kinds))
    f = np.load(os.path.join(tdir, "instrument", "train_flux.npy"), mmap_mode="r")[:20000]; w = np.load(os.path.join(tdir, "instrument", "train_ivar.npy"), mmap_mode="r")[:20000]
    f, w = torch.tensor(np.asarray(f), device=dev), torch.tensor(np.asarray(w), device=dev)
    inst = torch.tensor(fixed, device=dev)
    for g in ("offset", "inst", "sky"):
        pass
    for cam in "brz":
        a, b = CAMS[cam]; fc, wc = f[:, a:b], w[:, a:b]
        keep = (wc > 0).sum(1) > 0.5 * (b - a)
        m = ((wc * fc).sum(1) / wc.sum(1).clamp(min=1e-12))[keep]                                    # weighted mean flux of the camera segment of each SKY row
        i_off = [i for i, k in enumerate(kinds) if k == f"offset_{cam}"][0]
        sig[i_off] = float(1.4826 * (m - m.median()).abs().median())
        i_ins = [i for i, k in enumerate(kinds) if k == f"inst_{cam}"]; Xc = inst[i_ins][:, a:b]
        A = torch.einsum("ml,rl,nl->rmn", Xc, wc, Xc) + 1e-6 * torch.eye(len(i_ins), device=dev)
        c = torch.linalg.solve(A[keep].double(), torch.einsum("ml,rl->rm", Xc, wc * fc)[keep].double()[..., None])[..., 0].float()
        sd = 1.4826 * (c - c.median(0).values).abs().median(0).values
        for j, i in enumerate(i_ins):
            sig[i] = float(sd[j])
    fs, ws = (torch.tensor(np.load(os.path.join(tdir, "sky", f"train_{k}.npy")), device=dev) for k in ("flux", "ivar"))
    i_sky = [i for i, k in enumerate(kinds) if k == "sky"]; X = inst[i_sky]
    A = torch.einsum("ml,rl,nl->rmn", X, ws, X) + 1e-6 * torch.eye(len(i_sky), device=dev); bb = torch.einsum("ml,rl->rm", X, ws * fs)
    c = torch.linalg.solve(A.double(), bb.double()[..., None])[..., 0].float()
    sd = c.std(0)
    for j, i in enumerate(i_sky):
        sig[i] = SKY_FRAC * float(sd[j])
    return sig


class Model:
    def __init__(self, tdir, data_dir, dev, K=24, batch_seed=0):
        self.dev, self.K = dev, K
        d = lambda a: torch.tensor(a, device=dev)
        self.lam = d(LAM_OBS.astype(np.float32)); self.Rm = d(np.load(os.path.join(data_dir, "R_mean.npy")))
        fixed, self.kinds, mult, self.mk = load_fixed(tdir, dev)
        sig = prior_numbers(tdir, fixed, self.kinds, dev)
        self.prior_sigma = sig
        self.fixed_flux = d(fixed); self.mult = d(mult)
        self.mult_sig = d(np.array([SIG_MULT0[c] if dd == 0 else SIG_MULT12 for c, dd in self.mk], np.float32))
        self.sig_flux = d(sig.astype(np.float32))
        self.inst20 = d(np.load(os.path.join(tdir, "instrument", "K020.npz"))["templates_flux"])
        self.sky_bank = d(np.load(os.path.join(tdir, "sky", "train_flux.npy")))                 # VAC rows for sky injection
        ft, wt = (np.load(os.path.join(tdir, "instrument", f"train_{k}.npy"), mmap_mode="r")[:30000] for k in ("flux", "ivar"))
        ft, wt = d(np.asarray(ft)), d(np.asarray(wt))
        X = self.inst20; A = torch.einsum("ml,rl,nl->rmn", X, wt, X) + 1e-6 * torch.eye(20, device=dev); b = torch.einsum("ml,rl->rm", X, wt * ft)
        self.inst_bank = torch.linalg.solve(A.double(), b.double()[..., None])[..., 0].float()    # coefficients of real SKY rows on the global K=20 set
        self.curves = legendre_curves(3)
        self.sig_off = {c: float(sig[[i for i, k in enumerate(self.kinds) if k == f"offset_{c}"][0]]) for c in "brz"}
        self.s = None; self.aug_scale = 1.0; self.part_scale = {k: 1.0 for k in ("mult", "off", "inst", "sky")}

    def set_whitening(self, ivar):
        good = (ivar > 0).float(); s = torch.sqrt((ivar * good).sum(0) / good.sum(0).clamp(min=1))
        self.s = torch.where(s > 0, s, s[s > 0].median()).to(self.dev)
        rms = ((self.fixed_flux * self.s) ** 2).mean(1).sqrt()                                      # unit-rms scaling of the shared columns
        self.col_scale = rms
        self.fixed_w = self.fixed_flux * self.s / rms[:, None]
        self.prior_prec_fixed = (rms / self.sig_flux).clamp(max=1e6) ** 2
        self.mult_prec = 1.0 / self.mult_sig ** 2
        n = min(4000, len(ivar)); sk = self.sky_bank[torch.randint(0, len(self.sky_bank), (n,), device=self.dev)]
        x = float(((ivar[:n].to(self.dev) / self.s ** 2) * (sk * self.s) ** 2).mean())                           # mean chi2 per pixel of the unit-amplitude sky injection
        self.sky_frac = SKY_FRAC

    def design(self, phi, z):
        rl = torch.log10(self.lam[None, :] / (1 + z[:, None])); p = (rl - X0) / DLOG
        i0 = p.floor().long().clamp(0, NR - 2); fr = (p - i0).clamp(0, 1)
        X = phi[:, i0] * (1 - fr) + phi[:, i0 + 1] * fr                                           # (K, B, L)
        Xp = torch.nn.functional.pad(X, (5, 5))
        out = sum(self.Rm[d][None, None, :] * Xp[..., d:d + L] for d in range(11))
        return (out * self.s).permute(1, 0, 2)

    @staticmethod
    def solve(X, g, om, prec):
        Xw = X * om[:, None, :]
        A = torch.einsum("bml,bnl->bmn", Xw, X) + torch.diag(prec)[None]
        b = torch.einsum("bml,bl->bm", Xw, g)
        return torch.linalg.solve(A.double(), b.double()[..., None])[..., 0].float()

    def comp_index(self, comps):
        pref = {"off": "offset", "inst": "inst", "sky": "sky"}
        return [i for i, k in enumerate(self.kinds) if any(k.startswith(pref[c]) for c in comps if c in pref)]

    def predict(self, phi, z, g, om_vis, mode):
        """mode 'plain', 'nuis' (all components) or 'nuis:c1,c2' with components from off, inst, sky, mult."""
        A = self.design(phi, z); B = A.shape[0]; K = self.K
        pp = torch.full((K,), 1e-3, device=self.dev)
        if mode == "plain":
            th = self.solve(A, g, om_vis, pp)
            return torch.einsum("bm,bml->bl", th, A)
        comps = set(mode.split(":")[1].split(",")) if ":" in mode else {"off", "inst", "sky", "mult"}
        ix = self.comp_index(comps)
        S = self.fixed_w[ix][None].expand(B, -1, -1); pf = self.prior_prec_fixed[ix]
        X1 = torch.cat([A, S], 1) if len(ix) else A; prec1 = torch.cat([pp, pf])
        th1 = self.solve(X1, g, om_vis, prec1)
        if "mult" not in comps:
            return torch.einsum("bm,bml->bl", th1, X1)
        M0 = torch.einsum("bk,bkl->bl", th1[:, :K], A).detach()
        Mc = M0[:, None, :] * self.mult[None]
        X2 = torch.cat([X1, Mc], 1); prec2 = torch.cat([prec1, self.mult_prec])
        th2 = self.solve(X2, g, om_vis, prec2)
        return torch.einsum("bm,bml->bl", th2, X2)

    def predict_parts(self, phi, z, g, om_vis):
        """nuisance-mode prediction split into (physical part, nuisance part) in whitened units."""
        A = self.design(phi, z); B = A.shape[0]; K = self.K; pp = torch.full((K,), 1e-3, device=self.dev)
        S = self.fixed_w[None].expand(B, -1, -1)
        th1 = self.solve(torch.cat([A, S], 1), g, om_vis, torch.cat([pp, self.prior_prec_fixed]))
        M0 = torch.einsum("bk,bkl->bl", th1[:, :K], A); Mc = M0[:, None, :] * self.mult[None]
        X2 = torch.cat([A, S, Mc], 1); th2 = self.solve(X2, g, om_vis, torch.cat([pp, self.prior_prec_fixed, self.mult_prec]))
        phys = torch.einsum("bm,bml->bl", th2[:, :K], A)
        return phys, torch.einsum("bm,bml->bl", th2[:, K:], X2[:, K:])

    def augment(self, f, gen, scale=None, parts=("mult", "off", "inst", "sky")):
        """flux (B, L) in physical units -> augmented flux; amplitude of each effect is U(0, 2) times its typical size."""
        B, dev = f.shape[0], f.device
        scale = self.aug_scale if scale is None else scale
        u = lambda: torch.rand(B, 1, generator=gen, device=dev) * 2 * scale
        gain = torch.ones(B, L, device=dev)
        add = torch.zeros(B, L, device=dev)
        for c in "brz":
            cur = torch.tensor(self.curves[c], dtype=torch.float32, device=dev)
            sd = torch.tensor([0.0 if (c == "b") else SIG_MULT0[c], SIG_MULT12, SIG_MULT12], device=dev)
            a = torch.randn(B, 3, generator=gen, device=dev) * sd * u()
            gain = gain + (self.part_scale["mult"] * (a @ cur) if "mult" in parts else 0.0)
            W = torch.tensor(camera_weights()[c], dtype=torch.float32, device=dev)
            o = (torch.randn(B, 1, generator=gen, device=dev) * self.sig_off[c] * u()) * W[None]
            add = add + (self.part_scale["off"] * o if "off" in parts else 0.0)
        ci = self.inst_bank[torch.randint(0, len(self.inst_bank), (B,), generator=gen, device=dev)]
        add = add + (self.part_scale["inst"] * u() * (ci @ self.inst20) if "inst" in parts else 0.0)
        sk = self.sky_bank[torch.randint(0, len(self.sky_bank), (B,), generator=gen, device=dev)]
        add = add + (self.part_scale["sky"] * torch.randn(B, 1, generator=gen, device=dev) * self.sky_frac * u() * sk if "sky" in parts else 0.0)
        return f * gain + add


@torch.no_grad()
def aug_budget(model, data, n=2000, per_part=0.075, max_scale=4.0, seed=5):
    """Each injected effect is scaled so that it adds about `per_part` chi2 per pixel (mean of omega delta^2 over the whole spectrum, taken at the
    typical amplitude), capped at max_scale times its natural size. Natural sizes: multipliers = the measured per-camera scale scatter, offsets =
    robust scatter of SKY-row camera means, instrument = coefficients of real SKY rows on the global K=20 set, sky = 1% of a VAC spectrum."""
    f, w = data["f"][:n].to(model.dev), data["w"][:n].to(model.dev)
    model.part_scale = {k: 1.0 for k in model.part_scale}
    parts = {k: float((w * (model.augment(f, torch.Generator(device=model.dev).manual_seed(seed), parts=(k,)) - f) ** 2).mean()) for k in model.part_scale}
    model.aug_parts = parts
    model.part_scale = {k: min(max_scale, (per_part / max(v, 1e-12)) ** 0.5) for k, v in parts.items()}
    return parts, model.part_scale


def rest_init(f, w, z, K, n=6000, seed=0):
    """SVD of rest-frame resampled spectra (missing pixels filled with the weighted mean) -> (K, NR) initial templates."""
    rng = np.random.default_rng(seed); sel = rng.choice(len(f), min(n, len(f)), replace=False)
    xr = X0 + DLOG * np.arange(NR); lr = 10.0 ** xr
    F = np.zeros((len(sel), NR), np.float32); Wt = np.zeros_like(F)
    for i, r in enumerate(sel):
        lo = LAM_OBS / (1 + z[r]); g = w[r] > 0
        F[i] = np.interp(lr, lo, f[r], left=0, right=0); Wt[i] = np.interp(lr, lo, g.astype(np.float32), left=0, right=0) * (lr >= lo[0]) * (lr <= lo[-1])
    mean = (F * Wt).sum(0) / np.maximum(Wt.sum(0), 1e-3)
    F = np.where(Wt > 0.5, F, mean[None])
    _, _, Vt = np.linalg.svd(F[:2000], full_matrices=False)
    return np.vstack([mean[None] / np.linalg.norm(mean) * 10, Vt[:K - 1]]).astype(np.float32)


def train(model, data, K, mode, aug, steps, batch, lr, seed, log, phi0=None):
    dev = model.dev; gen = torch.Generator(device=dev).manual_seed(seed); gen_cpu = torch.Generator().manual_seed(seed)
    f, w, z = data["f"], data["w"], data["z"]
    phi = torch.nn.Parameter(phi0.clone() if phi0 is not None else torch.tensor(rest_init(f.numpy(), w.numpy(), z.numpy(), K), device=dev))
    if phi0 is None:
        with torch.no_grad():
            idx = torch.arange(256); A = model.design(phi, z[idx].to(dev))
            sc = (A ** 2).mean((0, 2)).sqrt()
            phi.mul_((1.0 / sc.clamp(min=1e-8))[:, None])
    opt = torch.optim.AdamW([phi], lr=lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); hist = []
    for step in range(steps):
        idx = torch.randint(0, len(f), (batch,), generator=gen_cpu)
        fc, wc, zz = f[idx].to(dev), w[idx].to(dev), z[idx].to(dev)
        fi = model.augment(fc, gen) if aug else fc
        hid = span_mask(batch, L, gen=gen, device=dev)
        g_in, g_clean = fi * model.s, fc * model.s; om = wc / model.s ** 2
        pred = model.predict(phi, zz, g_in, om * (~hid), mode)
        omh = om * hid
        loss = ((omh * (g_clean - pred) ** 2).sum(1) / (omh > 0).sum(1).clamp(min=1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([phi], 10.0); opt.step(); sched.step(); hist.append(float(loss.detach()))
        if step % 100 == 0 or step == steps - 1:
            log(f"    step {step} hidden chi2 {np.mean(hist[-50:]):.4f}")
    return phi.detach(), np.array(hist)


@torch.no_grad()
def evaluate(model, phi, data, mode, aug, chunk=256, seed=123):
    """hidden-pixel chi2 per pixel, per row, against the clean spectrum; fixed masks and augmentation draws."""
    gen = torch.Generator(device=model.dev).manual_seed(seed); out = []
    f, w, z = data["f"], data["w"], data["z"]
    for a in range(0, len(f), chunk):
        fc, wc, zz = f[a:a + chunk].to(model.dev), w[a:a + chunk].to(model.dev), z[a:a + chunk].to(model.dev)
        fi = model.augment(fc, gen) if aug else fc
        hid = span_mask(len(fc), L, gen=gen, device=model.dev)
        om = wc / model.s ** 2
        pred = model.predict(phi, zz, fi * model.s, om * (~hid), mode)
        omh = om * hid
        out.append(((omh * (fc * model.s - pred) ** 2).sum(1) / (omh > 0).sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out)


def load_data(ddir, name, dev, n=None):
    f = np.load(os.path.join(ddir, f"{name}_flux.npy")); w = np.load(os.path.join(ddir, f"{name}_ivar.npy")); z = np.load(os.path.join(ddir, f"{name}_meta.npz"))["z"]
    if n:
        f, w, z = f[:n], w[:n], z[:n]
    return {"f": torch.tensor(f), "w": torch.tensor(w), "z": torch.tensor(z)}


ARMS = {"A": ("plain", False), "B": ("plain", True), "C": ("nuis", False), "D": ("nuis", True), "E": ("nuis:inst", False), "F": ("nuis:inst,sky", False)}
EVAL_MODES = ("plain", "nuis", "nuis:inst", "nuis:inst,sky")


def main(ddir, tdir, out, K=24, steps=1500, batch=256, lr=3e-3, arms="ABCD", n_train=None, n_eval=None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    tr, ev = load_data(ddir, "train", dev, n_train), load_data(ddir, "eval", dev, n_eval)
    m = Model(tdir, ddir, dev, K=K); m.set_whitening(tr["w"])
    log(f"sky injection amplitude {m.sky_frac:.5f} of the VAC sky (cap {SKY_FRAC}); offset sigmas {m.sig_off}"); log(f"K={K} train {len(tr['f'])} eval {len(ev['f'])} on {dev}; NR={NR}; shared columns {len(m.kinds)}, multipliers {len(m.mult)}")
    parts, ps = aug_budget(m, tr); log(f"augmentation parts, natural chi2/px: {parts}; part scales: {ps}")
    res = {}
    rp = os.path.join(out, f"results_K{K}.json")
    if os.path.exists(rp):
        res = json.load(open(rp))
    for arm in arms:
        mode, aug = ARMS[arm]; t0 = time.time(); log(f"arm {arm}: train mode {mode}, augmentation {aug}")
        phi, hist = train(m, tr, K, mode, aug, steps, batch, lr, 0, log)
        r = {"seconds": time.time() - t0, "train_hist": hist[::10].tolist()}
        for em in EVAL_MODES:
            for ea in (False, True):
                v = evaluate(m, phi, ev, em, ea); r[f"eval_{em}_{'aug' if ea else 'real'}"] = [float(v.mean()), float(v.std() / np.sqrt(len(v)))]
                np.save(os.path.join(out, f"perrow_K{K}_{arm}_{em.replace(':', '-').replace(',', '+')}_{'aug' if ea else 'real'}.npy"), v.astype(np.float32))
        res[arm] = r; log(f"arm {arm} done: " + ", ".join(f"{k} {r[k][0]:.4f}" for k in r if k.startswith("eval_")))
        np.savez(os.path.join(out, f"phi_K{K}_{arm}.npz"), phi=phi.cpu().numpy(), x0=X0, dlog=DLOG, nr=NR)
        json.dump(res, open(rp, "w"))


def ablate(ddir, tdir, out, K=24, arms="ABCD", modes=("nuis:off", "nuis:inst", "nuis:sky", "nuis:mult", "nuis:off,inst", "nuis:inst,sky", "nuis:inst,mult")):
    """Evaluate trained templates under subsets of the nuisance group (real held-out data only)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, ev = load_data(ddir, "train", dev, 20000), load_data(ddir, "eval", dev)
    m = Model(tdir, ddir, dev, K=K); m.set_whitening(load_data(ddir, "train", dev)["w"]); res = {}
    for a in arms:
        phi = torch.tensor(np.load(os.path.join(out, f"phi_K{K}_{a}.npz"))["phi"], device=dev); res[a] = {}
        for md in ("plain",) + tuple(modes):
            v = evaluate(m, phi, ev, md, False); res[a][md] = [float(v.mean()), float(v.std() / np.sqrt(len(v)))]
            np.save(os.path.join(out, f"perrow_K{K}_{a}_{md.replace(':', '-').replace(',', '+')}_real.npy"), v.astype(np.float32))
            print(a, md, round(res[a][md][0], 4), flush=True)
    json.dump(res, open(os.path.join(out, f"ablation_K{K}.json"), "w"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True); ap.add_argument("--templates", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=24); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--arms", default="ABCD"); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--n-train", type=int, default=None); ap.add_argument("--n-eval", type=int, default=None); ap.add_argument("--ablate", action="store_true")
    a = ap.parse_args()
    if a.ablate:
        ablate(a.data, a.templates, a.out, a.K, a.arms); raise SystemExit
    main(a.data, a.templates, a.out, a.K, a.steps, a.batch, lr=a.lr, arms=a.arms, n_train=a.n_train, n_eval=a.n_eval)
