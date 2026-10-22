"""
Clean physics-linked synthetic DESI-like spectra on the native 7781-pixel grid (3600-9824 A, 0.8 A): the noise-free signals of the fidelity study (`metric.fidelity`).

Ported from the benchmark generator (scripts/narrow_bench/phys/physspec.py, clean part only). Every object draws a handful of PHYSICAL LATENTS and every feature follows from them with small scatter:

  ELG  mass (-> gas metallicity, velocity dispersion, dust), youth (-> EW, continuum slope, 4000 A break, density, Balmer absorption), AGN fraction and type; line ratios on the
       star-forming locus mixed with an AGN locus, fixed doublets, [OII] / [SII] density ratios; all lines share z and one gas velocity dispersion.
  LRG  age, velocity dispersion (-> [Z/H], alpha/Fe), dust: Lick-like absorption EWs and the 4000 A break move together; weak LINER emission when young.
  QSO  luminosity (correlated with z), EV1 (broad-line width, CIV blueshift, FeII vs narrow [OIII]), Baldwin effect, BAL troughs, narrow MgII absorbers.

Not simulated: stars, BGS-specific types, the Lyman-alpha forest (QSOs at z > 2.5 have a clean continuum blueward of Lyman alpha). Spectra are normalised so the median continuum in 4000-9500 A
(observed) is 1 and are strictly positive. The noise lives in `metric.fidelity` (`degrade`).
"""
import math
import torch

C = 299792.458
N_PIX = 7781
LAM0, LAM1 = 3600.0, 9824.0
KMAX = 24
CLASS_NAMES = ("ELG", "LRG", "QSO")
# natural redshift reach of the generators: a class is drawn in a band only when the band overlaps it (ELG z < 1.7, LRG z < 1.2, QSO z > 0.7)
CLASS_REACH = {"ELG": (0.0, 1.7), "LRG": (0.0, 1.2), "QSO": (0.7, 99.0)}


def wave_grid(dev="cpu"):
    return LAM0 + 0.8 * torch.arange(N_PIX, device=dev, dtype=torch.float32)


def kdust(lam):
    return 4.05 * (5500.0 / lam) ** 1.2


def sig_inst(lam):
    R = (2000.0 + (lam - LAM0) / (LAM1 - LAM0) * 3500.0).clamp_min(1000.0)
    return C / (2.3548 * R)


def render(wave, mu, sig_v, flux):
    """Gaussian lines: mu (B,K) observed centre (A), sig_v (B,K) intrinsic velocity sigma (km/s), flux (B,K) integrated flux (signed). -> (B,N), sigma_obs (B,K)."""
    mu = mu.clamp_min(1.0)
    sa = (mu * torch.sqrt(sig_v ** 2 + sig_inst(mu) ** 2) / C).clamp_min(0.05)
    d = (wave.view(1, 1, -1) - mu.unsqueeze(-1)) / sa.unsqueeze(-1)
    prof = torch.exp(-0.5 * d * d) / (math.sqrt(2 * math.pi) * sa.unsqueeze(-1))
    return (flux.unsqueeze(-1) * prof).sum(1), sa


class Rng:
    def __init__(self, g, dev):
        self.g, self.dev = g, dev

    def r(self, *s):
        return torch.rand(*s, device=self.dev, generator=self.g)

    def n(self, *s):
        return torch.randn(*s, device=self.dev, generator=self.g)


def _pad(t, k):
    return torch.cat([t, torch.zeros(t.shape[0], k - t.shape[1], device=t.device, dtype=t.dtype)], 1) if t.shape[1] < k else t


def _finish(wave, z, cont_fn, lam_rest, ew, sig):
    """lam_rest (B,K) rest wavelength of each feature, ew (B,K) signed rest EW (A; emission > 0), sig (B,K) velocity sigma.
    -> clean, cont (normalised so the median continuum in 4000-9500 A observed is 1), mu (B,KMAX), peak (B,KMAX) relative peak height."""
    zf = (1 + z).unsqueeze(1)
    cont = cont_fn(wave.unsqueeze(0) / zf)
    c_line = cont_fn(lam_rest)
    mu = lam_rest * zf
    inr = ((mu > LAM0 - 30) & (mu < LAM1 + 30)).to(mu.dtype)
    flux = ew * zf * c_line * inr
    lines, sa = render(wave, mu, sig, flux)
    peak = flux.abs() / (math.sqrt(2 * math.pi) * sa) / c_line.clamp_min(1e-6)
    sel = (wave > 4000) & (wave < 9500)
    norm = torch.where(sel.unsqueeze(0), cont, torch.full_like(cont, float("nan"))).nanmedian(1).values.unsqueeze(1).clamp_min(1e-6)
    clean = torch.maximum(cont + lines, 0.05 * cont)                                 # absorption cannot go below ~5% of the continuum
    return clean / norm, cont / norm, _pad(mu * inr, KMAX), _pad(peak * inr, KMAX)


# --------------------------------------------------------------------------------------------------------------------------------------------
def gen_elg(n, rg, wave, zr=(0.2, 1.6)):
    r, rn = rg.r, rg.n
    z = zr[0] + (zr[1] - zr[0]) * r(n)
    logM = rn(n)
    m = torch.sigmoid(1.2 * logM + 0.6 * rn(n))                                   # gas metallicity 0..1 (mass-metallicity relation)
    y = r(n)                                                                     # youth / sSFR
    sig = 10 ** (1.85 + 0.16 * logM + 0.05 * rn(n))                              # gas velocity dispersion, km/s
    E = 0.03 + 0.6 * torch.sigmoid(0.9 * logM + 1.0 * (y - 0.5) + 0.6 * rn(n) - 0.6)   # nebular reddening
    ne = 10 ** (1.6 + 0.9 * y + 0.25 * rn(n))                                    # electron density, cm^-3
    a = (r(n) < 0.25).float() * (0.05 + 0.75 * r(n))                             # AGN Hb fraction
    u = r(n)                                                                     # 0 = LINER-like, 1 = Seyfert-like
    ew_hb = 10 ** (0.5 + 0.9 * y + 0.12 * rn(n))                                 # rest-frame Hb EW (A)
    beta = -1.8 + 1.5 * (1 - y) + 0.2 * rn(n)
    d4 = (0.28 * (1 - y) ** 1.5 + 0.03 * rn(n)).clamp(0, 0.5)
    Ec = 0.44 * E

    def cont_fn(lam):
        c = (lam / 4500.0) ** beta.unsqueeze(1) * (1 - d4.unsqueeze(1) / (1 + torch.exp(-(4000.0 - lam) / 60.0)))
        return c * 10 ** (-0.4 * Ec.unsqueeze(1) * (kdust(lam) - kdust(5500.0)))

    xn = -1.45 + 1.15 * m + 0.05 * rn(n)                                         # log [NII]/Ha (SF locus)
    lo3 = 0.61 / (xn - 0.05) + 1.3 + 0.12 * (y - 0.5) + 0.05 * rn(n)             # log [OIII]/Hb (Kewley+01 boundary)
    sf = dict(Ha=2.86 + 0 * m, NII=10 ** xn * 2.86, SII=10 ** (-0.45 - 0.25 * (xn + 1.45) + 0.07 * rn(n)) * 2.86, OI=10 ** (-1.7 + 0.2 * (xn + 1) + 0.1 * rn(n)) * 2.86,
              OIII=10 ** lo3, OII=10 ** (0.35 + 0.35 * m - 0.35 * y + 0.07 * rn(n)), NeIII=10 ** (lo3 - 0.95 + 0.08 * rn(n)), HeI=(0.11 * (1 + 0.2 * rn(n))).clamp_min(0.02))
    lo3a = 0.3 + 0.6 * u + 0.07 * rn(n)
    ag = dict(Ha=3.1 + 0 * m, NII=10 ** (-0.1 + 0.2 * (1 - u) + 0.05 * rn(n)) * 3.1, SII=10 ** (-0.05 - 0.25 * u + 0.07 * rn(n)) * 3.1, OI=10 ** (-0.6 - 0.2 * u + 0.07 * rn(n)) * 3.1,
              OIII=10 ** lo3a, OII=10 ** (0.4 + 0.2 * u + 0.07 * rn(n)), NeIII=10 ** (lo3a - 0.7 + 0.08 * rn(n)), HeI=0.15 + 0 * m)
    R = {k: (1 - a) * sf[k] + a * ag[k] for k in sf}
    R.update(Hb=torch.ones_like(m), Hg=0.468 * torch.ones_like(m), Hd=0.259 * torch.ones_like(m))
    rOII = 0.35 + 1.15 / (1 + ne / 400.0)                                         # F(3729)/F(3727)
    rS = 0.44 + 1.01 / (1 + ne / 600.0)                                           # F(6718)/F(6733)
    spec = [("OII1", 3727.09, R["OII"] / (1 + rOII)), ("OII2", 3729.88, R["OII"] * rOII / (1 + rOII)), ("NeIII", 3869.86, R["NeIII"]), ("Hd", 4102.89, R["Hd"]), ("Hg", 4341.68, R["Hg"]),
            ("Hb", 4862.68, R["Hb"]), ("OIII4960", 4960.30, R["OIII"] / 2.98), ("OIII5008", 5008.24, R["OIII"]), ("HeI", 5877.25, R["HeI"]), ("OI", 6302.05, R["OI"]),
            ("NII6549", 6549.86, R["NII"] / 3.05), ("Ha", 6564.61, R["Ha"]), ("NII6585", 6585.27, R["NII"]), ("SII6718", 6718.29, R["SII"] * rS / (1 + rS)), ("SII6733", 6732.67, R["SII"] / (1 + rS))]
    c_hb = cont_fn(torch.full((n, 1), 4862.68, device=z.device))
    lam, ew, sg = [], [], []
    for name, l0, rel in spec:
        dust = 10 ** (-0.4 * E * (kdust(l0) - kdust(4862.68)))
        c_l = cont_fn(torch.full((n, 1), l0, device=z.device))
        lam.append(torch.full((n,), l0, device=z.device)); ew.append(ew_hb * rel * dust * c_hb[:, 0] / c_l[:, 0]); sg.append(sig)
    ea = 0.6 + 4.0 * torch.sqrt(y * (1 - y))                                      # Balmer absorption: A-type stars, strongest at intermediate youth
    cak = 0.5 + 1.5 * (1 - y)
    sig_st = torch.sqrt((1.3 * sig) ** 2 + 60.0 ** 2)
    for l0, w in ((4102.89, -ea), (4341.68, -0.8 * ea), (4862.68, -0.6 * ea), (3934.78, -cak), (3969.59, -0.7 * cak)):
        lam.append(torch.full((n,), l0, device=z.device)); ew.append(w); sg.append(sig_st)
    lam, ew, sg = torch.stack(lam, 1), torch.stack(ew, 1), torch.stack(sg, 1)
    clean, cont, mu, peak = _finish(wave, z, cont_fn, lam, ew, sg)
    par = dict(m=m, y=y, E=E, ne=ne, a=a, u=u, sig=sig, ew_hb=ew_hb, beta=beta, d4=d4, logM=logM, xn=xn, lo3=lo3, R=R, rOII=rOII, rS=rS)
    return clean, cont, z, mu, peak, par


def gen_lrg(n, rg, wave, zr=(0.3, 1.0)):
    r, rn = rg.r, rg.n
    z = zr[0] + (zr[1] - zr[0]) * r(n)
    t = 0.35 + 0.65 * r(n)                                                        # age (1 = old)
    ls = rn(n)
    sig = 10 ** (2.3 + 0.10 * ls)
    zh = 0.1 + 0.3 * ls + 0.15 * rn(n)                                            # [Z/H]
    ae = (0.2 + 0.1 * ls + 0.05 * rn(n)).clamp(0, 0.45)                           # [alpha/Fe]
    E = 0.01 + 0.06 * rn(n).abs()
    beta = 0.2 + 1.4 * t + 0.4 * zh + 0.15 * rn(n)
    d4 = (0.20 + 0.30 * t + 0.15 * zh + 0.02 * rn(n)).clamp(0.1, 0.65)
    duv = 0.3 + 0.4 * t

    def cont_fn(lam):
        c = (lam / 4500.0) ** beta.unsqueeze(1) * (1 - d4.unsqueeze(1) / (1 + torch.exp(-(4000.0 - lam) / 60.0))) * (1 - duv.unsqueeze(1) / (1 + torch.exp(-(3000.0 - lam) / 150.0)))
        return c * 10 ** (-0.4 * E.unsqueeze(1) * (kdust(lam) - kdust(5500.0)))

    cak = 7 + 2.5 * t + 2.0 * zh
    sig_a = torch.sqrt(sig ** 2 + 400.0 ** 2)                                      # stellar absorption lines are broad (blends, wings)
    hd = (6.9 * (1 - t) + 0.3 * rn(n)).clamp_min(0.0)
    ew = {3934.78: cak, 3969.59: 0.7 * cak, 4102.89: hd, 4341.68: 0.8 * hd, 4305.0: 3 + 3 * t + 2.5 * zh, 4862.68: 0.5 * hd + 0.8, 5176.0: 3 + 1.5 * zh + 7 * ae,
          5270.4: 2.3 + 1.0 * zh - 1.0 * ae, 5335.2: 2.1 + 1.0 * zh, 5893.0: 2.5 + 1.2 * zh + 2 * t, 2799.1: 6 * (1 - 0.5 * t) + 0 * zh, 2600.2: 4 * (1 - 0.5 * t) + 0 * zh, 2852.1: 2.0 + 0 * zh}
    lam, ews, sg = [], [], []
    for l0, w in ew.items():
        lam.append(torch.full((n,), l0, device=z.device)); ews.append(-(w.clamp_min(0.1)) * 10 ** (0.03 * rn(n))); sg.append(sig_a)
    has = (r(n) < (0.15 + 0.65 * (1 - t))).float()                                # weak LINER-like emission in the younger ones
    W = has * 10 ** (-0.1 + 0.9 * (1 - t) / 0.65 + 0.1 * rn(n))
    rO = 0.35 + 1.15 / (1 + 100.0 / 400.0)
    sge = 0.8 * sig + 30
    for l0, w in ((3727.09, W / (1 + rO)), (3729.88, W * rO / (1 + rO)), (4862.68, 0.2 * W), (5008.24, 0.6 * W), (6564.61, 0.8 * W), (6585.27, 0.88 * W), (6718.29, 0.35 * W)):
        lam.append(torch.full((n,), l0, device=z.device)); ews.append(w); sg.append(sge)
    clean, cont, mu, peak = _finish(wave, z, cont_fn, torch.stack(lam, 1), torch.stack(ews, 1), torch.stack(sg, 1))
    par = dict(t=t, sig=sig, zh=zh, ae=ae, E=E, beta=beta, d4=d4, ew=ew, W=W)
    return clean, cont, z, mu, peak, par


def gen_qso(n, rg, wave, zr=(0.8, 3.5)):
    r, rn = rg.r, rg.n
    z = zr[0] + (zr[1] - zr[0]) * r(n)
    Lum = 0.9 * (z - 2.1) / 0.8 + 0.7 * rn(n)
    ev1 = rn(n)
    sig_b = 10 ** (3.35 - 0.10 * ev1 + 0.04 * rn(n))
    alpha = -1.5 + 0.3 * rn(n)
    E = 0.08 * rn(n).abs()

    def cont_fn(lam):
        return (lam / 2500.0) ** alpha.unsqueeze(1) * 10 ** (-0.4 * E.unsqueeze(1) * (kdust(lam) - kdust(5500.0)))

    vsh = (500.0 * (ev1 + 0.5)).clamp(-200, 2500)                                 # CIV blueshift grows with EV1
    broad = [(1215.67, 70, 1.0, 1), (1240.14, 12, 1.0, 1), (1397.0, 9, 1.1, 1), (1549.06, 32, 1.2, 1), (1908.73, 18, 1.0, 1), (2798.75, 28, 0.85, 1), (4862.68, 55, 1.0, 0), (6564.61, 190, 1.0, 0)]
    lam, ews, sg = [], [], []
    for l0, e0, wf, uv in broad:
        e = e0 * 10 ** ((-0.17 if uv else -0.05) * Lum) * 10 ** (0.03 * rn(n))
        l = torch.full((n,), l0, device=z.device)
        if l0 == 1549.06:
            l = l * (1 - vsh / C)
            e = e * (1 - 0.12 * (vsh / 1000.0).clamp(0, 2.5))
        lam.append(l); ews.append(e); sg.append(sig_b * wf)
    sig_n = 10 ** (2.35 + 0.1 * rn(n))
    o3 = 12 * 10 ** (-0.35 * ev1)
    for l0, e in ((5008.24, o3), (4960.30, o3 / 2.98), (3727.09, 3.0 * 10 ** (-0.2 * ev1)), (4862.68, 0.12 * o3)):
        lam.append(torch.full((n,), l0, device=z.device)); ews.append(e); sg.append(sig_n)
    f2 = 10 ** (0.3 * ev1)                                                        # FeII strength anticorrelates with narrow [OIII]
    for l0, e0, uv in ((2400.0, 12, 1), (2600.0, 10, 1), (4570.0, 14, 0), (5250.0, 8, 0)):
        lam.append(torch.full((n,), l0, device=z.device)); ews.append(e0 * f2 * 10 ** ((-0.17 if uv else -0.05) * Lum)); sg.append(torch.full((n,), 2500.0, device=z.device))
    clean, cont, mu, peak = _finish(wave, z, cont_fn, torch.stack(lam, 1), torch.stack(ews, 1), torch.stack(sg, 1))
    nab = torch.poisson(torch.full((n,), 1.5, device=z.device), generator=rg.g).clamp(max=3)
    tau = torch.zeros_like(clean)                                                 # narrow MgII absorbers at random z_abs, not tied to z: transmission exp(-tau)
    for i in range(3):
        za = 0.35 + (z - 0.35).clamp_min(0.01) * r(n)
        tau0 = 10 ** (-0.7 + 1.7 * r(n)) * (nab > i).float()
        for l0, wt in ((2796.35, 1.0), (2803.53, 0.5)):
            mu_a = l0 * (1 + za)
            sa = mu_a * torch.sqrt(60.0 ** 2 + sig_inst(mu_a) ** 2) / C
            tau = tau + (wt * tau0).unsqueeze(1) * torch.exp(-0.5 * ((wave.unsqueeze(0) - mu_a.unsqueeze(1)) / sa.unsqueeze(1)) ** 2)
    clean = clean * torch.exp(-tau)
    bal = (r(n) < 0.10).float()
    depth = bal * (0.3 + 0.4 * r(n))
    vb = 9000.0 + 3000.0 * rn(n).clamp(-1, 2)
    tr = torch.zeros_like(clean)
    for l0, f in ((1549.06, 1.0), (1240.14, 0.8), (1397.0, 0.8)):
        mu_b = l0 * (1 + z) * (1 - vb / C)
        sa = mu_b * 3500.0 / C
        tr = tr + f * depth.unsqueeze(1) * torch.exp(-0.5 * ((wave.unsqueeze(0) - mu_b.unsqueeze(1)) / sa.unsqueeze(1)) ** 2)
    clean = clean * (1 - tr.clamp(max=0.95))
    par = dict(Lum=Lum, ev1=ev1, sig_b=sig_b, vsh=vsh, bal=bal, alpha=alpha, o3=o3)
    return clean, cont, z, mu, peak, par


# --------------------------------------------------------------------------------------------------------------------------------------------
class CleanSky:
    """Generator of clean synthetic suites. `suite(n, z_lo, z_hi)` -> dict(clean (n, 7781), z (n,), cls (n,) with CLASS_NAMES). The classes drawn in a band are the ones whose natural redshift
    reach overlaps it (CLASS_REACH), with the weights `classes` (ELG, LRG, QSO) renormalised over them."""

    def __init__(self, dev="cpu", seed=0, classes=(0.5, 0.2, 0.3)):
        self.dev = torch.device(dev)
        self.wave = wave_grid(self.dev)
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.rg = Rng(self.g, self.dev)
        self.classes = classes

    @torch.no_grad()
    def suite(self, n, z_lo, z_hi):
        z_lo = max(z_lo, 0.005)
        w = torch.tensor([c if (CLASS_REACH[name][0] < z_hi and CLASS_REACH[name][1] > z_lo) else 0.0 for name, c in zip(CLASS_NAMES, self.classes)], device=self.dev, dtype=torch.float32)
        if float(w.sum()) <= 0:
            raise ValueError(f"no class reaches z {z_lo}-{z_hi}")
        cls = torch.multinomial(w / w.sum(), n, replacement=True, generator=self.g)
        out = {"clean": [], "z": [], "cls": []}
        for c, fn in ((0, gen_elg), (1, gen_lrg), (2, gen_qso)):
            k = int((cls == c).sum())
            if k == 0:
                continue
            lo, hi = max(z_lo, CLASS_REACH[CLASS_NAMES[c]][0]), min(z_hi, CLASS_REACH[CLASS_NAMES[c]][1])
            cl, _, z, _, _, _ = fn(k, self.rg, self.wave, zr=(lo, hi))
            out["clean"].append(cl); out["z"].append(z); out["cls"].append(torch.full((k,), c, device=self.dev, dtype=torch.long))
        out = {k: torch.cat(v) for k, v in out.items()}
        perm = torch.randperm(n, device=self.dev, generator=self.g)
        return {k: v[perm] for k, v in out.items()}
