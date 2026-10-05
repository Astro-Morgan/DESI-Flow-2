"""
Figures for the pretraining run (matplotlib, headless). Everything is rebuilt from the JSONL logs, so a resumed run continues its curves.

curves.png  (every evaluation)
    row 1: redshift loss (train MSE of log1p z, validation RMSE of log1p z) | redshift prediction (validation z_pred vs z_true) | z scatter and
           catastrophic fraction vs step
    row 2: masked reconstruction (hidden-pixel chi2, train and validation) | total reconstruction (all good pixels: train and validation from the
           masked input, validation from the UNMASKED input) | gradient norm, cos(g_rec, g_z)
recon/step_XXXXXXX.png  five fixed validation spectra across the redshift range: 8-pixel data bins, the model with nothing hidden (blue), and
    the reconstruction of the hidden spans (red) from the masked input; below each, the residual in units of its noise.
"""
import json
from pathlib import Path
import numpy as np

LINES = {"Lya": 1215.67, "CIV": 1549.0, "CIII]": 1908.7, "MgII": 2798.0, "[OII]": 3727.4, "Hb": 4861.3, "[OIII]": 5006.8, "Ha": 6562.8}
WAVE = np.linspace(3600., 9824., 7781)


def _read(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _smooth(y, w=7):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")


def plot_curves(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    tr, ev = _read(out / "train_log.jsonl"), _read(out / "eval_log.jsonl")
    if not tr:
        return
    st = np.array([r["step"] for r in tr])
    es = np.array([e["step"] for e in ev])
    col = lambda rows, k: np.array([r.get(k, np.nan) for r in rows], float)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    a = ax[0, 0]
    if "z_loss" in tr[0]:
        a.plot(st, _smooth(col(tr, "z_loss")), color="tab:blue", label="train MSE(log1p z)")
        if ev and "z_rmse_log1p" in ev[0]:
            a.plot(es, col(ev, "z_rmse_log1p") ** 2, "o-", color="tab:red", label="validation (RMSE^2)")
        a.set_yscale("log"); a.legend(); a.set_title("redshift loss"); a.set_xlabel("step")
    a = ax[0, 1]
    sc = out / "z_scatter_latest.npz"
    if sc.exists():
        d = np.load(sc)
        a.plot(d["z"], d["zp"], ".", ms=2, alpha=0.4)
        lim = [0, max(float(d["z"].max()), 1) * 1.05]
        a.plot(lim, lim, "k-", lw=0.7); a.set_xlim(lim); a.set_ylim(-0.2, lim[1] * 1.1)
        a.set_xlabel("z true"); a.set_ylabel("z predicted"); a.set_title(f"redshift prediction, validation (step {int(d['step'])})")
    a = ax[0, 2]
    if ev and "z_sigma_nmad" in ev[0]:
        a.plot(es, col(ev, "z_sigma_nmad"), "o-", label="sigma_NMAD of dz/(1+z)")
        a.plot(es, col(ev, "z_cat_frac"), "s-", label="fraction |dz|/(1+z) > 0.15")
        a.set_yscale("log"); a.legend(); a.set_title("redshift accuracy, validation"); a.set_xlabel("step")
    a = ax[1, 0]
    a.plot(st, _smooth(col(tr, "chi2_hidden")), label="train (hidden px)")
    if ev:
        a.plot(es, col(ev, "val_masked_chi2"), "o-", color="tab:red", label="validation (hidden px)")
        b = int(np.nanargmin(col(ev, "val_masked_chi2")))
        a.plot(es[b], col(ev, "val_masked_chi2")[b], "*", ms=14, color="gold", mec="k", label="best (saved)")
    a.axhline(1, color="0.6", lw=0.7); a.set_yscale("log"); a.legend(); a.set_title("masked reconstruction, chi2/px (1 = noise)"); a.set_xlabel("step")
    a = ax[1, 1]
    a.plot(st, _smooth(col(tr, "chi2_total")), label="train, masked input (all good px)")
    if ev:
        a.plot(es, col(ev, "val_total_chi2"), "o-", color="tab:red", label="validation, masked input")
        a.plot(es, col(ev, "val_unmasked_chi2"), "s-", color="tab:green", label="validation, unmasked input")
    a.axhline(1, color="0.6", lw=0.7); a.set_yscale("log"); a.legend(); a.set_title("total reconstruction, chi2/px"); a.set_xlabel("step")
    a = ax[1, 2]
    a.plot(st, _smooth(col(tr, "grad_norm")), color="tab:purple", label="grad norm (pre-clip)")
    a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("grad norm")
    if "cos" in tr[0]:
        a2 = a.twinx()
        a2.plot(st, _smooth(col(tr, "cos"), 15), color="tab:orange", lw=0.8, label="cos(g_rec, g_z)")
        a2.axhline(0, color="0.7", lw=0.5); a2.set_ylabel("cos")
    a.set_title("optimization")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)


def plot_reconstructions(items, step, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(18, 3.1 * n), gridspec_kw={"width_ratios": [2.6, 1.4]})
    W = 8
    for r, it in enumerate(items):
        top, res_ax = axes[r]
        f, iv, g, h = it["flux"].astype(float), it["ivar"].astype(float), it["good"], it["hidden"]
        w = (iv * g)[:7776].reshape(-1, W)
        den = w.sum(1)
        ok = den > 0
        fb = np.where(ok, (w * f[:7776].reshape(-1, W)).sum(1) / np.maximum(den, 1e-30), np.nan)
        sb = 1 / np.sqrt(np.maximum(den, 1e-30))
        wb = WAVE[:7776].reshape(-1, W).mean(1)
        mb_u = np.where(ok, (w * it["denoised"][:7776].reshape(-1, W)).sum(1) / np.maximum(den, 1e-30), np.nan)
        hb = (w * h[:7776].reshape(-1, W)).sum(1) / np.maximum(den, 1e-30) > 0.5
        mb_r = np.where(ok & hb, (w * it["rec_masked"][:7776].reshape(-1, W)).sum(1) / np.maximum(den, 1e-30), np.nan)
        top.errorbar(wb, fb, sb, color="k", lw=0.5, elinewidth=0.35, alpha=0.8, label="data, 8-px bins")
        top.plot(WAVE, it["denoised"], color="tab:blue", lw=0.8, label="model, nothing hidden")
        top.plot(WAVE, np.where(h, it["rec_masked"], np.nan), color="tab:red", lw=1.0, label="reconstruction of hidden px")
        edges = np.flatnonzero(np.diff(np.concatenate([[0], h.astype(int), [0]])))
        for a_, e_ in zip(edges[0::2], edges[1::2]):
            top.axvspan(WAVE[a_], WAVE[min(e_, 7780)], color="tab:orange", alpha=0.13, lw=0)
        lo, hi = np.nanpercentile(fb, [1, 99.5])
        top.set_ylim(lo - 0.3 * (hi - lo), hi + 0.35 * (hi - lo))
        for name, lam in LINES.items():
            x = lam * (1 + it["z"])
            if 3700 < x < 9750:
                top.axvline(x, color="g", lw=0.5, ls=":")
                top.text(x, top.get_ylim()[1], name, fontsize=6, color="g", ha="center", va="top")
        rb = (fb - mb_u) / sb
        chi_bin = np.nanmean(rb ** 2)
        hid_chi = float(((f - it["rec_masked"]) ** 2 * iv)[h].mean()) if h.any() else float("nan")
        top.set_title(f"val #{it['val_pos']}: z = {it['z']:.3f}, z_pred = {it['z_pred']:.3f}, median S/N {it['snr']:.1f}; nothing-hidden binned chi2 {chi_bin:.2f}; "
                      f"hidden-px chi2 {hid_chi:.2f}", fontsize=9)
        res_ax.axhspan(-2, 2, color="0.87")
        res_ax.plot(wb, rb, color="tab:blue", lw=0.5, label="data - model (nothing hidden)")
        res_ax.plot(wb, np.where(hb, (fb - mb_r) / sb, np.nan), color="tab:red", lw=0.7, label="data - reconstruction (hidden bins)")
        res_ax.set_ylim(-8, 8)
        res_ax.set_ylabel("residual / sigma_bin", fontsize=8)
        if r == 0:
            top.legend(fontsize=7, ncol=3, loc="lower right")
            res_ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("observed wavelength (A)")
    axes[-1, 1].set_xlabel("observed wavelength (A)")
    fig.suptitle(f"step {step}", y=0.995)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
