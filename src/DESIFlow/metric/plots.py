"""
Figures of the metric S/N study:

  fidelity_vs_snr.png    (`metric.fidelity`) stress, bias and rank correlation of the metric against its clean value versus median per-pixel S/N, with the noise-realisation confidence interval, one
                         line per redshift band
  projection_vs_z.png    (`metric.dr2_projection`) per redshift bin and S/N cut: completeness, projected stress, median S/N of the kept objects, the S/N cut a stress target needs and the
                         completeness at that cut, objects kept

usage: python -m DESIFlow.metric.plots <fidelity.json | projection.json> [out_dir]
"""
import json, sys
from pathlib import Path
import numpy as np

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]   # blue steps 250 ... 700 (reference palette)
ORDINAL = [0, 2, 4, 6, 8]                                                                                      # five steps validated as an ordinal ramp (validate_palette.js --ordinal)
PLOT_CUTS = (1.0, 2.0, 4.0, 6.0, 8.0)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.findfont("Segoe UI", fallback_to_default=False); fam = ["Segoe UI"]
    except Exception:
        fam = ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5})
    return plt


def _ordinal(n):
    """n <= 5 colours: validated ordinal steps of the one-hue ramp, light -> dark."""
    pos = {1: [2], 2: [1, 3], 3: [0, 2, 4], 4: [0, 1, 3, 4]}.get(n, [0, 1, 2, 3, 4])
    return [RAMP[ORDINAL[i]] for i in pos]


def _ramp_colors(n):
    """Few lines: ordinal steps; many: the continuous one-hue ramp (a colour bar then carries the identity)."""
    if n <= 5:
        return _ordinal(n)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("z", RAMP)
    return [cm(p) for p in np.linspace(0, 1, n)]


def _zbar(plt, fig, z_lo, z_hi):
    """Colour bar for the one-hue redshift ramp in its own axes at the right edge (added AFTER tight_layout, which leaves room through its rect)."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    sm = plt.cm.ScalarMappable(norm=Normalize(z_lo, z_hi), cmap=LinearSegmentedColormap.from_list("z", RAMP))
    cax = fig.add_axes([0.957, 0.2, 0.012, 0.6])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("redshift band (light = low z)", color=INK2); cb.outline.set_visible(False); cax.grid(False)


def _fmt_snr_axis(ax):
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("median per-pixel S/N")


def fig_fidelity(fid, path):
    plt = _mpl()
    from matplotlib.ticker import NullFormatter
    bands = [b for b in fid["bands"] if b.get("rows")]
    if not bands:
        return None
    cols = _ramp_colors(len(bands))
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 5.0))
    ax_s, ax_b, ax_r = axs
    for b, col in zip(bands, cols):
        rows = sorted(b["rows"], key=lambda r: r["snr_realised"])
        x = np.array([r["snr_realised"] for r in rows])
        st = np.array([r["stress"] for r in rows]); bi = np.array([r["bias_rel"] for r in rows]); rho = np.array([r["spearman"] for r in rows])
        lab = f"z {b['z_lo']:g}-{b['z_hi']:g}" if len(bands) <= 5 else None
        ax_s.fill_between(x, st[:, 0], st[:, 4], color=col, alpha=0.25, lw=0)
        ax_s.plot(x, st[:, 2], "-o", color=col, lw=1.8, ms=3.5, mew=0, label=lab)
        ax_b.fill_between(x, bi[:, 0], bi[:, 4], color=col, alpha=0.25, lw=0)
        ax_b.plot(x, bi[:, 2], "-o", color=col, lw=1.8, ms=3.5, mew=0, label=lab)
        ax_r.fill_between(x, rho[:, 0], rho[:, 4], color=col, alpha=0.25, lw=0)
        ax_r.plot(x, rho[:, 2], "-o", color=col, lw=1.8, ms=3.5, mew=0, label=lab)
    ax_s.set_yscale("log"); ax_s.axhline(1.0, color=AXIS, lw=1.0)
    ax_s.set_yticks([0.03, 0.1, 0.3, 1, 3]); ax_s.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"]); ax_s.yaxis.set_minor_formatter(NullFormatter())
    ax_b.axhline(0.0, color=AXIS, lw=1.0)
    ax_s.set_title("stress against the clean distances\n(0 = clean geometry; 1 = errors as large as the distances)", loc="left", fontsize=9.5, color=INK)
    ax_b.set_title("relative bias of the distances: (mean D - mean D_clean) / mean D_clean\n(a ratio to the clean mean, so it can pass 1 while every D stays <= 1)", loc="left", fontsize=9.5, color=INK)
    ax_r.set_title("rank correlation with the clean distances\n(does the ORDER of the distances survive)", loc="left", fontsize=9.5, color=INK)
    for ax in axs:
        _fmt_snr_axis(ax)
    if len(bands) <= 5:
        ax_s.legend(loc="upper right", handlelength=1.6); ax_b.legend(loc="upper right", handlelength=1.6); ax_r.legend(loc="lower left", handlelength=1.6)
    c = fid.get("config", {})
    fig.text(0.01, 0.995, "Fidelity of the target metric to its clean value versus median per-pixel S/N (synthetic spectra)", ha="left", va="top", fontsize=12.5, color=INK, weight="bold")
    fig.text(0.01, 0.945, f"{bands[0]['n_suite']} clean ELG / LRG / QSO spectra per band, redshift-aligned, NW-smoothed, weighted-starlet Hellinger over the mutual window; all objects degraded to the same S/N; "
             f"{c.get('reps', '?')} noise realisations per point, line = median, band = 95% interval; noise: {'real donors' if c.get('noise_donors') else 'analytic DESI-like profile'}",
             ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1 if len(bands) <= 5 else 0.95, 0.9))
    if len(bands) > 5:
        _zbar(plt, fig, bands[0]["z_lo"], bands[-1]["z_hi"])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_projection(proj, path):
    plt = _mpl()
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    bins = proj["bins"]
    edges = [(b["z_lo"], b["z_hi"]) for b in bins]
    cen = np.array([(lo + hi) / 2 for lo, hi in edges]); half = np.array([(hi - lo) / 2 for lo, hi in edges])
    allcuts = sorted({t["cut"] for t in proj["table"]})
    show = [c for c in PLOT_CUTS if c in allcuts] or [c for c in allcuts if c > 0][:5]
    cols = _ordinal(len(show))
    per = {(t["z_lo"], t["z_hi"], t["cut"]): t for t in proj["table"]}

    def series(key, c):
        return np.array([(per.get((lo, hi, c), {}).get(key) if per.get((lo, hi, c), {}).get(key) is not None else np.nan) for lo, hi in edges], float)

    fig, axs = plt.subplots(2, 3, figsize=(14.2, 8.2), sharex=True)
    ax_c, ax_s, ax_m, ax_n, ax_cn, ax_k = axs.ravel()
    for c, col in zip(show, cols):
        lab = f"S/N >= {c:g}"
        for ax, key in ((ax_c, "completeness"), (ax_s, "stress"), (ax_m, "snr_median_kept"), (ax_k, "n_kept")):
            y = series(key, c)
            ax.errorbar(cen, y, xerr=half, fmt="-o", color=col, lw=1.7, ms=4, mew=0, elinewidth=1.0, capsize=0, label=lab)
        lo_, hi_ = series("stress_ci_lo", c), series("stress_ci_hi", c)
        ax_s.fill_between(cen, lo_, hi_, color=col, alpha=0.18, lw=0)
    ax_s.set_yscale("log"); ax_s.axhline(1.0, color=AXIS, lw=1.0)
    ax_s.set_yticks([0.1, 0.2, 0.4, 0.7, 1, 2]); ax_s.set_yticklabels(["0.1", "0.2", "0.4", "0.7", "1", "2"]); ax_s.yaxis.set_minor_formatter(NullFormatter())
    ax_m.set_yscale("log"); ax_m.set_yticks([0.5, 1, 2, 4, 8, 16]); ax_m.set_yticklabels(["0.5", "1", "2", "4", "8", "16"]); ax_m.yaxis.set_minor_formatter(NullFormatter())
    ax_k.set_yscale("log"); ax_k.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0))); ax_k.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}")); ax_k.yaxis.set_minor_formatter(NullFormatter())
    ax_c.set_ylim(-0.02, 1.04)
    ts = sorted({n["stress_target"] for n in proj["needs"]})
    tcols = [RAMP[i] for i in ([0, 4, 8] if len(ts) == 3 else ORDINAL[:len(ts)])]
    for x, col in zip(ts, tcols):
        rows = [next((n for n in proj["needs"] if n["z_lo"] == lo and n["z_hi"] == hi and n["stress_target"] == x), {}) for lo, hi in edges]
        for ax, key in ((ax_n, "snr_needed"), (ax_cn, "completeness")):
            y = np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows], float)
            ax.errorbar(cen, y, xerr=half, fmt="-o", color=col, lw=1.7, ms=4, mew=0, elinewidth=1.0, capsize=0, label=f"stress <= {x:g}")
    ax_n.set_yscale("log"); ax_n.set_yticks([0.5, 1, 2, 4, 8, 16, 32]); ax_n.set_yticklabels(["0.5", "1", "2", "4", "8", "16", "32"]); ax_n.yaxis.set_minor_formatter(NullFormatter())
    ax_cn.set_ylim(-0.02, 1.04)
    ax_c.set_title("completeness: fraction of the bin with S/N >= cut", loc="left", fontsize=9.5, color=INK)
    ax_s.set_title("projected stress of the metric for the kept objects\n(simulated curve at their median S/N; band = 95% interval)", loc="left", fontsize=9.5, color=INK)
    ax_m.set_title("median per-pixel S/N of the kept objects", loc="left", fontsize=9.5, color=INK)
    ax_n.set_title("S/N cut a stress target needs\n(blank: not reached by S/N 64)", loc="left", fontsize=9.5, color=INK)
    ax_cn.set_title("completeness at that S/N cut", loc="left", fontsize=9.5, color=INK)
    ax_k.set_title("objects kept", loc="left", fontsize=9.5, color=INK)
    for ax in axs[1]:
        ax.set_xlabel("redshift (bars: bin extent)")
    h, l = ax_c.get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.997), ncol=len(l), handlelength=1.6, columnspacing=1.4)
    ax_n.legend(loc="upper left", handlelength=1.6)
    fig.text(0.01, 0.995, "DR2 projection: what an S/N cut keeps and how faithful the metric is, by redshift", ha="left", va="top", fontsize=12.5, color=INK, weight="bold")
    fig.text(0.01, 0.955, f"{proj['n_valid']} objects; the metric is not run on them: the synthetic-spectrum fidelity curve of each redshift band is read at the S/N statistics of the survey bin", ha="left", va="top", fontsize=8.8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    res = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).parent
    if "table" in res:
        print(fig_projection(res, out / "projection_vs_z.png"))
    else:
        print(fig_fidelity(res, out / "fidelity_vs_snr.png"))
