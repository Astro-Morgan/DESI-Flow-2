"""
Where does the reconstruction residual live? Per-scale excess variance of (data - model), by S/N bin.

    python -m DESIFlow.training.diagnose --data-dir DIR --run-dir RUN [--ckpt best.pt] [--n 2048]

For every validation spectrum the residual d - s*model (native pixels, quoted ivar) goes through the model's own preprocessor (so it sits on the
same 23 km/s log-lambda lattice with propagated ivar as the encoder input) and the weighted starlet (scale j ~ 23.2 * 2^j km/s). The same is done
for simulated pure-noise spectra drawn from the quoted ivar. The excess

    E_j = (mean squared starlet coefficient of the residual at scale j) / (the same for pure noise)

is 1 where the residual is noise at that scale and above 1 where the model misses structure (or the quoted noise is too small). Scale 1 is
almost pure noise in real spectra, so E_1 doubles as a check of the ivar calibration at each S/N; E_j - E_1 at coarser scales is structured misfit.
Two variants: nothing hidden (the denoised output), and the fixed validation hidden spans only (the training condition).
Also printed per S/N bin: pixel chi2 (floor 1), chi2 of 8-pixel bins, and the fraction of hidden pixels beyond 5 sigma (what a Huber(5) loss clips).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from DESIFlow.eidos.eidos import Eidos
from DESIFlow.eidos.reader import velocity
from DESIFlow.eidos.starlet import RawStarletPath
from DESIFlow.training.data import PretrainData
from DESIFlow.training.evaluation import Evaluator
from DESIFlow.training.masking import apply_hidden, N_PIX
from DESIFlow.training.scales import scale_power, scale_excess      # noqa: F401  (re-exported)

SNR_EDGES = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 1e9]


def load_model(run_dir, ckpt, dev):
    from DESIFlow.training.pretrain import SMALL
    ck = torch.load(Path(run_dir) / ckpt, map_location=dev, weights_only=False)
    model = Eidos(**({} if ck.get("model_kind", "full") == "full" else SMALL)).to(dev).eval()
    model.load_state_dict(ck["model"])
    return model, ck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=2048, help="validation spectra to use")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    run = Path(args.run_dir)
    model, ck = load_model(run, args.ckpt, dev)
    seed = ck["args"]["seed"] if "args" in ck else 0
    data = PretrainData(args.data_dir)
    val_idx = np.load(run / "splits.npz")["val"][:args.n]
    a_ck = ck.get("args", {})                                                         # the masks the run was validated with (older runs: 5-40%)
    ev = Evaluator(data, val_idx, dev, seed=seed, chunk=args.batch, mask_frac=(a_ck.get("mask_frac_min", 0.05), a_ck.get("mask_frac_max", 0.40)))
    qv = velocity(torch.linspace(3600., 9824., N_PIX, dtype=torch.float64, device=dev), model.wave0)
    path = RawStarletPath(9).to(dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    n = ev.n
    E = {"unmasked": [], "hidden": []}
    px = {"chi2_unmasked": [], "chi2_hidden": [], "bin8": [], "beyond5": []}
    for a in range(0, n, args.batch):
        sl = slice(a, a + args.batch)
        flux, ivar, mask, hid = ev.flux[sl], ev.ivar[sl], ev.mask[sl], ev.hidden[sl]
        good = (mask == 0) & (ivar > 0)
        x = torch.stack([flux, ivar, mask], 1)
        with torch.no_grad():
            lat = model.encode(x)
            res_u = flux - 10 ** lat[:, 0, 0, None] * model.decoder(lat[:, 1:], qv)
            lat_m = model.encode(apply_hidden(x, hid))
            res_m = flux - 10 ** lat_m[:, 0, 0, None] * model.decoder(lat_m[:, 1:], qv)
        E["unmasked"].append(scale_excess(model.preprocessor, path, res_u, ivar, mask, gen=gen).cpu())
        E["hidden"].append(scale_excess(model.preprocessor, path, res_m, ivar, torch.where(hid, mask, torch.ones_like(mask)), gen=gen).cpu())
        r2u, r2m = res_u ** 2 * ivar, res_m ** 2 * ivar
        px["chi2_unmasked"].append(((r2u * good).sum(1) / good.sum(1).clamp_min(1)).cpu())
        px["chi2_hidden"].append(((r2m * hid).sum(1) / hid.sum(1).clamp_min(1)).cpu())
        w8 = (ivar * good)[:, :7776].reshape(len(x), -1, 8)
        num, den = (w8 * res_u[:, :7776].reshape(len(x), -1, 8)).sum(-1), w8.sum(-1)
        px["bin8"].append((((num ** 2 / den.clamp_min(1e-30)) * (den > 0)).sum(1) / (den > 0).sum(1).clamp_min(1)).cpu())
        px["beyond5"].append((((res_m * ivar.sqrt()).abs() > 5) & hid).sum(1).float().div(hid.sum(1).clamp_min(1)).cpu())
    E = {k: torch.cat(v).numpy() for k, v in E.items()}
    px = {k: torch.cat(v).numpy() for k, v in px.items()}
    snr = ev.snr
    bins = np.digitize(snr, SNR_EDGES) - 1
    J = E["unmasked"].shape[1] - 1
    labels = [f"s{j} ({23.2 * 2 ** j:.0f} km/s)" for j in range(1, J + 1)] + ["coarse"]
    report = {"n": int(n), "snr_edges": SNR_EDGES, "scales": labels, "bins": []}
    print(f"{n} validation spectra, checkpoint step {ck.get('step')}; E_j = residual power / pure-noise power per starlet scale (1 = noise)")
    for b in range(len(SNR_EDGES) - 1):
        m = bins == b
        if m.sum() < 2:
            continue
        row = {"snr": [SNR_EDGES[b], SNR_EDGES[b + 1]], "n": int(m.sum()), "chi2_unmasked": float(px["chi2_unmasked"][m].mean()),
               "chi2_hidden": float(px["chi2_hidden"][m].mean()), "bin8": float(px["bin8"][m].mean()), "frac_hidden_beyond_5sigma": float(px["beyond5"][m].mean()),
               "E_unmasked": np.median(E["unmasked"][m], 0).tolist(), "E_hidden": np.median(E["hidden"][m], 0).tolist()}
        report["bins"].append(row)
        print(f"\nS/N {SNR_EDGES[b]:g}-{SNR_EDGES[b + 1]:g} (n={m.sum()}): chi2 unmasked {row['chi2_unmasked']:.2f}, hidden {row['chi2_hidden']:.2f}, 8-px bins {row['bin8']:.2f}; "
              f"{100 * row['frac_hidden_beyond_5sigma']:.1f}% of hidden px beyond 5 sigma")
        print("   scale          " + " ".join(f"{l.split()[0]:>6s}" for l in labels))
        print("   E unmasked     " + " ".join(f"{v:6.2f}" for v in row["E_unmasked"]))
        print("   E hidden only  " + " ".join(f"{v:6.2f}" for v in row["E_hidden"]))
    json.dump(report, open(run / "diagnose.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
        for key, a_ in (("E_unmasked", ax[0]), ("E_hidden", ax[1])):
            for row in report["bins"]:
                a_.plot(range(1, J + 2), row[key], "o-", label=f"S/N {row['snr'][0]:g}-{row['snr'][1]:g} (n={row['n']})")
            a_.axhline(1, color="0.5", lw=0.7)
            a_.set_yscale("log")
            a_.set_xticks(range(1, J + 2))
            a_.set_xticklabels([f"s{j}" for j in range(1, J + 1)] + ["c"])
            a_.set_xlabel("starlet scale (width ~ 23 * 2^j km/s)")
            a_.set_title(key.replace("E_", "excess residual power, ") + " (1 = noise)")
        ax[0].legend(fontsize=7)
        mid = [np.sqrt(max(r["snr"][0], 0.5) * min(r["snr"][1], 32)) for r in report["bins"]]
        ax[2].plot(mid, [r["chi2_unmasked"] for r in report["bins"]], "o-", label="chi2/px, unmasked")
        ax[2].plot(mid, [r["chi2_hidden"] for r in report["bins"]], "s-", label="chi2/px, hidden")
        ax[2].plot(mid, [r["E_unmasked"][0] for r in report["bins"]], "^-", label="E_1 (noise calibration)")
        ax[2].axhline(1, color="0.5", lw=0.7)
        ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("median S/N per pixel"); ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run / "diagnose.png", dpi=110)
        print(f"\nwrote {run / 'diagnose.json'} and {run / 'diagnose.png'}")
    except ImportError:
        print("matplotlib missing: figure skipped")
    return report


if __name__ == "__main__":
    main()
