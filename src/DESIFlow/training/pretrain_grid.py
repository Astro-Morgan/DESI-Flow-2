"""
Pretraining of the GRID Eidos (eidos/grid_eidos.py): masked denoising on native-grid spectra + two redshift heads (linear regression and a softmax classifier over ln(1+z) bins)
with PCGrad, a ramp and a precision gate (the recipe of the physics-linked benchmark campaign, scripts/narrow_bench/phys/physeidos2.py).

    python -m DESIFlow.training.pretrain_grid --data-dir DIR --out RUN_DIR [--model small|full] [--pos cbias|bias|rope] [--steps N] [--batch B --accum K] [--resume]

Recipe (all measured in the benchmark; see memory 'eidos-grid-architecture-handoff-2026-10' and 'eidos-autodecoder-ceiling-test'):
    reconstruction   ivar-weighted per-object MSE on ALL good pixels, hidden and visible (training/loss.py weighted_mse_loss), masked input, the input of bright objects degraded
    heads            ZHead 'flat' regression (log loss, eps 1000 km/s, lr x0.02) + ZClassHead softmax (lr x1) on the flattened per-token latents; trained from step 0
    PCGrad           z gradient on the shared encoder (CNN + token attention + bottleneck): anti-parallel component removed, capped at the reconstruction gradient's norm, times the
                     RAMP (0 until --ramp-abs A, linear to 1 at B); the decoder-side parameters get reconstruction only; the heads get g_rec (through the z feed) + g_z;
                     surgery acts on the accumulated WHOLE-batch gradients (--accum micro-batches are exactly equivalent to one big batch)
    z feed (decoder) additive, zero-init, GATED: the gate stays 0 until an EMA of the classifier's training-batch sigma_NMAD (masked inputs, trusted z) drops below --gate-thr
                     (0.0014 ~ 0.0008 on a clean validation set), then ramps to 1 over --gate-ramp-steps. The encoder never receives z.
Outputs in RUN_DIR as training/pretrain.py: config.json, splits.npz, train_log.jsonl, eval_log.jsonl, curves.png, recon/, best.pt, last.pt (model + heads + optimizer + gate state;
resume with --resume). SIGTERM / SIGUSR1 / --max-hours save last.pt and exit.
The sizes of --model full are NOT validated (the benchmark ran the small CNN with d 128, 6 blocks, 8 numbers per token).
"""
import argparse, json, math, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from DESIFlow.eidos.grid_eidos import EidosGrid
from DESIFlow.eidos.reader import velocity
from DESIFlow.training.data import PretrainData, StepBatches
from DESIFlow.training.evaluation import Evaluator
from DESIFlow.training.loss import weighted_mse_loss
from DESIFlow.training.masking import SpanMasker, apply_hidden, N_PIX
from DESIFlow.training.pcgrad import route_grads, acc_grads, route_apply
from DESIFlow.training.plots import plot_curves, plot_reconstructions
from DESIFlow.training.pretrain import SMALL, SELECT, lr_at, atomic_save, append_jsonl, summarize_data
from DESIFlow.training.zhead import ZHead, ZClassHead

C_KMS = 299792.458


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-file", default=None)
    ap.add_argument("--model", choices=["small", "full"], default="small", help="CNN size (small = the benchmark's; full = the default CNN, UNVALIDATED with the grid head)")
    ap.add_argument("--pos", choices=["cbias", "bias", "rope"], default="cbias", help="token attention positions: learned offset table + content-conditioned offset preference g(x_i).R[offset] "
                    "(default; benchmark winner on random-span hidden-line medians, +0.08..+0.10, 1.55x encoder step time), the plain table, or rotary on the physical token velocities")
    ap.add_argument("--dim", type=int, default=None, help="token attention width (default: 128 small, 256 full)")
    ap.add_argument("--layers", type=int, default=None, help="token attention blocks (default: 6 small, 8 full)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bott", type=int, default=8, help="numbers per token (813 x 8 = 6504 < 7781 flux values)")
    ap.add_argument("--hyb-grid", type=int, default=2, choices=[1, 2, 4], help="decoder output grid = this x 813 points")
    ap.add_argument("--hyb-dim", type=int, default=96)
    ap.add_argument("--hyb-conv-norm", action="store_true", help="GroupNorm(1) in the decoder conv stack (default off: it hurt the brightest-line objects)")
    ap.add_argument("--no-heads", action="store_true", help="no z heads (and no z feed): reconstruction only")
    ap.add_argument("--no-zfeed", action="store_true", help="z heads but no z feed into the decoder")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per step (batch must be divisible): memory is that of batch / accum, the result is identical")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0, help="global grad-norm clip of the model (the heads get 4x)")
    ap.add_argument("--rb-mult", type=float, default=5.0, help="LR multiplier of the learned offset table and of R (--pos bias | cbias)")
    ap.add_argument("--reg-lr-mult", type=float, default=0.02)
    ap.add_argument("--cls-lr-mult", type=float, default=1.0)
    ap.add_argument("--hidden-weight", type=float, default=1.0)
    ap.add_argument("--w-cap", type=float, default=100.0)
    ap.add_argument("--loss-pixels", choices=["all", "hidden"], default="all")
    ap.add_argument("--mask-frac-min", type=float, default=0.05)
    ap.add_argument("--mask-frac-max", type=float, default=0.75)
    ap.add_argument("--degrade", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--q-floor", type=float, default=1.0)
    ap.add_argument("--degrade-power", type=float, default=2.0)
    ap.add_argument("--lsf-aug", type=float, default=0.0, help="LSF augmentation: a fraction --lsf-prob of the training objects are re-observed with an extra Gaussian LSF of sigma ~ U(0, X) "
                    "Angstrom (input AND target broadened together, like a different instrument; 0 = off). DESI's own LSF sigma is ~0.76 A")
    ap.add_argument("--lsf-prob", type=float, default=0.5)
    ap.add_argument("--z-eps-kms", type=float, default=1000.0)
    ap.add_argument("--w-cls", type=float, default=1.0, help="weight of the classifier loss relative to the regression loss")
    ap.add_argument("--no-z-cap", action="store_true", help="do not cap the z gradient on the encoder at the reconstruction gradient's norm")
    ap.add_argument("--no-surgery", action="store_true", help="plain sum of the two gradients on the encoder (no PCGrad projection)")
    ap.add_argument("--ramp-abs", type=int, nargs=2, default=[4000, 8000], help="z gradient on the encoder: 0 until A, linear to 1 at B (steps)")
    ap.add_argument("--gate-thr", type=float, default=0.0014, help="decoder z feed opens when the training-batch sigma_NMAD EMA of the classifier is below this")
    ap.add_argument("--gate-ema", type=float, default=0.02)
    ap.add_argument("--gate-ramp-steps", type=int, default=500)
    ap.add_argument("--gate-min-step", type=int, default=1000)
    ap.add_argument("--gate-force-step", type=int, default=0, help="open the decoder z feed at this step even if the classifier never reached --gate-thr (real data: noisy spectra and z labels may keep the "
                    "training-batch sigma_NMAD above the synthetic benchmark's 0.0014; 0 = off). The feed is zero-initialised, so opening it early is safe")
    ap.add_argument("--cls-ymax", default="auto", help="upper edge of the classifier's ln(1+z) bins: a number, or 'auto' = max(1.6, ln(1+max trusted training z) rounded up to 0.1). The benchmark used 1.6 (z <= 3.95); "
                    "the DR2 pretraining set reaches z = 6 (ln 7 = 1.95), where a 1.6 head would clamp the labels")
    ap.add_argument("--val-n", type=int, default=4096)
    ap.add_argument("--test-n", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--recon-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--select", choices=list(SELECT), default="wmse")
    ap.add_argument("--plot-z", default="0.3,1.0,2.0,3.0,4.0")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap


def ramp(t, a, b):
    return 1.0 if t >= b else 0.0 if t <= a else (t - a) / max(b - a, 1e-9)


class Heads(nn.Module):
    """The two redshift heads on the content latents: regression (ZHead, ln(1+z)) and classifier (ZClassHead). z(content) -> (z_reg (B,), z_cls (B,), logits)."""

    def __init__(self, n_latents, d_latent, y_max=1.6):
        super().__init__()
        self.reg, self.cls = ZHead(n_latents, d_latent, "flat"), ZClassHead(n_latents, d_latent, y_max=y_max)

    def forward(self, content):
        c = content.float()
        z_reg, logits = self.reg(c), self.cls(c)
        return z_reg, self.cls.predict(logits), logits


class EvalView:
    """The Eidos API the Evaluator uses (encode / decoder / preprocessor / train-eval), with the z feed applied: the heads are read off the latents it is given."""

    def __init__(self, net, heads, state):
        self.net, self.heads, self.state, self.preprocessor = net, heads, state, net.preprocessor

    @property
    def training(self):
        return self.net.training

    def eval(self):
        self.train(False)

    def train(self, mode=True):
        self.net.train(mode)
        if self.heads is not None:
            self.heads.train(mode)

    def encode(self, x):
        return self.net.encode(x)

    def decoder(self, content, qv):
        z = None
        if self.heads is not None and self.net.zfeed:
            z_reg, z_cls, _ = self.heads(content)
            z = (z_reg, z_cls)
        return self.net.decoder(content, qv, self.net.z_feed(z, self.state["gate"]))


class ZPredictor(nn.Module):
    """head(content) -> the classifier's ln(1+z), the interface the Evaluator expects of a z head"""

    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, content):
        return self.heads(content)[1]


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    (out / "recon").mkdir(parents=True, exist_ok=True)
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("no GPU visible to PyTorch; run on a GPU node or pass --cpu")
    if args.batch % args.accum:
        raise SystemExit("--batch must be divisible by --accum")
    dev = torch.device("cpu" if args.cpu else "cuda")
    if dev.type == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    workers = 0 if os.name == "nt" else args.workers

    data = PretrainData(args.data_dir, args.group_file)
    train_idx, val_idx, test_idx = data.splits(args.val_n, args.test_n, args.seed)
    np.savez(out / "splits.npz", train=train_idx, val=val_idx, test=test_idx)
    print(f"split: {len(train_idx)} train / {len(val_idx)} validation / {len(test_idx)} test rows", flush=True)
    summarize_data(data, train_idx)

    torch.manual_seed(args.seed)
    cnn_kwargs = SMALL["cnn_kwargs"] if args.model == "small" else {}
    d, layers = args.dim or (128 if args.model == "small" else 256), args.layers or (6 if args.model == "small" else 8)
    net = EidosGrid(cnn_kwargs=cnn_kwargs, d=d, n_layers=layers, n_heads=args.heads, bott=args.bott, pos=args.pos, zfeed=not (args.no_heads or args.no_zfeed),
                    decoder_kwargs=dict(factor=args.hyb_grid, d=args.hyb_dim, conv_norm=args.hyb_conv_norm)).to(dev)
    heads = None
    if not args.no_heads:
        torch.manual_seed(10_000 + args.seed)
        zt = data.z[train_idx][data.trusted(train_idx)]
        y_max = float(args.cls_ymax) if args.cls_ymax != "auto" else max(1.6, math.ceil(math.log1p(float(zt.max())) * 10 - 1e-9) / 10) if len(zt) else 1.6
        heads = Heads(net.n_latents, net.d_latent, y_max).to(dev)
        print(f"classifier head: ln(1+z) bins up to {y_max:g} ({heads.cls.n_bins} bins of 0.002; trusted training z up to {float(zt.max()) if len(zt) else float('nan'):.3f})", flush=True)
    shared, rec_only = net.partition_params()
    heads_p = list(heads.parameters()) if heads else []
    named = [(n, p) for n, p in net.named_parameters()] + [("heads." + n, p) for n, p in (heads.named_parameters() if heads else [])]
    is_rb = lambda n: n.endswith((".rb", ".R"))                                          # offset table and offset profiles: lr x rb_mult, no weight decay
    groups = [{"params": [p for n, p in named if not n.startswith("heads.") and not is_rb(n) and p.ndim >= 2], "weight_decay": args.wd, "mult": 1.0},
              {"params": [p for n, p in named if not n.startswith("heads.") and not is_rb(n) and p.ndim < 2], "weight_decay": 0.0, "mult": 1.0},
              {"params": [p for n, p in named if is_rb(n)], "weight_decay": 0.0, "mult": args.rb_mult},
              {"params": [p for n, p in named if n.startswith("heads.reg")], "weight_decay": args.wd, "mult": args.reg_lr_mult},
              {"params": [p for n, p in named if n.startswith("heads.cls")], "weight_decay": args.wd, "mult": args.cls_lr_mult}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=args.lr, betas=(0.9, 0.99))
    n_par = {"encoder": sum(p.numel() for p in shared), "decoder+feed": sum(p.numel() for p in rec_only), "heads": sum(p.numel() for p in heads_p)}
    print(f"EidosGrid {args.model} pos={args.pos} d={d} layers={layers} bott={args.bott}: {n_par} (total {sum(n_par.values()) / 1e6:.2f}M); z feed {net.zfeed}; device {dev}", flush=True)
    qv = velocity(torch.linspace(3600., 9824., N_PIX, dtype=torch.float64, device=dev), net.wave0)
    z_eps = args.z_eps_kms / C_KMS
    state = {"gate": 0.0 if net.zfeed else 1.0}

    start, best, trig, ema = 0, float("inf"), None, None
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        if heads is not None and ck.get("heads") is not None:
            heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"])
        start, best, trig, ema = ck["step"], ck["best"], ck.get("trig"), ck.get("ema")
        print(f"resumed from step {start} (best {args.select} metric so far {best:.4f}; gate trigger {trig})", flush=True)
    elif (out / "train_log.jsonl").exists():
        for f in ("train_log.jsonl", "eval_log.jsonl"):
            (out / f).unlink(missing_ok=True)

    def commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    json.dump({"args": vars(args), "git_commit": commit(), "torch": torch.__version__, "params": n_par,
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}, open(out / "config.json", "w"), indent=1)

    view = EvalView(net, heads, state)
    zpred = ZPredictor(heads) if heads is not None else None
    evaluator = Evaluator(data, val_idx, dev, seed=args.seed, chunk=args.eval_batch, plot_z=[float(s) for s in args.plot_z.split(",")], mask_frac=(args.mask_frac_min, args.mask_frac_max),
                          z_eps=z_eps, z_loss="log")
    masker = SpanMasker(frac=(args.mask_frac_min, args.mask_frac_max))
    ds = StepBatches(data, train_idx, args.batch, args.seed, masker, degrade=args.degrade, q_floor=args.q_floor, degrade_power=args.degrade_power,
                     lsf_aug=args.lsf_aug, lsf_prob=args.lsf_prob)
    print(f"{ds.steps_per_epoch} steps per epoch at batch {args.batch}; steps {start} -> {args.steps} = {(args.steps - start) / ds.steps_per_epoch:.2f} epochs", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=range(start, args.steps), num_workers=workers, prefetch_factor=4 if workers else None, pin_memory=dev.type == "cuda")

    stop = {"flag": False}
    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *a: stop.__setitem__("flag", True))
    t0 = time.time()

    def save_last(step_done):
        atomic_save({"model": net.state_dict(), "heads": heads.state_dict() if heads else None, "opt": opt.state_dict(), "step": step_done, "best": best, "trig": trig, "ema": ema,
                     "args": vars(args), "model_kind": "grid-" + args.model}, out / "last.pt")

    def evaluate(step_done, do_recon):
        nonlocal best
        t = time.time()
        metrics, scatter = evaluator.run(view, zpred, qv)
        row = {"step": step_done, "gate": state["gate"], **metrics, "eval_seconds": time.time() - t}
        append_jsonl(out / "eval_log.jsonl", row)
        if scatter is not None:
            np.savez(out / "z_scatter_latest.npz", z=scatter[0], zp=scatter[1], step=step_done)
        metric = metrics[SELECT[args.select]]
        if metric < best:
            best = metric
            atomic_save({"model": net.state_dict(), "heads": heads.state_dict() if heads else None, "step": step_done, "metric": metric, "select": SELECT[args.select],
                         "args": vars(args), "model_kind": "grid-" + args.model}, out / "best.pt")
            row["new_best"] = True
        try:
            plot_curves(out)
            if do_recon:
                items = evaluator.plot_items(view, zpred, qv)
                plot_reconstructions(items, step_done, out / "recon" / f"step_{step_done:07d}.png")
                plot_reconstructions(items, step_done, out / "recon" / "latest.png")
        except ImportError:
            print("matplotlib missing: skipping figures", flush=True)
        zs = f" | z sigma_NMAD {metrics['z_sigma_nmad']:.4f} cat {metrics['z_cat_frac']:.3f}" if zpred is not None else ""
        print(f"[eval @ {step_done}] gate {state['gate']:.2f} | wmse {metrics['val_wmse']:.4f} (unmasked {metrics['val_unmasked_wmse']:.4f}) | masked chi2 {metrics['val_masked_chi2']:.3f} "
              f"| total {metrics['val_total_chi2']:.3f} | unmasked {metrics['val_unmasked_chi2']:.3f}{zs} | {args.select}-best {best:.4f}{' *NEW BEST*' if row.get('new_best') else ''} ({row['eval_seconds']:.0f}s)", flush=True)

    net.train(); heads.train() if heads else None
    acc, n_acc, bad_streak, t_win, step = {}, 0, 0, time.time(), start
    mb = args.batch // args.accum
    for step, batch in zip(range(start, args.steps), loader):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args) * g["mult"]
        zsc = ramp(step, *args.ramp_abs) if heads is not None else 0.0                      # z gradient reaching the shared encoder
        if net.zfeed:
            state["gate"] = 0.0 if trig is None else min(1.0, (step - trig) / max(args.gate_ramp_steps, 1))
        b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        flux, ivar, mask, hidden = b["flux"], b["ivar"], b["mask"], b["hidden"]
        x = torch.stack([b["flux_in"], b["ivar_in"], mask], 1)
        good = (mask == 0) & (ivar > 0)
        score = good if args.loss_pixels == "all" else hidden
        for p in net.parameters():
            p.grad = None
        for p in heads_p:
            p.grad = None
        g_rec_acc = g_z_acc = None
        loss_rec_v = loss_z_v = 0.0
        dz_list, ok, r2h = [], True, torch.zeros(4, device=dev)
        for mi in range(args.accum):
            sl = slice(mi * mb, (mi + 1) * mb)
            lat = net.encode(apply_hidden(x[sl], hidden[sl]))
            z, logits = None, None
            if heads is not None:
                z_reg, z_cls, logits = heads(lat[:, 1:])
                z = (z_reg, z_cls) if net.zfeed else None
            pred = net.decode_native(lat, z=z, gate=state["gate"])                          # (mb, 7781) flux / s on the native pixels
            s = 10 ** lat[:, 0, 0].float()
            loss_rec = weighted_mse_loss(pred.float(), s, flux[sl], ivar[sl], score[sl], hidden[sl], args.hidden_weight, args.w_cap) / args.accum
            loss_z = None
            if heads is not None:
                zt, zok = b["z"][sl], b["z_ok"][sl]
                parts = [ZHead.log_loss(z_reg, zt, zok, z_eps), None if args.w_cls == 0 else heads.cls.loss(logits, zt, zok)]
                parts = [p for p in parts if p is not None]
                if parts:
                    loss_z = (parts[0] + (args.w_cls * parts[1] if len(parts) > 1 else 0.0)) / args.accum
                with torch.no_grad():
                    dz_list.append((torch.exp(z_cls.detach() - torch.log1p(zt)) - 1.0)[zok.bool()])
            if not torch.isfinite(loss_rec) or (loss_z is not None and not torch.isfinite(loss_z)):
                ok = False
                break
            with torch.no_grad():
                r2 = (flux[sl] - s.unsqueeze(-1) * pred.float()) ** 2 * ivar[sl]
                mh, mv = score[sl] & hidden[sl], score[sl] & ~hidden[sl]
                r2h += torch.stack([(r2 * mh).sum(), mh.sum().float(), (r2 * mv).sum(), mv.sum().float()])
            gr, gz = route_grads(loss_rec, loss_z, shared, rec_only, heads_p)
            g_rec_acc, g_z_acc = acc_grads(g_rec_acc, gr), acc_grads(g_z_acc, gz)
            loss_rec_v += loss_rec.item()
            loss_z_v += 0.0 if loss_z is None else loss_z.item()
        if not ok:
            bad_streak += 1
            print(f"step {step}: non-finite loss, step skipped ({bad_streak} in a row)", flush=True)
            if bad_streak >= 20:
                raise RuntimeError("20 consecutive non-finite losses")
            continue
        if heads is not None and net.zfeed and trig is None and dz_list:
            dzb = torch.cat(dz_list)
            if dzb.numel():
                nm = 1.4826 * float((dzb - dzb.median()).abs().median())
                ema = nm if ema is None else (1 - args.gate_ema) * ema + args.gate_ema * nm
                if step >= args.gate_min_step and ema < args.gate_thr:
                    trig = step
                    print(f"step {step}: classifier sigma_NMAD EMA {ema:.5f} < {args.gate_thr}: the decoder z feed opens (ramp over {args.gate_ramp_steps} steps)", flush=True)
        if heads is not None and net.zfeed and trig is None and args.gate_force_step and step >= args.gate_force_step:
            trig = step
            print(f"step {step}: --gate-force-step reached (classifier sigma_NMAD EMA {ema if ema is not None else float('nan'):.5f} never fell below {args.gate_thr}): the decoder z feed opens anyway", flush=True)
        stats = route_apply(g_rec_acc, g_z_acc, shared, rec_only, heads_p, surgery=not args.no_surgery, z_cap=not args.no_z_cap, z_scale=zsc)
        gn = float(torch.nn.utils.clip_grad_norm_(shared + rec_only, args.clip))
        if heads_p:
            torch.nn.utils.clip_grad_norm_(heads_p, args.clip * 4)
        if not math.isfinite(gn):
            bad_streak += 1
            opt.zero_grad(set_to_none=True)
            print(f"step {step}: non-finite gradient, step skipped ({bad_streak} in a row)", flush=True)
            if bad_streak >= 20:
                raise RuntimeError("20 consecutive non-finite gradients")
            continue
        bad_streak = 0
        opt.step()
        chi2_h, chi2_v = float(r2h[0] / r2h[1].clamp_min(1)), float(r2h[2] / r2h[3].clamp_min(1))
        row = {"loss_rec": loss_rec_v, "chi2_hidden": chi2_h, "chi2_visible": chi2_v, "chi2_total": float((r2h[0] + r2h[2]) / (r2h[1] + r2h[3]).clamp_min(1)), "grad_norm": gn,
               "clipped": float(gn > args.clip), "gate": state["gate"], "zscale": zsc, "g_mean": b["g"].mean().item(), "frac_degraded": (b["g"] > 1.05).float().mean().item()}
        if heads is not None:
            row["z_loss"] = loss_z_v
        row.update(stats)
        for k, v in row.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        done = step + 1
        if done % args.log_every == 0:
            w = {k: v / n_acc for k, v in acc.items()}
            w.update({"step": done, "lr": lr_at(step, args), "samples_per_s": args.batch * n_acc / (time.time() - t_win), "elapsed_h": (time.time() - t0) / 3600, "ema_nmad": ema})
            if dev.type == "cuda":
                w["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            append_jsonl(out / "train_log.jsonl", w)
            print(f"step {done:6d} | lr {w['lr']:.2e} | loss {w['loss_rec']:8.4f} | chi2 hidden {w['chi2_hidden']:7.3f} total {w['chi2_total']:7.3f}"
                  + (f" | z loss {w['z_loss']:.4f} ema {(ema if ema is not None else float('nan')):.4f} gate {state['gate']:.2f} zgrad {zsc:.2f}" if heads is not None else "")
                  + f" | gnorm {w['grad_norm']:.2f} | {w['samples_per_s']:6.1f} samples/s" + (f" | peak {w['gpu_mem_gb']:.1f} GB" if "gpu_mem_gb" in w else ""), flush=True)
            acc, n_acc, t_win = {}, 0, time.time()
        last = done == args.steps
        timeout = args.max_hours > 0 and (time.time() - t0) / 3600 > args.max_hours
        if done % args.eval_every == 0 or last:
            evaluate(done, do_recon=(done % args.recon_every == 0 or last))
            save_last(done)
        elif done % args.save_every == 0:
            save_last(done)
        if stop["flag"] or timeout:
            print(f"stopping at step {done} ({'signal' if stop['flag'] else '--max-hours'}); saving last.pt", flush=True)
            save_last(done)
            break
    summary = {"steps_done": step + 1, "best": best, "select": SELECT[args.select], "hours": (time.time() - t0) / 3600, "gate_trigger": trig}
    json.dump(summary, open(out / "summary.json", "w"))
    print("done:", summary, flush=True)
    return summary


if __name__ == "__main__":
    main()
