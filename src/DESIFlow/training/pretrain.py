"""
Eidos pretraining: masked denoising (hide spans / cameras of native pixels, score the reconstruction of the hidden pixels) plus a linear
redshift head whose gradient is surgically removed where it opposes reconstruction (PCGrad), on a large memory-mapped set of native-grid spectra.

    python -m DESIFlow.training.pretrain --data-dir DIR --out RUN_DIR [--model full|small] [--arm pcgrad|plain|none] [--steps N] [--resume]

Single GPU. (Multi-GPU needs the PCGrad gradients all-reduced by hand: DDP only synchronizes gradients that come out of .backward().)

Outputs in RUN_DIR:
    config.json, splits.npz            run settings (+ git commit), train / validation / test row indices
    train_log.jsonl, eval_log.jsonl    per-window training means, per-evaluation validation metrics
    curves.png, z_scatter_latest.npz   loss curves (redshift loss, redshift prediction, masked and total reconstruction) and the z scatter
    recon/step_XXXXXXX.png, recon/latest.png   five fixed validation spectra across the redshift range
    best.pt                            weights with the best validation reconstruction (--select, default hidden-pixel chi2)
    last.pt                            model + head + optimizer + step, rewritten every --save-every steps (resume with --resume)
SIGTERM / SIGUSR1 (Slurm: #SBATCH --signal=B:USR1@300) or --max-hours make the run save last.pt and exit cleanly.
"""
import argparse, json, math, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from DESIFlow.eidos.eidos import Eidos
from DESIFlow.eidos.reader import velocity
from DESIFlow.training.data import PretrainData, StepBatches
from DESIFlow.training.evaluation import Evaluator
from DESIFlow.training.loss import hidden_loss
from DESIFlow.training.masking import SpanMasker, apply_hidden, N_PIX
from DESIFlow.training.pcgrad import backward_with_surgery
from DESIFlow.training.plots import plot_curves, plot_reconstructions
from DESIFlow.training.zhead import ZHead

SMALL = dict(cnn_kwargs=dict(branch_channels=8, mix_channels=32, down_channels=(32, 64, 64, 64), d_token=64),
             perceiver_kwargs=dict(n_latents=16, d_latent=64, n_heads=2, n_cross=2),
             decoder_kwargs=dict(d_query=64, n_heads=2))
SELECT = {"masked": "val_masked_chi2", "total": "val_total_chi2", "unmasked": "val_unmasked_chi2"}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="directory with flux.npy, ivar.npy, mask.npy, z.npy (row-matched)")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--group-file", default=None, help="optional (N,) ids (TARGETID / HEALPix): rows of one group never straddle train/val/test")
    ap.add_argument("--model", choices=["full", "small"], default="full")
    ap.add_argument("--arm", choices=["none", "plain", "pcgrad"], default="pcgrad", help="no z head / z head with plain gradient sum / z head with PCGrad")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr-frac", type=float, default=0.1, help="cosine decay floor as a fraction of --lr")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1000.0, help="global grad-norm clip (the grad norm starts in the thousands and falls to tens)")
    ap.add_argument("--huber", type=float, default=5.0, help="Huber threshold in sigma for the reconstruction loss (0 = plain chi2)")
    ap.add_argument("--z-weight", type=float, default=1.0)
    ap.add_argument("--head-lr-mult", type=float, default=0.02,
                    help="LR multiplier for the z head. The flat head reads 64x256 latents: one Adam step at the full LR can move its output by ~9 "
                         "(target std 0.35), which blows up the z loss and its gradient into the encoder")
    ap.add_argument("--val-n", type=int, default=4096)
    ap.add_argument("--test-n", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--recon-every", type=int, default=1000, help="steps between reconstruction figures (multiple of --eval-every)")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--select", choices=list(SELECT), default="masked", help="validation metric that decides best.pt")
    ap.add_argument("--plot-z", default="0.3,1.0,2.0,3.0,4.0", help="target redshifts of the plotted validation spectra")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop cleanly (saving last.pt) after this wall time; 0 = off")
    ap.add_argument("--resume", action="store_true", help="continue from RUN_DIR/last.pt if it exists")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (NOT validated for the rotary / steering path; TF32 is on by default)")
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap


def lr_at(step, args):
    w = min(1.0, (step + 1) / max(args.warmup, 1))
    cos = 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / max(args.steps, 1)))
    return args.lr * w * (args.min_lr_frac + (1 - args.min_lr_frac) * cos)


def gather_hidden_queries(hidden):
    """hidden (B, N) bool -> pixel indices (B, Q) with every sample's hidden pixels first (Q = the largest hidden count) and a validity mask."""
    q = max(int(hidden.sum(1).max()), 1)
    order = torch.argsort((~hidden).to(torch.int8), dim=1, stable=True)[:, :q]
    return order, hidden.gather(1, order)


def sample_pixels(select, k):
    """k random pixels per sample among those flagged in `select` (B, N) bool -> indices (B, k), validity (B, k)."""
    score = torch.rand(select.shape, device=select.device)
    score = torch.where(select, score, torch.full_like(score, 2.0))
    idx = score.topk(min(k, select.shape[1]), dim=1, largest=False).indices
    return idx, select.gather(1, idx)


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def summarize_data(data, train_idx, n=1000):
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(train_idx, size=min(n, len(train_idx)), replace=False))
    f, iv, m = data.read(pick)
    good = (m == 0) & (iv > 0)
    sn = np.array([np.median((f[i] * np.sqrt(iv[i]))[good[i]]) if good[i].any() else 0 for i in range(len(pick))])
    z = data.z
    print(f"data: {data.n} rows x {N_PIX} px; z {np.nanmin(z):.3f}..{np.nanmax(z):.3f}, quartiles {np.nanpercentile(z, [25, 50, 75]).round(2)}; "
          f"from {len(pick)} train rows: good pixel fraction {good.mean():.3f}, median S/N {np.median(sn):.2f} (10-90%: {np.percentile(sn, 10):.2f}..{np.percentile(sn, 90):.2f})", flush=True)


def normalize_argv(argv):
    """--recon_every is accepted as --recon-every (values are left alone)."""
    out = []
    for a in (sys.argv[1:] if argv is None else argv):
        if a.startswith("--"):
            name, eq, val = a.partition("=")
            a = name.replace("_", "-") + eq + val
        out.append(a)
    return out


def main(argv=None):
    args = build_parser().parse_args(normalize_argv(argv))
    out = Path(args.out)
    (out / "recon").mkdir(parents=True, exist_ok=True)
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("no GPU visible to PyTorch (a CPU-only torch build, or no GPU allocated to this shell). Check "
                           "`python -c 'import torch; print(torch.cuda.is_available())'`, or get a GPU node: "
                           "`salloc -N 1 -C gpu -q interactive -t 04:00:00 -A <acct>_g --gpus-per-node=1 -c 32`. Pass --cpu to force the CPU.")
    dev = torch.device("cpu" if args.cpu else "cuda")
    if dev.type == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    workers = args.workers
    if os.name == "nt" and workers > 0:
        print("Windows: DataLoader workers would pickle the memory maps; using --workers 0", flush=True)
        workers = 0

    data = PretrainData(args.data_dir, args.group_file)
    train_idx, val_idx, test_idx = data.splits(args.val_n, args.test_n, args.seed)
    np.savez(out / "splits.npz", train=train_idx, val=val_idx, test=test_idx)
    print(f"split: {len(train_idx)} train / {len(val_idx)} validation / {len(test_idx)} test rows" + (" (grouped)" if data.group is not None else ""), flush=True)
    summarize_data(data, train_idx)

    torch.manual_seed(args.seed)
    cfg = {} if args.model == "full" else SMALL
    model = Eidos(**cfg).to(dev)
    head = None
    if args.arm != "none":
        torch.manual_seed(10_000 + args.seed)
        head = ZHead(model.n_latents, model.d_latent, "flat").to(dev)
    named = list(model.named_parameters()) + [("zhead." + n, p) for n, p in (head.named_parameters() if head else [])]
    params = [p for _, p in named]
    shared = [p for n, p in model.named_parameters() if n.startswith(("cnn", "perceiver"))]
    dec = list(model.decoder.parameters())
    hp = list(head.parameters()) if head else []
    decay = [p for n, p in named if p.ndim >= 2 and not n.startswith("zhead")]
    no_decay = [p for n, p in named if p.ndim < 2 and not n.startswith("zhead")]
    groups = [{"params": decay, "weight_decay": args.wd, "mult": 1.0}, {"params": no_decay, "weight_decay": 0.0, "mult": 1.0},
              {"params": hp, "weight_decay": 0.0, "mult": args.head_lr_mult}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=args.lr, betas=(0.9, 0.99))
    n_par = {k: sum(p.numel() for n, p in named if n.startswith(k)) for k in ("cnn", "perceiver", "decoder", "zhead")}
    print(f"model {args.model}: {n_par} (total {sum(n_par.values()) / 1e6:.2f}M); arm {args.arm}; device {dev}", flush=True)
    qv = velocity(torch.linspace(3600., 9824., N_PIX, dtype=torch.float64, device=dev), model.wave0)

    start, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if head is not None and ck.get("head") is not None:
            head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck["best"]
        print(f"resumed from step {start} (best {args.select} metric so far {best:.4f})", flush=True)
    elif (out / "train_log.jsonl").exists():
        for f in ("train_log.jsonl", "eval_log.jsonl"):
            (out / f).unlink(missing_ok=True)                               # fresh run in a reused directory

    def commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    json.dump({"args": vars(args), "git_commit": commit(), "torch": torch.__version__, "params": n_par,
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}, open(out / "config.json", "w"), indent=1)

    plot_z = [float(s) for s in args.plot_z.split(",")]
    evaluator = Evaluator(data, val_idx, dev, seed=args.seed, chunk=args.eval_batch, plot_z=plot_z, huber_c=args.huber or None)
    print(f"plotted validation spectra (z): {[round(float(evaluator.z[p]), 3) for p in evaluator.plot_pos]}", flush=True)
    masker = SpanMasker()
    ds = StepBatches(data, train_idx, args.batch, args.seed, masker)
    print(f"{ds.steps_per_epoch} steps per epoch at batch {args.batch}; steps {start} -> {args.steps} = {(args.steps - start) / ds.steps_per_epoch:.2f} epochs", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, sampler=range(start, args.steps), num_workers=workers,
                                         prefetch_factor=4 if workers else None, pin_memory=dev.type == "cuda")

    stop = {"flag": False}
    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *a: stop.__setitem__("flag", True))
    t0 = time.time()
    autocast = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if args.bf16 else (lambda: torch.autocast(dev.type, enabled=False))
    huber = args.huber or None

    def save_last(step_done):
        atomic_save({"model": model.state_dict(), "head": head.state_dict() if head else None, "opt": opt.state_dict(), "step": step_done,
                     "best": best, "args": vars(args), "model_kind": args.model}, out / "last.pt")

    def evaluate(step_done, recon):
        nonlocal best
        t = time.time()
        metrics, scatter = evaluator.run(model, head, qv)
        row = {"step": step_done, **metrics, "eval_seconds": time.time() - t}
        append_jsonl(out / "eval_log.jsonl", row)
        if scatter is not None:
            np.savez(out / "z_scatter_latest.npz", z=scatter[0], zp=scatter[1], step=step_done)
        metric = metrics[SELECT[args.select]]
        if metric < best:
            best = metric
            atomic_save({"model": model.state_dict(), "head": head.state_dict() if head else None, "step": step_done, "metric": metric,
                         "select": SELECT[args.select], "args": vars(args), "model_kind": args.model}, out / "best.pt")
            row["new_best"] = True
        try:
            plot_curves(out)
            if recon:
                items = evaluator.plot_items(model, head, qv)
                plot_reconstructions(items, step_done, out / "recon" / f"step_{step_done:07d}.png")
                plot_reconstructions(items, step_done, out / "recon" / "latest.png")
        except ImportError:
            print("matplotlib missing: skipping figures (logs are still written)", flush=True)
        zs = f" | z sigma_NMAD {metrics['z_sigma_nmad']:.4f} cat {metrics['z_cat_frac']:.3f} rmse(log1p) {metrics['z_rmse_log1p']:.3f}" if head else ""
        print(f"[eval @ {step_done}] masked chi2 {metrics['val_masked_chi2']:.3f} | total (masked in) {metrics['val_total_chi2']:.3f} | unmasked {metrics['val_unmasked_chi2']:.3f} "
              f"bin8 {metrics['val_unmasked_bin8']:.2f}{zs} | {args.select}-best {best:.3f}{' *NEW BEST*' if row.get('new_best') else ''} ({row['eval_seconds']:.0f}s)", flush=True)

    model.train()
    acc, n_acc, bad_streak = {}, 0, 0
    t_win, t_end = time.time(), time.time()
    step = start
    for step, batch in zip(range(start, args.steps), loader):
        t_wait = time.time() - t_end
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args) * g["mult"]
        b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        flux, ivar, mask, hidden = b["flux"], b["ivar"], b["mask"], b["hidden"]
        x = torch.stack([flux, ivar, mask], 1)
        good = (mask == 0) & (ivar > 0)
        for p in params:
            p.grad = None
        with autocast():
            lat = model.encode(apply_hidden(x, hidden))
            qi, qw = gather_hidden_queries(hidden)
            pred = model.decoder(lat[:, 1:], qv[qi])
        s = 10 ** lat[:, 0, 0].float()
        fq, iq = flux.gather(1, qi), ivar.gather(1, qi)
        loss_rec = hidden_loss(pred.float(), s, fq, iq, qw, huber)
        loss_z, stats = None, {}
        if head is not None:
            loss_z = args.z_weight * ZHead.loss(head(lat[:, 1:].float()), b["z"], b["z_ok"])
        if not torch.isfinite(loss_rec):
            bad_streak += 1
            print(f"step {step}: non-finite loss, step skipped ({bad_streak} in a row)", flush=True)
            if bad_streak >= 20:
                raise RuntimeError("20 consecutive non-finite losses")
            continue
        with torch.no_grad():
            r2 = (fq - s.unsqueeze(-1) * pred.float()) ** 2 * iq
            chi2_h = (r2 * qw).sum() / qw.sum().clamp_min(1)
            vi, vw = sample_pixels(good & ~hidden, 1024)
            with autocast():
                pv = model.decoder(lat[:, 1:].detach(), qv[vi])
            chi2_v = ((flux.gather(1, vi) - s.unsqueeze(-1) * pv.float()) ** 2 * ivar.gather(1, vi) * vw).sum() / vw.sum().clamp_min(1)
            n_h, n_v = hidden.sum(), (good & ~hidden).sum()
            chi2_t = (n_h * chi2_h + n_v * chi2_v) / (n_h + n_v).clamp_min(1)
        if head is None or loss_z is None:
            loss_rec.backward()
        else:
            stats = backward_with_surgery(loss_rec, loss_z, shared, dec, hp, surgery=(args.arm == "pcgrad"))
        gn = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        if not math.isfinite(gn):
            bad_streak += 1
            print(f"step {step}: non-finite gradient, step skipped ({bad_streak} in a row)", flush=True)
            if bad_streak >= 20:
                raise RuntimeError("20 consecutive non-finite gradients")
            continue
        bad_streak = 0
        opt.step()
        row = {"loss_rec": loss_rec.item(), "chi2_hidden": chi2_h.item(), "chi2_visible": chi2_v.item(), "chi2_total": chi2_t.item(), "grad_norm": gn,
               "clipped": float(gn > args.clip), "data_wait": t_wait}
        if loss_z is not None:
            row["z_loss"] = loss_z.item() / args.z_weight
        row.update(stats)
        for k, v in row.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        t_end = time.time()
        done = step + 1
        if done % args.log_every == 0:
            w = {k: v / n_acc for k, v in acc.items()}
            w.update({"step": done, "lr": lr_at(step, args), "samples_per_s": args.batch * n_acc / (time.time() - t_win), "elapsed_h": (time.time() - t0) / 3600})
            if dev.type == "cuda":
                w["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            append_jsonl(out / "train_log.jsonl", w)
            print(f"step {done:6d} | lr {w['lr']:.2e} | loss {w['loss_rec']:8.3f} | chi2 hidden {w['chi2_hidden']:7.3f} total {w['chi2_total']:7.3f}"
                  + (f" | z loss {w['z_loss']:.4f}" if "z_loss" in w else "") + f" | gnorm {w['grad_norm']:8.1f} | {w['samples_per_s']:6.1f} samples/s"
                  f" (data wait {1e3 * w['data_wait']:.0f} ms/step)" + (f" | peak {w['gpu_mem_gb']:.1f} GB" if "gpu_mem_gb" in w else ""), flush=True)
            acc, n_acc, t_win = {}, 0, time.time()
        last = done == args.steps
        timeout = args.max_hours > 0 and (time.time() - t0) / 3600 > args.max_hours
        if done % args.eval_every == 0 or last:
            evaluate(done, recon=(done % args.recon_every == 0 or last))
            save_last(done)
        elif done % args.save_every == 0:
            save_last(done)
        if stop["flag"] or timeout:
            print(f"stopping at step {done} ({'signal' if stop['flag'] else '--max-hours'}); saving last.pt", flush=True)
            save_last(done)
            break
    summary = {"steps_done": step + 1, "best": best, "select": SELECT[args.select], "hours": (time.time() - t0) / 3600}
    json.dump(summary, open(out / "summary.json", "w"))
    print("done:", summary, flush=True)
    return summary


if __name__ == "__main__":
    main()
