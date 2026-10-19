# Grid Eidos pretraining

Entry point: `python -m DESIFlow.training.pretrain_grid` (source: `src/DESIFlow/training/pretrain_grid.py`; model: `src/DESIFlow/eidos/grid_eidos.py`). Single GPU. Same data, masking,
degradation, loss and logging as `docs/pretraining_nersc.md` (read that first for the data format and the outputs); what changes is the model and the redshift machinery.

## Model

```
raw (B, 3, 7781)  ->  Preprocessor  ->  CNN (weighted-starlet branches)  ->  813 tokens (one per 16 log-lambda px = 370 km/s)
  -> proj + n x TokenBlock (self-attention with a learned relative-position table plus a content-conditioned offset preference [--pos cbias, default]; the plain table [--pos bias] or rotary positions [--pos rope] are the alternatives)
  -> to_lat: 8 numbers per token  = latents (B, 1 + 813, 8)   (813 x 8 = 6504 < 7781 flux values; slot 0 = scale token; Plato reads slots 1..)
  -> HybridDecoder: a 1626-point output grid (2 per token) cross-attends the lifted latents, self-attends, then a conv stack up-samples to the log grid; interpolated at the queried wavelengths
```

Redshift: two heads on the flattened latents (`ZHead` linear regression of ln(1+z); `ZClassHead` softmax over 801 ln(1+z) bins), trained from step 0 and PCGrad'd against reconstruction
(the z gradient on the shared encoder has its anti-parallel component removed, is capped at the reconstruction gradient's norm and multiplied by a ramp `--ramp-abs A B`). The
decoder (never the encoder) receives z through an additive, zero-initialised feed on the grid queries, behind a gate that opens when the classifier's training-batch sigma_NMAD EMA falls below
`--gate-thr 0.0014`. Everything about why these choices: memory nodes `eidos-grid-architecture-handoff-2026-10`, `eidos-autodecoder-ceiling-test`, `eidos-grid-migration-into-src`.

## Run

```bash
export PYTHONPATH=$PWD/src
python -u -m DESIFlow.training.pretrain_grid --data-dir $SCRATCH/pretraining_data --out $SCRATCH/eidos_grid/run1 \
    --model small --pos cbias --batch 128 --accum 2 --steps 40000 --warmup 500 --ramp-abs 1000 2000 --gate-min-step 250 --gate-ramp-steps 200 --resume
sbatch slurm/pretrain_grid_perlmutter.sbatch      # edit the account / paths first; resubmitting continues from last.pt
```

- `--batch` is the effective batch (PCGrad needs whole-batch gradients); `--accum K` splits it into K micro-batches of batch/K (identical result, memory of batch/K). The small model used 7.7 GB at 32 objects with the plain table and about 10.5 GB with `--pos cbias` (the per-token offset preference is a (batch, heads, 813, 813) tensor in every block).
- The step-based schedules (`--ramp-abs`, `--gate-min-step`, `--gate-ramp-steps`, `--warmup`) were tuned in OBJECTS: the benchmark used ramp 4000-8000 steps at B=32 and 1000-2000 at B=128.
- `--pos`: `cbias` (default) adds to the learned offset table a term g(x_i).R[offset] in which the asking token's content (a small MLP, rank 8) chooses which signed offsets it attends to (see `eidos/tokens.py`).
  Benchmark one-factor tests (G0, B=32, 40k steps, no z heads, one seed): random-span hidden-line medians cbias 0.52 / 0.44 / 0.58 (std / dr2 / hi) vs the plain table 0.44 / 0.34 / 0.48 vs rotary 0.29 / 0.24 / 0.33;
  complex-hidden and visible lines equal; cbias costs 1.55x the training time of the encoder step (264 vs 171 ms at 32 objects) and 0.37M parameters. Use `--pos bias` for the cheaper baseline.
- `--no-heads` = reconstruction only; `--no-zfeed` = heads without the decoder feed; `--no-surgery`, `--no-z-cap` for the one-factor ablations; `--hyb-grid 1|2|4`, `--hyb-conv-norm` for the decoder.
- `--model full` (default CNN, d 256, 8 blocks) is NOT validated with this head: the benchmark ran `--model small` (d 128, 6 blocks).
- `--lsf-aug X --lsf-prob 0.5`: LSF augmentation, off by default. A fraction of the training objects are re-observed with an extra Gaussian LSF of sigma ~ U(0, X) Angstrom (DESI's own sigma is ~0.76 A; Spec-S5 = the same spectrographs on a 6 m mirror, so the LSF is the expected difference). Input AND loss target are broadened together (the model learns to reconstruct whatever width it sees, not to deconvolve); it can only broaden. Kernel, ivar propagation and masking rules: `broaden_lsf` in `training/data.py` (tests/test_lsf_aug.py). The benchmark generator has the same option (`--lsf-aug LO HI`, `--val-lsf LO HI` for a broadened validation set 'wide').
- Defaults follow the benchmark: AdamW lr 6e-4, clip 1.0 (heads 4x), regression head lr x0.02, `--loss-pixels all`, masks 5-75%, brightness-tied degradation on.

## What to look at

`eval_log.jsonl` has the usual validation metrics (`val_wmse`, masked / unmasked chi2, z sigma_NMAD) plus `gate`; the training log prints `gate`, `zgrad` (the ramp), the classifier EMA that drives the gate, and the PCGrad statistics
(`cos`, `z_cap`, `along`). The decoder only matters for pretraining: downstream (Plato) reads `encode(x)[:, 1:]`.

## First real-data run on NERSC (checklist)

Everything above was validated on SYNTHETIC physics-linked spectra (benchmark: median per-pixel S/N 3, z <= 3.5). The DR2 pretraining set (608,090 rows) has median S/N 1.8 (10-90%: 0.5-7.2), z up to 6, quartiles 0.50 / 0.89 / 1.57.

1. **Ship the code.** The grid pretraining depends on the v3 edits of `training/{loss,evaluation,pretrain}.py`, `eidos/embeddings.py` (ScaleToken for narrow latents) and `tests/test_pretrain.py` that were uncommitted when this was written (`weighted_mse_loss` is not in the older commit): commit them together with the new files, push, `git pull` on Perlmutter.
2. **Environment + tests on the login node (CPU, ~5 min):** `export PYTHONPATH=$PWD/src; python tests/test_eidos_grid.py; python tests/test_pretrain_grid.py; python tests/test_lsf_aug.py` (the benchmark-equivalence test skips itself when `scripts/` is absent).
3. **Debug-queue smoke (30 min):** `sbatch -A <project>_g -q debug -t 00:30:00 --export=ALL,ENV_CMD="module load pytorch/<version>",DATA=...,OUT=...,STEPS=200,EVAL_EVERY=100,GATE_MIN=50,GATE_FORCE=150 slurm/pretrain_grid_perlmutter.sbatch` (the GPU account ends in `_g`; `ENV_CMD` is how you get your PyTorch environment, without commas; the script prints the job, host, torch version, GPU and checks the data files before starting; the log is `eidos_grid_<jobid>.out` in the submit directory). Read from the log: peak memory (expect ~21 GB at 64 objects per micro-batch with cbias; raise `ACCUM` if the GPU has less), samples/s and `data wait` (raise `--workers` if it is not ~0), the printed classifier range (`ln(1+z) bins up to 2.0` for this set), and that the gate opens at step 150 at the latest.
4. **Production:** `sbatch slurm/pretrain_grid_perlmutter.sbatch` (STEPS 40000 = 8.5 epochs at batch 128; resubmit to continue from `last.pt`; keep STEPS fixed across the chain because the cosine schedule depends on it). Throughput guess from earlier runs: the 12.4M-parameter Perceiver Eidos ran 136-143 samples/s on an A100; the laptop RTX 4090 runs this stack at ~58 samples/s, so expect roughly 90-130 samples/s on an A100 => 11-16 h for 40k steps (one to two 12 h jobs). Measure in step 3 before committing.
5. **What to look at (eval_log.jsonl / train log):** `val_wmse` and masked chi2 falling smoothly (the v2 run was erratic; the v1 run reached z sigma_NMAD 0.044 with the old regression head); `gate` (the decoder z feed opens when the classifier's training-batch sigma_NMAD EMA < 0.0014 -- on S/N ~2 spectra it may not, then `--gate-force-step` opens it at step 8000); z sigma_NMAD and catastrophic fraction on the validation set from the classifier and the regression head; the E1 / scale-excess diagnostics (`python -m DESIFlow.training.diagnose`, see docs/pretraining_nersc.md); the PCGrad statistics (`cos`, `along` >= 1, `z_cap`).
6. **Known risks:** (a) the benchmark gates and ramps were tuned in objects at q ~ 3; real data are noisier and the classifier may need many more steps to reach sub-0.001 precision -- judge the encoder by masked reconstruction, not by the gate time; (b) heavy-tailed noise and bad ivar are handled by `--w-cap 100` as before; (c) z labels are taken as trusted (NaN/negative excluded): catastrophic labels make the classifier loss noisy, which the soft Gaussian labels and PCGrad tolerate in the benchmark but this is untested on real labels; (d) `--model full` is unvalidated; (e) cbias is 1.55x the encoder step time of the plain table -- `POS=bias` is the cheaper fallback if throughput is the problem.
7. **Optional, after the first run:** `--lsf-aug 0.6` for the Spec-S5 LSF robustness (off by default); a second seed; the full-size model.

## Tests

`python tests/test_eidos_grid.py` (modules, z feed, parameter routing, accumulated PCGrad, exact equivalence with the benchmark implementation when `scripts/narrow_bench/phys` is present) and
`python tests/test_pretrain_grid.py` (tiny CPU run: outputs, ramp, gate, resume).
