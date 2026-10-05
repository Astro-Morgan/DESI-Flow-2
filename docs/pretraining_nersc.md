# Eidos pretraining on NERSC

Entry point: `python -m DESIFlow.training.pretrain` (source: `src/DESIFlow/training/pretrain.py`). Single GPU. Masked denoising of native-grid
spectra plus a linear redshift head trained with PCGrad (the head's gradient is projected off wherever it opposes reconstruction).

## Data

A directory with four row-matched arrays on the native DESI grid (`coadd_cameras` output, 3600-9824 A, 0.8 A, 7781 pixels), C-order:

| file | shape | meaning |
|---|---|---|
| `flux.npy` | (N, 7781) | flux, 1e-17 erg/s/cm2/A |
| `ivar.npy` | (N, 7781) | inverse variance |
| `mask.npy` | (N, 7781) | DESI pixel mask, 0 = good (any nonzero is bad) |
| `z.npy`    | (N,)      | redshift, taken as trusted (NaN or negative rows are excluded from the z loss) |
| `group.npy` (optional) | (N,) | TARGETID / HEALPix ids; rows of one group never straddle train / validation / test |

The arrays are memory-mapped (`mmap_mode="r"`), never loaded whole; non-finite values and negative ivar are zeroed on read. Without `group.npy`
the split is by row, so if one TARGETID can appear in several rows (several surveys / programs / healpix files), pass its ids as `group.npy`.

## Run

```bash
export PYTHONPATH=$PWD/src
python -u -m DESIFlow.training.pretrain --data-dir $SCRATCH/pretraining_data --out $SCRATCH/eidos_pretrain/run1 --steps 30000 --batch 64 --resume
sbatch slurm/pretrain_perlmutter.sbatch          # edit the account / paths first; resubmitting continues from last.pt
```

Defaults: full 12.4M-parameter model, `--arm pcgrad` (z head + PCGrad; `plain` = same head, plain gradient sum; `none` = reconstruction only),
AdamW lr 5e-4 with 1000 warm-up steps and cosine decay to 10%, clip 1000, Huber 5 sigma, TF32 on, batch 64, 4096 validation + 4096 test rows held out.
`--head-lr-mult 0.02`: the flat z head has 16,384 inputs and one full-LR Adam step can move its output by ~9 (target std 0.35).

## Outputs (RUN_DIR)

| file | content |
|---|---|
| `curves.png` | redshift loss; z predicted vs true (validation); sigma_NMAD and catastrophic fraction; masked reconstruction (train / validation, best marked); total reconstruction (train and validation from masked input, validation from unmasked input); gradient norm and cos(g_rec, g_z) |
| `recon/step_XXXXXXX.png`, `recon/latest.png` | five fixed validation spectra across the redshift range (every `--recon-every` steps): data in 8-px bins, model with nothing hidden (blue), reconstruction of the hidden spans (red), residuals in sigma, lines marked at the true z |
| `train_log.jsonl`, `eval_log.jsonl` | training means per `--log-every` steps; validation metrics per evaluation |
| `best.pt` | weights at the best validation hidden-pixel chi2 (`--select masked|total|unmasked`) |
| `last.pt` | model + head + optimizer + step, for `--resume` |
| `config.json`, `splits.npz` | settings, git commit, row indices of the three splits |

chi2 values are mean (flux - s*model)^2 * ivar per pixel: 1 = noise. "Masked reconstruction" = hidden pixels only; "total" = every good pixel;
"unmasked" = nothing hidden. A wide gap between masked and unmasked chi2 means the model is not yet using context well.

## Memory and speed (estimates from a laptop RTX 4090; check the printed numbers)

Batch 16 used 5.0 GB and ran 33-38 samples/s on the laptop, so batch 64 should need about 20 GB (batch 128 ~40 GB: too tight on a 40 GB A100).
The log prints samples/s, GPU peak memory and `data wait` (ms per step spent waiting for the loader; if it is not near zero, raise `--workers`).

## Cloning and running on NERSC

- Use SSH keys (or a GitHub token) on the login node; the repo's `data/`, `legacy/` and `scripts/` are NOT tracked, so they will not be there. Tests that read
  `data/data.fits` (`test_eidos*.py`, `test_preprocessing.py` partly) need that file copied over; `test_pretrain.py` falls back to synthetic spectra.
- Clone into `$HOME` or the project directory (CFS); keep the data and the run directories on `$SCRATCH` (purged after ~8 weeks without access). For the large arrays set Lustre striping before copying: `lfs setstripe -c 8 $SCRATCH/pretraining_data`.
- Run on a compute node (`salloc`/`sbatch`), not the login node. Environment: `module load pytorch` (check `module avail pytorch`) or your own conda env with torch >= 2.1, numpy, matplotlib (astropy is needed only for the tests that read FITS).
- `git pull` on the cluster after each push; the module is run from source (`PYTHONPATH=$PWD/src`), nothing to install.
- Shell scripts must keep LF line endings (`.gitattributes` enforces it for `*.sbatch` / `*.sh`).

## Limits

Single GPU only (PCGrad calls `autograd.grad` outside `.backward()`, so DDP would not synchronize those gradients; multi-GPU needs a manual all-reduce).
`--bf16` is available but the rotary / steering path has only been validated in fp32 / TF32. No ivar noise floor yet (DR2 ivar is under-estimated at high S/N); the Huber tail is the only protection.
