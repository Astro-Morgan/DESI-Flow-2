#!/bin/bash
# GRID Eidos pretraining on one Perlmutter GPU node (1 GPU). Edit the account and paths, then:  sbatch slurm/pretrain_grid_perlmutter.sbatch
# Quick check first:  sbatch -q debug -t 00:30:00 --export=ALL,STEPS=200,EVAL_EVERY=100,GATE_MIN=50 slurm/pretrain_grid_perlmutter.sbatch
# Resubmitting the same script continues from RUN_DIR/last.pt (--resume), so a chain of jobs trains through the walltime limit.
# The step-based schedules below are the B=128 values of the benchmark (ramp 1000-2000 steps = 128k-256k objects); scale them with the batch size.
#SBATCH -A CHANGE_ME_g            # your GPU allocation (name usually ends in _g)
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -J eidos_grid
#SBATCH -o %x_%j.out
#SBATCH --signal=B:USR1@300       # 5 min before the limit the batch shell gets USR1; the trap below forwards it so the run saves last.pt and exits

REPO=${REPO:-$SLURM_SUBMIT_DIR}                                   # the cloned repo (submit from its root)
DATA=${DATA:-$SCRATCH/pretraining_data}                           # flux.npy ivar.npy mask.npy z.npy
OUT=${OUT:-$SCRATCH/eidos_grid/run1}
STEPS=${STEPS:-40000}
EVAL_EVERY=${EVAL_EVERY:-2000}
BATCH=${BATCH:-128}
ACCUM=${ACCUM:-2}                                                 # micro-batches of BATCH/ACCUM: the small grid model needs ~21 GB at 64 objects with --pos cbias (~15 GB with the plain table)
POS=${POS:-cbias}                                                 # cbias (default: table + content-conditioned offsets, 1.55x encoder step time) | bias | rope
GATE_MIN=${GATE_MIN:-250}
GATE_FORCE=${GATE_FORCE:-8000}                                    # open the decoder z feed at this step even if the classifier never reaches sigma_NMAD 0.0014 (real spectra: median S/N 1.8)

# module load pytorch            # NERSC's PyTorch module (check `module avail pytorch`), or: source activate your_env
export PYTHONPATH=$REPO/src:$PYTHONPATH
export OMP_NUM_THREADS=4
mkdir -p "$OUT"
cd "$REPO"

srun python -u -m DESIFlow.training.pretrain_grid \
    --data-dir "$DATA" --out "$OUT" --resume \
    --model small --pos "$POS" --batch "$BATCH" --accum "$ACCUM" --steps "$STEPS" --warmup 500 \
    --ramp-abs 1000 2000 --gate-min-step "$GATE_MIN" --gate-force-step "$GATE_FORCE" --gate-ramp-steps 200 \
    --eval-every "$EVAL_EVERY" --recon-every "$EVAL_EVERY" --save-every 1000 \
    --workers 16 --max-hours 11.5 &
PID=$!
trap 'kill -USR1 $PID' USR1
wait $PID
wait $PID                                                         # the first wait returns early when the trap fires
