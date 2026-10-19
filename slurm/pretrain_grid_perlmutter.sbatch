#!/bin/bash
# GRID Eidos pretraining on one Perlmutter GPU node (1 GPU). Edit the account and paths, then:  sbatch slurm/pretrain_grid_perlmutter.sbatch
# Quick check first (the account of a GPU job ends in _g; ENV_CMD has no commas; every default below can be overridden through --export):
#   sbatch -A desi_g -q debug -t 00:30:00 --export=ALL,ENV_CMD="module load pytorch/2.11.0",STEPS=200,EVAL_EVERY=100,GATE_MIN=50,GATE_FORCE=150 slurm/pretrain_grid_perlmutter.sbatch
# The job log is  eidos_grid_<jobid>.out  in the directory you ran sbatch from (sacct -j JOBID -X --format=JobID,WorkDir%100 shows it).
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

ENV_CMD=${ENV_CMD:-}             # how to get the PyTorch environment, e.g. --export=ALL,ENV_CMD="module load pytorch/2.11.0"  (no commas) or ENV_CMD="source activate myenv"
if [ -n "$ENV_CMD" ]; then eval "$ENV_CMD"; fi
export PYTHONPATH=$REPO/src:$PYTHONPATH
export OMP_NUM_THREADS=4
mkdir -p "$OUT"
cd "$REPO"
echo "job $SLURM_JOB_ID on $(hostname) | repo $REPO | data $DATA | out $OUT | pos $POS batch $BATCH accum $ACCUM steps $STEPS"
python -c "import sys, torch; print('python', sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
    || { echo "ERROR: cannot import torch; pass the environment with --export=ALL,ENV_CMD=\"module load pytorch/...\""; exit 2; }
python -c "import DESIFlow.training.pretrain_grid" || { echo "ERROR: cannot import DESIFlow from $REPO/src (submit from the repo root or set REPO=...)"; exit 3; }
for f in flux ivar mask z; do [ -f "$DATA/$f.npy" ] || { echo "ERROR: $DATA/$f.npy not found"; exit 4; }; done

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
