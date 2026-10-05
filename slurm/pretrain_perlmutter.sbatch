#!/bin/bash
# Eidos pretraining on one Perlmutter GPU node (1 GPU). Edit the three paths and the account, then:  sbatch slurm/pretrain_perlmutter.sbatch
# Quick check first:  sbatch -q debug -t 00:30:00 --export=ALL,STEPS=200,EVAL_EVERY=100 slurm/pretrain_perlmutter.sbatch
# Resubmitting the same script continues from RUN_DIR/last.pt (--resume), so a chain of jobs trains through the walltime limit.
#SBATCH -A CHANGE_ME_g            # your GPU allocation (name usually ends in _g)
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -J eidos_pretrain
#SBATCH -o %x_%j.out
#SBATCH --signal=B:USR1@300       # 5 min before the limit the batch shell gets USR1; the trap below forwards it so the run saves last.pt and exits

REPO=${REPO:-$SLURM_SUBMIT_DIR}                                   # the cloned repo (submit from its root)
DATA=${DATA:-$SCRATCH/pretraining_data}                           # flux.npy ivar.npy mask.npy z.npy
OUT=${OUT:-$SCRATCH/eidos_pretrain/run1}
STEPS=${STEPS:-30000}
EVAL_EVERY=${EVAL_EVERY:-1000}

# module load pytorch            # NERSC's PyTorch module (check `module avail pytorch`), or: source activate your_env
export PYTHONPATH=$REPO/src:$PYTHONPATH
export OMP_NUM_THREADS=4
mkdir -p "$OUT"
cd "$REPO"

srun python -u -m DESIFlow.training.pretrain \
    --data-dir "$DATA" --out "$OUT" --resume \
    --model full --arm pcgrad --batch 64 --steps "$STEPS" \
    --eval-every "$EVAL_EVERY" --recon-every "$EVAL_EVERY" --save-every 1000 \
    --workers 16 --max-hours 11.5 &
PID=$!
trap 'kill -USR1 $PID' USR1
wait $PID
wait $PID                                                         # the first wait returns early when the trap fires
