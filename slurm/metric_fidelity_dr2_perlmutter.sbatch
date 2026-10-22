#!/bin/bash
# Metric fidelity versus S/N on clean synthetic spectra, projected onto the DR2 set (src/DESIFlow/metric, docs/metric_fidelity_dr2.md). One Perlmutter GPU node, 1 GPU.
# Edit the account and paths, then:  sbatch slurm/metric_fidelity_dr2_perlmutter.sbatch
# Quick check first (the GPU account ends in _g; ENV_CMD has no commas; every default below can be overridden through --export):
#   sbatch -A desi_g -q debug -t 00:30:00 --export=ALL,ENV_CMD="module load pytorch/2.11.0",MAX_ROWS=20000,N_SUITE=100,REPS=4,OUT=$SCRATCH/metric_snr/debug slurm/metric_fidelity_dr2_perlmutter.sbatch
# The job log is  metric_fidelity_<jobid>.out  in the directory you ran sbatch from. Resubmitting the same script continues (--resume): finished bands and the S/N of every object are reused.
# Stage 1 simulates the fidelity curves on clean synthetic spectra (skipped when FIDELITY points to an existing fidelity.json, e.g. one made on a laptop); stage 2 streams the DR2 memory maps
# once for the median per-pixel S/N of every spectrum (CPU / I/O) and projects the curves. Cost guide: stage 1 ~ 1 s per (S/N, noise realisation) per band on a laptop RTX 4090.
#SBATCH -A CHANGE_ME_g            # your GPU allocation (name usually ends in _g)
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH -t 02:00:00
#SBATCH -J metric_fidelity
#SBATCH -o %x_%j.out

REPO=${REPO:-$SLURM_SUBMIT_DIR}                                   # the cloned repo (submit from its root)
DATA=${DATA:-$SCRATCH/V2_Real/pretraining_data}                   # flux.npy ivar.npy mask.npy z.npy (the DR2 pretraining set)
OUT=${OUT:-$SCRATCH/metric_snr/dr2}
FIDELITY=${FIDELITY:-}                                            # an existing fidelity.json: skip stage 1
N_SUITE=${N_SUITE:-250}                                           # clean synthetic spectra per redshift band
REPS=${REPS:-24}                                                  # noise realisations per S/N and band (the confidence interval)
DONORS=${DONORS:-1}                                               # 1: the noise profiles are those of random DR2 spectra; 0: the built-in analytic DESI-like profile
USE_MASKS=${USE_MASKS:-0}                                         # 1: also apply the donors' masks (needs DONORS=1)
MAX_ROWS=${MAX_ROWS:-}                                            # random subset of the survey for stage 2 (testing only)
EXTRA_FID=${EXTRA_FID:-}                                          # e.g. EXTRA_FID="--z-edges 0 0.5 1 1.5 2 3 6.5 --snr 32 16 8 4 2 1 0.5"
EXTRA_PROJ=${EXTRA_PROJ:-}                                        # e.g. EXTRA_PROJ="--cuts 0 1 2 3 4 --stress-targets 0.25 0.5"

ENV_CMD=${ENV_CMD:-}             # how to get the PyTorch environment, e.g. --export=ALL,ENV_CMD="module load pytorch/2.11.0"  (no commas) or ENV_CMD="source activate myenv"
if [ -n "$ENV_CMD" ]; then eval "$ENV_CMD"; fi
export PYTHONPATH=$REPO/src:$PYTHONPATH
export OMP_NUM_THREADS=8
mkdir -p "$OUT"
cd "$REPO"
echo "job $SLURM_JOB_ID on $(hostname) | repo $REPO | data $DATA | out $OUT | fidelity ${FIDELITY:-<simulate>} | n_suite $N_SUITE reps $REPS donors $DONORS masks $USE_MASKS max_rows ${MAX_ROWS:-all}"
python -c "import sys, torch; print('python', sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
    || { echo "ERROR: cannot import torch; pass the environment with --export=ALL,ENV_CMD=\"module load pytorch/...\""; exit 2; }
python -c "import DESIFlow.metric.fidelity, DESIFlow.metric.dr2_projection" || { echo "ERROR: cannot import DESIFlow from $REPO/src (submit from the repo root or set REPO=...)"; exit 3; }
for f in flux ivar mask z; do [ -f "$DATA/$f.npy" ] || { echo "ERROR: $DATA/$f.npy not found"; exit 4; }; done
python -c "import matplotlib" 2>/dev/null || echo "NOTE: matplotlib is not importable here; the tables are written and the figures are skipped (python -m DESIFlow.metric.plots <json> draws them wherever it is available)"

if [ -z "$FIDELITY" ]; then
    FIDELITY=$OUT/fidelity/fidelity.json
    echo "=== stage 1: fidelity simulation ==="
    FID_ARGS=()
    [ "$DONORS" = 1 ] && FID_ARGS+=(--noise-donors "$DATA")
    [ "$USE_MASKS" = 1 ] && FID_ARGS+=(--use-masks)
    srun python -u -m DESIFlow.metric.fidelity --out "$OUT/fidelity" --resume --n-suite "$N_SUITE" --reps "$REPS" "${FID_ARGS[@]}" $EXTRA_FID || exit 5
fi
echo "=== stage 2: DR2 projection with $FIDELITY ==="
srun python -u -m DESIFlow.metric.dr2_projection --data-dir "$DATA" --fidelity "$FIDELITY" --out "$OUT/projection" ${MAX_ROWS:+--max-rows $MAX_ROWS} $EXTRA_PROJ
