#!/bin/bash

#SBATCH --job-name=renes_trpo
#SBATCH --output=logs/renes_%A_%a.out
#SBATCH --error=logs/renes_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --partition=nocona
#SBATCH --exclusive

# --- Create logs directory if it doesn't exist
mkdir -p logs

# --- Set up environment
ml load gcc

# Initialize conda using the full path
. "/home/mibeebe/sw/el8-x86_64/miniconda3/etc/profile.d/conda.sh"
conda activate cs5388

# --- Array of solvers
declare -a solvers=("fp" "ce" "prd" "alpha_rank")
solver=${solvers[$SLURM_ARRAY_TASK_ID]}

# --- Print solver information to output files
echo "----------------------------------------"
echo "  Running with solver: ${solver}"
echo "  Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "  Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "  Start time: $(date)"
echo "----------------------------------------"

# --- Create a directory for this specific run
run_dir="results/${solver}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $run_dir

# --- Run the training script with the specific solver
python main_ppo_renes_pt_trpo.py            \
    --meta-solver $solver                   \
    --num-processes $SLURM_CPUS_PER_TASK    \
    --log-dir $run_dir                      \
    --save-dir $run_dir

# python main_ppo_renes_pt.py                 \
#     --meta-solver $solver                   \
#     --num-processes $SLURM_CPUS_PER_TASK    \
#     --log-dir $run_dir                      \
#     --save-dir $run_dir                     \
#     2>&1 | tee $run_dir/training.log

# --- Copy the slurm output files to the run directory
cp logs/renes_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.* $run_dir/

