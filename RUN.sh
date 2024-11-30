#!/bin/bash

#SBATCH --job-name=cs5388_renes
#SBATCH --output=logs/solver_%A_%a.out
#SBATCH --error=logs/solver_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --partition=nocona
#SBATCH --exclusive

# --- Create logs directory if it doesn't exist
mkdir -p logs

# --- Set up python environment
conda activate cs5388

# --- Array of solvers
declare -a solvers=("alpha_rank" "fp" "ce" "prd")
solver=${solvers[$SLURM_ARRAY_TASK_ID]}

# --- Create a directory for this specific run
run_dir="results/${solver}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $run_dir

# --- Run the training script with the specific solver
python main_ppo_renes_pt.py \
    --meta-solver $solver \
    --num-processes $SLURM_CPUS_PER_TASK \
    --log-dir $run_dir \
    --save-dir $run_dir \
    2>&1 | tee $run_dir/training.log

# --- Copy the slurm output files to the run directory
cp logs/solver_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.* $run_dir/

