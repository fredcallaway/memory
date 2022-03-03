#!/usr/bin/env bash
#SBATCH --job-name=exp2
#SBATCH --output=slurm_out/%A_%a
#SBATCH --array=1-250
#SBATCH --time=60
#SBATCH --mem-per-cpu=4500
#SBATCH --cpus-per-task=1

~/bin/julia17 exp2_main.jl $SLURM_ARRAY_TASK_ID

