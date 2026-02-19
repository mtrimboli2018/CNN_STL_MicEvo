#!/bin/bash
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --nodelist=gpu05
#SBATCH --output=output_trial.txt
#SBATCH --error=error_trial.txt

python3 CustomRuntime_simvp.py
