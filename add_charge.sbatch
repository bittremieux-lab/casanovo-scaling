#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --partition=one_day
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate casanovo

python massivekb_data/add_charge.py