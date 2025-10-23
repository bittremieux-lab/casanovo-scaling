#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=one_day
#SBATCH --mem=50G
#SBATCH --cpus-per-task=3

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate datasail

zenodo_get 6602020 -R 5 -e -g *lysn_meta* -o /mnt/data/cdens/casanovo-scaling/prospect_data/non-tryptic_data &
zenodo_get 6602020 -R 5 -e -g *HLA* -o /mnt/data/cdens/casanovo-scaling/prospect_data/non-tryptic_data &

zenodo_get 8003138 -R 5 -e -g *aspn* -o /mnt/data/cdens/casanovo-scaling/prospect_data/non-tryptic_data &

wait
