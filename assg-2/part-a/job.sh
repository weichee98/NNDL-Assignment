#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=part-a
#SBATCH --output=part-a.out
#SBATCH --error=part-a.err

module load anaconda
python train.py
python figure.py