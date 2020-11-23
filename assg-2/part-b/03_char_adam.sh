#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=char_adam
#SBATCH --output=char_adam.out
#SBATCH --error=char_adam.err

module load anaconda
python char_train.py -O Adam