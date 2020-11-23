#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=char_sgd
#SBATCH --output=char_sgd.out
#SBATCH --error=char_sgd.err

module load anaconda
python char_train.py -O SGD