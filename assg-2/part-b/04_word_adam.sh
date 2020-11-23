#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=word_adam
#SBATCH --output=word_adam.out
#SBATCH --error=word_adam.err

module load anaconda
python word_train.py -O Adam