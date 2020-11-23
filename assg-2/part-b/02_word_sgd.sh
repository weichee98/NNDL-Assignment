#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=word_sgd
#SBATCH --output=word_sgd.out
#SBATCH --error=word_sgd.err

module load anaconda
python word_train.py -O SGD