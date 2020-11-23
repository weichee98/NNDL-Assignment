#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=part-a
#SBATCH --output=part-a.out
#SBATCH --error=part-a.err

module load anaconda
export PYTHONPATH=$PYTHONPATH:/home/UG/yeww0006/assg-1/part_a/utils
python src/q1_train.py -D ../ctg_data_cleaned.csv -P src/params.json
python analyze/q1_analyze.py -D result/AQ1.json
python src/q2_train.py -D ../ctg_data_cleaned.csv -P src/params.json
python analyze/q2_analyze.py -D result/AQ2.json
python src/q3_train.py -D ../ctg_data_cleaned.csv -P src/params.json
python analyze/q3_analyze.py -D result/AQ3.json
python src/q4_train.py -D ../ctg_data_cleaned.csv -P src/params.json
python analyze/q4_analyze.py -D result/AQ4.json
python src/q5_train.py -D ../ctg_data_cleaned.csv
python analyze/q5_analyze.py -D result/AQ4.json result/AQ5.json