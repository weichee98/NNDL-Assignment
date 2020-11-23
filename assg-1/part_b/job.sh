#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --job-name=part-b
#SBATCH --output=part-b.out
#SBATCH --error=part-b.err

module load anaconda
export PYTHONPATH=$PYTHONPATH:/home/UG/yeww0006/assg-1/part_b/utils
# python src/q1_train.py -D ../admission_predict.csv -P src/params.json
# python analyze/q1_analyze.py -D result/BQ1.json

# python src/q2_train.py -D ../admission_predict.csv -P src/params.json
# python analyze/q2_analyze.py -D result/BQ1.json result/BQ2.json -DS ../admission_predict.csv -P src/params.json
# python src/q3_train.py -D ../admission_predict.csv -P src/params.json
# python analyze/q3_analyze.py -D result/BQ3.json

# python src/q2_train2.py -D ../admission_predict.csv -P src/params.json
# python analyze/q2_analyze2.py -D result/BQ1.json result/BQ2.json -DS ../admission_predict.csv -P src/params.json
# python src/q3_train2.py -D ../admission_predict.csv -P src/params.json
# python analyze/q3_analyze.py -D result/BQ3.json

python src/additional_train.py -D ../admission_predict.csv -P src/params.json
python analyze/additional_analyze.py -D result/additional.json