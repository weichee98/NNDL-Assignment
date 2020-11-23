import argparse
import pandas as pd
import matplotlib.pyplot as plt

from utils.logger import Logger
from utils.dict_json import read_json_to_dict
from utils.acc_loss import training_result, loss_converge_epoch

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to result json file for Q3', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q3_analyze.py...')

logger.log('Loading \"' + args.data + '\"')
histories = read_json_to_dict(args.data)

def compare():
    df = pd.DataFrame(histories['Q3'])
    val_mse = df[df.index == 'val_mse'].apply(lambda x: x.explode()).reset_index(drop=True)
    f, ax = plt.subplots(1, 1, figsize=(15, 5))
    val_mse.plot()
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('mean squared error', fontsize=15)
    plt.legend(loc='upper right', fontsize=12)
    plt.title('MSE for Different Models', fontsize=15, pad=20)
    plt.xlim(500, 10000)
    plt.ylim(0.0055, 0.0115)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('result/BQ3_compare.png')
    logger.log('Saved result to \"result/BQ3_compare.png\"')

    df.applymap(lambda x: training_result(x, mode='loss')).to_csv('result/BQ3_compare.csv')
    logger.log('Saved result to \"result/BQ3_compare.csv\"')

    df.applymap(lambda x: loss_converge_epoch(x)).to_csv('result/BQ3_converge_epoch.csv')
    logger.log('Saved result to \"result/BQ3_converge_epoch.csv\"')

compare()

logger.end('Stopped q3_analyze.py')