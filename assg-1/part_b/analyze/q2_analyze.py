import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from utils.logger import Logger
from utils.dict_json import read_json_to_dict, write_dict_to_json
from utils.acc_loss import training_result, loss_converge_epoch

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to result json file for Q1 and Q2', required=True, nargs=2)
parser.add_argument('-DS', '--dataset', help='Path to the dataset csv file', required=True)
parser.add_argument('-P', '--params', help='Path to hyperparameters json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q2_analyze.py...')

logger.log('Loading \"' + args.data[0] + '\" and \"' + args.data[1] + '\"')
hist1 = read_json_to_dict(args.data[0])
hist2 = read_json_to_dict(args.data[1])


# Hyperparameters
hyperparameters = read_json_to_dict(args.params)


# Setup data to be used
df = pd.DataFrame(hist2['Q2']).applymap(lambda x: training_result(x, mode='loss'))
columns = hist2['columns']

def decode_col_name(col):
    idx = list(map(int, col.split(",")))
    removed_col = [columns[i] for i in idx]
    remaining_cols = list(set(columns) - set(removed_col))
    return ", ".join(remaining_cols)

val_mse = df[df.index == 'val_mse'].squeeze()


def corr_coef():
    logger.log('Plotting correlation coefficient of features...')
    cdf = pd.read_csv(args.dataset)
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    sb.heatmap(cdf[cdf.columns[1:]].corr(), annot=True, fmt='.3f', square=True, cmap='Blues')
    plt.tight_layout()
    plt.savefig('result/BQ2_corr.png')
    logger.log('Saved result to \"result/BQ2_corr.png\"')


def rfe_n_features(n):
    logger.log('Analyzing Q2 RFE %s features...' % (n))
    f, ax = plt.subplots(1, 1, figsize=(10, 12))

    cols = [c for c in df.columns if len(c.split(",")) == len(columns) - n]
    to_plot = val_mse[cols]
    to_plot = to_plot.rename(lambda x: decode_col_name(x))
    to_plot.plot(kind='bar')
    ax.set_title('Mean Squared Error against Features Used', fontsize=20, pad=20)
    ax.set_xlabel('features used', fontsize=18)
    ax.set_ylabel('mean squared error', fontsize=18)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    y_extra = (max(to_plot) - min(to_plot)) * 0.2
    ax.set_ylim(
        (min(to_plot) - y_extra, max(to_plot) + y_extra)
    )

    plt.tight_layout()
    plt.savefig('result/BQ2_rfe_%s.png' % (n))
    logger.log('Saved result to \"result/BQ2_rfe_%s.png\"' % (n))


def compare():
    logger.log('Compare all features, 6 features and 5 features')

    mse_all = hist1['Q1']['a']['val_mse']
    best_6 = val_mse[[c for c in df.columns if len(c.split(",")) == 1]].idxmin()
    mse_6 = hist2['Q2'][best_6]['val_mse']
    best_5 = val_mse[[c for c in df.columns if len(c.split(",")) == 2]].idxmin()
    mse_5 = hist2['Q2'][best_5]['val_mse']

    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(mse_all, label='all features')
    plt.plot(mse_6, label='6 features')
    plt.plot(mse_5, label='5 features')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('mean squared error', fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('MSE for Different Numbers of Input Features', fontsize=20, pad=20)
    plt.xlim(1000, 5000)
    plt.ylim(0.0058, 0.010)
    plt.xticks(fontsize=12, wrap=True)
    plt.yticks(fontsize=12, wrap=True)

    plt.tight_layout()
    plt.savefig('result/BQ2_compare.png')
    logger.log('Saved result to \"result/BQ2_compare.png\"')

    final = {
        'mse_converge': {
            "all_features": loss_converge_epoch(mse_all),
            "6_features": loss_converge_epoch(mse_6),
            "5_features": loss_converge_epoch(mse_5)
        },

        'final_mse': {
            "all_features": training_result(mse_all, mode='loss'),
            "6_features": training_result(mse_6, mode='loss'),
            "5_features": training_result(mse_5, mode='loss')
        }
    }
    final_df = pd.DataFrame(final)
    final_df.to_csv('result/BQ2_result.csv')
    logger.log('Saved result to \"result/BQ2_result.csv\"')

    # update hyperparameter
    result = min(final['final_mse'].keys(), key=lambda x: final['final_mse'][x])
    if result == 'all_features':
        removed = []
    elif result == '6_features':
        removed = list(map(int, best_6.split(",")))
    elif result == '5_features':
        removed = list(map(int, best_5.split(",")))
    
    hyperparameters['input_shape'] = (len(columns) - len(removed),)
    hyperparameters['removed'] = removed
    write_dict_to_json(hyperparameters, args.params)

corr_coef()
rfe_n_features(6)
rfe_n_features(5)
compare()

logger.end('Stopped q2_analyze.py')