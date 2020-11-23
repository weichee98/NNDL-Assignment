import argparse

import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.dict_json import read_json_to_dict
from utils.acc_loss import loss_converge_epoch

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to result json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q1_analyze.py...')

logger.log('Loading \"' + args.data + '\"')
histories = read_json_to_dict(args.data)

def q1a():
    logger.log('Analyzing Q1(a)...')
    training_error = histories['Q1']['a']['mse']
    testing_error = histories['Q1']['a']['val_mse']

    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(training_error, label='training')
    plt.plot(testing_error, label='testing')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('mean squared error', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('Training and Testing Error against Training Epochs', fontsize=20)
    ax.axvline(
        x=loss_converge_epoch(testing_error), 
        linestyle='--', linewidth=1, color='r'
    )
    plt.tight_layout()

    plt.savefig('result/BQ1_error.png')
    logger.log('Saved result to \"result/BQ1_error.png\"')

def q1b():
    logger.log('Analyzing Q1(b)...')
    real_values = histories['Q1']['c']['real'][:50]
    predicted_values = histories['Q1']['c']['predicted'][:50]

    f, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(x=real_values, y=predicted_values, marker='x')
    plt.xlabel('real values', fontsize=18)
    plt.ylabel('predicted values', fontsize=18)
    plt.title('Predicted Values against Real Values', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    min_x = min(min(real_values), min(predicted_values))
    max_x = max(max(real_values), max(predicted_values))
    r = (min_x - 0.1 * (max_x - min_x)), (max_x + 0.1 * (max_x - min_x))
    plt.plot(r, r, color="r", linestyle="dashed", linewidth=1.0)

    plt.savefig('result/BQ1_real_predict.png')
    logger.log('Saved result to \"result/BQ1_real_predict.png\"')

q1a()
q1b()
logger.end('Stopped q1_analyze.py')