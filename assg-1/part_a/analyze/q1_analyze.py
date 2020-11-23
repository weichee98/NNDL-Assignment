import argparse

import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.dict_json import read_json_to_dict
from utils.acc_loss import acc_converge_epoch, loss_converge_epoch

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to result json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q1_analyze.py...')

logger.log('Loading \"' + args.data + '\"')
histories = read_json_to_dict(args.data)

def q1a():
    logger.log('Analyzing Q1(a)...')
    training_accuracy = histories['Q1']['accuracy']
    testing_accuracy = histories['Q1']['val_accuracy']

    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(training_accuracy[1:], label='training')
    plt.plot(testing_accuracy[1:], label='testing')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    plt.title('Training and Testing Accuracies against Training Epochs.', fontsize=20)
    ax.axvline(
        x=acc_converge_epoch(testing_accuracy), 
        linestyle='--', linewidth=1, color='r'
    )
    plt.tight_layout()
    plt.savefig('result/AQ1_accuracy.png')
    logger.log('Saved result to \"result/AQ1_accuracy.png\"')

def q1b():
    logger.log('Analyzing Q1(b)...')
    training_loss = histories['Q1']['loss']
    testing_loss = histories['Q1']['val_loss']

    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(training_loss, label='training')
    plt.plot(testing_loss, label='testing')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('Training and Testing Losses against Training Epochs', fontsize=20)
    ax.axvline(
        x=loss_converge_epoch(testing_loss, mean_thres=0.01), 
        linestyle='--', linewidth=1, color='r'
    )
    plt.tight_layout()
    plt.savefig('result/AQ1_loss.png')
    logger.log('Saved result to \"result/AQ1_loss.png\"')

q1a()
q1b()
logger.end('Stopped q1_analyze.py')