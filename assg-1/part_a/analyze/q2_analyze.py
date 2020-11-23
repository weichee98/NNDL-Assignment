import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.dict_json import read_json_to_dict
from utils.acc_loss import acc_converge_epoch, loss_converge_epoch, training_result

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to result json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q2_analyze.py...')

logger.log('Loading \"' + args.data + '\"')
histories = read_json_to_dict(args.data)
batch_epoch = pd.DataFrame(histories['Q2']['cv']['accuracy'])
time_per_epoch = pd.DataFrame(histories['Q2']['cv']['time'])
epoch_to_converge = batch_epoch.applymap(lambda x: acc_converge_epoch(x))
cv_accuracy = batch_epoch.applymap(lambda x: training_result(x))

total_time_to_converge = (epoch_to_converge * time_per_epoch).mean()
print(epoch_to_converge)
epoch_to_converge = epoch_to_converge.mean()
print(cv_accuracy)
cv_accuracy = cv_accuracy.mean()
batch_size = histories['Q2']['optimal']['optimal_batch']

def q2a1():
    logger.log('Analyzing Q2(a1)...')
    
    f, ax = plt.subplots(5, 1, figsize=(10, 25))
    for i, batch in enumerate(batch_epoch.columns, start=0):
        ax[i].plot(batch_epoch[batch][0][1:], label='Fold 1')
        ax[i].plot(batch_epoch[batch][1][1:], label='Fold 2')
        ax[i].plot(batch_epoch[batch][2][1:], label='Fold 3')
        ax[i].plot(batch_epoch[batch][3][1:], label='Fold 4')
        ax[i].plot(batch_epoch[batch][4][1:], label='Fold 5')
        ax[i].legend(loc='lower right', fontsize=12)
        ax[i].set_xlabel('epoch', fontsize=18)
        ax[i].set_ylabel('cross validation accuracy', fontsize=18)
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].tick_params(axis='y', labelsize=18)
        ax[i].set_title('Batch Size %s' % batch, fontsize=20)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('result/AQ2_cv_epoch.png')
    logger.log('Saved result to \"result/AQ2_cv_epoch.png\"')


def q2a2():
    logger.log('Analyzing Q2(a2)...')
    f, ax = plt.subplots(1, 2, figsize=(20, 6))

    time_per_epoch.mean().plot(marker='o', ax=ax[0])
    ax[0].set_title('Time per Epoch against Batch Size', fontsize=20)
    ax[0].set_xlabel('batch size', fontsize=18)
    ax[0].set_ylabel('time per epoch (s)', fontsize=18)
    ax[0].tick_params(axis='x', labelsize=18)
    ax[0].tick_params(axis='y', labelsize=18)
    
    epoch_to_converge.plot(marker='o', ax=ax[1])
    ax[1].set_title('Convergence Epoch against Batch Size', fontsize=20)
    ax[1].set_xlabel('batch size', fontsize=18)
    ax[1].set_ylabel('convergence epoch', fontsize=18)
    ax[1].tick_params(axis='x', labelsize=18)
    ax[1].tick_params(axis='y', labelsize=18)

    plt.tight_layout(pad=3.0)
    plt.savefig('result/AQ2_time.png')
    logger.log('Saved result to \"result/AQ2_time.png\"')

def q2b():
    logger.log('Analyzing Q2(b)...')
    f, ax = plt.subplots(1, 2, figsize=(20, 6))

    total_time_to_converge.plot(marker='o', ax=ax[0])
    ax[0].set_title('Total Time to Converge against Batch Size', fontsize=20)
    ax[0].set_xlabel('batch size', fontsize=18)
    ax[0].set_ylabel('total time (s)', fontsize=18)
    ax[0].tick_params(axis='x', labelsize=18)
    ax[0].tick_params(axis='y', labelsize=18)

    cv_accuracy.plot(marker='o', ax=ax[1])
    ax[1].set_title('Cross Validation Accuracy against Batch Size', fontsize=20)
    ax[1].set_xlabel('batch size', fontsize=18)
    ax[1].set_ylabel('cross validation accuracy', fontsize=18)
    ax[1].tick_params(axis='x', labelsize=18)
    ax[1].tick_params(axis='y', labelsize=18)

    plt.tight_layout(pad=3.0)
    plt.savefig('result/AQ2_decide.png')
    logger.log('Saved result to \"result/AQ2_decide.png\"')

def q2c():
    logger.log('Analyzing Q2(c)...')
    training_accuracy = histories['Q2']['optimal']['accuracy'][1:]
    testing_accuracy = histories['Q2']['optimal']['val_accuracy'][1:]

    f, ax = plt.subplots(figsize=(12, 6))
    plt.plot(training_accuracy, label='training')
    plt.plot(testing_accuracy, label='testing')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    plt.title('Training and Testing Accuracies against Training Epochs for Batch Size %s' % batch_size, fontsize=20)
    # ax.axvline(
    #     x=acc_converge_epoch(testing_accuracy), 
    #     linestyle='--', linewidth=1, color='r'
    # )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('result/AQ2_final.png')
    logger.log('Saved result to \"result/AQ2_final.png\"')

q2a1()
q2a2()
q2b()
q2c()
logger.end('Stopped q2_analyze.py')