import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.dict_json import read_json_to_dict
from utils.acc_loss import acc_converge_epoch, loss_converge_epoch, smooth_curve, training_result

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to result json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q3_analyze.py...')

logger.log('Loading \"' + args.data + '\"')
histories = read_json_to_dict(args.data)
neuron_epoch = pd.DataFrame(histories['Q3']['cv']['accuracy'])
cv_accuracy = neuron_epoch.applymap(lambda x: training_result(x))
print(cv_accuracy)
cv_accuracy = cv_accuracy.mean()
num_neurons = histories['Q3']['optimal']['optimal_num']

def q3a1():
    logger.log('Analyzing Q3(a1)...')
    
    f, ax = plt.subplots(5, 1, figsize=(10, 25))
    for i, num in enumerate(neuron_epoch.columns, start=0):
        ax[i].plot(neuron_epoch[num][0][1:], label='Fold 1')
        ax[i].plot(neuron_epoch[num][1][1:], label='Fold 2')
        ax[i].plot(neuron_epoch[num][2][1:], label='Fold 3')
        ax[i].plot(neuron_epoch[num][3][1:], label='Fold 4')
        ax[i].plot(neuron_epoch[num][4][1:], label='Fold 5')

        ax[i].legend(loc='lower right', fontsize=12)
        ax[i].set_xlabel('epoch', fontsize=18)
        ax[i].set_ylabel('cross validation accuracy', fontsize=18)
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].tick_params(axis='y', labelsize=18)
        ax[i].set_title('Number of Neurons %s' % num, fontsize=20)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('result/AQ3_cv_epoch.png')
    logger.log('Saved result to \"result/AQ3_cv_epoch.png\"')


def q3a2():
    logger.log('Analyzing Q3(a2)...')
    f, ax = plt.subplots(5, 1, figsize=(10, 25))

    for i in range(5):
        for num in neuron_epoch.columns:
            ax[i].plot(smooth_curve(neuron_epoch[num][i][1:]), label='%s Neurons' % num)
        ax[i].legend(loc='lower right', fontsize=12)
        ax[i].set_xlabel('epoch', fontsize=18)
        ax[i].set_ylabel('cross validation accuracy', fontsize=18)
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].tick_params(axis='y', labelsize=18)
        ax[i].set_title('Smoothed CV Accuracies for Fold %s' % (i + 1), fontsize=20)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('result/AQ3_cv_epoch2.png')
    logger.log('Saved result to \"result/AQ3_cv_epoch2.png\"')

def q3b():
    logger.log('Analyzing Q3(b)...')
    f, ax = plt.subplots(1, 1, figsize=(10, 5))

    cv_accuracy.plot(kind='bar')
    ax.set_title('Cross Validation Accuracy against Number of Neurons', fontsize=20)
    ax.set_xlabel('number of neurons', fontsize=18)
    ax.set_ylabel('cross validation accuracy', fontsize=18)
    y_extra = (max(cv_accuracy) - min(cv_accuracy)) * 0.2
    ax.set_ylim(
        (min(cv_accuracy) - y_extra, max(cv_accuracy) + y_extra)
    )
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    for p in ax.patches:
        ax.annotate("{:.5f}".format(p.get_height()), (p.get_x() + p.get_width() * 0.1, p.get_height() + 0.0001), fontsize=12)
    plt.tight_layout(pad=3.0)
    plt.savefig('result/AQ3_decide.png')
    logger.log('Saved result to \"result/AQ3_decide.png\"')

def q3c():
    logger.log('Analyzing Q3(c)...')
    training_accuracy = histories['Q3']['optimal']['accuracy'][1:]
    testing_accuracy = histories['Q3']['optimal']['val_accuracy'][1:]

    f, ax = plt.subplots(figsize=(12, 6))
    plt.plot(training_accuracy, label='training')
    plt.plot(testing_accuracy, label='testing')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    plt.title('Training and Testing Accuracies against Training Epochs for Number of Neurons %s' % num_neurons, fontsize=20, pad=20)
    # ax.axvline(
    #     x=acc_converge_epoch(testing_accuracy), 
    #     linestyle='--', linewidth=1, color='r'
    # )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('result/AQ3_final.png')
    logger.log('Saved result to \"result/AQ3_final.png\"')

q3a1()
q3a2()
q3b()
q3c()
logger.end('Stopped q3_analyze.py')