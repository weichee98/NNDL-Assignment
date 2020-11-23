import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils.logger import Logger
from utils.dict_json import read_json_to_dict
from utils.acc_loss import acc_converge_epoch, loss_converge_epoch, training_result

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', nargs=2, help='Path to the result json files for Q4 and additional', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting additional_analyze.py...')

logger.log('Loading \"' + args.data[0] + '\" and \"' + args.data[1] + '\"')
hist1 = read_json_to_dict(args.data[0])
hist2 = read_json_to_dict(args.data[1])
if hist1['seed'] == hist2['seed']:
    hist1.update(hist2)
    histories = hist1
else:
    raise Exception('Results for Q4 and additional have different seed values')


def additional_compare():
    q4_accuracy = histories['Q4']['optimal']['val_accuracy'][1:]
    q5_accuracy = histories['Q5']['val_accuracy'][1:]

    f, ax = plt.subplots(figsize=(10, 5))
    plt.plot(q4_accuracy, label='3-level network')
    plt.plot(q5_accuracy, label='4-level network')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    plt.title('3-Level Network vs Modified 4-Level Network', fontsize=20, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('result/additional_compare.png')
    logger.log('Saved result to \"result/additional_compare.png\"')

def additional_result():
    compare = {
        'acc_converge': {
            3: acc_converge_epoch(histories['Q4']['optimal']['val_accuracy']),
            4: acc_converge_epoch(histories['Q5']['val_accuracy'])
        },

        'loss_converge': {
            3: loss_converge_epoch(histories['Q4']['optimal']['val_loss']),
            4: loss_converge_epoch(histories['Q5']['val_loss'])
        },

        'final_acc': {
            3: training_result(histories['Q4']['optimal']['val_accuracy']),
            4: training_result(histories['Q5']['val_accuracy'])
        },

        'final_loss': {
            3: training_result(histories['Q4']['optimal']['val_loss'], mode='loss'),
            4: training_result(histories['Q5']['val_loss'], mode='loss')
        }
    }
    compare_df = pd.DataFrame(compare)
    compare_df.to_csv('result/additional_result.csv')
    logger.log('Saved result to \"result/additional_result.csv\"')


additional_compare()
additional_result()
logger.end('Stopped additional_analyze.py')