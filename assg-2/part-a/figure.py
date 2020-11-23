import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from utils.acc_loss import smooth_curve, training_result


def Q1():

    num_ch_c1 = 50
    num_ch_c2 = 60
    optimizer_ = 'SGD'

    result_log = f'./logs/Q1_{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout'
    result_df = pd.read_csv(result_log)

    # Save the plot for losses
    train_loss = result_df['loss']
    test_loss = result_df['val_loss']
    smooth_loss = smooth_curve(result_df['val_loss'])
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
    plt.plot(range(1, len(test_loss) + 1), test_loss, label='Test', alpha=0.5)
    plt.plot(range(1, len(smooth_loss) + 1), smooth_loss, label='Smooth Test')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(
        f'./results/Q1_{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_loss.png'
    )
    plt.close()

    # Save the plot for accuracies
    train_acc = result_df['accuracy']
    test_acc = result_df['val_accuracy']
    smooth_acc = smooth_curve(result_df['val_accuracy'])
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
    plt.plot(range(1, len(test_acc) + 1), test_acc, label='Test', alpha=0.5)
    plt.plot(range(1, len(smooth_acc) + 1), smooth_acc, label='Smooth Test')
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(
        f'./results/Q1_{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_accuracy.png'
    )
    plt.close()


def Q2():
    hist_df = pd.read_csv('./results/c1c2_test_accuracy.csv', index_col=0)
    opt_c1 = hist_df.max().idxmax()
    opt_c2 = hist_df[opt_c1].idxmax()

    sb.heatmap(hist_df, square=True, annot=True, fmt='.3f', cmap='Blues')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('number of channel in C2')
    plt.xlabel('number of channel in C1')
    plt.savefig('./results/Q2_model_accuracy_heatmap.png')
    plt.close()

    return opt_c1, opt_c2


def Q3(num_ch_c1, num_ch_c2):

    config = {
        0: {'use_dropout': False, 'optimizer_': 'SGD'},
        1: {'use_dropout': False, 'optimizer_': 'SGD-momentum'},
        2: {'use_dropout': False, 'optimizer_': 'RMSProp'},
        3: {'use_dropout': False, 'optimizer_': 'Adam'},
        4: {'use_dropout': True, 'optimizer_': 'SGD'}
    }

    accuracies = dict()
    losses = dict()

    for c in config.values():
        use_dropout = c['use_dropout']
        optimizer_ = c['optimizer_']

        if use_dropout:
            dropout = 'dropout'
        else:
            dropout = 'no_dropout'

        result_log = f'./logs/{num_ch_c1}_{num_ch_c2}_{optimizer_}_{dropout}'

        result_df = pd.read_csv(result_log)
        if use_dropout:
            label = f'{optimizer_} Dropout'
        else:
            label = f'{optimizer_} No Dropout'
        accuracies[label] = result_df['val_accuracy']
        losses[label] = result_df['val_loss']

        # Save the plot for losses
        train_loss = result_df['loss']
        test_loss = result_df['val_loss']
        smooth_loss = smooth_curve(result_df['val_loss'])
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
        plt.plot(range(1, len(smooth_loss) + 1), smooth_loss, label='Test')
        plt.title(f'{label} Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(
            f'./results/Q3_{num_ch_c1}_{num_ch_c2}_{optimizer_}_{dropout}_loss.png'
        )
        plt.close()

        # Save the plot for accuracies
        train_acc = result_df['accuracy']
        test_acc = result_df['val_accuracy']
        smooth_acc = smooth_curve(result_df['val_accuracy'])
        plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
        plt.plot(range(1, len(smooth_acc) + 1), smooth_acc, label='Test')
        plt.title(f'{label} Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(
            f'./results/Q3_{num_ch_c1}_{num_ch_c2}_{optimizer_}_{dropout}_accuracy.png'
        )
        plt.close()

    accuracies = pd.DataFrame(accuracies)
    losses = pd.DataFrame(losses)

    accuracies.apply(smooth_curve, axis=0).plot()
    plt.title('Model Accuracy Comparison')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('./results/Q3_model_accuracy_comparison_epoch.png')
    plt.close()

    losses.apply(smooth_curve, axis=0).plot()
    plt.yscale('log')
    plt.title('Model Loss Comparison')
    plt.ylabel('loss (log)')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig('./results/Q3_model_loss_comparison_epoch.png')
    plt.close()

    accuracies = accuracies.apply(lambda x: training_result(x, mode='acc'), axis=0)
    losses = losses.apply(lambda x: training_result(x, mode='loss'), axis=0)
    
    f, ax = plt.subplots(1, 1, figsize=(5, 6))
    accuracies.plot(kind='bar')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylabel('accuracy')
    y_extra = (max(accuracies) - min(accuracies)) * 0.2
    ax.set_ylim(
        (max(0, min(accuracies) - y_extra), max(accuracies) + y_extra)
    )
    for p in ax.patches:
        ax.annotate("{:.5f}".format(p.get_height()), (p.get_x() - p.get_width() * 0.15, p.get_height() + y_extra / 3))
    plt.tight_layout()
    plt.savefig('./results/Q3_model_accuracy_comparison_bar.png')
    plt.close()

    f, ax = plt.subplots(1, 1, figsize=(5, 6))
    losses.plot(kind='bar')
    ax.set_title('Model Loss Comparison')
    ax.set_ylabel('loss')
    y_extra = (max(losses) - min(losses)) * 0.2
    ax.set_ylim(
        (max(0, min(losses) - y_extra), max(losses) + y_extra)
    )
    for p in ax.patches:
        ax.annotate("{:.5f}".format(p.get_height()), (p.get_x() - p.get_width() * 0.15, p.get_height() + y_extra / 3))
    plt.tight_layout()
    plt.savefig('./results/Q3_model_loss_comparison_bar.png')
    plt.close()


if __name__ == '__main__':
    Q1()
    opt_c1, opt_c2 = Q2()
    Q3(opt_c1, opt_c2)