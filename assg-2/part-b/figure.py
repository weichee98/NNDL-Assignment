import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from utils.acc_loss import smooth_curve, training_result


def LossAccPlot(file, model_name):
    result = pd.read_csv(file)
    train_loss = result['loss']
    test_loss = result['val_loss']
    train_acc = result['accuracy']
    test_acc = result['val_accuracy']

    file_name = file.split('/')[-1].split('.')[0]

    # Accuracy
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
    plt.plot(range(1, len(test_acc) + 1), test_acc, label='Test')
    plt.title(model_name + ' Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'./results/{file_name}_accuracy.png')
    plt.close()

    # Loss
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
    plt.plot(range(1, len(test_loss) + 1), test_loss, label='Test')
    plt.title(model_name + ' Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'./results/{file_name}_loss.png')
    plt.close()



def SGDvsAdam(sgd_file, adam_file):
    sgd = pd.read_csv(sgd_file)
    adam = pd.read_csv(adam_file)

    sgd_train_loss = sgd['loss']
    sgd_test_loss = sgd['val_loss']
    sgd_train_acc = sgd['accuracy']
    sgd_test_acc = sgd['val_accuracy']

    adam_train_loss = adam['loss']
    adam_test_loss = adam['val_loss']
    adam_train_acc = adam['accuracy']
    adam_test_acc = adam['val_accuracy']

    sgd_name = sgd_file.split('/')[-1].split('.')[0]
    model_name = sgd_name.replace('_SGD', '')
    title_model = ' '.join(model_name.split('_')).title()

    # SGD vs Adam Test Loss
    plt.plot(range(1, len(sgd_test_loss) + 1), sgd_test_loss, label='SGD')
    plt.plot(range(1, len(adam_test_loss) + 1), adam_test_loss, label='Adam')
    plt.title(f'{title_model} Loss Comparison')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'./results/{model_name}_loss_SGDvsAdam.png')
    plt.close()

    # SGD vs Adam Test Loss
    plt.plot(range(1, len(sgd_test_acc) + 1), sgd_test_acc, label='SGD')
    plt.plot(range(1, len(adam_test_acc) + 1), adam_test_acc, label='Adam')
    plt.title(f'{title_model} Accuracy Comparison')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'./results/{model_name}_accuracy_SGDvsAdam.png')
    plt.close()


def Q1_4():
    models = {
        'char_cnn_Adam_no_dropout': 'Char CNN',
        'char_gru_Adam_no_dropout': 'Char GRU',
        'word_cnn_Adam_no_dropout': 'Word CNN',
        'word_gru_Adam_no_dropout': 'Word GRU'
    }
    for model in models:
        LossAccPlot('./logs/' + model, models[model])


def Q5():
    models = {
        'char_cnn_Adam_no_dropout': 'Char CNN No Dropout',
        'char_gru_Adam_no_dropout': 'Char GRU No Dropout',
        'word_cnn_Adam_no_dropout': 'Word CNN No Dropout',
        'word_gru_Adam_no_dropout': 'Word GRU No Dropout',
        'char_cnn_Adam_dropout': 'Char CNN Dropout',
        'char_gru_Adam_dropout': 'Char GRU Dropout',
        'word_cnn_Adam_dropout': 'Word CNN Dropout',
        'word_gru_Adam_dropout': 'Word GRU Dropout'
    }
    
    time = pd.read_csv('./results/time.csv', index_col=0)
    time = time.div(250)

    f, ax = plt.subplots()
    time.plot(kind='barh', ax=ax)
    ax.set_title('Model Timing')
    ax.set_xlabel('time per epoch(s)')
    ax.set_ylabel('model')
    x_extra = (max(time.max()) - min(time.min())) * 0.2
    ax.set_xlim(
        (max(0, min(time.min()) - x_extra), max(time.max()) + x_extra)
    )
    for p in ax.patches:
        ax.annotate(
            "{:.2f}".format(p.get_width()), 
            (p.get_width() + 0.1, p.get_y() + p.get_height() / 2), 
            va='center'
        )
    plt.tight_layout()
    plt.savefig(f'./results/Q5_time.png')
    plt.close()

    results = {model: pd.read_csv('./logs/' + model) for model in models.keys()}
    acc_df = pd.DataFrame({
        'No Dropout': {
            models[model].replace(' No Dropout', ''): results[model]['val_accuracy'] 
            for model in models if 'no_dropout' in model
        },
        'Dropout': {
            models[model].replace(' Dropout', ''): results[model]['val_accuracy'] 
            for model in models if 'dropout' in model and 'no_dropout' not in model
        }
    })
    loss_df = pd.DataFrame({
        'No Dropout': {
            models[model].replace(' No Dropout', ''): results[model]['val_loss'] 
            for model in models if 'no_dropout' in model
        },
        'Dropout': {
            models[model].replace(' Dropout', ''): results[model]['val_loss'] 
            for model in models if 'dropout' in model and 'no_dropout' not in model
        }
    })

    acc_df = acc_df.applymap(lambda x: training_result(x, mode='acc'))
    f, ax = plt.subplots()
    acc_df.plot(kind='barh', ax=ax)
    ax.set_title('Model Test Accuracies Comparison')
    ax.set_xlabel('accuracy')
    ax.set_ylabel('model')
    x_extra = (max(acc_df.max()) - min(acc_df.min())) * 0.2
    ax.set_xlim(
        (max(0, min(acc_df.min()) - x_extra), max(acc_df.max()) + x_extra)
    )
    for p in ax.patches:
        ax.annotate(
            "{:.5f}".format(p.get_width()), 
            (p.get_width() + 0.005, p.get_y() + p.get_height() / 2), 
            va='center'
        )
    plt.tight_layout()
    plt.savefig(f'./results/Q5_accuracy_comparison.png')
    plt.close()

    loss_df = loss_df.applymap(lambda x: training_result(x, mode='loss'))
    f, ax = plt.subplots()
    loss_df.plot(kind='barh', ax=ax)
    ax.set_title('Model Test Loss Comparison')
    ax.set_xlabel('loss')
    ax.set_ylabel('model')
    x_extra = (max(loss_df.max()) - min(loss_df.min())) * 0.2
    ax.set_xlim(
        (max(0, min(loss_df.min()) - x_extra), max(loss_df.max()) + x_extra)
    )
    for p in ax.patches:
        ax.annotate(
            "{:.5f}".format(p.get_width()), 
            (p.get_width() + 0.05, p.get_y() + p.get_height() / 2), 
            va='center'
        )
    plt.tight_layout()
    plt.savefig(f'./results/Q5_loss_comparison.png')
    plt.close()

    acc_results = pd.DataFrame({
        models[model]: results[model]['val_accuracy'] for model in results
    })
    acc_results.plot()
    plt.title('Model Test Accuracies')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/Q5_accuracy_epoch.png')
    plt.close()

    loss_results = pd.DataFrame({
        models[model]: results[model]['val_loss'] for model in results
    })
    loss_results.plot()
    plt.title('Model Test Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/Q5_loss_epoch.png')
    plt.close()


def Q6():
    models = {
        'char_gru_Adam_no_dropout': 'Char GRU',
        'char_vanilla_Adam': 'Char Vanilla',
        'char_lstm_Adam': 'Char LSTM',
        'char_gru_2_layers_Adam': 'Char 2-Layer GRU',
        'char_gru_Adam_gradient_clipping': 'Char GRU with\nGradient Clipping',
        'word_gru_Adam_no_dropout': 'Word GRU',
        'word_vanilla_Adam': 'Word Vanilla',
        'word_lstm_Adam': 'Word LSTM',
        'word_gru_2_layers_Adam': 'Word 2-Layer GRU',
        'word_gru_Adam_gradient_clipping': 'Word GRU with\nGradient Clipping'
    }
    for model in models:
        LossAccPlot('./logs/' + model, models[model])

    results = {model: pd.read_csv('./logs/' + model) for model in models.keys()}
    acc_df = pd.DataFrame({
        'Char': {
            models[model].replace('Char ', ''): results[model]['val_accuracy'] 
            for model in models if 'char' in model
        },
        'Word': {
            models[model].replace('Word ', ''): results[model]['val_accuracy'] 
            for model in models if 'word' in model
        }
    })
    loss_df = pd.DataFrame({
        'Char': {
            models[model].replace('Char ', ''): results[model]['val_loss'] 
            for model in models if 'char' in model
        },
        'Word': {
            models[model].replace('Word ', ''): results[model]['val_loss'] 
            for model in models if 'word' in model
        }
    })

    acc_df = acc_df.applymap(lambda x: training_result(x, mode='acc'))
    f, ax = plt.subplots(figsize=(7.4, 5.8))
    acc_df.plot(kind='barh', ax=ax)
    ax.set_title('Model Test Accuracies Comparison')
    ax.set_xlabel('accuracy')
    ax.set_ylabel('model')
    x_extra = (max(acc_df.max()) - min(acc_df.min())) * 0.2
    ax.set_xlim(
        (max(0, min(acc_df.min()) - x_extra), max(acc_df.max()) + x_extra)
    )
    for p in ax.patches:
        ax.annotate(
            "{:.5f}".format(p.get_width()), 
            (p.get_width() + 0.005, p.get_y() + p.get_height() / 2), 
            va='center'
        )
    ax.legend(loc='right', bbox_to_anchor=(1.0, 0.3))
    plt.tight_layout()
    plt.savefig(f'./results/Q6_accuracy_comparison.png')
    plt.close()

    loss_df = loss_df.applymap(lambda x: training_result(x, mode='loss'))
    f, ax = plt.subplots(figsize=(7.4, 5.8))
    loss_df.plot(kind='barh', ax=ax)
    ax.set_title('Model Test Loss Comparison')
    ax.set_xlabel('loss')
    ax.set_ylabel('model')
    x_extra = (max(loss_df.max()) - min(loss_df.min())) * 0.2
    ax.set_xlim(
        (max(0, min(loss_df.min()) - x_extra), max(loss_df.max()) + x_extra)
    )
    for p in ax.patches:
        ax.annotate(
            "{:.5f}".format(p.get_width()), 
            (p.get_width() + 0.05, p.get_y() + p.get_height() / 2), 
            va='center'
        )
    plt.tight_layout()
    plt.savefig(f'./results/Q6_loss_comparison.png')
    plt.close()

    char_acc_results = pd.DataFrame({
        models[model].replace('Char ', ''): results[model]['val_accuracy'] 
        for model in results if 'char' in model
    })
    char_acc_results.plot()
    plt.title('Char Model Test Accuracies')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/Q6_char_accuracy_epoch.png')
    plt.close()

    char_loss_results = pd.DataFrame({
        models[model].replace('Char ', ''): results[model]['val_loss'] 
        for model in results if 'char' in model
    })
    char_loss_results.plot()
    plt.title('Char Model Test Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/Q6_char_loss_epoch.png')
    plt.close()

    word_acc_results = pd.DataFrame({
        models[model].replace('Word ', ''): results[model]['val_accuracy'] 
        for model in results if 'word' in model
    })
    word_acc_results.plot()
    plt.title('Word Model Test Accuracies')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/Q6_word_accuracy_epoch.png')
    plt.close()

    word_loss_results = pd.DataFrame({
        models[model].replace('Word ', ''): results[model]['val_loss'] 
        for model in results if 'word' in model
    })
    word_loss_results.plot()
    plt.title('Word Model Test Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/Q6_word_loss_epoch.png')
    plt.close()


if __name__ == '__main__':
    models = [
        ('cnn_SGD_no_dropout', 'cnn_Adam_no_dropout'),
        ('cnn_SGD_dropout', 'cnn_Adam_dropout'),
        ('gru_SGD_no_dropout', 'gru_Adam_no_dropout'),
        ('gru_SGD_dropout', 'gru_Adam_dropout'),
        ('vanilla_SGD', 'vanilla_Adam'),
        ('lstm_SGD', 'lstm_Adam'),
        ('gru_2_layers_SGD', 'gru_2_layers_Adam'),
        ('gru_SGD_gradient_clipping', 'gru_Adam_gradient_clipping')
    ]
    Q1_4()
    Q5()
    Q6()
    