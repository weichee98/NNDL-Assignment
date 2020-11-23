import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from utils.load_data import load_data
from utils.keract import get_activations, display_activations
from utils.acc_loss import training_result
from utils.logger import Logger

# This is required when using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


logger = Logger()
EPOCHS = 1000
BATCH_SIZE = 128
LR = 0.001


def make_model(num_ch_c1, num_ch_c2, use_dropout=False, dropout_rate=0.5):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(filters=num_ch_c1, kernel_size=9, activation='relu', padding='valid', name='C1'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', name='S1'))
    model.add(tf.keras.layers.Conv2D(filters=num_ch_c2, kernel_size=5, activation='relu', padding='valid', name='C2'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', name='S2'))
    model.add(tf.keras.layers.Flatten())
    if use_dropout:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(300, name='F3'))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='F4'))
    return model


def Q1(x_train, y_train, x_test, y_test):
    logger.log('Start Q1')
    num_ch_c1 = 50
    num_ch_c2 = 60

    epochs = EPOCHS
    batch_size = BATCH_SIZE
    learning_rate = LR
    use_dropout = False

    logger.log('Create Model')
    model = make_model(num_ch_c1, num_ch_c2, use_dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer_ = 'SGD'
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    print(model.summary())

    logger.log('Training')
    with tqdm(total=epochs) as pbar:
        update = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda batch, logs: pbar.update(1)
        )
        logs = tf.keras.callbacks.CSVLogger(
            f'./logs/Q1_{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout', separator=',', append=False
        )
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=(x_test, y_test),
            verbose=0,
            callbacks=[update, logs]
        )
    model.save(f'./models/Q1_{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout.h5')

    logger.log('Plotting Output for C1, C2, S1, S2')
    
    activations_1 = get_activations(model=model, x=x_test[:1], layer_names=['C1', 'S1', 'C2', 'S2'])
    activations_2 = get_activations(model=model, x=x_test[1:2], layer_names=['C1', 'S1', 'C2', 'S2'])
    display_activations(activations_1, cmap="gray", save=True, directory='./results', filename='Test Image 1', fig_size=(20, 20))
    display_activations(activations_2, cmap="gray", save=True, directory='./results', filename='Test Image 2', fig_size=(20, 20))

    plt.imshow(x_test[0])
    plt.title('Test Image 1')
    plt.savefig(
        f'./results/Q1_{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_image1.png'
    )
    plt.close()

    plt.imshow(x_test[1])
    plt.title('Test Image 2')
    plt.savefig(
        f'./results/Q1_{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_image2.png'
    )
    plt.close()

    logger.end('Done Q1')


def Q2(x_train, y_train, x_test, y_test):
    logger.log('Start Q2')
    num_ch_c1 = [10, 30, 50, 70, 90]
    num_ch_c2 = [20, 40, 60, 80, 100]

    epochs = EPOCHS
    batch_size = BATCH_SIZE
    learning_rate = LR
    use_dropout = False

    histories = {c1: {c2: None for c2 in num_ch_c2} for c1 in num_ch_c1}

    for c1 in num_ch_c1:
        for c2 in num_ch_c2:
            logger.log(f'Train C1={c1} C2={c2}')
            model = make_model(c1, c2, use_dropout)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            optimizer_ = 'SGD'
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

            with tqdm(total=epochs) as pbar:
                update = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda batch, logs: pbar.update(1)
                )
                logs = tf.keras.callbacks.CSVLogger(
                    f'./logs/{c1}_{c2}_{optimizer_}_no_dropout', separator=',', append=False
                )
                history = model.fit(
                    x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    verbose=0,
                    callbacks=[update, logs]
                )
            model.save(f'./models/{c1}_{c2}_{optimizer_}_no_dropout.h5')
            histories[c1][c2] = history.history['val_accuracy']

    hist_df = pd.DataFrame(histories)
    hist_df = hist_df.applymap(lambda x: training_result(x, mode='acc'))
    hist_df.to_csv('./results/c1c2_test_accuracy.csv', header=True, index=True)
    opt_c1 = hist_df.max().idxmax()
    opt_c2 = hist_df[opt_c1].idxmax()
    logger.log(f'Best C1={opt_c1} C2={opt_c2}')
    logger.end('Done Q2')

    return opt_c1, opt_c2


def Q3(x_train, y_train, x_test, y_test, num_ch_c1, num_ch_c2):
    logger.log('Start Q3')
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    learning_rate = LR
    
    config = {
        0: {'use_dropout': False, 'optimizer_': 'SGD-momentum', 'optimizer': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.1)},
        1: {'use_dropout': False, 'optimizer_': 'RMSProp', 'optimizer': tf.keras.optimizers.RMSprop(learning_rate=learning_rate)},
        2: {'use_dropout': False, 'optimizer_': 'Adam', 'optimizer':tf.keras.optimizers.Adam(learning_rate=learning_rate) },
        3: {'use_dropout': True, 'dropout_rate': 0.5, 'optimizer_': 'SGD', 'optimizer': tf.keras.optimizers.SGD(learning_rate=learning_rate)}
    }

    for c in config.values():
        use_dropout = c['use_dropout']
        dropout_rate = c['dropout_rate'] if use_dropout else 0
        model = make_model(num_ch_c1=num_ch_c1, num_ch_c2=num_ch_c2, use_dropout=use_dropout, dropout_rate=dropout_rate)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer_ = c['optimizer_']
        optimizer = c['optimizer']
        model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

        logger.log(f'Train Optimizer={optimizer_} Dropout={use_dropout}')
        with tqdm(total=epochs) as pbar:
            update = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda batch, logs: pbar.update(1)
            )
            if use_dropout:
                logs = tf.keras.callbacks.CSVLogger(
                    f'./logs/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout', separator=',', append=False
                )
            else:
                logs = tf.keras.callbacks.CSVLogger(
                    f'./logs/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout', separator=',', append=False
                )
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                validation_data=(x_test, y_test),
                verbose=0,
                callbacks=[update, logs]
            )
        
        if use_dropout:
            model.save(f'./models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout.h5')
        else:
            model.save(f'./models/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout.h5')

    logger.end('Done Q3')


if __name__ == '__main__':
    seed = 10
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load train and test data
    x_train, y_train = load_data('data_batch_1')
    x_test, y_test = load_data('test_batch_trim')

    # x_train = x_train[:BATCH_SIZE]
    # y_train = y_train[:BATCH_SIZE]
    # x_test = x_test[:BATCH_SIZE]
    # y_test = y_test[:BATCH_SIZE]

    # Create folder to store models and results
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    Q1(x_train, y_train, x_test, y_test)
    opt_c1, opt_c2 = Q2(x_train, y_train, x_test, y_test)
    Q3(x_train, y_train, x_test, y_test, opt_c1, opt_c2)


    
