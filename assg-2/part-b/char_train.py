import os
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf

from utils.load_data import read_data_chars
from utils.logger import Logger
from utils.model import CharCNN, CharVanilla, CharLSTM, CharGRU1, CharGRU2


parser = argparse.ArgumentParser()
parser.add_argument('-O', '--optimizer', help='Optimizer to use (SGD or Adam)', required=True, default='SGD')
args = parser.parse_args()


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
NUM_CHAR = 256
EPOCHS = 250
BATCH_SIZE = 128
LR = 0.01
DROP_RATE = 0.5
CLIPPING = 2.0


def train(model, train_ds, test_ds, title, filename, dropout=0, clipping=False):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    if args.optimizer.upper() == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
    elif args.optimizer.title() == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    else:
        raise Exception('Invalid optimizer, please input either SGD or Adam')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def train_step(model, x, label, drop_rate):
        with tf.GradientTape() as tape:
            out = model(x, drop_rate)
            loss = loss_object(label, out)
            gradients = tape.gradient(loss, model.trainable_variables)
            if clipping:
                gradients, _ = tf.clip_by_global_norm(gradients, clipping)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, out)

    def test_step(model, x, label):
        out = model(x, drop_rate=0)
        t_loss = loss_object(label, out)
        test_loss(t_loss)
        test_accuracy(label, out)

    logger.log('Start' + title)

    training_loss = []
    testing_loss = []
    training_acc = []
    testing_acc = []

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for chars, labels in train_ds:
            train_step(model, chars, labels, drop_rate=dropout)

        for chars, labels in test_ds:
            test_step(model, chars, labels)

        training_loss.append(train_loss.result().numpy())
        testing_loss.append(test_loss.result().numpy())
        training_acc.append(train_accuracy.result().numpy())
        testing_acc.append(test_accuracy.result().numpy())
        
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        logger.log(template.format(epoch + 1, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()))

    histories = pd.DataFrame({
        'loss': training_loss,
        'val_loss': testing_loss,
        'accuracy': training_acc,
        'val_accuracy': testing_acc
    })
    histories.to_csv('./logs/' + filename)
    logger.end('Done' + title)


if __name__ == '__main__':
    seed = 10
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load train and test data
    char_x_train, char_y_train, char_x_test, char_y_test = read_data_chars(
        train_file='./train_medium.csv',
        test_file='./test_medium.csv'
    )
    print(char_x_train.shape)
    char_train_ds = tf.data.Dataset.from_tensor_slices(
        (char_x_train, char_y_train)).shuffle(10000).batch(BATCH_SIZE)
    char_test_ds = tf.data.Dataset.from_tensor_slices((char_x_test, char_y_test)).batch(BATCH_SIZE)

    # Create folder to store models and results
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    train(
        model=CharCNN(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char CNN Without Dropout', filename=f'char_cnn_{args.optimizer}_no_dropout'
    )

    train(
        model=CharCNN(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char CNN With Dropout', filename=f'char_cnn_{args.optimizer}_dropout', dropout=DROP_RATE
    )


    train(
        model=CharGRU1(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char GRU Without Dropout', filename=f'char_gru_{args.optimizer}_no_dropout'
    )

    train(
        model=CharGRU1(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char GRU With Dropout', filename=f'char_gru_{args.optimizer}_dropout', dropout=DROP_RATE
    )
    
    train(
        model=CharVanilla(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char Vanilla RNN', filename=f'char_vanilla_{args.optimizer}'
    )

    train(
        model=CharLSTM(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char LSTM', filename=f'char_lstm_{args.optimizer}'
    )

    train(
        model=CharGRU2(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char GRU 2-Layers', filename=f'char_gru_2_layers_{args.optimizer}'
    )

    train(
        model=CharGRU1(NUM_CHAR), train_ds=char_train_ds, test_ds=char_test_ds, 
        title='Train Char GRU With Gradient Clipping', filename=f'char_gru_{args.optimizer}_gradient_clipping', clipping=CLIPPING
    )


    
