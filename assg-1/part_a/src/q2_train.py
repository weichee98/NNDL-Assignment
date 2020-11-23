import argparse
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.logger import Logger
from utils.seed import SEED, init_seed
from utils.preprocess_dataset import PreprocessDataset
from utils.dict_json import filter_dict, write_dict_to_json, read_json_to_dict
from utils.acc_loss import acc_converge_epoch, training_result

init_seed()

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to dataset csv file', required=True)
parser.add_argument('-P', '--params', help='Path to hyperparameters json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q2_train.py...')

logger.log('Loading dataset from \"' + args.data + '\"...')
df = pd.read_csv(args.data)
dataset = PreprocessDataset(
    df=df, 
    feature_columns=df.columns[:21],
    label_column=df.columns[-1],
    test_ratio=0.3,
    fold=5
)

try:
    hyperparameters = read_json_to_dict(args.params)
except:
    hyperparameters = {
        "input_shape": (21,),
        "num_classes": 3,
        "batch_size": 32,
        "num_neurons": 10,
        "alpha": 0.01,
        "beta": 1e-6
    }
    write_dict_to_json(hyperparameters, args.params)

histories = {
    'seed': SEED
}

def cross_validation():
    input_shape = hyperparameters['input_shape']
    num_classes = hyperparameters['num_classes']
    epochs = 500
    batch_sizes = [4, 8, 16, 32, 64]
    num_neurons = hyperparameters['num_neurons']
    alpha = hyperparameters['alpha']

    histories['Q2'] = {
        'cv': {
            'accuracy': {batch: [] for batch in batch_sizes},
            'time': {batch: [] for batch in batch_sizes},
        },
        'optimal': dict()
    }

    logger.log('Starting cross validation...')
    X, y = dataset.get_train()
    X_test, y_test = dataset.get_test()

    for fold, (train_index, valid_index) in enumerate(dataset.get_kfold(), start=1):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        for batch in batch_sizes:
            logger.log('Fold %s Batch Size %s' % (fold, batch))
            with tqdm(total=epochs, desc='Fold %s Batch Size %s' % (fold, batch)) as pbar:
                update = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda batch, logs: pbar.update(1)
                )
            
                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=input_shape),
                    tf.keras.layers.Dense(num_neurons, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])

                model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=alpha),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy']
                )

                start = time.time()
                history = model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=batch,
                    epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    verbose=0,
                    callbacks=[update]
                )
                end = time.time()
            
                histories['Q2']['cv']['accuracy'][batch].append(history.history['val_accuracy'])
                histories['Q2']['cv']['time'][batch].append((end - start) / epochs)
    

    batch_epoch = pd.DataFrame(histories['Q2']['cv']['accuracy'])
    time_per_epoch = pd.DataFrame(histories['Q2']['cv']['time'])
    epoch_to_converge = batch_epoch.applymap(lambda x: acc_converge_epoch(x))
    cv_accuracy = batch_epoch.applymap(lambda x: training_result(x)).mean()
    total_time_to_converge = (epoch_to_converge * time_per_epoch).mean()

    def standardize(arr):
        return (arr - arr.mean()) / arr.std()

    def deciding_factor(total_time, acc):
        time_score = np.exp(-standardize(total_time))
        acc_score = np.exp(standardize(acc))
        print(acc_score * time_score)
        return int((acc_score * time_score).idxmax())

    hyperparameters['batch_size'] = deciding_factor(total_time_to_converge, cv_accuracy)
    logger.log('Optimal batch size: %s' % hyperparameters['batch_size'])
    write_dict_to_json(hyperparameters, args.params)

    logger.log('Done cross validation')


def optimal():
    input_shape = hyperparameters['input_shape']
    num_classes = hyperparameters['num_classes']
    epochs = 1000
    batch_size = hyperparameters['batch_size']
    num_neurons = hyperparameters['num_neurons']
    alpha = hyperparameters['alpha']

    logger.log('Create optimal model...')
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(num_neurons, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=alpha),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    X_train, y_train = dataset.get_train()
    X_test, y_test = dataset.get_test()

    logger.log('Training...')
    with tqdm(total=epochs) as pbar:
        update = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda batch, logs: pbar.update(1)
        )
        
        hist = model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[update]
        )
        
    histories['Q2']['optimal'] = {
        'optimal_batch': hyperparameters['batch_size'],
        'accuracy': hist.history['accuracy'],
        'val_accuracy': hist.history['val_accuracy'],
        'loss': hist.history['loss'],
        'val_loss': hist.history['val_loss']
    }

    logger.log('Done training')


cross_validation()
optimal()

# output to json
logger.log('Saving result to \"result/AQ2.json\"')
write_dict_to_json(filter_dict(histories, ['seed', 'Q2']), 'result/AQ2.json')

logger.end('Stopped q2_train.py')