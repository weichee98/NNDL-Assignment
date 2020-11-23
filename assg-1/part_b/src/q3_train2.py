import argparse
import time
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.logger import Logger
from utils.seed import SEED, init_seed
from utils.preprocess_dataset import PreprocessDataset
from utils.dict_json import filter_dict, write_dict_to_json, read_json_to_dict
from utils.acc_loss import training_result

init_seed()

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to dataset csv file', required=True)
parser.add_argument('-P', '--params', help='Path to hyperparameters json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting q3_train.py...')


# Loading dataset
logger.log('Loading dataset from \"' + args.data + '\"...')
df = pd.read_csv(args.data)


# Defining hyperparameters
try:
    hyperparameters = read_json_to_dict(args.params)
except:
    hyperparameters = {
        "input_shape": (7,),
        "features_left": df.columns[1:8].tolist(),
        "batch_size": 8,
        "num_neurons": 10,
        "alpha": 1e-3,
        "beta": 1e-3
    }
    write_dict_to_json(hyperparameters, args.params)

epochs = 10000
num_neurons = 50
dropout = 0.2
batch_size = hyperparameters['batch_size']
alpha = hyperparameters['alpha']
beta = hyperparameters['beta']


# Histories of results
histories = {
    'seed': SEED,
    'Q3': dict()
}

features_left = hyperparameters['features_left']
input_shape = hyperparameters['input_shape']

dataset = PreprocessDataset(
    df=df, 
    feature_columns=features_left,
    label_column=df.columns[-1],
    test_ratio=0.3,
    fold=5
)
X_train, y_train = dataset.get_train()
X_test, y_test = dataset.get_test()


# Creating models
models = {

    "3-level": tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=beta))
    ]),

    "4-level": tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=beta))
    ]),

    "4-level+Dropout": tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=beta))
    ]),

    "5-level": tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=beta))
    ]),

    "5-level+Dropout": tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=beta))
    ]),

}


def compare():

    for k, model in models.items():
        logger.log('Training ' + k)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=alpha),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse']
        )

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

            histories['Q3'][k] = {
                'mse': hist.history['mse'],
                'val_mse': hist.history['val_mse'],
                'loss': hist.history['loss'],
                'val_loss': hist.history['val_loss']
            }


compare()

# output to json
logger.log('Saving result to \"result/BQ3_2.json\"')
write_dict_to_json(filter_dict(histories, ['seed', 'Q3']), 'result/BQ3_2.json')

logger.end('Stopped q3_train.py')