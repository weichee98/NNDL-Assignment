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
logger.log('Starting q2_train.py...')


# Loading dataset
logger.log('Loading dataset from \"' + args.data + '\"...')
df = pd.read_csv(args.data)


# Defining hyperparameters
try:
    hyperparameters = read_json_to_dict(args.params)
except:
    hyperparameters = {
        "input_shape": (7,),
        "batch_size": 8,
        "num_neurons": 10,
        "alpha": 1e-3,
        "beta": 1e-3
    }
    write_dict_to_json(hyperparameters, args.params)

epochs = 5000
batch_size = hyperparameters['batch_size']
num_neurons = hyperparameters['num_neurons']
alpha = hyperparameters['alpha']
beta = hyperparameters['beta']


# Histories of results
histories = {
    'seed': SEED,
    'columns': df.columns[1:8].tolist(),
    'Q2': dict()
}


def rfe(n_features, features_left=[]):
    if len(features_left) <= n_features:
        return

    logger.log('RFE %s' % (len(features_left) - 1))
    results = dict()
    
    for i, col in enumerate(features_left):

        columns = list(features_left[0:i]) + list(features_left[i + 1:len(features_left)])
        dataset = PreprocessDataset(
            df=df, 
            feature_columns=columns,
            label_column=df.columns[-1],
            test_ratio=0.3,
            fold=5
        )
        X_train, y_train = dataset.get_train()
        X_test, y_test = dataset.get_test()

        logger.log('Features %s' % (", ".join(columns)))

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(len(columns),)),
            tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=beta))
        ])

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

        key = ", ".join(columns)
        histories['Q2'][key] = {
            'mse': hist.history['mse'],
            'val_mse': hist.history['val_mse'],
            'loss': hist.history['loss'],
            'val_loss': hist.history['val_loss']
        }
        results[key] = training_result(hist.history['val_mse'], mode='loss')
    
    cols = min(results.keys(), key=lambda x: results[x])
    logger.log('Features left %s' % (cols))
    rfe(n_features, cols.split(", "))

rfe(
    n_features=5,
    features_left=list(df.columns[1:8])
)

# output to json
logger.log('Saving result to \"result/BQ2.json\"')
write_dict_to_json(filter_dict(histories, ['seed', 'columns', 'Q2']), 'result/BQ2.json')

logger.end('Stopped q2_train.py')