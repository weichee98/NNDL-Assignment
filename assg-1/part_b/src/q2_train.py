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
dataset = PreprocessDataset(
    df=df, 
    feature_columns=df.columns[1:8],
    label_column=df.columns[-1],
    test_ratio=0.3,
    fold=5
)
X_train, y_train = dataset.get_train()
X_test, y_test = dataset.get_test()


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
write_dict_to_json(filter_dict(histories, ['seed', 'columns', 'Q2']), 'result/BQ2.json')


# RFE functions
def remove_features(f_arr, remove):
    n_features = set(range(f_arr.shape[1]))
    remaining_features = list(n_features - set(remove))
    return f_arr[:, remaining_features]

def rfe(n_to_remove, removed=[]):
    if n_to_remove == 0:
        return
    
    logger.log('RFE %s' % (n_to_remove))
    results = dict()
    
    for i in range(X_train.shape[1]):
        if i in removed:
            continue

        logger.log('Removed %s' % (removed + [i]))

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_train.shape[1] - len(removed) - 1,)),
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
                x=remove_features(X_train, removed + [i]),
                y=y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(remove_features(X_test, removed + [i]), y_test),
                verbose=0,
                callbacks=[update]
            )

        key = ",".join(map(str, removed + [i]))
        histories['Q2'][key] = {
            'mse': hist.history['mse'],
            'val_mse': hist.history['val_mse'],
            'loss': hist.history['loss'],
            'val_loss': hist.history['val_loss']
        }
        results[i] = training_result(hist.history['val_mse'], mode='loss')
    
    i = min(results.keys(), key=lambda x: results[x])
    logger.log('Feature %s is removed!' % (i))
    rfe(n_to_remove - 1, removed + [i])

rfe(
    n_to_remove=2,
    removed=[]
)

# output to json
logger.log('Saving result to \"result/BQ2.json\"')
write_dict_to_json(filter_dict(histories, ['seed', 'columns', 'Q2']), 'result/BQ2.json')

logger.end('Stopped q2_train.py')