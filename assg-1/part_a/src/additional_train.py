import argparse

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.logger import Logger
from utils.seed import SEED, init_seed
from utils.preprocess_dataset import PreprocessDataset
from utils.dict_json import filter_dict, write_dict_to_json, read_json_to_dict

init_seed()

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--data', help='Path to dataset csv file', required=True)
parser.add_argument('-P', '--params', help='Path to hyperparameters json file', required=True)
args = parser.parse_args()

logger = Logger()
logger.log('Starting additional_train.py...')

# import dataset
logger.log('Loading dataset from \"' + args.data + '\"...')
df = pd.read_csv(args.data)
dataset = PreprocessDataset(
    df=df, 
    feature_columns=df.columns[:21],
    label_column=df.columns[-1],
    test_ratio=0.3,
    fold=5
)

hyperparameters = read_json_to_dict(args.params)

# output dict
histories = {
    'seed': SEED
}

# define parameters
input_shape = hyperparameters['input_shape']
num_classes = hyperparameters['num_classes']

epochs = 1000
batch_size = hyperparameters['batch_size']
num_neurons = hyperparameters['num_neurons']
alpha = hyperparameters['alpha']
beta = hyperparameters['beta']

# create and compile model
logger.log('Creating model...')
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
    tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=beta)),
    tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l=beta))
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=alpha),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# train
logger.log('Training...')
X_train, y_train = dataset.get_train()
X_test, y_test = dataset.get_test()

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
    
histories['Q5'] = {
    'accuracy': hist.history['accuracy'],
    'val_accuracy': hist.history['val_accuracy'],
    'loss': hist.history['loss'],
    'val_loss': hist.history['val_loss']
}

# output to json
logger.log('Saving result to \"result/additional.json\"')
write_dict_to_json(filter_dict(histories, ['seed', 'Q5']), 'result/additional.json')

logger.end('Stopped additional_train.py')