import numpy as np
import tensorflow as tf

SEED = 10

def init_seed():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)