import tensorflow as tf


class CharCNN(tf.keras.Model):
    def __init__(self, VOCAB_SIZE, MAX_LABEL=15,
                 N_FILTERS1=10, FILTER_SHAPE1=[20, 256], POOLING_WINDOW1=4, POOLING_STRIDE1=2, 
                 N_FILTERS2=10, FILTER_SHAPE2=[20, 1], POOLING_WINDOW2=4, POOLING_STRIDE2=2):
        super(CharCNN, self).__init__()
        self.one_hot_size = VOCAB_SIZE
        self.conv1 = tf.keras.layers.Conv2D(N_FILTERS1, FILTER_SHAPE1, padding='VALID', activation='relu', name='C1')
        self.pool1 = tf.keras.layers.MaxPool2D(POOLING_WINDOW1, POOLING_STRIDE1, padding='SAME', name='S1')
        self.conv2 = tf.keras.layers.Conv2D(N_FILTERS2, FILTER_SHAPE2, padding='VALID', activation='relu', name='C2')
        self.pool2 = tf.keras.layers.MaxPool2D(POOLING_WINDOW2, POOLING_STRIDE2, padding='SAME', name='S2')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = tf.one_hot(x, self.one_hot_size)
        x = x[..., tf.newaxis] 
        x = self.conv1(x)
        x = self.pool1(x)
        x = tf.nn.dropout(x, drop_rate)
        x = self.conv2(x)
        x = self.pool2(x)
        x = tf.nn.dropout(x, drop_rate)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits


class CharVanilla(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, HIDDEN_DIM=20, MAX_LABEL=15):
        super(CharVanilla, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.one_hot_size = VOCAB_SIZE
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.SimpleRNNCell(HIDDEN_DIM), unroll=True)
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = tf.one_hot(x, self.one_hot_size)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class CharLSTM(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, HIDDEN_DIM=20, MAX_LABEL=15):
        super(CharLSTM, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.one_hot_size = VOCAB_SIZE
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(HIDDEN_DIM), unroll=True)
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = tf.one_hot(x, self.one_hot_size)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class CharGRU1(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, HIDDEN_DIM=20, MAX_LABEL=15):
        super(CharGRU1, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.one_hot_size = VOCAB_SIZE
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(HIDDEN_DIM), unroll=True)
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = tf.one_hot(x, self.one_hot_size)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class CharGRU2(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, HIDDEN_DIM=20, MAX_LABEL=15, NUM_GRU_LAYERS=2):
        super(CharGRU2, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.one_hot_size = VOCAB_SIZE
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.GRUCell(HIDDEN_DIM) for _ in range(NUM_GRU_LAYERS)]
            ),
            unroll=True
        )
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = tf.one_hot(x, self.one_hot_size)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class WordCNN(tf.keras.Model):
    def __init__(self, VOCAB_SIZE, EMBEDDING_SIZE=20, MAX_LABEL=15,
                 N_FILTERS1=10, FILTER_SHAPE1=[20, 20], POOLING_WINDOW1=4, POOLING_STRIDE1=2, 
                 N_FILTERS2=10, FILTER_SHAPE2=[20, 1], POOLING_WINDOW2=4, POOLING_STRIDE2=2):
        super(WordCNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.conv1 = tf.keras.layers.Conv2D(N_FILTERS1, FILTER_SHAPE1, padding='VALID', activation='relu', name='C1')
        self.pool1 = tf.keras.layers.MaxPool2D(POOLING_WINDOW1, POOLING_STRIDE1, padding='SAME', name='S1')
        self.conv2 = tf.keras.layers.Conv2D(N_FILTERS2, FILTER_SHAPE2, padding='VALID', activation='relu', name='C2')
        self.pool2 = tf.keras.layers.MaxPool2D(POOLING_WINDOW2, POOLING_STRIDE2, padding='SAME', name='S2')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = self.embedding(x)
        x = x[..., tf.newaxis]
        x = self.conv1(x)
        x = self.pool1(x)
        x = tf.nn.dropout(x, drop_rate)
        x = self.conv2(x)
        x = self.pool2(x)
        x = tf.nn.dropout(x, drop_rate)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits


class WordVanilla(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, EMBEDDING_SIZE=20, HIDDEN_DIM=20, MAX_LABEL=15):
        super(WordVanilla, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.vocab_size = VOCAB_SIZE
        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.SimpleRNNCell(HIDDEN_DIM), unroll=True)
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = self.embedding(x)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class WordLSTM(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, EMBEDDING_SIZE=20, HIDDEN_DIM=20, MAX_LABEL=15):
        super(WordLSTM, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.vocab_size = VOCAB_SIZE
        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(HIDDEN_DIM), unroll=True)
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = self.embedding(x)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class WordGRU1(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, EMBEDDING_SIZE=20, HIDDEN_DIM=20, MAX_LABEL=15):
        super(WordGRU1, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.vocab_size = VOCAB_SIZE
        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(HIDDEN_DIM), unroll=True)
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = self.embedding(x)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits


class WordGRU2(tf.keras.Model):

    def __init__(self, VOCAB_SIZE, EMBEDDING_SIZE=20, HIDDEN_DIM=20, MAX_LABEL=15, NUM_GRU_LAYERS=2):
        super(WordGRU2, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.vocab_size = VOCAB_SIZE
        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.rnn = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(
                [tf.keras.layers.GRUCell(HIDDEN_DIM) for _ in range(NUM_GRU_LAYERS)]
            ),
            unroll=True
        )
        self.dense = tf.keras.layers.Dense(MAX_LABEL, activation='softmax')

    def call(self, x, drop_rate):
        x = self.embedding(x)
        x = self.rnn(x)
        x = tf.nn.dropout(x, drop_rate)
        logits = self.dense(x)
        return logits