import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Embedding, Input, \
    Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
import keras.layers as layers
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Layer
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import random
from app.coreFunction import dataset
from app.coreFunction import utils





def dig_lists(l):
    output = []
    for e in l:
        if isinstance(e, list):
            output += dig_lists(e)
        else:
            output.append(e)
    return (output)


def pad_sequences(comment_to_id, maxlen, padding, truncating):
    features = np.zeros((len(comment_to_id), maxlen), dtype=int)
    for i, comment in enumerate(comment_to_id):
        if len(comment) <= maxlen and padding == 'pre':
            features[i, -len(comment):] = np.array(comment)[:maxlen]
        if len(comment) <= maxlen and padding == 'post':
            features[i, :len(comment)] = np.array(comment)[:maxlen]
        if len(comment) > maxlen and truncating == 'post':
            features[i, :] = np.array(comment)[:maxlen]
        if len(comment) > maxlen and truncating == 'pre':
            features[i, :] = np.array(comment)[len(comment) - maxlen:]
    return features


def split_dataset(pad_comments, labels, split_frac):
    split_index = int(len(pad_comments) * split_frac)
    data_list = list(zip(pad_comments, labels))
    random.shuffle(data_list)
    pad_comments, labels = zip(*data_list)
    x_train, x_test = pad_comments[:split_index], pad_comments[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]
    return x_train, y_train, x_test, y_test


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def softmax(x, axis=1):
    """Softmax activation function."""
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


densor1 = Dense(50, activation="tanh")
densor2 = Dense(3, activation="relu")
# We are using a custom softmax(axis = 1) loaded in this notebook
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)


def one_step_attention(a):
    e = densor1(a)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context


def get_model():
    inp = Input(shape=(400,))
    x = Embedding(50000, 400)(inp)
    x1 = Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNLSTM(120, return_sequences=True))(x1)
    x3 = Bidirectional(CuDNNLSTM(140, return_sequences=True))(x2)
    # x3 = BatchNormalization()(x3)
    pooling = concatenate([x3, x2, x1, x])
    # pooling = GlobalAveragePooling1D()(pooling)
    # pooling = BatchNormalization()(pooling)
    # x4 = Dense(1200, activation='relu')(pooling)

    # x = Embedding(50000, 400)(pooling)
    # attnInput = Embedding(10000, 400)(pooling)
    pooling = Dropout(0.05)(pooling)
    context = one_step_attention(pooling)

    # x4 = CuDNNLSTM(60, return_sequences=True)(context)
    context = Flatten()(context)

    context = Dropout(0.3)(context)
    preds = Dense(80, activation='selu', kernel_initializer='lecun_normal')(context)
    model = Model(inputs=[inp], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model



"""
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.title('model f1')
plt.ylabel('f1')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""

