import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback
from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
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
from dataset import *

data_files = [
    'D:/senior/aiCourse/dataSource/comment_classification/output/train.json']
vocab_file = 'D:/senior/aiCourse/dataSource/comment_classification/output/vocab.txt'
label_file = 'D:/senior/aiCourse/dataSource/comment_classification/labels.txt'
enb_file = 'D:/senior/aiCourse/dataSource/comment_classification/embedding/embedding.txt'
batch_size = 128
reverse = False
split_word = True
max_len = 400
data_files = ['../../../dataSource/comment_classification/output/train.json']
vocab_file = '../../../dataSource/comment_classification/output/vocab.txt'
label_file = '../../../dataSource/comment_classification/labels.txt'
enb_file = '../../../dataSource/comment_classification/output/embedding.txt'
dataset1 = DataSet(data_files, vocab_file, label_file, batch_size,
                   reverse=reverse, split_word=split_word, max_len=max_len)
for i, (source, lengths, targets, _) in enumerate(dataset1.get_next()):
    print(len(source[9]))
    break


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
            features[i, :] = np.array(comment)[len(comment)-maxlen:]
    return features


def split_dataset(pad_comments, labels, split_frac):
    split_index = int(len(pad_comments)*split_frac)
    data_list = list(zip(pad_comments, labels))
    random.shuffle(data_list)
    pad_comments, labels = zip(*data_list)
    x_train, x_test = pad_comments[:split_index], pad_comments[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]
    return x_train, y_train, x_test, y_test


comment_to_id = [x[0] for x in dataset1._raw_data]
pad_comments = pad_sequences(
    comment_to_id, maxlen=max_len, padding='post', truncating='post')

labels = [x[2].flatten().tolist() for x in dataset1._raw_data]
x_train, y_train, x_test, y_test = split_dataset(
    pad_comments[:20000], labels, 0.8)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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


densor1 = Dense(32, activation="tanh")
densor2 = Dense(2, activation="relu")
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
    x = Embedding(50000, 256)(inp)
    x1 = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
    x2 = Bidirectional(CuDNNGRU(100, return_sequences=True))(x1)
    x3 = Bidirectional(CuDNNGRU(100, return_sequences=True))(x2)
    conc = concatenate([x1, x2, x3])
    x = Dropout(0.5)(conc)
    context = one_step_attention(x)
    context = Flatten()(context)
    merged = Dropout(0.5)(context)
    preds = Dense(80, activation='relu')(merged)
    model = Model(inputs=[inp], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    print(model.summary())
    return model


def get_model1():

    inp = Input(shape=(400,))
    x = Embedding(50000, 256)(inp)
    x1 = Bidirectional(CuDNNGRU(100, return_sequences=False))(x)

    preds = Dense(80, activation='relu')(x1)
    model = Model(inputs=[inp], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    print(model.summary())
    return model


model = get_model1()

print(model.summary())
history = model.fit(x_train, y_train, epochs=2,
                    validation_data=(x_test, y_test), verbose=1)
