from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Layer
from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D, Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import roc_auc_score
import keras.layers as layers
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras import initializers as initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
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
    pad_comments[:10000], labels, 0.8)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


def get_model():
    embed_size = 256
    inp = Input(shape=(400, ))
    x = Embedding(input_dim=len(dataset1.w2i), output_dim=embed_size,)(inp)
    print(x)
    x = SpatialDropout1D(0.2)(x)
    x1 = Bidirectional(GRU(100, return_sequences=True))(x)
    x2 = Bidirectional(GRU(100, return_sequences=True))(x1)
    x3 = Bidirectional(GRU(100, return_sequences=True))(x2)

    avg_pool = GlobalAveragePooling1D()(x3)
    max_pool = GlobalMaxPooling1D()(x3)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(80, activation="relu")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()

print(model.summary())
history = model.fit(x_train, y_train, epochs=5，validation_data=(x_test, y_test), verbose=1, batch_size=batch_size)
