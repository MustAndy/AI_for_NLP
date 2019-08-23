from app.coreFunction import model
from app.coreFunction import utils
from app.coreFunction import dataset
import numpy as np
import datetime
data_files = [
    'D:/senior/aiCourse/dataSource/comment_classification/output/validation.json']
vocab_file = 'D:/senior/aiCourse/dataSource/comment_classification/output/vocab.txt'
label_file = 'D:/senior/aiCourse/dataSource/comment_classification/labels.txt'
enb_file = 'D:/senior/aiCourse/dataSource/comment_classification/embedding/embedding.txt'

data_files = ['../../../../dataSource/comment_classification/output/validation.json',
              '../../../../dataSource/comment_classification/output/train.json']
vocab_file = '../../../../dataSource/comment_classification/output/vocab.txt'
label_file = '../../../../dataSource/comment_classification/labels.txt'
enb_file = '../../../../dataSource/comment_classification/output/embedding.txt'

max_len = 400


def loadData():
    batch_size = 100
    reverse = False
    split_word = True

    datasetTemp = dataset.Dataset(data_files, vocab_file, label_file, batch_size,
                                  reverse=reverse, split_word=split_word, max_len=max_len)
    return datasetTemp


def loadInputData(datasetTemp):
    row = datasetTemp.get_shuffle_row()
    comment_to_id = [x[0] for x in row]
    labels = [x[2].flatten().tolist() for x in row]
    pad_comments = model.pad_sequences(comment_to_id, maxlen=max_len, padding='post', truncating='post')

    x_train, y_train, x_test, y_test = model.split_dataset(
        pad_comments[:110000], labels[:110000], 0.8)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def preLoad():
    datasetTemp = loadData()
    x_train, y_train, x_test, y_test = loadInputData(datasetTemp)
    return datasetTemp,x_train, y_train, x_test, y_test


def train():
    starttime = datetime.datetime.now()
    datasetTemp, x_train, y_train, x_test, y_test = preLoad()
    modelTrain = model.get_model()
    history = []
    for i in range(0,4):
        history.append(modelTrain.fit(x_train[i*20000:(i+1)*20000], y_train[i*20000:(i+1)*20000],
                                      epochs=2,batch_size=100,validation_data=(x_test, y_test), verbose=1))
        preds = modelTrain.predict(x_test)
        result = utils.Evaluation(preds,y_test)
        marking, f1Score = result.printing()
        if marking:
            print('Model saved')
            modelTrain.save('./saveModel/model_'+str(f1Score)+'.h5')

    history.append(modelTrain.fit(x_train[80000:], y_train[80000:],
                                  epochs=2, batch_size=200,validation_data=(x_test, y_test), verbose=1))
    preds = modelTrain.predict(x_test)
    result = utils.Evaluation(preds,y_test)
    marking, f1Score = result.printing()
    if marking:
        modelTrain.save('./saveModel/model_'+str(f1Score)+'.h5')
    endtime = datetime.datetime.now()
    print ('Using {} minutes'.format(((endtime - starttime).seconds)/60))