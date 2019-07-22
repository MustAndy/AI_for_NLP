import pandas as pd
from collections import Counter
import numpy as np
import os
from opencc import OpenCC
import jieba
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.merge import add
from keras.layers import LSTM,Bidirectional, Dropout,GRU
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import regularizers

dataSource = 'D:/senior/aiCourse/dataSource/comment_classification/'
stopWord_path = 'D:/senior/aiCourse/dataSource/stop_word.txt'
trainCSV_path = os.path.join(dataSource,'train/sentiment_analysis_trainingset.csv')
testCSV_path = os.path.join(dataSource,'test/sentiment_analysis_testa.csv')

