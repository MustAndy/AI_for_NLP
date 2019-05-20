from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re
import jieba
import pandas as pd
import os
def cut(string): return ' '.join(jieba.cut(string))
DATA_PATH = 'd:/senior/aiCourse/dataSource/'


MODEL_PATH = os.path.join(DATA_PATH,'w2vmodel/wiki_news_cutting.model')
all_word2vec = Word2Vec.load(MODEL_PATH)
print(all_word2vec.similar_by_word('说'))

