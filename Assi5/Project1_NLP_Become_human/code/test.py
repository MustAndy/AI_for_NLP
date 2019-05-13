from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re
import jieba
import pandas as pd

def cut(string): return ' '.join(jieba.cut(string))

all_word2vec = Word2Vec.load('./wiki_news_cutting.pickle')
print(all_word2vec.similar_by_word('说'))

