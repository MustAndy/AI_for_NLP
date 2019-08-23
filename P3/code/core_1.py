import os
import pandas as pd
from collections import Counter
import numpy as np
from opencc import OpenCC
import jieba
import random
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import datetime
from functools import reduce
def cut(string): return ' '.join(jieba.cut(string))
import re
import networkx
import matplotlib.font_manager as fm
fp1 = fm.FontProperties(fname="D://senior/aiCourse/dataSource/SourceHanSerifSC-Regular.otf")
path_root = 'D:/senior/aiCourse/dataSource/'


def load_data():
    pure_file = os.path.join(path_root, 'P3/pureContent.csv')
    if not os.path.exists(pure_file):
        news_file = os.path.join(path_root, 'sqlResult_1558435.csv')
        news_content = pd.read_csv(news_file, encoding='gb18030')
        news_content['content'] = news_content['content'].fillna('')
        pure_content = pd.DataFrame()
        pure_content['content'] = news_content['content']
        pure_content = pure_content.fillna('')
        pure_content['tokenized_content'] = pure_content['content'].apply(cut)
        pure_content.to_csv(pure_file, encoding='gb18030')
    else:
        print('File found! ')
        pure_content = pd.read_csv(pure_file, encoding='gb18030')

    return pure_content


def get_summarization_simple_with_text_rank(text, constraint=200):
    return get_summarization_simple(text, sentence_ranking_by_text_ranking, constraint)


# 建立句子和标点符号之间的关系，例如，建立一个字典
def get_summarization_simple(text, score_fn, constraint=200):
    sub_sentence = split_sentence(text)
    ranking_sentence = score_fn(sub_sentence)
    selected_text = set()
    current_text = ''

    for sen, _ in ranking_sentence:
        if len(current_text) < constraint:
            current_text += sen
            selected_text.add(sen)
        else:
            break

    summarized = []
    for sen in sub_sentence:  # print the selected sentence by sequent
        if sen in selected_text:
            summarized.append(sen)
    return summarized


def get_connect_graph_by_text_rank(tokenized_text: str, window=3):
    keywords_graph = networkx.Graph()
    tokeners = tokenized_text.split()
    for ii, t in enumerate(tokeners):
        word_tuples = [(tokeners[connect], t)
                       for connect in range(ii-window, ii+window+1)
                       if connect >= 0 and connect < len(tokeners)]
        keywords_graph.add_edges_from(word_tuples)

    return keywords_graph


def sentence_ranking_by_text_ranking(split_sentence):
    sentence_graph = get_connect_graph_by_text_rank(' '.join(split_sentence))
    ranking_sentence = networkx.pagerank(sentence_graph)
    ranking_sentence = sorted(ranking_sentence.items(), key=lambda x: x[1], reverse=True)
    return ranking_sentence


def load_model(pure_content):
    model_file = os.path.join(path_root, 'P3/wordmodel_50.model')
    savedContent = os.path.join(path_root, 'P3/all_corpus.txt')

    if not os.path.exists(model_file):
        if not os.path.exists(savedContent):
            with open(savedContent , 'w', encoding='utf-8') as f:
                f.write(' '.join(pure_content['tokenized_content'].tolist()))
        model = FastText(LineSentence(savedContent), workers=8,window=5, size=50, iter=10, min_count=1)
        model.save(model_file)
    else:
        print('model Found!')
        model = FastText.load(model_file)
    return model


#分割句子，将句子按照逗号和句号分隔开。
def split_sentence(sentence):
    pattern = re.compile('[。，,.]：')
    split = pattern.sub(' ', sentence).split()  # split sentence
    return split




if __name__=='__main__':
    starttime = datetime.datetime.now()
    pure_content = load_data()
    model = load_model(pure_content)
    lengthes_of_text = map(len, pure_content['content'].tolist())

    sharp_news = pure_content.iloc[7]['content']
    print(sharp_news+'\n')
    print(' '.join(get_summarization_simple_with_text_rank(sharp_news, constraint=250)))

    endtime = datetime.datetime.now()
    print ('Using {} minutes'.format(((endtime - starttime).seconds)/60))
