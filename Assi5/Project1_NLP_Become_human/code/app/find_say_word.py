from collections import defaultdict
from functools import lru_cache
import datetime
import pickle
import os


@lru_cache(maxsize=10 * 100 * 100)
def find_most_similar(node, topn_in, model):
    new_expanding_temp = [w for w, s in model.most_similar(node, topn=topn_in)]
    return new_expanding_temp


def get_related_words(initial_words, model):
    """
    @initial_words are initial words we already know
    @model is the word2vec model
    """
    starttime = datetime.datetime.now()

    unseen = initial_words

    seen = defaultdict(int)

    max_size = 1000  # could be greater

    while unseen and len(seen) < max_size:
        # if len(seen) % 50 == 0:
        #    print('seen length : {}'.format(len(seen)))

        node = unseen.pop(0)

        if seen[node]:
            if seen[node] > 10:
                seen[node] += seen[node] / 50
            else:
                seen[node] += 1
            new_expanding = find_most_similar(node, 10, model)
            unseen += new_expanding
            continue

        new_expanding = find_most_similar(node, 10, model)
        unseen += new_expanding
        seen[node] += 1

        # optimal: 1. score function could be revised
        # optimal: 2. using dymanic programming to reduce computing time
    endtime = datetime.datetime.now()
    print('Using : {} s for finding \'say\''.format((endtime - starttime).seconds))

    return seen


def load_say_word(DATA_PATH,all_word2vec):
    say_word_file= os.path.join(DATA_PATH,'say.pickle')
    if os.path.exists(say_word_file):
        print('Pickle file found ! Now loading!')
        result = pickle.load(open(say_word_file,'rb'))
    else:
        print('Pickle file not found ! Starting searching, please wait for about 2 minutes.')
        related_words = get_related_words(['说', '表示'], all_word2vec)
        pickle.dump(related_words,open(say_word_file,'wb'),pickle.HIGHEST_PROTOCOL)
        result = pickle.load(open(say_word_file,'rb'))

    return result


