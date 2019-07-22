import codecs
import json
from collections import namedtuple

import numpy as np
import tensorflow as tf
import sys
import os

UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


def read_vocab(vocab_file):
    """read vocab from file

    Args:
        vocab_file ([type]): path to the vocab file, the vocab file should contains a word each line

    Returns:
        list of words
    """

    if not os.path.isfile(vocab_file):
        raise ValueError("%s is not a vaild file" % vocab_file)

    vocab = []
    word2id = {}
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        for i, line in enumerate(f):
            word = line.strip()
            if not word:
                raise ValueError("Got empty word at line %d" % (i + 1))
            vocab.append(word)
            word2id[word] = len(word2id)

    print("# vocab size: ", len(vocab))
    return vocab, word2id


def load_embed_file(embed_file):
    """Load embed_file into a python dictionary.
    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:
    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547
    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for i, line in enumerate(f):
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(
                    vec), "All embedding size should be same, but got {0} at line {1}".format(len(vec), i + 1)
            else:
                emb_size = len(vec)
    return emb_dict, emb_size


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")

    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _padding(tokens_list, max_len):
    ret = np.zeros((len(tokens_list), max_len), np.int32)
    for i, t in enumerate(tokens_list):
        t = t + (max_len - len(t)) * [EOS_ID]
        ret[i] = t
    return ret


def _tokenize(content, w2i, max_tokens=1200, reverse=False, split=True):
    def get_tokens(content):
        tokens = content.strip().split()
        ids = []
        for t in tokens:
            if t in w2i:
                ids.append(w2i[t])
            else:
                for c in t:
                    ids.append(w2i.get(c, UNK_ID))
        return ids

    if split:
        ids = get_tokens(content)
    else:
        ids = [w2i.get(t, UNK_ID) for t in content.strip().split()]
    if reverse:
        ids = list(reversed(ids))
    tokens = [SOS_ID] + ids[:max_tokens] + [EOS_ID]
    return tokens


class DataItem(namedtuple("DataItem", ('content', 'length', 'labels', 'id'))):
    pass


class DataSet(object):
    def __init__(self, data_files, vocab_file, label_file, batch_size=32, reverse=False, split_word=True, max_len=1200):
        self.reverse = reverse
        self.split_word = split_word
        self.data_files = data_files
        self.batch_size = batch_size
        self.max_len = max_len

        self.vocab, self.w2i = read_vocab(vocab_file)
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.label_names, self.l2i = read_vocab(label_file)
        self.i2l = {v: k for k, v in self.l2i.items()}

        self.tag_l2i = {"1": 0, "0": 1, "-1": 2, "-2": 3}
        self.tag_i2l = {v: k for k, v in self.tag_l2i.items()}

        self._raw_data = []
        self.items = []
        self._preprocess()


    def get_label(self, labels, l2i, normalize=False):
        one_hot_labels = np.zeros(len(l2i), dtype=np.float32)
        for n in labels:
            if n:
                one_hot_labels[l2i[n]] = 1

        if normalize:
            one_hot_labels = one_hot_labels / len(labels)
        return one_hot_labels


    def _preprocess(self):
        print_out("# Start to preprocessing data...")
        for fname in self.data_files:
            print_out("# load data from %s ..." % fname)
            for line in open(fname,encoding = 'utf-8'):
                
                item = json.loads(line.strip(),encoding = 'utf-8')
                content = item['content']
                content = _tokenize(content, self.w2i, self.max_len, self.reverse, self.split_word)
                item_labels = []
                for label_name in self.label_names:
                    labels = [item[label_name]]
                    labels = self.get_label(labels, self.tag_l2i)
                    item_labels.append(labels)
                self._raw_data.append(
                    DataItem(content=content, labels=np.asarray(item_labels), length=len(content), id=int(item['id'])))
                self.items.append(item)

        self.num_batches = len(self._raw_data) // self.batch_size
        self.data_size = len(self._raw_data)
        print_out("# Got %d data items with %d batches" % (self.data_size, self.num_batches))

    def _shuffle(self):
        # code from https://github.com/fastai/fastai/blob/3f2079f7bc07ef84a750f6417f68b7b9fdc9525a/fastai/text.py#L125
        idxs = np.random.permutation(self.data_size)
        sz = self.batch_size * 50
        ck_idx = [idxs[i:i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=lambda x: self._raw_data[x].length, reverse=True) for s in ck_idx])
        sz = self.batch_size
        ck_idx = [sort_idx[i:i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self._raw_data[ck[0]].length for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


    def process_batch(self, batch):
        contents = [item.content for item in batch]
        lengths = [item.length for item in batch]
        contents = _padding(contents, max(lengths))
        lengths = np.asarray(lengths)
        targets = np.asarray([item.labels for item in batch])
        ids = [item.id for item in batch]
        return contents, lengths, targets, ids

    def get_next(self, shuffle=True):
        if shuffle:
            idxs = self._shuffle()
        else:
            idxs = range(self.data_size)

        batch = []
        for i in idxs:
            item = self._raw_data[i]
            if len(batch) >= self.batch_size:
                yield self.process_batch(batch)
                batch = [item]
            else:
                batch.append(item)
        if len(batch) > 0:
            yield self.process_batch(batch)