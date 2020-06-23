#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from Bio import SeqIO
from itertools import product
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder)
from gensim.models import (Word2Vec, FastText)


def get_sequences_from_fasta(path):
    seqs = list()
    for seq_record in SeqIO.parse(path, "fasta"):
        s = str(seq_record.seq.upper())
        if 'N' not in s:
            seqs.append(s)
    return seqs


def set_org_paths(org, fasta_path):
    neg_path = os.path.join(fasta_path, '{}_neg.fa'.format(org))
    pos_path = os.path.join(fasta_path, '{}_pos.fa'.format(org))

    return neg_path, pos_path


def slice_sequences(sequences, tss_position, upstream, downstream):
    sequences = [x[(tss_position - upstream):(tss_position + downstream)] for x in sequences]
    return sequences


def get_kmers(seq, k=1, step=1):
    numChunks = ((len(seq) - k) / step) + 1
    mers = list()
    i = 0
    for i in range(0, int(numChunks * step), int(step)):
        s = seq[i:i + k]
        mers.append(s)
    # print('mers', len(mers), mers)
    return mers


def get_all_possible_k_mers(k):
    nucs = ['A', 'C', 'G', 'T']
    tups = list(product(nucs, repeat=k))
    mers = [''.join(x) for x in tups]
    # print(mers)
    return mers


def tokenize_sequences(seqs, k=1, step=1):
    print(' -' * 30)
    print('Tokenize with k = {}'.format(k))
    print('Tokenize sequences.')
    tokens = [get_kmers(s, k=k, step=step) for s in seqs]
    print('Number of tokens per sequence:\t{}'.format(len(tokens[0])))
    print('Sample:\t{}'.format(tokens[0]))
    return tokens


def get_kmer_to_int_encoder(k):
    mers = get_all_possible_k_mers(k)
    label_encoder = LabelEncoder()
    label_encoder.fit(mers)
    return label_encoder


def encode_kmers_to_int_sequences(tokens, k=1):
    encoder_opt = 'categorical'
    encoder = get_kmer_to_int_encoder(k)
    new_seqs = np.vstack([encoder.transform(s) for s in tokens])
    new_seqs = np.expand_dims(new_seqs, axis=2)
    return new_seqs, encoder_opt


def get_kmer_to_onehot_enconder(k):
    mers = np.array(get_all_possible_k_mers(k)).reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(mers)
    return onehot_encoder


def encode_kmers_to_onehot_sequences(tokens, k=1):
    encoder_opt = 'onehot'
    encoder = get_kmer_to_onehot_enconder(k)
    new_seqs = np.stack([encoder.transform(np.array(list(s)).reshape(-1, 1)) for s in tokens])
    new_seqs = np.expand_dims(new_seqs, axis=3)
    return new_seqs, encoder_opt

def encode_word2vec(tokens, k=1):
    encoder_opt = 'embbed'
    if not type(tokens) == list:
        tokens = tokens.tolist()
    model = Word2Vec(min_count=1, workers=4, size=10, window=3)
    model.build_vocab(sentences=tokens)
    model.train(sentences=tokens, total_examples=len(tokens), epochs=10)
    new_seqs = np.array([model.wv.__getitem__(t) for t in tokens])
    # print(seqs)
    # print(seqs.shape)
    return new_seqs, encoder_opt


def encode_fastdna(tokens, k=1):
    from utils import tsne_plot
    encoder_opt = 'embbed'
    if not type(tokens) == list:
        tokens = tokens.tolist()
    model = FastText(min_count=1, workers=4, size=3, window=7)
    model.build_vocab(sentences=tokens)
    model.train(sentences=tokens, total_examples=len(tokens), epochs=100)

    # tsne_plot(model)
    # raise Exception

    new_seqs = np.array([model.wv.__getitem__(t) for t in tokens])
    # print(seqs)
    # print(seqs.shape)
    return new_seqs, encoder_opt


def get_y(neg_data, pos_data):
    neg_labels = [0 for _, _ in enumerate(neg_data)]
    pos_labels = [1 for _, _ in enumerate(pos_data)]
    y = np.array(neg_labels + pos_labels)
    return y


def encode_tokens(tokens, encoder_type='label'):
    encoders = ('label', 'onehot', 'dna2vec', 'fastdna')
    encoder_functions = [
        encode_kmers_to_int_sequences,
        encode_kmers_to_onehot_sequences,
        encode_word2vec,
        encode_fastdna,
    ]
    k = len(tokens[0][0])
    new_seq, encoder_opt = encoder_functions[encoders.index(encoder_type)](tokens, k=k)
    print(type(new_seq))
    add_dims = lambda x: np.expand_dims(x, len(x.shape))
    while len(new_seq.shape) != 4:
        new_seq = add_dims(new_seq)
    print('new_seq.shape', new_seq.shape)
    return new_seq.astype(float), encoder_opt


class PromoterData(object):
    def __init__(self, fasta_path):
        self.fasta_path = os.path.join(*(os.path.split(fasta_path)))  # Normalize path for OS
        self.org_name = None
        self.neg_path = None
        self.pos_path = None
        self.negative_sequences = None
        self.positive_sequences = None
        self.tss_position = None
        self.downstream = None
        self.upstream = None
        self.y = None
        self.X = None
        self.data: DataChunk = None
        self.tokens = None

    def set_organism_sequences(self, org_name):
        print(' -' * 30)
        self.org_name = org_name

        print('Setting paths for organism: {}'.format(org_name))
        self.neg_path, self.pos_path = set_org_paths(self.org_name, self.fasta_path)
        print('Negative:\t {}'.format(self.neg_path))
        print('Positive:\t {}'.format(self.pos_path))

        print('\nGetting negative sequences for organism "{}" in "{}"'.format(self.org_name, self.neg_path))
        self.negative_sequences = get_sequences_from_fasta(self.neg_path)
        print('Number of negative sequences: \t{}'.format(len(self.negative_sequences)))
        print('Sample:\t{}'.format(self.negative_sequences[0]))

        print('\nGetting positive sequences for organism "{}" in "{}"'.format(self.org_name, self.pos_path))
        self.positive_sequences = get_sequences_from_fasta(self.pos_path)
        print('Number of positive sequences: \t{}'.format(len(self.positive_sequences)))
        print('Sample:\t{}'.format(self.positive_sequences[0]))

        seqs_lenght = len(self.negative_sequences[0])
        print('\nLength of sequences: \t{}'.format(seqs_lenght))

    def set_tokens(self, k=None, slice_seq=None):
        if slice_seq:
            (_tss, _up, _down) = slice_seq
        pos = self.negative_sequences if not slice_seq else slice_sequences(self.negative_sequences, _tss, _up, _down)
        neg = self.positive_sequences if not slice_seq else slice_sequences(self.positive_sequences, _tss, _up, _down)
        neg_tokens = tokenize_sequences(neg, k=k)
        pos_tokens = tokenize_sequences(pos, k=k)
        tokens = np.vstack((neg_tokens, pos_tokens))
        return tokens

    def get_y(self):
        y = get_y(self.negative_sequences, self.positive_sequences)
        self.y = y
        return y

    def encode_dataset(self, _data=None):
        self.data = _data
        self.tokens = dict()
        for d in self.data:
            print(d)
            _slice = d.get_slice()
            _k = d.get_k()
            if _slice not in self.tokens.keys():
                self.tokens[_slice] = dict()
            if _k not in self.tokens[_slice].keys():
                self.tokens[_slice][_k] = self.set_tokens(k=_k, slice_seq=_slice)
            d.set_x(self.tokens[_slice][_k])
        self.X = [x.get_x() for x in self.data]
        return self.data

    def load_partition(self, split_index_A, split_index_B):
        X_split_A = list()
        X_split_B = list()
        for _, x in enumerate(self.X):
            X_split_A.append(x[split_index_A])
            X_split_B.append(x[split_index_B])
        y_split_A = self.y[split_index_A]
        y_split_B = self.y[split_index_B]
        return (X_split_A, y_split_A), (X_split_B, y_split_B)


class DataChunk(object):
    def __init__(self, k=1, encode='onehot', _slice=None):
        self._k = k
        self.encode = encode
        self.num_words = self._k ** 4
        self._x = None
        self._data_type = None
        self._tss_position = _slice[0] if _slice else None
        self._upstream = _slice[1] if _slice else None
        self._downstream = _slice[2] if _slice else None

    def set_x(self, tokens):
        self._x, self._data_type = encode_tokens(tokens, encoder_type=self.encode)

    def get_x(self):
        return self._x

    def get_encode(self):
        return self.encode

    def get_data_type(self):
        return self._data_type

    def get_k(self):
        return self._k

    def get_slice(self):
        return (self._tss_position, self._upstream, self._downstream) if self._tss_position else None

    def shape(self):
        return self._x.shape

    def load_partition(self, split_index_a, split_index_b):
        X_split_a = list()
        X_split_b = list()
        for _, x in enumerate(self._x):
            X_split_a.append(x[split_index_a])
            X_split_b.append(x[split_index_b])
        y_split_a = self.y[split_index_a]
        y_split_b = self.y[split_index_b]
        return (X_split_a, y_split_a), (X_split_b, y_split_b)


if __name__ == "__main__":
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    pass
