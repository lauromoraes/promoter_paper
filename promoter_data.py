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


def get_kmer_to_int_enconder(k):
    mers = get_all_possible_k_mers(k)
    label_encoder = LabelEncoder()
    label_encoder.fit(mers)
    return label_encoder


def encode_kmers_to_int_sequences(seqs, k=1):
    encoder = get_kmer_to_int_enconder(k)
    new_seqs = np.vstack([encoder.transform(s) for s in seqs])
    return new_seqs


def get_kmer_to_onehot_enconder(k):
    mers = np.array(get_all_possible_k_mers(k)).reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(mers)
    return onehot_encoder


def encode_kmers_to_onehot_sequences(seqs, k=1):
    encoder = get_kmer_to_onehot_enconder(k)
    new_seqs = np.stack([encoder.transform(np.array(list(s)).reshape(-1, 1)) for s in seqs])
    new_seqs = np.expand_dims(new_seqs, axis=3)
    return new_seqs


def encode_dna_to_vec(tokens, k=1):
    if not type(tokens) == list:
        tokens = tokens.tolist()
    model = Word2Vec(min_count=1, workers=4, size=10, window=3)
    model.build_vocab(sentences=tokens)
    model.train(sentences=tokens, total_examples=len(tokens), epochs=10)
    seqs = np.array([model.wv.__getitem__(t) for t in tokens])
    # print(seqs)
    # print(seqs.shape)
    return seqs


def encode_fastdna_to_vec(tokens, k=1):
    if not type(tokens) == list:
        tokens = tokens.tolist()
    model = FastText(min_count=1, workers=4, size=10, window=3)
    model.build_vocab(sentences=tokens)
    model.train(sentences=tokens, total_examples=len(tokens), epochs=10)
    seqs = np.array([model.wv.__getitem__(t) for t in tokens])
    # print(seqs)
    # print(seqs.shape)
    return seqs


def get_y(neg_data, pos_data):
    neg_labels = [0 for _, _ in enumerate(neg_data)]
    pos_labels = [1 for _, _ in enumerate(pos_data)]
    y = np.array(neg_labels + pos_labels)
    return y


def encode_tokens(tokens, encoder_type=0):
    encoder_types = [encode_kmers_to_int_sequences, encode_kmers_to_onehot_sequences, encode_dna_to_vec,
                     encode_fastdna_to_vec]
    k = len(tokens[0][0])
    seq = encoder_types[encoder_type](tokens, k=k)
    return seq


def encode_nucleotides_sequences(sequences, k=1, encoder_type=0):
    tokens = tokenize_sequences(sequences, k=k)
    new_sequences = encode_tokens(tokens, encoder_type=encoder_type)
    return new_sequences


class PromoterData(object):
    def __init__(self, fasta_path):
        self.fasta_path = os.path.join(*(os.path.split(fasta_path)))  # Normalize path for OS
        self.k = 1
        self.org_name = None
        self.neg_path = None
        self.pos_path = None
        self.negative_sequences = None
        self.positive_sequences = None
        self.tss_position = None
        self.downstream = None
        self.upstream = None
        self.tokens = None
        self.y = None
        self.X = None

    def set_organism_sequences(self, org_name, slice_seqs=False, tss_position=None, downstream=None, upstream=None):
        print(' -' * 30)
        self.org_name = org_name
        self.tss_position = tss_position
        self.downstream = downstream
        self.upstream = upstream

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

        if slice_seqs:
            print('\nSlicing sequences with:\nTss = {} |\tUpstream = {} |\t Downstream = {}'.format(tss_position,
                                                                                                    upstream,
                                                                                                    downstream))
            self.negative_sequences = slice_sequences(self.negative_sequences, tss_position, upstream, downstream)
            self.positive_sequences = slice_sequences(self.positive_sequences, tss_position, upstream, downstream)
            print('Length of sequences changed from {} to {}'.format(seqs_lenght, len(self.negative_sequences[0])))

    def set_k(self, k=1):
        self.k = k

    def set_tokens(self, k=None):
        if not k:
            k = self.k
        neg_tokens = tokenize_sequences(self.negative_sequences, k=k)
        pos_tokens = tokenize_sequences(self.positive_sequences, k=k)
        tokens = np.vstack((neg_tokens, pos_tokens))
        self.tokens = tokens
        return tokens

    def get_y(self):
        y = get_y(self.negative_sequences, self.positive_sequences)
        self.y = y
        return y

    def encode_dataset(self, encoder_types=0, tokens=None):
        X = list()
        if not tokens:
            tokens = self.set_tokens()
        if type(encoder_types) == list or type(encoder_types) == tuple:
            for encoder in encoder_types:
                x = encode_tokens(tokens, encoder_type=encoder)
                X.append(x)
        elif type(encoder_types) == int:
            x = encode_tokens(tokens, encoder_type=encoder_types)
            X.append(x)
        self.X = X
        return X

    def load_partition(self, split_index_A, split_index_B):
        X_split_A = list()
        X_split_B = list()
        for _, x in enumerate(self.X):
            X_split_A.append(x[split_index_A])
            X_split_B.append(x[split_index_B])
        y_split_A = self.y[split_index_A]
        y_split_B = self.y[split_index_B]
        return (X_split_A, y_split_A), (X_split_B, y_split_B)


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="ticks", color_codes=True)

    organism = 'Bacillus'
    k = 3
    tss_pos = 59
    downstream = 20
    upstream = 20
    data = PromoterData('./fasta')
    data.set_organism_sequences(organism, slice_seqs=True, tss_position=tss_pos, downstream=downstream,
                                upstream=upstream)
    K = data.encode_dataset(encoder_types=0)

    print(' -' * 30)
    print('\t>>> END <<<')
