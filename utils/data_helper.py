# -*- coding: utf-8 -*-
# -------------------------------------
# Description:
# @Author: tiniwu (tiniwu@tencent.com)
# CopyRight: Tencent Company


# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from string import punctuation
from collections import Counter
import codecs
import jieba
from torch.utils.data import DataLoader, TensorDataset, Dataset

vocab_to_int = {"_pad_": 0, "__unk__": 1}


def token_to_id(sentence):
    return [vocab_to_int.get(word, 1) for word in sentence]


def data_process_similarity(file_path, max_seq_length=16, is_train=True):
    print("loading data")
    text_a, text_b, labels = [], [], []
    global vocab_to_int
    with codecs.open(file_path, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processor"):
            items = line.strip('\n').split('\t')
            text_a.append(jieba.lcut(items[0]))
            text_b.append(jieba.lcut(items[1]))
            labels.append(int(items[2]))

    # all_text_a = ''.join([c for c in text_a if c not in punctuation])
    # all_text_b = ''.join([c for c in text_b if c not in punctuation])
    if is_train:
        all_text = [item for sublist in text_a + text_b for item in sublist]
        count_words = Counter(all_text)

        total_words = len(count_words)
        sorted_words = count_words.most_common(total_words)
        vocab_to_int = {w: i+2 for i, (w, c) in enumerate(sorted_words)}
    # vocab_to_int['pad'] = 0
    # print(vocab_to_int)
    encoded_labels = np.array(labels)
    sentences_id_a = [token_to_id(sentence) for sentence in text_a]
    sentences_id_b = [token_to_id(sentence) for sentence in text_b]
    encoded_a = pad_features(sentences_id_a, max_seq_length)
    encoded_b = pad_features(sentences_id_b, max_seq_length)
    return encoded_a, encoded_b, encoded_labels

def data_process_class(file_path, max_seq_length=64):
    print("loading data")
    text_a,  labels = [], []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processor"):
            items = line.strip('\n').split(',')
            text_a.append(jieba.lcut(items[2]))
            labels.append(int(items[1]))

    # all_text_a = ''.join([c for c in text_a if c not in punctuation])
    # all_text_b = ''.join([c for c in text_b if c not in punctuation])
    all_text = [item for sublist in text_a for item in sublist]
    count_words = Counter(all_text)

    total_words = len(count_words)
    sorted_words = count_words.most_common(total_words)
    vocab_to_int = {w: i+1 for i, (w, c) in enumerate(sorted_words)}
    # vocab_to_int['pad'] = 0
    # print(vocab_to_int)
    encoded_labels = np.array(labels)
    sentences_id_a = [token_to_id(sentence, vocab_to_int) for sentence in text_a]

    encoded_a = pad_features(sentences_id_a, max_seq_length)

    return encoded_a, encoded_labels, vocab_to_int


def pad_features(sentences_id, max_seq_length):
    features = np.zeros((len(sentences_id), max_seq_length), dtype=int)
    for i, sentence_id in enumerate(sentences_id):
        sentence_length = len(sentence_id)
        if sentence_length <= max_seq_length:
            zeroes = list(np.zeros(max_seq_length - sentence_length))
            new = sentence_id + zeroes
        elif sentence_length > max_seq_length:
            new = sentence_id[0:max_seq_length]
        # print(new)
        features[i, :] = np.asarray(new)
    return features

def load_dataset(x1,x2,y):
    train_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=3)
    data_iter = iter(train_loader)
    sample_x, sample_y = data_iter.next()
    print(sample_x.size(),sample_y.size())
    print(sample_x, sample_y)

# encoded_a, encoded_b, encoded_labels, vocab_to_int = data_process("../data/ATEC/test.txt")
# load_dataset((encoded_a, encoded_b), encoded_labels)

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence