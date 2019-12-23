# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Description:
# @Author: tiniwu (tiniwu@tencent.com)
# CopyRight: Tencent Company


import random
from annoy import AnnoyIndex
from gensim.models import KeyedVectors
import pickle
import sys
import logging
import codecs
import time
from bert_serving.client import BertClient
bc = BertClient()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()


class AnnoyTools:
    def __init__(self, config, is_build_annoy=False):
        self.word_embedding_path = config['word_embedding_path']
        self.annoy_path = config['annoy_path']
        self.word2index = {}
        self.index2word = {}
        self.annoy_index = AnnoyIndex(768, 'angular')
        self.word2index_path = config['word2index_path']
        self.index2word_path = config['index2word2_path']
        self.annoy_tree_num = config['annoy_tree_num']
        if is_build_annoy:
            self._save_annoy_index()
            logger.info("Build and save annoy index.")
        else:
            self._load_annoy_index()
            logger.info("Load the saved annoy index.")

    def _save_annoy_index(self):
        try:
            with codecs.open(self.word_embedding_path, 'r', encoding="utf-8") as f:
                count = 0
                for line in f:

                    count += 1
                    result = line.strip('\n').split()
                    if len(result) == 2: continue
                    word = result[0]
                    # index2word[count] = word
                    self.word2index[word] = count
                    vector = list(map(float, result[1:]))
                    self.annoy_index.add_item(count, vector)
        except Exception as e:
            logger.info(e)
        self.index2word = {v: k for k, v in self.word2index.items()}
        with open(self.word2index_path, 'wb') as f1, open(self.index2word_path, 'wb') as f2:
            pickle.dump(self.word2index, f1, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.index2word, f2, protocol=pickle.HIGHEST_PROTOCOL)
        self.annoy_index.build(self.annoy_tree_num)
        logger.info("Save annoy tree done.")
        self.annoy_index.save(self.annoy_path)

    def _load_annoy_index(self):
        self.annoy_index.load(self.annoy_path)
        with open(self.word2index_path, "rb") as f1, open(self.index2word_path, 'rb') as f2:
            self.word2index = pickle.load(f1)
            self.index2word = pickle.load(f2)
            logger.info("Loaded the saved word2index and index2word.")

    def get_similar_by_query(self, query, topk=21):
        query_vec = bc.encode([query])[0]
        idxes, dists = self.annoy_index.get_nns_by_vector(query_vec, topk, include_distances=True)
        idxes = [self.index2word[i] for i in idxes]
        similars = list(zip(idxes, dists))

        result = [(i, 0.1*(abs(1 - score)) + 0.5) for i, score in zip(idxes, dists)]
        print(result)
        return similars

    def _read_vector(self):
        model = KeyedVectors.load_word2vec_format("words.vector", binary=True)
        model.wv.save_word2vec_format(self.word_embedding_path, binary=False)


if __name__ == "__main__":
    config = {
        'word_embedding_path': "data/sports_vector.txt",
        'word2index_path': "data/word2id.pkl",
        'index2word2_path': 'data/id2word.pkl',
        'annoy_path': "data/sports.annoy",
        'annoy_tree_num': 100
    }
    if len(sys.argv) == 2:
        annoy_tools = AnnoyTools(config)
        query = sys.argv[1]
        start = time.time()
        logger.info(annoy_tools.get_similar_by_query(query))
        stop = time.time()
        logger.info("time/query by annoy Search = %.2f s" % (float(stop - start)))
    else:
        annoy_tools = AnnoyTools(config, True)
        start = time.time()
        logger.info(annoy_tools.get_similar_by_query("科比球鞋"))
        stop = time.time()
        logger.info("time/query by annoy Search = %.2f s" % (float(stop - start)))

