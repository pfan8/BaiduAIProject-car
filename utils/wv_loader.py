# -*- coding:utf-8 -*-
# Created by LuoJie at 11/22/19
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np
import codecs
# 引入日志配置
import logging
from utils.pickle_io import *

import utils.config as config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_w2v_model(save_wv_model_path=None):
    # 保存词向量模型
    wv_model = Word2Vec.load(config.save_wv_model_path)
    return wv_model


def get_vocab():
    wv_model = Word2Vec.load(config.save_wv_model_path)
    id2word = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    word2id = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    return word2id, id2word


def load_vocab(file_path=None):
    """
    读取字典
    :param file_path: pickle文件路径
    :return: 返回读取后的字典
    """
    if file_path is not None:
        word2id = load_file(file_path)
        id2word = {i:word for word,i in word2id}
    else:
        word2id = load_file(config.word2id_path)
        id2word = load_file(config.id2word_path)
    return word2id, id2word


def load_embedding_matrix():
    """
    加载 embedding_matrix_path
    """
    return load_file(config.embedding_matrix_path)
