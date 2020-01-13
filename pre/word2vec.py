import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import logging
import sys
sys.path.insert(1, '.')
from utils.utils import * # 自定义工具类

def get_wv_model(model='word2vec', size=300, min_count=5, workers=8):
    '''
        训练Word2vec或FastText，并保存模型
        :param model :word2vec | fasttext
        :param size :词向量维度
        :param min_count :过滤低频词的阈值
        :param workers :并行数量
        :return :词向量模型
    '''
    # 引入日志配置
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    line_gen = LineSentence(merger_data_path) # generator，节省内存
    if model == 'word2vec':
        model = Word2Vec(line_gen, size=size, window=5, min_count=min_count, workers=workers)
    elif model == 'fasttext':
        model = FastText(line_gen, size=size, min_count=min_count, workers=workers)
    else:
        raise ValueError('model_paddle can only be word2vec or fasttext')
    return model

def get_embedding_matrix(vocab, model):
    '''
        获取vocab的词向量
        :param vocab :id2word dict
        :param model :word2vec | fasttext
        :return embedding_matrix with [V,E] shape
    '''
    embedding_matrix = np.zeros((len(vocab), model.wv.vector_size))
    for i,w in vocab.items():
        if w in model.wv:
            embedding_matrix[i] = model.wv[w]
    return embedding_matrix

    
if __name__ == '__main__':
    # 加载分词的dataframe
    merger_data_path = 'data/merged_train_test_seg_data.csv'
    merged_df = pd.read_csv(merger_data_path, header=None)
    print('数据数量：{}'.format(len(merged_df)))
    
    # 训练Word2vec和Fasttext模型，并保存
    print('训练Word2vec...')
    w2v_model = get_wv_model('word2vec')
    output_dir = 'output/word2vec'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    w2v_model.save(os.path.join(output_dir, 'word2vec.model_paddle'))
    # print('训练FastText...')
    # ft_model = get_wv_model('fasttext')
    # output_dir = 'output/fasttext'
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)
    # ft_model.save(os.path.join(output_dir, 'fasttext.model_paddle'))

    # '''构建vocab'''
    # # Word2vec Embedding Matrix
    # print('构建Word2vec词向量...')
    # w2v_matrix = w2v_model.wv.vectors
    # save_file(w2v_matrix, 'output/word2vec_emb')
    # FastText Embedding Matrix
    # print('构建FastText词向量...')
    # ft_matrix = ft_model.wv.vectors
    # save_file(ft_matrix, 'output/fasttext_emb')