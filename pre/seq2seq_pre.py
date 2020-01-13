import sys
sys.path.insert(1, '.')
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import utils.config as config
from utils.multi_proc_utils import parallelize
from utils.wv_loader import get_vocab
from utils.pickle_io import *
import re
import jieba
jieba.load_userdict(config.user_dict)
import logging
# seq2seq预处理

def seq2seq_pre(sentence):
    # 要加空格，否则会连接在一起
    sentence = re.sub('车主说', ' TOKEN1 ', sentence, flags=re.MULTILINE)
    sentence = re.sub('技师说', ' TOKEN2 ', sentence, flags=re.MULTILINE)
    sentence = re.sub('\[图片\]', ' TOKEN3 ', sentence, flags=re.MULTILINE)
    sentence = re.sub('\[语音\]', ' TOKEN4 ', sentence, flags=re.MULTILINE)
    sentence = re.sub('(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|#|\-)*\b', ' TOKEN5 ', sentence, flags=re.MULTILINE)
    sentence = re.sub('\|','',sentence, flags=re.MULTILINE)
    # 训练词向量时，已加载词典到jieba，直接调用
    words = jieba.cut(sentence)
    return ' '.join(words)

def process_seq2seq(df):
    '''
    seq2seq批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(seq2seq_pre)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(seq2seq_pre)
    return df

def mark_proc(sentence, max_len, vocab, update=False):
    '''
    < start > < end > < pad > < unk >
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1.过滤过多空格导致的空值''
    words = [x for x in words if len(x)]
    # 2. 截取规定长度的词数
    words = words[:max_len]
    if update:
        sentence = words
    else:
        # 5. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
        sentence = [word if word in vocab else '<UNK>' for word in words]
        # 3. 填充< start > < end >
        sentence = ['<START>'] + sentence + ['<STOP>']
        # 4. 判断长度，填充　< pad >
        sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)

def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' ')+1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))

if __name__ == '__main__':
    train_df = pd.read_csv(config.train_data_path)
    test_df = pd.read_csv(config.test_data_path)
    train_df.dropna(subset=['Question','Dialogue','Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question','Dialogue'], how='any', inplace=True)
    
    train_df = parallelize(train_df, process_seq2seq)
    test_df = parallelize(test_df, process_seq2seq)

    # 构建DataSet
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 获取输入数据 适当的最大长度
    train_x_max_len = get_max_len(train_df['X'])
    test_x_max_len = get_max_len(test_df['X'])

    x_max_len = max(train_x_max_len, test_x_max_len)
    if x_max_len > 300:
        print('X长度（{}）超过最大长度300，重置为300'.format(x_max_len))
        x_max_len = 300

    # 获取标签数据 适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    if train_y_max_len > 300:
        print('Y长度（{}）超过最大长度300，重置为300'.format(train_y_max_len))
        train_y_max_len = 300

    # 取80%数据集中数据训练词向量，构建vocab，剩余20%出现OOV则标记为<UNK>
    update_nums = int(len(train_df)*0.8)
    # 训练集X处理
    train_df['X'] = train_df['X'].apply(lambda x: mark_proc(x, x_max_len, vocab=None, update=True))
    # 训练集Y处理
    train_df['Y'] = train_df['Report'].apply(lambda x: mark_proc(x, train_y_max_len, vocab=None, update=True))
    # 保存更新词向量的数据
    update_df = train_df.sample(update_nums)
    update_df['X'].to_csv(config.train_x_update_vocab_path, index=None, header=False)
    update_df['Y'].to_csv(config.train_y_update_vocab_path, index=None, header=False)
    # 重新训练Word2Vec
    w2v_model = Word2Vec.load(config.save_wv_model_path)
    print('start retrain w2v model_paddle')
    w2v_model.build_vocab(LineSentence(config.train_x_update_vocab_path), update=True)
    w2v_model.train(LineSentence(config.train_x_update_vocab_path)
                    , epochs=config.wv_train_epochs
                    , total_examples=w2v_model.corpus_count)
    print('1/2')
    w2v_model.build_vocab(LineSentence(config.train_y_update_vocab_path), update=True)
    w2v_model.train(LineSentence(config.train_y_update_vocab_path)
                    , epochs=config.wv_train_epochs
                    , total_examples=w2v_model.corpus_count)
    print('2/2')
    # 构建Vocab
    vocab = w2v_model.wv.vocab

    # 更新词向量后，根据vocab重新处理数据集，标注<UNK>
    # 训练集X处理
    train_df['X'] = train_df['X'].apply(lambda x: mark_proc(x, x_max_len, vocab))
    # 训练集Y处理
    train_df['Y'] = train_df['Report'].apply(lambda x: mark_proc(x, train_y_max_len, vocab))
    # 测试集X处理
    test_df['X'] = test_df['X'].apply(lambda x: mark_proc(x, x_max_len, vocab))

    # 保存中间结果数据
    train_df['X'].to_csv(config.train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(config.train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(config.test_x_pad_path, index=None, header=False)
    # 新数据再次训练Word2Vec（带有<UNK>）
    print('start retrain w2v model_paddle')
    w2v_model.build_vocab(LineSentence(config.train_x_pad_path), update=True)
    w2v_model.train(LineSentence(config.train_x_pad_path)
                    , epochs=config.wv_train_epochs
                    , total_examples=w2v_model.corpus_count)
    print('1/3')
    w2v_model.build_vocab(LineSentence(config.train_y_pad_path), update=True)
    w2v_model.train(LineSentence(config.train_y_pad_path)
                    , epochs=config.wv_train_epochs
                    , total_examples=w2v_model.corpus_count)
    print('2/3')
    w2v_model.build_vocab(LineSentence(config.test_x_pad_path), update=True)
    w2v_model.train(LineSentence(config.test_x_pad_path)
                    , epochs=config.wv_train_epochs
                    , total_examples=w2v_model.corpus_count)
    # 保存Word2Vec模型
    w2v_model.save(config.save_wv_model_path)

    # Word2vec Embedding Matrix
    print('构建Word2vec Vocab...')
    w2v_matrix = w2v_model.wv.vectors
    save_file(w2v_matrix, config.embedding_matrix_path)

    # 保存id2word和word2id
    vocab = w2v_model.wv.vocab
    id2word = {i:x[0] for i,x in enumerate(vocab.items()) }
    word2id = {x:i for i,x in id2word.items()}
    save_file(id2word, config.id2word_path)
    save_file(word2id, config.word2id_path)