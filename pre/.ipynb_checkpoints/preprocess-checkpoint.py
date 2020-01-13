import numpy as np
import pandas as pd
import re
import jieba
import os
import pickle
from multiprocessing import cpu_count, Pool

def save_file(data, filepath):
    '''
    封装保存数据API，适合数据不大的case
    :param data: 待保存数据，任意python object
    :filepath: 保存路径
    '''
    dirs = filepath.split(os.sep)[:-1]
    DIR = '.'
    while len(dirs):
        DIR += os.sep + dirs.pop(0)
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
    # 如果不是.pkl后缀，添加
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def clean_sentence(sentence):
    '''
    删去多余内容
    :param sentence:待处理字符串
    :return:过滤之后字符串
    '''
    if isinstance(sentence, str):
        # 过滤链接，需要先处理，否则除去特殊字符后不匹配
        sentence = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|#|\-)*\b', '', sentence, flags=re.MULTILINE)
        # 去除特殊字符
        sentence =  re.sub(
                        r'车主说|技师说|\[语音\]|\[图片\]|你好|您好|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+',
                        '', sentence)
        return sentence
    else:
        return ''

# 加载停用词
def load_stop_words(stop_word_path):
    '''
    加载停用词词典
    :param stop_word_path:停用词路径
    :return 停用词list
    '''
    with open(stop_word_path, encoding='utf8') as f:
        stop_words = f.readlines()
        stop_words = [x.strip() for x in stop_words]
    return stop_words

def filter_stopwords(words):
    '''
    过滤停用词（已加载停用词到stopwords）
    :param words: 待处理语句list
    :return: 过滤后停用词
    '''
    return [x for x in words if x not in stop_words]

def process_sentence(sentence):
    '''
    预处理流程
    :param sentence:待处理字符串
    :return 处理后字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 分词
    words = jieba.cut(sentence)
    # 过滤停用词
    words = filter_stopwords(words)
    # 以空格连接词组
    return ' '.join(words)

# 批处理
def process_dataframe(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(process_sentence)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(process_sentence)
    return df

def parallelize(df, func):
    '''
    多核运行func程序处理df
    :param df: 待处理dataframe
    :param func: 批处理流程
    :return: 处理后df
    '''
    # cpu数量
    cores = cpu_count()
    # 分块数量
    partitions = cores
    # split data
    data_split = np.array_split(df, partitions)
    # open process pool
    pool = Pool()
    data = pd.concat(pool.map(func, data_split))
    # close process pool
    pool.close()
    # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    pool.join()
    return data


if __name__ == "__main__":
    train_df = pd.read_csv('data/AutoMaster_TrainSet.csv')
    test_df = pd.read_csv('data/AutoMaster_TestSet.csv')

    train_df.dropna(subset=['Question','Dialogue','Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question','Dialogue'], how='any', inplace=True)

    # 加载汽车词典
    jieba.load_userdict('data/car_dict.txt')

    train_df = parallelize(train_df, process_dataframe)
    test_df = parallelize(test_df, process_dataframe)

    train_df['merge'] = train_df[['Question','Dialogue','Report']].apply(lambda x : ' '.join(x), axis=1)
    test_df['merge'] = test_df[['Question','Dialogue']].apply(lambda x : ' '.join(x), axis=1)

    merged_df = pd.concat([train_df['merge'], test_df['merge']], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df),len(test_df),len(merged_df)))

    merged_df.to_csv('data/merged_train_test_seg_data.csv', index=None, header=False)
    print('merged data saved!')

    vocab = set(' '.join(merged_df).split(' '))
    save_file(vocab, 'data/vocab')