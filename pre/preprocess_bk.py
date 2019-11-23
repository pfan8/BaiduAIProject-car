import numpy as np
import pandas as pd
import jieba
import os
import sys
import pickle
import glob
from tqdm import tqdm
from string import digits, punctuation

def save_file(data, filepath):
    '''
        保存数据为pickle文件，支持自动生成目录
    '''
    dirs = filepath.split(os.sep)[:-1]
    DIR = '.'
    while len(dirs):
        DIR += os.sep + dirs.pop(0)
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def get_tokens(cut_func, corpus, batch_num=10000):
    '''
        根据tokenizer(cut_func)，对corpus进行分词
        分batch操作，从而进度可视化
    '''
    result = []
    for i in tqdm(range(0, len(corpus), batch_num)):
        result.extend([cut_func(x) for x in corpus[i:i+batch_num]])
    return result

def get_vocab(data_path, alias=''):
    '''
        建立字典
        data_path : 数据路径
        alias : 输出时候的文件夹名，默认不构建单独的文件夹
    '''
    print('='*25 + 'DataSet {}'.format(alias) + '='*25)

    out_dir = 'output' + os.sep
    if len(alias):
        out_dir += alias + os.sep

    # 读取corpus
    dataset = pd.read_csv(data_path, sep=',')
    corpus = []
    corpus.extend(dataset.Brand.dropna().tolist())
    save_file(corpus, '{}brand_corpus'.format(out_dir))
    lb = len(corpus)
    corpus.extend(dataset.Model.dropna().tolist())
    save_file(corpus[lb:], '{}model_corpus'.format(out_dir))
    lm = len(corpus)
    corpus.extend(dataset.Question.str.replace(r'\[语音\]|\[图片\]','').dropna().tolist())
    save_file(corpus[lm:], '{}question_corpus'.format(out_dir))
    lq = len(corpus)
    corpus.extend(dataset.Dialogue.str.replace(r'\[语音\]|\[图片\]','').dropna().tolist())
    save_file(corpus[lq:], '{}dialogue_corpus'.format(out_dir))
    ld = len(corpus)
    if 'Report' in dataset.columns:
        corpus.extend(dataset.Report.str.replace(r'\[语音\]|\[图片\]','').dropna().tolist())
        save_file(corpus[lb:], '{}report_corpus'.format(out_dir))

    # 加载汽车行业字典
    jieba.load_userdict('car_dict.txt')
    cut = lambda word : ' '.join(jieba.cut(word))
    print('Tokenize...')
    tokens = get_tokens(cut, corpus)
    words = [x.split(' ') for x in tokens]
    words = [x for sublist in words for x in sublist if x != ''] # flatten list

    # 取消过滤停用词（seq2seq模型和word2vec需要上下文）
#     print('Filter Stopwords...')
#     stopwords = []
#     files = glob.glob('stopwords/*.txt')
#     for file in files:
#         with open(file) as f:
#             stopwords.extend([x.rstrip() for x in f.readlines()])

#     # 加入特殊符号
#     stopwords.extend(list(u'[^a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：' + digits + punctuation + '\u4e00-\u9fa5]+'))
#     stopwords = set(stopwords)

#     words = [x for x in words if x not in stopwords]

    # 构建字典
    print('Establish Vocab...')
    from collections import Counter
    result = sorted(Counter(words).items(),key= lambda x:(-x[1],x[0]))
    result.insert(0, ('<BOS>', -1))
    result.insert(0, ('<EOS>', -1))
    result.insert(0, ('<PAD>', -1))
    result.insert(0, ('<UNK>', -1))

    id2word = {i+1:x[0] for i,x in enumerate(result)}
    word2id = {x:i for i,x in id2word.items()}

    print('Save Result...')
    save_file(id2word, '{}id2word'.format(out_dir))
    save_file(word2id, '{}word2id'.format(out_dir))
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python preprocess.py DATA_PATH ALIAS(option)')
        sys.exit()
    data_path = sys.argv[1]
    alias = sys.argv[2] if len(sys.argv)>2 else ''
    get_vocab(data_path, alias)
