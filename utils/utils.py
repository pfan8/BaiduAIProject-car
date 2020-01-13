import os
import pickle

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
        
def load_file(filepath):
    '''
    封装提取数据API，文件格式为pkl
    :param filepath: 文件路径
    :retunn: 提取的数据
    '''
    with open(filepath, 'rb') as f:
        return pickle.load(f)
