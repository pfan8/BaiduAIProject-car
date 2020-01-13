# -*- coding:utf-8 -*-
# Created by LuoJie at 11/27/19


def save_vocab(file_path, data):
    with open(file_path) as f:
        for i in data:
            f.write(i)


def save_dict(save_path, dict_data):
    """
    保存字典
    :param save_path: 保存路径
    :param dict_data: 字典路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("{}\t{}\n".format(k, v))


