# coding: UTF-8
import os
import sys
sys.path.append('script')

import re
import pandas as pd
import jieba
from sklearn.utils import shuffle


class ProcessData(object):
    '''
    数据处理
    '''
    def __init__(self):
        pass


    def _segment_line_(self, line):
        """
        预处理和分词
        :param line: str
        :return: str
        """
        line = re.sub(
            "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '', line)
        tokens = jieba.cut(line, cut_all=False)
        return " ".join(tokens)


    def process_tiku(self, his_dfs, class_filename_map):
        '''

        :param his_dfs:
        :param class_filename_map:
        :return:
        '''
        for one_title in his_dfs:
            his_dfs[one_title]['item'] = his_dfs[one_title]['item'].progress_apply(
                lambda x: self._segment_line_(x))
            his_dfs[one_title]['label'] = class_filename_map[one_title][1]
        dataset_df = pd.concat([one_df for _, one_df in his_dfs.items()])  # 拼接三个df
        dataset_df = shuffle(dataset_df)
        return dataset_df