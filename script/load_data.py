# coding: UTF-8
import os
import sys
sys.path.append('script')

import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class LoadTiku(object):
    '''
    加载题库数据
    '''
    def __init__(self, tiku_fpath=None):
        '''
        初始化
        :param tiku_fpath: 题库原始文件路径
        '''
        self.tiku_fpath = tiku_fpath


    def load_tiku(self, tiku_fpath=None, sep=','):
        '''
        加载题库数据
        :param tiku_fpath: 文件地址
        :return:  tiku_df
        '''
        fpath = tiku_fpath if tiku_fpath else self.tiku_fpath
        tiku_df = pd.read_csv(fpath, sep=sep)
        return tiku_df