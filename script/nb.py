# coding: UTF-8
import os
import sys
sys.path.append('script')

from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB


class NB(object):
    '''
    朴素贝叶斯
    '''
    def __init__(self, nb_type='Multinomial'):
        self.nb_model = self._get_model_(nb_type)


    def _get_model_(self, nb_type):
        '''
        获取模型
        :param nb_type:
        :return: model class
        '''
        if 'Gaussian' == nb_type:
            return GaussianNB()
        elif 'Multinomial' == nb_type:
            return MultinomialNB()
        elif 'Complement' == nb_type:
            return ComplementNB()
        elif 'Bernoulli' == nb_type:
            return BernoulliNB()
        else:
            raise NameError('no type for {}'.format(nb_type))





