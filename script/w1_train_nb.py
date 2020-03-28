# coding: UTF-8
import os
import sys
sys.path.append('script')

import pandas as pd

import load_data
import process_data
import nb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == "__main__":
    # S1: 加载数据
    ori_data_path = 'data/ori_data/百度题库/高中_历史/origin/'
    class_filename_map = {'ancient': ['古代史.csv', 1],
                          'contemporary': ['现代史.csv', 2],
                          'modern': ['近代史.csv', 3]}
    his_dfs = {title: pd.read_csv(os.path.join(ori_data_path, file_label[0])) for title, file_label in
               class_filename_map.items()}  # 加载数据

    # S2: 处理数据
    process_data_obj = process_data.ProcessData()  # 初始化数据处理对象
    dataset_df = process_data_obj.process_tiku(his_dfs, class_filename_map)
    vectorizer = TfidfVectorizer(max_features=2500, min_df=5)
    X = vectorizer.fit_transform(dataset_df['item'])
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), dataset_df['label'], test_size=0.2,
                                                        random_state=42)

    # S3：模型训练
    gau_nb_obj = nb.NB('Gaussian')
    gau_nb_obj.nb_model.fit(X_train, y_train)
    gau_y_pred = gau_nb_obj.nb_model.predict(X_test)

    mul_nb_obj = nb.NB('Multinomial')
    mul_nb_obj.nb_model.fit(X_train, y_train)
    mul_y_pred = mul_nb_obj.nb_model.predict(X_test)

    com_nb_obj = nb.NB('Complement')
    com_nb_obj.nb_model.fit(X_train, y_train)
    com_y_pred = com_nb_obj.nb_model.predict(X_test)

    ber_nb_obj = nb.NB('Bernoulli')
    ber_nb_obj.nb_model.fit(X_train, y_train)
    ber_y_pred = ber_nb_obj.nb_model.predict(X_test)
    # S4：评估
    print('GauusianNB classification_report: \n')
    print(classification_report(y_test, gau_y_pred))
    print('MultinomialNB classification_report: \n')
    print(classification_report(y_test, mul_y_pred))
    print('ComplementNB classification_report: \n')
    print(classification_report(y_test, com_y_pred))
    print('BernoulliNB classification_report: \n')
    print(classification_report(y_test, ber_y_pred))
