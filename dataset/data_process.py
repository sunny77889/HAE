from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(2)
import sys

sys.path.append('../../')
import os
from collections import Counter

import pandas as pd
import glob
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def combin_file(source_path, des_path):
    csv_list = glob.glob(source_path + '*.csv')
    print(u'共发现%s个CSV文件' % len(csv_list))
    csv_files_list = []
    for file in csv_list:  # 循环读取同文件夹下的csv文件
        fd = pd.read_csv(file, index_col=None, encoding='unicode_escape')
        csv_files_list.append(fd)
    results = pd.concat(csv_files_list)
    results.to_csv(des_path, index=False)


def del_feas_flow(csv_file):
    fd = pd.read_csv(csv_file, index_col=None)
    fd = fd.drop(columns=['Flow ID', ' Timestamp',
                          ' Fwd Header Length.1'])  # 删除无关列
    fd = fd[~fd.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
    print(fd.shape)
    # fd['Tot pkts']=fd['Tot Fwd Pkts']+fd['Tot Bwd Pkts']
    # incomplete_flow=fd[fd['Tot pkts']<=3]
    fd.to_csv(csv_file, index=False)


def fea_select(data_file):
    data = pd.read_csv(data_file, index_col=None)
    data['Binary_label'] = data[' Label']
    data.loc[data[' Label'] == 'BENIGN', 'Binary_label'] = 0
    data.loc[data[' Label'] != 'BENIGN', 'Binary_label'] = 1
    labels = data['Binary_label'].astype(int)
    data = data.iloc[:, :-2]
    rf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    rf.fit(data, labels)
    types_labels = data.columns.values

    fea_imp = rf.feature_importances_
    indices = np.argsort(fea_imp)[::-1]
    del_sf = []
    sf = []
    for f in range(data.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, types_labels[indices[f]], fea_imp[indices[f]]))
        if fea_imp[indices[f]] < 0.003:
            del_sf.append(types_labels[indices[f]])
        else:
            sf.append(types_labels[indices[f]])
    data[sf].corr().to_csv('corr.csv', index=None)
    corr_values = np.array(data[sf].corr())
    print(type(corr_values))

    for i in range(corr_values.shape[1]):
        for j in range(i + 1, corr_values.shape[1]):
            if corr_values[i][j] > 0.9:
                del_sf.append(sf[i])
    return del_sf


# del_fea=['Bwd IAT Total', ' Down/Up Ratio', 'Idle Mean', 'FIN Flag Count', ' Active Max', ' Idle Min', ' Active Min', ' Bwd IAT Std', ' Bwd IAT Mean', ' Bwd IAT Min', 'Active Mean', ' Active Std', ' Bwd IAT Max', ' Idle Std', 'Fwd PSH Flags', ' SYN Flag Count', ' CWE Flag Count', ' Fwd URG Flags', ' RST Flag Count', ' ECE Flag Count', ' Bwd PSH Flags', 'Bwd Avg Bulk Rate', ' Bwd Avg Packets/Bulk', ' Bwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', 'Fwd Avg Bytes/Bulk', ' Bwd URG Flags', ' Fwd Avg Bulk Rate', ' Bwd Packet Length Mean', ' Bwd Packet Length Mean', ' Bwd Packet Length Mean', ' Bwd Packet Length Mean', ' Bwd Packet Length Mean', ' Bwd Packet Length Mean', ' Bwd Packet Length Mean', ' Packet Length Std', ' Packet Length Std', ' Packet Length Std', ' Packet Length Std', ' Packet Length Std', ' Packet Length Std', ' Packet Length Std', ' Average Packet Size', ' Average Packet Size', ' Average Packet Size', ' Bwd Packet Length Std', ' Bwd Packet Length Std', ' Bwd Packet Length Std', ' Packet Length Variance', 'Bwd Packet Length Max', 'Bwd Packet Length Max', ' Avg Bwd Segment Size', ' Avg Bwd Segment Size', ' Subflow Bwd Bytes', ' Subflow Bwd Bytes', ' Subflow Bwd Bytes', ' Subflow Bwd Bytes', ' Subflow Bwd Bytes', ' Packet Length Mean', ' Fwd Packet Length Max', ' Fwd Packet Length Max', ' Fwd Packet Length Max', ' Fwd Packet Length Mean', ' Subflow Fwd Bytes', ' Total Length of Bwd Packets', ' Total Length of Bwd Packets', ' Total Length of Bwd Packets', ' Total Length of Bwd Packets', ' Flow IAT Max', ' Flow IAT Max', ' Flow IAT Max', ' Flow IAT Max', ' Subflow Bwd Packets', ' Subflow Bwd Packets', ' Subflow Bwd Packets', ' Fwd IAT Std', ' Fwd IAT Std', ' Flow Duration', 'Subflow Fwd Packets', 'Subflow Fwd Packets', ' Idle Max', ' Idle Max', ' Total Backward Packets', ' Fwd IAT Max', ' Flow Packets/s', ' Fwd IAT Mean']
def norm_num(data_file,dataset_path):
    unnorm_cols=[' Source IP', ' Source Port', ' Destination IP', ' Destination Port',' Label']
    rawdata = pd.read_csv(data_file, index_col=None)

    data = pd.get_dummies(data=rawdata, columns=[' Protocol'])  # one_hot
    print(data.columns.values)

    ben_data = data[rawdata[' Label'] == "BENIGN"]
    mal_data = data[rawdata[' Label'] != "BENIGN"]

    train_data, ben_test_data = train_test_split(ben_data, test_size=len(mal_data) / len(data))  #保证测试集中正常流量：异常=1:1
    test_data=pd.concat([ben_test_data, mal_data])
    y_test= np.array([0] * len(ben_test_data) + [1] * len(mal_data))


    train_ids=pd.DataFrame(train_data,columns=unnorm_cols)
    test_ids = pd.DataFrame(test_data,columns=unnorm_cols)
    test_data = test_data.drop(columns=unnorm_cols)
    train_data = train_data.drop(columns=unnorm_cols)

    norm = MinMaxScaler()  # 归一化
    train_data = norm.fit_transform(train_data)
    test_data = norm.transform(test_data)
    test_data = np.c_[test_data, test_ids, y_test]
    train_data = np.c_[train_data, train_ids, np.array([0] * len(train_data))]
    print(train_data.shape)
    print(test_data.shape)
    np.save(dataset_path+'train_data', train_data)
    np.save(dataset_path+'test_data', test_data)



if __name__ == '__main__':
    csv_data_file='./raw_data/'  #原始特征文件路径
    dataset_path = './' #预处理生成的文件路径
    data_file = dataset_path+'com_data.csv' #原始特征文件合并的文件

    combin_file(csv_data_file, data_file)
    print('合并完毕')

    del_feas_flow(data_file)
    print('数据清洗完成')

    norm_num(data_file, dataset_path)
    print("归一化完成")
