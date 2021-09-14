from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(4)
import sys

import numpy as np
import os
from hae.models import AE_Tree
from hae.tools import read_dataset, evaluation, evaluation_types
from time import time
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_path = "./dataset/train_data.npy"
test_path = "./dataset/test_data.npy"
saved_model_path = './savedModel/hae/'


def HAE_train(AE_num, epoch_num):
    '''
    HAE模型训练
    :param AE_num: 集成的AE的数目
    :param epoch_num: 每个AE迭代训练的次数
    '''
    print("Read data......")

    train_data, train_ids, test_data, test_ids = read_dataset(train_path, test_path)  # ids:[sip,sport,dip,dport,types,labels]

    clf = AE_Tree(n_clf=AE_num, original_dim=train_data.shape[1])

    # 模型训练
    print("HAE Training......")
    t1 = time()
    clf.fit(X=train_data, k=2/3, epochs=epoch_num, batch_size=128)
    print("HAE Train Time:".format(time() - t1))
    print("Saving Model......")
    # clf.save(saved_model_path)


def HAE_test(AE_num):
    '''
    HAE模型测试
    :param AE_num: 实际测试使用“AE_num”个AE进行集成测试
    '''
    print("Read data......")
    train_data, train_ids, test_data, test_ids = read_dataset(train_path, test_path)  # ids:[sip,sport,dip,dport,types,labels]

    clf = AE_Tree(n_clf=AE_num, original_dim=train_data.shape[1])

    # 测试集评估
    print("load model.....")
    clf.load(saved_model_path)
    t1 = time()
    pre_data = clf.predict(test_data)
    print('HAE Test Time:', time() - t1)

    print("Test Metrics")
    test_labels = test_ids[:, -1]
    test_types = test_ids[:, -2]
    test_binary_labels = np.array(test_labels).astype(int)
    print(Counter(test_binary_labels))
    pre_data = pre_data.astype(int)  # 测试样本预测值，0，1序列
    test_ip_port = test_ids[:, :4]  # 源ip，源port，目的IP、目的port

    evaluation(test_binary_labels, pre_data)
    evaluation_types(pre_data, test_types)

    # # 模型预测(单条流)
    # flow=test_data[0]
    # print("load model.....")
    # clf.load(saved_model_path)
    # normal = clf.online_dect(flow)
    # if normal == False:
    #     print('异常')


if __name__ == '__main__':
    HAE_train(AE_num=5,epoch_num=10)
    # HAE_test(AE_num=2)
