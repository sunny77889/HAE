#%%
from numpy.random import seed
from tensorflow.python.keras.engine.base_layer import Layer

seed(1)
import tensorflow as tf

tf.random.set_seed(2)
import os
import sys

sys.path.append('/home/xiaoqing/HAE/')
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import (LSTM, AveragePooling1D, MaxPool1D,
                                     MaxPooling1D)
from tensorflow.keras.models import Model, Sequential
# from pyod.models.ocsvm import OCSVM
from tensorflow.keras.optimizers import Adam
from tools import evaluation, evaluation_types, plot_roc, read_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
seq_len=6
def mse_loss(y_true, y_pred):
    n = len(y_true)
    ml = tf.zeros(seq_len, dtype=tf.float32)
    for i in range(n):
        kk = tf.losses.mean_squared_error(y_true[i], y_pred[i])
        ml+=kk
    return ml/float(seq_len)


def lstm_autoencoder(original_dim):
    model = Sequential()
    model.add(LSTM(64, activation='tanh',return_sequences=True, input_shape=(6,original_dim)))
    model.add(LSTM(32, activation='tanh',return_sequences=True))
    model.add(AveragePooling1D(pool_size=32, data_format='channels_first'))
    model.add(LSTM(64, activation='tanh',return_sequences=True))
    model.add(LSTM(original_dim, activation='tanh', return_sequences=True))
    adam = Adam(0.0001)
    model.compile(optimizer=adam, loss=mse_loss)
    model.summary()
    return model


class MinPooling1D(Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MinPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MinPooling1D, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return K.min(x, axis=1, keepdims=True)

def model_train(train_data, saved_model_path):
    train_seq_len=len(train_data)//6
    fd=train_data.shape[1] #流特征维度
    train_seq=train_data[:train_seq_len*6].reshape(train_seq_len, 6, fd)

    lstm_ae = lstm_autoencoder(fd)
    history = lstm_ae.fit(train_seq,train_seq,batch_size=128,epochs=30,validation_data=None)
    
    lstm_ae.save_weights(saved_model_path)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('train_loss')
    plt.show()

def model_test(train_data, test_data,saved_model_path):

    test_seq_len=len(test_data)//6
    tl = test_seq_len*6
    fd=train_data.shape[1] #流特征维度
    test_seq=test_data[:test_seq_len*6, :fd].reshape(test_seq_len, 6, fd) 

    lstm_ae = lstm_autoencoder(fd)
    lstm_ae.load_weights(saved_model_path)
    pred = lstm_ae.predict(test_seq.astype(np.float32))
    print(pred.shape)
    test_labels = test_data[:, -1]
    test_types=test_data[:, -2]
    test_binary_labels = np.array(test_labels[:tl]).astype(int)
    test_losses = tf.losses.mean_squared_error(test_data[:tl,:fd].astype(np.float32), pred.reshape(-1, fd))
    threshold = plot_roc(test_binary_labels, test_losses) #根据roc曲线确定最佳阈值
    pred=np.ones(len(test_losses))
    pred[test_losses<threshold]=0
    evaluation(test_binary_labels, pred)
    evaluation_types(pred, test_types[:tl])

if __name__ =='__main__':
    train_path = '../dataset/train_data.npy'
    test_path='../dataset/test_data.npy'
    saved_model_path = './savedModel/LSTMAE/'
    
    train_data=np.load(train_path, allow_pickle=True)
    test_data=np.load(test_path, allow_pickle=True)
    model_train(train_data, saved_model_path)
    # model_test(train_data, test_data, saved_model_path)
    

