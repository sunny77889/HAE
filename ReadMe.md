## 基于分层自编码器的异常网络流量检测HAE
### 模型训练
`hae_dec.py` HAE模型训练和测试的主函数
`model.py` 模型类：HAE、AE、ABAE
`tools.py` 工具类：数据读取、模型评估


`/savedModel/`HAE模型保存与加载路径
`dataset/` 原始特征数据存放路径 
`dataset/data_process.py` 数据预处理,主要针对CIC2017数据集


### 对比实验
`/experiments/ABAE_dec.py`对比模型：ABAE（基于Adaboost对AE进行集成）
`/experiments/AE.py`对比模型：AE（自编码器）
`/experiments/ML.py`对比模型：机器学习方法：PCA，IForest，HBOS

### 操作流程
1. 下载CIC2017数据集（https://www.unb.ca/cic/datasets/ids-2017.html） 将csv格式的数据文件放在`dataset/rawdata/`路径下。
2. 运行`dataset/data_process.py`进行数据预处理
3. 运行`hae_dec.py` 进行HAE模型训练和测试
4. 运行`experiments/`目录下文件进行对比模型的训练与测试
