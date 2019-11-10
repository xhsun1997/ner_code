## Name Entity Recognition

该项目使用tensorflow框架实现命名实体识别。命名实体识别是识别提及命名实体的文本范围，并将其分类为预定义的类别，例如人，位置，组织等。命名实体识别是各种自然语言应用程序（例如智能问答，文本摘要和机器翻译）的基础。

### 该项目主要利用以下三个模型：

1. Hidden Markov Model
2. BiLSTM-CRF Model
3. CNN-BiLSTM-CRF Model

隐马尔可夫模型(Hidden Markov Model)是概率图模型的一种，被广泛的用来处理序列标注问题。随着人工智能的兴起，深度学习已经被广泛的用在各种自然语言处理任务，在实体识别方面对比早期的方法已经取得了显著的效果。
在实体识别的所有神经网络模型中，BiLSTM-CRF模型是十分流行的同时效果也是十分显著的。具体的做法是将所有的单词和对应的标签嵌入到向量空间，每一个单词和对应的标签都表示为实值向量，然后送进BiLSTM网络训练，为了把达到更高的预测准确率，在BiLSTM网络上连接一个CRF层。CRF是条件随机场(Conditional Random Field)，条件随机场同样是概率图模型的一种，不同于HMM，CRF是无向图。之所以CRF可以提高预测的准确率是因为它考虑到了标签与标签之间的关系，而BiLSTM只能针对预测文本和标签之间的关系，例如 B-PER后面不会出现I-LOC，但是BiLSTM所输出的预测结果不会考虑到这种情况。为了考虑到字符级别的特征，可以用卷积神经网络CNN提取出字符级别的表示向量，然后联合单词级别的向量再送进BiLSTM网络。

我这里综合考虑了不同网络结构产生的不同效果。即对于BiLSTM_CRF.py文件有两种网络结构：BiLSTM以及BiLSTM-CRF。对于CNN_BiLSTM_CRF.py也有两种结构：CNN_BiLSTM以及CNN_BiLSTM_CRF。这四种神经网络结构以及HMM模型所得出的准确率，召回率以及F1分数放在model_results文件夹的五个.txt文件里。另外，模型用的word embeddings是用的斯坦福大学发表的GloVe word embeddings。由于文件较大无法上传到github上，地址在(https://nlp.stanford.edu/projects/glove/)。
利用HMM模型可以简单的测试两个句子如下图所示

![result.png](https://i.loli.net/2019/11/10/3Upsy2wbRNLtxmD.png)  ![result_1.png](https://i.loli.net/2019/11/10/xGzucS84VONg2ZL.png)

参考论文:

1. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
2. Named Entity Recognition with Bidirectional LSTM-CNNs
3. Neural Architecture for Name Entity Recognition
4. Bidirectional LSTM-CRF Models for Sequence Tagging