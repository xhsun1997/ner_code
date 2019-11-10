## Name Entity Recognition

该项目使用tensorflow框架实现命名实体识别。命名实体识别是识别提及命名实体的文本范围，并将其分类为预定义的类别，例如人，位置，组织等。命名实体识别是各种自然语言应用程序（例如智能问答，文本摘要和机器翻译）的基础。

该项目主要利用以下三个模型：

1. Hidden Markov Model
2. BiLSTM-CRF Model
3. CNN-BiLSTM-CRF Model

隐马尔可夫模型(Hidden Markov Model)是概率图模型的一种，被广泛的用来处理序列标注问题。随着人工智能的兴起，深度学习已经被广泛的用在各种自然语言处理任务，在实体识别方面对比早期的方法已经取得了显著的效果。
在实体识别的所有神经网络模型中，BiLSTM-CRF模型是十分流行的同时效果也是十分显著的。具体的做法是将所有的单词和对应的标签嵌入到向量空间，每一个单词和对应的标签都表示为实值向量，然后送进BiLSTM网络训练，为了把达到更高的预测准确率，在BiLSTM网络上连接一个CRF层。CRF是条件随机场(Conditional Random Field)，条件随机场同样是概率图模型的一种，不同于HMM，CRF是无向图。为了考虑到字符级别的特征，可以在将训练的词和
### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/xhsun1997/ner_code/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
