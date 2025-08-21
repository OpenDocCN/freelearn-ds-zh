# 第十一章：使用 Word2Vec 创建和可视化单词向量

在本章中，我们将涵盖以下内容：

+   获取数据

+   导入必要的库

+   准备数据

+   构建和训练模型

+   进一步可视化

+   进一步分析

# 介绍

在对文本数据进行神经网络训练并使用 LSTM 单元生成文本之前，重要的是要了解文本数据（如单词、句子、客户评论或故事）在输入神经网络之前是如何转换为单词向量的。本章将描述如何将文本转换为语料库，并从语料库生成单词向量，这样就可以使用欧几里得距离计算或余弦距离计算等技术轻松地对相似单词进行分组。

# 获取数据

第一步是获取一些要处理的数据。在本章中，我们需要大量的文本数据，将其转换为标记并进行可视化，以了解神经网络如何根据欧几里得距离和余弦距离对单词向量进行排名。这是了解不同单词如何相互关联的重要步骤。反过来，这可以用于设计更好、更高效的语言和文本处理模型。

# 准备工作

考虑以下内容：

+   模型的文本数据需要以`.txt`格式的文件存在，并且您必须确保文件放置在当前工作目录中。文本数据可以是来自 Twitter 动态、新闻动态、客户评论、计算机代码或以`.txt`格式保存在工作目录中的整本书。在我们的案例中，我们已经将《权力的游戏》书籍作为模型的输入文本。然而，任何文本都可以替换书籍，并且相同的模型也会起作用。

+   许多经典文本已不再受版权保护。这意味着您可以免费下载这些书籍的所有文本，并将它们用于实验，比如创建生成模型。获取不再受版权保护的免费书籍的最佳途径是 Project Gutenberg（[`www.gutenberg.org/`](https://www.gutenberg.org/)）。

# 操作方法...

步骤如下：

1.  首先访问 Project Gutenberg 网站并浏览您感兴趣的书籍。单击书籍，然后单击 UTF-8，这样您就可以以纯文本格式下载书籍。链接如下截图所示：

![](img/00318.jpeg)

Project Gutenberg 数据集下载页面

1.  点击纯文本 UTF-8 后，您应该会看到一个类似以下截图的页面。右键单击页面，然后单击“另存为...”接下来，将文件重命名为您选择的任何名称，并保存在您的工作目录中：

![](img/00319.jpeg)

1.  现在，您应该在当前工作目录中看到一个带有指定文件名的`.txt`文件。

1.  Project Gutenberg 为每本书添加了标准的页眉和页脚；这不是原始文本的一部分。在文本编辑器中打开文件，然后删除页眉和页脚。

# 工作原理...

功能如下：

1.  使用以下命令检查当前工作目录：`pwd`。

1.  可以使用`cd`命令更改工作目录，如下截图所示：

![](img/00320.jpeg)

1.  请注意，在我们的案例中，文本文件包含在名为`USF`的文件夹中，因此这被设置为工作目录。您可以类似地将一个或多个`.txt`文件存储在工作目录中，以便作为模型的输入。

1.  UTF-8 指定了文本文件中字符的编码类型。**UTF-8**代表**Unicode 转换格式**。**8**表示它使用**8 位**块来表示一个字符。

1.  UTF-8 是一种折衷的字符编码，可以像 ASCII 一样紧凑（如果文件只是纯英文文本），但也可以包含任何 Unicode 字符（文件大小会略有增加）。

1.  不需要文本文件以 UTF-8 格式，因为我们将在稍后阶段使用 codecs 库将所有文本编码为 Latin1 编码格式。

# 还有更多...

有关 UTF-8 和 Latin1 编码格式的更多信息，请访问以下链接：

+   [`en.wikipedia.org/wiki/UTF-8`](https://en.wikipedia.org/wiki/UTF-8)

+   [`www.ic.unicamp.br/~stolfi/EXPORT/www/ISO-8859-1-Encoding.html`](http://www.ic.unicamp.br/~stolfi/EXPORT/www/ISO-8859-1-Encoding.html)

# 另请参阅

访问以下链接以更好地了解神经网络中单词向量的需求：

[`medium.com/deep-math-machine-learning-ai/chapter-9-1-nlp-word-vectors-d51bff9628c1`](https://medium.com/deep-math-machine-learning-ai/chapter-9-1-nlp-word-vectors-d51bff9628c1)

以下是与将单词转换为向量相关的一些其他有用文章：

[`monkeylearn.com/blog/word-embeddings-transform-text-numbers/`](https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/)

[`towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817`](https://towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817)

# 导入必要的库

在开始之前，我们需要导入以下库和依赖项，这些库需要导入到我们的 Python 环境中。这些库将使我们的任务变得更加容易，因为它们具有现成的可用函数和模型，可以代替我们自己进行操作。这也使得代码更加简洁和可读。

# 做好准备

以下库和依赖项将需要创建单词向量和绘图，并在 2D 空间中可视化 n 维单词向量：

+   `未来`

+   `codecs`

+   `glob`

+   ``multiprocessing``

+   `os`

+   ``pprint``

+   `re`

+   `nltk`

+   `Word2Vec`

+   `sklearn`

+   `numpy`

+   `matplotlib`

+   `pandas`

+   `seaborn`

# 如何做...

步骤如下：

1.  在 Jupyter 笔记本中键入以下命令以导入所有所需的库： 

```scala
from __future__ import absolute_import, division, print_function
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy
as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%pylab inline
```

1.  您应该看到一个类似以下截图的输出：

![](img/00321.jpeg)

1.  接下来，使用以下命令导入`stopwords`和`punkt`库：

```scala
nltk.download("punkt")
nltk.download("stopwords")
```

1.  您看到的输出必须看起来像以下截图：

![](img/00322.jpeg)

# 它是如何工作的...

本节将描述用于此配方的每个库的目的。

1.  `future`库是 Python 2 和 Python 3 之间的缺失链接。它充当两个版本之间的桥梁，并允许我们使用两个版本的语法。

1.  `codecs`库将用于对文本文件中所有单词进行编码。这构成了我们的数据集。

1.  Regex 是用于快速查找文件的库。`glob`函数允许快速高效地在大型数据库中搜索所需的文件。

1.  `multiprocessing`库允许我们执行并发，这是一种运行多个线程并使每个线程运行不同进程的方式。这是一种通过并行化使程序运行更快的方式。

1.  `os`库允许与操作系统进行简单交互，如 Mac、Windows 等，并执行诸如读取文件之类的功能。

1.  `pprint`库提供了一种能够以可用作解释器输入的形式对任意 Python 数据结构进行漂亮打印的功能。

1.  `re`模块提供了类似于 Perl 中的正则表达式匹配操作。

1.  NLTK 是一个自然语言工具包，能够在非常简短的代码中对单词进行标记。当输入整个句子时，`nltk`函数会分解句子并输出每个单词的标记。基于这些标记，单词可以被组织成不同的类别。NLTK 通过将每个单词与一个名为**词汇表**的巨大预训练单词数据库进行比较来实现这一点。

1.  `Word2Vec`是 Google 的模型，它在一个巨大的单词向量数据集上进行了训练。它将语义上相似的单词归为一类。这将是本节中最重要的库。

1.  `sklearn.manifold`允许使用**t-分布随机邻居嵌入**（**t-SNE**）技术对数据集进行降维。由于每个单词向量是多维的，我们需要某种形式的降维技术，将这些单词的维度降低到一个较低的维度空间，以便在 2D 空间中进行可视化。

# 还有更多...

`Numpy`是常用的`math`库。`Matplotlib`是我们将利用的`plotting`库，而`pandas`通过允许轻松重塑、切片、索引、子集和操纵数据，提供了很大的灵活性。

`Seaborn`库是另一个统计数据可视化库，我们需要与`matplotlib`一起使用。`Punkt`和`Stopwords`是两个数据处理库，简化了诸如将语料库中的文本拆分为标记（即通过标记化）和删除`stopwords`等任务。

# 另请参阅

有关使用的一些库的更多信息，请访问以下链接：

+   [`docs.python.org/3/library/codecs.html`](https://docs.python.org/3/library/codecs.html)

+   [`docs.python.org/2/library/pprint.html`](https://docs.python.org/2/library/pprint.html)

+   [`docs.python.org/3/library/re.html`](https://docs.python.org/3/library/re.html)

+   [`www.nltk.org/`](https://www.nltk.org/)

+   [`www.tensorflow.org/tutorials/word2vec`](https://www.tensorflow.org/tutorials/word2vec)

+   [`scikit-learn.org/stable/modules/manifold.html`](http://scikit-learn.org/stable/modules/manifold.html)

# 准备数据

在将数据馈送到模型之前，需要执行一些数据预处理步骤。本节将描述如何清理数据并准备数据，以便将其馈送到模型中。

# 准备就绪

首先将所有`.txt`文件中的文本转换为一个大的语料库。这是通过从每个文件中读取每个句子并将其添加到一个空语料库中来完成的。然后执行一些预处理步骤，以删除诸如空格、拼写错误、`stopwords`等不规则性。然后必须对清理后的文本数据进行标记化，并通过循环将标记化的句子添加到一个空数组中。

# 如何做...

步骤如下：

1.  键入以下命令以在工作目录中搜索`.txt`文件并打印找到的文件的名称：

```scala
book_names = sorted(glob.glob("./*.txt"))
print("Found books:")
book_names
```

在我们的案例中，工作目录中保存了五本名为`got1`、`got2`、`got3`、`got4`和`got5`的书籍。

1.  创建一个`corpus`，读取每个句子，从第一个文件开始，对其进行编码，并使用以下命令将编码字符添加到`corpus`中：

```scala
corpus = u''
for book_name in book_names:
print("Reading '{0}'...".format(book_name))
with codecs.open(book_name,"r","Latin1") as book_file:
corpus += book_file.read()
print("Corpus is now {0} characters long".format(len(corpus)))
print()
```

1.  执行前面步骤中的代码，应该会产生以下截图中的输出：

![](img/00323.jpeg)

1.  使用以下命令从`punkt`加载英语 pickle`tokenizer`：

```scala
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
```

1.  使用以下命令将整个`corpus`标记化为句子：

```scala
raw_sentences = tokenizer.tokenize(corpus)
```

1.  以以下方式定义将句子拆分为其组成单词并删除不必要字符的函数：

```scala
def sentence_to_wordlist(raw):
     clean = re.sub("[^a-zA-Z]"," ", raw)
     words = clean.split()
     return words
```

1.  将每个句子的每个单词标记化的原始句子全部添加到一个新的句子数组中。使用以下代码完成：

```scala
sentences = []
for raw_sentence in raw_sentences:
  if len(raw_sentence) > 0:
  sentences.append(sentence_to_wordlist(raw_sentence))
```

1.  从语料库中打印一个随机句子，以直观地查看`tokenizer`如何拆分句子并从结果创建一个单词列表。使用以下命令完成：

```scala
print(raw_sentences[50])
print(sentence_to_wordlist(raw_sentences[50]))
```

1.  使用以下命令计算数据集中的总标记数：

```scala
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))
```

# 工作原理...

执行分词器并对语料库中的所有句子进行分词应该会产生以下截图中的输出：

![](img/00324.jpeg)

接下来，删除不必要的字符，如连字符和特殊字符，是以以下方式完成的。使用用户定义的`sentence_to_wordlist()`函数拆分所有句子会产生以下截图中显示的输出：

![](img/00325.jpeg)

将原始句子添加到名为`sentences[]`的新数组中，将产生如下截图所示的输出：

![](img/00326.jpeg)

在打印语料库中的总标记数时，我们注意到整个语料库中有 1,110,288 个标记。这在以下截图中有所说明：

![](img/00327.jpeg)

功能如下：

1.  使用 NLTK 中的预训练`tokenizer`通过将每个句子计为一个标记来标记整个语料库。每个标记化的句子都被添加到变量`raw_sentences`中，该变量存储了标记化的句子。

1.  接下来，常见的停用词被移除，并且通过将每个句子分割成单词来清理文本。

1.  打印一个随机句子以及其单词列表，以了解其工作原理。在我们的案例中，我们选择打印`raw_sentences`数组中的第 50 个句子。

1.  计算并打印句子数组中的总标记数（在我们的案例中是句子）。在我们的案例中，我们看到`tokenizer`创建了 1,110,288 个标记。

# 还有...

有关将段落和句子标记化的更多信息，请访问以下链接：

+   [`textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize`](https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize)

+   [`stackoverflow.com/questions/37605710/tokenize-a-paragraph-into-sentence-and-then-into-words-in-nltk`](https://stackoverflow.com/questions/37605710/tokenize-a-paragraph-into-sentence-and-then-into-words-in-nltk)

+   [`pythonspot.com/tokenizing-words-and-sentences-with-nltk/`](https://pythonspot.com/tokenizing-words-and-sentences-with-nltk/)

# 另请参阅

有关正则表达式工作原理的更多信息，请访问以下链接：

[`stackoverflow.com/questions/13090806/clean-line-of-punctuation-and-split-into-words-python`](https://stackoverflow.com/questions/13090806/clean-line-of-punctuation-and-split-into-words-python)

# 构建和训练模型

一旦我们将文本数据以数组形式的标记输入到模型中，我们就能够为模型定义一些超参数。本节将描述如何执行以下操作：

+   声明模型超参数

+   使用`Word2Vec`构建模型

+   在准备好的数据集上训练模型

+   保存和检查点训练好的模型

# 准备工作

需要声明的一些模型超参数包括以下内容：

+   生成的单词向量的维度

+   最小词数阈值

+   在训练模型时运行的并行线程数

+   上下文窗口长度

+   降采样（对于频繁出现的单词）

+   设置种子

一旦前面提到的超参数被声明，就可以使用`Gensim`库中的`Word2Vec`函数构建模型。

# 如何做...

步骤如下：

1.  使用以下命令声明模型的超参数：

```scala
num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1
```

1.  使用声明的超参数，使用以下代码行构建模型：

```scala
got2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
```

1.  使用标记化的句子构建模型的词汇表，并通过所有标记进行迭代。这是使用以下方式的`build_vocab`函数完成的：

```scala
got2vec.build_vocab(sentences,progress_per=10000, keep_raw_vocab=False, trim_rule=None)
```

1.  使用以下命令训练模型：

```scala
got2vec.train(sentences, total_examples=got2vec.corpus_count, total_words=None, epochs=got2vec.iter, start_alpha=None, end_alpha=None, word_count=0, queue_factor=2, report_delay=1.0, compute_loss=False)
```

1.  如果尚不存在，请创建一个名为 trained 的目录。使用以下命令保存和检查点`trained`模型：

```scala
if not os.path.exists("trained"):
     os.makedirs("trained")
got2vec.wv.save(os.path.join("trained", "got2vec.w2v"), ignore=[])
```

1.  要在任何时候加载保存的模型，请使用以下命令：

```scala
got2vec = w2v.KeyedVectors.load(os.path.join("trained", "got2vec.w2v"))
```

# 它是如何工作的...

功能如下：

1.  模型参数的声明不会产生任何输出。它只是在内存中留出空间来存储变量作为模型参数。以下截图描述了这个过程：![](img/00328.jpeg)

1.  模型是使用前述超参数构建的。在我们的案例中，我们将模型命名为`got2vec`，但模型可以根据您的喜好进行命名。模型定义如下截图所示：![](img/00329.jpeg)

1.  在模型上运行`build_vocab`命令应该会产生如下截图所示的输出：![](img/00330.jpeg)

1.  通过定义以下截图中所见的参数来训练模型：![](img/00331.jpeg)

1.  上述命令产生如下截图所示的输出：

![](img/00332.jpeg)

1.  保存、检查点和加载模型的命令产生如下输出，如截图所示：

![](img/00333.jpeg)

# 还有更多...

考虑以下内容：

+   在我们的案例中，我们注意到`build_vocab`函数从 1,110,288 个单词的列表中识别出 23,960 个不同的单词类型。然而，对于不同的文本语料库，这个数字会有所不同。

+   每个单词都由一个 300 维向量表示，因为我们已经声明维度为 300。增加这个数字会增加模型的训练时间，但也会确保模型很容易地泛化到新数据。

+   发现 1e![](img/00334.jpeg)3 的下采样率是一个很好的比率。这是为了让模型知道何时对频繁出现的单词进行下采样，因为它们在分析时并不重要。这些单词的例子包括 this, that, those, them 等。

+   设置一个种子以使结果可重现。设置种子也使得调试变得更容易。

+   由于模型不是很复杂，使用常规 CPU 计算训练模型大约需要 30 秒。

+   当检查点被检查时，模型被保存在工作目录内的`trained`文件夹下。

# 另请参阅

有关`Word2Vec`模型和 Gensim 库的更多信息，请访问以下链接：[`radimrehurek.com/gensim/models/word2vec.html`](https://radimrehurek.com/gensim/models/word2vec.html)

[`radimrehurek.com/gensim/models/word2vec.html`](https://radimrehurek.com/gensim/models/word2vec.html)

# 进一步可视化

本节将描述如何压缩所有训练过的单词的维度，并将其全部放入一个巨大的矩阵以进行可视化。由于每个单词都是一个 300 维的向量，所以需要将其降低到更低的维度，以便我们在 2D 空间中进行可视化。

# 准备工作

一旦模型在训练后保存和检查点后，开始将其加载到内存中，就像在上一节中所做的那样。在本节中将使用的库和模块有：

+   `tSNE`

+   `pandas`

+   `Seaborn`

+   `numpy`

# 如何做...

步骤如下：

1.  使用以下命令压缩 300 维单词向量的维度：

```scala
 tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
```

1.  将所有单词向量放入一个巨大的矩阵（命名为`all_word_vectors_matrix`），并使用以下命令查看它：

```scala
 all_word_vectors_matrix = got2vec.wv.syn0
 print (all_word_vectors_matrix)
```

1.  使用以下命令将所有学习到的表示拟合到二维空间中：

```scala
 all_word_vectors_matrix_2d =  tsne.fit_transform(all_word_vectors_matrix)
```

1.  使用以下代码收集所有单词向量及其相关单词：

```scala
 points = pd.DataFrame(
     [
            (word, coords[0], coords[1])
             for word, coords in [
              (word, all_word_vectors_matrix_2d[got2vec.vocab[word].index])
                for word in got2vec.vocab
         ]
    ],
    columns=["word", "x", "y"]
)
```

1.  使用以下命令可以获取前十个点的`X`和`Y`坐标以及相关单词：

```scala
points.head(10)
```

1.  使用以下命令绘制所有点：

```scala
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(15, 15))
```

1.  可以放大绘图图表的选定区域以进行更仔细的检查。通过使用以下函数对原始数据进行切片来实现这一点：

```scala
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
        ]
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word,                                                  fontsize=11)
```

1.  使用以下命令绘制切片数据。切片数据可以被视为原始所有数据点的放大区域：

```scala
plot_region(x_bounds=(20.0, 25.0), y_bounds=(15.5, 20.0))
```

# 工作原理...

功能如下：

1.  t-SNE 算法是一种非线性降维技术。计算机在计算过程中很容易解释和处理许多维度。然而，人类一次只能可视化两到三个维度。因此，当试图从数据中得出见解时，这些降维技术非常有用。

1.  将 300 维向量应用 t-SNE 后，我们能够将其压缩为只有两个维度来绘制和查看。

1.  通过将 `n_components` 指定为 2，我们让算法知道它必须将数据压缩到二维空间。完成此操作后，我们将所有压缩后的向量添加到一个名为 `all_word_vectors_matrix` 的巨大矩阵中，如下图所示：![](img/00335.jpeg)

1.  t-SNE 算法需要对所有这些单词向量进行训练。在常规 CPU 上，训练大约需要五分钟。

1.  一旦 t-SNE 完成对所有单词向量的训练，它会为每个单词输出 2D 向量。可以通过将它们全部转换为数据框架来将这些向量绘制为点。如下图所示完成此操作：![](img/00336.jpeg)

1.  我们看到上述代码生成了许多点，其中每个点代表一个单词及其 X 和 Y 坐标。检查数据框架的前二十个点时，我们看到如下图所示的输出：

![](img/00337.jpeg)

1.  通过使用 `all_word_vectors_2D` 变量绘制所有点，您应该会看到类似以下截图的输出：![](img/00338.jpeg)

1.  上述命令将生成从整个文本生成的所有标记或单词的绘图，如下图所示：![](img/00339.jpeg)

1.  我们可以使用 `plot_region` 函数来放大绘图中的某个区域，以便我们能够实际看到单词及其坐标。这一步骤如下图所示：![](img/00340.jpeg)

1.  通过设置 `x_bounds` 和 `y_bounds` 的值，可以可视化绘图的放大区域，如下图所示：![](img/00341.jpeg)

1.  可以通过改变 `x_bounds` 和 `y_bounds` 的值来可视化相同绘图的不同区域，如下两个截图所示：![](img/00342.jpeg)

![](img/00343.jpeg)

# 另请参阅

还有以下额外的要点：

+   有关 t-SNE 算法工作原理的更多信息，请访问以下链接：

+   [`www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm`](https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm)

+   有关余弦距离相似性和排名的更多信息，请访问以下链接：

[`code.google.com/archive/p/word2vec/`](https://code.google.com/archive/p/word2vec/)

+   使用以下链接来探索 `Seaborn` 库的不同功能：

[`seaborn.pydata.org/`](https://seaborn.pydata.org/)

# 进一步分析

本节将描述可在可视化后对数据执行的进一步分析。例如，探索不同单词向量之间的余弦距离相似性。

# 准备工作

以下链接是关于余弦距离相似性工作原理的出色博客，并讨论了一些涉及的数学内容：

[`blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/`](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)

# 如何做...

考虑以下内容：

+   可以使用 `Word2Vec` 的不同功能执行各种自然语言处理任务。其中之一是在给定某个单词时找到最语义相似的单词（即具有高余弦相似性或它们之间的欧几里德距离较短的单词向量）。可以使用 `Word2Vec` 的 `most_similar` 函数来执行此操作，如下图所示：![](img/00344.jpeg)此截图显示了与单词 `Lannister` 相关的所有最接近的单词：

![](img/00345.jpeg)此截图显示了与单词 `Jon` 相关的所有单词的列表：![](img/00346.jpeg)

# 工作原理...

考虑以下内容：

+   有各种方法来衡量单词之间的语义相似性。我们在本节中使用的方法是基于余弦相似性的。我们还可以通过以下代码来探索单词之间的线性关系：

```scala
 def nearest_similarity_cosmul(start1, end1, end2):
    similarities = got2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
)
start2 = similarities[0][0]
print("{start1} is related to {end1}, as {start2} is related to         {end2}".format(**locals()))
return start2
```

+   要找到给定一组词的最近词的余弦相似度，请使用以下命令：

```scala
nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")
```

+   上述过程如下截图所示：![](img/00347.jpeg)

+   结果如下：![](img/00348.jpeg)

+   如本节所示，词向量是所有自然语言处理任务的基础。在深入研究更复杂的自然语言处理模型（如循环神经网络和长短期记忆（LSTM）单元）之前，了解它们以及构建这些模型所涉及的数学是很重要的。

# 另请参阅

可以进一步阅读有关使用余弦距离相似性、聚类和其他机器学习技术在排名词向量中的应用的内容，以更好地理解。以下是一些有用的关于这个主题的已发表论文的链接：

+   [`s3.amazonaws.com/academia.edu.documents/32952068/pg049_Similarity_Measures_for_Text_Document_Clustering.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1530163881&Signature=YG6YjvJb2z0JjmfHzaYujA2ioIo%3D&response-content-disposition=inline%3B%20filename%3DSimilarity_Measures_for_Text_Document_Cl.pdf`](https://s3.amazonaws.com/academia.edu.documents/32952068/pg049_Similarity_Measures_for_Text_Document_Clustering.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1530163881&Signature=YG6YjvJb2z0JjmfHzaYujA2ioIo%3D&response-content-disposition=inline%3B%20filename%3DSimilarity_Measures_for_Text_Document_Cl.pdf)

+   [`csis.pace.edu/ctappert/dps/d861-12/session4-p2.pdf`](http://csis.pace.edu/ctappert/dps/d861-12/session4-p2.pdf)
