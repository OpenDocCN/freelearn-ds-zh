# 第六章：分析学习：使用 TensorFlow 进行 AI 和图像识别

|   | *“人工智能、深度学习、机器学习——无论你在做什么，如果你不理解它——就学习它。因为不然，你将在 3 年内变成恐龙。”* |   |
| --- | --- | --- |
|   | --*马克·库班* |

这是一个涵盖流行行业用例的系列示例应用中的第一章，毫无疑问，我从一个与机器学习，特别是通过图像识别示例应用的深度学习相关的用例开始。这几年，**人工智能**（**AI**）领域经历了加速增长，许多实际应用已成为现实，比如自动驾驶汽车，具备高级自动语音识别的聊天机器人，这些技术在某些任务中完全能够替代人工操作员，而越来越多的人，无论是学术界还是产业界，开始参与其中。然而，仍然存在一种看法，认为进入的门槛很高，并且掌握机器学习背后的数学概念是前提条件。在本章中，我们尝试通过示例来演示，事实并非如此。

我们将以简要介绍机器学习开始本章，并介绍其一个子集——深度学习。接着我们将介绍一个非常流行的深度学习框架——TensorFlow，我们将利用它来构建一个图像识别模型。在本章的第二部分，我们将展示如何通过实现一个名为 PixieApp 的示例应用来将我们构建的模型投入实际使用，用户可以输入一个网站链接，获取该网站的所有图片，并将其作为输入传递给模型进行分类。

在本章结束时，你应该确信，即使没有机器学习博士学位，也完全可以构建有意义的应用并将其投入实际使用。

# 什么是机器学习？

我认为很好地捕捉到机器学习直觉的一个定义来自斯坦福大学的副教授 Andrew Ng，在他的 Coursera 课程*机器学习*中提到（[`www.coursera.org/learn/machine-learning`](https://www.coursera.org/learn/machine-learning)）：

> 机器学习是让计算机通过学习来完成任务，而不是通过显式编程。

上述定义中的关键词是*学习*，在此上下文中，*学习*的含义与我们人类的学习方式非常相似。继续这一类比，从小开始，我们就被教导如何通过示范或者通过自身的试错过程完成一项任务。广义来说，机器学习算法可以分为两种类型，这两种类型对应于人类学习的两种方式：

+   **监督学习**：算法从已正确标注的示例数据中学习。这些数据也叫做训练数据，或者有时被称为*地面真实*。

+   **无监督学习**：算法能够从未标记的数据中自行学习。

下面的表格概述了每个类别中最常用的机器学习算法及其解决的问题类型：

![什么是机器学习？](img/B09699_06_01.jpg)

机器学习算法列表

这些算法的输出被称为**模型**，并用于对从未见过的新输入数据进行预测。构建和部署这些模型的整个端到端过程在不同类型的算法中是非常一致的。

下图展示了这个过程的高层次工作流：

![什么是机器学习？](img/B09699_06_02.jpg)

机器学习模型工作流

和往常一样，工作流从数据开始。在监督学习的情况下，数据将作为示例使用，因此必须正确标记答案。然后，输入数据被处理以提取内在特性，称为**特征**，我们可以将它们看作是代表输入数据的数值。随后，这些特征被输入到一个机器学习算法中，构建出一个模型。在典型设置中，原始数据会被拆分为训练数据、测试数据和盲数据。在模型构建阶段，测试数据和盲数据用于验证和优化模型，以确保模型不会过度拟合训练数据。过度拟合发生在模型参数过于紧密地跟随训练数据，导致在使用未见过的数据时出现错误。当模型达到预期的准确度时，它会被部署到生产环境中，并根据宿主应用的需求对新数据进行预测。

在本节中，我们将提供一个非常高层次的机器学习介绍，配以简化的数据流水线工作流，足以让你理解模型是如何构建和部署的。如果你是初学者，我强烈推荐 Andrew Ng 在 Coursera 上的*机器学习*课程（我自己也时常回顾）。在接下来的部分，我们将介绍机器学习的一个分支——深度学习，我们将用它来构建图像识别示例应用。

# 什么是深度学习？

让计算机学习、推理和思考（做决策）是一门被称为**认知计算**的科学，其中机器学习和深度学习是重要组成部分。下图展示了这些领域如何与 AI 这一广泛领域相关：

![什么是深度学习？](img/B09699_06_03.jpg)

深度学习在 AI 中的位置

正如图示所示，深度学习是机器学习算法的一种类型。或许不为人所广知的是，深度学习领域已经存在相当长的时间，但直到最近才被广泛应用。兴趣的复燃是由于近年来计算机、云计算和存储技术的巨大进步，这些技术推动了人工智能的指数增长，并催生了许多新的深度学习算法，每个算法都特别适合解决特定问题。

正如我们在本章稍后讨论的，深度学习算法特别擅长学习复杂的非线性假设。它们的设计实际上是受到人脑工作方式的启发，例如，输入数据通过多个计算单元层进行处理，以将复杂的模型表示（例如图像）分解为更简单的表示，然后将结果传递到下一层，依此类推，直到到达负责输出结果的最终层。这些层的组合也被称为**神经网络**，构成一层的计算单元被称为**神经元**。本质上，一个神经元负责接收多个输入，并将其转换为单一输出，然后这个输出可以输入到下一层的其他神经元。

以下图示表示了一个用于图像分类的多层神经网络：

![什么是深度学习？](img/B09699_06_04.jpg)

图像分类的神经网络高级表示

上述神经网络也被称为**前馈网络**，因为每个计算单元的输出作为输入传递到下一层，从输入层开始。中间层被称为**隐藏层**，包含由网络自动学习的中间特征。在我们的图像示例中，某些神经元可能负责检测角落，而其他神经元则可能专注于边缘，依此类推。最终的输出层负责为每个输出类别分配一个置信度（得分）。

一个重要的问题是，神经元的输出是如何从输入生成的？在不深入探讨涉及的数学内容的前提下，每个人工神经元会对其输入的加权和应用激活函数 ![什么是深度学习？](img/B09699_06_26.jpg)，以决定它是否应该*激活*。

以下公式计算加权和：

![什么是深度学习？](img/B09699_06_27.jpg)

其中 ![什么是深度学习？](img/B09699_06_28.jpg) 是层 *i* 和 *i + 1* 之间的权重矩阵。这些权重是在稍后讨论的训练阶段中计算得出的。

### 注意

**注意**：前面公式中的偏置表示偏置神经元的权重，它是每一层中添加的一个额外神经元，其 x 值为 +1。偏置神经元很特殊，因为它贡献了下一层的输入，但与上一层没有连接。然而，它的权重仍然像其他神经元一样被正常学习。偏置神经元的直觉是，它为线性回归方程提供了常数项 b：

![什么是深度学习？](img/B09699_06_29.jpg)

当然，应用神经元激活函数 ![什么是深度学习？](img/B09699_06_30.jpg) 在 *A* 上，不能简单地产生一个二进制（0 或 1）值，因为如果多个类别都被赋予了 1 的分数，我们就无法正确地排序最终的候选答案。相反，我们使用提供 0 到 1 之间非离散分数的激活函数，并设置一个阈值（例如 0.5）来决定是否激活神经元。

最常用的激活函数之一是 sigmoid 函数：

![什么是深度学习？](img/B09699_06_31.jpg)

下图展示了如何使用 sigmoid 激活函数根据输入和权重计算神经元的输出：

![什么是深度学习？](img/B09699_06_05.jpg)

使用 sigmoid 函数计算神经元输出

其他常用的激活函数包括双曲正切 ![什么是深度学习？](img/B09699_06_32.jpg) 和 **修正线性单元**（**ReLu**）：![什么是深度学习？](img/B09699_06_33.jpg)。当有很多层时，ReLu 的表现更好，因为它提供了稀疏的*激活*神经元，从而减少噪音并加快学习速度。

前馈传播用于模型评分时，但在训练神经网络的权重矩阵时，一种常用的方法叫做**反向传播**（[`en.wikipedia.org/wiki/Backpropagation`](https://en.wikipedia.org/wiki/Backpropagation)）。

以下高层步骤描述了训练是如何进行的：

1.  随机初始化权重矩阵（最好使用较小的值，例如 ![什么是深度学习？](img/B09699_06_34.jpg)）。

1.  使用之前描述的前向传播方法，对所有训练样本进行计算，使用你选择的激活函数计算每个神经元的输出。

1.  为你的神经网络实现一个成本函数。**成本函数**量化了与训练样本的误差。可以与反向传播算法一起使用的成本函数有多种，例如均方误差（[`en.wikipedia.org/wiki/Mean_squared_error`](https://en.wikipedia.org/wiki/Mean_squared_error)）和交叉熵（[`en.wikipedia.org/wiki/Cross_entropy`](https://en.wikipedia.org/wiki/Cross_entropy)）。

1.  使用反向传播来最小化你的成本函数并计算权重矩阵。反向传播的基本思想是从输出层的激活值开始，计算与训练数据的误差，并将这些误差反向传递到隐藏层。然后，这些误差会被调整，以最小化步骤 3 中实现的成本函数。

### 注意

**注意**：详细解释这些成本函数以及它们如何被优化超出了本书的范围。若想深入了解，我强烈推荐阅读 MIT 出版社的《深度学习》一书（Ian Goodfellow、Yoshua Bengio 和 Aaron Courville）。

在本节中，我们从高层次讨论了神经网络的工作原理以及它们是如何训练的。当然，我们只触及了这项激动人心的技术的皮毛，但希望你应该能大致了解它们的工作方式。在接下来的部分，我们将开始研究 TensorFlow，这是一个帮助抽象实现神经网络底层复杂性的编程框架。

# 开始使用 TensorFlow

除了 TensorFlow ([`www.tensorflow.org`](https://www.tensorflow.org)) 之外，我还可以选择多个开源深度学习框架用于这个示例应用程序。

以下是一些最流行的框架：

+   PyTorch ([`pytorch.org`](http://pytorch.org))

+   Caffee2 ([`caffe2.ai`](https://caffe2.ai))

+   MXNet ([`mxnet.apache.org`](https://mxnet.apache.org))

+   Keras ([`keras.io`](https://keras.io))：一个高级神经网络抽象 API，能够运行其他深度学习框架，如 TensorFlow、CNTK ([`github.com/Microsoft/cntk`](https://github.com/Microsoft/cntk)) 和 Theano ([`github.com/Theano/Theano`](https://github.com/Theano/Theano))

TensorFlow API 支持多种语言：Python、C++、Java、Go，最近还包括 JavaScript。我们可以将 API 分为两类：高级和低级，具体如下图所示：

![开始使用 TensorFlow](img/B09699_06_06.jpg)

TensorFlow 高级 API 架构

为了开始使用 TensorFlow API，让我们构建一个简单的神经网络，学习 XOR 转换。

提醒一下，XOR 运算符只有四个训练样本：

| **X** | **Y** | **结果** |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

有趣的是，线性分类器 ([`en.wikipedia.org/wiki/Linear_classifier`](https://en.wikipedia.org/wiki/Linear_classifier)) 无法学习 XOR 转换。然而，我们可以通过一个简单的神经网络来解决这个问题，该网络有两个输入层神经元、一个隐藏层（包含两个神经元）和一个输出层（包含一个神经元，进行二分类），如下所示：

![开始使用 TensorFlow](img/B09699_06_07.jpg)

XOR 神经网络

### 注意

**注意**：你可以通过以下命令直接从 Notebook 安装 TensorFlow：

```py
!pip install tensorflow

```

像往常一样，在成功安装任何内容后，别忘了重启内核。

为了创建输入层和输出层的张量，我们使用`tf.placeholder` API，如下代码所示：

```py
import tensorflow as tf
x_input = tf.placeholder(tf.float32)
y_output = tf.placeholder(tf.float32)
```

然后，我们使用`tf.Variable` API ([`www.tensorflow.org/programmers_guide/variables`](https://www.tensorflow.org/programmers_guide/variables)) 初始化矩阵的随机值！TensorFlow 入门 和 ![TensorFlow 入门](img/B09699_new_02.jpg)，分别对应隐藏层和输出层：

```py
eps = 0.01
W1 = tf.Variable(tf.random_uniform([2,2], -eps, eps))
W2 = tf.Variable(tf.random_uniform([2,1], -eps, eps))
```

对于激活函数，我们使用 sigmoid 函数：

### 注意

**注意**：为了简化，我们省略了偏置的介绍。

```py
layer1 = tf.sigmoid(tf.matmul(x_input, W1))
output_layer = tf.sigmoid(tf.matmul(layer1, W2))
```

对于损失函数，我们使用**MSE**（即**均方误差**）：

```py
cost = tf.reduce_mean(tf.square(y_output - output_layer))
```

在图中的所有张量就位后，我们可以使用`tf.train.GradientDescentOptimizer`，学习率为`0.05`，来最小化我们的损失函数，开始训练：

```py
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
training_data = ([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        sess.run(train,
            feed_dict={x_input: training_data[0], y_output: training_data[1]})
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode1.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode1.py)

上述代码首次引入了 TensorFlow `Session` 的概念，这是框架的基础部分。实际上，任何 TensorFlow 操作必须在`Session`的上下文中执行，使用其`run`方法。会话还维护需要显式释放的资源，通过`close`方法来释放。为了方便，`Session`类通过提供`__enter__`和`__exit__`方法支持上下文管理协议。这允许调用者使用`with`语句 ([`docs.python.org/3/whatsnew/2.6.html#pep-343-the-with-statement`](https://docs.python.org/3/whatsnew/2.6.html#pep-343-the-with-statement)) 来调用 TensorFlow 操作，并自动释放资源。

以下伪代码展示了一个典型的 TensorFlow 执行结构：

```py
with tf.Session() as sess:
    with-block statement with TensorFlow operations
```

在本节中，我们快速探讨了低级 TensorFlow API，构建了一个简单的神经网络，学习了 XOR 转换。在下一节中，我们将探讨提供高级抽象层的更高层次的估计器 API。

## 使用 DNNClassifier 进行简单的分类

### 注意

**注意**：本节讨论了一个示例 PixieApp 的源代码。如果你想跟着操作，可能更容易直接下载完整的 Notebook 文件，位于这个位置：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/TensorFlow%20classification.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/TensorFlow%20classification.ipynb)

在我们开始使用低级 TensorFlow API 中的张量、图和会话之前，先熟悉一下 `Estimators` 包提供的高级 API 是很有帮助的。在这一部分，我们构建了一个简单的 PixieApp，它接受 pandas DataFrame 作为输入，并训练一个具有类别输出的分类模型。

### 注意

**注意**：分类输出基本上有两种类型：类别型和连续型。在类别型分类模型中，输出只能从有限的预定义值列表中选择，且可能有或没有逻辑顺序。我们通常称二分类为只有两个类别的分类模型。另一方面，连续输出可以有任何数值。

用户首先需要选择一个数值列进行预测，然后使用数据框中所有其他数值列训练一个分类模型。

### 注意

**注意**：这个示例应用的一些代码改编自 [`github.com/tensorflow/models/tree/master/samples/core/get_started`](https://github.com/tensorflow/models/tree/master/samples/core/get_started)。

对于这个示例，我们将使用内置的示例数据集 #7：波士顿犯罪数据，两周的样本数据，但你也可以使用任何其他数据集，只要它有足够的数据和数值列。

提醒一下，你可以使用以下代码浏览 PixieDust 内置的数据集：

```py
import pixiedust
pixiedust.sampleData()
```

![使用 DNNClassifier 进行简单分类](img/B09699_06_08.jpg)

PixieDust 中的内置数据集列表

以下代码使用 `sampleData()` API 加载 *波士顿犯罪* 数据集：

```py
import pixiedust
crimes = pixiedust.sampleData(7, forcePandas=True)
```

和往常一样，我们首先通过 `display()` 命令探索数据。这里的目标是寻找一个合适的列进行预测：

```py
display(crimes)
```

![使用 DNNClassifier 进行简单分类](img/B09699_06_09.jpg)

犯罪数据集的表格视图

看起来 `nonviolent` 是一个适合二分类的良好候选项。现在让我们展示一个条形图，以确保该列的数据分布良好：

![使用 DNNClassifier 进行简单分类](img/B09699_06_10.jpg)

在选项对话框中选择非暴力列

点击 **OK** 会生成以下图表：

![使用 DNNClassifier 进行简单分类](img/B09699_06_11.jpg)

非暴力犯罪分布

不幸的是，数据倾向于非暴力犯罪，但我们有接近 2,000 个暴力犯罪的数据点，对于这个示例应用程序来说，应该足够了。

我们现在准备创建 `do_training` 方法，使用 `tf.estimator.DNNClassifier` 创建一个分类模型。

### 注意

**注意**：你可以在这里找到更多关于 `DNNClassifier` 和其他高级 TensorFlow 估算器的信息：

[`www.tensorflow.org/api_docs/python/tf/estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator)

`DNNClassifier` 构造函数有很多可选参数。在我们的示例应用中，我们只会使用其中三个，但我鼓励你查看文档中的其他参数：

+   `feature_columns`：`feature_column._FeatureColumn`模型输入的可迭代对象。在我们的例子中，我们可以使用 Python 推导式仅通过 pandas DataFrame 的数值列创建一个数组。

+   `hidden_units`：每个单元隐藏层数的可迭代对象。在这里，我们只使用两个层，每个层有 10 个节点。

+   `n_classes`：标签类别的数量。我们将通过对预测列进行分组并计算行数来推断此数字。

这是`do_training`方法的代码：

```py
def do_training(train, train_labels, test, test_labels, num_classes):
    #set TensorFlow logging level to INFO
    tf.logging.set_verbosity(tf.logging.INFO)

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        # Compute feature_columns from dataframe keys using a list comprehension
        feature_columns =
            [tf.feature_column.numeric_column(key=key) for key in train.keys()],
        hidden_units=[10, 10],
        n_classes=num_classes)

    # Train the Model
    classifier.train(
        input_fn=lambda:train_input_fn(train, train_labels,100),
        steps=1000
    )

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test, test_labels,100)
    )

    return (classifier, eval_result)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode2.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode2.py)

`classifier.train`方法使用一个`train_input_fn`方法，负责以小批量的形式提供训练输入数据（即真实标签），返回一个`tf.data.Dataset`或`(features, labels)`元组。我们的代码还通过`classifier.evaluate`进行模型评估，通过对测试数据集进行评分并将结果与给定标签进行比较来验证准确性。结果随后作为函数输出的一部分返回。

此方法需要一个与`train_input_fn`类似的`eval_input_fn`方法，唯一的区别是在评估过程中我们不使数据集可重复。由于这两个方法共享大部分相同的代码，我们使用一个名为`input_fn`的辅助方法，该方法由两个方法调用，并带有适当的标志：

```py
def input_fn(features, labels, batch_size, train):
    # Convert the inputs to a Dataset and shuffle.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)).shuffle(1000)
    if train:
        #repeat only for training
 dataset = dataset.repeat()
    # Return the dataset in batch
    return dataset.batch(batch_size)

def train_input_fn(features, labels, batch_size):
    return input_fn(features, labels, batch_size, train=True)

def eval_input_fn(features, labels, batch_size):
    return input_fn(features, labels, batch_size, train=False)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode3.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode3.py)

下一步是构建 PixieApp，它将从传递给`run`方法的 pandas DataFrame 创建分类器。主屏幕构建了所有数值列的下拉列表，并要求用户选择一个将用作分类器输出的列。这是通过以下代码完成的，使用 Jinja2 `{%for ...%}` 循环遍历作为输入传递的 DataFrame，DataFrame 通过`pixieapp_entity`变量引用。

### 注意

**注意**：以下代码使用`[[SimpleClassificationDNN]]`符号表示它是指定类的不完整代码。请勿尝试运行此代码，直到提供完整实现为止。

```py
[[SimpleClassificationDNN]]
from pixiedust.display.app import *
@PixieApp
class SimpleClassificationDNN():
    @route()
    def main_screen(self):
        return """
<h1 style="margin:40px">
    <center>The classificiation model will be trained on all the numeric columns of the dataset</center>
</h1>
<style>
    div.outer-wrapper {
        display: table;width:100%;height:300px;
    }
    div.inner-wrapper {
        display: table-cell;vertical-align: middle;height: 100%;width: 100%;
    }
</style>
<div class="outer-wrapper">
    <div class="inner-wrapper">
        <div class="col-sm-3"></div>
        <div class="input-group col-sm-6">
          <select id="cols{{prefix}}" style="width:100%;height:30px" pd_options="predictor=$val(cols{{prefix}})">
              <option value="0">Select a predictor column</option>
              {%for col in this.pixieapp_entity.columns.values.tolist()%}
 <option value="{{col}}">{{col}}</option>
 {%endfor%}
          </select>
        </div>
    </div>
</div>     
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode4.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode4.py)

使用`crimes`数据集，我们通过以下代码运行 PixieApp：

```py
app = SimpleClassificationDNN()
app.run(crimes)
```

### 注意

**注意**：此时 PixieApp 代码尚不完整，但我们仍然可以看到欢迎页面的结果，如下图所示：

![使用 DNNClassifier 的简单分类](img/B09699_06_12.jpg)

显示输入 pandas DataFrame 列表的主屏幕

当用户选择预测列（例如 `nonviolent`）时，通过属性 `pd_options="predictor=$val(cols{{prefix}})"` 会触发一个新的 `prepare_training` 路由。该路由将显示两个条形图，分别显示训练集和测试集的输出类别分布，这些数据是通过从原始数据集中以 80/20 的比例随机选取得到的。

### 注意

**注意**：我们在训练集和测试集之间使用 80/20 的分割比例，从我的经验来看，这种做法很常见。当然，这不是绝对规则，根据具体情况可以进行调整。

屏幕片段还包括一个按钮，用于启动训练分类器。

`prepare_training` 路由的代码如下所示：

```py
[[SimpleClassificationDNN]]
@route(predictor="*")
@templateArgs
def prepare_training(self, predictor):
        #select only numerical columns
        self.dataset = self.pixieapp_entity.dropna(axis=1).select_dtypes(
            include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        )
        #Compute the number of classed by counting the groups
        self.num_classes = self.dataset.groupby(predictor).size().shape[0]
        #Create the train and test feature and labels
        self.train_x=self.dataset.sample(frac=0.8)
        self.full_train = self.train_x.copy()
        self.train_y = self.train_x.pop(predictor)
        self.test_x=self.dataset.drop(self.train_x.index)
        self.full_test = self.test_x.copy()
        self.test_y=self.test_x.pop(predictor)

        bar_chart_options = {
          "rowCount": "100",
          "keyFields": predictor,
          "handlerId": "barChart",
          "noChartCache": "true"
        }

        return """
<div class="container" style="margin-top:20px">
    <div class="row">
        <div class="col-sm-5">
            <h3><center>Train set class distribution</center></h3>
            <div pd_entity="full_train" pd_render_onload>
                <pd_options>{{bar_chart_options|tojson}}</pd_options>
            </div>
        </div>
        <div class="col-sm-5">
            <h3><center>Test set class distribution</center></h3>
            <div pd_entity="full_test" pd_render_onload>
                <pd_options>{{bar_chart_options|tojson}}</pd_options>
            </div>
        </div>
    </div>
</div>

<div style="text-align:center">
 <button class="btn btn-default" type="submit" pd_options="do_training=true">
 Start Training
 </button>
</div>
"""
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode5.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode5.py)

**注意**：由于我们计算了 `bar_chart_options` 变量一次，并且在 Jinja2 模板中使用它，所以使用了 `@templateArgs`。

选择 `nonviolent` 预测列将给我们以下截图结果：

![使用 DNNClassifier 进行简单分类](img/B09699_06_13.jpg)

预训练屏幕

**开始训练** 按钮通过属性 `pd_options="do_training=true",` 调用 `do_training` 路由，该路由触发我们之前创建的 `do_training` 方法。注意，我们使用了 `@captureOutput` 装饰器，因为我们将 TensorFlow 日志级别设置为 `INFO`，所以我们希望捕获日志消息并将其显示给用户。这些日志消息会通过 *stream* 模式返回到浏览器，PixieDust 会自动将它们显示为专门创建的 `<div>` 元素，并随着数据的到达动态追加到该元素中。当训练完成时，路由返回一个 HTML 片段，生成一个表格，显示 `do_training` 方法返回的评估指标，如下所示的代码：

```py
[[SimpleClassificationDNN]]
@route(do_training="*")
   @captureOutput
def do_training_screen(self):
 self.classifier, self.eval_results = \
 do_training(
self.train_x, self.train_y, self.test_x, self.test_y, self.num_classes
 )
        return """
<h2>Training completed successfully</h2>
<table>
    <thead>
        <th>Metric</th>
        <th>Value</th>
    </thead>
    <tbody>
{%for key,value in this.eval_results.items()%}
<tr>
    <td>{{key}}</td>
    <td>{{value}}</td>
</tr>
{%endfor%}
    </tbody>
</table>
        """
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode6.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode6.py)

以下截图显示了模型成功创建后的结果，并包括分类模型的评估指标表，准确率为 87%：

![使用 DNNClassifier 进行简单分类](img/B09699_06_14.jpg)

显示成功训练结果的最终屏幕

这个 PixieApp 使用 `crimes` 数据集作为参数运行，如下所示的代码所示：

```py
app = SimpleClassificationDNN()
app.run(crimes)
```

一旦模型成功训练，你可以通过在 `app.classifier` 变量上调用 `predict` 方法来分类新数据。与 `train` 和 `evaluate` 方法类似，`predict` 也接受一个 `input_fn`，用于构造输入特征。

### 注意

**注意**：有关 `predict` 方法的更多细节，请参见此处：

[`www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier#predict`](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier#predict)

这个示例应用程序通过使用高层次的估算器 API，为熟悉 TensorFlow 框架提供了一个很好的起点。

### 注意

**注意**：此示例应用程序的完整笔记本可以在这里找到：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/TensorFlow%20classification.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/TensorFlow%20classification.ipynb)

在下一部分，我们将开始使用低级 TensorFlow API（包括张量、图和会话）构建我们的图像识别示例应用程序。

# 图像识别示例应用程序

当谈到构建一个开放式应用程序时，你应该从定义**MVP**（即**最小可行产品**）版本的需求开始，该版本仅包含足够的功能，足以使其对用户有用且有价值。在做技术决策时，确保你能够尽快获得一个完整的端到端实现，而不会投入过多时间，这是一个非常重要的标准。其核心思想是，你需要从小做起，这样你可以快速迭代并改进应用程序。

对于我们图像识别示例应用程序的 MVP，我们将使用以下要求：

+   不要从头开始构建模型；而是重用公开可用的预训练通用**卷积神经网络**（**CNN**：[`en.wikipedia.org/wiki/Convolutional_neural_network`](https://en.wikipedia.org/wiki/Convolutional_neural_network)）模型，如 MobileNet。我们可以稍后使用迁移学习（[`en.wikipedia.org/wiki/Transfer_learning`](https://en.wikipedia.org/wiki/Transfer_learning)）用自定义训练图像重新训练这些模型。

+   对于 MVP，我们虽然只关注评分而不涉及训练，但仍应确保应用程序对用户有吸引力。所以让我们构建一个 PixieApp，允许用户输入网页的 URL，并显示从页面中抓取的所有图片，包括我们的模型推断的分类输出。

+   由于我们正在学习深度学习神经网络和 TensorFlow，如果我们能够在 Jupyter Notebook 中直接显示 TensorBoard 图形可视化（[`www.tensorflow.org/programmers_guide/graph_viz`](https://www.tensorflow.org/programmers_guide/graph_viz)），而不强迫用户使用其他工具，那将会非常棒。这将提供更好的用户体验，并增强用户与应用程序的互动。

### 注意

**注意**：本节中的应用程序实现是根据以下教程改编的：

[`codelabs.developers.google.com/codelabs/tensorflow-for-poets`](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets)

## 第一部分 – 加载预训练的 MobileNet 模型

### 注意

**注意**：你可以下载完成的 Notebook 来跟进本节讨论：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%201.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%201.ipynb)

有很多公开可用的图像分类模型，使用 CNNs，且在如 ImageNet 等大型图像数据库上进行了预训练。ImageNet 发起了多个公开挑战，如**ImageNet 大规模视觉识别挑战赛**（**ILSVRC**）或 Kaggle 上的*ImageNet 物体定位挑战*（[`www.kaggle.com/c/imagenet-object-localization-challenge`](https://www.kaggle.com/c/imagenet-object-localization-challenge)），并取得了非常有趣的结果。

这些挑战催生了多个模型，如 ResNet、Inception、SqueezeNet、VGGNet 或 Xception，每个模型都使用不同的神经网络架构。详细讲解每个架构超出了本书的范围，但即使你还不是机器学习专家（我也绝对不是），我也鼓励你在网上阅读相关内容。为了这个示例应用，我选择了 MobileNet 模型，因为它小巧、快速且非常准确。它提供了一个包含 1,000 个类别的图像分类模型，足以满足此示例应用的需求。

为了确保代码的稳定性，我已在 GitHub 仓库中创建了模型的副本：[`github.com/DTAIEB/Thoughtful-Data-Science/tree/master/chapter%206/Visual%20Recognition/mobilenet_v1_0.50_224`](https://github.com/DTAIEB/Thoughtful-Data-Science/tree/master/chapter%206/Visual%20Recognition/mobilenet_v1_0.50_224)。

在这个目录中，你可以找到以下文件：

+   `frozen_graph.pb`：TensorFlow 图的序列化二进制版本

+   `labels.txt`：包含 1,000 个图像类别及其索引的文本文件

+   `quantized_graph.pb`：采用 8 位定点表示的模型图的压缩形式

加载模型的过程包括构建一个`tf.graph`对象及相关标签。由于未来可能会加载多个模型，因此我们首先定义一个字典，用来提供有关模型的元数据：

```py
models = {
    "mobilenet": {
        "base_url":"https://github.com/DTAIEB/Thoughtful-Data-Science/raw/master/chapter%206/Visual%20Recognition/mobilenet_v1_0.50_224",
        "model_file_url": "frozen_graph.pb",
        "label_file": "labels.txt",
        "output_layer": "MobilenetV1/Predictions/Softmax"
    }
}
```

### 注意

你可以在这里找到文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode7.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode7.py)

在前面的`models`字典中，每个键代表特定模型的元数据：

+   `base_url`：指向文件存储的 URL

+   `model_file_url`：假定相对于`base_url`的模型文件名称

+   `label_file`：假定相对于`base_url`的标签文件名称

+   `output_layer`：提供每个类别最终得分的输出层名称

我们实现了一个`get_model_attribute`辅助方法，以便从`model`元数据中读取内容，这在我们整个应用程序中都非常有用：

```py
# helper method for reading attributes from the model metadata
def get_model_attribute(model, key, default_value = None):
    if key not in model:
        if default_value is None:
            raise Exception("Require model attribute {} not found".format(key))
        return default_value
    return model[key]
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode8.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode8.py)

为了加载图形，我们下载二进制文件，使用`ParseFromString`方法将其加载到`tf.GraphDef`对象中，然后我们调用`tf.import_graph_def`方法，将图形作为当前内容管理器：

```py
import tensorflow as tf
import requests
# Helper method for resolving url relative to the selected model
def get_url(model, path):
    return model["base_url"] + "/" + path

# Download the serialized model and create a TensorFlow graph
def load_graph(model):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(
        requests.get( get_url( model, model["model_file_url"] ) ).content
    )
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode9.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode9.py)

加载标签的方法返回一个 JSON 对象或一个数组（稍后我们会看到这两者都需要）。以下代码使用 Python 列表推导式迭代`requests.get`调用返回的行。然后，它使用`as_json`标志将数据格式化为适当的形式：

```py
# Load the labels
def load_labels(model, as_json = False):
    labels = [line.rstrip() \
      for line in requests.get(get_url(model, model["label_file"]) ).text.split("\n") if line != ""]
    if as_json:
        return [{"index": item.split(":")[0],"label":item.split(":")[1]} for item in labels]
    return labels
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode10.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode10.py)

下一步是调用模型来分类图像。为了简化操作并可能提高其价值，我们要求用户提供一个包含待分类图像的 HTML 页面的 URL。我们将使用 BeautifulSoup4 库来帮助解析页面。要安装 BeautifulSoup4，只需运行以下命令：

```py
!pip install beautifulsoup4

```

### 注意

**注意**：像往常一样，安装完成后不要忘记重启内核。

以下`get_image_urls`方法接受一个 URL 作为输入，下载 HTML，实例化一个 BeautifulSoup 解析器，并提取所有`<img>`元素和`background-image`样式中找到的图像。BeautifulSoup 提供了一个非常优雅且易于使用的 API 来解析 HTML。在这里，我们只使用`find_all`方法来查找所有的`<img>`元素，并使用`select`方法选择所有具有内联样式的元素。读者很快会注意到，我们没有探索通过 HTML 创建图像的其他方式，例如，作为 CSS 类声明的图像。像往常一样，如果你有兴趣和时间改进它，我非常欢迎你在 GitHub 仓库中提交拉取请求（关于如何创建拉取请求，请参阅此处：[`help.github.com/articles/creating-a-pull-request`](https://help.github.com/articles/creating-a-pull-request)）。

`get_image_urls`的代码如下：

```py
from bs4 import BeautifulSoup as BS
import re

# return an array of all the images scraped from an html page
def get_image_urls(url):
    # Instantiate a BeautifulSoup parser
    soup = BS(requests.get(url).text, "html.parser")

    # Local helper method for extracting url
    def extract_url(val):
        m = re.match(r"url\((.*)\)", val)
        val = m.group(1) if m is not None else val
        return "http:" + val if val.startswith("//") else val

    # List comprehension that look for <img> elements and backgroud-image styles
    return [extract_url(imgtag['src']) for imgtag in soup.find_all('img')] + [ \
        extract_url(val.strip()) for key,val in \
        [tuple(selector.split(":")) for elt in soup.select("[style]") \
            for selector in elt["style"].strip(" ;").split(";")] \
            if key.strip().lower()=='background-image' \
        ]
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode11.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode11.py)

对于发现的每一张图片，我们还需要一个辅助函数来下载这些图片，这些图片将作为输入传递给模型进行分类。

以下`download_image`方法将图片下载到临时文件：

```py
import tempfile
def download_image(url):
   response = requests.get(url, stream=True)
   if response.status_code == 200:
      with tempfile.NamedTemporaryFile(delete=False) as f:
 for chunk in response.iter_content(2048):
 f.write(chunk)
         return f.name
   else:
      raise Exception("Unable to download image: {}".format(response.status_code))
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode12.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode12.py)

给定一张本地路径的图片，我们现在需要通过调用`tf.image`包中的正确解码方法将其解码为张量，也就是`.png`文件需要使用`decode_png`方法。

### 注意

**注意**：在数学中，张量是向量的一个推广，向量由方向和大小定义，张量则支持更高的维度。向量是 1 阶张量，同样，标量是 0 阶张量。直观地讲，我们可以把 2 阶张量看作一个二维数组，其中的值是通过乘以两个向量得到的结果。在 TensorFlow 中，张量是 n 维数组。

在对图片读取器张量进行一些转换（转换为正确的十进制表示、调整大小和归一化）之后，我们在归一化器张量上调用`tf.Session.run`以执行之前定义的步骤，如以下代码所示：

```py
# decode a given image into a tensor
def read_tensor_from_image_file(model, file_name):
    file_reader = tf.read_file(file_name, "file_reader")
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);

    # Read some info from the model metadata, providing default values
    input_height = get_model_attribute(model, "input_height", 224)
    input_width = get_model_attribute(model, "input_width", 224)
    input_mean = get_model_attribute(model, "input_mean", 0)
    input_std = get_model_attribute(model, "input_std", 255)

    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode13.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode13.py)

在所有部分准备好后，我们现在可以实现`score_image`方法，该方法接受`tf.graph`、模型元数据和图片的 URL 作为输入参数，并根据置信度分数返回前五个候选分类，包括它们的标签：

```py
import numpy as np

# classify an image given its url
def score_image(graph, model, url):
    # Get the input and output layer from the model
    input_layer = get_model_attribute(model, "input_layer", "input")
    output_layer = get_model_attribute(model, "output_layer")

    # Download the image and build a tensor from its data
    t = read_tensor_from_image_file(model, download_image(url))

    # Retrieve the tensors corresponding to the input and output layers
    input_tensor = graph.get_tensor_by_name("import/" + input_layer + ":0");
    output_tensor = graph.get_tensor_by_name("import/" + output_layer + ":0");

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_tensor, {input_tensor: t})
    results = np.squeeze(results)
    # select the top 5 candidate and match them to the labels
    top_k = results.argsort()[-5:][::-1]
 labels = load_labels(model)
 return [(labels[i].split(":")[1], results[i]) for i in top_k]

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode14.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode14.py)

我们现在可以使用以下步骤来测试代码：

1.  选择`mobilenet`模型并加载对应的图

1.  获取从 Flickr 网站抓取的图片 URL 列表

1.  对每个图片 URL 调用`score_image`方法并打印结果

代码如下所示：

```py
model = models['mobilenet']
graph = load_graph(model)
image_urls = get_image_urls("https://www.flickr.com/search/?text=cats")
for url in image_urls:
    results = score_image(graph, model, url)
    print("Result for {}: \n\t{}".format(url, results))
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode15.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode15.py)

结果非常准确（除了第一张是空白图片），如以下截图所示：

![第一部分 – 加载预训练的 MobileNet 模型](img/B09699_06_15.jpg)

对与猫相关的 Flickr 页面上发现的图片进行分类

我们的图像识别示例应用程序的第一部分现已完成；您可以在以下位置找到完整的 Notebook：[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%201.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%201.ipynb)。

在接下来的部分中，我们将通过构建 PixieApp 的用户界面来构建一个更加用户友好的体验。

## 第二部分 – 创建一个 PixieApp 用于我们的图像识别示例应用程序

### 注意

**注意**：您可以在此下载完成的 Notebook，以便跟随本部分的讨论：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%202.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%202.ipynb)

提醒一下，PixieApp 的`setup`方法（如果定义的话）会在应用程序开始运行之前执行。我们用它来选择模型并初始化图形：

```py
from pixiedust.display.app import *

@PixieApp
class ScoreImageApp():
    def setup(self):
        self.model = models["mobilenet"]
        self.graph = load_graph( self.model )
    ...
```

### 注意

您可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode16.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode16.py)

在 PixieApp 的主屏幕上，我们使用一个输入框让用户输入网页的 URL，如下所示的代码片段所示：

```py
[[ScoreImageApp]]
@route()
def main_screen(self):
   return """
<style>
    div.outer-wrapper {
        display: table;width:100%;height:300px;
    }
    div.inner-wrapper {
        display: table-cell;vertical-align: middle;height: 100%;width: 100%;
    }
</style>
<div class="outer-wrapper">
    <div class="inner-wrapper">
        <div class="col-sm-3"></div>
        <div class="input-group col-sm-6">
          <input id="url{{prefix}}" type="text" class="form-control"
              value="https://www.flickr.com/search/?text=cats"
              placeholder="Enter a url that contains images">
          <span class="input-group-btn">
            <button class="btn btn-default" type="button" pd_options="image_url=$val(url{{prefix}})">Go</button>
          </span>
        </div>
    </div>
</div>
"""
```

### 注意

您可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode17.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode17.py)

为了方便起见，我们将输入文本初始化为默认值`https://www.flickr.com/search/?text=cats`。

我们现在可以使用以下代码来运行并测试主屏幕：

```py
app = ScoreImageApp()
app.run()
```

主屏幕看起来是这样的：

![Part 2 – 创建一个 PixieApp 用于我们的图像识别示例应用程序](img/B09699_06_16.jpg)

图像识别 PixieApp 的主屏幕

### 注意

**注意**：这对于测试是好的，但我们应该记住，`do_process_url`路由尚未实现，因此，点击**Go**按钮将会回退到默认路由。

现在让我们实现`do_process_url`路由，它会在用户点击**Go**按钮时触发。该路由首先调用`get_image_urls`方法获取图像 URL 列表。然后，我们使用 Jinja2 构建一个 HTML 片段，显示所有图像。对于每个图像，我们异步调用`do_score_url`路由，运行模型并显示结果。

以下代码展示了`do_process_url`路由的实现：

```py
[[ScoreImageApp]]
@route(image_url="*")
@templateArgs
def do_process_url(self, image_url):
    image_urls = get_image_urls(image_url)
    return """
<div>
{%for url in image_urls%}
<div style="float: left; font-size: 9pt; text-align: center; width: 30%; margin-right: 1%; margin-bottom: 0.5em;">
<img src="img/{{url}}" style="width: 100%">
  <div style="display:inline-block" pd_render_onload pd_options="score_url={{url}}">
  </div>
</div>
{%endfor%}
<p style="clear: both;">
</div>
        """
```

### 注意

您可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode18.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode18.py)

注意`@templateArgs`装饰器的使用，它允许 Jinja2 片段引用本地的`image_urls`变量。

最后，在 `do_score_url` 路由中，我们调用 `score_image` 并将结果以列表形式显示：

```py
[[ScoreImageApp]]
@route(score_url="*")
@templateArgs
def do_score_url(self, score_url):
    results = score_image(self.graph, self.model, score_url)
    return """
<ul style="text-align:left">
{%for label, confidence in results%}
<li><b>{{label}}</b>: {{confidence}}</li>
{%endfor%}
</ul>
"""
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode19.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode19.py)

以下截图展示了包含猫咪图像的 Flickr 页面结果：

![第二部分 – 为我们的图像识别示例应用创建 PixieApp](img/B09699_06_17.jpg)

猫咪的图像分类结果

### 注意

提醒您，您可以在此位置找到完整的 Notebook：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%202.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%202.ipynb)

我们的 MVP 应用程序几乎完成。在下一节中，我们将直接在 Notebook 中集成 TensorBoard 图形可视化。

## 第三部分 – 集成 TensorBoard 图形可视化

### 注意

**注意**：本节中描述的部分代码改编自位于此处的 `deepdream` notebook：

[`github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)

您可以在这里下载完整的 Notebook 来跟随本节内容：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%203.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%203.ipynb)

TensorFlow 提供了一套非常强大的可视化工具，帮助调试和优化应用程序性能。请花点时间在这里探索 TensorBoard 的功能：[`www.tensorflow.org/programmers_guide/summaries_and_tensorboard`](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)。

这里的一个问题是，将 TensorBoard 服务器配置为与您的 Notebook 一起使用可能会很困难，特别是当您的 Notebooks 托管在云端时，且您几乎无法访问底层操作系统。在这种情况下，配置和启动 TensorBoard 服务器可能会变得几乎不可能。在本节中，我们展示了如何通过将模型图形可视化直接集成到 Notebook 中来解决这个问题，无需任何配置。为了提供更好的用户体验，我们希望将 TensorBoard 可视化功能添加到我们的 PixieApp 中。我们通过将主布局更改为选项卡布局，并将 TensorBoard 可视化分配到单独的选项卡中来实现这一点。方便的是，PixieDust 提供了一个名为 `TemplateTabbedApp` 的基础 PixieApp，它负责构建选项卡布局。当使用 `TemplateTabbedApp` 作为基类时，我们需要在 `setup` 方法中配置选项卡，如下所示：

```py
[[ImageRecoApp]]
from pixiedust.apps.template import TemplateTabbedApp
@PixieApp
class ImageRecoApp(TemplateTabbedApp):
    def setup(self):
        self.apps = [
            {"title": "Score", "app_class": "ScoreImageApp"},
            {"title": "Model", "app_class": "TensorGraphApp"},
            {"title": "Labels", "app_class": "LabelsApp"}
        ]
        self.model = models["mobilenet"]
        self.graph = self.load_graph(self.model)

app = ImageRecoApp()
app.run()
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode20.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode20.py)

需要注意的是，在前面的代码中，我们已经将`LabelsApp`子 PixieApp 添加到了标签页列表中，尽管它尚未实现。因此，正如预期的那样，如果直接运行这段代码，`Labels`标签将会失败。

`self.apps`包含一个对象数组，用于定义标签页：

+   `title`：标签页标题

+   `app_class`: 选中标签时运行的 PixieApp

在`ImageRecoApp`中，我们配置了三个与三个子 PixieApps 相关联的标签页：我们在*第二部分 – 为图像识别示例应用创建 PixieApp*中已经创建的`ScoreImageApp`，用于显示模型图的`TensorGraphApp`，以及用于显示模型中所有标注类别的表格的`LabelsApp`。

结果显示在以下截图中：

![第三部分 – 集成 TensorBoard 图形可视化](img/B09699_06_18.jpg)

包含 Score、Model 和 Labels 的标签布局

使用`TemplateTabbedApp`超类的另一个优点是，子 PixieApps 是分开定义的，这使得代码更易于维护和重用。

首先来看一下`TensorGraphApp` PixieApp。它的主路由返回一个 HTML 片段，该片段从`https://tensorboard.appspot.com`的 Iframe 加载`tf-graph-basic.build.html`，并使用 JavaScript 加载监听器应用通过`tf.Graph.as_graph_def`方法计算得到的序列化图定义。为了确保图定义保持在合理的大小，并避免在浏览器客户端上不必要的性能下降，我们调用`strip_consts`方法删除具有大尺寸常量值的张量。

`TensorGraphApp`的代码如下所示：

```py
@PixieApp
class TensorGraphApp():
    """Visualize TensorFlow graph."""
    def setup(self):
        self.graph = self.parent_pixieapp.graph

    @route()
    @templateArgs
    def main_screen(self):
        strip_def = self.strip_consts(self.graph.as_graph_def())
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph'+ self.getPrefix()).replace('"', '&quot;')

        return """
<iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{{code}}"></iframe>
"""

    def strip_consts(self, graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add() 
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped {} bytes>".format(size).encode("UTF-8")
        return strip_def
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode21.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode21.py)

**注意**：子 PixieApps 可以通过`self.parent_pixieapp`变量访问其父 PixieApp。

`TensorGraphApp`子 PixieApp 的结果屏幕如以下截图所示。它提供了选定模型的 TensorFlow 图的交互式可视化，允许用户浏览不同的节点，并深入探索模型。然而，重要的是要注意，整个可视化是在浏览器内运行的，而没有使用 TensorBoard 服务器。因此，TensorBoard 中的一些功能，如运行时统计信息，是禁用的。

![第三部分 – 集成 TensorBoard 图形可视化](img/B09699_06_19.jpg)

显示 MobileNet V1 的模型图

在`LabelsApp` PixieApp 中，我们只是将标签作为 JSON 格式加载，并使用`handlerId=tableView`选项在 PixieDust 表格中显示它。

```py
[[LabelsApp]]
@PixieApp
class LabelsApp():
    def setup(self):
        self.labels = self.parent_pixieapp.load_labels(
            self.parent_pixieapp.model, as_json=True
        )

    @route()
    def main_screen(self):
        return """
<div pd_render_onload pd_entity="labels">
    <pd_options>
    {
        "table_noschema": "true",
 "handlerId": "tableView",
        "rowCount": "10000"
    }
    </pd_options>
</div>
        """
```

### 注意

您可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode22.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode22.py)

**注意**：我们通过将`table_noschema`设置为`true`来配置表格，以避免显示模式架构，但为了方便起见，我们保留了搜索栏。

结果如下截图所示：

![第三部分 – 集成 TensorBoard 图形可视化](img/B09699_06_20.jpg)

可搜索的模型类别表格

我们的 MVP 图像识别示例应用程序现在已经完成；您可以在此找到完整的 Notebook：[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%203.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%203.ipynb)。

在下一节中，我们将通过允许用户使用自定义图像重新训练模型来改进应用程序。

## 第四部分 – 使用自定义训练数据重新训练模型

### 注意

**注意**：您可以在此下载完整的 Notebook 以便跟随本节的讨论：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%204.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%204.ipynb)

本节中的代码相当广泛，部分与主题无关的辅助函数将被省略。然而，和往常一样，更多关于代码的信息请参阅 GitHub 上的完整 Notebook。

在本节中，我们将使用自定义训练数据重新训练 MobileNet 模型，并用它来分类那些在通用模型中得分较低的图像。

### 注意

**注意**：本节中的代码改编自*TensorFlow for poets*教程：

[`github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/retrain.py`](https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/retrain.py)

正如大多数时候一样，获取高质量的训练数据可能是最具挑战性且耗时的任务。在我们的示例中，我们需要为每个要训练的类别获取大量图像。为了简便和可复现性，我们使用了 ImageNet 数据库，该数据库方便地提供了获取 URL 和相关标签的 API。我们还将下载的文件限制为`.jpg`格式。当然，如果需要，您也可以自行获取训练数据。

我们首先从 2011 年秋季发布的版本下载所有图片 URL 的列表，链接在这里：[`image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz`](http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz)，并将文件解压到你选择的本地目录（例如，我选择了`/Users/dtaieb/Downloads/fall11_urls.txt`）。我们还需要下载所有`synsets`的 WordNet ID 与单词的映射文件，链接在这里：[`image-net.org/archive/words.txt`](http://image-net.org/archive/words.txt)，这个文件将帮助我们找到包含我们需要下载的 URL 的 WordNet ID。

以下代码将分别加载两个文件到 pandas DataFrame 中：

```py
import pandas
wnid_to_urls = pandas.read_csv('/Users/dtaieb/Downloads/fall11_urls.txt',
                sep='\t', names=["wnid", "url"],
                header=0, error_bad_lines=False,
                warn_bad_lines=False, encoding="ISO-8859-1")
wnid_to_urls['wnid'] = wnid_to_urls['wnid'].apply(lambda x: x.split("_")[0])
wnid_to_urls = wnid_to_urls.dropna()

wnid_to_words = pandas.read_csv('/Users/dtaieb/Downloads/words.txt',
                sep='\t', names=["wnid", "description"],
                header=0, error_bad_lines=False,
                warn_bad_lines=False, encoding="ISO-8859-1")
wnid_to_words = wnid_to_words.dropna()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode23.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode23.py)

请注意，我们需要清理`wnid_to_urls`数据集中的`wnid`列，因为它包含一个后缀，表示该图片在类别中的索引。

然后我们可以定义一个方法`get_url_for_keywords`，它返回一个字典，字典的键是类别，值是包含 URL 的数组：

```py
def get_url_for_keywords(keywords):
    results = {}
    for keyword in keywords:
        df = wnid_to_words.loc[wnid_to_words['description'] == keyword]
        row_list = df['wnid'].values.tolist()
        descriptions = df['description'].values.tolist()
        if len(row_list) > 0:
            results[descriptions[0]] = \
            wnid_to_urls.loc[wnid_to_urls['wnid'] == \
            row_list[0]]["url"].values.tolist()
    return results
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode24.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode24.py)

我们可以使用 PixieDust 的`display`轻松查看数据分布。和往常一样，随时可以自己进行更多探索：

![第四部分 – 使用自定义训练数据重新训练模型](img/B09699_06_21.jpg)

按类别分布的图片

现在我们可以构建代码来下载与我们选择的类别列表对应的图片。在我们的例子中，我们选择了水果：`["apple", "orange", "pear", "banana"]`。这些图片将下载到 PixieDust 主目录的子目录中（使用 PixieDust 的`Environment`助手类，来自`pixiedust.utils`包），并限制下载图片的数量为`500`，以提高速度：

### 注意

**注意**：以下代码使用了 Notebook 中先前定义的方法和导入内容。在尝试运行以下代码之前，请确保先运行相应的单元格。

```py
from pixiedust.utils.environment import Environment
root_dir = ensure_dir_exists(os.path.join(Environment.pixiedustHome, "imageRecoApp")
image_dir = root_dir
image_dict = get_url_for_keywords(["apple", "orange", "pear", "banana"])
with open(os.path.join(image_dir, "retrained_label.txt"), "w") as f_label:
    for key in image_dict:
        f_label.write(key + "\n")
        path = ensure_dir_exists(os.path.join(image_dir, key))
        count = 0
        for url in image_dict[key]:
            download_image_into_dir(url, path)
            count += 1
            if count > 500:
                break;
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode25.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode25.py)

代码的下一部分处理训练集中的每张图片，使用以下步骤：

### 注意

**注意**：如前所述，代码比较长，并且部分代码被省略，这里仅解释了重要部分。请不要直接运行以下代码，完整实现请参阅完整的 Notebook。

1.  使用以下代码解码`.jpeg`文件：

    ```py
    def add_jpeg_decoding(model):
        input_height = get_model_attribute(model,
                       "input_height")
        input_width = get_model_attribute(model, "input_width")
        input_depth = get_model_attribute(model, "input_depth")
        input_mean = get_model_attribute(model, "input_mean",
                     0)
        input_std = get_model_attribute(model, "input_std",
                    255)

        jpeg_data = tf.placeholder(tf.string,
                    name='DecodeJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data,
                        channels=input_depth)
        decoded_image_as_float = tf.cast(decoded_image,
                                 dtype=tf.float32)
        decoded_image_4d =  tf.expand_dims(
                           decoded_image_as_float,
                           0)
        resize_shape = tf.stack([input_height, input_width])
        resize_shape_as_int = tf.cast(resize_shape,
                              dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(
                        decoded_image_4d,
                        resize_shape_as_int)
        offset_image = tf.subtract(resized_image, input_mean)
        mul_image = tf.multiply(offset_image, 1.0 / input_std)
        return jpeg_data, mul_image
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode26.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode26.py)

1.  创建瓶颈值（根据需要进行缓存），通过调整图像大小和缩放来标准化图像。这是在以下代码中完成的：

    ```py
    def run_bottleneck_on_image(sess, image_data,
        image_data_tensor,decoded_image_tensor,
        resized_input_tensor,bottleneck_tensor):
        # First decode the JPEG image, resize it, and rescale the pixel values.
        resized_input_values = sess.run(decoded_image_tensor,
            {image_data_tensor: image_data})
        # Then run it through the recognition network.
        bottleneck_values = sess.run(
            bottleneck_tensor,
            {resized_input_tensor: resized_input_values})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode27.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode27.py)

1.  使用`add_final_training_ops`方法添加最终训练操作，放在一个公共命名空间下，方便在可视化图时进行操作。训练步骤如下：

    1.  使用`tf.truncated_normal` API 生成随机权重：

        ```py
              initial_value = tf.truncated_normal(
                  [bottleneck_tensor_size, class_count],
                  stddev=0.001)
                  layer_weights = tf.Variable(
                      initial_value, name='final_weights')
        ```

    1.  添加偏置，初始化为零：

        ```py
              layer_biases = tf.Variable(tf.zeros([class_count]),
                  name='final_biases')
        ```

    1.  计算加权和：

        ```py
              logits = tf.matmul(bottleneck_input, layer_weights) +
                  layer_biases
        ```

    1.  添加`cross_entropy`成本函数：

        ```py
              cross_entropy =
                  tf.nn.softmax_cross_entropy_with_logits(
                  labels=ground_truth_input, logits=logits)
              with tf.name_scope('total'):
                  cross_entropy_mean = tf.reduce_mean(
                  cross_entropy)
        ```

    1.  最小化成本函数：

        ```py
              optimizer = tf.train.GradientDescentOptimizer(
                  learning_rate)
              train_step = optimizer.minimize(cross_entropy_mean)
        ```

为了可视化重新训练后的图，我们首先需要更新`TensorGraphApp` PixieApp，让用户选择可视化的模型：通用的 MobileNet 还是自定义模型。通过在主路由中添加`<select>`下拉菜单并附加`pd_script`元素来更新状态：

```py
[[TensorGraphApp]]
return """
{%if this.custom_graph%}
<div style="margin-top:10px" pd_refresh>
    <pd_script>
self.graph = self.custom_graph if self.graph is not self.custom_graph else self.parent_pixieapp.graph
    </pd_script>
    <span style="font-weight:bold">Select a model to display:</span>
    <select>
 <option {%if this.graph!=this.custom_graph%}selected{%endif%} value="main">MobileNet</option>
 <option {%if this.graph==this.custom_graph%}selected{%endif%} value="custom">Custom</options>
    </select>
{%endif%}
<iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{{code}}"></iframe>
"""
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode28.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode28.py)

重新运行我们的`ImageReco` PixieApp 生成以下截图：

![Part 4 – 使用自定义训练数据重新训练模型](img/B09699_06_22.jpg)

可视化重新训练后的图

点击火车节点将显示运行反向传播算法的嵌套操作，以最小化前面`add_final_training_ops`中指定的`cross_entropy_mean`成本函数：

```py
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode29.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode29.py)

以下截图展示了**train**命名空间的详细信息：

![Part 4 – 使用自定义训练数据重新训练模型](img/B09699_06_23.jpg)

训练过程中的反向传播

类似地，我们可以在`LabelsApp`中添加下拉切换，以切换通用 MobileNet 和自定义模型之间的可视化：

```py
[[LabelsApp]]
@PixieApp
class LabelsApp():
    def setup(self):
        ...

    @route()
    def main_screen(self):
        return """
{%if this.custom_labels%}
<div style="margin-top:10px" pd_refresh>
    <pd_script>
self.current_labels = self.custom_labels if self.current_labels is not self.custom_labels else self.labels
    </pd_script>
    <span style="font-weight:bold">
        Select a model to display:</span>
    <select>
        <option {%if this.current_labels!=this.labels%}selected{%endif%} value="main">MobileNet</option>
        <option {%if this.current_labels==this.custom_labels%}selected{%endif%} value="custom">Custom</options>
    </select>
{%endif%}
<div pd_render_onload pd_entity="current_labels">
    <pd_options>
    {
        "table_noschema": "true",
        "handlerId": "tableView",
        "rowCount": "10000",
        "noChartCache": "true"

    }
    </pd_options>
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode30.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode30.py)

结果显示在以下截图中：

![Part 4 – 使用自定义训练数据重新训练模型](img/B09699_06_24.jpg)

显示每个模型的标签信息

第四部分 MVP 的最后一步是更新 `score_image` 方法，使其同时使用两个模型对图像进行分类，并将结果以字典形式存储，其中每个模型有一个条目。我们定义了一个本地方法 `do_score_image`，该方法返回前 5 个候选答案。

该方法会为每个模型调用，并将结果填充到一个字典中，字典的键是模型名称：

```py
# classify an image given its url
def score_image(graph, model, url):
    # Download the image and build a tensor from its data
    t = read_tensor_from_image_file(model, download_image(url))

    def do_score_image(graph, output_layer, labels):
        # Retrieve the tensors corresponding to the input and output layers
        input_tensor = graph.get_tensor_by_name("import/" +
            input_layer + ":0");
        output_tensor = graph.get_tensor_by_name( output_layer +
            ":0");

        with tf.Session(graph=graph) as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            results = sess.run(output_tensor, {input_tensor: t})
        results = np.squeeze(results)
        # select the top 5 candidates and match them to the labels
        top_k = results.argsort()[-5:][::-1]
        return [(labels[i].split(":")[1], results[i]) for i in top_k]

    results = {}
    input_layer = get_model_attribute(model, "input_layer",
        "input")
    labels = load_labels(model)
    results["mobilenet"] = do_score_image(graph, "import/" +
        get_model_attribute(model, "output_layer"), labels)
    if "custom_graph" in model and "custom_labels" in model:
        with open(model["custom_labels"]) as f:
            labels = [line.rstrip() for line in f.readlines() if line != ""]
            custom_labels = ["{}:{}".format(i, label) for i,label in zip(range(len(labels)), labels)]
        results["custom"] = do_score_image(model["custom_graph"],
            "final_result", custom_labels)
    return results
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode31.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode31.py)

由于我们修改了 `score_image` 方法返回的值，我们需要调整 `ScoreImageApp` 中返回的 HTML 片段，以便遍历 `results` 字典中的所有模型条目：

```py
@route(score_url="*")
@templateArgs
def do_score_url(self, score_url):
    scores_dict = score_image(self.graph, self.model, score_url)
    return """
{%for model, results in scores_dict.items()%}
<div style="font-weight:bold">{{model}}</div>
<ul style="text-align:left">
{%for label, confidence in results%}
<li><b>{{label}}</b>: {{confidence}}</li>
{%endfor%}
</ul>
{%endfor%}
    """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode32.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/sampleCode32.py)

在这些更改生效后，PixieApp 将会自动调用可用的自定义模型，并且如果存在自定义模型，它会显示两个模型的结果。

下图显示了与 *香蕉* 相关的图像的结果：

![Part 4 – 使用自定义训练数据重新训练模型](img/B09699_06_25.jpg)

使用通用的 MobileNet 和自定义训练模型进行评分

读者会注意到自定义模型的分数相当低。一个可能的解释是，训练数据获取是完全自动化的，并且没有人工筛选。对这个示例应用程序的一个可能改进是，将训练数据获取和再训练步骤移到一个独立的 PixieApp 标签页中。我们还应当给用户机会验证图像，并拒绝质量差的图像。让用户重新标注错误分类的图像也是一个不错的主意。

### 注意

第四部分的完整 Notebook 可以在这里找到：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%204.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%206/Tensorflow%20VR%20Part%204.ipynb)

本节讨论了使用 TensorFlow 在 Jupyter Notebook 中构建图像识别示例应用程序的增量方法，特别关注如何通过 PixieApps 实现算法的操作化。我们首先通过 TensorFlow `DNNClassifier` 估计器，从 pandas DataFrame 中构建了一个简单的分类模型。接着，我们将图像识别示例应用程序的 MVP 版本分为四部分来构建：

1.  我们加载了预训练的 MobileNet 模型

1.  我们为我们的图像识别示例应用程序创建了一个 PixieApp

1.  我们将 TensorBoard 图形可视化集成到 PixieApp 中

1.  我们使用户能够使用来自 ImageNet 的自定义训练数据重新训练模型

# 概述

机器学习是一个庞大的领域，享有巨大的增长，无论是在研究还是开发方面。在本章中，我们只探讨了与机器学习算法相关的极小一部分前沿技术，具体来说，是使用深度学习神经网络进行图像识别。对于一些刚刚开始接触机器学习的读者，示例 PixieApp 及其关联的算法代码可能一次性难以消化。然而，底层的目标是展示如何逐步构建一个应用程序，并利用机器学习模型。我们恰好使用了一个卷积神经网络模型进行图像识别，但任何其他模型都可以使用。

希望你已经对 PixieDust 和 PixieApp 编程模型如何帮助你完成自己的项目有了一个不错的了解，我强烈建议你以这个示例应用程序为起点，使用你选择的机器学习方法来构建自己的自定义应用程序。我还推荐将你的 PixieApp 部署为一个 web 应用程序，并通过 PixieGateway 微服务进行测试，看看它是否是一个可行的解决方案。

在下一章，我们将介绍另一个与大数据和自然语言处理相关的重要行业应用案例。我们将构建一个示例应用程序，通过自然语言理解服务分析社交媒体趋势。
