# 第六章：介绍 ML 包

在上一章中，我们使用了 Spark 的 MLlib 包，该包严格在 RDD 上操作。在本章中，我们将转向 Spark 的 ML 部分，该部分严格在 DataFrame 上操作。此外，根据 Spark 文档，Spark 的主要机器学习 API 现在是包含在`spark.ml`包中的基于 DataFrame 的模型集。

那么，让我们开始吧！

### 注意

在本章中，我们将重用上一章中我们使用的数据集的一部分。数据可以从[`www.tomdrabas.com/data/LearningPySpark/births_transformed.csv.gz`](http://www.tomdrabas.com/data/LearningPySpark/births_transformed.csv.gz)下载。

在本章中，你将学习以下内容：

+   准备转换器、估计器和管道

+   使用 ML 包中的模型预测婴儿生存的机会

+   评估模型的性能

+   执行参数超调

+   使用包中可用的其他机器学习模型

# 包的概述

在最高级别，该包公开了三个主要抽象类：一个`Transformer`，一个`Estimator`和一个`Pipeline`。我们将很快通过一些简短的示例来解释每个类。我们将在本章的最后部分提供一些模型的具体示例。

## 转换器

`Transformer`类，正如其名称所暗示的，通过（通常）向 DataFrame 中添加新列来*转换*你的数据。

在高级别，当从`Transformer`抽象类派生时，每个新的`Transformer`都需要实现一个`.transform(...)`方法。这个方法，作为一个首要且通常是唯一必需的参数，需要传递一个要转换的 DataFrame。当然，在 ML 包中，这会*方法各异*：其他*常用*参数是`inputCol`和`outputCol`；然而，这些参数通常默认为一些预定义的值，例如，对于`inputCol`参数，默认值可能是`'features'`。

`spark.ml.feature`提供了许多`Transformers`，我们将在下面简要描述它们（在我们本章后面使用它们之前）：

+   `Binarizer`：给定一个阈值，该方法将连续变量转换为二进制变量。

+   `Bucketizer`：类似于`Binarizer`，该方法接受一系列阈值（`splits`参数）并将连续变量转换为多项式变量。

+   `ChiSqSelector`: 对于分类目标变量（例如分类模型），此功能允许你选择一个预定义数量的特征（由`numTopFeatures`参数参数化），这些特征最好地解释了目标中的方差。选择是通过名称暗示的方法完成的，即使用卡方检验。这是两步方法之一：首先，你需要`.fit(...)`你的数据（以便方法可以计算卡方检验）。调用`.fit(...)`方法（你传递 DataFrame 作为参数）返回一个`ChiSqSelectorModel`对象，然后你可以使用该对象通过`.transform(...)`方法转换你的 DataFrame。

    ### 注意

    更多关于卡方检验的信息可以在这里找到：[`ccnmtl.columbia.edu/projects/qmss/the_chisquare_test/about_the_chisquare_test.html`](http://ccnmtl.columbia.edu/projects/qmss/the_chisquare_test/about_the_chisquare_test.html)。

+   `CountVectorizer`: 这对于标记化文本（例如`[['Learning', 'PySpark', 'with', 'us'],['us', 'us', 'us']]`）非常有用。它是两步方法之一：首先，你需要使用`.fit(...)`（即从你的数据集中学习模式），然后你才能使用由`.fit(...)`方法返回的`CountVectorizerModel`进行`.transform(...)`。此转换器的输出，对于前面提供的标记化文本，将类似于以下内容：`[(4, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0]),(4, [3], [3.0])]`。

+   `DCT`: 离散余弦变换（Discrete Cosine Transform）将一个实数值向量转换为一个长度相同的向量，但其中的余弦函数以不同的频率振荡。这种变换对于从你的数据或数据压缩中提取一些基本频率非常有用。

+   `ElementwiseProduct`: 这是一个返回元素为传递给方法的向量与作为`scalingVec`参数传递的向量乘积的向量的方法。例如，如果你有一个`[10.0, 3.0, 15.0]`向量，并且你的`scalingVec`是`[0.99, 3.30, 0.66]`，那么你将得到的向量将如下所示：`[9.9, 9.9, 9.9]`。

+   `HashingTF`: 这是一个哈希技巧转换器，它接受一个标记化文本列表并返回一个向量（具有预定义的长度）和计数。从 PySpark 的文档中：

    > "由于使用了简单的模运算将哈希函数转换为列索引，因此建议将 numFeatures 参数设置为 2 的幂；否则，特征将不会均匀地映射到列上。"

+   `IDF`: 此方法计算文档列表的**逆文档频率**。请注意，文档需要已经表示为向量（例如，使用`HashingTF`或`CountVectorizer`）。

+   `IndexToString`: 这是`StringIndexer`方法的补充。它使用`StringIndexerModel`对象的编码将字符串索引反转回原始值。顺便提一下，请注意，这有时可能不起作用，你需要从`StringIndexer`指定值。

+   `MaxAbsScaler`: 将数据重新缩放到`[-1.0, 1.0]`范围内（因此，它不会移动数据的中心）。

+   `MinMaxScaler`: 与`MaxAbsScaler`类似，但不同之处在于它将数据缩放到`[0.0, 1.0]`范围内。

+   `NGram`: 此方法接受一个标记化文本的列表，并返回*n-gram*：后续单词的成对、三元组或*n-mores*。例如，如果您有一个`['good', 'morning', 'Robin', 'Williams']`向量，您将得到以下输出：`['good morning', 'morning Robin', 'Robin Williams']`。

+   `Normalizer`: 此方法使用 p-norm 值（默认为 L2）将数据缩放到单位范数。

+   `OneHotEncoder`: 此方法将分类列编码为二进制向量列。

+   `PCA`: 使用主成分分析进行数据降维。

+   `PolynomialExpansion`: 对向量执行多项式展开。例如，如果您有一个符号写为`[x, y, z]`的向量，该方法将生成以下展开：`[x, x*x, y, x*y, y*y, z, x*z, y*z, z*z]`。

+   `QuantileDiscretizer`: 与`Bucketizer`方法类似，但您不是传递`splits`参数，而是传递`numBuckets`。然后，方法通过计算数据上的近似分位数来决定应该是什么分割。

+   `RegexTokenizer`: 这是一个使用正则表达式的字符串分词器。

+   `RFormula`: 对于那些热衷于使用 R 的用户，您可以通过传递一个公式，例如`vec ~ alpha * 3 + beta`（假设您的`DataFrame`有`alpha`和`beta`列），它将根据表达式生成`vec`列。

+   `SQLTransformer`: 与之前类似，但您可以使用 SQL 语法而不是 R-like 公式。

    ### 提示

    `FROM`语句应选择`__THIS__`，表示您正在访问 DataFrame。例如：`SELECT alpha * 3 + beta AS vec FROM __THIS__`。

+   `StandardScaler`: 将列标准化，使其具有 0 均值和标准差等于 1。

+   `StopWordsRemover`: 从标记化文本中移除停用词（如`'the'`或`'a'`）。

+   `StringIndexer`: 给定一个列中所有单词的列表，这将生成一个索引向量。

+   `Tokenizer`: 这是一个默认的分词器，它将字符串转换为小写，然后根据空格分割。

+   `VectorAssembler`: 这是一个非常有用的转换器，它将多个数值（包括向量）列合并成一个具有向量表示的单列。例如，如果您在 DataFrame 中有三个列：

    ```py
    df = spark.createDataFrame(
        [(12, 10, 3), (1, 4, 2)], 
        ['a', 'b', 'c']) 
    ```

    调用以下内容的输出：

    ```py
    ft.VectorAssembler(inputCols=['a', 'b', 'c'], 
            outputCol='features')\
        .transform(df) \
        .select('features')\
        .collect() 
    ```

    它看起来如下：

    ```py
    [Row(features=DenseVector([12.0, 10.0, 3.0])), 
     Row(features=DenseVector([1.0, 4.0, 2.0]))]
    ```

+   `VectorIndexer`: 这是一个将分类列索引到索引向量的方法。它以*列-by-列*的方式工作，从列中选择不同的值，排序并返回映射中的值的索引，而不是原始值。

+   `VectorSlicer`: 在特征向量上工作，无论是密集的还是稀疏的：给定一个索引列表，它从特征向量中提取值。

+   `Word2Vec`: 这种方法将一个句子（字符串）作为输入，并将其转换成 `{string, vector}` 格式的映射，这种表示在自然语言处理中非常有用。

    ### 注意

    注意，ML 包中有很多方法旁边都有一个 E 字母；这意味着该方法目前处于测试版（或实验性）状态，有时可能会失败或产生错误的结果。请小心。

## 估计器

估计器可以被视为需要估计以进行预测或对观测值进行分类的统计模型。

如果从抽象的 `Estimator` 类派生，新的模型必须实现 `.fit(...)` 方法，该方法根据 DataFrame 中的数据和一些默认或用户指定的参数来拟合模型。

PySpark 中有很多估计器可用，我们现在将简要描述 Spark 2.0 中可用的模型。

### 分类

机器学习包为数据科学家提供了七种分类模型可供选择。这些模型从最简单的（例如逻辑回归）到更复杂的模型不等。我们将在下一节中简要介绍每种模型：

+   `LogisticRegression`: 分类领域的基准模型。逻辑回归使用 logit 函数来计算观测值属于特定类别的概率。在撰写本文时，PySpark ML 仅支持二元分类问题。

+   `DecisionTreeClassifier`: 一个构建决策树以预测观测值类别的分类器。指定 `maxDepth` 参数限制树的生长深度，`minInstancePerNode` 确定树节点中所需的最小观测值数量以进一步分割，`maxBins` 参数指定连续变量将被分割成的最大箱数，而 `impurity` 指定用于衡量和计算分割信息增益的指标。

+   `GBTClassifier`: 一个用于分类的 **梯度提升树** 模型。该模型属于集成模型家族：这些模型将多个弱预测模型组合成一个强模型。目前，`GBTClassifier` 模型支持二元标签以及连续和分类特征。

+   `RandomForestClassifier`: 该模型生成多个决策树（因此得名——森林），并使用这些决策树的 `mode` 输出来对观测值进行分类。`RandomForestClassifier` 支持二元和多项式标签。

+   `NaiveBayes`: 基于贝叶斯定理，该模型使用条件概率理论对观测值进行分类。PySpark ML 中的 `NaiveBayes` 模型支持二元和多项式标签。

+   `多层感知器分类器`：一种模仿人类大脑性质的分类器。深深植根于人工神经网络理论，该模型是一个黑盒，即模型的内部参数不易解释。该模型至少由三个全连接的`层`（在创建模型对象时需要指定的参数）组成的人工神经元：输入层（需要等于数据集中的特征数量），若干隐藏层（至少一个），以及一个输出层，其神经元数量等于标签中的类别数量。输入层和隐藏层中的所有神经元都有 sigmoid 激活函数，而输出层神经元的激活函数是 softmax。

+   `一对抗`：将多类分类减少为二类分类。例如，在多项式标签的情况下，模型可以训练多个二元逻辑回归模型。例如，如果`label == 2`，则模型将构建一个逻辑回归，其中将`label == 2`转换为`1`（所有剩余的标签值将设置为`0`），然后训练一个二元模型。然后对所有模型进行评分，概率最高的模型获胜。

### 回归

PySpark ML 包中提供了七个回归任务模型。与分类类似，这些模型从一些基本的（如强制性的线性回归）到更复杂的模型：

+   `加速失效时间回归`：拟合加速失效时间回归模型。这是一个参数模型，假设某个特征的一个边缘效应会加速或减慢预期寿命（或过程失效）。它非常适合具有明确阶段的流程。

+   `决策树回归器`：与分类模型类似，但有一个明显的区别，即标签是连续的而不是二元的（或多项式的）。

+   `梯度提升回归器`：与`决策树回归器`类似，区别在于标签的数据类型。

+   `广义线性回归`：具有不同核函数（链接函数）的线性模型族。与假设误差项正态性的线性回归相比，GLM 允许标签具有不同的误差项分布：PySpark ML 包中的`广义线性回归`模型支持`高斯`、`二项式`、`伽马`和`泊松`误差分布族，以及众多不同的链接函数。

+   `等调回归`：一种回归类型，将自由形式的非递减线拟合到您的数据。对于具有有序和递增观测值的数据集来说很有用。

+   `线性回归`：回归模型中最简单的一种，它假设特征与连续标签之间存在线性关系，以及误差项的正态性。

+   `随机森林回归器`：类似于`决策树回归器`或`梯度提升回归器`，`随机森林回归器`拟合的是连续标签而不是离散标签。

### 聚类

聚类是一组无监督模型，用于在数据中找到潜在的规律。PySpark ML 包目前提供了四个最流行的模型：

+   `BisectingKMeans`: k 均值聚类方法和层次聚类的组合。算法开始时将所有观测值放在一个簇中，然后迭代地将数据分割成`k`个簇。

    ### 注意

    查阅此网站以获取有关伪算法的更多信息：[`minethedata.blogspot.com/2012/08/bisecting-k-means.html`](http://minethedata.blogspot.com/2012/08/bisecting-k-means.html)。

+   `KMeans`: 这是著名的 k 均值算法，它将数据分离成`k`个簇，迭代地寻找使每个观测值与其所属簇的质心之间的平方距离之和最小的质心。

+   `GaussianMixture`: 此方法使用具有未知参数的`k`个高斯分布来剖析数据集。通过最大化对数似然函数，使用期望最大化算法找到高斯参数。

    ### 小贴士

    注意，对于具有许多特征的集合，由于维度诅咒和高斯分布的数值问题，此模型可能表现不佳。

+   `LDA`: 该模型用于自然语言处理应用中的主题建模。

PySpark ML 中还有一个推荐模型可用，但我们将在此处不对其进行描述。

## Pipeline

PySpark ML 中的`Pipeline`是一个端到端转换-估计过程（具有不同的阶段）的概念，它接收一些原始数据（以 DataFrame 形式），执行必要的数据整理（转换），并最终估计一个统计模型（估计器）。

### 小贴士

`Pipeline`可以是纯粹的转换型，即仅由`Transformer`组成。

可以将`Pipeline`视为多个离散阶段的链。当在`Pipeline`对象上执行`.fit(...)`方法时，所有阶段都按照在`stages`参数中指定的顺序执行；`stages`参数是一个`Transformer`和`Estimator`对象的列表。`Pipeline`对象的`.fit(...)`方法执行`Transformer`的`.transform(...)`方法和`Estimator`的`.fit(...)`方法。

通常，前一个阶段的输出成为下一个阶段的输入：当从`Transformer`或`Estimator`抽象类派生时，需要实现`.getOutputCol()`方法，该方法返回创建对象时指定的`outputCol`参数的值。

# 使用 ML 预测婴儿生存的机会

在本节中，我们将使用上一章的数据集的一部分来展示 PySpark ML 的思想。

### 注意

如果你在阅读上一章时还没有下载数据，可以在此处访问：[`www.tomdrabas.com/data/LearningPySpark/births_transformed.csv.gz`](http://www.tomdrabas.com/data/LearningPySpark/births_transformed.csv.gz)。

在本节中，我们将再次尝试预测婴儿生存的机会。

## 加载数据

首先，我们使用以下代码加载数据：

```py
import pyspark.sql.types as typ
labels = [
    ('INFANT_ALIVE_AT_REPORT', typ.IntegerType()),
    ('BIRTH_PLACE', typ.StringType()),
    ('MOTHER_AGE_YEARS', typ.IntegerType()),
    ('FATHER_COMBINED_AGE', typ.IntegerType()),
    ('CIG_BEFORE', typ.IntegerType()),
    ('CIG_1_TRI', typ.IntegerType()),
    ('CIG_2_TRI', typ.IntegerType()),
    ('CIG_3_TRI', typ.IntegerType()),
    ('MOTHER_HEIGHT_IN', typ.IntegerType()),
    ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
    ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
    ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
    ('DIABETES_PRE', typ.IntegerType()),
    ('DIABETES_GEST', typ.IntegerType()),
    ('HYP_TENS_PRE', typ.IntegerType()),
    ('HYP_TENS_GEST', typ.IntegerType()),
    ('PREV_BIRTH_PRETERM', typ.IntegerType())
]
schema = typ.StructType([
    typ.StructField(e[0], e[1], False) for e in labels
])
births = spark.read.csv('births_transformed.csv.gz', 
                        header=True, 
                        schema=schema)
```

我们指定 DataFrame 的模式；我们的数据集现在只有 17 列。

## 创建转换器

在我们可以使用数据集估计模型之前，我们需要进行一些转换。由于统计模型只能操作数值数据，我们将不得不对`BIRTH_PLACE`变量进行编码。

在我们进行任何操作之前，由于我们将在本章后面使用许多不同的特征转换，让我们导入它们：

```py
import pyspark.ml.feature as ft
```

为了对`BIRTH_PLACE`列进行编码，我们将使用`OneHotEncoder`方法。然而，该方法不能接受`StringType`列；它只能处理数值类型，因此我们首先将列转换为`IntegerType`：

```py
births = births \
    .withColumn('BIRTH_PLACE_INT', births['BIRTH_PLACE'] \
    .cast(typ.IntegerType()))
```

完成这些后，我们现在可以创建我们的第一个`Transformer`：

```py
encoder = ft.OneHotEncoder(
    inputCol='BIRTH_PLACE_INT', 
    outputCol='BIRTH_PLACE_VEC')
```

现在让我们创建一个包含所有特征的单一列。我们将使用`VectorAssembler`方法：

```py
featuresCreator = ft.VectorAssembler(
    inputCols=[
        col[0] 
        for col 
        in labels[2:]] + \
    [encoder.getOutputCol()], 
    outputCol='features'
)
```

传递给`VectorAssembler`对象的`inputCols`参数是一个列表，其中包含所有要组合在一起形成`outputCol`（即`'features'`）的列。请注意，我们使用编码器对象的输出（通过调用`.getOutputCol()`方法），因此我们不必记住在编码器对象中更改输出列名称时更改此参数的值。

现在是时候创建我们的第一个估计器了。

## 创建估计器

在这个例子中，我们将（再次）使用逻辑回归模型。然而，在本章的后面，我们将展示一些来自 PySpark ML 模型`.classification`集合的更复杂模型，因此我们加载整个部分：

```py
import pyspark.ml.classification as cl
```

一旦加载，让我们使用以下代码创建模型：

```py
logistic = cl.LogisticRegression(
    maxIter=10, 
    regParam=0.01, 
    labelCol='INFANT_ALIVE_AT_REPORT')
```

如果我们的目标列名为`'label'`，我们就不必指定`labelCol`参数。此外，如果我们的`featuresCreator`的输出不是名为`'features'`，我们就必须通过（最方便的）在`featuresCreator`对象上调用`getOutputCol()`方法来指定`featuresCol`。

## 创建 pipeline

现在剩下的只是创建一个`Pipeline`并拟合模型。首先，让我们从 ML 包中加载`Pipeline`：

```py
from pyspark.ml import Pipeline
```

创建一个`Pipeline`非常简单。以下是我们的 pipeline 在概念上的样子：

![创建 pipeline](img/B05793_06_01.jpg)

将这个结构转换为`Pipeline`是一件轻而易举的事情：

```py
pipeline = Pipeline(stages=[
        encoder, 
        featuresCreator, 
        logistic
    ])
```

就这样！我们的`pipeline`现在已经创建好了，我们可以（终于！）估计模型了。

## 模型拟合

在拟合模型之前，我们需要将我们的数据集分成训练集和测试集。方便的是，DataFrame API 有`.randomSplit(...)`方法：

```py
births_train, births_test = births \
    .randomSplit([0.7, 0.3], seed=666)
```

第一个参数是一个列表，其中包含应该分别进入`births_train`和`births_test`子集的数据集比例。`seed`参数为随机化器提供一个种子。

### 注意

只要列表的元素之和为 1，你就可以将数据集分成超过两个子集，并将输出解包成尽可能多的子集。

例如，我们可以将出生数据集分成三个子集，如下所示：

```py
train, test, val = births.\
    randomSplit([0.7, 0.2, 0.1], seed=666)
```

前面的代码会将出生数据集的 70%随机放入`train`对象中，20%会进入`test`，而`val` DataFrame 将保留剩余的 10%。

现在是时候最终运行我们的流水线和估计我们的模型了：

```py
model = pipeline.fit(births_train)
test_model = model.transform(births_test)
```

流水线对象的`.fit(...)`方法将我们的训练数据集作为输入。在内部，`births_train`数据集首先传递给`encoder`对象。在`encoder`阶段创建的 DataFrame 随后传递给`featuresCreator`，它创建`'features'`列。最后，这一阶段的输出传递给`logistic`对象，它估计最终模型。

`.fit(...)`方法返回`PipelineModel`对象（前面代码片段中的`model`对象），然后可以用于预测；我们通过调用`.transform(...)`方法并传递之前创建的测试数据集来实现这一点。以下命令中的`test_model`看起来是这样的：

```py
test_model.take(1)
```

它生成了以下输出：

![拟合模型](img/B05793_06_02.jpg)

如你所见，我们得到了来自`Transfomers`和`Estimators`的所有列。逻辑回归模型输出几个列：`rawPrediction`是特征和β系数的线性组合的值，`probability`是每个类计算的概率，最后是`prediction`，即我们的最终类别分配。

## 评估模型的性能

显然，我们现在想测试我们的模型表现如何。PySpark 在包的`.evaluation`部分暴露了多个用于分类和回归的评估方法：

```py
import pyspark.ml.evaluation as ev
```

我们将使用`BinaryClassficationEvaluator`来测试我们的模型表现如何：

```py
evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='INFANT_ALIVE_AT_REPORT')
```

`rawPredictionCol`可以是估计器生成的`rawPrediction`列或`probability`。

让我们看看我们的模型表现如何：

```py
print(evaluator.evaluate(test_model, 
    {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(test_model, 
   {evaluator.metricName: 'areaUnderPR'}))
```

前面的代码生成了以下结果：

![评估模型的性能](img/B05793_06_03.jpg)

ROC 曲线下 74%的面积和 PR 曲线下 71%的面积显示了一个定义良好的模型，但并没有什么异常之处；如果我们有其他特征，我们可以进一步提高这个值，但这不是本章（甚至整本书）的目的。

## 保存模型

PySpark 允许你保存`Pipeline`定义以供以后使用。它不仅保存了流水线结构，还保存了所有`Transformers`和`Estimators`的定义：

```py
pipelinePath = './infant_oneHotEncoder_Logistic_Pipeline'
pipeline.write().overwrite().save(pipelinePath)
```

因此，你可以稍后加载它并直接使用它来`.fit(...)`和预测：

```py
loadedPipeline = Pipeline.load(pipelinePath)
loadedPipeline \
    .fit(births_train)\
    .transform(births_test)\
    .take(1)
```

前面的代码生成了相同的结果（正如预期）：

![保存模型](img/B05793_06_04.jpg)

然而，如果你想要保存估计的模型，你也可以这样做；你不需要保存`Pipeline`，而是需要保存`PipelineModel`。

### 小贴士

注意，不仅`PipelineModel`可以被保存：几乎所有通过在`Estimator`或`Transformer`上调用`.fit(...)`方法返回的模型都可以被保存并重新加载以供重用。

要保存你的模型，请参考以下示例：

```py
from pyspark.ml import PipelineModel

modelPath = './infant_oneHotEncoder_Logistic_PipelineModel'
model.write().overwrite().save(modelPath)

loadedPipelineModel = PipelineModel.load(modelPath)
test_reloadedModel = loadedPipelineModel.transform(births_test)
```

前面的脚本使用了`PipelineModel`类的类方法`.load(...)`来重新加载估计的模型。你可以将`test_reloadedModel.take(1)`的结果与之前展示的`test_model.take(1)`的输出进行比较。

# 参数超调

很少情况下，我们的第一个模型就是我们能做的最好的。仅仅通过查看我们的指标并接受模型因为它通过了我们预想的性能阈值，这几乎不是寻找最佳模型的科学方法。

参数超调的概念是找到模型的最佳参数：例如，正确估计逻辑回归模型所需的最大迭代次数或决策树的最大深度。

在本节中，我们将探讨两个概念，这些概念可以帮助我们找到模型的最佳参数：网格搜索和训练-验证拆分。

## 网格搜索

网格搜索是一个穷举算法，它遍历定义的参数值列表，估计单独的模型，并根据某些评估指标选择最佳模型。

应该指出的是：如果你定义了太多的参数想要优化，或者这些参数的值太多，选择最佳模型可能需要很长时间，因为随着参数和参数值的增加，估计的模型数量会迅速增长。

例如，如果你想要微调两个参数，并且每个参数有两个值，那么你需要拟合四个模型。增加一个额外的参数并赋予两个值，将需要估计八个模型，而将我们的两个参数增加一个额外的值（使每个参数变为三个值），将需要估计九个模型。正如你所见，如果不小心，这会迅速变得难以控制。请查看以下图表以直观地检查这一点：

![网格搜索](img/B05793_06_05.jpg)

在这个警示故事之后，让我们开始微调我们的参数空间。首先，我们加载包的`.tuning`部分：

```py
import pyspark.ml.tuning as tune
```

接下来，让我们指定我们的模型和想要遍历的参数列表：

```py
logistic = cl.LogisticRegression(
    labelCol='INFANT_ALIVE_AT_REPORT')
grid = tune.ParamGridBuilder() \
    .addGrid(logistic.maxIter,  
             [2, 10, 50]) \
    .addGrid(logistic.regParam, 
             [0.01, 0.05, 0.3]) \
    .build()
```

首先，我们指定我们想要优化参数的模型。接下来，我们决定我们将优化哪些参数，以及测试这些参数的哪些值。我们使用`.tuning`子包中的`ParamGridBuilder()`对象，并使用`.addGrid(...)`方法向网格中添加参数：第一个参数是我们想要优化的模型的参数对象（在我们的例子中，这些是`logistic.maxIter`和`logistic.regParam`），第二个参数是我们想要遍历的值的列表。在`.ParamGridBuilder`上调用`.build()`方法将构建网格。

接下来，我们需要一种比较模型的方法：

```py
evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='INFANT_ALIVE_AT_REPORT')
```

因此，我们再次使用 `BinaryClassificationEvaluator`。现在是时候创建一个逻辑，为我们执行验证工作：

```py
cv = tune.CrossValidator(
    estimator=logistic, 
    estimatorParamMaps=grid, 
    evaluator=evaluator
)
```

`CrossValidator` 需要估计器、估计器参数映射和评估器来完成其工作。模型遍历值网格，估计模型，并使用评估器比较它们的性能。

我们不能直接使用数据（因为 `births_train` 和 `births_test` 仍然有未编码的 `BIRTHS_PLACE` 列），因此我们创建了一个纯转换的 `Pipeline`：

```py
pipeline = Pipeline(stages=[encoder ,featuresCreator])
data_transformer = pipeline.fit(births_train)
```

完成这些后，我们就可以找到我们模型的最佳参数组合：

```py
cvModel = cv.fit(data_transformer.transform(births_train))
```

`cvModel` 将返回最佳估计模型。现在我们可以使用它来查看它是否比我们之前的模型表现更好：

```py
data_train = data_transformer \
    .transform(births_test)
results = cvModel.transform(data_train)
print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderPR'}))
```

上一段代码将产生以下结果：

![网格搜索](img/B05793_06_06.jpg)

如您所见，我们得到了一个稍微好一点的结果。最佳模型有哪些参数？答案是有点复杂，但以下是您可以提取它的方法：

```py
results = [
    (
        [
            {key.name: paramValue} 
            for key, paramValue 
            in zip(
                params.keys(), 
                params.values())
        ], metric
    ) 
    for params, metric 
    in zip(
        cvModel.getEstimatorParamMaps(), 
        cvModel.avgMetrics
    )
]
sorted(results, 
       key=lambda el: el[1], 
       reverse=True)[0]
```

上一段代码产生以下输出：

![网格搜索](img/B05793_06_07.jpg)

## 训练-验证分割

`TrainValidationSplit` 模型，为了选择最佳模型，将输入数据集（训练数据集）随机分割成两个子集：较小的训练集和验证集。分割只进行一次。

在这个例子中，我们还将使用 `ChiSqSelector` 来选择前五个特征，从而限制我们模型的复杂性：

```py
selector = ft.ChiSqSelector(
    numTopFeatures=5, 
    featuresCol=featuresCreator.getOutputCol(), 
    outputCol='selectedFeatures',
    labelCol='INFANT_ALIVE_AT_REPORT'
)
```

`numTopFeatures` 指定了要返回的特征数量。我们将选择器放在 `featuresCreator` 之后，因此我们在 `featuresCreator` 上调用 `.getOutputCol()`。

我们之前已经介绍了如何创建 `LogisticRegression` 和 `Pipeline`，因此我们在这里不再解释它们的创建方法：

```py
logistic = cl.LogisticRegression(
    labelCol='INFANT_ALIVE_AT_REPORT',
    featuresCol='selectedFeatures'
)
pipeline = Pipeline(stages=[encoder, featuresCreator, selector])
data_transformer = pipeline.fit(births_train)
```

`TrainValidationSplit` 对象的创建方式与 `CrossValidator` 模型相同：

```py
tvs = tune.TrainValidationSplit(
    estimator=logistic, 
    estimatorParamMaps=grid, 
    evaluator=evaluator
)
```

如前所述，我们将数据拟合到模型中，并计算结果：

```py
tvsModel = tvs.fit(
    data_transformer \
        .transform(births_train)
)
data_train = data_transformer \
    .transform(births_test)
results = tvsModel.transform(data_train)
print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(results, 
     {evaluator.metricName: 'areaUnderPR'}))
```

上一段代码输出了以下内容：

![训练-验证分割](img/B05793_06_08.jpg)

好吧，具有较少特征的模型肯定比完整模型表现差，但差距并不大。最终，这是一个在更复杂的模型和不太复杂的模型之间的性能权衡。

# PySpark ML 的其他功能实战

在本章的开头，我们描述了 PySpark ML 库的大部分功能。在本节中，我们将提供一些使用 `Transformers` 和 `Estimators` 的示例。

## 特征提取

我们已经使用了 PySpark 子模块中相当多的模型。在本节中，我们将向您展示如何使用我们认为最有用的模型。

### NLP 相关的特征提取器

如前所述，`NGram` 模型接受一个分词文本的列表，并生成单词对（或 n-gram）。

在这个例子中，我们将从 PySpark 的文档中摘取一段内容，展示在传递给`NGram`模型之前如何清理文本。以下是我们的数据集的样子（为了简洁而省略）：

### 小贴士

要查看以下代码片段的完整视图，请从我们的 GitHub 仓库下载代码：[`github.com/drabastomek/learningPySpark`](https://github.com/drabastomek/learningPySpark)。

我们从`Pipelines`中 DataFrame 使用的描述中复制了这四个段落：[`spark.apache.org/docs/latest/ml-pipeline.html#dataframe`](http://spark.apache.org/docs/latest/ml-pipeline.html#dataframe)。

```py
text_data = spark.createDataFrame([
    ['''Machine learning can be applied to a wide variety 
        of data types, such as vectors, text, images, and 
        structured data. This API adopts the DataFrame from 
        Spark SQL in order to support a variety of data
        types.'''],
    (...)
    ['''Columns in a DataFrame are named. The code examples 
        below use names such as "text," "features," and 
        "label."''']
], ['input'])
```

我们单列 DataFrame 中的每一行只是一堆文本。首先，我们需要对这段文本进行分词。为此，我们将使用`RegexTokenizer`而不是仅仅使用`Tokenizer`，因为我们可以指定我们想要文本在何处被分割的模式：

```py
tokenizer = ft.RegexTokenizer(
    inputCol='input', 
    outputCol='input_arr', 
    pattern='\s+|[,.\"]')
```

此处的模式在任意数量的空格处分割文本，同时也移除了逗号、句号、反斜杠和引号。`tokenizer`输出中的一个单行看起来类似于这样：

![NLP - 相关特征提取器](img/B05793_06_09.jpg)

如您所见，`RegexTokenizer`不仅将句子分割成单词，而且还对文本进行了规范化，使得每个单词都是小写。

然而，我们的文本中仍然有很多垃圾信息：例如`be`、`a`或`to`等单词在分析文本时通常不会提供任何有用的信息。因此，我们将使用`StopWordsRemover(...)`移除这些所谓的`停用词`：

```py
stopwords = ft.StopWordsRemover(
    inputCol=tokenizer.getOutputCol(), 
    outputCol='input_stop')
```

该方法的输出如下所示：

![NLP - 相关特征提取器](img/B05793_06_10.jpg)

现在我们只有有用的单词。所以，让我们构建我们的`NGram`模型和`Pipeline`：

```py
ngram = ft.NGram(n=2, 
    inputCol=stopwords.getOutputCol(), 
    outputCol="nGrams")
pipeline = Pipeline(stages=[tokenizer, stopwords, ngram])
```

现在我们有了`pipeline`，我们将按照之前非常相似的方式继续：

```py
data_ngram = pipeline \
    .fit(text_data) \
    .transform(text_data)
data_ngram.select('nGrams').take(1)
```

上述代码产生以下输出：

![NLP - 相关特征提取器](img/B05793_06_11.jpg)

就这样。我们已经得到了我们的 n-gram，现在我们可以将它们用于进一步的 NLP 处理。

### 离散化连续变量

我们经常处理一个高度非线性且很难只用一个系数来拟合模型中的连续特征。

在这种情况下，可能很难只用一个系数来解释这种特征与目标之间的关系。有时，将值分组到离散的桶中是有用的。

首先，让我们使用以下代码创建一些假数据：

```py
import numpy as np
x = np.arange(0, 100)
x = x / 100.0 * np.pi * 4
y = x * np.sin(x / 1.764) + 20.1234
```

现在，我们可以使用以下代码创建一个 DataFrame：

```py
schema = typ.StructType([
    typ.StructField('continuous_var', 
                    typ.DoubleType(), 
                    False
   )
])
data = spark.createDataFrame(
    [[float(e), ] for e in y], 
    schema=schema)
```

![离散化连续变量](img/B05793_06_12.jpg)

接下来，我们将使用`QuantileDiscretizer`模型将我们的连续变量分割成五个桶（`numBuckets`参数）：

```py
discretizer = ft.QuantileDiscretizer(
    numBuckets=5, 
    inputCol='continuous_var', 
    outputCol='discretized')
```

让我们看看我们得到了什么：

```py
data_discretized = discretizer.fit(data).transform(data)
```

我们的功能现在看起来如下：

![离散化连续变量](img/B05793_06_13.jpg)

我们现在可以将这个变量视为分类变量，并使用`OneHotEncoder`对其进行编码以供将来使用。

### 标准化连续变量

标准化连续变量不仅有助于更好地理解特征之间的关系（因为解释系数变得更容易），而且还有助于计算效率，并防止遇到一些数值陷阱。这是使用 PySpark ML 如何做到这一点。

首先，我们需要创建我们连续变量的向量表示（因为它只是一个单一的浮点数）：

```py
vectorizer = ft.VectorAssembler(
    inputCols=['continuous_var'], 
    outputCol= 'continuous_vec')
```

接下来，我们构建我们的`normalizer`和`pipeline`。通过将`withMean`和`withStd`设置为`True`，该方法将移除均值并将方差缩放到单位长度：

```py
normalizer = ft.StandardScaler(
    inputCol=vectorizer.getOutputCol(), 
    outputCol='normalized', 
    withMean=True,
    withStd=True
)
pipeline = Pipeline(stages=[vectorizer, normalizer])
data_standardized = pipeline.fit(data).transform(data)
```

这是转换后的数据看起来会是什么样子：

![标准化连续变量](img/B05793_06_14.jpg)

如你所见，数据现在围绕 0 振荡，具有单位方差（绿色线）。

## 分类

到目前为止，我们只使用了 PySpark ML 中的`LogisticRegression`模型。在本节中，我们将使用`RandomForestClassfier`再次对婴儿生存的机会进行建模。

在我们能够做到这一点之前，我们需要将`label`特征转换为`DoubleType`：

```py
import pyspark.sql.functions as func
births = births.withColumn(
    'INFANT_ALIVE_AT_REPORT', 
    func.col('INFANT_ALIVE_AT_REPORT').cast(typ.DoubleType())
)
births_train, births_test = births \
    .randomSplit([0.7, 0.3], seed=666)
```

现在我们已经将标签转换为双精度，我们准备好构建我们的模型。我们以类似的方式前进，区别在于我们将重用本章早些时候的`encoder`和`featureCreator`。`numTrees`参数指定我们的随机森林中应该有多少决策树，而`maxDepth`参数限制了树的深度：

```py
classifier = cl.RandomForestClassifier(
    numTrees=5, 
    maxDepth=5, 
    labelCol='INFANT_ALIVE_AT_REPORT')
pipeline = Pipeline(
    stages=[
        encoder,
        featuresCreator, 
        classifier])
model = pipeline.fit(births_train)
test = model.transform(births_test)
```

现在我们来看看`RandomForestClassifier`模型与`LogisticRegression`相比的表现：

```py
evaluator = ev.BinaryClassificationEvaluator(
    labelCol='INFANT_ALIVE_AT_REPORT')
print(evaluator.evaluate(test, 
    {evaluator.metricName: "areaUnderROC"}))
print(evaluator.evaluate(test, 
    {evaluator.metricName: "areaUnderPR"}))
```

我们得到以下结果：

![分类](img/B05793_06_15.jpg)

好吧，正如你所见，结果比逻辑回归模型好大约 3 个百分点。让我们测试一下只有一个树的模型表现如何：

```py
classifier = cl.DecisionTreeClassifier(
    maxDepth=5, 
    labelCol='INFANT_ALIVE_AT_REPORT')
pipeline = Pipeline(stages=[
    encoder,
    featuresCreator, 
    classifier])
model = pipeline.fit(births_train)
test = model.transform(births_test)
evaluator = ev.BinaryClassificationEvaluator(
    labelCol='INFANT_ALIVE_AT_REPORT')
print(evaluator.evaluate(test, 
    {evaluator.metricName: "areaUnderROC"}))
print(evaluator.evaluate(test, 
    {evaluator.metricName: "areaUnderPR"}))
```

上述代码给出了以下结果：

![分类](img/B05793_06_16.jpg)

表现相当不错！实际上，它在精确度-召回率关系方面表现得比随机森林模型更好，只是在 ROC 曲线下的面积上略差一些。我们可能已经找到了一个赢家！

## 聚类

聚类是机器学习的一个重要部分：在现实世界中，我们往往没有目标特征的奢侈，因此我们需要回到无监督学习范式，试图在数据中揭示模式。

### 在出生数据集中寻找聚类

在这个例子中，我们将使用`k-means`模型来寻找出生数据中的相似性：

```py
import pyspark.ml.clustering as clus
kmeans = clus.KMeans(k = 5, 
    featuresCol='features')
pipeline = Pipeline(stages=[
        assembler,
        featuresCreator, 
        kmeans]
)
model = pipeline.fit(births_train)
```

在估计了模型之后，让我们看看我们是否能在聚类之间找到一些差异：

```py
test = model.transform(births_test)
test \
    .groupBy('prediction') \
    .agg({
        '*': 'count', 
        'MOTHER_HEIGHT_IN': 'avg'
    }).collect()
```

上述代码产生了以下输出：

![在出生数据集中寻找聚类](img/B05793_06_17.jpg)

好吧，`MOTHER_HEIGHT_IN`在聚类 2 中显著不同。通过查看结果（这里我们不会做，很明显的原因）可能会发现更多差异，并帮助我们更好地理解数据。

### 主题挖掘

聚类模型不仅限于数值数据。在自然语言处理（NLP）领域，如主题提取等问题依赖于聚类来检测具有相似主题的文档。我们将通过一个这样的例子来讲解。

首先，让我们创建我们的数据集。数据是从互联网上随机选择的段落中形成的：其中三个涉及自然和国家公园的主题，剩下的三个涉及技术。

### 小贴士

由于显而易见的原因，代码片段再次被简化。请参考 GitHub 上的源文件以获取完整表示。

```py
text_data = spark.createDataFrame([
    ['''To make a computer do anything, you have to write a 
    computer program. To write a computer program, you have 
    to tell the computer, step by step, exactly what you want 
    it to do. The computer then "executes" the program, 
    following each step mechanically, to accomplish the end 
    goal. When you are telling the computer what to do, you 
    also get to choose how it's going to do it. That's where 
    computer algorithms come in. The algorithm is the basic 
    technique used to get the job done. Let's follow an 
    example to help get an understanding of the algorithm 
    concept.'''],
    (...),
    ['''Australia has over 500 national parks. Over 28 
    million hectares of land is designated as national 
    parkland, accounting for almost four per cent of 
    Australia's land areas. In addition, a further six per 
    cent of Australia is protected and includes state 
    forests, nature parks and conservation reserves.National 
    parks are usually large areas of land that are protected 
    because they have unspoilt landscapes and a diverse 
    number of native plants and animals. This means that 
    commercial activities such as farming are prohibited and 
    human activity is strictly monitored.''']
], ['documents'])
```

首先，我们再次使用`RegexTokenizer`和`StopWordsRemover`模型：

```py
tokenizer = ft.RegexTokenizer(
    inputCol='documents', 
    outputCol='input_arr', 
    pattern='\s+|[,.\"]')
stopwords = ft.StopWordsRemover(
    inputCol=tokenizer.getOutputCol(), 
    outputCol='input_stop')
```

在我们的管道中接下来的是`CountVectorizer`：这是一个计算文档中单词并返回计数向量的模型。向量的长度等于所有文档中不同单词的总数，这在以下代码片段中可以看到：

```py
stringIndexer = ft.CountVectorizer(
    inputCol=stopwords.getOutputCol(), 
    outputCol="input_indexed")
tokenized = stopwords \
    .transform(
        tokenizer\
            .transform(text_data)
    )

stringIndexer \
    .fit(tokenized)\
    .transform(tokenized)\
    .select('input_indexed')\
    .take(2)
```

上述代码将产生以下输出：

![主题挖掘](img/B05793_06_18.jpg)

如您所见，文本中有 262 个不同的单词，现在每个文档都由每个单词出现的计数来表示。

现在是时候开始预测主题了。为此，我们将使用`LDA`模型——**潜在狄利克雷分配**模型：

```py
clustering = clus.LDA(k=2, 
    optimizer='online', 
    featuresCol=stringIndexer.getOutputCol())
```

`k`参数指定了我们期望看到多少个主题，`optimizer`参数可以是`'online'`或`'em'`（后者代表期望最大化算法）。

将这些难题组合起来，到目前为止，是我们的管道中最长的：

```py
pipeline = ml.Pipeline(stages=[
        tokenizer, 
        stopwords,
        stringIndexer, 
        clustering]
)
```

我们是否正确地揭示了主题？好吧，让我们看看：

```py
topics = pipeline \
    .fit(text_data) \
    .transform(text_data)
topics.select('topicDistribution').collect()
```

我们得到以下结果：

![主题挖掘](img/B05793_06_19.jpg)

看起来我们的方法正确地发现了所有主题！但是，不要习惯看到这样的好结果：遗憾的是，现实世界的数据很少是这样的。

## 回归

我们在介绍机器学习库的章节中，如果不能构建一个回归模型，就无法完成。

在本节中，我们将尝试预测`MOTHER_WEIGHT_GAIN`，给定这里描述的一些特征；这些特征包含在以下列出的特征中：

```py
features = ['MOTHER_AGE_YEARS','MOTHER_HEIGHT_IN',
            'MOTHER_PRE_WEIGHT','DIABETES_PRE',
            'DIABETES_GEST','HYP_TENS_PRE', 
            'HYP_TENS_GEST', 'PREV_BIRTH_PRETERM',
            'CIG_BEFORE','CIG_1_TRI', 'CIG_2_TRI', 
            'CIG_3_TRI'
           ]
```

首先，由于所有特征都是数值的，我们将它们收集在一起，并使用`ChiSqSelector`来选择仅前六个最重要的特征：

```py
featuresCreator = ft.VectorAssembler(
    inputCols=[col for col in features[1:]], 
    outputCol='features'
)
selector = ft.ChiSqSelector(
    numTopFeatures=6, 
    outputCol="selectedFeatures", 
    labelCol='MOTHER_WEIGHT_GAIN'
)
```

为了预测体重增加，我们将使用梯度提升树回归器：

```py
import pyspark.ml.regression as reg
regressor = reg.GBTRegressor(
    maxIter=15, 
    maxDepth=3,
    labelCol='MOTHER_WEIGHT_GAIN')
```

最后，再次，我们将所有这些组合到一个`Pipeline`中：

```py
pipeline = Pipeline(stages=[
        featuresCreator, 
        selector,
        regressor])
weightGain = pipeline.fit(births_train)
```

在创建了`weightGain`模型之后，让我们看看它在测试数据上的表现如何：

```py
evaluator = ev.RegressionEvaluator(
    predictionCol="prediction", 
    labelCol='MOTHER_WEIGHT_GAIN')
print(evaluator.evaluate(
     weightGain.transform(births_test), 
    {evaluator.metricName: 'r2'}))
```

我们得到以下输出：

![回归](img/B05793_06_20.jpg)

很遗憾，模型的表现并不比掷硬币好。看起来，如果没有与`MOTHER_WEIGHT_GAIN`标签更好相关联的附加独立特征，我们将无法充分解释其变异性。

# 摘要

在本章中，我们详细介绍了如何使用 PySpark ML：PySpark 的官方主要机器学习库。我们解释了 `Transformer` 和 `Estimator` 是什么，并展示了它们在 ML 库中引入的另一个概念：`Pipeline` 中的作用。随后，我们还介绍了如何使用一些方法来微调模型的超参数。最后，我们给出了一些如何使用库中的一些特征提取器和模型的示例。

在下一章中，我们将深入探讨图论和 GraphFrames，这些工具有助于更好地以图的形式表示机器学习问题。
