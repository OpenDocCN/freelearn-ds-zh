# 第六章. 使用 MLlib 开始机器学习

本章分为以下食谱：

+   创建向量

+   创建一个标记的点

+   创建矩阵

+   计算汇总统计量

+   计算相关性

+   进行假设检验

+   使用 ML 创建机器学习管道

# 简介

以下是维基百科对机器学习的定义：

> *"机器学习是一个科学学科，它探索了构建和研究的算法，这些算法可以从数据中学习。"*

实质上，机器学习是利用过去的数据来对未来进行预测。机器学习高度依赖于统计分析和方法。

在统计学中，有四种类型的测量尺度：

| 尺度类型 | 描述 |
| --- | --- |
| 名义尺度 | =, ≠标识类别不能是数字示例：男性，女性 |
| 序数尺度 | =, ≠, <, >名义尺度 + 从最不重要到最重要的排名示例：公司等级 |
| 间隔尺度 | =, ≠, <, >, +, -间隔尺度 + 观测之间的距离分配给观测的数字表示顺序任何连续值之间的差异与其他值相同 60°的温度不是 30°的两倍 |
| 比例尺度 | =, ≠, <, >, +, ×, ÷间隔尺度 + 观测之间的比例$20 是$10 的两倍 |

在数据之间可以做出的另一个区分是连续数据和离散数据之间的区别。连续数据可以取任何值。属于间隔和比例尺度的多数数据是连续的。

离散变量只能取特定的值，并且值之间存在明确的界限。例如，一栋房子可以有二或三个房间，但不能有 2.75 个房间。属于名义和序数尺度的数据总是离散的。

MLlib 是 Spark 的机器学习库。在本章中，我们将重点关注机器学习的基础知识。

# 创建向量

在理解向量之前，让我们先关注什么是点。点只是一组数字。这组数字或坐标定义了点在空间中的位置。坐标的数量决定了空间的维度。

我们可以用最多三个维度来可视化空间。超过三个维度的空间被称为**超空间**。让我们将这个空间隐喻付诸实践。

让我们从一个人开始。一个人有以下维度：

+   重量

+   身高

+   年龄

我们在这里工作在三维空间中。因此，点（160,69,24）的解释将是 160 磅体重，69 英寸身高，24 岁年龄。

### 注意

点和向量是同一件事。向量中的维度被称为**特征**。另一种方式，我们可以将特征定义为一个现象的观察中的单个可测量属性。

Spark 有本地向量和矩阵，还有分布式矩阵。分布式矩阵由一个或多个 RDD 支持。本地向量具有数字索引和双精度值，并存储在单个机器上。

MLlib 中有两种类型的本地向量：密集和稀疏。密集向量由其值的数组支持，而稀疏向量由两个并行数组支持，一个用于索引，另一个用于值。

因此，人员数据（160,69,24）将使用密集向量表示为 [160.0,69.0,24.0]，使用稀疏向量格式表示为 (3,[0,1,2],[160.0,69.0,24.0])。

是否将向量制作成稀疏或密集，取决于它有多少空值或 0。让我们以一个包含 10,000 个值且其中 9,000 个值为 0 的向量的例子。如果我们使用密集向量格式，它将是一个简单的结构，但 90% 的空间将被浪费。稀疏向量格式在这里会更好，因为它只会保留非零的索引。

稀疏数据非常常见，Spark 支持用于它的 `libsvm` 格式，该格式每行存储一个特征向量。

## 如何做到这一点…

1.  启动 Spark shell：

    ```py
    $ spark-shell

    ```

1.  显式导入 MLlib 向量（不要与其他向量类混淆）：

    ```py
    Scala> import org.apache.spark.mllib.linalg.{Vectors,Vector}

    ```

1.  创建一个密集向量：

    ```py
    scala> val dvPerson = Vectors.dense(160.0,69.0,24.0)

    ```

1.  创建一个稀疏向量：

    ```py
    scala> val svPerson = Vectors.sparse(3,Array(0,1,2),Array(160.0,69.0,24.0))

    ```

## 它是如何工作的…

以下为 `vectors.dense` 方法的签名：

```py
def dense(values: Array[Double]): Vector
```

这里，值表示向量中元素的双精度数组。

以下为 `Vectors.sparse` 方法的签名：

```py
def sparse(size: Int, indices: Array[Int], values: Array[Double]): Vector
```

这里，`size` 表示向量的大小，`indices` 是索引数组，`values` 是值数组，作为双精度值。请确保您指定 `double` 作为数据类型或至少在一个值中使用小数；否则，它将为只有整数的数据集抛出异常。

# 创建一个标记点

标记点是一个带有相关标签的本地向量（稀疏/密集），在监督学习中用于帮助训练算法。你将在下一章中了解更多关于它的信息。

标签存储在 `LabeledPoint` 中的双精度值。这意味着当您有分类标签时，它们需要映射到双精度值。您分配给类别的值无关紧要，这只是方便的问题。

| 类型 | 标签值 |
| --- | --- |
| 二元分类 | 0 或 1 |
| 多类分类 | 0, 1, 2… |
| 回归 | 十进制值 |

## 如何做到这一点…

1.  启动 Spark shell：

    ```py
    $spark-shell

    ```

1.  显式导入 MLlib 向量：

    ```py
    scala> import org.apache.spark.mllib.linalg.{Vectors,Vector}

    ```

1.  导入 `LabeledPoint`：

    ```py
    scala> import org.apache.spark.mllib.regression.LabeledPoint

    ```

1.  创建一个带有正标签的标记点和密集向量：

    ```py
    scala> val willBuySUV = LabeledPoint(1.0,Vectors.dense(300.0,80,40))

    ```

1.  创建一个带有负标签的标记点和密集向量：

    ```py
    scala> val willNotBuySUV = LabeledPoint(0.0,Vectors.dense(150.0,60,25))

    ```

1.  创建一个带有正标签的标记点和稀疏向量：

    ```py
    scala> val willBuySUV = LabeledPoint(1.0,Vectors.sparse(3,Array(0,1,2),Array(300.0,80,40)))

    ```

1.  创建一个带有负标签的标记点和稀疏向量：

    ```py
    scala> val willNotBuySUV = LabeledPoint(0.0,Vectors.sparse(3,Array(0,1,2),Array(150.0,60,25)))

    ```

1.  创建一个包含相同数据的 `libsvm` 文件：

    ```py
    $vi person_libsvm.txt (libsvm indices start with 1)
    0  1:150 2:60 3:25
    1  1:300 2:80 3:40

    ```

1.  将 `person_libsvm.txt` 上传到 `hdfs`：

    ```py
    $ hdfs dfs -put person_libsvm.txt person_libsvm.txt

    ```

1.  进行更多导入：

    ```py
    scala> import org.apache.spark.mllib.util.MLUtils
    scala> import org.apache.spark.rdd.RDD

    ```

1.  从 `libsvm` 文件加载数据：

    ```py
    scala> val persons = MLUtils.loadLibSVMFile(sc,"person_libsvm.txt")

    ```

# 创建矩阵

矩阵是一个简单的表格，用于表示多个特征向量。可以存储在一台机器上的矩阵称为**本地矩阵**，而可以分布在整个集群上的矩阵称为**分布式矩阵**。

本地矩阵具有基于整数的索引，而分布式矩阵具有基于长整型的索引。两者都有双精度值。

分布式矩阵有三种类型：

+   `RowMatrix`：这每一行都是一个特征向量。

+   `IndexedRowMatrix`：这也有行索引。

+   `CoordinateMatrix`：这是一个简单的 `MatrixEntry` 矩阵。`MatrixEntry` 代表矩阵中的一个条目，由其行和列索引表示。

## 如何做到这一点…

1.  启动 Spark shell：

    ```py
    $spark-shell

    ```

1.  导入矩阵相关类：

    ```py
    scala> import org.apache.spark.mllib.linalg.{Vectors,Matrix, Matrices}

    ```

1.  创建一个密集的本地矩阵：

    ```py
    scala> val people = Matrices.dense(3,2,Array(150d,60d,25d, 300d,80d,40d))

    ```

1.  创建一个 `personRDD` 作为向量的 RDD：

    ```py
    scala> val personRDD = sc.parallelize(List(Vectors.dense(150,60,25), Vectors.dense(300,80,40)))

    ```

1.  导入 `RowMatrix` 和相关类：

    ```py
    scala> import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix,RowMatrix, CoordinateMatrix, MatrixEntry}

    ```

1.  创建 `personRDD` 的行矩阵：

    ```py
    scala> val personMat = new RowMatrix(personRDD)

    ```

1.  打印行数：

    ```py
    scala> print(personMat.numRows)

    ```

1.  打印列数：

    ```py
    scala> print(personMat.numCols)

    ```

1.  创建一个索引行的 RDD：

    ```py
    scala> val personRDD = sc.parallelize(List(IndexedRow(0L, Vectors.dense(150,60,25)), IndexedRow(1L, Vectors.dense(300,80,40))))

    ```

1.  创建一个索引行矩阵：

    ```py
    scala> val pirmat = new IndexedRowMatrix(personRDD)

    ```

1.  打印行数：

    ```py
    scala> print(pirmat.numRows)

    ```

1.  打印列数：

    ```py
    scala> print(pirmat.numCols)

    ```

1.  将索引行矩阵转换回行矩阵：

    ```py
    scala> val personMat = pirmat.toRowMatrix

    ```

1.  创建一个矩阵条目的 RDD：

    ```py
    scala> val meRDD = sc.parallelize(List(
     MatrixEntry(0,0,150),
     MatrixEntry(1,0,60),
    MatrixEntry(2,0,25),
    MatrixEntry(0,1,300),
    MatrixEntry(1,1,80),
    MatrixEntry(2,1,40)
    ))

    ```

1.  创建一个坐标矩阵：

    ```py
    scala> val pcmat = new CoordinateMatrix(meRDD)

    ```

1.  打印行数：

    ```py
    scala> print(pcmat.numRows)

    ```

1.  打印列数：

    ```py
    scala> print(pcmat.numCols)

    ```

# 计算摘要统计

摘要统计用于总结观察结果，以获得数据的整体感觉。摘要包括以下内容：

+   数据的中心趋势——均值、众数、中位数

+   数据的分布——方差、标准差

+   边界条件——最小值、最大值

这个菜谱涵盖了如何生成摘要统计。

## 如何做到这一点…

1.  启动 Spark shell：

    ```py
    $ spark-shell

    ```

1.  导入矩阵相关类：

    ```py
    scala> import org.apache.spark.mllib.linalg.{Vectors,Vector}
    scala> import org.apache.spark.mllib.stat.Statistics

    ```

1.  创建一个 `personRDD` 作为向量的 RDD：

    ```py
    scala> val personRDD = sc.parallelize(List(Vectors.dense(150,60,25), Vectors.dense(300,80,40)))

    ```

1.  计算列的摘要统计：

    ```py
    scala> val summary = Statistics.colStats(personRDD)

    ```

1.  打印这个摘要的平均值：

    ```py
    scala> print(summary.mean)

    ```

1.  打印方差：

    ```py
    scala> print(summary.variance)

    ```

1.  打印每列的非零值：

    ```py
    scala> print(summary.numNonzeros)

    ```

1.  打印样本大小：

    ```py
    scala> print(summary.count)

    ```

1.  打印每列的最大值：

    ```py
    scala> print(summary.max)

    ```

# 计算相关性

相关性是两个变量之间的统计关系，当一个变量变化时，会导致另一个变量的变化。相关性分析衡量两个变量相关性的程度。

如果一个变量的增加导致另一个变量的增加，这被称为**正相关**。如果一个变量的增加导致另一个变量的减少，这被称为**负相关**。

Spark 支持两种相关算法：皮尔逊和斯皮尔曼。皮尔逊算法适用于两个连续变量，例如一个人的身高和体重或房屋大小和房屋价格。斯皮尔曼处理一个连续变量和一个分类变量，例如邮编和房屋价格。

## 准备工作

让我们使用一些真实数据，这样我们才能更有意义地计算相关性。以下是美国加利福尼亚州萨拉托加市 2014 年初的房屋面积和价格：

| 房屋面积（平方英尺） | 价格 |
| --- | --- |
| 2100 | $1,620,000 |
| 2300 | $1,690,000 |
| 2046 | $1,400,000 |
| 4314 | $2,000,000 |
| 1244 | $1,060,000 |
| 4608 | $3,830,000 |
| 2173 | $1,230,000 |
| 2750 | $2,400,000 |
| 4010 | $3,380,000 |
| 1959 | $1,480,000 |

## 如何做到这一点…

1.  启动 Spark shell：

    ```py
    $ spark-shell

    ```

1.  导入统计和相关类：

    ```py
    scala> import org.apache.spark.mllib.linalg._
    scala> import org.apache.spark.mllib.stat.Statistics

    ```

1.  创建一个房屋面积的 RDD：

    ```py
    scala> val sizes = sc.parallelize(List(2100, 2300, 2046, 4314, 1244, 4608, 2173, 2750, 4010, 1959.0))

    ```

1.  创建一个房屋价格的 RDD：

    ```py
    scala> val prices = sc.parallelize(List(1620000 , 1690000, 1400000, 2000000, 1060000, 3830000, 1230000, 2400000, 3380000, 1480000.00))

    ```

1.  计算相关性：

    ```py
    scala> val correlation = Statistics.corr(sizes,prices)
    correlation: Double = 0.8577177736252577 

    ```

    `0.85` 表示非常强的正相关。

    由于这里没有特定的算法，默认使用皮尔逊相关系数。`corr` 方法被重载，可以接受算法名称作为第三个参数。

1.  使用皮尔逊相关系数计算相关性：

    ```py
    scala> val correlation = Statistics.corr(sizes,prices)

    ```

1.  使用斯皮尔曼相关系数计算相关性：

    ```py
    scala> val correlation = Statistics.corr(sizes,prices,"spearman")

    ```

在前面的例子中，两个变量都是连续的，因此斯皮尔曼假设大小是离散的。斯皮尔曼的一个更好的例子可能是邮编与价格。

# 进行假设检验

假设检验是一种确定给定假设是否为真的概率的方法。假设样本数据表明女性倾向于为民主党投票。这种模式对于更大的人群来说可能或可能不是真的。如果这种模式仅仅因为偶然出现在样本数据中怎么办？

另一种看待假设检验目标的方式是回答这个问题：如果一个样本中存在某种模式，那么这种模式仅仅因为偶然出现的概率是多少？

我们如何操作？有句话说，最好的证明方式就是试图证明其相反。

要证伪的假设被称为**零假设**。假设检验与分类数据一起工作。让我们看看政党归属的民意调查的例子。

| 党派 | 男性 | 女性 |
| --- | --- | --- |
| 民主党派 | 32 | 41 |
| 共和党派 | 28 | 25 |
| 独立党派 | 34 | 26 |

## 如何操作...

1.  启动 Spark shell：

    ```py
    $ spark-shell

    ```

1.  导入相关类：

    ```py
    scala> import org.apache.spark.mllib.stat.Statistics
    scala> import org.apache.spark.mllib.linalg.{Vector,Vectors}
    scala> import org.apache.spark.mllib.linalg.{Matrix, Matrices}

    ```

1.  为民主党派创建一个向量：

    ```py
    scala> val dems = Vectors.dense(32.0,41.0)

    ```

1.  为共和党派创建一个向量：

    ```py
    scala> val reps= Vectors.dense(28.0,25.0)

    ```

1.  为独立党派创建一个向量：

    ```py
    scala> val indies = Vectors.dense(34.0,26.0)

    ```

1.  对观察数据与均匀分布进行卡方拟合优度检验：

    ```py
    scala> val dfit = Statistics.chiSqTest(dems)
    scala> val rfit = Statistics.chiSqTest(reps)
    scala> val ifit = Statistics.chiSqTest(indies)

    ```

1.  打印拟合优度结果：

    ```py
    scala> print(dfit)
    scala> print(rfit)
    scala> print(ifit)

    ```

1.  创建输入矩阵：

    ```py
    scala> val mat = Matrices.dense(2,3,Array(32.0,41.0, 28.0,25.0, 34.0,26.0))

    ```

1.  进行卡方独立性检验：

    ```py
    scala> val in = Statistics.chiSqTest(mat)

    ```

1.  打印独立性检验结果：

    ```py
    scala> print(in)

    ```

# 使用 ML 创建机器学习管道

Spark ML 是 Spark 中用于构建机器学习管道的新库。这个库与 MLlib 一起开发。它有助于将多个机器学习算法组合成一个单一的管道，并使用 DataFrame 作为数据集。

## 准备工作

让我们先了解 Spark ML 中的一些基本概念。它使用转换器将一个 DataFrame 转换为另一个 DataFrame。一个简单的转换示例可以是添加一个列。你可以将其视为关系世界中“alter table”的等价物。

估计器另一方面代表机器学习算法，它从数据中学习。估计器的输入是 DataFrame，输出是转换器。每个估计器都有一个 `fit()` 方法，用于训练算法。

机器学习管道被定义为一系列阶段；每个阶段可以是估计器或转换器。

我们将要在这个食谱中使用的例子是某人是否是篮球运动员。为此，我们将有一个包含一个估计器和一个转换器的管道。

估计器获取训练数据以训练算法，然后转换器进行预测。

目前，假设我们使用的是机器学习算法`LogisticRegression`。我们将在后续章节中解释`LogisticRegression`以及其他算法的细节。

## 如何操作...

1.  启动 Spark shell：

    ```py
    $ spark-shell

    ```

1.  导入必要的库：

    ```py
    scala> import org.apache.spark.mllib.linalg.{Vector,Vectors}
    scala> import org.apache.spark.mllib.regression.LabeledPoint
    scala> import org.apache.spark.ml.classification.LogisticRegression

    ```

1.  为是一名篮球运动员、身高 80 英寸、体重 250 磅的勒布朗创建一个标记点：

    ```py
    scala> val lebron = LabeledPoint(1.0,Vectors.dense(80.0,250.0))

    ```

1.  为不是篮球运动员、身高 70 英寸、体重 150 磅的蒂姆创建一个标记点：

    ```py
    scala> val tim = LabeledPoint(0.0,Vectors.dense(70.0,150.0))

    ```

1.  为是一名篮球运动员、身高 80 英寸、体重 207 磅的布里特尼创建一个标记点：

    ```py
    scala> val brittany = LabeledPoint(1.0,Vectors.dense(80.0,207.0))

    ```

1.  为不是篮球运动员、身高 65 英寸、体重 120 磅的斯泰西创建一个标记点：

    ```py
    scala> val stacey = LabeledPoint(0.0,Vectors.dense(65.0,120.0))

    ```

1.  创建一个训练 RDD：

    ```py
    scala> val trainingRDD = sc.parallelize(List(lebron,tim,brittany,stacey))

    ```

1.  创建一个训练 DataFrame：

    ```py
    scala> val trainingDF = trainingRDD.toDF

    ```

1.  创建一个`LogisticRegression`估计器：

    ```py
    scala> val estimator = new LogisticRegression

    ```

1.  通过拟合估计器与训练 DataFrame 来创建一个转换器：

    ```py
    scala> val transformer = estimator.fit(trainingDF)

    ```

1.  现在，让我们创建一些测试数据——约翰身高 90 英寸，体重 270 磅，是一名篮球运动员：

    ```py
    scala> val john = Vectors.dense(90.0,270.0)

    ```

1.  创建另一组测试数据——汤姆身高 62 英寸，体重 150 磅，不是篮球运动员：

    ```py
    scala> val tom = Vectors.dense(62.0,120.0)

    ```

1.  创建一个训练 RDD：

    ```py
    scala> val testRDD = sc.parallelize(List(john,tom))

    ```

1.  创建一个`Features`案例类：

    ```py
    scala> case class Feature(v:Vector)

    ```

1.  将`testRDD`映射到一个`Features`的 RDD：

    ```py
    scala> val featuresRDD = testRDD.map( v => Feature(v))

    ```

1.  将`featuresRDD`转换为具有列名`"features"`的 DataFrame：

    ```py
    scala> val featuresDF = featuresRDD.toDF("features")

    ```

1.  通过向`featuresDF`添加`predictions`列来转换`featuresDF`：

    ```py
    scala> val predictionsDF = transformer.transform(featuresDF)

    ```

1.  打印`predictionsDF`：

    ```py
    scala> predictionsDF.foreach(println)

    ```

1.  如您所见，`PredictionDF`除了保留特征外，还创建了三个列——`rawPrediction`、`probability`和`prediction`。让我们只选择`features`和`prediction`：

    ```py
    scala> val shorterPredictionsDF = predictionsDF.select("features","prediction")

    ```

1.  将预测结果重命名为`isBasketBallPlayer`：

    ```py
    scala> val playerDF = shorterPredictionsDF.toDF("features","isBasketBallPlayer")

    ```

1.  打印`playerDF`的 schema：

    ```py
    scala> playerDF.printSchema

    ```
