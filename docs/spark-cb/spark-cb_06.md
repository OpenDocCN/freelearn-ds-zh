# 第六章：使用 MLlib 开始机器学习

本章分为以下配方：

+   创建向量

+   创建标记点

+   创建矩阵

+   计算摘要统计信息

+   计算相关性

+   进行假设检验

+   使用 ML 创建机器学习管道

# 介绍

以下是维基百科对机器学习的定义：

> *"机器学习是一门探索从数据中学习的算法的构建和研究的科学学科。"*

基本上，机器学习是利用过去的数据来预测未来。机器学习在很大程度上依赖于统计分析和方法。

在统计学中，有四种测量标度：

| 规模类型 | 描述 |
| --- | --- |
| 名义标度 | =，≠识别类别不能是数字示例：男性，女性 |
| 序数标度 | =，≠，<，>名义标度+从最不重要到最重要的排名示例：公司等级制度 |
| 间隔标度 | =，≠，<，>，+，-序数标度+观察之间的距离分配的数字指示顺序任何连续值之间的差异与其他值相同 60°温度不是 30°的两倍 |
| 比例标度 | =，≠，<，>，+，×，÷间隔标度+观察的比率$20 是$10 的两倍 |

数据之间可以进行的另一个区分是连续数据和离散数据。连续数据可以取任何值。大多数属于间隔和比例标度的数据是连续的。

离散变量只能取特定的值，值之间有明确的界限。例如，一所房子可以有两间或三间房间，但不能有 2.75 间。属于名义和序数标度的数据始终是离散的。

MLlib 是 Spark 的机器学习库。在本章中，我们将专注于机器学习的基础知识。

# 创建向量

在了解向量之前，让我们专注于点是什么。一个点只是一组数字。这组数字或坐标定义了点在空间中的位置。坐标的数量确定了空间的维度。

我们可以用最多三个维度来可视化空间。具有三个以上维度的空间称为**超空间**。让我们利用这个空间的隐喻。

让我们从一个人开始。一个人具有以下维度：

+   重量

+   身高

+   年龄

我们在三维空间中工作。因此，点（160,69,24）的解释将是 160 磅的体重，69 英寸的身高和 24 岁的年龄。

### 注意

点和向量是同一回事。向量中的维度称为**特征**。换句话说，我们可以将特征定义为被观察现象的个体可测属性。

Spark 有本地向量和矩阵，还有分布式矩阵。分布式矩阵由一个或多个 RDD 支持。本地向量具有数字索引和双值，并存储在单台机器上。

MLlib 中有两种本地向量：密集和稀疏。密集向量由其值的数组支持，而稀疏向量由两个并行数组支持，一个用于索引，另一个用于值。

因此，人的数据（160,69,24）将使用密集向量表示为[160.0,69.0,24.0]，使用稀疏向量格式表示为（3，[0,1,2]，[160.0,69.0,24.0]）。

是将向量稀疏还是密集取决于它有多少空值或 0。让我们以一个包含 10,000 个值的向量为例，其中有 9,000 个值为 0。如果我们使用密集向量格式，它将是一个简单的结构，但会浪费 90%的空间。稀疏向量格式在这里会更好，因为它只保留非零的索引。

稀疏数据非常常见，Spark 支持`libsvm`格式，该格式每行存储一个特征向量。

## 如何做…

1.  启动 Spark shell：

```scala
$ spark-shell

```

1.  显式导入 MLlib 向量（不要与其他向量类混淆）：

```scala
Scala> import org.apache.spark.mllib.linalg.{Vectors,Vector}

```

1.  创建密集向量：

```scala
scala> val dvPerson = Vectors.dense(160.0,69.0,24.0)

```

1.  创建稀疏向量：

```scala
scala> val svPerson = Vectors.sparse(3,Array(0,1,2),Array(160.0,69.0,24.0))

```

## 它是如何工作的...

以下是`vectors.dense`的方法签名：

```scala
def dense(values: Array[Double]): Vector
```

这里，值表示向量中元素的双精度数组。

以下是`Vectors.sparse`的方法签名：

```scala
def sparse(size: Int, indices: Array[Int], values: Array[Double]): Vector
```

这里，`size`表示向量的大小，`indices`是索引数组，`values`是双精度值数组。确保您指定`double`作为数据类型，或者至少在一个值中使用十进制；否则，对于只有整数的数据集，它将抛出异常。

# 创建一个带标签的点

带标签的点是一个带有相关标签的本地向量（稀疏/密集），在监督学习中用于帮助训练算法。您将在下一章中了解更多相关信息。

标签以双精度值存储在`LabeledPoint`中。这意味着当您有分类标签时，它们需要被映射为双精度值。您分配给类别的值是无关紧要的，只是一种便利。

| 类型 | 标签值 |
| --- | --- |
| 二元分类 | 0 或 1 |
| 多类分类 | 0, 1, 2… |
| 回归 | 十进制值 |

## 如何做…

1.  启动 Spark shell：

```scala
$spark-shell

```

1.  显式导入 MLlib 向量：

```scala
scala> import org.apache.spark.mllib.linalg.{Vectors,Vector}

```

1.  导入`LabeledPoint`：

```scala
scala> import org.apache.spark.mllib.regression.LabeledPoint

```

1.  使用正标签和密集向量创建一个带标签的点：

```scala
scala> val willBuySUV = LabeledPoint(1.0,Vectors.dense(300.0,80,40))

```

1.  使用负标签和密集向量创建一个带标签的点：

```scala
scala> val willNotBuySUV = LabeledPoint(0.0,Vectors.dense(150.0,60,25))

```

1.  使用正标签和稀疏向量创建一个带标签的点：

```scala
scala> val willBuySUV = LabeledPoint(1.0,Vectors.sparse(3,Array(0,1,2),Array(300.0,80,40)))

```

1.  使用负标签和稀疏向量创建一个带标签的点：

```scala
scala> val willNotBuySUV = LabeledPoint(0.0,Vectors.sparse(3,Array(0,1,2),Array(150.0,60,25)))

```

1.  创建一个包含相同数据的`libsvm`文件：

```scala
$vi person_libsvm.txt (libsvm indices start with 1)
0  1:150 2:60 3:25
1  1:300 2:80 3:40

```

1.  将`person_libsvm.txt`上传到`hdfs`：

```scala
$ hdfs dfs -put person_libsvm.txt person_libsvm.txt

```

1.  做更多的导入：

```scala
scala> import org.apache.spark.mllib.util.MLUtils
scala> import org.apache.spark.rdd.RDD

```

1.  从`libsvm`文件加载数据：

```scala
scala> val persons = MLUtils.loadLibSVMFile(sc,"person_libsvm.txt")

```

# 创建矩阵

矩阵只是一个表示多个特征向量的表。可以存储在一台机器上的矩阵称为**本地矩阵**，可以分布在集群中的矩阵称为**分布式矩阵**。

本地矩阵具有基于整数的索引，而分布式矩阵具有基于长整数的索引。两者的值都是双精度。

有三种类型的分布式矩阵：

+   `RowMatrix`：每行都是一个特征向量。

+   `IndexedRowMatrix`：这也有行索引。

+   `CoordinateMatrix`：这只是一个`MatrixEntry`的矩阵。`MatrixEntry`表示矩阵中的一个条目，由其行和列索引表示。

## 如何做…

1.  启动 Spark shell：

```scala
$spark-shell

```

1.  导入与矩阵相关的类：

```scala
scala> import org.apache.spark.mllib.linalg.{Vectors,Matrix, Matrices}

```

1.  创建一个密集的本地矩阵：

```scala
scala> val people = Matrices.dense(3,2,Array(150d,60d,25d, 300d,80d,40d))

```

1.  创建一个`personRDD`作为向量的 RDD：

```scala
scala> val personRDD = sc.parallelize(List(Vectors.dense(150,60,25), Vectors.dense(300,80,40)))

```

1.  导入`RowMatrix`和相关类：

```scala
scala> import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix,RowMatrix, CoordinateMatrix, MatrixEntry}

```

1.  创建一个`personRDD`的行矩阵：

```scala
scala> val personMat = new RowMatrix(personRDD)

```

1.  打印行数：

```scala
scala> print(personMat.numRows)

```

1.  打印列数：

```scala
scala> print(personMat.numCols)

```

1.  创建一个索引行的 RDD：

```scala
scala> val personRDD = sc.parallelize(List(IndexedRow(0L, Vectors.dense(150,60,25)), IndexedRow(1L, Vectors.dense(300,80,40))))

```

1.  创建一个索引行矩阵：

```scala
scala> val pirmat = new IndexedRowMatrix(personRDD)

```

1.  打印行数：

```scala
scala> print(pirmat.numRows)

```

1.  打印列数：

```scala
scala> print(pirmat.numCols)

```

1.  将索引行矩阵转换回行矩阵：

```scala
scala> val personMat = pirmat.toRowMatrix

```

1.  创建一个矩阵条目的 RDD：

```scala
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

```scala
scala> val pcmat = new CoordinateMatrix(meRDD)

```

1.  打印行数：

```scala
scala> print(pcmat.numRows)

```

1.  打印列数：

```scala
scala> print(pcmat.numCols)

```

# 计算摘要统计

汇总统计用于总结观察结果，以获得对数据的整体感觉。摘要包括以下内容：

+   数据的中心趋势-均值、众数、中位数

+   数据的分布-方差、标准差

+   边界条件-最小值、最大值

这个示例介绍了如何生成摘要统计信息。

## 如何做…

1.  启动 Spark shell：

```scala
$ spark-shell

```

1.  导入与矩阵相关的类：

```scala
scala> import org.apache.spark.mllib.linalg.{Vectors,Vector}
scala> import org.apache.spark.mllib.stat.Statistics

```

1.  创建一个`personRDD`作为向量的 RDD：

```scala
scala> val personRDD = sc.parallelize(List(Vectors.dense(150,60,25), Vectors.dense(300,80,40)))

```

1.  计算列的摘要统计：

```scala
scala> val summary = Statistics.colStats(personRDD)

```

1.  打印这个摘要的均值：

```scala
scala> print(summary.mean)

```

1.  打印方差：

```scala
scala> print(summary.variance)

```

1.  打印每列中非零值的数量：

```scala
scala> print(summary.numNonzeros)

```

1.  打印样本大小：

```scala
scala> print(summary.count)

```

1.  打印每列的最大值：

```scala
scala> print(summary.max)

```

# 计算相关性

相关性是两个变量之间的统计关系，当一个变量改变时，会导致另一个变量的改变。相关性分析衡量了这两个变量相关的程度。

如果一个变量的增加导致另一个变量的增加，这被称为**正相关**。如果一个变量的增加导致另一个变量的减少，这是**负相关**。

Spark 支持两种相关算法：Pearson 和 Spearman。Pearson 算法适用于两个连续变量，例如一个人的身高和体重或房屋大小和房价。Spearman 处理一个连续和一个分类变量，例如邮政编码和房价。

## 准备就绪

让我们使用一些真实数据，这样我们可以更有意义地计算相关性。以下是 2014 年初加利福尼亚州萨拉托加市房屋的大小和价格：

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

## 如何做…

1.  启动 Spark shell：

```scala
$ spark-shell

```

1.  导入统计和相关类：

```scala
scala> import org.apache.spark.mllib.linalg._
scala> import org.apache.spark.mllib.stat.Statistics

```

1.  创建一个房屋面积的 RDD：

```scala
scala> val sizes = sc.parallelize(List(2100, 2300, 2046, 4314, 1244, 4608, 2173, 2750, 4010, 1959.0))

```

1.  创建一个房价的 RDD：

```scala
scala> val prices = sc.parallelize(List(1620000 , 1690000, 1400000, 2000000, 1060000, 3830000, 1230000, 2400000, 3380000, 1480000.00))

```

1.  计算相关性：

```scala
scala> val correlation = Statistics.corr(sizes,prices)
correlation: Double = 0.8577177736252577 

```

`0.85` 表示非常强的正相关性。

由于这里没有特定的算法，所以默认是 Pearson。`corr`方法被重载以将算法名称作为第三个参数。

1.  用 Pearson 计算相关性：

```scala
scala> val correlation = Statistics.corr(sizes,prices)

```

1.  用 Spearman 计算相关性：

```scala
scala> val correlation = Statistics.corr(sizes,prices,"spearman")

```

在前面的例子中，两个变量都是连续的，所以 Spearman 假设大小是离散的。Spearman 使用的更好的例子是邮政编码与价格。

# 进行假设检验

假设检验是确定给定假设为真的概率的一种方法。假设一个样本数据表明女性更倾向于投票给民主党。这可能对更大的人口来说是真的，也可能不是。如果这个模式只是样本数据中的偶然现象呢？

观察假设检验目标的另一种方式是回答这个问题：如果一个样本中有一个模式，那么这个模式存在的机会是多少？

我们怎么做？有一句话说，证明某事最好的方法是试图证伪它。

要证伪的假设被称为**零假设**。假设检验适用于分类数据。让我们看一个党派倾向的民意调查的例子。

| 党派 | 男性 | 女性 |
| --- | --- | --- |
| 民主党 | 32 | 41 |
| 共和党 | 28 | 25 |
| 独立 | 34 | 26 |

## 如何做…

1.  启动 Spark shell：

```scala
$ spark-shell

```

1.  导入相关的类：

```scala
scala> import org.apache.spark.mllib.stat.Statistics
scala> import org.apache.spark.mllib.linalg.{Vector,Vectors}
scala> import org.apache.spark.mllib.linalg.{Matrix, Matrices}

```

1.  为民主党创建一个向量：

```scala
scala> val dems = Vectors.dense(32.0,41.0)

```

1.  为共和党创建一个向量：

```scala
scala> val reps= Vectors.dense(28.0,25.0)

```

1.  为独立党创建一个向量：

```scala
scala> val indies = Vectors.dense(34.0,26.0)

```

1.  对观察数据进行卡方拟合度检验：

```scala
scala> val dfit = Statistics.chiSqTest(dems)
scala> val rfit = Statistics.chiSqTest(reps)
scala> val ifit = Statistics.chiSqTest(indies)

```

1.  打印拟合度检验结果：

```scala
scala> print(dfit)
scala> print(rfit)
scala> print(ifit)

```

1.  创建输入矩阵：

```scala
scala> val mat = Matrices.dense(2,3,Array(32.0,41.0, 28.0,25.0, 34.0,26.0))

```

1.  进行卡方独立性检验：

```scala
scala> val in = Statistics.chiSqTest(mat)

```

1.  打印独立性检验结果：

```scala
scala> print(in)

```

# 使用 ML 创建机器学习管道

Spark ML 是 Spark 中构建机器学习管道的新库。这个库正在与 MLlib 一起开发。它有助于将多个机器学习算法组合成一个单一的管道，并使用 DataFrame 作为数据集。

## 准备就绪

让我们首先了解一些 Spark ML 中的基本概念。它使用转换器将一个 DataFrame 转换为另一个 DataFrame。简单转换的一个例子可以是追加列。你可以把它看作是关系世界中的"alter table"的等价物。

另一方面，估计器代表一个机器学习算法，它从数据中学习。估计器的输入是一个 DataFrame，输出是一个转换器。每个估计器都有一个`fit()`方法，它的工作是训练算法。

机器学习管道被定义为一系列阶段；每个阶段可以是估计器或者转换器。

我们在这个示例中要使用的例子是某人是否是篮球运动员。为此，我们将有一个估计器和一个转换器的管道。

估计器获取训练数据来训练算法，然后转换器进行预测。

暂时假设`LogisticRegression`是我们正在使用的机器学习算法。我们将在随后的章节中解释`LogisticRegression`的细节以及其他算法。

## 如何做…

1.  启动 Spark shell：

```scala
$ spark-shell

```

1.  进行导入：

```scala
scala> import org.apache.spark.mllib.linalg.{Vector,Vectors}
scala> import org.apache.spark.mllib.regression.LabeledPoint
scala> import org.apache.spark.ml.classification.LogisticRegression

```

1.  为篮球运动员 Lebron 创建一个标记点，身高 80 英寸，体重 250 磅：

```scala
scala> val lebron = LabeledPoint(1.0,Vectors.dense(80.0,250.0))

```

1.  为不是篮球运动员的 Tim 创建一个标记点，身高 70 英寸，体重 150 磅：

```scala
scala> val tim = LabeledPoint(0.0,Vectors.dense(70.0,150.0))

```

1.  为篮球运动员 Brittany 创建一个标记点，身高 80 英寸，体重 207 磅：

```scala
scala> val brittany = LabeledPoint(1.0,Vectors.dense(80.0,207.0))

```

1.  为不是篮球运动员的 Stacey 创建一个标记点，身高 65 英寸，体重 120 磅：

```scala
scala> val stacey = LabeledPoint(0.0,Vectors.dense(65.0,120.0))

```

1.  创建一个训练 RDD：

```scala
scala> val trainingRDD = sc.parallelize(List(lebron,tim,brittany,stacey))

```

1.  创建一个训练 DataFrame：

```scala
scala> val trainingDF = trainingRDD.toDF

```

1.  创建一个`LogisticRegression`估计器：

```scala
scala> val estimator = new LogisticRegression

```

1.  通过拟合训练 DataFrame 来创建一个转换器：

```scala
scala> val transformer = estimator.fit(trainingDF)

```

1.  现在，让我们创建一个测试数据—John 身高 90 英寸，体重 270 磅，是篮球运动员：

```scala
scala> val john = Vectors.dense(90.0,270.0)

```

1.  创建另一个测试数据—Tom 身高 62 英寸，体重 150 磅，不是篮球运动员：

```scala
scala> val tom = Vectors.dense(62.0,120.0)

```

1.  创建一个训练 RDD：

```scala
scala> val testRDD = sc.parallelize(List(john,tom))

```

1.  创建一个`Features` case 类：

```scala
scala> case class Feature(v:Vector)

```

1.  将`testRDD`映射到`Features`的 RDD：

```scala
scala> val featuresRDD = testRDD.map( v => Feature(v))

```

1.  将`featuresRDD`转换为具有列名`"features"`的 DataFrame：

```scala
scala> val featuresDF = featuresRDD.toDF("features")

```

1.  通过向其添加`predictions`列来转换`featuresDF`：

```scala
scala> val predictionsDF = transformer.transform(featuresDF)

```

1.  打印`predictionsDF`：

```scala
scala> predictionsDF.foreach(println)

```

1.  `PredictionDF`，如您所见，除了保留特征之外，还创建了三列—`rawPrediction`、`probability`和`prediction`。让我们只选择`features`和`prediction`：

```scala
scala> val shorterPredictionsDF = predictionsDF.select("features","prediction")

```

1.  将预测重命名为`isBasketBallPlayer`：

```scala
scala> val playerDF = shorterPredictionsDF.toDF("features","isBasketBallPlayer")

```

1.  打印`playerDF`的模式：

```scala
scala> playerDF.printSchema

```
