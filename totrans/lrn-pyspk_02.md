# 第 2 章。弹性分布式数据集

弹性分布式数据集（RDDs）是一组不可变的 JVM 对象的分布式集合，允许您非常快速地进行计算，它们是 Apache Spark 的 *骨干*。

正如其名所示，数据集是分布式的；它根据某些键分割成块，并分布到执行节点。这样做可以非常快速地运行针对此类数据集的计算。此外，如[第 1 章](ch01.html "第 1 章。理解 Spark")中已提到的，“理解 Spark”，RDD 会跟踪（记录）对每个块应用的所有转换，以加快计算并提供回退机制，以防出现错误且该部分数据丢失；在这种情况下，RDD 可以重新计算数据。这种数据血缘关系是防止数据丢失的另一道防线，是数据复制的补充。

本章涵盖了以下主题：

+   RDD 的内部工作原理

+   创建 RDD

+   全局与局部作用域

+   转换

+   操作

# RDD 的内部工作原理

RDDs 以并行方式运行。这是在 Spark 中工作的最大优势：每个转换都是并行执行的，这极大地提高了速度。

对数据集的转换是惰性的。这意味着只有当对数据集调用操作时，任何转换才会执行。这有助于 Spark 优化执行。例如，考虑以下分析师通常为了熟悉数据集而执行的非常常见的步骤：

1.  计算某个列中 distinct 值的出现次数。

1.  选择以 `A` 开头的记录。

1.  将结果打印到屏幕上。

虽然前面提到的步骤听起来很简单，但如果只对以字母 `A` 开头的项感兴趣，就没有必要对所有其他项的 distinct 值进行计数。因此，而不是按照前面几点所述的执行流程，Spark 只能计数以 `A` 开头的项，然后将结果打印到屏幕上。

让我们用代码来分解这个例子。首先，我们使用 `.map(lambda v: (v, 1))` 方法让 Spark 映射 `A` 的值，然后选择以 `'A'` 开头的记录（使用 `.filter(lambda val: val.startswith('A'))` 方法）。如果我们调用 `.reduceByKey(operator.add)` 方法，它将减少数据集并 *添加*（在这个例子中，是计数）每个键的出现次数。所有这些步骤 **转换** 数据集。

其次，我们调用 `.collect()` 方法来执行步骤。这一步是对我们数据集的 **操作** - 它最终计算数据集的 distinct 元素。实际上，操作可能会颠倒转换的顺序，先过滤数据然后再映射，从而将较小的数据集传递给归约器。

### 注意

如果您目前还不理解前面的命令，请不要担心 - 我们将在本章后面详细解释它们。

# 创建 RDD

在 PySpark 中创建 RDD 有两种方式：你可以 `.parallelize(...)` 一个集合（`list` 或某些元素的 `array`）：

[PRE0]

或者你可以引用位于本地或外部位置的文件（或文件）：

[PRE1]

### 注意

我们从（于 2016 年 7 月 31 日访问）[ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/mortality/mort2014us.zip](ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/mortality/mort2014us.zip) 下载了死亡率数据集 `VS14MORT.txt` 文件；记录模式在本文档中解释 [http://www.cdc.gov/nchs/data/dvs/Record_Layout_2014.pdf](http://www.cdc.gov/nchs/data/dvs/Record_Layout_2014.pdf)。我们故意选择这个数据集：记录的编码将帮助我们解释如何在本章后面使用 UDFs 来转换你的数据。为了你的方便，我们还在这里托管了文件：[http://tomdrabas.com/data/VS14MORT.txt.gz](http://tomdrabas.com/data/VS14MORT.txt.gz)

`sc.textFile(..., n)` 的最后一个参数指定了数据集被分割成的分区数。

### 提示

一个经验法则是将你的数据集分成两到四个分区，每个分区在你的集群中。

Spark 可以从多种文件系统中读取：本地文件系统，如 NTFS、FAT 或 Mac OS 扩展（HFS+），或分布式文件系统，如 HDFS、S3、Cassandra 等。

### 提示

注意你的数据集是从哪里读取或保存的：路径不能包含特殊字符 `[]`。注意，这也适用于存储在 Amazon S3 或 Microsoft Azure Data Storage 上的路径。

支持多种数据格式：文本、parquet、JSON、Hive 表以及来自关系数据库的数据可以使用 JDBC 驱动程序读取。请注意，Spark 可以自动处理压缩数据集（如我们前面的例子中的 Gzipped 数据集）。

根据数据的读取方式，持有数据的对象将略有不同。当我们 `.paralellize(...)` 一个集合时，从文件读取的数据将表示为 `MapPartitionsRDD` 而不是 `ParallelCollectionRDD`。

## 模式

RDDs 是无模式的（与我们在下一章中将要讨论的 DataFrames 不同）。因此，当使用 RDDs 时，Spark 在并行化数据集（如下代码片段所示）时是完全可以接受的：

[PRE2]

因此，我们可以混合几乎所有东西：一个 `tuple`、一个 `dict` 或一个 `list`，Spark 不会抱怨。

一旦你 `.collect()` 收集数据集（即运行一个操作将其返回到驱动程序），你就可以像在 Python 中正常那样访问对象中的数据：

[PRE3]

它将产生以下输出：

[PRE4]

`.collect()` 方法将 RDD 的所有元素返回到驱动程序，其中它被序列化为一个列表。

### 注意

我们将在本章后面更详细地讨论使用 `.collect()` 的注意事项。

## 从文件读取

当你从文本文件中读取时，文件中的每一行形成一个 RDD 的元素。

`data_from_file.take(1)` 命令将产生以下（有些难以阅读）输出：

![从文件读取](img/B05793_02_01.jpg)

为了使其更易于阅读，让我们创建一个元素列表，这样每行都表示为一个值列表。

## Lambda 表达式

在这个例子中，我们将从 `data_from_file` 的神秘记录中提取有用的信息。

### 注意

请参阅我们 GitHub 仓库中这本书的详细信息，关于此方法的细节。在这里，由于空间限制，我们只展示完整方法的简略版，特别是我们创建正则表达式模式的部分。代码可以在以下位置找到：[https://github.com/drabastomek/learningPySpark/tree/master/Chapter03/LearningPySpark_Chapter03.ipynb](https://github.com/drabastomek/learningPySpark/tree/master/Chapter03/LearningPySpark_Chapter03.ipynb)。

首先，让我们在以下代码的帮助下定义该方法，该代码将解析不可读的行，使其变得可使用：

[PRE5]

### 小贴士

在这里需要提醒的是，定义纯 Python 方法可能会减慢你的应用程序，因为 Spark 需要不断地在 Python 解释器和 JVM 之间切换。只要可能，你应该使用内置的 Spark 函数。

接下来，我们导入必要的模块：`re` 模块，因为我们将会使用正则表达式来解析记录，以及 `NumPy` 以便于一次性选择多个元素。

最后，我们创建一个 `Regex` 对象来提取所需的信息，并通过它解析行。

### 注意

我们不会深入描述正则表达式。关于这个主题的良善汇编可以在以下位置找到：[https://www.packtpub.com/application-development/mastering-python-regular-expressions](https://www.packtpub.com/application-development/mastering-python-regular-expressions)。

一旦解析了记录，我们尝试将列表转换为 `NumPy` 数组并返回它；如果失败，我们返回一个包含默认值 `-99` 的列表，这样我们知道这个记录没有正确解析。

### 小贴士

我们可以通过使用 `.flatMap(...)` 隐式过滤掉格式不正确的记录，并返回一个空列表 `[]` 而不是 `-99` 值。有关详细信息，请参阅：[http://stackoverflow.com/questions/34090624/remove-elements-from-spark-rdd](http://stackoverflow.com/questions/34090624/remove-elements-from-spark-rdd)

现在，我们将使用 `extractInformation(...)` 方法来分割和转换我们的数据集。请注意，我们只传递方法签名到 `.map(...)`：该方法将每次在每个分区中 `hand over` 一个 RDD 元素到 `extractInformation(...)` 方法：

[PRE6]

运行 `data_from_file_conv.take(1)` 将产生以下结果（简略）：

![Lambda 表达式](img/B05793_02_02.jpg)

# 全局与局部作用域

作为潜在的 PySpark 用户，你需要习惯 Spark 的固有并行性。即使你精通 Python，在 PySpark 中执行脚本也需要稍微转变一下思维方式。

Spark可以以两种模式运行：本地和集群。当您在本地运行Spark时，您的代码可能与您目前习惯的Python运行方式不同：更改可能更多的是语法上的，但有一个额外的变化，即数据和代码可以在不同的工作进程之间复制。

然而，如果您不小心，将相同的代码部署到集群中可能会让您感到困惑。这需要理解Spark如何在集群上执行作业。

在集群模式下，当提交作业以执行时，作业被发送到驱动程序节点（或主节点）。驱动程序节点为作业创建一个DAG（见[第一章](ch01.html "第一章. 理解Spark")，*理解Spark*），并决定哪些执行器（或工作节点）将运行特定任务。

然后，驱动程序指示工作进程执行其任务，并在完成后将结果返回给驱动程序。然而，在发生之前，驱动程序会为每个任务准备闭包：一组变量和方法，这些变量和方法在驱动程序上存在，以便工作进程可以在RDD上执行其任务。

这组变量和方法在执行器上下文中本质上是*静态的*，也就是说，每个执行器都会从驱动程序获取变量和方法的一个*副本*。如果在运行任务时，执行器更改这些变量或覆盖方法，它将这样做**而不**影响其他执行器的副本或驱动程序的变量和方法。这可能会导致一些意外的行为和运行时错误，有时很难追踪。

### 注意

查看PySpark文档中的这个讨论以获取更实际的示例：[http://spark.apache.org/docs/latest/programming-guide.html#local-vs-cluster-modes](http://spark.apache.org/docs/latest/programming-guide.html#local-vs-cluster-modes)。

# 转换

转换塑造了您的数据集。这包括映射、过滤、连接和转码数据集中的值。在本节中，我们将展示RDD上可用的某些转换。

### 注意

由于空间限制，我们在此仅包含最常用的转换和操作。对于完整的方法集，我们建议您查看PySpark关于RDD的文档[http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD)。

由于RDD是无模式的，在本节中我们假设您知道生成的数据集的模式。如果您无法记住解析的数据集中信息的位置，我们建议您参考GitHub上`extractInformation(...)`方法的定义，代码在`第三章`。

## .map(...)转换

可以说，您将最常使用`.map(...)`转换。该方法应用于RDD的每个元素：在`data_from_file_conv`数据集的情况下，您可以将其视为对每行的转换。

在这个例子中，我们将创建一个新的数据集，将死亡年份转换为数值：

[PRE7]

运行`data_2014.take(10)`将产生以下结果：

![The .map(...) transformation](img/B05793_02_03.jpg)

### 小贴士

如果你不太熟悉`lambda`表达式，请参阅此资源：[https://pythonconquerstheuniverse.wordpress.com/2011/08/29/lambda_tutorial/](https://pythonconquerstheuniverse.wordpress.com/2011/08/29/lambda_tutorial/)。

你当然可以引入更多的列，但你需要将它们打包成一个`tuple`、`dict`或`list`。让我们也包含行中的第17个元素，以便我们可以确认我们的`.map(...)`按预期工作：

[PRE8]

上述代码将产生以下结果：

![The .map(...) transformation](img/B05793_02_04.jpg)

## The .filter(...) transformation

另一个最常使用的转换方法是`.filter(...)`方法，它允许你从数据集中选择符合特定标准的元素。作为一个例子，从`data_from_file_conv`数据集中，让我们计算有多少人在2014年发生了事故死亡：

[PRE9]

### 小贴士

注意，前面的命令可能需要一段时间，具体取决于你的电脑有多快。对我们来说，它花了点超过两分钟的时间才返回结果。

## The .flatMap(...) transformation

`.flatMap(...)`方法与`.map(...)`类似，但它返回一个扁平化的结果而不是列表。如果我们执行以下代码：

[PRE10]

它将产生以下输出：

![The .flatMap(...) transformation](img/B05793_02_05.jpg)

你可以将这个结果与之前生成`data_2014_2`的命令的结果进行比较。注意，正如之前提到的，`.flatMap(...)`方法可以在你需要解析输入时用来过滤掉一些格式不正确的记录。在底层，`.flatMap(...)`方法将每一行视为一个列表，然后简单地*添加*所有记录；通过传递一个空列表，格式不正确的记录将被丢弃。

## The .distinct(...) transformation

此方法返回指定列中的唯一值列表。如果你想要了解你的数据集或验证它，这个方法非常有用。让我们检查`gender`列是否只包含男性和女性；这将验证我们是否正确解析了数据集。让我们运行以下代码：

[PRE11]

此代码将产生以下结果：

![The .distinct(...) transformation](img/B05793_02_06.jpg)

首先，我们只提取包含性别的列。接下来，我们使用`.distinct()`方法来选择列表中的唯一值。最后，我们使用`.collect()`方法来在屏幕上打印这些值。

### 小贴士

注意，这是一个昂贵的操作，应该谨慎使用，并且仅在必要时使用，因为它会在数据周围进行洗牌。

## The .sample(...) transformation

`.sample(...)` 方法从数据集中返回一个随机样本。第一个参数指定采样是否带替换，第二个参数定义要返回的数据的分数，第三个是伪随机数生成器的种子：

[PRE12]

在这个例子中，我们从原始数据集中选择了 10% 的随机样本。为了确认这一点，让我们打印数据集的大小：

[PRE13]

前面的命令产生了以下输出：

![.sample(...) 转换](img/B05793_02_07.jpg)

我们使用 `.count()` 操作来计算相应 RDD 中的所有记录数。

## .leftOuterJoin(...) 转换

`.leftOuterJoin(...)`，就像在 SQL 世界中一样，基于两个数据集中找到的值将两个 RDD 连接起来，并返回来自左 RDD 的记录，在两个 RDD 匹配的地方附加来自右 RDD 的记录：

[PRE14]

在 `rdd3` 上运行 `.collect(...)` 将产生以下结果：

![.leftOuterJoin(...) 转换](img/B05793_02_08.jpg)

### 提示

这是一种成本较高的方法，应该谨慎使用，并且仅在必要时使用，因为它会在数据周围进行洗牌，从而影响性能。

你在这里看到的是来自 RDD `rdd1` 的所有元素及其来自 RDD `rdd2` 的对应值。正如你所见，值 `'a'` 在 `rdd3` 中出现了两次，并且 `'a'` 在 RDD `rdd2` 中也出现了两次。`rdd1` 中的值 `b` 只出现一次，并与来自 `rdd2` 的值 `'6'` 相连接。有两件事*缺失*：`rdd1` 中的值 `'c'` 在 `rdd2` 中没有对应的键，因此在返回的元组中的值显示为 `None`，并且，由于我们执行的是左外连接，`rdd2` 中的值 `'d'` 如预期那样消失了。

如果我们使用 `.join(...)` 方法，我们只会得到 `'a'` 和 `'b'` 的值，因为这两个值在这两个 RDD 之间相交。运行以下代码：

[PRE15]

它将产生以下输出：

![.leftOuterJoin(...) 转换](img/B05793_02_09.jpg)

另一个有用的方法是 `.intersection(...)`，它返回在两个 RDD 中相等的记录。执行以下代码：

[PRE16]

输出如下：

![.leftOuterJoin(...) 转换](img/B05793_02_10.jpg)

## .repartition(...) 转换

重新分区数据集会改变数据集被分割成的分区数量。这个功能应该谨慎使用，并且仅在真正必要时使用，因为它会在数据周围进行洗牌，这在实际上会导致性能的显著下降：

[PRE17]

前面的代码打印出 `4` 作为新的分区数量。

与 `.collect()` 相比，`.glom()` 方法产生一个列表，其中每个元素是另一个列表，包含在指定分区中存在的数据集的所有元素；返回的主要列表具有与分区数量相同的元素数量。

# 操作

与转换不同，动作在数据集上执行计划的任务；一旦你完成了数据的转换，你就可以执行转换。这可能不包含任何转换（例如，`.take(n)` 将只从RDD返回 `n` 条记录，即使你没有对它进行任何转换）或执行整个转换链。

## .take(...) 方法

这可能是最有用（并且使用最频繁，例如 `.map(...)` 方法）。该方法比 `.collect(...)` 更受欢迎，因为它只从单个数据分区返回 `n` 个顶部行，而 `.collect(...)` 则返回整个RDD。这在处理大型数据集时尤为重要：

[PRE18]

如果你想要一些随机的记录，可以使用 `.takeSample(...)` 代替，它接受三个参数：第一个参数指定采样是否带替换，第二个参数指定要返回的记录数，第三个参数是伪随机数生成器的种子：

[PRE19]

## .collect(...) 方法

此方法将RDD的所有元素返回给驱动器。正如我们刚刚已经对此提出了警告，我们在这里不再重复。

## .reduce(...) 方法

`.reduce(...)` 方法使用指定的方法对RDD的元素进行归约。

你可以使用它来对RDD的元素进行求和：

[PRE20]

这将产生`15`的总和。

我们首先使用 `.map(...)` 转换创建 `rdd1` 所有值的列表，然后使用 `.reduce(...)` 方法处理结果。`reduce(...)` 方法在每个分区上运行求和函数（这里表示为 `lambda`），并将求和结果返回给驱动节点，在那里进行最终聚合。

### 注意

这里需要提醒一句。作为reducer传递的函数需要是**结合律**，也就是说，当元素的顺序改变时，结果不会改变，并且**交换律**，也就是说，改变操作数的顺序也不会改变结果。

结合律的例子是 *(5 + 2) + 3 = 5 + (2 + 3)*，交换律的例子是 *5 + 2 + 3 = 3 + 2 + 5*。因此，你需要小心传递给reducer的函数。

如果你忽略了前面的规则，你可能会遇到麻烦（假设你的代码能正常运行的话）。例如，假设我们有一个以下RDD（只有一个分区！）：

[PRE21]

如果我们以我们想要将当前结果除以下一个结果的方式来减少数据，我们期望得到`10`的值：

[PRE22]

然而，如果你将数据分区成三个分区，结果将会是错误的：

[PRE23]

它将产生 `0.004`。

`.reduceByKey(...)` 方法的工作方式与 `.reduce(...)` 方法类似，但它是在键键基础上进行归约：

[PRE24]

上述代码会产生以下结果：

![.reduce(...) 方法](img/B05793_02_11.jpg)

## .count(...) 方法

`.count(...)` 方法计算RDD中元素的数量。使用以下代码：

[PRE25]

此代码将产生 `6`，这是 `data_reduce` RDD 中元素的确切数量。

`.count(...)` 方法产生的结果与以下方法相同，但它不需要将整个数据集移动到驱动程序：

[PRE26]

如果您的数据集是键值形式，您可以使用 `.countByKey()` 方法来获取不同键的计数。运行以下代码：

[PRE27]

此代码将产生以下输出：

![.count(...) 方法](img/B05793_02_12.jpg)

## `.saveAsTextFile(...)` 方法

如其名所示，`.saveAsTextFile(...)` 方法将 RDD 保存为文本文件：每个分区保存到一个单独的文件中：

[PRE28]

要读取它，你需要将其解析回字符串，因为所有行都被视为字符串：

[PRE29]

读取的键列表与我们最初的有匹配：

![.saveAsTextFile(...) 方法](img/B05793_02_13.jpg)

## `.foreach(...)` 方法

这是一个将相同的函数以迭代方式应用于 RDD 中每个元素的方法；与 `.map(..)` 相比，`.foreach(...)` 方法以一对一的方式对每条记录应用定义的函数。当您想将数据保存到 PySpark 本地不支持的数据库时，它非常有用。

在这里，我们将使用它来打印（到 CLI - 而不是 Jupyter Notebook）存储在 `data_key` RDD 中的所有记录：

[PRE30]

如果你现在导航到 CLI，你应该看到所有记录被打印出来。注意，每次的顺序很可能是不同的。

# 摘要

RDD 是 Spark 的骨架；这些无模式的数据库结构是我们将在 Spark 中处理的最基本的数据结构。

在本章中，我们介绍了通过 `.parallelize(...)` 方法以及从文本文件中读取数据来创建 RDD 的方法。还展示了处理非结构化数据的一些方法。

Spark 中的转换是惰性的 - 只有在调用操作时才会应用。在本章中，我们讨论并介绍了最常用的转换和操作；PySpark 文档包含更多内容[http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD)。

Scala 和 Python RDD 之间的一个主要区别是速度：Python RDD 可能比它们的 Scala 对应物慢得多。

在下一章中，我们将向您介绍一种数据结构，它使 PySpark 应用程序的性能与 Scala 编写的应用程序相当 - 数据帧。
