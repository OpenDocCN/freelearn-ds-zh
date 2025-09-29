# 第 3 章 数据帧

DataFrame 是一个不可变的分布式数据集合，它组织成命名的列，类似于关系数据库中的表。作为 Apache Spark 1.0 中的实验性功能 `SchemaRDD` 的一部分引入，它们在 Apache Spark 1.3 发布中更名为 `DataFrames`。对于熟悉 Python Pandas `DataFrame` 或 R `DataFrame` 的读者，Spark DataFrame 是一个类似的概念，它允许用户轻松地处理结构化数据（例如，数据表）；也有一些差异，所以请调整您的期望。

通过对分布式数据集合施加结构，这允许 Spark 用户在 Spark SQL 或使用表达式方法（而不是 lambda 函数）中查询结构化数据。在本章中，我们将包括使用这两种方法的代码示例。通过结构化您的数据，这允许 Apache Spark 引擎——特别是 Catalyst 优化器——显著提高 Spark 查询的性能。在 Spark 早期 API（即 RDDs）中，由于 Java JVM 和 Py4J 之间的通信开销，执行 Python 中的查询可能会显著变慢。

### 注意

如果您熟悉在 Spark 早期版本（即 Spark 1.x）中与 DataFrame 一起工作，您会注意到在 Spark 2.0 中，我们使用 SparkSession 而不是 `SQLContext`。各种 Spark 上下文：`HiveContext`、`SQLContext`、`StreamingContext` 和 `SparkContext` 已合并到 SparkSession 中。这样，您将只作为读取数据、处理元数据、配置和集群资源管理的入口点与这个会话一起工作。

更多信息，请参阅 *如何在 Apache Spark 2.0 中使用 SparkSession*([http://bit.ly/2br0Fr1](http://bit.ly/2br0Fr1))。

在本章中，您将了解以下内容：

+   Python 到 RDD 通信

+   快速回顾 Spark 的 Catalyst 优化器

+   使用 DataFrames 加速 PySpark

+   创建 DataFrame

+   简单的 DataFrame 查询

+   与 RDD 交互

+   使用 DataFrame API 进行查询

+   使用 Spark SQL 进行查询

+   使用 DataFrame 进行准点航班性能分析

# Python 到 RDD 通信

每当使用 RDDs 执行 PySpark 程序时，执行作业可能会有很大的开销。如以下图所示，在 PySpark 驱动程序中，`Spark Context` 使用 `Py4j` 通过 `JavaSparkContext` 启动 JVM。任何 RDD 转换最初都映射到 Java 中的 `PythonRDD` 对象。

一旦这些任务被推送到 Spark Worker(s)，`PythonRDD` 对象将使用管道启动 Python `subprocesses`，以发送 *代码和数据* 到 Python 中进行处理：

![Python 到 RDD 通信](img/B05793_03_01.jpg)

虽然这种方法允许 PySpark 将数据处理分布到多个工作者的多个 Python 子进程中，但如您所见，Python 和 JVM 之间存在大量的上下文切换和通信开销。

### 注意

关于PySpark性能的优秀资源是Holden Karau的*改进PySpark性能：Spark性能超越JVM*：[http://bit.ly/2bx89bn](http://bit.ly/2bx89bn)。

# Catalyst优化器刷新

如[第1章](ch01.html "第1章. 理解Spark")中所述，*理解Spark*，Spark SQL引擎之所以如此快速，其中一个主要原因是**Catalyst优化器**。对于有数据库背景的读者来说，这个图看起来与关系数据库管理系统（**RDBMS**）的逻辑/物理规划器和基于成本的优化模型/基于成本的优化类似：

![Catalyst优化器刷新](img/B05793_03_02.jpg)

这的重要性在于，与立即处理查询相反，Spark引擎的Catalyst优化器编译并优化一个逻辑计划，并有一个成本优化器来确定生成的最有效物理计划。

### 注意

如前几章所述，虽然Spark SQL引擎既有基于规则的优化也有基于成本的优化，包括（但不限于）谓词下推和列剪枝。针对Apache Spark 2.2版本，jira项目*[SPARK-16026]基于成本的优化器框架*在[https://issues.apache.org/jira/browse/SPARK-16026](https://issues.apache.org/jira/browse/SPARK-16026)是一个涵盖广播连接选择之外基于成本的优化器框架的通用票据。更多信息，请参阅[http://bit.ly/2li1t4T](http://bit.ly/2li1t4T)上的*Spark基于成本优化设计规范*。

作为**Project Tungsten**的一部分，通过生成字节码（代码生成或`codegen`）而不是解释每一行数据来进一步提高性能。更多关于Tungsten的详细信息，请参阅[第1章](ch01.html "第1章. 理解Spark")中*理解Spark*章节的*Project Tungsten*部分。

如前所述，优化器基于函数式编程结构，并设计有两个目的：简化向Spark SQL添加新的优化技术和功能，并允许外部开发者扩展优化器（例如，添加数据源特定的规则、支持新的数据类型等）。

### 注意

更多信息，请参阅Michael Armbrust的优秀演示文稿，*结构化Spark：SQL DataFrames、Datasets和Streaming*：[http://bit.ly/2cJ508x](http://bit.ly/2cJ508x)。

要进一步了解*Catalyst优化器*，请参阅[http://bit.ly/2bDVB1T](http://bit.ly/2bDVB1T)上的*深入Spark SQL的Catalyst优化器*。

此外，有关*Project Tungsten*的更多信息，请参阅[http://bit.ly/2bQIlKY](http://bit.ly/2bQIlKY)上的*Project Tungsten：将Apache Spark带到裸金属更近一步*，以及[http://bit.ly/2bDWtnc](http://bit.ly/2bDWtnc)上的*Apache Spark作为编译器：在笔记本电脑上每秒处理十亿行数据*。

# 使用DataFrames加速PySpark

DataFrame和*Catalyst Optimizer*（以及*Project Tungsten*）的重要性在于与未优化的RDD查询相比，PySpark查询性能的提升。如图所示，在引入DataFrame之前，Python查询速度通常比使用RDD的相同Scala查询慢两倍。通常，这种查询性能的下降是由于Python和JVM之间的通信开销：

![使用DataFrame加速PySpark](img/B05793_03_03.jpg)

来源：*在Apache-spark中介绍DataFrame用于大规模数据科学*，请参阅[http://bit.ly/2blDBI1](http://bit.ly/2blDBI1)

使用DataFrame，不仅Python性能有了显著提升，现在Python、Scala、SQL和R之间的性能也实现了对等。

### 提示

重要的是要注意，虽然DataFrame使得PySpark通常运行得更快，但也有一些例外。最突出的是Python UDF的使用，这会导致Python和JVM之间的往返通信。注意，这将是最坏的情况，如果计算是在RDD上完成的，情况将相似。

即使Catalyst Optimizer的代码库是用Scala编写的，Python也可以利用Spark的性能优化。基本上，它是一个大约2,000行代码的Python包装器，允许PySpark DataFrame查询显著加快。

总的来说，Python DataFrame（以及SQL、Scala DataFrame和R DataFrame）都能够利用Catalyst Optimizer（如下面的更新图所示）：

![使用DataFrame加速PySpark](img/B05793_03_04.jpg)

### 注意

有关更多信息，请参阅博客文章*在Apache Spark中介绍DataFrame用于大规模数据科学*，请参阅[http://bit.ly/2blDBI1](http://bit.ly/2blDBI1)，以及Reynold Xin在Spark Summit 2015上的演讲，*从DataFrame到Tungsten：一瞥Spark的未来*，请参阅[http://bit.ly/2bQN92T](http://bit.ly/2bQN92T)。

# 创建DataFrame

通常，你将通过使用SparkSession（或在PySpark shell中调用`spark`）导入数据来创建DataFrame。

### 提示

在Spark 1.x版本中，你通常必须使用`sqlContext`。

在未来的章节中，我们将讨论如何将数据导入你的本地文件系统、**Hadoop分布式文件系统**（**HDFS**）或其他云存储系统（例如，S3或WASB）。对于本章，我们将专注于在Spark中直接生成自己的DataFrame数据或利用Databricks社区版中已有的数据源。

### 注意

关于如何注册Databricks社区版的说明，请参阅附录章节，*免费Spark云服务提供*。

首先，我们不会访问文件系统，而是通过生成数据来创建一个DataFrame。在这种情况下，我们首先创建`stringJSONRDD` RDD，然后将其转换为DataFrame。此代码片段创建了一个包含游泳者（他们的ID、姓名、年龄和眼睛颜色）的JSON格式的RDD。

## 生成我们自己的 JSON 数据

下面，我们将首先生成 `stringJSONRDD` RDD：

[PRE0]

现在我们已经创建了 RDD，我们将使用 SparkSession 的 `read.json` 方法（即 `spark.read.json(...)`）将其转换为 DataFrame。我们还将使用 `.createOrReplaceTempView` 方法创建一个临时表。

### 注意

在 Spark 1.x 中，此方法为 `.registerTempTable`，它作为 Spark 2.x 的一部分已被弃用。

## 创建 DataFrame

下面是创建 DataFrame 的代码：

[PRE1]

## 创建临时表

下面是创建临时表的代码：

[PRE2]

如前几章所述，许多 RDD 操作是转换，它们只有在执行动作操作时才会执行。例如，在前面的代码片段中，`sc.parallelize` 是一个转换，它在将 RDD 转换为 DataFrame 时执行，即使用 `spark.read.json`。注意，在这个代码片段笔记本的截图（靠近左下角）中，Spark 作业直到包含 `spark.read.json` 操作的第二个单元格才会执行。

### 小贴士

这些截图来自 Databricks Community Edition，但所有代码示例和 Spark UI 截图都可以在任何 Apache Spark 2.x 版本中执行/查看。

为了进一步强调这一点，在以下图例的右侧面板中，我们展示了执行 DAG 图。

### 注意

更好地理解 Spark UI DAG 可视化的一个绝佳资源是博客文章《通过可视化理解您的 Apache Spark 应用程序》([http://bit.ly/2cSemkv](http://bit.ly/2cSemkv))。

在以下截图，你可以看到 Spark 作业的 `parallelize` 操作来自于生成 RDD `stringJSONRDD` 的第一个单元格，而 `map` 和 `mapPartitions` 操作是创建 DataFrame 所需的操作：

![创建临时表](img/B05793_03_05.jpg)

Spark UI 中 spark.read.json(stringJSONRDD) 作业的 DAG 可视化。

在以下截图，你可以看到 `parallelize` 操作的 *阶段* 来自于生成 RDD `stringJSONRDD` 的第一个单元格，而 `map` 和 `mapPartitions` 操作是创建 DataFrame 所需的操作：

![创建临时表](img/B05793_03_06.jpg)

Spark UI 中 spark.read.json(stringJSONRDD) 作业的 DAG 可视化阶段。

重要的是要注意，`parallelize`、`map` 和 `mapPartitions` 都是 RDD 转换。包裹在 DataFrame 操作（在这种情况下为 `spark.read.json`）中的不仅仅是 RDD 转换，还包括将 RDD 转换为 DataFrame 的 *动作*。这是一个重要的说明，因为尽管你正在执行 DataFrame *操作*，但在调试操作时，你需要记住你将需要在 Spark UI 中理解 *RDD 操作*。

注意，创建临时表是 DataFrame 转换，并且不会在执行 DataFrame 动作之前执行（例如，在下一节中要执行的 SQL 查询中）。

### 注意

DataFrame 转换和操作与 RDD 转换和操作类似，因为存在一组惰性操作（转换）。但是，与 RDD 相比，DataFrame 操作的惰性程度较低，这主要是由于 Catalyst 优化器。有关更多信息，请参阅 Holden Karau 和 Rachel Warren 的书籍 *High Performance Spark*，[http://highperformancespark.com/](http://highperformancespark.com/)。

# 简单的 DataFrame 查询

现在您已经创建了 `swimmersJSON` DataFrame，我们将能够运行 DataFrame API，以及针对它的 SQL 查询。让我们从一个简单的查询开始，显示 DataFrame 中的所有行。

## DataFrame API 查询

要使用 DataFrame API 执行此操作，您可以使用 `show(<n>)` 方法，该方法将前 `n` 行打印到控制台：

### 小贴士

运行 `.show()` 方法将默认显示前 10 行。

[PRE3]

这将产生以下输出：

![DataFrame API 查询](img/B05793_03_07.jpg)

## SQL 查询

如果您更喜欢编写 SQL 语句，可以编写以下查询：

[PRE4]

这将产生以下输出：

![SQL 查询](img/B05793_03_08.jpg)

我们正在使用 `.collect()` 方法，它返回所有记录作为 **Row** 对象的列表。请注意，您可以使用 `collect()` 或 `show()` 方法对 DataFrame 和 SQL 查询进行操作。只需确保如果您使用 `.collect()`，这仅适用于小型 DataFrame，因为它将返回 DataFrame 中的所有行并将它们从执行器移动到驱动器。您还可以使用 `take(<n>)` 或 `show(<n>)`，这允许您通过指定 `<n>` 来限制返回的行数：

### 小贴士

注意，如果您使用 Databricks，可以使用 `%sql` 命令并在笔记本单元中直接运行 SQL 语句，如上所述。

![SQL 查询](img/B05793_03_09.jpg)

# 与 RDD 互操作

将现有的 RDD 转换为 DataFrame（或 Datasets[T]）有两种不同的方法：使用反射推断模式，或程序化指定模式。前者允许您编写更简洁的代码（当您的 Spark 应用程序已经知道模式时），而后者允许您在运行时仅当列及其数据类型被揭示时构建 DataFrame。请注意，**反射**是指 *模式反射*，而不是 Python `反射`。

## 使用反射推断模式

在构建 DataFrame 和运行查询的过程中，我们跳过了这样一个事实，即此 DataFrame 的模式是自动定义的。最初，通过将键/值对列表作为 `**kwargs` 传递给行类来构建行对象。然后，Spark SQL 将此行对象 RDD 转换为 DataFrame，其中键是列，数据类型通过采样数据推断。

### 小贴士

`**kwargs` 构造函数允许你在运行时传递一个可变数量的参数给一个方法。

返回到代码，在最初创建 `swimmersJSON` DataFrame，没有指定模式的情况下，你会注意到通过使用 `printSchema()` 方法来定义模式：

[PRE5]

这将给出以下输出：

![使用反射推断模式](img/B05793_03_10.jpg)

但如果我们想指定模式，因为在这个例子中我们知道 `id` 实际上是一个 `long` 而不是一个 `string` 呢？

## 编程指定模式

在这个例子中，让我们通过引入 Spark SQL 数据类型（`pyspark.sql.types`）来编程指定模式，并生成一些 `.csv` 数据：

[PRE6]

首先，我们将按照下面的 `[schema]` 变量将模式编码为字符串。然后我们将使用 `StructType` 和 `StructField` 定义模式：

[PRE7]

注意，`StructField` 类可以从以下几个方面进行分解：

+   `name`: 此字段的名称

+   `dataType`: 该字段的类型

+   `nullable`: 指示此字段的值是否可以为 null

最后，我们将我们将创建的 `schema`（模式）应用到 `stringCSVRDD` RDD（即生成的 `.csv` 数据）上，并创建一个临时视图，这样我们就可以使用 SQL 来查询它：

[PRE8]

通过这个例子，我们对模式有了更细粒度的控制，可以指定 `id` 是一个 `long`（与前文中的字符串相反）：

[PRE9]

这将给出以下输出：

![编程指定模式](img/B05793_03_11.jpg)

### 小贴士

在许多情况下，模式可以被推断（如前文所述）并且你不需要指定模式，就像前一个例子中那样。

# 使用 DataFrame API 查询

如前文所述，你可以从使用 `collect()`、`show()` 或 `take()` 开始来查看 DataFrame 中的数据（后两个选项包括限制返回行数的选项）。

## 行数

要获取 DataFrame 中的行数，你可以使用 `count()` 方法：

[PRE10]

这将给出以下输出：

[PRE11]

## 运行过滤语句

要运行一个过滤语句，你可以使用 `filter` 子句；在下面的代码片段中，我们使用 `select` 子句来指定要返回的列：

[PRE12]

这个查询的输出是选择 `id` 和 `age` 列，其中 `age` = `22`：

![运行过滤语句](img/B05793_03_12.jpg)

如果我们只想获取那些眼睛颜色以字母 `b` 开头的游泳者的名字，我们可以使用类似 SQL 的语法，即 `like`，如下面的代码所示：

[PRE13]

输出如下：

![运行过滤语句](img/B05793_03_13.jpg)

# 使用 SQL 查询

让我们运行相同的查询，但这次我们将使用针对同一 DataFrame 的 SQL 查询。回想一下，这个 DataFrame 是可访问的，因为我们为 `swimmers` 执行了 `.createOrReplaceTempView` 方法。

## 行数

以下代码片段用于使用 SQL 获取 DataFrame 中的行数：

[PRE14]

输出如下：

![行数](img/B05793_03_14.jpg)

## 使用where子句运行过滤语句

要使用SQL运行过滤语句，您可以使用`where`子句，如下面的代码片段所示：

[PRE15]

此查询的输出是仅选择`age`等于`22`的`id`和`age`列：

![使用where子句运行过滤语句](img/B05793_03_15.jpg)

与DataFrame API查询类似，如果我们只想获取具有以字母`b`开眼的游泳者的名字，我们也可以使用`like`语法：

[PRE16]

输出如下：

![使用where子句运行过滤语句](img/B05793_03_16.jpg)

### 提示

更多信息，请参阅[Spark SQL、DataFrames和Datasets指南](http://bit.ly/2cd1wyx)。

### 注意

在使用Spark SQL和DataFrames时，一个重要的注意事项是，虽然处理CSV、JSON和多种数据格式都很方便，但Spark SQL分析查询最常用的存储格式是*Parquet*文件格式。它是一种列式格式，被许多其他数据处理系统支持，并且Spark SQL支持读取和写入Parquet文件，自动保留原始数据的模式。更多信息，请参阅最新的*Spark SQL编程指南 > Parquet文件*，链接为：[http://spark.apache.org/docs/latest/sql-programming-guide.html#parquet-files](http://spark.apache.org/docs/latest/sql-programming-guide.html#parquet-files)。此外，还有许多与Parquet相关的性能优化，包括但不限于[Parquet的自动分区发现和模式迁移](https://databricks.com/blog/2015/03/24/spark-sql-graduates-from-alpha-in-spark-1-3.html)和[Apache Spark如何使用Parquet元数据快速计数](https://github.com/dennyglee/databricks/blob/master/misc/parquet-count-metadata-explanation.md)。

# DataFrame场景 – 准时飞行性能

为了展示您可以使用DataFrames执行的查询类型，让我们看看准时飞行性能的使用案例。我们将分析*航空公司准时性能和航班延误原因：准时数据*([http://bit.ly/2ccJPPM](http://bit.ly/2ccJPPM))，并将其与从*Open Flights机场、航空公司和航线数据*([http://bit.ly/2ccK5hw](http://bit.ly/2ccK5hw))获得的机场数据集合并，以更好地理解与航班延误相关的变量。

### 提示

在本节中，我们将使用 Databricks Community Edition（Databricks 产品的免费提供），您可以在[https://databricks.com/try-databricks](https://databricks.com/try-databricks)获取。我们将使用 Databricks 内部的可视化和预加载数据集，以便您更容易专注于编写代码和分析结果。

如果您希望在自己的环境中运行此操作，您可以在本书的 GitHub 仓库中找到可用的数据集，网址为 [https://github.com/drabastomek/learningPySpark](https://github.com/drabastomek/learningPySpark)。

## 准备源数据集

我们将首先通过指定文件路径位置并使用 SparkSession 导入来处理源机场和飞行性能数据集：

[PRE17]

注意，我们使用 CSV 读取器（`com.databricks.spark.csv`）导入数据，它适用于任何指定的分隔符（注意，机场数据是制表符分隔的，而飞行性能数据是逗号分隔的）。最后，我们缓存飞行数据集，以便后续查询更快。

## 连接飞行性能和机场

使用 DataFrame/SQL 的更常见任务之一是将两个不同的数据集连接起来；这通常是一项性能要求较高的操作。使用 DataFrame，这些连接的性能优化默认情况下已经包括在内：

[PRE18]

在我们的场景中，我们正在查询华盛顿州的按城市和起始代码的总延误。这需要通过**国际航空运输协会**（**IATA**）代码将飞行性能数据与机场数据连接起来。查询的输出如下：

![连接飞行性能和机场](img/B05793_03_17.jpg)

使用笔记本（如 Databricks、iPython、Jupyter 和 Apache Zeppelin），您可以更轻松地执行和可视化您的查询。在以下示例中，我们将使用 Databricks 笔记本。在我们的 Python 笔记本中，我们可以使用 `%sql` 函数在该笔记本单元格中执行 SQL 语句：

[PRE19]

这与之前的查询相同，但由于格式化，更容易阅读。在我们的 Databricks 笔记本示例中，我们可以快速将此数据可视化成条形图：

![连接飞行性能和机场](img/B05793_03_18.jpg)

## 可视化我们的飞行性能数据

让我们继续可视化我们的数据，但按美国大陆的所有州进行细分：

[PRE20]

输出的条形图如下：

![可视化我们的飞行性能数据](img/B05793_03_19.jpg)

但是，将此数据作为地图查看会更酷；点击图表左下角的条形图图标，您可以选择许多不同的原生导航，包括地图：

![可视化我们的飞行性能数据](img/B05793_03_20.jpg)

DataFrame 的一个关键好处是信息结构类似于表格。因此，无论您是使用笔记本还是您喜欢的 BI 工具，您都将能够快速可视化您的数据。

### 小贴士

您可以在 [http://bit.ly/2bkUGnT](http://bit.ly/2bkUGnT) 找到 `pyspark.sql.DataFrame` 方法的完整列表。

您可以在 [http://bit.ly/2bTAzLT](http://bit.ly/2bTAzLT) 找到 `pyspark.sql.functions` 的完整列表。

# Spark Dataset API

在关于 Spark DataFrames 的讨论之后，让我们快速回顾一下 Spark Dataset API。Apache Spark 1.6 中引入的 Spark Datasets 的目标是提供一个 API，使用户能够轻松地表达对域对象的转换，同时提供强大 Spark SQL 执行引擎的性能和优势。作为 Spark 2.0 版本发布的一部分（如以下图所示），DataFrame API 被合并到 Dataset API 中，从而统一了所有库的数据处理能力。由于这种统一，开发者现在需要学习或记住的概念更少，并且可以使用一个单一的高级和 *类型安全* API —— 被称为 Dataset：

![Spark Dataset API](img/B05793_03_21.jpg)

从概念上讲，Spark DataFrame 是一个 Dataset[Row] 集合的 *别名*，其中 Row 是一个通用的 *未类型化* JVM 对象。相比之下，Dataset 是一个由您在 Scala 或 Java 中定义的 case 类决定的 *强类型化* JVM 对象集合。这一点尤其重要，因为这意味着 Dataset API 由于缺乏类型增强的好处，*不支持* PySpark。注意，对于 Dataset API 中 PySpark 中不可用的部分，可以通过转换为 RDD 或使用 UDFs 来访问。有关更多信息，请参阅 jira [SPARK-13233]：Python Dataset 在 [http://bit.ly/2dbfoFT](http://bit.ly/2dbfoFT)。

# 摘要

使用 Spark DataFrames，Python 开发者可以利用一个更简单的抽象层，这个层也可能会显著更快。Python 在 Spark 中最初较慢的一个主要原因是 Python 子进程和 JVM 之间的通信层。对于 Python DataFrame 用户，我们提供了一个围绕 Scala DataFrames 的 Python 包装器，它避免了 Python 子进程/JVM 通信开销。Spark DataFrames 通过 Catalyst 优化器和 Project Tungsten 实现了许多性能提升，这些我们在本章中已进行了回顾。在本章中，我们还回顾了如何使用 Spark DataFrames，并使用 DataFrames 实现了一个准时航班性能场景。

在本章中，我们通过生成数据或利用现有数据集创建了 DataFrame 并与之工作。

在下一章中，我们将讨论如何转换和理解您自己的数据。
