# 第三章. Spark SQL

大多数企业始终在处理大量的结构化数据。即使处理非结构化数据的方法有很多，许多应用场景仍然需要结构化数据。处理结构化数据和非结构化数据之间主要的区别是什么？如果数据源是结构化的，并且数据处理引擎事先知道数据结构，数据处理引擎在处理数据时可以进行很多优化，甚至在处理之前。这在数据处理量巨大且周转时间非常关键时非常关键。

企业数据的激增要求赋予最终用户在简单且易于使用的应用程序用户界面中查询和处理数据的能力。关系数据库管理系统（RDBMS）供应商联合起来，**结构化查询语言**（**SQL**）应运而生，作为解决这一问题的方案。在过去的几十年里，所有处理数据的人如果还不是高级用户，也熟悉了 SQL。

社交网络和微博等大型互联网应用产生了超出许多传统数据处理工具消费能力的数据。在处理如此庞大的数据海洋时，从其中挑选和选择正确的数据变得更加重要。Spark 是一个高度流行的数据处理平台，其基于 RDD 的编程模型与 Hadoop MapReduce 数据处理框架相比，降低了数据处理工作量。但是，Spark 基于 RDD 的编程模型的初始版本在让最终用户，如数据科学家、数据分析师和业务分析师使用 Spark 方面仍然难以捉摸。他们无法利用基于 RDD 的 Spark 编程模型的主要原因是因为它需要一定程度的函数式编程。解决这个问题的方法是 Spark SQL。Spark SQL 是建立在 Spark 之上的库。它公开了 SQL 接口和 DataFrame API。DataFrame API 支持编程语言 Scala、Java、Python 和 R。

如果提前知道数据的结构，如果数据符合行和列的模型，那么数据的来源并不重要，Spark SQL 可以将其全部一起使用，并像所有数据都来自单一来源一样进行处理。此外，查询方言是通用的 SQL。

在本章中，我们将涵盖以下主题：

+   数据结构

+   Spark SQL

+   聚合

+   多数据源连接

+   数据集

+   数据目录

# 理解数据结构

这里所讨论的数据结构需要进一步阐明。我们所说的数据结构是什么意思？存储在关系型数据库管理系统（RDBMS）中的数据以行/列或记录/字段的方式存储。每个字段都有一个数据类型，每个记录都是相同或不同数据类型的字段集合。在 RDBMS 的早期阶段，字段的数据类型是标量型的，而在最近版本中，它扩展到包括集合数据类型或复合数据类型。因此，无论记录包含标量数据类型还是复合数据类型，这里要强调的重要一点是，底层数据是有结构的。许多数据处理范式都采用了在内存中镜像 RDBMS 或其他存储中持久化的底层数据结构的概念，以简化数据处理。

换句话说，如果一个关系型数据库表中的数据正在被数据处理应用程序处理，如果相同的表样数据结构在内存中对程序、最终用户和程序员可用，那么建模应用程序和查询数据对它们来说就很容易了。例如，假设有一组以逗号分隔的数据项，每行有固定数量的值，这些值在所有行中的特定位置具有特定的数据类型。这是一个结构化数据文件。它是一个数据表，非常类似于 RDBMS 表。

在 R 等编程语言中，有一个用于在内存中存储数据表的 DataFrame 抽象。Python 数据分析库 Pandas 也有类似的数据框概念。一旦这种数据结构在内存中可用，程序就可以提取数据，并根据需要对其进行切片和切块。相同的数据表概念在 Spark 中得到了扩展，称为 DataFrame，它建立在 RDD 之上，Spark SQL 中有一个非常全面的 API 称为 DataFrame API，用于处理 DataFrame 中的数据。在 DataFrame 抽象之上还开发了一种类似 SQL 的查询语言，以满足最终用户查询和处理底层结构化数据的需求。总之，DataFrame 是一个按行和列组织的数据表，并为每个列命名。

基于 Spark 构建的 Spark SQL 库是基于名为 *"Spark SQL: Relational Data Processing in Spark"* 的研究论文开发的。它讨论了 Spark SQL 的四个目标，以下为原文照搬：

+   支持在 Spark 程序内部（在本地 RDD 上）以及使用程序员友好的 API 在外部数据源上进行关系处理

+   使用成熟的数据库管理系统（DBMS）技术提供高性能

+   便于支持新的数据源，包括半结构化数据和适用于查询联合的外部数据库

+   允许使用高级分析算法进行扩展，例如图处理和机器学习

DataFrame 存储结构化数据，并且是分布式的。它允许选择、过滤和聚合数据。听起来很像是 RDD 吗？RDD 和 DataFrame 之间的关键区别在于，DataFrame 存储了比 RDD 更多的关于数据结构的信息，例如列的数据类型和名称。这使得 DataFrame 能够比 Spark 对 RDD 进行处理的转换和操作更有效地优化处理。在这里需要提到的另一个最重要的方面是，所有 Spark 支持的编程语言都可以用来开发使用 Spark SQL DataFrame API 的应用程序。从所有实际应用的角度来看，Spark SQL 是一个分布式 SQL 引擎。

### 小贴士

之前在 Spark 1.3 版本中工作过的人一定熟悉 SchemaRDD，DataFrame 的概念正是建立在 SchemaRDD 之上，并且保持了 API 级别的兼容性。

# 为什么选择 Spark SQL？

毫无疑问，SQL 是进行数据分析的通用语言，而 Spark SQL 是 Spark 工具集家族中用于数据分析的解决方案。那么，它提供了什么？它提供了在 Spark 上运行 SQL 的能力。无论数据来自 CSV、Avro、Parquet、Hive，还是来自 Cassandra 这样的 NoSQL 数据存储，甚至是 RDBMS，Spark SQL 都可以用来分析数据，并与 Spark 程序混合使用。这里提到的许多数据源都由 Spark SQL 内置支持，而许多其他数据源则由外部包支持。在这里需要强调的最重要的一点是 Spark SQL 处理来自非常广泛的数据源的能力。一旦数据作为 Spark 中的 DataFrame 可用，Spark SQL 就可以以完全分布式的方式处理数据，将来自各种数据源的数据帧组合起来进行处理和查询，就像整个数据集都来自单一来源一样。

在上一章中，我们已经详细讨论了 RDD，并将其介绍为 Spark 编程模型。Spark SQL 的 DataFrame API 和 SQL 方言的使用是否正在取代基于 RDD 的编程模型？当然不是！基于 RDD 的编程模型是 Spark 中通用的和基本的数据处理模型。基于 RDD 的编程需要使用真正的编程技术。Spark 的转换和操作使用了大量的函数式编程结构。尽管与 Hadoop MapReduce 或其他任何范式相比，基于 RDD 的编程模型所需的代码量较少，但仍然需要编写一些函数式代码。这对许多数据科学家、数据分析师和业务分析师来说是一个障碍，他们可能需要进行大量的探索性数据分析或使用数据进行原型设计。Spark SQL 完全消除了这些限制。基于简单易用的**领域特定语言**（**DSL**）的方法来从数据源读取和写入数据，类似于 SQL 的语言来选择、过滤和聚合，以及从广泛的数据源读取数据的能力，使得任何了解数据结构的人都能轻松使用它。

### 注意

使用 RDD 的最佳用例是什么，使用 Spark SQL 的最佳用例又是什么？答案非常简单。如果数据是有结构的，如果它可以被安排在表格中，并且如果每一列都可以被赋予一个名称，那么就使用 Spark SQL。这并不意味着 RDD 和 DataFrame 是两个截然不同的实体。它们可以很好地交互。从 RDD 到 DataFrame 以及相反的转换都是可能的。许多通常应用于 RDD 的 Spark 转换和操作也可以应用于 DataFrame。

通常，在设计应用阶段，业务分析师通常使用 SQL 对应用数据进行大量分析，并将其用于应用需求和测试工件。在设计大数据应用时，也需要同样的东西，在这种情况下，除了业务分析师外，数据科学家也会在团队中。在基于 Hadoop 的生态系统中，Hive 被广泛用于大数据的数据分析。现在 Spark SQL 将这种能力带到了任何支持大量数据源的平台。如果有一个在通用硬件上的独立 Spark 安装，就可以进行大量此类活动来分析数据。在通用硬件上以独立模式部署的基本 Spark 安装就足以处理大量数据。

SQL-on-Hadoop 策略引入了许多应用程序，例如 Hive 和 Impala 等，为存储在 Hadoop 分布式文件系统（HDFS）中的底层大数据提供了类似 SQL 的接口。Spark SQL 在这个空间中处于什么位置？在深入探讨这个问题之前，先简要提及 Hive 和 Impala。Hive 是一种基于 MapReduce 的数据仓库技术，由于查询处理使用了 MapReduce，因此 Hive 查询在完成查询之前需要进行大量的 I/O 操作。Impala 通过在内存中处理数据并利用描述数据的 Hive 元存储提出了一个绝妙的解决方案。Spark SQL 使用 SQLContext 来执行所有数据操作。但它也可以使用 HiveContext，HiveContext 比 SQLContext 功能更丰富、更先进。HiveContext 可以执行 SQLContext 可以执行的所有操作，并且在此基础上，它还可以从 Hive 元存储和表中读取数据，还可以访问 Hive 用户定义的函数。显然，使用 HiveContext 的唯一要求是应该有一个已经存在的 Hive 设置 readily available。这样，Spark SQL 可以轻松与 Hive 共存。

### 注意

从 Spark 2.0 开始，SparkSession 成为基于 Spark SQL 的应用程序的新起点，它是 SQLContext 和 HiveContext 的组合，同时支持与 SQLContext 和 HiveContext 的向后兼容。

Spark SQL 可以使用其 Hive 查询语言比 Hive 更快地处理 Hive 表中的数据。Spark SQL 的另一个非常有趣的功能是它可以读取不同版本的 Hive 数据，这是一个非常棒的功能，使得数据源整合在数据处理中成为可能。

### 注意

提供了 Spark SQL 和 DataFrame API 的库提供了可以通过 JDBC/ODBC 访问的接口。这开启了一个全新的数据分析世界。例如，一个通过 JDBC/ODBC 连接到数据源的 **商业智能**（BI）工具可以使用 Spark SQL 支持的许多数据源。此外，BI 工具可以将计算密集型的连接聚合操作推送到 Spark 基础设施中巨大的工作节点集群。

# Spark SQL 的结构

与 Spark SQL 库的交互主要通过两种方法进行。一种是通过类似 SQL 的查询，另一种是通过 DataFrame API。在深入了解基于 DataFrame 的程序的工作原理之前，先看看基于 RDD 的程序是如何工作的是个好主意。

Spark 转换和 Spark 操作被转换为 Java 函数，并在 RDD 上执行，RDD 实际上就是 Java 对象对数据进行操作。由于 RDD 是一个纯 Java 对象，在编译时或运行时都无法知道将要处理什么数据。在执行引擎之前没有可用的元数据来优化 Spark 转换或 Spark 操作。没有预先可用的多个执行路径或查询计划来处理这些数据，因此无法评估各种执行路径的有效性。

在这里，因为没有与数据关联的模式，所以没有执行优化的查询计划。在 DataFrame 的情况下，结构是预先知道的。正因为如此，查询可以提前优化，数据缓存也可以提前建立。

下面的*图 1*给出了关于同一内容的想法：

![Spark SQL 的解剖结构](img/image_03_002.jpg)

图 1

对 DataFrame 进行的类似 SQL 的查询和 DataFrame API 调用被转换为语言无关的表达式。对应于 SQL 查询或 DataFrame API 的语言无关表达式称为未解析的逻辑计划。

通过对 DataFrame 元数据中的列名进行验证，将未解析的逻辑计划转换为逻辑计划。通过应用标准规则，如表达式简化、表达式评估和其他优化规则，进一步优化逻辑计划，形成优化的逻辑计划。优化的逻辑计划被转换为多个物理计划。这些物理计划是通过在逻辑计划中使用 Spark 特定的操作来创建的。选择最佳的物理计划，并将结果查询推送到 RDD 以对数据进行操作。由于 SQL 查询和 DataFrame API 调用被转换为语言无关的查询表达式，因此这些查询的性能在所有支持的语言中都是一致的。这也是为什么 DataFrame API 被所有 Spark 支持的语言（如 Scala、Java、Python 和 R）支持的原因。在未来，由于这个原因，许多更多的语言可能会支持 DataFrame API 和 Spark SQL。

在这里需要提及的是 Spark SQL 的查询计划和优化。在 DataFrame 上通过 SQL 查询或通过 DataFrame API 进行的任何查询操作，在物理上对底层的 base RDD 应用相应的操作之前，都经过了高度优化。在真正的 RDD 操作发生之前，有许多过程。

*图 2*给出了整个查询优化过程的一些想法：

![Spark SQL 的解剖结构](img/image_03_004.jpg)

图 2

可以对 DataFrame 调用两种类型的查询。它们是 SQL 查询或 DataFrame API 调用。它们经过适当的分析，以得出逻辑查询执行计划。然后，在逻辑查询计划上应用优化，以得出优化的逻辑查询计划。从最终的优化逻辑查询计划中，生成一个或多个物理查询计划。对于每个物理查询计划，都会计算出成本模型，并根据最优成本选择合适的物理查询计划，并生成高度优化的代码，针对 RDD 运行。这就是 DataFrame 上任何类型查询性能一致的原因。这也是为什么来自所有这些不同语言的 DataFrame API 调用（Scala、Java、Python 和 R）都能提供一致性能的原因。

让我们再次回顾一下更大的图景，如*图 3*所示，以设定上下文并在我们进入和讨论用例之前了解这里正在讨论的内容：

![Spark SQL 的解剖结构](img/image_03_006.jpg)

图 3

这里将要讨论的用例将展示混合 SQL 查询与 Spark 程序的能力。将选择多个数据源，使用 DataFrame 从这些源读取数据，并展示统一的数据访问。演示中使用的编程语言仍然是 Scala 和 Python。使用 R 操作 DataFrame 的用法将在本书的议程上，并有一个专门的章节介绍。

# DataFrame 编程

用于阐明使用 DataFrame 进行 Spark SQL 编程的用例如下：

+   交易记录以逗号分隔值的形式出现。

+   从列表中过滤出只有有效的交易记录。账户号码应以 `SB` 开头，交易金额应大于零。

+   找到所有交易金额大于 1000 的高价值交易记录。

+   找到所有账户号码无效的交易记录。

+   找到所有交易金额小于或等于零的交易记录。

+   找到所有无效交易记录的合并列表。

+   找到所有交易金额的总和。

+   找到所有交易金额的最大值。

+   找到所有交易金额的最小值。

+   找到所有有效的账户号码。

这正是前一章中使用的相同的一组用例，但在这里编程模型完全不同。使用这组用例，这里展示了两种编程模型。一种是使用 SQL 查询，另一种是使用 DataFrame API。

## 使用 SQL 进行编程

在 Scala REPL 提示符下，尝试以下语句：

```py
scala> // Define the case classes for using in conjunction with DataFrames 
scala> case class Trans(accNo: String, tranAmount: Double) 
defined class Trans 
scala> // Functions to convert the sequence of strings to objects defined by the case classes 
scala> def toTrans =  (trans: Seq[String]) => Trans(trans(0), trans(1).trim.toDouble) 
toTrans: Seq[String] => Trans 
scala> // Creation of the list from where the RDD is going to be created 
scala> val acTransList = Array("SB10001,1000", "SB10002,1200", "SB10003,8000", "SB10004,400", "SB10005,300", "SB10006,10000", "SB10007,500", "SB10008,56", "SB10009,30","SB10010,7000", "CR10001,7000", "SB10002,-10") 
acTransList: Array[String] = Array(SB10001,1000, SB10002,1200, SB10003,8000, SB10004,400, SB10005,300, SB10006,10000, SB10007,500, SB10008,56, SB10009,30, SB10010,7000, CR10001,7000, SB10002,-10) 
scala> // Create the RDD 
scala> val acTransRDD = sc.parallelize(acTransList).map(_.split(",")).map(toTrans(_)) 
acTransRDD: org.apache.spark.rdd.RDD[Trans] = MapPartitionsRDD[2] at map at <console>:30 
scala> // Convert RDD to DataFrame 
scala> val acTransDF = spark.createDataFrame(acTransRDD) 
acTransDF: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Register temporary view in the DataFrame for using it in SQL 
scala> acTransDF.createOrReplaceTempView("trans") 
scala> // Print the structure of the DataFrame 
scala> acTransDF.printSchema 
root 
 |-- accNo: string (nullable = true) 
 |-- tranAmount: double (nullable = false) 
scala> // Show the first few records of the DataFrame 
scala> acTransDF.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Use SQL to create another DataFrame containing the good transaction records 
scala> val goodTransRecords = spark.sql("SELECT accNo, tranAmount FROM trans WHERE accNo like 'SB%' AND tranAmount > 0") 
goodTransRecords: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Register temporary view in the DataFrame for using it in SQL 
scala> goodTransRecords.createOrReplaceTempView("goodtrans") 
scala> // Show the first few records of the DataFrame 
scala> goodTransRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
+-------+----------+ 
scala> // Use SQL to create another DataFrame containing the high value transaction records 
scala> val highValueTransRecords = spark.sql("SELECT accNo, tranAmount FROM goodtrans WHERE tranAmount > 1000") 
highValueTransRecords: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> highValueTransRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10006|   10000.0| 
|SB10010|    7000.0| 
+-------+----------+ 
scala> // Use SQL to create another DataFrame containing the bad account records 
scala> val badAccountRecords = spark.sql("SELECT accNo, tranAmount FROM trans WHERE accNo NOT like 'SB%'") 
badAccountRecords: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> badAccountRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
+-------+----------+ 
scala> // Use SQL to create another DataFrame containing the bad amount records 
scala> val badAmountRecords = spark.sql("SELECT accNo, tranAmount FROM trans WHERE tranAmount < 0") 
badAmountRecords: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> badAmountRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Do the union of two DataFrames and create another DataFrame 
scala> val badTransRecords = badAccountRecords.union(badAmountRecords) 
badTransRecords: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> badTransRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Calculate the sum 
scala> val sumAmount = spark.sql("SELECT sum(tranAmount) as sum FROM goodtrans") 
sumAmount: org.apache.spark.sql.DataFrame = [sum: double] 
scala> // Show the first few records of the DataFrame 
scala> sumAmount.show 
+-------+ 
|    sum| 
+-------+ 
|28486.0| 
+-------+ 
scala> // Calculate the maximum 
scala> val maxAmount = spark.sql("SELECT max(tranAmount) as max FROM goodtrans") 
maxAmount: org.apache.spark.sql.DataFrame = [max: double] 
scala> // Show the first few records of the DataFrame 
scala> maxAmount.show 
+-------+ 
|    max| 
+-------+ 
|10000.0| 
+-------+ 
scala> // Calculate the minimum 
scala> val minAmount = spark.sql("SELECT min(tranAmount) as min FROM goodtrans") 
minAmount: org.apache.spark.sql.DataFrame = [min: double] 
scala> // Show the first few records of the DataFrame 
scala> minAmount.show 
+----+ 
| min| 
+----+ 
|30.0| 
+----+ 
scala> // Use SQL to create another DataFrame containing the good account numbers 
scala> val goodAccNos = spark.sql("SELECT DISTINCT accNo FROM trans WHERE accNo like 'SB%' ORDER BY accNo") 
goodAccNos: org.apache.spark.sql.DataFrame = [accNo: string] 
scala> // Show the first few records of the DataFrame 
scala> goodAccNos.show 
+-------+ 
|  accNo| 
+-------+ 
|SB10001| 
|SB10002| 
|SB10003| 
|SB10004| 
|SB10005| 
|SB10006| 
|SB10007| 
|SB10008| 
|SB10009| 
|SB10010| 
+-------+ 
scala> // Calculate the aggregates using mixing of DataFrame and RDD like operations 
scala> val sumAmountByMixing = goodTransRecords.map(trans => trans.getAsDouble).reduce(_ + _) 
sumAmountByMixing: Double = 28486.0 
scala> val maxAmountByMixing = goodTransRecords.map(trans => trans.getAsDouble).reduce((a, b) => if (a > b) a else b) 
maxAmountByMixing: Double = 10000.0 
scala> val minAmountByMixing = goodTransRecords.map(trans => trans.getAsDouble).reduce((a, b) => if (a < b) a else b) 
minAmountByMixing: Double = 30.0 

```

零售银行交易记录包含账户号码、交易金额，并使用 SparkSQL 处理以获得用例所需的预期结果。以下是前面脚本所做操作的摘要：

+   定义了一个 Scala 案例类来描述要输入到 DataFrame 中的交易记录的结构。

+   使用必要的交易记录定义了一个数组。

+   RDD 是从数组中生成的，将逗号分隔的值拆分，使用在脚本的第一步中定义的 Scala 案例类映射创建对象，并将 RDD 转换为 DataFrame。这是 RDD 和 DataFrame 之间互操作性的一个用例。

+   使用名称注册了一个表与 DataFrame。这个注册的表名可以在 SQL 语句中使用。

+   然后，所有其他活动只是使用`spark.sql`方法发出 SQL 语句。在这里，spark 对象是 SparkSession 类型。

+   所有这些 SQL 语句的结果存储为 DataFrame，就像 RDD 的`collect`操作一样，DataFrame 的`show`方法用于将值提取到 Spark 驱动程序中。

+   聚合值计算以两种不同的方式进行。一种是在 SQL 语句方式中，这是最简单的方式。另一种是使用常规的 RDD 风格的 Spark 转换和 Spark 操作。这表明 DataFrame 也可以像 RDD 一样操作，Spark 转换和 Spark 操作可以应用于 DataFrame 之上。

+   有时，通过函数式风格的运算使用函数进行一些数据处理活动是很简单的。因此，这里有一个灵活性，可以混合 SQL、RDD 和 DataFrame，以获得一个非常方便的编程模型来处理数据。

+   使用 DataFrame 的`show`方法以表格格式显示 DataFrame 的内容。

+   使用`printSchema`方法显示了 DataFrame 的结构视图。这类似于数据库表的`describe`命令。

在 Python 交互式解释器提示符下，尝试以下语句：

```py
>>> from pyspark.sql import Row 
>>> # Creation of the list from where the RDD is going to be created 
>>> acTransList = ["SB10001,1000", "SB10002,1200", "SB10003,8000", "SB10004,400", "SB10005,300", "SB10006,10000", "SB10007,500", "SB10008,56", "SB10009,30","SB10010,7000", "CR10001,7000", "SB10002,-10"] 
>>> # Create the DataFrame 
>>> acTransDF = sc.parallelize(acTransList).map(lambda trans: trans.split(",")).map(lambda p: Row(accNo=p[0], tranAmount=float(p[1]))).toDF() 
>>> # Register temporary view in the DataFrame for using it in SQL 
>>> acTransDF.createOrReplaceTempView("trans") 
>>> # Print the structure of the DataFrame 
>>> acTransDF.printSchema() 
root 
 |-- accNo: string (nullable = true) 
 |-- tranAmount: double (nullable = true) 
>>> # Show the first few records of the DataFrame 
>>> acTransDF.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
>>> # Use SQL to create another DataFrame containing the good transaction records 
>>> goodTransRecords = spark.sql("SELECT accNo, tranAmount FROM trans WHERE accNo like 'SB%' AND tranAmount > 0") 
>>> # Register temporary table in the DataFrame for using it in SQL 
>>> goodTransRecords.createOrReplaceTempView("goodtrans") 
>>> # Show the first few records of the DataFrame 
>>> goodTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
+-------+----------+ 
>>> # Use SQL to create another DataFrame containing the high value transaction records 
>>> highValueTransRecords = spark.sql("SELECT accNo, tranAmount FROM goodtrans WHERE tranAmount > 1000") 
>>> # Show the first few records of the DataFrame 
>>> highValueTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10006|   10000.0| 
|SB10010|    7000.0| 
+-------+----------+ 
>>> # Use SQL to create another DataFrame containing the bad account records 
>>> badAccountRecords = spark.sql("SELECT accNo, tranAmount FROM trans WHERE accNo NOT like 'SB%'") 
>>> # Show the first few records of the DataFrame 
>>> badAccountRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
+-------+----------+ 
>>> # Use SQL to create another DataFrame containing the bad amount records 
>>> badAmountRecords = spark.sql("SELECT accNo, tranAmount FROM trans WHERE tranAmount < 0") 
>>> # Show the first few records of the DataFrame 
>>> badAmountRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|     -10.0| 
+-------+----------+ 
>>> # Do the union of two DataFrames and create another DataFrame 
>>> badTransRecords = badAccountRecords.union(badAmountRecords) 
>>> # Show the first few records of the DataFrame 
>>> badTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
>>> # Calculate the sum 
>>> sumAmount = spark.sql("SELECT sum(tranAmount)as sum FROM goodtrans") 
>>> # Show the first few records of the DataFrame 
>>> sumAmount.show() 
+-------+ 
|    sum| 
+-------+ 
|28486.0| 
+-------+ 
>>> # Calculate the maximum 
>>> maxAmount = spark.sql("SELECT max(tranAmount) as max FROM goodtrans") 
>>> # Show the first few records of the DataFrame 
>>> maxAmount.show() 
+-------+ 
|    max| 
+-------+ 
|10000.0| 
+-------+ 
>>> # Calculate the minimum 
>>> minAmount = spark.sql("SELECT min(tranAmount)as min FROM goodtrans") 
>>> # Show the first few records of the DataFrame 
>>> minAmount.show() 
+----+ 
| min| 
+----+ 
|30.0| 
+----+ 
>>> # Use SQL to create another DataFrame containing the good account numbers 
>>> goodAccNos = spark.sql("SELECT DISTINCT accNo FROM trans WHERE accNo like 'SB%' ORDER BY accNo") 
>>> # Show the first few records of the DataFrame 
>>> goodAccNos.show() 
+-------+ 
|  accNo| 
+-------+ 
|SB10001| 
|SB10002| 
|SB10003| 
|SB10004| 
|SB10005| 
|SB10006| 
|SB10007| 
|SB10008| 
|SB10009| 
|SB10010| 
+-------+ 
>>> # Calculate the sum using mixing of DataFrame and RDD like operations 
>>> sumAmountByMixing = goodTransRecords.rdd.map(lambda trans: trans.tranAmount).reduce(lambda a,b : a+b) 
>>> sumAmountByMixing 
28486.0 
>>> # Calculate the maximum using mixing of DataFrame and RDD like operations 
>>> maxAmountByMixing = goodTransRecords.rdd.map(lambda trans: trans.tranAmount).reduce(lambda a,b : a if a > b else b) 
>>> maxAmountByMixing 
10000.0 
>>> # Calculate the minimum using mixing of DataFrame and RDD like operations 
>>> minAmountByMixing = goodTransRecords.rdd.map(lambda trans: trans.tranAmount).reduce(lambda a,b : a if a < b else b) 
>>> minAmountByMixing 
30.0 

```

在前面的 Python 代码片段中，除了导入库和 lambda 函数定义等一些语言特定的结构之外，编程风格几乎与 Scala 代码相同，大多数时候都是如此。这是 Spark 统一编程模型的优势。如前所述，当业务分析师或数据分析师提供数据访问的 SQL 时，很容易将其与 Spark 中的数据处理代码集成。这种统一的编程风格对组织使用所选语言在 Spark 中开发数据处理应用程序非常有用。

### 小贴士

在 DataFrame 上，如果适用 Spark 转换，则返回 Dataset 而不是 DataFrame。Dataset 的概念在本章末尾介绍。DataFrame 和 Dataset 之间有非常紧密的联系，这一点在介绍 Dataset 的章节中解释。在开发应用程序时，必须小心处理这种情况。例如，在 Scala REPL 中尝试前面的代码片段中的以下转换时，它将返回一个数据集：`val amount = goodTransRecords.map(trans => trans.getAsDouble)amount: org.apache.spark.sql.Dataset[Double] = [value: double]`

## 使用 DataFrame API 编程

在本节中，代码片段将在适当的语言 REPL 中运行，作为上一节的延续，这样就不需要重复设置数据和其它初始化。与前面的代码片段类似，最初给出一些 DataFrame 特定的基本命令。这些命令被经常使用，用于查看内容并对 DataFrame 及其内容进行一些基本测试。这些是在数据分析的探索阶段通常使用的命令，常常用于深入了解底层数据的结构和内容。

在 Scala REPL 提示符下，尝试以下语句：

```py
scala> acTransDF.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Create the DataFrame using API for the good transaction records 
scala> val goodTransRecords = acTransDF.filter("accNo like 'SB%'").filter("tranAmount > 0") 
goodTransRecords: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> goodTransRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
+-------+----------+ 
scala> // Create the DataFrame using API for the high value transaction records 
scala> val highValueTransRecords = goodTransRecords.filter("tranAmount > 1000") 
highValueTransRecords: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> highValueTransRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10006|   10000.0| 
|SB10010|    7000.0| 
+-------+----------+ 
scala> // Create the DataFrame using API for the bad account records 
scala> val badAccountRecords = acTransDF.filter("accNo NOT like 'SB%'") 
badAccountRecords: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> badAccountRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
+-------+----------+ 
scala> // Create the DataFrame using API for the bad amount records 
scala> val badAmountRecords = acTransDF.filter("tranAmount < 0") 
badAmountRecords: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> badAmountRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Do the union of two DataFrames 
scala> val badTransRecords = badAccountRecords.union(badAmountRecords) 
badTransRecords: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> badTransRecords.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Calculate the aggregates in one shot 
scala> val aggregates = goodTransRecords.agg(sum("tranAmount"), max("tranAmount"), min("tranAmount")) 
aggregates: org.apache.spark.sql.DataFrame = [sum(tranAmount): double, max(tranAmount): double ... 1 more field] 
scala> // Show the first few records of the DataFrame 
scala> aggregates.show 
+---------------+---------------+---------------+ 
|sum(tranAmount)|max(tranAmount)|min(tranAmount)| 
+---------------+---------------+---------------+ 
|        28486.0|        10000.0|           30.0| 
+---------------+---------------+---------------+ 
scala> // Use DataFrame using API for creating the good account numbers 
scala> val goodAccNos = acTransDF.filter("accNo like 'SB%'").select("accNo").distinct().orderBy("accNo") 
goodAccNos: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string] 
scala> // Show the first few records of the DataFrame 
scala> goodAccNos.show 
+-------+ 
|  accNo| 
+-------+ 
|SB10001| 
|SB10002| 
|SB10003| 
|SB10004| 
|SB10005| 
|SB10006| 
|SB10007| 
|SB10008| 
|SB10009| 
|SB10010| 
+-------+ 
scala> // Persist the data of the DataFrame into a Parquet file 
scala> acTransDF.write.parquet("scala.trans.parquet") 
scala> // Read the data into a DataFrame from the Parquet file 
scala> val acTransDFfromParquet = spark.read.parquet("scala.trans.parquet") 
acTransDFfromParquet: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> acTransDFfromParquet.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
|SB10001|    1000.0| 
|SB10004|     400.0| 
|SB10007|     500.0| 
|SB10010|    7000.0| 
+-------+----------+

```

下面是从 DataFrame API 视角对前面脚本所做操作的总结：

+   本处使用的是包含前面章节中使用的数据超集的 DataFrame。

+   接下来演示了记录的过滤。这里，需要注意的最重要的一点是，过滤谓词必须与 SQL 语句中的谓词完全相同。过滤器可以串联使用。

+   聚合方法一次性计算为结果 DataFrame 中的三个列。

+   本组中的最后几个语句在一个单链语句中执行选择、过滤、选择不同记录和排序操作。

+   最后，事务记录以 Parquet 格式持久化，从 Parquet 存储中读取并创建一个 DataFrame。关于持久化格式的更多细节将在下一节中介绍。

+   在此代码片段中，Parquet 格式的数据存储在从相应 REPL 调用的当前目录中。当它作为一个 Spark 程序运行时，目录再次将是从该目录调用 Spark submit 的当前目录。

在 Python REPL 提示符下，尝试以下语句：

```py
>>> acTransDF.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
>>> # Print the structure of the DataFrame 
>>> acTransDF.printSchema() 
root 
 |-- accNo: string (nullable = true) 
 |-- tranAmount: double (nullable = true) 
>>> # Create the DataFrame using API for the good transaction records 
>>> goodTransRecords = acTransDF.filter("accNo like 'SB%'").filter("tranAmount > 0") 
>>> # Show the first few records of the DataFrame 
>>> goodTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
+-------+----------+ 
>>> # Create the DataFrame using API for the high value transaction records 
>>> highValueTransRecords = goodTransRecords.filter("tranAmount > 1000") 
>>> # Show the first few records of the DataFrame 
>>> highValueTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10006|   10000.0| 
|SB10010|    7000.0| 
+-------+----------+ 
>>> # Create the DataFrame using API for the bad account records 
>>> badAccountRecords = acTransDF.filter("accNo NOT like 'SB%'") 
>>> # Show the first few records of the DataFrame 
>>> badAccountRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
+-------+----------+ 
>>> # Create the DataFrame using API for the bad amount records 
>>> badAmountRecords = acTransDF.filter("tranAmount < 0") 
>>> # Show the first few records of the DataFrame 
>>> badAmountRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|     -10.0| 
+-------+----------+ 
>>> # Do the union of two DataFrames and create another DataFrame 
>>> badTransRecords = badAccountRecords.union(badAmountRecords) 
>>> # Show the first few records of the DataFrame 
>>> badTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
>>> # Calculate the sum 
>>> sumAmount = goodTransRecords.agg({"tranAmount": "sum"}) 
>>> # Show the first few records of the DataFrame 
>>> sumAmount.show() 
+---------------+ 
|sum(tranAmount)| 
+---------------+ 
|        28486.0| 
+---------------+ 
>>> # Calculate the maximum 
>>> maxAmount = goodTransRecords.agg({"tranAmount": "max"}) 
>>> # Show the first few records of the DataFrame 
>>> maxAmount.show() 
+---------------+ 
|max(tranAmount)| 
+---------------+ 
|        10000.0| 
+---------------+ 
>>> # Calculate the minimum 
>>> minAmount = goodTransRecords.agg({"tranAmount": "min"}) 
>>> # Show the first few records of the DataFrame 
>>> minAmount.show() 
+---------------+ 
|min(tranAmount)| 
+---------------+ 
|           30.0| 
+---------------+ 
>>> # Create the DataFrame using API for the good account numbers 
>>> goodAccNos = acTransDF.filter("accNo like 'SB%'").select("accNo").distinct().orderBy("accNo") 
>>> # Show the first few records of the DataFrame 
>>> goodAccNos.show() 
+-------+ 
|  accNo| 
+-------+ 
|SB10001| 
|SB10002| 
|SB10003| 
|SB10004| 
|SB10005| 
|SB10006| 
|SB10007| 
|SB10008| 
|SB10009| 
|SB10010| 
+-------+ 
>>> # Persist the data of the DataFrame into a Parquet file 
>>> acTransDF.write.parquet("python.trans.parquet") 
>>> # Read the data into a DataFrame from the Parquet file 
>>> acTransDFfromParquet = spark.read.parquet("python.trans.parquet") 
>>> # Show the first few records of the DataFrame 
>>> acTransDFfromParquet.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
|SB10001|    1000.0| 
|SB10004|     400.0| 
|SB10007|     500.0| 
|SB10010|    7000.0| 
+-------+----------+ 

```

在前面的 Python 代码片段中，除了聚合计算中的一些细微差异外，编程结构几乎与 Scala 对应版本相似。

前面的 Scala 和 Python 部分的最后几行是关于将 DataFrame 内容持久化到媒体中的。在任何类型的数据处理操作中，写入和读取操作都是非常必要的，但大多数工具都没有统一的写入和读取方式。Spark SQL 是不同的。DataFrame API 提供了一套丰富的持久化机制。将 DataFrame 的内容写入许多支持的持久化存储非常简单。所有这些写入和读取操作都有非常简单的 DSL 风格接口。以下是一些 DataFrame 可以写入和读取的内置格式。

除了这些，还有许多其他通过第三方包支持的外部数据源：

+   JSON

+   Parquet

+   Hive

+   MySQL

+   PostgreSQL

+   HDFS

+   纯文本

+   Amazon S3

+   ORC

+   JDBC

在前面的代码片段中已经演示了 DataFrame 到 Parquet 以及从 Parquet 读写的过程。所有之前内在支持的数据存储都有非常简单的 DSL 风格语法用于持久化和读取，这使得编程风格再次统一。DataFrame API 参考是了解如何处理每个数据存储细节的绝佳来源。

本章中的示例代码以 Parquet 和 JSON 格式持久化数据。所选的数据存储位置名称为 `python.trans.parquet`、`scala.trans.parquet` 等。这只是为了表明使用了哪种编程语言以及数据的格式。这并不是一个正确的约定，而是一种便利。当程序的一次运行完成后，这些目录就会被创建。下次运行相同的程序时，它将尝试创建相同的目录，并导致错误。解决方案是在后续运行之前手动删除这些目录，然后继续。适当的错误处理机制和精细编程的其他细微之处可能会分散注意力，因此故意从本书中省略。

# 理解 Spark SQL 中的聚合

在 SQL 中，数据的聚合非常灵活。在 Spark SQL 中也是如此。在这里，Spark SQL 可以在分布式数据源上执行与在单个机器上的单个数据源上运行 SQL 语句相同的事情。在前一章中，讨论了一个 MapReduce 用例来进行数据聚合，这里同样使用它来展示 Spark SQL 的聚合能力。在本节中，用例也是以 SQL 查询方式和 DataFrame API 方式来处理的。

在此处阐述 MapReduce 类型的数据处理所选择的用例如下：

+   零售银行交易记录包含账户号码和以逗号分隔的交易金额字符串

+   找到所有交易的账户级别摘要以获取账户余额

在 Scala REPL 提示符下，尝试以下语句：

```py
scala> // Define the case classes for using in conjunction with DataFrames 
scala> case class Trans(accNo: String, tranAmount: Double) 
defined class Trans 
scala> // Functions to convert the sequence of strings to objects defined by the case classes 
scala> def toTrans =  (trans: Seq[String]) => Trans(trans(0), trans(1).trim.toDouble) 
toTrans: Seq[String] => Trans 
scala> // Creation of the list from where the RDD is going to be created 
scala> val acTransList = Array("SB10001,1000", "SB10002,1200","SB10001,8000", "SB10002,400", "SB10003,300", "SB10001,10000","SB10004,500","SB10005,56", "SB10003,30","SB10002,7000","SB10001,-100", "SB10002,-10") 
acTransList: Array[String] = Array(SB10001,1000, SB10002,1200, SB10001,8000, SB10002,400, SB10003,300, SB10001,10000, SB10004,500, SB10005,56, SB10003,30, SB10002,7000, SB10001,-100, SB10002,-10) 
scala> // Create the DataFrame 
scala> val acTransDF = sc.parallelize(acTransList).map(_.split(",")).map(toTrans(_)).toDF() 
acTransDF: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Show the first few records of the DataFrame 
scala> acTransDF.show 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10001|    8000.0| 
|SB10002|     400.0| 
|SB10003|     300.0| 
|SB10001|   10000.0| 
|SB10004|     500.0| 
|SB10005|      56.0| 
|SB10003|      30.0| 
|SB10002|    7000.0| 
|SB10001|    -100.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Register temporary view in the DataFrame for using it in SQL 
scala> acTransDF.createOrReplaceTempView("trans") 
scala> // Use SQL to create another DataFrame containing the account summary records 
scala> val acSummary = spark.sql("SELECT accNo, sum(tranAmount) as TransTotal FROM trans GROUP BY accNo") 
acSummary: org.apache.spark.sql.DataFrame = [accNo: string, TransTotal: double] 
scala> // Show the first few records of the DataFrame 
scala> acSummary.show 
+-------+----------+ 
|  accNo|TransTotal| 
+-------+----------+ 
|SB10005|      56.0| 
|SB10004|     500.0| 
|SB10003|     330.0| 
|SB10002|    8590.0| 
|SB10001|   18900.0| 
+-------+----------+ 
scala> // Create the DataFrame using API for the account summary records 
scala> val acSummaryViaDFAPI = acTransDF.groupBy("accNo").agg(sum("tranAmount") as "TransTotal") 
acSummaryViaDFAPI: org.apache.spark.sql.DataFrame = [accNo: string, TransTotal: double] 
scala> // Show the first few records of the DataFrame 
scala> acSummaryViaDFAPI.show 
+-------+----------+ 
|  accNo|TransTotal| 
+-------+----------+ 
|SB10005|      56.0| 
|SB10004|     500.0| 
|SB10003|     330.0| 
|SB10002|    8590.0| 
|SB10001|   18900.0| 
+-------+----------+

```

在此代码片段中，一切与前面章节的代码非常相似。唯一的区别是，在这里，SQL 查询以及 DataFrame API 中都使用了聚合操作。

在 Python REPL 提示符下，尝试以下语句：

```py
>>> from pyspark.sql import Row 
>>> # Creation of the list from where the RDD is going to be created 
>>> acTransList = ["SB10001,1000", "SB10002,1200", "SB10001,8000","SB10002,400", "SB10003,300", "SB10001,10000","SB10004,500","SB10005,56","SB10003,30","SB10002,7000", "SB10001,-100","SB10002,-10"] 
>>> # Create the DataFrame 
>>> acTransDF = sc.parallelize(acTransList).map(lambda trans: trans.split(",")).map(lambda p: Row(accNo=p[0], tranAmount=float(p[1]))).toDF() 
>>> # Register temporary view in the DataFrame for using it in SQL 
>>> acTransDF.createOrReplaceTempView("trans") 
>>> # Use SQL to create another DataFrame containing the account summary records 
>>> acSummary = spark.sql("SELECT accNo, sum(tranAmount) as transTotal FROM trans GROUP BY accNo") 
>>> # Show the first few records of the DataFrame 
>>> acSummary.show()     
+-------+----------+ 
|  accNo|transTotal| 
+-------+----------+ 
|SB10005|      56.0| 
|SB10004|     500.0| 
|SB10003|     330.0| 
|SB10002|    8590.0| 
|SB10001|   18900.0| 
+-------+----------+ 
>>> # Create the DataFrame using API for the account summary records 
>>> acSummaryViaDFAPI = acTransDF.groupBy("accNo").agg({"tranAmount": "sum"}).selectExpr("accNo", "`sum(tranAmount)` as transTotal") 
>>> # Show the first few records of the DataFrame 
>>> acSummaryViaDFAPI.show() 
+-------+----------+ 
|  accNo|transTotal| 
+-------+----------+ 
|SB10005|      56.0| 
|SB10004|     500.0| 
|SB10003|     330.0| 
|SB10002|    8590.0| 
|SB10001|   18900.0| 
+-------+----------+

```

在 Python 的 DataFrame API 中，与 Scala 的对应版本相比，有一些微小的语法差异。

# 理解 SparkSQL 中的多数据源连接

在前一章中，已经讨论了基于键的多个 RDD 的连接。在本节中，使用 Spark SQL 实现了相同的用例。这里给出的用于阐明使用键连接多个数据集的用例已选定。

第一个数据集包含一个零售银行主记录摘要，包括账户号码、名和姓。第二个数据集包含零售银行账户余额，包括账户号码和余额金额。这两个数据集的关键是账户号码。将这两个数据集连接起来，创建一个包含账户号码、名、姓和余额金额的数据集。从这个报告中，挑选出余额金额最高的前三个账户。

在本节中，还演示了从多个数据源连接数据的概念。首先，从两个数组创建 DataFrame。它们以 Parquet 和 JSON 格式持久化。然后，从磁盘读取它们以形成 DataFrame，并将它们连接起来。

在 Scala REPL 提示符下，尝试以下语句：

```py
scala> // Define the case classes for using in conjunction with DataFrames 
scala> case class AcMaster(accNo: String, firstName: String, lastName: String) 
defined class AcMaster 
scala> case class AcBal(accNo: String, balanceAmount: Double) 
defined class AcBal 
scala> // Functions to convert the sequence of strings to objects defined by the case classes 
scala> def toAcMaster =  (master: Seq[String]) => AcMaster(master(0), master(1), master(2)) 
toAcMaster: Seq[String] => AcMaster 
scala> def toAcBal =  (bal: Seq[String]) => AcBal(bal(0), bal(1).trim.toDouble) 
toAcBal: Seq[String] => AcBal 
scala> // Creation of the list from where the RDD is going to be created 
scala> val acMasterList = Array("SB10001,Roger,Federer","SB10002,Pete,Sampras", "SB10003,Rafael,Nadal","SB10004,Boris,Becker", "SB10005,Ivan,Lendl") 
acMasterList: Array[String] = Array(SB10001,Roger,Federer, SB10002,Pete,Sampras, SB10003,Rafael,Nadal, SB10004,Boris,Becker, SB10005,Ivan,Lendl) 
scala> // Creation of the list from where the RDD is going to be created 
scala> val acBalList = Array("SB10001,50000", "SB10002,12000","SB10003,3000", "SB10004,8500", "SB10005,5000") 
acBalList: Array[String] = Array(SB10001,50000, SB10002,12000, SB10003,3000, SB10004,8500, SB10005,5000) 
scala> // Create the DataFrame 
scala> val acMasterDF = sc.parallelize(acMasterList).map(_.split(",")).map(toAcMaster(_)).toDF() 
acMasterDF: org.apache.spark.sql.DataFrame = [accNo: string, firstName: string ... 1 more field] 
scala> // Create the DataFrame 
scala> val acBalDF = sc.parallelize(acBalList).map(_.split(",")).map(toAcBal(_)).toDF() 
acBalDF: org.apache.spark.sql.DataFrame = [accNo: string, balanceAmount: double] 
scala> // Persist the data of the DataFrame into a Parquet file 
scala> acMasterDF.write.parquet("scala.master.parquet") 
scala> // Persist the data of the DataFrame into a JSON file 
scala> acBalDF.write.json("scalaMaster.json") 
scala> // Read the data into a DataFrame from the Parquet file 
scala> val acMasterDFFromFile = spark.read.parquet("scala.master.parquet") 
acMasterDFFromFile: org.apache.spark.sql.DataFrame = [accNo: string, firstName: string ... 1 more field] 
scala> // Register temporary view in the DataFrame for using it in SQL 
scala> acMasterDFFromFile.createOrReplaceTempView("master") 
scala> // Read the data into a DataFrame from the JSON file 
scala> val acBalDFFromFile = spark.read.json("scalaMaster.json") 
acBalDFFromFile: org.apache.spark.sql.DataFrame = [accNo: string, balanceAmount: double] 
scala> // Register temporary view in the DataFrame for using it in SQL 
scala> acBalDFFromFile.createOrReplaceTempView("balance") 
scala> // Show the first few records of the DataFrame 
scala> acMasterDFFromFile.show 
+-------+---------+--------+ 
|  accNo|firstName|lastName| 
+-------+---------+--------+ 
|SB10001|    Roger| Federer| 
|SB10002|     Pete| Sampras| 
|SB10003|   Rafael|   Nadal| 
|SB10004|    Boris|  Becker| 
|SB10005|     Ivan|   Lendl| 
+-------+---------+--------+ 
scala> acBalDFFromFile.show 
+-------+-------------+ 
|  accNo|balanceAmount| 
+-------+-------------+ 
|SB10001|      50000.0| 
|SB10002|      12000.0| 
|SB10003|       3000.0| 
|SB10004|       8500.0| 
|SB10005|       5000.0| 
+-------+-------------+ 
scala> // Use SQL to create another DataFrame containing the account detail records 
scala> val acDetail = spark.sql("SELECT master.accNo, firstName, lastName, balanceAmount FROM master, balance WHERE master.accNo = balance.accNo ORDER BY balanceAmount DESC") 
acDetail: org.apache.spark.sql.DataFrame = [accNo: string, firstName: string ... 2 more fields] 
scala> // Show the first few records of the DataFrame 
scala> acDetail.show 
+-------+---------+--------+-------------+ 
|  accNo|firstName|lastName|balanceAmount| 
+-------+---------+--------+-------------+ 
|SB10001|    Roger| Federer|      50000.0| 
|SB10002|     Pete| Sampras|      12000.0| 
|SB10004|    Boris|  Becker|       8500.0| 
|SB10005|     Ivan|   Lendl|       5000.0| 
|SB10003|   Rafael|   Nadal|       3000.0| 
+-------+---------+--------+-------------+

```

继续使用相同的 Scala REPL 会话，以下代码行通过 DataFrame API 获取相同的结果：

```py
scala> // Create the DataFrame using API for the account detail records 
scala> val acDetailFromAPI = acMasterDFFromFile.join(acBalDFFromFile, acMasterDFFromFile("accNo") === acBalDFFromFile("accNo"), "inner").sort($"balanceAmount".desc).select(acMasterDFFromFile("accNo"), acMasterDFFromFile("firstName"), acMasterDFFromFile("lastName"), acBalDFFromFile("balanceAmount")) 
acDetailFromAPI: org.apache.spark.sql.DataFrame = [accNo: string, firstName: string ... 2 more fields] 
scala> // Show the first few records of the DataFrame 
scala> acDetailFromAPI.show 
+-------+---------+--------+-------------+ 
|  accNo|firstName|lastName|balanceAmount| 
+-------+---------+--------+-------------+ 
|SB10001|    Roger| Federer|      50000.0| 
|SB10002|     Pete| Sampras|      12000.0| 
|SB10004|    Boris|  Becker|       8500.0| 
|SB10005|     Ivan|   Lendl|       5000.0| 
|SB10003|   Rafael|   Nadal|       3000.0| 
+-------+---------+--------+-------------+ 
scala> // Use SQL to create another DataFrame containing the top 3 account detail records 
scala> val acDetailTop3 = spark.sql("SELECT master.accNo, firstName, lastName, balanceAmount FROM master, balance WHERE master.accNo = balance.accNo ORDER BY balanceAmount DESC").limit(3) 
acDetailTop3: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [accNo: string, firstName: string ... 2 more fields] 
scala> // Show the first few records of the DataFrame 
scala> acDetailTop3.show 
+-------+---------+--------+-------------+ 
|  accNo|firstName|lastName|balanceAmount| 
+-------+---------+--------+-------------+ 
|SB10001|    Roger| Federer|      50000.0| 
|SB10002|     Pete| Sampras|      12000.0| 
|SB10004|    Boris|  Becker|       8500.0| 
+-------+---------+--------+-------------+

```

在代码的前一部分中选定的连接类型是内连接。而不是这样，可以通过 SQL 查询方式或 DataFrame API 方式使用任何其他类型的连接。在这个特定用例中，可以看到 DataFrame API 变得有点笨拙，而 SQL 查询看起来非常直接。这里的要点是，根据具体情况，在应用程序代码中，可以混合使用 SQL 查询方式和 DataFrame API 方式来产生所需的结果。以下脚本中给出的 DataFrame `acDetailTop3` 是一个例子。

在 Python REPL 提示符下，尝试以下语句：

```py
>>> from pyspark.sql import Row 
>>> # Creation of the list from where the RDD is going to be created 
>>> AcMaster = Row('accNo', 'firstName', 'lastName') 
>>> AcBal = Row('accNo', 'balanceAmount') 
>>> acMasterList = ["SB10001,Roger,Federer","SB10002,Pete,Sampras", "SB10003,Rafael,Nadal","SB10004,Boris,Becker", "SB10005,Ivan,Lendl"] 
>>> acBalList = ["SB10001,50000", "SB10002,12000","SB10003,3000", "SB10004,8500", "SB10005,5000"] 
>>> # Create the DataFrame 
>>> acMasterDF = sc.parallelize(acMasterList).map(lambda trans: trans.split(",")).map(lambda r: AcMaster(*r)).toDF() 
>>> acBalDF = sc.parallelize(acBalList).map(lambda trans: trans.split(",")).map(lambda r: AcBal(r[0], float(r[1]))).toDF() 
>>> # Persist the data of the DataFrame into a Parquet file 
>>> acMasterDF.write.parquet("python.master.parquet") 
>>> # Persist the data of the DataFrame into a JSON file 
>>> acBalDF.write.json("pythonMaster.json") 
>>> # Read the data into a DataFrame from the Parquet file 
>>> acMasterDFFromFile = spark.read.parquet("python.master.parquet") 
>>> # Register temporary table in the DataFrame for using it in SQL 
>>> acMasterDFFromFile.createOrReplaceTempView("master") 
>>> # Register temporary table in the DataFrame for using it in SQL 
>>> acBalDFFromFile = spark.read.json("pythonMaster.json") 
>>> # Register temporary table in the DataFrame for using it in SQL 
>>> acBalDFFromFile.createOrReplaceTempView("balance") 
>>> # Show the first few records of the DataFrame 
>>> acMasterDFFromFile.show() 
+-------+---------+--------+ 
|  accNo|firstName|lastName| 
+-------+---------+--------+ 
|SB10001|    Roger| Federer| 
|SB10002|     Pete| Sampras| 
|SB10003|   Rafael|   Nadal| 
|SB10004|    Boris|  Becker| 
|SB10005|     Ivan|   Lendl| 
+-------+---------+--------+ 
>>> # Show the first few records of the DataFrame 
>>> acBalDFFromFile.show() 
+-------+-------------+ 
|  accNo|balanceAmount| 
+-------+-------------+ 
|SB10001|      50000.0| 
|SB10002|      12000.0| 
|SB10003|       3000.0| 
|SB10004|       8500.0| 
|SB10005|       5000.0| 
+-------+-------------+ 
>>> # Use SQL to create another DataFrame containing the account detail records 
>>> acDetail = spark.sql("SELECT master.accNo, firstName, lastName, balanceAmount FROM master, balance WHERE master.accNo = balance.accNo ORDER BY balanceAmount DESC") 
>>> # Show the first few records of the DataFrame 
>>> acDetail.show() 
+-------+---------+--------+-------------+ 
|  accNo|firstName|lastName|balanceAmount| 
+-------+---------+--------+-------------+ 
|SB10001|    Roger| Federer|      50000.0| 
|SB10002|     Pete| Sampras|      12000.0| 
|SB10004|    Boris|  Becker|       8500.0| 
|SB10005|     Ivan|   Lendl|       5000.0| 
|SB10003|   Rafael|   Nadal|       3000.0| 
+-------+---------+--------+-------------+ 
>>> # Create the DataFrame using API for the account detail records 
>>> acDetailFromAPI = acMasterDFFromFile.join(acBalDFFromFile, acMasterDFFromFile.accNo == acBalDFFromFile.accNo).sort(acBalDFFromFile.balanceAmount, ascending=False).select(acMasterDFFromFile.accNo, acMasterDFFromFile.firstName, acMasterDFFromFile.lastName, acBalDFFromFile.balanceAmount) 
>>> # Show the first few records of the DataFrame 
>>> acDetailFromAPI.show() 
+-------+---------+--------+-------------+ 
|  accNo|firstName|lastName|balanceAmount| 
+-------+---------+--------+-------------+ 
|SB10001|    Roger| Federer|      50000.0| 
|SB10002|     Pete| Sampras|      12000.0| 
|SB10004|    Boris|  Becker|       8500.0| 
|SB10005|     Ivan|   Lendl|       5000.0| 
|SB10003|   Rafael|   Nadal|       3000.0| 
+-------+---------+--------+-------------+ 
>>> # Use SQL to create another DataFrame containing the top 3 account detail records 
>>> acDetailTop3 = spark.sql("SELECT master.accNo, firstName, lastName, balanceAmount FROM master, balance WHERE master.accNo = balance.accNo ORDER BY balanceAmount DESC").limit(3) 
>>> # Show the first few records of the DataFrame 
>>> acDetailTop3.show() 
+-------+---------+--------+-------------+ 
|  accNo|firstName|lastName|balanceAmount| 
+-------+---------+--------+-------------+ 
|SB10001|    Roger| Federer|      50000.0| 
|SB10002|     Pete| Sampras|      12000.0| 
|SB10004|    Boris|  Becker|       8500.0| 
+-------+---------+--------+-------------+ 

```

在前面的章节中，已经展示了在 DataFrame 上应用 RDD 操作。这显示了 Spark SQL 与 RDDs 交互以及反之亦然的能力。同样，SQL 查询和 DataFrame API 可以混合使用，以便在解决应用程序中的实际用例时，能够灵活地使用计算的最简单方法。

# 介绍数据集

当涉及到开发数据处理应用程序时，Spark 编程范式提供了许多抽象供选择。Spark 编程的基础始于可以轻松处理非结构化、半结构化和结构化数据的 RDDs。Spark SQL 库在处理结构化数据时提供了高度优化的性能。这使得基本的 RDDs 在性能方面看起来有些不足。为了填补这一差距，从 Spark 1.6 版本开始，引入了一种新的抽象，名为 Dataset，它补充了基于 RDD 的 Spark 编程模型。它在 Spark 转换和 Spark 操作方面几乎与 RDD 相同，同时，它像 Spark SQL 一样高度优化。Dataset API 在编写程序时提供了强大的编译时类型安全性，因此，Dataset API 仅在 Scala 和 Java 中可用。

在涵盖 Spark 编程模型的章节中讨论的交易银行业务案例在此再次提出，以阐明基于 dataset 的编程模型，因为这种编程模型与基于 RDD 的编程非常相似。该案例主要涉及一组银行交易记录以及在这些记录上执行的各种处理，以从中提取各种信息。案例描述在此不再重复，通过查看注释和代码不难理解。

以下代码片段演示了创建 Dataset 所使用的方法，以及它的使用、将 RDD 转换为 DataFrame 以及将 DataFrame 转换为 dataset 的过程。RDD 到 DataFrame 的转换已经讨论过，但在此再次捕获以保持概念的一致性。这主要是为了证明 Spark 中的各种编程模型和数据抽象具有高度的互操作性。

在 Scala REPL 提示符下，尝试以下语句：

```py
scala> // Define the case classes for using in conjunction with DataFrames and Dataset 
scala> case class Trans(accNo: String, tranAmount: Double)  
defined class Trans 
scala> // Creation of the list from where the Dataset is going to be created using a case class. 
scala> val acTransList = Seq(Trans("SB10001", 1000), Trans("SB10002",1200), Trans("SB10003", 8000), Trans("SB10004",400), Trans("SB10005",300), Trans("SB10006",10000), Trans("SB10007",500), Trans("SB10008",56), Trans("SB10009",30),Trans("SB10010",7000), Trans("CR10001",7000), Trans("SB10002",-10)) 
acTransList: Seq[Trans] = List(Trans(SB10001,1000.0), Trans(SB10002,1200.0), Trans(SB10003,8000.0), Trans(SB10004,400.0), Trans(SB10005,300.0), Trans(SB10006,10000.0), Trans(SB10007,500.0), Trans(SB10008,56.0), Trans(SB10009,30.0), Trans(SB10010,7000.0), Trans(CR10001,7000.0), Trans(SB10002,-10.0)) 
scala> // Create the Dataset 
scala> val acTransDS = acTransList.toDS() 
acTransDS: org.apache.spark.sql.Dataset[Trans] = [accNo: string, tranAmount: double] 
scala> acTransDS.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Apply filter and create another Dataset of good transaction records 
scala> val goodTransRecords = acTransDS.filter(_.tranAmount > 0).filter(_.accNo.startsWith("SB")) 
goodTransRecords: org.apache.spark.sql.Dataset[Trans] = [accNo: string, tranAmount: double] 
scala> goodTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
+-------+----------+ 
scala> // Apply filter and create another Dataset of high value transaction records 
scala> val highValueTransRecords = goodTransRecords.filter(_.tranAmount > 1000) 
highValueTransRecords: org.apache.spark.sql.Dataset[Trans] = [accNo: string, tranAmount: double] 
scala> highValueTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10006|   10000.0| 
|SB10010|    7000.0| 
+-------+----------+ 
scala> // The function that identifies the bad amounts 
scala> val badAmountLambda = (trans: Trans) => trans.tranAmount <= 0 
badAmountLambda: Trans => Boolean = <function1> 
scala> // The function that identifies bad accounts 
scala> val badAcNoLambda = (trans: Trans) => trans.accNo.startsWith("SB") == false 
badAcNoLambda: Trans => Boolean = <function1> 
scala> // Apply filter and create another Dataset of bad amount records 
scala> val badAmountRecords = acTransDS.filter(badAmountLambda) 
badAmountRecords: org.apache.spark.sql.Dataset[Trans] = [accNo: string, tranAmount: double] 
scala> badAmountRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Apply filter and create another Dataset of bad account records 
scala> val badAccountRecords = acTransDS.filter(badAcNoLambda) 
badAccountRecords: org.apache.spark.sql.Dataset[Trans] = [accNo: string, tranAmount: double] 
scala> badAccountRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
+-------+----------+ 
scala> // Do the union of two Dataset and create another Dataset 
scala> val badTransRecords  = badAmountRecords.union(badAccountRecords) 
badTransRecords: org.apache.spark.sql.Dataset[Trans] = [accNo: string, tranAmount: double] 
scala> badTransRecords.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10002|     -10.0| 
|CR10001|    7000.0| 
+-------+----------+ 
scala> // Calculate the sum 
scala> val sumAmount = goodTransRecords.map(trans => trans.tranAmount).reduce(_ + _) 
sumAmount: Double = 28486.0 
scala> // Calculate the maximum 
scala> val maxAmount = goodTransRecords.map(trans => trans.tranAmount).reduce((a, b) => if (a > b) a else b) 
maxAmount: Double = 10000.0 
scala> // Calculate the minimum 
scala> val minAmount = goodTransRecords.map(trans => trans.tranAmount).reduce((a, b) => if (a < b) a else b) 
minAmount: Double = 30.0 
scala> // Convert the Dataset to DataFrame 
scala> val acTransDF = acTransDS.toDF() 
acTransDF: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> acTransDF.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Use Spark SQL to find out invalid transaction records 
scala> acTransDF.createOrReplaceTempView("trans") 
scala> val invalidTransactions = spark.sql("SELECT accNo, tranAmount FROM trans WHERE (accNo NOT LIKE 'SB%') OR tranAmount <= 0") 
invalidTransactions: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> invalidTransactions.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+ 
scala> // Interoperability of RDD, DataFrame and Dataset 
scala> // Create RDD 
scala> val acTransRDD = sc.parallelize(acTransList) 
acTransRDD: org.apache.spark.rdd.RDD[Trans] = ParallelCollectionRDD[206] at parallelize at <console>:28 
scala> // Convert RDD to DataFrame 
scala> val acTransRDDtoDF = acTransRDD.toDF() 
acTransRDDtoDF: org.apache.spark.sql.DataFrame = [accNo: string, tranAmount: double] 
scala> // Convert the DataFrame to Dataset with the type checking 
scala> val acTransDFtoDS = acTransRDDtoDF.as[Trans] 
acTransDFtoDS: org.apache.spark.sql.Dataset[Trans] = [accNo: string, tranAmount: double] 
scala> acTransDFtoDS.show() 
+-------+----------+ 
|  accNo|tranAmount| 
+-------+----------+ 
|SB10001|    1000.0| 
|SB10002|    1200.0| 
|SB10003|    8000.0| 
|SB10004|     400.0| 
|SB10005|     300.0| 
|SB10006|   10000.0| 
|SB10007|     500.0| 
|SB10008|      56.0| 
|SB10009|      30.0| 
|SB10010|    7000.0| 
|CR10001|    7000.0| 
|SB10002|     -10.0| 
+-------+----------+

```

很明显，基于 dataset 的编程在许多数据处理用例中具有良好的适用性；同时，它与其他 Spark 内部的数据处理抽象具有高度的互操作性。

### 小贴士

在前面的代码片段中，DataFrame 通过类型指定`acTransRDDToDF.as[Trans]`转换为 Dataset。这种类型的转换在从外部数据源（如 JSON、Avro 或 Parquet 文件）读取数据时确实是必需的。那时就需要强类型检查。通常，结构化数据被读取到 DataFrame 中，然后可以通过以下方式一次性转换为具有强类型安全检查的 DataSet：`spark.read.json("/transaction.json").as[Trans]`

如果检查本章中的 Scala 代码片段，当在 DataFrame 上调用某些方法时，返回的不是 DataFrame 对象，而是一个 `org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]` 类型的对象。这是 DataFrame 和 dataset 之间的重要关系。换句话说，DataFrame 是一个 `org.apache.spark.sql.Row` 类型的 dataset。如果需要，可以使用 `toDF()` 方法显式地将此 `org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]` 类型的对象转换为 DataFrame。

选项过多会让人感到困惑。在这里的 Spark 编程模型中，也存在同样的问题。但相较于许多其他编程范式，它并不那么令人困惑。每当需要以非常高的灵活性处理各种数据，并且需要最低级别的 API 控制如库开发时，基于 RDD 的编程模型是理想的。每当需要以灵活的方式处理结构化数据，并且在整个支持的编程语言中具有优化性能时，基于 DataFrame 的 Spark SQL 编程模型是理想的。

当需要以优化性能要求以及编译时类型安全，但不是非常复杂的 Spark 转换和 Spark 动作使用要求来处理非结构化数据时，基于 dataset 的编程模型是理想的。在数据处理应用程序开发层面，如果选择的编程语言允许，最好使用 dataset 和 DataFrame 以获得更好的性能。

# 理解数据目录

本章的前几节介绍了 DataFrame 和 dataset 的编程模型。这两个编程模型都可以处理结构化数据。结构化数据包含元数据或描述数据结构的描述性数据。Spark SQL 为数据处理应用程序提供了一个名为 Catalog API 的最小化 API，用于查询和应用程序中的元数据。Catalog API 提供了一个包含许多数据库的目录抽象。对于常规的 SparkSession，它将只有一个数据库，即默认数据库。但如果 Spark 与 Hive 一起使用，则整个 Hive 元存储将通过 Catalog API 可用。以下代码片段展示了 Scala 和 Python 中 Catalog API 的使用示例。

从相同的 Scala REPL 提示符继续，尝试以下语句：

```py
scala> // Get the catalog object from the SparkSession object
scala> val catalog = spark.catalog
catalog: org.apache.spark.sql.catalog.Catalog = org.apache.spark.sql.internal.CatalogImpl@14b8a751
scala> // Get the list of databases
scala> val dbList = catalog.listDatabases()
dbList: org.apache.spark.sql.Dataset[org.apache.spark.sql.catalog.Database] = [name: string, description: string ... 1 more field]
scala> // Display the details of the databases
scala> dbList.select("name", "description", "locationUri").show()**+-------+----------------+--------------------+**
**| name| description| locationUri|**
**+-------+----------------+--------------------+**
**|default|default database|file:/Users/RajT/...|**
**+-------+----------------+--------------------+**
scala> // Display the details of the tables in the database
scala> val tableList = catalog.listTables()
tableList: org.apache.spark.sql.Dataset[org.apache.spark.sql.catalog.Table] = [name: string, database: string ... 3 more fields]
scala> tableList.show()**+-----+--------+-----------+---------+-----------+**
 **| name|database|description|tableType|isTemporary|**
**+-----+--------+-----------+---------+-----------+**
**|trans| null| null|TEMPORARY| true|**
**+-----+--------+-----------+---------+-----------+**
scala> // The above list contains the temporary view that was created in the Dataset use case discussed in the previous section
// The views created in the applications can be removed from the database using the Catalog APIscala> catalog.dropTempView("trans")
// List the available tables after dropping the temporary viewscala> val latestTableList = catalog.listTables()
latestTableList: org.apache.spark.sql.Dataset[org.apache.spark.sql.catalog.Table] = [name: string, database: string ... 3 more fields]
scala> latestTableList.show()**+----+--------+-----------+---------+-----------+**
**|name|database|description|tableType|isTemporary|**
**+----+--------+-----------+---------+-----------+**
**+----+--------+-----------+---------+-----------+** 

```

同样，Catalog API 也可以从 Python 代码中使用。由于在 Python 中 dataset 示例不适用，因此表列表将为空。在 Python REPL 提示符下，尝试以下语句：

```py
>>> #Get the catalog object from the SparkSession object
>>> catalog = spark.catalog
>>> #Get the list of databases and their details.
>>> catalog.listDatabases()   [Database(name='default', description='default database', locationUri='file:/Users/RajT/source-code/spark-source/spark-2.0/spark-warehouse')]
// Display the details of the tables in the database
>>> catalog.listTables()
>>> []

```

当编写数据处理应用程序时，Catalog API 非常方便，它可以根据元存储中的内容动态处理数据，尤其是在与 Hive 结合使用时。

# 参考文献

如需更多信息，您可以参考：

+   [`amplab.cs.berkeley.edu/wp-content/uploads/2015/03/SparkSQLSigmod2015.pdf`](https://amplab.cs.berkeley.edu/wp-content/uploads/2015/03/SparkSQLSigmod2015.pdf)

+   [`pandas.pydata.org/`](http://pandas.pydata.org/)

# 摘要

Spark SQL 是 Spark 核心基础设施之上的一个非常有用的库。这个库使得 Spark 编程对那些熟悉命令式编程风格但不太擅长函数式编程的程序员更加包容。除此之外，Spark SQL 是 Spark 数据处理库家族中处理结构化数据的最佳库。基于 Spark SQL 的数据处理应用程序可以使用类似 SQL 的查询或 DataFrame API 的命令式程序风格进行编写。本章还演示了混合 RDD 和 DataFrame、混合类似 SQL 的查询和 DataFrame API 的各种策略。这为应用程序开发者提供了极大的灵活性，让他们可以以最舒适的方式或更符合用例的方式编写数据处理程序，同时不牺牲性能。

数据集 API 是 Spark 中基于数据集的下一代编程模型，提供优化的性能和编译时类型安全。

目录 API 是一个非常实用的工具，可以根据元存储的内容动态处理数据。

R 是数据科学家的语言。在 Spark SQL 支持 R 作为编程语言之前，对于他们来说，主要的分布式数据处理并不容易。现在，使用 R 作为首选语言，他们可以无缝地编写分布式数据处理应用程序，就像他们使用个人机器上的 R 数据框一样。下一章将讨论在 Spark SQL 中使用 R 进行数据处理。
