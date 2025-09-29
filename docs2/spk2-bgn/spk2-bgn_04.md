# 第四章. 使用 R 语言进行 Spark 编程

R 是一种流行的统计计算编程语言，被许多人使用，并且可以在 **通用公共许可证**（**GNU**）下免费获得。R 语言起源于由 John Chambers 创建的编程语言 S，由 Ross Ihaka 和 Robert Gentleman 开发。许多数据科学家使用 R 来满足他们的计算需求。R 语言具有许多内置的统计函数和许多标量数据类型，并为向量、矩阵、数据框等提供了复合数据结构，用于统计计算。R 语言高度可扩展，因此可以创建外部包。一旦创建了外部包，就必须安装和加载它，以便任何程序可以使用它。这些包的集合在目录下形成了一个 R 库。换句话说，R 语言自带了一套基础包，以及可以安装在其上的附加包，以形成满足所需计算需求所需的库。除了函数外，数据集也可以打包在 R 包中。

本章将涵盖以下主题：

+   SparkR 的需求

+   R 语言基础

+   数据框

+   聚合

+   使用 SparkR 的多数据源连接

# SparkR 的需求

纯 R 语言基础安装无法与 Spark 交互。**SparkR** 包暴露了 R 与 Spark 生态系统通信所需的所有对象和函数。与 Scala、Java 和 Python 相比，R 语言的 Spark 编程有所不同，SparkR 包主要暴露了基于 DataFrame 的 Spark SQL 编程的 R API。目前，R 无法直接操作 Spark 的 RDD。因此，从实际应用的角度来看，R API 对 Spark 的访问仅限于 Spark SQL 抽象。Spark **MLlib** 也可以使用 R 进行编程，因为 Spark MLlib 使用 DataFrame。

SparkR 如何帮助数据科学家更好地进行数据处理？基础 R 安装要求所有要存储（或可访问）的数据都在安装 R 的计算机上。数据处理发生在可访问 R 安装的单一计算机上。此外，如果数据量超过计算机上的主内存，R 将无法进行所需的处理。使用 SparkR 包，可以访问一个全新的节点集群，用于数据存储和数据处理。借助 SparkR 包，可以使用 R 访问 Spark DataFrame 以及 R DataFrame。

了解两种数据框类型的区别非常重要，即 R 数据框和 Spark 数据框。R 数据框是完全局部的，是 R 语言的数结构。Spark 数据框是由 Spark 基础设施管理的结构化数据的并行集合。

R 数据框可以转换为 Spark 数据框，Spark 数据框也可以转换为 R 数据框。

当 Spark DataFrame 转换为 R DataFrame 时，它应该适合计算机可用的内存。这种转换是一个很好的特性，并且有必要这样做。通过将 R DataFrame 转换为 Spark DataFrame，数据可以分布式并行处理。通过将 Spark DataFrame 转换为 R DataFrame，可以使用其他 R 函数执行的大量计算、图表和绘图操作。简而言之，SparkR 包将分布式和并行计算能力引入 R。

在使用 R 进行数据处理时，由于数据量巨大以及需要将其放入计算机的主内存中，数据处理通常在多个批次中完成，并将结果合并以计算最终结果。如果使用 Spark 与 R 处理数据，可以完全避免这种多批次处理。

通常，报告、图表和绘图是在汇总和总结的原始数据上完成的。原始数据的大小可能很大，不一定适合在一个计算机中。在这种情况下，可以使用 Spark 与 R 处理整个原始数据，最后，使用汇总和总结的数据来生成报告、图表或绘图。

由于无法处理大量数据以及使用 R 进行数据分析，很多时候，ETL 工具被用来在原始数据上执行预处理或转换，并且仅在最终阶段使用 R 进行数据分析。由于 Spark 能够大规模处理数据，Spark 与 R 可以替代整个 ETL 管道，并使用 R 执行所需的数据分析。

许多 R 用户使用 **dplyr** R 包来操作 R 中的数据集。这个包提供了与 R DataFrames 一起快速的数据操作能力。就像操作本地 R DataFrame 一样，它还可以访问一些 RDBMS 表中的数据。除了这些原始的数据操作能力之外，它还缺少 Spark 中可用的许多数据处理功能。因此，Spark 与 R 是 dplyr 等包的良好替代品。

SparkR 包是另一个 R 包，但这并没有阻止任何人使用已经使用的任何 R 包。同时，它通过利用 Spark 的巨大数据处理能力，补充了 R 的数据处理能力。 

# R 语言基础

这不是任何形式的 R 编程指南。但是，为了使不熟悉 R 的人能够欣赏本章所涵盖的内容，简要地介绍 R 语言的基本知识是很重要的。这里涵盖了语言特性的非常基本的介绍。

R 随带一些内置数据类型来存储数值、字符和布尔值。有复合数据结构可用，其中最重要的有，即向量、列表、矩阵和数据框。向量是由给定类型的值按顺序排列的集合。列表是元素按顺序排列的集合，这些元素可以是不同类型的。例如，列表可以包含两个向量，其中一个向量包含数值，另一个向量包含布尔值。矩阵是一个二维数据结构，在行和列中存储数值。数据框是一个二维数据结构，包含行和列，其中列可以有不同的数据类型，但单个列不能包含不同的数据类型。

以下是一些使用变量（向量的特例）、数值向量、字符向量、列表、矩阵、数据框以及为数据框分配列名的代码示例。变量名尽可能具有自描述性，以便读者在没有额外解释的情况下理解。以下在常规 R REPL 上运行的代码片段给出了 R 的数据结构概念：

```py
$ r 
R version 3.2.2 (2015-08-14) -- "Fire Safety" 
Copyright (C) 2015 The R Foundation for Statistical Computing 
Platform: x86_64-apple-darwin13.4.0 (64-bit) 

R is free software and comes with ABSOLUTELY NO WARRANTY. 
You are welcome to redistribute it under certain conditions. 
Type 'license()' or 'licence()' for distribution details. 

  Natural language support but running in an English locale 

R is a collaborative project with many contributors. 
Type 'contributors()' for more information and 
'citation()' on how to cite R or R packages in publications. 

Type 'demo()' for some demos, 'help()' for on-line help, or 
'help.start()' for an HTML browser interface to help. 
Type 'q()' to quit R. 

Warning: namespace 'SparkR' is not available and has been replaced 
by .GlobalEnv when processing object 'goodTransRecords' 
[Previously saved workspace restored] 
> 
> x <- 5 
> x 
[1] 5 
> aNumericVector <- c(10,10.5,31.2,100) 
> aNumericVector 
[1]  10.0  10.5  31.2 100.0 
> aCharVector <- c("apple", "orange", "mango") 
> aCharVector 
[1] "apple"  "orange" "mango"  
> aBooleanVector <- c(TRUE, FALSE, TRUE, FALSE, FALSE) 
> aBooleanVector 
[1]  TRUE FALSE  TRUE FALSE FALSE 
> aList <- list(aNumericVector, aCharVector) 
> aList 
[[1]] 
[1]  10.0  10.5  31.2 100.0 
[[2]] 
[1] "apple"  "orange" "mango" 
> aMatrix <- matrix(c(100, 210, 76, 65, 34, 45),nrow=3,ncol=2,byrow = TRUE) 
> aMatrix 
     [,1] [,2] 
[1,]  100  210 
[2,]   76   65 
[3,]   34   45 
> bMatrix <- matrix(c(100, 210, 76, 65, 34, 45),nrow=3,ncol=2,byrow = FALSE) 
> bMatrix 
     [,1] [,2] 
[1,]  100   65 
[2,]  210   34 
[3,]   76   45 
> ageVector <- c(21, 35, 52)  
> nameVector <- c("Thomas", "Mathew", "John")  
> marriedVector <- c(FALSE, TRUE, TRUE)  
> aDataFrame <- data.frame(ageVector, nameVector, marriedVector)  
> aDataFrame 
  ageVector nameVector marriedVector 
1        21     Thomas         FALSE 
2        35     Mathew          TRUE 
3        52       John          TRUE 
> colnames(aDataFrame) <- c("Age","Name", "Married") 
> aDataFrame 
  Age   Name Married 
1  21 Thomas   FALSE 
2  35 Mathew    TRUE 
3  52   John    TRUE 

```

这里讨论的主要话题将围绕数据框展开。这里展示了与数据框常用的一些函数。所有这些命令都应在常规 R REPL 中执行，作为执行前一个代码片段的会话的延续：

```py
> # Returns the first part of the data frame and return two rows 
> head(aDataFrame,2) 
  Age   Name Married 
1  21 Thomas   FALSE 
2  35 Mathew    TRUE 

> # Returns the last part of the data frame and return two rows 
> tail(aDataFrame,2) 
  Age   Name Married  
2  35 Mathew    TRUE 
3  52   John    TRUE 
> # Number of rows in a data frame 
> nrow(aDataFrame) 
[1] 3 
> # Number of columns in a data frame 
> ncol(aDataFrame) 
[1] 3 
> # Returns the first column of the data frame. The return value is a data frame 
> aDataFrame[1] 
  Age 
1  21 
2  35 
3  52 
> # Returns the second column of the data frame. The return value is a data frame 
> aDataFrame[2] 
    Name 
1 Thomas 
2 Mathew 
3   John 
> # Returns the named columns of the data frame. The return value is a data frame 
> aDataFrame[c("Age", "Name")] 
  Age   Name 
1  21 Thomas 
2  35 Mathew 
3  52   John 
> # Returns the contents of the second column of the data frame as a vector.  
> aDataFrame[[2]] 
[1] Thomas Mathew John   
Levels: John Mathew Thomas 
> # Returns the slice of the data frame by a row 
> aDataFrame[2,] 
  Age   Name Married 
2  35 Mathew    TRUE 
> # Returns the slice of the data frame by multiple rows 
> aDataFrame[c(1,2),] 
  Age   Name Married 
1  21 Thomas   FALSE 
2  35 Mathew    TRUE 

```

# R 和 Spark 中的 DataFrames

当使用 R 与 Spark 一起工作时，很容易对 DataFrame 数据结构感到困惑。如前所述，它在 R 和 Spark SQL 中都存在。以下代码片段处理将 R DataFrame 转换为 Spark DataFrame 以及相反操作。当使用 R 编程 Spark 时，这将是一个非常常见的操作。以下代码片段应在 Spark 的 R REPL 中执行。从现在开始，所有关于 R REPL 的引用都是指 Spark 的 R REPL：

```py
$ cd $SPARK_HOME 
$ ./bin/sparkR 

R version 3.2.2 (2015-08-14) -- "Fire Safety" 
Copyright (C) 2015 The R Foundation for Statistical Computing 
Platform: x86_64-apple-darwin13.4.0 (64-bit) 

R is free software and comes with ABSOLUTELY NO WARRANTY. 
You are welcome to redistribute it under certain conditions. 
Type 'license()' or 'licence()' for distribution details. 

  Natural language support but running in an English locale 

R is a collaborative project with many contributors. 
Type 'contributors()' for more information and 
'citation()' on how to cite R or R packages in publications. 

Type 'demo()' for some demos, 'help()' for on-line help, or 
'help.start()' for an HTML browser interface to help. 
Type 'q()' to quit R. 

[Previously saved workspace restored] 

Launching java with spark-submit command /Users/RajT/source-code/spark-source/spark-2.0/bin/spark-submit   "sparkr-shell" /var/folders/nf/trtmyt9534z03kq8p8zgbnxh0000gn/T//RtmpmuRsTC/backend_port2d121acef4  
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties 
Setting default log level to "WARN". 
To adjust logging level use sc.setLogLevel(newLevel). 
16/07/16 21:08:50 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable 

 Welcome to 
    ____              __  
   / __/__  ___ _____/ /__  
  _\ \/ _ \/ _ `/ __/  '_/  
 /___/ .__/\_,_/_/ /_/\_\   version  2.0.1-SNAPSHOT  
    /_/  

 Spark context is available as sc, SQL context is available as sqlContext 
During startup - Warning messages: 
1: 'SparkR::sparkR.init' is deprecated. 
Use 'sparkR.session' instead. 
See help("Deprecated")  
2: 'SparkR::sparkRSQL.init' is deprecated. 
Use 'sparkR.session' instead. 
See help("Deprecated")  
> 
> # faithful is a data set and the data frame that comes with base R 
> # Obviously it is an R DataFrame 
> head(faithful) 
  eruptions waiting 
1     3.600      79 
2     1.800      54 
3     3.333      74 
4     2.283      62 
5     4.533      85 
6     2.883      55 
> tail(faithful) 
    eruptions waiting 
267     4.750      75 
268     4.117      81 
269     2.150      46 
270     4.417      90 
271     1.817      46 
272     4.467      74 
> # Convert R DataFrame to Spark DataFrame  
> sparkFaithful <- createDataFrame(faithful) 
> head(sparkFaithful) 
  eruptions waiting 
1     3.600      79 
2     1.800      54 
3     3.333      74 
4     2.283      62 
5     4.533      85 
6     2.883      55 
> showDF(sparkFaithful) 
+---------+-------+ 
|eruptions|waiting| 
+---------+-------+ 
|      3.6|   79.0| 
|      1.8|   54.0| 
|    3.333|   74.0| 
|    2.283|   62.0| 
|    4.533|   85.0| 
|    2.883|   55.0| 
|      4.7|   88.0| 
|      3.6|   85.0| 
|     1.95|   51.0| 
|     4.35|   85.0| 
|    1.833|   54.0| 
|    3.917|   84.0| 
|      4.2|   78.0| 
|     1.75|   47.0| 
|      4.7|   83.0| 
|    2.167|   52.0| 
|     1.75|   62.0| 
|      4.8|   84.0| 
|      1.6|   52.0| 
|     4.25|   79.0| 
+---------+-------+ 
only showing top 20 rows 
> # Try calling a SparkR function showDF() on an R DataFrame. The following error message will be shown 
> showDF(faithful) 
Error in (function (classes, fdef, mtable)  :  
  unable to find an inherited method for function 'showDF' for signature '"data.frame"' 
> # Convert the Spark DataFrame to an R DataFrame 
> rFaithful <- collect(sparkFaithful) 
> head(rFaithful) 
  eruptions waiting 
1     3.600      79 
2     1.800      54 
3     3.333      74 
4     2.283      62 
5     4.533      85 
6     2.883      55 

```

在支持的功能方面，R DataFrame 和 Spark DataFrame 之间没有完全的兼容性和互操作性。

### 小贴士

作为一种良好的实践，在 R 程序中最好使用约定的命名约定来命名 R DataFrame 和 Spark DataFrame，以便在两种不同类型之间有所区分。并非所有在 R DataFrame 上支持的功能都在 Spark DataFrame 上支持，反之亦然。始终参考 Spark 的正确版本 R API。

那些大量使用图表和绘图的人在使用 R DataFrame 与 Spark DataFrame 结合时必须格外小心。R 的图表和绘图仅与 R DataFrame 一起工作。如果需要使用 Spark DataFrame 中处理的数据生成图表或绘图，则必须将其转换为 R DataFrame 才能进行图表和绘图。以下代码片段将给出一个想法。我们将再次使用 faithful 数据集，在 Spark 的 R REPL 中进行阐明：

```py
head(faithful) 
  eruptions waiting 
1     3.600      79 
2     1.800      54 
3     3.333      74 
4     2.283      62 
5     4.533      85 
6     2.883      55 
> # Convert the faithful R DataFrame to Spark DataFrame   
> sparkFaithful <- createDataFrame(faithful) 
> # The Spark DataFrame sparkFaithful NOT producing a histogram 
> hist(sparkFaithful$eruptions,main="Distribution of Eruptions",xlab="Eruptions") 
Error in hist.default(sparkFaithful$eruptions, main = "Distribution of Eruptions",  :  
  'x' must be numeric 
> # The R DataFrame faithful producing a histogram 
> hist(faithful$eruptions,main="Distribution of Eruptions",xlab="Eruptions")

```

此图仅用于演示 Spark DataFrame 不能用于图表和绘图，而必须使用 R DataFrame 进行相同的操作：

![R 和 Spark 中的 DataFrame](img/image_04_002.jpg)

图 1

当与 Spark DataFrame 一起使用时，由于数据类型的不兼容，图表和绘图库出现了错误。

### 小贴士

需要牢记的最重要的一点是，R DataFrame 是一个内存驻留的数据结构，而 Spark DataFrame 是跨节点集群分布的数据集的并行集合。因此，所有使用 R DataFrame 的函数不必与 Spark DataFrame 一起工作，反之亦然。

让我们再次回顾一下更大的图景，如图 2 所示，以设置上下文并了解在这里讨论的内容，然后再进入并处理使用案例。在前一章中，使用 Scala 和 Python 编程语言介绍了相同主题。在这一章中，将使用与 Spark SQL 编程中使用的相同的一组使用案例，但使用 R 来实现：

![R 和 Spark 中的 DataFrame](img/image_04_004.jpg)

图 2

这里将要讨论的使用案例将展示在 R 中混合 SQL 查询与 Spark 程序的能力。将选择多个数据源，使用 DataFrame 从这些源读取数据，并演示统一的数据访问。

# 使用 R 进行 Spark DataFrame 编程

用于阐明使用 DataFrame 进行 Spark SQL 编程的使用案例如下：

+   交易记录是逗号分隔值。

+   从列表中过滤出仅包含良好交易记录。账户号码应以`SB`开头，交易金额应大于零。

+   查找所有交易金额大于 1000 的高价值交易记录。

+   查找所有账户号码错误的交易记录。

+   查找所有交易金额小于或等于零的交易记录。

+   查找所有不良交易记录的合并列表。

+   查找所有交易金额的总和。

+   查找所有交易金额的最大值。

+   查找所有交易金额的最小值。

+   查找所有账户号码良好的记录。

这正是上一章中使用的一组用例，但在这里，编程模型完全不同。在这里，编程是在 R 中完成的。使用这组用例，展示了两种类型的编程模型。一种是使用 SQL 查询，另一种是使用 DataFrame API。

### 小贴士

运行以下代码片段所需的数据文件与 R 代码所在的同一目录中可用。

在以下代码片段中，数据是从文件系统中读取的。由于所有这些代码片段都是在 Spark 的 R REPL 中执行的，因此所有数据文件都必须保存在`$SPARK_HOME`目录中。

## 使用 SQL 进行编程

在 R REPL 提示符下，尝试以下语句：

```py
> # TODO - Change the data directory location to the right location in the system in which this program is being run 
> DATA_DIR <- "/Users/RajT/Documents/CodeAndData/R/" 
> # Read data from a JSON file to create DataFrame 
>  
> acTransDF <- read.json(paste(DATA_DIR, "TransList1.json", sep = "")) 
> # Print the structure of the DataFrame 
> print(acTransDF) 
SparkDataFrame[AccNo:string, TranAmount:bigint] 
> # Show sample records from the DataFrame 
> showDF(acTransDF) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10001|      1000| 
|SB10002|      1200| 
|SB10003|      8000| 
|SB10004|       400| 
|SB10005|       300| 
|SB10006|     10000| 
|SB10007|       500| 
|SB10008|        56| 
|SB10009|        30| 
|SB10010|      7000| 
|CR10001|      7000| 
|SB10002|       -10| 
+-------+----------+ 
> # Register temporary view definition in the DataFrame for SQL queries 
> createOrReplaceTempView(acTransDF, "trans") 
> # DataFrame containing good transaction records using SQL 
> goodTransRecords <- sql("SELECT AccNo, TranAmount FROM trans WHERE AccNo like 'SB%' AND TranAmount > 0") 
> # Register temporary table definition in the DataFrame for SQL queries 

> createOrReplaceTempView(goodTransRecords, "goodtrans") 
> # Show sample records from the DataFrame 
> showDF(goodTransRecords) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10001|      1000| 
|SB10002|      1200| 
|SB10003|      8000| 
|SB10004|       400| 
|SB10005|       300| 
|SB10006|     10000| 
|SB10007|       500| 
|SB10008|        56| 
|SB10009|        30| 
|SB10010|      7000| 
+-------+----------+ 
> # DataFrame containing high value transaction records using SQL 
> highValueTransRecords <- sql("SELECT AccNo, TranAmount FROM goodtrans WHERE TranAmount > 1000") 
> # Show sample records from the DataFrame 
> showDF(highValueTransRecords) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10002|      1200| 
|SB10003|      8000| 
|SB10006|     10000| 
|SB10010|      7000| 
+-------+----------+ 
> # DataFrame containing bad account records using SQL 
> badAccountRecords <- sql("SELECT AccNo, TranAmount FROM trans WHERE AccNo NOT like 'SB%'") 
> # Show sample records from the DataFrame 
> showDF(badAccountRecords) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|CR10001|      7000| 
+-------+----------+ 
> # DataFrame containing bad amount records using SQL 
> badAmountRecords <- sql("SELECT AccNo, TranAmount FROM trans WHERE TranAmount < 0") 
> # Show sample records from the DataFrame 
> showDF(badAmountRecords) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10002|       -10| 
+-------+----------+ 
> # Create a DataFrame by taking the union of two DataFrames 
> badTransRecords <- union(badAccountRecords, badAmountRecords) 
> # Show sample records from the DataFrame 
> showDF(badTransRecords) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|CR10001|      7000| 
|SB10002|       -10| 
+-------+----------+ 
> # DataFrame containing sum amount using SQL 
> sumAmount <- sql("SELECT sum(TranAmount) as sum FROM goodtrans") 
> # Show sample records from the DataFrame 
> showDF(sumAmount) 
+-----+ 
|  sum| 
+-----+ 
|28486| 
+-----+ 
> # DataFrame containing maximum amount using SQL 
> maxAmount <- sql("SELECT max(TranAmount) as max FROM goodtrans") 
> # Show sample records from the DataFrame 
> showDF(maxAmount) 
+-----+ 
|  max| 
+-----+ 
|10000| 
+-----+ 
> # DataFrame containing minimum amount using SQL 
> minAmount <- sql("SELECT min(TranAmount)as min FROM goodtrans") 
> # Show sample records from the DataFrame 
> showDF(minAmount) 
+---+ 
|min| 
+---+ 
| 30| 
+---+ 
> # DataFrame containing good account number records using SQL 
> goodAccNos <- sql("SELECT DISTINCT AccNo FROM trans WHERE AccNo like 'SB%' ORDER BY AccNo") 
> # Show sample records from the DataFrame 
> showDF(goodAccNos) 
+-------+ 
|  AccNo| 
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

```

零售银行交易记录包含账户号码和交易金额，使用 SparkSQL 进行处理以获得用例的预期结果。以下是前面脚本所做操作的摘要：

+   与 Spark 支持的其他编程语言不同，R 没有 RDD 编程能力。因此，不是从集合中构建 RDD，而是从包含交易记录的 JSON 文件中读取数据。

+   从 JSON 文件创建 Spark DataFrame。

+   使用 DataFrame 注册一个带有名称的表。这个注册的表名可以在 SQL 语句中使用。

+   然后，所有其他活动都是通过 SparkR 包中的 SQL 函数发出 SQL 语句。

+   所有这些 SQL 语句的结果都存储为 Spark DataFrame，并使用 showDF 函数将值提取到调用 R 程序中。

+   聚合值计算也是通过 SQL 语句完成的。

+   DataFrame 的内容使用 SparkR 的`showDF`函数以表格格式显示。

+   使用打印函数可以显示 DataFrame 的结构视图。这类似于数据库表的 describe 命令。

在前面的 R 代码中，编程风格与 Scala 代码不同，因为它是一个 R 程序。使用 SparkR 库，正在使用 Spark 功能。但函数和其他抽象并没有真正不同的风格。

### 注意

在本章中，将会有使用 DataFrame 的实例。很容易混淆哪个是 R DataFrame，哪个是 Spark DataFrame。因此，特别注意通过限定 DataFrame 来具体说明，例如 R DataFrame 和 Spark DataFrame。

## 使用 R DataFrame API 进行编程

在本节中，代码片段将在相同的 R REPL 中运行。与前面的代码片段一样，最初给出了一些 DataFrame 特定的基本命令。这些命令被经常使用，用于查看内容并对 DataFrame 及其内容进行一些基本测试。这些是在数据分析的探索阶段经常使用的命令，以获得更多对底层数据结构和内容的洞察。

在 R 的 REPL 提示符下，尝试以下语句：

```py
> # Read data from a JSON file to create DataFrame 
> acTransDF <- read.json(paste(DATA_DIR, "TransList1.json", sep = "")) 
> print(acTransDF) 
SparkDataFrame[AccNo:string, TranAmount:bigint] 
> # Show sample records from the DataFrame 
> showDF(acTransDF) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10001|      1000| 
|SB10002|      1200| 
|SB10003|      8000| 
|SB10004|       400| 
|SB10005|       300| 
|SB10006|     10000| 
|SB10007|       500| 
|SB10008|        56| 
|SB10009|        30| 
|SB10010|      7000| 
|CR10001|      7000| 
|SB10002|       -10| 
+-------+----------+ 
> # DataFrame containing good transaction records using API 
> goodTransRecordsFromAPI <- filter(acTransDF, "AccNo like 'SB%' AND TranAmount > 0") 
> # Show sample records from the DataFrame 
> showDF(goodTransRecordsFromAPI) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10001|      1000| 
|SB10002|      1200| 
|SB10003|      8000| 
|SB10004|       400| 
|SB10005|       300| 
|SB10006|     10000| 
|SB10007|       500| 
|SB10008|        56| 
|SB10009|        30| 
|SB10010|      7000| 
+-------+----------+ 
> # DataFrame containing high value transaction records using API 
> highValueTransRecordsFromAPI = filter(goodTransRecordsFromAPI, "TranAmount > 1000") 
> # Show sample records from the DataFrame 
> showDF(highValueTransRecordsFromAPI) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10002|      1200| 
|SB10003|      8000| 
|SB10006|     10000| 
|SB10010|      7000| 
+-------+----------+ 
> # DataFrame containing bad account records using API 
> badAccountRecordsFromAPI <- filter(acTransDF, "AccNo NOT like 'SB%'") 
> # Show sample records from the DataFrame 
> showDF(badAccountRecordsFromAPI) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|CR10001|      7000| 
+-------+----------+ 
> # DataFrame containing bad amount records using API 
> badAmountRecordsFromAPI <- filter(acTransDF, "TranAmount < 0") 
> # Show sample records from the DataFrame 
> showDF(badAmountRecordsFromAPI) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10002|       -10| 
+-------+----------+ 
> # Create a DataFrame by taking the union of two DataFrames 
> badTransRecordsFromAPI <- union(badAccountRecordsFromAPI, badAmountRecordsFromAPI) 
> # Show sample records from the DataFrame 
> showDF(badTransRecordsFromAPI) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|CR10001|      7000| 
|SB10002|       -10| 
+-------+----------+ 
> # DataFrame containing sum amount using API 
> sumAmountFromAPI <- agg(goodTransRecordsFromAPI, sumAmount = sum(goodTransRecordsFromAPI$TranAmount)) 
> # Show sample records from the DataFrame 
> showDF(sumAmountFromAPI) 
+---------+ 
|sumAmount| 
+---------+ 
|    28486| 
+---------+ 
> # DataFrame containing maximum amount using API 
> maxAmountFromAPI <- agg(goodTransRecordsFromAPI, maxAmount = max(goodTransRecordsFromAPI$TranAmount)) 
> # Show sample records from the DataFrame 
> showDF(maxAmountFromAPI) 
+---------+ 
|maxAmount| 
+---------+ 
|    10000| 
+---------+ 
> # DataFrame containing minimum amount using API 
> minAmountFromAPI <- agg(goodTransRecordsFromAPI, minAmount = min(goodTransRecordsFromAPI$TranAmount))  
> # Show sample records from the DataFrame 
> showDF(minAmountFromAPI) 
+---------+ 
|minAmount| 
+---------+ 
|       30| 
+---------+ 
> # DataFrame containing good account number records using API 
> filteredTransRecordsFromAPI <- filter(goodTransRecordsFromAPI, "AccNo like 'SB%'") 
> accNosFromAPI <- select(filteredTransRecordsFromAPI, "AccNo") 
> distinctAccNoFromAPI <- distinct(accNosFromAPI) 
> sortedAccNoFromAPI <- arrange(distinctAccNoFromAPI, "AccNo") 
> # Show sample records from the DataFrame 
> showDF(sortedAccNoFromAPI) 
+-------+ 
|  AccNo| 
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
> # Persist the DataFrame into a Parquet file  
> write.parquet(acTransDF, "r.trans.parquet") 
> # Read the data from the Parquet file 
> acTransDFFromFile <- read.parquet("r.trans.parquet")  
> # Show sample records from the DataFrame 
> showDF(acTransDFFromFile) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10007|       500| 
|SB10008|        56| 
|SB10009|        30| 
|SB10010|      7000| 
|CR10001|      7000| 
|SB10002|       -10| 
|SB10001|      1000| 
|SB10002|      1200| 
|SB10003|      8000| 
|SB10004|       400| 
|SB10005|       300| 
|SB10006|     10000| 
+-------+----------+ 

```

下面是从 DataFrame API 角度对前面脚本所做操作的总结：

+   本节使用的是包含上一节中使用的数据集的超集的 DataFrame。

+   接下来演示记录的过滤。这里，需要注意的最重要方面是过滤谓词必须与 SQL 语句中的谓词完全相同。过滤器不能链式使用。

+   接下来计算聚合方法。

+   本集合中的最后几个语句正在进行选择、过滤、选择不同的记录以及排序操作。

+   最后，事务记录以 Parquet 格式持久化，从 Parquet 存储中读取，并创建了一个 Spark DataFrame。关于持久化格式的更多细节已在上一章中介绍，概念保持不变。只是 DataFrame API 的语法有所不同。

+   在这个代码片段中，Parquet 格式数据存储在当前目录中，从该目录调用相应的 REPL。当它作为一个 Spark 程序运行时，目录再次将是 Spark 提交调用的当前目录。

最后几个语句是关于将 DataFrame 内容持久化到媒体中的。如果与上一章中基于 Scala 和 Python 的持久化机制进行比较，这里也是以类似的方式进行。

# 理解 Spark R 中的聚合

在 SQL 中，数据的聚合非常灵活。在 Spark SQL 中也是如此。在这里，Spark SQL 可以在分布式数据源上执行与在单台机器上的单个数据源上运行 SQL 语句相同的事情。在介绍基于 RDD 的编程的章节中，讨论了一个 MapReduce 用例来进行数据聚合，这里同样使用它来展示 Spark SQL 的聚合能力。在本节中，使用 SQL 查询方式以及 DataFrame API 方式来处理用例。

下面给出用于阐明 MapReduce 类型数据处理的使用案例：

+   零售银行交易记录包含以逗号分隔的账户号码和交易金额字符串

+   找到所有交易的账户级别摘要以获取账户余额

在 R 的 REPL 提示符下，尝试以下语句：

```py
> # Read data from a JSON file to create DataFrame 
> acTransDFForAgg <- read.json(paste(DATA_DIR, "TransList2.json", sep = "")) 
> # Register temporary view definition in the DataFrame for SQL queries 
> createOrReplaceTempView(acTransDFForAgg, "transnew") 
> # Show sample records from the DataFrame 
> showDF(acTransDFForAgg) 
+-------+----------+ 
|  AccNo|TranAmount| 
+-------+----------+ 
|SB10001|      1000| 
|SB10002|      1200| 
|SB10001|      8000| 
|SB10002|       400| 
|SB10003|       300| 
|SB10001|     10000| 
|SB10004|       500| 
|SB10005|        56| 
|SB10003|        30| 
|SB10002|      7000| 
|SB10001|      -100| 
|SB10002|       -10| 
+-------+----------+ 
> # DataFrame containing account summary records using SQL 
> acSummary <- sql("SELECT AccNo, sum(TranAmount) as TransTotal FROM transnew GROUP BY AccNo") 
> # Show sample records from the DataFrame 
> showDF(acSummary) 
+-------+----------+ 
|  AccNo|TransTotal| 
+-------+----------+ 
|SB10001|     18900| 
|SB10002|      8590| 
|SB10003|       330| 
|SB10004|       500| 
|SB10005|        56| 
+-------+----------+ 
> # DataFrame containing account summary records using API 
> acSummaryFromAPI <- agg(groupBy(acTransDFForAgg, "AccNo"), TranAmount="sum") 
> # Show sample records from the DataFrame 
> showDF(acSummaryFromAPI) 
+-------+---------------+ 
|  AccNo|sum(TranAmount)| 
+-------+---------------+ 
|SB10001|          18900| 
|SB10002|           8590| 
|SB10003|            330| 
|SB10004|            500| 
|SB10005|             56| 
+-------+---------------+ 

```

在 R DataFrame API 中，与 Scala 或 Python 的对应版本相比，存在一些语法差异，主要是因为这是一个纯 API 编程模型。

# 理解 SparkR 中的多数据源连接

在上一章中，已经讨论了基于键的多个 DataFrame 的连接。在本节中，使用 Spark SQL 的 R API 实现了相同的用例。用于阐明使用键连接多个数据集的用例将在以下章节中给出。

第一个数据集包含一个零售银行主记录摘要，包括账户号码、名和姓。第二个数据集包含零售银行账户余额，包括账户号码和余额金额。这两个数据集的关键是账户号码。将这两个数据集连接起来，创建一个包含账户号码、名、姓和余额金额的单一数据集。从这份报告中，挑选出余额金额最高的前三个账户。

Spark DataFrame 是从持久化的 JSON 文件创建的。除了 JSON 文件，还可以是任何支持的数据文件。然后它们从磁盘读取以形成 DataFrame，并将它们连接在一起。

在 R 交互式命令行提示符下，尝试以下语句：

```py
> # Read data from JSON file 
> acMasterDF <- read.json(paste(DATA_DIR, "MasterList.json", sep = "")) 
> # Show sample records from the DataFrame 
> showDF(acMasterDF) 
+-------+---------+--------+ 
|  AccNo|FirstName|LastName| 
+-------+---------+--------+ 
|SB10001|    Roger| Federer| 
|SB10002|     Pete| Sampras| 
|SB10003|   Rafael|   Nadal| 
|SB10004|    Boris|  Becker| 
|SB10005|     Ivan|   Lendl| 
+-------+---------+--------+ 
> # Register temporary view definition in the DataFrame for SQL queries 
> createOrReplaceTempView(acMasterDF, "master")  
> acBalDF <- read.json(paste(DATA_DIR, "BalList.json", sep = "")) 
> # Show sample records from the DataFrame 
> showDF(acBalDF) 
+-------+---------+ 
|  AccNo|BalAmount| 
+-------+---------+ 
|SB10001|    50000| 
|SB10002|    12000| 
|SB10003|     3000| 
|SB10004|     8500| 
|SB10005|     5000| 
+-------+---------+ 

> # Register temporary view definition in the DataFrame for SQL queries 
> createOrReplaceTempView(acBalDF, "balance") 
> # DataFrame containing account detail records using SQL by joining multiple DataFrame contents 
> acDetail <- sql("SELECT master.AccNo, FirstName, LastName, BalAmount FROM master, balance WHERE master.AccNo = balance.AccNo ORDER BY BalAmount DESC") 
> # Show sample records from the DataFrame 
> showDF(acDetail) 
+-------+---------+--------+---------+ 
|  AccNo|FirstName|LastName|BalAmount| 
+-------+---------+--------+---------+ 
|SB10001|    Roger| Federer|    50000| 
|SB10002|     Pete| Sampras|    12000| 
|SB10004|    Boris|  Becker|     8500| 
|SB10005|     Ivan|   Lendl|     5000| 
|SB10003|   Rafael|   Nadal|     3000| 
+-------+---------+--------+---------+ 

> # Persist data in the DataFrame into Parquet file 
> write.parquet(acDetail, "r.acdetails.parquet") 
> # Read data into a DataFrame by reading the contents from a Parquet file 

> acDetailFromFile <- read.parquet("r.acdetails.parquet") 
> # Show sample records from the DataFrame 
> showDF(acDetailFromFile) 
+-------+---------+--------+---------+ 
|  AccNo|FirstName|LastName|BalAmount| 
+-------+---------+--------+---------+ 
|SB10002|     Pete| Sampras|    12000| 
|SB10003|   Rafael|   Nadal|     3000| 
|SB10005|     Ivan|   Lendl|     5000| 
|SB10001|    Roger| Federer|    50000| 
|SB10004|    Boris|  Becker|     8500| 
+-------+---------+--------+---------+ 

```

从相同的 R 交互式命令行会话继续，以下代码行通过 DataFrame API 得到相同的结果：

```py
> # Change the column names 
> acBalDFWithDiffColName <- selectExpr(acBalDF, "AccNo as AccNoBal", "BalAmount") 
> # Show sample records from the DataFrame 
> showDF(acBalDFWithDiffColName) 
+--------+---------+ 
|AccNoBal|BalAmount| 
+--------+---------+ 
| SB10001|    50000| 
| SB10002|    12000| 
| SB10003|     3000| 
| SB10004|     8500| 
| SB10005|     5000| 
+--------+---------+ 
> # DataFrame containing account detail records using API by joining multiple DataFrame contents 
> acDetailFromAPI <- join(acMasterDF, acBalDFWithDiffColName, acMasterDF$AccNo == acBalDFWithDiffColName$AccNoBal) 
> # Show sample records from the DataFrame 
> showDF(acDetailFromAPI) 
+-------+---------+--------+--------+---------+ 
|  AccNo|FirstName|LastName|AccNoBal|BalAmount| 
+-------+---------+--------+--------+---------+ 
|SB10001|    Roger| Federer| SB10001|    50000| 
|SB10002|     Pete| Sampras| SB10002|    12000| 
|SB10003|   Rafael|   Nadal| SB10003|     3000| 
|SB10004|    Boris|  Becker| SB10004|     8500| 
|SB10005|     Ivan|   Lendl| SB10005|     5000| 
+-------+---------+--------+--------+---------+ 
> # DataFrame containing account detail records using SQL by selecting specific fields 
> acDetailFromAPIRequiredFields <- select(acDetailFromAPI, "AccNo", "FirstName", "LastName", "BalAmount") 
> # Show sample records from the DataFrame 
> showDF(acDetailFromAPIRequiredFields) 
+-------+---------+--------+---------+ 
|  AccNo|FirstName|LastName|BalAmount| 
+-------+---------+--------+---------+ 
|SB10001|    Roger| Federer|    50000| 
|SB10002|     Pete| Sampras|    12000| 
|SB10003|   Rafael|   Nadal|     3000| 
|SB10004|    Boris|  Becker|     8500| 
|SB10005|     Ivan|   Lendl|     5000| 
+-------+---------+--------+---------+ 

```

在代码的前一部分中选择的连接类型是内连接。而不是这样，可以使用任何其他类型的连接，无论是通过 SQL 查询方式还是通过 DataFrame API 方式。在使用 DataFrame API 进行连接之前有一个注意事项是，两个 Spark DataFrame 的列名必须不同，以避免在结果 Spark DataFrame 中产生歧义。在这个特定的用例中，可以看到 DataFrame API 处理起来有点困难，而 SQL 查询方式看起来非常直接。

在前面的章节中，已经介绍了 Spark SQL 的 R API。一般来说，如果可能的话，最好尽可能多地使用 SQL 查询方式编写代码。DataFrame API 正在变得更好，但它不如 Scala 或 Python 等其他语言灵活。

与本书中的其他章节不同，这是一个独立的章节，旨在向 R 程序员介绍 Spark。本章中讨论的所有用例都是在 Spark 的 R 交互式命令行中运行的。但在现实世界的应用中，这种方法并不理想。R 命令必须组织在脚本文件中，并提交到 Spark 集群以运行。最简单的方法是使用已经存在的`$SPARK_HOME/bin/spark-submit <R 脚本文件路径>`脚本，其中完全限定的 R 文件名是根据命令执行的当前目录给出的。

# 参考文献

更多信息请参阅：[`spark.apache.org/docs/latest/api/R/index.html`](https://spark.apache.org/docs/latest/api/R/index.html)

# 摘要

本章介绍了 R 语言的快速浏览，随后特别提到了需要区分理解 R DataFrame 与 Spark DataFrame 之间的差异。接着，使用与之前章节相同的用例，介绍了基于 R 的基本 Spark 编程。本章涵盖了 Spark 的 R API，并使用 SQL 查询方式和 DataFrame API 方式实现了用例。这有助于数据科学家理解 Spark 的强大功能，并使用 SparkR 包（Spark 附带）将其应用于 R 应用程序中。这开启了使用 Spark 与 R 处理结构化数据的大数据处理之门。

在各种语言中基于 Spark 的数据处理主题已经被讨论，现在是时候专注于一些带有图表和绘图的数据分析了。Python 附带了许多图表和绘图库，可以生成高质量的图片。下一章将讨论使用 Spark 处理的数据进行图表和绘图。
