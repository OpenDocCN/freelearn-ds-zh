# 前言

Apache Spark 是一个开源的并行处理框架，已经存在了相当长的时间。Apache Spark 的许多用途之一是在集群计算机上进行数据分析应用程序。

本书将帮助您实施一些实用和经过验证的技术，以改进 Apache Spark 中的编程和管理方面。您不仅将学习如何使用 Spark 和 Python API 来创建高性能的大数据分析，还将发现测试、保护和并行化 Spark 作业的技术。

本书涵盖了 PySpark 的安装和设置、RDD 操作、大数据清理和整理，以及将数据聚合和总结为有用报告。您将学习如何从所有流行的数据托管平台（包括 HDFS、Hive、JSON 和 S3）获取数据，并使用 PySpark 处理大型数据集，获得实际的大数据经验。本书还将帮助您在本地机器上开发原型，然后逐步处理生产环境和大规模的混乱数据。

# 本书的受众

本书适用于开发人员、数据科学家、业务分析师或任何需要可靠地分析大量大规模真实世界数据的人。无论您是负责创建公司的商业智能功能，还是为机器学习模型创建出色的数据平台，或者希望使用代码放大业务影响，本书都适合您。

# 本书涵盖的内容

第一章《安装 Pyspark 并设置开发环境》涵盖了 PySpark 的安装，以及学习 Spark 的核心概念，包括弹性分布式数据集（RDDs）、SparkContext 和 Spark 工具，如 SparkConf 和 SparkShell。

第二章《使用 RDD 将大数据导入 Spark 环境》解释了如何使用 RDD 将大数据导入 Spark 环境，使用各种工具与修改数据进行交互，以便提取有用的见解。

第三章《使用 Spark 笔记本进行大数据清理和整理》介绍了如何在笔记本应用程序中使用 Spark，从而促进 RDD 的有效使用。

第四章《将数据聚合和总结为有用报告》描述了如何使用 map 和 reduce 函数计算平均值，执行更快的平均值计算，并使用键/值对数据点的数据透视表。

第五章《使用 MLlib 进行强大的探索性数据分析》探讨了 Spark 执行回归任务的能力，包括线性回归和 SVM 等模型。

第六章《使用 SparkSQL 为大数据添加结构》解释了如何使用 Spark SQL 模式操作数据框，并使用 Spark DSL 构建结构化数据操作的查询。

第七章《转换和操作》介绍了 Spark 转换以推迟计算，然后考虑应避免的转换。我们还将使用`reduce`和`reduceByKey`方法对数据集进行计算。

第八章《不可变设计》解释了如何使用 DataFrame 操作进行转换，以讨论高度并发环境中的不可变性。

第九章《避免洗牌和减少运营成本》涵盖了洗牌和应该使用的 Spark API 操作。然后我们将测试在 Apache Spark 中引起洗牌的操作，以了解应避免哪些操作。

第十章《以正确格式保存数据》解释了如何以正确格式保存数据，以及如何使用 Spark 的标准 API 将数据保存为纯文本。

第十一章《使用 Spark 键/值 API》，讨论了可用于键/值对的转换。我们将研究键/值对的操作，并查看键/值数据上可用的分区器。

第十二章《测试 Apache Spark 作业》更详细地讨论了在不同版本的 Spark 中测试 Apache Spark 作业。

第十三章，*利用 Spark GraphX API*，介绍了如何利用 Spark GraphX API。我们将对 Edge API 和 Vertex API 进行实验。

# 充分利用本书

本书需要一些 PySpark、Python、Java 和 Scala 的基本编程经验。

# 下载示例代码文件

您可以从您在[www.packt.com](http://www.packt.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  单击“代码下载和勘误”。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Big-Data-Analytics-with-PySpark`](https://github.com/PacktPublishing/Hands-On-Big-Data-Analytics-with-PySpark)。如果代码有更新，将在现有的 GitHub 存储库上进行更新。

我们还有其他代码包，来自我们丰富的书籍和视频目录，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。请查看！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781838644130_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781838644130_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。以下是一个例子：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。”

代码块设置如下：

```py
test("Should use immutable DF API") {
    import spark.sqlContext.implicits._
    //given
    val userData =
        spark.sparkContext.makeRDD(List(
            UserData("a", "1"),
            UserData("b", "2"),
            UserData("d", "200")
        )).toDF()
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
class ImmutableRDD extends FunSuite {
    val spark: SparkContext = SparkSession
        .builder().master("local[2]").getOrCreate().sparkContext

test("RDD should be immutable") {
    //given
    val data = spark.makeRDD(0 to 5)
```

任何命令行输入或输出都以以下方式编写：

```py
total_duration/(normal_data.count())
```

**粗体**：表示一个新术语、一个重要词或屏幕上看到的词。例如，菜单或对话框中的词会以这种方式出现在文本中。以下是一个例子：“从管理面板中选择系统信息。”

警告或重要说明会出现在这样的地方。

提示和技巧会出现在这样的地方。
