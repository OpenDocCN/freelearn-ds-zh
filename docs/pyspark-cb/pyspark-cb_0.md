# 前言

Apache Spark 是一个开源框架，用于高效的集群计算，具有强大的数据并行性和容错性接口。本书提供了有效和节省时间的配方，利用 Python 的力量并将其应用于 Spark 生态系统。

您将首先了解 Apache Spark 的架构，并了解如何为 Spark 设置 Python 环境。然后，您将熟悉 PySpark 中可用的模块，并开始轻松使用它们。除此之外，您还将了解如何使用 RDDs 和 DataFrames 抽象数据，并了解 PySpark 的流处理能力。然后，您将继续使用 ML 和 MLlib 来解决与 PySpark 的机器学习能力相关的任何问题，并使用 GraphFrames 解决图处理问题。最后，您将探索如何使用 spark-submit 命令将应用程序部署到云中。

本书结束时，您将能够使用 Apache Spark 的 Python API 解决与构建数据密集型应用程序相关的任何问题。

# 本书的读者对象

如果您是一名 Python 开发人员，并且希望通过实践掌握 Apache Spark 2.x 生态系统的最佳使用方法，那么本书适合您。对 Python 的深入理解（以及对 Spark 的一些熟悉）将帮助您充分利用本书。

# 本书涵盖的内容

第一章，*安装和配置 Spark*，向我们展示了如何安装和配置 Spark，可以作为本地实例、多节点集群或虚拟环境。

第二章，*使用 RDDs 抽象数据*，介绍了如何使用 Apache Spark 的弹性分布式数据集（RDDs）。

第三章，*使用 DataFrames 抽象数据*，探讨了当前的基本数据结构 DataFrames。

第四章，*为建模准备数据*，介绍了如何清理数据并为建模做准备。

第五章，*使用 MLlib 进行机器学习*，介绍了如何使用 PySpark 的 MLlib 模块构建机器学习模型。

第六章，*ML 模块的机器学习*，介绍了 PySpark 当前支持的机器学习模块 ML 模块。

第七章，*使用 PySpark 进行结构化流处理*，介绍了如何在 PySpark 中使用 Apache Spark 结构化流处理。

第八章，*GraphFrames - 使用 PySpark 进行图论*，展示了如何使用 GraphFrames 处理 Apache Spark。

# 为了充分利用本书

您需要以下内容才能顺利完成各章内容：

+   Apache Spark（可从[`spark.apache.org/downloads.html`](http://spark.apache.org/downloads.html)下载）

+   Python

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的帐户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)注册，直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，然后按照屏幕上的说明操作。

下载文件后，请确保使用以下最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/PySpark-Cookbook`](https://github.com/PacktPublishing/PySpark-Cookbook)。如果代码有更新，将在现有的 GitHub 存储库中进行更新。

我们还有其他代码包，来自我们丰富的图书和视频目录，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。去看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图像。您可以在此处下载：[`www.packtpub.com/sites/default/files/downloads/PySparkCookbook_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/PySparkCookbook_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄。例如："接下来，我们调用三个函数：`printHeader`、`checkJava`和`checkPython`。"

代码块设置如下：

```py
if [ "${_check_R_req}" = "true" ]; then
 checkR
fi
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```py
 if [ "$_machine" = "Mac" ]; then
    curl -O $_spark_source
 elif [ "$_machine" = "Linux"]; then
    wget $_spark_source
```

任何命令行输入或输出均按以下格式编写：

```py
tar -xvf sbt-1.0.4.tgz
sudo mv sbt-1.0.4/ /opt/scala/
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以这种方式出现在文本中。例如："转到文件 | 导入应用程序；单击路径选择旁边的按钮。"

警告或重要说明会以这种方式出现。

技巧和窍门会以这种方式出现。

# 章节

在本书中，您会经常看到几个标题（*准备工作*、*如何做...*、*工作原理...*、*还有更多...*和*另请参阅*）。

为了清晰地说明如何完成食谱，使用以下各节：

# 准备工作

本节告诉您食谱中可以期待什么，并描述如何设置食谱所需的任何软件或任何初步设置。

# 如何做...

本节包含了遵循食谱所需的步骤。

# 工作原理...

本节通常包括对前一节发生的事情的详细解释。

# 还有更多...

本节包括有关食谱的其他信息，以使您对食谱更加了解。

# 另请参阅

本节提供了有关食谱的其他有用信息的链接。
