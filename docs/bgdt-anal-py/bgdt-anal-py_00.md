# 第一章：*前言*

## 关于

本节简要介绍了作者、这本书的内容、你入门所需的技术技能，以及完成活动和练习所需的硬件和软件要求。

### 关于本书

实时处理大数据具有挑战性，因为它涉及到可扩展性、信息不一致性和容错性等问题。《Python 大数据分析》教你如何使用工具来应对数据的爆炸性增长。在本书中，你将学习如何将数据聚合成有用的维度以供后续分析，提取统计量，并将数据集转换成特征以供其他系统使用。

本书以介绍如何使用 pandas 在 Python 中进行数据处理开始。然后，你将熟悉统计分析和绘图技术。通过多种动手实践，你将学会如何使用 Dask 分析分布在多台计算机上的数据。随着进度的推进，你将学习如何在数据无法完全放入内存时，如何聚合数据以用于绘图。你还将探索 Hadoop（HDFS 和 YARN），这将帮助你处理更大的数据集。本书还涵盖了 Spark，并解释了它如何与其他工具互动。

本书结束时，你将能够启动自己的 Python 环境，处理大文件，并操控数据生成统计信息、指标和图表。

### 关于作者

**伊凡·马林**是 Daitan Group（总部位于坎皮纳斯的软件公司）的一名系统架构师和数据科学家。他为大规模数据设计大数据系统，并使用 Python 和 Spark 实现端到端的机器学习管道。他还是圣保罗数据科学、机器学习和 Python 的积极组织者，并曾在大学层级开设过 Python 数据科学课程。

**安基特·舒克拉**是世界技术公司（World Wide Technology，一家领先的美国技术解决方案提供商）的一名数据科学家，他负责开发和部署机器学习与人工智能解决方案，以解决商业问题并为客户创造实际的经济价值。他还参与公司的研发计划，负责生产知识产权，建立新领域的能力，并在企业白皮书中发布前沿研究。除了调试 AI/ML 模型外，他还喜欢阅读，并且是个大食客。

**萨朗·VK**是 StraitsBridge Advisors 的首席数据科学家，他的职责包括需求收集、解决方案设计、开发以及使用开源技术开发和产品化可扩展的机器学习、人工智能和分析解决方案。与此同时，他还支持售前和能力建设。

### 学习目标

+   使用 Python 读取并将数据转换成不同格式

+   使用磁盘上的数据生成基本统计信息和指标

+   使用分布在集群上的计算任务

+   将各种来源的数据转换为存储或查询格式

+   为统计分析、可视化和机器学习准备数据

+   以有效的视觉形式呈现数据

### 方法

《Python 大数据分析》采取实践方法，帮助理解如何使用 Python 和 Spark 处理数据并将其转化为有用的内容。它包含多个活动，使用实际商业场景，让你在高度相关的背景下实践并应用你的新技能。

### 受众

《Python 大数据分析》是为希望掌握数据控制与转化成有价值洞察方法的 Python 开发者、数据分析师和数据科学家设计的。对统计测量和关系型数据库的基本知识将帮助你理解本书中解释的各种概念。

### 最低硬件要求

为了获得最佳学生体验，我们建议以下硬件配置：

处理器：Intel 或 AMD 4 核或更高

内存：8 GB RAM

存储：20 GB 可用空间

### 软件要求

你需要提前安装以下软件。

以下任一操作系统：

+   Windows 7 SP1 32/64 位

+   Windows 8.1 32/64 位 或 Windows 10 32/64 位

+   Ubuntu 14.04 或更高版本

+   macOS Sierra 或更高版本

+   浏览器：Google Chrome 或 Mozilla Firefox

+   Jupyter lab

你还需要提前安装以下软件：

+   Python 3.5+

+   Anaconda 4.3+

以下 Python 库已包含在 Anaconda 安装中：

+   matplotlib 2.1.0+

+   iPython 6.1.0+

+   requests 2.18.4+

+   NumPy 1.13.1+

+   pandas 0.20.3+

+   scikit-learn 0.19.0+

+   seaborn 0.8.0+

+   bokeh 0.12.10+

这些 Python 库需要手动安装：

+   mlxtend

+   version_information

+   ipython-sql

+   pdir2

+   graphviz

### 约定

文本中的代码字、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账户名如下所示：“要将数据转换为正确的数据类型，我们可以使用转换函数，例如 `to_datetime`、`to_numeric` 和 `astype`。”

一段代码如下所示：

```py
before the sort function:[23, 66, 12, 54, 98, 3]
after the sort function: [3, 12, 23, 54, 66, 98]
```

新术语和重要词汇用粗体显示。屏幕上看到的词汇，例如在菜单或对话框中，按如下方式出现在文本中：“**Pandas** ([`pandas.pydata.org`](https://pandas.pydata.org)) 是一个广泛应用于数据科学社区的数据处理和分析库。”

### 安装与设置

安装 Anaconda：

1.  访问 [`www.anaconda.com/download/`](https://www.anaconda.com/download/) 在浏览器中。

1.  根据你使用的操作系统点击 Windows、Mac 或 Linux。

1.  接下来，点击下载选项，确保下载最新版本。

1.  下载后打开安装程序。

1.  按照安装程序中的步骤操作，就完成了！你的 Anaconda 分发版已经准备好。

PySpark 可以在 PyPi 上获得。要安装 PySpark，请运行以下命令：

```py
pip install pyspark --upgrade
```

更新 Jupyter 并安装依赖项：

1.  搜索 Anaconda Prompt 并打开它。

1.  输入以下命令来更新 Conda 和 Jupyter：

    ```py
    #Update conda
    conda update conda
    #Update Jupyter
    conda update Jupyter
    #install packages
    conda install numpy
    conda install pandas
    conda install statsmodels
    conda install matplotlib
    conda install seaborn
    ```

1.  从 Anaconda Prompt 打开 Jupyter Notebook，请使用以下命令：

    ```py
    jupyter notebook
    pip install -U scikit-learn
    ```

### 安装代码包

将课堂代码包复制到 `C:/Code` 文件夹中。

### 附加资源

本书的代码包也托管在 GitHub 上，网址为：[`github.com/TrainingByPackt/Big-Data-Analysis-with-Python`](https://github.com/TrainingByPackt/Big-Data-Analysis-with-Python)。

我们还提供了其他代码包，来自我们丰富的图书和视频目录，网址为：[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。赶快去看看吧！
