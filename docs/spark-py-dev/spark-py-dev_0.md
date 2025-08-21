# 前言

*Python 开发人员的 Spark*旨在将 Python 的优雅和灵活性与 Apache Spark 的强大和多功能性相结合。Spark 是用 Scala 编写的，并在 Java 虚拟机上运行。然而，它是多语言的，并为 Java、Scala、Python 和 R 提供了绑定和 API。Python 是一种设计良好的语言，具有广泛的专业库。本书探讨了 PySpark 在 PyData 生态系统中的应用。一些著名的 PyData 库包括 Pandas、Blaze、Scikit-Learn、Matplotlib、Seaborn 和 Bokeh。这些库是开源的。它们由数据科学家和 Python 开发人员社区开发、使用和维护。PySpark 与 PyData 生态系统很好地集成在一起，得到了 Anaconda Python 发行版的认可。本书提出了一个构建数据密集型应用程序的旅程，以及涵盖以下步骤的架构蓝图：首先，使用 Spark 建立基础设施。其次，获取、收集、处理和存储数据。第三，从收集的数据中获得见解。第四，实时传输数据并实时处理。最后，可视化信息。

本书的目标是通过构建分析社交网络上 Spark 社区互动的应用程序来学习 PySpark 和 PyData 库。重点是 Twitter 数据。

# 本书内容

第一章，“设置 Spark 虚拟环境”，介绍了如何创建一个分隔的虚拟机作为我们的沙盒或开发环境，以实验 Spark 和 PyData 库。它涵盖了如何安装 Spark 和 Python Anaconda 发行版，其中包括 PyData 库。在此过程中，我们解释了关键的 Spark 概念、Python Anaconda 生态系统，并构建了一个 Spark 词频统计应用程序。

第二章，“使用 Spark 构建批处理和流处理应用程序”，奠定了*数据密集型应用程序架构*的基础。它描述了应用程序架构蓝图的五个层次：基础设施、持久性、集成、分析和参与。我们与三个社交网络建立了 API 连接：Twitter、GitHub 和 Meetup。本章提供了连接到这三个非平凡 API 的工具，以便您在以后阶段创建自己的数据混搭。

第三章，“使用 Spark 处理数据”，介绍了如何从 Twitter 收集数据，并使用 Pandas、Blaze 和 SparkSQL 以及它们各自的数据框架数据结构进行处理。我们继续使用 Spark SQL 进行进一步的调查和技术，利用 Spark 数据框架数据结构。

第四章，“使用 Spark 从数据中学习”，概述了 Spark MLlib 算法库的不断扩展。它涵盖了监督学习和无监督学习、推荐系统、优化和特征提取算法。我们通过 Python Scikit-Learn 和 Spark MLlib K-means 聚类将 Twitter 收集的数据集进行了处理，以区分与*Apache Spark*相关的推文。

第五章，“使用 Spark 流式传输实时数据”，奠定了流式架构应用程序的基础，并描述了它们的挑战、约束和好处。我们用 TCP 套接字来说明流式传输的概念，然后直接从 Twitter firehose 进行实时推文摄取和处理。我们还描述了 Flume，这是一个可靠、灵活和可扩展的数据摄取和传输管道系统。Flume、Kafka 和 Spark 的结合在不断变化的环境中提供了无与伦比的稳健性、速度和灵活性。我们在本章结束时对两种流式架构范式——Lambda 和 Kappa 架构进行了一些评论和观察。

第六章，*可视化洞察和趋势*，侧重于一些关键的可视化技术。它涵盖了如何构建词云并展示它们直观的力量，以揭示成千上万条推文中携带的关键词、情绪和表情。然后，我们专注于使用 Bokeh 进行交互式地图可视化。我们从零开始构建世界地图，并创建关键推文的散点图。我们最终的可视化是将伦敦的实际谷歌地图叠加在一起，突出即将举行的聚会及其各自的主题。

# 本书所需内容

您需要好奇心、毅力和对数据、软件工程、应用架构和可扩展性以及简洁美观的可视化的热情。范围广泛。

您需要对 Python 或具有面向对象和函数式编程能力的类似语言有很好的理解。有使用 Python、R 或任何类似工具进行数据整理的初步经验会有所帮助。

您需要欣赏如何构想、构建和扩展数据应用程序。

# 本书的受众

目标受众包括以下内容：

+   数据科学家是主要的利益相关方。本书将帮助您释放 Spark 的力量，并利用您的 Python、R 和机器学习背景。

+   专注于 Python 的软件开发人员将很容易扩展他们的技能，使用 Spark 作为处理引擎和 Python 可视化库和 Web 框架创建数据密集型应用程序。

+   数据架构师可以创建快速数据管道，并构建包含批处理和流处理的著名 Lambda 架构，以实时渲染数据洞察，使用 Spark 和 Python 丰富的生态系统，也将受益于本书。

# 约定

在本书中，您会发现一些区分不同类型信息的文本样式。以下是一些这些样式的示例，以及它们的含义解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“在存储 Jupyter 或 IPython 笔记本的目录`examples/AN_Spark`中使用`IPYNB`启动 PySpark”。

代码块设置如下：

```py
# Word count on 1st Chapter of the Book using PySpark

# import regex module
import re
# import add from operator module
from operator import add

# read input file
file_in = sc.textFile('/home/an/Documents/A00_Documents/Spark4Py 20150315')
```

任何命令行输入或输出都以以下方式编写：

```py
# install anaconda 2.x.x
bash Anaconda-2.x.x-Linux-x86[_64].sh

```

**新术语**和**重要单词**以粗体显示。例如，屏幕上看到的单词，比如菜单或对话框中的单词，会在文本中以这种方式出现：“安装 VirtualBox 后，让我们打开 Oracle VM VirtualBox Manager 并单击**New**按钮。”

### 注意

警告或重要说明会以这种方式出现在一个框中。

### 提示

提示和技巧会以这种方式出现。
