# 前言

*Spark for Python Developers*旨在结合 Python 的优雅性和灵活性以及 Apache Spark 的强大功能和多功能性。Spark 是用 Scala 编写的，在 Java 虚拟机上运行。尽管如此，它仍然是多语言的，并为 Java、Scala、Python 和 R 提供了绑定和 API。Python 是一种设计良好的语言，拥有丰富的专用库集。本书在 PyData 生态系统中探讨了 PySpark。一些突出的 PyData 库包括 Pandas、Blaze、Scikit-Learn、Matplotlib、Seaborn 和 Bokeh。这些库是开源的。它们由数据科学家和 Python 开发者社区开发、使用和维护。PySpark 与 PyData 生态系统很好地集成，正如 Anaconda Python 发行版所认可的那样。本书提出了一条构建数据密集型应用程序的路线，以及一个涵盖以下步骤的架构蓝图：首先，使用 Spark 设置基本基础设施。其次，获取、收集、处理和存储数据。第三，从收集的数据中获得洞察。第四，流式传输实时数据并实时处理。最后，可视化信息。

本书的目标是通过构建分析 Spark 社区在社交网络上互动的应用程序来学习 PySpark 和 PyData 库。重点是 Twitter 数据。

# 本书涵盖的内容

第一章, *设置 Spark 虚拟环境*，涵盖了如何创建一个隔离的虚拟机作为我们的沙盒或开发环境，以实验 Spark 和 PyData 库。它涵盖了如何安装 Spark 和 Python Anaconda 发行版，该发行版包括 PyData 库。在这个过程中，我们解释了关键 Spark 概念、Python Anaconda 生态系统，并构建了一个 Spark 词频应用程序。

第二章, *使用 Spark 构建批处理和流式应用程序*，为*数据密集型应用程序架构*奠定了基础。它描述了应用程序架构蓝图中的五个层级：基础设施、持久化、集成、分析和参与。我们与三个社交网络建立了 API 连接：Twitter、GitHub 和 Meetup。本章提供了连接这三个非平凡 API 的工具，以便您可以在以后阶段创建自己的数据融合。

第三章, *使用 Spark 处理数据*，涵盖了如何从 Twitter 中提取数据，并使用 Pandas、Blaze 和 SparkSQL 及其各自的数据框数据结构实现来处理这些数据。我们继续使用 Spark SQL 进行进一步调查和技术研究，利用 Spark 数据框数据结构。

第四章, *使用 Spark 从数据中学习*，概述了 Spark MLlib 算法库不断扩大的算法库。它涵盖了监督学习和无监督学习、推荐系统、优化和特征提取算法。我们将从 Twitter 收集的数据集通过 Python Scikit-Learn 和 Spark MLlib K-means 聚类来分离与*Apache Spark*相关的推文。

第五章, *使用 Spark 进行实时数据流*，阐述了流式架构应用的基础，并描述了它们的挑战、限制和优势。我们使用 TCP 套接字来阐述流式概念，然后直接从 Twitter 的实时数据流中获取并处理实时推文。我们还描述了 Flume，这是一个可靠、灵活且可扩展的数据摄取和传输管道系统。Flume、Kafka 和 Spark 的结合在不断变化的环境中提供了无与伦比的鲁棒性、速度和敏捷性。我们在本章的最后对两种流式架构范式，即 Lambda 架构和 Kappa 架构，进行了一些评论和观察。

第六章, *可视化洞察和趋势*，专注于几种关键的可视化技术。它涵盖了如何构建词云并展示其直观的力量，以揭示成千上万条推文中携带的大量关键词、情绪和梗。然后我们专注于使用 Bokeh 的交互式映射可视化。我们从零开始构建世界地图，并创建关键推文的散点图。我们的最终可视化是在伦敦的实际谷歌地图上叠加，突出即将举行的聚会及其相应的话题。

# 您需要这本书所需的东西

您需要好奇心、毅力以及对数据、软件工程、应用架构和可扩展性以及美丽简洁的视觉化的热情。范围广泛。

您需要具备对 Python 或具有面向对象和函数式编程能力的类似语言的良好理解。使用 Python、R 或任何类似工具进行数据整理的初步经验将有所帮助。

您需要欣赏如何构思、构建和扩展数据应用。

# 这本书面向的对象

目标受众包括以下人群：

+   数据科学家是主要感兴趣的各方。这本书将帮助您释放 Spark 的力量，并利用您的 Python、R 和机器学习背景。

+   专注于 Python 的软件开发者将能够迅速扩展他们的技能，使用 Spark 作为处理引擎和 Python 可视化库以及 Web 框架来创建数据密集型应用。

+   能够创建快速数据管道并构建包含批处理和流处理以实时提供数据洞察的著名 Lambda 架构的数据架构师也将从这本书中受益。

# 惯例

在这本书中，您将找到许多不同风格的文章，以区分不同类型的信息。以下是一些这些风格的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称将如下所示：“在存储 Jupyter 或 IPython 笔记本的`examples/AN_Spark`目录中，使用`IPYNB`启动 PySpark”。

代码块将如下设置：

```py
# Word count on 1st Chapter of the Book using PySpark

# import regex module
import re
# import add from operator module
from operator import add

# read input file
file_in = sc.textFile('/home/an/Documents/A00_Documents/Spark4Py 20150315')
```

任何命令行输入或输出将如下所示：

```py
# install anaconda 2.x.x
bash Anaconda-2.x.x-Linux-x86[_64].sh

```

**新术语**和**重要词汇**将以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，将以如下方式显示：“安装 VirtualBox 后，让我们打开 Oracle VM VirtualBox 管理器并点击**新建**按钮。”

### 注意

警告或重要提示将以这样的框显示。

### 小贴士

小技巧和窍门如下所示。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正能从中获得最大价值的标题非常重要。

要发送给我们一般反馈，只需发送电子邮件到`<feedback@packtpub.com>`，并在您的邮件主题中提及书籍标题。

如果您在某个主题上具有专业知识，并且您对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您已经是 Packt 书籍的骄傲所有者，我们有一些事情可以帮助您从您的购买中获得最大价值。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载您购买的所有 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 错误清单

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何错误清单，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**错误清单提交表**链接，并输入您的错误清单详情来报告它们。一旦您的错误清单得到验证，您的提交将被接受，错误清单将被上传到我们的网站，或添加到该标题的错误清单部分。任何现有的错误清单都可以通过从[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看。

## 侵权

互联网上对版权材料的盗版是一个持续存在的问题，涉及所有媒体。在 Packt，我们非常重视保护我们的版权和许可证。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们提供有价值内容的能力方面的帮助。

## 问题和建议

如果你在本书的任何方面遇到问题，可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决。
