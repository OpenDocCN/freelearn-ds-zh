# 前言

当今数据的复杂结构需要高级解决方案来进行数据转换及其语义表示，以便使信息更容易为用户获取。Apache Hadoop 以及众多其他大数据工具，使您能够相对轻松地构建此类解决方案。本书列出了一些独特思想和技巧，使您能够克服成为大数据架构专家道路上的不同数据处理和分析挑战。

本书首先迅速阐述了企业数据架构的原则，并展示了它们与 Apache Hadoop 生态系统的关系。您将全面了解使用 Hadoop 的数据生命周期管理，随后在 Hadoop 中建模结构化和非结构化数据。本书还将向您展示如何利用 Apache Spark 等工具设计实时流管道，以及如何使用 Elasticsearch 等工具构建高效的搜索引擎解决方案。您将在 Hadoop 上构建企业级分析解决方案，并学习如何使用 Tableau 和 Python 等工具可视化您的数据。

本书还涵盖了在本地和云上部署大数据解决方案的技术，以及管理和管理您的 Hadoop 集群的专家技术。

在本书结束时，您将拥有构建满足任何数据或洞察需求的大数据系统的全部知识，利用现代大数据框架和工具的全套功能。您将具备成为真正的数据专家所需的技能和知识。

# 这本书面向谁

本书面向希望快速进入 Hadoop 行业并成为大数据架构专家的大数据专业人士。项目经理和希望在大数据和 Hadoop 领域建立职业生涯的主机专业人员也将发现本书很有用。为了从本书中获得最佳效果，需要对 Hadoop 有一定的了解。

# 这本书涵盖的内容

第一章，*企业数据架构原则*，展示了如何在 Hadoop 集群中存储和建模数据。

第二章，*Hadoop 生命周期管理*，涵盖了各种数据生命周期阶段，包括数据创建、共享、维护、归档、保留和删除。它还进一步详细介绍了数据安全工具和模式。

第三章，*Hadoop 设计考虑因素*，涵盖了关键数据架构原则和实践。读者将了解现代数据架构师如何适应大数据架构用例。

第四章，*数据移动技术*，涵盖了将数据传输到和从我们的 Hadoop 集群的不同方法，以充分利用其实力。

第五章，*Hadoop 中的数据建模*，展示了如何使用云基础设施构建企业应用程序。

第六章，*设计实时流数据管道*，涵盖了设计实时数据分析的不同工具和技术。

第七章，*大规模数据处理框架*，描述了企业数据架构原则以及管理和保护这些数据的重要性。

第八章，*构建企业搜索平台*，提供了使用 Elasticsearch 构建搜索解决方案的详细架构设计。

第九章，*设计数据可视化解决方案*，展示了如何使用 Apache Ambari 部署您的 Hadoop 集群。

第十章，*使用云开发应用程序*，涵盖了可视化数据的不同方法和选择正确可视化方法所涉及的因素。

第十一章，*生产 Hadoop 集群部署*，涵盖了从我们的数据中提取价值的不同数据处理解决方案。

# 为了充分利用本书

如果 Hadoop 的安装如前几章所述完成得很好，那就太好了。对 Hadoop 的详细或基本了解将是一个额外的优势。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载与勘误。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

下载文件后，请确保您使用最新版本解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Modern-Big-Data-Processing-with-Hadoop`](https://github.com/PacktPublishing/Modern-Big-Data-Processing-with-Hadoop)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/ModernBigDataProcessingwithHadoop_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/ModernBigDataProcessingwithHadoop_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。以下是一个示例：“将下载的 `WebStorm-10*.dmg` 磁盘镜像文件挂载为系统中的另一个磁盘。”

代码块应如下设置：

```py
export HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-$YARN_HOME/etc/hadoop}"
export HADOOP_COMMON_HOME="${HADOOP_COMMON_HOME:-$YARN_HOME}"
export HADOOP_HDFS_HOME="${HADOOP_HDFS_HOME:-$YARN_HOME}"  
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
$ hadoop fs -cat /tmp/output-7/part*
 NewDelhi, 440
 Kolkata, 390
 Bangalore, 270
```

任何命令行输入或输出都应如下编写：

```py
useradd hadoop
passwd hadoop1 
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下显示。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要提示将如下所示。

小贴士和技巧将如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送电子邮件给我们。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告此错误。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们将不胜感激，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 `copyright@packtpub.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评论？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/).
