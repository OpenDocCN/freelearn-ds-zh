# 前言

Apache Storm 是一个强大的框架，用于创建复杂的流程，可以处理大量数据。它通过通用的喷嘴和螺栓概念，以及简单的部署和监控工具，允许开发者专注于他们工作流程的具体细节，无需重新发明轮子。

然而，Storm 是用 Java 编写的。虽然它支持除了 Java 之外的其他编程语言，但工具不完整，文档很少，示例也很少。

本书的一位作者创建了 Petrel，这是第一个完全支持使用 100% Python 创建 Storm 拓扑的框架。他亲身经历了在 Java 工具集上构建 Python Storm 拓扑的挑战。本书填补了这一空白，为所有经验水平的 Python 开发者提供了一个资源，帮助他们使用 Storm 构建自己的应用程序。

# 本书涵盖的内容

第一章，*熟悉 Storm*，提供了关于 Storm 的用例、不同的安装模式和 Storm 的配置的详细信息。

第二章，*Storm 的解剖结构*，介绍了 Storm 特定的术语、流程、Storm 的容错性、在 Storm 中调整并行性以及保证元组处理，并对这些内容的每个方面进行了详细解释。

第三章，*介绍 Petrel*，介绍了一个用于在 Python 中构建 Storm 拓扑的框架。本章介绍了 Petrel 的安装，并包含了一个简单的示例。

第四章，*示例拓扑 – Twitter*，提供了一个深入示例，展示了实时计算 Twitter 数据统计信息的拓扑。示例介绍了使用滴答元组，这对于需要按计划计算统计信息或其他操作的拓扑非常有用。在本章中，您还可以看到拓扑如何访问配置数据。

第五章，*使用 Redis 和 MongoDB 进行持久化*，更新了示例 Twitter 拓扑以使用 Redis，这是一个流行的键值存储。它展示了如何使用内置的 Redis 操作简化复杂的 Python 计算逻辑。本章以将 Twitter 数据存储在 MongoDB（一个流行的 NoSQL 数据库）中的示例结束，并使用其聚合功能生成报告。

第六章，*Petrel 在实践中*，教授了使开发者更有效地使用 Storm 的实用技能。您将学习如何使用 Petrel 为运行在 Storm 之外的喷嘴和螺栓组件创建自动化测试。您还可以看到如何使用图形调试器调试运行在 Storm 内部的拓扑。

附录, *使用 Supervisord 管理 Storm*，是使用集群上的管理器对 Storm 进行监控和控制的实际演示。

# 您需要这本书什么

您需要一个装有 Python 2.7、Java 7 JDK 和 Apache Storm 0.9.3 的计算机。推荐使用 Ubuntu，但不是必需的。

# 这本书面向的对象

这本书适合初学者以及希望使用 Storm 实时处理大数据的 Python 高级开发者。虽然熟悉 Java 运行环境有助于安装和配置 Storm，但本书中的所有代码示例都是用 Python 编写的。

# 规范

在这本书中，您将找到许多不同风格的文本，以区分不同类型的信息。以下是一些这些风格的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示："可以使用`storm.yaml`进行 Storm 配置，该文件位于`conf`文件夹中"。

代码块如下设置：

```py
import nltk.corpus

from petrel import storm
from petrel.emitter import BasicBolt

class SplitSentenceBolt(BasicBolt):
    def __init__(self):
        super(SplitSentenceBolt, self).__init__(script=__file__)
        self.stop = set(nltk.corpus.stopwords.words('english'))
        self.stop.update(['http', 'https', 'rt'])
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
import logging
from collections import defaultdict

from petrel import storm
from petrel.emitter import BasicBolt
```

任何命令行输入或输出都如下所示：

```py
tail -f petrel24748_totalrankings.log
```

**新术语**和**重要词汇**以粗体显示。屏幕上看到的单词，例如在菜单或对话框中，在文本中如下所示："最后，点击**创建您的 Twitter 应用程序**"。

### 注意

警告或重要提示将以如下框的形式出现。

### 小贴士

小贴士和技巧如下所示。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正从中受益的标题非常重要。

要向我们发送一般反馈，只需发送一封电子邮件到`<feedback@packtpub.com>`，并在邮件的主题中提及书名。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经成为 Packt 书籍的骄傲拥有者，我们有一些东西可以帮助您充分利用您的购买。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载您购买的所有 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 错误清单

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者感到沮丧，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站，或添加到该标题的勘误部分下的现有勘误列表中。您可以通过从[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看任何现有勘误。

## 盗版

互联网上版权材料的盗版是一个持续存在的问题，跨越所有媒体。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

如果您在本书的任何方面遇到问题，请通过`<copyright@packtpub.com>`联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

## 问题

如果您在本书的任何方面遇到问题，可以通过`<questions@packtpub.com>`联系我们，我们将尽力解决。
