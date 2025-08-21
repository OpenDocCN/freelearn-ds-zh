# 前言

本书专为经验丰富的数据科学家和开发人员设计，是您利用 Databricks 进行端到端机器学习项目的权威指南。假设读者具备扎实的 Python、统计学、机器学习生命周期基础，并对 Spark 有初步了解，本书旨在帮助专业人士从 DIY 环境或其他云平台过渡到 Databricks 生态系统。

从简洁的机器学习领域概述开始，接着深入探讨 Databricks 的功能和 MLflow 框架。涵盖数据准备、模型选择和训练等关键要素，同时利用 Databricks 特征存储进行高效的特征工程。使用 Databricks AutoML 快速启动项目，并学习如何通过 Databricks 工作流自动化模型再训练和部署。

本书结束时，您将熟练掌握使用 MLflow 进行实验追踪、团队协作，并解决诸如模型可解释性和治理等高级需求。本书包含大量实用的代码示例，重点介绍当前的普遍可用功能，同时帮助您快速适应机器学习、Databricks 和 MLflow 中的新兴技术。

# 本书适合谁阅读？

本书为精通 Python、统计学和机器学习生命周期的数据科学家和开发人员编写，是您过渡到 Databricks 的指南。特别适合那些从 DIY 或其他云平台迁移的读者，假设读者具备 Spark 入门知识，涵盖从头到尾的机器学习工作流。

# 本书内容涵盖

*第一章**，机器学习过程及其挑战*，概述了各个领域中不同的数据科学用例。它概述了机器学习项目中涉及的不同阶段和角色，从数据工程到分析、特征工程、以及机器学习模型的训练和部署。

*第二章**，Databricks 上的机器学习概述*，指导您完成注册 Databricks 试用帐户的过程，并探索专为机器学习从业者工作空间设计的机器学习功能。

*第三章**，利用特征存储*，为您介绍特征存储的概念。我们将引导您通过使用 Databricks 的离线特征存储来创建特征表，并演示它们的有效使用。此外，我们还将讨论在机器学习工作流中使用特征存储的优势。

*第四章**，理解 Databricks 上的 MLflow 组件*，帮助您了解 MLflow 是什么、其组件及其使用的好处。我们还将讲解如何在 MLflow 模型注册中心注册模型。

*第五章**, 使用 Databricks AutoML 创建基准模型*，介绍了什么是 AutoML，它的重要性，以及 Databricks 在 AutoML 方面的做法。我们还将使用 AutoML 创建一个基准模型。

*第六章**, 模型版本控制和 Webhooks*，教你如何利用 MLflow 模型注册表来管理模型版本、从不同阶段过渡到生产环境，并使用 webhooks 设置警报和监控。

*第七章**, 模型部署方法*，介绍了利用 Databricks 平台部署 ML 模型的不同选项。

*第八章**, 使用 Databricks Jobs 自动化 ML 工作流*，解释了 Databricks 作业是什么，以及如何将其作为强大的工具来自动化机器学习工作流。我们将介绍如何使用 Jobs API 设置 ML 训练工作流。

*第九章**, 模型漂移检测和再训练*，教你如何检测和防止生产环境中的模型漂移。

*第十章**, 使用 CI/CD 自动化模型再训练和重新部署*，演示了如何将 Databricks 的 ML 开发和部署设置为 CI/CD 管道。我们将使用书中之前学习的所有概念。

# 为了最大限度地利用本书

在深入探讨本书提供的动手实践和代码示例之前，了解软件和知识的前提条件非常重要。以下是概述你所需要的内容的总结表：

| **前提条件** | **描述** |
| --- | --- |
| Databricks Runtime | 本书针对 Databricks Runtime 13.3 LTS 及以上版本进行编写。 |
| Python 熟练度（3.x） | 你应当熟练掌握至少 Python 3.x，因为代码示例主要是用这个版本编写的。 |
| 统计学和 ML 基础 | 假设你对统计学和机器学习生命周期有深入的理解。 |
| Spark 知识（3.0 或以上） | 需要具备对 Apache Spark 3.0 或以上版本的基础了解，因为 Databricks 是基于 Spark 构建的。 |
| Delta Lake 功能（可选） | 对 Delta Lake 功能的基础知识理解可以增强你的理解，但并非强制要求。 |

为了充分利用本书中描述的所有功能和代码示例，你需要一个 Databricks 试用账户，试用期为 14 天。我们建议你计划好学习进度，在此期间完成动手活动。如果你发现这个平台有价值，并希望在试用期后继续使用，考虑联系你的 Databricks 联系人设置付费工作区。

**如果你使用的是本书的数字版本，我们建议你自己输入代码，或者从本书的 GitHub 仓库获取代码（下一个章节中有链接）。这样可以帮助你避免因复制粘贴代码而可能出现的错误。**

完成本书后，我们强烈建议你浏览 Databricks 文档中的最新功能，无论是私有预览还是公共预览。这将为你提供机器学习在 Databricks 上发展的未来方向，帮助你走在前沿，充分利用新兴功能。

# 下载示例代码文件

你可以从 GitHub 下载本书的示例代码文件，链接地址为 [`github.com/PacktPublishing/Practical-Machine-Learning-on-Databricks`](https://github.com/PacktPublishing/Practical-Machine-Learning-on-Databricks)。如果代码有更新，它将在 GitHub 仓库中进行更新。

我们还提供了其他代码包，来自我们丰富的书籍和视频目录，链接地址为 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。快来看看吧！

# 使用的约定

本书中使用了许多文本约定。

`文本中的代码`：表示文本中的代码字、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。例如：“在第五个单元格中，我们首先初始化一些参数，如现有用户名 `experiment_name`，这是与我们的 AutoML 关联的实验名称，以及 `registry_model_name`，这是模型在模型注册表中的名称。”

代码块设置如下：

```py
iris = load_iris() X = iris.data  # Features 
y = iris.target  # Labels
```

任何命令行输入或输出如下所示：

```py
from sklearn.datasets import load_iris  # Importing the Iris datasetfrom sklearn.model_selection import train_test_split  # Importing train_test_split function
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression model
```

**粗体**：表示一个新术语、一个重要单词或屏幕上显示的单词。例如，菜单或对话框中的单词会显示为 **粗体**。例如：“要查看你的运行时包含哪些库，可以参考 Databricks 运行时发布说明中的 **系统环境** 小节，检查你特定的运行时版本。”

提示或重要说明

如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果你对本书的任何部分有疑问，可以通过电子邮件联系 customercare@packtpub.com，并在邮件主题中注明书名。

**勘误表**：虽然我们已尽力确保内容的准确性，但错误仍然可能发生。如果你在本书中发现了错误，我们将非常感激你向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) 并填写表单。

**盗版**：如果你在互联网上遇到我们作品的任何非法复制品，我们将感激你提供相关网址或网站名称。请通过 copyright@packt.com 联系我们，并附上该材料的链接。

**如果你有兴趣成为作者**：如果你对某个话题有专业知识，并且有兴趣撰写或参与编写一本书，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 评论

请留下评论。一旦你阅读并使用了本书，为什么不在你购买它的网站上留下评论呢？潜在的读者可以通过你的公正意见做出购买决策，我们在 Packt 可以了解你对我们产品的看法，而我们的作者也能看到你对他们书籍的反馈。谢谢！

关于 Packt 的更多信息，请访问 [packtpub.com](http://packtpub.com)。

# 分享你的想法

一旦你阅读了 *Practical Machine Learning on Databricks*，我们很想听听你的想法！请 [点击这里直接进入亚马逊评论页面](https://packt.link/r/1-801-81203-9) 并分享你的反馈。

你的评论对我们和技术社区非常重要，将帮助我们确保提供卓越的内容质量。

# 下载本书的免费 PDF 版本

感谢购买本书！

你喜欢随时随地阅读，但又无法将印刷版书籍带到处吗？你的电子书购买无法与所选设备兼容吗？

不用担心，现在每本 Packt 书籍都会免费附带 DRM 无保护的 PDF 版本。

在任何地方、任何设备上阅读。直接从你最喜欢的技术书籍中搜索、复制和粘贴代码到你的应用程序中。

特权不止于此，你还可以获得独家折扣、新闻通讯和每天送到你邮箱的精彩免费内容

按照以下简单步骤来享受优惠：

1.  扫描二维码或访问以下链接

![](img/B17875_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781801812030`](https://packt.link/free-ebook/9781801812030)

1.  提交你的购买证明

1.  就是这样！我们会直接将你的免费 PDF 和其他福利发送到你的邮箱

# 第一部分：简介

本部分主要关注数据科学用例、数据科学项目的生命周期以及涉及的角色（数据工程师、分析师和科学家），以及组织中机器学习开发的挑战。

本节包含以下章节：

+   *第一章*，*机器学习过程及其挑战*

+   *第二章*，*Databricks 上的机器学习概述*
