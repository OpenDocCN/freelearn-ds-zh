# 前言

随着数据集大小的爆炸性增长，廉价云存储的引入，以及近实时数据处理成为行业标准，许多组织转向湖屋架构，它将传统数据仓库的快速商业智能（BI）速度与云中大数据的可扩展 ETL 处理相结合。Databricks 数据智能平台——建立在包括 Apache Spark、Delta Lake、MLflow 和 Unity Catalog 在内的多个开源技术之上——消除了摩擦点，加速了现代数据应用的设计和部署，这些应用是为湖屋而构建的。

在本书中，您将首先概览 Delta Lake 格式，涵盖 Databricks 数据智能平台的核心概念，并掌握使用 Delta Live Tables 框架构建数据管道的技巧。我们将深入探讨数据转换的应用，如何实现 Databricks 镶嵌架构，以及如何持续监控数据在数据湖屋中的质量。您将学习如何使用 Databricks Auto Loader 功能对接收到的数据做出响应，并通过 Databricks 工作流实现实时数据处理的自动化。您还将学习如何使用 CI/CD 工具，如**Terraform**和**Databricks 资产包**（**DABs**），自动化部署数据管道更改到不同的部署环境，并在此过程中监控、控制和优化云成本。在本书结束时，您将掌握如何使用 Databricks 数据智能平台构建一个生产就绪的现代数据应用。

由于 Databricks 最近被评为 2024 年 Gartner 数据科学与机器学习平台魔力象限的领导者，预计未来几年对掌握 Databricks 数据智能平台技能的需求将持续增长。

# 本书适用人群

本书适用于数据工程师、数据科学家和负责组织企业数据处理的数据管理员。本书将简化在 Databricks 上学习高级数据工程技术的过程，使实现前沿的湖屋架构对具有不同技术水平的个人都变得可访问。然而，为了最大化利用本书中的代码示例，您需要具备 Apache Spark 和 Python 的初级知识。

# 本书内容

*第一章*，*Delta Live Tables 介绍*，讨论了如何使用 Delta Live Tables 框架构建近实时数据管道。它涵盖了管道设计的基础知识，以及 Delta Lake 格式的核心概念。本章最后通过一个简单的示例展示了如何从头到尾构建一个 Delta Live Table 数据管道。

*第二章*，*使用 Delta Live Tables 应用数据转化*，探讨了如何使用 Delta Live Tables 进行数据转化，指导你清洗、精炼和丰富数据以满足特定的业务需求。你将学习如何使用 Delta Live Tables 从多种输入源中摄取数据，在 Unity Catalog 中注册数据集，并有效地将变更应用于下游表。

*第三章*，*使用 Delta Live Tables 管理数据质量*，介绍了对新到数据强制执行数据质量要求的几种方法。你将学习如何使用 Delta Live Tables 框架中的期望定义数据质量约束，并实时监控管道的数据质量。

*第四章*，*扩展 DLT 管道*，解释了如何扩展**Delta Live Tables**（**DLT**）管道，以应对典型生产环境中不可预测的需求。你将深入了解如何使用 DLT UI 和 Databricks Pipeline REST API 配置管道设置。此外，你还将更好地理解日常运行在后台的 DLT 维护任务，以及如何优化表格布局以提升性能。

*第五章*，*通过 Unity Catalog 精通湖仓中的数据治理*，提供了一个全面的指南，帮助你使用 Unity Catalog 增强湖仓的数据治理和合规性。你将学习如何在 Databricks 工作区启用 Unity Catalog，使用元数据标签启用数据发现，并实施数据集的精细粒度行级和列级访问控制。

*第六章*，*在 Unity Catalog 中管理数据位置*，探讨了如何使用 Unity Catalog 有效地管理存储位置。你将学习如何在组织内部跨不同角色和部门管理数据访问，同时确保安全性和可审计性，利用 Databricks 数据智能平台实现这一目标。

*第七章*，*使用 Unity Catalog 查看数据血缘*，讨论了追踪数据来源、可视化数据转化过程，以及通过在 Unity Catalog 中追踪数据血缘来识别上下游依赖关系。在本章结束时，你将掌握验证数据来源是否可信的技能。

*第八章*，*使用 Terraform 部署、维护和管理 DLT 管道*，介绍了如何使用 Databricks Terraform 提供程序来部署 DLT 管道。你将学习如何设置本地开发环境，并自动化构建和部署管道，同时涵盖最佳实践和未来考虑事项。

*第九章*，*利用 Databricks 资产包简化数据管道部署*，探讨了如何使用 DABs 简化数据分析项目的部署，并促进跨团队协作。你将通过几个实践案例深入理解 DABs 的实际应用。

*第十章*，*监控生产中的数据管道*，深入探讨了在 Databricks 中监控数据管道这一关键任务。你将学习到在 Databricks 数据智能平台中追踪管道健康、性能和数据质量的各种机制。

# 为了充分利用本书

虽然这不是强制要求，但为了充分利用本书，建议你具备 Python 和 Apache Spark 的初学者水平知识，并且至少对如何在 Databricks 数据智能平台中导航有一定了解。为了配合本书中的实践练习和代码示例，建议在本地安装以下依赖项：

| **书中涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python 3.6+ | Windows、macOS 或 Linux |
| Databricks CLI 0.205+ |

此外，建议你拥有一个 Databricks 账户和工作区，以便登录、导入笔记本、创建集群和创建新的数据管道。如果你没有 Databricks 账户，可以在 Databricks 网站上注册免费试用：[`www.databricks.com/try-databricks`](https://www.databricks.com/try-databricks)。

**如果你使用的是本书的数字版本，我们建议你自己输入代码，或者从本书的 GitHub 仓库获取代码（下一个章节中有链接）。这样做可以帮助你避免与代码复制和粘贴相关的潜在错误** **。**

# 下载示例代码文件

你可以从 GitHub 下载本书的示例代码文件：[`github.com/PacktPublishing/Building-Modern-Data-Applications-Using-Databricks-Lakehouse`](https://github.com/PacktPublishing/Building-Modern-Data-Applications-Using-Databricks-Lakehouse)。如果代码有更新，将会在 GitHub 仓库中更新。

我们还从丰富的书籍和视频目录中提供其他代码包，可以在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 找到。快来看看吧！

# 使用的约定

本书中使用了若干文本约定。

**文本中的代码**：表示文本中的代码词汇、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。举个例子：“数据生成器笔记本的结果应为三个表：**youtube_channels**，**youtube_channel_artists**，和 **combined_table**。”

代码块的设置如下：

```py
@dlt.table(
    name="random_trip_data_raw",
    comment="The raw taxi trip data ingested from a landing zone.",
    table_properties={
        "quality": "bronze"
    }
)
```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目会以粗体显示：

```py
@dlt.table(
    name="random_trip_data_raw",
    comment="The raw taxi trip data ingested from a landing zone.",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "false"
    }
)
```

任何命令行输入或输出均按如下格式书写：

```py
$ databricks bundle validate
```

**粗体**：表示新术语、重要词汇或您在屏幕上看到的词语。例如，菜单或对话框中的文字会以**粗体**显示。以下是一个例子：“点击 Databricks 工作区右上方的**运行全部**按钮，执行所有笔记本单元格，验证所有单元格是否成功执行。”

提示或重要说明

显示如下。

# 与我们联系

我们始终欢迎读者的反馈。

**常见反馈**：如果您对本书的任何内容有疑问，请通过电子邮件联系我们：customercare@packtpub.com，并在邮件主题中注明书名。

**勘误**：虽然我们已经尽力确保内容的准确性，但错误还是可能发生。如果您在本书中发现了错误，我们将非常感激您向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们将非常感激您提供该位置地址或网站名称。请通过版权@packtpub.com 与我们联系，附上相关链接。

**如果您有兴趣成为作者**：如果您在某个领域拥有专业知识，并且有兴趣撰写或参与编写书籍，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

阅读完*《使用 Databricks Lakehouse 构建现代数据应用》*后，我们非常期待听到您的想法！请[点击这里直接访问亚马逊的书评页面](https://packt.link/r/1-801-07323-6)，并分享您的反馈。

您的反馈对我们和技术社区非常重要，它将帮助我们确保提供优质的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

喜欢随时随地阅读，但又无法随身携带纸质书籍吗？

您的电子书购买无法与您选择的设备兼容吗？

不用担心，现在每本 Packt 书籍都会提供免费的无 DRM 版本 PDF。

任何地方、任何设备上都能阅读。可以搜索、复制并粘贴您最喜欢的技术书籍中的代码，直接用于您的应用程序。

优惠不止于此，您还可以独享折扣、新闻通讯和每日发送到您邮箱的优质免费内容

按照这些简单步骤获取福利：

1.  扫描二维码或访问以下链接

![img](img/B22011_QR_Free_PDF.jpg)

[`packt.link/free-ebook/978-1-80107-323-3`](https://packt.link/free-ebook/978-1-80107-323-3)

1.  提交您的购买凭证

1.  就是这样！我们会将您的免费 PDF 和其他福利直接发送到您的邮箱

# 第一部分：湖仓的近实时数据管道

本书的第一部分将介绍**Delta Live Tables**（**DLT**）框架的核心概念。我们将讨论如何从各种输入源中获取数据，并将最新的更改应用到下游表格中。我们还将探讨如何对传入的数据施加要求，以便在可能污染湖仓的数据质量问题发生时，及时通知数据团队。

本部分包含以下章节：

+   *第一章* , *Delta Live Tables 简介*

+   *第二章* , *使用 Delta Live Tables 应用数据转换*

+   *第三章* , *使用 Delta Live Tables 管理数据质量*

+   *第四章* , *扩展 DLT 数据管道*
