# 前言

*学习 Jupyter* 讨论了如何使用 Jupyter 记录脚本并生成数据分析项目的结果。Jupyter 让数据科学家能够记录完整的分析过程，类似于其他科学家使用实验室笔记本记录测试、进展、结果和结论的方式。Jupyter 支持多种操作系统，本书将介绍在 Windows 和 macOS 中使用 Jupyter，以及满足您特定需求所需的各种步骤。通过添加语言引擎，Jupyter 支持多种脚本语言，使用户能够以原生方式使用特定的脚本。

# 本书适合谁阅读

本书是为那些希望以自然编程的方式向他人展示软件解决方案的读者编写的。Jupyter 提供了一种机制，可以执行多种语言并直接存储结果以供显示，就像用户在自己的机器上运行了这些脚本一样。

# 本书内容

第一章，*Jupyter 入门*，探讨了笔记本中可用的各种用户界面元素。我们将学习如何在 macOS 或 PC 上安装软件。我们将揭示笔记本的结构。我们将看到开发笔记本时的典型工作流程。我们将逐步了解笔记本中的用户界面操作。最后，我们将查看一些高级用户为笔记本提供的配置选项。

第二章，*Jupyter Python 脚本编写*，带您了解一个简单的笔记本及其基础结构。然后，我们将看到一个使用 pandas 的示例，并查看图形示例。最后，我们将看到一个使用 Python 脚本生成随机数的示例。

第三章，*Jupyter R 脚本编写*，增加了在我们的 Jupyter Notebook 中使用 R 脚本的功能。我们将添加一个不包含在标准 R 安装中的 R 库，并编写一个 `Hello World` R 脚本。接着，我们将了解 R 数据访问内置库以及一些自动生成的简单图形和统计数据。我们将使用 R 脚本生成几种不同方式的 3D 图形。然后，我们将进行标准的聚类分析（我认为这是 R 的基本用途之一），并使用其中一个预测工具。我们还将构建一个预测模型并测试其准确性。

第四章，*Jupyter Julia 脚本*，增加了在 Jupyter Notebook 中使用 Julia 脚本的功能。我们将添加一个不包含在标准 Julia 安装中的 Julia 库。我们将展示 Julia 的基本特性，并概述在 Jupyter 中使用 Julia 时遇到的一些局限性。我们将使用一些可用的图形包来展示图形。最后，我们将看到并行处理的实际应用，一个小的控制流示例，以及如何为 Julia 脚本添加单元测试。

第五章，*Jupyter Java 编程*，讲解了如何将 Java 引擎安装到 Jupyter 中。我们将看到 Java 在 Jupyter 中的不同输出展示示例。接着，我们将研究如何使用可选字段。我们将看到 Java 在 Jupyter 中的编译错误。接下来，我们将看到几个 lambda 示例。我们将使用集合完成多个任务。最后，我们将为一个标准数据集生成汇总统计。

第六章，*Jupyter JavaScript 编程*，展示了如何在 Jupyter Notebook 中添加 JavaScript。我们将看到在 Jupyter 中使用 JavaScript 的一些局限性。我们将查看一些典型的 Node.js 编程包的示例，包括图形、统计、内置 JSON 处理，以及使用第三方工具创建图形文件。我们还将看到如何在 Jupyter 中使用 Node.js 开发多线程应用程序。最后，我们将使用机器学习开发决策树。

第七章，*Jupyter Scala*，讲解了如何为 Jupyter 安装 Scala。我们将使用 Scala 编程来访问大型数据集。我们将看到 Scala 如何操作数组。我们将生成 Scala 中的随机数。这里有高阶函数和模式匹配的示例。我们将使用 case 类。我们将看到 Scala 中不变性的示例。我们将使用 Scala 包构建集合，并将讨论 Scala 特质。

第八章，*Jupyter 与大数据*，讨论了如何通过 Python 编程在 Jupyter 中使用 Spark 功能。首先，我们将在 Windows 和 macOS 机器上安装 Spark 插件。我们将编写一个初步的脚本，仅读取文本文件中的行。然后，我们将进一步处理，计算文件中的单词数。我们将为结果添加排序。这里有一个估算 pi 的脚本。我们将分析网页日志文件中的异常。我们将确定一组素数，并分析文本流中的一些特征。

第九章，*互动小部件*，讲解了如何向我们的 Jupyter 安装添加小部件。我们将使用 interact 和 interactive 小部件来生成多种用户输入控件。接下来，我们将深入研究小部件包，探讨可用的用户控件、容器中的属性以及控件发出的事件，并学习如何构建控件容器。

第十章，*共享和转换 Jupyter Notebooks*，介绍了如何在笔记本服务器上共享笔记本。我们将把笔记本添加到 Web 服务器并通过 GitHub 分发。我们还将探讨如何将笔记本转换为不同格式，例如 HTML 和 PDF。

第十一章，*多用户 Jupyter Notebooks*，展示了如何让多个用户同时使用一个笔记本。我们将展示共享错误发生的例子。我们将安装一个解决该问题的 Jupyter 服务器，并使用 Docker 缓解这个问题。

第十二章，*接下来怎么办？*，探讨了未来可能会融入 Jupyter 的一些功能。

# 为了最大限度地利用本书

本书中的步骤假设你拥有一台现代的 Windows 或 macOS 设备，并且能够访问互联网。书中有几个步骤需要安装软件，因此你需要具有该设备的管理员权限才能进行操作。

本书的预期是，你有一个或多个自己喜欢的实现语言，想在 Jupyter 上使用。

# 下载示例代码文件

你可以从 [www.packtpub.com](http://www.packtpub.com) 账户下载本书的示例代码文件。如果你从其他地方购买了本书，可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给你。

你可以按照以下步骤下载代码文件：

1.  请在 [www.packtpub.com](http://www.packtpub.com/support) 上登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载 & 勘误。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载完成后，请确保使用最新版本的以下软件解压或提取文件：

+   WinRAR/7-Zip（适用于 Windows）

+   Zipeg/iZip/UnRarX（适用于 Mac）

+   7-Zip/PeaZip（适用于 Linux）

本书的代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/Learning-Jupyter-5-Second-Edition`](https://github.com/PacktPublishing/Learning-Jupyter-5-Second-Edition)。如果代码有更新，更新内容会直接发布在现有的 GitHub 仓库中。

我们还提供了来自丰富书籍和视频目录中的其他代码包，地址是 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快来看看吧！

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码字、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账户名。以下是一个示例：“默认文件名 `untitled1.txt` 是可编辑的。”

代码块如下设置：

```py
var mycell = Jupyter.notebook.get_selected_cell();
var cell_config = mycell.config;
var code_patch = {
    CodeCell:{
      cm_config:{indentUnit:2}
    }
 }
cell_config.update(code_patch)
```

任何命令行输入或输出的书写方式如下：

```py
jupyter trust /path/to/notebook.ipynb
```

**粗体**：表示新术语、重要词汇或屏幕上显示的词语。例如，菜单或对话框中的词汇会以这样的方式出现在文本中。以下是一个示例：“显示了三个选项卡：文件、运行和集群。”

警告或重要说明如下所示。

提示和技巧以此方式展示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请通过电子邮件发送至 `feedback@packtpub.com`，并在邮件主题中注明书名。如果您对本书的任何内容有疑问，请通过 `questions@packtpub.com` 与我们联系。

**勘误**：尽管我们已尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现任何错误，请联系我们。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并填写相关细节。

**盗版**：如果您在互联网上发现任何我们作品的非法复制版本，恳请您提供该网址或网站名称。请通过 `copyright@packtpub.com` 与我们联系，并附上该材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上具有专业知识，并且有意编写或为书籍贡献内容，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评审

请留下评审。在您阅读并使用本书后，为什么不在购买该书的网站上留下您的评论呢？潜在读者可以根据您的客观意见做出购买决策，Packt 也能了解您对我们产品的看法，而我们的作者也能看到您对其书籍的反馈。谢谢！

如需了解有关 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/)。
