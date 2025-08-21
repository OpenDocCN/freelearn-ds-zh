# 前言

数据分析是我们许多人之前可能甚至不知道自己已经做过的事情。这是收集和分析信息的基本艺术，以适应各种目的——从视觉检查到机器学习技术。通过数据分析，我们可以从数字领域四处散布的信息中获取意义。它使我们能够解决最奇特的问题，甚至在此过程中提出新问题。

Haskell 作为我们进行强大数据分析的桥梁。对于一些人来说，Haskell 是一种保留给学术界和工业界最精英研究人员的编程语言。然而，我们看到它正吸引着全球开源开发者中最快增长的文化之一。Haskell 的增长表明，人们正在发现其优美的函数式纯净性、强大的类型安全性和卓越的表达力。翻开本书的页面，看到这一切的实际运用。

*Haskell 数据分析烹饪书*不仅仅是计算机领域两个迷人主题的融合。它还是 Haskell 编程语言的学习工具，以及简单数据分析实践的介绍。将其视为算法和代码片段的瑞士军刀。尝试每天一个配方，就像大脑的武术训练。从催化的示例中轻松翻阅本书，获得创意灵感。最重要的是，深入探索 Haskell 中的数据分析领域。

当然，如果没有 Lonku（[`lonku.tumblr.com`](http://lonku.tumblr.com)）提供的精彩章节插图和 Packt Publishing 提供的有益布局和编辑支持，这一切都是不可能的。

# 本书内容涵盖

第一章, *数据的探寻*，识别了从各种外部来源（如 CSV、JSON、XML、HTML、MongoDB 和 SQLite）读取数据的核心方法。

第二章, *完整性与检验*，解释了通过关于修剪空白、词法分析和正则表达式匹配的配方清理数据的重要性。

第三章, *单词的科学*，介绍了常见的字符串操作算法，包括基数转换、子串匹配和计算编辑距离。

第四章, *数据哈希*，涵盖了诸如 MD5、SHA256、GeoHashing 和感知哈希等重要的哈希函数。

第五章, *树的舞蹈*，通过包括树遍历、平衡树和 Huffman 编码等示例，建立对树数据结构的理解。

第六章, *图基础*，展示了用于图网络的基础算法，如图遍历、可视化和最大团检测。

第七章，*统计与分析*，开始了对重要数据分析技术的探索，其中包括回归算法、贝叶斯网络和神经网络。

第八章，*聚类与分类*，涉及典型的分析方法，包括 k-means 聚类、层次聚类、构建决策树以及实现 k 最近邻分类器。

第九章，*并行与并发设计*，介绍了 Haskell 中的高级主题，如分叉 I/O 操作、并行映射列表和性能基准测试。

第十章，*实时数据*，包含来自 Twitter、Internet Relay Chat（IRC）和套接字的流式数据交互。

第十一章，*可视化数据*，涉及多种绘制图表的方法，包括折线图、条形图、散点图和 `D3.js` 可视化。

第十二章，*导出与展示*，以一系列将数据导出为 CSV、JSON、HTML、MongoDB 和 SQLite 的算法结束本书。

# 本书所需内容

+   首先，您需要一个支持 Haskell 平台的操作系统，如 Linux、Windows 或 Mac OS X。

+   您必须安装 Glasgow Haskell Compiler 7.6 或更高版本及 Cabal，这两者都可以从 [`www.haskell.org/platform`](http://www.haskell.org/platform) 获取。

+   您可以在 GitHub 上获取每个食谱的配套源代码，网址为 [`github.com/BinRoot/Haskell-Data-Analysis-Cookbook`](https://github.com/BinRoot/Haskell-Data-Analysis-Cookbook)。

# 本书适合谁阅读

+   对那些已经开始尝试使用 Haskell 并希望通过有趣的示例来启动新项目的人来说，这本书是不可或缺的。

+   对于刚接触 Haskell 的数据分析师，本书可作为数据建模问题的函数式方法参考。

+   对于初学 Haskell 语言和数据分析的读者，本书提供了最大的学习潜力，可以帮助您掌握书中涉及的新话题。

# 约定

在本书中，您将看到多种文本样式，用以区分不同类型的信息。以下是一些这些样式的示例，并附有其含义的解释。

文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名将如下所示：“将 `readString` 函数应用于输入，并获取所有日期文档。”

一块代码块如下所示：

```py
main :: IO () 
main = do 
  input <- readFile "input.txt"
  print input
```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
main :: IO () 
main = do 
  input <- readFile "input.txt"
  print input
```

任何命令行输入或输出都将如下所示：

```py
$ runhaskell Main.hs

```

**新术语**和**重要单词**以粗体显示。你在屏幕上、菜单或对话框中看到的词语，通常以这种形式出现在文本中：“在**下载**部分，下载 cabal 源代码包。”

### 注意

警告或重要的注意事项以框框的形式呈现。

### 提示

提示和技巧以这种形式出现。

# 读者反馈

我们始终欢迎来自读者的反馈。让我们知道你对这本书的看法——你喜欢什么，或者可能不喜欢什么。读者反馈对我们开发能够让你真正受益的书籍至关重要。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并通过邮件主题注明书籍名称。

如果你在某个领域有专业知识，并且有兴趣撰写或参与撰写一本书，查看我们的作者指南：[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在你已经是一本 Packt 图书的骄傲拥有者，我们为你提供了一些帮助，以便你能够最大限度地从你的购买中受益。

## 下载示例代码

你可以从你在[`www.packtpub.com`](http://www.packtpub.com)的账户中下载你购买的所有 Packt 图书的示例代码文件。如果你是从其他地方购买的这本书，可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，将文件直接通过电子邮件发送给你。此外，我们强烈建议你从 GitHub 获取所有源代码，网址为[`github.com/BinRoot/Haskell-Data-Analysis-Cookbook`](https://github.com/BinRoot/Haskell-Data-Analysis-Cookbook)。

## 勘误表

尽管我们已尽一切努力确保内容的准确性，但错误难免发生。如果你在我们的书籍中发现错误——可能是文本错误或代码错误——我们将非常感激你报告给我们。通过这样做，你可以帮助其他读者避免困扰，并帮助我们改进后续版本的书籍。如果你发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择你的书籍，点击**勘误**提交**表单**链接，并输入勘误的详细信息。一旦你的勘误被验证，提交将被接受，并且勘误将被上传到我们的网站，或添加到该书勘误列表中。任何现有的勘误都可以通过选择你书籍标题，访问[`www.packtpub.com/support`](http://www.packtpub.com/support)查看。代码修订也可以在附带的 GitHub 仓库进行修改，仓库地址为[`github.com/BinRoot/Haskell-Data-Analysis-Cookbook`](https://github.com/BinRoot/Haskell-Data-Analysis-Cookbook)。

## 盗版

互联网版权材料的盗版问题在各类媒体中普遍存在。我们在 Packt 非常重视版权和许可的保护。如果您在互联网遇到我们作品的任何非法复制，无论其形式如何，请立即向我们提供该位置地址或网站名称，以便我们采取补救措施。

请通过 `<copyright@packtpub.com>` 与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您的帮助，以保护我们的作者，以及我们为您提供有价值内容的能力。

## 问题

如果您在书籍的任何方面遇到问题，可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决。
