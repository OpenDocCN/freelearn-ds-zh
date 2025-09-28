# 前言

您正在阅读的书的标题是*时间序列索引*，这应该暗示了其内容。

本书讨论和探索的时间序列索引称为 iSAX。iSAX 被认为是时间序列中最好的索引之一，这也是选择它的主要原因。除了将 iSAX 和 SAX 表示实现为 Python 3 包之外，本书还展示了如何在子序列级别处理时间序列以及如何理解学术论文中呈现的信息。

但本书并未止步于此，因为它提供了用于更好地了解您的时序数据的 Python 脚本，以及用于可视化时序数据和 iSAX 索引的代码，以便更好地理解数据以及特定 iSAX 索引的结构。

# 本书面向的对象

本书面向的开发者、研究人员和任何级别的大学学生，他们希望在后续级别处理时间序列，并在过程中使用现代时间序列索引。

虽然展示的代码是用 Python 3 编写的，但一旦您理解了代码背后的思想和概念，这些包和脚本可以轻松地移植到任何其他现代编程语言，如 Rust、Swift、Go、C、Kotlin、Java、JavaScript 等。

# 本书涵盖的内容

*第一章*，*时间序列与所需 Python 知识简介*，主要介绍了您需要了解的基础知识，包括时间序列的重要性以及如何设置合适的 Python 环境来运行本书的代码并进行时间序列实验。

*第二章*，*实现 SAX*，解释了 SAX 及其表示，并展示了用于计算时间序列或子序列的 SAX 表示的 Python 代码。它还展示了用于计算可以提供时间序列更高概述的统计量的 Python 脚本，并绘制时间序列数据的直方图。

*第三章*，*iSAX – 所需理论*，介绍了 iSAX 索引构建和使用背后的理论，并展示了如何通过大量可视化逐步手动构建一个小 iSAX 索引。

*第四章*，*iSAX - 实现*，介绍了开发一个用于创建适合内存的 iSAX 索引的 Python 包，并展示了如何将这个 Python 包付诸实践的 Python 脚本。

*第五章*，*连接和比较 iSAX 索引*，展示了如何使用`isax`包创建的 iSAX 索引，以及如何连接和比较它们。本章最后讨论了测试 Python 代码的主题。最后，我们展示了如何为`isax`包编写一些简单的测试。

*第六章*，*可视化 iSAX 索引*，主要介绍了如何使用 JavaScript 编程语言和 JSON 格式通过各种类型的可视化来可视化 iSAX 索引。

*第七章*，*使用 iSAX 近似 MPdist*，介绍了如何使用 iSAX 索引来近似计算两个时间序列之间的矩阵轮廓向量和 MPdist 距离。

*第八章*，*结论和下一步行动*，如果您对时间序列或数据库非常感兴趣，它将提供有关下一步要查找什么和哪里的指导，建议研究经典书籍和研究论文。

# 为了最大限度地利用本书

本书需要一台装有相对较新 Python 3 安装和本地安装 Python 包能力的 UNIX 机器。这包括运行 macOS 和 Linux 最新版本的任何机器。所有代码都在 Microsoft Windows 机器上进行了测试。

我们建议您使用用于 Python 包、依赖和环境管理的软件来拥有稳定的 Python 3 环境。我们使用 Anaconda，但任何类似的工具都可以正常工作。

最后，如果您真的想最大限度地利用本书，那么您需要尽可能多地与提供的 Python 代码进行实验，创建自己的 iSAX 索引和可视化，也许还可以将代码移植到不同的编程语言中。

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件：[`github.com/PacktPublishing/Time-Series-Indexing`](https://github.com/PacktPublishing/Time-Series-Indexing)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图和图表彩色图像的 PDF 文件。您可以从这里下载：[`packt.link/Pzq1j`](https://packt.link/Pzq1j)

# 使用的约定

本书中使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“我们将使用以下滑动窗口大小进行实验：`16`、`256`、`1024`、`4096`和`16384`。”

代码块应如下设置：

```py
def query(ISAX, q):
    global totalQueries
    totalQueries = totalQueries + 1
    Accesses = 0
    # Create TS Node
```

当我们希望将您的注意力引到代码块的一个特定部分时，相关的行或项目将以粗体显示：

```py
    # Query iSAX for TS1
    for idx in range(0, len(ts1)-windowSize+1):
        currentQuery = ts1[idx:idx+windowSize]
        found, ac = query(i1, currentQuery)
        if found == False:
            print("This cannot be happening!")
            return
```

任何命令行输入或输出都应如下编写：

```py
$ ./accessSplit.py -s 8 -c 32 -t 500 -w 16384 500k.gz
Max Cardinality: 32 Segments: 8 Sliding Window: 16384 Threshold: 500 Default Promotion: False
OVERFLOW: 01111_10000_10000_01111_10000_01111_10000_01111
Number of splits: 6996
Number of subsequence accesses: 19201125
```

小贴士或重要注意事项

看起来像这样。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请通过电子邮件 customercare@packtpub.com 与我们联系，并在邮件主题中提及书名。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，错误仍然可能发生。如果你在这本书中发现了错误，我们非常感谢你能向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) 并填写表格。

**盗版**: 如果你在网上遇到任何形式的我们作品的非法副本，我们非常感谢你能提供位置地址或网站名称。请通过 mailto:copyright@packtpub.com 与我们联系，并提供材料的链接。

**如果你有兴趣成为作者**: 如果你在一个领域有专业知识，并且你感兴趣的是撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 分享你的想法

一旦你阅读了 *时间序列索引*，我们非常乐意听到你的想法！请[点击此处直接跳转到该书的亚马逊评论页面](https://packt.link/r/1838821953)并分享你的反馈。

你的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。

# 下载这本书的免费 PDF 副本

感谢你购买这本书！

你喜欢在路上阅读，但又无法携带你的印刷书籍到处走吗？

你的电子书购买是否与你的选择设备不兼容？

别担心，现在每购买一本 Packt 书籍，你都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何设备上阅读。直接从你最喜欢的技术书籍中搜索、复制和粘贴代码到你的应用程序中。

优惠不会就此停止，你还可以获得独家折扣、时事通讯和每日收件箱中的精彩免费内容。

按照以下简单步骤获取这些好处：

1.  扫描下面的二维码或访问以下链接

![](img/B14769_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781838821951`](https://packt.link/free-ebook/9781838821951)

1.  提交你的购买证明

1.  就这些！我们将直接将你的免费 PDF 和其他好处发送到你的电子邮件。
