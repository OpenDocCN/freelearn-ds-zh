# 前言

2010 年代的数据科学家和机器学习工程师主要进行静态分析。我们创建文档来传达决策，填充了图表和指标，展示我们的发现或我们创建的模型。创建完整的 Web 应用，让用户与分析互动，至少可以说是繁琐的！这时，Streamlit 应运而生，它是一个为数据工作者量身打造的 Python 库，旨在每一步都为用户提供便捷的 Web 应用创建体验。

Streamlit 缩短了数据驱动的 web 应用开发时间，让数据科学家可以在几小时内使用 Python 创建 Web 应用原型，而不再是几天。

本书采用实践式学习方式，帮助你掌握让你迅速上手 Streamlit 的技巧和窍门。你将从创建一个基本应用的 Streamlit 基础开始，并逐步建立这个基础，生成高质量的图形数据可视化，并测试机器学习模型。随着你逐步深入，你将学习如何通过个人及工作相关的数据驱动型 Web 应用实例，并了解一些更复杂的话题，如使用 Streamlit 组件、美化应用，以及快速部署你的新应用。

# 本书的适用人群

本书适合数据科学家、机器学习工程师或爱好者，尤其是那些想要使用 Streamlit 创建 Web 应用的人。无论你是初级数据科学家，想要部署你的第一个机器学习项目，以提升简历，还是资深数据科学家，想通过动态分析说服同事，本书都适合你！

# 本书内容概览

*第一章*，*Streamlit 简介*，通过创建你的第一个应用来教授 Streamlit 的基本知识。

*第二章*，*上传、下载和操作数据*，探讨了数据；数据应用需要数据！你将学习如何在生产应用中高效、有效地使用数据。

*第三章*，*数据可视化*，讲解如何在 Streamlit 应用中使用你最喜欢的 Python 可视化库。无需学习新的可视化框架！

*第四章*，*使用 Streamlit 进行机器学习与人工智能*，讲解了机器学习。曾经想过在几小时内将你新开发的复杂机器学习模型部署到用户可用的应用中吗？在这里，你将获得深入的示例和技巧，包括与 Hugging Face 和 OpenAI 模型的合作。

*第五章*，*在 Streamlit 社区云上部署 Streamlit*，讲解了 Streamlit 自带的一键部署功能。你将在这里学到如何消除部署过程中的摩擦！

*第六章*，*美化 Streamlit 应用*，介绍了 Streamlit 充满的各种功能，帮助你打造华丽的 Web 应用。你将在这一章学到所有的小技巧和窍门。

*第七章*，*探索 Streamlit 组件*，教你如何通过开源集成（即 Streamlit 组件）利用 Streamlit 周围蓬勃发展的开发者生态系统。就像 LEGO，一样更强大。

*第八章*，*使用 Hugging Face 和 Heroku 部署 Streamlit 应用*，教你如何使用 Hugging Face 和 Heroku 部署 Streamlit 应用，作为 Streamlit Community Cloud 的替代方案。

*第九章*，*连接数据库*，将帮助你将生产数据库中的数据添加到 Streamlit 应用中，从而扩展你能制作的应用种类。

*第十章*，*使用 Streamlit 改进求职申请*，将帮助你通过 Streamlit 应用向雇主展示你的数据科学能力，涵盖从简历制作应用到面试带回家任务的应用。

*第十一章*，*数据项目 – 在 Streamlit 中进行项目原型制作*，介绍了如何为 Streamlit 社区和其他用户制作应用，这既有趣又富有教育意义。你将通过一些项目示例，学习如何开始自己的项目。

*第十二章*，*Streamlit 高级用户*，提供了更多关于 Streamlit 的信息，Streamlit 作为一个年轻的库，已经被广泛使用。通过对 Streamlit 创始人、数据科学家、分析师和工程师的深入访谈，向最优秀的人学习。

# 致谢

本书的完成离不开我的技术审阅者 Chanin Nantasenamat 的帮助。你可以在 X/Twitter 上找到他，链接为 [`twitter.com/thedataprof`](https://twitter.com/thedataprof)，也可以在 YouTube 上找到他，链接为 [`www.youtube.com/dataprofessor`](https://www.youtube.com/dataprofessor)。所有错误由我负责，但所有避免的错误都归功于他！

# 最大化利用本书

本书假设你至少是 Python 初学者，这意味着你已经熟悉基本的 Python 语法，并且之前接受过 Python 的教程或课程。本书同样适合对数据科学感兴趣的读者，涵盖统计学和机器学习等主题，但并不要求具备数据科学背景。如果你知道如何创建列表、定义变量，并且写过 `for` 循环，那么你已经具备足够的 Python 知识来开始了！

如果你正在使用本书的数字版，建议你自己输入代码，或者通过本书的 GitHub 仓库获取代码（下节将提供链接）。这样做可以帮助你避免因复制和粘贴代码而产生的潜在错误。

# 下载示例代码文件

你可以从 GitHub 下载本书的示例代码文件，链接为 [`github.com/tylerjrichards/Streamlit-for-Data-Science`](https://github.com/tylerjrichards/Streamlit-for-Data-Science)。如果代码有更新，它会在这些 GitHub 仓库中同步更新。

我们还提供了来自我们丰富书籍和视频目录中的其他代码包，访问链接 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。快去看看吧！

# 下载彩色图片

我们还提供一个包含本书中截图和图表彩色图像的 PDF 文件。您可以在此下载：[`packt.link/6dHPZ`](https://packt.link/6dHPZ)。

# 使用的规范

本书中使用了几个文本规范：

`文本中的代码`：表示文本中的代码词语、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。以下是一个例子：“…它的格式将是`ec2-10-857-84-485.compute-1.amazonaws.com`。我编造了这些数字，但你的应该接近这个格式。”

一块代码的格式如下：

```py
import pandas as pd
penguin_df = pd.read_csv('penguins.csv')
print(penguin_df.head()) 
```

任何命令行输入或输出都以以下形式编写：

```py
git add .
git commit -m 'added heroku files'
git push 
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词会以**粗体**显示。以下是一个例子：“我们将使用**Amazon Elastic Compute Cloud**，简称**Amazon EC2**。”

提示或重要说明

以这种形式出现。

# 与我们联系

我们始终欢迎读者的反馈。

**一般反馈**：通过电子邮件联系 `feedback@packtpub.com`，并在邮件主题中提及书名。如果您对本书的任何内容有问题，请通过 `questions@packtpub.com` 与我们联系。

**勘误**：虽然我们已经尽力确保内容的准确性，但错误仍然会发生。如果您在本书中发现错误，感谢您将其报告给我们。请访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，点击**提交勘误**，并填写表格。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，感谢您提供该地址或网站名称。请通过 `copyright@packtpub.com` 联系我们，并附上相关内容的链接。

**如果您有兴趣成为作者**：如果您在某个主题上有专业知识，并且有兴趣撰写或贡献书籍内容，请访问 [`authors.packtpub.com`](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读完《Streamlit for Data Science, Second Edition》，我们非常希望听到您的想法！请[点击这里直接进入亚马逊评价页面](https://packt.link/r/180324822X)并分享您的反馈。

您的评价对我们和技术社区非常重要，能帮助我们确保提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

您喜欢随时随地阅读，但又无法携带印刷版书籍吗？

您购买的电子书是否无法与您的设备兼容？

不用担心，现在每本 Packt 书籍都附带免费的无 DRM 版 PDF。

在任何设备上随时随地阅读，搜索、复制并粘贴您最喜欢的技术书籍中的代码，直接导入到您的应用程序中。

优惠不仅如此，你还可以每天通过电子邮件独享折扣、新闻通讯以及精彩的免费内容

按照以下简单步骤获取优惠：

1.  扫描二维码或访问以下链接

![](img/B18444_Free_PDF.png)

[`packt.link/free-ebook/9781803248226`](https://packt.link/free-ebook/9781803248226)

1.  提交你的购买凭证

1.  就这些！我们会将你的免费 PDF 和其他福利直接发送到你的邮箱
