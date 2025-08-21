# 前言

统计学是数据科学领域任何任务的绝对必要先决条件，但对于进入数据科学领域的开发人员来说，可能也是最令人生畏的障碍。本书将带你踏上从几乎一无所知到能够熟练使用各种统计方法处理典型数据科学任务的统计之旅。

# 本书所需的内容

本书适合那些有数据开发背景的人员，他们有兴趣可能进入数据科学领域，并希望通过富有洞察力的程序和简单的解释获得关于统计学的简明信息。只需带上你的数据开发经验和开放的心态！

# 本书适用对象

本书适合那些有兴趣进入数据科学领域，并希望通过富有洞察力的程序和简单的解释获得关于统计学的简明信息的开发人员。

# 约定

在本书中，你会发现一些文本样式，用来区分不同类型的信息。以下是一些样式示例及其含义说明。

文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入以及 Twitter 用户名等如下所示：在统计学中，`boxplot` 是一种简单的方法，用于获取关于统计数据集的形状、变异性和中心（或中位数）信息，因此我们将使用 `boxplot` 来分析数据，以查看我们是否能识别出中位数 `Coin-in` 以及是否有任何离群值。

代码块设置如下：

```py
MyFile <-"C:/GammingData/SlotsResults.csv" 
MyData <- read.csv(file=MyFile, header=TRUE, sep=",") 
```

**新术语**和**重要词汇**以粗体显示。

警告或重要提示如下所示。

提示和技巧如下所示。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们你对这本书的看法——喜欢或不喜欢的部分。读者的反馈对我们非常重要，因为它帮助我们开发出你能够真正受益的书籍。如果你想给我们提供一般性反馈，只需发送电子邮件到`feedback@packtpub.com`，并在邮件主题中提及书名。如果你在某个领域拥有专业知识，且有兴趣撰写或为书籍提供贡献，请查看我们的作者指南：[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在你已经成为一本 Packt 书籍的骄傲拥有者，我们为你提供了一些帮助，以便你最大限度地利用你的购买。

# 下载示例代码

你可以从你的账户下载本书的示例代码文件：[`www.packtpub.com`](http://www.packtpub.com)。如果你是从其他地方购买了这本书，可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)，并注册以便将文件直接通过电子邮件发送给你。你可以按照以下步骤下载代码文件：

1.  使用你的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的 SUPPORT 标签上。

1.  点击代码下载与勘误。

1.  在搜索框中输入书名。

1.  选择你要下载代码文件的书籍。

1.  从下拉菜单中选择你购买这本书的地点。

1.  点击代码下载。

下载文件后，请确保使用最新版本的工具解压或提取文件夹：

+   Windows 版 WinRAR / 7-Zip

+   Mac 版 Zipeg / iZip / UnRarX

+   Linux 版 7-Zip / PeaZip

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Statistics-for-Data-Science`](https://github.com/PacktPublishing/Statistics-for-Data-Science)。我们还提供了其他来自我们丰富书籍和视频目录的代码包，网址为[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。赶紧去看看吧！

# 下载本书的彩色图片

我们还为你提供了一份 PDF 文件，其中包含本书中使用的屏幕截图/图表的彩色图片。这些彩色图片将帮助你更好地理解输出的变化。你可以从[`www.packtpub.com/sites/default/files/downloads/StatisticsforDataScience_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/StatisticsforDataScience_ColorImages.pdf)下载该文件。

# 勘误

尽管我们已尽最大努力确保内容的准确性，但错误仍然可能发生。如果你在我们的书籍中发现错误——可能是文本或代码中的错误——我们非常感谢你向我们报告。通过这样做，你不仅能帮助其他读者避免困扰，还能帮助我们改进后续版本的书籍。如果你发现了勘误，请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)报告，选择你的书籍，点击“勘误提交表单”链接，填写勘误详情。一旦勘误得到验证，你的提交将被接受，并且该勘误将被上传到我们的网站或加入该书名的现有勘误列表中。要查看之前提交的勘误，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索框中输入书名。所需的信息将显示在勘误部分。

# 盗版

互联网盗版一直是各类媒体中的一个持续性问题。我们在 Packt 非常重视保护我们的版权和许可证。如果你在互联网上发现任何形式的非法复制品，请立即提供相关网址或网站名称，以便我们采取相应措施。请通过`copyright@packtpub.com`与我们联系，并附上涉嫌盗版材料的链接。感谢你在保护我们的作者以及我们为你提供有价值内容的能力方面的帮助。

# 问题

如果你在使用本书的过程中遇到任何问题，可以通过`questions@packtpub.com`与我们联系，我们将尽力解决问题。
