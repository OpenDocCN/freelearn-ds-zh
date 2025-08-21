# 前言

数据科学是一个交叉学科，涵盖了机器学习、统计学和数据挖掘，旨在通过算法和统计分析从现有数据中获得新知识。本书将教您数据科学中分析数据的七种最重要方法。每一章首先以简单的概念解释其算法或分析，并通过一个简单的例子进行支持。随后，更多的例子和练习将帮助您构建并扩展对特定分析类型的知识。

# 本书适合谁阅读

本书适合那些熟悉 Python，并且有一定统计学背景的有志数据科学专业人士。对于那些目前正在实现一两个数据科学算法并希望进一步扩展技能的开发者，本书将非常有用。

# 如何最大化利用本书

为了最大化地利用本书，首先，您需要保持积极的态度来思考问题——许多新内容会在每章末尾的 *Problems* 部分的练习中呈现出来。然后，您还需要能够在您选择的操作系统上运行 Python 程序。作者在 Linux 操作系统上使用命令行运行了这些程序。

# 下载示例代码文件

您可以从您的 [www.packt.com](http://www.packt.com) 账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，将文件直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册 [www.packt.com](http://www.packt.com)。

1.  选择 SUPPORT 标签。

1.  点击“Code Downloads & Errata”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

文件下载后，请确保使用最新版本的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Data-Science-Algorithms-in-a-Week-Second-Edition`](https://github.com/PacktPublishing/Data-Science-Algorithms-in-a-Week-Second-Edition)。如果代码有更新，将会在现有的 GitHub 仓库中更新。

我们还有其他代码包，来自我们丰富的书籍和视频目录，您可以在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 查看。不要错过！

# 下载彩色图片

我们还提供了一个包含本书中使用的截图/图表彩色图片的 PDF 文件。您可以在这里下载：`www.packtpub.com/sites/default/files/downloads/9781789806076_ColorImages.pdf`。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：指示文本中的代码字词，数据库表名，文件夹名，文件名，文件扩展名，路径名，虚拟 URL，用户输入和 Twitter 句柄。例如："将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。"

代码块设置如下：

```py
def dic_key_count(dic, key):
if key is None:
return 0
if dic.get(key, None) is None:
return 0
else:
return int(dic[key])
```

当我们希望引起您对代码块的特定部分的注意时，相关的行或条目会用粗体标记：

```py
def construct_general_tree(verbose, heading, complete_data,
enquired_column, m):
available_columns = []
for col in range(0, len(heading)):
if col != enquired_column:
```

任何命令行输入或输出如下所示：

```py
$ python naive_bayes.py chess.csv
```

**粗体**：表示新术语，重要单词或屏幕上显示的单词。例如，菜单或对话框中的单词在文本中出现如此。例如："从管理面板中选择系统信息。"

警告或重要提示如下所示。

提示和技巧如下所示。

# 联系我们

我们随时欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在消息主题中提及书名，并发送邮件至`customercare@packtpub.com`与我们联系。

**勘误表**：尽管我们已尽一切努力确保内容的准确性，但错误确实会发生。如果您在本书中发现了错误，请向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书，单击勘误提交表单链接并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，请提供地址或网站名称给我们。请通过`copyright@packt.com`与我们联系，并附上链接到该资料的链接。

**如果您有兴趣成为作者**：如果您在某个专题上有专业知识，并且有意撰写或为书籍做贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。一旦您阅读并使用了这本书，为什么不在购买它的网站上留下评论呢？潜在的读者可以看到并使用您的公正意见来做购买决定，我们在 Packt 可以了解您对我们产品的看法，而我们的作者也可以看到您对他们的书的反馈。谢谢！

要获取有关 Packt 的更多信息，请访问[packt.com](http://www.packt.com/)。
