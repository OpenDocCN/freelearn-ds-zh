# 前言

随着时间的推移，Python 已经成为空间分析的首选编程语言，产生了许多用于读取、转换、分析和可视化空间数据的包。在如此多的包可用的情况下，为学生和经验丰富的专业人士创建一本包含 Python 3 必需地理空间 Python 库的参考书是有意义的。

这本书也出现在一个激动人心的时刻：新技术正在改变人们处理地理空间数据的方式——物联网、机器学习和数据科学是地理空间数据不断被使用的领域。这解释了为什么包含新的 Python 库，如 CARTOframes 和 MapboxGL，以及 Jupyter 也被包括在内，以探索这些新趋势。同时，基于网页和云的 GIS 正在越来越多地成为新的标准。这在本书第二部分章节中得到了体现，其中介绍了交互式地理空间网络地图和 REST API。

这些较新的库与许多在多年中变得至关重要的旧库相结合，至今仍然非常受欢迎，如 Shapely、Rasterio 和 GeoPandas。对于新进入这个领域的人来说，将给出对流行库的适当介绍，将它们置于适当的视角，并通过使用真实世界数据的代码示例比较它们的语法。 

最后，这本书标志着从 Python 2 到 3.x 的过渡。本书涵盖的所有库都是用 Python 3.x 编写的，以便读者可以使用 Jupyter Notebook 访问它们，这也是本书推荐的 Python 编码环境。

# 本书面向的对象

这本书适合任何与位置信息以及 Python 工作的人。学生、开发人员和地理空间专业人士都可以使用这本参考书，因为它涵盖了 GIS 数据管理、分析技术和用 Python 3 构建的代码库。

# 本书涵盖的内容

第一章，*包安装和管理*，解释了如何安装和管理本书中使用的代码库。

第二章，*地理空间代码库简介*，涵盖了用于处理和分析地理空间数据的主要代码库。

第三章，*地理空间数据库简介*，介绍了用于数据存储和分析的地理空间数据库。

第四章，*数据类型、存储和转换*，专注于 GIS 中存在的许多不同数据类型（矢量数据类型和栅格数据类型）。

第五章，*矢量数据分析*，涵盖了 Shapely、OGR 和 GeoPandas 等 Python 库，这些库用于分析和处理矢量数据。

第六章，*栅格数据处理*，探讨了使用 GDAL 和 Rasterio 处理栅格数据集以执行地理空间分析。

第七章，*使用地理数据库进行地理处理*，向读者展示了如何使用包含空间列的数据库表执行空间 SQL 地理处理。

第八章，*自动化 QGIS 分析*，教导读者如何使用 PyQGIS 在 QGIS 映射套件内自动化分析。

第九章，*ArcGIS API for Python 和 ArcGIS Online*，介绍了 ArcGIS API for Python，它允许用户使用 Python 3 与 Esri 的云平台 ArcGIS Online 进行交互。

第十章，*使用 GPU 数据库进行地理处理*，介绍了使用 Python 工具与基于云的数据交互以搜索和处理数据。

第十一章，*Flask 和 GeoAlchemy2*，描述了如何使用 Flask Python 网络框架和 GeoAlchemy ORM 来执行空间数据查询。

第十二章，*GeoDjango*，介绍了使用 Django Python 网络框架和 GeoDjango ORM 来执行空间数据查询。

第十三章，*地理空间 REST API*，教导读者如何为地理空间数据创建 REST API。

第十四章，*云地理数据库分析和可视化*，向读者介绍了 CARTOframes Python 包，使 Carto 地图、分析和数据服务能够集成到数据科学工作流程中。

第十五章，*自动化云制图*，介绍了 Jupyter Notebooks 的新位置数据可视化库。

第十六章，*使用 Hadoop 进行 Python 地理处理*，解释了如何使用分布式服务器进行地理空间分析。

# 为了充分利用本书

由于本书涵盖 Python，因此假设读者对 Python 语言有基本的了解，可以安装 Python 库，并且知道如何编写和运行 Python 脚本。至于额外的知识，前六章可以很容易地理解，无需任何地理空间数据分析的先验知识。然而，后面的章节假设读者对空间数据库、大数据平台、数据科学、Web API 和 Python 网络框架有一定的了解。

# 下载示例代码文件

您可以从 [www.packtpub.com](http://www.packtpub.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

按照以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择 SUPPORT 选项卡。

1.  点击代码下载与勘误。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本解压缩或提取文件夹：

+   Windows 版的 WinRAR/7-Zip

+   Mac 版的 Zipeg/iZip/UnRarX

+   Linux 版的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Mastering-Geospatial-Analysis-with-Python`](https://github.com/PacktPublishing/Mastering-Geospatial-Analysis-with-Python)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包可供使用，请访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。你可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/MasteringGeospatialAnalysiswithPython_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/MasteringGeospatialAnalysiswithPython_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“选择一个文件夹并保存密钥，现在它将有一个`.ppk`文件扩展名。”

代码块设置如下：

```py
cursor.execute("SELECT * from art_pieces")
data=cursor.fetchall()
data
```

当我们希望将你的注意力引到代码块的一个特定部分时，相关的行或项目将被设置为粗体：

```py
from pymapd import connect
connection = connect(user="mapd", password= "{password}", 
     host="{my.host.com}", dbname="mapd")
cursor = connection.cursor()
sql_statement = """SELECT name FROM county;"""
cursor.execute(sql_statement)
```

任何命令行输入或输出都按照以下方式编写：

```py
conda install -c conda-forge geos
```

**粗体**: 表示新术语、重要单词或你在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“要从 EC2 仪表板生成密钥对，请在向下滚动后从左侧面板的 NETWORK & SECURITY 组中选择 Key Pairs。”

警告或重要提示看起来像这样。

技巧和窍门看起来像这样。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**: 请通过`feedback@packtpub.com`发送邮件，并在邮件主题中提及书名。如果你对本书的任何方面有疑问，请通过`questions@packtpub.com`发送邮件给我们。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将不胜感激，如果你能向我们报告这个错误。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择你的书，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供给我们地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
