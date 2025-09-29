# 地理空间 Python 工具箱

本书的前三章涵盖了地理空间分析的历史、分析师使用的地理空间数据类型以及地理空间行业中的主要软件和库。我们在某些地方使用了一些简单的 Python 示例来说明某些观点，但我们主要关注地理空间分析领域，而不考虑任何特定技术。从现在开始，我们将使用 Python 来征服地理空间分析，并将继续使用这种方法完成本书的其余部分。本章解释了您工具箱中所需的软件，以便在地理空间领域做几乎所有您想做的事情。

我们将发现用于访问第二章“学习地理空间数据”中发现的矢量数据和栅格数据不同类型的 Python 库。其中一些库是纯 Python，还有一些是我们第三章“地理空间技术景观”中查看的不同软件包的绑定。

在本章中，我们将涵盖以下主题：

+   安装第三方 Python 模块

+   Python 虚拟环境

+   Conda

+   Docker

+   用于获取数据的 Python 网络库

+   Python 基于标签的解析器

+   Python JSON 库

+   OGR

+   PyShp

+   DBFPY

+   Shapely

+   GDAL

+   Fiona

+   NumPy

+   GeoPandas

+   Python 图像库（PIL）

+   PNGCanvas

+   ReportLab

+   GeoPDF

+   Python NetCDF 库

+   Python HDF 库

+   OSMnx

+   空间索引库

+   Jupyter

+   Conda

在可能的情况下，我们将检查纯 Python 解决方案。Python 是一种非常强大的编程语言，但某些操作，尤其是在遥感领域，计算量过于庞大，因此在使用纯 Python 或其他解释型语言时不太实用。幸运的是，可以通过 Python 以某种方式解决地理空间分析的各个方面，即使它绑定到高度高效的 C/C++/其他编译语言库。

我们将避免使用覆盖地理空间分析以外的其他领域的广泛科学库，以使解决方案尽可能简单。使用 Python 进行地理空间分析有许多原因，但其中最强有力的论据之一是其可移植性。

此外，Python 已被移植到 Java 作为 Jython 发行版，以及到.NET **公共语言运行时**（**CLR**）作为 IronPython。Python 还有如 Stackless Python 这样的版本，适用于大量并发程序。还有专为在集群计算机上运行分布式处理而设计的 Python 版本。Python 还可在许多托管应用程序服务器上使用，这些服务器不允许您安装自定义可执行文件，例如具有 Python API 的 Google App Engine 平台。

# 技术要求

+   Python 3.6 或更高版本

+   RAM：最小 6 GB（Windows），推荐 8 GB（macOS），建议 8 GB

+   存储：最小 7200 RPM SATA，可用空间 20 GB；推荐 SSD，可用空间 40 GB

+   处理器：最小 Intel Core i3 2.5 GHz；推荐 Intel Core i5

# 安装第三方 Python 模块

使用纯 Python（使用标准库）编写的模块将在 Python 网站提到的 20 个平台中的任何一个上运行。每次你添加一个依赖于绑定到其他语言外部库的第三方模块时，你都会降低 Python 的固有可移植性。你还在代码中添加了另一层复杂性，通过添加另一种语言来彻底改变代码。纯 Python 保持简单。此外，Python 对外部库的绑定通常是由自动或半自动生成的。

这些自动生成的绑定非常通用且晦涩，它们只是通过使用该 API 的方法名将 Python 连接到 C/C++ API，而不是遵循 Python 的最佳实践。当然，也有一些值得注意的例外，这些例外是由项目需求驱动的，可能包括速度、独特的库功能或经常更新的库，在这些库中，自动生成的接口更可取。

我们将在 Python 的标准库中包含的模块和必须安装的模块之间做出区分。在 Python 中，`words` 模块和库是通用的。要安装库，你可以从 **Python 包索引（PyPI**） 获取，或者在许多地理空间模块的情况下，下载一个专门的安装程序。

PyPI 作为官方的软件仓库，提供了一些易于使用的设置程序，简化了包的安装。你可以使用 `easy_install` 程序，它在 Windows 上特别有用，或者使用在 Linux 和 Unix 系统上更常见的 `pip` 程序。一旦安装，你就可以通过运行以下代码来安装第三方包：

```py
easy_install <package name>
```

要安装 `pip`，请运行以下代码：

```py
pip install <package name>
```

本书将提供不在 PyPI 上可用的开源软件包的链接和安装说明。你可以通过下载 Python 源代码并将其放入当前工作目录，或者将其放入 Python 的 `site-packages` 目录中来手动安装第三方 Python 模块。这两个目录在尝试导入模块时都可用在 Python 的搜索路径中。如果你将模块放入当前工作目录，它只会在你从该目录启动 Python 时可用。

如果您将其放在`site-packages`目录中，每次启动 Python 时它都将可用。`site-packages`目录专门用于第三方模块。为了定位您安装的`site-packages`目录，您需要询问 Python 的`sys`模块。`sys`模块有一个`path`属性，其中包含 Python 搜索路径中的所有目录。`site-packages`目录应该是最后一个。您可以通过指定索引`-1`来定位它，如下面的代码所示：

```py
>>> import sys
>>> sys.path[-1]
'C:\\Python34\\lib\\site-packages'
```

如果该调用没有返回`site-packages`路径，只需查看整个列表以定位它，如下面的代码所示：

```py
>> sys.path
['', 'C:\\WINDOWS\\system32\\python34.zip', 'C:\\Python34\\DLLs',
'C:\\Python34\\lib', 'C:\\Python34\\lib\\plat-win
', 'C:\\Python34\\lib\\lib-tk', 'C:\\Python34',
'C:\\Python34\\lib\\site-packages']
```

这些安装方法将在本书的其余部分中使用。您可以在[`python.org/download/`](http://python.org/download/)找到最新的 Python 版本、您平台安装的源代码以及编译说明。

Python 的`virtualenv`模块允许您轻松地为特定项目创建一个隔离的 Python 副本，而不会影响您的主 Python 安装或其他项目。使用此模块，您可以拥有具有相同库的不同版本的不同项目。一旦您有一个工作代码库，您就可以将其与您使用的模块或甚至 Python 本身的变化隔离开来。`virtualenv`模块简单易用，可以用于本书中的任何示例；然而，关于其使用的明确说明并未包含。

要开始使用`virtualenv`，请遵循以下简单指南：[`docs.python-guide.org/en/latest/dev/virtualenvs/`](http://docs.python-guide.org/en/latest/dev/virtualenvs/)。

# Python 虚拟环境

Python 地理空间分析需要我们使用许多具有许多依赖关系的模块。这些模块通常使用特定版本的 C 或 C++库相互构建。当您向系统中添加 Python 模块时，经常会遇到版本冲突。有时，当您升级特定模块时，由于 API 的变化，它可能会破坏您现有的 Python 程序——或者您可能同时运行 Python 2 和 Python 3 以利用为每个版本编写的库。您需要的是一种安全安装新模块的方法，而不会破坏工作系统或代码。解决这个问题的方法是使用`virtualenv`模块的 Python 虚拟环境。

Python 的`virtualenv`模块为每个项目创建隔离的、独立的 Python 环境，这样您就可以避免冲突的模块污染您的主 Python 安装。您可以通过激活或停用特定环境来打开或关闭该环境。`virtualenv`模块在效率上很高，因为它在创建环境时实际上并不复制您整个系统 Python 安装。让我们开始吧：

1.  安装`virtualenv`就像运行以下代码一样简单：

```py
pip install virtualenv
```

1.  然后，为您的虚拟 Python 环境创建一个目录。命名它 whatever you want：

```py
mkdir geospatial_projects
```

1.  现在，您可以使用以下命令创建您的第一个虚拟环境：

```py
virtualenv geospatial_projects/project1
```

1.  然后，在输入以下命令后，你可以激活该环境：

```py
source geospatial_projects/project1/bin/activate
```

1.  现在，当你在这个目录中运行任何 Python 命令时，它将使用隔离的虚拟环境。当你完成时，你可以使用以下简单的命令来停用该环境：

```py
deactivate
```

这就是安装、激活以供使用以及停用 `virtualenv` 模块的方法。然而，你还需要了解另一个环境。我们将在下一节中检查它。

# Conda

在这里也值得提一下 Conda，它是一个开源的、跨平台的包管理系统，也可以创建和管理类似于 `virtualenv` 的环境。Conda 使得安装复杂的包变得容易，包括地理空间包。它还支持 Python 之外的其他语言，包括 R、Node.js 和 Java。

Conda 可在此处找到：[`docs.conda.io/en/latest/`](https://docs.conda.io/en/latest/)。

现在，让我们来看看如何安装 GDAL，这样我们就可以开始处理地理空间数据了。

# 安装 GDAL

**地理空间数据抽象库**（**GDAL**），包括 OGR，对于本书中的许多示例至关重要，也是更复杂的 Python 设置之一。因此，我们将在这里单独讨论它。最新的 GDAL 绑定可在 PyPI 上找到；然而，由于 GDAL 库需要额外的资源，安装需要额外的步骤。

有三种方法可以安装 GDAL 以用于 Python。你可以使用其中任何一种：

+   从源代码编译它。

+   作为更大软件包的一部分安装它。

+   安装二进制发行版，然后安装 Python 绑定。

如果你也有编译 C 库以及所需编译软件的经验，那么第一个选项会给你最大的控制权。然而，如果你只是想尽快开始，那么这个选项并不推荐，因为即使是经验丰富的软件开发者也会发现编译 GDAL 和相关的 Python 绑定具有挑战性。在主要平台上的 GDAL 编译说明可以在 [`trac.osgeo.org/gdal/wiki/BuildHints`](http://trac.osgeo.org/gdal/wiki/BuildHints) 找到；PyPI GDAL 页面上也有基本的构建说明；请查看 [`pypi.python.org/pypi/GDAL`](https://pypi.python.org/pypi/GDAL)。

第二个选项无疑是最快和最简单的方法。**开源地理空间基金会**（**OSGeo**）分发了一个名为 OSGeo4W 的安装程序，只需点击一下按钮即可在 Windows 上安装所有顶级开源地理空间包。OSGeo4W 可在 [`trac.osgeo.org/osgeo4w/`](http://trac.osgeo.org/osgeo4w/) 找到。

虽然这些包最容易使用，但它们带有自己的 Python 版本。如果你已经安装了 Python，那么仅为了使用某些库就安装另一个 Python 发行版可能会出现问题。在这种情况下，第三个选项可能适合你。

第三个选项安装了针对您的 Python 版本预编译的二进制文件。这种方法在安装简便性和定制之间提供了最佳折衷。但是，您必须确保二进制发行版和相应的 Python 绑定彼此兼容，与您的 Python 版本兼容，并且在许多情况下与您的操作系统配置兼容。

# Windows

每年，Windows 上 Python 的 GDAL 安装都变得越来越容易。要在 Windows 上安装 GDAL，您必须检查您是否正在运行 32 位或 64 位版本的 Python：

1.  要这样做，只需在命令提示符中启动 Python 解释器，如下面的代码所示：

```py
Python 3.4.2 (v3.4.2:ab2c023a9432, Oct 6 2014, 22:15:05) [MSC v.1600
32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more
information.
```

1.  基于此实例，我们可以看到 Python 版本为 3.4.2 的`win32`，这意味着它是 32 位版本。一旦您有了这些信息，请访问以下 URL：[`www.lfd.uci.edu/~gohlke/pythonlibs/#gdal`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)。

1.  这个网页包含了适用于几乎所有开源科学库的 Python Windows 二进制文件和绑定。在该网页的 GDAL 部分，找到与您的 Python 版本匹配的版本。版本名称使用 C Python 的缩写`cp`，后跟主要的 Python 版本号，以及 32 位 Windows 的`win32`或 64 位 Windows 的`win_amd64`。

在前面的例子中，我们会下载名为`GDAL-1.11.3-cp34-none-win32.whl`的文件。

1.  此下载包是较新的 Python `pip` wheel 格式。要安装它，只需打开命令提示符并输入以下代码：

```py
pip install GDAL-1.11.3-cp34-none-win32.whl
```

1.  一旦安装了包，打开 Python 解释器并运行以下命令，通过检查其版本来验证 GDAL 是否已安装：

```py
Python 3.4.2 (v3.4.2:ab2c023a9432, Oct 6 2014, 22:15:05) [MSC v.1600 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from osgeo import gdal
>>> gdal.__version__
1.11.3
```

现在，GDAL 应该返回版本号`1.11.3`。

如果您在使用`easy_install`或`pip`和 PyPI 安装模块时遇到问题，请尝试从与 GDAL 示例相同的网站下载并安装 wheel 包。

# Linux

Linux 上的 GDAL 安装因发行版而异。以下[`gdal.org`](https://gdal.org)二进制网页列出了几个发行版的安装说明：[`trac.osgeo.org/gdal/wiki/DownloadingGdalBinaries`](http://trac.osgeo.org/gdal/wiki/DownloadingGdalBinaries)。让我们开始吧：

1.  通常，您的包管理器会安装 GDAL 和 Python 绑定。例如，在 Ubuntu 上，要安装 GDAL，您需要运行以下代码：

```py
sudo apt-get install gdal-bin
```

1.  然后，要安装 Python 绑定，您可以运行以下命令：

```py
sudo apt-get install python3-gdal
```

1.  大多数 Linux 发行版已经配置好了编译软件，它们的说明比 Windows 上的简单得多。

1.  根据安装情况，您可能需要将`gdal`和`ogr`作为`osgeo`包的一部分导入，如下面的命令所示：

```py
>>> from osgeo import gdal
>>> from osgeo import ogr
```

# macOS X

要在 macOS X 上安装 GDAL，您还可以使用 Homebrew 包管理系统，该系统可在[`brew.sh/`](http://brew.sh/)找到。

或者，您可以使用 MacPorts 软件包管理系统，该系统可在 [`www.macports.org/`](https://www.macports.org/) 获取。

这两个系统都有很好的文档记录，并包含适用于 Python 3 的 GDAL 包。你实际上只需要它们用于需要正确编译的二进制文件（用 C 语言编写，具有许多依赖项并包含许多科学和地理空间库）的库。

# Python 网络库用于获取数据

大多数地理空间数据共享都是通过互联网完成的，Python 在处理几乎任何协议的网络库方面都准备得很充分。自动数据下载通常是自动化地理空间过程的一个重要步骤。数据通常从网站的 **统一资源定位符** (**URL**) 或 **文件传输协议** (**FTP**) 服务器检索，由于地理空间数据集通常包含多个文件，因此数据通常以 ZIP 文件的形式分发。

Python 的一个优点是其文件类似对象的概念。大多数用于读取和写入数据的 Python 库都使用一组标准方法，允许您从不同类型的资源访问数据，就像您在磁盘上写入一个简单的文件一样。Python 标准库中的网络模块也使用这种约定。这种方法的优点是它允许您将文件类似对象传递给其他库和方法，这些库和方法可以识别该约定，而无需为以不同方式分发的不同类型的数据进行大量设置。

# Python 的 urllib 模块

Python 的 `urllib` 包旨在简单访问任何具有 URL 地址的文件。Python 3 中的 `urllib` 包由几个模块组成，这些模块处理管理网络请求和响应的不同部分。这些模块实现了 Python 的一些文件类似对象约定，从其 `open()` 方法开始。当你调用 `open()` 时，它会准备与资源的连接，但不会访问任何数据。有时，你只想获取一个文件并将其保存到磁盘上，而不是将其加载到内存中。这个功能可以通过 `urllib.request.retrieve()` 方法获得。

以下示例使用 `urllib.request.retrieve()` 方法下载名为 `hancock.zip` 的压缩形状文件，该文件在其他示例中使用。我们定义了 URL 和本地文件名作为变量。URL 作为参数传递，以及我们想要使用的文件名，以将其保存到我们的本地机器上，在这种情况下，只是 `hancock.zip`：

```py
>>> import urllib.request
>>> import urllib.parse
>>> import urllib.error
>>> url = "https://github.com/GeospatialPython/
Learn/raw/master/hancock.zip"
>>> fileName = "hancock.zip"
>>> urllib.request.urlretrieve(url, fileName)
('hancock.zip', <httplib.HTTPMessage instance at 0x00CAD378>)
```

来自底层`httplib`模块的消息确认文件已下载到当前目录。URL 和文件名也可以直接作为字符串传递给`retrieve()`方法。如果你只指定文件名，下载将保存到当前工作目录。你也可以指定一个完全限定的路径名来将其保存到其他位置。你还可以指定一个回调函数作为第三个参数，该函数将接收文件的下载状态信息，这样你就可以创建一个简单的下载状态指示器或执行其他操作。

`urllib.request.urlopen()`方法允许你以更高的精度和控制访问在线资源。正如我们之前提到的，它实现了大多数 Python 文件类似对象方法，除了`seek()`方法，它允许你在文件中的任意位置跳转。你可以逐行读取在线文件，将所有行作为列表读取，读取指定数量的字节，或者遍历文件的每一行。所有这些功能都在内存中执行，因此你不需要将数据存储在磁盘上。这种能力对于访问可能需要处理但不想保存到磁盘上的在线频繁更新的数据非常有用。

在下面的示例中，我们通过访问**美国地质调查局**（**USGS**）地震源来演示这个概念，查看在过去一小时内发生的所有地震。这些数据以**逗号分隔值**（**CSV**）文件的形式分发，我们可以像文本文件一样逐行读取。CSV 文件类似于电子表格，可以在文本编辑器或电子表格程序中打开：

1.  首先，你需要打开 URL 并读取包含文件列名的标题。

1.  然后，你需要读取第一行，其中包含最近地震的记录，如下面的代码行所示：

```py
>>> url = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/
summary/all_hour.csv"
>>> earthquakes = urllib.request.urlopen(url)
>>> earthquakes.readline()
'time,latitude,longitude,depth,mag,magType,nst,gap,dmin,rms,net,
id,updated,place
\n'
>>> earthquakes.readline()
'2013-06-14T14:37:57.000Z,64.8405,-147.6478,13.1,0.6,Ml,
6,180,0.09701805,0.2,ak,
ak10739050,2013-06-14T14:39:09.442Z,"3km E of Fairbanks,
Alaska"\n'
```

1.  我们也可以遍历这个文件，这是一种读取大文件的内存高效方式。

1.  如果你在这个 Python 解释器中运行这个示例，你需要按下*Enter*或*return*键两次来执行循环。这个动作是必要的，因为它向解释器发出信号，表明你已经完成了循环的构建。在下面的示例中，我们简化了输出：

```py
>>> for record in earthquakes: print(record)
2013-06-14T14:30:40.000Z,62.0828,-145.2995,22.5,1.6,
Ml,8,108,0.08174669,0.86,ak,
ak10739046,2013-06-14T14:37:02.318Z,"13km ESE of Glennallen,
Alaska"
...
2013-06-14T13:42:46.300Z,38.8162,-122.8148,3.5,0.6,
Md,,126,0.00898315,0.07,nc,nc
72008115,2013-06-14T13:53:11.592Z,"6km NW of The Geysers,
California"
```

# Python 请求模块

`urllib`模块已经存在很长时间了。另一个第三方模块已经被开发出来，使得常见的 HTTP 请求更加容易。`requests`模块具有以下功能：

+   保持连接和连接池

+   国际域名和 URL

+   会话中保持 cookie 持久性

+   浏览器风格的 SSL 验证

+   自动内容解码

+   基本摘要认证

+   精美的键/值 cookie

+   自动解压缩

+   Unicode 响应体

+   HTTP(S)代理支持

+   多部分文件上传

+   流式下载

+   连接超时

+   分块请求

+   `.netrc`支持

在下面的例子中，我们将下载与使用`urllib`模块下载的相同 ZIP 文件，但这次我们将使用`requests`模块。首先，我们需要安装`requests`模块：

```py
pip install requests
```

然后，我们可以导入它：

```py
import requests
```

然后，我们可以设置 URL 和输出文件名的变量：

```py
url = "https://github.com/GeospatialPython/Learning/raw/master/hancock.zip"
fileName = "hancock.zip"
```

使用`requests`模块的`get()`方法检索 ZIP 文件非常简单：

```py
r = requests.get(url)
```

现在，我们可以从`.zip`文件中获取内容并将其写入我们的输出文件：

```py
with open(fileName, 'wb') as f:
 f.write(r.content)
```

`requests`模块还有许多其他高级功能，使用起来与这个例子一样简单。现在我们知道了如何通过 HTTP 协议获取信息，让我们来检查 FTP 协议，它通常用于从在线存档访问地理空间数据。

# FTP

FTP 允许你使用 FTP 客户端软件浏览在线目录并下载数据。直到大约 2004 年，当地理空间网络服务变得非常普遍之前，FTP 是分发地理空间数据最常见的方式之一。现在 FTP 不太常见，但在搜索数据时偶尔会遇到它。再次强调，Python 的内置标准库有一个名为`ftplib`的合理 FTP 模块，其主要类名为`FTP()`。

在下面的例子中，我们将执行以下操作：

1.  我们将访问由美国**国家海洋和大气管理局**（**NOAA**）托管的 FTP 服务器，以访问包含全球海啸监测网络**深海评估和报告**（**DART**）浮标数据的文本文件。这个特定的浮标位于秘鲁海岸。

1.  我们将定义服务器和目录路径，然后我们将访问服务器。所有 FTP 服务器都需要用户名和密码。大多数公共服务器都有一个名为 anonymous 的用户，密码也是 anonymous，就像这个服务器一样。

1.  使用 Python 的`ftplib`，你可以不带任何参数调用`login()`方法以默认匿名用户身份登录。否则，你可以添加用户名和密码作为字符串参数。

1.  登录后，我们将切换到包含 DART 数据文件的目录。

1.  要下载文件，我们将打开一个名为 out 的本地文件，并将它的`write()`方法作为回调函数传递给`ftplib.ftp.retrbinary()`方法，该方法同时下载文件并将其写入我们的本地文件。

1.  文件下载完成后，我们可以关闭它以保存它。

1.  然后，我们将读取文件并查找包含浮标纬度和经度的行，以确保数据已成功下载，如下面的代码行所示：

```py
import ftplib

server = "ftp.ngdc.noaa.gov"
dir = "hazards/DART/20070815_peru"
fileName = "21415_from_20070727_08_55_15_tides.txt"
ftp = ftplib.FTP(server)
ftp.login()
ftp.cwd(dir)

with open(fileName, "wb") as out:
    ftp.retrbinary("RETR " + fileName, out.write)

with open(fileName) as dart:
 for line in dart:
     if "LAT, " in line:
     print(line)
     break
```

输出如下：

```py
 LAT,   LON      50.1663    171.8360
```

在这个例子中，我们以二进制写入模式打开了本地文件，并使用了`retrbinary()``ftplib`方法，而不是使用 ASCII 模式的`retrlines()`。二进制模式适用于 ASCII 和二进制文件，因此始终是一个更安全的赌注。实际上，在 Python 中，文件的二进制读写模式仅在 Windows 上需要。

如果你只是从 FTP 服务器下载一个简单的文件，许多 FTP 服务器也有一个网络界面。在这种情况下，你可以使用`urllib`来读取文件。FTP URL 使用以下格式来访问数据：

```py
ftp://username:password@server/directory/file
```

这种格式对于密码保护的目录来说是不安全的，因为你正在通过互联网传输你的登录信息。但对于匿名 FTP 服务器来说，没有额外的安全风险。为了演示这一点，以下示例通过使用`urllib`而不是`ftplib`来访问我们刚刚看到的相同文件：

```py
>>> dart = urllib.request.urlopen("ftp://" + server + "/" + dir +
"/" + fileName)
>>> for line in dart:
... line = str(line, encoding="utf8")
... if "LAT," in line:
... print(line)
... break
...
LAT, LON 50.1663 171.8360
```

现在我们已经可以下载文件了，让我们学习如何解压缩它们。

# ZIP 和 TAR 文件

地理空间数据集通常由多个文件组成。因此，它们通常以 ZIP 或 TAR 文件归档的形式分发。这些格式也可以压缩数据，但它们捆绑多个文件的能力是它们被用于地理空间数据的主要原因。虽然 TAR 格式不包含压缩算法，但它结合了 gzip 压缩，并将其作为程序选项提供。Python 有用于读取和写入 ZIP 和 TAR 归档的标准模块。这些模块分别称为`zipfile`和`tarfile`。

以下示例使用`urllib`下载的`hancock.zip`文件中的`hancock.shp`、`hancock.shx`和`hancock.dbf`文件，用于在之前的示例中使用。此示例假定 ZIP 文件位于当前目录中：

```py
>>> import zipfile
>>> zip = open("hancock.zip", "rb")
>>> zipShape = zipfile.ZipFile(zip)
>>> shpName, shxName, dbfName = zipShape.namelist()
>>> shpFile = open(shpName, "wb")
>>> shxFile = open(shxName, "wb")
>>> dbfFile = open(dbfName, "wb")
>>> shpFile.write(zipShape.read(shpName))
>>> shxFile.write(zipShape.read(shxName))
>>> dbfFile.write(zipShape.read(dbfName))
>>> shpFile.close()
>>> shxFile.close()
>>> dbfFile.close()
```

这个例子比必要的更详细，为了清晰起见。我们可以通过在`zipfile.namelist()`方法周围使用`for`循环来缩短这个例子，并使其更健壮，而不必明确地将不同的文件定义为变量。这种方法是一种更灵活且更 Pythonic 的方法，可以用于具有未知内容的 ZIP 归档，如下面的代码行所示：

```py
>>> import zipfile
>>> zip = open("hancock.zip", "rb")
>>> zipShape = zipfile.ZipFile(zip)
>>> for fileName in zipShape.namelist():
... out = open(fileName, "wb")
... out.write(zipShape.read(fileName))
... out.close()
>>>
```

现在你已经了解了`zipfile`模块的基础知识，让我们用我们刚刚解压的文件创建一个 TAR 归档。在这个例子中，当我们打开 TAR 归档进行写入时，我们指定写入模式为`w:gz`以进行 gzip 压缩。我们还指定文件扩展名为`tar.gz`以反映这种模式，如下面的代码行所示：

```py
>>> import tarfile
>>> tar = tarfile.open("hancock.tar.gz", "w:gz")
>>> tar.add("hancock.shp")
>>> tar.add("hancock.shx")
>>> tar.add("hancock.dbf")
>>> tar.close()
```

我们可以使用简单的`tarfile.extractall()`方法提取文件。首先，我们使用`tarfile.open()`方法打开文件，然后提取它，如下面的代码行所示：

```py
>>> tar = tarfile.open("hancock.tar.gz", "r:gz")
>>> tar.extractall()
>>> tar.close()
```

我们将通过结合本章学到的元素以及第二章“学习地理空间数据”中向量数据部分的元素来工作一个额外的示例。我们将从`hancock.zip`文件中读取边界框坐标，而无需将其保存到磁盘。我们将使用 Python 的文件-like 对象约定来传递数据。然后，我们将使用 Python 的`struct`模块来读取边界框，就像我们在第二章“学习地理空间数据”中所做的那样。

在这种情况下，我们将未压缩的`.shp`文件读入一个变量，并通过指定数据起始和结束索引（由冒号`:`分隔）使用 Python 数组切片来访问数据。我们能够使用列表切片，因为 Python 允许您将字符串视为列表。在这个例子中，我们还使用了 Python 的`StringIO`模块，以文件对象的形式在内存中临时存储数据，该对象实现了包括`seek()`方法在内的各种方法，而`seek()`方法在大多数 Python 网络模块中是缺失的，如下面的代码行所示：

```py
>>> import urllib.request
>>> import urllib.parse
>>> import urllib.error
>>> import zipfile
>>> import io
>>> import struct
>>> url =
"https://github.com/GeospatialPython/Learn/raw/master/hancock.zip"
>>> cloudshape = urllib.request.urlopen(url)
>>> memoryshape = io.BytesIO(cloudshape.read())
>>> zipshape = zipfile.ZipFile(memoryshape)
>>> cloudshp = zipshape.read("hancock.shp")
# Access Python string as an array
>>> struct.unpack("<dddd", cloudshp[36:68])
(-89.6904544701547, 30.173943486533133, -89.32227546981174,
30.6483914869749)
```

如您从迄今为止的示例中看到的，Python 的标准库包含了很多功能。大多数时候，您不需要下载第三方库就能访问在线文件。

# Python 标记和基于标签的解析器

基于标签的数据，尤其是不同的 XML 方言，已经成为分发地理空间数据的一种非常流行的方式。既适合机器阅读又适合人类阅读的格式通常易于处理，尽管它们为了可用性牺牲了存储效率。这些格式对于非常大的数据集可能难以管理，但在大多数情况下工作得很好。

尽管大多数格式都是某种形式的 XML（例如 KML 或 GML），但有一个显著的例外。**已知文本**（**WKT**）格式相当常见，但使用外部标记和方括号（`[]`）来包围数据，而不是像 XML 那样使用尖括号包围数据。

Python 对 XML 有标准库支持，还有一些优秀的第三方库可用。所有合适的 XML 格式都遵循相同的结构，因此您可以使用通用的 XML 库来读取它。因为 XML 是基于文本的，所以通常很容易将其作为字符串编写，而不是使用 XML 库。大多数输出 XML 的应用程序都是这种方式。

使用 XML 库编写 XML 的主要优势是您的输出通常经过验证。在创建自己的 XML 格式时，很容易出错。单个缺失的引号就可以使 XML 解析器崩溃，并给试图读取您数据的人抛出错误。当这些错误发生时，它们几乎使您的数据集变得无用。您会发现这个问题在基于 XML 的地理空间数据中非常普遍。您会发现一些解析器对不正确的 XML 比其他解析器更宽容。通常，可靠性比速度或内存效率更重要。

在[`lxml.de/performance.html`](http://lxml.de/performance.html)提供的分析中，提供了不同 Python XML 解析器在内存和速度方面的基准。

# minidom 模块

Python 的`minidom`模块是一个非常古老且易于使用的 XML 解析器。它是 Python XML 包内建的一组 XML 工具的一部分。它可以解析 XML 文件或作为字符串输入的 XML。`minidom`模块最适合小于约 20 MB 的小到中等大小的 XML 文档，因为在此之前的速度开始下降。

为了演示 `minidom` 模块，我们将使用一个示例 KML 文件，这是 Google KML 文档的一部分，你可以下载。以下链接中的数据代表从 GPS 设备传输的时间戳点位置：[`github.com/GeospatialPython/Learn/raw/master/time-stamp-point.kml`](https://github.com/GeospatialPython/Learn/raw/master/time-stamp-point.kml)。让我们开始吧：

1.  首先，我们将通过从文件中读取数据并创建一个 `minidom` 解析器对象来解析这些数据。文件包含一系列 `<Placemark>` 标签，这些标签包含一个点和收集该点的时间戳。因此，我们将获取文件中所有 `Placemarks` 的列表，并且可以通过检查该列表的长度来计数，如下面的代码行所示：

```py
>>> from xml.dom import minidom
>>> kml = minidom.parse("time-stamp-point.kml")
>>> Placemarks = kml.getElementsByTagName("Placemark")
>>> len(Placemarks)
361
```

1.  如你所见，我们检索了所有的 `Placemark`，总数为 `361`。现在，让我们看看列表中的第一个 `Placemark` 元素：

```py
>>> Placemarks[0]
<DOM Element: Placemark at 0x2045a30>
```

现在，每个 `<Placemark>` 标签都是一个 DOM 元素数据类型。为了真正看到这个元素是什么，我们调用 `toxml()` 方法，如下所示：

```py
>>> Placemarks[0].toxml()
u'<Placemark>\n <TimeStamp>\n \<when>2007-01-14T21:05:02Z</when>\n
</TimeStamp>\n <styleUrl>#paddle-a</styleUrl>\n <Point>\n
<coordinates>-122.536226,37.86047,0</coordinates>\n
</Point>\n </Placemark>'
```

1.  `toxml()` 函数将 `Placemark` 标签内包含的所有内容输出为一个字符串对象。如果我们想将此信息打印到文本文件中，我们可以调用 `toprettyxml()` 方法，这将添加额外的缩进来使 XML 更易于阅读。

1.  现在，如果我们只想从这个 placemark 中获取坐标会怎样？坐标隐藏在 `coordinates` 标签中，该标签位于 `point` 标签内，嵌套在 `Placemark` 标签内。`minidom` 对象的每个元素都称为 **节点**。嵌套节点称为子节点或子节点。子节点不仅包括标签，还包括分隔标签的空白，以及标签内的数据。因此，我们可以通过标签名称钻到 `coordinates` 标签，但之后我们需要访问 `data` 节点。所有的 `minidom` 元素都有 `childNodeslist`，以及一个 `firstChild()` 方法来访问第一个节点。

1.  我们将结合这些方法来获取第一个坐标的 `data` 节点的 `data` 属性，我们使用列表中的索引 `0` 来引用坐标标签列表：

```py
>>> coordinates =
Placemarks[0].getElementsByTagName("coordinates")
>>> point = coordinates[0].firstChild.data
>>> point
u'-122.536226,37.86047,0'
```

如果你刚接触 Python，你会注意到这些示例中的文本输出被标记为字母 `u`。这种标记是 Python 表示支持国际化并使用不同字符集的多语言 Unicode 字符串的方式。Python 3.4.3 对此约定略有改变，不再标记 Unicode 字符串，而是用 `b` 标记 UTF-8 字符串。

1.  我们可以更进一步，将这个 `point` 字符串转换为可用的数据，通过分割字符串并将结果字符串转换为 Python 浮点类型来实现，如下所示：

```py
>>> x,y,z = point.split(",")
>>> x
u'-122.536226'
>>> y
u'37.86047'
>>> z
u'0'
>>> x = float(x)
>>> y = float(y)
>>> z = float(z)
>>> x,y,z
(-122.536226, 37.86047, 0.0)
```

1.  使用 Python 列表推导式，我们可以一步完成这个操作，如下面的代码行所示：

```py
>>> x,y,z = [float(c) for c in point.split(",")]
>>> x,y,z
(-122.536226, 37.86047, 0.0)
```

这个例子只是触及了`minidom`库所能做到的一小部分。关于这个库的精彩教程，请查看以下教程：[`www.edureka.co/blog/python-xml-parser-tutorial/`](https://www.edureka.co/blog/python-xml-parser-tutorial/)。

# ElementTree

`minidom`模块是纯 Python 编写，易于使用，自 Python 2.0 以来一直存在。然而，Python 2.5 在标准库中增加了一个更高效但更高级的 XML 解析器，称为`ElementTree`。`ElementTree`很有趣，因为它已经实现了多个版本。

有一个纯 Python 版本和一个用 C 编写的更快版本，称为`cElementTree`。你应该尽可能使用`cElementTree`，但可能你所在的平台不包括基于 C 的版本。当你导入`cElementTree`时，你可以测试它是否可用，并在必要时回退到纯 Python 版本：

```py
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
```

`ElementTree`的一个伟大特性是它实现了 XPath 查询语言的一个子集。XPath 代表 XML Path，允许你使用路径式语法搜索 XML 文档。如果你经常处理 XML，学习 XPath 是必不可少的。你可以在以下链接中了解更多关于 XPath 的信息：[`www.w3schools.com/xml/xpath_intro.asp`](https://www.w3schools.com/xml/xpath_intro.asp)。

这个特性的一个问题是，如果文档指定了命名空间，就像大多数 XML 文档一样，你必须将那个命名空间插入到查询中。`ElementTree`不会自动为你处理命名空间。你的选择是手动指定它或尝试从根元素的标签名中通过字符串解析提取它。

我们将使用`ElementTree`重复`minidomXML`解析示例：

1.  首先，我们将解析文档，然后手动定义 KML 命名空间；稍后，我们将使用 XPath 表达式和`find()`方法来查找第一个`Placemark`元素。

1.  最后，我们将找到坐标和子节点，然后获取包含纬度和经度的文本。

在这两种情况下，我们都可以直接搜索`coordinates`标签。但是，通过获取`Placemark`元素，它给我们提供了选择，稍后如果需要的话，可以获取相应的 timestamp 子元素，如下面的代码所示：

```py
>>> tree = ET.ElementTree(file="time-stamp-point.kml")
>>> ns = "{http://www.opengis.net/kml/2.2}"
>>> placemark = tree.find(".//%sPlacemark" % ns)
>>> coordinates =
placemark.find("./{}Point/{}coordinates".format(ns, ns))
>>> coordinates.text
'-122.536226,37.86047,0'
```

在这个例子中，请注意我们使用了 Python 字符串格式化语法，它基于 C 中的字符串格式化概念。当我们为 placemark 变量定义 XPath 表达式时，我们使用了`%`占位符来指定字符串的插入。然后，在字符串之后，我们使用了`%`运算符后跟变量名来在占位符处插入`ns`命名空间变量。在`coordinates`变量中，我们使用了`ns`变量两次，因此在字符串之后指定了包含两次`ns`的元组。

字符串格式化是 Python 中一种简单但极其强大且有用的工具，值得学习。你可以在以下链接中找到更多信息：[`docs.python.org/3.4/library/string.html`](https://docs.python.org/3.4/library/string.html)。

# 使用 ElementTree 和 Minidom 构建 XML

大多数时候，XML 可以通过连接字符串来构建，如下面的命令所示：

```py
xml = "<?xml version="1.0" encoding="utf-8"?>"
xml += "<kml >"
xml += " <Placemark>"
xml += " <name>Office</name>"
xml += " <description>Office Building</description>"
xml += " <Point>"
xml += " <coordinates>"
xml += " -122.087461,37.422069"
xml += " </coordinates>"
xml += " </Point>"
xml += " </Placemark>"
xml += "</kml>"
```

然而，这种方法很容易出错，这会创建无效的 XML 文档。一种更安全的方法是使用 XML 库。让我们使用 `ElementTree` 构建这个简单的 KML 文档：

1.  我们将定义 `rootKML` 元素并为其分配一个命名空间。

1.  然后，我们将系统地添加子元素到根元素，将元素包装为 `ElementTree` 对象，声明 XML 编码，并将其写入名为 `placemark.xml` 的文件中，如下面的代码行所示：

```py
>>> root = ET.Element("kml")
>>> root.attrib["xmlns"] = "http://www.opengis.net/kml/2.2"
>>> placemark = ET.SubElement(root, "Placemark")
>>> office = ET.SubElement(placemark, "name")
>>> office.text = "Office"
>>> point = ET.SubElement(placemark, "Point")
>>> coordinates = ET.SubElement(point, "coordinates")
>>> coordinates.text = "-122.087461,37.422069, 37.422069"
>>> tree = ET.ElementTree(root)
>>> tree.write("placemark.kml",
xml_declaration=True,encoding='utf-8',method="xml")
```

输出与之前的字符串构建示例相同，但 `ElementTree` 不缩进标签，而是将其作为一条长字符串写入。`minidom` 模块具有类似的接口，这在 Mark Pilgrim 的《深入 Python》一书中有所记录，该书籍在前面看到的 `minidom` 示例中有所引用。

XML 解析器，如 `minidom` 和 `ElementTree`，在格式完美的 XML 文档上工作得非常好。不幸的是，绝大多数的 XML 文档并不遵循这些规则，并包含格式错误或无效字符。你会发现你经常被迫处理这些数据，并且必须求助于非常规的字符串解析技术来获取你实际需要的少量数据。但是，多亏了 Python 和 Beautiful Soup，你可以优雅地处理糟糕的甚至极差的基于标签的数据。

Beautiful Soup 是一个专门设计用来鲁棒地处理损坏的 XML 的模块。它面向 HTML，HTML 以格式错误而闻名，但也适用于其他 XML 方言。Beautiful Soup 可在 PyPI 上找到，因此可以使用 `easy_install` 或 `pip` 来安装它，如下面的命令所示：

```py
easy_install beautifulsoup4
```

或者，你可以执行以下命令：

```py
pip install beautifulsoup4
```

然后，为了使用它，你只需简单地导入它：

```py
>>> from bs4 import BeautifulSoup
```

为了尝试它，我们将使用来自智能手机应用程序的 **GPS 交换格式**（**GPX**）跟踪文件，该文件存在故障并导出了略微损坏的数据。您可以从以下链接下载此样本文件：[`raw.githubusercontent.com/GeospatialPython/Learn/master/broken_data.gpx`](https://raw.githubusercontent.com/GeospatialPython/Learn/master/broken_data.gpx)。

这个 2,347 行的数据文件处于原始状态，除了它缺少一个关闭的`</trkseg>`标签，这个标签应该位于文件的末尾，就在关闭的`</trk>`标签之前。这个错误是由源程序中的数据导出函数引起的。这个缺陷很可能是原始开发者手动生成导出时的 GPX XML 并忘记添加此关闭标签的代码行所导致的。看看如果我们尝试用`minidom`解析这个文件会发生什么：

```py
>>> gpx = minidom.parse("broken_data.gpx")
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "C:\Python34\lib\xml\dom\minidom.py", line 1914, in parse
return expatbuilder.parse(file)
File "C:\Python34\lib\xml\dom\expatbuilder.py", line 924, in
parse
result = builder.parseFile(fp)
File "C:\Python34\lib\xml\dom\expatbuilder.py", line 207, in
parseFile
parser.Parse(buffer, 0)
xml.parsers.expat.ExpatError: mismatched tag: line 2346, column 2
```

如您从错误信息的最后一行所看到的，`minidom`中的底层 XML 解析器确切地知道问题所在——文件末尾存在一个`mismatched`标签。然而，它拒绝做任何更多的事情，只是报告了错误。您必须拥有完美无瑕的 XML，或者根本不使用，以避免这种情况。

现在，让我们尝试使用相同数据的更复杂和高效的`ElementTree`模块：

```py
>>> ET.ElementTree(file="broken_data.gpx")
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "C:\Python34\lib\xml\etree\ElementTree.py", line 611, in
__init__
self.parse(file)
File "C:\Python34\lib\xml\etree\ElementTree.py", line 653, in
parse
parser.feed(data)
File "C:\Python34\lib\xml\etree\ElementTree.py", line 1624, in
feed
self._raiseerror(v)
File "C:\Python34\lib\xml\etree\ElementTree.py", line 1488, in
_raiseerror
raise err
xml.etree.ElementTree.ParseError: mismatched tag: line 2346,
column 2
```

如您所见，不同的解析器面临相同的问题。在地理空间分析中，格式不良的 XML 是一个过于常见的现实，每个 XML 解析器都假设世界上所有的 XML 都是完美的，除了一个。这就是 Beautiful Soup 的用武之地。这个库毫不犹豫地将不良 XML 撕成可用的数据，并且它可以处理比缺失标签更严重的缺陷。即使缺少标点符号或其他语法，它也能正常工作，并给出它能给出的最佳数据。它最初是为解析 HTML 而开发的，HTML 因其格式不良而臭名昭著，但它与 XML 也相当兼容，如下所示：

```py
>>> from bs4 import BeautifulSoup
>>> gpx = open("broken_data.gpx")
>>> soup = BeautifulSoup(gpx.read(), features="xml")
>>>
```

Beautiful Soup 没有任何抱怨！为了确保数据实际上是可以使用的，让我们尝试访问一些数据。Beautiful Soup 的一个令人惊叹的特性是它将标签转换为解析树的属性。如果有多个具有相同名称的标签，它将获取第一个。我们的样本数据文件有数百个`<trkpt>`标签。让我们访问第一个：

```py
>>> soup.trkpt
<trkpt lat="30.307267000" lon="-89.332444000">
<ele>10.7</ele><time>2013-05-16T04:39:46Z</time></trkpt>
```

我们现在确信数据已经被正确解析，并且我们可以访问它。如果我们想访问所有的`<trkpt>`标签，我们可以使用`findAll()`方法来获取它们，然后使用内置的 Python `len()`函数来计数，如下所示：

```py
>>> tracks = soup.findAll("trkpt")
>>> len(tracks)
2321
```

如果我们将解析的数据写回到文件中，Beautiful Soup 会输出修正后的版本。我们将使用 Beautiful Soup 模块的`prettify()`方法将固定数据保存为一个新的 GPX 文件，以格式化 XML 并添加漂亮的缩进，如下面的代码行所示：

```py
>>> fixed = open("fixed_data.gpx", "w")
>>> fixed.write(soup.prettify())
>>> fixed.close()
```

Beautiful Soup 是一个非常丰富的库，具有更多功能。要进一步探索它，请访问在线的 Beautiful Soup 文档[`www.crummy.com/software/BeautifulSoup/bs4/documentation.html`](http://www.crummy.com/software/BeautifulSoup/bs4/documentation.html)。

虽然`minidom`、`ElementTree`和`cElementTree`是 Python 标准库的一部分，但还有一个更强大、更受欢迎的 Python XML 库，称为`lxml`。`lxml`模块通过`ElementTree` API 提供了对`libxml2`和`libxslt` C 库的 Pythonic 接口。更好的事实是，`lxml`还可以与 Beautiful Soup 一起解析基于标签的坏数据。在某些安装中，`beautifulsoup4`可能需要`lxml`。`lxml`模块可通过 PyPI 获取，但需要为 C 库执行一些额外步骤。更多信息可在以下链接的`lxml`主页上找到：[`lxml.de/`](http://lxml.de/)。

# Well-Known Text (WKT)

WKT 格式已经存在多年，是一种简单的基于文本的格式，用于表示几何形状和空间参考系统。它主要用作实现 OGC Simple Features for SQL 规范的系统的数据交换格式。请看以下多边形的 WKT 表示示例：

```py
POLYGON((0 0,4 0,4 4,0 4,0 0),(1 1, 2 1, 2 2, 1 2,1 1))
```

目前，读取和写入 WKT 的最佳方式是使用 Shapely 库。Shapely 提供了一个非常 Python 导向或 Pythonic 的接口，用于我们第三章中描述的**Geometry Engine - Open Source**（**GEOS**）库，*地理空间技术景观*。

您可以使用`easy_install`或`pip`安装 Shapely。您还可以使用上一节中提到的网站上的 wheel。Shapely 有一个 WKT 模块，可以加载和导出这些数据。让我们使用 Shapely 加载之前的多边形样本，然后通过计算其面积来验证它是否已作为多边形对象加载：

```py
>>> import shapely.wkt
>>> wktPoly = "POLYGON((0 0,4 0,4 4,0 4,0 0),(1 1, 2 1, 2 2, 1 2,
1 1))"
>>> poly = shapely.wkt.loads(wktPoly)
>>> poly.area
15.0
```

我们可以通过简单地调用其`wkt`属性，将任何 Shapely 几何形状转换回 WKT，如下所示：

```py
>>> poly.wkt
'POLYGON ((0.0 0.0, 4.0 0.0, 4.0 4.0, 0.0 4.0, 0.0 0.0), (1.0 1.0,
2.0 1.0, 2.0 2.0, 1.0 2.0, 1.0 1.0))'
```

Shapely 还可以处理 WKT 的二进制对应物，称为 W**ell-Known Binary**（**WKB**），它用于在数据库中以二进制对象的形式存储 WKT 字符串。Shapely 使用其`wkb`模块以与`wkt`模块相同的方式加载 WKB，并且可以通过调用该对象的`wkb`属性来转换几何形状。

Shapely 是处理 WKT 数据最 Pythonic 的方式，但您也可以使用 OGR 库的 Python 绑定，这是我们本章早些时候安装的。

对于这个例子，我们将使用一个包含一个简单多边形的 shapefile，它可以作为一个 ZIP 文件下载。您可以通过以下链接获取：[`github.com/GeospatialPython/Learn/raw/master/polygon.zip`](https://github.com/GeospatialPython/Learn/raw/master/polygon.zip)。

在以下示例中，我们将打开 shapefile 数据集中的`polygon.shp`文件，调用所需的`GetLayer()`方法，获取第一个（也是唯一一个）要素，然后将其导出为 WKT 格式：

```py
>>> from osgeo import ogr
>>> shape = ogr.Open("polygon.shp")
>>> layer = shape.GetLayer()
>>> feature = layer.GetNextFeature()
>>> geom = feature.GetGeometryRef()
>>> wkt = geom.ExportToWkt()
>>> wkt
' POLYGON ((-99.904679362176353 51.698147686745074,
-75.010398603076666 46.56036851832075,-75.010398603076666
46.56036851832075,-75.010398603076666 46.56036851832075,
-76.975736557742451 23.246272688996914,-76.975736557742451
23.246272688996914,-76.975736557742451 23.246272688996914,
-114.31715769639194 26.220870210283724,-114.31715769639194
26.220870210283724,-99.904679362176353 51.698147686745074))'
```

注意，使用 OGR 时，你必须逐个读取每个要素并单独导出它，因为`ExporttoWkt()`方法是在要素级别。现在我们可以使用包含导出的`wkt`变量来读取 WKT 字符串。我们将将其导入`ogr`，并获取多边形的边界框，也称为包围盒，如下所示：

```py
>>> poly = ogr.CreateGeometryFromWkt(wkt)
>>> poly.GetEnvelope()
(-114.31715769639194, -75.01039860307667, 23.246272688996914,
51.698147686745074)
```

Shapely 和 OGR 用于读取和写入有效的 WKT 字符串。当然，就像 XML 一样，它也是文本，你可以在必要时将少量 WKT 作为字符串进行操作。接下来，我们将探讨一个在地理空间领域变得越来越常见的现代文本格式。

# Python JSON 库

**JavaScript 对象表示法**（**JSON**）正在迅速成为许多领域的首选数据交换格式。轻量级的语法以及它与 JavaScript 数据结构的相似性，使得它非常适合 Python，Python 也借鉴了一些数据结构。

以下 GeoJSON 样本文档包含一个单独的点：

```py
{
    "type": "Feature",
    "id": "OpenLayers.Feature.Vector_314",
    "properties": {},
    "geometry": {
        "type": "Point",
        "coordinates": [
            97.03125,
            39.7265625
        ]
    },
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    }
}
```

这个样本只是一个带有新属性的基本点，这些属性将被存储在几何的属性数据结构中。在前面的例子中，ID、坐标和 CRS 信息将根据你的特定数据集而变化。

让我们使用 Python 修改这个示例 GeoJSON 文档。首先，我们将样本文档压缩成一个字符串，以便更容易处理：

```py
>>> jsdata = """{
    "type": "Feature",
   "id": "OpenLayers.Feature.Vector_314",
  "properties": {},
    "geometry": {
        "type": "Point",
        "coordinates": [
           97.03125,
           39.7265625
       ]
    },
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    }
}"""
```

现在，我们可以使用前面代码中创建的 GeoJSON `jsdata` 字符串变量，在以下示例中使用。

# json 模块

GeoJSON 看起来非常类似于 Python 的嵌套字典和列表集合。为了好玩，让我们尝试使用 Python 的`eval()`函数将其解析为 Python 代码：

```py
>>> point = eval(jsdata)
>>> point["geometry"]
{'type': 'Point', 'coordinates': [97.03125, 39.7265625]}
```

哇！这成功了！我们只需一步就将那个随机的 GeoJSON 字符串转换成了原生的 Python 数据。记住，JSON 数据格式基于 JavaScript 语法，这恰好与 Python 相似。此外，随着你对 GeoJSON 数据的深入了解以及处理更大的数据，你会发现 JSON 允许使用 Python 不允许的字符。使用 Python 的`eval()`函数也被认为是非常不安全的。但就保持简单而言，这已经是最简单的方法了！

由于 Python 追求简单，更高级的方法并没有变得更加复杂。让我们使用 Python 的`json`模块，它是标准库的一部分，将相同的字符串正确地转换为 Python：

```py
>>> import json
>>> json.loads(jsdata)
{u'geometry': {u'type': u'Point', u'coordinates': [97.03125,
39.7265625]}, u'crs': {u'type': u'name', u'properties': {u'name':
u'urn:ogc:def:crs:OGC:1.3:CRS84'}}, u'type': u'Feature', u'id': u'OpenLayers.Feature.Vector_314',
u'properties':
{}}
```

作为旁注，在先前的例子中，CRS84 属性是常见 WGS84 坐标系统的同义词。`json`模块添加了一些很好的功能，例如更安全的解析和将字符串转换为 Unicode。我们可以以几乎相同的方式将 Python 数据结构导出为 JSON：

```py
>>> pydata = json.loads(jsdata)
>>> json.dumps(pydata)
'{"geometry": {"type": "Point", "coordinates": [97.03125,
39.7265625]}, "crs": {"type": "name", "properties": {"name":
"urn:ogc:def:crs:OGC:1.3:CRS84"}}, "type" : "Feature", "id": "OpenLayers.Feature.Vector_314", "properties":
{}}'
```

当你导出数据时，它会以一长串难以阅读的字符串形式输出。我们可以通过传递`dumps()`方法一个缩进值来打印数据，使其更容易阅读：

```py
print(json.dumps(pydata, indent=4))

{
    "type": "Feature",
    "id":
    "OpenLayers.Feature.Vector_314",
    "properties": {},
    "geometry": {
        "type": "Point",
        "coordinates": [
            97.03125,
            39.7265625
        ]
    },
    "crs": {
        "type": "name",
        "properties": {           
                 "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
   }
}
```

现在我们已经了解了 `json` 模块，让我们看看地理空间版本 `geojson`。

# geojson 模块

我们可以愉快地继续使用 `json` 模块读取和写入 GeoJSON 数据，但还有更好的方法。PyPI 上可用的 `geojson` 模块提供了一些独特的优势。首先，它了解 GeoJSON 规范的要求，这可以节省大量的输入。让我们使用此模块创建一个简单的点并将其导出为 GeoJSON：

```py
>>> import geojson
>>> p = geojson.Point([-92, 37])
```

这次，当我们转储 JSON 数据以供查看时，我们将添加一个缩进参数，其值为 `4`，这样我们就可以得到格式良好的缩进 JSON 数据，更容易阅读：

```py
>>> geojs = geojson.dumps(p, indent=4)
>>> geojs
```

我们的结果如下：

```py
{
    "type": "Point",
    "coordinates": [
        -92,
        37
    ]
}
POINT (-92 37)
```

注意到 `geojson` 模块为不同的数据类型提供了一个接口，并使我们免于手动设置类型和坐标属性。现在，想象一下如果你有一个具有数百个要素的地理对象。你可以通过编程构建这个数据结构，而不是构建一个非常大的字符串。

`geojson` 模块也是 Python `geo_interface` 规范的参考实现。这个接口允许协作程序以 Pythonic 的方式无缝交换数据，而无需程序员显式导出和导入 GeoJSON 字符串。因此，如果我们想将使用 `geojson` 模块创建的点传递给 Shapely 模块，我们可以执行以下命令，该命令将 `geojson` 模块的点对象直接读取到 Shapely 中，然后将其导出为 WKT：

```py
>>> from shapely.geometry import asShape
>>> point = asShape(p)
>>> point.wkt
'POINT (-92.0000000000000000 37.0000000000000000)'

```

越来越多的地理空间 Python 库正在实现 `geojson` 和 `geo_interface` 功能，包括 PyShp、Fiona、Karta 和 ArcGIS。对于 QGIS，存在第三方实现。

GeoJSON 是一种简单且易于人类和计算机阅读的文本格式。现在，我们将查看一些二进制矢量格式。

# OGR

我们提到了 OGR 作为处理 WKT 字符串的方法，但它的真正力量在于作为一个通用的矢量库。本书力求提供纯 Python 解决方案，但没有单个库甚至接近 OGR 可以处理的格式多样性。

让我们使用 OGR Python API 读取一个示例点形状文件。示例形状文件可以作为 ZIP 文件在此处下载：[`github.com/GeospatialPython/Learn/raw/master/point.zip`](https://github.com/GeospatialPython/Learn/raw/master/point.zip)。

这个点形状文件有五个带有单个数字、正坐标的点。属性列表了点的创建顺序，这使得它在测试中非常有用。这个简单的例子将读取点形状文件，并遍历每个要素；然后，它将打印每个点的 *x* 和 *y* 值，以及第一个属性字段的值：

```py
>>> import ogr
>>> shp = ogr.Open("point.shp")
>>> layer = shp.GetLayer()
>>> for feature in layer:
... geometry = feature.GetGeometryRef()
... print(geometry.GetX(), geometry.GetY(),
feature.GetField("FIRST_FLD"))
...
1.0 1.0 First
3.0 1.0 Second
4.0 3.0 Third
2.0 2.0 Fourth
0.0 0.0 Appended
```

这个例子很简单，但 OGR 在脚本变得更加复杂时可能会变得相当冗长。接下来，我们将看看处理形状文件的一种更简单的方法。

# PyShp

PyShp 是一个简单的纯 Python 库，用于读取和写入形状文件。它不执行任何几何操作，仅使用 Python 的标准库。它包含在一个易于移动、压缩到小型嵌入式平台和修改的单个文件中。它也与 Python 3 兼容。它还实现了`__geo_interface__`。PyShp 模块可在 PyPI 上找到。

让我们用 PyShp 重复之前的 OGR 示例：

```py
>>> import shapefile
>>> shp = shapefile.Reader("point.shp")
>>> for feature in shp.shapeRecords():
... point = feature.shape.points[0]
... rec = feature.record[0]
... print(point[0], point[1], rec)
...
1.0 1.0 First
3.0 1.0 Second
4.0 3.0 Third
2.0 2.0 Fourth
0.0 0.0 Appended //
```

# dbfpy

OGR 和 PyShp 都读取和写入`.dbf`文件，因为它们是形状文件规范的一部分。`.dbf`文件包含形状文件的属性和字段。然而，这两个库对`.dbf`的支持非常基础。偶尔，你可能需要进行一些重型的 DBF 工作。`dbfpy3`模块是一个专门用于处理`.dbf`文件的纯 Python 模块。它目前托管在 GitHub 上。你可以通过指定下载文件来强制`easy_install`找到下载：

```py
easy_install -f
 https://github.com/GeospatialPython/dbfpy3/archive/master.zip
```

如果您使用`pip`安装软件包，请使用以下命令：

```py
pip install
 https://github.com/GeospatialPython/dbfpy3/archive/master.zip
```

以下形状文件包含超过 600 条`.dbf`记录，代表美国人口普查局的区域，这使得它成为尝试`dbfpy`的好样本：[`github.com/GeospatialPython/Learn/raw/master/GIS_CensusTract.zip`](https://github.com/GeospatialPython/Learn/raw/master/GIS_CensusTract.zip).

让我们打开这个形状文件的`.dbf`文件并查看第一条记录：

```py
>>> from dbfpy3 import dbf
>>> db = dbf.Dbf("GIS_CensusTract_poly.dbf")
>>> db[0]
GEODB_OID: 4029 (<type 'int'>)
OBJECTID: 4029 (<type 'int'>)
PERMANE0: 61be9239-8f3b-4876-8c4c-0908078bc597 (<type 'str'>)
SOURCE_1: NA (<type 'str'>)
SOURCE_2: 20006 (<type 'str'>)
SOURCE_3: Census Tracts (<type 'str'>)
SOURCE_4: Census Bureau (<type 'str'>)
DATA_SE5: 5 (<type 'str'>)
DISTRIB6: E4 (<type 'str'>)
LOADDATE: 2007-03-13 (<type 'datetime.date'>)
QUALITY: 2 (<type 'str'>)
SCALE: 1 (<type 'str'>)
FCODE: 1734 (<type 'str'>)
STCO_FI7: 22071 (<type 'str'>)
STATE_NAME: 22 (<type 'str'>)
COUNTY_8: 71 (<type 'str'>)
CENSUST9: 22071001734 (<type 'str'>)
POPULAT10: 1760 (<type 'int'>)
AREASQKM: 264.52661934 (<type 'float'>)
GNIS_ID: NA (<type 'str'>)
POPULAT11: 1665 (<type 'int'>)
DB2GSE_12: 264526619.341 (<type 'float'>)
DB2GSE_13: 87406.406192 (<type 'float'>)
```

该模块快速且容易地为我们提供列名和数据值，而不是将它们作为单独的列表处理，这使得它们更容易管理。现在，让我们将包含在`POPULAT10`中的人口字段增加`1`：

```py
>>> rec = db[0]
>>> field = rec["POPULAT10"]
>>> rec["POPULAT10"] = field + 1
>>> rec.store()
>>> del rec
>>> db[0]["POPULAT10"]
1761
```

请记住，OGR 和 PyShp 都可以执行此相同的过程，但如果您只是对`.dbf`文件进行大量更改，`dbfp3y`会使它稍微容易一些。

# Shapely

Shapely 在**已知文本**（**WKT**）部分被提及，因为它具有导入和导出功能。然而，它的真正目的是作为一个通用的几何库。Shapely 是 GEOS 库的几何操作的高级 Python 接口。实际上，Shapely 故意避免读取或写入文件。它完全依赖于从其他模块导入和导出数据，并专注于几何操作。

让我们做一个快速的 Shapely 演示，我们将定义一个 WKT 多边形，然后将其导入 Shapely。然后，我们将测量面积。我们的计算几何将包括通过五个任意单位对多边形进行缓冲，这将返回一个新的更大的多边形，我们将测量其面积：

```py
>>> from shapely import wkt, geometry
>>> wktPoly = "POLYGON((0 0,4 0,4 4,0 4,0 0))"
>>> poly = wkt.loads(wktPoly)
>>> poly.area
16.0
>>> buf = poly.buffer(5.0)
>>> buf.area
174.41371226364848
```

然后，我们可以执行缓冲区面积与原始多边形面积的差值，如下所示：

```py
>>> buf.difference(poly).area
158.413712264
```

如果您不能使用纯 Python，那么像 Shapely 一样干净、功能强大的 Python API 无疑是下一个最佳选择。

# Fiona

Fiona 库提供了围绕 OGR 库的简单 Python API，用于数据访问，仅此而已。这种方法使其易于使用，并且在使用 Python 时比 OGR 更简洁。Fiona 默认输出 GeoJSON。您可以在 [`www.lfd.uci.edu/~gohlke/pythonlibs/#fiona`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona) 找到 Fiona 的 wheel 文件。

例如，我们将使用本章前面查看的 `dbfpy` 示例中的 `GIS_CensusTract_poly.shp` 文件。

首先，我们将导入 `fiona` 和 Python 的 `pprint` 模块以格式化输出。然后，我们将打开 shapefile 并检查其驱动程序类型：

```py
>>> import fiona
>>> from pprint import pprint
>>> f = fiona.open("GIS_CensusTract_poly.shp")
>>> f.driver
```

# ESRI shapefile

接下来，我们将检查其坐标参考系统并获取数据边界框，如下所示：

```py
>>> f.crs
{'init': 'epsg:4269'}
>>> f.bounds
(-89.8744162216216, 30.161122135135138, -89.1383837783784,
30.661213864864862)
```

现在，我们将以 `geojson` 格式查看数据模式，并使用 `pprint` 模块进行格式化，如下面的代码行所示：

```py
>>> pprint(f.schema)
{'geometry': 'Polygon',
'properties': {'GEODB_OID': 'float:11',
'OBJECTID': 'float:11',
'PERMANE0': 'str:40',
'SOURCE_1': 'str:40',
'SOURCE_2': 'str:40',
'SOURCE_3': 'str:100',
'SOURCE_4': 'str:130',
'DATA_SE5': 'str:46',
'DISTRIB6': 'str:188',
'LOADDATE': 'date',
'QUALITY': 'str:35',
'SCALE': 'str:52',
'FCODE': 'str:38',
'STCO_FI7': 'str:5',
'STATE_NAME': 'str:140',
'COUNTY_8': 'str:60',
'CENSUST9': 'str:20',
'POPULAT10': 'float:11',
'AREASQKM': 'float:31.15',
'GNIS_ID': 'str:10',
'POPULAT11': 'float:11',
'DB2GSE_12': 'float:31.15',
'DB2GSE_13': 'float:31.15'}}
```

接下来，让我们获取特征数量的统计：

```py
>>> len(f)
45
```

最后，我们将打印一条记录作为格式化的 GeoJSON，如下所示：

```py
pprint(f[1])
{'geometry': {'coordinates': [[[(-89.86412366375093,
30.661213864864862), (-89.86418691770497, 30.660764012731285),
(-89.86443391770518, 30.659652012730202),
...
'type': 'MultiPolygon'},
'id': '1',
'properties': {'GEODB_OID': 4360.0,
'OBJECTID': 4360.0,
'PERMANE0': '9a914eef-9249-44cf-a05f-af4b48876c59',
'SOURCE_1': 'NA',
'SOURCE_2': '20006',
...
'DB2GSE_12': 351242560.967882,
'DB2GSE_13': 101775.283967268},
'type': 'Feature'}
```

# GDAL

GDAL 是用于栅格数据的占主导地位的地理空间库。其栅格功能非常显著，以至于它是任何语言中几乎所有地理空间工具包的一部分，Python 也不例外。要了解 GDAL 在 Python 中的基本工作原理，请下载以下示例栅格卫星图像的 ZIP 文件并解压：[`github.com/GeospatialPython/Learn/raw/master/SatImage.zip`](https://github.com/GeospatialPython/Learn/raw/master/SatImage.zip)。让我们打开这张图片，看看它有多少波段以及每个轴上有多少像素：

```py
>>> from osgeo import gdal
>>> raster = gdal.Open("SatImage.tif")
>>> raster.RasterCount
3
>>> raster.RasterXSize
2592
>>> raster.RasterYSize
2693
```

通过在 OpenEV 中查看，我们可以看到以下图像有三个波段，2,592 列像素和 2,693 行像素：

![](img/1831299e-4500-49f1-8ef2-f8ee35f21442.png)

GDAL 是 Python 中一个极快的地理空间栅格读取器和写入器。它还可以很好地重新投影图像，除了能够执行一些其他技巧之外。然而，GDAL 的真正价值来自于它与下一个 Python 模块的交互，我们现在将对其进行检查。

# NumPy

NumPy 是一个专为 Python 和科学计算设计的极快的多维 Python 数组处理器，但用 C 语言编写。它可以通过 PyPI 或作为 wheel 文件（可在 [`www.lfd.uci.edu/~gohlke/pythonlibs/#numpy`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) 找到）获得，并且可以轻松安装。除了其惊人的速度外，NumPy 的魔力还包括其与其他库的交互。NumPy 可以与 GDAL、Shapely、**Python 图像库**（**PIL**）以及许多其他领域的科学计算 Python 库交换数据。

作为 NumPy 能力的快速示例，我们将将其与 GDAL 结合起来读取我们的示例卫星图像，然后创建其直方图。GDAL 和 NumPy 之间的接口是一个名为 `gdal_array` 的 GDAL 模块，它依赖于 NumPy。Numeric 是 NumPy 模块的传统名称。`gdal_array` 模块导入 NumPy。

在以下示例中，我们将使用`gdal_array`，它导入 NumPy，将图像作为数组读取，获取第一个波段，并将其保存为 JPEG 图像：

```py
>>> from osgeo import gdal_array
>>> srcArray = gdal_array.LoadFile("SatImage.tif")
>>> band1 = srcArray[0]
>>> gdal_array.SaveArray(band1, "band1.jpg", format="JPEG")
```

这个操作在 OpenEV 中给出了以下灰度图像：

![](img/9e69e56e-1fcd-4829-9bbd-e3f7e5e1e51e.png)

# PIL

PIL 最初是为遥感开发的，但已经发展成为一个通用的 Python 图像编辑库。像 NumPy 一样，它是用 C 语言编写的，以提高速度，但它是专门为 Python 设计的。除了图像创建和处理外，它还有一个有用的栅格绘图模块。PIL 也可以通过 PyPI 获得；然而，在 Python 3 中，你可能想使用 Pillow 模块，它是 PIL 的升级版本。正如你将在以下示例中看到的那样，我们可以使用 Python 的 try 语句以两种可能的方式导入 PIL，这取决于你的安装方式。

在这个示例中，我们将结合 PyShp 和 PIL 将前一个示例中的`hancock` shapefile 转换为栅格并保存为图像。我们将使用类似于第一章中的 SimpleGIS 的世界到像素坐标转换，即使用 Python 进行地理空间分析。我们将创建一个图像作为 PIL 中的画布，然后使用 PIL 的`ImageDraw`模块来渲染多边形。最后，我们将将其保存为 PNG 图像，如下面的代码行所示：

```py
>>> try:
>>> import Image
>>> import ImageDraw
>>> except:
>>> from PIL import Image
>>> from PIL import ImageDraw
>>> import shapefile
>>> r = shapefile.Reader("hancock.shp")
>>> xdist = r.bbox[2] - r.bbox[0]
>>> ydist = r.bbox[3] - r.bbox[1]
>>> iwidth = 400
>>> iheight = 600
>>> xratio = iwidth/xdist
>>> yratio = iheight/ydist
>>> pixels = []
>>> for x,y in r.shapes()[0].points:
... px = int(iwidth - ((r.bbox[2] - x) * xratio))
... py = int((r.bbox[3] - y) * yratio)
... pixels.append((px,py))
...
>>> img = Image.new("RGB", (iwidth, iheight), "white")
>>> draw = ImageDraw.Draw(img)
>>> draw.polygon(pixels, outline="rgb(203, 196, 190)",
fill="rgb(198, 204, 189)")
>>> img.save("hancock.png")
```

这个示例创建以下图像：

![](img/fe5e5ae1-a9ff-4c7e-b9e1-fa1da206b56b.png)

# PNGCanvas

有时，你可能发现 PIL 对于你的目的来说过于强大，或者你不允许安装 PIL，因为你没有使用 Python 模块的 C 语言创建和编译的机器的管理权限。在这些情况下，你通常可以使用轻量级的纯 Python PNGCanvas 模块来解决问题。你可以使用`easy_install`或 pip 来安装它。

使用此模块，我们可以重复使用 PIL 执行的光栅形状文件示例，但使用纯 Python，如下所示：

```py
>>> import shapefile
>>> import pngcanvas
>>> r = shapefile.Reader("hancock.shp")
>>> xdist = r.bbox[2] - r.bbox[0]
>>> ydist = r.bbox[3] - r.bbox[1]
>>> iwidth = 400
>>> iheight = 600
>>> xratio = iwidth/xdist
>>> yratio = iheight/ydist
>>> pixels = []
>>> for x,y in r.shapes()[0].points:
... px = int(iwidth - ((r.bbox[2] - x) * xratio))
... py = int((r.bbox[3] - y) * yratio)
... pixels.append([px,py])
...
>>> c = pngcanvas.PNGCanvas(iwidth,iheight)
>>> c.polyline(pixels)
>>> f = open("hancock_pngcvs.png", "wb")
>>> f.write(c.dump())
>>> f.close()
```

这个示例提供了一个简单的轮廓，因为 PNGCanvas 没有内置的填充方法：

![](img/a5135eda-4450-4838-8565-64ba2695193f.png)

# GeoPandas

Pandas 是一个高性能的 Python 数据分析库，可以处理大型表格数据集（类似于数据库），有序/无序，标记矩阵或未标记的统计数据。GeoPandas 是 Pandas 的一个地理空间扩展，基于 Shapely、Fiona、PyProj、Matplotlib 和 Descartes 构建，所有这些都必须安装。它允许你轻松地在 Python 中执行操作，否则可能需要像 PostGIS 这样的空间数据库。你可以从[`www.lfd.uci.edu/~gohlke/pythonlibs/#panda`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#panda)下载 GeoPandas 的 wheel 文件。

以下脚本打开一个 shapefile 并将其转换为 GeoJSON。然后，它使用`matplotlib`创建一个地图：

```py
>>> import geopandas
>>> import matplotlib.pyplot as plt
>>> gdf = geopandas.GeoDataFrame
>>> census = gdf.from_file("GIS_CensusTract_poly.shp")
>>> census.plot()
>>> plt.show()
```

以下图像是先前命令的结果地图：

![](img/930444cb-0f37-407e-875f-dc4c5398b1ff.png)

# PyMySQL

流行的 MySQL（可在 [`dev.mysql.com/downloads`](http://dev.mysql.com/downloads) 获取）数据库正在逐渐发展空间功能。它支持 OGC 几何形状和一些空间函数。它还提供了 PyMySQL 库中的纯 Python API。有限的空间函数使用平面几何和边界矩形，而不是球面几何和形状。MySQL 的最新开发版本包含一些额外的函数，这些函数提高了这一功能。

在以下示例中，我们将创建一个名为 `spatial_db` 的 MySQL 数据库。然后，我们将添加一个名为 `PLACES` 的表，其中包含一个几何列。接下来，我们将添加两个城市作为点位置。最后，我们将使用 MySQL 的 `ST_Distance` 函数计算距离，并将结果从度数转换为英里。

首先，我们将导入我们的 `mysql` 库并设置数据库连接：

```py
# Import the python mysql library
import pymysql
# Establish a database connection on our local
# machine as the root database user.
conn = pymysql.connect(host='localhost', port=3306,
user='root', passwd='', db='mysql')
```

接下来，我们获取数据库游标：

```py
# Get the database cursor needed to change
# the database
cur = conn.cursor()
```

现在，我们检查数据库是否已存在，如果存在则将其删除：

```py
# If the database already exists, delete
# it and recreate it either way.
cur.execute("DROP DATABASE IF EXISTS spatial_db")
cur.execute("CREATE DATABASE spatial_db")
# Close the cursor and the connection
cur.close()
conn.close()
```

现在，我们设置一个新的连接并获取游标：

```py
# Set up a new connection and cursor
conn = pymysql.connect(host='localhost', port=3306,
user='root', passwd='', db='spatial_db')
cur = conn.cursor()
```

接下来，我们可以创建我们的新表并添加我们的字段：

```py
# Create our geospatial table
cur.execute("CREATE TABLE PLACES (id int NOT NULL
# Add name and location fields. The location
# field is spatially enabled to hold GIS data
AUTO_INCREMENT PRIMARY KEY, Name varchar(50) NOT NULL, location
Geometry NOT NULL)")
```

添加了字段后，我们就可以为一些城市的地理位置插入记录了：

```py
# Insert a name and location for the city of
# New Orleans
cur.execute("INSERT INTO PLACES (name, location) VALUES ('NEW
ORLEANS', GeomFromText('POINT(30.03 90.03)'))")
# Insert a name and location for the city of
# Memphis.
cur.execute("INSERT INTO PLACES (name, location) VALUES
('MEMPHIS', GeomFromText('POINT(35.05 90.00)'))")
```

然后，我们可以将更改提交到数据库：

```py
# Commit the changes to the database
conn.commit()
```

现在，我们可以查询数据库了！首先，我们将获取所有点位置列表：

```py
# Now let's read the data. Select all of
# the point locations from the database.
cur.execute("SELECT AsText(location) FROM PLACES")
```

现在，我们将从查询结果中提取两个点：

```py
# We know there's only two points, so we'll
# just parse them.
p1, p2 = [p[0] for p in cur.fetchall()]
```

在我们能够测量距离之前，我们需要将点列表转换为地理空间几何形状：

```py
# Now we'll convert the data
# to geometries to measure the distance
# between the two cities
cur.execute("SET @p1 = ST_GeomFromText('{}')".format(p1))
cur.execute("SET @p2 = ST_GeomFromText('{}')".format(p2))
```

最后，我们可以使用 `Distance` 存储过程来测量两个几何形状之间的距离：

```py
# Now we do the measurement function which
# is also a database query.
cur.execute("SELECT ST_Distance(@p1, @p2)")
d = float(cur.fetchone()[0])
# Print the distance as a formatted
# string object.
print("{:.2f} miles from New Orleans to Memphis".format(d *
70))
cur.close()
conn.close()
```

输出如下：

```py
351.41 miles from New Orleans to Memphis
```

其他空间数据库选项也可用，包括 PostGIS 和 SpatiaLite；然而，这些空间引擎在 Python 3 中的支持最多处于开发阶段。您可以通过 OGR 库访问 PostGIS 和 MySQL；然而，MySQL 的支持有限。

# PyFPDF

纯 Python 的 PyFPDF 库是一种创建 PDF（包括地图）的轻量级方式。由于 PDF 格式是一种广泛使用的标准，PDF 通常用于分发地图。您可以通过 PyPI 以 `fpdf` 的方式安装它。该软件的官方名称是 PyFPDF，因为它 PHP 语言模块 `fpdf` 的一部分。此模块使用一个称为单元格的概念，在页面的特定位置布局项目。作为一个快速示例，我们将从 PIL 示例中导入的 `hancock.png` 图像放入名为 `map.pdf` 的 PDF 中，以创建一个简单的 PDF 地图。地图顶部将有标题文本，说明汉考克县边界，然后是地图图像：

```py
>>> import fpdf
>>> # PDF constructor:
>>> # Portrait, millimeter units, A4 page size
>>> pdf=fpdf.FPDF("P", "mm", "A4")
>>> # create a new page
>>> pdf.add_page()
>>> # Set font: arial, bold, size 20
>>> pdf.set_font('Arial','B',20)
>>> # Layout cell: 160 x 25mm, title, no border, centered
>>> pdf.cell(160,25,'Hancock County Boundary', \
>>> border=0, align="C")
>>> # Write the image specifying the size
>>> pdf.image("hancock.png",25,50,110,160)
>>> # Save the file: filename, F = to file System
>>> pdf.output('map.pdf','F')
```

如果您在 Adobe Acrobat Reader 或其他 PDF 阅读器（如 Sumatra PDF）中打开名为 `map.pdf` 的 PDF 文件，您会看到图像现在位于 A4 页面的中心。地理空间产品通常作为更大报告的一部分，PyFPDF 模块简化了自动生成 PDF 报告的过程。

# 地理空间 PDF

**便携式文档格式**，或**PDF**，是一种存储和以跨平台和应用独立的方式呈现数字化文本和图像的文件格式。PDF 是一种广泛使用的文档格式，它也被扩展用于存储地理空间信息。

PDF 规范从 1.7 版本开始包括地理空间 PDF 的扩展，这些扩展将文档的部分映射到物理空间，也称为地理参照。您可以创建点、线或多边形作为地理空间几何形状，这些几何形状也可以有属性。

在 PDF 内部编码地理空间信息有两种方法。一家名为 TerraGo 的公司制定了一个规范，该规范已被开放地理空间联盟作为最佳实践采用，但不是一个标准。这种格式被称为**GeoPDF**。Adobe Systems 提出的扩展，即创建 PDF 规范的 ISO 32000，目前正在纳入规范的 2.0 版本。

TerraGo 的地理空间 PDF 产品符合 OGC 最佳实践文档和 Adobe PDF 扩展。但是，TerraGo 超越了这些功能，包括图层和其他 GIS 功能。然而，您必须使用 TerraGo 的 Adobe Acrobat 或其他软件的插件来访问这些功能。至少，TerraGo 支持至少在 PDF 软件中显示所需的功能。

在 Python 中，有一个名为`geopdf`的库，它与 TerraGo 无关，但支持 OGC 最佳实践。这个库最初是由 Prominent Edge 的 Tyler Garner 开发的，用于 Python 2。它已被移植到 Python 3。

从 GitHub 安装`geopdf`就像运行以下命令一样简单：

```py
pip install https://github.com/GeospatialPython/geopdf-py3/archive/master.zip
```

以下示例重新创建了我们在第一章，“使用 Python 学习地理空间分析”，在*简单 GIS*部分创建的地图，作为一个地理空间 PDF。`geopdf`库依赖于 Python 的 ReportLab PDF 库。我们需要执行的步骤如下：

1.  创建一个 PDF 绘图画布。

1.  为科罗拉多州绘制一个矩形。

1.  设置一个函数将地图坐标转换为屏幕坐标。

1.  绘制并标注城市和人口。

1.  将该州的所有角落注册为地理空间 PDF 坐标，这些坐标将整个地图进行地理参照。

Python 代码的注释解释了每一步发生了什么：

```py
# Import the geopdf library
from geopdf import GeoCanvas
# Import the necessary Reportlab modules
from reportlab.pdfbase.pdfdoc import PDFString, PDFArray
# Create a canvas with a name for our pdf.
canvas = GeoCanvas('SimpleGIS.pdf')
# Draw a rectangle to represent the State boundary
canvas.rect(100, 400, 400, 250, stroke=1)
# DATA MODEL
# All layers will have a name, 1+ points, and population count
NAME = 0
POINTS = 1
POP = 2
# Create the state layer
state = ["COLORADO", [[-109, 37], [-109, 41], [-102, 41], [-102, 37]], 5187582]
# Cities layer list
# city = [name, [point], population]
cities = []
# Add Denver
cities.append(["DENVER", [-104.98, 39.74], 634265])
# Add Boulder
cities.append(["BOULDER", [-105.27, 40.02], 98889])
# Add Durango
cities.append(["DURANGO", [-107.88, 37.28], 17069])
# MAP GRAPHICS RENDERING
map_width = 400
map_height = 250
# State Bounding Box
# Use Python min/max function to get state bounding box
minx = 180
maxx = -180
miny = 90
maxy = -90
for x, y in state[POINTS]:
    if x < minx:
        minx = x
    elif x > maxx:
        maxx = x
    if y < miny:
        miny = y
    elif y > maxy:
        maxy = y
# Get earth distance on each axis
dist_x = maxx - minx
dist_y = maxy - miny
# Scaling ratio each axis
# to map points from world to screen
x_ratio = map_width / dist_x
y_ratio = map_height / dist_y
def convert(point):
    """Convert lat/lon to screen coordinates"""
    lon = point[0]
    lat = point[1]
    x = map_width - ((maxx - lon) * x_ratio)
    y = map_height - ((maxy - lat) * y_ratio)
    # Python turtle graphics start in the middle of   
    # the screen so we must offset the points so they     
    # are centered
    x = x + 100
    y = y + 400
    return [x, y]

# Set up our map labels
canvas.setFont("Helvetica", 20)
canvas.drawString(250, 500, "COLORADO")

# Use smaller text for cities
canvas.setFont("Helvetica", 8)

# Draw points and label the cities
for city in cities:
pixel = convert(city[POINTS])
print(pixel)

# Place a point for the city
canvas.circle(pixel[0], pixel[1], 5, stroke=1, fill=1)  

# Label the city
canvas.drawString(pixel[0] + 10, pixel[1], city[NAME] + ", Population: " + str(city[POP]))

# A series of registration point pairs (pixel x, 
# pixel y, x, y) to spatially enable the PDF. We only 
# need to do the state boundary.
# The cities will be contained with in it.
registration = PDFArray([
PDFArray(map(PDFString, ['100', '400', '{}'.format(minx), '{}'.format(maxy)])),
PDFArray(map(PDFString, ['500', '400', '{}'.format(maxx), '{}'.format(maxy)])),
PDFArray(map(PDFString, ['100', '150', '{}'.format(minx), '{}'.format(miny)])),
PDFArray(map(PDFString, ['500', '150', '{}'.format(maxx), '{}'.format(miny)]))
])
# Add the map registration
canvas.addGeo(Registration=registration)
# Save our geopdf
canvas.save()
```

# Rasterio

我们在本章早期介绍的 GDAL 库功能非常强大，但它并不是为 Python 设计的。`rasterio`库通过将 GDAL 包装在一个非常简单、干净的 Pythonic API 中，解决了这个问题，用于栅格数据操作。

此示例使用本章 GDAL 示例中的卫星图像。我们将打开图像并获取一些元数据，如下所示

```py
>>> import rasterio
>>> ds = rasterio.open("SatImage.tif")
>>> ds.name
'SatImage.tif'
>>> ds.count
3
>>> ds.width
2592
>>> ds.height
2693
```

# OSMnx

`osmnx`库结合了**Open Street Map**（**OSM**）和强大的 NetworkX 库来管理用于路由的街道网络。这个库有数十个依赖项，它将这些依赖项整合起来以执行下载、分析和可视化街道网络的复杂步骤。

您可以使用`pip`尝试安装`osmnx`：

```py
pip install osmnx
```

然而，您可能会因为依赖项而遇到一些安装问题。在这种情况下，使用我们将在本章后面介绍的 Conda 系统会更容易一些。

以下示例使用`osmnx`从 OSM 下载城市的街道数据，从中创建街道网络，并计算一些基本统计数据：

```py
>>> import osmnx as ox
>>> G = ox.graph_from_place('Bay Saint Louis, MS , USA', network_type='drive')
>>> stats = ox.basic_stats(G)
>>> stats["street_length_avg"]
172.1468804611654
```

# Jupyter

当您处理地理空间或其他科学数据时，您应该了解 Jupyter 项目。Jupyter Notebook 应用程序在网页浏览器中创建和显示笔记本文档，这些文档是可读和可执行的代码和数据。它非常适合分享软件教程，并在地理空间 Python 世界中变得非常普遍。

您可以在这里找到 Jupyter Notebooks 和 Python 的良好介绍：[`jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html`](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html)。

# Conda

Conda 是一个开源的包管理系统，它使得安装和更新复杂的库变得更加容易。它与多种语言兼容，包括 Python。Conda 对于设置库和测试它们非常有用，这样我们就可以在开发环境中尝试新事物。通常，自定义配置生产环境会更好，但 Conda 是原型化新想法的绝佳方式。

您可以从[`conda.io/en/latest/`](https://conda.io/en/latest/)开始使用 Conda。

# 摘要

在本章中，我们概述了 Python 特定的地理空间分析工具。许多这些工具都包含了绑定到我们在第三章，“地理空间技术景观”中讨论的库，以提供针对特定操作（如 GDAL 的栅格访问函数）的最佳解决方案。我们还尽可能地包括了纯 Python 库，并且在我们处理即将到来的章节时将继续包括纯 Python 算法。

在下一章中，我们将开始应用所有这些工具进行 GIS 分析。

# 进一步阅读

以下链接将帮助您进一步探索本章的主题。第一个链接是关于 XPath 查询语言，我们使用 Elementree 来过滤 XML 元素。第二个链接是 Python 字符串库的文档，这在本书中对于操作数据至关重要。第三，我们有`lxml`库，这是更强大和快速的 XML 库之一。最后，我们有 Conda，它为 Python 中的科学操作提供了一个全面、易于使用的框架，包括地理空间技术：

+   想要获取 XPath 的更多信息，请查看以下链接：[`www.w3schools.com/xsl/xpath_intro.asp`](http://www.w3schools.com/xsl/xpath_intro.asp)

+   想要了解更多关于 Python `string`模块的详细信息，请查看以下链接：[`docs.python.org/3.4/library/string.html`](https://docs.python.org/3.4/library/string.html)

+   LXML 的文档可以在以下链接中找到：[`lxml.de/`](http://lxml.de/)

+   你可以在以下链接中了解更多关于 Conda 的信息：[`conda.io/en/latest/`](https://conda.io/en/latest/)
