# 地理空间Python工具箱

本书的前三章涵盖了地理空间分析的历史、分析师使用的地理空间数据类型以及地理空间行业中的主要软件和库。我们在某些地方使用了一些简单的Python示例来说明某些观点，但我们主要关注地理空间分析领域，而不考虑任何特定技术。从现在开始，我们将使用Python来征服地理空间分析，并将继续使用这种方法完成本书的其余部分。本章解释了您工具箱中所需的软件，以便在地理空间领域做几乎所有您想做的事情。

我们将发现用于访问[第2章](a7a60707-fb99-41d3-959c-7ed43a469c55.xhtml)“学习地理空间数据”中发现的矢量数据和栅格数据不同类型的Python库。其中一些库是纯Python，还有一些是我们[第3章](a5e439d1-e7fd-46b4-8fd3-8f811bfe73e4.xhtml)“地理空间技术景观”中查看的不同软件包的绑定。

在本章中，我们将涵盖以下主题：

+   安装第三方Python模块

+   Python虚拟环境

+   Conda

+   Docker

+   用于获取数据的Python网络库

+   Python基于标签的解析器

+   Python JSON库

+   OGR

+   PyShp

+   DBFPY

+   Shapely

+   GDAL

+   Fiona

+   NumPy

+   GeoPandas

+   Python图像库（PIL）

+   PNGCanvas

+   ReportLab

+   GeoPDF

+   Python NetCDF库

+   Python HDF库

+   OSMnx

+   空间索引库

+   Jupyter

+   Conda

在可能的情况下，我们将检查纯Python解决方案。Python是一种非常强大的编程语言，但某些操作，尤其是在遥感领域，计算量过于庞大，因此在使用纯Python或其他解释型语言时不太实用。幸运的是，可以通过Python以某种方式解决地理空间分析的各个方面，即使它绑定到高度高效的C/C++/其他编译语言库。

我们将避免使用覆盖地理空间分析以外的其他领域的广泛科学库，以使解决方案尽可能简单。使用Python进行地理空间分析有许多原因，但其中最强有力的论据之一是其可移植性。

此外，Python已被移植到Java作为Jython发行版，以及到.NET **公共语言运行时**（**CLR**）作为IronPython。Python还有如Stackless Python这样的版本，适用于大量并发程序。还有专为在集群计算机上运行分布式处理而设计的Python版本。Python还可在许多托管应用程序服务器上使用，这些服务器不允许您安装自定义可执行文件，例如具有Python API的Google App Engine平台。

# 技术要求

+   Python 3.6或更高版本

+   RAM：最小6 GB（Windows），推荐8 GB（macOS），建议8 GB

+   存储：最小 7200 RPM SATA，可用空间 20 GB；推荐 SSD，可用空间 40 GB

+   处理器：最小 Intel Core i3 2.5 GHz；推荐 Intel Core i5

# 安装第三方 Python 模块

使用纯 Python（使用标准库）编写的模块将在 Python 网站提到的 20 个平台中的任何一个上运行。每次你添加一个依赖于绑定到其他语言外部库的第三方模块时，你都会降低 Python 的固有可移植性。你还在代码中添加了另一层复杂性，通过添加另一种语言来彻底改变代码。纯 Python 保持简单。此外，Python 对外部库的绑定通常是由自动或半自动生成的。

这些自动生成的绑定非常通用且晦涩，它们只是通过使用该 API 的方法名将 Python 连接到 C/C++ API，而不是遵循 Python 的最佳实践。当然，也有一些值得注意的例外，这些例外是由项目需求驱动的，可能包括速度、独特的库功能或经常更新的库，在这些库中，自动生成的接口更可取。

我们将在 Python 的标准库中包含的模块和必须安装的模块之间做出区分。在 Python 中，`words` 模块和库是通用的。要安装库，你可以从 **Python 包索引（PyPI**） 获取，或者在许多地理空间模块的情况下，下载一个专门的安装程序。

PyPI 作为官方的软件仓库，提供了一些易于使用的设置程序，简化了包的安装。你可以使用 `easy_install` 程序，它在 Windows 上特别有用，或者使用在 Linux 和 Unix 系统上更常见的 `pip` 程序。一旦安装，你就可以通过运行以下代码来安装第三方包：

[PRE0]

要安装 `pip`，请运行以下代码：

[PRE1]

本书将提供不在 PyPI 上可用的开源软件包的链接和安装说明。你可以通过下载 Python 源代码并将其放入当前工作目录，或者将其放入 Python 的 `site-packages` 目录中来手动安装第三方 Python 模块。这两个目录在尝试导入模块时都可用在 Python 的搜索路径中。如果你将模块放入当前工作目录，它只会在你从该目录启动 Python 时可用。

如果您将其放在`site-packages`目录中，每次启动Python时它都将可用。`site-packages`目录专门用于第三方模块。为了定位您安装的`site-packages`目录，您需要询问Python的`sys`模块。`sys`模块有一个`path`属性，其中包含Python搜索路径中的所有目录。`site-packages`目录应该是最后一个。您可以通过指定索引`-1`来定位它，如下面的代码所示：

[PRE2]

如果该调用没有返回`site-packages`路径，只需查看整个列表以定位它，如下面的代码所示：

[PRE3]

这些安装方法将在本书的其余部分中使用。您可以在[http://python.org/download/](http://python.org/download/)找到最新的Python版本、您平台安装的源代码以及编译说明。

Python的`virtualenv`模块允许您轻松地为特定项目创建一个隔离的Python副本，而不会影响您的主Python安装或其他项目。使用此模块，您可以拥有具有相同库的不同版本的不同项目。一旦您有一个工作代码库，您就可以将其与您使用的模块或甚至Python本身的变化隔离开来。`virtualenv`模块简单易用，可以用于本书中的任何示例；然而，关于其使用的明确说明并未包含。

要开始使用`virtualenv`，请遵循以下简单指南：[http://docs.python-guide.org/en/latest/dev/virtualenvs/](http://docs.python-guide.org/en/latest/dev/virtualenvs/)。

# Python虚拟环境

Python地理空间分析需要我们使用许多具有许多依赖关系的模块。这些模块通常使用特定版本的C或C++库相互构建。当您向系统中添加Python模块时，经常会遇到版本冲突。有时，当您升级特定模块时，由于API的变化，它可能会破坏您现有的Python程序——或者您可能同时运行Python 2和Python 3以利用为每个版本编写的库。您需要的是一种安全安装新模块的方法，而不会破坏工作系统或代码。解决这个问题的方法是使用`virtualenv`模块的Python虚拟环境。

Python的`virtualenv`模块为每个项目创建隔离的、独立的Python环境，这样您就可以避免冲突的模块污染您的主Python安装。您可以通过激活或停用特定环境来打开或关闭该环境。`virtualenv`模块在效率上很高，因为它在创建环境时实际上并不复制您整个系统Python安装。让我们开始吧：

1.  安装`virtualenv`就像运行以下代码一样简单：

[PRE4]

1.  然后，为您的虚拟Python环境创建一个目录。命名它 whatever you want：

[PRE5]

1.  现在，您可以使用以下命令创建您的第一个虚拟环境：

[PRE6]

1.  然后，在输入以下命令后，你可以激活该环境：

[PRE7]

1.  现在，当你在这个目录中运行任何 Python 命令时，它将使用隔离的虚拟环境。当你完成时，你可以使用以下简单的命令来停用该环境：

[PRE8]

这就是安装、激活以供使用以及停用 `virtualenv` 模块的方法。然而，你还需要了解另一个环境。我们将在下一节中检查它。

# Conda

在这里也值得提一下 Conda，它是一个开源的、跨平台的包管理系统，也可以创建和管理类似于 `virtualenv` 的环境。Conda 使得安装复杂的包变得容易，包括地理空间包。它还支持 Python 之外的其他语言，包括 R、Node.js 和 Java。

Conda 可在此处找到：[https://docs.conda.io/en/latest/](https://docs.conda.io/en/latest/)。

现在，让我们来看看如何安装 GDAL，这样我们就可以开始处理地理空间数据了。

# 安装 GDAL

**地理空间数据抽象库**（**GDAL**），包括 OGR，对于本书中的许多示例至关重要，也是更复杂的 Python 设置之一。因此，我们将在这里单独讨论它。最新的 GDAL 绑定可在 PyPI 上找到；然而，由于 GDAL 库需要额外的资源，安装需要额外的步骤。

有三种方法可以安装 GDAL 以用于 Python。你可以使用其中任何一种：

+   从源代码编译它。

+   作为更大软件包的一部分安装它。

+   安装二进制发行版，然后安装 Python 绑定。

如果你也有编译 C 库以及所需编译软件的经验，那么第一个选项会给你最大的控制权。然而，如果你只是想尽快开始，那么这个选项并不推荐，因为即使是经验丰富的软件开发者也会发现编译 GDAL 和相关的 Python 绑定具有挑战性。在主要平台上的 GDAL 编译说明可以在 [http://trac.osgeo.org/gdal/wiki/BuildHints](http://trac.osgeo.org/gdal/wiki/BuildHints) 找到；PyPI GDAL 页面上也有基本的构建说明；请查看 [https://pypi.python.org/pypi/GDAL](https://pypi.python.org/pypi/GDAL)。

第二个选项无疑是最快和最简单的方法。**开源地理空间基金会**（**OSGeo**）分发了一个名为 OSGeo4W 的安装程序，只需点击一下按钮即可在 Windows 上安装所有顶级开源地理空间包。OSGeo4W 可在 [http://trac.osgeo.org/osgeo4w/](http://trac.osgeo.org/osgeo4w/) 找到。

虽然这些包最容易使用，但它们带有自己的 Python 版本。如果你已经安装了 Python，那么仅为了使用某些库就安装另一个 Python 发行版可能会出现问题。在这种情况下，第三个选项可能适合你。

第三个选项安装了针对您的Python版本预编译的二进制文件。这种方法在安装简便性和定制之间提供了最佳折衷。但是，您必须确保二进制发行版和相应的Python绑定彼此兼容，与您的Python版本兼容，并且在许多情况下与您的操作系统配置兼容。

# Windows

每年，Windows上Python的GDAL安装都变得越来越容易。要在Windows上安装GDAL，您必须检查您是否正在运行32位或64位版本的Python：

1.  要这样做，只需在命令提示符中启动Python解释器，如下面的代码所示：

[PRE9]

1.  基于此实例，我们可以看到Python版本为3.4.2的`win32`，这意味着它是32位版本。一旦您有了这些信息，请访问以下URL：[http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)。

1.  这个网页包含了适用于几乎所有开源科学库的Python Windows二进制文件和绑定。在该网页的GDAL部分，找到与您的Python版本匹配的版本。版本名称使用C Python的缩写`cp`，后跟主要的Python版本号，以及32位Windows的`win32`或64位Windows的`win_amd64`。

在前面的例子中，我们会下载名为`GDAL-1.11.3-cp34-none-win32.whl`的文件。

1.  此下载包是较新的Python `pip` wheel格式。要安装它，只需打开命令提示符并输入以下代码：

[PRE10]

1.  一旦安装了包，打开Python解释器并运行以下命令，通过检查其版本来验证GDAL是否已安装：

[PRE11]

现在，GDAL应该返回版本号`1.11.3`。

如果您在使用`easy_install`或`pip`和PyPI安装模块时遇到问题，请尝试从与GDAL示例相同的网站下载并安装wheel包。

# Linux

Linux上的GDAL安装因发行版而异。以下[https://gdal.org](https://gdal.org)二进制网页列出了几个发行版的安装说明：[http://trac.osgeo.org/gdal/wiki/DownloadingGdalBinaries](http://trac.osgeo.org/gdal/wiki/DownloadingGdalBinaries)。让我们开始吧：

1.  通常，您的包管理器会安装GDAL和Python绑定。例如，在Ubuntu上，要安装GDAL，您需要运行以下代码：

[PRE12]

1.  然后，要安装Python绑定，您可以运行以下命令：

[PRE13]

1.  大多数Linux发行版已经配置好了编译软件，它们的说明比Windows上的简单得多。

1.  根据安装情况，您可能需要将`gdal`和`ogr`作为`osgeo`包的一部分导入，如下面的命令所示：

[PRE14]

# macOS X

要在macOS X上安装GDAL，您还可以使用Homebrew包管理系统，该系统可在[http://brew.sh/](http://brew.sh/)找到。

或者，您可以使用 MacPorts 软件包管理系统，该系统可在 [https://www.macports.org/](https://www.macports.org/) 获取。

这两个系统都有很好的文档记录，并包含适用于 Python 3 的 GDAL 包。你实际上只需要它们用于需要正确编译的二进制文件（用 C 语言编写，具有许多依赖项并包含许多科学和地理空间库）的库。

# Python 网络库用于获取数据

大多数地理空间数据共享都是通过互联网完成的，Python 在处理几乎任何协议的网络库方面都准备得很充分。自动数据下载通常是自动化地理空间过程的一个重要步骤。数据通常从网站的 **统一资源定位符** (**URL**) 或 **文件传输协议** (**FTP**) 服务器检索，由于地理空间数据集通常包含多个文件，因此数据通常以 ZIP 文件的形式分发。

Python 的一个优点是其文件类似对象的概念。大多数用于读取和写入数据的 Python 库都使用一组标准方法，允许您从不同类型的资源访问数据，就像您在磁盘上写入一个简单的文件一样。Python 标准库中的网络模块也使用这种约定。这种方法的优点是它允许您将文件类似对象传递给其他库和方法，这些库和方法可以识别该约定，而无需为以不同方式分发的不同类型的数据进行大量设置。

# Python 的 urllib 模块

Python 的 `urllib` 包旨在简单访问任何具有 URL 地址的文件。Python 3 中的 `urllib` 包由几个模块组成，这些模块处理管理网络请求和响应的不同部分。这些模块实现了 Python 的一些文件类似对象约定，从其 `open()` 方法开始。当你调用 `open()` 时，它会准备与资源的连接，但不会访问任何数据。有时，你只想获取一个文件并将其保存到磁盘上，而不是将其加载到内存中。这个功能可以通过 `urllib.request.retrieve()` 方法获得。

以下示例使用 `urllib.request.retrieve()` 方法下载名为 `hancock.zip` 的压缩形状文件，该文件在其他示例中使用。我们定义了 URL 和本地文件名作为变量。URL 作为参数传递，以及我们想要使用的文件名，以将其保存到我们的本地机器上，在这种情况下，只是 `hancock.zip`：

[PRE15]

来自底层`httplib`模块的消息确认文件已下载到当前目录。URL和文件名也可以直接作为字符串传递给`retrieve()`方法。如果你只指定文件名，下载将保存到当前工作目录。你也可以指定一个完全限定的路径名来将其保存到其他位置。你还可以指定一个回调函数作为第三个参数，该函数将接收文件的下载状态信息，这样你就可以创建一个简单的下载状态指示器或执行其他操作。

`urllib.request.urlopen()`方法允许你以更高的精度和控制访问在线资源。正如我们之前提到的，它实现了大多数Python文件类似对象方法，除了`seek()`方法，它允许你在文件中的任意位置跳转。你可以逐行读取在线文件，将所有行作为列表读取，读取指定数量的字节，或者遍历文件的每一行。所有这些功能都在内存中执行，因此你不需要将数据存储在磁盘上。这种能力对于访问可能需要处理但不想保存到磁盘上的在线频繁更新的数据非常有用。

在下面的示例中，我们通过访问**美国地质调查局**（**USGS**）地震源来演示这个概念，查看在过去一小时内发生的所有地震。这些数据以**逗号分隔值**（**CSV**）文件的形式分发，我们可以像文本文件一样逐行读取。CSV文件类似于电子表格，可以在文本编辑器或电子表格程序中打开：

1.  首先，你需要打开URL并读取包含文件列名的标题。

1.  然后，你需要读取第一行，其中包含最近地震的记录，如下面的代码行所示：

[PRE16]

1.  我们也可以遍历这个文件，这是一种读取大文件的内存高效方式。

1.  如果你在这个Python解释器中运行这个示例，你需要按下*Enter*或*return*键两次来执行循环。这个动作是必要的，因为它向解释器发出信号，表明你已经完成了循环的构建。在下面的示例中，我们简化了输出：

[PRE17]

# Python请求模块

`urllib`模块已经存在很长时间了。另一个第三方模块已经被开发出来，使得常见的HTTP请求更加容易。`requests`模块具有以下功能：

+   保持连接和连接池

+   国际域名和URL

+   会话中保持cookie持久性

+   浏览器风格的SSL验证

+   自动内容解码

+   基本摘要认证

+   精美的键/值cookie

+   自动解压缩

+   Unicode响应体

+   HTTP(S)代理支持

+   多部分文件上传

+   流式下载

+   连接超时

+   分块请求

+   `.netrc`支持

在下面的例子中，我们将下载与使用`urllib`模块下载的相同ZIP文件，但这次我们将使用`requests`模块。首先，我们需要安装`requests`模块：

[PRE18]

然后，我们可以导入它：

[PRE19]

然后，我们可以设置URL和输出文件名的变量：

[PRE20]

使用`requests`模块的`get()`方法检索ZIP文件非常简单：

[PRE21]

现在，我们可以从`.zip`文件中获取内容并将其写入我们的输出文件：

[PRE22]

`requests`模块还有许多其他高级功能，使用起来与这个例子一样简单。现在我们知道了如何通过HTTP协议获取信息，让我们来检查FTP协议，它通常用于从在线存档访问地理空间数据。

# FTP

FTP允许你使用FTP客户端软件浏览在线目录并下载数据。直到大约2004年，当地理空间网络服务变得非常普遍之前，FTP是分发地理空间数据最常见的方式之一。现在FTP不太常见，但在搜索数据时偶尔会遇到它。再次强调，Python的内置标准库有一个名为`ftplib`的合理FTP模块，其主要类名为`FTP()`。

在下面的例子中，我们将执行以下操作：

1.  我们将访问由美国**国家海洋和大气管理局**（**NOAA**）托管的FTP服务器，以访问包含全球海啸监测网络**深海评估和报告**（**DART**）浮标数据的文本文件。这个特定的浮标位于秘鲁海岸。

1.  我们将定义服务器和目录路径，然后我们将访问服务器。所有FTP服务器都需要用户名和密码。大多数公共服务器都有一个名为anonymous的用户，密码也是anonymous，就像这个服务器一样。

1.  使用Python的`ftplib`，你可以不带任何参数调用`login()`方法以默认匿名用户身份登录。否则，你可以添加用户名和密码作为字符串参数。

1.  登录后，我们将切换到包含DART数据文件的目录。

1.  要下载文件，我们将打开一个名为out的本地文件，并将它的`write()`方法作为回调函数传递给`ftplib.ftp.retrbinary()`方法，该方法同时下载文件并将其写入我们的本地文件。

1.  文件下载完成后，我们可以关闭它以保存它。

1.  然后，我们将读取文件并查找包含浮标纬度和经度的行，以确保数据已成功下载，如下面的代码行所示：

[PRE23]

输出如下：

[PRE24]

在这个例子中，我们以二进制写入模式打开了本地文件，并使用了`retrbinary()``ftplib`方法，而不是使用ASCII模式的`retrlines()`。二进制模式适用于ASCII和二进制文件，因此始终是一个更安全的赌注。实际上，在Python中，文件的二进制读写模式仅在Windows上需要。

如果你只是从FTP服务器下载一个简单的文件，许多FTP服务器也有一个网络界面。在这种情况下，你可以使用`urllib`来读取文件。FTP URL使用以下格式来访问数据：

[PRE25]

这种格式对于密码保护的目录来说是不安全的，因为你正在通过互联网传输你的登录信息。但对于匿名FTP服务器来说，没有额外的安全风险。为了演示这一点，以下示例通过使用`urllib`而不是`ftplib`来访问我们刚刚看到的相同文件：

[PRE26]

现在我们已经可以下载文件了，让我们学习如何解压缩它们。

# ZIP和TAR文件

地理空间数据集通常由多个文件组成。因此，它们通常以ZIP或TAR文件归档的形式分发。这些格式也可以压缩数据，但它们捆绑多个文件的能力是它们被用于地理空间数据的主要原因。虽然TAR格式不包含压缩算法，但它结合了gzip压缩，并将其作为程序选项提供。Python有用于读取和写入ZIP和TAR归档的标准模块。这些模块分别称为`zipfile`和`tarfile`。

以下示例使用`urllib`下载的`hancock.zip`文件中的`hancock.shp`、`hancock.shx`和`hancock.dbf`文件，用于在之前的示例中使用。此示例假定ZIP文件位于当前目录中：

[PRE27]

这个例子比必要的更详细，为了清晰起见。我们可以通过在`zipfile.namelist()`方法周围使用`for`循环来缩短这个例子，并使其更健壮，而不必明确地将不同的文件定义为变量。这种方法是一种更灵活且更Pythonic的方法，可以用于具有未知内容的ZIP归档，如下面的代码行所示：

[PRE28]

现在你已经了解了`zipfile`模块的基础知识，让我们用我们刚刚解压的文件创建一个TAR归档。在这个例子中，当我们打开TAR归档进行写入时，我们指定写入模式为`w:gz`以进行gzip压缩。我们还指定文件扩展名为`tar.gz`以反映这种模式，如下面的代码行所示：

[PRE29]

我们可以使用简单的`tarfile.extractall()`方法提取文件。首先，我们使用`tarfile.open()`方法打开文件，然后提取它，如下面的代码行所示：

[PRE30]

我们将通过结合本章学到的元素以及[第2章](a7a60707-fb99-41d3-959c-7ed43a469c55.xhtml)“学习地理空间数据”中向量数据部分的元素来工作一个额外的示例。我们将从`hancock.zip`文件中读取边界框坐标，而无需将其保存到磁盘。我们将使用Python的文件-like对象约定来传递数据。然后，我们将使用Python的`struct`模块来读取边界框，就像我们在[第2章](a7a60707-fb99-41d3-959c-7ed43a469c55.xhtml)“学习地理空间数据”中所做的那样。

在这种情况下，我们将未压缩的`.shp`文件读入一个变量，并通过指定数据起始和结束索引（由冒号`:`分隔）使用Python数组切片来访问数据。我们能够使用列表切片，因为Python允许您将字符串视为列表。在这个例子中，我们还使用了Python的`StringIO`模块，以文件对象的形式在内存中临时存储数据，该对象实现了包括`seek()`方法在内的各种方法，而`seek()`方法在大多数Python网络模块中是缺失的，如下面的代码行所示：

[PRE31]

如您从迄今为止的示例中看到的，Python的标准库包含了很多功能。大多数时候，您不需要下载第三方库就能访问在线文件。

# Python标记和基于标签的解析器

基于标签的数据，尤其是不同的XML方言，已经成为分发地理空间数据的一种非常流行的方式。既适合机器阅读又适合人类阅读的格式通常易于处理，尽管它们为了可用性牺牲了存储效率。这些格式对于非常大的数据集可能难以管理，但在大多数情况下工作得很好。

尽管大多数格式都是某种形式的XML（例如KML或GML），但有一个显著的例外。**已知文本**（**WKT**）格式相当常见，但使用外部标记和方括号（`[]`）来包围数据，而不是像XML那样使用尖括号包围数据。

Python对XML有标准库支持，还有一些优秀的第三方库可用。所有合适的XML格式都遵循相同的结构，因此您可以使用通用的XML库来读取它。因为XML是基于文本的，所以通常很容易将其作为字符串编写，而不是使用XML库。大多数输出XML的应用程序都是这种方式。

使用XML库编写XML的主要优势是您的输出通常经过验证。在创建自己的XML格式时，很容易出错。单个缺失的引号就可以使XML解析器崩溃，并给试图读取您数据的人抛出错误。当这些错误发生时，它们几乎使您的数据集变得无用。您会发现这个问题在基于XML的地理空间数据中非常普遍。您会发现一些解析器对不正确的XML比其他解析器更宽容。通常，可靠性比速度或内存效率更重要。

在[http://lxml.de/performance.html](http://lxml.de/performance.html)提供的分析中，提供了不同Python XML解析器在内存和速度方面的基准。

# minidom模块

Python的`minidom`模块是一个非常古老且易于使用的XML解析器。它是Python XML包内建的一组XML工具的一部分。它可以解析XML文件或作为字符串输入的XML。`minidom`模块最适合小于约20 MB的小到中等大小的XML文档，因为在此之前的速度开始下降。

为了演示 `minidom` 模块，我们将使用一个示例 KML 文件，这是 Google KML 文档的一部分，你可以下载。以下链接中的数据代表从 GPS 设备传输的时间戳点位置：[https://github.com/GeospatialPython/Learn/raw/master/time-stamp-point.kml](https://github.com/GeospatialPython/Learn/raw/master/time-stamp-point.kml)。让我们开始吧：

1.  首先，我们将通过从文件中读取数据并创建一个 `minidom` 解析器对象来解析这些数据。文件包含一系列 `<Placemark>` 标签，这些标签包含一个点和收集该点的时间戳。因此，我们将获取文件中所有 `Placemarks` 的列表，并且可以通过检查该列表的长度来计数，如下面的代码行所示：

[PRE32]

1.  如你所见，我们检索了所有的 `Placemark`，总数为 `361`。现在，让我们看看列表中的第一个 `Placemark` 元素：

[PRE33]

现在，每个 `<Placemark>` 标签都是一个 DOM 元素数据类型。为了真正看到这个元素是什么，我们调用 `toxml()` 方法，如下所示：

[PRE34]

1.  `toxml()` 函数将 `Placemark` 标签内包含的所有内容输出为一个字符串对象。如果我们想将此信息打印到文本文件中，我们可以调用 `toprettyxml()` 方法，这将添加额外的缩进来使 XML 更易于阅读。

1.  现在，如果我们只想从这个 placemark 中获取坐标会怎样？坐标隐藏在 `coordinates` 标签中，该标签位于 `point` 标签内，嵌套在 `Placemark` 标签内。`minidom` 对象的每个元素都称为 **节点**。嵌套节点称为子节点或子节点。子节点不仅包括标签，还包括分隔标签的空白，以及标签内的数据。因此，我们可以通过标签名称钻到 `coordinates` 标签，但之后我们需要访问 `data` 节点。所有的 `minidom` 元素都有 `childNodeslist`，以及一个 `firstChild()` 方法来访问第一个节点。

1.  我们将结合这些方法来获取第一个坐标的 `data` 节点的 `data` 属性，我们使用列表中的索引 `0` 来引用坐标标签列表：

[PRE35]

如果你刚接触 Python，你会注意到这些示例中的文本输出被标记为字母 `u`。这种标记是 Python 表示支持国际化并使用不同字符集的多语言 Unicode 字符串的方式。Python 3.4.3 对此约定略有改变，不再标记 Unicode 字符串，而是用 `b` 标记 UTF-8 字符串。

1.  我们可以更进一步，将这个 `point` 字符串转换为可用的数据，通过分割字符串并将结果字符串转换为 Python 浮点类型来实现，如下所示：

[PRE36]

1.  使用 Python 列表推导式，我们可以一步完成这个操作，如下面的代码行所示：

[PRE37]

这个例子只是触及了`minidom`库所能做到的一小部分。关于这个库的精彩教程，请查看以下教程：[https://www.edureka.co/blog/python-xml-parser-tutorial/](https://www.edureka.co/blog/python-xml-parser-tutorial/)。

# ElementTree

`minidom`模块是纯Python编写，易于使用，自Python 2.0以来一直存在。然而，Python 2.5在标准库中增加了一个更高效但更高级的XML解析器，称为`ElementTree`。`ElementTree`很有趣，因为它已经实现了多个版本。

有一个纯Python版本和一个用C编写的更快版本，称为`cElementTree`。你应该尽可能使用`cElementTree`，但可能你所在的平台不包括基于C的版本。当你导入`cElementTree`时，你可以测试它是否可用，并在必要时回退到纯Python版本：

[PRE38]

`ElementTree`的一个伟大特性是它实现了XPath查询语言的一个子集。XPath代表XML Path，允许你使用路径式语法搜索XML文档。如果你经常处理XML，学习XPath是必不可少的。你可以在以下链接中了解更多关于XPath的信息：[https://www.w3schools.com/xml/xpath_intro.asp](https://www.w3schools.com/xml/xpath_intro.asp)。

这个特性的一个问题是，如果文档指定了命名空间，就像大多数XML文档一样，你必须将那个命名空间插入到查询中。`ElementTree`不会自动为你处理命名空间。你的选择是手动指定它或尝试从根元素的标签名中通过字符串解析提取它。

我们将使用`ElementTree`重复`minidomXML`解析示例：

1.  首先，我们将解析文档，然后手动定义KML命名空间；稍后，我们将使用XPath表达式和`find()`方法来查找第一个`Placemark`元素。

1.  最后，我们将找到坐标和子节点，然后获取包含纬度和经度的文本。

在这两种情况下，我们都可以直接搜索`coordinates`标签。但是，通过获取`Placemark`元素，它给我们提供了选择，稍后如果需要的话，可以获取相应的timestamp子元素，如下面的代码所示：

[PRE39]

在这个例子中，请注意我们使用了Python字符串格式化语法，它基于C中的字符串格式化概念。当我们为placemark变量定义XPath表达式时，我们使用了`%`占位符来指定字符串的插入。然后，在字符串之后，我们使用了`%`运算符后跟变量名来在占位符处插入`ns`命名空间变量。在`coordinates`变量中，我们使用了`ns`变量两次，因此在字符串之后指定了包含两次`ns`的元组。

字符串格式化是 Python 中一种简单但极其强大且有用的工具，值得学习。你可以在以下链接中找到更多信息：[https://docs.python.org/3.4/library/string.html](https://docs.python.org/3.4/library/string.html)。

# 使用 ElementTree 和 Minidom 构建 XML

大多数时候，XML 可以通过连接字符串来构建，如下面的命令所示：

[PRE40]

然而，这种方法很容易出错，这会创建无效的 XML 文档。一种更安全的方法是使用 XML 库。让我们使用 `ElementTree` 构建这个简单的 KML 文档：

1.  我们将定义 `rootKML` 元素并为其分配一个命名空间。

1.  然后，我们将系统地添加子元素到根元素，将元素包装为 `ElementTree` 对象，声明 XML 编码，并将其写入名为 `placemark.xml` 的文件中，如下面的代码行所示：

[PRE41]

输出与之前的字符串构建示例相同，但 `ElementTree` 不缩进标签，而是将其作为一条长字符串写入。`minidom` 模块具有类似的接口，这在 Mark Pilgrim 的《深入 Python》一书中有所记录，该书籍在前面看到的 `minidom` 示例中有所引用。

XML 解析器，如 `minidom` 和 `ElementTree`，在格式完美的 XML 文档上工作得非常好。不幸的是，绝大多数的 XML 文档并不遵循这些规则，并包含格式错误或无效字符。你会发现你经常被迫处理这些数据，并且必须求助于非常规的字符串解析技术来获取你实际需要的少量数据。但是，多亏了 Python 和 Beautiful Soup，你可以优雅地处理糟糕的甚至极差的基于标签的数据。

Beautiful Soup 是一个专门设计用来鲁棒地处理损坏的 XML 的模块。它面向 HTML，HTML 以格式错误而闻名，但也适用于其他 XML 方言。Beautiful Soup 可在 PyPI 上找到，因此可以使用 `easy_install` 或 `pip` 来安装它，如下面的命令所示：

[PRE42]

或者，你可以执行以下命令：

[PRE43]

然后，为了使用它，你只需简单地导入它：

[PRE44]

为了尝试它，我们将使用来自智能手机应用程序的 **GPS 交换格式**（**GPX**）跟踪文件，该文件存在故障并导出了略微损坏的数据。您可以从以下链接下载此样本文件：[https://raw.githubusercontent.com/GeospatialPython/Learn/master/broken_data.gpx](https://raw.githubusercontent.com/GeospatialPython/Learn/master/broken_data.gpx)。

这个2,347行的数据文件处于原始状态，除了它缺少一个关闭的`</trkseg>`标签，这个标签应该位于文件的末尾，就在关闭的`</trk>`标签之前。这个错误是由源程序中的数据导出函数引起的。这个缺陷很可能是原始开发者手动生成导出时的GPX XML并忘记添加此关闭标签的代码行所导致的。看看如果我们尝试用`minidom`解析这个文件会发生什么：

[PRE45]

如您从错误信息的最后一行所看到的，`minidom`中的底层XML解析器确切地知道问题所在——文件末尾存在一个`mismatched`标签。然而，它拒绝做任何更多的事情，只是报告了错误。您必须拥有完美无瑕的XML，或者根本不使用，以避免这种情况。

现在，让我们尝试使用相同数据的更复杂和高效的`ElementTree`模块：

[PRE46]

如您所见，不同的解析器面临相同的问题。在地理空间分析中，格式不良的XML是一个过于常见的现实，每个XML解析器都假设世界上所有的XML都是完美的，除了一个。这就是Beautiful Soup的用武之地。这个库毫不犹豫地将不良XML撕成可用的数据，并且它可以处理比缺失标签更严重的缺陷。即使缺少标点符号或其他语法，它也能正常工作，并给出它能给出的最佳数据。它最初是为解析HTML而开发的，HTML因其格式不良而臭名昭著，但它与XML也相当兼容，如下所示：

[PRE47]

Beautiful Soup没有任何抱怨！为了确保数据实际上是可以使用的，让我们尝试访问一些数据。Beautiful Soup的一个令人惊叹的特性是它将标签转换为解析树的属性。如果有多个具有相同名称的标签，它将获取第一个。我们的样本数据文件有数百个`<trkpt>`标签。让我们访问第一个：

[PRE48]

我们现在确信数据已经被正确解析，并且我们可以访问它。如果我们想访问所有的`<trkpt>`标签，我们可以使用`findAll()`方法来获取它们，然后使用内置的Python `len()`函数来计数，如下所示：

[PRE49]

如果我们将解析的数据写回到文件中，Beautiful Soup会输出修正后的版本。我们将使用Beautiful Soup模块的`prettify()`方法将固定数据保存为一个新的GPX文件，以格式化XML并添加漂亮的缩进，如下面的代码行所示：

[PRE50]

Beautiful Soup是一个非常丰富的库，具有更多功能。要进一步探索它，请访问在线的Beautiful Soup文档[http://www.crummy.com/software/BeautifulSoup/bs4/documentation.html](http://www.crummy.com/software/BeautifulSoup/bs4/documentation.html)。

虽然`minidom`、`ElementTree`和`cElementTree`是Python标准库的一部分，但还有一个更强大、更受欢迎的Python XML库，称为`lxml`。`lxml`模块通过`ElementTree` API提供了对`libxml2`和`libxslt` C库的Pythonic接口。更好的事实是，`lxml`还可以与Beautiful Soup一起解析基于标签的坏数据。在某些安装中，`beautifulsoup4`可能需要`lxml`。`lxml`模块可通过PyPI获取，但需要为C库执行一些额外步骤。更多信息可在以下链接的`lxml`主页上找到：[http://lxml.de/](http://lxml.de/)。

# Well-Known Text (WKT)

WKT格式已经存在多年，是一种简单的基于文本的格式，用于表示几何形状和空间参考系统。它主要用作实现OGC Simple Features for SQL规范的系统的数据交换格式。请看以下多边形的WKT表示示例：

[PRE51]

目前，读取和写入WKT的最佳方式是使用Shapely库。Shapely提供了一个非常Python导向或Pythonic的接口，用于我们[第3章](a5e439d1-e7fd-46b4-8fd3-8f811bfe73e4.xhtml)中描述的**Geometry Engine - Open Source**（**GEOS**）库，*地理空间技术景观*。

您可以使用`easy_install`或`pip`安装Shapely。您还可以使用上一节中提到的网站上的wheel。Shapely有一个WKT模块，可以加载和导出这些数据。让我们使用Shapely加载之前的多边形样本，然后通过计算其面积来验证它是否已作为多边形对象加载：

[PRE52]

我们可以通过简单地调用其`wkt`属性，将任何Shapely几何形状转换回WKT，如下所示：

[PRE53]

Shapely还可以处理WKT的二进制对应物，称为W**ell-Known Binary**（**WKB**），它用于在数据库中以二进制对象的形式存储WKT字符串。Shapely使用其`wkb`模块以与`wkt`模块相同的方式加载WKB，并且可以通过调用该对象的`wkb`属性来转换几何形状。

Shapely是处理WKT数据最Pythonic的方式，但您也可以使用OGR库的Python绑定，这是我们本章早些时候安装的。

对于这个例子，我们将使用一个包含一个简单多边形的shapefile，它可以作为一个ZIP文件下载。您可以通过以下链接获取：[https://github.com/GeospatialPython/Learn/raw/master/polygon.zip](https://github.com/GeospatialPython/Learn/raw/master/polygon.zip)。

在以下示例中，我们将打开shapefile数据集中的`polygon.shp`文件，调用所需的`GetLayer()`方法，获取第一个（也是唯一一个）要素，然后将其导出为WKT格式：

[PRE54]

注意，使用OGR时，你必须逐个读取每个要素并单独导出它，因为`ExporttoWkt()`方法是在要素级别。现在我们可以使用包含导出的`wkt`变量来读取WKT字符串。我们将将其导入`ogr`，并获取多边形的边界框，也称为包围盒，如下所示：

[PRE55]

Shapely和OGR用于读取和写入有效的WKT字符串。当然，就像XML一样，它也是文本，你可以在必要时将少量WKT作为字符串进行操作。接下来，我们将探讨一个在地理空间领域变得越来越常见的现代文本格式。

# Python JSON库

**JavaScript对象表示法**（**JSON**）正在迅速成为许多领域的首选数据交换格式。轻量级的语法以及它与JavaScript数据结构的相似性，使得它非常适合Python，Python也借鉴了一些数据结构。

以下GeoJSON样本文档包含一个单独的点：

[PRE56]

这个样本只是一个带有新属性的基本点，这些属性将被存储在几何的属性数据结构中。在前面的例子中，ID、坐标和CRS信息将根据你的特定数据集而变化。

让我们使用Python修改这个示例GeoJSON文档。首先，我们将样本文档压缩成一个字符串，以便更容易处理：

[PRE57]

现在，我们可以使用前面代码中创建的GeoJSON `jsdata` 字符串变量，在以下示例中使用。

# json模块

GeoJSON看起来非常类似于Python的嵌套字典和列表集合。为了好玩，让我们尝试使用Python的`eval()`函数将其解析为Python代码：

[PRE58]

哇！这成功了！我们只需一步就将那个随机的GeoJSON字符串转换成了原生的Python数据。记住，JSON数据格式基于JavaScript语法，这恰好与Python相似。此外，随着你对GeoJSON数据的深入了解以及处理更大的数据，你会发现JSON允许使用Python不允许的字符。使用Python的`eval()`函数也被认为是非常不安全的。但就保持简单而言，这已经是最简单的方法了！

由于Python追求简单，更高级的方法并没有变得更加复杂。让我们使用Python的`json`模块，它是标准库的一部分，将相同的字符串正确地转换为Python：

[PRE59]

作为旁注，在先前的例子中，CRS84属性是常见WGS84坐标系统的同义词。`json`模块添加了一些很好的功能，例如更安全的解析和将字符串转换为Unicode。我们可以以几乎相同的方式将Python数据结构导出为JSON：

[PRE60]

当你导出数据时，它会以一长串难以阅读的字符串形式输出。我们可以通过传递`dumps()`方法一个缩进值来打印数据，使其更容易阅读：

[PRE61]

现在我们已经了解了 `json` 模块，让我们看看地理空间版本 `geojson`。

# geojson 模块

我们可以愉快地继续使用 `json` 模块读取和写入 GeoJSON 数据，但还有更好的方法。PyPI 上可用的 `geojson` 模块提供了一些独特的优势。首先，它了解 GeoJSON 规范的要求，这可以节省大量的输入。让我们使用此模块创建一个简单的点并将其导出为 GeoJSON：

[PRE62]

这次，当我们转储 JSON 数据以供查看时，我们将添加一个缩进参数，其值为 `4`，这样我们就可以得到格式良好的缩进 JSON 数据，更容易阅读：

[PRE63]

我们的结果如下：

[PRE64]

注意到 `geojson` 模块为不同的数据类型提供了一个接口，并使我们免于手动设置类型和坐标属性。现在，想象一下如果你有一个具有数百个要素的地理对象。你可以通过编程构建这个数据结构，而不是构建一个非常大的字符串。

`geojson` 模块也是 Python `geo_interface` 规范的参考实现。这个接口允许协作程序以 Pythonic 的方式无缝交换数据，而无需程序员显式导出和导入 GeoJSON 字符串。因此，如果我们想将使用 `geojson` 模块创建的点传递给 Shapely 模块，我们可以执行以下命令，该命令将 `geojson` 模块的点对象直接读取到 Shapely 中，然后将其导出为 WKT：

[PRE65]

越来越多的地理空间 Python 库正在实现 `geojson` 和 `geo_interface` 功能，包括 PyShp、Fiona、Karta 和 ArcGIS。对于 QGIS，存在第三方实现。

GeoJSON 是一种简单且易于人类和计算机阅读的文本格式。现在，我们将查看一些二进制矢量格式。

# OGR

我们提到了 OGR 作为处理 WKT 字符串的方法，但它的真正力量在于作为一个通用的矢量库。本书力求提供纯 Python 解决方案，但没有单个库甚至接近 OGR 可以处理的格式多样性。

让我们使用 OGR Python API 读取一个示例点形状文件。示例形状文件可以作为 ZIP 文件在此处下载：[https://github.com/GeospatialPython/Learn/raw/master/point.zip](https://github.com/GeospatialPython/Learn/raw/master/point.zip)。

这个点形状文件有五个带有单个数字、正坐标的点。属性列表了点的创建顺序，这使得它在测试中非常有用。这个简单的例子将读取点形状文件，并遍历每个要素；然后，它将打印每个点的 *x* 和 *y* 值，以及第一个属性字段的值：

[PRE66]

这个例子很简单，但 OGR 在脚本变得更加复杂时可能会变得相当冗长。接下来，我们将看看处理形状文件的一种更简单的方法。

# PyShp

PyShp是一个简单的纯Python库，用于读取和写入形状文件。它不执行任何几何操作，仅使用Python的标准库。它包含在一个易于移动、压缩到小型嵌入式平台和修改的单个文件中。它也与Python 3兼容。它还实现了`__geo_interface__`。PyShp模块可在PyPI上找到。

让我们用PyShp重复之前的OGR示例：

[PRE67]

# dbfpy

OGR和PyShp都读取和写入`.dbf`文件，因为它们是形状文件规范的一部分。`.dbf`文件包含形状文件的属性和字段。然而，这两个库对`.dbf`的支持非常基础。偶尔，你可能需要进行一些重型的DBF工作。`dbfpy3`模块是一个专门用于处理`.dbf`文件的纯Python模块。它目前托管在GitHub上。你可以通过指定下载文件来强制`easy_install`找到下载：

[PRE68]

如果您使用`pip`安装软件包，请使用以下命令：

[PRE69]

以下形状文件包含超过600条`.dbf`记录，代表美国人口普查局的区域，这使得它成为尝试`dbfpy`的好样本：[https://github.com/GeospatialPython/Learn/raw/master/GIS_CensusTract.zip](https://github.com/GeospatialPython/Learn/raw/master/GIS_CensusTract.zip).

让我们打开这个形状文件的`.dbf`文件并查看第一条记录：

[PRE70]

该模块快速且容易地为我们提供列名和数据值，而不是将它们作为单独的列表处理，这使得它们更容易管理。现在，让我们将包含在`POPULAT10`中的人口字段增加`1`：

[PRE71]

请记住，OGR和PyShp都可以执行此相同的过程，但如果您只是对`.dbf`文件进行大量更改，`dbfp3y`会使它稍微容易一些。

# Shapely

Shapely在**已知文本**（**WKT**）部分被提及，因为它具有导入和导出功能。然而，它的真正目的是作为一个通用的几何库。Shapely是GEOS库的几何操作的高级Python接口。实际上，Shapely故意避免读取或写入文件。它完全依赖于从其他模块导入和导出数据，并专注于几何操作。

让我们做一个快速的Shapely演示，我们将定义一个WKT多边形，然后将其导入Shapely。然后，我们将测量面积。我们的计算几何将包括通过五个任意单位对多边形进行缓冲，这将返回一个新的更大的多边形，我们将测量其面积：

[PRE72]

然后，我们可以执行缓冲区面积与原始多边形面积的差值，如下所示：

[PRE73]

如果您不能使用纯Python，那么像Shapely一样干净、功能强大的Python API无疑是下一个最佳选择。

# Fiona

Fiona 库提供了围绕 OGR 库的简单 Python API，用于数据访问，仅此而已。这种方法使其易于使用，并且在使用 Python 时比 OGR 更简洁。Fiona 默认输出 GeoJSON。您可以在 [http://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona](http://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona) 找到 Fiona 的 wheel 文件。

例如，我们将使用本章前面查看的 `dbfpy` 示例中的 `GIS_CensusTract_poly.shp` 文件。

首先，我们将导入 `fiona` 和 Python 的 `pprint` 模块以格式化输出。然后，我们将打开 shapefile 并检查其驱动程序类型：

[PRE74]

# ESRI shapefile

接下来，我们将检查其坐标参考系统并获取数据边界框，如下所示：

[PRE75]

现在，我们将以 `geojson` 格式查看数据模式，并使用 `pprint` 模块进行格式化，如下面的代码行所示：

[PRE76]

接下来，让我们获取特征数量的统计：

[PRE77]

最后，我们将打印一条记录作为格式化的 GeoJSON，如下所示：

[PRE78]

# GDAL

GDAL 是用于栅格数据的占主导地位的地理空间库。其栅格功能非常显著，以至于它是任何语言中几乎所有地理空间工具包的一部分，Python 也不例外。要了解 GDAL 在 Python 中的基本工作原理，请下载以下示例栅格卫星图像的 ZIP 文件并解压：[https://github.com/GeospatialPython/Learn/raw/master/SatImage.zip](https://github.com/GeospatialPython/Learn/raw/master/SatImage.zip)。让我们打开这张图片，看看它有多少波段以及每个轴上有多少像素：

[PRE79]

通过在 OpenEV 中查看，我们可以看到以下图像有三个波段，2,592 列像素和 2,693 行像素：

![](img/1831299e-4500-49f1-8ef2-f8ee35f21442.png)

GDAL 是 Python 中一个极快的地理空间栅格读取器和写入器。它还可以很好地重新投影图像，除了能够执行一些其他技巧之外。然而，GDAL 的真正价值来自于它与下一个 Python 模块的交互，我们现在将对其进行检查。

# NumPy

NumPy 是一个专为 Python 和科学计算设计的极快的多维 Python 数组处理器，但用 C 语言编写。它可以通过 PyPI 或作为 wheel 文件（可在 [http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) 找到）获得，并且可以轻松安装。除了其惊人的速度外，NumPy 的魔力还包括其与其他库的交互。NumPy 可以与 GDAL、Shapely、**Python 图像库**（**PIL**）以及许多其他领域的科学计算 Python 库交换数据。

作为 NumPy 能力的快速示例，我们将将其与 GDAL 结合起来读取我们的示例卫星图像，然后创建其直方图。GDAL 和 NumPy 之间的接口是一个名为 `gdal_array` 的 GDAL 模块，它依赖于 NumPy。Numeric 是 NumPy 模块的传统名称。`gdal_array` 模块导入 NumPy。

在以下示例中，我们将使用`gdal_array`，它导入NumPy，将图像作为数组读取，获取第一个波段，并将其保存为JPEG图像：

[PRE80]

这个操作在OpenEV中给出了以下灰度图像：

![](img/9e69e56e-1fcd-4829-9bbd-e3f7e5e1e51e.png)

# PIL

PIL最初是为遥感开发的，但已经发展成为一个通用的Python图像编辑库。像NumPy一样，它是用C语言编写的，以提高速度，但它是专门为Python设计的。除了图像创建和处理外，它还有一个有用的栅格绘图模块。PIL也可以通过PyPI获得；然而，在Python 3中，你可能想使用Pillow模块，它是PIL的升级版本。正如你将在以下示例中看到的那样，我们可以使用Python的try语句以两种可能的方式导入PIL，这取决于你的安装方式。

在这个示例中，我们将结合PyShp和PIL将前一个示例中的`hancock` shapefile转换为栅格并保存为图像。我们将使用类似于[第1章](6b5bd08a-170c-4471-a3f3-d79d5b91f017.xhtml)中的SimpleGIS的世界到像素坐标转换，即使用Python进行地理空间分析。我们将创建一个图像作为PIL中的画布，然后使用PIL的`ImageDraw`模块来渲染多边形。最后，我们将将其保存为PNG图像，如下面的代码行所示：

[PRE81]

这个示例创建以下图像：

![](img/fe5e5ae1-a9ff-4c7e-b9e1-fa1da206b56b.png)

# PNGCanvas

有时，你可能发现PIL对于你的目的来说过于强大，或者你不允许安装PIL，因为你没有使用Python模块的C语言创建和编译的机器的管理权限。在这些情况下，你通常可以使用轻量级的纯Python PNGCanvas模块来解决问题。你可以使用`easy_install`或pip来安装它。

使用此模块，我们可以重复使用PIL执行的光栅形状文件示例，但使用纯Python，如下所示：

[PRE82]

这个示例提供了一个简单的轮廓，因为PNGCanvas没有内置的填充方法：

![](img/a5135eda-4450-4838-8565-64ba2695193f.png)

# GeoPandas

Pandas是一个高性能的Python数据分析库，可以处理大型表格数据集（类似于数据库），有序/无序，标记矩阵或未标记的统计数据。GeoPandas是Pandas的一个地理空间扩展，基于Shapely、Fiona、PyProj、Matplotlib和Descartes构建，所有这些都必须安装。它允许你轻松地在Python中执行操作，否则可能需要像PostGIS这样的空间数据库。你可以从[http://www.lfd.uci.edu/~gohlke/pythonlibs/#panda](http://www.lfd.uci.edu/~gohlke/pythonlibs/#panda)下载GeoPandas的wheel文件。

以下脚本打开一个shapefile并将其转换为GeoJSON。然后，它使用`matplotlib`创建一个地图：

[PRE83]

以下图像是先前命令的结果地图：

![](img/930444cb-0f37-407e-875f-dc4c5398b1ff.png)

# PyMySQL

流行的 MySQL（可在 [http://dev.mysql.com/downloads](http://dev.mysql.com/downloads) 获取）数据库正在逐渐发展空间功能。它支持 OGC 几何形状和一些空间函数。它还提供了 PyMySQL 库中的纯 Python API。有限的空间函数使用平面几何和边界矩形，而不是球面几何和形状。MySQL 的最新开发版本包含一些额外的函数，这些函数提高了这一功能。

在以下示例中，我们将创建一个名为 `spatial_db` 的 MySQL 数据库。然后，我们将添加一个名为 `PLACES` 的表，其中包含一个几何列。接下来，我们将添加两个城市作为点位置。最后，我们将使用 MySQL 的 `ST_Distance` 函数计算距离，并将结果从度数转换为英里。

首先，我们将导入我们的 `mysql` 库并设置数据库连接：

[PRE84]

接下来，我们获取数据库游标：

[PRE85]

现在，我们检查数据库是否已存在，如果存在则将其删除：

[PRE86]

现在，我们设置一个新的连接并获取游标：

[PRE87]

接下来，我们可以创建我们的新表并添加我们的字段：

[PRE88]

添加了字段后，我们就可以为一些城市的地理位置插入记录了：

[PRE89]

然后，我们可以将更改提交到数据库：

[PRE90]

现在，我们可以查询数据库了！首先，我们将获取所有点位置列表：

[PRE91]

现在，我们将从查询结果中提取两个点：

[PRE92]

在我们能够测量距离之前，我们需要将点列表转换为地理空间几何形状：

[PRE93]

最后，我们可以使用 `Distance` 存储过程来测量两个几何形状之间的距离：

[PRE94]

输出如下：

[PRE95]

其他空间数据库选项也可用，包括 PostGIS 和 SpatiaLite；然而，这些空间引擎在 Python 3 中的支持最多处于开发阶段。您可以通过 OGR 库访问 PostGIS 和 MySQL；然而，MySQL 的支持有限。

# PyFPDF

纯 Python 的 PyFPDF 库是一种创建 PDF（包括地图）的轻量级方式。由于 PDF 格式是一种广泛使用的标准，PDF 通常用于分发地图。您可以通过 PyPI 以 `fpdf` 的方式安装它。该软件的官方名称是 PyFPDF，因为它 PHP 语言模块 `fpdf` 的一部分。此模块使用一个称为单元格的概念，在页面的特定位置布局项目。作为一个快速示例，我们将从 PIL 示例中导入的 `hancock.png` 图像放入名为 `map.pdf` 的 PDF 中，以创建一个简单的 PDF 地图。地图顶部将有标题文本，说明汉考克县边界，然后是地图图像：

[PRE96]

如果您在 Adobe Acrobat Reader 或其他 PDF 阅读器（如 Sumatra PDF）中打开名为 `map.pdf` 的 PDF 文件，您会看到图像现在位于 A4 页面的中心。地理空间产品通常作为更大报告的一部分，PyFPDF 模块简化了自动生成 PDF 报告的过程。

# 地理空间 PDF

**便携式文档格式**，或**PDF**，是一种存储和以跨平台和应用独立的方式呈现数字化文本和图像的文件格式。PDF是一种广泛使用的文档格式，它也被扩展用于存储地理空间信息。

PDF规范从1.7版本开始包括地理空间PDF的扩展，这些扩展将文档的部分映射到物理空间，也称为地理参照。您可以创建点、线或多边形作为地理空间几何形状，这些几何形状也可以有属性。

在PDF内部编码地理空间信息有两种方法。一家名为TerraGo的公司制定了一个规范，该规范已被开放地理空间联盟作为最佳实践采用，但不是一个标准。这种格式被称为**GeoPDF**。Adobe Systems提出的扩展，即创建PDF规范的ISO 32000，目前正在纳入规范的2.0版本。

TerraGo的地理空间PDF产品符合OGC最佳实践文档和Adobe PDF扩展。但是，TerraGo超越了这些功能，包括图层和其他GIS功能。然而，您必须使用TerraGo的Adobe Acrobat或其他软件的插件来访问这些功能。至少，TerraGo支持至少在PDF软件中显示所需的功能。

在Python中，有一个名为`geopdf`的库，它与TerraGo无关，但支持OGC最佳实践。这个库最初是由Prominent Edge的Tyler Garner开发的，用于Python 2。它已被移植到Python 3。

从GitHub安装`geopdf`就像运行以下命令一样简单：

[PRE97]

以下示例重新创建了我们在[第1章](6b5bd08a-170c-4471-a3f3-d79d5b91f017.xhtml)，“使用Python学习地理空间分析”，在*简单GIS*部分创建的地图，作为一个地理空间PDF。`geopdf`库依赖于Python的ReportLab PDF库。我们需要执行的步骤如下：

1.  创建一个PDF绘图画布。

1.  为科罗拉多州绘制一个矩形。

1.  设置一个函数将地图坐标转换为屏幕坐标。

1.  绘制并标注城市和人口。

1.  将该州的所有角落注册为地理空间PDF坐标，这些坐标将整个地图进行地理参照。

Python代码的注释解释了每一步发生了什么：

[PRE98]

# Rasterio

我们在本章早期介绍的GDAL库功能非常强大，但它并不是为Python设计的。`rasterio`库通过将GDAL包装在一个非常简单、干净的Pythonic API中，解决了这个问题，用于栅格数据操作。

此示例使用本章GDAL示例中的卫星图像。我们将打开图像并获取一些元数据，如下所示

[PRE99]

# OSMnx

`osmnx`库结合了**Open Street Map**（**OSM**）和强大的NetworkX库来管理用于路由的街道网络。这个库有数十个依赖项，它将这些依赖项整合起来以执行下载、分析和可视化街道网络的复杂步骤。

您可以使用`pip`尝试安装`osmnx`：

[PRE100]

然而，您可能会因为依赖项而遇到一些安装问题。在这种情况下，使用我们将在本章后面介绍的Conda系统会更容易一些。

以下示例使用`osmnx`从OSM下载城市的街道数据，从中创建街道网络，并计算一些基本统计数据：

[PRE101]

# Jupyter

当您处理地理空间或其他科学数据时，您应该了解Jupyter项目。Jupyter Notebook应用程序在网页浏览器中创建和显示笔记本文档，这些文档是可读和可执行的代码和数据。它非常适合分享软件教程，并在地理空间Python世界中变得非常普遍。

您可以在这里找到Jupyter Notebooks和Python的良好介绍：[https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html)。

# Conda

Conda是一个开源的包管理系统，它使得安装和更新复杂的库变得更加容易。它与多种语言兼容，包括Python。Conda对于设置库和测试它们非常有用，这样我们就可以在开发环境中尝试新事物。通常，自定义配置生产环境会更好，但Conda是原型化新想法的绝佳方式。

您可以从[https://conda.io/en/latest/](https://conda.io/en/latest/)开始使用Conda。

# 摘要

在本章中，我们概述了Python特定的地理空间分析工具。许多这些工具都包含了绑定到我们在[第3章](a5e439d1-e7fd-46b4-8fd3-8f811bfe73e4.xhtml)，“地理空间技术景观”中讨论的库，以提供针对特定操作（如GDAL的栅格访问函数）的最佳解决方案。我们还尽可能地包括了纯Python库，并且在我们处理即将到来的章节时将继续包括纯Python算法。

在下一章中，我们将开始应用所有这些工具进行GIS分析。

# 进一步阅读

以下链接将帮助您进一步探索本章的主题。第一个链接是关于XPath查询语言，我们使用Elementree来过滤XML元素。第二个链接是Python字符串库的文档，这在本书中对于操作数据至关重要。第三，我们有`lxml`库，这是更强大和快速的XML库之一。最后，我们有Conda，它为Python中的科学操作提供了一个全面、易于使用的框架，包括地理空间技术：

+   想要获取XPath的更多信息，请查看以下链接：[http://www.w3schools.com/xsl/xpath_intro.asp](http://www.w3schools.com/xsl/xpath_intro.asp)

+   想要了解更多关于Python `string`模块的详细信息，请查看以下链接：[https://docs.python.org/3.4/library/string.html](https://docs.python.org/3.4/library/string.html)

+   LXML的文档可以在以下链接中找到：[http://lxml.de/](http://lxml.de/)

+   你可以在以下链接中了解更多关于Conda的信息：[https://conda.io/en/latest/](https://conda.io/en/latest/)
