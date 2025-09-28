# 地理空间代码库简介

本章将介绍用于处理和分析地理空间数据的主要代码库。您将了解每个库的特点，它们之间的关系，如何安装它们，在哪里可以找到额外的文档，以及典型用例。这些说明假设用户在其机器上安装了最新的（2.7 或更高版本）Python，并且不涉及 Python 的安装。接下来，我们将讨论所有这些包是如何结合在一起的，以及它们在本书的其余部分是如何被涵盖的。

本章将涵盖以下库：

+   GDAL/OGR

+   GEOS

+   Shapely

+   Fiona

+   Python Shapefile 库 (`pyshp`)

+   `pyproj`

+   Rasterio

+   GeoPandas

# 地理空间数据抽象库（GDAL）和 OGR 简单特征库

**地理空间数据抽象库（Geospatial Data Abstraction Library）**（**GDAL**）/**OGR 简单特征库**结合了两个通常一起作为 GDAL 下载的独立库。这意味着安装 GDAL 包也提供了对 OGR 功能的访问，这就是为什么它们在这里一起被介绍。GDAL 被首先介绍的原因是其他包是在 GDAL 之后编写的，所以按时间顺序，它排在前面。您将注意到，本章中涵盖的一些包扩展了 GDAL 的功能或在其底层使用它。

GDAL 是由 Frank Warmerdam 在 1990 年代创建的，并于 2000 年 6 月首次发布。后来，GDAL 的发展被转移到了**开源地理空间基金会（Open Source Geospatial Foundation）**（**OSGeo**）。技术上，GDAL 与您平均的 Python 包略有不同，因为 GDAL 包本身是用 C 和 C++ 编写的，这意味着为了能够在 Python 中使用它，您需要编译 GDAL 及其相关的 Python 绑定。然而，使用 `conda` 和 Anaconda 可以相对容易地快速开始。由于它是用 C 和 C++ 编写的，因此在线 GDAL 文档是用库的 C++ 版本编写的。对于 Python 开发者来说，这可能是一个挑战，但许多函数都有文档，并且可以使用内置的 `pydoc` 工具或通过在 Python 中使用 `help` 函数进行查阅。

由于其历史原因，在 Python 中使用 GDAL 感觉更像是在使用 C++ 而不是纯 Python。例如，OGR 中的命名约定与 Python 的不同，因为您使用大写字母而不是小写字母来表示函数。这些差异解释了为什么选择其他一些 Python 库，如 Rasterio 和 Shapely，这些库也包含在本章中，虽然它们是从 Python 开发者的角度编写的，但提供了相同的 GDAL 功能。

GDAL 是一个庞大且广泛使用的栅格数据数据库。它支持读取和写入许多栅格文件格式，最新版本支持多达 200 种不同的文件格式。正因为如此，它在地理空间数据管理和分析中是不可或缺的。与其它 Python 库一起使用，GDAL 能够实现一些强大的遥感功能。它也是行业标准，存在于商业和开源 GIS 软件中。

OGR 库用于读取和写入矢量格式地理空间数据，支持读取和写入多种不同的格式。OGR 使用一个一致的模式来管理许多不同的矢量数据格式。我们将在 第五章 “矢量数据分析” 中讨论这个模型。你可以使用 OGR 进行矢量重投影、矢量数据格式转换、矢量属性数据过滤等操作。

GDAL/OGR 库不仅对 Python 程序员很有用，还被许多 GIS 供应商和开源项目所使用。截至撰写本文时，最新的 GDAL 版本是 2.2.4，该版本于 2018 年 3 月发布。

# 安装 GDAL

以前，Python 的 GDAL 安装相当复杂，需要你调整系统设置和路径变量。然而，仍然可以通过各种方式安装 GDAL，但我们建议你使用 Anaconda3 或 `conda`，因为这是最快、最简单的方法来开始。其他选项包括使用 `pip` 安装，或使用在线仓库，如 [`gdal.org`](http://gdal.org) 或 Tamas Szekeres 的 Windows 二进制文件 ([`www.gisinternals.com/release.php`](http://www.gisinternals.com/release.php))。

然而，这可能会比这里描述的选项复杂一些。安装 GDAL 的难点在于，该库的特定版本（以 C 语言编写，并安装在你的本地 Python 文件之外的独立系统目录中）有一个配套的 Python 版本，并且需要编译才能在 Python 中使用。此外，Python 的 GDAL 依赖于一些额外的 Python 库，这些库包含在安装中。虽然可以在同一台机器上使用多个 GDAL 版本，但这里推荐的方法是在虚拟环境中安装它，使用 Anaconda3、`conda` 或 `pip` 安装。这将保持你的系统设置干净，避免额外的路径变量或防止某些东西无法正常工作。

# 使用 Anaconda3 安装 GDAL

如果你使用 Anaconda3，最简单的方法是通过 Anaconda Navigator 创建一个虚拟环境，选择 Python 3.6 作为首选版本。然后，从未安装的 Python 软件包列表中选择 `gdal`。这将安装 `gdal` 版本 2.1.0。

安装后，你可以通过进入 Python 命令行并输入以下命令来检查一切是否正常工作：

```py
>> import gdal
>> import ogr
```

你可以按照以下方式检查 GDAL 的版本号：

```py
>> gdal.VersionInfo() # returns '2010300'
```

这意味着你正在运行 GDAL 版本 2.1.3。

# 使用 conda 安装 GDAL

使用`conda`安装 GDAL 比 Anaconda3 提供了更多选择 Python 版本的自由度。如果您打开终端，可以使用`conda search gdal`命令打印出可用的`gdal`版本及其对应的 Python 版本。如果您想了解每个包的依赖项，请输入`conda info gdal`。GDAL 的特定版本依赖于特定的包版本，如果您已经安装了这些包，例如 NumPy，这可能会成为一个问题。然后，您可以使用虚拟环境安装和运行 GDAL 及其依赖项，并使用相应的 Python 版本，例如：

```py
(C:\Users\<UserName> conda create -n myenv python=3.4
(C:\Users\<UserName> activate myenv # for Windows only. macOS and Linux users type "source activate myenv"
(C:\Users\<UserName> conda install gdal=2.1.0
```

您将被询问是否继续。如果您确认使用`y`并按*Enter*键，将安装一组额外的包。这些被称为**依赖项**，是 GDAL 运行所需的包。

如您所见，当您输入 `conda search gdal` 时，`conda` 并不会列出最新的 GDAL 版本，即 2.2.2。记住，在第一章，*包安装与管理*中，我们提到`conda`并不总是提供与其他方式可用的最新测试版本。这是一个例子。

# 使用 pip 安装 GDAL

**Python 包索引**（PyPI）也提供了 GDAL，这意味着您可以使用`pip`在您的机器上安装它。安装过程与前面描述的`conda`安装过程类似，但这次使用`pip install`命令。同样，如果您使用 Windows，建议在安装 GDAL 时使用虚拟环境，而不是需要您在系统环境设置中创建路径变量的根安装。

# 使用 pip 安装第二个 GDAL 版本

如果您有一台 Windows 机器，并且已经在您的机器上安装了一个工作的 GDAL 版本，但想使用`pip`安装额外的版本，您可以使用以下链接安装您选择的 GDAL 版本，然后从您激活的虚拟环境中运行以下命令来正确安装它：

GDAL 下载仓库：[`www.lfd.uci.edu/~gohlke/pythonlibs/#gdal`](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)

```py
>> pip install path\to\GDAL‑2.1.3‑cp27‑cp27m‑win32.whl
```

`GDAL-2.1.3-cp27m-win32.whl` 是下载的 GDAL 仓库的名称。

# 其他推荐的 GDAL 资源

GDAL/OGR Python API 的完整文档可在：[`gdal.org/python/`](http://gdal.org/python/)找到。

主页，[`gdal.org`](http://gdal.org)，也提供了 GDAL 的下载链接以及针对开发者和用户的详细文档。

# GEOS

**几何引擎开源**（**GEOS**）是**Java 拓扑套件**（**JTS**）和选定功能的 C/C++端口。GEOS 旨在包含 JTS 在 C++中的完整功能。它可以在许多平台上编译，包括 Python。正如你稍后将会看到的，Shapely 库使用了 GEOS 库中的函数。实际上，有许多应用程序使用 GEOS，包括 PostGIS 和 QGIS。在第十二章中介绍的 GeoDjango，*GeoDjango*，以及其他地理空间库，如 GDAL，也使用了 GEOS。GEOS 还可以与 GDAL 一起编译，为 OGR 提供所有其功能。

JTS 是一个用 Java 编写的开源地理空间计算几何库。它提供了各种功能，包括几何模型、几何函数、空间结构和算法以及输入/输出功能。使用 GEOS，你可以访问以下功能——地理空间函数（如`within`和`contains`）、地理空间操作（并集、交集等）、空间索引、**开放地理空间联盟**（**OGC**）**已知文本**（**WKT**）和**已知二进制**（**WKB**）输入/输出、C 和 C++ API 以及线程安全性。

# 安装 GEOS

GEOS 可以使用`pip install`、`conda`和 Anaconda3 进行安装：

```py
>> conda install -c conda-forge geos
>> pip install geos
```

关于 GEOS 和其他文档的详细安装信息可在此处获取：[`trac.osgeo.org/geos/`](https://trac.osgeo.org/geos/)

# Shapely

Shapely 是一个用于平面特征操作和分析的 Python 包，它使用来自 GEOS 库（PostGIS 的引擎）和 JTS 的端口中的函数。Shapely 不关心数据格式或坐标系，但可以轻松地与这些包集成。Shapely 只处理几何形状的分析，不提供读取和写入地理空间文件的能力。它是由 Sean Gillies 开发的，他也是 Fiona 和 Rasterio 背后的那个人。

Shapely 支持在`shapely.geometry`模块中以类形式实现的八个基本几何类型——点、多点、线字符串、多线字符串、线环、多边形、几何集合。除了表示这些几何形状外，Shapely 还可以通过多种方法和属性来操作和分析几何形状。

Shapely 在处理几何形状时主要具有与 OGR 相同的类和函数。Shapely 与 OGR 的区别在于，Shapely 具有更 Pythonic 和非常直观的接口，优化得更好，并且拥有完善的文档。使用 Shapely，你将编写纯 Python 代码，而使用 GEOS，你将在 Python 中编写 C++代码。对于**数据整理**，这是一个用于数据管理和分析术语，你最好使用纯 Python 编写，而不是 C++，这也解释了为什么创建了这些库。

关于 Shapely 的更多信息，请参阅[`toblerity.org/shapely/manual.html`](https://toblerity.org/shapely/manual.html)上的文档。此页面还提供了有关在不同平台上安装 Shapely 以及如何从源代码构建 Shapely 以兼容依赖 GEOS 的其他模块的详细信息。这指的是安装 Shapely 可能需要您升级已安装的 NumPy 和 GEOS。

# 安装 Shapely

Shapely 可以使用 `pip` 安装、`conda` 和 Anaconda3 进行安装：

```py
>> pip install shapely
>> conda install -c scitools shapely
```

Windows 用户也可以从[`www.lfd.uci.edu/~gohlke/pythonlibs/#shapely`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)获取 wheels。wheel 是 Python 的一个预构建包格式，包含一个 ZIP 格式存档，具有特殊格式的文件名和 `.whl` 扩展名。Shapely 1.6 需要 Python 版本高于 2.6，以及 GEOS 版本高于或等于 3.3。

还请参阅[`pypi.python.org/pypi/Shapely`](https://pypi.python.org/pypi/Shapely)以获取有关安装和使用 Shapely 的更多信息。

# Fiona

Fiona 是 OGR 的 API。它可以用于读取和写入数据格式。使用它的主要原因之一是它比 OGR 更接近 Python，同时更可靠且错误更少。它利用两种标记语言，WKT 和 WKB，来表示矢量数据的空间信息。因此，它可以很好地与其他 Python 库（如 Shapely）结合使用。您将使用 Fiona 进行输入和输出，而使用 Shapely 创建和操作地理空间数据。

虽然 Fiona 与 Python 兼容且是我们的推荐，但用户也应了解一些缺点。它比 OGR 更可靠，因为它使用 Python 对象来复制矢量数据，而不是 C 指针，这也意味着它们使用更多的内存，这会影响性能。

# 安装 Fiona

您可以使用 `pip` 安装、`conda` 或 Anaconda3 来安装 Fiona：

```py
>> conda install -c conda-forge fiona
>> conda install -c conda-forge/label/broken fiona
>> pip install fiona
```

Fiona 需要 Python 2.6、2.7、3.3 或 3.4 以及 GDAL/OGR 1.8+。Fiona 依赖于模块 `six`、`cligj`、`munch`、`argparse` 和 `ordereddict`（后两个模块是 Python 2.7+ 的标准模块）。

有关更多下载信息，请参阅 Fiona 的自述文件页面[`toblerity.org/fiona/README.html`](https://toblerity.org/fiona/README.html)。

# Python shapefile library (pyshp)

**Python shapefile 库**（**pyshp**）是一个纯 Python 库，用于读取和写入 shapefiles。`pyshp` 库的唯一目的是处理 shapefiles——它只使用 Python 标准库。您不能用它进行几何运算。如果您只处理 shapefiles，这个仅包含一个文件的库比使用 GDAL 更简单。

# 安装 pyshp

您可以使用 `pip` 安装、`conda` 和 Anaconda3 来安装 `pyshp`：

```py
>> pip install pyshp
>> conda install pyshp
```

更多文档可在 PyPi 上找到：[`pypi.python.org/pypi/pyshp/1.2.3`](https://pypi.python.org/pypi/pyshp/1.2.3)

`pyshp` 的源代码可在以下网址找到：[`github.com/GeospatialPython/pyshp`](https://github.com/GeospatialPython/pyshp)。

# pyproj

`pyproj` 是一个执行地图变换和大地测量计算的 Python 包。它是一个 Cython 包装器，提供 Python 接口到 PROJ.4 函数，这意味着你可以在 Python 中访问现有的 C 代码库。

PROJ.4 是一个投影库，可以在许多坐标系之间转换数据，并且也通过 GDAL 和 OGR 提供。PROJ.4 仍然受欢迎和广泛使用的原因有两个：

+   首先，因为它支持如此多的不同坐标系

+   其次，因为它提供了执行此操作的方法——接下来将要介绍的 Rasterio 和 GeoPandas 两个 Python 库，都使用 `pyproj` 和 PROJ.4 功能。

使用 PROJ.4 单独而不是与 GDAL 等包一起使用，其区别在于它允许你重新投影单个点，而使用 PROJ.4 的包不提供此功能。

`pyproj` 包提供了两个类——`Proj` 类和 `Geod` 类。`Proj` 类执行地图计算，而 `Geod` 类执行大地测量计算。

# 安装 pyproj

使用 `pip install`、`conda` 和 Anaconda3 安装 `pyproj`：

```py
>> conda install -c conda-forge pyproj
>> pip install pyproj
```

以下链接包含有关 `pyproj` 的更多信息：[`jswhit.github.io/pyproj/`](https://jswhit.github.io/pyproj/)

你可以在 [`proj4.org/`](http://proj4.org/) 上找到更多关于 PROJ.4 的信息。

# Rasterio

Rasterio 是一个基于 GDAL 和 NumPy 的 Python 库，用于处理栅格数据，它以 Python 开发者为出发点，而不是 C 语言，使用 Python 语言类型、协议和习惯用法编写。Rasterio 的目标是使 GIS 数据对 Python 程序员更加易于访问，并帮助 GIS 分析师学习重要的 Python 标准。Rasterio 依赖于 Python 的概念，而不是 GIS。

Rasterio 是来自 Mapbox 卫星团队的开放源代码项目，Mapbox 是网站和应用程序的定制在线地图提供商。这个库的名字应该读作 *raster-i-o* 而不是 *ras-te-rio*。Rasterio 是在名为 **Mapbox Cloudless Atlas** 的项目之后出现的，该项目旨在从卫星图像中创建一个看起来很漂亮的底图。

软件要求之一是使用开源软件和具有方便的多维数组语法的编程语言。尽管 GDAL 提供了经过验证的算法和驱动程序，但使用 GDAL 的 Python 绑定感觉就像是在用 C++ 开发。

因此，Rasterio 被设计成一个顶层 Python 包，中间是扩展模块（使用 Cython），底部是 GDAL 共享库。对于栅格库的其他要求是能够读写 NumPy ndarrays 到和从数据文件，使用 Python 类型、协议和习惯用法而不是 C 或 C++ 来解放程序员，使他们不必用两种语言编码。

对于地理配准，Rasterio 跟随 `pyproj` 的步伐。在读取和写入的基础上增加了一些功能，其中之一是功能模块。可以使用 `rasterio.warp` 模块进行地理空间数据的重投影。

Rasterio 的项目主页可在此处找到：[`github.com/mapbox/rasterio`](https://github.com/mapbox/rasterio)

# Rasterio 依赖项

如前所述，Rasterio 使用 GDAL，这意味着它是其依赖之一。Python 包依赖项包括 `affine`、`cligj`、`click`、`enum34` 和 `numpy`。

Rasterio 的文档可在此处找到：[`mapbox.github.io/rasterio/`](https://mapbox.github.io/rasterio/)

# Rasterio 安装

要在 Windows 机器上安装 Rasterio，你需要下载适用于你系统的 `rasterio` 和 GDAL 二进制文件，并运行：

```py
>> pip install -U pip
>> pip install GDAL-1.11.2-cp27-none-win32.whl
>> pip install rasterio-0.24.0-cp27-none-win32.whl
```

使用 `conda`，你可以这样安装 `rasterio`：

```py
>> conda config --add channels conda-forge # this enables the conda-forge channel
>> conda install rasterio
```

`conda-forge` 是一个额外的通道，可以从该通道安装包。

不同平台的详细安装说明可在此处找到：[`mapbox.github.io/rasterio/installation.html`](https://mapbox.github.io/rasterio/installation.html)

# GeoPandas

GeoPandas 是一个用于处理矢量数据的 Python 库。它基于 SciPy 堆栈中的 `pandas` 库。SciPy 是一个流行的数据检查和分析库，但遗憾的是，它不能读取空间数据。GeoPandas 的创建是为了填补这一空白，以 `pandas` 数据对象为起点。该库还增加了来自地理 Python 包的功能。

GeoPandas 提供了两个数据对象——一个基于 `pandas` Series 对象的 GeoSeries 对象和一个基于 `pandas` DataFrame 对象的 GeoDataFrame，为每一行添加一个几何列。GeoSeries 和 GeoDataFrame 对象都可以用于空间数据处理，类似于空间数据库。几乎为每种矢量数据格式提供了读写功能。此外，由于 Series 和 DataFrame 对象都是 pandas 数据对象的子类，你可以使用相同的属性来选择或子集数据，例如 `.loc` 或 `.iloc`。

GeoPandas 是一个能够很好地利用新工具（如 Jupyter Notebooks）的库，而 GDAL 则允许你通过 Python 代码与矢量和栅格数据集中的数据记录进行交互。GeoPandas 通过将所有记录加载到 GeoDataFrame 中，以便你可以在屏幕上一起查看它们，采取了一种更直观的方法。数据绘图也是如此。这些功能在 Python 2 中缺失，因为开发者依赖于没有广泛数据可视化能力的 IDE，而现在这些功能可以通过 Jupyter Notebooks 获得。

# GeoPandas 安装

安装 GeoPandas 有多种方式。你可以使用 `pip` install、`conda` install、Anaconda3 或 GitHub。使用终端窗口，你可以按以下方式安装：

```py
>> pip install geopandas
>> conda install -c conda-forge geopandas
```

详细安装信息可在此处找到：[`geopandas.org/install.html`](http://geopandas.org/install.html)

GeoPandas 也可以通过 PyPi 获取：[`pypi.python.org/pypi/geopandas/0.3.0`](https://pypi.python.org/pypi/geopandas/0.3.0)

GeoPandas 也可以通过 Anaconda Cloud 获取：[`anaconda.org/IOOS/geopandas`](https://anaconda.org/IOOS/geopandas)

# GeoPandas 依赖项

GeoPandas 依赖于以下 Python 库：`pandas`、Shapely、Fiona、`pyproj`、NumPy 和 `six`。这些库在安装 `GeoPandas` 时会更新或安装。

`Geopandas` 的文档可在 [`geopandas.org`](http://geopandas.org) 查找。

# 它们是如何协同工作的

我们提供了处理和分析地理空间数据最重要的开源软件包的概述**。**那么问题随之而来，何时使用某个软件包以及为什么。GDAL、OGR 和 GEOS 对于地理空间处理和分析是必不可少的，但它们不是用 Python 编写的，因此需要为 Python 开发者提供 Python 二进制文件。Fiona、Shapely 和 `pyproj` 是为了解决这些问题而编写的，以及较新的 Rasterio 库。为了更 Pythonic 的方法，这些较新的软件包比带有 Python 二进制文件的较老 C++ 软件包更受欢迎（尽管它们在底层使用）。

然而，了解所有这些软件包的起源和历史是很好的，因为它们都被广泛使用（并且有很好的理由）。下一章，将讨论地理空间数据库，将基于本章的信息。第五章，*矢量数据分析*，和 第六章，*栅格数据处理*，将专门处理这里讨论的库，更深入地探讨使用这些库进行栅格和矢量数据处理的细节。

到目前为止，你应该对处理和分析最重要的软件包有一个全局的了解，包括它们的历史以及它们是如何相互关联的。你应该对特定用例可用的选项有一个概念，以及为什么一个软件包比另一个软件包更可取。然而，正如编程中经常发生的那样，对于特定问题可能有多个解决方案。例如，处理 shapefiles 时，你可以根据你的偏好和问题使用 `pyshp`、GDAL、Shapely 或 GeoPandas。

# 摘要

在本章中，我们介绍了用于处理和分析地理空间数据的主要代码库。你学习了每个库的特点，它们是如何相互关联或相互区别的，如何安装它们，在哪里可以找到额外的文档，以及典型用例。GDAL 是一个包含两个独立库的主要库，即 OGR 和 GDAL。许多其他库和软件应用程序在底层使用 GDAL 功能，例如 Fiona 和 Rasterio，这两者都在本章中进行了介绍。这些是为了使与 GDAL 和 OGR 一起以更 Pythonic 的方式工作而创建的。

下一章将介绍空间数据库。这些数据库用于数据存储和分析，例如 SpatiaLite 和 PostGIS。你还将学习如何使用不同的 Python 库来连接这些数据库。
