# 必需的包

在本附录中，我将向您展示如何安装复制本书中示例所需的软件。我将向您展示如何在 Linux 和 macOS 上这样做，特别是 Ubuntu 17.10 和 High Sierra。如果您使用 Windows，则适用相同的原理，但具体细节可能略有不同。然而，我相信在任何情况下都不会太难。

执行本书中所有代码需要两种类型的要求：外部和内部。R 语言之外的应用软件，我称之为外部要求。R 语言内部的应用软件，即 R 包，我称之为内部要求。我将向您介绍这两种要求的安装过程。

# 外部要求 - R 语言之外的应用软件

本书代码复现所需的某些 R 包有外部依赖项，这些依赖项可以是安装或执行依赖项。我们将在以下章节中介绍每个外部依赖项的安装。安装外部依赖项并不困难，但它可能是一个需要我们在 R 之外做一些工作的不熟悉的过程。一旦我们成功安装了这些外部依赖项，安装 R 包应该会变得容易。

在我们继续之前，我想说的是，在尝试安装 R 包之前，您并不总是事先知道需要哪些外部依赖项。通常，您只需尝试安装包，看看会发生什么。如果没有出现任何问题，那么您就准备好了。如果出现问题，控制台输出的信息将提示您下一步需要做什么。大多数时候，快速在线搜索错误或查看包的文档就足够了解如何继续。随着您经验的增加，您将能够快速诊断并解决任何问题。

下表显示了我们需要安装的外部软件，以及它在哪些章节中使用，为什么使用它，以及您可以从哪里获取它的 URL。下表中提供的 Fortran 和 C++编译器的 URL 适用于 macOS。在 Linux 的情况下，我没有提供任何，因为我们将通过终端通过包管理器安装它们，您不需要导航到外部网站下载它们的安装程序。最后，所有这些软件都是免费的，您应该安装最新版本。R 包运行所需的外部软件如下表所示：

| **软件** | **章节** | **原因** | **下载 URL** |
| --- | --- | --- | --- |
| MySQL 社区服务器 | 4 | 提供 MySQL 数据库 | [`dev.mysql.com/downloads/mysql/`](https://dev.mysql.com/downloads/mysql/) |
| GDAL 系统 | 5 | Linux 中的 3D 图形 | [`www.gdal.org/index.html`](http://www.gdal.org/index.html) |
| XQuartz 系统 | 5 | macOS 中的 3D 图形 | [`www.xquartz.org/`](https://www.xquartz.org/) |
| Fortran 编译器 | 9 | 编译 Fortran | [`gcc.gnu.org/wiki/GFortranBinaries`](https://gcc.gnu.org/wiki/GFortranBinaries) |
| C++ 编译器 | 9 | 编译 C++ | [`developer.apple.com/xcode/`](https://developer.apple.com/xcode/) |

根据您的配置，您执行的某些终端命令（在 Linux 和 macOS 上）可能需要在前面加上 `sudo` 字符串，以便它们实际上可以修改您的系统。您可以在维基百科上关于 `sudo` 命令的文章中找到更多信息（[`en.wikipedia.org/wiki/Sudo`](https://en.wikipedia.org/wiki/Sudo)），以及您操作系统的文档中。

# RMySQL R 包的依赖项

第四章，*模拟销售数据和与数据库协同工作*，依赖于 MySQL 数据库。这意味着即使系统中没有 MySQL 数据库，RMySQL R 包也可以正常安装，但当 R 使用它来与 MySQL 数据库接口时，您必须有一个可用的且配置适当的数据库正在运行，否则您将遇到错误。

现在，我将向您展示如何在 Ubuntu 17.10 和 macOS High Sierra 中安装 MySQL 社区数据库。在安装过程中，您可能会被要求输入可选的用户名和密码，如果是这样，您应该抓住这个机会并实际指定这些值，而不是留空，因为我们将在 R 中需要实际值。如果您这样做，您可以跳过以下关于设置用户名/密码组合的部分。

# Ubuntu 17.10

在 Ubuntu 中安装 MySQL 非常简单。你只需更新你的包管理器并安装 `mysql-server` 包，如下所示：

```py
$ apt-get update
$ apt-get install mysql-server 
```

数据库应该会自动为您执行，您可以通过查看下一节标题为 *Both* 的内容来验证。如果不是这样，您可以使用以下命令来启动数据库：

```py
$ sudo service mysql start 
```

查阅 Rackspace 的帖子 *在 Ubuntu 上安装 MySQL 服务器* ([`support.rackspace.com/how-to/installing-mysql-server-on-ubuntu/`](https://support.rackspace.com/how-to/installing-mysql-server-on-ubuntu/)) 以获取更详细的说明。

# macOS High Sierra

您需要做的第一件事是安装 **Xcode** ([`developer.apple.com/xcode/`](https://developer.apple.com/xcode/))。为此，您需要在您的计算机上打开 App Store，搜索 `Xcode` 并安装它。如果您有任何与 macOS 相关的开发工作，您很可能已经安装了它，因为它是大多数 macOS 下开发的基本依赖项。

接下来，我建议您使用出色的 **Homebrew** 软件包管理器 ([`brew.sh/`](https://brew.sh/))）。它是您在 Ubuntu 中可以获得的与 `apt-get` 类似工具的近似物。要安装它，您需要在您的终端中执行以下行。请注意，命令中的实际 URL 可能会更改，并且您应该确保它与 Homebrew 网站上显示的相匹配。

以下命令使用 "`\`" 符号进行了分割。如果您想将其作为单行使用，您可以删除这样的符号并将两行合并为一行。

让我们看看以下命令：

```py
$ /usr/bin/ruby -e "$(curl -fsSL \
 https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

一旦您安装了 Xcode 和 Homebrew，然后您只需在您的终端中执行以下行即可安装 MySQL，并且您应该已经设置好了：

```py
$ brew install mysql
```

如果您以这种方式安装 MySQL 时遇到任何问题，您可以尝试更手动的方法，通过访问 MySQL 社区下载页面([`dev.mysql.com/downloads/mysql/`](https://dev.mysql.com/downloads/mysql/))，下载适当的 DMG 文件，并将其作为任何其他 macOS 应用程序安装。

# 在 Linux 和 macOS 上设置用户/密码

一旦您在计算机上安装了 MySQL 数据库，您需要确保您可以使用明确的用户/密码组合访问它。如果您已经设置了它们，您应该能够像前面所示那样访问数据库。

`<YOUR_PASSWORD>`值显示在第二行，并且没有命令提示符(**$**)，因为它不应该包含在第一行中，您应该等待 MySQL 请求它，这通常是在执行第一行之后，这是一个不可见的提示，意味着您不会看到您正在输入的内容（出于安全原因）：

```py
$ mysql -u <YOU_USER> -p
<YOUR_PASSWORD>
```

如果您看到类似前面的信息，并且得到类似`mysql>`的命令提示符，那么您已经设置好了，当您从 R 中连接到数据库时应该使用该用户/密码组合：

```py
$ mysql

Welcome to the MySQL monitor. Commands end with ; or \g.
Your MySQL connection id is 15
Server version: 5.7.20-0ubuntu0.17.10.1 (Ubuntu)

Copyright (c) 2000, 2017, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql>
```

如果您无法连接，或者您没有明确的用户/密码组合，那么我们需要创建一个。为此，您需要弄清楚如何登录到您的 MySQL 服务器，这取决于安装数据库时使用的配置（即使是类似操作系统也可能不同）。您很可能会通过在终端中执行以下命令来访问它：

```py
$ mysql
```

一旦您进入 MySQL 命令提示符，您应该执行以下行来为您的本地安装创建用户/密码组合。完成此操作后，您应该已经设置好了，并且您应该能够使用之前显示的用户/密码组合明确登录：

```py
mysql> CREATE USER ''@'localhost' IDENTIFIED BY '';
mysql> GRANT ALL ON *.* TO ''@'localhost'; 
```

最后，为了明确起见，当您在相应章节中看到以下代码时，您需要使用在这里创建的相同的用户/密码替换`<YOUR_USER>`和`<YOUR_PASSWORD>`占位符：

```py
db <- dbConnect(
    MySQL(),
    user = <YOU_USER>,
    password = <YOUR_PASSWORD>,
    host = "localhost"
)
```

# rgl 和 rgdal R 包的依赖项

第五章，*通过可视化进行销售沟通*，使用了`rgl`和`rgdal`包来创建 3D 和地理数据图表。这两个包是我们将在本书中看到的具有最复杂外部依赖的包，因此我们将提供不同的安装方法，以防其中一个对您不起作用。

我们需要在 Ubuntu 或 macOS 的 Xquartz 中安装 **GDAL**（地理空间数据抽象库）系统库 ([`www.gdal.org/`](http://www.gdal.org/)) 以及地理空间和 **X11** ([`www.x.org/wiki/`](https://www.x.org/wiki/)) 以创建具有动态内容的窗口。

在 Windows 的情况下，您不需要像 X11 或 Xquartz 这样的外部工具，因为 Windows 本地处理必要的窗口。

# Ubuntu 17.10

要安装 GDAL 和 X11，我们需要在 Ubuntu 中安装各种系统库。最简单的方法是使用以下行。如果您没有遇到问题，那么您应该已经设置好了：

```py
$ apt-get update
$ apt-get install r-cran-rgl 
```

如果使用之前的行导致错误或无法正常工作，您可以尝试使用以下行以更明确的方式安装 GDAL。最后两行可以合并为一句，如果需要，它们被拆分是因为空间限制：

```py
# apt-get update
$ apt-get install mesa-common-dev libglu1-mesa-dev libgdal1-dev
$ apt-get install libx11-dev libudunits2-dev libproj-dev 
```

如果您在使用之前的命令时遇到某种错误，您可以尝试添加 `ubuntugis` 仓库信息，更新您的包管理器，然后重试之前的代码：

```py
$ add-apt-repository ppa:ubuntugis/ubuntugis-unstable
$ apt-get update
```

# macOS High Sierra

在 macOS 中安装 GDAL，您可以使用我们之前提到的 Homebrew 包安装。当然，此时您也可以在您的电脑上安装 Xcode：

```py
$ brew install proj geos udunits
$ brew install gdal2 --with-armadillo --with-complete --with-libkml --with-unsupported 
```

最后，我们需要安装 Xquartz 系统（类似于 Ubuntu 的 X11）。为此，请访问 Xquartz 网站 ([`www.xquartz.org/`](https://www.xquartz.org/))，下载适当的 DMG 文件，并像安装任何其他 macOS 应用程序一样安装它。

# Rcpp 包和 .Fortran() 函数的依赖项

第九章，*实现高效简单移动平均*，展示了如何将代码委托给 Fortran 和 C++ 以提高速度。这些语言都有自己的编译器，必须使用它们来编译相应的代码，以便 R 可以使用。如何在章节中编译此类代码。在这里，我们将展示如何安装编译器。

C++ 代码的编译器称为 `gcc`，Fortran 的编译器称为 `gfortran`。您可能已经在电脑上安装了它们，因为它们是 R 的依赖项，但如果没有，安装它们也很容易。

# Ubuntu 17.10

在 Ubuntu 中安装这两个编译器，只需在您的终端中执行以下行：

```py
$ apt-get update
$ apt-get install gcc ggfortran 
```

# macOS High Sierra

在 macOS 中安装 C++ 编译器，只需安装 **Xcode** ([`developer.apple.com/xcode/`](https://developer.apple.com/xcode/))。正如我们之前提到的，它可以通过您电脑中应有的 App Store 应用程序安装。

要安装 Fortran 编译器，您可以使用前面所示 Homebrew 包管理器。但是，如果由于某种原因它不起作用，您也可以尝试使用 GNU 网站上找到的二进制文件 ([`gcc.gnu.org/wiki/GFortranBinaries`](https://gcc.gnu.org/wiki/GFortranBinaries))：

```py
$ brew install gfortran
```

# 内部要求 - R 包

R *包* 是一组相关的函数、帮助文件和数据文件，它们被捆绑在一起。在撰写本文时，**综合 R 档案网络**（**CRAN**）([`cran.r-project.org/`](https://cran.r-project.org/))提供了超过 12,000 个 R 包。当使用 R 时，这是一个巨大的优势，因为您不必重新发明轮子来利用可能实现您所需功能的非常高质量的包，如果没有这样的包，您还可以贡献自己的包！

即使 CRAN 没有您需要的功能包，它可能存在于 GitLab、GitHub、Bitbucket 和其他 Git 托管网站的个人 Git 仓库中。实际上，我们将要安装的两个包来自 GitHub，而不是 CRAN，具体是`ggbiplot`和`ggthemr`。最后，您可能需要安装特定版本的包，就像我们将要做的`caret`包一样。

本书使用的所有包都在以下表中列出，包括它们使用的章节、您应该安装的版本以及我们为什么在书中使用它们的原因。在本书的示例中，我们使用了以下表中未显示的 R 包，但由于它们是内置的，我们不需要自己安装它们，因此没有显示。例如，`methods`和`parallel`包就是这种情况，它们分别用于与 S4 对象系统一起工作以及执行并行计算。我们需要安装的 R 包在以下表中列出：

| **包** | **章节** | **版本** | 原因 |
| --- | --- | --- | --- |
| `ggplot2` | 2, 3, 5, 9, and 10 | 最新版 | 高质量图表 |
| `ggbiplot` | 2 | 最新版 | 主成分图 |
| `viridis` | 2 and 5 | 最新版 | 图表颜色调色板 |
| `corrplot` | 2 and 3 | 最新版 | 相关性图 |
| `progress` | 2 and 3 | 最新版 | 显示迭代进度 |
| `RMySQL` | 4 | 最新版 | MySQL 数据库接口 |
| `ggExtra` | 5 | 最新版 | 带边缘分布的图表 |
| `threejs` | 5 | 最新版 | 交互式地球图表 |
| `leaflet` | 5 | 最新版 | 交互式高质量地图 |
| `plotly` | 5 | 最新版 | 交互式高质量图表 |
| `rgl` | 5 | 最新版 | 交互式 3D 图表 |
| `rgdal` | 5 | 最新版 | 地理数据处理 |
| `plyr` | 5 | 最新版 | 数据框追加 |
| `lsa` | 6 | 最新版 | 余弦相似度计算 |
| `rilba` | 6 | 最新版 | 高效 SVD 分解 |
| `caret` | 6 and 7 | 最新版 | 机器学习框架 |
| `twitteR` | 6 | 最新版 | Twitter API 接口 |
| `quanteda` | 6 | 最新版 | 文本数据处理 |
| `sentimentr` | 6 | 最新版 | 文本数据情感分析 |
| `randomForest` | 6 | 最新版 | 随机森林模型 |
| `ggrepel` | 7 | 最新版 | 避免图表中标签重叠 |
| `rmarkdown` | 7 | 最新版 | 带有可执行代码的 Markdown 文档 |
| `R6` | 8 | 最新版 | R6 对象模型 |
| `jsonlite` | 8 | 最新 | 从 JSON API 检索数据 |
| `lubridate` | 8, 9, 10 | 最新 | 简单地转换日期 |
| `microbenchmark` | 9 | 最新 | 基准函数的性能 |
| `shiny` | 10 | 最新 | 创建现代网络应用程序 |
| `shinythemes` | 10 | 最新 | 应用 Shiny 应用程序的主题 |
| `ggthemr` | 10 | 最新 | 应用`ggplot2`图形的主题 |

要安装这些包，你可以使用`install.packages(ggplot2)`命令，并将包更改为之前表中显示的每个包。然而，安装所有包的更有效的方法是将我们想要安装的所有包的名称向量发送到`install.packages()`函数，如下面的代码所示。最后，请注意，你可以发送`dependencies = TRUE`参数来告诉 R 尝试为你安装任何缺失的依赖项：

```py
install.packages(c(
    "ggplot2",
    "viridis",
    "corrplot",
    "progress",
    "RMySQL",
    "ggExtra",
    "threejs",
    "leaflet",
    "plotly",
    "rgl",
    "rgdal",
    "plyr",
    "lsa",
    "rilba",
    "twitteR",
    "quanteda",
    "sentimentr",
    "randomForest",
    "ggrepel",
    "rmarkdown",
    "R6",
    "jsonlite",
    "lubridate",
    "microbenchmark",
    "shiny",
    "shinythemes",
    dependencies = TRUE
))
```

注意，前面的向量省略了三个包：`ggbiplot`、`ggthemr`和`caret`。它们被省略是因为前两个只能直接从 GitHub（不是 CRAN）安装，而第三个需要一个特定的版本，因为最新的版本包含一个影响我们编写此内容时的一些代码的 bug。要安装`ggbiplot`包，我们需要 GitHub 上包的所有者的用户名。如果你访问包的 URL（[`github.com/vqv/ggbiplot`](https://github.com/vqv/ggbiplot)），你会看到它是`vqv`。现在，为了执行实际的安装，我们使用`devtools`包中的`install_github()`函数，并向它提供一个包含用户名（`vqv`）和存储库名称（`ggbiplot`）的字符串，这两个名称由斜杠（`/`）分隔。

如果你愿意，你可以将`devtools`包加载到内存中，然后直接调用`install_github()`函数。

让我们看看以下命令：

```py
devtools::install_github("vqv/ggbiplot")
```

类似地，要安装`ggthemr`包（[`github.com/cttobin/ggthemr`](https://github.com/cttobin/ggthemr)），我们使用以下命令行：

```py
devtools::install_github("cttobin/ggthemr")
```

最后，要安装`caret`包，我们可以使用 CRAN，但我们必须指定我们想要的版本，在这个例子中是`6.0.76`。为了完成这个任务，我们使用来自同一`devtools`包的`install_version`函数。在这种情况下，我们向它发送包的名称和我们想要的版本：

```py
devtools::install_version("caret", version = "6.0.76")
```

到现在为止，你应该拥有复制书中代码所需的一切。如果你遇到任何问题，我确信在线和 Stack Overflow（[`stackoverflow.com/`](https://stackoverflow.com/)）的搜索将非常有帮助。

# 加载 R 包

到目前为止，你应该能够加载这本书所需的 R 包，你可以使用`library()`或`require()`函数来这样做，这两个函数都接受你想要加载的包的名称作为参数。

你可能想知道为什么需要将包加载到 R 中才能使用它们。如果每个包都默认加载到 R 中，你可能会认为你正在使用一个函数，但实际上却在使用另一个函数。更糟糕的是，可能存在内部冲突：两个不同的包可能会使用完全相同的函数名，导致出现奇怪和意外的结果。通过只加载你需要的包，你可以最小化这些冲突的可能性。
