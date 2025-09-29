Python 和安装 Jupyter Notebook 概述

既然你对数据素养背后的概念和数据分析的演变有了更好的理解，让我们设置自己的环境，以便你可以处理数据。在本章中，我们将介绍 Python 编程语言，以及一个流行的工具 Jupyter Notebook，它用于运行数据分析的命令。我们将逐步讲解安装过程，并讨论关键概念，以了解为什么它们对于数据分析是必需的。到本章结束时，你将拥有一个工作站，可以运行一个`hello world`程序，这将有助于增强你进一步学习更深入概念的信心。

在本章中，我们将涵盖以下内容：

+   安装 Python 和使用 Jupyter Notebook

+   存储和检索数据文件

+   Hello World！——运行你的第一个 Python 代码

+   探索 Python 包

# 技术要求

这是本书的 GitHub 仓库链接：[`github.com/PacktPublishing/Practical-Data-Analysis-using-Jupyter-Notebook/tree/master/Chapter02`](https://github.com/PacktPublishing/Practical-Data-Analysis-using-Jupyter-Notebook/tree/master/Chapter02).

此外，你可以从以下链接下载并安装所需的软件：[`www.anaconda.com/products/individual`](https://www.anaconda.com/products/individual)。

# 安装 Python 和使用 Jupyter Notebook

我首先要承认，这个章节在未来可能会变得过时，因为在你工作站上安装开源软件可能是一个痛苦的过程，在某些情况下，它正被预装虚拟机或云版本所取代。例如，微软提供免费 Azure 订阅选项，用于云托管的 Jupyter Notebook。

理解软件版本、硬件、**操作系统**（**OS**）差异和库依赖的所有依赖项可能很复杂。此外，你的 IT 部门在企业环境中对软件安装的规定可能有安全限制，禁止访问你的工作站文件系统。很可能会随着云计算的更多创新，大多数步骤已经在事先完成，从而消除了安装软件的需要。

话虽如此，我将带你通过安装 Python 和 Jupyter Notebook 的过程，沿途指出提示和陷阱，以教育你关键概念。我会将使用这些技术工具处理数据比作开车。驾驶的能力不应该依赖于你修理汽车引擎的能力！仅仅知道你需要一个引擎就足够你驾驶并前进。所以，我的重点是快速设置你的工作站以进行数据分析，而不关注这些强大技术背后的细节层次。

创建 Jupyter 笔记本应用程序的开源项目始于 2014 年的 iPython。iPython 中存在的许多功能今天仍然存在于 Jupyter 中，例如，用于运行 Python 命令的交互式 GUI 和并行处理。有一个内核来控制你的计算机 CPU、内存和文件系统之间的输入/输出。最后，笔记本还有一个功能，可以将所有命令、代码、图表和注释收集到一个单一的、带有`.ipynb`扩展名的可共享文件中。

为了提供一些关于 Jupyter 笔记本在数据分析中如何变得流行的背景信息，我发现了一个由 Peter Parente 创建的公共 GitHub 仓库，该仓库收集了自 2014 年以来在 GitHub 上找到的每日`.pynb`文件数量。增长是指数级的，因为到 2019 年 11 月，数量从 65,000 多增长到 570 万，这意味着在过去五年中每年都在翻倍！

使用 Jupyter 笔记本的第一个先决条件是安装 Python。我们将使用 3.3 或更高版本的 Python，并且有两种方法可以用来安装软件：直接下载或包管理器。直接下载将使你在工作站上的安装有更多的控制权，但这也需要额外的时间来管理依赖库。话虽如此，使用包管理器安装 Python 已经成为首选方法，因此，我在本章中介绍了这种方法。

Python 是一种功能强大的编程语言，支持多个操作系统平台，包括 Windows、macOS 和 Linux。我鼓励你阅读更多关于这种强大软件语言的历史以及其创造者 Guido van Rossum 的信息。

Python 在本质上是一种命令行编程语言，因此你必须习惯于从提示符运行一些命令。当我们完成安装后，你将拥有一个 Python 命令行窗口，如果你的工作站运行 Windows 操作系统，它将看起来如下截图所示：

![图片](img/fa36b410-bffb-450e-a3bd-5644cafed652.png)

将 Python 安装视为达到目的的手段，因为我们真正想要用作数据分析师的是 Jupyter 笔记本，它也被称为用于运行代码和调用库的**集成开发环境**（**IDE**），它是一个自包含的**图形用户界面**（**GUI**）。

由于我推荐使用包管理器进行安装，你必须做出的第一个决定是选择哪个包管理器来在你的计算机上安装。包管理器旨在简化开源库、你的操作系统和软件之间的版本和依赖层。最常见的是`conda`、`pip`或`docker`。

通过研究差异，我更喜欢`conda`而不是`pip`，尤其是对于刚开始的人来说，如果您不熟悉运行命令行命令和在 PC 上直接管理软件安装，我更推荐 Anaconda。因为它包括了 Python、几个用于数据分析的流行库以及 Jupyter，所有这些都在下载包中。

记住，目标是让 Jupyter Notebook 在您的工作站上运行起来，所以请随意选择安装替代方案，尤其是如果您更喜欢**命令行界面**（**CLI**）的话。

## 安装 Anaconda

按照以下步骤安装 Anaconda。对于这个教程，我选择了 Windows 操作系统安装程序，但无论选择哪个，安装截图都将类似：

1.  根据您的工作站操作系统选择所需的安装程序来下载软件。为此，请导航到 Anaconda Distribution 页面，该页面应类似于以下截图，并可在[`www.anaconda.com/`](https://www.anaconda.com/)找到：

![图片](img/10ad98c5-b081-4d46-abb2-fedd0f5009cc.png)

1.  下载软件并在您的 PC 上启动安装程序后，您应该会看到如下所示的设置向导截图：

![图片](img/f03f00bf-ce2c-4981-a1b0-e88013a970b3.png)

1.  在安装向导中选择默认选项，您应该会看到如下所示的类似消息：

![图片](img/17b9dc57-440a-480f-a62b-1056333b715b.png)

1.  现在 Anaconda 安装完成后，您必须从您的 PC 上启动 Anaconda Navigator 应用程序，如下所示（使用 Windows 操作系统）。由于有多个操作系统选项可用，如 Windows、macOS 或 Ubuntu，您的屏幕将与此截图有所不同：

![图片](img/9695d77c-a300-48bb-aa80-3c755ea6a732.png)

我认为安装过程类似于为什么艺术家需要购买画布、画架和材料来开始绘画。现在我们已经安装并可以使用名为 Anaconda 的工作环境了，您就可以启动 Jupyter 并创建您的第一个笔记本。

## 运行 Jupyter 并为数据分析安装 Python 包

软件安装到您的 PC 后，启动 Jupyter 笔记本可以通过两种方式之一完成。第一种是通过 Anaconda Prompt 中的命令行提示符使用`jupyter notebook`命令，其外观类似于以下截图：

![图片](img/5c46c68a-fbd6-4901-967b-dcd8890b91c6.png)

您还可以使用 Anaconda Navigator 软件，在 Jupyter Notebook 的“我的应用”中点击启动按钮，如下所示：

![图片](img/b5a6c3de-96f5-4ec5-b010-f71702061099.png)

两种选项都将启动一个新的网络浏览器会话，使用 `http://localhost:8888/tree` URL，这被称为 Jupyter 仪表板。如果您没有看到以下截图**所示的内容，您可能需要重新安装 Anaconda 软件，或者检查防火墙端口是否阻止了命令或依赖项。在企业环境中，您可能需要审查您的公司政策或请求 IT 支持：

![](img/8d3cb440-3ca7-4e07-a425-726613ec9423.png)

如果您想尝试 JupyterLab 而不是 Jupyter Notebook，任何一种解决方案都可以工作。JupyterLab 使用与经典 Jupyter Notebook 完全相同的 Notebook 服务器和文件格式，因此它与现有的笔记本和内核完全兼容。经典笔记本和 JupyterLab 可以在同一台计算机上并排运行。您可以轻松地在两个界面之间切换。

注意，Jupyter 默认根据其安装方式访问您工作站上的文件系统。在大多数情况下，这应该足够了，但如果您想更改默认的项目 `home`/`root` 文件夹，您可以使用 Anaconda Prompt 轻易地更改它。只需在输入 `jupyter notebook` 命令之前运行 `cd` 命令来更改目录。

例如，我在 Windows PC 的本地 `c:\` 驱动器路径上首先创建了一个 `project` 文件夹，然后使用以下命令运行 Anaconda Prompt 窗口：

```py
>cd \
>cd projects
>jupyter notebook
```

如果您按照这个示例操作，如果您使用的是 Windows 操作系统，您的命令提示符窗口应该看起来像以下截图：

![](img/6b16a838-8be3-4f93-a346-96012c3003da.png)

完成后，Jupyter 会话中显示的文件和文件夹列表将是空的，您的会话将类似于以下截图：

![](img/614d30e9-6514-40ce-85e7-8998e9f2aa45.png)

现在，Jupyter 软件应该已经在您的工作站上积极运行，准备好浏览所有可用的功能，我们将在下一部分介绍。

# 存储和检索数据文件

我喜欢使用 Jupyter 的原因是它是一个包含数据分析的自包含解决方案。我的意思是，您可以在一个地方与文件系统交互，添加、更新和删除文件夹，以及运行 Python 命令。随着您继续使用这个工具，我认为您会发现与在多个窗口、应用程序或工作站上的系统之间跳转相比，保持在一个生态系统中导航要容易得多。

让我们从熟悉添加、编辑或删除文件的菜单选项开始。Jupyter 默认通过列出从安装目录路径可访问的所有文件和文件夹来列出仪表板。这可以配置为更改起始文件夹，但我们将使用 Windows 默认设置。在以下截图中，我已经用字母突出显示了 Jupyter 仪表板的重要部分，以便于参考：

![](img/4a735e04-42f9-4dff-829b-9bfb213c3ad8.png)

在**A**节中，当在个人工作站上运行时，URL 默认为`http://localhost:888/tree`。如果笔记本托管在服务器或云上，此 URL 将更改。注意，当您在**B**节中选择文件夹或文件时，URL 地址将更改为跟随您选择的位置和路径。

在**B**节中，您将找到仪表板可见的文件夹或文件的层次结构。如果您点击任何文件，它将尝试在编辑器中打开它，无论该文件是否可由 Jupyter 使用。编辑器可读取的文件扩展名包括`.jpeg`、`.png`和`.svg`格式的图像；半结构化数据文件，如`.json`、`.csv`和`.xml`；以及代码，如`.html`、`.py`（Python）和`.js`（JavaScript）。请注意，当打开文件时，URL 路径将从`tree`参数词更改为`edit`。

如果编辑器无法识别文件，它将在第一行提供错误信息，并告诉您原因，类似于以下截图：

![图片](img/4eca4ee4-8e01-4f62-9e8d-4ecc13d9aaed.png)

在**C**节中，您可以选择和过滤仪表板上显示的一个或多个文件或文件夹。这可以在创建多个笔记本和组织用于分析的数据文件时用于组织项目工作空间。一旦选择了任何文件或文件夹，标题“选择项目以执行操作”将更改为操作按钮**重命名**、**复制**和一个红色垃圾桶图标，它将删除文件或文件夹，如以下截图所示：

![图片](img/1ac2ee11-4b4e-486d-8afe-8081180cfd7b.png)

在仪表板上，您还会注意到标记为“文件”、“运行”和“集群”的标签页。这些标签页由 Jupyter 应用程序使用，以帮助您保持方向并跟踪正在积极运行的过程。集群是一个高级功能，超出了本书的范围。我们已经在**B**节中介绍了“文件”标签页。

让我们讨论一下“运行”标签页。它有两个部分：终端，它将包含 Windows 操作系统中的 Powershell 等系统 shell 命令；以及笔记本，它将显示所有正在使用的活动笔记本。一旦我们创建了一些笔记本，我鼓励您刷新浏览器以查看哪些笔记本文件是活动的，以便更好地理解这一功能。如果需要终止一个无响应或占用过多计算机资源（CPU/RAM）的活动笔记本，请使用“关闭”按钮。

在**D**节中，您将看到一个“上传”按钮，允许您将文件添加到仪表板中的任何已导航文件夹。新按钮包含一个子菜单，可以创建文本文件、文件夹或 Python 3 笔记本。

# Hello World！– 运行您的第一个 Python 代码

现在我们对仪表板及其导航有了更好的理解，让我们创建我们的第一个笔记本并运行一些 Python 代码。最简单的方法是点击新建按钮，然后在子菜单中选择 Python 3。这将打开浏览器中的一个新标签页或窗口，其外观类似于以下截图：

![图片](img/24497712-89c5-4d68-bbe5-f734efb41129.png)

我建议将任何笔记本的无标题文件重命名，以便以后更容易找到它们。为此，从文件菜单中选择重命名，如以下截图所示，并将其重命名为`hello_world`或相关的项目名称。一旦点击 OK 按钮，页面顶部的标题栏将显示新名称：

![图片](img/b6637b32-60cb-4a5a-8178-376e9c37e9f8.png)

通过重命名笔记本，将创建一个以`.ipynb`扩展名的新文件，其中包含所有内容的 JSON 格式。这有助于使笔记本文件可共享，并有助于版本控制，它是文件中更改的审计跟踪。

您可以通过选择编辑菜单中的编辑笔记本元数据来查看实际的 JSON 元数据内容。结果将类似于以下截图：

![图片](img/8ed1d340-88bd-40da-8de1-1be467390a5b.png)

笔记本的 UI 看起来与其他今天使用的现代网络软件非常相似，因为它是为了便于导航而设计的。以下菜单选项是易于使用的图标，统称为笔记本工具栏，它支持键盘快捷键，以优化您使用工具时的工作流程。您可以在帮助菜单中找到用户界面之旅和键盘快捷键，如以下截图所示。我建议您浏览它们，以查看所有可用的功能：

![图片](img/dd688ac9-0ccf-4829-a7f2-0f25a158fed2.png)

一旦您对帮助菜单选项感到舒适，让我们通过在笔记本单元格中输入`print("hello world")`命令来编写您的第一个代码，该单元格默认为`In []:`。记住，如果您使用鼠标在笔记本中导航，您必须点击单元格以选择它并出现光标。

在命令后按*Enter*键只会创建一个用于更多输入的第二行。您必须使用键盘快捷键、单元格菜单或工具栏图标来执行任何命令。

一旦你在单元格中输入了`print("hello world")`命令并点击了以下选项之一。运行命令的选项如下：

+   点击工具栏中的![图片]按钮。

+   从单元格菜单中选择运行单元格。

+   按下*Shift* + *Enter* 或 *Ctrl* + *Enter* 键。

屏幕应类似于以下截图：

![图片](img/84fcd699-effb-48a3-893e-6c03deaf6f14.png)

恭喜你，你已经创建了你的第一个 Jupyter 笔记本并运行了你的第一个命令！点击文件菜单中的关闭和停止选项返回到仪表板。

## 创建项目文件夹层次结构

现在我们已经覆盖了基础知识，让我们遍历一个目录以找到特定的文件，并创建一个项目文件夹层次结构，为未来的数据分析学习模块做准备。我建议在你的工作站上创建一个起始的`projects`文件夹，以保持所有笔记本和数据井然有序。标准的企业目录结构因公司而异，但设置一个带有子文件夹的基本结构使过程可移植，并有助于与他人共享工作。以下截图显示了项目文件夹模板示例：

![图片](img/459e5358-387b-49b4-963c-bf108290fff2.png)

在整本书中，我将使用章节编号作为`projectname`来使每个目录子文件夹，如`data`和`notebooks`，模块化、独立且易于跟踪。你的工作站目录结构和树应该与本书的 GitHub 仓库相匹配，以便更容易同步你的文件和文件夹。

按照“照我说的做，别照我做的做”的经典方式，以及由于不同操作系统版本之间相对路径的限制，示例使用相同的文件夹以防止本书中出错。要继续，你可以克隆或下载本书 GitHub 仓库中的所有文件和子文件夹，在 Jupyter 仪表板上创建所有文件夹和文件，或者在您的工作站上创建它们。完成后，本章的项目文件夹应该看起来如下面的截图所示：

![图片](img/c916a856-888e-4cc4-bdf4-4342f3543fca.png)

## 上传文件

现在我们有了项目文件夹，让我们遍历以下步骤来上传一个用于分析的文件。你必须提前从*技术要求*部分找到的 GitHub 仓库 URL 下载文件：

1.  点击`data`文件夹名称。

1.  点击`source`子文件夹名称。

1.  点击屏幕右上角的“上传”按钮。

1.  选择`evolution_of_data_analysis.csv`。

1.  点击蓝色的“上传”按钮继续。完成后，你将在仪表板上看到一个文件，如下面的截图所示：

![图片](img/b472628b-fd01-469d-a9e8-ba5bee2e5c54.png)

1.  返回到`notebooks`文件夹，通过点击“新建”菜单创建一个新的笔记本文件。类似于`hello_world`示例，在子菜单中选择 Python 3 以创建默认的`未命名`笔记本。

如前所述，我总是在移动之前重命名`未命名`笔记本，所以将笔记本重命名为`evolution_of_data_analysis`。

1.  要从笔记本中读取文件中的数据，你必须运行几个 Python 命令。这些命令可以全部在一个单元中运行，也可以作为三个单独的单元条目运行。打开我们之前上传的 CSV 文件的命令如下：

```py
f = open("../data/source/evolution_of_data_analysis.csv","r")
print(f.read())
f.close()
```

让我们逐行查看命令。首先，我们将文件打开命令的值赋给`f`变量，以缩短下一行中附加命令的长度。注意`evolution_of_data_analysis.csv`文件包含了目录路径`"../data/source/"`，这是必需的，因为活动笔记本`evolution_of_data_analysis`位于不同的文件夹中。打开命令还包括一个参数`r`，这意味着我们只想读取文件，而不编辑内容。

第二行是通过传递`f`变量和`read()`函数来打印文件内容。这将结果显示在输出单元格中，类似于以下截图：

![图片](img/a6680d18-35ac-4ad2-a65f-1ed2ce1a3434.png)

最后一行是一个最佳实践，用于关闭文件，以避免在以后使用文件或在操作系统文件系统中出现冲突。一旦你验证可以在笔记本中看到 CSV 文件的内容，请从文件菜单中选择关闭和停止选项，返回到仪表板。

# 探索 Python 包

在结束本章之前，让我们探索与数据分析相关的不同 Python 包，并验证它们是否可在 Jupyter 笔记本应用程序中使用。这些包随着时间的推移而发展，并且是开源的，因此程序员可以贡献并改进源代码。

Python 包的版本将随着时间的推移而增加，具体取决于你在机器上安装`conda`或`pip`（包管理器）的时间。如果你在运行命令时收到错误，请验证它们是否与本书中使用的版本匹配。

当我们在未来的章节中使用它们的出色功能时，我们将更深入地介绍每个单独的包。本章的重点是验证特定的库是否可用，并且有几种不同的方法可以使用，例如检查工作站上的特定文件安装文件夹或从 Python 命令行运行命令。我发现最简单的方法是在新笔记本中运行几个简单的命令。

导航回`notebooks`文件夹，通过点击菜单中的新建并从子菜单中选择 Python 3 来创建一个新的笔记本文件，以创建默认的`Untitled`笔记本。为了保持最佳实践的连贯性，在继续之前，请确保将笔记本重命名为`verify_python_packages`。

## 检查 pandas

验证每个 Python 包是否可用的步骤与代码略有不同。第一个将是`pandas`，这将使完成常见的数据分析技术（如数据透视、清理、合并和分组数据集）变得更加容易，而无需返回到原始记录。

要验证`pandas`库是否在 Jupyter 中可用，请按照以下步骤操作：

1.  在`In []:`单元格中输入`import pandas as pd`。

1.  使用在*安装 Python 和使用 Jupyter Notebook*部分中讨论的先前方法运行单元格：

+   从工具栏中点击按钮。

+   从单元格菜单中选择“运行单元格”。

+   按下 *Shift* + *Enter* 或 *Ctrl* + *Enter* 键。

1.  在下一个 `In []:` 单元格中输入 `np.__version__` 命令。

1.  使用步骤 2 中推荐的方法运行单元格。

1.  验证显示为 `Out []` 的输出单元格。

`pandas` 的版本应该是 **0.18.0** 或更高。

现在，您将为本书中使用的以下每个必需的包重复这些步骤：`numpy`、`sklearn`、`matplotlib` 和 `scipy`。请注意，我已经使用了每个库的常用快捷名称，以使其与行业中的最佳实践保持一致。

例如，`pandas` 已缩短为 `pd`，因此当您从每个库调用功能时，您只需使用快捷名称即可。

根据所需的分析类型、数据输入的变体以及 Python 生态系统的进步，可以使用并应该使用额外的包。

## 检查 NumPy

**NumPy** 是 Python 的一种强大且常用的数学扩展，用于对称为数组的值列表执行快速数值计算。我们将在 第三章 “NumPy 入门” 中了解更多关于 NumPy 功能的强大之处。

要验证 `numpy` 库是否在 Jupyter 中可用，请按照以下步骤操作：

1.  在 `In []:` 单元格中输入 `import numpy as np`。

1.  使用在 *安装 Python 和使用 Jupyter Notebook* 部分中讨论的推荐方法运行单元格：

+   从工具栏中点击按钮。

+   从单元格菜单中选择“运行单元格”。

+   按下 *Shift* + *Enter* 或 *Ctrl* + *Enter* 键。

1.  在下一个 `In []:` 单元格中输入 `np.__version__` 命令。

1.  使用步骤 2 中推荐的方法运行单元格。

1.  验证显示为 `Out []` 的输出单元格。

NumPy 的版本应该是 **1.10.4** 或更高。

## 检查 sklearn

`sklearn` 是一个用于聚类和回归分析的先进开源数据科学库。虽然我们不会利用这个库的所有高级功能，但安装它将使未来的课程更容易进行。

要验证 `sklearn` 库是否在 Jupyter 中可用，请按照以下步骤操作：

1.  在 `In []:` 单元格中输入 `import sklearn as sk`。

1.  使用在 *安装 Python 和使用 Jupyter Notebook* 部分中讨论的推荐方法运行单元格：

+   从工具栏中点击按钮。

+   从单元格菜单中选择“运行单元格”。

+   按下 *Shift* + *Enter* 或 *Ctrl* + *Enter* 键。

1.  在下一个 `In []:` 单元格中输入 `sk.__version__` 命令。

1.  使用步骤 2 中推荐的方法运行单元格。

1.  验证显示为 `Out []` 的输出单元格。

`sklearn` 的版本应该是 **0.17.1** 或更高。

## 检查 Matplotlib

**Matplotlib** Python 库包用于使用 Python 进行数据可视化和绘制图表。

要验证 `matplotlib` 库是否在 Jupyter 中可用，请按照以下步骤操作：

1.  在 `In []:` 单元格中输入 `import matplotlib as mp`。

1.  使用在*安装 Python 和使用 Jupyter Notebook*部分中较早讨论的首选方法运行单元格：

+   点击工具栏中的按钮。

+   从单元格菜单中选择“运行单元格”。

+   按下*Shift* + *Enter*或*Ctrl* + *Enter*键。

1.  在下一个`In []:`单元格中输入`mp.__version__`命令。

1.  使用步骤 2 中讨论的首选方法运行单元格。

1.  验证显示为`Out []`的输出单元格。

`matplotlib`的版本应该是**1.5.1**或更高。

## 检查 SciPy

**SciPy**是一个依赖于 NumPy 的库，它包括用于数据分析的额外数学函数。

要验证`scipy`库是否在 Jupyter 中可用，请按照以下步骤操作：

1.  在`In []:`单元格中输入`in import scipy as sc`。

1.  使用在*安装 Python 和使用 Jupyter Notebook*部分中讨论的首选方法运行单元格：

+   点击工具栏中的按钮。

+   从单元格菜单中选择“运行单元格”。

+   按下*Shift* + *Enter*或*Ctrl* + *Enter*键。

1.  在下一个`In []:`单元格中输入`sc.__version__`命令。

1.  使用步骤 2 中讨论的首选方法运行单元格。

1.  验证显示为`Out []`的输出单元格。

`scipy`的版本应该是**0.17.0**或更高。

一旦完成所有步骤，你的笔记本应该看起来类似于以下截图：

![](img/f32216ab-105d-4160-8239-345deeb542c2.png)

# 摘要

恭喜，我们现在已经设置好了一个可以用于处理数据的环境。我们首先使用名为 Anaconda 的`conda`包安装程序安装了 Python 和 Jupyter 笔记本应用程序。接下来，我们启动了 Jupyter 应用程序，并讨论了如何导航仪表板和笔记本的所有功能。我们创建了一个工作目录，它可以作为所有数据分析项目的模板。

我们通过创建一个`hello_world`笔记本来运行我们的第一个 Python 代码，并介绍了 Jupyter 中可用的核心功能。最后，我们验证并探索了不同的 Python 包（NumPy、pandas、sklearn、Matplotlib 和 SciPy）及其在数据分析中的应用。你现在应该已经熟悉并准备好在 Jupyter 笔记本中运行额外的 Python 代码命令了。

在下一章中，我们将通过一些实践课程来扩展你的数据素养技能。我们将讨论 NumPy 的基础库，它用于分析称为数组的结构化数据。

# 未来阅读

这里有一些链接，你可以参考以获取更多关于本章相关主题的信息：

+   Python 的历史：[`docs.python.org/3/license.html`](https://docs.python.org/3/license.html)

+   `pip`和`conda`Python 包管理器的区别：[`stackoverflow.com/questions/20994716/what-is-the-difference-between-pip-and-conda`](https://stackoverflow.com/questions/20994716/what-is-the-difference-between-pip-and-conda)

+   理解`conda`和`pip`：[`www.anaconda.com/understanding-conda-and-pip/`](https://www.anaconda.com/understanding-conda-and-pip/)

+   Jupyter Notebook 教程：[`www.dataquest.io/blog/jupyter-notebook-tutorial/`](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

+   云端 Jupyter Notebook 服务的比较：[`discourse.jupyter.org/t/in-depth-comparison-of-cloud-based-services-that-run-jupyter-notebook/460/7`](https://discourse.jupyter.org/t/in-depth-comparison-of-cloud-based-services-that-run-jupyter-notebook/460/7)

+   JupyterLab 简介：**[`ipython-books.github.io/36-introducing-jupyterlab/`](https://ipython-books.github.io/36-introducing-jupyterlab/**) 

+   修改 Jupyter 启动文件夹的参考信息：[`stackoverflow.com/questions/35254852/how-to-change-the-jupyter-start-up-folder`](https://stackoverflow.com/questions/35254852/how-to-change-the-jupyter-start-up-folder)

+   Jupyter 项目的历史：[`github.com/jupyter/design/wiki/Jupyter-Logo`](https://github.com/jupyter/design/wiki/Jupyter-Logo)

+   安装 Jupyter 后文件和目录位置的参考信息：[`jupyter.readthedocs.io/en/latest/projects/jupyter-directories.html`](https://jupyter.readthedocs.io/en/latest/projects/jupyter-directories.html)

+   在 Jupyter 中处理不同文件类型：[`jupyterlab.readthedocs.io/en/stable/user/file_formats.html`](https://jupyterlab.readthedocs.io/en/stable/user/file_formats.html)

+   微软托管 Jupyter Notebook 网站：[`notebooks.azure.com/`](https://notebooks.azure.com/)

+   GitHub 上公共 Jupyter Notebook 的数量：[`github.com/parente/nbestimate`](https://github.com/parente/nbestimate)
