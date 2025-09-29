# 第七章：打包和分发你的应用程序

我们现在已经到达了应用程序开发过程的最后一步。我们有一个包含许多基本 GIS 功能的工作应用程序。然而，到目前为止，它只能在我们的特定开发环境中运行在我们的电脑上。如果你想让你之外的人从你的应用程序中受益，或者只是让你更容易携带和使用你的应用程序在多台电脑上，你需要打包应用程序，以便更容易安装。在本章中，我们将介绍以下最终步骤：

+   分配一个图标作为我们应用程序的标志

+   将你的开发环境转换为包含可执行文件（`.exe`）的自包含文件夹结构，以便运行你的应用程序

+   为你的应用程序提供一个安装向导以实现更持久的安装

# 附加应用程序标志

到目前为止，你可能已经注意到我们的应用程序在窗口的左上角显示了一个小而相当通用的红色图标，并且在打开的应用程序列表下方。这是 Tkinter GUI 应用程序的标准 Tkinter 标志，包括 Python 中的 IDLE 编辑器。对于你自己的应用程序，显然你想要自己的图标。

## 图标图像文件

首先，你必须找到或创建你想要的图标。现在，为了将标志分配给你的应用程序，它需要是 `.ico` 格式，这种格式包含同一图像在不同分辨率下的多个版本，以实现最佳显示。很可能会发现或创建的图像是一个普通的图像文件，例如 `.png`、`.bmp` 或 `.gif`，因此我们需要将其转换。我们将使用 Python 和 PIL 进行一次性处理，因为我们已经安装了它们。

使用这种 PIL 方法，我们可能会遇到一个小障碍，我们可能需要通过破解的方式绕过去。py2exe（我们将使用它来为我们的应用程序创建 EXE 文件）的在线文档警告我们，为了将图标分配给 EXE 文件，图标文件的各个分辨率保存的顺序很重要。大小必须按照从大到小的顺序分配，否则将无法工作。

在 PIL 中，我们遇到了一个障碍，在 2.8.1 或更低版本中，它会在幕后自动以从大到小的顺序排列图像大小，而不管你最初指定的顺序如何。幸运的是，当我提出问题时，PIL/Pillow 开发团队非常响应，所以问题已经得到解决，并且一旦下一个稳定版本 2.8.2 发布，应该不再成为问题。

如果新的修补过的 PIL 版本还没有发布，我们仍然可以自己轻松修复。尽管我们很勇敢，但我们还是深入到 PIL 的内部工作文件中，这些文件位于`C:/Python27/Lib/site-packages/PIL`。在`IcoImagePlugin.py`文件中，脚本的上部有一个`_save`函数。在那里你会看到它使用以下代码按从小到大的顺序对指定的尺寸参数进行排序：`sizes = sorted(sizes, key=lambda x: x[0])`。我们只需要删除或注释掉那行代码，这样用户就可以完全决定保存尺寸的顺序。

现在，我们已经准备好转换你选择的标志图像。我们只需要做一次，而且相当简单，所以我们就在交互式 Python Shell 窗口中这样做，而不是使用常规的文件编辑器。如果你已经在 Python IDLE 文件编辑器中，只需在顶部菜单中点击**运行**从 Python Shell。本质上我们只是导入 PIL，加载你选择的图像文件，并将其保存到一个新的`.ico`扩展名的文件中。在保存时，我们给出一个包含我们想要支持的图标分辨率的宽高元组的尺寸参数列表，按降序排列。将此图标图像保存到`pythongis/app`文件夹中是有意义的。运行以下命令：

```py
>>> import PIL, PIL.Image
>>> img = PIL.Image.open("your/path/to/icon.png")
>>> img.save("your/path/to/pythongis/app/icon.ico", sizes=[(255,255),(128,128),(64,64),(48,48),(32,32),(16,16),(8,8)])

```

## 分配图标

现在我们有了图标文件，我们可以将其分配给我们的应用程序。这是通过将图标分配给 Tkinter 来完成的，它将我们的图标放置在应用程序窗口的左上角，并在活动应用程序的 Windows 任务栏下方。我们在`app/builder.py`文件中的`run`函数中这样做，只需将我们的根应用程序窗口指向图标的路径。图标文件与`app/builder.py`在同一文件夹中，因此有人可能会认为到`logo.ico`的相对路径就足够了，但显然，对于分配 GUI 图标这个特定任务，Tkinter 需要完整的绝对路径。为此，我们利用全局`__file__`变量，它指向运行脚本的绝对路径：

```py
    # assign logo from same directory as this file
    import sys, os
    curfolder,curfile = os.path.split(__file__)
    logopath = os.path.join(curfolder, "logo.ico")
    window.iconbitmap(logopath)
```

如果你现在运行应用程序，你应该会看到图标出现在左上角和底部。尽管我们已经告诉 Tkinter 在应用程序内部使用图标，但这不会影响我们在 Windows 资源管理器中浏览和查看 EXE 文件时的外观。我们将如何在接下来的打包和创建 EXE 文件的过程中看到这一点。

# 应用程序启动脚本

由于我们想要一个可以打开并运行我们的应用程序的 EXE 文件，我们需要一个脚本，该脚本明确定义了如何启动我们的应用程序。我们用于本书整个测试目的的`guitester.py`脚本正是这样做的。因此，我们将我们的测试脚本重命名为`mygisapp.py`（或你希望给你的应用程序取的任何名字）。我们的主`pythongis`文件夹的位置应该如下所示：

![应用程序启动脚本](img/5407OS_07_01.jpg)

由于我们所做的只是将之前的 `guitester.py` 脚本重命名为 `mygisapp.py`，内容应该保持不变，它看起来应该是这样的：

```py
import pythongis as pg
pg.app.run()
```

# 包装你的应用程序

应用程序启动定义好后，我们现在就可以准备包装它了。包装我们的应用程序意味着我们的应用程序将包含所有必要的文件，这些文件被分组到一个文件夹树中（目前它们散布在您电脑的多个位置），以及一个用户可以双击以运行应用程序的 EXE 文件。

## 安装 py2exe

Python 中有许多用于包装项目的库，我们选择使用 py2exe，因为它非常容易安装：

1.  前往 [www.py2exe.org](http://www.py2exe.org)。

1.  点击顶部的 **Download** 链接，它将带你去到 [`sourceforge.net/projects/py2exe/files/py2exe/0.6.9/`](http://sourceforge.net/projects/py2exe/files/py2exe/0.6.9/)。

1.  下载并运行适用于 Python 2.7 的最新版本，目前是 `py2exe-0.6.9.win32-py2.7.exe`。

    ### 注意

    py2exe 是针对 Windows 平台的；你必须在 Windows 上构建，并且你的程序只能在 Windows 上使用。

    Windows 的另一个替代方案将是 PyInstaller：[`pythonhosted.org/PyInstaller/`](http://pythonhosted.org/PyInstaller/)。

    Mac OS X 的对应工具是 py2app：[`pythonhosted.org/py2app/`](https://pythonhosted.org/py2app/)。

    对于 Linux，你可以使用 cx_Freeze：[`cx-freeze.sourceforge.net/`](http://cx-freeze.sourceforge.net/)。

## 制定包装策略

包装应用程序有许多方法，所以在我们深入之前，我们应该首先了解 py2exe 的工作原理并相应地制定一个包装策略。给定一个用户想要包装的脚本，py2exe 做的是遍历脚本，递归地检测所有导入语句，从而确定哪些库必须包含在最终的包中。然后它创建一个名为 `dist` 的文件夹（它还创建了一个名为 `build` 的文件夹，但对我们来说那个是不相关的），这个文件夹变成了包含所有必需文件和基于我们的启动脚本的 EXE 文件的发行文件夹。

一个关键的决定是我们如何选择捆绑我们的包。我们可以将大多数所需的文件和依赖项捆绑到 EXE 文件本身或 ZIP 文件中，或者不捆绑任何东西，保持所有内容在文件夹结构中松散。起初，捆绑可能看起来是最整洁和最佳组织的选择。不幸的是，py2exe（与其他打包库一样）通常无法正确检测或复制所有必要的文件（尤其是`.dll`和`.pyc`文件）从依赖项中，导致我们的应用程序启动失败。我们可以指定一些选项来帮助 py2exe 正确检测和包含所有内容，但对于大型项目来说，这可能会变得繁琐，并且仍然可能无法纠正每个错误。通过将所有内容作为文件和文件夹而不是捆绑起来，我们实际上可以在 py2exe 完成工作后进入并纠正 py2exe 犯的一些错误。

使用非捆绑方法，我们可以获得更大的控制权，因为 EXE 文件变得像 Python 解释器一样，`dist`文件夹顶层的所有内容都变成了 Python 的`site-packages`文件夹，用于导入库。这样，通过手动将依赖项完整地从`site-packages`复制到`dist`文件夹，它们就可以以与 Python 通常从`site-packages`导入它们相同的方式导入。py2exe 将检测并正确处理我们的内置 Python 库的导入，但对于更高级的第三方依赖项，包括我们的主要`pythongis`库，我们希望自行添加。我们可以在创建下一个构建脚本时将这种策略付诸实践。

## 创建构建脚本

要打包一个项目，py2exe 需要一个非常简单的指令脚本。将其保存为与我们的主`pythongis`文件夹位于同一目录下的`setup.py`。以下是目录结构层次：

![创建构建脚本](img/5407OS_07_02.jpg)

我们从`setup.py`文件开始，通过链接到应由 EXE 文件运行的`mygisapp.py`启动脚本，并指向我们的图标文件路径，这样当浏览时 EXE 文件看起来就会是这样。在选项中，我们根据我们的非捆绑策略将`skip_archive`设置为`True`。我们还阻止 py2exe 尝试从`pyagg`包中读取和复制两个二进制文件，这会导致不必要的错误，因为这些文件只是为了跨版本和跨平台兼容性而提供的。

如果在应用程序演变过程中遇到其他构建错误，可以使用`dll_excludes`忽略`.dll`和`.pyd`文件或模块或包的排除，这可以是一个好的方法来忽略这些错误，并在构建后复制粘贴所需的文件。以下是我们刚才描述的步骤的代码，写在`setup.py`脚本中：

```py
############
### allow building the exe by simply running this script
import sys
sys.argv.append("py2exe") 

############
### imports
from distutils.core import setup
import py2exe

###########
### options
WINDOWS = [{"script": "mygisapp.py",
            "icon_resources": [(1,"pythongis/app/logo.ico")] }]
OPTIONS = {"skip_archive": True,
           "dll_excludes": ["python26.dll","python27.so"],
           "excludes": [] }

###########
### build
setup(windows=WINDOWS,
      options={"py2exe": OPTIONS}
      )
```

`setup`函数将在`setup.py`旁边的`dist`文件夹和`pythongis`文件夹中构建。正如我们在包装策略中之前所述，如果第三方库有如`.dll`、`.pyd`、图片或其他数据文件的高级文件布局，py2exe 可能无法正确复制所有这些库。因此，我们选择在脚本中添加一些额外的代码，在构建过程之后将更高级的依赖项，如`PIL`、`Pyagg`、`Rtree`和`Shapely`从`site-packages`（假设你没有将它们安装到其他位置）以及我们整个`pythongis`库复制并覆盖到`dist`文件夹中。你必须确保`site-packages`的路径与你的平台匹配。

```py
###########
### manually copy pythongis package to dist
### ...because py2exe may not copy all files
import os
import shutil
frompath = "pythongis"
topath = os.path.join("dist","pythongis")
shutil.rmtree(topath) # deletes the folder copied by py2exe
shutil.copytree(frompath, topath)

###########
### and same with advanced dependencies
### ...only packages, ie folders
site_packages_folder = "C:/Python27/Lib/site-packages"
advanced_dependencies = ["PIL", "pyagg", "rtree", "shapely"]
for dependname in advanced_dependencies:
    frompath = os.path.join(site_packages_folder, dependname)
    topath = os.path.join("dist", dependname)
    shutil.rmtree(topath) # deletes the folder copied by py2exe
    shutil.copytree(frompath, topath) 
```

创建了`setup.py`脚本后，你只需运行脚本以打包你的应用程序。py2exe 复制所有内容可能需要一分钟左右。完成后，将在与`setup.py`和`pythongis`相同的文件夹中有一个可用的`dist`文件夹：

![创建构建脚本](img/5407OS_07_03.jpg)

在`dist`文件夹内，将有一个`mygisapp.exe`文件（假设这是你的启动脚本名称），它应该看起来像你选择的图标，并且当运行时，应该成功启动你的类似图标的应用程序窗口。当你在`dist`文件夹内时，检查 py2exe 是否意外包含了你试图避免的任何库。例如，Shapely 有可选的 NumPy 支持，并且会尝试导入 NumPy，这会导致 py2exe 即使你没有使用它也会将其添加到你的`dist`文件夹中。通过将不想要的包添加到设置脚本中的排除选项来避免这种情况。

## 添加 Visual C 运行时 DLL

如果你是在 Windows 上操作，在我们应用程序完全独立之前，还有最后一个至关重要的步骤。Python 编程环境依赖于我们在安装 Python 时包含的 Microsoft Visual C 运行时 DLL。然而，存在许多版本的此 DLL，因此并非所有计算机或用户都会有我们应用程序需要的特定版本。py2exe 默认不会包含所需的 DLL，因此我们必须将其包含在我们的`dist`文件夹中。在安装中包含 DLL 是一个简单的复制和粘贴过程，按照以下步骤操作：

1.  尽管从技术上讲，我们已经在电脑上某个地方有了 DLL 文件，但我认为找到正确的一个有足够的变数和陷阱，因此最好是通过干净安装（免费的）Microsoft Visual C redistributable 程序来获取它。下载并安装 Python 版本所使用的版本，对于 32 位 Python 2.7，是**Microsoft Visual C++ 2008 Redistributable Package (x86**)，可以从[`www.microsoft.com/download/en/details.aspx?displaylang=en&id=29`](http://www.microsoft.com/download/en/details.aspx?displaylang=en&id=29)获取。

    ### 注意

    对于其他 Python 版本和位架构及其所需的 VC++和 DLL 版本的概述，请参阅这篇出色的帖子：

    [`stackoverflow.com/questions/9047072/windows-python-version-and-vc-redistributable-version`](http://stackoverflow.com/questions/9047072/windows-python-version-and-vc-redistributable-version)

1.  安装完成后，转到新安装的文件夹，它应该是类似于`C:\Program Files\Microsoft Visual Studio 9.0\VC\redist\x86\`的路径，尽管这可能会根据你的版本和位架构而有所不同。

1.  一旦到了那里，你将找到一个名为`Microsoft.VC90.CRT`的文件夹，其中包含以下文件：

    +   `Microsoft.VC90.CRT.manifest`

    +   `msvcm90.dll`

    +   `msvcp90.dll`

    +   `msvcr90.dll`

    作为免费许可证的一部分，Microsoft 要求你将整个文件夹包含在你的应用程序中，所以请将其复制到你的`dist`文件夹中。

1.  现在，你的 EXE 文件应该总是能够找到所需的 DLLs。如果你遇到麻烦或需要更多信息，请查看官方 py2exe 教程中的 DLL 部分，链接为[`www.py2exe.org/index.cgi/Tutorial#Step5`](http://www.py2exe.org/index.cgi/Tutorial#Step5)。

你现在已经成功打包了你的应用程序，并使其变得便携！注意整个应用程序仅重 30MB，这使得上传、下载甚至通过电子邮件发送变得轻而易举。如果你使用推荐的 32 位 Python 构建了应用程序和包，你的程序应该能在任何 Windows 7 或 8 计算机上运行（在 Python 和 EXE 文件的眼中，它们基本上是相同的）。这包括 32 位和 64 位 Windows，因为 64 位代码与 32 位代码向后兼容。如果你使用了 64 位 Python，它将仅适用于那些拥有 64 位 Windows 的用户，这并不理想。

# 创建安装程序

到目前为止，你可以理论上将你的`dist`文件夹作为一个便携式 GIS 应用程序放在 U 盘上，或者通过 ZIP 存档与他人共享。这在一定程度上是可以的，但如果你希望面向更广泛的受众，这不是分发应用程序最专业或最可信的方式。为了运行应用程序，用户（包括你自己）必须在一个他们不理解的长名单中找到 EXE 文件。这仅仅是太多应该直接从盒子里出来的血腥细节和手动工作。

更常见的是，人们习惯于接收一个安装程序文件，该文件指导用户在更永久的位置安装程序，并从他们那里创建快捷方式。这不仅看起来更专业，而且也处理了用户的高级步骤。作为最终步骤，我们将为我们的 GIS 应用程序创建这样一个安装程序，使用广泛推荐的安装软件**Inno Setup**。

## 安装 Inno Setup

要安装 Inno Setup，请按照以下步骤操作：

1.  前往[`www.jrsoftware.org/`](http://www.jrsoftware.org/)。

1.  点击左侧的**Inno Setup**链接。

1.  点击左侧的**下载**链接。

1.  在稳定标题下，下载并安装名为`isetup-5.5.5.exe`的文件。

## 设置应用程序的安装程序

一旦运行 Inno Setup，您将看到一个欢迎屏幕，您应该选择**使用脚本向导创建一个新的脚本文件**，如图所示：

![设置应用程序的安装程序](img/5407OS_07_04.jpg)

这为您提供了一个逐步向导，您将执行以下操作：

1.  在向导的第一屏，保持复选框未勾选，并点击**下一步**。

1.  在第二屏，提供应用程序的名称和版本，以及出版者名称和适用的网站。

1.  在第三屏，保留默认的安装位置。

1.  第四屏是最关键的一屏：您通过点击**添加文件夹**（您可能需要将其重命名为应用程序的名称）来告诉安装程序您的 EXE 文件位置以及整个自包含的`dist`文件夹位置。

1.  在第五屏，保留默认的开始菜单选项。

1.  在第六屏，您可以提供许可证文本文件，以及/或一些自定义信息文本，用于显示在安装的开始和结束部分。

1.  在第七屏，选择安装程序的语言。

1.  在第八屏，将**自定义编译器输出文件夹**设置为您的应用程序名称（安装后的程序文件夹名称），将**编译器输出基本文件名**（安装程序文件名称）设置为`[您的应用程序名称]_setup`，并将**自定义安装图标文件**设置为之前创建的图标。

1.  在第九屏，点击**完成**以创建安装程序。

1.  当提示保存设置脚本时，选择**是**，并将其保存与您的`setup.py`脚本一起，以便您可以在以后重建或修改安装程序。

这样，您现在应该有一个带有您图标的应用程序安装文件，它将引导用户安装您新创建的 GIS 应用程序。您所有的辛勤工作现在都整齐地封装在一个文件中，最终可以与更广泛的受众共享和使用。

# 摘要

在本章中，您完成了创建 GIS 应用程序的最终打包步骤。您通过给应用程序添加一个显示在可执行文件和应用程序窗口中的标志图标来给它一个完美的收尾。然后，我们将应用程序打包在一个自包含的文件夹中，可以在任何 Windows 7 或 8 计算机上运行（包括 32 位和 64 位系统，前提是您使用了 32 位 Python）。最后，我们通过创建一个安装向导来给应用程序一个更“官方”的介绍和安装，使其看起来更专业。您的应用程序的最终用户不需要知道 Python 编程或它被用来制作程序的事实。他们唯一需要的是运行您友好的安装文件，然后他们可以通过点击 Windows 桌面或开始菜单上新添加的快捷方式来开始使用您的应用程序。

完成了从零开始到结束制作一个简单的 Python GIS 应用程序的步骤后，继续阅读最后一章，我们将快速回顾所学到的知识，并考虑为你进一步扩展和定制你自己的应用程序的可能路径和建议。
