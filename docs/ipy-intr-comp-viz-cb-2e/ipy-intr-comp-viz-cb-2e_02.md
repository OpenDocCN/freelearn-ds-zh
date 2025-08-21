# 第二章：交互式计算中的最佳实践

在本章中，我们将涵盖以下主题：

+   选择（或不选择）Python 2 和 Python 3

+   使用 IPython 进行高效的交互式计算工作流

+   学习分布式版本控制系统 Git 的基础

+   使用 Git 分支的典型工作流

+   进行可重现交互式计算实验的十条技巧

+   编写高质量的 Python 代码

+   使用 nose 编写单元测试

+   使用 IPython 调试代码

# 引言

这是关于交互式计算中良好实践的特别章节。如果本书的其余部分讨论的是内容，那么本章讨论的是形式。它描述了如何高效且正确地使用本书所讨论的工具。我们将在讨论可重现计算实验之前，先介绍版本控制系统 Git 的基本要点（特别是在 IPython notebook 中）。

我们还将讨论一些软件开发中的一般主题，例如代码质量、调试和测试。关注这些问题可以极大地提高我们最终产品的质量（例如软件、研究和出版物）。我们这里只是略微涉及，但你将找到许多参考资料，帮助你深入了解这些重要主题。

# 选择（或不选择）Python 2 和 Python 3

在这第一个食谱中，我们将简要讨论一个横向且有些平凡的话题：Python 2 还是 Python 3？

**Python 3** 自 2008 年推出以来，许多 Python 用户仍然停留在 Python 2。通过改进了 Python 2 的多个方面，Python 3 打破了与之前版本的兼容性。因此，迁移到 Python 3 可能需要投入大量精力。

即使没有太多破坏兼容性的变化，一个在 Python 2 中运行良好的程序可能在 Python 3 中完全无法运行。例如，你的第一个 `Hello World` Python 2 程序在 Python 3 中无法运行；`print "Hello World!"` 在 Python 3 中会引发 `SyntaxError`。实际上，`print` 现在是一个函数，而不是一个语句。你应该写成 `print("Hello World!")`，这在 Python 2 中也能正常工作。

无论你是开始一个新项目，还是需要维护一个旧的 Python 库，选择 Python 2 还是 Python 3 的问题都会出现。在这里，我们提供一些论点和提示，帮助你做出明智的决策。

### 注意

当我们提到 Python 2 时，我们特别指的是 Python 2.6 或 Python 2.7，因为这些 Python 2.*x* 的最后版本比 Python 2.5 及更早版本更接近 Python 3。支持 Python 2.5+ 和 Python 3.*x* 同时运行更为复杂。

同样，当我们提到 Python 3 或 Python 3.*x* 时，我们特别指的是 Python 3.3 或更高版本。

## 如何做到……

首先，Python 2 和 Python 3 之间有什么区别？

### Python 3 相对于 Python 2 的主要差异

这里是一些差异的部分列表：

+   `print` 不再是一个语句，而是一个函数（括号是必须的）。

+   整数除法会返回浮点数，而不是整数。

+   一些内置函数返回的是迭代器或视图，而不是列表。例如，`range` 在 Python 3 中的行为类似于 Python 2 中的 `xrange`，而后者在 Python 3 中已经不存在。

+   字典不再有 `iterkeys()`、`iteritems()` 和 `itervalues()` 方法了。你应该改用 `keys()`、`items()` 和 `values()` 函数。

+   以下是来自官方 Python 文档的引用：

    > *“你曾经以为自己了解的二进制数据和 Unicode，现在都变了。”*

+   使用 `%` 进行字符串格式化已不推荐使用；请改用 `str.format`。

+   `exec` 是一个函数，而不是一个语句。

Python 3 带来了许多关于语法和标准库内容的改进和新特性。你将在本食谱末尾的参考资料中找到更多细节。

现在，你的项目基本上有两种选择：坚持使用单一分支（Python 2 或 Python 3），或同时保持与两个分支的兼容性。

### Python 2 还是 Python 3？

自然，许多人会偏爱 Python 3；它代表着未来，而 Python 2 是过去。为什么还要去支持一个已被废弃的 Python 版本呢？不过，这里有一些可能需要保持 Python 2 兼容性的情况：

+   你需要维护一个用 Python 2 编写的大型项目，而更新到 Python 3 会花费太高（即使存在半自动化的更新工具）。

+   你的项目有一些依赖项无法与 Python 3 一起使用。

    ### 注意

    本书中我们将使用的大多数库支持 Python 2 和 Python 3。并且本书的代码兼容这两个分支。

+   你的最终用户所使用的环境不太支持 Python 3。例如，他们可能在某个大型机构工作，在许多服务器上部署新的 Python 版本成本太高。

在这种情况下，你可以选择继续使用 Python 2，但这意味着你的代码可能在不久的将来变得过时。或者，你可以选择 Python 3 和它那一堆崭新的功能，但也有可能会把 Python 2 的用户抛在后头。你也可以用 Python 2 编写代码，并为 Python 3 做好准备。这样，你可以减少将来迁移到 Python 3 时所需的修改。

幸运的是，你不一定要在 Python 2 和 Python 3 之间做出选择。实际上，有方法可以同时支持这两个版本。即使这可能比单纯选择一个分支多花一些工作，但在某些情况下它可能会非常有趣。不过，需要注意的是，如果选择这种做法，你可能会错过许多仅支持 Python 3 的特性。

### 同时支持 Python 2 和 Python 3

支持两个分支的基本方法有两种：使用 **2to3** 工具，或者编写在两个分支中都能*正常运行*的代码。

### 使用 2to3 工具

`2to3` 是标准库中的一个程序，能够自动将 Python 2 代码转换为 Python 3。例如，运行 `2to3 -w example.py` 可以将单个 Python 2 模块迁移到 Python 3。你可以在 [`docs.python.org/2/library/2to3.html`](https://docs.python.org/2/library/2to3.html) 找到更多关于 2to3 工具的信息。

你可以配置安装脚本，使得 `2to3` 在用户安装你的包时自动运行。Python 3 用户将自动获得转换后的 Python 3 版本的包。

这个解决方案要求你的程序有一个坚实的测试套件，并且有一个持续集成系统，能够测试 Python 2 和 Python 3（请参阅本章稍后的单元测试食谱）。这是确保你的代码在两个版本中都能正常工作的方式。

### 编写在 Python 2 和 Python 3 中都能运行的代码。

你还可以编写既能在 Python 2 中运行，又能在 Python 3 中运行的代码。如果从头开始一个新项目，这个解决方案会更简单。一个广泛使用的方法是依赖一个轻量且成熟的模块，名为 **six**，由 Benjamin Peterson 开发。这个模块只有一个文件，因此你可以轻松地将其与包一起分发。无论何时你需要使用一个仅在某个 Python 分支中支持的函数或特性时，都需要使用 six 中实现的特定函数。这个函数要么包装，要么模拟相应的功能，因此它能在两个分支中正常工作。你可以在 [`pythonhosted.org/six/`](http://pythonhosted.org/six/) 上找到关于 six 的更多信息。

这种方法要求你改变一些习惯。例如，在 Python 2 中迭代字典的所有项时，你会写如下代码：

```py
for k, v in d.iteritems():
    # ...
```

现在，不再使用前面的代码，而是使用 six 编写以下代码：

```py
from six import iteritems
for k, v in iteritems(d):
    # ...
```

Python 2 中字典的 `iteritems()` 方法在 Python 3 中被 `items()` 替代。six 模块的 `iteritems` 函数根据 Python 版本内部调用一个方法。

### 提示

**下载示例代码**

你可以从你的账号中下载所有已购买的 Packt 图书的示例代码文件，网址为 [`www.packtpub.com`](http://www.packtpub.com)。如果你是在其他地方购买的此书，可以访问 [`www.packtpub.com/support`](http://www.packtpub.com/support) 注册并直接将文件通过电子邮件发送给你。

## 还有更多...

正如我们所看到的，关于 Python 2 或 Python 3 的问题，有很多选择可以考虑。简而言之，你应该考虑以下选项：

+   请仔细决定是否真的需要支持 Python 2：

    +   如果是，请通过避免使用 Python 2 专有的语法或特性，为 Python 3 准备好你的代码。你可以使用 six、2to3 或类似的工具。

    +   如果没有必要，坚决使用 Python 3。

+   在所有情况下，确保你的项目拥有一个坚实的测试套件、出色的代码覆盖率（接近 100%），并且有一个持续集成系统，能够在你正式支持的所有 Python 版本中测试你的代码。

这里有几个相关的参考资料：

+   一本关于将代码迁移到 Python 3 的优秀免费书籍，作者：Lennart Regebro，访问地址：[`python3porting.com/`](http://python3porting.com/)

+   关于将代码迁移到 Python 3 的官方推荐，访问地址：[`docs.python.org/3/howto/pyporting.html`](https://docs.python.org/3/howto/pyporting.html)

+   关于 Python 2/Python 3 问题的官方维基页面，访问地址：[`wiki.python.org/moin/Python2orPython3`](https://wiki.python.org/moin/Python2orPython3)

+   Nick Coghlan 提供的 Python 3 问答，访问地址：[`python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html`](http://python-notes.curiousefficiency.org/en/latest/python3/questions_and_answers.html)

+   *Python 3 中的新特性*，请见 [`docs.python.org/3.3/whatsnew/3.0.html`](https://docs.python.org/3.3/whatsnew/3.0.html)

+   *你无法使用的 Python 十大酷炫特性，因为你拒绝升级到 Python 3*，由 Aaron Meurer 提供的演讲，访问地址：[`asmeurer.github.io/python3-presentation/slides.html`](http://asmeurer.github.io/python3-presentation/slides.html)

+   在编写兼容性代码时使用 `__future__` 模块，访问地址：[`docs.python.org/2/library/__future__.html`](https://docs.python.org/2/library/__future__.html)

+   Python 2 和 Python 3 的关键区别，访问地址：[`sebastianraschka.com/Articles/2014_python_2_3_key_diff.html`](https://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html)

## 另见

+   *编写高质量 Python 代码* 这篇食谱

+   *使用 nose 编写单元测试* 这篇食谱

# 使用 IPython 进行高效的交互式计算工作流

有多种方法可以使用 IPython 进行交互式计算。其中一些在灵活性、模块化、可重用性和可复现性方面更为优秀。我们将在本节中回顾和讨论这些方法。

任何交互式计算工作流都基于以下循环：

+   编写一些代码

+   执行它

+   解释结果

+   重复

这个基本循环（也叫做**读取-求值-打印循环**，或称**REPL**）在进行数据或模型模拟的探索性研究时特别有用，或者在逐步构建复杂算法时也很有用。一个更经典的工作流（编辑-编译-运行-调试循环）通常是编写一个完整的程序，然后进行全面分析。这种方法通常比较繁琐。更常见的做法是通过做小规模实验、调整参数来迭代地构建算法解决方案，这正是交互式计算的核心。

**集成开发环境**（**IDEs**），提供了全面的软件开发设施（如源代码编辑器、编译器和调试器），广泛应用于经典工作流。然而，在交互式计算方面，存在一些替代 IDE 的工具。我们将在这里进行回顾。

## 如何实现...

以下是几个可能的交互式计算工作流程，按复杂度递增排列。当然，IPython 是所有这些方法的核心。

### IPython 终端

IPython 是 Python 中交互式计算的*事实标准*。IPython 终端（`ipython`命令）提供了一个专门为 REPL 设计的命令行界面。它比本地 Python 解释器（`python`命令）更强大。IPython 终端是一个便捷的工具，用于快速实验、简单的 Shell 交互以及查找帮助。忘记了 NumPy 的` savetxt`函数的输入参数吗？只需在 IPython 中输入`numpy.savetxt?`（当然，你首先需要使用`import numpy`）。有些人甚至把 IPython 终端当作（复杂的）计算器来使用！

然而，当单独使用时，终端很快变得有限。主要问题是终端不是一个代码编辑器，因此输入超过几行的代码可能会变得不方便。幸运的是，有多种方法可以解决这个问题，下面的章节将详细介绍这些方法。

### IPython 与文本编辑器

*非文本编辑器*问题的最简单解决方案或许并不令人意外，那就是结合使用 IPython 和文本编辑器。在这种工作流程中，`%run`魔法命令成为了核心工具：

+   在你喜欢的文本编辑器中编写一些代码，并将其保存在`myscript.py` Python 脚本文件中。

+   在 IPython 中，假设你处于正确的目录，输入`%run myscript.py`。

+   脚本被执行。标准输出会实时显示在 IPython 终端中，并且会显示可能的错误。脚本中定义的顶级变量在脚本执行完毕后，可以在 IPython 终端中访问。

+   如果需要在脚本中进行代码更改，请重复此过程。

### 提示

通过适当的键盘快捷键，IPython-文本编辑器工作流程可以变得更加高效。例如，你可以自动化你的文本编辑器，当按下*F8*时，在正在运行的 IPython 解释器中执行以下命令：

```py
%run <CURRENT_FILE_NAME>

```

这里描述了这种方法（在 Windows 上，使用 Notepad++和 AutoHotKey）：

[`cyrille.rossant.net/python-ide-windows/`](http://cyrille.rossant.net/python-ide-windows/)

使用一个好的文本编辑器，这个工作流程可以非常高效。由于每次执行`%run`时都会重新加载脚本，因此你的更改会自动生效。当你的脚本导入了其他 Python 模块并且你修改了这些模块时，情况会变得更复杂，因为这些模块不会随着`%run`被重新加载。你可以使用深度重新加载来解决这个问题：

```py
import myscript
from IPython.lib.deepreload import reload as dreload
dreload(myscript)
```

`myscript`中导入的模块将会被重新加载。一个相关的 IPython 魔法命令是`%autoreload`（你首先需要执行`%load_ext autoreload`）。此命令会尝试自动重新加载交互命名空间中导入的模块，但并不总是成功。你可能需要显式地重新加载已更改的模块，使用`reload(module)`（在 Python 2 中）或`imp.reload(module)`（Python 3 中）。

### IPython 笔记本

IPython 笔记本在高效的交互式工作流程中起着核心作用。它是代码编辑器和终端的精心设计的结合，将两者的优点融为一体，提供一个统一的环境。

你可以在笔记本的单元格中开始编写所有代码。你可以在同一地方编写、执行和测试代码，从而提高生产力。你可以在 Markdown 单元格中加入长注释，并使用 Markdown 标题来结构化你的笔记本。

一旦你的一部分代码足够成熟且不再需要进一步修改，你可以（并且应该）将它们重构为可重用的 Python 组件（函数、类和模块）。这将清理你的笔记本，并促进代码的未来重用。需要强调的是，不断将代码重构为可重用组件非常重要。IPython 笔记本当前不容易被第三方代码重用，并且它们并不针对这一点进行设计。笔记本适合进行初步分析和探索性研究，但它们不应该阻止你定期清理并将代码重构为 Python 组件。

笔记本的一个主要优势是，它们能为你提供一份文档，记录你在代码中所做的一切。这对可重复研究极为有用。笔记本保存在人类可读的 JSON 文档中，因此它们与 Git 等版本控制系统相对兼容。

### 集成开发环境

集成开发环境（IDEs）特别适用于经典的软件开发，但它们也可以用于交互式计算。一款好的 Python IDE 将强大的文本编辑器（例如，包含语法高亮和自动补全功能的编辑器）、IPython 终端和调试器结合在统一的环境中。

对于大多数平台，有多个商业和开源 IDE。**Eclipse**/**PyDev** 是一个流行的（尽管略显笨重的）开源跨平台环境。**Spyder** 是另一个开源 IDE，具有良好的 IPython 和 matplotlib 集成。**PyCharm** 是众多支持 IPython 的商业环境之一。

微软的 Windows IDE，Visual Studio，有一个名为 **Python Tools for Visual Studio**（**PTVS**）的开源插件。这个工具为 Visual Studio 带来了 Python 支持。PTVS 原生支持 IPython。你不一定需要付费版本的 Visual Studio；你可以下载一个免费的包，将 PTVS 和 Visual Studio 打包在一起。

## 还有更多…

以下是一些 Python IDE 的链接：

+   [`pydev.org`](http://pydev.org) 是 PyDev for Eclipse 的官方网站。

+   [`code.google.com/p/spyderlib/`](http://code.google.com/p/spyderlib/) 是 Spyder，一个开源 IDE。

+   [www.jetbrains.com/pycharm/](http://www.jetbrains.com/pycharm/) 是 PyCharm 的官方网站。

+   [`pytools.codeplex.com`](http://pytools.codeplex.com) 是微软 Visual Studio 在 Windows 上的 PyTools。

+   [`code.google.com/p/pyscripter/`](http://code.google.com/p/pyscripter/) 是 PyScripter 的官方网站。

+   [www.iep-project.org](http://www.iep-project.org) 为 IEP，Python 的交互式编辑器

## 另见

+   *学习分布式版本控制系统 Git 的基础* 方法

+   *使用 IPython 调试代码* 的方法

# 学习分布式版本控制系统 Git 的基础

使用 **分布式版本控制系统** 在当今已经变得非常自然，如果你正在阅读本书，你可能已经在使用某种版本控制系统。然而，如果你还没有，请认真阅读这个方法。你应该始终为你的代码使用版本控制系统。

## 准备工作

著名的分布式版本控制系统包括 **Git**、**Mercurial** 和 **Bazaar**。在这一章中，我们选择了流行的 Git 系统。你可以从 [`git-scm.com`](http://git-scm.com) 下载 Git 程序和 Git GUI 客户端。在 Windows 上，你也可以安装 **msysGit**（[`msysgit.github.io`](http://msysgit.github.io)）和 **TortoiseGit**（[`code.google.com/p/tortoisegit/`](https://code.google.com/p/tortoisegit/)）。

### 注意

与 SVN 或 CVS 等集中式系统相比，分布式系统通常更受欢迎。分布式系统允许本地（离线）更改，并提供更灵活的协作系统。

支持 Git 的在线服务商包括 **GitHub**（[`github.com`](https://github.com)）、**Bitbucket**（[`bitbucket.org`](https://bitbucket.org)）、**Google code**（[`code.google.com`](https://code.google.com)）、**Gitorious**（[`gitorious.org`](https://gitorious.org)）和 **SourceForge**（[`sourceforge.net`](https://sourceforge.net)）。在撰写本书时，所有这些网站创建账户都是免费的。GitHub 提供免费的无限制公共仓库，而 Bitbucket 提供免费的无限制公共和私有仓库。GitHub 为学术用户提供特殊功能和折扣（[`github.com/edu`](https://github.com/edu)）。将你的 Git 仓库同步到这样的网站，在你使用多台计算机时特别方便。

你需要安装 Git（可能还需要安装 GUI）才能使用此方法（参见 [`git-scm.com/downloads`](http://git-scm.com/downloads)）。我们还建议你在以下这些网站之一创建账户。GitHub 是一个很受欢迎的选择，特别是因为它用户友好的网页界面和发达的社交功能。GitHub 还提供了非常好的 Windows 客户端（[`windows.github.com`](https://windows.github.com)）和 Mac OS X 客户端（[`mac.github.com`](https://mac.github.com)）。我们在本书中使用的大多数 Python 库都是在 GitHub 上开发的。

## 如何操作…

我们将展示两种初始化仓库的方法。

### 创建一个本地仓库

这种方法最适合开始在本地工作时使用。可以通过以下步骤实现：

1.  在开始一个新项目或计算实验时，最先要做的就是在本地创建一个新文件夹：

    ```py
    $ mkdir myproject
    $ cd myproject

    ```

1.  我们初始化一个 Git 仓库：

    ```py
    $ git init

    ```

1.  让我们设置我们的姓名和电子邮件地址：

    ```py
    $ git config --global user.name "My Name"
    $ git config --global user.email "me@home"

    ```

1.  我们创建一个新文件，并告诉 Git 跟踪它：

    ```py
    $ touch __init__.py
    $ git add __init__.py

    ```

1.  最后，让我们创建我们的第一次提交：

    ```py
    $ git commit -m "Initial commit."

    ```

### 克隆一个远程仓库

当仓库需要与 GitHub 等在线提供商同步时，这种方法最好。我们来执行以下步骤：

1.  我们在在线提供商的网页界面上创建了一个新的仓库。

1.  在新创建项目的主页面上，我们点击**克隆**按钮并获取仓库 URL，然后在终端输入：

    ```py
    $ git clone /path/to/myproject.git

    ```

1.  我们设置我们的姓名和电子邮件地址：

    ```py
    $ git config --global user.name "My Name"
    $ git config --global user.email "me@home"

    ```

1.  让我们创建一个新文件并告诉 Git 跟踪它：

    ```py
    $ touch __init__.py
    $ git add __init__.py

    ```

1.  我们创建我们的第一次提交：

    ```py
    $ git commit -m "Initial commit."

    ```

1.  我们将本地更改推送到远程服务器：

    ```py
    $ git push origin

    ```

当我们拥有一个本地仓库（通过第一种方法创建）时，我们可以使用 `git remote add` 命令将其与远程服务器同步。

## 它是如何工作的…

当你开始一个新项目或新的计算实验时，在你的计算机上创建一个新文件夹。你最终会在这个文件夹中添加代码、文本文件、数据集和其他资源。分布式版本控制系统会跟踪你在项目发展过程中对文件所做的更改。它不仅仅是一个简单的备份，因为你对任何文件所做的每个更改都会保存相应的时间戳。你甚至可以随时恢复到之前的状态；再也不用担心破坏你的代码了！

具体来说，你可以随时通过执行**提交**来拍摄项目的快照。该快照包括所有已暂存（或已跟踪）的文件。你完全控制哪些文件和更改将被跟踪。使用 Git，你可以通过 `git add` 将文件标记为下次提交的暂存文件，然后用 `git commit` 提交你的更改。`git commit -a` 命令允许你提交所有已经被跟踪的文件的更改。

在提交时，你需要提供一个消息来描述你所做的更改。这样可以使仓库的历史更加详细和富有信息。

### 注

**你应该多频繁地提交？**

答案是非常频繁的。Git 只有在你提交更改时才会对你的工作负责。在两次提交之间发生的内容可能会丢失，所以你最好定期提交。此外，提交是快速且便宜的，因为它们是本地的；也就是说，它们不涉及与外部服务器的任何远程通信。

Git 是一个分布式版本控制系统；你的本地仓库不需要与外部服务器同步。然而，如果你需要在多台计算机上工作，或者如果你希望拥有远程备份，你应该进行同步。与远程仓库的同步可以通过 `git push`（将你的本地提交发送到远程服务器）、`git fetch`（下载远程分支和对象）或 `git pull`（同步远程更改到本地仓库）来完成。

## 还有更多…

本教程中展示的简化工作流是线性的。然而，在实际操作中，Git 的工作流通常是非线性的；这就是分支的概念。我们将在下一个教程中描述这个概念，*使用 Git 分支的典型工作流*。

这里有一些关于 Git 的优秀参考资料：

+   实操教程，见于 [`try.github.io`](https://try.github.io)

+   Git 指导游，见于 [`gitimmersion.com`](http://gitimmersion.com)

+   Atlassian Git 教程，见于 [www.atlassian.com/git](http://www.atlassian.com/git)

+   在线课程，见于 [www.codeschool.com/courses/try-git](http://www.codeschool.com/courses/try-git)

+   Lars Vogel 编写的 Git 教程，见于 [www.vogella.com/tutorials/Git/article.html](http://www.vogella.com/tutorials/Git/article.html)

+   GitHub Git 教程，见于 [`git-lectures.github.io`](http://git-lectures.github.io)

+   针对科学家的 Git 教程，见于 [`nyuccl.org/pages/GitTutorial/`](http://nyuccl.org/pages/GitTutorial/)

+   GitHub 帮助，见于 [`help.github.com`](https://help.github.com)

+   由 Scott Chacon 编写的 *Pro Git*，见于 [`git-scm.com`](http://git-scm.com)

## 另见

+   *Git 分支的典型工作流* 方案

# Git 分支的典型工作流

像 Git 这样的分布式版本控制系统是为复杂的、非线性的工作流设计的，这类工作流通常出现在交互式计算和探索性研究中。一个核心概念是**分支**，我们将在本方案中讨论这一点。

## 准备工作

你需要在本地 Git 仓库中工作才能进行此方案（见前一方案，*学习分布式版本控制系统 Git 的基础知识*）。

## 如何执行…

1.  我们创建一个名为 `newidea` 的新分支：

    ```py
    $ git branch newidea

    ```

1.  我们切换到这个分支：

    ```py
    $ git checkout newidea

    ```

1.  我们对代码进行更改，例如，创建一个新文件：

    ```py
    $ touch newfile.py

    ```

1.  我们添加此文件并提交我们的更改：

    ```py
    $ git add newfile.py
    $ git commit -m "Testing new idea."

    ```

1.  如果我们对更改感到满意，我们将该分支合并到 *master* 分支（默认分支）：

    ```py
    $ git checkout master
    $ git merge newidea

    ```

    否则，我们删除该分支：

    ```py
    $ git checkout master
    $ git branch -d newidea

    ```

其他感兴趣的命令包括：

+   `git status`：查看仓库的当前**状态**

+   `git log`：显示提交日志

+   `git branch`：显示现有的**分支**并突出当前分支

+   `git diff`：显示提交或分支之间的**差异**

### 暂存

可能发生的情况是，当我们正在进行某项工作时，需要在另一个提交或另一个分支中进行其他更改。我们可以提交尚未完成的工作，但这并不理想。更好的方法是将我们正在工作的副本**暂存**到安全位置，以便稍后恢复所有未提交的更改。它是如何工作的：

1.  我们使用以下命令保存我们的未提交更改：

    ```py
    $ git stash

    ```

1.  我们可以对仓库进行任何操作：检出一个分支、提交更改、从远程仓库拉取或推送等。

1.  当我们想要恢复未提交的更改时，输入以下命令：

    ```py
    $ git stash pop

    ```

我们可以在仓库中有多个暂存的状态。有关暂存的更多信息，请使用 `git stash --help`。

## 它是如何工作的…

假设为了测试一个新想法，你需要对多个文件中的代码进行非琐碎的修改。你创建了一个新的分支，测试你的想法，并最终得到了修改过的代码版本。如果这个想法是死路一条，你可以切换回原始的代码分支。然而，如果你对这些修改感到满意，你可以**合并**它到主分支。

这种工作流的优势在于，主分支可以独立于包含新想法的分支进行发展。当多个协作者在同一个仓库中工作时，这特别有用。然而，这也是一种很好的习惯，尤其是当只有一个贡献者时。

合并并不总是一个简单的操作，因为它可能涉及到两个分歧的分支，且可能存在冲突。Git 会尝试自动解决冲突，但并不总是成功。在这种情况下，你需要手动解决冲突。

合并的替代方法是**重基（rebasing）**，当你在自己的分支上工作时，如果主分支发生了变化，重基将非常有用。将你的分支重基到主分支上，可以让你将分支点移到一个更近期的点。这一过程可能需要你解决冲突。

Git 分支是轻量级对象。创建和操作它们的成本很低。它们是为了频繁使用而设计的。掌握所有相关概念和`git`命令（尤其是 `checkout`、`merge` 和 `rebase`）非常重要。前面的食谱中包含了许多很好的参考资料。

## 还有更多……

许多人曾思考过有效的工作流。例如，一个常见但复杂的工作流，叫做 git-flow，可以在 [`nvie.com/posts/a-successful-git-branching-model/`](http://nvie.com/posts/a-successful-git-branching-model/) 中找到描述。然而，在小型和中型项目中，使用一个更简单的工作流可能更为适宜，比如 [`scottchacon.com/2011/08/31/github-flow.html`](http://scottchacon.com/2011/08/31/github-flow.html) 中描述的工作流。后者详细阐述了这个食谱中展示的简化示例。

与分支相关的概念是**分叉（forking）**。同一个仓库可以在不同的服务器上有多个副本。假设你想为存储在 GitHub 上的 IPython 代码做贡献。你可能没有权限修改他们的仓库，但你可以将其复制到你的个人账户中——这就是所谓的分叉。在这个副本中，你可以创建一个分支，并提出一个新功能或修复一个 bug。然后，你可以提出一个**拉取请求（pull request）**，请求 IPython 的开发者将你的分支合并到他们的主分支。他们可以审核你的修改，提出建议，并最终决定是否合并你的工作（或不合并）。GitHub 就是围绕这个想法构建的，因此提供了一种清晰、现代的方式来协作开发开源项目。

在合并 pull 请求之前进行代码审查，有助于提高协作项目中的代码质量。当至少两个人审查任何一段代码时，合并错误代码或不正确代码的可能性就会降低。

当然，关于 Git 还有很多要说的。版本控制系统通常是复杂且功能强大的，Git 也不例外。掌握 Git 需要时间和实验。之前的食谱中包含了许多优秀的参考资料。

这里有一些关于分支和工作流的进一步参考资料：

+   可用的 Git 工作流，见于 [www.atlassian.com/git/workflows](http://www.atlassian.com/git/workflows)

+   在 [`pcottle.github.io/learnGitBranching/`](http://pcottle.github.io/learnGitBranching/) 学习 Git 分支

+   NumPy 项目（以及其他项目）推荐的 Git 工作流，描述于 [`docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html`](http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html)

+   Fernando Perez 在 IPython 邮件列表上关于高效 Git 工作流的帖子，见于 [`mail.scipy.org/pipermail/ipython-dev/2010-October/006746.html`](http://mail.scipy.org/pipermail/ipython-dev/2010-October/006746.html)

## 另见

+   *学习分布式版本控制系统 Git 的基础* 食谱

# 进行可重现的互动计算实验的十个技巧

在这篇食谱中，我们提出了十个技巧，帮助你进行高效且**可重现**的互动计算实验。这些更多是指导性建议，而非绝对规则。

首先，我们将展示如何通过减少重复性任务的时间、增加思考核心工作的时间来提高生产力。

其次，我们将展示如何在计算工作中实现更高的可重现性。值得注意的是，学术研究要求实验可重现，以便任何结果或结论可以被其他研究者独立验证。方法中的错误或操控往往会导致错误的结论，从而产生有害的后果。例如，在 2010 年 Carmen Reinhart 和 Kenneth Rogoff 发表的经济学研究论文《债务时期的增长》中，计算错误部分导致了一项存在全球影响力的 flawed 研究，对政策制定者产生了影响（请见 [`en.wikipedia.org/wiki/Growth_in_a_Time_of_Debt`](http://en.wikipedia.org/wiki/Growth_in_a_Time_of_Debt)）。

## 如何做…

1.  仔细而一致地组织你的目录结构。具体的结构并不重要，重要的是在整个项目中保持文件命名规范、文件夹、子文件夹等的一致性。以下是一个简单的例子：

    +   `my_project/`

        +   `data/`

        +   `code/`

            +   `common.py`

            +   `idea1.ipynb`

            +   `idea2.ipynb`

        +   `figures/`

        +   `notes/`

        +   `README.md`

1.  使用轻量级标记语言（如 **Markdown** ([`daringfireball.net/projects/markdown/`](http://daringfireball.net/projects/markdown/)) 或 **reStructuredText** (**reST**)）在文本文件中写下笔记。所有与项目、文件、数据集、代码、图形、实验室笔记本等相关的元信息应写入文本文件。

1.  与此相关，在代码中记录所有非平凡的内容，包括注释、文档字符串等。你可以使用文档生成工具，如 **Sphinx** ([`sphinx-doc.org`](http://sphinx-doc.org))。然而，在你工作时，不要花费太多时间记录不稳定和前沿的代码；它可能频繁变化，且你的文档很快就会过时。编写代码时要确保它易于理解，无需过多注释（为变量和函数命名合理，使用 Pythonic 编程模式等）。另请参见下一个章节，*编写高质量的 Python 代码*。

1.  对于所有基于文本的文件，使用分布式版本控制系统，如 Git，但不要用于二进制文件（除非是非常小的文件且确实需要）。每个项目应使用一个版本库。将版本库同步到远程服务器上，使用免费或付费的托管服务提供商（如 GitHub 或 Bitbucket）或你自己的服务器（你的主办机构可能能够为你设置一个）。使用特定的系统来存储和共享二进制数据文件，例如 [figshare.com](http://figshare.com) 或 [datadryad.org](http://datadryad.org)。

1.  首先在 IPython 笔记本中编写所有交互式计算代码，只有在代码成熟和稳定时，才将其重构为独立的 Python 组件。

1.  为了完全可重现性，确保记录下整个软件堆栈中所有组件的确切版本（操作系统、Python 发行版、模块等）。一种选择是使用虚拟环境，如 **virtualenv** 或 **conda**。

1.  使用 Python 的原生 **pickle** 模块、**dill** ([`pypi.python.org/pypi/dill`](https://pypi.python.org/pypi/dill)) 或 **Joblib** ([`pythonhosted.org/joblib/`](http://pythonhosted.org/joblib/)) 缓存长时间计算的中间结果。Joblib 特别实现了一个 NumPy 友好的 **memoize** 模式（不要与 memorize 混淆），该模式允许你缓存计算密集型函数的结果。还可以查看 **ipycache** IPython 扩展 ([`github.com/rossant/ipycache`](https://github.com/rossant/ipycache))；它在笔记本中实现了一个 `%%cache` 单元魔法。

    ### 注意

    **在 Python 中保存持久化数据**

    对于纯内部使用，你可以使用 Joblib、NumPy 的`save`和`savez`函数来保存数组，使用 pickle 来保存任何其他 Python 对象（尽量选择原生类型，如列表和字典，而非自定义类）。对于共享用途，建议使用文本文件来保存小型数据集（少于 10k 个数据点），例如，使用 CSV 格式存储数组，使用 JSON 或 YAML 格式存储高度结构化的数据。对于较大的数据集，你可以使用 HDF5（请参见第四章中的*使用 HDF5 和 PyTables 操作大型数组*和*使用 HDF5 和 PyTables 操作大型异构表格*的配方）。

1.  在开发并测试大数据集上的算法时，先在数据的小部分上运行并进行比较，再转向整个数据集。

1.  在批量运行任务时，使用并行计算来充分利用你的多核处理单元，例如，使用`IPython.parallel`、Joblib、Python 的多处理包，或任何其他并行计算库。

1.  尽可能使用 Python 函数或脚本来自动化你的工作。对于用户公开的脚本，使用命令行参数，但在可能的情况下，更倾向于使用 Python 函数而非脚本。在 Unix 系统中，学习终端命令以提高工作效率。对于 Windows 或基于 GUI 的系统上的重复性任务，使用自动化工具，如 AutoIt（[www.autoitscript.com/site/autoit/](http://www.autoitscript.com/site/autoit/)）或 AutoHotKey（[www.autohotkey.com](http://www.autohotkey.com)）。学习你经常使用程序的快捷键，或者创建你自己的快捷键。

### 提示

例如，你可以创建一个键盘快捷键来启动当前目录下的 IPython 笔记本服务器。以下链接包含一个 AutoHotKey 脚本，可以在 Windows 资源管理器中实现这一操作：

[`cyrille.rossant.net/start-an-ipython-notebook-server-in-windows-explorer/`](http://cyrille.rossant.net/start-an-ipython-notebook-server-in-windows-explorer/)

## 它是如何工作的…

本文档中的建议最终旨在优化你的工作流程，涵盖人类时间、计算机时间和质量。使用一致的约定和结构来编写代码，可以让你更轻松地组织工作。记录所有内容可以节省每个人的时间，包括（最终）你自己！如果明天你被公交车撞了（我真心希望你不会），你应该确保你的替代者能够迅速接手，因为你的文档写得非常认真细致。（你可以在[`en.wikipedia.org/wiki/Bus_factor`](http://en.wikipedia.org/wiki/Bus_factor)找到更多关于“公交车因子”的信息。）

使用分布式版本控制系统和在线托管服务可以让你在多个地点协同工作同一个代码库，且无需担心备份问题。由于你可以回溯代码历史，因此几乎不可能无意间破坏代码。

IPython 笔记本是一个用于可重复交互计算的出色工具。它让你可以详细记录工作过程。此外，IPython 笔记本的易用性意味着你无需担心可重复性；只需在笔记本中进行所有交互式工作，将其放入版本控制中，并定期提交。不要忘记将你的代码重构为独立的可重用组件。

确保优化你在电脑前花费的时间。当处理一个算法时，经常会发生这样的循环：你做了一点修改，运行代码，获取结果，再做另一个修改，依此类推。如果你需要尝试很多修改，你应该确保执行时间足够快（不超过几秒钟）。在实验阶段，使用高级优化技术未必是最佳选择。你应该缓存结果，在数据子集上尝试算法，并以较短的时间运行模拟。当你想测试不同的参数值时，也可以并行启动批处理任务。

最后，极力避免重复任务。对于日常工作中频繁发生的任务，花时间将其自动化是值得的。虽然涉及 GUI 的任务更难自动化，但借助 AutoIt 或 AutoHotKey 等免费工具，还是可以实现自动化的。

## 还有更多…

以下是一些关于计算可重复性的参考资料：

+   *高效的可重复科学工作流程*，Trevor Bekolay 的演讲， 可在[`bekolay.org/scipy2013-workflow/`](http://bekolay.org/scipy2013-workflow/)找到。

+   *可重复计算研究的十条简单规则*，*Sandve 等*，*PLoS 计算生物学*，*2013 年*，可在[`dx.doi.org/10.1371/journal.pcbi.1003285`](http://dx.doi.org/10.1371/journal.pcbi.1003285)找到。

+   Konrad Hinsen 的博客，[`khinsen.wordpress.com`](http://khinsen.wordpress.com)。

+   Software Carpentry 是一个为科学家举办研讨会的志愿者组织；这些研讨会涵盖了科学编程、交互式计算、版本控制、测试、可重复性和任务自动化等内容。你可以在[`software-carpentry.org`](http://software-carpentry.org)找到更多信息。

## 另见

+   *高效的交互式计算工作流程与 IPython* 配方

+   *编写高质量的 Python 代码* 配方

# 编写高质量的 Python 代码

编写代码很容易，编写高质量的代码则要困难得多。质量不仅体现在实际代码（变量名、注释、文档字符串等）上，还包括架构（函数、模块、类等）。通常，设计一个良好的代码架构比实现代码本身更具挑战性。

在本配方中，我们将提供一些如何编写高质量代码的建议。这是学术界一个特别重要的话题，因为越来越多没有软件开发经验的科学家需要编程。

本教程末尾给出的参考资料包含了比我们在此提到的更多细节。

## 如何实现...

1.  花时间认真学习 Python 语言。查看标准库中所有模块的列表——你可能会发现你已经实现的某些函数已经存在。学会编写 *Pythonic* 代码，不要将其他语言（如 Java 或 C++）的编程习惯直接翻译到 Python 中。

1.  学习常见的 **设计模式**；这些是针对软件工程中常见问题的通用可复用解决方案。

1.  在代码中使用断言（`assert` 关键字）来防止未来的 bug (**防御性编程**)。

1.  采用自下而上的方法开始编写代码；编写实现专注任务的独立 Python 函数。

1.  不要犹豫定期重构你的代码。如果你的代码变得过于复杂，思考如何简化它。

1.  尽量避免使用类。如果可以使用函数代替类，请选择函数。类只有在需要在函数调用之间存储持久状态时才有用。尽量让你的函数保持 *纯净*（没有副作用）。

1.  一般来说，优先使用 Python 原生类型（列表、元组、字典和 Python collections 模块中的类型）而不是自定义类型（类）。原生类型能带来更高效、更可读和更具可移植性的代码。

1.  在函数中选择关键字参数而不是位置参数。参数名称比参数顺序更容易记住。它们使你的函数自文档化。

1.  小心命名你的变量。函数和方法的名称应以动词开头。变量名应描述它是什么。函数名应描述它做什么。命名的正确性至关重要，不能被低估。

1.  每个函数都应该有一个描述其目的、参数和返回值的文档字符串，如下例所示。你还可以查看像 NumPy 这样的流行库中所采用的约定。重要的是在你的代码中保持一致性。你可以使用 Markdown 或 reST 等标记语言：

    ```py
    def power(x, n):
        """Compute the power of a number.

        Arguments:
          * x: a number.
          * n: the exponent.

        Returns:
           * c: the number x to the power of n.

        """
        return x ** n
    ```

1.  遵循（至少部分遵循）Guido van Rossum 的 Python 风格指南，也叫 **Python 增强提案第 8 号**（**PEP8**），可在 [www.python.org/dev/peps/pep-0008/](http://www.python.org/dev/peps/pep-0008/) 查阅。这是一本长篇指南，但它能帮助你编写可读性强的 Python 代码。它涵盖了许多小细节，如操作符之间的空格、命名约定、注释和文档字符串。例如，你会了解到将代码行限制在 79 个字符以内（如果有助于提高可读性，可以例外为 99 个字符）是一个良好的实践。这样，你的代码可以在大多数情况下（如命令行界面或移动设备上）正确显示，或者与其他文件并排显示。或者，你可以选择忽略某些规则。总的来说，在涉及多个开发人员的项目中，遵循常见的指南是有益的。

1.  你可以通过**pep8** Python 包自动检查代码是否符合 PEP8 中的大多数编码风格规范。使用`pip install pep8`进行安装，并通过`pep8 myscript.py`执行。

1.  使用静态代码分析工具，如 Pylint（[www.pylint.org](http://www.pylint.org)）。它可以让你静态地找到潜在的错误或低质量的代码，即在不运行代码的情况下进行检查。

1.  使用空行来避免代码杂乱（参见 PEP8）。你也可以通过显著的注释来标记一个较长 Python 模块的各个部分，例如：

    ```py
    # Imports
    # -------
    import numpy

    # Utility functions
    # -----------------
    def fun():
        pass
    ```

1.  一个 Python 模块不应包含超过几百行代码。一个模块的代码行数过多可能意味着你需要将其拆分成多个模块。

1.  将重要的项目（包含数十个模块）组织为子包，例如：

    +   `core/`

    +   `io/`

    +   `utils/`

    +   `__init__.py`

1.  看看主要的 Python 项目是如何组织的。例如，IPython 的代码就很有条理，按照具有明确职责的子包层次结构进行组织。阅读这些代码本身也非常有启发性。

1.  学习创建和分发新 Python 包的最佳实践。确保你了解 setuptools、pip、wheels、virtualenv、PyPI 等工具。此外，我们强烈建议你认真研究 conda（[`conda.pydata.org`](http://conda.pydata.org)），这是由 Continuum Analytics 开发的一个强大且通用的打包系统。打包是 Python 中一个混乱且快速发展的领域，因此请只阅读最新的参考资料。在*更多内容…*部分中有一些参考资料。

## 它是如何工作的…

编写可读的代码意味着其他人（或者几年后你自己）会更快地理解它，也更愿意使用它。这也有助于 bug 追踪。

模块化代码也更容易理解和重用。将程序的功能实现为独立的函数，并按照包和模块的层次结构进行组织，是实现高质量代码的绝佳方式。

使用函数而不是类可以更容易地保持代码的松耦合。意大利面式代码（Spaghetti code）真的很难理解、调试和重用。

在处理一个新项目时，可以在自下而上的方法和自上而下的方法之间交替进行。自下而上的方法让你在开始思考程序的整体架构之前先对代码有一定的经验。然而，仍然要确保通过思考组件如何协同工作来知道自己最终的目标。

## 还有更多内容…

已经有很多关于如何编写优美代码的文章——请参阅以下参考资料。你可以找到许多关于这个主题的书籍。在接下来的教程中，我们将介绍确保代码不仅看起来漂亮，而且能够按预期工作的标准技术：单元测试、代码覆盖率和持续集成。

这里有一些参考资料：

+   *《Python 厨房秘籍》*，由 David Beazley 和 Brian K. Jones 编写，包含许多 Python 3 高级配方，可在 [`shop.oreilly.com/product/0636920027072.do`](http://shop.oreilly.com/product/0636920027072.do) 获取

+   *《Python 旅行者指南！》*，可在 [`docs.python-guide.org/en/latest/`](http://docs.python-guide.org/en/latest/) 获取

+   维基百科上的设计模式，详见 [`en.wikipedia.org/wiki/Software_design_pattern`](http://en.wikipedia.org/wiki/Software_design_pattern)

+   Python 设计模式，描述于 [`github.com/faif/python-patterns`](https://github.com/faif/python-patterns)

+   Tahoe-LAFS 编码标准，详见 [`tahoe-lafs.org/trac/tahoe-lafs/wiki/CodingStandards`](https://tahoe-lafs.org/trac/tahoe-lafs/wiki/CodingStandards)

+   *《如何成为一名伟大的软件开发者》*，由 Peter Nixey 编写，可在 [`peternixey.com/post/83510597580/how-to-be-a-great-software-developer`](http://peternixey.com/post/83510597580/how-to-be-a-great-software-developer) 阅读

+   *《为什么你应该用尽可能少的功能编写有缺陷的软件》*，由 Brian Granger 主讲，可在 [www.youtube.com/watch?v=OrpPDkZef5I](http://www.youtube.com/watch?v=OrpPDkZef5I) 观看

+   *《程序包指南》*，可在 [`guide.python-distribute.org`](http://guide.python-distribute.org) 获取

+   *Python 包装用户指南*，可在 [`python-packaging-user-guide.readthedocs.org`](http://python-packaging-user-guide.readthedocs.org) 获取

## 另见

+   *《进行可重复交互式计算实验的十个技巧》* 配方

+   *使用 nose 编写单元测试* 配方

# 使用 nose 编写单元测试

手动测试对确保我们的软件按预期工作并且不包含关键性错误至关重要。然而，手动测试存在严重限制，因为每次更改代码时，可能会引入新的缺陷。我们不可能在每次提交时都手动测试整个程序。

现如今，自动化测试已经成为软件工程中的标准实践。在这个配方中，我们将简要介绍自动化测试的重要方面：单元测试、测试驱动开发、测试覆盖率和持续集成。遵循这些实践对于开发高质量的软件是绝对必要的。

## 准备工作

Python 有一个原生的单元测试模块（`unittest`），你可以直接使用。还有其他第三方单元测试包，如 py.test 或 nose，我们在这里选择了 nose。nose 使得编写测试套件变得稍微容易一些，并且拥有一个外部插件库。除非用户自己想运行测试套件，否则他们不需要额外的依赖。你可以通过 `pip install nose` 安装 nose。

## 如何实现...

在这个示例中，我们将为一个从 URL 下载文件的函数编写单元测试。即使在没有网络连接的情况下，测试套件也应能够运行并成功通过。我们通过使用一个模拟的 HTTP 服务器来欺骗 Python 的 `urllib` 模块，从而解决这一问题。

### 注意

本食谱中使用的代码片段是为 Python 3 编写的。要使它们在 Python 2 中运行，需要做一些更改，我们在代码中已标明了这些更改。Python 2 和 Python 3 的版本都可以在本书的网站上找到。

你可能对`requests`模块也感兴趣；它为 HTTP 请求提供了一个更简单的 API（[`docs.python-requests.org/en/latest/`](http://docs.python-requests.org/en/latest/)）。

1.  我们创建了一个名为`datautils.py`的文件，里面包含以下代码：

    ```py
    In [1]: %%writefile datautils.py
    # Version 1.
    import os
    from urllib.request import urlopen  # Python 2: use urllib2

    def download(url):
        """Download a file and save it in the current folder.
        Return the name of the downloaded file."""
        # Get the filename.
        file = os.path.basename(url)
        # Download the file unless it already exists.
        if not os.path.exists(file):
            with open(file, 'w') as f:
                f.write(urlopen(url).read())
        return file
    Writing datautils.py
    ```

1.  我们创建了一个名为`test_datautils.py`的文件，里面包含以下代码：

    ```py
    In [2]: %%writefile test_datautils.py
    # Python 2: use urllib2
    from urllib.request import (HTTPHandler, install_opener, 
                                build_opener, addinfourl)
    import os
    import shutil
    import tempfile
    from io import StringIO  # Python 2: use StringIO
    from datautils import download

    TEST_FOLDER = tempfile.mkdtemp()
    ORIGINAL_FOLDER = os.getcwd()

    class TestHTTPHandler(HTTPHandler):
        """Mock HTTP handler."""
        def http_open(self, req):
            resp = addinfourl(StringIO('test'), '',
                              req.get_full_url(), 200)
            resp.msg = 'OK'
            return resp

    def setup():
        """Install the mock HTTP handler for unit tests."""
        install_opener(build_opener(TestHTTPHandler))
        os.chdir(TEST_FOLDER)

    def teardown():
        """Restore the normal HTTP handler."""
        install_opener(build_opener(HTTPHandler))
        # Go back to the original folder.
        os.chdir(ORIGINAL_FOLDER)
        # Delete the test folder.
        shutil.rmtree(TEST_FOLDER)

    def test_download1():
        file = download("http://example.com/file.txt")
        # Check that the file has been downloaded.
        assert os.path.exists(file)
        # Check that the file contains the contents of
        # the remote file.
        with open(file, 'r') as f:
            contents = f.read()
        print(contents)
        assert contents == 'test'
    Writing test_datautils.py
    ```

1.  现在，为了启动测试，我们在终端中执行以下命令：

    ```py
    $ nosetests
    .
    Ran 1 test in 0.042s
    OK

    ```

1.  我们的第一个单元测试通过了！现在，让我们添加一个新的测试。我们在`test_datautils.py`文件的末尾添加一些代码：

    ```py
    In [4]: %%writefile test_datautils.py -a

            def test_download2():
                file = download("http://example.com/")
                assert os.path.exists(file)
    Appending to test_datautils.py
    ```

1.  我们使用`nosetests`命令再次启动测试：

    ```py
    $ nosetests
    .E
    ERROR: test_datautils.test_download2
    Traceback (most recent call last):
     File "datautils.py", line 12, in download
     with open(file, 'wb') as f:
    IOError: [Errno 22] invalid mode ('wb') or filename: ''
    Ran 2 tests in 0.032s
    FAILED (errors=1)

    ```

1.  第二个测试失败了。在实际应用中，我们可能需要调试程序。这应该不难，因为错误被隔离在一个单独的测试函数中。在这里，通过检查回溯错误和代码，我们发现错误是由于请求的 URL 没有以正确的文件名结尾。因此，推断的文件名`os.path.basename(url)`为空。我们通过以下方法来修复这个问题：将`datautils.py`中的`download`函数替换为以下函数：

    ```py
    In [6]: %%file datautils.py
    # Version 2.
    import os
    from urllib.request import urlopen  # Python 2: use urllib2

    def download(url):
        """Download a file and save it in the current folder.
        Return the name of the downloaded file."""
        # Get the filename.
        file = os.path.basename(url)
        # Fix the bug, by specifying a fixed filename if the
        # URL does not contain one.
        if not file:
            file = 'downloaded'
        # Download the file unless it already exists.
        if not os.path.exists(file):
            with open(file, 'w') as f:
                f.write(urlopen(url).read())
        return file
    Overwriting datautils.py
    ```

1.  最后，让我们再次运行测试：

    ```py
    $ nosetests
    ..
    Ran 2 tests in 0.036s
    OK

    ```

### 提示

默认情况下，`nosetests`会隐藏标准输出（除非发生错误）。如果你希望标准输出显示出来，可以使用`nosetests --nocapture`。

## 它是如何工作的...

每个名为`xxx.py`的 Python 模块应该有一个对应的`test_xxx.py`模块。这个测试模块包含执行并测试`xxx.py`模块中功能的函数（单元测试）。

根据定义，一个给定的单元测试必须专注于一个非常具体的功能。所有单元测试应该是完全独立的。将程序编写为一组经过充分测试、通常是解耦的单元，迫使你编写更易于维护的模块化代码。

然而，有时你的模块函数在运行之前需要一些预处理工作（例如，设置环境、创建数据文件或设置 Web 服务器）。单元测试框架可以处理这些事情；只需编写`setup()`和`teardown()`函数（称为**fixtures**），它们将分别在测试模块开始和结束时被调用。请注意，测试模块运行前后的系统环境状态应该完全相同（例如，临时创建的文件应在`teardown`中删除）。

这里，`datautils.py` 模块包含一个名为 `download` 的函数，该函数接受一个 URL 作为参数，下载文件并将其保存到本地。这个模块还带有一个名为 `test_datautils.py` 的测试模块。你应该在你的程序中使用相同的约定（`test_<modulename>` 作为 `modulename` 模块的测试模块）。这个测试模块包含一个或多个以 `test_` 为前缀的函数。nose 就是通过这种方式自动发现项目中的单元测试。nose 也接受其他类似的约定。

### 提示

nose 会运行它在你的项目中找到的所有测试，但你当然可以更精确地控制要运行的测试。键入 `nosetests --help` 可以获取所有选项的列表。你也可以查阅 [`nose.readthedocs.org/en/latest/testing.html`](http://nose.readthedocs.org/en/latest/testing.html) 上的文档。

测试模块还包含 `setup` 和 `teardown` 函数，这些函数会被 nose 自动识别为测试夹具。在 `setup` 函数中创建一个自定义的 HTTP 处理程序对象。该对象会捕获所有 HTTP 请求，即使是那些具有虚构 URL 的请求。接着，`setup` 函数会进入一个测试文件夹（该文件夹是通过 `tempfile` 模块创建的），以避免下载的文件和现有文件之间可能的冲突。一般来说，单元测试不应该留下任何痕迹；这也是我们确保测试完全可重复的方式。同样，`teardown` 函数会删除测试文件夹。

### 提示

在 Python 3.2 及更高版本中，你还可以使用 `tempfile.TemporaryDirectory` 来创建一个临时目录。

第一个单元测试从一个虚拟 URL 下载文件，并检查它是否包含预期的内容。默认情况下，如果单元测试没有抛出异常，则视为通过。这时，`assert` 语句就非常有用，如果语句为 `False`，则会抛出异常。nose 还提供了方便的例程和装饰器，用于精确地确定某个单元测试期望通过或失败的条件（例如，它应该抛出某个特定的异常才算通过，或者它应该在 *X* 秒内完成等）。

### 提示

NumPy 提供了更多方便的类似 assert 的函数（请参见 [`docs.scipy.org/doc/numpy/reference/routines.testing.html`](http://docs.scipy.org/doc/numpy/reference/routines.testing.html)）。这些函数在处理数组时特别有用。例如，`np.testing.assert_allclose(x, y)` 会断言 `x` 和 `y` 数组几乎相等，精度可以指定。

编写完整的测试套件需要时间。它对你代码的架构提出了严格（但良好的）约束。这是一次真正的投资，但从长远来看总是值得的。此外，知道你的项目有一个完整的测试套件支持，真是让人放心。

首先，从一开始就考虑单元测试迫使你思考模块化架构。对于一个充满相互依赖的单体程序，编写单元测试是非常困难的。

其次，单元测试使得发现和修复 bug 变得更加容易。如果在程序中引入了更改后，某个单元测试失败，隔离并重现 bug 就变得非常简单。

第三，单元测试帮助你避免**回归**，即已修复的 bug 在后续版本中悄然重现。当你发现一个新 bug 时，应该为它编写一个特定的失败单元测试。修复它时，让这个测试通过。现在，如果这个 bug 后来再次出现，这个单元测试将会失败，你就可以立即解决它。

假设你编写了一个多层次的复杂程序，每一层的基础上都有一个基于*第 n*层的*n+1*层。有了一套成功的单元测试作为*第 n*层的保障，你就能确信它按预期工作。当你在处理*n+1*层时，你可以专注于这一层，而不必总是担心下面一层是否有效。

单元测试并不是全部，它只关注独立的组件。为了确保程序中各组件的良好集成，还需要进一步的测试层级。

## 还有更多...

单元测试是一个广泛的话题，我们在这个配方中仅仅触及了表面。这里提供了一些进一步的信息。

### 测试覆盖率

使用单元测试是好事，但测量**测试覆盖率**更好：它量化了我们的代码有多少被你的测试套件覆盖。Ned Batchelder 的**coverage**模块（[`nedbatchelder.com/code/coverage/`](http://nedbatchelder.com/code/coverage/)）正是做这件事。它与 nose 非常好地集成。

首先，使用`pip install coverage`安装 coverage。然后使用以下命令运行你的测试套件：

```py
$ nosetests --with-cov --cover-package datautils

```

该命令指示 nose 仅为`datautils`包启动测试套件并进行覆盖率测量。

[coveralls.io](http://coveralls.io)服务将测试覆盖率功能引入持续集成服务器（参见*单元测试与持续集成*部分）。它与 GitHub 无缝集成。

### 带有单元测试的工作流

注意我们在这个例子中使用的特定工作流。在编写`download`函数后，我们创建了第一个通过的单元测试。然后我们创建了第二个失败的测试。我们调查了问题并修复了函数，第二个测试通过了。我们可以继续编写越来越复杂的单元测试，直到我们确信该函数在大多数情况下按预期工作。

### 提示

运行`nosetests --pdb`以在失败时进入 Python 调试器。这对于快速找出单元测试失败的原因非常方便。

这就是**测试驱动开发**，它要求在编写实际代码之前编写单元测试。这个工作流迫使我们思考代码的功能和使用方式，而不是它是如何实现的。

### 单元测试与持续集成

养成每次提交时运行完整测试套件的好习惯。实际上，我们甚至可以通过 **持续集成** 完全透明且自动地做到这一点。我们可以设置一个服务器，在每次提交时自动在云端运行我们的测试套件。如果某个测试失败，我们会收到一封自动邮件，告诉我们问题所在，以便我们修复。

有很多持续集成系统和服务：Jenkins/Hudson、[`drone.io`](https://drone.io)、[`stridercd.com`](http://stridercd.com)、[`travis-ci.org`](https://travis-ci.org) 等等。其中一些与 GitHub 项目兼容。例如，要在 GitHub 项目中使用 Travis CI，可以在 Travis CI 上创建账户，将 GitHub 项目与此账户关联，然后在仓库中添加一个 `.travis.yml` 文件，其中包含各种设置（有关更多详情，请参见以下参考资料）。

总结来说，单元测试、代码覆盖率和持续集成是所有重大项目应遵循的标准实践。

这里有一些参考资料：

+   测试驱动开发，详见 [`en.wikipedia.org/wiki/Test-driven_development`](http://en.wikipedia.org/wiki/Test-driven_development)

+   *未经测试的代码是坏代码：企业软件交付中的测试自动化*，由 Martin Aspeli 编写，详见 [www.deloittedigital.com/eu/blog/untested-code-is-broken-code-test-automation-in-enterprise-software-deliver](http://www.deloittedigital.com/eu/blog/untested-code-is-broken-code-test-automation-in-enterprise-software-deliver)

+   Travis CI 在 Python 中的文档，见 [`about.travis-ci.org/docs/user/languages/python/`](http://about.travis-ci.org/docs/user/languages/python/)

# 使用 IPython 调试代码

调试是软件开发和交互式计算的一个不可或缺的部分。一种常见的调试技术是在代码中的各个地方放置 `print` 语句。谁没做过这个呢？它可能是最简单的解决方案，但肯定不是最有效的（它是穷人版的调试器）。

IPython 完美适配调试，集成的调试器非常易于使用（实际上，IPython 只是提供了一个友好的界面来访问原生的 Python 调试器 `pdb`）。特别是，IPython 调试器中支持 Tab 补全。本节内容描述了如何使用 IPython 调试代码。

### 注意

早期版本的 IPython 笔记本不支持调试器，也就是说，调试器可以在 IPython 终端和 Qt 控制台中使用，但在笔记本中无法使用。这个问题在 IPython 1.0 中得到了解决。

## 如何做到这一点...

在 Python 中有两种非互斥的调试方式。在事后调试模式下，一旦抛出异常，调试器会立即进入代码，这样我们就可以调查导致异常的原因。在逐步调试模式下，我们可以在断点处停止解释器，并逐步恢复执行。这个过程使我们能够在代码执行时仔细检查变量的状态。  

两种方法其实可以同时使用；我们可以在事后调试模式下进行逐步调试。  

### 事后调试模式  

当在 IPython 中抛出异常时，执行 `%debug` 魔法命令启动调试器并逐步进入代码。并且，`%pdb on` 命令告诉 IPython 一旦抛出异常，就自动启动调试器。  

一旦进入调试器，你可以访问几个特殊命令，下面列出的是最重要的一些：  

+   `p varname` **打印**一个变量的值  

+   `w` 显示你在堆栈中的当前位置  

+   `u` 在堆栈中向**上**跳  

+   `d` 在堆栈中向**下**跳  

+   `l` 显示你当前位置周围的**代码行**  

+   `a` 显示当前函数的**参数**  

调用堆栈包含代码执行当前位置的所有活动函数的列表。你可以轻松地在堆栈中上下导航，检查函数参数的值。虽然这个模式使用起来相当简单，但它应该能帮助你解决大部分问题。对于更复杂的问题，可能需要进行逐步调试。  

### 步骤调试  

你有几种方法来启动逐步调试模式。首先，为了在代码中某处设置断点，插入以下命令：  

```py
import pdb; pdb.set_trace()

```

其次，你可以使用以下命令从 IPython 运行一个脚本：  

```py
%run -d -b extscript.py:20 script

```

这个命令在调试器控制下运行 `script.py` 文件，并在 `extscript.py` 的第 20 行设置一个断点（该文件在某个时刻由 `script.py` 导入）。最后，一旦进入调试器，你就可以开始逐步调试。  

步骤调试就是精确控制解释器的执行过程。从脚本开始或者从断点处，你可以使用以下命令恢复解释器的执行：  

+   `s` 执行当前行并尽快停下来（**逐步**调试，也就是最细粒度的执行模式）  

+   `n` 继续执行直到到达当前函数中的**下一**行  

+   `r` 继续执行直到当前函数**返回**  

+   `c` **继续**执行直到到达下一个断点

+   `j 30` 将你带到当前文件的第 30 行  

你可以通过 `b` 命令或 `tbreak`（临时断点）动态添加断点。你还可以清除所有或部分断点，启用或禁用它们，等等。你可以在 [`docs.python.org/3/library/pdb.html`](https://docs.python.org/3/library/pdb.html) 找到调试器的完整细节。

## 还有更多...

要使用 IPython 调试代码，你通常需要先通过 IPython 执行它，例如使用 `%run`。然而，你可能并不总是有一个简单的方法来做到这一点。例如，你的程序可能通过一个自定义的命令行 Python 脚本运行，可能是由一个复杂的 bash 脚本启动，或者集成在一个 GUI 中。在这些情况下，你可以在代码的任何位置嵌入一个 IPython 解释器（由 Python 启动），而不是用 IPython 运行整个程序（如果你只需要调试代码的一小部分，使用整个程序可能会显得过于复杂）。

要将 IPython 嵌入到你的程序中，只需在代码中的某个地方插入以下命令：

```py
from IPython import embed
embed()

```

当你的 Python 程序执行到这段代码时，它会暂停并在该特定位置启动一个交互式的 IPython 终端。你将能够检查所有局部变量，执行你想要的任何代码，并且在恢复正常执行之前，可能会调试你的代码。

### 提示

rfoo，访问链接 [`code.google.com/p/rfoo/`](https://code.google.com/p/rfoo/)，让你可以远程检查和修改正在运行的 Python 脚本的命名空间。

### GUI 调试器

大多数 Python 集成开发环境（IDE）都提供图形化调试功能（参见 *使用 IPython 的高效交互式计算工作流*）。有时候，GUI 比命令行调试器更为方便。我们还可以提到 Winpdb（[winpdb.org](http://winpdb.org)），一个图形化、平台无关的 Python 调试器。
