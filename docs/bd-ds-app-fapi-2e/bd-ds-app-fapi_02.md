

# 第一章：Python 开发环境设置

在我们开始 FastAPI 之旅之前，我们需要按照 Python 开发者日常使用的最佳实践和约定来配置 Python 环境，以运行他们的项目。在本章结束时，您将能够在一个受限的环境中运行 Python 项目，并安装第三方依赖项，这样即使您在处理另一个使用不同 Python 版本或依赖项的项目时，也不会产生冲突。

在本章中，我们将讨论以下主要内容：

+   使用 `pyenv` 安装 Python 发行版

+   创建 Python 虚拟环境

+   使用 `pip` 安装 Python 包

+   安装 HTTPie 命令行工具

# 技术要求

在本书中，我们假设您可以访问基于 Unix 的环境，例如 Linux 发行版或 macOS。

如果您还没有这样做，macOS 用户应该安装 *Homebrew* 包管理器（[`brew.sh`](https://brew.sh)），它在安装命令行工具时非常有用。

如果您是 Windows 用户，您应该启用 **Windows 子系统 Linux**（**WSL**）（[`docs.microsoft.com/windows/wsl/install-win10`](https://docs.microsoft.com/windows/wsl/install-win10)）并安装与 Windows 环境并行运行的 Linux 发行版（如 Ubuntu），这将使您能够访问所有必需的工具。目前，WSL 有两个版本：WSL 和 WSL2。根据您的 Windows 版本，您可能无法安装最新版本。然而，如果您的 Windows 安装支持，建议使用 WSL2。

# 使用 pyenv 安装 Python 发行版

Python 已经捆绑在大多数 Unix 环境中。为了确保这一点，您可以在命令行中运行以下命令，查看当前安装的 Python 版本：

```py

  $ python3 --version
```

显示的版本输出将根据您的系统有所不同。您可能认为这足以开始，但它带来了一个重要问题：*您无法为您的项目选择 Python 版本*。每个 Python 版本都引入了新功能和重大变化。因此，能够切换到较新的版本以便为新项目利用新特性，同时还能运行可能不兼容的旧项目是非常重要的。这就是为什么我们需要 `pyenv`。

**pyenv** 工具（[`github.com/pyenv/pyenv`](https://github.com/pyenv/pyenv)）帮助您管理并在系统中切换多个 Python 版本。它允许您为整个系统设置默认的 Python 版本，也可以为每个项目设置。

在开始之前，您需要在系统上安装一些构建依赖项，以便 `pyenv` 可以在您的系统上编译 Python。官方文档提供了明确的指导（[`github.com/pyenv/pyenv/wiki#suggested-build-environment`](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)），但以下是您应该运行的命令：

1.  安装构建依赖项：

    +   对于 macOS 用户，请使用以下命令：

        ```py

        $ brew install openssl readline sqlite3 xz zlib tcl-tk
        ```

    +   对于 Ubuntu 用户，使用以下命令：

        ```py
        $ sudo apt update; sudo apt install make build-essential libssl-dev zlib1g-dev \libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
        ```

包管理器

`brew` 和 `apt` 是通常所说的软件包管理器。它们的作用是自动化系统上软件的安装和管理。因此，你不必担心从哪里下载它们，以及如何安装和卸载它们。这些命令只是告诉包管理器更新其内部的软件包索引，然后安装所需的软件包列表。

1.  安装 `pyenv`：

    ```py

    $ curl https://pyenv.run | bash
    ```

macOS 用户提示

如果你是 macOS 用户，你也可以使用 Homebrew 安装它：`brew` `install pyenv`。

1.  这将下载并执行一个安装脚本，为你处理所有事情。最后，它会提示你添加一些行到 shell 脚本中，以便 `pyenv` 能被 shell 正确发现：

    +   如果你的 shell 是 `bash`（大多数 Linux 发行版和旧版 macOS 的默认 shell），运行以下命令：

        ```py

        zsh (the default in the latest version of macOS), run the following commands:

        ```

        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrcecho 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrcecho 'eval "$(pyenv init -)"' >> ~/.zshrc

        ```py

        ```

什么是 shell，我怎么知道自己在使用哪个 shell？

Shell 是你启动命令行时正在运行的底层程序。它负责解释并执行你的命令。随着时间的推移，已经开发出了多种变种程序，如 `bash` 和 `zsh`。尽管它们在某些方面有所不同，尤其是在配置文件的命名上，但它们大多是兼容的。要查找你使用的是哪个 shell，你可以运行 `echo $``SHELL` 命令。

1.  重新加载你的 shell 配置以应用这些更改：

    ```py

    pyenv tool:

    ```

    $ pyenv>>> pyenv 2.3.6>>> 用法：pyenv <command> [<args>]

    ```py

    ```

1.  我们现在可以安装我们选择的 Python 发行版。虽然 FastAPI 兼容 Python 3.7 及以上版本，但本书中我们将使用 Python 3.10，这个版本在处理异步范式和类型提示方面更为成熟。本书中的所有示例都是用这个版本测试的，但应该在更新版本中也能顺利运行。让我们安装 Python 3.10：

    ```py

    $ pyenv install 3.10
    ```

这可能需要几分钟，因为系统需要从源代码编译 Python。

那 Python 3.11 呢？

你可能会想，既然 Python 3.11 已经发布并可用，为什么我们在这里使用 Python 3.10？在写这本书的时候，我们将使用的所有库并不都正式支持最新版本。这就是我们选择使用一个更成熟版本的原因。不过别担心：你在这里学到的内容对未来的 Python 版本仍然是相关的。

1.  最后，你可以使用以下命令设置默认的 Python 版本：

    ```py

    $ pyenv global 3.10
    ```

这将告诉系统，除非在特定项目中另有指定，否则默认始终使用 Python 3.10。

1.  为确保一切正常，运行以下命令检查默认调用的 Python 版本：

    ```py

    $ python --versionPython 3.10.8
    ```

恭喜！现在你可以在系统上处理任何版本的 Python，并随时切换！

为什么显示的是 3.10.8 而不是仅仅 3.10？

3.10 版本对应 Python 的一个主要版本。Python 核心团队定期发布主要版本，带来新特性、弃用和有时会有破坏性更改。然而，当发布新主要版本时，之前的版本并没有被遗忘：它们继续接收错误和安全修复。这就是版本号第三部分的意义。

当你阅读本书时，你很可能已经安装了更新版本的 Python 3.10，例如 3.10.9，这意味着修复已发布。你可以在这个官方文档中找到更多关于 Python 生命周期以及 Python 核心团队计划支持旧版本的时间的信息：[`devguide.python.org/versions/`](https://devguide.python.org/versions/)。

# 创建 Python 虚拟环境

和今天的许多编程语言一样，Python 的强大来自于庞大的第三方库生态系统，其中当然包括 FastAPI，这些库帮助你非常快速地构建复杂且高质量的软件。`pip`。

默认情况下，当你使用 `pip` 安装第三方包时，它会为*整个系统*安装。与一些其他语言不同，例如 Node.js 的 `npm`，它默认会为当前项目创建一个本地目录来安装这些依赖项。显然，当你在多个 Python 项目中工作，且这些项目的依赖项版本冲突时，这可能会导致问题。它还使得仅检索部署项目所需的依赖项变得困难。

这就是为什么 Python 开发者通常使用**虚拟环境**的原因。基本上，虚拟环境只是项目中的一个目录，其中包含你的 Python 安装副本和项目的依赖项。这个模式非常常见，以至于用于创建虚拟环境的工具已与 Python 一起捆绑：

1.  创建一个将包含你的项目的目录：

    ```py

    $ mkdir fastapi-data-science$ cd fastapi-data-science
    ```

针对使用 WSL 的 Windows 用户的提示

如果你使用的是带有 WSL 的 Windows，我们建议你在 Windows 驱动器上创建工作文件夹，而不是在 Linux 发行版的虚拟文件系统中。这样，你可以在 Windows 中使用你最喜欢的文本编辑器或**集成开发环境**(**IDE**)编辑源代码文件，同时在 Linux 中运行它们。

为此，你可以通过 `/mnt/c` 在 Linux 命令行中访问你的 `C:` 驱动器。因此，你可以使用常规的 Windows 路径访问个人文档，例如 `cd /mnt/c/Users/YourUsername/Documents`。

1.  你现在可以创建一个虚拟环境：

    ```py

    $ python -m venv venv
    ```

基本上，这个命令告诉 Python 运行标准库中的 `venv` 包，在 `venv` 目录中创建一个虚拟环境。这个目录的名称是一个约定，但你可以根据需要选择其他名称。

1.  完成此操作后，你需要激活这个虚拟环境。它会告诉你的 shell 会话使用本地目录中的 Python 解释器和依赖项，而不是全局的。运行以下命令：

    ```py

    $ source venv/bin/activatee
    ```

完成此操作后，你可能会注意到提示符添加了虚拟环境的名称：

```py

(venv) $
```

请记住，这个虚拟环境的激活仅对*当前会话*有效。如果你关闭它或打开其他命令提示符，你将需要重新激活它。这很容易被忘记，但经过一些 Python 实践后，它会变得自然而然。

现在，你可以在项目中安全地安装 Python 包了！

# 使用 pip 安装 Python 包

正如我们之前所说，`pip`是内置的 Python 包管理器，它将帮助我们安装第三方库。

关于替代包管理器，如 Poetry、Pipenv 和 Conda 的一些话

在探索 Python 社区时，你可能会听说过像 Poetry、Pipenv 和 Conda 这样的替代包管理器。这些管理器的出现是为了解决`pip`的一些问题，特别是在子依赖项管理方面。虽然它们是非常好的工具，但我们将在*第十章*《部署 FastAPI 项目》中看到，大多数云托管平台期望使用标准的`pip`命令来管理依赖项。因此，它们可能不是 FastAPI 应用程序的最佳选择。

开始之前，让我们先安装 FastAPI 和 Uvicorn：

```py

(venv) $ pip install fastapi "uvicorn[standard]"
```

我们将在后面的章节中讨论它，但运行 FastAPI 项目需要 Uvicorn。

“标准”在“uvicorn”后面是什么意思？

你可能注意到`uvicorn`后面方括号中的`standard`一词。有时，一些库有一些子依赖项，这些依赖项不是必需的来使库工作。通常，它们是为了可选功能或特定项目需求而需要的。方括号在这里表示我们想要安装`uvicorn`的标准子依赖项。

为了确保安装成功，我们可以打开 Python 交互式 Shell 并尝试导入`fastapi`包：

```py

(venv) $ python>>> from fastapi import FastAPI
```

如果没有错误出现，恭喜你，FastAPI 已经安装并准备好使用了！

# 安装 HTTPie 命令行工具

在进入主题之前，我们还需要安装最后一个工具。正如你可能知道的，FastAPI 主要用于构建**REST API**。因此，我们需要一个工具来向我们的 API 发送 HTTP 请求。为了做到这一点，我们有几种选择：

+   **FastAPI** **自动文档**

+   **Postman**：一个 GUI 工具，用于执行 HTTP 请求

+   **cURL**：广泛使用的命令行工具，用于执行网络请求

即使像 FastAPI 自动文档和 Postman 这样的可视化工具很好用且容易操作，它们有时缺乏一些灵活性，并且可能不如命令行工具高效。另一方面，cURL 是一个非常强大的工具，具有成千上万的选项，但对于测试简单的 REST API 来说，它可能显得复杂和冗长。

这就是我们将介绍 **HTTPie** 的原因，它是一个旨在简化 HTTP 请求的命令行工具。与 cURL 相比，它的语法更加友好，更容易记住，因此你可以随时运行复杂的请求。而且，它内置支持 JSON 和语法高亮。由于它是一个 **命令行界面** (**CLI**) 工具，我们保留了命令行的所有优势：例如，我们可以直接将 JSON 文件通过管道传输，并作为 HTTP 请求的主体发送。它可以通过大多数包管理器安装：

+   macOS 用户可以使用此命令：

    ```py

    $ brew install httpie
    ```

+   Ubuntu 用户可以使用此命令：

    ```py

    $ sudo apt-get update && sudo apt-get install httpie
    ```

让我们看看如何对一个虚拟 API 执行简单请求：

1.  首先，让我们获取数据：

    ```py

    $ http GET https://603cca51f4333a0017b68509.mockapi.io/todos>>>HTTP/1.1 200 OKAccess-Control-Allow-Headers: X-Requested-With,Content-Type,Cache-Control,access_tokenAccess-Control-Allow-Methods: GET,PUT,POST,DELETE,OPTIONSAccess-Control-Allow-Origin: *Connection: keep-aliveContent-Length: 58Content-Type: application/jsonDate: Tue, 08 Nov 2022 08:28:30 GMTEtag: "1631421347"Server: CowboyVary: Accept-EncodingVia: 1.1 vegurX-Powered-By: Express[    {        "id": "1",        "text": "Write the second edition of the book"    }]
    ```

如你所见，你可以使用 `http` 命令调用 HTTPie，简单地输入 HTTP 方法和 URL。它以清晰且格式化的方式输出 HTTP 头和 JSON 请求体。

1.  HTTPie 还支持非常快速地在请求体中发送 JSON 数据，而无需手动格式化 JSON：

    ```py

    $ http -v POST https://603cca51f4333a0017b68509.mockapi.io/todos text="My new task"POST /todos HTTP/1.1Accept: application/json, */*;q=0.5Accept-Encoding: gzip, deflateConnection: keep-aliveContent-Length: 23Content-Type: application/jsonHost: 603cca51f4333a0017b68509.mockapi.ioUser-Agent: HTTPie/3.2.1{"text": "My new task"}HTTP/1.1 201 CreatedAccess-Control-Allow-Headers: X-Requested-With,Content-Type,Cache-Control,access_tokenAccess-Control-Allow-Methods: GET,PUT,POST,DELETE,OPTIONSAccess-Control-Allow-Origin: *Connection: keep-aliveContent-Length: 31Content-Type: application/jsonDate: Tue, 08 Nov 2022 08:30:10 GMTServer: CowboyVary: Accept-EncodingVia: 1.1 vegurX-Powered-By: Express{    "id": "2",    "text": "My new task"}
    ```

只需输入属性名称及其值，用 `=` 分隔，HTTPie 就会理解这是请求体的一部分（JSON 格式）。注意，这里我们指定了 `-v` 选项，告诉 HTTPie 在响应之前 *输出请求*，这对于检查我们是否正确指定了请求非常有用。

1.  最后，让我们看看如何指定 *请求头部*：

    ```py

    $ http -v GET https://603cca51f4333a0017b68509.mockapi.io/todos "My-Header: My-Header-Value"GET /todos HTTP/1.1Accept: */*Accept-Encoding: gzip, deflateConnection: keep-aliveHost: 603cca51f4333a0017b68509.mockapi.ioMy-Header: My-Header-ValueUser-Agent: HTTPie/3.2.1HTTP/1.1 200 OKAccess-Control-Allow-Headers: X-Requested-With,Content-Type,Cache-Control,access_tokenAccess-Control-Allow-Methods: GET,PUT,POST,DELETE,OPTIONSAccess-Control-Allow-Origin: *Connection: keep-aliveContent-Length: 90Content-Type: application/jsonDate: Tue, 08 Nov 2022 08:32:12 GMTEtag: "1849016139"Server: CowboyVary: Accept-EncodingVia: 1.1 vegurX-Powered-By: Express[    {        "id": "1",        "text": "Write the second edition of the book"    },    {        "id": "2",        "text": "My new task"    }]
    ```

就是这样！只需输入你的头部名称和值，用冒号分隔，告诉 HTTPie 这是一个头部。

# 概述

现在你已经掌握了所有必要的工具和配置，可以自信地运行本书中的示例以及所有未来的 Python 项目。理解如何使用 `pyenv` 和虚拟环境是确保在切换到另一个项目或在其他人的代码上工作时一切顺利的关键技能。你还学会了如何使用 `pip` 安装第三方 Python 库。最后，你了解了如何使用 HTTPie，这是一种简单高效的方式来运行 HTTP 查询，能够提高你在测试 REST API 时的生产力。

在下一章，我们将重点介绍 Python 作为编程语言的一些独特之处，并理解什么是 *Pythonic*。
