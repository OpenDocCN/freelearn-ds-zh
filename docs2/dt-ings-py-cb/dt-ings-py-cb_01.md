# 1

# 数据摄取简介

欢迎来到数据世界的奇妙世界！你准备好开始一段令人兴奋的数据摄取之旅了吗？如果是这样，这本书正是你开始的最佳选择！摄取数据是进入大数据世界的第一步。

**数据摄取**是一个涉及收集和导入数据，并且还要妥善存储数据的过程，以便后续的 **提取、转换和加载**（**ETL**）管道可以利用这些数据。为了实现这一点，我们必须谨慎选择我们将使用的工具以及如何正确配置它们。

在我们的书籍之旅中，我们将使用 **Python** 和 **PySpark** 从不同的数据源检索数据，并学习如何正确存储它们。为了协调所有这些，我们将实现 **Airflow** 的基本概念，以及高效的监控，以确保我们的管道得到覆盖。

本章将介绍有关数据摄取的一些基本概念以及如何设置你的环境以开始任务。

在本章中，你将构建并学习以下食谱：

+   设置 Python 和环境

+   安装 PySpark

+   为 MongoDB 配置 Docker

+   为 Airflow 配置 Docker

+   日志库

+   创建模式

+   在摄取中应用数据治理

+   实现数据复制

# 技术要求

本章中的命令使用 Linux 语法。如果你不使用基于 Linux 的系统，你可能需要调整命令：

+   Docker 或 Docker Desktop

+   选择你喜欢的 SQL 客户端（推荐）；我们推荐 DBeaver，因为它有一个社区免费版本

你可以在这个 GitHub 仓库中找到本章的代码：[`github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook`](https://github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook)。

注意

Windows 用户可能会收到类似于 **Docker Desktop 需要更新的 WSL 内核版本**的错误消息。这可以通过遵循以下步骤来解决：[`docs.docker.com/desktop/windows/wsl/`](https://docs.docker.com/desktop/windows/wsl/)。

# 设置 Python 和其环境

在数据领域，如 **Java**、**Scala** 或 **Python** 这样的语言被广泛使用。前两种语言因其与大数据工具环境（如 **Hadoop** 和 **Spark**）的兼容性而被使用，其中核心部分运行在 **Java 虚拟机**（**JVM**）上。然而，在过去的几年里，由于语言的通用性、易于理解以及社区构建的许多开源库，Python 在数据工程和数据科学领域的使用显著增加。

## 准备工作

让我们为我们的项目创建一个文件夹：

1.  首先，打开你的系统命令行。由于我使用的是 **Windows Subsystem for Linux**（**WSL**），我将打开 WSL 应用程序。

1.  进入你的主目录并创建一个文件夹，如下所示：

    ```py
    $ mkdir my-project
    ```

1.  进入这个文件夹：

    ```py
    $ cd my-project
    ```

1.  按照以下方式检查你的操作系统上的 Python 版本：

    ```py
    $ python -–version
    ```

根据您的操作系统，您可能在这里看到输出，例如，WSL 20.04 用户可能看到以下输出：

```py
Command 'python' not found, did you mean:
 command 'python3' from deb python3
 command 'python' from deb python-is-python3
```

如果您的 Python 路径配置为使用 `python` 命令，您将看到类似以下输出的内容：

```py
Python 3.9.0
```

有时，您的 Python 路径可能配置为使用 `python3` 来调用。您可以使用以下命令尝试：

```py
$ python3 --version
```

输出将与 `python` 命令类似，如下所示：

```py
Python 3.9.0
```

1.  现在，让我们检查我们的 `pip` 版本。这个检查是必要的，因为一些操作系统安装了多个 Python 版本：

    ```py
    $ pip --version
    ```

您应该看到类似的输出：

```py
pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.9)
```

如果您的系统没有安装 `3.8x` 或其他版本的语言，请继续执行 *如何操作* 步骤；否则，您就可以开始以下 *安装 PySpark* 菜谱了。

## 如何操作…

我们将使用来自 Python.org 的官方安装程序。您可以在以下链接找到它：[`www.python.org/downloads/`](https://www.python.org/downloads/):

注意

对于 Windows 用户，检查您的操作系统版本很重要，因为 Python 3.10 可能还不兼容 Windows 7，或者您的处理器类型（32 位或 64 位）。

1.  下载其中一个稳定版本。

在撰写本文时，与这里展示的工具和资源兼容的稳定推荐版本是 `3.8`、`3.9` 和 `3.10`。我将使用 `3.9` 版本，并使用以下链接下载它：[`www.python.org/downloads/release/python-390/`](https://www.python.org/downloads/release/python-390/)。向下滚动页面，您将找到根据操作系统列出的 Python 安装程序的链接，如下面的截图所示。

![图 1.1 – Python.org 3.9 版本下载文件](img/Figure_1.1_B19453.jpg)

图 1.1 – Python.org 3.9 版本下载文件

1.  下载安装文件后，双击它，并按照向导窗口中的说明操作。为了避免复杂性，请选择显示的推荐设置。

以下截图显示了在 Windows 上的样子：

![图 1.2 – Windows 的 Python 安装程序](img/Figure_1.2_B19453.jpg)

图 1.2 – Windows 的 Python 安装程序

1.  如果您是 Linux 用户，您可以使用以下命令从源安装它：

    ```py
    $ wget https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
    $ tar -xf Python-3.9.1.tgz
    $ ./configure –enable-optimizations
    $ make -j 9
    ```

安装 Python 后，您应该能够执行 `pip` 命令。如果不能，请参考此处提供的 `pip` 官方文档页面：[`pip.pypa.io/en/stable/installation/`](https://pip.pypa.io/en/stable/installation/).

## 它是如何工作的…

Python 是一种 **解释型语言**，其解释器扩展了用 **C** 或 **C++** 制成的几个功能。语言包还附带了一些内置库，当然，还有解释器。

解释器的工作方式类似于 Unix shell，可以在 `usr/local/bin` 目录中找到：[`docs.python.org/3/tutorial/interpreter.xhtml`](https://docs.python.org/3/tutorial/interpreter.xhtml).

最后，请注意，本书中许多 Python 第三方包需要安装 `pip` 命令。这是因为 `pip`（即 **Pip Installs Packages** 的缩写）是 Python 的默认包管理器；因此，它用于安装、升级和管理来自 **Python 包索引**（**PyPI**）的 Python 包及其依赖项。

## 更多内容…

即使你的机器上没有 Python 版本，你仍然可以使用命令行或 **HomeBrew**（适用于 **macOS** 用户）来安装它们。Windows 用户也可以从 MS Windows Store 下载它们。

注意

如果你选择从 Windows Store 下载 Python，请确保使用由 Python 软件基金会制作的应用程序。

## 参见

你可以使用 `pip` 安装方便的第三方应用程序，例如 Jupyter。这是一个开源的、基于网络的、交互式（且用户友好）的计算平台，常被数据科学家和数据工程师使用。你可以从这里安装它：[`jupyter.org/install`](https://jupyter.org/install)。

# 安装 PySpark

为了处理、清理和转换大量数据，我们需要一个提供弹性和分布式处理的工具，这就是为什么 **PySpark** 是一个好的选择。它通过 Spark 库提供了一个 API，让你可以使用其应用程序。

## 准备工作

在开始 PySpark 安装之前，我们需要在我们的操作系统中检查我们的 Java 版本：

1.  这里，我们检查 Java 版本：

    ```py
    $ java -version
    ```

你应该看到类似的输出：

```py
openjdk version "1.8.0_292"
OpenJDK Runtime Environment (build 1.8.0_292-8u292-b10-0ubuntu1~20.04-b10)
OpenJDK 64-Bit Server VM (build 25.292-b10, mixed mode)
```

如果一切正常，你应该看到命令的输出中显示上述消息，以及 **OpenJDK 18** 版本或更高。然而，某些系统默认没有安装任何 Java 版本，为了覆盖这种情况，我们需要进行到 *步骤 2*。

1.  现在，我们下载**Java 开发工具包**（**JDK**）。

访问 [`www.oracle.com/java/technologies/downloads/`](https://www.oracle.com/java/technologies/downloads/)，选择你的 **操作系统**，下载 JDK 的最新版本。在撰写本文时，是 JDK 19。

JDK 的下载页面看起来如下：

![图 1.3 – JDK 19 下载官方网页](img/Figure_1.3_B19453.jpg)

图 1.3 – JDK 19 下载官方网页

执行下载的应用程序。点击应用程序以启动安装过程。以下窗口将出现：

注意

根据你的操作系统，安装窗口可能略有不同。

![图 1.4 – Java 安装向导窗口](img/Figure_1.4_B19453.jpg)

图 1.4 – Java 安装向导窗口

点击 **下一步**回答以下两个问题，应用程序将开始安装。你不需要担心 JDK 将安装在哪里。默认情况下，应用程序配置为与标准兼容，以便与其他工具的安装兼容。

1.  接下来，我们再次检查我们的 Java 版本。再次执行命令时，你应该看到以下版本：

    ```py
    $ java -version
    openjdk version "1.8.0_292"
    OpenJDK Runtime Environment (build 1.8.0_292-8u292-b10-0ubuntu1~20.04-b10)
    OpenJDK 64-Bit Server VM (build 25.292-b10, mixed mode)
    ```

## 如何操作…

这里是执行此菜谱的步骤：

1.  从 PyPi 安装 PySpark：

    ```py
    $ pip install pyspark
    ```

如果命令运行成功，安装输出的最后一行将如下所示：

```py
Successfully built pyspark
Installing collected packages: py4j, pyspark
Successfully installed py4j-0.10.9.5 pyspark-3.3.2
```

1.  执行`pyspark`命令以打开交互式 shell。当你在命令行中执行`pyspark`命令时，你应该看到以下消息：

    ```py
    $ pyspark
    Python 3.8.10 (default, Jun 22 2022, 20:18:18)
    [GCC 9.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    22/10/08 15:06:11 WARN Utils: Your hostname, DESKTOP-DVUDB98 resolves to a loopback address: 127.0.1.1; using 172.29.214.162 instead (on interface eth0)
    22/10/08 15:06:11 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    22/10/08 15:06:13 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    Welcome to
          ____              __
         / __/__  ___ _____/ /__
        _\ \/ _ \/ _ `/ __/  '_/
       /__ / .__/\_,_/_/ /_/\_\   version 3.1.2
          /_/
    Using Python version 3.8.10 (default, Jun 22 2022 20:18:18)
    Spark context Web UI available at http://172.29.214.162:4040
    Spark context available as 'sc' (master = local[*], app id = local-1665237974112).
    SparkSession available as 'spark'.
    >>>
    ```

你可以在这里看到一些有趣的消息，例如 Spark 的版本和 PySpark 使用的 Python 版本。

1.  最后，我们按照以下方式退出交互式 shell：

    ```py
    >>> exit()
    $
    ```

## 它是如何工作的...

如本食谱开头所见，Spark 是一个运行在 JVM 之上的强大框架。它也是一个开源工具，可以从大量数据中创建健壮和分布式处理输出。随着 Python 语言在过去几年中的普及，需要一个解决方案来适配 Spark 以便与 Python 一起运行。

PySpark 是一个通过 Py4J 与 Spark API 交互的接口，它动态地允许 Python 代码与 JVM 交互。我们首先需要在我们的操作系统上安装 Java 才能使用 Spark。当我们安装 PySpark 时，它已经包含了 Spark 和 Py4J 组件，这使得启动应用程序和构建代码变得容易。

## 还有更多...

Anaconda 是安装 PySpark 和其他数据科学工具的便捷方式。这个工具封装了所有手动过程，并为与 Python 组件（如**NumPy**、**pandas**或**Jupyter**）交互和安装提供了一个友好的界面：

1.  要安装 Anaconda，请访问官方网站并选择**产品** | **Anaconda 发行版**：[`www.anaconda.com/products/distribution`](https://www.anaconda.com/products/distribution)。

1.  根据你的操作系统下载发行版。

关于如何安装 Anaconda 和其他强大命令的更详细信息，请参阅[`docs.anaconda.com/`](https://docs.anaconda.com/)。

### 使用 virtualenv 与 PySpark

可以配置并使用`virtualenv`与 PySpark，如果你选择这种安装类型，Anaconda 会自动完成。然而，对于其他安装方法，我们需要采取一些额外步骤来确保我们的 Spark 集群（本地或服务器上）能够运行它，这包括指定`virtualenv /bin/`文件夹和你的 PySpark 路径。

## 参见

关于这个主题有一篇不错的文章，*使用 VirtualEnv 与 PySpark*，作者是 jzhang，可以在这里找到：[`community.cloudera.com/t5/Community-Articles/Using-VirtualEnv-with-PySpark/ta-p/245932`](https://community.cloudera.com/t5/Community-Articles/Using-VirtualEnv-with-PySpark/ta-p/245932)。

# 配置 Docker 用于 MongoDB

**MongoDB**是一个**非 SQL**（**NoSQL**）文档型数据库，广泛用于存储**物联网**（**IoT**）数据、应用程序日志等。NoSQL 数据库是一个非关系型数据库，它以不同于 MySQL 或 PostgreSQL 等关系型数据库的方式存储非结构化数据。现在不必过于担心这个问题；我们将在*第五章*中更详细地介绍它。

你的集群生产环境可以处理大量数据并创建具有弹性的数据存储。

## 准备工作

遵循代码组织的良好实践，让我们开始在项目内创建一个文件夹来存储 Docker 镜像：

在我们的项目目录内创建一个文件夹以存储 MongoDB Docker 镜像和数据，如下所示：

```py
my-project$ mkdir mongo-local
my-project$ cd mongo-local
```

## 如何操作…

下面是尝试这个菜谱的步骤：

1.  首先，我们按照以下方式从 Docker Hub 拉取 Docker 镜像：

    ```py
    my-project/mongo-local$ docker pull mongo
    ```

你应该在命令行中看到以下消息：

```py
Using default tag: latest
latest: Pulling from library/mongo
(...)
bc8341d9c8d5: Pull complete
(...)
Status: Downloaded newer image for mongo:latest
docker.io/library/mongo:latest
```

注意

如果你使用的是 WSL 1 版本而不是版本 2，可能会出现错误。你可以通过遵循以下步骤轻松修复此问题：[`learn.microsoft.com/en-us/windows/wsl/install`](https://learn.microsoft.com/en-us/windows/wsl/install)。

1.  然后，我们按照以下方式运行 MongoDB 服务器：

    ```py
    my-project/mongo-local$ docker run \
    --name mongodb-local \
    -p 27017:27017 \
    -e MONGO_INITDB_ROOT_USERNAME="your_username" \
    -e MONGO_INITDB_ROOT_PASSWORD="your_password"\
    -d mongo:latest
    ```

然后，我们检查我们的服务器。为此，我们可以使用命令行来查看哪些 Docker 镜像正在运行：

```py
my-project/mongo-local$ docker ps
```

然后，我们在屏幕上看到以下内容：

![图 1.5 – MongoDB 和 Docker 运行容器](img/Figure_1.5_B19453.jpg)

图 1.5 – MongoDB 和 Docker 运行容器

我们甚至可以在 Docker Desktop 应用程序中检查我们的容器是否正在运行：

![图 1.6 – 运行中的 MongoDB 容器的 Docker Desktop 视图](img/Figure_1.6_B19453.jpg)

图 1.6 – 运行中的 MongoDB 容器的 Docker Desktop 视图

1.  最后，我们需要停止我们的容器。我们需要使用 `Container ID` 来停止容器，这是我们之前在检查正在运行的 Docker 镜像时看到的。我们将在 *第五章* 中重新运行它：

    ```py
    my-project/mongo-local$ docker stop 427cc2e5d40e
    ```

## 它是如何工作的…

MongoDB 的架构使用 `main` 节点与客户端请求交互的概念，例如查询和文档操作。它自动在其分片之间分配请求，分片是这里更大数据集合的子集。

![图 1.7 – MongoDB 架构](img/Figure_1.7_B19453.jpg)

图 1.7 – MongoDB 架构

由于我们可能在机器内还有其他运行的项目或软件应用程序，因此隔离开发中使用的任何数据库或应用程序服务器是一种良好的做法。这样，我们确保没有任何东西干扰我们的本地服务器，调试过程可以更容易管理。

此 Docker 镜像设置在本地创建了一个 MongoDB 服务器，甚至允许我们进行额外的更改，以便模拟任何其他测试或开发场景。

我们使用的命令如下：

+   `--name` 命令定义了我们给容器起的名字。

+   `-p` 命令指定了容器将打开的端口，以便我们可以通过 `localhost:27017` 访问它。

+   `-e` 命令定义了环境变量。在这种情况下，我们为 MongoDB 容器设置了 `root` 用户名和密码。

+   `-d` 是分离模式——也就是说，Docker 进程将在后台运行，我们不会看到输入或输出。然而，我们仍然可以使用 `docker ps` 来检查容器状态。

+   `mongo:latest`表示 Docker 正在拉取此镜像的最新版本。

## 更多内容...

对于频繁用户，手动配置 MongoDB 容器的其他参数，例如版本、镜像端口、数据库名称和数据库凭证，也是可能的。

此镜像的一个带有示例值的版本也作为`docker-compose`文件在官方文档中提供：[`hub.docker.com/_/mongo`](https://hub.docker.com/_/mongo)。

MongoDB 的`docker-compose`文件看起来类似于以下内容：

```py
# Use your own values for username and password
version: '3.1'
services:
  mongo:
    image: mongo
    restart: always
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: example
  mongo-express:
    image: mongo-express
    restart: always
    ports:
      - 8081:8081
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: root
      ME_CONFIG_MONGODB_ADMINPASSWORD: example
      ME_CONFIG_MONGODB_URL: mongodb://root:example@mongo:27017/
```

## 参见

您可以在完整的 Docker Hub 文档中查看 MongoDB：[`hub.docker.com/_/mongo`](https://hub.docker.com/_/mongo)。

# 配置 Docker 用于 Airflow

在这本书中，我们将使用**Airflow**来编排数据摄取并提供日志以监控我们的管道。

Airflow 可以直接安装在您的本地机器和任何服务器上，使用 PyPi([`pypi.org/project/apache-airflow/`](https://pypi.org/project/apache-airflow/))或 Docker 容器([`hub.docker.com/r/apache/airflow`](https://hub.docker.com/r/apache/airflow)). Docker Hub 上可以找到 Airflow 的官方和受支持的版本，并且由**Apache 基金会**社区维护。

然而，配置我们的 Airflow 还有一些额外的步骤。幸运的是，Apache 基金会也有一个包含使 Airflow 工作所需的所有其他要求的`docker-compose`文件。我们只需要完成几个额外的步骤。

## 准备工作

让我们从在我们的机器上初始化 Docker 应用程序开始。您可以使用桌面版本或 CLI 命令。

确保您处于项目文件夹内。创建一个文件夹来存储 Airflow 内部组件和`docker-compose.yaml`文件：

```py
my-project$ mkdir airflow-local
my-project$ cd airflow-local
```

## 如何操作...

1.  首先，我们从 Airflow 官方文档中直接获取`docker-compose.yaml`文件：

    ```py
    my-project/airflow-local$ curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.3.0/docker-compose.yaml'
    ```

您应该看到类似以下输出：

![图 1.8 – Airflow 容器镜像下载进度](img/Figure_1.8_B19453.jpg)

图 1.8 – Airflow 容器镜像下载进度

注意

在您下载时检查此`docker-compose`文件的最稳定版本，因为在此书出版后可能会有更合适的新版本。

1.  接下来，我们创建`dags`、`logs`和`plugins`文件夹，如下所示：

    ```py
    my-project/airflow-local$ mkdir ./dags ./logs ./plugins
    ```

1.  然后，我们创建并设置 Airflow 用户，如下所示：

    ```py
    my-project/airflow-local$ echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
    ```

注意

如果您有任何与`AIRFLOW_UID`变量相关的错误消息，您可以在与您的`docker-compose.yaml`文件相同的文件夹中创建一个`.env`文件，并将变量定义为`AIRFLOW_UID=50000`。

1.  然后，我们初始化数据库：

    ```py
    my-project/airflow-local$ docker-compose up airflow-init
    ```

执行命令后，您应该看到类似以下输出：

```py
Creating network "airflow-local_default" with the default driver
Creating volume "airflow-local_postgres-db-volume" with default driver
Pulling postgres (postgres:13)...
13: Pulling from library/postgres
(...)
Status: Downloaded newer image for postgres:13
Pulling redis (redis:latest)...
latest: Pulling from library/redis
bd159e379b3b: Already exists
(...)
Status: Downloaded newer image for redis:latest
Pulling airflow-init (apache/airflow:2.3.0)...
2.3.0: Pulling from apache/airflow
42c077c10790: Pull complete
(...)
Status: Downloaded newer image for apache/airflow:2.3.0
Creating airflow-local_postgres_1 ... done
Creating airflow-local_redis_1    ... done
Creating airflow-local_airflow-init_1 ... done
Attaching to airflow-local_airflow-init_1
(...)
airflow-init_1       | [2022-10-09 09:49:26,250] {manager.py:213} INFO - Added user airflow
airflow-init_1       | User "airflow" created with role "Admin"
(...)
airflow-local_airflow-init_1 exited with code 0
```

1.  然后，我们启动 Airflow 服务：

    ```py
    my-project/airflow-local$ docker-compose up
    ```

1.  然后，我们需要检查 Docker 进程。使用以下 CLI 命令，您将看到正在运行的 Docker 镜像：

    ```py
    my-project/airflow-local$ docker ps
    ```

这些是我们看到的镜像：

![图 1.9 – docker ps 命令输出](img/Figure_1.9_B19453.jpg)

图 1.9 – docker ps 命令输出

在 Docker 桌面应用程序中，您也可以看到相同的容器正在运行，但界面更加友好：

![图 1.10 – 运行中的 Airflow 容器在 Docker 桌面上的视图](img/Figure_1.10_B19453.jpg)

图 1.10 – 运行中的 Airflow 容器在 Docker 桌面上的视图

1.  然后，我们通过网页浏览器访问 Airflow：

在您首选的浏览器中，输入 `http://localhost:8080/home`。将出现以下屏幕：

![图 1.11 – Airflow UI 登录页面](img/Figure_1.11_B19453.jpg)

图 1.11 – Airflow UI 登录页面

1.  然后，我们登录到 Airflow 平台。由于它是一个用于测试和学习的本地应用程序，Airflow 中用于管理访问的默认凭据（用户名和密码）是 `airflow`。

登录后，将出现以下屏幕：

![图 1.12 – Airflow UI 主页面](img/Figure_1.12_B19453.jpg)

图 1.12 – Airflow UI 主页面

1.  然后，我们停止我们的容器。我们可以停止我们的容器，直到我们达到*第九章*，届时我们将探索 Airflow 中的数据摄取：

    ```py
    my-project/airflow-local$ docker-compose stop
    ```

## 它是如何工作的…

Airflow 是一个开源平台，允许批量数据处理管道的开发、监控和调度。然而，它需要其他组件，如内部数据库，以存储元数据才能正确工作。在这个例子中，我们使用 PostgreSQL 存储元数据，并使用 **Redis** 缓存信息。

所有这些都可以直接在我们的机器环境中逐一安装。尽管看起来很简单，但由于与操作系统、其他软件版本等问题可能存在兼容性问题，可能并不总是如此。

Docker 可以创建一个隔离的环境，并提供所有使其工作的要求。使用 `docker-compose`，它变得更加简单，因为我们可以在其他组件健康的情况下创建只能创建的组件之间的依赖关系。

您还可以打开我们为这个配方下载的 `docker-compose.yaml` 文件，并查看它以更好地探索它。我们还会在第 *第九章* 中详细讨论它。

## 参见

如果您想了解更多关于此 `docker-compose` 文件的工作方式，您可以在 Apache Airflow 官方 Docker 文档的 Apache Airflow 文档页面上查看：[`airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.xhtml`](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.xhtml)。

# 创建模式

**模式**被认为是数据库或表的蓝图。虽然一些数据库严格要求模式定义，但其他数据库可以在没有它的情况下工作。然而，在某些情况下，与数据模式一起工作可能是有益的，以确保应用程序数据架构得到维护，并且可以接收所需的数据输入。

## 准备工作

让我们想象我们需要为学校创建一个数据库来存储有关学生、课程和讲师的信息。有了这些信息，我们知道我们至少有到目前为止的三个表。

![图 1.13 – 三个实体的表图](img/Figure_1.13_B19453.jpg)

图 1.13 – 三个实体的表图

在这个菜谱中，我们将通过使用**实体关系图**（**ERD**），即数据库中实体之间关系的视觉表示，来展示模式是如何连接的。

## 如何做…

下面是尝试这一步骤的步骤：

1.  我们定义了模式类型。以下图示帮助我们了解如何进行这一步骤：

![图 1.14 – 帮助你决定使用哪个模式的图示](img/Figure_1.14_B19453.jpg)

图 1.14 – 帮助你决定使用哪个模式的图示

1.  然后，我们定义每个表列的字段和数据类型：

![图 1.15 – 每个表的列的定义](img/Figure_1.15_B19453.jpg)

图 1.15 – 每个表的列的定义

1.  接下来，我们定义哪些字段可以是空的或`NULL`：

![图 1.16 – 定义哪些列可以是 NULL 的图示](img/Figure_1.16_B19453.jpg)

图 1.16 – 定义哪些列可以是 NULL 的图示

1.  然后，我们创建表之间的关系：

![图 1.17 – 表的关系图](img/Figure_1.17_B19453.jpg)

图 1.17 – 表的关系图

## 它是如何工作的…

在设计数据模式时，我们首先需要做的是定义它们的类型。正如我们在*步骤 1*中的图示中可以看到，应用模式架构取决于数据的目的。

之后，设计表。决定如何定义数据类型可能因项目或目的而异，但决定列可以接收哪些值是很重要的。例如，如果我们知道房间的识别号总是数字，`Teacher`表上的`officeRoom`可以是`Integer`类型，或者如果对识别方式不确定（例如，`Room 3-D`），则可以是`String`类型。

在*步骤 3*中涵盖的另一个重要主题是如何定义哪些列可以接受`NULL`字段。学生的姓名字段可以空着吗？如果不能，我们需要创建一个约束来禁止这种类型的插入。

最后，根据模式类型，制定表之间的关系定义。

## 相关内容

如果你想了解更多关于数据库模式设计和它们的应用的文章，请阅读 Mark Smallcombe 的这篇文章：[`www.integrate.io/blog/database-schema-examples/`](https://www.integrate.io/blog/database-schema-examples/)。

# 在摄取中应用数据治理

**数据治理**是一套确保数据安全、可用、妥善存储、文档化、隐私和准确的方法论。

## 准备工作

**数据摄取**是数据管道过程的开始，但这并不意味着数据治理没有得到广泛应用。最终数据管道输出中的治理状态取决于它在摄取过程中的实施情况。

下面的图示展示了数据摄取通常是如何进行的：

![图 1.18 – 数据摄取过程](img/Figure_1.18_B19453.jpg)

图 1.18 – 数据摄取过程

让我们分析图中的步骤：

1.  **从源获取数据**：第一步是定义数据的类型、其周期性、我们将从哪里收集它以及为什么我们需要它。

1.  **编写数据摄取的脚本**：基于上一步的答案，我们可以开始规划我们的代码将如何表现以及一些基本步骤。

1.  **在临时数据库或其他类型的存储中存储数据**：在摄取和转换阶段之间，数据通常存储在临时数据库或存储库中。

![图 1.19 – 数据治理支柱](img/Figure_1.19_B19453.jpg)

图 1.19 – 数据治理支柱

## 如何做…

逐步，让我们将图 *1.19* 中的支柱归因于摄取阶段：

1.  在数据源级别应用对可访问性的关注，定义允许查看或检索数据的人员。

1.  接下来，有必要对数据进行编目以更好地理解它。由于数据摄取在这里仅作介绍，因此更相关的是涵盖数据源。

1.  质量支柱将应用于摄取和暂存区域，在那里我们控制数据并保持其质量与源保持一致。

1.  然后，让我们定义所有权。我们知道数据源 *属于* 一个业务领域或公司。然而，当我们摄取数据并将其放入临时或暂存存储时，它就变成了我们的责任来维护它。

1.  最后一个支柱涉及在整个管道中保持数据的安全。在所有步骤中，安全性都是至关重要的，因为我们可能正在处理私人或敏感信息。

![图 1.20 – 增加数据摄取](img/Figure_1.20_B19453.jpg)

图 1.20 – 增加数据摄取

## 它是如何工作的…

虽然一些文章定义了“支柱”来创建良好的治理实践，但了解如何应用它们最好的方式是理解它们的组成。正如你在上一节“如何做…*”中看到的，我们将一些项目分配给了我们的管道，现在我们可以理解它们是如何与以下主题相连的：

+   **数据可访问性**：数据可访问性是指一个群体、组织或项目中的成员如何查看和使用数据。信息需要易于使用。同时，它需要对参与过程的人员可用。例如，敏感数据可访问性应限制在某些人或程序。在我们构建的图中，我们将其应用于我们的数据源，因为我们需要理解和检索数据。同样地，它也可以应用于临时存储需求。

+   **数据编目**：编目和记录数据对于业务和工程团队至关重要。当我们知道哪些类型的信息依赖于我们的数据库或数据湖，并且可以快速访问这些文档时，解决问题的行动时间就会缩短。

再次强调，记录我们的数据源可以使摄取过程更快，因为我们每次需要摄取数据时都需要进行发现。

+   **数据质量**：质量始终与数据摄入、处理和加载相关。通过其周期性跟踪和监控数据的预期收入和结果至关重要。例如，如果我们预计每天摄入 300 GB 的数据，突然降至 1 GB，那么肯定出了大问题，这将影响我们最终输出的质量。其他质量参数可以是列数、分区等，我们将在本书的后面部分探讨。

+   **所有权**：谁负责数据？这个定义对于在出现问题时与所有者取得联系或分配责任以保持和维护数据至关重要。

+   **安全性**：如今，数据安全是一个令人担忧的话题。随着关于数据隐私的许多法规的出现，数据工程师和科学家至少需要了解加密、敏感数据以及如何避免数据泄露的基础知识。甚至用于工作的语言和库也需要评估。这就是为什么这个条目被归入*图 1*的三个步骤中。19。

除了我们探讨的主题外，全球数据治理项目有一个至关重要的角色，称为**数据管理员**，负责管理组织的资产数据，并确保数据准确、一致和安全。总之，数据管理是管理和监督组织的资产数据。

## 参考内容

你可以在这里了解更多关于最近发现的数据工程中最常用工具之一的一个漏洞：[`www.ncsc.gov.uk/information/log4j-vulnerability-what-everyone-needs-to-know`](https://www.ncsc.gov.uk/information/log4j-vulnerability-what-everyone-needs-to-know)。

# 实施数据复制

**数据复制**是在数据环境中应用的一个过程，用于创建数据的多个副本并将它们存储在不同的位置、服务器或站点。这种技术通常用于创建更好的可用性，以避免在停机或自然灾害影响数据中心时数据丢失。

## 准备工作

你会在论文和文章中找到关于**数据复制**决策的不同类型（甚至名称）。在这个菜谱中，你将学习如何决定哪种复制更适合你的应用程序或软件。

## 如何做…

让我们开始构建实施数据复制的基本支柱：

1.  首先，我们需要决定复制的规模，这可以通过存储数据的一部分或全部来完成。

1.  下一步是考虑复制何时进行。它可以在新数据到达存储时同步进行，或在特定时间段内进行。

1.  最后一个基本支柱是数据是增量形式还是批量形式。

最后，我们将得到一个如下所示的图表：

![图 1.21 – 数据复制模型决策图](img/Figure_1.21_B19453.jpg)

图 1.21 – 数据复制模型决策图

## 它是如何工作的…

分析前面的图示，我们有三个主要问题需要回答，关于扩展、频率以及我们的复制将是增量还是批量。

对于第一个问题，我们决定复制将是完全的还是有选择的。换句话说，数据将始终被复制，无论进行了何种类型的交易或更改，或者只是部分数据将被复制。一个真实的例子是跟踪所有商店销售或仅跟踪最昂贵的销售。

第二个问题，与频率相关，是决定何时进行复制。这个问题也需要考虑相关成本。实时复制通常更昂贵，但同步性保证了几乎没有任何数据不一致性。

最后，考虑数据将如何传输到复制站点也是相关的。在大多数情况下，一个带有脚本的调度器可以复制小批量数据并降低运输成本。然而，在数据摄取过程中，可以使用大量复制，例如将当前批次的原始数据从源复制到冷存储。

## 还有更多...

在过去几年中，数据复制的一种使用增加的方法是**冷存储**，用于保留不常使用或甚至不活跃的数据。与此类复制相关的成本微乎其微，并保证了数据的长期性。你可以在所有云服务提供商中找到冷存储解决方案，例如**Amazon Glacier**、**Azure Cool Blob**和**Google Cloud Storage Nearline**。

除了复制之外，像**通用数据保护条例**（**GDPR**）这样的监管合规法律也得益于这种存储方式，因为对于某些案例场景，用户数据需要保留数年。

在本章中，我们探讨了基本概念，并为本书后续章节和食谱奠定了基础。我们从 Python 安装开始，准备我们的 Docker 容器，并了解了数据治理和复制概念。在接下来的章节中，你将观察到几乎所有主题都是相互关联的，你将理解在 ETL 过程开始时理解它们的相关性。

# 进一步阅读

+   [`www.manageengine.com/device-control/data-replication.xhtml`](https://www.manageengine.com/device-control/data-replication.xhtml)
