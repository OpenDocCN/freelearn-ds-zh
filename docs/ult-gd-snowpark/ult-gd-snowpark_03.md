

# 第二章：使用 Snowpark 建立基础

在上一章中，你学习了 Snowpark 的基础知识、其优势以及它如何允许开发者使用 Python 开发复杂的数据应用程序。本章将专注于使用 Snowpark 建立坚实的基础。在这里，你将学习如何配置和操作 Snowpark，选择编码风格和结构，并深入探索 Snowpark 的基本原理。这将帮助你获得实际的知识和技能，以高效地使用 Snowpark，包括设置环境、结构化代码以及利用它处理不同的工作负载。

本章将涵盖以下主要内容：

+   配置 Snowpark 开发环境

+   使用 Snowpark 操作

+   为 Snowpark 建立项目结构

# 技术要求

对于本章，你需要一个活跃的 Snowflake 账户，并在本地安装了配置好的 Anaconda Python。你可以参考以下文档以获取安装说明：

+   你可以在[`signup.snowflake.com/`](https://signup.snowflake.com/)注册 Snowflake 试用账户

+   要配置 Anaconda，请遵循[`conda.io/projects/conda/en/latest/user-guide/getting-started.html`](https://code.visualstudio.com/docs/python/python-tutorial)中的指南

+   此外，要安装和设置 VS Code 中的 Python，请遵循[`code.visualstudio.com/docs/python/python-tutorial`](https://code.visualstudio.com/docs/python/python-tutorial)中的指南

+   要了解如何在 VS Code 中操作 Jupyter Notebook，请访问[`code.visualstudio.com/docs/datascience/jupyter-notebooks`](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

本章的支持材料可在本书的 GitHub 仓库中找到，网址为[`github.com/PacktPublishing/The-Ultimate-Guide-To-Snowpark`](https://github.com/PacktPublishing/The-Ultimate-Guide-To-Snowpark)。

# 配置 Snowpark 开发环境

开发 Snowpark 的第一步是设置 Snowpark 开发环境。开发者有很大的灵活性，可以选择他们可以用来开始使用 Snowpark 的**集成开发环境**（**IDEs**）；他们唯一需要做的是安装 Snowpark 客户端**应用程序编程接口**（**API**）并连接到他们的 Snowflake 账户。开发环境可以是一个包含你喜欢的 IDE 的本地 Python 环境，或者 Snowsight 中的新 Snowflake Python 工作表。本节将涵盖设置 Snowpark Python 工作表和本地开发环境。

## Snowpark Python 工作表

Snowflake 发布了 *Snowflake Worksheets for Python*，这是一种新的工作表类型，允许在 Snowflake 内部开发基于 Python 的 Snowpark 环境。这一颠覆性功能允许开发者轻松利用 Snowpark Python 的力量在 Snowsight 中执行数据处理和创建数据管道、**机器学习**（**ML**）模型和应用，通过直接将 Snowpark Python 集成到浏览器中，无需设置 Python 环境或在客户端安装开源库。相反，开发者可以轻松使用 Anaconda 中的预存软件包或从阶段导入自己的 Python 文件到工作表中。此外，他们可以快速将 Python 工作表作为存储过程部署。

### 使用 Python 工作表的前提条件

要启用和使用 Snowflake Python 工作表，您必须首先在 Snowsight 中确认 Anaconda 的服务条款。以下步骤应由组织管理员执行：

1.  首先，您必须登录到您的 Snowflake 账户。在左侧面板的 **切换角色** 部分中，切换到用户上下文中的 **ORGADMIN** 角色：

![图 2.1 – Snowflake 中的 ORGADMIN 角色](img/B19923_02_1.jpg)

图 2.1 – Snowflake 中的 ORGADMIN 角色

1.  然后，转到 **管理员** | **账单与** **条款**：

![图 2.2 – 管理员 | 账单与条款](img/B19923_02_2.jpg)

图 2.2 – 管理员 | 账单与条款

1.  点击 **Anaconda** **Python 软件包** 旁边的 **启用**：

![图 2.3 – 启用 Anaconda Python 软件包](img/B19923_02_3.jpg)

图 2.3 – 启用 Anaconda Python 软件包

1.  您将看到一个弹出窗口，如下面的截图所示。点击 **确认并继续** 以启用软件包：

![图 2.4 – Anaconda 的条款和服务](img/B19923_02_4.jpg)

图 2.4 – Anaconda 的条款和服务

每次创建新的 Snowflake 环境，您都需要启用 Anaconda 软件包。现在我们已经启用了这些软件包，让我们看看如何使用 Python 工作表。

### 在 Snowflake Snowsight 中创建数据库和模式

要使用 Snowflake 的 Snowsight UI 界面创建名为 `SNOWPARK_DEFINITIVE_GUIDE` 的数据库和名为 `MY_SCHEMA` 的模式，请执行以下步骤：

1.  前往 Snowsight 网络界面，使用您的 Snowflake 凭据登录，并导航到 **数据** | **数据库** 部分。这通常位于左侧导航菜单中：

![图 2.5 – 数据库部分](img/B19923_02_5.jpg)

图 2.5 – 数据库部分

1.  在右上角寻找一个按钮或链接，上面写着 **+ 数据库** 并点击它。在出现的对话框中，输入 **SNOWPARK_DEFINITIVE_GUIDE** 作为新数据库的名称。如有需要，您可以指定其他设置，例如 **注释**，然后点击 **创建**：

![图 2.6 – 新数据库对话框](img/B19923_02_6.jpg)

图 2.6 – 新数据库对话框

1.  创建数据库后，点击 **SNOWPARK_DEFINITIVE_GUIDE** 数据库。这将带您进入以下页面：

![图 2.7 – SNOWPARK_DEFINTIVE_GUIDE 页面](img/B19923_02_7.jpg)

图 2.7 – SNOWPARK_DEFINTIVE_GUIDE 页面

1.  在此页面上，寻找一个写着**+ 模式**的按钮或链接并点击它。在出现的对话框中，将新模式的名字输入为**MY_SCHEMA**。可选地，你可以指定其他设置，例如**注释**和**管理访问**。点击**创建**来创建模式：

![图 2.8 – 新模式对话框](img/B19923_02_8.jpg)

图 2.8 – 新模式对话框

在本书中，除非特定章节明确指示，否则我们将始终使用相同的数据库和模式。

### 在 Snowflake 中使用 Python 工作表

Python 工作表附带一个示例代码模板，可以作为起点使用。整个体验都基于 Snowflake 的 Snowsight UI。按照以下步骤创建和使用 Snowflake 中的 Python 工作表：

1.  从 Snowsight UI 的菜单导航到**工作表**部分：

![图 2.9 – 工作表菜单选项](img/B19923_02_9.jpg)

图 2.9 – 工作表菜单选项

1.  在**工作表**面板中，点击右侧的**+**图标并选择**Python 工作表**来创建 Python 工作表：

![图 2.10 – 创建 Python 工作表](img/B19923_02_10.jpg)

图 2.10 – 创建 Python 工作表

1.  选择一个数据库和模式，这样你就可以在工作表上下文中工作。Python 代码的作用域将基于这些细节：

![图 2.11 – 工作表上下文](img/B19923_02_11.jpg)

图 2.11 – 工作表上下文

1.  工作表有一个处理函数，当工作表执行时会被调用。**设置**菜单允许你配置你的工作表。默认的处理函数是**main()**，返回类型可以指定为**Table()**、**Variant**或**String**。结果将以你选择的格式显示：

![图 2.12 – 工作表设置](img/B19923_02_12.jpg)

图 2.12 – 工作表设置

工作表首选项，如代码检查和换行，也可以在此处自定义。

1.  可以通过点击工作表右上角的**运行**按钮来执行 Python 代码：

![图 2.13 – 执行工作表](img/B19923_02_13.jpg)

图 2.13 – 执行工作表

你可以在工作表中开始开发和执行 Python 代码以查看结果。

### 在 Python 工作表中管理 Anaconda 包

Python 工作表附带一个集成的 Anaconda 环境，支持导入最常见的 Anaconda 库，无需担心依赖管理。它还支持从内部阶段导入自定义包。在本节中，我们将探讨如何管理 Python 中的 Anaconda 包。为此，请执行以下步骤：

1.  从菜单导航到**包**标签并选择**Anaconda 包**。这将显示预安装的包及其版本：

![图 2.14 – Anaconda 包](img/B19923_02_14.jpg)

图 2.14 – Anaconda 包

1.  您可以使用搜索栏搜索和安装所需的包：

![图 2.15 – 搜索包](img/B19923_02_15.jpg)

图 2.15 – 搜索包

1.  您还可以通过从相应的下拉列表中选择可用版本来修改包的版本：

![图 2.16 – 包版本](img/B19923_02_16.jpg)

图 2.16 – 包版本

1.  您可以使用以下查询通过 SQL 检查 Snowflake 中可用的 Python 包：

    ```py
    SELECT distinct package_name
    FROM information_schema.packages
    WHERE language = 'python';
    ```

    此查询从信息模式中的包视图获取结果：

![图 2.17 – Anaconda 包查询](img/B19923_02_17.jpg)

图 2.17 – Anaconda 包查询

1.  如果您需要检查特定包的版本信息，可以通过包的名称过滤查询：

    ```py
    SELECT *
    FROM information_schema.packages
    WHERE (package_name='numpy' AND language = 'python');
    ```

    上述代码的输出如下：

![图 2.18 – Anaconda Python 包查询](img/B19923_02_18.jpg)

图 2.18 – Anaconda Python 包查询

此查询显示了我账户中可用的 Anaconda 包列表。在下一节中，我们将学习如何管理自定义 Python 包。

### 在 Python 工作表中管理自定义包

Python 工作表也支持导入自定义 Python 包的功能，这些包可以在 Python 代码中使用。但是，必须将包上传到内部阶段，这是 Snowflake 账户的存储部分，并从中导入。在 Snowsight 中，您可以将文件加载到命名的内部阶段区域，这样您就可以方便地在 Python 工作表中查看和使用它们，或者使用 SQL 将数据加载到表中。但是，需要注意的是，Snowsight 不支持将文件加载到用户或表阶段。

要创建一个命名的内部阶段，请确保您使用的角色在相关数据库和模式上具有**使用**权限，以及在模式上具有**创建阶段**权限。让我们开始吧：

1.  登录 Snowsight。

1.  访问**数据**部分并导航到**数据库**。

1.  选择您想要创建阶段并加载文件的数据库和模式。

1.  点击**创建**，选择**阶段**，然后点击**Snowflake 管理的**：

![图 2.19 – 创建阶段](img/B19923_02_19.jpg)

图 2.19 – 创建阶段

1.  为阶段提供一个名称，并选择启用目录表以使阶段可视化。完成后，点击**创建**：

![图 2.20 – 创建内部阶段](img/B19923_02_20.jpg)

图 2.20 – 创建内部阶段

要使用 Snowsight 将文件加载到 Snowflake 管理的命名内部阶段，请按照以下步骤操作：

1.  访问**数据**部分并选择**数据库**。

1.  选择您创建阶段的数据库模式，并选择阶段本身。

1.  点击**+ 文件**将所需的文件加载到阶段。

1.  在出现的**上传您的文件**对话框中，选择您想要上传的文件（可以同时选择多个文件）：

![图 2.21 – 上传自定义包](img/B19923_02_21.jpg)

图 2.21 – 上传自定义包

1.  可选地，在阶段内指定或创建一个路径，用于存储文件。

1.  点击 **上传**。

1.  一旦文件成功加载到阶段，您就可以根据需求执行各种操作。

这样，包已经被上传到阶段，并准备好导入。

重要提示

允许的最大文件大小为 50 MB。您需要在数据库和模式上具有 **USAGE** 权限，以及在阶段上具有 **WRITE** 权限。

一旦创建阶段并上传了包，您就可以导入模块，以便在程序中使用它。要导入包，请按照以下步骤操作：

1.  从 **包** 菜单中选择 **阶段包**：

![图 2.22 – 阶段包](img/B19923_02_22.jpg)

图 2.22 – 阶段包

1.  输入包的路径。您可以使用 **@Stage/path/to/package.py** 在同一数据库和模式中引用阶段。如果阶段在不同的数据库和模式中，您可以使用 **@Database.Schema.Stage/path/to/package.py**：

![图 2.23 – 导入阶段包](img/B19923_02_23.jpg)

图 2.23 – 导入阶段包

1.  点击 **导入** 以安装包。现在模块将在 **已安装包** 下可见：

![图 2.24 – 安装阶段包](img/B19923_02_24.jpg)

图 2.24 – 安装阶段包

1.  您可以通过使用 **import <包名>** 将包导入到您的代码中。

在下一节中，我们将介绍如何使用 UI 部署 Python 存储过程。

### 部署 Python 存储过程

我们工作表中的 Python 脚本可以无缝地部署为存储过程。然后可以在常规 SQL 上下文中使用它，或安排为任务执行。要部署存储过程，请按照以下步骤操作：

1.  点击工作表右上角的 **部署** 按钮。

1.  给存储过程起一个名称；也可以指定一个可选的注释，提供有关它的信息。如果存储过程已存在，则勾选 **如果存在则替换** 复选框以替换现有存储过程：

![图 2.25 – 部署存储过程](img/B19923_02_25.jpg)

图 2.25 – 部署存储过程

1.  点击 **部署** 以部署存储过程。它将在您在工作表上下文中提供的数据库和模式下部署。

1.  现在您可以使用以下代码在工作表中执行存储过程：

    ```py
    CALL DATABASE.SCHEMA.MY_STORED_PROCEDURE();
    ```

在下一节中，我们将介绍 Python 工作表的各项功能。

### Python 工作表的功能

Python 工作表包含一些对开发者友好的功能，并支持生产力。以下是 Python 工作表的某些独特特性：

+   **交互式 Python 环境**：工作表支持 Python 语言，并支持 Snowpark**用户自定义函数**（**UDFs**）和存储过程。诸如语法高亮、关键字类型敏感的自动完成以及诸如未声明的变量或无效方法使用等便捷的诊断功能有助于提高开发者的生产力：

![图 2.26 – 交互式 Python 环境](img/B19923_02_26.jpg)

图 2.26 – 交互式 Python 环境

+   **Python 库支持**：Python 工作表附带一个集成的 Anaconda 环境，支持导入最常见的 Anaconda 库，无需担心依赖管理。它还支持从内部阶段导入自定义包。

+   **Snowpark 调试**：Python 工作表可以使用**show()**或**print()**函数在 Snowsight 中显示 DataFrame 的结果。DataFrame 的预览也可以返回以显示表格格式的输出，这在调试 Python 程序时非常有用：

![图 2.27 –  Snowpark 调试](img/B19923_02_27.jpg)

图 2.27 – Snowpark 调试

+   **协作**：Snowpark Python 工作表可以与开发者共享。这使得协作变得更加容易，因为多个开发者可以同时访问并工作在工作表上：

![图 2.28 – 工作表协作](img/B19923_02_28.jpg)

图 2.28 – 工作表协作

+   **图表和数据探索**：Python 工作表提供了一种方便的方式来可视化和探索数据以及 DataFrame 作为图表，这有助于数据探索并提供了一种快速分析数据的方法：

![图 2.29 – 图表和数据探索](img/B19923_02_29.jpg)

图 2.29 – 图表和数据探索

在下一节中，我们将介绍 Python 工作表的局限性。

### Python 工作表的局限性

Python 工作表在 Snowsight 中相对较新，在功能方面存在一些局限性。让我们更仔细地看看：

+   默认情况下，低于**WARN**级别的日志级别不会出现在**结果**区域中。

+   Python 工作表不支持断点或代码部分的选执行。相反，整个代码都在工作表中运行。

+   Python 工作表无法显示它生成的 Python 代码中的图像或其他元素。它只能显示通过 DataFrame 返回的结果。

+   它仅支持 Python 3.8 以及使用 Python 3.8 的库。

Snowflake 不断更新 Python 工作表的功能，为开发者提供改进。因此，Python 工作表使得开发者能够在 Snowflake 环境中更容易、更快地编写 Python 代码。

## 在本地环境中进行 Snowpark 开发

Snowpark 也可以方便地从本地环境使用您喜欢的 IDE 进行开发。Snowflake 与 Anaconda 的合作的关键优势在于 Anaconda 的 Snowflake Snowpark for Python 通道，其中包含运行 Snowpark 所需的 Python 包。要使用此通道，Anaconda 或 Miniconda 应该安装在机器上。在本节中，我们将向您介绍如何使用 Anaconda 在本地开发环境中设置 Snowpark。

重要提示

本书将使用 Anaconda 进行 Snowpark 开发，因为这是一种推荐的方法，并利用 Anaconda 的包管理器的优势，轻松设置和管理 Snowpark 开发环境。

Snowpark API 需要安装 Python 3.8。Snowflake 建议您使用 Anaconda 以便于包管理。您可以通过在 **命令行界面** （**CLI**） 中运行以下命令来检查您拥有的 Python 版本：

```py
python –-version
```

您的输出应类似于以下内容：

![图 2.30 – Python CLI 版本](img/B19923_02_30.jpg)

图 2.30 – Python CLI 版本

您也可以通过运行以下命令在 Python 代码中检查 Python 版本：

```py
from platform import python_version
print(python_version())
```

这将输出类似于以下内容：

![图 2.31 – Python 版本](img/B19923_02_31.jpg)

图 2.31 – Python 版本

Python 安装完成后，需要创建虚拟环境。因此，让我们创建一个。

### 创建虚拟环境

建议您创建一个 Python 虚拟环境，以确保在处理 Snowpark 时拥有无缝的开发者体验；它隔离了 Snowpark API，并允许您管理所有开发所需的依赖项。要使用 Anaconda 创建虚拟环境，请运行以下命令：

```py
conda create --name def_gui_3.8_env --override-channels --channel https://repo.anaconda.com/pkgs/snowflake python=3.8
```

此命令创建了一个名为 `def_gui_3.8_env` 的新 Python 虚拟环境，并使用 Python 3.8 安装了必要的包，例如 `numpy` 和 `pandas`，来自 Anaconda 的 Snowflake 通道。

### 安装 Snowpark Python 包

在安装包之前，让我们使用以下命令激活我们的 Python 虚拟环境：

```py
conda activate def_gui_3.8_env
```

您可以使用以下命令从 Anaconda 的 Snowflake 通道安装 Snowpark 包：

```py
conda install --channel https://repo.anaconda.com/pkgs/snowflake Snowflake-snowpark-python
```

接下来，让我们安装一些额外的包。

### 安装额外的 Python 包

要安装开发所需的额外包，例如 `pandas` 和 `numpy`，您可以使用相同的 Anaconda Snowflake 通道：

```py
conda install --channel https://repo.anaconda.com/pkgs/snowflake numpy pandas
```

虚拟环境现在已准备好用于开发，并连接到您喜欢的 IDE、Jupyter Notebook 或 VS Code 开发环境。同样，我们也可以安装 IPython Notebook。

### 配置 Jupyter Notebook

Jupyter Notebook 是开发者中最受欢迎的 IDE 之一。在本节中，我们将介绍如何配置 Jupyter IDE 以支持 Snowpark，因为本书的示例使用 Jupyter。Jupyter Notebook 需要安装在本地的环境中。Jupyter 环境与 Anaconda 一起安装。因此，让我们打开 Jupyter Notebook：

1.  如果在上一节中没有激活，则必须激活 **def_gui_3.8_env** 虚拟环境以进行开发。要激活虚拟环境，请运行以下命令：

    ```py
    conda activate def_gui_3.8_env
    ```

1.  通过运行以下命令启动 Jupyter Notebook：

    ```py
    jupyter notebook
    ```

1.  在 Jupyter Notebook 网页上，点击右上角的 **New** 按钮。从下拉菜单中，在 **Notebooks** 部分下选择 **Python 3**。这将打开一个新的笔记本，其中包含一个空单元格，准备好执行代码。

重要提示

本书将使用 Jupyter Notebook 来展示所有示例。

### 导入 Snowpark 模块

Snowpark API 的 Python 类是 `snowflake.snowpark` 模块的一部分。您可以通过指定它们的名称从模块中导入特定的类。例如，要导入 `average` 函数，您可以使用以下代码：

```py
from snowflake.snowpark.functions import avg
```

现在开发环境已经设置好了，让我们学习如何使用 Snowpark 进行操作。

# 使用 Snowpark 操作

Snowpark for Python 由直接在 Python 引擎上执行的客户端 API、UDF 和存储过程组成。以下截图显示了您可以选择的各种 Snowpark 对象：

![图 2.32 – Snowpark Python 对象](img/B19923_02_32.jpg)

图 2.32 – Snowpark Python 对象

Snowpark 使用 DataFrame 对象来查询和处理数据。使用 Snowpark 操作的指导原则是将数据保持在 Snowflake 中，并使用各种 Snowflake 对象在 Snowflake 内部进行处理。以下图显示了 Snowpark 的架构：

![图 2.33 – Snowpark Python 架构](img/B19923_02_33.jpg)

图 2.33 – Snowpark Python 架构

在下一节中，我们将介绍 Python 引擎。

## Python 引擎

Python 引擎是一个由 Anaconda 驱动的安全沙盒 Python 环境，它在 Snowflake 的虚拟仓库上执行，并托管在 Snowflake 的计算基础设施上。这允许您使用 Python 处理数据，而无需将数据从环境中提取出来。Python 引擎由 UDF 引擎和存储过程引擎组成。UDF 引擎是一个受限的引擎，不能在 Snowflake 外部读取或写入数据，而存储过程引擎则更为宽容，并包含一个用于与 Snowflake 数据库交互的会话对象。

## 客户端 API

客户端 API 是`snowflake-snowpark-python`库，它可以在任何 Python 环境中安装。它提供会话和 DataFrame API 作为方法来支持在 Snowflake 的 Python 引擎中下推查询。客户端 API 包括会话、DataFrame 等显著对象。让我们详细看看这些。

### 与会话一起工作

Snowpark 会话是 Snowpark API 的一部分，连接到 Snowflake 以与之交互并使用 Snowpark 对象执行操作。Snowpark API 中的`Session`函数负责会话操作。要导入 Snowpark 的`Session`函数，请运行以下命令：

```py
from snowflake.snowpark import Session
```

Snowpark 会话包含一个 Python 字典，其中包含建立 Snowflake 连接所需的参数值：

```py
connection_parameters = {
    "account": "<your snowflake account identifier>",
    "user": "<your snowflake username>",
    "password": "<your snowflake password>",
    "role": "<your snowflake role>", # optional
    "warehouse": "<your snowflake warehouse>",  # optional
    "database": "<your snowflake database>",  # optional
    "schema": "<your snowflake schema>" # optional
}
```

连接包含以下必填参数：

+   **账户**: Snowflake 账户标识符为**<orgname>-<account_name>**。无需指定**snowflakecomputing.com**后缀。

+   **用户**: Snowflake 用户的用户名。

+   **密码**: 用于验证 Snowflake 的密码。

以下是可以传递给连接的可选参数：

+   **角色**: 在 Snowpark 会话中使用的角色。如果留空，将使用用户的默认角色。

+   **仓库**: 用于执行进程的仓库。如果留空，将使用用户的默认仓库。

+   **数据库**: 执行上下文中的数据库。如果留空，将使用用户的默认数据库。

+   **模式**: 执行上下文中的模式。如果留空，将使用用户的默认模式。

您可以通过将此字典传递给会话构建器来创建`session`对象，然后使用它来建立会话：

```py
session = Session.builder.configs(connection_parameters).create()
```

会话创建后，`session`对象充当与 Snowflake 交互的句柄，通过各种方法执行读取和操作数据等操作。以下是一些可用的会话方法：

```py
print("Session Current Account:", session.get_current_account())
```

最后，您可以使用`close()`方法关闭会话。这将终止会话及其关联的持续查询：

```py
session.close()
```

Snowflake 建议在进程完成后关闭会话。可以使用`Session`方法来建立会话并与`Session`对象交互。

#### 高级认证

Snowpark 会话也支持高级认证，如密钥对认证，如果已为连接到 Snowflake 的用户配置，则可以使用。私钥必须序列化然后传递给会话构建器的连接对象。我们将使用 hazmat 加密包来序列化私钥。此私钥作为变量提供：

```py
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
private_key_plain_text = '''-----BEGIN PRIVATE KEY-----
< your private key >
-----END PRIVATE KEY-----'''
private_key_passphrase = '<your private key passphase>'
```

我们使用口令对私钥进行编码，然后将其序列化以分配给传递给 Snowflake 连接对象的变量：

```py
private_key_encoded = private_key_plain_text.encode()
private_key_passphrase_encoded = private_key_passphrase.encode()
private_key_loaded = serialization.load_pem_private_key(
    private_key_encoded,
    password = private_key_passphrase_encoded,
    backend = default_backend()
)
private_key_serialized = private_key_loaded.private_bytes(
    encoding = serialization.Encoding.DER,
    format = serialization.PrivateFormat.PKCS8,
    encryption_algorithm = serialization.NoEncryption()
)
```

连接参数将以序列化的形式传递私钥，并且可以使用会话构建器建立会话：

```py
connection_parameters = {
    "account": "<your snowflake account identifier>",
    "user": "< your snowflake username>",
    "private_key": private_key_serialized,
    "warehouse": "<your snowflake warehouse>",
    "database": "<your snowflake database>",
    "schema": "<your snowflake schema>"
}
session = Session.builder.configs(connection_parameters).create()
```

接下来，让我们讨论 Snowpark DataFrames。

### Snowpark DataFrames

Snowpark for Python 由一个客户端 API 组成，它提供了一个基于 DataFrame 的方法，使用 DataFrame 对象查询和处理数据。以下图表解释了所使用的代码块：

![图 2.34 – Snowpark DataFrame API](img/B19923_02_34.jpg)

图 2.34 – Snowpark DataFrame API

Snowpark DataFrame API 提供了更高的效率，开发者所需的努力更少，因为它具有简洁且易于理解和调试的语法。以下图表比较了 DataFrame 和 SQL 查询：

![图 2.35 – Snowpark DataFrame 与查询](img/B19923_02_35.jpg)

图 2.35 – Snowpark DataFrame 与查询

此外，Snowpark DataFrames 可以直接从表、视图和`SELECT`语句中读取数据，这些语句支持下推，以便它们可以在 Snowflake 中执行。Snowpark 客户端 API 还支持将 pandas DataFrame 转换为 Snowflake DataFrame，以便数据可以写回 Snowflake。最后，Snowpark DataFrames 支持懒加载，因为数据计算是在动作调用时执行的。

注意

Snowpark 中的懒加载意味着数据操作不会在定义时立即执行。相反，Snowpark 会构建一系列转换，直到你明确请求结果时才执行。这种方法优化了性能和资源使用，允许你高效且交互式地构建复杂的数据工作流。懒加载是处理大型数据集和优化 Snowpark 中的数据处理任务的关键特性。

#### 使用 DataFrames

DataFrames 是 Snowpark 中的数据集对象，用于查询和处理数据。它们代表提供懒加载的关系数据集。DataFrame 以下推方式执行 SQL，并可以执行创建对象、从 Python 代码中读取、写入和操作数据等操作。`Session`对象中的各种方法用于处理 DataFrames。

让我们创建一个名为`SAMPLE_EMPLOYEE_DATA`的员工数据表：

```py
session.sql('CREATE OR REPLACE TABLE SAMPLE_EMPLOYEE_DATA (id INT,name VARCHAR, age INT, email VARCHAR, city VARCHAR,country VARCHAR)').collect()
```

上一段代码将创建一个包含员工数据所需字段的表格。为了操作目的，让我们向表格中插入一些数据：

```py
session.sql("""
    INSERT INTO SAMPLE_EMPLOYEE_DATA VALUES
    (1,'John Doe',25,'johndoe@example.com','New York','USA'),
    (2,'Jane Smith',30,'janesmith@example.com','Los Angeles','USA'),
    (3,'Michael Johnson',35,'michaeljohnson@example.com','London', 
       'UK'),
    (4,'Sarah Williams',28,'sarahwilliams@example.com','Leeds', 
       'UK'),
    (5,'David Brown',32,'davidbrown@example.com','Tokyo','Japan'),
    (6,'Emily Davis',29,'emilydavis@example.com','Sydney',
       'Australia'),
    (7,'James Miller',27,'jamesmiller@example.com','Dallas','USA'),
    (8,'Emma Wilson',33,'emmawilson@example.com','Berlin','Germany'),
    (9,'Alexander Taylor',31,'alexandertaylor@example.com',
       'Rome','Italy'),
    (10,'Olivia Anderson',26,'oliviaanderson@example.com',
        'Melbourne','Australia')
""").collect()
```

上一段代码将填充表格，我们可以查询这些数据。要查询数据，我们可以直接传递 SQL 语句：

```py
session.sql("SELECT count(*) FROM SAMPLE_EMPLOYEE_DATA").collect()
```

上一段代码在 Snowflake 中执行后会返回结果。我们还可以将这些结果存储在 DataFrame 中，以便在 Python 中操作：

```py
from snowflake.snowpark.functions import col
df_subset_row = session.table(
    "SAMPLE_EMPLOYEE_DATA").filter(col("id") == 1)
```

以下代码将结果保存到`df_subset_row` DataFrame 中，可以使用`show()`显示：

```py
df_subset_row.show()
```

这是输出：

![图 2.36 – DataFrame 数据](img/B19923_02_36.jpg)

图 2.36 – DataFrame 数据

在下一节中，我们将探讨 Snowpark UDFs 和存储过程。

关于代码片段的说明

下一个部分中提供的示例已经有意简化。我们的主要目标是掌握和区分概念，而不是深入研究复杂场景。然而，在接下来的章节中，我们将深入研究更复杂的示例。

## UDFs

Snowpark for Python 支持 UDF，允许开发者编写可重用的自定义 lambda 和函数，通过 DataFrames 处理数据。像内置函数一样，UDF 可以从 SQL 中调用，这增强了 SQL 的功能，使其能够执行它原本不具备或执行不佳的操作。UDF 还提供了一种封装功能的方法，以便您可以从代码的多个地方重复调用它。例如，您可以编写一个返回单个值（称为标量函数）的 UDF，也称为 UDF，或者返回一组值（称为表函数）的 UDF，也称为**用户自定义表函数**（**UDTF**）。这些 UDF 可以从 Python 工作表内部开发，或者通过在本地开发环境中使用 Snowpark 来开发。

### 标量 UDF

标量 UDF 对每一行调用一次，并为每个输入行返回一个输出行。这些 UDF 的调用方式与标准 SQL 函数相同，以列或表达式作为参数。它产生一个由单个列/值组成的行作为输出。数据在多节点虚拟仓库的每个节点内并行处理。

#### 使用 UDF

一旦创建了一个 Snowpark 会话，UDF 可以被转换成一个标准函数，该函数可以在 Snowflake 中注册：

```py
def <main Python function name>(<arguments>):
    return <function output>
from snowflake.snowpark.types \
    import <specific Snowpark DataType object>
snowpark_session.add_packages(
    'List of native packages in Anaconda Channel')
snowpark_session.add_import('Path to Local File')
snowpark_session.udf.register(
    func = <Main Function Name>
  , return_type = <Return Type of Snowpark DataType object >
  , input_types = <[Input Types of Snowflake DataType object]>
  , is_permanent = True
  , name = '<UDF name>'
  , replace = True
  , stage_location = '@<UDF stage name>'
)
```

以下模板定义了在 Snowflake 中使用 Python 创建 UDF 的步骤。它涉及定义 UDF 将使用的首选 Python 函数，并在 Snowflake 中注册它。接下来，指定函数和 UDF 的名称，以及 UDF 文件将上传的 Snowflake 阶段。最后，导入 UDF 返回值的所需 Snowpark `DataType` 对象，并确定其特定对象。这不是我们唯一可以遵循的模板——我们还可以利用装饰器来完成这项工作。但对于初学者来说，这可以非常有帮助，以便模板化和组织 UDFS、UDTFs 和存储过程。

此外，还需要导入和确定 UDF 输入参数所需的任何必要的 Snowpark `DataType` 对象。模板还允许您在 UDF 中包含额外的包和导入。我们还可以指定 UDF 是否应该是临时的，以及是否应该覆盖具有相同名称的现有 UDF：

```py
def last_name_finder(input_name:str):
    last_name = input_name.split()[1]
    return last_name
from snowflake.snowpark.types \
    import StringType,IntegerType,ArrayType
test = session.udf.register(
    func = last_name_finder
  , return_type = StringType()
  , input_types = [StringType()]
  , is_permanent = True
  , name = 'LAST_NAME_FINDER'
  , replace = True
  , stage_location = '@MY_STAGE'
)
```

这个简单的函数获取输入名称并将其分割，以便返回最后一个名称。UDF 被注册到内部的 `My_Stage` 阶段，并部署到 Snowflake。UDF 可以直接在 SQL 中作为函数调用：

```py
session.sql('''SELECT
    NAME,
    LAST_NAME_FINDER(NAME) AS LAST_NAME
    FROM SAMPLE_EMPLOYEE_DATA
''').show()
```

输出如下：

![图 2.37 – UDF Snowpark 执行](img/B19923_02_37.jpg)

图 2.37 – UDF Snowpark 执行

在此示例中，我们使用`Name`列调用了`LAST_NAME_FINDER`函数，该函数通过分割返回了姓氏。该函数也可以在`DataFrame`函数内部调用，如下所示：

```py
from snowflake.snowpark.functions import col, call_udf
df = session.table("SAMPLE_EMPLOYEE_DATA")
df.with_column(
    "last_name",call_udf("LAST_NAME_FINDER", col("name"))).show()
```

上述代码生成了以下输出：

![图 2.38 – UDF DataFrame 执行](img/B19923_02_38.jpg)

图 2.38 – UDF DataFrame 执行

接下来，让我们看看 UDTF。

### UDTF

表格 UDF，也称为 UDTF，需要在数据批次上执行有状态的操作，并且对每行数据调用一次，就像标量 UDF 一样，但它们可以为每个输入行返回多行输出。UDTF 处理方法包含一个额外的可选参数，该参数有助于为每个分区初始化处理程序并最终处理每个部分。UDTF 是一种类似于 UDF 但具有表格输出的 UDF。因此，它们可以在 Python 工作表和 Snowpark 开发环境中开发。

#### 使用 UDTF

在 Snowpark 中创建 UDTF 与创建 UDF 类似，即在创建 Snowpark 会话之后，可以直接在标准命令中创建 UDTF，该命令可以在 Snowflake 中注册。

以下 Snowpark UDTF 模板提供了使用 Python 在 Snowpark 中创建 UDTF 的基本轮廓。以下代码显示了此模板中的关键元素：

```py
# Define Python class locally
'''
Define main Python class which is
leveraged to process partitions.
Executes in the following order:
- __init__ | Executes once per partition
- process | Executes once per input row within the partition
- end_partition | Executes once per partition
'''
class <name of main Python class> :
    '''
    Optional __init__ method to
    execute logic for a partition
    before breaking out into rows
    '''
    def __init__(self) :
```

此模板在 Snowflake 中创建 UDTF。首先，为 UDTF 定义一个主要的 Python 处理程序类，该类可以利用脚本中的其他函数或从外部源导入。需要注意的是，只能将一个主要的 Python 处理程序类分配给 UDTF。

在此模板中，您需要将`<主 Python 类名>`替换为对您的 UDTF 类有意义的名称。这是您将定义处理 UDTF 内部数据逻辑的主要类。

`__init__`方法被标记为可选，这意味着您可以选择是否将其包含在您的 UDTF 实现中。如果您选择包含它，此方法将在每个分区之前执行一次，然后才会进入单个行。

您可以使用`__init__`方法执行任何针对您的 UDTF 的分区级设置或初始化。例如，您可能用它来初始化变量、打开连接或设置在整个 UDTF 执行过程中将使用的数据结构：

```py
'''
Method to process each input row
within a partition, returning a
tabular value as tuples.
'''
def process(self, <arguments>) :
    '''
    Enter Python code here that
    executes for each input row.
    This likely ends with a set of yield
    clauses that output tuples,
    for example:
    '''
    yield (<field_1_value_1>, <field_2_value_1>, ...)
    yield (<field_1_value_2>, <field_2_value_2>, ...)
    '''
    Alternatively, this may end with
    a single return clause containing
    an iterable of tuples, for example:
    '''
    return [
        (<field_1_value_1>, <field_2_value_1>, ...)
      , (<field_1_value_2>, <field_2_value_2>, ...)
    ]
```

此方法负责处理分区内的每个输入行并生成作为元组的表格数据。在`process`方法内部，您可以编写为分区中的每个输入行执行的定制 Python 代码。此方法的关键部分是使用`yield`语句或`return`语句生成元组作为输出。

在 `yield` 语句方面，您可以针对每个输入行输出一个或多个元组，从而在生成表格结果方面提供灵活性。或者，您可以使用带有元组列表的 `return` 语句来实现相同的结果。本质上，`process` 方法是您 UDTF 的核心逻辑，其中您将每个输入行中的数据操作和转换成表格格式，使其适合进一步处理或分析：

```py
'''
Optional end_partition method to
execute logic for a partition
after processing all input rows
'''
def end_partition(self) :
    # Python code at the partition level
    '''
    This ends with a set of yield
    clauses that output tuples,
    for example:
    '''
    yield (<field_1_value_1>, <field_2_value_1>, ...)
    yield (<field_1_value_2>, <field_2_value_2>, ...)
    '''
    Alternatively, this ends with
    a single return clause containing
    an iterable of tuples, for example:
    '''
    return [
        (<field_1_value_1>, <field_2_value_1>, ...)
      , (<field_1_value_2>, <field_2_value_2>, ...)
    ]
```

此方法用于在处理完该分区的所有输入行后执行特定于分区的逻辑。在 `end_partition` 方法内部，您可以编写自定义的 Python 代码，根据在该分区中处理的数据执行计算或生成结果。此方法还可以用于像 `process` 方法一样产生或返回作为元组的表格数据，但此数据通常代表整个分区的聚合或汇总信息。

您可以选择使用 `yield` 语句输出一个或多个元组，或者可以使用带有元组列表的 `return` 语句来提供分区级别的结果。这允许您执行特定于分区的操作并以结构化格式返回结果。

Snowpark UDTF 模板中的 `end_partition` 方法用于执行分区级别的逻辑，并在处理完该分区内的所有输入行后返回表格数据或特定于该分区的结果。这对于需要整个分区数据的聚合或计算等任务特别有用。

以下代码模板提供了如何在 Snowflake 中注册 UDTF 以及定义 UDTF 的相应选项的详细信息：

```py
from snowflake.snowpark.types import StructType, StructField
from snowflake.snowpark.types \
    import <specific Snowpark DataType objects>
snowpark_session.add_packages(
    '<list of required packages natively available in Snowflake( 
        i.e. included in Anaconda Snowpark channel)>')
snowpark_session.add_import('<path\\to\\local\\directory\\or\\file>')
snowpark_session.udtf.register(
    handler = <name of main Python class>
  , output_schema = StructType(
        <list of StructField objects with specific field \
        name and Snowpark DataType objects>)
  , input_types = <list of input DataType() \
        objects for input parameters>
  , is_permanent = True
  , name = '<UDTF name>'
  , replace = True
  , stage_location = '@<UDTF stage name>'
)
```

Snowflake Snowpark `Session` 对象的 `udtf()` 方法用于在 Snowflake 中创建 UDTF。这个过程涉及几个步骤：确定 UDTF 将使用的 Python 类，指定 Snowflake 中 UDTF 的名称（可以是完全限定名称或与 Snowpark `Session` 对象在同一命名空间中创建），以及提供将上传 UDTF 文件的 Snowflake 阶段名称。

特定的 Snowpark `DataType` 对象被导入以定义 UDTF 的结构。这包括导入用于定义表格结构的对象，如表模式（使用 `StructType`）和表内的字段（使用 `StructField`）。此外，还导入了一个特定的 Snowpark `DataType` 对象，用于传递给 UDTF 并由 UDTF 返回的值。UDTF 的输出模式使用导入的 Snowpark `DataType` 对象定义，将它们嵌入到 `StructField` 和 `StructType` 对象中。此外，还定义了一个特定 Snowpark `DataType` 对象的列表，用于 UDTF 的输入参数。在导入时包括所有这些 `DataType` 对象并确保它们与在 `handler` 类中传递给 `process` 方法的预期参数相匹配至关重要。

模板允许创建一个临时的 UDTF，它仅在特定的 Snowflake Snowpark `Session` 对象中存在。此外，还有一个选项可以覆盖具有相同名称的现有 UDTF；如果设置为 `False` 且已存在 UDTF，则将返回错误。最后，模板简要提到了向 UDTF 添加额外的包和导入，这是可选的，可以使用提供的行来完成。

以下示例说明了如何使用 Snowpark UDTF 来计算 Snowflake 表中数值数据的平均值。这展示了 UDTF 在 Snowpark 中用于自定义数据处理任务的实用应用：

```py
class CalculateAverage:
    def __init__(self) :
        self._values = []
    def process(self, input_measure: int) :
        self._values.append(input_measure)
    def end_partition(self) :
        values_list = self._values
        average = sum(values_list) / len(values_list)
        yield(average ,)
```

`CalculateAverage` Snowpark UDTF 的设计是为了计算 Snowflake 表中数值列的平均值。它是通过累积每个数据分区的输入值，然后在分区结束时计算平均值来实现的。

`process` 方法逐个收集输入值并将它们存储在列表中。当分区结束时（在 `end_partition` 方法中），它通过将所有收集到的值相加并除以值的数量来计算平均值。最后，它将计算出的平均值作为 UDTF 的输出产生。这个 UDTF 简化了在 Snowflake SQL 查询中计算平均值的流程，尤其是在处理大型数据集时：

```py
from snowflake.snowpark.types import StructType, StructField
from snowflake.snowpark.types \
    import FloatType,IntegerType,StringType
output_schema = StructType([
    StructField("Avg_Age", FloatType())
])
session.udtf.register(
    handler = CalculateAverage
  , output_schema = output_schema
  , input_types = [IntegerType()]
  , is_permanent = True
  , name = 'AVERAGE_AGE'
  , replace = True
  , stage_location = '@MY_STAGE'
)
```

在这个例子中，我们创建了一个名为 `Average_Age` 的 UDTF 函数，该函数通过获取年龄作为输入来计算平均年龄。该函数被上传到 `MY_STAGE` 并在 Snowflake 中注册。

该函数可以执行以从样本员工数据中获取每个国家的平均年龄：

```py
session.sql('''
    SELECT
        COUNTRY, Avg_Age
    FROM
        SAMPLE_EMPLOYEE_DATA,
        table(AVERAGE_AGE(AGE) over (partition by COUNTRY))
''').show()
```

这将显示以下输出：

![图 2.39 – UDTF Snowpark 执行](img/B19923_02_39.jpg)

图 2.39 – UDTF Snowpark 执行

输出显示了 UDTF 的执行输出。在下一节中，我们将介绍矢量化 UDF。

### 矢量化 UDF

矢量化 UDF 与标量 UDF 类似，允许您定义接收输入行批次的 Python 函数作为 pandas DataFrame，并返回结果集合作为 pandas 数组或序列。

矢量化 UDF 将数据批次的操作并行化，并在与逐行序列处理相比的行集上提供显著的性能优势。此外，它们减少了使用在 pandas DataFrame 和数组上操作的库的复杂性。

#### 与矢量化 UDF 一起工作

在通过将矢量化 DataFrame 作为输入传递给标准 UDF 建立会话后，之前查看的相同示例可以在 Snowpark 环境中执行：

```py
import pandas as pd
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.types \
    import IntegerType, PandasSeriesType,StringType
@pandas_udf(
    name='column_adder'
  , stage_location = '@MY_STAGE'
  , input_types=[PandasSeriesType(StringType()), \
        PandasSeriesType(StringType())]
  , return_type=PandasSeriesType(StringType())
  , is_permanent=True
  , replace=True)
def column_adder(
    column1: pd.Series, column2: pd.Series) -> pd.Series:
    return column1 + "," + column2
df = session.table("SAMPLE_EMPLOYEE_DATA")
df.withColumn('City_Country', column_adder(col('CITY'), \
    col('COUNTRY'))).show()
```

示例 UDF 返回 `Sample` `Employee` 表中每行的 `CITY_COUNTRY` 列中的城市和国家：

![图 2.40 – Snowpark 中的矢量化 UDF](img/B19923_02_40.jpg)

图 2.40 – Snowpark 中的矢量化 UDF

输出显示了矢量化 UDF 的执行输出。在下一节中，我们将介绍存储过程。

### 存储过程

Python 存储过程是一系列可以参数化和按需执行的代码语句。它们在比 UDF 更不受限制的环境中运行，支持与 Snowflake 对象交互，以及在对表执行 DDL 和 DML 操作。

Snowpark 中的存储过程用于在 Snowflake 数据处理框架中执行任务和流。这些存储过程封装了特定的逻辑或功能，使用户能够无缝地对数据进行各种操作。任务通常与批量处理相关联，涉及在预定时间间隔内对数据集执行预定义的操作或工作流。存储过程使用户能够自动化这些任务，确保数据处理的持续性和效率。

另一方面，流是持续的数据管道，实时捕获数据源的变化。存储过程通过定义如何处理和集成到目标目的地来在管理和处理流中发挥关键作用。使用 Snowpark，用户可以创建存储过程来处理这些流数据场景，包括数据转换、过滤以及将数据加载到 Snowflake 表中。

#### 使用存储过程

可以使用以下模板在 Snowpark 中创建存储过程：

```py
# Define Python function locally
def <Python Function Name>(
    snowpark_session: snowflake.snowpark.Session, <arguments>):
    return <Output>
# Imports Required For Stored Procedure
from snowflake.snowpark.types \
    import <specific Snowpark DataType object>
# Optional: Import additional packages or files
snowpark_session.add_packages(
    'List of native packages in Anaconda Channel')
snowpark_session.add_import('Path to Local File')
# Upload Stored Procedure to Snowflake
snowpark_session.sproc.register(
    func = <Function name to register>
  , return_type = <Return Type of Snowpark DataType object>
  , input_types = <[Input Types of Snowflake DataType object]>
  , is_permanent = True
  , name = '<Stored Procedure name>'
  , replace = True
  , stage_location = '@<Stored Procedure stage name>'
    <optional: , execute_as = 'CALLER'>
)
```

在这里，我们定义了将在存储过程中使用的 Python 主函数以及一个名为`snowpark_session`的附加参数，它允许我们与 Snowflake 对象交互。接下来，我们使用`sproc.register()`方法创建存储过程，指定 Python 函数、存储过程名称和文件上传的 Snowflake 阶段。最后，我们导入存储过程的返回值和输入参数的特定 Snowpark `DataType`对象。

`snowflake_session`参数是隐含理解的，不包括在输入参数中。可选行允许额外的包和导入。在这里，我们可以确定存储过程是否为临时。我们还可以决定是否覆盖同名现有存储过程，并指定它将以调用者或所有者的身份执行：

```py
def subset_table(snowpark_session:Session):
    df = snowpark_session.table(
        'SAMPLE_EMPLOYEE_DATA').select("NAME","AGE")
    return df.collect()
from snowflake.snowpark.types import StringType
session.add_packages('snowflake-snowpark-python')
session.sproc.register(
    func = subset_table
  , return_type = StringType()
  , input_types = []
  , is_permanent = True
  , name = 'SPROC_SUBSET_TABLE'
  , replace = True
  , stage_location = '@MY_STAGE'
)
```

该存储过程从`Employee Data`表中返回列名和年龄。它注册为`SPROC_SUBSET_TABLE`并通过`My_Stage`上传：

```py
session.sql(''' CALL SPROC_SUBSET_TABLE()''').show()
```

下面是输出结果：

![图 2.41 – 存储过程执行](img/B19923_02_41.jpg)

图 2.41 – 存储过程执行

可以通过运行`CALL`命令来执行存储过程。

### UDF 与存储过程之间的区别

UDF 和存储过程在功能和用途方面有显著差异。以下图显示了 UDF 和存储过程的基本区别以及它们的使用目的：

![图 2.42 – UDF 与存储过程对比](img/B19923_02_42.jpg)

图 2.42 – UDF 与存储过程对比

以下表格显示了 UDF 和存储过程基于其属性之间的差异和相似性：

|  | **UDF** | **存储过程** |
| --- | --- | --- |
| 目的 | 执行计算并返回结果。UDF 需要返回一个值。 | 通过执行 SQL 语句执行复杂操作。它们不需要显式返回一个值。 |
| 使用 | 当需要将逻辑作为返回值的 SQL 语句的一部分调用时，可以使用 UDF。 | 当需要执行数据库操作或管理任务时。 |
| 输出 | UDF 总是需要返回一个结果。 | 存储过程不需要返回结果。 |
| 上下文 | UDF 的返回值可以直接在 SQL 上下文中访问。 | 存储过程的返回值在 SQL 上下文中不可访问。 |
| 执行 | UDF 可以在另一个 SQL 语句的上下文中调用。此外，可以在单个 SQL 语句中调用多个 UDF。 | 存储过程是独立调用的。因此，SQL 语句中只调用一个存储过程。 |
| 安全性 | UDF 不能直接访问数据库或在其上执行操作。 | 存储过程可以访问数据库并在其上执行数据操作。 |

表 2.1 – UDF 和存储过程的比较

存储过程和 UDF 都可以一起使用，以扩展 Python 在 Snowflake 中的执行功能。

# 为 Snowpark 建立项目结构

为了帮助在 Python 中开发 Snowpark 并使其易于创建 Snowpark 项目，Snowflake 已发布了 Python Snowpark 项目模板。这些模板包含您开发、测试和部署 Snowpark 所需的一切——它们提供了开发 UDF 和存储过程所需的所有样板代码，以及单元测试，甚至还有用于 CI/CD 的 GitHub Actions 工作流程文件。

该项目模板已作为开源发布在 GitHub 上，这使得开发者可以轻松克隆和使用项目。要克隆项目，请按照以下步骤操作：

1.  从[`github.com/Snowflake-Labs/snowpark-python-template`](https://github.com/Snowflake-Labs/snowpark-python-template)下载文件或克隆仓库。可以使用 GitHub CLI 从模板创建一个新的 GitHub 仓库，如下所示：

    ```py
    gh repo create <new-repo-name> --template="Snowflake-Labs/snowpark-python-template"
    ```

    需要指定新仓库的名称。该仓库将与 Snowpark 项目模板类似。

1.  设置以下环境变量，以便您可以配置必要的 Snowflake 详细信息：

    ```py
    SNOWSQL_ACCOUNT=<replace with your account identifier>
    SNOWSQL_USER=<replace with your username>
    SNOWSQL_PWD=<replace with your password>
    SNOWSQL_DATABASE=<replace with your database>
    SNOWSQL_SCHEMA=<replace with your schema>
    SNOWSQL_WAREHOUSE=<replace with your warehouse>
    ```

    这些环境变量是连接到 Snowflake 环境以及 Snowpark 建立会话所必需的。

1.  创建一个 Anaconda 虚拟环境，并从**environment.yml**文件安装依赖项：

    ```py
    conda env create -f environment.yml
    snowpark that can be used for development.
    ```

1.  您可以通过执行**app.py**存储过程来测试连接并检查环境是否已设置。导航到项目文件夹并运行以下命令：

    ```py
    python src/procs/app.py
    ```

    这将生成一个名为**Hello World**的输出，用于建立与 Snowflake 的连接。

Snowpark 项目支持函数和存储过程。项目结构包括用于存储过程的`procs`目录，用于用户定义函数（UDFs）的`udf`目录，以及用于 UDFs 和存储过程之间共享的实用方法和类的`util`目录。

`test`目录包含可以通过`pytest`测试的测试用例。还有一个可以通过 GitHub Actions 部署的 GitHub 工作流程。我们将在接下来的章节中详细介绍这一点。

# 摘要

Snowpark 非常灵活，支持复杂的发展模式。在本章中，我们学习了如何配置 Snowpark 开发环境以及不同的 Snowpark 对象，例如会话、UDFs 和存储过程，以及如何使用它们。我们还学习了在查看一些我们可以用来开始开发的示例代码之前，如何在本地和 Python 工作表中设置 Snowpark 开发项目。

在下一章中，我们将介绍如何使用 Snowpark 进行数据处理，以及如何摄取、准备和分析数据。
