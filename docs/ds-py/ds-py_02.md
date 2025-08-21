# 第二章 Python 和 Jupyter Notebooks 助力你的数据分析

> “最好的代码是你不必写的那一行！”

– *未知*

在上一章中，我从开发者的角度，基于真实经验探讨了数据科学，并讨论了成功部署企业所需的三个战略支柱：数据、服务和工具。我还讨论了一个观点，即数据科学不仅仅是数据科学家的专属领域，它实际上是一个团队合作的过程，其中开发者扮演着特殊的角色。

在本章中，我将介绍一个解决方案——基于 Jupyter Notebooks、Python 和 PixieDust 开源库——它专注于三个简单的目标：

+   通过降低非数据科学家的准入门槛，实现数据科学的普及

+   增加开发者与数据科学家之间的合作

+   让数据科学分析更易于操作化

### 注意

该解决方案仅聚焦于工具层面，而不涉及数据和服务，尽管我们将在第六章 *数据分析研究：使用 TensorFlow 的 AI 与图像识别*中讨论一些内容，数据和服务应独立实现。

# 为什么选择 Python？

像许多开发者一样，在构建数据密集型项目时，使用 Python 并不是我的首选。说实话，经过多年的 Java 工作经验后，最初 Scala 对我来说更具吸引力，尽管学习曲线相当陡峭。Scala 是一种非常强大的语言，它优雅地结合了面向对象编程和函数式编程，而这些在 Java 中是极为缺乏的（至少直到 Java 8 开始引入 Lambda 表达式）。

Scala 还提供了非常简洁的语法，这转化为更少的代码行、更高的生产力，最终也减少了错误。这在你大部分工作都涉及数据操作时非常有用。喜欢 Scala 的另一个原因是，当使用像 Apache Spark 这样的“大数据”框架时，它的 API 支持更好，而这些框架本身就是用 Scala 编写的。还有很多其他理由使得 Scala 更受青睐，比如它是一个强类型系统，能够与 Java 互操作，拥有在线文档，并且性能高。

因此，对于像我这样的开发者，刚开始涉足数据科学时，Scala 似乎是一个更自然的选择，但实际上，剧透一下，我们最终选择了 Python。这一选择有多个原因：

+   Python 作为一种语言也有很多优点。它是一种动态编程语言，具有与 Scala 类似的优势，比如函数式编程和简洁的语法等。

+   在过去几年里，Python 在数据科学家中迅速崛起，超过了长期竞争对手 R，成为数据科学领域的首选语言，这一点通过在 Google Trends 上搜索`Python Data Science`、`Python Machine Learning`、`R Data Science`和`R Machine Learning`可以看到：![为什么选择 Python？](img/B09699_02_01.jpg)

    2017 年的兴趣趋势

在良性循环中，Python 日益增长的流行度推动了一个庞大而不断扩展的生态系统，涵盖了各种各样的库，这些库可以通过 pip Python 包安装器轻松导入到项目中。数据科学家现在可以使用许多强大的开源 Python 库进行数据处理、数据可视化、统计学、数学、机器学习和自然语言处理。

即使是初学者，也可以通过流行的 scikit-learn 包 ([`scikit-learn.org`](http://scikit-learn.org)) 快速构建机器学习分类器，而不需要成为机器学习专家，或者使用 Matplotlib ([`matplotlib.org`](https://matplotlib.org)) 或 Bokeh ([`bokeh.pydata.org`](https://bokeh.pydata.org)) 快速绘制丰富的图表。

此外，Python 也成为了开发者的顶级语言之一，正如 IEEE Spectrum 2017 年调查所示 ([`spectrum.ieee.org/computing/software/the-2017-top-programming-languages`](https://spectrum.ieee.org/computing/software/the-2017-top-programming-languages)):

![为什么选择 Python？](img/B09699_02_02.jpg)

按编程语言划分的使用统计

这一趋势在 GitHub 上也得到了确认，Python 现在在所有仓库的总数中排名第三，仅次于 Java 和 JavaScript：

![为什么选择 Python？](img/B09699_02_03.jpg)

按编程语言划分的 GitHub 仓库统计

上面的图表展示了一些有趣的统计数据，说明了 Python 开发者社区的活跃程度。在 GitHub 上，Python 相关的活跃仓库是第三大，拥有健康的总代码推送和每个仓库的开启问题数量。

Python 还在网页开发中变得无处不在，许多知名网站都采用了如 Django ([`www.djangoproject.com`](https://www.djangoproject.com))、Tornado ([`www.tornadoweb.org`](http://www.tornadoweb.org))和 TurboGears ([`turbogears.org`](http://turbogears.org))等网页开发框架。最近，Python 也开始渗透到云服务领域，所有主要云服务提供商都将其作为某种程度的服务纳入其产品中。

显然，Python 在数据科学领域有着光明的前景，特别是当与强大的工具如 Jupyter Notebooks 一起使用时，Jupyter Notebooks 在数据科学家社区中已经变得非常流行。Notebook 的价值主张在于，它们非常容易创建，非常适合快速运行实验。此外，Notebook 支持多种高保真度的序列化格式，可以捕捉指令、代码和结果，然后很容易与团队中的其他数据科学家分享，或作为开源项目供所有人使用。例如，我们看到越来越多的 Jupyter Notebooks 在 GitHub 上被分享，数量已超过 250 万，并且还在不断增加。

下图显示了在 GitHub 上搜索所有扩展名为 `.ipynb` 的文件的结果，这是最流行的 Jupyter Notebooks 序列化格式（JSON 格式）：

![为什么选择 Python？](img/B09699_02_04.jpg)

在 GitHub 上搜索 Jupyter Notebooks 的结果

这很好，但 Jupyter Notebooks 常常被认为仅仅是数据科学家的工具。我们将在接下来的章节中看到，它们可以做得更多，而且可以帮助所有类型的团队解决数据问题。例如，它们可以帮助业务分析师快速加载和可视化数据集，使开发者能够直接在 Notebook 中与数据科学家合作，利用他们的分析并构建强大的仪表盘，或允许 DevOps 将这些仪表盘轻松地部署到可扩展的企业级微服务中，这些微服务可以作为独立的 Web 应用程序运行，或者作为可嵌入的组件。正是基于将数据科学工具带给非数据科学家的愿景，才创建了 PixieDust 开源项目。

# 介绍 PixieDust

### 小贴士

**有趣的事实**

我经常被问到我如何想出“PixieDust”这个名字，对此我回答说，我只是想让 Notebook 对非数据科学家来说变得简单，就像魔法一样。

PixieDust ([`github.com/ibm-watson-data-lab/pixiedust`](https://github.com/ibm-watson-data-lab/pixiedust)) 是一个开源项目，主要由三个组件组成，旨在解决本章开头提到的三个目标：

+   一个为 Jupyter Notebooks 提供的辅助 Python 库，提供简单的 API 来将数据从各种来源加载到流行的框架中，如 pandas 和 Apache Spark DataFrame，然后交互式地可视化和探索数据集。

+   一种基于 Python 的简单编程模型，使开发者能够通过创建强大的仪表盘（称为 PixieApps）将分析“产品化”到 Notebook 中。正如我们将在接下来的章节中看到的，PixieApps 与传统的 **BI**（即 **商业智能**）仪表盘不同，因为开发者可以直接使用 HTML 和 CSS 创建任意复杂的布局。此外，他们还可以在其业务逻辑中嵌入对 Notebook 中创建的任何变量、类或函数的访问。

+   一种名为 PixieGateway 的安全微服务 Web 服务器，可以将 PixieApps 作为独立的 Web 应用程序运行，或作为可以嵌入任何网站的组件。PixieApps 可以通过 Jupyter Notebook 使用图形向导轻松部署，并且无需任何代码更改。此外，PixieGateway 支持将任何由 PixieDust 创建的图表作为可嵌入的网页共享，使数据科学家可以轻松地将结果传达给 Notebook 外部的受众。

需要注意的是，PixieDust `display()` API 主要支持两种流行的数据处理框架：

+   **pandas** ([`pandas.pydata.org`](https://pandas.pydata.org)): 迄今为止最受欢迎的 Python 数据分析包，pandas 提供了两种主要的数据结构：DataFrame 用于处理类似二维表格的数据集，Series 用于处理一维列状数据集。

    ### 注意

    目前，PixieDust `display()` 只支持 pandas DataFrame。

+   **Apache Spark DataFrame** ([`spark.apache.org/docs/latest/sql-programming-guide.html`](https://spark.apache.org/docs/latest/sql-programming-guide.html)): 这是一个高层数据结构，用于操作 Spark 集群中分布式数据集。Spark DataFrame 建立在低层的 **RDD**（即 **Resilient Distributed Dataset**）之上，并且增加了支持 SQL 查询的功能。

另一种 PixieDust `display()` 支持的较少使用的格式是 JSON 对象数组。在这种情况下，PixieDust 会使用这些值来构建行，并将键作为列，例如如下所示：

```py
my_data = [
{"name": "Joe", "age": 24},
{"name": "Harry", "age": 35},
{"name": "Liz", "age": 18},
...
]

```

此外，PixieDust 在数据处理和渲染层面都具有高度的可扩展性。例如，你可以向可视化框架添加新的数据类型，或者如果你特别喜欢某个绘图库，你可以轻松地将其添加到 PixieDust 支持的渲染器列表中（更多细节请参见接下来的章节）。

你还会发现，PixieDust 包含了一些与 Apache Spark 相关的附加工具，例如以下内容：

+   **PackageManager**：这使你可以在 Python Notebook 中安装 Spark 包。

+   **Scala 桥接**：这使你可以在 Python Notebook 中直接使用 Scala，通过 `%%scala` 魔法命令。变量会自动从 Python 转移到 Scala，反之亦然。

+   **Spark 作业进度监控器**：通过在单元格输出中显示进度条，跟踪任何 Spark 作业的状态。

在我们深入了解 PixieDust 的三个组件之前，建议先获取一个 Jupyter Notebook，可以通过注册云端托管解决方案（例如，Watson Studio，网址：[`datascience.ibm.com`](https://datascience.ibm.com)）或在本地机器上安装开发版来实现。

### 注意

你可以按照以下说明在本地安装 Notebook 服务器：[`jupyter.readthedocs.io/en/latest/install.html`](http://jupyter.readthedocs.io/en/latest/install.html)。

要在本地启动 Notebook 服务器，只需从终端运行以下命令：

```py
jupyter notebook --notebook-dir=<<directory path where notebooks are stored>>

```

Notebook 主页将自动在浏览器中打开。有许多配置选项可以控制 Notebook 服务器的启动方式。这些选项可以添加到命令行或持久化到 Notebook 配置文件中。如果你想尝试所有可能的配置选项，可以使用`--generate-config`选项生成一个配置文件，如下所示：

```py
jupyter notebook --generate-config

```

这将生成以下 Python 文件`<home_directory>/.jupyter/jupyter_notebook_config.py`，其中包含一组已禁用的自动文档选项。例如，如果你不希望 Jupyter Notebook 启动时自动打开浏览器，找到包含`sc.NotebookApp.open_browser`变量的行，取消注释并将其设置为`False`：

```py
## Whether to open in a browser after starting. The specific browser used is
#  platform dependent and determined by the python standard library 'web browser'
#  module, unless it is overridden using the --browser (NotebookApp.browser)
#  configuration option.
c.NotebookApp.open_browser = False

```

在做完更改后，只需保存`jupyter_notebook_config.py`文件并重新启动 Notebook 服务器。

下一步是使用`pip`工具安装 PixieDust 库：

1.  从 Notebook 本身，输入以下命令在单元格中执行：

    ```py
    !pip install pixiedust

    ```

    ### 注意

    **注意**：感叹号语法是 Jupyter Notebook 特有的，表示后续的命令将作为系统命令执行。例如，你可以使用`!ls`列出当前工作目录下的所有文件和目录。

1.  使用**单元格** | **运行单元格**菜单或工具栏上的**运行**图标来运行单元格。你也可以使用以下键盘快捷键来运行单元格：

    +   *Ctrl* + *Enter*：运行并保持当前单元格选中

    +   *Shift* + *Enter*：运行并选中下一个单元格

    +   *Alt* + *Enter*：运行并在下方创建一个新的空单元格

1.  重新启动内核以确保`pixiedust`库已正确加载到内核中。

以下截图显示了首次安装`pixiedust`后的结果：

![介绍 PixieDust](img/B09699_02_05.jpg)

在 Jupyter Notebook 上安装 PixieDust 库

### 提示

我强烈推荐使用 Anaconda（[`anaconda.org`](https://anaconda.org)），它提供了优秀的 Python 包管理功能。如果你像我一样喜欢尝试不同版本的 Python 和库依赖，我建议你使用 Anaconda 虚拟环境。

它们是轻量级的 Python 沙箱，创建和激活非常简单（请参见[`conda.io/docs/user-guide/tasks/manage-environments.html`](https://conda.io/docs/user-guide/tasks/manage-environments.html)）：

+   创建新环境：`conda create --name env_name`

+   列出所有环境：`conda env list`

+   激活环境：`source activate env_name`

我还推荐你可以选择性地熟悉源代码，源代码可以在[`github.com/ibm-watson-data-lab/pixiedust`](https://github.com/ibm-watson-data-lab/pixiedust)和[`github.com/ibm-watson-data-lab/pixiegateway`](https://github.com/ibm-watson-data-lab/pixiegateway)找到。

我们现在准备好在下一节中探索 PixieDust API，从`sampleData()`开始。

# SampleData – 一个简单的数据加载 API

将数据加载到 Notebook 中是数据科学家最常做的重复性任务之一，但根据使用的框架或数据源，编写代码可能会很困难且耗时。

让我们以一个具体的例子来说明，尝试从一个开放数据网站（例如 [`data.cityofnewyork.us`](https://data.cityofnewyork.us)）加载 CSV 文件到 pandas 和 Apache Spark DataFrame。

### 注意

**注意**：接下来的所有代码都假定在 Jupyter Notebook 中运行。

对于 pandas，代码相当简单，因为它提供了一个直接从 URL 加载的 API：

```py
import pandas
data_url = "https://data.cityofnewyork.us/api/views/e98g-f8hy/rows.csv?accessType=DOWNLOAD"
building_df = pandas.read_csv(data_url)
building_df
```

最后一条语句，调用`building_df,`，将在输出单元中打印其内容。由于 Jupyter 会将单元格最后一条调用变量的语句作为打印指令，因此可以在不显式使用 `print` 的情况下实现此操作：

![SampleData – 一个简单的数据加载 API](img/B09699_02_06.jpg)

pandas DataFrame 的默认输出

然而，对于 Apache Spark，我们首先需要将数据下载到文件中，然后使用 Spark CSV 连接器将其加载到 DataFrame 中：

```py
#Spark CSV Loading
from pyspark.sql import SparkSession
try:
    from urllib import urlretrieve
except ImportError:
    #urlretrieve package has been refactored in Python 3
    from urllib.request import urlretrieve

data_url = "https://data.cityofnewyork.us/api/views/e98g-f8hy/rows.csv?accessType=DOWNLOAD"
urlretrieve (data_url, "building.csv")

spark = SparkSession.builder.getOrCreate()
building_df = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', True)\
  .load("building.csv")
building_df
```

输出略有不同，因为 `building_df` 现在是一个 Spark DataFrame：

![SampleData – 一个简单的数据加载 API](img/B09699_02_07.jpg)

Spark DataFrame 的默认输出

尽管这段代码并不复杂，但每次都需要重复执行，并且很可能需要花时间做 Google 搜索来记住正确的语法。数据也可能采用不同的格式，例如 JSON，这需要调用不同的 API，无论是对于 pandas 还是 Spark。数据可能也不是很规范，可能在 CSV 文件中有错误行，或者 JSON 语法不正确。所有这些问题不幸的是并不罕见，并且符合数据科学的 80/20 法则，该法则表明数据科学家平均花费 80% 的时间来获取、清洗和加载数据，仅有 20% 的时间用于实际分析。

PixieDust 提供了一个简单的 `sampleData` API 来帮助改善这种情况。当不带参数调用时，它会显示一个预先策划的、准备好进行分析的数据集列表：

```py
import pixiedust
pixiedust.sampleData()
```

结果如下所示：

![SampleData – 一个简单的数据加载 API](img/B09699_02_08.jpg)

PixieDust 内置数据集

预构建的策划数据集列表可以根据组织的需要进行自定义，这是朝着我们*数据*支柱迈出的好步骤，如上一章所述。

用户可以简单地再次调用 `sampleData` API，传入预构建数据集的 ID，如果 Jupyter 内核中有 Spark 框架可用，则会获得一个 Spark DataFrame；如果不可用，则会回退为 pandas DataFrame。

在下面的示例中，我们在与 Spark 连接的 Notebook 上调用 `sampleData()`。我们还调用 `enableSparkJobProgressMonitor()` 来显示涉及操作的 Spark 作业的实时信息。

### 注意

**注意**：Spark 作业是在 Spark 集群中的特定节点上运行的进程，处理特定子集的数据。对于从数据源加载大量数据的情况，每个 Spark 作业会分配一个特定的子集来处理（实际大小取决于集群中的节点数和总体数据的大小），并与其他作业并行运行。

在一个单独的单元格中，我们运行以下代码来启用 Spark 作业进度监视器：

```py
pixiedust.enableSparkJobProgressMonitor()
```

结果如下：

```py
Successfully enabled Spark Job Progress Monitor
```

接着，我们调用 `sampleData` 来加载 `cars` 数据集：

```py
cars = pixiedust.sampleData(1)
```

结果如下所示：

![SampleData – a simple API for loading data](img/B09699_02_09.jpg)

使用 PixieDust sampleData API 加载内置数据集

用户还可以传入指向可下载文件的任意 URL；PixieDust 目前支持 JSON 和 CSV 文件。在这种情况下，PixieDust 会自动下载该文件，缓存到临时区域，检测格式，并根据 Notebook 中是否可用 Spark 将其加载到 Spark 或 pandas DataFrame 中。请注意，即使 Spark 可用，用户也可以通过使用 `forcePandas` 关键字参数强制加载到 pandas 中：

```py
import pixiedust
data_url = "https://data.cityofnewyork.us/api/views/e98g-f8hy/rows.csv?accessType=DOWNLOAD"
building_dataframe = pixiedust.sampleData(data_url, forcePandas=True)
```

结果如下：

```py
Downloading 'https://data.cityofnewyork.us/api/views/e98g-f8hy/rows.csv?accessType=DOWNLOAD' from https://data.cityofnewyork.us/api/views/e98g-f8hy/rows.csv?accessType=DOWNLOAD
Downloaded 13672351 bytes
Creating pandas DataFrame for 'https://data.cityofnewyork.us/api/views/e98g-f8hy/rows.csv?accessType=DOWNLOAD'. Please wait...
Loading file using 'pandas'
Successfully created pandas DataFrame for 'https://data.cityofnewyork.us/api/views/e98g-f8hy/rows.csv?accessType=DOWNLOAD'
```

`sampleData()` API 足够智能，能够识别指向 ZIP 和 GZ 类型压缩文件的 URL。在这种情况下，它会自动解压原始二进制数据，并加载归档中包含的文件。对于 ZIP 文件，它会查看归档中的第一个文件，而对于 GZ 文件，它会简单地解压内容，因为 GZ 文件不是归档文件，不包含多个文件。`sampleData()` API 然后会从解压后的文件加载 DataFrame。

例如，我们可以直接从伦敦开放数据网站提供的 ZIP 文件加载区信息，并使用 `display()` API 将结果显示为饼图，具体如下：

```py
import pixiedust
london_info = pixiedust.sampleData("https://files.datapress.com/london/dataset/london-borough-profiles/2015-09-24T15:50:01/London-borough-profiles.zip")
```

结果如下（假设你的 Notebook 已连接到 Spark，否则将加载一个 pandas DataFrame）：

```py
Downloading 'https://files.datapress.com/london/dataset/london-borough-profiles/2015-09-24T15:50:01/London-borough-profiles.zip' from https://files.datapress.com/london/dataset/london-borough-profiles/2015-09-24T15:50:01/London-borough-profiles.zip
Extracting first item in zip file...
File extracted: london-borough-profiles.csv
Downloaded 948147 bytes
Creating pySpark DataFrame for 'https://files.datapress.com/london/dataset/london-borough-profiles/2015-09-24T15:50:01/London-borough-profiles.zip'. Please wait...
Loading file using 'com.databricks.spark.csv'
Successfully created pySpark DataFrame for 'https://files.datapress.com/london/dataset/london-borough-profiles/2015-09-24T15:50:01/London-borough-profiles.zip'
```

然后我们可以在 `london_info` DataFrame 上调用 `display()`，如图所示：

```py
display(london_info)
```

我们在图表菜单中选择 **饼图**，在 **选项** 对话框中，将 `Area name` 列拖放到 **键** 区域，将 `Crime rates per thousand population 2014/15` 拖放到 **值** 区域，如下图所示：

![SampleData – a simple API for loading data](img/B09699_02_10.jpg)

用于可视化 london_info DataFrame 的图表选项

在 **选项** 对话框中点击 **确定** 按钮后，我们得到以下结果：

![SampleData – a simple API for loading data](img/B09699_02_11.jpg)

从指向压缩文件的 URL 创建饼图

许多时候，您可能找到了一个很棒的数据集，但文件中包含错误，或者对您重要的数据格式不正确，或者被埋在一些非结构化文本中需要提取到自己的列中。这个过程也被称为**数据整理**，可能非常耗时。在接下来的部分中，我们将看到一个名为`pixiedust_rosie`的 PixieDust 扩展，它提供了一个`wrangle_data`方法，可以帮助处理这个过程。

# 使用 pixiedust_rosie 整理数据

在受控实验中工作，大多数情况下与在真实世界中工作并不相同。我的意思是，在开发过程中，我们通常会选择（或者我应该说制造）一个设计良好、符合模式规范、没有数据缺失等特性的样本数据集。目标是专注于验证假设并构建算法，而不是数据清洗，这可能非常痛苦和耗时。然而，在开发过程的早期尽可能接近真实数据确实有不可否认的好处。为了帮助完成这项任务，我与两位 IBM 同事 Jamie Jennings 和 Terry Antony 合作，他们志愿构建了一个名为`pixiedust_rosie`的 PixieDust 扩展。

这个 Python 包实现了一个简单的`wrangle_data()`方法来自动清理原始数据。`pixiedust_rosie`包目前支持 CSV 和 JSON 格式，但未来将添加更多格式。底层数据处理引擎使用了**Rosie 模式语言（RPL）**开源组件，这是一个专为开发人员设计、更高效、可扩展到大数据的正则表达式引擎。您可以在这里找到更多关于 Rosie 的信息：[`rosie-lang.org`](http://rosie-lang.org)。

要开始使用，您需要使用以下命令安装`pixiedust_rosie`包：

```py
!pip install pixiedust_rosie

```

`pixiedust_rosie`包依赖于`pixiedust`和`rosie`，如果系统上尚未安装，将自动下载。

`wrangle_data()`方法与`sampleData()` API 非常相似。如果不带参数调用，它将显示预先筛选数据集的列表，如下所示：

```py
import pixiedust_rosie
pixiedust_rosie.wrangle_data()
```

这将产生以下结果：

![使用 pixiedust_rosie 整理数据](img/B09699_02_12.jpg)

预先筛选的数据集列表可用于`wrangle_data()`。

您还可以通过预先筛选的数据集的 ID 或 URL 链接来调用它，例如：

```py
url = "https://github.com/ibm-watson-data-lab/pixiedust_rosie/raw/master/sample-data/Healthcare_Cost_and_Utilization_Project__HCUP__-_National_Inpatient_Sample.csv"
pixiedust_rosie.wrangle_data(url)
```

在上面的代码中，我们在由`url`变量引用的 CSV 文件上调用`wrangle_data()`。该函数首先下载文件到本地文件系统，并对数据的一个子集执行自动化数据分类，以推断数据架构。随后启动一个架构编辑器 PixieApp，提供一组向导屏幕，允许用户配置架构。例如，用户将能够删除和重命名列，更重要的是，通过提供 Rosie 模式将现有列解构为新列。

工作流在以下图示中进行说明：

![使用 pixiedust_rosie 处理数据](img/B09699_02_13.jpg)

`wrangle_data()`工作流

`wrangle_data()`向导的第一个屏幕显示了 Rosie 数据分类器推断出的架构，如下图所示：

![使用 pixiedust_rosie 处理数据](img/B09699_02_14.jpg)

`wrangle_data()`架构编辑器

上面的架构小部件显示了列名、`Rosie 类型`（特定于 Rosie 的高级类型表示）和`列类型`（映射到支持的 pandas 类型）。每一行还包含三个操作按钮：

+   **删除列**：这将从架构中删除列。此列将不会出现在最终的 pandas DataFrame 中。

+   **重命名列**：这将改变列的名称。

+   **转换列**：通过将列解构为新列来转换列。

用户随时可以预览数据（如上面的 SampleData 小部件所示），以验证架构配置是否按预期运行。

当用户点击转换列按钮时，显示一个新屏幕，允许用户指定用于构建新列的模式。在某些情况下，数据分类器能够自动检测这些模式，在这种情况下，会添加一个按钮询问用户是否应用这些建议。

下图显示了带有自动化建议的**转换选定列**屏幕：

![使用 pixiedust_rosie 处理数据](img/B09699_02_15.jpg)

转换列屏幕

此屏幕显示四个小部件，包含以下信息：

+   Rosie 模式输入框是您可以输入自定义 Rosie 模式的位置，用于表示此列的数据。然后，您使用**提取变量**按钮告诉架构编辑器应该将模式中的哪一部分提取到新列中（更多细节稍后解释）。

+   有一个帮助小部件，提供指向 RPL 文档的链接。

+   显示当前列的数据预览。

+   显示应用了 Rosie 模式的数据预览。

当用户点击**提取变量**按钮时，部件将更新为如下：

![使用 pixiedust_rosie 处理数据](img/B09699_02_16.jpg)

将 Rosie 变量提取到列中

此时，用户可以选择编辑定义，然后点击**创建列**按钮，将新列添加到架构中。**新列示例**小部件随后会更新，显示数据预览。如果模式定义包含错误语法，则此小部件会显示错误：

![使用 pixiedust_rosie 清理数据](img/B09699_02_17.jpg)

应用模式定义后的新列预览

当用户点击**提交列**按钮时，主架构编辑器界面会再次显示，新增的列会被添加进来，如下图所示：

![使用 pixiedust_rosie 清理数据](img/B09699_02_18.jpg)

带有新列的架构编辑器

最后一步是点击**完成**按钮，将架构定义应用到原始文件中，并创建一个 pandas 数据框，该数据框将作为变量在笔记本中使用。此时，用户将看到一个对话框，其中包含一个可以编辑的默认变量名，如下图所示：

![使用 pixiedust_rosie 清理数据](img/B09699_02_19.jpg)

编辑结果 Pandas 数据框的变量名称

点击**完成**按钮后，`pixiedust_rosie` 会遍历整个数据集，应用架构定义。完成后，它会在当前单元格下方创建一个新单元格，其中包含生成的代码，调用`display()` API 来显示新生成的 pandas 数据框，如下所示：

```py
#Code generated by pixiedust_rosie
display(wrangled_df)
```

运行前面的单元格将让你探索和可视化新数据集。

我们在本节中探讨的`wrangle_data()`功能是帮助数据科学家减少数据清理时间、更多时间进行数据分析的第一步。在下一节中，我们将讨论如何帮助数据科学家进行数据探索和可视化。

# Display – 一个简单的交互式数据可视化 API

数据可视化是数据科学中另一个非常重要的任务，它在探索和形成假设中是不可或缺的。幸运的是，Python 生态系统拥有许多强大的库，专门用于数据可视化，例如以下这些流行的例子：

+   Matplotlib: [`matplotlib.org`](http://matplotlib.org)

+   Seaborn: [`seaborn.pydata.org`](https://seaborn.pydata.org)

+   Bokeh: [`bokeh.pydata.org`](http://bokeh.pydata.org)

+   Brunel: [`brunelvis.org`](https://brunelvis.org)

然而，类似于数据加载和清理，在笔记本中使用这些库可能会很困难且耗时。每个库都有自己独特的编程模型，API 学习和使用起来并不总是容易，特别是如果你不是一个有经验的开发者。另一个问题是，这些库没有提供一个高层次的接口来与常用的数据处理框架（如 pandas（也许 Matplotlib 除外）或 Apache Spark）进行对接，因此，在绘制数据之前，需要进行大量的数据准备工作。

为了帮助解决这个问题，PixieDust 提供了一个简单的`display()` API，使得 Jupyter Notebook 用户可以通过交互式图形界面绘制数据，无需编写任何代码。这个 API 并不直接创建图表，而是通过调用渲染器的 API 来处理数据准备工作，根据用户的选择委托给相应的渲染器。

`display()` API 支持多种数据结构（如 pandas、Spark 和 JSON）以及多种渲染器（如 Matplotlib、Seaborn、Bokeh 和 Brunel）。

举个例子，让我们使用内置的汽车性能数据集，开始通过调用`display()` API 来可视化数据：

```py
import pixiedust
cars = pixiedust.sampleData(1, forcePandas=True) #car performance data
display(cars)
```

当命令首次在单元格中调用时，系统会显示一个表格视图，随着用户在菜单中的导航，所选的选项会以 JSON 格式存储在单元格的元数据中，确保下次运行该单元格时可以重新使用。所有可视化的输出布局遵循相同的模式：

+   有一个可扩展的顶级菜单，用于在不同的图表之间切换。

+   有一个下载菜单，允许将文件下载到本地计算机。

+   有一个过滤切换按钮，允许用户通过过滤数据来精炼他们的探索。我们将在*过滤*部分讨论过滤功能。

+   有一个展开/折叠 Pixiedust 输出按钮，用于折叠或展开输出内容。

+   有一个**选项**按钮，点击后会弹出一个对话框，包含当前可视化的特定配置。

+   有一个**分享**按钮，可以让你将可视化结果发布到网络上。

    ### 注

    **注**：此按钮仅在你部署了 PixieGateway 后可用，详细内容将在第四章，*将数据分析发布到网络 - PixieApp 工具*中讨论。

+   在可视化的右侧有一组上下文相关的选项。

+   有主可视化区域。

![显示 – 一个简单的交互式数据可视化 API](img/B09699_02_20.jpg)

表格渲染器的可视化输出布局

要开始创建图表，首先在菜单中选择合适的类型。PixieDust 默认支持六种类型的图表：**柱状图**、**折线图**、**散点图**、**饼图**、**地图**和**直方图**。正如我们在第五章，*Python 和 PixieDust 最佳实践及高级概念*中看到的，PixieDust 还提供 API，允许你通过添加新的菜单项或为现有菜单添加选项来自定义这些菜单：

![显示 – 一个简单的交互式数据可视化 API](img/B09699_02_21.jpg)

PixieDust 图表菜单

当首次调用图表菜单时，将显示一个选项对话框，用于配置一组基本的配置选项，例如使用*X*轴和*Y*轴的内容、聚合类型等。为了节省时间，对话框将预填充 PixieDust 从 DataFrame 自动推测的数据架构。

在以下示例中，我们将创建一个条形图，显示按马力划分的平均油耗：

![显示 – 一种简单的交互式数据可视化 API](img/B09699_02_22.jpg)

条形图对话框选项

点击**OK**将在单元格输出区域显示交互式界面：

![显示 – 一种简单的交互式数据可视化 API](img/B09699_02_23.jpg)

条形图可视化

画布显示图表在中心区域，并在侧边展示与所选图表类型相关的上下文选项。例如，我们可以在**Cluster By**下拉框中选择**origin**字段，按原产国展示细分：

![显示 – 一种简单的交互式数据可视化 API](img/B09699_02_24.jpg)

聚类条形图可视化

如前所述，PixieDust 的`display()`实际上并不创建图表，而是根据选定的选项准备数据，并且通过渲染引擎的 API 调用做重载工作，使用正确的参数。这种设计的目标是让每种图表类型支持多种渲染器，无需额外编程，尽可能为用户提供自由的探索空间。

开箱即用，PixieDust 支持以下渲染器，前提是已安装相应的库。对于未安装的库，PixieDust 日志中将生成警告，并且对应的渲染器将不会在菜单中显示。我们将在第五章中详细介绍 PixieDust 日志，*Python 和 PixieDust 最佳实践与高级概念*。

+   Matplotlib ([`matplotlib.org`](https://matplotlib.org))

+   Seaborn ([`seaborn.pydata.org`](https://seaborn.pydata.org))

    ### 注意

    该库需要使用以下命令安装：`!pip install seaborn.`

+   Bokeh ([`bokeh.pydata.org`](https://bokeh.pydata.org))

    ### 注意

    该库需要使用以下命令安装：`!pip install bokeh.`

+   Brunel ([`brunelvis.org`](https://brunelvis.org))

    ### 注意

    该库需要使用以下命令安装：`!pip install brunel.`

+   Google Map ([`developers.google.com/maps`](https://developers.google.com/maps))

+   Mapbox ([`www.mapbox.com`](https://www.mapbox.com))

    ### 注意

    **注意**：Google Map 和 Mapbox 需要 API 密钥，您可以在各自的站点上获取。

你可以使用**Renderer**下拉框在不同的渲染器之间切换。例如，如果我们想要更多的交互性来探索图表（如缩放和平移），我们可以使用 Bokeh 渲染器，而不是 Matplotlib，后者仅提供静态图像：

![显示 – 一个简单的交互式数据可视化 API](img/B09699_02_25.jpg)

使用 Bokeh 渲染器的簇状条形图

另一个值得一提的图表类型是 Map，当你的数据包含地理空间信息时，它特别有趣，例如经度、纬度或国家/州信息。PixieDust 支持多种类型的地理映射渲染引擎，包括流行的 Mapbox 引擎。

### 注意

在使用 Mapbox 渲染器之前，建议从 Mapbox 网站获取 API 密钥，网址如下：([`www.mapbox.com/help/how-access-tokens-work`](https://www.mapbox.com/help/how-access-tokens-work))。不过，如果没有密钥，PixieDust 将提供一个默认密钥。

为了创建一个地图图表，下面我们使用*东北马萨诸塞百万美元住宅销售*数据集：

```py
import pixiedust
homes = pixiedust.sampleData(6, forcePandas=True) #Million dollar home sales in NE Mass
display(homes)
```

首先，在图表下拉菜单中选择**Map**，然后在选项对话框中，选择`LONGITUDE`和`LATITUDE`作为键，并在提供的输入框中输入 Mapbox 访问令牌。你可以在**Values**区域添加多个字段，它们将作为工具提示显示在地图上：

![显示 – 一个简单的交互式数据可视化 API](img/B09699_02_26.jpg)

Mapbox 图表的选项对话框

当点击**OK**按钮时，你将获得一个交互式地图，你可以使用样式（简单、分区图或密度图）、颜色和底图（亮色、卫星图、暗色和户外）选项来自定义该地图：

![显示 – 一个简单的交互式数据可视化 API](img/B09699_02_27.jpg)

交互式 Mapbox 可视化

每种图表类型都有自己的一套上下文选项，这些选项不难理解，在此我鼓励你尝试每一个选项。如果遇到问题或有改进的想法，你可以随时在 GitHub 上创建一个新问题，网址为[`github.com/ibm-watson-data-lab/pixiedust/issues`](https://github.com/ibm-watson-data-lab/pixiedust/issues)，或者更好的是，提交一个包含代码更改的拉取请求（关于如何做这件事的更多信息可以在这里找到：[`help.github.com/articles/creating-a-pull-request`](https://help.github.com/articles/creating-a-pull-request)）。

为了避免每次单元格运行时重新配置图表，PixieDust 将图表选项存储为 JSON 对象在单元格元数据中，并最终保存到 Notebook 中。你可以通过选择**View** | **Cell Toolbar** | **Edit Metadata**菜单手动检查这些数据，如下所示的截图：

![显示 – 一个简单的交互式数据可视化 API](img/B09699_02_28.jpg)

显示编辑元数据按钮

一个**Edit Metadata**按钮将显示在单元格顶部，点击该按钮后会显示 PixieDust 的配置：

![显示 – 一个简单的交互式数据可视化 API](img/B09699_02_29.jpg)

编辑单元格元数据对话框

当我们在下一节讨论 PixieApps 时，这个 JSON 配置将变得非常重要。

# 过滤

为了更好地探索数据，PixieDust 还提供了一个内置的简单图形界面，可以让你快速筛选正在可视化的数据。你可以通过点击顶级菜单中的筛选切换按钮快速调出筛选器。为了简化操作，筛选器只支持基于单一列构建谓词，这在大多数情况下足以验证简单假设（根据反馈，未来可能会增强此功能，支持多个谓词）。筛选器的 UI 会自动让你选择要筛选的列，并根据其类型显示不同的选项：

+   **数值类型**：用户可以选择一个数学比较符并输入操作数的值。为了方便，UI 还会显示与所选列相关的统计值，这些值可以在选择操作数时使用：![筛选](img/B09699_02_30.jpg)

    对汽车数据集中的 mpg 数值列进行筛选

+   **字符串类型**：用户可以输入一个表达式来匹配列值，可以是正则表达式或普通字符串。为了方便，UI 还会显示有关如何构建正则表达式的基本帮助：

![筛选](img/B09699_02_31.jpg)

对汽车数据集中的 name 字符串类型列进行筛选

点击 **应用** 按钮时，当前的可视化将更新，以反映筛选器的配置。需要注意的是，筛选器适用于整个单元格，而不仅仅是当前的可视化。因此，在切换图表类型时，筛选器仍然会继续应用。筛选器配置也会保存在单元格元数据中，因此在保存笔记本并重新运行单元格时，筛选器配置会被保留。

例如，以下截图将 `cars` 数据集可视化为一个柱状图，显示 `mpg` 大于 `23` 的行，根据统计框，23 是数据集的均值，并按年份聚集。在选项对话框中，我们选择 `mpg` 列作为键，`origin` 作为值：

![筛选](img/B09699_02_32.jpg)

筛选后的汽车数据集柱状图

总结一下，在这一节中，我们讨论了 PixieDust 如何帮助解决三项复杂且耗时的数据科学任务：数据加载、数据清洗和数据可视化。接下来，我们将看到 PixieDust 如何帮助增强数据科学家与开发人员之间的协作。

# 使用 PixieApps 弥合开发人员和数据科学家之间的鸿沟

解决难度较大的数据问题只是数据科学团队任务的一部分。他们还需要确保数据科学结果能正确地投入实际应用，为组织创造商业价值。数据分析的操作化非常依赖于使用案例。例如，这可能意味着创建一个仪表板，用于为决策者整合见解，或者将一个机器学习模型（如推荐引擎）集成到一个网页应用中。

在大多数情况下，这就是数据科学与软件工程相遇的地方（或者有人会说，*关键时刻*）。团队之间的持续合作——而不是一次性的交接——是任务成功完成的关键。往往他们还需要应对不同的编程语言和平台，导致软件工程团队进行大量的代码重写。

我们在*Twitter 标签情感分析*项目中亲身体验到了这一点，当时我们需要构建一个实时仪表盘来可视化结果。数据分析部分是用 Python 编写的，使用了 pandas、Apache Spark 和一些绘图库，如 Matplotlib 和 Bokeh，而仪表盘则是用 Node.js([`nodejs.org`](https://nodejs.org))和 D3([`d3js.org`](https://d3js.org))编写的。

我们还需要在分析和仪表盘之间构建数据接口，而且由于我们需要系统是实时的，我们选择使用 Apache Kafka 来流式传输格式化的分析结果事件。

以下图示概括了一种我称之为**交接模式**的方法，在这种模式下，数据科学团队构建分析并将结果部署在数据接口层中。然后，应用程序将使用这些结果。数据层通常由数据工程师处理，这也是我们在第一章中讨论的角色之一，*编程与数据科学——一套新工具*：

![通过 PixieApps 弥合开发者与数据科学家的差距](img/B09699_02_33.jpg)

数据科学与工程之间的交接

这种交接模式的问题在于，它不利于快速迭代。数据层的任何更改都需要与软件工程团队进行同步，以避免破坏应用程序。PixieApps 的理念是，在构建应用程序的同时，尽量保持与数据科学环境的接近，而在我们的案例中，数据科学环境就是 Jupyter Notebook。通过这种方式，分析结果直接从 PixieApp 中调用，PixieApp 嵌入在 Jupyter Notebook 中运行，因此数据科学家和开发人员可以轻松合作并迭代，从而实现快速改进。

PixieApp 定义了一个简单的编程模型，用于构建具有直接访问 IPython Notebook 内核（即运行 Notebook 代码的 Python 后台进程）的单页面应用程序。本质上，PixieApp 是一个 Python 类，封装了展示和业务逻辑。展示部分由一组特殊的方法组成，称为路由，这些方法返回任意的 HTML 片段。每个 PixieApp 都有一个默认路由，返回启动页面的 HTML 片段。开发人员可以使用自定义 HTML 属性来调用其他路由并动态更新页面的全部或部分内容。例如，一个路由可以调用一个从 Notebook 中创建的机器学习算法，或者使用 PixieDust 显示框架生成图表。

以下图示展示了 PixieApps 如何与 Jupyter Notebook 客户端前端和 IPython 内核交互的高层架构：

![通过 PixieApps 缩小开发人员和数据科学家之间的差距](img/B09699_02_34.jpg)

PixieApp 与 Jupyter 内核的交互

作为 PixieApp 外观的预览，下面是一个*hello world*示例应用程序，具有一个按钮，显示我们在前一节中创建的汽车 DataFrame 的条形图：

```py
#import the pixieapp decorators
from pixiedust.display.app import *

#Load the cars dataframe into the Notebook
cars = pixiedust.sampleData(1)

@PixieApp   #decorator for making the class a PixieApp
class HelloWorldApp():
    #decorator for making a method a
    #route (no arguments means default route)
    @route()
    def main_screen(self):
        return """
        <button type="submit" pd_options="show_chart=true" pd_target="chart">Show Chart</button>
        <!--Placeholder div to display the chart-->
        <div id="chart"></div>
        """

    @route(show_chart="true")
    def chart(self):
        #Return a div bound to the cars dataframe
        #using the pd_entity attribute
        #pd_entity can refer a class variable or
        #a global variable scoped to the notebook
        return """
        <div pd_render_onload pd_entity="cars">
            <pd_options>
                {
                  "title": "Average Mileage by Horsepower",
                  "aggregation": "AVG",
                  "clusterby": "origin",
                  "handlerId": "barChart",
                  "valueFields": "mpg",
                  "rendererId": "bokeh",
                  "keyFields": "horsepower"
                }
            </pd_options>
        </div>
        """
#Instantiate the application and run it
app = HelloWorldApp()
app.run()
```

当上面的代码在 Notebook 单元格中运行时，我们会得到以下结果：

![通过 PixieApps 缩小开发人员和数据科学家之间的差距](img/B09699_02_35.jpg)

Hello World PixieApp

你可能对上面的代码有很多疑问，但不用担心。在接下来的章节中，我们将涵盖所有关于 PixieApp 的技术细节，包括如何在端到端的管道中使用它们。

# 操作化数据科学分析的架构

在前一节中，我们展示了 PixieApps 与 PixieDust 显示框架结合，提供了一种便捷的方法来构建强大的仪表板，直接连接到您的数据分析，允许算法和用户界面之间的快速迭代。这非常适合快速原型设计，但 Notebook 不适合用于生产环境，其中目标用户是业务线用户。一个明显的解决方案是使用传统的三层 Web 应用架构重写 PixieApp，例如，如下所示：

+   用于展示层的 React ([`reactjs.org`](https://reactjs.org))

+   Web 层的 Node.js

+   一个面向机器学习评分或运行其他分析任务的 Web 分析层的数据访问库

然而，这只会比现有流程提供微小的改进，在这种情况下，现有流程仅包含通过 PixieApp 进行迭代实现的能力。

一个更好的解决方案是直接将 PixieApps 作为 Web 应用进行部署和运行，包括将分析嵌入周围的 Notebook 中，并且在此过程中，无需任何代码更改。

使用这种模型，Jupyter Notebooks 将成为简化开发生命周期的核心工具，如下图所示：

![用于实现数据科学分析的架构](img/B09699_02_36.jpg)

数据科学流水线开发生命周期

1.  数据科学家使用 Python Notebook 来加载、丰富和分析数据，并创建分析（机器学习模型、统计分析等）

1.  在同一个 Notebook 中，开发人员创建 PixieApp 来实现这些分析。

1.  一旦准备好，开发人员将 PixieApp 发布为 Web 应用，业务用户可以轻松地通过交互方式使用它，而无需访问 Notebooks。

PixieDust 提供了一个实现该解决方案的组件 PixieGateway。PixieGateway 是一个 Web 应用服务器，负责加载和运行 PixieApps。它建立在 Jupyter 内核网关之上（[`github.com/jupyter/kernel_gateway`](https://github.com/jupyter/kernel_gateway)），而 Jupyter 内核网关本身是建立在 Tornado Web 框架之上的，因此遵循如下所示的架构：

![用于实现数据科学分析的架构](img/B09699_02_37.jpg)

PixieGateway 架构图

1.  PixieApp 直接从 Notebook 发布到 PixieGateway 服务器，并生成一个 URL。在后台，PixieGateway 为 PixieApp 分配一个 Jupyter 内核来运行。根据配置，PixieApp 可以与其他应用共享内核实例，或者根据需求拥有专用内核。PixieGateway 中间件可以通过管理多个内核实例的生命周期进行横向扩展，这些内核实例可以是本地服务器上的，或者是集群上的远程内核。

    ### 注意

    **注意**：远程内核必须是 Jupyter 内核网关。

    使用发布向导，用户可以选择性地为应用定义安全性。提供多种选项，包括基本认证、OAuth 2.0 和 Bearer Token。

1.  业务用户通过浏览器使用第一步生成的 URL 访问应用。

1.  PixieGateway 提供了一个全面的管理控制台，用于管理服务器，包括配置应用程序、配置和监控内核、访问日志进行故障排除等。

1.  PixieGateway 为每个活跃用户管理会话，并使用 IPython 消息协议（[`jupyter-client.readthedocs.io/en/latest/messaging.html`](http://jupyter-client.readthedocs.io/en/latest/messaging.html)）通过 WebSocket 或 ZeroMQ 调度请求到相应的内核进行执行，具体取决于内核是本地的还是远程的。

在将分析产品化时，这种解决方案相比经典的三层 Web 应用架构提供了显著的改进，因为它将 Web 层和数据层合并为一个**Web 分析层**，如下图所示：

![用于实现数据科学分析的架构](img/B09699_02_38.jpg)

经典三层与 PixieGateway 网络架构的比较

在经典的三层架构中，开发人员必须维护多个 REST 接口，这些接口调用数据层的分析功能，并对数据进行处理，以满足展示层的要求，从而正确显示数据。因此，必须在这些接口中添加大量工程工作，增加了开发和代码维护的成本。相比之下，在 PixieGateway 的两层架构中，开发人员不需要担心创建接口，因为服务器负责通过内置的通用接口将请求分发到适当的内核。换句话说，PixieApp 的 Python 方法会自动成为展示层的接口，而无需任何代码更改。这种模型有利于快速迭代，因为 Python 代码的任何变化都能在重新发布后直接反映到应用程序中。

PixieApps 非常适合快速构建单页面应用和仪表板。然而，你可能还希望生成更简单的一页报告并与用户分享。为此，PixieGateway 还允许你通过 **共享** 按钮共享由 `display()` API 生成的图表，生成一个链接到包含图表的网页的 URL。反过来，用户可以通过复制并粘贴为该页面生成的代码，将图表嵌入到网站或博客文章中。

### 注释

**注**：我们将在第四章中详细介绍 PixieGateway，*将数据分析发布到 Web - PixieApp 工具*，包括如何在本地和云端安装新实例。

为了演示此功能，我们使用之前创建的车辆 DataFrame：

![操作化数据科学分析的架构](img/B09699_02_39.jpg)

分享图表对话框

如果共享成功，下一页将显示生成的 URL 和嵌入到网页或博客文章中的代码片段：

![操作化数据科学分析的架构](img/B09699_02_40.jpg)

确认共享图表

点击链接将会带你到该页面：

![操作化数据科学分析的架构](img/B09699_02_41.jpg)

将图表显示为网页

# 总结

在本章中，我们讨论了为什么我们的数据科学工具策略以 Python 和 Jupyter Notebook 为中心。我们还介绍了 PixieDust 的功能，通过以下特点提高用户生产力：

+   数据加载与清洗

+   无需编码即可进行数据可视化和探索

+   一个基于 HTML 和 CSS 的简单编程模型，称为 PixieApp，用于构建与 Notebook 直接交互的工具和仪表板

+   一种点选机制，将图表和 PixieApp 直接发布到网页

在下一章中，我们将深入探讨 PixieApp 编程模型，讨论 API 的各个方面，并附上大量代码示例。
