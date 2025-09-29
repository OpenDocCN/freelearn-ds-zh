# 第二章：Python 中的探索性数据分析与可视化

分析管道不是一步从原始数据中构建的。相反，开发是一个迭代过程，涉及更详细地了解数据，并系统地细化模型和输入以解决问题。这个周期的一个关键部分是交互式数据分析和可视化，这可以为我们的预测建模提供初步的想法，或者为为什么应用程序没有按预期行为提供线索。

电子表格程序是此类探索的一种交互式工具：它们允许用户导入表格信息，旋转和汇总数据，并生成图表。然而，如果数据太大而无法使用此类电子表格应用程序怎么办？如果数据不是表格形式，或者无法有效地以线形或条形图的形式显示呢？在前一种情况下，我们可能只需获得一台更强大的计算机，但后一种情况则更为复杂。简而言之，许多传统的数据可视化工具并不适合复杂的数据类型，如文本或图像。此外，电子表格程序通常假设数据是最终形式，而实际上我们在分析之前通常需要清理原始数据。我们可能还希望计算比简单平均值或总和更复杂的统计数据。最后，使用相同的编程工具来清理和可视化我们的数据，以及生成模型本身并测试其性能，可以使得开发过程更加流畅。

在本章中，我们介绍了交互式 Python（IPython）笔记本应用程序（Pérez, Fernando，和 Brian E. Granger. *IPython：一个交互式科学计算系统*。《科学计算中的计算》9.3（2007）：21-29）。这些笔记本形成了一个数据准备、探索和建模环境，它运行在网页浏览器中。在 IPython 笔记本的输入单元中键入的命令在接收时被翻译并执行：这种交互式编程对于数据探索很有帮助，因为我们可能需要改进我们的努力，并逐步开发更详细的分析。在这些笔记本中记录我们的工作将有助于在调试期间回溯，并作为可以轻松与同事分享的见解的记录。

在本章中，我们将讨论以下主题：

+   将原始数据读取到 IPython 笔记本中，使用 Pandas 库对其进行清理和处理。

+   使用 IPython 处理数值、分类、地理空间或时间序列数据，并执行基本统计分析。

+   基本探索性分析：汇总统计（均值、方差、中位数）、分布（直方图和核密度）、以及自相关（时间序列）。

+   Spark RDDs 和 DataFrames 的分布式数据处理简介。

# 在 IPython 中探索分类和数值数据

我们将通过将文本文件加载到 DataFrame 中，计算一些汇总统计量，并可视化分布来开始我们的 IPython 探索。为此，我们将使用来自互联网电影数据库（[`www.imdb.com/`](http://www.imdb.com/)）的一组电影评分和元数据来调查哪些因素可能与该网站上电影的评分相关。此类信息可能有助于，例如，开发基于此类用户反馈的推荐系统。

## 安装 IPython 笔记本

要跟随示例，您应该在计算机上安装 Windows、Linux 或 Mac OSX 操作系统，并能够访问互联网。有几种安装 IPython 的选项：由于每个资源都包括安装指南，我们提供了可用的来源摘要，并将读者指引到相关文档以获取更深入的说明。

+   对于大多数用户，Anaconda（Continuum Analytics）或 Canopy（Enthought）之类的预捆绑 Python 环境提供了一个包含 IPython 和我们将在这项练习中使用的所有库的即用型发行版：这些产品是自包含的，因此您不需要担心版本冲突或依赖关系管理。

+   对于更有雄心的用户，您可以选择安装 Python 发行版，然后使用`pip`或`easy_install`之类的包管理器安装所需的库。

## 笔记本界面

让我们按照以下步骤开始：

1.  一旦您安装了 IPython，打开计算机上的命令提示符（终端）并输入：

    ```py
    jupyter notebook

    ```

    注意，根据您安装程序的位置，`jupyter`命令可能需要将启动`jupyter`的二进制文件放在您的系统路径中。您应该在终端中看到一系列如下命令：

    ![笔记本界面](img/B04881_chapter02_01.jpg)

    这将启动**内核**，即计算笔记本中输入的命令结果的 Python 解释器。如果您想停止笔记本，请按*Ctrl* + *C*，然后输入**yes**，内核将关闭。

1.  当内核启动时，您的默认网络浏览器也应该打开，显示一个主页，如下所示：![笔记本界面](img/B04881_chapter02_02.jpg)

1.  **文件**标签（见上图）将显示您启动 IPython 进程的目录中的所有文件。点击**运行**将显示所有正在运行的笔记本列表——开始时没有：![笔记本界面](img/B04881_chapter02_30.jpg)

1.  最后，集群面板会列出外部集群，如果我们决定通过将命令提交到多个机器上处理来并行化我们的计算，那么我们会考虑这一点。现在我们不必担心这个问题，但当我们开始训练预测模型时，它将变得非常有用，这项任务通常可以通过在多台计算机或处理器之间分配工作来加速。

1.  返回到 **文件** 选项卡，你会在右上角注意到两个选项。一个是 **上传** 文件：当我们在本地上运行 IPython 时，它同样可以运行在远程服务器上，分析师可以通过浏览器访问笔记本。在这种情况下，为了与存储在我们自己机器上的文件交互，我们可以使用这个按钮打开提示并选择要上传到服务器的文件，然后我们可以在笔记本中分析它们。**新建** 选项卡允许你创建新的文件夹、文本文件、在浏览器中运行的 Python 终端或笔记本。

    现在，让我们通过双击 **B04881_chapter02_code01.ipynb** 来打开本章的示例笔记本。这会打开笔记本：

    ![笔记本界面](img/B04881_chapter02_03.jpg) 笔记本由一系列单元格组成，这些是我们可以输入 Python 代码、执行它并查看命令结果的地方。每个单元格中的 Python 代码可以通过点击工具栏上的 ![笔记本界面](img/B04881_chapter02_55.jpg) 按钮来执行，并且可以通过点击 ![笔记本界面](img/B04881_chapter02_56.jpg) 按钮在当前单元格下方插入一个新的单元格。

1.  虽然第一个单元格中的导入语句可能看起来与你在命令行或脚本中使用 Python 的经验相似，但 `%matplotlib` 内联命令实际上不是 Python：它是对笔记本的标记指令，指示 `matplotlib` 图像要在浏览器中内联显示。我们在笔记本的开始处输入这个命令，以便我们所有的后续绘图都使用这个设置。要运行导入语句，点击 ![笔记本界面](img/B04881_chapter02_57.jpg) 按钮，或者按 *Ctrl* + *Enter*。当命令执行时，单元格上的 `ln[1]` 可能会短暂地变为 `[*]`。在这种情况下，将没有输出，因为我们只是导入了库依赖项。现在我们的环境已经准备好了，我们可以开始检查一些数据。

## 加载数据和检查数据

首先，我们将使用 Pandas 库将 `movies.csv` 中的数据导入到 DataFrame 对象中（McKinney, Wes. *Python for data analysis: Data wrangling with Pandas*, NumPy, and IPython. O'Reilly Media, Inc., 2012）。这个 DataFrame 类似于传统的电子表格软件，并允许强大的扩展，如自定义转换和聚合。这些可以与 NumPy 中可用的数值方法结合，以进行更高级的数据统计分析。让我们继续我们的分析：

1.  如果这是一个新的笔记本，要添加新的单元格，我们会去工具栏，点击 **插入** 并选择 **在下方插入单元格**，或者使用 ![加载数据和检查数据](img/B04881_chapter02_57.jpg) 按钮。然而，在这个例子中，所有单元格已经生成，因此我们在第二个单元格中运行以下命令：

    ```py
    >>> imdb_ratings = pd.read_csv('movies.csv')

    ```

    我们现在已经使用 Pandas 库创建了 `imdb_ratings` DataFrame 对象，并可以开始分析数据。

1.  让我们从使用 `head()` 和 `tail()` 查看数据的开始和结束部分开始。请注意，默认情况下，此命令返回前五行数据，但我们可以向命令提供一个整数参数来指定要返回的行数。此外，默认情况下，文件的第一行假定包含列名，在这种情况下是正确的。键入：

    ```py
    >>> imdb_ratings.head()

    ```

    输出如下：

    ![加载数据并检查](img/B04881_chapter02_04.jpg)

    我们可以通过键入以下内容来查看数据的最后 15 行：

    ```py
    >>> imdb_ratings.tail(15)

    ```

    ![加载数据并检查](img/B04881_chapter02_05.jpg)

1.  查看单个行可以让我们了解文件包含的数据类型：我们还可以使用 `describe()` 命令查看每个列中所有行的摘要，该命令返回记录数、平均值和其他聚合统计信息。尝试键入：

    ```py
    >>> imdb_ratings.describe()

    ```

    这给出了以下输出：

    ![加载数据并检查](img/B04881_chapter02_06.jpg)

1.  列名和它们的数据类型可以通过 `columns` 和 `dtypes` 属性访问。键入：

    ```py
    >>> imdb_ratings.columns

    ```

    给出列名：

    ![加载数据并检查](img/B04881_chapter02_58.jpg)

    如果我们发出以下命令：

    ```py
    >>> imdb_ratings.dtypes

    ```

1.  如我们所见，当我们首次加载文件时，列的数据类型已被自动推断：![加载数据并检查](img/B04881_chapter02_07.jpg)

1.  如果我们想访问单个列的数据，我们可以使用 `{DataFrame_name}.{column_name}` 或 `{DataFrame_name}['column_name']`（类似于 Python 字典）。例如，键入：

    ```py
    >>> imdb_ratings.year.head()

    ```

    或

    ```py
    >>> imdb_ratings['year'].head()

    ```

    输出如下：

    ![加载数据并检查](img/B04881_chapter02_08.jpg)

不费吹灰之力，我们就可以使用这些简单的命令来对数据进行一系列诊断性提问。我们使用 `describe()` 生成的汇总统计信息是否合理（例如，最高评分应该是 10，而最低是 1）？数据是否正确解析到我们预期的列中？

回顾我们使用 `head()` 命令可视化的前五行数据，这次初步检查也揭示了一些我们可能需要考虑的格式问题。在 **budget** 列中，有几个条目具有 `NaN` 值，表示缺失值。如果我们打算尝试根据包括 **budget** 在内的特征预测电影评分，我们可能需要制定一个规则来填充这些缺失值，或者以正确表示给算法的方式对它们进行编码。

## 基本操作 - 分组、过滤、映射和转置

现在我们已经了解了 Pandas DataFrame 的基本特征，让我们开始应用一些转换和计算，这些转换和计算超出了我们通过 `describe()` 获得的简单统计信息。例如，如果我们想计算属于每个发布年份的电影数量，我们可以使用以下命令：

```py
>>> imdb_ratings.value_counts()

```

输出如下：

![基本操作 - 分组、过滤、映射和转置](img/B04881_chapter02_09.jpg)

注意，默认情况下，结果是按每年记录数排序的（在这个数据集中，2002 年上映的电影最多）。如果我们想按发行年份排序怎么办？`sort_index()` 命令按其索引（属于的年份）对结果进行排序。索引类似于图表的轴，其值表示每个轴刻度处的点。使用以下命令：

```py
>>> imdb_ratings.year.value_counts().sort_index(ascending=False)

```

给出以下输出：

![基本操作 – 分组、过滤、映射和转置](img/B04881_chapter02_31.jpg)

我们还可以使用 DataFrame 来开始对数据进行分析性提问，就像在数据库查询中那样进行逻辑切片和子选择。例如，让我们使用以下命令选择 1999 年之后上映并具有 R 评分的电影子集：

```py
>>> imdb_ratings[(imdb_ratings.year > 1999) & (imdb_ratings.mpaa == 'R')].head()

```

这给出了以下输出：

![基本操作 – 分组、过滤、映射和转置](img/B04881_chapter02_11.jpg)

同样，我们可以根据任何列（s）对数据进行分组，并使用 `groupby` 命令计算聚合统计信息，将执行计算的数组作为参数传递给 `aggregate`。让我们使用 NumPy 中的平均值和标准差函数来找到特定年份上映电影的平均评分和变化：

```py
>>> imdb_ratings.groupby('year').rating.aggregate([np.mean,np.std])

```

这给出了：

![基本操作 – 分组、过滤、映射和转置](img/B04881_chapter02_32.jpg)

然而，有时我们想要提出的问题需要我们重塑或转换我们给出的原始数据。这在后面的章节中会经常发生，当我们为预测模型开发特征时。Pandas 提供了许多执行这种转换的工具。例如，虽然根据类型对数据进行聚合也可能很有趣，但我们注意到在这个数据集中，每个类型都由一个单独的列表示，其中 1 或 0 表示一部电影是否属于某个特定类型。对我们来说，有一个单独的列来指示电影属于哪个类型，以便在聚合操作中使用会更有用。我们可以使用带有参数 1 的 `idxmax()` 命令来创建这样的列，以表示跨列的最大值（0 将表示沿行的最大索引），它返回所选列中值最大的列。键入：

```py
>>>imdb_ratings['genre']=imdb_ratings[['Action','Animation','Comedy','Drama','Documentary','Romance']].idxmax(1)

```

当我们使用以下内容检查这个新的类型列时，给出以下结果：

```py
>>> imdb_ratings['genre'].head()

```

![基本操作 – 分组、过滤、映射和转置](img/B04881_chapter02_33.jpg)

我们也许还希望用代表特定类型的颜色来绘制数据。为了为每个类型生成一个颜色代码，我们可以使用以下命令的自定义映射函数：

```py
>>> genres_map = {"Action": 'red', "Animation": 'blue', "Comedy": 'yellow', "Drama": 'green', "Documentary": 'orange', "Romance": 'purple'}
>>> imdb_ratings['genre_color'] = imdb_ratings['genre'].apply(lambda x: genres_map[x])

```

我们可以通过键入以下内容来验证输出：

```py
>>> imdb_ratings['genre_color'].head()

```

这给出了以下结果：

![基本操作 – 分组、过滤、映射和转置](img/B04881_chapter02_34.jpg)

我们还可以转置表格并使用 `pivot_table` 命令进行统计分析，该命令可以在类似于电子表格的行和列分组上执行聚合计算。例如，为了计算每个年份每个类型的平均评分，我们可以使用以下命令：

```py
>>>pd.pivot_table(imdb_ratings,values='rating',index='year',columns=['genre'],aggfunc=np.mean)

```

这给出了以下输出：

![基本操作 – 分组、过滤、映射和转置](img/B04881_chapter02_35.jpg)

现在我们已经进行了一些探索性计算，让我们来看看这些信息的可视化。

## 使用 Matplotlib 绘图

IPython 笔记本的一个实用功能是能够将数据与我们的分析内联绘图。例如，如果我们想可视化电影长度的分布，我们可以使用以下命令：

```py
>>> imdb_ratings.length.plot()

```

![使用 Matplotlib 绘图](img/B04881_chapter02_16.jpg)

然而，这并不是一个非常吸引人的图像。为了制作一个更美观的图表，我们可以使用 `style.use()` 命令更改默认样式。让我们将样式更改为 `ggplot`，这是在 `ggplot` 图形库（Wickham, Hadley. *ggplot: An Implementation of the Grammar of Graphics*. R 包版本 0.4.0 (2006)）中使用的。键入以下命令：

```py
>>> matplotlib.style.use('ggplot')
>>> imdb_ratings.length.plot()

```

给出了一张更吸引人的图形：

![使用 Matplotlib 绘图](img/B04881_chapter02_36.jpg)

如您所见，默认图表是折线图。折线图按 DataFrame 中的行号从左到右将每个数据点（电影时长）作为一条线绘制。为了按电影类型绘制密度图，我们可以使用 `groupby` 命令并带有 `type=kde` 参数。**KDE** 是 **Kernel Density Estimate**（Rosenblatt, Murray. Remarks on some nonparametric estimates of a density function. *The Annals of Mathematical Statistics 27.3 (1956): 832-837*; Parzen, Emanuel. On estimation of a probability density function and mode. The annals of mathematical statistics 33.3 (1962): 1065-1076）的缩写，意味着对于每个点（电影时长），我们使用以下方程估计密度（具有该运行时的人口比例）：

![使用 Matplotlib 绘图](img/B04881_chapter02_60.jpg)

其中 `f(x)` 是概率密度的估计，n 是我们数据集中的记录数，`h` 是带宽参数，`K` 是核函数。例如，如果 `K` 是以下高斯核函数：

![使用 Matplotlib 绘图](img/B04881_chapter02_59.jpg)

其中 σ 是正态分布的标准差，μ 是正态分布的均值，那么核密度估计（KDE）表示围绕给定点 x 的正态分布“窗口”中所有其他数据点的平均密度。这个窗口的宽度由 *h* 给出。因此，KDE 通过在给定点绘制连续概率估计而不是绝对计数来允许我们绘制直方图的平滑表示。为此 KDE 图，我们还可以添加轴的注释，并使用以下命令将最大运行时间限制为 2 小时：

```py
>>> plot1 = imdb_ratings.groupby('genre').length.plot(kind='kde',xlim=(0,120),legend='genre')
>>>plot1[0].set_xlabel('Number of Minutes')
>>>plot1[0].set_title('Distribution of Films by Runtime Minutes')

```

这给出了以下图表：

![使用 Matplotlib 绘图](img/B04881_chapter02_39.jpg)

我们可以看到，不出所料，许多动画电影较短，而其他类别平均长度约为 90 分钟。我们还可以使用以下命令绘制类似的密度曲线来检查不同类型之间的评分分布：

```py
>>> plot2 = imdb_ratings.groupby('genre').rating.plot(kind='kde',xlim=(0,10),legend='genre')
>>> plot2[0].set_xlabel('Ratings')
>>> plot2[0].set_title('Distribution of Ratings')

```

这给出了以下图表：

![使用 Matplotlib 绘图](img/B04881_chapter02_40.jpg)

有趣的是，纪录片平均评分最高，而动作电影评分最低。我们也可以使用以下命令使用箱线图来可视化相同的信息：

```py
>>> pd.pivot_table(imdb_ratings,values='rating',index='title',columns=['genre']).\
plot(kind='box',by='genre').\
set_ylabel('Rating')

```

这给出了以下箱线图：

![使用 Matplotlib 绘图](img/B04881_chapter02_41.jpg)

我们也可以使用笔记本开始为数据集自动进行这种类型的绘图。例如，我们经常想查看每个变量的边缘图（其单维分布）与所有其他变量的比较，以便在数据集的列之间找到相关性。我们可以使用内置的`scatter_matrix`函数来完成此操作：

```py
>>> from pandas.tools.plotting import scatter_matrix
>>> scatter_matrix(imdb_ratings[['year','length','budget','rating','votes']], alpha=0.2, figsize=(6, 6), diagonal='kde')

```

这将允许我们绘制我们选择的变量的成对分布，从而为我们提供它们之间潜在相关性的概述：

![使用 Matplotlib 绘图](img/B04881_chapter02_42.jpg)

这张单张图实际上提供了很多信息。例如，它显示一般来说，预算较高的电影评分较高，而 20 世纪 20 年代制作的电影的平均评分高于之前的电影。使用这种散点矩阵，我们可以寻找可能指导预测模型发展的相关性，例如根据其他电影特征预测评分。我们只需要给这个函数提供 DataFrame 中用于绘图的列子集（因为我们想排除无法以这种方式可视化的非数值数据），我们就可以为任何新的数据集重复这种分析。

如果我们想更详细地可视化这些分布呢？作为一个例子，让我们使用以下命令来根据类型将长度和评分之间的相关性分解：

```py
>>> fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))
>>> row = 0
>>> col = 0
>>> for index, genre in imdb_ratings.groupby('genre'):
…    if row > 2:
…        row = 0
…       col += 1
…    genre.groupby('genre').\
....plot(ax=axes[row,col],kind='scatter',x='length',y='rating',s=np.sqrt(genre['votes']),c=genre['genre_color'],xlim=(0,120),ylim=(0,10),alpha=0.5,label=index)
…    row += 1

```

在这个命令中，我们创建一个 3x2 的网格来容纳六个类型的绘图。然后我们按类型迭代数据组，如果我们已经到达第三行，我们就重置并移动到第二列。然后我们绘制数据，使用我们之前生成的`genre_color`列以及索引（类型组）来标记绘图。我们通过每个点（代表一部电影）收到的投票数来调整每个点的大小。生成的散点图显示了长度和类型之间的关系，点的大小给出了我们对点值的信心程度。

![使用 Matplotlib 绘图](img/B04881_chapter02_43.jpg)

现在我们已经使用分类数据和数值数据进行了基本分析，让我们继续一个特殊的数值数据案例——时间序列。

# 时间序列分析

虽然 `imdb` 数据包含了电影发行年份，但本质上我们感兴趣的是单个电影和评分，而不是随时间推移可能相互关联的一系列事件。这种后一种类型的数据——时间序列——提出了不同的问题。数据点是否相互关联？如果是，它们在什么时间段内相关？信号有多嘈杂？Pandas DataFrames 有许多内置的时间序列分析工具，我们将在下一节中探讨。

## 清洗和转换

在我们之前的例子中，我们能够以提供的数据形式使用这些数据。然而，并不总是有保证这种情况会发生。在我们的第二个例子中，我们将查看过去一个世纪美国按年份的石油价格时间序列（Makridakis, Spyros, Steven C. Wheelwright, 和 Rob J. Hyndman. 《预测方法和应用》，John Wiley & Sons. Inc, 纽约(1998)。我们将再次通过将此数据加载到笔记本中，并使用 `tail()` 通过输入来检查它：

```py
>>> oil_prices = pd.read_csv('oil.csv')
>>> oil_prices.tail()

```

这给出了以下输出：

![清洗和转换](img/B04881_chapter02_10.jpg)

最后一行是意外的，因为它看起来根本不像一个年份。实际上，它是电子表格中的一个页脚注释。由于它实际上不是数据的一部分，我们需要将其从数据集中删除，这可以通过以下命令完成：

```py
>>> oil_prices = oil_prices[~np.isnan(oil_prices[oil_prices.columns[1]])] 

```

这将从数据集中删除第二列是 NaN（不是正确格式的数字）的行。我们可以通过再次使用 tail 命令来验证我们已经清理了数据集。

我们希望清理数据的第二个方面是格式。如果我们查看列的格式，使用：

```py
>>> oil_prices.dtypes

```

我们看到年份默认不是解释为 Python 日期类型：

![清洗和转换](img/B04881_chapter_02_25.jpg)

我们希望 **年份** 列是 Python 日期类型。Pandas 提供了使用 `convert_object()` 命令执行此转换的内置功能：

```py
>>> oil_prices = oil_prices.convert_objects(convert_dates='coerce')

```

同时，我们可以使用 `rename` 命令将价格列重命名为更简洁的名称：

```py
>>> oil_prices.rename(columns = {oil_prices.columns[1]: 'Oil_Price_1997_Dollars'},inplace=True)

```

然后，我们可以通过使用 head() 命令来验证输出显示了这些变化：

![清洗和转换](img/B04881_chapter02_26.jpg)

现在我们有了可以开始对这个时间序列进行一些诊断的数据格式。

## 时间序列诊断

我们可以使用上一节中介绍的 `matplotlib` 命令来绘制这些数据，如下所示：

```py
>>> oil_prices.plot(x='Year',y='Oil_Price_1997_Dollars')

```

这将生成以下时间序列图：

![时间序列诊断](img/B04881_chapter02_44.jpg)

对于这些数据，我们可能会有许多自然的问题。每年石油价格的波动是完全随机的，还是每年之间的测量相互关联？数据中似乎存在一些周期，但很难量化这种相关性的程度。我们可以使用一个视觉工具来帮助诊断这个特征，这个工具是`lag_plot`，在 Pandas 中使用以下命令可用：

```py
>>> from pandas.tools.plotting import lag_plot
>>> lag_plot(oil_prices.Oil_Price_1997_Dollars)

```

![时间序列诊断](img/B04881_chapter02_45.jpg)

滞后图简单地绘制了每年石油价格（x 轴）与随后一年石油价格（y 轴）的关系。如果没有相关性，我们预计会看到一个圆形云。这里的线性模式表明数据中存在某种结构，这与每年价格上升或下降的事实相符。这种相关性与预期相比有多强？我们可以使用自相关图来回答这个问题，使用以下命令：

```py
>>> from pandas.tools.plotting import autocorrelation_plot
>>> autocorrelation_plot(oil_prices['Oil_Price_1997_Dollars'])

```

这给出了以下自相关图：

![时间序列诊断](img/B04881_chapter02_46.jpg)

在这个图中，不同滞后（年份差异）的点之间的相关性被绘制出来，同时还有 95%置信区间（实线）和 99%置信区间（虚线）线，表示随机数据预期相关性的范围。根据这个可视化，似乎在滞后小于 10 年的情况下存在异常的相关性，这与上述第一幅图中峰值价格期间的近似持续时间相吻合。

## 将信号和相关性连接起来

最后，让我们看看一个比较石油价格时间序列与其他数据集的例子，即美国给定年份的汽车事故死亡人数（*美国年度机动车死亡名单*。维基百科。维基媒体基金会。网络。2016 年 5 月 2 日。[`en.wikipedia.org/wiki/List_of_motor_vehicle_deaths_in_U.S._by_year`](https://en.wikipedia.org/wiki/List_of_motor_vehicle_deaths_in_U.S._by_year)）。

例如，我们可能会假设，随着石油价格的上涨，平均消费者将驾驶得少，从而导致未来的车祸。同样，我们需要在将数据集时间转换为日期格式之前，先将数字转换为字符串，使用以下命令：

```py
>>> car_crashes=pd.read_csv("car_crashes.csv")
>>> car_crashes.Year=car_crashes.Year.astype(str)
>>> car_crashes=car_crashes.convert_objects(convert_dates='coerce') 

```

使用`head()`命令检查前几行确认我们已经成功格式化了数据：

![连接信号和相关性](img/B04881_chapter02_47.jpg)

我们可以将这些数据与石油价格统计数据合并，并比较两个趋势随时间的变化。请注意，我们需要通过除以 1000 来重新缩放崩溃数据，以便在以下命令中可以轻松地将其与同一轴上的数据进行查看：

```py
>>> car_crashes['Car_Crash_Fatalities_US']=car_crashes['Car_Crash_Fatalities_US']/1000

```

然后，我们使用`merge()`将数据连接起来，通过`on`变量指定用于匹配每个数据集中行的列，并使用以下命令绘制结果：

```py
>>> oil_prices_car_crashes = pd.merge(oil_prices,car_crashes,on='Year')
>>> oil_prices_car_crashes.plot(x='Year')

```

结果图如下所示：

![连接信号和相关性](img/B04881_chapter02_48.jpg)

这两个信号的相关性如何？我们再次可以使用`auto_correlation`图来探索这个问题：

```py
>>> autocorrelation_plot(oil_prices_car_crashes[['Car_Crash_Fatalities_US','Oil_Price_1997_Dollars']])

```

这给出了：

![信号连接和相关性](img/B04881_chapter02_49.jpg)

因此，似乎这种相关性超出了 20 年或更短预期波动范围，比仅从油价中出现的关联范围更长。

### 小贴士

**处理大型数据集**

本节中给出的示例规模较小。在实际应用中，我们可能需要处理无法装在计算机上的数据集，或者需要进行分析，这些分析计算量如此之大，以至于必须在多台机器上分割运行才能在合理的时间内完成。对于这些用例，可能无法使用我们用 Pandas DataFrames 展示的形式使用 IPython Notebook。对于处理此类规模的数据，有多个替代应用程序可用，包括 PySpark ([`spark.apache.org/docs/latest/api/python/`](http://spark.apache.org/docs/latest/api/python/))、H20 ([`www.h2o.ai/`](http://www.h2o.ai/)) 和 XGBoost ([`github.com/dmlc/xgboost`](https://github.com/dmlc/xgboost))。我们也可以通过笔记本使用这些工具中的许多，从而实现对极大数据量的交互式操作和建模。

# 处理地理空间数据

对于我们的最后一个案例研究，让我们探索使用 Pandas 库的扩展 GeoPandas 来分析地理空间数据。您需要在您的 IPython 环境中安装 GeoPandas 才能跟随此示例。如果尚未安装，您可以使用 `easy_install` 或 pip 安装它。

## 加载地理空间数据

除了我们的其他依赖项之外，我们将使用以下命令导入 `GeoPandas` 库：

```py
>>> import GeoPandas as geo.

```

我们为此示例加载数据集，非洲国家坐标（"Africa." Maplibrary.org。网络。2016 年 5 月 2 日。[`www.mapmakerdata.co.uk.s3-website-eu-west-1.amazonaws.com/library/stacks/Africa/`](http://www.mapmakerdata.co.uk.s3-website-eu-west-1.amazonaws.com/library/stacks/Africa/))，这些坐标以前包含在一个形状（`.shp`）文件中，现在将其导入到 **GeoDataFrame** 中，这是 Pandas DataFrame 的扩展，使用以下方式：

```py
>>> africa_map = geo.GeoDataFrame.from_file('Africa_SHP/Africa.shp')

```

使用 `head()` 查看前几行：

![加载地理空间数据](img/B04881_chapter02_50.jpg)

我们可以看到，数据由标识符列组成，以及一个表示国家形状的几何对象。`GeoDataFrame` 还有一个 `plot()` 函数，我们可以传递一个 `column` 参数，指定用于生成每个多边形颜色的字段，如下所示：

```py
>>> africa_map.plot(column='CODE')

```

这给出了以下可视化：

![加载地理空间数据](img/B04881_chapter02_51.jpg)

然而，目前这个颜色代码是基于国家名称的，所以对地图的洞察力不大。相反，让我们尝试根据每个国家的人口来着色每个国家，使用每个国家的人口密度信息（*Population by Country – Thematic Map – World*。*Population by Country – Thematic Map-World*。网络。2016 年 5 月 2 日，[`www.indexmundi.com/map/?v=21`](http://www.indexmundi.com/map/?v=21)）。首先，我们使用以下方式读取人口：

```py
>>> africa_populations = pd.read_csv('Africa_populations.tsv',sep='\t')

```

注意，在这里我们已将`sep='\t'`参数应用于`read_csv()`，因为该文件中的列不是像迄今为止的其他示例那样以逗号分隔。现在我们可以使用合并操作将此数据与地理坐标连接起来：

```py
>>> africa_map = pd.merge(africa_map,africa_populations,left_on='COUNTRY',right_on='Country_Name')

```

与上面提到的石油价格和事故死亡率的例子不同，这里我们希望用来连接数据的列在每个数据集中都有不同的名称，因此我们必须使用`left_on`和`right_on`参数来指定每个表中所需的列。然后我们可以使用以下方法使用来自人口数据的颜色绘制地图：

```py
>>> africa_map.plot(column='Population',colormap='hot')

```

这给出了新的地图如下：

![加载地理空间数据](img/B04881_chapter02_52.jpg)

现在，我们可以清楚地看到人口最多的国家（埃塞俄比亚、刚果民主共和国和埃及）以白色突出显示。

## 在云端工作

在前面的例子中，我们假设您正在通过您的网络浏览器在本地计算机上运行 IPython 笔记本。如前所述，应用程序也可以在远程服务器上运行，用户可以通过界面上传文件以远程交互。这种外部服务的一种方便形式是云平台，如**Amazon Web Services**（**AWS**）、Google Compute Cloud 和 Microsoft Azure。除了提供托管平台以运行笔记本等应用程序外，这些服务还提供存储，可以存储比我们个人电脑能存储的更大的数据集。通过在云端运行我们的笔记本，我们可以更轻松地使用共享的数据访问和处理基础设施与这些分布式存储系统进行交互，同时也强制执行所需的安全性和数据治理。最后，通过这些云服务提供的廉价计算资源也可能使我们能够扩展我们在后面章节中描述的计算类型，通过添加额外的服务器来处理笔记本后端输入的命令。

# PySpark 简介

到目前为止，我们主要关注可以适应单个机器的数据库集。对于更大的数据集，我们可能需要通过分布式文件系统如 Amazon S3 或 HDFS 来访问它们。为此，我们可以利用开源分布式计算框架 PySpark ([`spark.apache.org/docs/latest/api/python/`](http://spark.apache.org/docs/latest/api/python/))。PySpark 是一个分布式计算框架，它使用**弹性分布式数据集**（**RDDs**）的抽象来处理对象的并行集合，这使得我们可以像它适合单个机器一样程序化地访问数据集。在后面的章节中，我们将演示如何在 PySpark 中构建预测模型，但在此介绍中，我们关注 PySpark 中的数据处理函数。

## 创建 SparkContext

任何 Spark 应用程序的第一步是生成 SparkContext。SparkContext 包含任何特定作业的配置（例如内存设置或工作任务的数目），并允许我们通过指定主节点来连接到 Spark 集群。我们使用以下命令启动 SparkContext：

```py
>>> sc = SparkContext('local','job_.{0}'.format(uuid.uuid4()))

```

第一个参数给出了我们的 Spark 主节点的 URL，即协调 Spark 作业执行并将任务分配给集群中工作机器的机器。所有 Spark 作业都包含两种任务：**驱动器**（负责发布命令并收集作业进度的信息）和**执行器**（在 RDD 上执行操作）。这些任务可以创建在同一台机器上（如我们的示例所示），也可以在不同的机器上，这样就可以使用多台计算机的并行计算来分析无法在单台机器内存中容纳的数据集。在这种情况下，我们将本地运行，因此将主节点的参数指定为 `localhost`，但否则这可以是集群中远程机器的 URL。第二个参数只是我们给应用程序起的名字，我们使用 `uuid` 库生成的唯一 ID 来指定它。如果此命令成功，你应该在你的终端中看到运行笔记本的位置出现以下堆栈跟踪：

![创建 SparkContext](img/B04881_chapter02_53.jpg)

我们可以使用地址 `http://localhost:4040` 打开 SparkUI，它看起来如下所示：

![创建 SparkContext](img/B04881_chapter02_54.jpg)

你可以在右上角看到我们的作业名称，一旦开始运行它们，我们可以使用这个页面来跟踪作业的进度。现在 SparkContext 已经准备好接收命令，并且我们可以在 `ui` 中看到我们在笔记本中执行的任何操作的进度。如果你想停止 SparkContext，我们可以简单地使用以下命令：

```py
>>> sc.stop()

```

注意，如果我们本地运行，我们一次只能在一个 `localhost` 上启动一个 SparkContext，所以如果我们想更改上下文，我们需要停止并重新启动它。一旦我们创建了基本的 SparkContext，我们就可以实例化其他上下文对象，这些对象包含特定类型数据集的参数和功能。对于这个例子，我们将使用 SqlContext，它允许我们对 DataFrame 进行操作并使用 SQL 逻辑查询数据集。我们使用 SparkContext 作为参数来生成 SqlContext：

```py
>>> sqlContext = SQLContext(sc)

```

## 创建 RDD

要生成我们的第一个 RDD，让我们再次加载电影数据集，并使用除了索引和行号之外的所有列将其转换为元组列表：

```py
>>> data = pd.read_csv("movies.csv")
>>> rdd_data = sc.parallelize([ list(r)[2:-1] for r in data.itertuples()])

```

`itertuples()` 命令将 pandas DataFrame 的每一行返回为一个元组，然后我们通过将其转换为列表并取索引 `2` 及以上（代表所有列，但不是行的索引，这是 Pandas 自动插入的，以及行号，这是文件中的原始列之一）来切片。为了将此本地集合转换为 RDD，我们调用 `sc.parallelize`，它将集合转换为 RDD。我们可以使用 `getNumPartitions()` 函数检查这个分布式集合中有多少个分区：

```py
>>> rdd_data.getNumPartitions()

```

由于我们刚刚在本地创建了此数据集，它只有一个分区。我们可以使用 `repartition()`（增加分区数量）和 `coalesce()`（减少）函数来更改 RDD 中的分区数量，这可以改变对数据每个子集所做的负载。你可以验证以下命令更改了我们示例中的分区数量：

```py
>>> rdd_data.repartition(10).getNumPartitions() 
>>> rdd_data.coalesce(2).getNumPartitions()

```

如果我们想检查 RDD 中的小部分数据，可以使用 `take()` 函数。以下命令将返回五行：

```py
rdd_data.take(5)

```

你可能会注意到，在输入需要将结果打印到笔记本的命令之前，Spark UI 上没有任何活动，例如 `getNumPartitions()` 或 `take()`。这是因为 Spark 遵循惰性执行模型，只有在需要用于下游操作时才返回结果，否则等待这样的操作。除了提到的那些之外，其他将强制执行的操作包括写入磁盘和 `collect()`（下面将描述）。

为了使用 PySpark DataFrames API（类似于 Pandas DataFrames）来加载数据，而不是使用 RDD（它没有我们上面展示的 DataFrame 操作的许多实用函数），我们需要一个 **JavaScript 对象表示法**（**JSON**）格式的文件。我们可以使用以下命令生成此文件，该命令将每行的元素映射到一个字典，并将其转换为 JSON：

```py
>>> rdd_data.map( lambda x: json.JSONEncoder().encode({ str(k):str(v) for (k,v) in zip(data.columns[2:-1],x)})).\
>>> saveAsTextFile('movies.json')

```

如果你检查输出目录，你会注意到我们实际上保存了一个名为 `movies.json` 的目录，其中包含单个文件（与我们的 RDD 中的分区数量一样多）。这是数据在 **Hadoop 分布式文件系统**（**HDFS**）中存储在目录中的相同方式。

注意，我们刚刚只是触及了 RDD 可以执行的所有操作的一小部分。我们可以执行其他操作，如过滤、按键对 RDD 进行分组、投影每行的子集、在组内排序数据、与其他 RDD 进行连接，以及许多其他操作。所有可用的转换和操作的完整范围在 [`spark.apache.org/docs/latest/api/python/pyspark.html`](http://spark.apache.org/docs/latest/api/python/pyspark.html) 中有文档说明。

## 创建 Spark DataFrame

现在我们有了 JSON 格式的文件，我们可以使用以下方式将其加载为 Spark DataFrame：

```py
>>> df = sqlContext.read.json("movies.json")

```

如果我们打算对这份数据执行许多操作，我们可以将其缓存（在临时存储中持久化），这样我们就可以在 Spark 自己的内部存储格式上操作数据，该格式针对重复访问进行了优化。我们可以使用以下命令缓存数据集。

```py
>>> df.cache()

```

`SqlContext` 还允许我们为数据集声明一个表别名：

```py
>>> df.registerTempTable('movies')

```

然后，我们可以像查询关系数据库系统中的表一样查询这些数据：

```py
>>> sqlContext.sql(' select * from movies limit 5 ').show()

```

与 Pandas DataFrame 类似，我们可以通过特定列对它们进行聚合：

```py
>>> df.groupby('year').count().collect()

```

我们也可以使用与 Pandas 相似的语法来访问单个列：

```py
>>> df.year

```

如果我们希望将所有数据带到一台机器上，而不是在可能分布在几台计算机上的数据集分区上操作，我们可以调用 `collect()` 命令。使用此命令时请谨慎：对于大型数据集，它将导致所有数据分区被合并并发送到驱动器，这可能会潜在地超载驱动器的内存。`collect()` 命令将返回一个行对象数组，我们可以使用 `get()` 来访问单个元素（列）：

```py
>>> df.collect()[0].get(0)

```

我们感兴趣在数据上执行的所有操作可能都不在 DataFrame API 中可用，因此如果需要，我们可以使用以下命令将 DataFrame 转换为行 RDD：

```py
>>> rdd_data = df.rdd

```

我们甚至可以使用以下方法将 PySpark DataFrame 转换为 Pandas DataFrame：

```py
>>> df.toPandas()

```

在后面的章节中，我们将介绍如何在 Spark 中设置应用程序和构建模型，但你现在应该能够执行许多与 Pandas 中相同的基本数据操作。

# 摘要

我们现在已经检查了许多开始构建分析应用程序所需的任务。使用 IPython 笔记本，我们介绍了如何将文件中的数据加载到 Pandas 的 DataFrame 中，重命名数据集中的列，过滤掉不需要的行，转换列数据类型，以及创建新列。此外，我们还从不同的来源合并了数据，并使用聚合和交叉操作执行了一些基本的统计分析。我们还使用直方图、散点图和密度图以及自相关和日志图来可视化数据，以及使用坐标文件在地图上叠加地理空间数据。此外，我们还使用 PySpark 处理了电影数据集，创建了 RDD 和 PySpark DataFrame，并对这些数据类型执行了一些基本操作。

我们将在未来的部分中构建这些工具，通过操作原始输入来开发用于构建预测分析管道的特征。我们还将利用类似工具来可视化和理解我们开发的预测模型的特征和性能，以及报告它们可能提供的见解。
