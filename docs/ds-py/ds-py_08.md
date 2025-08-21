# 第八章 分析学习：预测 - 金融时间序列分析与预测

> “在做出重要决策时，依靠直觉是可以的，但始终要用数据进行验证”

– *David Taieb*

时间序列的研究是数据科学中一个非常重要的领域，广泛应用于各行各业，包括天气、医学、销售以及金融等。它是一个广泛且复杂的主题，详尽讨论超出了本书的范围，但我们会在本章中简要提及一些重要的概念，并保持足够的高层次内容，不要求读者具备特定的知识。我们还将展示 Python 如何特别适合进行时间序列分析，使用诸如 pandas ([`pandas.pydata.org`](https://pandas.pydata.org))进行数据分析、NumPy ([`www.numpy.org`](http://www.numpy.org))进行科学计算，Matplotlib ([`matplotlib.org`](https://matplotlib.org))和 Bokeh ([`bokeh.pydata.org`](https://bokeh.pydata.org))进行数据可视化。

本章从 NumPy 库的介绍开始，介绍其最重要的 API，这些 API 将在构建描述性分析时得到充分应用，分析代表股票历史金融数据的时间序列。我们将使用 Python 库，如`statsmodels` ([`www.statsmodels.org/stable/index.html`](https://www.statsmodels.org/stable/index.html))，展示如何进行统计探索，找到如平稳性、**自相关函数**（**ACF**）和**偏自相关函数**（**PACF**）等属性，这些对于发现数据趋势并创建预测模型非常有用。接着，我们将通过构建一个 PixieApp 来操作这些分析，汇总股票历史金融数据的重要统计信息和可视化。

在第二部分，我们将尝试构建一个时间序列预测模型，预测股票的未来趋势。我们将使用自回归整合滑动平均模型，简称**ARIMA**，通过使用时间序列中的先前值来预测下一个值。ARIMA 是目前最流行的模型之一，尽管基于递归神经网络的新模型开始逐渐获得人气。

和往常一样，我们将在本章结束时，结合在`StockExplorer` PixieApp 中构建 ARIMA 时间序列预测模型。

# 开始使用 NumPy

NumPy 库是 Python 在数据科学家社区中获得广泛关注的主要原因之一。它是一个基础库，许多流行库的构建都依赖于它，例如 pandas ([`pandas.pydata.org`](https://pandas.pydata.org))、Matplotlib ([`matplotlib.org`](https://matplotlib.org))、SciPy ([`www.scipy.org`](https://www.scipy.org))和 scikit-learn ([`scikit-learn.org`](http://scikit-learn.org))。

NumPy 提供的关键功能包括：

+   一个功能强大的多维 NumPy 数组，称为 ndarray，具有非常高效的数学运算性能（至少与常规 Python 列表和数组相比）

+   通用函数，也简称`ufunc`，用于提供非常高效且易于使用的逐元素操作，适用于一个或多个 ndarray

+   强大的 ndarray 切片和选择功能

+   广播函数，使得在遵守某些规则的前提下，可以对不同形状的 ndarray 进行算术运算

在开始探索 NumPy APIs 之前，有一个 API 是绝对必须了解的：`lookfor()`。使用此方法，你可以通过查询字符串查找函数，这在 NumPy 提供的数百个强大 API 中非常有用。

例如，我可以查找一个计算数组平均值的函数：

```py
import numpy as np
np.lookfor("average")
```

结果如下：

```py
Search results for 'average'
----------------------------
numpy.average
    Compute the weighted average along the specified axis.
numpy.irr
    Return the Internal Rate of Return (IRR).
numpy.mean
    Compute the arithmetic mean along the specified axis.
numpy.nanmean
    Compute the arithmetic mean along the specified axis, ignoring NaNs.
numpy.ma.average
    Return the weighted average of array over the given axis.
numpy.ma.mean
    Returns the average of the array elements along given axis.
numpy.matrix.mean
    Returns the average of the matrix elements along the given axis.
numpy.chararray.mean
    Returns the average of the array elements along given axis.
numpy.ma.MaskedArray.mean
    Returns the average of the array elements along given axis.
numpy.cov
    Estimate a covariance matrix, given data and weights.
numpy.std
    Compute the standard deviation along the specified axis.
numpy.sum
    Sum of array elements over a given axis.
numpy.var
    Compute the variance along the specified axis.
numpy.sort
    Return a sorted copy of an array.
numpy.median
    Compute the median along the specified axis.
numpy.nanstd
    Compute the standard deviation along the specified axis, while
numpy.nanvar
    Compute the variance along the specified axis, while ignoring NaNs.
numpy.nanmedian
    Compute the median along the specified axis, while ignoring NaNs.
numpy.partition
    Return a partitioned copy of an array.
numpy.ma.var
    Compute the variance along the specified axis.
numpy.apply_along_axis
    Apply a function to 1-D slices along the given axis.
numpy.ma.apply_along_axis
    Apply a function to 1-D slices along the given axis.
numpy.ma.MaskedArray.var
    Compute the variance along the specified axis.
```

在几秒钟内，我可以找到几个候选函数，而不必离开我的 Notebook 去查阅文档。在之前的例子中，我可以找到一些有趣的函数——`np.average`和`np.mean`——但我仍然需要了解它们的参数。同样，我不想花时间查阅文档，打断我的工作流，因此我使用 Jupyter Notebooks 的一个不太为人知的功能，提供函数的签名和文档字符串内联显示。要调用函数的内联帮助，只需将光标定位在函数末尾并使用*Shift* + *Tab*组合键。第二次按*Shift* + *Tab*时，将展开弹出窗口，显示更多文本，如下截图所示：

### 注意

**注意**：*Shift* + *Tab* 仅适用于函数。

![开始使用 NumPy](img/B09699_08_01.jpg)

在 Jupyter Notebook 中提供内联帮助。

使用此方法，我可以快速地遍历候选函数，直到找到适合我需求的那个。

需要注意的是，`np.lookfor()`不仅限于查询 NumPy 模块；你也可以在其他模块中进行搜索。例如，以下代码在`statsmodels`包中搜索与`acf`（自相关函数）相关的方法：

```py
import statsmodels
np.lookfor("acf", module = statsmodels)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode1.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode1.py)

这将产生以下结果：

```py
Search results for 'acf'
------------------------
statsmodels.tsa.vector_ar.var_model.var_acf
    Compute autocovariance function ACF_y(h) up to nlags of stable VAR(p)
statsmodels.tsa.vector_ar.var_model._var_acf
    Compute autocovariance function ACF_y(h) for h=1,...,p
statsmodels.tsa.tests.test_stattools.TestPACF
    Set up for ACF, PACF tests.
statsmodels.sandbox.tsa.fftarma.ArmaFft.acf2spdfreq
    not really a method
statsmodels.tsa.stattools.acf
    Autocorrelation function for 1d arrays.
statsmodels.tsa.tests.test_stattools.TestACF_FFT
    Set up for ACF, PACF tests.
...
```

## 创建一个 NumPy 数组

有很多方法可以创建 NumPy 数组。以下是最常用的方法：

+   从 Python 列表或元组使用`np.array()`，例如，`np.array([1, 2, 3, 4])`。

+   来自 NumPy 工厂函数之一：

    +   `np.random`：一个提供非常丰富函数集的模块，用于随机生成值。该模块由以下几类组成：

        简单的随机数据：`rand`、`randn`、`randint`等

        排列：`shuffle`、`permutation`

        分布：`geometric`、`logistic`等

        ### 注意

        你可以在这里找到关于`np.random`模块的更多信息：

        [`docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html`](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html)

    +   `np.arange`：返回一个在给定区间内均匀分布的 ndarray。

        函数签名：`numpy.arange([start, ]stop, [step, ]dtype=None)`

        例如：`np.arange(1, 100, 10)`

        结果：`array([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91])`

    +   `np.linspace`：与`np.arange`类似，返回一个在给定区间内均匀分布的 ndarray，不同之处在于，使用`linspace`时你指定的是想要的样本数量，而不是步长数量。

        例如：`np.linspace(1,100,8, dtype=int)`

        结果：`array([ 1, 15, 29, 43, 57, 71, 85, 100])`

    +   `np.full`, `np.full_like`, `np.ones`, `np.ones_like`, `np.zeros`, `np.zeros_like`：创建一个用常数值初始化的 ndarray。

        例如：`np.ones( (2,2), dtype=int)`

        结果：`array([[1, 1], [1, 1]])`

    +   `np.eye`, `np.identity`, `np.diag`：创建一个对角线元素为常数值的 ndarray：

        例如：`np.eye(3,3)`

        结果：`array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])`

    ### 注意

    **注意**：当未提供`dtype`参数时，NumPy 会尝试从输入参数中推断数据类型。然而，可能会发生返回的类型不是正确的情况；例如，当应该是整数时，返回了浮动类型。此时，你应当使用`dtype`参数强制指定数据类型。例如：

    ```py
    np.arange(1, 100, 10, dtype=np.integer)
    ```

    为什么 NumPy 数组比 Python 的列表和数组更快？

    如前所述，NumPy 数组的操作比 Python 对应的操作要快得多。这是因为 Python 是动态语言，它不知道所处理数据的类型，因此必须不断查询与数据类型相关的元数据，以便将操作分派到正确的方法。而 NumPy 经过高度优化，能够处理大型多维数组，尤其是通过将 CPU 密集型操作委托给外部经过高度优化的 C 库，这些 C 库已被预编译。

    为了能够做到这一点，NumPy 对 ndarray 施加了两个重要的约束：

+   **ndarray 是不可变的**：因此，如果你想改变一个 ndarray 的形状或大小，或者想添加/删除元素，你必须总是创建一个新的 ndarray。例如，下面的代码使用`arange()`函数创建一个包含均匀分布值的一维数组，然后将其重塑为一个 4x5 的矩阵：

    ```py
    ar = np.arange(20)
    print(ar)
    print(ar.reshape(4,5))
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode2.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode2.py)

    结果如下：

    ```py
    before:
       [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    after: 
       [[ 0  1  2  3  4]
       [ 5  6  7  8  9]
       [10 11 12 13 14]
       [15 16 17 18 19]]
    ```

+   **ndarray 中的元素必须是相同类型**：ndarray 在其`dtype`成员中存储元素类型。当使用`nd.array()`函数创建新的 ndarray 时，NumPy 会自动推断一个适合所有元素的类型。

    例如：`np.array([1,2,3]).dtype` 将返回 `dtype('int64')`。

    `np.array([1,2,'3']).dtype` 将返回 `dtype('<U21')`，其中`<`表示小端字节序（见[`en.wikipedia.org/wiki/Endianness`](https://en.wikipedia.org/wiki/Endianness)），`U21`表示 21 个字符的 Unicode 字符串。

### 注意

**注意**：你可以在这里找到有关所有支持的数据类型的详细信息：

[`docs.scipy.org/doc/numpy/reference/arrays.dtypes.html`](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)

## 对 ndarray 的操作

通常，我们需要对 ndarray 进行数据汇总。幸运的是，NumPy 提供了一套非常丰富的函数（也称为**归约函数**），可以直接对 ndarray 或 ndarray 的轴进行汇总。

作为参考，NumPy 的轴对应数组的一个维度。例如，一个二维 ndarray 有两个轴：一个沿行方向，称为轴 0；一个沿列方向，称为轴 1。

以下图示展示了二维数组中的轴：

![Operations on ndarray](img/B09699_08_02.jpg)

二维数组中的轴

接下来我们讨论的大多数归约函数都将接受轴作为参数。它们分为以下几类：

+   **数学函数**：

    +   三角函数：`np.sin`，`np.cos`，等等

    +   双曲线函数：`np.sinh`，`np.cosh`，等等

    +   四舍五入：`np.around`，`np.floor`，等等

    +   和、积、差：`np.sum`，`np.prod`，`np.cumsum`，等等

    +   指数和对数：`np.exp`，`np.log`，等等

    +   算术运算：`np.add`，`np.multiply`，等等

    +   杂项：`np.sqrt`，`np.absolute`，等等

    ### 注意

    **注意**：所有这些一元函数（只接受一个参数的函数）都直接作用于 ndarray。例如，我们可以使用`np.square`一次性对数组中的所有值进行平方：

    代码：`np.square(np.arange(10))`

    结果：`array([ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81])`

    你可以在这里找到更多关于 NumPy 数学函数的信息：

    [`docs.scipy.org/doc/numpy/reference/routines.math.html`](https://docs.scipy.org/doc/numpy/reference/routines.math.html)

+   **统计函数**：

    +   顺序统计：`np.amin`，`np.amax`，`np.percentile`，等等

    +   平均数和方差：`np.median`，`np.var`，`np.std`，等等

    +   相关性：`np.corrcoef`，`np.correlate`，`np.cov`，等等

    +   直方图：`np.histogram`，`np.bincount`，等等

### 注意

**注意**：pandas 与 NumPy 紧密集成，让你能够在 pandas 的 DataFrame 上应用这些 NumPy 操作。在本章接下来的时间序列分析中，我们将充分利用这一功能。

以下代码示例创建一个 pandas DataFrame 并对所有列进行平方运算：

![操作示例](img/B09699_08_03.jpg)

将 NumPy 操作应用于 pandas DataFrame

## NumPy 数组的选择

NumPy 数组支持与 Python 数组和列表类似的切片操作。因此，使用 `np.arrange()` 方法创建的 ndarray，我们可以执行以下操作：

```py
sample = np.arange(10)
print("Sample:", sample)
print("Access by index: ", sample[2])
print("First 5 elements: ", sample[:5])
print("From 8 to the end: ", sample[8:])
print("Last 3 elements: ", sample[-3:])
print("Every 2 elements: ", sample[::2])
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode3.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode3.py)

这将产生以下结果：

```py
Sample: [0 1 2 3 4 5 6 7 8 9]
Access by index:  2
First 5 elements:  [0 1 2 3 4]
From index 8 to the end:  [8 9]
Last 3 elements:  [7 8 9]
Every 2 elements:  [0 2 4 6 8]
```

使用切片进行选择也适用于具有多个维度的 NumPy 数组。我们可以对数组的每个维度使用切片操作。而这在 Python 数组和列表中不可行，它们仅允许使用整数或切片进行索引。

### 注意

**注意**：作为参考，Python 中的切片语法如下：

```py
start:end:step
```

作为示例，让我们创建一个形状为 `(3,4)` 的 NumPy 数组，即 3 行 * 4 列：

```py
my_nparray = np.arange(12).reshape(3,4)
print(my_nparray)
```

返回：

```py
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
```

假设我想选择矩阵的中间部分，即 [5, 6]。我可以简单地对行和列应用切片，例如，`[1:2]` 选择第二行，`[1:3]` 选择第二行的第二和第三个值：

```py
print(my_nparray[1:2, 1:3])
```

返回：

```py
array([[5, 6]])
```

另一个有趣的 NumPy 特性是我们还可以使用谓词来用布尔值对 ndarray 进行索引。

例如：

```py
print(sample > 5 )
```

返回：

```py
[False False False False False False  True  True  True  True]
```

然后，我们可以使用布尔 ndarray 以简单优雅的语法选择数据的子集。

例如：

```py
print( sample[sample > 5] )
```

返回：

```py
[6 7 8 9]
```

### 注意

这是 NumPy 所有选择能力的一个小预览。有关更多 NumPy 选择的信息，您可以访问：

[`docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html`](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html)

## 广播

广播是 NumPy 的一个非常方便的特性，它允许你对具有不同形状的 ndarrays 执行算术运算。**广播**这个术语来源于这样一个事实：较小的数组会自动被复制以适应较大的数组，从而使它们具有兼容的形状。然而，有一套规则决定了广播如何工作。

### 注意

您可以在这里找到更多关于广播的信息：

[`docs.scipy.org/doc/numpy/user/basics.broadcasting.html`](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

NumPy 广播的最简单形式是**标量广播**，它允许你在 ndarray 和标量（即数字）之间执行逐元素的算术操作。

例如：

```py
my_nparray * 2
```

返回：

```py
array([[ 0,  2,  4,  6],
       [ 8, 10, 12, 14],
       [16, 18, 20, 22]])
```

### 注意

**注意**：在接下来的讨论中，我们假设我们要操作的两个 ndarray 具有不同的维度。

广播操作对于较小的数组仅需要遵循一个规则：其中一个数组必须至少有一个维度等于 1。其思想是沿着不匹配的维度复制较小的数组，直到它们的维度一致。

以下图示来自 [`www.scipy-lectures.org/`](http://www.scipy-lectures.org/) 网站，生动地展示了两数组相加的不同情况：

![Broadcasting](img/B09699_08_04.jpg)

广播流的解释

来源：[`www.scipy-lectures.org/_images/numpy_broadcasting.png`](http://www.scipy-lectures.org/_images/numpy_broadcasting.png)

上图展示的三种使用情况是：

+   **数组的维度匹配**：按常规执行元素级的求和。

+   **较小的数组只有 1 行**：复制行，直到维度与第一个数组匹配。如果较小的数组只有 1 列，则会使用相同的算法。

+   **第一个数组只有 1 列，第二个数组只有 1 行**：

    +   将第一个数组中的列复制，直到列数与第二个数组相同。

    +   将第二个数组中的行复制，直到行数与第一个数组相同。

以下代码示例展示了 NumPy 广播的实际应用：

```py
my_nparray + np.array([1,2,3,4])
```

结果：

```py
array([[ 1,  3,  5,  7],
       [ 5,  7,  9, 11],
       [ 9, 11, 13, 15]])
```

本节中，我们提供了 NumPy 的基础介绍，至少足够让我们入门并跟随接下来章节中涵盖的代码示例。在下一节中，我们将开始讨论时间序列，并通过统计数据探索来寻找模式，帮助我们识别数据中的潜在结构。

# 时间序列的统计探索

对于示例应用，我们将使用 Quandl 数据平台金融 API 提供的股票历史金融数据 ([`www.quandl.com/tools/api`](https://www.quandl.com/tools/api)) 和 `quandl` Python 库 ([`www.quandl.com/tools/python`](https://www.quandl.com/tools/python))。

要开始使用，我们需要通过在独立的单元格中运行以下命令来安装 `quandl` 库：

```py
!pip install quandl

```

### 注意

**注意**：和往常一样，安装完成后不要忘记重启内核。

访问 Quandl 数据是免费的，但每天限 50 次请求，不过你可以通过创建一个免费账户并获取 API 密钥来绕过此限制：

1.  访问 [`www.quandl.com`](https://www.quandl.com)，通过点击右上角的 **SIGN UP** 按钮创建一个新账户。

1.  在注册向导的三步中填写表单。（我选择了 **Personal**，但根据你的情况，你可能想选择 **Business** 或 **Academic**。）

1.  完成过程后，你应该会收到一封包含激活账户链接的电子邮件确认。

1.  账户激活后，登录 Quandl 平台网站，在右上角菜单点击 **Account Settings**，然后转到 **API KEY** 标签。

1.  复制本页面提供的 API 密钥。此值将用于在 `quandl` Python 库中通过编程方式设置密钥，如以下代码所示：

    ```py
    import quandl
    quandl.ApiConfig.api_key = "YOUR_KEY_HERE"
    ```

`quandl` 库主要由两个 API 组成：

+   `quandl.get(dataset, **kwargs)`：这将返回一个 pandas DataFrame 或一个 NumPy 数组，表示请求的数据集。`dataset`参数可以是一个字符串（单一数据集）或一个字符串列表（多个数据集）。每个数据集遵循`database_code/dataset_code`的语法，其中`database_code`是数据发布者，`dataset_code`与资源相关。（接下来我们将介绍如何获取所有`database_code`和`dataset_code`的完整列表）。

    关键字参数使你能够精细化查询。你可以在 GitHub 上的`quandl`代码中找到支持的所有参数的完整列表：[`github.com/quandl/quandl-python/blob/master/quandl/get.py`](https://github.com/quandl/quandl-python/blob/master/quandl/get.py)。

    一个有趣的关键字参数`returns`控制方法返回的数据结构，它可以取以下两个值：

    +   `pandas`：返回一个 pandas DataFrame

    +   `numpy`：返回一个 NumPy 数组

+   `quandl.get_table(datatable_code, **kwargs)`：返回一个非时间序列数据集（称为`datatable`），用于描述某个资源。在本章中我们不会使用这个方法，但你可以通过查看代码了解更多：[`github.com/quandl/quandl-python/blob/master/quandl/get_table.py`](https://github.com/quandl/quandl-python/blob/master/quandl/get_table.py)。

为了获取`database_code`的列表，我们使用 Quandl REST API：`https://www.quandl.com/api/v3/databases?api_key=YOUR_API_KEY&page=n`，它使用了分页功能。

### 注意

**注意**：在前面的 URL 中，将`YOUR_API_KEY`值替换为你实际的 API 密钥。

返回的 payload 是以下 JSON 格式：

```py
{
  "databases": [{
         "id": 231,
         "name": "Deutsche Bundesbank Data Repository",
         "database_code": "BUNDESBANK",
         "description": "Data on the German economy, ...",
         "datasets_count": 49358,
         "downloads": 43209922,
         "premium": false,
         "image": "https://quandl--upload.s3.amazonaws/...thumb_bundesbank.png",
         "favorite": false,
         "url_name": "Deutsche-Bundesbank-Data-Repository"
       },...
],
  "meta": {
    "query": "",
    "per_page": 100,
    "current_page": 1,
    "prev_page": null,
    "total_pages": 3,
    "total_count": 274,
    "next_page": 2,
    "current_first_item": 1,
    "current_last_item": 100
  }
}
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode4.json`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode4.json)

我们使用`while`循环加载所有可用的页面，依赖于`payload['meta']['next_page']`值来判断何时停止。在每次迭代中，我们将`database_code`信息的列表追加到一个名为`databases`的数组中，如下所示：

```py
import requests
databases = []
page = 1
while(page is not None):
    payload = requests.get("https://www.quandl.com/api/v3/databases?api_key={}&page={}"\
                    .format(quandl.ApiConfig.api_key, page)).json()
 databases += payload['databases']
 page = payload['meta']['next_page']

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode5.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode5.py)

`databases`变量现在包含一个包含每个`database_code`元数据的 JSON 对象数组。我们使用 PixieDust 的`display()` API 以漂亮的可搜索表格形式查看数据：

```py
import pixiedust
display(databases)
```

在下面的 PixieDust 表格截图中，我们使用第二章中描述的**筛选**按钮，*用 Python 和 Jupyter Notebook 驱动数据分析*，来查看每个数据库中可用数据集的统计信息，例如最小值、最大值和均值：

![时间序列的统计探索](img/B09699_08_05.jpg)

Quandl 数据库代码列表

在寻找包含 **纽约证券交易所**（**NYSE**）股票信息的数据库时，我找到了 `XNYS` 数据库，如下所示：

### 注意

**注意**：确保在图表选项对话框中将显示的值数量增加到 `300`，以便所有结果都能在表格中显示。

![时间序列的统计探索](img/B09699_08_06.jpg)

寻找包含纽约证券交易所（NYSE）股票数据的数据库

不幸的是，`XNYS` 数据库是私有的，需要付费订阅。我最终使用了 `WIKI` 数据库代码，尽管它没有出现在之前 API 请求返回的列表中，但我在一些代码示例中找到了它。

我随后使用 `https://www.quandl.com/api/v3/databases/{database_code}/codes` REST API 获取数据集列表。幸运的是，这个 API 返回的是一个压缩在 ZIP 文件中的 CSV 文件，PixieDust 的 `sampleData()` 方法可以轻松处理，如下代码所示：

```py
codes = pixiedust.sampleData( "https://www.quandl.com/api/v3/databases/WIKI/codes?api_key=" + quandl.ApiConfig.api_key)
display(codes)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode6.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode6.py)

在 PixieDust 表格界面中，我们点击 **选项** 对话框，将显示的值数量增加到 `4000`，以便可以显示整个数据集（共有 3,198 条），并使用搜索框查找特定的股票，如下图所示：

### 注意

**注意**：搜索框只会搜索浏览器中显示的行，当数据集过大时，这可能是一个较小的子集。由于本例中数据集过大，增加显示的行数并不实际；因此，建议使用 **筛选器**，它可以确保查询整个数据集。

`quandl` API 返回的 CSV 文件没有表头，但 `PixieDust.sampleData()` 期望文件中包含表头。这是当前的一个限制，将来会进行改进。

![时间序列的统计探索](img/B09699_08_07.jpg)

WIKI 数据库的数据集列表

在接下来的部分中，我们加载了微软（Microsoft）股票（股票代码 MSFT）过去几年的历史时间序列数据，并开始探索其统计属性。在以下代码中，我们使用 `quandl.get()` 来获取 `WIKI/MSFT` 数据集。我们添加了一个名为 `daily_spread` 的列，通过调用 pandas 的 `diff()` 方法计算每日涨跌，这个方法返回当前调整后的收盘价与前一个收盘价之间的差异。请注意，返回的 pandas DataFrame 使用日期作为索引，但 PixieDust 当前不支持按索引绘制时间序列图表。因此，在以下代码中，我们调用 `reset_index()` 将 `DateTime` 索引转换为一个名为 `Date` 的新列，其中包含日期信息：

```py
msft = quandl.get('WIKI/MSFT')
msft['daily_spread'] = msft['Adj. Close'].diff()
msft = msft.reset_index()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode7.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode7.py)

对于我们的第一次数据探索，我们使用`display()`创建了一个基于 Bokeh 渲染器的股票调整后收盘价随时间变化的折线图。

以下截图显示了**选项**配置及其生成的折线图：

![时间序列的统计探索](img/B09699_08_08.jpg)

MSFT 股价随时间变化，已调整股息分配、股票拆分及其他公司行为

我们还可以生成一个图表，显示该期间每天的收益，以下截图展示了该图表：

![时间序列的统计探索](img/B09699_08_09.jpg)

MSFT 股票的每日收益

## 假设投资

作为练习，我们尝试创建一个图表，显示在所选股票（MSFT）中，假设投资 10,000 美元随着时间的推移会如何变化。为此，我们必须计算一个数据框，包含每一天的总投资价值，考虑到我们在上一段计算的每日收益，并使用 PixieDust 的`display()` API 来可视化数据。

我们利用 pandas 的能力，根据日期条件选择行，首先过滤数据框，只选择我们感兴趣的时间段内的数据点。然后通过将初始投资 10,000 美元除以该期间第一天的收盘价来计算购买的股票数量，并加上初始投资价值。所有这些计算都变得非常简单，得益于 pandas 高效的系列计算和底层的 NumPy 基础库。我们使用`np.cumsum()`方法（[`docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.cumsum.html`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.cumsum.html)）来计算所有每日收益的累积和，并加上初始投资金额 10,000 美元。

最后，我们通过使用`resample()`方法将频率从每日转换为每月，并使用该月的平均值计算新值，从而使图表更易于阅读。

以下代码计算了从 2016 年 5 月开始的增长数据框（DataFrame）：

```py
import pandas as pd
tail = msft[msft['Date'] > '2016-05-16']
investment = np.cumsum((10000 / tail['Adj. Close'].values[0]) * tail['daily_spread']) + 10000
investment = investment.astype(int)
investment.index = tail['Date']
investment = investment.resample('M').mean()
investment = pd.DataFrame(investment).reset_index()
display(investment)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode8.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode8.py)

以下截图显示了`display()` API 生成的图表，包括配置选项：

![假设投资](img/B09699_08_10.jpg)

假设投资组合增长

## 自相关函数（ACF）和偏自相关函数（PACF）

在尝试生成预测模型之前，了解时间序列是否具有可识别的模式（如季节性或趋势）是至关重要的。一种常见的技术是查看数据点如何根据指定的时间滞后与前一个数据点进行关联。直观地说，自相关性会揭示内部结构，例如识别高相关性（正相关或负相关）发生的时期。你可以尝试不同的滞后值（也就是对于每个数据点，你会考虑多少个之前的数据点）来找到合适的周期性。

计算自相关函数通常需要计算一组数据点的皮尔逊相关系数（[`en.wikipedia.org/wiki/Pearson_correlation_coefficient`](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)），这并不是一件简单的事情。好消息是，`statsmodels` Python 库提供了一个`tsa`包（**tsa**代表**时间序列分析**），它提供了紧密集成于 pandas Series 的辅助方法，用于计算自相关函数（ACF）。

### 注意

**注意**：如果尚未完成，请使用以下命令安装`statsmodels`包，并在完成后重启内核：

```py
!pip install statsmodels
```

以下代码使用来自`tsa.api.graphics`包的`plot_acf()`函数来计算并可视化 MSFT 股票时间序列的自相关函数（ACF）：

```py
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
smt.graphics.plot_acf(msft['Adj. Close'], lags=100)
plt.show()
```

以下是结果：

![自相关函数（ACF）和偏自相关函数（PACF）](img/B09699_08_11.jpg)

滞后为 100 时的 MSFT 自相关函数（ACF）

上面的图表展示了数据在多个前期数据点（滞后）上的自相关性，这些滞后值由*x*坐标轴表示。因此，在滞后为`0`时，你总是会有一个自相关值为`1.0`（你总是与你自己完全相关），滞后`1`显示的是与前一个数据点的自相关性，滞后`2`显示的是与两个数据点之前的自相关性。我们可以清楚地看到，随着滞后的增加，自相关性逐渐下降。在上面的图表中，我们只使用了 100 个滞后，并且看到自相关性仍然在 0.9 左右，表明数据间的长时间间隔并没有关联。这表明数据存在趋势，从整体价格图上来看，这一点非常明显。

为了验证这个假设，我们绘制了一个具有更大`lags`参数的 ACF 图，例如`1000`（考虑到我们的数据序列有超过 10,000 个数据点，这并不为过），如下图所示：

![自相关函数（ACF）和偏自相关函数（PACF）](img/B09699_08_12.jpg)

滞后为 1000 时的 MSFT 自相关函数（ACF）

我们现在可以清楚地看到，自相关性在约`600`个滞后值时低于显著性水平。

为了更好地说明 ACF 如何工作，让我们生成一个周期性的时间序列，该序列没有趋势，看看我们可以学到什么。例如，我们可以在用 `np.linspace()` 生成的一系列均匀间隔的点上使用 `np.cos()`：

```py
smt.graphics.plot_acf(np.cos(np.linspace(0, 1000, 100)), lags=50)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode9.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode9.py)

结果如下：

![自相关函数（ACF）和偏自相关函数（PACF）](img/B09699_08_13.jpg)

没有趋势的周期性序列的 ACF

在前面的图表中，我们可以看到自相关再次在固定间隔（大约每 5 个滞后期）处出现峰值，清晰地显示了周期性（在处理真实世界数据时，这也称为季节性）。

使用 ACF 来检测时间序列中的结构有时会导致问题，尤其是当你有强周期性时。在这种情况下，无论你试图多么往回计算自相关，总会看到自相关在周期的倍数处出现一个峰值，这可能会导致错误的解释。为了解决这个问题，我们使用 PACF，它使用较短的滞后期，且与 ACF 不同，它不会重复使用之前在较短时间段内发现的相关性。ACF 和 PACF 的数学原理相当复杂，但读者只需要理解其背后的直觉，并愉快地使用像 `statsmodels` 这样的库来进行繁重的计算。我获取有关 ACF 和 PACF 更多信息的一个资源可以在这里找到：[`www.mathworks.com/help/econ/autocorrelation-and-partial-autocorrelation.html`](https://www.mathworks.com/help/econ/autocorrelation-and-partial-autocorrelation.html)。

回到我们的 MSFT 股票时间序列，以下代码展示了如何使用 `smt.graphics` 包绘制其 PACF：

```py
import statsmodels.tsa.api as smt
smt.graphics.plot_pacf(msft['Adj. Close'], lags=50)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode10.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode10.py)

结果显示在下面的截图中：

![自相关函数（ACF）和偏自相关函数（PACF）](img/B09699_08_14.jpg)

MSFT 股票时间序列的偏自相关

我们将在本章后面讨论 ARIMA 模型进行时间序列预测时，再回到 ACF 和 PACF。

在本节中，我们讨论了多种探索数据的方法。当然，这些方法绝非详尽无遗，但我们可以看到像 Jupyter、pandas、NumPy 和 PixieDust 等工具如何让实验变得更容易，并在必要时快速失败。在下一节中，我们将构建一个 PixieApp，将所有这些图表汇聚在一起。

# 将一切整合在一起，使用 StockExplorer PixieApp

对于我们 `StockExplorer` PixieApp 的第一个版本，我们希望实现用户选择的股票数据时间序列的数据探索。与我们构建的其他 PixieApps 类似，第一个屏幕有一个简单的布局，包含一个输入框，用户可以在其中输入以逗号分隔的股票代码列表，并有一个 **Explore** 按钮来开始数据探索。主屏幕由一个垂直导航栏组成，每个数据探索类型都有一个菜单。为了使 PixieApp 代码更加模块化，并且更易于维护和扩展，我们将在自己的子 PixieApp 中实现每个数据探索屏幕，这些子 PixieApp 通过垂直导航栏触发。同时，每个子 PixieApp 都继承自一个名为 `BaseSubApp` 的基类，提供所有子类共有的功能。以下图显示了整体 UI 布局以及所有子 PixieApps 的类图：

![将所有内容整合在一起，使用 StockExplorer PixieApp](img/B09699_08_15.jpg)

股票探索 PixieApp 的 UI 布局

我们先来看一下欢迎屏幕的实现。它是在 `StockExplorer` PixieApp 类的默认路由中实现的。以下代码显示了 `StockExplorer` 类的部分实现，仅包括默认路由。

### 注意

**注意**：在提供完整实现之前，不要尝试运行此代码。

```py
@PixieApp
class StockExplorer():
    @route()
    def main_screen(self):
        return """
<style>
    div.outer-wrapper {
        display: table;width:100%;height:300px;
    }
    div.inner-wrapper {
        display: table-cell;vertical-align: middle;height: 100%;width: 100%;
    }
</style>
<div class="outer-wrapper">
    <div class="inner-wrapper">
        <div class="col-sm-3"></div>
        <div class="input-group col-sm-6">
          <input id="stocks{{prefix}}" type="text"
              class="form-control"
              value="MSFT,AMZN,IBM"
              placeholder="Enter a list of stocks separated by comma e.g MSFT,AMZN,IBM">
          <span class="input-group-btn">
 <button class="btn btn-default" type="button" pd_options="explore=true">
                <pd_script>
self.select_tickers('$val(stocks{{prefix}})'.split(','))
                </pd_script>
                Explore
            </button>
          </span>
        </div>
    </div>
</div>
"""
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode11.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode11.py)

上述代码与我们迄今为止看到的其他示例 PixieApps 非常相似。**Explore** 按钮包含以下两个 PixieApp 属性：

+   一个 `pd_script` 子元素，它调用一个 Python 代码片段来设置股票代码。我们还使用 `$val` 指令来获取用户输入的股票代码值：

    ```py
    <pd_script>
       self.select_tickers('$val(stocks{{prefix}})'.split(','))
    </pd_script>
    ```

+   `pd_options` 属性，它指向 `explore` 路由：

    ```py
    pd_options="explore=true"
    ```

`select_tickers` 辅助方法将股票代码列表存储在字典成员变量中，并选择第一个作为活动股票代码。出于性能考虑，我们只在需要时加载数据，也就是在第一次设置活动股票代码时，或者当用户在 UI 中点击特定股票代码时。

### 注意

**注意**：与前几章一样，`[[StockExplorer]]` 表示以下代码是 `StockExplorer` 类的一部分。

```py
[[StockExplorer]]
def select_tickers(self, tickers):
        self.tickers = {ticker.strip():{} for ticker in tickers}
        self.set_active_ticker(tickers[0].strip())

def set_active_ticker(self, ticker):
    self.active_ticker = ticker
 if 'df' not in self.tickers[ticker]:
        self.tickers[ticker]['df'] = quandl.get('WIKI/{}'.format(ticker))
        self.tickers[ticker]['df']['daily_spread'] = self.tickers[ticker]['df']['Adj. Close'] - self.tickers[ticker]['df']['Adj. Open']
        self.tickers[ticker]['df'] = self.tickers[ticker]['df'].reset_index()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode12.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode12.py)

特定股票代码的数据懒加载到 pandas DataFrame 是在`set_active_ticker()`中完成的。我们首先检查 DataFrame 是否已经加载，通过查看`df`键是否存在，如果不存在，我们调用`quandl` API，传入`dataset_code`：`'WIKI/{ticker}'`。我们还添加了一个列，用于计算股票的每日波动，这将在基本探索界面中显示。最后，我们需要调用`reset_index()`（[`pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html)）对 DataFrame 进行处理，将索引（`DateTimeIndex`）转换为名为`Date`的列。原因是 PixieDust 的`display()`尚不支持可视化包含`DateTimeIndex`的 DataFrame。

在`explore`路由中，我们返回一个构建整个屏幕布局的 HTML 片段。如前面的示意图所示，我们使用`btn-group-vertical`和`btn-group-toggle`的 Bootstrap 类来创建垂直导航栏。菜单列表及其关联的子 PixieApp 定义在`tabs` Python 变量中，并且我们使用 Jinja2 的`{%for loop%}`来构建内容。我们还添加了一个占位符`<div>`元素，`id ="analytic_screen{{prefix}}"`，它将成为子 PixieApp 屏幕的接收容器。

`explore`路由的实现如下所示：

```py
[[StockExplorer]] 
@route(explore="*")
 @templateArgs
    def stock_explore_screen(self):
 tabs = [("Explore","StockExploreSubApp"),
 ("Moving Average", "MovingAverageSubApp"),
 ("ACF and PACF", "AutoCorrelationSubApp")]
        return """
<style>
    .btn:active, .btn.active {
        background-color:aliceblue;
    }
</style>
<div class="page-header">
    <h1>Stock Explorer PixieApp</h1>
</div>
<div class="container-fluid">
    <div class="row">
        <div class="btn-group-vertical btn-group-toggle col-sm-2"
             data-toggle="buttons">
 {%for title, subapp in tabs%}
            <label class="btn btn-secondary {%if loop.first%}active{%endif%}"
                pd_options="show_analytic={{subapp}}"
                pd_target="analytic_screen{{prefix}}">
                <input type="radio" {%if loop.first%}checked{%endif%}>
                    {{title}}
            </label>
 {%endfor%}
        </div>
        <div id="analytic_screen{{prefix}}" class="col-sm-10">
    </div>
</div>
"""
```

### 注意

你可以在此处找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode13.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode13.py)

在前面的代码中，请注意我们使用了`@templateArgs`装饰器，因为我们想在 Jinja2 模板中使用方法实现局部创建的`tabs`变量。

垂直导航栏中的每个菜单都指向相同的`analytic_screen{{prefix}}`目标，并通过`{{subapp}}`引用的选定子 PixieApp 类名调用`show_analytic`路由。

反过来，`show_analytic`路由仅返回一个包含`<div>`元素的 HTML 片段，该元素具有`pd_app`属性，引用子 PixieApp 类名。我们还使用`pd_render_onload`属性，要求 PixieApp 在浏览器 DOM 加载时立即渲染`<div>`元素的内容。

以下代码用于`show_analytic`路由：

```py
    @route(show_analytic="*")
    def show_analytic_screen(self, show_analytic):
        return """
<div pd_app="{{show_analytic}}" pd_render_onload></div>
"""
```

### 注意

你可以在此处找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode14.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode14.py)

## BaseSubApp – 所有子 PixieApp 的基类

现在让我们来看看每个子 PixieApp 的实现，以及如何使用基类`BaseSubApp`来提供共享功能。对于每个子 PixieApp，我们希望用户能够通过标签页界面选择股票代码，如下图所示：

![BaseSubApp – 所有子 PixieApp 的基类](img/B09699_08_16.jpg)

MSFT、IBM、AMZN 股票代码的选项卡小部件

为了避免为每个子 PixieApp 重复 HTML 片段，我们使用了一种我特别喜欢的技术，即创建一个名为`add_ticker_selection_markup`的 Python 装饰器，它动态改变函数的行为（有关 Python 装饰器的更多信息，请参见 [`wiki.python.org/moin/PythonDecorators`](https://wiki.python.org/moin/PythonDecorators)）。这个装饰器是在`BaseSubApp`类中创建的，并且会自动为路由预先附加选项卡选择小部件的 HTML 标记，代码如下所示：

```py
[[BaseSubApp]]
def add_ticker_selection_markup(refresh_ids):
    def deco(fn):
        def wrap(self, *args, **kwargs):
            return """
<div class="row" style="text-align:center">
 <div class="btn-group btn-group-toggle"
 style="border-bottom:2px solid #eeeeee"
 data-toggle="buttons">
 {%for ticker, state in this.parent_pixieapp.tickers.items()%}
 <label class="btn btn-secondary {%if this.parent_pixieapp.active_ticker == ticker%}active{%endif%}"
 pd_refresh=\"""" + ",".join(refresh_ids) + """\" pd_script="self.parent_pixieapp.set_active_ticker('{{ticker}}')">
 <input type="radio" {%if this.parent_pixieapp.active_ticker == ticker%}checked{%endif%}> 
 {{ticker}}
 </label>
 {%endfor%}
 </div>
</div>
            """ + fn(self, *args, **kwargs)
        return wrap
    return deco
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode15.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode15.py)

初看之下，上面的代码可能看起来很难读，因为`add_ticker_selection_markup`装饰器方法包含了两层匿名嵌套方法。我们来尝试解释它们的目的，包括主要的`add_ticker_selection_markup`装饰器方法：

+   `add_ticker_selection_markup`：这是主要的装饰器方法，它接受一个名为`refresh_ids`的参数，这个参数将在生成的标记中使用。该方法返回一个名为`deco`的匿名函数，`deco`接收一个函数参数。

+   `deco`：这是一个包装方法，它接受一个名为`fn`的参数，该参数是指向原始函数的指针，装饰器应用于该函数。当用户代码中调用该函数时，该方法返回一个名为`wrap`的匿名函数，`wrap`将在原始函数的地方被调用。

+   `wrap`：这是最终的包装方法，它接受三个参数：

    +   `self`：指向函数所属类的指针

    +   `*args`：原方法定义的任意可变参数（可能为空）

    +   `**kwargs`：原方法定义的任意关键字参数（可能为空）

    `wrap`方法可以通过 Python 闭包机制访问其作用域外的变量。在这种情况下，它使用`refresh_ids`生成选项卡小部件的标记，然后用`self`、`args`和`kwargs`参数调用`fn`函数。

### 注意

**注意**：如果上述解释看起来依然让你感到困惑，即使你读了多次，也不用担心。你现在只需使用这个装饰器，它不会影响你理解本章的其他内容。

## StockExploreSubApp – 第一个子 PixieApp

现在我们可以实现第一个子 PixieApp，名为`StockExploreSubApp`。在主屏幕中，我们创建了两个`<div>`元素，每个元素都有一个`pd_options`属性，该属性调用`show_chart`路由，并将`Adj. Close`和`daily_spread`作为值。然后，`show_chart`路由返回一个`<div>`元素，其中`pd_entity`属性指向`parent_pixieapp.get_active_df()`方法，并且包含一个`<pd_options>`元素，里面包含一个 JSON 负载，用于显示一个 Bokeh 线图，`Date`作为* x *轴，任何作为参数传递的值作为* y *轴的列。我们还用`BaseSubApp.add_ticker_selection_markup`装饰器装饰了该路由，使用前面两个`<div>`元素的 ID 作为`refresh_ids`参数。

以下代码显示了`StockExplorerSubApp`子 PixieApp 的实现：

```py
@PixieApp
class StockExploreSubApp(BaseSubApp):
    @route()
 @BaseSubApp.add_ticker_selection_markup(['chart{{prefix}}', 'daily_spread{{prefix}}'])
    def main_screen(self):
        return """
<div class="row" style="min-height:300px">
    <div class="col-xs-6" id="chart{{prefix}}" pd_render_onload pd_options="show_chart=Adj. Close">
    </div>
    <div class="col-xs-6" id="daily_spread{{prefix}}" pd_render_onload pd_options="show_chart=daily_spread">
    </div>
</div>
"""

    @route(show_chart="*")
    def show_chart_screen(self, show_chart):
        return """
<div pd_entity="parent_pixieapp.get_active_df()" pd_render_onload>
    <pd_options>
    {
      "handlerId": "lineChart",
      "valueFields": "{{show_chart}}",
      "rendererId": "bokeh",
      "keyFields": "Date",
      "noChartCache": "true",
      "rowCount": "10000"
    }
    </pd_options>
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode16.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode16.py)

在前面的`show_chart`路由中，`pd_entity`使用了`parent_pixieapp`中定义的`get_active_df()`方法，该方法在`StockExplorer`主类中定义，代码如下：

```py
[[StockExplorer]]
def get_active_df(self):
    return self.tickers[self.active_ticker]['df']
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode17.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode17.py)

提醒一下，`StockExploreSubApp`通过`StockExplorer`路由中的`Explore`路由内声明的`tabs`数组变量中的元组与菜单关联：

```py
tabs = [("Explore","StockExploreSubApp"), ("Moving Average", "MovingAverageSubApp"),("ACF and PACF", "AutoCorrelationSubApp")]
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode18.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode18.py)

下图显示了`StockExploreSubApp`：

![StockExploreSubApp – 第一个子 PixieApp](img/B09699_08_17.jpg)

StockExploreSubApp 主屏幕

## MovingAverageSubApp – 第二个子 PixieApp

第二个子 PixieApp 是`MovingAverageSubApp`，它展示了所选股票代码的移动平均线图，并且可以通过滑块控件配置延迟。与股票选择标签类似，延迟滑块将在另一个子 PixieApp 中使用。我们可以使用与股票选择标签控件相同的装饰器技术，但这里我们希望能将延迟滑块放置在页面的任何位置。因此，我们将使用一个在`BaseSubApp`类中定义的`pd_widget`控件，名为`lag_slider`，并返回一个用于滑块控件的 HTML 片段。它还添加了一个`<script>`元素，使用 jQuery UI 模块中可用的 jQuery `slider`方法（有关更多信息，请参见[`api.jqueryui.com/slider`](https://api.jqueryui.com/slider)）。我们还添加了一个`change`事件处理函数，当用户选择新值时会被调用。在这个事件处理程序中，我们调用`pixiedust.sendEvent`函数，发布一个类型为`lagSlider`的事件，并包含新的延迟值的有效载荷。调用者有责任添加一个`<pd_event_handler>`元素来监听该事件并处理有效载荷。

以下代码展示了`lag_slider` `pd_widget`的实现

```py
[[BaseSubApp]]
@route(widget="lag_slider")
def slider_screen(self):
    return """
<div>
    <label class="field">Lag:<span id="slideval{{prefix}}">50</span></label>
    <i class="fa fa-info-circle" style="color:orange"
       data-toggle="pd-tooltip"
       title="Selected lag used to compute moving average, ACF or PACF"></i>
    <div id="slider{{prefix}}" name="slider" data-min=30 
         data-max=300
         data-default=50 style="margin: 0 0.6em;">
    </div>
</div>
<script>
$("[id^=slider][id$={{prefix}}]").each(function() {
    var sliderElt = $(this)
    var min = sliderElt.data("min")
    var max = sliderElt.data("max")
    var val = sliderElt.data("default")
 sliderElt.slider({
        min: isNaN(min) ? 0 : min,
        max: isNaN(max) ? 100 : max,
        value: isNaN(val) ? 50 : val,
        change: function(evt, ui) {
            $("[id=slideval{{prefix}}]").text(ui.value);
            pixiedust.sendEvent({type:'lagSlider',value:ui.value})
        },
        slide: function(evt, ui) {
            $("[id=slideval{{prefix}}]").text(ui.value);
        }
    });
})
</script>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode19.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode19.py)

在`MovingAverageSubApp`中，我们在默认路由中使用`add_ticker_selection_markup`装饰器，并以`chart{{prefix}}`作为参数，添加股票选择标签，并添加一个名为`lag_slider`的`pd_widget`的`<div>`元素，包括一个`<pd_event_handler>`来设置`self.lag`变量并刷新`chart` div。`chart` div 使用`pd_entity`属性，并调用`get_moving_average_df()`方法，该方法调用来自所选 pandas DataFrame 的 pandas Series 上的`rolling`方法（[`pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rolling.html`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rolling.html)）并对其调用`mean()`方法。由于 PixieDust 的`display()`尚不支持 pandas Series，我们使用 series 索引构建一个 pandas DataFrame，作为一个名为`x`的列，并在`get_moving_average_df()`方法中返回它。

以下代码展示了`MovingAverageSubApp`子 PixieApp 的实现

```py
@PixieApp
class MovingAverageSubApp(BaseSubApp):
    @route()
 @BaseSubApp.add_ticker_selection_markup(['chart{{prefix}}'])
    def main_screen(self):
        return """
<div class="row" style="min-height:300px">
    <div class="page-header text-center">
        <h1>Moving Average for {{this.parent_pixieapp.active_ticker}}</h1>
    </div>
    <div class="col-sm-12" id="chart{{prefix}}" pd_render_onload pd_entity="get_moving_average_df()">
        <pd_options>
        {
          "valueFields": "Adj. Close",
          "keyFields": "x",
          "rendererId": "bokeh",
          "handlerId": "lineChart",
          "rowCount": "10000"
        }
        </pd_options>
    </div>
</div>
<div class="row">
    <div pd_widget="lag_slider">
        <pd_event_handler 
            pd_source="lagSlider"
 pd_script="self.lag = eventInfo['value']"
 pd_refresh="chart{{prefix}}">
        </pd_event_handler>
    </div>
</div>
"""
    def get_moving_average_df(self):
        ma = self.parent_pixieapp.get_active_df()['Adj. Close'].rolling(window=self.lag).mean()
        ma_df = pd.DataFrame(ma)
        ma_df["x"] = ma_df.index
        return ma_df
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode20.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode20.py)

以下截图展示了`MovingAverageSubApp`所显示的图表：

![MovingAverageSubApp – 第二个子 PixieApp](img/B09699_08_18.jpg)

MovingAverageSubApp 截图

## AutoCorrelationSubApp – 第三个子 PixieApp

对于第三个子 PixieApp，我们调用了`AutoCorrelationSubApp`；我们展示了所选股票 DataFrame 的 ACF 和 PACF，它们是使用`statsmodels`包计算得出的。

以下代码展示了`AutoCorrelationSubApp`的实现，它还使用了`add_ticker_selection_markup`装饰器和名为`lag_slider`的`pd_widget`：

```py
import statsmodels.tsa.api as smt
@PixieApp
class AutoCorrelationSubApp(BaseSubApp):
    @route()
    @BaseSubApp.add_ticker_selection_markup(['chart_acf{{prefix}}', 'chart_pacf{{prefix}}'])
    def main_screen(self):
        return """
<div class="row" style="min-height:300px">
    <div class="col-sm-6">
        <div class="page-header text-center">
            <h1>Auto-correlation Function</h1>
        </div>
        <div id="chart_acf{{prefix}}" pd_render_onload pd_options="show_acf=true">
        </div>
    </div>
    <div class="col-sm-6">
        <div class="page-header text-center">
            <h1>Partial Auto-correlation Function</h1>
        </div>
        <div id="chart_pacf{{prefix}}" pd_render_onload pd_options="show_pacf=true">
        </div>
    </div>
</div> 

<div class="row">
    <div pd_widget="lag_slider">
        <pd_event_handler 
            pd_source="lagSlider"
            pd_script="self.lag = eventInfo['value']"
            pd_refresh="chart_acf{{prefix}},chart_pacf{{prefix}}">
        </pd_event_handler>
    </div>
</div>
"""
 @route(show_acf='*')
 @captureOutput
    def show_acf_screen(self):
        smt.graphics.plot_acf(self.parent_pixieapp.get_active_df()['Adj. Close'], lags=self.lag)

 @route(show_pacf='*')
 @captureOutput
    def show_pacf_screen(self):
        smt.graphics.plot_pacf(self.parent_pixieapp.get_active_df()['Adj. Close'], lags=self.lag)
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode21.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode21.py)

在前面的代码中，我们定义了两个路由：`show_acf` 和 `show_pacf`，分别调用`smt.graphics`包中的`plot_acf`和`plot_pacf`方法。我们还使用`@captureOutput`装饰器，告诉 PixieApp 框架捕获由`plot_acf`和`plot_pacf`生成的输出。

以下截图展示了`AutoCorrelationSubApp`所显示的图表：

![AutoCorrelationSubApp – 第三个子 PixieApp](img/B09699_08_19.jpg)

AutoCorrelationSubApp 截图

在本节中，我们展示了如何组合一个简单的 PixieApp，用于对时间序列进行基本的数据探索，并显示各种统计图表。完整的 Notebook 可以在此找到：[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/StockExplorer%20-%20Part%201.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/StockExplorer%20-%20Part%201.ipynb)。

在接下来的部分中，我们尝试使用一个非常流行的模型来构建一个时间序列预测模型，称为**自回归积分滑动平均**（**ARIMA**）。

# 使用 ARIMA 模型的时间序列预测

ARIMA 是最流行的时间序列预测模型之一，正如其名字所示，它由三个部分组成：

+   **AR**：代表**自回归**，即应用线性回归算法，使用一个观测值及其自身的滞后观测值作为训练数据。

    AR 模型使用以下公式：

    ![使用 ARIMA 模型的时间序列预测](img/B09699_08_31.jpg)

    其中 ![使用 ARIMA 模型的时间序列预测](img/B09699_08_32.jpg) 是从先前的观测中学习到的模型权重，![使用 ARIMA 模型的时间序列预测](img/B09699_08_33.jpg) 是观测值*t*的残差误差。

    我们还称*p*为自回归模型的阶数，定义为前述公式中包含的滞后观测值的数量。

    例如：

    *AR(2)* 定义为：

    ![使用 ARIMA 模型的时间序列预测](img/B09699_08_34.jpg)

    *AR(1)* 定义为：

    ![使用 ARIMA 模型的时间序列预测](img/B09699_08_35.jpg)

+   **I**：代表**积分**。为了使 ARIMA 模型有效，假设时间序列是平稳的，或者可以被转化为平稳的。如果一个序列的均值和方差随时间变化不大，我们称之为平稳序列 ([`en.wikipedia.org/wiki/Stationary_process`](https://en.wikipedia.org/wiki/Stationary_process))。

    ### 注意

    **注意**：严格平稳性也有一个概念，要求一组观测值的联合概率分布在时间平移时不发生变化。

    使用数学符号，严格平稳性转化为：

    ![使用 ARIMA 模型的时间序列预测](img/B09699_08_36.jpg) 和![使用 ARIMA 模型的时间序列预测](img/B09699_08_37.jpg) 对于任意的*t*, *m* 和 *k* 都是相同的，*F*是联合概率分布。

    实际操作中，这个条件太严格，前述较弱的定义通常更为常用。

    我们可以通过使用观测值与其前一个值之间的对数差分来对时间序列进行平稳化，如下方的公式所示：

    ![使用 ARIMA 模型的时间序列预测](img/B09699_08_38.jpg)

    可能在将时间序列真正转化为平稳序列之前，需要进行多次对数差分转换。我们称*d*为我们使用对数差分转换序列的次数。

    例如：

    *I(0)* 定义为不需要对数差分（该模型称为 ARMA 模型）。

    *I(1)* 定义为需要 1 次对数差分。

    *I(2)* 定义为需要 2 次对数差分。

    ### 注意

    **注意**：在预测一个值之后，记得做与转换次数相同的反向转换。

+   **MA**：代表**滑动平均**。MA 模型使用当前观测值的均值的残差误差和滞后观测值的加权残差误差。我们可以使用以下公式来定义该模型：![使用 ARIMA 模型的时间序列预测](img/B09699_08_39.jpg)

    其中

    ![使用 ARIMA 模型的时间序列预测](img/B09699_08_40.jpg)

    是时间序列的均值，![使用 ARIMA 模型的时间序列预测](img/B09699_08_33.jpg) 是序列中的残差误差，![使用 ARIMA 模型的时间序列预测](img/B09699_08_41.jpg) 是滞后残差误差的权重。

    我们称*q*为滑动平均窗口的大小。

    例如：

    *MA(0)* 定义为不需要滑动平均（该模型称为 AR 模型）。

    *MA(1)* 定义为使用 1 的滑动平均窗口。公式为：

    ![使用 ARIMA 模型的时间序列预测](img/B09699_08_42.jpg)

根据前述定义，我们使用符号*ARIMA(p,d,q)*来定义一个 ARIMA 模型，其中*p*为自回归模型的阶数，*d*为积分/差分的阶数，*q*为滑动平均窗口的大小。

实现构建 ARIMA 模型的所有代码可能非常耗时。幸运的是，`statsmodels`库在`statsmodels.tsa.arima_model`包中实现了一个`ARIMA`类，提供了训练模型所需的所有计算，使用`fit()`方法来训练模型，使用`predict()`方法来预测值。它还处理对数差分，使时间序列变得平稳。诀窍是找到用于构建最佳 ARIMA 模型的参数*p*、*d*和*q*。为此，我们使用以下的 ACF 和 PACF 图：

+   *p*值对应于 ACF 图首次越过统计显著性阈值的滞后数（在*x*坐标轴上）。

+   同样，*q*值对应于 PACF 图首次越过统计显著性阈值的滞后数（在*x*坐标轴上）。

## 构建 MSFT 股票时间序列的 ARIMA 模型

提醒一下，MSFT 股票时间序列的价格图表如下所示：

![构建 MSFT 股票时间序列的 ARIMA 模型](img/B09699_08_20.jpg)

MSFT 股票序列图

在我们开始构建模型之前，让我们首先保留最后 14 天的数据用于测试，其余部分用于训练。

以下代码定义了两个新变量：`train_set`和`test_set`：

```py
train_set, test_set = msft[:-14], msft[-14:]
```

### 注意

**注意**：如果你仍然不熟悉前面的切片表示法，请参考本章开头关于 NumPy 的部分。

从前面的图表中，我们可以清楚地观察到自 2012 年起的增长趋势，但没有明显的季节性。因此，我们可以放心地假设没有平稳性。我们首先尝试应用一次对数差分变换，并绘制相应的 ACF 和 PACF 图。

在以下代码中，我们通过对`Adj. Close`列使用`np.log()`来构建`logmsft` pandas Series，然后使用`logmsft`与滞后 1（使用`shift()`方法）的差异来构建`logmsft_diff` pandas DataFrame。像之前一样，我们还调用`reset_index()`将`Date`索引转换为列，以便 PixieDust 的`display()`可以处理它：

```py
logmsft = np.log(train_set['Adj. Close'])
logmsft.index = train_set['Date']
logmsft_diff = pd.DataFrame(logmsft - logmsft.shift()).reset_index()
logmsft_diff.dropna(inplace=True)
display(logmsft_diff)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode22.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode22.py)

结果如下面的截图所示：

![构建 MSFT 股票时间序列的 ARIMA 模型](img/B09699_08_21.jpg)

对数差分后的 MSFT 股票序列

从前面的图形来看，我们可以合理地认为，我们已成功将时间序列平稳化，且其均值为 0。我们还可以使用更严格的方法，通过使用 Dickey-Fuller 检验（[`en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test`](https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test)）来检验平稳性，该检验测试*AR(1)*模型中是否存在单位根的原假设。

### 注意

**注意**：在统计学中，统计假设检验是通过取样来挑战假设是否成立，并判断该假设是否成立。我们查看 p 值（[`en.wikipedia.org/wiki/P-value`](https://en.wikipedia.org/wiki/P-value)），它有助于判断结果的显著性。有关统计假设检验的更多细节可以在这里找到：

[`en.wikipedia.org/wiki/Statistical_hypothesis_testing`](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)

以下代码使用 `statsmodels.tsa.stattools` 包中的 `adfuller` 方法：

```py
from statsmodels.tsa.stattools import adfuller
import pprint

ad_fuller_results = adfuller(
logmsft_diff['Adj. Close'], autolag = 'AIC', regression = 'c'
)
labels = ['Test Statistic','p-value','#Lags Used','Number of Observations Used']
pp = pprint.PrettyPrinter(indent=4)
pp.pprint({labels[i]: ad_fuller_results[i] for i in range(4)})
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode23.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode23.py)我们使用 `pprint` 包，它对 *漂亮打印* 任意 Python 数据结构非常有用。有关 `pprint` 的更多信息可以在这里找到：

[`docs.python.org/3/library/pprint.html`](https://docs.python.org/3/library/pprint.html)

结果（在以下链接详细解释：[`www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.adfuller.html`](http://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.adfuller.html)）显示在这里：

```py
{
    'Number of lags used': 3,
    'Number of Observations Used': 8057,
    'Test statistic': -48.071592138591136,
    'MacKinnon's approximate p-value': 0.0
}
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode24.json`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode24.json)

p 值低于显著性水平；因此，我们可以拒绝零假设，即 *AR(1)* 模型中存在单位根，这使我们有信心认为时间序列是平稳的。

然后我们绘制 ACF 和 PACF 图，这将为我们提供 ARIMA 模型的 *p* 和 *q* 参数：

以下代码构建了 ACF 图：

```py
import statsmodels.tsa.api as smt
smt.graphics.plot_acf(logmsft_diff['Adj. Close'], lags=100)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode25.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode25.py)

结果显示在以下截图中：

![构建 MSFT 股票时间序列的 ARIMA 模型](img/B09699_08_22.jpg)

对数差分 MSFT 数据框的 ACF

从前面的 ACF 图中，我们可以看到相关性首次超过统计显著性阈值时，滞后值为 1。因此，我们将使用 *p = 1* 作为 ARIMA 模型的 AR 顺序。

我们对 PACF 做相同的操作：

```py
smt.graphics.plot_pacf(logmsft_diff['Adj. Close'], lags=100)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode26.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode26.py)

结果显示在以下截图中：

![构建 MSFT 股票时间序列的 ARIMA 模型](img/B09699_08_23.jpg)

对数差分 MSFT 数据框的 PACF

从之前的 PACF 图中，我们还可以看到在滞后为 1 时，相关性首次越过了统计显著性阈值。因此，我们将使用*q = 1*作为 ARIMA 模型的 MA 阶数。

我们还只应用了对数差分转换一次。因此，我们将使用*d = 1*作为 ARIMA 模型的积分部分。

### 注意

**注意**：当调用`ARIMA`类时，如果使用*d = 0*，你可能需要手动进行对数差分，在这种情况下，你需要自己在预测值上恢复转换。如果不进行此操作，`statsmodels`包会在返回预测值之前自动恢复转换。

以下代码使用*p = 1*、*d = 1*、*q = 1*作为`ARIMA`构造函数的顺序元组参数，对`train_set`时间序列训练 ARIMA 模型。接着我们调用`fit()`方法进行训练并获取模型：

```py
from statsmodels.tsa.arima_model import ARIMA

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    arima_model_class = ARIMA(train_set['Adj. Close'], dates=train_set['Date'], order=(1,1,1))
    arima_model = arima_model_class.fit(disp=0)

    print(arima_model.resid.describe())
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode27.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode27.py)

**注意**：我们使用`warnings`包来避免在使用较旧版本的 NumPy 和 pandas 时可能出现的多个弃用警告。

在之前的代码中，我们使用`train_set['Adj. Close']`作为`ARIMA`构造函数的参数。由于我们使用的是 Series 数据，因此我们还需要传递`train_set['Date']`系列作为`dates`参数。请注意，如果我们传递的是带有`DateIndex`索引的 pandas DataFrame，那么就不需要使用`dates`参数了。`ARIMA`构造函数的最后一个参数是`order`参数，它是一个包含*p*、*d*和*q*的元组，正如本节开头所讨论的那样。

然后，我们调用`fit()`方法，它返回实际的 ARIMA 模型，我们将使用该模型来预测数值。为了展示目的，我们使用`arima_model.resid.describe()`打印模型的残差误差统计信息。

结果如下所示：

```py
count    8.061000e+03
mean    -5.785533e-07
std      4.198119e-01
min     -5.118915e+00
25%     -1.061133e-01
50%     -1.184452e-02
75%      9.848486e-02
max      5.023380e+00
dtype: float64
```

平均残差误差是 ![构建 MSFT 股票时间序列的 ARIMA 模型](img/B09699_08_43.jpg)，该误差非常接近零，因此表明模型可能存在过拟合训练数据的情况。

现在我们已经有了模型，接下来尝试对其进行诊断。我们定义了一个名为`plot_predict`的方法，它接受一个模型、一个日期系列和一个数字，表示我们想要回溯的时间段。然后我们调用 ARIMA 的`plot_predict()`方法，绘制一个包含预测值和观察值的图表。

以下代码展示了`plot_predict()`方法的实现，包括两次调用`100`和`10`：

```py
def plot_predict(model, dates_series, num_observations):
    fig = plt.figure(figsize = (12,5))
    model.plot_predict(
        start = str(dates_series[len(dates_series)-num_observations]),
        end = str(dates_series[len(dates_series)-1])
    )
    plt.show()

plot_predict(arima_model, train_set['Date'], 100)
plot_predict(arima_model, train_set['Date'], 10)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode28.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode28.py)

结果如下所示：

![为 MSFT 股票时间序列构建 ARIMA 模型](img/B09699_08_24.jpg)

观察值与预测值图表

上述图表展示了预测值与训练集实际观察值的接近程度。现在我们使用之前被保留的测试集，进一步诊断模型。对于这一部分，我们使用`forecast()`方法来预测下一个数据点。对于`test_set`的每个值，我们从一个叫做 history 的观察数组中构建一个新的 ARIMA 模型，这个数组包含了训练数据并增添了每个预测值。

以下代码展示了`compute_test_set_predictions()`方法的实现，该方法接收`train_set`和`test_set`作为参数，并返回一个包含`forecast`列（包含所有预测值）和`test`列（包含相应实际观察值）的 pandas DataFrame：

```py
def compute_test_set_predictions(train_set, test_set):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = train_set['Adj. Close'].values
        forecast = np.array([])
        for t in range(len(test_set)):
            prediction = ARIMA(history, order=(1,1,0)).fit(disp=0).forecast()
            history = np.append(history, test_set['Adj. Close'].iloc[t])
            forecast = np.append(forecast, prediction[0])
        return pd.DataFrame(
 {"forecast": forecast,
 "test": test_set['Adj. Close'],
 "Date": pd.date_range(start=test_set['Date'].iloc[len(test_set)-1], periods = len(test_set))
 }
 )

results = compute_test_set_predictions(train_set, test_set)
display(results)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode29.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode29.py)

以下截图展示了结果图表：

![为 MSFT 股票时间序列构建 ARIMA 模型](img/B09699_08_25.jpg)

预测值与实际值的图表

我们可以使用流行的`mean_squared_error`方法来衡量误差，该方法来自 scikit-learn 包（[`en.wikipedia.org/wiki/Mean_squared_error`](https://en.wikipedia.org/wiki/Mean_squared_error)）并定义如下：

![为 MSFT 股票时间序列构建 ARIMA 模型](img/B09699_08_44.jpg)

其中 ![为 MSFT 股票时间序列构建 ARIMA 模型](img/B09699_08_45.jpg) 是实际值，![为 MSFT 股票时间序列构建 ARIMA 模型](img/B09699_08_46.jpg) 是预测值。

以下代码定义了一个`compute_mean_squared_error`方法，该方法接收一个测试集和一个预测集，并返回均方误差的值：

```py
from sklearn.metrics import mean_squared_error
def compute_mean_squared_error(test_series, forecast_series):
    return mean_squared_error(test_series, forecast_series)

print('Mean Squared Error: {}'.format(
compute_mean_squared_error( test_set['Adj. Close'], results.forecast))
)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode30.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode30.py)

结果如下所示：

```py
Mean Squared Error: 6.336538843075749
```

## StockExplorer PixieApp 第二部分 - 使用 ARIMA 模型添加时间序列预测

在本节中，我们通过添加一个菜单来改进`StockExplorer` PixieApp，该菜单为选定的股票代码提供基于 ARIMA 模型的时间序列预测。我们创建了一个名为`ForecastArimaSubApp`的新类，并更新了主`StockExplorer`类中的`tabs`变量。

```py
[[StockExplorer]]
@route(explore="*")
@templateArgs
def stock_explore_screen(self):
   tabs = [("Explore","StockExploreSubApp"),
           ("Moving Average", "MovingAverageSubApp"),
           ("ACF and PACF", "AutoCorrelationSubApp"),
 ("Forecast with ARIMA", "ForecastArimaSubApp")]
   ...
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode31.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode31.py)

`ForecastArimaSubApp`子 PixieApp 由两个屏幕组成。第一个屏幕显示时间序列图表以及 ACF 和 PACF 图表。这个屏幕的目标是为用户提供必要的数据探索，帮助他们确定 ARIMA 模型的*p*、*d*和*q*阶数，如前一节所解释的那样。通过查看时间序列图表，我们可以判断时间序列是否平稳（提醒一下，这是构建 ARIMA 模型的前提条件）。如果不是，用户可以点击**添加差分**按钮，尝试通过对数差分转换使 DataFrame 平稳。然后，三个图表会使用转换后的 DataFrame 进行更新。

以下代码展示了`ForecastArimaSubApp`子 PixieApp 的默认路由：

```py
from statsmodels.tsa.arima_model import ARIMA

@PixieApp
class ForecastArimaSubApp(BaseSubApp):
    def setup(self):
        self.entity_dataframe = self.parent_pixieapp.get_active_df().copy()
        self.differencing = False

    def set_active_ticker(self, ticker):
 BaseSubApp.set_active_ticker(self, ticker)
        self.setup()

    @route()
 @BaseSubApp.add_ticker_selection_markup([])
    def main_screen(self):
        return """
<div class="page-header text-center">
    <h2>1\. Data Exploration to test for Stationarity
        <button class="btn btn-default"
                pd_script="self.toggle_differencing()" pd_refresh>
            {%if this.differencing%}Remove differencing{%else%}Add differencing{%endif%}
        </button>
        <button class="btn btn-default"
                pd_options="do_forecast=true">
            Continue to Forecast
        </button>
    </h2>
</div>

<div class="row" style="min-height:300px">
    <div class="col-sm-10" id="chart{{prefix}}" pd_render_onload pd_options="show_chart=Adj. Close">
    </div>
</div>

<div class="row" style="min-height:300px">
    <div class="col-sm-6">
        <div class="page-header text-center">
            <h3>Auto-correlation Function</h3>
        </div>
        <div id="chart_acf{{prefix}}" pd_render_onload pd_options="show_acf=true">
        </div>
    </div>
    <div class="col-sm-6">
        <div class="page-header text-center">
            <h3>Partial Auto-correlation Function</h3>
        </div>
        <div id="chart_pacf{{prefix}}" pd_render_onload pd_options="show_pacf=true">
        </div>
    </div>
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode32.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode32.py)

上面的代码遵循了我们现在应该熟悉的模式：

+   定义一个`setup`方法，确保 PixieApp 启动时会调用。在这个方法中，我们从父级 PixieApp 复制选中的 DataFrame，并维护一个名为`self.differencing`的变量，用于跟踪用户是否点击了**添加差分**按钮。

+   我们创建了一个默认路由，显示由以下组件组成的第一个屏幕：

    +   一个带有两个按钮的头部：`添加差分`（用于使时间序列平稳）和`继续预测`（用于显示第二个屏幕，稍后我们将讨论）。当差分已应用时，`添加差分`按钮会切换为`移除差分`。

    +   一个`<div>`元素，它调用`show_chart`路由来显示时间序列图表。

    +   一个`<div>`元素，它调用`show_acf`路由来显示 ACF 图表。

    +   一个`<div>`元素，它调用`show_pacf`路由来显示 PACF 图表。

+   我们使用一个空数组`[]`作为`@BaseSubApp.add_ticker_selection_markup`装饰器的参数，确保当用户选择另一个股票代码时，整个屏幕会刷新，并且从第一个屏幕重新开始。我们还需要重置内部变量。为此，我们对`add_ticker_selection_markup`进行了修改，定义了`BaseSubApp`中的一个新方法`set_active_ticker`，它是父级 PixieApp 中的`set_active_ticker`方法的封装。这个设计的目的是让子类能够重写这个方法，并根据需要注入额外的代码。我们还修改了`pd_script`属性，用于当用户选择新的股票代码时调用该方法，如下代码所示：

    ```py
    [[BaseSubApp]]
    def add_ticker_selection_markup(refresh_ids):
            def deco(fn):
                def wrap(self, *args, **kwargs):
                    return """
    <div class="row" style="text-align:center">
        <div class="btn-group btn-group-toggle"
             style="border-bottom:2px solid #eeeeee"
             data-toggle="buttons">
            {%for ticker, state in this.parent_pixieapp.tickers.items()%}
            <label class="btn btn-secondary {%if this.parent_pixieapp.active_ticker == ticker%}active{%endif%}"
                pd_refresh=\"""" + ",".join(refresh_ids) + """\" pd_script="self.set_active_ticker('{{ticker}}')">
                <input type="radio" {%if this.parent_pixieapp.active_ticker == ticker%}checked{%endif%}> 
                    {{ticker}}
            </label>
            {%endfor%}
        </div>
    </div>
                    """ + fn(self, *args, **kwargs)
                return wrap
            return deco

     def set_active_ticker(self, ticker):
     self.parent_pixieapp.set_active_ticker(ticker)

    ```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode33.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode33.py)

在`ForecastArimaSubApp`子 PixieApp 中，我们覆盖了`set_active_tracker`方法，首先调用父类方法，然后调用`self.setup()`来重新初始化内部变量：

```py
[[ForecastArimaSubApp]]
def set_active_ticker(self, ticker):
        BaseSubApp.set_active_ticker(self, ticker)
        self.setup()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode34.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode34.py)

第一个预测屏幕的路由实现非常简单。`Add differencing` / `Remove differencing`按钮具有`pd_script`属性，调用`self.toggle_differencing()`方法，并且具有`pd_refresh`属性来更新整个页面。它还定义了三个`<div>`元素，分别调用`show_chart`、`show_acf`和`show_pacf`路由，如以下代码所示：

```py
[[ForecastArimaSubApp]]
@route()
    @BaseSubApp.add_ticker_selection_markup([])
    def main_screen(self):
        return """
<div class="page-header text-center">
  <h2>1\. Data Exploration to test for Stationarity
    <button class="btn btn-default"
            pd_script="self.toggle_differencing()" pd_refresh>
    {%if this.differencing%}Remove differencing{%else%}Add differencing{%endif%}
    </button>
    <button class="btn btn-default" pd_options="do_forecast=true">
        Continue to Forecast
    </button>
  </h2>
</div>

<div class="row" style="min-height:300px">
  <div class="col-sm-10" id="chart{{prefix}}" pd_render_onload pd_options="show_chart=Adj. Close">
  </div>
</div>

<div class="row" style="min-height:300px">
    <div class="col-sm-6">
        <div class="page-header text-center">
            <h3>Auto-correlation Function</h3>
        </div>
        <div id="chart_acf{{prefix}}" pd_render_onload pd_options="show_acf=true">
        </div>
    </div>
    <div class="col-sm-6">
      <div class="page-header text-center">
         <h3>Partial Auto-correlation Function</h3>
      </div>
      <div id="chart_pacf{{prefix}}" pd_render_onload pd_options="show_pacf=true">
      </div>
    </div>
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode35.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode35.py)

`toggle_differencing()`方法通过`self.differencing`变量跟踪当前差分状态，并且要么从`parent_pixieapp`复制活动数据框，要么对`self.entity_dataframe`变量应用对数差分转换，如以下代码所示：

```py
def toggle_differencing(self):
   if self.differencing:
       self.entity_dataframe = self.parent_pixieapp.get_active_df().copy()
       self.differencing = False
   else:
       log_df = np.log(self.entity_dataframe['Adj. Close'])
       log_df.index = self.entity_dataframe['Date']
       self.entity_dataframe = pd.DataFrame(log_df - log_df.shift()).reset_index()
       self.entity_dataframe.dropna(inplace=True)
       self.differencing = True
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode36.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode36.py)

`show_acf`和`show_pacf`路由实现非常简单。它们分别调用`smt.graphics.plot_acf`和`smt.graphics.plot_pacf`方法。它们还使用`@captureOutput`装饰器将图表图像传递到目标小部件：

```py
@route(show_acf='*')
@captureOutput
def show_acf_screen(self):
    smt.graphics.plot_acf(self.entity_dataframe['Adj. Close'], lags=50)

@route(show_pacf='*')
@captureOutput
def show_pacf_screen(self):
    smt.graphics.plot_pacf(self.entity_dataframe['Adj. Close'], lags=50)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode37.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode37.py)

以下截图展示了没有差分的预测子 PixieApp 的数据探索页面：

![StockExplorer PixieApp Part 2 – 使用 ARIMA 模型添加时间序列预测](img/B09699_08_26.jpg)

没有应用差分的第一个预测屏幕

正如预期的那样，图表与非平稳的时间序列一致。当用户点击**Add differencing**按钮时，显示以下屏幕：

![StockExplorer PixieApp Part 2 – 使用 ARIMA 模型添加时间序列预测](img/B09699_08_27.jpg)

应用差分的第一个预测屏幕

下一步是实现 `do_forecast` 路由，它由 **继续预测** 按钮触发。这个路由负责构建 ARIMA 模型；它首先展示一个配置页面，其中有三个输入框，允许用户输入 *p*、*d* 和 *q* 顺序，这些顺序是通过查看数据探索界面中的图表推断得出的。我们添加了一个 `Go` 按钮来继续使用 `build_arima_model` 路由构建模型，稍后我们会在本节中讨论这个路由。页面头部还有一个 `Diagnose Model` 按钮，触发另一个页面，用于评估模型的准确性。

`do_forecast` 路由的实现如下所示。请注意，我们使用 `add_ticker_selection_markup` 并传递一个空数组，以便在用户选择另一个股票代码时刷新整个页面：

```py
[[ForecastArimaSubApp]] 
@route(do_forecast="true")
 @BaseSubApp.add_ticker_selection_markup([])
    def do_forecast_screen(self):
        return """
<div class="page-header text-center">
    <h2>2\. Build Arima model
        <button class="btn btn-default"
                pd_options="do_diagnose=true">
            Diagnose Model
        </button>
    </h2>
</div>
<div class="row" id="forecast{{prefix}}">
    <div style="font-weight:bold">Enter the p,d,q order for the ARIMA model you want to build</div>

    <div class="form-group" style="margin-left: 20px">
        <label class="control-label">Enter the p order for the AR model:</label>
        <input type="text" class="form-control"
               id="p_order{{prefix}}"
               value="1" style="width: 100px;margin-left:10px">

        <label class="control-label">Enter the d order for the Integrated step:</label>
        <input type="text" class="form-control"
               id="d_order{{prefix}}" value="1"
               style="width: 100px;margin-left:10px">

        <label class="control-label">Enter the q order for the MA model:</label>
        <input type="text" class="form-control" 
               id="q_order{{prefix}}" value="1"
               style="width: 100px;margin-left:10px">
    </div>

    <center>
        <button class="btn btn-default"
               pd_target="forecast{{prefix}}"
            pd_options="p_order=$val(p_order{{prefix}});d_order=$val(p_order{{prefix}});q_order=$val(p_order{{prefix}})">
        Go
        </button>
    </center>
</div>
"""
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode38.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode38.py)

以下截图显示了 **构建 ARIMA 模型** 页面配置的界面：

![StockExplorer PixieApp 第二部分 – 使用 ARIMA 模型添加时间序列预测](img/B09699_08_28.jpg)

构建 ARIMA 模型页面的配置页面

**Go** 按钮有一个 `pd_options` 属性，该属性调用一个有三个状态的路由：`p_order`、`d_order` 和 `q_order`，这些值取自与每个属性相关的三个输入框。

构建 ARIMA 模型的路由在下面的代码中显示。它首先将活动数据框（DataFrame）拆分为训练集和测试集，保留 14 个观测值作为测试集。然后它构建模型并计算残差误差。一旦模型成功构建，我们返回一个包含图表的 HTML 标记，图表显示训练集的预测值与实际训练集的值对比。这是通过调用 `plot_predict` 路由实现的。最后，我们还通过创建一个 `<div>` 元素，并为其设置 `pd_entity` 属性指向残差变量，使用 `<pd_options>` 子元素配置所有统计数据的表格视图，来显示模型的残差误差统计信息。

显示预测与实际训练集的对比图表使用了 `plot_predict` 路由，该路由调用了我们在笔记本中之前创建的 `plot_predict` 方法。我们还使用了 `@captureOutput` 装饰器，将图表图像发送到正确的组件。

`plot_predict` 路由的实现如下所示：

```py
    @route(plot_predict="true")
    @captureOutput
    def plot_predict(self):
        plot_predict(self.arima_model, self.train_set['Date'], 100)

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode39.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode39.py)

`build_arima_model` 路由的实现如下所示：

```py
@route(p_order="*",d_order="*",q_order="*")
def build_arima_model_screen(self, p_order, d_order, q_order):
    #Build the arima model
    self.train_set = self.parent_pixieapp.get_active_df()[:-14]
    self.test_set = self.parent_pixieapp.get_active_df()[-14:]
    self.arima_model = ARIMA(
        self.train_set['Adj. Close'], dates=self.train_set['Date'],
        order=(int(p_order),int(d_order),int(q_order))
    ).fit(disp=0)
    self.residuals = self.arima_model.resid.describe().to_frame().reset_index()
    return """
<div class="page-header text-center">
    <h3>ARIMA Model succesfully created</h3>
<div>
<div class="row">
    <div class="col-sm-10 col-sm-offset-3">
        <div pd_render_onload pd_options="plot_predict=true">
        </div>
        <h3>Predicted values against the train set</h3>
    </div>
</div>
<div class="row">
    <div pd_render_onload pd_entity="residuals">
        <pd_options>
 {
 "handlerId": "tableView",
 "table_noschema": "true",
 "table_nosearch": "true",
 "table_nocount": "true"
 }
 </pd_options>
    </div>
    <h3><center>Residual errors statistics</center></h3>
<div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode40.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode40.py)

以下截图显示了 **构建 ARIMA 模型** 页面结果：

![StockExplorer PixieApp Part 2 – 使用 ARIMA 模型添加时间序列预测](img/B09699_08_29.jpg)

模型构建页面

预测子应用的最终页面是由 `do_diagnose` 路由调用的 *诊断模型* 页面。在这个页面中，我们只是简单地显示了一个由 `compute_test_set_predictions` 方法返回的 DataFrame 的折线图，这个方法我们之前在 Notebook 中使用 `train_set` 和 `test_set` 变量创建过。这个图表的 `<div>` 元素使用了 `pd_entity` 属性，调用了一个中介类方法 `compute_test_set_predictions`。它还有一个 `<pd_options>` 子元素，包含显示折线图的 `display()` 选项。

以下代码展示了 `do_diagnose_screen` 路由的实现：

```py
    def compute_test_set_predictions(self):
        return compute_test_set_predictions(self.train_set, self.test_set)

    @route(do_diagnose="true")
    @BaseSubApp.add_ticker_selection_markup([])
    def do_diagnose_screen(self):
        return """
<div class="page-header text-center"><h2>3\. Diagnose the model against the test set</h2></div>
<div class="row">
    <div class="col-sm-10 center" pd_render_onload pd_entity="compute_test_set_predictions()">
        <pd_options>
 {
 "keyFields": "Date",
 "valueFields": "forecast,test",
 "handlerId": "lineChart",
 "rendererId": "bokeh",
 "noChartCache": "true" 
 }
        </pd_options>
    </div>
</div>
"""
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode41.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/sampleCode41.py)

以下截图显示了诊断页面的结果：

![StockExplorer PixieApp Part 2 – 使用 ARIMA 模型添加时间序列预测](img/B09699_08_30.jpg)

模型诊断页面

在这一部分中，我们展示了如何改进 `StockExplorer` 示例 PixieApp，加入使用 ARIMA 模型的预测能力。顺便提一下，我们演示了如何使用 PixieApp 编程模型创建一个三步向导，首先进行一些数据探索，然后配置模型的参数并构建模型，最后对模型进行测试集的诊断。

### 注意

完整的 Notebook 实现可以在此找到：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/StockExplorer%20-%20Part%202.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%208/StockExplorer%20-%20Part%202.ipynb)

# 总结

在本章中，我们触及了时间序列分析和预测的话题。当然，我们只是略微触及了表面，实际上还有更多内容需要探索。这也是一个对行业非常重要的领域，特别是在金融领域，相关研究非常活跃。例如，我们看到越来越多的数据科学家尝试基于循环神经网络（[`en.wikipedia.org/wiki/Recurrent_neural_network`](https://en.wikipedia.org/wiki/Recurrent_neural_network)）算法构建时间序列预测模型，并取得了巨大成功。我们还展示了如何将 Jupyter Notebooks 与 PixieDust 以及 `pandas`、`numpy` 和 `statsmodels` 等库的生态系统结合使用，帮助加速分析开发，并将其操作化为可以被业务用户使用的应用。

在下一章，我们将探讨另一个重要的数据科学应用场景：图形。我们将构建一个与航班旅行相关的示例应用，并讨论我们何时以及如何应用图算法来解决数据问题。
