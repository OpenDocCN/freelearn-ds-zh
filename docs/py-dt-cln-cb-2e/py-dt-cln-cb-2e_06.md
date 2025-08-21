

# 第六章：使用 Series 操作清理和探索数据

我们可以将本书前几章的配方视为本质上是诊断性的。我们导入了一些原始数据，然后生成了关于关键变量的描述性统计数据。这使我们对这些变量的值分布情况有了一个大致的了解，并帮助我们识别出异常值和意外值。然后，我们检查了变量之间的关系，以寻找模式，以及这些模式的偏差，包括逻辑上的不一致性。简而言之，到目前为止，我们的主要目标是弄清楚数据到底发生了什么。

然而，在数据探索和清理项目开始不久后，我们通常需要更改一些变量在某些观察中的初始值。例如，我们可能需要创建一个基于一个或多个其他列值的新列，或者我们可能想要改变某些值，这些值可能在某个范围内，比如小于 0，或者超过某个阈值，可能需要将它们设置为均值，或者设置为缺失。幸运的是，pandas Series 对象提供了大量用于操作数据值的方法。

本章的配方展示了如何使用 pandas 方法来更新 Series 的值，一旦我们确定了需要做什么。理想情况下，我们需要花时间仔细检查数据，在操作变量值之前。我们应该首先有集中趋势的度量、分布形状和范围的指示、相关性和可视化，然后再更新变量的值，或者基于它们创建新的变量。在更新变量值之前，我们还应该对异常值和缺失值有一个清晰的认识，了解它们如何影响汇总统计数据，并对填补新值或其他调整的初步计划有所准备。

完成这些工作后，我们就可以开始进行一些数据清理任务了。这些任务通常涉及直接操作 pandas Series 对象，无论是修改现有 Series 的值，还是创建一个新的 Series。这通常涉及条件性地更改值，仅更改满足特定标准的值，或者基于该 Series 的现有值或另一个 Series 的值，分配多个可能的值。

我们分配这些值的方式在很大程度上取决于 Series 的数据类型，无论是要更改的 Series 还是标准 Series。查询和清理字符串数据与处理日期或数值数据的任务有很大不同。对于字符串数据，我们通常需要评估某个字符串片段是否具有某个值，去除字符串中的一些无意义字符，或将其转换为数值或日期值。对于日期数据，我们可能需要查找无效的或超出范围的日期，甚至计算日期间隔。

幸运的是，pandas Series 提供了大量用于操作字符串、数值和日期值的工具。在本章中，我们将探讨一些最有用的工具。具体来说，我们将涵盖以下几个例子：

+   从 pandas Series 中获取值

+   显示 pandas Series 的汇总统计信息

+   更改 Series 值

+   有条件地更改 Series 值

+   评估和清理字符串 Series 数据

+   处理日期

+   使用 OpenAI 进行 Series 操作

让我们开始吧！

# 技术要求

你需要 pandas、NumPy 和 Matplotlib 来完成本章中的例子。我使用的是 pandas 2.1.4，但该代码也可以在 pandas 1.5.3 或更高版本上运行。

本章节中的代码可以从本书的 GitHub 仓库下载，[`github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition`](https://github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition)。

# 从 pandas Series 中获取值

pandas Series 是一种一维的类数组结构，它采用 NumPy 数据类型。每个 Series 也有一个索引，这是数据标签的数组。如果在创建 Series 时没有指定索引，它将使用默认的 0 到 N-1 的索引。

创建 pandas Series 的方式有很多种，包括从列表、字典、NumPy 数组或标量中创建。在数据清洗工作中，我们最常通过选择 DataFrame 的列来访问数据 Series，使用属性访问（`dataframename.columname`）或括号符号（`dataframename['columnname']`）。属性访问不能用来设置 Series 的值，但括号符号可以用于所有 Series 操作。

在本例中，我们将探讨从 pandas Series 中获取值的几种方法。这些技术与我们在 *第三章*《数据度量》中介绍的从 pandas DataFrame 获取行的方法非常相似。

## 做好准备

在本例中，我们将使用 **国家纵向调查** (**NLS**) 的数据，主要是关于每个受访者的高中 **平均绩点** (**GPA**) 数据。

**数据说明**

**国家青少年纵向调查**由美国劳工统计局进行。该调查始于 1997 年，调查对象为 1980 年至 1985 年间出生的一群人，并且每年都会进行跟踪调查，直到 2023 年。调查数据可供公众使用，网址：[nlsinfo.org](https://nlsinfo.org)。

## 如何实现…

在这个例子中，我们使用括号操作符以及 `loc` 和 `iloc` 访问器来选择 Series 值。让我们开始吧：

1.  导入所需的`pandas`和 NLS 数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97f.csv", low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

**注意**

是否使用括号运算符、`loc` 访问器或 `iloc` 访问器，主要是个人偏好问题。通常，当你知道要访问的行的索引标签时，使用 `loc` 访问器更方便。而当通过绝对位置引用行更为简便时，括号运算符或 `iloc` 访问器可能会是更好的选择。这个例子中展示了这一点。

1.  从 GPA 总体列创建一个 Series。

使用 `head` 显示前几个值及其对应的索引标签。`head` 默认显示的值数量是 5。Series 的索引与 DataFrame 的索引相同，即 `personid`：

```py
gpaoverall = nls97.gpaoverall
type(gpaoverall) 
```

```py
pandas.core.series.Series 
```

```py
gpaoverall.head() 
```

```py
personid
135335   3.09
999406   2.17
151672    NaN
750699   2.53
781297   2.43
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall.index 
```

```py
Index([135335, 999406, 151672, 750699, 781297, 613800,
       403743, 474817, 530234, 351406,
       ...
       290800, 209909, 756325, 543646, 411195, 505861,
       368078, 215605, 643085, 713757],
      dtype='int64', name='personid', length=8984) 
```

1.  使用括号运算符选择 GPA 值。

使用切片创建一个 Series，包含从第一个值到第五个值的所有值。注意我们得到了与 *第 2 步* 中 `head` 方法相同的值。在 `gpaoverall[:5]` 中不包含冒号左边的值意味着它将从开头开始。`gpaoverall[0:5]` 将返回相同的结果。同样，`gpaoverall[-5:]` 显示的是从第五个到最后一个位置的值。这与 `gpaoverall.tail()` 返回的结果相同：

```py
gpaoverall[:5] 
```

```py
135335   3.09
999406   2.17
151672    NaN
750699   2.53
781297   2.43
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall.tail() 
```

```py
personid
505861    NaN
368078    NaN
215605   3.22
643085   2.30
713757    NaN
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall[-5:] 
```

```py
personid
505861    NaN
368078    NaN
215605   3.22
643085   2.30
713757    NaN
Name: gpaoverall, dtype: float64 
```

1.  使用 `loc` 访问器选择值。

我们将一个索引标签（在此案例中为 `personid` 的值）传递给 `loc` 访问器以返回一个标量。如果我们传递一个索引标签列表，无论是一个还是多个，我们将得到一个 Series。我们甚至可以传递一个通过冒号分隔的范围。这里我们将使用 `gpaoverall.loc[135335:151672]`：

```py
gpaoverall.loc[135335]
3.09 
```

```py
gpaoverall.loc[[135335]] 
```

```py
personid
135335   3.09
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall.loc[[135335,999406,151672]] 
```

```py
personid
135335   3.09
999406   2.17
151672    NaN
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall.loc[135335:151672] 
```

```py
personid
135335   3.09
999406   2.17
151672    NaN
Name: gpaoverall, dtype: float64 
```

1.  使用 `iloc` 访问器选择值。

`iloc` 与 `loc` 的区别在于，它接受的是行号列表，而不是标签。它的工作方式类似于括号运算符切片。在这一步中，我们传递一个包含 0 的单项列表。然后，我们传递一个包含五个元素的列表 `[0,1,2,3,4]`，以返回一个包含前五个值的 Series。如果我们传递 `[:5]` 给访问器，也会得到相同的结果：

```py
gpaoverall.iloc[[0]] 
```

```py
personid
135335   3.09
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall.iloc[[0,1,2,3,4]] 
```

```py
personid
135335   3.09
999406   2.17
151672    NaN
750699   2.53
781297   2.43
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall.iloc[:5] 
```

```py
personid
135335   3.09
999406   2.17
151672    NaN
750699   2.53
781297   2.43
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall.iloc[-5:] 
```

```py
personid
505861    NaN
368078    NaN
215605   3.22
643085   2.30
713757    NaN
Name: gpaoverall, dtype: float64 
```

访问 pandas Series 值的这些方法——括号运算符、`loc` 访问器和 `iloc` 访问器——都有许多使用场景，特别是 `loc` 访问器。

## 它是如何工作的...

在 *第 3 步* 中，我们使用 `[]` 括号运算符执行了类似标准 Python 的切片操作来创建一个 Series。这个运算符允许我们根据位置轻松选择数据，方法是使用列表或通过切片符号表示的值范围。该符号的形式为 `[start:end:step]`，如果没有提供 `step`，则假定 `step` 为 1。当 `start` 使用负数时，它表示从原始 Series 末尾开始的行数。

在*步骤 4*中使用的`loc`访问器通过索引标签选择数据。由于`personid`是 Series 的索引，我们可以将一个或多个`personid`值的列表传递给`loc`访问器，以获取具有这些标签及相关 GPA 值的 Series。我们还可以将一个标签范围传递给访问器，它将返回一个包含从冒号左侧到右侧（包括）的索引标签的 GPA 值的 Series。例如，`gpaoverall.loc[135335:151672]`将返回`personid`在`135335`到`151672`之间（包括这两个值）的 GPA 值的 Series。

如*步骤 5*所示，`iloc`访问器使用的是行位置，而不是索引标签。我们可以传递一个整数列表或使用切片表示法传递一个范围。

# 显示 pandas Series 的汇总统计数据

pandas 有许多生成汇总统计数据的方法。我们可以分别使用`mean`、`median`、`max`和`min`方法轻松获得 Series 的平均值、中位数、最大值或最小值。非常方便的`describe`方法将返回所有这些统计数据，以及其他一些数据。我们还可以使用`quantile`方法获得 Series 中任意百分位的值。这些方法可以应用于 Series 的所有值，或者仅用于选定的值。接下来的示例中将展示如何使用这些方法。

## 准备工作

我们将继续使用 NLS 中的总体 GPA 列。

## 如何操作...

让我们仔细看看整个数据框和选定行的总体 GPA 分布。为此，请按照以下步骤操作：

1.  导入`pandas`和`numpy`并加载 NLS 数据：

    ```py
    import pandas as pd
    import numpy as np
    nls97 = pd.read_csv("data/nls97f.csv",
    low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  获取一些描述性统计数据：

    ```py
    gpaoverall = nls97.gpaoverall
    gpaoverall.mean() 
    ```

    ```py
    2.8184077281812145 
    ```

    ```py
    gpaoverall.describe() 
    ```

    ```py
    count	6,004.00
    mean	2.82
    std	0.62
    min	0.10
    25%	2.43
    50%	2.86
    75%	3.26
    max	4.17
    Name: gpaoverall, dtype: float64 
    ```

    ```py
    gpaoverall.quantile(np.arange(0.1,1.1,0.1)) 
    ```

    ```py
    0.10	2.02
    0.20	2.31
    0.30	2.52
    0.40	2.70
    0.50	2.86
    0.60	3.01
    0.70	3.17
    0.80	3.36
    0.90	3.60
    1.00	4.17
    Name: gpaoverall, dtype: float64 
    ```

1.  显示 Series 子集的描述性统计数据：

    ```py
    gpaoverall.loc[gpaoverall.between(3,3.5)].head(5) 
    ```

    ```py
    personid
    135335   3.09
    370417   3.41
    684388   3.00
    984178   3.15
    730045   3.44
    Name: gpaoverall, dtype: float64 
    ```

    ```py
    gpaoverall.loc[gpaoverall.between(3,3.5)].count() 
    ```

    ```py
    1679 
    ```

    ```py
    gpaoverall.loc[(gpaoverall<2) | (gpaoverall>4)].sample(5, random_state=10) 
    ```

    ```py
    personid
    382527   1.66
    436086   1.86
    556245   4.02
    563504   1.94
    397487   1.84
    Name: gpaoverall, dtype: float64 
    ```

    ```py
    gpaoverall.loc[gpaoverall>gpaoverall.quantile(0.99)].\
    ...   agg(['count','min','max']) 
    ```

    ```py
    count     60.00
    min       3.98
    max       4.17
    Name: gpaoverall, dtype: float64 
    ```

1.  测试所有值中的某一条件。

检查 GPA 值是否超过 4，并确保所有值都大于或等于 0。（我们通常期望 GPA 在 0 到 4 之间。）还要统计缺失值的数量：

```py
(gpaoverall>4).any() # any person has GPA greater than 4 
```

```py
True 
```

```py
(gpaoverall>=0).all() # all people have GPA greater than or equal 0 
```

```py
False 
```

```py
(gpaoverall>=0).sum() # of people with GPA greater than or equal 0 
```

```py
6004 
```

```py
(gpaoverall==0).sum() # of people with GPA equal to 0 
```

```py
0 
```

```py
gpaoverall.isnull().sum() # of people with missing value for GPA 
```

```py
2980 
```

1.  基于不同列的值，显示 Series 的子集描述性统计数据。

显示 2020 年工资收入高于第 75 百分位的个人以及低于第 25 百分位的个人的高中平均 GPA：

```py
nls97.loc[nls97.wageincome20 > nls97.wageincome20.quantile(0.75),'gpaoverall'].mean() 
```

```py
3.0672837022132797 
```

```py
nls97.loc[nls97.wageincome20 < nls97.wageincome20.quantile(0.25),'gpaoverall'].mean() 
```

```py
2.6852676399026763 
```

1.  显示包含分类数据的 Series 的描述性统计和频率：

    ```py
    nls97.maritalstatus.describe() 
    ```

    ```py
    count      6675
    unique     5
    top        Married
    freq       3068
    Name: maritalstatus, dtype: object 
    ```

    ```py
    nls97.maritalstatus.value_counts() 
    ```

    ```py
    Married           3068
    Never-married     2767
    Divorced           669
    Separated          148
    Widowed             23
    Name: maritalstatus, dtype: int64 
    ```

一旦我们有了 Series，我们可以使用 pandas 的多种工具来计算该 Series 的描述性统计数据。

## 它是如何工作的……

Series 的`describe`方法非常有用，因为它能很好地展示连续变量的集中趋势和分布情况。查看每个十分位的值通常也很有帮助。我们在*步骤 2*中通过将 0.1 到 1.0 之间的值列表传递给 Series 的`quantile`方法来获得这些信息。

我们可以在 Series 的子集上使用这些方法。在*第 3 步*中，我们获得了 GPA 值在 3 到 3.5 之间的计数。我们还可以根据与汇总统计量的关系来选择值；例如，`gpaoverall>gpaoverall.quantile(0.99)` 选择 GPA 值大于第 99^(百分位)的值。然后，我们通过方法链将结果 Series 传递给 `agg` 方法，返回多个汇总统计量（`agg(['count','min','max'])`）。

有时，我们需要测试某个条件是否在 Series 中的所有值上都成立。`any` 和 `all` 方法对于此操作非常有用。`any` 当 Series 中至少有一个值满足条件时返回 `True`（例如，`(gpaoverall>4).any()`）。`all` 当 Series 中所有值都满足条件时返回 `True`。当我们将测试条件与 `sum` 链接时（`(gpaoverall>=0).sum()`），我们可以得到所有 `True` 值的计数，因为 pandas 在执行数值操作时将 `True` 视为 1。

`(gpaoverall>4)` 是一种简写方式，用于创建一个与 `gpaoverall` 具有相同索引的布尔 Series。当 `gpaoverall` 大于 4 时，其值为 `True`，否则为 `False`：

```py
(gpaoverall>4) 
```

```py
personid
135335    False
999406    False
151672    False
750699    False
781297    False
505861    False
368078    False
215605    False
643085    False
713757    False
Name: gpaoverall, Length: 8984, dtype: bool 
```

我们有时需要为通过另一个 Series 过滤后的 Series 生成汇总统计数据。在*第 5 步*中，我们通过计算工资收入高于第三四分位数的个体的平均高中 GPA，以及工资收入低于第一四分位数的个体的平均高中 GPA，来完成这项工作。

`describe` 方法对于连续变量（如 `gpaoverall`）最为有用，但在与分类变量（如 `maritalstatus`）一起使用时也能提供有价值的信息（见*第 6 步*）。它返回非缺失值的计数、不同值的数量、最常出现的类别以及该类别的频率。

然而，在处理分类数据时，`value_counts` 方法更为常用。它提供了 Series 中每个类别的频率。

## 还有更多……

使用 Series 是 pandas 数据清理任务中的基础，数据分析师很快就会发现，本篇中使用的工具已成为他们日常数据清理工作流的一部分。通常，从初始数据导入阶段到使用 Series 方法（如 `describe`、`mean`、`sum`、`isnull`、`all` 和 `any`）之间不会间隔太长时间。

## 另见

本章我们只是浅尝辄止地介绍了数据聚合的内容。我们将在*第九章*《聚合时修复脏数据》中更详细地讨论这一点。

# 更改 Series 值

在数据清理过程中，我们经常需要更改数据 Series 中的值或创建一个新的 Series。我们可以更改 Series 中的所有值，或仅更改部分数据的值。我们之前用来从 Series 获取值的大部分技术都可以用来更新 Series 的值，尽管需要进行一些小的修改。

## 准备工作

在本道菜谱中，我们将处理 NLS 数据中的总体高中 GPA 列。

## 如何实现…

我们可以为所有行或选择的行更改 pandas Series 中的值。我们可以通过对其他 Series 执行算术操作和使用汇总统计来更新 Series。让我们来看看这个过程：

1.  导入`pandas`并加载 NLS 数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97f.csv",
    low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  基于标量编辑所有值。

将`gpaoverall`乘以 100：

```py
nls97.gpaoverall.head() 
```

```py
personid
135335   3.09
999406   2.17
151672    NaN
750699   2.53
781297   2.82
Name: gpaoverall, dtype: float64 
```

```py
gpaoverall100 = nls97['gpaoverall'] * 100
gpaoverall100.head() 
```

```py
personid
135335   309.00
999406   217.00
151672      NaN
750699   253.00
781297   243.00
Name: gpaoverall, dtype: float64 
```

1.  使用索引标签设置值。

使用`loc`访问器通过索引标签指定要更改的值：

```py
nls97.loc[[135335], 'gpaoverall'] = 3
nls97.loc[[999406,151672,750699],'gpaoverall'] = 0
nls97.gpaoverall.head() 
```

```py
personid
135335   3.00
999406   0.00
151672   0.00
750699   0.00
781297   2.43
Name: gpaoverall, dtype: float64 
```

1.  使用运算符在多个 Series 之间设置值。

使用`+`运算符计算孩子的数量，这个数量是住在家里的孩子和不住在家里的孩子的总和：

```py
nls97['childnum'] = nls97.childathome + nls97.childnotathome
nls97.childnum.value_counts().sort_index() 
```

```py
0.00		23
1.00		1364
2.00		1729
3.00		1020
4.00		420
5.00		149
6.00		55
7.00		21
8.00		7
9.00		1
12.00		2
Name: childnum, dtype: int64 
```

1.  使用索引标签设置汇总统计值。

使用`loc`访问器从`100061`到`100292`选择`personid`：

```py
nls97.loc[135335:781297,'gpaoverall'] = nls97.gpaoverall.mean()
nls97.gpaoverall.head() 
```

```py
personid
135335   2.82
999406   2.82
151672   2.82
750699   2.82
781297   2.82
Name: gpaoverall, dtype: float64 
```

1.  使用位置设置值。

使用`iloc`访问器按位置选择。可以使用整数或切片表示法（`start:end:step`）放在逗号左边，指示应该更改的行。逗号右边使用整数来选择列。`gpaoverall`列在第 16 个位置（由于列索引是从零开始的，所以是第 15 个位置）：

```py
nls97.iloc[0, 15] = 2
nls97.iloc[1:4, 15] = 1
nls97.gpaoverall.head() 
```

```py
personid
135335   2.00
999406   1.00
151672   1.00
750699   1.00
781297   2.43
Name: gpaoverall, dtype: float64 
```

1.  在筛选后设置 GPA 值。

将所有超过`4`的 GPA 值更改为`4`：

```py
nls97.gpaoverall.nlargest() 
```

```py
personid
312410     4.17
639701     4.11
850001     4.10
279096     4.08
620216     4.07
Name: gpaoverall, dtype: float64 
```

```py
nls97.loc[nls97.gpaoverall>4, 'gpaoverall'] = 4
nls97.gpaoverall.nlargest() 
```

```py
personid
588585   4.00
864742   4.00
566248   4.00
990608   4.00
919755   4.00
Name: gpaoverall, dtype: float64 
```

前面的步骤展示了如何使用标量、算术操作和汇总统计值更新 Series 中的值。

## 它是如何工作的…

首先需要注意的是，在*步骤 2*中，pandas 将标量乘法进行了向量化。它知道我们想将标量应用于所有行。实质上，`nls97['gpaoverall'] * 100`创建了一个临时 Series，所有值都设置为 100，且拥有与`gpaoverall` Series 相同的索引。然后，它将`gpaoverall`与这个 100 值的 Series 相乘。这就是所谓的广播。

我们可以运用本章第一道菜谱中学到的许多内容，比如如何从 Series 中获取值，来选择特定的值进行更新。这里的主要区别是，我们使用 DataFrame 的`loc`和`iloc`访问器（`nls97.loc`），而不是 Series 的访问器（`nls97.gpaoverall.loc`）。这样做是为了避免令人头疼的`SettingwithCopyWarning`，该警告提示我们在 DataFrame 的副本上设置值。`nls97.gpaoverall.loc[[135335]] = 3`会触发这个警告，而`nls97.loc[[135335], 'gpaoverall'] = 3`则不会。

在*步骤 4*中，我们看到了 pandas 如何处理两个或多个 Series 之间的数值操作。加法、减法、乘法和除法等操作就像我们在标准 Python 中对标量进行的操作，只不过是向量化的。（这得益于 pandas 的索引对齐功能。请记住，同一个 DataFrame 中的 Series 会有相同的索引。）如果你熟悉 NumPy，那么你已经有了对这个过程的良好理解。

## 还有更多内容…

注意到`nls97.loc[[135335], 'gpaoverall']`返回一个 Series，而`nls97.loc[[135335], ['gpaoverall']]`返回一个 DataFrame，这是很有用的：

```py
type(nls97.loc[[135335], 'gpaoverall']) 
```

```py
<class 'pandas.core.series.Series'> 
```

```py
type(nls97.loc[[135335], ['gpaoverall']]) 
```

```py
<class 'pandas.core.frame.DataFrame'> 
```

如果`loc`访问器的第二个参数是字符串，它将返回一个 Series。如果它是一个列表，即使列表只包含一个项，它也会返回一个 DataFrame。

对于我们在本案例中讨论的任何操作，记得关注 pandas 如何处理缺失值。例如，在*步骤 4*中，如果`childathome`或`childnotathome`缺失，那么操作将返回`missing`。我们将在下一章的*识别和修复缺失值*案例中讨论如何处理这种情况。

## 另请参阅

*第三章*，*测量你的数据*，更详细地介绍了`loc`和`iloc`访问器的使用，特别是在*选择行*和*选择与组织列*的案例中。

# 有条件地更改 Series 值

更改 Series 值通常比前一个案例所示的更为复杂。我们经常需要根据该行数据中一个或多个其他 Series 的值来设置 Series 值。当我们需要根据*其他*行的值设置 Series 值时，这会变得更加复杂；比如，某个人的先前值，或者一个子集的平均值。我们将在本案例和下一个案例中处理这些复杂情况。

## 准备工作

在本案例中，我们将处理土地温度数据和 NLS 数据。

**数据说明**

土地温度数据集包含了来自全球 12,000 多个站点在 2023 年的平均温度数据（单位：摄氏度），尽管大多数站点位于美国。该原始数据集来自全球历史气候网络集成数据库，已由美国国家海洋和大气管理局（NOAA）提供给公众使用，网址：[`www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly`](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly)。

## 如何做……

我们将使用 NumPy 的`where`和`select`方法，根据该 Series 的值、其他 Series 的值和汇总统计来赋值。然后我们将使用`lambda`和`apply`函数来构建更复杂的赋值标准。我们开始吧：

1.  导入`pandas`和`numpy`，然后加载 NLS 和土地温度数据：

    ```py
    import pandas as pd
    import numpy as np
    nls97 = pd.read_csv("data/nls97f.csv", low_memory=False)
    nls97.set_index("personid", inplace=True)
    landtemps = pd.read_csv("data/landtemps2023avgs.csv") 
    ```

1.  使用 NumPy 的`where`函数创建一个包含两个值的分类 Series。

我们来快速检查一下`elevation`值的分布情况：

```py
landtemps.elevation.quantile(np.arange(0.2,1.1,0.2)) 
```

```py
0.2      47.9
0.4     190.5
0.6     395.0
0.8   1,080.0
1.0   9,999.0
Name: elevation, dtype: float64 
```

```py
landtemps['elevation_group'] = np.where(landtemps.elevation>landtemps.elevation.quantile(0.8),'High','Low')
landtemps.elevation_group = landtemps.elevation_group.astype('category')
landtemps.groupby(['elevation_group'],
  observed=False)['elevation'].\
  agg(['count','min','max']) 
```

```py
 count     min    max
elevation_group                  
High              2428   1,080    9,999
Low               9709    -350    1,080 
```

**注意**

你可能已经注意到，我们将`False`值传递给了`groupby`的`observed`属性。这是所有 pandas 版本在 2.1.0 之前的默认值。在后续的 pandas 版本中，`groupby`的默认`observed=True`。当`observed`为`True`且`groupby`中包含分类列时，只会显示观察到的值。这不会影响前一步的汇总统计结果。我仅在此处提到它，以提醒你即将发生的默认值变化。在本章其余部分我将忽略它。

1.  使用 NumPy 的`where`方法创建一个包含三个值的分类 Series。

将 80^(th)百分位以上的值设置为`'High'`，介于中位数和 80^(th)百分位之间的值设置为`'Medium'`，剩余的值设置为`'Low'`：

```py
landtemps['elevation_group'] = \
  np.where(landtemps.elevation>
    landtemps.elevation.quantile(0.8),'High',
    np.where(landtemps.elevation>landtemps.elevation.\
      median(),'Medium','Low'))
landtemps.elevation_group = landtemps.elevation_group.astype('category')
landtemps.groupby(['elevation_group'])['elevation'].\
  agg(['count','min','max']) 
```

```py
 count     min     max
elevation_group                        
High              2428    1,080   9,999
Low               6072     -350     271
Medium            3637      271   1,080 
```

1.  使用 NumPy 的`select`方法来评估一系列条件。

设置一组测试条件，并为结果设置另一个列表。我们希望 GPA 低于 2 且没有学位的个人归为一个类别，GPA 较高但没有学位的个人归为第二个类别，拥有学位但 GPA 较低的个人归为第三个类别，剩余的个人归为第四个类别：

```py
test = [(nls97.gpaoverall<2) &
  (nls97.highestdegree=='0\. None'),
   nls97.highestdegree=='0\. None',
   nls97.gpaoverall<2]
result = ['1\. Low GPA/No Dip','2\. No Diploma',
 '3\. Low GPA']
nls97['hsachieve'] = np.select(test, result, '4\. Did Okay')
nls97[['hsachieve','gpaoverall','highestdegree']].\
  sample(7, random_state=6) 
```

```py
 hsachieve    gpaoverall     highestdegree
personid                                              
102951     1\. Low GPA/No Dip           1.4           0\. None
583984           4\. Did Okay           3.3    2\. High School
116430           4\. Did Okay           NaN     3\. Associates
859586           4\. Did Okay           2.3    2\. High School
288527           4\. Did Okay           2.7      4\. Bachelors
161698           4\. Did Okay           3.4      4\. Bachelors
943703         2\. No Diploma           NaN           0\. None 
```

```py
nls97.hsachieve.value_counts().sort_index() 
```

```py
hsachieve
1\. Low GPA/No Dip              90
2\. No Diploma                 787
3\. Low GPA                    464
4\. Did Okay                  7643
Name: count, dtype: int64 
```

虽然 NumPy 的`select`方法在相对简单的条件赋值中非常方便，但当赋值操作较为复杂时，它可能会变得难以使用。在这种情况下，我们可以使用自定义函数，而不是使用`select`。

1.  让我们使用`apply`和自定义函数来执行与前一步相同的 Series 值赋值操作。我们创建一个名为`gethsachieve`的函数，包含将值分配给新变量`hsachieve2`的逻辑。我们将此函数传递给`apply`并指定`axis=1`，以便将该函数应用于所有行。

我们将在下一步中使用相同的技术来处理一个更复杂的赋值操作，该操作基于更多的列和条件。

```py
def gethsachieve(row):
  if (row.gpaoverall<2 and row.highestdegree=="0\. None"):
    hsachieve2 = "1\. Low GPA/No Dip"
  elif (row.highestdegree=="0\. None"):
    hsachieve2 = "2\. No Diploma"
  elif (row.gpaoverall<2):
    hsachieve2 = "3\. Low GPA"
  else:
    hsachieve2 = '4\. Did Okay'
  return hsachieve2
nls97['hsachieve2'] = nls97.apply(gethsachieve,axis=1)
nls97.groupby(['hsachieve','hsachieve2']).size() 
```

```py
 hsachieve          hsachieve2      
1\. Low GPA/No Dip  1\. Low GPA/No Dip      90
2\. No Diploma      2\. No Diploma         787
3\. Low GPA         3\. Low GPA            464
4\. Did Okay        4\. Did Okay          7643
dtype: int64 
```

请注意，在这一步中，我们得到了与前一步中`hsachieve`相同的`hsachieve2`值。

1.  现在，让我们使用`apply`和自定义函数进行更复杂的计算，该计算基于多个变量的值。

以下的`getsleepdeprivedreason`函数创建一个变量，用于根据调查对象可能因为什么原因导致每晚睡眠时间少于 6 小时来对其进行分类。我们根据 NLS 调查中关于受访者的就业状态、与受访者同住的孩子数、工资收入和最高完成的学业年级等信息来进行分类：

```py
def getsleepdeprivedreason(row):
  sleepdeprivedreason = "Unknown"
  if (row.nightlyhrssleep>=6):
    sleepdeprivedreason = "Not Sleep Deprived"
  elif (row.nightlyhrssleep>0):
    if (row.weeksworked20+row.weeksworked21 < 80):
      if (row.childathome>2):
        sleepdeprivedreason = "Child Rearing"
      else:
        sleepdeprivedreason = "Other Reasons"
    else:
      if (row.wageincome20>=62000 or row.highestgradecompleted>=16):
        sleepdeprivedreason = "Work Pressure"
      else:
        sleepdeprivedreason = "Income Pressure"
  else:
    sleepdeprivedreason = "Unknown"
  return sleepdeprivedreason 
```

1.  使用`apply`来对所有行运行该函数：

    ```py
    nls97['sleepdeprivedreason'] = nls97.apply(getsleepdeprivedreason, axis=1)
    nls97.sleepdeprivedreason = nls97.sleepdeprivedreason.astype('category')
    nls97.sleepdeprivedreason.value_counts() 
    ```

    ```py
    sleepdeprivedreason
    Not Sleep Deprived    5595
    Unknown               2286
    Income Pressure        453
    Work Pressure          324
    Other Reasons          254
    Child Rearing           72
    Name: count, dtype: int64 
    ```

1.  如果我们只需要处理特定的列，并且不需要将它们传递给自定义函数，我们可以使用`lambda`函数与`transform`。让我们通过使用`lambda`在一个语句中测试多个列来尝试这个方法。

`colenr`列记录了每个人在每年 2 月和 10 月的入学状态。我们想要测试是否有任何一列大学入学状态的值为`3. 4 年制大学`。使用`filter`创建一个包含`colenr`列的 DataFrame。然后，使用`transform`调用一个 lambda 函数，测试每个`colenr`列的第一个字符。（我们只需查看第一个字符，判断它是否为 3。）接着将其传递给`any`，评估是否有任何（一个或多个）列的第一个字符为 3。（由于空间限制，我们只显示 2000 年至 2004 年之间的大学入学状态，但我们会检查 1997 年到 2022 年之间所有大学入学状态列的值。）这可以通过以下代码看到：

```py
nls97.loc[[999406,750699],
  'colenrfeb00':'colenroct04'].T 
```

```py
personid                 999406                750699
colenrfeb00     1\. Not enrolled       1\. Not enrolled
colenroct00   3\. 4-year college       1\. Not enrolled
colenrfeb01   3\. 4-year college       1\. Not enrolled
colenroct01   2\. 2-year college       1\. Not enrolled
colenrfeb02     1\. Not enrolled     2\. 2-year college
colenroct02   3\. 4-year college       1\. Not enrolled
colenrfeb03   3\. 4-year college       1\. Not enrolled
colenroct03   3\. 4-year college       1\. Not enrolled
colenrfeb04   3\. 4-year college       1\. Not enrolled
colenroct04   3\. 4-year college       1\. Not enrolled 
```

```py
nls97['baenrollment'] = nls97.filter(like="colenr").\
...   transform(lambda x: x.str[0:1]=='3').\
...   any(axis=1)
nls97.loc[[999406,750699], ['baenrollment']].T 
```

```py
personid      999406  750699
baenrollment    True   False 
```

```py
nls97.baenrollment.value_counts() 
```

```py
baenrollment
False    4987
True     3997
Name: count, dtype: int64 
```

上述步骤展示了几种我们可以用来有条件地设置 Series 值的技巧。

## 它是如何工作的……

如果你曾在 SQL 或 Microsoft Excel 中使用过`if-then-else`语句，那么 NumPy 的`where`对你应该是熟悉的。它的形式是`where`（测试条件，`True`时的表达式，`False`时的表达式）。在*第 2 步*中，我们测试了每行的海拔值是否大于 80^(百分位数)的值。如果为`True`，则返回`'High'`；否则返回`'Low'`。这是一个基本的`if-then-else`结构。

有时，我们需要将一个测试嵌套在另一个测试中。在*第 3 步*中，我们为海拔创建了三个组：高，中和低。我们在`False`部分（第二个逗号之后）没有使用简单的语句，而是使用了另一个`where`语句。这将它从`else`语句变成了`else if`语句。它的形式是`where`（测试条件，`True`时的语句，`where`（测试条件，`True`时的语句，`False`时的语句））。

当然，可以添加更多嵌套的`where`语句，但并不推荐这样做。当我们需要评估一个稍微复杂一些的测试时，NumPy 的`select`方法非常有用。在*第 4 步*中，我们将测试的列表以及该测试的结果列表传递给了`select`。我们还为没有任何测试为`True`的情况提供了一个默认值`4. Did Okay`。当多个测试为`True`时，会使用第一个为`True`的测试。

一旦逻辑变得更加复杂，我们可以使用`apply`。DataFrame 的`apply`方法可以通过指定`axis=1`将 DataFrame 的每一行传递给一个函数。*第 5 步*演示了如何使用`apply`和用户定义的函数复现与*第 4 步*相同的逻辑。

在*第 6 步*和*第 7 步*中，我们创建了一个 Series，基于工作周数、与受访者同住的子女数量、工资收入和最高学历来分类缺乏睡眠的原因。如果受访者在 2020 年和 2021 年大部分时间没有工作，且有两个以上的孩子与其同住，则`sleepdeprivedreason`被设置为“育儿”。如果受访者在 2020 年和 2021 年大部分时间没有工作，且有两个或更少的孩子与其同住，则`sleepdeprivedreason`被设置为“其他原因”。如果受访者在 2020 年和 2021 年大部分时间有工作，则如果他们有高薪或完成了四年的大学学业，`sleepdeprivedreason`为“工作压力”，否则为“收入压力”。当然，这些分类有些人为，但它们确实展示了如何通过函数基于其他 Series 之间的复杂关系来创建 Series。

在*第 8 步*中，我们使用了`transform`调用一个 lambda 函数，测试每个大学入学值的第一个字符是否是 3。但首先，我们使用`filter`方法从 DataFrame 中选择所有的大学入学列。我们本可以将`lambda`函数与`apply`搭配使用以实现相同的结果，但`transform`通常更高效。

你可能注意到，我们在*第 2 步*和*第 3 步*中创建的新 Series 的数据类型被更改为`category`。这个新 Series 最初是`object`数据类型。我们通过将类型更改为`category`来减少内存使用。

我们在*第 2 步*中使用了另一个非常有用的方法，虽然是有点偶然的。`landtemps.groupby(['elevation_group'])`创建了一个 DataFrame 的`groupby`对象，我们将其传递给一个聚合（`agg`）函数。这样我们就可以获得每个`elevation_group`的计数、最小值和最大值，从而验证我们的分组分类是否按预期工作。

## 还有更多……

自从我有一个数据清理项目没有涉及 NumPy 的`where`或`select`语句，或者`lambda`或`apply`语句以来，已经有很长一段时间了。在某些时候，我们需要基于一个或多个其他 Series 的值来创建或更新一个 Series。熟练掌握这些技术是个好主意。

每当有一个内置的 pandas 函数能够完成我们的需求时，最好使用它，而不是使用`apply`。`apply`的最大优点是它非常通用且灵活，但也正因为如此，它比优化过的函数更占用资源。然而，当我们想要基于现有 Series 之间复杂的关系创建一个 Series 时，它是一个很好的工具。

执行*第 6 步*和*第 7 步*的另一种方式是将一个 lambda 函数添加到`apply`中。这会产生相同的结果：

```py
def getsleepdeprivedreason(childathome, nightlyhrssleep, wageincome, weeksworked20, weeksworked21, highestgradecompleted):
...   sleepdeprivedreason = "Unknown"
...   if (nightlyhrssleep>=6):
...     sleepdeprivedreason = "Not Sleep Deprived"
...   elif (nightlyhrssleep>0):
...     if (weeksworked16+weeksworked17 < 80):
...       if (childathome>2):
...         sleepdeprivedreason = "Child Rearing"
...       else:
...         sleepdeprivedreason = "Other Reasons"
...     else:
...       if (wageincome>=62000 or highestgradecompleted>=16):
...         sleepdeprivedreason = "Work Pressure"
...       else:
...         sleepdeprivedreason = "Income Pressure"
...   else:
...     sleepdeprivedreason = "Unknown"
...   return sleepdeprivedreason
...
nls97['sleepdeprivedreason'] = nls97.apply(lambda x: getsleepdeprivedreason(x.childathome, x.nightlyhrssleep, x.wageincome, x.weeksworked16, x.weeksworked17, x.highestgradecompleted), axis=1) 
```

这种方法的一个优点是，它更清晰地显示了哪些 Series 参与了计算。

## 另请参见

我们将在*第九章*《聚合时修复杂乱数据》中详细讲解 DataFrame 的`groupby`对象。我们在*第三章*《了解你的数据》中已经探讨了多种选择 DataFrame 列的技术，包括`filter`。

# 评估和清理字符串 Series 数据

Python 和 pandas 中有许多字符串清理方法。这是件好事。由于存储在字符串中的数据种类繁多，因此在进行字符串评估和操作时，拥有广泛的工具非常重要：当按位置选择字符串片段时，当检查字符串是否包含某个模式时，当拆分字符串时，当测试字符串长度时，当连接两个或更多字符串时，当改变字符串大小写时，等等。在本食谱中，我们将探索一些最常用于字符串评估和清理的方法。

## 准备工作

在本食谱中，我们将使用 NLS 数据。（实际上，NLS 数据对于这个食谱来说有点过于干净。为了演示如何处理带有尾随空格的字符串，我在`maritalstatus`列的值后添加了尾随空格。）

## 如何实现...

在本食谱中，我们将执行一些常见的字符串评估和清理任务。我们将使用`contains`、`endswith`和`findall`来分别搜索模式、尾随空格和更复杂的模式。

我们还将创建一个处理字符串值的函数，在将值分配给新 Series 之前，使用`replace`进行更简单的处理。让我们开始吧：

1.  导入`pandas`和`numpy`，然后加载 NLS 数据：

    ```py
    import pandas as pd
    import numpy as np
    nls97 = pd.read_csv("data/nls97ca.csv", low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  测试字符串中是否存在某个模式。

使用`contains`来检查`govprovidejobs`（政府应该提供就业）响应中的“绝对不”与“可能不”值。在`where`调用中，首先处理缺失值，确保它们不会出现在第一个`else`分支中（即第二个逗号后的部分）：

```py
nls97.govprovidejobs.value_counts() 
```

```py
2\. Probably           617
3\. Probably not       462
1\. Definitely         454
4\. Definitely not     300
Name: govprovidejobs, dtype: int64 
```

```py
nls97['govprovidejobsdefprob'] = \
  np.where(nls97.govprovidejobs.isnull(),
    np.nan,
      np.where(nls97.govprovidejobs.str.\
      contains("not"),"No","Yes"))
pd.crosstab(nls97.govprovidejobs, nls97.govprovidejobsdefprob) 
```

```py
govprovidejobsdefprob       No       Yes
govprovidejobs                          
1\. Definitely                0       454
2\. Probably	                0       617
3\. Probably not            462         0
4\. Definitely not          300         0 
```

1.  处理字符串中的前导或尾随空格。

创建一个永婚状态的 Series。首先，检查`maritalstatus`的值。注意有两个表示从未结婚的异常值。它们是“Never-married”后有一个额外的空格，而其他“Never-married”的值则没有尾随空格。使用`startswith`和`endswith`分别测试是否有前导空格或尾随空格。使用`strip`去除尾随空格后再测试永婚状态。`strip`去除前导和尾随空格（`lstrip`去除前导空格，`rstrip`去除尾随空格，所以在这个例子中，`rstrip`也能起作用）：

```py
nls97.maritalstatus.value_counts() 
```

```py
Married           3066
Never-married     2764
Divorced           663
Separated          154
Widowed             23
Never-married        2
Name: count, dtype: int64 
```

```py
nls97.maritalstatus.str.startswith(' ').any() 
```

```py
False 
```

```py
nls97.maritalstatus.str.endswith(' ').any() 
```

```py
True 
```

```py
nls97['evermarried'] = \
  np.where(nls97.maritalstatus.isnull(),np.nan,
    np.where(nls97.maritalstatus.str.\
      strip()=="Never-married","No","Yes"))
pd.crosstab(nls97.maritalstatus, nls97.evermarried) 
```

```py
evermarried        No    Yes
maritalstatus             
Divorced            0    663
Married             0   3066
Never-married    2764      0
Never-married       2      0
Separated           0    154
Widowed             0     23 
```

1.  使用`isin`将字符串值与值列表进行比较：

    ```py
    nls97['receivedba'] = \
      np.where(nls97.highestdegree.isnull(),np.nan,
        np.where(nls97.highestdegree.str[0:1].\
          isin(['4','5','6','7']),"Yes","No"))
    pd.crosstab(nls97.highestdegree, nls97.receivedba) 
    ```

    ```py
    receivedba             No      Yes
    highestdegree                  
    0\. None               953        0
    1\. GED               1146        0
    2\. High School       3667        0
    3\. Associates         737        0
    4\. Bachelors            0     1673
    5\. Masters              0      603
    6\. PhD                  0       54
    7\. Professional         0      120 
    ```

我们有时需要找出字符串中特定字符的位置。这有时是因为我们需要获取该点之前或之后的文本，或者以不同方式处理这些文本。让我们用之前处理过的“最高学历”列来尝试。我们将创建一个新列，该列不包含数字前缀。例如，*2. 高中*将变为*高中*。

1.  使用`find`获取`highestdegree`值中句点的位置，并提取该位置后的文本。

在此之前，我们将*99\. Unknown*分配给缺失值。虽然这不是必要的，但它帮助我们明确处理所有值（包括缺失值）的方式，同时增加了有用的复杂性。完成后，前导数字可以是 1 位或 2 位数字。

接下来，我们创建一个 lambda 函数`onlytext`，它将用于识别我们想要的文本的位置，然后利用它提取该文本。然后，我们使用`highestdegree` Series 的`transform`方法调用`onlytext`函数：

```py
nls97.fillna({"highestdegree":"99\. Unknown"},
  inplace=True)
onlytext = lambda x: x[x.find(".") + 2:]
highestdegreenonum = nls97.highestdegree.\
  astype(str).transform(onlytext)
highestdegreenonum.value_counts(dropna=False).\
  sort_index() 
```

```py
highestdegree
Associates       737
Bachelors       1673
GED             1146
High School     3667
Masters          603
None             953
PhD               54
Professional     120
Unknown           31
Name: count, dtype: int64 
```

你可能注意到，在句点和我们想要的文本开始之间有一个空格。为了处理这一点，`onlytext`函数会从句点后的两个空格处开始提取文本。

**注意**

为了实现我们想要的结果，我们并不需要给 lambda 函数命名。我们本可以直接在`transform`方法中输入 lambda 函数。然而，由于 NLS 数据中有多个列具有相似的前缀，创建一个可重用的函数来处理其他列是一个不错的选择。

有时我们需要查找字符串中某个特定值或某种类型的值（比如数字）出现的所有位置。pandas 的`findall`函数可以用来返回字符串中一个或多个匹配的值。它会返回一个包含满足给定条件的字符串片段的列表。在深入更复杂的例子之前，我们先做一个简单的示范。

使用`findall`计算每个`maritalstatus`值中`r`出现的次数，展示前几行数据。首先，展示`maritalstatus`的值，然后展示每个值对应的`findall`返回的列表：

```py
nls97.maritalstatus.head() 
```

```py
personid
100061           Married
100139           Married
100284     Never-married
100292               NaN
100583           Married
Name: maritalstatus, dtype: object 
```

```py
nls97.maritalstatus.head().str.findall("r") 
```

```py
personid
100061       [r, r]
100139       [r, r]
100284    [r, r, r]
100292          NaN
100583       [r, r]
Name: maritalstatus, dtype: object 
```

1.  我们还将展示`r`出现的次数。

使用`concat`将`maritalstatus`值、`findall`返回的列表和列表的长度显示在同一行：

```py
pd.concat([nls97.maritalstatus.head(),
   nls97.maritalstatus.head().str.findall("r"),
   nls97.maritalstatus.head().str.findall("r").\
       str.len()],
   axis=1) 
```

```py
 maritalstatus    maritalstatus   maritalstatus
personid                                            
100061             Married           [r, r]               2
100139             Married           [r, r]               2
100284       Never-married        [r, r, r]               3
100292                 NaN              NaN             NaN
100583             Married            [r, r]              2 
```

我们也可以使用`findall`返回不同类型的值。例如，我们可以使用正则表达式返回字符串中的所有数字列表。在接下来的几步中，我们将展示这一过程。

1.  使用`findall`创建一个包含所有数字的列表，该列表来源于`weeklyhrstv`（每周花费的电视观看时间）字符串。传递给`findall`的`"\d+"`正则表达式表示我们只想要数字：

    ```py
    pd.concat([nls97.weeklyhrstv.head(),\
    ...   nls97.weeklyhrstv.str.findall("\d+").head()], axis=1) 
    ```

    ```py
     weeklyhrstv             weeklyhrstv
    personid                                      
    100061     11 to 20 hours a week      [11, 20]
    100139     3 to 10 hours a week        [3, 10]
    100284     11 to 20 hours a week      [11, 20]
    100292     NaN                             NaN
    100583     3 to 10 hours a week        [3, 10] 
    ```

1.  使用`findall`创建的列表，从`weeklyhrstv`文本中创建一个数值 Series。

我们来定义一个函数，它为每个`weeklyhrstv`值提取`findall`创建的列表中的最后一个元素。`getnum`函数还会调整该数字，使其更接近这两个数字的中点，当存在多个数字时。然后我们使用`apply`调用这个函数，将`findall`为每个值创建的列表传递给它。`crosstab`显示新的`weeklyhrstvnum`列达到了我们的预期效果：

```py
def getnum(numlist):
...   highval = 0
...   if (type(numlist) is list):
...     lastval = int(numlist[-1])
...     if (numlist[0]=='40'):
...       highval = 45
...     elif (lastval==2):
...       highval = 1
...     else:
...       highval = lastval - 5
...   else:
...     highval = np.nan
...   return highval
...
nls97['weeklyhrstvnum'] = nls97.weeklyhrstv.str.\
...   findall("\d+").apply(getnum)
nls97[['weeklyhrstvnum','weeklyhrstv']].head(7) 
```

```py
 weeklyhrstvnum                 weeklyhrstv
personid                                           
100061                15       11 to 20 hours a week
100139                 5        3 to 10 hours a week
100284                15       11 to 20 hours a week
100292               NaN                         NaN
100583                 5        3 to 10 hours a week
100833                 5        3 to 10 hours a week
100931                 1  Less than 2 hours per week 
```

```py
pd.crosstab(nls97.weeklyhrstv, nls97.weeklyhrstvnum) 
```

```py
weeklyhrstvnum                1       5      15     25     35      45
weeklyhrstv                                                 
11 to 20 hours a week         0       0    1145      0      0       0
21 to 30 hours a week         0       0       0    299      0       0
3 to 10 hours a week          0    3625       0      0      0       0
31 to 40 hours a week         0       0       0      0    116       0
Less than 2 hrs.           1350       0       0      0      0       0
More than 40 hrs.             0       0       0      0      0     176 
```

1.  用替代值替换 Series 中的值。

`weeklyhrscomputer`（每周在计算机上花费的时间）Series 目前的值排序不太理想。我们可以通过将这些值替换为表示顺序的字母来解决此问题。我们将首先创建一个包含旧值的列表，以及一个包含新值的列表。然后，使用 Series 的 `replace` 方法将旧值替换为新值。每当 `replace` 在旧值列表中找到一个值时，它会将其替换为新值列表中相同位置的值：

```py
comphrsold = ['Less than 1 hour a week',
  '1 to 3 hours a week','4 to 6 hours a week',
  '7 to 9 hours a week','10 hours or more a week']
comphrsnew = ['A. Less than 1 hour a week',
  'B. 1 to 3 hours a week','C. 4 to 6 hours a week',
  'D. 7 to 9 hours a week','E. 10 hours or more a week']
nls97.weeklyhrscomputer.value_counts().sort_index() 
```

```py
1 to 3 hours a week         733
10 hours or more a week    3669
4 to 6 hours a week         726
7 to 9 hours a week         368
Less than 1 hour a week     296
Name: weeklyhrscomputer, dtype: int64 
```

```py
nls97.weeklyhrscomputer.replace(comphrsold, comphrsnew, inplace=True)
nls97.weeklyhrscomputer.value_counts().sort_index() 
```

```py
A. Less than 1 hour a week     296
B. 1 to 3 hours a week         733
C. 4 to 6 hours a week         726
D. 7 to 9 hours a week         368
E. 10 hours or more a week    3669
Name: weeklyhrscomputer, dtype: int64 
```

本食谱中的步骤演示了我们在 pandas 中可以执行的一些常见字符串评估和操作任务。

## 工作原理……

我们经常需要检查一个字符串，以查看其中是否存在某种模式。我们可以使用字符串的 `contains` 方法来实现这一点。如果我们确切知道期望的模式的位置，可以使用标准的切片符号 `[start:stop:step]` 来选择从起始位置到结束位置减一的文本。（`step` 的默认值为 1。）例如，在*步骤 4* 中，我们使用 `nls97.highestdegree.str[0:1]` 获取了 `highestdegree` 的第一个字符。然后，我们使用 `isin` 来测试第一个字符串是否出现在一个值列表中。 (`isin` 适用于字符数据和数值数据。)

有时，我们需要从字符串中提取多个满足条件的值。在这种情况下，`findall` 非常有用，因为它会返回一个满足条件的所有值的列表。它还可以与正则表达式配合使用，当我们寻找的内容比字面值更为通用时。在*步骤 8* 和 *步骤 9* 中，我们在寻找任何数字。

## 还有更多……

在根据另一个 Series 的值创建 Series 时，处理缺失值时需要特别小心。缺失值可能会满足 `where` 调用中的 `else` 条件，而这并非我们的意图。在*步骤 2*、*步骤 3* 和 *步骤 4* 中，我们确保通过在 `where` 调用的开始部分进行缺失值检查，正确处理了缺失值。

我们在进行字符串比较时，也需要注意字母的大小写。例如，`Probably` 和 `probably` 并不相等。解决这一问题的一种方法是在进行比较时，使用 `upper` 或 `lower` 方法，以防大小写的差异没有实际意义。`upper("Probably") == upper("PROBABLY")` 实际上是 `True`。

# 处理日期

处理日期通常并不简单。数据分析师需要成功地解析日期值，识别无效或超出范围的日期，填补缺失的日期，并计算时间间隔。在这些步骤中，每个环节都会遇到意想不到的挑战，但一旦我们成功解析了日期值并获得了 pandas 中的 datetime 值，就算是迈出了成功的一步。在本食谱中，我们将首先解析日期值，然后再处理接下来的其他挑战。

## 准备工作

在本食谱中，我们将处理 NLS 和 COVID-19 每日病例数据。COVID-19 每日数据包含每个国家每天的报告数据。（NLS 数据实际上对于这个目的来说过于干净。为了说明如何处理缺失的日期值，我将其中一个出生月份的值设置为缺失。）

**数据说明**

我们的《全球数据》提供了 COVID-19 的公共数据，链接：[`ourworldindata.org/covid-cases`](https://ourworldindata.org/covid-cases)。本食谱中使用的数据是于 2024 年 3 月 3 日下载的。

## 如何操作…

在这个食谱中，我们将把数字数据转换为日期时间数据，首先通过确认数据中是否包含有效的日期值，然后使用`fillna`来替换缺失的日期。接下来，我们将计算一些日期间隔；也就是说，计算 NLS 数据中受访者的年龄，以及 COVID-19 每日数据中自首例 COVID-19 病例以来的天数。让我们开始吧：

1.  导入`pandas`和`dateutils`中的`relativedelta`模块，然后加载 NLS 和 COVID-19 每日病例数据：

    ```py
    import pandas as pd
    from dateutil.relativedelta import relativedelta
    covidcases = pd.read_csv("data/covidcases.csv")
    nls97 = pd.read_csv("data/nls97c.csv")
    nls97.set_index("personid", inplace=True) 
    ```

1.  显示出生月份和年份的值。

请注意，出生月份有一个缺失值。除此之外，我们将用来创建`birthdate`序列的数据看起来相当干净：

```py
nls97[['birthmonth','birthyear']].isnull().sum() 
```

```py
birthmonth    1
birthyear     0
dtype: int64 
```

```py
nls97.birthmonth.value_counts(dropna=False).\
  sort_index() 
```

```py
birthmonth
1      815
2      693
3      760
4      659
5      689
6      720
7      762
8      782
9      839
10     765
11     763
12     736
NaN      1
Name: count, dtype: int64 
```

```py
nls97.birthyear.value_counts().sort_index() 
```

```py
1980     1691
1981     1874
1982     1841
1983     1807
1984     1771
Name: birthyear, dtype: int64 
```

1.  使用`fillna`方法为缺失的出生月份设置值。

将`birthmonth`的平均值（四舍五入为最接近的整数）传递给`fillna`。这将用`birthmonth`的平均值替换缺失的`birthmonth`值。请注意，现在又有一个人将`birthmonth`的值设为 6：

```py
nls97.fillna({"birthmonth":\
 int(nls97.birthmonth.mean())}, inplace=True)
nls97.birthmonth.value_counts(dropna=False).\
  sort_index() 
```

```py
birthmonth
1     815
2     693
3     760
4     659
5     689
6     721
7     762
8     782
9     839
10    765
11    763
12    736
Name: count, dtype: int64 
```

1.  使用`month`和年份`integers`来创建日期时间列。

我们可以将字典传递给 pandas 的`to_datetime`函数。字典需要包含年、月和日的键。请注意，`birthmonth`、`birthyear`和`birthdate`没有缺失值：

```py
nls97['birthdate'] = pd.to_datetime(dict(year=nls97.birthyear, month=nls97.birthmonth, day=15))
nls97[['birthmonth','birthyear','birthdate']].head() 
```

```py
 birthmonth     birthyear             birthdate
personid                                                 
100061     5              1980                 1980-05-15
100139     9              1983                 1983-09-15
100284     11             1984                 1984-11-15
100292     4              1982                 1982-04-15
100583     6              1980                 1980-06-15 
```

```py
nls97[['birthmonth','birthyear','birthdate']].isnull().sum() 
```

```py
birthmonth    0
birthyear     0
birthdate     0
dtype: int64 
```

1.  使用日期时间列计算年龄。

首先，定义一个函数，当给定起始日期和结束日期时，计算年龄。请注意，我们创建了一个**Timestamp**对象`rundate`，并将其赋值为`2024-03-01`，以用作年龄计算的结束日期：

```py
def calcage(startdate, enddate):
...   age = enddate.year - startdate.year
...   if (enddate.month<startdate.month or (enddate.month==startdate.month and enddate.day<startdate.day)):
...     age = age -1
...   return age
...
rundate = pd.to_datetime('2024-03-01')
nls97["age"] = nls97.apply(lambda x: calcage(x.birthdate, rundate), axis=1)
nls97.loc[100061:100583, ['age','birthdate']] 
```

```py
 age     birthdate
personid                   
100061     43    1980-05-15
100139     40    1983-09-15
100284     39    1984-11-15
100292     41    1982-04-15
100583     43    1980-06-15 
```

1.  我们可以改用`relativedelta`模块来计算年龄。我们只需要执行以下操作：

    ```py
    nls97["age2"] = nls97.\
      apply(lambda x: relativedelta(rundate,
        x.birthdate).years,
        axis=1) 
    ```

1.  我们应该确认我们得到的值与*步骤 5*中的值相同：

    ```py
    (nls97['age']!=nls97['age2']).sum() 
    ```

    ```py
    0 
    ```

    ```py
    nls97.groupby(['age','age2']).size() 
    ```

    ```py
    age  age2
    39   39      1463
    40   40      1795
    41   41      1868
    42   42      1874
    43   43      1690
    44   44       294
    dtype: int64 
    ```

1.  将字符串列转换为日期时间列。

`casedate`列是`object`数据类型，而不是`datetime`数据类型：

```py
covidcases.iloc[:, 0:6].dtypes 
```

```py
iso_code        object
continent       object
location        object
casedate        object
total_cases	   float64
new_cases       float64
dtype: object 
```

```py
covidcases.iloc[:, 0:6].sample(2, random_state=1).T 
```

```py
 628         26980
iso_code            AND           PRT
casedate     2020-03-15    2022-12-04
continent        Europe        Europe
location        Andorra      Portugal
total_cases           2     5,541,211
new_cases             1         3,963 
```

```py
covidcases['casedate'] = pd.to_datetime(covidcases.casedate, format='%Y-%m-%d')
covidcases.iloc[:, 0:6].dtypes 
```

```py
iso_code             object
continent            object
location             object
casedate     datetime64[ns]
total_cases         float64
new_cases           float64
dtype: object 
```

1.  显示日期时间列的描述性统计数据：

    ```py
    covidcases.casedate.nunique() 
    ```

    ```py
    214 
    ```

    ```py
    covidcases.casedate.describe() 
    ```

    ```py
    count                            36501
    mean     2021-12-16 05:41:07.954302720
    min                2020-01-05 00:00:00
    25%                2021-01-31 00:00:00
    50%                2021-12-12 00:00:00
    75%                2022-10-09 00:00:00
    max                2024-02-04 00:00:00
    Name: casedate, dtype: object 
    ```

1.  创建一个`timedelta`对象来捕捉日期间隔。

对于每一天，计算自报告首例病例以来，每个国家的天数。首先，创建一个 DataFrame，显示每个国家新病例的第一天，然后将其与完整的 COVID-19 病例数据合并。接着，对于每一天，计算从`firstcasedate`到`casedate`的天数：

```py
firstcase = covidcases.loc[covidcases.new_cases>0,['location','casedate']].\
...   sort_values(['location','casedate']).\
...   drop_duplicates(['location'], keep='first').\
...   rename(columns={'casedate':'firstcasedate'})
covidcases = pd.merge(covidcases, firstcase, left_on=['location'], right_on=['location'], how="left")
covidcases['dayssincefirstcase'] = covidcases.casedate - covidcases.firstcasedate
covidcases.dayssincefirstcase.describe() 
```

```py
count                          36501
mean     637 days 01:36:55.862579112
std      378 days 15:34:06.667833980
min                  0 days 00:00:00
25%                315 days 00:00:00
50%                623 days 00:00:00
75%                931 days 00:00:00
max               1491 days 00:00:00
Name: dayssincefirstcase, dtype: object 
```

本食谱展示了如何解析日期值并创建日期时间序列，以及如何计算时间间隔。

## 如何操作…

在 pandas 中处理日期时，第一项任务是将其正确转换为 pandas datetime Series。在 *步骤 3*、*4* 和 *8* 中，我们处理了一些最常见的问题：缺失值、从整数部分转换日期和从字符串转换日期。`birthmonth` 和 `birthyear` 在 NLS 数据中是整数。我们确认这些值是有效的日期月份和年份。如果，举例来说，存在月份值为 0 或 20，则转换为 pandas datetime 将失败。

`birthmonth` 或 `birthyear` 的缺失值将导致 `birthdate` 缺失。我们使用 `fillna` 填充了 `birthmonth` 的缺失值，将其分配为 `birthmonth` 的平均值。在 *步骤 5* 中，我们使用新的 `birthdate` 列计算了每个人截至 2024 年 3 月 1 日的年龄。我们创建的 `calcage` 函数会根据出生日期晚于 3 月 1 日的个体进行调整。

数据分析师通常会收到包含日期字符串的文件。当发生这种情况时，`to_datetime` 函数是分析师的得力助手。它通常足够智能，能够自动推断出字符串日期数据的格式，而无需我们明确指定格式。然而，在 *步骤 8* 中，我们告诉 `to_datetime` 使用 `%Y-%m-%d` 格式处理我们的数据。

*步骤 9* 告诉我们有 214 个独特的日期报告了 COVID-19 病例。第一次报告的日期是 2020 年 1 月 5 日，最后一次报告的日期是 2024 年 2 月 4 日。

*步骤 10* 中的前两条语句涉及了一些技巧（排序和去重），我们将在 *第九章*《汇总时修复杂乱数据》和 *第十章*《合并 DataFrame 时处理数据问题》中详细探讨。这里你只需要理解目标：创建一个按 `location`（国家）每行数据表示的 DataFrame，并记录首次报告的 COVID-19 病例日期。我们通过仅选择全数据中 `new_cases` 大于 0 的行来做到这一点，然后按 `location` 和 `casedate` 排序，并保留每个 `location` 的第一行。接着，我们将 `casedate` 改名为 `firstcasedate`，然后将新的 `firstcase` DataFrame 与 COVID-19 日病例数据合并。

由于 `casedate` 和 `firstcasedate` 都是日期时间列，从后者减去前者将得到一个 timedelta 值。这为我们提供了一个 Series，表示每个国家每个报告日期自 `new_cases` 首次出现后的天数。报告病例日期（`casedate`）和首次病例日期（`firstcasedate`）之间的最大持续时间（`dayssincefirstcase`）是 1491 天，约为 4 年多。这个间隔计算对于我们想要按病毒在一个国家明显存在的时间来追踪趋势，而不是按日期来追踪趋势时非常有用。

## 另请参见

与其在*步骤 10*中使用`sort_values`和`drop_duplicates`，我们也可以使用`groupby`来实现类似的结果。在*第九章*中，我们将深入探索`groupby`，*在聚合时修复杂乱数据*。我们还在*步骤 10*中做了一个合并。*第十章*，*合并 DataFrame 时解决数据问题*，将专门讨论这个主题。

# 使用 OpenAI 进行 Series 操作

本章之前食谱中演示的许多 Series 操作可以借助 AI 工具完成，包括通过 PandasAI 与 OpenAI 的大型语言模型一起使用。在这个食谱中，我们研究如何使用 PandasAI 查询 Series 的值，创建新的 Series，有条件地设置 Series 的值，并对 DataFrame 进行一些基础的重塑。

## 准备就绪

在这个食谱中，我们将再次使用 NLS 和 COVID-19 每日数据。我们还将使用 PandasAI，它可以通过`pip install pandasai`安装。你还需要从[openai.com](https://openai.com)获取一个令牌，以便向 OpenAI API 发送请求。

## 如何操作...

以下步骤创建一个 PandasAI `SmartDataframe`对象，然后使用该对象的聊天方法提交一系列 Series 操作的自然语言指令：

1.  我们首先需要从 PandasAI 导入`OpenAI`和`SmartDataframe`模块。我们还需要实例化一个`llm`对象：

    ```py
    import pandas as pd
    from pandasai.llm.openai import OpenAI
    from pandasai import SmartDataframe
    llm = OpenAI(api_token="Your API Token") 
    ```

1.  我们加载 NLS 和 COVID-19 数据并创建一个`SmartDataframe`对象。我们传入`llm`对象以及一个 pandas DataFrame：

    ```py
    covidcases = pd.read_csv("data/covidcases.csv")
    nls97 = pd.read_csv("data/nls97f.csv")
    nls97.set_index("personid", inplace=True)
    nls97sdf = SmartDataframe(nls97, config={"llm": llm}) 
    ```

1.  现在，我们准备好在我们的`SmartDataframe`上生成 Series 的汇总统计信息。我们可以请求单个 Series 的平均值，或者多个 Series 的平均值：

    ```py
    nls97sdf.chat("Show average of gpaoverall") 
    ```

    ```py
    2.8184077281812128 
    ```

    ```py
    nls97sdf.chat("Show average for each weeks worked column") 
    ```

    ```py
     Average Weeks Worked
                      0
    weeksworked00 26.42
    weeksworked01 29.78
    weeksworked02 31.83
    weeksworked03 33.51
    weeksworked04 35.10
    weeksworked05 37.34
    weeksworked06 38.44
    weeksworked07 39.29
    weeksworked08 39.33
    weeksworked09 37.51
    weeksworked10 37.12
    weeksworked11 38.06
    weeksworked12 38.15
    weeksworked13 38.79
    weeksworked14 38.73
    weeksworked15 39.67
    weeksworked16 40.19
    weeksworked17 40.37
    weeksworked18 40.01
    weeksworked19 41.22
    weeksworked20 38.35
    weeksworked21 36.17
    weeksworked22 11.43 
    ```

1.  我们还可以通过另一个 Series 来汇总 Series 的值，通常是一个分类的 Series：

    ```py
    nls97sdf.chat("Show satmath average by gender") 
    ```

    ```py
     Female   Male
    0  486.65 516.88 
    ```

1.  我们还可以通过`SmartDataframe`的`chat`方法创建一个新的 Series。我们不需要使用实际的列名。例如，PandasAI 会自动识别我们想要的是`childathome` Series，当我们写下*child at home*时：

    ```py
    nls97sdf = nls97sdf.chat("Set childnum to child at home plus child not at home")
    nls97sdf[['childnum','childathome','childnotathome']].\
      sample(5, random_state=1) 
    ```

    ```py
     childnum  childathome  childnotathome
    personid                                      
    211230        2.00         2.00            0.00
    990746        3.00         3.00            0.00
    308169        3.00         1.00            2.00
    798458         NaN          NaN             NaN
    312009         NaN          NaN             NaN 
    ```

1.  我们可以使用`chat`方法有条件地创建 Series 值：

    ```py
    nls97sdf = nls97sdf.chat("evermarried is 'No' when maritalstatus is 'Never-married', else 'Yes'")
    nls97sdf.groupby(['evermarried','maritalstatus']).size() 
    ```

    ```py
    evermarried  maritalstatus
    No           Never-married    2767
    Yes          Divorced          669
                 Married          3068
                 Separated         148
                 Widowed            23
    dtype: int64 
    ```

1.  PandasAI 对你在这里使用的语言非常灵活。例如，以下内容提供了与*步骤 6*相同的结果：

    ```py
    nls97sdf = nls97sdf.chat("if maritalstatus is 'Never-married' set evermarried2 to 'No', otherwise 'Yes'")
    nls97sdf.groupby(['evermarried2','maritalstatus']).size() 
    ```

    ```py
    evermarried2  maritalstatus
    No            Never-married    2767
    Yes           Divorced          669
                  Married          3068
                  Separated         148
                  Widowed            23
    dtype: int64 
    ```

1.  我们可以对多个同名的列进行计算：

    ```py
    nls97sdf = nls97sdf.chat("set weeksworkedavg to the average for weeksworked columns") 
    ```

这将计算所有`weeksworked00`到`weeksworked22`列的平均值，并将其分配给一个名为`weeksworkedavavg`的新列。

1.  我们可以根据汇总统计轻松地填补缺失的值：

    ```py
    nls97sdf.gpaenglish.describe() 
    ```

    ```py
    count   5,798
    mean      273
    std        74
    min         0
    25%       227
    50%       284
    75%       323
    max       418
    Name: gpaenglish, dtype: float64 
    ```

    ```py
    nls97sdf = nls97sdf.chat("set missing gpaenglish to the average")
    nls97sdf.gpaenglish.describe() 
    ```

    ```py
    count   8,984
    mean      273
    std        59
    min         0
    25%       264
    50%       273
    75%       298
    max       418
    Name: gpaenglish, dtype: float64 
    ```

1.  我们还可以使用 PandasAI 进行一些重塑，类似于我们在之前的食谱中所做的。回顾一下，我们处理了 COVID-19 病例数据，并希望获取每个国家的第一行数据。让我们首先以传统方式做一个简化版本：

    ```py
    firstcase = covidcases.\
      sort_values(['location','casedate']).\
      drop_duplicates(['location'], keep='first')
    firstcase.set_index('location', inplace=True)
    firstcase.shape 
    ```

    ```py
    (231, 67) 
    ```

    ```py
    firstcase[['iso_code','continent','casedate',
      'total_cases','new_cases']].head(2).T 
    ```

    ```py
    location       Afghanistan        Albania
    iso_code               AFG            ALB
    continent             Asia         Europe
    casedate        2020-03-01     2020-03-15
    total_cases           1.00          33.00
    new_cases             1.00          33.00 
    ```

1.  我们可以通过创建一个`SmartDataframe`并使用`chat`方法来获得相同的结果。这里使用的自然语言非常简单，*显示每个国家的第一个 casedate、location 和其他值*：

    ```py
    covidcasessdf = SmartDataframe(covidcases, config={"llm": llm})
    firstcasesdf = covidcasessdf.chat("Show first casedate and location and other values for each country.")
    firstcasesdf.shape 
    ```

    ```py
    (231, 7) 
    ```

    ```py
    firstcasesdf[['location','continent','casedate',
      'total_cases','new_cases']].head(2).T 
    ```

    ```py
    iso_code                  ABW             AFG
    location                Aruba     Afghanistan
    continent       North America            Asia
    casedate           2020-03-22      2020-03-01
    total_cases              5.00            1.00
    new_cases                5.00            1.00 
    ```

请注意，PandasAI 会智能地选择需要获取的列。我们只获取我们需要的列，而不是所有列。我们也可以直接将我们想要的列名传递给`chat`。

这就是一点 PandasAI 和 OpenAI 的魔力！通过传递一句相当普通的句子给`chat`方法，就完成了所有的工作。

## 它是如何工作的…

使用 PandasAI 时，大部分工作其实就是导入相关库，并实例化大型语言模型和 `SmartDataframe` 对象。完成这些步骤后，只需向 `SmartDataframe` 的 `chat` 方法发送简单的句子，就足以总结 Series 值并创建新的 Series。

PandasAI 擅长从 Series 中生成简单的统计信息。我们甚至不需要精确记住 Series 名称，正如我们在*步骤 3*中所见。我们可能使用的自然语言往往比传统的 pandas 方法（如 `groupby`）更直观。在*步骤 4*中传递给 `chat` 的*按性别显示 satmath 平均值*就是一个很好的例子。

对 Series 进行的操作，包括创建新的 Series，也是相当简单的。在*步骤 5*中，我们通过指示 `SmartDataframe` 将住在家中的孩子数与不住在家中的孩子数相加，创建了一个表示孩子总数的 Series（*childnum*）。我们甚至没有提供字面上的 Series 名称，*childathome* 和 *childnotathome*。PandasAI 会自动理解我们的意思。

*步骤 6* 和 *7* 展示了使用自然语言进行 Series 操作的灵活性。如果我们在*步骤 6*中将*evermarried 为 ‘No’ 当 maritalstatus 为 ‘Never-married’，否则为 ‘Yes’*传递给`chat`，或者在*步骤 7*中将*如果 maritalstatus 为 ‘Never-married’，则将 evermarried2 设置为 ‘No’，否则为 ‘Yes’*传递给`chat`，我们都会得到相同的结果。

我们还可以通过简单的自然语言指令对 DataFrame 进行较为广泛的重塑，正如在*步骤 11*中所示。我们将*and other values*添加到指令中，以获取除了*casedate*之外的列。PandasAI 还会自动识别出*location*作为索引是有意义的。

## 还有更多…

鉴于 PandasAI 工具仍然非常新，数据科学家们现在才开始弄清楚如何将这些工具最佳地集成到我们的数据清理和分析工作流程中。PandasAI 有两个明显的应用场景：1）检查我们以传统方式进行的 Series 操作的准确性；2）在 pandas 或 NumPy 工具不够直观时（如 pandas 的 `groupby` 或 NumPy 的 `where` 函数），以更直观的方式进行 Series 操作。

PandasAI 还可以用于构建交互式界面来查询数据存储，如数据仪表盘。我们可以使用 AI 工具帮助终端用户更有效地查询组织数据。正如我们在*第三章*《衡量你的数据》中所看到的，PandasAI 在快速创建可视化方面也非常出色。

## 另请参见

在*第九章*，*聚合数据时修复混乱数据*中，我们将进行更多的数据聚合操作，包括跨行聚合数据和重新采样日期和时间数据。

# 总结

本章探讨了多种 pandas Series 方法，用于探索和处理不同类型的数据：数值、字符串和日期。我们学习了如何从 Series 中获取值以及如何生成摘要统计信息。我们还了解了如何更新 Series 中的值，以及如何针对数据子集或根据条件进行更新。我们还探讨了处理字符串或日期 Series 时的特定挑战，以及如何使用 Series 方法来应对这些挑战。最后，我们看到如何利用 PandasAI 来探索和修改 Series。在下一章，我们将探索如何识别和修复缺失值。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`discord.gg/p8uSgEAETX`](https://discord.gg/p8uSgEAETX )

![](img/QR_Code10336218961138498953.png)
