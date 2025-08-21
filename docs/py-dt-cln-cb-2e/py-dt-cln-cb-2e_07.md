

# 第七章：识别并修复缺失值

我想我可以代表许多数据分析师和科学家来说，鲜少有什么看似微小而琐碎的事情能像缺失值那样对我们的分析产生如此大的影响。我们花费大量时间担心缺失值，因为它们可能对我们的分析产生戏剧性的、令人惊讶的影响。尤其是当缺失值不是随机的，而是与因变量相关时，情况尤其如此。例如，如果我们正在做一个收入的纵向研究，但教育水平较低的个体每年更可能跳过收入问题，那么很可能会对我们关于教育水平的参数估计产生偏差。

当然，识别缺失值只解决了问题的一部分。我们还需要决定如何处理它们。我们是删除任何包含缺失值的观测值，还是基于像均值这样的样本统计量插补一个值？或者，基于更有针对性的统计量，例如某个类别的均值，来插补？对于时间序列或纵向数据，我们是否应该考虑用最接近的时间值来填补？或者，是否应该使用更复杂的多变量技术进行插补，可能是基于回归或 *k*-最近邻方法？

对于前面所有的问题，答案是“是的”。在某个阶段，我们会希望使用这些技术中的每一个。我们希望在做出最终缺失值插补选择时，能够回答为什么或为什么不使用这些可能性。每种方法都将根据情况有其合理性。

本章将介绍识别每个变量的缺失值以及识别缺失值较多的观测值的技术。接着，我们将探讨一些插补策略，例如将缺失值设置为整体均值、某个特定类别的均值或前向填充。我们还将研究多变量插补技术，并讨论它们在何种情况下是合适的。

具体来说，本章将探讨以下几种方法：

+   识别缺失值

+   清理缺失值

+   使用回归进行插补

+   使用 *k*-最近邻方法进行插补

+   使用随机森林进行插补

+   使用 PandasAI 进行插补

# 技术要求

你将需要 pandas、NumPy 和 Matplotlib 来完成本章中的示例。我使用的是 pandas 2.1.4，但代码同样适用于 pandas 1.5.3 或更高版本。

本章中的代码可以从本书的 GitHub 仓库下载：[`github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition`](https://github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition)。

# 识别缺失值

由于识别缺失值是分析师工作流程中的重要部分，我们使用的任何工具都需要使定期检查缺失值变得容易。幸运的是，pandas 使得识别缺失值变得非常简单。

## 准备就绪

本章我们将使用 **国家纵向调查**（**NLS**）数据。NLS 数据每个调查响应者有一条观察记录。每年的就业、收入和大学入学数据都存储在带有后缀的列中，后缀表示年份，如 `weeksworked21` 和 `weeksworked22` 分别代表 2021 年和 2022 年的工作周数。

我们还将再次使用 COVID-19 数据。该数据集包含每个国家的观察值，记录了总 COVID-19 病例和死亡人数，以及每个国家的人口统计数据。

**数据说明**

青年国家纵向调查由美国劳工统计局进行。此调查始于 1997 年，针对的是 1980 至 1985 年出生的群体，每年进行一次跟踪，直到 2023 年。对于此项工作，我从调查的数百个数据项中提取了关于年级、就业、收入和对政府态度的 104 个变量。NLS 数据可以从 [nlsinfo.org/](https://nlsinfo.org) 下载。

*Our World in Data* 提供了用于公共使用的 COVID-19 数据，网址为 [`ourworldindata.org/covid-cases`](https://ourworldindata.org/covid-cases)。该数据集包括总病例和死亡人数、已做测试数量、医院床位数以及人口统计数据，如中位年龄、国内生产总值和预期寿命。此处使用的数据集是在 2024 年 3 月 3 日下载的。

## 如何操作...

我们将使用 pandas 函数来识别缺失值和逻辑缺失值（即尽管数据本身不缺失，但却代表缺失的非缺失值）。

1.  让我们从加载 NLS 和 COVID-19 数据开始：

    ```py
    import pandas as pd
    import numpy as np
    nls97 = pd.read_csv("data/nls97g.csv",
        low_memory=False)
    nls97.set_index("personid", inplace=True)
    covidtotals = pd.read_csv("data/covidtotalswithmissings.csv",
        low_memory=False)
    covidtotals.set_index("iso_code", inplace=True) 
    ```

1.  接下来，我们统计每个变量的缺失值数量。我们可以使用 `isnull` 方法来测试每个值是否缺失。如果值缺失，它将返回 True，否则返回 False。然后，我们可以使用 `sum` 来统计 True 值的数量，因为 `sum` 会将每个 True 值视为 1，False 值视为 0。我们指定 `axis=0` 来对列进行求和，而不是对行进行求和：

    ```py
    covidtotals.shape 
    ```

    ```py
    (231, 16) 
    ```

    ```py
    demovars = ['pop_density','aged_65_older',
       'gdp_per_capita','life_expectancy','hum_dev_ind']
    covidtotals[demovars].isnull().sum(axis=0) 
    ```

    ```py
    pop_density        22
    aged_65_older      43
    gdp_per_capita     40
    life_expectancy     4
    hum_dev_ind        44
    dtype: int64 
    ```

231 个国家中有 43 个国家的 `aged_65_older` 变量存在空值。几乎所有国家都有 `life_expectancy` 数据。

1.  如果我们想要了解每一行的缺失值数量，可以在求和时指定 `axis=1`。以下代码创建了一个 Series，`demovarsmisscnt`，它记录了每个国家人口统计变量的缺失值数量。178 个国家的所有变量都有值，但 16 个国家缺少 5 个变量中的 4 个值，4 个国家所有变量都缺少值：

    ```py
    demovarsmisscnt = covidtotals[demovars].isnull().sum(axis=1)
    demovarsmisscnt.value_counts().sort_index() 
    ```

    ```py
    0    178
    1      8
    2     14
    3     11
    4     16
    5      4
    Name: count, dtype: int64 
    ```

1.  让我们看一看一些缺失值超过 4 的国家。这些国家几乎没有人口统计数据：

    ```py
    covidtotals.loc[demovarsmisscnt>=4, ['location'] + demovars].\
      sample(5, random_state=1).T 
    ```

    ```py
    iso_code                      FLK                       SPM  \
    location         Falkland Islands  Saint Pierre and Miquelon 
    pop_density                   NaN                        NaN 
    aged_65_older                 NaN                        NaN 
    gdp_per_capita                NaN                        NaN 
    life_expectancy                81                         81 
    hum_dev_ind                   NaN                        NaN 
    iso_code              GGY         MSR           COK
    location         Guernsey  Montserrat  Cook Islands
    pop_density           NaN         NaN           NaN
    aged_65_older         NaN         NaN           NaN
    gdp_per_capita        NaN         NaN           NaN
    life_expectancy       NaN          74            76
    hum_dev_ind           NaN         NaN           NaN 
    ```

1.  我们还将检查总病例和死亡人数的缺失值。每百万人的病例和每百万人的死亡人数分别有一个缺失值：

    ```py
    totvars = ['location','total_cases_pm','total_deaths_pm']
    covidtotals[totvars].isnull().sum(axis=0) 
    ```

    ```py
    location           0
    total_cases_pm     1
    total_deaths_pm    1
    dtype: int64 
    ```

1.  我们可以轻松检查某个国家是否同时缺失每百万的病例数和每百万的死亡人数。我们看到有`230`个国家两者都没有缺失，而仅有一个国家同时缺失这两项数据：

    ```py
    totvarsmisscnt = covidtotals[totvars].isnull().sum(axis=1)
    totvarsmisscnt.value_counts().sort_index() 
    ```

    ```py
    0    230
    2      1
    Name: count, dtype: int64 
    ```

有时我们会遇到需要转换为实际缺失值的逻辑缺失值。这发生在数据集设计者使用有效值作为缺失值的代码时。这些通常是像 9、99 或 999 这样的值，取决于变量允许的数字位数。或者它可能是一个更复杂的编码方案，其中有不同的代码表示缺失值的不同原因。例如，在 NLS 数据集中，代码揭示了受访者未回答问题的原因：-3 是无效跳过，-4 是有效跳过，-5 是非访谈。

1.  NLS 数据框的最后 4 列包含了关于受访者母亲和父亲完成的最高学位、父母收入以及受访者出生时母亲年龄的数据。我们将从 `motherhighgrade` 列开始，检查这些列的逻辑缺失值。

    ```py
    nlsparents = nls97.iloc[:,-4:]
    nlsparents.loc[nlsparents.motherhighgrade.between(-5,-1),
       'motherhighgrade'].value_counts() 
    ```

    ```py
    motherhighgrade
    -3    523
    -4    165
    Name: count, dtype: int64 
    ```

1.  有 523 个无效跳过值和 165 个有效跳过值。我们来看几个至少在这四个变量中有一个非响应值的个体：

    ```py
    nlsparents.loc[nlsparents.transform(lambda x: x.between(-5,-1)).any(axis=1)] 
    ```

    ```py
     motherage   parentincome   fatherhighgrade   motherhighgrade
    personid                                                         
    135335             26             -3                16                 8
    999406             19             -4                17                15
    151672             26          63000                -3                12
    781297             34             -3                12                12
    613800             25             -3                -3                12
                      ...            ...               ...               ...
    209909             22           6100                -3                11
    505861             21             -3                -4                13
    368078             19             -3                13                11
    643085             21          23000                -3                14
    713757             22          23000                -3                14
    [3831 rows x 4 columns] 
    ```

1.  对于我们的分析，非响应的原因并不重要。我们只需要统计每列的非响应数量，无论非响应的原因是什么：

    ```py
    nlsparents.transform(lambda x: x.between(-5,-1)).sum() 
    ```

    ```py
    motherage            608
    parentincome        2396
    fatherhighgrade     1856
    motherhighgrade      688
    dtype: int64 
    ```

1.  在我们进行分析之前，应该将这些值设置为缺失值。我们可以使用 `replace` 将所有介于 -5 和 -1 之间的值设置为缺失值。当我们检查实际缺失值时，我们得到预期的计数：

    ```py
    nlsparents.replace(list(range(-5,0)), np.nan, inplace=True)
    nlsparents.isnull().sum() 
    ```

    ```py
    motherage             608
    parentincome         2396
    fatherhighgrade      1856
    motherhighgrade       688
    dtype: int64 
    ```

## 它是如何工作的…

在*步骤 8*和*步骤 9*中，我们充分利用了 lambda 函数和 `transform` 来跨多个列搜索指定范围的值。`transform` 的工作方式与 `apply` 类似。两者都是 DataFrame 或 Series 的方法，允许我们将一个或多个数据列传递给一个函数。在这种情况下，我们使用了 lambda 函数，但我们也可以使用命名函数，就像我们在*第六章*《使用 Series 操作清理和探索数据》中的*条件性更改 Series 值*教程中所做的那样。

这个教程展示了一些非常实用的 pandas 技巧，用于识别每个变量的缺失值数量以及具有大量缺失值的观测数据。我们还研究了如何找到逻辑缺失值并将其转换为实际缺失值。接下来，我们将首次探索如何清理缺失值。

# 清理缺失值

在本教程中，我们介绍了一些最直接处理缺失值的方法。这包括删除缺失值的观测数据；为缺失值分配样本范围内的统计量（如均值）；以及基于数据的适当子集的均值为缺失值分配值。

## 如何操作…

我们将查找并移除来自 NLS 数据中那些主要缺失关键变量数据的观测值。我们还将使用 pandas 方法为缺失值分配替代值，例如使用变量均值：

1.  让我们加载 NLS 数据并选择一些教育数据。

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index("personid", inplace=True)
    schoolrecordlist = ['satverbal','satmath','gpaoverall',
      'gpaenglish',  'gpamath','gpascience','highestdegree',
      'highestgradecompleted']
    schoolrecord = nls97[schoolrecordlist]
    schoolrecord.shape 
    ```

    ```py
    (8984, 8) 
    ```

1.  我们可以使用前面章节中探讨的技术来识别缺失值。`schoolrecord.isnull().sum(axis=0)` 会给出每列的缺失值数量。绝大多数观测值在 `satverbal` 上存在缺失值，共 7,578 个缺失值（总共 8,984 个观测值）。只有 31 个观测值在 `highestdegree` 上有缺失值：

    ```py
    schoolrecord.isnull().sum(axis=0) 
    ```

    ```py
    satverbal			7578
    satmath			7577
    gpaoverall			2980
    gpaenglish			3186
    gpamath			3218
    gpascience			3300
    highestdegree		31
    highestgradecompleted	2321
    dtype: int64 
    ```

1.  我们可以创建一个 Series `misscnt`，它记录每个观测值的缺失变量数量，方法是 `misscnt = schoolrecord.isnull().sum(axis=1)`。949 个观测值的教育数据中有 7 个缺失值，10 个观测值的所有 8 个列都有缺失值。在以下代码中，我们还查看了一些具有 7 个或更多缺失值的观测值。看起来 `highestdegree` 通常是唯一一个存在的变量，这并不奇怪，因为我们已经发现 `highestdegree` 很少缺失：

    ```py
    misscnt = schoolrecord.isnull().sum(axis=1)
    misscnt.value_counts().sort_index() 
    ```

    ```py
    0	1087
    1	312
    2	3210
    3	1102
    4	176
    5	101
    6	2037
    7	949
    8	10
    dtype: int64 
    ```

    ```py
    schoolrecord.loc[misscnt>=7].head(4).T 
    ```

    ```py
    personid               403743  101705   943703   406679
    satverbal                 NaN     NaN      NaN      NaN
    satmath                   NaN     NaN      NaN      NaN
    gpaoverall                NaN     NaN      NaN      NaN
    gpaenglish                NaN     NaN      NaN      NaN
    gpamath                   NaN     NaN      NaN      NaN
    gpascience                NaN     NaN      NaN      NaN
    highestdegree          1\. GED  1\. GED  0\. None  0\. None
    highestgradecompleted     NaN     NaN      NaN      NaN 
    ```

1.  我们将删除那些在 8 个变量中有 7 个或更多缺失值的观测值。我们可以通过将 `dropna` 的 `thresh` 参数设置为 `2` 来实现。这样会删除那些非缺失值少于 2 个的观测值。删除缺失值后，我们得到了预期的观测数：`8984` - `949` - `10` = `8025`：

    ```py
    schoolrecord = schoolrecord.dropna(thresh=2)
    schoolrecord.shape 
    ```

    ```py
    (8025, 8) 
    ```

    ```py
    schoolrecord.isnull().sum(axis=1).value_counts().sort_index() 
    ```

    ```py
    0	1087
    1	312
    2	3210
    3	1102
    4	176
    5	101
    6	2037
    dtype: int64 
    ```

`gpaoverall` 存在相当多的缺失值，共计 2,980 个，虽然我们有三分之二的有效观测值 `((8984-2980)/8984)`。如果我们能够很好地填补缺失值，这个变量可能是可以保留的。相比于直接删除这些观测值，这样做可能更可取。如果我们能避免丢失这些数据，尤其是如果缺失 `gpaoverall` 的个体与其他个体在一些重要预测变量上有所不同，我们不希望失去这些数据。

1.  最直接的方法是将`gpaoverall`的总体均值分配给缺失值。以下代码使用 pandas Series 的 `fillna` 方法将所有缺失的 `gpaoverall` 值替换为 Series 的均值。`fillna` 的第一个参数是你想要填充所有缺失值的值，在本例中是 `schoolrecord.gpaoverall.mean()`。请注意，我们需要记得将 `inplace` 参数设置为 True，才能真正覆盖现有值：

    ```py
    schoolrecord = nls97[schoolrecordlist]
    schoolrecord.gpaoverall.agg(['mean','std','count']) 
    ```

    ```py
    mean      282
    std        62
    count   6,004
    Name: gpaoverall, dtype: float64 
    ```

    ```py
    schoolrecord.fillna({"gpaoverall":\
     schoolrecord.gpaoverall.mean()},
     inplace=True)
    schoolrecord.gpaoverall.isnull().sum() 
    ```

    ```py
    0 
    ```

    ```py
    schoolrecord.gpaoverall.agg(['mean','std','count']) 
    ```

    ```py
    mean      282
    std        50
    count   8,984
    Name: gpaoverall, dtype: float64 
    ```

均值当然没有改变，但标准差有了显著减少，从 62 降到了 50。这是使用数据集均值来填补所有缺失值的一个缺点。

1.  NLS 数据集中的`wageincome20`也有相当多的缺失值。以下代码显示了 3,783 个观测值缺失。我们使用`copy`方法进行深拷贝，并将`deep`设置为 True。通常我们不会这样做，但在这种情况下，我们不想改变底层 DataFrame 中`wageincome20`的值。我们这样做是因为接下来的代码块中我们会尝试使用不同的填充方法：

    ```py
    wageincome20 = nls97.wageincome20.copy(deep=True)
    wageincome20.isnull().sum() 
    ```

    ```py
    3783 
    ```

    ```py
    wageincome20.head().T 
    ```

    ```py
    personid
    135335       NaN
    999406   115,000
    151672       NaN
    750699    45,000
    781297   150,000
    Name: wageincome20, dtype: float64 
    ```

1.  与其将`wageincome`的平均值分配给缺失值，我们可以使用另一种常见的填充技术。我们可以将前一个观测值中的最近非缺失值赋给缺失值。我们可以使用 Series 对象的`ffill`方法来实现这一点（注意，首次观测值不会填充，因为没有前一个值可用）：

    ```py
    wageincome20.ffill(inplace=True)
    wageincome20.head().T 
    ```

    ```py
    personid
    135335       NaN
    999406   115,000
    151672   115,000
    750699    45,000
    781297   150,000
    Name: wageincome20, dtype: float64 
    ```

    ```py
    wageincome20.isnull().sum() 
    ```

    ```py
    1 
    ```

**注意**

如果你在 pandas 2.2.0 之前的版本中使用过`ffill`，你可能还记得以下语法：

`wageincome.fillna(method="ffill", inplace=True)`

这种语法在 pandas 2.2.0 版本中已被弃用。向后填充的语法也是如此，我们接下来将使用这种方法。

1.  我们也可以使用`bfill`方法进行向后填充。这会将缺失值填充为最近的后续值。这样会得到如下结果：

    ```py
    wageincome20 = nls97.wageincome20.copy(deep=True)
    wageincome20.head().T 
    ```

    ```py
    personid
    135335       NaN
    999406   115,000
    151672       NaN
    750699    45,000
    781297   150,000
    Name: wageincome20, dtype: float64 
    ```

    ```py
    wageincome20.std() 
    ```

    ```py
    59616.290306039584 
    ```

    ```py
    wageincome20.bfill(inplace=True)
    wageincome20.head().T 
    ```

    ```py
    personid
    135335   115,000
    999406   115,000
    151672    45,000
    750699    45,000
    781297   150,000
    Name: wageincome20, dtype: float64 
    ```

    ```py
    wageincome20.std() 
    ```

    ```py
    58199.4895818016 
    ```

如果缺失值是随机分布的，那么前向或后向填充相比使用平均值有一个优势。它更可能接近非缺失值的分布。注意，在后向填充后，标准差变化不大。

有时，根据相似观测值的平均值或中位数来填充缺失值是有意义的；例如，具有相同相关变量值的观测值。让我们在下一步中尝试这种方法。

1.  在 NLS DataFrame 中，2020 年的工作周数与获得的最高学历有相关性。以下代码显示了不同学历水平下的工作周数平均值如何变化。工作周数的平均值是 38，但没有学位的人为 28，拥有职业学位的人为 48。在这种情况下，给没有学位的人的缺失工作周数分配 28 可能比分配 38 更合适：

    ```py
    nls97.weeksworked20.mean() 
    ```

    ```py
    38.35403815808349 
    ```

    ```py
    nls97.groupby(['highestdegree'])['weeksworked20'].mean() 
    ```

    ```py
    highestdegree
    0\. None           28
    1\. GED            34
    2\. High School    37
    3\. Associates     41
    4\. Bachelors      42
    5\. Masters        45
    6\. PhD            47
    7\. Professional   48
    Name: weeksworked20, dtype: float64 
    ```

1.  以下代码为缺失`weeksworked20`的观测值分配了相同学历水平组中的工作周数平均值。我们通过使用`groupby`创建一个分组 DataFrame，`groupby(['highestdegree'])['weeksworked20']`来实现这一点。然后，我们在`transform`内使用`fillna`方法，将缺失值填充为该学历组的平均值。注意，我们确保只对学历信息不缺失的观测值进行填充，`nls97.highestdegree.notnull()`。对于同时缺失学历和工作周数的观测值，仍然会存在缺失值：

    ```py
    nls97.loc[nls97.highestdegree.notnull(), 'weeksworked20imp'] = \
      nls97.loc[nls97.highestdegree.notnull()].\
      groupby(['highestdegree'])['weeksworked20'].\
      transform(lambda x: x.fillna(x.mean()))
    nls97[['weeksworked20imp','weeksworked20','highestdegree']].\
      head(10) 
    ```

    ```py
     weeksworked20imp  weeksworked20      highestdegree
    personid                                               
    135335                  42            NaN       4\. Bachelors
    999406                  52             52     2\. High School
    151672                   0              0       4\. Bachelors
    750699                  52             52     2\. High School
    781297                  52             52     2\. High School
    613800                  52             52     2\. High School
    403743                  34            NaN             1\. GED
    474817                  51             51         5\. Masters
    530234                  52             52         5\. Masters
    351406                  52             52       4\. Bachelors 
    ```

## 它的工作原理是...

当可用数据非常少时，删除某个观测值可能是合理的。我们在*步骤 4*中已经做过了。另一种常见的方法是我们在*步骤 5*中使用的，即将该变量的整体数据集均值分配给缺失值。在这个例子中，我们看到了这种方法的一个缺点。我们可能会导致变量方差显著减小。

在*步骤 9*中，我们基于数据子集的均值为变量赋值。如果我们为变量 X[1]填充缺失值，并且 X[1]与 X[2]相关联，我们可以使用 X[1]和 X[2]之间的关系来填充 X[1]的值，这比使用数据集的均值更有意义。当 X[2]是分类变量时，这通常非常直接。在这种情况下，我们可以填充 X[1]在 X[2]的关联值下的均值。

这些填充策略——删除缺失值观测、分配数据集的均值或中位数、使用前向或后向填充，或使用相关变量的组均值——适用于许多预测分析项目。当缺失值与目标变量或依赖变量没有相关性时，这些方法效果最佳。当这种情况成立时，填充缺失值能让我们保留这些观测中的其他信息，而不会偏倚估计结果。

然而，有时情况并非如此，需要更复杂的填充策略。接下来的几个教程将探讨用于清理缺失数据的多变量技术。

## 参见

如果你对我们在*步骤 10*中使用`groupby`和`transform`的理解仍然有些不清楚，不必担心。在*第九章*，*聚合时清理杂乱数据*中，我们将更深入地使用`groupby`、`transform`和`apply`。

# 使用回归法填充缺失值

我们在上一教程的结尾处，给缺失值分配了组均值，而不是整体样本均值。正如我们所讨论的，这在决定组的变量与缺失值变量相关时非常有用。使用回归法填充值在概念上与此类似，但通常是在填充基于两个或更多变量时使用。

回归填充通过回归模型预测的相关变量值来替代变量的缺失值。这种特定的填充方法被称为确定性回归填充，因为填充值都位于回归线上，并且不会引入误差或随机性。

这种方法的一个潜在缺点是，它可能会大幅度减少缺失值变量的方差。我们可以使用随机回归填充来解决这一缺点。在本教程中，我们将探讨这两种方法。

## 准备工作

我们将在本教程中使用`statsmodels`模块来运行线性回归模型。`statsmodels`通常包含在 Python 的科学发行版中，但如果你还没有安装，可以通过`pip install statsmodels`来安装它。

## 如何做到这一点...

NLS 数据集上的`wageincome20`列存在大量缺失值。我们可以使用线性回归来填补这些值。工资收入值是 2020 年的报告收入。

1.  我们首先重新加载 NLS 数据，并检查`wageincome20`以及可能与`wageincome20`相关的列的缺失值。同时加载`statsmodels`库：

    ```py
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index("personid", inplace=True)
    nls97[['wageincome20','highestdegree','weeksworked20','parentincome']].info() 
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    Index: 8984 entries, 135335 to 713757
    Data columns (total 4 columns):
     #   Column         Non-Null Count  Dtype
    ---  ------         --------------  -----
     0   wageincome20   5201 non-null   float64
     1   highestdegree  8952 non-null   object
     2   weeksworked20  6971 non-null   float64
     3   parentincome   6588 non-null   float64
    dtypes: float64(3), object(1)
    memory usage: 350.9+ KB 
    ```

1.  我们对超过 3,000 个观测值的`wageincome20`缺失值。其他变量的缺失值较少。让我们将`highestdegree`列转换为数值，以便在回归模型中使用它：

    ```py
    nls97['hdegnum'] = nls97.highestdegree.str[0:1].astype('float')
    nls97.groupby(['highestdegree','hdegnum']).size() 
    ```

    ```py
    highestdegree    hdegnum
    0\. None          0                877
    1\. GED           1                1167
    2\. High School   2                3531
    3\. Associates    3                766
    4\. Bachelors     4                1713
    5\. Masters       5                704
    6\. PhD           6                64
    7\. Professional  7                130
    dtype: int64 
    ```

1.  正如我们已经发现的那样，我们需要将`parentincome`的逻辑缺失值替换为实际缺失值。之后，我们可以运行一些相关性分析。每个变量与`wageincome20`都有一定的正相关性，特别是`hdegnum`。

    ```py
    nls97.parentincome.replace(list(range(-5,0)), np.nan, inplace=True)
    nls97[['wageincome20','hdegnum','weeksworked20','parentincome']].corr() 
    ```

    ```py
     wageincome20  hdegnum  weeksworked20  parentincome
    wageincome20      1.00     0.38           0.22          0.27
    hdegnum           0.38     1.00           0.22          0.32
    weeksworked20     0.22     0.22           1.00          0.09
    parentincome      0.27     0.32           0.09          1.00 
    ```

1.  我们应该检查一下，具有工资收入缺失值的观测对象在某些重要方面是否与那些没有缺失值的观测对象不同。以下代码显示，这些观测对象的学位获得水平、父母收入和工作周数显著较低。在这种情况下，使用整体均值来分配值显然不是最佳选择：

    ```py
    nls97weeksworked = nls97.loc[nls97.weeksworked20>0]
    nls97weeksworked.shape 
    ```

    ```py
    (5889, 111) 
    ```

    ```py
    nls97weeksworked['missingwageincome'] = \
      np.where(nls97weeksworked.wageincome20.isnull(),1,0)
    nls97weeksworked.groupby(['missingwageincome'])[['hdegnum',
      'parentincome','weeksworked20']].\
      agg(['mean','count']) 
    ```

    ```py
     hdegnum       parentincome        weeksworked20
              mean count         mean count           mean count
    missingwageincome                                          
    0         2.81  4997    48,270.85  3731          47.97  5012
    1         2.31   875    40,436.23   611          30.70   877 
    ```

注意，我们在这里仅处理具有正值工作周数的行。对于 2020 年未工作的人来说，在 2020 年有工资收入是没有意义的。

1.  我们来试试回归插补。我们首先用平均值替换缺失的`parentincome`值。我们将`hdegnum`折叠为达到以下三种学位水平的人群：少于本科、本科及以上。我们将它们设置为哑变量，当`False`或`True`时，值为`0`或`1`。这是处理回归分析中分类数据的一种经过验证的方法。它允许我们基于组成员身份估计不同的 y 截距。

（*Scikit-learn*具有预处理功能，可以帮助我们处理这些任务。我们将在下一章节中介绍其中一些。）

```py
nls97weeksworked.parentincome. \
  fillna(nls97weeksworked.parentincome.mean(), inplace=True)
nls97weeksworked['degltcol'] = \
  np.where(nls97weeksworked.hdegnum<=2,1,0)
nls97weeksworked['degcol'] = \
  np.where(nls97weeksworked.hdegnum.between(3,4),1,0)
nls97weeksworked['degadv'] = \
  np.where(nls97weeksworked.hdegnum>4,1,0) 
```

1.  接下来，我们定义一个函数，`getlm`，用于使用`statsmodels`模块运行线性模型。该函数具有目标变量或依赖变量名称`ycolname`以及特征或自变量名称`xcolnames`的参数。大部分工作由`statsmodels`的`fit`方法完成，即`OLS(y, X).fit()`：

    ```py
    def getlm(df, ycolname, xcolnames):
      df = df[[ycolname] + xcolnames].dropna()
      y = df[ycolname]
      X = df[xcolnames]
      X = sm.add_constant(X)
      lm = sm.OLS(y, X).fit()
      coefficients = pd.DataFrame(zip(['constant'] + xcolnames,
        lm.params, lm.pvalues), columns=['features','params',
        'pvalues'])
      return coefficients, lm 
    ```

1.  现在我们可以使用 `getlm` 函数来获取参数估计和模型摘要。所有系数都是正的，并且在 95% 水平下显著，*p*-值小于 `0.05`。正如我们预期的那样，工资收入随着工作周数和父母收入的增加而增加。拥有大学学位的收入比没有大学学位的人多 $18.5K。拥有研究生学位的人比那些学历较低的人多了近 $45.6K。（`degcol` 和 `degadv` 的系数是相对于没有大学学位的人来解释的，因为这个变量被省略掉了。）

    ```py
    xvars = ['weeksworked20','parentincome','degcol','degadv']
    coefficients, lm = getlm(nls97weeksworked, 'wageincome20', xvars) 
    ```

    ```py
    coefficients
                      features     params  pvalues
    0                 constant -22,868.00     0.00
    1            weeksworked20   1,281.84     0.00
    2             parentincome       0.26     0.00
    3                   degcol  18,549.57     0.00
    4                   degadv  45,595.94     0.00 
    ```

1.  我们使用这个模型来插补缺失的工资收入值。由于我们的模型包含了常数项，因此我们需要在预测中添加一个常数。我们可以将预测结果转换为 DataFrame，然后将其与其他 NLS 数据合并。让我们也来看一些预测值，看看它们是否合理。

    ```py
    pred = lm.predict(sm.add_constant(nls97weeksworked[xvars])).\
      to_frame().rename(columns= {0: 'pred'})
    nls97weeksworked = nls97weeksworked.join(pred)
    nls97weeksworked['wageincomeimp'] = \
      np.where(nls97weeksworked.wageincome20.isnull(),\
      nls97weeksworked.pred, nls97weeksworked.wageincome20)
    nls97weeksworked[['wageincomeimp','wageincome20'] + xvars].\
      sample(10, random_state=7) 
    ```

    ```py
     wageincomeimp  wageincome20  weeksworked20  parentincome  \
    personid                                                    
    696721   380,288       380,288             52        81,300 
    928568    38,000        38,000             41        47,168 
    738731    38,000        38,000             51        17,000 
    274325    40,698           NaN              7        34,800 
    644266    63,954           NaN             52        78,000 
    438934    70,000        70,000             52        31,000 
    194288     1,500         1,500             13        39,000 
    882066    52,061           NaN             52        32,000 
    169452   110,000       110,000             52        48,600 
    284731    25,000        25,000             52        47,168 
              degcol  degadv
    personid                
    696721         1       0
    928568         0       0
    738731         1       0
    274325         0       1
    644266         0       0
    438934         1       0
    194288         0       0
    882066         0       0
    169452         1       0
    284731         0       0 
    ```

1.  我们应该查看一下我们的工资收入插补的汇总统计，并将其与实际的工资收入值进行比较。（记住，`wageincomeimp` 列包含了当 `wageincome20` 没有缺失时的实际值，其他情况下则是插补值。）`wageincomeimp` 的均值略低于 `wageincome20`，这是我们预期的结果，因为工资收入缺失的人群通常在相关变量上表现较低。但是标准差也较低。这可能是确定性回归插补的结果：

    ```py
    nls97weeksworked[['wageincomeimp','wageincome20']].\
      agg(['count','mean','std']) 
    ```

    ```py
     wageincomeimp  wageincome20
    count          5,889         5,012
    mean          59,290        63,424
    std           57,529        60,011 
    ```

1.  随机回归插补会在基于我们模型残差的预测中添加一个正态分布的误差。我们希望这个误差的均值为零，且标准差与我们的残差相同。我们可以使用 NumPy 的 `normal` 函数来实现这一点，代码为 `np.random.normal(0, lm.resid.std(), nls97.shape[0])`。其中，`lm.resid.std()` 获取模型残差的标准差。最后一个参数 `nls97.shape[0]` 指示我们需要生成多少个值；在这个例子中，我们需要为每一行数据生成一个值。

我们可以将这些值与数据合并，然后将误差 `randomadd` 加到我们的预测值中。我们设置了一个种子，以便可以重现结果：

```py
np.random.seed(0)
randomadd = np.random.normal(0, lm.resid.std(),
   nls97weeksworked.shape[0])
randomadddf = pd.DataFrame(randomadd, columns=['randomadd'],
   index=nls97weeksworked.index)
nls97weeksworked = nls97weeksworked.join(randomadddf)
nls97weeksworked['stochasticpred'] = \
   nls97weeksworked.pred + nls97weeksworked.randomadd 
```

1.  这应该会增加方差，但不会对均值产生太大影响。让我们验证一下这一点。我们首先需要用随机预测值替换缺失的工资收入值：

    ```py
    nls97weeksworked['wageincomeimpstoc'] = \
     np.where(nls97weeksworked.wageincome20.isnull(),
     nls97weeksworked.stochasticpred, nls97weeksworked.wageincome20)
    nls97weeksworked[['wageincomeimpstoc','wageincome20']].\
     agg(['count','mean','std']) 
    ```

    ```py
     wageincomeimpstoc wageincome20
    count                      5,889        5,012
    mean                      59,485       63,424
    std                       60,773       60,011 
    ```

这似乎起作用了。基于我们的随机预测插补的变量，标准差几乎与工资收入变量相同。

## 工作原理...

回归插补是一种有效的方式，可以利用我们拥有的所有数据来填补某一列的缺失值。它通常优于我们在上一篇文章中研究的插补方法，尤其是在缺失值不是随机时。然而，确定性回归插补有两个重要的局限性：它假设回归变量（我们的预测变量）与待插补变量之间存在线性关系，并且它可能会显著降低插补变量的方差，正如我们在*步骤 8 和 9*中看到的那样。

如果我们使用随机回归插补，就不会人为地减少方差。我们在*步骤 10*中就做了这个操作。这样，我们得到了更好的结果，尽管它并没有解决回归变量与插补变量之间可能存在的非线性关系问题。

在我们开始广泛使用机器学习之前，回归插补是我们常用的多变量插补方法。现在，我们可以选择使用像*k*-最近邻和随机森林等算法来执行此任务，这些方法在某些情况下比回归插补更具优势。与回归插补不同，KNN 插补不假设变量之间存在线性关系，也不假设这些变量是正态分布的。我们将在下一部分中探讨 KNN 插补。

# 使用 K 最近邻进行插补

**k-最近邻**（**KNN**）是一种流行的机器学习技术，因为它直观易懂，易于运行，并且在变量和观察值数量不大的情况下，能提供很好的结果。正因如此，它经常用于插补缺失值。正如其名字所示，KNN 识别出与每个观察值变量最相似的*k*个观察值。当用于插补缺失值时，KNN 使用最近邻来确定应该使用哪些填充值。

## 准备工作

我们将使用来自 scikit-learn 1.3.0 版本的 KNN 插补器。如果你还没有安装 scikit-learn，可以通过`pip install scikit-learn`进行安装。

## 如何操作…

我们可以使用 KNN 插补来执行与上一篇文章中回归插补相同的插补操作。

1.  我们首先从`scikit-learn`导入`KNNImputer`，并重新加载 NLS 数据：

    ```py
    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  接下来，我们准备变量。我们将学位获得情况合并为三个类别——低于大学、大学和大学以上学位——每个类别用不同的虚拟变量表示。我们还将家长收入的逻辑缺失值转换为实际的缺失值：

    ```py
    nls97['hdegnum'] = \
     nls97.highestdegree.str[0:1].astype('float')
    nls97['parentincome'] = \
     nls97.parentincome.\
       replace(list(range(-5,0)),
       np.nan) 
    ```

1.  让我们创建一个仅包含工资收入和一些相关变量的 DataFrame。我们还只选择那些有工作周数为正值的行：

    ```py
    wagedatalist = ['wageincome20','weeksworked20',
      'parentincome','hdegnum']
    wagedata = \
     nls97.loc[nls97.weeksworked20>0, wagedatalist]
    wagedata.shape 
    ```

    ```py
    (5889, 6) 
    ```

1.  现在，我们可以使用 KNN 填补器的`fit_transform`方法，为传入的 DataFrame `wagedata`中的所有缺失值生成填补值。`fit_transform`返回一个 NumPy 数组，包含了`wagedata`中所有非缺失值以及填补的值。我们将这个数组转换成一个使用`wagedata`相同索引的 DataFrame。这样在下一步中合并数据会更加方便。（对于一些有使用 scikit-learn 经验的人来说，这一步应该是熟悉的，我们将在下一章中详细讲解。）

我们需要指定用于最近邻数目的值，即*k*。我们使用一个通用的经验法则来确定*k*的值，即观察数量的平方根除以 2（sqrt(*N*)/2）。在这个例子中，*k*的值为 38。

```py
impKNN = KNNImputer(n_neighbors=38)
newvalues = impKNN.fit_transform(wagedata)
wagedatalistimp = ['wageincomeimp','weeksworked20imp',
 'parentincomeimp','hdegnumimp']
wagedataimp = pd.DataFrame(newvalues, columns=wagedatalistimp, index=wagedata.index) 
```

1.  我们将填补后的数据与原始的 NLS 工资数据进行合并，并查看一些观测值。请注意，在 KNN 填补过程中，我们不需要对相关变量的缺失值进行任何预处理填补。（在回归填补中，我们将父母收入设为数据集的均值。）

    ```py
    wagedata = wagedata.\
     join(wagedataimp[['wageincomeimp','weeksworked20imp']])
    wagedata[['wageincome20','wageincomeimp','weeksworked20',
     'weeksworked20imp']].sample(10, random_state=7) 
    ```

    ```py
     wageincome20 wageincomeimp weeksworked20 weeksworked20imp
    personid                             
    696721         380,288       380,288            52               52
    928568          38,000        38,000            41               41
    738731          38,000        38,000            51               51
    274325             NaN        11,771             7                7
    644266             NaN        59,250            52               52
    438934          70,000        70,000            52               52
    194288           1,500         1,500            13               13
    882066             NaN        61,234            52               52
    169452         110,000       110,000            52               52
    284731          25,000        25,000            52               52 
    ```

1.  让我们看看原始变量和填补变量的汇总统计数据。毫不奇怪，填补后的工资收入均值低于原始均值。正如我们在前一个菜谱中发现的，缺失工资收入的观测值通常具有较低的学历、较少的工作周数和较低的父母收入。我们还失去了一些工资收入的方差。

    ```py
    wagedata[['wageincome20','wageincomeimp']].\
     agg(['count','mean','std']) 
    ```

    ```py
     wageincome20       wageincomeimp
    count                5,012               5,889
    mean                63,424              59,467
    std                 60,011              57,218 
    ```

很简单！前面的步骤为工资收入以及其他缺失值的变量提供了合理的填补，并且我们几乎没有进行数据准备。

## 它是如何工作的...

这道菜谱的大部分工作是在*第 4 步*中完成的，我们将 DataFrame 传递给了 KNN 填补器的`fit_transform`方法。KNN 填补器返回了一个 NumPy 数组，为我们数据中的所有列填补了缺失值，包括工资收入。它基于*k*个最相似的观测值来进行填补。我们将这个 NumPy 数组转换为一个 DataFrame，并在*第 5 步*中与初始 DataFrame 合并。

KNN 在进行填补时并不假设基础数据的分布。而回归填补则假设线性回归的标准假设成立，即变量之间存在线性关系且数据服从正态分布。如果不是这种情况，KNN 可能是更好的填补方法。

我们确实需要对*k*的适当值做出初步假设，这就是所谓的超参数。模型构建者通常会进行超参数调优，以找到最佳的*k*值。KNN 的超参数调优超出了本书的范围，但我在我的书《*数据清洗与机器学习探索*》中详细讲解了这一过程。在*第 4 步*中，我们对*k*的合理假设做出了初步判断。

## 还有更多...

尽管有这些优点，KNN 插补也有其局限性。正如我们刚才讨论的，我们必须通过初步假设来调整模型，选择一个合适的*k*值，这个假设仅基于我们对数据集大小的了解。随着*k*值的增加，可能会存在过拟合的风险——即过度拟合目标变量的非缺失值数据，以至于我们对缺失值的估计不可靠。超参数调优可以帮助我们确定最佳的*k*值。

KNN 也在计算上比较昂贵，对于非常大的数据集可能不切实际。最后，当待插补的变量与预测变量之间的相关性较弱，或者这些变量之间高度相关时，KNN 插补可能表现不佳。与 KNN 插补相比，随机森林插补能够帮助我们避免 KNN 和回归插补的缺点。接下来我们将探讨随机森林插补。

## 另见

我在我的书《*数据清洗与机器学习探索*》中对 KNN 有更详细的讨论，并且有真实世界数据的示例。这些讨论将帮助您更好地理解算法的工作原理，并与其他非参数机器学习算法（如随机森林）进行对比。我们将在下一个配方中探讨随机森林用于插补值。

# 使用随机森林进行插补

随机森林是一种集成学习方法，使用自助聚合（也称为 bagging）来提高模型准确性。它通过重复计算多棵树的平均值来做出预测，从而逐步改进估计值。在这个配方中，我们将使用 MissForest 算法，它是将随机森林算法应用于缺失值插补的一种方法。

MissForest 通过填充缺失值的中位数或众数（分别适用于连续或分类变量）开始，然后使用随机森林来预测值。使用这个转换后的数据集，其中缺失值被初始预测替换，MissForest 会生成新的预测，可能会用更好的预测值替换初始预测。MissForest 通常会经历至少 4 次迭代。

## 做好准备

要运行这个配方中的代码，您需要安装`MissForest`和`MiceForest`模块。可以通过`pip`安装这两个模块。

## 如何做到……

运行 MissForest 比使用我们在前一个配方中使用的 KNN 插补器还要简单。我们将对之前处理过的工资收入数据进行插补。

1.  让我们从导入`MissForest`模块并加载 NLS 数据开始。我们导入`missforest`，并且还导入`miceforest`，我们将在后续步骤中讨论它：

    ```py
    import pandas as pd
    import numpy as np
    from missforest.missforest import MissForest
    import miceforest as mf
    nls97 = pd.read_csv("data/nls97g.csv",low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  我们应该做与前一个配方中相同的数据清洗：

    ```py
    nls97['hdegnum'] = \
     nls97.highestdegree.str[0:1].astype('float')
    nls97['parentincome'] = \
     nls97.parentincome.\
       replace(list(range(-5,0)),
       np.nan)
    wagedatalist = ['wageincome20','weeksworked20','parentincome',
     'hdegnum']
    wagedata = \
     nls97.loc[nls97.weeksworked20>0, wagedatalist] 
    ```

1.  现在我们准备运行 MissForest。请注意，这个过程与我们使用 KNN 插补器的过程非常相似：

    ```py
    imputer = MissForest()
    wagedataimp = imputer.fit_transform(wagedata)
    wagedatalistimp = \
     ['wageincomeimp','weeksworked20imp','parentincomeimp']
    wagedataimp.rename(columns=\
      {'wageincome20':'wageincome20imp',
      'weeksworked20':'weeksworked20imp',
      'parentincome':'parentincomeimp'}, inplace=True)
    wagedata = \
     wagedata.join(wagedataimp[['wageincome20imp',
    'weeksworked20imp']]) 
    ```

1.  让我们看一下我们的一些插补值和一些汇总统计信息。插补后的值具有较低的均值。考虑到我们已经知道缺失值并非随机分布，且具有较低学位和工作周数的人更有可能缺失工资收入，这一点并不令人惊讶：

    ```py
    wagedata[['wageincome20','wageincome20imp',
     'weeksworked20','weeksworked20imp']].\
     sample(10, random_state=7) 
    ```

    ```py
     wageincome20 wageincome20imp weeksworked20 weeksworked20imp
    personid                                                          
    696721       380,288         380,288            52               52
    928568        38,000          38,000            41               41
    738731        38,000          38,000            51               51
    274325           NaN           6,143             7                7
    644266           NaN          85,050            52               52
    438934        70,000          70,000            52               52
    194288         1,500           1,500            13               13
    882066           NaN          74,498            52               52
    169452       110,000         110,000            52               52
    284731        25,000          25,000            52               52 
    ```

    ```py
    wagedata[['wageincome20','wageincome20imp',
     'weeksworked20','weeksworked20imp']].\
     agg(['count','mean','std']) 
    ```

    ```py
     wageincome20 wageincome20imp weeksworked20 weeksworked20imp
    count          5,012           5,889         5,889            5,889
    mean          63,424          59,681            45               45
    std           60,011          57,424            14               14 
    ```

MissForest 使用随机森林算法生成高精度的预测。与 KNN 不同，它不需要为*k*选择初始值进行调优。它的计算成本也低于 KNN。或许最重要的是，随机森林插补对变量之间的低相关性或高度相关性不那么敏感，尽管在这个示例中这并不是问题。

## 它是如何工作的...

我们在这里基本上遵循与前一个食谱中 KNN 插补相同的过程。我们首先稍微清理数据，从最高阶的文本中提取数值变量，并将父母收入的逻辑缺失值替换为实际缺失值。

然后，我们将数据传递给`MissForest`插补器的`fit_transform`方法。该方法返回一个包含所有列插补值的数据框。

## 还有更多...

我们本可以使用链式方程多重插补（MICE），它可以通过随机森林实现，作为替代插补方法。该方法的一个优势是，MICE 为插补添加了一个随机成分，可能进一步减少了过拟合的可能性，甚至优于`missforest`。

`miceforest`的运行方式与`missforest`非常相似。

1.  我们使用在*步骤 1*中创建的`miceforest`实例创建一个`kernel`：

    ```py
    kernel = mf.ImputationKernel(
     data=wagedata[wagedatalist],
     save_all_iterations=True,
     random_state=1
    )
    kernel.mice(3,verbose=True) 
    ```

    ```py
    Initialized logger with name mice 1-3
    Dataset 0
    1 | degltcol | degcol | degadv | weeksworked20 | parentincome | wageincome20
    2 | degltcol | degcol | degadv | weeksworked20 | parentincome | wageincome20
    3 | degltcol | degcol | degadv | weeksworked20 | parentincome | wageincome20 
    ```

    ```py
    wagedataimpmice = kernel.complete_data() 
    ```

1.  然后，我们可以查看插补结果：

    ```py
    wagedataimpmice.rename(columns=\
     {'wageincome20':'wageincome20impmice',
     'weeksworked20':'weeksworked20impmice',
     'parentincome':'parentincomeimpmice'},
     inplace=True)
    wagedata = wagedata[wagedatalist].\
     join(wagedataimpmice[['wageincome20impmice',
      'weeksworked20impmice']])
    wagedata[['wageincome20','wageincome20impmice',
     'weeksworked20','weeksworked20impmice']].\
     agg(['count','mean','std']) 
    ```

    ```py
     wageincome20 wageincome20impmice weeksworked20 \
    count        5,012               5,889         5,889
    mean        63,424              59,191            45
    std         60,011              58,632            14
          weeksworked20impmice
    count                5,889
    mean                    45
    std                     14 
    ```

这产生了与`missforest`非常相似的结果。这两种方法都是缺失值插补的优秀选择。

# 使用 PandasAI 进行插补

本章中我们探讨的许多缺失值插补任务也可以通过 PandasAI 完成。正如我们在之前的章节中讨论的那样，AI 工具可以帮助我们检查使用传统工具所做的工作，并能建议我们没有想到的替代方法。然而，理解 PandasAI 或其他 AI 工具的工作原理始终是有意义的。

在这个食谱中，我们将使用 PandasAI 来识别缺失值，基于汇总统计插补缺失值，并根据机器学习算法分配缺失值。

## 准备工作

在这个食谱中，我们将使用 PandasAI。可以通过`pip install` `pandasai`进行安装。你还需要从[openai.com](https://openai.com)获取一个令牌，以便向 OpenAI API 发送请求。

## 如何操作...

在这个食谱中，我们将使用 AI 工具来完成本章中之前执行过的许多任务。

1.  我们首先导入`pandas`和`numpy`库，以及`OpenAI`和`pandasai`。在这个食谱中，我们将与 PandasAI 的`SmartDataFrame`模块进行大量的工作。我们还将加载 NLS 数据：

    ```py
    import pandas as pd
    import numpy as np
    from pandasai.llm.openai import OpenAI
    from pandasai import SmartDataframe
    llm = OpenAI(api_token="Your API Key")
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  我们对父母收入和最高学位变量进行与之前示例相同的数据清理：

    ```py
    nls97['hdegnum'] = nls97.highestdegree.str[0:1].astype('category')
    nls97['parentincome'] = \
     nls97.parentincome.\
       replace(list(range(-5,0)),
       np.nan) 
    ```

1.  我们创建了一个仅包含工资和学位数据的 DataFrame，然后从 PandasAI 中创建一个`SmartDataframe`：

    ```py
    wagedatalist = ['wageincome20','weeksworked20',
      'parentincome','hdegnum']
    wagedata = nls97[wagedatalist]
    wagedatasdf = SmartDataframe(wagedata, config={"llm": llm}) 
    ```

1.  显示所有变量的非缺失计数、平均值和标准差。我们向`SmartDataFrame`对象的`chat`方法发送一个自然语言命令来执行此操作。由于`hdegnum`（最高学位）是一个分类变量，`chat`不会显示均值或标准差：

    ```py
    wagedatasdf.chat("Show the counts, means, and standard deviations as table") 
    ```

    ```py
     count     mean      std
    wageincome20        5,201   62,888   59,616
    weeksworked20       6,971       38       21
    parentincome        6,588   46,362   42,144 
    ```

1.  我们将基于每个变量的均值填充缺失值。此时，`chat`方法将返回一个 pandas DataFrame。收入和工作周数的缺失值不再存在，但 PandasAI 识别出学位类别变量不应根据均值填充：

    ```py
    wagedatasdf = \
     wagedatasdf.chat("Impute missing values based on average.")
    wagedatasdf.chat("Show the counts, means, and standard deviations as table") 
    ```

    ```py
     count     mean      std
    wageincome20       8,984   62,888    45,358
    weeksworked20      8,984       38       18
    parentincome       8,984   46,362   36,088 
    ```

1.  我们再来看一下最高学位的值。注意到最频繁的值是`2`，你可能记得之前的内容中，`2`代表的是高中文凭。

    ```py
    wagedatasdf.hdegnum.value_counts(dropna=False).sort_index() 
    ```

    ```py
    hdegnum
    0    877
    1   1167
    2   3531
    3    766
    4   1713
    5    704
    6     64
    7    130
    NaN   32
    Name: count, dtype: int64 
    ```

1.  我们可以将学位变量的缺失值设置为其最频繁的非缺失值，这是一种常见的处理分类变量缺失值的方法。现在，所有的缺失值都被填充为`2`：

    ```py
    wagedatasdf = \
     wagedatasdf.chat("Impute missings based on most frequent value")
    wagedatasdf.hdegnum.value_counts(dropna=False).sort_index() 
    ```

    ```py
    hdegnum
    0   877
    1  1167
    2  3563
    3   766
    4  1713
    5   704
    6    64
    7   130
    Name: count, dtype: int64 
    ```

1.  我们本可以使用内置的`SmartDataframe`函数`impute_missing_values`。这个函数将使用前向填充来填补缺失值。对于最高学位变量`hdegnum`，没有填充任何值。

    ```py
    wagedatasdf = SmartDataframe(wagedata, config={"llm": llm})
    wagedatasdf = \
     wagedatasdf.impute_missing_values()
    wagedatasdf.chat("Show the counts, means, and standard deviations as table") 
    ```

    ```py
     count     mean       std
    wageincome20         8,983   62,247    59,559
    weeksworked20        8,983       39        21
    parentincome         8,982   46,096    42,632 
    ```

1.  我们可以使用 KNN 方法填充收入和工作周数的缺失值。我们从一个未更改的 DataFrame 开始。在填充后，`wageincome20`的均值比原来要低，如*步骤 4*所示。这并不奇怪，因为我们在其他示例中看到，缺失`wageincome20`的个体在与`wageincome20`相关的其他变量上也有较低的值。`wageincome20`和`parentincome`的标准差变化不大。`weeksworked20`的均值和标准差几乎没有变化，这很好。

    ```py
    wagedatasdf = SmartDataframe(wagedata, config={"llm": llm})
    wagedatasdf = wagedatasdf.chat("Impute missings for float variables based on knn with 47 neighbors")
    wagedatasdf.chat("Show the counts, means, and standard deviations as table") 
    ```

    ```py
     Counts     Means    Std Devs
    hdegnum          8952       NaN         NaN
    parentincome     8984    44,805      36,344
    wageincome20     8984    58,356      47,378
    weeksworked20    8984        38          18 
    ```

## 它是如何工作的……

每当我们将自然语言命令传递给`SmartDataframe`的`chat`方法时，Pandas 代码会被生成并执行该命令。有些代码用于生成非常熟悉的摘要统计数据。然而，它也能生成用于运行机器学习算法的代码，如 KNN 或随机森林。如前几章所述，执行`chat`后查看`pandasai.log`文件始终是个好主意，这样可以了解所生成的代码。

本示例展示了如何使用 PandasAI 来识别和填充缺失值。AI 工具，特别是大语言模型，使得通过自然语言命令生成代码变得容易，就像我们在本章早些时候创建的代码一样。

# 总结

在本章中，我们探讨了最流行的缺失值插补方法，并讨论了每种方法的优缺点。通常情况下，赋予一个整体样本均值并不是一个好方法，特别是当缺失值的观测值与其他观测值在重要方面存在差异时。我们也可以显著降低方差。前向或后向填充方法可以帮助我们保持数据的方差，但在观测值之间的接近性具有意义时，效果最佳，例如时间序列或纵向数据。在大多数非平凡的情况下，我们将需要使用多元技术，如回归、KNN 或随机森林插补。在本章中，我们已经探讨了所有这些方法，接下来的章节中，我们将学习特征编码、转换和标准化。

# 留下评价！

喜欢这本书吗？通过在亚马逊上留下评价帮助像你一样的读者。扫描下面的二维码，获取一本你选择的免费电子书。

![](img/Review_copy.png)
