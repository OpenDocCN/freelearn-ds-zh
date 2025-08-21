

# 第九章：聚合时修复凌乱的数据

本书的前几章介绍了生成整个 DataFrame 汇总统计数据的技巧。我们使用了 `describe`、`mean` 和 `quantile` 等方法来实现这一点。本章讨论了更复杂的聚合任务：按类别变量聚合以及使用聚合来改变 DataFrame 的结构。

在数据清理的初始阶段之后，分析师会花费大量时间进行 Hadley Wickham 所说的 *拆分-应用-合并*——即我们按组对数据进行子集化，对这些子集应用某些操作，然后得出对整个数据集的结论。更具体一点来说，这涉及到通过关键类别变量生成描述性统计数据。对于 `nls97` 数据集，这可能是性别、婚姻状况以及最高学历。而对于 COVID-19 数据，我们可能会按国家或日期对数据进行分段。

通常，我们需要聚合数据以为后续分析做准备。有时，DataFrame 的行被细分得比所需的分析单位更细，这时必须先进行某些聚合操作，才能开始分析。例如，我们的 DataFrame 可能包含多年来按物种每天记录的鸟类观察数据。由于这些数据波动较大，我们可能决定通过只处理每月甚至每年按物种统计的总观测量来平滑这些波动。另一个例子是家庭和汽车修理支出，我们可能需要按年度总结这些支出。

使用 NumPy 和 pandas 有多种聚合数据的方法，每种方法都有其特定的优点。本章将探讨最有用的方法：从使用 `itertuples` 进行循环，到在 NumPy 数组上进行遍历，再到使用 DataFrame 的 `groupby` 方法和透视表的多种技巧。熟悉 pandas 和 NumPy 中可用的全套工具非常有帮助，因为几乎所有的数据分析项目都需要进行某种聚合，而聚合通常是我们数据清理过程中最重要的步骤之一，选择合适的工具往往取决于数据的特征，而不是我们的个人偏好。

本章中的具体实例包括：

+   使用 `itertuples` 循环遍历数据（反模式）

+   使用 NumPy 数组按组计算汇总

+   使用 `groupby` 按组组织数据

+   使用更复杂的聚合函数与 `groupby`

+   使用用户定义的函数和 `groupby` 中的 apply

+   使用 `groupby` 改变 DataFrame 的分析单位

+   使用 pandas 的 `pivot_table` 函数改变分析单位

# 技术要求

本章的实例需要 pandas、NumPy 和 Matplotlib。我使用的是 pandas 2.1.4，但代码同样适用于 pandas 1.5.3 或更高版本。

本章的代码可以从本书的 GitHub 仓库下载，[`github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition`](https://github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition)。

# 使用`itertuples`循环遍历数据（反模式）

在本食谱中，我们将遍历数据框的每一行，并为一个变量生成自己的总计。在本章后续的食谱中，我们将使用 NumPy 数组，然后是一些 pandas 特定技术，来完成相同的任务。

开始这一章时使用一个我们通常被警告不要使用的技术，可能看起来有些奇怪。但在 35 年前，我曾在 SAS 中做过类似的日常循环操作，甚至在 10 年前的 R 中偶尔也会使用。因此，即使我很少以这种方式实现代码，我仍然会从概念上考虑如何遍历数据行，有时会按组排序。我认为即使在使用其他对我们更有效的 pandas 方法时，保持这种概念化的思维是有益的。

我不想给人留下 pandas 特定技术总是明显更高效的印象。pandas 用户可能会发现自己比预期更多地使用`apply`，这种方法比循环稍微快一点。

## 准备工作

在本食谱中，我们将使用 COVID-19 每日病例数据。每行代表一天，每个国家一行，包含当天的新病例数和新死亡人数。它反映了截至 2024 年 3 月的总数。

我们还将使用来自巴西 87 个气象站 2023 年的陆地温度数据。大多数气象站每个月有一个温度读数。

**数据说明**

我们的数据来源于[Our World in Data](https://ourworldindata.org/covid-cases)，提供 COVID-19 的公共数据。该数据集包括总病例数和死亡人数、施行的检测次数、医院床位，以及人口统计数据，如中位年龄、国内生产总值和糖尿病患病率。此食谱中使用的数据集是在 2024 年 3 月 3 日下载的。

陆地温度数据框包含了 2023 年来自全球超过 12,000 个站点的平均温度（以^°C 为单位），尽管大多数站点位于美国。原始数据是从全球历史气候网整合数据库中提取的。美国国家海洋和大气管理局将其公开提供，网址为[`www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly`](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly)。

## 如何操作…

我们将使用`itertuples`数据框方法来遍历 COVID-19 每日数据和巴西的月度陆地温度数据。我们将添加逻辑来处理缺失数据和关键变量值在不同时间段之间的意外变化：

1.  导入 `pandas` 和 `numpy`，并加载 COVID-19 和陆地温度数据：

    ```py
    import pandas as pd
    coviddaily = pd.read_csv("data/coviddaily.csv", parse_dates=["casedate"])
    ltbrazil = pd.read_csv("data/ltbrazil.csv") 
    ```

1.  按位置和日期对数据进行排序：

    ```py
    coviddaily = coviddaily.sort_values(['location','casedate']) 
    ```

1.  使用 `itertuples` 遍历行。

使用 `itertuples`，它允许我们将所有行作为命名元组进行遍历。对每个国家的所有日期求新增病例的总和。每当国家（`location`）发生变化时，将当前的累计值附加到 `rowlist` 中，然后将计数重置为 `0`（请注意，`rowlist` 是一个列表，每次国家发生变化时，我们都会向 `rowlist` 中添加一个字典。字典列表是暂时存储数据的一个好地方，数据最终可以转为 DataFrame）。

```py
prevloc = 'ZZZ'
rowlist = []
casecnt = 0
for row in coviddaily.itertuples():
...   if (prevloc!=row.location):
...     if (prevloc!='ZZZ'):
...       rowlist.append({'location':prevloc, 'casecnt':casecnt})
...     casecnt = 0
...     prevloc = row.location
...   casecnt += row.new_cases
...
rowlist.append({'location':prevloc, 'casecnt':casecnt})
len(rowlist) 
```

```py
231 
```

```py
rowlist[0:4] 
```

```py
[{'location': 'Afghanistan', 'casecnt': 231539.0},
 {'location': 'Albania', 'casecnt': 334863.0},
 {'location': 'Algeria', 'casecnt': 272010.0},
 {'location': 'American Samoa', 'casecnt': 8359.0}] 
```

1.  从汇总值列表 `rowlist` 创建一个 DataFrame。

将我们在上一步创建的列表传递给 pandas 的 `DataFrame` 方法：

```py
covidtotals = pd.DataFrame(rowlist)
covidtotals.head() 
```

```py
 location      casecnt
0         Afghanistan      231,539
1             Albania      334,863
2             Algeria      272,010
3      American Samoa        8,359
4             Andorra       48,015 
```

1.  现在，我们对陆地温度数据做同样的处理。我们首先按 `station` 和 `month` 排序。

同时，删除温度缺失的行：

```py
ltbrazil = ltbrazil.sort_values(['station','month'])
ltbrazil = ltbrazil.dropna(subset=['temperature']) 
```

1.  排除每一周期之间变化较大的行。

计算年度平均温度，排除比上个月的温度高出或低于 3°C 的值：

```py
prevstation = 'ZZZ'
prevtemp = 0
rowlist = []
tempcnt = 0
stationcnt = 0
for row in ltbrazil.itertuples():
...   if (prevstation!=row.station):
...     if (prevstation!='ZZZ'):
...       rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt, 'stationcnt':stationcnt})
...     tempcnt = 0
...     stationcnt = 0
...     prevstation = row.station
...   # choose only rows that are within 3 degrees of the previous temperature 
...   if ((0 <= abs(row.temperature-prevtemp) <= 3) or (stationcnt==0)):
...     tempcnt += row.temperature
...     stationcnt += 1
...   prevtemp = row.temperature
...
rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt, 'stationcnt':stationcnt})
rowlist[0:5] 
```

```py
[{'station': 'ALTAMIRA', 'avgtemp': 27.729166666666668, 'stationcnt': 12},
 {'station': 'ALTA_FLORESTA_AERO',
  'avgtemp': 32.49333333333333,
  'stationcnt': 9},
 {'station': 'ARAXA', 'avgtemp': 21.52142857142857, 'stationcnt': 7},
 {'station': 'BACABAL', 'avgtemp': 28.59166666666667, 'stationcnt': 6},
 {'station': 'BAGE', 'avgtemp': 19.615000000000002, 'stationcnt': 10}] 
```

1.  根据汇总值创建一个 DataFrame。

将我们在上一步创建的列表传递给 pandas 的 `DataFrame` 方法：

```py
ltbrazilavgs = pd.DataFrame(rowlist)
ltbrazilavgs.head() 
```

```py
 station      avgtemp      stationcnt
0                ALTAMIRA           28              12
1      ALTA_FLORESTA_AERO           32               9
2                   ARAXA           22               7
3                 BACABAL           29               6
4                    BAGE           20              10 
```

这将为我们提供一个包含 2023 年平均温度和每个站点观测次数的 DataFrame。

## 它是如何工作的...

在 *第 2 步* 中通过 `location` 和 `casedate` 对 COVID-19 每日数据进行排序后，我们逐行遍历数据，并在 *第 3 步* 中对新增病例进行累计。每当遇到一个新国家时，我们将累计值重置为 `0`，然后继续计数。请注意，我们实际上并不会在遇到下一个国家之前就附加新增病例的总结。这是因为在我们遇到下一个国家之前，无法判断当前行是否是某个国家的最后一行。这不是问题，因为我们会在将累计值重置为 `0` 之前将总结附加到 `rowlist` 中。这也意味着我们需要采取特别的措施来输出最后一个国家的总数，因为没有下一个国家。我们通过在循环结束后执行最后一次附加操作来做到这一点。这是一种相当标准的数据遍历和按组输出总数的方法。

我们在 *第 3 步* 和 *第 4 步* 中创建的汇总 `DataFrame` 可以通过本章中介绍的其他 pandas 技巧更高效地创建，无论是在分析师的时间上，还是在计算机的工作负载上。但当我们需要进行更复杂的计算时，特别是那些涉及跨行比较值的计算，这个决策就变得更加困难。

*第 6 步* 和 *第 7 步* 提供了这个示例。我们想要计算每个站点一年的平均温度。大多数站点每月有一次读数。然而，我们担心可能存在一些异常值，这些异常值是指一个月与下个月之间温度变化超过 3°C。我们希望将这些读数排除在每个站点的均值计算之外。在遍历数据时，通过存储上一个温度值（`prevtemp`）并将其与当前值进行比较，可以相对简单地做到这一点。

## 还有更多...

我们本可以在*第 3 步*中使用`iterrows`，而不是`itertuples`，语法几乎完全相同。由于这里不需要`iterrows`的功能，我们使用了`itertuples`。与`iterrows`相比，`itertuples`方法对系统资源的消耗较少。因为使用`itertuples`时，你是遍历元组，而使用`iterrows`时是遍历 Series，并且涉及到类型检查。

处理表格数据时，最难完成的任务是跨行计算：在行之间求和、基于不同一行的值进行计算以及生成累计总和。无论使用何种语言，这些计算都很复杂且资源密集。然而，特别是在处理面板数据时，很难避免这些任务。某些变量在特定时期的值可能由前一时期的值决定。这通常比我们在本段中所做的累积总和更加复杂。

数十年来，数据分析师们一直试图通过遍历行、仔细检查分类和汇总变量中的数据问题，然后根据情况处理求和来解决这些数据清理挑战。尽管这种方法提供了最大的灵活性，但 pandas 提供了许多数据聚合工具，这些工具运行更高效，编码也更简单。挑战在于如何匹配循环解决方案的能力，以应对无效、不完整或不典型的数据。我们将在本章后面探讨这些工具。

# 使用 NumPy 数组按组计算汇总

我们可以使用 NumPy 数组完成在上一段中所做的大部分工作。我们还可以使用 NumPy 数组来获取数据子集的汇总值。

## 做好准备

我们将再次使用 COVID-19 每日数据和巴西土地温度数据。

## 如何做……

我们将 DataFrame 的值复制到 NumPy 数组中。然后，我们遍历该数组，按组计算总和并检查值的意外变化：

1.  导入`pandas`和`numpy`，并加载 COVID-19 和土地温度数据：

    ```py
    import pandas as pd
    coviddaily = pd.read_csv("data/coviddaily.csv", parse_dates=["casedate"])
    ltbrazil = pd.read_csv("data/ltbrazil.csv") 
    ```

1.  创建一个位置列表：

    ```py
    loclist = coviddaily.location.unique().tolist() 
    ```

1.  使用 NumPy 数组按位置计算总和。

创建一个包含位置和新增病例数据的 NumPy 数组。接下来，我们可以遍历在上一步骤中创建的位置列表，并为每个位置选择所有新增病例值（`casevalues[j][1]`）（根据位置（`casevalues[j][0]`））。然后，我们为该位置求和新增病例值：

```py
rowlist = []
casevalues = coviddaily[['location','new_cases']].to_numpy()
for locitem in loclist:
...   cases = [casevalues[j][1] for j in range(len(casevalues))\
...     if casevalues[j][0]==locitem]
...   rowlist.append(sum(cases))
...
len(rowlist) 
```

```py
231 
```

```py
len(loclist) 
```

```py
231 
```

```py
rowlist[0:5] 
```

```py
[231539.0, 334863.0, 272010.0, 8359.0, 48015.0] 
```

```py
casetotals = pd.DataFrame(zip(loclist,rowlist), columns=(['location','casetotals']))
casetotals.head() 
```

```py
 location      casetotals
0         Afghanistan         231,539
1             Albania         334,863
2             Algeria         272,010
3      American Samoa           8,359
4             Andorra          48,015 
```

1.  对陆地温度数据进行排序，并删除温度缺失值的行：

    ```py
    ltbrazil = ltbrazil.sort_values(['station','month'])
    ltbrazil = ltbrazil.dropna(subset=['temperature']) 
    ```

1.  使用 NumPy 数组来计算年度平均温度。

排除两个时间段之间变化较大的行：

```py
prevstation = 'ZZZ'
prevtemp = 0
rowlist = []
tempvalues = ltbrazil[['station','temperature']].to_numpy()
tempcnt = 0
stationcnt = 0
for j in range(len(tempvalues)):
...   station = tempvalues[j][0]
...   temperature = tempvalues[j][1]
...   if (prevstation!=station):
...     if (prevstation!='ZZZ'):
...       rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt, 'stationcnt':stationcnt})
...     tempcnt = 0
...     stationcnt = 0
...     prevstation = station
...   if ((0 <= abs(temperature-prevtemp) <= 3) or (stationcnt==0)):
...     tempcnt += temperature
...     stationcnt += 1
...   prevtemp = temperature
...
rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt, 'stationcnt':stationcnt})
rowlist[0:5] 
```

```py
[{'station': 'ALTAMIRA', 'avgtemp': 27.729166666666668, 'stationcnt': 12},
 {'station': 'ALTA_FLORESTA_AERO',
  'avgtemp': 32.49333333333333,
  'stationcnt': 9},
 {'station': 'ARAXA', 'avgtemp': 21.52142857142857, 'stationcnt': 7},
 {'station': 'BACABAL', 'avgtemp': 28.59166666666667, 'stationcnt': 6},
 {'station': 'BAGE', 'avgtemp': 19.615000000000002, 'stationcnt': 10}] 
```

1.  创建一个包含陆地温度平均值的 DataFrame：

    ```py
    ltbrazilavgs = pd.DataFrame(rowlist)
    ltbrazilavgs.head() 
    ```

    ```py
     station      avgtemp      stationcnt
    0                ALTAMIRA           28              12
    1      ALTA_FLORESTA_AERO           32               9
    2                   ARAXA           22               7
    3                 BACABAL           29               6
    4                    BAGE           20              10 
    ```

这将给我们一个 DataFrame，其中包含每个站点的平均温度和观测次数。请注意，我们得到的结果与前一个示例的最后一步相同。

## 工作原理…

当我们处理表格数据，但需要在行间进行计算时，NumPy 数组非常有用。这是因为访问数组中的“行”的方式与访问“列”的方式没有太大区别。例如，`casevalues[5][0]`（数组的第六“行”和第一“列”）与 `casevalues[20][1]` 的访问方式是相同的。遍历 NumPy 数组也比遍历 pandas DataFrame 更快。

我们在*第 3 步*中利用了这一点。我们通过列表推导式获取给定位置的所有数组行（`if casevalues[j][0]==locitem`）。由于我们还需要在将要创建的汇总值 DataFrame 中包含 `location` 列表，我们使用 `zip` 来组合这两个列表。

我们在*第 4 步*开始处理陆地温度数据，首先按 `station` 和 `month` 排序，然后删除温度缺失值的行。*第 5 步*中的逻辑与前一个示例中的*第 6 步*几乎相同。主要的区别是，我们需要引用数组中站点（`tempvalues[j][0]`）和温度（`tempvalues[j][1]`）的位置。

## 还有更多…

当你需要遍历数据时，NumPy 数组通常比通过 `itertuples` 或 `iterrows` 遍历 pandas DataFrame 更快。此外，如果你尝试使用 `itertuples` 来运行*第 3 步*中的列表推导式，虽然是可行的，但你将需要等待较长时间才能完成。通常，如果你想对某一数据段做快速汇总，使用 NumPy 数组是一个合理的选择。

## 另见

本章剩余的示例依赖于 pandas DataFrame 中强大的 `groupby` 方法来生成分组总数。

# 使用 groupby 按组组织数据

在大多数数据分析项目中，我们必须按组生成汇总统计信息。虽然可以使用前一个示例中的方法完成这项任务，但在大多数情况下，pandas DataFrame 的 `groupby` 方法是一个更好的选择。如果 `groupby` 能够处理聚合任务——而且通常可以——那么它很可能是完成该任务的最有效方式。我们将在接下来的几个示例中充分利用 `groupby`。我们将在本示例中介绍基础知识。

## 准备工作

我们将在本食谱中处理 COVID-19 每日数据。

## 如何做到…

我们将创建一个 pandas 的`groupby` DataFrame，并使用它生成按组的汇总统计：

1.  导入`pandas`和`numpy`，并加载 COVID-19 每日数据：

    ```py
    import pandas as pd
    coviddaily = pd.read_csv("data/coviddaily.csv", parse_dates=["casedate"]) 
    ```

1.  创建一个 pandas 的`groupby` DataFrame：

    ```py
    countrytots = coviddaily.groupby(['location'])
    type(countrytots) 
    ```

    ```py
    <class 'pandas.core.groupby.generic.DataFrameGroupBy'> 
    ```

1.  为每个国家创建第一次出现的行的 DataFrame。

为了节省空间，我们只显示前五行和前五列：

```py
countrytots.first().iloc[0:5, 0:5] 
```

```py
 iso_code     casedate     continent      new_cases  \
location                                                 
Afghanistan        AFG   2020-03-01          Asia              1  
Albania            ALB   2020-03-15        Europe             33  
Algeria            DZA   2020-03-01        Africa              1  
American Samoa     ASM   2021-09-19       Oceania              1  
Andorra            AND   2020-03-08        Europe              1  
                    new_deaths 
location                   
Afghanistan                  0 
Albania                      1 
Algeria                      0 
American Samoa               0 
Andorra                      0 
```

1.  为每个国家创建最后几行的 DataFrame：

    ```py
    countrytots.last().iloc[0:5, 0:5] 
    ```

    ```py
     iso_code       casedate     continent      new_cases  \
    location                                                 
    Afghanistan         AFG     2024-02-04          Asia            210  
    Albania             ALB     2024-01-28        Europe             45  
    Algeria             DZA     2023-12-03        Africa             19  
    American Samoa      ASM     2023-09-17       Oceania             18  
    Andorra             AND     2023-05-07        Europe             41  
                        new_deaths 
    location                   
    Afghanistan                  0 
    Albania                      0 
    Algeria                      0 
    American Samoa               0 
    Andorra                      0 
    ```

    ```py
    type(countrytots.last()) 
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'> 
    ```

1.  获取某个国家的所有行：

    ```py
    countrytots.get_group(('Zimbabwe')).iloc[0:5, 0:5] 
    ```

    ```py
     iso_code       casedate      location     continent      new_cases
    36305 ZWE     2020-03-22      Zimbabwe        Africa              2
    36306 ZWE     2020-03-29      Zimbabwe        Africa              5
    36307 ZWE     2020-04-05      Zimbabwe        Africa              2
    36308 ZWE     2020-04-12      Zimbabwe        Africa              7
    36309 ZWE     2020-04-19      Zimbabwe        Africa             10 
    ```

1.  遍历各组。

仅显示马耳他和科威特的行：

```py
for name, group in countrytots:
...   if (name[0] in ['Malta','Kuwait']):
...     print(group.iloc[0:5, 0:5])
... 
```

```py
 iso_code       casedate     location     continent      new_cases
17818   KWT     2020-03-01       Kuwait          Asia             45
17819   KWT     2020-03-08       Kuwait          Asia             16
17820   KWT     2020-03-15       Kuwait          Asia             43
17821   KWT     2020-03-22       Kuwait          Asia             72
17822   KWT     2020-03-29       Kuwait          Asia             59
   iso_code       casedate     location     continent      new_cases
20621   MLT     2020-03-08        Malta        Europe              3
20622   MLT     2020-03-15        Malta        Europe             28
20623   MLT     2020-03-22        Malta        Europe             78
20624   MLT     2020-03-29        Malta        Europe             50
20625   MLT     2020-04-05        Malta        Europe             79 
```

1.  显示每个国家的行数：

    ```py
    countrytots.size() 
    ```

    ```py
    location
    Afghanistan              205
    Albania                  175
    Algeria                  189
    American Samoa            58
    Andorra                  158
    Vietnam                  192
    Wallis and Futuna         23
    Yemen                    122
    Zambia                   173
    Zimbabwe                 196
    Length: 231, dtype: int64 
    ```

1.  按国家显示汇总统计：

    ```py
    countrytots.new_cases.describe().head(3).T 
    ```

    ```py
    location      Afghanistan      Albania      Algeria
    count                 205          175          189
    mean                1,129        1,914        1,439
    std                 1,957        2,637        2,205
    min                     1           20            1
    25%                   242          113           30
    50%                   432          522          723
    75%                 1,106        3,280        1,754
    max                12,314       15,405       14,774 
    ```

    ```py
    countrytots.new_cases.sum().head() 
    ```

    ```py
    location
    Afghanistan          231,539
    Albania              334,863
    Algeria              272,010
    American Samoa         8,359
    Andorra               48,015
    Name: new_cases, dtype: float64 
    ```

这些步骤展示了当我们希望按分类变量生成汇总统计时，`groupby` DataFrame 对象是多么有用。

## 它是如何工作的…

在*步骤 2*中，我们使用`pandas`的`groupby`方法创建一个`groupby`对象，传入一个列或多个列进行分组。一旦我们拥有了一个`groupby`的 DataFrame，我们可以使用与整个 DataFrame 生成汇总统计相同的工具来按组生成统计数据。`describe`、`mean`、`sum`等方法可以在`groupby`的 DataFrame 或由其创建的系列上按预期工作，区别在于汇总统计会针对每个组执行。

在*步骤 3 和 4*中，我们使用`first`和`last`来创建包含每个组的第一次和最后一次出现的 DataFrame。在*步骤 5*中，我们使用`get_group`来获取某个特定组的所有行。我们还可以遍历各组，并使用`size`来统计每个组的行数。

在*步骤 8*中，我们从 DataFrame 的`groupby`对象创建一个 Series 的`groupby`对象。使用结果对象的聚合方法，我们可以按组生成 Series 的汇总统计。从这个输出可以清楚地看到，`new_cases`的分布因国家而异。例如，我们可以立刻看到，即使是前三个国家，它们的四分位数间距也差异很大。

## 还有更多…

从*步骤 8*得到的输出非常有用。保存每个重要连续变量的输出是值得的，尤其是当按组的分布有显著不同的时候。

pandas 的`groupby` DataFrame 非常强大且易于使用。*步骤 8*展示了创建我们在本章前两篇食谱中按组生成的汇总统计有多么简单。除非我们处理的 DataFrame 很小，或者任务涉及非常复杂的跨行计算，否则`groupby`方法是优于循环的选择。

# 使用更复杂的聚合函数与`groupby`

在前一个示例中，我们创建了一个 `groupby` DataFrame 对象，并使用它来按组运行汇总统计数据。在这个示例中，我们通过链式操作一行代码创建分组、选择聚合变量和选择聚合函数。我们还利用了 `groupby` 对象的灵活性，允许我们以多种方式选择聚合列和函数。

## 准备工作

本示例将使用 **国家青年纵向调查**（**National Longitudinal Survey of Youth**，简称 **NLS**）数据。

**数据说明**

**国家纵向调查**（**National Longitudinal Surveys**），由美国劳工统计局管理，是针对 1997 年高中毕业生开展的纵向调查。参与者每年接受一次调查，直到 2023 年。这些调查数据可通过 [nlsinfo.org](https://nlsinfo.org) 公开访问。

## 如何操作…

我们在这个示例中使用 `groupby` 做了比之前示例更复杂的聚合操作，利用了其灵活性：

1.  导入 `pandas` 并加载 NLS 数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  查看数据的结构：

    ```py
    nls97.iloc[:,0:7].info() 
    ```

    ```py
    <class 'pandas.core.frame.DataFrame'>
    Index: 8984 entries, 135335 to 713757
    Data columns (total 7 columns):
     #   Column                 Non-Null Count  Dtype 
    ---  ------                 --------------  ----- 
     0   gender                 8984 non-null   object
     1   birthmonth             8984 non-null   int64 
     2   birthyear              8984 non-null   int64 
     3   sampletype             8984 non-null   object
     4   ethnicity              8984 non-null   object
     5   highestgradecompleted  6663 non-null   float64
     6   maritalstatus          6675 non-null   object
    dtypes: float64(1), int64(2), object(4)
    memory usage: 561.5+ KB 
    ```

1.  查看一些类别数据：

    ```py
    catvars = ['gender','maritalstatus','highestdegree']
    for col in catvars:
    ...   print(col, nls97[col].value_counts().\
    ...     sort_index(), sep="\n\n", end="\n\n\n")
    ... 
    ```

    ```py
    gender
    Female    4385
    Male      4599
    Name: count, dtype: int64
    maritalstatus
    Divorced          669
    Married          3068
    Never-married    2767
    Separated         148
    Widowed            23
    Name: count, dtype: int64
    highestdegree
    0\. None             877
    1\. GED             1167
    2\. High School     3531
    3\. Associates       766
    4\. Bachelors       1713
    5\. Masters          704
    6\. PhD               64
    7\. Professional     130
    Name: count, dtype: int64 
    ```

1.  查看一些描述性统计信息：

    ```py
    contvars = ['satmath','satverbal',
    ...   'weeksworked06','gpaoverall','childathome']
    nls97[contvars].describe() 
    ```

    ```py
     satmath   satverbal   weeksworked06   gpaoverall    childathome
    count     1,407       1,406           8,419        6,004          4,791
    mean        501         500              38          282              2
    std         115         112              19           62              1
    min           7          14               0           10              0
    25%         430         430              27          243              1
    50%         500         500              51          286              2
    75%         580         570              52          326              3
    max         800         800              52          417              9 
    ```

1.  按性别查看 **学术能力评估测试**（**SAT**）数学成绩。

我们将列名传递给 `groupby`，根据该列进行分组：

```py
nls97.groupby('gender')['satmath'].mean() 
```

```py
gender
Female	487
Male	517
Name: satmath, dtype: float64 
```

1.  按性别和最高学历查看 SAT 数学成绩。

我们可以将列名列表传递给 `groupby`，以便按多个列进行分组：

```py
nls97.groupby(['gender','highestdegree'])['satmath'].\
  mean() 
```

```py
gender  highestdegree 
Female  0\. None           414
        1\. GED            405
        2\. High School    426
        3\. Associates     448
        4\. Bachelors      503
        5\. Masters        504
        6\. PhD            569
        7\. Professional   593
Male    0\. None           545
        1\. GED            320
        2\. High School    465
        3\. Associates     490
        4\. Bachelors      536
        5\. Masters        568
        6\. PhD            624
        7\. Professional   594
Name: satmath, dtype: float64 
```

1.  按性别和最高学历查看 SAT 数学和语言成绩。

我们可以使用列表来汇总多个变量的值，在这种情况下是 `satmath` 和 `satverbal`：

```py
nls97.groupby(['gender','highestdegree'])[['satmath','satverbal']].mean() 
```

```py
 satmath  satverbal
gender highestdegree                     
Female 0\. None              414        408
       1\. GED               405        390
       2\. High School       426        440
       3\. Associates        448        453
       4\. Bachelors         503        508
       5\. Masters           504        529
       6\. PhD               569        561
       7\. Professional      593        584
Male   0\. None              545        515
       1\. GED               320        360
       2\. High School       465        455
       3\. Associates        490        469
       4\. Bachelors         536        521
       5\. Masters           568        540
       6\. PhD               624        627
       7\. Professional      594        599 
```

1.  对一个变量做多个聚合函数。

使用 `agg` 函数返回多个汇总统计数据：

```py
nls97.groupby(['gender','highestdegree'])\
  ['gpaoverall'].agg(['count','mean','max','std']) 
```

```py
 count    mean    max    std
gender highestdegree                        
Female 0\. None              134     243    400     66
       1\. GED               231     230    391     66
       2\. High School      1152     277    402     53
       3\. Associates        294     291    400     50
       4\. Bachelors         742     322    407     48
       5\. Masters           364     329    417     43
       6\. PhD                26     345    400     44
       7\. Professional       55     353    411     41
Male   0\. None              180     222    400     65
       1\. GED               346     223    380     63
       2\. High School      1391     263    396     49
       3\. Associates        243     272    383     49
       4\. Bachelors         575     309    405     49
       5\. Masters           199     324    404     50
       6\. PhD                23     342    401     55
       7\. Professional       41     345    410     35 
```

1.  使用字典进行更复杂的聚合：

    ```py
    pd.options.display.float_format = '{:,.1f}'.format
    aggdict = {'weeksworked06':['count', 'mean',
    ...  'max','std'], 'childathome':['count', 'mean',
    ...  'max', 'std']}
    nls97.groupby(['highestdegree']).agg(aggdict) 
    ```

    ```py
     weeksworked06                 \
                            count  mean   max    std  
    highestdegree                                 
    0\. None                   666  29.7   52.0   21.6  
    1\. GED                   1129  32.9   52.0   20.7  
    2\. High School           3262  39.4   52.0   18.6  
    3\. Associates             755  40.2   52.0   18.0  
    4\. Bachelors             1683  42.3   52.0   16.2  
    5\. Masters                703  41.8   52.0   16.6   
    6\. PhD                     63  38.5   52.0   18.4  
    7\. Professional           127  27.8   52.0   20.4  
                    childathome              
                          count   mean   max   std 
    highestdegree                            
    0\. None                 408    1.8   8.0   1.6 
    1\. GED                  702    1.7   9.0   1.5 
    2\. High School         1881    1.9   7.0   1.3 
    3\. Associates           448    1.9   6.0   1.1 
    4\. Bachelors            859    1.9   8.0   1.1 
    5\. Masters              379    1.9   6.0   0.9 
    6\. PhD                   33    1.9   3.0   0.8 
    7\. Professional          60    1.8   4.0   0.8 
    ```

    ```py
    nls97.groupby(['maritalstatus']).agg(aggdict) 
    ```

    ```py
     weeksworked06                 \
                          count   mean    max    std  
    maritalstatus                               
    Divorced                666   37.5   52.0   19.0  
    Married                3035   40.3   52.0   17.9  
    Never-married          2735   37.2   52.0   19.1  
    Separated               147   33.6   52.0   20.3  
    Widowed                  23   37.1   52.0   19.3  
                  childathome              
                          count   mean    max    std 
    maritalstatus                          
    Divorced                530   1.5     5.0    1.2 
    Married                2565   2.1     8.0    1.1 
    Never-married          1501   1.6     9.0    1.3 
    Separated               132   1.5     8.0    1.4 
    Widowed                  18   1.8     5.0    1.4 
    ```

我们为 `weeksworked06` 和 `childathome` 显示了相同的汇总统计数据，但我们也可以为每个变量指定不同的聚合函数，使用与 *步骤 9* 中相同的语法。

## 如何操作…

我们首先查看 DataFrame 中关键列的汇总统计信息。在 *步骤 3* 中，我们获得了类别变量的频率，在 *步骤 4* 中，我们得到了连续变量的一些描述性统计信息。生成按组统计数据之前，先查看整个 DataFrame 的汇总值是个不错的主意。

接下来，我们准备使用 `groupby` 创建汇总统计数据。这涉及三个步骤：

1.  根据一个或多个类别变量创建 `groupby` DataFrame。

1.  选择用于汇总统计数据的列。

1.  选择聚合函数。

在这个示例中，我们使用了链式操作，一行代码完成了三件事。因此，`nls97.groupby('gender')['satmath'].mean()` 在*步骤 5*中做了三件事情：`nls97.groupby('gender')` 创建了一个 `groupby` DataFrame 对象，`['satmath']` 选择了聚合列，`mean()` 是聚合函数。

我们可以像在*步骤 5*中那样传递列名，或者像在*步骤 6*中那样传递列名列表，来通过一个或多个列进行分组。我们可以使用一个变量列表来选择多个变量进行聚合，正如在*步骤 7*中使用`[['satmath','satverbal']]`一样。

我们可以链式调用特定的汇总函数，例如`mean`、`count`或`max`。另外，我们也可以将一个列表传递给`agg`，选择多个聚合函数，像在*步骤 8*中使用`agg(['count','mean','max','std'])`。我们可以使用熟悉的 pandas 和 NumPy 聚合函数，或者使用用户定义的函数，后者我们将在下一个例子中探讨。

从*步骤 8*中可以得出的另一个重要结论是，`agg`将每个聚合列一次只发送给一个函数。每个聚合函数中的计算会对`groupby` DataFrame 中的每个组执行。另一种理解方式是，它允许我们一次对一个组执行通常在整个 DataFrame 上执行的相同函数，自动化地将每个组的数据传递给聚合函数。

## 更多内容…

我们首先了解 DataFrame 中类别变量和连续变量的分布情况。通常，我们会通过分组数据，查看连续变量（例如工作周数）如何因类别变量（例如婚姻状况）而有所不同。在此之前，了解这些变量在整个数据集中的分布情况非常有帮助。

`nls97`数据集仅对约 1,400 个受访者中的 8,984 人提供 SAT 分数，因此在根据不同群体查看 SAT 分数时需要小心。这意味着按性别和最高学位（特别是博士学位获得者）统计的某些计数值可能太小，无法可靠。在 SAT 数学和语言类分数上有异常值（如果我们定义异常值为高于第三四分位数或低于第一四分位数的 1.5 倍四分位距）。

对于所有的最高学位和婚姻状况（除了丧偶）值，我们都有可接受的工作周数和居住在家中的孩子数的计数值。获得专业学位的人的平均工作周数出乎意料，它低于任何其他群体。接下来的好步骤是查看这种现象在多年中的持续性。（我们这里只看的是 2006 年的工作周数数据，但有 20 年的工作周数数据可用。）

## 另请参见

`nls97`文件是伪装成个体级数据的面板数据。我们可以恢复面板数据结构，从而促进对就业和学校注册等领域的时间序列分析。我们在*第十一章：数据整理与重塑*的例子中会进行相关操作。

# 使用用户定义函数和 apply 与 groupby

尽管 pandas 和 NumPy 提供了众多聚合函数，但有时我们需要编写自己的函数来获得所需的结果。在某些情况下，这需要使用`apply`。

## 准备工作

本例中我们将使用 NLS 数据。

## 如何操作…

我们将创建自己的函数，定义我们按组需要的汇总统计量：

1.  导入 `pandas` 和 NLS 数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index("personid", inplace=True) 
    ```

1.  创建一个函数来定义四分位数范围：

    ```py
    def iqr(x):
    ...   return x.quantile(0.75) - x.quantile(0.25) 
    ```

1.  运行四分位数范围函数。

创建一个字典，指定每个分析变量运行的聚合函数：

```py
aggdict = {'weeksworked06':['count', 'mean', iqr], 'childathome':['count', 'mean', iqr]}
nls97.groupby(['highestdegree']).agg(aggdict) 
```

```py
 weeksworked06           childathome
                      count   mean    iqr       count   mean    iqr  
highestdegree                            
0\. None                 666   29.7    47.0        408    1.8   3.0
1\. GED                 1129   32.9    40.0        702    1.7   3.0
2\. High School         3262   39.4    21.0       1881    1.9   2.0
3\. Associates           755   40.2    19.0        448    1.9   2.0
4\. Bachelors           1683   42.3    13.5        859    1.9   1.0
5\. Masters              703   41.8    13.5        379    1.9   1.0
6\. PhD                   63   38.5    22.0         33    1.9   2.0
7\. Professional         127   27.8    43.0         60    1.8   1.0 
```

1.  定义一个函数来返回选定的汇总统计量：

    ```py
    def gettots(x):
    ...   out = {}
    ...   out['qr1'] = x.quantile(0.25)
    ...   out['med'] = x.median()
    ...   out['qr3'] = x.quantile(0.75)
    ...   out['count'] = x.count()
    ...   return out 
    ```

1.  使用 `apply` 运行函数。

这将创建一个具有多重索引的 Series，基于 `highestdegree` 值和所需的汇总统计量：

```py
nls97.groupby(['highestdegree'])['weeksworked06'].\
  apply(gettots) 
```

```py
highestdegree        
0\. None          qr1         5
                 med        35
                 qr3        52
                 count     666
1\. GED           qr1        12
                 med        42
                 qr3        52
                 count   1,129
2\. High School   qr1        31
                 med        52
                 qr3        52
                 count   3,262
3\. Associates    qr1        33
                 med        52
                 qr3        52
                 count     755
4\. Bachelors     qr1        38
                 med        52
                 qr3        52
                 count   1,683
5\. Masters       qr1        38
                 med        52
                 qr3        52
                 count     703
6\. PhD           qr1        30
                 med        50
                 qr3        52
                 count      63
7\. Professional  qr1         6
                 med        30
                 qr3        49
                 count     127
Name: weeksworked06, dtype: float64 
```

1.  使用 `reset_index` 来使用默认索引，而不是由 `groupby` DataFrame 创建的索引：

    ```py
    nls97.groupby(['highestdegree'])['weeksworked06'].\
      apply(gettots).reset_index() 
    ```

    ```py
     highestdegree   level_1    weeksworked06
    0           0\. None       qr1                5
    1           0\. None       med               35
    2           0\. None       qr3               52
    3           0\. None     count              666
    4            1\. GED       qr1               12
    5            1\. GED       med               42
    6            1\. GED       qr3               52
    7            1\. GED     count            1,129
    8    2\. High School       qr1               31
    9    2\. High School       med               52
    10   2\. High School       qr3               52
    11   2\. High School     count            3,262
    12    3\. Associates       qr1               33
    13    3\. Associates       med               52
    14    3\. Associates       qr3               52
    15    3\. Associates     count              755
    16     4\. Bachelors       qr1               38
    17     4\. Bachelors       med               52
    18     4\. Bachelors       qr3               52
    19     4\. Bachelors     count            1,683
    20       5\. Masters       qr1               38
    21       5\. Masters       med               52
    22       5\. Masters       qr3               52
    23       5\. Masters     count              703
    24           6\. PhD       qr1               30
    25           6\. PhD       med               50
    26           6\. PhD       qr3               52
    27           6\. PhD     count               63
    28  7\. Professional       qr1                6
    29  7\. Professional       med               30
    30  7\. Professional       qr3               49
    31  7\. Professional     count              127 
    ```

1.  反而用 `unstack` 链接，以基于汇总变量创建列。

这将创建一个 DataFrame，`highestdegree` 值作为索引，聚合值作为列：

```py
nlssums = nls97.groupby(['highestdegree'])\
  ['weeksworked06'].apply(gettots).unstack()
nlssums 
```

```py
 qr1   med   qr3   count
highestdegree                       
0\. None            5    35    52     666
1\. GED            12    42    52   1,129
2\. High School    31    52    52   3,262
3\. Associates     33    52    52     755
4\. Bachelors      38    52    52   1,683
5\. Masters        38    52    52     703
6\. PhD            30    50    52      63
7\. Professional    6    30    49     127 
```

```py
nlssums.info() 
```

```py
<class 'pandas.core.frame.DataFrame'>
Index: 8 entries, 0\. None to 7\. Professional
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   qr1     8 non-null      float64
 1   med     8 non-null      float64
 2   qr3     8 non-null      float64
 3   count   8 non-null      float64
dtypes: float64(4)
memory usage: 320.0+ bytes 
```

`unstack` 在我们希望将索引的某些部分旋转到列轴时非常有用。

## 它是如何工作的……

我们在 *第 2 步* 中定义了一个非常简单的函数，用于按组计算四分位数范围。然后，我们在 *第 3 步* 中将该函数调用包含在我们的聚合函数列表中。

*第 4 步* 和 *第 5 步* 稍微复杂一些。我们定义了一个计算第一和第三四分位数以及中位数并统计行数的函数。它返回一个包含这些值的 Series。通过将 `groupby` DataFrame 与 *第 5 步* 中的 `apply` 结合，我们可以让 `gettots` 函数返回每个组的该 Series。

*第 5 步* 给出了我们想要的数字，但可能不是最好的格式。例如，如果我们想将数据用于其他操作——比如可视化——我们需要链式调用一些额外的方法。一种可能性是使用 `reset_index`。这将用默认索引替换多重索引。另一种选择是使用 `unstack`。这将根据索引的第二级（具有 `qr1`、`med`、`qr3` 和 `count` 值）创建列。

## 还有更多……

有趣的是，随着教育程度的提高，工作周数和家中孩子数量的四分位数范围显著下降。那些教育程度较低的群体在这些变量上似乎有更大的变异性。这应该被更仔细地检查，并且对于统计检验有影响，因为统计检验假设各组之间的方差是相同的。

## 另见

在 *第十一章*《整理与重塑数据》中，我们对 `stack` 和 `unstack` 做了更多的操作。

# 使用 groupby 改变 DataFrame 的分析单位

在前一个步骤的最后，我们创建的 DataFrame 是我们努力按组生成多个汇总统计量时的一个意外副产品。有时我们确实需要聚合数据来改变分析的单位——例如，从每个家庭的月度公用事业费用到每个家庭的年度公用事业费用，或从学生按课程的成绩到学生的整体 **平均绩点** (**GPA**)。

`groupby` 是一个很好的工具，特别适用于折叠分析单位，特别是在需要进行汇总操作时。当我们只需要选择未重复的行时——也许是每个个体在给定间隔内的第一行或最后一行——那么 `sort_values` 和 `drop_duplicates` 的组合就能胜任。但是，我们经常需要在折叠之前对每组的行进行一些计算。这时 `groupby` 就非常方便了。

## 准备工作

我们将再次处理每日病例数据，该数据每天每个国家有一行记录。我们还将处理巴西陆地温度数据，该数据每个气象站每个月有一行记录。

## 如何做…

我们将使用 `groupby` 创建一个按组的汇总值的 DataFrame：

1.  导入 `pandas` 并加载 COVID-19 和陆地温度数据：

    ```py
    import pandas as pd
    coviddaily = pd.read_csv("data/coviddaily.csv", parse_dates=["casedate"])
    ltbrazil = pd.read_csv("data/ltbrazil.csv") 
    ```

1.  让我们查看数据的样本，以便回顾其结构。每个国家（`location`）每天有一行记录，包括当天的新病例数和死亡数（我们使用随机种子以便每次生成相同的值）：

    ```py
    coviddaily[['location','casedate',
      'new_cases','new_deaths']]. \
      set_index(['location','casedate']). \
      sample(10, random_state=1) 
    ```

    ```py
     new_cases  \
    location                 casedate               
    Andorra                  2020-03-15           1  
    Portugal                 2022-12-04       3,963  
    Eswatini                 2022-08-07          22  
    Singapore                2020-08-30         451  
    Georgia                  2020-08-02          46  
    British Virgin Islands   2020-08-30          14  
    Thailand                 2023-01-29         472  
    Bolivia                  2023-12-17         280  
    Montenegro               2021-08-15       2,560  
    Eswatini                 2022-04-17         132  
                                         new_deaths 
    location                 casedate               
    Andorra                  2020-03-15           0 
    Portugal                 2022-12-04          69 
    Eswatini                 2022-08-07           2 
    Singapore                2020-08-30           0 
    Georgia                  2020-08-02           1 
    British Virgin Islands   2020-08-30           0 
    Thailand                 2023-01-29          29 
    Bolivia                  2023-12-17           0 
    Montenegro               2021-08-15           9 
    Eswatini                 2022-04-17           0 
    ```

1.  现在，我们可以将 COVID-19 数据从每天每个国家转换为每天所有国家的汇总数据。为了限制要处理的数据量，我们仅包括 2023 年 2 月至 2024 年 1 月之间的日期。

    ```py
    coviddailytotals = coviddaily.loc[coviddaily.\
      casedate.between('2023-02-01','2024-01-31')].\
      groupby(['casedate'], as_index=False)\
      [['new_cases','new_deaths']].\
      sum()
    coviddailytotals.head(10) 
    ```

    ```py
     casedate  new_cases  new_deaths
    0 2023-02-05  1,385,583      69,679
    1 2023-02-12  1,247,389      10,105
    2 2023-02-19  1,145,666       8,539
    3 2023-02-26  1,072,712       7,771
    4 2023-03-05  1,028,278       7,001
    5 2023-03-12    894,678       6,340
    6 2023-03-19    879,074       6,623
    7 2023-03-26    833,043       6,711
    8 2023-04-02    799,453       5,969
    9 2023-04-09    701,000       5,538 
    ```

1.  让我们看一看巴西平均温度数据的一些行：

    ```py
    ltbrazil.head(2).T 
    ```

    ```py
     0                1
    locationid           BR000082400      BR000082704
    year                        2023             2023
    month                          1                1
    temperature                   27               27
    latitude                      -4               -8
    longitude                    -32              -73
    elevation                     59              194
    station      FERNANDO_DE_NORONHA  CRUZEIRO_DO_SUL
    countryid                     BR               BR
    country                   Brazil           Brazil
    latabs                         4                8 
    ```

1.  创建一个包含巴西每个气象站平均温度的 DataFrame。

首先删除具有缺失温度值的行：

```py
ltbrazil = ltbrazil.dropna(subset=['temperature'])
ltbrazilavgs = ltbrazil.groupby(['station'],
...   as_index=False).\
...   agg({'latabs':'first','elevation':'first',
...   'temperature':'mean'})
ltbrazilavgs.head(10) 
```

```py
 station  latabs  elevation  temperature
0             ALTAMIRA       3        112           28
1   ALTA_FLORESTA_AERO      10        289           32
2                ARAXA      20      1,004           22
3              BACABAL       4         25           29
4                 BAGE      31        242           20
5       BARRA_DO_CORDA       6        153           28
6            BARREIRAS      12        439           27
7  BARTOLOMEU_LISANDRO      22         17           26
8                BAURU      22        617           25
9                BELEM       1         10           28 
```

让我们更详细地看一看这些示例中的聚合函数是如何工作的。

## 它是如何工作的…

在 *步骤 3* 中，首先选择我们需要的日期。我们基于 `casedate` 创建一个 DataFrame 的 `groupby` 对象，选择 `new_cases` 和 `new_deaths` 作为聚合变量，并选择 `sum` 作为聚合函数。这将为每个组（`casedate`）产生 `new_cases` 和 `new_deaths` 的总和。根据您的目的，您可能不希望 `casedate` 成为索引，如果没有将 `as_index` 设置为 `False` 将会发生这种情况。

我们经常需要在不同的聚合变量上使用不同的聚合函数。我们可能想要对一个变量取第一个（或最后一个）值，并对另一个变量的值按组取平均值。这就是我们在 *步骤 5* 中所做的。我们通过将一个字典传递给 `agg` 函数来实现这一点，字典的键是我们的聚合变量，值是要使用的聚合函数。

# 使用 pivot_table 改变 DataFrame 的分析单位

在前一个示例中，我们可以使用 pandas 的 `pivot_table` 函数而不是 `groupby`。`pivot_table` 可以用于根据分类变量的值生成汇总统计信息，就像我们用 `groupby` 做的那样。`pivot_table` 函数还可以返回一个 DataFrame，这在本示例中将会看到。

## 准备工作

我们将再次处理 COVID-19 每日病例数据和巴西陆地温度数据。温度数据每个气象站每个月有一行记录。

## 如何做…

让我们从 COVID-19 数据创建一个 DataFrame，显示每一天在所有国家中的总病例数和死亡人数：

1.  我们首先重新加载 COVID-19 和温度数据：

    ```py
    import pandas as pd
    coviddaily = pd.read_csv("data/coviddaily.csv", parse_dates=["casedate"])
    ltbrazil = pd.read_csv("data/ltbrazil.csv") 
    ```

1.  现在，我们可以调用`pivot_table`函数了。我们将一个列表传递给`values`，以指示要进行汇总计算的变量。我们使用`index`参数来表示我们希望按`casedate`进行汇总，并通过将其传递给`aggfunc`来表示我们只希望求和。注意，我们得到的总数与之前使用`groupby`时的结果相同：

    ```py
    coviddailytotals = \
      pd.pivot_table(coviddaily.loc[coviddaily.casedate. \
      between('2023-02-01','2024-01-31')],
      values=['new_cases','new_deaths'], index='casedate',
      aggfunc='sum')
    coviddailytotals.head(10) 
    ```

    ```py
     new_cases  new_deaths
    casedate                        
    2023-02-05  1,385,583      69,679
    2023-02-12  1,247,389      10,105
    2023-02-19  1,145,666       8,539
    2023-02-26  1,072,712       7,771
    2023-03-05  1,028,278       7,001
    2023-03-12    894,678       6,340
    2023-03-19    879,074       6,623
    2023-03-26    833,043       6,711
    2023-04-02    799,453       5,969
    2023-04-09    701,000       5,538 
    ```

1.  让我们尝试使用`pivot_table`处理土地温度数据，并进行更复杂的聚合。我们希望得到每个站点的纬度（`latabs`）和海拔高度的第一个值，以及平均温度。回想一下，纬度和海拔值对于一个站点来说是固定的。我们将所需的聚合操作作为字典传递给`aggfunc`。同样，我们得到的结果与前一个例子中的结果一致：

    ```py
    ltbrazil = ltbrazil.dropna(subset=['temperature'])
    ltbrazilavgs = \
      pd.pivot_table(ltbrazil, index=['station'],
      aggfunc={'latabs':'first','elevation':'first',
      'temperature':'mean'})
    ltbrazilavgs.head(10) 
    ```

    ```py
     elevation  latabs  temperature
    station                                           
    ALTAMIRA                   112       3           28
    ALTA_FLORESTA_AERO         289      10           32
    ARAXA                    1,004      20           22
    BACABAL                     25       4           29
    BAGE                       242      31           20
    BARRA_DO_CORDA             153       6           28
    BARREIRAS                  439      12           27
    BARTOLOMEU_LISANDRO         17      22           26
    BAURU                      617      22           25
    BELEM                       10       1           28 
    ```

## 工作原理……

如我们所见，无论是使用`groupby`还是`pivot_table`，我们得到的结果是相同的。分析师应该选择他们自己和团队成员觉得最直观的方法。由于我的工作流程更常使用`groupby`，所以在聚合数据以创建新的 DataFrame 时，我更倾向于使用这种方法。

# 概述

在本章中，我们探讨了使用 NumPy 和 pandas 进行数据聚合的多种策略。我们还讨论了每种技术的优缺点，包括如何根据数据和聚合任务选择最有效、最直观的方法。由于大多数数据清理和处理项目都会涉及某种分割-应用-合并的操作，因此熟悉每种方法是很有必要的。在下一章中，我们将学习如何合并 DataFrame 并处理后续的数据问题。

# 加入我们社区的 Discord

加入我们社区的 Discord 空间，与作者和其他读者讨论：

[`discord.gg/p8uSgEAETX`](https://discord.gg/p8uSgEAETX )

![](img/QR_Code10336218961138498953.png)
