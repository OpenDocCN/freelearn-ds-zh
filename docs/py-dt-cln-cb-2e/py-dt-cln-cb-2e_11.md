

# 第十一章：整理和重塑数据

引用列夫·托尔斯泰的智慧（“幸福的家庭都是相似的；每个不幸的家庭都有其不幸的方式。”），哈德利·威克姆告诉我们，所有整洁的数据本质上是相似的，但所有不整洁的数据都有其独特的混乱方式。我们多少次盯着某些数据行，心里想，*“这…怎么回事…为什么他们这么做？”* 这有点夸张。尽管数据结构不良的方式有很多，但在人类创造力方面是有限的。我们可以将数据集偏离标准化或整洁形式的最常见方式进行分类。

这是哈德利·威克姆在他关于整洁数据的开创性著作中的观察。我们可以依赖这项工作，以及我们自己在处理结构奇特的数据时的经验，为我们需要进行的重塑做好准备。不整洁的数据通常具有以下一种或多种特征：缺乏明确的按列合并关系；一对多关系中的*一方*数据冗余；多对多关系中的数据冗余；列名中存储值；将多个值存储在一个变量值中；数据未按分析单位进行结构化。（尽管最后一种情况不一定是数据不整洁的表现，但我们将在接下来的几个菜谱中回顾的某些技术也适用于常见的分析单位问题。）

在本章中，我们使用强大的工具来应对像前面那样的数据清理挑战。具体而言，我们将讨论以下内容：

+   移除重复行

+   修复多对多关系

+   使用`stack`和`melt`将数据从宽格式重塑为长格式

+   多组列的合并

+   使用`unstack`和`pivot`将数据从长格式重塑为宽格式

# 技术要求

完成本章的任务，您将需要 pandas、NumPy 和 Matplotlib。我使用的是 pandas 2.1.4，但该代码可以在 pandas 1.5.3 或更高版本上运行。

本章中的代码可以从本书的 GitHub 仓库下载，[`github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition`](https://github.com/PacktPublishing/Python-Data-Cleaning-Cookbook-Second-Edition)。

# 移除重复行

数据在分析单位上的重复有几种原因：

+   当前的 DataFrame 可能是一次一对多合并的结果，其中一方是分析单位。

+   该 DataFrame 是反复测量或面板数据被压缩为平面文件，这只是第一种情况的特殊情况。

+   我们可能正在处理一个分析文件，其中多个一对多关系已被展平，形成多对多关系。

当*单一*侧面是分析单元时，*多重*侧面的数据可能需要以某种方式进行合并。例如，如果我们在分析一组大学生的结果，学生是分析单元；但我们也可能拥有每个学生的课程注册数据。为了准备数据进行分析，我们可能需要首先统计每个学生的课程数量、总学分或计算 GPA，最后得到每个学生的一行数据。通过这个例子，我们可以概括出，在去除重复数据之前，我们通常需要对*多重*侧面的信息进行聚合。

在这个示例中，我们查看了 pandas 处理去除重复行的技巧，并考虑了在这个过程中什么时候需要进行聚合，什么时候不需要。在下一示例中，我们将解决多对多关系中的重复问题。

## 准备工作

在本示例中，我们将处理 COVID-19 每日病例数据。该数据集为每个国家提供每一天的一行数据，每行记录当天的新病例和新死亡人数。每个国家还有人口统计数据和病例死亡的累计总数，因此每个国家的最后一行提供了总病例数和总死亡人数。

**数据说明**

Our World in Data 提供了 COVID-19 公共使用数据，网址是[`ourworldindata.org/covid-cases`](https://ourworldindata.org/covid-cases)。该数据集包括总病例和死亡人数、已进行的测试、医院床位以及诸如中位年龄、国内生产总值和糖尿病患病率等人口统计数据。本示例中使用的数据集是 2024 年 3 月 3 日下载的。

## 如何操作…

我们使用`drop_duplicates`去除每个国家在 COVID-19 每日数据中的重复人口统计数据。当我们需要先进行一些聚合再去除重复数据时，我们也可以探索`groupby`作为`drop_duplicates`的替代方法：

1.  导入`pandas`和 COVID-19 每日病例数据：

    ```py
    import pandas as pd
    covidcases = pd.read_csv("data/covidcases.csv") 
    ```

1.  为每日病例和死亡列、病例总数列以及人口统计列创建列表（`total_cases`和`total_deaths`列分别是该国家的病例和死亡的累计总数）：

    ```py
    dailyvars = ['casedate','new_cases','new_deaths']
    totvars = ['location','total_cases','total_deaths']
    demovars = ['population','population_density',
    ...   'median_age','gdp_per_capita',
    ...   'hospital_beds_per_thousand','region']
    covidcases[dailyvars + totvars + demovars].head(2).T 
    ```

    ```py
     0               1
    casedate                        2020-03-01      2020-03-15
    new_cases                                1               6
    new_deaths                               0               0
    location                       Afghanistan     Afghanistan
    total_cases                              1               7
    total_deaths                             0               0
    population                        41128772        41128772
    population_density                      54              54
    median_age                              19              19
    gdp_per_capita                       1,804           1,804
    hospital_beds_per_thousand               0               0
    region                          South Asia      South Asia 
    ```

1.  创建一个仅包含每日数据的 DataFrame：

    ```py
    coviddaily = covidcases[['location'] + dailyvars]
    coviddaily.shape 
    ```

    ```py
    (36501, 4) 
    ```

    ```py
    coviddaily.head() 
    ```

    ```py
     location       casedate     new_cases     new_deaths
    0  Afghanistan     2020-03-01             1              0
    1  Afghanistan     2020-03-15             6              0
    2  Afghanistan     2020-03-22            17              0
    3  Afghanistan     2020-03-29            67              2
    4  Afghanistan     2020-04-05           183              3 
    ```

1.  为每个国家选择一行。

检查预计有多少个国家（位置），方法是获取唯一位置的数量。按位置和病例日期排序。然后使用`drop_duplicates`选择每个位置的一个行，并使用 keep 参数指示我们希望为每个国家选择最后一行：

```py
covidcases.location.nunique() 
```

```py
231 
```

```py
coviddemo = \
  covidcases[['casedate'] + totvars + demovars].\
  sort_values(['location','casedate']).\
  drop_duplicates(['location'], keep='last').\
  rename(columns={'casedate':'lastdate'})
coviddemo.shape 
```

```py
(231, 10) 
```

```py
coviddemo.head(2).T 
```

```py
 204                379
lastdate                      2024-02-04         2024-01-28
location                     Afghanistan            Albania
total_cases                      231,539            334,863
total_deaths                       7,982              3,605
population                      41128772            2842318
population_density                    54                105
median_age                            19                 38
gdp_per_capita                     1,804             11,803
hospital_beds_per_thousand             0                  3
region                        South Asia     Eastern Europe 
```

1.  对每个组的值进行求和。

使用 pandas 的 DataFrame groupby 方法来计算每个国家的病例和死亡总数。（我们在这里计算病例和死亡的总和，而不是使用 DataFrame 中已存在的病例和死亡的累计总数。）同时，获取一些在每个国家的所有行中都重复的列的最后一个值：`median_age`、`gdp_per_capita`、`region`和`casedate`。（我们只选择 DataFrame 中的少数几列。）请注意，数字与*第 4 步*中的一致：

```py
covidtotals = covidcases.groupby(['location'],
...   as_index=False).\
...   agg({'new_cases':'sum','new_deaths':'sum',
...     'median_age':'last','gdp_per_capita':'last',
...     'region':'last','casedate':'last',
...     'population':'last'}).\
...   rename(columns={'new_cases':'total_cases',
...     'new_deaths':'total_deaths',
...     'casedate':'lastdate'})
covidtotals.head(2).T 
```

```py
 0                     1
location              Afghanistan               Albania
total_cases               231,539               334,863
total_deaths                7,982                 3,605
median_age                     19                    38
gdp_per_capita              1,804                11,803
region                 South Asia        Eastern Europe
lastdate               2024-02-04            2024-01-28
population               41128772               2842318 
```

选择使用`drop_duplicates`还是`groupby`来消除数据冗余，取决于我们是否需要在压缩*多*方之前进行任何聚合。

## 它是如何工作的……

COVID-19 数据每个国家每天有一行，但实际上很少有数据是每日数据。只有`casedate`、`new_cases`和`new_deaths`可以视为每日数据。其他列则显示累计病例和死亡人数，或是人口统计数据。累计数据是冗余的，因为我们已有`new_cases`和`new_deaths`的实际值。人口统计数据在所有日期中对于每个国家来说值是相同的。

国家（及其相关人口统计数据）与每日数据之间有一个隐含的一对多关系，其中*一*方是国家，*多*方是每日数据。我们可以通过创建一个包含每日数据的 DataFrame 和另一个包含人口统计数据的 DataFrame 来恢复这种结构。我们在*步骤 3*和*4*中做到了这一点。当我们需要跨国家的总数时，我们可以自己生成，而不是存储冗余数据。

然而，运行总计变量并非完全没有用处。我们可以使用它们来检查我们关于病例总数和死亡总数的计算。*步骤 5*展示了如何在需要执行的不仅仅是去重时，使用`groupby`来重构数据。在这种情况下，我们希望对每个国家的`new_cases`和`new_deaths`进行汇总。

## 还有更多……

我有时会忘记一个小细节。在改变数据结构时，某些列的含义可能会发生变化。在这个例子中，`casedate`变成了每个国家最后一行的日期。我们将该列重命名为`lastdate`。

## 另请参见

我们在*第九章*《聚合时修复混乱数据》中更详细地探讨了`groupby`。

Hadley Wickham 的*整洁数据*论文可以在[`vita.had.co.nz/papers/tidy-data.pdf`](https://vita.had.co.nz/papers/tidy-data.pdf)找到。

# 修复多对多关系

有时我们必须处理从多对多合并创建的数据表。这是一种在左侧和右侧的合并列值都被重复的合并。正如我们在前一章中讨论的那样，数据文件中的多对多关系通常代表多个一对多关系，其中*一*方被移除。数据集 A 和数据集 B 之间有一对多关系，数据集 A 和数据集 C 之间也有一对多关系。我们有时面临的问题是，收到的数据文件将 B 和 C 合并在一起，而将 A 排除在外。

处理这种结构的数据的最佳方式是重新创建隐含的一对多关系，若可能的话。我们通过首先创建一个类似 A 的数据集来实现；也就是说，假设有一个多对多关系，我们可以推测 A 的结构是怎样的。能够做到这一点的关键是为数据两边的多对多关系识别出一个合适的合并列。这个列或这些列将在 B 和 C 数据集中重复，但在理论上的 A 数据集中不会重复。

本教程中使用的数据就是一个很好的例子。我们使用克利夫兰艺术博物馆的收藏数据。每个博物馆收藏品都有多行数据。这些数据包括收藏品的信息（如标题和创作日期）；创作者的信息（如出生和死亡年份）；以及该作品在媒体中的引文。当有多个创作者和多个引文时（这通常发生），行数会重复。更精确地说，每个收藏品的行数是引文和创作者数量的笛卡尔积。所以，如果有 5 条引文和 2 个创作者，我们将看到该项目有 10 行数据。

我们想要的是一个收藏品文件，每个收藏品一行（并且有一个唯一标识符），一个创作者文件，每个创作者对应一行，和一个引文文件，每个引文对应一行。在本教程中，我们将创建这些文件。

你们中的一些人可能已经注意到，这里还有更多的整理工作要做。我们最终需要一个单独的创作者文件，每个创作者一行，另一个文件只包含创作者 ID 和收藏品 ID。我们需要这种结构，因为一个创作者可能会为多个项目创作。我们在这个教程中忽略了这个复杂性。

我应该补充的是，这种情况并不是克利夫兰艺术博物馆的错，该博物馆慷慨地提供了一个 API，可以返回作为 JSON 文件的收藏数据。使用 API 的个人有责任创建最适合自己研究目的的数据文件。直接从更灵活的 JSON 文件结构中工作也是可能的，而且通常是一个不错的选择。我们将在 *第十二章*，*使用用户定义的函数、类和管道自动清理数据* 中演示如何操作。

## 准备工作

我们将使用克利夫兰艺术博物馆收藏的数据。CSV 文件包含有关创作者和引文的数据，这些数据通过 `itemid` 列合并，`itemid` 用来标识收藏品。每个项目可能有一行或多行关于引文和创作者的数据。

**数据说明**

克利夫兰艺术博物馆提供了一个公共访问数据的 API：[`openaccess-api.clevelandart.org/`](https://openaccess-api.clevelandart.org/)。API 提供的数据远不止引文和创作者的数据。本文中的数据是 2024 年 4 月下载的。

## 如何实现…

我们通过恢复数据中隐含的多个一对多关系来处理 DataFrame 之间的多对多关系：

1.  导入 `pandas` 和博物馆的 `collections` 数据。为了更方便显示，我们还将限制 `collection` 和 `title` 列中值的长度：

    ```py
    import pandas as pd
    cma = pd.read_csv("data/cmacollections.csv")
    cma['category'] = cma.category.str.strip().str[0:15]
    cma['title'] = cma.title.str.strip().str[0:30] 
    ```

1.  显示博物馆的一些 `collections` 数据。注意，几乎所有的数据值都是冗余的，除了 `citation`。

同时，显示唯一的 `itemid`、`citation` 和 `creator` 值的数量。有 986 个独特的集合项，12,941 个引用，以及 1,062 对 item/creator 组合：

```py
cma.shape 
```

```py
(17001, 9) 
```

```py
cma.head(4).T 
```

```py
 0                      1  \
itemid                         75551                  75551  
citation             Purrmann, Hans.        <em>Henri Matis  
creatorid                      2,130                  2,130  
creator              Henri Matisse (        Henri Matisse (  
title                         Tulips                 Tulips  
birth_year                      1869                   1869  
death_year                      1954                   1954  
category             Mod Euro - Pain        Mod Euro - Pain  
creation_date                   1914                   1914  
                                   2                      3 
itemid                         75551                  75551 
citation             Flam, Jack D. <        <em>Masters of  
creatorid                      2,130                  2,130 
creator              Henri Matisse (        Henri Matisse ( 
title                         Tulips                 Tulips 
birth_year                      1869                   1869 
death_year                      1954                   1954 
category             Mod Euro - Pain        Mod Euro - Pain 
creation_date                   1914                   1914 
```

```py
cma.itemid.nunique() 
```

```py
986 
```

```py
cma.drop_duplicates(['itemid','citation']).\
  itemid.count() 
```

```py
12941 
```

```py
cma.drop_duplicates(['itemid','creatorid']).\
  itemid.count() 
```

```py
1062 
```

1.  显示一个包含重复引用和创作者的集合项。

只显示前 6 行（实际上共有 28 行）。注意，引用数据对于每个创作者都是重复的：

```py
cma.set_index(['itemid'], inplace=True)
cma.loc[124733, ['title','citation',
  'creation_date','creator','birth_year']].head(6) 
```

```py
 title              citation  \
itemid                                     
124733       Dead Blue Roller       Weigel, J. A. G  
124733       Dead Blue Roller       Weigel, J. A. G  
124733       Dead Blue Roller       Winkler, Friedr  
124733       Dead Blue Roller       Winkler, Friedr  
124733       Dead Blue Roller       Francis, Henry   
124733       Dead Blue Roller       Francis, Henry   
            creation_date               creator birth_year 
itemid                                                
124733               1583       Hans Hoffmann (       1545 
124733               1583       Albrecht Dürer        1471 
124733               1583       Hans Hoffmann (       1545 
124733               1583       Albrecht Dürer        1471 
124733               1583       Hans Hoffmann (       1545 
124733               1583       Albrecht Dürer        1471 
```

1.  创建一个集合 DataFrame。`title`、`category` 和 `creation_date` 应该是每个集合项唯一的，因此我们创建一个仅包含这些列的 DataFrame，并带有 `itemid` 索引。我们得到预期的行数 `986`：

    ```py
    collectionsvars = \
      ['title','category','creation_date']
    cmacollections = cma[collectionsvars].\
      reset_index().\
      drop_duplicates(['itemid']).\
      set_index(['itemid'])
    cmacollections.shape 
    ```

    ```py
    (986, 3) 
    ```

    ```py
    cmacollections.head() 
    ```

    ```py
     title  \
    itemid                                  
    75551                           Tulips  
    75763   Procession or Pardon at Perros  
    78982       The Resurrection of Christ  
    84662                The Orange Christ  
    86110   Sunset Glow over a Fishing Vil  
                 category   creation_date 
    itemid                                
    75551   Mod Euro - Pain          1914 
    75763   Mod Euro - Pain          1891 
    78982   P - German befo          1622 
    84662   Mod Euro - Pain          1889 
    86110   ASIAN - Hanging   1460s–1550s 
    ```

1.  让我们看看新 DataFrame `cmacollections` 中的同一项，该项在 *步骤 3* 中已经展示过：

    ```py
    cmacollections.loc[124733] 
    ```

    ```py
    title            Dead Blue Roller
    category              DR - German
    creation_date                1583
    Name: 124733, dtype: object 
    ```

1.  创建一个引用（citations）DataFrame。

这将只包含 `itemid` 和 `citation`：

```py
cmacitations = cma[['citation']].\
  reset_index().\
  drop_duplicates(['itemid','citation']).\
  set_index(['itemid'])
cmacitations.loc[124733] 
```

```py
 citation
itemid                     
124733       Weigel, J. A. G
124733       Winkler, Friedr
124733       Francis, Henry
124733       Kurz, Otto. <em
124733       Minneapolis Ins
124733       Pilz, Kurt. "Ha
124733       Koschatzky, Wal
124733       Johnson, Mark M
124733       Kaufmann, Thoma
124733        Koreny, Fritz.
124733       Achilles-Syndra
124733       Schoch, Rainer,
124733       DeGrazia, Diane
124733       Dunbar, Burton 
```

1.  创建一个创作者 DataFrame：

    ```py
    creatorsvars = \
      ['creator','birth_year','death_year']
    cmacreators = cma[creatorsvars].\
      reset_index().\
      drop_duplicates(['itemid','creator']).\
      set_index(['itemid'])
    cmacreators.loc[124733] 
    ```

    ```py
     creator        birth_year      death_year
    itemid                                                     
    124733     Hans Hoffmann (             1545            1592
    124733     Albrecht Dürer              1471            1528 
    ```

1.  统计出生在 1950 年后创作者的集合项数量。

首先，将 `birth_year` 值从字符串转换为数字。然后，创建一个只包含年轻艺术家的 DataFrame：

```py
cmacreators['birth_year'] = \
  cmacreators.birth_year.str.\
  findall("\d+").str[0].astype(float)
youngartists = \
  cmacreators.loc[cmacreators.birth_year>1950,
  ['creator']].assign(creatorbornafter1950='Y')
youngartists.shape[0]==youngartists.index.nunique() 
```

```py
True 
```

```py
youngartists 
```

```py
 creator creatorbornafter1950
itemid                                     
168529  Richard Barnes                     Y
369885  Simone Leigh (A                    Y
371392  Belkis Ayón (Cu                    Y
378931  Teresa Margolle                    Y 
```

1.  现在，我们可以将 `youngartists` DataFrame 与集合 DataFrame 合并，创建一个标记，用于标识至少有一个创作者出生在 1950 年后 的集合项：

    ```py
    cmacollections = \
      pd.merge(cmacollections, youngartists,
      left_on=['itemid'], right_on=['itemid'], how='left')
    cmacollections.fillna({'creatorbornafter1950':'N'}, inplace=True)
    cmacollections.shape 
    ```

    ```py
    (986, 9) 
    ```

    ```py
    cmacollections.creatorbornafter1950.value_counts() 
    ```

    ```py
    creatorbornafter1950
    N    982
    Y      4
    Name: count, dtype: int64 
    ```

现在我们有了三个 DataFrame——集合项（`cmacollections`）、引用（`cmacitations`）和创作者（`cmacreators`）——而不是一个。`cmacollections` 与 `cmacitations` 和 `cmacreators` 都存在一对多关系。

## 它是如何工作的……

如果你主要直接处理企业数据，你可能很少会看到这种结构的文件，但许多人并没有这么幸运。如果我们从博物馆请求关于其收藏的媒体引用和创作者的数据，得到类似这样的数据文件并不完全令人惊讶，其中引用和创作者的数据是重复的。但看起来像是集合项唯一标识符的存在，让我们有希望恢复集合项与其引用之间、一对多的关系，以及集合项与创作者之间、一对多的关系。

*步骤 2* 显示有 986 个独特的 `itemid` 值。这表明在 17,001 行的 DataFrame 中，可能只包含 986 个集合项。共有 12,941 对独特的 `itemid` 和 `citation`，即每个集合项平均有约 13 条引用。共有 1,062 对 `itemid` 和 `creator`。

*步骤 3* 展示了集合项目值（如 `title`）的重复情况。返回的行数等于左右合并条件的笛卡尔积。对于 *Dead Blue Roller* 项目，共有 28 行（我们在*步骤 3*中只展示了其中 6 行），因为它有 14 个引用和 2 个创作者。每个创作者的行会被重复 14 次；每个引用重复一次，针对每个创作者。每个引用会出现两次；一次针对每个创作者。对于非常少的用例，保留这种状态的数据是有意义的。

我们的“北极星”是 `itemid` 列，它帮助我们将数据转化为更好的结构。在*步骤 4*中，我们利用它来创建集合 DataFrame。我们仅保留每个 `itemid` 值的一行，并获取与集合项目相关的其他列，而非引用或创作者——`title`、`category` 和 `creation_date`（因为 `itemid` 是索引，我们需要先重置索引，然后再删除重复项）。

我们按照相同的程序，在*步骤 6* 和 *步骤 7* 中分别创建 `citations` 和 `creators` DataFrame。我们使用 `drop_duplicates` 保留 `itemid` 和 `citation` 的唯一组合，和 `itemid` 和 `creator` 的唯一组合。这让我们得到了预期的行数：14 行 `citations` 数据和 2 行 `creators` 数据。

*步骤 8* 展示了我们如何使用这些 DataFrame 来构建新列并进行分析。我们想要计算至少有一个创作者出生在 1950 年之后的集合项目数量。分析的单位是集合项目，但我们需要从创作者 DataFrame 中获取信息来进行计算。由于 `cmacollections` 和 `cmacreators` 之间是多对一的关系，我们确保在创作者 DataFrame 中每个 `itemid` 只获取一行数据，即使某个项目有多个创作者出生在 1950 年之后：

```py
youngartists.shape[0]==youngartists.index.nunique() 
```

## 还有更多...

当我们处理定量数据时，多对多合并所产生的重复数据最为棘手。如果原始文件中包含了每个集合项目的评估价值，这些值将像 `title` 一样被重复。如果我们对评估价值生成描述性统计信息，结果会偏差很大。例如，如果 *Dead Blue Roller* 项目的评估价值为 1,000,000 美元，在汇总评估价值时，我们将得到 28,000,000 美元，因为有 28 个重复值。

这突显了标准化和整洁数据的重要性。如果有评估价值列，我们会将其包含在*步骤 4*中创建的 `cmacollections` DataFrame 中。这个值将不会被重复，并且我们能够为集合生成汇总统计数据。

我发现始终回到分析单位是非常有帮助的，它与整洁数据的概念有重叠，但在某些方面有所不同。如果我们只关心 1950 年后出生的创作者的数量，而不是 1950 年后出生的创作者所对应的收藏项数量，*第 8 步*中的方法会完全不同。在这种情况下，分析单位将是创作者，我们只会使用创作者数据框。

## 另请参见

我们在*第十章*的*处理合并数据框时的数据问题*部分中讨论了多对多合并。

我们在*第十二章*的*使用用户定义函数、类和管道自动化数据清理*部分中的*处理非表格数据结构的类*示例中，展示了处理这种结构数据的完全不同方式。

# 使用 stack 和 melt 将数据从宽格式转换为长格式

Wickham 确定的一种不整洁数据类型是将变量值嵌入列名中。虽然在企业或关系型数据中这种情况很少发生，但在分析数据或调查数据中却相当常见。变量名可能会有后缀，指示时间段，如月份或年份。或者，调查中相似的变量可能有类似的名称，比如 `familymember1age` 和 `familymember2age`，因为这样便于使用，并且与调查设计者对变量的理解一致。

调查数据中这种混乱相对频繁发生的一个原因是，一个调查工具上可能有多个分析单位。一个例子是美国的十年一次人口普查，它既询问家庭问题，也询问个人问题。调查数据有时还包括重复测量或面板数据，但通常每个受访者只有一行数据。在这种情况下，新测量值或新回答会存储在新列中，而不是新行中，列名将与早期时期的响应列名相似，唯一的区别是后缀的变化。

**美国青年纵向调查**（**NLS**）是一个很好的例子。它是面板数据，每个个体每年都进行调查。然而，分析文件中每个受访者只有一行数据。类似“在某一年工作了多少周”这样的问题的回答会放入新的列中。整理 NLS 数据意味着将如 `weeksworked17` 到 `weeksworked21`（即 2017 年到 2021 年间的工作周数）等列，转换成仅有一列表示工作周数，另一列表示年份，且每个人有五行数据（每年一行），而不是一行数据。这有时被称为将数据从*宽格式转换为长格式*。

令人惊讶的是，pandas 有几个函数使得像这样的转换相对容易：`stack`、`melt` 和 `wide_to_long`。我们在这个示例中使用 `stack` 和 `melt`，并在接下来的部分探讨 `wide_to_long`。

## 准备工作

我们将处理 NLS 中每年工作周数和大学入学状态的数据。DataFrame 中每行对应一位调查参与者。

**数据说明**

**国家纵向调查**（**NLS**），由美国劳工统计局管理，是对 1997 年开始时在高中的个体进行的纵向调查。参与者每年接受一次调查，直到 2023 年。调查数据可供公众使用，网址为[nlsinfo.org](https://nlsinfo.org)。

## 如何操作…

我们将使用`stack`和`melt`将 NLS 工作周数据从宽格式转换为长格式，同时提取列名中的年份值：

1.  导入`pandas`和 NLS 数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False) 
    ```

1.  查看一些工作周数的值。

首先，设置索引：

```py
nls97.set_index(['originalid'], inplace=True)
weeksworkedcols = ['weeksworked17','weeksworked18',
  'weeksworked19','weeksworked20','weeksworked21']
nls97.loc[[2,3],weeksworkedcols].T 
```

```py
originalid          2       3
weeksworked17      52      52
weeksworked18      52      52
weeksworked19      52       9
weeksworked20      52       0
weeksworked21      46       0 
```

```py
nls97.shape 
```

```py
(8984, 110) 
```

1.  使用`stack`将数据从宽格式转换为长格式。

首先，仅选择`weeksworked##`列。使用 stack 将原始 DataFrame 中的每个列名移入索引，并将`weeksworked##`的值移入相应的行。重置`index`，使得`weeksworked##`列名成为`level_1`列（我们将其重命名为`year`）的值，`weeksworked##`的值成为 0 列（我们将其重命名为`weeksworked`）的值：

**数据说明**

对于未来升级到`pandas 3.0`，我们需要在`stack`函数中提到关键字参数`(future_stack=True)`。

```py
weeksworked = nls97[weeksworkedcols].\
  stack().\
  reset_index().\
  rename(columns={'level_1':'year',0:'weeksworked'})
weeksworked.loc[weeksworked.originalid.isin([2,3])] 
```

```py
 originalid                year       weeksworked
5            2       weeksworked17                52
6            2       weeksworked18                52
7            2       weeksworked19                52
8            2       weeksworked20                52
9            2       weeksworked21                46
10           3       weeksworked17                52
11           3       weeksworked18                52
12           3       weeksworked19                 9
13           3       weeksworked20                 0
14           3       weeksworked21                 0 
```

1.  修正`year`值。

获取年份值的最后几位数字，将其转换为整数，并加上 2,000：

```py
weeksworked['year'] = \
  weeksworked.year.str[-2:].astype(int)+2000
weeksworked.loc[weeksworked.originalid.isin([2,3])] 
```

```py
 originalid       year       weeksworked
5                 2       2017                52
6                 2       2018                52
7                 2       2019                52
8                 2       2020                52
9                 2       2021                46
10                3       2017                52
11                3       2018                52
12                3       2019                 9
13                3       2020                 0
14                3       2021                 0 
```

```py
weeksworked.shape 
```

```py
(44920, 3) 
```

1.  或者，使用`melt`将数据从宽格式转换为长格式。

首先，重置`index`并选择`originalid`和`weeksworked##`列。使用`melt`的`id_vars`和`value_vars`参数，指定`originalid`作为`ID`变量，并将`weeksworked##`列作为要旋转或熔化的列。使用`var_name`和`value_name`参数将列名重命名为`year`和`weeksworked`。`value_vars`中的列名成为新`year`列的值（我们使用原始后缀将其转换为整数）。`value_vars`列的值被分配到新`weeksworked`列中的相应行：

```py
weeksworked = nls97.reset_index().\
  loc[:,['originalid'] + weeksworkedcols].\
  melt(id_vars=['originalid'],
  value_vars=weeksworkedcols,
  var_name='year', value_name='weeksworked')
weeksworked['year'] = \
  weeksworked.year.str[-2:].astype(int)+2000
weeksworked.set_index(['originalid'], inplace=True)
weeksworked.loc[[2,3]] 
```

```py
 year       weeksworked
originalid                       
2                2017                52
2                2018                52
2                2019                52
2                2020                52
2                2021                46
3                2017                52
3                2018                52
3                2019                 9
3                2020                 0
3                2021                 0 
```

1.  使用`melt`重塑大学入学列。

这与`melt`函数在处理工作周数列时的作用相同：

```py
colenrcols = \
  ['colenroct17','colenroct18','colenroct19',
  'colenroct20','colenroct21']
colenr = nls97.reset_index().\
  loc[:,['originalid'] + colenrcols].\
  melt(id_vars=['originalid'], value_vars=colenrcols,
var_name='year', value_name='colenr')
colenr['year'] = colenr.year.str[-2:].astype(int)+2000
colenr.set_index(['originalid'], inplace=True)
colenr.loc[[2,3]] 
```

```py
 year                     colenr
originalid                                
2                2017            1\. Not enrolled
2                2018            1\. Not enrolled
2                2019            1\. Not enrolled
2                2020            1\. Not enrolled
2                2021            1\. Not enrolled
3                2017            1\. Not enrolled
3                2018            1\. Not enrolled
3                2019            1\. Not enrolled
3                2020            1\. Not enrolled
3                2021            1\. Not enrolled 
```

1.  合并工作周数和大学入学数据：

    ```py
    workschool = \
      pd.merge(weeksworked, colenr, on=['originalid','year'], how="inner")
    workschool.shape 
    ```

    ```py
    (44920, 3) 
    ```

    ```py
    workschool.loc[[2,3]] 
    ```

    ```py
     year       weeksworked                colenr
    originalid                                        
    2           2017                52       1\. Not enrolled
    2           2018                52       1\. Not enrolled
    2           2019                52       1\. Not enrolled
    2           2020                52       1\. Not enrolled
    2           2021                46       1\. Not enrolled
    3           2017                52       1\. Not enrolled
    3           2018                52       1\. Not enrolled
    3           2019                 9       1\. Not enrolled
    3           2020                 0       1\. Not enrolled
    3           2021                 0       1\. Not enrolled 
    ```

这将通过熔化工作周数和大学入学列，生成一个 DataFrame。

## 它是如何工作的…

我们可以使用`stack`或`melt`将数据从宽格式重塑为长格式，但`melt`提供了更多的灵活性。`stack`会将所有列名移动到索引中。我们在*第 4 步*中看到，堆叠后得到了预期的行数`44920`，这等于 5*8984，即初始数据中的行数。

使用`melt`，我们可以根据不同于索引的`ID`变量旋转列名和值。我们通过`id_vars`参数来实现这一点。我们使用`value_vars`参数指定要旋转的变量。

在 *步骤 6* 中，我们还重新塑造了大学入学的列。为了将重新塑造后的工作周和大学入学数据合并为一个 DataFrame，我们合并了 *步骤 5* 和 *步骤 6* 中创建的两个 DataFrame。我们将在下一个配方中看到如何一步完成 *步骤 5* 到 *步骤 7* 的工作。

# 融合多个列组

在前一个配方中，当我们需要融合多个列组时，我们使用了两次 `melt` 然后合并了结果的 DataFrame。那样也可以，但我们可以用 `wide_to_long` 函数在一步内完成相同的任务。`wide_to_long` 的功能比 `melt` 强大，但使用起来稍微复杂一些。

## 准备工作

我们将在本配方中使用 NLS 的工作周和大学入学数据。

## 如何操作...

我们将使用 `wide_to_long` 一次性转换多个列组：

1.  导入 `pandas` 并加载 NLS 数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index('personid', inplace=True) 
    ```

1.  查看一些工作周和大学入学的数据：

    ```py
    weeksworkedcols = ['weeksworked17','weeksworked18',
      'weeksworked19','weeksworked20','weeksworked21']
    colenrcols = ['colenroct17','colenroct18',
       'colenroct19','colenroct20','colenroct21']
    nls97.loc[nls97.originalid.isin([2,3]),
      ['originalid'] + weeksworkedcols + colenrcols].T 
    ```

    ```py
    personid                     999406                151672
    originalid                        2                     3
    weeksworked17                    52                    52
    weeksworked18                    52                    52
    weeksworked19                    52                     9
    weeksworked20                    52                     0
    weeksworked21                    46                     0
    colenroct17         1\. Not enrolled       1\. Not enrolled
    colenroct18         1\. Not enrolled       1\. Not enrolled
    colenroct19         1\. Not enrolled       1\. Not enrolled
    colenroct20         1\. Not enrolled       1\. Not enrolled
    colenroct21         1\. Not enrolled       1\. Not enrolled 
    ```

1.  运行 `wide_to_long` 函数。

将一个列表传递给 `stubnames` 以指示所需的列组。（所有列名以列表中每一项的相同字符开头的列都会被选中进行转换。）使用 `i` 参数指示 ID 变量（`originalid`），并使用 `j` 参数指定基于列后缀（如 `17`、`18` 等）命名的列（`year`）：

```py
workschool = pd.wide_to_long(nls97[['originalid']
...   + weeksworkedcols + colenrcols],
...   stubnames=['weeksworked','colenroct'],
...   i=['originalid'], j='year').reset_index()
workschool['year'] = workschool.year+2000
workschool = workschool.\
...   sort_values(['originalid','year'])
workschool.set_index(['originalid'], inplace=True)
workschool.loc[[2,3]] 
```

```py
 year       weeksworked             colenroct
originalid                                        
2                2017                52       1\. Not enrolled
2                2018                52       1\. Not enrolled
2                2019                52       1\. Not enrolled
2                2020                52       1\. Not enrolled
2                2021                46       1\. Not enrolled
3                2017                52       1\. Not enrolled
3                2018                52       1\. Not enrolled
3                2019                 9       1\. Not enrolled
3                2020                 0       1\. Not enrolled
3                2021                 0       1\. Not enrolled 
```

`wide_to_long` 一步完成了我们在前一个配方中使用 `melt` 需要多个步骤才能完成的工作。

## 工作原理...

`wide_to_long` 函数几乎为我们完成了所有工作，尽管它的设置比 `stack` 或 `melt` 要复杂一些。我们需要向函数提供列组的字符（在这个例子中是 `weeksworked` 和 `colenroct`）。由于我们的变量名称带有表示年份的后缀，`wide_to_long` 会将这些后缀转换为有意义的值，并将它们融合到用 `j` 参数命名的列中。这几乎就像魔法一样！

## 还有更多...

本配方中 `stubnames` 列的后缀是相同的：17 到 21。但这不一定是必然的。当某个列组有后缀，而另一个没有时，后者列组对应后缀的值将会缺失。通过排除 DataFrame 中的 `weeksworked17` 并添加 `weeksworked16`，我们可以看到这一点：

```py
weeksworkedcols = ['weeksworked16','weeksworked18',
  'weeksworked19','weeksworked20','weeksworked21']
workschool = pd.wide_to_long(nls97[['originalid']
...   + weeksworkedcols + colenrcols],
...   stubnames=['weeksworked','colenroct'],
...   i=['originalid'], j='year').reset_index()
workschool['year'] = workschool.year+2000
workschool = workschool.sort_values(['originalid','year'])
workschool.set_index(['originalid'], inplace=True)
workschool.loc[[2,3]] 
```

```py
 year       weeksworked             colenroct
originalid                                                  
2                2016                53                   NaN
2                2017               NaN       1\. Not enrolled
2                2018                52       1\. Not enrolled
2                2019                52       1\. Not enrolled
2                2020                52       1\. Not enrolled
2                2021                46       1\. Not enrolled
3                2016                53                   NaN
3                2017               NaN       1\. Not enrolled
3                2018                52       1\. Not enrolled
3                2019                 9       1\. Not enrolled
3                2020                 0       1\. Not enrolled
3                2021                 0       1\. Not enrolled 
```

现在，2017 年的 `weeksworked` 值缺失了，2016 年的 `colenroct` 值也缺失了。

# 使用 `unstack` 和 `pivot` 将数据从长格式转换为宽格式

有时候，我们实际上需要将数据从整洁格式转换为杂乱格式。这通常是因为我们需要将数据准备为某些不擅长处理关系型数据的软件包分析，或者因为我们需要提交数据给某个外部机构，而对方要求以杂乱格式提供数据。`unstack` 和 `pivot` 在需要将数据从长格式转换为宽格式时非常有用。`unstack` 做的是与我们使用 `stack` 的操作相反的事，而 `pivot` 做的则是与 `melt` 相反的操作。

## 准备工作

我们在本食谱中继续处理关于工作周数和大学入学的 NLS 数据。

## 如何操作……

我们使用`unstack`和`pivot`将融化的 NLS 数据框恢复到其原始状态：

1.  导入`pandas`并加载堆叠和融化后的 NLS 数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97g.csv", low_memory=False)
    nls97.set_index(['originalid'], inplace=True) 
    ```

1.  再次堆叠数据。

这重复了本章早期食谱中的堆叠操作：

```py
weeksworkedcols = ['weeksworked17','weeksworked18',
  'weeksworked19','weeksworked20','weeksworked21']
weeksworkedstacked = nls97[weeksworkedcols].\
  stack()
weeksworkedstacked.loc[[2,3]] 
```

```py
originalid              
2           weeksworked17        52
            weeksworked18        52
            weeksworked19        52
            weeksworked20        52
            weeksworked21        46
3           weeksworked17        52
            weeksworked18        52
            weeksworked19         9
            weeksworked20         0
            weeksworked21         0
dtype: float64 
```

1.  再次融化数据。

这重复了本章早期食谱中的`melt`操作：

```py
weeksworkedmelted = nls97.reset_index().\
...   loc[:,['originalid'] + weeksworkedcols].\
...   melt(id_vars=['originalid'],
...   value_vars=weeksworkedcols,
...   var_name='year', value_name='weeksworked')
weeksworkedmelted.loc[weeksworkedmelted.\
  originalid.isin([2,3])].\
  sort_values(['originalid','year']) 
```

```py
 originalid                year       weeksworked
1                    2       weeksworked17                52
8985                 2       weeksworked18                52
17969                2       weeksworked19                52
26953                2       weeksworked20                52
35937                2       weeksworked21                46
2                    3       weeksworked17                52
8986                 3       weeksworked18                52
17970                3       weeksworked19                 9
26954                3       weeksworked20                 0
35938                3       weeksworked21                 0 
```

1.  使用`unstack`将堆叠的数据从长格式转换为宽格式：

    ```py
    weeksworked = weeksworkedstacked.unstack()
    weeksworked.loc[[2,3]].T 
    ```

    ```py
    originalid          2       3
    weeksworked17      52      52
    weeksworked18      52      52
    weeksworked19      52       9
    weeksworked20      52       0
    weeksworked21      46       0 
    ```

1.  使用`pivot`将融化的数据从长格式转换为宽格式。

`pivot`比`unstack`稍微复杂一点。我们需要传递参数来执行 melt 的反向操作，告诉 pivot 使用哪个列作为列名后缀（`year`），并从哪里获取要取消融化的值（在本例中来自`weeksworked`列）：

```py
weeksworked = weeksworkedmelted.\
...   pivot(index='originalid',
...   columns='year', values=['weeksworked']).\
...   reset_index()
weeksworked.columns = ['originalid'] + \
...   [col[1] for col in weeksworked.columns[1:]]
weeksworked.loc[weeksworked.originalid.isin([2,3])].T 
```

```py
 1       2
originalid          2       3
weeksworked17      52      52
weeksworked18      52      52
weeksworked19      52       9
weeksworked20      52       0
weeksworked21      46       0 
```

这将 NLS 数据返回到其原始的无序形式。

## 它是如何工作的……

我们首先在*步骤 2*和*步骤 3*分别执行`stack`和`melt`。这将数据框从宽格式转换为长格式。然后我们使用`unstack`（*步骤 4*）和`pivot`（*步骤 5*）将数据框从长格式转换回宽格式。

`unstack`使用由`stack`创建的多重索引来确定如何旋转数据。

`pivot`函数需要我们指定索引列（`originalid`），将附加到列名中的列（`year`），以及包含要取消融化值的列名称（`weeksworked`）。`pivot`将返回多级列名。我们通过从第二级提取`[col[1] for col in weeksworked.columns[1:]]`来修复这个问题。

# 总结

本章中我们探讨了关键的 tidy 数据主题。这些主题包括处理重复数据，可以通过删除冗余数据的行或按组聚合来处理。我们还将以多对多格式存储的数据重构为 tidy 格式。最后，我们介绍了将数据从宽格式转换为长格式的几种方法，并在必要时将其转换回宽格式。接下来是本书的最后一章，我们将学习如何使用用户定义的函数、类和管道来自动化数据清理。

# 加入我们的社区，参与 Discord 讨论

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`discord.gg/p8uSgEAETX`](https://discord.gg/p8uSgEAETX)

![](img/QR_Code10336218961138498953.png)
