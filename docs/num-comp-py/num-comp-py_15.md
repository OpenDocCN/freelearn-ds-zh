# 第十五章：合并 Pandas 对象

有多种选项可以将两个或多个 DataFrame 或 Series 组合在一起。`append`方法是最不灵活的，它只允许向 DataFrame 添加新行。`concat`方法非常灵活，可以在任意轴上合并任意数量的 DataFrame 或 Series。`join`方法通过将一个 DataFrame 的列与其他 DataFrame 的索引对齐来提供快速查找。`merge`方法提供类似 SQL 的功能，用于将两个 DataFrame 合并。

在本章中，我们将涵盖以下主题：

+   向 DataFrame 追加新行

+   合并多个 DataFrame

+   比较特朗普总统和奥巴马总统的支持率

+   理解`concat`、`join`和`merge`之间的区别

+   连接到 SQL 数据库

# 向 DataFrame 追加新行

在进行数据分析时，创建新列比创建新行更为常见。这是因为新的一行数据通常代表一个新的观察结果，而作为分析师，通常不是你负责不断捕获新数据。数据捕获通常由其他平台，如关系数据库管理系统，来处理。然而，作为一项必要功能，你仍然需要掌握它，因为它有时会用到。

# 准备就绪

在本例中，我们将首先使用`.loc`索引器将行添加到一个小数据集中，然后过渡到使用`append`方法。

# 如何操作...

1.  阅读名字数据集，并输出：

```py
>>> names = pd.read_csv('data/names.csv')
>>> names
```

![](img/4954144c-e882-4317-8aea-1ed33e15dc93.png)

1.  让我们创建一个包含一些新数据的列表，并使用`.loc`索引器将一个行标签设置为这些新数据：

```py
>>> new_data_list = ['Aria', 1]
>>> names.loc[4] = new_data_list
>>> names
```

![](img/1da729da-b2ad-4cb5-abc6-aa45ba1f1c00.png)

1.  `.loc`索引器使用标签来引用行。在这种情况下，行标签完全匹配整数位置。也可以使用非整数标签追加更多行：

```py
>>> names.loc['five'] = ['Zach', 3]
>>> names
```

![](img/2a083be8-7f33-46f1-a95f-316aaa26f946.png)

1.  为了更明确地将变量与值关联，你可以使用字典。此外，在这一步中，我们可以动态选择新的索引标签为 DataFrame 的长度：

```py
>>> names.loc[len(names)] = {'Name':'Zayd', 'Age':2}
>>> names
```

![](img/8ede39cb-b883-4529-9504-08a6509b4762.png)

1.  一个 Series 也可以保存新的数据，并且与字典完全相同：

```py
>>> names.loc[len(names)] = pd.Series({'Age':32,
                                       'Name':'Dean'})
>>> names
```

![](img/7a5690e9-355e-44db-936a-2513551b28ec.png)

1.  上述操作都使用`.loc`索引运算符对`names` DataFrame 进行原地修改。没有返回 DataFrame 的副本。在接下来的几个步骤中，我们将了解`append`方法，它不会修改调用的 DataFrame。相反，它会返回一个新的 DataFrame 副本，附加了行。让我们从原始的`names` DataFrame 开始，尝试追加一行。`append`的第一个参数必须是另一个 DataFrame、Series、字典或这些的列表，而不是像步骤 2 中的列表。让我们看看当我们尝试用字典和`append`一起使用时会发生什么：

```py
>>> names = pd.read_csv('data/names.csv')
>>> names.append({'Name':'Aria', 'Age':1})
TypeError: Can only append a Series if ignore_index=True or if the Series has a name
```

1.  这个错误信息看起来稍有不准确。我们传递的是一个 DataFrame 而不是 Series，但它仍然给出了如何修正的指示：

```py
>>> names.append({'Name':'Aria', 'Age':1}, ignore_index=True)
```

![](img/d1209a67-6e9c-400b-878f-3b4c6454fa05.png)

1.  这虽然可行，但 `ignore_index` 是一个狡猾的参数。当设置为 `True` 时，旧的索引将被完全移除，并被从 0 到 n-1 的 `RangeIndex` 所替代。例如，下面我们为 `names` DataFrame 指定一个索引：

```py
>>> names.index = ['Canada', 'Canada', 'USA', 'USA']
>>> names
```

![](img/5ce47674-b461-431f-ac0e-0deb1442f0f1.png)

1.  重新运行步骤 7 中的代码，你将得到相同的结果。原始索引会被完全忽略。

1.  让我们继续使用这个 `names` 数据集，其中索引是这些国家字符串，并使用一个带有 `name` 属性的 Series 来使用 `append` 方法：

```py
>>> s = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
>>> s
Age        3
Name    Zach
Name: 4, dtype: object

>>> names.append(s)
```

![](img/c54f0095-e1c3-4527-b793-a90c3efa3aa3.png)

1.  `append` 方法比 `.loc` 索引器更灵活。它支持同时附加多行数据。实现这一点的一种方式是使用 Series 列表：

```py
>>> s1 = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
>>> s2 = pd.Series({'Name': 'Zayd', 'Age': 2}, name='USA')
>>> names.append([s1, s2])
```

![](img/49370d22-759a-4752-8eff-b1253447829e.png)

1.  只有两列的小型 DataFrame 足够简单，可以手动写出所有的列名和值。当它们变大时，这个过程将变得相当痛苦。例如，来看一下 2016 年的棒球数据集：

```py
>>> bball_16 = pd.read_csv('data/baseball16.csv')
>>> bball_16.head()
```

![](img/5c669c06-e152-40f1-a4ce-af3f4f91803a.png)

1.  这个数据集包含 22 列，如果你手动输入新行数据时，很容易输入错误的列名或者完全忘记某些列名。为了避免这些错误，下面我们选择一行作为 Series，并链式调用 `to_dict` 方法，将这一行作为字典提取出来：

```py
>>> data_dict = bball_16.iloc[0].to_dict()
>>> print(data_dict)
{'playerID': 'altuvjo01', 'yearID': 2016, 'stint': 1, 'teamID': 'HOU', 'lgID': 'AL', 'G': 161, 'AB': 640, 'R': 108, 'H': 216, '2B': 42, '3B': 5, 'HR': 24, 'RBI': 96.0, 'SB': 30.0, 'CS': 10.0, 'BB': 60, 'SO': 70.0, 'IBB': 11.0, 'HBP': 7.0, 'SH': 3.0, 'SF': 7.0, 'GIDP': 15.0}
```

1.  使用字典推导式清除旧值，将任何之前的字符串值设为空字符串，其他所有的值设为缺失值。这个字典现在可以作为你想输入任何新数据的模板：

```py
>>> new_data_dict = {k: '' if isinstance(v, str) else 
                        np.nan for k, v in data_dict.items()}
>>> print(new_data_dict)
{'playerID': '', 'yearID': nan, 'stint': nan, 'teamID': '', 'lgID': '', 'G': nan, 'AB': nan, 'R': nan, 'H': nan, '2B': nan, '3B': nan, 'HR': nan, 'RBI': nan, 'SB': nan, 'CS': nan, 'BB': nan, 'SO': nan, 'IBB': nan, 'HBP': nan, 'SH': nan, 'SF': nan, 'GIDP': nan}
```

# 它是如何工作的...

`.loc` 索引操作符用于根据行和列标签选择并分配数据。传递给它的第一个值代表行标签。在步骤 2 中，`names.loc[4]` 指的是标签等于整数 4 的那一行。这个标签在 DataFrame 中当前并不存在。赋值语句通过列表提供的数据创建了一行新的数据。正如食谱中所提到的，这个操作会修改 `names` DataFrame 本身。如果之前存在标签等于整数 4 的行，这个命令会覆盖它。就地修改使得这个索引操作符比 `append` 方法更具风险，后者永远不会修改原始的调用 DataFrame。

任何有效的标签都可以与 `.loc` 索引操作符一起使用，如步骤 3 所示。不管新的标签值是什么，新的行总是会附加到最后。即使用列表赋值有效，为了清晰起见，最好使用字典，这样我们可以清楚地知道每个值与哪些列关联，就像在步骤 4 中所做的那样。

第 5 步展示了一个小技巧，可以动态地将新标签设置为 DataFrame 当前的行数。存储在 Series 中的数据也会被正确分配，只要索引标签与列名匹配。

剩下的步骤使用 `append` 方法，这是一种简单的方法，只能将新行附加到 DataFrame 中。大多数 DataFrame 方法都可以通过 `axis` 参数同时处理行和列。唯一的例外是 `append`，它只能向 DataFrame 添加行。

使用列名映射到值的字典不足以使 append 操作生效，正如第 6 步的错误信息所示。要正确地附加没有行名的字典，你必须将 `ignore_index` 参数设置为 `True`。第 10 步展示了如何通过将字典转换为 Series 来保留原始索引。确保使用 `name` 参数，这将作为新的索引标签。在这种方式下，通过将 Series 列表作为第一个参数传递，可以添加任意数量的行。

当你想要以这种方式附加大量的行到一个 DataFrame 时，你可以通过将单行转换为字典，并使用字典推导式清除所有旧值，将它们替换为一些默认值，从而避免大量的输入错误。

# 还有更多...

向 DataFrame 中添加单行是一个相当昂贵的操作，如果你发现自己正在写循环逐行添加数据到 DataFrame，那么你做错了。首先，我们创建 1,000 行新数据作为 Series 列表：

```py
>>> random_data = []
>>> for i in range(1000):
        d = dict()
        for k, v in data_dict.items():
            if isinstance(v, str):
                d[k] = np.random.choice(list('abcde'))
            else:
                d[k] = np.random.randint(10)
        random_data.append(pd.Series(d, name=i + len(bball_16)))

>>> random_data[0].head()
2B    3
3B    9
AB    3
BB    9
CS    4
Name: 16, dtype: object
```

让我们来计时，看看循环通过每一项并逐行附加需要多少时间：

```py
>>> %%timeit
>>> bball_16_copy = bball_16.copy()
>>> for row in random_data:
        bball_16_copy = bball_16_copy.append(row)
4.88 s ± 190 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

仅 1,000 行数据就花了近五秒钟。如果我们改为传递整个 Series 列表，速度会大大提高：

```py
>>> %%timeit
>>> bball_16_copy = bball_16.copy()
>>> bball_16_copy = bball_16_copy.append(random_data)
78.4 ms ± 6.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

通过传递 Series 列表，时间被减少到了不到一秒的十分之一。Pandas 内部将 Series 列表转换为一个单一的 DataFrame，然后进行 append 操作。

# 将多个 DataFrame 连接在一起

多功能的 `concat` 函数可以将两个或更多的 DataFrame（或 Series）垂直或水平地连接在一起。通常，处理多个 pandas 对象时，连接不会随意发生，而是根据索引对齐每个对象。

# 准备好

在这个示例中，我们使用 `concat` 函数水平和垂直地合并 DataFrame，然后通过更改参数值来得到不同的结果。

# 如何做到这一点...

1.  读取 2016 年和 2017 年的股票数据集，并将它们的股票代码设置为索引：

```py
>>> stocks_2016 = pd.read_csv('data/stocks_2016.csv', 
                              index_col='Symbol')
>>> stocks_2017 = pd.read_csv('data/stocks_2017.csv',
                              index_col='Symbol')
```

![](img/78a928de-3aa5-444b-b6c3-8f74e590a2ca.png)    ![](img/82f29089-c05d-4bf1-8208-be3212c8b259.png)

1.  将所有 `stock` 数据集放入一个列表中，然后调用 `concat` 函数将它们连接在一起：

```py
>>> s_list = [stocks_2016, stocks_2017]
>>> pd.concat(s_list)
```

![](img/d273561a-b8ee-4400-84c6-6fe99b2870ec.png)

1.  默认情况下，`concat`函数将 DataFrame 垂直连接，将其一一叠加。前述的 DataFrame 存在一个问题，就是无法识别每一行的年份。`concat`函数允许通过`keys`参数为结果 DataFrame 中的每个部分加上标签。这个标签会出现在连接后的框架的最外层索引级别，并强制创建一个多重索引。此外，`names`参数可以重命名每个索引级别，以便更清晰地显示：

```py
>>> pd.concat(s_list, keys=['2016', '2017'], 
              names=['Year', 'Symbol'])
```

![](img/010d420f-feba-4c21-9d7c-3f60a806cf2e.png)

1.  也可以通过将`axis`参数设置为*columns*或*1*来进行水平连接：

```py
>>> pd.concat(s_list, keys=['2016', '2017'],
              axis='columns', names=['Year', None])
```

![](img/6d248ec3-bfb5-496c-96ae-2b5d267e5c77.png)

1.  注意，每当某个股票代码在某一年出现而在另一年没有出现时，缺失值会出现。默认情况下，`concat`函数使用外连接，保留列表中每个 DataFrame 的所有行。然而，它也提供了仅保留两个 DataFrame 中索引值相同的行的选项，这被称为内连接。我们可以将`join`参数设置为*inner*来改变这一行为：

```py
>>> pd.concat(s_list, join='inner', keys=['2016', '2017'],
              axis='columns', names=['Year', None])
```

![](img/4740c498-d459-403c-bd63-17d7554797d5.png)

# 它是如何工作的...

第一个参数是`concat`函数唯一需要的参数，必须是一个包含 pandas 对象的序列，通常是一个 DataFrame 或 Series 的列表或字典。默认情况下，所有这些对象会被垂直堆叠在一起。在这个例子中，仅连接了两个 DataFrame，但任何数量的 pandas 对象都可以工作。当我们进行垂直连接时，DataFrame 会根据列名对齐。

在这个数据集中，所有列名相同，因此 2017 年的每一列都准确地与 2016 年对应的列对齐。然而，当它们像第 4 步那样进行水平连接时，只有两个索引标签在两年间匹配——*AAPL*和*TSLA*。因此，这些股票代码在两年之间都没有缺失值。使用`concat`可以进行两种类型的对齐，分别是*outer*（默认）和*inner*，由`join`参数指定。

# 还有更多内容...

`append`方法是`concat`的简化版本，仅能将新行追加到 DataFrame 中。内部实现上，`append`只是调用了`concat`函数。例如，这个示例中的第二步可以通过以下方式实现：

```py
>>> stocks_2016.append(stocks_2017)
```

# 比较特朗普和奥巴马的支持率

美国现任总统的公众支持度是一个经常出现在新闻头条的话题，并通过民意调查正式测量。近年来，这些民意调查的频率急剧增加，每周都会有大量的新数据发布。不同的民意调查公司有不同的问题和方法来收集数据，因此数据之间存在相当大的差异。加利福尼亚大学圣巴巴拉分校的美国总统项目提供了一个综合的支持度数据，每天更新一次。

与本书中的大多数示例不同，这些数据并不是直接以 CSV 文件的形式提供的。作为数据分析师，你通常需要在网上寻找数据，并使用一个能够抓取这些数据并转化为你能够在本地工作站解析的格式的工具。

# 准备就绪

在本节中，我们将使用 `read_html` 函数，它具有强大的功能，可以从网上的表格中抓取数据并将其转化为 DataFrame。你还将学习如何检查网页，找到某些元素的底层 HTML。我使用的是 Google Chrome 浏览器，建议你也使用它，或者 Firefox 浏览器，进行网页相关的操作。

# 如何操作...

1.  访问 *美国总统项目* 中关于总统唐纳德·特朗普的支持度页面（[`www.presidency.ucsb.edu/data/popularity.php?pres=45`](http://www.presidency.ucsb.edu/data/popularity.php?pres=45)）。你应该会看到一个包含时间序列图的数据页面，紧跟其后的是一个表格：

![](img/aa1d10d8-b146-408e-a53e-c0949b87086e.png)

1.  `read_html` 函数能够从网页中抓取表格并将其数据放入 DataFrame。它在处理简单的 HTML 表格时效果最佳，并提供了一些有用的参数，帮助你在同一页面有多个表格时选择你想要的具体表格。我们可以直接使用 `read_html` 的默认值，这将返回所有表格作为 DataFrame 的列表：

```py
>>> base_url = 'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
>>> trump_url = base_url.format(45)
>>> df_list = pd.read_html(trump_url)
>>> len(df_list)
14
```

1.  该函数返回了 14 个表格，乍一看似乎很不可思议，因为网页上似乎只显示了一个大家普遍认为是表格的元素。`read_html` 函数正式搜索以 *<table* 开头的 HTML 表格元素。让我们通过右键点击批准数据表格并选择检查或检查元素来检查 HTML 页面：

![](img/e0ac10b2-f06d-41b7-8812-e98ff764ef59.png)

1.  这会打开控制台，这是一个非常强大的网页开发工具。在本节中，我们只需要它来完成几个任务。所有控制台都允许你在 HTML 中搜索特定的词汇。让我们搜索 `table` 这个词。我的浏览器找到了 15 个不同的 HTML 表格，与 `read_html` 返回的数量非常接近：

![](img/fa149f24-7f39-49b5-9ea0-17bb8a32f54c.png)

1.  让我们开始检查 `df_list` 中的 DataFrame：

```py
>>> df0 = df_list[0]
>>> df0.shape
(308, 1794)

>>> df0.head(7)
```

![](img/f2c5a7e0-0994-47a1-89a9-dc3872414d23.png)

1.  回顾网页，从 2017 年 1 月 22 日开始，直到数据抓取的日期——2017 年 9 月 25 日，审批表中几乎每天都有一行数据。大约是八个月或 250 行数据，接近第一个表格中的 308 行。浏览其余的表格时，你会发现很多空的无意义的表格，以及一些实际上不像表格的网页其他部分。让我们使用`read_html`函数的一些参数，帮助我们选择我们想要的表格。我们可以使用`match`参数搜索表格中的特定字符串。让我们搜索包含*Start Date*的表格：

```py
>>> df_list = pd.read_html(trump_url, match='Start Date')
>>> len(df_list)
3
```

1.  通过在表格中搜索特定的字符串，我们将表格的数量减少到了只有三个。另一个有用的参数是`attrs`，它接受一组 HTML 属性及其对应的值的字典。我们希望为我们的特定表格找到一些独特的属性。为此，让我们再次右键点击数据表。这次，确保点击在表头的最上方。例如，右键点击*President*，然后再次选择“检查”或“检查元素”：

![](img/8453c5fd-4db1-4cb7-9373-7c92d335c3ce.png)

1.  你选择的元素应该被高亮显示。但这实际上不是我们感兴趣的元素。继续查找，直到你遇到一个以*<table*开头的 HTML 标签。等号左侧的所有词是属性或`attrs`，右侧是它们的值。让我们在搜索时使用*align*属性，其值为*center*：

```py
>>> df_list = pd.read_html(trump_url, match='Start Date',
                           attrs={'align':'center'})
>>> len(df_list)
1

>>> trump = df_list[0]
>>> trump.shape
(249, 19)

>>> trump.head(8)
```

![](img/42221dad-03c0-4e9b-9bf7-2e6aec18e4d6.png)

1.  我们只匹配了一个表格，而且行数非常接近第一个和最后一个日期之间的总天数。查看数据后，似乎我们确实找到了我们想要的表格。六个列名似乎出现在第 4 行。我们可以更进一步，精确选择我们想跳过的行以及我们想用作列名的行，使用`skiprows`和`header`参数。我们还可以使用`parse_dates`参数确保开始日期和结束日期被正确转换为合适的数据类型：

```py
>>> df_list = pd.read_html(trump_url, match='Start Date',
                           attrs={'align':'center'}, 
                           header=0, skiprows=[0,1,2,3,5], 
                           parse_dates=['Start Date',
                                        'End Date'])
>>> trump = df_list[0]
>>> trump.head()
```

![](img/99e47525-f4c0-44cf-84be-62eceaa334de.png)

1.  这几乎正是我们想要的，除了那些有缺失值的列。我们可以使用`dropna`方法删除所有值缺失的列：

```py
>>> trump = trump.dropna(axis=1, how='all')
>>> trump.head()
```

![](img/907ada91-650c-4666-8dac-f7347b19b9fe.png)

1.  我们将使用`ffill`方法按顺序填充`President`列中的缺失值。让我们先检查其他列中是否有缺失值：

```py
>>> trump.isnull().sum()
President         242
Start Date          0
End Date            0
Approving           0
Disapproving        0
unsure/no data      0
dtype: int64

>>> trump = trump.ffill()
trump.head()
```

![](img/da9fc9fb-8b2e-4221-893d-105d8bca02d1.png)

1.  最后，检查数据类型是否正确非常重要：

```py
>>> trump.dtypes
President                 object
Start Date        datetime64[ns]
End Date          datetime64[ns]
Approving                  int64
Disapproving               int64
unsure/no data             int64
dtype: object
```

1.  让我们创建一个函数，将所有步骤合并为一个，自动化获取任何总统的审批数据的过程：

```py
>>> def get_pres_appr(pres_num):
        base_url =\
'http://www.presidency.ucsb.edu/data/popularity.php?pres={}'
        pres_url = base_url.format(pres_num)
        df_list = pd.read_html(pres_url, match='Start Date',
                               attrs={'align':'center'}, 
                               header=0, skiprows=[0,1,2,3,5], 
                               parse_dates=['Start Date',
                                            'End Date'])
        pres = df_list[0].copy()
        pres = pres.dropna(axis=1, how='all')
        pres['President'] = pres['President'].ffill()
        return pres.sort_values('End Date') \
                   .reset_index(drop=True)
```

1.  唯一的参数`pres_num`表示每位总统的顺序编号。巴拉克·奥巴马是美国的第 44 任总统；传递 44 给`get_pres_appr`函数以获取他的支持率：

```py
>>> obama = get_pres_appr(44)
>>> obama.head()
```

![](img/6247a2a4-700b-48c1-97ee-065dee3a30d6.png)

1.  我们可以追溯到 1941 年总统富兰克林·罗斯福的第三任期的总统支持率数据。通过自定义函数和`concat`函数的结合，我们可以从该网站获取所有总统的支持率数据。现在，我们只获取过去五任总统的支持率数据，并输出每位总统的前三行数据：

```py
>>> pres_41_45 = pd.concat([get_pres_appr(x) for x in range(41,46)],
                            ignore_index=True)
>>> pres_41_45.groupby('President').head(3)
```

![](img/b5510c6f-12c7-4faa-8711-6c8ece33933d.png)

1.  在继续之前，让我们确定是否有任何日期对应多个支持率：

```py
>>> pres_41_45['End Date'].value_counts().head(8)
1990-08-26    2
1990-03-11    2
1999-02-09    2
2013-10-10    2
1990-08-12    2
1992-11-22    2
1990-05-22    2
1991-09-30    1
Name: End Date, dtype: int64
```

1.  只有少数几天有重复值。为了简化分析，我们只保留重复日期的第一行：

```py
>>> pres_41_45 = pres_41_45.drop_duplicates(subset='End Date')
```

1.  让我们获取一些数据的摘要统计：

```py
>>> pres_41_45.shape
(3679, 6)

>>> pres_41_45['President'].value_counts()
Barack Obama          2786
George W. Bush         270
Donald J. Trump        243
William J. Clinton     227
George Bush            153
Name: President, dtype: int64

>>> pres_41_45.groupby('President', sort=False) \
                       .median().round(1)
```

![](img/118778a4-f5d5-4d65-a816-2a35a1024802.png)

1.  让我们将每位总统的支持率绘制在同一张图表上。为此，我们将按每位总统分组，遍历每个组，并单独绘制每个日期的支持率：

```py
>>> from matplotlib import cm
>>> fig, ax = plt.subplots(figsize=(16,6))

>>> styles = ['-.', '-', ':', '-', ':']
>>> colors = [.9, .3, .7, .3, .9]
>>> groups = pres_41_45.groupby('President', sort=False)

>>> for style, color, (pres, df) in zip(styles, colors, groups):
        df.plot('End Date', 'Approving', ax=ax,
                label=pres, style=style, color=cm.Greys(color), 
                title='Presedential Approval Rating')
```

![](img/8b6bbb45-0f79-4cbd-95f5-e47498c98c9e.png)

1.  这张图表将所有总统按顺序排列。我们可以通过将他们的支持率与在任天数进行对比，简化比较。让我们创建一个新的变量来表示在任天数：

```py
>>> days_func = lambda x: x - x.iloc[0]
>>> pres_41_45['Days in Office'] = pres_41_45.groupby('President') \
                                             ['End Date'] \
                                             .transform(days_func)
>>> pres_41_45.groupby('President').head(3)
```

![](img/b63c31b6-e297-40e4-b17a-49bddb24e807.png)

1.  我们已经成功地给每一行添加了自总统任期开始以来的相对天数。有趣的是，新列`Days in Office`的值是以字符串形式表示的。让我们检查一下它的数据类型：

```py
>>> pres_41_45.dtypes
...
Days in Office    timedelta64[ns]
dtype: object
```

1.  `Days in Office`列是一个`timedelta64`对象，精度为纳秒。这远远超过了所需的精度。让我们通过只取天数，将数据类型更改为整数：

```py
>>> pres_41_45['Days in Office'] = pres_41_45['Days in Office'] \
                                             .dt.days
>>> pres_41_45['Days in Office'].head()
0     0
1    32
2    35
3    43
4    46
Name: Days in Office, dtype: int64
```

1.  我们可以像第 19 步中那样绘制这组数据，但还有一种完全不同的方法，它不涉及任何循环。默认情况下，当在 DataFrame 上调用`plot`方法时，pandas 会尝试将每一列数据作为折线图绘制，并使用索引作为 x 轴。了解这一点后，让我们将数据透视，以便每位总统都有自己的一列支持率数据：

```py
>>> pres_pivot = pres_41_45.pivot(index='Days in Office',
                                  columns='President',
                                  values='Approving')
>>> pres_pivot.head()
```

![](img/16d28567-1ef0-412b-9a27-918e22994808.png)

1.  现在每位总统都有自己的一列支持率数据，我们可以直接绘制每一列，而无需分组。为了减少图表中的杂乱，我们将仅绘制巴拉克·奥巴马和唐纳德·J·特朗普的数据：

```py
>>> plot_kwargs = dict(figsize=(16,6), color=cm.gray([.3, .7]), 
                       style=['-', '--'], title='Approval Rating')
>>> pres_pivot.loc[:250, ['Donald J. Trump', 'Barack Obama']] \
              .ffill().plot(**plot_kwargs)
```

![](img/82e84686-c9f1-4e28-bdfc-393437ce7825.png)

# 如何工作...

通常会多次调用`read_html`，直到找到你想要的表格（或表格）。你可以使用两个主要参数来指定表格，`match`和`attrs`。提供给`match`的字符串用于精确匹配表格中的实际文本。这是网页上显示的文本。而`attrs`参数则是用来查找 HTML 表格属性，这些属性位于表格标签`<table>`后面。要查看更多表格属性，请访问 W3 Schools 的这个页面（[`bit.ly/2hzUzdD`](https://www.w3schools.com/TagS/tag_table.asp)）。

一旦我们在步骤 8 中找到了表格，我们仍然可以利用一些其他参数来简化操作。HTML 表格通常不能直接转化为漂亮的 DataFrame。常常会缺少列名、额外的行以及数据错位。在这个步骤中，`skiprows`接收一个包含要跳过的行号的列表。当读取文件时，这些行号对应于步骤 8 中 DataFrame 输出中的缺失值行。`header`参数也用来指定列名的位置。注意，`header`等于零，乍一看可能会觉得是错误的。每当`header`参数与`skiprows`一起使用时，行会先被跳过，这样每行会得到一个新的整数标签。正确的列名在第 4 行，但由于我们跳过了第 0 到第 3 行，它的新的整数标签是 0。

在步骤 11 中，`ffill`方法通过用最后一个非缺失值向下填充来填补任何缺失值。这个方法只是`fillna(method='ffill')`的快捷方式。

步骤 13 构建了一个包含所有前面步骤的函数，用来自动获取任何总统的支持率，前提是你有订单号。这个函数有一些不同之处。我们不是对整个 DataFrame 应用`ffill`方法，而是只对`President`列应用它。在特朗普的 DataFrame 中，其他列没有缺失数据，但这并不保证所有抓取的表格在其他列中也没有缺失数据。函数的最后一行将日期按更自然的方式从旧到新排序，这样会改变索引的顺序，因此我们通过`reset_index`将其重置，使其重新从零开始。

步骤 16 展示了一种常见的 pandas 用法，先将多个索引相似的 DataFrame 收集到一个列表中，然后用`concat`函数将它们合并在一起。合并成一个 DataFrame 后，我们应该对其进行视觉检查，确保其准确性。检查的一种方法是通过对数据进行分组，然后在每个组上使用`head`方法来快速查看每个总统部分的前几行。

步骤 18 中的摘要统计数据很有趣，因为每位继任总统的中位支持率都低于前任。如果对数据进行外推，可能会天真地预测未来几位总统的支持率为负数。

第 19 步中的绘图代码相当复杂。你可能会想，为什么一开始就需要遍历一个`groupby`对象。在当前的 DataFrame 结构中，它无法基于单列的值绘制不同组的数据。然而，第 23 步会展示如何设置 DataFrame，以便 pandas 可以直接绘制每个总统的数据，而不需要像这样的循环。

要理解第 19 步中的绘图代码，首先你需要知道`groupby`对象是可迭代的，在遍历时，它会返回一个包含当前组（这里就是总统的名字）和该组对应的子 DataFrame 的元组。这个`groupby`对象与控制绘图颜色和线条样式的值一起被打包。我们从 matplotlib 导入了色彩图模块`cm`，它包含了几十种不同的色彩图。传递一个 0 到 1 之间的浮动值可以从色彩图中选择一个特定的颜色，我们在`plot`方法中使用`color`参数来应用它。还需要注意的是，我们必须创建图形`fig`和绘图表面`ax`，以确保每条支持率线都绘制在同一张图上。在循环的每次迭代中，我们使用相同的绘图表面和同名的参数`ax`。

为了更好地比较各位总统，我们创建了一个新的列，表示在职天数。我们从每位总统组的其他日期中减去第一天的日期。当两个`datetime64`列相减时，结果是一个`timedelta64`对象，表示某段时间的长度，这里是天数。如果我们保持纳秒级的精度，x 轴也会显示过多的精度，因此使用特殊的`dt`访问器返回天数。

关键步骤出现在第 23 步。我们重新结构化数据，使每个总统都有一个专门的列来表示其支持率。Pandas 为每一列绘制了一条单独的线。最后，在第 24 步，我们使用`.loc`索引器同时选择前 250 天（行）以及特朗普和奥巴马的列。`ffill`方法用于在总统的某一天有缺失值时进行填充。在 Python 中，可以通过在函数前加`**`的方式将包含参数名称及其值的字典传递给函数，这个过程叫做**字典解包**。

# 还有更多内容……

第 19 步中的绘图显示了相当多的噪音，如果对数据进行平滑处理，可能会更容易解释。一个常见的平滑方法叫做**滚动平均**。Pandas 为 DataFrame 和`groupby`对象提供了`rolling`方法。它的工作方式类似于`groupby`方法，通过返回一个等待执行额外操作的对象。当创建时，你必须将窗口大小作为第一个参数传递，这个大小可以是一个整数或者一个日期偏移字符串。

在这个例子中，我们使用日期偏移字符串*90D*计算 90 天的移动平均值。`on`参数指定了滚动窗口计算的列：

```py
>>> pres_rm = pres_41_45.groupby('President', sort=False) \
                        .rolling('90D', on='End Date')['Approving'] \
                        .mean()
>>> pres_rm.head()
President    End Date   
George Bush  1989-01-26    51.000000
             1989-02-27    55.500000
             1989-03-02    57.666667
             1989-03-10    58.750000
             1989-03-13    58.200000
Name: Approving, dtype: float64
```

在此，我们可以使用 `unstack` 方法重新结构化数据，使其与步骤 23 的输出类似，然后制作我们的图表：

```py
>>> styles = ['-.', '-', ':', '-', ':']
>>> colors = [.9, .3, .7, .3, .9]
>>> color = cm.Greys(colors)
>>> title='90 Day Approval Rating Rolling Average'
>>> plot_kwargs = dict(figsize=(16,6), style=styles,
                       color = color, title=title)
>>> correct_col_order = pres_41_45.President.unique()

>>> pres_rm.unstack('President')[correct_col_order].plot(**plot_kwargs)
```

![](img/d120b3da-ec23-4575-b869-80eb4a9d486c.png)

# 另见

+   matplotlib 的 colormap 参考（[`bit.ly/2yJZOvt`](https://matplotlib.org/examples/color/colormaps_reference.html)）

+   所有日期偏移及其别名的列表（[`bit.ly/2xO5Yg0`](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)）

# 理解 `concat`、`join` 和 `merge` 之间的区别

`merge` 和 `join` 的 DataFrame 方法（而不是 Series）以及 `concat` 函数都提供了非常相似的功能，用于将多个 pandas 对象合并在一起。由于它们如此相似，并且在某些情况下可以互相替代，因此理解何时以及如何正确使用它们可能会让人困惑。为了帮助澄清它们之间的差异，以下是一个简要的概述：

+   `concat`：

    +   Pandas 函数

    +   垂直或水平合并两个或更多 pandas 对象

    +   仅在索引上对齐

    +   每当索引中出现重复项时会报错

    +   默认执行外连接，并可选择执行内连接

+   `join`：

    +   DataFrame 方法

    +   水平合并两个或更多 pandas 对象

    +   将调用的 DataFrame 的列/索引与其他对象的索引（而非列）对齐

    +   通过执行笛卡尔积处理连接列/索引中的重复值

    +   默认执行左连接，并可选择执行内连接、外连接和右连接

+   `merge`：

    +   DataFrame 方法

    +   精确地水平合并两个 DataFrame

    +   将调用的 DataFrame 的列/索引与另一个 DataFrame 的列/索引对齐

    +   通过执行笛卡尔积处理连接列/索引中的重复值

    +   默认执行内连接，并可选择执行左连接、外连接和右连接

`join` 方法的第一个参数是 `other`，它可以是单个 DataFrame/Series，或者是任何数量的 DataFrame/Series 列表。

# 准备就绪

在这个示例中，我们将完成合并 DataFrame 所需的步骤。第一种情况使用 `concat` 更为简单，而第二种情况则使用 `merge` 更为简单。

# 如何执行...

1.  让我们读取 2016 年、2017 年和 2018 年的股票数据，并使用循环将它们放入一个 DataFrame 列表中，而不是通过三次不同的 `read_csv` 调用。当前，Jupyter notebooks 只允许在一行上显示一个 DataFrame。但有一种方法可以借助 `IPython` 库自定义 HTML 输出。用户定义的 `display_frames` 函数接受一个 DataFrame 列表并将它们全部输出在同一行上：

```py
>>> from IPython.display import display_html

>>> years = 2016, 2017, 2018
>>> stock_tables = [pd.read_csv('data/stocks_{}.csv'.format(year),
                                index_col='Symbol') 
                    for year in years]

>>> def display_frames(frames, num_spaces=0):
        t_style = '<table style="display: inline;"'
        tables_html = [df.to_html().replace('<table', t_style) 
                       for df in frames]

        space = '&nbsp;' * num_spaces
        display_html(space.join(tables_html), raw=True)

>>> display_frames(stock_tables, 30)
>>> stocks_2016, stocks_2017, stocks_2018 = stock_tables
```

![](img/334685b5-02c5-4a06-a159-caccb833fc01.png)

1.  `concat` 函数是唯一能够垂直合并 DataFrame 的函数。让我们通过传递 `stock_tables` 列表来实现：

```py
>>> pd.concat(stock_tables, keys=[2016, 2017, 2018])
```

![](img/0c817ede-3963-45a8-961e-4ba60ab03d7c.png)

1.  还可以通过将`axis`参数更改为`columns`来水平合并 DataFrame：

```py
>>> pd.concat(dict(zip(years,stock_tables)), axis='columns')
```

![](img/761c169f-57ab-4ac7-aae1-e5e83bda06a3.png)

1.  现在我们已经开始水平合并 DataFrame，可以使用`join`和`merge`方法来复制`concat`的这一功能。在这里，我们使用`join`方法来合并`stock_2016`和`stock_2017`两个 DataFrame。默认情况下，DataFrame 会根据它们的索引进行对齐。如果某些列有相同的名称，则必须为`lsuffix`或`rsuffix`参数提供值，以便在结果中区分它们：

```py
>>> stocks_2016.join(stocks_2017, lsuffix='_2016',
                     rsuffix='_2017', how='outer')
```

![](img/7003e693-6693-46b5-9880-13e184c2612d.png)

1.  为了精确复制第 3 步中`concat`函数的输出，我们可以将一个 DataFrame 列表传递给`join`方法：

```py
>>> other = [stocks_2017.add_suffix('_2017'),
             stocks_2018.add_suffix('_2018')]
>>> stocks_2016.add_suffix('_2016').join(other, how='outer')
```

![](img/146587e7-f778-4917-8eac-06e4f8208f7a.png)

1.  让我们检查它们是否确实完全相等：

```py
>>> stock_join = stocks_2016.add_suffix('_2016').join(other, 
                                                      how='outer')
>>> stock_concat = pd.concat(dict(zip(years,stock_tables)),
                             axis='columns')
>>> level_1 = stock_concat.columns.get_level_values(1)
>>> level_0 = stock_concat.columns.get_level_values(0).astype(str)
>>> stock_concat.columns = level_1 + '_' + level_0
>>> stock_join.equals(stock_concat)
True
```

1.  现在，让我们来看一下`merge`，它与`concat`和`join`不同，能够将两个 DataFrame 精确地合并在一起。默认情况下，`merge`会尝试根据每个 DataFrame 中相同名称的列来对齐列中的值。不过，你也可以选择通过将布尔参数`left_index`和`right_index`设置为`True`，让它根据索引进行对齐。让我们将 2016 年和 2017 年的库存数据合并在一起：

```py
>>> stocks_2016.merge(stocks_2017, left_index=True, 
                      right_index=True)
```

![](img/075849a5-3ab9-4b9c-a914-be9e715ea643.png)

1.  默认情况下，`merge`使用内连接，并自动为同名列提供后缀。让我们改用外连接，然后再执行一次外连接，将 2018 年的数据合并在一起，从而完全复制`concat`的功能：

```py
>>> step1 = stocks_2016.merge(stocks_2017, left_index=True, 
                              right_index=True, how='outer',
                              suffixes=('_2016', '_2017'))

>>> stock_merge = step1.merge(stocks_2018.add_suffix('_2018'), 
                              left_index=True, right_index=True,
                              how='outer')

>>> stock_concat.equals(stock_merge)
True
```

1.  现在让我们转向我们关心对齐列值而非索引或列标签的数据集。`merge`方法正是为这种情况而设计的。让我们来看一下两个新的小数据集，`food_prices`和`food_transactions`：

```py
>>> names = ['prices', 'transactions']
>>> food_tables = [pd.read_csv('data/food_{}.csv'.format(name)) 
                    for name in names]
>>> food_prices, food_transactions = food_tables
>>> display_frames(food_tables, 30)
```

![](img/f98e02af-0bc9-4214-bff8-783bdda27391.png)

1.  如果我们想要找到每笔交易的总金额，我们需要在`item`和`store`列上连接这些表：

```py
>>> food_transactions.merge(food_prices, on=['item', 'store'])
```

![](img/32a7e87a-018f-40bb-bb3e-123be98fdf88.png)

1.  价格现在已经与对应的商品和商店正确对齐，但存在一个问题。客户 2 购买了四个`steak`商品。由于`steak`商品在商店`B`的每个表格中都出现了两次，它们之间发生了笛卡尔积，导致了四行数据的出现。另请注意，`coconut`商品缺失，因为没有相应的价格。我们来解决这两个问题：

```py
>>> food_transactions.merge(food_prices.query('Date == 2017'),
                            how='left')
```

![](img/cbeec36e-7160-4664-a8a2-983ad7e70c41.png)

1.  我们可以使用`join`方法来复制这一过程，但必须首先将`food_prices` DataFrame 的连接列放入索引中：

```py
>>> food_prices_join = food_prices.query('Date == 2017') \
                                  .set_index(['item', 'store'])
>>> food_prices_join
```

![](img/a6e73c76-1791-4dd5-b906-a5f9b0f386b1.png)

1.  `join`方法仅与传递的 DataFrame 的索引对齐，但可以使用调用 DataFrame 的索引或列。要使用列进行对齐，你需要将它们传递给`on`参数：

```py
>>> food_transactions.join(food_prices_join, on=['item', 'store'])
```

1.  输出与第 11 步的结果完全匹配。要使用`concat`方法复制此操作，您需要将`item`和`store`列放入两个 DataFrame 的索引中。然而，在这个特定的案例中，由于至少一个 DataFrame 中存在重复的索引值（`item`为`steak`且`store`为`B`），因此会产生错误。

```py
>>> pd.concat([food_transactions.set_index(['item', 'store']), 
               food_prices.set_index(['item', 'store'])],
              axis='columns')
Exception: cannot handle a non-unique multi-index!
```

# 它是如何工作的...

在同时导入多个 DataFrame 时，重复编写`read_csv`函数可能会变得繁琐。自动化这个过程的一种方法是将所有文件名放入列表中，并使用`for`循环遍历它们。这在第 1 步中通过列表推导实现。

本步骤的其余部分构建了一个函数，用于在 Jupyter notebook 中将多个 DataFrame 显示在同一行输出上。所有 DataFrame 都有一个`to_html`方法，它返回表格的原始 HTML 字符串表示。每个表格的 CSS（层叠样式表）通过修改`display`属性为*inline*，使得元素水平并排显示，而不是垂直显示。为了在 notebook 中正确渲染表格，必须使用 IPython 库提供的辅助函数`read_html`。

在第 1 步结束时，我们将 DataFrame 列表解包到各自适当命名的变量中，以便每个单独的表格可以轻松且清晰地引用。拥有 DataFrame 列表的好处是，它正好符合`concat`函数的需求，如第 2 步所示。注意第 2 步如何使用`keys`参数为每个数据块命名。通过将字典传递给`concat`，也可以实现这一点，如第 3 步所示。

在第 4 步，我们必须将`join`的类型更改为`outer`，以包括传递的 DataFrame 中那些没有在调用 DataFrame 中找到的索引的所有行。在第 5 步，传递的 DataFrame 列表不能有任何重复的列。虽然有`rsuffix`参数，它仅在传递单个 DataFrame 时有效，而非传递 DataFrame 列表。为了解决这个限制，我们事先使用`add_suffix`方法更改列名，然后调用`join`方法。

在第 7 步，我们使用`merge`，它默认按照两个 DataFrame 中相同的列名进行对齐。若要更改此默认行为，并按照某个或两个 DataFrame 的索引进行对齐，可以将`left_index`或`right_index`参数设置为`True`。第 8 步通过两次调用`merge`完成了复制。如你所见，当你根据索引对齐多个 DataFrame 时，`concat`通常是比`merge`更好的选择。

在第 9 步，我们转变思路，聚焦于`merge`占优势的情况。`merge`方法是唯一能够根据列值对调用的 DataFrame 和传递的 DataFrame 进行对齐的方法。第 10 步展示了合并两个 DataFrame 的简单方法。`on`参数并非必需，但为了清晰起见提供。

不幸的是，当合并数据帧时很容易复制或丢弃数据，如第 10 步所示。在合并数据后进行一些合理性检查非常重要。在这个例子中，`food_prices`数据集在商店`B`中对`steak`的价格有重复，因此我们在第 11 步通过查询仅保留当前年份的行来删除了这一行。我们还改为左连接以确保每笔交易都被保留，无论价格是否存在。

在这些情况下可以使用联接，但是传递的数据帧中的所有列都必须首先移动到索引中。最后，当您打算根据其列中的值对齐数据时，使用`concat`将是一个不好的选择。

# 还有更多内容……

可以在不知道文件名的情况下从特定目录读取所有文件到数据帧中。Python 提供了几种迭代目录的方法，其中`glob`模块是一个流行的选择。天然气价格目录包含五个不同的 CSV 文件，每个文件从 2007 年开始每周记录一个特定等级的天然气价格。每个文件只有两列——每周的日期和价格。这是一个完美的情况，可以通过`concat`函数迭代所有文件，并将它们全部组合在一起。`glob`模块具有`glob`函数，它接受一个参数——要迭代的目录的位置作为字符串。要获取目录中的所有文件，请使用字符串 ***。在这个例子中，**.csv* 仅返回以 *.csv* 结尾的文件。`glob`函数的结果是一组字符串文件名，可以直接传递给`read_csv`函数：

```py
>>> import glob

>>> df_list = []
>>> for filename in glob.glob('data/gas prices/*.csv'):
        df_list.append(pd.read_csv(filename, index_col='Week',
                       parse_dates=['Week']))

>>> gas = pd.concat(df_list, axis='columns')
>>> gas.head()
```

![](img/8a01d07f-d580-4dce-9d3c-f47cf9a4c0ea.png)

# 另请参见

+   IPython `read_html` 函数的官方文档（[`bit.ly/2fzFRzd`](http://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display_html)）

+   参考*Exploding indexes*章节的食谱（来自第十二章，*索引对齐*）

# 连接到 SQL 数据库

要成为一名严肃的数据分析师，你几乎肯定需要学习一定量的 SQL。世界上大部分数据存储在接受 SQL 语句的数据库中。有许多关系型数据库管理系统，其中 SQLite 是最流行和易于使用的之一。

# 准备工作

我们将探索由 SQLite 提供的 Chinook 样例数据库，其中包含 11 张音乐商店数据表。首次深入研究合适的关系数据库时，最好做的事情之一是研究数据库图表（有时称为实体关系图），以更好地理解表之间的关系。在浏览此处的食谱时，以下图表将非常有帮助：

![](img/bfb39fe0-ea36-4d28-a0d2-245d6385b744.png)

为了使这个步骤有效，您需要安装`sqlalchemy` Python 包。如果您安装了 Anaconda 发行版，那么它应该已经可用。在与数据库建立连接时，SQLAlchemy 是首选的 pandas 工具。在这个步骤中，您将学习如何连接到 SQLite 数据库。然后，您将提出两个不同的查询，并通过使用`merge`方法将表格连接在一起来回答它们。

# 如何做到...

1.  在我们开始从`chinook`数据库读取表格之前，我们需要设置我们的 SQLAlchemy 引擎：

```py
>>> from sqlalchemy import create_engine
>>> engine = create_engine('sqlite:///data/chinook.db')
```

1.  现在我们可以回到 pandas 的世界，并在剩余的步骤中继续待在那里。让我们完成一个简单的命令，并使用`read_sql_table`函数读取`tracks`表格。表格的名称是第一个参数，SQLAlchemy 引擎是第二个参数：

```py
>>> tracks = pd.read_sql_table('tracks', engine)
>>> tracks.head()
>>> genres = pd.read_sql_table('genres', engine)
```

![](img/e34917a4-21a5-4bba-b011-70d354bc6d96.png)

1.  在接下来的步骤中，我们将通过数据库图表回答一些不同的具体查询。首先，让我们找出每种流派歌曲的平均长度：

```py
>>> genre_track = genres.merge(tracks[['GenreId', 'Milliseconds']], 
                               on='GenreId', how='left') \
                        .drop('GenreId', axis='columns')

>>> genre_track.head()
```

![](img/41707069-07b1-4426-a5c2-ffa53f5c92de.png)

1.  现在我们可以轻松地找出每种流派歌曲的平均长度。为了帮助解释，我们将`Milliseconds`列转换为`timedelta`数据类型：

```py
>>> genre_time = genre_track.groupby('Name')['Milliseconds'].mean()
>>> pd.to_timedelta(genre_time, unit='ms').dt.floor('s')
                                             .sort_values()
Name
Rock And Roll        00:02:14
Opera                00:02:54
Hip Hop/Rap          00:02:58
...
Drama                00:42:55
Science Fiction      00:43:45
Sci Fi & Fantasy     00:48:31
Name: Milliseconds, dtype: timedelta64[ns]
```

1.  现在让我们找出每位顾客的总花费金额。我们将需要将`customers`、`invoices`和`invoice_items`表格连接在一起：

```py
>>> cust = pd.read_sql_table('customers', engine, 
                             columns=['CustomerId','FirstName',
                                      'LastName'])
>>> invoice = pd.read_sql_table('invoices', engine, 
                                 columns=['InvoiceId','CustomerId'])
>>> ii = pd.read_sql_table('invoice_items', engine, 
                            columns=['InvoiceId', 'UnitPrice',
                                     'Quantity'])

>>> cust_inv = cust.merge(invoice, on='CustomerId') \
                   .merge(ii, on='InvoiceId')
>>> cust_inv.head()
```

![](img/6e860d89-e3e3-4d5e-8ed2-c768bf668f23.png)

1.  现在我们可以将数量乘以单价，然后找出每位顾客的总花费金额：

```py
>>> total = cust_inv['Quantity'] * cust_inv['UnitPrice']
>>> cols = ['CustomerId', 'FirstName', 'LastName']
>>> cust_inv.assign(Total = total).groupby(cols)['Total'] \
                                  .sum() \
                                  .sort_values(ascending=False) \
                                  .head()
CustomerId  FirstName  LastName  
6           Helena     Holý          49.62
26          Richard    Cunningham    47.62
57          Luis       Rojas         46.62
46          Hugh       O'Reilly      45.62
45          Ladislav   Kovács        45.62
Name: Total, dtype: float64
```

# 它的工作原理...

`create_engine`函数需要一个连接字符串才能正常工作。SQLite 的连接字符串非常简单，只是数据库的位置，它位于数据目录中。其他关系型数据库管理系统具有更复杂的连接字符串。您需要提供用户名、密码、主机名、端口号，以及可选的数据库。您还需要提供 SQL 方言和驱动程序。连接字符串的一般形式如下：`dialect+driver://username:password@host:port/database`。您的特定关系型数据库的驱动程序可能需要单独安装。

一旦我们创建了引擎，在第二步中使用`read_sql_table`函数将整个表格选择到 DataFrames 中非常容易。数据库中的每个表格都有一个主键，用于在图表中唯一标识每一行。在第三步中，我们通过`GenreId`将流派链接到音轨。由于我们只关心音轨长度，因此在执行合并之前，我们将音轨 DataFrame 减少到所需的列。表格合并后，我们可以通过基本的`groupby`操作回答查询。

我们更进一步，将整数形式的毫秒转换为更易于阅读的 Timedelta 对象。关键是将正确的度量单位作为字符串传递。现在我们有了一个 Timedelta Series，可以使用`dt`属性访问`floor`方法，它会将时间舍入到最接近的秒。

第五步所需的查询涉及三个表。我们可以通过将它们传递给`columns`参数，大大减少表格内容，仅保留我们需要的列。在使用`merge`时，如果连接列的名称相同，则这些列不会被保留。第六步中，我们本可以通过以下方式分配一个表示价格乘以数量的列：

```py
cust_inv['Total'] = cust_inv['Quantity'] * cust_inv['UnitPrice']
```

以这种方式分配列是完全没有问题的。我们选择动态创建一个新的列，使用 assign 方法以便支持连续的方法链。

# 还有更多内容...

如果你熟悉 SQL，可以将 SQL 查询写成字符串并传递给`read_sql_query`函数。例如，以下查询将重现第四步的输出：

```py
>>> sql_string1 = '''
    select 
        Name, 
        time(avg(Milliseconds) / 1000, 'unixepoch') as avg_time
    from (
            select 
                g.Name, 
                t.Milliseconds
            from 
                genres as g 
            join
                tracks as t
                on 
                    g.genreid == t.genreid
         )
    group by 
        Name
    order by 
         avg_time
'''
>>> pd.read_sql_query(sql_string1, engine)
```

![](img/ccb4217b-d635-47d8-bfd3-c5952e29082b.png)

要重现第六步的答案，请使用以下 SQL 查询：

```py
>>> sql_string2 = '''
    select 
          c.customerid, 
          c.FirstName, 
          c.LastName, 
          sum(ii.quantity * ii.unitprice) as Total
    from
         customers as c
    join
         invoices as i
              on c.customerid = i.customerid
    join
        invoice_items as ii
              on i.invoiceid = ii.invoiceid
    group by
        c.customerid, c.FirstName, c.LastName
    order by
        Total desc
'''
>>> pd.read_sql_query(sql_string2, engine)
```

![](img/307645d3-c9aa-4981-8e8c-eb934a124bf9.png)

# 另见

+   所有*SQLAlchemy*的引擎配置（[`bit.ly/2kb07vV`](http://docs.sqlalchemy.org/en/latest/core/engines.html)）

+   Pandas 官方文档关于*SQL 查询*（[`bit.ly/2fFsOQ8`](http://pandas.pydata.org/pandas-docs/stable/io.html#sql-queries)）
