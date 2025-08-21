# 第五章：计算基础 – Python 入门

本章将介绍 Python 在分析中的应用。该部分主要面向对 Python 不熟悉的初学者程序员或开发人员。本章结束时，你将对 Python 基础语言的特性有基本的了解，这对于医疗分析和机器学习至关重要。你还将了解如何开始使用 `pandas` 和 `scikit-learn`，这两个 Python 数据分析的关键库。

如果你想跟随 Jupyter Notebook 操作，我们建议你参考第一章，*医疗分析入门*，来启动一个新的 Jupyter 会话。本章的笔记本也可以在书籍的官方代码库中在线获取。

# 变量和类型

Python 中的基本变量类型包括字符串和数字类型。在本节中，我们将介绍这两种类型。

# 字符串

在 Python 中，**字符串**是一种存储文本字符的变量类型，这些字符可以是字母、数字、特殊字符和标点符号。在 Python 中，我们使用单引号或双引号来表示一个变量是字符串，而不是数字：

```py
var = 'Hello, World!'
print(var)
```

字符串不能用于数字的数学运算，但它们可以用于其他有用的操作，正如我们在以下示例中看到的：

```py
string_1 = '1'
string_2 = '2'
string_sum = string_1 + string_2
print(string_sum)
```

上述代码的结果是打印字符串 `'12'`，而不是 `'3'`。在 Python 中，`+` 运算符对两个字符串进行操作时，执行的是拼接操作（将第二个字符串附加到第一个字符串的末尾），而不是相加。

其他作用于字符串的运算符包括 `*` 运算符（用于重复字符串 ![](img/14655201-6690-4b4c-ad90-85b3c37d6601.png) 多次，例如，`string_1 * 3`）以及 `<` 和 `>` 运算符（用于比较字符串的 ASCII 值）。

要将数据从数字类型转换为字符串，我们可以使用 `str()` 方法。

由于字符串是字符序列，我们可以对其进行索引和切片（就像我们对其他数据容器所做的那样，稍后你会看到）。切片是字符串的连续部分。为了索引/切片它们，我们使用方括号中的整数来表示字符的位置：

```py
test_string = 'Healthcare'
print(test_string[0])
```

输出如下所示：

```py
H
```

要切片字符串，我们需要在方括号中包含起始位置和结束位置，二者用冒号隔开。请注意，结束位置会包含所有字符，*直到但不包括*该结束位置，正如我们在以下示例中看到的：

```py
print(test_string[0:6])
```

输出如下：

```py
Health
```

前面我们提到了 `str()` 方法。字符串有很多其他方法。它们的完整列表可以在在线 Python 文档中查看，网址是 [www.python.org](http://www.python.org)。这些方法包括大小写转换、查找特定子字符串和去除空格等操作。这里我们将讨论另外一个方法——`split()` 方法。`split()` 方法作用于字符串，并接受一个 `separator` 参数。

输出是一个字符串列表；列表中的每个项目是原始字符串的一个组成部分，按`separator`分隔开。这对于解析由标点符号（如`,`或`;`）分隔的字符串非常有用。我们将在下一节讨论列表。下面是`split()`方法的示例：

```py
test_split_string = 'Jones,Bill,49,Atlanta,GA,12345'
output = test_split_string.split(',')
print(output)
```

输出结果如下：

```py
['Jones', 'Bill', '49', 'Atlanta', 'GA', '12345']
```

# 数值类型

在 Python 中，最常用于分析的两种数值类型是**整数**和**浮点数**。要将数据转换为这些类型，可以分别使用`int()`和`float()`函数。常见的数值操作都可以通过常用运算符来实现：`+`、`-`、`*`、`/`、`<`和`>`。包含特殊数值方法的模块，如`math`和`random`，对于分析尤其有用。更多有关数值类型的信息，可以参考在线 Python 文档（参见上一节中的链接）。

请注意，在某些版本的 Python 中，使用`/`运算符对两个整数进行除法运算时，会执行**向下取整除法**（即忽略小数点后的部分）；例如，`10/4`会等于`2`，而不是`2.5`。这是一个隐蔽而严重的错误，可能会影响数值计算。然而，在本书中使用的 Python 版本里，我们不需要担心这个问题。

**布尔类型**是一个特殊的整数类型，用于表示`True`和`False`值。要将整数转换为布尔类型，可以使用`bool()`函数。零会被转换为`False`；其他任何整数都会被转换为`True`。布尔变量的行为像 1（True）和 0（False），不过在转换为字符串时，它们分别返回`True`和`False`。

# 数据结构和容器

在上一节中，我们讲解了存储单一值的变量类型。接下来，我们将讨论能够存储多个值的数据结构。这些数据结构包括列表、元组、字典和集合。在 Python 中，列表和元组通常被称为序列。在本书中，我们将“数据结构”和“数据容器”这两个术语互换使用。

# 列表

列表是一个广泛使用的数据结构，可以包含多个值。我们来看一下列表的一些特点：

+   要创建一个列表，我们使用方括号`[]`。

    示例：`my_list = [1, 2, 3]`。

+   列表可以包含任意组合的数值类型、字符串、布尔类型、元组、字典，甚至其他列表。

    示例：`my_diverse_list = [51, 'Health', True, [1, 2, 3]]`。

+   列表和字符串一样，都是序列，支持索引和切片操作。

    例如，在上面的示例中，`my_diverse_list[0]`会等于`51`。`my_diverse_list[0:2]`会等于`[51, 'Health']`。要访问嵌套列表中的`3`，我们可以使用`my_diverse_list[3][2]`。

+   列表是**可变**的（不同于字符串和元组），这意味着我们可以通过索引来更改单个元素。

    例如，如果我们输入了 `my_diverse_list[2] = False` 命令，那么我们的新 `my_diverse_list` 将等于 `[51, 'Health', False, [1, 2, 3]]`。

列表在数据分析中的显著优势包括其丰富的辅助方法，如 `append()`、`extend()` 和 `join()`，以及它们与 `pandas` 和 `numpy` 数据结构的互换性。

# 元组

元组类似于列表。要创建元组，我们使用圆括号 `()`。示例：`my_tuple = (1, 2, 3)`。元组与列表的主要区别在于元组是 **不可变的**，因此我们不能更改元组中的任何元素。如果我们尝试 `my_tuple[0] = 4`，则会抛出错误。由于它们的值是不可变的，元组在设置常量变量时非常有用。

# 字典

**字典**是 Python 中常见的数据结构。它用于存储从键到值的单向映射。例如，如果我们想创建一个字典来存储病人姓名及其对应的房间号，我们可以使用以下代码：

```py
rooms = {
    'Smith': '141-A',
    'Davis': '142',
    'Williams': '144',
    'Johnson': '145-B'
}
```

让我们更详细地讨论一下前面的代码片段：

+   `rooms` 字典中的名称被称为 **键**。字典中的键必须是唯一的。要访问它们，我们可以使用 `keys()` 函数，`rooms.keys()`。

+   `rooms` 字典中的房间号被称为 **值**。要访问所有值，我们可以使用 `values()` 函数，`rooms.values()`。要访问单个值，我们只需提供其键的名称，用方括号括起来。例如，`rooms['Smith']` 将返回 `'141-A'`。因此，我们可以说字典将键映射到其值。

+   要访问包含每个键及其对应值的嵌套元组列表，我们可以使用 `items()` 函数，`rooms.items()`。

+   字典的值不一定只是字符串；事实上，值可以是任何数据类型/结构。键可以是特定的变量，例如整数或字符串。虽然值是可变的，但键是不可变的。

+   字典没有固有的顺序，因此不支持按数字索引和切片操作。

# 集合

虽然在 Python 中集合不像它的流行表亲列表那样受到关注，但集合在数据分析中扮演着重要角色，因此我们在这里包括它们。要创建集合，我们使用内置的 `set()` 函数。关于集合，你需要知道三件事：

+   它们是不可变的

+   它们是无序的

+   集合的元素是唯一的

因此，如果你熟悉基础集合论，Python 中的集合与其数学对应物非常相似。集合方法也复制了典型的集合操作，包括 `union()`、`intersection()`、`add()` 和 `remove()`。当你想对数据结构（如列表或元组）执行典型的集合操作时，这些函数会派上用场，前提是将其转换为集合。

# Python 编程– 通过示例说明

在前面的章节中，我们讨论了变量类型和数据容器。Python 编程中还有很多其他方面，如 if/else 语句、循环和推导式的控制流；函数；以及类和面向对象编程。通常，Python 程序会被打包成**模块**，即独立的脚本，可以通过命令行运行执行计算任务。

让我们通过一个“模块”来介绍一些 Python 的概念（你可以使用 Jupyter Notebook 来实现）：

```py
from math import pow

LB_TO_KG = 0.453592
IN_TO_M = 0.0254

class Patient:
    def __init__(self, name, weight_lbs, height_in):
        self.name = name
        self.weight_lbs = weight_lbs
        self.weight_kg = weight_lbs * LB_TO_KG
        self.height_in = height_in
        self.height_m = height_in * IN_TO_M

    def calculate_bmi(self):
        return self.weight_kg / pow(self.height_m, 2)

    def get_height_m(self):
        return self.height_m

if __name__ == '__main__':
    test_patients = [
        Patient('John Smith', 160, 68),
        Patient('Patty Johnson', 180, 73)
    ]
    heights = [patient.get_height_m() for patient in test_patients]
    print(
        "John's height: ", heights[0], '\n',
        "Patty's height: ", heights[1], '\n',
        "John's BMI: ", test_patients[0].calculate_bmi(), '\n',
        "Patty's BMI: ", test_patients[1].calculate_bmi()
    )
```

当你运行这段代码时，你应该会看到以下输出：

```py
John's height:  1.7271999999999998 
 Patty's height:  1.8541999999999998 
 John's BMI:  24.327647271211504 
 Patty's BMI:  23.74787410486812
```

上述代码是一个 Python 模块，打印出两个虚拟患者的身高和**体重指数**（**BMI**）。让我们更详细地看看这段代码的每个元素：

+   代码块的第一行是**导入语句**。这使我们能够导入其他模块中已编写的函数和类，这些模块可能是与 Python 一起分发的、开源软件编写的，或者是我们自己编写的。**模块**可以简单地理解为一个包含 Python 函数、常量和/或类的文件，它的扩展名为`.py`。要导入整个模块，我们只需使用`import`关键字后跟模块名称，例如`import math`。请注意，我们还使用了`from`关键字，因为我们只想导入特定的函数——`pow()`函数。这样也避免了每次想计算幂时都需要输入`math.pow()`的麻烦。

+   接下来的两行包含了我们将用来进行单位转换的**常量**。常量通常用大写字母表示。

+   接下来，我们定义了一个`Patient`类，包含一个**构造函数**和两个**方法**。构造函数接受三个参数——姓名、身高和体重——并将特定`Patient`实例的三个属性设置为这些值。它还将体重从磅转换为千克，将身高从英寸转换为米，并将这些值存储在两个额外的属性中。

+   这两个方法被编码为**函数**，使用`def`关键字。`calculate_bmi()`返回患者的 BMI，而`get_height()`则简单地返回身高（以米为单位）。

+   接下来，我们有一个简短的`if`语句。这个`if`语句的作用是：只有在作为命令行调用的主模块时才执行后续代码。其他`if`语句可能包含多个`elif`子句，还可以包含一个最终的`else`子句。

+   接下来，我们创建了一个包含两位患者的信息的列表，分别是 John Smith 和 Patty Johnson，以及他们的身高和体重数据。

+   下一行使用了列表**推导式**来创建两个患者的身高列表。推导式在 Python 编程中非常流行，也可以用于字典操作。

+   最后，我们的`print`语句会将四个数字作为输出（两个身高和两个 BMI 值）。

本章末尾提供了更多关于基础 Python 编程语言的参考资料。你也可以访问在线文档 [www.python.org](http://www.python.org)。

# pandas 简介

到目前为止，我们讨论的几乎所有功能都是 *基础* Python 的功能；也就是说，使用这些功能不需要额外的包或库。事实是，本书中我们编写的大部分代码将涉及几个常用于分析的 *外部* Python 包。**pandas** 库（[`pandas.pydata.org`](http://pandas.pydata.org)）是后续编程章节的核心部分。pandas 在机器学习中的功能有三方面：

+   从平面文件导入数据到你的 Python 会话

+   使用 pandas DataFrame 及其函数库来整理、操作、格式化和清洗数据

+   将数据从你的 Python 会话导出到平面文件

让我们逐一回顾这些功能。

平面文件是存储与医疗相关数据的常用方式（还有 HL7 格式，本书不涉及）。**平面文件**是数据的文本文件表示形式。使用平面文件，数据可以像数据库一样以行和列的形式表示，不同的是，标点符号或空白字符用作列的分隔符，而回车符则用作行的分隔符。我们将在第七章，*在医疗领域创建预测模型* 中看到一个平面文件的示例。

pandas 允许我们从各种其他 Python 结构和平面文件中导入数据到一个表格化的 Python 数据结构，称为 **DataFrame**，包括 Python 字典、pickle 对象、**逗号分隔值**（**csv**）文件、**定宽格式**（**fwf**）文件、Microsoft Excel 文件、JSON 文件、HTML 文件，甚至是 SQL 数据库表。

一旦数据进入 Python，你可以使用一些附加功能来探索和转换数据。需要对某一列执行数学运算，比如求和吗？需要执行类似 SQL 的操作，如 JOIN 或添加列（请参阅第三章，*机器学习基础*）？需要按条件过滤行吗？这些功能都可以通过 pandas 的 API 实现。我们将在第六章，*衡量医疗质量* 和第七章，*在医疗领域创建预测模型* 中充分利用 pandas 的一些功能。

最后，当我们完成数据探索、清洗和整理后，如果我们愿意，可以选择将数据导出为列出的多种格式之一。或者我们可以将数据转换为 NumPy 数组并训练机器学习模型，正如我们在本书后面会做的那样。

# 什么是 pandas DataFrame？

**pandas DataFrame**可以看作是一种二维的、类似矩阵的数据结构，由行和列组成。pandas DataFrame 类似于 R 中的 dataframe 或 SQL 中的表。与传统矩阵和其他 Python 数据结构相比，它的优势包括可以在同一 DataFrame 中包含不同类型的列、提供广泛的预定义函数以便于数据操作，以及支持快速转换为其他文件格式（包括数据库、平面文件格式和 NumPy 数组）的单行接口（便于与 scikit-learn 的机器学习功能集成）。因此，`pandas`确实是连接许多机器学习管道的粘合剂，从数据导入到算法应用。

pandas 的局限性包括较慢的性能以及缺乏内建的并行处理功能。因此，如果你正在处理数百万或数十亿个数据点，**Apache Spark**（[`spark.apache.org/`](https://spark.apache.org/)）可能是一个更好的选择，因为它的语言内置了并行处理功能。

# 导入数据

在本节中，我们演示了如何通过字典、平面文件和数据库将数据加载到 Python 中。

# 从 Python 数据结构导入数据到 pandas

使用`pandas` DataFrame 的第一步是通过`pandas`构造函数`DataFrame()`来创建一个 DataFrame。构造函数接受多种 Python 数据结构作为输入。它还可以接收 NumPy 数组和 pandas 的**Series**，Series 是另一种一维的`pandas`数据结构，类似于列表。这里我们演示如何将一个字典的列表转换为 DataFrame：

```py
import pandas as pd
data = {
    'col1': [1, 2, 3],
    'col2': [4, 5, 6],
    'col3': ['x', 'y', 'z']
}

df = pd.DataFrame(data)
print(df)
```

输出如下：

```py
   col1  col2 col3
0     1     4    x
1     2     5    y
2     3     6    z
```

# 从平面文件导入数据到 pandas

因为医疗保健数据通常采用平面文件格式，如`.csv`或`.fwf`，所以了解`read_csv()`和`read_fwf()`函数非常重要，这两个函数分别用于将数据从这两种格式导入`pandas`。这两个函数都需要作为必需参数提供平面文件的完整路径，并且还有十多个可选参数，用于指定诸如列的数据类型、标题行、要包含在 DataFrame 中的列等选项（完整的函数参数列表可以在线查看）。通常，最简单的方法是将所有列导入为字符串类型，然后再将列转换为其他数据类型。在下面的示例中，使用`read_csv()`函数从一个包含一个标题行（`row #0`）的平面`.csv`文件中读取数据，并创建一个名为`data`的 DataFrame：

```py
pt_data = pd.read_csv(data_full_path,header=0,dtype='str')
```

因为定宽文件没有显式的字符分隔符，`read_fwf()`函数需要一个额外的参数`widths`，它是一个整数列表，指定每一列的宽度。`widths`的长度应该与文件中的列数匹配。作为替代，`colspecs`参数接收一个元组列表，指定每列的起始点和终止点：

```py
pt_data = pd.read_fwf(source,widths=data_widths,header=None,dtype='str')
```

# 从数据库导入数据到 pandas

`pandas`库还支持从 SQL 数据库直接导入表格的函数。这些函数包括`read_sql_query()`和`read_sql_table()`。在使用这些函数之前，必须先建立与数据库的连接，以便将其传递给函数。以下示例展示了如何使用`read_sql_query()`函数将 SQLite 数据库中的表读取到 DataFrame 中：

```py
import sqlite3

conn = sqlite3.connect(pt_db_full_path)
table_name = 'TABLE1'
pt_data = pd.read_sql_query('SELECT * from ' + table_name + ';',conn) 
```

如果你希望连接到标准数据库，如 MySQL 数据库，代码将类似，唯一不同的是连接语句，它将使用针对 MySQL 数据库的相应函数。

# DataFrame 的常见操作

在本节中，我们将介绍一些对执行分析有用的 DataFrame 操作。有关更多操作的描述，请参阅官方的 pandas 文档，网址为[`pandas.pydata.org/`](https://pandas.pydata.org/)。

# 添加列

添加列是数据分析中常见的操作，无论是从头开始添加新列还是转换现有列。这里我们将介绍这两种操作。

# 添加空白列或用户初始化的列

要添加一个新的 DataFrame 列，你可以在 DataFrame 名称后加上新列的名称（用单引号和方括号括起来），并将其设置为你喜欢的任何值。要添加一个空字符串或整数的列，你可以将列设置为`""`或`numpy.nan`，后者需要事先导入`numpy`。要添加一个零的列，可以将列设置为`0`。以下示例说明了这些要点：

```py
df['new_col1'] = ""
df['new_col2'] = 0
print(df)
```

输出如下：

```py
   col1  col2 col3 new_col1  new_col2
0     1     4    x                  0
1     2     5    y                  0
2     3     6    z                  0
```

# 通过转换现有列添加新列

在某些情况下，你可能希望添加一个新列，该列是现有列的函数。在以下示例中，新的列`example_new_column_3`作为现有列`old_column_1`和`old_column_2`的和被添加。`axis=1`参数表示你希望对列进行横向求和，而不是对列进行纵向求和：

```py
df['new_col3'] = df[[
    'col1','col2'
]].sum(axis=1)

print(df)
```

输出如下：

```py
   col1  col2 col3 new_col1  new_col2  new_col3
0     1     4    x                  0         5
1     2     5    y                  0         7
2     3     6    z                  0         9
```

以下第二个示例使用 pandas 的`apply()`函数完成类似的任务。`apply()`是一个特殊的函数，因为它允许你将任何函数应用于 DataFrame 中的列（包括你自定义的函数）：

```py
old_column_list = ['col1','col2']
df['new_col4'] = df[old_column_list].apply(sum, axis=1)
print(df)
```

输出如下：

```py
   col1  col2 col3 new_col1  new_col2  new_col3  new_col4
0     1     4    x                  0         5         5
1     2     5    y                  0         7         7
2     3     6    z                  0         9         9
```

# 删除列

要删除列，可以使用 pandas 的`drop()`函数。它接受单个列名或列名列表，在此示例中，额外的可选参数指示沿哪个轴删除列，并且是否在原地删除列：

```py
df.drop(['col1','col2'], axis=1, inplace=True)
print(df)
```

输出如下：

```py
  col3 new_col1  new_col2  new_col3  new_col4
0    x                  0         5         5
1    y                  0         7         7
2    z                  0         9         9
```

# 对多个列应用函数

要对 DataFrame 中的多个列应用函数，可以使用`for`循环遍历列的列表。在以下示例中，预定义的列列表从字符串类型转换为数字类型：

```py
df['new_col5'] = ['7', '8', '9']
df['new_col6'] = ['10', '11', '12']

for str_col in ['new_col5','new_col6']:
    df[[str_col]] = df[[str_col]].apply(pd.to_numeric)

print(df)
```

这是输出结果：

```py
  col3 new_col1  new_col2  new_col3  new_col4  new_col5  new_col6
0    x                  0         5         5         7        10
1    y                  0         7         7         8        11
2    z                  0         9         9         9        12
```

# 合并 DataFrame

DataFrame 也可以彼此组合，只要它们在合并轴上有相同数量的条目。在此示例中，两个 DataFrame 被垂直连接（例如，它们包含相同数量的列，行按顺序堆叠）。DataFrame 也可以水平连接（如果它们包含相同数量的行），通过指定`axis`参数来完成。请注意，列名和行名应在所有 DataFrame 之间相互对应；如果不对应，则会形成新的列，并为任何缺失的值插入 NaN。

首先，我们创建一个新的 DataFrame 名称，`df2`：

```py
df2 = pd.DataFrame({
    'col3': ['a', 'b', 'c', 'd'],
    'new_col1': '',
    'new_col2': 0,
    'new_col3': [11, 13, 15, 17],
    'new_col4': [17, 19, 21, 23],
    'new_col5': [7.5, 8.5, 9.5, 10.5],
    'new_col6': [13, 14, 15, 16]
});
print(df2)
```

输出结果如下：

```py
  col3 new_col1  new_col2  new_col3  new_col4  new_col5  new_col6
0    a                  0        11        17       7.5        13
1    b                  0        13        19       8.5        14
2    c                  0        15        21       9.5        15
3    d                  0        17        23      10.5        16
```

接下来，我们进行连接操作。我们将可选的`ignore_index`参数设置为`True`，以避免重复的行索引：

```py
df3 = pd.concat([df, df2] ignore_index = True)
print(df3)
```

输出结果如下：

```py
  col3 new_col1  new_col2  new_col3  new_col4  new_col5  new_col6
0    x                  0         5         5       7.0        10
1    y                  0         7         7       8.0        11
2    z                  0         9         9       9.0        12
3    a                  0        11        17       7.5        13
4    b                  0        13        19       8.5        14
5    c                  0        15        21       9.5        15
6    d                  0        17        23      10.5        16
```

# 将 DataFrame 列转换为列表

要将列的内容提取到列表中，可以使用`tolist()`函数。转换为列表后，数据可以通过`for`循环和推导式进行迭代：

```py
my_list = df3['new_col3'].tolist()
print(my_list)
```

输出结果如下：

```py
[5, 7, 9, 11, 13, 15, 17]
```

# 获取和设置 DataFrame 值

`pandas`库提供了两种主要方法来选择性地获取和设置 DataFrame 中的值：`loc`和`iloc`。`loc`方法主要用于**基于标签的索引**（例如，使用索引/列名识别行/列），而`iloc`方法主要用于**基于整数的索引**（例如，使用行/列在 DataFrame 中的整数位置来识别）。您希望访问的行和列的具体标签/索引通过方括号紧跟在 DataFrame 名称后面提供，行标签/索引位于列标签/索引之前，并由逗号分隔。让我们来看一些示例。

# 使用基于标签的索引（loc）获取/设置值

DataFrame 的`.loc`属性用于通过条目的标签选择值。它可以用于从 DataFrame 中检索单个标量值（使用行列的单个字符串标签），或从 DataFrame 中检索多个值（使用行/列标签的列表）。还可以将单索引和多索引结合使用，从单行或单列中获取多个值。以下代码行演示了如何从`df` DataFrame 中检索单个标量值：

```py
value = df3.loc[0,'new_col5']
print(value)
```

输出结果为`7.0`。

也可以使用`.loc`属性和等号设置单个/多个值：

```py
df3.loc[[2,3,4],['new_col4','new_col5']] = 1
print(df3)
```

输出结果如下：

```py
  col3 new_col1  new_col2  new_col3  new_col4  new_col5  new_col6
0    x                  0         5         5       7.0        10
1    y                  0         7         7       8.0        11
2    z                  0         9         1       1.0        12
3    a                  0        11         1       1.0        13
4    b                  0        13         1       1.0        14
5    c                  0        15        21       9.5        15
6    d                  0        17        23      10.5        16
```

# 使用基于整数标签的 iloc 获取/设置值

`.iloc`属性与`.loc`属性非常相似，只是它使用被访问的行和列的整数位置，而不是它们的标签。在以下示例中，第 101 行（不是第 100 行，因为索引从 0 开始）和第 100 列的值被转移到`scalar_value`中：

```py
value2 = df3.iloc[0,5]
print(value2)
```

输出结果为`7.0`。

请注意，与`.loc`类似，包含多个值的列表可以传递给`.iloc`属性，以一次性更改 DataFrame 中的多个条目。

# 使用切片获取/设置多个连续的值

有时，我们希望获取或设置的多个值恰好位于相邻（连续）的列中。在这种情况下，我们可以在方括号内使用**切片**来选择多个值。通过切片，我们指定希望访问的数据的起始点和终止点。我们可以在`.loc`和`.iloc`中使用切片，尽管使用整数和`.iloc`的切片更为常见。以下代码行展示了如何通过切片从 DataFrame 中提取部分内容（我们也可以使用等号进行赋值）。请注意，切片也可以用于访问列表和元组中的值（如本章之前所述）：

```py
partial_df3 = df3.loc[1:3,'new_col2':'new_col4']
print(partial_df3)
```

输出如下：

```py
   new_col2  new_col3  new_col4
1         0         7         7
2         0         9         1
3         0        11         1
```

# 使用`at`和`iat`快速获取/设置标量值

如果我们确定只希望获取/设置 DataFrame 中的单个值，可以使用 `.at` 和 `.iat` 属性，分别配合单一标签/整数。只需记住，`.iloc` 和 `.iat` 中的 `i` 代表“整数”：

```py
value3 = df3.iat[3,3]
print(value3)
```

输出结果是`11`。

# 其他操作

另外两个常见的操作是使用布尔条件筛选行和排序行。这里我们将回顾每个操作。

# 使用布尔索引筛选行

到目前为止，我们已经讨论了如何使用标签、整数和切片来选择 DataFrame 中的值。有时，选择符合特定条件的某些行会更加方便。例如，如果我们希望将分析限制在年龄大于或等于 50 岁的人群中。

pandas DataFrame 支持**布尔索引**，即使用布尔值的向量进行索引，以指示我们希望包含哪些值，前提是布尔向量的长度等于 DataFrame 中的行数。由于涉及 DataFrame 列的条件语句正是这样，我们可以使用此类条件语句来索引 DataFrame。在以下示例中，`df` DataFrame 被筛选，只包括`age`列值大于或等于`50`的行：

```py
df3_filt = df3[df3['new_col3'] > 10]
print(df3_filt)
```

输出如下：

```py
  col3 new_col1  new_col2  new_col3  new_col4  new_col5  new_col6
3    a                  0        11         1       1.0        13
4    b                  0        13         1       1.0        14
5    c                  0        15        21       9.5        15
6    d                  0        17        23      10.5        16
```

条件语句可以使用逻辑运算符如`|`或`&`进行链式连接。

# 排序行

如果你希望按某一列的值对 DataFrame 进行排序，可以使用 `sort_values()` 函数；只需将列名作为第一个参数传递即可。`ascending`是一个可选参数，允许你指定排序方向：

```py
df3 = df3.sort_values('new_col4', ascending=True)
print(df3)
```

输出如下：

```py
  col3 new_col1  new_col2  new_col3  new_col4  new_col5  new_col6
2    z                  0         9         1       1.0        12
3    a                  0        11         1       1.0        13
4    b                  0        13         1       1.0        14
0    x                  0         5         5       7.0        10
1    y                  0         7         7       8.0        11
5    c                  0        15        21       9.5        15
6    d                  0        17        23      10.5        16
```

# 类似 SQL 的操作

对于那些习惯于在 SQL 中处理异构类型表格的人来说，转向使用 Python 进行类似的分析可能看起来是一项艰巨的任务。幸运的是，有许多 `pandas` 函数可以结合使用，从而得到与常见 SQL 查询相似的结果，使用诸如分组和连接等操作。`pandas` 文档中甚至有一个子部分（[`pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html`](https://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html)）描述了如何使用 `pandas` DataFrame 执行类似 SQL 的操作。在本节中，我们提供了两个这样的示例。

# 获取聚合行计数

有时，您可能希望获取某一列中特定值的出现次数或统计。例如，您可能拥有一个医疗保健数据集，想要知道在患者就诊期间，特定支付方式被使用了多少次。在 SQL 中，您可以编写一个查询，使用 `GROUP BY` 子句与聚合函数（在本例中为 `COUNT(*)`）结合，来获取支付方式的统计信息：

```py
SELECT payment, COUNT(*)
FROM data
GROUP BY payment;
```

在`pandas`中，您可以通过将`groupby()`和`size()`函数链式调用来实现相同的结果：

```py
tallies = df3.groupby('new_col4').size()
print(tallies)
```

输出如下：

```py
new_col4
1     3
5     1
7     1
21    1
23    1
dtype: int64
```

# 连接 DataFrame

在第四章，*计算基础 - 数据库*中，我们讨论了使用 `JOIN` 操作合并来自两个数据库表的数据。要使用 JOIN 操作，您需要指定两个表的名称，以及 JOIN 的类型（左、右、外部或内部）和连接的列：

```py
SELECT *
FROM left_table OUTER JOIN right_table
ON left_table.index = right_table.index;
```

在 pandas 中，您可以使用 `merge()` 或 `join()` 函数来实现表连接。默认情况下，`join()` 函数基于表的索引连接数据；但是，可以通过指定 `on` 参数来使用其他列。如果连接的两个表中有重复的列名，您需要指定 `rsuffix` 或 `lsuffix` 参数，以重命名列，使它们不再具有相同的名称：

```py
df_join_df2 = df.join(df2, how='outer', rsuffix='r')
print(df_join_df2)
```

输出如下（请注意第 3 行中的`NaN`值，这是`df`中不存在的一行）：

```py
  col3 new_col1  new_col2  new_col3  new_col4  new_col5  new_col6 col3r  \
0    x                0.0       5.0       5.0       7.0      10.0     a   
1    y                0.0       7.0       7.0       8.0      11.0     b   
2    z                0.0       9.0       9.0       9.0      12.0     c   
3  NaN      NaN       NaN       NaN       NaN       NaN       NaN     d   

  new_col1r  new_col2r  new_col3r  new_col4r  new_col5r  new_col6r  
0                    0         11         17        7.5         13  
1                    0         13         19        8.5         14  
2                    0         15         21        9.5         15  
3                    0         17         23       10.5         16 
```

# scikit-learn 简介

整本书都围绕**scikit-learn**（[`scikit-learn.org/stable/`](http://scikit-learn.org/stable/)）展开。scikit-learn 库包含许多子模块。本书将只使用其中的一些子模块（在第七章，*在医疗保健中创建预测模型*）。例如，包括 `sklearn.linear_model` 和 `sklearn.ensemble` 子模块。在这里，我们将概述一些更常用的子模块。为了方便起见，我们已将相关模块分组为数据科学管道的各个部分，这些部分在第一章，*医疗保健分析简介*中讨论过。

# 示例数据

scikit-learn 在`sklearn.datasets`子模块中包含了几个示例数据集。至少有两个数据集，`sklearn.datasets.load_breast_cancer`和`sklearn.datasets.load_diabetes`，是与健康相关的。这些数据集已经预处理过，且规模较小，仅包含几十个特征和几百个患者。在第七章《医疗保健中的预测模型制作》中，我们使用的数据要大得多，且更像现代医疗机构提供的数据。然而，这些示例数据集对于实验 scikit-learn 功能仍然非常有用。

# 数据预处理

数据预处理功能存在于`sklearn.preprocessing`子模块中，其他相关功能在以下章节中讨论。

# 分类变量的独热编码

几乎每个数据集都包含一些分类数据。**分类数据**是离散数据，其中值可以取有限数量的可能值（通常编码为“字符串”）。由于 Python 的 scikit-learn 只能处理数值数据，因此在使用 scikit-learn 进行机器学习之前，我们必须找到其他方法来对分类变量进行编码。

使用**独热编码**，也称为**1-of-K 编码方案**，一个具有*k*个可能值的单一分类变量被转换为*k*个不同的二元变量，每个二元变量仅在该观测值的列值等于它所代表的值时为正。在第七章《医疗保健中的预测模型制作》中，我们提供了独热编码的详细示例，并使用 pandas 的`get_dummies()`函数对真实的临床数据集进行独热编码。scikit-learn 也有一个类可以用于执行独热编码，这个类是`sklearn.preprocessing`模块中的`OneHotEncoder`类。

关于如何使用`OneHotEncoder`的说明，可以访问 scikit-learn 文档：[`scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features`](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)。

# 缩放和中心化

对于一些机器学习算法，通常建议不仅转换类别变量（使用之前讨论过的独热编码），还需要转换连续变量。回顾第一章，*《医疗分析导论》*中提到，连续变量是数值型的，可以取任何有理数值（尽管在许多情况下它们限制为整数）。一个特别常见的做法是**标准化**每个连续变量，使得*变量的均值为零，标准差为一*。例如，考虑`AGE`变量：它通常范围从 0 到 100 左右，均值可能约为 40。假设对于某个人群，`AGE`变量的均值为 40，标准差为 20。如果我们对`AGE`变量进行中心化和重缩放，年龄为 40 的人在转换后的变量中将表示为零。年龄为 20 岁的人将表示为-1，年龄为 60 岁的人将表示为 1，年龄为 80 岁的人将表示为 2，年龄为 50 岁的人将表示为 0.5。这种转换可以防止具有更大范围的变量在机器学习算法中被过度表示。

scikit-learn 有许多内置类和函数，用于数据的中心化和缩放，包括`sklearn.preprocessing.StandardScaler()`、`sklearn.preprocessing.MinMaxScaler()`和`sklearn.preprocessing.RobustScaler()`。这些不同的工具专门用于处理不同类型的连续数据，如正态分布变量或具有许多异常值的变量。

有关如何使用缩放类的说明，您可以查看 scikit-learn 文档：[`scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling`](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)。

# 二值化

**二值化**是另一种转换方法，它将连续变量转换为二进制变量。例如，如果我们有一个名为`AGE`的连续变量，我们可以通过设置年龄 50 岁为阈值，对年龄进行二值化，将 50 岁及以上的年龄设为 1，将低于 50 岁的年龄设为 0。二值化在处理有大量变量时节省了时间和内存；然而，实际上，原始的连续值通常表现得更好，因为它们包含更多的信息。

虽然也可以使用之前演示过的代码在 pandas 中执行二值化，但 scikit-learn 提供了一个`Binarizer`类，也可以用来对特征进行二值化。有关如何使用`Binarizer`类的说明，您可以访问[`scikit-learn.org/stable/modules/preprocessing.html#binarization`](http://scikit-learn.org/stable/modules/preprocessing.html#binarization)。

# 填充

在第一章，*医疗分析入门*中，我们提到了处理缺失数据的重要性。**插补**是处理缺失值的一种策略，通过用基于现有数据估算的值来填补缺失值。在医疗领域，常见的两种插补方法是**零插补**，即将缺失数据视为零（例如，如果某一诊断值为 `NULL`，很可能是因为该信息没有出现在病历中）；以及**均值插补**，即将缺失数据视为现有数据分布的均值（例如，如果某患者缺少年龄，我们可以将其插补为 40）。我们在第四章，*计算基础——数据库*中演示了各种插补方法，我们将在第七章，*在医疗保健中构建预测模型*中编写我们自己的插补函数。

Scikit-learn 提供了一个 `Imputer` 类来执行不同类型的插补。你可以在 [`scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values`](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values) 查看如何使用它的详细信息。

# 特征选择

在机器学习中，常常有一种误解，认为数据越多越好。对于观测数据（例如，数据集中的行数），这种说法通常是正确的。然而，对于特征来说，更多的特征并不总是更好。在某些情况下，使用较少的特征可能反而表现得更好，因为多个高度相关的特征可能会对预测产生偏差，或者特征的数量超过了观测值的数量。

在其他情况下，性能可能与使用一半特征时相同，或者稍微差一点，但较少的特征可能因为多种原因而更为可取，包括时间考虑、内存可用性，或者便于向非技术相关人员解释和解释。无论如何，通常对数据进行特征选择是一个好主意。即使你不打算删除任何特征，进行特征选择并对特征重要性进行排序，也能为你提供对模型的深入洞察，帮助理解其预测行为和性能。

`sklearn.feature_selection` 模块中有许多类和函数是为特征选择而构建的，不同的类集合对应于不同的特征选择方法。例如，单变量特征选择涉及测量每个预测变量与目标变量之间的统计依赖性，这可以通过 `SelectKBest` 或 `SelectPercentile` 类等实现。`VarianceThreshold` 类移除在观测值中方差较低的特征，例如那些几乎总是为零的特征。而 `SelectFromModel` 类在模型拟合后，修剪那些不满足一定强度要求（无论是系数还是特征重要性）的特征。

要查看 scikit-learn 中所有特征选择类的完整列表，请访问[`scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection`](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)。

# 机器学习算法

机器学习算法提供了一种数学框架，用于对新的观测值进行预测。scikit-learn 支持数十种不同的机器学习算法，这些算法具有不同的优缺点。我们将在这里简要讨论一些算法及其相应的 scikit-learn API 功能。我们将在第七章中使用这些算法，*在医疗保健中构建预测模型*。

# 广义线性模型

正如我们在第三章中讨论的，*机器学习基础*，**线性模型**可以简单地理解为特征的加权组合（例如加权和），用于预测目标值。特征由观测值决定；每个特征的权重由模型决定。线性回归预测连续变量，而逻辑回归可以看作是线性回归的扩展形式，其中预测的目标值经过**logit 变换**，转化为一个范围在零到一之间的变量。这种变换对于执行二分类任务非常有用，比如当有两个可能的结果时。

在 scikit-learn 中，这两种算法由 `sklearn.linear_model.LogisticRegression` 和 `sklearn.linear_model.LinearRegression` 类表示。我们将在第七章中演示逻辑回归，*在医疗保健中构建预测模型*。

# 集成方法

**集成方法**涉及使用不同机器学习模型的组合来进行预测。例如，**随机森林**是由多个决策树分类器组成的集合，这些树通过为每棵树选择和使用特定的特征集来实现去相关。此外，**AdaBoost**是一种算法，它通过在数据上拟合许多弱学习器来做出有效预测。这些算法由`sklearn.ensemble`模块提供支持。

# 其他机器学习算法

其他一些流行的机器学习算法包括朴素贝叶斯算法、k-近邻算法、神经网络、决策树和支持向量机。这些算法在 scikit-learn 中分别由`sklearn.naive_bayes`、`sklearn.neighbors`、`sklearn.neural_network`、`sklearn.tree`和`sklearn.svm`模块提供支持。在第七章 *在医疗保健中构建预测模型*中，我们将使用临床数据集构建神经网络模型。

# 性能评估

最后，一旦我们使用所需的算法构建了模型，衡量其性能就变得至关重要。`sklearn.metrics`模块对于这一点非常有用。如第三章 *机器学习基础*中所讨论的，混淆矩阵对于分类任务特别重要，并且由`sklearn.metrics.confusion_matrix()`函数支持。确定接收者操作特征（ROC）曲线并计算**曲线下面积**（**AUC**）可以分别通过`sklearn.metrics.roc_curve()`和`sklearn.metrics.roc_auc_score()`函数完成。精确率-召回率曲线是 ROC 曲线的替代方法，特别适用于不平衡数据集，并且由`sklearn.metrics.precision_recall_curve()`函数提供支持。

# 其他分析库

在这里，我们提到三个常用于分析的主要包：NumPy、SciPy 和 matplotlib。

# NumPy 与 SciPy

**NumPy** ([www.numpy.org](http://www.numpy.org/)) 是 Python 的矩阵库。通过使用`numpy.array()`及类似的构造，可以创建大型矩阵并对其进行各种数学操作（包括矩阵加法和乘法）。NumPy 还具有许多用于操作矩阵形状的函数。NumPy 的另一个特点是提供了熟悉的数学函数，如`sin()`、`cos()`和`exp()`。

**SciPy** ([www.scipy.org](http://www.scipy.org)) 是一个包含许多高级数学模块的工具箱。与机器学习相关的子包包括`cluster`、`stats`、`sparse`和`optimize`。SciPy 是一个重要的包，它使 Python 能够进行科学计算。

# matplotlib

**matplotlib** ([`matplotlib.org`](https://matplotlib.org)) 是一个流行的 Python 二维绘图库。根据其官网介绍，用户“只需几行代码就可以生成图表、直方图、功率谱、条形图、误差图、散点图等。”它的绘图库提供了丰富的选项和功能，支持高度自定义。

# 总结

在本章中，我们快速浏览了基础的 Python 语言，以及两个在数据分析中非常重要的 Python 库：pandas 和 scikit-learn。我们现在已经完成了本书的基础章节。

在第六章《衡量医疗质量》中，我们将深入探讨一些真实的医疗服务提供者的表现数据，并使用 pandas 进行分析。
