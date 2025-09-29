# 第四章。复杂 ArcPy 脚本和泛化函数

在本章中，我们将从基于 ModelBuilder 自动生成的简单脚本创建转向包含高级 Python 和 ArcPy 概念的复杂脚本，例如函数。函数可以在编写脚本时提高代码效率和节省时间。当创建模块或其他可重用代码时，它们也非常有用，允许将标准编程操作脚本化并准备好供将来使用。

本章将涵盖以下主题：

+   创建函数以避免代码重复

+   创建辅助函数以处理 ArcPy 的限制

+   将函数泛化以使其可重用

# Python 函数–避免代码重复

编程语言共享一个几十年来帮助程序员的观念：函数。从广义上讲，函数的概念是创建代码块，将对数据进行操作，根据程序员的所需对其进行转换，并将转换后的数据返回到代码的主体部分。在前几章中，我们已经介绍了 Python 的一些内置函数，例如 `int` 函数，它将字符串或浮点数转换为整数；现在是我们编写自己的函数的时候了。

函数之所以被使用，是因为它们在编程中解决了许多不同的需求。函数减少了编写重复代码的需要，从而减少了创建脚本所需的时间。它们可以用来创建数字范围（`range()` 函数），或者确定列表的最大值（`max` 函数），或者创建一个 SQL 语句来从要素类中选择一组行。它们甚至可以被复制并用于另一个脚本，或者作为可以导入到脚本中的模块的一部分。函数的重用不仅使编程更有用，而且减少了繁琐的工作。当脚本编写者开始编写函数时，这是将编程作为 GIS 工作流程一部分的重大步骤。

## 函数的技术定义

函数，在其他编程语言中也称为子程序或过程，是一段代码块，其设计目的是接受输入数据并对其进行转换，或者在没有输入要求的情况下调用时，向主程序提供数据。在理论上，函数只会转换提供给函数作为参数的数据；它不应更改函数中未包含的脚本的其他任何部分。为了实现这一点，引入了命名空间的概念。如第一章中所述，“ArcGIS 的 Python 简介”，命名空间用于隔离脚本中的变量；变量要么是全局的，可以在脚本的主体以及函数中使用，要么是局部的，仅在函数内部可用。

命名空间使得在函数内部使用变量名成为可能，并允许它表示一个值，同时也在脚本的另一部分使用相同的变量名。当从其他程序员那里导入模块时，这一点尤为重要；在该模块及其函数内部，它包含的变量可能具有与主脚本中变量名相同的名称。

在像 Python 这样的高级编程语言中，有内置对函数的支持，包括定义函数名称和数据输入（也称为参数）的能力。函数是通过使用`def`关键字加上函数名称以及可能包含或不包含参数的括号来创建的。参数也可以定义默认值，因此只有当参数与默认值不同时才需要传递给函数。函数返回的值也容易定义。

## 第一个函数

让我们创建一个函数，以了解编写函数时可能实现的功能。首先，我们需要通过提供`def`关键字和括号内的名称来调用函数。当调用`firstFunction()`时，它将返回一个字符串：

```py
def firstFunction():
 'a simple function returning a string'
 return "My First Function"
>>>firstFunction()

```

输出如下：

```py
'My First Function'

```

注意，这个函数有一个文档字符串或 doc 字符串（一个简单返回字符串的函数），它描述了函数的功能；这个字符串可以在以后被调用，以了解函数的功能，使用`__doc__`内部函数：

```py
>>>print firstFunction.__doc__

```

输出如下：

```py
'a simple function returning a string' 

```

函数被定义并赋予了一个名称，然后添加括号并跟一个冒号。接下来的行必须进行缩进（一个好的 IDE 会自动添加缩进）。该函数没有任何参数，因此括号是空的。然后函数使用`return`关键字从函数返回一个值，在这种情况下是一个字符串。

接下来，通过在函数名称后添加括号来调用函数。当它被调用时，它将执行它被指示执行的操作：返回一个字符串。

## 带参数的函数

现在，让我们创建一个接受参数并根据需要转换它们的函数。这个函数将接受一个数字并将其乘以 3：

```py
def secondFunction(number):
 'this function multiples numbers by 3'
 return number *3
>>> secondFunction(4)

```

输出如下：

```py
12

```

然而，该函数有一个缺点；无法保证传递给函数的值是一个数字。我们需要在函数中添加一个条件，以确保它不会抛出异常：

```py
def secondFunction(number):
 'this function multiples numbers by 3'
 if type(number) == type(1) or type(number) == type(1.0):
 return number *3
>>> secondFunction(4.0)

```

输出如下：

```py
12.0
>>>secondFunction(4)

```

输出如下：

```py
12
>>>secondFunction("String")
>>> 

```

现在函数接受一个参数，检查它的数据类型，并返回参数的倍数，无论是整数还是函数。如果它是一个字符串或其他数据类型，如最后一个示例所示，则不返回任何值。

我们应该讨论的简单函数的另一个调整是参数默认值。通过在函数定义中包含默认值，我们避免了提供很少更改的参数。例如，如果我们想在简单函数中使用不同于 3 的乘数，我们可以这样定义它：

```py
def thirdFunction(number, multiplier=3):
 'this function multiples numbers by 3'
 if type(number) == type(1) or type(number) == type(1.0):
 return number *multiplier
>>>thirdFunction(4)

```

输出如下：

```py
12
>>>thirdFunction(4,5)

```

输出如下：

```py
20

```

当只提供要乘以的数字时，该函数将正常工作，因为乘数默认值为 3。然而，如果我们需要另一个乘数，可以在调用函数时添加另一个值来调整其值。请注意，第二个值不必是数字，因为它没有类型检查。此外，函数中的默认值（s）必须跟在无默认值的参数之后（或者所有参数都可以有默认值，并且可以按顺序或按名称向函数提供参数）。

这些简单的函数结合了我们之前章节中讨论的许多概念，包括内置函数如 `type`、`条件语句`、`参数`、`参数默认值` 和 `函数返回值`。我们现在可以继续使用 ArcPy 创建函数。

## 使用函数替换重复代码

函数的主要用途之一是确保相同的代码不必反复编写。让我们回到上一章的例子，并将脚本中的代码转换为函数，以便能够对旧金山的任何公交线路进行相同的分析。

我们可以将脚本的第一部分转换为函数的是三个 ArcPy 函数。这样做将使脚本适用于 Bus Stop 特征类中的任何站点，并具有可调整的缓冲距离：

```py
bufferDist = 400
buffDistUnit  = "Feet"
lineName = '71 IB'
busSignage = 'Ferry Plaza'
sqlStatement = "NAME = '{0}' AND BUS_SIGNAG = '{1}'"
def selectBufferIntersect(selectIn,selectOut,bufferOut,intersectIn, intersectOut, sqlStatement, bufferDist, buffDistUnit, lineName, busSignage): 
 'a function to perform a bus stop analysis'
 arcpy.Select_analysis(selectIn, selectOut, sqlStatement.format(lineName, busSignage))
 arcpy.Buffer_analysis(selectOut, bufferOut, "{0} {1}".format(bufferDist), "FULL", "ROUND", "NONE", "")
 arcpy.Intersect_analysis("{0} #;{1} #".format(bufferOut, intersectIn), intersectOut, "ALL", "", "INPUT")
 return intersectOut

```

此函数演示了如何调整分析以接受输入和输出特征类变量作为参数，以及一些新变量。

函数添加了一个变量来替换 SQL 语句，以及变量来调整公交站，并且调整了缓冲距离语句，以便可以调整距离和单位。在脚本中先前定义的特征类名称变量都已替换为局部变量名；虽然全局变量名可以保留，但这会降低函数的可移植性。

下一个函数将接受 `selectBufferIntersect()` 函数的结果，并使用搜索游标对其进行搜索，将结果传递到字典中。然后，该函数将返回字典供以后使用：

```py
def createResultDic(resultFC):
 'search results of analysis and create results dictionary' 
 dataDictionary = {} 
 with arcpy.da.SearchCursor(resultFC, ["STOPID","POP10"]) as cursor:
 for row in cursor:
 busStopID = row[0]
 pop10 = row[1]
 if busStopID not in dataDictionary.keys():
 dataDictionary[busStopID] = [pop10]
 else:
 dataDictionary[busStopID].append(pop10)
 return dataDictionary

```

此函数仅需要一个参数：来自 `searchBufferIntersect()` 函数返回的特征类。首先创建一个持有字典的结果，然后通过搜索游标填充，使用 `busStopid` 属性作为键，并将人口普查区块人口属性添加到分配给该键的列表中。

字典在被填充了排序后的数据后，从函数中返回，用于最终函数`createCSV()`。这个函数接受字典和输出 CSV 文件的名称作为字符串：

```py
def createCSV(dictionary, csvname):
  'a function takes a dictionary and creates a CSV file'
    with open(csvname, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for busStopID in dictionary.keys():
            popList = dictionary[busStopID]
            averagePop = sum(popList)/len(popList)
            data = [busStopID, averagePop]
            csvwriter.writerow(data)
```

最终的函数使用`csv`模块创建 CSV 文件。现在文件名，一个字符串，是一个可定制参数（这意味着脚本名称可以是任何有效的文件路径和具有`.csv`扩展名的文本文件）。`csvfile`参数传递给 CSV 模块的 writer 方法，并分配给变量`csvwriter`，然后访问和处理字典，并将其作为列表传递给`csvwriter`以写入`CSV`文件。`csv.writer()`方法将列表中的每个项目处理成 CSV 格式，并保存最终结果。使用 Excel 或记事本等文本编辑器打开`CSV`文件。

要运行这些函数，我们将在脚本中调用它们，在函数定义之后：

```py
analysisResult = selectBufferIntersect(Bus_Stops,Inbound71, Inbound71_400ft_buffer,CensusBlocks2010, Intersect71Census, bufferDist, lineName, busSignage )
dictionary = createResultDic(analysisResult)
createCSV(dictionary,r'C:\Projects\Output\Averages.csv')

```

现在，脚本已经被分为三个函数，它们替换了第一个修改后的脚本中的代码。修改后的脚本看起来像这样：

```py
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 8662_Chapter4Modified1.py
# Created on: 2014-04-22 21:59:31.00000
#   (generated by ArcGIS/ModelBuilder)
# Description: 
# Adjusted by Silas Toms
# 2014 05 05
# ---------------------------------------------------------------------------

# Import arcpy module
import arcpy
import csv

# Local variables:
Bus_Stops = r"C:\Projects\PacktDB.gdb\SanFrancisco\Bus_Stops"
CensusBlocks2010 = r"C:\Projects\PacktDB.gdb\SanFrancisco\CensusBlocks2010"
Inbound71 = r"C:\Projects\PacktDB.gdb\Chapter3Results\Inbound71"
Inbound71_400ft_buffer = r"C:\Projects\PacktDB.gdb\Chapter3Results\Inbound71_400ft_buffer"
Intersect71Census = r"C:\Projects\PacktDB.gdb\Chapter3Results\Intersect71Census"
bufferDist = 400
lineName = '71 IB'
busSignage = 'Ferry Plaza'
def selectBufferIntersect(selectIn,selectOut,bufferOut,intersectIn,
 intersectOut, bufferDist,lineName, busSignage ):
 arcpy.Select_analysis(selectIn, 
 selectOut, 
 "NAME = '{0}' AND BUS_SIGNAG = '{1}'".format(lineName, busSignage))
 arcpy.Buffer_analysis(selectOut, 
 bufferOut, 
 "{0} Feet".format(bufferDist), 
 "FULL", "ROUND", "NONE", "")
 arcpy.Intersect_analysis("{0} #;{1} #".format(bufferOut,intersectIn), 
 intersectOut, "ALL", "", "INPUT")
 return intersectOut

def createResultDic(resultFC):
 dataDictionary = {}

 with arcpy.da.SearchCursor(resultFC, 
 ["STOPID","POP10"]) as cursor:
 for row in cursor:
 busStopID = row[0]
 pop10 = row[1]
 if busStopID not in dataDictionary.keys():
 dataDictionary[busStopID] = [pop10]
 else:
 dataDictionary[busStopID].append(pop10)
 return dataDictionary

def createCSV(dictionary, csvname):
 with open(csvname, 'wb') as csvfile:
 csvwriter = csv.writer(csvfile, delimiter=',')
 for busStopID in dictionary.keys():
 popList = dictionary[busStopID]
 averagePop = sum(popList)/len(popList)
 data = [busStopID, averagePop]
 csvwriter.writerow(data)
analysisResult = selectBufferIntersect(Bus_Stops,Inbound71, Inbound71_400ft_buffer,CensusBlocks2010,Intersect71Census, bufferDist,lineName, busSignage )
dictionary = createResultDic(analysisResult)
createCSV(dictionary,r'C:\Projects\Output\Averages.csv')
print "Data Analysis Complete"

```

函数的进一步泛化，虽然我们已经从原始脚本中创建了可以用来提取更多关于旧金山公交车站数据的函数，但我们的新函数仍然非常特定于它们被创建的数据集和分析。这对于那些不需要创建可重用函数的漫长而繁重分析来说非常有用。函数的第一个用途是消除重复代码的需要。接下来的目标是使代码可重用。让我们讨论一些方法，我们可以将这些函数从一次性函数转换为可重用函数甚至模块。

首先，让我们检查第一个函数：

```py
def selectBufferIntersect(selectIn,selectOut,bufferOut,intersectIn,
 intersectOut, bufferDist,lineName, busSignage ):
 arcpy.Select_analysis(selectIn, 
 selectOut, 
 "NAME = '{0}' AND BUS_SIGNAG = '{1}'".format(lineName, busSignage))
 arcpy.Buffer_analysis(selectOut, 
 bufferOut, 
 "{0} Feet".format(bufferDist), 
 "FULL", "ROUND", "NONE", "")
 arcpy.Intersect_analysis("{0} #;{1} #".format(bufferOut,intersectIn), 
 intersectOut, "ALL", "", "INPUT")
 return intersectOut

```

这个函数似乎非常特定于公交车站分析。实际上，它如此特定，以至于尽管有几种方法可以调整它使其更通用（即，在其他可能不涉及相同步骤的脚本中也有用），我们不应该将其转换为单独的函数。当我们创建一个单独的函数时，我们引入了太多的变量到脚本中，试图简化它，这是一种事倍功半的努力。相反，让我们专注于泛化 ArcPy 工具本身的方法。

第一步将是将三个 ArcPy 工具分开，并检查每个工具可以调整什么。Select 工具应该调整以接受字符串作为 SQL 选择语句。SQL 语句可以由另一个函数生成，或者通过在运行时接受的参数生成（例如，通过 Script 工具传递给脚本，这将在下一章中讨论）。

例如，如果我们想使脚本接受每次运行脚本时多个公交车站（例如，每条线路的进站和出站车站），我们可以创建一个函数，该函数接受所需车站的列表和 SQL 模板，并返回一个 SQL 语句，可以插入到 Select 工具中。以下是一个示例：

```py
def formatSQLIN(dataList, sqlTemplate):
 'a function to generate a SQL statement'
 sql = sqlTemplate #"OBJECTID IN "
 step = "("
 for data in dataList:
 step += str(data)
 sql += step + ")"
 return sql

def formatSQL(dataList, sqlTemplate):
 'a function to generate a SQL statement'
 sql = ''
 for count, data in enumerate(dataList):
 if count != len(dataList)-1:
 sql += sqlTemplate.format(data) + ' OR '
 else:
 sql += sqlTemplate.format(data)
 return sql

```

```py
>>> dataVals = [1,2,3,4]
>>> sqlOID = "OBJECTID = {0}"
>>> sql = formatSQL(dataVals, sqlOID)
>>> print sql

```

输出如下：

```py
OBJECTID = 1 OR OBJECTID = 2 OR OBJECTID = 3 OR OBJECTID = 4
```

这个新的函数`formatSQL()`是一个非常实用的函数。让我们通过将函数与其后的结果进行比较来回顾它所做的工作。该函数被定义为接受两个参数：一个值列表和一个 SQL 模板。第一个局部变量是空字符串`sql`，它将通过字符串添加来添加。该函数旨在将值插入到变量`sql`中，通过使用字符串格式化将它们添加到模板中，然后将模板添加到 SQL 语句字符串中（注意`sql +=`等同于`sql = sql +`）。此外，使用一个运算符（`OR`）使 SQL 语句包含所有匹配该模式的数据行。此函数使用内置的`enumerate`函数来计数列表的迭代次数；一旦它达到了列表中的最后一个值，运算符就不会添加到 SQL 语句中。

注意，我们还可以向函数添加一个参数，使其能够使用`AND`运算符而不是`OR`，同时仍然保留`OR`作为默认选项：

```py
def formatSQL2(dataList, sqlTemplate, operator=" OR "):
 'a function to generate a SQL statement'
 sql = ''
 for count, data in enumerate(dataList):
 if count != len(dataList)-1:
 sql += sqlTemplate.format(data) + operator
 else:
 sql += sqlTemplate.format(data)
 return sql

>>> sql = formatSQL2(dataVals, sqlOID," AND ")
>>> print sql

```

输出如下：

```py
OBJECTID = 1 AND OBJECTID = 2 AND OBJECTID = 3 AND OBJECTID = 4

```

虽然在 ObjectIDs 上使用`AND`运算符没有意义，但还有其他情况下使用它是有意义的，因此保留`OR`作为默认选项，同时允许使用`AND`。无论如何，这个函数现在可以用来生成我们的多站公交车站 SQL 语句（现在忽略公交标志字段）：

```py
>>> sqlTemplate = "NAME = '{0}'"
>>> lineNames = ['71 IB','71 OB']
>>> sql = formatSQL2(lineNames, sqlTemplate)
>>> print sql

```

输出如下：

```py
NAME = '71 IB' OR NAME = '71 OB'

```

然而，我们不能忽视进站线路的公交标志字段，因为该线路有两个起点，所以我们需要调整函数以接受多个值：

```py
def formatSQLMultiple(dataList, sqlTemplate, operator=" OR "):
    'a function to generate a SQL statement'
    sql = ''
    for count, data in enumerate(dataList):
        if count != len(dataList)-1:
            sql += sqlTemplate.format(*data) + operator
        else:
            sql += sqlTemplate.format(*data)
    return sql
```

```py
>>> sqlTemplate = "(NAME = '{0}' AND BUS_SIGNAG = '{1}')"
>>> lineNames = [('71 IB', 'Ferry Plaza'),('71 OB','48th Avenue')]
>>> sql = formatSQLMultiple(lineNames, sqlTemplate)
>>> print sql

```

输出如下：

```py
(NAME = '71 IB' AND BUS_SIGNAG = 'Ferry Plaza') OR (NAME = '71 OB' AND BUS_SIGNAG = '48th Avenue')

```

这个函数的细微差别在于数据变量前的星号，它允许数据变量内的值通过解包元组被正确地格式化到 SQL 模板中。注意，SQL 模板已被创建，通过使用括号来分隔每个条件。现在函数（们）可以重用，SQL 语句现在可以插入到选择工具中：

```py
sql = formatSQLMultiple(lineNames, sqlTemplate)
arcpy.Select_analysis(Bus_Stops, Inbound71, sql)
```

接下来是缓冲工具。我们已经通过添加一个距离变量来使其通用化。在这种情况下，我们只需再添加一个变量，即单位变量，这将使缓冲单元可以从英尺调整到米或其他任何允许的单位。我们将保留其他默认设置不变。

这里是调整后的缓冲工具版本：

```py
bufferDist = 400
bufferUnit = "Feet"
arcpy.Buffer_analysis(Inbound71, 
 Inbound71_400ft_buffer, 
 "{0} {1}".format(bufferDist, bufferUnit), 
 "FULL", "ROUND", "NONE", "")

```

现在，缓冲距离和缓冲单元都由前一个脚本中定义的变量控制，如果决定距离不足且可能需要调整变量，这将使其易于调整。

调整 ArcPy 工具的下一步是编写一个函数，该函数将允许使用交集工具将任意数量的特征类相互交集。这个新函数将与之前的`formatSQL`函数类似，因为它们将使用字符串格式化和附加来允许将特征类列表处理成交集工具可以接受的正确字符串格式。然而，由于这个函数将被构建成尽可能通用，它必须设计成可以接受任意数量的要交集的特征类：

```py
def formatIntersect(features):
    'a function to generate an intersect string'
    formatString = ''
    for count, feature in enumerate(features):
        if count != len(features)-1:
            formatString += feature + " #;"
        else:
            formatString += feature + " #"
        return formatString
```

```py
>>> shpNames = ["example.shp","example2.shp"]
>>> iString = formatIntersect(shpNames)
>>> print iString

```

输出结果如下：

```py
example.shp #;example2.shp #

```

现在我们已经编写了`formatIntersect()`函数，需要创建的是要传递给函数的特征类列表。函数返回的字符串然后可以传递给交集工具：

```py
intersected = [Inbound71_400ft_buffer, CensusBlocks2010]
iString = formatIntersect(intersected)
# Process: Intersect
arcpy.Intersect_analysis(iString, 
                         Intersect71Census, "ALL", "", "INPUT")
```

由于我们避免创建仅适用于此脚本或分析的函数，我们现在有两个（或更多）有用的函数可以在后续分析中使用，并且我们知道如何操作 ArcPy 工具以接受我们想要提供给它们的任何数据。

## 函数的更一般化

我们最初创建的用于搜索结果并生成结果电子表格的其他函数，也可以通过一些调整变得更加通用。

如果我们想要生成关于距离公交站点的每个普查区更详细的信息（例如，如果我们有一个包含收入数据和人口数据的普查区数据集），我们需要将一个属性列表传递给函数，以便从最终的特征类中提取这些属性。为了实现这一点，需要调整`createResultDic()`函数以接受这个属性列表：

```py
def createResultDic(resultFC, key, values):
    dataDictionary = {}
    fields = [key]
    fields.extend(values)
    with arcpy.da.SearchCursor(resultFC, fields) as cursor:
        for row in cursor:
            busStopID = row[0]
            data = row[1:]
            if busStopID not in dataDictionary.keys():
                dataDictionary[busStopID] = [data]
            else:
                dataDictionary[busStopID].append(data)
    return dataDictionary
```

这个新的`createResultDic()`函数将为每个公交站生成一个列表的列表（即，每行的值包含在一个列表中，并添加到主列表中），然后可以通过知道列表中每个值的位位置来稍后解析。这种解决方案在需要将数据排序到字典中时很有用。

然而，这是一种不令人满意的结果排序方式。如果字段列表没有传递给字典，并且无法知道列表中数据的顺序怎么办？相反，我们希望能够使用 Python 字典的功能按字段名称对数据进行排序。在这种情况下，我们将使用嵌套字典来创建通过它们包含的数据类型（即，人口、收入或其他字段）可访问的结果列表：

```py
def createResultDic(resultFC, key, values):
    dataDic = {}
    fields = []
   if type(key) == type((1,2)) or type(key) == type([1,2]):
            fields.extend(key)
            length = len(key)
    else:
        fields = [key]
        length = 1
    fields.extend(values)
    with arcpy.da.SearchCursor(resultFC, fields) as cursor:
        for row in cursor:
            busStopID = row[:length]
            data = row[length:]
            if busStopID not in dataDictionary.keys():

                dataDictionary[busStopID] = {}

            for counter,field in enumerate(values):
                if field not in dataDictionary[busStopID].keys():
                    dataDictionary[busStopID][field] = [data[counter]]
                else:
                    dataDictionary[busStopID][field].append(data[counter])
    return dataDictionary
```

```py
>>> rFC = r'C:\Projects\PacktDB.gdb\Chapter3Results\Intersect71Census'
>>> key = 'STOPID'
>>> values = 'HOUSING10','POP10'
>>> dic = createResultDic(rFC, key, values)
>>> dic[1122023]

```

输出结果如下：

```py
{'HOUSING10': [104, 62, 113, 81, 177, 0, 52, 113, 0, 104, 81, 177, 52], 'POP10': [140, 134, 241, 138, 329, 0, 118, 241, 0, 140, 138, 329, 118]}

```

在这个例子中，函数作为参数传递给一个要素类，`STOPID`以及要合并的字段。`fields`变量被创建出来，以便将所需的字段传递给搜索光标。光标返回每一行作为一个元组；元组的第一个成员是`busStopID`，其余的元组是该公交车站相关联的数据。然后，函数使用一个条件来评估该公交车站是否已经被分析过；如果没有，它将被添加到字典中，并分配一个第二级内部字典，该字典将用于存储与该站点相关的结果。通过使用字典，我们可以然后对结果进行排序，并将它们分配给正确的字段。

之前的例子展示了请求一个特定公交车站（`1122023`）的数据结果。由于这里传递了两个字段，数据已经被组织成两组，字段名称现在成为内部字典的键。正因为这种组织方式，我们现在可以为每个字段创建平均值，而不仅仅是单个值。

说到平均值，我们将搜索光标分析结果的平均值计算工作留给了`createCSV()`函数。这也应该避免，因为它通过添加额外的数据处理职责（这些职责应该由另一个函数负责）降低了`createCSV()`函数的有用性。让我们首先通过调整`createCSV()`函数来解决这个问题：

```py
def createCSV(data, csvname, mode ='ab'):
    with open(csvname, mode) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(data)
```

这是一个简化版的函数，但它非常有用。通过这样调整函数，我们限制它只做两件事：打开 CSV 文件并向其中添加一行数据。因为我们使用了`ab`模式，如果 CSV 文件存在，我们只会向其中添加数据而不是覆盖它（如果不存在，它将被创建）。这种添加模式可以通过传递`wb`作为模式来覆盖，这将每次生成一个新的脚本。

现在我们可以对分析结果进行排序，计算平均值，并将它们传递给我们的新`createCSV`脚本。为此，我们将遍历由`createResultDic()`函数创建的字典：

```py
csvname = r'C:\Projects\Output\Averages.csv'
dataKey = 'STOPID'
fields = 'HOUSING10','POP10'
dictionary = createResultDic(Intersect71Census, dataKey, fields)

header = [dataKey]
for field in fields:
    header.append(field)

createCSV(header,csvname, 'wb' )

for counter, busStop in enumerate(dictionary.keys()):
    datakeys  = dictionary[busStop]
    averages = [busStop]
    for key in datakeys:
        data = datakeys[key]
        average = sum(data)/len(data)
        averages.append(average)
    createCSV(averages,csvname)
```

最后一步展示了如何创建 CSV 文件：通过遍历字典中的数据，然后为每个公交车站的平均值。然后，这些平均值被添加到一个列表中，该列表包含每个公交车站的名称（以及在这个例子中它所属的线路），然后传递给`createCSV()`函数以写入到`CSV`文件中。

这里是最终的代码。请注意，我已经将许多自动生成的注释转换成了打印语句，以提供关于脚本状态的反馈：

```py
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 8662_Chapter4Modified2.py
# Created on: 2014-04-22 21:59:31.00000
#   (generated by ArcGIS/ModelBuilder)
# Description: 
# Adjusted by Silas Toms
# 2014 04 23
# ---------------------------------------------------------------------------

# Import arcpy module
import arcpy
import csv

Bus_Stops = r"C:\Projects\PacktDB.gdb\SanFrancisco\Bus_Stops"
CensusBlocks2010 = r"C:\Projects\PacktDB.gdb\SanFrancisco\CensusBlocks2010"
Inbound71 = r"C:\Projects\PacktDB.gdb\Chapter4Results\Inbound71"
Inbound71_400ft_buffer = r"C:\Projects\PacktDB.gdb\Chapter4Results\Inbound71_400ft_buffer"
Intersect71Census = r"C:\Projects\PacktDB.gdb\Chapter4Results\Intersect71Census"
bufferDist = 400
bufferUnit = "Feet"
lineNames = [('71 IB', 'Ferry Plaza'),('71 OB','48th Avenue')]
sqlTemplate = "NAME = '{0}' AND BUS_SIGNAG = '{1}'"
intersected = [Inbound71_400ft_buffer, CensusBlocks2010]
dataKey = 'NAME','STOPID'
fields = 'HOUSING10','POP10'
csvname = r'C:\Projects\Output\Averages.csv'

def formatSQLMultiple(dataList, sqlTemplate, operator=" OR "):
    'a function to generate a SQL statement'
    sql = ''
    for count, data in enumerate(dataList):
        if count != len(dataList)-1:
            sql += sqlTemplate.format(*data) + operator
        else:
            sql += sqlTemplate.format(*data)
    return sql

def formatIntersect(features):
    'a function to generate an intersect string'
    formatString = ''
    for count, feature in enumerate(features):
        if count != len(features)-1:
            formatString += feature + " #;"
        else:
            formatString += feature + " #"
    return formatString

def createResultDic(resultFC, key, values):
    dataDictionary = {}
    fields = []
    if type(key) == type((1,2)) or type(key) == type([1,2]):
        fields.extend(key)
        length = len(key)
    else:
        fields = [key]
        length = 1
    fields.extend(values)
    with arcpy.da.SearchCursor(resultFC, fields) as cursor:
        for row in cursor:
            busStopID = row[:length]
            data = row[length:]
            if busStopID not in dataDictionary.keys():

                dataDictionary[busStopID] = {}

            for counter,field in enumerate(values):
                if field not in dataDictionary[busStopID].keys():
                    dataDictionary[busStopID][field] = [data[counter]]
                else:
                    dataDictionary[busStopID][field].append(data[counter])

    return dataDictionary

def createCSV(data, csvname, mode ='ab'):
    with open(csvname, mode) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(data)

sql = formatSQLMultiple(lineNames, sqlTemplate)

print 'Process: Select'
arcpy.Select_analysis(Bus_Stops, 
                      Inbound71, 
                      sql)

print 'Process: Buffer'
arcpy.Buffer_analysis(Inbound71, 
                      Inbound71_400ft_buffer, 
                      "{0} {1}".format(bufferDist, bufferUnit), 
                      "FULL", "ROUND", "NONE", "")

iString = formatIntersect(intersected)
print iString

print 'Process: Intersect'
arcpy.Intersect_analysis(iString, 
                          Intersect71Census, "ALL", "", "INPUT")

print 'Process Results'
dictionary = createResultDic(Intersect71Census, dataKey, fields)

print 'Create CSV'
header = [dataKey]
for field in fields:
    header.append(field)
createCSV(header,csvname, 'wb' )

for counter, busStop in enumerate(dictionary.keys()):
    datakeys  = dictionary[busStop]
    averages = [busStop]

    for key in datakeys:
        data = datakeys[key]
        average = sum(data)/len(data)
        averages.append(average)
    createCSV(averages,csvname)

print "Data Analysis Complete"
```

# 摘要

在本章中，我们讨论了如何将自动生成的代码进行泛化，同时添加可以在其他脚本中重用的功能，这将使生成必要的代码组件，如 SQL 语句，变得更加容易。我们还讨论了何时最好不过度创建函数，以避免使它们过于特定。

在下一章中，我们将探讨强大的数据访问模块及其搜索游标、更新游标和插入游标。
