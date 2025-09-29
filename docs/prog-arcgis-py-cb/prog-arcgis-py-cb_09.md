# 第九章。列出和描述 GIS 数据

在本章中，我们将介绍以下食谱：

+   使用 ArcPy 列表函数

+   获取要素类或表中的字段列表

+   使用 Describe() 函数返回关于要素类的描述信息

+   使用 Describe() 函数返回栅格图像的描述信息

# 简介

Python 通过脚本提供批量处理数据的能力。这有助于您自动化工作流程并提高数据处理效率。例如，您可能需要遍历磁盘上的所有数据集并对每个数据集执行特定操作。第一步通常是先进行初步的数据收集，然后再进行地理处理任务的主要部分。这种初步的数据收集通常是通过使用 ArcPy 中找到的一个或多个列表方法来完成的。这些列表作为真正的 Python 列表对象返回。然后，这些列表对象可以用于进一步处理。ArcPy 提供了许多用于生成数据列表的函数。这些方法适用于许多不同类型的 GIS 数据。在本章中，我们将检查 ArcPy 提供的许多用于创建数据列表的函数。在 第二章 中，*管理地图文档和图层*，我们也介绍了一些列表函数。然而，这些函数与使用 `arcpy.mapping` 模块有关，特别是用于处理地图文档和图层。本章中我们介绍的列表函数直接位于 ArcPy 中，并且更具有通用性。

我们还将介绍 Describe() 函数，以返回包含属性组的动态对象。这些动态生成的 Describe 对象包含依赖于所描述数据类型的属性组。例如，当 Describe() 函数针对要素类运行时，将返回特定于要素类的属性。此外，所有数据，无论数据类型如何，都会获得一组通用属性，我们将在稍后讨论。

# 使用 ArcPy 列表函数

获取数据列表通常是多步骤地理处理操作的第一步。ArcPy 提供了许多列表函数，您可以使用它们收集信息列表，无论是要素类、表、工作空间等。在收集数据列表后，您通常会对列表中的项目执行地理处理操作。例如，您可能希望向文件地理数据库中的所有要素类添加新字段。为此，您首先需要获取工作空间中所有要素类的列表。在本食谱中，您将通过使用 `ListFeatureClasses()` 函数来学习如何使用 ArcPy 中的列表函数。所有 ArcPy 列表函数的工作方式相同。

## 准备工作

ArcPy 提供了获取字段列表、索引、数据集、特征类、文件、栅格、表格等列表的函数。所有列表函数执行相同类型的基本操作。`ListFeatureClasses()`函数可用于生成工作空间中所有特征类的列表。`ListFeatureClasses()`函数有三个可选参数可以传递给函数，这些参数将用于限制返回的列表。第一个可选参数是一个通配符，可用于根据名称限制返回的特征类，第二个可选参数可用于根据数据类型（如点、线、多边形等）限制返回的特征类。第三个可选参数通过特征数据集限制返回的特征类。在本菜谱中，您将学习如何使用`ListFeatureClasses()`函数返回特征类列表。您还将学习如何限制返回的列表。

## 如何操作…

按照以下步骤学习如何使用`ListFeatureClasses()`函数检索工作空间中的特征类列表：

1.  打开**IDLE**并创建一个新的脚本窗口。

1.  将脚本保存为`C:\ArcpyBook\Ch9\ListFeatureClasses.py。`

1.  导入`arcpy`模块：

    ```py
    import arcpy
    ```

1.  设置工作空间：

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    ```

    ### 注意

    您应该始终记住在调用使用 IDLE 或任何其他 Python 开发环境开发的脚本中的任何列表函数之前，使用环境设置设置工作空间。如果没有这样做，列表函数将不知道应该从哪个数据集中提取列表。如果在 ArcMap 中运行脚本，并且没有设置工作空间，它将返回默认地理数据库中的特征类。

1.  调用`ListFeatureClasses()`函数并将结果分配给名为`fcList`的变量：

    ```py
    fcList = arcpy.ListFeatureClasses()
    ```

1.  遍历`fcList`中的每个特征类并将它们打印到屏幕上：

    ```py
    for fc in fcList:
        print(fc)
    ```

1.  您可以通过检查`C:\ArcpyBook\code\Ch9\ListFeatureClasses_Step1.py`解决方案文件来验证您的作品。

1.  保存并运行脚本。您应该看到以下输出。

    ```py
    Crimes2009
    CityBoundaries
    CrimesBySchoolDistrict
    SchoolDistricts
    BexarCountyBoundaries
    Texas_Counties_LowRes

    ```

1.  通过传递给`ListFeatureClasses()`函数的第一个参数的通配符，可以限制由该函数返回的特征类列表。通配符用于根据名称限制列表的内容。例如，您可能只想返回以`C`开头的特征类列表。为了实现这一点，您可以使用星号与字符组合。更新`ListFeatureClasses()`函数以包含一个通配符，该通配符将找到所有以大写`C`开头并具有任意数量的字符的特征类：

    ```py
    fcList = arcpy.ListFeatureClasses("C*")
    ```

1.  您可以通过检查`C:\ArcpyBook\code\Ch9\ListFeatureClasses_Step2.py`解决方案文件来验证您的作品。

1.  保存并运行脚本以查看以下输出：

    ```py
    Crimes2009
    CityBoundaries
    CrimesBySchoolDistrict

    ```

1.  除了使用通配符来限制 `ListFeatureClasses()` 函数返回的列表外，还可以应用类型限制，无论是与通配符结合使用还是单独使用。例如，您可以限制返回的特征类列表只包含以 `C` 开头且具有 `polygon` 数据类型的特征类。更新 `ListFeatureClasses()` 函数以包含一个通配符，该通配符将找到所有以大写 `C` 开头且具有多边形数据类型的特征类：

    ```py
    fcs = arcpy.ListFeatureClasses("C*", "Polygon")
    ```

1.  您可以通过检查 `C:\ArcpyBook\code\Ch9\ListFeatureClasses_Step3.py` 解决方案文件来验证您的作品。

1.  保存并运行脚本。您将看到以下输出：

    ```py
    CityBoundaries
    CrimesBySchoolDistrict

    ```

## 它是如何工作的…

在调用任何列表函数之前，您需要设置工作空间环境设置，该设置将当前工作空间设置为从中生成列表的工作空间。`ListFeatureClasses()` 函数可以接受三个可选参数，这些参数将限制返回的特征类。这三个可选参数包括通配符、特征类型和特征数据集。在这个配方中，我们应用了两个可选参数，包括通配符和特征类型。大多数其他列表函数的工作方式相同。参数类型将不同，但调用函数的方式基本上是相同的。

## 更多内容…

而不是返回工作空间中的特征类列表，您可能需要获取表列表。`ListTables()` 函数返回工作空间中的独立表列表。此列表可以通过名称或表类型进行筛选。表类型可以包括 dBase、INFO 和 ALL。列表中的所有值都是 `string` 数据类型，并包含表名。其他列表函数包括 `ListFields()`、`ListRasters()`、`ListWorkspaces()`、`ListIndexes()`、`ListDatasets()`、`ListFiles()` 和 `ListVersions()`。

# 获取特征类或表中的字段列表

特征类和表包含一个或多个属性信息列。您可以通过 `ListFields()` 函数获取特征类中的字段列表。

## 准备工作

`ListFields()` 函数返回一个包含特征类或表中的每个字段的单个 `Field` 对象的列表。一些函数，如 `ListFields()` 和 `ListIndexes()`，需要输入数据集来操作。您可以使用通配符或字段类型来约束返回的列表。每个 `Field` 对象包含各种只读属性，包括 `Name`、`AliasName`、`Type`、`Length` 等。

## 如何做到这一点…

按照以下步骤学习如何返回特征类中的字段列表。

1.  打开 **IDLE** 并创建一个新的脚本窗口。

1.  将脚本保存为 `C:\ArcpyBook\Ch9\ListOfFields.py`。

1.  导入 `arcpy` 模块：

    ```py
    import arcpy
    ```

1.  设置工作空间：

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    ```

1.  在 try 块中对 `Burglary` 特征类调用 `ListFields()` 方法：

    ```py
    try:
        fieldList = arcpy.ListFields("Burglary")
    ```

1.  遍历字段列表中的每个字段，并打印出名称、类型和长度。确保根据需要缩进：

    ```py
      for fld in fieldList:
        print("%s is a type of %s with a length of %i" % (fld.name, fld.type, fld.length))
    ```

1.  添加 `Exception` 块：

    ```py
    except Exception as e:
        print(e.message)
    ```

1.  整个脚本应如下所示：

    ```py
    import arcpy

    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    try:
       fieldList = arcpy.ListFields("Burglary")
       for fld in fieldList:
       print("%s is a type of %s with a length of %i" % (fld.name, fld.type, fld.length))
    except Exception as e:
        print(e.message)
    ```

1.  您可以通过检查`C:\ArcpyBook\code\Ch9\ListOfFields.py`解决方案文件来检查您的工作。

1.  保存并运行脚本。您应该看到以下输出：

    ```py
    OBJECTID is a type of OID with a length of 4
    Shape is a type of Geometry with a length of 0
    CASE is a type of String with a length of 11
    LOCATION is a type of String with a length of 40
    DIST is a type of String with a length of 6
    SVCAREA is a type of String with a length of 7
    SPLITDT is a type of Date with a length of 8
    SPLITTM is a type of Date with a length of 8
    HR is a type of String with a length of 3
    DOW is a type of String with a length of 3
    SHIFT is a type of String with a length of 1
    OFFCODE is a type of String with a length of 10
    OFFDESC is a type of String with a length of 50
    ARCCODE is a type of String with a length of 10
    ARCCODE2 is a type of String with a length of 10
    ARCTYPE is a type of String with a length of 10
    XNAD83 is a type of Double with a length of 8
    YNAD83 is a type of Double with a length of 8

    ```

## 它是如何工作的...

`ListFields()`函数返回要素类或表中的字段列表。此函数接受一个必需参数，即函数应执行的要素类或表的引用。您可以使用通配符或字段类型来限制返回的字段。在本例中，我们仅指定了一个要素类，表示将返回所有字段。对于返回的每个字段，我们打印了名称、字段类型和字段长度。正如我之前在讨论`ListFeatureClasses()`函数时提到的，`ListFields()`和其他所有列表函数通常在脚本中的多步过程中作为第一步被调用。例如，您可能想更新包含在人口普查区要素类中的`population`字段的人口统计数据。为此，您可以获取要素类中所有字段的列表，通过查找包含有关人口信息的特定字段名称来遍历此列表，然后更新每行的人口信息。或者，`ListFields()`函数接受通配符作为其参数之一，因此如果您已经知道`population`字段的名称，您就可以将其作为通配符传递，这样只会返回单个字段。

# 使用`Describe()`函数返回关于要素类的描述性信息

所有数据集都包含描述性信息。例如，要素类具有名称、形状类型、空间参考等信息。在脚本中继续进一步处理之前，这些信息对于您寻找特定信息时非常有价值。例如，您可能只想在`polyline`要素类上执行缓冲区操作，而不是点或多边形。使用`Describe()`函数，您可以获取任何数据集的基本描述性信息。您可以将这些信息视为元数据。

## 准备工作

`Describe()`函数为您提供了获取数据集基本信息的功能。这些数据集可能包括要素类、表、ArcInfo 覆盖、图层文件、工作空间、栅格等。返回一个`Describe`对象，并包含基于描述的数据类型的特定属性。`Describe`对象上的属性组织到属性组中，所有数据集至少属于一个属性组。例如，对地理数据库执行`Describe()`操作将返回`GDB FeatureClass`、`FeatureClass`、`Table`和`Dataset`属性组。每个属性组都包含可以检查的特定属性。

`Describe()` 函数接受一个字符串参数，该参数是指向数据源的一个指针。在下面的代码示例中，我们传递一个包含在文件地理数据库中的要素类。该函数返回一个包含一组动态属性的 `Describe` 对象，这些属性被称为 **属性组**。然后我们可以像在这个例子中那样通过简单地使用打印函数打印属性来访问这些各种属性：

```py
arcpy.env.workspace = "c:/ArcpyBook/Ch9/CityOfSanAntonio.gdb"
desc = arcpy.Describe("Schools")
print("The feature type is: " + desc.featureType)
The feature type is: Simple
print("The shape type is: " + desc.shapeType)
The shape type is: Polygon
print("The name is: " + desc.name)
The name is: Schools
print("The path to the data is: " + desc.path)
The path to the data is: c:/ArcpyBook/Ch9/CityOfSanAntonio.gdb
```

所有数据集，无论其类型如何，都包含在 `Describe` 对象上的一组默认属性。这些属性是只读的。一些更常用的属性包括 `dataType`、`catalogPath`、`name`、`path` 和 `file`。

在本例中，你将编写一个脚本，使用 `Describe()` 函数获取要素类的描述信息。

## 如何操作…

按照以下步骤学习如何获取要素类的描述信息：

1.  打开 **IDLE** 并创建一个新的脚本窗口。

1.  将脚本保存为 `C:\ArcpyBook\Ch9\DescribeFeatureClass.py`。

1.  导入 `arcpy` 模块：

    ```py
    import arcpy
    ```

1.  设置工作空间：

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    ```

1.  开始一个 `try` 块：

    ```py
    try:
    ```

1.  在 `Burglary` 要素类上调用 `Describe()` 函数并打印出形状类型：

    ```py
    descFC = arcpy.Describe("Burglary")
    print("The shape type is: " + descFC.ShapeType)
    ```

1.  获取要素类中的字段列表并打印出每个字段的名称、类型和长度：

    ```py
    flds = descFC.fields
    for fld in flds:
        print("Field: " + fld.name)
        print("Type: " + fld.type
        print("Length: " + str(fld.length))
    ```

1.  获取要素类的地理范围并打印出定义范围的坐标：

    ```py
    ext = descFC.extent
    print("XMin: %f" % (ext.XMin))
    print("YMin: %f" % (ext.YMin))
    print("XMax: %f" % (ext.XMax))
    print("YMax: %f" % (ext.YMax))
    ```

1.  添加 `Exception` 块：

    ```py
    except Exception as e:
        print(e.message)
    ```

1.  整个脚本应该如下所示：

    ```py
    import arcpy
    arcpy.env.workspace = "c:/ArcpyBook/data/CityOfSanAntonio.gdb"
    try:
        descFC = arcpy.Describe("Burglary")
        print("The shape type is: " + descFC.ShapeType)
        flds = descFC.fields
        for fld in flds:
            print("Field: " + fld.name)
            print("Type: " + fld.type)
            print("Length: " + str(fld.length))
        ext = descFC.extent
        print("XMin: %f" % (ext.XMin))
        print("YMin: %f" % (ext.YMin))
        print("XMax: %f" % (ext.XMax))
        print("YMax: %f" % (ext.YMax))
    except:
        print(arcpy.GetMessages())
    ```

1.  你可以通过检查 `C:\ArcpyBook\code\Ch9\DescribeFeatureClass.py` 解决方案文件来检查你的工作。

1.  保存并运行脚本。你应该看到以下输出：

    ```py
    The shape type is: Point
    Field: OBJECTID
    Type: OID
    Length: 4
    Field: Shape
    Type: Geometry
    Length: 0
    Field: CASE
    Type: String
    Length: 11
    Field: LOCATION
    Type: String
    Length: 40
    .....
    .....
    XMin: -103.518030
    YMin: -6.145758
    XMax: -98.243208
    YMax: 29.676404

    ```

## 它是如何工作的…

对要素类执行 `Describe()` 操作，正如我们在脚本中所做的，返回一个 `FeatureClass` 属性组以及分别访问 `Table` 和 `Dataset` 属性组。除了返回 `FeatureClass` 属性组外，你还可以访问 `Table` 属性组。

`Table` 属性组非常重要，主要是因为它让你可以访问独立表或要素类中的字段。你还可以通过这个属性组访问表或要素类上的任何索引。`Table` 属性中的 `Fields` 返回一个包含每个要素类中 `Field` 对象的 Python 列表。每个字段都有许多只读属性，包括 `name`、`alias`、`length`、`type`、`scale`、`precision` 等。最显然有用的属性是名称和类型。在这个脚本中，我们打印出了字段名称、类型和长度。注意使用 Python `for` 循环来处理 Python 列表中的每个 `field`。

最后，我们通过使用`Dataset`属性组中的范围属性返回的`Extent`对象打印出了图层的地域范围。`Dataset`属性组包含许多有用的属性。也许，最常用的属性包括`extent`和`spatialReference`，因为许多地理处理工具和脚本在执行过程中某个时刻都需要这些信息。您还可以获取`datasetType`和版本信息以及其他几个属性。

# 使用`Describe()`函数返回关于栅格图像的描述性信息

栅格文件也包含描述性信息，这些信息可以通过`Describe()`函数返回。

## 准备工作

栅格数据集也可以通过使用`Describe()`函数来描述。在这个菜谱中，您将通过返回其范围和空间参考来描述栅格数据集。`Describe()`函数包含对通用`Dataset`属性组的引用，同时也包含对数据集的`SpatialReference`对象的引用。然后可以使用`SpatialReference`对象来获取数据集的详细空间参考信息。

## 如何操作…

按照以下步骤学习如何获取关于栅格图像文件描述性信息的方法。

1.  打开**IDLE**并创建一个新的脚本窗口。

1.  将脚本保存为`C:\ArcpyBook\Ch9\DescribeRaster.py`。

1.  导入`arcpy`模块：

    ```py
    import arcpy
    ```

1.  设置工作空间：

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data "
    ```

1.  开始一个`try`块：

    ```py
    try:
    ```

1.  在栅格数据集上调用`Describe()`函数：

    ```py
    descRaster = arcpy.Describe("AUSTIN_EAST_NW.sid")
    ```

1.  获取栅格数据集的范围并打印出来：

    ```py
    ext = descRaster.extent
    print("XMin: %f" % (ext.XMin))
    print("YMin: %f" % (ext.YMin))
    print("XMax: %f" % (ext.XMax))
    print("YMax: %f" % (ext.YMax))
    ```

1.  获取对`SpatialReference`对象的引用并打印出来：

    ```py
    sr = descRaster.SpatialReference
    print(sr.name)
    print(sr.type)
    ```

1.  添加`Exception`块：

    ```py
    except Exception as e:
        print(e.message)
    ```

1.  整个脚本应如下所示：

    ```py
    import arcpy
    arcpy.env.workspace = "c:/ArcpyBook/data"
    try:
        descRaster = arcpy.Describe("AUSTIN_EAST_NW.sid")
        ext = descRaster.extent
        print("XMin: %f" % (ext.XMin))
        print("YMin: %f" % (ext.YMin))
        print("XMax: %f" % (ext.XMax))
        print("YMax: %f" % (ext.YMax))

        sr = descRaster.SpatialReference
        print(sr.name)
        print(sr.type)
    except Exception as e:
        print(e.message)
    ```

1.  您可以通过检查`C:\ArcpyBook\code\Ch9\DescribeRaster.py`解决方案文件来验证您的操作。

1.  保存并运行脚本。您应该看到以下输出：

    ```py
    XMin: 3111134.862457
    YMin: 10086853.262238
    XMax: 3131385.723907
    YMax: 10110047.019228
    NAD83_Texas_Central
    Projected

    ```

## 它是如何工作的…

这个菜谱与之前的菜谱非常相似。不同之处在于我们使用的是对栅格数据集的`Describe()`函数，而不是`vector`要素类。在这两种情况下，我们都使用范围对象返回了数据集的地理范围。然而，在脚本中，我们还获取了栅格数据集的`SpatialReference`对象，并打印出有关该对象的信息，包括其名称和类型。
