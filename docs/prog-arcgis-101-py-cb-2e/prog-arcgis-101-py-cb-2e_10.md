# 第十章。列出和描述 GIS 数据

在本章中，我们将介绍以下内容：

+   获取工作空间中要素类的列表

+   使用通配符限制返回的对象列表

+   限制返回的对象列表中的要素类型

+   获取要素类或表中的字段列表

+   使用 Describe()函数返回关于要素类的描述性信息

+   使用 Describe()函数返回关于图像的描述性信息

+   使用 Describe()函数返回工作空间信息

# 简介

Python 通过脚本提供批量处理数据的能力。这有助于您自动化工作流程并提高数据处理效率。例如，您可能需要遍历磁盘上的所有数据集并对每个数据集执行特定操作。您的第一步通常是先进行初步的数据收集，然后再进行地理处理任务的主要内容。这种初步的数据收集通常是通过使用 ArcPy 中找到的一个或多个**List**方法来完成的。这些列表作为真正的 Python 列表对象返回。然后，这些列表对象可以被迭代以进行进一步处理。ArcPy 提供了一些函数，可以用来生成数据列表。这些方法适用于许多不同类型的 GIS 数据。在本章中，我们将检查 ArcPy 提供的用于创建数据列表的许多函数。在第三章中，我们也介绍了一些列表函数。然而，这些函数与使用`arcpy.mapping`模块有关，特别是用于处理地图文档和图层。本章中我们介绍的列表函数直接位于`arcpy`中，并且更具有通用性。

我们还将介绍用于返回包含属性组的动态对象的`Describe()`函数。这些动态生成的`Describe`对象将包含依赖于所描述数据类型的属性组。例如，当`Describe()`函数针对要素类运行时，将返回特定于要素类的属性。此外，所有数据，无论数据类型如何，都会获得一组通用属性，我们将在后面讨论。

# 获取工作空间中要素类的列表

就像本章我们将检查的所有列表函数一样，获取工作空间中要素类的列表通常是您的脚本执行的多步过程中的第一步。例如，您可能想要向文件地理数据库中的所有要素类添加一个新字段。为此，您首先需要获取工作空间中所有要素类的列表。

## 准备工作

ArcPy 提供了获取字段、索引、数据集、要素类、文件、栅格、表等列表的函数。`ListFeatureClasses()`函数可以用来生成工作空间中所有要素类的列表。`ListFeatureClasses()`有三个可选参数可以传递给函数，这些参数将用于限制返回的列表。第一个可选参数是一个**通配符**，可以用来根据名称限制返回的要素类。第二个可选参数可以用来根据数据类型（点、线、多边形等）限制返回的要素类。第三个可选参数通过要素数据集限制返回的要素类。在本例中，我们将返回工作空间中的所有要素类。

## 如何操作…

按照以下步骤学习如何使用`ListFeatureClasses()`函数检索工作空间中要素类的列表：

1.  打开 IDLE 并创建一个新的脚本窗口。

1.  将脚本保存为`c:\ArcpyBook\Ch10\ListFeatureClasses.py`。

1.  导入`arcpy`模块：

    ```py
    import arcpy
    ```

1.  设置工作空间：

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    ```

    ### 注意

    你必须始终记住在调用任何列表函数之前使用环境设置设置工作空间。否则，列表函数将不知道应该从哪个数据集提取列表。

1.  调用`ListFeatureClasses()`函数并将结果赋值给名为`fcList`的变量：

    ```py
    fcList = arcpy.ListFeatureClasses()
    ```

1.  遍历`fcList`中的每个要素类并在屏幕上打印它们：

    ```py
    for fc in fcList:
      print fc
    ```

1.  保存并运行脚本。你应该看到以下输出：

    ```py
    Crimes2009
    CityBoundaries
    CrimesBySchoolDistrict
    SchoolDistricts
    BexarCountyBoundaries
    Texas_Counties_LowRes
    Burglary

    ```

## 它是如何工作的…

在调用任何列表函数之前，你需要设置工作空间环境设置，这将设置当前工作空间，你将从中生成列表。`ListFeatureClasses()`函数可以接受三个可选参数，这些参数将限制返回的要素类。大多数其他列表函数以相同的方式工作。然而，在这种情况下，我们没有传递任何参数就调用了`ListFeatureClasses()`函数。这将返回当前工作空间中所有要素类的 Python 列表对象，然后使用`for`循环进行迭代。列表中返回的每个要素类都表示为一个包含要素类名称的字符串。

## 还有更多…

除了返回工作空间中要素类的列表之外，你可能还需要获取一个表列表。`ListTables()`函数返回工作空间中独立表的列表。此列表可以根据名称或表类型进行筛选。表类型可以包括`dBase`、`INFO`和`ALL`。列表中的所有值都是`string`数据类型，并包含表名。

# 使用通配符限制返回的要素类列表

默认情况下，`ListFeatureClasses()` 函数将返回工作空间中的所有特征类。你通常会想以某种方式限制此列表。可以将三个可选参数传递给 `ListFeatureClasses()` 以限制返回的特征类。所有参数都是可选的。第一个参数是一个通配符，用于根据字符组合限制返回的列表。其他可以用来限制列表的参数包括数据类型和特征数据集。

## 准备工作

通过将通配符作为第一个参数传入，可以限制 `ListFeatureClasses()` 函数返回的特征类列表。通配符用于根据名称限制列表内容。例如，你可能只想返回以字母 `B` 开头的特征类列表。为此，你使用星号与任意数量的字符的组合。以下代码示例显示了如何使用通配符来限制列表内容：

```py
fcs = arcpy.ListFeatureClasses("B*")
```

在本食谱中，你将学习如何通过使用通配符来限制返回的特征类列表。

## 如何操作...

按照以下步骤学习如何通过传递给第一个参数的通配符来限制 `ListFeatureClasses()` 函数返回的特征类列表：

1.  打开 IDLE 和 `c:\ArcpyBook\Ch10\ListFeatureClasses.py` 脚本。

1.  添加一个通配符，以限制返回的特征类列表仅包含以字母 `C` 开头的特征类：

    ```py
    fcs = arcpy.ListFeatureClasses("C*")
    ```

1.  保存并运行脚本以查看以下输出：

    ```py
    Crimes2009
    CityBoundaries
    CrimesBySchoolDistrict

    ```

## 它是如何工作的...

`ListFeatureClasses()` 函数可以接受三个可选参数，包括一个通配符，该通配符将根据名称限制特征类列表。在这种情况下，我们使用了通配符字符（`*`）来限制返回的特征类列表，使其仅包含以字母 `C` 开头的特征类。

# 通过特征类型限制返回的特征类列表。

除了使用通配符来限制 `ListFeatureClasses()` 返回的特征类外，还可以通过特征类型进行过滤。

## 准备工作

除了使用通配符来限制 `ListFeatureClasses()` 函数返回的列表外，还可以结合通配符或单独应用类型限制。例如，以下代码示例显示了两者结合使用，以将返回的列表限制为仅包含以字母 `B` 开头的 `polygon` 特征类。在本食谱中，你将通过使用特征类型参数和通配符来限制返回的特征类。

```py
fcs = arcpy.ListFeatureClasses("B*", "Polygon")
```

## 如何操作...

按照以下步骤学习如何通过特征类型限制 `ListFeatureClasses()` 函数返回的特征类列表：

1.  打开 IDLE 和 `c:\ArcpyBook\Ch10\ListFeatureClasses.py` 脚本。

1.  向 `ListFeatureClasses()` 函数添加第二个参数，以限制返回的要素类仅限于以字母 `C` 开头且类型为 `polygon` 的那些：

    ```py
    fcs = arcpy.ListFeatureClasses("C*","Polygon")
    ```

1.  保存并运行脚本以查看以下输出：

    ```py
    CityBoundaries
    CrimesBySchoolDistrict

    ```

## 它是如何工作的...

在这个菜谱中，我们已将要素类限制为仅包含多边形要素。其他有效的要素类型包括点、折线和区域。

## 更多...

可以传递给 `ListFeatureClasses()` 函数的第三个可选参数是要素数据集名称。这将过滤列表，只返回特定要素数据集中的要素类。当此可选参数不包括在 `ListFeatureClasses()` 调用中时，只返回当前工作空间中的独立要素类。

# 获取要素类或表中的字段列表

要素类和表包含一个或多个属性信息列。您可以通过 `ListFields()` 函数获取要素类中的字段列表。

## 准备工作

`ListFields()` 函数返回一个包含每个字段单独 `Field` 对象的列表，这些对象对应于要素类或表中的每个字段。一些函数，例如 `ListFields()` 和 `ListIndexes()`，需要输入数据集来操作。您可以使用通配符或字段类型来限制返回的列表。每个 `Field` 对象包含各种只读属性，包括 `Name`、`AliasName`、`Type`、`Length` 等。

## 如何操作...

按照以下步骤学习如何返回要素类中的字段列表：

1.  打开 IDLE 并创建一个新的脚本窗口。

1.  将脚本保存为 `c:\ArcpyBook\Ch10\ListOfFields.py`。

1.  导入 `arcpy` 模块。

    ```py
    import arcpy
    ```

1.  设置工作空间：

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    ```

1.  在 `try` 块中对 `Burglary` 要素类调用 `ListFields()` 方法：

    ```py
    try:
      fieldList = arcpy.ListFields("Burglary")
    ```

1.  遍历字段列表中的每个字段，并打印名称、类型和长度。确保根据需要缩进：

    ```py
      for fld in fieldList:
        print "%s is a type of %s with a length of %i" % (fld.name, fld.type, fld.length)
    ```

1.  添加 `except` 块：

    ```py
    except Exception e:
      print e.message();
    ```

1.  整个脚本应如下所示：

    ```py
    import arcpy

    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    try:
     fieldList = arcpy.ListFields("Burglary")
      for fld in fieldList:
        print "%s is a type of %s with a length of %i" % (fld.name, fld.type, fld.length)
    except Exception e:
      print e.message()
    ```

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

`ListFields()` 函数返回要素类或表格的字段列表。此函数接受一个必需参数，即函数应针对其执行的功能类的引用或表格。您可以使用通配符或字段类型来限制返回的字段。在本例中，我们仅指定了一个要素类，这表示将返回所有字段。对于返回的每个字段，我们打印了名称、字段类型和字段长度。正如我之前在讨论 `ListFeatureClasses()` 函数时提到的，`ListFields()` 和所有其他列表函数通常在脚本中的多步过程中作为第一步被调用。例如，您可能只想更新人口普查区要素类中包含的人口统计信息。为此，您可以获取要素类中所有字段的列表，遍历此列表以查找包含人口信息的特定字段名称，然后更新每行的人口信息。或者，`ListFields()` 函数接受通配符作为其参数之一。因此，如果您事先知道人口字段的名称，可以将该名称作为通配符传递，从而只返回单个字段。

# 使用 `Describe()` 函数返回要素类的描述信息

所有数据集都包含描述性信息。例如，要素类有一个名称、形状类型、空间参考等。当您在脚本中继续进一步处理之前寻求特定信息时，这些信息对您的脚本可能很有价值。例如，您可能只想对折线要素类执行缓冲区操作，而不是对点或多边形执行。使用 `Describe()` 函数，您可以获取任何数据集的基本描述信息。您可以将此信息视为元数据。

## 准备工作

`Describe()` 函数为您提供了获取数据集基本信息的功能。这些数据集可能包括要素类、表格、ArcInfo 覆盖、图层文件、工作空间、栅格以及其他类型。返回一个 `Describe` 对象，其中包含基于描述的数据类型的特定属性。`Describe` 对象上的属性被组织成属性组，所有数据集至少属于一个属性组。例如，对地理数据库执行 `Describe()` 操作将返回 **GDB** 的 `FeatureClass`、`Table` 和 `Dataset` 属性组。每个属性组都包含可以检查的特定属性。

所有数据集，无论其类型如何，都包含在`Describe`对象上的默认属性集。这些属性是只读的。一些更常用的属性包括`dataType`、`catalogPath`、`name`、`path`和`file`。

```py
arcpy.env.workspace = "c:/ArcpyBook/Ch10/CityOfSanAntonio.gdb"
desc = arcpy.Describe("Schools")
print "The feature type is: " + desc.featureType
The feature type is: Simple
print "The shape type is: " + desc.shapeType
The shape type is: Polygon
print "The name is: " + desc.name
The name is: Schools
print "The path to the data is: " + desc.path
The path to the data is: c:/ArcpyBook/Ch10/CityOfSanAntonio.gdb

```

添加`except`块：

打开 IDLE 并创建一个新的脚本窗口。

## `Describe()`函数接受一个字符串参数，该参数是指向数据源的指针。在下面的代码示例中，我们传递一个包含在文件地理数据库中的要素类。该函数返回一个包含一组称为属性组的动态属性的`Describe`对象。然后我们可以通过简单地使用`print`函数打印属性来访问这些各种属性，就像在这个例子中我们所做的那样：

`Table`属性组之所以重要，主要是因为它允许你访问独立表或要素类中的字段。你还可以通过此属性组访问表或要素类上的任何索引。表属性返回一个包含每个要素类中`Field`对象的 Python 列表。每个字段都有许多只读属性，包括`name`、`alias`、`length`、`type`、`scale`、`precision`等。最显然有用的属性是`name`和`type`。在这个脚本中，我们打印了字段名称、类型和长度。注意使用 Python `for`循环来处理 Python 列表中的每个字段。

1.  它是如何工作的...

1.  整个脚本应如下所示：

1.  对要素类执行`Describe()`函数，我们在脚本中已经这样做，返回一个`FeatureClass`属性组以及访问`Table`和`Dataset`属性组的权限。除了返回`FeatureClass`属性组外，你还可以访问`Table`属性组。

1.  导入 arcpy 模块

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data/CityOfSanAntonio.gdb"
    ```

1.  在`Burglary`要素类上调用`Describe()`函数并打印形状类型：

    ```py
    try:
    ```

1.  设置工作空间：

    ```py
    descFC = arcpy.Describe("Burglary")
    print "The shape type is: " + descFC.ShapeType
    ```

1.  在这个菜谱中，你将编写一个脚本，使用`Describe()`函数获取关于要素类的描述性信息。

    ```py
    flds = descFC.fields
    for fld in flds:
      print "Field: " + fld.name
      print "Type: " + fld.type
      print "Length: " + str(fld.length)
    ```

1.  如何做到这一点...

    ```py
    ext = descFC.extent
    print "XMin: %f" % (ext.XMin)
    print "YMin: %f" % (ext.YMin)
    print "XMax: %f" % (ext.XMax)
    print "YMax: %f" % (ext.YMax)
    ```

1.  开始一个`try`块：

    ```py
    except Exception e:
      print e.message()
    ```

1.  获取要素类中的字段列表并打印每个字段的名称、类型和长度：

    ```py
    import arcpy
    arcpy.env.workspace = "c:/ArcpyBook/data/CityOfSanAntonio.gdb"
    try:
      descFC = arcpy.Describe("Burglary")
      print "The shape type is: " + descFC.ShapeType
      flds = descFC.fields
      for fld in flds:
        print "Field: " + fld.name
        print "Type: " + fld.type
        print "Length: " + str(fld.length)
      ext = descFC.extent
      print "XMin: %f" % (ext.XMin)
      print "YMin: %f" % (ext.YMin)
      print "XMax: %f" % (ext.XMax)
      print "YMax: %f" % (ext.YMax)
    except:
      print arcpy.GetMessages()
    ```

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

## 获取要素类的地理范围并打印定义范围的坐标：

按照以下步骤学习如何获取关于要素类的描述性信息：

将脚本保存为`c:\ArcpyBook\Ch10\DescribeFeatureClass.py`。

最后，我们通过使用`Extent`对象，该对象由`Dataset`属性组上的`extent`属性返回，打印出层的地理范围。`Dataset`属性组包含许多有用的属性。也许，最常用的属性包括`extent`和`spatialReference`，因为许多地理处理工具和脚本在执行过程中某个时刻都需要这些信息。你还可以获取`datasetType`和版本信息以及其他几个属性。

# 使用`Describe()`函数返回关于图像的描述性信息

栅格文件也包含描述性信息，这些信息可以通过`Describe()`函数返回。

## 准备工作

栅格数据集也可以通过使用`Describe()`函数来描述。在这个菜谱中，你将通过返回其范围和空间参考来描述一个栅格数据集。`Describe()`函数还包含对通用`Dataset`属性组的引用，该属性组包含对数据集的`SpatialReference`对象的引用。然后可以使用`SpatialReference`对象来获取数据集的详细空间参考信息。

## 如何做…

按照以下步骤学习如何获取关于栅格图像文件描述性信息的方法：

1.  打开 IDLE 并创建一个新的脚本窗口。

1.  将脚本保存为`c:\ArcpyBook\Ch10\DescribeRaster.py`。

1.  导入`arcpy`模块：

    ```py
    import arcpy
    ```

1.  设置工作区：

    ```py
    arcpy.env.workspace = "C:/ArcpyBook/data"
    ```

1.  开始一个`try`块：

    ```py
    try:
    ```

1.  在`try`语句中调用`Describe()`函数针对栅格数据集，确保将下一行代码缩进：

    ```py
    descRaster = arcpy.Describe("AUSTIN_EAST_NW.sid")
    ```

1.  获取栅格数据集的范围并打印：

    ```py
    ext = descRaster.extent
    print "XMin: %f" % (ext.XMin)
    print "YMin: %f" % (ext.YMin)
    print "XMax: %f" % (ext.XMax)
    print "YMax: %f" % (ext.YMax)
    ```

1.  获取对`SpatialReference`对象的引用并打印：

    ```py
    sr = descRaster.SpatialReference
    print sr.name
    print sr.type
    ```

1.  添加`except`块：

    ```py
    except Exception e:
      print e.message()
    ```

1.  整个脚本应该如下所示：

    ```py
    import arcpy
    arcpy.env.workspace = "c:/ArcpyBook/data"
    try:
      descRaster = arcpy.Describe("AUSTIN_EAST_NW.sid")
      ext = descRaster.extent
      print "XMin: %f" % (ext.XMin)
      print "YMin: %f" % (ext.YMin)
      print "XMax: %f" % (ext.XMax)
      print "YMax: %f" % (ext.YMax)

      sr = descRaster.SpatialReference
      print sr.name
      print sr.type
    except:
      print arcpy.GetMessages()
    ```

1.  保存并运行脚本。你应该看到以下输出：

    ```py
    XMin: 3111134.862457
    YMin: 10086853.262238
    XMax: 3131385.723907
    YMax: 10110047.019228
    NAD83_Texas_Central
    Projected

    ```

## 它是如何工作的…

这个菜谱与上一个非常相似。不同之处在于我们正在使用`Describe()`函数针对栅格数据集而不是针对矢量要素类。在两种情况下，我们都使用了`extent`对象来返回数据集的地理范围。然而，在脚本中我们还获取了栅格数据集的`SpatialReference`对象并打印了关于该对象的信息，包括名称和类型。

# 使用`Describe()`函数返回工作区信息

可以使用多种类型的地理数据库与 ArcGIS 一起使用，包括个人、文件和企业。正如我们在第八章中看到的，*查询和选择数据*，查询的构建将取决于数据集所在的地理数据库类型。你的脚本可能或可能不知道地理数据库类型。为了使你的脚本在查询时更加健壮，你可以使用`Describe()`函数针对工作区来捕获这些信息并相应地构建你的查询。

## 准备工作

`Workspace` 属性组提供了关于工作空间（如文件夹、个人或文件地理数据库，或企业地理数据库）的信息。这些属性在获取 ArcSDE 连接信息时尤其有用。通过此属性组可以获取的信息包括当工作空间是 ArcSDE 工作空间时的连接信息、与地理数据库关联的域以及工作空间类型，这些类型可以是 `FileSystem`、`LocalDatabase` 或 `RemoteDatabase`。`LocalDatabase` 指的是个人或文件地理数据库，而 `RemoteDatabase` 指的是 `ArcSDE` 地理数据库。在本例中，你将使用 `Workspace` 属性组来获取文件地理数据库的信息。

## 如何操作...

按照以下步骤学习如何获取工作空间的描述信息：

1.  打开 IDLE 并创建一个新的脚本窗口。

1.  将脚本保存为 `c:\ArcpyBook\Ch10\DescribeWorkspace.py`。

1.  导入 `arcpy` 模块：

    ```py
    import arcpy
    ```

1.  开始一个 `try` 块：

    ```py
    try:
    ```

1.  在 `CityOfSanAntonio` 地理数据库上调用 `Describe()` 函数，并确保将此语句缩进放在 try 语句内。下面的两个打印语句也应该缩进。

    ```py
    descRaster = arcpy.Describe("c:/ArcpyBook/data/CityOfSanAntonio.gdb")
    ```

1.  打印工作空间类型：

    ```py
    print descWorkspace.workspaceType
    ```

1.  打印详细的工作空间信息：

    ```py
    print descWorkspace.workspaceFactoryProgID
    ```

1.  添加 `except` 块：

    ```py
    except Exception e:
      print e.message()
    ```

1.  保存并运行脚本。你应该看到以下输出：

    ```py
    LocalDatabase
    esriDataSourcesGDB.FileGDBWorkspaceFactory.1

    ```

## 工作原理...

`workspaceType` 属性返回三个值之一：`FileSystem`、`LocalDatabase` 或 `RemoteDatabase`。`localDatabase` 值表示你正在使用个人或文件地理数据库。然而，它并不更具体。要获取具体的地理数据库，你可以检索 `workspaceFactoryProgID` 属性，这将指示地理数据库的类型。在这种情况下，它是一个文件地理数据库。
