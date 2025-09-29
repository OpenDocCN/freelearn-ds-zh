# 第三章。将空间数据从一个格式移动到另一个格式

在本章中，我们将涵盖以下主题：

+   使用 ogr2ogr 将 Shapefile 转换为 PostGIS 表

+   使用 ogr2ogr 将 Shapefiles 文件夹批量导入 PostGIS

+   从 PostGIS 批量导出一系列表到 Shapefiles

+   将 OpenStreetMap（OSM）XML 转换为 Shapefile

+   将 Shapefile（矢量）转换为 GeoTiff（栅格）

+   使用 GDAL 将栅格（GeoTiff）转换为矢量（Shapefile）

+   从存储在 Microsoft Excel 中的点数据创建 Shapefile

+   将 ESRI ASCII DEM 转换为图像高度图

# 简介

地理空间数据有数百种格式，将数据从一种格式转换为另一种格式是一项简单的任务。在数据类型之间进行转换的能力，如栅格或矢量，属于数据处理任务，可用于更好的地理空间分析。以下是一个栅格和矢量数据集的示例，以便您可以查看我在谈论的内容：

![简介](img/B03543_03_01.jpg)

最佳实践方法是运行存储在常见格式中的数据（如 PostgreSQL PostGIS 数据库或一组具有共同坐标系统的 Shapefiles）上的分析函数或模型。例如，对存储在多种格式中的输入数据进行分析也是可能的，但如果出现问题或结果不符合预期，您可能会发现问题的细节。

本章将探讨一些常见的数据格式，并演示如何使用最常用的工具将这些格式从一种转换为另一种。

# 使用 ogr2ogr 将 Shapefile 转换为 PostGIS 表

将数据从一种格式转换为另一种格式最简单的方法是直接使用随 GDAL 安装提供的*ogr2ogr*工具。这个强大的工具可以转换 200 多种地理空间格式。在这个解决方案中，我们将从 Python 脚本中执行*ogr2ogr*实用程序以执行通用的矢量数据转换。因此，Python 代码被用来执行这个命令行工具并传递变量，这样您就可以创建自己的数据导入或导出脚本。

如果你对编码并不特别感兴趣，只想完成工作以移动你的数据，使用这个工具也是推荐的。当然，纯 Python 解决方案是可能的，但它无疑更倾向于满足开发人员（或 Python 纯主义者）的需求。由于本书的目标读者是开发人员、分析师或研究人员，这种类型的配方既简单又易于扩展。

## 准备工作

要运行此脚本，您需要在您的系统上安装 GDAL 工具应用程序。Windows 用户可以访问 OSGeo4W（[`trac.osgeo.org/osgeo4w`](http://trac.osgeo.org/osgeo4w)）并下载 32 位或 64 位 Windows 安装程序。只需双击安装程序即可启动脚本，如下所示：

1.  导航到底部选项，**高级安装** | **下一步**。

1.  点击**下一步**从互联网下载 GDAL 工具（第一个默认选项）。

1.  点击**下一步**接受路径的默认位置或更改为你喜欢的位置。

1.  点击**下一步**接受本地保存下载的位置（默认）。

1.  点击**下一步**接受直接连接（默认）。

1.  点击**下一步**选择默认下载站点。

1.  现在，你终于可以看到菜单了。点击**+**打开**Commandline_Utilities**标签页，你应该能看到这个截图所示的内容：![准备中](img/B03543_03_02.jpg)

1.  现在，选择**gdal: The GDAL/OGR library and commandline tools**来安装它。

1.  点击**下一步**开始下载和安装。

Ubuntu/Linux 用户可以使用以下步骤安装 GDAL 工具：

1.  执行以下简单的单行命令：

    ```py
    $ sudo apt-get install gdal-bin

    ```

    这将使你能够直接从终端执行`ogr2ogr`。

    要导入的 Shapefile 位于你的`/ch02/geodata/`文件夹中，如果你已经从 GitHub [`github.com/mdiener21/python-geospatial-analysis-cookbook/`](https://github.com/mdiener21/python-geospatial-analysis-cookbook/)下载了整个源代码和代码。温哥华开放地理数据门户 [`data.vancouver.ca/datacatalogue/index.htm`](http://data.vancouver.ca/datacatalogue/index.htm) 是我们的数据源，它提供了一个本地自行车道的数据集。

1.  接下来，让我们设置带有 PostGIS 扩展的 PostgreSQL 数据库。为此，我们首先创建一个新用户来管理我们的新数据库和表，如下所示：

    ```py
    Sudo su createuser  –U postgres –P pluto

    ```

1.  为新角色输入密码。

1.  再次输入新角色的密码。

1.  为`postgres`用户输入密码，因为你将使用此`postgres`用户创建用户。

1.  `–P`选项会提示你为名为`pluto`的新用户设置密码。在以下示例中，我们的密码是`stars`；我建议为你的生产数据库使用一个更安全的密码。

    ### 小贴士

    Windows 用户可以导航到`c:\Program Files\PostgreSQL\9.3\bin\`文件夹，并执行以下命令，然后按照之前的方式遵循屏幕上的说明：

    ```py
    Createuser.exe –U postgres –P pluto

    ```

1.  要创建数据库，我们将使用与`postgres`用户相同的`createdb`命令行来创建一个名为`py_geoan_cb`的数据库，并将`pluto`用户指定为数据库所有者。以下是执行此操作的命令：

    ```py
    $ sudo su createdb –O pluto –U postgres py_geoan_cb

    ```

    ### 小贴士

    Windows 用户可以访问`c:\Program Files\PostgreSQL\9.3\bin\`并执行以下`createdb.exe`命令：

    ```py
    createdb.exe –O pluto –U postgres py_geoan_cb

    ```

    接下来，我们将为我们的新创建的数据库创建 PostGIS 扩展：

    ```py
    psql –U postgres -d py_geoan_cb -c "CREATE EXTENSION postgis;"

    ```

    Windows 用户也可以在`c:\Program Files\PostgreSQL\9.3\bin\`文件夹中执行`psql`，如下所示：

    ```py
    psql.exe –U postgres –d py_geoan_cb –c "CREATE EXTENSION postgis;"

    ```

1.  最后，我们将创建一个名为**geodata**的模式来存储我们新的空间表。在 PostgreSQL 的默认`public`模式之外存储空间数据是常见的。

    ```py
    $ sudo -u postgres psql -d py_geoan_cb -c "CREATE SCHEMA geodata AUTHORIZATION pluto;"

    ```

    ### 小贴士

    Windows 用户可以使用以下命令来完成此操作：

    ```py
    psql.exe –U postgres –d py_geoan_cb –c "CREATE SCHEMA geodata AUTHORIZATION pluto;"

    ```

## 如何操作...

1.  现在，让我们开始将我们的 Shapefile 导入到 PostGIS 数据库中，这将自动从我们的 Shapefile 创建一个新表：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    import subprocess

    # database options
    db_schema = "SCHEMA=geodata"
    overwrite_option = "OVERWRITE=YES"
    geom_type = "MULTILINESTRING"
    output_format = "PostgreSQL"

    # database connection string
    db_connection = """PG:host=localhost port=5432
      user=pluto dbname=py_test password=stars"""

    # input shapefile
    input_shp = "../geodata/bikeways.shp"

    # call ogr2ogr from python
    subprocess.call(["ogr2ogr","-lco", db_schema, "-lco", overwrite_option,
      "-nlt", geom_type, "-f", output_format, db_connection,  input_shp])
    ```

1.  接下来，我们将从命令行调用我们的脚本：

    ```py
    $ python ch03-01_shp2pg.py

    ```

## 它是如何工作的...

我们首先导入标准的 Python `subprocess`模块，该模块将调用*ogr2ogr*命令行工具。接下来，我们将设置一系列变量，这些变量用作输入参数，并为 ogr2ogr 执行提供各种选项。

从`SCHEMA=geodata`的 PostgreSQL 数据库开始，我们为我们的新表设置了一个非默认的数据库模式。将空间数据表存储在公共模式之外的一个单独的模式中是一种最佳实践，公共模式是默认模式。这种做法将使备份和恢复变得容易得多，并使数据库组织得更好。

接下来，我们创建一个设置为`yes`的`overwrite_option`变量，这样我们就可以在创建时覆盖任何同名表。当您想完全用新数据替换表时，这很有用；否则，建议使用`-append`选项。我们还指定了几何类型，因为有时 ogr2ogr 并不总是能正确猜测我们的 Shapefile 的几何类型，所以设置这个值可以节省您这方面的担忧。

现在，我们将使用`PostgreSQL`关键字设置我们的`output_format`变量，告诉 ogr2ogr 我们希望将数据输出到 PostgreSQL 数据库。然后是`db_connection`变量，它指定了我们的数据库连接信息。我们绝对不能忘记数据库必须已经存在，以及`geodata`模式；否则，我们将得到一个错误。

最后的`input_shp`变量是我们 Shapefile 的完整路径，包括`.shp`文件扩展名。我们将调用 subprocess 模块，它将调用 ogr2ogr 命令行工具，并传递运行工具所需的变量选项。我们向该函数传递一个参数数组，数组中的第一个对象是 ogr2ogr 命令行工具的名称。在名称之后，我们在数组中传递一个选项，以完成调用。

### 注意

Subprocess 可以用来直接调用任何命令行工具。Subprocess 接受由空格分隔的参数列表。这种参数传递相当挑剔，所以请确保您紧跟其后，不要添加任何额外的空格或逗号。

最后但同样重要的是，我们需要从命令行执行我们的脚本，通过调用 Python 解释器并传递脚本实际上导入我们的 Shapefile。现在转到**PgAdmin** PostgreSQL 数据库查看器，看看是否成功。或者，更好的是，打开 Quantum GIS ([www.qgis.org](http://www.qgis.org))并查看新创建的表。

## 参见

如果您想查看 ogr2ogr 命令可用的完整选项列表，只需在命令行中输入以下内容：

```py
$ ogr2ogr –help

```

您将看到可用的完整选项列表。此外，请访问[`gdal.org/ogr2ogr.html`](http://gdal.org/ogr2ogr.html)以阅读可用的文档。

### 注意

对于那些好奇如何在不使用 Python 的情况下运行此调用的人来说，直接调用`ogr2ogr`的调用方式如下：

```py
ogr2ogr -lco SCHEMA=geodata -nlt MULTILINE -f "Postgresql" PG:"host=localhost port=5432 user=postgres dbname=py_geoan_cb password=secret" /home/mdiener/ch03/geodata/bikeways.shp

```

# 使用 ogr2ogr 将 Shapefile 文件夹批量导入 PostGIS

我们希望扩展我们最后的脚本，以便遍历一个充满 Shapefiles 的文件夹并将它们导入到 PostGIS 中。大多数导入任务都涉及多个文件，因此这是一个非常实用的任务。

## 如何操作...

我们的脚本将以函数的形式重用之前的代码，这样我们就可以批量处理要导入到 PostgreSQL PostGIS 数据库的 Shapefiles 列表。

1.  为了简化起见，我们将从单个文件夹创建我们的 Shapefiles 列表：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    import subprocess
    import os
    import ogr

    def discover_geom_name(ogr_type):
        """

        :param ogr_type: ogr GetGeomType()
        :return: string geometry type name
        """
        return {ogr.wkbUnknown            : "UNKNOWN",
                ogr.wkbPoint              : "POINT",
                ogr.wkbLineString         : "LINESTRING",
                ogr.wkbPolygon            : "POLYGON",
                ogr.wkbMultiPoint         : "MULTIPOINT",
                ogr.wkbMultiLineString    : "MULTILINESTRING",
                ogr.wkbMultiPolygon       : "MULTIPOLYGON",
                ogr.wkbGeometryCollection : "GEOMETRYCOLLECTION",
                ogr.wkbNone               : "NONE",
                ogr.wkbLinearRing         : "LINEARRING"}.get(ogr_type)

    def run_shp2pg(input_shp):
        """
        input_shp is full path to shapefile including file ending
        usage:  run_shp2pg('/home/geodata/myshape.shp')
        """

        db_schema = "SCHEMA=geodata"
        db_connection = """PG:host=localhost port=5432
                        user=pluto dbname=py_geoan_cb password=stars"""
        output_format = "PostgreSQL"
        overwrite_option = "OVERWRITE=YES"
        shp_dataset = shp_driver.Open(input_shp)
        layer = shp_dataset.GetLayer(0)
        geometry_type = layer.GetLayerDefn().GetGeomType()
        geometry_name = discover_geom_name(geometry_type)
        print (geometry_name)

        subprocess.call(["ogr2ogr", "-lco", db_schema, "-lco", overwrite_option,
                         "-nlt", geometry_name, "-skipfailures",
                         "-f", output_format, db_connection, input_shp])

    # directory full of shapefiles
    shapefile_dir = os.path.realpath('../geodata')

    # define the ogr spatial driver type
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')

    # empty list to hold names of all shapefils in directory
    shapefile_list = []

    for shp_file in os.listdir(shapefile_dir):
        if shp_file.endswith(".shp"):
            # apped join path to file name to outpout "../geodata/myshape.shp"
            full_shapefile_path = os.path.join(shapefile_dir, shp_file)
            shapefile_list.append(full_shapefile_path)

    # loop over list of Shapefiles running our import function
    for each_shapefile in shapefile_list:
        run_shp2pg(each_shapefile)
        print ("importing Shapefile: " + each_shapefile)
    ```

1.  现在，我们可以再次从命令行简单地运行我们的新脚本，如下所示：

    ```py
    $ python ch03-02_batch_shp2pg.py

    ```

## 它是如何工作的...

在这里，我们正在重用之前脚本中的代码，但已将其转换为名为`run_shp2pg(input_shp)`的 Python 函数，该函数接受一个参数，即我们想要导入的 Shapefile 的完整路径。输入参数必须包含 Shapefile 扩展名，`.shp`。

我们有一个辅助函数，它通过读取 Shapefile 要素层并输出几何类型作为字符串来获取几何类型，这样`ogr`命令就知道期待什么。这并不总是有效，可能会发生一些错误。`–skipfailures`选项将忽略插入过程中抛出的任何错误，并继续填充我们的表。

首先，我们需要定义包含所有待导入 Shapefiles 的文件夹。接下来，我们可以创建一个名为`shapefile_list`的空列表对象，它将保存我们想要导入的所有 Shapefiles 的列表。

第一个`for`循环使用标准 Python `os.listdir()`函数获取指定目录中所有 Shapefiles 的列表。我们不想获取这个文件夹中的所有文件。我们只想获取以`.shp`结尾的文件；因此，有一个`if`语句，如果文件以`.shp`结尾，则评估为`True`。一旦找到`.shp`文件，我们需要将文件路径和文件名连接起来，创建一个包含路径和 Shapefile 名称的单个字符串，即`full_shapefile_path`变量。在最后部分，我们将每个新文件及其附加路径添加到我们的`shapefile_list`列表对象中，以便我们可以遍历最终的列表。

现在，是时候遍历我们新列表中的每个 Shapefile，并对列表中的每个 Shapefile 运行我们的`run_shp2pg(input_shp)`函数，将其导入到我们的 PostgreSQL PostGIS 数据库中。

## 还有更多...

如果你有很多 Shapefiles（我的意思是真的很多，比如 100 个或更多），性能将是一个考虑因素，因此将需要很多具有空闲资源的机器。

# 从 PostGIS 批量导出表到 Shapefiles

现在，我们将改变方向，看看我们如何可以从 PostGIS 数据库批量导出一系列表到 Shapefiles 文件夹。我们将在 Python 脚本中使用 ogr2ogr 命令行工具，这样你就可以将其包含在你的应用程序编程工作流程中。在接近结尾的地方，你还可以看到所有这些是如何在一个单独的命令行中完成的。

## 如何操作...

1.  以下脚本将触发 `ogr2ogr` 命令并遍历表列表以将 Shapefile 格式导出到现有文件夹。因此，让我们看看如何按照以下步骤进行：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    #
    import subprocess
    import os

    # folder to hold output Shapefiles
    destination_dir = os.path.realpath('../geodata/temp')

    # list of postGIS tables
    postgis_tables_list = ["bikeways", "highest_mountains"]

    # database connection parameters
    db_connection = """PG:host=localhost port=5432 user=pluto
            dbname=py_geoan_cb password=stars active_schema=geodata"""

    output_format = "ESRI Shapefile"

    # check if destination directory exists
    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)
        for table in postgis_tables_list:
            subprocess.call(["ogr2ogr", "-f", output_format, destination_dir,
                             db_connection, table])
            print("running ogr2ogr on table: " + table)
    else:
        print("oh no your destination directory " + destination_dir +
              " already exist please remove it then run again")

    # commandline call without using python will look like this
    # ogr2ogr -f "ESRI Shapefile" mydatadump \
    # PG:"host=myhost user=myloginname dbname=mydbname password=mypassword" neighborhood parcels
    ```

1.  现在，我们将按照以下方式从命令行调用我们的脚本：

    ```py
    $ python ch03-03_batch_postgis2shp.py

    ```

## 它是如何工作的...

从简单的 `subprocess` 和 `os` 模块导入开始，我们立即定义了我们想要存储导出 Shapefiles 的目标目录。该变量后面跟着我们想要导出的表名列表。此列表只能包括位于同一 PostgreSQL 模式中的文件。该模式定义为 `active_schema`，这样 `ogr2ogr` 就知道在哪里找到要导出的表。

再次强调，我们将输出格式定义为**ESRI Shapefile**。现在，我们将检查目标文件夹是否存在。如果存在，我们将继续并调用我们的循环。然后，我们将遍历存储在 `postgis_tables_list` 变量中的表列表。如果目标文件夹不存在，您将在屏幕上看到错误信息。

## 更多内容...

编写应用程序并从脚本内部执行 ogr2ogr 命令确实既快又简单。另一方面，对于一次性工作，当导出 Shapefile 列表时，您只需执行命令行工具即可。为了以一行命令完成此操作，请参阅以下信息框。

### 注意

如果您只想执行一次而不在脚本环境中执行，以下是一个调用 ogr2ogr 批量 PostGIS 表到 Shapefiles 的一行示例：

```py
ogr2ogr -f "ESRI Shapefile" /home/ch03/geodata/temp PG:"host=localhost user=pluto dbname=py_geoan_cb password=stars" bikeways highest_mountains

```

您想要导出的表列表位于末尾，由空格分隔。导出 Shapefiles 的目标位置是 `../geodata/temp`。请注意，此 `/temp` 目录必须存在。

# 将 OpenStreetMap (OSM) XML 转换为 Shapefile

OpenStreetMap (OSM) 拥有丰富的免费数据，但为了与其他大多数应用程序一起使用，我们需要将其转换为其他格式，例如 Shapefile 或 PostgreSQL PostGIS 数据库。本食谱将使用 **ogr2ogr** 工具在 Python 脚本中为我们执行转换。这种方法的优点再次是简单性。

## 准备工作

要开始，您需要下载 OSM 数据，请访问 [`www.openstreetmap.org/export#map=17/37.80721/-122.47305`](http://www.openstreetmap.org/export#map=17/37.80721/-122.47305) 并将文件（`.osm`）保存到您的 `/ch03/geodata` 目录中。下载按钮位于左侧栏上，按下后应立即开始下载（参见图表）。我们正在测试的区域位于旧金山，就在**金门大桥**之前。

![准备工作](img/B03543_03_03.jpg)

如果你选择从 OSM 下载另一个区域，请随意，但请确保你选择一个与我示例相似的小区域。如果你选择一个更大的区域，OSM 网络工具会给出警告并禁用下载按钮。原因很简单：如果数据集非常大，它可能更适合其他工具，例如**osm2pgsql**([`wiki.openstreetmap.org/wiki/Osm2pgsql`](http://wiki.openstreetmap.org/wiki/Osm2pgsql))进行转换。如果你需要获取大区域的 OSM 数据并将其导出为 Shapefile，建议使用其他工具，例如**osm2pgsql**，它首先将你的数据导入 PostgreSQL 数据库。然后，使用**pgsql2shp**工具从 PostGIS 数据库导出数据到 Shapefile。

### 小贴士

一个名为**imposm**的 Python 工具可以用来将 OSM 数据导入 PostGIS 数据库，并且可以在[`imposm.org/`](http://imposm.org/)找到。它的第 2 版是用 Python 编写的，第 3 版是用*go*编程语言编写的，如果你想尝试这个，也可以。

## 如何操作...

使用以下步骤将 OpenStreetMap (OSM) XML 转换为 Shapefile：

1.  使用子进程模块，我们将执行**ogr2ogr**将我们下载的 OSM 数据转换为新的 Shapefile：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    # convert / import osm xml .osm file into a Shapefile
    import subprocess
    import os
    import shutil

    # specify output format
    output_format = "ESRI Shapefile"

    # complete path to input OSM xml file .osm
    input_osm = '../geodata/OSM_san_francisco_westbluff.osm'

    # Windows users can uncomment these two lines if needed
    # ogr2ogr = r"c:/OSGeo4W/bin/ogr2ogr.exe"
    # ogr_info = r"c:/OSGeo4W/bin/ogrinfo.exe"

    # view what geometry types are available in our OSM file
    subprocess.call([ogr_info, input_osm])

    destination_dir = os.path.realpath('../geodata/temp')

    if os.path.isdir(destination_dir):
        # remove output folder if it exists
        shutil.rmtree(destination_dir)
        print("removing existing directory : " + destination_dir)
        # create new output folder
        os.mkdir(destination_dir)
        print("creating new directory : " + destination_dir)

        # list of geometry types to convert to Shapefile
        geom_types = ["lines", "points", "multilinestrings", "multipolygons"]

        # create a new Shapefile for each geometry type
        for g_type in geom_types:

            subprocess.call([ogr2ogr,
                   "-skipfailures", "-f", output_format,
                     destination_dir, input_osm,
                     "layer", g_type,
                     "--config","OSM_USE_CUSTOM_INDEXING", "NO"])
            print("done creating " + g_type)

    # if you like to export to SPATIALITE from .osm
    # subprocess.call([ogr2ogr, "-skipfailures", "-f",
    #         "SQLITE", "-dsco", "SPATIALITE=YES",
    #         "my2.sqlite", input_osm])
    ```

1.  现在，我们可以从命令行调用我们的脚本：

    ```py
    $ python ch03-04_osm2shp.py

    ```

前往你的`../geodata`文件夹查看新创建的 Shapefiles，并尝试在 Quantum GIS 中打开它们，Quantum GIS 是一款免费的 GIS 软件([www.qgis.org](http://www.qgis.org))。

## 它是如何工作的...

这个脚本应该很清晰，因为我们使用子进程模块调用来执行 ogr2ogr 命令行工具。我们将指定我们的 OSM 数据集作为输入文件，包括文件的完整路径。Shapefile 的名称不需要提供，因为 ogr2ogr 将输出一系列 Shapefiles，每个 Shapefile 根据在 OSM 文件中找到的几何类型分别对应一个。我们只需要指定我们希望 ogr2ogr 将 Shapefiles 导出到的文件夹名称，如果该文件夹不存在，则会自动创建。

### 注意

Windows 用户：如果你没有将 ogr2ogr 工具映射到你的环境变量中，你可以简单地取消第 16 行和第 17 行的注释，并将显示的路径替换为你机器上 Windows 可执行文件的路径。

第一次子进程调用会在屏幕上打印出我们 OSM 文件中找到的几何类型。这在大多数情况下很有用，可以帮助识别可用内容。Shapefiles 每个文件只能支持一种几何类型，这也是为什么 ogr2ogr 会输出一个包含多个 Shapefiles 的文件夹，每个 Shapefile 代表一个单独的几何类型。

最后，我们调用子进程来执行 ogr2ogr，传入输出文件类型为 ESRI Shapefile，输出文件夹和 OSM 数据集的名称。

# 将 Shapefile（矢量）转换为 GeoTiff（栅格）

在格式之间移动数据也包括从矢量到栅格或相反。在这个菜谱中，我们使用 Python 的`gdal`和`ogr`模块将数据从矢量（Shapefile）移动到栅格（GeoTiff）。

## 准备工作

我们需要再次进入我们的虚拟环境，所以启动它，这样我们就可以访问我们在第一章中安装的`gdal`和`ogr`Python 模块，*设置您的地理空间 Python 环境*。

如同往常，使用`workon pygeoan_cb`命令或此命令进入您的 Python 虚拟环境：

```py
$ source venvs/pygeoan_cb/bin/activate

```

还需要一个 Shapefile，所以请确保下载源文件并访问`/ch03/geodata`文件夹（[`github.com/mdiener21/python-geospatial-analysis-cookbook/archive/master.zip`](https://github.com/mdiener21/python-geospatial-analysis-cookbook/archive/master.zip)）。

## 如何操作...

让我们深入进去，将我们的高尔夫球场多边形 Shapefile 转换为 GeoTif；下面是代码：

1.  导入`ogr`和`gdal`库，然后定义我们的输出像素大小以及分配给空值的值：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    from osgeo import ogr
    from osgeo import gdal

    # set pixel size
    pixel_size = 1
    no_data_value = -9999
    ```

1.  设置我们想要转换的输入 Shapefile，以及当脚本执行时将创建的新 GeoTiff 栅格：

    ```py
    # Shapefile input name
    # input projection must be in Cartesian system in meters
    # input wgs 84 or EPSG: 4326 will NOT work!!!
    input_shp = r'../geodata/ply_golfcourse-strasslach3857.shp'

    # TIF Raster file to be created
    output_raster = r'../geodata/ply_golfcourse-strasslach.tif'
    ```

1.  现在我们需要创建输入 Shapefile 对象，获取图层信息，并最终设置范围值：

    ```py
    # Open the data source get the layer object
    # assign extent coordinates
    open_shp = ogr.Open(input_shp)
    shp_layer = open_shp.GetLayer()
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    ```

1.  在这里，我们需要计算分辨率距离到像素值的转换：

    ```py
    # calculate raster resolution
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    ```

1.  我们的新栅格类型是 GeoTiff，因此我们必须明确告诉 GDAL 获取此驱动程序。然后，驱动程序能够通过传递文件名或我们想要创建的新栅格（称为*x*方向分辨率），然后是*y*方向分辨率，接着是波段数；在这种情况下，是 1。最后，我们设置了一种新的`GDT_Byte`栅格类型：

    ```py
    # set the image type for export
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    new_raster = driver.Create(output_raster, x_res, y_res, 1, gdal.GDT_Byte)
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    ```

1.  现在我们可以访问新的栅格波段，并为新的栅格分配无数据值和内部数据值。所有内部值都将接收一个值为 255，类似于我们在`burn_values`变量中设置的值：

    ```py
    # get the raster band we want to export too
    raster_band = new_raster.GetRasterBand(1)

    # assign the no data value to empty cells
    raster_band.SetNoDataValue(no_data_value)

    # run vector to raster on new raster with input Shapefile
    gdal.RasterizeLayer(new_raster, [1], shp_layer, burn_values=[255])
    ```

1.  我们开始了；让我们运行这个脚本来看看我们的新栅格是什么样子：

    ```py
    $ python ch03-05_shp2raster.py

    ```

如果您使用**QGIS**（[`www.qgis.org`](http://www.qgis.org)）打开，我们的结果栅格应该看起来像以下截图所示：

![如何操作...](img/B03543_03_04.jpg)

## 它是如何工作的...

这个代码涉及几个步骤，所以请跟随，因为一些点可能会导致问题，如果你不确定要输入什么值。我们首先导入*gdal*和*ogr*模块，因为它们将通过输入 Shapefile（矢量）和输出 GeoTiff（栅格）为我们完成工作。

`pixel_size`变量非常重要，因为它将决定我们将创建的新栅格的大小。在这个例子中，我们只有两个多边形，所以我们设置`pixel_size = 1`以保持它们之间精细的边界。如果你有一个 Shapefile 中跨越全球的许多多边形，更明智的做法是将此值设置为 25 或更多。否则，你可能会得到一个 10GB 的栅格，你的机器将整夜运行！`no_data_value`是必需的，以告诉 GDAL 在输入多边形周围的空空间中设置什么值，我们将其设置为`-9999`以便于识别。

接下来，我们简单地设置输入的 Shapefile 存储在 EPSG:3857 Web Mercator 和输出 GeoTiff。如果你想使用其他数据集，请确保相应地更改文件名。我们首先使用 OGR 模块打开 Shapefile 并检索其层信息和范围信息。范围很重要，因为它用于计算输出栅格的宽度和高度值，这些值必须是整数，由`x_res`和`y_res`变量表示。

### 注意

注意，你的 Shapefile 投影必须是米为单位，而不是度。这一点非常重要，因为例如在 EPSG:4326, WGS 84 中，这将不会工作。原因在于坐标单位是经纬度。这意味着 WGS84 不是一个平面投影，不能直接绘制。我们的`x_res`和`y_res`值将评估为 0，因为我们无法使用度来获得真实的比例。这是由于我们无法简单地从坐标*x*中减去坐标*y*，因为单位是度而不是平面米投影。

现在，让我们继续到栅格设置，我们定义要导出的栅格类型为`Gtiff`。然后，我们将通过栅格类型获取正确的 GDAL 驱动程序。一旦设置栅格类型，我们就可以创建一个新的空栅格数据集，传入栅格文件名、宽度、栅格的像素高度、栅格波段数，以及最后在 GDAL 术语中的栅格类型，例如`gdal.GDT_Byte`。这五个参数是创建新栅格的必填项。

接下来，我们调用`SetGeoTransform`，它处理像素/行栅格空间和投影坐标空间之间的转换。我们希望激活`波段 1`，因为这是我们栅格中唯一的波段。然后，我们将所有围绕多边形的空空间分配为无数据值。

最后一步是调用`gdal.RasterizeLayer()`函数，并传入我们的新栅格、波段、Shapefile 以及分配给栅格内部的值。所有在多边形内部的像素将被分配值为 255。

## 参见

如果你感兴趣，可以访问`gdal_rasterize`命令行工具[`www.gdal.org/gdal_rasterize.html`](http://www.gdal.org/gdal_rasterize.html)。你可以直接从命令行运行它。

# 使用 GDAL 将栅格（GeoTiff）转换为矢量（Shapefile）

我们已经看到了如何从矢量转换为栅格，现在是时候从栅格转换为矢量了。这种方法更为常见，因为我们的大部分矢量数据都来源于遥感数据，如卫星图像、正射影像或某些其他遥感数据集，如`lidar`。

## 准备工作

如同往常，在您的 Python 虚拟环境中输入`workon pygeoan_cb`命令：

```py
$ source venvs/pygeoan_cb/bin/activate

```

## 如何操作...

这个食谱只需要四个步骤利用 OGR 和 GDAL，所以请为您的代码打开一个新文件：

1.  导入`ogr`和`gdal`模块，并直接打开我们想要转换的栅格，通过传递磁盘上的文件名并获取一个栅格波段：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    from osgeo import ogr
    from osgeo import gdal

    #  get raster data source
    open_image = gdal.Open( "../geodata/cadaster_borders-2tone-black-white.png" )
    input_band = open_image.GetRasterBand(3)
    ```

1.  将输出矢量文件设置为 Shapefile 格式，使用 output_shp，然后获取一个 Shapefile 驱动程序。现在，我们可以从我们的驱动程序创建输出，并创建一个图层，如下所示：

    ```py
    #  create output data source
    output_shp = "../geodata/cadaster_raster"
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")

    # create output file name
    output_shapefile = shp_driver.CreateDataSource( output_shp + ".shp" )
    new_shapefile = output_shapefile.CreateLayer(output_shp, srs = None )
    ```

1.  最后一步是运行`gdal.Polygonize`函数，它通过将我们的栅格转换为矢量来完成繁重的工作，如下所示：

    ```py
    gdal.Polygonize(input_band, None, new_shapefile, -1, [], callback=None)
    new_shapefile.SyncToDisk()
    ```

1.  按照以下方式执行新脚本：

    ```py
    $ python ch03-06_raster2shp.py

    ```

## 它是如何工作的...

在所有我们的食谱中，使用`ogr`和`gdal`的方式相似；我们必须定义输入并获取适当的文件驱动程序来打开文件。GDAL 库非常强大，我们只需一行代码就可以通过`gdal.Polygonize`函数将栅格转换为矢量。前面的代码仅仅是设置代码，用于定义我们想要使用哪种格式，然后设置适当的驱动程序来输入和输出我们的新文件。

# 从存储在 Microsoft Excel 中的点数据创建 Shapefile

Excel 文件现在非常普遍，分析师或开发者经常收到需要映射的 Excel 文件。当然，我们可以将其保存为`.csv`文件，然后使用伟大的 Python 标准*csv*模块，但这需要额外的手动步骤。我们将看看如何读取一个包含欧洲最高山脉列表的非常简单的 Excel 文件。这个数据集来源于[`www.geonames.org`](http://www.geonames.org)。

## 准备工作

我们将需要一个新 Python 库来读取 Microsoft Excel 文件，这个库是**xlrd** ([`www.python-excel.org`](http://www.python-excel.org))。

### 注意

这个库只能读取 Excel 文件；如果您想要写入 Excel 文件，请下载并安装**xlwt**。

首先，从您的`workon pygeoan_cb` Linux 机器启动虚拟环境，运行`pip install xlrd`，然后您就可以开始比赛了。

要写入新的 Shapefile，我们将使用我们在第一章中安装的 pyshp 库，这样就不需要做任何事情。

数据位于您的下载目录中的`/ch03/geodata`，在您完成这个食谱后，输出 Shapefile 也将被写入这个位置。

## 如何操作...

因此，让我们从一些代码开始：

1.  首先导入`xlrd`和 pyshp 模块；注意导入名称是`shapefile`，而不是模块名称所暗示的 pyshp：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    import xlrd
    import shapefile
    ```

1.  使用 xlrd 模块打开 Excel 文件，并创建一个变量来保存 Excel 工作表。我们通过索引号引用 Excel 文件中的第一个工作表，始终从第一个工作表的(0)开始：

    ```py
    excel_file = xlrd.open_workbook("../geodata/highest-mountains-europe.xlsx")

    # get the first sheet
    sh = excel_file.sheet_by_index(0)
    ```

1.  按照以下方式创建 Shapefile 对象：

    ```py
    w = shapefile.Writer(shapefile.POINT)
    ```

1.  定义新的 Shapefile 字段及其数据类型。*F*代表浮点数，*C*代表字符：

    ```py
    w.field('GeoNameId','F')
    w.field('Name', 'C')
    w.field('Country', 'C')
    w.field('Latitude', 'F')
    w.field('Longitude', 'F')
    w.field('Altitude', 'F')
    ```

1.  遍历 Excel 文件中的每一行，并创建几何值及其属性：

    ```py
    for row_number in range(sh.nrows):
        # skips over the first row since it is the header row
        if row_number == 0:
            continue
        else:
            x_coord = sh.cell_value(rowx=row_number, colx=4)
            y_coord = sh.cell_value(rowx=row_number, colx=3)
            w.point(x_coord, y_coord)

            w.record(GeoNameId=sh.cell_value(rowx=row_number, colx=0), Name=sh.cell_value(rowx=row_number, colx=1),
                     Country=sh.cell_value(rowx=row_number, colx=2), Latitude=sh.cell_value(rowx=row_number, colx=3),
                     Longitude=sh.cell_value(rowx=row_number, colx=4),Altitude=sh.cell_value(rowx=row_number, colx=5))
            print "Adding row: " + str(row_number) + " creating mount: " + sh.cell_value(rowx=row_number, colx=1)
    ```

1.  最后，我们将在`/ch03/geodata`文件夹中创建新的 Shapefile，如下所示：

    ```py
    w.save('../geodata/highest-mountains')
    ```

1.  按照以下方式从命令行执行我们新的`ch03-07_excel2shp.py`脚本：

    ```py
    $ python ch03-07_excel2shp.py

    ```

## 它是如何工作的...

Python 代码的阅读方式类似于描述代码的工作方式，而且几乎所有的解释都非常简单。我们首先导入新的*xlrd*模块以及写入 Shapefile 所需的 Shapefile 模块。查看我们的 Excel 文件，我们可以看到哪些字段可用，并定位到*x*坐标（经度）和*y*坐标（纬度）的位置。这个位置索引号通过从 0 开始计数来记住起始点。

我们的 Excel 文件还有一个标题行，当然，这个标题行不应该包含在新数据属性中；这就是为什么我们要检查行号是否等于 0——即第一行——然后继续。continue 语句允许代码继续执行而不会出错，并进入`else`语句，在那里我们定义列的索引位置。每个列都使用`pyshp`语法引用，通过名称引用列，使代码更容易阅读。

我们调用`w.point` pyshp 函数来创建点几何形状，传入我们的 x 和 y 坐标作为浮点数。`xlrd`模块会自动将值转换为浮点数，这很方便。我们最终需要做的只是使用 pyshp 的保存函数将数据写入我们的`/ch03/geodata`文件夹。不需要添加`.shp`扩展名；pyshp 会为我们处理并输出`.shp`、`.dbf`和`.shx`。

### 注意

注意，`.prj`投影文件不会自动输出。如果您希望将投影信息一起导出，您需要手动创建它，如下所示：

```py
# create the PRJ file
filename = 'highest-mountains'
prj = open("%s.prj" % filename, "w")
epsg = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
prj.write(epsg)
prj.close()
```

# 将 ESRI ASCII DEM 转换为图像高度图

为了让这一章有一个圆满的结尾，这里是我们迄今为止见过的最复杂的转换，也是最有趣的。输入是一个存储在*ASCII*格式中的高程数据集，更具体地说，是 Arc/Info ASCII Grid，简称 AAIGrid，文件扩展名为(`.asc`)。我们的输出是一个*高度图*图像([`en.wikipedia.org/wiki/Heightmap`](http://en.wikipedia.org/wiki/Heightmap))。高度图图像是一种存储高度高程为像素值的图像。高度图也简单地称为**数字高程模型**(**DEM**)。使用图像存储高程数据的优点是它是*网络兼容的*，我们可以使用它进行 3D 可视化，例如，如第十章中所示，*可视化您的分析*。

我们需要小心处理输出图像格式，因为仅仅存储 8 位图像将限制我们只能存储 0 到 255 的高度值，这通常是不够的。输出图像应存储至少 16 位，给我们一个从-32,767 到 32,767 的范围。如果我是正确的，地球上最高的山是珠穆朗玛峰，高度为 8,848 米，所以 16 位图像应该足够存储我们的高程数据。

## 准备工作

运行此练习需要一个 DEM，请确保您已下载了包含在[`github.com/mdiener21/python-geospatial-analysis-cookbook/archive/master.zip`](https://github.com/mdiener21/python-geospatial-analysis-cookbook/archive/master.zip)中的代码和地理数据，并下载所需的示例 DEM 进行处理。您不需要在虚拟环境中运行您的脚本，因为此脚本将执行标准 Python 模块和与 GDAL 一起安装的几个 GDAL 内置工具。这仅仅意味着您需要确保您的 GDAL 实用程序已正确安装并在您的机器上运行。（有关参考安装，请参阅第二章，*处理投影*。）

## 如何做到这一点...

我们将通过在 Python 脚本中调用由`gdal`安装的几个 GDAL 实用脚本来执行此脚本：

1.  我们将首先导入`subprocess`标准模块；这将用于执行我们的 GDAL 实用函数。然后，我们将设置基路径，我们将在这里存储我们的地理数据，包括输入文件、临时文件和输出文件：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    import subprocess
    from osgeo import gdal

    path_base = "../geodata/"
    ```

1.  使用伟大的 OSGeo4w 安装程序安装 GDAL 的 Windows 用户可能希望直接指定 GDAL 实用程序的路径，如果它不在 Windows 环境变量中，如下所示：

    ```py
    # gdal_translate converts raster data between different formats

    command_gdal_translate = "c:/OSGeo4W/bin/gdal_translate.exe"
    command_gdalinfo = "c:/OSGeo4W/bin/gdalinfo.exe"
    ```

1.  Linux 用户可以使用以下变量：

    ```py
    command_gdal_translate = "gdal_translate"
    command_gdalinfo = "gdalinfo"
    command_gdaldem = "gdaldem"
    ```

1.  我们将创建一组变量来存储我们的输入 DEM、输出文件、临时文件以及我们的最终输出文件。这些变量将基路径文件夹与文件名连接起来，如下所示：

    ```py
    orig_dem_asc = path_base + "original_dem.asc"

    temp_tiff = path_base + "temp_image.tif"

    output_envi = path_base + "final_envi.bin"
    ```

1.  然后，我们将调用`gdal_translate`命令来创建我们的新临时 GeoTiff，如下所示：

    ```py
    # transform dem to tiff
    dem2tiff = command_gdal_translate + " " + orig_dem_asc + " " + temp_tiff
    print ("now executing this command: " + dem2tiff)
    subprocess.call(dem2tiff.split(), shell=False)
    ```

1.  接下来，我们将打开临时 GeoTiff，读取关于 tiff 的信息，以找出存储在我们数据中的最小和最大高度值。这虽然不是完成脚本所必需的，但非常有用，可以帮助你识别最大和最小高度值：

    ```py
    ds = gdal.Open(temp_tiff, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    print 'Band Type=', gdal.GetDataTypeName(band.DataType)
    min = band.GetMinimum()
    max = band.GetMaximum()
    if min is None or max is None:
        (min, max) = band.ComputeRasterMinMax(1)
    print 'Min=%.3f, Max=%.3f' % (min, max)
    min_elevation = str(int(round(min)))
    max_elevation = str(int(round(max)))
    ```

1.  然后，使用以下参数调用`gdal_translate`工具，将缩放范围从原始的最小/最大值设置为新的范围，从 0 到 65,535 个值。指定`-ot`输出类型为 vENVI 格式，使用我们的临时 GeoTiff 作为输入：

    ```py
    tif_2_envi = command_gdal_translate + " -scale -ot UInt16 -outsize 500 500 -of ENVI " \
                 + temp_tiff + " " + output_envi
    ```

1.  让我们从命令行运行我们新的`ch03-08_dem2heightmap.py`脚本：

    ```py
    subprocess.call(tif_2_envi.split(),shell=False)

    ```

1.  让我们从命令行运行我们新的`ch03-08_dem2heightmap.py`脚本：

    ```py
    python ch03-08_dem2heightmap.py

    ```

结果是，你会在/ch03/geodata/文件夹中找到一个名为.bin 的新文件，该文件存储了你的新的 ENVI 16 位图像，包括所有你的高程数据。现在，这个高度图可以用于你的 3D 软件，例如 Blender([www.blender.org](http://www.blender.org))、Unity([www.unity3d.com](http://www.unity3d.com))，或者在一个更酷的 Web 应用程序中使用 JavaScript 库，如`threejs`。

## 它是如何工作的...

让我们从导入开始，然后指定我们的输入和输出存储的基本路径。之后，我们将看到实际使用的`gdal_translate`转换命令。Windows 和 Linux 的命令由你自己决定是否使用，这取决于你如何设置你的机器。然后，我们设置变量来定义输入 DEM、临时 GeoTiff 和输出 ENVI 高度图图像。

最后，我们可以使用`gdal_translate`工具将我们的 DEM ASCII 文件转换为 GeoTiff 格式的第一次转换。现在为了获取我们数据的一些信息，我们将最小和最大高度值打印到屏幕上。在转换过程中，这非常有用，可以让你检查输出数据是否确实包含了输入的高度值，并且在转换过程中没有出现错误。

最后，我们只需再次调用`gdal_translate`工具，将我们的 GeoTiff 转换为 ENVI 高度图图像。`-scale`参数没有参数时，会自动将我们的 16 位图像填充为从 0 到 65,535 的值。下一个参数是`-ot`，指定输出类型为 16 位，后面跟着`-outsize 500 500`，设置输出图像大小为 500 x 500 像素。最后，`-of ENVI`是我们的输出格式，后面跟着输入 GeoTiff 的名称和输出高度图的名称。

使用 DEM 时的一个典型工作流程如下：

1.  下载一个 DEM，通常是一个非常大的文件，覆盖一个大的地理区域。

1.  将 DEM 裁剪到较小的感兴趣区域。

1.  将裁剪区域转换为另一种格式。

1.  将 DEM 导出为高度图图像。

### 注意

我们介绍了 `.split()` 方法，它将返回一个由字符分隔的 Python 字符串列表。在我们的例子中，分隔字符是一个 *单个空格* 字符，但你也可以根据任何其他字符或字符组合进行分割（请参阅 Python 文档中的[`docs.python.org/2/library/string.html#string.split`](https://docs.python.org/2/library/string.html#string.split)）。这有助于我们减少在代码中需要执行的连接操作数量。
