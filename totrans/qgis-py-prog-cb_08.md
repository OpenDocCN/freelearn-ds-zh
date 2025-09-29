# 第八章 QGIS 工作流程

在本章中，我们将介绍以下食谱：

+   创建 NDVI

+   地理编码地址

+   创建栅格足迹

+   执行网络分析

+   沿街道进行路由

+   跟踪 GPS

+   创建图集

+   寻找最低成本路径

+   执行最近邻分析

+   创建热图

+   创建点密度图

+   收集现场数据

+   使用高程数据计算道路坡度

+   在地图上定位照片

+   图像变化检测

# 简介

在本章中，我们将使用 Python 在 QGIS 中执行各种常见的地理空间任务，这些任务可能是完整的流程，或者是更大流程的关键部分。

# 创建 NDVI

**归一化植被指数**（**NDVI**）是用于检测感兴趣区域内绿色植被的最古老的遥感算法之一，它使用图像的红光和近红外波段。植物中的叶绿素吸收可见光，包括红光波段，而植物细胞的结构反射近红外光。NDVI 公式提供了近红外光与总入射辐射的比率，这作为植被密度的指标。本食谱将使用 Python 控制 QGIS 栅格计算器，以使用农田的多光谱图像创建 NDVI。

## 准备工作

从[`geospatialpython.googlecode.com/svn/farm-field.tif`](https://geospatialpython.googlecode.com/svn/farm-field.tif)下载图像并将其放置在您的`qgis_data`目录下的`rasters`文件夹中。

## 如何操作...

我们将加载栅格作为 QGIS 栅格图层，执行 NDVI 算法，并最终将颜色渐变应用于栅格，以便我们能够轻松地可视化图像中的绿色植被。为此，我们需要执行以下步骤：

1.  在 QGIS **Python 控制台**中，导入以下库：

    ```py
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    from qgis.analysis import *

    ```

1.  现在，使用以下代码将栅格图像作为图层加载：

    ```py
    rasterName = "farm"
    raster = QgsRasterLayer("/Users/joellawhead/qgis_data/\
    rasters/farm-field.tif", rasterName)

    ```

1.  然后，使用以下代码在 QGIS 栅格计算器中为两个波段创建条目：

    ```py
    ir = QgsRasterCalculatorEntry()
    r = QgsRasterCalculatorEntry()

    ```

1.  现在，使用以下代码行，将栅格图层分配为每个计算器条目的栅格组件：

    ```py
    ir.raster = raster
    r.raster = raster

    ```

1.  为每个条目选择适当的波段，以便计算器使用我们需要的 NDVI 数据。红光和红外波段编号通常列在栅格的元数据中：

    ```py
    ir.bandNumber = 2
    r.bandNumber = 1

    ```

1.  接下来，使用特殊的 QGIS 命名约定为每个条目分配一个参考 ID，如以下示例所示，以图层的名称作为前缀，后跟一个`@`符号和波段编号作为后缀：

    ```py
    ir.ref = rasterName + "@2"
    r.ref = rasterName + "@1"

    ```

1.  使用以下代码构建栅格计算器表达式：

    ```py
    references = (ir.ref, r.ref, ir.ref, r.ref)
    exp = "1.0 * (%s - %s) / 1.0 + (%s + %s)" % references

    ```

1.  然后，指定 NDVI 图像的输出名称：

    ```py
    output = "/Users/joellawhead/qgis_data/rasters/ndvi.tif"

    ```

1.  通过定义栅格的范围、其列和行的宽度和高度以及我们在上一步中定义的栅格条目来设置其余栅格计算器调用的变量：

    ```py
    e = raster.extent()
    w = raster.width()
    h = raster.height()
    entries = [ir,r]

    ```

1.  现在，使用我们的表达式创建 NDVI：

    ```py
    ndvi =  QgsRasterCalculator(exp, output, "GTiff", e, w, h, entries)
    ndvi.processCalculation()

    ```

1.  接下来，将 NDVI 输出作为栅格图层加载：

    ```py
    lyr = QgsRasterLayer(output, "NDVI")

    ```

1.  我们必须对图像执行直方图拉伸，否则值之间的差异将难以看到。拉伸是通过 QGIS 对比增强算法来执行的：

    ```py
    algorithm = QgsContrastEnhancement.StretchToMinimumMaximum
    limits = QgsRaster.ContrastEnhancementMinMax
    lyr.setContrastEnhancement(algorithm, limits)

    ```

1.  接下来，构建一个颜色渐变着色器来着色 NDVI，如下所示：

    ```py
    s = QgsRasterShader()
    c = QgsColorRampShader()
    c.setColorRampType(QgsColorRampShader.INTERPOLATED)

    ```

1.  然后，为图像中的每种颜色添加条目。每个条目由一个下限值范围、一个颜色和一个标签组成。条目中的颜色将从下限值开始，直到遇到一个更高的值或最大值。请注意，我们将使用变量别名来表示 QGIS `ColorRampItem` 对象的极长名称：

    ```py
    i = []
    qri = QgsColorRampShader.ColorRampItem
    i.append(qri(0, QColor(0,0,0,0), 'NODATA')) 
    i.append(qri(214, QColor(120,69,25,255), 'Lowest Biomass'))
    i.append(qri(236, QColor(255,178,74,255), 'Lower Biomass'))
    i.append(qri(258, QColor(255,237,166,255), 'Low Biomass'))
    i.append(qri(280, QColor(173,232,94,255), 'Moderate Biomass'))
    i.append(qri(303, QColor(135,181,64,255), 'High Biomass'))
    i.append(qri(325, QColor(3,156,0,255), 'Higher Biomass'))
    i.append(qri(400, QColor(1,100,0,255), 'Highest Biomass'))

    ```

1.  现在，我们可以将条目添加到着色器并将其应用于图像：

    ```py
    c.setColorRampItemList(i)
    s.setRasterShaderFunction(c)
    ps = QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), 1, s)
    lyr.setRenderer(ps)

    ```

1.  最后，将分类的 NDVI 图像添加到地图中以可视化它：

    ```py
    QgsMapLayerRegistry.instance().addMapLayer(lyr)

    ```

## 它是如何工作的...

QGIS 栅格计算器正如其名所示。它允许你在图像上执行数组数学。QGIS 栅格菜单和处理工具箱都有几个栅格处理工具，但栅格计算器可以执行可以由单个数学方程定义的自定义分析。NDVI 算法是红外波段减去红波段除以红外波段加上红波段，即 *(IR-R)/(IR+R)*。在我们的计算器表达式中，我们将方程的两边都乘以 `1.0` 以避免除以零错误。如果你将结果加载到 QGIS 中，你的输出应该看起来类似于以下图像。在这个屏幕截图中，`NODATA` 值用黑色表示；然而，你的 QGIS 安装可能默认使用白色：

![它是如何工作的...](img/00054.jpeg)

# 地理编码地址

地理编码是将地址转换为地球坐标的过程。地理编码需要一个综合的数据集，将邮政编码、城市、街道和街道号码（或街道号码范围）与坐标关联起来。为了拥有适用于世界上任何地址且精度合理的地理编码器，你需要使用云服务，因为地理编码数据集非常密集，可能相当大。为超过几平方英里的任何区域创建地理编码数据集需要大量的资源。有几个服务可用，包括 Google 和 MapQuest。在 QGIS 中，通过 QGIS Python 地理编码插件访问这些服务是最简单的方法。在这个菜谱中，我们将使用此插件以编程方式对地址进行地理编码。

## 准备工作

为了进行这个练习，你需要安装由 Alessandro Pasotti 开发的 QGIS Python 地理编码插件，如下所示：

1.  从 QGIS **插件** 菜单中选择 **管理并安装插件…**

1.  在 **插件** 对话框的搜索框中，搜索 `地理编码`。

1.  选择 **地理编码** 插件并点击 **安装插件** 按钮。

## 如何操作...

在这个菜谱中，我们将使用 Python 访问地理编码插件方法，向插件提供一个地址，并打印出结果坐标。为此，我们需要执行以下步骤：

1.  在 QGIS **Python 控制台** 中，使用以下代码导入 OpenStreetMap 的 `geoCoding` 对象：

    ```py
    from GeoCoding.geopy.geocoders import Nominatim

    ```

1.  接下来，我们将创建我们的地理编码器：

    ```py
    geocoder = Nominatim()

    ```

1.  然后，使用以下代码，我们将对地址进行地理编码：

    ```py
    location = geocoder.geocode("The Ugly Pirate, Bay Saint Louis, MS 39520")

    ```

1.  最后，我们将打印结果以查看坐标：

    ```py
    print location

    ```

1.  检查是否在控制台打印出以下输出：

    ```py
    (u'The Ugly Pirate, 144, Demontluzin Street, Bay St. Louis, Hancock County, Mississippi, 39520, United States of America', (30.3124059, -89.3281418))

    ```

## 它是如何工作的...

**地理编码**插件旨在与 QGIS GUI 界面一起使用。然而，像大多数 QGIS 插件一样，它是用 Python 编写的，我们可以通过 Python 控制台访问它。

### 小贴士

这个技巧并不适用于每个插件。有时，用户界面与插件的 GUI 结合得太紧密，以至于你无法在不触发 GUI 的情况下程序化地使用插件的方法。

然而，在大多数情况下，你可以使用插件不仅扩展 QGIS，还可以利用其强大的 Python API。如果你自己编写插件，考虑使其可访问 QGIS Python 控制台，以便使其更有用。

## 更多内容...

地理编码插件还提供 Google 地理编码引擎作为一项服务。请注意，Google 地图 API，包括地理编码，附带一些限制，可以在[`developers.google.com/maps-engine/documentation/limits`](https://developers.google.com/maps-engine/documentation/limits)找到。

# 创建栅格足迹

对由大量文件组成的栅格数据集进行编目的一种常见方法是通过创建一个包含每个栅格文件范围的多边形足迹的矢量数据集。矢量足迹文件可以轻松地加载到 QGIS 中或在网络上提供服务。本菜谱演示了一种从充满栅格文件目录创建足迹矢量的方法。我们将把这个程序作为一个处理工具箱脚本构建，这比构建 QGIS 插件更容易，并且提供了一个 GUI 和一个干净的编程 API。

## 准备工作

从[`geospatialpython.googlecode.com/svn/scenes.zip`](https://geospatialpython.googlecode.com/svn/scenes.zip)下载示例栅格图像场景。将`scenes`目录解压缩到您的`qgis_data`目录下的名为`rasters`的目录中。

对于这个菜谱，我们将按照以下步骤创建一个新的处理工具箱脚本：

1.  在 QGIS 处理工具箱中，展开**脚本**树菜单。

1.  接下来，展开**工具**树菜单。

1.  最后，双击**创建新脚本**项以打开处理脚本编辑器。

## 如何操作...

首先，我们将使用处理工具箱的命名约定，这将同时定义我们的 GUI 以及输入和输出变量。然后，我们将创建逻辑，该逻辑处理栅格目录并计算图像范围，最后我们将创建矢量文件。为此，我们需要执行以下步骤：

1.  首先，我们使用注释来定义我们的输入变量，告诉 Processing Toolbox 在脚本被用户调用时将这些添加到 GUI 中。第一个条目定义了脚本的分组菜单，将我们的脚本放置在工具箱中，第二个条目定义了包含栅格的目录，第三个条目是我们 shapefile 的输出名称。脚本必须以这些注释开始。每个条目还声明了 Processing Toolbox API 允许的类型。这些注释中变量的名称对脚本可用：

    ```py
    ##Vector=group
    ##Input_Raster_Directory=folder
    ##Output_Footprints_Vector=output vector

    ```

1.  接下来，我们导入我们将需要的 Python 库，使用以下命令：

    ```py
    import os
    from qgis.core import *

    ```

1.  现在，我们获取栅格目录中的文件列表。以下脚本没有尝试根据类型过滤文件。如果目录中有其他类型的非栅格文件，它们也将被包括在内：

    ```py
    files = os.listdir(Input_Raster_Directory)

    ```

1.  然后，我们声明几个变量，它们将保存我们的栅格范围和坐标参考字符串，如下所示：

    ```py
    footprints = []
    crs = ""

    ```

1.  现在，我们遍历栅格，将它们作为栅格图层加载以获取它们的范围，将它们作为 Python 字典中的点数据存储，并将它们添加到我们的临时存储的足迹列表中。如果栅格无法处理，将使用 Processing Toolbox 进度对象发出警告：

    ```py
    for f in files:
     try:
     fn = os.path.join(Input_Raster_Directory, f)
     lyr = QgsRasterLayer(fn, "Input Raster")
     crs = lyr.crs()
     e = lyr.extent()
     ulx = e.xMinimum()
     uly = e.yMaximum()
     lrx = e.xMaximum()
     lry = e.yMinimum()
     ul = (ulx, uly)
     ur = (lrx, uly)
     lr = (lrx, lry)
     ll = (ulx, lry)
     fp = {}
     points = []
     points.append(QgsPoint(*ul))
     points.append(QgsPoint(*ur))
     points.append(QgsPoint(*lr)) 
     points.append(QgsPoint(*ll)) 
     points.append(QgsPoint(*ul))
     fp["points"] = points
     fp["raster"] = fn
     footprints.append(fp)
     except:
     progress.setInfo("Warning: The file %s does not appear to be a \
    valid raster file." % f)

    ```

1.  使用以下代码，我们将创建一个内存矢量图层来在写入 shapefile 之前构建足迹矢量：

    ```py
    vectorLyr =  QgsVectorLayer("Polygon?crs=%s&field=raster:string(100)" \
    % crs, "Footprints" , "memory")
    vpr = vectorLyr.dataProvider()

    ```

1.  现在，我们将我们的范围列表转换为要素：

    ```py
    features = []
    for fp in footprints:
     poly = QgsGeometry.fromPolygon([fp["points"]])
     f = QgsFeature()
     f.setGeometry(poly)
     f.setAttributes([fp["raster"]])
     features.append(f)
    vpr.addFeatures(features)
    vectorLyr.updateExtents()

    ```

1.  然后，我们将设置 shapefile 的文件驱动程序和 CRS：

    ```py
    driver = "Esri Shapefile"
    epsg = crs.postgisSrid()
    srs = "EPSG:%s" % epsg

    ```

1.  最后，我们将写入选定的输出文件，指定我们保存到磁盘的图层；输出文件的名称；文件编码，这可能会根据输入而变化；坐标参考系统；以及输出文件类型的驱动程序，在这种情况下是 shapefile：

    ```py
    error = QgsVectorFileWriter.writeAsVectorFormat\ (vectorLyr, Output_Footprints_Vector, \"utf-8", srs, driver)
    if error == QgsVectorFileWriter.NoError:
     pass
    else:
     progress.setInfo("Unable to output footprints.")

    ```

## 它是如何工作的...

记住这一点很重要：Processing Toolbox 脚本可以在几个不同的环境中运行：作为一个 GUI 进程，例如插件，作为一个来自 Python 控制台的程序化脚本，一个 Python 插件，或者图形模型器框架。因此，遵循文档化的 Processing Toolbox API 非常重要，以确保它在这些所有环境中都能按预期工作。这包括定义清晰的输入和输出，并使用进度对象。进度对象是提供进度条和消息反馈给用户的正确方式。尽管 API 允许你定义让用户选择不同 OGR 和 GDAL 输出的输出，但目前似乎只支持 shapefile 和 GeoTiff。

## 还有更多...

Processing Toolbox 内的图形模型器工具允许你将不同的处理算法可视化地链接起来，以创建复杂的工作流程。另一个有趣的插件是 Processing Workflows 插件，它不仅允许你链接算法，还提供了一个带有用户说明的漂亮的选项卡界面，以帮助初学者通过复杂的地理空间工作流程。

下面的截图显示了在 OpenStreetMap 底图上的栅格足迹：

![还有更多...](img/00055.jpeg)

# 执行网络分析

网络分析允许您在定义的网络中找到两点之间的最有效路线。这些线可能代表街道、水系统中的管道、互联网或任何数量的连接系统。网络分析抽象出这个常见问题，以便相同的技术和算法可以应用于广泛的各类应用。在本菜谱中，我们将使用通用线网络，通过 Dijkstra 算法进行分析，这是用于找到最短路径的最古老算法之一。QGIS 内置了所有这些功能。

## 准备工作

首先，从以下链接下载矢量数据集，其中包含两个 shapefiles，并将其解压缩到`qgis_data`目录下的名为`shapes`的目录中：

[`geospatialpython.googlecode.com/svn/network.zip`](https://geospatialpython.googlecode.com/svn/network.zip)

## 如何操作...

我们将通过定义线网络的起点和终点来创建一个网络图，然后使用这个图来确定两点之间沿线网络的最短路线。为此，我们需要执行以下步骤：

1.  在 QGIS **Python 控制台**中，我们首先导入所需的库，包括 QGIS 网络分析器：

    ```py
    from qgis.core import *
    from qgis.gui import *
    from qgis.networkanalysis import *
    from PyQt4.QtCore import *

    ```

1.  接下来，我们将加载我们的线网络 shapefile 以及包含我们想要网络分析器在选择路线时考虑的网络上的点的 shapefile：

    ```py
    network = QgsVectorLayer("/Users/joellawhead/qgis_data/shapes/\Network.shp", "Network Layer", "ogr")
    waypoints = QgsVectorLayer("/Users/joellawhead/qgis_data/shapes/\ NetworkPoints.shp", "Waypoints", "ogr")

    ```

1.  现在，我们将创建一个图导演来定义图的属性。`director`对象接受我们的线 shapefile、一个用于方向信息的字段 ID 以及一些其他涉及网络方向属性的文档化整数代码。在我们的例子中，我们将告诉导演忽略方向。`properter`对象是一种基本的路由策略算法，它被添加到网络图中，并考虑线长度：

    ```py
    director = QgsLineVectorLayerDirector(network, -1, '', '', '', 3)
    properter = QgsDistanceArcProperter()
    director.addProperter(properter)
    crs = network.crs()

    ```

1.  现在，我们创建`GraphBuilder`对象，将线网络实际转换为图：

    ```py
    builder = QgsGraphBuilder(crs)

    ```

1.  我们定义了我们的路线的起点和终点：

    ```py
    ptStart = QgsPoint(-0.8095638694, -0.1578175511)
    ptStop = QgsPoint(0.8907435677, 0.4430834924)

    ```

1.  然后，我们告诉导演将我们的点层转换为网络中的连接点，这些点定义了网络中的航路点，也可以选择提供阻力值：

    ```py
    tiePoints = director.makeGraph(builder, [ptStart, ptStop])

    ```

1.  现在，我们可以使用以下代码来构建图：

    ```py
    graph = builder.graph()

    ```

1.  我们现在将起点和终点定位为图中的连接点：

    ```py
    tStart = tiePoints[0]
    tStop = tiePoints[1]
    idStart = graph.findVertex(tStart)
    idStop = graph.findVertex(tStop)

    ```

1.  然后，我们可以告诉分析器使用我们的起点来找到网络中的最短路线：

    ```py
    (tree, cost) = QgsGraphAnalyzer.dijkstra(graph, idStart, 0)

    ```

1.  接下来，我们将遍历生成的树并获取输出路线上的点：

    ```py
    p = []
    curPos = idStop
    while curPos != idStart:
    p.append(graph.vertex(graph.arc(tree[curPos]).inVertex()).point())
    curPos = graph.arc(tree[curPos]).outVertex()
    p.append(tStart)

    ```

1.  现在，我们将加载我们的两个输入 shapefiles 到地图上，并创建一个橡皮筋来可视化路线：

    ```py
    QgsMapLayerRegistry.instance().addMapLayers([network,waypoints])
    rb = QgsRubberBand(iface.mapCanvas())
    rb.setColor(Qt.red)

    ```

1.  最后，我们将路线点添加到橡皮筋上，以便查看网络分析器的输出：

    ```py
    for pnt in p:
     rb.addPoint(pnt)

    ```

## 它是如何工作的...

这个食谱是一个非常简单的例子，用作调查一个非常复杂且强大的工具的起点。线网络形状文件可以有一个字段定义每条线在某个方向上为单向或双向。点形状文件提供网络上的航点，以及阻力值，这些值可能代表海拔、交通密度或其他因素，这些因素会使路线不那么理想。输出将类似于以下图像：

![如何工作...](img/00056.jpeg)

更多信息和网络分析工具的示例可以在 QGIS 文档中找到，网址为 [`docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/network_analysis.html`](http://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/network_analysis.html)。

# 沿街道进行路由

有时候，你可能想要找到两个地址之间的最佳驾驶路线。街道导航现在已经变得如此普遍，以至于我们把它视为理所当然。然而，如果你探索这本书中关于地理编码和网络分析的食谱，你将开始看到街道导航实际上是一个多么复杂的挑战。为了在 QGIS 中执行路由操作，我们将使用用 Python 编写的 QGIS GeoSearch 插件，这样我们就可以从控制台访问它。

## 准备工作

为了进行这个练习中的路由，你需要安装 QGIS Python GeoSearch 插件，以及 QGIS OpenLayers 插件来在 Google 地图上叠加结果，如下所示：

1.  从 QGIS **插件**菜单中选择 **管理并安装插件…**。

1.  如果你已经安装了 QGIS GeoCoding 插件，那么你必须卸载它，因为有时它与 GeoSearch 插件冲突。因此，在插件列表中选择此选项，并点击 **卸载插件**按钮。

1.  在 **插件**对话框的搜索框中，搜索 `GeoSearch`。

1.  选择 **GeoSearch 插件**并点击 **安装插件**按钮。

1.  接下来，在 **插件**搜索对话框中，搜索 `OpenLayers`。

1.  选择 **OpenLayers 插件**并点击 **安装插件**按钮。

## 如何操作...

我们将调用 GeoSearch 插件的路线功能，该功能使用 Google 的路线引擎，并通过 OpenLayers 插件在 Google 地图上显示结果。为此，我们需要执行以下步骤：

1.  在 **QGIS Python 控制台**中，我们首先导入 **QGIS utils 库**以及 **GeoSearch** 插件所需的部分：

    ```py
    import qgis.utils
    from GeoSearch import geosearchdialog, GoogleMapsApi

    ```

1.  接下来，我们将使用 QGIS utils 库来访问 **OpenLayers 插件**：

    ```py
    openLyrs = qgis.utils.plugins['openlayers_plugin']

    ```

1.  GeoSearch 插件并不是真正为程序化使用而设计的，因此为了调用此插件，我们必须通过 GUI 界面调用它，但然后我们需要传递空白值，这样它就不会触发 GUI 插件界面：

    ```py
    g = geosearchdialog.GeoSearchDialog(iface)
    g.SearchRoute([])

    ```

1.  现在，使用以下代码，我们可以安全地创建我们的路由引擎对象：

    ```py
    d = GoogleMapsApi.directions.Directions()

    ```

1.  接下来，我们创建我们的起点和终点地址：

    ```py
    origin = "Boston, MA"
    dest = "2517 Main Rd, Dedham, ME 04429"

    ```

1.  然后，我们可以使用最简单的选项来计算路线，如下所示：

    ```py
    route = d.GetDirections(origin, dest, mode = "driving", \ waypoints=None, avoid=None, units="imperial")

    ```

1.  现在，我们使用**OpenLayers 插件**将谷歌地图基础地图添加到 QGIS 地图中：

    ```py
    layerType = openLyrs._olLayerTypeRegistry.getById(4)
    openLyrs.addLayer(layerType)

    ```

1.  最后，我们使用**GeoSearch 插件**在基础地图上创建我们的路线图层：

    ```py
    g.CreateVectorLayerGeoSearch_Route(route)

    ```

## 它是如何工作的...

尽管它们是用 Python 构建的，但 GeoSearch 和 OpenLayers 插件都不是为程序员使用 Python 而设计的。然而，我们仍然可以在脚本中不费太多麻烦地使用这些工具。为了利用 GeoSearch 插件提供的某些路由选项，您可以使用其 GUI 查看可用选项，然后将这些选项添加到您的脚本中。请注意，大多数插件没有真正的 API，因此插件在未来版本中的微小更改可能会破坏您的脚本。

# 跟踪 GPS

QGIS 能够连接到使用 NMEA 标准的 GPS。QGIS 可以通过串行连接到 GPS 或通过 QGIS GPS 信息面板使用名为 gpsd 的开源软件与之通信。GPS 的位置信息可以在 QGIS 地图上显示，QGIS 甚至可以自动平移地图以跟随 GPS 点。在这个菜谱中，我们将使用 QGIS API 处理 NMEA 语句并更新全球地图上的一个点。连接到不同 GPS 设备所需的信息可能会有很大差异，因此我们将使用在线 NMEA 语句生成器来获取一些模拟的 GPS 信息。

## 准备工作

这个菜谱不需要任何准备。

## 如何操作...

我们将从免费的在线生成器中抓取一批 NMEA GPS 语句，使用在线 geojson 数据创建一个全球基础地图，创建一个矢量点图层来表示 GPS，并最终遍历语句，使我们的轨迹点在地图上移动。

要做到这一点，我们需要执行以下步骤：

1.  首先，我们需要使用 QGIS **Python 控制台**导入一些标准 Python 库：

    ```py
    import urllib
    import urllib2
    import time

    ```

1.  接下来，我们将连接到在线 NMEA 生成器，下载一批语句，并将它们转换成列表，如下所示：

    ```py
    url = 'http://freenmea.net/api/emitnmea'
    values = {'types' : 'default'}
    data = urllib.urlencode(values)
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    results = response.read().split("\n")

    ```

1.  接下来，我们可以添加我们的世界国家基础地图，使用 geojson 服务：

    ```py
    wb = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    basemap = QgsVectorLayer(wb, "Countries", "ogr")
    qmr = QgsMapLayerRegistry.instance()
    qmr.addMapLayer(basemap)

    ```

1.  现在，我们可以创建我们的 GPS 点图层并访问其数据提供者：

    ```py
    vectorLyr = QgsVectorLayer('Point?crs=epsg:4326', \'GPS Point' , "memory")
    vpr = vectorLyr.dataProvider()

    ```

1.  然后，我们需要一些变量来存储我们在遍历位置时的当前坐标，我们还将访问`mapCanvas`对象：

    ```py
    cLat = None
    cLon = None
    canvas = iface.mapCanvas()

    ```

1.  接下来，我们将创建一个 GPS 连接对象用于数据处理。如果我们使用的是实时 GPS 对象，我们将使用此行输入设备信息：

    ```py
    c = QgsNMEAConnection(None)

    ```

1.  现在，我们设置一个标志来决定我们是否正在处理第一个点：

    ```py
    firstPt = True

    ```

1.  我们现在可以遍历 NMEA 语句了，但我们必须检查语句类型，以确定我们正在使用哪种类型的信息。在实时 GPS 连接中，QGIS 会自动处理这部分，因此这部分代码将不再必要：

    ```py
    for r in results:
     l = len(r)
     if "GGA" in r:
     c.processGGASentence(r,l)
     elif "RMC" in r:
     c.processRMCSentence(r,l)
     elif "GSV" in r:
     c.processGSVSentence(r,l)
     elif "VTG" in r:
     c.processVTGSentence(r,l)
     elif "GSA" in r:
     c.processGSASentence(r,l)

    ```

1.  然后，我们可以获取当前的 GPS 信息：

    ```py
     i=c.currentGPSInformation()

    ```

1.  现在，在我们尝试更新地图之前，我们将检查这些信息，以确保 GPS 位置确实自上次循环以来发生了变化：

    ```py
     if i.latitude and i.longitude:
     lat = i.latitude
     lon = i.longitude
     if lat==cLat and lon==cLon:
     continue
     cLat = lat
     cLon = lon
     pnt = QgsGeometry.fromPoint(QgsPoint(lon,lat))

    ```

1.  现在我们有一个新点，我们检查这是否是第一个点，如果是，就将整个图层添加到地图中。否则，我们编辑图层并添加一个新功能，如下所示：

    ```py
     if firstPt:
     firstPt = False
     f = QgsFeature()
     f.setGeometry(pnt)
     vpr.addFeatures([f])
     qmr.addMapLayer(vectorLyr)
     else:
     print lon, lat
     vectorLyr.startEditing()
     vectorLyr.changeGeometry(1,pnt)
     vectorLyr.commitChanges()

    ```

1.  最后，我们刷新地图并观察跟踪点跳到新的位置：

    ```py
     vectorLyr.setCacheImage(None)
     vectorLyr.updateExtents()
     vectorLyr.triggerRepaint()
     time.sleep(1)

    ```

## 它是如何工作的...

一个实时 GPS 将在地图上沿线性、增量路径移动。在这个配方中，我们使用了随机生成的点在世界各地跳跃，但概念是相同的。要连接实时 GPS，您需要首先使用 QGIS 的 GPS 信息 GUI 建立连接，或者至少获取正确的连接信息，然后使用 Python 来自动化后续操作。一旦您有了位置信息，您就可以轻松地使用 Python 操作 QGIS 地图。

## 还有更多...

NMEA 标准虽然老旧且广泛使用，但按照现代标准来说，它是一个设计不佳的协议。现在几乎每部智能手机都有 GPS，但它们并不使用 NMEA 协议。然而，几乎每个智能手机平台都有几个应用程序可以将手机的 GPS 输出为 NMEA 语句，这些语句可以被 QGIS 使用。在本章的后面部分，在*收集现场数据*的配方中，我们将演示另一种跟踪手机、GPS 或甚至数字设备的估计位置的方法，这种方法更简单，也更现代。

# 创建地图册

地图册是一个自动生成的文档，也可以称为**地图集**。地图册将数据集分解成更小的、更详细的地图，这些地图基于覆盖层，将大地图缩放到覆盖层中的每个特征，以便制作地图册的一页。覆盖层可能与地图册每页上展示的地图层相同，也可能不同。在这个配方中，我们将创建一个展示世界上所有国家的地图册。

## 准备工作

对于这个配方，您需要从[`geospatialpython.googlecode.com/svn/countries.zip`](https://geospatialpython.googlecode.com/svn/countries.zip)下载世界国家数据集，并将其放入`qgis_data`目录中名为`shapes`的文件夹内。

接下来，您需要安装`PyPDF2`库。在 Linux 或 OS X 上，只需打开控制台并运行以下命令：

```py
sudo easy_install PyPDF2

```

在 Windows 上，从开始菜单打开 OSGEO4W 控制台并运行以下命令：

```py
easy_install PyPDF2

```

最后，在您的`qgis_data`目录中，创建一个名为`atlas`的文件夹来存储地图册的输出。

## 如何做到这一点...

我们将构建一个 QGIS 组合，并将其设置为地图册模式。然后，我们将添加一个组合地图，其中每个国家都将被展示，以及一个概述地图。接下来，我们将运行地图册过程，以生成地图册的每一页作为单独的 PDF 文件。最后，我们将单个 PDF 文件合并成一个 PDF 文件。为此，我们需要执行以下步骤：

1.  首先，导入所有需要的库：

    ```py
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    from qgis.core import *
    import PyPDF2
    import os

    ```

1.  接下来，创建与输出文件相关的变量，包括地图册的名称、覆盖层以及单个 PDF 文件的命名模式：

    ```py
    filenames = []
    mapbook = "/Users/joellawhead/qgis_data/atlas/mapbook.pdf"
    coverage = "/Users/joellawhead/qgis_data/shapes/countries.shp"
    atlasPattern = "/Users/joellawhead/qgis_data/atlas/output_"

    ```

1.  现在，使用以下代码将覆盖层添加到地图中：

    ```py
    vlyr = QgsVectorLayer(coverage, "Countries", "ogr")
    QgsMapLayerRegistry.instance().addMapLayer(vlyr)

    ```

1.  接下来，建立地图渲染器：

    ```py
    mr = QgsMapRenderer()
    mr.setLayerSet([vlyr.id()])
    mr.setProjectionsEnabled(True)
    mr.setMapUnits(QGis.DecimalDegrees)
    crs = QgsCoordinateReferenceSystem()
    crs.createFromSrid(4326)
    mr.setDestinationCrs(crs)

    ```

1.  然后，设置组合：

    ```py
    c = QgsComposition(mr)
    c.setPaperSize(297, 210)

    ```

1.  为覆盖层创建一个符号：

    ```py
    gray = {"color": "155,155,155"}
    mapSym = QgsFillSymbolV2.createSimple(gray)
    renderer = QgsSingleSymbolRendererV2(mapSym)
    vlyr.setRendererV2(renderer)

    ```

1.  现在，将第一个作曲地图添加到组合中，如下所示：

    ```py
    atlasMap = QgsComposerMap(c, 20, 20, 130, 130)
    atlasMap.setFrameEnabled(True)
    c.addComposerMap(atlasMap)

    ```

1.  然后，创建图集框架：

    ```py
    atlas = c.atlasComposition()
    atlas.setCoverageLayer(vlyr)
    atlas.setHideCoverage(False)
    atlas.setEnabled(True)
    c.setAtlasMode(QgsComposition.ExportAtlas)

    ```

1.  接下来，建立概述地图：

    ```py
    ov = QgsComposerMap(c, 180, 20, 50, 50)
    ov.setFrameEnabled(True)
    ov.setOverviewFrameMap(atlasMap.id())
    c.addComposerMap(ov)
    rect = QgsRectangle(vlyr.extent())
    ov.setNewExtent(rect)

    ```

1.  然后，创建概述地图符号：

    ```py
    yellow = {"color": "255,255,0,255"}
    ovSym = QgsFillSymbolV2.createSimple(yellow)
    ov.setOverviewFrameMapSymbol(ovSym)

    ```

1.  接下来，你需要用国家的名称标记每一页，这个名称存储在 shapefile 的`CNTRY_NAME`字段中：

    ```py
    lbl = QgsComposerLabel(c)
    c.addComposerLabel(lbl)
    lbl.setText('[% "CNTRY_NAME" %]')
    lbl.setFont(QgsFontUtils.getStandardTestFont())
    lbl.adjustSizeToText()
    lbl.setSceneRect(QRectF(150, 5, 60, 15))

    ```

1.  现在，我们将告诉图集为每个国家使用自动缩放，以便最佳地适应每个地图窗口：

    ```py
    atlasMap.setAtlasDriven(True)
    atlasMap.setAtlasScalingMode(QgsComposerMap.Auto)
    atlasMap.setAtlasMargin(0.10)

    ```

1.  现在，我们告诉图集遍历所有特征并创建 PDF 地图，如下所示：

    ```py
    atlas.setFilenamePattern("'%s' || $feature" % atlasPattern)
    atlas.beginRender()
    for i in range(0, atlas.numFeatures()):
     atlas.prepareForFeature(i)
     filename = atlas.currentFilename() + ".pdf"
     print "Writing file %s" % filename
     filenames.append(filename)
     c.exportAsPDF(filename)
    atlas.endRender()

    ```

1.  最后，我们将使用 PyPDF2 库将单个 PDF 文件合并成一个 PDF 文件，如下所示：

    ```py
    output = PyPDF2.PdfFileWriter()
    for f in filenames:
     pdf = open(f, "rb")
     page = PyPDF2.PdfFileReader(pdf)
     output.addPage(page.getPage(0))
     os.remove(f)
    print "Writing final mapbook..."
    book = open(mapbook, "wb")
    output.write(book)
    with open(mapbook, 'wb') as book:
     output.write(book)

    ```

## 它是如何工作的...

你可以根据需要自定义创建单个页面的模板。GUI 图集工具可以将图集导出为单个文件，但这个功能在 PyQIS 中不可用，所以我们使用纯 Python 的 PyPDF2 库。你还可以在 GUI 中创建模板，保存它，然后用 Python 加载它，但如果你在代码中已经有了布局，通常更容易进行更改。你还应该知道，PDF 页面只是图像。地图以栅格形式导出，因此图集将不可搜索，文件大小可能很大。

# 寻找最低成本路径

**最低成本路径**（**LCP**）分析是网络分析的栅格等效，用于在栅格中找到两点之间的最优路径。在这个菜谱中，我们将对数字高程模型（DEM）执行 LCP 分析。

## 准备工作

你需要下载以下 DEM 并将 ZIP 文件解压到你的`qgis_data/rasters`目录：[`geospatialpython.googlecode.com/svn/lcp.zip`](https://geospatialpython.googlecode.com/svn/lcp.zip)

## 如何做到这一点...

我们将加载我们的 DEM 和包含起点和终点的两个 shapefile。然后，我们将通过处理工具箱使用 GRASS 来创建累积成本层，该层根据栅格中每个单元格的高度、周围其他单元格的值以及其到和从终点的距离为每个栅格单元格分配一个成本。

然后，我们将使用 SAGA 处理算法在两点之间找到最低成本路径。最后，我们将输出加载到地图上。为此，我们需要执行以下步骤：

1.  首先，我们将导入 QGIS 处理 Python 库：

    ```py
    import processing

    ```

1.  现在，我们将设置层的路径，如下所示：

    ```py
    path = "/Users/joellawhead/qgis_data/rasters"/"
    dem = path + "dem.asc"
    start = path + "start-point.shp"
    finish = path + "end-point.shp"

    ```

1.  我们需要将 DEM 的范围作为字符串用于算法：

    ```py
    demLyr = QgsRasterLayer(dem, "DEM")
    ext = demLyr.extent()
    xmin = ext.xMinimum()
    ymin = ext.yMinimum()
    xmax = ext.xMaximum()
    ymax = ext.xMaximum()
    box = "%s,%s,%s,%s" % (xmin,xmax,ymin,ymax)

    ```

1.  使用以下代码，我们将端点作为层来建立：

    ```py
    a = QgsVectorLayer(start, "Start", "ogr")
    b = QgsVectorLayer(finish, "End", "ogr")

    ```

1.  然后，我们将创建累积成本栅格，指定算法名称、成本层（DEM）、起点层、终点层、速度或精度选项、保留空值选项、感兴趣的范围、单元格大小（默认为 0）和一些其他默认值：

    ```py
    tmpCost = processing.runalg("grass:r.cost",dem,a,b,\
    False,False,box,0,-1,0.0001,None)
    cost = tmpCost["output"]

    ```

1.  我们还需要将点合并成一个单独的层，以便 SAGA 算法使用：

    ```py
    tmpMerge = processing.runalg("saga:mergeshapeslayers",\start,finish,None)
    merge = tmpMerge["OUT"]

    ```

1.  接下来，我们为 LCP 算法设置输入和输出：

    ```py
    vLyr = QgsVectorLayer(merge, "Destination Points", "ogr")
    rLyr = QgsRasterLayer(cost, "Accumulated Cost")
    line = path + "path.shp"

    ```

1.  然后，我们使用以下代码运行 LCP 分析：

    ```py
    results = processing.runalg("saga:leastcostpaths",\lyr,rLyr,demLyr,None,line)

    ```

1.  最后，我们可以加载路径来查看它：

    ```py
    path = QgsVectorLayer(line, "Least Cost Path", "ogr")
    QgsMapLayerRegistry.instance().addMapLayers([demLyr, \ vLyr, path])

    ```

## 它是如何工作的...

GRASS 也有 LCP 算法，但 SAGA 算法更容易使用。GRASS 在创建成本网格方面做得很好。处理工具箱算法允许您创建在 QGIS 关闭时删除的临时文件。因此，我们使用临时文件来处理中间产品，包括成本网格和合并后的形状文件。

# 执行邻近邻分析

邻近邻分析将一个点与一个或多个数据集中的最近点相关联。在这个菜谱中，我们将一组点与另一个数据集中的最近点相关联。在这种情况下，我们将找到国家不明飞行物（UFO）报告中心目录中每个不明飞行物目击事件的最近大城市。这项分析将告诉您哪些大城市有最多的不明飞行物活动。不明飞行物目录数据仅包含经纬度点，因此我们将使用邻近邻分析来为地点分配名称。

## 准备工作

下载以下 ZIP 文件并将其解压到 `qgis_data` 目录下的名为 `ufo` 的目录中：

[`geospatialpython.googlecode.com/svn/ufo.zip`](https://geospatialpython.googlecode.com/svn/ufo.zip)

您还需要 MMQGIS 插件：

1.  从 QGIS **插件** 菜单中选择 **管理并安装插件…**

1.  在 **插件** 对话框的搜索框中，搜索 `mmqgis`。

1.  选择 **MMQGIS 插件** 并点击 **安装插件** 按钮。

## 如何操作...

这个菜谱很简单。在这里，我们将在 MMQGIS 插件中加载图层并运行邻近邻算法，如下所示：

1.  首先，我们将导入 MMQGIS 插件：

    ```py
    from mmqgis import mmqgis_library as mmqgis

    ```

1.  接下来，如图所示，我们将加载所有数据集：

    ```py
    srcPath = "/qgis_data/ufo/ufo-sightings.shp"
    dstPath = "/qgis_data/ufo/major-cities.shp"
    usPth = "/qgis_data/ufo/continental-us.shp"
    output = "/qgis_data/ufo/alien_invasion.shp"
    srcName = "UFO Sightings"
    dstName = "Major Cities"
    usName = "Continental US"
    source = QgsVector(srcPath, srcName, "ogr")
    dest = QgsVector(dstPath, dstName, "ogr")
    us = QgsVector(usPath, usName, "ogr")

    ```

1.  最后，我们将运行并加载算法，该算法将从每个不明飞行物目击点绘制到最近城市的线条：

    ```py
    mmqgis.mmqgis_hub_distance(iface, srcName, dstName, \"NAME", "Miles", True, output, True)

    ```

## 它是如何工作的...

QGIS 中有几个不同的邻近邻算法，但 MMQGIS 版本是一个优秀的实现，并且具有最好的可视化。与其他章节中的其他菜谱一样，插件没有故意设计的 Python API，因此探索其功能的一个好方法是使用 GUI 界面，然后再查看 Python 代码。以下图像显示了输出，其中不明飞行物目击点由较小的点表示，而连接到城市的枢纽线则由较大、较暗的点表示。

![它是如何工作的...](img/00057.jpeg)

# 创建热力图

**热力图**用于使用显示密度的栅格图像来显示数据的地理聚类。聚类也可以使用数据中的一个字段进行加权，不仅显示地理密度，还可以显示强度因子。在这个菜谱中，我们将使用地震点数据来创建地震影响的热力图，并通过地震的震级来加权聚类。

## 准备工作

这个菜谱不需要准备。

## 如何操作...

我们将使用 GeoJSON 格式创建一个包含全球国家边界和地震位置的地图，接下来，我们将运行 SAGA 的核密度估计算法来生成热力图图像。我们将从输出中创建一个图层，为其添加颜色着色器，并将其添加到地图中。

要执行此操作，我们需要执行以下步骤：

1.  首先，我们将导入 Python 控制台中需要的 Python 库：

    ```py
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    import processing

    ```

1.  接下来，使用以下代码，我们将定义我们的地图图层和输出栅格名称：

    ```py
    countries = "https://raw.githubusercontent.com/johan/\world.geo.json/master/countries.geo.json"
    quakes = "https://geospatialpython.googlecode.com/\svn/quakes2014.geojson"
    output = "/Users/joellawhead/qgis_data/rasters/heat.tif"

    ```

1.  现在我们将图层添加到地图中：

    ```py
    basemap = QgsVectorLayer(countries, "World", "ogr")
    quakeLyr = QgsVectorLayer(quakes, "Earthquakes", "ogr")
    QgsMapLayerRegistry.instance().addMapLayers([quakeLyr, basemap])

    ```

1.  我们需要获取地震图层的范围，以便 Processing 工具箱算法可以使用：

    ```py
    ext = quakeLyr.extent()
    xmin = ext.xMinimum()
    ymin = ext.yMinimum()
    xmax = ext.xMaximum()
    ymax = ext.xMaximum()
    box = "%s,%s,%s,%s" % (xmin,xmax,ymin,ymax)

    ```

1.  现在，我们可以通过指定 `mag` 或震级字段作为权重因子来运行核密度估计算法：

    ```py
    processing.runalg("saga:kerneldensityestimation",quakeLyr,"mag",10,0,0,box,1,output)

    ```

1.  接下来，我们将输出加载为图层：

    ```py
    heat = QgsRasterLayer(output, "Earthquake Heatmap")

    ```

1.  然后，我们创建颜色渐变着色器并将其应用于图层：

    ```py
    algorithm = QgsContrastEnhancement.StretchToMinimumMaximum
    limits = QgsRaster.ContrastEnhancementMinMax
    heat.setContrastEnhancement(algorithm, limits)
    s = QgsRasterShader() 
    c = QgsColorRampShader() 
    c.setColorRampType(QgsColorRampShader.INTERPOLATED) 
    i = [] 
    qri = QgsColorRampShader.ColorRampItem
    i.append(qri(0, QColor(255,255,178,255), \
    'Lowest Earthquake Impact')) 
    i.append(qri(0.106023, QColor(254,204,92,255), \
    'Lower Earthquake Impact')) 
    i.append(qri(0.212045, QColor(253,141,60,255), \
    'Moderate Earthquake Impact')) 
    i.append(qri(0.318068, QColor(240,59,32,255), \
    'Higher Earthquake Impact'))
    i.append(qri(0.42409, QColor(189,0,38,255), \
    'Highest Earthquake Impact')) 
    c.setColorRampItemList(i) 
    s.setRasterShaderFunction(c) 
    ps = QgsSingleBandPseudoColorRenderer(heat.dataProvider(),\ 1,  s) 
    heat.setRenderer(ps) 

    ```

1.  最后，我们将 `Heatmap` 添加到我们的地图中：

    ```py
    QgsMapLayerRegistry.instance().addMapLayers([heat])

    ```

## 如何工作...

核密度估计算法会查看点数据集并形成簇。值越高，簇的密度就越大。然后，算法根据权重因子（即地震的震级）增加值。输出图像当然是灰度 GeoTIFF，但我们使用颜色渐变着色器来使可视化更容易理解。以下截图显示了预期的输出：

![如何工作...](img/00058.jpeg)

## 更多内容...

QGIS 提供了一个功能强大的插件，名为热力图，它能够自动在各种数据上良好地工作。然而，它是用 C++ 编写的，并且没有 Python API。

# 创建点密度图

点密度图使用点密度来表示多边形内的字段值。我们将使用这项技术来表示美国人口普查局的一些区域的人口密度。

## 准备工作

您需要从 [`geospatialpython.googlecode.com/files/GIS_CensusTract.zip`](https://geospatialpython.googlecode.com/files/GIS_CensusTract.zip) 下载人口区域图层并将其提取到 `qgis_data` 目录下的 `census` 目录中。

## 如何操作...

我们将加载人口图层，创建一个内存图层，遍历人口图层的要素，为每 100 人计算一个特征内的随机点，并将点添加到内存图层中。为此，我们需要执行以下步骤：

1.  在 QGIS 的 **Python 控制台** 中，我们将导入 `random` 模块：

    ```py
    import random

    ```

1.  接下来，我们将加载人口图层：

    ```py
    src = "/Users/joellawhead/qgis_data/census/\
    GIS_CensusTract_poly.shp"
    tractLyr = QgsVectorLayer(src, "Census Tracts", "ogr")

    ```

1.  然后，我们将创建我们的内存图层：

    ```py
    popLyr =  QgsVectorLayer('Point?crs=epsg:4326', "Population" , "memory")

    ```

1.  我们需要人口值的索引：

    ```py
    i = tractLyr.fieldNameIndex('POPULAT11')

    ```

1.  现在，我们以迭代器的方式获取人口图层的特点：

    ```py
    features = tractLyr.getFeatures()

    ```

1.  我们需要一个数据提供者来为内存图层提供支持，以便我们可以编辑它：

    ```py
    vpr = popLyr.dataProvider()

    ```

1.  我们将创建一个列表来存储我们的随机点：

    ```py
    dotFeatures = []

    ```

1.  然后，我们可以遍历要素并计算密度点：

    ```py
    for feature in features:
     pop = feature.attributes()[i]
     density = pop / 100
     found = 0
     dots = []
     g = feature.geometry()
     minx =  g.boundingBox().xMinimum()
     miny =  g.boundingBox().yMinimum()
     maxx =  g.boundingBox().xMaximum()
     maxy =  g.boundingBox().yMaximum()
     while found < density:
     x = random.uniform(minx,maxx)
     y = random.uniform(miny,maxy)
     pnt = QgsPoint(x,y)
     if g.contains(pnt):
     dots.append(pnt)
     found += 1
     geom = QgsGeometry.fromMultiPoint(dots)
     f = QgsFeature()
     f.setGeometry(geom)
     dotFeatures.append(f)

    ```

1.  现在，我们可以使用以下代码将我们的要素添加到内存图层中，并将它们添加到地图中以查看结果：

    ```py
    vpr.addFeatures(dotFeatures)
    popLyr.updateExtents()
    QgsMapLayerRegistry.instance().addMapLayers(\ [popLyr,tractLyr])

    ```

## 如何工作...

这种方法稍微有些低效；它使用了一种暴力方法，可能会将随机生成的点放置在不规则的多边形之外。我们使用要素的范围来尽可能包含随机点，然后使用几何包含方法来验证点是否在多边形内部。以下截图显示了输出样本：

![工作原理...](img/00059.jpeg)

# 收集现场数据

几十年来，从现场收集观测数据并将其输入 GIS 系统需要花费数小时的手动录入，或者在最好情况下，在旅行结束后加载数据。智能手机和具有蜂窝连接的笔记本电脑彻底改变了这一过程。在本教程中，我们将使用一个简单但有趣的基于 geojson 的框架，通过任何具有网络浏览器的互联网连接设备输入信息和地图位置，并在 QGIS 中更新地图。

## 准备工作

本教程不需要任何准备工作。

## 如何操作...

我们将在 QGIS 地图上加载世界边界图层和现场数据图层，访问现场数据移动网站创建一个条目，然后刷新 QGIS 地图以查看更新。为此，我们需要执行以下步骤：

1.  在 QGIS 的**Python 控制台**中，添加以下 geojson 图层：

    ```py
    wb = "https://raw.githubusercontent.com/johan/\ world.geo.json/master/countries.geo.json"
    basemap = QgsVectorLayer(wb, "Countries", "ogr")
    observations = \
    QgsVectorLayer("http://bit.ly/QGISFieldApp", \
    "Field Observations", "ogr")
    QgsMapLayerRegistry.instance().addMapLayers(\ [basemap, observations])

    ```

1.  现在，在您的计算机上的浏览器中，或者在具有数据连接的移动设备上，访问[`geospatialpython.github.io/qgis/fieldwork.html`](http://geospatialpython.github.io/qgis/fieldwork.html)。应用程序将请求您允许使用您的位置，您应该暂时允许程序运行。

1.  在表单中输入信息，然后点击**发送**按钮。

1.  验证您是否可以在[`api.myjson.com/bins/3ztvz`](https://api.myjson.com/bins/3ztvz)看到 geojson 数据，包括您的提交。

1.  最后，通过缩放或平移更新 QGIS 中的地图，并定位您的记录。

## 工作原理...

```py
 code for the mobile page on GitHub.com (https://github.com/GeospatialPython/qgis).
```

以下图像显示了 iPhone 上的移动现场应用程序：

![工作原理...](img/00060.jpeg)

此图像显示了在 QGIS 中对应数据的显示方式：

![工作原理...](img/00061.jpeg)

# 使用高程数据计算道路坡度

常见的地理空间工作流程是将栅格值分配给相应的矢量图层，以便可以对矢量图层进行样式设置或进行进一步分析。本教程将使用这一概念来展示如何使用坡度栅格将值映射到道路矢量图层，以表示道路的陡峭程度。

## 准备工作

您需要从[`geospatialpython.googlecode.com/svn/road.zip`](https://geospatialpython.googlecode.com/svn/road.zip)下载一个压缩目录，并将名为`road`的目录放置在您的`qgis_data`目录中。

## 如何操作...

我们将从 DEM 开始，计算其坡度。然后，我们将加载一个道路矢量层，并将其分割成 500 米的区间长度。接下来，我们将加载该层，并使用每个段落的绿色、黄色和红色值对其进行样式化，以显示坡度的范围。我们将在 DEM 的阴影图上叠加这个层，以获得良好的可视化效果。为此，我们需要执行以下步骤：

1.  首先，我们需要在 QGIS **Python 控制台**中导入 QGIS 处理模块、QGIS 常量模块、Qt GUI 模块和 os 模块：

    ```py
    from qgis.core import *
    from PyQt4.QtGui import *
    import processing

    ```

1.  现在，我们需要将我们项目的坐标参考系统（CRS）设置为我们的数字高程模型（DEM）的 CRS，即 EPSG 代码 26910，这样我们就可以以米为单位而不是十进制度数来处理数据：

    ```py
    myCrs = QgsCoordinateReferenceSystem(26910, QgsCoordinateReferenceSystem.EpsgCrsId)
    iface.mapCanvas().mapRenderer().setDestinationCrs(myCrs)
    iface.mapCanvas().setMapUnits(QGis.Meters)
    iface.mapCanvas().refresh()

    ```

1.  现在，我们将设置所有层的路径。为此，我们将使用我们创建的中间层，这样如果需要，我们可以在一个地方更改它们：

    ```py
    src_dir = "/Users/joellawhead/qgis_data/road/" 
    dem = os.path.join(src_dir, "dem.asc")
    road = os.path.join(src_dir, "road.shp")
    slope = os.path.join(src_dir, "slope.tif")
    segRoad = os.path.join(src_dir, "segRoad.shp")
    steepness = os.path.join(src_dir, "steepness.shp")
    hillshade = os.path.join(src_dir, "hillshade.tif") 

    ```

1.  我们将加载 DEM 和道路层，以便我们可以获取处理算法的范围：

    ```py
    demLyr = QgsRasterLayer(dem, "DEM")
    roadLyr = QgsVectorLayer(road, "Road", "ogr")

    ```

1.  现在，使用以下代码构建一个包含 DEM 范围的字符串：

    ```py
    ext = demLyr.extent()
    xmin = ext.xMinimum()
    ymin = ext.yMinimum()
    xmax = ext.xMaximum()
    ymax = ext.yMaximum()
    demBox = "%s,%s,%s,%s" % (xmin,xmax,ymin,ymax)

    ```

1.  接下来，计算坡度网格：

    ```py
    processing.runalg("grass:r.slope",dem,0,0,1,0,True,\ demBox,0,slope)

    ```

1.  然后，我们可以将道路层的范围作为一个字符串获取：

    ```py
    ext = roadLyr.extent()
    xmin = ext.xMinimum()
    ymin = ext.yMinimum()
    xmax = ext.xMaximum()
    ymax = ext.yMaximum()
    roadBox = "%s,%s,%s,%s" % (xmin,xmax,ymin,ymax)

    ```

1.  现在，我们将道路层分割成 500 米的段落，以便为坡度评估提供一个有意义的长度：

    ```py
    processing.runalg("grass:v.split.length",road,500,\
    roadBox,-1,0.0001,0,segRoad)

    ```

1.  接下来，我们将坡度和分割层添加到地图界面中，以便进行下一个算法，但我们将使用`addMapLayers`方法中的布尔`False`选项将它们隐藏起来：

    ```py
    slopeLyr = QgsRasterLayer(slope, "Slope")
    segRoadLyr = QgsVectorLayer(segRoad, \
    "Segmented Road", "ogr")
    QgsMapLayerRegistry
    .instance().addMapLayers([\ segRoadLyr,slopeLyr], False)

    ```

1.  现在，我们可以将坡度值传输到分割道路层，以创建坡度层：

    ```py
    processing.runalg("saga:addgridvaluestoshapes",\ segRoad,slope,0,steepness)

    ```

1.  现在，我们可以加载坡度层：

    ```py
    steepLyr = QgsVectorLayer(steepness, \ "Road Gradient", "ogr")

    ```

1.  我们将样式化坡度层，使用交通信号灯的红色、黄色和绿色值，其中红色表示最陡：

    ```py
    roadGrade = ( ("Rolling Hill", 0.0, 20.0, "green"), 
    ("Steep", 20.0, 40.0, "yellow"),
    ("Very Steep", 40.0, 90.0, "red"))
    ranges = []
    for label, lower, upper, color in roadGrade:
     sym = QgsSymbolV2.defaultSymbol(steepLyr.geometryType())
     sym.setColor(QColor(color))
     sym.setWidth(3.0)
     rng = QgsRendererRangeV2(lower, upper, sym, label)
     ranges.append(rng)

    field = "slope"
    renderer = QgsGraduatedSymbolRendererV2(field, ranges)
    steepLyr.setRendererV2(renderer)

    ```

1.  接下来，我们将从 DEM 创建阴影图以进行可视化，并将所有内容加载到地图上：

    ```py
    processing.runalg("saga:analyticalhillshading",dem,\
    0,315,45,4,hillshade)
    hs = QgsRasterLayer(hillshade, "Terrain")
    QgsMapLayerRegistry.instance().addMapLayers([steepLyr, hs])

    ```

## 它是如何工作的...

对于我们每个 500 米的线段，算法会平均其下方的坡度值。这个工作流程相当简单，同时也为你提供了一个更复杂版本所需的所有构建块。在执行涉及相对较小区域测量的计算时，使用投影数据是最佳选择。以下图像显示了输出效果：

![它是如何工作的...](img/00062.jpeg)

# 在地图上定位照片

配备 GPS 的相机拍摄的照片，包括智能手机，在文件的头部存储位置信息，这种格式称为 EXIF 标签。这些标签在很大程度上基于 TIFF 图像标准使用的相同头部标签。在这个菜谱中，我们将使用这些标签为一些照片创建地图上的位置，并提供链接以打开它们。

## 准备工作

您需要从[`github.com/GeospatialPython/qgis/blob/gh-pages/photos.zip?raw=true`](https://github.com/GeospatialPython/qgis/blob/gh-pages/photos.zip?raw=true)下载一些带有地理标签的样本照片，并将它们放置在`qgis_data`目录中名为`photos`的目录中。

## 如何操作...

QGIS 需要**Python 图像库**（**PIL**），它应该已经包含在您的安装中。PIL 可以解析 EXIF 标签。我们将收集照片的文件名，解析位置信息，将其转换为十进制度，创建点矢量图层，添加照片位置，并将操作链接添加到属性中。为此，我们需要执行以下步骤：

1.  在 QGIS **Python 控制台**中，导入我们将需要的库，包括用于解析图像数据的 k 库和用于执行通配符文件搜索的`glob`模块：

    ```py
    import glob
    import Image
    from ExifTags import TAGS

    ```

1.  接下来，我们将创建一个可以解析标题数据的函数：

    ```py
    def exif(img):
     exif_data = {}
     try: 
     i = Image.open(img)
     tags = i._getexif()
     for tag, value in tags.items():
     decoded = TAGS.get(tag, tag)
     exif_data[decoded] = value
     except:
     pass
     return exif_data

    ```

1.  现在，我们将创建一个可以将度-分-秒转换为十进制度数的函数，这是 JPEG 图像中存储坐标的方式：

    ```py
    def dms2dd(d, m, s, i):
     sec = float((m * 60) + s)
     dec = float(sec / 3600)
     deg = float(d + dec)
     if i.upper() == 'W':
     deg = deg * -1
     elif i.upper() == 'S':
     deg = deg * -1
     return float(deg)

    ```

1.  接下来，我们将定义一个函数来解析标题数据中的位置数据：

    ```py
    def gps(exif):
     lat = None
     lon = None
     if exif['GPSInfo']: 
     # Lat
     coords = exif['GPSInfo']
     i = coords[1]
     d = coords[2][0][0]
     m = coords[2][1][0]
     s = coords[2][2][0]
     lat = dms2dd(d, m ,s, i)
     # Lon
     i = coords[3]
     d = coords[4][0][0]
     m = coords[4][1][0]
     s = coords[4][2][0]
     lon = dms2dd(d, m ,s, i)
     return lat, lon

    ```

1.  接下来，我们将遍历`photos`目录，获取文件名，解析位置信息，并构建一个简单的字典来存储信息，如下所示：

    ```py
    photos = {}
    photo_dir = "/Users/joellawhead/qgis_data/photos/"
    files = glob.glob(photo_dir + "*.jpg")
    for f in files:
     e = exif(f)
     lat, lon = gps(e)
     photos[f] = [lon, lat]

    ```

1.  现在，我们将设置用于编辑的矢量图层：

    ```py
    lyr_info = "Point?crs=epsg:4326&field=photo:string(75)" 
    vectorLyr =  QgsVectorLayer(lyr_info, \"Geotagged Photos" , "memory")
    vpr = vectorLyr.dataProvider()

    ```

1.  我们将向矢量图层添加照片详细信息：

    ```py
    features = []
    for pth, p in photos.items():
     lon, lat = p
     pnt = QgsGeometry.fromPoint(QgsPoint(lon,lat))
     f = QgsFeature()
     f.setGeometry(pnt)
     f.setAttributes([pth])
     features.append(f)
    vpr.addFeatures(features)
    vectorLyr.updateExtents()

    ```

1.  现在，我们可以将图层添加到地图并将它设置为活动图层：

    ```py
    QgsMapLayerRegistry.instance().addMapLayer(vectorLyr)
    iface.setActiveLayer(vectorLyr)
    activeLyr = iface.activeLayer()

    ```

1.  最后，我们将添加一个允许您点击并打开照片的操作：

    ```py
    actions = activeLyr.actions() 
    actions.addAction(QgsAction.OpenUrl, "Photos", \'[% "photo" %]')

    ```

## 如何工作...

使用包含的 PIL EXIF 解析器，获取位置信息并将其添加到矢量图层相对简单。此菜谱的有趣部分是 QGIS 打开照片的操作。此操作是打开 URL 的默认选项。但是，您也可以使用 Python 表达式作为操作来执行各种任务。以下截图显示了数据可视化和照片弹出窗口的示例：

![如何工作...](img/00063.jpeg)

## 还有更多...

另一个名为 Photo2Shape 的插件可用，但它需要您安装外部 EXIF 标签解析器。

# 图像变化检测

变化检测允许您在两个图像正确正射校正的同一区域自动突出显示它们之间的差异。在此菜谱中，我们将对两个相隔数年的图像进行简单的差异变化检测，以查看城市发展和自然环境的变化。

## 准备工作

您可以从[`github.com/GeospatialPython/qgis/blob/gh-pages/change-detection.zip?raw=true`](https://github.com/GeospatialPython/qgis/blob/gh-pages/change-detection.zip?raw=true)下载此菜谱的两个图像，并将它们放入`qgis_data`目录下的`rasters`目录中名为`change-detection`的目录中。请注意，该文件大小为 55 兆字节，因此下载可能需要几分钟。

## 如何操作...

我们将使用 QGIS 栅格计算器来减去图像以获取差异，这将突出显示显著的变化。我们还将向输出添加颜色渐变着色器以可视化变化。为此，我们需要执行以下步骤：

1.  首先，我们需要将所需的库导入到 QGIS 控制台：

    ```py
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    from qgis.analysis import *

    ```

1.  现在，我们将为我们的图像设置路径名和光栅名：

    ```py
    before = "/Users/joellawhead/qgis_data/rasters/change-detection/before.tif"
    after = "/Users/joellawhead/qgis_data/rasters/change-detection/after.tif"
    beforeName = "Before"
    afterName = "After"

    ```

1.  接下来，我们将图像设置为光栅图层：

    ```py
    beforeRaster = QgsRasterLayer(before, beforeName)
    afterRaster = QgsRasterLayer(after, afterName)

    ```

1.  然后，我们可以构建计算器条目：

    ```py
    beforeEntry = QgsRasterCalculatorEntry()
    afterEntry = QgsRasterCalculatorEntry()
    beforeEntry.raster = beforeRaster
    afterEntry.raster = afterRaster
    beforeEntry.bandNumber = 1
    afterEntry.bandNumber = 2
    beforeEntry.ref = beforeName + "@1"
    afterEntry.ref = afterName + "@2"
    entries = [afterEntry, beforeEntry]

    ```

1.  现在，我们将设置一个简单的表达式，用于进行遥感计算：

    ```py
    exp = "%s - %s" % (afterEntry.ref, beforeEntry.ref)

    ```

1.  然后，我们可以设置输出文件路径、光栅范围以及像素宽度和高度：

    ```py
    output = "/Users/joellawhead/qgis_data/rasters/change-detection/change.tif"
    e = beforeRaster.extent()
    w = beforeRaster.width()
    h = beforeRaster.height()

    ```

1.  现在，我们进行计算：

    ```py
    change = QgsRasterCalculator(exp, output, "GTiff", e, w, h, entries)
    change.processCalculation()

    ```

1.  最后，我们将输出加载为图层，创建颜色渐变着色器，将其应用于图层，并将其添加到地图中，如图所示：

    ```py
    lyr = QgsRasterLayer(output, "Change")
    algorithm = QgsContrastEnhancement.StretchToMinimumMaximum
    limits = QgsRaster.ContrastEnhancementMinMax
    lyr.setContrastEnhancement(algorithm, limits)
    s = QgsRasterShader() 
    c = QgsColorRampShader() 
    c.setColorRampType(QgsColorRampShader.INTERPOLATED) 
    i = [] 
    qri = QgsColorRampShader.ColorRampItem
    i.append(qri(0, QColor(0,0,0,0), 'NODATA')) 
    i.append(qri(-101, QColor(123,50,148,255), 'Significant Itensity Decrease')) 
    i.append(qri(-42.2395, QColor(194,165,207,255), 'Minor Itensity Decrease')) 
    i.append(qri(16.649, QColor(247,247,247,0), 'No Change'))
    i.append(qri(75.5375, QColor(166,219,160,255), 'Minor Itensity Increase')) 
    i.append(qri(135, QColor(0,136,55,255), 'Significant Itensity Increase'))
    c.setColorRampItemList(i) 
    s.setRasterShaderFunction(c) 
    ps = QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), 1,  s) 
    lyr.setRenderer(ps) 
    QgsMapLayerRegistry.instance().addMapLayer(lyr)

    ```

## 它是如何工作的...

概念很简单。我们从新图像数据中减去旧图像数据。专注于城市区域往往具有较高的反射性，导致图像像素值较高。如果在新图像中添加了一座建筑，它将比周围环境更亮。如果移除了一座建筑，新图像在该区域将变暗。这在一定程度上也适用于植被。
