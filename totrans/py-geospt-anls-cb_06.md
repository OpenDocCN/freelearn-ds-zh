# 第六章. 叠加分析

在本章中，我们将涵盖以下主题：

+   使用对称差操作在多边形中打孔

+   不合并的多边形联合

+   使用合并（溶解）的多边形联合

+   执行恒等函数（差集 + 交集）

# 简介

发现当两个数据集叠加在一起时它们在空间上的相互关系被称为叠加分析。叠加可以比作一张描图纸。例如，你可以在你的底图上叠加描图纸，看看哪些区域重叠。这个过程在过去和现在都是空间分析和建模的一个变革。因此，计算机辅助 GIS 计算可以自动识别两个几何集在空间上接触的位置。

本章的目标是让你对最常见的叠加分析函数有一个感觉，例如联合、交集和对称差。这些基于**维度扩展的九交模型**（**DE-9IM**），可以在[`en.wikipedia.org/wiki/DE-9IM`](http://en.wikipedia.org/wiki/DE-9IM)找到，并描述了我们的可能叠加列表。我们在这里使用或命名的所有过程都是使用这九个谓词的组合推导出来的。

![简介](img/50790OS_06_01.jpg)

我们将在第九章中深入探讨这些拓扑规则，*拓扑检查和数据验证*。

# 使用对称差操作在多边形中打孔

为什么，哦，为什么我们要在多边形中打孔并创建一个甜甜圈？嗯，这是出于几个原因，例如，你可能想从与森林多边形重叠的湖泊多边形中移除，因为它位于森林中间，因此包含在你的面积计算中。

另一个例子是我们有一组代表高尔夫球道发球区的多边形，以及另一组代表与这些球道重叠的绿色区域的绿色多边形。我们的任务是计算正确的球道平方米数。绿色区域将在球道多边形中形成我们的甜甜圈。

这被翻译成空间操作术语，意味着我们需要执行一个`对称差`操作，或者在 ESRI 术语中，一个“擦除”操作。

![使用对称差操作在多边形中打孔](img/50790OS_06_02.jpg)

## 准备工作

在这个例子中，我们将创建两组可视化来查看我们的结果。我们的输出将生成**已知文本**（**WKT**），它使用**Openlayers 3**网络地图客户端在你的浏览器中显示。

对于这个例子，请确保你已经下载了所有代码到 GitHub 提供的`/ch06`文件夹，并且这个文件夹结构包含以下文件：

```py
code
¦   ch06-01_sym_diff.py
¦   foldertree.txt
¦   utils.py
¦
+---ol3
    +---build
    ¦       ol-debug.js
    ¦       ol-deps.js
    ¦       ol.js
    ¦
    +---css
    ¦       layout.css
    ¦       ol.css
    ¦
    +---data
    ¦       my_polys.js
    ¦
    +---html
    ¦       ch06-01_sym_diff.html
    ¦
    +---js
    ¦       map_sym_diff.js
    ¦
    +---resources
        ¦   jquery.min.js
        ¦   logo-32x32-optimized.png
        ¦   logo-32x32.png
        ¦   logo.png
        ¦   textured_paper.jpeg
        ¦
        +---bootstrap
            +---css
            ¦       bootstrap-responsive.css
            ¦       bootstrap-responsive.min.css
            ¦       bootstrap.css
            ¦       bootstrap.min.css
            ¦
            +---img
            ¦       glyphicons-halflings-white.png
            ¦       glyphicons-halflings.png
            ¦
            +---js
                    bootstrap.js
                    bootstrap.min.js

geodata
    pebble-beach-fairways-3857.geojson
    pebble-beach-greens-3857.geojson
    results_sym_diff.js
```

在文件夹结构就绪的情况下，当你运行代码时，所有输入和输出都将找到它们正确的家。

## 如何做到这一点...

我们想像往常一样从命令行运行此代码，它将在您的虚拟环境中运行：

1.  从您的 `/ch06/code` 文件夹执行以下语句：

    ```py
    >> python Ch06-01_sym_diff.py

    ```

1.  以下代码是 Shapely 中有趣操作发生的地方：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    import json
    from os.path import realpath
    from shapely.geometry import MultiPolygon
    from shapely.geometry import asShape
    from shapely.wkt import dumps

    # define our files input and output locations
    input_fairways = realpath("../geodata/pebble-beach-fairways-3857.geojson")
    input_greens = realpath("../geodata/pebble-beach-greens-3857.geojson")
    output_wkt_sym_diff = realpath("ol3/data/results_sym_diff.js")

    # open and load our geojson files as python dictionary
    with open(input_fairways) as fairways:
        fairways_data = json.load(fairways)

    with open(input_greens) as greens:
        greens_data = json.load(greens)

    # create storage list for our new shapely objects
    fairways_multiply = []
    green_multply = []

    # create shapely geometry objects for fairways
    for feature in fairways_data['features']:
        shape = asShape(feature['geometry'])
        fairways_multiply.append(shape)

    # create shapely geometry objects for greens
    for green in greens_data['features']:
        green_shape = asShape(green['geometry'])
        green_multply.append(green_shape)

    # create shapely MultiPolygon objects for input analysis
    fairway_plys = MultiPolygon(fairways_multiply)
    greens_plys = MultiPolygon(green_multply)

    # run the symmetric difference function creating a new Multipolygon
    result = fairway_plys.symmetric_difference(greens_plys)

    # write the results out to well known text (wkt) with shapely dump
    def write_wkt(filepath, features):
        with open(filepath, "w") as f:
            # create a js variable called ply_data used in html
            # Shapely dumps geometry out to WKT
            f.write("var ply_data = '" + dumps(features) + "'")

    # write to our output js file the new polygon as wkt
    write_wkt(output_wkt_sym_diff, result)
    ```

    您的输出将保存在 `/ch06/code/ol3/html/` 文件夹中，文件名为 `ch06-01_sym_diff.html`。只需在您的本地网页浏览器中打开此文件，例如 Chrome、Firefox 或 Safari。我们的输出网络地图是通过根据我们的需求修改 Openlayers 3 示例代码页面创建的。生成的网络地图应在您的本地网页浏览器中显示以下地图：

    ![如何做到这一点...](img/50790OS_06_03.jpg)

您现在可以清楚地看到航道内部有一个洞。

## 它是如何工作的...

首先，我们使用两个 **GeoJSON** 数据集作为我们的输入，它们都具有 EPSG: 3857，并源自 OSM EPSG: 4326。转换过程在此未涉及；有关如何在两个坐标系之间转换数据的更多信息，请参阅 第二章，*处理投影*。

我们的第一项任务是使用标准的 Python `json` 模块将两个 GeoJSON 文件读入 Python 字典对象。接下来，我们设置一些空列表，这些列表将存储 Shapely 几何对象列表，用作我们的输入以生成分析所需的 `MultiPolygons`。我们使用 Shapely 内置的 `asShape()` 函数创建 Shapely 几何对象，以便我们可以执行空间操作。这是通过访问字典的 `['geometry']` 元素来实现的。然后我们将每个几何形状追加到我们的空列表中。然后，这个列表被输入到 Shapely 的 `MultiPolygon()` 函数中，该函数将为我们创建一个 MultiPolygon，并用作我们的输入。

实际上运行我们的 `symmetric_difference` 过程发生在我们输入 `fairways_plys` MultiPolygon 作为输入，并传递参数 `greens_ply` MultiPolygon 时。输出存储在 `result` 变量中，它本身也是一个 MultiPolygon。别忘了，MultiPolygon 只是一个我们可以迭代的多边形列表。

接下来，我们将查看一个名为 `write_wkt(filepath, features)` 的函数。这个函数将我们的结果 MultiPolygon Shapely 几何形状输出到 `Well Known Text (WKT)` 格式。我们不仅输出这个 `WKT`，而是创建一个新的 JavaScript 文件，`ol3/data/ch06-01_results_sym_diff.js`，包含我们的 `WKT` 输出。代码输出一个字符串，创建一个名为 `ply_data` 的 JavaScript 变量。这个 `ply_data` 变量随后被用于位于 `/ch06/code/ol3/html/sym_diff.html` 的我们的 HTML 文件中，以使用 Openlayers 3 绘制我们的 `WKT` 向量层。然后我们调用我们的函数，它执行写入到 `WKT` JavaScript 文件的操作。

这个示例是第一个将我们的结果可视化为网络地图的示例。在 第十一章，*使用 GeoDjango 进行网络分析*中，我们将探索一个功能齐全的网络映射应用程序；对于那些迫不及待的人，您可能想要提前跳读。接下来的示例将继续使用 Openlayers 3 作为我们的数据查看器，而不再使用 Matplotlib。

最后，我们的简单一行对称差执行需要大量的辅助代码来处理导入 GeoJSON 数据和以可以显示 Openlayers 3 网络地图的格式导出结果。

# 不合并的合并多边形

为了演示合并的概念，我们将从一个 **国家海洋和大气管理局** (**NOAA**) 的气象数据示例中获取例子。它提供了您下载数据的 Shapefiles 的令人惊叹的每分钟更新。我们将查看一周的天气预警集合，并将这些与州边界结合起来，以查看预警在州边界内确切发生的位置。

![不合并的合并多边形](img/50790OS_06_04.jpg)

前面的截图显示了在 QGIS 中进行合并操作之前的多边形。

## 准备工作

确保您的虚拟环境始终处于运行状态，并运行以下命令：

```py
$ source venvs/pygeo_analysis_cookbook/bin/activate

```

接下来，切换到您的 `/ch06/code/` 文件夹以查找完成的代码示例，或者在 `/ch06/` 工作文件夹中创建一个空文件，并按照代码进行操作。

## 如何操作...

`pyshp` 和 `shapely` 库是我们这个练习的两个主要工具：

1.  您可以直接在命令提示符中运行此文件以查看结果，如下所示：

    ```py
    >> python ch06-02_union.py

    ```

    然后，您可以通过双击打开 `/ch06/code/ol3/html/ch06-02_union.html` 文件夹中的结果，以在您的本地网络浏览器中启动它们。如果一切顺利，您应该看到以下网络地图：

    ![如何操作...](img/50790OS_06_05.jpg)

1.  现在，让我们看看使这一切发生的代码：

    ```py
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    import json
    from os.path import realpath
    import shapefile  # pyshp
    from geojson import Feature, FeatureCollection
    from shapely.geometry import asShape, MultiPolygon
    from shapely.ops import polygonize
    from shapely.wkt import dumps

    def create_shapes(shapefile_path):
        """
        Convert Shapefile Geometry to Shapely MultiPolygon
        :param shapefile_path: path to a shapefile on disk
        :return: shapely MultiPolygon
        """
        in_ply = shapefile.Reader(shapefile_path)

        # using pyshp reading geometry
        ply_shp = in_ply.shapes()
        ply_records = in_ply.records()
        ply_fields = in_ply.fields
        print ply_records
        print ply_fields

        if len(ply_shp) > 1:
            # using python list comprehension syntax
            # shapely asShape to convert to shapely geom
            ply_list = [asShape(feature) for feature in ply_shp]

            # create new shapely multipolygon
            out_multi_ply = MultiPolygon(ply_list)

            # # equivalent to the 2 lines above without using list comprehension
            # new_feature_list = []
            # for feature in features:
            #     temp = asShape(feature)
            #     new_feature_list.append(temp)
            # out_multi_ply = MultiPolygon(new_feature_list)

            print "converting to MultiPolygon: " + str(out_multi_ply)
        else:
            print "one or no features found"
            shply_ply = asShape(ply_shp)
            out_multi_ply = MultiPolygon(shply_ply)

        return out_multi_ply

    def create_union(in_ply1, in_ply2, result_geojson):
        """
        Create union polygon
        :param in_ply1: first input shapely polygon
        :param in_ply2: second input shapely polygon
        :param result_geojson: output geojson file including full file path
        :return: shapely MultiPolygon
        """
        # union the polygon outer linestrings together
        outer_bndry = in_ply1.boundary.union(in_ply2.boundary)

        # rebuild linestrings into polygons
        output_poly_list = polygonize(outer_bndry)

        out_geojson = dict(type='FeatureCollection', features=[])

        # generate geojson file output
        for (index_num, ply) in enumerate(output_poly_list):
            feature = dict(type='Feature', properties=dict(id=index_num))
            feature['geometry'] = ply.__geo_interface__
            out_geojson['features'].append(feature)

        # create geojson file on disk
        json.dump(out_geojson, open(result_geojson, 'w'))

        # create shapely MultiPolygon
        ply_list = []
        for fp in polygonize(outer_bndry):
            ply_list.append(fp)

        out_multi_ply = MultiPolygon(ply_list)

        return out_multi_ply

    def write_wkt(filepath, features):
        """

        :param filepath: output path for new JavaScript file
        :param features: shapely geometry features
        :return:
        """
        with open(filepath, "w") as f:
            # create a JavaScript variable called ply_data used in html
            # Shapely dumps geometry out to WKT
            f.write("var ply_data = '" + dumps(features) + "'")

    def output_geojson_fc(shply_features, outpath):
        """
        Create valid GeoJSON python dictionary
        :param shply_features: shapely geometries
        :param outpath:
        :return: GeoJSON FeatureCollection File
        """

        new_geojson = []
        for feature in shply_features:
            feature_geom_geojson = feature.__geo_interface__
            myfeat = Feature(geometry=feature_geom_geojson,
                             properties={'name': "mojo"})
            new_geojson.append(myfeat)

        out_feat_collect = FeatureCollection(new_geojson)

        with open(outpath, "w") as f:
            f.write(json.dumps(out_feat_collect))

    if __name__ == "__main__":

        # define our inputs
        shp1 = realpath("../geodata/temp1-ply.shp")
        shp2 = realpath("../geodata/temp2-ply.shp")

        # define outputs
        out_geojson_file = realpath("../geodata/res_union.geojson")
        output_union = realpath("../geodata/output_union.geojson")
        out_wkt_js = realpath("ol3/data/results_union.js")

        # create our shapely multipolygons for geoprocessing
        in_ply_1_shape = create_shapes(shp1)
        in_ply_2_shape = create_shapes(shp2)

        # run generate union function
        result_union = create_union(in_ply_1_shape, in_ply_2_shape, out_geojson_file)

        # write to our output js file the new polygon as wkt
        write_wkt(out_wkt_js, result_union)

        # write the results out to well known text (wkt) with shapely dump
        geojson_fc = output_geojson_fc(result_union, output_union)
    ```

## 它是如何工作的...

在代码开始部分快速浏览一下正在发生的事情应该有助于澄清。在我们的 Python 代码中，我们有四个函数和九个变量来分割输入和输出数据的负载。我们的代码运行发生在代码末尾的 `if __name__ == "main":` 调用中。我们开始定义两个变量来处理我们将要 **合并** 的输入。这两个是我们的输入 Shapefiles，其他三个输出是 GeoJSON 和 JavaScript 文件。

`create_shapes()` 函数将我们的 Shapefile 转换为 Shapely `MultiPolygon` 几何对象。在 Python 类内部，列表推导用于生成一个新列表，其中包含多边形对象，这些对象是我们用于创建输出 `MultiPolygon` 的输入多边形列表。接下来，我们将简单地运行这个函数，传入我们的输入 Shapefiles。

接下来是我们的`create_union()`函数，我们在这里进行真正的合并工作。我们首先将两个几何边界合并在一起，生成一个包含 LineStrings 的并集集合，代表输入多边形的边界。这样做的原因是我们不希望丢失两个多边形的几何形状，当直接传递给 Shapely 的合并函数时，它们将默认溶解成一个大的多边形。因此，我们需要使用`polygonize()` Shapely 函数重建多边形。

`polygonize`函数创建了一个 Python **生成器**对象，而不是一个简单的几何对象。这是一个类似于**列表**的**迭代器**，我们需要遍历它以获取它为我们创建的各个多边形。

我们在下一个代码段中正是这样做的，使用 Python 的`enumerate()`函数为每个我们用作 id 字段的属性结果自动创建一个 ID。在我们的循环之后，我们使用标准的 Python `json.dump()`方法导出我们新创建的 GeoJSON 文件，并使用 Python 的`open()`方法以写入模式将其写入磁盘。

最后，在我们的`create_union()`函数中，我们准备输出我们的结果**并集**多边形作为一个 Shapely MultiPolygon 对象。这通过简单地遍历`polygonize()`迭代器，输出一个列表，该列表输入到 Shapely 的`MultiPolygon()`函数中。最后，我们执行合并函数，传入我们的两个输入几何形状，并指定输出 GeoJSON 文件。

因此，我们可以像在之前的练习中一样，使用一个名为`write_wkt()`的小函数在我们的网络地图中查看我们的结果。这个小小的函数接受我们想要创建的输出 JavaScript 文件的文件路径以及 MultiPolygon 结果的几何形状。Shapely 然后将几何形状以写入 JavaScript 文件的方式转换为 Well Known Text 格式。

最后，一个名为`output_geojson_fc()`的小函数被用来输出另一个 GeoJSON 文件，这次使用 Python 的`geojson`库。这仅仅展示了另一种创建 GeoJSON 文件的方法。由于 GeoJSON 是一个纯文本文件，因此根据您的个人编程偏好，可以以许多独特的方式创建它。

# 通过合并（溶解）合并多边形

为了展示合并的概念，我们将从 NOAA 气象数据中举一个例子。它提供了令人惊叹的逐分钟更新 Shapefiles，以满足您下载数据的愿望。我们将查看一周的天气预警收集，并将这些预警合并在一起，得到本周发布的总预警区域。

这里展示了我们期望的结果的概念可视化：

![通过合并（溶解）合并多边形](img/50790OS_06_06.jpg)

大部分数据位于佛罗里达州附近，但在夏威夷和加利福尼亚州也有一些多边形。要查看原始数据或寻找新数据，请查看以下链接：

+   [`www.nws.noaa.gov/geodata/catalog/wsom/html/pubzone.htm`](http://www.nws.noaa.gov/geodata/catalog/wsom/html/pubzone.htm)

+   [`nws.noaa.gov/regsci/gis/week.html`](http://nws.noaa.gov/regsci/gis/week.html)

+   [`www.nws.noaa.gov/geodata/index.html`](http://www.nws.noaa.gov/geodata/index.html)

如果你想查看州界，你可以在[`www.census.gov/geo/maps-data/data/cbf/cbf_state.html`](https://www.census.gov/geo/maps-data/data/cbf/cbf_state.html)找到它们。

这里是佛罗里达州在联盟之前的数据样本的样子，它使用 QGIS 进行了可视化：

![使用合并（溶解）合并多边形](img/50790OS_06_07.jpg)

## 准备工作

需要遵循常规的业务顺序才能开始这段代码。启动你的虚拟环境，并检查你的数据是否全部下载并位于你的`/ch06/geodata/`文件夹中。如果一切准备就绪，就直接开始编写代码。

## 如何操作...

我们的数据至少有点杂乱，所以请按照我们的步骤概述的解决方案进行操作，以便我们能够处理并运行分析函数`union`：

```py
# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from shapely.geometry import MultiPolygon
from shapely.ops import cascaded_union
from os.path import realpath
from utils import create_shapes
from utils import out_geoj
from utils import write_wkt

def check_geom(in_geom):
    """
    :param in_geom: input valid Shapely geometry objects
    :return: Shapely MultiPolygon cleaned
    """
    plys = []
    for g in in_geom:
        # if geometry is NOT valid
        if not g.is_valid:
            print "Oh no invalid geometry"
            # clean polygon with buffer 0 distance trick
            new_ply = g.buffer(0)
            print "now lets make it valid"
            # add new geometry to list
            plys.append(new_ply)
        else:
            # add valid geometry to list
            plys.append(g)
    # convert new polygons into a new MultiPolygon
    out_new_valid_multi = MultiPolygon(plys)
    return out_new_valid_multi

if __name__ == "__main__":

    # input NOAA Shapefile
    shp = realpath("../geodata/temp-all-warn-week.shp")

    # output union_dissolve results as GeoJSON
    out_geojson_file = realpath("../geodata/ch06-03_union_dissolve.geojson")

    out_wkt_js = realpath("ol3/data/ch06-03_results_union.js")

    # input Shapefile and convert to Shapely geometries
    shply_geom = create_shapes(shp)

    # Check the Shapely geometries if they are valid if not fix them
    new_valid_geom = check_geom(shply_geom)

    # run our union with dissolve
    dissolve_result = cascaded_union(new_valid_geom)

    # output the resulting union dissolved polygons to GeoJSON file
    out_geoj(dissolve_result, out_geojson_file)

    write_wkt(out_wkt_js, dissolve_result)
```

你的结果网络地图将看起来像这样：

![如何操作...](img/50790OS_06_08.jpg)

## 它是如何工作的...

我们开始越来越多地重用现在藏在我们`/ch06/code/utils.py`模块中的代码。正如你在导入中看到的那样，我们使用三个函数进行数据的标准输入和输出。主应用程序从定义我们的 NOAA 输入 Shapefile 和定义输出 GeoJSON 文件开始。然后，如果我们运行代码，它将由于数据有效性问题而崩溃。因此，我们创建了一个新函数来检查我们的输入数据中的无效几何形状。这个新函数将捕获这些无效几何形状并将它们转换为有效的多边形。

Shapely 有一个名为`is_valid`的几何属性，它访问 GEOS 引擎，根据 OGC 规范中的简单特征来检查几何的有效性。

### 小贴士

如果你正在寻找所有可能的无效数据可能性，你可以在开放地理空间联盟网站上找到更多信息。查看第*28*页的简单特征标准；你将在[`portal.opengeospatial.org/files/?artifact_id=25355`](http://portal.opengeospatial.org/files/?artifact_id=25355)找到无效多边形的示例。

这些异常的原因是，当数据重叠和处理时，几何形状会以不是总是最优的角度组合或切割。

最后，我们有了干净的数据可以工作，通过运行 Shapely 的`cascaded_union()`函数，这将溶解我们所有的重叠多边形。我们的结果多边形进一步推入我们的`out_geoj()`函数，该函数最终将新的几何形状写入我们`/ch06/geodata`文件夹中的磁盘。

# 执行身份函数（差集+交集）

在 ESRI 地理处理术语中，有一个名为`identity`的重叠功能。当你想要保留所有原始几何边界，并且仅与输入特征的重叠相结合时，这是一个非常有用的功能。

![执行身份函数（差集 + 交集）](img/50790OS_06_09.jpg)

这归结为一个公式，需要同时调用`difference`和`intersect`。我们首先找到差集（`输入特征 - 交集`），然后添加交集以创建我们的结果如下：

```py
 (input feature – intersection) + intersection = result

```

## 如何做到这一点...

1.  对于所有好奇的人，如果你想知道如何做到这一点，请输入以下代码；它将帮助你记忆肌肉：

    ```py
    ##!/usr/bin/env python
    # -*- coding: utf-8 -*-
    from shapely.geometry import asShape, MultiPolygon
    from utils import shp2_geojson_obj, out_geoj, write_wkt
    from os.path import realpath

    def create_polys(shp_data):
        """
        :param shp_data: input GeoJSON
        :return: MultiPolygon Shapely geometry
        """
        plys = []
        for feature in shp_data['features']:
            shape = asShape(feature['geometry'])
            plys.append(shape)

        new_multi = MultiPolygon(plys)
        return new_multi

    def create_out(res1, res2):
        """

        :param res1: input feature
        :param res2: identity feature
        :return: MultiPolygon identity results
        """
        identity_geoms = []

        for g1 in res1:
            identity_geoms.append(g1)
        for g2 in res2:
            identity_geoms.append(g2)

        out_identity = MultiPolygon(identity_geoms)
        return out_identity

    if __name__ == "__main__":
        # out two input test Shapefiles
        shp1 = realpath("../geodata/temp1-ply.shp")
        shp2 = realpath("../geodata/temp2-ply.shp")

        # output resulting GeoJSON file
        out_geojson_file = realpath("../geodata/result_identity.geojson")

        output_wkt_identity = realpath("ol3/data/ch06-04_results_identity.js")

        # convert our Shapefiles to GeoJSON
        # then to python dictionaries
        shp1_data = shp2_geojson_obj(shp1)
        shp2_data = shp2_geojson_obj(shp2)

        # transform our GeoJSON data into Shapely geom objects
        shp1_polys = create_polys(shp1_data)
        shp2_polys = create_polys(shp2_data)

        # run the difference and intersection
        res_difference = shp1_polys.difference(shp2_polys)
        res_intersection = shp1_polys.intersection(shp2_polys)

        # combine the difference and intersection polygons into results
        result_identity = create_out(res_difference, res_intersection)

        # export identity results to a GeoJSON
        out_geoj(result_identity, out_geojson_file)

        # write out new JavaScript variable with wkt geometry
        write_wkt(output_wkt_identity, result_identity )
    ```

    现在生成的多边形可以在你的浏览器中可视化。现在只需打开`/ch06/code/ol3/html/ch06-04_identity.html`文件，你将看到这张地图：

    ![如何做到这一点...](img/50790OS_06_10.jpg)

## 它是如何工作的...

我们在我们的`util.py`工具文件中隐藏了两颗宝石，名为`shp2_geojson_obj`和`out_geoj`。第一个函数接收我们的 Shapefile 并返回一个 Python 字典对象。我们的函数实际上创建了一个有效的 GeoJSON，以 Python 字典的形式，可以很容易地使用标准的`json.dumps()`Python 模块转换为 JSON 字符串。

在处理完这些前置工作之后，我们可以跳到创建 Shapely 几何体，这些几何体可以用于我们的分析。`create_polys()`函数正是这样做的：它接收我们的几何体，返回一个`MultiPolygon`。这个`MultiPolygon`用于计算我们的差集和交集。

因此，最后，我们可以从 Shapely 的差集函数开始进行分析计算，使用我们的`temp1-ply.shp`作为输入特征，`temp2-poly.shp`作为身份特征。差集函数只返回不与另一个特征相交的输入特征的几何体。接下来，我们执行交集函数，它只返回两个输入之间的重叠几何体。

我们的配方几乎完成了；我们只需要将这两个新结果结合起来，以产生我们新的身份结果的多边形。`create_out()`函数接受两个参数，第一个是我们的输入特征，第二个是我们的结果交集特征。顺序非常重要；否则你的结果将被反转。所以请确保你输入正确的顺序。

我们遍历每个几何体，将它们组合成一个名为`result_identity`的复杂新`MultiPolygon`。然后将其泵入我们的`out_geoj()`函数，该函数将写入一个新的 GeoJSON 文件到你的`/ch06/geodata`文件夹。

我们的`out_geoj()`函数位于`utils.py`文件中，可能需要简要说明。输入是一个几何列表和输出 GeoJSON 文件在磁盘上的文件路径。我们简单地创建一个新的字典，然后遍历每个几何体，使用内置的 Shapely `__geo_interface__`将 Shapely 几何体导出到 GeoJSON 文件。

### 注意

如果你想了解`__geo_interface__`，请自行查阅并了解它是什么以及为什么它如此酷，请访问[`gist.github.com/sgillies/2217756`](https://gist.github.com/sgillies/2217756)。

对于那些正在寻找两个效用函数的各位，这里就是供您阅读的版本：

```py
def shp2_geojson_obj(shapefile_path):
    # open shapefile
    in_ply = shapefile.Reader(shapefile_path)
    # get a list of geometry and records
    shp_records = in_ply.shapeRecords()
    # get list of fields excluding first list object
    fc_fields = in_ply.fields[1:]

    # using list comprehension to create list of field names
    field_names = [field_name[0] for field_name in fc_fields ]
    my_fc_list = []
    # run through each shape geometry and attribute
    for x in shp_records:
        field_attributes = dict(zip(field_names, x.record))
        geom_j = x.shape.__geo_interface__
        my_fc_list.append(dict(type='Feature', geometry=geom_j,
                               properties=field_attributes))

    geoj_json_obj = {'type': 'FeatureCollection',
                    'features': my_fc_list}

    return geoj_json_obj
def out_geoj(list_geom, out_geoj_file):
    out_geojson = dict(type='FeatureCollection', features=[])

    # generate geojson file output
    for (index_num, ply) in enumerate(list_geom):
        feature = dict(type='Feature', properties=dict(id=index_num))
        feature['geometry'] = ply.__geo_interface__
        out_geojson['features'].append(feature)

    # create geojson file on disk
    json.dump(out_geojson, open(out_geoj_file, 'w'))
```
