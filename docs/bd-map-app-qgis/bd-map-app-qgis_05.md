# 第五章。在外部应用程序中使用 QGIS

在 第一章，*使用 QGIS 入门*中，我们简要地查看了一个使用 PyQt 和 PyQGIS 库构建的独立 Python 程序。在本章中，我们将使用相同的技巧，使用 PyQGIS 构建一个完整的即插即用地图应用程序。在这个过程中，我们将：

+   设计和构建一个简单但完整的独立地图应用程序

+   学习如何在我们的 Python 程序运行之前使用包装脚本来处理平台特定的依赖项

+   在单独的 Python 模块中定义我们应用程序的用户界面，以便我们将 UI 与应用程序的业务逻辑分开

+   根据用户的偏好动态显示和隐藏地图图层

+   学习如何使用基于规则的渲染器根据地图当前的缩放级别选择性地显示特征

+   看看如何使用数据定义的属性来计算用于标签的字体大小，基于特征的属性

+   实现谷歌地图风格的平移和缩放

# 介绍 Lex

我们的地图应用程序将显示世界地图，允许用户缩放和平移，并在地图上显示各种地标。如果用户点击一个地标，将显示该地标的详细信息。

我们将把我们的应用程序称为 **Lex**，它是 **L**andmark **ex**plorer 的缩写。Lex 将使用两个免费提供的地理空间数据集：一个高分辨率的阴影地形图，以及一个全面的地点名称数据库，我们将使用它作为显示的地标列表：

![介绍 Lex](img/00047.jpeg)

我们将使用 PyQt 构建 Lex 应用程序，并利用 QGIS 内置的 PyQGIS 库来完成大部分繁重的工作。

对于 Lex 应用程序，我们的要求如下：

+   它必须作为一个即插即用应用程序运行。双击启动器脚本必须启动 PyQt 程序，加载所有数据，并向用户展示一个完整的工作应用程序。

+   用户界面必须尽可能专业，包括键盘快捷方式和美观的工具栏图标。

+   当用户点击一个地标时，应显示该地标的名称和管辖区域、时区和经纬度。

+   外观和感觉应尽可能类似于谷歌地图。

    ### 注意

    这个最后的要求是一个重要的点，因为 QGIS 内置的缩放和平移工具比我们希望在即插即用地图应用程序中拥有的要复杂。大多数用户已经熟悉谷歌地图的行为，我们希望模仿这种行为，而不是使用 QGIS 提供的默认平移和缩放工具。

不再拖延，让我们开始构建我们的应用程序。我们的第一步将是下载应用程序将基于的地理空间数据。

# 获取数据

Lex 将使用两个地图层：一个**底图层**显示阴影高程栅格图像，以及一个**地标层**根据一组地名显示单个地标。这两个数据集都可以从自然地球数据网站下载。访问[`www.naturalearthdata.com`](http://www.naturalearthdata.com)，并点击**获取数据**链接跳转到**下载**页面。

通过点击**栅格**链接可以找到底图数据。我们希望使用最高分辨率的可用数据，因此请使用**大比例尺数据，1:10m**部分中的链接。

虽然你可以使用这些数据集作为底图，但我们将下载**自然地球 I 带阴影高程、水和排水**数据集。确保你下载这个数据集的高分辨率版本，这样当用户放大时，栅格图像仍然看起来很好。

对于地标，我们将使用“人口密集地区”数据集。返回主下载页面，在**大比例尺数据，1:10m**部分点击**文化**链接。向下滚动到**人口密集地区**部分，并点击**下载人口密集地区**链接。

下载完成后，你应该在电脑上有两个 ZIP 存档：

`NE1_HR_LC_SR_W_DR.zip`

`ne_10m_populated_places.zip`

创建一个名为`data`的文件夹，解压缩前面的两个 ZIP 存档，并将生成的目录放入你的`data`文件夹中。

# 设计应用程序

我们现在有一份我们映射应用的需求列表，以及我们想要显示的地理空间数据。然而，在我们开始编码之前，退一步思考我们应用的用户界面是个好主意。

我们的应用程序将有一个主窗口，我们将称之为**地标探索器**。为了使其易于使用，我们将显示一个地图画布以及一个简单的工具栏。我们的基本窗口布局将如下所示：

![设计应用程序](img/00048.jpeg)

除了主窗口外，我们的 Lex 应用程序还将有一个包含以下菜单的菜单栏：

![设计应用程序](img/00049.jpeg)

工具栏将使新用户通过点击工具栏图标来使用 Lex 变得容易，而经验丰富的用户可以利用广泛的键盘快捷键来访问程序的功能。

带着这个设计思路，让我们开始编码。

# 创建应用程序框架

首先创建一个用于存放应用程序源代码的文件夹，并将你之前创建的数据文件夹移动到其中。接下来，我们想要使用我们在第一章中学习的技术来创建我们应用程序的基本框架，即*使用 QGIS 入门*。创建一个名为`lex.py`的模块，并将以下内容输入到该文件中：

```py
import os, os.path, sys

from qgis.core import *
from qgis.gui import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *

class MapExplorer(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Landmark Explorer")
        self.resize(800, 400)

def main():
    QgsApplication.setPrefixPath(os.environ['QGIS_PREFIX'], True)
    QgsApplication.initQgis()

    app = QApplication(sys.argv)

    window = MapExplorer()
    window.show()
    window.raise_()

    app.exec_()
    app.deleteLater()
    QgsApplication.exitQgis()

if __name__ == "__main__":
    main()
```

我们只是导入所需的各个库，并使用我们之前学到的技术设置一个外部 PyQGIS 应用程序。然后我们创建并显示一个空白窗口，以便应用程序在启动时能做些事情。

由于我们希望 Lex 应用程序能在任何操作系统上运行，我们不会将 QGIS 的路径硬编码到我们的源代码中。相反，我们将编写一个 **封装脚本**，在启动我们的 Python 程序之前设置所需的环境变量。由于这些封装脚本依赖于操作系统，您需要为您的操作系统创建一个适当的封装脚本。

### 注意

注意，我们在 `lex.py` 模块中使用 `os.environ['QGIS_PREFIX']` 以避免将 QGIS 应用程序的路径硬编码到我们的源代码中。我们的封装脚本将负责在应用程序运行之前设置这个环境变量。

如果您使用的是 Microsoft Windows 计算机上的计算机，您的封装脚本看起来可能如下所示：

```py
SET OSGEO4W_ROOT=C:\OSGeo4W
SET QGIS_PREFIX=%OSGEO4W_ROOT%\apps\qgis
SET PATH=%QGIS_PREFIX%\bin;%OSGWO4W_ROOT\bin;%PATH%
SET PYTHONPATH=%QGIS_PREFIX%\python;%OSEO4W_ROOT%\apps\Python27;%PYTHONPATH%
SET PYTHONHOME=%OSGEO4W_ROOT%\apps\Python27
python lex.py
```

将此脚本命名为有意义的名称，例如，`run.bat`，并将其放在与您的 `lex.py` 模块相同的目录中。

如果您使用的是运行 Linux 的计算机，您的封装脚本将被命名为类似 `run.sh` 的名称，并看起来如下所示：

```py
export PYTHONPATH="/path/to/qgis/build/output/python/"
export LD_LIBRARY_PATH="/path/to/qgis/build/output/lib/"
export QGIS_PREFIX="/path/to/qgis/build/output/"
python lex.py
```

您需要修改路径以指向 QGIS 已安装的目录。

对于运行 Mac OS X 的用户，您的封装脚本也将被命名为 `run.sh`，并包含以下内容：

```py
export PYTHONPATH="$PYTHONPATH:/Applications/QGIS.app/Contents/Resources/python"
export DYLD_FRAMEWORK_PATH="/Applications/QGIS.app/Contents/Frameworks"
export QGIS_PREFIX="/Applications/QGIS.app/Contents/Resources"
python lex.py
```

注意，对于 Mac OS X 和 Linux 系统，我们必须设置框架或库路径。这允许 PyQGIS 的 Python 封装器找到它们所依赖的底层 C++ 共享库。

### 提示

如果您在 Linux 或 Mac OS X 下运行，您还必须使您的封装脚本可执行。为此，请在 bash shell 或终端窗口中输入 `chmod +x run.sh`。

一旦您创建了您的 shell 脚本，尝试运行它。如果一切顺利，您的 PyQt 应用程序应该启动并显示一个空白窗口，如下所示：

![创建应用程序框架](img/00050.jpeg)

如果它不起作用，您需要检查您的封装脚本和/或您的 `lex.py` 模块。您可能需要修改目录路径以匹配您的 QGIS 和 Python 安装。

# 添加用户界面

现在我们程序正在运行，我们可以开始实现用户界面（UI）。一个典型的 PyQt 应用程序将使用 Qt Designer 将应用程序的 UI 存储在一个模板文件中，然后将其编译成一个 Python 模块，以便在您的应用程序中使用。

由于描述如何使用 Qt Designer 来布局带有工具栏和菜单的窗口需要很多页面，我们将采取捷径，直接在 Python 中创建用户界面。同时，我们还将创建我们的 UI 模块，就像它是使用 Qt Designer 创建的一样；这使我们的应用程序 UI 保持独立，同时也展示了如果使用 Qt Designer 设计用户界面，我们的应用程序将如何工作。

创建一个名为`ui_explorerWindow.py`的新模块，并将以下代码输入到该模块中：

```py
from PyQt4 import QtGui, QtCore

import resources

class Ui_ExplorerWindow(object):
    def setupUi(self, window):
        window.setWindowTitle("Landmark Explorer")

        self.centralWidget = QtGui.QWidget(window)
        self.centralWidget.setMinimumSize(800, 400)
        window.setCentralWidget(self.centralWidget)

        self.menubar = window.menuBar()
        self.fileMenu = self.menubar.addMenu("File")
        self.viewMenu = self.menubar.addMenu("View")
        self.modeMenu = self.menubar.addMenu("Mode")

        self.toolBar = QtGui.QToolBar(window)
        window.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.actionQuit = QtGui.QAction("Quit", window)
        self.actionQuit.setShortcut(QtGui.QKeySequence.Quit)

        self.actionShowBasemapLayer = QtGui.QAction("Basemap", window)
        self.actionShowBasemapLayer.setShortcut("Ctrl+B")
        self.actionShowBasemapLayer.setCheckable(True)

        self.actionShowLandmarkLayer = QtGui.QAction("Landmarks", window)
        self.actionShowLandmarkLayer.setShortcut("Ctrl+L")
        self.actionShowLandmarkLayer.setCheckable(True)

        icon = QtGui.QIcon(":/icons/mActionZoomIn.png")
        self.actionZoomIn = QtGui.QAction(icon, "Zoom In", window)
        self.actionZoomIn.setShortcut(QtGui.QKeySequence.ZoomIn)

        icon = QtGui.QIcon(":/icons/mActionZoomOut.png")
        self.actionZoomOut = QtGui.QAction(icon, "Zoom Out", window)
        self.actionZoomOut.setShortcut(QtGui.QKeySequence.ZoomOut)

        icon = QtGui.QIcon(":/icons/mActionPan.png")
        self.actionPan = QtGui.QAction(icon, "Pan", window)
        self.actionPan.setShortcut("Ctrl+1")
        self.actionPan.setCheckable(True)

        icon = QtGui.QIcon(":/icons/mActionExplore.png")
        self.actionExplore = QtGui.QAction(icon, "Explore", window)
        self.actionExplore.setShortcut("Ctrl+2")
        self.actionExplore.setCheckable(True)

        self.fileMenu.addAction(self.actionQuit)

        self.viewMenu.addAction(self.actionShowBasemapLayer)
        self.viewMenu.addAction(self.actionShowLandmarkLayer)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.actionZoomIn)
        self.viewMenu.addAction(self.actionZoomOut)

        self.modeMenu.addAction(self.actionPan)
        self.modeMenu.addAction(self.actionExplore)

        self.toolBar.addAction(self.actionZoomIn)
        self.toolBar.addAction(self.actionZoomOut)
        self.toolBar.addAction(self.actionPan)
        self.toolBar.addAction(self.actionExplore)

        window.resize(window.sizeHint())
```

此模块实现了我们的 Lex 应用程序的用户界面，为每个工具栏和菜单项定义了一个`QtAction`对象，创建了一个用于容纳我们的地图画布的小部件，并在`QtMainWindow`对象内布局一切。此模块的结构与 Qt Designer 和`pyuic4`命令行工具将用户界面模板提供给 Python 代码的方式相同。

注意，`Ui_ExplorerWindow`类使用了多个工具栏图标。我们需要创建这些图标图像并在资源描述文件中定义它们，就像我们在上一章中创建`resources.py`模块一样。

我们将需要以下图标图像：

+   `mActionZoomIn.png`

+   `mActionZoomOut.png`

+   `mActionPan.png`

+   `mActionExplore.png`

如果你愿意，你可以从 QGIS 源代码库中下载这些图像文件（SVG 格式）[`github.com/qgis/QGIS/tree/master/images/themes/default`](https://github.com/qgis/QGIS/tree/master/images/themes/default)，但你需要将它们从`.svg`转换为`.png`以避免图像文件格式问题。如果你不想自己转换图标，这些图像作为本书提供的源代码的一部分可用。完成后，将这些四个文件放置在 Lex 应用程序的主目录中。

### 小贴士

注意，`mActionExplore.png`图标文件是源代码库中`mActionIdentify.svg`图像的转换副本。我们将图像文件重命名为与 Lex 应用程序中工具的名称相匹配。

接下来，我们需要创建我们的`resources.qrc`文件，以便 PyQt 可以使用这些图像。创建此文件并输入以下内容：

```py
<RCC>
    <qresource prefix="/icons">
        <file>mActionZoomIn.png</file>
        <file>mActionZoomOut.png</file>
        <file>mActionPan.png</file>
        <file>mActionExplore.png</file>
    </qresource>
</RCC>
```

你需要使用`pyrcc4`编译此文件。这将为你提供用户界面所需的`resources.py`模块。

现在我们已经定义了我们的用户界面，让我们修改`lex.py`模块以使用它。将以下`import`语句添加到模块的顶部：

```py
from ui_explorerWindow import Ui_ExplorerWindow
import resources
```

接下来，我们想要用我们新的 UI 替换`MapExplorer`窗口的占位实现。`MapExplorer`类的定义应该如下所示：

```py
class MapExplorer(QMainWindow, Ui_ExplorerWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setupUi(self)
```

如果一切顺利，我们的应用程序现在应该运行带有完整的用户界面——工具栏、菜单和我们的地图画布的空间：

![添加用户界面](img/00051.jpeg)

当然，我们的用户界面目前还没有任何功能，但我们的 Lex 应用程序开始看起来像是一个真正的程序。现在，让我们实现 UI 背后的行为。

# 连接操作

你可能已经注意到，菜单命令和工具栏图标目前都没有任何作用——即使是**退出**命令也不工作。在我们操作之前，我们必须将它们连接到适当的方法。为此，请将以下内容添加到`MapExplorer.__init__()`方法中，紧接在调用`setupUi()`之后：

```py
        self.connect(self.actionQuit,
                     SIGNAL("triggered()"), qApp.quit)
        self.connect(self.actionShowBasemapLayer,
                     SIGNAL("triggered()"), self.showBasemapLayer)
        self.connect(self.actionShowLandmarkLayer,
                     SIGNAL("triggered()"),
                     self.showLandmarkLayer)
        self.connect(self.actionZoomIn,
                     SIGNAL("triggered()"), self.zoomIn)
        self.connect(self.actionZoomOut,
                     SIGNAL("triggered()"), self.zoomOut)
        self.connect(self.actionPan,
                     SIGNAL("triggered()"), self.setPanMode)
        self.connect(self.actionExplore,
                     SIGNAL("triggered()"), self.setExploreMode)
```

我们将我们的 **退出** 动作连接到 `qApp.quit()` 方法。对于其他动作，我们将在 `MapExplorer` 类本身内部调用方法。让我们为这些方法定义一些占位符：

```py
    def showBasemapLayer(self):
        pass

    def showLandmarkLayer(self):
        pass

    def zoomIn(self):
        pass

    def zoomOut(self):
        pass

    def setPanMode(self):
        pass

    def setExploreMode(self):
        pass
```

我们将在地图画布设置好并运行之后实现这些方法。

# 创建地图画布

我们的 `Ui_ExplorerWindow` 类定义了一个名为 `centralWidget` 的实例变量，它作为窗口内容的占位符。由于我们想在窗口中放置一个 QGIS 地图画布，让我们实现创建地图画布并将其放置到这个中央小部件中的代码。将以下内容添加到 `MapExplorer` 窗口的 `__init__()` 方法的末尾（在 `lex.py` 中）：

```py
        self.mapCanvas = QgsMapCanvas()
        self.mapCanvas.useImageToRender(False)
        self.mapCanvas.setCanvasColor(Qt.white)
        self.mapCanvas.show()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mapCanvas)
        self.centralWidget.setLayout(layout)
```

接下来，我们希望将底图和地标图图层填充到地图画布中。为此，我们将定义一个新的方法，称为 `loadMap()`，并在适当的时候调用它。将以下方法添加到您的 `MapExplorer` 类中：

```py
    def loadMap(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(cur_dir, "data",
                                "NE1_HR_LC_SR_W_DR",
                                "NE1_HR_LC_SR_W_DR.tif")
        self.basemap_layer = QgsRasterLayer(filename, "basemap")
        QgsMapLayerRegistry.instance().addMapLayer(
                self.basemap_layer)

        filename = os.path.join(cur_dir, "data",
                                "ne_10m_populated_places",
                                "ne_10m_populated_places.shp")
        self.landmark_layer = QgsVectorLayer(filename,
                                             "landmarks", "ogr")
        QgsMapLayerRegistry.instance().addMapLayer(
               self.landmark_layer)

        self.showVisibleMapLayers()
        self.mapCanvas.setExtent(QgsRectangle(-127.7, 24.4, -79.3, 49.1))
```

此方法加载我们放置在 `data` 目录中的栅格和矢量数据集。然后我们调用一个新的方法 `showVisibleMapLayers()` 来使这些图层可见，并在应用程序首次启动时设置地图画布的范围以显示美国大陆。

让我们实现 `showVisibleMapLayers()` 方法：

```py
    def showVisibleMapLayers(self):
        layers = []
        if self.actionShowLandmarkLayer.isChecked():
            layers.append(QgsMapCanvasLayer(self.landmark_layer))
        if self.actionShowBasemapLayer.isChecked():
            layers.append(QgsMapCanvasLayer(self.basemap_layer))
        self.mapCanvas.setLayerSet(layers)
```

由于用户可以选择单独显示或隐藏底图和地标图层，我们只显示用户选择显示的图层。我们还将其放入一个单独的方法中，以便在用户切换图层的可见性时调用它。

在我们的地图可以显示之前，还有一些事情要做。首先，在调用 `window.raise_()` 之后，立即在 `main()` 函数中添加以下行：

```py
    window.loadMap()
```

这将在窗口显示后加载地图。接下来，将以下内容添加到主窗口的 `__init__()` 方法的末尾：

```py
        self.actionShowBasemapLayer.setChecked(True)
        self.actionShowLandmarkLayer.setChecked(True)
```

这使得两个图层在程序启动时可见。最后，让我们实现我们之前定义的两个方法，以便用户可以选择显示哪些图层：

```py
    def showBasemapLayer(self):
        self.showVisibleMapLayers()

    def showLandmarkLayer(self):
        self.showVisibleMapLayers()
```

运行程序应显示两个地图图层，您可以使用 **视图** 菜单中的命令显示或隐藏每个图层：

![创建地图画布](img/00052.jpeg)

# 标记点

如前图所示，每个地标仅由一个彩色点表示。为了使程序更有用，我们希望显示每个地标的名称。这可以通过使用 QGIS 内置的 "PAL" 标签引擎来完成。将以下代码添加到您的 `loadMap()` 方法中，在调用 `self.showVisibleMapLayers()` 之前立即执行：

```py
        p = QgsPalLayerSettings()
        p.readFromLayer(self.landmark_layer)
        p.enabled = True
        p.fieldName = "NAME"
        p.placement = QgsPalLayerSettings.OverPoint
        p.displayAll = True
        p.setDataDefinedProperty(QgsPalLayerSettings.Size,
                                 True, True, "12", "")
        p.quadOffset = QgsPalLayerSettings.QuadrantBelow
        p.yOffset = 1
        p.labelOffsetInMapUnits = False
        p.writeToLayer(self.landmark_layer)

        labelingEngine = QgsPalLabeling()
        self.mapCanvas.mapRenderer().setLabelingEngine(labelingEngine)
```

这将为地图上的每个点添加标签。不幸的是，有很多点，结果地图几乎无法阅读：

![标记点](img/00053.jpeg)

# 过滤地标

我们的标签之所以难以阅读，是因为显示的地标太多。然而，并非所有地标在所有缩放级别都相关——我们希望在地图缩放时隐藏太小而无法使用的地标，同时当用户放大时仍然显示这些地标。为此，我们将使用 `QgsRuleBasedRendererV2` 对象并利用 `SCALERANK` 属性来选择性地隐藏对于当前缩放级别来说太小的不必要特征。

在调用 `self.showVisibleMapLayers()` 之前，将以下代码添加到您的 `loadMap()` 方法中：

```py
        symbol = QgsSymbolV2.defaultSymbol(self.landmark_layer.geometryType())
        renderer = QgsRuleBasedRendererV2(symbol)
        root_rule = renderer.rootRule()
        default_rule = root_rule.children()[0]

        rule = default_rule.clone()
        rule.setFilterExpression("(SCALERANK >= 0) and (SCALERANK <= 1)")
        rule.setScaleMinDenom(0)
        rule.setScaleMaxDenom(99999999)
        root_rule.appendChild(rule)

        rule = default_rule.clone()
        rule.setFilterExpression("(SCALERANK >= 2) and (SCALERANK <= 4)")
        rule.setScaleMinDenom(0)
        rule.setScaleMaxDenom(10000000)
        root_rule.appendChild(rule)

        rule = default_rule.clone()
        rule.setFilterExpression("(SCALERANK >= 5) and (SCALERANK <= 7)")
        rule.setScaleMinDenom(0)
        rule.setScaleMaxDenom(5000000)
        root_rule.appendChild(rule)

        rule = default_rule.clone()
        rule.setFilterExpression("(SCALERANK >= 7) and (SCALERANK <= 10)")
        rule.setScaleMinDenom(0)
        rule.setScaleMaxDenom(2000000)
        root_rule.appendChild(rule)

        root_rule.removeChildAt(0)
        self.landmark_layer.setRendererV2(renderer)
```

这将在地图缩放时隐藏过小的地标（即具有过大 `SCALERANK` 值的地标）。现在，我们的地图看起来更加合理：

![过滤地标](img/00054.jpeg)

目前，我们还想添加一个功能；目前，所有标签的大小都是相同的。然而，我们希望较大的地标显示更大的标签。为此，将您的程序中的 `p.setDataDefinedProperty(...)` 行替换为以下内容：

```py
        expr = ("CASE WHEN SCALERANK IN (0,1) THEN 18" +
                "WHEN SCALERANK IN (2,3,4) THEN 14 " +
                "WHEN SCALERANK IN (5,6,7) THEN 12 " +
                "WHEN SCALERANK IN (8,9,10) THEN 10 " +
                "ELSE 9 END")
        p.setDataDefinedProperty(QgsPalLayerSettings.Size, True,
                                 True, expr, "")
```

这根据特征的 `SCALERANK` 属性值计算字体大小。正如您所想象的，以这种方式使用数据定义属性可以非常有用。

# 实现缩放工具

接下来，我们希望支持缩放和放大。如前所述，Lex 应用程序的一个要求是它必须像 Google Maps 而不是 QGIS 一样工作，这是一个我们必须支持的地方。QGIS 有一个缩放工具，用户点击它，然后在地图上点击或拖动以缩放。在 Lex 中，用户将直接点击工具栏图标来进行缩放。幸运的是，这很容易做到；只需以下方式实现 `zoomIn()` 和 `zoomOut()` 方法：

```py
    def zoomIn(self):
        self.mapCanvas.zoomIn()

    def zoomOut(self):
        self.mapCanvas.zoomOut()
```

现在，尝试运行您的程序。在您缩放和放大时，您可以看到各种地标的出现和消失，您也应该能够看到根据每个特征的 `SCALERANK` 值使用的不同字体大小。

# 实现平移工具

平移（即点击并拖动地图以移动）是另一个 QGIS 默认行为并不完全符合我们期望的领域。QGIS 包括一个 `classQgsMapToolPan` 类，它实现了平移；然而，它还包含了一些可能会让来自 Google Maps 的用户感到困惑的功能。特别是，如果用户点击而不拖动，地图将重新居中到点击的点。我们不会使用 `classQgsMapToolPan`，而是将实现我们自己的平移地图工具。幸运的是，这很简单：只需在 `MapExplorer` 类定义之后，将以下类定义添加到您的 `lex.py` 模块中：

```py
class PanTool(QgsMapTool):
    def __init__(self, mapCanvas):
        QgsMapTool.__init__(self, mapCanvas)
        self.setCursor(Qt.OpenHandCursor)
        self.dragging = False

    def canvasMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.dragging = True
            self.canvas().panAction(event)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragging:
            self.canvas().panActionEnd(event.pos())
            self.dragging = False
```

我们需要将以下内容添加到主窗口的 `__init__()` 方法末尾，以创建我们的平移工具的实例：

```py
        self.panTool = PanTool(self.mapCanvas)
        self.panTool.setAction(self.actionPan)
```

我们现在可以实施我们的 `setPanMode()` 方法来使用这个地图工具：

```py
    def setPanMode(self):
        self.actionPan.setChecked(True)
        self.mapCanvas.setMapTool(self.panTool)
```

最后，我们希望在应用程序启动时选择平移模式。为此，在调用 `window.loadMap()` 之后，将以下内容添加到您的 `main()` 函数中：

```py
    window.setPanMode()
```

# 实现探索模式

到目前为止，用户可以选择显示哪些地图图层，并且可以缩放和平移地图视图。唯一缺少的是应用程序的整个目的：探索地标。为此，我们必须实现应用程序的 **探索** 模式。

在上一章中，我们看到了如何使用 `QgsMapToolIdentify` 子类来响应用户点击矢量要素。我们将在这里使用相同的逻辑来实现一个新的地图工具，我们将称之为 `ExploreTool`。在 `PanTool` 类定义之后，将以下类定义添加到您的 `lex.py` 模块中：

```py
class ExploreTool(QgsMapToolIdentify):
    def __init__(self, window):
        QgsMapToolIdentify.__init__(self, window.mapCanvas)
        self.window = window

    def canvasReleaseEvent(self, event):
        found_features = self.identify(event.x(), event.y(),
                                       self.TopDownStopAtFirst,
                                       self.VectorLayer)
        if len(found_features) > 0:
            layer = found_features[0].mLayer
            feature = found_features[0].mFeature
            geometry = feature.geometry()

            info = []

            name = feature.attribute("NAME")
            if name != None: info.append(name)

            admin_0 = feature.attribute("ADM0NAME")
            admin_1 = feature.attribute("ADM1NAME")
            if admin_0 and admin_1:
                info.append(admin_1 + ", " + admin_0)

            timezone = feature.attribute("TIMEZONE")
            if timezone != None:
                info.append("Timezone: " + timezone)

            longitude = geometry.asPoint().x()
            latitude  = geometry.asPoint().y()
            info.append("Lat/Long: %0.4f, %0.4f" % (latitude,
                                                    longitude))

            QMessageBox.information(self.window,
                                    "Feature Info",
                                    "\n".join(info))
```

此工具识别用户点击的地标要素，提取该要素的相关属性，并在消息框中显示结果。要使用我们新的地图工具，我们必须将以下内容添加到 `MapExplorer` 窗口的 `__init__()` 方法的末尾：

```py
        self.exploreTool = ExploreTool(self)
        self.exploreTool.setAction(self.actionExplore)
```

我们接下来需要实现我们的 `setExploreMode()` 方法来使用这个工具：

```py
        def setExploreMode(self):
        self.actionPan.setChecked(False)
        self.actionExplore.setChecked(True)
        self.mapCanvas.setMapTool(self.exploreTool)
```

注意，当用户切换到探索模式时，我们必须取消勾选平移模式操作。这确保了两种模式是互斥的。我们必须采取的最后一步是修改我们的 `setPanMode()` 方法，以便当用户切换回平移模式时取消勾选探索模式操作。为此，将以下突出显示的行添加到您的 `setPanMode()` 方法中：

```py
    def setPanMode(self):
        self.actionPan.setChecked(True)
 self.actionExplore.setChecked(False)
        self.mapCanvas.setMapTool(self.panTool)
```

这完成了我们的 Lex 程序。现在用户可以放大和缩小，平移地图，并点击要素以获取有关该地标的更多信息：

![实现探索模式](img/00055.jpeg)

# 进一步的改进和增强

当然，虽然 Lex 是一个有用且完整的地图应用程序，但它实际上只是一个起点。免费提供的已有人口数据集提供的信息并不构成一个特别有趣的地标集，而且我们的应用程序仍然相当基础。以下是一些您可以对 Lex 应用程序进行的建议性改进：

+   添加一个 **搜索** 操作，用户可以输入要素的名称，Lex 将缩放和平移地图以显示该要素。

+   让用户选择任意两个地标，并显示这两个点之间的距离，单位为千米和英里。

+   允许用户加载他们自己的地标集，无论是从 shapefile 还是 Excel 电子表格中。当从 shapefile 加载时，用户可能会被提示选择要显示的每个要素的属性。当从电子表格（例如使用 `xlrd` 库）加载数据时，不同的列将包含纬度和经度值，以及要显示的标签和其他数据。

+   看看将 Lex 应用程序和 QGIS 本身捆绑成一个双击即可安装的操作系统安装程序涉及哪些内容。*PyQGIS 开发者手册*中提供了一些关于如何做到这一点的技巧，并且有各种工具，如 **py2exe** 和 **py2app**，您可以用它们作为起点。

实现这些额外功能是了解 PyQGIS 以及如何在您自己的独立地图程序中使用它的绝佳方式。

# 摘要

在本章中，我们使用 PyQGIS 设计并实现了一个简单但完整的即用型地图应用程序。在这个过程中，我们学习了如何使用包装脚本将特定平台的设置排除在您的 Python 程序之外。我们还看到了即使我们不使用 Qt Designer 创建用户界面模板，我们也可以在单独的模块中定义我们应用程序的 UI。

我们学习了如何使用 QGIS 内置的 "PAL" 标签引擎在矢量地图层中为每个要素显示标签。我们看到 `QgsRuleBasedRendererV2` 对象可以用来根据地图的缩放因子显示或隐藏某些要素，并且数据定义的属性允许我们计算诸如标签字体大小之类的值；我们还看到了如何使用 `CASE...WHEN` 表达式以复杂的方式计算数据定义的属性。

最后，我们看到了如何在地图应用程序中实现 Google Maps 风格的平移和缩放。

在下一章中，我们将了解 QGIS Python API 的更多高级功能以及我们如何在我们的地图应用程序中使用它们。
