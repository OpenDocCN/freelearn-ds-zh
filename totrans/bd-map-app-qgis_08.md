# 第八章. 使用 Python 和 QGIS 构建完整的地图应用程序

在本章中，我们将设计和开始构建一个完整的交钥匙地图应用程序。虽然我们的示例应用程序可能看起来有些专业，但设计和实现这个应用程序的过程以及我们使用的很大一部分代码，将适用于你可能会自己编写的所有类型的地图应用程序。

由于我们创建的应用程序复杂，我们将分两章实现。在本章中，我们将通过以下方式为地图应用程序打下基础：

+   设计应用程序

+   构建高分辨率底图，我们的矢量数据将在底图上显示

+   实现应用程序的整体结构

+   定义应用程序的用户界面

在下一章中，我们将实现地图工具，使用户能够输入和操作地图数据，编辑属性，并计算两点之间的最短路径。

# 介绍 ForestTrails

想象一下，你为一家负责开发和维护大型娱乐森林的公司工作。人们使用森林中的各种通道和专门建造的小径进行步行、骑自行车和骑马。你的任务是编写一个计算机程序，让用户创建一个数据库，包含通道和路径，以协助森林的持续维护。为了简单起见，我们将使用**路径**一词来指代通道或小径。每个路径都将具有以下属性：

+   **类型**：轨道是步行小径、自行车小径、马术小径还是通道

+   **名称**：并非所有小径和通道都有名称，尽管有些有

+   **方向**：一些小径和通道是单向的，而其他则可以双向通行

+   **状态**：轨道目前是否开放或关闭

由于娱乐森林持续发展，新的路径正在定期添加，而现有的路径有时会被修改，甚至在不再需要时被移除。这意味着你不能将路径集硬编码到你的程序中；你需要包含一个*路径编辑*模式，以便用户可以添加、编辑和删除路径。

你被赋予的特定要求是制作一套方向指南，以便轨道维护团队可以从一个给定的起点到达森林中的任何地方。为了实现这一点，程序将允许用户选择起点和终点，并计算并显示这两个点之间的**最短可用路径**。

# 设计 ForestTrails 应用程序

根据我们的需求集，很明显，路径可以用 LineString 几何形状来表示。我们还需要一个合适的底图，这些几何形状将在底图上显示。这意味着我们的应用程序至少将包含以下两个地图层：

![设计 ForestTrails 应用程序](img/00088.jpeg)

由于我们希望数据持久化，我们将使用 SpatiaLite 数据库来存储我们的轨迹数据，而底图则是一个我们加载并显示的 GeoTIFF 栅格图像。

除了这两个主要地图层之外，我们还将使用基于内存的层来在地图上显示以下临时信息：

+   当前选定的起点

+   当前选定的终点

+   这两点之间的最短路径

为了使事情更简单，我们将将这些信息分别显示在不同的地图层中。这意味着我们的应用程序将总共拥有五个地图层：

+   `basemapLayer`

+   `trackLayer`

+   `startPointLayer`

+   `endPointLayer`

+   `shortestPathLayer`

除了地图本身之外，我们的应用程序还将具有一个工具栏和一个菜单栏，这两个栏都允许用户访问系统的各种功能。以下操作将在工具栏和菜单栏中可用：

+   **放大**：这将允许用户放大地图。

+   **缩小**：这允许用户缩小地图。

+   **平移**：这是我们之前实现的平移模式，允许用户在地图上移动。

+   **编辑**：单击此项目将打开轨迹编辑模式。如果我们已经在轨迹编辑模式中，再次单击它将提示用户在关闭编辑模式之前保存他们的更改。

+   **添加轨迹**：这允许用户添加新轨迹。请注意，此项目仅在轨迹编辑模式下可用。

+   **编辑轨迹**：这允许用户编辑现有轨迹。只有当用户处于轨迹编辑模式时，此功能才可用。

+   **删除轨迹**：这允许用户删除轨迹。此功能仅在轨迹编辑模式下可用。

+   **获取信息**：这启用了获取信息地图工具。当用户点击一个轨迹时，此工具将显示该轨迹的属性，并允许用户更改这些属性。

+   **设置起点**：这允许用户为最短路径计算设置当前起点。

+   **设置终点**：此项目允许用户在地图上单击以设置最短路径计算的目标点。

+   **找到最短路径**：这将显示当前起始点和终点之间的最短可用路径。再次单击此项目将隐藏路径。

这让我们对我们的应用程序的外观和工作方式有了很好的了解。现在，让我们开始编写 ForestTrails 程序，通过实现应用程序及其主窗口的基本逻辑。

# 创建应用程序

我们的应用程序将是一个独立的 Python 程序，使用 PyQt 和 PyQGIS 库构建。以我们在第五章中实现的 Lex 应用程序为起点，*在外部应用程序中使用 QGIS*，让我们看看我们如何组织 ForestTrails 系统的源文件。我们将从以下基本结构开始：

![创建应用程序](img/00089.jpeg)

这与我们在 Lex 应用程序中使用的结构非常相似，所以其中大部分内容对你来说应该是熟悉的。主要区别在于我们使用两个子目录来存放额外的文件。让我们看看每个文件和目录将用于什么：

+   `constants.py`：这个模块将包含 ForestTrails 系统中使用的各种常量。

+   `data`：这是一个目录，我们将用它来存放我们的栅格底图以及包含我们轨迹的 SpatiaLite 数据库。

+   `forestTrails.py`：这是我们的应用程序的主程序。

+   `Makefile`：这个文件告诉 make 工具如何将 `resources.qrc` 文件编译成我们的应用程序可以使用的 `resources.py` 模块。

+   `mapTools.py`：这个模块实现了我们的各种地图工具。

+   `resources`：这是一个目录，我们将在这里放置各种图标和其他资源。由于我们有这么多图标文件，将这些文件放入子目录而不是让主目录充斥着这些文件是有意义的。

+   `resources.qrc`：这是我们的应用程序的资源描述文件。

+   `run_lin.sh`：这个 bash shell 脚本用于在 Linux 系统上运行我们的应用程序。

+   `run_mac.sh`：这个 bash shell 脚本用于在 Mac OS X 系统上运行我们的应用程序。

+   `run_win.bat`：这个批处理文件用于在 MS Windows 机器上运行我们的应用程序。

+   `ui_mainWindow.py`：这个 Python 模块定义了我们主窗口的用户界面。

## 布局应用程序

让我们一步一步地实现 ForestTrails 系统。创建一个目录来存放 ForestTrails 系统的源代码，然后在其中创建 `data` 和 `resources` 子目录。由于主目录中的许多文件都很直接，我们不妨直接创建以下文件：

+   `Makefile` 应该看起来像这样：

    ```py
    RESOURCE_FILES = resources.py

    default: compile

    compile: $(RESOURCE_FILES)

    %.py : %.qrc
      pyrcc4 -o $@ $<

    %.py : %.ui
      pyuic4 -o $@ $<

    clean:
      rm $(RESOURCE_FILES)
      rm *.pyc
    ```

    ### 提示

    注意，如果你的 `pyrcc4` 命令在非标准位置，你可能需要修改此文件，以便 `make` 可以找到它。

+   按照以下方式创建 `resources.qrc` 文件：

    ```py
    <RCC>
    <qresource>
    <file>resources/mActionZoomIn.png</file>
    <file>resources/mActionZoomOut.png</file>
    <file>resources/mActionPan.png</file>
    <file>resources/mActionEdit.svg</file>
    <file>resources/mActionAddTrack.svg</file>
    <file>resources/mActionEditTrack.png</file>
    <file>resources/mActionDeleteTrack.svg</file>
    <file>resources/mActionGetInfo.svg</file>
    <file>resources/mActionSetStartPoint.svg</file>
    <file>resources/mActionSetEndPoint.svg</file>
    <file>resources/mActionFindShortestPath.svg</file>
    </qresource>
    </RCC>
    ```

    注意，我们已经包含了将被用于我们的工具栏动作的各种图像文件。所有这些文件都在我们的 `resources` 子目录中。我们将在稍后查看如何获取这些图像文件。

+   `run-lin.sh` 文件应该看起来像这样：

    ```py
    #!/bin/sh
    export PYTHONPATH="/path/to/qgis/build/output/python/"
    export LD_LIBRARY_PATH="/path/to/qgis/build/output/lib/"
    export QGIS_PREFIX="/path/to/qgis/build/output/"
    python forestTrails.py
    ```

+   类似地，`run-mac.sh` 应该包含以下内容：

    ```py
    export PYTHONPATH="$PYTHONPATH:/Applications/QGIS.app/Contents/Resources/python"
    export DYLD_FRAMEWORK_PATH="/Applications/QGIS.app/Contents/Frameworks"
    export QGIS_PREFIX="/Applications/QGIS.app/Contents/Resources"
    python forestTrails.py
    ```

+   `run-win.bat` 文件应该包含：

    ```py
    SET OSGEO4W_ROOT=C:\OSGeo4W
    SET QGIS_PREFIX=%OSGEO4W_ROOT%\apps\qgis
    SET PATH=%PATH%;%QGIS_PREFIX%\bin
    SET PYTHONPATH=%QGIS_PREFIX%\python;%PYTHONPATH%
    python forestTrails.py
    ```

    ### 注意

    如果你的 QGIS 安装在一个非标准位置，你可能需要修改相应的脚本，以便可以找到所需的库。

由于 `resources.qrc` 文件导入了我们的各种工具栏图标并使它们可供应用程序使用，我们将想要设置这些图标文件。现在让我们来做这件事。

## 定义工具栏图标

我们总共需要为 11 个工具栏动作显示图标：

![定义工具栏图标](img/00090.jpeg)

您可以自由创建或下载这些工具栏动作的自定义图标，或者您可以使用本章提供的源代码中包含的图标文件。文件格式并不重要，只要在`resoures.qrc`文件中包含正确的后缀，并在`ui_mainWindow.py`中初始化工具栏动作时即可。

确保将这些文件放入`resources`子目录中，并运行`make`来构建`resources.py`模块，以便这些图标可供您的应用程序使用。

在完成这些基础工作后，我们就可以开始定义应用程序代码本身了。让我们从`constants.py`模块开始。

## `constants.py`模块

此模块将包含我们用来表示轨道属性值的各种常量；通过在同一个地方定义它们，我们确保属性值被一致地使用，我们不必记住确切的值。例如，轨道层的`type`属性可以有以下值：

+   `ROAD`

+   `WALKING`

+   `BIKE`

+   `HORSE`

而不是每次需要这些值时都硬编码它们，我们将定义这些值在`constants.py`模块中。创建此模块并将以下代码输入其中：

```py
TRACK_TYPE_ROAD    = "ROAD"
TRACK_TYPE_WALKING = "WALKING"
TRACK_TYPE_BIKE    = "BIKE"
TRACK_TYPE_HORSE   = "HORSE"

TRACK_DIRECTION_BOTH     = "BOTH"
TRACK_DIRECTION_FORWARD  = "FORWARD"
TRACK_DIRECTION_BACKWARD = "BACKWARD"

TRACK_STATUS_OPEN   = "OPEN"
TRACK_STATUS_CLOSED = "CLOSED"
```

我们将在继续的过程中添加更多常量，但这已经足够我们开始了。

## `forestTrails.py`模块

此模块定义了 ForestTrails 应用程序的主程序。它看起来与我们在第五章中定义的`lex.py`模块非常相似，即*在外部应用程序中使用 QGIS*。创建您的`forestTrails.py`文件，并将以下`import`语句输入其中：

```py
import os, os.path, sys

from qgis.core import *
from qgis.gui import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from ui_mainWindow import Ui_MainWindow

import resources
from constants import *
from mapTools import *
```

接下来，我们想在类中定义我们应用程序的主窗口，我们将称之为`ForestTrailsWindow`。这是应用程序代码的大部分将得到实现的地方；这个类将变得相当复杂，但我们将从简单开始，只定义窗口本身，并为所有工具栏动作定义空占位符方法。

让我们定义类本身和`__init__()`方法来初始化一个新窗口：

```py
class ForestTrailsWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setupUi(self)

        self.connect(self.actionQuit, SIGNAL("triggered()"),
                     self.quit)
        self.connect(self.actionZoomIn, SIGNAL("triggered()"),
                     self.zoomIn)
        self.connect(self.actionZoomOut, SIGNAL("triggered()"),
                     self.zoomOut)
        self.connect(self.actionPan, SIGNAL("triggered()"),
                     self.setPanMode)
        self.connect(self.actionEdit, SIGNAL("triggered()"),
                     self.setEditMode)
        self.connect(self.actionAddTrack, SIGNAL("triggered()"),
                     self.addTrack)
        self.connect(self.actionEditTrack, SIGNAL("triggered()"),
                     self.editTrack)
        self.connect(self.actionDeleteTrack,SIGNAL("triggered()"),
                     self.deleteTrack)
        self.connect(self.actionGetInfo, SIGNAL("triggered()"),
                     self.getInfo)
        self.connect(self.actionSetStartPoint,
                     SIGNAL("triggered()"),
                self.setStartPoint)
        self.connect(self.actionSetEndPoint,
                     SIGNAL("triggered()"),
                  self.setEndPoint)
        self.connect(self.actionFindShortestPath,
                     SIGNAL("triggered()"),
                     self.findShortestPath)

        self.mapCanvas = QgsMapCanvas()
        self.mapCanvas.useImageToRender(False)
        self.mapCanvas.setCanvasColor(Qt.white)
        self.mapCanvas.show()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mapCanvas)
        self.centralWidget.setLayout(layout)
```

这与 Lex 应用程序的`__init__()`方法非常相似；我们将在`ui_mainWindow.py`模块中定义`Ui_MainWindow`类来设置应用程序的用户界面。这就是所有那些`actionXXX`实例变量将被定义的地方。在我们的`__init__()`方法中，我们将这些动作连接到各种方法，当用户从工具栏或菜单栏选择动作时，这些方法将做出响应。

`__init__()`方法的其余部分只是设置地图画布并将其布局在窗口内。有了这个方法，我们现在可以定义所有那些动作处理方法。我们可以直接从`lex.py`借用其中两个：

```py
    def zoomIn(self):
        self.mapCanvas.zoomIn()

    def zoomOut(self):
        self.mapCanvas.zoomOut()
```

对于其余部分，我们将推迟实现它们，直到应用程序更加完整。为了允许我们的程序运行，我们将为剩余的动作处理程序设置空占位符方法：

```py
    def quit(self):
        pass

    def setPanMode(self):
        pass

    def setEditMode(self):
        pass

    def addTrack(self):
        pass

    def editTrack(self):
        pass

    def deleteTrack(self):
        pass

    def getInfo(self):
        pass

    def setStartingPoint(self):
        pass

    def setEndingPoint(self):
        pass

    def findShortestPath(self):
        pass
```

`forestTrails.py`模块的最后部分是`main()`函数，当程序运行时会被调用：

```py
def main():
    QgsApplication.setPrefixPath(os.environ['QGIS_PREFIX'], True)
    QgsApplication.initQgis()

    app = QApplication(sys.argv)

    window = ForestTrailsWindow()
    window.show()
    window.raise_()
    window.setPanMode()

    app.exec_()
    app.deleteLater()
    QgsApplication.exitQgis()

if __name__ == "__main__":
    main()
```

再次强调，这与我们在 Lex 应用程序中看到的代码几乎相同。

这完成了`forestTrails.py`模块的初始实现。我们的下一步是创建一个模块，用于存放我们所有的地图工具。

## mapTools.py 模块

我们在 Lex 应用程序中使用了`mapTools.py`来分别定义我们的各种地图工具，而不在主程序中定义。我们在这里也将这样做。不过，目前我们的`mapTools.py`模块几乎是空的：

```py
from qgis.core import *
from qgis.gui import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from constants import *
```

显然，随着我们开始实现各种地图工具，我们还将添加更多内容，但就目前而言，这已经足够了。

## ui_mainWindow.py 模块

这是我们需要为 ForestTrails 系统的初始实现定义的最后一个模块。与 Lex 应用程序一样，这个模块定义了一个`Ui_MainWindow`类，它实现了应用程序的用户界面，并为各种菜单和工具栏项定义了`QAction`对象。我们将首先导入我们的类需要的模块：

```py
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import resources
```

接下来，我们将定义`Ui_MainWindow`类和`setupUi()`方法，它将完成所有工作：

```py
class Ui_MainWindow(object):
    def setupUi(self, window):
```

`setupUi()`方法的第一部分设置了窗口的标题，创建了一个`centralWidget`实例变量来保存地图视图，并初始化应用程序的菜单和工具栏：

```py
        window.setWindowTitle("Forest Trails")

        self.centralWidget = QWidget(window)
        self.centralWidget.setMinimumSize(800, 400)
        window.setCentralWidget(self.centralWidget)

        self.menubar = window.menuBar()
        self.fileMenu = self.menubar.addMenu("File")
        self.mapMenu = self.menubar.addMenu("Map")
        self.editMenu = self.menubar.addMenu("Edit")
        self.toolsMenu = self.menubar.addMenu("Tools")

        self.toolBar = QToolBar(window)
        window.addToolBar(Qt.TopToolBarArea, self.toolBar)
```

接下来，我们想要定义各种工具栏和菜单项的`QAction`对象。对于每个动作，我们将定义动作的图标和键盘快捷键，并检查动作是否**可勾选**（即用户点击时保持选中状态）：

```py
        self.actionQuit = QAction("Quit", window)
        self.actionQuit.setShortcut(QKeySequence.Quit)

        icon = QIcon(":/resources/mActionZoomIn.png")
        self.actionZoomIn = QAction(icon, "Zoom In", window)
        self.actionZoomIn.setShortcut(QKeySequence.ZoomIn)

        icon = QIcon(":/resources/mActionZoomOut.png")
        self.actionZoomOut = QAction(icon, "Zoom Out", window)
        self.actionZoomOut.setShortcut(QKeySequence.ZoomOut)

        icon = QIcon(":/resources/mActionPan.png")
        self.actionPan = QAction(icon, "Pan", window)
        self.actionPan.setShortcut("Ctrl+1")
        self.actionPan.setCheckable(True)

        icon = QIcon(":/resources/mActionEdit.svg")
        self.actionEdit = QAction(icon, "Edit", window)
        self.actionEdit.setShortcut("Ctrl+2")
        self.actionEdit.setCheckable(True)

        icon = QIcon(":/resources/mActionAddTrack.svg")
        self.actionAddTrack = QAction(icon, "Add Track", window)
        self.actionAddTrack.setShortcut("Ctrl+A")
        self.actionAddTrack.setCheckable(True)

        icon = QIcon(":/resources/mActionEditTrack.png")
        self.actionEditTrack = QAction(icon, "Edit", window)
        self.actionEditTrack.setShortcut("Ctrl+E")
        self.actionEditTrack.setCheckable(True)

        icon = QIcon(":/resources/mActionDeleteTrack.svg")
        self.actionDeleteTrack = QAction(icon, "Delete", window)
        self.actionDeleteTrack.setShortcut("Ctrl+D")
        self.actionDeleteTrack.setCheckable(True)

        icon = QIcon(":/resources/mActionGetInfo.svg")
        self.actionGetInfo = QAction(icon, "Get Info", window)
        self.actionGetInfo.setShortcut("Ctrl+I")
        self.actionGetInfo.setCheckable(True)

        icon = QIcon(":/resources/mActionSetStartPoint.svg")
        self.actionSetStartPoint = QAction(
                icon, "Set Start Point", window)
        self.actionSetStartPoint.setCheckable(True)

        icon = QIcon(":/resources/mActionSetEndPoint.svg")
        self.actionSetEndPoint = QAction(
                icon, "Set End Point", window)
        self.actionSetEndPoint.setCheckable(True)

        icon = QIcon(":/resources/mActionFindShortestPath.svg")
        self.actionFindShortestPath = QAction(
                icon, "Find Shortest Path", window)
        self.actionFindShortestPath.setCheckable(True)
```

然后我们将各种动作添加到应用程序的菜单中：

```py
        self.fileMenu.addAction(self.actionQuit)

        self.mapMenu.addAction(self.actionZoomIn)
        self.mapMenu.addAction(self.actionZoomOut)
        self.mapMenu.addAction(self.actionPan)
        self.mapMenu.addAction(self.actionEdit)

        self.editMenu.addAction(self.actionAddTrack)
        self.editMenu.addAction(self.actionEditTrack)
        self.editMenu.addAction(self.actionDeleteTrack)
        self.editMenu.addAction(self.actionGetInfo)

        self.toolsMenu.addAction(self.actionSetStartPoint)
        self.toolsMenu.addAction(self.actionSetEndPoint)
        self.toolsMenu.addAction(self.actionFindShortestPath)
```

最后，我们将动作添加到工具栏中，并告诉窗口根据内容调整大小：

```py
        self.toolBar.addAction(self.actionZoomIn)
        self.toolBar.addAction(self.actionZoomOut)
        self.toolBar.addAction(self.actionPan)
        self.toolBar.addAction(self.actionEdit)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionAddTrack)
        self.toolBar.addAction(self.actionEditTrack)
        self.toolBar.addAction(self.actionDeleteTrack)
        self.toolBar.addAction(self.actionGetInfo)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSetStartPoint)
        self.toolBar.addAction(self.actionSetEndPoint)
        self.toolBar.addAction(self.actionFindShortestPath)

        window.resize(window.sizeHint())
```

这完成了`ui_mainWindow.py`模块的实现。我们现在有一个完整的小型应用程序，应该能够运行。让我们试试它。

## 运行应用程序

现在你已经输入了所有这些代码，是时候检查它是否工作。让我们尝试使用适当的启动脚本运行应用程序。打开一个终端或命令行窗口，导航到`forestTrails`目录，并运行相应的启动脚本。

如果一切顺利，你应该会看到应用程序的主窗口以及工具栏和菜单项：

![运行应用程序](img/00091.jpeg)

当然，主窗口的地图视图是空的，工具栏或菜单项还没有任何功能，但至少我们为我们的应用程序提供了一个工作的框架。我们的下一步是获取应用程序的基础地图，设置我们的地图层，然后开始实现各种工具栏和菜单栏项。

# 获取基础地图

为了继续本章的这一部分，您将需要访问 GDAL 命令行工具。GDAL 可能已经安装在您的计算机上，因为 QGIS 使用了它。如果您还没有安装 GDAL，请访问[www.gdal.org](http://www.gdal.org)并点击**下载**链接，将副本下载并安装到您的机器上。

编写地图应用的一个挑战是在上面显示您的地理空间数据的优质底图。在我们的案例中，我们希望底图显示森林的航空照片。我们将使用新西兰罗托鲁瓦的 Whakarewarewa 森林作为我们的 ForestTrails 应用。幸运的是，*新西兰土地信息网站*提供了合适的航空照片。

访问以下网页，该网页提供了新西兰丰盛湾的高分辨率航空照片：

[`data.linz.govt.nz/layer/1760-bay-of-plenty-025m-rural-aerial-photos-2011-2012/`](https://data.linz.govt.nz/layer/1760-bay-of-plenty-025m-rural-aerial-photos-2011-2012/)

我们想要下载一个覆盖 Whakarewarewa 森林的底图，该森林位于罗托鲁瓦市以南。在页面右侧的地图上，平移并缩放到以下地图区域：

![获取底图](img/00092.jpeg)

地图中心黑暗的圆形区域是罗托鲁阿湖。进一步放大并向下平移到罗托鲁阿以南的区域：

![获取底图](img/00093.jpeg)

这张地图显示了我们要下载的 Whakarewarewa 森林图像。接下来，点击右上角的**裁剪**工具 (![获取底图](img/00094.jpeg)) 并选择以下地图区域：

![获取底图](img/00095.jpeg)

在选择了适当的地图区域后，点击右上角的**“下载或订购”**链接。出现的窗口为您提供下载底图的选择。请确保您选择以下选项：

+   地图投影将为 NZGD2000

+   原始图像格式将为 TIFF，保持原始分辨率

    ### 注意

    您需要注册才能下载文件，但注册过程只需几秒钟，且不收费。

生成的下载文件大小约为 2.8 GB，略低于本站文件下载的 3 GB 限制。如果文件太大，您将不得不选择较小的区域进行下载。

下载文件后，您将得到一个包含多个 TIFF 格式栅格图像文件的 ZIP 存档。接下来，我们需要将这些图像合并成一个单独的`.tif`文件作为我们的底图。为此，我们将使用 GDAL 附带的`gdal_merge.py`命令：

```py
gdal_merge.py -o /dst/path/basemap.tif *.tif

```

选择`basemap.tif`文件的适当目的地（例如，通过将`/dst/path`替换为合理的位置，例如桌面路径）。如果当前目录未设置为包含下载的`.tif`文件的文件夹，您还需要在命令中指定源路径。

这个命令将需要一段时间来拼接各种图像，但结果应该是一个名为 `basemap.tif` 的单个大文件。这是一个包含您所选航空照片的 TIFF 格式栅格图像，并且地理参考到地球表面的适当部分。

不幸的是，我们无法直接使用此文件。要了解原因，请在下载的文件上运行`gdalinfo`命令：

```py
gdalinfo basemap.tif

```

此外，这告诉我们文件使用的是哪个坐标参考系统：

```py
    Coordinate System is:
    PROJCS["NZGD2000 / New Zealand Transverse Mercator 2000",
        GEOGCS["NZGD2000",
            DATUM["New_Zealand_Geodetic_Datum_2000",
                SPHEROID["GRS 1980",6378137,298.2572221010002,
                    AUTHORITY["EPSG","7019"]],
                AUTHORITY["EPSG","6167"]],
            PRIMEM["Greenwich",0],
            UNIT["degree",0.0174532925199433],
            AUTHORITY["EPSG","4167"]],
        ...
```

如您所见，下载的底图使用的是**新西兰横轴墨卡托 2000**坐标系。我们需要将其转换为 WGS84（地理纬度/经度坐标）坐标系，以便在 ForestTrails 程序中使用。为此，我们将使用`gdalwarp`命令，如下所示：

```py
 gdalwarp -t_srs EPSG:4326 basemap.tif basemap_wgs84.tif

```

如果您使用`gdalinfo`查看生成的图像，您会看到它已被转换为纬度/经度坐标系：

```py
    Coordinate System is:
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433],
        AUTHORITY["EPSG","4326"]]
```

### 注意

您可能会想知道为什么我们没有直接以 WGS84 坐标系下载文件。我们以原始 CRS 下载文件，因为这使我们能够更好地控制最终图像。自己重新投影图像也更容易看到图像在重新投影时发生了哪些变化。

到目前为止，一切顺利。然而，如果我们查看生成的图像，我们会看到另一个问题：

![获取底图](img/00096.jpeg)

从 NZGD2000 到 WGS84 的转换使底图略微旋转，因此地图的边界看起来不太好。现在，我们需要裁剪地图以去除不需要的边界。为此，我们将再次使用`gdal_warp`命令，这次带有目标范围：

```py
gdalwarp -te 176.241 -38.2333 176.325 -38.1557 basemap_wgs84.tif basemap_trimmed.tif

```

### 提示

如果您在下载底图时选择了略微不同的边界，您可能需要调整纬度/经度值。`gdalinfo`显示的角落坐标值将为您提供有关要使用哪些值的线索。

生成的文件是我们用于 ForestTrails 程序的理想栅格底图：

![获取底图](img/00097.jpeg)

将最终图像复制到您的`forestTrails/data`目录，并将其重命名为`basemap.tif`。

# 定义地图层

我们知道我们希望在应用程序中总共拥有五个地图层。底图层将显示我们刚刚下载的`basemap.tif`文件，而轨迹层将使用 SpatiaLite 数据库来存储和显示用户输入的轨迹数据。其余的地图层将显示内存中持有的临时特征。

让我们从在`forestTrails.py`模块中定义一个新的方法开始，以初始化我们将用于轨迹层的 SpatiaLite 数据库：

```py
    def setupDatabase(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        dbName = os.path.join(cur_dir, "data", "tracks.sqlite")
        if not os.path.exists(dbName):
            fields = QgsFields()
            fields.append(QgsField("id", QVariant.Int))
            fields.append(QgsField("type", QVariant.String))
            fields.append(QgsField("name", QVariant.String))
            fields.append(QgsField("direction", QVariant.String))
            fields.append(QgsField("status", QVariant.String))

            crs = QgsCoordinateReferenceSystem(4326,
                        QgsCoordinateReferenceSystem.EpsgCrsId)

            writer = QgsVectorFileWriter(dbName, 'utf-8', fields,
                                         QGis.WKBLineString,
                                         crs, 'SQLite',
                                         ["SPATIALITE=YES"])

            if writer.hasError() != QgsVectorFileWriter.NoError:
                print "Error creating tracks database!"

            del writer
```

如您所见，我们检查我们的`data`子目录中是否存在 SpatiaLite 数据库文件，并在必要时创建一个新的数据库。我们定义了将保存各种轨迹属性的各种字段，并使用`QgsVectorFileWriter`对象创建数据库。

你还需要修改 `main()` 函数以调用 `setupDatabase()` 方法。在调用 `window.raise_()` 之后添加以下行到这个函数中：

```py
    window.setupDatabase()
```

现在我们已经为轨迹层设置了数据库，我们可以定义我们的各种地图层。我们将创建一个名为 `setupMapLayers()` 的新方法来完成这个任务。让我们首先定义一个 `layers` 变量来保存各种地图层，并初始化我们的基础地图层：

```py
    def setupMapLayers(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        layers = []

        filename = os.path.join(cur_dir, "data", "basemap.tif")
        self.baseLayer = QgsRasterLayer(filename, "basemap")
        QgsMapLayerRegistry.instance().addMapLayer(self.baseLayer)
        layers.append(QgsMapCanvasLayer(self.baseLayer))
```

接下来，我们想要设置我们的 **tracks** 层。由于这个层存储在 SpatiaLite 数据库中，我们必须使用 `QgsDataSourceURI` 对象将数据库连接到地图层。以下代码展示了如何完成这个操作：

```py
        uri = QgsDataSourceURI()
        uri.setDatabase(os.path.join(cur_dir, "data",
 "tracks.sqlite"))
        uri.setDataSource('', 'tracks', 'GEOMETRY')

        self.trackLayer = QgsVectorLayer(uri.uri(), "Tracks",
                                         "spatialite")
        QgsMapLayerRegistry.instance().addMapLayer(
            self.trackLayer)
        layers.append(QgsMapCanvasLayer(self.trackLayer))
```

现在，我们可以设置一个基于内存的地图层来显示最短路径：

```py
        self.shortestPathLayer = QgsVectorLayer(
            "LineString?crs=EPSG:4326",
            "shortestPathLayer", "memory")
        QgsMapLayerRegistry.instance().addMapLayer(
            self.shortestPathLayer)
        layers.append(QgsMapCanvasLayer(self.shortestPathLayer))
```

我们在 第六章 中看到了如何创建基于内存的地图层，*掌握 QGIS Python API*，所以这里不应该有任何惊喜；我们只是在定义一个用于保存 LineString 几何的最短路径层。

接下来，我们想要设置另一个基于内存的地图层来显示用户的选定起点：

```py
        self.startPointLayer = QgsVectorLayer(
                                   "Point?crs=EPSG:4326",
                                   "startPointLayer", "memory")
        QgsMapLayerRegistry.instance().addMapLayer(
            self.startPointLayer)
        layers.append(QgsMapCanvasLayer(self.startPointLayer))
```

此外，我们还想为终点设置另一个地图层：

```py
        self.endPointLayer = QgsVectorLayer(
             "Point?crs=EPSG:4326",
             "endPointLayer", "memory")
        QgsMapLayerRegistry.instance().addMapLayer(
            self.endPointLayer)
        layers.append(QgsMapCanvasLayer(self.endPointLayer))
```

这完成了我们五个地图层的所有设置。`setupMapLayers()` 方法的最后一部分将这些各种层添加到地图画布上。请注意，因为我们按从后向前的顺序定义了地图层（换句话说，`layers` 中的第一个条目是底图，它应该出现在后面），在将它们添加到地图画布之前，我们必须反转这些层。以下是相关代码：

```py
        layers.reverse()
        self.mapCanvas.setLayerSet(layers)
        self.mapCanvas.setExtent(self.baseLayer.extent())
```

我们最后要做的就是从我们的 `main()` 函数中添加对 `setupMapLayers()` 的调用。在 `window.setupDatabase()` 行之后立即添加以下内容：

```py
window.setupMapLayers()
```

现在我们已经设置了地图层，我们可以再次运行我们的程序。目前还没有矢量数据，但底图应该是可见的，我们可以使用工具栏图标进行缩放：

![定义地图层](img/00098.jpeg)

# 定义地图渲染器

现在我们有了地图层，我们将想要设置适当的符号和渲染器来将矢量数据绘制到地图上。让我们首先定义一个名为 `setupRenderers()` 的方法，它为我们的各种地图层创建渲染器。我们的第一个渲染器将显示轨迹层，我们使用 `QgsRuleBasedRendererV2` 对象根据轨迹类型、轨迹是否开放以及是否为双向或只能单向使用来以不同的方式显示轨迹。以下是相关代码：

```py
    def setupRenderers(self):
        root_rule = QgsRuleBasedRendererV2.Rule(None)

        for track_type in (TRACK_TYPE_ROAD, TRACK_TYPE_WALKING,
                           TRACK_TYPE_BIKE, TRACK_TYPE_HORSE):
            if track_type == TRACK_TYPE_ROAD:
                width = ROAD_WIDTH
            else:
                width = TRAIL_WIDTH

            lineColor = "light gray"
            arrowColor = "dark gray"

            for track_status in (TRACK_STATUS_OPEN,TRACK_STATUS_CLOSED):
                for track_direction in (TRACK_DIRECTION_BOTH,
                                        TRACK_DIRECTION_FORWARD,
                                        TRACK_DIRECTION_BACKWARD):
                    symbol = self.createTrackSymbol(width,lineColor, arrowColor,track_status,track_direction)
                    expression = ("(type='%s') and " +
                                  "(status='%s') and " +
                                  "(direction='%s')") % (track_type,track_status,                            track_direction)

                    rule = QgsRuleBasedRendererV2.Rule(symbol,filterExp=expression)
                    root_rule.appendChild(rule)

        symbol = QgsLineSymbolV2.createSimple({'color' : "black"})
        rule = QgsRuleBasedRendererV2.Rule(symbol, elseRule=True)
        root_rule.appendChild(rule)

        renderer = QgsRuleBasedRendererV2(root_rule)
        self.trackLayer.setRendererV2(renderer)
```

如您所见，我们遍历所有可能的轨迹类型。根据轨迹类型，我们选择合适的线宽。我们还选择用于线条和箭头的颜色——目前，我们只是为每种轨迹类型使用相同的颜色。然后，我们遍历所有可能的状态和方向值，并调用名为`createTrackSymbol()`的辅助方法来为该轨迹类型、状态和方向创建合适的符号。然后，我们创建一个`QgsRuleBasedRendererV2.Rule`对象，该对象使用该符号为给定类型、状态和方向的轨迹。最后，我们为渲染器定义一个“else”规则，如果轨迹没有预期的属性值，则将其显示为简单的黑色线条。

我们剩余的地图层将使用简单的线条或标记符号来显示最短路径以及起点和终点。以下是`setupRenderers()`方法的其余部分，它定义了这些地图渲染器：

```py
        symbol = QgsLineSymbolV2.createSimple({'color' : "blue"})
        symbol.setWidth(ROAD_WIDTH)
        symbol.setOutputUnit(QgsSymbolV2.MapUnit)
        renderer = QgsSingleSymbolRendererV2(symbol)
        self.shortestPathLayer.setRendererV2(renderer)

        symbol = QgsMarkerSymbolV2.createSimple(
                            {'color' : "green"})
        symbol.setSize(POINT_SIZE)
        symbol.setOutputUnit(QgsSymbolV2.MapUnit)
        renderer = QgsSingleSymbolRendererV2(symbol)
        self.startPointLayer.setRendererV2(renderer)

        symbol = QgsMarkerSymbolV2.createSimple({'color' : "red"})
        symbol.setSize(POINT_SIZE)
        symbol.setOutputUnit(QgsSymbolV2.MapUnit)
        renderer = QgsSingleSymbolRendererV2(symbol)
        self.endPointLayer.setRendererV2(renderer)
```

现在我们已经定义了`setupRenderers()`方法本身，让我们修改我们的`main()`函数来调用它。在调用`setupMapLayers()`之后立即添加以下行：

```py
window.setupRenderers()
```

为了完成我们的地图渲染器的实现，我们还需要做一些其他的事情。首先，我们需要定义我们用来设置轨迹渲染器的`createTrackSymbol()`辅助方法。将以下内容添加到您的`ForestTrailsWindow`类中：

```py
    def createTrackSymbol(self, width, lineColor, arrowColor,
                          status, direction):
        symbol = QgsLineSymbolV2.createSimple({})
        symbol.deleteSymbolLayer(0) # Remove default symbol layer.

        symbolLayer = QgsSimpleLineSymbolLayerV2()
        symbolLayer.setWidth(width)
        symbolLayer.setWidthUnit(QgsSymbolV2.MapUnit)
        symbolLayer.setColor(QColor(lineColor))
        if status == TRACK_STATUS_CLOSED:
            symbolLayer.setPenStyle(Qt.DotLine)
        symbol.appendSymbolLayer(symbolLayer)

        if direction == TRACK_DIRECTION_FORWARD:
            registry = QgsSymbolLayerV2Registry.instance()
            markerLineMetadata = registry.symbolLayerMetadata(
                "MarkerLine")
            markerMetadata     = registry.symbolLayerMetadata(
                "SimpleMarker")

            symbolLayer = markerLineMetadata.createSymbolLayer(
                                {'width': '0.26',
                                 'color': arrowColor,
                                 'rotate': '1',
                                 'placement': 'interval',
                                 'interval' : '20',
                                 'offset': '0'})
            subSymbol = symbolLayer.subSymbol()
            subSymbol.deleteSymbolLayer(0)
            triangle = markerMetadata.createSymbolLayer(
                                {'name': 'filled_arrowhead',
                                 'color': arrowColor,
                                 'color_border': arrowColor,
                                 'offset': '0,0',
                                 'size': '3',
                                 'outline_width': '0.5',
                                 'output_unit': 'mapunit',
                                 'angle': '0'})
            subSymbol.appendSymbolLayer(triangle)

            symbol.appendSymbolLayer(symbolLayer)
        elif direction == TRACK_DIRECTION_BACKWARD:
            registry = QgsSymbolLayerV2Registry.instance()
            markerLineMetadata = registry.symbolLayerMetadata(
                "MarkerLine")
            markerMetadata     = registry.symbolLayerMetadata(
                "SimpleMarker")

            symbolLayer = markerLineMetadata.createSymbolLayer(
                                {'width': '0.26',
                                 'color': arrowColor,
                                 'rotate': '1',
                                 'placement': 'interval',
                                 'interval' : '20',
                                 'offset': '0'})
            subSymbol = symbolLayer.subSymbol()
            subSymbol.deleteSymbolLayer(0)
            triangle = markerMetadata.createSymbolLayer(
                                {'name': 'filled_arrowhead',
                                 'color': arrowColor,
                                 'color_border': arrowColor,
                                 'offset': '0,0',
                                 'size': '3',
                                 'outline_width': '0.5',
                                 'output_unit': 'mapunit',
                                 'angle': '180'})
            subSymbol.appendSymbolLayer(triangle)

            symbol.appendSymbolLayer(symbolLayer)

        return symbol
```

这个方法的复杂部分是绘制箭头到轨迹上以指示轨迹方向的代码。除此之外，我们只是使用指定的颜色和宽度绘制线条来表示轨迹，如果轨迹是闭合的，我们将其绘制为虚线。

我们在这里的最终任务是向我们的`constants.py`模块添加一些条目来表示我们的渲染器使用的各种大小和线宽。将以下内容添加到该模块的末尾：

```py
ROAD_WIDTH  = 0.0001
TRAIL_WIDTH = 0.00003
POINT_SIZE  = 0.0004
```

所有这些值都在地图单位中。

不幸的是，我们目前看不到这些渲染器被使用，因为我们还没有任何矢量要素来显示，但我们需要现在实现它们，以便我们的代码在需要时能够工作。我们将在下一章中看到这些渲染器的实际效果，当用户开始添加轨迹并在地图上选择起点和终点时。

# 平移工具

为了让用户在地图上移动，我们将使用我们在早期章节中实现的`PanTool`类。将以下类定义添加到`mapTools.py`模块中：

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

在我们的`forestTrails.py`模块中，添加以下新方法：

```py
    def setupMapTools(self):
        self.panTool = PanTool(self.mapCanvas)
        self.panTool.setAction(self.actionPan)
```

此方法将初始化我们的应用程序将使用的各种地图工具；我们将随着进展添加到这个方法中。现在，在调用`window.setupRenderers()`之后，向您的`main()`函数中添加以下内容：

```py
    window.setupMapTools()
```

我们现在可以用真实的东西替换我们的`setPanMode()`的模拟实现：

```py
    def setPanMode(self):
        self.mapCanvas.setMapTool(self.panTool)
```

如果您现在运行程序，您会看到用户现在可以放大和缩小，并使用平移工具在基本地图上移动。

# 实现轨迹编辑模式

本章的最后一个任务是实现轨道编辑模式。我们在上一章学习了如何为地图层打开编辑模式，然后使用各种地图工具让用户添加、编辑和删除功能。我们将在第九章，*完成 ForestTrails 应用程序*中开始实现实际的地图工具，但现在，让我们定义我们的轨道编辑模式本身。

`setEditMode()`方法用于进入和退出轨道编辑模式。用这个新实现替换你之前定义的占位符方法：

```py
    def setEditMode(self):
        if self.editing:
            if self.modified:
                reply = QMessageBox.question(self, "Confirm",
                                             "Save Changes?",
                                             QMessageBox.Yes |
                                             QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.trackLayer.commitChanges()
                else:
                    self.trackLayer.rollBack()
            else:
                self.trackLayer.commitChanges()
            self.trackLayer.triggerRepaint()
            self.editing = False
            self.setPanMode()
        else:
            self.trackLayer.startEditing()
            self.trackLayer.triggerRepaint()
            self.editing  = True
            self.modified = False
            self.setPanMode()
        self.adjustActions()
```

如果用户目前正在编辑轨道并已进行了某些更改，我们将询问用户他们是否想要保存更改，然后提交更改或撤销更改。如果没有进行任何更改，我们将撤销（关闭矢量层的编辑模式）并切换回平移模式。

我们在这里使用了一些实例变量来监控轨道编辑的状态：`self.editing`将在我们正在编辑轨道时设置为`True`，而`self.modified`将在用户在轨道层中更改了任何内容时设置为`True`。我们必须在我们的`ForestTrailsWindow.__init__()`方法中添加以下内容来初始化这两个实例变量：

```py
        self.editing  = False
        self.modified= False
```

另有一个我们之前没有见过的方法：`adjustActions()`。这个方法将根据应用程序的当前状态启用/禁用和检查/取消选中各种操作：例如，当我们进入轨道编辑模式时，我们的`adjustActions()`方法将启用添加、编辑和删除工具，当用户离开轨道编辑模式时，这些工具将再次被禁用。

我们目前无法实现所有的`adjustActions()`，因为我们还没有定义应用程序将使用的各种地图工具。现在，我们将编写这个方法的前半部分：

```py
    def adjustActions(self):
       if self.editing:
            self.actionAddTrack.setEnabled(True)
            self.actionEditTrack.setEnabled(True)
            self.actionDeleteTrack.setEnabled(True)
            self.actionGetInfo.setEnabled(True)
            self.actionSetStartPoint.setEnabled(False)
            self.actionSetEndPoint.setEnabled(False)
            self.actionFindShortestPath.setEnabled(False)
        else:
            self.actionAddTrack.setEnabled(False)
            self.actionEditTrack.setEnabled(False)
            self.actionDeleteTrack.setEnabled(False)
            self.actionGetInfo.setEnabled(False)
            self.actionSetStartPoint.setEnabled(True)
            self.actionSetEndPoint.setEnabled(True)
            self.actionFindShortestPath.setEnabled(True)
```

我们还需要在调用`setPanMode()`之后在我们的`main()`函数中添加对`adjustActions()`的调用：

```py
    window.adjustActions()
```

实现了轨道编辑模式后，用户可以点击**编辑**工具栏图标进入轨道编辑模式，再次点击它以退出该模式。当然，我们目前还不能进行任何更改，但代码本身已经就位。

我们还想在我们的应用程序中添加一个功能；如果用户对轨道层进行了某些更改然后尝试退出应用程序，我们希望给用户一个保存更改的机会。为此，我们将实现一个`quit()`方法，并将其链接到`actionQuit`操作：

```py
    def quit(self):
        if self.editing and self.modified:
            reply = QMessageBox.question(self, "Confirm",
                                         "Save Changes?",
                                         QMessageBox.Yes |
                                         QMessageBox.No |
                                         QMessageBox.Cancel,
                                         QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.curEditedLayer.commitChanges()
            elif reply == QMessageBox.No:
                self.curEditedLayer.rollBack()

            if reply != QMessageBox.Cancel:
                qApp.quit()
        else:
            qApp.quit()
```

这与`setEditMode()`方法中允许用户退出轨道编辑模式的部分非常相似，只不过我们在最后调用`qApp.quit()`来退出应用程序。我们还有一个方法需要定义，它拦截关闭窗口的尝试并调用`self.quit()`。这会在用户在编辑时关闭窗口时提示用户保存他们的更改。以下是此方法的定义：

```py
    def closeEvent(self, event):
        self.quit()
```

# 摘要

在本章中，我们设计和开始实施了一个完整的映射应用程序，用于维护休闲森林内轨迹和道路的地图。我们实现了应用程序本身，定义了我们的地图层，为我们的应用程序获取了高分辨率的基础地图，并实现了缩放、平移以及编辑轨迹层所需的代码。

在下一章中，我们将通过实现地图工具来完善我们的 ForestTrails 系统的实施，使用户能够添加、编辑和删除轨迹。我们还将实现编辑轨迹属性和查找两点之间最短可用路径的代码。
