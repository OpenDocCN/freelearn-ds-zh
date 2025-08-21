# 第五章：Python 和 PixieDust 最佳实践与高级概念

> “我们信仰上帝，其他的都带上数据。”

– *W. Edwards Deming*

本书的剩余章节中，我们将深入探讨行业用例的架构，包括示范数据管道的实现，广泛应用我们到目前为止学到的技术。在开始查看代码之前，让我们通过一些最佳实践和高级 PixieDust 概念来完善我们的工具箱，这些将有助于我们实现示例应用：

+   使用`@captureOutput`装饰器调用第三方 Python 库

+   提高 PixieApp 的模块化和代码复用性

+   PixieDust 对流数据的支持

+   使用 PixieApp 事件添加仪表盘钻取功能

+   使用自定义显示渲染器扩展 PixieDust

+   调试：

    +   使用 pdb 调试在 Jupyter Notebook 中运行的逐行 Python 代码

    +   使用 PixieDebugger 进行可视化调试

    +   使用 PixieDust 日志框架排查问题

    +   客户端 JavaScript 调试技巧

+   在 Python Notebook 中运行 Node.js

# 使用`@captureOutput`装饰器集成第三方 Python 库的输出

假设你希望将自己的 PixieApp 在已经使用一段时间的第三方库中复用，以执行某个任务，例如，使用 scikit-learn 机器学习库([`scikit-learn.org`](http://scikit-learn.org))进行集群计算并将其作为图形显示。问题是，大多数情况下，你调用的是一个高级方法，它并不会返回数据，而是直接在单元格输出区域绘制某些内容，比如图表或报告表格。从 PixieApp 路由调用此方法将不起作用，因为路由的合同要求返回一个 HTML 片段字符串，该字符串将由框架处理。在这种情况下，该方法很可能没有返回任何内容，因为它将结果直接写入单元格输出区域。解决方案是在路由方法中使用`@captureOutput`装饰器——这是 PixieApp 框架的一部分。

## 使用@captureOutput 创建词云图像

为了更好地说明前面描述的`@captureOutput`场景，让我们以一个具体示例为例，在这个示例中，我们想要构建一个 PixieApp，使用`wordcloud` Python 库([`pypi.python.org/pypi/wordcloud`](https://pypi.python.org/pypi/wordcloud))从用户通过 URL 提供的文本文件生成词云图像。

我们首先通过在自己的单元格中运行以下命令来安装`wordcloud`库：

```py
!pip install wordcloud

```

### 注意

**注意**：确保在安装完`wordcloud`库后重新启动内核。

PixieApp 的代码如下所示：

```py
from pixiedust.display.app import *
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@PixieApp
class WordCloudApp():
    @route()
    def main_screen(self):
        return """
        <div style="text-align:center">
            <label>Enter a url: </label>
            <input type="text" size="80" id="url{{prefix}}">
            <button type="submit"
                pd_options="url=$val(url{{prefix}})"
                pd_target="wordcloud{{prefix}}">
                Go
            </button>
        </div>
        <center><div id="wordcloud{{prefix}}"></div></center>
        """

    @route(url="*")
    @captureOutput
    def generate_word_cloud(self, url):
        text = requests.get(url).text
        plt.axis("off")
        plt.imshow(
            WordCloud(max_font_size=40).generate(text),
            interpolation='bilinear'
        )

app = WordCloudApp()
app.run()
```

### 注意

你可以在这里找到代码：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode1.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode1.py)

注意，通过简单地在 `generate_word_cloud` 路由上添加 `@captureOutput` 装饰器，我们不再需要返回 HTML 片段字符串。我们可以直接调用 Matplotlib 的 `imshow()` 函数，将图像发送到系统输出。PixieApp 框架将负责捕获输出，并将其打包为 HTML 片段字符串，然后注入到正确的 div 占位符中。结果如下：

### 注意

**注意**：我们使用以下来自 GitHub 上 `wordcloud` 仓库的输入 URL：

[`github.com/amueller/word_cloud/blob/master/examples/constitution.txt`](https://github.com/amueller/word_cloud/blob/master/examples/constitution.txt)

### 注意

另一个可以使用的好链接是：

[`raw.githubusercontent.com/amueller/word_cloud/master/examples/a_new_hope.txt`](https://raw.githubusercontent.com/amueller/word_cloud/master/examples/a_new_hope.txt)

![使用 @captureOutput 创建词云图像](img/B09699_05_01.jpg)

简单的 PixieApp，生成来自文本的词云

任何直接绘制到单元格输出的函数都可以与 `@captureOutput` 装饰器一起使用。例如，你可以使用 Matplotlib 的 `show()` 方法或 IPython 的 `display()` 方法与 HTML 或 JavaScript 类一起使用。你甚至可以使用 `display_markdown()` 方法，通过 Markdown 标记语言输出富文本，如下代码所示：

```py
from pixiedust.display.app import *
from IPython.display import display_markdown

@PixieApp
class TestMarkdown():
    @route()
    @captureOutput
    def main_screen(self):
        display_markdown("""
# Main Header:
## Secondary Header with bullet
1\. item1
2\. item2
3\. item3

Showing image of the PixieDust logo
![alt text](https://github.com/pixiedust/pixiedust/raw/master/docs/_static/PixieDust%202C%20\(256x256\).png "PixieDust Logo")
    """, raw=True)

TestMarkdown().run()
```

这将产生以下结果：

![使用 @captureOutput 创建词云图像](img/B09699_05_02.jpg)

PixieApp 使用 @captureOutput 与 Markdown

# 增加模块化和代码重用性

将你的应用程序拆分为较小的、自包含的组件始终是一种良好的开发实践，因为这样可以使代码更具可重用性，并且更易于维护。PixieApp 框架提供了两种创建和运行可重用组件的方法：

+   动态调用其他 PixieApp 使用 `pd_app` 属性

+   将应用程序的一部分打包为可重用的小部件

使用 `pd_app` 属性，你可以通过完全限定的类名动态调用另一个 PixieApp（从现在开始我们称之为子 PixieApp）。子 PixieApp 的输出将放置在宿主 HTML 元素（通常是一个 div 元素）中，或者通过使用 `runInDialog=true` 选项放入对话框中。你还可以使用 `pd_options` 属性初始化子 PixieApp，在这种情况下，框架将调用相应的路由。

为了更好地理解 `pd_app` 的工作原理，让我们通过将生成 `WordCloud` 图像的代码重构为其自己的 PixieApp，称为 `WCChildApp`，来重写我们的 `WordCloud` 应用程序。

以下代码实现了 `WCChildApp` 作为常规 PixieApp，但请注意，它不包含默认路由。它只有一个名为 `generate_word_cloud` 的路由，应该由另一个 PixieApp 使用 `url` 参数来调用：

```py
from pixiedust.display.app import *
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt

@PixieApp
class WCChildApp():
    @route(url='*')
    @captureOutput
    def generate_word_cloud(self, url):
        text = requests.get(url).text
        plt.axis("off")
        plt.imshow(
            WordCloud(max_font_size=40).generate(text),
            interpolation='bilinear'
        )
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode2.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode2.py)

现在，我们可以构建主 PixieApp，当用户指定 URL 后点击**Go**按钮时，它将调用`WCChildApp`：

```py
@PixieApp
class WordCloudApp():
    @route()
    def main_screen(self):
        return """
        <div style="text-align:center">
            <label>Enter a url: </label>
            <input type="text" size="80" id="url{{prefix}}">
            <button type="submit"
                pd_options="url=$val(url{{prefix}})"
                pd_app="WCChildApp"
                pd_target="wordcloud{{prefix}}">
                Go
            </button>
        </div>
        <center><div id="wordcloud{{prefix}}"></div></center>
        """

app = WordCloudApp()
app.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode3.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode3.py)

在上述代码中，`Go`按钮具有以下属性：

+   `pd_app="WCChildApp"`：使用子 PixieApp 的类名。请注意，如果你的子 PixieApp 位于一个导入的 Python 模块中，那么你需要使用完整限定名。

+   `pd_options="url=$val(url{{prefix}})"`：将用户输入的 URL 作为子 PixieApp 的初始化选项进行存储。

+   `pd_target="wordcloud{{prefix}}"`：告诉 PixieDust 将子 PixieApp 的输出放置在 ID 为`wordcloud{{prefix}}`的 div 中。

`pd_app`属性是通过封装组件的逻辑和展示来模块化代码的强大方式。`pd_widget`属性提供了另一种实现类似结果的方式，不过这次组件不是由外部调用，而是通过继承来调用。

每种方法都有其优缺点：

+   `pd_widget`技术作为一个路由实现，肯定比`pd_app`更加轻量化，因为`pd_app`需要创建一个全新的 PixieApp 实例。请注意，`pd_widget`和`pd_app`（通过`parent_pixieapp`变量）都可以访问宿主应用程序中的所有变量。

+   `pd_app`属性提供了更清晰的组件分离，并且比小部件具有更多灵活性。例如，你可以有一个按钮，根据某些用户选择动态调用多个 PixieApps。

    ### 注意

    **注意**：正如我们将在本章后面看到的，这实际上就是 PixieDust 显示选项对话框时使用的方式。

如果你发现自己需要在 PixieApp 中有多个相同组件的副本，请问问自己该组件是否需要将其状态保持在类变量中。如果是这样，最好使用`pd_app`，如果不是，那么使用`pd_widget`也可以。

## 使用`pd_widget`创建小部件

创建小部件可以按照以下步骤进行：

1.  创建一个 PixieApp 类，该类包含一个带有特殊参数`widget`的路由

1.  使主类继承自 PixieApp 小部件

1.  使用`pd_widget`属性在 div 元素中调用小部件

再次举例说明，让我们用小部件重写`WordCloud`应用程序：

```py
from pixiedust.display.app import *
import requests
from word cloud import WordCloud
import matplotlib.pyplot as plt

@PixieApp
class WCChildApp():
    @route(widget='wordcloud')
    @captureOutput
    def generate_word_cloud(self):
        text = requests.get(self.url).text if self.url else ""
        plt.axis("off")
        plt.imshow(
            WordCloud(max_font_size=40).generate(text),
            interpolation='bilinear'
        )
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode4.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode4.py)

注意前面的代码中，`url` 现在被引用为一个类变量，因为我们假设基类会提供它。代码必须测试 `url` 是否为 `None`，因为在启动时，它的值正是 `None`。我们之所以这样实现，是因为 `pd_widget` 是一个无法轻松动态生成的属性（你需要使用一种二级途径来生成带有 `pd_widget` 属性的 div 片段）。

现在主要的 PixieApp 类看起来是这样的：

```py
@PixieApp
class WordCloudApp(WCChildApp):
    @route()
    def main_screen(self):
        self.url=None
        return """
        <div style="text-align:center">
            <label>Enter a url: </label>
            <input type="text" size="80" id="url{{prefix}}">
            <button type="submit"
                pd_script="self.url = '$val(url{{prefix}})'"
                pd_refresh="wordcloud{{prefix}}">
                Go
            </button>
        </div>
        <center><div pd_widget="wordcloud" id="wordcloud{{prefix}}"></div></center>
        """

app = WordCloudApp()
app.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode5.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode5.py)

包含 `pd_widget` 属性的 div 在启动时被渲染出来，但由于 `url` 仍然是 `None`，因此实际上并没有生成词云。`Go` 按钮有一个 `pd_script` 属性，该属性将 `self.url` 设置为用户提供的值。它还有一个 `pd_refresh` 属性，该属性设置为 `pd_widget` div，这将再次调用 `wordcloud` 小部件，但这一次 URL 会被初始化为正确的值。

在这一节中，我们已经看到了两种将代码模块化以便重用的方法，以及它们各自的优缺点。我强烈建议你自己动手实验这些代码，感受一下在何时使用每种技巧。如果你觉得这部分内容还是有些模糊，不用担心；希望在接下来的章节中使用这些技巧时，它会变得更加清晰。

在下一节中，我们将转向并研究 PixieDust 中的流式数据支持。

## PixieDust 对流式数据的支持

随着**物联网**（**Internet of Things**）设备的兴起，能够分析和可视化实时数据流变得越来越重要。例如，你可以有像温度计这样的传感器，或者像心脏起搏器这样的便携式医疗设备，持续地将数据流传输到像 Kafka 这样的流式服务中。PixieDust 通过提供简单的集成 API，简化了在 Jupyter Notebook 中与实时数据的交互，使得 `PixieApp` 和 `display()` 框架的集成更加便捷。

在可视化层面，PixieDust 使用 Bokeh ([`bokeh.pydata.org`](https://bokeh.pydata.org)) 支持高效的数据源更新，将流式数据绘制到实时图表中（请注意，目前仅支持折线图和散点图，但未来会添加更多图表类型）。`display()` 框架还支持使用 Mapbox 渲染引擎进行流式数据的地理空间可视化。

要启用流式可视化，你需要使用一个继承自 `StreamingDataAdapter` 的类，这是 PixieDust API 中的一个抽象类。这个类充当了流式数据源和可视化框架之间的通用桥梁。

### 注意

**注意**：我建议你花些时间查看这里的 `StreamingDataAdapter` 代码：

[`github.com/pixiedust/pixiedust/blob/0c536b45c9af681a4da160170d38879298aa87cb/pixiedust/display/streaming/__init__.py`](https://github.com/pixiedust/pixiedust/blob/0c536b45c9af681a4da160170d38879298aa87cb/pixiedust/display/streaming/__init__.py)

以下图表展示了`StreamingDataAdapter`数据结构如何融入`display()`框架：

![PixieDust 支持流数据](img/B09699_05_03.jpg)

StreamingDataAdapter 架构

在实现`StreamingDataAdapter`的子类时，必须重写基类提供的`doGetNextData()`方法，该方法会被重复调用以获取新数据并更新可视化。你还可以选择性地重写`getMetadata()`方法，将上下文传递给渲染引擎（稍后我们将使用此方法配置 Mapbox 渲染）。

`doGetNextData()`的抽象实现如下：

```py
@abstractmethod
def doGetNextData(self):
    """Return the next batch of data from the underlying stream.
    Accepted return values are:
    1\. (x,y): tuple of list/numpy arrays representing the x and y axis
    2\. pandas dataframe
    3\. y: list/numpy array representing the y axis. In this case, the x axis is automatically created
    4\. pandas serie: similar to #3
    5\. json
    6\. geojson
    7\. url with supported payload (json/geojson)
    """
    Pass
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode6.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode6.py)

上面的文档字符串解释了`doGetNextData()`方法允许返回的不同类型的数据。

作为一个示例，我们希望可视化一个虚构无人机在地图上实时地绕地球游荡的位置。其当前位置由一个 REST 服务提供，网址为：[`wanderdrone.appspot.com`](https://wanderdrone.appspot.com)。

负载使用 GeoJSON（[`geojson.org`](http://geojson.org)），例如：

```py
{
    "geometry": {
        "type": "Point",
        "coordinates": [
            -93.824908715741202, 10.875051131034805
        ]
    },
    "type": "Feature",
    "properties": {}
}
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode7.json`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode7.json)

为了实时渲染我们的无人机位置，我们创建了一个`DroneStreamingAdapter`类，继承自`StreamingDataAdapter`，并在`doGetNextData()`方法中返回无人机位置服务的 URL，如下代码所示：

```py
from pixiedust.display.streaming import *

class DroneStreamingAdapter(StreamingDataAdapter):
    def getMetadata(self):
        iconImage = "rocket-15"
        return {
            "layout": {"icon-image": iconImage, "icon-size": 1.5},
            "type": "symbol"
        }
    def doGetNextData(self):
        return "https://wanderdrone.appspot.com/"
adapter = DroneStreamingAdapter()
display(adapter)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode8.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode8.py)

在`getMetadata()`方法中，我们返回了特定于 Mapbox 的样式属性（如文档中所述：[`www.mapbox.com/mapbox-gl-js/style-spec`](https://www.mapbox.com/mapbox-gl-js/style-spec)），该样式使用火箭 Maki 图标（[`www.mapbox.com/maki-icons`](https://www.mapbox.com/maki-icons)）作为无人机的符号。

通过几行代码，我们能够创建一个实时的无人机位置地理空间可视化，结果如下：

![PixieDust 支持流数据](img/B09699_05_04.jpg)

无人机的实时地理空间映射

### 注意

你可以在 PixieDust 仓库的以下位置找到此示例的完整笔记本：

[`github.com/pixiedust/pixiedust/blob/master/notebook/pixieapp-streaming/Mapbox%20Streaming.ipynb`](https://github.com/pixiedust/pixiedust/blob/master/notebook/pixieapp-streaming/Mapbox%20Streaming.ipynb)

### 将流媒体功能添加到您的 PixieApp

在下一个示例中，我们展示如何使用 PixieDust 提供的开箱即用的 `MessageHubStreamingApp` PixieApp 来可视化来自 Apache Kafka 数据源的流数据：[`github.com/pixiedust/pixiedust/blob/master/pixiedust/apps/messageHub/messageHubApp.py`](https://github.com/pixiedust/pixiedust/blob/master/pixiedust/apps/messageHub/messageHubApp.py)。

### 注意

**注意**：`MessageHubStreamingApp` 与 IBM Cloud Kafka 服务 Message Hub（[`console.bluemix.net/docs/services/MessageHub/index.html#messagehub`](https://console.bluemix.net/docs/services/MessageHub/index.html#messagehub)）一起使用，但它可以很容易地适配任何其他 Kafka 服务。

如果您不熟悉 Apache Kafka，不用担心，我们将在第七章中介绍相关内容，*分析研究：使用 Twitter 情感分析的 NLP 和大数据*。

该 PixieApp 允许用户选择与服务实例关联的 Kafka 主题，并实时显示事件。假设所选主题的事件有效负载使用 JSON 格式，它会展示从事件数据采样推断出的模式。用户随后可以选择一个特定字段（必须是数值型），并显示该字段随时间变化的平均值的实时图表。

![将流媒体功能添加到您的 PixieApp](img/B09699_05_05.jpg)

流媒体数据的实时可视化

提供流媒体功能的关键 PixieApp 属性是 `pd_refresh_rate,` 它在指定的间隔执行特定的内核请求（拉取模型）。在前面的应用中，我们使用它来更新实时图表，如下所示，由 `showChart` 路由返回的 HTML 片段：

```py
    @route(topic="*",streampreview="*",schemaX="*")
    def showChart(self, schemaX):
        self.schemaX = schemaX
        self.avgChannelData = self.streamingData.getStreamingChannel(self.computeAverages)
        return """
<div class="well" style="text-align:center">
    <div style="font-size:x-large">Real-time chart for {{this.schemaX}}(average).</div>
</div>

<div pd_refresh_rate="1000" pd_entity="avgChannelData"></div>
        """
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode9.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode9.py)

上述 div 通过 `pd_entity` 属性与 `avgChannelData` 实体绑定，并负责创建每秒更新的实时图表（*pd_refresh_rate=1000 ms*）。反过来，`avgChannelData` 实体通过调用 `getStreamingChannel(),` 创建，并传递给 `self`。`computeAverage` 函数负责更新所有流数据的平均值。需要注意的是，`avgChannelData` 是一个继承自 `StreamingDataAdapter` 的类，因此可以传递给 `display()` 框架，用于构建实时图表。

最后一步是让 PixieApp 返回`displayHandler`，这是`display()`框架所需要的。可以通过如下方式重写`newDisplayHandler()`方法来实现：

```py
def newDisplayHandler(self, options, entity):
    if self.streamingDisplay is None:
        self.streamingDisplay = LineChartStreamingDisplay(options, entity)
    else:
        self.streamingDisplay.options = options
    return self.streamingDisplay
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode10.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode10.py)

在前面的代码中，我们使用它创建了一个由 PixieDust 提供的`LineChartStreamingDisplay`实例，该实例位于`pixiedust.display.streaming.bokeh`包中（[`github.com/pixiedust/pixiedust/blob/master/pixiedust/display/streaming/bokeh/lineChartStreamingDisplay.py`](https://github.com/pixiedust/pixiedust/blob/master/pixiedust/display/streaming/bokeh/lineChartStreamingDisplay.py)），并传入了`avgChannelData`实体。

如果你想看到这个应用的实际效果，你需要在 IBM Cloud 上创建一个消息中心服务实例（[`console.bluemix.net/catalog/services/message-hub`](https://console.bluemix.net/catalog/services/message-hub)），并使用其凭证在 Notebook 中调用此 PixieApp，代码如下：

```py
from pixiedust.apps.messageHub import *
MessageHubStreamingApp().run(
    credentials={
        "username": "XXXX",
        "password": "XXXX",
        "api_key" : "XXXX",
        "prod": True
    }
)
```

如果你有兴趣了解更多关于 PixieDust 流媒体的内容，你可以在这里找到其他流媒体应用示例：

+   一个简单的 PixieApp 示例，演示如何从随机生成的数据创建流媒体可视化：[`github.com/pixiedust/pixiedust/blob/master/notebook/pixieapp-streaming/PixieApp%20Streaming-Random.ipynb`](https://github.com/pixiedust/pixiedust/blob/master/notebook/pixieapp-streaming/PixieApp%20Streaming-Random.ipynb)

    +   显示如何构建实时股票行情可视化的 PixieApp：[`github.com/pixiedust/pixiedust/blob/master/notebook/pixieapp-streaming/PixieApp%20Streaming-Stock%20Ticker.ipynb`](https://github.com/pixiedust/pixiedust/blob/master/notebook/pixieapp-streaming/PixieApp%20Streaming-Stock%20Ticker.ipynb)

    接下来的主题将介绍 PixieApp 事件，它可以让你在应用程序的不同组件之间添加交互性。

## 使用 PixieApp 事件添加仪表盘钻取功能

PixieApp 框架支持使用浏览器中可用的发布-订阅模式在不同组件之间发送和接收事件。使用这种模式的巨大优势在于，它借鉴了松耦合模式（[`en.wikipedia.org/wiki/Loose_coupling`](https://en.wikipedia.org/wiki/Loose_coupling)），使得发送和接收组件可以彼此独立。这样，它们的实现可以相互独立执行，并且不受需求变化的影响。这在你的 PixieApp 使用来自不同团队的不同 PixieApp 组件时非常有用，或者当事件来自用户与图表交互（例如，点击地图）时，你希望提供钻取功能。

每个事件都携带一个包含任意键值对的 JSON 负载。负载必须至少包含以下一个键（或两者）：

+   `targetDivId`：标识发送事件元素的 DOM ID

+   `type`：标识事件类型的字符串

发布者可以通过两种方式触发事件：

+   **声明式**：使用`pd_event_payload`属性来指定负载内容。该属性遵循与`pd_options`相同的规则：

    +   每个键值对必须使用`key=value`的表示法进行编码

    +   事件将由点击或变化事件触发

    +   必须支持`$val()`指令，以动态注入用户输入的内容

    +   使用`<pd_event_payload>`子元素输入原始 JSON

        ```py
        <button type="submit" pd_event_payload="type=topicA;message=Button clicked">
            Send event A
        </button>
        ```

        ```py
        <button type="submit">
            <pd_event_payload>
            {
                "type":"topicA",
                "message":"Button Clicked"
            }
            </pd_event_payload>
            Send event A
        </button>
        ```

    示例：

    或者，我们可以使用以下方法：

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode11.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode11.html)

+   **编程式**：在某些情况下，你可能想通过 JavaScript 直接触发事件。在这种情况下，你可以使用`pixiedust`全局对象的`sendEvent(payload, divId)`方法。`divId`是一个可选参数，指定事件的来源。如果省略`divId`参数，则默认为当前发送事件的元素的`divId`。因此，通常情况下，你应使用`pixiedust.sendEvent`而不带`divId`，来自用户事件的 JavaScript 处理程序，例如点击和悬停。

    示例：

    ```py
    <table
    onclick="pixiedust.sendEvent({type:'topicB',text:event.srcElement.innerText})">
        <tr><td>Row 1</td></tr>
        <tr><td>Row 2</td></tr>
        <tr><td>Row 3</td></tr>
    </table>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode12.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode12.html)

订阅者可以通过声明一个`<pd_event_handler>`元素来监听事件，该元素可以接受 PixieApp 内核执行属性中的任何一个，如`pd_options`和`pd_script`。它还必须使用`pd_source`属性来筛选它们想要处理的事件。`pd_source`属性可以包含以下值之一：

+   `targetDivId`：只接受来自指定 ID 元素的事件

+   `type`：只有指定类型的事件才会被接受

+   `"*"`：表示接受任何事件

示例：

```py
<div class="col-sm-6" id="listenerA{{prefix}}">
    Listening to button event
    <pd_event_handler
        pd_source="topicA"
        pd_script="print(eventInfo)"
        pd_target="listenerA{{prefix}}">
    </pd_event_handler>
</div>
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode13.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode13.html)

下图展示了组件如何相互交互：

![使用 PixieApp 事件添加仪表板下钻功能](img/B09699_05_06.jpg)

组件之间的事件发送/接收

在以下代码示例中，我们通过构建两个发布者，一个按钮元素和一个表格来说明 PixieDust 事件系统，其中每一行都是一个事件源。我们还有两个监听器，作为 div 元素实现：

```py
from pixiedust.display.app import *
@PixieApp
class TestEvents():
    @route()
    def main_screen(self):
        return """
<div>
    <button type="submit">
        <pd_event_payload>
        {
            "type":"topicA",
            "message":"Button Clicked"
        }
        </pd_event_payload>
        Send event A
    </button>
    <table onclick="pixiedust.sendEvent({type:'topicB',text:event.srcElement.innerText})">
        <tr><td>Row 1</td></tr>
        <tr><td>Row 2</td></tr>
        <tr><td>Row 3</td></tr>
    </table>
</div>
<div class="container" style="margin-top:30px">
    <div class="row">
        <div class="col-sm-6" id="listenerA{{prefix}}">
            Listening to button event
            <pd_event_handler pd_source="topicA" pd_script="print(eventInfo)" pd_target="listenerA{{prefix}}">
            </pd_event_handler>
        </div>
        <div class="col-sm-6" id="listenerB{{prefix}}">
            Listening to table event
            <pd_event_handler pd_source="topicB" pd_script="print(eventInfo)" pd_target="listenerB{{prefix}}">
            </pd_event_handler>
        </div>
    </div>
</div>
        """
app = TestEvents()
app.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode14.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode14.py)

上述代码会生成以下结果：

![使用 PixieApp 事件添加仪表板钻取功能](img/B09699_05_07.jpg)

PixieApp 事件的用户交互流程

PixieApp 事件使你能够创建具有钻取功能的复杂仪表板。还要知道，你可以利用 `display()` 框架为某些图表自动发布的事件。例如，内置渲染器（如 Google Maps、Mapbox 和 Table）将在用户点击图表上的某个地方时自动生成事件。这对于快速构建各种具有钻取功能的互动仪表板非常有用。

在下一个主题中，我们将讨论如何使用 PixieDust 可扩展性 API 创建自定义可视化。

## 扩展 PixieDust 可视化

PixieDust 被设计为高度可扩展。你可以创建自己的可视化并控制它何时可以被调用，这取决于正在显示的实体。PixieDust 框架提供了多个可扩展性层。最底层也是最强大的那一层让你创建自己的 `Display` 类。然而，大多数可视化具有许多共同的属性，例如标准选项（聚合、最大行数、标题等），或者是一个缓存机制，用来防止在用户仅选择了一个不需要重新处理数据的小选项时重新计算所有内容。

为了防止用户每次都从头开始，PixieDust 提供了第二层可扩展性，称为 **renderer**，它包含了这里描述的所有功能。

以下图示说明了不同的层级：

![扩展 PixieDust 可视化](img/B09699_05_08.jpg)

PixieDust 扩展层

要开始使用 **Display 扩展层**，你需要通过创建一个继承自 `pixiedust.display.DisplayHandlerMeta` 的类，将你的可视化显示在菜单中。此类包含两个需要重写的方法：

+   `getMenuInfo(self,entity,dataHandler)`：如果传入的实体参数不被支持，返回一个空数组，否则返回一个包含一组 JSON 对象的数组，其中包含菜单信息。每个 JSON 对象必须包含以下信息：

    +   `id`：一个唯一的字符串，用于标识你的工具。

    +   `categoryId`：一个唯一的字符串，用于标识菜单类别或组。稍后会提供所有内置类别的完整列表。

    +   `title`：一个任意字符串，用于描述菜单。

    +   `icon`：一个字体图标的名称，或者一个图片的 URL。

+   `newDisplayHandler(self,options,entity)`：当用户激活菜单时，将调用`newDisplayHandler()`方法。该方法必须返回一个继承自`pixiedust.display.Display`的类实例。要求该类实现`doRender()`方法，该方法负责创建可视化效果。

让我们以为 pandas DataFrame 创建自定义表格渲染为例。我们首先创建`DisplayHandlerMeta`类来配置菜单和工厂方法：

```py
from pixiedust.display.display import *
import pandas
@PixiedustDisplay()
class SimpleDisplayMeta(DisplayHandlerMeta):
    @addId
    def getMenuInfo(self,entity,dataHandler):
        if type(entity) is pandas.core.frame.DataFrame:
            return [
               {"categoryId": "Table", "title": "Simple Table", "icon": "fa-table", "id": "simpleTest"}
            ]
        return []
    def newDisplayHandler(self,options,entity):
        return SimpleDisplay(options,entity)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode15.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode15.py)

请注意，前面的`SimpleDisplayMeta`类需要使用`@PixiedustDisplay`进行装饰，这对于将此类添加到 PixieDust 插件的内部注册表是必需的。在`getMenuInfo()`方法中，我们首先检查实体类型是否为*pandas DataFrame*，如果不是，则返回一个空数组，表示此插件不支持当前实体，因此不会对菜单做出任何贡献。如果类型正确，我们将返回一个包含菜单信息的 JSON 对象的数组。

工厂方法`newDisplayHandler()`接受`options`和`entity`作为参数。`options`参数是一个包含用户选择的各种键/值对的字典。如我们稍后将看到的，可视化效果可以定义任意的键/值对来反映其功能，PixieDust 框架将自动将其保存在单元格元数据中。

例如，你可以为在 UI 中将 HTTP 链接显示为可点击的选项添加一个功能。在我们的示例中，我们返回一个定义好的`SimpleDisplay`实例，如下所示：

```py
class SimpleDisplay(Display):
    def doRender(self, handlerId):
        self._addHTMLTemplateString("""
<table class="table table-striped">
   <thead>
       {%for column in entity.columns.tolist()%}
       <th>{{column}}</th>
       {%endfor%}
   </thead>
   <tbody>
       {%for _, row in entity.iterrows()%}
       <tr>
           {%for value in row.tolist()%}
           <td>{{value}}</td>
           {%endfor%}
       </tr>
       {%endfor%}
   </tbody>
</table>
        """)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode16.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode16.py)

如前所述，`SimpleDisplay` 类必须继承自 `Display` 类，并实现 `doRender()` 方法。在该方法的实现中，您可以访问 `self.entity` 和 `self.options` 变量，以调整如何在屏幕上呈现信息。在前面的示例中，我们使用 `self._addHTMLTemplateString()` 方法来创建将渲染可视化的 HTML 片段。与 PixieApp 路由类似，传递给 `self._addHTMLTemplateString()` 的字符串可以利用 Jinja2 模板引擎，并自动访问如 `entity` 等变量。如果您不想在 Python 文件中硬编码模板字符串，您可以将其提取到一个独立的文件中，该文件必须放置在名为 `templates` 的目录中，并且该目录必须与调用 Python 文件位于同一目录下。然后，您需要使用 `self._addHTMLTemplate()` 方法，该方法将文件名作为参数（无需指定 `templates` 目录）。

### 注意

将 HTML 片段外部化为独立文件的另一个好处是，您在进行更改时不必每次都重启内核，这可以节省很多时间。由于 Python 的工作方式，如果 HTML 片段嵌入在源代码中，情况就不同了，在这种情况下，您必须重新启动内核才能使 HTML 片段的任何更改生效。

还需要注意的是，`self._addHTMLTemplate()` 和 `self._addHTMLTemplateString()` 接受关键字参数，这些参数将传递给 Jinja2 模板。例如：

```py
self._addHTMLTemplate('simpleTable.html', custom_arg = "Some value")
```

现在我们可以运行一个单元格，显示例如 `cars` 数据集：

### 注意

**注意**：**简单表格**扩展只适用于 pandas，而不适用于 Spark DataFrame。因此，如果您的 Notebook 连接到 Spark，您需要在调用 `sampleData()` 时使用 `forcePandas = True`。

![扩展 PixieDust 可视化](img/B09699_05_09.jpg)

在 pandas DataFrame 上运行自定义可视化插件

如 PixieDust 扩展层架构图所示，您还可以使用 **渲染器扩展层** 来扩展 PixieDust，**渲染器扩展层**比 **显示扩展层** 更具规定性，但开箱即用提供了更多的功能，如选项管理和中间数据计算缓存。从用户界面的角度来看，用户可以通过图表区域右上角的 **渲染器** 下拉菜单在不同的渲染器之间切换。

PixieDust 附带了一些内置渲染器，如 Matplotlib、Seaborn、Bokeh、Mapbox、Brunel 和 Google Maps，但它并不声明对底层可视化库（包括 Bokeh、Brunel 或 Seaborn）的硬依赖。因此，用户必须手动安装这些库，否则它们将不会出现在菜单中。

以下截图展示了在给定图表上切换渲染器的机制：

![扩展 PixieDust 可视化](img/B09699_05_10.jpg)

切换渲染器

添加新的渲染器类似于添加显示可视化（使用相同的 API），尽管实际上更简单，因为您只需构建一个类（无需构建元数据类）。以下是您需要遵循的步骤：

1.  创建一个从专门的 `BaseChartDisplay class` 继承的 Display 类。实现所需的 `doRenderChart()` 方法。

1.  使用 `@PixiedustRenderer` 装饰器注册 `rendererId`（在所有渲染器中必须是唯一的）和正在渲染的图表类型。

    注意，相同的 `rendererId` 可以用于渲染器中包含的所有图表。PixieDust 提供了一组核心图表类型：

    +   `tableView`

    +   `barChart`

    +   `lineChart`

    +   `scatterPlot`

    +   `pieChart`

    +   `mapView`

    +   `histogram`

1.  *(可选)* 使用 `@commonChartOptions` 装饰器创建一组动态选项。

1.  *(可选)* 通过覆盖 `get_options_dialog_pixieapp()` 方法来自定义选项对话框，返回 `pixiedust.display.chart.options.baseOptions` 包中继承自 `BaseOptions` 类的 PixieApp 类的完全限定名称。

例如，让我们使用扩展层的渲染器重新编写前述的自定义 `SimpleDisplay` 表可视化：

```py
from pixiedust.display.chart.renderers import PixiedustRenderer
from pixiedust.display.chart.renderers.baseChartDisplay import BaseChartDisplay

@PixiedustRenderer(rendererId="simpletable", id="tableView")
class SimpleDisplayWithRenderer(BaseChartDisplay):
    def get_options_dialog_pixieapp(self):
        return None #No options needed

    def doRenderChart(self):
        return self.renderTemplateString("""
<table class="table table-striped">
   <thead>
       {%for column in entity.columns.tolist()%}
       <th>{{column}}</th>
       {%endfor%}
   </thead>
   <tbody>
       {%for _, row in entity.iterrows()%}
       <tr>
           {%for value in row.tolist()%}
           <td>{{value}}</td>
           {%endfor%}
       </tr>
       {%endfor%}
   </tbody>
</table>
        """)
```

### 注意

您可以在此处找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode17.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode17.py)

我们使用 `@PixiedustRenderer` 装饰器装饰这个类，指定一个名为 `simpletable` 的唯一 `rendererId`，并将其与 PixieDust 框架定义的 `tableView` 图表类型关联。我们在 `get_options_dialog_pixieapp()` 方法中返回 `None`，表示此扩展不支持自定义选项。因此，**选项**按钮将不会显示。在 `doRenderChart()` 方法中，我们返回 HTML 片段。由于我们想使用 Jinja2，我们需要使用 `self.renderTemplateString` 方法进行渲染。

现在，我们可以使用 `cars` 数据集测试这个新的渲染器。

### 注意

再次运行代码时，请确保将 `cars` 数据集加载为 pandas DataFrame。如果您已经运行了**简单表**的第一个实现，并且正在重用笔记本电脑，则可能仍会看到旧的**简单表**菜单。如果是这种情况，请重新启动内核并重试。

下图显示了作为渲染器的简单表可视化：

![扩展 PixieDust 可视化](img/B09699_05_11.jpg)

测试简单表的渲染器实现

您可以在这里找到更多关于此主题的材料：[`pixiedust.github.io/pixiedust/develop.html`](https://pixiedust.github.io/pixiedust/develop.html)。希望到目前为止，您已经对您可以编写的类型自定义有了一个很好的了解，以集成到 `display()` 框架中。

在接下来的章节中，我们将讨论开发者非常重要的一个主题：调试。

## 调试

能够快速调试应用程序对于项目的成功至关重要。如果没有这样做，我们在打破数据科学与工程之间的壁垒所取得的生产力和协作上的进展，大部分（如果不是全部）将会丧失。还需要注意的是，我们的代码在不同的位置运行，即 Python 在服务器端，JavaScript 在客户端，调试必须在这两个地方进行。对于 Python 代码，让我们来看两种排查编程错误的方法。

### 在 Jupyter Notebook 中使用 pdb 调试

pdb（[`docs.python.org/3/library/pdb.html`](https://docs.python.org/3/library/pdb.html)）是一个交互式命令行 Python 调试器，每个 Python 发行版中都默认包含。

调用调试器的方式有多种：

+   启动时，从命令行：

    ```py
    python -m pdb <script_file>

    ```

+   在代码中以编程方式：

    ```py
    import pdb
    pdb.run("<insert a valid python statement here>")
    ```

+   通过在代码中设置显式的断点，使用`set_trace()`方法：

    ```py
    import pdb
    def my_function(arg1, arg2):
        pdb.set_trace()
        do_something_here()
    ```

    ### 注意

    您可以在此处找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode18.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode18.py)

+   在发生异常后，通过调用`pdb.pm()`进行事后调试。

在交互式调试器中，您可以调用命令、检查变量、运行语句、设置断点等。

### 注意

完整的命令列表可以在此处找到：

[`docs.python.org/3/library/pdb.html`](https://docs.python.org/3/library/pdb.html)

好消息是 Jupyter Notebook 对交互式调试器提供了一流的支持。要调用调试器，只需使用`%pdb`单元魔法命令来启用/禁用它，并且如果触发异常，调试器将自动在出错的行停止执行。

魔法命令（[`ipython.readthedocs.io/en/stable/interactive/magics.html`](http://ipython.readthedocs.io/en/stable/interactive/magics.html)）是特定于 IPython 内核的构造。它们与语言无关，因此理论上可以在内核支持的任何语言中使用（例如 Python、Scala 和 R）。

有两种类型的魔法命令：

+   **行魔法**：语法为`%<magic_command_name> [optional arguments]`，例如`%matplotlib inline`，它配置 Matplotlib 将图表以内联方式输出到 Notebook 的输出单元格中。

    它们可以在单元格代码中的任何位置调用，甚至可以返回值，赋值给 Python 变量，例如：

    ```py
    #call the pwd line magic to get the current working directory
    #and assign the result into a Python variable called pwd
    pwd = %pwd
    print(pwd)
    ```

    ### 注意

    您可以在此处找到所有行魔法命令的列表：

    [`ipython.readthedocs.io/en/stable/interactive/magics.html#line-magics`](http://ipython.readthedocs.io/en/stable/interactive/magics.html#line-magics)

+   **单元魔法**：语法为`%%<magic_command_name> [optional arguments]`。例如，我们可以调用 HTML 单元魔法在输出单元格中显示 HTML：

    ```py
    %%html
    <div>Hello World</div>
    ```

    单元格魔法命令必须位于单元格的顶部；如果放在其他位置将导致执行错误。单元格魔法命令下方的所有内容都会作为参数传递给处理程序，并根据单元格魔法命令的规范进行解释。例如，HTML 单元格魔法命令期望单元格的其余部分是 HTML 格式。

以下代码示例调用了一个引发`ZeroDivisionError`异常的函数，并且激活了`pdb`的自动调用：

### 注意

**注意**：一旦启用`pdb`，它将保持开启，直到整个 Notebook 会话结束。

![使用 pdb 在 Jupyter Notebook 中调试](img/B09699_05_12.jpg)

交互式命令行调试

这里有一些可以用来排查问题的重要`pdb`命令：

+   `s(tep)`：进入被调用的函数并停在下一行语句处。

+   `n(ext)`：继续执行到下一行，而不进入嵌套函数。

+   `l(list)`：列出当前行周围的代码。

+   `c(ontinue)`：继续运行程序并停在下一个断点，或者当其他异常被触发时停下。

+   `d(own)`：向下移动堆栈帧。

+   `u(p)`：向上移动堆栈帧。

+   `<any expression>`：在当前框架上下文中评估并显示一个表达式。例如，你可以使用`locals()`来获取当前框架作用域内的所有局部变量列表。

如果发生了异常，而且你没有设置自动调用`pdb`，你仍然可以在事后通过在另一个单元格中使用`%debug`魔法命令来调用调试器，如下图所示：

![使用 pdb 在 Jupyter Notebook 中调试](img/B09699_05_13.jpg)

使用%debug 进行事后调试会话

类似于普通的 Python 脚本，你也可以使用`pdb.set_trace()`方法显式地设置一个断点。然而，建议使用由 IPython 核心模块提供的增强版`set_trace()`，它支持语法高亮：

![使用 pdb 在 Jupyter Notebook 中调试](img/B09699_05_14.jpg)

显式断点

在下一个主题中，我们将介绍一个由 PixieDust 提供的增强版 Python 调试器。

### 使用 PixieDebugger 进行可视化调试

使用标准命令行调试工具 Python 的 pdb 来调试代码是一个不错的工具，但它有两个主要的局限性：

+   它是命令行导向的，这意味着命令必须手动输入，结果会按顺序附加到单元格输出中，这使得它在进行高级调试时不太实用。

+   它不能与 PixieApps 一起使用

PixieDebugger 的功能解决了这两个问题。你可以在 Jupyter Notebook 的任何 Python 代码中使用它来进行可视化调试。要在单元格中启用 PixieDebugger，只需在单元格顶部添加`%%pixie_debugger`单元格魔法命令。

### 注意

**注意**：如果你还没有这么做，请记得在尝试使用`%%pixie_debugger`之前，先在单独的单元格中导入`pixiedust`。

例如，以下代码尝试计算`cars`数据集中名为`chevrolet`的汽车数量：

```py
%%pixie_debugger
import pixiedust
cars = pixiedust.sampleData(1, forcePandas=True)

def count_cars(name):
    count = 0
    for row in cars.itertuples():
        if name in row.name:
            count += 1     return count

count_cars('chevrolet')
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode19.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode19.py)

运行前面的代码单元将触发以下截图所示的可视化调试器。用户界面让你逐行调试代码，具备检查局部变量、评估 Python 表达式和设置断点的功能。代码执行工具栏提供了用于管理代码执行的按钮：恢复执行、跳过当前行、进入代码中的特定函数、运行到当前函数的末尾，以及上下显示栈帧一层：

![使用 PixieDebugger 进行可视化调试](img/B09699_05_15.jpg)

PixieDebugger 在工作中

没有参数时，`pixie_debugger`单元格魔法将会在代码中的第一个可执行语句处停止。你可以通过使用`-b`开关轻松配置它在特定位置停止，后面跟着一个断点列表，这些断点可以是行号或方法名。

从前面的示例代码开始，让我们在`count_cars()`方法和**第 11 行**添加断点：

```py
%%pixie_debugger -b count_cars 11
import pixiedust
cars = pixiedust.sampleData(1, forcePandas=True)

def count_cars(name):
    count = 0
    for row in cars.itertuples():
        if name in row.name:
            count += 1
    return count

count_cars('chevrolet')
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode20.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode20.py)

运行前面的代码将触发 PixieDebugger，在`count_cars()`方法的第一个可执行语句处停止。它还在第 11 行添加了一个断点，如果用户恢复执行，执行流程将在此停止，如下图所示：

![使用 PixieDebugger 进行可视化调试](img/B09699_05_16.jpg)

带有预定义断点的 PixieDebugger

### 注意

**注意**：要运行到特定的代码行而不设置显式断点，只需在左侧面板的行号区域悬停，然后点击出现的图标。

像`%debug`行魔法一样，你还可以使用`%pixie_debugger`行魔法来调用 PixieDebugger 进行事后调试。

### 使用 PixieDebugger 调试 PixieApp 路由

PixieDebugger 完全集成到 PixieApp 框架中。每当触发路由时发生异常，生成的回溯信息将会增加两个额外的按钮：

+   **事后调试**：调用 PixieDebugger 开始事后故障排除会话，允许你检查变量并分析堆栈帧

+   **调试路线**：回放当前路线，在 PixieDebugger 中停止在第一个可执行语句处

例如，以下是实现一个 PixieApp 的代码，允许用户通过提供列名和查询条件来搜索`cars`数据集：

```py
from pixiedust.display.app import *

import pixiedust
cars = pixiedust.sampleData(1, forcePandas=True)

@PixieApp
class DisplayCars():
    @route()
    def main_screen(self):
        return """
        <div>
            <label>Column to search</label>
            <input id="column{{prefix}}" value="name">
            <label>Query</label>
            <input id="search{{prefix}}">
            <button type="submit" pd_options="col=$val(column{{prefix}});query=$val(search{{prefix}})"
                pd_target="target{{prefix}}">
                Search
            </button>
        </div>
        <div id="target{{prefix}}"></div>
        """
    @route(col="*", query="*")
    def display_screen(self, col, query):
        self.pdf = cars.loc[cars[col].str.contains(query)]
        return """
        <div pd_render_onload pd_entity="pdf">
            <pd_options>
            {
              "handlerId": "tableView",
              "table_noschema": "true",
              "table_nosearch": "true",
              "table_nocount": "true"
            }
            </pd_options>
        </div>
        """
app = DisplayCars()
app.run()
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode21.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode21.py)

搜索列的默认值为`name`，但如果用户输入的列名不存在，将会生成如下的回溯信息：

![Debugging PixieApp routes with PixieDebugger](img/B09699_05_17.jpg)

增强的回溯信息，带有用于调用 PixieDebugger 的按钮

点击**Debug Route**将自动启动 PixieDebugger，并在路由的第一个可执行语句处停下来，如下图所示：

![Debugging PixieApp routes with PixieDebugger](img/B09699_05_18.jpg)

调试 PixieApp 路由

你也可以通过使用`debug_route`关键字参数来让 PixieDebugger 在`display_screen()`路由处停下来，而无需等待回溯信息的生成，方法如下：

```py
...
app = DisplayCars()
app.run(debug_route="display_screen")
```

PixieDebugger 是第一个为 Jupyter Notebook 提供的可视化 Python 调试器，提供了 Jupyter 用户社区长期要求的功能。然而，实时调试并不是开发者使用的唯一工具。在接下来的部分，我们将通过检查日志记录消息来进行调试。

### 使用 PixieDust 日志记录进行故障排查

习惯上最好在代码中使用日志记录消息，而 PixieDust 框架提供了一种简便的方式，可以直接从 Jupyter Notebook 创建和读取日志消息。首先，你需要通过调用`getLogger()`方法创建一个日志记录器，方法如下：

```py
import pixiedust
my_logger = pixiedust.getLogger(__name__)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode22.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode22.py)

你可以将任何内容作为`getLogger()`方法的参数。然而，为了更好地识别特定消息的来源，建议使用`__name__`变量，它返回当前模块的名称。`my_logger`变量是一个标准的 Python 日志记录器对象，提供各种级别的日志记录方法：

+   `debug(msg, *args, **kwargs)`：以`DEBUG`级别记录一条消息。

+   `info(msg, *args, **kwargs)`：以`INFO`级别记录一条消息。

+   `warning(msg, *args, **kwargs)`：以`WARNING`级别记录一条消息。

+   `error(msg, *args, **kwargs)`：以`ERROR`级别记录一条消息。

+   `critical(msg, *args, **kwargs)`：以`CRITICAL`级别记录一条消息。

+   `exception(msg, *args, **kwargs)`：以`EXCEPTION`级别记录一条消息。此方法仅应在异常处理程序中调用。

### 注意

**注意**：你可以在这里找到更多关于 Python 日志框架的信息：

[`docs.python.org/2/library/logging.html`](https://docs.python.org/2/library/logging.html)

然后你可以通过`%pixiedustLog`单元魔法直接从 Jupyter Notebook 查询日志消息，该魔法需要以下参数：

+   `-l`: 按日志级别过滤，例如 `CRITICAL`、`FATAL`、`ERROR`、`WARNING`、`INFO` 和 `DEBUG`

+   `-f`: 过滤包含特定字符串的消息，例如 `Exception`

+   `-m`: 返回的最大日志消息数

在以下示例中，我们使用 `%pixiedustLog` 魔法来显示所有调试消息，将这些消息限制为最后五条：

![使用 PixieDust 日志排查问题](img/B09699_05_19.jpg)

显示最后五条日志消息

为了方便使用，在处理 Python 类时，你还可以使用 `@Logger` 装饰器，它会自动创建一个以类名为标识符的日志记录器。

这是一个使用 `@Logger` 装饰器的代码示例：

```py
from pixiedust.display.app import *
from pixiedust.utils import Logger

@PixieApp
@Logger()
class AppWithLogger():
    @route()
    def main_screen(self):
        self.info("Calling default route")
        return "<div>hello world</div>"

app = AppWithLogger()
app.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode23.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode23.py)

在单元格中运行前述 PixieApp 后，你可以调用 `%pixiedustLog` 魔法来显示消息：

![使用 PixieDust 日志排查问题](img/B09699_05_20.jpg)

使用特定术语查询日志

这完成了我们关于服务器端调试的讨论。在下一节中，我们将探讨一种执行客户端调试的技术

### 客户端调试

PixieApp 编程模型的设计原则之一是尽量减少开发者编写 JavaScript 的需要。框架将通过监听用户输入事件（如点击或更改事件）自动触发内核请求。然而，在某些情况下，编写少量的 JavaScript 是不可避免的。这些 JavaScript 片段通常是特定路由 HTML 片段的一部分，并动态注入到浏览器中，这使得调试变得非常困难。

一种流行的技巧是在 JavaScript 代码中加入 `console.log` 调用，以便将消息打印到浏览器的开发者控制台。

### 注意

**注意**：每种浏览器都有自己调用开发者控制台的方式。例如，在 Google Chrome 中，你可以使用 **查看** | **开发者** | **JavaScript 控制台**，或 *Command* + *Alt* + *J* 快捷键。

另一个我特别喜欢的调试技巧是通过在 JavaScript 代码中编程插入一个断点，使用 `debugger;` 语句。除非浏览器开发者工具已打开并启用了源代码调试，否则此语句没有任何效果。在这种情况下，执行将自动在 `debugger;` 语句处中断。

以下是一个 PixieApp 示例，使用 JavaScript 函数解析 `$val()` 指令引用的动态值：

```py
from pixiedust.display.app import *

@PixieApp
class TestJSDebugger():
    @route()
    def main_screen(self):
        return """
<script>
function FooJS(){
    debugger;
    return "value"
}
</script>
<button type="submit" pd_options="state=$val(FooJS)">Call route</button>
        """

    @route(state="*")
    def my_route(self, state):
        return "<div>Route called with state <b>{{state}}</b></div>"

app = TestJSDebugger()
app.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode24.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode24.py)

在上面的代码中，按钮动态地使用包含调试器语句的`FooJS` JavaScript 函数设置状态的值。执行应用程序并在开发者工具打开时点击按钮，将自动在浏览器中启动调试会话：

![客户端调试](img/B09699_05_21.jpg)

在客户端使用调试器调试 JavaScript 代码；语句

# 在 Python Notebook 中运行 Node.js

尽管我在本书开头已经明确表示，Python 已成为数据科学领域的明确领导者，但它在开发者社区中的使用仍然处于边缘地位，而传统语言（如 Node.js）仍然是首选。认识到对于一些开发者而言，学习像 Python 这样的新语言可能是进入数据科学的成本太高，我与我的 IBM 同事 Glynn Bird 合作，构建了一个名为`pixiedust_node`的 PixieDust 扩展库（[`github.com/pixiedust/pixiedust_node`](https://github.com/pixiedust/pixiedust_node)），让开发者可以在 Python Notebook 的单元格中运行 Node.js/JavaScript 代码。这个库的目标是通过让开发者可以重用他们喜欢的 Node.js 库（例如，用于加载和处理现有数据源中的数据）来帮助他们更容易地进入 Python 世界。

要安装`pixiedust_node`库，只需在自己的单元格中运行以下命令：

```py
!pip install pixiedust_node

```

### 注意

**注意**：安装完成后，不要忘记重启内核。

**重要**：你需要确保在与 Jupyter Notebook Server 同一台机器上安装 Node.js 运行时版本 6 或更高版本。

一旦内核重启，我们导入`pixiedust_node`模块：

```py
import pixiedust_node
```

你应该在输出中看到关于 PixieDust 和`pixiedust_node`的信息，如下所示：

![在 Python Notebook 中运行 Node.js](img/B09699_05_22.jpg)

pixiedust_node 欢迎输出

当导入`pixiedust_node`时，Python 端会创建一个 Node 子进程，并启动一个特殊线程来读取子进程的输出，将其传递给 Python 端，以便在当前执行的 Notebook 单元格中显示。这个子进程负责启动**REPL**会话（**读取-求值-打印循环**：[`en.wikipedia.org/wiki/Read-eval-print_loop`](https://en.wikipedia.org/wiki/Read-eval-print_loop)），它将执行所有从 Notebook 发送的脚本，并使所有创建的类、函数和变量在所有执行中可重用。

它还定义了一组旨在与 Notebook 和 PixieDust `display()` API 交互的函数：

+   `print(data)`：在当前执行的 Notebook 单元格中输出 data 的值。

+   `display(data)`：调用 PixieDust 的`display()` API，使用从数据转换的 pandas DataFrame。如果数据无法转换为 pandas DataFrame，则默认使用`print`方法。

+   `html(data)`：以 HTML 格式在当前执行的 Notebook 单元格中显示数据。

+   `image(data)`：期望数据是一个指向图像的 URL，并在当前执行的单元格中显示图像。

+   `help()`：显示所有前述方法的列表。

此外，`pixiedust_node` 使两个变量，`npm` 和 `node`，在笔记本中全局可用：

+   `node.cancel()`：停止当前在 Node.js 子进程中执行的代码。

+   `node.clear()`：重置 Node.js 会话；所有现有变量将被删除。

+   `npm.install(package)`：安装一个 npm 包并使其在 Node.js 会话中可用。该包在会话之间保持持久。

+   `npm.uninstall(package)`：从系统和当前 Node.js 会话中删除 npm 包。

+   `npm.list()`：列出当前安装的所有 npm 包。

`pixiedust_node` 创建一个单元格魔法，允许你运行任意 JavaScript 代码。只需在单元格顶部使用 `%%node` 魔法并像往常一样运行，代码将被执行在 Node.js 子进程的 REPL 会话中。

以下代码使用 JavaScript `Date` 对象（[`www.w3schools.com/Jsref/jsref_obj_date.asp`](https://www.w3schools.com/Jsref/jsref_obj_date.asp)）显示一个包含当前日期时间的字符串：

```py
%%node
var date = new Date()
print("Today's date is " + date)
```

这将输出以下内容：

```py
"Today's date is Sun May 27 2018 20:36:35 GMT-0400 (EDT)"
```

以下图示说明了前述单元格的执行流程：

![在 Python 笔记本中运行 Node.js](img/B09699_05_23.jpg)

Node.js 脚本执行的生命周期

JavaScript 代码由 `pixiedust_node` 魔法处理并发送到 Node 子进程执行。在代码执行过程中，其输出将由特殊线程读取并显示回当前在笔记本中执行的单元格。请注意，JavaScript 代码可能会进行异步调用，在这种情况下，执行将立即返回，而异步调用可能还没有完成。在这种情况下，笔记本会显示单元格代码已完成，即使异步代码可能稍后会生成更多输出。无法确定异步代码何时完成。因此，开发人员必须小心地管理此状态。

`pixiedust_node` 还具有在 Python 和 JavaScript 之间共享变量的能力，反之亦然。因此，你可以声明一个 Python 变量（例如整数数组），在 JavaScript 中应用转换（也许使用你喜欢的库），然后再返回 Python 中处理。

以下代码在两个单元格中运行，一个纯 Python 单元格声明一个整数数组，另一个 JavaScript 单元格将每个元素乘以 2：

![在 Python 笔记本中运行 Node.js](img/B09699_05_24.jpg)

反向方向也同样有效。以下代码首先在 JavaScript 的 node 单元格中创建一个 JSON 变量，然后在 Python 单元格中创建并显示一个 pandas DataFrame：

```py
%%node
data = {
    "name": ["Bob","Alice","Joan","Christian"],
    "age": [20, 25, 19, 45]
}
print(data)
```

结果如下：

```py
{"age": [20, 25, 19, 45], "name": ["Bob", "Alice", "Joan", "Christian"]}
```

然后，在 Python 单元格中，我们使用 PixieDust 的 `display()`：

```py
df = pandas.DataFrame(data)
display(df)
```

使用以下选项：

![在 Python 笔记本中运行 Node.js](img/B09699_05_25.jpg)

从 Node 单元创建的数据的 display() 选项

我们得到以下结果：

![在 Python 笔记本中运行 Node.js](img/B09699_05_26.jpg)

从 Node 单元创建的数据生成的柱状图

我们也可以直接从 Node 单元使用`pixiedust_node`提供的 `display()` 方法，达到相同的结果，如下所示：

```py
%%node
data = {
    "name": ["Bob","Alice","Joan","Christian"],
    "age": [20, 25, 19, 45]
}
display(data)
```

如果你想了解更多关于`pixiedust_node`的信息，我强烈推荐阅读这篇博客文章：[`medium.com/ibm-watson-data-lab/nodebooks-node-js-data-science-notebooks-aa140bea21ba`](https://medium.com/ibm-watson-data-lab/nodebooks-node-js-data-science-notebooks-aa140bea21ba)。像往常一样，我鼓励读者通过贡献代码或提出改进意见来参与这些工具的改进。

# 总结

在本章中，我们探索了各种高级概念、工具和最佳实践，增加了更多工具到我们的工具箱中，涵盖了从 PixieApps（流式处理、如何通过将第三方库与`@captureOutput`集成来实现路由、PixieApp 事件、以及通过`pd_app`实现更好的模块化）到开发者必备工具 PixieDebugger 的内容。我们还详细介绍了如何使用 PixieDust `display()` API 创建自定义可视化。我们还讨论了`pixiedust_node`，它是 PixieDust 框架的扩展，允许那些更熟悉 JavaScript 的开发者在他们喜爱的语言中处理数据。

在本书的剩余部分，我们将利用这些学到的知识，构建行业应用数据管道，从第六章，*数据分析研究：使用 TensorFlow 进行 AI 和图像识别*中的 *深度学习视觉识别* 应用开始。

本书结尾提供了关于 PixieApp 编程模型的开发者快速参考指南，详见附录，*PixieApp 快速参考*。
