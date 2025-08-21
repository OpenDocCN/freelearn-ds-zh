# 前言

## 关于

本节简要介绍了作者、书籍内容、你开始时所需的技术技能，以及完成书中所有活动和练习所需的硬件和软件要求。

## 关于本书

《应用数据科学：Python 和 Jupyter》将教授你入门级数据科学所需的技能。你将了解一些最常用的库，这些库是 Anaconda 发行版的一部分，然后通过真实数据集探索机器学习模型，帮助你获得在现实世界中应用这些技能的能力。最后，你将学习如何轻松地从开放网络抓取并收集数据，将这些新技能应用到可操作的实际场景中。

### 关于作者

Alex Galea 自从毕业于加拿大圭尔夫大学，获得物理学硕士学位后，便一直从事数据分析工作。在研究量子气体作为研究生项目的一部分时，他对 Python 产生了浓厚的兴趣。最近，Alex 从事网络数据分析工作，Python 依然在他的工作中扮演着重要角色。他经常写博客，分享工作和个人项目，通常这些项目围绕数据展开，并且通常涉及 Python 和 Jupyter Notebooks。

### 目标

+   快速启动 Jupyter 生态系统

+   确定潜在的研究领域并进行探索性数据分析

+   规划机器学习分类策略并训练分类模型

+   使用验证曲线和降维技术来调优和增强你的模型

+   从网页抓取表格数据并将其转换为 Pandas DataFrame

+   创建交互式、适用于网络的可视化图表，以清晰地传达你的发现

### 目标读者

《应用数据科学：Python 和 Jupyter》非常适合来自各行各业的专业人士，鉴于数据科学的流行和普及。你需要具备一定的 Python 使用经验，任何涉及 Pandas、Matplotlib 和 Pandas 等库的工作经验都将帮助你更好地入门。

### 方法

《应用数据科学：Python 和 Jupyter》涵盖了标准数据工作流程的各个方面，完美结合了理论、实际编程操作和生动的示例插图。每个模块都是建立在前一章内容的基础上的。本书包含多个使用实际商业场景的活动，让你在高度相关的情境中实践并应用新学到的技能。

### 最低硬件要求

最低硬件要求如下：

+   处理器：Intel i5（或同等配置）

+   内存：8 GB RAM

+   硬盘：10 GB

+   互联网连接

### 软件要求

你还需要提前安装以下软件：

+   Python 3.5+

+   Anaconda 4.3+

+   包含在 Anaconda 安装中的 Python 库：

+   matplotlib 2.1.0+

+   ipython 6.1.0+

+   requests 2.18.4+

+   beautifulsoup4 4.6.0+

+   numpy 1.13.1+

+   pandas 0.20.3+

+   scikit-learn 0.19.0+

+   seaborn 0.8.0+

+   bokeh 0.12.10+

+   需要手动安装的 Python 库：

+   mlxtend

+   version_information

+   ipython-sql

+   pdir2

+   graphviz

### 安装与设置

在开始本书之前，我们将安装包含 Python 和 Jupyter Notebook 的 Anaconda 环境。

### 安装 Anaconda

1.  在浏览器中访问 https://www.anaconda.com/download/。

1.  根据你使用的操作系统，点击 Windows、Mac 或 Linux。

1.  接下来，点击下载选项。确保下载最新版本。

1.  下载后打开安装程序。

1.  按照安装程序中的步骤操作，完成后即可！你的 Anaconda 发行版已经准备好。

### 更新 Jupyter 并安装依赖项

1.  搜索 Anaconda Prompt 并打开它。

1.  输入以下命令以更新 conda 和 Jupyter：

    ```py
    #Update conda
    conda update conda
    #Update Jupyter
    conda update Jupyter
    #install packages
    conda install numpy
    conda install pandas
    conda install statsmodels
    conda install matplotlib
    conda install seaborn
    ```

1.  若要从 Anaconda Prompt 打开 Jupyter Notebook，请使用以下命令：

    ```py
    jupyter notebook
    pip install -U scikit-learn
    ```

### 额外资源

本书的代码包也托管在 GitHub 上，网址为[`github.com/TrainingByPackt/Applied-Data-Science-with-Python-and-Jupyter`](https://github.com/TrainingByPackt/Applied-Data-Science-with-Python-and-Jupyter)。

我们还有来自丰富书籍和视频目录的其他代码包，网址为 https://github.com/PacktPublishing/。快去看看吧！

### 约定

文本中的代码单词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名显示如下：

"最后，图像会保存为高分辨率 PNG 格式到`figures`文件夹。"

代码块如下设置：

```py
y = df['MEDV'].copy()
del df['MEDV']
df = pd.concat((y, df), axis=1)
```

任何命令行输入或输出都如下所示：

```py
jupyter notebook
```

新术语和重要单词以粗体显示。你在

屏幕上的内容，例如菜单或对话框中的内容，以如下方式显示："点击右上角的**新建**，并从下拉菜单中选择一个内核。"
