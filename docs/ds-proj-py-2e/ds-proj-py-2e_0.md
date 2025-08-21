# 前言

# 关于本书

如果数据是新石油，那么机器学习就是钻井机。随着公司能够获取越来越多的原始数据，能够提供支持商业决策的先进预测模型的能力变得愈发宝贵。

在本书中，你将基于一个现实的 数据集进行从头到尾的项目，内容被分解为易于操作的小练习。这种方式采用了案例研究的方法，模拟了你在实际数据科学项目中可能遇到的工作环境。

你将学习如何使用包括 pandas、Matplotlib 和 scikit-learn 在内的关键 Python 包，并掌握数据探索和数据处理的过程，随后再进行算法的拟合、评估和调优，如正则化逻辑回归和随机森林。

本书现已出版第二版，将带领你走过从探索数据到交付机器学习模型的全过程。该版本已更新至 2021 年，新增了关于 XGBoost、SHAP 值、算法公平性以及在现实世界中部署模型的伦理问题的内容。

在完成这本数据科学书籍后，你将具备构建自己的机器学习模型并从真实数据中获得洞见的技能、理解力和信心。

## 关于作者

*Stephen Klosterman* 是一位机器学习数据科学家，拥有数学、环境科学和生态学的背景。他的教育背景包括哈佛大学的生物学博士学位，并曾在数据科学课程中担任助教。他的专业经验涵盖了环境、医疗和金融领域。在工作中，他喜欢研究和开发能够创造价值并且易于理解的机器学习解决方案。在业余时间，他喜欢跑步、骑行、划桨板和音乐。

## 目标

+   使用 pandas Python 包加载、探索和处理数据

+   使用 Matplotlib 创建有效的数据可视化

+   使用 scikit-learn 和 XGBoost 实现预测性机器学习模型

+   使用 lasso 回归和岭回归减少模型的过拟合

+   构建决策树的集成模型，使用随机森林和梯度提升

+   评估模型性能并解释模型预测

+   通过明确的商业建议提供有价值的洞见

## 目标读者

*Python 数据科学项目（第二版）*适合任何想要入门数据科学和机器学习的人。如果你希望通过数据分析和预测建模来生成商业洞察，推动你的职业发展，那么本书是一个完美的起点。为了快速掌握所涉及的概念，建议你具有 Python 或其他类似语言（如 R、Matlab、C 等）的编程基础。此外，了解基本统计学知识，包括概率与线性回归等课程内容，或者在阅读本书时自行学习这些知识将会对你有所帮助。

## 方法

*Python 数据科学项目*采用案例学习方法，通过真实世界数据集的背景来教授概念。清晰的解释将加深你的理解，而富有趣味的练习和具有挑战性的活动将通过实践巩固你的知识。

## 关于各章节

*第一章*，*数据探索与清洗*，将帮助你开始使用 Python 和 Jupyter 笔记本。随后，本章将探索案例数据集，深入进行探索性数据分析、质量保证以及使用 pandas 进行数据清洗。

*第二章*，*Scikit-Learn 简介与模型评估*，将向你介绍二分类模型的评估指标。你将学习如何使用 scikit-learn 构建和评估二分类模型。

*第三章*，*逻辑回归与特征探索的细节*，深入探讨逻辑回归和特征探索。你将学习如何生成多特征与响应变量的相关性图，并将逻辑回归视为线性模型进行解读。

*第四章*，*偏差-方差权衡*，通过研究如何扩展逻辑回归模型来解决过拟合问题，探索了机器学习中过拟合、欠拟合和偏差-方差权衡的基础概念。

*第五章*，*决策树与随机森林*，将向你介绍基于树的机器学习模型。你将学习如何为机器学习任务训练决策树、可视化训练后的决策树，并训练随机森林并可视化结果。

*第六章*，*梯度提升、XGBoost 与 SHAP 值*，向你介绍了两个关键概念：梯度提升和**Shapley 加性解释**（**SHAP**）。你将学习如何训练 XGBoost 模型，并了解如何使用 SHAP 值为任何数据集的模型预测提供个性化的解释。

*第七章*，*测试集分析、财务洞察与客户交付*，介绍了几种分析模型测试集的技术，以推导出未来模型性能的可能洞察。本章还描述了交付和部署模型时需要考虑的关键因素，例如交付格式和如何监控模型的使用情况。

## 硬件要求

为了获得最佳的学习体验，我们推荐以下硬件配置：

+   处理器：Intel Core i5 或同等处理器

+   内存：4 GB RAM

+   存储：35 GB 可用空间

## 软件要求

你还需要预先安装以下软件：

+   操作系统：Windows 7 SP1 64 位、Windows 8.1 64 位或 Windows 10 64 位，Ubuntu Linux，或最新版本的 OS X

+   浏览器：Google Chrome/Mozilla Firefox 最新版本

+   Notepad++/Sublime Text 作为 IDE（这是可选的，因为你可以在浏览器中使用 Jupyter Notebook 完成所有练习）

+   安装 Python 3.8+（本书使用 Python 3.8.2）（来自 https://python.org 或通过 Anaconda 安装，见下文推荐）。在撰写时，用于*第六章*的 SHAP 库（*梯度提升、XGBoost 和 SHAP 值*）与 Python 3.9 不兼容。因此，如果你使用的是 Python 3.9 作为基础环境，建议按照下一节的说明设置 Python 3.8 环境。

+   根据需要安装 Python 库（如 Jupyter、NumPy、Pandas、Matplotlib 等，建议通过 Anaconda 安装，见下文）

## 安装与设置

在开始本书之前，建议安装 Anaconda 包管理器，并使用它来协调 Python 及其包的安装。

### 代码包

请查找本书的代码包，托管在 GitHub 上：https://github.com/PacktPublishing/Data-Science-Projects-with-Python-Second-Ed。

### Anaconda 和环境设置

你可以访问以下链接来安装 Anaconda：https://www.anaconda.com/products/individual。滚动到页面底部，下载与你系统相关的安装程序。

推荐在 Anaconda 中创建一个环境，以进行本书中的练习和活动，这些活动已在此处指示的软件版本上经过测试。安装 Anaconda 后，如果你使用的是 macOS 或 Linux，请打开终端；如果使用的是 Windows，请打开命令提示符窗口，然后执行以下操作：

1.  创建一个包含大多数所需包的环境。你可以根据需要命名它；在这里它被命名为 `dspwp2`。请将以下语句复制粘贴或直接在终端中输入：

    ```py
    conda create -n dspwp2 python=3.8.2 jupyter=1.0.0 pandas=1.2.1 scikit-learn=0.23.2 numpy=1.19.2 matplotlib=3.3.2 seaborn=0.11.1 python-graphviz=0.15 xlrd=2.0.1
    ```

1.  当提示时，键入`'y'`并按[Enter]键。

1.  激活环境：

    ```py
    conda activate dspwp2
    ```

1.  安装剩余的包：

    ```py
    conda install -c conda-forge xgboost=1.3.0 shap=0.37.0
    ```

1.  当提示时，键入`'y'`并按[Enter]键。

1.  你已经可以使用该环境了。完成后，要停用它：

    ```py
    conda deactivate
    ```

我们还在 https://github.com/PacktPublishing/ 上提供了来自我们丰富书籍和视频目录的其他代码包。快去看看吧！

## 约定

文章中的代码、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账户名显示如下：“通过在命令行输入`conda list`，你可以看到在你的环境中安装的所有包。”

一块代码的设置如下：

```py
import numpy as np #numerical computation
import pandas as pd #data wrangling
import matplotlib.pyplot as plt #plotting package
#Next line helps with rendering plots
%matplotlib inline
import matplotlib as mpl #add'l plotting functionality
mpl.rcParams['figure.dpi'] = 400 #high res figures
import graphviz #to visualize decision trees
```

新术语和重要单词以粗体显示。屏幕上看到的词汇，例如菜单或对话框中的词汇，会像这样出现在文本中：“从`New`菜单创建一个新的 Python 3 笔记本，如下所示。”

## 代码展示

跨越多行的代码使用反斜杠（ \ ）进行分割。当代码执行时，Python 会忽略反斜杠，并将下一行的代码视为当前行的直接延续。

例如：

```py
my_new_lr = LogisticRegression(penalty='l2', dual=False,\ 
                               tol=0.0001, C=1.0,\ 
                               fit_intercept=True,\ 
                               intercept_scaling=1,\ 
                               class_weight=None,\ 
                               random_state=None,\ 
                               solver='lbfgs',\ 
                               max_iter=100,\ 
                               multi_class='auto',\ 
                               verbose=0, warm_start=False,\ 
                               n_jobs=None, l1_ratio=None)
```

注释被添加到代码中以帮助解释特定的逻辑。单行注释使用 `#` 符号表示，如下所示：

```py
import pandas as pd
import matplotlib.pyplot as plt #import plotting package
#render plotting automatically
%matplotlib inline
```

## 与我们联系

我们始终欢迎读者的反馈。

`customercare@packtpub.com`。

**勘误**：虽然我们已经尽力确保内容的准确性，但错误仍然可能发生。如果你在本书中发现错误，我们将非常感激你能报告给我们。请访问 www.packtpub.com/support/errata 并填写表格。

`copyright@packt.com`，并附上材料链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有兴趣写作或为书籍贡献内容，请访问 authors.packtpub.com。

## 请留下评论

请通过在亚马逊上留下详细且公正的评论告诉我们你的想法。我们欢迎所有反馈——它帮助我们继续制作优秀的产品，并帮助有志开发者提升技能。请花几分钟时间提供你的想法——这对我们意义重大。你可以通过点击以下链接留下评论： https://packt.link/r/1800564481。
