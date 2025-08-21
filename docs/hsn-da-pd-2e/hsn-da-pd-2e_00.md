# 前言

数据科学通常被描述为一个跨学科的领域，其中编程技能、统计知识和领域知识相交。它已经迅速成为我们社会中最热门的领域之一，懂得如何处理数据已成为当今职业生涯中的必备技能。不论行业、角色或项目如何，数据技能的需求都很高，而学习数据分析是产生影响的关键。

数据科学领域涉及许多不同的方面：数据分析师更侧重于提取业务洞察，而数据科学家则更侧重于将机器学习技术应用于业务问题。数据工程师则专注于设计、构建和维护供数据分析师和数据科学家使用的数据管道。机器学习工程师与数据科学家的技能集有很多相似之处，并且像数据工程师一样，他们是熟练的软件工程师。数据科学领域涵盖了许多领域，但对于所有这些领域，数据分析都是基础构建模块。本书将为你提供入门所需的技能，无论你的旅程将带你走向何方。

传统的数据科学技能集包括了解如何从各种来源（如数据库和 API）收集数据并进行处理。Python 是一种流行的数据科学语言，提供了收集和处理数据以及构建生产级数据产品的手段。由于它是开源的，通过利用他人编写的库来解决常见的数据任务和问题，使得开始进行数据科学变得容易。

Pandas 是与 Python 中的数据科学同义的强大且流行的库。本书将通过使用 Pandas 在现实世界的数据集上进行数据分析，为你提供动手实践的入门，包括涉及股市、模拟黑客攻击、天气趋势、地震、葡萄酒和天文数据的实际案例。Pandas 通过使我们能够高效地处理表格数据，简化了数据处理和可视化的过程。

一旦我们掌握了如何进行数据分析，我们将探索许多应用。我们将构建 Python 包，并尝试进行股票分析、异常检测、回归、聚类和分类，同时借助常用于数据可视化、数据处理和机器学习的额外库，如 Matplotlib、Seaborn、NumPy 和 Scikit-learn。在你完成本书后，你将能充分准备好，开展自己的 Python 数据科学项目。

# 本书适用对象

本书面向那些有不同经验背景的人，旨在学习 Python 中的数据科学，可能是为了应用于项目、与数据科学家合作和/或与软件工程师一起进行机器学习生产代码的工作。如果你的背景与以下之一（或两个）相似，你将从本书中获得最大的收益：

+   你在其他语言（如 R、SAS 或 MATLAB）中有数据科学的经验，并希望学习 pandas，将你的工作流程迁移到 Python。

+   你有一定的 Python 经验，并希望学习如何使用 Python 进行数据科学。

# 本书内容

*第一章*，*数据分析简介*，教你数据分析的基本原理，为你打下统计学基础，并指导你如何设置环境以便在 Python 中处理数据并使用 Jupyter Notebooks。

*第二章*，*操作 Pandas DataFrame*，介绍了 `pandas` 库，并展示了如何处理 `DataFrame` 的基础知识。

*第三章*，*使用 Pandas 进行数据整理*，讨论了数据操作的过程，展示了如何探索 API 获取数据，并引导你通过 `pandas` 进行数据清洗和重塑。

*第四章*，*聚合 Pandas DataFrame*，教你如何查询和合并 `DataFrame`，如何对它们执行复杂的操作，包括滚动计算和聚合，以及如何有效处理时间序列数据。

*第五章*，*使用 Pandas 和 Matplotlib 可视化数据*，展示了如何在 Python 中创建你自己的数据可视化，首先使用 `matplotlib` 库，然后直接从 `pandas` 对象中创建。

*第六章*，*使用 Seaborn 绘图及自定义技术*，继续讨论数据可视化，教你如何使用 `seaborn` 库来可视化你的长格式数据，并为你提供自定义可视化的工具，使其达到可用于展示的效果。

*第七章*，*金融分析 – 比特币与股票市场*，带你了解如何创建一个用于分析股票的 Python 包，并结合从 *第一章*，*数据分析简介*，到 *第六章*，*使用 Seaborn 绘图及自定义技术*，所学的所有内容，并将其应用于金融领域。

*第八章*，*基于规则的异常检测*，介绍了如何模拟数据并应用从 *第一章*，*数据分析简介*，到 *第六章*，*使用 Seaborn 绘图及自定义技术*，所学的所有知识，通过基于规则的异常检测策略来捕捉试图认证进入网站的黑客。

*第九章*，*在 Python 中入门机器学习*，介绍了机器学习以及如何使用`scikit-learn`库构建模型。

*第十章*，*更好的预测 - 优化模型*，向你展示了调整和提高机器学习模型性能的策略。

*第十一章*，*机器学习异常检测*，重新探讨了通过机器学习技术进行登录尝试数据的异常检测，同时让你了解实际工作流程的样子。

*第十二章*，*前路漫漫*，讲解了提升技能和进一步探索的资源。

# 为了最大程度地从本书中受益

你应该熟悉 Python，特别是 Python 3 及以上版本。你还需要掌握如何编写函数和基本脚本，理解标准编程概念，如变量、数据类型和控制流程（if/else、for/while 循环），并能将 Python 用作函数式编程语言。具备一些面向对象编程的基础知识会有所帮助，但并不是必需的。如果你的 Python 水平尚未达到这一程度，Python 文档中有一个有用的教程，能帮助你迅速入门：[`docs.python.org/3/tutorial/index.html`](https://docs.python.org/3/tutorial/index.html)。

本书的配套代码可以在 GitHub 上找到，地址为[`github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas-2nd-edition`](https://github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas-2nd-edition)。为了最大程度地从本书中受益，建议你在阅读每一章时，在 Jupyter 笔记本中进行跟随操作。我们将在*第一章*，*数据分析导论*中介绍如何设置环境并获取这些文件。请注意，如果需要，还可以参考 Python 101 笔记本，它提供了一个速成课程/复习资料：[`github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas-2nd-edition/blob/master/ch_01/python_101.ipynb`](https://github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas-2nd-edition/blob/master/ch_01/python_101.ipynb)。

最后，务必完成每章末尾的练习。有些练习可能相当具有挑战性，但它们会让你对材料的理解更加深入。每章练习的解答可以在[`github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas-2nd-edition/tree/master/solutions`](https://github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas-2nd-edition/tree/master/solutions)中找到，位于各自的文件夹内。

# 下载彩色图像

我们还提供了一份 PDF 文件，其中包含本书中使用的截图/图表的彩色图像。你可以在这里下载：[`static.packt-cdn.com/downloads/9781800563452_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781800563452_ColorImages.pdf)。

# 使用的约定

本书中使用了一些文本约定。

`文本中的代码`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL 和用户输入。以下是一个例子：“使用`pip`安装`requirements.txt`文件中的包。”

一段代码会如下所示。该行的开头将以`>>>`为前缀，接下来的行将以`...`为前缀：

```py
>>> df = pd.read_csv(
...     'data/fb_2018.csv', index_col='date', parse_dates=True
... )
>>> df.head()
```

任何没有前缀`>>>`或`...`的代码我们不会执行——它仅供参考：

```py
try:
    del df['ones']
except KeyError:
    pass # handle the error here
```

当我们希望将你的注意力引导到代码块的某一部分时，相关的行或项会被加粗显示：

```py
>>> df.price.plot(
...     title='Price over Time', ylim=(0, None)
... )
```

结果将显示在没有任何前缀的行中：

```py
>>> pd.Series(np.random.rand(2), name='random')
0 0.235793
1 0.257935
Name: random, dtype: float64
```

任何命令行输入或输出都如下所示：

```py
# Windows:
C:\path\of\your\choosing> mkdir pandas_exercises
# Linux, Mac, and shorthand:
$ mkdir pandas_exercises
```

**加粗**：表示新术语、重要词汇或屏幕上看到的词。例如，菜单或对话框中的词会以这种方式出现在文本中。以下是一个例子：“使用**文件浏览器**窗格，双击**ch_01**文件夹，该文件夹包含我们将用来验证安装的 Jupyter Notebook。”

提示或重要注意事项

以这种方式显示。

# 联系我们

我们始终欢迎读者的反馈。

`customercare@packtpub.com`。

**勘误表**：尽管我们已尽力确保内容的准确性，但错误仍然会发生。如果你在本书中发现错误，我们将不胜感激，如果你能向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择你的书籍，点击“勘误提交表单”链接，并填写相关细节。

`copyright@packt.com`，并链接到该材料。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有兴趣写作或为书籍做贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 评审

请留下评论。一旦你阅读并使用了本书，为什么不在你购买的站点上留下评论呢？潜在的读者可以看到并利用你的公正意见来做出购买决策，我们在 Packt 可以了解你对我们产品的看法，而我们的作者也能看到你对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问[packt.com](http://packt.com)。
