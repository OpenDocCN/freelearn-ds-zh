# 8

# 结论和下一步

您可以将 iSAX 用作传统索引或更复杂的东西，就像我们在*第七章*中所做的那样。通过这种 iSAX 的实际应用，我们已到达本书的最后一章！感谢您阅读这本书，这是一本多人合作的成果，而不仅仅是作者的个人努力。

时序和时序数据挖掘，在学术界和工业界都是热门话题，主要是因为如今数据通常以时序方式出现。似乎这还不够，这些时序数据中包含大量我们需要快速且准确地处理的数据。

本章将为您提供方向，如果您真正对时序或数据库感兴趣，您将了解接下来应该关注哪里以及什么。

**免责声明**：所有提到的书籍和研究论文都是个人喜好。您的个人品味或研究兴趣可能有所不同。

在本章中，我们将涵盖以下主要主题：

+   总结到目前为止我们所学的所有内容

+   iSAX 的其他变体

+   关于时序的有趣研究论文

+   关于数据库的有趣研究论文

+   有用书籍

# 总结到目前为止我们所学的所有内容

时序无处不在！但随着我们收集更多数据，时序数据往往会变得越来越大。因此，我们需要更快地处理和搜索大型时序数据，以便从数据中得出有用的结论。

iSAX 索引旨在帮助您快速搜索您的时序数据。我希望这本书已经为您提供了开始使用时序和子序列，以及 Python 中的 iSAX 索引所必需的工具和知识。然而，这些知识和所展示的技术很容易转移到其他编程语言，包括但不限于*Swift*、Java、C、*C++*、Ruby、Kotlin、*Go*、*Rust*和*JavaScript*。

我们相信，我们已经提供了适量的关于使用适当理论和实践进行时序索引的知识，以便您能够成功处理时序并开发 iSAX 索引。

下一个部分将介绍 iSAX 的改进版本。

# iSAX 的其他变体

本书已向您介绍了 iSAX 的初始形式。存在更多 iSAX 的变体，使它的操作和构建更快。然而，核心功能保持不变。您可以通过阅读以下研究论文来深入了解：

+   *iSAX 2.0：使用 iSAX 索引和挖掘十亿个时序*，由 Alessandro Camerra、Themis Palpanas、Jin Shieh 和 Eamonn Keogh 撰写

+   *超越十亿个时序：使用 iSAX2+索引和挖掘非常大的时序数据集*，由 Alessandro Camerra、Jin Shieh、Themis Palpanas、Thanawin Rakthanmanon 和 Eamonn Keogh 撰写

+   *DPiSAX：大规模分布式分区 iSAX*，由 Djamel Edine Yagoubi、Reza Akbarinia、Florent Masseglia 和 Themis Palpanas 撰写

+   *《数据系列索引的演变：iSAX 系列数据系列索引：iSAX、iSAX2.0、iSAX2+、ADS、ADS+、ADS-Full、ParIS、ParIS+、MESSI、DPiSAX、ULISSE、Coconut-Trie/Tree、Coconut-LSM》*，由 Themis Palpanas 撰写

请记住，这些都是高级研究论文，你可能一开始会觉得难以理解。然而，如果你坚持不懈，你最终会理解它们的。

在数据库领域，存在许多索引，因为索引对于快速回答 SQL 查询是必不可少的。即使它们与时间序列没有直接关联，你也可能想看看它们，并调整它们以便与时间序列和子序列一起工作。

一个非常著名的索引称为**R 树**，它是一种基于**B+树**的层次数据结构。你可以通过阅读 Antonin Guttman 撰写的《R-trees: A Dynamic Index Structure for Spatial Searching》来了解更多关于 R 树索引的信息。

最后，在撰写这本书的过程中，出现了一个新的基于 SAX 表示的时间序列索引，它试图纠正 iSAX 的缺点。这个新索引的名称是**Dumpy**。Dumpy 背后的核心逻辑与 iSAX 相同，但在构建索引的过程中进行了一些调整，以便在索引内部有更好的子序列分布。

你可以通过阅读[`arxiv.org/abs/2304.08264`](https://arxiv.org/abs/2304.08264)上的论文来了解更多关于 Dumpy 的信息。

下一节将提到一些关于时间序列的有趣研究论文。

# 时间序列有趣的论文

这里有一份关于时间序列聚类和**异常检测**的研究论文列表，你可能觉得很有趣：

+   *《时间序列数据中的异常/离群值检测综述》*，由 Ane Blazquez-Garcia、Angel Conde、Usue Mori 和 Jose A. Lozano 撰写

+   *《时间序列中的异常检测：全面评估》*，由 Sebastian Schmidl、Phillip Wenig 和 Thorsten Papenbrock 撰写

+   *《时间序列异常检测技术综述：迈向未来展望》*，由 Kamran Shaukat、Talha Mahboob Alam、Suhuai Luo、Shakir Shabbir、Ibrahim A. Hameed、Jiaming Li、Syed Konain Abbas 和 Umair Javed 撰写

+   *《时间序列聚类——十年回顾》*，由 Saeed Aghabozorgi、Ali Seyed Shirkhorshidi 和 Teh Ying Wah 撰写

+   *《离散序列的异常检测：综述》*，由 Varun Chandola、Arindam Banerjee 和 Vipin Kumar 撰写

下一节将提到一些关于数据库的有趣研究论文。

# 数据库有趣的论文

由于时间序列与数据库相关联，我将给你一个关于数据库的经典研究论文的小列表：

+   *《全局查询优化》*，由 Timos K. Sellis 撰写

+   *《POSTGRES 的设计》*，由 Michael Stonebraker 和 Lawrence A. Rowe 撰写

+   *《INGRES 的设计与实现》*，由 Michael Stonebraker、Gerald Held、Eugene Wong 和 Peter Kreps 撰写

+   *《大型共享数据库的关系数据模型》*，由 E. F. Codd 撰写

+   *《西雅图数据库研究报告*》，可在[`dl.acm.org/doi/10.1145/3524284`](https://dl.acm.org/doi/10.1145/3524284)找到

下一节将提出一些关于数据库的有趣且宝贵的书籍。

# 有用书籍

在本书的最后一部分，我将列出与计算机科学和软件工程相关的有用书籍，从数据库领域开始。

## 数据库有用书籍

由于时间序列与数据库相关联，我将为您提供一个关于数据库的经典书籍小清单：

+   *《数据库系统阅读材料，第 4 版*》，由 Joseph M. Hellerstein 和 Michael Stonebraker 编辑

+   *《数据库内部机制*》，由 Alex Petrov 撰写

+   *《数据挖掘导论，第 2 版*》，由 Pang-Ning Tan、Michael Steinbach、Anuj Karpatne 和 Vipin Kumar 合著

+   *《数据库系统：完整指南，第 2 版*》，由 Hector Garcia-Molina、**杰弗里·D·乌尔曼**和 Jennifer Widom 合著

+   *《数据库管理系统，第 3 版*》，由 Raghu Ramakrishnan 和 Johannes Gehrke 合著

数据库和时间序列并非与操作系统和计算机编程孤立。因此，在处理数据库时，拥有强大的计算机科学背景将大有裨益。下一小节将介绍一些有助于此的书籍。

## 建立强大的计算机科学背景

本小节将介绍一些有助于您建立强大计算机科学背景的书籍。以下列出了以下书籍：

+   *《算法导论，第 4 版*》，由 Thomas H. Cormen、Charles E. Leiserson、Ronald L. Rivest 和 Clifford Stein 合著

+   *《编程珠玑，第 2 版*》，由**Jon Bentley**撰写

+   *《更多编程珠玑：程序员的自白*》，由 Jon Bentley 撰写

+   *《代码大全：软件构造实用手册*》，由**史蒂夫·麦克康奈尔**撰写

+   *《编写解释器*》，由 Robert Nystrom 撰写

+   *《算法设计手册，第 3 版*》，由 Steven S. Skiena 撰写

+   *《统计学习元素：数据挖掘、推理和预测，第 2 版*》，由 Trevor Hastie、Robert Tibshirani 和 Jerome Friedman 合著

+   *《编译原理、技术和工具，第 2 版*》，由**阿尔弗雷德·阿霍**、Jeffrey Ullman、**拉维·塞西**和 Monica Lam 合著

+   *《用 Go 编写编译器*》，由 Thorsten Ball 撰写

您不必从头到尾阅读每一页。然而，这份书单将为您在计算机科学领域打下坚实的基础。

下一小节将提出一些能帮助您成为更好的 UNIX 和 Linux 开发者和高级用户的书籍。

## UNIX 和 Linux 相关书籍

本小节将介绍一些与 UNIX 和 Linux 操作系统相关的书籍。以下列出了以下书籍：

+   *《UNIX 编程环境*》，由**Brian W. Kernighan**和 Rob Pike 合著

+   *《编程实践*》，由 Brian W. Kernighan 和 Rob Pike 合著

+   *《UNIX 实用工具*》，由 Shelley Powers、Jerry Peek、Tim O’Reilly 和 Mike Loukides 合著

+   *《UNIX 编程环境高级编程（第三版）》，作者**W. Richard Stevens**和 Stephen Rago*

+   *《UNIX 网络编程》，作者 W. Richard Stevens*

+   *《C 程序设计语言（第二版）》，作者 Brian W. Kernighan 和**丹尼斯** **M. Ritchie**

下一个小节将介绍与 Python 编程语言相关的实用书籍。

## Python 编程语言书籍

本小节介绍了与 Python 编程语言相关的书籍。以下列出了以下书籍：

+   *《流畅 Python》，作者 Luciano Ramalho*

+   *《Effective Pandas》，作者 Matt Harrison*

+   *《精通 Python：利用 Python 的全部功能编写强大而高效的代码（第二版）》，作者 Rick van Hattem*

+   *《专家 Python 编程：通过学习最佳编码实践和高级编程概念掌握 Python（第四版）》，作者 Michal Jaworski 和 Tarek Ziade*

+   *《使用 Python 进行时间序列分析食谱》，作者 Tarek A. Atwan*

+   *《Python 数据分析（第三版）》，作者 Avinash Navlani、Armando Fandango 和 Ivan Idris*

+   *《使用 PyTorch 和 Scikit-Learn 进行机器学习：用 Python 开发机器学习和深度学习模型》，作者 Sebastian Raschka、Yuxi (Hayden) Liu 和 Vahid Mirjalili*

有了这个，我们就完成了你可以从中受益的书籍列表。

# 摘要

在这本书的最后一章，我们列出了一份长长的有趣书籍和研究论文列表。时间序列、时间序列数据挖掘以及数据库，总的来说，都是一些有趣的领域，如果你以尊重的态度对待它们，并持续实验和学习新事物，这些领域将使你终身忙碌。我可以向你保证，在这些领域里，无论是学术界还是工业界，你都不会感到无聊。

如果你从这本书中只保留一个要点，那就是要**始终保持好奇心和实验精神**。

然而，第一步是找到你最感兴趣的事情，并跟随那个方向。如果你打算在某个事物上花费大量时间，你肯定应该找到它既有趣又具有挑战性！

非常感谢您选择这本书。如果您有任何关于改进潜在未来版本的书籍的建议，请随时提出。您的评论和建议可能会产生差异！

# 有用链接

+   *《异常分析》，作者 Charu C. Aggarwal，Springer，2013*

+   `darts` Python 包：[`pypi.org/project/darts/`](https://pypi.org/project/darts/)

+   *《R-Trees：理论与应用》，作者 Yannis Manolopoulos、Alexandros Nanopoulos、Apostolos N. Papadopoulos 和 Yannis Theodoridis*

+   流畅 Python：[`www.fluentpython.com/`](https://www.fluentpython.com/)

+   使用矩阵轮廓进行时间序列数据挖掘第一部分：[`www.youtube.com/watch?v=1ZHW977t070`](https://www.youtube.com/watch?v=1ZHW977t070)

+   使用矩阵轮廓进行时间序列数据挖掘第二部分：[`www.youtube.com/watch?v=LnQneYvg84M`](https://www.youtube.com/watch?v=LnQneYvg84M)

+   切比雪夫多项式：[`en.wikipedia.org/wiki/Chebyshev_polynomials`](https://en.wikipedia.org/wiki/Chebyshev_polynomials)

+   `numba` Python 包：[`pypi.org/project/numba/`](https://pypi.org/project/numba/)

+   iSAX 页面：[`www.cs.ucr.edu/~eamonn/iSAX/iSAX.xhtml`](https://www.cs.ucr.edu/~eamonn/iSAX/iSAX.xhtml)

+   计算机理论：[`en.wikipedia.org/wiki/Theory_of_computation`](https://en.wikipedia.org/wiki/Theory_of_computation)

# 练习

尝试以下操作：

+   作为练习，了解`pandas.read_csv()`函数支持的压缩文件类型。

+   如果你时间足够，尝试用 Go、Swift 或 Rust 等其他编程语言实现 iSAX 索引。

+   如果你真的很喜欢 Python，你可以尝试优化`isax`包的代码。

+   这是一个**非常困难**的练习：你可以尝试通过允许使用`numba` Python 包来提高 iSAX 索引的搜索性能。我个人无法用 Python 编写这样的程序。

+   这又是一个**非常困难**的练习：尝试创建一个在**你的 GPU**上运行的搜索算法的并行版本！如果你做到了，请告诉我！
