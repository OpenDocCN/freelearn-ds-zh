# 前言

问候和祝福！本文是使用 Python 和 CUDA 进行 GPU 编程的入门指南。**GPU**可能代表**图形编程单元**，但我们应该清楚，这本书**不是**关于图形编程——它本质上是对**通用 GPU 编程**，或简称**GPGPU 编程**的介绍。在过去的十年里，已经很明显，GPU 非常适合除了渲染图形之外的计算，尤其是对于需要大量计算吞吐量的并行计算。为此，NVIDIA 发布了 CUDA 工具包，这使得几乎任何有 C 编程知识的人都可以轻松进入 GPGPU 编程的世界。

《Python 和 CUDA 动手实践 GPU 编程》的目的是让您尽可能快地进入 GPGPU 编程的世界。我们努力为每一章提供有趣和有意义的示例和练习；特别是，我们鼓励您在阅读过程中输入这些示例并在您喜欢的 Python 环境中运行它们（Spyder、Jupyter 和 PyCharm 都是合适的选择）。这样，您最终将学会所有必要的函数和命令，并了解如何编写 GPGPU 程序。

初始时，GPGPU 并行编程看起来非常复杂和令人畏惧，尤其是如果你过去只做过 CPU 编程。你需要学习许多新的概念和约定，这可能会让你感觉像是从头开始。在这些时候，你必须相信你学习这个领域的努力不会白费。只要有一点主动性和纪律性，等你读完这本书的时候，这个主题对你来说就会像第二本能一样自然。

开心编程！

# 本书面向对象

本书特别针对某个人——那就是 2014 年的我，当时我正在尝试为我的数学博士研究开发基于 GPU 的模拟。我翻阅了多本关于 GPU 编程的书籍和手册，试图弄清楚这个领域的最基本概念；大多数文本似乎很高兴在每一页上向读者展示无穷无尽的硬件原理和术语，而实际的*编程*则被放在次要位置。

本书主要面向那些真正想进行*GPU 编程*的人，但又不希望被繁琐的技术细节和硬件原理所困扰。在本文中，我们将使用正确的 C/C++（CUDA C）来编程 GPU，但我们将通过 PyCUDA 模块将其*内联*在 Python 代码中。PyCUDA 允许我们只编写所需的底层 GPU 代码，而它将自动为我们处理编译、链接和将代码加载到 GPU 上的所有冗余工作。

# 本书涵盖内容

第一章，*为什么进行 GPU 编程？*，给出了我们学习这个领域的一些动机，以及如何应用 Amdahl 定律来估计将串行程序转换为利用 GPU 可能带来的性能提升。

第二章，*设置您的 GPU 编程环境*，解释了如何在 Windows 和 Linux 下设置适合 CUDA 的 Python 和 C++开发环境。

第三章，*开始使用 PyCUDA*，展示了我们从 Python 编程 GPU 所需的最基本技能。我们将特别看到如何使用 PyCUDA 的 gpuarray 类将数据传输到和从 GPU，以及如何使用 PyCUDA 的 ElementwiseKernel 函数编译简单的 CUDA 内核。

第四章，*内核、线程、块和网格*，教授编写有效的 CUDA 内核的基础知识，这些内核是启动在 GPU 上的并行函数。我们将看到如何编写 CUDA 设备函数（由 CUDA 内核直接调用的“串行”函数），以及了解 CUDA 的抽象网格/块结构及其在启动内核中的作用。

第五章，*流、事件、上下文和并发*，涵盖了 CUDA 流的观念，这是一个允许我们并发地在 GPU 上启动和同步多个内核的功能。我们将看到如何使用 CUDA 事件来计时内核启动，以及如何创建和使用 CUDA 上下文。

第六章，*调试和性能分析您的 CUDA 代码*，填补了我们在纯 CUDA C 编程方面的空白，并展示了如何使用 NVIDIA Nsight IDE 进行调试和开发，以及如何使用 NVIDIA 性能分析工具。

第七章，*使用 Scikit-CUDA 中的 CUDA 库*，通过 Python Scikit-CUDA 模块简要介绍了几个重要的标准 CUDA 库，包括 cuBLAS、cuFFT 和 cuSOLVER。

第八章，*CUDA 设备函数库和 Thrust*，展示了如何在我们的代码中使用 cuRAND 和 CUDA Math API 库，以及如何使用 CUDA Thrust C++容器。

第九章，*深度神经网络实现*，作为一个总结，我们学习如何从头开始构建整个深度神经网络，应用我们在文本中学到的许多想法。

第十章，*与编译后的 GPU 代码一起工作*，展示了如何使用 PyCUDA 和 Ctypes 将我们的 Python 代码与预编译的 GPU 代码接口。

第十一章，*CUDA 中的性能优化*，教授了一些非常底层的性能优化技巧，特别是与 CUDA 相关，如 warp 混洗、向量内存访问、使用内联 PTX 汇编和原子操作。

第十二章，*从这里开始*，概述了一些教育职业道路，这些道路将建立在您现在在 GPU 编程方面的坚实基础之上。

# 为了充分利用这本书

这实际上是一个相当技术性的主题。为此，我们不得不对读者的编程背景做一些假设。为此，我们将假设以下：

+   您在 Python 中具有中级编程经验。

+   您熟悉标准的 Python 科学包，如 NumPy、SciPy 和 Matplotlib。

+   您在任何基于 C 的编程语言（C、C++、Java、Rust、Go 等）中具有中级能力。

+   您理解 C 语言中动态内存分配的概念（特别是如何使用 C 的`malloc`和`free`函数。）

GPU 编程主要适用于那些非常科学或数学性质较强的领域，因此，许多（如果不是大多数）示例都将使用一些数学知识。因此，我们假设读者对大学一年级或二年级的数学有一定了解，包括：

+   三角学（正弦函数：sin、cos、tan 等）

+   微积分（积分、导数、梯度）

+   统计学（均匀分布和正态分布）

+   线性代数（向量、矩阵、向量空间、维度）

如果您没有学习过这些主题，或者已经有一段时间没有学习了，请不要担心，因为我们将随着我们的进展尝试回顾一些关键的编程和数学概念。

在这里，我们还将做出另一个假设。记住，我们将只使用 CUDA，这是一种 NVIDIA 硬件的专有编程语言。因此，在我们开始之前，我们需要拥有一些特定的硬件。所以，我将假设读者可以访问以下：

+   基于 64 位 x86 Intel/AMD 的 PC

+   4 吉字节（GB）或更多的 RAM

+   入门级 NVIDIA GTX 1050 GPU（帕斯卡架构）或更好

读者应该知道，大多数较老的 GPU 将可能适用于本文中的大多数，如果不是所有示例，但本文中的示例仅在 Windows 10 下的 GTX 1050 和 Linux 下的 GTX 1070 上进行了测试。有关设置和配置的详细说明请参阅第二章，*设置您的 GPU 编程环境*。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载和勘误表。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载完成后，请确保您使用最新版本的以下软件解压或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

书籍的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781788993913_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781788993913_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“我们现在可以使用`cublasSaxpy`函数。”

代码块设置如下：

```py
cublas.cublasDestroy(handle)
print 'cuBLAS returned the correct value: %s' % np.allclose(np.dot(A,x), y_gpu.get())
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
def compute_gflops(precision='S'):

if precision=='S':
    float_type = 'float32'
elif precision=='D':
    float_type = 'float64'
else:
    return -1
```

任何命令行输入或输出都按以下方式编写：

```py
$ run cublas_gemm_flops.py
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。

警告或重要提示看起来像这样。

技巧和窍门看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`将邮件发送给我们。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，我们将不胜感激，如果您能向我们报告，请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书，点击勘误提交表链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
