# 第十二章：从哪里开始

这本书就像一次大胆的山地徒步旅行一样，是一段旅程……但现在，我们终于到达了旅行的终点。我们现在站在入门级 GPU 编程的山顶上，我们自豪地回望我们的故乡——串行编程城，当我们想到我们旧的一维编程传统是多么的纯真时，我们会微笑。我们勇敢地克服了许多陷阱和危险才到达这个点，我们甚至可能犯了一些错误，比如在 Linux 中安装了损坏的 NVIDIA 驱动模块，或者在我们假期探望父母时，通过缓慢的 100k 连接下载了错误的 Visual Studio 版本。但这些挫折只是暂时的，留下的伤口变成了老茧，使我们更坚强地对抗（GPU）自然的力量。

但是，在我们眼角的余光中，我们看到了两块木制标志，离我们站立的地方几米远；我们转移目光，从我们过去的那个小村庄转向它们。第一个标志上有一个箭头指向我们现在面对的方向，上面只有一个词——过去。另一个标志指向相反的方向，上面也只有一个词——未来。我们转身朝向“未来”的方向，我们看到一个巨大的闪耀着光芒的大都市在我们面前延伸到地平线，向我们招手。现在我们终于喘过气来，我们可以开始走向未来……

在本章中，我们将回顾一些你现在可用的选项，以便你能够继续在 GPU 编程相关领域进行教育和职业发展。无论你是试图建立职业生涯，一个为了乐趣而从事这项工作的爱好者，一个为了课程学习 GPU 的工程学生，一个试图增强自己技术背景的程序员或工程师，或者一个试图将 GPU 应用于研究项目的学术科学家，你现在都有很多很多的选择。就像我们的比喻性大都市一样，很容易迷路，很难确定我们应该去哪里。我们希望在这一章中提供类似简短导游的东西，为你提供一些你可以继续前进的选项。

在本章中，我们将探讨以下路径：

+   高级 CUDA 和 GPGPU 编程

+   图形

+   机器学习和计算机视觉

+   区块链技术

# 深入了解 CUDA 和 GPGPU 编程

你拥有的第一个选择当然是学习更多关于 CUDA 以及特别地**通用型 GPU 编程**（**GPGPU**）。在这种情况下，你可能已经找到了这个领域的良好应用，并希望编写更加高级或优化的 CUDA 代码。你可能仅仅因为兴趣而觉得这很有趣，或者你可能想要成为一名 CUDA/GPU 程序员。在有了这本书提供的强大的 GPU 编程基础之后，我们现在将探讨这个领域中的一些高级主题，这些主题我们现在已经准备好去学习了。

# 多 GPU 系统

第一个想到的主要话题可能是学习如何编程安装了多个 GPU 的系统。许多专业工作站和服务器都安装了多个 GPU，目的是处理远超过单个顶级 GPU 所能处理的数据量。为此，存在一个被称为多 GPU 编程的子领域。大部分工作都集中在负载均衡上，这是一种使用每个 GPU 达到其峰值容量的艺术，确保没有 GPU 因为过多的工作而饱和，而其他 GPU 则没有得到充分利用。这里的另一个话题是 GPU 间通信，它通常关注的是使用 CUDA 的 GPUDirect **对等**（**P2P**）内存访问，一个 GPU 直接将内存数组复制到或从另一个 GPU 的问题。

NVIDIA 在这里提供了一个关于多 GPU 编程的简要介绍：[`www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf`](https://www.nvidia.com/docs/IO/116711/sc11-multi-gpu.pdf)。

# 集群计算和 MPI

另一个话题是集群计算，即编写利用包含 GPU 的众多服务器集体使用的程序。这些是填充了知名互联网公司如 Facebook 和 Google 数据处理设施以及政府和军队使用的科学超级计算设施的**服务器农场**。集群通常使用一种称为**消息传递接口**（**MPI**）的编程范式进行编程，这是一个接口，用于与 C++或 Fortran 等语言一起使用，允许你编程连接到同一网络的多台计算机。

关于使用 CUDA 与 MPI 的更多信息，请参阅此处：[`devblogs.nvidia.com/introduction-cuda-aware-mpi/`](https://devblogs.nvidia.com/introduction-cuda-aware-mpi/)。

# OpenCL 和 PyOpenCL

CUDA 不是唯一可以用来编程 GPU 的语言。CUDA 最主要的竞争对手被称为开放计算语言，或 OpenCL。CUDA 是一个封闭且专有的系统，仅能在 NVIDIA 硬件上运行，而 OpenCL 是一个由非营利组织 Khronos Group 开发和支持的开放标准。OpenCL 可以用来编程不仅限于 NVIDIA GPU，还包括 AMD Radeon GPU 和英特尔 HD GPU——大多数主要科技公司都承诺在其产品中支持 OpenCL。此外，PyCUDA 的作者、UIUC 的安德烈亚斯·克洛克纳教授还编写了另一个优秀的（且免费）Python 库，名为 PyOpenCL，它提供了与 PyCUDA 几乎相同的语法和概念，为 OpenCL 提供了一个同样用户友好的接口。

关于 OpenCL 的信息由 NVIDIA 提供：[`developer.nvidia.com/opencl`](https://developer.nvidia.com/opencl).

关于免费 PyOpenCL 库的信息可以从安德烈亚斯·克洛克纳的网站获取：

[`mathema.tician.de/software/pyopencl/`](https://mathema.tician.de/software/pyopencl/).

# 图形

显然，GPU 中的 **G** 代表 **图形**，在这本书中我们并没有看到很多关于它的内容。尽管机器学习应用现在是 NVIDIA 的主要业务，但一切始于制作漂亮的图形。我们将在此提供一些资源，无论您是想开发视频游戏引擎、渲染 CGI 电影，还是开发 CAD 软件，都可以从这里开始。CUDA 实际上可以与图形应用结合使用，实际上它被用于 Adobe 的 Photoshop 和 After Effects 等专业软件，以及许多最近的游戏，如 *Mafia* 和 *Just Cause* 系列。我们将简要介绍一些您可能考虑从这里开始的主要 API。

# OpenGL

开放图形语言，或 OpenGL，是一个自 90 年代初就存在的行业开放标准。虽然它在某些方面显示出其年代感，但它是一个稳定的 API，得到了广泛的支持，如果您编写了一个利用这个 API 的程序，那么它在任何相对现代的 GPU 上都能保证运行。CUDA 样例文件夹实际上包含了大量 OpenGL 如何与 CUDA 交互的示例（特别是在 `2_Graphics` 子目录中），感兴趣的读者可以考虑查看这些示例。（在 Windows 中的默认位置是 `C:\ProgramData\NVIDIA Corporation\CUDA Samples`，在 Linux 中的默认位置是 `/usr/local/cuda/samples`。）

关于 OpenGL 的信息可以直接从 NVIDIA 的网站获取：[`developer.nvidia.com/opengl`](https://developer.nvidia.com/opengl).

PyCUDA 还提供了一个用于 NVIDIA OpenGL 驱动程序的接口。相关信息请在此处查看：[`documen.tician.de/pycuda/gl.html`](https://documen.tician.de/pycuda/gl.html).

# DirectX 12

DirectX 12 是微软知名且受支持的图形 API 的最新版本。虽然它是为 Windows PC 和微软 Xbox 游戏机开发的专有技术，但这些系统显然拥有数亿用户的广泛安装基础。此外，除了 NVIDIA 显卡外，Windows PC 还支持各种 GPU，Visual Studio IDE 提供了极大的易用性。DirectX 12 实际上支持低级 GPGPU 编程概念，并可以利用多个 GPU。

微软的 DirectX 12 编程指南可在以下链接获取：[`docs.microsoft.com/en-us/windows/desktop/direct3d12/directx-12-programming-guide`](https://docs.microsoft.com/en-us/windows/desktop/direct3d12/directx-12-programming-guide).

# Vulkan

Vulkan 可以被视为 Khronos Group 开发的 DirectX 12 的开放等效版本，它是 OpenGL 的下一代继任者。除了 Windows 外，Vulkan 还支持 macOS、Linux，以及索尼 PlayStation 4、任天堂 Switch 和 Xbox One 游戏机。Vulkan 具有与 DirectX 12 相似的功能，如准 GPGPU 编程。Vulkan 正在为 DirectX 12 提供一些严重的竞争，例如 2016 年的 *DOOM* 重制游戏。

《*Vulkan 入门指南*》可在 Khronos Group 这里获取：[`www.khronos.org/blog/beginners-guide-to-vulkan`](https://www.khronos.org/blog/beginners-guide-to-vulkan).

# 机器学习和计算机视觉

当然，本章中不容忽视的是机器学习和其孪生兄弟计算机视觉。不言而喻，机器学习（尤其是深度神经网络和卷积神经网络等子领域）是如今支撑着 NVIDIA 首席执行官黄仁勋的屋顶的东西。（好吧，我们承认这是本世纪最夸张的夸张……）如果你需要提醒为什么 GPU 在这个领域如此适用和有用，请再次查看第九章，*深度神经网络的实现*。大量的并行计算和数学运算，以及用户友好的数学库，使得 NVIDIA GPU 成为机器学习行业的硬件支柱。

# 基础知识

虽然你现在已经了解了底层 GPU 编程的许多复杂性，但你无法立即将这些知识应用到机器学习上。如果你在这个领域没有基本技能，比如如何对数据集进行基本的统计分析，你真的应该停下来熟悉它们。斯坦福大学教授、谷歌大脑的创始人 Andrew Ng 提供了许多免费材料，这些材料可以在网上和 YouTube 上找到。Ng 教授的工作通常被认为是机器学习教育材料的黄金标准。

Ng 教授在网上提供了一门免费的机器学习入门课程：[`www.ml-class.org`](http://www.ml-class.org).

# cuDNN

NVIDIA 为深度神经网络原语提供了一个优化的 GPU 库，称为 cuDNN。这些原语包括前向传播、卷积、反向传播、激活函数（如 sigmoid、ReLU 和 tanh）和梯度下降等操作。cuDNN 是大多数主流深度神经网络框架（如 Tensorflow）用作 NVIDIA GPU 后端的东西。这是由 NVIDIA 免费提供的，但必须从 CUDA 工具包中单独下载。

关于 cuDNN 的更多信息在这里：[`developer.nvidia.com/cudnn`](https://developer.nvidia.com/cudnn)。

# Tensorflow 和 Keras

Tensorflow 当然是谷歌知名的神经网络框架。这是一个免费且开源的框架，可以用 Python 和 C++ 使用，自 2015 年以来对公众开放。

在这里可以找到 Tensorflow 的教程：[`www.tensorflow.org/tutorials/`](https://www.tensorflow.org/tutorials/)。

Keras 是一个更高级的库，它为 Tensorflow 提供了一个更*用户友好*的接口，最初由 Google Brain 的 Francois Chollet 编写。读者实际上可以考虑先从 Keras 开始，然后再转向 Tensorflow。

关于 Keras 的信息在这里：[`keras.io/`](https://keras.io/)。

# Chainer

Chainer 是由日本东京大学的博士生 Seiya Tokui 开发的另一个神经网络 API。虽然它的主流程度不如 Tensorflow，但由于其惊人的速度和效率，它非常受尊重。此外，读者可能会对 Chainer 特别感兴趣，因为它是最初使用 PyCUDA 开发的。（后来改为使用 CuPy，这是 PyCUDA 的一个分支，旨在提供一个与 NumPy 更相似的接口。）

关于 Chainer 的信息在这里：[`chainer.org/`](https://chainer.org/)。

# OpenCV

自 2001 年以来，开源计算机视觉库（OpenCV）一直存在。这个库提供了许多来自经典计算机视觉和图像处理工具，在深度神经网络时代仍然非常有用。近年来，OpenCV 中的大多数算法都已移植到 CUDA，并且它与 PyCUDA 非常容易接口。

关于 OpenCV 的信息在这里：[`opencv.org/`](https://opencv.org/)。

# 区块链技术

最后，但同样重要的是，是**区块链技术**。这是支撑比特币和以太坊等加密货币的底层加密技术。当然，这是一个非常新的领域，最初由比特币神秘的创造者中本聪在 2008 年发表的白皮书中描述。GPU 在发明后几乎立即被应用于这个领域——生成货币单位的过程归结为破解加密难题的暴力破解，而 GPU 可以并行尝试破解比今天公众可用的任何其他硬件更多的组合。这个过程被称为**挖矿**。

对区块链技术感兴趣的人建议阅读中本聪关于比特币的原始白皮书，可在以下链接找到：[`bitcoin.org/bitcoin.pdf`](https://bitcoin.org/bitcoin.pdf)。

GUIMiner，一个基于 CUDA 的开源比特币矿工，可在以下链接找到：[`guiminer.org/`](https://guiminer.org/)。

# 摘要

在本章中，我们讨论了那些对进一步学习 GPU 编程感兴趣的人的一些选项和路径，这些内容超出了本书的范围。我们首先讨论了扩展你在纯 CUDA 和 GPGPU 编程方面的背景——本书未涉及的一些你可以学习的内容包括编程具有多个 GPU 和网络集群的系统。我们还探讨了除 CUDA 之外的一些并行编程语言/API，例如 MPI 和 OpenCL。接下来，我们讨论了一些对那些有兴趣将 GPU 应用于图形渲染的人可用的知名 API，例如 Vulkan 和 DirectX 12。然后，我们探讨了机器学习，并介绍了一些你应该具备的基本背景以及一些用于开发深度神经网络的框架。最后，我们简要地看了看区块链技术以及基于 GPU 的加密货币挖矿。

作为作者，我想对推动这本书完成并到达这里的每一个人说**谢谢**，感谢你们的支持。GPU 编程是我遇到的最棘手的编程子领域之一，我希望我的文字能帮助你掌握基本知识。作为读者，你现在可以尽情享受你所能找到的最丰富、热量最高的巧克力蛋糕的一小块——只要知道你已经**赚到了**。 (但只能吃一小块！)

# 问题

1.  使用谷歌或其他搜索引擎查找本章未提及的至少一种 GPU 编程应用。

1.  尝试找到至少一种可以用来编程本章未提及的 GPU 的编程语言或 API。

1.  查找谷歌的新 Tensor Processing Unit (TPU)芯片。这些芯片与 GPU 有何不同？

1.  你认为使用 Wi-Fi 还是有线以太网电缆将计算机连接成集群更好？
