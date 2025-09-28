# 附录 B：在云端利用 Jupyter 笔记本

在本附录中，我们将涵盖以下主题：

+   Jupyter Notebook

+   Google Colaboratory

+   IBM 量子实验室

+   伴随 Jupyter 笔记本

## Jupyter Notebook

Jupyter Notebook 是一个免费的 Web 应用程序，用于创建和共享结合 Markdown 格式的可执行代码和叙述文本的计算文档 [Jupyter_0]。它提供了一个简单、流畅、以文档为中心的体验。Project Jupyter 是一个非营利性、开源项目。

## Google Colaboratory

Google Colaboratory（或简称 Colab）是一个完全在云端运行的免费 Jupyter Notebook 环境，无需下载或安装任何软件即可提供共享的在线 Jupyter 笔记本实例 [Colab_0] [Colab_1]。您只需拥有一个有效的 Gmail 账户即可保存和访问 Google Colab Jupyter 笔记本。

## IBM Quantum Lab

IBM Quantum Lab 是一个无需安装的云启用 Jupyter 笔记本环境 [IBM_QLab0] [IBM_QLab1]。IBM Quantum Composer 是一个图形量子编程工具，允许您拖放操作来构建量子电路并在真实量子硬件或模拟器上运行它们 [IBM_comp1] [IBM_comp2]。它允许公众免费访问 IBM 提供的基于云的量子计算服务。在 Quantum Lab 中，您可以在定制的 Jupyter Notebook 环境中编写结合 Qiskit 代码、方程、可视化和叙述文本的脚本。

## 伴随 Jupyter 笔记本

我们在此处提供了一个 GitHub 仓库，其中包含本书的伴随 Jupyter 笔记本：https://github.com/PacktPublishing/Quantum-Chemistry-and-Computing-for-the-Curious。这些伴随笔记本会自动安装相关库，如下所示：

+   **数值 Python**（**NumPy**）[NumPy]，一个开源的 Python 库，几乎在科学和工程的每个领域都得到应用。

+   Qiskit [Qiskit]，一个开源 SDK，用于在脉冲、电路和应用模块级别与量子计算机交互。

+   Qiskit 可视化支持，以启用可视化功能和使用 Jupyter 笔记本。

+   Qiskit Nature [Qiskit_Nature] [Qiskit_Nat_0]，一个独特的平台，用于弥合自然科学与量子模拟之间的差距。

+   **基于 Python 的化学模拟框架**（**PySCF**）[PySCF]是一个由 Python 驱动的开源电子结构模块集合。

+   **量子工具箱在 Python 中**（**QuTiP**）[QuTiP]旨在成为一个解决量子力学问题的一般框架，例如由少量量子系统和谐振子组成的系统。

+   **原子模拟环境**（**ASE**）[ASE_0]，一套用于设置、操作、运行、可视化和分析原子模拟的工具和 Python 模块。代码在 GNU LGPL 许可下免费提供。

+   PyQMC [PyQMC]，一个实现实空间量子蒙特卡洛技术的 Python 模块。它主要用于与 PySCF 交互。

+   h5py [H5py] 包，一个用于 HDF5 二进制数据格式的 Python 接口。

+   SciPy [SciPy_0]，一个用于科学计算和技术计算的免费开源 Python 库。SciPy 提供了优化、积分、插值、特征值问题、代数方程、微分方程、统计学以及许多其他类别的算法。

+   SymPy [SymPy]，一个用于符号数学的 Python 库。

所有配套的 Jupyter 笔记本在 Google Colab 和量子实验室环境中都运行成功。

第六章“超越玻恩-奥本海默”的配套 Jupyter 笔记本不包括安装 Psi4，这是一种用于高通量量子化学的开源软件 [Psi4_0]，我们曾用它来对二氧化碳 (CO2) 分子的振动频率进行分析进行简单计算。我们建议对安装此软件包感兴趣的读者查阅“开始使用 Psi4” [Psi4_1] 文档和参考文献 [Psi4_3]。

# 参考文献

[ASE_0] 原子模拟环境 (ASE)，[`wiki.fysik.dtu.dk/ase/index.html`](https://wiki.fysik.dtu.dk/ase/index.html)

[Colab_0] 欢迎来到 Colaboratory，Google Colab 常见问题解答，[`research.google.com/colaboratory/faq.html`](https://research.google.com/colaboratory/faq.html)

[Colab_1] 欢迎来到 Colaboratory，[`colab.research.google.com/`](https://colab.research.google.com/)

[H5py] 快速入门指南，[`docs.h5py.org/en/stable/quick.html`](https://docs.h5py.org/en/stable/quick.html)

[IBM_QLab0] IBM 量子实验室，[`quantum-computing.ibm.com/lab`](https://quantum-computing.ibm.com/lab)

[IBM_QLab1] 欢迎来到量子实验室，[`quantum-computing.ibm.com/lab/docs/iql/`](https://quantum-computing.ibm.com/lab/docs/iql/)

[IBM_comp1] 欢迎来到 IBM 量子作曲家，[`quantum-computing.ibm.com/composer/docs/iqx/`](https://quantum-computing.ibm.com/composer/docs/iqx/)

[IBM_comp2] IBM 量子作曲家，[`quantum-computing.ibm.com/composer/files/new`](https://quantum-computing.ibm.com/composer/files/new)

[Jupyter_0] Jupyter，[`jupyter.org/`](https://jupyter.org/)

[NumPy] NumPy：初学者的绝对基础，[`numpy.org/doc/stable/user/absolute_beginners.html`](https://numpy.org/doc/stable/user/absolute_beginners.html)

[Psi4_0] Psi4 手册主索引，[`psicode.org/psi4manual/master/index.html`](https://psicode.org/psi4manual/master/index.html%20)

[Psi4_1] 开始使用 PSI4，[`psicode.org/installs/v15/`](https://psicode.org/installs/v15/%20)

[Psi4_3] Smith DGA, Burns LA, Simmonett AC, Parrish RM, Schieber MC, Galvelis R, Kraus P, Kruse H, Di Remigio R, Alenaizan A, James AM, Lehtola S, Misiewicz JP, Scheurer M, Shaw RA, Schriber JB, Xie Y, Glick ZL, Sirianni DA, O'Brien JS, Waldrop JM, Kumar A, Hohenstein EG, Pritchard BP, Brooks BR, Schaefer HF 3rd, Sokolov AY, Patkowski K, DePrince AE 3rd, Bozkaya U, King RA, Evangelista FA, Turney JM, Crawford TD, Sherrill CD, Psi4 1.4: 开源软件用于高通量量子化学，J Chem Phys. 2020 年 5 月 14 日;152(18):184108\. doi: 10.1063/5.0006002\. PMID: 32414239; PMCID: PMC7228781, [`www.ncbi.nlm.nih.gov/pmc/articles/PMC7228781/pdf/JCPSA6-000152-184108_1.pdf`](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7228781/pdf/JCPSA6-000152-184108_1.pdf%20)

[PyQMC] PyQMC，一个实现实空间量子蒙特卡罗技术的 Python 模块，[`github.com/WagnerGroup/pyqmc`](https://github.com/WagnerGroup/pyqmc)

[PySCF] 基于 Python 的化学模拟框架（PySCF），[`pyscf.org/`](https://pyscf.org/)

[Qiskit] Qiskit，[`qiskit.org/`](https://qiskit.org/)

[Qiskit_Nat_0] Qiskit_Nature，[`github.com/Qiskit/qiskit-nature/blob/main/README.md`](https://github.com/Qiskit/qiskit-nature/blob/main/README.md)

[Qiskit_Nature] 介绍 Qiskit Nature，Qiskit，Medium，2021 年 4 月 6 日，[`medium.com/qiskit/introducing-qiskit-nature-cb9e588bb004`](https://medium.com/qiskit/introducing-qiskit-nature-cb9e588bb004)

[QuTiP] QuTiP，在布洛赫球上的绘图，[`qutip.org/docs/latest/guide/guide-bloch.html`](https://qutip.org/docs/latest/guide/guide-bloch.html)

[SciPy_0] SciPy，Python 符号数学库，[`scipy.org/`](https://scipy.org/)

[SymPy] SymPy，Python 符号数学库，[`www.sympy.org/en/index.html`](https://www.sympy.org/en/index.html)
