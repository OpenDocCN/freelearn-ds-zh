# 第七章：结论

“我没有特殊的天赋。我只是对知识充满热情。”

– 阿尔伯特·爱因斯坦

![](img/B18268_Figure_7.1.jpg)

图 7.1 – 量子化学与量子计算之间的循环依赖关系 [作者]

# 7.1\. 量子计算

计算的初始量子电路模型作为逻辑量子门按时间顺序执行量子寄存器状态的单位演化序列，已经演变成一个紧密集成的动态量子电路模型，允许对中电路测量结果进行并发经典处理 [Corcoles] [IBM_mid]。这种新的动态量子计算范式为从经典计算到量子增强计算的平稳过渡铺平了道路。

一个经常被忽视的潜在量子优势是量子计算的能源效率 [Auffeves] [Q Daily] [Quantum_AI] [Thibault]。量子霸权实验 [Arute] 中的能耗比高性能计算机实现实验预期结果的精确计算所需的能耗小三个数量级。确定能够实现能源量子优势的条件是一个开放的研究课题 [Auffeves]。

最近展示的新量子寄存器对费米子对，其中信息存储在处于两个振动状态叠加的原子对的振动运动中，为分子可编程量子模拟器开辟了前景 [Hartke]。分子振动光谱的精确计算，这些计算在从天体化学到生物化学以及气候变化缓解中都有应用，可能在量子计算机上比电子能量计算更容易实现 [Sawaya]。

# 7.2\. 量子化学

在*第四章**，分子哈密顿量* 和 *第五章**，变分量子本征值求解器（VQE）算法* 中，我们使用 Python 和开源量子化学软件包 PySCF、ASE、PyQMC 和 Qiskit 展示了量子计算化学的一些方法，求解了氢分子、氢化锂分子和宏分子 ONCHHC 的基态能量水平，并绘制了这些分子的 BOPES。使用最简单的基组（STO-3G）和噪声无模拟的量子电路（状态向量模拟器），不同的计算方法得到了良好的吻合。

科学或工业应用需要化学反应机制的高度精确相对能量估计，大约为 1 毫电子哈特里 (mHA) 或甚至 0.1 mHA。对于参与感兴趣化学反应的每种分子物种的总电子能量，也需要相同的精度 [Burg]。在关于计算与原生物化学相关的分子电子结构的算法基准的作者们得出结论：“对于 NISQ 处理器来说，利用 VQE 并实现近化学精度将极具挑战性” [Lolur]。他们指出，主要挑战来自于将费米子哈密顿量映射到量子比特哈密顿量产生的大量 Pauli 项、大量的变分参数以及大量的能量评估。此外，为了获得 1 mHA 的精确能量估计，VQE 的基必须接近真实基态，误差小于百万分之一 [Troyer]。

在 *第六章**，超越玻恩-奥本海默* 中，我们解释了非玻恩-奥本海默 (non-BO) 计算如何包括在基态振动以上预测化学状态所需的效果。使用创新的混合经典-量子算法实现这些 non-BO 计算是一个开放的研究课题。谷歌量子算法团队的负责人 Ryan Babbush 开发了化学的第一个量子模拟算法。在他的这些算法的介绍 [Babbush] 中，他提到在许多情况下非-BO 模拟都很重要，例如涉及氢键的低温度、电子与原子核之间的隧穿或耦合，或者直接从量子动力学计算动力学、反应散射系数或热速率常数。对这些算法的研究 [Su] 显示出对第二量子化算法的潜在优势。然而，这些算法需要能够运行具有巨大数量门 (![](img/Formula_07_001.png) 到 ![](img/Formula_07_002.png)) 的量子电路的具有数千个逻辑量子比特的容错量子计算机，这远远超出了当前或近期的 NISQ 时代处理器的功能。

# 参考文献

[Arute] Arute, F., Arya, K., Babbush, R. et al., 使用可编程超导处理器实现量子霸权，自然 574, 505–510 (2019), [`doi.org/10.1038/s41586-019-1666-5`](https://doi.org/10.1038/s41586-019-1666-5)

[Auffeves] Alexia Auffèves, 优化量子计算器的能耗：一个跨学科挑战，物理反思 N°69 (2021) 16-20，2021 年 7 月 12 日，[`doi.org/10.1051/refdp/202169016`](https://doi.org/10.1051/refdp/202169016)

[Babbush] Ryan Babbush，2021 年 2 月 24 日，第一量子化化学量子模拟的承诺，Google AI Quantum，化学的容错未来是第一量子化的！，[`www.youtube.com/watch?v=iugrIX616yg`](https://www.youtube.com/watch?v=iugrIX616yg)

[Burg] Vera von Burg，Guang Hao Low，Thomas Häner，Damian S. Steiger，Markus Reiher，Martin Roetteler，Matthias Troyer，量子计算增强的计算催化，2021 年 3 月 3 日，10.1103/PhysRevResearch.3.033055，[`arxiv.org/abs/2007.14460`](https://arxiv.org/abs/2007.14460)

[Corcoles] A. D. Córcoles, Maika Takita, Ken Inoue, Scott Lekuch, Zlatko K. Minev, Jerry M. Chow, and Jay M. Gambetta, 利用超导量子比特在量子算法中利用动态量子电路，物理评论快报 127, 100501，2021 年 8 月 31 日，[`journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.100501`](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.100501)

[Hartke] Hartke, T., Oreg, B., Jia, N. et al., 费米子对量子寄存器，自然 601, 537–541 (2022)，[`doi.org/10.1038/s41586-021-04205-8`](https://doi.org/10.1038/s41586-021-04205-8)

[IBM_mid] 中断电路测量教程，IBM 量子系统，[`quantum-computing.ibm.com/lab/docs/iql/manage/systems/midcircuit-measurement/`](https://quantum-computing.ibm.com/lab/docs/iql/manage/systems/midcircuit-measurement/)

[Lolur] Lolur, Phalgun, Magnus Rahm, Marcus Skogh, Laura García-Álvarez 和 Göran Wendin，通过在高性能计算机上模拟原生物分子的基态能量来基准测试变分量子本征求解器，arXiv:2010.13578v2 [quant-ph]，2021 年 1 月 5 日，[`arxiv.org/pdf/2010.13578.pdf`](https://arxiv.org/pdf/2010.13578.pdf)

[Q Daily] 量子技术 | 我们可持续的未来，量子日报，2021 年 7 月 29 日，[`www.youtube.com/watch?v=iB2_ibvEcsE`](https://www.youtube.com/watch?v=iB2_ibvEcsE)

[Quantum_AI] 量子人工智能可持续性研讨会，Q4Climate，演讲者：Karl Thibault 博士，Michał Stęchły 先生，2021 年 9 月 1 日，[`quantum.ieee.org/conferences/quantum-ai`](https://quantum.ieee.org/conferences/quantum-ai)

[Sawaya] Nicolas P. D. Sawaya，Francesco Paesani，Daniel P. Tabor，振动光谱的近中和长期量子算法方法，2021 年 2 月 1 日，arXiv:2009.05066 [quant-ph]，[`arxiv.org/abs/2009.05066`](https://arxiv.org/abs/2009.05066)

[Su] Yuan Su, Dominic W. Berry, Nathan Wiebe, Nicholas Rubin, and Ryan Babbush, 在第一量子化中实现容错量子化学模拟，2021 年 10 月 11 日，PRX 量子 2, 040332，DOI:10.1103/PRXQuantum.2.040332，[`doi.org/10.1103/PRXQuantum.2.040332`](https://doi.org/10.1103/PRXQuantum.2.040332)

[Thibault] Casey Berger, Agustin Di Paolo, Tracey Forrest, Stuart Hadfield, Nicolas Sawaya, Michał Stęchły, Karl Thibault, 量子技术应对气候变化：初步评估，IV. 量子计算机的能源效率，作者：Karl Thibault，arXiv:2107.05362 [quant-ph]，2021 年 6 月 23 日，[`arxiv.org/abs/2107.05362`](https://arxiv.org/abs/2107.05362)

[Troyer] Matthias Troyer, Matthias Troyer: 在化学模拟中实现实用量子优势，QuCQC 2021，[`www.youtube.com/watch?v=2MsfbPlKgyI`](https://www.youtube.com/watch?v=2MsfbPlKgyI)
