# 附录 A：准备数学概念

按照惯例，#表示在*第九章**，术语表*中有一个互补条目。

在本附录中，我们将涵盖以下主题：

+   使用的符号

+   数学定义

# 技术要求

本章的配套 Jupyter 笔记本可以从 GitHub 下载，网址为[`github.com/PacktPublishing/Quantum-Chemistry-and-Computing-for-the-Curious`](https://github.com/PacktPublishing/Quantum-Chemistry-and-Computing-for-the-Curious)，已在 Google Colab 环境中测试，这是一个免费的环境，完全运行在云端，并在 IBM Quantum Lab 环境中。请参阅*附录 B**——在云端利用 Jupyter 笔记本*，获取更多信息。配套的 Jupyter 笔记本会自动安装以下列表中的库：

+   **数值 Python**（**NumPy**）[NumPy]，一个开源的 Python 库，几乎被应用于科学和工程的各个领域

+   **SymPy** [SymPy]，一个用于符号数学的 Python 库

+   **Qiskit** [Qiskit]，一个开源 SDK，用于在脉冲、电路和应用模块级别与量子计算机交互

+   Qiskit 可视化支持以启用其可视化功能和使用 Jupyter 笔记本

## 安装 NumPy、SimPy 和 Qiskit 以及导入各种模块

使用以下命令安装 NumPy：

```py
pip install numpy
```

使用以下命令安装 SymPy：

```py
pip install simpy
```

使用以下命令安装 Qiskit：

```py
pip install qiskit
```

### 导入 NumPy 和一个返回复数数组 LaTeX 表示的函数

使用以下命令导入 NumPy：

```py
import numpy as np
```

导入所需的函数和类方法。`array_to_latex function()` 返回一个具有 1 或 2 维度的复数数组的 LaTeX 表示：

```py
from qiskit.visualization import array_to_latex
```

# 使用的符号

我们将在适当的地方使用以下符号：

+   ![](img/Formula_A_001.png)，等等——小写希腊字母表示标量。

+   ![](img/Formula_A_002.png)，等等——小写拉丁字母表示粒子空间中的列向量。这些向量有 n 个分量，表示为 ![](img/Formula_A_003.png)，等等，其中 k 是一个整数。

+   ![](img/Formula_A_004.png)，等等——大写拉丁字母表示粒子空间中的矩阵。这些是 n x n 的矩阵。

+   ![](img/Formula_A_005.png)，等等——撇号（'）和字母 ![](img/Formula_A_006.png) 分别表示向量和矩阵的转置。

+   ![](img/Formula_A_007.png)，等等——星号（*）用于向量和矩阵的复共轭。

+   ![](img/Formula_A_008.png)，![](img/Formula_A_009.png)，等等—— dagger 符号，![](img/Formula_A_010.png)，用于向量和矩阵的复共轭转置。

+   ![](img/Formula_A_011.png)，等等——负一的幂 ![](img/Formula_A_012.png) 表示矩阵的逆。

+   ![](img/Formula_A_013.png)，等等 – 符号![](img/Formula_A_014.png)表示矩阵和/或向量的克罗内克积或张量积。

+   ![](img/Formula_A_015.png)，等等。 – 符号![](img/Formula_A_016.png)表示平方矩阵的克罗内克和。

+   ![](img/Formula_A_017.png) – 至少存在一个。

+   ![](img/Formula_A_018.png) – 对于所有。

+   ![](img/Formula_A_019.png) – 例如，是![](img/Formula_A_020.png)的成员意味着![](img/Formula_A_021.png)属于实数集合![](img/Formula_A_022.png)。

# 数学定义

## 泡利不相容原理（PEP） #

在 1925 年，泡利描述了电子的 PEP，该原理指出，同一原子的两个电子不可能同时具有以下四个量子数的相同值：![](img/Formula_A_023.png)，主量子数；![](img/Formula_A_024.png)，角动量量子数；![](img/Formula_A_025.png)，磁量子数；和![](img/Formula_A_026.png)，自旋量子数。

在发现各种基本粒子之后，电子的 PEP（泡利不相容原理）已被推广到所有基本粒子和复合系统。记住，费米子是具有半整数值自旋（![](img/Formula_A_027.png)）的粒子，而玻色子是具有整数值自旋（![](img/Formula_A_028.png)）的粒子。PEP 的一般公式指出，量子系统的总波函数![](img/Formula_A_029.png)必须对所有相同粒子的集合具有某些对称性，即电子和相同核子，玻色子和费米子，在配对粒子排列操作下：

+   对于费米子，总波函数必须相对于相同配对粒子的交换具有反对称性（![](img/Formula_A_030.png)）：

![](img/Formula_A_032.png)

这意味着波函数的空间部分是反对称的，而自旋部分是对称的，或者反之亦然。

+   对于玻色子，总波函数必须相对于配对粒子的交换具有对称性（![](img/Formula_A_033.png)）：

![](img/Formula_A_035.png)

这意味着空间波函数和自旋函数要么都是对称的，要么都是反对称的。

+   对于同时包含相同费米子和相同玻色子的复合系统，上述操作必须同时成立。

## 角动量量子数 #

也称为轨道量子数或方位量子数，用 ![](img/Formula_A_036.png) 表示，它描述了电子亚层，并通过关系式：![](img/Formula_A_037.png)给出了轨道角动量的模。在化学和光谱学中，![](img/Formula_A_038.png) 被称为 ![](img/Formula_A_039.png) 轨道，![](img/Formula_A_040.png) 是 ![](img/06_013.png) 轨道，![](img/Formula_A_042.png) 是 ![](img/Formula_A_043.png) 轨道，而 ![](img/Formula_A_044.png) 是 ![](img/Formula_A_045.png) 轨道。技术上，在 ![](img/Formula_A_046.png) 轨道之外还有更多的轨道，即 ![](img/Formula_A_047.png)，![](img/Formula_A_048.png) 等等，这些具有更高的能级。

## 职位数算子 #

一个算子 ![](img/Formula_A_049.png)，其中 ![](img/Formula_A_050.png) 和 ![](img/Formula_A_051.png) 是湮灭算子，而 ![](img/Formula_A_052.png) 是作用在局部电子模式上的创建算子，它们满足以下反对易关系：

![](img/Formula_A_053.jpg)

![](img/Formula_A_054.png)

其中 ![](img/Formula_A_055.png) 是狄拉克δ函数，而 ![](img/Formula_A_056.png) 是两个算子 ![](img/Formula_A_057.png) 和 ![](img/Formula_A_058.png) 的反对易子。

## 量子相位估计 (QPE) #

给定一个幺正算子 ![](img/Formula_05_052.png)，其本征态和本征值 ![](img/Formula_A_060.png)，准备状态 ![](img/Formula_A_061.png) 的能力，以及应用 ![](img/Formula_A_062.png) 本身的能力，QPE 算法计算 ![](img/Formula_A_063.png)，其中 ![](img/Formula_06_123.png) 是用于估计 ![](img/Formula_A_065.png) 所使用的量子比特数，从而允许我们测量 ![](img/Formula_A_066.png) 如我们所希望的那样精确。

## 复数

复数的形式为 ![](img/Formula_A_067.png)，其中 ![](img/Formula_A_068.png) 和 ![](img/Formula_A_069.png) 是实数，而 ![](img/Formula_A_070.png) (![](img/Formula_A_071.png) 在 Python 中称为虚数单位)定义为满足方程 ![](img/Formula_A_072.png) 的数。复数的模长为：![](img/Formula_A_073.png)。复数 ![](img/Formula_A_074.png) 的共轭复数为 ![](img/Formula_A_075.png)。欧拉公式 ![](img/Formula_A_076.png) 在复数的乘法和指数运算中非常方便。复数集加上加法和乘法运算构成一个域，表示为 ![](img/Formula_A_077.png)。由复数组成的代数表达式遵循代数的标准规则；与实数不同的是，![](img/Formula_A_078.png) 被替换为 ![](img/Formula_A_079.png)。

## 向量空间

在复数域 ![](img/Formula_A_081.png) 或实数域 ℝ 上，一个向量空间 ![](img/Formula_A_080.png) 是一组称为向量的对象集合，这些向量可以相加，并且可以由数字相乘（“缩放”）。

以下 Python 代码说明了具有两个复数分量的向量：

```py
x = np.array([[1j],
             [2]])
array_to_latex(x, prefix='x = ')
```

![](img/Formula_A_082.png)

我们使用 Python 3.5 中引入的@运算符来乘以一个向量，如下所示，将向量![](img/Formula_A_083.png)乘以虚数单位![](img/Formula_A_084.png)（Python 中的![](img/Formula_A_071.png)），其中![](img/Formula_A_086.png)被替换为![](img/Formula_A_0791.png)：

```py
α = 1j
print('α =', α)
y = α*x
array_to_latex(y, prefix=' y = α*x =')
```

![](img/Formula_A_088.png)

![](img/Formula_A_089.jpg)

## 线性算子

定义在向量空间![](img/Formula_A_091.png)上，且在![](img/Formula_A_092.png)上的函数![](img/Formula_A_090.png)是线性算子，如果它具有以下两个性质：

+   对于任何![](img/Formula_A_093.png)在![](img/Formula_A_094.png)中，![](img/Formula_A_095.png)

+   对于任何![](img/Formula_A_096.png)在![](img/Formula_A_097.png)中，![](img/Formula_A_098.png)在![](img/Formula_A_099.png)中，![](img/Formula_A_100.png)

## 矩阵

矩阵是一组以方形或矩形排列的元素。元素可以是数字、矩阵、函数或代数表达式。矩阵的阶数或形状写作（行数）x（列数）。索引以*行，列*格式书写，例如，![](img/Formula_A_101.png)是第![](img/Formula_A_102.png)行和第![](img/Formula_A_103.png)列的元素。矩阵代表向量空间中的线性算子。在某些正交归一基中，使用相同的符号表示算子和它的矩阵是方便的。

## 特征值和特征向量

根据定义，线性算子![](img/Formula_A_104.png)在向量空间![](img/Formula_A_105.png)上定义，且在![](img/Formula_A_099.png)上，其特征向量是一个非零向量![](img/Formula_05_045.png)，它具有以下性质：![](img/Formula_A_108.png)，其中![](img/Formula_A_109.png)是![](img/Formula_A_110.png)中的一个标量，称为与特征向量![](img/Formula_A_111.png)相关的特征值。

对于有限维空间![](img/Formula_A_112.png)，上述定义等价于![](img/Formula_A_113.png)，其中![](img/Formula_A_114.png)是![](img/Formula_A_115.png)的矩阵表示。

## 向量和矩阵的转置、共轭和共轭转置

一些向量![](img/Formula_A_116.png)或一些矩阵![](img/Formula_A_114.png)的转置，通常表示为![](img/Formula_A_118.png)，是通过交换向量![](img/Formula_A_119.png)或矩阵![](img/Formula_A_120.png)的行和列索引得到的。以下 Python 代码说明了向量![](img/Formula_A_121.png)的转置：

```py
x = array_to_latex(x.transpose(), prefix='x^T = ')
```

![](img/Formula_A_122.png)

一些向量![](img/Formula_A_123.png)或一些矩阵![](img/Formula_A_124.png)的复共轭，通常表示为![](img/Formula_A_125.png)、![](img/Formula_A_126.png)，是通过对所有元素执行复共轭得到的。

```py
x = array_to_latex(x.conjugate(), prefix='x^* = ')
```

![](img/Formula_A_127.png)

在量子力学中，某些向量 ![](img/Formula_A_128.png) 或矩阵 ![](img/Formula_02_430.png) 的复共轭转置通常表示为 ![](img/Formula_A_130.png)、![](img/Formula_A_131.png)。符号 ![](img/Formula_A_132.png) 被称为 dagger。![](img/Formula_02_429.png) 被称为伴随或厄米共轭 ![](img/Formula_02_304.png)：

```py
x = array_to_latex(x.conjugate().transpose(), prefix='(x^*)^T = ')
```

![公式 A_135](img/Formula_A_135.png)

## 狄拉克符号 #

在狄拉克符号，也称为括号-括号符号中：

+   拉括号 ![](img/Formula_A_136.png) 表示一个矢量，它代表量子系统的状态。

+   拉括号 ![](img/Formula_A_137.png) 表示一个线性函数，它将每个向量映射到一个复数。

+   线性函数 ![](img/Formula_A_138.png) 对矢量 ![](img/Formula_A_139.png) 的作用表示为 ![](img/Formula_A_140.png)。

它们之间的关系如下：

![公式 A_141](img/Formula_A_141.png) ![公式 A_142](img/Formula_A_142.png)

## 两个向量的内积

在向量空间 ![](img/Formula_A_105.png) 上，对 ![](img/Formula_A_144.png) 的内积是一个复函数 (·, ·)，它返回一个标量，并满足以下条件：

+   对于任何 ![](img/Formula_A_145.png) 在 ![](img/Formula_A_080.png) 中，![](img/Formula_A_147.png)。此外，![](img/Formula_A_148.png) 当且仅当 ![](img/Formula_A_149.png)。

+   ![公式 A_150.png].

+   ![公式 A_151](img/Formula_A_151.png).

在 ![](img/Formula_A_152.png) 上，标准厄米内积是：![](img/Formula_A_153.png) .

使用狄拉克符号，向量 ![](img/Formula_A_154.png) 和 ![](img/Formula_A_155.png) 的内积表示为 ![](img/Formula_A_156.png)，并且与将括号 ![](img/Formula_A_157.png) 应用到矢量 ![](img/Formula_A_155.png) 上的结果相同：

![公式 A_159](img/Formula_A_159.png) ![公式 A_160](img/Formula_A_160.png)

![公式 A_161.jpg]

Python 的 `numpy.vdot` 函数返回两个向量的厄米内积：

```py
array_to_latex(x, prefix='x = ')
```

![公式 A_162.png]

```py
array_to_latex(y, prefix='y = ')
```

![公式 A_163](img/Formula_A_163.png)

```py
print("np.vdot(x, y) = ", np.vdot(x, y)
```

![公式 A_164](img/Formula_A_164.png)

## 向量的范数

内积产生一个由 ![](img/Formula_A_165.png) 定义的范数。除了三角不等式 ![](img/Formula_A_166.png) 之外，范数还满足施瓦茨不等式 ![](img/Formula_A_167.png)。向量的范数或向量的模通常被称为向量的长度。

Python 的 `numpy.linalg.norm` 函数返回向量的范数：

```py
print("Norm of vector x: {:.3f}".format(np.linalg.norm(x)))
```

![公式 A_168](img/Formula_A_168.png)

## 希尔伯特空间

内积空间 ![](img/Formula_A_169.png) 如果在诱导范数下是完备的，即如果每个柯西序列都收敛：对于每个序列 ![](img/Formula_A_170.png) 有 ![](img/Formula_A_171.png) 使得 ![](img/Formula_A_172.png)，在 ![](img/Formula_A_105.png) 中存在一个 ![](img/Formula_02_529.png)，使得 ![](img/Formula_A_175.png)。这个性质允许使用微积分技术。

## 矩阵与向量的乘法

Python 3.5 中引入的 `@` 运算符实现了矩阵与向量的乘法：

```py
A = np.array([[1, 2],
              [3, 1j]])
array_to_latex(A, prefix='A = ')
```

![公式 A_176](img/Formula_A_176.png)

```py
a = np.array([[1],
             [1]])
array_to_latex(a, prefix='a = ')
```

![](img/Formula_A_177.png)

```py
array_to_latex(A@ , prefix='A@  = ')
```

![](img/Formula_A_180.png)

## 矩阵加法

相同形状的两个矩阵的加法是通过将相应的项相加来实现的：

![](img/Formula_A_181.png)

```py
A = np.array([[1, 0],
              [0, 1j]])
array_to_latex(A, prefix='A = ')
```

![](img/Formula_A_182.png)

```py
B = np.array([[0, 1],
              [1j, 0]])
array_to_latex(B, prefix='B = ')
```

![](img/Formula_A_183.png)

```py
array_to_latex(A+B, prefix='A+B = ')
```

![](img/Formula_A_184.png)

## 矩阵乘法

设 ![](img/Formula_02_304.png) 是一个 m 行 n 列的矩阵，![](img/Formula_A_186.png) 是一个 n 行 p 列的矩阵，那么乘积 ![](img/Formula_A_187.png) 是一个 m 行 p 列的矩阵，其定义如下：

![](img/Formula_A_188.png)

Python 3.5 中引入的 `@` 运算符实现了矩阵乘法：

```py
A = np.array([[1, 0],
              [0, 1j]])
array_to_latex(A, prefix='A = ')
```

![](img/Formula_A_189.png)

```py
B = np.array([[1, 1, 1j],
              [1, -1, 0]])
array_to_latex(B, prefix='B = ')
```

![](img/Formula_A_190.png)

```py
array_to_latex(A@B, prefix='A@B = ')
```

![](img/Formula_A_191.png)

## 矩阵逆

当某些矩阵 ![](img/Formula_A_192.png) 存在时，其逆矩阵表示为 ![](img/Formula_A_193.png)，是一个矩阵，使得 ![](img/Formula_A_194.png)，其中 ![](img/Formula_A_195.png) 是单位矩阵，对于任何矩阵 ![](img/Formula_02_304.png) : ![](img/Formula_A_197.png) 。`numpy.linalg.inv` 函数计算矩阵的乘法逆：

```py
from numpy.linalg import inv
a = np.array([[1., 2.], [3., 4.]])
array_to_latex(A, prefix='A =')
```

![](img/Formula_A_198.png)

```py
array_to_latex(inv(A), prefix='A^{-1} = ')
```

![](img/Formula_A_199.jpg)

## 张量积

给定维度为 ![](img/Formula_02_458.png) 的向量空间 ![](img/Formula_02_350.png) 和维度为 ![](img/Formula_A_202.png) 的向量空间 ![](img/Formula_A_203.png) 在 ![](img/Formula_A_204.png) 上，张量积 ![](img/Formula_A_205.png) 是另一个维度为 ![](img/Formula_A_207.png) 的向量空间 ![](img/Formula_A_206.png) 在 ![](img/Formula_A_099.png) 上。

![](img/Formula_A_209.png) 和 ![](img/Formula_A_210.png) 是 ![](img/Formula_A_211.png) 上的线性映射，![](img/Formula_A_212.png) 是 ![](img/Formula_A_213.png) 上的线性映射：

### 双线性

![](img/Formula_A_214.png)

![](img/Formula_A_215.png)

![](img/Formula_A_216.png)

### 结合律

![](img/Formula_A_217.png)

![](img/Formula_A_218.png)

### 线性映射的性质

![](img/Formula_A_219.png)

![](img/Formula_A_220.png)

如果内积空间 ![](img/Formula_A_221.png) 是两个内积空间 ![](img/Formula_A_222.png)，![](img/Formula_A_169.png) 的张量积，那么对于每一对向量 ![](img/Formula_A_224.png)，![](img/Formula_A_225.png)，在 ![](img/Formula_A_227.png) 中存在一个相关的张量积 ![](img/Formula_A_226.png)。

在狄拉克记号中，我们表示张量积 ![](img/Formula_A_228.png) 为 ![](img/Formula_A_229.png) 或 ![](img/Formula_A_230.png)。

![](img/Formula_A_231.png) 和 ![](img/Formula_A_232.png) 的内积是 ![](img/Formula_A_233.png)。

## 克朗内克积或矩阵或向量的张量积

表示为 ![](img/Formula_A_234.png) 的两个矩阵的克朗内克积或张量积是由第二个矩阵的块按第一个矩阵的比例缩放而成的复合矩阵。设 ![](img/Formula_A_124.png) 是一个 m 行 n 列的矩阵，![](img/Formula_A_236.png) 是一个 p 行 q 列的矩阵，那么克朗内克积 ![](img/Formula_A_237.png) 是一个 pm 行 qn 列的块矩阵：

![](img/Formula_A_238.png)

Python 的 `numpy.kron` 函数实现了克罗内克积：

```py
A = np.array([[1,2],
              [3, 4]])
array_to_latex(A, prefix='A =')
```

![公式 A_239](img/Formula_A_239.png)

```py
B = np.array([[0, 5],
              [6, 7]])
array_to_latex(B, prefix='B =')
```

![公式 A_240](img/Formula_A_240.png)

```py
C = np.kron(A,B)
array_to_latex(C, prefix='A \otimes B =')
```

![公式 A_241](img/Formula_A_241.jpg)

## 克罗内克和

任何两个方阵的克罗内克和，![公式 A_192](img/Formula_A_192.png) n×n 和 ![公式 A_243](img/Formula_A_243.png) m×m，记为 ![公式 A_244](img/Formula_A_244.png)，定义为：

![公式 A_245](img/Formula_A_245.png)

其中 ![公式 A_246](img/Formula_A_246.png) 是阶数为 ![公式 A_247](img/Formula_A_247.png) 的单位矩阵，![公式 A_248](img/Formula_A_248.png) 是阶数为 ![公式 A_249](img/Formula_A_249.png) 的单位矩阵。

## 外积

基底 ![公式 A_139](img/Formula_A_139.png) 和 bra ![公式 A_251](img/Formula_A_251.png) 的外积是一个秩为 1 的算子 ![公式 A_252](img/Formula_A_252.png)，其规则为：

![公式 A_253](img/Formula_A_253.png)

对于有限维向量空间，外积是一个简单的矩阵乘法：

![公式 A_254](img/Formula_A_254.jpg)

Python 的 `numpy.outer` 函数实现了外积：

```py
array_to_latex(x, prefix='x = ')
```

![公式 A_255](img/Formula_A_255.png)

```py
array_to_latex(y, prefix='y = ')
```

![公式 A_256](img/Formula_A_256.png)

```py
array_to_latex(np.outer(x, y), prefix='np.outer(x, y) = ')
```

![公式 A_257](img/Formula_A_257.png)

### 将矩阵表示为外积之和

任何矩阵都可以用外积表示。例如，对于一个 2×2 矩阵：

![公式 A_258](img/Formula_A_258.png) ![公式 A_259](img/Formula_A_259.png)

![公式 A_260](img/Formula_A_260.png) ![公式 A_261](img/Formula_A_261.png)

![公式 A_262](img/Formula_A_262.png)

## 厄米算子

某些向量 ![公式 A_263](img/Formula_A_263.png) 或矩阵 ![公式 A_057](img/Formula_A_057.png) 的复共轭转置通常表示为 ![公式 _02_426](img/Formula_02_426.png)，![公式 A_266](img/Formula_A_266.png) 在量子力学中。符号 ![公式 A_267](img/Formula_A_267.png) 被称为 dagger。![公式 A_268](img/Formula_A_268.png) 被称为伴随或厄米共轭。

一个线性算子 ![公式 A_270](img/Formula_A_270.png) 如果它是自己的伴随算子，则称为厄米算子或自伴算子：![公式 A_271](img/Formula_A_271.png)。

谱定理表明，如果 ![公式 A_270](img/Formula_A_270.png) 是厄米算子，那么它必须有一组正交归一的特征向量

![公式 A_273](img/Formula_A_273.png)

其中 ![公式 A_274](img/Formula_A_274.png) 具有实特征值 ![公式 A_275](img/Formula_A_275.png)，![公式 _02_480](img/Formula_02_480.png) 是特征向量的数量，或者也是希尔伯特空间的维度。厄米算子以特征值集合 ![公式 A_277](img/Formula_A_277.png) 和相应的特征向量 ![公式 A_278](img/Formula_A_278.png) 为唯一谱表示：

![公式 A_279](img/Formula_A_279.png)

## 单位算子

一个线性算子 ![公式 A_02_433](img/Formula_A_02_433.png) 如果其伴随算子存在并且满足 ![公式 A_281](img/Formula_A_281.png)，其中 ![公式 A_282](img/Formula_A_282.png) 是单位矩阵，根据定义，它将乘以的任何向量保持不变，则称为单位算子。

单位算子保持内积：

![公式 A_283](img/Formula_A_283.png)

因此，单位算子也保持了通常称为量子态长度的范数：

![公式 A_284](img/Formula_A_284.png)

对于任何幺正矩阵 ![](img/Formula_02_458.png)，任何特征向量 ![](img/Formula_A_286.png) 和 ![](img/Formula_A_287.png) 以及它们的特征值 ![](img/Formula_A_288.png) 和 ![](img/Formula_A_289.png)，![](img/Formula_A_290.png) 和 ![](img/Formula_A_291.png)，特征值 ![](img/Formula_A_292.png) 和 ![](img/Formula_02_466.png) 的形式为 ![](img/Formula_A_294.png)，如果 ![](img/Formula_A_295.png) 则特征向量 ![](img/Formula_A_296.png) 和 ![](img/Formula_A_297.png) 是正交的：![](img/Formula_A_298.png)。

有用的一点是，由于对于任何 ![](img/Formula_02_472.png) ，![](img/Formula_A_300.png)：

![](img/Formula_A_301.png)

## 密度矩阵 #

任何量子态，无论是**混合**还是**纯**，都可以用一个**密度矩阵**(![](img/Formula_A_302.png))来描述，这是一个归一化的正厄米算子，其中 ![](img/Formula_A_303.png)。根据谱定理，存在一个正交基，在*第 2.3.1 节，厄米算子*中定义，使得密度是所有特征值的和(![](img/Formula_02_480.png))：

![](img/Formula_A_305.jpg)

其中 ![](img/Formula_A_306.png) 从 1 到 ![](img/Formula_A_307.png)，![](img/Formula_A_308.png) 是正的或零特征值 (![](img/Formula_A_309.png))，特征值的和是密度矩阵的迹操作 (![](img/Formula_A_310.png))，等于 1：

![](img/Formula_A_311.jpg)

例如，当密度为 ![](img/Formula_A_312.png)，![](img/Formula_A_313.png) 时，密度的迹为：

![](img/Formula_A_314.png)

下面是一些纯量子态密度矩阵的例子：

![](img/Formula_A_315.jpg)![](img/Formula_A_316.jpg)![](img/Formula_A_317.jpg)

由 ![](img/Formula_02_247.png) 个纯量子态 ![](img/Formula_A_319.png) 组成的混合量子态的密度矩阵，每个态具有经典发生的概率 ![](img/Formula_A_320.png)，定义为：

![](img/Formula_A_321.png)

其中每个 ![](img/Formula_A_322.png) 是正的或零，它们的和等于一：

![](img/Formula_A_323.png)

我们在图 AA.1 中总结了纯态和混合态之间的区别，它与*图 2.20*相同。

![](img/B18268_Appendix_table1.1.jpg)

图 AA.1 – 纯态和混合量子态的密度矩阵

## 泡利矩阵

存在三个泡利矩阵(![](img/Formula_A_330.png) , ![](img/Formula_A_331.png) 和 ![](img/Formula_A_332.png))：

![](img/Formula_A_333.png) , ![](img/Formula_A_334.png) , ![](img/Formula_A_335.png)

它们是厄米算子和幺正算子，使得每个矩阵的平方等于 ![](img/Formula_A_336.png) 单位矩阵：

![](img/Formula_A_337.png)

每个泡利矩阵都等于其逆矩阵：

![](img/Formula_A_338.png)

![](img/Formula_A_339.png)

![](img/Formula_A_340.png)

我们在以下表格中总结了泡利矩阵和作用于量子比特的操作，这些操作产生相关的特征向量：

![](img/B18268_Appendix_table1.2.jpg)

### 将矩阵分解为泡利矩阵张量的加权求和

可以证明任何矩阵都可以分解为恒等矩阵和泡利矩阵 ![](img/Formula_A_353.png) 的张量的加权求和，其中 ![](img/Formula_A_354.png) ，权重为 ![](img/Formula_A_355.png) 和 ![](img/Formula_02_331.png) 个量子比特：

![](img/Formula_A_357.png)

对于厄米矩阵，所有权重 ![](img/Formula_A_355.png) 都是实数。

我们为任何 2x2 矩阵 ![](img/Formula_A_359.png) 提供一个证明。

![](img/Formula_A_360.jpg)![](img/Formula_A_361.jpg)![](img/Formula_A_362.jpg)![](img/Formula_A_363.jpg)

由于 ![](img/Formula_A_364.png) 因此 ![](img/Formula_A_365.png) 我们有：

![](img/Formula_A_366.jpg)![](img/Formula_A_367.jpg)

从 2x2 矩阵的分解作为外积之和开始：

![](img/Formula_A_368.png)

因此我们可以写出：

![](img/Formula_A_369.jpg)![](img/Formula_A_370.jpg)

## 反对易子 #

两个算子 ![](img/Formula_A_371.png) 的运算，定义为：![](img/Formula_A_372.png)。

## 反对易 #

可以定义一组作用于局部电子模式的费米子湮灭算符 ![](img/Formula_A_373.png) 和创建算符 ![](img/Formula_A_374.png) ，它们满足以下反对易关系：

![](img/Formula_A_375.png)

![](img/Formula_A_376.jpg)

## 交换子

两个算子 ![](img/Formula_A_377.png) 的运算，定义为：![](img/Formula_A_378.png)。对于任何算子 ![](img/Formula_02_304.png) 和 ![](img/Formula_06_150.png)，![](img/Formula_A_381.png) 当且仅当 ![](img/Formula_02_425.png) 和 ![](img/Formula_A_383.png) 交换。可以证明，如果一个量子系统有两个同时物理可观测的量，那么表示它们的厄米算子必须交换。对于任何算子 ![](img/Formula_02_425.png) ， ![](img/Formula_A_385.png) 和 ![](img/Formula_A_386.png) ，我们有以下关系，这些关系对于计算交换子是有用的：

![](img/Formula_A_387.png)

![](img/Formula_A_388.png)

![](img/Formula_A_389.png)

![](img/Formula_A_390.png)

![](img/Formula_A_391.png)

![](img/Formula_A_392.png)

![](img/Formula_A_393.png)

### 费米子，费米子，电子湮灭算符 #

一种数学运算，允许我们表示准粒子的激发或跃迁。激发需要初始状态比最终状态具有更低的能量水平。

一个算子 ![](img/Formula_A_394.png) ，它将位于 ![](img/Formula_A_395.png) 费米子轨道中的粒子数减少一个单位：

![](img/Formula_A_396.png)

其中：

![](img/Formula_A_397.png) 和 ![](img/Formula_A_398.png) 是位于 ![](img/Formula_A_395.png) 费米子轨道中的粒子数。

![](img/Formula_A_400.png) 是一个预因子，如果 ![](img/Formula_A_401.png) 费米子轨道中没有电子，即如果 ![](img/Formula_A_402.png) ，则湮灭斯莱特行列式中的状态。

相位因子 ![](img/Formula_A_403.png) 保持整个状态叠加的反对称性质。

### 费米子，费米子，电子创建算符 #

一种数学运算，允许我们表示准粒子的去激发（去激发）或跃迁。去激发需要初始状态比最终状态具有更高的能量水平。

一个算符 ![](img/Formula_A_404.png)，它将位于 ![](img/Formula_A_405.png) 费米子轨道中的粒子数增加一个单位：

![](img/Formula_A_406.png)

其中：

![](img/Formula_A_407.png) 和 ![](img/Formula_A_408.png) 是位于 ![](img/Formula_A_409.png) 费米子轨道中的粒子数。

![](img/Formula_A_410.png) 是一个预因子，如果我们有一个电子在 ![](img/Formula_A_411.png) 费米子轨道中，它将湮灭该状态，即如果 ![](img/Formula_A_412.png)。

相位因子 ![](img/Formula_A_413.png) 保持整个状态叠加的反对称性质。

### 费米子，费米子，电子激发算符 #

一个算符 ![](img/Formula_A_414.png)，它将电子从占据的自旋轨道 ![](img/Formula_A_415.png) 激发到未占据轨道 ![](img/Formula_A_416.png)。

## 总波函数 #

描述系统的物理行为，并由大写希腊字母 Psi 表示：![](img/Formula_A_417.png)。它包含量子系统的所有信息，包括作为参数的复数 (![](img/Formula_A_418.png))。一般来说，![](img/Formula_A_419.png) 是系统中所有粒子 ![](img/Formula_A_420.png) 的函数，其中粒子的总数是 ![](img/Formula_06_024.png)。此外，![](img/Formula_A_422.png) 包括每个粒子的空间位置 (![](img/Formula_A_423.png))、每个粒子的自旋方向坐标 (![](img/Formula_A_424.png)) 和时间 ![](img/Formula_A_425.png)：

![](img/Formula_A_426.png)

其中 ![](img/Formula_A_427.png) 和 ![](img/Formula_A_428.png) 是单粒子坐标的向量：

![](img/Formula_A_429.png)

![](img/Formula_A_430.png)

单粒子系统的总波函数是空间 ![](img/Formula_A_431.png)、自旋 ![](img/Formula_A_432.png) 和时间 ![](img/Formula_02_017.png) 函数的乘积：

![](img/Formula_A_434.png)

# 参考文献

[Micr_Algebra] 线性代数，QuantumKatas/tutorials/LinearAlgebra/：[`github.com/microsoft/QuantumKatas/tree/main/tutorials/LinearAlgebra`](https://github.com/microsoft/QuantumKatas/tree/main/tutorials/LinearAlgebra)

[Micr_Complex] 复数运算，QuantumKatas/tutorials/ComplexArithmetic/：https://github.com/microsoft/QuantumKatas/tree/main/tutorials/ComplexArithmetic

[NumPy] NumPy：初学者的绝对基础：[`numpy.org/doc/stable/user/absolute_beginners.html`](https://numpy.org/doc/stable/user/absolute_beginners.html)

[Qiskit] Qiskit：[`qiskit.org/`](https://qiskit.org/)

[Qiskit_Alg] 线性代数，Qiskit: [`qiskit.org/textbook/ch-appendix/linear_algebra.html`](https://qiskit.org/textbook/ch-appendix/linear_algebra.html)

[SymPy] SymPy，一个用于符号数学的 Python 库：[`www.sympy.org/en/index.html`](https://www.sympy.org/en/index.html)
