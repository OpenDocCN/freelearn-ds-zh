# 前言

由于硬件行业取得的进步，我们的存储容量增加了，因此许多组织都希望为了分析目的存储所有类型的事件。这催生了一个新的机器学习时代。机器学习领域非常复杂，编写这些算法并非易事。Apache Mahout 为我们提供了机器学习领域的现成算法，并使我们免于复杂的算法实现任务。

这本书的目的是涵盖 Apache Mahout 中可用的分类算法。无论你已经使用其他工具工作过分类算法，还是对这个领域完全陌生，这本书都会帮助你。因此，开始阅读这本书，探索最受欢迎的开源项目之一，Apache Mahout 中的分类算法，该项目享有强大的社区支持。

# 本书涵盖的内容

第一章, *数据分析中的分类*，介绍了数据分析中的分类概念。本章将涵盖分类的基础知识、相似度矩阵以及该领域可用的算法。

第二章, *Apache Mahout*，介绍了 Apache Mahout 及其安装过程。此外，本章将讨论为什么它是进行分类的好选择。

第三章, *使用 Mahout 学习逻辑回归/SGD*，讨论了逻辑回归和随机梯度下降，以及开发者如何使用 Mahout 实现 SGD。

第四章, *使用 Mahout 学习朴素贝叶斯分类*，讨论了贝叶斯定理、朴素贝叶斯分类以及我们如何使用 Mahout 构建朴素贝叶斯分类器。

第五章, *使用 Mahout 学习隐马尔可夫模型*，涵盖了 HMM 以及如何使用 Mahout 的 HMM 算法。

第六章, *使用 Mahout 学习随机森林*，详细讨论了随机森林算法，以及如何使用 Mahout 的随机森林实现。

第七章, *使用 Mahout 学习多层感知器*，讨论了 Mahout 作为神经网络早期实现的工具。本章我们将讨论多层感知器。此外，我们将使用 Mahout 的 MLP 实现。

第八章, *即将发布的 Mahout 变化*，讨论了 Mahout 作为一个正在进行中的工作。我们将讨论即将发布的 Mahout 版本中的新主要变化。

第九章, *使用 Apache Mahout 构建电子邮件分类系统*，提供了两个电子邮件分类的使用案例——垃圾邮件分类和基于邮件所属项目的电子邮件分类。我们将创建模型，并在一个模拟真实工作环境的程序中使用此模型。

# 您需要为此书准备的内容

要使用本书中的示例，您应该在您的系统上安装以下软件：

+   Java 1.6 或更高版本

+   Eclipse

+   Hadoop

+   Mahout；我们将在本书的第二章 Apache Mahout 中讨论其安装

+   Maven，根据您如何安装 Mahout

# 本书面向的对象

如果您是一位对 Hadoop 生态系统和机器学习方法有一定经验的 数据科学家，并想尝试使用 Mahout 在大型数据集上进行分类，这本书非常适合您。Java 知识是必需的。

# 规范

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称将如下所示：“提取源代码并确保文件夹包含`pom.xml`文件。”

代码块将以如下方式设置：

```py
    public static Map<String, Integer> readDictionary(Configuration conf, Path dictionaryPath) {
        Map<String, Integer> dictionary = new HashMap<String, Integer>();
        for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(dictionaryPath, true, conf)) {
            dictionary.put(pair.getFirst().toString(), pair.getSecond().get());
        }
        return dictionary;
    }
```

当我们希望将您的注意力引向代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
    public static Map<String, Integer> readDictionary(Configuration conf, Path dictionaryPath) {
 Map<String, Integer> dictionary = new HashMap<String, Integer>();
        for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(dictionaryPath, true, conf)) {
            dictionary.put(pair.getFirst().toString(), pair.getSecond().get());
        }
        return dictionary;
    }
```

任何命令行输入或输出都将如下所示：

```py
hadoop fs -mkdir /user/hue/KDDTrain 
hadoop fs -mkdir /user/hue/KDDTest
hadoop fs –put /tmp/KDDTrain+_20Percent.arff  /user/hue/KDDTrain
hadoop fs –put /tmp/KDDTest+.arff  /user/hue/KDDTest

```

**新术语**和**重要词汇**将以粗体显示。你在屏幕上看到的单词，例如在菜单或对话框中，将以如下方式显示：“现在，导航到**mahout-distribution-0.9**的位置并点击**完成**。”

### 注意

警告或重要注意事项将以如下方框显示。

### 小贴士

小贴士和技巧将以如下方式显示。

# 读者反馈

我们欢迎读者的反馈。告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出您真正能从中获得最大收益的标题。

要向我们发送一般反馈，请简单地发送电子邮件至 `<feedback@packtpub.com>`，并在邮件主题中提及书籍标题。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)上下载您购买的所有 Packt Publishing 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 下载本书的彩色图像

我们还为您提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。这些彩色图像将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/4959OS_ColoredImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/4959OS_ColoredImages.pdf)下载此文件。

## 勘误

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分下。

## 盗版

互联网上对版权材料的盗版是一个持续存在的问题，涉及所有媒体。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现任何形式的非法复制我们的作品，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过<mailto:copyright@packtpub.com> copyright@packtpub.com>与我们联系，并提供疑似盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

## 问题

如果您对本书的任何方面有问题，您可以通过<mailto:questions@packtpub.com> questions@packtpub.com>与我们联系，我们将尽力解决问题。
