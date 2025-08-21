# 第十章：使用 Spark 进行高级文本处理

在第四章中，*使用 Spark 获取、处理和准备数据*，我们涵盖了与特征提取和数据处理相关的各种主题，包括从文本数据中提取特征的基础知识。在本章中，我们将介绍 Spark ML 中可用的更高级的文本处理技术，以处理大规模文本数据集。

在本章中，我们将：

+   通过详细示例，说明数据处理、特征提取和建模流程，以及它们与文本数据的关系

+   基于文档中的单词评估两个文档之间的相似性

+   使用提取的文本特征作为分类模型的输入

+   介绍自然语言处理的最新发展，将单词本身建模为向量，并演示使用 Spark 的 Word2Vec 模型评估两个单词之间的相似性，基于它们的含义

我们将研究如何使用 Spark 的 MLlib 以及 Spark ML 进行文本处理示例，以及文档的聚类。

# 文本数据有何特殊之处？

处理文本数据可能会很复杂，主要有两个原因。首先，文本和语言具有固有的结构，不容易使用原始单词来捕捉（例如，含义、上下文、不同类型的单词、句子结构和不同语言等）。因此，天真的特征提取通常相对无效。

其次，文本数据的有效维度非常大，潜在无限。想想仅英语单词的数量，再加上各种特殊单词、字符、俚语等等。然后，再加入其他语言以及互联网上可能找到的各种文本类型。文本数据的维度很容易超过数千万甚至数亿个单词，即使是相对较小的数据集。例如，数十亿个网站的 Common Crawl 数据集包含超过 8400 亿个单词。

为了解决这些问题，我们需要提取更结构化的特征的方法，以及处理文本数据的巨大维度的方法。

# 从数据中提取正确的特征

**自然语言处理**（**NLP**）领域涵盖了处理文本的各种技术，从文本处理和特征提取到建模和机器学习。在本章中，我们将重点关注 Spark MLlib 和 Spark ML 中可用的两种特征提取技术：**词频-逆文档频率**（**tf-idf**）术语加权方案和特征哈希。

通过 tf-idf 的示例，我们还将探讨在特征提取过程中的处理、标记化和过滤如何帮助减少输入数据的维度，以及改善我们提取的特征的信息内容和有用性。

# 术语加权方案

在第四章中，*使用 Spark 获取、处理和准备数据*，我们研究了向量表示，其中文本特征被映射到一个简单的二进制向量，称为**词袋**模型。实践中常用的另一种表示称为词频-逆文档频率。

tf-idf 根据文本（称为**文档**）中术语的频率对每个术语进行加权。然后，基于该术语在所有文档中的频率（数据集中的文档集通常称为**语料库**），应用全局归一化，称为**逆文档频率**。tf-idf 的标准定义如下：

*tf-idf(t,d) = tf(t,d) x idf(t)*

这里，*tf(t,d)*是文档*d*中术语*t*的频率（出现次数），*idf(t)*是语料库中术语*t*的逆文档频率；定义如下：

*idf(t) = log(N / d)*

在这里，*N*是文档的总数，*d*是术语*t*出现的文档数。

tf-idf 公式意味着在文档中多次出现的术语在向量表示中会获得更高的权重，相对于在文档中出现少次的术语。然而，IDF 归一化会减少在所有文档中非常常见的术语的权重。最终结果是真正罕见或重要的术语应该被分配更高的权重，而更常见的术语（假定具有较低重要性）在权重方面应该影响较小。

关于词袋模型（或向量空间模型）的更多学习资源是《信息检索导论》，作者是克里斯托弗·D·曼宁、普拉巴卡尔·拉加万和亨里希·舒兹，剑桥大学出版社（在[`nlp.stanford.edu/IR-book/html/htmledition/irbook.html`](http://nlp.stanford.edu/IR-book/html/htmledition/irbook.html)以 HTML 形式提供）。

它包含有关文本处理技术的部分，包括标记化、停用词去除、词干提取和向量空间模型，以及诸如 tf-idf 之类的加权方案。

也可以在[`en.wikipedia.org/wiki/Tf%E2%80%93idf`](http://en.wikipedia.org/wiki/Tf%E2%80%93idf)找到概述。

# 特征哈希

**特征哈希**是一种处理高维数据的技术，通常与文本和分类数据集一起使用，其中特征可以具有许多唯一值（通常有数百万个值）。在先前的章节中，我们经常对分类特征（包括文本）使用*1-of-K*编码方法。虽然这种方法简单而有效，但在面对极高维数据时可能会失效。

构建和使用*1-of-K*特征编码需要我们保留每个可能特征值到向量中的索引的映射。此外，创建映射本身的过程至少需要对数据集进行一次额外的遍历，并且在并行场景中可能会很棘手。到目前为止，我们经常使用简单的方法来收集不同的特征值，并将此集合与一组索引进行压缩，以创建特征值到索引的映射。然后将此映射广播（无论是在我们的代码中明确地还是由 Spark 隐式地）到每个工作节点。

然而，在处理文本时常见的数千万维甚至更高维的特征时，这种方法可能会很慢，并且可能需要大量的内存和网络资源，无论是在 Spark 主节点（收集唯一值）还是工作节点（广播生成的映射到每个工作节点，以便它可以将特征编码应用于其本地的输入数据）。

特征哈希通过使用哈希函数将特征的值哈希为一个数字（通常是整数值），并基于此值为特征分配向量索引。例如，假设“美国”地理位置的分类特征的哈希值为`342`。我们将使用哈希值作为向量索引，该索引处的值将为`1.0`，以表示“美国”特征的存在。所使用的哈希函数必须是一致的（即对于给定的输入，每次返回相同的输出）。

这种编码方式与基于映射的编码方式相同，只是我们需要提前为我们的特征向量选择一个大小。由于大多数常见的哈希函数返回整数范围内的值，我们将使用*模*运算将索引值限制为我们向量的大小，这通常要小得多（根据我们的要求，通常是几万到几百万）。

特征哈希的优点在于我们不需要构建映射并将其保存在内存中。它也很容易实现，非常快速，可以在线和实时完成，因此不需要先通过我们的数据集。最后，因为我们选择了一个明显小于数据集原始维度的特征向量维度，我们限制了模型在训练和生产中的内存使用；因此，内存使用量不随数据的大小和维度而扩展。

然而，存在两个重要的缺点，如下所示：

+   由于我们不创建特征到索引值的映射，因此也无法进行特征索引到值的反向映射。例如，这使得在我们的模型中确定哪些特征最具信息量变得更加困难。

+   由于我们限制了特征向量的大小，我们可能会遇到**哈希冲突**。当两个不同的特征被哈希到特征向量中的相同索引时，就会发生这种情况。令人惊讶的是，只要我们选择一个相对于输入数据维度合理的特征向量维度，这似乎并不会严重影响模型性能。如果哈希向量很大，冲突的影响就很小，但收益仍然很大。有关更多细节，请参阅此论文：[`www.cs.jhu.edu/~mdredze/publications/mobile_nlp_feature_mixing.pdf`](http://www.cs.jhu.edu/~mdredze/publications/mobile_nlp_feature_mixing.pdf)

有关哈希的更多信息可以在[`en.wikipedia.org/wiki/Hash_function`](http://en.wikipedia.org/wiki/Hash_function)找到。

引入哈希用于特征提取和机器学习的关键论文是：

*Kilian Weinberger*，*Anirban Dasgupta*，*John Langford*，*Alex Smola*和*Josh Attenberg*。*大规模多任务学习的特征哈希*。*ICML 2009 年会议论文*，网址为[`alex.smola.org/papers/2009/Weinbergeretal09.pdf`](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf)。

# 从 20 个新闻组数据集中提取 tf-idf 特征

为了说明本章中的概念，我们将使用一个名为**20 Newsgroups**的著名文本数据集；这个数据集通常用于文本分类任务。这是一个包含 20 个不同主题的新闻组消息的集合。有各种形式的可用数据。为了我们的目的，我们将使用数据集的`bydate`版本，该版本可在[`qwone.com/~jason/20Newsgroups`](http://qwone.com/~jason/20Newsgroups)上找到。

该数据集将可用数据分为训练集和测试集，分别占原始数据的 60%和 40%。在这里，测试集中的消息出现在训练集中的消息之后。该数据集还排除了一些标识实际新闻组的消息头；因此，这是一个适合测试分类模型实际性能的数据集。

有关原始数据集的更多信息可以在*UCI 机器学习库*页面上找到，网址为[`kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.data.html`](http://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.data.html)。

要开始，请下载数据并使用以下命令解压文件：

```scala
>tar xfvz 20news-bydate.tar.gz

```

这将创建两个文件夹：一个名为`20news-bydate-train`，另一个名为`20news-bydate-test`。让我们来看看训练数据集文件夹下的目录结构：

```scala
>cd 20news-bydate-train/ >ls

```

您将看到它包含许多子文件夹，每个子文件夹对应一个新闻组：

```scala
alt.atheism                comp.windows.x          rec.sport.hockey
  soc.religion.christian
comp.graphics              misc.forsale            sci.crypt
  talk.politics.guns comp.os.ms-windows.misc    rec.autos               sci.electronics
  talk.politics.mideast
comp.sys.ibm.pc.hardware   rec.motorcycles         sci.med
  talk.politics.misc
comp.sys.mac.hardware      rec.sport.baseball      sci.space
  talk.religion.misc

```

在每个新闻组文件夹下有许多文件；每个文件包含一个单独的消息帖子：

```scala
> ls rec.sport.hockey
52550 52580 52610 52640 53468 53550 53580 53610 53640 53670 53700 
53731 53761 53791
...

```

我们可以查看其中一条消息的一部分以查看格式：

```scala
> head -20 rec.sport.hockey/52550
From: dchhabra@stpl.ists.ca (Deepak Chhabra)
Subject: Superstars and attendance (was Teemu Selanne, was +/-
  leaders)
Nntp-Posting-Host: stpl.ists.ca
Organization: Solar Terresterial Physics Laboratory, ISTS
Distribution: na
Lines: 115

Dean J. Falcione (posting from jrmst+8@pitt.edu) writes:
[I wrote:]

>>When the Pens got Mario, granted there was big publicity,etc, etc,
>>and interest was immediately generated. Gretzky did the same thing for
>>LA.
>>However, imnsho, neither team would have seen a marked improvement in
>>attendance if the team record did not improve. In the year before Lemieux
>>came, Pittsburgh finished with 38 points. Following his arrival, the Pens
>>finished with 53, 76, 72, 81, 87, 72, 88, and 87 points, with a couple of
 ^^
>>Stanley Cups thrown in.
...

```

正如我们所看到的，每条消息都包含一些包含发件人、主题和其他元数据的标题字段，然后是消息的原始内容。

# 探索 20 个新闻组数据

我们将使用一个 Spark 程序来加载和分析数据集。

```scala
object TFIDFExtraction { 

  def main(args: Array[String]) { 

 } 
}

```

查看目录结构时，您可能会认出，我们再次有数据包含在单独的文本文件中（每个消息一个文本文件）。因此，我们将再次使用 Spark 的`wholeTextFiles`方法将每个文件的内容读入 RDD 中的记录。

在接下来的代码中，`PATH`指的是您提取`20news-bydate` ZIP 文件的目录：

```scala
val sc = new SparkContext("local[2]", "First Spark App") 

val path = "../data/20news-bydate-train/*" 
val rdd = sc.wholeTextFiles(path) 
// count the number of records in the dataset 
println(rdd.count)

```

如果您设置断点，您将看到以下行显示，指示 Spark 检测到的文件总数：

```scala
...
INFO FileInputFormat: Total input paths to process : 11314
...

```

命令运行完毕后，您将看到总记录数，应该与前面的`要处理的总输入路径`屏幕输出相同：

```scala
11314

```

现在让我们打印`rdd`的第一个元素，其中已加载数据：

```scala
16/12/30 20:42:02 INFO DAGScheduler: Job 1 finished: first at 
TFIDFExtraction.scala:27, took 0.067414 s
(file:/home/ubuntu/work/ml-resources/spark- 
ml/Chapter_10/data/20news- bydate-train/alt.atheism/53186,From:  
ednclark@kraken.itc.gu.edu.au (Jeffrey Clark)
Subject: Re: some thoughts.
Keywords: Dan Bissell
Nntp-Posting-Host: kraken.itc.gu.edu.au
Organization: ITC, Griffith University, Brisbane, Australia
Lines: 70
....

```

接下来，我们将查看可用的新闻组主题：

```scala
val newsgroups = rdd.map { case (file, text) => 
  file.split("/").takeRight(2).head } 
println(newsgroups.first()) 
val countByGroup = newsgroups.map(n => (n, 1)).reduceByKey(_ +
  _).collect.sortBy(-_._2).mkString("n") 
println(countByGroup)

```

这将显示以下结果：

```scala
(rec.sport.hockey,600)
(soc.religion.christian,599)
(rec.motorcycles,598)
(rec.sport.baseball,597)
(sci.crypt,595)
(rec.autos,594)
(sci.med,594)
(comp.windows.x,593)
(sci.space,593)
(sci.electronics,591)
(comp.os.ms-windows.misc,591)
(comp.sys.ibm.pc.hardware,590)
(misc.forsale,585)
(comp.graphics,584)
(comp.sys.mac.hardware,578)
(talk.politics.mideast,564)
(talk.politics.guns,546)
(alt.atheism,480)
(talk.politics.misc,465)
(talk.religion.misc,377)

```

我们可以看到消息数量在主题之间大致相等。

# 应用基本标记化

我们文本处理管道的第一步是将每个文档中的原始文本内容拆分为一组术语（也称为**标记**）。这就是**标记化**。我们将首先应用简单的**空格**标记化，同时将每个文档的每个标记转换为小写：

```scala
val text = rdd.map { case (file, text) => text } 
val whiteSpaceSplit = text.flatMap(t => t.split(" 
  ").map(_.toLowerCase)) 
println(whiteSpaceSplit.distinct.count)

```

在前面的代码中，我们使用了`flatMap`函数而不是`map`，因为现在我们想要一起检查所有标记以进行探索性分析。在本章的后面，我们将在每个文档的基础上应用我们的标记方案，因此我们将使用`map`函数。

运行此代码片段后，您将看到应用我们的标记化后的唯一标记总数：

```scala
402978

```

如您所见，即使对于相对较少的文本，原始标记的数量（因此，我们的特征向量的维度）也可能非常高。

让我们看一下随机选择的文档。我们将使用 RDD 的 sample 方法：

```scala
def sample( 
      withReplacement: Boolean, 
      fraction: Double, 
      seed: Long = Utils.random.nextLong): RDD[T] 

Return a sampled subset of this RDD. 
@param withReplacement can elements be sampled multiple times    
  (replaced when sampled out) 
@param fraction expected size of the sample as a fraction of this   
  RDD's size without replacement: probability that each element is    
  chosen; fraction must be [0, 1] with replacement: expected number   
  of times each element is chosen; fraction must be >= 0 
@param seed seed for the random number generator 

      println(nonWordSplit.distinct.sample( 
      true, 0.3, 42).take(100).mkString(","))

```

请注意，我们将`sample`函数的第三个参数设置为随机种子。我们将此函数设置为`42`，以便每次调用`sample`时都获得相同的结果，以便您的结果与本章中的结果相匹配。

这将显示以下结果：

```scala
atheist,resources
summary:,addresses,,to,atheism
keywords:,music,,thu,,11:57:19,11:57:19,gmt
distribution:,cambridge.,290

archive-name:,atheism/resources
alt-atheism-archive-  
name:,december,,,,,,,,,,,,,,,,,,,,,,addresses,addresses,,,,,,,
religion,to:,to:,,p.o.,53701.
telephone:,sell,the,,fish,on,their,cars,,with,and,written

inside.,3d,plastic,plastic,,evolution,evolution,7119,,,,,san,san,
san,mailing,net,who,to,atheist,press

aap,various,bible,,and,on.,,,one,book,is:

"the,w.p.,american,pp.,,1986.,bible,contains,ball,,based,based,
james,of

```

# 改进我们的标记化

前面的简单方法会产生大量的标记，并且不会过滤掉许多非单词字符（如标点符号）。大多数标记方案都会去除这些字符。我们可以通过使用正则表达式模式在非单词字符上拆分每个原始文档来实现这一点：

```scala
val nonWordSplit = text.flatMap(t => 
  t.split("""W+""").map(_.toLowerCase)) 
println(nonWordSplit.distinct.count)

```

这显著减少了唯一标记的数量：

```scala
130126

```

如果我们检查前几个标记，我们会发现我们已经消除了文本中大部分不太有用的字符：

```scala
println( 
nonWordSplit.distinct.sample(true, 0.3, 
  50).take(100).mkString(","))

```

您将看到以下结果显示：

```scala
jejones,ml5,w1w3s1,k29p,nothin,42b,beleive,robin,believiing,749,
steaminess,tohc4,fzbv1u,ao,
instantaneous,nonmeasurable,3465,tiems,tiems,tiems,eur,3050,pgva4,
animating,10011100b,413,randall_clark,
mswin,cannibal,cannibal,congresswoman,congresswoman,theoreticians,
34ij,logically,kxo,contoler,
contoler,13963,13963,ets,sask,sask,sask,uninjured,930420,pws,vfj,
jesuit,kocharian,6192,1tbs,octopi,
012537,012537,yc0,dmitriev,icbz,cj1v,bowdoin,computational,
jkis_ltd,
caramate,cfsmo,springer,springer,
005117,shutdown,makewindow,nowadays,mtearle,discernible,
discernible,qnh1,hindenburg,hindenburg,umaxc,
njn2e5,njn2e5,njn2e5,x4_i,x4_i,monger,rjs002c,rjs002c,rjs002c,
warms,ndallen,g45,herod,6w8rg,mqh0,suspects,
floor,flq1r,io21087,phoniest,funded,ncmh,c4uzus

```

虽然我们用于拆分文本的非单词模式效果相当不错，但我们仍然留下了数字和包含数字字符的标记。在某些情况下，数字可能是语料库的重要部分。对于我们的目的，管道中的下一步将是过滤掉数字和包含数字的标记。

我们可以通过应用另一个正则表达式模式来实现这一点，并使用它来过滤不匹配模式的标记，`val regex = """[⁰-9]*""".r`。

```scala
val regex = """[⁰-9]*""".r 
val filterNumbers = nonWordSplit.filter(token => 
  regex.pattern.matcher(token).matches) 
println(filterNumbers.distinct.count)

```

这进一步减少了标记集的大小：

```scala
84912

println(filterNumbers.distinct.sample(true, 0.3,      
50).take(100).mkString(","))

```

让我们再看一下过滤后的标记的另一个随机样本。

您将看到以下输出：

```scala
jejones,silikian,reunion,schwabam,nothin,singen,husky,tenex,
eventuality,beleive,goofed,robin,upsets,aces,nondiscriminatory,
underscored,bxl,believiing,believiing,believiing,historians,
nauseam,kielbasa,collins,noport,wargame,isv,bellevue,seetex,seetex,
negotiable,negotiable,viewed,rolled,unforeseen,dlr,museum,museum,
wakaluk,wakaluk,dcbq,beekeeper,beekeeper,beekeeper,wales,mop,win,
ja_jp,relatifs,dolphin,strut,worshippers,wertheimer,jaze,jaze,
logically,kxo,nonnemacher,sunprops,sask,bbzx,jesuit,logos,aichi,
remailing,remailing,winsor,dtn,astonished,butterfield,miserable,
icbz,icbz,poking,sml,sml,makeing,deterministic,deterministic,
deterministic,rockefeller,rockefeller,explorers,bombardments,
bombardments,bombardments,ray_bourque,hour,cfsmo,mishandles,
scramblers,alchoholic,shutdown,almanac_,bruncati,karmann,hfd,
makewindow,perptration,mtearle

```

我们可以看到我们已经删除了所有数字字符。这仍然给我们留下了一些奇怪的*单词*，但我们在这里不会太担心这些。

# 删除停用词

停用词是指在语料库中（以及大多数语料库中）几乎所有文档中都出现多次的常见词。典型的英语停用词包括 and、but、the、of 等。在文本特征提取中，通常会排除停用词。

在使用 tf-idf 加权时，加权方案实际上会为我们处理这个问题。由于停用词的 idf 得分非常低，它们往往具有非常低的 tf-idf 权重，因此重要性较低。然而，在某些情况下，对于信息检索和搜索任务，可能希望包括停用词。然而，在特征提取过程中排除停用词仍然是有益的，因为它减少了最终特征向量的维度以及训练数据的大小。

我们可以查看我们语料库中出现次数最多的一些标记，以了解其他需要排除的停用词：

```scala
val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + 
  _) 
val oreringDesc = Ordering.by(String, Int), Int 
println(tokenCounts.top(20)(oreringDesc).mkString("n"))

```

在上述代码中，我们在过滤掉数字字符后获取了标记，并生成了每个标记在整个语料库中出现次数的计数。现在我们可以使用 Spark 的 top 函数来检索按计数排名的前 20 个标记。请注意，我们需要为 top 函数提供一个排序方式，告诉 Spark 如何对我们的 RDD 元素进行排序。在这种情况下，我们希望按计数排序，因此我们将指定键值对的第二个元素。

运行上述代码片段将产生以下前几个标记：

```scala
(the,146532)
(to,75064)
(of,69034)
(a,64195)
(ax,62406)
(and,57957)
(i,53036)
(in,49402)
(is,43480)
(that,39264)
(it,33638)
(for,28600)
(you,26682)
(from,22670)
(s,22337)
(edu,21321)
(on,20493)
(this,20121)
(be,19285)
(t,18728)

```

正如我们所预期的，这个列表中有很多常见词，我们可能会将其标记为停用词。让我们创建一个包含其中一些常见词的停用词集

以及其他常见词。然后我们将在过滤掉这些停用词后查看标记：

```scala
val stopwords = Set( 
  "the","a","an","of","or","in","for","by","on","but", "is", 
  "not", "with", "as", "was", "if", 
  "they", "are", "this", "and", "it", "have", "from", "at", "my",  
  "be", "that", "to" 
val tokenCountsFilteredStopwords = tokenCounts.filter {  
  case (k, v) => !stopwords.contains(k)  
  } 

println(tokenCountsFilteredStopwords.top(20)   
  (oreringDesc).mkString("n"))

```

您将看到以下输出：

```scala
(ax,62406)
(i,53036)
(you,26682)
(s,22337)
(edu,21321)
(t,18728)
(m,12756)
(subject,12264)
(com,12133)
(lines,11835)
(can,11355)
(organization,11233)
(re,10534)
(what,9861)
(there,9689)
(x,9332)
(all,9310)
(will,9279)
(we,9227)
(one,9008)

```

您可能会注意到在这个前面的列表中仍然有相当多的常见词。在实践中，我们可能会有一个更大的停用词集。然而，我们将保留一些（部分是为了稍后使用 tf-idf 加权时常见词的影响）。

您可以在这里找到常见停用词列表：[`xpo6.com/list-of-english-stop-words/`](http://xpo6.com/list-of-english-stop-words/)

我们将使用的另一个过滤步骤是删除长度为一个字符的任何标记。这背后的原因类似于删除停用词-这些单字符标记不太可能在我们的文本模型中提供信息，并且可以进一步减少特征维度和模型大小。我们将通过另一个过滤步骤来实现这一点：

```scala
val tokenCountsFilteredSize =  
  tokenCountsFilteredStopwords.filter {  
    case (k, v) => k.size >= 2  
  } 
println(tokenCountsFilteredSize.top(20)  
  (oreringDesc).mkString("n"))

```

同样，我们将在此过滤步骤之后检查剩下的标记：

```scala
(ax,62406)
(you,26682)
(edu,21321)
(subject,12264)
(com,12133)
(lines,11835)
(can,11355)
(organization,11233)
(re,10534)
(what,9861)
(there,9689)
(all,9310)
(will,9279)
(we,9227)
(one,9008)
(would,8905)
(do,8674)
(he,8441)
(about,8336)
(writes,7844)

```

除了我们没有排除的一些常见词之外，我们看到一些潜在更具信息量的词开始出现。

# 根据频率排除术语

在标记化过程中，通常会排除语料库中整体出现非常少的术语。例如，让我们来检查语料库中出现次数最少的术语（注意我们在这里使用不同的排序方式来返回按升序排序的结果）：

```scala
val oreringAsc = Ordering.by(String, Int), Int 
println(tokenCountsFilteredSize.top(20)(oreringAsc)
  .mkString("n"))

```

您将得到以下结果：

```scala
(lennips,1)
(bluffing,1)
(preload,1)
(altina,1)
(dan_jacobson,1)
(vno,1)
(actu,1)
(donnalyn,1)
(ydag,1)
(mirosoft,1)
(xiconfiywindow,1)
(harger,1)
(feh,1)
(bankruptcies,1)
(uncompression,1)
(d_nibby,1)
(bunuel,1)
(odf,1)
(swith,1)
(lantastic,1)

```

正如我们所看到的，有许多术语在整个语料库中只出现一次。通常情况下，我们希望将我们提取的特征用于其他任务，如文档相似性或机器学习模型，只出现一次的标记对于学习来说是没有用的，因为相对于这些标记，我们将没有足够的训练数据。我们可以应用另一个过滤器来排除这些罕见的标记：

```scala
val rareTokens = tokenCounts.filter{ case (k, v) => v < 2 }.map {  
  case (k, v) => k }.collect.toSet 
val tokenCountsFilteredAll = tokenCountsFilteredSize.filter {    
  case (k, v) => !rareTokens.contains(k) } 
println(tokenCountsFilteredAll.top(20)    
  (oreringAsc).mkString("n"))

```

我们可以看到，我们剩下的标记至少在语料库中出现了两次：

```scala
(sina,2)
(akachhy,2)
(mvd,2)
(hizbolah,2)
(wendel_clark,2)
(sarkis,2)
(purposeful,2)
(feagans,2)
(wout,2)
(uneven,2)
(senna,2)
(multimeters,2)
(bushy,2)
(subdivided,2)
(coretest,2)
(oww,2)
(historicity,2)
(mmg,2)
(margitan,2)
(defiance,2)

```

现在，让我们统计一下唯一标记的数量：

```scala
println(tokenCountsFilteredAll.count)

```

您将看到以下输出：

```scala
51801

```

正如我们所看到的，通过在我们的标记化流程中应用所有过滤步骤，我们已将特征维度从`402,978`减少到`51,801`。

现在，我们可以将所有过滤逻辑组合成一个函数，然后将其应用到我们 RDD 中的每个文档：

```scala
def tokenize(line: String): Seq[String] = { 
  line.split("""W+""") 
    .map(_.toLowerCase) 
    .filter(token => regex.pattern.matcher(token).matches) 
    .filterNot(token => stopwords.contains(token)) 
    .filterNot(token => rareTokens.contains(token)) 
    .filter(token => token.size >= 2) 
    .toSeq 
}

```

我们可以检查这个函数是否给我们相同的结果，使用以下代码片段：

```scala
println(text.flatMap(doc => tokenize(doc)).distinct.count)

```

这将输出`51801`，给我们与逐步流程相同的唯一标记计数。

我们可以按如下方式对 RDD 中的每个文档进行标记化：

```scala
val tokens = text.map(doc => tokenize(doc)) 
println(tokens.first.take(20))

```

您将看到类似以下的输出，显示我们第一个文档的标记化版本的前部分：

```scala
WrappedArray(mathew, mantis, co, uk, subject, alt, atheism, 
faq, atheist, resources, summary, books, addresses, music,         
anything, related, atheism, keywords, faq)

```

# 关于词干的一点说明

文本处理和标记化中的一个常见步骤是**词干提取**。这是将整个单词转换为**基本形式**（称为**词干**）的过程。例如，复数可能会转换为单数（*dogs*变成*dog*），而*walking*和*walker*这样的形式可能会变成*walk*。词干提取可能会变得非常复杂，通常需要专门的 NLP 或搜索引擎软件（例如 NLTK、OpenNLP 和 Lucene 等）来处理。在这个例子中，我们将忽略词干提取。

对词干提取的全面处理超出了本书的范围。您可以在[`en.wikipedia.org/wiki/Stemming`](http://en.wikipedia.org/wiki/Stemming)找到更多细节。

# 特征哈希

首先，我们解释什么是特征哈希，以便更容易理解下一节中的 tf-idf 模型。

特征哈希将字符串或单词转换为固定长度的向量，这样可以更容易地处理文本。

Spark 目前使用 Austin Appleby 的 MurmurHash 3 算法（MurmurHash3_x86_32）将文本哈希为数字。

您可以在这里找到实现

```scala
private[spark] def murmur3Hash(term: Any): Int = {
  term match {
  case null => seed
  case b: Boolean => hashInt(if (b) 1 else 0, seed)
  case b: Byte => hashInt(b, seed)
  case s: Short => hashInt(s, seed)
  case i: Int => hashInt(i, seed)
  case l: Long => hashLong(l, seed)
  case f: Float => hashInt(java.lang.Float
    .floatToIntBits(f), seed)
  case d: Double => hashLong(java.lang.Double.
    doubleToLongBits(d), seed)
  case s: String => val utf8 = UTF8String.fromString(s)
    hashUnsafeBytes(utf8.getBaseObject, utf8.getBaseOffset, 
    utf8.numBytes(), seed)
  case _ => throw new SparkException( 
  "HashingTF with murmur3 algorithm does not " +
    s"support type ${term.getClass.getCanonicalName} of input  
  data.")
  }
}

```

请注意，函数`hashInt`、`hasLong`等是从`Util.scala`中调用的

# 构建 tf-idf 模型

现在，我们将使用 Spark ML 将每个文档（以处理后的标记形式）转换为向量表示。第一步将是使用`HashingTF`实现，它利用特征哈希将输入文本中的每个标记映射到术语频率向量中的索引。然后，我们将计算全局 IDF，并使用它将术语频率向量转换为 tf-idf 向量。

对于每个标记，索引将是标记的哈希值（依次映射到特征向量的维度）。每个标记的值将是该标记的 tf-idf 加权值（即，术语频率乘以逆文档频率）。

首先，我们将导入我们需要的类并创建我们的`HashingTF`实例，传入一个`dim`维度参数。虽然默认的特征维度是 2²⁰（大约 100 万），我们将选择 2¹⁸（大约 26 万），因为大约有 5 万个标记，我们不应该遇到显著数量的哈希碰撞，而较小的维度对于说明目的来说更加节省内存和处理资源：

```scala
import org.apache.spark.mllib.linalg.{ SparseVector => SV } 
import org.apache.spark.mllib.feature.HashingTF 
import org.apache.spark.mllib.feature.IDF 
val dim = math.pow(2, 18).toInt 
val hashingTF = new HashingTF(dim) 
val tf = hashingTF.transform(tokens) 
tf.cache

```

请注意，我们使用`SV`的别名导入了 MLlib 的`SparseVector`。这是因为稍后，我们将使用 Breeze 的`linalg`模块，它本身也导入`SparseVector`。这样，我们将避免命名空间冲突。

`HashingTF`的`transform`函数将每个输入文档（即标记序列）映射到 MLlib 的`Vector`。我们还将调用`cache`将数据固定在内存中，以加速后续操作。

让我们检查转换后数据集的第一个元素：

请注意，`HashingTF.transform`返回一个`RDD[Vector]`，因此我们将返回的结果转换为 MLlib`SparseVector`的实例。

`transform`方法也可以通过接受一个`Iterable`参数（例如，作为`Seq[String]`的文档）来处理单个文档。这将返回一个单一的向量。

```scala
val v = tf.first.asInstanceOf[SV] 
println(v.size) 
println(v.values.size) 
println(v.values.take(10).toSeq) 
println(v.indices.take(10).toSeq)

```

您将看到以下输出显示：

```scala
262144
706
WrappedArray(1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0)
WrappedArray(313, 713, 871, 1202, 1203, 1209, 1795, 1862, 3115,     
3166)

```

我们可以看到每个稀疏向量的特征频率的维度为 262,144（或者我们指定的 2¹⁸）。然而，向量中非零条目的数量只有 706。输出的最后两行显示了向量中前几个条目的频率计数和索引。

现在，我们将通过创建一个新的`IDF`实例并调用`fit`方法来计算语料库中每个术语的逆文档频率。然后，我们将通过`IDF`的`transform`函数将我们的术语频率向量转换为 tf-idf 向量：

```scala
val idf = new IDF().fit(tf) 
val tfidf = idf.transform(tf) 
val v2 = tfidf.first.asInstanceOf[SV] 
println(v2.values.size) 
println(v2.values.take(10).toSeq) 
println(v2.indices.take(10).toSeq)

```

当您检查 tf-idf 转换后向量的 RDD 中的第一个元素时，您将看到类似于这里显示的输出：

```scala
706
WrappedArray(2.3869085659322193, 4.670445463955571, 
6.561295835827856, 4.597686109673142,  ...
WrappedArray(313, 713, 871, 1202, 1203, 1209, 1795, 1862, 3115,     
3166)

```

我们可以看到非零条目的数量没有改变（为`706`），术语的向量索引也没有改变。改变的是每个术语的值。早些时候，这些值代表了文档中每个术语的频率，但现在，新值代表了由 IDF 加权的频率。

当我们执行以下两行时，IDF 加权就出现了

```scala
val idf = new IDF().fit(tf) 
val tfidf = idf.transform(tf)

```

# 分析 tf-idf 加权

接下来，让我们调查一些术语的 tf-idf 加权，以说明术语的普遍性或稀有性的影响。

首先，我们可以计算整个语料库中的最小和最大 tf-idf 权重：

```scala
val minMaxVals = tfidf.map { v => 
  val sv = v.asInstanceOf[SV] 
  (sv.values.min, sv.values.max) 
} 
val globalMinMax = minMaxVals.reduce { case ((min1, max1), 
  (min2, max2)) => 
  (math.min(min1, min2), math.max(max1, max2)) 
} 
println(globalMinMax)

```

正如我们所看到的，最小的 tf-idf 是零，而最大的 tf-idf 显着更大：

```scala
(0.0,66155.39470409753)

```

我们现在将探讨附加到各种术语的 tf-idf 权重。在停用词的上一节中，我们过滤掉了许多经常出现的常见术语。请记住，我们没有删除所有这些潜在的停用词。相反，我们在语料库中保留了一些术语，以便我们可以说明应用 tf-idf 加权方案对这些术语的影响。

Tf-idf 加权将倾向于为常见术语分配较低的权重。为了证明这一点，我们可以计算我们先前计算的顶部出现列表中的一些术语的 tf-idf 表示，例如`you`，`do`和`we`：

```scala
val common = sc.parallelize(Seq(Seq("you", "do", "we"))) 
val tfCommon = hashingTF.transform(common) 
val tfidfCommon = idf.transform(tfCommon) 
val commonVector = tfidfCommon.first.asInstanceOf[SV] 
println(commonVector.values.toSeq)

```

如果我们形成这篇文章的 tf-idf 向量表示，我们会看到每个术语分配的以下值。请注意，由于特征散列，我们不确定哪个术语代表什么。但是，这些值说明了对这些术语应用的加权相对较低：

```scala
WrappedArray(0.9965359935704624, 1.3348773448236835, 
0.5457486182039175)

```

现在，让我们将相同的转换应用于一些我们可能直观地认为与特定主题或概念更相关的不太常见的术语：

```scala
val uncommon = sc.parallelize(Seq(Seq("telescope", 
  "legislation", "investment"))) 
val tfUncommon = hashingTF.transform(uncommon) 
val tfidfUncommon = idf.transform(tfUncommon) 
val uncommonVector = tfidfUncommon.first.asInstanceOf[SV] 
println(uncommonVector.values.toSeq)

```

从以下结果中我们可以看到，tf-idf 加权确实比更常见的术语要高得多：

```scala
WrappedArray(5.3265513728351666, 5.308532867332488, 
5.483736956357579)

```

# 使用 tf-idf 模型

尽管我们经常提到训练 tf-idf 模型，但实际上它是一个特征提取过程或转换，而不是一个机器学习模型。Tf-idf 加权通常用作其他模型的预处理步骤，例如降维、分类或回归。

为了说明 tf-idf 加权的潜在用途，我们将探讨两个例子。第一个是使用 tf-idf 向量计算文档相似性，而第二个涉及使用 tf-idf 向量作为输入特征训练多标签分类模型。

# 20 个新闻组数据集和 tf-idf 特征的文档相似性

您可能还记得第五章中的*使用 Spark 构建推荐引擎*，两个向量之间的相似度可以使用距离度量来计算。两个向量越接近（即距离度量越小），它们就越相似。我们用于计算电影之间相似度的一种度量是余弦相似度。

就像我们为电影所做的那样，我们也可以计算两个文档之间的相似性。使用 tf-idf，我们已将每个文档转换为向量表示。因此，我们可以使用与我们用于比较两个文档的电影向量相同的技术。

直觉上，如果两个文档共享许多术语，我们可能期望这两个文档彼此更相似。相反，如果它们各自包含许多彼此不同的术语，我们可能期望这两个文档更不相似。由于我们通过计算两个向量的点积来计算余弦相似度，而每个向量由每个文档中的术语组成，我们可以看到具有高重叠术语的文档将倾向于具有更高的余弦相似度。

现在，我们可以看到 tf-idf 在起作用。我们可能合理地期望，即使非常不同的文档也可能包含许多重叠的相对常见的术语（例如，我们的停用词）。然而，由于 tf-idf 加权较低，这些术语对点积的影响不大，因此对计算的相似度也没有太大影响。

例如，我们可能期望从`冰球`新闻组中随机选择的两条消息之间相对相似。让我们看看是否是这种情况：

```scala
val hockeyText = rdd.filter { case (file, text) => 
  file.contains("hockey") } 
val hockeyTF = hockeyText.mapValues(doc => 
  hashingTF.transform(tokenize(doc))) 
val hockeyTfIdf = idf.transform(hockeyTF.map(_._2))

```

在前面的代码中，我们首先过滤了原始输入 RDD，只保留了冰球主题内的消息。然后应用了我们的标记化和词项频率转换函数。请注意，使用的`transform`方法是适用于单个文档（以`Seq[String]`形式）的版本，而不是适用于 RDD 文档的版本。

最后，我们应用了`IDF`转换（请注意，我们使用的是已经在整个语料库上计算过的相同 IDF）。

一旦我们有了我们的`冰球`文档向量，我们可以随机选择其中的两个向量，并计算它们之间的余弦相似度（就像之前一样，我们将使用 Breeze 进行线性代数功能，特别是首先将我们的 MLlib 向量转换为 Breeze`SparseVector`实例）：

```scala
import breeze.linalg._ 
val hockey1 = hockeyTfIdf.sample( 
  true, 0.1, 42).first.asInstanceOf[SV] 
val breeze1 = new SparseVector(hockey1.indices,
  hockey1.values, hockey1.size) 
val hockey2 = hockeyTfIdf.sample(true, 0.1, 
  43).first.asInstanceOf[SV] 
val breeze2 = new SparseVector(hockey2.indices,
  hockey2.values, hockey2.size) 
val cosineSim = breeze1.dot(breeze2) / 
  (norm(breeze1) * norm(breeze2)) 
println(cosineSim)

```

我们可以看到文档之间的余弦相似度大约为 0.06：

```scala
0.06700095047242809

```

虽然这可能看起来相当低，但要记住，由于处理文本数据时通常会出现大量唯一术语，因此我们特征的有效维度很高。因此，我们可以期望，即使两个文档是关于相同主题的，它们之间的术语重叠也可能相对较低，因此绝对相似度得分也会较低。

相比之下，我们可以将此相似度得分与使用相同方法在`计算机图形`新闻组中随机选择的另一个文档与我们的`冰球`文档之间计算的相似度进行比较：

```scala
val graphicsText = rdd.filter { case (file, text) => 
  file.contains("comp.graphics") } 
val graphicsTF = graphicsText.mapValues(doc => 
  hashingTF.transform(tokenize(doc))) 
val graphicsTfIdf = idf.transform(graphicsTF.map(_._2)) 
val graphics = graphicsTfIdf.sample(true, 0.1, 
  42).first.asInstanceOf[SV] 
val breezeGraphics = new SparseVector(graphics.indices, 
  graphics.values, graphics.size) 
val cosineSim2 = breeze1.dot(breezeGraphics) / (norm(breeze1) * 
  norm(breezeGraphics)) 
println(cosineSim2)

```

余弦相似度显著较低，为`0.0047`：

```scala
0.001950124251275256

```

最后，很可能来自另一个与体育相关的主题的文档与我们的`冰球`文档更相似，而不像来自与计算机相关的主题的文档。但是，我们可能预期`棒球`文档与我们的`冰球`文档不太相似。让我们通过计算`棒球`新闻组中的随机消息与我们的`冰球`文档之间的相似度来看看是否如此：

```scala
// compare to sport.baseball topic 
val baseballText = rdd.filter { case (file, text) => 
  file.contains("baseball") } 
val baseballTF = baseballText.mapValues(doc => 
  hashingTF.transform(tokenize(doc))) 
val baseballTfIdf = idf.transform(baseballTF.map(_._2)) 
val baseball = baseballTfIdf.sample(true, 0.1, 
  42).first.asInstanceOf[SV] 
val breezeBaseball = new SparseVector(baseball.indices, 
  baseball.values, baseball.size) 
val cosineSim3 = breeze1.dot(breezeBaseball) / (norm(breeze1) * 
   norm(breezeBaseball)) 
println(cosineSim3)

```

事实上，正如我们预期的那样，我们发现`棒球`和`冰球`文档的余弦相似度为`0.05`，这显著高于`计算机图形`文档，但也略低于另一个`冰球`文档：

```scala
0.05047395039466008

```

源代码：

[`github.com/ml-resources/spark-ml/blob/branch-ed2/Chapter_10/scala-2.0.x/src/main/scala/TFIDFExtraction.scala`](https://github.com/ml-resources/spark-ml/blob/branch-ed2/Chapter_10/scala-2.0.x/src/main/scala/TFIDFExtraction.scala)

# 使用 tf-idf 在 20 个新闻组数据集上训练文本分类器

使用 tf-idf 向量时，我们预期余弦相似度度量将捕捉文档之间的相似性，基于它们之间的术语重叠。类似地，我们预期机器学习模型，如分类器，将能够学习每个术语的加权；这将使其能够区分不同类别的文档。也就是说，应该可以学习到存在（和加权）某些术语与特定主题之间的映射。

在 20 个新闻组的例子中，每个新闻组主题都是一个类别，我们可以使用我们的 tf-idf 转换后的向量来训练分类器。

由于我们正在处理一个多类分类问题，我们将在 MLlib 中使用朴素贝叶斯模型，该模型支持多个类别。作为第一步，我们将导入我们将使用的 Spark 类：

```scala
import org.apache.spark.mllib.regression.LabeledPoint 
import org.apache.spark.mllib.classification.NaiveBayes 
import org.apache.spark.mllib.evaluation.MulticlassMetrics.

```

我们将保留我们的聚类代码在一个名为`文档聚类`的对象中

```scala
object DocumentClassification { 

  def main(args: Array[String]) { 
    val sc = new SparkContext("local[2]", "") 
    ... 
}

```

接下来，我们需要提取 20 个主题并将它们转换为类映射。我们可以像对*1-of-K*特征编码一样做，为每个类分配一个数字索引：

```scala
val newsgroupsMap = 
  newsgroups.distinct.collect().zipWithIndex.toMap 
val zipped = newsgroups.zip(tfidf) 
val train = zipped.map { case (topic, vector) => 
  LabeledPoint(newsgroupsMap(topic), vector) } 
train.cache

```

在前面的代码片段中，我们取了`newsgroups` RDD，其中每个元素都是主题，并使用`zip`函数将其与我们的 tf-idf 向量 RDD 中的每个元素组合在一起。然后，我们在我们的新压缩 RDD 中的每个键值元素上进行映射，并创建一个`LabeledPoint`实例，其中`label`是类索引，`features`是 tf-idf 向量。

请注意，`zip`操作符假定每个 RDD 具有相同数量的分区以及每个分区中相同数量的元素。如果不是这种情况，它将失败。我们可以做出这种假设，因为我们实际上已经通过对相同原始 RDD 进行一系列`map`转换来创建了`tfidf` RDD 和`newsgroups` RDD，并保留了分区结构。

现在我们有了正确形式的输入 RDD，我们可以简单地将其传递给朴素贝叶斯的`train`函数：

```scala
val model = NaiveBayes.train(train, lambda = 0.1)

```

让我们评估模型在测试数据集上的性能。我们将从`20news-bydate-test`目录加载原始测试数据，再次使用`wholeTextFiles`将每条消息读入 RDD 元素。然后，我们将从文件路径中提取类标签，方式与我们对`newsgroups` RDD 所做的方式相同。

```scala
val testPath = "/PATH/20news-bydate-test/*" 
val testRDD = sc.wholeTextFiles(testPath) 
val testLabels = testRDD.map { case (file, text) => 
  val topic = file.split("/").takeRight(2).head 
  newsgroupsMap(topic) 
}

```

对测试数据集中的文本进行转换的过程与训练数据相同-我们将应用我们的`tokenize`函数，然后进行词项频率转换，然后再次使用从训练数据中计算的相同 IDF 来将 TF 向量转换为 tf-idf 向量。最后，我们将测试类标签与 tf-idf 向量进行压缩，并创建我们的测试`RDD[LabeledPoint]`：

```scala
val testTf = testRDD.map { case (file, text) => 
  hashingTF.transform(tokenize(text)) } 
val testTfIdf = idf.transform(testTf) 
val zippedTest = testLabels.zip(testTfIdf) 
val test = zippedTest.map { case (topic, vector) => 
  LabeledPoint(topic, vector) }

```

请注意，重要的是我们使用训练集的 IDF 来转换测试数据，因为这样可以更真实地估计模型在新数据上的性能，新数据可能包含模型尚未训练过的术语。如果基于测试数据集重新计算 IDF 向量，这将是“作弊”，更重要的是，可能会导致通过交叉验证选择的最佳模型参数的不正确估计。

现在，我们准备计算模型的预测和真实类标签。我们将使用此 RDD 来计算模型的准确性和多类加权 F-度量：

```scala
val predictionAndLabel = test.map(p =>       
  (model.predict(p.features),   p.label)) 
val accuracy = 1.0 * predictionAndLabel.filter
  (x => x._1 == x._2).count() / test.count() 
val metrics = new MulticlassMetrics(predictionAndLabel) 
println(accuracy) 
println(metrics.weightedFMeasure)

```

加权 F-度量是精确度和召回率性能的综合度量（类似于 ROC 曲线下面积，值越接近 1.0 表示性能越好），然后通过在类别之间进行加权平均来组合。

我们可以看到，我们简单的多类朴素贝叶斯模型的准确性和 F-度量都接近 80％：

```scala
0.7928836962294211
0.7822644376431702

```

# 评估文本处理的影响

文本处理和 tf-idf 加权是旨在减少原始文本数据的维度并提取一些结构的特征提取技术的例子。通过比较在原始文本数据上训练的模型与在处理和 tf-idf 加权文本数据上训练的模型的性能，我们可以看到应用这些处理技术的影响。

# 比较 20 个新闻组数据集上的原始特征和处理后的 tf-idf 特征

在这个例子中，我们将简单的哈希词项频率转换应用于使用文档文本的简单空格拆分获得的原始文本标记。我们将在这些数据上训练一个模型，并评估在测试集上的性能，就像我们对使用 tf-idf 特征训练的模型一样：

```scala
val rawTokens = rdd.map { case (file, text) => text.split(" ") } 
val rawTF = texrawTokenst.map(doc => hashingTF.transform(doc)) 
val rawTrain = newsgroups.zip(rawTF).map { case (topic, vector)  
  => LabeledPoint(newsgroupsMap(topic), vector) } 
val rawModel = NaiveBayes.train(rawTrain, lambda = 0.1) 
val rawTestTF = testRDD.map { case (file, text) => 
  hashingTF.transform(text.split(" ")) } 
val rawZippedTest = testLabels.zip(rawTestTF) 
val rawTest = rawZippedTest.map { case (topic, vector) => 
  LabeledPoint(topic, vector) } 
val rawPredictionAndLabel = rawTest.map(p => 
  (rawModel.predict(p.features), p.label)) 
val rawAccuracy = 1.0 * rawPredictionAndLabel.filter(x => x._1 
  == x._2).count() / rawTest.count() 
println(rawAccuracy) 
val rawMetrics = new MulticlassMetrics(rawPredictionAndLabel) 
println(rawMetrics.weightedFMeasure)

```

也许令人惊讶的是，原始模型表现得相当不错，尽管准确性和 F-度量都比 tf-idf 模型低几个百分点。这在一定程度上也反映了朴素贝叶斯模型适合以原始频率计数形式的数据。

```scala
0.7661975570897503
0.7653320418573546

```

# 使用 Spark 2.0 进行文本分类

在本节中，我们将使用 libsvm 版本的*20newsgroup*数据，使用 Spark DataFrame-based API 对文本文档进行分类。在当前版本的 Spark 中，支持 libsvm 版本 3.22 ([`www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/`](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/))

从以下链接下载 libsvm 格式的数据并将输出文件夹复制到 Spark-2.0.x 下。

访问以下链接以获取*20newsgroup libsvm*数据：[`1drv.ms/f/s!Av6fk5nQi2j-iF84quUlDnJc6G6D`](https://1drv.ms/f/s!Av6fk5nQi2j-iF84quUlDnJc6G6D)

从`org.apache.spark.ml`中导入适当的包并创建 Wrapper Scala：

```scala
package org.apache.spark.examples.ml 

import org.apache.spark.SparkConf 
import org.apache.spark.ml.classification.NaiveBayes 
import        

org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 

import org.apache.spark.sql.SparkSession 

object DocumentClassificationLibSVM { 
  def main(args: Array[String]): Unit = { 

  } 
}

```

接下来，我们将将`libsvm`数据加载到 Spark DataFrame 中：

```scala
val spConfig = (new SparkConf).setMaster("local")
  .setAppName("SparkApp") 
val spark = SparkSession 
  .builder() 
  .appName("SparkRatingData").config(spConfig) 
  .getOrCreate() 

val data = spark.read.format("libsvm").load("./output/20news-by-
  date-train-libsvm/part-combined") 

val Array(trainingData, testData) = data.randomSplit(Array(0.7,
  0.3), seed = 1L)

```

从`org.apache.spark.ml.classification.NaiveBayes`类中实例化`NaiveBayes`模型并训练模型：

```scala
val model = new NaiveBayes().fit(trainingData) 
val predictions = model.transform(testData) 
predictions.show()

```

以下表格是预测 DataFrame 的输出`.show()`命令：

```scala
+----+-------------------+--------------------+-----------------+----------+
|label|     features     |    rawPrediction   |   probability   |prediction|
+-----+------------------+--------------------+-----------------+----------+
|0.0|(262141,[14,63,64...|[-8972.9535882773...|[1.0,0.0,1.009147...| 0.0|
|0.0|(262141,[14,329,6...|[-5078.5468878602...|[1.0,0.0,0.0,0.0,...| 0.0|
|0.0|(262141,[14,448,5...|[-3376.8302696656...|[1.0,0.0,2.138643...| 0.0|
|0.0|(262141,[14,448,5...|[-3574.2782864683...|[1.0,2.8958758424...| 0.0|
|0.0|(262141,[14,535,3...|[-5001.8808481928...|[8.85311976855360...| 12.0|
|0.0|(262141,[14,573,8...|[-5950.1635030844...|[1.0,0.0,1.757049...| 0.0|
|0.0|(262141,[14,836,5...|[-8795.2012408412...|[1.0,0.0,0.0,0.0,...| 0.0|
|0.0|(262141,[14,991,2...|[-1892.8829282793...|[0.99999999999999...| 0.0|
|0.0|(262141,[14,1176,...|[-4746.2275710890...|[1.0,5.8201E-319,...| 0.0|
|0.0|(262141,[14,1379,...|[-7104.8373572933...|[1.0,8.9577444139...| 0.0|
|0.0|(262141,[14,1582,...|[-5473.6206675848...|[1.0,5.3185120345...| 0.0|
|0.0|(262141,[14,1836,...|[-11289.582479676...|[1.0,0.0,0.0,0.0,...| 0.0|
|0.0|(262141,[14,2325,...|[-3957.9187837274...|[1.0,2.1880375223...| 0.0|
|0.0|(262141,[14,2325,...|[-7131.2028421844...|[1.0,2.6110663778...| 0.0|
|0.0|(262141,[14,3033,...|[-3014.6430319605...|[1.0,2.6341580467...| 0.0|
|0.0|(262141,[14,4335,...|[-8283.7207917560...|[1.0,8.9559011053...| 0.0|
|0.0|(262141,[14,5173,...|[-6811.3466537480...|[1.0,7.2593916980...| 0.0|
|0.0|(262141,[14,5232,...|[-2752.8846541292...|[1.0,1.8619374091...| 0.0|
|0.0|(262141,[15,5173,...|[-8741.7756643949...|[1.0,0.0,2.606005...| 0.0|
|0.0|(262141,[168,170,...|[-41636.025208445...|[1.0,0.0,0.0,0.0,...| 0.0|
+----+--------------------+-------------------+-------------------+--------+

```

测试模型的准确性：

```scala
val accuracy = evaluator.evaluate(predictions) 
println("Test set accuracy = " + accuracy) 
spark.stop()

```

如下输出所示，该模型的准确性高于`0.8`：

```scala
Test set accuracy = 0.8768458357944477
Accuracy is better as the Naive Bayes implementation has improved 
from Spark 1.6 to Spark 2.0

```

# Word2Vec 模型

到目前为止，我们已经使用了词袋向量，可选地使用一些加权方案，如 tf-idf 来表示文档中的文本。另一个最近流行的模型类别与将单个单词表示为向量有关。

这些模型通常在某种程度上基于语料库中单词之间的共现统计。一旦计算出向量表示，我们可以以类似于使用 tf-idf 向量的方式使用这些向量（例如，将它们用作其他机器学习模型的特征）。这样一个常见的用例是根据它们的向量表示计算两个单词之间的相似性。

Word2Vec 是指这些模型中的一个特定实现，通常被称为**分布式向量表示**。MLlib 模型使用**skip-gram**模型，该模型旨在学习考虑单词出现上下文的向量表示。

虽然对 Word2Vec 的详细处理超出了本书的范围，但 Spark 的文档[`spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec`](http://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec)中包含有关算法的更多详细信息以及参考实现的链接。

Word2Vec 的主要学术论文之一是*Tomas Mikolov*，*Kai Chen*，*Greg Corrado*和*Jeffrey Dean*。*Efficient Estimation of Word Representations in Vector Space*。*在 2013 年 ICLR 研讨会论文集中*。

它可以在[`arxiv.org/pdf/1301.3781.pdf`](http://arxiv.org/pdf/1301.3781.pdf)上找到。

在词向量表示领域的另一个最近的模型是 GloVe，网址为[`www-nlp.stanford.edu/projects/glove/`](http://www-nlp.stanford.edu/projects/glove/)。

您还可以利用第三方库进行词性标注。例如，Stanford NLP 库可以连接到 scala 代码中。有关如何执行此操作的更多详细信息，请参阅此讨论线程([`stackoverflow.com/questions/18416561/pos-tagging-in-scala`](http://stackoverflow.com/questions/18416561/pos-tagging-in-scala))。

# 在 20 Newsgroups 数据集上使用 Spark MLlib 的 Word2Vec

在 Spark 中训练 Word2Vec 模型相对简单。我们将传入一个 RDD，其中每个元素都是一个术语序列。我们可以使用我们已经创建的标记化文档的 RDD 作为模型的输入。

```scala
object Word2VecMllib {
  def main(args: Array[String]) {
  val sc = new SparkContext("local[2]", "Word2Vector App")
  val path = "./data/20news-bydate-train/alt.atheism/*"
  val rdd = sc.wholeTextFiles(path)
  val text = rdd.map { case (file, text) => text }
  val newsgroups = rdd.map { case (file, text) =>             
    file.split("/").takeRight(2).head }
  val newsgroupsMap =       
    newsgroups.distinct.collect().zipWithIndex.toMap
  val dim = math.pow(2, 18).toInt
  var tokens = text.map(doc => TFIDFExtraction.tokenize(doc))
  import org.apache.spark.mllib.feature.Word2Vec
  val word2vec = new Word2Vec()
  val word2vecModel = word2vec.fit(tokens)
    word2vecModel.findSynonyms("philosophers", 5).foreach(println)
  sc.stop()
  }
}

```

我们的代码在 Scala 对象`Word2VecMllib`中：

```scala
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{SparseVector => SV}
object Word2VecMllib {
  def main(args: Array[String]) {
  }
}

```

让我们从加载文本文件开始：

```scala
val sc = new SparkContext("local[2]", "Word2Vector App")
val path = "./data/20news-bydate-train/alt.atheism/*"
val rdd = sc.wholeTextFiles(path)
val text = rdd.map { case (file, text) => text }
val newsgroups = rdd.map { case (file, text) =>
  file.split("/").takeRight(2).head }
val newsgroupsMap =      
  newsgroups.distinct.collect().zipWithIndex.toMap
val dim = math.pow(2, 18).toInt
  var tokens = text.map(doc => TFIDFExtraction.tokenize(doc))

```

我们使用 tf-idf 创建的标记作为 Word2Vec 的起点。让我们首先初始化对象并设置一个种子：

```scala
import org.apache.spark.mllib.feature.Word2Vec
 val word2vec = new Word2Vec()

```

现在，让我们通过在 tf-idf 标记上调用`word2vec.fit()`来创建模型：

```scala
val word2vecModel = word2vec.fit(tokens)

```

在训练模型时，您将看到一些输出。

训练完成后，我们可以轻松地找到给定术语的前 20 个同义词（即，与输入术语最相似的术语，由单词向量之间的余弦相似性计算得出）。例如，要找到与`philosopher`最相似的 20 个术语，请使用以下代码行：

```scala
word2vecModel.findSynonyms(philosophers", 5).foreach(println)
sc.stop()

```

从以下输出中可以看出，大多数术语与曲棍球或其他相关：

```scala
(year,0.8417112940969042) (motivations,0.833017707021745) (solution,0.8284719617235932) (whereas,0.8242997325042509) (formed,0.8042383351975712)

```

# 在 20 个新闻组数据集上使用 Spark ML 的 Word2Vec

在本节中，我们将看看如何使用 Spark ML DataFrame 和 Spark 2.0.X 中的新实现来创建 Word2Vector 模型。

我们将从数据集创建一个 DataFrame：

```scala
val spConfig = (new SparkConf).setMaster("local").setAppName("SparkApp")
val spark = SparkSession
  .builder
  .appName("Word2Vec Sample").config(spConfig)
  .getOrCreate()
import spark.implicits._
val rawDF = spark.sparkContext
  .wholeTextFiles("./data/20news-bydate-train/alt.atheism/*")
  val temp = rawDF.map( x => {
    (x._2.filter(_ >= ' ').filter(! _.toString.startsWith("(")) )
    })
  val textDF = temp.map(x => x.split(" ")).map(Tuple1.apply)
    .toDF("text")

```

接下来将创建`Word2Vec`类，并在上面创建的 DataFrame `textDF`上训练模型：

```scala
val word2Vec = new Word2Vec()
  .setInputCol("text")
  .setOutputCol("result")
  .setVectorSize(3)
  .setMinCount(0)
val model = word2Vec.fit(textDF)
val result = model.transform(textDF)
  result.select("result").take(3).foreach(println)
)

```

现在让我们尝试找一些`hockey`的同义词：

以下

```scala
val ds = model.findSynonyms("philosophers", 5).select("word")
  ds.rdd.saveAsTextFile("./output/philiosphers-synonyms" +             System.nanoTime())
  ds.show(

```

将生成以下输出：

```scala
 +--------------+ | word         | +--------------+ | Fess         | | guide        | |validinference| | problems.    | | paperback    | +--------------+

```

正如您所看到的，结果与我们使用 RDD 得到的结果非常不同。这是因为 Spark 1.6 和 Spark 2.0/2.1 中的 Word2Vector 转换两种实现不同。

# 总结

在本章中，我们深入研究了更复杂的文本处理，并探索了 MLlib 的文本特征提取能力，特别是 tf-idf 术语加权方案。我们介绍了使用生成的 tf-idf 特征向量来计算文档相似性和训练新闻组主题分类模型的示例。最后，您学会了如何使用 MLlib 的尖端 Word2Vec 模型来计算文本语料库中单词的向量表示，并使用训练好的模型找到具有类似给定单词的上下文含义的单词。我们还研究了如何在 Spark ML 中使用 Word2Vec

在下一章中，我们将看一看在线学习，您将学习 Spark Streaming 与在线学习模型的关系。
