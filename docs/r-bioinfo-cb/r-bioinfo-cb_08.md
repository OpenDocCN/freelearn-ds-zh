# 第八章：与数据库和远程数据源的工作

大规模模式生物有机体测序项目，如**人类基因组计划**（**HGP**）或 1001 植物基因组测序项目，已经使大量基因组数据公开可用。同样，个别实验室的开放数据共享也使基因组和转录组的原始测序数据得到了广泛的共享。通过编程处理这些数据意味着可能需要解析或本地存储一些非常庞大或复杂的文件。因此，许多努力已投入使这些资源通过 API 和其他可查询接口（如 BioMart）尽可能易于访问。在本章中，我们将介绍一些食谱，帮助我们在不下载整个基因组文件的情况下搜索注释，并能够在数据库中找到相关信息。我们还将了解如何在代码中提取实验的原始读取数据，并借此机会研究如何对下载的数据进行质量控制。

本章将覆盖以下食谱：

+   从 BioMart 检索基因和基因组注释

+   检索和处理 SNP

+   获取基因本体信息

+   从 SRA/ENA 查找实验和读取数据

+   对高通量测序读取进行质量控制和过滤

+   完成使用外部程序的读到参考比对

+   可视化读到参考比对的质量控制图

# 技术要求

你所需的示例数据可以在本书的 GitHub 仓库中找到，地址为 [`github.com/PacktPublishing/R-Bioinformatics-Cookbook`](https://github.com/PacktPublishing/R-Bioinformatics-Cookbook)[.](https://github.com/danmaclean/R_Bioinformatics_Cookbook) 如果你想按原样使用代码示例，你需要确保这些数据位于你的工作目录的子目录中。

以下是你需要的 R 包。一般来说，你可以使用`install.packages("package_name")`来安装这些包。在`Bioconductor`下列出的包需要使用专用的安装器进行安装。具体安装方法将在本节中描述。如果需要进一步操作，安装过程将在包使用的食谱中进行描述：

+   `Bioconductor`

    +   `biomaRt`

    +   `ramwas`

    +   `ShortRead`

    +   `SRAdb`

Bioconductor 非常庞大，并且拥有自己的安装管理器。你可以使用以下代码来安装管理器：

```py
if (!requireNamespace("BiocManager"))
    install.packages("BiocManager")
```

然后，你可以使用以下代码安装这些包：

```py
BiocManager::install("package_name")
```

更多信息请参见 [`www.bioconductor.org/install/`](https://www.bioconductor.org/install/)。

通常，在 R 中，用户会加载一个库并直接通过名称使用其中的函数。这在交互式会话中非常方便，但在加载多个包时可能会导致混淆。为了明确我在某一时刻使用的是哪个包和函数，我会偶尔使用`packageName::functionName()`的惯例。

有时，在配方中间，我会中断代码，这样您就可以看到一些中间输出或者重要对象的结构。每当这种情况发生时，您将看到每行以`##`（双重井号）符号开头的代码块。考虑以下命令：

`letters[1:5]`

这将给我们以下输出：

`## a b c d e`

请注意，输出行前缀为`##`。

# 从`BioMart`检索基因和基因组注释

一旦准备好了某个基因组序列的草稿，就会进行大量的生物信息学工作，以找到基因和其他功能特征或重要的基因座。这些注释很多，执行和验证都很困难，通常需要大量的专业知识和时间，并且不希望重复。因此，基因组项目联合体通常会通过某种方式共享他们的注释，通常是通过一些公共数据库。`BioMart`是一种常见的数据结构和 API，通过它可以获取注释数据。在这个配方中，我们将看看如何以编程方式访问这些数据库，以获取我们感兴趣的基因的注释信息。

# 准备工作

对于这个配方，我们需要名为`biomaRt`的`Bioconductor`包以及一个可用的互联网连接。我们还需要知道要连接的`BioMart`服务器 —— 全球约有 40 个这样的服务器，提供各种信息。最常访问的是`Ensembl`数据库，这些是这些包的默认设置。您可以在这里查看所有`BioMart`的列表：[`www.biomart.org/notice.html`](http://www.biomart.org/notice.html)。我们将开发的代码将适用于这些`BioMart`中的任何一个，只需稍微修改表名和 URL。

# 如何实现...

可以通过以下步骤从`BioMart`检索基因和基因组注释：

1.  列出所选示例数据库`gramene`中的`mart`列表：

```py
library(biomaRt)
listMarts(host = "ensembl.gramene.org")
```

1.  创建到所选`mart`的连接：

```py
gramene_connection <- useMart(biomart = "ENSEMBL_MART_PLANT", host = "ensembl.gramene.org")
```

1.  列出该`mart`中的数据集：

```py
data_sets <-  listDatasets(gramene_connection)
head(data_sets)

data_set_connection <- useMart("atrichopoda_eg_gene", biomart = "ENSEMBL_MART_PLANT", host = "ensembl.gramene.org")
```

1.  列出我们实际可以检索的数据类型：

```py
attributes <- listAttributes(data_set_connection)
head(attributes)
```

1.  获取所有染色体名称的向量：

```py
chrom_names <- getBM(attributes = c("chromosome_name"), mart = data_set_connection )
head(chrom_names)
```

1.  创建一些用于查询数据的过滤器：

```py
filters <- listFilters(data_set_connection)
head(filters)
```

1.  获取第一个染色体上的基因 ID：

```py
first_chr <- chrom_names$chromosome_name[1]
genes <- getBM(attributes = c("ensembl_gene_id", "description"), filters = c("chromosome_name"), values = c(first_chr), mart = data_set_connection )head(genes)
head(genes)
```

# 工作原理...

该配方围绕对数据库进行一系列不同的查找操作，每次获取一些更多信息来处理。

在*第一步*中，我们使用`listMarts()`函数获取指定主机 URL 上所有可用的`BioMart`列表。在需要连接到不同服务器时，请根据需要更改 URL。我们得到一个可用`mart`的数据框，并使用该信息。

在*第二步*中，我们使用`useMart()`函数创建一个名为`gramene_connection`的连接对象，传入服务器 URL 和*第一步*中的具体`BioMart`。

在*步骤* *3*中，我们将`gramene_connection`传递给`listDatasets()`函数，以检索该`biomart`中的数据集。选择其中一个数据集（`atrichopda_eg_gene`）后，我们可以运行`useMart()`函数来创建一个到该`biomart`中数据集的连接，并将对象命名为`data_set_connection`。

在*步骤 4*中，我们几乎完成了确定可以使用哪些数据集的工作。在这里，我们使用在`listAttributes()`函数中创建的`data_set_connection`，获取我们可以从该数据集中检索的各种信息类型的列表。

在*步骤 5*中，我们最终通过主要函数`getBM()`获取一些实际信息。我们将`attributes`参数设置为我们希望返回的数据的名称；在这里，我们获取`chromosome_name`的所有值，并将它们保存到一个向量`chrom_names`中。

在*步骤 6*中，我们设置过滤器——即接收哪些值的限制。我们首先询问`data_set_connection`对象，使用`listFilters()`函数查看我们可以使用哪些过滤器。从返回的`filters`对象中可以看到，我们可以在`chromosome_name`上进行过滤，所以我们将使用这个。

在*步骤 7*中，我们设置一个完整的查询。在这里，我们打算获取第一个染色体上的所有基因。请注意，我们已经在*步骤 5*中获得了一个染色体列表，因此我们将使用`chrom_names`对象中的第一个元素作为过滤器，并将其保存在`first_chr`中。为了执行查询，我们使用`getBM()`函数，并指定`ensembl_gene_id`和`description`属性。我们将`filter`参数设置为我们希望过滤的数据类型，并将`values`参数设置为我们希望保留的过滤器值。我们还将`data_set_connection`对象作为要使用的 BioMart 传递。最终生成的`genes`对象包含了第一个染色体上的`ensembl_gene_id`和描述信息，如下所示：

```py
## ensembl_gene_id           description
## 1 AMTR_s00001p00009420    hypothetical protein 
## 2 AMTR_s00001p00015790    hypothetical protein 
## 3 AMTR_s00001p00016330    hypothetical protein 
## 4 AMTR_s00001p00017690    hypothetical protein 
## 5 AMTR_s00001p00018090    hypothetical protein 
## 6 AMTR_s00001p00019800    hypothetical protein
```

# 获取和处理 SNP

SNP 和其他多态性是重要的基因组特征，我们通常希望在特定基因组区域内检索已知的 SNP。在这里，我们将介绍如何在两个不同的 BioMart 中执行此操作，这些 BioMart 存储了不同类型的 SNP 数据。在第一部分中，我们将再次使用 Gramene 来查看如何获取植物 SNP。在第二部分中，我们将了解如何在主要的 Ensembl 数据库中查找人类 SNP 的信息。

# 准备工作

如前所述，我们只需要`biomaRt`包，它来自`Bioconductor`，并且需要一个正常工作的互联网连接。

# 如何操作……

获取和处理 SNP 可以通过以下步骤完成：

1.  从 Gramene 获取数据集、属性和过滤器列表：

```py
library(biomaRt)
listMarts(host = "ensembl.gramene.org")
gramene_connection <- useMart(biomart = "ENSEMBL_MART_PLANT_SNP", host = "ensembl.gramene.org")
data_sets <- listDatasets(gramene_connection)
head(data_sets)
data_set_connection <- useMart("athaliana_eg_snp", biomart = "ENSEMBL_MART_PLANT_SNP", host = "ensembl.gramene.org")

listAttributes(data_set_connection)
listFilters(data_set_connection)
```

1.  查询实际的 SNP 信息：

```py
snps <- getBM(attributes = c("refsnp_id", "chr_name", "chrom_start", "chrom_end"), filters = c("chromosomal_region"), values = c("1:200:200000:1"), mart = data_set_connection )
head(snps)

```

# 它是如何工作的……

*步骤 1*将与之前的食谱中的*步骤 1*到*6*类似，我们将建立初始连接并让它列出我们可以在此 BioMart 中使用的数据集、属性和过滤器；这是相同的模式，每次使用 BioMart 时都会重复（直到我们能熟记它）。

在*步骤 2*中，我们利用收集到的信息提取目标区域的 SNP。同样，我们使用`getBM()`函数并设置`chromosomal_region`过滤器。这允许我们指定一个描述基因组中特定位点的值。`value`参数采用`Chromosome:Start:Stop:Strand`格式的字符串；具体而言，`1:200:20000:1`，这将返回染色体 1 上从第 200 到 20000 个碱基的所有 SNP，且位于正链上（注意，正链 DNA 的标识符为`1`，负链 DNA 的标识符为`-1`）。

# 还有更多...

从 Ensembl 中查找人类 SNP 的步骤基本相同。唯一的区别是，由于 Ensembl 是默认服务器，我们可以在`useMart()`函数中省略服务器信息。对于人类的类似查询会像这样：

```py
data_set_connection <- useMart("hsapiens_snp", biomart = "ENSEMBL_MART_SNP")
human_snps <- getBM(attributes = c("refsnp_id", "allele", "minor_allele", "minor_allele_freq"), filters = c("chromosomal_region"), value = c("1:200:20000:1"), mart = data_set_connection)

```

# 另见

如果你拥有`dbSNP refsnp ID`编号，可以通过`rnsps`包和`ncbi_snp_query()`函数直接查询这些 ID。只需将有效的`refsnp` ID 向量传递给该函数。

# 获取基因本体论信息

**基因本体论**（**GO**）是一个非常有用的受限词汇，包含用于基因和基因产物的注释术语，描述了注释实体的生物过程、分子功能或细胞成分。因此，这些术语在基因集富集分析和其他功能-组学方法中非常有用。在本节中，我们将看看如何准备一个基因 ID 列表，并为它们获取 GO ID 和描述信息。

# 准备工作

由于我们仍在使用`biomaRt`包，所以只需要该包以及一个有效的互联网连接。

# 如何操作...

获取基因本体论信息的步骤如下：

1.  连接到 Ensembl BioMart 并找到适当的属性和过滤器：

```py
library(biomaRt)

ensembl_connection <- useMart(biomart = "ENSEMBL_MART_ENSEMBL")
 listDatasets(ensembl_connection)

data_set_connection <- useMart("hsapiens_gene_ensembl", biomart = "ENSEMBL_MART_ENSEMBL")

att <- listAttributes(data_set_connection)
fil <- listFilters(data_set_connection)

```

1.  获取基因列表，并使用它们的 ID 获取 GO 注释：

```py
genes <- getBM(attributes = c("ensembl_gene_id"), filters = c("chromosomal_region"), value = c("1:200:2000000:1"), mart = data_set_connection)

go_ids <- getBM(attributes = c("go_id", "goslim_goa_description"), filters = c("ensembl_gene_id"), values = genes$ensembl_gene_id, mart = data_set_connection )
```

# 工作原理...

如同前两个步骤，*步骤 1* 包括找到适合的 biomart、数据集、属性和过滤器的值。

在*步骤 2*中，我们使用`getBM()`函数获取特定染色体区域中的`ensembl_gene_id`属性，将结果保存在`genes`对象中。然后我们再次使用该函数，使用`ensembl_gene_id`作为过滤器，`go_id`和`goslim_goa_description`来获取仅选定基因的 GO 注释。

# 查找 SRA/ENA 中的实验和读取数据

**短序列数据归档**（**SRA**）和**欧洲核苷酸库**（**ENA**）是记录原始高通量 DNA 序列数据的数据库。每个数据库都是相同高通量序列数据集的镜像版本，这些数据集是来自世界各地各个生物学领域的科学家提交的。通过这些数据库免费获取高通量序列数据意味着我们可以设想并执行对现有数据集的新分析。通过对数据库进行搜索，我们可以识别出我们可能想要处理的序列数据。在本配方中，我们将研究如何使用`SRAdb`包查询 SRA/ENA 上的数据集，并以编程方式检索选定数据集的数据。

# 准备工作

这个配方的两个关键元素是来自`Bioconductor`的`SRAdb`包和一个有效的互联网连接。

# 如何操作...

查找来自 SRA/ENA 的实验和数据读取可以通过以下步骤完成：

1.  下载 SQL 数据库并建立连接：

```py
library(SRAdb)
sqlfile <- file.path(system.file('extdata', package='SRAdb'), 'SRAmetadb_demo.sqlite')
sra_con <- dbConnect(SQLite(),sqlfile)
```

1.  获取研究信息：

```py
dbGetQuery(sra_con, "select study_accession, study_description from study where study_description like '%coli%' ")
```

1.  获取关于该研究包含内容的信息：

```py
sraConvert( c('ERP000350'), sra_con = sra_con )
```

1.  获取可用文件的列表：

```py
listSRAfile( c("ERR019652","ERR019653"), sra_con, fileType = 'sra' )
```

1.  下载序列文件：

```py
getSRAfile( c("ERR019652","ERR019653"), sra_con, fileType = 'fastq', destDir = file.path(getwd(), "datasets", "ch8") )
```

# 它是如何工作的...

加载库后，第一步是设置一个本地的 SQL 文件，名为`sqlfile`。该文件包含了关于 SRA 上研究的所有信息。在我们的示例中，我们使用的是包内的一个小版本（因此，我们通过`system.file()`函数提取它）；真实的文件大小超过 50GB，因此我们暂时不会使用它，但可以通过以下替换代码获取：`sqlfile <- getSRAdbfile()`。一旦我们拥有一个`sqlfile`对象，就可以使用`dbConnect()`函数创建与数据库的连接。我们将连接保存在名为`sra_con`的对象中，以便重用。

接下来，我们使用`dbGetQuery()`函数对`sqlfile`数据库执行查询。该函数的第一个参数是数据库文件，第二个参数是一个完整的 SQL 格式的查询。写出的查询非常直观；我们希望返回包含`coli`一词的描述时的`study_accession`和`study_description`。更复杂的查询是可能的——如果你准备好用 SQL 编写它们。关于这方面的教程超出了本配方的范围，但有许多书籍专门介绍这个主题；你可以尝试阅读 Upom Malik、Matt Goldwasser 和 Benjamin Johnston 合著的《SQL for Data Analytics》，由 Packt Publishing 出版：[`www.packtpub.com/big-data-and-business-intelligence/sql-data-analysis`](https://www.packtpub.com/big-data-and-business-intelligence/sql-data-analysis)。该查询返回一个类似于以下内容的 dataframe 对象：

```py
## study_accession    study_description
## ERP000350    Transcriptome sequencing of E.coli K12 in LB media in early exponential phase and transition to stationary phase

```

*步骤 3* 使用我们提取的访问号，利用`sraConvert()`函数获取与该研究相关的所有提交、样本、实验和运行信息。这将返回如下表格，我们可以看到该研究的运行 ID，展示了包含序列的实际文件：

```py
##    study submission    sample experiment       run
## 1 ERP000350 ERA014184 ERS016116 ERX007970 ERR019652 
## 2 ERP000350 ERA014184 ERS016115 ERX007969 ERR019653 
```

在*步骤 4*中，我们使用 `listSRAfile()` 函数获取运行中特定序列的实际 FTP 地址。这将提供 SRA 格式文件的地址，这是一个压缩且方便的格式，如果你希望了解的话：

```py
     run     study    sample experiment    ftp
## 1 ERR019652 ERP000350 ERS016116 ERX007970 ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByRun/sra/ERR/ERR019/ERR019652/ERR019652.sra 
## 2 ERR019653 ERP000350 ERS016115 ERX007969 ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByRun/sra/ERR/ERR019/ERR019653/ERR019653.sra
```

但是在*步骤 5*中，我们使用 `getSRAfile()` 函数，并将 `fileType` 参数设置为 `fastq`，以便以标准的 `fastq` 格式获取数据。文件将下载到 `destDir` 参数指定的文件夹中。

# 还有更多...

不要忘记定期刷新本地的 SQL 数据库，并使用以下代码获取完整版本：`sqlfile <- getSRAdbfile()`。

# 对高通量序列读取进行质量控制和过滤

当我们有一组新的序列读取需要处理时，无论是来自新实验还是数据库，我们都需要执行质量控制步骤，去除任何序列接头、去除质量差的读取，或者根据需要修剪质量差的序列。在这个配方中，我们将介绍如何在 R 中使用 `Bioconductor ShortRead` 包来实现这一点。

# 准备工作

你需要安装 `ShortRead` 包，并且需要运行本章中 *查找 SRA/ENA 中的实验和读取* 这一配方的代码。在该配方的最后一步，会创建两个文件，我们将使用其中一个。运行该代码后，文件应该位于本书的 `datasets/ch8/ERRR019652.fastq.gz` 目录中。

# 如何执行...

对高通量序列读取进行质量控制和过滤可以通过以下步骤完成：

1.  加载库并连接到文件：

```py
library(ShortRead)
fastq_file <- readFastq(file.path(getwd(), "datasets", "ch8", "ERR019652.fastq.gz") )
```

1.  过滤掉质量低于 20 的任何核苷酸的读取：

```py
qualities <- rowSums(as(quality(fastq_file), "matrix") <= 20) 
fastq_file <- fastq_file[qualities == 0] 
```

1.  修剪读取的右侧：

```py
cut_off_txt <- rawToChar(as.raw(40))
trimmed <- trimTails(fastq_file, k =2, a= cut_off_txt)
```

1.  设置自定义过滤器以移除 *N* 和同源重复序列：

```py
custom_filter_1 <- nFilter(threshold=0)
custom_filter_2 <- polynFilter(threshold = 10, nuc = c("A", "T", "C", "G"))
custom_filter <- compose(custom_filter_1, custom_filter_2)
passing_rows <- custom_filter(trimmed)
trimmed <- trimmed[passing_rows]
```

1.  写出保留的读取：

```py
writeFastq(trimmed, file = file.path(getwd(), "datasets", "ch8", "ERR019652.trimmed.fastq.gzip"), compress = TRUE)
```

# 它是如何工作的...

第一步将读取加载到 `ShortReadQ` 对象中，该对象表示 DNA 读取及其相关的质量评分；这个特殊的对象使我们能够在一次操作中处理序列和质量。

第二步让我们找到所有质量分数都高于 20 的读取。这里的代码有点特殊，所以请花时间理解它。首先，我们在 `fastq_file` 上使用 `quality()` 函数提取质量值，然后将其传递给 `as()` 函数，要求返回一个矩阵。接着，我们对该矩阵使用 `rowSums()` 计算每行的总和，最终通过比较得到一个逻辑向量 `qualities`，它指示哪些 `rowSums()` 的值小于 20。在下一行中，我们使用 `qualities` 向量来对 `fastq_file` 进行子集筛选，去除低质量的读取。

在*步骤 3*中，我们修剪读取的右侧（以纠正读取质量低于阈值的地方）。这里的主要功能是`trimTails()`，它接受两个参数：`k`，开始修剪所需的失败字母数，以及`a`，开始修剪的字母。这当然意味着我们所认为的 Phred 数值质量分数（如*步骤 2*中，我们仅使用了 20）需要根据质量分数的文本编码转换为其 ASCII 等价物。这就是第一行发生的事情；数字 40 通过`as.raw()`转换为原始字节，然后通过`rawToChar()`转换为字符。生成的文本可以通过将其存储在`cut_off_txt`变量中来使用。

*步骤 4*应用了一些自定义过滤器。第一行，`custom_filter_1`，为包含名为*N*的碱基的序列创建过滤器，阈值参数允许序列包含零个*N*。第二行，`custom_filter_2`，为长度等于或大于阈值的同质聚合物读取创建过滤器。`nuc`参数指定要考虑的核苷酸。一旦指定了过滤器，我们必须使用`compose()`函数将它们合并成一个单一的过滤器，该函数返回一个我们称之为`custom_filter()`的过滤器函数，然后对修剪后的对象进行调用。它返回一个`SRFFilterResult`对象，可以用来对读取进行子集化。

最后，在*步骤 5*中，我们使用`writeFastQ()`函数将保留的读取写入文件。

# 使用外部程序完成读取到参考的比对

高通量读取的比对是本书中许多方法的一个重要前提，包括 RNAseq 和 SNP/INDEL 调用。在第一章，*执行定量 RNAseq*，以及第二章，*使用 HTS 数据查找遗传变异*中我们对其进行了深入讨论，但我们没有涉及如何实际执行比对。我们通常不会在 R 中执行此操作；进行这些比对所需的程序是强大的，并且作为独立进程从命令行运行。但 R 可以控制这些外部进程，因此我们将探讨如何运行外部进程，以便你可以从 R 包装脚本中控制它们，最终使你能够开发端到端的分析管道。

# 准备中...

本教程仅使用基础的 R 语言，因此你无需安装任何额外的包。你需要准备参考基因组 FASTA 文件`datasets/ch8/ecoli_genome.fa`，以及我们在*寻找 SRA/ENA 实验和读取数据*教程中创建的`datasets/ch8/ERR019653.fastq,gz`文件。本教程还需要系统中安装 BWA 和`samtools`的可执行文件。相关软件的网页可以在[`samtools.sourceforge.net/`](http://samtools.sourceforge.net/)和[`bio-bwa.sourceforge.net/`](http://bio-bwa.sourceforge.net/)找到。如果你已经安装了`conda`，可以通过`conda install -c bioconda bwa`和`conda install -c bioconda samtools`来安装它们。

# 如何操作……

使用以下步骤完成读取到参考的比对，借助外部程序：

1.  设置文件和可执行文件路径：

```py
bwa <- "/Users/macleand/miniconda2/bin/bwa"
samtools <- "/Users/macleand/miniconda2/bin/samtools"
reference <- file.path(getwd(), "datasets", "ch8", "ecoli_genome.fa")
```

1.  准备`index`命令并执行：

```py
command <- paste(bwa, "index", reference)
system(command, wait = TRUE)
```

1.  准备`alignment`命令并执行：

```py
reads <- file.path(getwd(), "datasets", "ch8", "ERR019653.fastq.gz")
output <- file.path(getwd(), "datasets", "ch8", "aln.bam")
command <- paste(bwa, "mem", reference, reads, "|", samtools, "view -S -b >", output)
system(command, wait = TRUE)
```

# 它是如何工作的……

第一步非常简单：我们仅仅创建了几个变量来保存程序和文件所在目录的路径。`bwa`和`samtools`分别保存了这些程序在系统中的路径。请注意，你的系统上的路径可能与此不同。在 Linux 和 macOS 系统中，你可以通过`which`命令在终端中查找路径；在 Windows 机器上，你可以使用`where`命令或等效命令。

在*步骤 2*中，我们概述了运行系统命令的基本模式。首先，通过`paste()`函数，我们将命令作为字符串创建，并保存在一个名为`command`的变量中。在这里，我们正在准备一个命令行，在执行 BWA 进行读取比对之前创建所需的索引。然后，我们将该命令作为第一个参数传递给`system()`函数，后者会实际执行该命令。命令作为一个全新的进程在后台启动，默认情况下，一旦进程开始，控制权会立即返回给 R 脚本。如果你打算在后台进程输出后立即在 R 中继续工作，你需要将`system()`函数中的`wait`参数设置为`TRUE`，这样 R 进程将在后台进程完成后才继续。

在*步骤 3*中，我们扩展了模式，创建了读取和输出变量，并组合了一个更长的命令行，展示了如何构建一个有效的命令行。然后我们重复执行`system`命令。此过程将生成一个最终的 BAM 文件，位于`datasets/ch8/aln.bam`。

# 可视化读取到参考比对的质量控制

一旦完成读取的比对，通常建议检查比对的质量，确保读取的模式和期望的插入距离等没有异常。这在草拟参考基因组中尤其有用，因为高通量读取的异常比对可能会揭示参考基因组的拼接错误或其他结构重排。在本教程中，我们将使用一个名为`ramwas`的包，它有一些易于访问的图形，可以帮助我们评估比对的质量。

# 正在准备中...

对于这个配方，我们需要准备好的 `bam_list.txt` 和 `sample_list.txt` 信息文件，位于本书的 `datasets/ch8` 目录下。我们还需要来自同一位置的小文件 `ERR019652.small.bam` 和 `ERR019653.small.bam`。

# 如何执行...

可通过以下步骤可视化读取到参考基因组的比对质量控制：

1.  设置运行的参数：

```py
library(ramwas)
param <- ramwasParameters( dirbam = ".", filebamlist = "bam_list.txt", 
                            dirproject = file.path(getwd(), "datasets", "ch8"), 
                            filebam2sample = "sample_list.txt")
```

1.  执行质量控制：

```py
ramwas1scanBams(param)
qc <- readRDS(file.path(getwd(), "datasets", "ch8", "rds_qc", "ERR019652.small.qc.rds")$qc
```

1.  查看图表：

```py
plot(qc$hist.score1)
plot(qc$bf.hist.score1)
plot(qc$hist.length.matched)
plot(qc$bf.hist.length.matched)
```

# 它是如何工作的...

*步骤 1* 使用 `ramwasParameters()` 函数设置一个包含参数的对象。我们只需提供信息文件（`bam_list.txt` 和 `sample_list.txt`），分别指定要使用的 BAM 文件及其包含的样本位置。`dirproject` 参数指定结果应该写入系统的路径。请注意，这些结果会被写入磁盘，而不是直接返回内存。

*步骤 2* 使用参数通过 `ramwas1scanBams()` 函数运行质量控制（QC）。结果被写入磁盘，因此我们使用基本的 R `readRDS()` 函数加载返回的 RDS 文件。`qc` 对象包含许多成员，代表了不同的比对质量控制方面。

*步骤 3* 使用通用的 `plot` 函数来创建一些 `qc` 对象中的质量控制统计图。
