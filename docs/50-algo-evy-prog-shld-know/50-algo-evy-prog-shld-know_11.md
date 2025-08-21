

# 第九章：自然语言处理的算法

> 语言是思维最重要的工具。
> 
> —马文·明斯基

本章介绍了**自然语言处理**（**NLP**）的算法。首先介绍了 NLP 的基础知识。然后介绍了为 NLP 任务准备数据。接下来，解释了文本数据向量化的概念。然后，我们讨论了词嵌入。最后，展示了一个详细的用例。

本章由以下几部分组成：

+   介绍 NLP

+   **基于词袋模型**（**BoW-based**）的 NLP

+   词嵌入介绍

+   案例研究：餐厅评论情感分析

到本章结束时，你将理解用于自然语言处理（NLP）的基本技术。你还应该理解 NLP 如何用于解决一些有趣的现实世界问题。

让我们从基本概念开始。

# 介绍 NLP

NLP 是机器学习算法的一个分支，处理计算机与人类语言之间的互动。它涉及分析、处理和理解人类语言，以使计算机能够理解并回应人类的沟通。NLP 是一个综合性的学科，涉及使用计算机语言学算法以及人机交互技术和方法来处理复杂的非结构化数据。

NLP 通过处理人类语言并将其分解为基本部分，如单词、短语和句子，来工作。目标是使计算机理解文本的含义并做出适当回应。NLP 算法利用各种技术，如统计模型、机器学习和深度学习，来分析和处理大量的自然语言数据。对于复杂问题，我们可能需要使用多种技术的组合来找到有效的解决方案。

NLP 的一个重大挑战是处理人类语言的复杂性和歧义性。语言种类繁多，具有复杂的语法结构和习惯用语。此外，单词和短语的含义可能根据使用的上下文而有所不同。NLP 算法必须能够处理这些复杂性，以实现有效的语言处理。

让我们从一些在讨论 NLP 时使用的术语开始。

# 理解 NLP 术语

NLP 是一个广泛的研究领域。在这一部分，我们将探讨一些与 NLP 相关的基本术语：

+   **语料库**：语料库是一个大型且结构化的文本或语音数据集合，作为 NLP 算法的资源。它可以由各种类型的文本数据组成，如书面文本、口语语言、转录对话和社交媒体帖子。语料库通过有意地从各种在线和离线来源收集和组织数据来创建，包括互联网。虽然互联网是获取数据的丰富来源，但决定将哪些数据包含在语料库中，需要根据特定研究或分析的目标进行有目的的选择和对齐。

    语料库（corpora）是语料（corpus）的复数形式，可以进行注释，意味着它们可能包含关于文本的额外细节，例如词性标签和命名实体。这些注释语料库提供了特定的信息，能够增强 NLP 算法的训练和评估，使它们在该领域成为极具价值的资源。

+   **标准化**：这个过程涉及将文本转换为标准形式，例如将所有字符转换为小写字母或去除标点符号，使其更容易进行分析。

+   **分词**：分词将文本拆分成更小的部分，称为词元，通常是单词或子词，从而实现更结构化的分析。

+   **命名实体识别**（**NER**）：NER 用于识别和分类文本中的命名实体，例如人名、地点、组织等。

+   **停用词**：这些是常用词，例如 *and*、*the* 和 *is*，它们在文本处理过程中通常会被过滤掉，因为它们可能不会提供显著的意义。

+   **词干提取和词形还原**：词干提取是将单词还原为其词根形式，而词形还原是将单词转换为其基本或词典形式。这两种技术有助于分析单词的核心含义。

接下来，让我们研究 NLP 中使用的不同文本预处理技术：

+   **词嵌入**：这是一种将单词转换为数值形式的方法，其中每个单词都表示为一个向量，位于一个可能具有多个维度的空间中。在这个背景下，“高维向量”指的是一个数字数组，其中维度数量或单独的成分是相当大的——通常在数百甚至数千维之间。使用高维向量的思想是为了捕捉单词之间复杂的关系，使得具有相似含义的单词在这个多维空间中更接近。向量维度越多，它能够捕捉的关系就越细致。因此，在词嵌入中，语义相关的单词会在这个高维空间中彼此更接近，从而使得算法能够更容易地理解和处理语言，反映出人类的理解方式。

+   **语言建模**：语言建模是开发统计模型的过程，这些模型可以根据给定文本语料库中发现的模式和结构，预测或生成单词或字符的序列。

+   **机器翻译**：使用自然语言处理（NLP）技术和模型自动将文本从一种语言翻译成另一种语言的过程。

+   **情感分析**：通过分析文本中使用的单词、短语及其上下文来确定一段文本中表达的态度或情感的过程。

## NLP 中的文本预处理

文本预处理是 NLP 中的一个关键阶段，在这一阶段，原始文本数据会经过转换，变得适合机器学习算法。这个转化过程涉及将无序且通常杂乱无章的文本转化为所谓的“结构化格式”。结构化格式意味着数据被组织成更加系统和可预测的模式，通常涉及分词、词干提取和删除不需要的字符等技术。这些步骤有助于清理文本，减少无关信息或“噪音”，并以一种更便于机器学习模型理解的方式整理数据。通过这种方法，原始文本中的不一致性和不规则性得以转化，形成一种能够提高后续 NLP 任务准确性、性能和效率的形式。在本节中，我们将探索用于文本预处理的各种技术，以实现这种结构化格式。

### 分词

提醒一下，分词是将文本分解成更小单位（即令牌）的关键过程。这些令牌可以是单个词语，甚至是子词。在 NLP 中，分词通常被视为准备文本数据进行进一步分析的第一步。分词之所以如此重要，源于语言本身的特性，理解和处理文本需要将其分解为可管理的部分。通过将连续的文本流转化为单独的令牌，我们创造了一种结构化格式，类似于人类自然阅读和理解语言的方式。这种结构化使得机器学习模型能够以清晰且系统化的方式分析文本，从而识别数据中的模式和关系。随着我们深入研究 NLP 技术，这种令牌化的格式成为许多其他预处理和分析步骤的基础。

```py
is tokenizing the given text using the Natural Language Toolkit (nltk) library in Python. The nltk is a widely used library in Python, specifically designed for working with human language data. It provides easy-to-use interfaces and tools for tasks such as classification, tokenization, stemming, tagging, parsing, and more, making it a valuable asset for NLP. For those who wish to leverage these capabilities in their Python projects, the nltk library can be downloaded and installed directly from the Python Package Index (PyPI) by using the command pip install nltk. By incorporating the nltk library into your code, you can access a rich set of functions and resources that streamline the development and execution of various NLP tasks, making it a popular choice among researchers, educators, and developers in the field of computational linguistics. Let us start by importing relevant functions and using them:
```

```py
from nltk.tokenize import word_tokenize
corpus = 'This is a book about algorithms.'
tokens = word_tokenize(corpus)
print(tokens) 
```

输出将是如下所示的列表：

```py
['This', 'is', 'a', 'book', 'about', 'algorithms', '.'] 
```

在这个示例中，每个令牌都是一个单词。最终令牌的粒度将根据目标而有所不同——例如，每个令牌可以是一个单词、一句话或一段话。

要基于句子对文本进行分词，可以使用`nltk.tokenize`模块中的`sent_tokenize`函数：

```py
from nltk.tokenize import sent_tokenize
corpus = 'This is a book about algorithms. It covers various topics in depth.' 
```

在这个例子中，`corpus`变量包含了两个句子。`sent_tokenize`函数将语料库作为输入，并返回一个句子的列表。当你运行修改后的代码时，将得到以下输出：

```py
sentences = sent_tokenize(corpus)
print(sentences) 
```

```py
['This is a book about algorithms.', 'It covers various topics in depth.'] 
```

有时我们可能需要将较大的文本拆分为按段落划分的块。`nltk`可以帮助完成这项任务。这项功能在文档摘要等应用中尤其有用，因为在这些应用中，理解段落级别的结构可能至关重要。将文本按段落进行分词看似简单，但根据文本的结构和格式，可能会变得复杂。一个简单的方法是通过两个换行符来拆分文本，这通常用于纯文本文档中的段落分隔。

这里是一个基本示例：

```py
def tokenize_paragraphs(text):
    # Split by two newline characters
    paragraphs = text.split('\n\n') 
    return [p.strip() for p in paragraphs if p] 
```

接下来，让我们看看如何清理数据。

### 清理数据

清理数据是 NLP 中的一个关键步骤，因为原始文本数据通常包含噪音和无关信息，这些信息可能会妨碍 NLP 模型的性能。清理数据的目标是对文本数据进行预处理，去除噪音和无关信息，并将其转换为适合使用 NLP 技术分析的格式。请注意，数据清理是在数据被分词之后进行的。原因在于，清理可能涉及到依赖于分词揭示的结构的操作。例如，删除特定单词或修改单词形式可能在文本被分词为独立的词汇后更为准确。

让我们研究一些用于清理数据并为机器学习任务做准备的技术：

#### 大小写转换

大小写转换是自然语言处理（NLP）中的一种技术，它将文本从一种大小写格式转换为另一种格式，例如从大写转换为小写，或者从标题式大小写转换为大写。

例如，标题式大小写的“Natural Language Processing”可以转换为小写，即“natural language processing”。

这一简单而有效的步骤有助于标准化文本，从而简化其在各种 NLP 算法中的处理。通过确保文本处于统一的大小写格式，有助于消除由于大小写变化可能产生的不一致性。

#### 标点符号移除

在 NLP 中，标点符号移除是指在分析之前，从原始文本数据中删除标点符号的过程。标点符号是如句号（`.`）、逗号（`,`）、问号（`?`）和感叹号（`!`）等符号，它们在书面语言中用于表示停顿、强调或语调。虽然它们在书面语言中至关重要，但它们会为原始文本数据增加噪音和复杂性，进而妨碍 NLP 模型的性能。

合理的疑虑是，删除标点符号可能会影响句子的意义。请考虑以下示例：

`"她是只猫。"`

`"她是只猫??"`

没有标点符号时，两行文本都变成了“她是只猫”，可能失去了问号所传达的独特强调。

然而，值得注意的是，在许多 NLP 任务中，如主题分类或情感分析，标点符号可能不会显著影响整体理解。此外，模型可以依赖于文本结构、内容或上下文中的其他线索来推导含义。在标点符号细微差别至关重要的情况下，可能需要使用专门的模型和预处理技术来保留所需的信息。

#### 处理数字在 NLP 中的应用

文本数据中的数字可能给 NLP 带来挑战。下面是两种处理文本中数字的主要策略，既考虑了传统的去除方法，也考虑了标准化的替代选项。

在某些自然语言处理（NLP）任务中，数字可能被视为噪声，特别是当关注点集中在像词频或情感分析等方面时。这就是为什么一些分析师可能选择去除数字的原因：

+   **缺乏相关性**：在某些文本分析情境中，数字字符可能不携带重要的含义。

+   **扭曲词频统计**：数字可能会扭曲词频统计，尤其是在像主题建模这样的模型中。

+   **减少复杂性**：去除数字可以简化文本数据，可能提升 NLP 模型的性能。

然而，一种替代方法是将所有数字转换为标准表示，而不是将其丢弃。这种方法承认数字可以携带重要信息，并确保其在一致格式中保留其值。在数字数据对文本含义起着至关重要作用的语境中，这种方法特别有用。

决定是否去除或保留数字需要理解所解决的问题。一个算法可能需要定制，以根据文本的上下文和特定的 NLP 任务来区分数字是否重要。分析数字在文本领域中的作用以及分析的目标，可以引导这一决策过程。

在 NLP 中处理数字并不是一成不变的方法。是否去除、标准化或仔细分析数字，取决于任务的独特要求。理解这些选项及其影响，有助于做出符合文本分析目标的明智决策。

#### 去除空格

在 NLP 中，空格去除指的是去除不必要的空格字符，如多个空格和制表符字符。在文本数据的语境中，空格不仅是单词之间的空白，还包括其他“看不见”的字符，这些字符在文本中创建了间距。在 NLP 中，空格去除指的是去除这些不必要的空格字符。去除不必要的空格可以减少文本数据的大小，并使其更易于处理和分析。

下面是一个简单的例子来说明空格去除：

+   输入文本：`"The quick brown fox \tjumps over the lazy dog."`

+   处理过的文本：`"The quick brown fox jumps over the lazy dog."`

在上述示例中，去除了额外的空格和一个制表符字符（由 `\t` 表示），从而创建了更干净且更标准化的文本字符串。

#### 停用词去除

停用词去除是指从文本语料库中删除常见词汇，称为停用词。停用词是在语言中频繁出现，但不具有重要意义或不有助于理解文本的词汇。英语中的停用词包括 *the,* 和*, is, in* 和 *for*。停用词去除有助于减少数据的维度，并提高算法的效率。通过去除那些对分析没有重要贡献的词汇，可以将计算资源集中在真正重要的词汇上，从而提高各种自然语言处理算法的效率。

请注意，停用词去除不仅仅是减少文本大小；它是为了专注于分析中真正重要的词汇。虽然停用词在语言结构中扮演着重要角色，但在自然语言处理中的去除，可以提升分析的效率和重点，特别是在情感分析等任务中，主要关心的是理解潜在的情感或观点。

#### 词干提取和词形还原

在文本数据中，大多数单词可能以略微不同的形式出现。将每个单词简化为它的原型或词干，这一过程称为**词干提取**。它用于根据单词的相似含义将单词分组，以减少需要分析的单词总数。本质上，词干提取减少了问题的整体条件性。英语中最常用的词干提取算法是 Porter 算法。

例如，让我们看一些例子：

+   示例 1：`{use, used, using, uses} => use`

+   示例 2：`{easily, easier, easiest} => easi`

需要注意的是，词干提取有时会导致拼写错误，如示例 2 中生成的 `easi`。

词干提取是一种简单且快速的处理过程，但它可能并不总是产生正确的结果。在需要正确拼写的情况下，词形还原是一种更合适的方法。词形还原考虑上下文并将单词还原为其基本形式。单词的基本形式，也称为词根，是其最简单且最具意义的版本。它代表了单词在字典中的形式，去除了任何词尾变化，形成一个正确的英语单词，从而产生更准确、更有意义的词根。

引导算法识别相似性是一个精确且深思熟虑的任务。与人类不同，算法需要明确的规则和标准来建立连接，这些连接对我们来说可能看起来是显而易见的。理解这一差异并知道如何提供必要的引导，是开发和调整算法在各种应用中的重要技能。

# 使用 Python 清理数据

让我们看一下如何使用 Python 清理文本。

首先，我们需要导入必要的库：

```py
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Make sure to download the NLTK resources
nltk.download('punkt')
nltk.download('stopwords') 
```

接下来，这是执行文本清理的主函数：

```py
def clean_text(text):
    """
    Cleans input text by converting case, removing punctuation, numbers,
    white spaces, stop words and stemming
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove white spaces
    text = text.strip()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_text)

    # Stemming
    ps = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_text = [ps.stem(word) for word in tokens]
    text = ' '.join(stemmed_text)

    return text 
```

让我们测试一下`clean_text()`函数：

```py
corpus="7- Today, Ottawa is becoming cold again "
clean_text(corpus) 
```

结果将是：

```py
today ottawa becom cold 
```

注意输出中的单词`becom`。由于我们使用了词干提取技术，输出中的并非所有单词都是正确的英语单词。

所有前面的处理步骤通常是必需的；实际的处理步骤取决于我们要解决的问题。它们会因使用场景而异——例如，如果文本中的数字表示一些在我们尝试解决的问题背景下可能有意义的内容，那么在标准化阶段我们可能不需要去除这些数字。

一旦数据被清理，我们需要将结果存储在一个专门为此目的设计的数据结构中。这个数据结构被称为**术语文档矩阵**（**TDM**），接下来会详细解释。

# 理解术语文档矩阵

TDM（术语文档矩阵）是自然语言处理（NLP）中使用的数学结构。它是一个表格，用于统计文档集合中术语（单词）的频率。每一行表示一个独特的术语，每一列表示一个特定的文档。它是文本分析中的一个重要工具，可以让你看到每个单词在不同文本中出现的频率。

对于包含单词`cat`和`dog`的文档：

+   文档 1：`cat cat dog`

+   文档 2：`dog dog cat`

|  | **Document 1** | **Document 2** |
| --- | --- | --- |
| **cat** | 2 | 1 |
| **dog** | 1 | 2 |

这种矩阵结构允许高效地存储、组织和分析大型文本数据集。在 Python 中，可以使用`sklearn`库中的`CountVectorizer`模块来创建一个 TDM，方法如下：

```py
from sklearn.feature_extraction.text import CountVectorizer
# Define a list of documents
documents = ["Machine Learning is useful", "Machine Learning is fun", "Machine Learning is AI"]
# Create an instance of CountVectorizer
vectorizer = CountVectorizer()
# Fit and transform the documents into a TDM
tdm = vectorizer.fit_transform(documents)
# Print the TDM
print(tdm.toarray()) 
```

输出如下：

```py
[[0 0 1 1 1 1]
 [0 1 1 1 1 0]
 [1 0 1 1 1 0]] 
```

请注意，每个文档对应一行，每个不同的单词对应一列。这里有三个文档和六个不同的单词，结果是一个 3x6 的矩阵。

在这个矩阵中，数字表示每个单词（列）在对应文档（行）中出现的频率。例如，如果第一行第一列的数字是 1，这意味着第一个单词在第一个文档中出现了一次。

默认情况下，TDM 使用每个术语的频率，这是量化每个单词在每个文档中的重要性的一种简单方法。更精细的量化方法是使用 TF-IDF，这将在下一节中解释。

## 使用 TF-IDF

**词频-逆文档频率**（**TF-IDF**）是一种用于计算单词在文档中重要性的方法。它考虑了两个主要成分来确定每个术语的权重：**词频**（**TF**）和**逆文档频率**（**IDF**）。TF 关注一个词在特定文档中出现的频率，而 IDF 则检查这个词在整个文档集合（即语料库）中的稀有程度。在 TF-IDF 的上下文中，语料库指的是你正在分析的所有文档。如果我们正在处理一组书评，举例来说，语料库将包括所有的书评：

+   **TF**：TF 衡量一个术语在文档中出现的次数。它的计算方法是术语在文档中出现的次数与文档中术语总数的比值。术语出现得越频繁，TF 值就越高。

+   **IDF**：IDF 衡量一个术语在整个文档集合中的重要性。它的计算方法是语料库中总文档数与包含该术语的文档数之比的对数。术语在语料库中越稀有，它的 IDF 值就越高。

要使用 Python 计算 TF-IDF，请执行以下操作：

```py
from sklearn.feature_extraction.text import TfidfVectorizer
# Define a list of documents
documents = ["Machine Learning enables learning", "Machine Learning is fun", "Machine Learning is useful"]
# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit and transform the documents into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
# Get the feature names
feature_names = vectorizer.get_feature_names_out()
# Loop over the feature names and print the TF-IDF score for each term
for i, term in enumerate(feature_names):
    tfidf = tfidf_matrix[:, i].toarray().flatten()
    print(f"{term}: {tfidf}") 
```

这将输出：

```py
enables:   [0.60366655 0\.         0\.        ]
fun:       [0\.         0.66283998 0\.        ]
is:        [0\.         0.50410689 0.50410689]
learning:  [0.71307037 0.39148397 0.39148397]
machine:   [0.35653519 0.39148397 0.39148397]
useful:    [0\.         0\.         0.66283998] 
```

输出中的每一列对应一个文档，行表示各文档中术语的 TF-IDF 值。例如，术语`kids`只有在第二个文档中才有非零的 TF-IDF 值，这与我们的预期一致。

## 结果总结与讨论

TF-IDF 方法提供了一种有价值的方式来衡量术语在单个文档内以及在整个语料库中的重要性。计算得到的 TF-IDF 值揭示了每个术语在每个文档中的相关性，同时考虑了它们在给定文档中的频率以及在整个语料库中的稀有性。在提供的示例中，不同术语的 TF-IDF 值变化表明该模型能够区分那些在特定文档中独有的词汇和那些使用频率较高的词汇。这种能力可以在多个应用中加以利用，如文本分类、信息检索和特征选择，提升对文本数据的理解和处理。

# 词嵌入简介

NLP 的一项重大进展是我们能够创建单词的有意义的数字表示形式，采用稠密向量的形式。这种技术称为词嵌入。那么，稠密向量到底是什么呢？假设你有一个单词 `apple`（苹果）。在词嵌入中，`apple` 可能被表示为一系列数字，例如 `[0.5, 0.8, 0.2]`，其中每个数字都是连续的、多维空间中的一个坐标。“稠密”意味着这些数字大多数或全部都不为零，不像稀疏向量那样许多元素为零。简单来说，词嵌入将文本中的每个单词转化为一个独特的、多维的空间点。这样，含义相似的单词将最终在这个空间中彼此接近，从而使算法能够理解单词之间的关系。Yoshua Bengio 在他的论文 *A Neural Probabilistic Language Model* 中首次提出了这个术语。NLP 问题中的每个单词可以被视为一个类别对象。

在词嵌入中，尝试建立每个单词的邻域，并利用它来量化单词的意义和重要性。一个单词的邻域是指围绕特定单词的一组单词。

为了真正理解词嵌入的概念，我们来看一个涉及四个常见水果词汇的具体例子：`apple`（苹果）、`banana`（香蕉）、`orange`（橙子）和`pear`（梨）。这里的目标是将这些单词表示为稠密向量，这些向量是数字数组，其中每个数字捕捉单词的特定特征或特性。

为什么要以这种方式表示单词呢？在自然语言处理（NLP）中，将单词转换为稠密向量可以使算法量化不同单词之间的关系。本质上，我们是在将抽象的语言转化为可以用数学方法衡量的内容。

考虑我们水果单词的甜度、酸度和多汁度特征。我们可以对每个水果的这些特征进行从 0 到 1 的评分，0 表示该特征完全缺失，1 表示该特征非常明显。评分可能是这样的：

```py
"apple": [0.5, 0.8, 0.2] – moderately sweet, quite acidic, not very juicy
"banana": [0.2, 0.3, 0.1] – not very sweet, moderately acidic, not juicy
"orange": [0.9, 0.6, 0.9] – very sweet, somewhat acidic, very juicy
"pear": [0.4, 0.1, 0.7] – moderately sweet, barely acidic, quite juicy 
```

这些数字是主观的，可以通过味觉测试、专家意见或其他方法得出，但它们的作用是将单词转化为算法可以理解并使用的格式。

通过可视化，你可以想象一个三维空间，其中每个坐标轴代表一个特征（甜度、酸度或多汁度），每个水果的向量将其放置在这个空间中的特定位置。具有相似口味的单词（水果）会在这个空间中彼此更接近。

那么，为什么选择长度为 3 的稠密向量呢？这是基于我们选择的特征来表示的。在其他应用中，向量的长度可能不同，取决于你希望捕捉的特征数量。

这个例子展示了词嵌入是如何将一个单词转化为一个持有实际意义的数字向量的。这是让机器“理解”并处理人类语言的关键步骤。

# 使用 Word2Vec 实现词嵌入

Word2Vec 是一种用于获取单词向量表示的突出方法，通常称为单词嵌入。该算法并不是“生成单词”，而是创建代表每个单词语义的数值向量。

Word2Vec 的基本思想是利用神经网络来预测给定文本语料库中每个单词的上下文。神经网络通过输入单词及其周围的上下文单词进行训练，网络学习输出给定输入单词的上下文单词的概率分布。神经网络的权重随后被用作单词嵌入，这些嵌入可以用于各种自然语言处理任务：

```py
import gensim
# Define a text corpus
corpus = [['apple', 'banana', 'orange', 'pear'],
          ['car', 'bus', 'train', 'plane'],
          ['dog', 'cat', 'fox', 'fish']]
# Train a word2vec model on the corpus
model = gensim.models.Word2Vec(corpus, window=5, min_count=1, workers=4) 
```

让我们分解一下`Word2Vec()`函数的重要参数：

+   **sentences**：这是模型的输入数据。它应该是一个句子的集合，每个句子是一个单词列表。实际上，它是一个单词列表的列表，代表了你的整个文本语料库。

+   **size**：定义了单词嵌入的维度。换句话说，它设置了表示单词的向量中的特征或数值的数量。一个典型的值可能是`100`或`300`，具体取决于词汇的复杂性。

+   **window**：该参数设置目标单词与句子中用于预测的上下文单词之间的最大距离。例如，如果将窗口大小设置为`5`，算法将在训练过程中考虑目标单词前后五个立即相邻的单词。

+   **min_count**：通过设置此参数，可以排除在语料库中出现频率较低的单词。例如，如果将`min_count`设置为`2`，那么在所有句子中出现次数少于两次的单词将在训练过程中被忽略。

+   **workers**：指的是训练过程中使用的处理线程数量。增加该值可以通过启用并行处理来加速在多核机器上的训练。

一旦 Word2Vec 模型训练完成，使用它的强大方法之一是测量嵌入空间中单词之间的相似性或“距离”。这个相似性得分可以让我们洞察模型如何看待不同单词之间的关系。现在，让我们通过查看`car`和`train`之间的距离来检查模型：

```py
print(model.wv.similarity('car', 'train')) 
```

```py
-0.057745814 
```

现在让我们来看一下`car`和`apple`的相似度：

```py
print(model.wv.similarity('car', 'apple')) 
```

```py
0.11117952 
```

因此，输出给我们的是基于模型学习到的单词嵌入之间的相似性得分。

## 解释相似性得分

以下细节有助于解释相似性得分：

+   **非常相似**：接近 1 的得分表示强烈的相似性。具有此得分的单词通常共享上下文或语义意义。

+   **适度相似**：接近 0.5 的得分表示某种程度的相似性，可能是由于共享的属性或主题。

+   **相似度弱或没有相似性**：接近 0 或负数的得分表示意义之间几乎没有相似性，甚至存在对比。

因此，这些相似度分数提供了关于单词关系的定量见解。通过理解这些分数，你可以更好地分析文本语料库的语义结构，并将其用于各种 NLP 任务。

Word2Vec 提供了一种强大且高效的方式来表示文本数据，能够捕捉单词之间的语义关系、减少维度并提高下游 NLP 任务的准确性。让我们来看看 Word2Vec 的优缺点。

## Word2Vec 的优缺点

以下是使用 Word2Vec 的优点：

+   **捕捉语义关系**：Word2Vec 的嵌入在向量空间中的位置使得语义相关的单词靠得很近。通过这种空间安排，捕捉了语法和语义关系，如同义词、类比等，从而在信息检索和语义分析等任务中取得更好的表现。

+   **降维**：传统的单热编码（one-hot encoding）会创建一个稀疏且高维的空间，尤其是当词汇表很大时。Word2Vec 将其压缩为一个更加密集且低维的连续向量空间（通常为 100 到 300 维）。这种压缩表示保留了重要的语言模式，同时在计算上更高效。

+   **处理词汇外单词**：Word2Vec 可以通过利用上下文词来推断未出现在训练语料中的单词的嵌入。这个特性有助于更好地泛化到未见过或新的文本数据，增强了模型的鲁棒性。

现在让我们来看看使用 Word2Vec 的一些缺点：

+   **训练复杂性**：Word2Vec 模型的训练可能需要大量计算资源，特别是在拥有庞大词汇表和高维向量时。它们需要大量的计算资源，并可能需要优化技术，如负采样或层次化软最大（hierarchical softmax），以实现高效扩展。

+   **缺乏可解释性**：Word2Vec 嵌入的连续性和密集性使得它们难以被人类理解。与精心设计的语言特征不同，Word2Vec 中的维度不对应直观的特征，这使得理解捕获了单词的哪些具体方面变得困难。

+   **对文本预处理敏感**：Word2Vec 嵌入的质量和效果可能会根据应用于文本数据的预处理步骤而显著变化。诸如分词、词干提取、词形还原或去除停用词等因素必须谨慎考虑。预处理的选择可能会影响向量空间中的空间关系，从而可能影响模型在下游任务中的表现。

接下来，我们来看一个关于餐厅评论的案例研究，结合了本章介绍的所有概念。

# 案例研究：餐厅评论情感分析

我们将使用 Yelp 评论数据集，该数据集包含标记为正面（5 星）或负面（1 星）的评论。我们将训练一个可以将餐厅评论分类为负面或正面的模型。

让我们通过以下步骤实现这个处理管道。

## 导入所需的库并加载数据集

首先，我们导入所需的包：

```py
import numpy as np
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
```

然后我们从一个`.csv`文件导入数据集：

```py
url = 'https://storage.googleapis.com/neurals/data/2023/Restaurant_Reviews.tsv'
dataset = pd.read_csv(url, delimiter='\t', quoting=3)
dataset.head() 
```

```py
 Review     Liked
0                           Wow... Loved this place.        1
1                                 Crust is not good.        0
2          Not tasty and the texture was just nasty.        0
3     Stopped by during the late May bank holiday of...     1
4      The selection on the menu was great and so wer...    1 
```

## 构建一个干净的语料库：文本数据预处理

接下来，我们通过对数据集中的每条评论进行词干提取和停用词去除等文本预处理来清洗数据：

```py
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [
        ps.stem(word) for word in text 
        if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
corpus = [clean_text(review) for review in dataset['Review']] 
```

代码遍历数据集中的每一条评论（在这种情况下是`'Review'`列），并应用`clean_text`函数对每条评论进行预处理和清洗。代码创建了一个名为`corpus`的新列表。结果是一个存储在`corpus`变量中的已清洗和预处理过的评论列表。

## 将文本数据转换为数值特征

现在让我们定义特征（由`y`表示）和标签（由`X`表示）。记住，**特征**是描述数据特征的自变量或属性，作为预测的输入。

而**标签**是模型被训练来预测的因变量或目标值，表示与特征对应的结果：

```py
vectorizer = CountVectorizer(max_features=1500)
X = vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values 
```

让我们将数据分为测试数据和训练数据：

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 
```

为了训练模型，我们使用了在*第七章*中学习的朴素贝叶斯算法：

```py
classifier = GaussianNB()
classifier.fit(X_train, y_train) 
```

让我们预测测试集的结果：

```py
y_pred = classifier.predict(X_test) 
```

接下来，我们打印混淆矩阵。记住，混淆矩阵是一个帮助可视化分类模型表现的表格：

```py
cm = confusion_matrix(y_test, y_pred)
print(cm) 
```

```py
[[55 42]
 [12 91]] 
```

通过查看混淆矩阵，我们可以估算误分类情况。

## 分析结果

混淆矩阵让我们窥见了模型所做的误分类。在这个背景下，有：

+   55 个真正的正例（正确预测的正面评论）

+   42 个假正例（错误预测为正面的评论）

+   12 个假负例（错误预测为负面的评论）

+   91 个真正的负例（正确预测的负面评论）

55 个真正的正例和 91 个真正的负例表明我们的模型具有合理的能力，能够区分正面和负面评论。然而，42 个假正例和 12 个假负例突显了潜在的改进空间。

在餐厅评论的背景下，理解这些数字有助于商家和顾客评估整体情感。高比例的真正正例和真正负例表明模型能够被信任，提供准确的情感概述。这些信息对于想要提升服务的餐厅或寻求真实评论的潜在顾客来说，可能非常宝贵。另一方面，假正例和假负例的存在则表明模型可能需要调整，以避免误分类并提供更准确的洞察。

# 自然语言处理（NLP）的应用

自然语言处理（NLP）技术的持续进步彻底改变了我们与计算机和其他数字设备的互动方式。近年来，它在多个任务上取得了显著进展，并取得了令人印象深刻的成就，包括：

+   **主题识别**：在文本库中发现主题，并根据发现的主题对库中的文档进行分类。

+   **情感分析**：根据文本中的正面或负面情感对其进行分类。

+   **机器翻译**：在不同语言之间进行翻译。

+   **语音转文本**：将口语转换为文本。

+   **问答**：这是通过使用可用信息来理解和回应查询的过程。它涉及智能地解读问题，并根据现有知识或数据提供相关的答案。

+   **实体识别**：从文本中识别实体（如人、地点或事物）。

+   **假新闻检测**：根据内容标记假新闻。

# 总结

本章讨论了与 NLP 相关的基本术语，如语料库、词向量、语言建模、机器翻译和情感分析。此外，本章还介绍了 NLP 中至关重要的各种文本预处理技术，包括分词，它将文本分解成称为标记的小单位，以及其他技术，如词干提取和去除停用词。

本章还讨论了词向量，并展示了一个餐厅评论情感分析的案例。现在，读者应当对 NLP 中使用的基本技术及其在现实世界问题中的潜在应用有了更好的理解。

在下一章，我们将讨论如何训练处理顺序数据的神经网络。我们还将探讨如何利用深度学习进一步改善自然语言处理（NLP）技术和本章讨论的方法论。

# 了解更多信息，请访问 Discord

要加入本书的 Discord 社区 —— 你可以在这里分享反馈、向作者提问并了解新版本 —— 请扫描下面的二维码：

[`packt.link/WHLel`](https://packt.link/WHLel)

![](img/QR_Code1955211820597889031.png)
