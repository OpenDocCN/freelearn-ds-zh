

# 第十二章：大规模语言模型时代的文本预处理

在**大规模语言模型**（**LLMs**）时代，掌握文本预处理比以往任何时候都更加重要。随着 LLMs 在复杂性和能力上的不断提升，成功的**自然语言处理**（**NLP**）任务的基础依然在于文本数据的准备工作。在本章中，我们将讨论文本预处理，这是任何 NLP 任务的基础。我们还将探讨重要的预处理技术，并重点研究如何调整这些技术以最大化 LLMs 的潜力。

在本章中，我们将覆盖以下主题：

+   在大规模语言模型时代重新学习文本预处理

+   文本清洗技术

+   处理稀有词汇和拼写变体

+   词块划分

+   分词策略

+   将词元转化为嵌入

# 技术要求

本章的完整代码可以在以下 GitHub 仓库中找到：

[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/tree/main/chapter12`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/tree/main/chapter12)

让我们安装本章中将使用的必要库：

```py
pip install transformers==4.42.4
pip install beautifulsoup4==4.12.3
pip install langchain-text-splitters==0.2.2
pip install tiktoken==0.7.0
pip install langchain==0.2.10
pip install langchain-experimental==0.0.62
pip install langchain-huggingface==0.0.3
pip install presidio_analyzer==2.2.355
pip install presidio_anonymizer==2.2.355
pip install rapidfuzz-3.9.4 thefuzz-0.22.1
pip install stanza==1.8.2
pip install tf-keras-2.17.0
```

# 在大规模语言模型时代重新学习文本预处理

**文本预处理**是指对原始文本数据应用各种技术，目的是清理、组织并将其转化为适合分析或建模的格式。其主要目标是通过解决与非结构化文本相关的常见挑战来提高数据的质量。这包括清理无关字符、处理变体以及为后续的自然语言处理（NLP）任务准备数据等任务。

随着大规模语言模型（LLMs）的快速发展，自然语言处理（NLP）的格局发生了显著变化。然而，基础的预处理技术，如文本清洗和分词，依然至关重要，尽管它们在方法和重要性上有所变化。

从文本清洗开始，尽管大规模语言模型（LLMs）在处理输入文本噪声方面表现出显著的鲁棒性，但清洗后的数据仍然能带来更好的结果，尤其在微调任务中尤为重要。基础清洗技术，如去除 HTML 标签、处理特殊字符以及文本标准化，依然是相关的。然而，像拼写纠正这样的高级技术对于 LLMs 的必要性可能较低，因为它们通常能处理轻微的拼写错误。领域特定的清洗仍然非常重要，尤其是在处理专业词汇或术语时。

随着子词标记化方法的出现，Tokenization 也得到了发展，现代大多数 LLM（大规模语言模型）都使用如**字节对编码**（**BPE**）或 WordPiece 等方法。传统的基于词的标记化在 LLM 的背景下不再常见。一些传统的 NLP 预处理步骤，如停用词去除、词干提取和词形还原，变得不那么重要。停用词去除，即去除常见词汇，如“and”或“the”，变得不那么必要，因为 LLM 能够理解这些词在上下文中的重要性以及它们如何贡献于句子的意义。类似地，词干提取和词形还原（如将“running”还原为“run”）也不常使用，因为 LLM 能够准确理解不同词形，并理解它们在文本中的关系。这一转变使得对语言的理解更加细致，能够捕捉到一些严格预处理可能遗漏的细微差别。

关键的信息是，虽然 LLM 能够令人印象深刻地处理原始文本，但在某些情境下，预处理仍然至关重要，因为它可以提高模型在特定任务上的表现。记住：**垃圾进，垃圾出**。清洗和标准化文本也可以减少 LLM 处理的 token 数量，从而可能降低计算成本。新的方法正在出现，它们将传统的预处理与 LLM 的能力相结合，利用 LLM 本身来进行数据清洗和预处理任务。

总之，尽管 LLM 在许多 NLP 任务中减少了对广泛预处理的需求，但理解并谨慎应用这些基础技术仍然具有价值。在接下来的章节中，我们将重点介绍仍然相关的文本预处理技术。

# 文本清洗

文本清洗的主要目标是将非结构化的文本信息转化为标准化且更易处理的形式。在清洗文本时，常见的操作包括去除 HTML 标签、特殊字符和数字值，标准化字母大小写，处理空格和格式问题。这些操作共同有助于提升文本数据的质量并减少其歧义性。让我们深入探讨这些技术。

## 去除 HTML 标签和特殊字符

HTML 标签通常会出现在从网页中提取内容的过程中。这些标签，如`<p>`、`<a>`或`<div>`，在 NLP 的上下文中*没有语义意义*，必须被移除。清洗过程包括识别并去除 HTML 标签，保留实际的文本内容。

对于这个示例，让我们假设我们有一个产品的用户评论数据集，并希望为情感分析准备文本数据。你可以在 GitHub 代码库中找到这一部分的代码：[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/1.text_cleaning.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/1.text_cleaning.py)。在这个脚本中，数据生成也已提供，你可以一步一步跟着示例走。

重要提示

在本章中，我们包含了关键的代码片段，以说明最重要的概念。然而，要查看完整的代码，包括使用的库，并运行完整的端到端示例，请访问代码库。

我们将执行的第一个文本预处理步骤是移除 HTML 标签。让我们一步步查看代码：

1.  让我们为这个示例导入所需的库：

    ```py
    from bs4 import BeautifulSoup
    from transformers import BertTokenizer
    ```

1.  这里展示了示例用户评论：

    ```py
    reviews = [
      "<html>This product is <b>amazing!</b></html>",
      "The product is good, but it could be better!!!",
      "I've never seen such a terrible product. 0/10",
      "The product is AWESOME!!! Highly recommended!",
    ]
    ```

1.  接下来，我们创建一个使用`BeautifulSoup`来解析 HTML 内容并提取文本的函数，移除所有 HTML 标签：

    ```py
    def clean_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    ```

1.  然后，我们对所有评论进行预处理：

    ```py
    def preprocess_text(text):
        text = clean_html_tags(text)
        return text
    preprocessed_reviews = [preprocess_text(review) for review in reviews]
    ```

1.  最后，我们得到以下的预处理评论：

    ```py
    - This product is amazing!
    - The product is good, but it could be better!!!
    - I've never seen such a terrible product. 0/10
    - The product is AWESOME!!! Highly recommended!
    ```

如我们所见，所有 HTML 标签已经被移除，文本变得干净整洁。我们将继续通过添加另一个常见的预处理步骤来增强此示例：处理文本的大小写。

## 处理大小写

文本数据通常有各种大小写——大写、小写或两者的混合。不一致的大小写可能会导致语言处理任务中的歧义。因此，一种常见的文本清理做法是统一整个语料库中的字母大小写。这不仅有助于保持一致性，还能确保模型在不同大小写之间具有良好的泛化能力。

基于前面的示例，我们将扩展预处理函数，增加一个额外步骤：字母标准化：

1.  让我们首先回顾一下在上一步移除 HTML 标签后的评论是怎样的：

    ```py
    - This product is amazing!
    - The product is good, but it could be better!!!
    - I've never seen such a terrible product. 0/10
    - The product is AWESOME!!! Highly recommended!
    ```

1.  以下函数将把所有字符转换为小写字母：

    ```py
    def standardize_case(text):
        return text.lower()
    ```

1.  我们将扩展在前一个示例中介绍的`preprocess_text`函数，将文本中的所有字符转换为小写字母，使得文本对大小写不敏感：

    ```py
    def preprocess_text(text):
        text = clean_html_tags(text)
        text = standardize_case(text)
        return text
    ```

1.  让我们打印出预处理后的评论：

    ```py
    for preprocessed_review in preprocessed_reviews:
        print(f"- {preprocessed_review}")
    ```

    这里展示了小写处理后的评论：

    ```py
    - this product is amazing!
    - the product is good, but it could be better!!!
    - i've never seen such a terrible product. 0/10
    - the product is awesome!!! highly recommended!
    ```

注意所有字母都变成小写了！请继续更新大小写函数，按照以下方式将所有内容转换为大写：

```py
def standardize_case(text):
    return text.upper()
```

这里展示了大写字母的评论：

```py
- THIS PRODUCT IS AMAZING!
- THE PRODUCT IS GOOD, BUT IT COULD BE BETTER!!!
- I'VE NEVER SEEN SUCH A TERRIBLE PRODUCT. 0/10
- THE PRODUCT IS AWESOME!!! HIGHLY RECOMMENDED!
```

如果你在犹豫是否应该使用小写或大写，我们已经为你准备好了答案。

### 小写还是大写？

使用小写或大写文本的选择取决于 NLP 任务的具体要求。例如，情感分析等任务通常更适合小写处理，因为这能简化文本并减少变异性。相反，像**命名实体识别（NER）**这样的任务可能需要保留大小写信息，以便准确识别和区分实体。

例如，在德语中，所有名词都需要大写，因此保持大小写对于正确的语言表现至关重要。相比之下，英语通常不使用大小写来表达意义，因此对于一般文本分析来说，转换为小写可能更为合适。

在处理来自用户输入的文本数据时，如社交媒体帖子或评论，考虑大小写变化的作用非常重要。例如，一条推文可能会使用混合大小写来强调或表达语气，这对于情感分析可能是相关的。

现代大型语言模型（LLMs），如**双向编码器表示（BERT）**和 GPT-3，都是在混合大小写文本上训练的，能够有效处理大写和小写。这些模型利用大小写信息来增强上下文理解。它们的分词器本身设计能处理大小写敏感性，无需显式转换。

如果你的任务需要区分不同的大小写（例如，识别专有名词或首字母缩略词），最好保留原始的大小写。然而，始终参考你所使用的模型的文档和最佳实践。有些模型可能已优化为适应小写输入，如果文本转换为小写，可能会表现得更好。

下一步是学习如何处理文本中的数字值和符号。

## 处理数字值和符号

数字值、符号和数学表达式可能会出现在文本数据中，但并不总是对上下文产生有意义的贡献。清理它们需要根据任务的具体要求决定是保留、替换还是删除这些元素。

例如，在情感分析中，数字值可能不太相关，它们的存在可能会分散注意力。相反，对于与定量分析或金融情感相关的任务，保留数字信息变得至关重要。

在前面的示例基础上，我们将删除文本中的所有数字和符号：

1.  让我们回顾一下上一步预处理后的数据样貌：

    ```py
    - This product is amazing!
    - The product is good, but it could be better!!!
    - I've never seen such a terrible product. 0/10
    - The product is AWESOME!!! Highly recommended!
    ```

1.  现在，让我们添加一个函数，删除文本中所有*除字母字符外* *和空格*以外的字符：

    ```py
    def remove_numbers_and_symbols(text):
        return ''.join(e for e in text if e.isalpha() or e.isspace())
    ```

1.  应用文本预处理流程：

    ```py
    def preprocess_text(text):
        text = clean_html_tags(text)
        text = standardize_case(text)
        text = remove_numbers_and_symbols(text)
        return text
    ```

1.  让我们来看看预处理后的评论：

    ```py
    - this product is amazing
    - the product is good but it could be better
    - ive never seen such a terrible product
    - the product is awesome highly recommended
    ```

如你所见，在这个预处理步骤之后，文本中的所有标点符号和符号都已被移除。在文本预处理过程中，是否保留、替换或删除符号和标点符号，取决于你 NLP 任务的具体目标和数据集的特征。

### 保留符号和标点符号

随着大规模语言模型（LLMs）的发展，处理标点符号和符号的预处理方法已经发生了显著变化。现代大规模语言模型通过保留标点符号和符号受益，因为它们在多样化数据集上的广泛训练帮助模型更准确地理解上下文。保留这些符号有助于模型捕捉情感、强调和句子边界等细微差别。例如，感叹号和问号等标点符号在情感分析中发挥着重要作用，通过传达强烈的情感，提升了模型的表现。同样，在文本生成任务中，标点符号维持了可读性和结构，而在命名实体识别（NER）和翻译中，它有助于识别专有名词和句子边界。

另一方面，有些情况下，移除标点符号和符号可能会带来优势。现代大规模语言模型（LLMs）足够强大，能够处理噪声数据，但在某些应用中，通过移除标点符号简化文本可以优化预处理并*减少独特标记的数量*。这种方法对于主题建模和聚类等任务尤为有用，因为这些任务更侧重于内容而非结构元素。例如，移除标点符号有助于通过消除句子结构中的干扰来识别核心主题，而在文本分类中，当标点符号没有提供显著价值时，它可以帮助标准化输入数据。

另一种方法是用空格或特定标记替换标点符号和符号，这有助于在规范化文本时保持标记之间的某种分隔。这种方法对于自定义分词策略特别有用。在专业的自然语言处理（NLP）管道中，将标点符号替换为特定标记可以保留重要的区别，而不会给文本添加不必要的杂乱，从而促进更有效的分词和下游任务的预处理。

让我们通过一个简单的例子来看看如何移除或替换符号和标点符号。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/2.punctuation.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/2.punctuation.py)找到本节的代码：

+   创建示例文本：

    ```py
    text = "I love this product!!! It's amazing!!!"
    ```

+   选项 1：用空格替换符号和标点符号：

    ```py
    replaced_text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    print("Replaced Text:", replaced_text)
    ```

    这将打印以下输出：

    ```py
    I love this product    It s amazing
    ```

+   选项 2：移除符号和标点符号：

    ```py
    removed_text = "".join(char for char in text if char.isalnum() or char.isspace())
    print("Removed Text:", removed_text)
    ```

    这将打印以下输出：

    ```py
    I love this product Its amazing
    ```

移除符号和数字是文本分析中的一个重要预处理步骤，它通过消除非字母数字字符简化了文本。在本节的最后，我们将讨论解决空格问题，以提高文本的可读性并确保一致的格式。

## 处理空格和格式问题

空格和格式不一致在文本数据中是常见的，尤其是当数据来源多样时。清理过程涉及解决多个连续空格、前后空格以及格式样式差异等问题。空格的规范化确保了文本表示的一致性，减少了下游模型误解的风险。

解决空格和格式化问题在大语言模型（LLMs）的世界中依然至关重要。尽管现代 LLMs 对各种格式不一致表现出较强的鲁棒性，但有效管理空格和格式仍能提升模型表现并确保数据一致性。

规范化空格和格式化可以创建统一的数据集，这通过最小化噪声并将注意力集中在内容上而非格式差异，有助于模型训练和分析。通过适当的空格管理提高可读性，有助于人类和机器学习的解读，清晰地划定文本元素。此外，一致的空格处理对于准确的分词非常重要——这是许多 NLP 任务中的基础过程——它确保了单词和短语的精确识别和处理。

所以，让我们回到评论示例，并在流程中添加另一步骤来去除空格：

1.  让我们先从解决空格和格式化问题开始。此函数移除多余的空格，并确保单词之间只有一个空格：

    ```py
    def remove_extra_whitespace(text):
        return ' '.join(text.split())
    ```

1.  接下来，我们将在文本预处理管道中添加这一步骤：

    ```py
    def preprocess_text(text):
        text = clean_html_tags(text)
        text = standardize_case(text)
        text = remove_numbers_and_symbols(text)
        text = remove_extra_whitespace(text)
        return text
    ```

让我们在应用新步骤之前，先看一下评论，并集中注意力于这里标记的空格：

```py
- this productis amazing
- the product is good but it could be better
- ive never seen such a terribleproduct
- the product is awesome highly recommended
```

最后，在应用了空格移除之后，让我们检查清理后的数据集：

```py
- this product is amazing
- the product is good but it could be better
- ive never seen such a terrible product
- the product is awesome highly recommended
```

让我们从纯文本清理过渡到专注于保护数据。

## 去除个人身份信息

在预处理文本数据时，去除**个人身份信息**（**PII**）对于维护隐私、确保符合规定以及提高数据质量至关重要。例如，考虑一个包含用户名、电子邮件地址和电话号码的用户评论数据集。如果这些敏感信息未被匿名化或移除，将会带来诸如隐私侵犯和潜在滥用等重大风险。**通用数据保护条例**（**GDPR**）、**加利福尼亚消费者隐私法案**（**CCPA**）和**健康保险流动性与责任法案**（**HIPAA**）等法规要求对个人数据进行小心处理。未能移除 PII 可能会导致法律处罚和信任丧失。此外，包含可识别的细节可能会给机器学习模型引入偏差，影响其泛化能力。去除 PII 对于负责任的人工智能开发至关重要，因为这可以在保持个人隐私的同时创建和使用数据集，为研究和分析提供有价值的见解。

以下代码片段演示了如何使用`presidio-analyzer`和`presidio-anonymizer`库来检测和匿名化 PII（个人身份信息）。我们一步步来看一下代码。完整代码可以通过以下链接访问：[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/3.pii_detection.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/3.pii_detection.py)：

1.  让我们首先导入本示例所需的库：

    ```py
    import pandas as pd
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    ```

1.  我们创建一个示例 DataFrame，其中有一列名为`text`，包含包含不同类型 PII（例如，姓名、电子邮件地址和电话号码）的句子：

    ```py
    data = {
        'text': [
            "Hello, my name is John Doe. My email is john.doe@example.com",
            "Contact Jane Smith at jane.smith@work.com",
            "Call her at 987-654-3210.",
            "This is a test message without PII."
        ]
    }
    df = pd.DataFrame(data)
    ```

1.  我们初始化`AnalyzerEngine`来*检测*PII 实体，并初始化`AnonymizerEngine`来*匿名化*检测到的 PII 实体：

    ```py
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    ```

1.  接下来，我们将定义一个匿名化函数，该函数在文本中检测 PII 并根据实体类型应用掩码规则：

    ```py
    def anonymize_text(text):
        analyzer_results = analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], language="en")
        operators = {
            "PERSON": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 4, "from_end": True}),
            "EMAIL_ADDRESS": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 5, "from_end": True}),
            "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 6, "from_end": True})
        }
        anonymized_result = anonymizer.anonymize(
            text=text, analyzer_results=analyzer_results,
            operators=operators)
        return anonymized_result.text
    ```

    `anonymize_text`函数旨在通过匿名化特定类型的实体来保护给定文本中的敏感信息。它首先分析文本，识别出姓名（`PERSON`）、电子邮件地址（`EMAIL_ADDRESS`）和电话号码（`PHONE_NUMBER`）等实体。对于每种实体类型，它应用掩码操作来隐藏部分信息。具体来说，它会掩盖人名的最后四个字符、电子邮件地址的最后五个字符和电话号码的最后六个字符。该函数返回匿名化后的文本，确保个人信息被隐藏，同时保留文本的整体结构。

1.  将匿名化函数应用于 DataFrame：

    ```py
    df['anonymized_text'] = df['text'].apply(anonymize_text)
    ```

1.  显示 DataFrame：

    ```py
    0    Hello, my name is John. My email is john.d...
    1            Contact Jane S at jane.smith@wor*
    2                            Call her at 987-65.
    3                  This is a test message without PII.
    ```

通过使用这些配置，您可以根据特定需求定制匿名化过程，确保敏感信息得到适当保护。这种方法有助于您遵守隐私法规，并保护数据集中的敏感信息。

虽然删除 PII 对于保护隐私和确保数据合规性至关重要，但文本预处理的另一个关键方面是处理稀有词汇和拼写变体。

# 处理稀有词汇和拼写变体

大型语言模型（LLMs）的崛起彻底改变了我们与技术互动和处理信息的方式，特别是在处理拼写变化和罕见词汇的领域。在 LLMs 出现之前，管理这些语言挑战需要大量的人工努力，通常涉及专业知识和精心设计的算法。传统的拼写检查器和语言处理工具在处理罕见词和变化时常常力不从心，导致频繁的错误和低效。今天，像 GPT-4、Llama3 等 LLMs 通过利用庞大的数据集和复杂的机器学习技术，已经彻底改变了这一局面，它们能够理解并生成适应各种拼写变化和罕见术语的文本。这些模型能够识别并修正拼写错误，提供上下文适当的建议，并准确解释罕见词汇，从而提高文本处理的精准度和可靠性。

## 处理罕见词

在像 GPT-3 和 GPT-4 这样的 LLMs 时代，处理罕见词相比传统的自然语言处理（NLP）方法已经不再是一个大问题。这些模型在庞大而多样的数据集上进行了训练，使它们能够理解并生成带有罕见甚至未见过的词汇的文本。然而，在文本预处理和有效处理罕见词方面仍然需要一些注意事项。

那么，如何使用 LLMs 处理罕见词呢？我们需要理解一些关键概念，从分词开始。我们这里不深入探讨分词，因为稍后会有专门的部分进行讨论；现在，假设 LLMs 使用**子词分词**方法，将罕见词拆解成更常见的子词单元。这有助于通过将罕见词拆解成熟悉的组件来管理**词汇外**（**OOV**）词汇。关于 LLMs 的另一个有趣之处是，即使它们本身不认识某个词，它们也具备上下文理解能力，这意味着 LLMs 能够通过上下文推测罕见词的含义。

在下面的代码示例中，我们将测试 GPT-2 是否能处理罕见词。您可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/4.rare_words.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/4.rare_words.py)找到代码：

1.  让我们导入所需的库：

    ```py
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    ```

1.  初始化 GPT-2 的分词器和模型：

    ```py
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    ```

1.  使用罕见词定义文本提示：

    ```py
    text = "The quokka, a rare marsupial,"
    ```

1.  将输入文本编码为标记：

    ```py
    indexed_tokens = tokenizer.encode(text, return_tensors='pt')
    ```

1.  生成文本直到输出长度达到 50 个标记。模型根据输入提示生成文本，利用其对上下文的理解来处理罕见词：

    ```py
    output_text = model.generate(indexed_tokens, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    ```

    给定代码片段中的`generate`函数用于根据提供的输入标记生成模型的文本输出。此函数调用中的参数控制了文本生成过程中的各个方面：

    +   `indexed_tokens`：这表示模型将用来开始生成文本的输入序列。它由令牌化的文本组成，作为生成的起点。

    +   `max_length=50`：此参数设置生成文本的最大长度。模型将生成多达 50 个令牌，包括输入令牌，确保输出不超过此长度。

    +   `num_beams=5`：这控制梁搜索过程，模型在生成过程中跟踪最有可能的五个序列。梁搜索通过同时探索多个可能的结果并选择最可能的结果来提高生成文本的质量。

    +   `no_repeat_ngram_size=2`：这防止模型在生成文本中重复任何两个令牌（二元组）。通过确保相同的短语不会多次出现，它有助于生成更连贯和少重复的输出。

    +   `early_stopping=True`：此参数允许生成过程在所有梁都到达文本序列末端（例如，句子结束令牌）时提前停止。通过在已经生成了完整且合理的输出时避免不必要的继续，这可以使生成过程更高效。

这些参数可以根据所需的输出进行调整。例如，增加`max_length`会生成更长的文本，而修改`num_beams`可以在质量和计算成本之间进行平衡。调整`no_repeat_ngram_size`可以改变重复预防的严格性，而切换`early_stopping`可能会影响生成文本的效率和长度。*我建议你去尝试这些配置，看看它们的输出* *会如何受到影响*：

1.  生成的令牌被解码成可读的文本：

    ```py
    output_text_decoded = tokenizer.decode(output_text[0], skip_special_tokens=True)
    ```

1.  打印解码后的文本：

    ```py
    The quokka, a rare marsupial, is one of the world's most endangered species.
    ```

正如我们所见，模型理解了*短尾树袋鼠*的含义，并创建了一个单词序列，这是从提示中继续的额外文本，展示了 LLM 的语言生成能力。这是可能的，因为 LLM 将令牌转换为称为**嵌入**的数字表示，我们将在稍后看到，它捕捉了单词的含义。

我们讨论了在文本预处理中使用罕见词。现在让我们转向另一个挑战——拼写错误和拼写错误。

## 处理拼写变体和拼写错误

拼写变体和拼写错误的挑战在于它可能导致*相似单词的不同标记化方式*。在 LLM 时代，处理拼写和拼写错误已变得更加复杂。LLM 可以理解上下文并生成文本，通常会隐式地纠正这些错误。然而，显式预处理以纠正拼写错误仍然可以提高这些模型的性能，特别是在准确性至关重要的应用中。有多种方法可以解决拼写变体和错误，我们将在接下来的部分中看到，从拼写校正开始。

### 拼写校正

让我们通过使用 Hugging Face Transformers 的大语言模型（LLM）创建一个修正拼写错误的示例。我们将使用实验性的`oliverguhr/spelling-correction-english-base`拼写校正模型进行演示。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/5.spelling_checker.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/5.spelling_checker.py)找到完整的代码：

1.  定义拼写校正函数管道。在这个函数内部，我们使用`oliverguhr/spelling-correction-english-base`模型初始化拼写校正管道。这个模型是专门为拼写校正任务训练的：

    ```py
    def fix_spelling(text):
        spell_check = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")
    ```

1.  我们使用管道生成校正后的文本。`max_length`参数设置为`2048`，以便处理较长的输入文本：

    ```py
        corrected = spell_check(text, max_length=2048)[0]['generated_text']
        return corrected
    ```

1.  使用包含拼写错误的示例文本测试该函数：

    ```py
    sample_text = "My name si from Grece."
    corrected_text = fix_spelling(sample_text)
    Corrected text: My name is from Greece.
    ```

需要注意的是，这是一个实验性模型，它的表现可能会因输入文本的复杂性和上下文而有所不同。为了更稳健的拼写和语法校正，你可以考虑使用更高级的模型；然而，其中一些模型需要认证才能下载或签署协议。因此，为了简便起见，我们在这里使用了一个实验性模型。你可以将其替换为你能够访问的任何模型，从 Llama3 到 GPT4 等等。

拼写校正对于文本预处理任务的重要性将我们引入了模糊匹配的概念，这是一种通过容忍输入文本中的小错误和变化，进一步提高生成内容的准确性和相关性的技术。

### 模糊匹配

模糊匹配是一种用于比较字符串相似性的技术，即使它们并不完全相同。它就像是在寻找“有点相似”或“足够接近”的单词。因此，我们可以使用模糊匹配算法来识别和映射相似的单词，以及解决变体和小的拼写错误。我们可以通过添加使用`TheFuzz`库的模糊匹配来增强拼写校正功能。

让我们浏览一下你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/6.fuzzy_matching.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/6.fuzzy_matching.py)找到的代码：

1.  我们将从安装库开始：

    ```py
    pip install thefuzz==0.22.1
    ```

1.  让我们导入所需的库：

    ```py
    from transformers import pipeline
    from thefuzz import process, fuzz
    ```

1.  初始化拼写校正管道：

    ```py
    def fix_spelling(text, threshold=80):
        spell_check = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")
    ```

    `oliverguhr/spelling-correction-english-base` 模型专门为拼写修正任务进行了精细调整，使其成为一个高效且有效的拼写修正工具。该模型已经经过训练，能够识别并修正英语文本中的常见拼写错误，从而提高准确性。它经过优化，适用于文本到文本的生成，使其能够高效地生成输入文本的修正版本，并且计算开销最小。此外，模型的训练可能涉及了包含拼写错误及其修正的语料库，使其能够做出有根据且符合语境的修正。

1.  生成与上一节中相同的修正文本：

    ```py
        corrected = spell_check(text, max_length=2048)[0]['generated_text']
    ```

1.  将原始文本和修正后的文本分解为单词：

    ```py
        original_words = text.split()
        corrected_words = corrected.split()
    ```

1.  创建一个常见英语单词的词典（你可以扩展这个列表）：

    ```py
        common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'])
    ```

1.  模糊匹配每个单词：

    ```py
        final_words = []
        for orig, corr in zip(original_words, corrected_words):
            if orig.lower() in common_words:
                final_words.append(orig)
            else:
                matches = process.extractOne(orig, [corr], scorer=fuzz.ratio)
                if matches[1] >= threshold:
                    final_words.append(matches[0])
                else:
                    final_words.append(orig)
        return ' '.join(final_words)
    ```

1.  用包含拼写错误的一些示例文本测试函数：

    ```py
    sample_text = "Lets do a copmarsion of speling mistaks in this sentense."
    corrected_text = fix_spelling(sample_text)
    ```

1.  打印结果：

    ```py
    Original text: Lets do a copmarsion of speling mistaks in this sentense.
    Corrected text: Let's do a comparison of speling mistaks in this sentence.
    ```

如你所见，并非所有拼写错误都已被修正。通过针对模型常常遗漏的例子进行微调，我们可以获得更好的表现。然而，好消息是！大语言模型（LLM）的兴起使得拼写错误的修正变得不那么重要，因为这些模型设计上是为了理解和处理文本的上下文。即使单词拼写错误，LLM 也能通过分析周围的单词和整体句子结构推断出意图的含义。这种能力减少了对拼写完美的需求，因为焦点转向了传达信息，而不是确保每个单词的拼写都正确。

完成初步的文本预处理步骤后，下一步至关重要的是**分块**。这一过程涉及将清理过的文本分解成更小、更有意义的单元。我们将在接下来的部分讨论这一点。

# 分块

分块是自然语言处理（NLP）中的一个基本预处理步骤，它涉及将文本拆分成更小、更易管理的单元，或称“块”。这一过程对于多种应用至关重要，包括文本摘要、情感分析、信息提取等。

为什么分块变得越来越重要？通过将大型文档分解，分块提高了可管理性和效率，尤其是对于具有*令牌限制*的模型，防止过载并实现更平稳的处理。它还通过允许模型*专注于更小、更连贯的文本片段*来提高准确性，相较于分析整个文档，这样可以减少噪音和复杂性。此外，分块有助于在每个片段中保持上下文，这对于机器翻译和文本生成等任务至关重要，确保模型能够有效理解和处理文本。

分块可以通过多种方式实现；例如，摘要任务可能更适合段落级分块，而情感分析可能使用句子级分块来捕捉细微的情感变化。在接下来的部分中，我们将专注于固定长度分块、递归分块和语义分块，因为它们在数据领域中更为常见。

### 实现固定长度分块

**固定长度分块**涉及将文本分成*预定义长度*的块，可以按字符数或标记数来划分。通常更为优选，因为它实现简单且确保块的大小一致。然而，由于划分是随机的，它可能会把句子或语义单元拆开，导致上下文的丧失。它适用于需要统一块大小的任务，如某些类型的文本分类。

为了展示固定长度分块，我们将再次使用评论数据，但这次会包括一些较长的评论。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/7.fixed_chunking.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/7.fixed_chunking.py)查看完整示例：

1.  让我们先加载示例数据：

    ```py
    reviews = [
        "This smartphone has an excellent camera. The photos are sharp and the colors are vibrant. Overall, very satisfied with my purchase.",
        "I was disappointed with the laptop's performance. It frequently lags and the battery life is shorter than expected.",
        "The blender works great for making smoothies. It's powerful and easy to clean. Definitely worth the price.",
        "Customer support was unresponsive. I had to wait a long time for a reply, and my issue was not resolved satisfactorily.",
        "The book is a fascinating read. The storyline is engaging and the characters are well-developed. Highly recommend to all readers."
    ]
    ```

1.  导入`TokenTextSplitter`类：

    ```py
    from langchain_text_splitters import TokenTextSplitter
    ```

1.  初始化`TokenTextSplitter`类，设置块大小为`50`个标记，并且没有重叠：

    ```py
    text_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=0)
    ```

1.  将评论合并成一个文本块进行分块：

    ```py
    text_block = " ".join(reviews)
    ```

1.  将文本拆分为基于标记的块：

    ```py
    chunks = text_splitter.split_text(text_block)
    ```

1.  打印分块：

    ```py
    Chunk 1:
    This smartphone has an excellent camera. The photos are sharp and the colors are vibrant. Overall, very satisfied with my purchase. I was disappointed with the laptop's performance. It frequently lags and the battery life is shorter than expected. The blender works
    Chunk 2:
    great for making smoothies. It's powerful and easy to clean. Definitely worth the price. Customer support was unresponsive. I had to wait a long time for a reply, and my issue was not resolved satisfactorily. The book is a
    Chunk 3:
    fascinating read. The storyline is engaging and the characters are well-developed. Highly recommend to all readers.
    ```

为了了解不同块大小如何影响输出，你可以修改`chunk_size`参数。例如，你可以尝试`20`、`70`和`150`标记大小。这里，你可以看到如何调整代码来测试不同的块大小：

```py
chunk_sizes = [20, 70, 150]
for size in chunk_sizes:
    print(f"Chunk Size: {size}")
    text_splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=0)
    chunks = text_splitter.split_text(text_block)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(chunk)
        print("\n")
```

我们成功地将评论划分为所需的块，但在继续之前，理解`chunk_overlap=0`参数的重要性是至关重要的。

#### 块重叠

**块重叠**是指在拆分文本时，*相邻块之间共享*的字符或标记数。它是两个块之间“重叠”的文本量。

块重叠非常重要，因为它有助于保持上下文并增强文本的连贯性。通过确保相邻的块共享一些共同的内容，重叠*保持连续性*，防止重要信息在边界处丢失。例如，如果文档被分割成没有重叠的块，可能会有关键信息被分割成两个块，导致无法访问或丧失意义。在检索任务中，如搜索或问答，重叠确保即使相关细节跨越块边界，也能被捕捉到，从而提高检索过程的效果。例如，如果一个块在句子中间结束，重叠确保整个句子都会被考虑到，这是准确理解和生成回答所必需的。

让我们考虑一个简单的例子来说明块重叠：

```py
Original text:
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.
```

使用五个单词的块大小和一个单词的重叠，我们将得到以下结果：

```py
Chunk 1: "One of the most important"
Chunk 2: "important things I didn't understand"
Chunk 3: "understand about the world when"
Chunk 4: "when I was a child"
Chunk 5: "child is the degree to"
Chunk 6: "to which the returns for"
Chunk 7: "for performance are superlinear."
```

正如你所看到的，每个块与下一个块之间有*两个单词*的重叠，这有助于保持上下文并防止在块边界丢失意义。固定长度的分块将文本分割成大小均匀的段落，但这种方法有时会无法捕捉到有意义的文本单元，特别是在处理自然语言固有的变动性时。另一方面，转向段落分块，通过根据文本的自然结构进行分割，提供了一种更具上下文连贯性的方法。

### 实现递归字符分块

`RecursiveCharacterTextSplitter` 是一个复杂的文本分割工具，专为处理更复杂的文本分割任务而设计，特别是当处理需要分解成更小、更有意义的块的长文档时。与简单的文本分割器不同，后者只是将文本切割成固定或可变大小的块，`RecursiveCharacterTextSplitter` 使用递归方法来分割文本，确保每个块在上下文上既连贯又适合自然语言模型处理。从回顾示例开始，我们将演示如何使用 `RecursiveCharacterTextSplitter` 将文档分割成段落。完整的代码可以在 [`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/8.paragraph_chunking.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/8.paragraph_chunking.py) 中找到：

1.  我们创建一个 `RecursiveCharacterTextSplitter` 实例：

    ```py
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=200,
        chunk_overlap=0,
        length_function=len
        )
    ```

    `RecursiveCharacterTextSplitter` 实例是通过特定参数实例化的：

    +   `separators`：这是一个分隔符列表，用于分割文本。在这里，它包括双换行符（`\n\n`）、单换行符（`\n`）、空格（）、以及空字符串（`""`）。这有助于分割器使用自然的文本边界和空白来进行分块。

    +   `chunk_size`：这是每个块的最大大小，设置为 200 个字符。这意味着每个块将*最多*包含 200 个字符。

    +   `chunk_overlap`：这是相邻块之间重叠的字符数，设置为 0。也就是说，块之间没有重叠。

    +   `length_function`：这是一个用于衡量文本长度的函数，设置为 `len`，它计算字符串中的字符数。

1.  将文本拆分成块：

    ```py
    chunks = text_splitter.split_text(text_block)
    ```

1.  打印这些块。在第一个块中，用户对智能手机的相机非常满意，赞扬了照片的清晰度和生动的色彩。然而，用户对笔记本电脑的性能感到失望，提到了频繁的卡顿问题：

    ```py
    Chunk 1:
    This smartphone has an excellent camera. The photos are sharp and the colors are vibrant. Overall, very satisfied with my purchase. I was disappointed with the laptop's performance. It frequently lags
    ```

    用户对搅拌机很满意，指出其在制作果昔方面的高效性、强大的功率和易于清洁的特点。他们认为其性价比很高：

    ```py
    Chunk 2:
    and the battery life is shorter than expected. The blender works great for making smoothies. It's powerful and easy to clean. Definitely worth the price. Customer support was unresponsive. I had to
    ```

    用户在与客户支持的互动中有不好的体验，提到了长时间等待和未解决的问题。用户认为这本书非常吸引人，情节引人入胜，人物刻画深入，他们强烈推荐给读者：

    ```py
    Chunk 3:
    wait a long time for a reply, and my issue was not resolved satisfactorily. The book is a fascinating read. The storyline is engaging and the characters are well-developed. Highly recommend to all
    ```

    我们剩下一个单词：

    ```py
    Chunk 4:
    Readers.
    ```

现在，这些块并不完美，但让我们了解 `RecursiveCharacterTextSplitter` 的工作原理，这样你就可以根据自己的使用场景进行调整：

+   **块大小目标**：分割器的目标是生成大约 200 个字符的块，但这只是一个最大值，而不是严格要求。它将尽量创建接近 200 个字符的块，但不会超过这个限制。

+   **递归方法**：递归性质意味着它会重复应用这些规则，通过分隔符列表逐步找到合适的分割点。

+   **保持语义意义**：通过使用这种方法，分割器尝试将语义相关的内容保持在一起。例如，它会尽量避免在段落或句子中间进行分割。

+   `chunk_overlap` 设置为 `0`，这意味着块之间没有内容重复，每个块都是独立的。

+   `len` 函数用于衡量块的大小，即它计算的是字符而不是词元。

`length_function` 参数

`RecursiveCharacterTextSplitter` 中的 `length_function` 参数是一个灵活的选项，允许你定义*如何衡量文本块的长度*。虽然 `len` 是默认且最常见的选择，但也有很多其他选项，从基于词元的到基于单词的，再到自定义实现。

递归切块法专注于根据固定大小和自然分隔符创建块，而语义切块法则更进一步，通过根据文本的意义和上下文对其进行分组。这种方法确保了块不仅在长度上连贯，而且在语义上具有意义，从而提高了后续自然语言处理任务的相关性和准确性。

### 实现语义切块

语义分块涉及根据语义意义而非仅仅是句法规则或固定长度来拆分文本。在幕后，使用*嵌入*将相关的句子聚集在一起（我们将在*第十三章*中深入探讨嵌入，章节标题为《图像与音频预处理与 LLMs》）。我们通常使用语义分块处理需要深度理解上下文的任务，例如问答系统和主题分析。让我们深入了解语义分块背后的过程：

1.  **文本输入**：过程从文本输入开始，可以是一个文档、一组句子或任何需要处理的文本数据。

1.  **嵌入生成**：文本的每个片段（通常是句子或小组句子（块））都被转换为高维向量表示，使用嵌入生成。这些嵌入是由预训练语言模型生成的，关键是要理解这些嵌入捕捉了文本的语义含义。换句话说，我们将文本转化为一个数值表示，*它编码了* *其意义*！

1.  **相似度测量**：然后将这些嵌入进行比较，以衡量文本不同部分之间的语义相似度。常用的技术如余弦相似度，用于量化不同片段之间的相关性。

1.  **聚类**：根据相似度评分，将句子或文本片段聚集在一起。聚类算法将语义相似的句子归为同一组，这样可以确保每个组内的内容保持语义一致性和上下文连贯性。

1.  **块创建**：聚类后的句子被组合成块。这些块被设计为语义上有意义的文本单元，可以更有效地被 NLP 模型处理。

让我们回到产品评论的示例，看看通过语义分块生成了什么样的块。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/9.semantic_chunking.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/9.semantic_chunking.py)找到代码：

1.  使用`HuggingFaceEmbeddings`初始化`SemanticChunker`：

    ```py
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    ```

1.  将文本拆分成块：

    ```py
    docs = text_splitter.create_documents([text_block])
    ```

1.  打印块：

    ```py
    Chunk 1:
    This smartphone has an excellent camera. The photos are sharp and the colors are vibrant. Overall, very satisfied with my purchase. I was disappointed with the laptop's performance. It frequently lags and the battery life is shorter than expected. The blender works great for making smoothies. It's powerful and easy to clean.
    Chunk 2:
    Definitely worth the price. Customer support was unresponsive. I had to wait a long time for a reply, and my issue was not resolved satisfactorily. The book is a fascinating read. The storyline is engaging and the characters are well-developed. Highly recommend to all readers.
    ```

每个块包含相关的句子，这些句子构成一个连贯的段落。例如，第一个块讨论了各种产品的性能，而第二个块则包括了客户支持体验和书评。这些块在每个段落内保持上下文一致，确保相关信息被组合在一起。需要改进的一点是，第一个块包含了不同产品（智能手机、笔记本电脑和搅拌机）的评价，而第二个块则将客户支持体验与书评混合，这可能被视为语义上不相关。在这种情况下，我们可以进一步将文本拆分成更小、更集中的块，以提高其连贯性，或/和调整语义切分器的参数。

```py
text_splitter = SemanticChunker(
    embeddings=embedding_model,
    buffer_size=200,
    add_start_index=True,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=0.9,
    number_of_chunks=4,
    sentence_split_regex=r'\.|\n|\s'
)
```

你可以在文档中找到更多关于这些参数的细节：

[`api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html`](https://api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)

然而，在我们这个案例中，改进切分的步骤可能是这样的：

+   使用不同的嵌入模型，看看哪一种为你的文本提供了最佳的嵌入。

+   调整缓冲区大小，以找到块大小和连贯性之间的最佳平衡

+   调整阈值类型和数量，以优化基于语义断点的块切分位置

+   自定义句子分割的正则表达式，以更好地适应文本的结构

从切分（chunking）到分词（tokenization）的过渡，意味着从一个将文本划分为更大、更具语法意义的段落（块）的过程，转向将文本划分为更小、更细粒度单元（词元）的过程。让我们来看一下**分词**是如何工作的。

# 分词

分词是将一段文本拆分成更小的单元或词元的过程，词元可以是单词、子词或字符。这个过程对于将文本转化为适合*计算处理*的格式至关重要，使得模型能够在更精细的粒度上学习模式。

在分词阶段，一些关键术语包括 `[CLS]` 用于分类，`[SEP]` 用于分隔等。词汇表中的每个词项都会被分配一个 ID，模型内部使用这个 ID 来表示该词项。这些 ID 是整数，通常范围从 0 到词汇表大小减一。

世界上所有的词汇都能放进一个词汇表里吗？答案是*不行*! OOV（Out-Of-Vocabulary）词是指模型词汇表中没有的词。

现在我们知道了常用的术语，让我们来探讨不同类型的分词以及与之相关的挑战。

## 单词分词

单词分词是将文本拆分为单个单词的过程。

例如，句子“Tokenization is crucial in NLP!”会被分词成`["Tokenization", "is", "crucial", "in", "NLP", "!"]`。

词语分词保留了完整的词语，这对于需要词语级理解的任务是有益的。它在词语边界明确的语言中效果良好。这是一种简单的解决方案，但可能导致 OOV 词汇的问题，特别是在医学文本和包含许多拼写错误的文本等专业领域中。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/10.word_tokenisation.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/10.word_tokenisation.py)找到完整的代码。

让我们看一个代码示例：

1.  下载必要的 NLTK 数据（只需运行一次）：

    ```py
    nltk.download('punkt')
    ```

1.  以以下文本作为示例：

    ```py
    text = "The quick brown fox jumps over the lazy dog. It's unaffordable!"
    ```

1.  执行词语分词：

    ```py
    word_tokens = word_tokenize(text)
    ```

1.  打印输出：

    ```py
    Tokens:
    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.', 'It', "'s", 'unaffordable', '!']
    ```

这种词语分词方法适用于处理简单、结构良好的文本，其中每个词语都通过空格和标点符号清晰分隔。它是一种简单的方法，与人类感知词语的方式非常契合。然而，相同词语的不同形式（例如，“run”、“running”、“ran”）被视为不同的标记，这可能会削弱模型的理解能力。它还可能导致词汇量过大，特别是在形态丰富或具有许多独特词汇的语言中。最后，这些模型在训练过程中没有出现过的词语会成为 OOV 标记。

鉴于词语分词的局限性，子词分词方法变得越来越流行。子词分词在词语级别分词和字符级别分词之间取得了平衡，解决了两者的许多不足之处。

## 子词分词

**子词分词**将文本拆分成比词语更小的单位，通常是子词。通过将词语拆分成已知的子词，它可以处理 OOV（未登录词）问题。它显著减少了词汇量和参数数量。接下来的部分将介绍子词分词的不同选项。

### 字节对编码（BPE）

BPE 从单个字符开始，迭代地合并最频繁的标记对以创建子词。它最初作为一种数据压缩算法开发，但已经被改编用于 NLP 任务中的分词。过程如下：

1.  从单个字符的词汇表开始。

1.  计算文本中所有字符对的频率。

1.  合并最频繁的字符对形成新的标记。

1.  重复该过程直到达到所需的词汇量。

这种基于频率的合并策略对于具有简单形态结构的语言（例如英语）或需要直接且稳健的分词时非常有用。由于基于频率的合并，它简单且计算高效。我们来演示一个如何实现 BPE 分词的例子。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/11.bpe_tokeniser.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/11.bpe_tokeniser.py)找到完整示例：

1.  加载预训练的分词器：

    ```py
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    ```

1.  加载示例文本：

    ```py
    text = "Tokenization in medical texts can include words like hyperlipidemia."
    ```

1.  对文本进行分词：

    ```py
    tokens = tokenizer.tokenize(text)
    ```

1.  将符号转换为输入 ID：

    ```py
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ```

1.  打印符号和 ID，如下所示：

    ```py
    Tokens: ['Token', 'ization', 'Ġin', 'Ġmedical', 'Ġtexts', 'Ġcan', 'Ġinclude', 'Ġwords', 'Ġlike', 'Ġhyper', 'lip', 'idem', 'ia', '.']
    Input IDs: [21920, 3666, 287, 1400, 1562, 649, 4551, 3545, 588, 20424, 3182, 1069, 257, 13]
    ```

分词输出中的特殊字符“Ġ”在 BPE 分词的上下文中具有特定含义。它表示紧随其后的符号原本前面有空格或位于文本的开头，因此它允许保留有关原始文本中单词边界和空格的信息。

让我们解释一下我们看到的输出：

+   `Token` 和 `ization`：这些是“Tokenization”的子词，分割时没有“Ġ”，因为*它们是同一个单词*的一部分。

+   `in`、`medical`、`texts`等：这些符号以**Ġ**开头，表示它们在原文中是*独立*的单词。

+   `hyper`、`lip`、`id` 和 `emia`：这些是`hyperlipidemia`（高脂血症）的子词。`hyper`表示这是一个新词，而后续的子词没有`hyperlipidemia`被分解成 `hyper`（表示*过量*的前缀）、`lip`（与脂肪相关）、`id`（连接元素）和 `emia`（表示*血液状况*的后缀）。

在探讨了 BPE 分词及其对文本处理的影响后，我们现在将注意力转向 WordPiece 分词，这是一种进一步优化 NLP 任务中子词单元处理的强大方法。

### WordPiece 分词

WordPiece，BERT 使用的分词方法，从一个字符的基本词汇表开始，迭代地添加最频繁的子词单元。过程如下所示：

1.  从单个字符的基本词汇表和一个用于未知词汇的特殊符号开始。

1.  迭代地合并最频繁的符号对（从字符开始）以形成新符号，直到达到预定义的词汇表大小。

1.  对于任何给定的词，使用词汇表中最长的匹配子词单元。这个过程称为**最大匹配**。

WordPiece 分词对于结构复杂的语言（例如韩语和日语）和高效处理多样化词汇至关重要。其有效性源于根据最大化可能性选择合并，因此可能会生成更有意义的子词。然而，一切都有代价，在这种情况下，由于可能性最大化步骤，它的计算密集性更高。让我们看一个使用 BERT 分词器执行 WordPiece 分词的代码示例。您可以在 [`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/12.tokenisation_wordpiece.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/12.tokenisation_wordpiece.py) 找到完整的代码。

1.  加载预训练分词器：

    ```py
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ```

1.  取一些样本文本：

    ```py
    text = "Tokenization in medical texts can include words like hyperlipidemia."
    ```

1.  对文本进行分词。该方法将输入文本分割成 WordPiece 标记。例如，`unaffordable` 被分解为 `un`, `##afford`, `##able`：

    ```py
    tokens = tokenizer.tokenize(text)
    ```

1.  将标记转换为输入 ID：

    ```py
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    Tokens:
    ['token', '##ization', 'in', 'medical', 'texts', 'can', 'include', 'words', 'like', 'hyper', '##lip', '##idem', '##ia']
    Input IDs:
    [19204, 10859, 1999, 2966, 4524, 2064, 2421, 2540, 2066, 15088, 17750, 28285, 3676]
    ```

`##` 前缀用于表示该标记是前一个标记的延续。因此，它有助于通过指示应将标记附加到前一个标记而不加空格来重构原始单词。

在审查了诸如 BPE 和 WordPiece 等分词方法之后，关键是考虑如何调整分词器以处理专门领域的数据，例如医学文本，以确保在这些特定领域中进行精确和上下文相关的处理。

## 领域特定数据

当处理诸如医学文本之类的领域特定数据时，确保分词器能够有效处理专门词汇至关重要。当领域具有高频率的独特术语或专门词汇时，标准分词器可能无法达到最佳性能。在这种情况下，领域特定分词器可以更好地捕捉领域的细微差别和术语，从而提高模型性能。当面临这一挑战时，有一些可选方案：

+   在领域特定文本语料库上训练一个分词器，创建包含专门术语的词汇表

+   考虑通过领域特定标记扩展现有的分词器，而不是从头开始训练

然而，如何确定您需要更进一步地调整数据集上的分词器呢？让我们找找答案。

### 评估是否需要专门的分词器

正如我们所解释的，当处理诸如医学文本之类的领域特定数据时，评估是否需要专门的分词器是至关重要的。让我们看看几个考虑的关键因素：

+   **分析 OOV 率**：确定您领域特定语料库中不包含在标准分词器词汇表中的单词的百分比。高 OOV 率表明您领域中许多重要术语未被识别，突显出需要专门的分词器来更好地处理独特的词汇。

+   **检查分词质量**：通过手动检查样本分词，查看标准分词器如何分割领域特定术语。如果关键术语（如医学术语）经常被分解为无意义的子词，这表明分词器不适合该领域，可能需要定制化。

+   **压缩比**：使用标准和领域特定分词器测量每个句子的平均词汇数量。领域特定分词器显示出显著较低的比率，表明它在压缩和表示领域知识方面更为高效，减少冗余，提高性能。

例如，在医学语料库中，术语如 *心肌梗死* 可能被标准分词器分成 `myo`、`cardial` 和 `infarction`，导致意义丧失。然而，专门的医学分词器可能将 *心肌梗死* 识别为一个单独的术语，保留其含义并提升下游任务如实体识别和文本生成的质量。类似地，如果标准分词器的 OOV 率为 15%，而专门分词器仅为 3%，这清楚地表明需要定制化的需求。最后，如果使用标准分词器的压缩比为每句话 1.8 个词汇，而专门分词器为 1.2 个词汇，则表明专门分词器在捕捉领域特定细微差别方面更为高效。

让我们实现一个小应用程序，以评估不同医疗数据的分词器。示例的代码可在 [`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/13.specialised_tokenisers.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/13.specialised_tokenisers.py) 上找到：

1.  初始化用于生物医学文本的 Stanza：

    ```py
    stanza.download('en', package='mimic', processors='tokenize')
    nlp = stanza.Pipeline('en', package='mimic', processors='tokenize')
    ```

1.  初始化标准 GPT-2 分词器：

    ```py
    standard_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ```

1.  将 `pad_token` 设置为 `eos_token`：

    ```py
    standard_tokenizer.pad_token = standard_tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    ```

1.  为模型设置 `pad_token_id`：

    ```py
    model.config.pad_token_id = model.config.eos_token_id
    ```

1.  定义一个包含与心肌梗死和心脏病相关的句子的样本医学语料库：

    ```py
    corpus = [
      "The patient suffered a myocardial infarction.",
      "Early detection of heart attack is crucial.",
      "Treatment for myocardial infarction includes medication.",
      "Patients with heart conditions require regular check-ups.",
      "Myocardial infarction can lead to severe complications."
    ]
    ```

1.  下面的 `stanza_tokenize` 函数使用 Stanza 流水线对文本进行分词并返回一个词汇列表：

    ```py
    def stanza_tokenize(text):
        doc = nlp(text)
        tokens = [word.text for sent in doc.sentences for word in sent.words]
        return tokens
    ```

1.  `calculate_oov_and_compression` 函数对语料库中的每个句子进行分词并计算 OOV 率以及平均每句话的词汇数，并返回所有的词汇。对于标准分词器，它检查词汇表中是否存在这些标记，而对于 Stanza，则不会显式检查 OOV 标记：

    ```py
    def calculate_oov_and_compression(corpus, tokenizer):
        oov_count = 0
        total_tokens = 0
        all_tokens = []
        for sentence in corpus:
            tokens = tokenizer.tokenize(sentence) if hasattr(tokenizer, 'tokenize') else stanza_tokenize(sentence)
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
            oov_count += tokens.count(tokenizer.oov_token) if hasattr(tokenizer, 'oov_token') else 0
        oov_rate = (oov_count / total_tokens) * 100 if total_tokens > 0 else 0
        avg_tokens_per_sentence = total_tokens / len(corpus)
    return oov_rate, avg_tokens_per_sentence, all_tokens
    ```

1.  `analyze_token_utilization`函数计算语料库中每个标记的频率，并返回一个标记利用率百分比的字典：

    ```py
    def analyze_token_utilization(tokens):
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        utilization = {token: count / total_tokens for token, count in token_counts.items()}
        return utilization
    ```

1.  `calculate_perplexity`函数计算给定文本的困惑度，这是衡量模型预测样本能力的指标：

    ```py
    def calculate_perplexity(tokenizer, model, text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    ```

1.  以下脚本通过计算 OOV 率、每句平均标记数、标记利用率和困惑度，评估了标准 GPT-2 分词器和 Stanza 医学分词器的性能。最后，它打印每个分词器的结果并比较它们在`myocardial` `infarction`术语上的表现：

    ```py
    for tokenizer_name, tokenizer in [("Standard GPT-2", standard_tokenizer), ("Stanza Medical", stanza_tokenize)]:
        oov_rate, avg_tokens, all_tokens = calculate_oov_and_compression(corpus, tokenizer)
        utilization = analyze_token_utilization(all_tokens)
        print(f"\n{tokenizer_name} Tokenizer:")
        print(f"OOV Rate: {oov_rate:.2f}%")
        print(f"Average Tokens per Sentence: {avg_tokens:.2f}")
        print("Top 5 Most Used Tokens:")
        for token, freq in sorted(utilization.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f" {token}: {freq:.2%}")
    ```

我们来看一下下表中展示的两个分词器的结果：

| **指标** | **标准** **GPT-2 分词器** | **Stanza** **医学分词器** |
| --- | --- | --- |
| **OOV 率** | 0.00% | 0.00% |
| **每句平均标记数** | 10.80 | 7.60 |
| **最常用的五个** **标记** | . : 9.26% | . : 13.16% |
| ocard : 5.56% | infarction : 7.89% |
| ial : 5.56% | myocardial : 5.26% |
| Ġinf : 5.56% | heart : 5.26% |
| ar : 5.56% | The : 2.63% |

表 12.1 – GPT-2 分词器与专用医学分词器的比较

如表中所示，两个分词器的 OOV 率均为 0.00%，这意味着语料库中的所有标记都被两个分词器识别。Stanza Medical 分词器的每句平均标记数（7.60）低于标准 GPT-2 分词器（10.80）。这表明 Stanza Medical 分词器在将领域特定术语压缩成更少的标记方面更为高效。标准 GPT-2 分词器将有意义的医学术语拆分成更小的子词，从而导致标记利用效率较低（例如，`ocard`，`ial`，`inf`，和`ar`）。然而，Stanza Medical 分词器保持了医学术语的完整性（例如，`infarction`和`myocardial`），使标记更加有意义且与上下文相关。基于分析，Stanza Medical 分词器应该更适用于医学文本处理，原因如下：

+   它将领域特定术语高效地分词成更少的标记

+   它保持医学术语的完整性和意义

+   它提供了更多有意义且与上下文相关的标记，这对于医学领域中的实体识别和文本生成等任务至关重要

标准 GPT-2 分词器在处理一般文本时很有用，但它将医学术语拆分成子词，这可能导致上下文和意义的丧失，因此不太适合专门的医学文本。

词汇大小权衡

更大的词汇表可以捕捉到更多的领域特定术语，但会增加模型的大小和计算需求。找到一个平衡点，既能充分覆盖领域术语，又不至于过度膨胀。

在评估了不同的分词方法的表现后，包括它们如何处理 OOV 词汇以及在压缩领域特定知识时的效率，下一步的逻辑是探讨这些分词输出是如何通过嵌入技术转化为有意义的数值表示的。这一过渡非常关键，因为嵌入构成了模型理解和处理分词文本的基础。

# 将 tokens 转换为嵌入

嵌入是词语、短语或整个文档在高维向量空间中的数值表示。实质上，我们将词语表示为数字数组，以捕捉它们的语义意义。这些数值数组旨在编码词语和句子的潜在意义，使得模型能够以有意义的方式理解和处理文本。让我们从分词到嵌入的过程进行探索。

该过程从分词开始，将文本拆分为可管理的单位，称为 tokens。例如，句子“The cat sat on the mat”可能被分词为单个词或子词单元，如 [“The”, `cat`, “sat”, “on”, “the”, “mat”]。一旦文本被分词，每个 token 会通过嵌入层或查找表映射到一个嵌入向量。这个查找表通常会用随机值初始化，然后进行训练，以捕捉词语之间的有意义关系。例如，`cat` 可能被表示为一个 300 维的向量。

像 BERT 或 GPT 这样的高级模型会生成上下文化的嵌入，其中一个词的向量表示会受到其周围词语的影响。这使得模型能够理解细微差别和上下文，比如区分“bank”在“river bank”和“financial bank”中的不同含义。

让我们更详细地了解这些模型。

## BERT – 上下文化嵌入模型

BERT 是谷歌开发的强大 NLP 模型。它属于基于 transformer 的模型家族，并且在大量文本数据上进行预训练，以学习词语的上下文化表示。

BERT 嵌入模型是 BERT 架构的一个*组件*，用于生成上下文化的词嵌入。与传统的词嵌入不同，传统词嵌入为每个词分配一个固定的向量表示，而 BERT 嵌入是上下文相关的，能够捕捉词语在整个句子中的意义。以下是如何使用 BERT 嵌入模型的解释 [`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/14.embedding_bert.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/14.embedding_bert.py)：

1.  加载预训练的 BERT 模型和分词器：

    ```py
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    ```

1.  编码输入文本：

    ```py
    input_text = "BERT embeddings capture contextual information."
    inputs= tokenizer(input_text, return_tensors="pt")
    ```

1.  获取 BERT 嵌入：

    ```py
    with torch.no_grad():
        outputs = model(inputs)
    ```

1.  打印嵌入：

    ```py
    print("Shape of the embeddings tensor:", last_hidden_states.shape)
    Shape of the embeddings tensor: torch.Size([1, 14, 768])
    ```

嵌入张量的形状将是（`1`，`sequence_length`，`hidden_size`）。其中，`sequence_length`是输入句子中的标记数，`hidden_size`是 BERT 模型中的隐藏状态大小（`bert-base-uncased`为`768`）。

`[CLS]` 标记嵌入表示整个输入句子，通常用于分类任务。它是输出张量中的第一个标记：

```py
CLS token embedding: [ 0.23148441 -0.32737488 ...  0.02315655]
```

句子中第一个实际单词的嵌入表示该特定单词的上下文化嵌入。句子中第一个实际单词的嵌入不仅仅是该单词的静态或孤立表示。相反，它是一个“上下文感知”或“上下文化”的嵌入，意味着它反映了该单词的含义如何受到句子中周围单词的影响。简单来说，这个嵌入不仅捕捉了单词的内在含义，还反映了基于周围单词提供的上下文，单词含义如何变化。这是像 BERT 这样的模型的一个关键特性，它根据单词在不同上下文中的使用，生成不同的嵌入。

```py
First word embedding: [ 0.00773875  0.24699381 ... -0.09120814]
```

这里需要理解的关键点是，我们从文本开始，输出是向量或嵌入。使用`transformers`库提供的分词器时，分词步骤是在幕后进行的。分词器将输入句子转换为标记及其对应的标记 ID，然后传递给 BERT 模型。请记住，句子中的每个单词都有自己的嵌入，反映了该单词在句子上下文中的含义。

BERT 的多功能性使其在各种 NLP 任务中表现出色。然而，随着对更高效和任务特定的嵌入需求的增加，出现了像**BAAI 通用嵌入**（**BGE**）这样的模型。BGE 设计为更小、更快，提供高质量的嵌入，优化了语义相似性和信息检索等任务。

## BGE

BAAI/bge-small-en 模型是**北京人工智能研究院**（**BAAI**）开发的一系列 BGE 模型的一部分。这些模型旨在生成高质量的文本嵌入，通常用于文本分类、语义搜索等各种 NLP 任务。

这些模型为文本生成嵌入（向量表示）。嵌入捕捉文本的语义含义，使其在诸如相似性搜索、聚类和分类等任务中非常有用。`bge-small-en`模型是该系列中的一个较小的、专为英语设计的模型。我们来看一个例子。此示例的完整代码可在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/15.embedding_bge.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/15.embedding_bge.py)查看：

1.  定义模型名称和参数：

    ```py
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    ```

1.  初始化嵌入模型：

    ```py
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    ```

1.  我们随机选取几句话进行嵌入：

    ```py
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I love machine learning and natural language processing."
    ]
    ```

1.  为每个句子生成嵌入：

    ```py
    embeddings = [bge_embeddings.embed_query(sentence) for sentence in sentences]
    ```

1.  打印嵌入：

    ```py
    [-0.07455343008041382, -0.004580824635922909, 0.021685084328055382, 0.06458176672458649, 0.020278634503483772]...
    Length of embedding: 384
    Embedding for sentence 2: [-0.025911744683980942, 0.0050039878115057945, -0.011821565218269825, -0.020849423483014107, 0.06114110350608826]...
    ```

像 bge-small-en 这样的 BGE 模型，设计上比 BERT 等较大、通用的模型更小、更高效，适合用于嵌入生成任务。这种高效性转化为更低的内存使用和更快的推理时间，使得 BGE 模型特别适合计算资源有限或实时处理至关重要的应用。尽管 BERT 是一个多功能的通用模型，能够处理广泛的 NLP 任务，但 BGE 模型特别针对生成高质量的嵌入进行了优化。这种优化使得 BGE 模型能够在特定任务中提供可比或甚至更优的性能，如语义搜索和信息检索，在这些任务中，嵌入的质量至关重要。通过专注于嵌入的精确度和语义丰富性，BGE 模型利用了先进的技术，如学习稀疏嵌入，结合了稠密和稀疏表示的优势。这种有针对性的优化使得 BGE 模型在需要细致文本表示和高效处理的场景中表现出色，相比更通用的 BERT 模型，它们在以嵌入为核心的应用中是更好的选择。

在 BERT 和 BGE 成功的基础上，**通用文本嵌入**（**GTEs**）的引入标志着又一步重要的进展。GTE 模型专门针对各种文本相关应用进行了精细调优，以提供强大且高效的嵌入。

## GTE

GTE 代表了下一代嵌入模型，旨在应对对专业化和高效文本表示日益增长的需求。GTE 模型在为特定任务（如语义相似性、聚类和信息检索）提供高质量嵌入方面表现出色。让我们看看它们的实际应用。完整的代码可在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/16.embedding_gte.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter12/16.embedding_gte.py)找到：

1.  加载 GTE-base 模型：

    ```py
    model = SentenceTransformer('thenlper/gte-base')
    ```

1.  我们随机选取一些文本进行嵌入：

    ```py
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I love machine learning and natural language processing.",
        "Embeddings are useful for many NLP tasks."
    ]
    ```

1.  生成嵌入：

    ```py
    embeddings = model.encode(texts
    ```

1.  打印嵌入的形状：

    ```py
    print(f"Shape of embeddings: {embeddings.shape}")
    Shape of embeddings: (3, 768)
    ```

1.  打印第一个嵌入的前几个值：

    ```py
    [-0.02376037 -0.04635307  0.02570779  0.01606994  0.05594607]
    ```

GTE 的一大亮点是其高效性。通过保持较小的模型尺寸和更快的推理时间，GTE 非常适合实时应用和计算资源受限的环境。这种高效性并不以牺牲性能为代价；GTE 模型在各种文本处理任务中依然能交付出色的结果。然而，它们减少的复杂性处理可能成为一个限制，因为较小的模型尺寸可能妨碍它们有效处理高度复杂或细致的文本。这可能导致对微妙上下文细节的捕捉不够准确，从而影响更复杂场景下的表现。此外，GTE 对高效性的专注可能导致其泛化能力下降；尽管在特定任务中表现优秀，但它可能在适应多样化或不常见的语言输入时遇到困难。而且，模型较小的尺寸可能限制其微调的灵活性，由于学习和存储特定领域复杂模式的能力较弱，可能无法很好地适应专业化任务或领域。

## 选择合适的嵌入模型

在为你的应用选择模型时，首先确定你的具体使用场景和领域。你是否需要一个用于分类、聚类、检索或摘要的模型，且你的领域是法律、医学还是通用文本，将显著影响你的选择。

接下来，评估模型的*大小和内存使用情况*。较大的模型通常能提供更好的性能，但也伴随着更高的计算要求和较高的延迟。在初期原型开发阶段，可以选择较小的模型，随着需求的发展，再考虑过渡到更大的模型。注意嵌入维度，因为更大的维度能够提供更丰富的数据表示，但也更具计算强度。在捕捉详细信息与保持操作效率之间找到平衡非常重要。

仔细评估*推理时间*，特别是如果你有实时应用需求；延迟较高的模型可能需要 GPU 加速才能达到性能标准。最后，使用像**Massive Text Embedding Benchmark**（**MTEB**）这样的基准测试来评估模型的性能，以便在不同的度量标准之间进行比较。考虑到内在评估，它考察模型对语义和句法关系的理解，以及外在评估，它则评估模型在特定下游任务上的表现。

## 利用嵌入解决实际问题

随着 BERT、BGE 和 GTE 等嵌入模型的进展，我们可以应对各个领域的广泛挑战。这些模型使我们能够解决不同的问题，具体如下：

+   **语义搜索**：嵌入通过捕捉查询和文档的上下文含义来提高搜索相关性，从而提升搜索结果的准确性。

+   **推荐系统**：它们根据用户的偏好和行为，提供个性化的内容推荐，量身定制推荐内容以满足个人需求。

+   **文本分类**：嵌入使得文档能够准确地归类到预定义的类别中，例如情感分析或主题识别。

+   **信息检索**：它们提高了从庞大数据集中检索相关文档的准确性，提升了信息检索系统的效率。

+   **自然语言理解**：嵌入支持如命名实体识别（NER）等任务，帮助系统识别和分类文本中的关键实体。

+   **聚类技术**：它们能够改善大型数据集中相似文档或主题的组织结构，帮助实现更好的聚类和数据管理。

+   **多模态数据处理**：嵌入对于整合和分析文本、图像和音频数据至关重要，有助于提供更全面的洞察和增强的决策能力。

让我们总结一下本章的学习内容。

# 总结

在本章中，我们回顾了文本预处理，这是自然语言处理中的一个关键步骤。我们介绍了不同的文本清理技术，从处理 HTML 标签和大小写到解决数字值和空格问题。我们深入探讨了分词，分析了词汇分词和子词分词，并提供了实用的 Python 示例。最后，我们介绍了多种文档嵌入方法，并介绍了当前最受欢迎的嵌入模型。

在下一章中，我们将继续探索非结构化数据，深入研究图像和音频的预处理。
