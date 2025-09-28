# 附录 A. Penn Treebank 词性标签

以下是一个表格，列出了与 NLTK 一起分发的 `treebank` 语料库中出现的所有词性标签。这里显示的标签和计数是通过以下代码获得的：

```py
>>> from nltk.probability import FreqDist
>>> from nltk.corpus import treebank
>>> fd = FreqDist()
>>> for word, tag in treebank.tagged_words():
...   fd.inc(tag)
>>> fd.items()
```

`FreqDist fd` 包含了 `treebank` 语料库中每个标签的所有计数。您可以通过执行 `fd[tag]` 来单独检查每个标签的计数，例如 `fd['DT']`。标点符号标签也显示出来，以及特殊标签如 `-NONE-`，它表示词性标签是未知的。大多数标签的描述可以在 [`www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html`](http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) 找到。

| 词性标签 | 出现频率 |
| --- | --- |
| # | 16 |
| $ | 724 |
| '' | 694 |
| , | 4,886 |
| -LRB- | 120 |
| -NONE- | 6,592 |
| -RRB- | 126 |
| . | 384 |
| : | 563 |
| `` | 712 |
| CC | 2,265 |
| CD | 3,546 |
| DT | 8,165 |
| EX | 88 |
| FW | 4 |
| IN | 9,857 |
| JJ | 5,834 |
| JJR | 381 |
| JJS | 182 |
| LS | 13 |
| MD | 927 |
| NN | 13,166 |
| NNP | 9,410 |
| NNPS | 244 |
| NNS | 6,047 |
| PDT | 27 |
| POS | 824 |
| PRP | 1,716 |
| PRP$ | 766 |
| RB | 2,822 |
| RBR | 136 |
| RBS | 35 |
| RP | 216 |
| SYM | 1 |
| TO | 2,179 |
| UH | 3 |
| VB | 2,554 |
| VBD | 3,043 |
| VBG | 1,460 |
| VBN | 2,134 |
| VBP | 1,321 |
| VBZ | 2,125 |
| WDT | 445 |
| WP | 241 |
| WP$ | 14 |
