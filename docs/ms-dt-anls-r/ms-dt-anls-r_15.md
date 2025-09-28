# 附录 A. 参考文献

虽然互联网上关于 R 和数据科学有很多优秀且免费的资源（例如 StackOverflow、GitHub 维基、[`www.r-bloggers.com/`](http://www.r-bloggers.com/)和一些免费电子书），但有时购买一本结构化的书籍会更好——就像你做的那样。

在这个附录中，我列出了一些我在学习 R 时发现有用的书籍和其他参考资料。如果你希望成为一个拥有良好 R 背景的专业数据科学家，并且不喜欢自学的方式，我建议你至少浏览一下这些材料。

为了确保可重复性，本书中使用的所有 R 包都列出了实际包版本和安装来源。

# R 的通用阅读材料

虽然即将到来的列表与本书的不同章节相关，但以下是一些关于 R 入门和高级主题的优秀资源列表：

+   *《快速 R》*，作者*Robert I. Kabacoff*，可在[`www.statmethods.net`](http://www.statmethods.net)找到

+   官方 R 手册可在[`cran.r-project.org/manuals.html`](https://cran.r-project.org/manuals.html)找到

+   *《R“元”书》*，作者*Joseph Ricker*，可在[`blog.revolutionanalytics.com/2014/03/an-r-meta-book.html`](http://blog.revolutionanalytics.com/2014/03/an-r-meta-book.html)找到

+   *《R 入门》*，*Wiley*，*2012*，作者*Andrie de Vries*和*Joris Meys*

+   *《R 实战》*，*Manning*，*2015*，作者*Robert I. Kabacoff*

+   *《R 速成》*，*O'Reilly*，*2010*，作者*Joseph Adler*

+   *《R 编程艺术》*，*2011*，作者*Norman Matloff*

+   *《R 地狱》*，作者*Partrick Burns*，可在[`www.burns-stat.com/documents/books/the-r-inferno/`](http://www.burns-stat.com/documents/books/the-r-inferno/)找到

+   *《高级 R》*，作者*Hadley Wickham*，*2015*，可在[`adv-r.had.co.nz`](http://adv-r.had.co.nz)找到

# 第一章：– 嗨，数据！

本章提到的 R 包版本（按顺序列出）：

+   hflights 0.1 (CRAN)

+   microbenchmark 1.4-2 (CRAN)

+   R.utils 2.0.2 (CRAN)

+   sqldf 0.4-10 (CRAN)

+   ff 2.2-13 (CRAN)

+   bigmemory 4.4.6 (CRAN)

+   data.table 1.9.4 (CRAN)

+   RMySQL 0.10.3 (CRAN)

+   RPostgreSQL 0.4 (CRAN)

+   ROracle 1.1-12 (CRAN)

+   dbConnect 1.0 (CRAN)

+   XLConnect 0.2-11 (CRAN)

+   xlsx 0.5.7 (CRAN)

相关 R 包：

+   mongolite 0.4 (CRAN)

+   MonetDB.R 0.9.7 (CRAN)

+   RcppRedis 0.1.5 (CRAN)

+   RCassandra 0.1-3 (CRAN)

+   RSQLite 1.0.0 (CRAN)

相关阅读：

+   R 数据导入/导出手册可在[`cran.r-project.org/doc/manuals/r-release/R-data.html`](https://cran.r-project.org/doc/manuals/r-release/R-data.html)找到

+   使用 R 进行高性能和并行计算，CRAN 任务视图可在[`cran.r-project.org/web/views/HighPerformanceComputing.html`](http://cran.r-project.org/web/views/HighPerformanceComputing.html)找到

+   Hadley Wickham 的关于数据库的 dplyr 小节可在[`cran.r-project.org/web/packages/dplyr/vignettes/databases.html`](https://cran.r-project.org/web/packages/dplyr/vignettes/databases.html)找到

+   RODBC vignette：[`cran.r-project.org/web/packages/RODBC/vignettes/RODBC.pdf`](https://cran.r-project.org/web/packages/RODBC/vignettes/RODBC.pdf)

+   Docker 文档：[`docs.docker.com`](http://docs.docker.com)

+   VirtualBox 手册：[`www.virtualbox.org/manual`](http://www.virtualbox.org/manual)

+   MySQL 下载地址：[`dev.mysql.com/downloads/mysql`](https://dev.mysql.com/downloads/mysql)

# 第二章：– 从网络获取数据

加载的 R 包版本（按章节中提到的顺序）：

+   RCurl 1.95-4.1 (CRAN)

+   rjson 0.2.13 (CRAN)

+   plyr 1.8.1 (CRAN)

+   XML 3.98-1.1 (CRAN)

+   wordcloud 2.4 (CRAN)

+   RSocrata 1.4 (CRAN)

+   quantmod 0.4 (CRAN)

+   Quandl 2.3.2 (CRAN)

+   devtools 1.5 (CRAN)

+   GTrendsR（BitBucket @ d507023f81b17621144a2bf2002b845ffb00ed6d）

+   weatherData 0.4 (CRAN)

相关 R 包：

+   jsonlite 0.9.16 (CRAN)

+   curl 0.6 (CRAN)

+   bitops 1.0-6 (CRAN)

+   xts 0.9-7 (CRAN)

+   RJSONIO 1.2-0.2 (CRAN)

+   RGoogleDocs 0.7 (OmegaHat.org)

相关阅读：

+   Chrome Devtools 手册：[`developer.chrome.com/devtools`](https://developer.chrome.com/devtools)

+   Chrome 开发者工具课程：[`discover-devtools.codeschool.com/`](http://discover-devtools.codeschool.com/)

+   XPath 在 Mozilla 开发者网络上的文档：[`developer.mozilla.org/en-US/docs/Web/XPath`](https://developer.mozilla.org/en-US/docs/Web/XPath)

+   Firefox 开发者工具：[`developer.mozilla.org/en-US/docs/Tools`](https://developer.mozilla.org/en-US/docs/Tools)

+   Firefox 的 Firebug：[`getfirebug.com/`](http://getfirebug.com/)

+   *使用 R 进行数据科学中的 XML 和 Web 技术*，由 *Deborah Nolan*、*Duncan Temple Lang（2014）* 编写，Springer 出版

+   *jsonlite 包：JSON 数据与 R 对象之间实用且一致的映射*，由 *Jeroen Ooms*（2014）编写，可在 [`arxiv.org/abs/1403.2805`](http://arxiv.org/abs/1403.2805) 查找

+   *Web 技术和服务 CRAN 任务视图*，由 *Scott Chamberlain*、*Karthik Ram*、*Christopher Gandrud* 和 *Patrick Mair*（2014）编写，可在 [`cran.r-project.org/web/views/WebTechnologies.html`](http://cran.r-project.org/web/views/WebTechnologies.html) 查找

# 第三章：– 过滤和汇总数据

加载的 R 包版本（按章节中提到的顺序）：

+   sqldf 0.4-10 (CRAN)

+   hflights 0.1 (CRAN)

+   dplyr 0.4.1 (CRAN)

+   data.table 1.9.4\. (CRAN)

+   plyr 1.8.2 (CRAN)

+   microbenchmark 1.4-2 (CRAN)

进一步阅读：

+   data.table 手册、vignette 和其他文档：[`github.com/Rdatatable/data.table/wiki/Getting-started`](https://github.com/Rdatatable/data.table/wiki/Getting-started)

+   *dplyr 简介*，*vignette*：[`cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html`](https://cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html)

# 第四章：– 数据重构

加载的 R 包版本（按章节中提到的顺序）：

+   hflights 0.1 (CRAN)

+   dplyr 0.4.1 (CRAN)

+   data.table 1.9.4\. (CRAN)

+   pryr 0.1 (CRAN)

+   reshape 1.4.2 (CRAN)

+   ggplot2 1.0.1 (CRAN)

+   tidyr 0.2.0 (CRAN)

进一步的 R 包：

+   jsonlite 0.9.16 (CRAN)

进一步阅读：

+   *data.table 简介*，*useR! 2014 会议的教程幻灯片*，由*Matt Dowle*于 2014 年提供，在 [`user2014.stat.ucla.edu/files/tutorial_Matt.pdf`](http://user2014.stat.ucla.edu/files/tutorial_Matt.pdf)

+   *使用 reshape 包重塑数据*，*Hadley Wickham*，*2006*，在 [`had.co.nz/reshape/introduction.pdf`](http://had.co.nz/reshape/introduction.pdf)

+   *探索数据和模型的实际工具*，由*Hadley Wickham*，*2008*，在 [`had.co.nz/thesis/`](http://had.co.nz/thesis/)

+   *双表动词*，*包示例*，在 [`cran.r-project.org/web/packages/dplyr/vignettes/two-table.html`](https://cran.r-project.org/web/packages/dplyr/vignettes/two-table.html)

+   *dplyr 简介*，*包示例*，[`cran.r-project.org/web/packages/dplyr/vignettes/introduction.html`](http://cran.r-project.org/web/packages/dplyr/vignettes/introduction.html)

+   *数据整理速查表*，*RStudio*，*2015* [`www.rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf`](https://www.rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf)

+   *使用 dplyr 进行数据操作*，*useR! 2014 会议的教程幻灯片和材料*，由*Hadley Wickham*提供，在 [`bit.ly/dplyr-tutorial`](http://bit.ly/dplyr-tutorial)

+   *整洁数据*，*统计软件杂志. 59(10): 1:23*，由*Hadley Wickham*，*2014*，在 [`vita.had.co.nz/papers/tidy-data.html`](http://vita.had.co.nz/papers/tidy-data.html)

# 第五章：– 构建模型（由 Renata Nemeth 和 Gergely Toth 撰写）

加载的 R 包版本（按章节中提到的顺序）：

+   gamlss.data 4.2-7 (CRAN)

+   scatterplot3d 0.3-35 (CRAN)

+   Hmisc 3.16-0 (CRAN)

+   ggplot2 1.0.1 (CRAN)

+   gridExtra 0.9.1 (CRAN)

+   gvlma 1.0.0.2 (CRAN)

+   partykit 1.0-1 (CRAN)

+   rpart 4.1-9 (CRAN)

进一步阅读：

+   *应用回归分析和其他多变量方法*，*Duxbury Press*，*2008*，由*David G. Kleinbaum*，*Lawrence L. Kupper*，*Azhar Nizam*，*Keith E. Muller*编写

+   *R 应用回归分析指南*，*Sage*，*网络指南*，由*John Fox*于*2011*年提供，在 [`socserv.socsci.mcmaster.ca/jfox/Books/Companion/appendix.html`](http://socserv.socsci.mcmaster.ca/jfox/Books/Companion/appendix.html)

+   *使用 R 进行实用回归和 ANOVA*，*Julian J*，*Faraway*，*2002*，在 [`cran.r-project.org/doc/contrib/Faraway-PRA.pdf`](https://cran.r-project.org/doc/contrib/Faraway-PRA.pdf)

+   *使用 R 进行线性模型*，*Julian J*，*CRC*，*Faraway*，*2014*，在 [`www.maths.bath.ac.uk/~jjf23/LMR/`](http://www.maths.bath.ac.uk/~jjf23/LMR/)

# 第六章：– 超越线性趋势线（由 Renata Nemeth 和 Gergely Toth 撰写）

加载的 R 包版本（按章节中提到的顺序）：

+   catdata 1.2.1 (CRAN)

+   vcdExtra 0.6.8 (CRAN)

+   lmtest 0.9-33 (CRAN)

+   BaylorEdPsych 0.5 (CRAN)

+   ggplot2 1.0.1 (CRAN)

+   MASS 7.3-40 (CRAN)

+   broom 0.3.7 (CRAN)

+   data.table 1.9.4\. (CRAN)

+   plyr 1.8.2 (CRAN)

进一步的 R 包：

+   LogisticDx 0.2 (CRAN)

进一步阅读：

+   *应用回归分析和其他多元方法. Duxbury Press*, *David G. Kleinbaum*, *Lawrence L. Kupper*, *Azhar Nizam*, 和 *Keith E. Muller* (2008)

+   *R 应用的回归分析伴侣*, *Sage*, *Web 补充* 由 *John Fox* (2011) 提供，[`socserv.socsci.mcmaster.ca/jfox/Books/Companion/appendix.html`](http://socserv.socsci.mcmaster.ca/jfox/Books/Companion/appendix.html)

# 第七章：– 非结构化数据

加载的 R 包版本（按章节中提到的顺序）：

+   tm 0.6-1 (CRAN)

+   wordcloud 2.5 (CRAN)

+   SnowballC 0.5.1 (CRAN)

进一步的 R 包：

+   coreNLP 0.4-1 (CRAN)

+   topicmodels 0.2-2 (CRAN)

+   textcat 1.0-3 (CRAN)

进一步阅读：

+   *Christopher D. Manning*, *Hinrich Schütze* (1999): *统计自然语言处理基础*. *MIT*.

+   *Daniel Jurafsky*, *James H. Martin* (2009): *语音与语言处理. 前沿出版社*.

+   *Christopher D. Manning*, *Prabhakar Raghavan*, *Hinrich Schütze* (2008): *信息检索导论. 剑桥大学出版社*. [`nlp.stanford.edu/IR-book/html/htmledition/irbook.html`](http://nlp.stanford.edu/IR-book/html/htmledition/irbook.html)

+   *Ingo Feinerer: R 中的 tm 包文本挖掘导论*. [`cran.r-project.org/web/packages/tm/vignettes/tm.pdf`](https://cran.r-project.org/web/packages/tm/vignettes/tm.pdf)

+   Ingo Feinerer (2008): R 中的文本挖掘框架及其应用. [`epub.wu.ac.at/1923/1/document.pdf`](http://epub.wu.ac.at/1923/1/document.pdf)

+   Yanchang Zhao: 使用 R 进行文本挖掘：Twitter 数据分析. [`www.rdatamining.com/docs/text-mining-with-r-of-twitter-data-analysis`](http://www.rdatamining.com/docs/text-mining-with-r-of-twitter-data-analysis)

+   Stefan Thomas Gries (2009): 使用 R 进行定量语料库语言学：实用导论. Routledge.

# 第八章：– 数据精炼

加载的 R 包版本（按章节中提到的顺序）：

+   hflights 0.1 (CRAN)

+   rapportools 1.0 (CRAN)

+   Defaults 1.1-1 (CRAN)

+   microbenchmark 1.4-2 (CRAN)

+   Hmisc 3.16-0 (CRAN)

+   missForest 1.4 (CRAN)

+   outliers 0.14 (CRAN)

+   lattice 0.20-31 (CRAN)

+   MASS 7.3-40 (CRAN)

进一步的 R 包：

+   imputeR 1.0.0 (CRAN)

+   VIM 4.1.0 (CRAN)

+   mvoutlier 2.0.6 (CRAN)

+   randomForest 4.6-10 (CRAN)

+   AnomalyDetection 1.0 (GitHub @ c78f0df02a8e34e37701243faf79a6c00120e797)

进一步阅读：

+   *推断和缺失数据*, *Biometrika 63(3)*, *581-592, Donald B. Rubin* (1976)

+   *缺失数据的统计分析*, *Wiley Roderick*, *J. A. Little* (2002)

+   *缺失数据灵活插补*, *CRC*, *Stef van Buuren* (2012)

+   *稳健统计方法 CRAN 任务视图*, *Martin Maechler* [`cran.r-project.org/web/views/Robust.html`](https://cran.r-project.org/web/views/Robust.html)

# 第九章：– 从大数据到小数据

加载的 R 包版本（按章节中提到的顺序）：

+   hflights 0.1 (CRAN)

+   MVN 3.9 (CRAN)

+   ellipse 0.3-8 (CRAN)

+   psych 1.5.4 (CRAN)

+   GPArotation 2014.11-1 (CRAN)

+   jpeg 0.1-8 (CRAN)

进一步的 R 包：

+   mvnormtest 0.1-9 (CRAN)

+   corrgram 1.8 (CRAN)

+   MASS 7.3-40 (CRAN)

+   sem 3.1-6 (CRAN)

+   ca 0.58 (CRAN)

进一步阅读：

+   *FactoMineR：用于多元分析的 R 包*，*JSS*，*Sebastien Le*，*Julie Josse*，*Francois Husson*，于*2008*年出版，可在[`factominer.free.fr/docs/article_FactoMineR.pdf`](http://factominer.free.fr/docs/article_FactoMineR.pdf)找到

+   *使用 R 进行示例性多元分析*，*CRC*，*Francois Husson*，*Sebastien Le*，*Jerome Pages*，于*2010*年

+   *因子简单性的索引*，*Psychometrika 39*，*31–36 Kaiser, H. F.*，于*1974*年

+   *R 中的主成分分析*，*Gregory B. Anderson*，可在[`www.ime.usp.br/~pavan/pdf/MAE0330-PCA-R-2013`](http://www.ime.usp.br/~pavan/pdf/MAE0330-PCA-R-2013)找到

+   *使用 R 的 sem 包进行结构方程建模*，*John Fox*，于*2006*年出版，可在[`socserv.mcmaster.ca/jfox/Misc/sem/SEM-paper.pdf`](http://socserv.mcmaster.ca/jfox/Misc/sem/SEM-paper.pdf)找到

+   *R 中的对应分析，带有二维和三维图形：ca 包*，*JSS*，由*Oleg Nenadic*，*Michael Greenacre*，于*2007*年出版，可在[`www.jstatsoft.org/v20/i03/paper`](http://www.jstatsoft.org/v20/i03/paper)找到

+   *PCA 的视觉解释*，*Victor Powell*，可在[`setosa.io/ev/principal-component-analysis/`](http://setosa.io/ev/principal-component-analysis/)找到

# 第十章：- 分类与聚类

在本章中提到的顺序中加载的 R 包版本：

+   NbClust 3.0 (CRAN)

+   cluster 2.0.1 (CRAN)

+   poLCA 1.4.1 (CRAN)

+   MASS 7.3-40 (CRAN)

+   nnet 7.3-9 (CRAN)

+   dplyr 0.4.1 (CRAN)

+   class 7.3-12 (CRAN)

+   rpart 4.1-9 (CRAN)

+   rpart.plot 1.5.2 (CRAN)

+   partykit 1.0-1 (CRAN)

+   party 1.0-2- (CRAN)

+   randomForest 4.6-10 (CRAN)

+   caret 6.0-47 (CRAN)

+   C50 0.1.0-24 (CRAN)

进一步的 R 包：

+   glmnet 2.0-2 (CRAN)

+   gbm 2.1.1 (CRAN)

+   xgboost 0.4-2 (CRAN)

+   h2o 3.0.0.30 (CRAN)

进一步阅读：

+   *统计学习的要素：数据挖掘、推理和预测*，*Springer*，由*Trevor Hastie*，*Robert Tibshirani*，*Jerome Friedman*，于*2009*年出版，可在[`statweb.stanford.edu/~tibs/ElemStatLearn/`](http://statweb.stanford.edu/~tibs/ElemStatLearn/)找到

+   *统计学习导论*，*Springer*，由*Gareth James*，*Daniela Witten*，*Trevor Hastie*，*Robert Tibshirani*，于*2013*年出版，可在[`www-bcf.usc.edu/~gareth/ISL/`](http://www-bcf.usc.edu/~gareth/ISL/)找到

+   *R 与数据挖掘：实例与案例研究*，由*Yanchang Zhao*，可在[`www.rdatamining.com/docs/r-and-data-mining-examples-and-case-studies`](http://www.rdatamining.com/docs/r-and-data-mining-examples-and-case-studies)找到

+   *机器学习基准*，由*Szilard Pafka*，于*2015*年出版，可在[`github.com/szilard/benchm-ml`](https://github.com/szilard/benchm-ml)找到

# 第十一章：- R 生态系统的社会网络分析

在本章中提到的顺序中加载的 R 包版本：

+   tools 3.2

+   plyr 1.8.2 (CRAN)

+   igraph 0.7.1 (CRAN)

+   visNetwork 0.3 (CRAN)

+   miniCRAN 0.2.4 (CRAN)

进一步阅读：

+   *使用 R 进行网络数据统计分析*，*Springer*，由*Eric D. Kolaczyk*，*Gábor Csárdi*在*2014*编写

+   *链接*，*Plume Publishing*，*Albert-László Barabási*在*2003*

+   *使用 R 和 SoNIA 进行社会网络分析实验室*，由*Sean J. Westwood*在*2010*编写，可在[`sna.stanford.edu/rlabs.php`](http://sna.stanford.edu/rlabs.php)找到

# 第十二章：– 时间序列分析

加载的 R 包版本（按章节中提到的顺序）：

+   hflights 0.1 (CRAN)

+   data.table 1.9.4 (CRAN)

+   forecast 6.1 (CRAN)

+   tsoutliers 0.6 (CRAN)

+   AnomalyDetection 1.0 (GitHub)

+   zoo 1.7-12 (CRAN)

进一步的 R 包：

+   xts 0.9-7 (CRAN)

进一步阅读：

+   *预测：原理与实践*，*OTexts*，*Rob J Hyndman*，*George Athanasopoulos*在*2013*，可在[`www.otexts.org/fpp`](https://www.otexts.org/fpp)找到

+   *时间序列分析与应用*，*Springer*，*Robert H. Shumway*，*David S. Stoffer*在*2011*编写，可在[`www.stat.pitt.edu/stoffer/tsa3/`](http://www.stat.pitt.edu/stoffer/tsa3/)找到

+   *R 时间序列分析小书*，*Avril Coghlan*在*2015*编写，可在[`a-little-book-of-r-for-time-series.readthedocs.org/en/latest/`](http://a-little-book-of-r-for-time-series.readthedocs.org/en/latest/)找到

+   *时间序列分析 CRAN 任务视图*，由*Rob J Hyndman*在[`cran.r-project.org/web/views/TimeSeries.html`](https://cran.r-project.org/web/views/TimeSeries.html)提供

# 第十三章：– 周围的数据

加载的 R 包版本（按章节中提到的顺序）：

+   hflights 0.1 (CRAN)

+   data.table 1.9.4 (CRAN)

+   ggmap 2.4 (CRAN)

+   maps 2.3-9 (CRAN)

+   maptools 0.8-36 (CRAN)

+   sp 1.1-0 (CRAN)

+   fields 8.2-1 (CRAN)

+   deldir 0.1-9 (CRAN)

+   OpenStreetMap 0.3.1 (CRAN)

+   rCharts 0.4.5 (GitHub @ 389e214c9e006fea0e93d73621b83daa8d3d0ba2)

+   leaflet 0.0.16 (CRAN)

+   diagram 1.6.3 (CRAN)

+   scales 0.2.4 (CRAN)

+   ape 3.2 (CRAN)

+   spdep 0.5-88 (CRAN)

进一步的 R 包：

+   raster 2.3-40 (CRAN)

+   rgeos 0.3-8 (CRAN)

+   rworldmap 1.3-1 (CRAN)

+   countrycode 0.18 (CRAN)

进一步阅读：

+   *使用 R 进行应用空间数据分析*，*Springer*，由*Roger Bivand*，*Edzer Pebesma*，*Virgilio Gómez-Rubio*在*2013*编写

+   *使用 R 进行生态和农业中的空间数据分析*，*CRC*，*Richard E. Plant*在*2012*

+   *使用 R 进行数值生态学*，*Springer*，由*Daniel Borcard*，*Francois Gillet*，和*Pierre Legendre*在*2012*编写

+   *R 空间分析和制图入门*，*Sage*，由*Chris Brunsdon*，*Lex Comber*在*2015*编写

+   *地理计算：实践入门*，*Sage*，由*Chris Brunsdon*，*Alex David Singleton*在*2015*编写

+   *空间数据分析 CRAN 任务视图*，由*Roger Bivand*在[`cran.r-project.org/web/views/Spatial.html`](https://cran.r-project.org/web/views/Spatial.html)提供

# 第十四章：– 分析 R 社区

加载的 R 包版本（按章节中提到的顺序）：

+   XML 3.98-1.1 (CRAN)

+   rworldmap 1.3-1 (CRAN)

+   ggmap 2.4 (CRAN)

+   fitdistrplus 1.0-4 (CRAN)

+   actuar 1.1-9 (CRAN)

+   RCurl 1.95-4.6 (CRAN)

+   data.table 1.9.4 (CRAN)

+   ggplot2 1.0.1 (CRAN)

+   forecast 6.1 (CRAN)

+   Rcapture 1.4-2 (CRAN)

+   fbRads 0.1 (GitHub @ 4adbfb8bef2dc49b80c87de604c420d4e0dd34a6)

+   twitteR 1.1.8 (CRAN)

+   tm 0.6-1 (CRAN)

+   wordcloud 2.5 (CRAN)

其他 R 包：

+   jsonlite 0.9.16 (CRAN)

+   curl 0.6 (CRAN)

+   countrycode 0.18 (CRAN)

+   VGAM 0.9-8 (CRAN)

+   stringdist 0.9.0 (CRAN)

+   lubridate 1.3.3 (CRAN)

+   rgithub 0.9.6 (GitHub @ 0ce19e539fd61417718a664fc1517f9f9e52439c)

+   Rfacebook 0.5 (CRAN)

进一步阅读：

+   *社交问答网站如何改变开源软件社区的知识共享*，*ACM*，由 *Bogdan Vasilescu*，*Alexander Serebrenik*，*Prem Devanbu* 和 *Vladimir Filkov* 在 *2014 年* 发布于 [`web.cs.ucdavis.edu/~filkov/papers/r_so.pdf`](http://web.cs.ucdavis.edu/~filkov/papers/r_so.pdf)

+   *R 活动在哪里？* 由 *James Cheshire* 在 *2013 年* 发布于 [`spatial.ly/2013/06/r_activity/`](http://spatial.ly/2013/06/r_activity/)

+   *关于 R 的七个快速事实* 由 *David Smith* 在 *2014 年* 发布于 [`blog.revolutionanalytics.com/2014/04/seven-quick-facts-about-r.html`](http://blog.revolutionanalytics.com/2014/04/seven-quick-facts-about-r.html)

+   *useR! 2013 全球参会者概览* 由 *Gergely Daroczi* 在 *2013 年* 发布于 [`blog.rapporter.net/2013/11/the-attendants-of-user-2013-around-world.html`](http://blog.rapporter.net/2013/11/the-attendants-of-user-2013-around-world.html)

+   *全球各地的 R 用户* 由 *Gergely Daroczi* 在 *2014 年* 发布于 [`blog.rapporter.net/2014/07/user-at-los-angeles-california.html`](http://blog.rapporter.net/2014/07/user-at-los-angeles-california.html)

+   *R 活动全球概况* 由 *Gergely Daroczi* 在 *2014 年* 发布于 [`rapporter.net/custom/R-activity`](http://rapporter.net/custom/R-activity)

+   *全球各地的 R 用户*（更新版）由 *Gergely Daroczi* 在 *2015 年* 发布 [`www.scribd.com/doc/270254924/R-users-all-around-the-world-2015`](https://www.scribd.com/doc/270254924/R-users-all-around-the-world-2015)
