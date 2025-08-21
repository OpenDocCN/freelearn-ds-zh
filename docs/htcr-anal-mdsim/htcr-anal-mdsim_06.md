# 第六章：测量医疗质量

本章面向所有读者，旨在向你展示美国当前基于价值的项目下，医疗提供者是如何被评估和奖励/处罚的。我们将查看从网络下载的实际提供者数据示例，并使用 Python 对这些数据进行整理，以提取我们需要的信息。通过本章学习后，你将能够定位到感兴趣的项目中基于提供者的数据，并使用`pandas`进行操作，识别出表现优秀的提供者以及那些可能从分析解决方案中受益的提供者。

# 医疗措施简介

**医疗措施**是对提供者的护理活动进行的计算，旨在显示护理人员提供的质量水平。随着提供者越来越多地根据他们提供的服务质量而非数量获得奖励，医疗措施在决定哪些护理提供者会受到奖励或处罚方面发挥着越来越重要的作用。**医疗保险和医疗补助服务中心**（**CMS**）是美国的联邦级机构之一，发布标准化的医疗措施；此外，各州也会发布相应的措施。提供者使用自己患者的数据计算这些措施，然后将计算结果提交给发布机构进行审计。结果在一定程度上决定了提供者从机构获得的报销金额。

医疗中的典型措施通常是与护理质量相关的比率或百分比。一个措施通常由两个部分组成：**分子**和**分母**。分母是指在特定时间范围内，提供者所接诊的符合资格的患者群体或接诊次数的量化。确定分母通常需要通过应用**纳入标准**和/或**排除标准**来筛选整个提供者群体，进而得到所需的测量群体或接诊池。一旦确定了分母，分子便是根据分母中获得某些积极或消极结果或事件的项数来计算的。

这个结果或事件通常由基础和/或临床研究建议作为患者护理的推荐部分（或不良护理的标志）。最后，将分子除以分母得到最终的百分比。这个百分比可以单独使用，也可以与其他措施结合，融入更复杂的公式和加权方案，以确定总体质量分数。

例如，一个机构希望评估某州门诊设施提供的糖尿病护理质量，可以通过调查文献中的糖尿病护理建议开始制定衡量标准。除其他事项外，糖尿病患者应该每年接受多次足部检查（检查溃疡和神经损伤）和糖化血红蛋白（HgbA1c）检测（检查血糖是否升高）。为了计算分母，纳入标准是患者在过去一年中至少接受过一次糖尿病（ICD 编码）诊断。该机构只希望考虑标准的成人群体；因此，18 岁以下的儿童和 65 岁以上的老年人将被排除在外。一个诊所可能有 4,000 名患者，其中 500 人符合这些标准；那么 500 就是该指标的分母。接下来需要计算两个分子：

+   这些患者中，在过去一年中接受了至少三次足部检查的数量

+   这些患者中，在过去一年中接受了至少两次 HgbA1c 检测的数量

例如，假设我们诊所的数字分别为 350 和 400。最终的指标是 350/500 = 0.70，代表糖尿病足部检查的表现，400/500 = 0.80，代表糖尿病血液检查的表现。然后，这些可以平均得出 0.75，作为该诊所糖尿病护理的总体评分。

衡量标准存在一些问题；没有一个标准能够避免漏洞，这些漏洞允许提供者操控他们的衡量分数，而不真正提高护理质量。此外，许多标准可能不公平地惩罚那些患者可能违背医疗建议或拒绝适当治疗的提供者。然而，如果要奖励护理质量，就必须有一种量化护理质量的方法，而在医疗领域，衡量标准是实现这一目标的重要手段。

# 美国医疗保险基于价值的项目

在第二章《医疗基础》中，我们讨论了**按服务项目付费**（**FFS**）报销模式在医学中的应用，在这种模式下，医生根据提供的护理量而非护理的价值获得报酬。最近，有一个推动趋势，旨在根据护理质量而非护理数量来奖励提供者。

为了促进从按服务项目付费（FFS）到基于价值的报销模式的转变，CMS 实施了基于价值的项目。这些项目根据医疗服务提供者为医疗保险患者提供的护理质量进行奖励或惩罚。2018 年，共有八个此类项目，具体如下：

+   **医院基于价值的采购（HVBP）项目**

+   **医院再入院减少（HRR）项目**

+   **医院获得性疾病（HAC）项目**

+   **终末期肾病（ESRD）质量倡议项目**

+   **熟练护理设施基于价值的项目** (**SNFVBP**)

+   **居家健康价值基础项目**（**HHVBP**）

+   **替代支付模型**（**APMs**）

+   **基于绩效的激励支付系统**（**MIPS**）

在接下来的部分中，我们将详细介绍这些项目。

# 医院基于价值的采购（HVBP）项目

HVBP 项目根据医院提供的医疗服务质量对其进行奖励，激励支付给接受医保患者的医院。HVBP 项目通过 2010 年的《平价医疗法案》设立，并于 2012 年开始实施。

# 领域与测量标准

在 2018 年，HVBP 项目大约包含 20 项测量标准，涵盖医院护理质量的四个不同领域。该列表不断扩展，到 2023 年预计将包括约 25 项测量标准。让我们在此看看每个领域和测量标准。

# 临床护理领域

临床护理领域通过使用死亡率等指标来主要评估临床护理质量。**死亡率**指特定疾病患者的死亡率。该领域使用了五项死亡率测量指标（列举如下）。第六项测量标准是全髋关节/膝关节置换（即关节置换）手术的并发症率：

+   **MORT-30-AMI**：急性心肌梗死患者的 30 天死亡率

+   **MORT-30-HF**：心力衰竭患者的 30 天死亡率

+   **MORT-30-PN**：肺炎患者的 30 天死亡率

+   **THA/TKA**：发生并发症的全髋关节置换/全膝关节置换手术数量

+   **MORT-30-COPD**：慢性阻塞性肺病（COPD）患者的 30 天死亡率

+   **MORT-30-CABG**：接受冠状动脉旁路移植手术（CABG）的患者 30 天死亡率

# 以患者和护理人员为中心的护理体验领域

以患者和护理人员为中心的护理体验领域的测量标准是通过**医院消费者评估医疗服务和系统**（**HCAHPS**）调查获得的。HCAHPS 调查在患者出院后不久，对随机抽取的医保患者进行。超过十个问题集中在以下八个测量指标上：

+   与护士的沟通

+   与医生的沟通

+   医院员工的响应性

+   药物沟通

+   医院环境的清洁度和安静度

+   出院信息

+   医院整体评分

+   三项护理过渡

# 安全领域

该领域的测量标准评估医院的安全性，特别是不良事件和院内感染等问题。该领域的所有测量标准都将在后续的 HAC 项目部分中描述（PC-01 测量标准除外，具体描述如下）：

+   **AHRQ 复合指标（PSI-90）**：有关详细描述，请参见 HAC 项目部分。

+   **导尿管相关尿路感染（CAUTI）**：有关详细描述，请参见 HAC 项目部分。

+   **中心静脉导管相关血流感染（CLABSI）**：有关详细描述，请参见 HAC 项目部分。

+   **艰难梭状芽孢杆菌感染**（CDI）：有关详细描述，请参阅 HAC 计划部分。

+   **耐甲氧西林金黄色葡萄球菌感染**（MRSA）：有关详细描述，请参阅 HAC 计划部分。

+   **手术部位感染（SSI）**：有关详细描述，请参阅 HAC 计划部分。

+   **PC-01 – 39 周未满时的选择性分娩**：指南建议妊娠尽可能接近 40 周时分娩。

# 效率和成本削减领域

本领域的四项指标检查与每家医院相关的护理成本。其中一项指标（MSPB）与每位患者的总体支出有关；其余三项指标涉及三种特定疾病的支出：

+   **每位受益人 Medicare 支出**（**MSPB**）

+   **急性心肌梗死**（**AMI**）支付

+   **心力衰竭**（**HF**）支付

+   **肺炎**（**PN**）支付

# 医院再入院减少（HRR）计划

测量住院病人护理质量的另一种方式是通过使用住院病人初次（起始）就诊时诊断为特定疾病的患者的再入院率。如果病人在医院获得了针对这些特定疾病的适当护理，那么预期再入院率应该等于或低于可接受的水平。高于基准再入院率的医院将面临较低的赔偿。因此，HRR 计划于 2012 年启动。该计划为减少住院病人在 30 天内因以下疾病再入院率的医院提供激励支付（最高可达其来自 Medicare 的住院支付的 3%）：

+   **急性心肌梗死**（**AMI**）

+   **心力衰竭**（**HF**）

+   **肺炎**（**PN**）

+   **慢性阻塞性肺疾病**（**COPD**）

+   **全髋**/ **膝关节置换术**（**THA**/ **TKA**）

+   **冠状动脉旁路移植手术**（**CABG**）

# 医院获得性疾病（HAC）计划

衡量住院病人护理质量的另一种方法是考虑该医院发生的院内感染或医源性疾病的数量。**医源性**疾病是指由医学检查或治疗引起的疾病，而**院内感染**则是指源自医院的疾病（通常是感染）。院内感染通常对多种抗生素具有耐药性，且相当难以治疗。

在 2014 年启动的 HACRP 计划下，如果医院的患者感染医院获得性感染的风险较高，则医院将面临总 Medicare 支付的 1%的处罚。更具体地说，医院如果符合特定得分阈值（基于患者发生五种常见医院获得性感染的频率以及其 AHRQ **患者安全指标**（**PSI**）90 综合指标的表现），将有资格获得 Medicare 报销的 1%的减少。

HAC 项目包括六项措施，涵盖了两大护理领域。六项措施中有五项与医院患者的感染率相关。第六项措施是一个综合性指标，评估各种不利的患者安全事件。

我们现在将更详细地了解这些领域和措施。

# 医疗获得性感染领域

以下是五种医疗获得性感染：

+   **导尿管相关尿路感染 (CAUTI)**：CAUTI 发生在使用不当（无菌）技术将尿管插入尿道时，导致细菌在尿路中繁殖。

+   **中心静脉导管相关血流感染 (CLABSI)**：类似地，当中心静脉导管不当插入体内时，细菌便可进入血液并定植（**败血症**）。

+   **艰难梭状芽孢杆菌感染 (CDI)**：住院治疗的病人在抗生素治疗后，原本的肠道菌群被消除，从而容易受到艰难梭状芽孢杆菌的感染，细菌在肠道内定植。医疗人员的卫生条件差和手部洗净不当是额外的风险因素。

+   **耐甲氧西林*金黄色葡萄球菌*（MRSA）感染**：MRSA 是一种常见且特别具有毒性的金黄色葡萄球菌株，通常感染皮肤和血液，并且对多种抗生素具有耐药性。它常常在医院传播，通过快速治疗和护理可以避免传播。

+   **手术部位感染（SSI）**：由于手术过程中或术后灭菌技术不当，导致伤口或手术部位感染。

# 患者安全领域

PSI 90 是由**医疗研究与质量局**（**AHRQ**）发布的患者安全/并发症指标。2017 年，它通过 10 项指标衡量了医院的患者安全和并发症发生率：

+   **PSI 03：压疮发生率**：压疮是由于病人长时间保持同一姿势卧床所形成的皮肤损伤。它常常被用作衡量医院护理质量/忽视情况的指标。

+   **PSI 06：医源性气胸发生率**：气胸是指肺壁破裂，导致空气积聚在肺部周围的腔隙中，从而妨碍患者正常呼吸。有些气胸是由医院手术引起的，这些被称为医源性气胸。

+   **PSI 08：住院跌倒伴髋部骨折发生率**：跌倒在老年患者中常见，尤其是在手术或治疗后。采取一些预防措施可以防止患者跌倒，未能做到这一点的医院常被认为提供了低质量的护理。

+   **PSI 09：围手术期出血或血肿发生率**：此指标衡量患者在医院接受手术时发生过量出血的情况。

+   **PSI 10: 术后急性肾损伤率**：在手术或操作后，患者因血流减少或 X 光对比剂的影响而面临肾脏损伤的风险。

+   **PSI 11: 术后呼吸衰竭发生率**：手术后，患者可能出现呼吸衰竭，这是一种危及生命的病症，需要将患者置于麻醉状态下的呼吸机上，并在**重症监护病房**（**ICU**）进行持续监护。通过指导患者进行正确的呼吸练习，可以减少呼吸衰竭事件的发生。

+   **PSI 12: 术后肺栓塞（PE）或深静脉血栓（DVT）发生率**：深静脉血栓是在下肢肌肉的静脉中形成的血块。肺栓塞是指血块通过血液循环进入肺部，导致生命威胁的并发症。许多 DVT 可以通过在住院期间使用肝素和其他治疗方法，鼓励患者保持活动来预防。

+   **PSI 13: 术后脓毒症发生率**：该指标衡量了接受手术的患者术后发生感染的频率。脓毒症是一种危及生命的病症，表现为细菌感染血液，影响器官功能。

+   **PSI 14: 术后伤口裂开率**：伤口裂开是指手术部位未能正确闭合或愈合，通常是手术操作不当或术后营养不良的表现。

+   **PSI 15: 未识别的腹腔/盆腔意外穿刺/撕裂率**：此指标衡量在腹部或盆腔手术过程中，意外穿刺/撕裂发生的频率。

更多信息请访问[`www.qualityindicators.ahrq.gov/News/PSI90_Factsheet_FAQ.pdf`](https://www.qualityindicators.ahrq.gov/News/PSI90_Factsheet_FAQ.pdf)。

# 终末期肾病（ESRD）质量激励计划

ESRD 质量激励计划衡量了 Medicare ESRD 患者在透析中心接受的护理质量。共有 16 项指标：11 项临床指标和 5 项报告指标，具体如下：

+   **NHSN 血流感染在血液透析门诊患者中的发生率**：不当的消毒技术可能导致血液透析时发生感染。该指标通过对比实际发生的感染数（分子）和预期感染数（分母），来评估感染的发生情况。

+   **ICH CAHPS**：该指标通过评估患者问卷调查反馈，审视透析中心提供护理的质量。

+   **标准化再入院率**：标准化再入院率是实际发生的非计划性 30 天再入院次数与预期非计划性 30 天再入院次数的比值。

+   **Kt/V 透析充分性措施 – 血液透析**：Kt/V 是一个公式，用于量化透析治疗的充分性。四项 Kt/V 措施检查有多少治疗会话符合不同透析患者群体的 Kt/V 阈值：

    +   **Kt/V 透析充分性措施 – 腹膜透析**

    +   **Kt/V 透析充分性措施 – 儿童血液透析**

    +   **Kt/V 透析充分性措施 – 儿童腹膜透析**

+   **标准化输血比率**：此措施比较透析患者中实际与预期的红细胞输血数量（输血是透析的不良后果）。

+   **血管通路 – 动静脉瘘**：血管通路措施量化了是否为患者提供了适当的血液通路。动静脉瘘措施评估了使用两根针头进行血液通路的动静脉瘘部位数量。

+   **血管通路 – 导管**：导管措施确定有多少导管在患者体内存在超过 90 天，这是一种感染风险。

+   **高钙血症**：该措施衡量患者经历高钙血症的月数，这是一种透析的不良反应。

+   **矿物质代谢报告**：这五项报告措施检查每个设施如何报告透析患者护理的各个方面。措施包括矿物质代谢报告、贫血管理报告、疼痛评估、抑郁症筛查和个人流感疫苗接种报告：

    +   贫血管理报告

    +   疼痛评估和跟踪报告

    +   临床抑郁症筛查和跟踪报告

    +   个人流感疫苗接种报告

# 熟练护理设施价值导向项目（SNFVBP）

SNFVBP 计划定于 2019 年开始。该计划将基于两项与结果相关的措施，部分决定政府向 SNF 支付的医疗保险报销：

+   30 天内全因再入院率

+   30 天内可能可预防的再入院率

这些标准适用于入住 SNF 的居民，且他们已被转院到其他医院。当该项目启动时，SNF 可能会通过与机器学习专家合作，预测哪些患者有再入院风险，从而获益。

有关 SNFVBP 的更多信息，请访问以下链接：[`www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/Other-VBPs/SNF-VBP.html`](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/Other-VBPs/SNF-VBP.html)。

# 家庭健康价值导向项目（HHVBP）

HHVBP 计划于 2016 年 1 月在美国 50 个州中的 9 个州启动。它根据护理质量向获得 Medicare 认证的**家庭健康机构** (**HHAs**)提供支付调整。该项目将使用 22 项措施来评估 HHAs 提供的护理质量。这些措施包括调查、过程和结果措施，并包括急诊利用和非计划性住院的相关措施。

# 优质激励支付系统（MIPS）

MIPS 是一个面向个体和团体门诊医师实践的基于价值的项目。该项目始于 2017 年，并通过 2015 年的《MACRA 法案》实施。与 APMs 项目一起，MIPS 构成了 Medicare 的**质量支付项目**（**QPP**）。它替代并整合了之前的基于价值的项目，如**医师质量报告系统**（**PQRS**）和**价值调整（VM）**项目。如果提供者的账单金额或 Medicare 患者数量达到一定要求，必须参与 MIPS。在 MIPS 中，提供者根据四个类别进行评估：

+   质量

+   推进护理信息

+   改善活动

+   成本

2017 年，确定实践最终 MIPS 得分的细分标准如下：60%为质量，25%为推进护理信息，15%为改善活动。从 2018 年开始，成本也将影响最终的 MIPS 得分。

让我们详细了解这四个绩效类别。

# 质量

在质量类别中，提供者从包含 271 项措施的列表中选择六项措施（截至 2018 年）。措施的例子包括*急性外耳道炎（耳部感染）：局部治疗*和*静脉曲张治疗伴大隐静脉消融：结果调查*。所有医学专业都有代表，提供者可以选择最适合自己的措施。然后，提供者根据措施的规范收集并提交相关数据。

# 推进护理信息

这一类别包括与推进健康信息技术相关的措施。此类别包含 15 项措施。措施的例子包括核对患者信息、向数据注册处报告临床数据，以及电子处方。

# 改善活动

在这一类别中，提供者必须证明他们已采取措施来改善实践中的护理协调、患者参与和患者安全等方面。提供者必须证明他们在 3 个月内完成了最多四项措施。

# 成本

对于最终类别，护理成本将从索赔数据中确定，提供者提供最具效率的护理将获得奖励。该类别将从 2018 年开始计入 MIPS 最终得分。

# 其他基于价值的项目

除了上述由 CMS 管理的基于价值的项目外，还有其他由其他机构管理的附加项目。我们来看看这些项目。

# 医疗效果数据与信息集（HEDIS）

HEDIS 用于衡量健康保险计划的质量，由**国家质量保证委员会**（**NCQA**）管理。HEDIS 包括约 90 个衡量标准，涵盖几乎所有医学专业。许多衡量标准与之前讨论过的指标，或者与 MIPS 临床护理类别中的 271 个衡量标准有相似之处。

# 州级指标

在 2018 年，几乎每个州都有某种形式的基于价值的程序和激励措施。通常，这些计划适用于 Medicaid 患者，因为 Medicaid 通常由州级管理。许多州也采用联邦发布的指标，并对其进行调整以适应自己的需求。例如，乔治亚州资助了 Georgia Families 计划（[`dch.georgia.gov/georgia-families`](https://dch.georgia.gov/georgia-families)），该计划允许乔治亚州的 Medicaid 患者选择健康保险计划。它通过使用 HEDIS 指标设定目标并衡量效果。

# 使用 Python 比较透析设施

在前一节中，我们概述了 CMS 实施的基于价值的激励计划。其中一个计划是 ESRD 质量激励计划，该计划根据透析设施为患有 ESRD 的 Medicare 患者提供的护理质量进行财务奖励。我们描述了 16 项评估每个 ESRD 病例的指标。

在本节中，我们将下载 CMS 发布的关于美国透析中心绩效的数据。我们将使用 Python 命令处理这些数据，以提取我们可以用来找出哪些中心表现良好，哪些中心可能从分析解决方案中受益的信息。通过适当的市场营销和销售努力的精准定位，将提高分析解决方案的效率。

# 下载数据

要下载透析设施对比数据，请完成以下步骤。

1.  访问以下网址：[`data.medicare.gov/data/dialysis-facility-compare`](https://data.medicare.gov/)。

1.  在页面上找到标有“DOWNLOAD CSV FLAT FILES (REVISED) NOW”的蓝色按钮。（为了获取正确的年份，你可能需要选择“GET ARCHIVED DATA”按钮。）点击该按钮，`.zip` 文件将开始下载。

1.  使用适当的 Windows/Mac 程序或 Linux 命令解压`.zip`文件。

1.  注意文件名为`ESRD QIP - Complete QIP Data - Payment Year 2018.csv`的目录和路径。

# 将数据导入到你的 Jupyter Notebook 会话中

要将`.csv`文件导入 Jupyter Notebook 会话，请像我们在第一章《医疗分析简介》中所做的那样打开 Jupyter Notebook 程序。打开一个新笔记本。然后，在第一个单元格中输入以下内容（用你的文件路径替换这里显示的路径），并点击播放按钮：

```py
import pandas as pd

df = pd.read_csv(
    'C:\\Users\\Vikas\\Desktop\\Bk\\Data\\DFCompare_Revised_FlatFiles\\' + 
    'ESRD QIP - Complete QIP Data - Payment Year 2018.csv', header=0
)
```

上述代码使用了`pandas`库的`read_csv()`函数将`.csv`文件导入为 DataFrame。`header`参数告诉笔记本该文件的第一行包含列名。

注意到反斜杠是成对出现的。这是因为`\`是 Python 中的转义字符。同时注意到文件名太长，无法放在一行内。在 Python 中，只要换行由括号和其他特定标点符号包围，语句就可以跨越多行，而不需要特殊处理。

# 探索数据的行与列

让我们来探索数据。在下一个单元格中，输入以下内容：

```py
print('Number of rows: ' + str(df.shape[0]))
print('Number of columns: ' + str(df.shape[1]))
```

输出结果如下：

```py
Number of rows: 6825
Number of columns: 153
```

在 2018 年文件中，应有 6,825 行和 153 列。每一行对应美国的一个透析设施。这里我们使用了 DataFrame 的`shape`属性，它返回一个元组，包含行数和列数。

我们还可以通过使用`head()`函数来查看 DataFrame 的可视化效果。`head()`函数接受一个参数`n`，用于指定打印 DataFrame 的前几行。在下一个单元格中，输入以下内容并按下播放按钮：

```py
print(df.head(n=5))
```

输出结果如下：

```py
                    Facility Name  CMS Certification Number (CCN)  \
0     CHILDRENS HOSPITAL DIALYSIS                           12306   
1                FMC CAPITOL CITY                           12500   
2                GADSDEN DIALYSIS                           12501   
3  TUSCALOOSA UNIVERSITY DIALYSIS                           12502   
4                  PCD MONTGOMERY                           12505   

...
```

你应该能够看到前五行中的一些列，如设施名称、地址和衡量得分。`head()`函数会打印出列的简要列表，从`.csv`文件的开始和结束处各选择一些列，并用省略号分隔。

让我们获取所有`153`列的完整列表。输入以下内容并按下*Enter*：

```py
print(df.columns)
```

输出结果如下：

```py
Index(['Facility Name', 'CMS Certification Number (CCN)', 'Alternate CCN 1',
       'Address 1', 'Address 2', 'City', 'State', 'Zip Code', 'Network',
       'VAT Catheter Measure Score',
       ...
       'STrR Improvement Measure Rate/Ratio',
       'STrR Improvement Period Numerator',
       'STrR Improvement Period Denominator', 'STrR Measure Score Applied',
       'National Avg STrR Measure Score', 'Total Performance Score',
       'PY2018 Payment Reduction Percentage', 'CMS Certification Date',
       'Ownership as of December 31, 2016', 'Date of Ownership Record Update'],
      dtype='object', length=153)
```

这里，我们使用 DataFrame 的`columns`属性，它会以列表的形式提供 DataFrame 的列名。不幸的是，`pandas`又一次将输出进行了缩略，因此我们无法看到所有`153`列。为了查看所有列，我们需要更明确地使用`for`循环逐个打印每一列：

```py
for column in df.columns:
    print(column)
```

输出结果如下：

```py
Facility Name
CMS Certification Number (CCN)
Alternate CCN 1
Address 1
Address 2
City
State
Zip Code
Network
VAT Catheter Measure Score
...
```

现在你将看到所有`153`列的名称。使用滚动条浏览所有列。你会发现每个 16 个衡量指标都与多个列相关联，此外还有像人口统计数据和总性能得分等附加列。

现在我们对数据集有了一个大致的概览，可以继续进行更深入的分析。

# 地理数据探索

在本节余下的部分，我们将使用许多类似 SQL 的操作来处理数据。这里是一些基本操作的 SQL 和`pandas`之间的转换表：

| **操作** | **SQL 语法** | **`pandas`函数** | **SQL 示例** | **`pandas`示例** |
| --- | --- | --- | --- | --- |
| 选择列 | `SELECT` | `[[]]` | `SELECT col1, col2, FROM df;` | `df[['col1','col2']]` |
| 选择行 | `WHERE` | `loc()`, `iloc()` | `SELECT * FROM df WHERE age=50;` | `df.loc[df['age']==50]` |
| 按列排序 | `ORDER BY` | `sort_values()` | `SELECT * FROM df ORDER BY col1;` | `df.sort_values('col1')` |
| 按列聚合 | `GROUP BY` | `groupby()` | `SELECT COUNT(*) FROM df GROUP BY col1;` | `df.groupby('col1').size()` |
| 限制行数 | `LIMIT` | `head()` | `SELECT * FROM df LIMIT 5;` | `df.head(n=5)` |

考虑到这些转换，我们可以开始按地理位置探索数据。

首先，6,825 个透析设施已经是一个庞大的数量。让我们尝试通过州来缩小范围。首先，我们统计每个州的透析设施数量：

```py
"""Equivalent SQL: SELECT COUNT(*) 
                   FROM df 
                   GROUP BY State;
"""
df_states = df.groupby('State').size()
print(df_states)
```

输出如下：

```py
State
AK      9
AL    170
AR     69
AS      1
AZ    120
CA    625
CO     75
CT     49
DC     23
DE     27
...
```

你应该能看到一个包含 50 行的表格（每个州一行，每行显示相关的计数）。

现在让我们按降序对这些行进行排序：

```py
"""Equivalent SQL: SELECT COUNT(*) AS Count 
                   FROM df 
                   GROUP BY State 
                   ORDER BY Count ASC;
"""
df_states = df.groupby('State').size().sort_values(ascending=False)
print(df_states)
```

输出如下：

```py
State
CA    625
TX    605
FL    433
GA    345
OH    314
IL    299
PA    294
NY    274
NC    211
MI    211
...
```

让我们进一步优化查询，将输出限制为 10 个州：

```py
"""Equivalent SQL: SELECT COUNT(*) AS Count 
                   FROM df 
                   GROUP BY State 
                   ORDER BY Count DESC
                   LIMIT 10;
"""
df_states = df.groupby('State').size().sort_values(ascending=False).head(n=10)
print(df_states)
```

根据结果，加利福尼亚州是拥有最多透析中心的州，其次是德克萨斯州。如果我们想要根据州来筛选透析设施，可以通过选择适当的行来实现：

```py
"""Equivalent SQL: SELECT *
                   FROM df
                   WHERE State='CA';
"""
df_ca = df.loc[df['State'] == 'CA']
print(df_ca)
```

# 根据总表现显示透析中心

几乎所有对这种以提供者为中心的数据的探索都会包括根据质量得分分析设施。接下来我们将深入探讨这一点。

首先，让我们统计不同透析设施所获得的得分：

```py
print(df.groupby('Total Performance Score').size())
```

输出如下：

```py
Total Performance Score
10           10
100          30
11            2
12            2
13            1
14            3
15            1
...
95           15
96            2
97           11
98            8
99           12
No Score    276
Length: 95, dtype: int64
```

需要注意的一点是，`Total Performance Score` 列的格式是字符串而不是整数格式，因此为了进行数值排序，我们必须先将该列转换为整数格式。其次，在运行前面的代码后，你会注意到 276 个透析设施在 `Total Performance Score` 列中的值为 `No Score`。这些行在转换为整数格式之前必须被删除，以避免出现错误。

在以下代码中，我们首先删除了 `No Score` 行，然后使用 `pandas` 的 `to_numeric()` 函数将字符串列转换为整数列：

```py
df_filt= df.loc[df['Total Performance Score'] != 'No Score']
df_filt['Total Performance Score'] = pd.to_numeric(
    df_filt['Total Performance Score']
)
```

现在，我们创建一个新的 DataFrame，仅选择我们感兴趣的几个列并进行排序，将表现最差的中心排在最前面。例如，这样的代码块对识别表现最差的透析中心非常有帮助。我们展示前五个结果：

```py
df_tps = df_filt[[
    'Facility Name',
    'State', 
    'Total Performance Score'
]].sort_values('Total Performance Score')
print(df_tps.head(n=5))
```

输出如下：

```py
                                   Facility Name State  \
5622   462320 PRIMARY CHILDREN'S DIALYSIS CENTER    UT   
698              PEDIATRIC DIALYSIS UNIT AT UCSF    CA   
6766                  VITAL LIFE DIALYSIS CENTER    FL   
4635  BELMONT COURT DIALYSIS - DOYLESTOWN CAMPUS    PA   
3763                       WOODMERE DIALYSIS LLC    NY   

      Total Performance Score  
5622                        5  
698                         7  
6766                        8  
4635                        8  
3763                        9
```

另外，如果我们希望分析每个州透析中心的平均总表现，可以通过结合使用 `numpy.mean()` 和 `groupby()` 来实现：

```py
import numpy as np

df_state_means = df_filt.groupby('State').agg({
    'Total Performance Score': np.mean
})
print(df_state_means.sort_values('Total Performance Score', ascending=False))
```

输出如下：

```py
       Total Performance Score
State                         
ID                   73.178571
WY                   71.777778
HI                   70.500000
UT                   70.421053
CO                   70.173333
WA                   70.146067
ME                   70.058824
OR                   70.046154
KS                   69.480769
AZ                   68.905983
...
```

根据这个查询的结果，爱达荷州和怀俄明州的透析中心表现最好。你也可以通过以下修改添加一列，显示每个州透析中心的数量：

```py
import numpy as np

df_state_means = df_filt.groupby('State').agg({
    'Total Performance Score': np.mean,
    'State': np.size
})
print(df_state_means.sort_values('Total Performance Score', ascending=False))
```

输出如下：

```py
       Total Performance Score  State
State                                
ID                   73.178571     28
WY                   71.777778      9
HI                   70.500000     26
UT                   70.421053     38
CO                   70.173333     75
WA                   70.146067     89
ME                   70.058824     17
OR                   70.046154     65
KS                   69.480769     52
AZ                   68.905983    117
...
```

结果表明，在只考虑拥有至少 100 个透析中心的州时，亚利桑那州的总表现最佳。

# 对透析中心的替代分析

本节中介绍的代码可以调整用于对透析中心进行各种不同类型的分析。例如，您可能希望根据透析中心所有者来衡量平均表现，而不是按 `State` 来衡量。这可以通过在最近的示例中更改分组的列来实现。或者，您也许想查看单个指标，而不是 `Total Performance Score`，这同样可以通过仅更改代码中的一列来完成。

现在，我们已经使用透析中心分析了服务提供商的表现，接下来我们将查看一个更复杂的数据集——住院医院表现数据集。

# 比较医院

在前面的示例中，我们使用 Python 分析了透析中心的表现。透析中心只是医疗服务提供者池中的一小部分——该池还包括医院、门诊诊所、疗养院、住院康复设施和临终关怀服务等。例如，当您从[`data.medicare.gov`](https://data.medicare.gov)下载透析设施比较数据时，您可能会注意到这些其他设施的表现数据。现在我们将研究一个更复杂的设施类型：住院医院。医院比较数据集包含了 CMS 基于价值的三个项目的数据。它是一个庞大的数据集，我们将使用这些数据展示一些高级的 Python 和 `pandas` 特性。

# 下载数据

要下载医院比较数据集，请完成以下步骤：

1.  访问[`data.medicare.gov/data/hospital-compare`](https://data.medicare.gov/data/hospital-compare)。

1.  在页面上找到标有“DOWNLOAD CSV FLAT FILES (REVISED) NOW” 的蓝色按钮。（如果要获取正确的年份，您可能需要选择“GET ARCHIVED DATA”按钮）。点击该按钮，`.zip` 文件将开始下载。

1.  使用适当的 Windows/Mac 程序或 Linux 命令提取 `.zip` 文件。

1.  请注意包含已提取 `.csv` 文件的路径。

# 将数据导入到您的 Jupyter Notebook 会话中

请注意，提取的医院比较文件夹包含 71 个文件，其中绝大多数是 `.csv` 文件。这是很多表格！让我们将一些表格导入到 Jupyter Notebook 中：

```py
import pandas as pd

pathname = 'C:\\Users\\Vikas\\Desktop\\Bk\\Data\\Hospital_Revised_Flatfiles\\'

files_of_interest = [
    'hvbp_tps_11_07_2017.csv', 
    'hvbp_clinical_care_11_07_2017.csv',
    'hvbp_safety_11_07_2017.csv',
    'hvbp_efficiency_11_07_2017.csv',
    'hvbp_hcahps_11_07_2017.csv'
]

dfs = {
    foi: pd.read_csv(pathname + foi, header=0) for foi in files_of_interest
}
```

上面的代码将与 HVBP 测量相关的表格加载到 Python 会话中。共有五个表格，其中四个表格对应该测量的四个领域，一个表格代表整体评分。

请注意，替代显式地创建和导入五个数据框，我们使用列表推导式创建了一个数据框字典。我们在 Python 章节中已经讲解了字典、列表和推导式。这在当前和即将到来的单元中节省了大量的输入工作。

# 探索表格

接下来，让我们探索表格，并检查每个表格中的行和列数：

```py
for k, v in dfs.items():
    print(
        k + ' - Number of rows: ' + str(v.shape[0]) + 
        ', Number of columns: ' + str(v.shape[1]) 
    )   
```

输出结果如下：

```py
hvbp_tps_11_07_2017.csv -  Number of rows: 2808, Number of columns: 16
hvbp_clinical_care_11_07_2017.csv -  Number of rows: 2808, Number of columns: 28
hvbp_safety_11_07_2017.csv -  Number of rows: 2808, Number of columns: 64
hvbp_efficiency_11_07_2017.csv -  Number of rows: 2808, Number of columns: 14
hvbp_hcahps_11_07_2017.csv -  Number of rows: 2808, Number of columns: 73
```

在前一个单元中，我们使用了字典的`items()`方法来遍历字典中的每个键-数据框对。

所有的表格都有相同的行数。由于每一行都对应着一个医院，因此可以安全地假设所有表格中的医院是相同的（我们稍后将验证这一假设）。

由于表格之间的分离，任何我们进行的分析都有其局限性。由于所有医院都是（假设）相同的，我们可以将所有列合并为一个表格。我们将使用`pandas`的`merge()`函数来实现这一点。使用`pandas`的`merge()`类似于使用`SQL JOIN`（你在第四章中学习了`SQL JOIN`，*计算基础 – 数据库*）。合并通过指定两个表格中共有的 ID 列来进行，这样就可以根据该列匹配行。为了查看五个 HVBP 表格中是否有共同的 ID 列，我们可以打印出每个表格的列名：

```py
for v in dfs.values():
    for column in v.columns:
        print(column)
    print('\n')
```

如果你浏览结果，你会注意到所有表格中都有`Provider Number`列。`Provider Number`是一个独特的标识符，可以用来链接这些表格。

# 合并 HVBP 表格

让我们尝试合并两个表格：

```py
df_master = dfs[files_of_interest[0]].merge(
    dfs[files_of_interest[1]], 
    on='Provider Number', 
    how='left',
```

```py
    copy=False
)

print(df_master.shape)
```

输出如下：

```py
(2808, 43)
```

我们的合并操作似乎成功了，因为`df_master`中的列数是前两个数据框列数的总和，减去一（`on`列没有被复制）。我们来看一下新数据框的列：

```py
print(df_master.columns)
```

输出如下：

```py
Index(['Provider Number', 'Hospital Name_x', 'Address_x', 'City_x', 'State_x',
       'Zip Code', 'County Name_x',
       'Unweighted Normalized Clinical Care Domain Score',
       'Weighted Normalized Clinical Care Domain Score',
       'Unweighted Patient and Caregiver Centered Experience of Care/Care Coordination Domain Score',
       'Weighted Patient and Caregiver Centered Experience of Care/Care Coordination Domain Score',
       'Unweighted Normalized Safety Domain Score',
       'Weighted Safety Domain Score',
       'Unweighted Normalized Efficiency and Cost Reduction Domain Score',
       'Weighted Efficiency and Cost Reduction Domain Score',
       'Total Performance Score', 'Hospital Name_y', 'Address_y', 'City_y',
       'State_y', 'ZIP Code', 'County Name_y',
       'MORT-30-AMI Achievement Threshold', 'MORT-30-AMI Benchmark',
       'MORT-30-AMI Baseline Rate', 'MORT-30-AMI Performance Rate',
       'MORT-30-AMI Achievement Points', 'MORT-30-AMI Improvement Points',
       'MORT-30-AMI Measure Score', 'MORT-30-HF Achievement Threshold',
       'MORT-30-HF Benchmark', 'MORT-30-HF Baseline Rate',
       'MORT-30-HF Performance Rate', 'MORT-30-HF Achievement Points',
       'MORT-30-HF Improvement Points', 'MORT-30-HF Measure Score',
       'MORT-30-PN Achievement Threshold', 'MORT-30-PN Benchmark',
       'MORT-30-PN Baseline Rate', 'MORT-30-PN Performance Rate',
       'MORT-30-PN Achievement Points', 'MORT-30-PN Improvement Points',
       'MORT-30-PN Measure Score'],
      dtype='object')
```

重复的列（`Hospital Name`、`Address`、`City`等）在合并后的表格中添加了后缀`_x`和`_y`，以指示它们来自哪个表格，确认了合并操作成功。

让我们使用一个`for`循环将其余的三个表格与`df_master`合并：

```py
for df in dfs.values():
    df.columns = [col if col not in ['Provider_Number'] else 'Provider Number' 
        for col in df.columns]

for num in [2,3,4]:
    df_master = df_master.merge(
        dfs[files_of_interest[num]],
        on='Provider Number',
        how='left',
        copy=False
    )

print(df_master.shape)
```

输出如下：

```py
(2808, 191)
```

在这一单元中，首先我们使用一个循环将所有列名从`Provider_Number`重命名为`Provider Number`，以便我们可以清晰地连接表格。

然后我们使用一个循环将每个剩余的表格与`df_master`合并。最终表格中的列数等于所有表格的列数总和，减去四。

为了确认合并是否成功，我们可以打印出新表格的列：

```py
for column in df_master.columns:
    print(column)
```

滚动输出确认了五个表格中的所有列都已包含。

我们将留给你使用*比较透析设施*部分的代码示例，进行更多的分析。

# 总结

在本章中，我们回顾了一些当今塑造美国医疗行业的突出基于价值的项目。我们已经看到这些项目如何通过使用度量标准来量化提供者的表现。此外，我们下载了用于比较透析设施和医院的数据，并通过一些 Python 代码示例来展示如何分析这些数据。

有人可能会争辩，书中这一章的分析可以通过使用诸如 Microsoft Excel 之类的电子表格应用程序来完成，而不必编程。在第七章，*在医疗保健中构建预测模型*，我们将使用医疗保健数据集训练预测模型，以预测急诊科的出院状态。正如你将看到的，这种类型的分析几乎肯定需要编写代码。

# 参考文献

Data.Medicare.gov（2018）。于 2018 年 4 月 28 日访问自[`data.medicare.gov`](https://data.medicare.gov)。

MIPS 概述（2018）。于 2018 年 4 月 28 日访问自[`qpp.cms.gov/mips/overview`](https://qpp.cms.gov/mips/overview)。

什么是基于价值的项目？（2017 年 11 月 9 日）。*美国医疗保险和医疗补助服务中心*。于 2018 年 4 月 28 日访问自[`www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/Value-Based-Programs.html`](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/Value-Based-Programs.html)。
