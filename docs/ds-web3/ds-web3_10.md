

# 第十章：安全与欺诈检测概述

“所有战争都基于欺骗。” —— 孙子

欺诈的历史与人类历史一样悠久。在加密世界中，欺诈并不令人惊讶，随着加密货币的主流化，意识到不同形式的欺诈变得愈加重要，以便能够识别它。

欺诈是企业、政府以及区块链行业尤其面临的重大问题。根据 2022 年普华永道全球欺诈年度报告，46%的受访组织表示在过去 24 个月内遭遇过某种形式的欺诈或经济犯罪。

政府出于税务目的以及打击洗钱和恐怖主义融资的需要，已经意识到这个问题。相关机构有责任执行法律，打击此类非法活动，即使涉及加密货币。对于所有涉及金融活动或货币服务业务的主体，都需要进行尽职调查，这一概念已扩展至包括集中式交易所。因此，主要的**集中式交易所**（**CEXs**）仅在验证身份证背后的人身份后才会为新用户开设账户。遵守一定的质量标准至关重要，未能遵守可能导致制裁。

加密行业也有兴趣通过合规的智能合约和无缝交易体验来巩固信任。智能合约漏洞、欺诈计划以及将加密货币作为支付手段来进行黑客攻击的行为并未助力这一目标。数据科学在解决这个问题方面已经有一段时间了，挑战在于如何将数据科学实践应用于区块链数据集，因为它具有较大的差异性且是伪匿名的。一些有趣的公司，如 Chainalysis、Elliptic 和 CipherTrace，处于区块链法医数据科学的前沿，帮助相关部门进行调查并支持用户的普遍信任。

在本章中，我们将研究地址的交易行为，训练一个机器学习模型，以确定我们是否遇到骗子。

特别地，我们将覆盖以下主题：

+   以太坊上的非法活动表现

+   以太坊交易数据的探索性数据分析

+   准备工作、模型训练和评估以标记欺诈交易

# 技术要求

你可以在本书的 GitHub 仓库中找到本章的所有数据和代码文件，网址为[`github.com/PacktPublishing/Data-Science-for-Web3/tree/main/Chapter10`](https://github.com/PacktPublishing/Data-Science-for-Web3/tree/main/Chapter10)。我们建议你浏览`Chapter10`文件夹中的代码文件，以便跟随学习。

在本章中，我们将使用以太坊实用程序库（`eth-utils`），该库包含用于与以太坊合作的 Python 开发人员常用的实用函数。根据我们的环境，我们可能需要导入其他低级别库，这些库由`eth-utils`使用。

如果您尚未安装`eth-utils`，可以使用以下代码片段进行安装：

```py
pip install eth-utils
```

`eth-utils`的文档可在[`eth-utils.readthedocs.io/en/stable/`](https://eth-utils.readthedocs.io/en/stable/)找到。如果由于缺少支持库而安装失败，您可以在`Chapter10/EDA.ipynb`中找到需要预先安装的完整库列表。

# 以太坊非法活动入门

欺诈和骗局之间存在技术上的差异。骗局是一种行为，我们在不知情的情况下支付虚假物品、转移资金或向犯罪分子提供我们的私钥。相反，欺诈指的是我们的地址上出现的任何未经授权的可疑活动。在本书中，我们将两个术语互换使用。

以太坊安全博客为刚入行加密行业的人提供了三条关键信息：

+   *始终* *持怀疑态度*

+   *没有人会免费或* *打折 ETH*

+   *没有人需要访问您的私钥或* *个人信息*

截至今日，一些最常见的欺诈行为包括以下几种：

+   **赠送骗局**: 这些骗局基本上通过犯罪分子承诺，如果我们向某个地址发送*X*数量的加密货币，它将被以加倍的金额返回给我们。这些方案通常是心理上的，只为受害者提供有限的时间参与这个“机会”，产生**错失良机** (**FOMO**) 的效果。此方案通常利用高调的 X（前身为 Twitter）账户、名人视频等。

+   **IT 支持骗局**: 这些骗子冒充区块链服务（如钱包、交易所、市场等）的 IT 支持或管理员人员。他们可能会向我们索要一些信息或验证我们的身份，并且通常会试图获取提取我们资金所需的最少信息。这类骗子主要出现在 Discord 讨论频道或 Telegram 上。通常可以在真实人名旁边看到短语“*永远不要主动私信*”。骗子会主动发起对话，试图建立联系和信任。值得记住的是，Web3 是一个去中心化的领域，因此不太可能看到支持团队在互联网上回答我们的问题。如果我们与一个有支持团队的集中式平台进行交互，团队将始终通过官方渠道联系我们。

+   **网络钓鱼诈骗**：这是一种社交工程攻击，通过电子邮件伪装来诱使受害者提供必要的信息以实施诈骗。邮件通常包含指向假网站的链接或下载恶意软件。正如我们在*第三章*中所看到的，在详细说明如何访问链外数据时，我们应该尝试通过受信任的链接访问网站——例如，通过 CoinMarketCap 或 CoinGecko。检测网络钓鱼有很多方法，但事实是它们随着时间的推移变得越来越有创意。

+   **经纪人诈骗**：这些是拥有大量社交媒体粉丝的交易经纪人，声称能够创造出色的利润。经纪人账户背后有一个真实的人物，他会与受害者互动，直到后者将资金转给经纪人“管理”。一旦资金转出，这些资金就会丢失。

每种诈骗都通过实际案例在这个博客中进一步解释：[`ethereum.org/en/security/`](https://ethereum.org/en/security/)。

无论欺诈方案如何，资金都会通过交易转移并保存在账户中。在接下来的部分，我们将分析账户行为，尝试确定哪些账户可以信任，哪些账户应该被标记为诈骗。

## 预处理

我们将使用在论文《以太坊区块链上非法账户的检测》中使用的平衡数据集。你可以在*进一步阅读*部分找到该论文的链接。这个数据集是一个平衡数据集，包含 48 列或特征，结合了合法和非法账户。该数据集通过使用 CryptoScamDB 数据库和 Etherscan 创建，后者是我们已经熟悉的工具。[cryptoscamdb.org](http://cryptoscamdb.org)管理一个开源数据集，跟踪恶意网址及其关联地址。

列及其描述如下：

| **特征** | **描述** |
| --- | --- |
| `Avg_min_between_sent_tnx` | 账户发送交易之间的平均时间（分钟） |
| `Avg_min_between_received_tnx` | 账户接收交易之间的平均时间（分钟） |
| `Time_Diff_between_first_and_last(Mins)` | 第一次和最后一次交易之间的时间差（分钟） |
| `Sent_tnx` | 发送的普通交易总数 |
| `Received_tnx` | 接收到的普通交易总数 |
| `Number_of_Created_Contracts` | 创建的合约交易总数 |
| `Unique_Received_From_Addresses` | 账户收到交易的唯一地址总数 |
| `Unique_Sent_To_Addresses` | 账户发送交易的唯一地址总数 |
| `Min_Value_Received` | 曾经接收到的最小以太币金额 |
| `Max_Value_Received` | 曾经接收到的最大以太币金额 |
| `Avg_Value_Received` | 曾经接收到的平均以太币金额 |
| `Min_Val_Sent` | 发送过的最小 Ether 值 |
| `Max_Val_Sent` | 发送过的最大 Ether 值 |
| `Avg_Val_Sent` | 发送的 Ether 平均值 |
| `Min_Value_Sent_To_Contract` | 发送到合约的最小 Ether 值 |
| `Max_Value_Sent_To_Contract` | 发送到合约的最大 Ether 值 |
| `Avg_Value_Sent_To_Contract` | 发送到合约的 Ether 平均值 |
| `Total_Transactions(Including_Tnx_to_Create_Contract)` | 总交易数量（包括创建合约的交易） |
| `Total_Ether_Sent` | 发送的总 Ether 数量（针对某个账户地址） |
| `Total_Ether_Received` | 接收的总 Ether 数量（针对某个账户地址） |
| `Total_Ether_Sent_Contracts` | 发送到合约地址的总 Ether 数量 |
| `Total_Ether_Balance` | 执行交易后的总 Ether 余额 |
| `Total_ERC20_Tnxs` | ERC20 代币转账交易的总数 |
| `ERC20_Total_Ether_Received` | 接收到的 ERC20 代币交易总量（以 Ether 计算） |
| `ERC20_Total_Ether_Sent` | 总共发送的 ERC20 代币交易数量（以 Ether 计算） |
| `ERC20_Total_Ether_Sent_Contract` | 发送到其他合约的 ERC20 代币总量（以 Ether 计算） |
| `ERC20_Uniq_Sent_Addr` | 发送到唯一账户地址的 ERC20 代币交易数量 |
| `ERC20_Uniq_Rec_Addr` | 从唯一地址接收的 ERC20 代币交易数量 |
| `ERC20_Uniq_Rec_Contract_Addr` | 从唯一合约地址接收的 ERC20 代币交易数量 |
| `ERC20_Avg_Time_Between_Sent_Tnx` | ERC20 代币发送交易之间的平均时间（以分钟计算） |
| `ERC20_Avg_Time_Between_Rec_Tnx` | ERC20 代币接收交易之间的平均时间（以分钟计算） |
| `ERC20_Avg_Time_Between_Contract_Tnx` | ERC20 代币发送交易之间的平均时间（针对合约交易） |
| `ERC20_Min_Val_Rec` | 从 ERC20 代币交易中接收到的最小 Ether 值 |
| `ERC20_Max_Val_Rec` | 从 ERC20 代币交易中接收到的最大 Ether 值 |
| `ERC20_Avg_Val_Rec` | 从 ERC20 代币交易中接收到的平均 Ether 值 |
| `ERC20_Min_Val_Sent` | 从 ERC20 代币交易中发送的最小 Ether 值 |
| `ERC20_Max_Val_Sent` | 从 ERC20 代币交易中发送的最大 Ether 值 |
| `ERC20_Avg_Val_Sent` | 通过 ERC20 代币交易发送的平均 Ether 值 |
| `ERC20_Uniq_Sent_Token_Name` | 发送的唯一 ERC20 代币数量 |
| `ERC20_Uniq_Rec_Token_Name` | 接收的唯一 ERC20 代币的数量 |
| `ERC20_Most_Sent_Token_Type` | 通过 ERC20 交易发送最多的代币类型 |
| `ERC20_Most_Rec_Token_Type` | 通过 ERC20 交易接收最多的代币类型 |

表 10.1 – 每列数据集的解释（来源 – 《检测以太坊区块链上的非法账户》论文第 10 页）

在`Chapter10/EDA.ipynb`中，我们分析了数据集并得出了以下结论：

1.  数据中有 4,681 个账户和 48 个特征。在这些地址中，有五个重复地址和五个无效地址。

    为了判断一个地址是否有效或无效，我们使用了 EIP-55 的一部分代码，并结合一个名为`address_validation()`的自定义公式。

    在`address_validation()`函数中，我们添加了一个额外的条件来计算每个地址的字符数，丢弃那些不是 42 个字符的地址。

    如果满足两个条件，则认为地址有效，并返回带有校验和的版本。否则，返回`not an ethereum address`标志：

    ```py
    def checksum_encode(addr):
        hex_addr = addr.hex()
        checksummed_buffer = ""
        hashed_address = eth_utils.keccak(text=hex_addr).hex()
        for nibble_index, character in enumerate(hex_addr):
            if character in "0123456789":
    checksummed_buffer += character
            elif character in "abcdef":
                hashed_address_nibble = int(hashed_address[nibble_index], 16)
                if hashed_address_nibble > 7:
                    checksummed_buffer += character.upper()
                else:
                    checksummed_buffer += character
            else:
                raise eth_utils.ValidationError(
                    f"Unrecognized hex character {character!r} at position {nibble_index}"
    )
        return "0x" + checksummed_buffer
    def test(addr_str):
        addr_bytes = eth_utils.to_bytes(hexstr=addr_str)
        checksum_encoded = checksum_encode(addr_bytes)
        try:
          assert checksum_encoded == addr_str, f"{checksum_encoded} != expected {addr_str}"
        except AssertionError:
          return checksum_encoded
    def address_validation(addr_str):
        if len(addr_str) == 42:
    result = test(addr_str)
        else:
            result = "not an ethereum address"
        return result
    ```

1.  第 25 到 49 列有一些缺失值，缺失值的百分比为 17.7%。我们识别数据框中的任何缺失值，并计算每列的缺失值百分比。

    我们使用以下代码片段：

    ```py
    Chapter10/Rebuilding.ipynb.
    ```

1.  欺诈交易和缺失值位于数据集的顶部。看起来最具欺诈性的账户似乎有缺失值。请参阅`Chapter10/EDA.ipynb`中的热力图。

1.  有 12 列数据方差较小（且只有一个值，即零）。方差较小的列可能对我们的训练没有帮助。这些列如下：

    `['min_value_sent_to_contract', 'max_val_sent_to_contract', 'avg_value_sent_to_contract', 'total_ether_sent_contracts', 'ERC20_avg_time_between_sent_tnx', 'ERC20_avg_time_between_rec_tnx', 'ERC20_avg_time_between_rec_2_tnx', 'ERC20_avg_time_between_contract_tnx', 'ERC20_min_val_sent_contract', 'ERC20_max_val_sent_contract', 'ERC20_avg_val_sent_contract']`

    我们使用以下代码片段来识别这些列：

    ```py
    variance_df= df.nunique()
    ```

1.  清理重复项和无效项后，有 2,497 个非欺诈账户和 2,179 个欺诈账户。

1.  我们运行了相关矩阵，并发现有五列数据高度相关。这些列如下：

    `['ERC20_max_val_rec', 'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent', 'ERC20_uniq_rec_token_name']`

    删除具有相似信息的列非常重要，因为冗余信息不会为我们的训练增加价值，反而可能使我们的算法学习变慢，并且如果我们需要解释模型，可能会使模型的可解释性变得更复杂。多重共线性也会影响某些模型，如线性模型。请参阅`Chapter10/EDA.ipynb`中的相关热力图。

1.  两列包含分类数据（`ERC20_most_sent_token_type` 和 `ERC20_most_rec_token_type`）非常稀疏，大多数代币只出现一次或为空。

    当我们按 ERC 代币进行分组时，没有明显的类别可以用来提供帮助。对这些列进行独热编码可能会导致一个稀疏的训练数据集。此外，每天都有新的代币被铸造，将这些信息加入我们的模型将创造一个很快过时的变量。

基于前面分析得出的结论，我们清理了数据以使其适应我们的需求。采取的步骤包括：

1.  我们删除了重复的地址。

1.  我们尝试从 Etherscan 填充缺失值。

1.  我们删除了低方差的列。

1.  我们删除了相关性为 0.95 或更高的列。

1.  我们删除了 `object` 类型的列。

1.  我们删除了 `Address` 列。

1.  我们用中位数填充了 NaN 值。

让我们扩展一下 *步骤 2* 的预处理工作，涉及从 Etherscan 填充缺失值。现实中的数据集往往不完整，数据分析师或数据科学家的职责就是填补那些部分完成的列。填补的方式有很多种，我们在 *第五章*中探讨了传统方法，适用于当没有更多数据可用时。

在 `Chapter10_Rebuilding` 中，我们尝试了另一种方法论，即直接访问数据源，查找缺失的数据。全书中，我们列出了多个数据源，随着这一领域的不断发展，新的链上数据源将逐步可用，帮助我们完善数据集。此外，Web3 的专业化将使我们能够对数据进行推断，因为我们能理解这些数据。当我们需要在数据集里反映的某个数据点是间接的，或者并非直接来源于链上数据时，这一点尤其重要。

在从论文《检测以太坊区块链上的非法账户》中提取的数据集里，有 17% 的行缺失数据。为了补充这部分数据，我们使用了一个我们已经分析了一段时间的数据源——Etherscan。我们利用了他们免费 API 的免费层，能够补充大部分缺失的行。API 文档链接为 [`docs.etherscan.io/`](https://docs.etherscan.io/)。补充这些列的步骤在 `Chapter10_Rebuilding` 中进行了说明，我们提取了缺失的数据点并将其添加到数据框中。如果 Etherscan 没有记录，我们可以推断该地址并未进行过该交易。

经过这一过程后，仍然留下了一些空值行，我们用列的中位数进行了填充。通过这最后一步，我们获得了一个完整的数据集，准备好进行训练。

`.csv` 文件已上传至本书的 GitHub，并可通过 `final.csv` 访问。

关于优秀预处理的一些说明

在该领域，关于数据构建和预处理的一个典型例子是与论文 *Exploiting Blockchain Data to Detect Smart Ponzi Schemes on Ethereum* 相关的一个数据集（论文链接在 *进一步阅读* 部分提供）。该论文的目的是训练一个机器学习算法来分类庞氏骗局的智能合约。为了实现这一目标，研究人员构建了一个包含 200 个庞氏骗局智能合约和 3,580 个非庞氏骗局智能合约的数据集。对于每个合约，他们提取了字节码、交易数据和内部交易。由于内部交易没有存储在链上，研究团队重新运行了一个以太坊客户端来重现这些交易。此外，为了将字节码转换为有意义的特征或类别，团队将其转换为操作码并记录了每种操作码在智能合约中的频率。

## 训练模型

一旦我们完成了数据清洗和预处理，我们将数据集进行洗牌，然后将其分割为训练集和测试集。接着，我们遍历在二分类任务中表现良好的多个模型，包括 `KNeighborsClassifier`、`DecisionTreeClassifier`、`AdaBoostClassifier`、`GradientBoostingClassifier` 和 `RandomForestClassifier`。

然而，仅仅因为一个模型在某个数据集上表现良好，并不意味着它在另一个数据集上也能取得相同效果。这时，调整模型变得非常重要。机器学习模型有一些超参数，需要调整以适应特定数据。定制这些超参数来适应我们的数据集，将提高模型的性能。为了执行这种优化，有一些可用的工具，比如 scikit-learn 中的 GridSearchCV。

`GridSearchCV` 返回的 `best_estimator_` 将包含在其中。如果我们不知道选择哪些参数，可以运行 `RandomizedSearchCV`，它会定义一个搜索空间并随机测试。两个类的文档可以在 [`scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.xhtml`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.xhtml) 和 [`scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.xhtml`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.xhtml) 中找到。

`GridSearchCV` 和 `RandomizedSearchCV` 都有一个 `cv` 参数，表示交叉验证。`K` 个大小相等的区间用于运行 `K` 次学习实验。然后将结果平均化，以减少模型性能的随机性并提高其稳健性。该类通常包含在其他实现中，例如网格搜索。然而，它也可以独立使用。

有一些变化，例如分层*K*折叠，它确保每个分割包含每个类别相同比例的观察值，以及重复*K*折叠，它为每个折叠重复操作，但每次都会打乱每个分区，使其成为一个新的样本。有关这些过程的文档可以在[`scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.xhtml`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.xhtml)找到。

我们的训练结果显示`GradientBoostingClassifier`模型表现最好。

## 评估结果

我们将使用混淆矩阵来展示模型的表现，因为它对于二分类问题非常有用。评估指标的选择将取决于所处理问题的类型。在欺诈检测的情况下，我们希望确保最小化假阴性率，因此我们将使用**召回率**作为评估指标。有关评估指标的详细解释，参见*第五章*。

结果在整个项目中的得分是 95.6%，这是一个不错的数字。

一个不平衡的数据集

一个类似的数据集也可以在 Kaggle 上找到，最初我们在*第六章*中进行了分析。这个数据集是一个不平衡的形式，因为只有 20%的行是欺诈性的。欺诈检测任务通常与不平衡数据有关，因为通常合法交易的数量要多于欺诈交易。某些模型可能会忽视少数类，而少数类有时正是我们感兴趣的需要检测的特定类别。为了处理不平衡数据，传统的方法是通过双向重采样数据集，或者通过过采样少数类或随机删除多数类行来进行欠采样。关于这些过程的更多文档可以在*进一步阅读*部分找到。

## 展示结果

数据实践者必须具备强大的沟通能力，以确保他们的发现能够让决策同事、客户以及所有使用这些见解的人容易理解。根据观众的不同，定制结果的展示方式至关重要。

我们可以通过**仪表盘**来展示分析结果，之前在本书中我们学习了如何使用 Dune analytics、Flipside 或 Covalent 来构建仪表盘。值得注意的是，并非所有的可视化分析平台都能灵活查询链上数据；有些平台仅限于传统数据库。像 Tableau 和 Power BI 这样的平台非常灵活，能够连接 API 并处理来自链上数据的复杂 SQL 查询。

我们还可以利用公司**幻灯片**演示文稿，并且可以通过非正式的**X（前身为 Twitter）讨论串**来传达数据分析的结果。无论选择何种媒介，目标都是捕获并保持观众的兴趣。以一个引人注目的问题和引言开始，保持句子简洁，只有在观众表现出兴趣时才深入细节，这些都是关键原则。一份关于创作引人入胜故事的宝贵资源可以在这个 X 讨论串中找到：[`twitter.com/alexgarcia_atx/status/1381066483330117632`](https://twitter.com/alexgarcia_atx/status/1381066483330117632).

无论演示平台如何，讲故事的技巧在传递发现时至关重要，要有吸引力的叙事。

# 摘要

总结来说，我们已经识别并讨论了加密货币领域的一个关键威胁，突出了有效交易监控和识别的必要性。为此，我们在以太坊地址层面开展了机器学习实验，利用 Etherscan 完成了我们的数据集。

我们评估并比较了各种机器学习模型，通过网格搜索超参数调优和交叉验证优化其性能。通过开展这个项目，我们深入探讨了一个法医专业人员活跃的主题，并且这个话题仍然是当前的新闻焦点。

区块链取证是数据科学应用中较为创新的领域之一，因为模型需要扩展并不断发展以适应新的变化，能够发现新类型的欺诈和诈骗。

在下一章中，我们将深入探讨价格预测。

# 进一步阅读

以下是供您进一步阅读的资源列表：

+   PwC. (2022). PwC 全球经济犯罪与欺诈调查报告 2022\. [`www.pwc.com/gx/en/forensics/gecsm-2022/PwC-Global-Economic-Crime-and-Fraud-Survey-2022.pdf`](https://www.pwc.com/gx/en/forensics/gecsm-2022/PwC-Global-Economic-Crime-and-Fraud-Survey-2022.pdf)

+   Furneaux, N. (2018). *研究加密货币：理解、提取和分析区块链证据*. John Wiley & Sons. 第 268 页。

+   Sfarrugia15/Ethereum_Fraud_Detection. (无日期). GitHub: [`github.com/sfarrugia15/Ethereum_Fraud_Detection`](https://github.com/sfarrugia15/Ethereum_Fraud_Detection)

+   Steven Farrugia, Joshua Ellul, George Azzopardi, *检测以太坊区块链上的非法账户，《专家系统与应用》*, 第 150 卷, 2020, 113318, ISSN 0957-4174: [`doi.org/10.1016/j.eswa.2020.113318`](https://doi.org/10.1016/j.eswa.2020.113318).

+   *利用区块链数据检测以太坊上的智能合约庞氏骗局*。（2019 年 3 月 18 日）。IEEE Xplore: [`ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8668768`](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8668768)

+   以太坊安全与防诈骗. (无日期). ethereum.org. [`ethereum.org/en/security/`](https://ethereum.org/en/security/)

+   Senilov, I.（2021 年 9 月 27 日）。*接近事务数据中的异常检测*。Medium: [`towardsdatascience.com/approaching-anomaly-detection-in-transactional-data-744d132d524e`](https://towardsdatascience.com/approaching-anomaly-detection-in-transactional-data-744d132d524e)

+   Janiobachmann.（2019 年 7 月 3 日）。*信用欺诈 || 处理不平衡数据集*。Kaggle：您的机器学习与数据科学社区：[`www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets`](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)

+   Jason Brownlee.（2020 年）。*用于不平衡分类的随机过采样与欠采样*。机器学习精通：[`machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/`](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/)

+   **聚类**：一旦我们完成了地址的聚类，就可以构建一个异常检测标识符，使用每个地址所属聚类的参数作为正常性标准。

    Price, W.（2021 年 5 月 28 日）。*聚类以太坊地址*。Medium: [`towardsdatascience.com/clustering-ethereum-addresses-18aeca61919d`](https://towardsdatascience.com/clustering-ethereum-addresses-18aeca61919d)

+   Ethereum_clustering/main.ipynb at master · willprice221/ethereum_clustering.（无日期）。GitHub。 [`github.com/willprice221/ethereum_clustering/blob/master/main.ipynb`](https://github.com/willprice221/ethereum_clustering/blob/master/main.ipynb)

+   故事讲述资源：Insider.（2019 年 6 月 21 日）。*皮克斯制作完美电影的秘密公式 | 电影艺术* [视频]。YouTube: [`www.youtube.com/watch?v=Y34eshkxE5o`](https://www.youtube.com/watch?v=Y34eshkxE5o)
