# 前言

FastAPI 是一个用于构建 Python 3.6 及其后续版本 API 的 Web 框架，基于标准的 Python 类型提示。通过本书，你将能够通过实际示例创建快速且可靠的数据科学 API 后端。

本书从 FastAPI 框架的基础知识和相关的现代 Python 编程概念开始。接着将带你深入了解该框架的各个方面，包括其强大的依赖注入系统，以及如何利用这一系统与数据库进行通信、实现身份验证和集成机器学习模型。稍后，你将学习与测试和部署相关的最佳实践，以运行高质量、强健的应用程序。你还将被介绍到 Python 数据科学包的广泛生态系统。随着学习的深入，你将学会如何使用 FastAPI 在 Python 中构建数据科学应用。书中还演示了如何开发快速高效的机器学习预测后端。为此，我们将通过两个涵盖典型 AI 用例的项目：实时物体检测和文本生成图像。

在完成本书的学习后，你不仅将掌握如何在数据科学项目中实现 Python，还将学会如何利用 FastAPI 设计和维护这些项目，以满足高编程标准。

# 本书读者对象

本书面向那些希望了解 FastAPI 及其生态系统，进而构建数据科学应用的 数据科学家 和 软件开发人员。建议具备基本的数据科学和机器学习概念知识，并了解如何在 Python 中应用这些知识。

# 本书涵盖内容

*第一章*，*Python 开发环境设置*，旨在设置开发环境，使你可以开始使用 Python 和 FastAPI。我们将介绍 Python 社区中常用的各种工具，帮助简化开发过程。

*第二章*，*Python 编程的特点*，向你介绍 Python 编程的具体特点，特别是代码块缩进、控制流语句、异常处理和面向对象范式。我们还将讲解诸如列表推导式和生成器等特性。最后，我们将了解类型提示和异步 I/O 的工作原理。

*第三章*，*使用 FastAPI 开发 RESTful API*，讲解了使用 FastAPI 创建 RESTful API 的基础：路由、参数、请求体验证和响应。我们还将展示如何使用专门的模块和分离的路由器来正确地组织 FastAPI 项目。

*第四章*，*在 FastAPI 中管理 Pydantic 数据模型*，更详细地介绍了如何使用 FastAPI 的底层数据验证库 Pydantic 来定义数据模型。我们将解释如何通过类继承实现相同模型的不同变体，避免重复代码。最后，我们将展示如何在这些模型上实现自定义数据验证逻辑。

*第五章*，*FastAPI 中的依赖注入*，解释了依赖注入是如何工作的，以及我们如何定义自己的依赖关系，以便在不同的路由器和端点之间重用逻辑。

*第六章*，*数据库和异步 ORM*，演示了如何设置与数据库的连接以读取和写入数据。我们将介绍如何使用 SQLAlchemy 与 SQL 数据库异步工作，以及它们如何与 Pydantic 模型交互。最后，我们还将展示如何与 MongoDB（一种 NoSQL 数据库）一起工作。

*第七章*，*在 FastAPI 中管理身份验证和安全性*，展示了如何实现一个基本的身份验证系统，以保护我们的 API 端点并返回经过身份验证的用户的相关数据。我们还将讨论关于 CORS 的最佳实践以及如何防范 CSRF 攻击。

*第八章*，*在 FastAPI 中定义 WebSocket 以进行双向交互通信*，旨在理解 WebSocket 以及如何创建它们并处理 FastAPI 接收到的消息。

*第九章*，*使用 pytest 和 HTTPX 异步测试 API*，展示了如何为我们的 REST API 端点编写测试。

*第十章*，*部署 FastAPI 项目*，介绍了在生产环境中平稳运行 FastAPI 应用程序的常见配置。我们还将探索几种部署选项：PaaS 平台、Docker 和传统服务器配置。

*第十一章*，*Python 中的数据科学介绍*，简要介绍了机器学习，然后介绍了 Python 中数据科学的两个核心库：NumPy 和 pandas。我们还将展示 scikit-learn 库的基础，它是一套用于执行机器学习任务的现成工具。

*第十二章*，*使用 FastAPI 创建高效的预测 API 端点*，展示了如何使用 Joblib 高效地存储训练好的机器学习模型。接着，我们将其集成到 FastAPI 后端，考虑到 FastAPI 内部的一些技术细节，以实现最大性能。最后，我们将展示如何使用 Joblib 缓存结果。

*第十三章*，*使用 Web**S**ockets 和 FastAPI 实现实时目标检测系统*，实现了一个简单的应用程序，用于在浏览器中执行目标检测，背后由 FastAPI WebSocket 和 Hugging Face 库中的预训练计算机视觉模型支持。

*第十四章*，*使用 Stable Diffusion 模型创建分布式文本到图像 AI 系统*，实现了一个能够通过文本提示生成图像的系统，采用流行的 Stable Diffusion 模型。由于这一任务资源消耗大且过程缓慢，我们将学习如何通过工作队列创建一个分布式系统，支持我们的 FastAPI 后端，并在后台执行计算。

*第十五章*，*监控数据科学系统的健康和性能*，涵盖了额外的内容，帮助您构建稳健的、生产就绪的系统。实现这一目标最重要的方面之一是拥有确保系统正常运行所需的所有数据，并尽早发现问题，以便我们采取纠正措施。在本章中，我们将学习如何设置适当的日志记录设施，以及如何实时监控软件的性能和健康状况。

# 为了最大限度地利用本书

在本书中，我们将主要使用 Python 编程语言。第一章将解释如何在操作系统上设置合适的 Python 环境。某些示例还涉及使用 JavaScript 运行网页，因此您需要一个现代浏览器，如 Google Chrome 或 Mozilla Firefox。

在*第十四章*中，我们将运行 Stable Diffusion 模型，这需要一台强大的机器。我们建议使用配备 16 GB RAM 和现代 NVIDIA GPU 的计算机，以便能够生成好看的图像。

| **本书中涉及的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python 3.10+ | Windows、macOS 或 Linux |
| Javascript |

# 下载示例代码文件

您可以从 GitHub 上下载本书的示例代码文件，网址是[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition)。如果代码有更新，它会在 GitHub 仓库中进行更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，您可以在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)查看。快去看看吧！

# 使用的约定

本书中使用了一些文本约定。

`文本中的代码`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账户名。示例：“显然，如果一切正常，我们将获得一个`Person`实例，并能够访问正确解析的字段。”

一段代码的设置如下：

```py

from fastapi import FastAPIapp = FastAPI()
@app.get("/users/{type}/{id}")
async def get_user(type: str, id: int):
    return {"type": type, "id": id}
```

当我们希望引起您注意某段代码时，相关行或项目会设置为粗体：

```py

class PostBase(BaseModel):    title: str
    content: str
    def excerpt(self) -> str:
        return f"{self.content[:140]}..."
```

任何命令行输入或输出将如下所示：

```py

$ http http://localhost:8000/users/abcHTTP/1.1 422 Unprocessable Entity
content-length: 99
content-type: application/json
date: Thu, 10 Nov 2022 08:22:35 GMT
server: uvicorn
```

提示或重要说明

以这种方式出现。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请通过电子邮件联系我们：customercare@packtpub.com，并在邮件主题中注明书名。

**勘误**：尽管我们已尽力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，我们将不胜感激您向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) 并填写表单。

**盗版**：如果您在互联网上发现任何我们作品的非法副本，请提供相关位置地址或网站名称。请通过 copyright@packtpub.com 与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域拥有专业知识，并且有兴趣写书或为书籍贡献内容，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

阅读完 *《使用 FastAPI 构建数据科学应用程序（第二版）》* 后，我们很想听听您的想法！请 [点击这里直接访问亚马逊的评论页面](https://packt.link/r/1-837-63274-X)并分享您的反馈。

您的评论对我们和技术社区非常重要，将帮助我们确保提供优质的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

您喜欢随时随地阅读，但又无法携带纸质书籍吗？

您购买的电子书无法与您选择的设备兼容吗？

不用担心，现在购买每本 Packt 书籍时，您可以免费获得该书的 DRM-free PDF 版本。

在任何地方、任何设备上阅读。直接将您最喜欢的技术书籍中的代码搜索、复制并粘贴到您的应用程序中。

优惠不止于此，您还可以获得独家的折扣、新闻通讯和每天送到您邮箱的精彩免费内容。

按照以下简单步骤即可获得优惠：

1.  扫描二维码或访问下面的链接

![](img/B19528_QR_Free_PDF.jpg)

https://packt.link/free-ebook/9781837632749

1.  提交您的购买证明

1.  就是这样！我们会直接将免费的 PDF 和其他优惠发送到您的邮箱。
