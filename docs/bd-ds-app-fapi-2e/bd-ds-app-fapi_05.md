

# 第四章：在 FastAPI 中管理 Pydantic 数据模型

本章将详细讲解如何使用 Pydantic 定义数据模型，这是 FastAPI 使用的底层数据验证库。我们将解释如何在不重复代码的情况下实现相同模型的变种，得益于类的继承。最后，我们将展示如何将自定义数据验证逻辑实现到 Pydantic 模型中。

本章我们将涵盖以下主要内容：

+   使用 Pydantic 定义模型及其字段类型

+   使用类继承创建模型变种

+   使用 Pydantic 添加自定义数据验证

+   使用 Pydantic 对象

# 技术要求

要运行代码示例，你需要一个 Python 虚拟环境，我们在 *第一章*，*Python 开发环境设置* 中进行了设置。

你可以在专门的 GitHub 仓库中找到本章的所有代码示例，链接为：[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04)。

# 使用 Pydantic 定义模型及其字段类型

Pydantic 是一个强大的库，用于通过 Python 类和类型提示定义数据模型。这种方法使得这些类与静态类型检查完全兼容。此外，由于它是常规的 Python 类，我们可以使用继承，并且还可以定义我们自己的方法来添加自定义逻辑。

在 *第三章*，*使用 FastAPI 开发 RESTful API* 中，你学习了如何使用 Pydantic 定义数据模型的基础：你需要定义一个继承自 `BaseModel` 的类，并将所有字段列为类的属性，每个字段都有一个类型提示来强制其类型。

在本节中，我们将重点关注模型定义，并查看我们在定义字段时可以使用的所有可能性。

## 标准字段类型

我们将从定义标准类型字段开始，这只涉及简单的类型提示。让我们回顾一下一个表示个人信息的简单模型。你可以在以下代码片段中看到它：

chapter04_standard_field_types_01.py

```py

from pydantic import BaseModelclass Person(BaseModel):
    first_name: str
    last_name: str
    age: int
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_01.py)

正如我们所说，你只需要写出字段的名称，并使用预期的类型对其进行类型提示。当然，我们不仅限于标量类型：我们还可以使用复合类型，如列表和元组，或像 datetime 和 enum 这样的类。在下面的示例中，你可以看到一个使用这些更复杂类型的模型：

chapter04_standard_field_types_02.py

```py

from datetime import datefrom enum import Enum
from pydantic import BaseModel, ValidationError
class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    NON_BINARY = "NON_BINARY"
class Person(BaseModel):
    first_name: str
    last_name: str
    gender: Gender
    birthdate: date
    interests: list[str]
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py)

在这个示例中有三点需要注意。

首先，我们使用标准 Python `Enum` 类作为 `gender` 字段的类型。这允许我们指定一组有效值。如果输入的值不在该枚举中，Pydantic 会引发错误，如以下示例所示：

chapter04_standard_field_types_02.py

```py

# Invalid gendertry:
    Person(
        first_name="John",
        last_name="Doe",
        gender="INVALID_VALUE",
        birthdate="1991-01-01",
        interests=["travel", "sports"],
    )
except ValidationError as e:
    print(str(e))
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py)

如果你运行前面的示例，你将得到如下输出：

```py

1 validation error for Persongender
  value is not a valid enumeration member; permitted: 'MALE', 'FEMALE', 'NON_BINARY' (type=type_error.enum; enum_values=[<Gender.MALE: 'MALE'>, <Gender.FEMALE: 'FEMALE'>, <Gender.NON_BINARY: 'NON_BINARY'>])
```

实际上，这正是我们在*第三章*《使用 FastAPI 开发 RESTful API》中所做的，用以限制 `path` 参数的允许值。

然后，我们将 `date` Python 类作为 `birthdate` 字段的类型。Pydantic 能够自动解析以 ISO 格式字符串或时间戳整数给出的日期和时间，并实例化一个合适的 `date` 或 `datetime` 对象。当然，如果解析失败，你也会得到一个错误。你可以在以下示例中进行实验：

chapter04_standard_field_types_02.py

```py

# Invalid birthdatetry:
    Person(
        first_name="John",
        last_name="Doe",
        gender=Gender.MALE,
        birthdate="1991-13-42",
        interests=["travel", "sports"],
    )
except ValidationError as e:
    print(str(e))
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py)

这是输出结果：

```py

1 validation error for Personbirthdate
  invalid date format (type=value_error.date)
```

最后，我们将 `interests` 定义为一个字符串列表。同样，Pydantic 会检查该字段是否是有效的字符串列表。

显然，如果一切正常，我们将得到一个 `Person` 实例，并能够访问正确解析的字段。这就是我们在以下代码片段中展示的内容：

chapter04_standard_field_types_02.py

```py

# Validperson = Person(
    first_name="John",
    last_name="Doe",
    gender=Gender.MALE,
    birthdate="1991-01-01",
    interests=["travel", "sports"],
)
# first_name='John' last_name='Doe' gender=<Gender.MALE: 'MALE'> birthdate=datetime.date(1991, 1, 1) interests=['travel', 'sports']
print(person)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_02.py)

如你所见，这非常强大，我们可以拥有相当复杂的字段类型。但这还不是全部：*字段本身可以是 Pydantic 模型*，允许你拥有子对象！在以下代码示例中，我们扩展了前面的代码片段，添加了一个 `address` 字段：

chapter04_standard_field_types_03.py

```py

class Address(BaseModel):    street_address: str
    postal_code: str
    city: str
    country: str
class Person(BaseModel):
    first_name: str
    last_name: str
    gender: Gender
    birthdate: date
    interests: list[str]
    address: Address
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_03.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_03.py)

我们只需定义另一个 Pydantic 模型，并将其作为类型提示使用。现在，你可以使用已经有效的`Address`实例来实例化`Person`，或者更好的是，使用字典。在这种情况下，Pydantic 会自动解析它并根据地址模型进行验证。

在下面的代码片段中，我们尝试输入一个无效的地址：

chapter04_standard_field_types_03.py

```py

# Invalid addresstry:
    Person(
        first_name="John",
        last_name="Doe",
        gender=Gender.MALE,
        birthdate="1991-01-01",
        interests=["travel", "sports"],
        address={
            "street_address": "12 Squirell Street",
            "postal_code": "424242",
            "city": "Woodtown",
            # Missing country
        },
    )
except ValidationError as e:
    print(str(e))
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_03.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_03.py)

这将生成以下验证错误：

```py

1 validation error for Personaddress -> country
  field required (type=value_error.missing)
```

Pydantic 清晰地显示了子对象中缺失的字段。再次强调，如果一切顺利，我们将获得一个`Person`实例及其关联的`Address`，如下面的代码片段所示：

chapter04_standard_field_types_03.py

```py

# Validperson = Person(
    first_name="John",
    last_name="Doe",
    gender=Gender.MALE,
    birthdate="1991-01-01",
    interests=["travel", "sports"],
    address={
        "street_address": "12 Squirell Street",
        "postal_code": "424242",
        "city": "Woodtown",
        "country": "US",
    },
)
print(person)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_03.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_standard_field_types_03.py)

## 可选字段和默认值

到目前为止，我们假设在实例化模型时，每个字段都必须提供。然而，通常情况下，有些值我们希望是可选的，因为它们可能对每个对象实例并不相关。有时，我们还希望为未指定的字段设置默认值。

正如你可能猜到的，这可以通过`| None`类型注解非常简单地完成，如以下代码片段所示：

chapter04_optional_fields_default_values_01.py

```py

from pydantic import BaseModelclass UserProfile(BaseModel):
    nickname: str
    location: str | None = None
    subscribed_newsletter: bool = True
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_optional_fields_default_values_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_optional_fields_default_values_01.py)

当定义一个字段时，使用`| None`类型提示，它接受`None`值。如上面的代码所示，默认值可以通过将值放在等号后面简单地赋值。

但要小心：*不要为动态类型*（如日期时间）赋予默认值。如果这样做，日期时间实例化只会在模型导入时评估一次。这样一来，你实例化的所有对象都会共享相同的值，而不是每次都生成一个新的值。你可以在以下示例中观察到这种行为：

chapter04_optional_fields_default_values_02.py

```py

class Model(BaseModel):    # Don't do this.
    # This example shows you why it doesn't work.
    d: datetime = datetime.now()
o1 = Model()
print(o1.d)
time.sleep(1)  # Wait for a second
o2 = Model()
print(o2.d)
print(o1.d < o2.d)  # False
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_optional_fields_default_values_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_optional_fields_default_values_02.py)

即使我们在实例化`o1`和`o2`之间等待了 1 秒钟，`d`日期时间仍然是相同的！这意味着日期时间只在类被导入时评估一次。

如果你想要一个默认的列表，比如`l: list[str] = ["a", "b", "c"]`，你也会遇到同样的问题。注意，这不仅仅适用于 Pydantic 模型，所有的 Python 对象都会存在这个问题，所以你应该牢记这一点。

那么，我们该如何赋予动态默认值呢？幸运的是，Pydantic 提供了一个`Field`函数，允许我们为字段设置一些高级选项，其中包括为创建动态值设置工厂。在展示这个之前，我们首先会介绍一下`Field`函数。

在*第三章*《使用 FastAPI 开发 RESTful API》中，我们展示了如何对请求参数应用一些验证，检查一个数字是否在某个范围内，或一个字符串是否匹配正则表达式。实际上，这些选项直接来自 Pydantic！我们可以使用相同的技术对模型的字段进行验证。

为此，我们将使用 Pydantic 的`Field`函数，并将其结果作为字段的默认值。在下面的示例中，我们定义了一个`Person`模型，其中`first_name`和`last_name`是必填字段，必须至少包含三个字符，`age`是一个可选字段，必须是介于`0`和`120`之间的整数。我们在下面的代码片段中展示了该模型的实现：

chapter04_fields_validation_01.py

```py

from pydantic import BaseModel, Field, ValidationErrorclass Person(BaseModel):
    first_name: str = Field(..., min_length=3)
    last_name: str = Field(..., min_length=3)
    age: int | None = Field(None, ge=0, le=120)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_fields_validation_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_fields_validation_01.py)

如你所见，语法与我们之前看到的`Path`、`Query`和`Body`非常相似。第一个位置参数定义了字段的*默认值*。如果字段是必填的，我们使用省略号`...`。然后，关键字参数用于设置字段的选项，包括一些基本的验证。

你可以在官方 Pydantic 文档中查看`Field`接受的所有参数的完整列表，网址为[`pydantic-docs.helpmanual.io/usage/schema/#field-customization`](https://pydantic-docs.helpmanual.io/usage/schema/#field-customization)。

### 动态默认值

在上一节中，我们曾提醒你不要将动态值设置为默认值。幸运的是，Pydantic 在`Field`函数中提供了`default_factory`参数来处理这种用例。这个参数要求你传递一个函数，这个函数将在模型实例化时被调用。因此，每次你创建一个新对象时，生成的对象将在运行时进行评估。你可以在以下示例中看到如何使用它：

chapter04_fields_validation_02.py

```py

from datetime import datetimefrom pydantic import BaseModel, Field
def list_factory():
    return ["a", "b", "c"]
class Model(BaseModel):
    l: list[str] = Field(default_factory=list_factory)
    d: datetime = Field(default_factory=datetime.now)
    l2: list[str] = Field(default_factory=list)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_fields_validation_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_fields_validation_02.py)

你只需将一个函数传递给这个参数。不要在其上放置参数：当你实例化新对象时，Pydantic 会自动调用这个函数。如果你需要使用特定的参数调用一个函数，你需要将它包装成自己的函数，正如我们为`list_factory`所做的那样。

还请注意，默认值所使用的第一个位置参数（如`None`或`...`）在这里完全省略了。这是有道理的：同时使用默认值和工厂是不一致的。如果你将这两个参数一起设置，Pydantic 会抛出错误。

## 使用 Pydantic 类型验证电子邮件地址和 URL

为了方便，Pydantic 提供了一些类，可以作为字段类型来验证一些常见模式，例如电子邮件地址或 URL。

在以下示例中，我们将使用`EmailStr`和`HttpUrl`来验证电子邮件地址和 HTTP URL。

要使`EmailStr`工作，你需要一个可选的依赖项`email-validator`，你可以使用以下命令安装：

```py

(venv)$ pip install email-validator
```

这些类的工作方式与其他类型或类相同：只需将它们作为字段的类型提示使用。你可以在以下代码片段中看到这一点：

chapter04_pydantic_types_01.py

```py

from pydantic import BaseModel, EmailStr, HttpUrl, ValidationErrorclass User(BaseModel):
    email: EmailStr
    website: HttpUrl
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py)

在以下示例中，我们检查电子邮件地址是否被正确验证：

chapter04_pydantic_types_01.py

```py

# Invalid emailtry:
    User(email="jdoe", website="https://www.example.com")
except ValidationError as e:
    print(str(e))
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py)

你将看到以下输出：

```py

1 validation error for Useremail
  value is not a valid email address (type=value_error.email)
```

我们还检查了 URL 是否被正确解析，如下所示：

chapter04_pydantic_types_01.py

```py

# Invalid URLtry:
    User(email="jdoe@example.com", website="jdoe")
except ValidationError as e:
    print(str(e))
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py)

你将看到以下输出：

```py

1 validation error for Userwebsite
  invalid or missing URL scheme (type=value_error.url.scheme)
```

如果你查看下面的有效示例，你会发现 URL 被解析为一个对象，这样你就可以访问它的不同部分，比如协议或主机名：

chapter04_pydantic_types_01.py

```py

# Validuser = User(email="jdoe@example.com", website="https://www.example.com")
# email='jdoe@example.com' website=HttpUrl('https://www.example.com', scheme='https', host='www.example.com', tld='com', host_type='domain')
print(user)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_pydantic_types_01.py)

Pydantic 提供了一套非常丰富的类型，可以帮助你处理各种情况。我们邀请你查阅官方文档中的完整列表：[`pydantic-docs.helpmanual.io/usage/types/#pydantic-types`](https://pydantic-docs.helpmanual.io/usage/types/#pydantic-types)。

现在你对如何通过使用更高级的类型或利用验证功能来细化定义 Pydantic 模型有了更清晰的了解。正如我们所说，这些模型是 FastAPI 的核心，你可能需要为同一个实体定义多个变体，以应对不同的情况。在接下来的部分中，我们将展示如何做到这一点，同时最小化重复。

# 使用类继承创建模型变体

在*第三章*，*使用 FastAPI 开发 RESTful API*中，我们看到一个例子，在这个例子中我们需要定义 Pydantic 模型的两个变体，以便将我们想要存储在后端的数据和我们想要展示给用户的数据分开。这是 FastAPI 中的一个常见模式：你定义一个用于创建的模型，一个用于响应的模型，以及一个用于存储在数据库中的数据模型。

我们在以下示例中展示了这种基本方法：

chapter04_model_inheritance_01.py

```py

from pydantic import BaseModelclass PostCreate(BaseModel):
    title: str
    content: str
class PostRead(BaseModel):
    id: int
    title: str
    content: str
class Post(BaseModel):
    id: int
    title: str
    content: str
    nb_views: int = 0
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_model_inheritance_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_model_inheritance_01.py)

这里我们有三个模型，涵盖了三种情况：

+   `PostCreate`将用于`POST`端点来创建新帖子。我们期望用户提供标题和内容；然而，**标识符**（**ID**）将由数据库自动确定。

+   `PostRead`将用于我们检索帖子数据时。我们当然希望获取它的标题和内容，还希望知道它在数据库中的关联 ID。

+   `Post` 将包含我们希望存储在数据库中的所有数据。在这里，我们还想存储查看次数，但希望将其保密，以便内部进行统计。

你可以看到这里我们重复了很多，特别是 `title` 和 `content` 字段。在包含许多字段和验证选项的大型示例中，这可能会迅速变得难以管理。

避免这种情况的方法是利用模型继承。方法很简单：找出每个变种中共有的字段，并将它们放入一个模型中，作为所有其他模型的基类。然后，你只需从这个模型继承来创建变体，并添加特定的字段。在以下示例中，我们可以看到使用这种方法后的结果：

chapter04_model_inheritance_02.py

```py

from pydantic import BaseModelclass PostBase(BaseModel):
    title: str
    content: str
class PostCreate(PostBase):
    pass
class PostRead(PostBase):
    id: int
class Post(PostBase):
    id: int
    nb_views: int = 0
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_model_inheritance_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_model_inheritance_02.py)

现在，每当你需要为整个实体添加一个字段时，所需要做的就是将其添加到 `PostBase` 模型中，如下所示的代码片段所示。

如果你希望在模型中定义方法，这也是非常方便的。记住，Pydantic 模型是普通的 Python 类，因此你可以根据需要实现尽可能多的方法！

chapter04_model_inheritance_03.py

```py

class PostBase(BaseModel):    title: str
    content: str
    def excerpt(self) -> str:
        return f"{self.content[:140]}..."
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_model_inheritance_03.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_model_inheritance_03.py)

在 `PostBase` 中定义 `excerpt` 方法意味着它将会在每个模型变种中都可用。

虽然这种继承方法不是强制要求的，但它大大有助于防止代码重复，并最终减少错误。我们将在下一节看到，使用自定义验证方法时，它将显得更加有意义。

# 使用 Pydantic 添加自定义数据验证

到目前为止，我们已经看到了如何通过 `Field` 参数或 Pydantic 提供的自定义类型为模型应用基本验证。然而，在一个实际项目中，你可能需要为特定情况添加自定义验证逻辑。Pydantic 允许通过定义 **validators** 来实现这一点，验证方法可以应用于字段级别或对象级别。

## 在字段级别应用验证

这是最常见的情况：为单个字段定义验证规则。要在 Pydantic 中定义验证规则，我们只需要在模型中编写一个静态方法，并用 `validator` 装饰器装饰它。作为提醒，装饰器是一种语法糖，它允许用通用逻辑包装函数或类，而不会影响可读性。

以下示例检查出生日期，确保这个人不超过 120 岁：

chapter04_custom_validation_01.py

```py

from datetime import datefrom pydantic import BaseModel, ValidationError, validator
class Person(BaseModel):
    first_name: str
    last_name: str
    birthdate: date
    @validator("birthdate")
    def valid_birthdate(cls, v: date):
        delta = date.today() - v
        age = delta.days / 365
        if age > 120:
            raise ValueError("You seem a bit too old!")
        return v
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_custom_validation_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_custom_validation_01.py)

如你所见，`validator` 是一个静态类方法（第一个参数，`cls`，是类本身），`v` 参数是要验证的值。它由 `validator` 装饰器装饰，要求第一个参数是需要验证的参数的名称。

Pydantic 对此方法有两个要求，如下所示：

+   如果值根据你的逻辑不合法，你应该抛出一个 `ValueError` 错误并提供明确的错误信息。

+   否则，你应该返回将被赋值给模型的值。请注意，它不需要与输入值相同：你可以根据需要轻松地更改它。这实际上是我们将在接下来的章节中做的，*在 Pydantic 解析之前应用验证*。

## 在对象级别应用验证

很多时候，一个字段的验证依赖于另一个字段——例如，检查密码确认是否与密码匹配，或在某些情况下强制要求某个字段为必填项。为了允许这种验证，我们需要访问整个对象的数据。为此，Pydantic 提供了 `root_validator` 装饰器，如下面的代码示例所示：

chapter04_custom_validation_02.py

```py

from pydantic import BaseModel, EmailStr, ValidationError, root_validatorclass UserRegistration(BaseModel):
    email: EmailStr
    password: str
    password_confirmation: str
    @root_validator()
    def passwords_match(cls, values):
        password = values.get("password")
        password_confirmation = values.get("password_confirmation")
        if password != password_confirmation:
            raise ValueError("Passwords don't match")
        return values
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_custom_validation_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_custom_validation_02.py)

使用此装饰器的方法类似于 `validator` 装饰器。静态类方法与 `values` 参数一起调用，`values` 是一个 *字典*，包含所有字段。这样，你可以获取每个字段并实现你的逻辑。

再次强调，Pydantic 对此方法有两个要求，如下所示：

+   如果根据你的逻辑，值不合法，你应该抛出一个 `ValueError` 错误并提供明确的错误信息。

+   否则，你应该返回一个 `values` 字典，这个字典将被赋值给模型。请注意，你可以根据需要在这个字典中修改某些值。

## 在 Pydantic 解析之前应用验证

默认情况下，验证器在 Pydantic 完成解析工作之后运行。这意味着你得到的值已经符合你指定的字段类型。如果类型不正确，Pydantic 会抛出错误，而不会调用你的验证器。

然而，有时你可能希望提供一些自定义解析逻辑，以允许你转换那些对于所设置类型来说原本不正确的输入值。在这种情况下，你需要在 Pydantic 解析器之前运行你的验证器：这就是 `validator` 中 `pre` 参数的作用。

在下面的示例中，我们展示了如何将一个由逗号分隔的字符串转换为列表：

chapter04_custom_validation_03.py

```py

from pydantic import BaseModel, validatorclass Model(BaseModel):
    values: list[int]
    @validator("values", pre=True)
    def split_string_values(cls, v):
        if isinstance(v, str):
            return v.split(",")
        return v
m = Model(values="1,2,3")
print(m.values)  # [1, 2, 3]
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_custom_validation_03.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_custom_validation_03.py)

你可以看到，在这里我们的验证器首先检查我们是否有一个字符串。如果有，我们将逗号分隔的字符串进行拆分，并返回结果列表；否则，我们直接返回该值。Pydantic 随后会运行它的解析逻辑，因此你仍然可以确保如果 `v` 是无效值，会抛出错误。

# 使用 Pydantic 对象

在使用 FastAPI 开发 API 接口时，你可能会处理大量的 Pydantic 模型实例。接下来，你需要实现逻辑，将这些对象与服务进行连接，比如数据库或机器学习模型。幸运的是，Pydantic 提供了一些方法，使得这个过程变得非常简单。我们将回顾一些开发过程中常用的使用场景。

## 将对象转换为字典

这可能是你在 Pydantic 对象上执行最多的操作：将其转换为一个原始字典，这样你就可以轻松地将其发送到另一个 API，或者例如用在数据库中。你只需在对象实例上调用 `dict` 方法。

以下示例重用了我们在本章的*标准字段类型*部分看到的 `Person` 和 `Address` 模型：

chapter04_working_pydantic_objects_01.py

```py

person = Person(    first_name="John",
    last_name="Doe",
    gender=Gender.MALE,
    birthdate="1991-01-01",
    interests=["travel", "sports"],
    address={
        "street_address": "12 Squirell Street",
        "postal_code": "424242",
        "city": "Woodtown",
        "country": "US",
    },
)
person_dict = person.dict()
print(person_dict["first_name"])  # "John"
print(person_dict["address"]["street_address"])  # "12 Squirell Street"
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_01.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_01.py)

如你所见，调用 `dict` 就足以将所有数据转换为字典。子对象也会递归地被转换：`address` 键指向一个包含地址属性的字典。

有趣的是，`dict` 方法支持一些参数，允许你选择要转换的属性子集。你可以指定你希望包括的属性，或者希望排除的属性，正如下面的代码片段所示：

chapter04_working_pydantic_objects_02.py

```py

person_include = person.dict(include={"first_name", "last_name"})print(person_include)  # {"first_name": "John", "last_name": "Doe"}
person_exclude = person.dict(exclude={"birthdate", "interests"})
print(person_exclude)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_02.py)

`include` 和 `exclude` 参数期望一个集合，集合中包含你希望包含或排除的字段的键。

对于像 `address` 这样的嵌套结构，你也可以使用字典来指定要包含或排除的子字段，以下示例演示了这一点：

chapter04_working_pydantic_objects_02.py

```py

person_nested_include = person.dict(    include={
        "first_name": ...,
        "last_name": ...,
        "address": {"city", "country"},
    }
)
# {"first_name": "John", "last_name": "Doe", "address": {"city": "Woodtown", "country": "US"}}
print(person_nested_include)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_02.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_02.py)

结果的 `address` 字典仅包含城市和国家。请注意，当使用这种语法时，像 `first_name` 和 `last_name` 这样的标量字段必须与省略号 `...` 一起使用。

如果你经常进行某种转换，将其放入一个方法中以便于随时重用是很有用的，以下示例演示了这一点：

chapter04_working_pydantic_objects_03.py

```py

class Person(BaseModel):    first_name: str
    last_name: str
    gender: Gender
    birthdate: date
    interests: list[str]
    address: Address
    def name_dict(self):
        return self.dict(include={"first_name", "last_name"})
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_03.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_03.py)

## 从子类对象创建实例

在 *通过类继承创建模型变体* 这一节中，我们研究了根据具体情况创建特定模型类的常见模式。特别地，你会有一个专门用于创建端点的模型，其中只有创建所需的字段，以及一个包含我们想要存储的所有字段的数据库模型。

让我们再看一下 `Post` 示例：

chapter04_working_pydantic_objects_04.py

```py

class PostBase(BaseModel):    title: str
    content: str
class PostCreate(PostBase):
    pass
class PostRead(PostBase):
    id: int
class Post(PostBase):
    id: int
    nb_views: int = 0
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_04.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_04.py)

假设我们有一个创建端点的 API。在这种情况下，我们会得到一个只有 `title` 和 `content` 的 `PostCreate` 实例。然而，在将其存储到数据库之前，我们需要构建一个适当的 `Post` 实例。

一种方便的做法是同时使用 `dict` 方法和解包语法。在以下示例中，我们使用这种方法实现了一个创建端点：

chapter04_working_pydantic_objects_04.py

```py

@app.post("/posts", status_code=status.HTTP_201_CREATED, response_model=PostRead)async def create(post_create: PostCreate):
    new_id = max(db.posts.keys() or (0,)) + 1
    post = Post(id=new_id, **post_create.dict())
    db.posts[new_id] = post
    return post
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_04.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_04.py)

如你所见，路径操作函数为我们提供了一个有效的`PostCreate`对象。然后，我们想将其转换为`Post`对象。

我们首先确定缺失的`id`属性，这是由数据库提供的。在这里，我们使用基于字典的虚拟数据库，因此我们只需取数据库中已存在的最大键并将其递增。在实际情况下，这个值会由数据库自动确定。

这里最有趣的一行是`Post`实例化。你可以看到，我们首先使用关键字参数分配缺失的字段，然后解包`post_create`的字典表示。提醒一下，`**`在函数调用中的作用是将像`{"title": "Foo", "content": "Bar"}`这样的字典转换为像`title="Foo", content="Bar"`这样的关键字参数。这是一种非常方便和动态的方式，将我们已有的所有字段设置到新的模型中。

请注意，我们还在路径操作装饰器中设置了`response_model`参数。我们在*第三章*，*使用 FastAPI 开发 RESTful API*中解释了这一点，但基本上，它提示 FastAPI 构建一个只包含`PostRead`字段的 JSON 响应，即使我们最终返回的是一个`Post`实例。

## 部分更新实例

在某些情况下，你可能需要允许部分更新。换句话说，你允许最终用户仅向你的 API 发送他们想要更改的字段，并省略不需要更改的字段。这是实现`PATCH`端点的常见方式。

为此，你首先需要一个特殊的 Pydantic 模型，所有字段都标记为可选，这样在缺少某个字段时不会引发错误。让我们看看在我们的`Post`示例中这是什么样的：

chapter04_working_pydantic_objects_05.py

```py

class PostBase(BaseModel):    title: str
    content: str
class PostPartialUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_05.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_05.py)

我们现在能够实现一个端点，接受`Post`字段的子集。由于这是一个更新操作，我们将通过其 ID 从数据库中检索现有的帖子。然后，我们需要找到一种方法，只更新负载中的字段，保持其他字段不变。幸运的是，Pydantic 再次提供了便捷的方法和选项来解决这个问题。

让我们看看如何在以下示例中实现这样的端点：

chapter04_working_pydantic_objects_05.py

```py

@app.patch("/posts/{id}", response_model=PostRead)async def partial_update(id: int, post_update: PostPartialUpdate):
    try:
        post_db = db.posts[id]
        updated_fields = post_update.dict(exclude_unset=True)
        updated_post = post_db.copy(update=updated_fields)
        db.posts[id] = updated_post
        return updated_post
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_05.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter04/chapter04_working_pydantic_objects_05.py)

我们的路径操作函数接受两个参数：`id`属性（来自路径）和`PostPartialUpdate`实例（来自请求体）。

首先要做的是检查这个`id`属性是否存在于数据库中。由于我们使用字典作为虚拟数据库，访问一个不存在的键会引发`KeyError`。如果发生这种情况，我们只需抛出一个`HTTPException`并返回`404`状态码。

现在是有趣的部分：更新现有对象。你可以看到，首先要做的是使用`dict`方法将`PostPartialUpdate`转换为字典。然而，这次我们将`exclude_unset`参数设置为`True`。这样做的效果是，*Pydantic 不会在结果字典中输出未提供的字段*：我们只会得到用户在有效负载中发送的字段。

然后，在我们现有的`post_db`数据库实例上，调用`copy`方法。这个方法是克隆 Pydantic 对象到另一个实例的一个有用方法。这个方法的好处在于它甚至接受一个`update`参数。这个参数期望一个字典，包含所有在复制过程中应该更新的字段：这正是我们想用`updated_fields`字典来做的！

就这样！我们现在有了一个更新过的`post`实例，只有在有效负载中需要的更改。你在使用 FastAPI 开发时，可能会经常使用`exclude_unset`参数和`copy`方法，所以一定要记住它们——它们会让你的工作更轻松！

# 总结

恭喜你！你已经学习了 FastAPI 的另一个重要方面：使用 Pydantic 设计和管理数据模型。现在，你应该对创建模型、应用字段级验证、使用内建选项和类型，以及实现你自己的验证方法有信心。你还了解了如何在对象级别应用验证，检查多个字段之间的一致性。你还学会了如何利用模型继承来避免在定义模型变体时出现代码重复。最后，你学会了如何正确处理 Pydantic 模型实例，从而以高效且可读的方式进行转换和更新。

到现在为止，你几乎已经掌握了 FastAPI 的所有功能。现在有一个最后非常强大的功能等着你去学习：**依赖注入**。这允许你定义自己的逻辑和数值，并将它们直接注入到路径操作函数中，就像你对路径参数和有效负载对象所做的那样，你可以在项目的任何地方重用它们。这是下一章的内容。
