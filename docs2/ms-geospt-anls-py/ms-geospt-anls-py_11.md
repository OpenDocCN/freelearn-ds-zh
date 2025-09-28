# Flask 和 GeoAlchemy2

Python 一直拥有强大的互联网功能。标准库包括 HTTP 处理、STMP 消息和 URL 请求的模型。已经编写了数千个第三方模块来扩展或改进内置的 Web 功能。随着时间的推移，一些模块聚集成了 Python Web 框架——编写用来管理创建和维护复杂和动态网站的代码库。

为了更好地理解如何使用 Python Web 框架以及如何添加地理空间功能，我们将实现 Flask **模型-视图-控制器**（**MVC**）框架。Flask 是一个纯 Python Web 框架，可以与 SQLAlchemy、GeoAlchemy2 和 Jinja2 HTML 模板系统结合使用，以创建具有地理空间功能的网页。

在本章中，你将了解：

+   Flask Web 框架

+   SQLAlchemy 数据库管理

+   GeoAlchemy2

+   使用对象关系映射（ORM）连接到 PostGIS

+   Jinja2 网页模板系统

# Flask 及其组件模块

与 Django 和 GeoDjango（在第十二章中介绍，*GeoDjango*）不同，Flask 不包含电池。相反，它允许根据需要安装多个支持模块。这给了程序员更多的自由，但也使得必须单独安装所需的组件。

我已经为这一章选择了一些模块，这些模块将使我们能够创建一个具有地理空间组件的 Flask 应用程序。以下各节将详细介绍如何设置、安装和利用这些模块来生成网站，使用具有 PostGIS 数据库后端（如第七章中所述，*使用地理数据库进行地理处理*）的演示网站，以及通过基于 Web 的界面执行空间查询的能力。

# 设置

为了确保 Flask 应用程序及其与 PostgreSQL 和 PostGIS 数据库组件的连接能够按需运行，必须安装一些重要的 Python 模块。这些模块将通过`pip`下载和安装，`pip`连接到**Python 包索引**（**PyPI**），这是一个位于[`pypi.python.org/pypi`](https://pypi.python.org/pypi)的在线注册模块仓库。

这些模块包括：

+   Flask，一个纯 Python MVC Web 框架（[`flask.pocoo.org/`](http://flask.pocoo.org/))。

+   Flask-SQLAlchemy，一个数据库 ORM 模块，可以连接到多种数据库后端（[`flask-sqlalchemy.pocoo.org/2.3/`](http://flask-sqlalchemy.pocoo.org/2.3/))。此模块安装 SQLAlchemy。

+   GeoAlchemy2，一个基于 SQLAlchemy 模块和 Postgres/PostGIS 后端（在第七章中介绍，*使用地理数据库进行地理处理*）的 Python 模块，用于允许使用地理空间数据列和 ORM 空间查询（[`geoalchemy-2.readthedocs.io/en/latest/`](https://geoalchemy-2.readthedocs.io/en/latest/))）。

+   Flask-WTForms，一个基于 WTForms（[`wtforms.readthedocs.io/en/latest/`](https://wtforms.readthedocs.io/en/latest/)）构建的 Web 表单模块，允许 Flask 携带每个网页的逻辑并处理输入（[`flask-wtf.readthedocs.io/en/stable/`](https://flask-wtf.readthedocs.io/en/stable/)).

+   SQLAlchemy-Utils，用于管理数据库的创建和删除([`github.com/kvesteri/sqlalchemy-utils/`](https://github.com/kvesteri/sqlalchemy-utils/)).

+   `psycopg2`，用于创建与 PostgreSQL 数据库的连接，并由 SQLAlchemy 模块使用([`initd.org/psycopg/`](http://initd.org/psycopg/)).

+   `pyshapefile`（或`pyshp`），用于读取本例中使用的 shapefiles 并将其添加到数据库表中([`pypi.python.org/pypi/pyshp`](https://pypi.python.org/pypi/pyshp)).

+   最后，`pygeoif`([`pypi.python.org/pypi/pygeoif`](https://pypi.python.org/pypi/pygeoif))用于将 shapefile 二进制编码的数据转换为**已知文本**（**WKT**）编码，以便将几何数据插入数据库。

其他重要的支持模块会自动与 Flask 和前面的模块一起安装，包括 Jinja2 模板系统([`jinja.pocoo.org/`](http://jinja.pocoo.org/))和 Werkzeug **Web 服务器网关接口**（**WSGI**）模块([`werkzeug.pocoo.org/`](http://werkzeug.pocoo.org/))。

# 使用 pip 安装模块

如果您的机器上安装了多个 Python 版本，并且您没有使用`virtualenv`模块的虚拟环境，请确保使用命令行调用的是 Python 3 版本的`pip`，使用`pip -V`选项：

```py
 C:\Python36\Scripts>pip -V
 pip 9.0.1 from c:\python36\lib\site-packages (python 3.6)
```

一旦确定命令行中正在调用正确的`pip`，就可以安装模块。让我们逐一查看所需的`pip`命令以及每个命令预期生成的输出示例。

# 使用 pip 安装 Flask

首先，安装 Flask 模块本身。使用`pip`命令`pip install flask`：

```py
C:\Python36\Scripts>pip install flask
```

`pip`将在 PyPI 上找到 Flask 及其所需依赖项，然后运行包含的`setup.py`指令（或等效指令）来安装模块：

![](img/22176846-a3f1-4e2d-a648-f807ca651a8d.png)

# 使用 pip 通过 pip 安装 Flask-SQLAlchemy

使用命令`pip install flask-sqlalchemy`安装`flask-sqlalchemy`轮文件及其所需依赖项：

```py
C:\Python36\Scripts>pip install flask-sqlalchemy
```

安装命令将在 PyPI 上查找`flask-sqlalchemy`轮文件（`pip`用于安装模块的预构建文件类型）并运行安装过程：

![](img/dcee7274-f35b-44f2-952c-4305f5475864.png)

# 使用 pip 安装 GeoAlchemy2

使用命令`pip install GeoAlchemy2`从 PyPI 调用模块，下载轮文件，并将其安装到 Python 安装的`Lib`/`site-packages`文件夹中：

```py
C:\Python36\Scripts>pip install GeoAlchemy2
```

# 使用 pip 安装 Flask-WTForms 和 WTForms

使用 WTForms 模块和 Flask-WTF 接口，我们可以创建使网页交互的 Web 表单。请使用 `pip install flask-wtf` 命令进行安装：

```py
C:\Python36\Scripts>pip install flask-wtf
```

# 使用 pip 安装 psycopg2

`pscycopg2` 是一个用于连接 PostgreSQL 数据库的 Python 模块。如果尚未安装（见第七章，*使用地理数据库进行地理处理*)，请使用 `pip install psycopg2` 命令进行安装：

```py
C:\Python36\Scripts>pip install psycopg2
```

# 使用 pip 安装 SQLAlchemy-Utils

这些实用程序允许快速创建数据库：

```py
C:\Python36\Scripts>pip install sqlalchemy-utils
```

# 使用 pip 安装 pyshapefile（或 pyshp）

`pyshapefile` 模块可以读取和写入形状文件：

```py
C:\Python36\Scripts>pip install pyshp
```

# 使用 pip 安装 pygeoif

`pygeoif` 模块允许进行地理空间数据格式转换：

```py
C:\Python36\Scripts>pip install pygeoif
```

# 编写 Flask 应用程序

为了探索 Flask 和 GeoAlchemy2 的基础知识，我们将构建一个 Flask Web 应用程序，并使用包含的 Web 服务器在本地进行测试和部署。此 Web 应用程序允许用户查找与全国不同场馆相关的县、州和国会选区。此应用程序将涉及从**美国地质调查局**（**USGS**）数据目录下载数据，并将有视图（处理 Web 请求的 Python 函数）使用 GeoAlchemy2 ORM 进行地理空间查询，并使用 SQLAlchemy ORM 进行表关系搜索。

此应用程序需要使用两个脚本，用于创建数据库和数据库表。这些脚本在进一步说明中会详细介绍，并位于书籍代码包的`Chapter11`文件夹中。最终产品将是一个使用`Leaflet` JavaScript 地图显示基于 ORM 的空间查询结果和关系查询的 Web 应用程序。

# 从数据源下载数据

要开始此项目，让我们从 USGS 数据目录下载数据。此项目将使用四个基于美国的形状文件——NBA 场馆形状文件、州形状文件、国会选区形状文件和县形状文件。

美国地质调查局在此处提供了大量可供下载的 USA 形状文件：[`www.sciencebase.gov/catalog/item/503553b3e4b0d5ec45b0db20`](https://www.sciencebase.gov/catalog/item/503553b3e4b0d5ec45b0db20)。

# 县、地区、州和场馆形状文件

`US_County_Boundaries` 数据是来自 USGS 数据目录的多边形形状文件，可在以下地址找到：[`www.sciencebase.gov/catalog/item/4f4e4a2ee4b07f02db615738`](https://www.sciencebase.gov/catalog/item/4f4e4a2ee4b07f02db615738)。

点击图片中显示的下载 zip 链接。将文件解压缩到项目文件夹中（例如，`C:\GeospatialPy3\Chapter11`），以便在整个章节中访问：

![图片](img/4b1862b4-a369-41a9-80dd-7d08f08e80a3.png)

`Arenas_NBA` 形状文件在此处可用：[`www.sciencebase.gov/catalog/item/4f4e4a0ae4b07f02db5fb54d`](https://www.sciencebase.gov/catalog/item/4f4e4a0ae4b07f02db5fb54d)。

`Congressional_Districts` 形状文件在此处可用：[`www.sciencebase.gov/catalog/item/4f4e4a06e4b07f02db5f8b58`](https://www.sciencebase.gov/catalog/item/4f4e4a06e4b07f02db5f8b58).

`US_States` 形状文件在此处可用：[`www.sciencebase.gov/catalog/item/4f4e4783e4b07f02db4837ce`](https://www.sciencebase.gov/catalog/item/4f4e4783e4b07f02db4837ce).

这些形状文件不是最新的（例如，Nets 场地仍然列在新泽西州，而不是在布鲁克林），但这里我们探索的是应用程序技术（以及它们如何处理几何数据类型），而不是数据本身，因此忽略数据的时效性。

# 创建数据库和数据表

要创建我们的数据库和将存储应用程序数据的表，我们将使用 SQLAlchemy 和 GeoAlchemy2 类和方法。以下代码位于名为 `Chapter11_0.py` 的脚本中。此代码将允许我们连接到 PostgreSQL 数据服务器以创建将构成网络应用程序后端的数据库和数据表。导入这些库：

```py
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database,
                             drop_database
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from sqlalchemy.ext.declarative import declarative_base
```

使用 `create_engine` 函数和连接字符串格式连接数据库服务器以生成和查询数据表，如下所示：

```py
conn_string = '{DBtype}://{user}:{pword}@{instancehost}:{port}/{database}'
engine = create_engine(conn_string, echo=True)
```

连接字符串在所有 Python 数据库模块中都会使用。它们通常包括对 **关系数据库管理系统**（**RDBMS**）类型的指定、用户名、密码、实例主机（即数据库服务器的 IP 地址或本地机器上安装的数据库服务器的 `localhost`）、可选端口号和数据库名称。例如，连接字符串可能看起来像这样：

```py
connstring = 'postgresql://postgres:bond007@localhost:5432/chapter11'
engine = create_engine(connstring, echo=True)
```

在此示例中，`postgresql` 是 RDBMS 类型，`postgres` 是用户，`bond007` 是密码，`localhost` 是实例主机，`5432` 是端口（也是 PostgreSQL 安装程序的默认端口；如果安装时未更改端口，则可以将其省略在连接字符串中），而 `chapter11` 是数据库名称。`echo=True` 语句用于将数据库交互的日志生成到标准输出窗口。要关闭这些消息，请将 echo 值更改为 `False`。

关于此模式的更详细解释可在此处找到：[`docs.sqlalchemy.org/en/latest/core/engines.html`](http://docs.sqlalchemy.org/en/latest/core/engines.html).

对于我们的数据库，我们可以使用以下格式。将 `{user}` 和 `{pword}`（包括括号）替换为您的 PostgreSQL 服务器用户名和密码：

```py
conn_string ='postgresql://{user}:{pword}@localhost:5432/chapter11'
engine = create_engine(conn_string, echo=True)
```

如果连接字符串有效，`create_engine` 函数将返回一个对象到 `engine` 变量，该变量将用于在整个脚本中执行数据库交互。

注释中的代码（`#drop_database(engine.url)`）已被注释掉，但如果需要使用脚本删除并重新创建数据库，可以取消注释。它调用 SQLAlchemy 的 `create_engine` 的 `url` 属性，该属性是连接字符串的引用：

```py
# Uncomment the line below if you need to recreate the database.
#drop_database(engine.url)
```

数据库及其包含的数据表是在一个依赖于 `database_exists` 函数的 `if not` 条件语句中创建的。如果条件返回 `True`（表示数据库不存在），则将 `engine` 变量传递给 `create_database` 函数：

```py
# Check to ensure that the database doesn't exist
# If it doesn't, create it and generate the PostGIS extention and tables
if not database_exists(engine.url):
    create_database(engine.url)
```

# 将 PostGIS 扩展表添加到新数据库

在 `create_database` 函数下方，我们需要使用 `engine.connect` 函数连接到数据库，并将 SQL 语句直接传递给数据库。这个 SQL 语句 `("CREATE EXTENSION postgis*"*)` 启用新数据库中的空间列和查询：

```py
    # Create a direct connection to the database using the engine.
    # This will allow the new database to use the PostGIS extension.
    conn = engine.connect()
    conn.execute("commit")
    try:
         conn.execute("CREATE EXTENSION postgis")
     except Exception as e:
         print(e)
         print("extension postgis already exists")
     conn.close()
```

这里使用 `try`/`except` 块以处理数据库已经启用空间的情况。检查 `print` 语句的输出以确保没有发生其他异常。

# 定义数据库表

在 Python MVC 网络框架的世界中，数据库表是**模型**。网站使用这些表来存储数据，它们由 Python 类生成并建模。这些类从包含大部分数据库管理代码的超类中继承或继承预写的功能，使我们只需使用基本数据类型（如字符串和整数）以及高级类（如几何形状）来定义表的列。

这些由类定义的表可以在多个关系数据库管理系统（RDBMS）中生成，而无需重新编写模型代码。虽然 GeoAlchemy2 只能在 PostgreSQL/PostGIS 上运行，但 SQLAlchemy 模型可以用于在包括 SQL Server、Oracle、Postgres、MySQL 等多种数据库中生成表。

# 声明性基础

对于 SQLAlchemy 数据库类，一个名为 `declarative_base` 的基类允许继承数据库方法和属性（这就是 SQLAlchemy 的超类魔法所在，处理多个 SQL 版本的数据库 SQL 语句，从而简化了写入任何关系数据库管理系统所需的代码）：

```py
    # Define the model Base
    Base = declarative_base()
```

# 数据库表模型类

一旦调用或实例化了基类，就可以将其传递给模型类。这些类，像所有 Python 类一样，可以包括函数、属性和方法，这些对于在类内部处理数据非常有用。在本章中，模型不包含任何内部函数，而是仅定义列。

在这里探索 SQLAlchemy 模型和它们的内部函数：[`docs.sqlalchemy.org/en/latest/orm/tutorial.html`](http://docs.sqlalchemy.org/en/latest/orm/tutorial.html)。

# 表属性

在 RDBMS 数据库中生成的数据表名称将对应于模型类的`__tablename__`属性。每个表的主键用于关系和查询，必须使用`primary_key`关键字定义。`Column`类和`String`、`Float`和`Integer`类型类从 SQLAlchemy 调用，并用于定义要在底层 RDBMS 中生成的表列（从而允许程序员避免为每个主要 RDBMS 使用的各种 SQL 编写`CREATE TABLE`语句）。

例如，`Arena`类将用于管理一个具有四个列的表——一个`String` `name`字段，两个`Float`字段（`longitude`和`latitude`），以及一个 SRID 或 EPSG 空间参考系统 ID 为`4326`的`POINT`几何类型，对应于 WGS 1984 坐标系（[`spatialreference.org/ref/epsg/wgs-84/`](http://spatialreference.org/ref/epsg/wgs-84/))）：

```py
    # Define the Arena class, which will model the Arena database table
    class Arena(Base):
        __tablename__ = 'arena'
       id = Column(Integer, primary_key=True)
       name = Column(String)
       longitude = Column(Float)
       latitude = Column(Float)
       geom = Column(Geometry(geometry_type='POINT', srid=4326))
```

与`Arena`类类似，以下类使用一个`String` `name`列。对于几何类型，它们也使用 SRID `4326`，但它们使用`MULTIPOLYGON`几何类型来存储用于建模这些地理的复杂多边形几何。对于具有关系的表，例如`County`、`District`和`State`类，还有用于管理表关系和表之间查询的特殊类。

这些特殊类包括`ForeignKey`类和`relationship`函数。`ForeignKey`类传递一个`id`参数，并将其传递给`Column`类，将子行与父行关联。`relationship`函数允许双向查询。`backref`关键字生成一个函数，该函数实例化连接表模型的一个实例：

```py
    # Define the County class
    class County(Base):
        __tablename__ = 'county'
        id = Column(Integer, primary_key=True)
        name = Column(String)
        state_id = Column(Integer, ForeignKey('state.id'))
        state_ref = relationship("State",backref='county')
        geom =   Column(Geometry(geometry_type='MULTIPOLYGON',srid=4326))

    # Define the District class
    class District(Base):
        __tablename__ = 'district'
        id = Column(Integer, primary_key=True)
        district = Column(String)
        name = Column(String)
        state_id = Column(Integer, ForeignKey('state.id'))
        state_ref = relationship("State",backref='district')
        geom = Column(Geometry(geometry_type='MULTIPOLYGON',srid=4326))

```

`County`类和`District`类将与`State`类建立`relationship`关系，允许调用`State`类的会话查询。这种`relationship`使得查找县或国会选区所在的美国州变得容易。`state_id`列建立`relationship`，而`state_ref`字段引用父`State`类。对于`State`类，县和区有自己的`backref`引用，允许父`State`类访问相关的县/区：

```py
    # Define the State class
    class State(Base):
        __tablename__ = 'state'
        id = Column(Integer, primary_key=True)
        name = Column(String)
        statefips = Column(String)
        stpostal = Column(String)
        counties = relationship('County', backref='state')
        districts = relationship('District', backref='state')
        geom =         
        Column(Geometry(geometry_type='MULTIPOLYGON',srid=4326))
```

# 创建表

实际生成表时，有两种方法可以使用。表模型类有一个内部的`__table__`方法，它有一个`create`函数，可以用来单独创建每个表。还有一个`drop`函数，可以用来删除表。

在脚本中，我们使用`try`*/*`except`块来生成表。如果发生异常（即，如果表已存在），则删除表并重新创建。以下是一个示例的`State`表创建语句：

```py
    # Generate the State table from the State class.
    # If it already exists, drop it and regenerate it
    try:
        State.__table__.create(engine)
    except:
        State.__table__.drop(engine)
        State.__table__.create(engine)
```

或者，可以使用`Base`方法的`metadata`和其`create_all`函数从定义的类生成所有数据库表：

```py
    Base.metadata.create_all(engine)
```

# 向新的数据表中插入数据

一旦数据库创建完成，并在数据库中定义和创建了数据库表，就可以添加数据。第二个脚本`Chapter11_1.py`将用于查找和读取下载的形状文件中的数据，并使用`for`循环读取数据并将其写入相应的数据库表。将使用 SQLAlchemy 会话管理器来查询和提交数据到表中。

# 导入所需的模块

为了处理和导入数据，将使用几个新模块。`pyshapefile`模块（或作为 shapefile 导入的`pyshp`）用于连接到形状文件，并读取它们包含的几何形状和属性数据。`pygeoif`模块是一个纯 Python 模块，实现了名为`geo_interface`的协议。

此协议允许 Python 对象级别的地理空间数据自省，例如，它将地理空间数据格式转换为 Python 对象。它将被用于将存储在二进制中的形状文件几何形状转换为可以插入数据库的 WKT 几何形状，使用 GeoAlchemy2 ORM：

```py
# The pyshapefile module is used to read shapefiles and
# the pygeoif module is used to convert between geometry types
import shapefile
import pygeoif
```

关于`geo_interface`协议的更多讨论请见此处：[`gist.github.com/sgillies/2217756`](https://gist.github.com/sgillies/2217756)。

要连接到数据库和表，导入 SQLAlchemy ORM 和其他 SQLAlchemy 函数：

```py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
```

要将数据添加到数据库表的几何列中，将使用 GeoAlchemy2 的`Geometry`数据类型：

```py
# The Geometry columns from GeoAlchemy2 extend the SQLAlchemy ORM 
from geoalchemy2 import Geometry
```

要使脚本能够找到下载的形状文件，请使用`Tkinter`模块及其`filedialog`方法，因为它内置在 Python 中，且与操作系统无关：

```py
# The built-in Tkinter GUI module allows for file dialogs
from tkinter import filedialog
from tkinter import Tk
```

将再次使用 SQLAlchemy 的`create_engine`函数创建数据库连接。本节还使用会话管理器生成一个`session`，并将其绑定到连接数据库的`engine`变量：

```py
# Connect to the database called chapter11 using SQLAlchemy functions
conn_string = 'postgresql://postgres:password@localhost/chapter11'
engine = create_engine(conn_string)
Session = sessionmaker(bind=engine)
session = Session()
```

`session`将允许对这些表进行查询和提交（即写入数据库）进行管理。我们将在`for`循环内查询数据库表，以创建县、区和州之间的数据库关系。

数据库表模型再次在脚本中定义，从`declarative_base`类派生。这些类定义将与上一个脚本中的定义相匹配。

# 定位和读取形状文件

要创建允许用户搜索和定位形状文件的文件对话框，请实例化 Tkinter 的`Tk`类并将其分配给变量`root`。`Tk`类创建了一个不必要的迷你控制台窗口，因此使用`root.withdraw`方法将其关闭：

```py
# Initiate the Tkinter module and withdraw the console it generates
root = Tk()
root.withdraw()
```

文件对话框是通过`filedialog.askopenfilename`方法生成的。该方法接受多个参数，包括文件对话框窗口的`title`、初始目录以及在使用文件对话框时应可见的文件扩展名。以下是一个示例代码，用于`选择竞技场形状文件`对话框：

```py
# Navigate to the Arena shapefile using the Tkinter file dialog
root.arenafile = filedialog.askopenfilename(initialdir = "/",
                              title = "Select Arena Shapefile",
                              filetypes = (("shapefiles","*.shp"),
                              ("all files", "*.*")))
```

在脚本中，这一过程会为每个下载的 shapefile 重复一次。在使用文件对话框后，找到的每个 shapefile 都会将字符串类型的文件路径传递给`root`变量，并将文件路径保存在一个属性中。

# 访问 shapefile 数据

要访问 shapefile 中的数据，通过将相应的文件路径属性传递给`Reader`类来调用`pyshp` `Reader`类。实例化的类将具有`records`和`shapes`方法，分别允许访问 shapefile 的属性数据和几何数据：

```py
# Read the Arena shapefile using the Reader class of the pyshp module
import shapefile
arena_shapefile = shapefile.Reader(root.arenafile)
arena_shapes = arena_shapefile.shapes()
arena_records = arena_shapefile.records()
```

一旦数据被读取并分配给可迭代变量，它们就可以使用`for`循环进行迭代。因为使用`pyshp` `Reader` `records`方法访问的数据与使用`shapes`方法访问的数据相对应，所以使用`enumerate`函数生成的循环计数器用于匹配当前记录和由`shapes`方法生成的几何形状列表中相应的几何数据索引。

对于`Arena` shapefile 的几何形状，`Reader`的`shapes`方法返回一个包含坐标对的列表。由于`Arena`类几何列是`POINT`数据类型，数据可以使用`POINT(X Y)` WKT 模板写入数据库表。SRID（`4326`）按照 GeoAlchemy2 扩展 WKT（EWKT）的要求包含在字符串的开头。

在这里了解更多关于 GeoAlcheym2 ORM 的信息：[`geoalchemy-2.readthedocs.io/en/0.4/orm_tutorial.html`](http://geoalchemy-2.readthedocs.io/en/0.4/orm_tutorial.html)。

在每次循环中，都会实例化一个新的`Arena`类并将其分配给变量`arena`。`name`字段从位于索引`6`的`Reader` `record`数据项中提取出来，并分配给`arena`变量，同时几何数据从`arena_shapes`数据项的`count`（即当前循环次数）中提取出来，并分配给`Arena`列的`arena.longitude`和`arena.latitude`。

这些坐标随后被传递给字符串`format`方法以格式化 EWKT 模板，并分配给`arena.geom`属性。一旦`arena`行的数据被分配，它就会使用`session.add`添加到会话中。最后，使用会话的`commit`方法将数据写入数据库：

```py
# Iterate through the Arena data read from the shapefile
for count, record in enumerate(arena_records):
    arena = Arena()
    arena.name = record[6]
    print(arena.name)
    point = arena_shapes[count].points[0]
    arena.longitude = point[0]
    arena.latitude = point[1]
    arena.geom = 'SRID=4326;POINT({0} {1})'.format(point[0],     
    point[1])
 session.add(arena)
session.commit()
```

对于`State`类（以及`County`和`District`类），使用索引从属性数据中提取名称、**联邦信息处理标准**（**FIPS**）代码和邮政编码缩写。`pygeoif`用于首先将几何形状转换为`pygeoif` `MultiPolygon`格式，然后转换为 WKT，并将其传递给字符串模板，作为 EWKT 写入`geom`字段：

```py
# Iterate through the State data read from the shapefile
for count, record in enumerate(state_records):
    state = State()
    state.name = record[1]
    state.statefips = record[0]
    state.stpostal = record[2]
    state_geo = state_shapes[count]
 gshape =     
    pygeoif.MultiPolygon(pygeoif.geometry.as_shape(state_geo))
 state.geom = 'SRID=4326;{0}'.format(gshape.wkt)
    session.add(state)
    if count % 10 == 0:
 session.commit()
session.commit()
```

由于州几何数据量较大，它们每`10`次循环提交一次到数据库。最后的`commit`捕获任何剩余的数据。

# 使用查询

对于`District`和`County`数据表，添加了一个最后的细节，即查询新添加的`state`数据以通过 FIPS 代码找到相关州。通过使用`session.query`查询`State`类，并使用`filter_by`方法（将来自区域记录的 FIPS 代码作为`filter`参数传递）过滤州的数据，然后指定应使用`first`结果，可以调用正确的`state`。使用变量的`id`字段来填充区域的`state_id`列以创建`relationship`：

```py
# This uses the STFIPS data to query the State table and find the state
for count, record in enumerate(district_records):
    district = District()
    district.district = record[0]
    district.name = record[1]
    state = session.query(State).filter_by(statefips=record[4]).first()
    district.state_id = state.id
    dist_geo = district_shapes[count]

   gshape=pygeoif.MultiPolygon(pygeoif.geometry.as_shape(dist_geo))
    district.geom = 'SRID=4326;{0}'.format(gshape.wkt)
    session.add(district)
    if count % 50 == 0:
        session.commit()
session.commit()
```

`County`表同样被循环遍历，并且还包括一个`State`查询。查看脚本以查看完整的代码。一旦所有数据都已写入数据表，`close`会话并`dispose`连接引擎：

```py
 session.close()
 engine.dispose()
```

# Flask 应用程序的组件

现在后台数据库和表已经创建并加载数据，表之间的关系也已经建模和生成，是时候编写创建 Flask 应用程序的脚本了。这些脚本将包含视图、模型和表单，以处理 Web 请求、查询数据库并返回 HTTP 响应。

该 Web 应用程序被称为 Arena 应用程序，因为它在下拉列表中列出存储在`arena`表中的所有 NBA`arenas`，并允许用户在地图上显示位置，同时显示包含关于`arena`信息的`popup`，这些信息来自空间查询和表关系。

Web 开发的 MVC 方法允许将 Web 应用程序的必要组件分离。这些组件包括数据库模型（前面描述的 SQLAlchemy 模型）、接受应用程序输入的 Web 表单，以及用于路由请求的控制器对象。组件的分离反映在单独的脚本中。使每个组件独立，可以更容易地调整而不会影响应用程序的其他组件。

数据库模型将包含在名为`models.py`的脚本中，以及所需的模块导入。Web 表单（创建网页组件，如下拉列表和输入字段的 Python 类）将包含在名为`forms.py`的脚本中。所有视图，包括 URL 端点和处理这些 URL 的 Web 请求，都将包含在名为`views.py`的脚本中。

控制器是从`Flask`类生成的对象，并分配给名为`app`的变量。每个 Web 应用程序的 URL 端点都使用`app.route`定义，并关联一个 Python 函数（视图），该函数包含处理 Web 请求并返回 HTTP 响应的逻辑。控制器用于将 Web 请求路由到正确的 URL 端点，并且可以区分`GET`和`POST HTTP`请求。它是在`views.py`脚本中创建的。

使用 HTML 模板来展示 Web 请求的处理结果。通过使用 Jinja2 模板系统，包含在 Web 表单中的数据将被传递到 HTML 模板，并以完整的网页形式发送回请求的 Web 浏览器。该应用程序的模板包含指向 JavaScript 库的链接，包括`Leaflet`，这使得网页能够在网页内展示地图。

# 文件夹结构和控制器对象

为了包含应用程序的各个独立组件，建议使用特定的文件夹结构。这将允许组件在需要时相互引用，同时仍然保持独立性。对组件某一部分的调整不应需要分离组件的重构（至少尽可能少）。

Arena 应用程序包含在一个名为`arenaapp`的文件夹中：

![图片](img/5c34b72a-3640-46e3-a156-2c8f1d69668b.png)

在`arenaapp`文件夹中有一个名为`app.py`的脚本和一个名为`application`的文件夹：

![图片](img/c947cb29-fc9f-4f4d-afaf-5bc9298ebf38.png)

`app.py`脚本从`application`导入`app`控制器对象，并调用`app.run`方法来启动 Web 应用程序：

```py
from application import app
app.run()
```

通过添加 Python 的`__init__.py`脚本，使得`application`文件夹可导入，并允许`app`访问组件脚本内的代码成为可能。这个特殊的脚本会告诉 Python 可执行文件该文件夹是一个模块：

![图片](img/f34e301a-ca18-4df3-b8ea-b28e672ab8b1.png)

在`__init__.py`中，定义并配置了`app`对象。`app`对象包含一个配置字典，允许 Web 应用程序连接到后端（`'SQLALCHEMY_DATABASE_URI'`）并执行会话管理和加密。虽然我们已将配置设置包含在这个脚本中，但请注意，较大的应用程序会将配置设置分离到单独的`config.py`脚本中：

```py
import flask
app = flask.Flask(__name__)
conn_string = 'postgresql://postgres:password@localhost:5432/chapter11'
app.config['SQLALCHEMY_DATABASE_URI'] = conn_string 
app.config['SECRET_KEY'] = "SECRET_KEY"
app.config['DEBUG'] = True
import application.views
```

为了便于调试应用程序，已将`DEBUG`配置设置为`True`。在生产环境中将其设置为`False`。将`'SECRET KEY'`替换为您的自己的密钥。

在这里了解更多关于配置 Flask Web 应用程序的信息：[`flask.pocoo.org/docs/latest/config/`](http://flask.pocoo.org/docs/latest/config/)。

# 模型

对于 Arena 应用程序，一个名为`models.py`的脚本包含了将用于应用程序的模型。如前所述，这些模型是包含数据库列定义的 Python 类，并且可以拥有内部函数来处理数据。我们的简化模型仅使用 SQLAlchemy 和 GeoAlchemy2 类包含数据列定义。

要连接到数据库，需要导入`app`对象。这使得应用程序配置变量，包括`app.config['SQLALCHEMY_DATABASE_URI']`（存储数据库连接字符串），对 SQLAlchemy 的`create_engine`函数可用：

```py
from application import app
# The database connections and session management are managed with SQLAlchemy functions
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()
```

为了简洁起见，我在这里省略了模型类定义的详细说明，因为它们之前已经解释过了。请在内置的`arenaapp`/`application`文件夹的`models.py`脚本中查找它们。

# 表单

Web 表单用于 Web 应用程序中接受用户数据并将其发送到服务器进行验证和处理。要生成所需的表单（例如下拉列表、输入字段甚至隐藏其内容的密码字段），使用 Flask-WTF 模块和 WTForms 模块。这些模块包含类，使得创建表单组件并确保输入到其中的数据对该字段有效成为可能。

对于我们的简单应用，只创建了一个表单。`ArenaForm`表单继承自`FlaskFor`类，并包含一个名为`description`的属性和字段`selections`。该字段是一个`SelectField`，它将在网页上创建一个下拉列表。它需要一个描述字符串，并使用关键字`choices`生成下拉列表中的选项列表。由于下拉列表的成员将在视图中动态生成（如下所述），因此在这里传递给`choices`关键字的是一个空列表：

```py
from flask_wtf import FlaskForm
from wtforms import SelectField
class ArenaForm(FlaskForm):
    description  = "Use the dropdown to select an arena."
    selections = SelectField('Select an Arena',choices=[])
```

其他字段类，例如`TextField`、`BooleanField`、`StringField`、`FloatField`、`PasswordField`以及许多其他字段，都可通过 WTForms 库用于实现复杂的 Web 应用程序。此外，由于它们是 Python 对象，表单可以动态更新以包含其他数据属性，正如我们进一步探讨时将看到的。

# 视图

Flask 视图是 Python 函数，当与`app`控制器对象及其`app.route` URL 定义配对时，允许我们编写 Python 代码来接受 Web 请求、处理它并返回响应。它们是 Web 应用程序的核心，使得将网页及其表单连接到数据库及其表成为可能。

要创建视图，我们将导入所有应用程序组件以及许多 Flask 函数。表单和模型从它们各自的脚本中导入，以及`app`对象：

```py
from application import app
from flask import render_template,jsonify, redirect, url_for, request
from .forms import * 
from .models import *
```

对于 Arena 应用，我们定义了两个视图，它们创建了两个应用程序 URL 端点。第一个视图`home`仅用于将请求重定向到 IP 地址根。使用 Flask 函数`redirect`和`url_for`，任何发送到`root`地址的 Web 请求都将被重定向到`arenas`视图：

```py
@app.route('/', methods=["GET"])
def home():
 return redirect(url_for('arenas'))
```

第二个视图`arenas`更为复杂。它接受`GET`和`POST`请求方法。根据`request`方法的不同，处理和返回的数据将不同，尽管它们都依赖于存储在`application`/`templates`文件夹（所有 Flask HTML 模板都存储在此处）中的模板`index.html`。以下是完整的视图：

```py
@app.route('/arenas', methods=["GET","POST"])
def arenas():
    form = ArenaForm(request.form)
    arenas = session.query(Arena).all()
    form.selections.choices = [(arena.id, 
                                arena.name) for arena in arenas]
    form.popup = "Select an Arena"
    form.latitude = 38.89517
    form.longitude = -77.03682    
    if request.method == "POST":
        arena_id = form.selections.data
        arena = session.query(Arena).get(arena_id)
        form.longitude = round(arena.longitude,4)
        form.latitude = round(arena.latitude,4)
        county=session.query(County).filter(
                        County.geom.ST_Contains(arena.geom)).first()
        if county != None:
            district=session.query(District).filter(
                       District.geom.ST_Intersects(arena.geom)).first()
            state = county.state_ref
            form.popup = """The {0} is located at {4}, {5}, which is in 
            {1} County, {3}, and in {3} Congressional District         
            {2}.""".format(arena.name,county.name, district.district, 
            state.name,                                          
            form.longitude, form.latitude)

        else:
                 form.popup = """The county, district, and state could                                                             
                 not be located using point in polygon analysis"""

        return render_template('index.html',form=form)
    return render_template('index.html',form=form)
```

# 视图剖析

视图的 URL 是 `http://{localhost}/arenas`***。*** 使用一个特殊的 Python 对象，称为 **装饰器**（例如 `@app.route`*）*，我们可以将我们想要使用的 URL 与将接受和处理请求处理的函数连接起来。函数和 URL 不需要具有相同的名称，尽管它们通常是这样的：

```py
@app.route('/arenas', methods=["GET","POST"])
def arenas():
```

# 使用表单

在装饰器和函数声明之下，调用了来自 `forms.py` 的 `ArenaForm`，并将 `request.form` 函数作为参数传递。这为 `ArenaForm` 添加了功能，并允许它根据需要访问请求的自身参数。

一旦 `ArenaForm` 对象传递给变量 `form`，它就可以用数据填充。这些数据将来自对 `Arena` 模型的 SQLAlchemy 会话 `query`。此查询请求 `Arena` 表的所有数据行，并使用 `all` 方法（而不是 `filter_by` 方法，后者将限制返回的行数）将其传递给变量 `arenas`。

由于 `ArenaForm` 的 `selections` 字段目前为空，我们将使用列表推导来遍历名为 `arenas` 的列表中包含的 `arena` 对象，将它们的 `id` 和 `name` 字段添加到列表内部的元组中。这填充了下拉列表，并确保列表中的每个选择项都有一个值（`id`）和一个标签（`name`）：

```py
    form = ArenaForm(request.form)
    arenas = session.query(Arena).all()
    form.selections.choices = [(arena.id, 
                                arena.name) for arena in arenas]
    form.popup = "Select an Arena"
    form.latitude = 38.89517
    form.longitude = -77.03682  
```

在填充选择项选项后，表单中添加了三个新属性——`popup`、`latitude` 和 `longitude`。最初，这些只是占位符，并不来自 `arena` 数据。然而，一旦网络应用程序运行，并且用户从下拉列表中选择 `arenas`，这些占位符值将被来自 `arenas` 表和其他表的查询得到的数据所替换。

# 评估请求方法

下一个行是一个使用 `request.method` 属性的 `if` 条件语句，用于检查 HTTP 请求方法是否为 `POST`：

```py
if request.method == "POST":
```

由于对 URL `arenas` 的初始请求是一个 `GET` 请求，代码最初评估 `if` 条件为 `False`，跳过视图底部缩进的代码部分，以返回模板 `index.html` 和现在已填充的 `form`：

```py
return render_template('index.html',form=form)
```

此函数使用 `render_template` 函数返回名为 `index.html` 的模板，并将填充的 `ArenaForm` 变量 `form` 传递到模板中，使得 Jinja2 模板系统能够生成完整的网页并发送到请求的网页浏览器。模板中所有双括号变量都填充了来自 `form` 的对应数据（例如，选择项被添加到下拉列表中）。

# POST 请求

如果用户从列表中选择一个`arena`并点击查找数据按钮，HTML 表单向视图发出`POST`请求。当`if`条件解析为`True`时，视图通过生成`arena`位置坐标对和自定义`popup`来处理请求，而不是使用默认坐标对和`popup`值：

```py
  if request.method == "POST":
       arena_id = form.selections.data
       arena = session.query(Arena).get(arena_id)
       form.longitude = round(arena.longitude,4)
       form.latitude = round(arena.latitude,4)
```

使用`form.selections.data`属性来检索从列表中选择的`arena`的`id`，并将其传递给一个名为`arena_id`的变量。然后使用这个`id`通过 SQLAlchemy ORM 的`get`方法查询数据库。`form.longitude`和`form.latitude`字段可以从查询返回的`arena`对象的字段中填充。

# 空间查询

为了找到县和国会选区，使用了两种 PostGIS 空间分析技术——`ST_Contains`和`ST_Intersects`。第一个查询确定`arena`是否包含在`county`内；如果不是，结果为空（或在 Python 中为`None`）：

```py
        county=session.query(County).filter(
                        County.geom.ST_Contains(arena.geom)).first()
        if county != None:
            district=session.query(District).filter(
                       District.geom.ST_Intersects(arena.geom)).first()
```

虽然`ST_Contains`可以用于两个查询，但我想要展示 GeoAlchemy2 ORM 在使用`Geometry`列时允许访问所有 PostGIS 函数。这些搜索结合了 SQLAlchemy 的`filter`方法和 GeoAlchemy2 ORM，使得基于空间分析返回查询结果成为可能。

# 关系查询

如果县查询成功，则执行区查询，然后使用关系属性（`state_ref`）来找到`county`所在的`state`：

```py
        state = county.state_ref
```

在`County`、`District`和`State`模型定义中建立的双向关系使得这一点成为可能。这个`state`对象是`State`模型类的一个成员，可以用来检索`state`的`name`。

为了创建自定义的`popup`，使用字符串模板格式化来填充`popup`，以描述请求的`arena`的具体信息。结果被分配给变量`form.popup`。

最后，填充后的`form`再次传递给`index.html`模板，但这次它包含所选`arena`的数据：

```py
    return render_template('index.html',form=form)
```

这是 The Oracle Arena 的应用程序查询结果截图：

![图片](img/54a1006e-af1e-4437-98ca-3af63d58a38c.png)

# 网页地图模板

在`index.html`模板中，通过双括号变量访问`form`数据。这些变量可以位于 JavaScript 或 HTML 中。在这个例子中，`form.latitude`和`form.longitude`变量位于定义地图初始中心点的地图 JavaScript 中：

```py
 var themap = L.map('map').setView([{{form.latitude}},                                                   {{form.longitude}}], 13);
```

为了在请求的`arena`位置创建带有自定义`popup`的`marker`，添加位置坐标和`popup`字段：

```py
  L.marker([{{form.latitude}},{{form.longitude}}]).addTo(themap)
    .bindPopup("{{form.popup}}").openPopup();
```

为了使`POST`请求成为可能，一个带有`POST`方法的 HTML 表单包含了`form.description`和`form.selection`（下拉列表）属性。HTML 表单的按钮在按下时生成`POST`请求：

```py
  <form method="post" class="form">
    <h3>{{form.description}}</h3>
    {{form.selections(class_='form-control',placeholder="")}}
    <br>
    <input type="submit" value="Find Data">
  </form>
```

# 在本地运行网络应用程序

要在本地运行应用程序，我们可以使用 Python 可执行文件调用位于`arenaapp`文件夹中的`app.py`脚本。打开命令行并传递脚本参数：

```py
C:\Python36>python C:\GeospatialPy3\Chapter11\Scripts\arenaapp\app.py
```

在 Web 服务器上运行此应用程序超出了本章的范围，但它涉及配置一个带有 WSGI 处理器的 Web 服务器，以便允许 Web 请求由 Python 可执行文件和`app.py`处理。对于 Apache Web 服务器，`mod_wsgi`模块很受欢迎。对于使用**Internet Information Services**（**IIS**）的 Windows 服务器，`wfastcgi`模块非常有用，并且可以从 Microsoft Web 平台安装程序中获取。

在这里了解更多关于 Apache 和`mod_wsgi`模块的信息：[`flask.pocoo.org/docs/latest/deploying/mod_wsgi/`](http://flask.pocoo.org/docs/latest/deploying/mod_wsgi/)。

对于 IIS，以下安装说明非常有用：[`netdot.co/2015/03/09/flask-on-iis/`](http://netdot.co/2015/03/09/flask-on-iis/)。

# 摘要

在本章中，我们学习了如何使用 Flask MVC 网络框架以及一些可用的组件模块，这些模块增加了额外的功能。这些模块包括 SQLAlchemy ORM、用于地理空间查询的 GeoAlchemy2 ORM、用于处理 Web 数据的 WTForms 以及用于创建 Web 页面模板的 Jinja2 模板系统。我们创建了数据库表，添加了数据表和表关系，并创建了一个利用地理空间和关系查询生成动态网页的 Web 应用程序。

一个有趣的挑战是在这里审查的代码基础上，探索为 Arena 应用程序添加编辑功能，允许用户在数据过时的情况下将`arenas`移动到正确的位置。探索 GeoAlchemy2 ORM 文档以获取更多高级功能。

在下一章中，我们将回顾一个类似 MVC 网络框架 Django 及其 GeoDjango 空间组件。Django 以更多电池包的哲学解决了与 Web 应用程序固有的相同问题，但与 Flask 相比，模块选择自由度较低。
