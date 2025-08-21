# 第十章：Storm 与 Redis、Elasticsearch 和 HBase 的集成

在上一章中，我们介绍了 Apache Hadoop 及其各个组件的概述。我们还介绍了 Storm-YARN 的概述，并介绍了如何在 Apache Hadoop 上部署 Storm-YARN。

在本章中，我们将解释如何将 Storm 与其他数据库集成以存储数据，以及如何在 Storm bolt 中使用 Esper 来支持窗口操作。

以下是本章将要涵盖的关键点：

+   将 Storm 与 HBase 集成

+   将 Storm 与 Redis 集成

+   将 Storm 与 Elasticsearch 集成

+   将 Storm 与 Esper 集成以执行窗口操作

# 将 Storm 与 HBase 集成

如前几章所述，Storm 用于实时数据处理。然而，在大多数情况下，您需要将处理后的数据存储在数据存储中，以便将存储的数据用于进一步的批量分析，并在存储的数据上执行批量分析查询。本节解释了如何将 Storm 处理的数据存储在 HBase 中。

在实施之前，我想简要介绍一下 HBase 是什么。HBase 是一个 NoSQL、多维、稀疏、水平可扩展的数据库，模型类似于**Google** **BigTable**。HBase 建立在 Hadoop 之上，这意味着它依赖于 Hadoop，并与 MapReduce 框架很好地集成。Hadoop 为 HBase 提供以下好处：

+   在通用硬件上运行的分布式数据存储

+   容错

我们假设您已经在系统上安装并运行了 HBase。您可以参考[HBase 安装文章](https://hbase.apache.org/cygwin.html)。

我们将创建一个示例 Storm 拓扑，演示如何使用以下步骤将 Storm 处理的数据存储到 HBase：

1.  使用`com.stormadvance`作为组 ID 和`stormhbase`作为 artifact ID 创建一个 Maven 项目。

1.  将以下依赖项和存储库添加到`pom.xml`文件中：

```scala
    <repositories> 
        <repository> 
            <id>clojars.org</id> 
            <url>http://clojars.org/repo</url> 
        </repository> 
    </repositories> 
    <dependencies> 
        <dependency> 
            <groupId>org.apache.storm</groupId> 
            <artifactId>storm-core</artifactId> 
            <version>1.0.2</version> 
            <scope>provided</scope> 
        </dependency> 
        <dependency> 
            <groupId>org.apache.hadoop</groupId> 
            <artifactId>hadoop-core</artifactId> 
            <version>1.1.1</version> 
        </dependency> 
        <dependency> 
            <groupId>org.slf4j</groupId> 
            <artifactId>slf4j-api</artifactId> 
            <version>1.7.7</version> 
        </dependency> 

        <dependency> 
            <groupId>org.apache.hbase</groupId> 
            <artifactId>hbase</artifactId> 
            <version>0.94.5</version> 
            <exclusions> 
                <exclusion> 
                    <artifactId>zookeeper</artifactId> 
                    <groupId>org.apache.zookeeper</groupId> 
                </exclusion> 

            </exclusions> 
        </dependency> 

        <dependency> 
            <groupId>junit</groupId> 
            <artifactId>junit</artifactId> 
            <version>4.10</version> 
        </dependency> 
    </dependencies> 
    <build> 
        <plugins> 
            <plugin> 
                <groupId>org.apache.maven.plugins</groupId> 
                <artifactId>maven-compiler-plugin</artifactId> 
                <version>2.5.1</version> 
                <configuration> 
                    <source>1.6</source> 
                    <target>1.6</target> 
                </configuration> 
            </plugin> 
            <plugin> 
                <artifactId>maven-assembly-plugin</artifactId> 
                <version>2.2.1</version> 
                <configuration> 
                    <descriptorRefs> 
                        <descriptorRef>jar-
                        with-dependencies</descriptorRef> 
                    </descriptorRefs> 
                    <archive> 
                        <manifest> 
                            <mainClass /> 
                        </manifest> 
                    </archive> 
                </configuration> 
                <executions> 
                    <execution> 
                        <id>make-assembly</id> 
                        <phase>package</phase> 
                        <goals> 
                            <goal>single</goal> 
                        </goals> 
                    </execution> 
                </executions> 
            </plugin> 
        </plugins> 
    </build> 
```

1.  在`com.stormadvance.stormhbase`包中创建一个`HBaseOperations`类。`HBaseOperations`类包含两个方法：

+   `createTable(String tableName, List<String> ColumnFamilies)`: 此方法将表名和 HBase 列族列表作为输入，以在 HBase 中创建表。

+   `insert(Map<String, Map<String, Object>> record, String rowId)`: 此方法将记录及其`rowID`参数作为输入，并将输入记录插入 HBase。以下是输入记录的结构：

```scala
{  

  "columnfamily1":  
  {  
    "column1":"abc",  
    "column2":"pqr"  
  },  
  "columnfamily2":  
  {  
    "column3":"bc",  
    "column4":"jkl"  
  }  
}  
```

这里，`columnfamily1`和`columnfamily2`是 HBase 列族的名称，`column1`、`column2`、`column3`和`column4`是列的名称。

`rowId`参数是 HBase 表行键，用于唯一标识 HBase 中的每条记录。

`HBaseOperations`类的源代码如下：

```scala
public class HBaseOperations implements Serializable{ 

    private static final long serialVersionUID = 1L; 

    // Instance of Hadoop Cofiguration class 
    Configuration conf = new Configuration(); 

    HTable hTable = null; 

    public HBaseOperations(String tableName, List<String> ColumnFamilies, 
            List<String> zookeeperIPs, int zkPort) { 
        conf = HBaseConfiguration.create(); 
        StringBuffer zookeeperIP = new StringBuffer(); 
        // Set the zookeeper nodes 
        for (String zookeeper : zookeeperIPs) { 
            zookeeperIP.append(zookeeper).append(","); 
        } 
        zookeeperIP.deleteCharAt(zookeeperIP.length() - 1); 

        conf.set("hbase.zookeeper.quorum", zookeeperIP.toString()); 

        // Set the zookeeper client port 
        conf.setInt("hbase.zookeeper.property.clientPort", zkPort); 
        // call the createTable method to create a table into HBase. 
        createTable(tableName, ColumnFamilies); 
        try { 
            // initilaize the HTable.  
            hTable = new HTable(conf, tableName); 
        } catch (IOException e) { 
            throw new RuntimeException("Error occure while creating instance of HTable class : " + e); 
        } 
    } 

    /** 
     * This method create a table into HBase 
     *  
     * @param tableName 
     *            Name of the HBase table 
     * @param ColumnFamilies 
     *            List of column famallies 
     *  
     */ 
    public void createTable(String tableName, List<String> ColumnFamilies) { 
        HBaseAdmin admin = null; 
        try { 
            admin = new HBaseAdmin(conf); 
            // Set the input table in HTableDescriptor 
            HTableDescriptor tableDescriptor = new HTableDescriptor( 
                    Bytes.toBytes(tableName)); 
            for (String columnFamaliy : ColumnFamilies) { 
                HColumnDescriptor columnDescriptor = new HColumnDescriptor( 
                        columnFamaliy); 
                // add all the HColumnDescriptor into HTableDescriptor 
                tableDescriptor.addFamily(columnDescriptor); 
            } 
            /* execute the creaetTable(HTableDescriptor tableDescriptor) of HBaseAdmin 
             * class to createTable into HBase. 
            */  
            admin.createTable(tableDescriptor); 
            admin.close(); 

        }catch (TableExistsException tableExistsException) { 
            System.out.println("Table already exist : " + tableName); 
            if(admin != null) { 
                try { 
                admin.close();  
                } catch (IOException ioException) { 
                    System.out.println("Error occure while closing the HBaseAdmin connection : " + ioException); 
                } 
            } 

        }catch (MasterNotRunningException e) { 
            throw new RuntimeException("HBase master not running, table creation failed : "); 
        } catch (ZooKeeperConnectionException e) { 
            throw new RuntimeException("Zookeeper not running, table creation failed : "); 
        } catch (IOException e) { 
            throw new RuntimeException("IO error, table creation failed : "); 
        } 
    } 

    /** 
     * This method insert the input record into HBase. 
     *  
     * @param record 
     *            input record 
     * @param rowId 
     *            unique id to identify each record uniquely. 
     */ 
    public void insert(Map<String, Map<String, Object>> record, String rowId) { 
        try { 
        Put put = new Put(Bytes.toBytes(rowId));         
        for (String cf : record.keySet()) { 
            for (String column: record.get(cf).keySet()) { 
                put.add(Bytes.toBytes(cf), Bytes.toBytes(column), Bytes.toBytes(record.get(cf).get(column).toString())); 
            }  
        } 
        hTable.put(put); 
        }catch (Exception e) { 
            throw new RuntimeException("Error occure while storing record into HBase"); 
        } 

    } 

    public static void main(String[] args) { 
        List<String> cFs = new ArrayList<String>(); 
        cFs.add("cf1"); 
        cFs.add("cf2"); 

        List<String> zks = new ArrayList<String>(); 
        zks.add("192.168.41.122"); 
        Map<String, Map<String, Object>> record = new HashMap<String, Map<String,Object>>(); 

        Map<String, Object> cf1 = new HashMap<String, Object>(); 
        cf1.put("aa", "1"); 

        Map<String, Object> cf2 = new HashMap<String, Object>(); 
        cf2.put("bb", "1"); 

        record.put("cf1", cf1); 
        record.put("cf2", cf2); 

        HBaseOperations hbaseOperations = new HBaseOperations("tableName", cFs, zks, 2181); 
        hbaseOperations.insert(record, UUID.randomUUID().toString()); 

    } 
} 
```

1.  在`com.stormadvance.stormhbase`包中创建一个`SampleSpout`类。此类生成随机记录并将其传递给拓扑中的下一个操作（bolt）。以下是`SampleSpout`类生成的记录的格式：

```scala
["john","watson","abc"]  
```

`SampleSpout`类的源代码如下：

```scala
public class SampleSpout extends BaseRichSpout { 
    private static final long serialVersionUID = 1L; 
    private SpoutOutputCollector spoutOutputCollector; 

    private static final Map<Integer, String> FIRSTNAMEMAP = new HashMap<Integer, String>(); 
    static { 
        FIRSTNAMEMAP.put(0, "john"); 
        FIRSTNAMEMAP.put(1, "nick"); 
        FIRSTNAMEMAP.put(2, "mick"); 
        FIRSTNAMEMAP.put(3, "tom"); 
        FIRSTNAMEMAP.put(4, "jerry"); 
    } 

    private static final Map<Integer, String> LASTNAME = new HashMap<Integer, String>(); 
    static { 
        LASTNAME.put(0, "anderson"); 
        LASTNAME.put(1, "watson"); 
        LASTNAME.put(2, "ponting"); 
        LASTNAME.put(3, "dravid"); 
        LASTNAME.put(4, "lara"); 
    } 

    private static final Map<Integer, String> COMPANYNAME = new HashMap<Integer, String>(); 
    static { 
        COMPANYNAME.put(0, "abc"); 
        COMPANYNAME.put(1, "dfg"); 
        COMPANYNAME.put(2, "pqr"); 
        COMPANYNAME.put(3, "ecd"); 
        COMPANYNAME.put(4, "awe"); 
    } 

    public void open(Map conf, TopologyContext context, 
            SpoutOutputCollector spoutOutputCollector) { 
        // Open the spout 
        this.spoutOutputCollector = spoutOutputCollector; 
    } 

    public void nextTuple() { 
        // Storm cluster repeatedly call this method to emit the continuous // 
        // stream of tuples. 
        final Random rand = new Random(); 
        // generate the random number from 0 to 4\. 
        int randomNumber = rand.nextInt(5); 
        spoutOutputCollector.emit (new Values(FIRSTNAMEMAP.get(randomNumber),LASTNAME.get(randomNumber),COMPANYNAME.get(randomNumber))); 
    } 

    public void declareOutputFields(OutputFieldsDeclarer declarer) { 
        // emits the field  firstName , lastName and companyName. 
        declarer.declare(new Fields("firstName","lastName","companyName")); 
    } 
} 

```

1.  在`com.stormadvance.stormhbase`包中创建一个`StormHBaseBolt`类。此 bolt 接收`SampleSpout`发出的元组，然后调用`HBaseOperations`类的`insert()`方法将记录插入 HBase。`StormHBaseBolt`类的源代码如下：

```scala
public class StormHBaseBolt implements IBasicBolt { 

    private static final long serialVersionUID = 2L; 
    private HBaseOperations hbaseOperations; 
    private String tableName; 
    private List<String> columnFamilies; 
    private List<String> zookeeperIPs; 
    private int zkPort; 
    /** 
     * Constructor of StormHBaseBolt class 
     *  
     * @param tableName 
     *            HBaseTableNam 
     * @param columnFamilies 
     *            List of column families 
     * @param zookeeperIPs 
     *            List of zookeeper nodes 
     * @param zkPort 
     *            Zookeeper client port 
     */ 
    public StormHBaseBolt(String tableName, List<String> columnFamilies, 
            List<String> zookeeperIPs, int zkPort) { 
        this.tableName =tableName; 
        this.columnFamilies = columnFamilies; 
        this.zookeeperIPs = zookeeperIPs; 
        this.zkPort = zkPort; 

    } 

    public void execute(Tuple input, BasicOutputCollector collector) { 
        Map<String, Map<String, Object>> record = new HashMap<String, Map<String, Object>>(); 
        Map<String, Object> personalMap = new HashMap<String, Object>(); 
        // "firstName","lastName","companyName") 
        personalMap.put("firstName", input.getValueByField("firstName")); 
        personalMap.put("lastName", input.getValueByField("lastName")); 

        Map<String, Object> companyMap = new HashMap<String, Object>(); 
        companyMap.put("companyName", input.getValueByField("companyName")); 

        record.put("personal", personalMap); 
        record.put("company", companyMap); 
        // call the inset method of HBaseOperations class to insert record into 
        // HBase 
        hbaseOperations.insert(record, UUID.randomUUID().toString()); 
    } 

    public void declareOutputFields(OutputFieldsDeclarer declarer) { 

    } 

    public Map<String, Object> getComponentConfiguration() { 
        // TODO Auto-generated method stub 
        return null; 
    } 

    public void prepare(Map stormConf, TopologyContext context) { 
        // create the instance of HBaseOperations class 
        hbaseOperations = new HBaseOperations(tableName, columnFamilies, 
                zookeeperIPs, zkPort); 
    } 

    public void cleanup() { 
        // TODO Auto-generated method stub 

    } 

} 
```

`StormHBaseBolt`类的构造函数以 HBase 表名、列族列表、ZooKeeper IP 地址和 ZooKeeper 端口作为参数，并设置类级变量。`StormHBaseBolt`类的`prepare()`方法将创建`HBaseOperatons`类的实例。

`StormHBaseBolt`类的`execute()`方法以输入元组作为参数，并将其转换为 HBase 结构格式。它还使用`java.util.UUID`类生成 HBase 行 ID。

1.  在`com.stormadvance.stormhbase`包中创建一个`Topology`类。这个类创建`spout`和`bolt`类的实例，并使用`TopologyBuilder`类将它们链接在一起。以下是主类的实现：

```scala
public class Topology {
    public static void main(String[] args) throws AlreadyAliveException, 
            InvalidTopologyException { 
        TopologyBuilder builder = new TopologyBuilder(); 

        List<String> zks = new ArrayList<String>(); 
        zks.add("127.0.0.1"); 

        List<String> cFs = new ArrayList<String>(); 
        cFs.add("personal"); 
        cFs.add("company"); 

        // set the spout class 
        builder.setSpout("spout", new SampleSpout(), 2); 
        // set the bolt class 
        builder.setBolt("bolt", new StormHBaseBolt("user", cFs, zks, 2181), 2) 
                .shuffleGrouping("spout"); 
        Config conf = new Config(); 
        conf.setDebug(true); 
        // create an instance of LocalCluster class for 
        // executing topology in local mode. 
        LocalCluster cluster = new LocalCluster(); 

        // LearningStormTopolgy is the name of submitted topology. 
        cluster.submitTopology("StormHBaseTopology", conf, 
                builder.createTopology()); 
        try { 
            Thread.sleep(60000); 
        } catch (Exception exception) { 
            System.out.println("Thread interrupted exception : " + exception); 
        } 
        System.out.println("Stopped Called : "); 
        // kill the LearningStormTopology 
        cluster.killTopology("StormHBaseTopology"); 
        // shutdown the storm test cluster 
        cluster.shutdown(); 

    } 
} 

```

在本节中，我们介绍了如何将 Storm 与 NoSQL 数据库 HBase 集成。在下一节中，我们将介绍如何将 Storm 与 Redis 集成。

# 将 Storm 与 Redis 集成

Redis 是一个键值数据存储。键值可以是字符串、列表、集合、哈希等。它非常快，因为整个数据集存储在内存中。以下是安装 Redis 的步骤：

1.  首先，您需要安装`make`、`gcc`和`cc`来编译 Redis 代码，使用以下命令：

```scala
    sudo yum -y install make gcc cc
```

1.  下载、解压并制作 Redis，并使用以下命令将其复制到`/usr/local/bin`：

```scala
    cd /home/$USER 
    Here, $USER is the name of the Linux user. 
    http://download.redis.io/releases/redis-2.6.16.tar.gz 
    tar -xvf redis-2.6.16.tar.gz 
    cd redis-2.6.16 
    make 
    sudo cp src/redis-server /usr/local/bin 
    sudo cp src/redis-cli /usr/local/bin
```

1.  执行以下命令将 Redis 设置为服务：

```scala
    sudo mkdir -p /etc/redis 
    sudo mkdir -p /var/redis 
    cd /home/$USER/redis-2.6.16/ 
    sudo cp utils/redis_init_script /etc/init.d/redis 
    wget https://bitbucket.org/ptylr/public-stuff/raw/41d5c8e87ce6adb3 
    4aa16cd571c3f04fb4d5e7ac/etc/init.d/redis 
    sudo cp redis /etc/init.d/redis 
    cd /home/$USER/redis-2.6.16/ 
    sudo cp redis.conf /etc/redis/redis.conf
```

1.  现在，运行以下命令将服务添加到`chkconfig`，设置为自动启动，并实际启动服务：

```scala
    chkconfig --add redis 
    chkconfig redis on 
    service redis start
```

1.  使用以下命令检查 Redis 的安装情况：

```scala
    redis-cli ping
```

如果测试命令的结果是`PONG`，则安装已成功。

我们假设您已经启动并运行了 Redis 服务。

接下来，我们将创建一个示例 Storm 拓扑，以解释如何将 Storm 处理的数据存储在 Redis 中。

1.  使用`com.stormadvance`作为`groupID`，`stormredis`作为`artifactID`创建一个 Maven 项目。

1.  在`pom.xml`文件中添加以下依赖和存储库：

```scala
<repositories> 
        <repository> 
            <id>central</id> 
            <name>Maven Central</name> 
            <url>http://repo1.maven.org/maven2/</url> 
        </repository> 
        <repository> 
            <id>cloudera-repo</id> 
            <name>Cloudera CDH</name> 
            <url>https://repository.cloudera.com/artifactory/cloudera-
            repos/</url> 
        </repository> 
        <repository> 
            <id>clojars.org</id> 
            <url>http://clojars.org/repo</url> 
        </repository> 
    </repositories> 
    <dependencies> 
        <dependency> 
            <groupId>storm</groupId> 
            <artifactId>storm</artifactId> 
            <version>0.9.0.1</version> 
        </dependency> 
                <dependency> 
            <groupId>com.fasterxml.jackson.core</groupId> 
            <artifactId>jackson-core</artifactId> 
            <version>2.1.1</version> 
        </dependency> 

        <dependency> 
            <groupId>com.fasterxml.jackson.core</groupId> 
            <artifactId>jackson-databind</artifactId> 
            <version>2.1.1</version> 
        </dependency> 
        <dependency> 
            <groupId>junit</groupId> 
            <artifactId>junit</artifactId> 
            <version>3.8.1</version> 
            <scope>test</scope> 
        </dependency> 
        <dependency> 
            <groupId>redis.clients</groupId> 
            <artifactId>jedis</artifactId> 
            <version>2.4.2</version> 
        </dependency> 
    </dependencies> 
```

1.  在`com.stormadvance.stormredis`包中创建一个`RedisOperations`类。`RedisOperations`类包含以下方法：

+   `insert(Map<String, Object> record, String id)`: 此方法接受记录和 ID 作为输入，并将输入记录插入 Redis。在`insert()`方法中，我们将首先使用 Jackson 库将记录序列化为字符串，然后将序列化记录存储到 Redis 中。每个记录必须具有唯一的 ID，因为它用于从 Redis 中检索记录。

以下是`RedisOperations`类的源代码：

```scala
public class RedisOperations implements Serializable { 

    private static final long serialVersionUID = 1L; 
    Jedis jedis = null; 

    public RedisOperations(String redisIP, int port) { 
        // Connecting to Redis on localhost 
        jedis = new Jedis(redisIP, port); 
    } 

    public void insert(Map<String, Object> record, String id) { 
        try { 
            jedis.set(id, new ObjectMapper().writeValueAsString(record)); 
        } catch (Exception e) { 
            System.out.println("Record not persist into datastore : "); 
        } 
    } 
} 
```

我们将使用在*将 Storm 与 HBase 集成*部分中创建的相同的`SampleSpout`类。

1.  在`com.stormadvance.stormredis`包中创建一个`StormRedisBolt`类。这个 bolt 接收`SampleSpout`类发出的元组，将它们转换为 Redis 结构，然后调用`RedisOperations`类的`insert()`方法将记录插入 Redis。以下是`StormRedisBolt`类的源代码：

```scala
    public class StormRedisBolt implements IBasicBolt{ 

    private static final long serialVersionUID = 2L; 
    private RedisOperations redisOperations = null; 
    private String redisIP = null; 
    private int port; 
    public StormRedisBolt(String redisIP, int port) { 
        this.redisIP = redisIP; 
        this.port = port; 
    } 

    public void execute(Tuple input, BasicOutputCollector collector) { 
        Map<String, Object> record = new HashMap<String, Object>(); 
        //"firstName","lastName","companyName") 
        record.put("firstName", input.getValueByField("firstName")); 
        record.put("lastName", input.getValueByField("lastName")); 
        record.put("companyName", input.getValueByField("companyName")); 
        redisOperations.insert(record, UUID.randomUUID().toString()); 
    } 

    public void declareOutputFields(OutputFieldsDeclarer declarer) { 

    } 

    public Map<String, Object> getComponentConfiguration() { 
        return null; 
    } 

    public void prepare(Map stormConf, TopologyContext context) { 
        redisOperations = new RedisOperations(this.redisIP, this.port); 
    } 

    public void cleanup() { 

    } 

} 

```

在`StormRedisBolt`类中，我们使用`java.util.UUID`类生成 Redis 键。

1.  在`com.stormadvance.stormredis`包中创建一个`Topology`类。这个类创建`spout`和`bolt`类的实例，并使用`TopologyBuilder`类将它们链接在一起。以下是主类的实现：

```scala
public class Topology { 
    public static void main(String[] args) throws AlreadyAliveException, 
            InvalidTopologyException { 
        TopologyBuilder builder = new TopologyBuilder(); 

        List<String> zks = new ArrayList<String>(); 
        zks.add("192.168.41.122"); 

        List<String> cFs = new ArrayList<String>(); 
        cFs.add("personal"); 
        cFs.add("company"); 

        // set the spout class 
        builder.setSpout("spout", new SampleSpout(), 2); 
        // set the bolt class 
        builder.setBolt("bolt", new StormRedisBolt("192.168.41.122",2181), 2).shuffleGrouping("spout"); 

        Config conf = new Config(); 
        conf.setDebug(true); 
        // create an instance of LocalCluster class for 
        // executing topology in local mode. 
        LocalCluster cluster = new LocalCluster(); 

        // LearningStormTopolgy is the name of submitted topology. 
        cluster.submitTopology("StormRedisTopology", conf, 
                builder.createTopology()); 
        try { 
            Thread.sleep(10000); 
        } catch (Exception exception) { 
            System.out.println("Thread interrupted exception : " + exception); 
        } 
        // kill the LearningStormTopology 
        cluster.killTopology("StormRedisTopology"); 
        // shutdown the storm test cluster 
        cluster.shutdown(); 
} 
} 
```

在本节中，我们介绍了 Redis 的安装以及如何将 Storm 与 Redis 集成。

# 将 Storm 与 Elasticsearch 集成

在本节中，我们将介绍如何将 Storm 与 Elasticsearch 集成。Elasticsearch 是一个基于 Lucene 开发的开源分布式搜索引擎平台。它提供了多租户能力、全文搜索引擎功能。

我们假设 Elasticsearch 正在您的环境中运行。如果您没有任何正在运行的 Elasticsearch 集群，请参考[`www.elastic.co/guide/en/elasticsearch/reference/2.3/_installation.html`](https://www.elastic.co/guide/en/elasticsearch/reference/2.3/_installation.html)在任何一个框中安装 Elasticsearch。按照以下步骤将 Storm 与 Elasticsearch 集成：

1.  使用`com.stormadvance`作为`groupID`，`storm_elasticsearch`作为`artifactID`创建一个 Maven 项目。

1.  在`pom.xml`文件中添加以下依赖和存储库：

```scala
<dependencies> 
        <dependency> 
            <groupId>org.elasticsearch</groupId> 
            <artifactId>elasticsearch</artifactId> 
            <version>2.4.4</version> 
        </dependency> 
        <dependency> 
            <groupId>junit</groupId> 
            <artifactId>junit</artifactId> 
            <version>3.8.1</version> 
            <scope>test</scope> 
        </dependency> 
        <dependency> 
            <groupId>org.apache.storm</groupId> 
            <artifactId>storm-core</artifactId> 
            <version>1.0.2</version> 
            <scope>provided</scope> 
        </dependency> 
    </dependencies> 
```

1.  在`com.stormadvance.storm_elasticsearch`包中创建一个`ElasticSearchOperation`类。`ElasticSearchOperation`类包含以下方法：

+   `insert(Map<String, Object> data, String indexName, String indexMapping, String indexId)`: 这个方法以记录数据、`indexName`、`indexMapping`和`indexId`作为输入，并将输入记录插入 Elasticsearch。

以下是`ElasticSearchOperation`类的源代码：

```scala
public class ElasticSearchOperation { 

    private TransportClient client; 

    public ElasticSearchOperation(List<String> esNodes) throws Exception { 
        try { 
            Settings settings = Settings.settingsBuilder() 
                    .put("cluster.name", "elasticsearch").build(); 
            client = TransportClient.builder().settings(settings).build(); 
            for (String esNode : esNodes) { 
                client.addTransportAddress(new InetSocketTransportAddress( 
                        InetAddress.getByName(esNode), 9300)); 
            } 

        } catch (Exception e) { 
            throw e; 
        } 

    } 

    public void insert(Map<String, Object> data, String indexName, String indexMapping, String indexId) { 
        client.prepareIndex(indexName, indexMapping, indexId) 
                .setSource(data).get(); 
    } 

    public static void main(String[] s){ 
        try{ 
            List<String> esNodes = new ArrayList<String>(); 
            esNodes.add("127.0.0.1"); 
            ElasticSearchOperation elasticSearchOperation  = new ElasticSearchOperation(esNodes); 
            Map<String, Object> data = new HashMap<String, Object>(); 
            data.put("name", "name"); 
            data.put("add", "add"); 
            elasticSearchOperation.insert(data,"indexName","indexMapping",UUID.randomUUID().toString()); 
        }catch(Exception e) { 
            e.printStackTrace(); 
            //System.out.println(e); 
        } 
    } 

} 
```

我们将使用在*将 Storm 与 HBase 集成*部分中创建的相同的`SampleSpout`类。

1.  在`com.stormadvance.storm_elasticsearch`包中创建一个`ESBolt`类。这个 bolt 接收`SampleSpout`类发出的元组，将其转换为`Map`结构，然后调用`ElasticSearchOperation`类的`insert()`方法将记录插入 Elasticsearch。以下是`ESBolt`类的源代码：

```scala
public class ESBolt implements IBasicBolt { 

    private static final long serialVersionUID = 2L; 
    private ElasticSearchOperation elasticSearchOperation; 
    private List<String> esNodes; 

    /** 
     *  
     * @param esNodes 
     */ 
    public ESBolt(List<String> esNodes) { 
        this.esNodes = esNodes; 

    } 

    public void execute(Tuple input, BasicOutputCollector collector) { 
        Map<String, Object> personalMap = new HashMap<String, Object>(); 
        // "firstName","lastName","companyName") 
        personalMap.put("firstName", input.getValueByField("firstName")); 
        personalMap.put("lastName", input.getValueByField("lastName")); 

        personalMap.put("companyName", input.getValueByField("companyName")); 
        elasticSearchOperation.insert(personalMap,"person","personmapping",UUID.randomUUID().toString()); 
    } 

    public void declareOutputFields(OutputFieldsDeclarer declarer) { 

    } 

    public Map<String, Object> getComponentConfiguration() { 
        // TODO Auto-generated method stub 
        return null; 
    } 

    public void prepare(Map stormConf, TopologyContext context) { 
        try { 
            // create the instance of ESOperations class 
            elasticSearchOperation = new ElasticSearchOperation(esNodes); 
        } catch (Exception e) { 
            throw new RuntimeException(); 
        } 
    } 

    public void cleanup() { 

    } 

} 
```

1.  在`com.stormadvance.storm_elasticsearch`包中创建一个`ESTopology`类。这个类创建了`spout`和`bolt`类的实例，并使用`TopologyBuilder`类将它们链接在一起。以下是主类的实现：

```scala
public class ESTopology {
    public static void main(String[] args) throws AlreadyAliveException, 
            InvalidTopologyException { 
        TopologyBuilder builder = new TopologyBuilder(); 

        //ES Node list 
        List<String> esNodes = new ArrayList<String>(); 
        esNodes.add("10.191.209.14"); 

        // set the spout class 
        builder.setSpout("spout", new SampleSpout(), 2); 
        // set the ES bolt class 
        builder.setBolt("bolt", new ESBolt(esNodes), 2) 
                .shuffleGrouping("spout"); 
        Config conf = new Config(); 
        conf.setDebug(true); 
        // create an instance of LocalCluster class for 
        // executing topology in local mode. 
        LocalCluster cluster = new LocalCluster(); 

        // ESTopology is the name of submitted topology. 
        cluster.submitTopology("ESTopology", conf, 
                builder.createTopology()); 
        try { 
            Thread.sleep(60000); 
        } catch (Exception exception) { 
            System.out.println("Thread interrupted exception : " + exception); 
        } 
        System.out.println("Stopped Called : "); 
        // kill the LearningStormTopology 
        cluster.killTopology("StormHBaseTopology"); 
        // shutdown the storm test cluster 
        cluster.shutdown(); 

    } 
} 
```

在本节中，我们介绍了如何通过在 Storm bolts 内部与 Elasticsearch 节点建立连接来将数据存储到 Elasticsearch 中。

# 将 Storm 与 Esper 集成

在本节中，我们将介绍如何在 Storm 中使用 Esper 进行窗口操作。Esper 是一个用于**复杂事件处理**（**CEP**）的开源事件序列分析和事件关联引擎。

请参阅[`www.espertech.com/products/esper.php`](http://www.espertech.com/products/esper.php)了解更多关于 Esper 的详细信息。按照以下步骤将 Storm 与 Esper 集成：

1.  使用`com.stormadvance`作为`groupID`，`storm_esper`作为`artifactID`创建一个 Maven 项目。

1.  在`pom.xml`文件中添加以下依赖项和存储库：

```scala
    <dependencies>
        <dependency>
            <groupId>com.espertech</groupId>
            <artifactId>esper</artifactId>
            <version>5.3.0</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>3.8.1</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.apache.storm</groupId>
            <artifactId>storm-core</artifactId>
            <version>1.0.2</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>
```

1.  在`com.stormadvance.storm_elasticsearch`包中创建一个`EsperOperation`类。`EsperOperation`类包含以下方法：

+   `esperPut(Stock stock)`: 这个方法以股票 bean 作为输入，将事件发送给 Esper 监听器。

`EsperOperation`类的构造函数初始化了 Esper 监听器并设置了 Esper 查询。Esper 查询在 5 分钟内缓冲事件并返回每个产品在 5 分钟窗口期内的总销售额。在这里，我们使用了固定批处理窗口。

以下是`EsperOperation`类的源代码：

```scala
public class EsperOperation { 

    private EPRuntime cepRT = null; 

    public EsperOperation() { 
        Configuration cepConfig = new Configuration(); 
        cepConfig.addEventType("StockTick", Stock.class.getName()); 
        EPServiceProvider cep = EPServiceProviderManager.getProvider( 
                "myCEPEngine", cepConfig); 
        cepRT = cep.getEPRuntime(); 

        EPAdministrator cepAdm = cep.getEPAdministrator(); 
        EPStatement cepStatement = cepAdm 
                .createEPL("select sum(price),product from " 
                        + "StockTick.win:time_batch(5 sec) " 
                        + "group by product"); 

        cepStatement.addListener(new CEPListener()); 
    } 

    public static class CEPListener implements UpdateListener { 

        public void update(EventBean[] newData, EventBean[] oldData) { 
            try { 
                System.out.println("#################### Event received: 
                "+newData); 
                for (EventBean eventBean : newData) { 
                    System.out.println("************************ Event 
                     received 1: " + eventBean.getUnderlying()); 
                } 

            } catch (Exception e) { 
                e.printStackTrace(); 
                System.out.println(e); 
            } 
        } 
    } 

    public void esperPut(Stock stock) { 
        cepRT.sendEvent(stock); 
    } 

    private static Random generator = new Random(); 

    public static void main(String[] s) throws InterruptedException { 
        EsperOperation esperOperation = new EsperOperation(); 
        // We generate a few ticks... 
        for (int i = 0; i < 5; i++) { 
            double price = (double) generator.nextInt(10); 
            long timeStamp = System.currentTimeMillis(); 
            String product = "AAPL"; 
            Stock stock = new Stock(product, price, timeStamp); 
            System.out.println("Sending tick:" + stock); 
            esperOperation.esperPut(stock); 
        } 
        Thread.sleep(200000); 
    } 

} 
```

1.  在`com.stormadvance.storm_esper`包中创建一个`SampleSpout`类。这个类生成随机记录并将它们传递给拓扑中的下一个操作（bolt）。以下是`SampleSpout`类生成的记录的格式：

```scala
    ["product type","price","sale date"] 
```

以下是`SampleSpout`类的源代码：

```scala
public class SampleSpout extends BaseRichSpout { 
    private static final long serialVersionUID = 1L; 
    private SpoutOutputCollector spoutOutputCollector; 

    private static final Map<Integer, String> PRODUCT = new 
    HashMap<Integer, String>(); 
    static { 
        PRODUCT.put(0, "A"); 
        PRODUCT.put(1, "B"); 
        PRODUCT.put(2, "C"); 
        PRODUCT.put(3, "D"); 
        PRODUCT.put(4, "E"); 
    } 

    private static final Map<Integer, Double> price = new 
    HashMap<Integer, Double>(); 
    static { 
        price.put(0, 500.0); 
        price.put(1, 100.0); 
        price.put(2, 300.0); 
        price.put(3, 900.0); 
        price.put(4, 1000.0); 
    } 

    public void open(Map conf, TopologyContext context, 
            SpoutOutputCollector spoutOutputCollector) { 
        // Open the spout 
        this.spoutOutputCollector = spoutOutputCollector; 
    } 

    public void nextTuple() { 
        // Storm cluster repeatedly call this method to emit the 
        continuous // 
        // stream of tuples. 
        final Random rand = new Random(); 
        // generate the random number from 0 to 4\. 
        int randomNumber = rand.nextInt(5); 

        spoutOutputCollector.emit (new 
        Values(PRODUCT.get(randomNumber),price.get(randomNumber), 
        System.currentTimeMillis())); 
        try { 
            Thread.sleep(1000); 
        } catch (InterruptedException e) { 
            // TODO Auto-generated catch block 
            e.printStackTrace(); 
        } 
    } 

    public void declareOutputFields(OutputFieldsDeclarer declarer) { 
        // emits the field  firstName , lastName and companyName. 
        declarer.declare(new Fields("product","price","timestamp")); 
    } 
} 
```

1.  在`com.stormadvance.storm_esper`包中创建一个`EsperBolt`类。这个 bolt 接收`SampleSpout`类发出的元组，将其转换为股票 bean，然后调用`EsperBolt`类的`esperPut()`方法将数据传递给 Esper 引擎。以下是`EsperBolt`类的源代码：

```scala
public class EsperBolt implements IBasicBolt { 

    private static final long serialVersionUID = 2L; 
    private EsperOperation esperOperation; 

    public EsperBolt() { 

    } 

    public void execute(Tuple input, BasicOutputCollector collector) { 

        double price = input.getDoubleByField("price"); 
        long timeStamp = input.getLongByField("timestamp"); 
        //long timeStamp = System.currentTimeMillis(); 
        String product = input.getStringByField("product"); 
        Stock stock = new Stock(product, price, timeStamp); 
        esperOperation.esperPut(stock); 
    } 

    public void declareOutputFields(OutputFieldsDeclarer declarer) { 

    } 

    public Map<String, Object> getComponentConfiguration() { 
        // TODO Auto-generated method stub 
        return null; 
    } 

    public void prepare(Map stormConf, TopologyContext context) { 
        try { 
            // create the instance of ESOperations class 
            esperOperation = new EsperOperation(); 
        } catch (Exception e) { 
            throw new RuntimeException(); 
        } 
    } 

    public void cleanup() { 

    } 
} 
```

1.  在`com.stormadvance.storm_esper`包中创建一个`EsperTopology`类。这个类创建了`spout`和`bolt`类的实例，并使用`TopologyBuilder`类将它们链接在一起。以下是主类的实现：

```scala
public class EsperTopology { 
    public static void main(String[] args) throws AlreadyAliveException, 
            InvalidTopologyException { 
        TopologyBuilder builder = new TopologyBuilder(); 

        // set the spout class 
        builder.setSpout("spout", new SampleSpout(), 2); 
        // set the ES bolt class 
        builder.setBolt("bolt", new EsperBolt(), 2) 
                .shuffleGrouping("spout"); 
        Config conf = new Config(); 
        conf.setDebug(true); 
        // create an instance of LocalCluster class for 
        // executing topology in local mode. 
        LocalCluster cluster = new LocalCluster(); 

        // EsperTopology is the name of submitted topology. 
        cluster.submitTopology("EsperTopology", conf, 
                builder.createTopology()); 
        try { 
            Thread.sleep(60000); 
        } catch (Exception exception) { 
            System.out.println("Thread interrupted exception : " + exception); 
        } 
        System.out.println("Stopped Called : "); 
        // kill the LearningStormTopology 
        cluster.killTopology("EsperTopology"); 
        // shutdown the storm test cluster 
        cluster.shutdown(); 

    } 
} 
```

# 总结

在本章中，我们主要关注了 Storm 与其他数据库的集成。此外，我们还介绍了如何在 Storm 中使用 Esper 执行窗口操作。

在下一章中，我们将介绍 Apache 日志处理案例研究。我们将解释如何通过 Storm 处理日志文件来生成业务信息。
