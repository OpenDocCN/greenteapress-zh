# 第二十一章：测验 5

> 原文：[`allendowney.github.io/DSIRP/quiz05.html`](https://allendowney.github.io/DSIRP/quiz05.html)

在开始这个测验之前：

1.  点击“复制到驱动器”以复制测验，

1.  点击“分享”，

1.  点击“更改”，然后选择“任何拥有此链接的人都可以编辑”

1.  点击“复制链接”和

1.  将链接粘贴到[此 Canvas 作业](https://canvas.olin.edu/courses/313/assignments/5075)中。

这个测验是开放笔记，开放互联网。

+   您可以向讲师寻求帮助，但不能向其他人寻求帮助。

+   您可以使用在互联网上找到的代码，但如果您从单个来源使用了超过几行代码，您应该注明出处。

## 安装和启动 Redis

对于这个测验，我们将在 Colab 上运行 Redis。以下单元安装并启动服务器，安装客户端，并实例化一个`Redis`对象。

```py
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    !pip  install  redis-server
    !/usr/local/lib/python*/dist-packages/redis_server/bin/redis-server  --daemonize  yes
else:
    !redis-server  --daemonize  yes 
```

```py
try:
    import redis
except ImportError:
    !pip  install  redis 
```

```py
import redis

r = redis.Redis() 
```

## 银行家琳达

在一项[著名的实验](https://en.wikipedia.org/wiki/Conjunction_fallacy)中，特沃斯基和卡尼曼提出了以下问题：

> 琳达今年 31 岁，单身，直言不讳，非常聪明。她主修哲学。作为学生，她对歧视和社会正义问题非常关注，并参与了反核示威活动。哪个更可能？
> 
> 1.  琳达是一名银行出纳员。
> 1.  
> 1.  琳达是一名银行出纳员，并且积极参与女权主义运动。

许多人选择第二个答案，可能是因为它似乎更符合描述。如果琳达*只是*一个银行出纳员，那似乎是不典型的；如果她还是一个女权主义者，那似乎更一致。

但第二个答案不能是“更可能”，正如问题所问的那样。为了了解原因，让我们探索一些数据。以下单元从[普遍社会调查](http://www.gss.norc.org/)下载数据。

```py
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)

download('https://github.com/AllenDowney/BiteSizeBayes/raw/master/gss_bayes.csv') 
```

以下单元将数据加载到 Pandas `DataFrame`中。如果您对 Pandas 不熟悉，我会解释您需要了解的内容。

```py
import pandas as pd

gss = pd.read_csv('gss_bayes.csv', index_col=0)
gss.index = pd.Index(range(len(gss)), name='caseid')
gss.head() 
```

`DataFrame`中每个受访者都有一行，称为“受访者”，我选择的每个变量都有一列。这些列是：

+   `caseid`：受访者的识别号码。

+   `year`：受访者接受调查的年份。

+   `age`：受访者接受调查时的年龄。

+   `sex`：男性或女性。

+   `polviews`：从自由主义到保守主义的政治观点范围。

+   `partyid`：政党隶属，民主党、独立党或共和党。

+   `indus10`：受访者所在行业的[代码](https://www.census.gov/cgi-bin/sssd/naics/naicsrch?chart=2007)。

我们将使用 Redis 集来探索这些变量之间的关系。具体来说，我们将回答与“琳达问题”相关的以下问题。

+   女性银行家的受访者人数，

+   自由主义女性银行家的受访者人数。

我们将看到第二个数字比第一个数字小。

## 遍历行

以下循环遍历`DataFrame`中的前 3 行，并打印`caseid`和行的内容。

```py
for caseid, row in gss.iterrows():
    print(caseid)
    print(row)
    if caseid >= 3:
        break 
```

以下循环遍历`DataFrame`并创建一个集，其中包含行业代码为 6870 的行的`caseid`，这表明受访者在银行业工作。

```py
bankers = set()

for caseid, row in gss.iterrows():
    if row.indus10 == 6870:
        bankers.add(caseid)

len(bankers) 
```

现在让我们使用 Redis 集来做同样的事情。

## 问题 1

以下循环创建一个 Redis 集，其中包含所有`indus10`为`6870`的受访者的`caseid`。

```py
banker_key = 'gss_set:bankers'

for caseid, row in gss.iterrows():
    if row.indus10 == 6870:
        r.sadd(banker_key, caseid) 
```

编写一个 Redis 命令来获取结果集中的元素数量。

这是[Redis 集命令的文档](https://redis.io/commands#set)。

## 问题 2

以下单元创建一个包含所有自我认同为女性的受访者的`caseid`的 Python 集。

```py
female = set()

for caseid, row in gss.iterrows():
    if row.sex == 2:
        female.add(caseid)

len(female) 
```

以下单元创建一个 Python 集，其中包括自我认同为“极端自由主义者”、“自由主义者”或“稍微自由主义者”的`caseid`。

```py
liberal = set()

for caseid, row in gss.iterrows():
    if row.polviews <= 3:
        liberal.add(caseid)

len(liberal) 
```

编写这些循环的 Redis 版本，创建这些集，并显示每个集中的元素数量。对于键，使用以下字符串：

```py
female_key = 'gss_set:female'
liberal_key = 'gss_set:liberal' 
```

在继续之前，请确保您在 Redis 上有三个集，并且每个集中的元素数量与我们使用 Python 集得到的结果一致。

如果你犯了一个错误，你可以使用`delete`来从一个新的空集合开始。或者你可以使用以下循环来从一个新的空数据库开始。

```py
#for key in r.keys():
#    r.delete(key) 
```

## 问题 3

Redis 的一个优点是它提供了在服务器上执行计算的函数，包括一个计算两个或更多集合的交集的函数。

编写 Redis 命令来计算：

1.  一组`caseid`值，用于受访者是女性银行家的情况。

1.  自由主义女性银行家的`caseid`值集合。

确认第二个集合实际上比第一个小。

## 问题 4

现在假设你想查找一个`caseid`并找到它所属的所有集合。

编写一个名为`find_tags`的函数，它接受一个`caseid`并返回一个字符串集合，其中每个字符串是包含`caseid`的集合的键。

例如，如果`caseid`是 33，结果应该是这个集合

```py
{b'gss_set:bankers', b'gss_set:female'} 
```

这表明这个受访者是一名女性银行家（但不是自由主义者）。

你可以使用以下示例来测试你的函数。你应该会发现`caseid`为 33 的受访者是一名女性银行家。

```py
find_tags(33) 
```

而`caseid`为 451 的受访者是一名自由主义女性银行家。

```py
find_tags(451) 
```

## 只是为了好玩的额外问题

假设有大量的集合，你经常想要查找一个`caseid`并找到它所属的集合。

遍历到目前为止我们定义的集合，并创建一个反向索引，将每个`caseid`映射到它所属的集合的键列表。

*Python 中的数据结构和信息检索*

版权所有 2021 年 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
