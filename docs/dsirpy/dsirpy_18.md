# 第十八章：Redis

> 原文：[`allendowney.github.io/DSIRP/redis.html`](https://allendowney.github.io/DSIRP/redis.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


单击此处在 Colab 上运行本章节

单击此处在 Colab 上运行本章节

## 持久性

仅存储在运行程序的内存中的数据称为“易失性”，因为当程序结束时它会消失。

在创建它的程序结束后仍然存在的数据称为“持久性”。一般来说，存储在文件系统中的文件以及存储在数据库中的数据是持久的。

使数据持久的一个简单方法是将其存储在文件中。例如，在程序结束之前，它可以将其数据结构转换为类似[JSON](https://en.wikipedia.org/wiki/JSON)的格式，然后将其写入文件。当它再次启动时，它可以读取文件并重建数据结构。

但是这种解决方案存在几个问题：

1.  读取和写入大型数据结构（如 Web 索引）将会很慢。

1.  整个数据结构可能无法适应单个运行程序的内存。

1.  如果程序意外结束（例如由于停电），自上次程序启动以来所做的任何更改都将丢失。

更好的选择是提供持久存储和能够读取和写入数据库部分而不是读取和写入整个数据库的数据库。

有许多种[数据库管理系统](https://en.wikipedia.org/wiki/Database)（DBMS）提供了这些功能。

我们将使用的数据库是 Redis，它以类似于 Python 数据结构的结构组织数据。除其他外，它提供列表、哈希（类似于 Python 字典）和集合。

Redis 是一个“键值数据库”，这意味着它表示从键到值的映射。在 Redis 中，键是字符串，值可以是几种类型之一。

## Redis 客户端和服务器

Redis 通常作为远程服务运行；实际上，这个名字代表“远程字典服务器”。要使用 Redis，您必须在某个地方运行 Redis 服务器，然后使用 Redis 客户端连接到它。

要开始，我们将在运行 Jupyter 服务器的同一台机器上运行 Redis 服务器。这将让我们快速开始，但如果我们在 Colab 上运行 Jupyter，数据库存在于 Colab 运行时环境中，当我们关闭笔记本时它就会消失。所以它并不是真正的持久化。

稍后我们将使用[RedisToGo](http://thinkdast.com/redistogo)，它在云中运行 Redis。RedisToGo 上的数据库是持久的。

以下单元格安装 Redis 服务器，并使用`daemonize`选项启动它，该选项在后台运行它，以便 Jupyter 服务器可以恢复。

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
341134:C 20 Dec 2021 15:10:27.756 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
341134:C 20 Dec 2021 15:10:27.756 # Redis version=5.0.3, bits=64, commit=00000000, modified=0, pid=341134, just started
341134:C 20 Dec 2021 15:10:27.756 # Configuration loaded 
```

## redis-py

要与 Redis 服务器通信，我们将使用[redis-py](https://redis-py.readthedocs.io/en/stable/index.html)。以下是我们如何使用它来连接到 Redis 服务器。

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

`set`方法向数据库添加键值对。在下面的示例中，键和值都是字符串。

```py
r.set('key', 'value') 
```

```py
True 
```

`get`方法查找一个键并返回相应的值。

```py
r.get('key') 
```

```py
b'value' 
```

结果实际上不是一个字符串；它是一个[bytearray](https://stackoverflow.com/questions/6224052/what-is-the-difference-between-a-string-and-a-byte-string)。

对于许多目的，bytearray 的行为类似于字符串，因此现在我们将把它视为字符串，并在出现差异时处理它们。

值可以是整数或浮点数。

```py
r.set('x', 5) 
```

```py
True 
```

Redis 还提供了一些理解数字的函数，比如`incr`。

```py
r.incr('x') 
```

```py
6 
```

但是如果您`get`一个数字值，结果是一个 bytearray。

```py
value = r.get('x')
value 
```

```py
b'6' 
```

如果您想对其进行数学运算，您必须将其转换回数字。

```py
int(value) 
```

```py
6 
```

如果要一次设置多个值，可以将字典传递给`mset`。

```py
d = dict(x=5, y='string', z=1.23)
r.mset(d) 
```

```py
True 
```

```py
r.get('y') 
```

```py
b'string' 
```

```py
r.get('z') 
```

```py
b'1.23' 
```

如果您尝试将任何其他类型存储在 Redis 数据库中，您将收到一个错误。

```py
from redis import DataError

t = [1, 2, 3]

try:
    r.set('t', t)
except DataError as e:
    print(e) 
```

```py
Invalid input of type: 'list'. Convert to a bytes, string, int or float first. 
```

我们可以使用`repr`函数创建列表的字符串表示，但该表示是特定于 Python 的。最好创建一个可以与任何语言一起使用的数据库。为此，我们可以使用 JSON 创建一个字符串表示。

`json`模块提供了一个`dumps`函数，它创建大多数 Python 对象的语言无关表示。

```py
import json

t = [1, 2, 3]
s = json.dumps(t)
s 
```

```py
'[1, 2, 3]' 
```

当我们读取其中一个字符串时，我们可以使用`loads`将其转换回 Python 对象。

```py
t = json.loads(s)
t 
```

```py
[1, 2, 3] 
```

**练习：**创建一个包含几个项目的字典，包括不同类型的键和值。使用`json`将字典制作成字符串表示，然后将其存储为 Redis 数据库中的值。检索它并将其转换回字典。

## Redis 数据类型

JSON 可以表示大多数 Python 对象，因此我们可以使用它来在 Redis 中存储任意数据结构。但在这种情况下，Redis 只知道它们是字符串；它无法将它们作为数据结构处理。例如，如果我们在 JSON 中存储数据结构，修改它的唯一方法是：

1.  获取整个结构，这可能很大，

1.  将其加载回 Python 结构，

1.  修改 Python 结构，

1.  将其转换回 JSON 字符串，

1.  用新值替换数据库中的旧值。

这并不是很有效。更好的选择是使用 Redis 提供的数据类型，您可以在[Redis 数据类型介绍](https://redis.io/topics/data-types-intro)中了解更多信息。

# 列表

`rpush`方法将新元素添加到列表的末尾（`r`表示列表的右侧）。

```py
r.rpush('t', 1, 2, 3) 
```

```py
3 
```

您无需执行任何特殊操作即可创建列表；如果不存在，Redis 会创建它。

`llen`返回列表的长度。

```py
r.llen('t') 
```

```py
3 
```

`lrange`从列表中获取元素。使用索引`0`和`-1`，它获取所有元素。

```py
r.lrange('t', 0, -1) 
```

```py
[b'1', b'2', b'3'] 
```

结果是一个 Python 列表，但元素是字节字符串。

`rpop`从列表的末尾移除元素。

```py
r.rpop('t') 
```

```py
b'3' 
```

您可以在[Redis 文档](https://redis.io/commands#list)中了解更多关于其他列表方法的信息。

您可以在[redis-py API 这里](https://redis-py.readthedocs.io/en/stable/index.html#redis.Redis.rpush)了解更多信息。

一般来说，Redis 的文档非常好；`redis-py`的文档有点粗糙。

**练习：**使用`lpush`将元素添加到列表的开头，使用`lpop`将元素移除。

注意：Redis 列表的行为类似于链表，因此您可以在常数时间内从任一端添加和删除元素。

```py
r.lpush('t', -3, -2, -1) 
```

```py
5 
```

```py
r.lpop('t') 
```

```py
b'-1' 
```

## 哈希

[Redis 哈希](https://redis.io/commands#hash)类似于 Python 字典，但为了使事情变得混乱，术语有点不同。

我们在 Python 字典中称为“键”的东西在 Redis 哈希中称为“字段”。

`hset`方法在哈希中设置字段-值对：

```py
r.hset('h', 'field', 'value') 
```

```py
1 
```

`hget`方法查找字段并返回相应的值。

```py
r.hget('h', 'field') 
```

```py
b'value' 
```

`hset`也可以接受 Python 字典作为参数：

```py
d = dict(a=1, b=2, c=3)
r.hset('h', mapping=d) 
```

```py
3 
```

要迭代哈希的元素，我们可以使用`hscan_iter`：

```py
for field, value in r.hscan_iter('h'):
    print(field, value) 
```

```py
b'field' b'value'
b'a' b'1'
b'b' b'2'
b'c' b'3' 
```

结果对于字段和值都是字节字符串。

**练习：**要向哈希添加多个项目，可以使用`hset`与关键字`mapping`和字典（或其他映射类型）。

使用`collections`模块的`Counter`对象来计算字符串中的字母，然后使用`hset`将结果存储在 Redis 哈希中。

然后使用`hscan_iter`来显示结果。

## 删除

在继续之前，让我们通过删除所有键值对来清理数据库。

```py
for key in r.keys():
    r.delete(key) 
```

## 变位词（再次！）

在之前的笔记本中，我们通过构建一个字典，其中键是字母排序的字符串，值是单词列表，制作了单词的变位词集合。

我们将首先使用 Python 数据结构解决这个问题；然后我们将其转换为 Redis。

以下单元格下载一个包含单词列表的文件。

```py
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)

download('https://github.com/AllenDowney/DSIRP/raw/main/american-english') 
```

这是一个生成器函数，它读取文件中的单词并逐个产生它们。

```py
def iterate_words(filename):
  """Read lines from a file and split them into words."""
    for line in open(filename):
        for word in line.split():
            yield word.strip() 
```

单词的“签名”是一个包含按排序顺序排列的单词字母的字符串。因此，如果两个单词是变位词，它们具有相同的签名。

```py
def signature(word):
    return ''.join(sorted(word)) 
```

以下循环创建了一个变位词列表的字典。

```py
anagram_dict = {}
for word in iterate_words('american-english'):
    key = signature(word)
    anagram_dict.setdefault(key, []).append(word) 
```

以下循环打印了所有具有 6 个或更多单词的变位词列表

```py
for v in anagram_dict.values():
    if len(v) >= 6:
        print(len(v), v) 
```

```py
6 ['abets', 'baste', 'bates', 'beast', 'beats', 'betas']
6 ['aster', 'rates', 'stare', 'tares', 'taser', 'tears']
6 ['caret', 'cater', 'crate', 'react', 'recta', 'trace']
7 ['carets', 'caster', 'caters', 'crates', 'reacts', 'recast', 'traces']
6 ['drapes', 'padres', 'parsed', 'rasped', 'spared', 'spread']
6 ['lapse', 'leaps', 'pales', 'peals', 'pleas', 'sepal']
6 ['least', 'slate', 'stale', 'steal', 'tales', 'teals']
6 ['opts', 'post', 'pots', 'spot', 'stop', 'tops']
6 ['palest', 'pastel', 'petals', 'plates', 'pleats', 'staple']
7 ['pares', 'parse', 'pears', 'rapes', 'reaps', 'spare', 'spear'] 
```

现在，要在 Redis 中执行相同的操作，我们有两个选项：

+   我们可以使用 Redis 列表存储变位词列表，使用签名作为键。

+   我们可以将整个数据结构存储在 Redis 哈希中。

第一个选项的问题是，Redis 数据库中的键就像全局变量一样。如果我们创建大量的键，很可能会遇到名称冲突。我们可以通过给每个键添加一个标识其目的的前缀来缓解这个问题。

以下循环实现了第一个选项，使用“Anagram”作为键的前缀。

```py
for word in iterate_words('american-english'):
    key = f'Anagram:{signature(word)}'
    r.rpush(key, word) 
```

这个选项的优点是它很好地利用了 Redis 列表。缺点是它进行了许多小型数据库事务，因此相对较慢。

我们可以使用`keys`获取具有给定前缀的所有键的列表。

```py
keys = r.keys('Anagram*')
len(keys) 
```

```py
96936 
```

**练习：**编写一个循环，遍历`keys`，使用`llen`获取每个列表的长度，并打印具有 6 个或更多元素的所有列表的元素。

在继续之前，我们可以像这样从数据库中删除键。

```py
r.delete(*keys) 
```

```py
96936 
```

第二个选项是在本地计算变位词列表的字典，然后将其存储为 Redis 哈希。

以下函数使用`dumps`将列表转换为可以作为 Redis 哈希中的值存储的字符串。

```py
hash_key = 'AnagramHash'
for field, t in anagram_dict.items():
    value = json.dumps(t)
    r.hset(hash_key, field, value) 
```

如果我们将所有列表在本地转换为 JSON 并使用一个`hset`命令存储所有字段值对，我们可以更快地完成相同的操作。

首先，我将删除我们刚刚创建的哈希。

```py
r.delete(hash_key) 
```

```py
1 
```

**练习：**创建一个 Python 字典，其中包含`anagram_dict`中的项目，但将值转换为 JSON。使用`hset`和`mapping`关键字将其存储为 Redis 哈希。

**练习：**编写一个循环，遍历字段值对，将每个值转换回 Python 列表，并打印具有 6 个或更多元素的列表。

## 关闭

如果您在自己的计算机上运行此笔记本，可以使用以下命令关闭 Redis 服务器。

如果您在 Colab 上运行，这并不是真正必要的：当 Colab 运行时关闭时，Redis 服务器将关闭（其中存储的所有内容将消失）。

```py
!killall  redis-server 
```

*Python 中的数据结构和信息检索*

版权所有 2021 年 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
