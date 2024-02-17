# 索引器

> 原文：[`allendowney.github.io/DSIRP/indexer.html`](https://allendowney.github.io/DSIRP/indexer.html)

[点击这里在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/indexer.ipynb)

[点击这里在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/indexer.ipynb)

## 对网页进行索引

在网页搜索的上下文中，索引是一种数据结构，使得可以查找搜索词并找到包含该词的页面成为可能。此外，我们还想知道搜索词在每个页面上出现的次数，这将有助于确定与该词相关的页面。

例如，如果用户提交搜索词“Python”和“programming”，我们将查找这两个搜索词并获得两组页面。包含单词“Python”的页面将包括关于蛇种的页面和关于编程语言的页面。包含单词“programming”的页面将包括关于不同编程语言的页面，以及单词的其他用法。通过选择同时包含两个术语的页面，我们希望消除无关的页面并找到关于 Python 编程的页面。

为了制作一个索引，我们需要遍历文档中的单词并对其进行计数。这就是我们将开始的地方。

以下是一个我们之前见过的最小的 HTML 文档，借用自 BeautifulSoup 文档。

```py
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
""" 
```

我们可以使用`BeautifulSoup`来解析文本并创建 DOM。

```py
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_doc)
type(soup) 
```

```py
bs4.BeautifulSoup 
```

以下是一个生成器函数，它遍历 DOM 的元素，找到`NavigableString`对象，遍历单词，并逐个产生它们。

从每个单词中，它删除了由`string`模块标识为空格或标点符号的字符。

```py
from bs4 import NavigableString
from string import whitespace, punctuation

def iterate_words(soup):
    for element in soup.descendants:
        if isinstance(element, NavigableString):
            for word in element.string.split():
                word = word.strip(whitespace + punctuation)
                if word:
                    yield word.lower() 
```

我们可以像这样循环遍历单词：

```py
for word in iterate_words(soup):
    print(word) 
```

```py
the
dormouse's
story
the
dormouse's
story
once
upon
a
time
there
were
three
little
sisters
and
their
names
were
elsie
lacie
and
tillie
and
they
lived
at
the
bottom
of
a
well 
```

并像这样对它们进行计数。

```py
from collections import Counter

counter = Counter(iterate_words(soup))
counter 
```

```py
Counter({'the': 3,
         "dormouse's": 2,
         'story': 2,
         'once': 1,
         'upon': 1,
         'a': 2,
         'time': 1,
         'there': 1,
         'were': 2,
         'three': 1,
         'little': 1,
         'sisters': 1,
         'and': 3,
         'their': 1,
         'names': 1,
         'elsie': 1,
         'lacie': 1,
         'tillie': 1,
         'they': 1,
         'lived': 1,
         'at': 1,
         'bottom': 1,
         'of': 1,
         'well': 1}) 
```

## 解析维基百科

现在让我们用维基百科页面的文本做同样的事情：

```py
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
filename = download(url) 
```

```py
fp = open(filename)
soup2 = BeautifulSoup(fp) 
```

```py
counter = Counter(iterate_words(soup2))
counter.most_common(10) 
```

```py
[('the', 3),
 ('and', 3),
 ("dormouse's", 2),
 ('story', 2),
 ('a', 2),
 ('were', 2),
 ('once', 1),
 ('upon', 1),
 ('time', 1),
 ('there', 1)] 
```

正如您所期望的那样，“python”是维基百科关于 Python 的页面上最常见的单词之一。单词“programming”没有进入前 10 名，但它也出现了很多次。

```py
counter['programming'] 
```

```py
0 
```

然而，有一些常见的单词，比如“the”和“from”，也出现了很多次。稍后，我们将回来考虑如何区分真正表示页面内容的单词和出现在每个页面上的常见单词。

但首先，让我们考虑制作一个索引。

## 索引

索引是从搜索词（如“python”）到包含该词的页面集合的映射。该集合还应指示该词在每个页面上出现的次数。

我们希望索引是持久的，所以我们将其存储在 Redis 上。

那么我们应该使用什么 Redis 类型？有几个选项，但一个合理的选择是每个单词一个哈希，其中字段是页面（由 URL 表示），值是计数。

为了管理索引的大小，除非给定搜索词至少出现三次，否则我们不会列出给定搜索词的页面。

让我们开始 Redis。

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
340987:C 20 Dec 2021 15:08:08.771 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
340987:C 20 Dec 2021 15:08:08.771 # Redis version=5.0.3, bits=64, commit=00000000, modified=0, pid=340987, just started
340987:C 20 Dec 2021 15:08:08.771 # Configuration loaded 
```

并确保 Redis 客户端已安装。

```py
try:
    import redis
except ImportError:
    !pip  install  redis 
```

让我们创建一个`Redis`对象，它创建到 Redis 数据库的连接。

```py
import redis

r = redis.Redis() 
```

如果您在不同的机器上运行 Redis 数据库，可以使用数据库的 URL 创建一个`Redis`对象，就像这样

```py
url = 'redis://redistogo:example@dory.redistogo.com:10534/'
r = redis.Redis.from_url(url) 
```

**练习：**编写一个名为`redis_index`的函数，它接受一个 URL 并对其进行索引。它应该下载给定 URL 的网页，遍历单词，并创建一个从单词到它们频率的`Counter`。

然后它应该遍历单词并将字段-值对添加到 Redis 哈希中。

+   哈希的键应该有前缀`Index:`；例如，单词`python`的键应该是`Index:python`。

+   哈希中的字段应该是 URL。

+   哈希中的值应该是单词计数。

使用您的函数来索引至少这两个页面：

```py
url1 = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
url2 = 'https://en.wikipedia.org/wiki/Python_(genus)' 
```

使用`hscan_iter`来迭代单词`python`在索引中的字段-值对。打印出这个单词出现的页面的 URL 以及它在每个页面上出现的次数。

## 关闭

如果您在自己的计算机上运行这个笔记本，您可以使用以下命令关闭 Redis 服务器。

如果您在 Colab 上运行，这并不是真正必要的：当 Colab 运行时关闭时，Redis 服务器将被关闭（其中存储的所有内容都将消失）。

```py
!killall  redis-server 
```

## RedisToGo

[RedisToGo](https://redistogo.com)是一个提供远程 Redis 数据库的托管服务。他们提供一个免费计划，其中包括一个小型数据库，非常适合测试我们的索引器。

如果您注册并转到您的实例列表，您应该会找到一个看起来像这样的 URL：

```py
redis://redistogo:digitsandnumbers@dory.redistogo.com:10534/ 
```

如果您将此 URL 传递给`Redis.from_url`，如上所述，您应该能够连接到 RedisToGo 上的数据库，并再次运行您的练习解决方案。

如果您稍后回来并阅读索引，您的数据应该仍然在那里！
