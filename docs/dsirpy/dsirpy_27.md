# 第二十七章：爬虫

> 原文：[`allendowney.github.io/DSIRP/crawler.html`](https://allendowney.github.io/DSIRP/crawler.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


[点击此处在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/crawler.ipynb)

## 爬取网络

在这一点上，我们已经有了构建网络爬虫所需的所有部分；是时候把它们放在一起了。

首先，从`philosophy.ipynb`中，我们有`WikiFetcher`，我们将使用它从维基百科下载页面，同时限制每秒约一次的请求。

```py
from urllib.request import urlopen
from bs4 import BeautifulSoup
from time import time, sleep

class WikiFetcher:
    next_request_time = None
    min_interval = 1  # second

    def fetch_wikipedia(self, url):
        self.sleep_if_needed()
        fp = urlopen(url)
        soup = BeautifulSoup(fp, 'html.parser')
        return soup

    def sleep_if_needed(self):
        if self.next_request_time:
            sleep_time = self.next_request_time - time()    
            if sleep_time > 0:
                sleep(sleep_time)

        self.next_request_time = time() + self.min_interval 
```

这里有一个例子：

```py
fetcher = WikiFetcher()

url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
soup = fetcher.fetch_wikipedia(url) 
```

结果是一个表示页面文档对象模型（DOM）的 BeautifulSoup 对象。

请注意，如果`url`是一个字节数组，`WikiFetcher`将无法工作，因为`urlopen`不适用于字节数组。

```py
url = b'https://en.wikipedia.org/wiki/Python_(programming_language)'
# soup = fetcher.fetch_wikipedia(url) 
```

要将字节数组转换为字符串，您必须对其进行解码。

```py
url_str = url.decode()
soup = fetcher.fetch_wikipedia(url_str) 
```

通常当您调用`decode`时，您应该[指定要使用的编码](https://docs.python.org/3.8/library/stdtypes.html#bytes.decode)。但在这种情况下，我们知道原始字符串是 URL，所以默认编码将起作用。

维基百科页面包含我们不想索引的样板内容，因此我们将选择包含页面“正文内容”的`div`元素。

```py
root = soup.find(class_='mw-body-content') 
```

## 查找链接

从`philosophy.ipynb`中，我们有一个遍历 DOM 并查找链接的函数。

```py
from bs4 import Tag

def link_generator(root):
    for element in root.descendants:
        if isinstance(element, Tag) and element.name == 'a':
            href = element.get('href', '')
            if href.startswith('/wiki'):
                yield element 
```

这个版本包括指向图像和其他我们可能不想索引的链接。

以下版本包括一个条件，检查链接是否具有`title`属性，这似乎选择了大部分“好”链接。

```py
def link_generator(root):
    for element in root.descendants:
        if isinstance(element, Tag) and element.name == 'a':
            title = element.get('title', '')
            href = element.get('href', '')
            if title and href.startswith('/wiki'):
                yield element 
```

这是我们下载的页面的前几个链接。

```py
for i, link in enumerate(link_generator(root)):
    print(link)
    if i == 5:
        break 
```

```py
<a href="/wiki/Programming_paradigm" title="Programming paradigm">Paradigm</a>
<a class="mw-redirect" href="/wiki/Multi-paradigm_programming_language" title="Multi-paradigm programming language">Multi-paradigm</a>
<a href="/wiki/Object-oriented_programming" title="Object-oriented programming">object-oriented</a>
<a href="/wiki/Procedural_programming" title="Procedural programming">procedural</a>
<a href="/wiki/Imperative_programming" title="Imperative programming">imperative</a>
<a href="/wiki/Functional_programming" title="Functional programming">functional</a> 
```

## 查找单词

从`indexer.ipynb`中，我们有以下函数，它遍历 DOM 并产生单词，去除标点并转换为小写。

```py
from bs4 import NavigableString
from string import whitespace, punctuation

def iterate_words(root):
    for element in root.descendants:
        if isinstance(element, NavigableString):
            for word in element.string.split():
                word = word.strip(whitespace + punctuation)
                if word:
                    yield word.lower() 
```

这是我们下载的页面的前几个单词。它们包括页面右侧边栏中的关键字，这些关键字不是主要文本的一部分，但可能是很好的索引，因为它们指示了页面的主题。

```py
for i, word in enumerate(iterate_words(root)):
    print(word)
    if i > 200:
        break 
```

```py
general-purpose
programming
language
mw-parser-output
infobox-subbox{padding:0;border:none;margin:-3px;width:auto;min-width:100%;font-size:100%;clear:none;float:none;background-color:transparent}.mw-parser-output
infobox-3cols-child{margin:auto
python
paradigm
multi-paradigm
object-oriented
1
procedural
imperative
functional
structured
reflective
designed
by
guido
van
rossum
developer
python
software
foundation
first
appeared
20
february
1991
30
years
ago
1991-02-20
2
stable
release
3.10.1
3
6
december
2021
14
days
ago
6
december
2021
preview
release
3.11.0a3
4
8
november
2021
42
days
ago
8
november
2021
typing
discipline
duck
dynamic
strong
typing
5
gradual
since
3.5
but
ignored
in
cpython
6
os
windows
linux/unix
macos
and
more
7
license
python
software
foundation
license
filename
extensions
py
pyi
pyc
pyd
pyo
prior
to
3.5
8
pyw
pyz
since
3.5
9
website
www
python
org
major
implementations
cpython
pypy
stackless
python
micropython
circuitpython
ironpython
jython
dialects
cython
rpython
starlark
10
influenced
by
abc
11
ada
12
algol
68
13
apl
14
c
15
c
16
clu
17
dylan
18
haskell
19
icon
20
java
21
lisp
22
modula-3
16
perl
standard
ml
14
influenced
apache
groovy
boo
cobra
coffeescript
23
d
f
genie
24
go
javascript
25
26
julia
27
nim
ring
28
ruby
29
swift
30
python
programming
at
wikibooks
python
is
an
interpreted
high-level
general-purpose
programming
language
its
design
philosophy
emphasizes
code
readability
with
its
use
of 
```

## Redis

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
340680:C 20 Dec 2021 15:03:31.520 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
340680:C 20 Dec 2021 15:03:31.520 # Redis version=5.0.3, bits=64, commit=00000000, modified=0, pid=340680, just started
340680:C 20 Dec 2021 15:03:31.520 # Configuration loaded 
```

并确保安装了 Redis 客户端。

```py
try:
    import redis
except ImportError:
    !pip  install  redis 
```

我们将创建一个`Redis`对象，它创建到 Redis 数据库的连接。

```py
import redis

r = redis.Redis() 
```

如果您的 Redis 数据库在另一台机器上运行，您可以使用数据库的 URL 创建一个`Redis`对象，就像这样

```py
url = 'redis://redistogo:example@dory.redistogo.com:10534/'
r = redis.Redis.from_url(url) 
```

如果您的数据库包含以前练习的值，或者如果您犯了一个错误并想重新开始，您可以使用以下函数来清除数据库。

```py
def clear_redis(r):
    for key in r.keys():
        r.delete(key)

# clear_redis(r) 
```

## 索引

从`indexer.ipynb`中，这是一个计算页面上单词数量并将结果添加到 Redis 哈希的函数。

对于每个单词，它在数据库中创建或更新一个哈希，将 URL 映射到单词计数。例如，如果单词`python`在页面上出现了 428 次，我们可以找到键为`Index:python`的哈希，并添加一个将 URL 映射到数字 428 的条目。

```py
from bs4 import BeautifulSoup
from collections import Counter

def redis_index(root, url):
    counter = Counter(iterate_words(root))
    for word, count in counter.items():
        if count >= 3:
            key = f'Index:{word}'
            # print(key, count)
            r.hset(key, url, count) 
```

之前的版本可能会很慢，因为它会向数据库发出许多小请求。我们可以使用管道对象来加快速度，就像这样：

```py
def redis_index_pipeline(root, url):
    counter = Counter(iterate_words(root))
    p = r.pipeline(transaction=False)
    for word, count in counter.items():
        if count >= 3:
            key = f'Index:{word}'
            # print(key, count)
            p.hset(key, url, count)
    p.execute() 
```

让我们看看哪个版本更快。

```py
url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
soup = fetcher.fetch_wikipedia(url)
root = soup.find(class_='mw-body-content') 
```

```py
%time redis_index(root, url) 
```

```py
CPU times: user 36.9 ms, sys: 11.8 ms, total: 48.8 ms
Wall time: 52.6 ms 
```

```py
%time redis_index_pipeline(root, url) 
```

```py
CPU times: user 26 ms, sys: 75 µs, total: 26.1 ms
Wall time: 25.1 ms 
```

我们可以使用`hscan_iter`来迭代单词`python`的索引中的字段-值对，并打印出这个单词出现的页面的 URL 以及它在每个页面上出现的次数。

```py
key = f'Index:python'

for page, count in r.hscan_iter(key):
    print(page, count) 
```

```py
b'https://en.wikipedia.org/wiki/Python_(programming_language)' b'428' 
```

请注意，当我们得到数字时，它是一个字节数组。如果我们想将其作为数字处理，我们必须转换为整数。

## 爬行

在`philosophy.ipynb`中，我们编写了一个简单的爬虫，它总是跟随第一个链接。

```py
from urllib.parse import urljoin

target = 'https://en.wikipedia.org/wiki/Philosophy'

def get_to_philosophy(url):
    visited = []

    for i in range(20):
        if url == target:
            print(f'Got there in {i} steps!')
            return visited

        if url in visited:
            raise ValueError(f'URL already visited {url}')
        else:
            print(url)
            visited.append(url)

        soup = fetcher.fetch_wikipedia(url)
        root = soup.find(class_='mw-body-content')
        link = next(link_generator(root))
        url = urljoin(url, link['href'])

    return visited 
```

```py
get_to_philosophy(url) 
```

```py
https://en.wikipedia.org/wiki/Python_(programming_language)
https://en.wikipedia.org/wiki/Programming_paradigm
https://en.wikipedia.org/wiki/Programming_model 
```

```py
---------------------------------------------------------------------------
ValueError  Traceback (most recent call last)
/tmp/ipykernel_336891/1601292495.py in <module>
----> 1 get_to_philosophy(url)

/tmp/ipykernel_336891/1250407340.py in get_to_philosophy(url)
  12 
  13         if url in visited:
---> 14             raise ValueError(f'URL already visited {url}')
  15         else:
  16             print(url)

ValueError: URL already visited https://en.wikipedia.org/wiki/Programming_paradigm 
```

现在我们想要一个运行广度优先搜索的爬虫。这是从`bfs.ipynb`中实现的 BFS：

```py
from collections import deque

def reachable_nodes_bfs(G, start):
    seen = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in seen:
            seen.add(node)
            neighbors = set(G[node]) - seen
            queue.extend(neighbors)
    return seen 
```

**练习：**编写一个名为`crawl`的函数，它以起始 URL 作为参数，并可选地指定要爬取的页面数。

它应该创建一个 URL 队列，并逐个处理队列，索引页面并将新链接添加到队列中。

对于初稿，我建议使用 Python 数据结构来跟踪队列和已经看到/索引的 URL 集合。

```py
url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
seen = crawl(url) 
```

```py
https://en.wikipedia.org/wiki/Python_(programming_language)
https://en.wikipedia.org/wiki/Programming_paradigm
https://en.wikipedia.org/wiki/Multi-paradigm_programming_language
https://en.wikipedia.org/wiki/Object-oriented_programming
https://en.wikipedia.org/wiki/Procedural_programming
https://en.wikipedia.org/wiki/Imperative_programming
https://en.wikipedia.org/wiki/Functional_programming
https://en.wikipedia.org/wiki/Structured_programming
https://en.wikipedia.org/wiki/Reflective_programming
https://en.wikipedia.org/wiki/Software_design 
```

```py
key = 'Index:the'
for page, count in r.hscan_iter(key):
    print(page, count) 
```

```py
b'https://en.wikipedia.org/wiki/Reflective_programming' b'53'
b'https://en.wikipedia.org/wiki/Functional_programming' b'227'
b'https://en.wikipedia.org/wiki/Object-oriented_programming' b'277'
b'https://en.wikipedia.org/wiki/Procedural_programming' b'49'
b'https://en.wikipedia.org/wiki/Multi-paradigm_programming_language' b'118'
b'https://en.wikipedia.org/wiki/Imperative_programming' b'81'
b'https://en.wikipedia.org/wiki/Structured_programming' b'161'
b'https://en.wikipedia.org/wiki/Programming_paradigm' b'118'
b'https://en.wikipedia.org/wiki/Software_design' b'152'
b'https://en.wikipedia.org/wiki/Python_(programming_language)' b'535' 
```

对于第二稿，考虑将这些结构存储在 Redis 中，这样它们就是持久的；这样，你可以稍后调用`crawl`，它将从离开的地方继续。或者你可以同时运行多个爬虫。

提示：当你从 Redis 读取一个 URL 时，你可能需要解码它以使其成为一个字符串。

```py
queue_key = 'Crawler:queue'

r.lpop(queue_key) 
```

```py
seen_key = 'Crawler:seen'

r.sismember(seen_key, 'anything') 
```

```py
False 
```

```py
url = 'https://en.wikipedia.org/wiki/Object-oriented_programming'
crawl_persistent(url) 
```

```py
b'https://en.wikipedia.org/wiki/Object-oriented_programming' 
```

```py
r.smembers(seen_key) 
```

```py
{b'https://en.wikipedia.org/wiki/Object-oriented_programming'} 
```

```py
r.lrange(queue_key, 0, -1) 
```

```py
[b'https://en.wikipedia.org/wiki/Object-orientation',
 b'https://en.wikipedia.org/wiki/List_of_object-oriented_programming_languages',
 b'https://en.wikipedia.org/wiki/Programming_paradigm',
 b'https://en.wikipedia.org/wiki/Action_language',
 b'https://en.wikipedia.org/wiki/Agent-oriented_programming',
 b'https://en.wikipedia.org/wiki/Array_programming',
 b'https://en.wikipedia.org/wiki/Automata-based_programming',
 b'https://en.wikipedia.org/wiki/Concurrent_computing',
 b'https://en.wikipedia.org/wiki/Relativistic_programming',
 b'https://en.wikipedia.org/wiki/Data-driven_programming',
 b'https://en.wikipedia.org/wiki/Declarative_programming',
 b'https://en.wikipedia.org/wiki/Imperative_programming',
 b'https://en.wikipedia.org/wiki/Functional_programming',
 b'https://en.wikipedia.org/wiki/Functional_logic_programming',
 b'https://en.wikipedia.org/wiki/Purely_functional_programming',
 b'https://en.wikipedia.org/wiki/Logic_programming',
 b'https://en.wikipedia.org/wiki/Abductive_logic_programming',
 b'https://en.wikipedia.org/wiki/Answer_set_programming',
 b'https://en.wikipedia.org/wiki/Concurrent_logic_programming',
 b'https://en.wikipedia.org/wiki/Functional_logic_programming',
 b'https://en.wikipedia.org/wiki/Inductive_logic_programming',
 b'https://en.wikipedia.org/wiki/Constraint_programming',
 b'https://en.wikipedia.org/wiki/Constraint_logic_programming',
 b'https://en.wikipedia.org/wiki/Concurrent_constraint_logic_programming',
 b'https://en.wikipedia.org/wiki/Dataflow_programming',
 b'https://en.wikipedia.org/wiki/Flow-based_programming',
 b'https://en.wikipedia.org/wiki/Reactive_programming',
 b'https://en.wikipedia.org/wiki/Functional_reactive_programming',
 b'https://en.wikipedia.org/wiki/Ontology_language',
 b'https://en.wikipedia.org/wiki/Differentiable_programming',
 b'https://en.wikipedia.org/wiki/Dynamic_programming_language',
 b'https://en.wikipedia.org/wiki/Event-driven_programming',
 b'https://en.wikipedia.org/wiki/Function-level_programming',
 b'https://en.wikipedia.org/wiki/Value-level_programming',
 b'https://en.wikipedia.org/wiki/Tacit_programming',
 b'https://en.wikipedia.org/wiki/Concatenative_programming_language',
 b'https://en.wikipedia.org/wiki/Generic_programming',
 b'https://en.wikipedia.org/wiki/Imperative_programming',
 b'https://en.wikipedia.org/wiki/Declarative_programming',
 b'https://en.wikipedia.org/wiki/Procedural_programming',
 b'https://en.wikipedia.org/wiki/Polymorphic_code',
 b'https://en.wikipedia.org/wiki/Intentional_programming',
 b'https://en.wikipedia.org/wiki/Language-oriented_programming',
 b'https://en.wikipedia.org/wiki/Domain-specific_language',
 b'https://en.wikipedia.org/wiki/Literate_programming',
 b'https://en.wikipedia.org/wiki/Natural-language_programming',
 b'https://en.wikipedia.org/wiki/Metaprogramming',
 b'https://en.wikipedia.org/wiki/Automatic_programming',
 b'https://en.wikipedia.org/wiki/Inductive_programming',
 b'https://en.wikipedia.org/wiki/Reflective_programming',
 b'https://en.wikipedia.org/wiki/Attribute-oriented_programming',
 b'https://en.wikipedia.org/wiki/Macro_(computer_science)',
 b'https://en.wikipedia.org/wiki/Template_metaprogramming',
 b'https://en.wikipedia.org/wiki/Non-structured_programming',
 b'https://en.wikipedia.org/wiki/Structured_programming',
 b'https://en.wikipedia.org/wiki/Array_programming',
 b'https://en.wikipedia.org/wiki/Nondeterministic_programming',
 b'https://en.wikipedia.org/wiki/Parallel_computing',
 b'https://en.wikipedia.org/wiki/Process-oriented_programming',
 b'https://en.wikipedia.org/wiki/Probabilistic_programming',
 b'https://en.wikipedia.org/wiki/Quantum_programming',
 b'https://en.wikipedia.org/wiki/Set_theoretic_programming',
 b'https://en.wikipedia.org/wiki/Stack-oriented_programming',
 b'https://en.wikipedia.org/wiki/Structured_programming',
 b'https://en.wikipedia.org/wiki/Non-structured_programming',
 b'https://en.wikipedia.org/wiki/Block_(programming)',
 b'https://en.wikipedia.org/wiki/Structured_concurrency',
 b'https://en.wikipedia.org/wiki/Actor_model',
 b'https://en.wikipedia.org/wiki/Class-based_programming',
 b'https://en.wikipedia.org/wiki/Concurrent_object-oriented_programming',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/Separation_of_concerns',
 b'https://en.wikipedia.org/wiki/Aspect-oriented_programming',
 b'https://en.wikipedia.org/wiki/Role-oriented_programming',
 b'https://en.wikipedia.org/wiki/Subject-oriented_programming',
 b'https://en.wikipedia.org/wiki/Recursion_(computer_science)',
 b'https://en.wikipedia.org/wiki/Symbolic_programming',
 b'https://en.wikipedia.org/wiki/Value-level_programming',
 b'https://en.wikipedia.org/wiki/Function-level_programming',
 b'https://en.wikipedia.org/wiki/Template:Programming_paradigms',
 b'https://en.wikipedia.org/wiki/Template_talk:Programming_paradigms',
 b'https://en.wikipedia.org/wiki/Programming_paradigm',
 b'https://en.wikipedia.org/wiki/Object_(computer_science)',
 b'https://en.wikipedia.org/wiki/Data',
 b'https://en.wikipedia.org/wiki/Field_(computer_science)',
 b'https://en.wikipedia.org/wiki/Attribute_(computing)',
 b'https://en.wikipedia.org/wiki/Method_(computer_science)',
 b'https://en.wikipedia.org/wiki/This_(computer_programming)',
 b'https://en.wikipedia.org/wiki/Class-based_programming',
 b'https://en.wikipedia.org/wiki/Instance_(computer_science)',
 b'https://en.wikipedia.org/wiki/Class_(computer_science)',
 b'https://en.wikipedia.org/wiki/Data_type',
 b'https://en.wikipedia.org/wiki/Multi-paradigm_programming_language',
 b'https://en.wikipedia.org/wiki/Imperative_programming',
 b'https://en.wikipedia.org/wiki/Procedural_programming',
 b'https://en.wikipedia.org/wiki/Java_(programming_language)',
 b'https://en.wikipedia.org/wiki/C%2B%2B',
 b'https://en.wikipedia.org/wiki/C_Sharp_(programming_language)',
 b'https://en.wikipedia.org/wiki/Python_(programming_language)',
 b'https://en.wikipedia.org/wiki/R_(programming_language)',
 b'https://en.wikipedia.org/wiki/PHP',
 b'https://en.wikipedia.org/wiki/Visual_Basic.NET',
 b'https://en.wikipedia.org/wiki/JavaScript',
 b'https://en.wikipedia.org/wiki/Ruby_(programming_language)',
 b'https://en.wikipedia.org/wiki/Perl',
 b'https://en.wikipedia.org/wiki/SIMSCRIPT',
 b'https://en.wikipedia.org/wiki/Object_Pascal',
 b'https://en.wikipedia.org/wiki/Objective-C',
 b'https://en.wikipedia.org/wiki/Dart_(programming_language)',
 b'https://en.wikipedia.org/wiki/Swift_(programming_language)',
 b'https://en.wikipedia.org/wiki/Scala_(programming_language)',
 b'https://en.wikipedia.org/wiki/Kotlin_(programming_language)',
 b'https://en.wikipedia.org/wiki/Common_Lisp',
 b'https://en.wikipedia.org/wiki/MATLAB',
 b'https://en.wikipedia.org/wiki/Smalltalk',
 b'https://en.wikipedia.org/wiki/Unified_Modeling_Language',
 b'https://en.wikipedia.org/wiki/Variable_(computer_science)',
 b'https://en.wikipedia.org/wiki/Method_(computer_programming)',
 b'https://en.wikipedia.org/wiki/MIT',
 b'https://en.wikipedia.org/wiki/Artificial_intelligence',
 b'https://en.wikipedia.org/wiki/Lisp_(programming_language)',
 b'https://en.wikipedia.org/wiki/Alan_Kay',
 b'https://en.wikipedia.org/wiki/Sketchpad',
 b'https://en.wikipedia.org/wiki/Ivan_Sutherland',
 b'https://en.wikipedia.org/wiki/ALGOL',
 b'https://en.wikipedia.org/wiki/Simula',
 b'https://en.wikipedia.org/wiki/Class_(computer_programming)',
 b'https://en.wikipedia.org/wiki/Object_(computer_science)',
 b'https://en.wikipedia.org/wiki/Dynamic_binding_(computing)',
 b'https://en.wikipedia.org/wiki/Physical_modelling',
 b'https://en.wikipedia.org/wiki/Smalltalk',
 b'https://en.wikipedia.org/wiki/Xerox_PARC',
 b'https://en.wikipedia.org/wiki/Alan_Kay',
 b'https://en.wikipedia.org/wiki/Dan_Ingalls',
 b'https://en.wikipedia.org/wiki/Adele_Goldberg_(computer_scientist)',
 b'https://en.wikipedia.org/wiki/Dynamic_programming',
 b'https://en.wikipedia.org/wiki/Interpreter_(computing)',
 b'https://en.wikipedia.org/wiki/Compiler',
 b'https://en.wikipedia.org/wiki/Lisp_(programming_language)#Language_innovations',
 b'https://en.wikipedia.org/wiki/Lisp_(programming_language)#Object_systems',
 b'https://en.wikipedia.org/wiki/Lisp_machine',
 b'https://en.wikipedia.org/wiki/Flavors_(programming_language)',
 b'https://en.wikipedia.org/wiki/Multiple_inheritance',
 b'https://en.wikipedia.org/wiki/Mixins',
 b'https://en.wikipedia.org/wiki/Common_Lisp_Object_System',
 b'https://en.wikipedia.org/wiki/Meta-object_protocol',
 b'https://en.wikipedia.org/wiki/Intel_iAPX_432',
 b'https://en.wikipedia.org/wiki/Linn_Products',
 b'https://en.wikipedia.org/wiki/Rekursiv',
 b'https://en.wikipedia.org/wiki/Byte_Magazine',
 b'https://en.wikipedia.org/wiki/Association_for_Computing_Machinery',
 b'https://en.wikipedia.org/wiki/Objective-C',
 b'https://en.wikipedia.org/wiki/Brad_Cox',
 b'https://en.wikipedia.org/wiki/ITT_Inc.',
 b'https://en.wikipedia.org/wiki/Bjarne_Stroustrup',
 b'https://en.wikipedia.org/wiki/C%2B%2B',
 b'https://en.wikipedia.org/wiki/Bertrand_Meyer',
 b'https://en.wikipedia.org/wiki/Eiffel_(programming_language)',
 b'https://en.wikipedia.org/wiki/Object-Oriented_Software_Construction',
 b'https://en.wikipedia.org/wiki/Design_by_Contract',
 b'https://en.wikipedia.org/wiki/File:Tiobeindex.png',
 b'https://en.wikipedia.org/wiki/TIOBE_index',
 b'https://en.wikipedia.org/wiki/Measuring_programming_language_popularity',
 b'https://en.wikipedia.org/wiki/Java_(programming_language)',
 b'https://en.wikipedia.org/wiki/Procedural_programming',
 b'https://en.wikipedia.org/wiki/C_(programming_language)',
 b'https://en.wikipedia.org/wiki/Paradigm',
 b'https://en.wikipedia.org/wiki/FoxPro',
 b'https://en.wikipedia.org/wiki/C%2B%2B',
 b'https://en.wikipedia.org/wiki/Delphi_(programming_language)',
 b'https://en.wikipedia.org/wiki/Wikipedia:Citation_needed',
 b'https://en.wikipedia.org/wiki/Graphical_user_interface',
 b'https://en.wikipedia.org/wiki/Cocoa_(software)',
 b'https://en.wikipedia.org/wiki/Mac_OS_X',
 b'https://en.wikipedia.org/wiki/Objective-C',
 b'https://en.wikipedia.org/wiki/Event-driven_programming',
 b'https://en.wikipedia.org/wiki/ETH_Z%C3%BCrich',
 b'https://en.wikipedia.org/wiki/Niklaus_Wirth',
 b'https://en.wikipedia.org/wiki/Data_abstraction',
 b'https://en.wikipedia.org/wiki/Modularity_(programming)',
 b'https://en.wikipedia.org/wiki/Modula-2',
 b'https://en.wikipedia.org/wiki/Oberon_(programming_language)',
 b'https://en.wikipedia.org/wiki/Ada_(programming_language)',
 b'https://en.wikipedia.org/wiki/BASIC',
 b'https://en.wikipedia.org/wiki/Fortran',
 b'https://en.wikipedia.org/wiki/Pascal_(programming_language)',
 b'https://en.wikipedia.org/wiki/COBOL',
 b'https://en.wikipedia.org/wiki/Python_(programming_language)',
 b'https://en.wikipedia.org/wiki/Ruby_programming_language',
 b'https://en.wikipedia.org/wiki/Java_(programming_language)',
 b'https://en.wikipedia.org/wiki/Sun_Microsystems',
 b'https://en.wikipedia.org/wiki/C_Sharp_(programming_language)',
 b'https://en.wikipedia.org/wiki/Visual_Basic.NET',
 b'https://en.wikipedia.org/wiki/.NET_Framework',
 b'https://en.wikipedia.org/wiki/Multi-paradigm',
 b'https://en.wikipedia.org/wiki/Comparison_of_programming_languages_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/List_of_object-oriented_programming_terms',
 b'https://en.wikipedia.org/wiki/Variable_(computer_science)',
 b'https://en.wikipedia.org/wiki/Data_type',
 b'https://en.wikipedia.org/wiki/Integer_(computer_science)',
 b'https://en.wikipedia.org/wiki/Character_(computing)',
 b'https://en.wikipedia.org/wiki/Data_structures',
 b'https://en.wikipedia.org/wiki/String_(computer_science)',
 b'https://en.wikipedia.org/wiki/List_(abstract_data_type)',
 b'https://en.wikipedia.org/wiki/Hash_table',
 b'https://en.wikipedia.org/wiki/Pointer_(computer_programming)',
 b'https://en.wikipedia.org/wiki/Subroutine',
 b'https://en.wikipedia.org/wiki/Structured_programming',
 b'https://en.wikipedia.org/wiki/Loop_(computing)',
 b'https://en.wikipedia.org/wiki/Conditional_(computer_programming)',
 b'https://en.wikipedia.org/wiki/Modular_programming',
 b'https://en.wikipedia.org/wiki/Namespace',
 b'https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Class-based_programming',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/Class_(computer_science)',
 b'https://en.wikipedia.org/wiki/Object_(computer_science)',
 b'https://en.wikipedia.org/wiki/Instance_(computer_science)',
 b'https://en.wikipedia.org/wiki/Method_(computer_science)',
 b'https://en.wikipedia.org/wiki/Field_(computer_science)',
 b'https://en.wikipedia.org/wiki/Class_variable',
 b'https://en.wikipedia.org/wiki/Instance_variable',
 b'https://en.wikipedia.org/wiki/Member_variable',
 b'https://en.wikipedia.org/wiki/Pointer_(computer_programming)',
 b'https://en.wikipedia.org/wiki/Abstraction_(computer_science)',
 b'https://en.wikipedia.org/wiki/Constructor_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Class-based_programming',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/Trait_(computer_programming)',
 b'https://en.wikipedia.org/wiki/Mixin',
 b'https://en.wikipedia.org/wiki/Class-based_programming',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/Equivalence_class',
 b'https://en.wikipedia.org/wiki/Delegation_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Single_inheritance',
 b'https://en.wikipedia.org/wiki/Dynamic_dispatch',
 b'https://en.wikipedia.org/wiki/Abstract_data_type',
 b'https://en.wikipedia.org/wiki/Multiple_dispatch',
 b'https://en.wikipedia.org/wiki/Message_passing',
 b'https://en.wikipedia.org/wiki/Information_hiding',
 b'https://en.wikipedia.org/wiki/Encapsulation_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Underscore',
 b'https://en.wikipedia.org/wiki/Code_refactoring',
 b'https://en.wikipedia.org/wiki/Coupling_(computer_programming)',
 b'https://en.wikipedia.org/wiki/Object_composition',
 b'https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Multiple_inheritance',
 b'https://en.wikipedia.org/wiki/Mixin',
 b'https://en.wikipedia.org/wiki/Abstract_class',
 b'https://en.wikipedia.org/wiki/Final_(Java)',
 b'https://en.wikipedia.org/wiki/Composition_over_inheritance',
 b'https://en.wikipedia.org/wiki/Go_(programming_language)',
 b'https://en.wikipedia.org/wiki/Open/closed_principle',
 b'https://en.wikipedia.org/wiki/Delegation_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Subtyping',
 b'https://en.wikipedia.org/wiki/Polymorphism_(computer_science)',
 b'https://en.wikipedia.org/wiki/Separation_of_concerns',
 b'https://en.wikipedia.org/wiki/Open_recursion',
 b'https://en.wikipedia.org/wiki/Name_binding',
 b'https://en.wikipedia.org/wiki/Wikipedia:Citing_sources',
 b'https://en.wikipedia.org/wiki/Wikipedia:Verifiability',
 b'https://en.wikipedia.org/wiki/Help:Referencing_for_beginners',
 b'https://en.wikipedia.org/wiki/Wikipedia:Verifiability#Burden_of_evidence',
 b'https://en.wikipedia.org/wiki/Help:Maintenance_template_removal',
 b'https://en.wikipedia.org/wiki/List_of_object-oriented_programming_languages',
 b'https://en.wikipedia.org/wiki/Simula',
 b'https://en.wikipedia.org/wiki/Computer_simulation',
 b'https://en.wikipedia.org/wiki/Smalltalk',
 b'https://en.wikipedia.org/wiki/Ruby_(programming_language)',
 b'https://en.wikipedia.org/wiki/Scala_(programming_language)',
 b'https://en.wikipedia.org/wiki/Smalltalk',
 b'https://en.wikipedia.org/wiki/Eiffel_(programming_language)',
 b'https://en.wikipedia.org/wiki/Emerald_(programming_language)',
 b'https://en.wikipedia.org/wiki/JADE_(programming_language)',
 b'https://en.wikipedia.org/wiki/Self_(programming_language)',
 b'https://en.wikipedia.org/wiki/Raku_(programming_language)',
 b'https://en.wikipedia.org/wiki/Java_(programming_language)',
 b'https://en.wikipedia.org/wiki/Python_(programming_language)',
 b'https://en.wikipedia.org/wiki/C%2B%2B',
 b'https://en.wikipedia.org/wiki/C_Sharp_(programming_language)',
 b'https://en.wikipedia.org/wiki/Delphi_(programming_language)',
 b'https://en.wikipedia.org/wiki/Object_Pascal',
 b'https://en.wikipedia.org/wiki/VB.NET',
 b'https://en.wikipedia.org/wiki/Procedural_programming',
 b'https://en.wikipedia.org/wiki/PHP',
 b'https://en.wikipedia.org/wiki/Perl',
 b'https://en.wikipedia.org/wiki/Visual_Basic',
 b'https://en.wikipedia.org/wiki/MATLAB',
 b'https://en.wikipedia.org/wiki/COBOL_2002',
 b'https://en.wikipedia.org/wiki/Fortran_2003',
 b'https://en.wikipedia.org/wiki/ABAP',
 b'https://en.wikipedia.org/wiki/Ada_(programming_language)',
 b'https://en.wikipedia.org/wiki/Pascal_(programming_language)',
 b'https://en.wikipedia.org/wiki/Oberon_(programming_language)',
 b'https://en.wikipedia.org/wiki/Abstract_data_type',
 b'https://en.wikipedia.org/wiki/Object-based',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/JavaScript',
 b'https://en.wikipedia.org/wiki/Lua_(programming_language)',
 b'https://en.wikipedia.org/wiki/Modula-2',
 b'https://en.wikipedia.org/wiki/CLU_(programming_language)',
 b'https://en.wikipedia.org/wiki/Tcl',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/Dynamic_programming_language',
 b'https://en.wikipedia.org/wiki/Python_(programming_language)',
 b'https://en.wikipedia.org/wiki/Windows_PowerShell',
 b'https://en.wikipedia.org/wiki/Ruby_(programming_language)',
 b'https://en.wikipedia.org/wiki/Groovy_(programming_language)',
 b'https://en.wikipedia.org/wiki/Perl',
 b'https://en.wikipedia.org/wiki/PHP',
 b'https://en.wikipedia.org/wiki/ColdFusion',
 b'https://en.wikipedia.org/wiki/Document_Object_Model',
 b'https://en.wikipedia.org/wiki/HTML',
 b'https://en.wikipedia.org/wiki/XHTML',
 b'https://en.wikipedia.org/wiki/XML',
 b'https://en.wikipedia.org/wiki/JavaScript',
 b'https://en.wikipedia.org/wiki/ECMAScript',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/Class-based_programming',
 b'https://en.wikipedia.org/wiki/Lua_(programming_language)',
 b'https://en.wikipedia.org/wiki/Distributed_Data_Management_Architecture',
 b'https://en.wikipedia.org/wiki/Smalltalk',
 b'https://en.wikipedia.org/wiki/IBM_i',
 b'https://en.wikipedia.org/wiki/Object_(IBM_i)',
 b'https://en.wikipedia.org/wiki/DRDA',
 b'https://en.wikipedia.org/wiki/Design_Patterns_(book)',
 b'https://en.wikipedia.org/wiki/Design_pattern_(computer_science)',
 b'https://en.wikipedia.org/wiki/Object-oriented_design',
 b'https://en.wikipedia.org/wiki/Program_semantics',
 b'https://en.wikipedia.org/wiki/Is_a',
 b'https://en.wikipedia.org/wiki/Mutable',
 b'https://en.wikipedia.org/wiki/Subtype_polymorphism',
 b'https://en.wikipedia.org/wiki/Type_checker',
 b'https://en.wikipedia.org/wiki/Behavioral_subtyping',
 b'https://en.wikipedia.org/wiki/Liskov_substitution_principle',
 b'https://en.wikipedia.org/wiki/Design_pattern_(computer_science)',
 b'https://en.wikipedia.org/wiki/Design_Patterns_(book)',
 b'https://en.wikipedia.org/wiki/Erich_Gamma',
 b'https://en.wikipedia.org/wiki/Richard_Helm',
 b'https://en.wikipedia.org/wiki/Ralph_Johnson_(computer_scientist)',
 b'https://en.wikipedia.org/wiki/John_Vlissides',
 b'https://en.wikipedia.org/wiki/Creational_pattern',
 b'https://en.wikipedia.org/wiki/Factory_method_pattern',
 b'https://en.wikipedia.org/wiki/Abstract_factory_pattern',
 b'https://en.wikipedia.org/wiki/Singleton_pattern',
 b'https://en.wikipedia.org/wiki/Builder_pattern',
 b'https://en.wikipedia.org/wiki/Prototype_pattern',
 b'https://en.wikipedia.org/wiki/Structural_pattern',
 b'https://en.wikipedia.org/wiki/Adapter_pattern',
 b'https://en.wikipedia.org/wiki/Bridge_pattern',
 b'https://en.wikipedia.org/wiki/Composite_pattern',
 b'https://en.wikipedia.org/wiki/Decorator_pattern',
 b'https://en.wikipedia.org/wiki/Facade_pattern',
 b'https://en.wikipedia.org/wiki/Flyweight_pattern',
 b'https://en.wikipedia.org/wiki/Proxy_pattern',
 b'https://en.wikipedia.org/wiki/Behavioral_pattern',
 b'https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern',
 b'https://en.wikipedia.org/wiki/Command_pattern',
 b'https://en.wikipedia.org/wiki/Interpreter_pattern',
 b'https://en.wikipedia.org/wiki/Iterator_pattern',
 b'https://en.wikipedia.org/wiki/Mediator_pattern',
 b'https://en.wikipedia.org/wiki/Memento_pattern',
 b'https://en.wikipedia.org/wiki/Observer_pattern',
 b'https://en.wikipedia.org/wiki/State_pattern',
 b'https://en.wikipedia.org/wiki/Strategy_pattern',
 b'https://en.wikipedia.org/wiki/Template_method_pattern',
 b'https://en.wikipedia.org/wiki/Visitor_pattern',
 b'https://en.wikipedia.org/wiki/Object-relational_impedance_mismatch',
 b'https://en.wikipedia.org/wiki/Object-relational_mapping',
 b'https://en.wikipedia.org/wiki/Object_database',
 b'https://en.wikipedia.org/wiki/Relational_database_management_systems',
 b'https://en.wikipedia.org/wiki/Relational_database',
 b'https://en.wikipedia.org/wiki/Object-relational_impedance_mismatch',
 b'https://en.wikipedia.org/wiki/Object-relational_mapping',
 b'https://en.wikipedia.org/wiki/Integrated_development_environment',
 b'https://en.wikipedia.org/wiki/Visual_FoxPro',
 b'https://en.wikipedia.org/wiki/Java_Data_Objects',
 b'https://en.wikipedia.org/wiki/Ruby_on_Rails',
 b'https://en.wikipedia.org/wiki/Object_database',
 b'https://en.wikipedia.org/wiki/Object-oriented_programming#Criticism',
 b'https://en.wikipedia.org/wiki/Bertrand_Meyer',
 b'https://en.wikipedia.org/wiki/Object-Oriented_Software_Construction',
 b'https://en.wikipedia.org/wiki/Circle-ellipse_problem',
 b'https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Niklaus_Wirth',
 b'https://en.wikipedia.org/wiki/Wirth%27s_law',
 b'https://en.wikipedia.org/wiki/KISS_principle',
 b'https://en.wikipedia.org/wiki/Steve_Yegge',
 b'https://en.wikipedia.org/wiki/Noun',
 b'https://en.wikipedia.org/wiki/Verb',
 b'https://en.wikipedia.org/wiki/Code_reuse',
 b'https://en.wikipedia.org/wiki/Software_maintenance',
 b'https://en.wikipedia.org/wiki/Control_flow',
 b'https://en.wikipedia.org/wiki/Thread_(computer_science)',
 b'https://en.wikipedia.org/wiki/Responsibility-driven_design',
 b'https://en.wikipedia.org/wiki/Data-driven_programming',
 b'https://en.wikipedia.org/wiki/SOLID_(object-oriented_design)',
 b'https://en.wikipedia.org/wiki/Single_responsibility_principle',
 b'https://en.wikipedia.org/wiki/Open/closed_principle',
 b'https://en.wikipedia.org/wiki/Liskov_substitution_principle',
 b'https://en.wikipedia.org/wiki/Interface_segregation_principle',
 b'https://en.wikipedia.org/wiki/Dependency_inversion_principle',
 b'https://en.wikipedia.org/wiki/GRASP_(object-oriented_design)',
 b'https://en.wikipedia.org/wiki/Craig_Larman',
 b'https://en.wikipedia.org/wiki/Luca_Cardelli',
 b'https://en.wikipedia.org/wiki/Joe_Armstrong_(programming)',
 b'https://en.wikipedia.org/wiki/Erlang_(programming_language)',
 b'https://en.wikipedia.org/wiki/Christopher_J._Date',
 b'https://en.wikipedia.org/wiki/Data_type',
 b'https://en.wikipedia.org/wiki/RDBMS',
 b'https://en.wikipedia.org/wiki/Alexander_Stepanov',
 b'https://en.wikipedia.org/wiki/Generic_programming',
 b'https://en.wikipedia.org/wiki/Paul_Graham_(computer_programmer)',
 b'https://en.wikipedia.org/wiki/Duplicate_code',
 b'https://en.wikipedia.org/wiki/Don%27t_repeat_yourself',
 b'https://en.wikipedia.org/wiki/Steve_Yegge',
 b'https://en.wikipedia.org/wiki/Functional_programming',
 b'https://en.wikipedia.org/wiki/Rich_Hickey',
 b'https://en.wikipedia.org/wiki/Clojure',
 b'https://en.wikipedia.org/wiki/Eric_S._Raymond',
 b'https://en.wikipedia.org/wiki/Unix',
 b'https://en.wikipedia.org/wiki/Open-source_software',
 b'https://en.wikipedia.org/wiki/C_(programming_language)',
 b'https://en.wikipedia.org/wiki/Rob_Pike',
 b'https://en.wikipedia.org/wiki/UTF-8',
 b'https://en.wikipedia.org/wiki/Go_(programming_language)',
 b'https://en.wikipedia.org/wiki/Roman_numerals',
 b'https://en.wikipedia.org/wiki/Data_structure',
 b'https://en.wikipedia.org/wiki/Algorithm',
 b'https://en.wikipedia.org/wiki/Data_type',
 b'https://en.wikipedia.org/wiki/Java_(programming_language)',
 b'https://en.wikipedia.org/wiki/Lookup_table',
 b'https://en.wikipedia.org/wiki/Formal_semantics_of_programming_languages',
 b'https://en.wikipedia.org/wiki/F-Coalgebra',
 b'https://en.wikipedia.org/wiki/Abstract_data_type',
 b'https://en.wikipedia.org/wiki/Existential_types',
 b'https://en.wikipedia.org/wiki/Module_(programming)',
 b'https://en.wikipedia.org/wiki/Dynamic_dispatch',
 b'https://en.wikipedia.org/wiki/Recursive_type',
 b'https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Record_(computer_science)',
 b'https://en.wikipedia.org/wiki/Function_literal',
 b'https://en.wikipedia.org/wiki/System_F-sub',
 b'https://en.wikipedia.org/wiki/Subtype_polymorphism',
 b'https://en.wikipedia.org/wiki/Parametric_polymorphism',
 b'https://en.wikipedia.org/wiki/Syntactic_sugar',
 b'https://en.wikipedia.org/wiki/Portal:Computer_programming',
 b'https://en.wikipedia.org/wiki/Comparison_of_programming_languages_(object-oriented_programming)',
 b'https://en.wikipedia.org/wiki/Comparison_of_programming_paradigms',
 b'https://en.wikipedia.org/wiki/Component-based_software_engineering',
 b'https://en.wikipedia.org/wiki/Design_by_contract',
 b'https://en.wikipedia.org/wiki/Object_association',
 b'https://en.wikipedia.org/wiki/Object_database',
 b'https://en.wikipedia.org/wiki/Object_model_reference',
 b'https://en.wikipedia.org/wiki/Object_modeling_language',
 b'https://en.wikipedia.org/wiki/Object-oriented_analysis_and_design',
 b'https://en.wikipedia.org/wiki/Object-relational_impedance_mismatch',
 b'https://en.wikipedia.org/wiki/The_Third_Manifesto',
 b'https://en.wikipedia.org/wiki/Object-relational_mapping',
 b'https://en.wikipedia.org/wiki/CADES',
 b'https://en.wikipedia.org/wiki/Common_Object_Request_Broker_Architecture',
 b'https://en.wikipedia.org/wiki/Distributed_Component_Object_Model',
 b'https://en.wikipedia.org/wiki/Distributed_Data_Management_Architecture',
 b'https://en.wikipedia.org/wiki/Jeroo',
 b'https://en.wikipedia.org/wiki/IDEF4',
 b'https://en.wikipedia.org/wiki/Interface_description_language',
 b'https://en.wikipedia.org/wiki/Lepus3',
 b'https://en.wikipedia.org/wiki/Unified_Modeling_Language',
 b'https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-321-53205-3',
 b'https://en.wikipedia.org/wiki/Phyllis_Fox',
 b'https://en.wikipedia.org/wiki/Louis_Hodes',
 b'https://en.wikipedia.org/wiki/David_Luckham',
 b'https://en.wikipedia.org/wiki/David_Park_(computer_scientist)',
 b'https://en.wikipedia.org/wiki/Steve_Russell_(computer_scientist)',
 b'https://en.wikipedia.org/wiki/Boston',
 b'https://en.wikipedia.org/wiki/Massachusetts',
 b'https://en.wikipedia.org/wiki/M.I.T._Computation_Center',
 b'https://en.wikipedia.org/wiki/Help:CS1_errors#missing_periodical',
 b'https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist)',
 b'https://en.wikipedia.org/wiki/MIT_Press',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-262-13011-0',
 b'https://en.wikipedia.org/wiki/Kristen_Nygaard',
 b'https://en.wikipedia.org/wiki/Ole-Johan_Dahl',
 b'https://en.wikipedia.org/wiki/Doi_(identifier)',
 b'https://en.wikipedia.org/wiki/S2CID_(identifier)',
 b'https://en.wikipedia.org/wiki/Bibcode_(identifier)',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-3-540-92144-8',
 b'https://en.wikipedia.org/wiki/FoxPro',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-81-259-2532-3',
 b'https://en.wikipedia.org/wiki/John_C._Mitchell',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/0-521-78098-5',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/0-12-633951-1',
 b'https://en.wikipedia.org/wiki/Types_and_Programming_Languages',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-262-16209-8',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-8053-0608-8',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-1-78588-713-0',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-201-54435-0',
 b'https://en.wikipedia.org/wiki/Object_Management_Group',
 b'https://en.wikipedia.org/wiki/Niklaus_Wirth',
 b'https://en.wikipedia.org/wiki/Computer_(magazine)',
 b'https://en.wikipedia.org/wiki/Doi_(identifier)',
 b'https://en.wikipedia.org/wiki/S2CID_(identifier)',
 b'https://en.wikipedia.org/wiki/Robert_Harper_(computer_scientist)',
 b'https://en.wikipedia.org/wiki/Luca_Cardelli',
 b'https://en.wikipedia.org/wiki/Doi_(identifier)',
 b'https://en.wikipedia.org/wiki/ISSN_(identifier)',
 b'https://en.wikipedia.org/wiki/S2CID_(identifier)',
 b'https://en.wikipedia.org/wiki/Wayback_Machine',
 b'https://en.wikipedia.org/wiki/Alexander_Stepanov',
 b'https://en.wikipedia.org/wiki/Doi_(identifier)',
 b'https://en.wikipedia.org/wiki/Paul_Graham_(computer_programmer)',
 b'https://en.wikipedia.org/wiki/Rob_Pike',
 b'https://en.wikipedia.org/wiki/Martin_Abadi',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-387-94775-4',
 b'https://en.wikipedia.org/wiki/Martin_Abadi',
 b'https://en.wikipedia.org/wiki/Luca_Cardelli',
 b'https://en.wikipedia.org/wiki/Springer_Verlag',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-387-94775-4',
 b'https://en.wikipedia.org/wiki/Harold_Abelson',
 b'https://en.wikipedia.org/wiki/Gerald_Jay_Sussman',
 b'https://en.wikipedia.org/wiki/MIT_Press',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-262-01153-2',
 b'https://en.wikipedia.org/wiki/Doi_(identifier)',
 b'https://en.wikipedia.org/wiki/ISSN_(identifier)',
 b'https://en.wikipedia.org/wiki/S2CID_(identifier)',
 b'https://en.wikipedia.org/wiki/Grady_Booch',
 b'https://en.wikipedia.org/wiki/Addison-Wesley',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-8053-5340-2',
 b'https://en.wikipedia.org/wiki/Oliver_Sims',
 b'https://en.wikipedia.org/wiki/John_Wiley_%26_Sons',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-471-19176-6',
 b'https://en.wikipedia.org/wiki/Erich_Gamma',
 b'https://en.wikipedia.org/wiki/Richard_Helm',
 b'https://en.wikipedia.org/wiki/Ralph_Johnson_(computer_scientist)',
 b'https://en.wikipedia.org/wiki/John_Vlissides',
 b'https://en.wikipedia.org/wiki/Bibcode_(identifier)',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-201-63361-0',
 b'https://en.wikipedia.org/wiki/Paul_Harmon_(management_author)',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-471-14717-6',
 b'https://en.wikipedia.org/wiki/Ivar_Jacobson',
 b'https://en.wikipedia.org/wiki/Use_case',
 b'https://en.wikipedia.org/wiki/Bibcode_(identifier)',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-201-54435-0',
 b'https://en.wikipedia.org/wiki/Alan_Kay',
 b'https://en.wikipedia.org/wiki/Bertrand_Meyer',
 b'https://en.wikipedia.org/wiki/Object-Oriented_Software_Construction',
 b'https://en.wikipedia.org/wiki/Prentice_Hall',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-13-629155-8',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-80-904661-8-0',
 b'https://en.wikipedia.org/wiki/James_Rumbaugh',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-13-629841-0',
 b'https://en.wikipedia.org/wiki/McGraw-Hill',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-07-319126-3',
 b'https://en.wikipedia.org/wiki/Hdl_(identifier)',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-3-446-17426-9',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-471-54364-0',
 b'https://en.wikipedia.org/wiki/Addison-Wesley',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-672-33016-2',
 b'https://en.wikipedia.org/wiki/Microsoft_Press',
 b'https://en.wikipedia.org/wiki/ISBN_(identifier)',
 b'https://en.wikipedia.org/wiki/Special:BookSources/978-0-7356-1965-4',
 b'https://en.wikipedia.org/wiki/Template:Types_of_programming_languages',
 b'https://en.wikipedia.org/wiki/Template_talk:Types_of_programming_languages',
 b'https://en.wikipedia.org/wiki/Programming_paradigm',
 b'https://en.wikipedia.org/wiki/Actor_model',
 b'https://en.wikipedia.org/wiki/Array_programming',
 b'https://en.wikipedia.org/wiki/Aspect-oriented_programming',
 b'https://en.wikipedia.org/wiki/Class-based_programming',
 b'https://en.wikipedia.org/wiki/Concatenative_programming_language',
 b'https://en.wikipedia.org/wiki/Concurrent_computing',
 b'https://en.wikipedia.org/wiki/Dataflow_programming',
 b'https://en.wikipedia.org/wiki/Declarative_programming',
 b'https://en.wikipedia.org/wiki/Differentiable_programming',
 b'https://en.wikipedia.org/wiki/Domain-specific_language',
 b'https://en.wikipedia.org/wiki/Dynamic_programming_language',
 b'https://en.wikipedia.org/wiki/Esoteric_programming_language',
 b'https://en.wikipedia.org/wiki/Event-driven_programming',
 b'https://en.wikipedia.org/wiki/Extensible_programming',
 b'https://en.wikipedia.org/wiki/Functional_programming',
 b'https://en.wikipedia.org/wiki/Imperative_programming',
 b'https://en.wikipedia.org/wiki/Logic_programming',
 b'https://en.wikipedia.org/wiki/Macro_(computer_science)',
 b'https://en.wikipedia.org/wiki/Metaprogramming',
 b'https://en.wikipedia.org/wiki/Object-based_language',
 b'https://en.wikipedia.org/wiki/Pipeline_(software)',
 b'https://en.wikipedia.org/wiki/Procedural_programming',
 b'https://en.wikipedia.org/wiki/Prototype-based_programming',
 b'https://en.wikipedia.org/wiki/Reflective_programming',
 b'https://en.wikipedia.org/wiki/Rule-based_system',
 b'https://en.wikipedia.org/wiki/Scripting_language',
 b'https://en.wikipedia.org/wiki/Stack-oriented_programming',
 b'https://en.wikipedia.org/wiki/Synchronous_programming_language',
 b'https://en.wikipedia.org/wiki/Tactile_programming_language',
 b'https://en.wikipedia.org/wiki/Template_processor',
 b'https://en.wikipedia.org/wiki/Assembly_language',
 b'https://en.wikipedia.org/wiki/Compiled_language',
 b'https://en.wikipedia.org/wiki/Interpreted_language',
 b'https://en.wikipedia.org/wiki/Machine_code',
 b'https://en.wikipedia.org/wiki/Low-level_programming_language',
 b'https://en.wikipedia.org/wiki/High-level_programming_language',
 b'https://en.wikipedia.org/wiki/Very_high-level_programming_language',
 b'https://en.wikipedia.org/wiki/First-generation_programming_language',
 b'https://en.wikipedia.org/wiki/Second-generation_programming_language',
 b'https://en.wikipedia.org/wiki/Third-generation_programming_language',
 b'https://en.wikipedia.org/wiki/Fourth-generation_programming_language',
 b'https://en.wikipedia.org/wiki/Fifth-generation_programming_language',
 b'https://en.wikipedia.org/wiki/Non-English-based_programming_languages',
 b'https://en.wikipedia.org/wiki/Visual_programming_language',
 b'https://en.wikipedia.org/wiki/Template:Software_engineering',
 b'https://en.wikipedia.org/wiki/Template_talk:Software_engineering',
 b'https://en.wikipedia.org/wiki/Software_engineering',
 b'https://en.wikipedia.org/wiki/Computer_programming',
 b'https://en.wikipedia.org/wiki/DevOps',
 b'https://en.wikipedia.org/wiki/Requirements_engineering',
 b'https://en.wikipedia.org/wiki/Site_reliability_engineering',
 b'https://en.wikipedia.org/wiki/Software_deployment',
 b'https://en.wikipedia.org/wiki/Software_design',
 b'https://en.wikipedia.org/wiki/Software_maintenance',
 b'https://en.wikipedia.org/wiki/Software_testing',
 b'https://en.wikipedia.org/wiki/Systems_analysis',
 b'https://en.wikipedia.org/wiki/Formal_methods',
 b'https://en.wikipedia.org/wiki/Data_modeling',
 b'https://en.wikipedia.org/wiki/Enterprise_architecture',
 b'https://en.wikipedia.org/wiki/Functional_specification',
 b'https://en.wikipedia.org/wiki/Modeling_language',
 b'https://en.wikipedia.org/wiki/Programming_paradigm',
 b'https://en.wikipedia.org/wiki/Software',
 b'https://en.wikipedia.org/wiki/Software_archaeology',
 b'https://en.wikipedia.org/wiki/Software_architecture',
 b'https://en.wikipedia.org/wiki/Software_configuration_management',
 b'https://en.wikipedia.org/wiki/Software_development_process',
 b'https://en.wikipedia.org/wiki/Software_quality',
 b'https://en.wikipedia.org/wiki/Software_quality_assurance',
 b'https://en.wikipedia.org/wiki/Software_verification_and_validation',
 b'https://en.wikipedia.org/wiki/Structured_analysis',
 b'https://en.wikipedia.org/wiki/Essential_Systems_Analysis',
 b'https://en.wikipedia.org/wiki/Agile_software_development',
 b'https://en.wikipedia.org/wiki/Aspect-oriented_programming',
 b'https://en.wikipedia.org/wiki/Ontology_(information_science)',
 b'https://en.wikipedia.org/wiki/Service-oriented_architecture',
 b'https://en.wikipedia.org/wiki/Systems_development_life_cycle',
 b'https://en.wikipedia.org/wiki/Agile_software_development',
 b'https://en.wikipedia.org/wiki/Enterprise_Unified_Process',
 b'https://en.wikipedia.org/wiki/Executable_UML',
 b'https://en.wikipedia.org/wiki/Incremental_build_model',
 b'https://en.wikipedia.org/wiki/Iterative_and_incremental_development',
 b'https://en.wikipedia.org/wiki/Software_prototyping',
 b'https://en.wikipedia.org/wiki/Rapid_application_development',
 b'https://en.wikipedia.org/wiki/Unified_Process',
 b'https://en.wikipedia.org/wiki/Scrum_(software_development)',
 b'https://en.wikipedia.org/wiki/Spiral_model',
 b'https://en.wikipedia.org/wiki/V-Model_(software_development)',
 b'https://en.wikipedia.org/wiki/Waterfall_model',
 b'https://en.wikipedia.org/wiki/Extreme_programming',
 b'https://en.wikipedia.org/wiki/ISO/IEC_15504',
 b'https://en.wikipedia.org/wiki/Capability_Maturity_Model_Integration',
 b'https://en.wikipedia.org/wiki/Data_model',
 b'https://en.wikipedia.org/wiki/Entity%E2%80%93relationship_model',
 b'https://en.wikipedia.org/wiki/Function_model',
 b'https://en.wikipedia.org/wiki/Information_model',
 b'https://en.wikipedia.org/wiki/Metamodeling',
 b'https://en.wikipedia.org/wiki/Object_model',
 b'https://en.wikipedia.org/wiki/Systems_modeling',
 b'https://en.wikipedia.org/wiki/View_model',
 b'https://en.wikipedia.org/wiki/IDEF',
 b'https://en.wikipedia.org/wiki/Unified_Modeling_Language',
 b'https://en.wikipedia.org/wiki/Universal_Systems_Language',
 b'https://en.wikipedia.org/wiki/Systems_Modeling_Language',
 b'https://en.wikipedia.org/wiki/Computer_science',
 b'https://en.wikipedia.org/wiki/Computer_engineering',
 b'https://en.wikipedia.org/wiki/Information_science',
 b'https://en.wikipedia.org/wiki/Project_management',
 b'https://en.wikipedia.org/wiki/Risk_management',
 b'https://en.wikipedia.org/wiki/Systems_engineering',
 b'https://en.wikipedia.org/wiki/Category:Software_engineering',
 b'https://en.wikipedia.org/wiki/Help:Authority_control'] 
```

```py
crawl_persistent() 
```

```py
b'https://en.wikipedia.org/wiki/Object-orientation' 
```

## 停用词

最常见的英语单词可能会出现在每个页面上。它们并不表示页面的内容，我们可能不想对它们进行索引。有时我们不索引的单词被称为[停用词](https://en.wikipedia.org/wiki/Stop_word)。

一旦你索引了一些页面，就可以使用索引来识别出现次数最多的单词，总计在所有页面中出现的次数。

```py
word_key = 'Index:the'
r.hvals(word_key) 
```

```py
[b'81', b'277', b'49', b'535', b'118', b'53', b'118', b'161', b'152', b'227'] 
```

```py
sum(int(x) for x in r.hvals(word_key)) 
```

```py
1771 
```

```py
counter = Counter()

for word_key in r.keys('Index*'):
    total = sum(int(x) for x in r.hvals(word_key))
    word = word_key.decode().split(':')[1]
    counter[word] = total 
```

```py
counter.most_common(20) 
```

```py
[('the', 1771),
 ('of', 1138),
 ('and', 1044),
 ('a', 924),
 ('to', 797),
 ('in', 726),
 ('programming', 719),
 ('is', 528),
 ('python', 453),
 ('on', 426),
 ('from', 406),
 ('for', 404),
 ('as', 402),
 ('software', 331),
 ('languages', 331),
 ('that', 328),
 ('retrieved', 323),
 ('design', 319),
 ('with', 282),
 ('language', 270)] 
```

以下单元格使用结果制作 Zipf 图，该图显示计数与“排名”在对数-对数尺度上的关系（最常见的单词排名为 1，下一个最常见的单词排名为 2，依此类推）。

Zipf 定律断言单词频率的分布遵循幂律，这意味着 Zipf 图大致上是一条直线。

```py
import numpy as np

res = []

for i, (word, count) in enumerate(counter.most_common()):
    res.append((i+1, count))

rank, count = np.transpose(res) 
```

```py
import matplotlib.pyplot as plt

plt.plot(rank, count)
plt.xlabel('Rank')
plt.ylabel('Count')
plt.title('Zipf plot')
plt.xscale('log')
plt.yscale('log') 
```

![_images/crawler_66_0.png](img/8dd14149b85e7d84c3e23e05ac0bbc2c.png)

## 关闭

如果你在自己的计算机上运行这个笔记本，你可以使用以下命令关闭 Redis 服务器。

如果你在 Colab 上运行，这并不是真正必要的：当 Colab 运行时关闭时，Redis 服务器也会关闭（其中存储的所有内容都会消失）。

```py
!killall  redis-server 
```

*Python 中的数据结构和信息检索*

版权所有 2021 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
