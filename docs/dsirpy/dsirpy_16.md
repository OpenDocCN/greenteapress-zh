# 到达哲学

> 原文：[`allendowney.github.io/DSIRP/philosophy.html`](https://allendowney.github.io/DSIRP/philosophy.html)

[单击此处在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/philosophy.ipynb)

# 到达哲学

本笔记本的目标是开发一个 Web 爬虫，测试“到达哲学”猜想。如[维基百科页面](https://en.wikipedia.org/wiki/Wikipedia:Getting_to_Philosophy)上所解释的：

> 单击英文维基百科文章的主文本中的第一个链接，然后重复这个过程以获取后续文章，通常会导航到哲学文章。2016 年 2 月，这对维基百科所有文章的 97%都成立……

更具体地说，链接不能在括号或斜体中，并且不能是外部链接，指向当前页面的链接或指向不存在页面的链接。

我们将使用`urllib`库下载维基百科页面，并使用 BeautifulSoup 解析 HTML 文本并导航文档对象模型（DOM）。

在开始处理维基百科页面之前，让我们先用一个最小的 HTML 文档热身，我从 BeautifulSoup 文档中改编而来。

```py
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
(<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>),
<i><a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and</i>
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
""" 
```

这个文档包含三个链接，但第一个在括号中，第二个是斜体，所以第三个是我们将要跟随的链接，以到达哲学。

这是我们如何解析这个文档并制作一个`BeautifulSoup`对象。

```py
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_doc)
type(soup) 
```

```py
bs4.BeautifulSoup 
```

要遍历 DOM 中的元素，我们可以编写自己的深度优先搜索实现，就像这样：

```py
def iterative_DFS(root):
    stack = [root]

    while(stack):
        element = stack.pop()
        yield element

        children = getattr(element, "contents", [])
        stack.extend(reversed(children)) 
```

例如，我们可以遍历元素并打印所有`NavigableString`元素：

```py
from bs4 import NavigableString

for element in iterative_DFS(soup):
    if isinstance(element, NavigableString):
        print(element.string, end='') 
```

```py
The Dormouse's story

The Dormouse's story
Once upon a time there were three little sisters; and their names were
(Elsie),
Lacie and
Tillie;
and they lived at the bottom of a well.
... 
```

但我们也可以使用`descendants`，它的功能相同。

```py
for element in soup.descendants:
    if isinstance(element, NavigableString):
        print(element.string, end='') 
```

```py
The Dormouse's story

The Dormouse's story
Once upon a time there were three little sisters; and their names were
(Elsie),
Lacie and
Tillie;
and they lived at the bottom of a well.
... 
```

## 检查括号

软件开发的一个理论认为，你应该先解决最难的问题，因为它会推动设计。然后你可以想出如何处理更容易的问题。

对于“到达哲学”，其中一个更难的问题是弄清楚链接是否在括号中。如果你有一个链接，你可以一直向外寻找括号，但在树中，这可能会变得复杂。

我选择的另一种方法是按顺序遍历文本，计算开放和关闭括号，并仅在它们没有被包含时产生链接。

```py
from bs4 import Tag

def link_generator(root):
    paren_stack = []

    for element in root.descendants:
        if isinstance(element, NavigableString):
            for char in element.string:
                if char == '(':
                    paren_stack.append(char)
                if char == ')':
                    paren_stack.pop()

        if isinstance(element, Tag) and element.name == "a":
            if len(paren_stack) == 0:
                yield element 
```

现在我们可以遍历不在括号中的链接。

```py
for link in link_generator(soup):
    print(link) 
```

```py
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a> 
```

## 检查斜体

要查看链接是否为斜体，我们可以：

1.  如果其父级是名称为`a`的`Tag`，它就是斜体。

1.  否则，我们必须检查父级的父级，依此类推。

1.  如果我们到达根节点而没有找到斜体标签，那么它就不是斜体。

例如，这是从`link_generator`中获取的第一个链接。

```py
link = next(link_generator(soup))
link 
```

```py
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> 
```

它的父标签是斜体标签。

```py
parent = link.parent
isinstance(parent, Tag) 
```

```py
True 
```

```py
parent.name 
```

```py
'i' 
```

**练习：**编写一个名为`in_italics`的函数，它接受一个元素并在它是斜体时返回`True`。

然后编写一个更通用的函数，名为`in_bad_element`，它接受一个元素并在以下情况下返回`True`：

+   元素或其祖先之一具有“不良”标签名称，如`i`，或

+   元素或其祖先之一是一个`div`，其`class`属性包含“不良”类名。

我们将需要这个函数的一般版本来排除维基百科页面上的无效链接。

## 与维基百科页面一起工作

实际的维基百科页面比简单的示例更复杂，因此需要一些努力来理解它们的结构，并确保我们选择正确的“第一个链接”。

以下单元格下载了 Python 的维基百科页面。

```py
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local) 
```

```py
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
download(url) 
```

现在我们可以解析它并制作`soup`。

```py
filename = basename(url)
fp = open(filename)
soup2 = BeautifulSoup(fp) 
```

如果您使用 Web 浏览器查看此页面，并使用检查元素工具来探索结构，您会看到文章的正文部分在一个带有类名`mw-body-content`的`div`元素中。

我们可以使用`find`来获取这个元素，并将其用作我们搜索的根。

```py
root = soup2.find(class_='mw-body-content') 
```

**练习：**编写一个名为`valid_link_generator`的生成器函数，它使用`link_generator`来查找不在括号中的链接；然后它应该过滤掉无效的链接，包括斜体链接，外部页面的链接等。

使用不同的页面测试您的函数，直到它可靠地找到似乎最符合规则精神的“第一个链接”。

## `WikiFetcher`

当您编写 Web 爬虫时，很容易下载太多的页面太快，这可能违反您正在下载的服务器的服务条款。为了避免这种情况，我们将使用一个名为`WikiFetcher`的对象，它有两个作用：

1.  它封装了下载和解析网页的代码。

1.  它测量请求之间的时间，如果我们在请求之间没有留足够的时间，它会休眠，直到合理的间隔时间过去。默认情况下，间隔是一秒。

这是`WikiFetcher`的定义：

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

`fetch_wikipedia`接受一个 URL 作为`String`，并返回一个代表页面内容的 BeautifulSoup 对象。

`sleep_if_needed`检查自上次请求以来的时间，并在经过的时间少于`min_interval`时休眠。

以下是一个演示它如何使用的示例：

```py
wf = WikiFetcher()
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

print(time())
wf.fetch_wikipedia(url)
print(time())
wf.fetch_wikipedia(url)
print(time()) 
```

```py
1640031013.2612915
1640031013.7938814
1640031014.7832372 
```

如果一切按计划进行，三个时间戳的间隔不应少于 1 秒，这与维基百科的[robots.txt](https://en.wikipedia.org/robots.txt)中的条款一致：

> 欢迎友好、低速的机器人查看文章页面，但请勿查看动态生成的页面。

**练习：**现在让我们把所有东西都整合起来。编写一个名为`get_to_philosophy`的函数，它以维基百科页面的 URL 作为参数。它应该：

1.  使用我们刚刚创建的`WikiFetcher`对象来下载和解析页面。

1.  遍历生成的`BeautifulSoup`对象，以找到符合规则精神的第一个有效链接。

1.  如果页面没有链接，或者第一个链接是我们已经看过的页面，程序应该指示失败并退出。

1.  如果链接与维基百科哲学页面的 URL 匹配，程序应该指示成功并退出。

1.  否则，它应该回到步骤 1（尽管您可能希望对跳数设置限制）。

程序应该构建一个它访问的 URL 列表，并在最后显示结果（无论成功与否）。

由于您找到的链接是相对的，您可能会发现`urljoin`函数有用：

```py
from urllib.parse import urljoin

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
relative_path = "/wiki/Interpreted_language"

urljoin(url, relative_path) 
```

```py
'https://en.wikipedia.org/wiki/Interpreted_language' 
```

```py
get_to_philosophy(url) 
```

```py
https://en.wikipedia.org/wiki/Python_(programming_language)
https://en.wikipedia.org/wiki/Interpreted_language
https://en.wikipedia.org/wiki/Computer_science
https://en.wikipedia.org/wiki/Computation
https://en.wikipedia.org/wiki/Calculation
https://en.wikipedia.org/wiki/Arithmetic
https://en.wikipedia.org/wiki/Mathematics
https://en.wikipedia.org/wiki/Epistemology
https://en.wikipedia.org/wiki/Outline_of_philosophy
Got there in 9 steps! 
```

```py
['https://en.wikipedia.org/wiki/Python_(programming_language)',
 'https://en.wikipedia.org/wiki/Interpreted_language',
 'https://en.wikipedia.org/wiki/Computer_science',
 'https://en.wikipedia.org/wiki/Computation',
 'https://en.wikipedia.org/wiki/Calculation',
 'https://en.wikipedia.org/wiki/Arithmetic',
 'https://en.wikipedia.org/wiki/Mathematics',
 'https://en.wikipedia.org/wiki/Epistemology',
 'https://en.wikipedia.org/wiki/Outline_of_philosophy'] 
```
