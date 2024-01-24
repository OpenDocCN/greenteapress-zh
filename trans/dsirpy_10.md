# 深度优先搜索

> 原文：[`allendowney.github.io/DSIRP/dfs.html`](https://allendowney.github.io/DSIRP/dfs.html)

[单击此处在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/dfs.ipynb)

本笔记本将“深度优先搜索”作为一种遍历树中节点的方法。这个算法适用于任何类型的树，但由于我们需要一个例子，我们将使用 BeautifulSoup，它是一个读取 HTML（和相关语言）并构建代表内容的树的 Python 模块。

## 使用 BeautifulSoup

当您下载网页时，内容是用超文本标记语言（HTML）编写的。例如，这是一个最小的 HTML 文档，我从[BeautifulSoup 文档](https://beautiful-soup-4.readthedocs.io)中借来的。文本来自刘易斯·卡罗尔的《爱丽丝梦游仙境》。

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

这是我们如何使用 BeautifulSoup 来读取它。

```py
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_doc)
type(soup) 
```

```py
bs4.BeautifulSoup 
```

结果是一个代表树根的`BeautifulSoup`对象。如果我们显示这个 soup，它会重现 HTML。

```py
soup 
```

```py
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
</body></html> 
```

`prettify`使用缩进来显示文档的结构。

```py
print(soup.prettify()) 
```

```py
<html>
 <head>
  <title>
   The Dormouse's story
  </title>
 </head>
 <body>
  <p class="title">
   <b>
    The Dormouse's story
   </b>
  </p>
  <p class="story">
   Once upon a time there were three little sisters; and their names were
   <a class="sister" href="http://example.com/elsie" id="link1">
    Elsie
   </a>
   ,
   <a class="sister" href="http://example.com/lacie" id="link2">
    Lacie
   </a>
   and
   <a class="sister" href="http://example.com/tillie" id="link3">
    Tillie
   </a>
   ;
and they lived at the bottom of a well.
  </p>
  <p class="story">
   ...
  </p>
 </body>
</html> 
```

`BeautifulSoup`对象有一个名为`children`的属性，返回它包含的对象的迭代器。

```py
soup.children 
```

```py
<list_iterator at 0x7f4cf816b850> 
```

我们可以使用 for 循环来遍历它们。

```py
for element in soup.children:
    print(type(element)) 
```

```py
<class 'bs4.element.Tag'> 
```

这个 soup 只包含一个孩子，那就是一个`Tag`。

`BeautifulSoup`还提供了`contents`，它以列表的形式返回孩子，这可能更方便。

```py
soup.contents 
```

```py
[<html><head><title>The Dormouse's story</title></head>
 <body>
 <p class="title"><b>The Dormouse's story</b></p>
 <p class="story">Once upon a time there were three little sisters; and their names were
 <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
 and they lived at the bottom of a well.</p>
 <p class="story">...</p>
 </body></html>] 
```

唯一的孩子是包含整个文档的 HTML 元素。让我们只获取这个元素：

```py
element = soup.contents[0]
element 
```

```py
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
</body></html> 
```

元素的类型是`Tag`：

```py
type(element) 
```

```py
bs4.element.Tag 
```

标签的名称是`html`。

```py
element.name 
```

```py
'html' 
```

现在让我们获取这个顶层元素的孩子：

```py
children = element.contents
children 
```

```py
[<head><title>The Dormouse's story</title></head>,
 '\n',
 <body>
 <p class="title"><b>The Dormouse's story</b></p>
 <p class="story">Once upon a time there were three little sisters; and their names were
 <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
 <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
 <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
 and they lived at the bottom of a well.</p>
 <p class="story">...</p>
 </body>] 
```

这个列表中有三个元素，但很难阅读，因为当您打印一个元素时，它会打印所有的 HTML。

我将使用以下函数以简单的形式打印元素。

```py
from bs4 import Tag, NavigableString

def print_element(element):
    if isinstance(element, Tag):
        print(f'{type(element).__name__}<{element.name}>')
    if isinstance(element, NavigableString):
        print(type(element).__name__) 
```

```py
print_element(element) 
```

```py
Tag<html> 
```

以及以下函数来打印元素列表。

```py
def print_element_list(element_list):
    print('[')
    for element in element_list:
        print_element(element)
    print(']') 
```

```py
print_element_list(element.contents) 
```

```py
[
Tag<head>
NavigableString
Tag<body>
] 
```

现在让我们尝试导航树。我将从`element`的第一个孩子开始。

```py
child = element.contents[0]
print_element(child) 
```

```py
Tag<head> 
```

并打印它的孩子。

```py
print_element_list(child.contents) 
```

```py
[
Tag<title>
] 
```

现在让我们获取第一个孩子的第一个孩子。

```py
grandchild = child.contents[0]
print_element(grandchild) 
```

```py
Tag<title> 
```

```py
grandchild = child.contents[0]
print_element(grandchild) 
```

```py
Tag<title> 
```

第一个孙子的第一个孩子。

```py
greatgrandchild = grandchild.contents[0]
print_element(greatgrandchild) 
```

```py
NavigableString 
```

```py
try:
    greatgrandchild.contents
except AttributeError as e:
    print('AttributeError:', e) 
```

```py
AttributeError: 'NavigableString' object has no attribute 'contents' 
```

```py
greatgrandchild 
```

```py
"The Dormouse's story" 
```

`NavigableString`没有孩子，所以我们已经到了尽头。

为了继续，我们需要回溯到孙子并选择第二个孩子。

这意味着我们必须跟踪我们已经看到的元素，以便从我们离开的地方继续。

这就是深度优先搜索的作用。

## 深度优先搜索

DFS 从树的根开始并选择第一个孩子。如果孩子有孩子，它再次选择第一个孩子。当它到达一个没有孩子的节点时，它回溯，向上移动到父节点，如果有下一个孩子，则再次选择；否则再次回溯。当它探索完根的最后一个孩子时，它就完成了。

有两种常见的实现 DFS 的方法，递归和迭代。递归实现如下：

```py
def recursive_DFS(element):
    if isinstance(element, NavigableString):
        print(element, end='')
        return

    for child in element.children:
        recursive_DFS(child) 
```

```py
recursive_DFS(soup) 
```

```py
The Dormouse's story

The Dormouse's story
Once upon a time there were three little sisters; and their names were
Elsie,
Lacie and
Tillie;
and they lived at the bottom of a well.
... 
```

这是一个使用列表表示元素堆栈的 DFS 的迭代版本：

```py
def iterative_DFS(root):
    stack = [root]

    while(stack):
        element = stack.pop()
        if isinstance(element, NavigableString):
            print(element, end='')
        else:
            children = reversed(element.contents)
            stack.extend(children) 
```

参数`root`是我们要遍历的树的根，所以我们首先创建堆栈并将根推入其中。

循环继续直到堆栈为空。每次循环时，它从堆栈中弹出一个`PageElement`。如果它得到一个`NavigableString`，它就打印内容。然后将孩子推入堆栈。为了以正确的顺序处理孩子，我们必须以相反的顺序将它们推入堆栈。

```py
iterative_DFS(soup) 
```

```py
The Dormouse's story

The Dormouse's story
Once upon a time there were three little sisters; and their names were
Elsie,
Lacie and
Tillie;
and they lived at the bottom of a well.
... 
```

**练习：**编写一个类似于`PageElement.find`的函数，它接受一个`PageElement`和一个标签名称，并返回具有给定名称的第一个标签。您可以以迭代或递归的方式编写它。

这是如何检查`PageElement`是否为`Tag`。

```py
from bs4 import Tag
isinstance(element, Tag) 
```

```py
def is_right_tag(element, tag_name):
    return (isinstance(element, Tag) and 
            element.name == tag_name) 
```

**练习：**编写一个类似于`PageElement.find_all`的生成器函数，它接受一个`PageElement`和一个标签名称，并产生所有具有给定名称的标签。您可以以迭代或递归的方式编写它。

*Python 中的数据结构和信息检索*

版权所有 2021 年 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
