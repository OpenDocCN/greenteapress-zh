# 测验 3

> 原文：[`allendowney.github.io/DSIRP/quiz03.html`](https://allendowney.github.io/DSIRP/quiz03.html)

在开始这个测验之前：

1.  点击“复制到驱动器”以复制测验，

1.  点击“分享”,

1.  点击“更改”，然后选择“任何人都可以编辑”

1.  点击“复制链接”和

1.  将链接粘贴到[此 Canvas 作业](https://canvas.olin.edu/courses/313/assignments/4985)。

这个测验是开放笔记，开放互联网。你唯一不能做的事情就是寻求帮助。

版权所有 2021 年 Allen Downey，[MIT 许可证](http://opensource.org/licenses/MIT)

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

## 问题 1

以下是从`search.ipynb`中实现的二叉搜索树（BST）。

```py
class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __repr__(self):
        return f'Node({self.data}, {repr(self.left)}, {repr(self.right)})' 
```

```py
class BSTree:
    def __init__(self, root=None):
        self.root = root

    def __repr__(self):
        return f'BSTree({repr(self.root)})' 
```

```py
def insert(tree, data):
    tree.root = insert_rec(tree.root, data)

def insert_rec(node, data):
    if node is None:
        return Node(data)

    if data < node.data:
        node.left = insert_rec(node.left, data)
    else:
        node.right = insert_rec(node.right, data)

    return node 
```

以下单元格从文件中读取单词并将它们添加到 BST 中。 但是如果你运行它，你会得到一个`RecursionError`。

```py
filename = 'american-english'
tree = BSTree()
for line in open(filename):
    for word in line.split():
        insert(tree, word.strip()) 
```

```py
---------------------------------------------------------------------------
RecursionError  Traceback (most recent call last)
<ipython-input-5-51d6de872b69> in <module>
  3 for line in open(filename):
  4     for word in line.split():
----> 5         insert(tree, word.strip())

<ipython-input-4-6a6c90da7b3d> in insert(tree, data)
  1 def insert(tree, data):
----> 2     tree.root = insert_rec(tree.root, data)
  3 
  4 def insert_rec(node, data):
  5     if node is None:

<ipython-input-4-6a6c90da7b3d> in insert_rec(node, data)
  9         node.left = insert_rec(node.left, data)
  10     else:
---> 11         node.right = insert_rec(node.right, data)
  12 
  13     return node

... last 1 frames repeated, from the frame below ...

<ipython-input-4-6a6c90da7b3d> in insert_rec(node, data)
  9         node.left = insert_rec(node.left, data)
  10     else:
---> 11         node.right = insert_rec(node.right, data)
  12 
  13     return node

RecursionError: maximum recursion depth exceeded 
```

但是，如果我们将单词放入列表中，对列表进行洗牌，然后将洗牌后的单词放入 BST 中，它就可以工作。

```py
word_list = []
for line in open(filename):
    for word in line.split():
        word_list.append(word.strip()) 
```

```py
from random import shuffle

shuffle(word_list) 
```

```py
tree = BSTree()
for word in word_list:
    insert(tree, word.strip()) 
```

写几个清晰、完整的句子来回答以下两个问题：

1.  我们为什么会得到`RecursionError`，为什么洗牌单词会解决问题？

1.  整个过程的增长顺序是什么；也就是说，将单词读入列表，对列表进行洗牌，然后将洗牌后的单词放入二叉搜索树。您可以假设`shuffle`是线性的。

## 问题 2

正如我们在课堂上讨论的那样，搜索问题有三个版本：

1.  检查元素是否在集合中；例如，这就是`in`运算符的作用。

1.  在有序集合中查找元素的索引；例如，这就是字符串方法`find`的作用。

1.  在键值对的集合中，找到与给定键对应的值；这就是字典方法`get`的作用。

在`search.ipynb`中，我们使用了 BST 来解决第一个问题。在这个练习中，您将修改它以解决第三个问题。

这里是代码（尽管请注意对象的名称是`MapNode`和`BSTMap`）。

```py
class MapNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __repr__(self):
        return f'Node({self.data}, {repr(self.left)}, {repr(self.right)})' 
```

```py
class BSTMap:
    def __init__(self, root=None):
        self.root = root

    def __repr__(self):
        return f'BSTMap({repr(self.root)})' 
```

```py
def insert_map(tree, data):
    tree.root = insert_map_rec(tree.root, data)

def insert_map_rec(node, data):
    if node is None:
        return MapNode(data)

    if data < node.data:
        node.left = insert_map_rec(node.left, data)
    else:
        node.right = insert_map_rec(node.right, data)

    return node 
```

修改此代码，以便它存储键和值，而不仅仅是集合的元素。然后编写一个名为`get`的函数，该函数接受`BSTMap`和一个键：

+   如果键在地图中，则应返回相应的值；

+   否则，它应该引发一个带有适当消息的`KeyError`。

您可以使用以下代码来测试您的实现。

```py
tree_map = BSTMap()

keys = 'uniqueltrs'
values = range(len(keys))
for key, value in zip(keys, values):
    print(key, value)
    insert_map(tree_map, key, value)

tree_map 
```

```py
u 0
n 1
i 2
q 3
u 4
e 5
l 6
t 7
r 8
s 9 
```

```py
BSTree(MapNode(u, MapNode(n, MapNode(i, MapNode(e, None, None), MapNode(l, None, None)), MapNode(q, None, MapNode(t, MapNode(r, None, MapNode(s, None, None)), None))), MapNode(u, None, None))) 
```

```py
for key in keys:
    print(key, get(tree_map, key)) 
```

```py
u 0
n 1
i 2
q 3
u 0
e 5
l 6
t 7
r 8
s 9 
```

以下应该引发一个`KeyError`。

```py
get(tree_map, 'b') 
```

## 替代解决方案

修改此代码，以便它存储键和值，而不仅仅是集合的元素。然后编写一个名为`get`的函数，该函数接受`BSTMap`和一个键：

+   如果键在地图中，则应返回相应的值；

+   否则，它应该引发一个带有适当消息的`KeyError`。

您可以使用以下代码来测试您的实现。

```py
tree_map = BSTMap()

keys = 'uniqueltrs'
values = range(len(keys))
for key, value in zip(keys, values):
    print(key, value)
    insert_map(tree_map, key, value)

tree_map 
```

```py
u 0
n 1
i 2
q 3
u 4
e 5
l 6
t 7
r 8
s 9 
```

```py
BSTree(MapNode(('u', 0), MapNode(('n', 1), MapNode(('i', 2), MapNode(('e', 5), None, None), MapNode(('l', 6), None, None)), MapNode(('q', 3), None, MapNode(('t', 7), MapNode(('r', 8), None, MapNode(('s', 9), None, None)), None))), MapNode(('u', 4), None, None))) 
```

```py
for key in keys:
    print(key, get(tree_map, key)) 
```

```py
u 0
n 1
i 2
q 3
u 0
e 5
l 6
t 7
r 8
s 9 
```
