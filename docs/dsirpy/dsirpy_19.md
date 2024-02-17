# 第十九章：链表

> 原文：[`allendowney.github.io/DSIRP/linked_list.html`](https://allendowney.github.io/DSIRP/linked_list.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


[单击此处在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/linked_list.ipynb)

## 链表

在链表上实现操作是编程课程和技术面试的重要内容。

我抵制它们，因为您很可能永远不会在专业工作中实现链表。如果您这样做了，那么某人已经做出了错误的决定。

但是，它们可以是很好的练习曲，也就是说，这些是您为了学习而练习但永远不会执行的曲目。

对于这些问题中的许多问题，根据要求，有几种可能的解决方案：

+   您是否允许修改现有列表，还是必须创建一个新列表？

+   如果您修改现有结构，您是否也应该返回对它的引用？

+   您是否允许分配临时结构，还是必须在原地执行所有操作？

对于所有这些问题，您都可以迭代或递归地编写解决方案。因此，每个问题都有许多可能的解决方案。

在考虑替代方案时，要牢记的一些因素是：

+   在时间和空间方面的性能。

+   可读性和可证明的正确性。

一般来说，性能应该是渐进有效的；例如，如果有一个常数时间的解决方案，线性解决方案是不可接受的。但是，我们可能愿意支付一些开销来实现无懈可击的正确性。

这是我们将用来表示列表中节点的类。

```py
class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __repr__(self):
        return f'Node({self.data}, {repr(self.next)})' 
```

我们可以这样创建节点：

```py
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)

node1 
```

```py
Node(1, None) 
```

然后像这样链接它们：

```py
node1.next = node2
node2.next = node3 
```

```py
node1 
```

```py
Node(1, Node(2, Node(3, None))) 
```

有两种方法可以思考`node1`是什么：

+   它“只是”一个节点对象，恰好包含对另一个节点的链接。

+   它是节点链表中的第一个节点。

当我们将一个节点作为参数传递时，有时我们认为它是一个节点，有时我们认为它是列表的开头。

## LinkedList 对象

对于一些操作，拥有另一个表示整个列表的对象将很方便（而不是它的一个节点）。

这是类的定义。

```py
class LinkedList:
    def __init__(self, head=None):
        self.head = head

    def __repr__(self):
        return f'LinkedList({repr(self.head)})' 
```

如果我们创建一个带有对`node1`的引用的`LinkedList`，我们可以将结果视为具有三个元素的列表。

```py
t = LinkedList(node1)
t 
```

```py
LinkedList(Node(1, Node(2, Node(3, None)))) 
```

## 搜索

**练习：**编写一个名为`find`的函数，该函数接受一个`LinkedList`和一个目标值；如果目标值出现在`LinkedList`中，则应返回包含它的`Node`；否则应返回`None`。

您可以使用这些示例来测试您的代码。

```py
find(t, 1) 
```

```py
Node(1, Node(2, Node(3, None))) 
```

```py
find(t, 3) 
```

```py
Node(3, None) 
```

```py
find(t, 5) 
```

## 推和弹

在链表的*左*侧添加和删除元素相对容易：

```py
def lpush(t, value):
    t.head = Node(value, t.head) 
```

```py
t = LinkedList()
lpush(t, 3)
lpush(t, 2)
lpush(t, 1)
t 
```

```py
LinkedList(Node(1, Node(2, Node(3, None)))) 
```

```py
def lpop(t):
    if t.head is None:
        raise ValueError('Tried to pop from empty LinkedList')
    node = t.head
    t.head = node.next
    return node.data 
```

```py
lpop(t), lpop(t), lpop(t) 
```

```py
(1, 2, 3) 
```

```py
t 
```

```py
LinkedList(None) 
```

从右侧添加和删除需要更长的时间，因为我们必须遍历列表。

**练习：**编写`rpush`和`rpop`。

您可以使用以下示例来测试您的代码。

```py
t = LinkedList()
rpush(t, 1)
t 
```

```py
LinkedList(Node(1, None)) 
```

```py
rpush(t, 2)
t 
```

```py
LinkedList(Node(1, Node(2, None))) 
```

```py
rpop(t) 
```

```py
2 
```

```py
rpop(t) 
```

```py
1 
```

```py
try:
    rpop(t)
except ValueError as e:
    print(e) 
```

```py
Tried to rpop from an empty list 
```

## 反转

反转链表是一个经典的面试问题，尽管在这一点上它是如此经典，您可能永远不会遇到它。

但是，这仍然是一个很好的练习，部分原因是有很多方法可以做到这一点。我这里的解决方案是基于[这个教程](https://www.geeksforgeeks.org/reverse-a-linked-list/)。

如果允许创建一个新列表，您可以遍历旧列表并将元素`lpush`到新列表上：

```py
def reverse(t):
    t2 = LinkedList()
    node = t.head
    while node:
        lpush(t2, node.data)
        node = node.next

    return t2 
```

```py
t = LinkedList(Node(1, Node(2, Node(3, None))))
reverse(t) 
```

```py
LinkedList(Node(3, Node(2, Node(1, None)))) 
```

这是一个不分配任何东西的递归版本

```py
def reverse(t):
    t.head = reverse_rec(t.head)

def reverse_rec(node):

    # if there are 0 or 1 nodes
    if node is None or node.next is None:
        return node

    # reverse the rest LinkedList
    rest = reverse_rec(node.next)

    # Put first element at the end
    node.next.next = node
    node.next = None

    return rest 
```

```py
t = LinkedList(Node(1, Node(2, Node(3, None))))
reverse(t)
t 
```

```py
LinkedList(Node(3, Node(2, Node(1, None)))) 
```

最后，这是一个不分配任何东西的迭代版本。

```py
def reverse(t):
    prev = None
    current = t.head
    while current :
        next = current.next
        current.next = prev
        prev = current
        current = next
    t.head = prev 
```

```py
t = LinkedList(Node(1, Node(2, Node(3, None))))
reverse(t)
t 
```

```py
LinkedList(Node(3, Node(2, Node(1, None)))) 
```

## 删除

链表的一个优点（与数组列表相比）是我们可以在列表的中间以常数时间添加和删除元素。

例如，以下函数接受一个节点并删除其后的节点。

```py
def remove_after(node):
    removed = node.next
    node.next = node.next.next
    return removed.data 
```

这是一个例子：

```py
t = LinkedList(Node(1, Node(2, Node(3, None))))
remove_after(t.head)
t 
```

```py
LinkedList(Node(1, Node(3, None))) 
```

**练习：**编写一个名为`remove`的函数，该函数接受一个 LinkedList 和一个目标值。它应该删除包含该值的第一个节点，如果找不到，则引发`ValueError`。

提示：这个有点棘手。

您可以使用此示例来测试您的代码。

```py
t = LinkedList(Node(1, Node(2, Node(3, None))))
remove(t, 2)
t 
```

```py
LinkedList(Node(1, Node(3, None))) 
```

```py
remove(t, 1)
t 
```

```py
LinkedList(Node(3, None)) 
```

```py
try:
    remove(t, 4)
except ValueError as e:
    print(e) 
```

```py
Value not found 
```

```py
remove(t, 3)
t 
```

```py
LinkedList(None) 
```

```py
try:
    remove(t, 5)
except ValueError as e:
    print(e) 
```

```py
Value not found (empty list) 
```

虽然`remove_after`是常数时间，但`remove`不是。因为我们必须遍历节点以找到目标，`remove`需要线性时间。

## 插入排序

同样，您可以在常数时间内将元素插入到链表的中间。

以下函数在列表中的给定节点后插入`data`。

```py
def insert_after(node, data):
    node.next = Node(data, node.next) 
```

```py
t = LinkedList(Node(1, Node(2, Node(3, None))))
insert_after(t.head, 5)
t 
```

```py
LinkedList(Node(1, Node(5, Node(2, Node(3, None))))) 
```

**练习：**编写一个名为`insert_sorted`（也称为`insort`）的函数，它接受一个链表和一个值，并在列表中按照递增排序顺序将该值插入到第一个位置，即最小的元素在开头。

```py
def insert_sorted(t, data):
    if t.head is None or t.head.data > data:
        lpush(t, data)
        return

    node = t.head
    while node.next:
        if node.next.data > data:
            insert_after(node, data)
            return
        node = node.next

    insert_after(node, data) 
```

您可以使用以下示例来测试您的代码。

```py
t = LinkedList()
insert_sorted(t, 1)
t 
```

```py
LinkedList(Node(1, None)) 
```

```py
insert_sorted(t, 3)
t 
```

```py
LinkedList(Node(1, Node(3, None))) 
```

```py
insert_sorted(t, 0)
t 
```

```py
LinkedList(Node(0, Node(1, Node(3, None)))) 
```

```py
insert_sorted(t, 2)
t 
```

```py
LinkedList(Node(0, Node(1, Node(2, Node(3, None))))) 
```

虽然`insert_after`是常数时间，但`insert_sorted`不是。因为我们必须遍历节点以找到插入点，`insert_sorted`需要线性时间。

*Python 中的数据结构和信息检索*

版权所有 2021 年 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
