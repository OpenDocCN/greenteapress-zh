# 测验 7

> 原文：[`allendowney.github.io/DSIRP/quiz07.html`](https://allendowney.github.io/DSIRP/quiz07.html)

在开始这个测验之前：

1.  点击“复制到驱动器”以复制测验，

1.  点击“分享”，

1.  点击“更改”，然后选择“任何人都可以编辑”

1.  点击“复制链接”和

1.  将链接粘贴到[此 Canvas 作业](https://canvas.olin.edu/courses/313/assignments/5183)中。

这个测验是开放笔记，开放互联网。

+   你可以向导师寻求帮助，但不能向其他人寻求帮助。

+   你可以使用在互联网上找到的代码，但如果你从单个来源使用了超过几行，你应该注明出处。

## 问题 1

某个函数被递归地定义如下：

$$ f(n, m) = f(n-1, m-1) + f(n-1, m) $$

有两种特殊情况：如果$m=0$或者$m=n$，函数的值为 1。

编写一个名为`f`的（Python）函数来计算这个（数学）函数。

你可以使用以下示例来测试你的函数。

```py
assert f(2, 1) == 2 
```

```py
assert f(4, 1) == 4 
```

```py
assert f(4, 2) == 6 
```

```py
assert f(5, 3) == 10 
```

```py
assert f(10, 5) == 252 
```

如果你尝试运行以下示例，你会发现它运行了很长时间。

```py
# f(100, 50) 
```

## 问题 2

编写一个名为`f_memo`的`f`版本，它使用适当的 Python 数据结构来“记忆化”`f`。换句话说，你应该记录你已经计算过的结果，并查找它们，而不是重新计算它们。

在 recursion.ipynb 中有一个记忆化的例子。

你可以使用这个示例来确认函数仍然有效。

```py
f_memo(10, 5) 
```

```py
252 
```

并使用这个示例来确认它更快。它应该少于一秒，结果应该是`100891344545564193334812497256`。

```py
%time f_memo(100, 50) 
```

```py
CPU times: user 2.7 ms, sys: 0 ns, total: 2.7 ms
Wall time: 2.7 ms 
```

```py
100891344545564193334812497256 
```

## LetterSet

接下来的两个问题基于我将称之为`LetterSet`的集合实现。

> 注意：在这个问题陈述中，“集合”指的是集合的概念，而不是 Python 对象`set`。在这些问题中，我们不会使用任何 Python`set`对象。

如果你提前知道集合中可能出现的元素，你可以使用[位数组](https://en.wikipedia.org/wiki/Bit_array)来有效地表示集合。例如，要表示一组字母，你可以使用一个包含 26 个布尔值的列表，每个字母在罗马字母表中都有一个（忽略大小写）。

这是一个表示集合的类定义。

```py
class LetterSet:
    def __init__(self, bits=None):
        if bits is None:
            bits = [False]*26
        self.bits = bits

    def __repr__(self):
        return f'LetterSet({repr(self.bits)})' 
```

如果列表中的所有元素都为 False，则集合为空。

```py
set1 = LetterSet()
set1 
```

```py
LetterSet([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]) 
```

要向集合添加一个字母，我们必须计算与给定字母对应的索引。以下函数使用内置的 Python 函数`ord`来计算给定字母的索引。

```py
def get_index(letter):
    return ord(letter.lower()) - ord('a') 
```

`a`的索引为 0，`Z`的索引为 25。

```py
get_index('a'), get_index('Z') 
```

```py
(0, 25) 
```

要添加一个字母，我们将列表的相应元素设置为`True`。

```py
def add(ls, letter):
    ls.bits[get_index(letter)] = True 
```

```py
add(set1, 'a')
add(set1, 'Z')
set1 
```

```py
LetterSet([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True]) 
```

要计算集合的元素数，我们可以使用内置的`sum`函数：

```py
def size(ls):
    return sum(ls.bits) 
```

```py
size(set1) 
```

```py
2 
```

## 问题 3

编写一个名为`is_in`的函数，它接受一个集合和一个字母，并在字母在集合中时返回`True`。在注释中，确定这个函数的增长顺序。

使用以下示例来测试你的代码。

```py
is_in(set1, 'a'), is_in(set1, 'b') 
```

```py
(True, False) 
```

## 问题 4

编写一个名为`intersect`的函数，它接受两个`LetterSet`对象，并返回一个表示两个集合交集的新的`LetterSet`。换句话说，新的`LetterSet`应该只包含出现在两个集合中的元素。

在注释中，确定这个函数的增长顺序。

使用以下示例来测试你的代码。

```py
intersect(set1, set1) 
```

```py
LetterSet([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True]) 
```

```py
set2 = LetterSet()
add(set2, 'a')
add(set2, 'b') 
```

```py
set3 = intersect(set1, set2)
set3 
```

```py
LetterSet([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]) 
```

```py
size(set3) 
```

```py
1 
```

## 只是为了有趣的奖励问题

表示大数字的一种方法是使用一个链表，其中每个节点包含一个数字。

这里有`DigitList`的类定义，它表示一个数字列表，以及`Node`，它包含一个数字和对列表中下一个`Node`的引用。

```py
class DigitList:
    def __init__(self, head=None):
        self.head = head 
```

```py
class Node:
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next 
```

在`DigitList`中，数字以相反的顺序存储，因此一个包含数字`1`、`2`和`3`的列表，按照顺序表示数字`321`。

```py
head = Node(1, Node(2, Node(3, None)))
head 
```

```py
<__main__.Node at 0x7f8dd0766940> 
```

```py
dl321 = DigitList(head)
dl321 
```

```py
<__main__.DigitList at 0x7f8dd0766370> 
```

以下函数接受一个`DigitList`并以相反的顺序打印数字。

```py
def print_dl(dl):
    print_dl_rec(dl.head)
    print()

def print_dl_rec(node):
    if node is not None:
        print_dl_rec(node.next)
        print(node.data, end='') 
```

```py
print_dl(dl321) 
```

```py
321 
```

```py
head = Node(4, Node(5, Node(6, None)))
dl654 = DigitList(head)
print_dl(dl654) 
```

```py
654 
```

编写一个名为`add`的函数，它接受两个`DigitList`对象，并返回一个表示它们之和的新的`DigitList`。

```py
divmod(11, 10) 
```

```py
(1, 1) 
```

你可以使用以下示例来测试你的代码。

```py
total = add(dl321, dl654)
print_dl(total)
321 + 654 
```

```py
975 
```

```py
975 
```

```py
head = Node(7, Node(8, None))
dl87 = DigitList(head)
print_dl(dl87) 
```

```py
87 
```

```py
print_dl(add(dl654, dl87))
654+87 
```

```py
741 
```

```py
741 
```

```py
print_dl(add(dl87, dl654))
87+654 
```

```py
741 
```

```py
741 
```

```py
zero = DigitList(None)
print_dl(add(dl87, zero))
87 + 0 
```

```py
87 
```

```py
87 
```

```py
print_dl(add(zero, dl87))
0 + 87 
```

```py
87 
```

```py
87 
```

*Python 中的数据结构和信息检索*

版权所有 2021 年 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
