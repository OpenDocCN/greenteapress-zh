# 第十七章：测验 4

> 原文：[`allendowney.github.io/DSIRP/quiz04.html`](https://allendowney.github.io/DSIRP/quiz04.html)

在开始这个测验之前：

1.  点击“复制到驱动器”以复制测验，

1.  点击“分享”，

1.  点击“更改”，然后选择“任何拥有此链接的人都可以编辑”

1.  点击“复制链接”和

1.  将链接粘贴到[此 Canvas 作业](https://canvas.olin.edu/courses/313/assignments/5032)中。

这个测验是开放笔记，开放互联网。唯一不能做的就是寻求帮助。

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

根据[Wikipedia](https://en.wikipedia.org/wiki/Gray_code)，格雷码是“二进制数制的一种排序，使得连续的两个值在只有一个位（二进制数字）上不同”。

“格雷码列表”是一个表，按顺序列出了每个十进制数的格雷码。例如，以下是直到 3 的十进制数的格雷码列表：

```py
number    Gray code
------    ---------
0         00
1         01
2         11
3         10 
```

在这段代码中，数字 3 的表示是位序列`10`。

[Wikipedia 页面的这一部分](https://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code)提供了一个用于构建具有给定二进制数字数量的格雷码列表的算法。

编写一个名为`gray_code`的函数，该函数以二进制数字`n`的数量作为参数，并返回表示格雷码列表的字符串列表。

例如，`gray_code(3)`应返回

```py
['000', '001', '011', '010', '110', '111', '101', '100'] 
```

您的函数可以是迭代的或递归的。

```py
# Buggy solution

def gray_code(n, codes=['0', '1']):
    if n <= 1:
        return codes

    r = codes[::-1]

    for i, code in enumerate(codes):
        codes[i] = '0' + code

    for i, code in enumerate(r):
        r[i] = '1' + code

    codes.extend(r)

    return gray_code(n-1, codes) 
```

您可以使用以下单元格来测试您的解决方案。

```py
gray_code(1)   # should be ['0', '1'] 
```

```py
['0', '1'] 
```

```py
gray_code(2)   # should be ['00', '01', '11', '10'] 
```

```py
['00', '01', '11', '10'] 
```

```py
gray_code(3)   # see above 
```

```py
['0000',
 '0001',
 '0011',
 '0010',
 '0110',
 '0111',
 '0101',
 '0100',
 '1100',
 '1101',
 '1111',
 '1110',
 '1010',
 '1011',
 '1001',
 '1000'] 
```

```py
gray_code(4)   # see above 
```

```py
['0000000',
 '0000001',
 '0000011',
 '0000010',
 '0000110',
 '0000111',
 '0000101',
 '0000100',
 '0001100',
 '0001101',
 '0001111',
 '0001110',
 '0001010',
 '0001011',
 '0001001',
 '0001000',
 '0011000',
 '0011001',
 '0011011',
 '0011010',
 '0011110',
 '0011111',
 '0011101',
 '0011100',
 '0010100',
 '0010101',
 '0010111',
 '0010110',
 '0010010',
 '0010011',
 '0010001',
 '0010000',
 '0110000',
 '0110001',
 '0110011',
 '0110010',
 '0110110',
 '0110111',
 '0110101',
 '0110100',
 '0111100',
 '0111101',
 '0111111',
 '0111110',
 '0111010',
 '0111011',
 '0111001',
 '0111000',
 '0101000',
 '0101001',
 '0101011',
 '0101010',
 '0101110',
 '0101111',
 '0101101',
 '0101100',
 '0100100',
 '0100101',
 '0100111',
 '0100110',
 '0100010',
 '0100011',
 '0100001',
 '0100000',
 '1100000',
 '1100001',
 '1100011',
 '1100010',
 '1100110',
 '1100111',
 '1100101',
 '1100100',
 '1101100',
 '1101101',
 '1101111',
 '1101110',
 '1101010',
 '1101011',
 '1101001',
 '1101000',
 '1111000',
 '1111001',
 '1111011',
 '1111010',
 '1111110',
 '1111111',
 '1111101',
 '1111100',
 '1110100',
 '1110101',
 '1110111',
 '1110110',
 '1110010',
 '1110011',
 '1110001',
 '1110000',
 '1010000',
 '1010001',
 '1010011',
 '1010010',
 '1010110',
 '1010111',
 '1010101',
 '1010100',
 '1011100',
 '1011101',
 '1011111',
 '1011110',
 '1011010',
 '1011011',
 '1011001',
 '1011000',
 '1001000',
 '1001001',
 '1001011',
 '1001010',
 '1001110',
 '1001111',
 '1001101',
 '1001100',
 '1000100',
 '1000101',
 '1000111',
 '1000110',
 '1000010',
 '1000011',
 '1000001',
 '1000000'] 
```

## 问题 2

假设您有一个非常大的数字序列，并且要求找到`k`个最大的元素。一种选择是对序列进行排序，但这将花费与序列长度`n`成正比的时间。而且您将不得不存储整个序列。

另一种方法是使用“有界堆”，即永远不包含超过`k`个元素的堆。

编写一个名为`k_largest`的函数，该函数以可迭代对象和整数`k`作为参数，并返回一个包含可迭代对象中`k`个最大元素的列表。不用担心并列。

您的实现不应存储超过`k`个元素，并且它的时间复杂度应与`n log k`成正比。

您可以使用以下单元格来测试您的函数。

```py
from random import shuffle

sequence = list(range(10))
shuffle(sequence)
sequence 
```

```py
[4, 3, 0, 7, 1, 5, 9, 6, 8, 2] 
```

```py
k_largest(sequence, 3)   # should return [7, 8, 9] 
```

```py
[7, 9, 8] 
```

## 问题 3

表达式树是表示数学表达式的树。例如，表达式`(1+2) * 3`由根节点处的乘法运算符和两个子节点表示：

+   左子节点是一个包含加法运算符和两个子节点的节点，数字 1 和数字 2。

+   右子节点是一个包含数字 3 的节点。

要表示表达式树，我们可以使用一个名为`Node`的`namedtuple`，其中包含三个属性，`data`，`left`和`right`。

```py
from collections import namedtuple

Node = namedtuple('Node', ['data', 'left', 'right']) 
```

在叶节点中，`data`包含一个数字。例如，这里有两个表示数字`1`和`2`的节点。

```py
operand1 = Node(1, None, None)
operand1 
```

```py
Node(data=1, left=None, right=None) 
```

```py
operand2 = Node(2, None, None)
operand2 
```

```py
Node(data=2, left=None, right=None) 
```

对于内部节点（即非叶节点），`data`包含一个函数。为了表示加法、减法和乘法，我将从`operator`模块导入函数。

```py
from operator import add, sub, mul 
```

现在我们可以用`add`函数作为根节点，两个操作数作为子节点来构建一个表达式树。

```py
etree = Node(add, operand1, operand2)
etree 
```

```py
Node(data=<built-in function add>, left=Node(data=1, left=None, right=None), right=Node(data=2, left=None, right=None)) 
```

要评估这棵树，我们可以提取函数和两个操作数，然后调用函数并将操作数作为参数传递。

```py
func = etree.data
left = operand1.data
right = operand2.data
func(left, right) 
```

```py
3 
```

编写一个名为`evaluate`的函数，该函数接受任意表达式树，对其进行评估，并返回一个整数。

您可能希望以递归的方式编写这个。

你可以使用以下示例测试您的函数：

```py
etree 
```

```py
Node(data=<built-in function add>, left=Node(data=1, left=None, right=None), right=Node(data=2, left=None, right=None)) 
```

```py
evaluate(etree)  # result should be 3 
```

```py
3 
```

```py
operand3 = Node(3, None, None)
etree2 = Node(mul, etree, operand3) 
```

```py
evaluate(etree2)  # result should be 9 
```

```py
9 
```

```py
operand4 = Node(4, None, None)
etree3 = Node(sub, etree2, operand4) 
```

```py
evaluate(etree3) # result should be 5 
```

```py
5 
```
