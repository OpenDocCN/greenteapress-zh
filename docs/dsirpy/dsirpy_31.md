# 第三十一章：PageRank

> 原文：[`allendowney.github.io/DSIRP/pagerank.html`](https://allendowney.github.io/DSIRP/pagerank.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


[点击这里在 Colab 上运行这个笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/pagerank.ipynb)

## 页面排名

信息检索的目标是找到相关和高质量的资源。“相关”和“质量”可能很难定义，它们取决于你正在搜索的资源类型。

在网络搜索的背景下，相关性通常取决于网页的内容：如果网页包含搜索词，我们认为它与搜索词相关。

质量通常取决于页面之间的链接。如果有许多链接指向特定页面，我们认为它更有可能是高质量的，特别是如果这些链接来自高质量的页面。

其中一个最早量化质量的算法是 PageRank，它是 Google 最初搜索引擎的核心。作为一个使用过早期搜索引擎（如 Alta Vista）的人，我可以亲口告诉你它带来了多大的改变。

PageRank 在 Page、Brin、Motwani 和 Winograd 的[“The PageRank citation ranking: Bringing order to the Web”](https://web.archive.org/web/20110818093436/http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)中有描述。

这是一个令人惊讶的简单算法；它可以高效地计算和更新；它在识别高质量页面方面非常有效。所以，让我们看看它是如何工作的。

举例来说，我将使用`random_k_out_graph`生成一个有`n`个节点的有向图：

+   每个节点有相同数量的出链，`k`，

+   入链的数量变化适中（由参数`alpha`控制）。

+   自链接和多链接都是允许的。

```py
import networkx as nx

G = nx.random_k_out_graph(n=8, k=2, alpha=0.75) 
```

这是图的样子。多个链接显示为略微粗的线。

```py
def draw_graph(G):
    nx.draw_circular(G, node_size=400, with_labels=True)

draw_graph(G) 
```

![_images/pagerank_6_0.png](img/d2eeb27b59adfdff0e5831278f78fe33.png)

NetworkX 提供了 PageRank 的实现，我们可以用它来计算每个页面的“重要性”指标。

```py
ranks_pr = nx.pagerank(G)
ranks_pr 
```

```py
{0: 0.1529214627122818,
 1: 0.3707896283472802,
 2: 0.14402452847002478,
 3: 0.018750000000000003,
 4: 0.18430400881769338,
 5: 0.018750000000000003,
 6: 0.018750000000000003,
 7: 0.09171037165271977} 
```

你应该看到有更多入链的节点得到了更高的分数。

## 随机游走

解释 PageRank 的一种方式是随机游走。假设你随机选择一个节点，然后随机选择其中一个出链，然后继续这样做，记录你访问的每个节点。

如果一个节点有很多入链，你可能会更频繁地访问它。如果这些入链来自有很多入链的节点，情况会更加如此。

然而，有一个问题：如果一个节点不包含出链，或者一组节点形成一个没有出链的循环，随机行走者可能会被困住。

为了避免这种情况，我们将修改随机游走，使得在每一步中，行走者有一定的概率跳转到一个随机节点，而不是按照链接进行。这个概率由参数`alpha`确定，它是按照链接的概率，所以`1-alpha`是进行随机跳转的概率。

以下函数实现了一个随机游走，带有这些随机跳转，并使用`Counter`来跟踪它访问每个节点的次数。

它返回`Counter`，其频率归一化为总和为一。如果一切按计划进行，这些值应该接近 PageRank 的结果。

```py
import numpy as np

def flip(p):
    return np.random.random() < p 
```

```py
from collections import Counter

def random_walk(G, alpha=0.85, iters=1000):
    counter = Counter()
    node = next(iter(G))

    for _ in range(iters):
        if flip(alpha):
            node = np.random.choice(list(G[node]))
        else:
            node = np.random.choice(list(G))

        counter[node] += 1

    total = sum(counter.values())
    for key in counter:
        counter[key] /= total
    return counter 
```

`alpha`的默认值是 0.85，这与`nx.pagerank`的默认值相同。

这是我们从随机游走中得到的分数。

```py
ranks_rw = random_walk(G)
ranks_rw 
```

```py
Counter({7: 0.093,
         2: 0.145,
         1: 0.39,
         4: 0.19,
         6: 0.019,
         0: 0.133,
         3: 0.009,
         5: 0.021}) 
```

为了将它们与 PageRank 的结果进行比较，我将它们放入 Pandas 的`DataFrame`中。

```py
import pandas as pd

s1 = pd.Series(ranks_pr)
s2 = pd.Series(ranks_rw)

df = pd.DataFrame(dict(PageRank=s1, RandomWalk=s2))
df['Diff'] = df['RandomWalk'] - df['PageRank']
df*100 
```

|  | PageRank | RandomWalk | Diff |
| --- | --- | --- | --- |
| 0 | 15.292146 | 13.3 | -1.992146 |
| 1 | 37.078963 | 39.0 | 1.921037 |
| 2 | 14.402453 | 14.5 | 0.097547 |
| 3 | 1.875000 | 0.9 | -0.975000 |
| 4 | 18.430401 | 19.0 | 0.569599 |
| 5 | 1.875000 | 2.1 | 0.225000 |
| 6 | 1.875000 | 1.9 | 0.025000 |
| 7 | 9.171037 | 9.3 | 0.128963 |

差异最多应该在几个百分点之间。

## 邻接矩阵

PageRank 的随机行走实现在概念上很简单，但计算起来并不是很高效。另一种方法是使用一个矩阵来表示每个节点到每个其他节点的链接，并计算该矩阵的特征向量。

在本节中，我将演示这个计算并解释它是如何工作的。这里的代码是基于[NetworkX 中 PageRank 的实现](https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html)。

NetworkX 提供了一个函数，用于创建表示图的[邻接矩阵](https://en.wikipedia.org/wiki/Adjacency_matrix)的 NumPy 数组。

```py
M = nx.to_numpy_array(G)
M 
```

```py
array([[1., 0., 0., 0., 0., 0., 0., 1.],
       [0., 1., 0., 0., 1., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 1.],
       [0., 1., 1., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 1., 0., 0., 0.],
       [1., 1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 1., 0., 0., 0., 0., 0.]]) 
```

在这个矩阵中，第`i`行第`j`列的元素表示从节点`i`到节点`j`的边的数量。

如果我们对每一行进行归一化，使其总和为一，结果的每个元素表示从一个节点到另一个节点的转移概率。

```py
M /= M.sum(axis=1)
M 
```

```py
array([[0.5, 0\. , 0\. , 0\. , 0\. , 0\. , 0\. , 0.5],
       [0\. , 0.5, 0\. , 0\. , 0.5, 0\. , 0\. , 0\. ],
       [0.5, 0.5, 0\. , 0\. , 0\. , 0\. , 0\. , 0\. ],
       [0\. , 0\. , 0.5, 0\. , 0\. , 0\. , 0\. , 0.5],
       [0\. , 0.5, 0.5, 0\. , 0\. , 0\. , 0\. , 0\. ],
       [0\. , 0.5, 0\. , 0\. , 0.5, 0\. , 0\. , 0\. ],
       [0.5, 0.5, 0\. , 0\. , 0\. , 0\. , 0\. , 0\. ],
       [0\. , 0.5, 0.5, 0\. , 0\. , 0\. , 0\. , 0\. ]]) 
```

我们可以使用这个矩阵来模拟同时有许多行走者的随机行走。例如，假设我们从每个节点开始有 100 个行走者，用数组`x`表示：

```py
N = len(G)
x = np.full(N, 100)
x 
```

```py
array([100, 100, 100, 100, 100, 100, 100, 100]) 
```

如果我们对`M`进行转置，我们会得到一个[转移矩阵](https://en.wikipedia.org/wiki/Stochastic_matrix)，其中第`i`行第`j`列的元素是从节点`j`移动到节点`i`的行走者的比例。

如果我们将转移矩阵乘以`x`，结果是一个数组，其中包含每个节点在一个时间步长后的行走者数量。

```py
x = M.T @ x
x 
```

```py
array([150., 300., 150.,   0., 100.,   0.,   0., 100.]) 
```

如果你运行那个单元格几次，你应该会发现它会收敛到一个稳定状态，在这个状态下，每个节点的行走者数量在一个时间步长到下一个时间步长几乎不会改变。

然而，你可能会注意到一些节点失去了所有的行走者。这是因为我们模拟的随机行走不包括随机跳跃。

要添加随机跳跃，我们可以创建另一个矩阵，其中包括从所有节点到所有其他节点的等概率转移。

```py
p = np.full((N, N), 1/N)
p 
```

```py
array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
       [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
       [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
       [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
       [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
       [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
       [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
       [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]) 
```

现在我们将使用参数`alpha`来计算`M`和`p`的加权和。

```py
alpha = 0.85
GM = alpha * M + (1 - alpha) * p 
```

结果是一个代表随机行走中转移的“Google 矩阵”，包括随机跳跃。

所以让我们再次从所有节点开始有相同数量的行走者，并模拟 10 个时间步长。

```py
x = np.full(N, 100)

for i in range(10):
    x = GM.T @ x 
```

如果我们对`x`进行归一化，使其总和为一，结果应该接近我们从 PageRank 得到的排名。

```py
ranks_am = x / x.sum()
ranks_am 
```

```py
array([0.15293199, 0.37077494, 0.14404034, 0.01875   , 0.18427767,
       0.01875   , 0.01875   , 0.09172506]) 
```

这是一个比较结果的表格。

```py
import pandas as pd

s1 = pd.Series(ranks_pr)
s2 = pd.Series(ranks_am)

df = pd.DataFrame(dict(PageRank=s1, AdjMatrix=s2))
df['Diff'] = df['AdjMatrix'] - df['PageRank']
df*100 
```

|  | PageRank | AdjMatrix | Diff |
| --- | --- | --- | --- |
| 0 | 15.292146 | 15.293199 | 0.001053 |
| 1 | 37.078963 | 37.077494 | -0.001469 |
| 2 | 14.402453 | 14.404034 | 0.001581 |
| 3 | 1.875000 | 1.875000 | 0.000000 |
| 4 | 18.430401 | 18.427767 | -0.002634 |
| 5 | 1.875000 | 1.875000 | 0.000000 |
| 6 | 1.875000 | 1.875000 | 0.000000 |
| 7 | 9.171037 | 9.172506 | 0.001469 |

## 特征向量

如果你从几乎任何向量开始，并重复地乘以一个矩阵，就像我们在前一节中所做的那样，结果将收敛到与矩阵对应的最大特征值的特征向量。

事实上，重复乘法是计算特征值的算法之一：它被称为[幂迭代](https://en.wikipedia.org/wiki/Power_iteration)。

我们也可以直接计算特征值，而不是使用迭代方法，这就是 Numpy 函数`eig`所做的。这里是 Google 矩阵的特征值和特征向量。

```py
eigenvalues, eigenvectors = np.linalg.eig(GM.T)
eigenvalues 
```

```py
array([ 1.00000000e+00+0.j       ,  4.25000000e-01+0.j       ,
       -2.12500000e-01+0.3680608j, -2.12500000e-01-0.3680608j,
        8.61220879e-17+0.j       ,  2.07547158e-18+0.j       ,
       -1.82225683e-17+0.j       ,  0.00000000e+00+0.j       ]) 
```

这是我们如何得到与最大特征值对应的特征向量。

```py
ind = np.argmax(eigenvalues)
ind, eigenvalues[ind] 
```

```py
(0, (0.9999999999999993+0j)) 
```

```py
largest = eigenvectors[:, ind]
largest 
```

```py
array([-0.32235148+0.j, -0.78161291+0.j, -0.30359969+0.j, -0.03952437+0.j,
       -0.38850772+0.j, -0.03952437+0.j, -0.03952437+0.j, -0.19332161+0.j]) 
```

结果包含复数，但虚部都是 0，所以我们可以只取实部。

```py
largest = largest.real 
```

然后对其进行归一化。

```py
ranks_ev = largest / largest.sum()
ranks_ev 
```

```py
array([0.15292059, 0.37079   , 0.14402491, 0.01875   , 0.1843045 ,
       0.01875   , 0.01875   , 0.09171   ]) 
```

结果是基于 Google 矩阵的特征向量的排名集。它们应该与 PageRank 的结果相同，除了小的浮点误差。

```py
import pandas as pd

s1 = pd.Series(ranks_pr)
s2 = pd.Series(ranks_ev)

df = pd.DataFrame(dict(PageRank=s1, Eigenvector=s2))
df['Diff'] = df['Eigenvector'] - df['PageRank']
df*100 
```

|  | PageRank | Eigenvector | Diff |
| --- | --- | --- | --- |
| 0 | 15.292146 | 15.292059 | -8.752734e-05 |
| 1 | 37.078963 | 37.079000 | 3.719912e-05 |
| 2 | 14.402453 | 14.402491 | 3.839473e-05 |
| 3 | 1.875000 | 1.875000 | 1.734723e-15 |
| 4 | 18.430401 | 18.430450 | 4.913262e-05 |
| 5 | 1.875000 | 1.875000 | 1.040834e-15 |
| 6 | 1.875000 | 1.875000 | 1.040834e-15 |
| 7 | 9.171037 | 9.171000 | -3.719912e-05 |

## 把所有东西放在一起

以下是计算 Google 矩阵和 PageRank 分数的 NetworkX 函数的简化版本。

```py
def google_matrix(G, alpha=0.85):
  """Returns the Google matrix of the graph.

 Parameters
 ----------
 G : graph
 A NetworkX graph.  Undirected graphs will be converted to a directed
 graph with two directed edges for each undirected edge.

 alpha : float
 The damping factor.

 Notes
 -----
 The matrix returned represents the transition matrix that describes the
 Markov chain used in PageRank. For PageRank to converge to a unique
 solution (i.e., a unique stationary distribution in a Markov chain), the
 transition matrix must be irreducible. In other words, it must be that
 there exists a path between every pair of nodes in the graph, or else there
 is the potential of "rank sinks."
 """
    M = np.asmatrix(nx.to_numpy_array(G))
    N = len(G)
    if N == 0:
        return M

    # Personalization vector
    p = np.repeat(1.0 / N, N)

    # Dangling nodes
    dangling_weights = p
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes 
    # (nodes with no out links)
    for node in dangling_nodes:
        M[node] = dangling_weights

    M /= M.sum(axis=1)  # Normalize rows to sum to 1

    return alpha * M + (1 - alpha) * p 
```

```py
def pagerank_numpy(G, alpha=0.85):
  """Returns the PageRank of the nodes in the graph.

 PageRank computes a ranking of the nodes in the graph G based on
 the structure of the incoming links. It was originally designed as
 an algorithm to rank web pages.

 Parameters
 ----------
 G : graph
 A NetworkX graph.  Undirected graphs will be converted to a directed
 graph with two directed edges for each undirected edge.

 alpha : float, optional
 Damping parameter for PageRank, default=0.85.

 Returns
 -------
 pagerank : dictionary
 Dictionary of nodes with PageRank as value.

 Examples
 --------
 >>> G = nx.DiGraph(nx.path_graph(4))
 >>> pr = nx.pagerank_numpy(G, alpha=0.9)

 Notes
 -----
 The eigenvector calculation uses NumPy's interface to the LAPACK
 eigenvalue solvers.  This will be the fastest and most accurate
 for small graphs.

 References
 ----------
 .. [1] A. Langville and C. Meyer,
 "A survey of eigenvector methods of web information retrieval."
 http://citeseer.ist.psu.edu/713792.html
 .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
 The PageRank citation ranking: Bringing order to the Web. 1999
 http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
 """
    if len(G) == 0:
        return {}
    M = google_matrix(G, alpha)

    # use numpy LAPACK solver
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)

    # eigenvector of largest eigenvalue is at ind, normalized
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm))) 
```

```py
pagerank_numpy(G) 
```

```py
{0: 0.15292058743886122,
 1: 0.370790000338484,
 2: 0.14402491241728307,
 3: 0.01875000000000002,
 4: 0.1843045001438557,
 5: 0.018750000000000013,
 6: 0.018750000000000013,
 7: 0.09170999966151594} 
```

*Python 中的数据结构和信息检索*

版权所有 2021 年 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
