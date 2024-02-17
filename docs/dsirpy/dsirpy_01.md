# 介绍

> 原文：[`allendowney.github.io/DSIRP/`](https://allendowney.github.io/DSIRP/)

*Python 中的数据结构和信息检索*是使用 Web 搜索引擎作为引人注目的例子介绍数据结构和算法。它在一定程度上基于使用 Java 的*[Think Data Structures](https://greenteapress.com/wp/think-data-structures/)*。

搜索引擎的元素包括：

+   爬虫，下载网页并跟踪到其他页面的链接，

+   索引器，它构建了从每个搜索词到出现的页面的映射，以及

+   检索器，查找搜索词并找到相关的高质量页面。

索引存储在 Redis 中，它是一个提供集合、列表和哈希映射等结构的数据存储。该书首先用 Python 呈现每个数据结构，然后用 Redis 呈现，这应该有助于读者看到哪些功能是必要的，哪些是实现细节。

就像我在[*Think Bayes*](https://greenteapress.com/wp/think-bayes/)中所做的那样，我完全是用 Jupyter 笔记本写的这本书，并使用 JupyterBook 将它们转换成 HTML。笔记本在 Colab 上运行，这是谷歌提供的一个在浏览器中运行笔记本的服务。因此，您可以阅读这本书，运行代码，并在不安装任何东西的情况下进行练习。

这个材料还在不断完善中，所以欢迎您的反馈。提供反馈的最佳方式是[点击这里，在 GitHub 存储库中创建一个问题](https://github.com/AllenDowney/DSIRP/issues)。

[概述幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vRFFocqlEH4YAbi8_xgZhfx9cvHFdMkhx_-yQ2aVVqc5quUQlm_mhuu7XoE9UOARsvwDe9X0kcA2DqS/pub)

## 笔记本

点击下面的链接在 Colab 上运行笔记本。

+   算法：第一天的活动检查变位词和找到变位词集。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/algorithms.ipynb)

+   分析：算法分析和大 O 符号介绍。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQXYlOUlPPTE9GGR3UBugxYT8n_TcIGR5ttG7Rz_aA8lAFLTCeYUC1HFnQyDQBKPOv6PC7_PQ5Q-xz6/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/analysis.ipynb)

+   时间：通过测量运行时间来检查渐近行为。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/timing.ipynb)

+   测验 1

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/quiz01.ipynb)

+   生成器函数：将迭代与程序逻辑分开

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vTOxX01R5LNdEZDqSkiG5YOlJQieAO2bePigUnz6Fx5fiJqTMtpoOzn0ltpaeuWbfLl74vz6YqWUmZK/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/generator.ipynb)

+   集合：使用 Python 集合在单词游戏中作弊。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/set.ipynb)

+   递归：练习递归函数。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/recursion.ipynb)

+   测验 2

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/quiz02.ipynb)

+   深度优先搜索：BeautifulSoup 中的树遍历。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vTQzIt8u_vdwhqeFjPIHUNDFlO0_2-GId567gTbSCtyfQM0nRWjlxbklUhWTGl4KDzVI4_JxcfYRfEa/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/dfs.ipynb)

+   搜索：线性搜索，二分法和二叉搜索树。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQItNQPqCoUITZggi-ML-OYZtecevxcsPVvbP1JvW55erx2tXaO3cibTrWE5E8myJ4wqRPLt7xby7ei/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/searching.ipynb)

+   哈希映射：最伟大的数据结构如何工作。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQXOQd5jpi4eHfIg9iqPCOSLVFEnaAvAiFhBAGZECl0wZ2XKJdbMSnGZsym8CvVq-IsxvvKu1tB7e2L/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/hashmap.ipynb)

+   测验 3

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/quiz03.ipynb)

+   堆：它是一个数组，它是一棵树，它是一个优先队列！

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQTHKlq7pvrOCgqgPhLodGUtrcA3sFGco4r8O041WvmKLi-JFDfUPpb4X6txEn1qe2RR_xBfvXlXtSD/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/heap.ipynb)

+   赫夫曼编码：使用我们学到的结构制作最佳前缀代码。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQjk8Ko3u59qdandz-R_KfmQiHc2oIBk5RcJlWMXubdIMDxYuZpVHqn26jLylm0_eMf_ZJ-rOgnBjpi/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/huffman.ipynb)

+   到达哲学：跟随维基百科链接，直到到达哲学。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQKVxHQKnp4LoiDipCvMh6GFRhgdiNFG_fqJ6vOfFb-ai9S1jLLbFvR1Qp4ocaAMNGL2FSaUd3-3H62/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/philosophy.ipynb)

+   测验 4

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/quiz04.ipynb)

+   Redis：介绍 Redis 数据存储。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/redis.ipynb)

+   链表：列表之前是树？奇怪但真实。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vRSKmupEcVRXzH4jj31Zk5To6PrmIej58HviUrbN0a7wKTKBZwdoVHcGSFKvWac-L1w3Js9R6eD33fn/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/linked_list.ipynb)

+   索引器：为快速查找制作互联网地图。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/indexer.ipynb)

+   测验 5

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/quiz05.ipynb)

+   双端队列：像链表一样，但更多。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/deque.ipynb)

+   图：用 NetworkX 表示图。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/graph.ipynb)

+   层次遍历搜索：使用`os`模块遍历文件系统。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQT31xIq3pY-JF9J2RezS-i3528RM-NSpa67PN3wjfNF_6T0uUw_pV253lFKCB7pc_zXsnglXKOU2Pw/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/level_order.ipynb)

+   广度优先搜索：图算法的基础。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vRXakv4ZkGq648UwqRCXUkmqUFwGx4kJ4OskY6F9_busCH2aXPjZKKsQhGP4ESdJJNDq8bJowB9zLJb/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/bfs.ipynb)

+   测验 6

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/quiz06.ipynb)

+   爬虫：跟踪链接并对互联网进行广度优先搜索。

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/crawler.ipynb)

+   归并排序：分治和在线性对数时间内合并。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vQbgVZohGR3tSm7LtnYVravKt_za_70Egy4hQwpGeLsjvhfmG16QfBjhph991EsIWsrfyABsRMmMAMk/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/mergesort.ipynb)

+   快速傅里叶变换：就像归并排序，但使用复数。

    [幻灯片](https://docs.google.com/presentation/d/e/2PACX-1vRuShFoETvJiCPAiM1xbxDBIM6MaXh2kMpjYB3FvRB4xzYsfi3vgZYgoQbxtGq8ODLjC8qhwn17f2_V/pub)

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/fft.ipynb)

+   测验 7 [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/quiz07.ipynb)

+   PageRank：随机游走，邻接矩阵和特征向量！

    幻灯片

    [笔记本](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/pagerank.ipynb)

2021 年艾伦 B.唐尼版权所有

许可证：[署名-非商业性使用-相同方式共享 4.0 国际（CC BY-NC-SA 4.0）](https://creativecommons.org/licenses/by-nc-sa/4.0/)
