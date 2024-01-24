# 数据科学要素

> 原文：[`allendowney.github.io/ElementsOfDataScience/README.html`](https://allendowney.github.io/ElementsOfDataScience/README.html)

《数据科学要素》是为没有编程经验的人们介绍数据科学的。我的目标是呈现 Python 的一个小而强大的子集，使您能够尽快在数据科学中进行真正的工作。

我不假设读者对编程、统计学或数据科学有任何了解。当我使用一个术语时，我会立即定义它，当我使用一个编程功能时，我会尝试解释它。

这本书是以 Jupyter 笔记本的形式呈现的。Jupyter 是一个可以在 Web 浏览器中运行的软件开发工具，因此您不需要安装任何软件。Jupyter 笔记本是一个包含文本、Python 代码和结果的文档。因此，您可以像阅读书籍一样阅读它，但您也可以修改代码、运行代码、开发新程序并对其进行测试。

这些笔记本包含练习，您可以在其中练习所学的知识。大多数练习都是快速的，但有一些更实质性。

这些材料还在不断完善中，因此欢迎提出建议。提供反馈的最佳方式是[点击此处并在此 GitHub 存储库中创建一个问题](https://github.com/AllenDowney/ElementsOfDataScience/issues)。

## 案例研究

除了下面的笔记本之外，《数据科学要素》课程还包括以下案例研究：

+   [政治立场案例研究](https://allendowney.github.io/PoliticalAlignmentCaseStudy/)：使用来自美国普遍社会调查的数据，这个案例研究探讨了受访者在各种话题上的观点如何随时间变化以及这些变化如何与政治立场（保守、中间或自由）相关。读者可以选择大约 120 个调查问题之一，看看回答如何随时间变化以及这些变化如何与政治立场相关。

+   [再犯案案例研究](https://allendowney.github.io/RecidivismCaseStudy/)：这个案例研究基于一篇著名的文章“机器偏见”，该文章于 2016 年由 Politico 发表。它涉及到 COMPAS，这是一个在刑事司法系统中用于评估被告如果获释会再次犯罪风险的统计工具。ProPublica 的文章得出结论，COMPAS 对黑人被告不公平，因为他们更有可能被错误分类为高风险。《华盛顿邮报》上的一篇回应文章指出“事实并不那么清楚”。利用原始文章中的数据，这个案例研究解释了用于评估二元分类器的（许多）指标，展示了定义算法公平性的挑战，并开始讨论数据科学的背景、伦理和社会影响。

+   [Bayes 的微观概率](https://allendowney.github.io/BiteSizeBayes/)：介绍概率，重点介绍 Bayes 定理。

+   [Python 中的天文数据](https://allendowney.github.io/AstronomicalData/)：使用来自盖亚空间望远镜的数据作为示例，介绍 SQL。

## 这些笔记本

对于下面的每个笔记本，您有三个选项：

+   如果您在 NBViewer 上查看笔记本，您可以阅读它，但无法运行代码。

+   如果您在 Colab 上运行笔记本，您将能够运行代码，做练习，并将修改后的笔记本保存在 Google Drive 中（如果您有一个）。

+   或者，如果您下载笔记本，您可以在自己的环境中运行它。但在这种情况下，您需要确保您拥有所需的库。

### 笔记本 1

**变量和值**：第一个笔记本解释了如何使用 Jupyter，并介绍了变量、值和数值计算。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/01_variables.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/01_variables.ipynb)

### 笔记本 2

时间和地点：本笔记本展示了如何在 Python 中表示时间、日期和位置，并使用 GeoPandas 库在地图上绘制点。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/02_times.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/02_times.ipynb)

### 笔记本 3

列表和数组：本笔记本介绍了列表和 NumPy 数组。它讨论了绝对误差、相对误差和百分比误差，以及总结它们的方法。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/03_arrays.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/03_arrays.ipynb)

### 笔记本 4

循环和文件：本笔记本介绍了`for`循环和`if`语句；然后使用它们来快速阅读《战争与和平》，并计算单词数。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/04_loops.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/04_loops.ipynb)

### 笔记本 5

字典：本笔记本介绍了 Python 最强大的功能之一，字典，并使用它们来计算文本中唯一单词及其频率。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/05_dictionaries.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/05_dictionaries.ipynb)

### 笔记本 6

绘图：本笔记本介绍了绘图库 Matplotlib，并使用它生成一些常见的数据可视化和一个不太常见的 Zipf 图。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/06_plotting.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/06_plotting.ipynb)

### 笔记本 7

数据框架：本笔记本介绍了数据框架，用于表示数据表。例如，它使用来自美国家庭增长调查的数据来找到婴儿的平均体重。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/07_dataframes.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/07_dataframes.ipynb)

### 笔记本 8

分布：本笔记本解释了分布是什么，并介绍了表示分布的 3 种方法：PMF、CDF 或 PDF。它还展示了如何将一个分布与另一个分布或数学模型进行比较。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/08_distributions.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/08_distributions.ipynb)

### 笔记本 9

关系：本笔记本探讨了使用散点图、小提琴图和箱线图来探索变量之间的关系。它使用相关系数量化关系的强度，并使用简单回归估计线的斜率。

[点击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/09_relationships.ipynb)

[或点击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/09_relationships.ipynb)

### 笔记本 10

回归：本笔记本介绍了多元回归，并使用它来探索年龄、教育和收入之间的关系。它使用可视化来解释多元模型。它还介绍了二元变量和逻辑回归。

单击此处在 Colab 上运行此笔记本

或单击此处下载它

### 笔记本 11

**重抽样**：这本笔记本介绍了我们可以使用的计算方法，以量化由于随机抽样而产生的变化，这是统计估计中几种误差来源之一。

单击此处在 Colab 上运行此笔记本

或单击此处下载它

### 笔记本 12

**自助法**：自助法是一种适合我们一直在处理的调查数据的重抽样方法。

单击此处在 Colab 上运行此笔记本

或单击此处下载它

### 笔记本 13

**假设检验**：假设检验是古典统计学的大敌。这本笔记本提出了一种计算方法，使得清楚地表明[只有一个测试](http://allendowney.blogspot.com/2016/06/there-is-still-only-one-test.html)。

单击此处在 Colab 上运行此笔记本

或单击此处下载它
