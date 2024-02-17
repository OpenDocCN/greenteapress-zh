# 前言

> 原文：[`allendowney.github.io/ThinkBayes2/preface.html`](https://allendowney.github.io/ThinkBayes2/preface.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


这本书的前提，以及*Think X*系列中的其他书籍，是如果你知道如何编程，你可以利用这一技能来学习其他主题。

大多数关于贝叶斯统计的书籍使用数学符号，并使用数学概念如微积分来呈现思想。这本书使用 Python 代码和离散逼近代替连续数学。因此，在数学书中的积分变成了求和，而概率分布上的大多数操作都是循环或数组操作。

我认为这种表达方式更容易理解，至少对于具有编程技能的人来说是这样。它也更加通用，因为当我们做建模决策时，我们可以选择最合适的模型，而不用太担心模型是否适合数学分析。

此外，它提供了从简单示例到现实世界问题的平稳路径。

## 这本书适合谁？

要开始阅读这本书，你应该对 Python 感到舒适。如果你熟悉 NumPy 和 Pandas，那会有所帮助，但我会在我们进行时解释你需要的内容。你不需要了解微积分或线性代数。你不需要任何统计学的先验知识。

在第一章中，我定义了概率并介绍了条件概率，这是贝叶斯定理的基础。第三章介绍了概率分布，这是贝叶斯统计的基础。

在后面的章节中，我们使用各种离散和连续分布，包括二项式、指数、泊松、贝塔、伽玛和正态分布。我会在引入每个分布时解释它们，并且我们将使用 SciPy 来计算它们，因此你不需要了解它们的数学属性。

## 建模

这本书的大多数章节都是由一个现实世界的问题所激发的，因此它们涉及一定程度的建模。在我们应用贝叶斯方法（或任何其他分析）之前，我们必须对包括在模型中的现实世界系统的哪些部分以及我们可以抽象掉的哪些细节做出决策。

例如，在第八章中，激励问题是预测足球比赛的赢家。我将进球建模为泊松过程，这意味着在比赛的任何时刻进球的可能性都是相等的。这并不完全正确，但对于大多数情况来说，这可能是一个足够好的模型。

我认为将建模作为问题解决的一个明确部分是很重要的，因为它提醒我们要考虑建模误差（即由于简化和模型假设而产生的误差）。

这本书中的许多方法都是基于离散分布，这让一些人担心数值误差。但对于现实世界的问题，数值误差几乎总是比建模误差要小。

此外，离散方法通常可以更好地进行建模决策，我宁愿对一个好模型得到一个近似解决方案，也不愿对一个坏模型得到一个精确解决方案。

## 与代码一起工作

阅读这本书只能让你走得更远；要真正理解它，你必须与代码一起工作。这本书的原始形式是一系列 Jupyter 笔记本。在阅读每一章之后，我鼓励你运行笔记本并完成练习。

如果你需要帮助，我的解决方案是可用的。

有几种方法可以运行这些笔记本：

+   如果你安装了 Python 和 Jupyter，你可以下载笔记本并在你的计算机上运行它们。

+   如果你没有可以运行 Jupyter 笔记本的编程环境，你可以使用 Colab，在浏览器中运行 Jupyter 笔记本而无需安装任何东西。

要在 Colab 上运行笔记本，请从[此页面](http://allendowney.github.io/ThinkBayes2/index.html)开始，该页面包含所有笔记本的链接。

如果您已经安装了 Python 和 Jupyter，您可以[将笔记本作为 Zip 文件下载](https://github.com/AllenDowney/ThinkBayes2/raw/master/ThinkBayes2Notebooks.zip)。

## 安装 Jupyter

如果您尚未安装 Python 和 Jupyter，我建议您安装 Anaconda，这是一个免费的 Python 发行版，包含了您所需的所有包。我发现 Anaconda 很容易安装。默认情况下，它会在您的主目录中安装文件，因此您不需要管理员权限。您可以从[此网站](https://www.anaconda.com/products/individual)下载 Anaconda。

Anaconda 包括大部分您在本书中运行代码所需的包。但是还有一些额外的包需要安装。

为了确保您拥有所需的一切（以及正确的版本），最好的选择是创建一个 Conda 环境。[下载此 Conda 环境文件](https://github.com/AllenDowney/ThinkBayes2/raw/master/environment.yml)，并运行以下命令来创建和激活一个名为`ThinkBayes2`的环境。

```py
conda env create -f environment.yml
conda activate ThinkBayes2 
```

如果您不想专门为这本书创建一个环境，您可以使用 Conda 安装所需的内容。以下命令应该可以获取您所需的一切：

```py
conda install python jupyter pandas scipy matplotlib
pip install empiricaldist 
```

如果您不想使用 Anaconda，您将需要以下包：

+   用于运行笔记本的 Jupyter，[`jupyter.org/`](https://jupyter.org/)；

+   用于基本数值计算的 NumPy，[`www.numpy.org/`](https://www.numpy.org/)；

+   用于科学计算的 SciPy，[`www.scipy.org/`](https://www.scipy.org/)；

+   用于处理数据的 Pandas，[`pandas.pydata.org/`](https://pandas.pydata.org/)；

+   用于可视化的 matplotlib，[`matplotlib.org/`](https://matplotlib.org/)；

+   用于表示分布的 empiricaldist，[`pypi.org/project/empiricaldist/`](https://pypi.org/project/empiricaldist/)。

尽管这些是常用的包，但并非所有 Python 安装都包含它们，并且在某些环境中安装它们可能很困难。如果您在安装时遇到问题，我建议使用 Anaconda 或其他包含这些包的 Python 发行版。

## 贡献者名单

如果您有建议或更正，请发送电子邮件至*downey@allendowney.com*。如果我根据您的反馈进行更改，我将把您加入贡献者名单（除非您要求省略）。

如果您至少包括错误出现的部分句子，那对我来说很容易搜索。页码和章节号也可以，但不太容易处理。谢谢！

+   首先，我必须感谢 David MacKay 的优秀著作《信息论、推断和学习算法》，这是我第一次了解贝叶斯方法的地方。在他的许可下，我使用了他书中的一些问题作为例子。

+   第二版中的一些示例和练习是从 Cameron Davidson-Pilon 那里借用的，并得到了许可。

+   这本书还受益于我与 Sanjoy Mahajan 的互动，特别是在 2012 年秋季，当时我在 Olin College 参加了他的贝叶斯推断课程。

+   本书中的许多示例是与我在 Olin College 的贝叶斯统计课程中的学生合作开发的。特别是，Red Line 示例最初是由 Brendan Ritter 和 Kai Austin 作为一个课程项目开始的。

+   我在波士顿 Python 用户组的项目夜间写了这本书的部分内容，所以我要感谢他们的陪伴和披萨。

+   Jasmine Kwityn 和 Dan Fauxsmith 在 O’Reilly Media 审校了第一版，并发现了许多改进的机会。

+   Linda Pescatore 发现了一个错别字并提出了一些建议。

+   Tomasz Miasko 提出了许多出色的更正和建议。

+   对于第二版，我要感谢 O’Reilly Media 的 Michele Cronin 和 Kristen Brown 以及技术审阅员 Ravin Kumar、Thomas Nield、Josh Starmer 和 Junpeng Lao。

+   本书基于的软件库的开发者和贡献者表示感激，特别是 Jupyter、NumPy、SciPy、Pandas、PyMC、ArviZ 和 Matplotlib。

其他发现拼写错误和错误的人包括 Greg Marra、Matt Aasted、Marcus Ogren、Tom Pollard、Paul A. Giannaros、Jonathan Edwards、George Purkins、Robert Marcus、Ram Limbu、James Lawry、Ben Kahle、Jeffrey Law、Alvaro Sanchez、Olivier Yiptong、Yuriy Pasichnyk、Kristopher Overholt、Max Hailperin、Markus Dobler、Brad Minch、Allen Minch、Nathan Yee、Michael Mera、Chris Krenn、Daniel Vianna。
