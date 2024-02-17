# 第零章 序言

> 原文：[`greenteapress.com/thinkstats2/html/thinkstats2001.html`](https://greenteapress.com/thinkstats2/html/thinkstats2001.html)

《统计思维》是对探索性数据分析实用工具的介绍。本书的组织遵循我在开始处理数据集时使用的流程：

+   导入和清理：无论数据处于什么格式，通常需要一些时间和精力来读取数据，清理和转换数据，并检查所有内容是否完整地通过了翻译过程。

+   单变量探索：我通常从逐个检查变量开始，了解变量的含义，查看值的分布，并选择适当的摘要统计信息。

+   成对探索：为了确定变量之间可能的关系，我查看表格和散点图，并计算相关性和线性拟合。

+   多变量分析：如果变量之间存在明显的关系，我会使用多元回归添加控制变量，并研究更复杂的关系。

+   估计和假设检验：在报告统计结果时，回答三个问题很重要：效应有多大？如果我们再次进行相同的测量，我们应该期望多大的变异性？明显的效应可能是由偶然性引起的吗？

+   可视化：在探索过程中，可视化是发现可能的关系和影响的重要工具。如果一个明显的效应经得起审查，可视化是传达结果的有效方式。

本书采用了计算方法，相对于数学方法有几个优点：

+   我主要使用 Python 代码来提出大部分想法，而不是数学符号。一般来说，Python 代码更易读；而且，因为它是可执行的，读者可以下载、运行和修改它。

+   每一章都包括读者可以进行的练习，以发展和巩固他们的学习。当您编写程序时，您用代码表达您的理解；当您调试程序时，您也在纠正您的理解。

+   一些练习涉及实验来测试统计行为。例如，您可以通过生成随机样本并计算它们的总和来探索中心极限定理（CLT）。由此产生的可视化演示了 CLT 的工作原理以及它何时不起作用。

+   一些在数学上难以理解的想法通过模拟很容易理解。例如，我们通过运行随机模拟来近似 p 值，这加强了 p 值的含义。

+   由于本书基于通用编程语言（Python），读者可以从几乎任何来源导入数据。他们不仅限于已经为特定统计工具清理和格式化的数据集。

我写这本书的时候假设读者熟悉核心 Python，包括面向对象的特性，但不熟悉 pandas、NumPy 和 SciPy。

我假设读者了解基本的数学知识，包括对数和求和。我在一些地方提到了微积分概念，但您不必进行任何微积分。

如果您从未学过统计学，我认为这本书是一个很好的起点。如果您已经上过传统的统计课程，我希望这本书能帮助修复损害。

为了演示我的统计分析方法，本书提供了一个贯穿所有章节的案例研究。它使用了两个来源的数据：

+   美国疾病控制和预防中心（CDC）进行的《全国家庭增长调查》（NSFG）旨在收集有关“家庭生活、婚姻和离婚、怀孕、不孕不育、避孕使用以及男性和女性健康”的信息。（参见[`cdc.gov/nchs/nsfg.htm`](http://cdc.gov/nchs/nsfg.htm)。）

+   由美国疾病控制和预防中心进行的行为风险因素监测系统（BRFSS）旨在“跟踪美国的健康状况和风险行为”。（见[`cdc.gov/BRFSS/`](http://cdc.gov/BRFSS/)。）

其他例子使用了来自美国国税局、美国人口普查局和波士顿马拉松的数据。

这本《统计思维》的第二版包括了第一版的章节，其中许多章节经过了大幅修订，并新增了关于回归、时间序列分析、生存分析和分析方法的新章节。上一版没有使用 pandas、SciPy 或 StatsModels，所以所有这些内容都是新的。

## 0.1  我是如何写这本书的

当人们写新教科书时，他们通常会从一堆旧教科书中开始阅读。因此，大多数书籍包含的材料几乎是按照相同的顺序排列的。

我没有这样做。事实上，在我写这本书的时候，我几乎没有使用任何印刷材料，原因有几个：

+   我的目标是探索这个材料的新方法，所以我不想接触太多现有的方法。

+   由于我将这本书以自由许可证的形式提供，我希望确保它的任何部分都不受版权限制。

+   我的读者中有许多人无法接触到印刷材料的图书馆，所以我试图引用一些在互联网上免费提供的资源。

+   一些老媒体的支持者认为，专门使用电子资源是懒惰和不可靠的。他们可能在第一部分是对的，但我认为他们在第二部分是错误的，所以我想测试一下我的理论。

我使用的资源比其他任何资源都要多的是维基百科。总的来说，我阅读的统计主题的文章都非常好（尽管我在途中做了一些小的修改）。我在整本书中都包含了对维基百科页面的引用，并鼓励你去跟踪这些链接；在许多情况下，维基百科页面会继续我描述的内容。本书中的词汇和符号通常与维基百科保持一致，除非我有充分的理由偏离。我发现有用的其他资源是 Wolfram MathWorld 和 Reddit 统计论坛，[`www.reddit.com/r/statistics`](http://www.reddit.com/r/statistics)。

## 0.2  使用代码

这本书中使用的代码和数据可以从[`allendowney.github.io/ThinkStats2/`](http://allendowney.github.io/ThinkStats2/)获取。

使用这些代码的最简单方法是在 Colab 上运行它，这是一个免费的服务，在 Web 浏览器中运行 Jupyter 笔记本。对于每一章，我都提供了两个笔记本：一个包含了章节和练习中的代码；另一个还包含了解决方案。

如果你想在自己的电脑上运行这些笔记本，你可以从 GitHub 上单独下载它们，或者从[`github.com/AllenDowney/ThinkStats2/archive/refs/heads/master.zip`](https://github.com/AllenDowney/ThinkStats2/archive/refs/heads/master.zip)下载整个存储库。

我使用了 Continuum Analytics 的 Anaconda 来编写这本书，这是一个免费的 Python 发行版，包括了你运行代码所需的所有软件包（以及更多）。我发现 Anaconda 很容易安装。默认情况下，它进行用户级安装，所以你不需要管理权限。你可以从[`continuum.io/downloads`](http://continuum.io/downloads)下载 Anaconda。

如果你不想使用 Anaconda，你将需要以下软件包：

+   用于表示和分析数据的 pandas，[`pandas.pydata.org/`](http://pandas.pydata.org/)；

+   用于基本数值计算的 NumPy，[`www.numpy.org/`](http://www.numpy.org/)；

+   科学计算包 SciPy，包括统计学，[`www.scipy.org/`](http://www.scipy.org/)；

+   回归和其他统计分析的 StatsModels，[`statsmodels.sourceforge.net/`](http://statsmodels.sourceforge.net/)；

+   可视化工具 matplotlib，[`matplotlib.org/`](http://matplotlib.org/)。

尽管这些是常用的软件包，但并非所有 Python 安装都包含它们，并且在某些环境中安装它们可能很困难。如果您在安装时遇到问题，我强烈建议使用 Anaconda 或其他包含这些软件包的 Python 发行版。

—

Allen B. Downey 是马萨诸塞州尼达姆富兰克林 W.奥林工程学院的计算机科学教授。

## 贡献者名单

如果您有建议或更正，请发送电子邮件至`downey@allendowney.com`。如果我根据您的反馈进行更改，我将把您加入贡献者名单（除非您要求省略）。

如果您至少包含错误所在句子的一部分，这样对我来说很容易搜索。页码和章节号也可以，但不太容易处理。谢谢！

+   Lisa Downey 和 June Downey 阅读了初稿，并提出了许多更正和建议。

+   Steven Zhang 发现了几个错误。

+   Andy Pethan 和 Molly Farison 帮助调试了一些解决方案，Molly 发现了几个错别字。

+   Dr. Nikolas Akerblom 知道 Hyracotherium 有多大。

+   Alex Morrow 澄清了一个代码示例。

+   Jonathan Street 在最后一刻发现了一个错误。

+   非常感谢 Kevin Smith 和 Tim Arnold 在 plasTeX 上的工作，我用它将这本书转换为 DocBook。

+   George Caplan 发送了几个改进清晰度的建议。

+   Julian Ceipek 发现了一个错误和一些错别字。

+   Stijn Debrouwere，Leo Marihart III，Jonathan Hammler 和 Kent Johnson 在第一版印刷中发现了错误。

+   Jörg Beyer 在书中发现了错别字，并在附带代码的 docstrings 中进行了许多更正。

+   Tommie Gannert 发送了一个包含许多更正的补丁文件。

+   Christoph Lendenmann 提交了几个勘误。

+   Michael Kearney 给我寄来了许多出色的建议。

+   Alex Birch 提出了许多有用的建议。

+   Lindsey Vanderlyn，Griffin Tschurwald 和 Ben Small 阅读了这本书的早期版本，并发现了许多错误。

+   John Roth，Carol Willing 和 Carol Novitsky 对这本书进行了技术审查。他们发现了许多错误，并提出了许多有用的建议。

+   David Palmer 发送了许多有用的建议和更正。

+   Erik Kulyk 发现了许多错别字。

+   Nir Soffer 为这本书和支持代码发送了几个出色的拉取请求。

+   GitHub 用户 flothesof 发送了许多更正。

+   正在进行这本书的日语翻译的 Toshiaki Kurokawa 发送了许多更正和有用的建议。

+   Benjamin White 建议使用更通俗的 Pandas 代码。

+   Takashi Sato 发现了一个代码错误。

+   Fardin Afdideh 为这本书和 Think X 系列中的其他书籍发送了几页更正和建议。

其他发现错别字和类似错误的人包括 Andrew Heine，Gábor Lipták，Dan Kearney，Alexander Gryzlov，Martin Veillette，Haitao Ma，Jeff Pickhardt，Rohit Deshpande，Joanne Pratt，Lucian Ursu，Paul Glezen，Ting-kuang Lin，Scott Miller，Luigi Patruno。
