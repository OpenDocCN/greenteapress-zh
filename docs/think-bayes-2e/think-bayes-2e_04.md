# 概率

> 原文：[`allendowney.github.io/ThinkBayes2/chap01.html`](https://allendowney.github.io/ThinkBayes2/chap01.html)

贝叶斯统计的基础是贝叶斯定理，而贝叶斯定理的基础是条件概率。

在本章中，我们将从条件概率开始，推导贝叶斯定理，并使用真实数据集进行演示。在下一章中，我们将使用贝叶斯定理来解决与条件概率相关的问题。在接下来的章节中，我们将从贝叶斯定理过渡到贝叶斯统计，并解释其中的区别。

## 银行家琳达

为了介绍条件概率，我将使用特韦斯基和卡尼曼的[著名实验中的一个例子](https://en.wikipedia.org/wiki/Conjunction_fallacy)，他们提出了以下问题：

> 琳达今年 31 岁，单身，直言不讳，非常聪明。她主修哲学。作为学生，她对歧视和社会正义问题非常关注，并参加了反核示威活动。哪个更有可能？
> 
> 1.  琳达是一名银行出纳。
> 1.  
> 1.  琳达是一名银行出纳，并积极参与女权主义运动。

许多人选择第二个答案，可能是因为它似乎更符合描述。如果琳达*只是*一个银行出纳，那似乎是不太一致的；如果她还是一个女权主义者，那似乎更一致。

但第二个答案不能是“更有可能”，正如问题所问的那样。假设我们找到了符合琳达描述的 1000 个人，其中有 10 个人是银行出纳。他们中有多少人也是女权主义者？最多，他们中的所有 10 个人都是；在这种情况下，这两个选项是*同样*可能的。如果少于 10 个人，第二个选项就*不太*可能。但第二个选项绝对不可能*更*可能。

如果你倾向于选择第二个选项，那么你是和一些人一样的。生物学家[史蒂芬·古尔德写道](https://doi.org/10.1080/09332480.1989.10554932)：

> 我特别喜欢这个例子，因为我知道[第二个]陈述最不可能，但是我脑海中的小人仍然在跳来跳去，对我喊道，“但她不可能只是一个银行出纳；读一下描述。”

如果你脑海中的小人仍然不开心，也许这一章会有所帮助。

## 概率

在这一点上，我应该提供一个“概率”的定义，但是这[事实上是令人惊讶地困难](https://en.wikipedia.org/wiki/Probability_interpretations)。为了避免在开始之前陷入困境，我们现在将使用一个简单的定义，并稍后加以完善：**概率**是有限集合的一个分数。

例如，如果我们对 1000 人进行调查，其中有 20 人是银行出纳，那么作为银行出纳的比例就是 0.02 或 2%。如果我们随机选择这个人口中的一个人，他们是银行出纳的概率就是 2%。我所说的“随机”是指数据集中的每个人被选择的机会是相同的。

有了这个定义和一个合适的数据集，我们可以通过计数来计算概率。为了演示，我将使用[General Social Survey](http://gss.norc.org/) (GSS)的数据。

我将使用 Pandas 来读取数据并将其存储在`DataFrame`中。

```py
import pandas as pd

gss = pd.read_csv('gss_bayes.csv')
gss.head() 
```

|  | caseid | year | age | sex | polviews | partyid | indus10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1974 | 21.0 | 1 | 4.0 | 2.0 | 4970.0 |
| 1 | 2 | 1974 | 41.0 | 1 | 5.0 | 0.0 | 9160.0 |
| 2 | 5 | 1974 | 58.0 | 2 | 6.0 | 1.0 | 2670.0 |
| 3 | 6 | 1974 | 30.0 | 1 | 5.0 | 4.0 | 6870.0 |
| 4 | 7 | 1974 | 48.0 | 1 | 5.0 | 4.0 | 7860.0 |

`DataFrame`中的每一行代表接受调查的每个人，每一列代表我选择的每个变量。

这些列是

+   `caseid`：受访者标识符。

+   `year`：受访者接受调查时的年份。

+   `age`：受访者接受调查时的年龄。

+   `sex`：男性或女性。

+   `polviews`：从自由主义到保守主义的政治观点。

+   `partyid`：政党隶属，民主党、独立党或共和党。

+   `indus10`：[代码](https://www.census.gov/cgi-bin/sssd/naics/naicsrch?chart=2007) 表示受访者所在行业。

让我们更详细地看看这些变量，从`indus10`开始。

## 银行家的比例

“银行业及相关活动”的代码是 6870，所以我们可以这样选择银行家：

```py
banker = (gss['indus10'] == 6870)
banker.head() 
```

```py
0    False
1    False
2    False
3     True
4    False
Name: indus10, dtype: bool 
```

结果是一个包含布尔值`True`和`False`的 Pandas `Series`。

如果我们在这个`Series`上使用`sum`函数，它会将`True`视为 1，`False`视为 0，因此总数就是银行家的数量。

```py
banker.sum() 
```

```py
728 
```

在这个数据集中，有 728 名银行家。

要计算银行家的*比例*，我们可以使用`mean`函数，它计算`Series`中`True`值的比例：

```py
banker.mean() 
```

```py
0.014769730168391155 
```

大约 1.5%的受访者在银行业工作，所以如果我们从数据集中随机选择一个人，他们是银行家的概率约为 1.5%。

## 概率函数

我将把上一节的代码放入一个函数中，该函数接受一个布尔系列并返回一个概率：

```py
def prob(A):
  """Computes the probability of a proposition, A."""    
    return A.mean() 
```

所以我们可以这样计算银行家的比例：

```py
prob(banker) 
```

```py
0.014769730168391155 
```

现在让我们看看这个数据集中的另一个变量。列`sex`的值是这样编码的：

```py
1    Male
2    Female 
```

因此，我们可以制作一个布尔系列，对女性受访者为`True`，否则为`False`。

```py
female = (gss['sex'] == 2) 
```

并使用它来计算女性受访者的比例。

```py
prob(female) 
```

```py
0.5378575776019476 
```

这个数据集中女性的比例比美国成年人口要高，因为 [GSS 不包括生活在机构中的人](https://gss.norc.org/faq) ，比如监狱和军事住房，这些人口更可能是男性。

## 政治观点和政党

我们将考虑的其他变量是`polviews`，描述受访者的政治观点，以及`partyid`，描述他们与政党的关联。

`polviews` 的值是一个七点量表：

```py
1	Extremely liberal
2	Liberal
3	Slightly liberal
4	Moderate
5	Slightly conservative
6	Conservative
7	Extremely conservative 
```

我将定义`liberal`为任何选择“极端自由派”，“自由派”或“稍微自由派”的人为`True`。

```py
liberal = (gss['polviews'] <= 3) 
```

这是按照这个定义自由派的受访者的比例。

```py
prob(liberal) 
```

```py
0.27374721038750255 
```

如果我们在这个数据集中选择一个随机人，他们是自由的概率约为 27%。

`partyid`的值是这样编码的：

```py
0	Strong democrat
1	Not strong democrat
2	Independent, near democrat
3	Independent
4	Independent, near republican
5	Not strong republican
6	Strong republican
7	Other party 
```

我将定义`democrat`，包括选择“强烈民主党”或“不强烈民主党”的受访者：

```py
democrat = (gss['partyid'] <= 1) 
```

这是按照这个定义民主党人的受访者的比例。

```py
prob(democrat) 
```

```py
0.3662609048488537 
```

## 连接

现在我们有了概率的定义和计算它的函数，让我们继续进行连接。

“连接”是逻辑`and`操作的另一个名称。如果你有两个[命题](https://en.wikipedia.org/wiki/Proposition)，`A`和`B`，连接`A 和 B`如果`A`和`B`都为`True`，则为`True`，否则为`False`。

如果我们有两个布尔系列，我们可以使用`&`运算符来计算它们的连接。例如，我们已经计算了受访者是银行家的概率。

```py
prob(banker) 
```

```py
0.014769730168391155 
```

以及他们是民主党人的概率：

```py
prob(democrat) 
```

```py
0.3662609048488537 
```

现在我们可以计算受访者是银行家*和*民主党人的概率：

```py
prob(banker & democrat) 
```

```py
0.004686548995739501 
```

正如我们所期望的，`prob(banker & democrat)`小于`prob(banker)`，因为并非所有的银行家都是民主党人。

我们期望连接是可交换的；也就是说，`A & B` 应该与 `B & A` 相同。为了检查，我们还可以计算 `prob(democrat & banker)`：

```py
prob(democrat & banker) 
```

```py
0.004686548995739501 
```

正如预期的那样，它们是相同的。

## 条件概率

条件概率是一个依赖于条件的概率，但这可能不是最有帮助的定义。以下是一些例子：

+   如果一个受访者是自由派，那么他是民主党人的概率是多少？

+   如果一个受访者是银行家，那么她是女性的概率是多少？

+   如果一个受访者是女性，那么她是自由派的概率是多少？

让我们从第一个开始，我们可以这样解释：“在所有自由派的受访者中，有多少是民主党人？”

我们可以分两步计算这个概率：

1.  选择所有自由派的受访者。

1.  计算所选受访者中是民主党人的比例。

要选择自由派受访者，我们可以使用方括号运算符`[]`，像这样：

```py
selected = democrat[liberal] 
```

`selected`包含自由派受访者的`democrat`的值，所以`prob(selected)`是自由派中是民主党人的比例：

```py
prob(selected) 
```

```py
0.5206403320240125 
```

超过一半的自由派是民主党人。如果这个结果比你预期的要低，记住：

1.  我们使用了一个相对严格的“民主党人”的定义，排除了“倾向”民主党的独立人士。

1.  数据集包括自 1974 年以来的受访者；在这个时间段的早期，与现在相比，政治观点和党派隶属之间的一致性较少。

让我们尝试第二个例子，“一个受访者是女性的概率是多少，假设他们是银行家？”我们可以解释为，“在所有是银行家的受访者中，有多少是女性？”

同样，我们将使用方括号运算符来选择只有银行家的受访者，然后使用`prob`来计算其中女性的比例。

```py
selected = female[banker]
prob(selected) 
```

```py
0.7706043956043956 
```

这个数据集中大约 77%的银行家是女性。

让我们把这个计算包装成一个函数。我将定义`conditional`来接受两个布尔系列，`proposition`和`given`，并计算在`given`条件下`proposition`的条件概率：

```py
def conditional(proposition, given):
  """Probability of A conditioned on given."""
    return prob(proposition[given]) 
```

我们可以使用`conditional`来计算一个受访者是自由派的概率，假设他们是女性。

```py
conditional(liberal, given=female) 
```

```py
0.27581004111500884 
```

大约 28%的女性受访者是自由派。

我在这个表达式中包含了关键词“给定”，以及参数“女性”，使其更易读。

## 条件概率不是可交换的

我们已经看到连接是可交换的；也就是说，`prob(A & B)`总是等于`prob(B & A)`。

但条件概率*不*是可交换的；也就是说，`conditional(A, B)`和`conditional(B, A)`不是一样的。

如果我们看一个例子，这一点应该很清楚。之前，我们计算了一个受访者是女性的概率，假设他们是银行家。

```py
conditional(female, given=banker) 
```

```py
0.7706043956043956 
```

结果显示大多数银行家是女性。这与一个受访者是银行家的概率，假设他们是女性的概率不同：

```py
conditional(banker, given=female) 
```

```py
0.02116102749801969 
```

只有大约 2%的女性受访者是银行家。

我希望这个例子能清楚地表明条件概率不是可交换的，也许对你来说已经很清楚了。然而，混淆`conditional(A, B)`和`conditional(B, A)`是一个常见的错误。我们稍后会看到一些例子。

## 条件和连接

我们可以结合条件概率和连接。例如，这是一个受访者是女性的概率，假设他们是自由派民主党人。

```py
conditional(female, given=liberal & democrat) 
```

```py
0.576085409252669 
```

大约 57%的自由派民主党人是女性。

这是他们是自由女性的概率，假设他们是银行家：

```py
conditional(liberal & female, given=banker) 
```

```py
0.17307692307692307 
```

大约 17%的银行家是自由女性。

## 概率定律

在接下来的几节中，我们将推导连接和条件概率之间的三个关系：

+   定理 1：使用连接来计算条件概率。

+   定理 2：使用条件概率来计算连接。

+   定理 3：使用`conditional(A, B)`来计算`conditional(B, A)`。

定理 3 也被称为贝叶斯定理。

我将使用数学符号来写这些定理：

+   $P(A)$是命题$A$的概率。

+   $P(A~\mathrm{and}~B)$是$A$和$B$的连接的概率，也就是说，两者都为真的概率。

+   $P(A | B)$是在$B$为真的条件下$A$的条件概率。$A$和$B$之间的竖线读作“给定”。

有了这个，我们准备好定理 1 了。

### 定理 1

银行家中女性的比例是多少？我们已经看到了计算答案的一种方法：

1.  使用方括号运算符来选择银行家，然后

1.  使用`mean`来计算女性银行家的比例。

我们可以把这些步骤写成这样：

```py
female[banker].mean() 
```

```py
0.7706043956043956 
```

或者我们可以使用`conditional`函数，它做的是同样的事情：

```py
conditional(female, given=banker) 
```

```py
0.7706043956043956 
```

但还有另一种方法来计算这个条件概率，即计算两个概率的比值：

1.  受访者中女性银行家的比例，

1.  受访者中是银行家的比例。

换句话说：在所有的银行家中，有多少比例是女性银行家？我们是这样计算这个比例的。

```py
prob(female & banker) / prob(banker) 
```

```py
0.7706043956043956 
```

结果是相同的。这个例子演示了一个关于条件概率和连接的一般规则。在数学符号中，它是这样的：

$$P(A|B) = \frac{P(A~\mathrm{and}~B)}{P(B)}$$

这就是定理 1。

### 定理 2

如果我们从定理 1 开始，并将两边都乘以$P(B)$，我们得到定理 2。

$$P(A~\mathrm{and}~B) = P(B) ~ P(A|B)$$

这个公式提出了计算连接的第二种方法：不使用`&`运算符，我们可以计算两个概率的乘积。

让我们看看它是否适用于“自由派”和“民主党人”。这里是使用`&`的结果：

```py
prob(liberal & democrat) 
```

```py
0.1425238385067965 
```

这里是使用定理 2 的结果：

```py
prob(democrat) * conditional(liberal, democrat) 
```

```py
0.1425238385067965 
```

它们是相同的。

### 定理 3

我们已经证明了连接是可交换的。在数学符号中，这意味着：

$$P(A~\mathrm{and}~B) = P(B~\mathrm{and}~A)$$

如果我们将定理 2 应用于两边，我们有

$$P(B) P(A|B) = P(A) P(B|A)$$

这里有一种解释方法：如果你想检查$A$和$B$，你可以以任何顺序进行：

1.  你可以先检查$B$，然后在$B$的条件下检查$A$，或者

1.  你可以先检查$A$，然后在$A$的条件下检查$B$。

如果我们除以$P(B)$，我们得到定理 3：

$$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$$

这，朋友们，就是贝叶斯定理。

为了看看它是如何工作的，让我们首先使用“条件”计算银行家中自由派的比例：

```py
conditional(liberal, given=banker) 
```

```py
0.2239010989010989 
```

现在使用贝叶斯定理：

```py
prob(liberal) * conditional(banker, liberal) / prob(banker) 
```

```py
0.2239010989010989 
```

它们是相同的。

## 总概率定律

除了这三个定理，我们还需要贝叶斯统计的另一件事：总概率定律。这是总概率定律的一种形式，用数学符号表示：

$$P(A) = P(B_1 \mathrm{and} A) + P(B_2 \mathrm{and} A)$$

换句话说，$A$的总概率是两种可能性的总和：要么$B_1$和$A$为真，要么$B_2$和$A$为真。但是只有在$B_1$和$B_2$是：

+   相互排斥，这意味着它们中只有一个可以为真，以及

+   完全穷尽，这意味着它们中的一个必须为真。

例如，让我们使用这个定律来计算受访者是银行家的概率。我们可以直接计算如下：

```py
prob(banker) 
```

```py
0.014769730168391155 
```

所以让我们确认一下，如果我们分别计算男性和女性银行家，我们会得到相同的结果。

在这个数据集中，所有受访者都被指定为男性或女性。最近，GSS 监事会宣布他们将在调查中添加更具包容性的性别问题（您可以在[`gender.stanford.edu/news-publications/gender-news/more-inclusive-gender-questions-added-general-social-survey`](https://gender.stanford.edu/news-publications/gender-news/more-inclusive-gender-questions-added-general-social-survey)了解更多关于这个问题及他们的决定）。

我们已经有一个布尔`Series`，对于女性受访者是`True`。这是男性受访者的补充`Series`。

```py
male = (gss['sex'] == 1) 
```

现在我们可以这样计算“银行家”的总概率。

```py
prob(male & banker) + prob(female & banker) 
```

```py
0.014769730168391155 
```

因为“男性”和“女性”是相互排斥且完全穷尽的（MECE），我们得到了直接计算“银行家”概率时得到的相同结果。

应用定理 2，我们也可以这样写总概率定律：

$$P(A) = P(B_1) P(A|B_1) + P(B_2) P(A|B_2)$$

我们可以用同样的例子来测试它：

```py
(prob(male) * conditional(banker, given=male) +
prob(female) * conditional(banker, given=female)) 
```

```py
0.014769730168391153 
```

当存在两个以上的条件时，将总概率定律写成求和形式更加简洁：

$$P(A) = \sum_i P(B_i) P(A|B_i)$$

只要条件$B_i$是相互排斥且完全穷尽的，这个结论就成立。例如，让我们考虑“政治观点”，它有七个不同的值。

```py
B = gss['polviews']
B.value_counts().sort_index() 
```

```py
polviews
1.0     1442
2.0     5808
3.0     6243
4.0    18943
5.0     7940
6.0     7319
7.0     1595
Name: count, dtype: int64 
```

在这个尺度上，`4.0`代表“中等”。所以我们可以这样计算中等银行家的概率：

```py
i = 4
prob(B==i) * conditional(banker, B==i) 
```

```py
0.005822682085615744 
```

我们可以使用`sum`和[生成器表达式](https://www.johndcook.com/blog/2020/01/15/generator-expression/)来计算总和。

```py
sum(prob(B==i) * conditional(banker, B==i)
    for i in range(1, 8)) 
```

```py
0.014769730168391157 
```

结果是相同的。

在这个例子中，使用全概率定理比直接计算概率要麻烦得多，但我保证它会很有用。

## 总结

到目前为止我们有：

**定理 1**给了我们一种使用连接来计算条件概率的方法：

$$P(A|B) = \frac{P(A~\mathrm{and}~B)}{P(B)}$$

**定理 2**给了我们一种使用条件概率来计算连接的方法：

$$P(A~\mathrm{and}~B) = P(B) P(A|B)$$

**定理 3**，也被称为贝叶斯定理，给了我们一种从$P(A|B)$到$P(B|A)$或者反过来的方法：

$$P(A|B) = \frac{P(A) P(B|A)}{P(B)}$$

**全概率定理**提供了一种通过加总各部分来计算概率的方法：

$$P(A) = \sum_i P(B_i) P(A|B_i)$$

此时你可能会问，“那又怎样呢？”如果我们有所有的数据，我们可以通过计数来计算任何我们想要的概率，任何连接或任何条件概率。我们不必使用这些公式。

你是对的，*如果*我们有所有的数据。但通常我们没有，这种情况下，这些公式就会非常有用——特别是贝叶斯定理。在下一章中，我们将看到。

## 练习

**练习：**使用`conditional`来计算以下概率：

+   一个受访者是自由主义者的概率是多少，假设他们是民主党人？

+   一个受访者是民主党人的概率是多少，假设他们是自由主义者？

仔细考虑你传递给`conditional`的参数的顺序。

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容隐藏代码单元格内容</summary>

```py
# Solution

conditional(liberal, given=democrat) 
```

```py
0.3891320002215698 
```</details> <details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容隐藏代码单元格内容</summary>

```py
# Solution

conditional(democrat, given=liberal) 
```

```py
0.5206403320240125 
```</details>

**练习：**让我们使用本章的工具来解决琳达问题的一个变种。

> 琳达今年 31 岁，单身，直言不讳，非常聪明。她主修哲学。作为学生，她对歧视和社会正义问题非常关注，并参加了反核示威。哪个更有可能？
> 
> 1.  琳达是一名银行家。
> 1.  
> 1.  琳达是一名银行家，并认为自己是自由主义民主党人。

为了回答这个问题，计算

+   Linda 是银行家的概率，假设她是女性，

+   Linda 是一名银行家和自由主义民主党人的概率，假设她是女性。

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容隐藏代码单元格内容</summary>

```py
# Solution

conditional(banker, given=female) 
```

```py
0.02116102749801969 
```</details> <details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容隐藏代码单元格内容</summary>

```py
# Solution

conditional(banker & liberal & democrat, given=female) 
```

```py
0.0023009316887329786 
```</details>

**练习：**有一句[著名的引语](https://quoteinvestigator.com/2014/02/24/heart-head/)关于年轻人、老年人、自由主义者和保守派，大致是这样的：

> 如果你在 25 岁不是自由主义者，你就没有心。如果你在 35 岁不是保守派，你就没有大脑。

无论你是否同意这个命题，它都暗示了一些我们可以计算的概率。与其使用具体的年龄 25 和 35，不如将“年轻”和“老年”定义为 30 岁以下或 65 岁以上：

```py
young = (gss['age'] < 30)
prob(young) 
```

```py
0.19435991073240008 
```

```py
old = (gss['age'] >= 65)
prob(old) 
```

```py
0.17328058429701765 
```

对于这些阈值，我选择了接近第 20 和第 80 百分位数的整数。根据你的年龄，你可能同意或不同意这些“年轻”和“老年”的定义。

我将“保守派”定义为政治观点为“保守派”、“稍微保守派”或“极端保守派”的人。

```py
conservative = (gss['polviews'] >= 5)
prob(conservative) 
```

```py
0.3419354838709677 
```

使用`prob`和`conditional`来计算以下概率。

+   随机选择的受访者是自由主义者的概率是多少？

+   一个年轻人是自由主义者的概率是多少？

+   受访者中有多少是老年保守派？

+   保守派中有多少是老年人？

对于每个陈述，思考它是在表达一个连接、一个条件概率，还是两者兼而有之。

对于条件概率，要注意参数的顺序。如果你对最后一个问题的答案大于 30%，那么你搞错了！

<details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容 隐藏代码单元格内容</summary>

```py
# Solution

prob(young & liberal) 
```

```py
0.06579427875836884 
```</details> <details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容 隐藏代码单元格内容</summary>

```py
# Solution

conditional(liberal, given=young) 
```

```py
0.338517745302714 
```</details> <details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容 隐藏代码单元格内容</summary>

```py
# Solution

prob(old & conservative) 
```

```py
0.06701156421180766 
```</details> <details class="hide above-input"><summary aria-label="Toggle hidden content">显示代码单元格内容 隐藏代码单元格内容</summary>

```py
# Solution

conditional(old, given=conservative) 
```

```py
0.19597721609113564 
```</details>