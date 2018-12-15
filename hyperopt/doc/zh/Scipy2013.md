# [Scipy2013](https://github.com/hyperopt/hyperopt/wiki/Scipy2013)

*[Font Tian](http://blog.csdn.net/fontthrone) translated this article on 23 December 2017*

[SciPy2013摘要提交](SciPy2013论文摘要)

## 标题
Hyperopt：用于优化机器学习算法的超参数的Python库

## 作者
詹姆斯·伯格斯特拉，丹·维明斯和戴维·C·考克斯

*(James Bergstra, Dan Yamins, and David D. Cox)*

## 简介
James Bergstra是滑铁卢大学理论神经科学中心的NSERC万津研究员。他的研究兴趣包括视觉系统模型和学习算法，深度学习，贝叶斯优化，高性能计算和音乐信息检索。此前他曾是哈佛大学罗兰科学研究院David Cox教授的计算机和生物视觉实验室的成员。他在2011年7月在Yoshua Bengio教授的指导下完成了蒙特利尔大学的博士研究，并撰写了关于如何将复杂的细胞纳入深度学习模型的论文。作为他的博士研究工作的一部分，他共同开发了Theano，一个流行的Python元编程系统，可以针对GPU进行高性能计算。

Dan Yamins是麻省理工学院脑与认知科学的博士后研究员。他的研究兴趣包括腹视觉流的计算模型，以及神经科学和计算机视觉应用的高性能计算。此前，他开发了用于大规模数据分析和工作流管理的Python语言软件工具。他在Radhika Nagpal的指导下在哈佛大学完成了博士学位，并撰写了关于空间分布式多代理系统计算模型的论文。

David Cox是分子和细胞生物学和计算机科学的助理教授，也是哈佛大学脑科学中心的成员。他完成了博士学位。在麻省理工学院脑与认知科学系，专攻计算神经科学。在加入MCB / CBS之前，他是哈佛罗兰研究所的一名初级研究员，这是一个多学科研究机构，致力于在传统领域的边界进行高风险，高回报的科学研究。

## 谈话摘要
大多数机器学习算法具有超参数，这些参数对端到端系统性能有很大影响，调整超参数以优化端到端性能可能是一项艰巨的任务。超参数有多种类型 - 连续值有边界和无边界值，有离散值的有序或不有序，有条件的甚至不总是适用（例如，可选的预处理阶段的参数） - 所以常规的连续和组合优化算法要么不直接应用，要么在没有利用搜索空间中的结构的情况下操作。通常，超参数的优化是由领域专家在不相关的问题之前进行的，或者在网格搜索的帮助下手动地处理问题。然而，

 对于更好的超参数优化算法（HOAs）有强烈的需求，原因有两个：

 1. HOAs将模型评估的实践正式化，从而可以在以后的日期和不同的人群中复制基准实验。

 2. 学习算法设计人员可以向非专家（如深度学习系统）提供灵活的完全可配置的实现，只要他们也提供相应的HOA。

Hyperopt通过Python库提供串行和并行化的HOA [2,3]。其设计的基础是（a）超参数搜索空间的描述，（b）超参数评估函数（机器学习系统）和（c）超参数搜索算法之间的通信协议。该协议使得可以使通用HOAs（例如捆绑的“TPE”算法）适用于一系列特定的搜索问题。具体的机器学习算法（或算法族）被实现为hyperopt 搜索空间在相关的项目中：Deep Belief Networks [4]，卷积视觉体系结构[5]和scikit-learn分类器[6]。我的演示文稿将解释hyperopt解决什么问题，如何使用它，以及如何从数据中提供准确的模型，而无需操作员干预。

## Submission References
[1] J. Bergstra and Y. Bengio (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research 13:281–305. [http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)

[2] J. Bergstra, D. Yamins and D. D. Cox (2013). Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. Proc. 30th International Conference on Machine Learning (ICML-13). [http://jmlr.csail.mit.edu/proceedings/papers/v28/bergstra13.pdf](http://jmlr.csail.mit.edu/proceedings/papers/v28/bergstra13.pdf)

[3] Hyperopt: [http://jaberg.github.com/hyperopt](http://jaberg.github.com/hyperopt)

[4] ... for Deep Belief Networks: [https://github.com/jaberg/hyperopt-dbn](https://github.com/jaberg/hyperopt-dbn)

[5] ... for convolutional vision architectures:[ https://github.com/jaberg/hyperopt-convnet](https://github.com/jaberg/hyperopt-convnet)

[6] ... for scikit-learn classifiers: [https://github.com/jaberg/hyperopt-sklearn](https://github.com/jaberg/hyperopt-sklearn)

More information about the presenting author can be found on his academic website:[ http://www.eng.uwaterloo.ca/~jbergstr/](http://www.eng.uwaterloo.ca/~jbergstr/)