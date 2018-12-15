# [FMin](https://github.com/hyperopt/hyperopt/wiki/FMin)

*[Font Tian](http://blog.csdn.net/fontthrone) translated this article on 22 December 2017*

这一页是关于  `hyperopt.fmin()` 的基础教程. 主要写了如何写一个可以利用 `fmin` 进行优化的函数,以及如何描述 `fmin` 的搜索空间。

Hyperopt的工作是通过一组可能的参数找到标量值，possibly-stochastic function的最佳值（注意在数学中stochastic与random并不完全相同,译者注）。虽然许多优化包假设这些输入是从矢量空间中抽取的，但Hyperopt的不同之处在于它鼓励您更详细地描述搜索空间。通过提供有关定义函数的位置以及您所认为最佳值位置的更多信息，可以让hyperopt中的算法更有效地进行搜索。

使用hyperopt的方式的过程总结：

 - 用于最小化的目标函数
 - 搜索空间
 - 存储搜索的所有点评估的数据库
 - 要使用的搜索算法

这个（最基本的）教程将介绍如何编写函数和搜索空间，使用默认 `Trials` 数据库和伪 `random` 搜索算法。第（1）部分是关于目标函数与
 `hyperopt` 之间通信的不同调用约定。第（2）部分是关于搜索空间的描述。

用 `MongoTrials` 代替 `Trials` 以进行并行搜索；除此之外，还有一个有关如何使用[`MongDB` 进行并行所有的wiki页面](https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB)

我们可以通过选择 `algo=hyperopt.tpe.suggest` 代替 `algo=hyperopt.random.suggest` 来进行搜索算法的选择.实际上,搜索算法是一个可调用的对象,同时其构造函数也传入配置参数.不过,这几乎已经是有关选择搜索算法的所有有关技术了.

## 1.定义一个用来最小化的函数

当涉及到指定一个最小化的目标函数时， `Hyperopt` 提供了一些增加灵活性/复杂性的选项。作为一个目标函数设计者可能会考虑的问题:

 - 除了函数返回值之外，还需要保存其他信息，例如计算目标时收集的其他统计信息和诊断信息？
 - 你是否想要使用比函数值更多的优化算法？
 - 你想在并行进程之间进行通信吗？（例如任务，或最小化算法）

接下来的几节将介绍实现一个目标函数的各种方法，这个目标函数可以使单个变量的二次目标函数最小化。在每一节中，我们将搜索从-10到+10的有界范围，我们可以用一个搜索空间来描述：

```
	space = hp.uniform('x', -10, 10)
```

而之后的第二节则介绍如何指定更加复杂的搜索空间

### 1.1 最简单的情况
`hyperopt` 的优化算法与您的目标函数之间进行通信的最简单协议是您的目标函数从搜索空间接收到有效点，并返回与该点关联的浮点 损失（也称为负效用）。

```
	from hyperopt import fmin, tpe, hp
	best = fmin(fn=lambda x: x ** 2,
	    space=hp.uniform('x', -10, 10),
	    algo=tpe.suggest,
	    max_evals=100)
	print best
```

这个协议的优点是可读性高并且易于上手。正如你所看到的，这只有一行代码。这种协议的缺点是：（1）这种功能不能将关于每个评估的额外信息返回到试验数据库中;（2）这种功能不能与搜索算法或其他并发功能评估相互作用。你会在下面的例子中看到为什么你可能想要做这些事情。

### 1.2通过试用对象附加额外的信息

如果你的目标函数很复杂，需要很长时间才能运行，那么你很可能会想要保存更多的统计数据和诊断信息，而不仅仅是最后出现的浮点数损失。对于这种情况， 你可以将`fmin` 函数编写成处理字典返回值的形式。比如，让你的损失函数可以返回你想要的所有统计数据和诊断的嵌套字典。但实际情况要比这个灵活一点：当使用 `mongodb` 时，字典必须是有效的JSON文档。这对于存储特定领域的数据而言还是很灵活的。

当目标函数返回一个字典时， `fmin` 函数会在返回值中查找一些特殊的键值对，并将其传递给优化算法。有两个必须写的键值对：

 - `status`- 其中的一个键 `hyperopt.STATUS_STRINGS` ，例如成功完成时的“OK”，以及在功能变得未定义的情况时的“失败”。
 - `loss` - 如果你尝试最小化浮点值函数值，如果状态是“ok”，那么这个值必须存在。

fmin函数也有一些可选的键

 - `attachments` - 键值对的字典，其键是短字符串（如文件名），其值可能是长字符串（如文件内容），每次访问记录时都不应从数据库加载。（另外，MongoDB限制正常的键值对的长度，所以一旦你的值是`megabytes`，你可能必须把它作为附件。）
 - `loss_variance` - 浮点型数值 - 随机目标函数的不确定性（键的直译是 `损失-方差` ）。
 - `true_loss '- 浮点型数值 - 当进行超参数优化时，如果使用这个名称存储模型的泛化错误，那么有时可以从内置的绘图例程中获得更加精确的输出。
 - `true_loss_variance` - 浮点型数值 - 泛化误差的不确定性（键的直译时 `真实的-损失-方差` ）。

下面时原文(译者补充):

- attachments - a dictionary of key-value pairs whose keys are short strings (like filenames) and whose values are potentially long strings (like file contents) that should not be loaded from a database every time we access the record. (Also, MongoDB limits the length of normal key-value pairs so once your value is in the megabytes, you may have to make it an attachment.)
- loss_variance - float - the uncertainty in a stochastic objective function
- true_loss - float - When doing hyper-parameter optimization, if you store the generalization error of your model with this name, then you can sometimes get spiffier output from the built-in plotting routines.
- true_loss_variance - float - the uncertainty in the generalization error


由于字典是为了适应各种后端存储机制，你应该确保它是JSON兼容的。只要它是字典，列表，元组，数字，字符串和日期时间的树形结构图，就可以。

**提示**：要存储 `numpy` 数组，请将它们序列化为一个字符串，并考虑将它们存储为附件( `attachments` )。

在字典返回样式中编写上面的函数，它看起来像这样：

```
	import pickle
	import time
	from hyperopt import fmin, tpe, hp, STATUS_OK

	def objective(x):
	    return {'loss': x ** 2, 'status': STATUS_OK }

	best = fmin(objective,
	    space=hp.uniform('x', -10, 10),
	    algo=tpe.suggest,
	    max_evals=100)

	print best

```

### 1.3 Trials对象

为了能够真正看到我们需要返回的字典，让我们修改目标函数返回更多的东西，并传递一个明确的`trials`参数`fmin`。

``` 
	import pickle
	import time
	from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

	def objective(x):
	    return {
		'loss': x ** 2,
		'status': STATUS_OK,
		# -- store other results like this
		'eval_time': time.time(),
		'other_stuff': {'type': None, 'value': [0, 1, 2]},
		# -- attachments are handled differently
		'attachments':
		    {'time_module': pickle.dumps(time.time)}
		}
	trials = Trials()
	best = fmin(objective,
	    space=hp.uniform('x', -10, 10),
	    algo=tpe.suggest,
	    max_evals=100,
	    trials=trials)

	print best
```
在这种情况下，对`fmin` 的调用与以前一样进行，但是通过直接传递一个 `Trials` 对象，我们可以检查实验期间计算的所有返回值。

举个例子：

 - `trials.trials` - 一个代表所有搜索内容的字典列表(a list of dictionaries)
 - `trials.results` - 一个在搜索过程中由“objective”返回的字典列表(a list of dictionaries)
 - `trials.losses()` - 一个关于损失的List列表（一个“OK”Trial的浮动值）
 - `trials.statuses()` - 一个状态字符串的List列表

这个试验对象可以保存，传递到内置的绘图程序，或用自己的自定义代码进行分析。

该附件(attachments)是通过一种特殊的机制，使得它可以使用相同的代码都处理 `Trials` 和 `MongoTrials` 。

您可以像这样检索 `trial` 附件，下面的代码可以检索第5次`trial`的“time_module”附件：

```
	msg = trials.trial_attachments(trials.trials[5])['time_module']
	time_module = pickle.loads(msg)
```
语法有点复杂难懂，主要是因为一开始的设计思想是将附件当做大字符串。所以当我们使用 `MongoTrials` 时，我们并不想下载我们不需要的额外数据。而字符串则可以通过设置为全局（globally）来连接 `trials` 对象 `trials.attachments` ，这就像是一个字符串对字符串的字典（a string-to-string dictionary)。

**注意**目前， `Trials` 对象的 `trial-specific attachments` 被放入同样的 `global trials attachment dictionary` 中，但是将来可能会改变，从而导致对 `MongoTrials` 对象产生不正确的结果。

### 1.4用于实时与 `MongoDB` 通信的 `Ctrl` 对象

可以 `fmin()` 给你的目标函数一个并行实验使用的 `mongodb` 的句柄。这种机制使得用部分结果更新数据库成为可能，并且与其他正在评估不同点的并发进程进行通信。你的目标函数甚至可以添加新的搜索点，就像 `random.suggest` 。

基本的技术包括：

 - 使用 `fmin_pass_expr_memo_ctrl` 装饰器
 - 在你自己的方法中使用 `pyll.rec_eval` ，从 `expr` 和 `memo` 中构建搜索空间中的点。
 - 使用 `ctrl`， 一个与实况对象（ `trials` ）进行交流的  `hyperopt.Ctrl`  实例。

如果在你读完这个简短的教程之后,感觉没有太多的意义，这很正常，我想要告诉你们的是对于当前的代码库什么是可行的，并提供一些术语 从而使你能够在 `HyperOpt` 源文件，单元测试，和示例项目中进行有效检索，如 术语 `HyperOpt ConvNet` 。如果你想要一些帮助快速掌握这部分代码可以给我发电子邮件或者  file a github issue。
## 2.定义一个搜索空间

搜索空间由嵌套函数表达式组成，包括随机表达式。随机表达式是超参数。从这个嵌套的随机程序中抽样定义了随机搜索算法。超参数优化算法通过将正常的“采样”逻辑替换为自适应探索策略来工作，而不是试图从搜索空间中指定的分布进行采样。

最好把搜索空间看作是随机的抽样程序。例如

```
	from hyperopt import hp
	space = hp.choice('a',
	    [
		('case 1', 1 + hp.lognormal('c1', 0, 1)),
		('case 2', hp.uniform('c2', -10, 10))
	    ])
```

运行这段代码片段的结果是一个变量 `space` ，它引用了表达式标识符及其参数的图表。实际上它没有采样，它只是一个描述如何采样点的图表。处理这种表达式图的代码在 `hyperopt.pyll`中  ，我将这些图作为 `pyll` 图或 `pyll` 程序（ pyll graphs or pyll programs）。

如果你喜欢，你可以通过抽样来评估样本空间。

```
	import hyperopt.pyll.stochastic
	print hyperopt.pyll.stochastic.sample(space)
```

通过 `space` 描述的搜索空间一共有三个超参数：

 - 'a' - 选择案例
 - 'c1' - 在'case 1'中使用的正值参数
 - 'c2' - 在'case 2'中使用的有界实值参数

有一件事要注意，每个可优化的随机表达式都有一个**标签**作为第一个参数。这些标签用于选择参数并将其返回给调用者，并以各种方式在内部进行。

第二件要注意的是，我们在图的中间使用了元组（包裹着'case 1'和'case 2'）。列表，字典和元组都被升级为“确定性函数表达式”，以便它们可以成为搜索空间随机程序 `search space stochastic program` 的一部分。

第三件要注意的是数字表达式 `1 + hp.lognormal('c1', 0, 1)` ，它嵌入到搜索空间的描述中。就优化算法而言，直接在搜索空间中加1和在目标函数本身的逻辑内加1并没有什么区别。那么作为程序设计者，您可以选择将这种处理放在你喜欢的地方，以达到您想要的种类模块化。请注意，搜索空间中的中间表达式结果可以是任意Python对象，即使在使用 `mongodb` 并行优化时也是如此。向搜索空间描述中添加新的非随机表达式是很容易的，参见下面（2.3节）了解如何去做。

第四点要注意的是，'c1'和'c2'是我们称之为条件参数的例子。每一个返回的样本中的“c1”和“c2”都有一个关于“a”的特定值。如果“a”是0，则使用“c1”而不是“c2”。如果'a'是1，那么使用'c2'而不是'c1'。只有这样做才有意义，应该用这种方式将参数编码为条件参数，而不是简单地忽略目标函数中的参数。如果您向程序展示了“c1”有时对目标函数没有影响的事实（因为它对目标函数的参数没有影响），那么Hyperot在搜索可以进行更有效地进行资源分配。

### 2.1参数表达式
目前 `hyperopt` 的优化算法所识别的随机表达式是：

 - hp.choice(label, options)

	- 返回其中一个选项，它应该是一个列表或元组。options元素本身可以是[嵌套]随机表达式。在这种情况下，仅出现在某些选项中的随机选择(stochastic choices)将成为条件参数。

 - hp.randint(label, upper)

	- 返回范围:[0，upper]中的随机整数。当更远的整数值相比较时,这种分布的语义是意味着邻整数值之间的损失函数没有更多的相关性。例如，这是描述随机种子的适当分布。如果损失函数可能更多的与相邻整数值相关联，那么你或许应该用“量化”连续分布的一个，比如 `quniform` ， `qloguniform` ， `qnormal` 或 `qlognormal` 。

 - hp.uniform(label, low, high)

	- 返回位于[low,hight]之间的均匀分布的值。
	- 在优化时，这个变量被限制为一个双边区间。

 - hp.quniform(label, low, high, q)

	- 返回一个值，如 `round（uniform（low，high）/ q）* q` 
	- 适用于目标仍然有点“光滑”的离散值，但是在它上下存在边界(双边区间)。

 - hp.loguniform(label, low, high)

	- 返回根据 `exp（uniform（low，high））` 绘制的值，以便返回值的对数是均匀分布的。
优化时，该变量被限制在[exp（low），exp（high）]区间内。

 - hp.qloguniform(label, low, high, q)

	- 返回类似 `round（exp（uniform（low，high））/ q）* q` 的值
	- 适用于一个离散变量，其目标是“平滑”，并随着值的大小变得更平滑，但是在它上下存在边界(双边区间)。

 - hp.normal(label, mu, sigma)

	- 返回正态分布的实数值，其平均值为 `mu` ，标准偏差为 `σ`。优化时，这是一个无约束(unconstrained)的变量。

 - hp.qnormal(label, mu, sigma, q)

	- 返回像 `round（正常（mu，sigma）/ q）* q` 的值
	- 适用于离散值，可能需要在 `mu` 附近的取值，但从基本上上是无约束(unbounded)的。

 - hp.lognormal(label, mu, sigma)(对数正态分布)

	- 返回根据 `exp（normal（mu，sigma））` 绘制的值，以便返回值的对数正态分布。优化时，这个变量被限制为正值。

 - hp.qlognormal(label, mu, sigma, q)

	- 返回类似 `round（exp（normal（mu，sigma））/ q）* q` 的值
	 - 适用于一个离散变量，其目标是“平滑”，并随着值的大小变得更平滑，变量的大小是从一个边界开始的(单边区间)。
### 2.2搜索空间示例：scikit-learn

为了看到所有可行方案，我们来看看如何在scikit-learn中描述分类算法超参数的空间。（这个想法正在hyperopt-sklearn中开发）

```
	from hyperopt import hp
	space = hp.choice('classifier_type', [
	    {
		'type': 'naive_bayes',
	    },
	    {
		'type': 'svm',
		'C': hp.lognormal('svm_C', 0, 1),
		'kernel': hp.choice('svm_kernel', [
		    {'ktype': 'linear'},
		    {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
		    ]),
	    },
	    {
		'type': 'dtree',
		'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
		'max_depth': hp.choice('dtree_max_depth',
		    [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
		'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
	    },
	    ])
```

### 2.3用 'pyll' 添加非随机表达式

你可以使用这样的节点作为 `pyll` 函数的参数（参见pyll）。如果你想知道更多关于这一点,File a github issue 。

```
	import hyperopt.pyll
	from hyperopt.pyll import scope

	@scope.define
	def foo(a, b=0):
	     print 'runing foo', a, b
	     return a + b / 2

	# -- this will print 0, foo is called as usual.
	print foo(0)

	# In describing search spaces you can use `foo` as you
	# would in normal Python. These two calls will not actually call foo,
	# they just record that foo should be called to evaluate the graph.

	space1 = scope.foo(hp.uniform('a', 0, 10))
	space2 = scope.foo(hp.uniform('a', 0, 10), hp.normal('b', 0, 1)

	# -- this will print an pyll.Apply node
	print space1

	# -- this will draw a sample by running foo()
	print hyperopt.pyll.stochastic.sample(space1)
```
### 2.4增加新的超参数
如果可能的话，应该避免添加新的用于描述参数搜索空间的随机表达式。为了使所有的搜索算法能够在所有的空间上工作，搜索算法必须就描述空间的超参数类型达成一致。作为library的维护者，我可能会不时地加入一些表达方式，但是就像我说的那样，我想尽可能地避免这种表达方式。添加并不是hyperopt的新类型的随机表达式意味着其是可扩展(Adding new kinds of stochastic expressions is not one of the ways hyperopt is meant to be extensible.)。
