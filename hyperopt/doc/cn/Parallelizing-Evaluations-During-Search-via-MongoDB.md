# [通过MongoDB在搜索时进行并行计算](https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB)
*[Font Tian](http://blog.csdn.net/fontthrone) translated this article on 23 December 2017*
Hyperopt旨在支持不同类型的试用数据库。默认试用数据库（`Trials`）是用Python列表和字典实现的。默认实现是一个参考实现，很容易处理，但不支持并行评估试验所需的异步更新。对于并行搜索，hyperopt包含一个 `MongoTrials` 支持异步更新的实现。

要运行并行搜索，您需要执行以下操作（在安装 `mongodb` 之后）：

 1. 在网络可见(network-visible)的地方启动一个mongod进程。

 2. 修改您的调用以 `hyperopt.fmin` 连接到该 `mongod` 进程的 `MongoTrials` 后端。

 3. 启动一个或多个 `hyperopt-mongo-worker` 连接到 `mongod` 进程的进程，并在 `fmin `blocks(块) 进行搜索。

### 1.启动一个mongod进程
一旦安装了 `mongodb` ，启动一个数据库进程（mongod）就像打字一样简单

```
	mongod --dbpath . --port 1234
	# or storing each db its own directory is nice:
	mongod --dbpath . --port 1234 --directoryperdb --journal --nohttpinterface
	# or consider starting mongod as a daemon:
	mongod --dbpath . --port 1234 --directoryperdb --fork --journal --logpath log.log --nohttpinterface
```

Mongo有一个预先分配几GB空间的习惯（你可以用 `--noprealloc` 来禁用这个空间）以获得更好的性能，所以当你想创建这个数据库的位置请思考一下这个问题。在联网的文件系统上创建数据库不仅会给您的数据库带来糟糕的性能，还会给网络上的其他人带来不好的体验，请小心。

另外，如果你的机器可以连接互联网，那么绑定到 `loopback` 接口并通过ssh连接，或者读取有关密码保护的 `mongodb` 文档。

本教程的其余部分是基于运行 `mongo` **端口1234**的的**本地主机**。

### 2.使用 `MongoTrials`
假设为了使用便捷，你想 `math.sin` 通过 `hyperopt`  来最小化函数。那么像下面的例子那样在进程中（串行）运行任务：

```
	import math
	from hyperopt import fmin, tpe, hp
	from hyperopt.mongoexp import MongoTrials

	trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')
	best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
```


 `MongoTrials` 的第一个参数告诉它要使用哪个 `mongod`进程，以及该进程中的哪个数据库 `（这里是 'foo_db' ）` 。第二个参数 `（exp_key='exp_1'）` 对于在数据库中标记一组特定的试验非常有用。 `exp_key` 参数是可选的。

注意目前有一个使用要求，数据库名称后必须加上 `“/ jobs”` 字段。

无论你总是把你的试验放在单独的数据库中，还是使用 `exp_key` 机制来区分它们都取决于你的代码。单独使用数据库的好处：可以从shell中操作它们（它们显示为不同的文件），并确保实验的更大的独立性/隔离性。使用exp_key的好处：hyperopt-mongo-worker进程（请参阅下文）在数据库级别进行轮询，以便它们可以同时支持多个使用同一个数据库的实验。

### 3.运行 hyperopt-mongo-worker
如果你运行上面的代码片段，你会发现它在调用 `fmin` 时阻塞（挂起） 。 `MongoTrials` 在内部将自己描述为 `fmin` 作为异步 `trials` 对象(an asynchronous trials object)，所以 `fmin` 在建议新的搜索点时并​实际上​不评估目标函数。相反，它只是坐在那里，耐心等待另一个进程做这个工作，并更新MongoDB的结果。 包含在 `bin` 目录中的`hyperopt-mongo-worker` 脚本就是为这个目的编写的。在你安装hyperopt的时候它应该已经安装在你的 `$PATH` 里面了。

在上面脚本中的调用 `fmin`  并被阻塞的同时，打开一个新的 `shell` 并输入

```
	hyperopt-mongo-worker --mongo=localhost:1234/foo_db --poll-interval=0.1
```
它会从MongoDB中取出一个工作项，评估 `math.sin` 函数，并将结果存回数据库。在 `fmin` 函数尝试了足够的点后，它将返回，上面的脚本将终止。然后 `hyperopt-mongo-worker`脚本会等待几分钟，等待更多的工作出现，然后终止。

在这种情况下，我们显式地设置轮询间隔，因为默认的时间设置是为了至少需要一两分钟完成的作业（搜索点评估）设置的。

### `MongoTrials` 是一个持久对象

如果你再次运行这个例子，

```
	best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
```

你会看到它立即返回，似乎没有进行任何计算。那是因为你连接的数据库已经有足够的尝试(trials); 实际上在你运行第一个实验时已经计算了它们。而如果你想进行一次新的搜索，你可以改变数据库名称或者`exp_key`。如果你想扩展本次搜索，那么你可以为了 `max_evals` 设置一个拥有更高数字的 `fmin` 。

又或者，您可以启动其他专门创建 `MongoTrials` 的进程来分析数据库中已用结果。而那些其他进程根本不需要调用 `fmin`。


