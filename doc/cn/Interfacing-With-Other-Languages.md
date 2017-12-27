# [其他语言接口](https://github.com/hyperopt/hyperopt/wiki/Interfacing-With-Other-Languages)
*[Font Tian](http://blog.csdn.net/fontthrone) translated this article on 23 December 2017*
## 两种接口策略

 基本上有两种方法将 `hyperopt` 与其他语言进行连接：

 1. 你可以为你的成本函数中编写一个Python包装器，这个函数不是用Python编写的(用Python调用其它语言,译者注)
 2. 您可以将 `hyperopt-mongo-worker` 替换为直接使用JSON与MongoDB通信。

## 包装对非Python代码的调用
使用hyperopt优化非python函数的参数（例如外部可执行文件）的最简单方法是在外部可执行文件周围编写一个Python函数包装器。假设你有一个可执行文件 `foo` 需要一个整数的命令行参数 `--n` 并打印出一个分数，你可以像这样包装它：

```
	import subprocess
	def foo_wrapper(n):
	    # Optional: write out a script for the external executable
	    # (we just call foo with the argument proposed by hyperopt)
	    proc = subprocess.Popen(['foo', '--n', n], stdout=subprocess.PIPE)
	    proc_out, proc_err = proc.communicate()
	    # <you might have to do some more elaborate parsing of foo's output here>
	    score = float(proc_out)
	    return score
```

当然，要优化 `n` 参数给 `foo` 你也需要调用 `hyperopt.fmin` ，并且定义搜索空间。我觉得你会想在Python中做这个部分。

```
	from hyperopt import fmin, hp, random

	best_n = fmin(foo_wrapper, hp.quniform('n', 1, 100, 1), algo=random.suggest)

	print best_n
```

当这里的搜索空间大于简单的搜索空间时，您可能需要或者必须包装函数来将其参数转换为外部可执行文件的某种 配置文件/脚本。

这种方法与MongoTrials完全兼容。

## 直接与MongoDB进行通信

通过直接与MongoDB进行通信，可以更直接地与搜索过程（使用 `MongoTrials` 时）进行交互，就像 `hyperopt-mongo-worker` 一样。该内容已经超过了本教程的范围，但Hannes Schultz（@ Contemporaryer）的hyperopt与他的MDBQ项目可以作为有不错的参考，这是一个独立的基于MongoDB的任务队列：

[https://github.com/temporaer/MDBQ/blob/master/src/example/hyperopt_client.cpp](https://github.com/temporaer/MDBQ/blob/master/src/example/hyperopt_client.cpp)

查看代码以及 [hyperopt / mongoexp.py](https://github.com/jaberg/hyperopt/blob/master/hyperopt/mongoexp.py) 的内容，了解工作进程如何在工作队列中保留作业，并将结果存储回MongoDB。






