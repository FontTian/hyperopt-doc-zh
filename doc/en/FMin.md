This page is a tutorial on basic usage of `hyperopt.fmin()`.
It covers how to write an objective function that fmin can optimize, and how to describe a search space that fmin can search.

Hyperopt's job is to find the best value of a scalar-valued, possibly-stochastic function over a set of possible arguments to that function.
Whereas many optimization packages will assume that these inputs are drawn from a vector space,
Hyperopt is different in that it encourages you to describe your search space in more detail.
By providing more information about where your function is defined, and where you think the best values are, you allow algorithms in hyperopt to search more efficiently.

The way to use hyperopt is to describe:

* the objective function to minimize
* the space over which to search
* the database in which to store all the point evaluations of the search
* the search algorithm to use

This (most basic) tutorial will walk through how to write functions and search spaces,
using the default `Trials` database, and the dummy `random` search algorithm.
Section (1) is about the different calling conventions for communication between an objective function and hyperopt.
Section (2) is about describing search spaces.

Parallel search is possible when replacing the `Trials` database with
a `MongoTrials` one;
there is another wiki page on the subject of [using mongodb for parallel search](Parallelizing-Evaluations-During-Search-via-MongoDB).

Choosing the search algorithm is as simple as passing `algo=hyperopt.tpe.suggest` instead of `algo=hyperopt.random.suggest`.
The search algorithms are actually callable objects, whose constructors
accept configuration arguments, but that's about all there is to say about the
mechanics of choosing a search algorithm.

## 1. Defining a Function to Minimize

Hyperopt provides a few levels of increasing flexibility / complexity when it comes to specifying an objective function to minimize.
The questions to think about as a designer are
* Do you want to save additional information beyond the function return value, such as other statistics and diagnostic information collected during the computation of the objective?
* Do you want to use optimization algorithms that require more than the function value?
* Do you want to communicate between parallel processes? (e.g. other workers, or the minimization algorithm)

The next few sections will look at various ways of implementing an objective
function that minimizes a quadratic objective function over a single variable.
In each section, we will be searching over a bounded range from -10 to +10,
which we can describe with a *search space*:
```
space = hp.uniform('x', -10, 10)
```

Below, Section 2, covers how to specify search spaces that are more complicated.

### 1.1 The Simplest Case

The simplest protocol for communication between hyperopt's optimization
algorithms and your objective function, is that your objective function
receives a valid point from the search space, and returns the floating-point
*loss* (aka negative utility) associated with that point.


```python
from hyperopt import fmin, tpe, hp
best = fmin(fn=lambda x: x ** 2,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100)
print best
```

This protocol has the advantage of being extremely readable and quick to
type. As you can see, it's nearly a one-liner.
The disadvantages of this protocol are
(1) that this kind of function cannot return extra information about each evaluation into the trials database,
and
(2) that this kind of function cannot interact with the search algorithm or other concurrent function evaluations.
You will see in the next examples why you might want to do these things.


### 1.2 Attaching Extra Information via the Trials Object

If your objective function is complicated and takes a long time to run, you will almost certainly want to save more statistics
and diagnostic information than just the one floating-point loss that comes out at the end.
For such cases, the fmin function is written to handle dictionary return values.
The idea is that your loss function can return a nested dictionary with all the statistics and diagnostics you want.
The reality is a little less flexible than that though: when using mongodb for example,
the dictionary must be a valid JSON document.
Still, there is lots of flexibility to store domain specific auxiliary results.

When the objective function returns a dictionary, the fmin function looks for some special key-value pairs
in the return value, which it passes along to the optimization algorithm.
There are two mandatory key-value pairs:
* `status` - one of the keys from `hyperopt.STATUS_STRINGS`, such as 'ok' for
  successful completion, and 'fail' in cases where the function turned out to
  be undefined.
* `loss` - the float-valued function value that you are trying to minimize, if
  the status is 'ok' then this has to be present.

The fmin function responds to some optional keys too:

* `attachments` -  a dictionary of key-value pairs whose keys are short
  strings (like filenames) and whose values are potentially long strings (like
  file contents) that should not be loaded from a database every time we
  access the record. (Also, MongoDB limits the length of normal key-value
  pairs so once your value is in the megabytes, you may *have* to make it an
  attachment.)
* `loss_variance` - float - the uncertainty in a stochastic objective function
* `true_loss` - float -
  When doing hyper-parameter optimization, if you store the generalization error of your model with this name, then you can sometimes get spiffier output from the built-in plotting routines.
* `true_loss_variance` - float - the uncertainty in the generalization error

Since dictionary is meant to go with a variety of back-end storage
mechanisms, you should make sure that it is JSON-compatible.  As long as it's
a tree-structured graph of dictionaries, lists, tuples, numbers, strings, and
date-times, you'll be fine.

**HINT:** To store numpy arrays, serialize them to a string, and consider storing
them as attachments.

Writing the function above in dictionary-returning style, it
would look like this:

```python
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

### 1.3 The Trials Object

To really see the purpose of returning a dictionary,
let's modify the objective function to return some more things,
and pass an explicit `trials` argument to `fmin`.

```python
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

In this case the call to fmin proceeds as before, but by passing in a trials object directly,
we can inspect all of the return values that were calculated during the experiment.

So for example:
* `trials.trials` - a list of dictionaries representing everything about the search
* `trials.results` - a list of dictionaries returned by 'objective' during the search
* `trials.losses()` - a list of losses (float for each 'ok' trial)
* `trials.statuses()` - a list of status strings

This trials object can be saved, passed on to the built-in plotting routines,
or analyzed with your own custom code.

The *attachments* are handled by a special mechanism that makes it possible to use the same code
for both `Trials` and `MongoTrials`.

You can retrieve a trial attachment like this, which retrieves the 'time_module' attachment of the 5th trial:
```python
msg = trials.trial_attachments(trials.trials[5])['time_module']
time_module = pickle.loads(msg)
```

The syntax is somewhat involved because the idea is that attachments are large strings,
so when using MongoTrials, we do not want to download more than necessary.
Strings can also be attached globally to the entire trials object via trials.attachments,
which behaves like a string-to-string dictionary.


**N.B.** Currently, the trial-specific attachments to a Trials object are tossed into the same global trials attachment dictionary, but that may change in the future and it is not true of MongoTrials.



### 1.4 The Ctrl Object for Realtime Communication with MongoDB

It is possible for `fmin()` to give your objective function a handle to the mongodb used by a parallel experiment. This mechanism makes it possible to update the database with partial results, and to communicate with other concurrent processes that are evaluating different points.
Your objective function can even add new search points, just like `random.suggest`.

The basic technique involves:

* Using the `fmin_pass_expr_memo_ctrl` decorator
* call `pyll.rec_eval` in your own function to build the search space point
  from `expr` and `memo`.
* use `ctrl`, an instance of `hyperopt.Ctrl` to communicate with the live
  trials object.

It's normal if this doesn't make a lot of sense to you after this short tutorial,
but I wanted to give some mention of what's possible with the current code base,
and provide some terms to grep for in the hyperopt source, the unit test,
and example projects, such as [hyperopt-convnet](https://github.com/jaberg/hyperopt-convnet).
Email me or file a github issue if you'd like some help getting up to speed with this part of the code.


## 2. Defining a Search Space

A search space consists of nested function expressions, including stochastic expressions.
The stochastic expressions are the hyperparameters.
Sampling from this nested stochastic program defines the random search algorithm.
The hyperparameter optimization algorithms work by replacing normal "sampling" logic with
adaptive exploration strategies, which make no attempt to actually sample from the distributions specified in the search space.

It's best to think of search spaces as stochastic argument-sampling programs. For example
```python
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])
```
The result of running this code fragment is a variable `space` that refers to a graph of expression identifiers and their arguments.
Nothing has actually been sampled, it's just a graph describing *how* to sample a point.
The code for dealing with this sort of expression graph is in `hyperopt.pyll` and I will refer to these graphs as *pyll graphs* or *pyll programs*.

If you like, you can evaluate a sample space by sampling from it.
```python
import hyperopt.pyll.stochastic
print hyperopt.pyll.stochastic.sample(space)
```

This search space described by `space` has 3 parameters:
* 'a' - selects the case
* 'c1' - a positive-valued parameter that is used in 'case 1'
* 'c2' - a bounded real-valued parameter that is used in 'case 2'

One thing to notice here is that every optimizable stochastic expression has a *label* as the first argument.
These labels are used to return parameter choices to the caller, and in various ways internally as well.

A second thing to notice is that we used tuples in the middle of the graph (around each of 'case 1' and 'case 2').
Lists, dictionaries, and tuples are all upgraded to "deterministic function expressions" so that they can be part of the search space stochastic program.

A third thing to notice is the numeric expression `1 + hp.lognormal('c1', 0, 1)`, that is embedded into the description of the search space.
As far as the optimization algorithms are concerned, there is no difference between adding the 1 directly in the search space
and adding the 1 within the logic of the objective function itself.
As the designer, you can choose where to put this sort of processing to achieve the kind modularity you want.
Note that the intermediate expression results within the search space can be arbitrary Python objects, even when optimizing in parallel using mongodb.
It is easy to add new types of non-stochastic expressions to a search space description, see below (Section 2.3) for how to do it.

A fourth thing to note is that 'c1' and 'c2' are examples what we will call *conditional parameters*.
Each of 'c1' and 'c2' only figures in the returned sample for a particular value of 'a'.
If 'a' is 0, then 'c1' is used but not 'c2'.
If 'a' is 1, then 'c2' is used but not 'c1'.
Whenever it makes sense to do so, you should encode parameters as conditional ones this way,
rather than simply ignoring parameters in the objective function.
If you expose the fact that 'c1' sometimes has no effect on the objective function (because it has no effect on the argument to the objective function) then search can be more efficient about credit assignment.


### 2.1 Parameter Expressions

The stochastic expressions currently recognized by hyperopt's optimization algorithms are:

* `hp.choice(label, options)`
   * Returns one of the options, which should be a list or tuple.
       The elements of `options` can themselves be [nested] stochastic expressions.
       In this case, the stochastic choices that only appear in some of the options become *conditional* parameters.

* `hp.randint(label, upper)`
   * Returns a random integer in the range [0, upper). The semantics of this
       distribution is that there is *no* more correlation in the loss function between nearby integer values,
       as compared with more distant integer values.  This is an appropriate distribution for describing random seeds    for example.
       If the loss function is probably more correlated for nearby integer values, then you should probably use one of the "quantized" continuous distributions, such as either `quniform`, `qloguniform`, `qnormal` or `qlognormal`.

* `hp.uniform(label, low, high)`
   * Returns a value uniformly between `low` and `high`.
   * When optimizing, this variable is constrained to a two-sided interval.

* `hp.quniform(label, low, high, q)`
    * Returns a value like round(uniform(low, high) / q) * q
    * Suitable for a discrete value with respect to which the objective is still somewhat "smooth", but which should be bounded both above and below.

* `hp.loguniform(label, low, high)`
    * Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the return value is uniformly distributed.
    * When optimizing, this variable is constrained to the interval [exp(low), exp(high)].

* `hp.qloguniform(label, low, high, q)`
    * Returns a value like round(exp(uniform(low, high)) / q) * q
    * Suitable for a discrete variable with respect to which the objective is "smooth" and gets smoother with the size of the value, but which should be bounded both above and below.

* `hp.normal(label, mu, sigma)`
    * Returns a real value that's normally-distributed with mean mu and standard deviation sigma. When optimizing, this is an unconstrained variable.

* `hp.qnormal(label, mu, sigma, q)`
    * Returns a value like round(normal(mu, sigma) / q) * q
    * Suitable for a discrete variable that probably takes a value around mu, but is fundamentally unbounded.

* `hp.lognormal(label, mu, sigma)`
    * Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the return value is normally distributed.
        When optimizing, this variable is constrained to be positive.

* `hp.qlognormal(label, mu, sigma, q)`
    * Returns a value like round(exp(normal(mu, sigma)) / q) * q
    * Suitable for a discrete variable with respect to which the objective is smooth and gets smoother with the size of the variable, which is bounded from one side.

### 2.2 A Search Space Example: scikit-learn

To see all these possibilities in action, let's look at how one might go about describing the space of hyperparameters of classification algorithms in scikit-learn.
(This idea is being developed in [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn))

```python
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


### 2.3 Adding Non-Stochastic Expressions with pyll

You can use such nodes as arguments to pyll functions (see pyll).
File a github issue if you want to know more about this.

In a nutshell, you just have to decorate a top-level (i.e. pickle-friendly) function so
that it can be used via the `scope` object.

```python
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


### 2.4 Adding New Kinds of Hyperparameter

Adding new kinds of stochastic expressions for describing parameter search spaces should be avoided if possible.
In order for all search algorithms to work on all spaces, the search algorithms must agree on the kinds of hyperparameter that describe the space.
As the maintainer of the library, I am open to the possibility that some kinds of expressions should be added from time to time, but like I said, I would like to avoid it as much as possible.
Adding new kinds of stochastic expressions is not one of the ways hyperopt is meant to be extensible.
