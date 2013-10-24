"""

Suppose we have hyperparameters:

    X, Y, Z

The function (f) we want to minimize has the property that Z is only used when
X takes value 1, but X and Y are always used.

We make this aspect of f apparent to the optimization engine by specifying a
sampling space of the form:

    P(Z | X) P(X) P(Y)

The semantics of optimization are such that we may ignore the independence
relationships in the prior, and construct densities over the space that are
less independent.

    P(Z | X, Y) P(X, Y)


Optimization in the style of TPE requires that we optimize the ratio of two
distributions over the space, which requires that we actually pick forms for
the various terms contributing to the joint distributions.

The nature of choice() nodes is such that by conditional structure, they also
tell us to rule out large parts of the search space (e.g. P(Z | X=0) is
ignored) and we want to retain that.  In this case, we retain that information
by ensuring that P(Z | X, Y) is fixed to the same distribution when X = 0, in
both the "lower" and "upper" distributions.


Supposing we had a real-valued Y, and Z and a binary-valued X, we could choose

    P(X, Y)
    = P(X) P(Y | X)
    = Bern(X) Normal(Y| X)
    \propto (X * pX1 + (1 - X) * pX0)
      * (  ((X) * exp((Y - muX1)^2 / sigmaX1^2))
         + ((1 - X) * exp((Y - muX0)^2 / sigmaX0^2)))

    P(Z | X, Y)
    = cZ1 * X * exp((Z - aY + b)^2 / sigmaZ^2) + cZ0 * (1 - X) * exp(-Z^2)


In which case the ratio between the "lower" and "upper" densities would look
like:

    \frac{P(X, Y, Z; lower-data)}{P(X, Y, Z; upper-data)}
    = \log(P(X, Y, Z; lower-data)) - \log(P(X, Y, Z; upper-data))

"""

