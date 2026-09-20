# Analytic gradient of a network's output w.r.t. every weight

Returns d(output)/d(w) in \`nnWeightLayout()\` order for the registered
network \`id\` at input \`x\`, reading the weights from the loader
buffer. Computed in plain C (thread-safe); this is the \`d(g)/d(w)\`
forcing factor for the forward-sensitivity variational states of the NN
weights.

## Usage

``` r
nnWeightGrad(id, x)
```

## Arguments

- id:

  integer network id (must be registered via \[nnSetMeta()\]).

- x:

  numeric input vector of length K.

## Value

numeric vector of length \`H\*K + 2\*H + 1\`.
