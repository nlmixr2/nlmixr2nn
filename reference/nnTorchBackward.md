# Accumulate the vector-Jacobian product d(sum G\*y)/d(w) into the grads

Accumulate the vector-Jacobian product d(sum G\*y)/d(w) into the grads

## Usage

``` r
nnTorchBackward(id, X, G)
```

## Arguments

- id:

  network id.

- X:

  numeric matrix of inputs, \`N\` rows by \`K\` columns.

- G:

  numeric cotangent vector of length \`N\` (d(loss)/d(output)).

## Value

invisibly NULL.
