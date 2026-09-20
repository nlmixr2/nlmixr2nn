# Set a network's externally-owned weight buffer

Stores the weights (in \`nnWeightLayout()\` order) that the rxode2
par-loader hook injects into the reserved \`par_ptr\` block on every
solve. This is how torch-trained weights reach the solve without being
nlmixr2 parameters.

## Usage

``` r
nnSetWeights(id, values)
```

## Arguments

- id:

  integer network id.

- values:

  numeric weight vector of length \`H\*K + 2\*H + 1\`.

## Value

invisibly TRUE.
