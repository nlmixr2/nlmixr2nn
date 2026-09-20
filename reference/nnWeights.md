# Network weights carried by a model or fit

Returns the weight vector a model currently uses: the values trained by
a fit when it has been fitted, otherwise the initial values drawn when
the model was parsed.

## Usage

``` r
nnWeights(x)
```

## Arguments

- x:

  an \`rxUi\` model or an \`nlmixr2\` fit built with \[nn()\].

## Value

a named numeric vector of weights, or \`NULL\` if the model has none.

## Author

Matthew L. Fidler
