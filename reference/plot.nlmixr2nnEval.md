# Plot a network's learned function

Plot a network's learned function

## Usage

``` r
# S3 method for class 'nlmixr2nnEval'
plot(x, true = NULL, ...)
```

## Arguments

- x:

  an \`nnEval()\` result.

- true:

  optional reference to overlay: a function of the first input, or a
  numeric vector aligned with the rows of \`x\`. Use it to compare a
  learned term against the mechanism that generated the data.

- ...:

  passed to the underlying plotting call.

## Value

a \`ggplot\` object when ggplot2 is available, otherwise \`NULL\`
invisibly (the plot is drawn).

## Author

Matthew L. Fidler
