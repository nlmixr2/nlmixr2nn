# Evaluate a model's neural network at chosen inputs

Evaluates the network embedded by \[nn()\] directly, at the weights the
model currently carries – the trained weights of a fit, or the initial
values drawn when the model was parsed. Use it to see the function the
network represents.

## Usage

``` r
nnEval(object, ..., net = 0L)
```

## Arguments

- object:

  an \`rxUi\` model or an \`nlmixr2\` fit containing an \[nn()\] term.
  Named \`object\` rather than \`x\` deliberately: the inputs are
  supplied by name through \`...\`, and a network input called \`x\` is
  entirely ordinary – it would otherwise be captured by the first formal
  instead.

- ...:

  the network inputs, given as named vectors (e.g. \`central = seq(0,
  10, 0.1)\`). Names are for labelling only; arguments are matched to
  the network's inputs positionally.

- net:

  which network to evaluate, for a model containing more than one
  \`nn()\` term (the id shown in the model text, counting from 0).

## Value

a \`data.frame\` of class \`"nlmixr2nnEval"\` with one column per input
and a \`value\` column holding the network output.

## Details

The inputs are supplied by name, in the order they appear in the
\`nn()\` call. Vectors of unequal length are recycled against each
other, so a one-input network takes a single sequence and a two-input
network can be swept over a grid built with \[expand.grid()\].

## Author

Matthew L. Fidler

## Examples

``` r
# \donttest{
mod <- function() {
  model({
    d/dt(central) <- -nn(central, nHidden = 4L)
  })
}
set.seed(1)
ui <- rxode2::rxode2(mod)
#>  
#>  
#> ℹ parameter labels from comments are typically ignored in non-interactive mode
#> ℹ Need to run with the source intact to parse comments
head(nnEval(ui, central = seq(0, 10, length.out = 5)))
#> nn() network 0 (softplus, 1 input) evaluated at 5 points
#>   central       value
#> 1     0.0 -0.04782756
#> 2     2.5  0.05737350
#> 3     5.0  0.09462795
#> 4     7.5  0.13444449
#> 5    10.0  0.17632374
# }
```
