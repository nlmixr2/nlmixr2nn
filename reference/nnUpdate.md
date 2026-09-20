# Register generated networks' weight-block layout for solving

Resolves each \`nn()\`-generated network's weight-block base index from
an assembled model's solve (\`par_ptr\`) parameter order, registers it
with the compiled layer via \[nnSetMeta()\], and seeds the loader weight
buffer with the model's current weight values via \[nnSetWeights()\].
The rxode2 par-loader hook then injects those weights into \`par_ptr\`
on every solve.

## Usage

``` r
nnUpdate(x, params = NULL)
```

## Arguments

- x:

  an rxode2 model / ui object, or a character vector of parameter names
  in solve order.

- params:

  parameter order of the model that will be solved, when the caller
  knows it; reconstructed from \`x\` when \`NULL\`.

## Value

invisibly, a data.frame of the registered networks.

## Details

The solve order differs from \`rxode2::rxModelVars(ui)\$params\` for
\`ui\` models: \`par_ptr\` places population parameters first in theta
(\`ntheta\`) order, then covariates. This uses the model \`iniDf\` theta
order when available, and otherwise the declared parameter order (raw
rxode2 models).
