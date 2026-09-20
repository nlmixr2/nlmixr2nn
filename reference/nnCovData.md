# Add placeholder weight-covariate columns for an nn() model to data

The \`nn()\` UDF declares each network's weights as covariates; this
adds a 0-valued column per weight to \`data\` so the model solves. The
par-loader overwrites these slots with the network weights on every
solve, so the placeholder value is irrelevant.

## Usage

``` r
nnCovData(data)
```

## Arguments

- data:

  a data.frame with the estimation/simulation data.

## Value

\`data\` with any missing weight-covariate columns added (value 0).
