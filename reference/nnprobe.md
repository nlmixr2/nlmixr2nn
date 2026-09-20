# Probe the current solve parameter vector (development/validation only)

\`nnprobe(idx, x)\` returns \`par_ptr\[idx\]\` of the first subject in
the current rxode2 solve; \`nnnpars(x)\` returns the number of
parameters. These validate that a package-exported custom function can
read the solve's parameter vector, which the neural-network functions
rely on.

## Usage

``` r
nnprobe(idx, x = 0)

nnnpars(x = 0)
```

## Arguments

- idx:

  zero-based parameter index.

- x:

  ignored placeholder argument.

## Value

numeric vector.
