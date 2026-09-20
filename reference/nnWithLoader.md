# Run an expression with the nn parameter-loader active

The nn weight loader is registered under the name
\`"nlmixr2nn:nnParLoader"\` and runs only while it is the active
injector, so it never overwrites an unrelated model's parameters. The
transparent \`nlmixr2()\` workflow activates it automatically; when
solving a model that contains \`nn()\` DIRECTLY (\`rxSolve()\` after
\`nnUpdate()\`/\`nnSetWeights()\`), wrap the solve in \`nnWithLoader()\`
so the weights reach \`par_ptr\`.

## Usage

``` r
nnWithLoader(expr)
```

## Arguments

- expr:

  expression to evaluate with the nn loader active.

## Value

the value of \`expr\`.

## Author

Matthew L. Fidler
