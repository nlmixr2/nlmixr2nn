# Weight-block layout for a single-hidden-layer MLP

Returns the ordered weight names for a \`K\`-input, \`H\`-hidden
network, matching the contiguous layout \`nn\<K\>()\` expects: \`W1\`
(H\*K, row-major), \`b1\` (H), \`W2\` (H), \`b2\` (1).

## Usage

``` r
nnWeightLayout(id, K, H)
```

## Arguments

- id:

  network id used to prefix names.

- K:

  input dimension.

- H:

  hidden width.

## Value

character vector of length \`H\*K + 2\*H + 1\`.
