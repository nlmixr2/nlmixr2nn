# Register a network's weight-block layout for solving

The single-hidden-layer MLP \`nn\<K\>(id, ...)\` reads its weights from
a contiguous block of the solve parameter vector. This records, for
network \`id\`, where that block starts (\`base\`, zero-based index into
\`rxode2::rxModelVars(model)\$params\`) and the network dimensions.

## Usage

``` r
nnSetMeta(id, base, K, H, act = "relu")
```

## Arguments

- id:

  integer network id (0-based, matches the first \`nn\<K\>()\`
  argument).

- base:

  zero-based index of the first weight (\`W1\[0\]\`) in the model's
  parameter order.

- K:

  input dimension.

- H:

  hidden width.

- act:

  activation: one of "relu", "softplus", "tanh".

## Value

invisibly TRUE.
