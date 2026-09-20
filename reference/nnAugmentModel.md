# Build the forward-sensitivity augmented NN model text

Supports one or more \`nn\<K\>(id, ...)\` outputs used in the ODE RHS.
The weight variational states are indexed by a GLOBAL weight index
across all networks (network 0's weights first, then network 1's, ...),
so a single-network model keeps the original
\`rx_sw\_\<state\>\_\<j\>\_\` layout; \`rx_drdg\_\<state\>\_\` gains a
\`\<id\>\_\` suffix only when more than one network is present.

## Usage

``` r
nnAugmentModel(modelText, H)
```

## Arguments

- modelText:

  rxode2 model text containing the \`nn\<K\>(id, ...)\` output(s).

- H:

  hidden width: a scalar for a single network, or a vector named by
  network id (character) for multiple networks.

## Value

augmented model text (character scalar): the base model, \`rx_drdg\_\*\`
outputs (dR/dg per state per network), and the
\`rx_sw\_\<state\>\_\<globalj\>\_\` variational states whose RHS is the
F_X.s block plus the per-network forcing.
