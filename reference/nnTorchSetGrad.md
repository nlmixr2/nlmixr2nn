# Set the network's parameter gradients from a flat vector (nnWeightLayout order)

Injects an externally computed gradient (e.g. the analytic dLoss/dw from
the ODE forward sensitivity) into the torch parameters' \`.grad\`, to be
applied by the next \[nnTorchStep()\].

## Usage

``` r
nnTorchSetGrad(id, grad)
```

## Arguments

- id:

  network id.

- grad:

  numeric gradient vector of length \`H\*K + 2\*H + 1\`.

## Value

invisibly NULL.
