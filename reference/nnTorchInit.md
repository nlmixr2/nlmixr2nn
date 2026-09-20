# Create a C++ torch module for a network

Create a C++ torch module for a network

## Usage

``` r
nnTorchInit(id, K, H, act = "relu", seed = NULL)
```

## Arguments

- id:

  network id.

- K:

  input dimension.

- H:

  hidden width.

- act:

  activation ("relu", "softplus", "tanh").

- seed:

  optional integer seed for weight initialization.

## Value

invisibly, the id.
