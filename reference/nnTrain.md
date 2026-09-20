# Fit an embedded neural network in an ODE by forward-sensitivity + torch

Fit an embedded neural network in an ODE by forward-sensitivity + torch

## Usage

``` r
nnTrain(
  model,
  data,
  pred,
  nnId = 0L,
  K,
  H,
  act = "tanh",
  params = numeric(0),
  inits = numeric(0),
  optimizer = "adam",
  lr = 0.05,
  iter = 50,
  sigma = 1,
  estSigma = TRUE,
  seed = NULL,
  verbose = FALSE
)
```

## Arguments

- model:

  rxode2 model text with one \`g = nn\<K\>(id, ...)\` output and the
  weight block declared via \`param(nnWeightLayout(...))\`.

- data:

  event data frame (columns id, time, evid, dv, amt as needed).

- pred:

  name of the predicted state that \`dv\` observes.

- nnId, K, H, act:

  network id, input dim, hidden width, activation.

- params:

  named numeric of fixed population parameters (non-weight).

- inits:

  named numeric of base-state initial conditions.

- optimizer:

  "adam" or "sgd"; \`lr\` learning rate; \`iter\` iterations.

- sigma:

  initial residual SD; \`estSigma\` profiles it by MLE each iteration.

- seed:

  optional torch init seed; \`verbose\` prints the LL trace.

## Value

list(weights, sigma, ll, llTrace).
