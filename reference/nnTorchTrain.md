# Train a torch module to a target with an L2 loss (standalone helper)

Runs \`steps\` optimizer iterations minimizing \`0.5\*sum((y -
target)^2)\` over the batch, using the cotangent \`G = y - target\` -\>
\[nnTorchBackward()\] -\> \[nnTorchStep()\]. This exercises the same VJP
path the likelihood-contribution gradient bridge will use (there the
cotangent comes from the adjoint sweep).

## Usage

``` r
nnTorchTrain(id, X, target, steps = 200L, lr = 0.05, type = "adam")
```

## Arguments

- id:

  network id.

- X:

  input matrix (\`N\` x \`K\`).

- target:

  numeric target vector (length \`N\`).

- steps:

  number of optimizer iterations.

- lr:

  learning rate.

- type:

  "adam" or "sgd".

## Value

numeric vector of the loss at each step.
