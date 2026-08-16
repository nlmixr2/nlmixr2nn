## torch training path: the vector-Jacobian product (cotangent on the NN output
## -> d(loss)/d(weights)) and the optimizer step.  This is the same gradient
## bridge the likelihood contribution will use (there the cotangent comes from
## the adjoint sweep); here we drive it with an explicit L2 cotangent.

test_that("VJP gradient matches finite differences of the loss", {
  skip_if_no_torch()
  id <- 0L; K <- 1L; H <- 6L
  nnTorchInit(id, K, H, act = "softplus", seed = 5)
  on.exit(nnTorchFree(id), add = TRUE)

  set.seed(1)
  X <- matrix(seq(-2, 2, length.out = 11), ncol = 1)
  target <- as.numeric(sin(X))

  ## analytic gradient of loss = 0.5*sum((y-target)^2) via the VJP
  w0 <- nnTorchWeights(id)
  nnTorchOptInit(id, "sgd", 0)          # optimizer needed for zero_grad
  nnTorchZeroGrad(id)
  y0 <- nnTorchForwardBatch(id, X)
  nnTorchBackward(id, X, y0 - target)   # cotangent G = dLoss/dy
  gAnalytic <- nnTorchGetGrad(id)

  ## finite-difference gradient of the same loss w.r.t. each weight
  lossAt <- function(w) {
    nnTorchSetWeights(id, w)
    0.5 * sum((nnTorchForwardBatch(id, X) - target)^2)
  }
  eps <- 1e-6
  gNum <- vapply(seq_along(w0), function(j) {
    wp <- w0; wp[j] <- wp[j] + eps
    wm <- w0; wm[j] <- wm[j] - eps
    (lossAt(wp) - lossAt(wm)) / (2 * eps)
  }, numeric(1))
  nnTorchSetWeights(id, w0)             # restore

  expect_equal(gAnalytic, gNum, tolerance = 1e-5)
})

test_that("training reduces the loss and fits the target", {
  skip_if_no_torch()
  id <- 1L
  nnTorchInit(id, K = 1L, H = 12L, act = "softplus", seed = 3)
  on.exit(nnTorchFree(id), add = TRUE)

  X <- matrix(seq(-2, 2, length.out = 41), ncol = 1)
  target <- as.numeric(sin(X))

  loss <- nnTorchTrain(id, X, target, steps = 800L, lr = 0.05, type = "adam")
  expect_lt(tail(loss, 1), 0.05 * loss[1])       # substantial decrease
  expect_lt(tail(loss, 1), 0.5)                  # actually fits reasonably

  ## trained module predicts the target well, and the loader buffer was synced
  ## (nnTorchStep copies weights into the buffer) so nnTorchWeights == module
  yhat <- nnTorchForwardBatch(id, X)
  expect_lt(max(abs(yhat - target)), 0.2)
})
