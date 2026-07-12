## Outer-problem training step: nnOuterStep turns the method-agnostic [f, states]
## matrix (from nlmixr2est's foceiOfv0 callback) into dLL/dw and a torch step.
## Here we validate the consumer arithmetic + optimizer step on a synthetic
## matrix (the full FOCEI end-to-end fit is exercised separately).

test_that("nnOuterStep assembles dLL/dw from the matrix and steps the optimizer", {
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")

  K <- 2L; H <- 1L; nW <- H * K + 2L * H + 1L        # 5 weights
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = K, H = H, act = "tanh")
  nnTorchInit(0L, K, H, act = "tanh", seed = 5); nnTorchOptInit(0L, "adam", 0.05)
  on.exit({ nnOuterUnregister(); nnClearMeta(); nnTorchFree(0L) }, add = TRUE)

  set.seed(1)
  nObs <- 20L
  f  <- runif(nObs, 1, 10)
  sw <- matrix(rnorm(nObs * nW), nObs, nW)           # synthetic rx_sw = df/dw per weight
  mat <- cbind(f, sw)                                # [f, s1..s_nW]; swCols = 2:(1+nW)
  dv <- f + rnorm(nObs, 0, 0.5)

  nnOuterRegister(0L, swCols = 2:(1 + nW), dv = dv)

  w0 <- nnTorchWeights(0L)
  nlmixr2nn:::nnOuterStep(mat)
  w1 <- nnTorchWeights(0L)

  ## manual dLL/dw = sum_obs (dv - f)/sigma^2 * rx_sw
  resid <- dv - f; sigma <- sqrt(mean(resid^2)); dLLdf <- resid / sigma^2
  dLLdw <- vapply(2:(1 + nW), function(cn) sum(dLLdf * mat[, cn]), numeric(1))

  expect_equal(nlmixr2nn:::.nnOuterEnv$dLLdw, dLLdw, tolerance = 1e-10)
  expect_false(isTRUE(all.equal(w0, w1)))            # optimizer moved the weights
  expect_length(nlmixr2nn:::.nnOuterEnv$llTrace, 1L)
})
