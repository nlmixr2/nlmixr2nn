## Analytic weight gradient d(output)/d(weight): the forcing factor d(g)/d(w) for
## the forward-sensitivity variational state of each NN weight.  Computed in
## plain C (thread-safe, no torch in the hot loop); validated here against torch
## autograd (a backward pass with unit cotangent yields d(output)/d(w)).

test_that("analytic nnWeightGrad matches torch autograd across activations/K", {
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  code <- c(relu = 0L, softplus = 1L, tanh = 2L, gelu = 3L, silu = 4L)
  for (act in c("softplus", "tanh", "gelu", "silu")) {
    for (K in c(1L, 2L, 3L)) {
      H <- 4L
      nnTorchInit(0L, K, H, act = act, seed = 7)
      w <- nnTorchWeights(0L)
      set.seed(K); x <- rnorm(K)
      ana <- .Call("_nlmixr2nn_nnWeightGradW", K, H, code[[act]],
                   as.double(w), as.double(x), PACKAGE = "nlmixr2nn")
      expect_length(ana, H * K + 2 * H + 1)
      ## torch autograd: d(1 * output)/dw
      nnTorchOptInit(0L, "sgd", 0)
      nnTorchZeroGrad(0L)
      nnTorchBackward(0L, matrix(x, 1, K), 1.0)
      tor <- nnTorchGetGrad(0L)
      expect_equal(ana, tor, tolerance = 1e-9, info = paste(act, K))
      nnTorchFree(0L)
    }
  }
})
