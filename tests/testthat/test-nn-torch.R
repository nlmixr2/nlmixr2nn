## torch backend: the C++ libtorch MLP module is the weight source; its weights
## reach the solve through nnUpdate -> loader hook -> par_ptr -> nn<K>.  The
## definitive check is that the ODE solve's network output equals the module's
## own forward pass.  Also covers weight get/set and save/load round-trips.

skip_if_no_torch <- function() {
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
}

test_that("torch module drives the ODE solve (solve == module forward)", {
  skip_if_no_torch()
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, n_hidden = 5, act = "softplus"); d/dt(A) <- -p * A })
  }
  ui <- rxode2::rxode2(mod)
  nnTorchModel(ui, seed = 1)          # create C++ torch module for network 0
  on.exit({ nnTorchFree(0); nnClearMeta() }, add = TRUE)
  nnUpdate(ui)                        # sync module weights into the loader buffer

  xs <- c(-1.5, -0.2, 0.4, 1.1, 3.0)
  ev <- do.call(rbind, lapply(seq_along(xs), function(i)
    data.frame(id = i, time = 0, x = xs[i], amt = 0, evid = 0)))
  s <- rxode2::rxSolve(ui, ev, returnType = "data.frame", covsInterpolation = "locf")
  for (i in seq_along(xs)) {
    expect_equal(s$y[s$id == i][1], nnTorchForward(0, xs[i]), tolerance = 1e-10)
  }
})

test_that("torch weight get/set round-trips through the module", {
  skip_if_no_torch()
  nnTorchInit(3, K = 2L, H = 4L, act = "tanh", seed = 7)
  on.exit(nnTorchFree(3), add = TRUE)
  w <- nnTorchWeights(3)
  expect_length(w, 2L * 4L + 2L * 4L + 1L)
  w2 <- w + 0.5
  nnTorchSetWeights(3, w2)
  expect_equal(nnTorchWeights(3), w2, tolerance = 1e-12)
})

test_that("save / load restores a trained module", {
  skip_if_no_torch()
  nnTorchInit(5, K = 1L, H = 3L, act = "relu", seed = 2)
  on.exit(nnTorchFree(5), add = TRUE)
  w <- nnTorchWeights(5)
  f <- tempfile(fileext = ".pt")
  nnTorchSave(5, f)
  nnTorchSetWeights(5, w + 1)          # perturb
  expect_gt(max(abs(nnTorchWeights(5) - w)), 0.5)
  nnTorchLoad(5, f)                    # restore
  expect_equal(nnTorchWeights(5), w, tolerance = 1e-10)
  unlink(f)
})
