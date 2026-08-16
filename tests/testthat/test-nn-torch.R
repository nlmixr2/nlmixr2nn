## torch backend: the C++ libtorch MLP module is the training optimizer, and it
## is loaded FROM the model's weights rather than generating its own -- that is
## what makes a fit reproducible under a single set.seed().
##
## The invariant worth pinning is therefore that the compiled evaluator the ODE
## integrates and the torch module agree ON THE SAME WEIGHTS.  (This used to be
## written the other way round: the module was the weight source and reached the
## solve through nnUpdate -> loader -> par_ptr.  A model now carries its own
## weights and they deliberately outrank anything an external buffer holds, so
## that phrasing no longer describes the design.)

test_that("the torch module and the ODE solve agree on the model's weights", {
  skip_if_no_torch()
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, n_hidden = 5, act = "softplus"); d/dt(A) <- -p * A })
  }
  set.seed(1)
  ui <- suppressMessages(rxode2::rxode2(mod))
  meta <- rxode2::rxUiDecompress(ui)$nnMeta[["0"]]

  ## load the module from the model, exactly as a fit does
  nnTorchInit(0, meta$K, meta$H, act = meta$act)
  on.exit({ try(nnTorchFree(0), silent = TRUE); nnClearMeta() }, add = TRUE)
  nnTorchSetWeights(0, unname(nnWeights(ui)))

  ## no loader, no nnUpdate, no nnCovData: the model carries its own weights
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
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
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
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
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
