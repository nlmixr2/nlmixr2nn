## Fit reproducibility.
##
## `set.seed()` fixing the PARSED model's weights is covered in
## test-nn-zeroboiler.R.  This covers the stronger claim -- that a whole fit is
## reproducible -- which was asserted for a while before it was true.
##
## It was not true because the input-scaling trial solve ran before the training
## loop's first weight injection, so it read whatever the previous fit had left
## in the loader buffer (zeros in a fresh session).  The derived scale therefore
## varied run to run from identical inputs, and every downstream number with it.
## The first run of a session differing from later ones is the signature to
## watch for if this regresses.

test_that("a fit is reproducible under a fixed seed, in one session", {
  skip_if_no_est()
  skip_if_no_torch()
  local_nn(threads = 1L)

  d <- nnSimData(ns = 6L, seed = 1L)
  runOnce <- function() {
    set.seed(7)
    f <- nnFit(nnModUde(), d, "focei",
               nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 2L,
                                        maxInnerIterations = 20L, calcTables = FALSE),
               nn = nnControl(mode = "iter", rounds = 2L, wSteps = 1L,
                              warmStart = "none"))
    list(objf = f$objf, w = unname(f$nnWeights),
         rmse = f$nnParHist$rmse, scale = .nnEnv$scales)
  }
  a <- runOnce()
  b <- runOnce()

  ## the derived input scale is the thing that used to drift
  expect_equal(a$scale, b$scale)
  ## and everything that depends on it
  expect_identical(a$objf, b$objf)
  expect_equal(a$w, b$w, tolerance = 0)
  expect_equal(a$rmse, b$rmse, tolerance = 0)
})

test_that("the input scale does not depend on leftover state from a previous fit", {
  ## The first fit in a session used to derive a different scale from the second,
  ## because the second inherited the first's weights in the loader buffer.
  ## Fitting a DIFFERENT model in between is the sharpest version of that.
  skip_if_no_est()
  skip_if_no_torch()
  local_nn(threads = 1L)

  d <- nnSimData(ns = 6L, seed = 1L)
  scaleOf <- function() {
    assign("scales", list(), envir = .nnEnv)
    set.seed(7)
    nnFit(nnModUde(), d, "focei",
          nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 1L,
                                   maxInnerIterations = 10L, calcTables = FALSE),
          nn = nnControl(mode = "iter", rounds = 1L, wSteps = 1L, warmStart = "none"))
    .nnEnv$scales
  }
  first <- scaleOf()

  ## an unrelated nn fit in between, leaving its own weights behind
  set.seed(11)
  invisible(nnFit(nnModQsp(), nnSimData(ns = 5L, seed = 3L, iiv = FALSE), "bobyqa",
                  nn = nnControl(rounds = 5L)))

  expect_equal(scaleOf(), first)
})
