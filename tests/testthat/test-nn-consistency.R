## The network a model EVALUATES must be the network its weights DESCRIBE.
##
## This is the invariant that nothing was checking, and its absence hid a real
## bug for a long time.  The compiled evaluator reads its weights from a
## contiguous block of the solve parameter vector, at an offset registered
## beforehand.  That offset differs between the base model, FOCEi's inner model,
## and the augmented sensitivity model -- and reading at the wrong one does not
## fail: it silently evaluates a DIFFERENT network.
##
## The symptom was a fit whose own `g` differed from the network at its trained
## weights by more than the entire range of `g`, while every existing test still
## passed, because the training loop optimizes whatever function it is actually
## evaluating -- self-consistently, and wrongly.

test_that("a plain solve evaluates exactly the network its weights describe", {
  skip_if_not_installed("rxode2")
  local_nn()
  m <- function() {
    model({
      g <- nn(centr, n_hidden = 3L, act = "tanh")
      d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
    })
  }
  set.seed(5)
  ui <- suppressMessages(rxode2::rxode2(m))
  ev <- data.frame(id = 1, time = c(0, .5, 1, 2, 4, 6, 8, 10),
                   evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7)))
  s <- rxode2::rxSolve(ui, ev, returnType = "data.frame")
  ## the solve's own g, against the network evaluated from the model's weights
  expect_equal(s$g, nnEval(ui, centr = s$centr)$value, tolerance = 1e-12)
  ## and g must actually vary -- a degenerate network would satisfy the above
  ## while telling us nothing
  expect_gt(diff(range(s$g)), 1e-6)
})

test_that("the weight-block base matches the model that is actually solved", {
  skip_if_not_installed("rxode2")
  skip_if_no_est()
  local_nn()
  m <- nnModUde()
  set.seed(7)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  u <- rxode2::rxUiDecompress(ui)
  wn <- u$nnMeta[["0"]]$weights

  ## the offset the registry is given...
  registered <- .nnSolveParams(ui)
  ## ...must be the offset in the model an estimation actually solves
  inner <- suppressWarnings(suppressMessages(u$foceiModel$inner))
  expect_equal(match(wn, registered), match(wn, rxode2::rxModelVars(inner)$params))
})

test_that("a fit evaluates the network its trained weights describe", {
  ## The end-to-end version, and the one that failed: it reported a difference of
  ## 0.178 where g's whole range was 0.023.
  skip_if_no_est()
  skip_if_no_torch()
  local_nn(threads = 1L)

  d <- nnSimData(ns = 6L, seed = 1L)
  set.seed(7)
  f <- nnFit(nnModUde(), d, "focei",
             nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 3L,
                                      maxInnerIterations = 20L, calcTables = TRUE),
             nn = nnControl(mode = "iter", rounds = 2L, wSteps = 1L,
                            warmStart = "none"))
  skip_if(is.null(f$g), "fit did not return the nn lhs")

  etas <- stats::setNames(f$eta$eta.nn, as.character(f$eta$ID))
  ev <- nnEval(f, centr = f$IPRED, eta.nn = unname(etas[as.character(f$ID)]))$value

  ## Compared against the SPREAD of g, not an absolute tolerance: what went wrong
  ## before was not a small numerical drift, it was a different function.  The
  ## residual here is only that IPRED is the prediction rather than the exact
  ## state the solver saw at each internal step.
  expect_lt(max(abs(f$g - ev)), 0.05 * diff(range(f$g)))
})
