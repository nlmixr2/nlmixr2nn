## est = "nn": the JOINT (DeepPumas-style) estimator.  Unlike est = "nnIter"
## (a full inner fit each round), it co-optimizes the population parameters and
## the network weights in one interleaved loop -- warm-started partial inner
## steps (outerPerRound outer iterations) + torch weight steps -- until both the
## objective and the weights stabilise.  It recovers the same MM+IIV truth.

test_that("est='nn' (joint) co-optimizes params + weights and recovers MM+IIV", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  set.seed(1); ns <- 8L; etaTrue <- rnorm(ns, 0, sqrt(0.15))
  mm <- rxode2::rxode2("d/dt(centr) = -(2*exp(eV))*centr/(3+centr)")
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8, 10),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
           params = c(eV = etaTrue[id]), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr + rnorm(nrow(s), 0, 0.1)))
  }))

  nnClearMeta()
  modF <- function() {
    ini({ add.sd <- 0.3; eta.nn ~ 0.2 })
    model({ g <- nn(centr, eta.nn, n_hidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  ctl <- nnControl(nlmixr2est::foceiControl(print = 0L, maxInnerIterations = 25L,
                                            calcTables = FALSE),
                   maxRounds = 50L, outerPerRound = 1L, wSteps = 2L, lr = 0.03,
                   tol = 5e-3, seed = 5L)
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modF, nnCovData(data), est = "nn", control = ctl)))

  expect_true(inherits(f, "nlmixr2FitCore") || inherits(f, "nlmixr2FitData"))
  expect_true(is.finite(f$objf))

  ## the joint objective descends (co-optimization) and the IIV is recovered
  ph <- f$nnParHist
  expect_true(all(c("objf", "wChange", "objfChange") %in% names(ph)))
  expect_lt(ph$objf[nrow(ph)], ph$objf[1L])          # objective co-descended
  expect_true(nrow(ph) <= 50L && f$nnRounds == nrow(ph))
  ebes <- setNames(f$eta$eta.nn, f$eta$ID)
  expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.7)

  ## trained weights carried on the fit
  nW <- 3L * 2L + 2L * 3L + 1L
  expect_equal(length(rxode2::rxForcedPars(f$ui)), nW)
})
