## Transparent nlmixr2nn workflow, JOINT (DeepPumas-style) mode:
##
##   nlmixr2(model, data, "focei", foceiControl(...), nn = nnControl(mode = "joint", ...))
##
## Unlike mode = "iter" (a full inner fit each round), joint mode co-optimizes the
## population parameters and the network weights in one interleaved loop -- warm-
## started partial inner steps (outerPerRound outer iterations) + torch weight
## steps -- until both the objective and the weights stabilise.  It recovers the
## same MM+IIV truth.  True interleave needs an inner estimator that exposes
## maxOuterIterations (the FOCEi family); others degrade to the iterative loop.

test_that("joint nn training co-optimizes params + weights and recovers MM+IIV", {
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
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modF, nnCovData(data), "focei",
      nlmixr2est::foceiControl(print = 0L, maxInnerIterations = 25L, calcTables = FALSE),
      nn = nnControl(mode = "joint", rounds = 50L, outerPerRound = 1L, wSteps = 2L,
                     lr = 0.03, tol = 5e-3, seed = 5L))))

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

test_that("joint nn training interleaves a variational inner estimator (ADVI iters knob)", {
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
  ## joint mode with an ADVI inner: interleave uses the `iters` knob -- each round
  ## a warm-started partial variational fit (outerPerRound iters, resuming from the
  ## previous round's ui) + a torch weight step, genuinely co-descending.
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modF, nnCovData(data), "advi",
      nlmixr2est::adviControl(),
      nn = nnControl(mode = "joint", rounds = 6L, outerPerRound = 30L, wSteps = 6L,
                     lr = 0.03, tol = 5e-3, seed = 5L))))

  expect_true(is.finite(f$objf))
  ph <- f$nnParHist
  expect_true(all(c("objf", "wChange", "objfChange") %in% names(ph)))
  expect_true(f$nnRounds == nrow(ph) && nrow(ph) <= 6L)
  ebes <- setNames(f$eta$eta.nn, f$eta$ID)
  expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.7)
})

test_that("nn training warm-starts from a model that already carries trained weights", {
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
  ## first fit produces trained weights baked into the fit ui as forcedPars
  f1 <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modF, nnCovData(data), "focei",
      nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 8L,
                               maxInnerIterations = 25L, calcTables = FALSE),
      nn = nnControl(mode = "iter", rounds = 5L, wSteps = 8L, lr = 0.03, seed = 5L))))
  w1 <- f1$nnWeights

  ## a fit STARTED from f1$finalUi must warm-start from those weights, NOT random
  ## init: .nnExistingWeights reads them and nnTorchSetWeights applies them.  With
  ## a different random seed, a cold start would give a different round-1 network;
  ## the warm start reproduces w1 at round 1 (before any further steps move it).
  warmUi <- rxode2::rxUiDecompress(f1$finalUi)
  aug <- nlmixr2nn:::.nnAugmentFromUi(warmUi)
  wExisting <- nlmixr2nn:::.nnExistingWeights(warmUi, aug)
  expect_equal(length(wExisting), length(w1))
  expect_equal(unname(wExisting), unname(w1), tolerance = 1e-8)

  ## a follow-on fit converges (does not regress) from the warm start
  nnClearMeta()
  f2 <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(warmUi, nnCovData(data), "focei",
      nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 4L,
                               maxInnerIterations = 25L, calcTables = FALSE),
      nn = nnControl(mode = "iter", rounds = 2L, wSteps = 3L, lr = 0.01, seed = 99L))))
  ebes <- setNames(f2$eta$eta.nn, f2$eta$ID)
  expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.7)
})

test_that("nlm-bridge population warm-start seeds the joint fit with a better start", {
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

  modF <- function() {
    ini({ add.sd <- 0.3; eta.nn ~ 0.2 })
    model({ g <- nn(centr, eta.nn, n_hidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  runWs <- function(ws) {
    nnClearMeta()
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modF, nnCovData(data), "focei",
        nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 6L,
                                 maxInnerIterations = 25L, calcTables = FALSE),
        nn = nnControl(mode = "iter", rounds = 5L, wSteps = 6L, lr = 0.03, seed = 5L,
                       warmStart = ws, warmPopIters = 40L))))
    ebes <- setNames(f$eta$eta.nn, f$eta$ID)
    list(r1 = f$nnParHist$rmse[1L], objf = f$objf,
         corr = abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)))
  }
  none <- runWs("none")
  ## nlm is simply an optimizer: the population (nlm-bridge) pre-fit works with any
  ## nlm-family optimizer -- a gradient-based one ("nlm") and a derivative-free one
  ## ("bobyqa") both land the network far closer to the data before the mixed-model
  ## loop (much lower round-1 rmse), with no random effect used, and still recover
  ## meaningful IIV (a strong population pre-fit can absorb some variation, so the
  ## eta correlation is looser than the default-path guarantee, but non-trivial).
  for (ws in c("nlm", "bobyqa")) {
    pop <- runWs(ws)
    expect_lt(pop$r1, none$r1)
    expect_true(is.finite(pop$objf))
    expect_gt(pop$corr, 0.4)
  }
})
