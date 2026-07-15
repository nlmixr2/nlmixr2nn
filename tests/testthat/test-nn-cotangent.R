## nnControl(cotangent = "exact"): the NN weight gradient uses the per-observation
## error-model score dLL/df captured from the inner fit's likelihood-contribution
## hook (the EXACT cotangent for ANY residual model) instead of the closed-form
## additive/proportional-Gaussian one.  On a Gaussian model the two agree; a
## non-add()/prop() model (e.g. lognormal) needs "exact".

test_that("exact cotangent matches the Gaussian closed form on an additive model", {
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
  runCt <- function(ct) {
    nnClearMeta()
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modF, nnCovData(data), "focei",
        nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 6L,
                                 maxInnerIterations = 25L, calcTables = FALSE),
        nn = nnControl(mode = "iter", rounds = 5L, wSteps = 1L, lr = 0.03, seed = 5L,
                       warmStart = "none", cotangent = ct))))
    ebes <- setNames(f$eta$eta.nn, f$eta$ID)
    abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue))
  }
  ## same error model -> the exact cotangent (inner fit's per-obs dLL/df at its
  ## converged f/r) agrees with the Gaussian closed form to numerical precision, so
  ## the IIV recovery matches -- confirming the exact path is wired correctly.
  expect_equal(runCt("exact"), runCt("gaussian"), tolerance = 1e-2)
})

test_that("exact cotangent trains under a non-Gaussian (lognormal) error model", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  set.seed(1); ns <- 8L; etaTrue <- rnorm(ns, 0, sqrt(0.15))
  mm <- rxode2::rxode2("d/dt(centr) = -(2*exp(eV))*centr/(3+centr)")
  ## lognormal measurement noise: dv = centr * exp(N(0, 0.1))
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8, 10),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
           params = c(eV = etaTrue[id]), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr * exp(rnorm(nrow(s), 0, 0.1))))
  }))
  modLN <- function() {
    ini({ lsd <- 0.2; eta.nn ~ 0.2 })
    model({ g <- nn(centr, eta.nn, n_hidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ lnorm(lsd) })
  }
  ## the Gaussian closed form cannot handle a non-add()/prop() model
  nnClearMeta()
  expect_error(
    suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modLN, nnCovData(data), "focei",
        nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 2L,
                                 maxInnerIterations = 10L, calcTables = FALSE),
        nn = nnControl(mode = "iter", rounds = 2L, wSteps = 1L, seed = 5L,
                       warmStart = "none")))),
    "cotangent")

  ## with the exact cotangent it trains and recovers the IIV
  nnClearMeta()
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modLN, nnCovData(data), "focei",
      nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 6L,
                               maxInnerIterations = 25L, calcTables = FALSE),
      nn = nnControl(mode = "iter", rounds = 5L, wSteps = 1L, lr = 0.03, seed = 5L,
                     warmStart = "none", cotangent = "exact"))))
  expect_true(is.finite(f$objf))
  ebes <- setNames(f$eta$eta.nn, f$eta$ID)
  expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.7)
})

test_that("exact cotangent trains under non-FOCEi-inner methods via a FOCEi posthoc", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  set.seed(1); ns <- 6L; etaTrue <- rnorm(ns, 0, sqrt(0.15))
  mm <- rxode2::rxode2("d/dt(centr) = -(2*exp(eV))*centr/(3+centr)")
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8),
           evid = c(1, rep(0, 6)), cmt = 1, amt = c(10, rep(0, 6))),
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
  ## saem's kernel and imp's importance draws do not fire the FOCEi contribution
  ## hook cleanly at the fitted etas; the exact path runs a FOCEi posthoc at the
  ## fit's estimates to capture the per-obs cotangent (+ refined EBEs), so both
  ## recover the IIV.
  ctls <- list(saem = nlmixr2est::saemControl(print = 0L, nBurn = 60L, nEm = 60L, covMethod = ""),
               impmap = nlmixr2est::impmapControl())
  for (est in names(ctls)) {
    nnClearMeta()
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modF, nnCovData(data), est, ctls[[est]],
        nn = nnControl(mode = "iter", rounds = 4L, wSteps = 1L, lr = 0.03, seed = 5L,
                       warmStart = "none", cotangent = "exact"))))
    expect_true(is.finite(f$objf))
    ebes <- setNames(f$eta$eta.nn, f$eta$ID)
    expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.5)
  }
})
