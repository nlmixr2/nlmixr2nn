## nnControl(cotangent = "exact"): the NN weight gradient uses the per-observation
## error-model score dLL/df captured from the inner fit's likelihood-contribution
## hook (the EXACT cotangent for ANY residual model) instead of the closed-form
## additive/proportional-Gaussian one.  On a Gaussian model the two agree; a
## non-add()/prop() model (e.g. lognormal) needs "exact".

test_that("exact cotangent matches the Gaussian closed form on an additive model", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
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
    model({ g <- nn(centr, eta.nn, nHidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  runCt <- function(ct) {
    ## Seed each run so both start from the SAME network.  Without this the two
    ## runs re-parse the model with the RNG already advanced by the first fit, so
    ## they begin at different weights -- and the comparison then measures that
    ## difference rather than the difference between the two cotangent sources.
    set.seed(42)
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modF, nnCovData(data), "focei",
        nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 6L,
                                 maxInnerIterations = 25L, calcTables = FALSE),
        nn = nnControl(mode = "iter", rounds = 5L, wSteps = 1L, lr = 0.03, seed = 5L,
                       warmStart = "none", cotangent = ct))))
    ebes <- setNames(f$eta$eta.nn, f$eta$ID)
    abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue))
  }
  ## Both cotangent sources train the same model to a comparable place.
  ##
  ## Read this for what it is: a WEAK proxy.  It compares two independent
  ## five-round fits, so it cannot establish that the two gradients agree -- it
  ## only catches one path being badly wired.  Measured, the two land about 4%
  ## apart on this model even from an identical starting network, which is within
  ## what two nonlinear mixed-model fits can differ by and tells us nothing about
  ## the gradients themselves.
  ##
  ## The defensible test is a direct comparison of the assembled dLL/dw from the
  ## two sources at one set of weights, checked against a finite difference of the
  ## objective (two agreeing wrong answers must not pass).  That needs the weight
  ## stepper to return its gradient, and is specified for the cotangent work in
  ## stage 2 -- it is the gate on making "exact" the default.
  expect_equal(runCt("exact"), runCt("gaussian"), tolerance = 0.05)
})

test_that("exact cotangent trains under a non-Gaussian (lognormal) error model", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
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
    model({ g <- nn(centr, eta.nn, nHidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ lnorm(lsd) })
  }
  ## A transform-both-sides endpoint has no closed-form Gaussian score, and the
  ## schedule now recognizes that and selects the exact one WITHOUT the user
  ## asking -- which is the whole point of inferring the schedule.  (This used to
  ## be an error instructing the user to pass cotangent="exact".)
  .env <- new.env(parent = emptyenv())
  .env$ui <- suppressWarnings(suppressMessages(rxode2::rxode2(modLN)))
  .env$data <- data
  .env$control <- list(maxOuterIterations = 2L)
  class(.env) <- c("focei", "environment")
  expect_equal(nlmixr2nn:::.nnInferSched(.env, nW = 16L)$cotangent, "exact")

  ## asking for the Gaussian score on such a model IS still an error, because
  ## there the user has stated something that cannot be right
  expect_error(
    nlmixr2nn:::.nnResolveSched(nnControl(cotangent = "gaussian"),
                                nlmixr2nn:::.nnInferSched(.env, nW = 16L)),
    "untransformed additive")

  ## with the exact cotangent it trains and recovers the IIV
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
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
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
    model({ g <- nn(centr, eta.nn, nHidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  ## saem's kernel and imp's importance draws do not fire the FOCEi contribution
  ## hook cleanly at the fitted etas; the exact path runs a FOCEi posthoc at the
  ## fit's estimates to capture the per-obs cotangent (+ refined EBEs), so both
  ## recover the IIV.
  ctls <- list(saem = nlmixr2est::saemControl(print = 0L, nBurn = 60L, nEm = 60L, covMethod = ""),
               impmap = nlmixr2est::impmapControl())
  ## Rounds, per estimator, because they are not interchangeable.  Neither of
  ## these resumes: each round is a FULL fit whose chain restarts, so the loop
  ## converges in rounds rather than within them.  SAEM at 4 rounds reaches an
  ## eta correlation of 0.30 with its rmse still falling; at 12 it reaches 0.89.
  ## That is the estimator needing iterations, not the gradient being wrong --
  ## and it is why the inferred schedule gives non-resuming estimators 30 rounds.
  rounds <- c(saem = 12L, impmap = 4L)
  for (est in names(ctls)) {
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modF, nnCovData(data), est, ctls[[est]],
        nn = nnControl(mode = "iter", rounds = rounds[[est]], wSteps = 1L, lr = 0.03,
                       seed = 5L, warmStart = "none", cotangent = "exact"))))
    expect_true(is.finite(f$objf))
    ebes <- setNames(f$eta$eta.nn, f$eta$ID)
    expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.5)
  }
})
