## End-to-end: an eta-free nn() model (a population UDE -- no latent random effect)
## trains transparently under FOCEI.  The covariate-declared weight block (loader-
## injected) assembles cleanly through FOCEI's theta expansion -- the fix that
## unblocked the native path (param()-declared weights were mangled) -- and the
## interceptor runs a population weight fit (no per-subject EBEs).

test_that("an eta-free nn() model assembles and trains under FOCEI (population UDE)", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  set.seed(1)
  d <- do.call(rbind, lapply(1:6, function(id) {
    t <- c(1, 2, 4, 6, 8)
    data.frame(ID = id, TIME = c(0, t), EVID = c(1, rep(0, 5)),
               AMT = c(10, rep(0, 5)), DV = c(NA, 9 * exp(-0.25 * t) + rnorm(5, 0, 0.2)),
               CMT = 1)
  }))

  mod <- function() {
    ini({ add.sd <- 0.5 })                        # no eta: a population UDE
    model({
      g <- nn(centr, nHidden = 3L, act = "tanh")
      d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr   # bounded rate (stable)
      centr ~ add(add.sd)
    })
  }
  ## transparent workflow: standard est + a small training schedule.  With no
  ## latent eta the interceptor trains the population network (no EBEs).
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(mod, nnCovData(d), "focei",
      nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 3L,
                               maxInnerIterations = 4L, calcTables = FALSE),
      nn = nnControl(mode = "iter", rounds = 3L, wSteps = 4L, lr = 0.03, seed = 1L))))

  expect_true(is.finite(f$objf))                 # FOCEI assembled + ran the nn() model
  expect_true(nrow(f$nnParHist) >= 1L)           # the population network trained
  nW <- 3L * 1L + 2L * 3L + 1L                    # H*K + 2H + 1, H=3 K=1
  expect_equal(length(f$nnWeights), nW)
})

test_that("a QSP no-BSV NN model fits with the nlm family (population weight fit)", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  ## QSP: ONE true system (no between-subject variability), noisy observations.
  set.seed(1)
  mm <- rxode2::rxode2("d/dt(centr) = -(2)*centr/(3+centr)")   # true MM rate, no IIV
  data <- do.call(rbind, lapply(1:6, function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8, 10),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr + rnorm(nrow(s), 0, 0.1)))
  }))
  ## eta-free NN model: learn the (nonlinear) concentration-dependent rate, no eta
  modQSP <- function() {
    ini({ add.sd <- 0.3 })
    model({ g <- nn(centr, nHidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  ## the true (noise-free) trajectory for a single dosing schedule
  dose1 <- data.frame(id = 1, time = c(0, 0.5, 1, 2, 4, 6, 8, 10),
                      evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7)))
  trueTraj <- rxode2::rxSolve(mm, dose1, returnType = "data.frame")
  trueTraj <- trueTraj[trueTraj$time > 0, "centr"]

  ## nlm is simply an optimizer: a gradient-based one ("nlm") and a derivative-free
  ## one ("bobyqa") both fit the population weights with NO random effect.  The nlm
  ## family now reads the injected weights natively (nlmixr2est's nlm model declares
  ## them as covariates), so the fit is a real nlm fit -- and, self-contained via
  ## rxForcedPars, its solved trajectory reproduces the true (concentration-
  ## dependent, non-degenerate) dynamics.
  for (est in c("nlm", "bobyqa")) {
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modQSP, nnCovData(data), est,
        nlmixr2est::getValidNlmixrControl(NULL, est),
        nn = nnControl(rounds = 60L, seed = 5L))))
    expect_true(is.finite(f$objf))
    expect_equal(length(f$nnWeights), 3L * 1L + 2L * 3L + 1L)
    ## the fit is self-contained: the trained weights ride along as forcedPars
    expect_equal(length(rxode2::rxForcedPars(f$ui)), 3L * 1L + 2L * 3L + 1L)
    ## user-facing check: solving the fitted model reproduces the true dynamics
    ## (a degenerate flat-rate network could not) -- this exercises the forcedPars
    ## weight injection, not the internal loader-state.
    sim <- rxode2::rxSolve(f$finalUi, nnCovData(dose1), returnType = "data.frame")
    simObs <- sim[sim$time > 0, "centr"]
    expect_gt(diff(range(simObs)), 1)                    # NOT a flat/degenerate fit
    expect_gt(cor(simObs, trueTraj), 0.99)               # reproduces the true curve
  }
})

test_that("multiple nn() networks in one model train jointly", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ try(nnTorchFree(0L), silent = TRUE); try(nnTorchFree(1L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  ## a two-compartment system with TWO networks: ka = f(WT), cl = f(AGE); no BSV
  set.seed(1); ns <- 8L; WTv <- runif(ns, -1, 1); AGEv <- runif(ns, -1, 1)
  tm <- rxode2::rxode2(paste0("d/dt(depot) = -exp(0.4*WT)*depot; ",
                              "d/dt(central) = exp(0.4*WT)*depot - exp(0.3*AGE)*central"))
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(tm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8),
           evid = c(1, rep(0, 6)), cmt = 1, amt = c(10, rep(0, 6)), WT = WTv[id], AGE = AGEv[id]),
           returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), WT = WTv[id], AGE = AGEv[id],
               dv = c(NA, s$central + rnorm(nrow(s), 0, 0.1)))
  }))

  mod2 <- function() {
    ini({ add.sd <- 0.3 })
    model({ ka <- exp(nn(WT, nHidden = 3L, act = "tanh"))    # network 0: WT -> ka
            cl <- exp(nn(AGE, nHidden = 3L, act = "tanh"))   # network 1: AGE -> cl
            d/dt(depot) <- -ka * depot
            d/dt(central) <- ka * depot - cl * central
            central ~ add(add.sd) })
  }
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(mod2, nnCovData(data), "bobyqa", nn = nnControl(rounds = 80L, seed = 5L))))

  expect_true(is.finite(f$objf))
  ## both networks' weights are carried (10 + 10) and baked as forcedPars
  nW1 <- 3L * 1L + 2L * 3L + 1L
  expect_equal(length(f$nnWeights), 2L * nW1)
  expect_equal(length(rxode2::rxForcedPars(f$ui)), 2L * nW1)
  ## both learned relationships reproduce the true two-compartment dynamics
  d1 <- nnCovData(data.frame(time = c(0, 0.5, 1, 2, 4, 6, 8), evid = c(1, rep(0, 6)),
                             cmt = 1, amt = c(10, rep(0, 6)), WT = 0.6, AGE = -0.5))
  sim <- rxode2::rxSolve(f$finalUi, d1, returnType = "data.frame")
  tru <- rxode2::rxSolve(tm, d1, returnType = "data.frame")
  expect_gt(cor(sim$central[sim$time > 0], tru$central[tru$time > 0]), 0.99)
})

test_that("a K=3 covariate network (QSP, no BSV) fits and reproduces the truth", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  set.seed(1); ns <- 8L
  WTv <- runif(ns, -1, 1); AGEv <- runif(ns, -1, 1); SEXv <- sample(c(-1, 1), ns, TRUE)
  tm <- rxode2::rxode2("d/dt(central) = -exp(0.4*WT + 0.3*AGE + 0.2*SEX)*central")
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(tm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8),
           evid = c(1, rep(0, 6)), cmt = 1, amt = c(10, rep(0, 6)),
           WT = WTv[id], AGE = AGEv[id], SEX = SEXv[id]), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), WT = WTv[id], AGE = AGEv[id], SEX = SEXv[id],
               dv = c(NA, s$central + rnorm(nrow(s), 0, 0.1)))
  }))

  mod3 <- function() {
    ini({ add.sd <- 0.3 })
    model({ cl <- exp(nn(WT, AGE, SEX, nHidden = 4L, act = "tanh"))   # K = 3 inputs
            d/dt(central) <- -cl * central
            central ~ add(add.sd) })
  }
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(mod3, nnCovData(data), "bobyqa", nn = nnControl(rounds = 60L, seed = 5L))))

  expect_true(is.finite(f$objf))
  expect_equal(length(f$nnWeights), 4L * 3L + 2L * 4L + 1L)          # H*K + 2H + 1
  d1 <- nnCovData(data.frame(time = c(0, 0.5, 1, 2, 4, 6, 8), evid = c(1, rep(0, 6)),
                             cmt = 1, amt = c(10, rep(0, 6)), WT = 0.6, AGE = -0.4, SEX = 1))
  sim <- rxode2::rxSolve(f$finalUi, d1, returnType = "data.frame")
  tru <- rxode2::rxSolve(tm, d1, returnType = "data.frame")
  expect_gt(cor(sim$central[sim$time > 0], tru$central[tru$time > 0]), 0.99)
})
