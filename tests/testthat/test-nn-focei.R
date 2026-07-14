## End-to-end: an eta-free nn() model (a population UDE -- no latent random effect)
## trains transparently under FOCEI.  The covariate-declared weight block (loader-
## injected) assembles cleanly through FOCEI's theta expansion -- the fix that
## unblocked the native path (param()-declared weights were mangled) -- and the
## interceptor runs a population weight fit (no per-subject EBEs).

test_that("an eta-free nn() model assembles and trains under FOCEI (population UDE)", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)
  nnClearMeta()

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
      g <- nn(centr, n_hidden = 3L, act = "tanh")
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
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
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
    model({ g <- nn(centr, n_hidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  trueRate <- function(cc) 2 / (3 + cc)
  cs <- c(1, 3, 6, 9)

  ## nlm is simply an optimizer: a gradient-based one ("nlm") and a derivative-free
  ## one ("bobyqa") both fit the population weights with NO random effect, and the
  ## learned rate is genuinely concentration-dependent (not a degenerate flat net)
  ## and tracks the truth.
  for (est in c("nlm", "bobyqa")) {
    nnClearMeta()
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(modQSP, nnCovData(data), est,
        nlmixr2est::getValidNlmixrControl(NULL, est),
        nn = nnControl(rounds = 60L, seed = 5L))))
    expect_true(is.finite(f$objf))
    expect_equal(length(f$nnWeights), 3L * 1L + 2L * 3L + 1L)
    nnSetWeights(0L, f$nnWeights)
    rateHat <- 1 / (1 + exp(-vapply(cs, function(cc) nn1(0L, cc), numeric(1))))
    expect_gt(diff(range(rateHat)), 0.1)                 # NOT a flat network
    expect_gt(cor(rateHat, trueRate(cs)), 0.8)           # tracks the true rate
    ## the fit is self-contained: the trained weights ride along as forcedPars
    expect_equal(length(rxode2::rxForcedPars(f$ui)), 3L * 1L + 2L * 3L + 1L)
  }
})
