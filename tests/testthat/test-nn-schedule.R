## The inferred training schedule.
##
## These tests do no fitting and need no torch: the whole point of inferring the
## schedule from predicates is that the decision is made before any solve, so it
## can be checked directly.

## a fake estimation environment, shaped like the one the interceptor receives
.schedEnv <- function(ui, est = "focei", control = list(maxOuterIterations = 5L),
                      data = NULL) {
  if (is.null(data)) {
    data <- data.frame(id = 1, time = c(0, 1, 2, 4), evid = c(1, 0, 0, 0),
                       amt = c(10, 0, 0, 0), cmt = 1, dv = c(NA, 8, 6, 4))
  }
  .e <- new.env(parent = emptyenv())
  .e$ui <- ui
  .e$data <- data
  .e$control <- control
  class(.e) <- c(est, "environment")
  .e
}

.schedEtaMod <- function() {
  ini({ add.sd <- 0.3; eta.nn ~ 0.2 })
  model({
    g <- nn(centr, eta.nn, n_hidden = 3L, act = "tanh")
    d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
    centr ~ add(add.sd)
  })
}

.schedPopMod <- function() {
  ini({ add.sd <- 0.3 })
  model({
    g <- nn(centr, n_hidden = 3L, act = "tanh")
    d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
    centr ~ add(add.sd)
  })
}

.schedLnormMod <- function() {
  ini({ lsd <- 0.2; eta.nn ~ 0.2 })
  model({
    g <- nn(centr, eta.nn, n_hidden = 3L, act = "tanh")
    d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
    centr ~ lnorm(lsd)
  })
}

test_that("a resumable estimator with a random effect interleaves", {
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  s <- .nnInferSched(.schedEnv(ui, "focei", list(maxOuterIterations = 5L)), nW = 16L)
  expect_equal(s$mode, "joint")
  expect_equal(s$outerPerRound, 1L)
  expect_equal(s$rounds, 200L)
})

test_that("a variational estimator interleaves in bigger steps", {
  ## `iters` is not an outer iteration -- one of them is not a step
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  s <- .nnInferSched(.schedEnv(ui, "emvi", list(iters = 100L)), nW = 16L)
  expect_equal(s$mode, "joint")
  expect_equal(s$outerPerRound, 30L)
})

test_that("a non-resuming estimator falls back to full fits, and far fewer", {
  ## SAEM's gain sequence restarts every call, so partial fits do not resume;
  ## 200 FULL mixed-effects fits is not a default anyone can afford
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  s <- .nnInferSched(.schedEnv(ui, "saem", list(maxOuterIterations = 5L)), nW = 16L)
  expect_equal(s$mode, "iter")
  expect_equal(s$rounds, 30L)
})

test_that("a population model with no random effect uses the cheap loop", {
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedPopMod)))
  s <- .nnInferSched(.schedEnv(ui, "focei", list(maxOuterIterations = 5L)), nW = 16L)
  expect_equal(s$mode, "iter")
  expect_equal(s$rounds, 60L)
})

test_that("an nlm-family estimator scales its iteration cap with the network", {
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedPopMod)))
  s <- .nnInferSched(.schedEnv(ui, "bobyqa", list()), nW = 16L)
  expect_equal(s$rounds, 80L)             # 5 * nW, inside [50, 500]
  expect_equal(s$warmStart, "bobyqa")
  ## and the cap is bounded at both ends
  expect_equal(.nnInferSched(.schedEnv(ui, "bobyqa", list()), nW = 2L)$rounds, 50L)
  expect_equal(.nnInferSched(.schedEnv(ui, "bobyqa", list()), nW = 500L)$rounds, 500L)
})

test_that("an nlm-family estimator on a model with a random effect is an error", {
  ## fitting the weights at eta = 0 and then materializing Omega with focei
  ## yields a fit whose weights and Omega come from different objectives
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  expect_error(.nnInferSched(.schedEnv(ui, "bobyqa", list()), nW = 16L),
               "population-only optimizer")
})

test_that("the cotangent source follows the error model, with no user flag", {
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  s <- .nnInferSched(.schedEnv(ui), nW = 16L)
  expect_equal(s$cotangent, "gaussian")   # untransformed add() -> closed form
  expect_equal(s$wSteps, 4L)

  ## a transform-both-sides endpoint has no closed-form score, so the exact one
  ## is selected automatically -- this used to be an error telling the user to
  ## pass a flag
  uiL <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedLnormMod)))
  sL <- .nnInferSched(.schedEnv(uiL), nW = 16L)
  expect_equal(sL$cotangent, "exact")
  expect_equal(sL$wSteps, 1L)             # the captured score is stale after one step
})

test_that("a model carrying trained weights is treated as a warm start", {
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  s <- .nnInferSched(.schedEnv(ui), nW = 16L, hasTrained = TRUE)
  expect_equal(s$warmStart, "none")       # the trained weights ARE the warm start
  expect_lt(s$lr, .nnInferSched(.schedEnv(ui), nW = 16L, hasTrained = FALSE)$lr)
})

test_that("multiple endpoints are refused with a message that names them", {
  m <- function() {
    ini({ add.sd <- 0.3; add.sd2 <- 0.2 })
    model({
      g <- nn(centr, n_hidden = 3L)
      d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
      eff <- centr * 2
      centr ~ add(add.sd)
      eff ~ add(add.sd2)
    })
  }
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  expect_error(.nnInferSched(.schedEnv(ui), nW = 16L), "single endpoint")
})

test_that("an explicit setting wins, and a wrong one is refused", {
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  inf <- .nnInferSched(.schedEnv(ui), nW = 16L)

  ## quality choice: honoured
  got <- .nnResolveSched(nnControl(rounds = 7L, lr = 0.5), inf)
  expect_equal(got$rounds, 7L)
  expect_equal(got$lr, 0.5)
  expect_equal(got$mode, inf$mode)        # untouched fields stay inferred

  ## efficiency conflict: message, then degrade
  uiS <- ui
  infS <- .nnInferSched(.schedEnv(uiS, "saem", list(maxOuterIterations = 5L)), nW = 16L)
  expect_message(.nnResolveSched(nnControl(mode = "joint"), infS), "no resumable")

  ## correctness conflict: error
  uiL <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedLnormMod)))
  infL <- .nnInferSched(.schedEnv(uiL), nW = 16L)
  expect_error(.nnResolveSched(nnControl(cotangent = "gaussian"), infL),
               "untransformed additive")
})

test_that("no nnControl() at all is the same as an empty one", {
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.schedEtaMod)))
  inf <- .nnInferSched(.schedEnv(ui), nW = 16L)
  expect_equal(.nnResolveSched(NULL, inf), inf)
  expect_equal(.nnResolveSched(nnControl(), inf), inf)
})
