## Shared test scaffolding.
##
## Two jobs: stop every file from re-inventing the same skip conditions and
## fixtures, and make each test clean up after itself so the suite does not
## depend on file order.
##
## The isolation matters more than it looks.  The compiled registry is keyed by a
## MODEL-LOCAL network id, so every single-network model in the session is
## network 0.  That is safe because each solve binds its own model first -- but a
## test that leaves a torch module or a half-built registry behind can still make
## the *next* test look broken.  `local_nn()` puts everything back.

## --- skips ------------------------------------------------------------------

skip_if_no_torch <- function() {
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable", PACKAGE = "nlmixr2nn")),
                 error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
}

skip_if_no_est <- function() {
  skip_on_cran()
  skip_if_not_installed("nlmixr2est")
}

## --- per-test isolation -----------------------------------------------------

## Save every piece of package state a test can disturb and restore it on exit.
## Returns invisibly; call it at the top of a test that fits, solves, or touches
## the registry.
local_nn <- function(threads = 1L, .env = parent.frame()) {
  e <- getFromNamespace(".nnEnv", "nlmixr2nn")
  old <- list(reg = e$reg,
              torchIds = e$torchIds,
              scales = e$scales,
              training = isTRUE(e$training),
              threads = rxode2::getRxThreads())
  withr::defer({
    ## free any torch module this test created, but not one it inherited
    for (i in setdiff(e$torchIds, old$torchIds)) {
      try(getFromNamespace("nnTorchFree", "nlmixr2nn")(i), silent = TRUE)
    }
    getFromNamespace("nnClearMeta", "nlmixr2nn")()
    assign("reg", old$reg, envir = e)
    assign("torchIds", old$torchIds, envir = e)
    assign("scales", old$scales, envir = e)
    assign("training", old$training, envir = e)
    getFromNamespace(".nnLoaderOff", "nlmixr2nn")()
    rxode2::setRxThreads(old$threads)
    ## the estimation interceptor is global; a test that removed it must not
    ## leave the next fit unclaimed
    try(getFromNamespace(".nnRegisterInterceptor", "nlmixr2nn")(), silent = TRUE)
  }, envir = .env)
  if (!is.null(threads)) rxode2::setRxThreads(threads)
  invisible()
}

## Assert that nothing leaked.  Opt-in per test rather than enforced per file:
## several older tests deliberately drive the torch module directly, and turning
## their leftovers into failures would report a mess rather than a defect.  The
## file-level teardown below still CLEANS UP, which is what actually buys
## order-independence.
expect_nn_isolated <- function() {
  e <- getFromNamespace(".nnEnv", "nlmixr2nn")
  expect_length(e$torchIds, 0L)
  expect_false(isTRUE(e$training))
}

## Leave the package in a clean state whatever a file did, so the next file
## starts from the same place regardless of run order.
nnResetState <- function() {
  e <- getFromNamespace(".nnEnv", "nlmixr2nn")
  for (i in e$torchIds) try(getFromNamespace("nnTorchFree", "nlmixr2nn")(i), silent = TRUE)
  try(getFromNamespace("nnClearMeta", "nlmixr2nn")(), silent = TRUE)
  assign("reg", list(), envir = e)
  assign("scales", list(), envir = e)
  assign("training", FALSE, envir = e)
  try(getFromNamespace(".nnLoaderOff", "nlmixr2nn")(), silent = TRUE)
  try(getFromNamespace(".nnRegisterInterceptor", "nlmixr2nn")(), silent = TRUE)
  invisible()
}

## --- fixtures ---------------------------------------------------------------

## Michaelis-Menten truth with optional between-subject variability on Vmax.
## This is the data-generating model nearly every estimation test used to build
## inline, a dozen times over.
nnSimData <- function(ns = 8L, seed = 1L, iiv = TRUE, sd = 0.1,
                      times = c(0, 0.5, 1, 2, 4, 6, 8, 10), amt = 10) {
  set.seed(seed)
  etaTrue <- if (iiv) stats::rnorm(ns, 0, sqrt(0.15)) else rep(0, ns)
  mm <- rxode2::rxode2("d/dt(centr) = -(2*exp(eV))*centr/(3+centr)")
  d <- do.call(rbind, lapply(seq_len(ns), function(id) {
    s <- rxode2::rxSolve(mm,
           data.frame(id = id, time = times, evid = c(1, rep(0, length(times) - 1L)),
                      cmt = 1, amt = c(amt, rep(0, length(times) - 1L))),
           params = c(eV = etaTrue[id]), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(amt, rep(0, nrow(s))),
               dv = c(NA, s$centr + stats::rnorm(nrow(s), 0, sd)))
  }))
  attr(d, "etaTrue") <- etaTrue
  d
}

## UDE correction with a latent eta as a network input (the headline use case)
nnModUde <- function() {
  ini({ add.sd <- 0.3; eta.nn ~ 0.2 })
  model({
    g <- nn(centr, eta.nn, nHidden = 3L, act = "tanh")
    d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
    centr ~ add(add.sd)
  })
}

## the same system with no between-subject variability (QSP)
nnModQsp <- function() {
  ini({ add.sd <- 0.3 })
  model({
    g <- nn(centr, nHidden = 3L, act = "tanh")
    d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
    centr ~ add(add.sd)
  })
}

## fits are noisy on stdout and irrelevant to what is being asserted
nnFit <- function(...) suppressWarnings(suppressMessages(nlmixr2est::nlmixr2(...)))

## --- file-level teardown ----------------------------------------------------
withr::defer(nnResetState(), teardown_env())
