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
