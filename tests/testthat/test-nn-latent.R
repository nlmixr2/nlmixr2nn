## DeepPumas-style latent-input random effect: instead of putting inter-individual
## variability on the network WEIGHTS (nni, which is over-parameterized and needs a
## finite-difference eta sensitivity), feed a small per-subject latent eta as an
## INPUT to a POPULATION network: g <- nn(state, eta.latent).  The eta is then a
## normal (visible) nlmixr2 eta, so its inner FOCEi sensitivity d(g)/d(eta) is the
## network's analytic input derivative (nn<K>_d<j>) -- exact, no finite differences,
## no etaFD directive -- and it is strongly identifiable.

test_that("a latent-input NN random effect fits with analytic (non-FD) sensitivity", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  nnClearMeta(); on.exit(nnClearMeta(), add = TRUE)

  set.seed(42)
  d <- do.call(rbind, lapply(1:16, function(id) {
    t <- c(0.25, 0.5, 1, 2, 4, 6, 8, 12)
    k <- 0.25 * exp(rnorm(1, 0, 0.4))               # per-subject elimination IIV
    data.frame(ID = id, TIME = c(0, t), EVID = c(1, rep(0, length(t))),
               AMT = c(10, rep(0, length(t))),
               DV = c(NA, 9 * exp(-k * t) + rnorm(length(t), 0, 0.15)), CMT = 1)
  }))

  mod <- function() {
    ini({ tk <- 0.25; add.sd <- 0.3; eta.nn ~ 0.3 })
    model({
      g <- nn(centr, eta.nn, n_hidden = 4L, act = "tanh")   # eta.nn is a latent input
      d/dt(centr) <- -(tk + 0.2 * tanh(g)) * centr
      centr ~ add(add.sd)
    })
  }
  ui <- rxode2::rxode2(mod)

  ## eta.nn is recognized as an eta (not a covariate); the weights are covariates.
  expect_true("eta.nn" %in% ui$eta)
  expect_false("eta.nn" %in% ui$allCovs)

  d <- nnCovData(d)
  nnUpdate(ui)
  set.seed(3)
  nnSetWeights(0L, rnorm(length(.nnEnv$reg[["0"]]$weights), 0, 0.5))

  f <- suppressWarnings(suppressMessages(nlmixr2est::nlmixr2(mod, d, est = "focei",
    control = nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 20L,
                                       maxInnerIterations = 40L, calcTables = FALSE))))

  expect_true(is.finite(f$objf))
  ## the latent eta is genuinely identifiable: its EBEs are non-degenerate (they move
  ## per subject) -- the exact opposite of per-weight nni, whose weight-eta EBEs stick
  ## near 0 with no gradient.  This EBE spread is the robust identifiability signal;
  ## the estimated Omega is a sane non-trivial variance.
  expect_gt(diff(range(f$eta$eta.nn)), 1.0)
  expect_gt(unname(f$omega[1, 1]), 0.05)
})
