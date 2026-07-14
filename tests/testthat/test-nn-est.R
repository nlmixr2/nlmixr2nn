## est = "nn": the packaged DeepPumas-style estimator.  Data is Michaelis-Menten
## elimination with IIV in Vmax.  nlmixr2(..., est = "nn", nnControl(foceiControl))
## alternates a FOCEi fit of the base latent-input model (residual error + latent
## Omega + per-subject EBEs) with torch weight steps driven by the augmented
## forward-sensitivity solve.  Over the rounds the network learns the (nonlinear)
## population shape while the latent eta recovers the per-subject Vmax variation.
## The returned fit is a standard nlmixr2 fit carrying the trained weights as
## rxForcedPars (self-contained predict/simulate) plus the training trace.

test_that("est='nn' recovers the population NN shape + IIV and is self-contained", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  set.seed(1)
  Vmax <- 2.0; Km <- 3.0; ns <- 8L
  etaTrue <- rnorm(ns, 0, sqrt(0.15))
  mm <- rxode2::rxode2("d/dt(centr) = -(Vmax*exp(eV))*centr/(Km+centr)")
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8, 10),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
           params = c(Vmax = Vmax, Km = Km, eV = etaTrue[id]), returnType = "data.frame")
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
  ctl <- nnControl(nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 8L,
                                            maxInnerIterations = 25L, calcTables = FALSE),
                   rounds = 5L, wSteps = 8L, lr = 0.03, seed = 5L)
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modF, nnCovData(data), est = "nn", control = ctl)))

  ## a standard nlmixr2 fit
  expect_true(inherits(f, "nlmixr2FitCore") || inherits(f, "nlmixr2FitData"))
  expect_true(is.finite(f$objf))

  ## the latent eta recovers the individual Vmax variation (|corr| -> ~1)
  ebes <- setNames(f$eta$eta.nn, f$eta$ID)
  expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.8)

  ## the training trace is attached and the residual error is driven down
  ph <- f$nnParHist
  expect_equal(nrow(ph), 5L)
  expect_lt(ph$add.sd[nrow(ph)], ph$add.sd[1L])

  ## the trained weights ride with the fit as forcedPars (self-contained)
  nW <- 3L * 2L + 2L * 3L + 1L                 # H*K + 2H + 1, H=3 K=2
  expect_equal(length(f$nnWeights), nW)
  expect_equal(length(rxode2::rxForcedPars(f$ui)), nW)
})

test_that("est='nn' injects NN input covariates into training (covariate-NN)", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  rxode2::setRxThreads(1L)

  ## elimination rate depends on a covariate WT (exp(0.5*WT)) with IIV; the NN
  ## input is the covariate WT (nested nn call) + the latent eta.
  set.seed(1)
  ns <- 8L; WTv <- runif(ns, -1, 1); etaTrue <- rnorm(ns, 0, sqrt(0.1))
  tm <- rxode2::rxode2("d/dt(central) = -exp(0.5*WT + eV)*central")
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(tm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8),
           evid = c(1, rep(0, 6)), cmt = 1, amt = c(10, rep(0, 6)), WT = WTv[id]),
           params = c(eV = etaTrue[id]), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), WT = WTv[id],
               dv = c(NA, s$central + rnorm(nrow(s), 0, 0.1)))
  }))

  nnClearMeta()
  modC <- function() {
    ini({ add.sd <- 0.3; eta.nn ~ 0.1 })
    model({ cl <- exp(nn(WT, eta.nn))           # NN input is the covariate WT
            d/dt(central) <- -cl * central
            central ~ add(add.sd) })
  }
  ctl <- nnControl(nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 8L,
                                            maxInnerIterations = 25L, calcTables = FALSE),
                   rounds = 5L, wSteps = 8L, lr = 0.03, seed = 5L)
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modC, nnCovData(data), est = "nn", control = ctl)))

  expect_true(is.finite(f$objf))
  ebes <- setNames(f$eta$eta.nn, f$eta$ID)
  expect_gt(abs(cor(ebes[order(as.integer(names(ebes)))], etaTrue)), 0.6)

  ## the trained network learned the WT effect: cl increases with WT, matching
  ## the true exp(0.5*WT) trend (the covariate genuinely reached the NN)
  nnSetWeights(0L, f$nnWeights)
  clhat <- vapply(c(-0.8, 0, 0.8), function(w) exp(nn2(0L, w, 0)), numeric(1))
  expect_true(clhat[1] < clhat[2] && clhat[2] < clhat[3])
  expect_lt(abs(clhat[3] / clhat[1] - exp(0.5 * 1.6)), 0.6)
})
