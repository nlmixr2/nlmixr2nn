## Weight persistence across save/reload: a fit carries its trained NN weights
## (rxForcedPars) AND shapes (nnMeta) on the ui, and the rxode2 ui-prep hook
## (.nnRehydrate) rebuilds the transient C/R registries from them -- so a fit
## saved to RDS and reloaded in a fresh session still solves/predicts.  A fresh
## session is emulated in-process with nnClearMeta() + an empty registry.

test_that("a fit persists nn weights + shapes and rehydrates them on solve", {
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
    model({ g <- nn(centr, eta.nn, n_hidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  fit <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(modF, nnCovData(data), "focei",
      nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 3L,
                               maxInnerIterations = 15L, calcTables = FALSE),
      nn = nnControl(mode = "iter", rounds = 3L, wSteps = 1L, lr = 0.03, seed = 5L,
                     warmStart = "none"))))

  ui <- rxode2::rxUiDecompress(fit$finalUi)
  ## both the weights and the shapes ride on the ui (serialize with saveRDS)
  expect_true(exists("nnMeta", envir = ui, inherits = FALSE))
  fp <- rxode2::rxForcedPars(ui)
  expect_false(is.null(fp))
  expect_true(length(fp) > 0L)

  ## in-session reference: plain ui solve at eta = 0 (weight cols supplied as 0,
  ## forcedPars overrides them with the trained values)
  ev <- data.frame(id = 1, time = c(0, 0.5, 1, 2, 4, 6, 8),
                   evid = c(1, rep(0, 6)), cmt = 1, amt = c(10, rep(0, 6)))
  pars <- c(add.sd = fit$theta[["add.sd"]], eta.nn = 0,
            stats::setNames(rep(0, length(fp)), names(fp)))
  ref <- rxode2::rxSolve(ui, ev, params = pars, returnType = "data.frame")$centr
  expect_false(anyNA(ref))

  ## emulate a fresh session: forget the transient C + R registries
  .nnEnv <- get(".nnEnv", envir = asNamespace("nlmixr2nn"))
  .savedReg <- .nnEnv$reg
  assign("reg", list(), envir = .nnEnv)
  on.exit(assign("reg", .savedReg, envir = .nnEnv), add = TRUE)

  ## the ui-prep hook (fired by rxSolve.rxUi) must rebuild everything -> exact match
  reload <- rxode2::rxSolve(ui, ev, params = pars, returnType = "data.frame")$centr
  expect_false(anyNA(reload))
  expect_equal(reload, ref, tolerance = 1e-10)
  ## and the R-side registry is restored (nnCovData / predict work again)
  expect_true(length(.nnEnv$reg) >= 1L)

  ## negative control: without the hook and with a cold registry, nn() cannot
  ## stride the weight block, so the solve is all NA -- proving the hook does the
  ## essential rehydration (forcedPars alone is not enough).
  ## nnClearMeta() here is not isolation boilerplate -- it is how this negative
  ## control is CONSTRUCTED.  The C registry still holds the binding the previous
  ## solve installed, so without clearing it the network can still stride its
  ## weights and the control proves nothing.
  assign("reg", list(), envir = .nnEnv)
  nnClearMeta()
  rxode2::rxRemoveUiPrep("nlmixr2nn:rehydrate")
  on.exit(rxode2::rxRegisterUiPrep("nlmixr2nn:rehydrate",
            get(".nnRehydrate", envir = asNamespace("nlmixr2nn"))), add = TRUE)
  dead <- suppressWarnings(
    rxode2::rxSolve(ui, ev, params = pars, returnType = "data.frame")$centr)
  expect_true(anyNA(dead))
})
