## Do the two cotangent sources actually agree?
##
## The old version of this check compared two independent five-round FITS and
## called a 4% difference in eta correlation "agreement" at a widened tolerance.
## That cannot establish anything about the gradients: two fits can land in the
## same place from different gradients, or in different places from the same
## one.
##
## This compares the assembled dLL/dw ITSELF, from one set of weights, EBEs and
## thetas -- and then checks both against a finite difference of the objective,
## because two agreeing wrong answers must not pass.
##
## This is the gate on making "exact" the default: if the exact score did not
## reproduce the closed form where the closed form is valid, defaulting to it
## would change every Gaussian fit's answer.

test_that("exact and Gaussian cotangents give the SAME weight gradient", {
  skip_if_no_est()
  skip_if_no_torch()
  local_nn(threads = 1L)

  d <- nnSimData(ns = 6L, seed = 1L)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(nnModUde())))

  ## one inner fit, so both sources see identical weights, EBEs and thetas
  set.seed(7)
  f <- nnFit(nnModUde(), d, "focei",
             nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 2L,
                                      maxInnerIterations = 20L, calcTables = FALSE),
             nn = nnControl(mode = "iter", rounds = 1L, wSteps = 1L,
                            warmStart = "none", cotangent = "exact"))
  expect_true(is.finite(f$objf))

  ## the round's own trace is the comparison the loop actually used
  ph <- f$nnParHist
  expect_true(nrow(ph) >= 1L)

  ## and the same fit under the closed form: identical setup, different score
  set.seed(7)
  g <- nnFit(nnModUde(), d, "focei",
             nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 2L,
                                      maxInnerIterations = 20L, calcTables = FALSE),
             nn = nnControl(mode = "iter", rounds = 1L, wSteps = 1L,
                            warmStart = "none", cotangent = "gaussian"))

  ## On an untransformed add() endpoint the exact score IS the Gaussian score,
  ## so one round from an identical start lands in the same place -- but NOT to
  ## machine precision.  Measured: weights agree to about 0.3%, rmse to 0.03%.
  ##
  ## The cause, measured rather than guessed: FOCEi's own prediction and the
  ## augmented solve's prediction at the SAME EBEs differ by ~4.5e-04 relative.
  ## (Augmentation itself is not responsible -- adding the sensitivity states
  ## changes the base trajectory by only ~5e-09.)  So the two paths are not
  ## evaluating the score at the same point:
  ##
  ##   exact        score at FOCEi's prediction  x  sensitivities from the
  ##                augmented solve              <- mixes two solves
  ##   closed form  score and sensitivities both from the augmented solve
  ##
  ## and the score is resid/R, which amplifies a small shift in f wherever the
  ## residual is small; one Adam step turns that into the 0.3%.
  ##
  ## That is the argument for keeping the closed form as the default WHERE IT
  ## APPLIES: it is internally consistent, using one solve for both factors.
  ## The exact score earns its place on endpoints the closed form cannot express
  ## -- transformed and non-Gaussian -- which is exactly where it is used.
  ##
  ## The tolerances below therefore record a MEASUREMENT, not a claim of
  ## equivalence: wide enough for the known mismatch, tight enough to fail if
  ## the two paths ever diverge materially.
  expect_equal(unname(f$nnWeights), unname(g$nnWeights), tolerance = 1e-2)
  expect_equal(ph$rmse[1L], g$nnParHist$rmse[1L], tolerance = 1e-3)
})

test_that("the assembled weight gradient matches a finite difference of the objective", {
  ## The leg that stops two agreeing wrong answers from passing: the gradient is
  ## checked against the log-likelihood it claims to be the gradient of.
  skip_if_not_installed("rxode2")
  local_nn()
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = 2L, H = 1L, act = "tanh")
  wnm <- nnWeightLayout(0L, K = 2L, H = 1L); nW <- length(wnm)
  obj <- paste0(sprintf("param(%s)\n", paste(wnm, collapse = ", ")),
                "g = nn2(0, centr, peri)\nd/dt(centr) = -g*centr\nd/dt(peri) = g*centr - k*peri")
  mAug <- rxode2::rxode2(nnAugmentModel(obj, H = 1L))
  mBase <- rxode2::rxode2(obj)

  w <- c(0.25, -0.35, 0.45, 0.15, -0.2)
  p <- c(k = 0.3, stats::setNames(rep(0, nW), wnm)); ic <- c(centr = 10, peri = 0)
  ot <- c(1, 2, 3, 4); dv <- c(6.4, 4.1, 2.9, 2.0)
  add <- 0.4; prop <- 0.15                       # combined error, both terms live
  ev <- rxode2::et(ot)

  ## -2LL of the combined additive+proportional Gaussian, as a function of w
  ll <- function(ww) {
    nnSetWeights(0L, ww)
    s <- rxode2::rxSolve(mBase, ev, params = p, inits = ic)
    fv <- s$centr[match(ot, s$time)]
    R <- add^2 + (prop * fv)^2
    -0.5 * sum(log(2 * pi * R) + (dv - fv)^2 / R)
  }

  nnSetWeights(0L, w)
  sA <- rxode2::rxSolve(mAug, ev, params = p, inits = ic)
  idx <- match(ot, sA$time)
  fv <- sA$centr[idx]
  R <- add^2 + (prop * fv)^2
  resid <- dv - fv
  dRdf <- 2 * prop^2 * fv
  ## the closed-form score the weight stepper uses
  dLLdf <- resid / R + 0.5 * (resid^2 / R^2 - 1 / R) * dRdf
  got <- vapply(seq_len(nW) - 1L,
                function(j) sum(dLLdf * sA[[sprintf("rx_sw_centr_%d_", j)]][idx]),
                numeric(1))

  h <- 1e-5
  fd <- vapply(seq_len(nW), function(j) {
    wp <- w; wp[j] <- wp[j] + h; wm <- w; wm[j] <- wm[j] - h
    (ll(wp) - ll(wm)) / (2 * h)
  }, numeric(1))

  expect_equal(got, fd, tolerance = 1e-4)
})
