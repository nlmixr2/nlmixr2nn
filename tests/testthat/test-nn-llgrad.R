## B4: log-likelihood gradient wrt NN weights.
## dLL/dw_j = sum_obs (dLL/df_obs) * (df_obs/dw_j), where df_obs/dw_j = rx_sw at
## the observation (the augmented forward sensitivity) and dLL/df is the error
## model's score.  During estimation dLL/df comes from the contribution bundle's
## obs hook; here we use a plain Gaussian and validate the assembled dLL/dw
## against a central finite difference of the total log-likelihood.

test_that("dLL/dw from the augmented solve matches FD of the Gaussian log-likelihood", {
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = 2L, H = 1L, act = "tanh")
  on.exit(nnClearMeta(), add = TRUE)
  wnm <- nnWeightLayout(0L, K = 2L, H = 1L); nW <- length(wnm)
  obj <- paste0(sprintf("param(%s)\n", paste(wnm, collapse = ", ")),
                "g = nn2(0, centr, peri)\nd/dt(centr) = -g*centr\nd/dt(peri) = g*centr - k*peri")
  aug   <- nnAugmentModel(obj, H = 1L)
  mAug  <- rxode2::rxode2(aug)
  mBase <- rxode2::rxode2(obj)

  w   <- c(0.3, -0.4, 0.5, 0.2, -0.1)
  p   <- c(k = 0.3, setNames(rep(0, nW), wnm)); ic <- c(centr = 10, peri = 0)
  ot  <- c(1, 2, 3, 4); dv <- c(6.5, 4.2, 3.0, 2.1); sigma <- 0.5
  ev  <- rxode2::et(ot)

  ll <- function(ww) {
    nnSetWeights(0L, ww)
    s <- rxode2::rxSolve(mBase, ev, params = p, inits = ic)
    f <- s$centr[match(ot, s$time)]
    sum(dnorm(dv, f, sigma, log = TRUE))
  }

  ## analytic dLL/dw from one augmented solve
  nnSetWeights(0L, w)
  sA  <- rxode2::rxSolve(mAug, ev, params = p, inits = ic)
  idx <- match(ot, sA$time)
  f   <- sA$centr[idx]
  dLLdf <- (dv - f) / sigma^2                       # Gaussian score wrt f
  dLLdw <- vapply(seq_len(nW) - 1L,
                  function(j) sum(dLLdf * sA[[sprintf("rx_sw_centr_%d_", j)]][idx]),
                  numeric(1))

  ## central finite difference of the log-likelihood
  h  <- 1e-5
  fd <- vapply(seq_len(nW), function(j) {
    wp <- w; wp[j] <- wp[j] + h; wm <- w; wm[j] <- wm[j] - h
    (ll(wp) - ll(wm)) / (2 * h)
  }, numeric(1))

  expect_equal(dLLdw, fd, tolerance = 1e-4)
})

## The transform-both-sides case, which is where the chain used to be wrong.
##
## nlmixr2est's likelihood hook reports d(LL)/d(f) with f the TRANSFORMED
## prediction (`rx_pred_`, after TBS), while the augmented solve's rx_sw are
## sensitivities of the NATURAL-scale state.  Multiplying one by the other skips
## the transform's own derivative, so for `centr ~ lnorm(sd)` the assembled
## gradient was short by a factor of 1/f at every observation.
##
## It still trained -- a positive per-observation rescaling is usually still
## uphill -- which is exactly why nothing caught it.  Only a finite difference
## of the real log-likelihood does.
test_that("dLL/dw is correct for a transform-both-sides (lnorm) endpoint", {
  skip_if_not_installed("rxode2")
  local_nn()
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = 2L, H = 1L, act = "tanh")
  wnm <- nnWeightLayout(0L, K = 2L, H = 1L); nW <- length(wnm)
  obj <- paste0(sprintf("param(%s)\n", paste(wnm, collapse = ", ")),
                "g = nn2(0, centr, peri)\nd/dt(centr) = -g*centr\nd/dt(peri) = g*centr - k*peri")
  mAug  <- rxode2::rxode2(nnAugmentModel(obj, H = 1L))
  mBase <- rxode2::rxode2(obj)

  w   <- c(0.3, -0.4, 0.5, 0.2, -0.1)
  p   <- c(k = 0.3, stats::setNames(rep(0, nW), wnm)); ic <- c(centr = 10, peri = 0)
  ot  <- c(1, 2, 3, 4); dv <- c(6.5, 4.2, 3.0, 2.1); sigma <- 0.5
  ev  <- rxode2::et(ot)

  ## lognormal (transform-both-sides): both sides on the log scale
  ll <- function(ww) {
    nnSetWeights(0L, ww)
    s <- rxode2::rxSolve(mBase, ev, params = p, inits = ic)
    f <- s$centr[match(ot, s$time)]
    sum(stats::dnorm(log(dv), log(f), sigma, log = TRUE))
  }

  nnSetWeights(0L, w)
  sA  <- rxode2::rxSolve(mAug, ev, params = p, inits = ic)
  idx <- match(ot, sA$time)
  f   <- sA$centr[idx]

  ## the score the likelihood hook reports: wrt the TRANSFORMED prediction
  dLLdfTrans <- (log(dv) - log(f)) / sigma^2
  ## ...chained to the natural scale by the transform's own derivative
  ep <- list(transform = "lnorm", lambda = 1, trLow = 0, trHi = 1)
  dLLdf <- dLLdfTrans * .nnTransformJac(ep, f)

  dLLdw <- vapply(seq_len(nW) - 1L,
                  function(j) sum(dLLdf * sA[[sprintf("rx_sw_centr_%d_", j)]][idx]),
                  numeric(1))

  h  <- 1e-5
  fd <- vapply(seq_len(nW), function(j) {
    wp <- w; wp[j] <- wp[j] + h; wm <- w; wm[j] <- wm[j] - h
    (ll(wp) - ll(wm)) / (2 * h)
  }, numeric(1))

  expect_equal(dLLdw, fd, tolerance = 1e-4)

  ## and the uncorrected chain -- what the code did before -- is genuinely wrong,
  ## so this test cannot pass for the wrong reason
  bad <- vapply(seq_len(nW) - 1L,
                function(j) sum(dLLdfTrans * sA[[sprintf("rx_sw_centr_%d_", j)]][idx]),
                numeric(1))
  expect_false(isTRUE(all.equal(bad, fd, tolerance = 1e-4)))
})
