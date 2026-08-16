## B5: end-to-end NN-in-ODE training via the forward-sensitivity gradient.
## Each iteration solves the augmented model to get df/dw (rx_sw), assembles the
## analytic dLL/dw with the Gaussian score, injects -dLL/dw into the torch
## optimizer (nnTorchSetGrad) and steps.  The log-likelihood must increase --
## i.e. the ODE forward sensitivity + torch optimizer actually train the weights.

test_that("forward-sensitivity gradient + torch step increases the log-likelihood", {
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  skip_if_no_torch()

  K <- 2L; H <- 1L; nW <- H * K + 2 * H + 1
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = K, H = H, act = "tanh")
  nnTorchInit(0L, K, H, act = "tanh", seed = 11)
  nnTorchOptInit(0L, "adam", 0.05)
  on.exit({ nnClearMeta(); nnTorchFree(0L) }, add = TRUE)

  wnm <- nnWeightLayout(0L, K, H)
  obj <- paste0(sprintf("param(%s)\n", paste(wnm, collapse = ", ")),
                "g = nn2(0, centr, peri)\nd/dt(centr) = -g*centr\nd/dt(peri) = g*centr - k*peri")
  mAug  <- rxode2::rxode2(nnAugmentModel(obj, H = H))
  mBase <- rxode2::rxode2(obj)

  p  <- c(k = 0.3, setNames(rep(0, nW), wnm)); ic <- c(centr = 10, peri = 0)
  ot <- c(0.5, 1, 2, 3, 4); dv <- c(8.0, 6.5, 4.2, 3.0, 2.1); sigma <- 0.5
  ev <- rxode2::et(ot)

  logLik <- function() {
    nnSetWeights(0L, nnTorchWeights(0L))
    s <- rxode2::rxSolve(mBase, ev, params = p, inits = ic)
    f <- s$centr[match(ot, s$time)]
    sum(dnorm(dv, f, sigma, log = TRUE))
  }

  ll0 <- logLik()
  for (iter in 1:15) {
    nnSetWeights(0L, nnTorchWeights(0L))
    sA  <- rxode2::rxSolve(mAug, ev, params = p, inits = ic)
    idx <- match(ot, sA$time)
    f   <- sA$centr[idx]
    dLLdf <- (dv - f) / sigma^2
    dLLdw <- vapply(seq_len(nW) - 1L,
                    function(j) sum(dLLdf * sA[[sprintf("rx_sw_centr_%d_", j)]][idx]),
                    numeric(1))
    nnTorchZeroGrad(0L)
    nnTorchSetGrad(0L, -dLLdw)     # minimize -LL
    nnTorchStep(0L)
  }
  ll1 <- logLik()

  expect_gt(ll1, ll0)             # training improved the fit
})
