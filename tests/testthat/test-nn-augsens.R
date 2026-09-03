## Forward sensitivity through an AUGMENTED (ANODE-style) model: two networks,
## two states, one of them driven by its own network and feeding back into both.
##
## This is the sharp case for R/nnAugment.R.  A single-network model exercises
## none of it: here the variational block has to couple the two states through
## the model Jacobian (`rx_sw_a1_j` and `rx_sw_centr_j` each appear in the
## other's RHS), index weights by a GLOBAL index across both networks, and get
## the per-network forcing `dR/dg * dg/dw` onto the right rows.  Any of those
## being wrong produces a gradient that still looks plausible and trains to the
## wrong place.
##
## Checked against finite differences of the actual log-likelihood, because that
## is the only reference that cannot share a bug with the thing under test.
##
## The model is written out by hand rather than via nn(aug=) -- that route is
## refused for now (it would renumber the user's compartments), and this file is
## about the sensitivity machinery, which is correct and worth keeping pinned.

test_that("the weight gradient through an augmented two-network model matches FD", {
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)

  K <- 2L; H <- 2L; nW <- H * K + 2L * H + 1L
  wn0 <- nnWeightLayout(0L, K, H); wn1 <- nnWeightLayout(1L, K, H)
  nnClearMeta(); on.exit(nnClearMeta(), add = TRUE)
  nnSetMeta(0L, base = 0L, K = K, H = H, act = "tanh")
  nnSetMeta(1L, base = nW, K = K, H = H, act = "tanh")

  ## `a1` relaxes rather than integrates, which is what nn(aug=) generates: the
  ## `- a1` shows up in the variational RHS as (-1 + nn2_d2(1, ...)) and is one
  ## of the things this test would catch if it went missing.
  obj <- paste0(sprintf("param(%s)\n", paste(c(wn0, wn1), collapse = ", ")),
                "g = nn2(0, centr, a1)\n",
                "d/dt(a1) = nn2(1, centr, a1) - a1\n",
                "d/dt(centr) = -g*centr")
  mAug  <- suppressMessages(rxode2::rxode2(nnAugmentModel(obj, H = c("0" = H, "1" = H))))
  mBase <- suppressMessages(rxode2::rxode2(obj))

  set.seed(4)
  w <- stats::rnorm(2L * nW, 0, 0.4)
  ## by NAME: the augmented state is a compartment too, and positional inits
  ## would quietly put the dose in the wrong one
  ic <- c(a1 = 0, centr = 10)
  ot <- c(0.5, 1, 2, 3, 4); dv <- c(8.0, 6.5, 4.2, 3.0, 2.1); sigma <- 0.5
  ev <- rxode2::et(ot)

  setW <- function(w) {
    nnSetWeights(0L, w[seq_len(nW)])
    nnSetWeights(1L, w[nW + seq_len(nW)])
    stats::setNames(w, c(wn0, wn1))
  }
  LL <- function(w) {
    s <- rxode2::rxSolve(mBase, ev, params = setW(w), inits = ic,
                         returnType = "data.frame")
    sum(stats::dnorm(dv, s$centr[match(ot, s$time)], sigma, log = TRUE))
  }

  sA <- rxode2::rxSolve(mAug, ev, params = setW(w), inits = ic,
                        returnType = "data.frame")
  idx <- match(ot, sA$time)
  dLLdf <- (dv - sA$centr[idx]) / sigma^2
  swCols <- sprintf("rx_sw_centr_%d_", seq_len(2L * nW) - 1L)
  ## every weight of BOTH networks gets a variational state for every model
  ## state -- the global indexing, asserted before it is used
  expect_true(all(swCols %in% names(sA)))
  expect_true(all(sprintf("rx_sw_a1_%d_", seq_len(2L * nW) - 1L) %in% names(sA)))

  ana <- vapply(swCols, function(cn) sum(dLLdf * sA[[cn]][idx]), numeric(1))
  h <- 1e-6
  fd <- vapply(seq_along(w), function(j) {
    wp <- w; wp[j] <- wp[j] + h
    wm <- w; wm[j] <- wm[j] - h
    (LL(wp) - LL(wm)) / (2 * h)
  }, numeric(1))

  expect_equal(unname(ana), fd, tolerance = 1e-5)
  ## and the SECOND network -- the one driving the augmented state, whose whole
  ## effect on the prediction is indirect -- must carry real signal, not zeros;
  ## a broken forcing term would still pass a loose all-close check on zeros
  expect_gt(max(abs(ana[nW + seq_len(nW)])), 1)
})

test_that("nn(aug=) is refused, and says why", {
  skip_if_not_installed("rxode2")
  expect_error(nn(centr, aug = 1, num = 1L), "renumber your compartments")
  expect_error(nn(centr, aug = 4, num = 1L), "aug")
  ## aug = 0 is the normal path and must stay untouched
  expect_silent(.r <- nn(centr, aug = 0, num = 1L))
  expect_match(.r$replace, "^nn1\\(0,")
})
