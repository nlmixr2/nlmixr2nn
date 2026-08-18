## A prediction fed by the network DIRECTLY, not through an ODE state.
##
## The weight sensitivity of a prediction has two parts:
##
##   through the states   sum_s d(pred)/d(s) * ds/dw
##   directly             d(pred)/dg * dg/dw
##
## Only the first was assembled.  For the usual model -- the network inside a
## d/dt() -- the direct part is zero, so nothing noticed.  For a model whose
## prediction IS the network, e.g. `y <- nn(centr)`, the state route carries
## none of the effect: the gradient came out exactly zero, so the network never
## moved and the fit reported success having learned nothing.

test_that("the augmented model emits the direct term when the prediction needs it", {
  skip_if_not_installed("rxode2")
  local_nn()
  m <- function() {
    ini({ lk <- -1; add.sd <- 0.3 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr
      y <- nn(centr, n_hidden = 2L, act = "tanh")
      y ~ add(add.sd)
    })
  }
  set.seed(4)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
  ## d(pred)/dg * dg/dw appears as an nnWg call in the prediction sensitivity
  expect_true(grepl("nnWg", paste(aug$text, collapse = " ")))
})

test_that("that gradient matches a finite difference, where it used to be zero", {
  skip_if_not_installed("rxode2")
  local_nn()
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = 1L, H = 2L, act = "tanh")
  wnm <- nnWeightLayout(0L, K = 1L, H = 2L); nW <- length(wnm)

  ## the state does NOT depend on the weights; the prediction does, directly
  obj <- paste0(sprintf("param(%s)\n", paste(wnm, collapse = ", ")),
                "y = nn1(0, centr)\nd/dt(centr) = -k*centr")
  mBase <- rxode2::rxode2(obj)

  ## build the prediction sensitivity the way .nnAugmentFromUi now does:
  ## state route + direct route
  st <- rxode2::rxStateOde(rxode2::rxS(rxode2::rxGetModel(obj), TRUE,
                                       promoteLinSens = FALSE))
  sym <- rxode2::rxS(rxode2::rxGetModel(obj), TRUE, promoteLinSens = FALSE)
  cl <- .nnParseCallAll(obj)[[1L]]
  dpds <- .nnDpDs(obj, st, "y")
  dpdg <- .nnDvarDg(sym, "y", cl)
  predsw <- vapply(seq_len(nW) - 1L, function(j)
    sprintf("rx_predsw_%d_ = (%s)*rx_sw_%s_%d_ + (%s)*nnWg1(0, %d, centr)",
            j, dpds[[1L]], st[1L], j, dpdg, j), character(1))
  mAug <- rxode2::rxode2(paste(c(nnAugmentModel(obj, H = 2L), predsw), collapse = "\n"))

  w <- c(0.3, -0.25, 0.4, 0.1, 0.35, -0.15, 0.2)
  p <- c(k = 0.25, stats::setNames(rep(0, nW), wnm)); ic <- c(centr = 8)
  ot <- c(1, 2, 3, 4); dv <- c(0.2, 0.1, 0.15, 0.05); sig <- 0.3
  ev <- rxode2::et(ot)

  ll <- function(ww) {
    nnSetWeights(0L, ww)
    s <- rxode2::rxSolve(mBase, ev, params = p, inits = ic)
    stats::setNames(sum(stats::dnorm(dv, s$y[match(ot, s$time)], sig, log = TRUE)), NULL)
  }

  nnSetWeights(0L, w)
  sA <- rxode2::rxSolve(mAug, ev, params = p, inits = ic)
  idx <- match(ot, sA$time)
  score <- (dv - sA$y[idx]) / sig^2
  got <- vapply(seq_len(nW) - 1L,
                function(j) sum(score * sA[[sprintf("rx_predsw_%d_", j)]][idx]),
                numeric(1))

  h <- 1e-5
  fd <- vapply(seq_len(nW), function(j) {
    wp <- w; wp[j] <- wp[j] + h; wm <- w; wm[j] <- wm[j] - h
    (ll(wp) - ll(wm)) / (2 * h)
  }, numeric(1))

  expect_equal(got, fd, tolerance = 1e-4)

  ## and the state route ALONE -- what was assembled before -- is exactly zero
  ## here, so this test cannot pass for the wrong reason
  stateOnly <- vapply(seq_len(nW) - 1L,
                      function(j) sum(score * sA[[sprintf("rx_sw_centr_%d_", j)]][idx]),
                      numeric(1))
  expect_equal(stateOnly, rep(0, nW))
  expect_gt(max(abs(fd)), 1)
})
