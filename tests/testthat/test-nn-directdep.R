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

test_that("the direct term uses each weight's OWN network, with a local index", {
  ## Weight indices are GLOBAL across networks (net 0's block, then net 1's),
  ## while nnWg<K> takes a LOCAL index within its own network.  Getting that
  ## mapping wrong would attribute one network's direct gradient to the other's
  ## weights -- a wrong gradient, not an error.  A single-network model cannot
  ## catch it, because there the two indexings coincide.
  skip_if_not_installed("rxode2")
  local_nn()
  m <- function() {
    ini({ lk <- -1; add.sd <- 0.3 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr
      ## two networks of DIFFERENT widths, both feeding the prediction directly
      y <- nn(centr, n_hidden = 2L, act = "tanh") + nn(centr, n_hidden = 3L, act = "tanh")
      y ~ add(add.sd)
    })
  }
  set.seed(6)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))

  nW0 <- 2L * 1L + 2L * 2L + 1L      # net 0: K=1, H=2  -> 7
  nW1 <- 3L * 1L + 2L * 3L + 1L      # net 1: K=1, H=3  -> 10
  expect_equal(aug$nW, nW0 + nW1)

  lines <- strsplit(aug$text, "\n")[[1]]
  psw <- grep("^rx_predsw_", lines, value = TRUE)
  expect_length(psw, nW0 + nW1)

  ## the first block belongs to network 0 with local indices 0..nW0-1 ...
  expect_true(grepl("nnWg1\\(0, 0,", psw[1L]))
  expect_true(grepl("nnWg1\\(0, 6,", psw[nW0]))
  ## ... and the second to network 1, with its local index restarting at 0
  expect_true(grepl("nnWg1\\(1, 0,", psw[nW0 + 1L]))
  expect_true(grepl("nnWg1\\(1, 9,", psw[nW0 + nW1]))
  ## no weight may be attributed to the other network
  expect_false(any(grepl("nnWg1\\(1,", psw[seq_len(nW0)])))
  expect_false(any(grepl("nnWg1\\(0,", psw[(nW0 + 1L):(nW0 + nW1)])))
})

test_that("a registry that disagrees with the model is refused, not indexed past its end", {
  ## .ownerOf is built from the parsed calls while the weight layout comes from
  ## the registry; a mismatch used to surface as "subscript out of bounds" from
  ## indexing a named vector past its end.
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
  set.seed(6)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  ## inject a network the model never calls
  .nnEnv$reg[["1"]] <- list(id = 1L, K = 1L, H = 2L, act = "tanh",
                            weights = nnWeightLayout(1L, 1L, 2L))
  expect_error(.nnAugmentFromUi(rxode2::rxUiDecompress(ui)),
               "do not match the ones the model calls")
})

test_that("the multi-network direct gradient matches a finite difference", {
  ## The text assertions above prove the string builder routes indices to the
  ## right network.  They do NOT prove the assembled number is the gradient --
  ## a regex test passes just as happily if the compiled layout is misaligned.
  ## This checks the value, across the combined two-network weight layout.
  skip_if_not_installed("rxode2")
  local_nn()
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)

  m <- function() {
    ini({ lk <- -1; add.sd <- 0.3 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr
      y <- nn(centr, n_hidden = 2L, act = "tanh") + nn(centr, n_hidden = 3L, act = "tanh")
      y ~ add(add.sd)
    })
  }
  set.seed(6)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
  nWt <- aug$nW

  ## put each network's slice of the GLOBAL weight vector where the augmented
  ## model reads it
  setW <- function(w) {
    for (net in aug$nets) {
      nnSetMeta(net$id, net$augBase, net$K, net$H, net$act)
      nnSetWeights(net$id, w[net$gIdx])
    }
  }
  pars <- c(lk = -1, stats::setNames(rep(0, nWt), aug$weights))
  ev <- rxode2::et(c(1, 2, 3, 4))
  ic <- c(centr = 8)
  ot <- c(1, 2, 3, 4); dv <- c(0.4, 0.25, 0.15, 0.1); sig <- 0.3

  solveY <- function(w) {
    setW(w)
    s <- rxode2::rxSolve(aug$mAug, ev, params = pars, inits = ic,
                         returnType = "data.frame")
    s$y[match(ot, s$time)]
  }
  ll <- function(w) sum(stats::dnorm(dv, solveY(w), sig, log = TRUE))

  set.seed(11)
  w <- stats::rnorm(nWt, 0, 0.3)
  setW(w)
  sA <- rxode2::rxSolve(aug$mAug, ev, params = pars, inits = ic,
                        returnType = "data.frame")
  idx <- match(ot, sA$time)
  score <- (dv - sA$y[idx]) / sig^2
  got <- vapply(seq_len(nWt) - 1L,
                function(j) sum(score * sA[[sprintf("rx_predsw_%d_", j)]][idx]),
                numeric(1))

  h <- 1e-5
  fd <- vapply(seq_len(nWt), function(j) {
    wp <- w; wp[j] <- wp[j] + h; wm <- w; wm[j] <- wm[j] - h
    (ll(wp) - ll(wm)) / (2 * h)
  }, numeric(1))

  expect_equal(got, fd, tolerance = 1e-4)
  ## both networks must actually contribute, or the test would pass with one
  ## network's block silently zero
  expect_gt(max(abs(fd[aug$nets[[1L]]$gIdx])), 1e-3)
  expect_gt(max(abs(fd[aug$nets[[2L]]$gIdx])), 1e-3)
})
