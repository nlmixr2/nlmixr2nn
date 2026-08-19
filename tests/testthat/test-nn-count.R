## Count endpoints (S2.6).
##
## For `y ~ pois(lam)` rxode2 compiles the PREDICTION to the log-density itself,
## and nlmixr2est's hook then reports d(LL)/d(f) = 1 -- true, and useless to
## chain, because the solve cannot differentiate a log-density it has no DV for.
## The chain therefore runs through the distribution's PARAMETER, and the score
## is that distribution's own derivative.  Both halves are checked against a
## finite difference of the model's real log-likelihood.

.countModel <- function(direct = FALSE) {
  if (direct) {
    ## the network feeds lam directly: the state route carries none of the effect
    function() {
      ini({ lk <- -1 })
      model({
        k <- exp(lk)
        d/dt(centr) <- -k * centr
        lam <- exp(nn(centr, n_hidden = 2L, act = "tanh")) + 0.5
        centr2 ~ pois(lam)
      })
    }
  } else {
    ## the network drives the ODE: the effect reaches lam through the state
    function() {
      ini({ lk <- -1 })
      model({
        k <- exp(lk)
        d/dt(centr) <- -k * centr + nn(centr, n_hidden = 2L, act = "tanh")
        lam <- centr / 4
        centr2 ~ pois(lam)
      })
    }
  }
}

test_that("the chain targets the distribution's parameter, not the prediction", {
  skip_if_not_installed("rxode2")
  local_nn()
  set.seed(5)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.countModel())))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
  ## `centr2` is the endpoint variable; `lam` is what the network moves
  expect_equal(aug$endpoint, "lam")
  expect_equal(aug$dist$dist, "pois")
  expect_equal(aug$dist$target, "lam")
  expect_true(is.na(aug$dist$size))
})

for (.direct in c(FALSE, TRUE)) {
  test_that(sprintf("the Poisson weight gradient matches a finite difference (%s)",
                    if (.direct) "network feeds lam directly" else "through the ODE"), {
    skip_if_not_installed("rxode2")
    local_nn()
    .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
    set.seed(if (.direct) 8L else 5L)
    ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.countModel(.direct))))
    aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
    nW <- aug$nW

    setW <- function(w) for (net in aug$nets) {
      nnSetMeta(net$id, net$augBase, net$K, net$H, net$act)
      nnSetWeights(net$id, w[net$gIdx])
    }
    pars <- c(lk = -1, stats::setNames(rep(0, nW), aug$weights))
    ev <- rxode2::et(c(1, 2, 3, 4)); ic <- c(centr = 8)
    ot <- c(1, 2, 3, 4); dv <- c(3, 2, 4, 1)
    solveAug <- function(w) {
      setW(w)
      rxode2::rxSolve(aug$mAug, ev, params = pars, inits = ic,
                      returnType = "data.frame")
    }
    ## the sensitivity states do not feed lam, so the augmented solve is also the
    ## reference for the finite difference
    ll <- function(w) {
      s <- solveAug(w)
      sum(stats::dpois(dv, s$lam[match(ot, s$time)], log = TRUE))
    }

    set.seed(21)
    w <- stats::rnorm(nW, 0, 0.3)
    s <- solveAug(w)
    idx <- match(ot, s$time)
    ## the score comes from rxode2's own llikPois derivative, not from algebra
    ## rewritten here -- and it had better equal the textbook one
    score <- .nnDistScore(aug$dist, s$lam[idx], dv)
    expect_equal(score, dv / s$lam[idx] - 1, tolerance = 1e-10)

    got <- vapply(seq_len(nW) - 1L,
                  function(j) sum(score * s[[sprintf("rx_predsw_%d_", j)]][idx]),
                  numeric(1))
    h <- 1e-5
    fd <- vapply(seq_len(nW), function(j) {
      wp <- w; wp[j] <- wp[j] + h; wm <- w; wm[j] <- wm[j] - h
      (ll(wp) - ll(wm)) / (2 * h)
    }, numeric(1))

    expect_equal(got, fd, tolerance = 1e-4)
    ## and the gradient is not trivially zero, or the comparison proves nothing
    expect_gt(max(abs(fd)), 1e-2)
  })
}

test_that("the binomial weight gradient matches a finite difference", {
  ## binom carries a SIZE as well as a probability.  rxode2 compiles
  ## `y ~ binom(a, b)` to llikBinom(DV, a, b) = (x, size, prob), and its .rxD
  ## entry differentiates prob only -- so the network drives b, and a has to be
  ## read alongside it or the score is formed against the wrong argument.
  skip_if_not_installed("rxode2")
  local_nn()
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
  m <- function() {
    ini({ lk <- -1 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr
      ## rxode2 requires the size to be modeled, so it is bound to a data
      ## column -- which is also how a real trial data set carries it
      nsz <- ntrials
      pr <- 1 / (1 + exp(-nn(centr, n_hidden = 2L, act = "tanh")))
      centr2 ~ binom(nsz, pr)
    })
  }
  set.seed(12)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
  expect_equal(aug$endpoint, "pr")          # the probability, not the size
  expect_equal(aug$dist$size, "nsz")
  nW <- aug$nW

  setW <- function(w) for (net in aug$nets) {
    nnSetMeta(net$id, net$augBase, net$K, net$H, net$act)
    nnSetWeights(net$id, w[net$gIdx])
  }
  pars <- c(lk = -1, stats::setNames(rep(0, nW), aug$weights))
  ot <- c(1, 2, 3, 4); dv <- c(7, 5, 6, 3)
  ## the number of trials is a data column, as it is in a real trial data set
  ev <- data.frame(id = 1L, time = ot, evid = 0, ntrials = c(10, 10, 12, 12))
  ic <- c(centr = 8)
  solveAug <- function(w) {
    setW(w)
    rxode2::rxSolve(aug$mAug, ev, params = pars, inits = ic, returnType = "data.frame")
  }
  ll <- function(w) {
    s <- solveAug(w)
    i <- match(ot, s$time)
    sum(stats::dbinom(dv, size = s$nsz[i], prob = s$pr[i], log = TRUE))
  }

  set.seed(31)
  w <- stats::rnorm(nW, 0, 0.3)
  s <- solveAug(w); idx <- match(ot, s$time)
  score <- .nnDistScore(aug$dist, s$pr[idx], dv, s$nsz[idx])
  expect_equal(score, dv / s$pr[idx] - (s$nsz[idx] - dv) / (1 - s$pr[idx]),
               tolerance = 1e-10)
  got <- vapply(seq_len(nW) - 1L,
                function(j) sum(score * s[[sprintf("rx_predsw_%d_", j)]][idx]),
                numeric(1))
  h <- 1e-5
  fd <- vapply(seq_len(nW), function(j) {
    wp <- w; wp[j] <- wp[j] + h; wm <- w; wm[j] <- wm[j] - h
    (ll(wp) - ll(wm)) / (2 * h)
  }, numeric(1))
  expect_equal(got, fd, tolerance = 1e-4)
  expect_gt(max(abs(fd)), 1e-2)
})

test_that("an unscorable distribution is refused, naming why", {
  skip_if_not_installed("rxode2")
  local_nn()
  m <- function() {
    ini({ lk <- -1 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr
      sz <- 10
      pr <- 1 / (1 + exp(-nn(centr, n_hidden = 2L, act = "tanh")))
      centr2 ~ nbinom(sz, pr)
    })
  }
  set.seed(4)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  ## nbinom is a real rxode2 distribution, just not one with a score here -- so
  ## this checks the refusal, not a parse failure
  expect_equal(as.character(ui$predDf$distribution[1L]), "nbinom")
  expect_error(.nnAugmentFromUi(rxode2::rxUiDecompress(ui)), "nbinom")
})

test_that("cotangent = \"exact\" is refused on a count endpoint", {
  ## Not a preference.  For a non-normal endpoint the hook's f IS the
  ## log-density, so it reports d(LL)/d(f) = 1; multiplying that by a
  ## sensitivity of the distribution's parameter is not a gradient of anything.
  ##
  ## Built from the REAL inferred schedule, not hand-written predicates: a rule
  ## reading a predicate that .nnInferSched() does not actually publish sees
  ## NULL and quietly goes the wrong way, and hand-written predicates would hide
  ## exactly that.
  skip_if_not_installed("rxode2")
  local_nn()
  mkEnv <- function(m) {
    e <- new.env(); class(e) <- c("focei", "environment")
    set.seed(9)
    assign("ui", suppressWarnings(suppressMessages(rxode2::rxode2(m))), envir = e)
    assign("data", data.frame(ID = 1, TIME = 1, DV = 1, EVID = 0), envir = e)
    assign("control", nlmixr2est::foceiControl(print = 0L), envir = e)
    e
  }
  ePois <- mkEnv(.countModel())
  infP <- .nnInferSched(ePois)
  expect_equal(infP$cotangent, "dist")
  expect_true(infP$predicates$distScore)
  expect_equal(infP$predicates$distribution, "pois")
  expect_error(.nnResolveSched(nnControl(cotangent = "exact"), infP),
               "log-density itself")
  ## asking for it explicitly is accepted
  expect_equal(.nnResolveSched(nnControl(cotangent = "dist"), infP)$cotangent, "dist")

  ## and on a normal endpoint the distribution score is refused instead
  eNorm <- mkEnv(function() {
    ini({ lk <- -1; add.sd <- 0.3 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr + nn(centr, n_hidden = 2L, act = "tanh")
      centr ~ add(add.sd)
    })
  })
  infN <- .nnInferSched(eNorm)
  expect_false(isTRUE(infN$predicates$distScore))
  expect_error(.nnResolveSched(nnControl(cotangent = "dist"), infN), "pois/binom")
})

test_that("a Poisson UDE trains end to end", {
  ## The pieces above are checked against finite differences; this checks that
  ## the whole loop actually descends on a count endpoint, through the ordinary
  ## nlmixr2() entry point with no nn helper call.
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  skip_on_cran()
  local_nn()

  set.seed(101)
  sim <- rxode2::rxode2({
    k <- 0.25
    d/dt(centr) <- -k * centr
    lam <- exp(-0.5 + 1.2 * centr / (3 + centr))     # the rate the net must learn
  })
  ev <- rxode2::et(seq(1, 6, by = 1)) |> rxode2::et(id = 1:8)
  s <- rxode2::rxSolve(sim, ev, inits = c(centr = 20))
  ## uppercase columns on purpose: the NONMEM spelling, see test-nn-datacols.R
  d <- data.frame(ID = s$id, TIME = s$time, DV = stats::rpois(nrow(s), s$lam),
                  EVID = 0, CMT = 2)

  m <- function() {
    ini({ lk <- -1.4 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr
      lam <- exp(nn(centr, n_hidden = 4L, act = "softplus"))
      centr2 ~ pois(lam)
    })
  }
  set.seed(7)
  fit <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(m, d, est = "focei", nlmixr2est::foceiControl(print = 0L),
                        nn = nnControl(mode = "iter", rounds = 4L, wSteps = 8L,
                                       lr = 0.05))))

  expect_true(all(is.finite(fit$nnWeights)))
  h <- fit$nnParHist
  expect_equal(nrow(h), 4L)
  ## the objective has to actually come down -- a count endpoint that merely
  ## runs proves nothing
  expect_lt(h$objf[nrow(h)], h$objf[1L])
  ## and the schedule chose the distribution's own score, not the hook's
  expect_equal(fit$nnSched$cotangent, "dist")
})
