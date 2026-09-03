## The zero-boilerplate guarantees.
##
## The whole point of nn() is that a model containing one is an ORDINARY
## rxode2/nlmixr2 model.  Every test in this file must therefore call nothing but
## rxode2()/rxSolve()/nlmixr2() and the two accessors -- no nnUpdate(), no
## nnCovData(), no nnTorchModel(), no nnWithLoader(), and above all no
## nnClearMeta().  If a guarantee here needs a helper call to hold, it does not
## hold.

.zbMod <- function() {
  ini({ lka <- 0.5; lVc <- 1 })
  model({
    ka <- exp(lka); Vc <- exp(lVc)
    d/dt(depot)   <- -ka * depot
    d/dt(central) <-  ka * depot - nn(central, nHidden = 4L)
    cp <- central / Vc
  })
}

.zbModB <- function() {
  ini({ lka <- 0.5; lVc <- 1 })
  model({
    ka <- exp(lka); Vc <- exp(lVc)
    d/dt(depot)   <- -ka * depot
    d/dt(central) <-  ka * depot - nn(central, nHidden = 6L, act = "tanh")
    cp <- central / Vc
  })
}

.zbEv <- function() {
  rxode2::et(amt = 100, cmt = "depot") |> rxode2::et(0, 24, by = 4)
}

test_that("a parsed model is seeded and solves with no setup call", {
  set.seed(1)
  ui <- suppressMessages(rxode2::rxode2(.zbMod))

  ## the weights are ON the model, not in some transient buffer
  w <- nnWeights(ui)
  expect_true(is.numeric(w))
  expect_length(w, 4L * 1L + 2L * 4L + 1L)   # H*K + 2H + 1
  expect_false(anyNA(w))

  s <- rxode2::rxSolve(ui, .zbEv())
  expect_false(anyNA(s$cp))
  ## an unbound C registry gives NA; an empty weight buffer gives an identically
  ## zero network.  Both were reachable before, and both are caught here.
  expect_true(any(s$cp != 0))
})

test_that("set.seed() reproduces a network, and different seeds differ", {
  set.seed(42); a <- nnWeights(suppressMessages(rxode2::rxode2(.zbMod)))
  set.seed(42); b <- nnWeights(suppressMessages(rxode2::rxode2(.zbMod)))
  set.seed(43); c <- nnWeights(suppressMessages(rxode2::rxode2(.zbMod)))
  expect_identical(a, b)
  expect_false(isTRUE(all.equal(unname(a), unname(c))))
})

test_that("building a model does not disturb the caller's random stream", {
  ## adding a network to a model must not shift unrelated draws in a script
  set.seed(99); ref <- stats::runif(3)
  set.seed(99); invisible(suppressMessages(rxode2::rxode2(.zbMod)))
  got <- stats::runif(3)
  expect_equal(got, ref)
})

test_that("two nn models are alive at once without contaminating each other", {
  ## every single-network model is network 0 in the compiled registry, so this
  ## is the test that the registry is bound per solve rather than accumulated.
  set.seed(10); uiA <- suppressMessages(rxode2::rxode2(.zbMod))
  set.seed(20); uiB <- suppressMessages(rxode2::rxode2(.zbModB))
  ev <- .zbEv()

  a1 <- rxode2::rxSolve(uiA, ev)
  b1 <- rxode2::rxSolve(uiB, ev)
  a2 <- rxode2::rxSolve(uiA, ev)

  expect_equal(a1$cp, a2$cp, tolerance = 1e-12)
  expect_false(isTRUE(all.equal(a1$cp, b1$cp)))
})

test_that("an unrelated model solved in between is unaffected", {
  plain <- function() {
    ini({ lk <- -1 })
    model({ k <- exp(lk); d/dt(A) <- -k * A })
  }
  ev <- .zbEv()
  pui <- suppressMessages(rxode2::rxode2(plain))
  ref <- rxode2::rxSolve(pui, rxode2::et(amt = 100, cmt = "A") |> rxode2::et(0, 24, by = 4))

  set.seed(3); ui <- suppressMessages(rxode2::rxode2(.zbMod))
  invisible(rxode2::rxSolve(ui, ev))
  got <- rxode2::rxSolve(pui, rxode2::et(amt = 100, cmt = "A") |> rxode2::et(0, 24, by = 4))
  expect_equal(got$A, ref$A, tolerance = 1e-12)
})

test_that("a seeded model survives saveRDS and reload", {
  set.seed(7)
  ui <- suppressMessages(rxode2::rxode2(.zbMod))
  ev <- .zbEv()
  s0 <- rxode2::rxSolve(ui, ev)

  f <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  saveRDS(ui, f)
  s1 <- rxode2::rxSolve(readRDS(f), ev)
  expect_equal(s1$cp, s0$cp, tolerance = 1e-12)
})

test_that("a fit marks its weights trained, and a refit warm-starts from them", {
  ## The refit guarantee, asserted on the MECHANISM rather than on how well a
  ## two-round refit happens to recover an eta -- that number moves with thread
  ## count and is not what this is about.
  skip_on_cran()
  skip_if_not_installed("nlmixr2est")
  skip_if_no_torch()

  set.seed(1)
  truth <- rxode2::rxode2("d/dt(centr) = -(2)*centr/(3+centr)")
  d <- do.call(rbind, lapply(1:6, function(id) {
    s <- rxode2::rxSolve(truth, data.frame(id = id, time = c(0, .5, 1, 2, 4, 6, 8, 10),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
           returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr + rnorm(nrow(s), 0, .1)))
  }))
  qsp <- function() {
    ini({ add.sd <- 0.3 })
    model({
      g <- nn(centr, nHidden = 3L, act = "tanh")
      d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
      centr ~ add(add.sd)
    })
  }
  f1 <- suppressWarnings(suppressMessages(nlmixr2est::nlmixr2(qsp, d, "bobyqa")))
  ui1 <- rxode2::rxUiDecompress(f1$finalUi)

  ## the fit says these weights are trained -- which is what stops a refit from
  ## re-applying input scaling to them and from re-running the population pre-fit
  expect_true(isTRUE(get("nnTrained", envir = ui1, inherits = FALSE)))
  ## and the trained values, not the parse-time draw, are what the model carries
  expect_equal(unname(nnWeights(ui1)), unname(f1$nnWeights), tolerance = 1e-12)

  ## feeding the fit back in starts from those weights
  f2 <- suppressWarnings(suppressMessages(nlmixr2est::nlmixr2(ui1, d, "bobyqa")))
  expect_true(is.finite(f2$objf))
  ## The refit stays in the same basin.  Deliberately NOT "strictly better": a
  ## derivative-free optimizer restarted at an optimum opens a fresh trust region
  ## and can settle a little off it.  A warm start means the refit BEGINS from
  ## the trained weights -- asserted above on the mechanism -- not that it is
  ## guaranteed to improve on a converged fit.
  expect_lt(abs(f2$objf - f1$objf), 0.05 * abs(f1$objf))
})

test_that("nnEval() reports exactly what the ODE right-hand side evaluates", {
  ## the accessor must not be a reimplementation that can drift from the
  ## compiled activations the solve actually integrates
  m <- function() {
    model({ y <- nn(u, nHidden = 5L); d/dt(A) <- -A * 0 })
  }
  set.seed(4)
  ui <- suppressMessages(rxode2::rxode2(m))
  us <- c(0.1, 0.5, 1, 2, 5)
  ev <- do.call(rbind, lapply(seq_along(us),
                              function(i) data.frame(id = i, time = c(0, 1), u = us[i])))
  s <- rxode2::rxSolve(ui, ev, returnType = "data.frame")
  ref <- vapply(seq_along(us), function(i) s$y[s$id == i][1], numeric(1))
  expect_equal(nnEval(ui, u = us)$value, ref, tolerance = 1e-12)
})

test_that("nnEval() rejects the wrong number of inputs and an unknown network", {
  set.seed(4)
  ui <- suppressMessages(rxode2::rxode2(.zbMod))
  expect_error(nnEval(ui, central = 1, other = 2), "takes 1 input")
  expect_error(nnEval(ui, central = 1, net = 3), "no network 3")
  expect_error(nnEval(), "first argument")
})

test_that("a vector nHidden is refused explicitly, not silently mishandled", {
  m <- function() {
    model({ d/dt(A) <- -nn(A, nHidden = c(4L, 4L)) })
  }
  expect_error(suppressMessages(rxode2::rxode2(m)), "single hidden layer")
})
