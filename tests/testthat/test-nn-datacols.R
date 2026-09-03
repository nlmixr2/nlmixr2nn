## ID/TIME/DV/EVID vs id/time/dv/evid.
##
## nlmixr2 accepts either spelling, and every test in this package happened to
## use the lower-case one.  The weight step read `data$time` unguarded, so on a
## data set written the NONMEM way -- which is what most users write -- the
## observation match returned all NA, every prediction was NA, and the assembled
## gradient was NaN.  Nothing said so: it surfaced only when a count endpoint,
## whose score asserts its input is finite, refused it.

test_that("the resolver finds a standard column in either case", {
  d <- data.frame(ID = 1, TIME = 0, DV = 1)
  expect_equal(.nnDataCol(d, "ID"), "ID")
  expect_equal(.nnDataCol(d, "time"), "TIME")
  expect_true(is.na(.nnDataCol(d, "EVID")))
  d2 <- data.frame(id = 1, time = 0, dv = 1)
  expect_equal(.nnDataCol(d2, "TIME"), "time")
})

test_that("the weight gradient does not depend on how the columns are spelled", {
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  local_nn()
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
  m <- function() {
    ini({ lk <- -1; add.sd <- 0.3 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr + nn(centr, nHidden = 2L, act = "tanh")
      centr ~ add(add.sd)
    })
  }
  set.seed(9)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
  for (net in aug$nets) {
    nnTorchInit(net$id, net$K, net$H, act = net$act)
    nnTorchSetWeights(net$id, stats::setNames(rep(0.2, length(net$gIdx)), NULL))
  }
  on.exit(for (net in aug$nets) try(nnTorchFree(net$id), silent = TRUE), add = TRUE)

  lower <- data.frame(id = rep(1:3, each = 4), time = rep(1:4, 3), evid = 0,
                      dv = c(6.1, 4.8, 3.9, 3.1, 5.9, 4.7, 3.8, 3.0,
                             6.3, 4.9, 4.0, 3.2))
  upper <- stats::setNames(lower, c("ID", "TIME", "EVID", "DV"))

  grad <- function(d) {
    ic <- .nnDataCol(d, "ID"); tc <- .nnDataCol(d, "TIME")
    ev <- d[[.nnDataCol(d, "EVID")]]
    obs <- is.na(ev) | ev == 0
    dv <- d[[.nnDataCol(d, "DV")]]
    ws <- .nnWeightStepper(aug, d, ic, tc, obs, dv,
                           stats::setNames(rep(0, aug$nW), aug$weights))
    ws(stats::setNames(numeric(0), character(0)),
       list(add = 0.3, prop = 0), c(lk = -1, add.sd = 0.3), step = FALSE)
  }
  gl <- grad(lower)
  gu <- grad(upper)

  ## the defect made these NaN, so "finite" is half the assertion and "equal" the
  ## other half -- NaN == NaN would not have been caught by equality alone
  expect_true(all(is.finite(unlist(gl$dLLdw))))
  expect_true(all(is.finite(unlist(gu$dLLdw))))
  expect_equal(gu$dLLdw, gl$dLLdw)
  expect_equal(gu$rmse, gl$rmse)
  ## and the gradient is not zero, or spelling could not have changed it
  expect_gt(max(abs(unlist(gl$dLLdw))), 1e-6)
})

test_that("data missing a required column is refused by name", {
  skip_if_not_installed("rxode2")
  local_nn()
  env <- new.env()
  class(env) <- c("focei", "environment")
  set.seed(9)
  m <- function() {
    ini({ lk <- -1; add.sd <- 0.3 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr + nn(centr, nHidden = 2L, act = "tanh")
      centr ~ add(add.sd)
    })
  }
  assign("ui", suppressWarnings(suppressMessages(rxode2::rxode2(m))), envir = env)
  assign("data", data.frame(ID = 1, DV = 1), envir = env)   # no TIME
  assign("control", nlmixr2est::foceiControl(print = 0L), envir = env)
  expect_error(.nnRunCtx(env, nnControl()), "TIME")
})
