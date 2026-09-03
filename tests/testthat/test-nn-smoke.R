## E5.5: {estimator} x {error model} x {eta / no eta}.
##
## Each combination is a different route through the engine -- which weight-block
## base is resolved, whether the kernel reads the loader or the data columns,
## which cotangent source applies -- and a route with no test is a route where a
## wrong answer has nowhere to show up.
##
## The model is the one the recovery tests already validate (a Michaelis-Menten
## truth the network has to learn), NOT a fresh one: on an ad-hoc model the
## network competes with a free structural parameter, the fit is near
## unidentifiable, and the objective wanders for reasons that have nothing to do
## with the code under test.  Asserting descent only means something on a problem
## known to be learnable.

.smokeData <- function(nId = 10L, err = "add") {
  set.seed(42)
  Vmax <- 1.2; Km <- 3
  etaTrue <- stats::rnorm(nId, 0, sqrt(0.2))
  mm <- rxode2::rxode2({ d/dt(centr) <- -(Vmax * exp(eV)) * centr / (Km + centr) })
  do.call(rbind, lapply(seq_len(nId), function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8, 10),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
           params = c(Vmax = Vmax, Km = Km, eV = etaTrue[id]), returnType = "data.frame")
    s <- s[s$time > 0, ]
    dv <- switch(err,
      add   = s$centr + stats::rnorm(nrow(s), 0, 0.1),
      lnorm = s$centr * exp(stats::rnorm(nrow(s), 0, 0.05)))
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), dv = c(NA, dv))
  }))
}

.smokeModel <- function(err = "add", hasEta = TRUE) {
  errLine <- switch(err, add = "centr ~ add(add.sd)", lnorm = "centr ~ lnorm(add.sd)")
  if (hasEta) {
    ini <- "add.sd <- 0.3; eta.nn ~ 0.2"
    net <- "nn(centr, eta.nn, nHidden = 3L, act = \"tanh\")"
  } else {
    ini <- "add.sd <- 0.3"
    net <- "nn(centr, nHidden = 3L, act = \"tanh\")"
  }
  eval(parse(text = sprintf(
    "function() { ini({ %s }); model({ g <- %s; d/dt(centr) <- -(1.0/(1.0 + exp(-g)))*centr; %s }) }",
    ini, net, errLine)))
}

.smokeControl <- function(est) {
  switch(est,
    focei  = nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 8L,
                                      maxInnerIterations = 25L, calcTables = FALSE),
    saem   = nlmixr2est::saemControl(print = 0L, nBurn = 60, nEm = 60, calcTables = FALSE),
    impmap = nlmixr2est::impmapControl(print = 0L, calcTables = FALSE),
    emvi   = nlmixr2est::fbviControl(print = 0L, calcTables = FALSE))
}

.smokeGrid <- expand.grid(est = c("focei", "saem", "impmap", "emvi"),
                          err = c("add", "lnorm"), stringsAsFactors = FALSE)

for (.i in seq_len(nrow(.smokeGrid))) local({
  .est <- .smokeGrid$est[.i]; .err <- .smokeGrid$err[.i]
  test_that(sprintf("smoke: est=%s, %s endpoint, latent eta", .est, .err), {
    skip_if_not_installed("rxode2")
    skip_if_no_torch()
    skip_on_cran()
    local_nn()
    set.seed(5)
    f <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(.smokeModel(.err, TRUE), .smokeData(err = .err), .est,
                          .smokeControl(.est),
                          nn = nnControl(mode = "iter", rounds = 4L, wSteps = 8L,
                                         lr = 0.03))))
    expect_true(all(is.finite(f$nnWeights)))
    expect_true(is.finite(f$objf))
    h <- f$nnParHist
    expect_true(all(is.finite(h$objf)))
    expect_true(all(is.finite(h$rmse)))
    ## it has to actually learn: a cell that merely runs proves nothing
    expect_lt(h$rmse[nrow(h)], h$rmse[1L])
  })
})

test_that("smoke: a population (no-BSV) model still trains", {
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  skip_on_cran()
  local_nn()
  set.seed(5)
  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(.smokeModel("add", FALSE), .smokeData(), "focei",
                        .smokeControl("focei"),
                        nn = nnControl(mode = "iter", rounds = 4L, wSteps = 8L,
                                       lr = 0.03))))
  expect_true(all(is.finite(f$nnWeights)))
  expect_true(is.finite(f$objf))
  expect_lt(f$nnParHist$rmse[nrow(f$nnParHist)], f$nnParHist$rmse[1L])
})
