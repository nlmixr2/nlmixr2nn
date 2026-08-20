## A row with EVID 0 and no DV.
##
## Dropped samples, BLQ records and placeholder times are everywhere in real
## PK/PD data, and none of them is an observation: nlmixr2est never evaluates
## them, so its likelihood hook never numbers them.  Counting them here did two
## silent things.  The captured cotangent is keyed by (subject, observation
## index), so one missing DV shifted every later index and the exact score could
## not be matched at all.  And the closed-form score of a missing DV is NA, which
## makes the whole summed gradient NA -- torch then wrote NaN into every weight
## while the fit returned an object with a number in it.
##
## The older standalone trainer (R/nnTrain.R) filtered these out; the engine that
## replaced it did not.

.mdvData <- function(nId = 6L) {
  set.seed(3)
  sim <- rxode2::rxode2({
    k <- 0.25
    d/dt(centr) <- -k * centr - 0.4 * centr / (2 + centr)
  })
  ev <- rxode2::et(1:5) |> rxode2::et(id = seq_len(nId))
  s <- rxode2::rxSolve(sim, ev, inits = c(centr = 10))
  data.frame(ID = s$id, TIME = s$time, EVID = 0,
             DV = s$centr + stats::rnorm(nrow(s), 0, 0.2))
}

.mdvModel <- function(err = "add") {
  if (err == "add") {
    function() {
      ini({ lk <- -1.4; add.sd <- 0.2 })
      model({
        k <- exp(lk)
        d/dt(centr) <- -k * centr + nn(centr, n_hidden = 3L, act = "softplus")
        centr ~ add(add.sd)
      })
    }
  } else {
    function() {
      ini({ lk <- -1.4; add.sd <- 0.2 })
      model({
        k <- exp(lk)
        d/dt(centr) <- -k * centr + nn(centr, n_hidden = 3L, act = "softplus")
        centr ~ lnorm(add.sd)
      })
    }
  }
}

test_that("a row with no DV is not an observation", {
  skip_if_not_installed("rxode2")
  local_nn()
  d <- .mdvData(2L)
  d$DV[3] <- NA_real_
  env <- new.env(); class(env) <- c("focei", "environment")
  set.seed(9)
  assign("ui", suppressWarnings(suppressMessages(rxode2::rxode2(.mdvModel()))),
         envir = env)
  assign("data", d, envir = env)
  assign("control", nlmixr2est::foceiControl(print = 0L), envir = env)
  skip_if_no_torch()
  ctx <- .nnRunCtx(env, .nnInferSched(env))
  on.exit(for (n in ctx$aug$nets) tryCatch(nnTorchFree(n$id), silent = TRUE), add = TRUE)
  expect_equal(sum(ctx$obs), nrow(d) - 1L)
  expect_false(ctx$obs[3L])
  expect_false(anyNA(ctx$dv[ctx$obs]))
})

test_that("one missing DV does not turn every weight into NaN", {
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  skip_on_cran()
  local_nn()
  d <- .mdvData()
  d$DV[8] <- NA_real_
  set.seed(7)
  fit <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(.mdvModel(), d, est = "focei",
                        nlmixr2est::foceiControl(print = 0L),
                        nn = nnControl(mode = "iter", rounds = 2L, wSteps = 4L,
                                       lr = 0.05))))
  ## the defect produced NaN weights, an NA rmse and a sentinel objective, and
  ## reported none of it
  expect_true(all(is.finite(fit$nnWeights)))
  expect_true(all(is.finite(fit$nnParHist$rmse)))
  expect_true(is.finite(fit$objf))
  ## and it still trains
  expect_lt(fit$nnParHist$rmse[2L], fit$nnParHist$rmse[1L])
})

test_that("the exact cotangent still aligns when a DV is missing", {
  ## The sharper half: a transformed endpoint has no closed form to fall back to,
  ## so a shifted observation index is a hard failure rather than a quiet
  ## downgrade.  This fit is only possible if the two indexings agree.
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  skip_on_cran()
  local_nn()
  d <- .mdvData()
  d$DV[8] <- NA_real_
  set.seed(7)
  fit <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(.mdvModel("lnorm"), d, est = "focei",
                        nlmixr2est::foceiControl(print = 0L),
                        nn = nnControl(mode = "iter", rounds = 2L, wSteps = 1L,
                                       lr = 0.05))))
  expect_equal(fit$nnSched$cotangent, "exact")
  expect_true(all(is.finite(fit$nnWeights)))
})

test_that("a non-finite weight gradient is refused, not stepped", {
  ## The backstop for the whole class: one torch step against NaN makes every
  ## weight NaN for the rest of the fit, and nothing downstream can tell.
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  local_nn()
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
  set.seed(9)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(.mdvModel())))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
  for (net in aug$nets) {
    nnTorchInit(net$id, net$K, net$H, act = net$act)
    nnTorchSetWeights(net$id, rep(0.2, length(net$gIdx)))
  }
  on.exit(for (net in aug$nets) tryCatch(nnTorchFree(net$id), silent = TRUE), add = TRUE)
  d <- .mdvData(2L)
  obs <- rep(TRUE, nrow(d))
  dv <- d$DV
  dv[2L] <- NA_real_                       # an NA the caller failed to exclude
  ws <- .nnWeightStepper(aug, d, "ID", "TIME", obs, dv,
                         stats::setNames(rep(0, aug$nW), aug$weights))
  expect_error(ws(stats::setNames(numeric(0), character(0)),
                  list(add = 0.2, prop = 0), c(lk = -1.4, add.sd = 0.2)),
               "not finite")
  ## and the weights were left alone
  expect_true(all(is.finite(nnTorchWeights(aug$nets[[1L]]$id))))
})
