## The embedded NN learns genuinely nonlinear dynamics it was never told.  Data is
## simulated from Michaelis-Menten (saturable) elimination d/dt(c) = -Vmax*c/(Km+c);
## the NN-ODE learns the concentration-dependent rate g(c) via forward-sensitivity
## gradients + the torch optimizer (no finite differences).  It should recover the
## MM rate Vmax/(Km+c).  A sigmoid transform bounds the rate (keeps the solve stable
## and exercises nnAugmentModel's dR/dg through a nonlinear transform).

test_that("nnTrain recovers a Michaelis-Menten rate (nonlinear NN-ODE, torch grads)", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)

  set.seed(1)
  Vmax <- 2.0; Km <- 3.0
  mm <- rxode2::rxode2("d/dt(centr) = -Vmax*centr/(Km+centr)")
  data <- do.call(rbind, lapply(1:5, function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 3, 4, 6, 8),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
           params = c(Vmax = Vmax, Km = Km), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr + rnorm(nrow(s), 0, 0.1)))
  }))

  K <- 1L; H <- 6L; wnm <- nnWeightLayout(0L, K, H)
  model <- paste0("param(", paste(wnm, collapse = ","), ")\n",
                  "g = nn1(0, centr)\n",
                  "d/dt(centr) = -(1.0/(1.0+exp(-g)))*centr\n")
  fit <- nnTrain(model, data, pred = "centr", nnId = 0L, K = K, H = H, act = "tanh",
                 optimizer = "adam", lr = 0.02, iter = 150, seed = 5, estSigma = TRUE)

  expect_gt(tail(fit$llTrace, 1), fit$llTrace[1] + 50)   # LL improves a lot
  nnSetWeights(0L, fit$weights)
  rate <- function(cc) 1 / (1 + exp(-nn1(0L, cc)))
  ## recovers the MM rate to within ~0.1 across the concentration range
  for (cc in c(2, 5, 8)) expect_lt(abs(rate(cc) - Vmax / (Km + cc)), 0.1)
})
