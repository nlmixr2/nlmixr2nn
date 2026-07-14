## nnTrain(): pooled multi-subject NN-in-ODE fit via forward sensitivity + torch.
## Bolus-dosed 2-subject data; the network output modulates elimination.  The fit
## must drive the pooled Gaussian log-likelihood up (the ODE-sensitivity gradient
## + torch optimizer actually train the embedded network on real event data).

test_that("nnTrain improves the pooled log-likelihood on dosed multi-subject data", {
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) skip("libtorch backend not available")

  K <- 2L; H <- 2L
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)

  wnm <- nnWeightLayout(0L, K, H)
  model <- paste0(
    sprintf("param(%s, k)\n", paste(wnm, collapse = ", ")),
    "g = nn2(0, centr, peri)\n",
    "d/dt(centr) = -g*centr\n",
    "d/dt(peri) = g*centr - k*peri")

  mk <- function(id, dv) data.frame(
    id = id, time = c(0, 0.5, 1, 2, 3, 4), evid = c(1, 0, 0, 0, 0, 0),
    cmt = 1, amt = c(10, 0, 0, 0, 0, 0), dv = c(NA, dv))
  data <- rbind(mk(1, c(7.8, 6.3, 4.1, 2.8, 1.9)),
                mk(2, c(8.4, 7.0, 4.8, 3.2, 2.3)))

  fit <- nnTrain(model, data, pred = "centr", nnId = 0L, K = K, H = H, act = "tanh",
                 params = c(k = 0.3), optimizer = "adam", lr = 0.05, iter = 30,
                 seed = 3, estSigma = TRUE)

  expect_length(fit$weights, H * K + 2 * H + 1)
  expect_true(all(is.finite(fit$llTrace)))
  expect_gt(fit$llTrace[length(fit$llTrace)], fit$llTrace[1])   # LL improved
  expect_true(fit$sigma > 0 && is.finite(fit$sigma))
})
