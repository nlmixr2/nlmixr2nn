## Inner-block individual weights (pop = FALSE): W_i = f(lW, etaW).  The inner
## weight-injection hook (nnInnerWeight) writes these into each subject's par_ptr
## during a FOCEI fit; here we validate the W_i arithmetic directly (no solve).

test_that("nnSetIndividual computes W_i = lW*exp(etaW) [prop] and lW+etaW [add]", {
  K <- 2L; H <- 1L; nW <- H * K + 2L * H + 1L      # 5 weights
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = K, H = H, act = "tanh")
  on.exit(nnClearMeta(), add = TRUE)
  lW <- as.double(seq_len(nW)) / 10
  nnSetWeights(0L, lW)

  ## weight-etas start at eta index 3 (0-based); the eta vector has 3 other etas
  ## first, then the nW weight-etas
  set.seed(2)
  etaW <- rnorm(nW, 0, 0.1)
  eta  <- c(0.1, -0.2, 0.05, etaW)                 # length 3 + nW = 8

  nnSetIndividual(0L, etaBase = 3L, etaModel = "prop")
  Wi <- .Call("_nlmixr2nn_nnIndividualWeightsW", 0L, as.double(eta))
  expect_length(Wi, nW)
  expect_equal(Wi, lW * exp(etaW), tolerance = 1e-12)

  nnSetIndividual(0L, etaBase = 3L, etaModel = "add")
  Wi2 <- .Call("_nlmixr2nn_nnIndividualWeightsW", 0L, as.double(eta))
  expect_equal(Wi2, lW + etaW, tolerance = 1e-12)

  ## a population network (no nnSetIndividual) yields no individual weights
  nnClearMeta(); nnSetMeta(1L, base = 0L, K = K, H = H, act = "tanh")
  nnSetWeights(1L, lW)
  expect_length(.Call("_nlmixr2nn_nnIndividualWeightsW", 1L, as.double(eta)), 0L)
})
