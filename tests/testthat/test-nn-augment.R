## B3: forward-sensitivity augmented NN model (nnAugmentModel).
## Validates (1) the emitted dR/dg outputs, and (2) the F_X.s variational block,
## the latter via an initial-condition sensitivity finite difference: with a unit
## IC on a weight-0 variational state and zero forcing, rx_sw_<state>_0_(t) equals
## d(state(t))/d(perturbed IC), which we finite-difference against base re-solves.
## (The forcing term b_ij = dR/dg * dg/dw is added by the dydt-force hook, wired
## in a later phase; here the variational block is exercised on its own.)

test_that("nnAugmentModel emits correct dR/dg and F_X.s variational block", {
  skip_if_not_installed("rxode2")
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = 2L, H = 1L, act = "tanh")  # nW = 5
  on.exit(nnClearMeta(), add = TRUE)
  obj <- "g = nn2(0, centr, peri)\nd/dt(centr) = -g*centr\nd/dt(peri) = g*centr - k*peri"

  aug <- nnAugmentModel(obj, H = 1L)
  ## 2 states * 5 weights = 10 variational states
  expect_equal(length(gregexpr("d/dt(rx_sw_", aug, fixed = TRUE)[[1]]), 10L)

  nnSetWeights(0L, as.double(seq_len(5)) / 10)  # fixed deterministic weights
  m <- rxode2::rxode2(aug)
  ev <- rxode2::et(seq(0, 5, by = 1))
  p <- c(k = 0.3)

  ## dR/dg outputs
  s0 <- rxode2::rxSolve(m, ev, params = p, inits = c(centr = 10, peri = 0))
  expect_equal(s0$rx_drdg_centr_, -s0$centr, tolerance = 1e-8)
  expect_equal(s0$rx_drdg_peri_,  s0$centr, tolerance = 1e-8)

  ## F_X.s block: unit IC on weight-0 variational states in the centr direction
  ## => rx_sw_centr_0_(t) = d centr(t)/d centr(0), rx_sw_peri_0_(t) = d peri(t)/d centr(0)
  sv <- rxode2::rxSolve(m, ev, params = p,
                        inits = c(centr = 10, peri = 0, rx_sw_centr_0_ = 1, rx_sw_peri_0_ = 0))
  h <- 1e-4
  sb  <- rxode2::rxSolve(m, ev, params = p, inits = c(centr = 10, peri = 0))
  sbh <- rxode2::rxSolve(m, ev, params = p, inits = c(centr = 10 + h, peri = 0))
  fdCentr <- (sbh$centr - sb$centr) / h
  fdPeri  <- (sbh$peri  - sb$peri)  / h
  expect_equal(sv$rx_sw_centr_0_, fdCentr, tolerance = 1e-3)
  expect_equal(sv$rx_sw_peri_0_,  fdPeri,  tolerance = 1e-3)
})
