## Phase 1: single-hidden-layer MLP evaluated inside an rxode2 ODE, with weights
## read from the solve parameter vector.  Validates the compiled forward value
## and the analytic input gradient/Hessian, and that the derivative chain is
## wired through rxode2's symbolic engine to the compiled derivative symbols.

test_that("nn functions register in rxode2 as thread-safe compiled functions", {
  skip_if_not_installed("rxode2")
  tr <- rxode2::rxode2parseGetTranslation()
  nn <- tr[tr$package == "rxode2nn", , drop = FALSE]
  expect_true(all(c("nn1", "nn2", "nn2_d1", "nn2_d2",
                    "nn2_d1_d1", "nn2_d1_d2", "nn2_d2_d2") %in% nn$rxFun))
  expect_true(all(nn$threadSafe == 1L))
  expect_true(all(nn$argMin == nn$argMax))
  expect_true("nn2" %in% rxode2::rxSupportedFuns())
})

test_that("symbolic derivative chain expands to compiled derivative symbols", {
  skip_if_not_installed("rxode2")
  expect_equal(rxode2::rxFromSE("Derivative(nn2(0,x1,x2),x1)"), "nn2_d1(0,x1,x2)")
  expect_equal(rxode2::rxFromSE("Derivative(nn2(0,x1,x2),x2)"), "nn2_d2(0,x1,x2)")
  expect_equal(rxode2::rxFromSE("Derivative(nn2(0,x1,x2),x1,x1)"), "nn2_d1_d1(0,x1,x2)")
  expect_equal(rxode2::rxFromSE("Derivative(nn2(0,x1,x2),x1,x2)"), "nn2_d1_d2(0,x1,x2)")
  expect_equal(rxode2::rxFromSE("Derivative(nn2(0,x1,x2),x2,x2)"), "nn2_d2_d2(0,x1,x2)")
  expect_equal(rxode2::rxFromSE("Derivative(nn1(0,x1),x1)"), "nn1_d1(0,x1)")
  expect_equal(rxode2::rxFromSE("Derivative(nn1(0,x1),x1,x1)"), "nn1_d1_d1(0,x1)")
})

test_that("compiled forward, gradient and Hessian match analytic references", {
  skip_if_not_installed("rxode2")
  set.seed(42)
  K <- 2L; H <- 3L
  actFn  <- function(z) ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
  actD2  <- function(z) { s <- 1 / (1 + exp(-z)); s * (1 - s) }
  W1 <- matrix(rnorm(H * K), H, K); b1 <- rnorm(H)
  W2 <- matrix(rnorm(H), 1, H);     b2 <- rnorm(1)
  wnames <- nnWeightLayout(0, K, H)
  wvals  <- c(as.vector(t(W1)), b1, as.vector(W2), b2)
  names(wvals) <- wnames

  refF <- function(x) as.numeric(W2 %*% actFn(W1 %*% x + b1) + b2)
  refH <- function(x) {
    z <- as.numeric(W1 %*% x + b1); Hm <- matrix(0, K, K)
    for (j in seq_len(H)) Hm <- Hm + W2[j] * actD2(z[j]) * outer(W1[j, ], W1[j, ])
    Hm
  }

  mtext <- paste0(
    "param(", paste(wnames, collapse = ","), ")\n",
    "f <- nn2(0, x1, x2)\n",
    "g1 <- nn2_d1(0, x1, x2)\n  g2 <- nn2_d2(0, x1, x2)\n",
    "h11 <- nn2_d1_d1(0, x1, x2)\n  h12 <- nn2_d1_d2(0, x1, x2)\n",
    "h22 <- nn2_d2_d2(0, x1, x2)\n  d/dt(A) <- 0\n")
  m <- rxode2::rxode2(mtext)
  params <- rxode2::rxModelVars(m)$params
  base <- match(wnames[1], params) - 1L
  expect_true(all(diff(match(wnames, params)) == 1L))   # contiguous block
  nnSetMeta(id = 0, base = base, K = K, H = H, act = "softplus")
  on.exit(nnClearMeta(), add = TRUE)

  grid <- expand.grid(x1 = c(-1, 0, 0.7, 2), x2 = c(-0.5, 0.3, 1.5))
  ev <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i)
    data.frame(id = i, time = 0, x1 = grid$x1[i], x2 = grid$x2[i], amt = 0, evid = 0)))
  s <- rxode2::rxSolve(m, ev, params = wvals, returnType = "data.frame",
                       covsInterpolation = "locf")

  for (i in seq_len(nrow(grid))) {
    x <- c(grid$x1[i], grid$x2[i]); row <- s[s$id == i, ][1, ]
    expect_equal(row$f, refF(x), tolerance = 1e-10)
    g <- numDeriv::grad(refF, x)
    expect_equal(c(row$g1, row$g2), g, tolerance = 1e-6)
    Hn <- refH(x)
    expect_equal(c(row$h11, row$h12, row$h22),
                 c(Hn[1, 1], Hn[1, 2], Hn[2, 2]), tolerance = 1e-12)
  }
})
