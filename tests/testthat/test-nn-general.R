## Generalized input dimension (K>2) and the added activations (GELU, SiLU).
## Reference MLP: y = W2 %*% act(W1 %*% x + b1) + b2, with analytic input
## gradient/Hessian, compared against the compiled nn<K> family and the symbolic
## derivative chain.

## activation + its first two derivatives (must match src/nnEval.c)
.actFns <- list(
  softplus = list(f = function(z) ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z))),
                  d1 = function(z) 1 / (1 + exp(-z)),
                  d2 = function(z) { s <- 1 / (1 + exp(-z)); s * (1 - s) }),
  tanh = list(f = tanh, d1 = function(z) 1 - tanh(z)^2,
              d2 = function(z) { t <- tanh(z); -2 * t * (1 - t^2) }),
  gelu = list(f = function(z) z * pnorm(z),
              d1 = function(z) pnorm(z) + z * dnorm(z),
              d2 = function(z) dnorm(z) * (2 - z^2)),
  silu = list(f = function(z) z / (1 + exp(-z)),
              d1 = function(z) { s <- 1 / (1 + exp(-z)); s * (1 + z * (1 - s)) },
              d2 = function(z) { s <- 1 / (1 + exp(-z)); s1 <- s * (1 - s);
                                 2 * s1 + z * s1 * (1 - 2 * s) }))

.mkRef <- function(W1, b1, W2, b2, act) {
  a <- .actFns[[act]]; H <- nrow(W1); K <- ncol(W1)
  list(
    f = function(x) as.numeric(W2 %*% a$f(W1 %*% x + b1) + b2),
    g = function(x) { z <- as.numeric(W1 %*% x + b1)
      vapply(seq_len(K), function(m) sum(W2 * a$d1(z) * W1[, m]), numeric(1)) },
    h = function(x) { z <- as.numeric(W1 %*% x + b1); Hm <- matrix(0, K, K)
      for (j in seq_len(H)) Hm <- Hm + W2[j] * a$d2(z[j]) * outer(W1[j, ], W1[j, ])
      Hm })
}

test_that("K=3 network forward/gradient/Hessian match analytic references", {
  skip_if_not_installed("rxode2")
  set.seed(21); K <- 3L; H <- 4L; act <- "softplus"
  W1 <- matrix(rnorm(H * K), H, K); b1 <- rnorm(H)
  W2 <- matrix(rnorm(H), 1, H);     b2 <- rnorm(1)
  ref <- .mkRef(W1, b1, W2, b2, act)
  wnames <- nnWeightLayout(0, K, H)
  wvals <- c(as.vector(t(W1)), b1, as.vector(W2), b2); names(wvals) <- wnames

  mtext <- paste0(
    "param(", paste(wnames, collapse = ","), ")\n",
    "f <- nn3(0, x1, x2, x3)\n",
    "g1 <- nn3_d1(0,x1,x2,x3)\n g2 <- nn3_d2(0,x1,x2,x3)\n g3 <- nn3_d3(0,x1,x2,x3)\n",
    "h12 <- nn3_d1_d2(0,x1,x2,x3)\n h33 <- nn3_d3_d3(0,x1,x2,x3)\n d/dt(A) <- 0\n")
  m <- rxode2::rxode2(mtext)
  base <- match(wnames[1], rxode2::rxModelVars(m)$params) - 1L
  nnSetMeta(0, base, K, H, act); on.exit(nnClearMeta(), add = TRUE)

  grid <- expand.grid(x1 = c(-1, 0.5), x2 = c(0.2, 1.3), x3 = c(-0.7, 0.9))
  ev <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i)
    data.frame(id = i, time = 0, x1 = grid$x1[i], x2 = grid$x2[i], x3 = grid$x3[i],
               amt = 0, evid = 0)))
  s <- rxode2::rxSolve(m, ev, params = wvals, returnType = "data.frame",
                       covsInterpolation = "locf")
  for (i in seq_len(nrow(grid))) {
    x <- c(grid$x1[i], grid$x2[i], grid$x3[i]); row <- s[s$id == i, ][1, ]
    expect_equal(row$f, ref$f(x), tolerance = 1e-10)
    g <- ref$g(x)
    expect_equal(c(row$g1, row$g2, row$g3), g, tolerance = 1e-10)
    Hm <- ref$h(x)
    expect_equal(row$h12, Hm[1, 2], tolerance = 1e-10)
    expect_equal(row$h33, Hm[3, 3], tolerance = 1e-10)
  }
})

test_that("K=3 symbolic derivative chain expands to compiled symbols", {
  skip_if_not_installed("rxode2")
  expect_equal(rxode2::rxFromSE("Derivative(nn3(0,x1,x2,x3),x2)"), "nn3_d2(0,x1,x2,x3)")
  expect_equal(rxode2::rxFromSE("Derivative(nn3(0,x1,x2,x3),x1,x3)"), "nn3_d1_d3(0,x1,x2,x3)")
})

test_that("GELU and SiLU compiled derivatives match analytic (via nn1)", {
  skip_if_not_installed("rxode2")
  for (act in c("gelu", "silu")) {
    set.seed(4); H <- 5L; K <- 1L
    W1 <- matrix(rnorm(H), H, 1); b1 <- rnorm(H)
    W2 <- matrix(rnorm(H), 1, H); b2 <- rnorm(1)
    ref <- .mkRef(W1, b1, W2, b2, act)
    wnames <- nnWeightLayout(0, K, H)
    wvals <- c(as.vector(t(W1)), b1, as.vector(W2), b2); names(wvals) <- wnames
    m <- rxode2::rxode2(paste0(
      "param(", paste(wnames, collapse = ","), ")\n",
      "f <- nn1(0,x1)\n g <- nn1_d1(0,x1)\n hh <- nn1_d1_d1(0,x1)\n d/dt(A) <- 0\n"))
    base <- match(wnames[1], rxode2::rxModelVars(m)$params) - 1L
    nnSetMeta(0, base, K, H, act)
    xs <- c(-1.5, 0.3, 2.0)
    ev <- do.call(rbind, lapply(seq_along(xs), function(i)
      data.frame(id = i, time = 0, x1 = xs[i], amt = 0, evid = 0)))
    s <- rxode2::rxSolve(m, ev, params = wvals, returnType = "data.frame",
                         covsInterpolation = "locf")
    for (i in seq_along(xs)) {
      row <- s[s$id == i, ][1, ]
      expect_equal(row$f, ref$f(xs[i]), tolerance = 1e-9, info = act)
      expect_equal(row$g, ref$g(xs[i])[1], tolerance = 1e-9, info = act)
      expect_equal(row$hh, ref$h(xs[i])[1, 1], tolerance = 1e-9, info = act)
    }
    nnClearMeta()
  }
})

test_that("torch module matches the compiled forward for gelu/silu", {
  skip_if_no_torch()
  for (act in c("gelu", "silu")) {
    id <- 0L; K <- 2L; H <- 4L
    nnTorchInit(id, K, H, act = act, seed = 8)
    w <- nnTorchWeights(id)
    ## reconstruct the reference from the module weights
    W1 <- matrix(w[1:(H * K)], H, K, byrow = TRUE); b1 <- w[(H * K + 1):(H * K + H)]
    W2 <- matrix(w[(H * K + H + 1):(H * K + 2 * H)], 1, H); b2 <- w[H * K + 2 * H + 1]
    ref <- .mkRef(W1, b1, W2, b2, act)
    for (x in list(c(0.5, -0.3), c(-1, 2))) {
      expect_equal(nnTorchForward(id, x), ref$f(x), tolerance = 1e-6, info = act)
    }
    nnTorchFree(id)
  }
})
