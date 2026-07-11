## Phase 2 + par-loader hook: the user-facing nn() code generator, base-index
## resolution from the solve parameter order, and the rxode2 par-loader hook
## that injects externally-owned (torch) weights into par_ptr each solve.

test_that("nn() expands to a compiled call, injects param(), seeds iniDf", {
  skip_if_not_installed("rxode2")
  set.seed(3)
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, n_hidden = 4, act = "softplus"); d/dt(A) <- -p * A })
  }
  ui <- rxode2::rxode2(mod)
  code <- paste(vapply(ui$lstExpr, deparse1, character(1)), collapse = "\n")
  expect_match(code, "nn1(0, x)", fixed = TRUE)     # opaque compiled call
  expect_match(code, "param(", fixed = TRUE)         # weight slots reserved
  wnames <- nnWeightLayout(0, 1L, 4L)
  expect_true(all(wnames %in% ui$iniDf$name))        # weights present as thetas
  expect_length(wnames, 1L * 4L + 2L * 4L + 1L)
})

test_that("nnUpdate resolves the base from solve (ntheta) order", {
  skip_if_not_installed("rxode2")
  set.seed(3)
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, n_hidden = 4, act = "softplus"); d/dt(A) <- -p * A })
  }
  ui <- rxode2::rxode2(mod)
  info <- nnUpdate(ui)
  on.exit(nnClearMeta(), add = TRUE)
  expect_equal(nrow(info), 1L)
  ## p is ntheta 1, weights follow -> block starts at par_ptr index 1
  expect_equal(info$base, 1L)
  expect_equal(info$K, 1L); expect_equal(info$H, 4L)
})

test_that("par-loader hook injects buffer weights into par_ptr each solve", {
  skip_if_not_installed("rxode2")
  set.seed(11)
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, n_hidden = 4, act = "softplus"); d/dt(A) <- -p * A })
  }
  ui <- rxode2::rxode2(mod)
  nnUpdate(ui)
  on.exit(nnClearMeta(), add = TRUE)
  H <- 4L
  sp <- function(z) ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
  refF <- function(w, x) {
    W1 <- matrix(w[1:H], H, 1); b1 <- w[(H+1):(2*H)]
    W2 <- matrix(w[(2*H+1):(3*H)], 1, H); b2 <- w[3*H+1]
    as.numeric(W2 %*% sp(W1 * x + b1) + b2)
  }
  ev <- data.frame(id = 1, time = 0, x = 0.5, amt = 0, evid = 0)
  solveY <- function() rxode2::rxSolve(ui, ev, returnType = "data.frame",
                                       covsInterpolation = "locf")$y[1]

  ## seeded (ini) weights reproduce the reference
  wIni <- unname(stats::setNames(ui$iniDf$est, ui$iniDf$name)[nnWeightLayout(0, 1L, H)])
  expect_equal(solveY(), refF(wIni, 0.5), tolerance = 1e-10)

  ## overriding the buffer changes the solve to the new weights
  set.seed(999); w2 <- rnorm(3 * H + 1, sd = 0.3)
  nnSetWeights(0, w2)
  y1 <- solveY()
  expect_equal(y1, refF(w2, 0.5), tolerance = 1e-10)
  expect_gt(abs(y1 - refF(wIni, 0.5)), 1e-6)          # genuinely used the buffer
})
