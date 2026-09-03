## Phase 2 + par-loader hook: the user-facing nn() code generator, base-index
## resolution from the solve parameter order, and the rxode2 par-loader hook
## that injects externally-owned (torch) weights into par_ptr each solve.  The
## weights are declared as COVARIATES (loader/inner-hook owned), not thetas, so
## the model assembles under FOCEI (which theta-expands and would mangle param()).

test_that("nn() expands to a compiled call and declares weights as covariates", {
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  set.seed(3)
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, nHidden = 4, act = "softplus"); d/dt(A) <- -p * A })
  }
  ui <- rxode2::rxode2(mod)
  code <- paste(vapply(ui$lstExpr, deparse1, character(1)), collapse = "\n")
  expect_match(code, "nn1(0, x)", fixed = TRUE)             # opaque compiled call
  wnames <- nnWeightLayout(0, 1L, 4L)
  ## weights are covariates (in the solve params), NOT thetas (not in iniDf)
  expect_true(all(wnames %in% rxode2::rxModelVars(ui)$params))
  expect_false(any(wnames %in% ui$iniDf$name[!is.na(ui$iniDf$ntheta)]))
  expect_length(wnames, 1L * 4L + 2L * 4L + 1L)
})

test_that("nnUpdate resolves a contiguous base from the solve parameter order", {
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  set.seed(3)
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, nHidden = 4, act = "softplus"); d/dt(A) <- -p * A })
  }
  ui <- rxode2::rxode2(mod)
  info <- nnUpdate(ui)
  on.exit(nnClearMeta(), add = TRUE)
  expect_equal(nrow(info), 1L)
  expect_gte(info$base, 0L)                                 # contiguous block resolved
  expect_equal(info$K, 1L); expect_equal(info$H, 4L)
})

test_that("a model's own weights drive the solve, and a stale buffer cannot override them", {
  skip_if_not_installed("rxode2")
  set.seed(11)
  mod <- function() {
    ini({ p <- 1 })
    model({ y <- nn(x, nHidden = 4, act = "softplus"); d/dt(A) <- -p * A })
  }
  ui <- suppressMessages(rxode2::rxode2(mod))
  on.exit(nnClearMeta(), add = TRUE)
  H <- 4L
  sp <- function(z) ifelse(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
  refF <- function(w, x) {
    W1 <- matrix(w[1:H], H, 1); b1 <- w[(H + 1):(2 * H)]
    W2 <- matrix(w[(2 * H + 1):(3 * H)], 1, H); b2 <- w[3 * H + 1]
    as.numeric(W2 %*% sp(W1 * x + b1) + b2)
  }
  ## no nnCovData(): a weight covariate is supplied by the model, not the data
  ev <- data.frame(id = 1, time = 0, x = 0.5, amt = 0, evid = 0)
  solveY <- function() rxode2::rxSolve(ui, ev, returnType = "data.frame",
                                       covsInterpolation = "locf")$y[1]

  ## the compiled evaluator reproduces an independent R MLP at the model's own
  ## parse-time weights
  w <- unname(nnWeights(ui))
  expect_equal(solveY(), refF(w, 0.5), tolerance = 1e-8)

  ## and a leftover loader buffer holding DIFFERENT weights does not change that.
  ## This is the protection that keeps a reloaded fit correct: in a fresh session
  ## the transient buffer is empty, and letting it win would zero the network.
  set.seed(999); other <- stats::rnorm(3 * H + 1, sd = 0.3)
  nnSetWeights(0, other)
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)
  expect_equal(solveY(), refF(w, 0.5), tolerance = 1e-8)
})

test_that("nn() accepts 1 to 4 inputs and rejects more", {
  skip_if_not_installed("rxode2")
  on.exit(nnClearMeta(), add = TRUE)
  for (kk in 1:4) {
    nnClearMeta()
    ins <- paste(c("centr", "WT", "AGE", "SEX")[seq_len(kk)], collapse = ", ")
    modTxt <- sprintf("function() { model({ g <- nn(%s, nHidden = 2L); d/dt(centr) <- -g*centr }) }", ins)
    ui <- eval(parse(text = paste0("rxode2::rxode2(", modTxt, ")")))
    reg <- nlmixr2nn:::.nnEnv$reg[[1L]]
    expect_equal(reg$K, kk)
    expect_equal(length(reg$weights), 2L * kk + 2L * 2L + 1L)   # H*K + 2H + 1, H=2
  }
  ## 5 inputs is rejected
  nnClearMeta()
  expect_error(
    rxode2::rxode2(function() { model({ g <- nn(a, b, c, d, e, nHidden = 2L); d/dt(a) <- -g*a }) }),
    "1 to 4 inputs")
})
