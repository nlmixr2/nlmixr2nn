## Input scaling.
##
## Networks are fed raw model quantities, which in pharmacometrics are nothing
## like order 1.  Without scaling a tanh network on a PK-scale state starts
## fully saturated: its input derivative collapses to ~1e-8, and that derivative
## is both the FOCEi sensitivity of a latent eta input and the weight-training
## signal.  These tests pin the scaling down at the unit level and then check the
## consequence that actually matters.

test_that(".nnTypicalScale describes magnitude robustly", {
  ## zeros are excluded: a compartment sits at exactly 0 before its first dose,
  ## and counting those would drag every scale toward 0
  expect_equal(.nnTypicalScale(c(0, 0, 2, 4)), 3)
  ## degenerate inputs must never produce a 0 or non-finite scale, because the
  ## scale is a divisor
  expect_equal(.nnTypicalScale(numeric(0)), 1)
  expect_equal(.nnTypicalScale(c(0, 0, 0)), 1)
  expect_equal(.nnTypicalScale(c(NA, NaN, Inf)), 1)
  ## sign does not matter, only magnitude
  expect_equal(.nnTypicalScale(c(-4, -2)), 3)
})

test_that(".nnRescaleW1 divides only the input side, and only per input", {
  K <- 2L; H <- 3L
  nW <- H * K + 2L * H + 1L
  w <- as.numeric(seq_len(nW))
  out <- .nnRescaleW1(w, K, H, c(2, 4))

  ## W1 is H*K row-major: hidden unit j, input k at index j*K + k
  for (j in seq_len(H) - 1L) {
    expect_equal(out[j * K + 1L], w[j * K + 1L] / 2)
    expect_equal(out[j * K + 2L], w[j * K + 2L] / 4)
  }
  ## b1, W2 and b2 are untouched -- scaling the input side is the whole point
  expect_equal(out[(H * K + 1L):nW], w[(H * K + 1L):nW])

  ## a unit scale is a no-op, and a malformed scale is refused rather than
  ## silently producing Inf/NaN weights
  expect_equal(.nnRescaleW1(w, K, H, c(1, 1)), w)
  expect_equal(.nnRescaleW1(w, K, H, c(1)), w)
  expect_false(anyNA(.nnRescaleW1(w, K, H, c(0, -1))))
})

test_that("scaling the first layer is equivalent to scaling the input", {
  ## the identity the whole approach rests on:
  ##   f(x/s) with weights W1  ==  f(x) with weights W1/s
  ## if this ever stops holding, scaling silently changes the model
  skip_if_not_installed("rxode2")
  K <- 2L; H <- 4L
  set.seed(11)
  w <- .nnInitDraw(K, H, "tanh", "ude", 0.1)
  s <- c(7, 0.5)
  x <- matrix(c(3.5, 12, -2, 0.25), ncol = K)

  act <- .nnActCode[["tanh"]]
  scaledInput <- .Call(`_nlmixr2nn_nnForwardW`, K, H, act, w,
                       sweep(x, 2L, s, "/"))
  scaledWeights <- .Call(`_nlmixr2nn_nnForwardW`, K, H, act,
                         .nnRescaleW1(w, K, H, s), x)
  expect_equal(scaledInput, scaledWeights, tolerance = 1e-12)
})

test_that("a latent eta input is never rescaled", {
  ## an eta is already standardized by its omega; dividing it out would change
  ## what the random effect means
  d <- data.frame(id = 1, time = 0:3, WT = c(70, 80, 90, 100))
  trial <- c(centr = 250)
  s <- .nnScalesForNet(c("centr", "eta.nn"), trial, d, etaNames = "eta.nn")
  expect_equal(s[2L], 1)
  expect_equal(s[1L], 250)
})

test_that("scales come from the solved model first, then the data", {
  d <- data.frame(id = 1, time = 0:3, WT = c(70, 80, 90, 100))
  trial <- c(centr = 250)
  ## a state is only knowable from the solve
  expect_equal(.nnScalesForNet("centr", trial, d, character(0)), 250)
  ## a covariate is read off the data
  expect_equal(.nnScalesForNet("WT", trial, d, character(0)), 85)
  ## an input that is neither (a compound expression) is left alone rather than
  ## guessed at
  expect_equal(.nnScalesForNet("centr/Vc", trial, d, character(0)), 1)
})

test_that("a tanh network on PK-scale data keeps a usable input gradient", {
  ## the end-to-end consequence: on an unscaled network this derivative is ~1e-8
  ## and the latent eta is unidentifiable.
  skip_if_not_installed("rxode2")
  m <- function() {
    model({
      y  <- nn(u, n_hidden = 5L, act = "tanh")
      dy <- nn1_d1(0, u)
      d/dt(A) <- -A * 0
    })
  }
  set.seed(5)
  ui <- suppressMessages(rxode2::rxode2(m))
  w <- nnWeights(ui)
  meta <- rxode2::rxUiDecompress(ui)$nnMeta[["0"]]

  ## emulate what the fit does: rescale by the typical input magnitude.
  ## a dosed amount decaying from 100 -- an entirely ordinary PK scale
  us <- c(1, 5, 20, 60, 100)
  scaled <- .nnRescaleW1(unname(w), meta$K, meta$H, .nnTypicalScale(us))

  grad <- function(weights, x, h = 1e-4) {
    f <- function(z) .Call(`_nlmixr2nn_nnForwardW`, meta$K, meta$H,
                           .nnActCode[[meta$act]], weights, matrix(z, ncol = 1L))
    (f(x + h) - f(x - h)) / (2 * h)
  }
  ## the claim is about the TOP of the input range: that is where an unscaled
  ## network is pinned and where a state spends the part of its trajectory the
  ## network is supposed to explain.  Taking a max over the range would report
  ## the one input that still happens to work.
  top <- max(us)
  gRaw <- abs(grad(unname(w), top))
  gScl <- abs(grad(scaled, top))

  ## unscaled is degenerate at the top of the range; scaled is a usable
  ## training/sensitivity signal there.  Stated as a ratio so the test asserts
  ## the effect rather than two hand-tuned absolute thresholds.
  expect_lt(gRaw, 1e-6)
  expect_gt(gScl / gRaw, 100)
})
