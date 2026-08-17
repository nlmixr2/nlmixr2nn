## The endpoint transformation's derivative, checked against rxode2's own
## transformation rather than against a second copy of my algebra.
##
## This is the factor that closes the chain between a score reported on the
## transformed scale and sensitivities computed on the natural scale.  Getting
## it wrong does not error -- it scales the gradient per observation, which
## still trains and still looks fine.

## finite difference of rxode2's transform, so the reference is the thing the
## model actually applies
.fdJac <- function(y, lambda, transform, low, hi, h = 1e-6) {
  f <- function(z) rxode2:::.rxTransform(z, lambda = lambda, low = low,
                                         high = hi, transform = transform)
  (f(y + h) - f(y - h)) / (2 * h)
}

test_that("the transform Jacobian matches a finite difference of rxode2's transform", {
  skip_if_not_installed("rxode2")
  ## every level rxode2 defines, so this cannot drift from its table
  low <- 0; hi <- 5
  for (nm in rxode2:::.rxTransformCombineLevels) {
    lambda <- if (grepl("boxCox|yeoJohnson", nm)) 0.4 else 1
    ## rxode2 returns NA for a Box-Cox applied after a bounding transform, at
    ## every value and every lambda -- those endpoints cannot be evaluated at
    ## all, so they are refused rather than compared (asserted separately below).
    if (grepl("\\+ boxCox", nm)) next
    y <- c(0.2, 0.5, 0.9, 1.7, 3.3)
    ep <- list(transform = nm, lambda = lambda, trLow = low, trHi = hi)

    got <- .nnTransformJac(ep, y)
    ref <- .fdJac(y, lambda = lambda, transform = nm, low = low, hi = hi)
    expect_equal(got, ref, tolerance = 1e-4,
                 info = paste("transformation:", nm))
  }
})

test_that("a combined transformation chains, it does not just multiply", {
  ## "logit + yeoJohnson" is yeoJohnson(logit(y)), so the outer derivative is
  ## evaluated at logit(y).  Multiplying two derivatives both taken at y is the
  ## plausible-looking wrong answer, and is asserted against here.
  skip_if_not_installed("rxode2")
  y <- c(0.4, 1.1, 2.6); low <- 0; hi <- 5; lambda <- 0.4
  ep <- list(transform = "logit + yeoJohnson", lambda = lambda, trLow = low, trHi = hi)
  inner <- list(transform = "logit", lambda = lambda, trLow = low, trHi = hi)
  outer <- list(transform = "yeoJohnson", lambda = lambda, trLow = low, trHi = hi)

  u <- log(((y - low) / (hi - low)) / (1 - (y - low) / (hi - low)))   # logit(y)
  expect_equal(.nnTransformJac(ep, y),
               .nnTransformJac(outer, u) * .nnTransformJac(inner, y))
  ## the naive product is genuinely different, so the test is not vacuous
  expect_false(isTRUE(all.equal(.nnTransformJac(ep, y),
                                .nnTransformJac(outer, y) * .nnTransformJac(inner, y))))
})

test_that("an untransformed endpoint is identity, and is detected as needing nothing", {
  y <- c(1, 2, 3)
  expect_equal(.nnTransformJac(list(transform = "untransformed"), y), rep(1, 3))
  expect_equal(.nnTransformJac(list(), y), rep(1, 3))
  expect_false(.nnNeedsTransformJac(list(transform = "untransformed")))
  expect_false(.nnNeedsTransformJac(list()))
  expect_true(.nnNeedsTransformJac(list(transform = "lnorm")))
  expect_true(.nnNeedsTransformJac(list(transform = "logit + boxCox")))
})

test_that("a Box-Cox after a bounding transform is refused, because rxode2 cannot evaluate it", {
  skip_if_not_installed("rxode2")
  ## the premise: rxode2 really does return NA for these, so refusing is right
  ys <- c(0.5, 2.5, 4.5)
  for (nm in c("logit + boxCox", "probit + boxCox")) {
    ref <- rxode2:::.rxTransform(ys, lambda = 0.4, low = 0, high = 5, transform = nm)
    expect_true(all(is.na(ref)), info = nm)
    expect_error(.nnTransformJac(list(transform = nm, lambda = 0.4, trLow = 0, trHi = 5), ys),
                 "not supported")
  }
})

test_that("an unknown transformation is refused rather than silently treated as identity", {
  ## silently returning 1 is the exact failure this code exists to remove
  expect_error(.nnTransformJac(list(transform = "notATransform"), c(1, 2)),
               "unsupported endpoint transformation")
})

test_that("boxCox at lambda 0 is the log transform", {
  skip_if_not_installed("rxode2")
  y <- c(0.5, 1.5, 4)
  expect_equal(.nnTransformJac(list(transform = "boxCox", lambda = 0), y), 1 / y)
})
