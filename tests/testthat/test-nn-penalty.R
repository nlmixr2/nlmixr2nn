## The weight penalty (R/nnPenalty.R).
##
## Two penalties shape weight training: `l2` shrinks the weights, `smooth`
## punishes curvature.  Almost everything here is checkable without a fit, a
## solve, or even torch -- the penalty is a pure function of the weight vector
## and a fixed grid -- so most of this file is cheap and runs everywhere.
##
## The two properties worth stating up front, because they are the ones that
## would break silently:
##
##   * lambda 0 is a STRICT no-op.  Not "close to" the unregularized gradient --
##     the identical object.  That is the escape hatch back to unpenalized
##     fitting, so it is asserted with expect_identical.
##   * the -2LL/-LL scale factor.  R/nnWeightStep.R minimizes -LL and
##     R/nnPopFit.R minimizes -2LL; the penalty is defined on -2LL.  Get that
##     wrong and one lambda means two different things inside a single fit, and
##     nothing about the fit says so.

## a two-input, three-hidden tanh net, hand-built so no model or solve is needed
.penNet <- function(K = 2L, H = 3L, act = "tanh", id = 0L) {
  .nW <- as.integer(H * K + 2L * H + 1L)
  list(id = id, K = K, H = H, act = act, nW = .nW, gIdx = seq_len(.nW),
       inputs = c("centr", "WT")[seq_len(K)])
}

.penAug <- function(nets = list(.penNet())) {
  list(nets = nets, nW = sum(vapply(nets, function(.n) .n$nW, integer(1))))
}

## a profile with a real range on every input, so grids exist
.penProfile <- function(K = 2L, scales = rep(1, K), lo = 0, hi = 1) {
  lapply(seq_len(K), function(.k) {
    list(name = paste0("x", .k), kind = "trial", scale = scales[.k],
         center = (lo + hi) / 2, range = c(lo, hi))
  })
}

.penW <- function(spec, seed = 42L) {
  set.seed(seed)
  stats::rnorm(spec$nets[[1L]]$nW)
}

## central finite difference of a scalar function of the weight vector
.penFD <- function(f, w, h = 1e-5) {
  vapply(seq_along(w), function(j) {
    .wp <- w; .wp[j] <- .wp[j] + h
    .wm <- w; .wm[j] <- .wm[j] - h
    (f(.wp) - f(.wm)) / (2 * h)
  }, numeric(1))
}

test_that("lambda 0 is a strict no-op, not an approximate one", {
  ## The escape hatch: nnControl(l2 = 0, smooth = 0) must reproduce an
  ## unregularized fit exactly, so the spec is NULL and the gradient comes back
  ## as the very same object rather than one that merely compares equal.
  .aug <- .penAug()
  expect_null(.nnPenSpec(.aug, list(.penProfile()), 0, 0))
  ## and a NULL spec is inert everywhere it is consumed
  .g <- c(1.5, -2.25, 0.5)
  expect_identical(.nnAddPen(.g, c(1, 2, 3), NULL, 1L), .g)
  expect_identical(.nnAddPen(.g, c(1, 2, 3), NULL, 2L), .g)
  expect_identical(.nnAddPenNet(.g, c(1, 2, 3), NULL, 0, 0, 1L), .g)
  expect_equal(.nnPenalty(NULL, c(1, 2, 3))$value, 0)
  expect_equal(.nnPenalty(NULL, c(1, 2, 3))$grad, c(0, 0, 0))
})

test_that("the inferred defaults turn regularization ON", {
  ## The package is unreleased, so the default is the right one rather than the
  ## historical one.  This guards against the defaults reverting to 0 through a
  ## merge, which would be invisible -- every test would still pass.
  skip_if_not_installed("rxode2")
  ## a local fixture rather than test-nn-schedule.R's: a filtered run loads only
  ## this file, so anything defined over there is not in scope here
  .mod <- function() {
    ini({ add.sd <- 0.3; eta.nn ~ 0.2 })
    model({
      g <- nn(centr, eta.nn, n_hidden = 3L, act = "tanh")
      d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
      centr ~ add(add.sd)
    })
  }
  .env <- new.env(parent = emptyenv())
  .env$ui <- suppressWarnings(rxode2::rxode2(.mod))
  .env$data <- data.frame(id = 1, time = c(0, 1, 2, 4), evid = c(1, 0, 0, 0),
                          amt = c(10, 0, 0, 0), cmt = 1, dv = c(NA, 8, 6, 4))
  .env$control <- list(maxOuterIterations = 5L)
  class(.env) <- c("focei", "environment")
  .s <- .nnInferSched(.env, nW = 13L, hasTrained = FALSE)
  expect_true(is.numeric(.s$l2) && .s$l2 > 0)
  expect_true(is.numeric(.s$smooth) && .s$smooth > 0)
  ## and an explicit 0 survives the overlay, so the escape hatch is reachable
  .r <- .nnResolveSched(nnControl(l2 = 0, smooth = 0), .s)
  expect_equal(.r$l2, 0)
  expect_equal(.r$smooth, 0)
})

test_that("the -2LL/-LL scale factor is right at both gradient sites", {
  ## R/nnWeightStep.R hands .nnAddPen a -LL-scale gradient (scale = 1); the
  ## population fit in R/nnPopFit.R works in -2LL (scale = 2).  Doubling the
  ## former must give the latter, or one lambda means two different things in
  ## one fit.
  .aug <- .penAug()
  .pen <- .nnPenSpec(.aug, list(.penProfile()), l2 = 0.1, smooth = 0.05)
  .w <- .penW(.pen)
  .g <- seq_along(.w) / 10
  expect_equal(2 * .nnAddPen(.g, .w, .pen, 1L),
               .nnAddPen(2 * .g, .w, .pen, 2L), tolerance = 1e-12)
  ## negative control: the same scale on both sides must NOT satisfy it, or the
  ## assertion above would hold for a penalty that was silently ignored
  expect_false(isTRUE(all.equal(2 * .nnAddPen(.g, .w, .pen, 1L),
                                .nnAddPen(2 * .g, .w, .pen, 1L))))
})

test_that("the L2 gradient matches a finite difference, and biases are free", {
  .aug <- .penAug()
  .pen <- .nnPenSpec(.aug, list(.penProfile()), l2 = 0.25, smooth = 0)
  .w <- .penW(.pen)
  .ana <- .nnPenalty(.pen, .w)$grad
  .fd <- .penFD(function(ww) .nnPenalty(.pen, ww)$value, .w)
  expect_equal(.ana, .fd, tolerance = 1e-7)

  ## biases stay free: the network can still move its output level without
  ## paying for it, so b1 and b2 carry exactly no gradient
  K <- 2L; H <- 3L
  .b1 <- (H * K + 1L):(H * K + H)
  .b2 <- H * K + 2L * H + 1L
  expect_equal(.ana[c(.b1, .b2)], rep(0, length(.b1) + 1L))
  expect_true(all(.ana[seq_len(H * K)] != 0))     # ...but W1 does
})

test_that("L2 penalizes the EFFECTIVE first-layer weight, not the raw one", {
  ## Input scaling is folded into W1 and then discarded (.nnRescaleW1), so an
  ## input of typical magnitude s carries a W1 column s times smaller at equal
  ## functional effect.  Penalizing raw W1 would let a large-magnitude input's
  ## weights grow essentially unpunished -- by s^2 -- which is the bug this test
  ## exists for.
  K <- 2L; H <- 3L
  .aug <- .penAug(list(.penNet(K, H)))
  .s <- 50
  .penUnit <- .nnPenSpec(.aug, list(.penProfile(K, scales = c(1, 1))), 0.1, 0)
  .penBig <- .nnPenSpec(.aug, list(.penProfile(K, scales = c(.s, 1))), 0.1, 0)
  .w <- .penW(.penUnit)
  ## the same network expressed against a scaled first input: W1's first column
  ## divided by s, exactly as .nnRescaleW1 would leave it
  .wScaled <- .nnRescaleW1(.w, K, H, c(.s, 1))
  expect_equal(.nnPenalty(.penUnit, .w)$value,
               .nnPenalty(.penBig, .wScaled)$value, tolerance = 1e-10)
  ## negative control: the naive raw-weight penalty does not have this property
  expect_false(isTRUE(all.equal(sum(.w^2), sum(.wScaled^2))))
})

test_that("the curvature gradient matches a finite difference", {
  skip_if_not_installed("rxode2")
  .aug <- .penAug()
  .pen <- .nnPenSpec(.aug, list(.penProfile(lo = -2, hi = 2)), l2 = 0, smooth = 0.3)
  .w <- .penW(.pen)
  .ana <- .nnPenalty(.pen, .w)$grad
  .fd <- .penFD(function(ww) .nnPenalty(.pen, ww)$value, .w)
  expect_equal(.ana, .fd, tolerance = 1e-6)
  expect_gt(.nnPenalty(.pen, .w)$value, 0)

  ## Negative control (the pattern from test-nn-llgrad.R): an assembly that
  ## drops the factor of 2 on d(C^2)/dC must FAIL the same comparison, so the
  ## test above cannot be passing for the wrong reason.
  expect_false(isTRUE(all.equal(.ana / 2, .fd, tolerance = 1e-6)))
})

test_that("curvature charges for wiggles, not for slope", {
  ## Penalizing the SECOND derivative is the whole point: a monotone network
  ## should be able to have any slope it likes for free, while an oscillating
  ## one pays.
  skip_if_not_installed("rxode2")
  K <- 1L; H <- 4L
  .aug <- .penAug(list(.penNet(K, H, act = "tanh")))
  .pen <- .nnPenSpec(.aug, list(.penProfile(K, lo = -3, hi = 3)), 0, 1)
  .nW <- .aug$nets[[1L]]$nW

  ## a flat network: every second-layer weight zero, so the output is constant
  .flat <- numeric(.nW)
  .flat[(H * K + H + 1L):(H * K + 2L * H)] <- 0
  expect_equal(.nnPenalty(.pen, .flat)$value, 0, tolerance = 1e-12)

  ## a near-linear network (tiny first-layer weights keep tanh in its linear
  ## region) versus a wiggly one of the SAME weight norm
  .lin <- numeric(.nW)
  .lin[seq_len(H * K)] <- 0.01
  .lin[(H * K + H + 1L):(H * K + 2L * H)] <- 1
  .wig <- .lin
  .wig[seq_len(H * K)] <- c(6, -6, 6, -6)[seq_len(H * K)]
  .wig[(H * K + 1L):(H * K + H)] <- c(-4, 0, 4, 8)[seq_len(H)]
  expect_gt(.nnPenalty(.pen, .wig)$value, .nnPenalty(.pen, .lin)$value)
})

test_that("an input with no resolvable range is held fixed, not swept", {
  ## An eta is not a solve column and its EBEs move every round, so there is no
  ## honest range to sweep it over; a compound expression such as nn(centr/Vc)
  ## has no single source at all.  Both are pinned while the others are gridded.
  .prof <- list(
    list(name = "centr", kind = "trial", scale = 250, center = 250, range = c(10, 500)),
    list(name = "eta_nn", kind = "eta", scale = 1, center = 0, range = NULL))
  .g <- .nnPenGrids(.prof, K = 2L)
  expect_length(.g, 1L)                       # only centr is gridable
  expect_equal(nrow(.g[[1L]]), .nnPenM)
  expect_true(all(.g[[1L]][, 2L] == 0))       # the eta pinned at 0
  expect_equal(range(.g[[1L]][, 1L]), c(10, 500))

  ## and a net whose inputs are ALL unrangeable simply has no curvature term
  .none <- list(list(name = "eta_nn", kind = "eta", scale = 1, center = 0, range = NULL))
  expect_length(.nnPenGrids(.none, K = 1L), 0L)
})

test_that("the input profile reads ranges from the solve and the data", {
  ## the profile is what feeds both the grid and the L2 multiplier, so its
  ## source priority has to be the same one input scaling uses
  .trial <- data.frame(centr = c(10, 100, 250, 400, 500))
  .data <- data.frame(id = 1, time = 0:4, WT = c(60, 70, 80, 90, 100))
  .p <- .nnInputProfile(c("centr", "WT", "eta_nn", "centr/Vc"), .trial, .data,
                        etaNames = "eta_nn")
  expect_equal(.p[[1L]]$kind, "trial")
  expect_false(is.null(.p[[1L]]$range))
  expect_equal(.p[[2L]]$kind, "data")
  expect_equal(.p[[2L]]$center, 80)
  expect_equal(.p[[3L]]$kind, "eta")
  expect_null(.p[[3L]]$range)                 # an eta is never gridded
  expect_equal(.p[[4L]]$kind, "expr")
  expect_null(.p[[4L]]$range)
})

test_that("a penalty spec survives a missing trial solve by falling back to l2", {
  ## The trial solve can fail (a bad initial estimate, an unsolvable model).
  ## That must cost the curvature term, not the fit: l2 needs no grid at all.
  .aug <- .penAug()
  .pen <- .nnPenSpec(.aug, NULL, l2 = 0.1, smooth = 0.1)
  expect_false(is.null(.pen))
  expect_true(all(vapply(.pen$nets, function(.s) length(.s$grids) == 0L, logical(1))))
  .w <- .penW(.pen)
  ## the value is then purely the l2 term, and still finite and differentiable
  expect_equal(.nnPenalty(.pen, .w)$grad,
               .penFD(function(ww) .nnPenalty(.pen, ww)$value, .w),
               tolerance = 1e-7)
})

test_that("each network is penalized against its own weight block", {
  ## a model with two nets of different shapes: the global gradient must place
  ## each net's contribution at that net's own global indices and nowhere else
  .n0 <- .penNet(K = 2L, H = 3L, id = 0L)
  .n1 <- .penNet(K = 1L, H = 2L, id = 1L)
  .n1$gIdx <- .n0$nW + seq_len(.n1$nW)
  .aug <- list(nets = list(.n0, .n1), nW = .n0$nW + .n1$nW)
  .pen <- .nnPenSpec(.aug, list(.penProfile(2L), .penProfile(1L)), 0.2, 0)
  set.seed(3)
  .w <- stats::rnorm(.aug$nW)
  .g <- .nnPenalty(.pen, .w)$grad
  expect_length(.g, .aug$nW)
  expect_equal(.g, .penFD(function(ww) .nnPenalty(.pen, ww)$value, .w),
               tolerance = 1e-7)
  ## zeroing one net's weights must leave the other net's gradient untouched
  .w2 <- .w; .w2[.n1$gIdx] <- 0
  expect_equal(.nnPenalty(.pen, .w2)$grad[.n0$gIdx], .g[.n0$gIdx])
})
