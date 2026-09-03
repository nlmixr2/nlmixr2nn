## The weight penalty (R/nnPenalty.R).
##
## Three penalties shape weight training: `l2` shrinks the weights, `smooth`
## punishes curvature, and `kinetic` punishes how hard the network pushes along
## the realized trajectory.  All three default to OFF.  `kinetic` is a FRACTION
## of the objective -- its normalizer is frozen from the first objective a fit
## produces -- so every test below that wants it to act has to freeze it first,
## and one that does not is testing that it stays inert.
##
## Almost everything here is checkable without a fit, a solve, or even torch --
## the penalty is a pure function of the weight vector and a fixed set of input
## points -- so most of this file is cheap and runs everywhere.
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

## a profile with a real range on every input, so grids exist.  `values` is the
## realized trajectory the kinetic term reads; `vals = FALSE` drops it, which is
## what an eta or a compound expression looks like.
.penProfile <- function(K = 2L, scales = rep(1, K), lo = 0, hi = 1,
                        vals = TRUE, n = 25L) {
  lapply(seq_len(K), function(.k) {
    list(name = paste0("x", .k), kind = "trial", scale = scales[.k],
         center = (lo + hi) / 2, range = c(lo, hi),
         values = if (vals) seq(lo, hi, length.out = n) + .k / 100 else NULL)
  })
}

.penW <- function(spec, seed = 42L) {
  set.seed(seed)
  stats::rnorm(spec$nets[[1L]]$nW)
}

## Freeze the kinetic normalizer at a reference objective, the way a fit does on
## its first objective evaluation.  Returns the spec so it can be used inline.
.penFreeze <- function(spec, w, obj = 100) {
  .nnPenFreeze(spec, w, obj)
  spec
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
  expect_identical(.nnAddPenNet(.g, c(1, 2, 3), NULL, NULL, 1L), .g)
  expect_equal(.nnPenalty(NULL, c(1, 2, 3))$value, 0)
  expect_equal(.nnPenalty(NULL, c(1, 2, 3))$grad, c(0, 0, 0))
})

test_that("the inferred defaults leave every penalty OFF", {
  ## All three lambdas default to 0, and that is a decision rather than an
  ## oversight: in these models the network IS the model, so a penalty big
  ## enough to suppress invented structure also attenuates real structure.  This
  ## guards against a default drifting back to nonzero through a merge, which
  ## would be invisible -- every other test would still pass.
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
  expect_equal(.s$l2, 0)
  expect_equal(.s$smooth, 0)
  expect_equal(.s$kinetic, 0)
  ## and an explicit value survives the overlay, so each knob is reachable
  .r <- .nnResolveSched(nnControl(l2 = 0.05, smooth = 0.05, kinetic = 0.05), .s)
  expect_equal(.r$l2, 0.05)
  expect_equal(.r$smooth, 0.05)
  expect_equal(.r$kinetic, 0.05)
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


## --------------------------------------------------------------------------
## kinetic: the trajectory-realized term (R/nnPenalty.R)
## --------------------------------------------------------------------------

test_that("kinetic 0 is a strict no-op alongside the other two", {
  ## the escape hatch has to survive the third lambda: all three at 0 is still
  ## the identical-object no-op, and `kinetic` alone is enough to build a spec
  .aug <- .penAug()
  expect_null(.nnPenSpec(.aug, list(.penProfile()), 0, 0, 0))
  .k <- .nnPenSpec(.aug, list(.penProfile()), 0, 0, 0.25)
  expect_false(is.null(.k))
  ## ... and it builds ONLY what it needs: points, not a curvature grid
  expect_false(is.null(.k$nets[[1L]]$pts))
  expect_length(.k$nets[[1L]]$grids, 0L)
})

test_that("the kinetic gradient matches a finite difference", {
  .pen <- .nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.25)
  .w <- .penW(.pen)
  .penFreeze(.pen, .w)
  expect_equal(.nnPenalty(.pen, .w)$grad,
               .penFD(function(ww) .nnPenalty(.pen, ww)$value, .w),
               tolerance = 1e-7)
})

test_that("kinetic is a fraction of the objective, and inert until frozen", {
  ## `kinetic * abs(obj)` per unit mean(f^2) -- the objective side is
  ## normalized, the term side is not (see the header of R/nnPenalty.R for the
  ## measurement that rules the other choice out)
  .pen <- .nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05)
  .w <- .penW(.pen)
  ## before freezing it contributes exactly nothing, and not approximately --
  ## the gradient comes back as the very same object
  .g <- seq_along(.w) / 10
  expect_identical(.nnPenalty(.pen, .w)$value, 0)
  expect_identical(.nnAddPen(.g, .w, .pen, 1L), .g)
  .raw <- .nnPenKinTerm(.pen$nets[[1L]], .w)$value
  .penFreeze(.pen, .w, obj = 400)
  expect_equal(.nnPenalty(.pen, .w)$value, 0.05 * 400 * .raw)
  ## a NEGATIVE objective normalizes on its magnitude, not its sign -- a -2LL of
  ## -124 is an ordinary objective here
  .p2 <- .penFreeze(.nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05),
                    .w, obj = -400)
  expect_equal(.nnPenalty(.p2, .w)$value, 0.05 * 400 * .raw)
})

test_that("kinetic scales with the objective, so one lambda travels", {
  ## the claim the fraction actually makes: doubling the objective -- twice the
  ## observations, or a different endpoint's -2LL -- doubles the penalty, so
  ## `kinetic` keeps meaning the same share of the fit rather than fading out.
  .w <- .penW(.nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05))
  .a <- .penFreeze(.nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05), .w, 200)
  .b <- .penFreeze(.nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05), .w, 400)
  expect_equal(2 * .nnPenalty(.a, .w)$value, .nnPenalty(.b, .w)$value)
})

test_that("kinetic does not anchor to the near-zero ude starting network", {
  ## The regression this file exists to prevent.  Dividing by the term's own
  ## value at w0 looks right and is not: `init = "ude"` starts the network near
  ## the zero function, so that reference is ~1e-4 and the normalizer comes out
  ## ~1e4 too large -- measured, kinetic = 0.05 then produced a penalty of 566
  ## against an objective of 76.  A network whose output grows to a normal size
  ## must not see the penalty grow past the objective it is a fraction of.
  .out <- (2L * 3L + 3L + 1L):(2L * 3L + 2L * 3L + 1L)
  .w <- .penW(.nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05))
  .tiny <- .w; .tiny[.out] <- 1e-2 * .tiny[.out]     # a ude-like starting point
  .pen <- .penFreeze(.nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05),
                     .tiny, obj = 400)
  ## frozen at the tiny network, then evaluated at the full-sized one
  expect_lt(.nnPenalty(.pen, .w)$value, 400)
})

test_that("freezing is idempotent and needs a finite objective", {
  .pen <- .nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.05)
  .w <- .penW(.pen)
  ## a non-finite objective must not arm the penalty with a garbage scale
  .nnPenFreeze(.pen, .w, NA_real_)
  expect_identical(.nnPenalty(.pen, .w)$value, 0)
  .nnPenFreeze(.pen, .w, Inf)
  expect_identical(.nnPenalty(.pen, .w)$value, 0)
  .nnPenFreeze(.pen, .w, 100)
  .v <- .nnPenalty(.pen, .w)$value
  ## a later, different objective must not move the scale -- one fit, one meaning
  .nnPenFreeze(.pen, .w, 999999)
  expect_equal(.nnPenalty(.pen, .w)$value, .v)
})

test_that("the kinetic value is lambda * mean(f^2) over the realized rows", {
  ## MEAN, not sum.  The curvature grid is a fixed 11 points so a sum over it is
  ## dataset-independent, but the kinetic point set is however many rows the
  ## trial solve produced -- summing there would make one lambda mean something
  ## different for every dataset.  So doubling the number of rows at the same
  ## dynamics must leave the penalty alone.
  .p1 <- .penProfile(n = 25L)
  .p2 <- lapply(.p1, function(.q) { .q$values <- rep(.q$values, 2L); .q })
  .w <- .penW(.nnPenSpec(.penAug(), list(.p1), 0, 0, 0.25))
  .a <- .penFreeze(.nnPenSpec(.penAug(), list(.p1), 0, 0, 0.25), .w)
  .b <- .penFreeze(.nnPenSpec(.penAug(), list(.p2), 0, 0, 0.25), .w)
  expect_equal(.nnPenalty(.a, .w)$value, .nnPenalty(.b, .w)$value)
})

test_that("kinetic charges for pushing hard, and a null network pays nothing", {
  ## the term exists to shrink the network's contribution to the derivative
  ## where the solution actually goes, so a zero network is free and scaling a
  ## trained one up is quadratically expensive
  .pen <- .nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 1)
  .w <- .penW(.pen)
  .penFreeze(.pen, .w)
  .zero <- numeric(length(.w))
  expect_equal(.nnPenalty(.pen, .zero)$value, 0)
  ## only the output layer scales f linearly; doubling W2 and b2 doubles f, so
  ## the penalty goes up 4x
  .K <- 2L; .H <- 3L
  .out <- (.K * .H + .H + 1L):(.K * .H + 2L * .H + 1L)
  .w2 <- .w; .w2[.out] <- 2 * .w2[.out]
  expect_equal(.nnPenalty(.pen, .w2)$value, 4 * .nnPenalty(.pen, .w)$value)
})

test_that("kinetic points are jointly realized rows, never fabricated ones", {
  ## row i must be a combination the model really produced.  An input the trial
  ## solve does not carry is held at its center rather than paired with rows it
  ## never occurred with -- that pairing is exactly what this penalty exists to
  ## avoid, so it is asserted rather than assumed.
  .p <- .penProfile(2L, lo = 0, hi = 1, n = 9L)
  .p[[2L]]$values <- NULL
  .pts <- .nnPenPoints(.p, 2L)
  expect_equal(nrow(.pts), 9L)
  expect_equal(.pts[, 1L], .p[[1L]]$values)
  expect_true(all(.pts[, 2L] == .p[[2L]]$center))
})

test_that("kinetic degrades to nothing when no input has a trajectory", {
  ## every input an eta or an expression: there is no realized trajectory, the
  ## term can only ever be a constant, and that is said once rather than charged
  ## for silently
  .p <- .penProfile(vals = FALSE)
  expect_null(.nnPenPoints(.p, 2L))
  expect_message(.nnPenSpec(.penAug(), list(.p), 0, 0, 0.25),
                 "realized trajectory")
})

test_that("the kinetic point set is capped and non-finite rows are dropped", {
  ## the term is re-evaluated on every weight step, so a long study is
  ## subsampled rather than carried whole
  .p <- .penProfile(n = 5000L)
  expect_lte(nrow(.nnPenPoints(.p, 2L)), .nnPenN)
  ## and a row the solve could not produce a finite value for is not a point
  .q <- .penProfile(n = 25L)
  .q[[1L]]$values[c(3L, 5L)] <- c(NA_real_, Inf)
  expect_equal(nrow(.nnPenPoints(.q, 2L)), 23L)
})

test_that("kinetic obeys the -2LL/-LL scale factor and adds to the others", {
  ## the invariant from the head of this file, with all three lambdas live --
  ## the one that breaks silently if a new term is added at the wrong site
  .pen <- .nnPenSpec(.penAug(), list(.penProfile()), 0.01, 1, 0.25)
  .w <- .penW(.pen)
  .penFreeze(.pen, .w)
  set.seed(11)
  .g <- stats::rnorm(length(.w))
  expect_equal(2 * .nnAddPen(.g, .w, .pen, 1L), .nnAddPen(2 * .g, .w, .pen, 2L))
  ## and the three terms are additive, so each can be reasoned about alone
  .l2s <- .nnPenSpec(.penAug(), list(.penProfile()), 0.01, 1, 0)
  .kin <- .penFreeze(.nnPenSpec(.penAug(), list(.penProfile()), 0, 0, 0.25), .w)
  expect_equal(.nnPenalty(.pen, .w)$value,
               .nnPenalty(.l2s, .w)$value + .nnPenalty(.kin, .w)$value)
})

test_that("a penalty spec built before kinetic existed still evaluates", {
  ## .nnPenalty() reads pen$kinetic, and a hand-built or serialized spec may not
  ## have the field.  NULL there would make `kinetic > 0` a zero-length
  ## condition and error out the whole fit rather than skipping the term.
  .pen <- .nnPenSpec(.penAug(), list(.penProfile()), 0.01, 1, 0)
  .w <- .penW(.pen)
  .old <- .pen
  .old$kinetic <- NULL
  .old$env <- NULL
  expect_equal(.nnPenalty(.old, .w)$value, .nnPenalty(.pen, .w)$value)
  expect_equal(.nnPenalty(.old, .w)$grad, .nnPenalty(.pen, .w)$grad)
})
