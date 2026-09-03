## Regularization of the network weights.
##
## An unregularized nn() invents variation the data does not support, and the
## amount scales with network capacity.  Before this, the only lever was to TRAIN
## LESS -- which is what nnControl(warmPopIters=) documents and apologises for.
## Two penalties replace that:
##
##   l2       shrink the weights
##   smooth   punish wiggles specifically
##   kinetic  punish how hard the network pushes along the realized trajectory
##
## All three shape the OPTIMIZATION only.  The reported objf (and hence AIC/BIC)
## stays the unpenalized -2 log-likelihood, so a regularized network model is
## still directly comparable to an analytic-covariate one.
##
## `kinetic` is the kinetic-energy / optimal-transport regularizer of the neural
## ODE literature (Worsham & Kalita 2025, whose measured effect on a pendulum
## NODE was the largest of any lever in their table -- mse 1.42 -> 0.70 AND a
## shorter run, because the learned dynamics got easier to integrate; the idea is
## from Finlay et al. 2020).  It is the one penalty here that pays for itself
## twice for us: nnAugment.R carries nStates x nWeights variational states
## alongside the primal, so every solver step a smoother RHS saves is multiplied
## by the whole sensitivity block.  It is OFF by default, unlike l2/smooth,
## because it is the only one that changes what the SOLVER does rather than only
## where the optimizer goes.
##
## THE OBJECTIVE SCALE.  Everything here is on the -2LL scale -- the OFV the user
## sees:
##
##   P = -2LL + l2 * sum(Weff^2) + smooth * sum(C^2) + kinetic * mean(f^2)
##
## Note sum for `smooth` and MEAN for `kinetic`.  That is not an inconsistency:
## the curvature grid is a FIXED 11 points, so a sum over it is dataset-
## independent, while the kinetic point set is however many rows the trial solve
## produced.  Summing there would make one lambda mean something different for
## every dataset -- a 2000-row study penalized 20x a 100-row one at identical
## dynamics.
##
## The two places a weight gradient is assembled do NOT share that scale:
## R/nnWeightStep.R works in -LL, R/nnPopFit.R in -2LL.  That is why .nnAddPen()
## takes a `scale` and why the invariant
##
##   2 * .nnAddPen(g, w, pen, 1L) == .nnAddPen(2 * g, w, pen, 2L)
##
## is asserted in tests/testthat/test-nn-penalty.R.  Do not "simplify" either
## call site without re-reading that test: getting it wrong makes one lambda mean
## two different things inside a single fit, and nothing about the fit will say so.

## Grid resolution for the curvature penalty.  Not exposed: 11 points is enough to
## see a wiggle and cheap enough to re-evaluate on every weight step.
.nnPenM <- 11L

## Per-weight multiplier for the L2 term, in nnWeightLayout() order.
##
## Plain sum(W1^2) is the WRONG thing to penalize in this package, because input
## scaling is folded into the first-layer weights and then discarded
## (.nnRescaleW1, R/nnScale.R): after scaling, an input of typical magnitude 500
## carries a W1 column ~500x smaller than one of magnitude 1 at equal functional
## effect, so a single lambda would penalize it 250000x less.  What is
## functionally comparable is the EFFECTIVE weight s_k * W1[j,k], so the
## multiplier is s_k^2 there.
##
## W2 is already scale-free and gets 1.  Biases get 0 -- they stay free, so the
## network can still move its output level without paying for it.
.nnL2Mult <- function(K, H, scales) {
  .m <- numeric(H * K + 2L * H + 1L)
  if (length(scales) != K || !all(is.finite(scales))) scales <- rep(1, K)
  for (j in seq_len(H) - 1L) {
    for (k in seq_len(K)) .m[j * K + k] <- scales[k]^2
  }
  .m[(H * K + H + 1L):(H * K + 2L * H)] <- 1
  .m
}

## The partial-dependence grids for one network: one M x K matrix per input that
## has a resolvable range, that input swept over its 10th-90th percentile and
## every other input pinned at its center.
##
## Sweeping the input's OWN observed range is what makes the curvature penalty
## scale-free without any extra normalization: C = f(x+h) - 2f(x) + f(x-h) is
## h^2 * f''(x), and h is proportional to that input's spread, so C is the
## curvature with respect to the STANDARDIZED input.  A grid on a fixed span
## would not have that property and `smooth` would mean something different for
## every input.
.nnPenGrids <- function(profile, K, M = .nnPenM) {
  .center <- vapply(profile, function(.p) as.numeric(.p$center), numeric(1))
  .grids <- list()
  for (.k in seq_len(K)) {
    .r <- profile[[.k]]$range
    if (is.null(.r)) next                       # an eta or a compound expression
    .g <- matrix(rep(.center, each = M), nrow = M, ncol = K)
    .g[, .k] <- seq(.r[1L], .r[2L], length.out = M)
    .grids[[length(.grids) + 1L]] <- .g
  }
  .grids
}

## Cap on the number of trajectory points the kinetic term is evaluated at.  The
## penalty is re-evaluated on every weight step, and each point costs one forward
## pass plus one nW-long weight gradient, so carrying the whole trajectory of a
## large study is not worth it: 200 rows spread evenly through the realized ones
## describe the input distribution just as well.  Not exposed.
.nnPenN <- 200L

## The JOINTLY REALIZED input rows for one network: one row per (subsampled)
## trial-solve row, each column that input's value there.
##
## Realized rather than gridded is the whole point of this term.  `smooth`
## measures curvature on a synthetic partial-dependence grid -- one input swept,
## the others pinned at their center -- which asks what the network does at input
## combinations that may never occur.  `kinetic` instead asks how hard the
## network is pushing WHERE THE SOLUTION ACTUALLY GOES, which is what makes it a
## statement about the dynamics the solver has to integrate rather than about the
## function in the abstract.
##
## Every column therefore has to come from the SAME frame, so that row i is a
## combination the model really produced.  An input the trial solve does not
## carry -- an eta, a compound expression such as `nn(central/Vc)`, or a covariate
## that lives only in the data at a different row count -- is held at its center
## instead.  Column-binding two frames of different length would fabricate input
## combinations that never happened, which is precisely what this penalty exists
## to avoid.
##
## Returns NULL when no input has realized values: the term would then be
## evaluated at the single center point, which is a legitimate number but not a
## meaningful one, and saying so once beats charging for it silently.
.nnPenPoints <- function(profile, K, nMax = .nnPenN) {
  .vals <- lapply(profile, function(.p) .p$values)
  .has <- !vapply(.vals, is.null, logical(1))
  if (!any(.has)) return(NULL)
  .n <- unique(vapply(.vals[.has], length, integer(1)))
  if (length(.n) != 1L || .n < 1L) return(NULL)
  .i <- if (.n <= nMax) seq_len(.n) else {
    unique(as.integer(round(seq(1, .n, length.out = nMax))))
  }
  .center <- vapply(profile, function(.p) as.numeric(.p$center), numeric(1))
  .pts <- matrix(rep(.center, each = length(.i)), nrow = length(.i), ncol = K)
  for (.k in seq_len(K)) {
    if (.has[[.k]]) .pts[, .k] <- .vals[[.k]][.i]
  }
  .pts <- .pts[is.finite(rowSums(.pts)), , drop = FALSE]
  if (nrow(.pts) < 1L) return(NULL)
  .pts
}

## The penalty specification for a whole model, built once per fit.
##
## Returns NULL when there is nothing to do -- both lambdas zero, no networks, or
## anything at all went wrong.  NULL is the strict no-op: .nnPenalty() then
## contributes exactly 0 to the objective and exactly 0 to every gradient
## coordinate, so nnControl(l2 = 0, smooth = 0) reproduces an unregularized fit
## bit for bit.
##
## `profiles` is one .nnInputProfile() per net, in aug$nets order, or NULL when
## the trial solve failed (in which case only l2 is available -- it needs no grid).
.nnPenSpec <- function(aug, profiles, l2, smooth, kinetic = 0, M = .nnPenM) {
  .lam <- function(v) if (is.null(v) || !is.finite(v)) 0 else as.numeric(v)
  l2 <- .lam(l2)
  smooth <- .lam(smooth)
  kinetic <- .lam(kinetic)
  if (l2 <= 0 && smooth <= 0 && kinetic <= 0) return(NULL)
  if (is.null(aug$nets) || !length(aug$nets)) return(NULL)
  .nets <- tryCatch(lapply(seq_along(aug$nets), function(.i) {
    .n <- aug$nets[[.i]]
    .p <- if (is.null(profiles)) NULL else profiles[[.i]]
    .sc <- if (is.null(.p)) rep(1, .n$K) else {
      vapply(.p, function(.q) as.numeric(.q$scale), numeric(1))
    }
    list(id = .n$id, K = .n$K, H = .n$H, nW = .n$nW, gIdx = .n$gIdx,
         act = .nnActCode[[.n$act]],
         mult = .nnL2Mult(.n$K, .n$H, .sc),
         grids = if (smooth > 0 && !is.null(.p)) .nnPenGrids(.p, .n$K, M) else list(),
         pts = if (kinetic > 0 && !is.null(.p)) .nnPenPoints(.p, .n$K) else NULL)
  }), error = function(e) NULL)
  if (is.null(.nets)) {
    warning("nlmixr2nn: the weight penalty could not be set up; ",
            "this fit runs unregularized", call. = FALSE)
    return(NULL)
  }
  if (smooth > 0 && !any(vapply(.nets, function(.s) length(.s$grids) > 0L, logical(1)))) {
    ## `smooth` was asked for but no network has a single input whose range is
    ## knowable, so the curvature term can never be anything but zero.  Say so
    ## once rather than let the knob quietly do nothing.
    message("nn: no network input has a resolvable range, so nnControl(smooth=) ",
            "has nothing to act on")
  }
  if (kinetic > 0 && !any(vapply(.nets, function(.s) !is.null(.s$pts), logical(1)))) {
    ## Same courtesy for the kinetic term, and it fails for a DIFFERENT reason
    ## worth naming: not "no range" but "no realized trajectory" -- the trial
    ## solve failed, or every input is an eta/expression the solve does not
    ## carry.
    message("nn: no network input has a realized trajectory (the trial solve ",
            "carries none of them), so nnControl(kinetic=) has nothing to act on")
  }
  list(nets = .nets, nW = aug$nW, l2 = l2, smooth = smooth, kinetic = kinetic)
}

## Value + gradient of the penalty for ONE network, on the -2LL scale.
## `w` is that network's own weight vector, in nnWeightLayout() order.
.nnPenaltyNet <- function(spec, w, l2, smooth, kinetic = 0) {
  .val <- 0
  .grad <- numeric(length(w))
  if (l2 > 0) {
    .val <- .val + l2 * sum(spec$mult * w^2)
    .grad <- .grad + 2 * l2 * spec$mult * w
  }
  if (smooth > 0 && length(spec$grids)) {
    for (.g in spec$grids) {
      .M <- nrow(.g)
      if (.M < 3L) next
      .f <- .Call(`_nlmixr2nn_nnForwardW`, spec$K, spec$H, spec$act,
                  as.double(w), .g)
      .i <- seq.int(2L, .M - 1L)
      .C <- .f[.i + 1L] - 2 * .f[.i] + .f[.i - 1L]
      .val <- .val + smooth * sum(.C^2)
      ## d(out)/dw at every grid point, once; each row then feeds up to three of
      ## the second differences above.  The registry-based nnWeightGrad reads the
      ## live solve's par_ptr and returns ZEROS outside a solve -- the explicit
      ## weight form is the only correct one here.
      .J <- vapply(seq_len(.M), function(.r) {
        .Call(`_nlmixr2nn_nnWeightGradW`, spec$K, spec$H, spec$act,
              as.double(w), as.double(.g[.r, ]))
      }, numeric(length(w)))
      for (.t in seq_along(.i)) {
        .m <- .i[.t]
        .dC <- .J[, .m + 1L] - 2 * .J[, .m] + .J[, .m - 1L]
        .grad <- .grad + 2 * smooth * .C[.t] * .dC
      }
    }
  }
  if (kinetic > 0 && !is.null(spec$pts)) {
    ## mean(f^2) over the realized input rows: the network's contribution to the
    ## derivative, averaged where the solution actually goes.  Squared rather
    ## than the plain norm Worsham & Kalita's implementation uses, for one
    ## practical reason -- ||.|| is not differentiable at 0, and a UDE network
    ## initialized as a small perturbation of the mechanistic model
    ## (init = "ude") starts very close to exactly there.
    .N <- nrow(spec$pts)
    .f <- .Call(`_nlmixr2nn_nnForwardW`, spec$K, spec$H, spec$act,
                as.double(w), spec$pts)
    .val <- .val + kinetic * mean(.f^2)
    ## d(out)/dw at every point: nW x N, so the chain below is one matrix-vector
    ## product rather than a loop.  As in the curvature term, the explicit-weight
    ## form is the only correct one -- the registry-based nnWeightGrad reads the
    ## live solve's par_ptr and returns zeros outside a solve.
    .J <- vapply(seq_len(.N), function(.r) {
      .Call(`_nlmixr2nn_nnWeightGradW`, spec$K, spec$H, spec$act,
            as.double(w), as.double(spec$pts[.r, ]))
    }, numeric(length(w)))
    .grad <- .grad + (2 * kinetic / .N) * as.numeric(.J %*% .f)
  }
  list(value = .val, grad = .grad)
}

## Value + gradient of the whole model's penalty, on the -2LL scale.
## `w` is the GLOBAL weight vector (every network concatenated in aug$nets
## order); the returned gradient matches it.
.nnPenalty <- function(pen, w) {
  if (is.null(pen)) return(list(value = 0, grad = numeric(length(w))))
  .val <- 0
  .grad <- numeric(length(w))
  for (.s in pen$nets) {
    .r <- .nnPenaltyNet(.s, w[.s$gIdx], pen$l2, pen$smooth,
                        ## a spec built before `kinetic` existed (or by hand in a
                        ## test) has no such field; NULL would make `kinetic > 0`
                        ## a zero-length condition and error out the whole fit
                        if (is.null(pen$kinetic)) 0 else pen$kinetic)
    .val <- .val + .r$value
    .grad[.s$gIdx] <- .grad[.s$gIdx] + .r$grad
  }
  list(value = .val, grad = .grad)
}

## Add the penalty to a gradient expressed on `scale` * (-LL).
##
##   scale = 1L  the torch weight step (R/nnWeightStep.R), objective -LL
##   scale = 2L  the population pre-fit (R/nnPopFit.R),     objective -2LL
##
## The penalty is defined on -2LL, so a -LL-scale gradient takes half of it.
.nnAddPen <- function(g, w, pen, scale) {
  if (is.null(pen)) return(g)
  g + .nnPenalty(pen, w)$grad * (scale / 2)
}

## Same, for a single network's local gradient and weights.
.nnAddPenNet <- function(g, w, spec, l2, smooth, kinetic, scale) {
  if (is.null(spec)) return(g)
  g + .nnPenaltyNet(spec, w, l2, smooth, kinetic)$grad * (scale / 2)
}
