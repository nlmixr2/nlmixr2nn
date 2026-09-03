## Regularization of the network weights.
##
## An unregularized nn() invents variation the data does not support, and the
## amount scales with network capacity.  Before this, the only lever was to TRAIN
## LESS -- which is what nnControl(warmPopIters=) documents and apologises for.
## Two penalties replace that:
##
##   l2      shrink the weights
##   smooth  punish wiggles specifically
##
## Both shape the OPTIMIZATION only.  The reported objf (and hence AIC/BIC) stays
## the unpenalized -2 log-likelihood, so a regularized network model is still
## directly comparable to an analytic-covariate one.
##
## THE OBJECTIVE SCALE.  Everything here is on the -2LL scale -- the OFV the user
## sees:
##
##   P = -2LL + l2 * kW * sum(Weff^2) + smooth * kC * sum(C^2)
##
## `kW` and `kC` are NORMALIZERS frozen at the first evaluation that knows an
## objective, each chosen so that ITS OWN term starts out at exactly lambda times
## the objective.  That is what makes lambda dimensionless: `l2 = 0.05` means
## "the weight term starts at 5% of the objective", in any model, under any
## estimator, on any endpoint.  The two terms are normalized SEPARATELY because
## they are on wildly different natural scales -- at equal lambda the curvature
## sum measured ~180x smaller than the weight sum -- so one shared normalizer
## would leave the mix as arbitrary as the magnitude was.
##
## Without it a single default cannot work.  The raw penalty is an ABSOLUTE
## quantity while -2LL and its curvature vary a lot between endpoint types and
## estimators, so a lambda that is light for an additive focei fit dominates a
## lnorm saem one -- measured: l2 = 0.1 raw already drove the saem/lnorm smoke
## cell's rmse UP, while l2 = 1 raw was needed before shrinkage was visible on an
## overfitting additive fit.  Those two ranges did not overlap.  Normalizing
## makes them the same range.
##
## They are frozen, not recomputed, so P stays a fixed objective that a gradient
## can descend rather than a moving target.  Until it is known the penalty is
## INACTIVE (contributes exactly 0): a penalty applied at an unknown relative
## scale is the thing this normalizer exists to prevent.
##
## The two places a weight gradient is assembled do NOT share that scale:
## R/nnWeightStep.R works in -LL, R/nnPopFit.R in -2LL.  That is why .nnAddPen()
## takes a `scale` and why the invariant
##
##   twice the -LL-scale addition equals the -2LL-scale addition of twice the
##   gradient
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
##
## A LATENT ETA's input column gets 0 as well, and that exemption is what makes a
## nonzero default possible at all.  The eta reaches the model only THROUGH the
## network, so shrinking the weights it multiplies shrinks the random effect
## itself: penalizing the network and identifying a latent eta pull against each
## other directly, by construction rather than by degree.  Measured, without the
## exemption: at a lambda strong enough to shrink visibly the eta recovery
## correlation collapsed from >0.7 to 0.14, while every lambda weak enough to
## keep recovery left the weight norm within 0.3% of unpenalized.  The safe and
## the useful range simply did not meet.
##
## Exempting the eta column removes the conflict rather than splitting the
## difference: the penalty still shrinks the network's response to states and
## covariates, which is where invented structure actually lives, and leaves the
## random effect's own channel alone.
.nnL2Mult <- function(K, H, scales, exempt = logical(K)) {
  .m <- numeric(H * K + 2L * H + 1L)
  if (length(scales) != K || !all(is.finite(scales))) scales <- rep(1, K)
  if (length(exempt) != K) exempt <- logical(K)
  for (j in seq_len(H) - 1L) {
    for (k in seq_len(K)) .m[j * K + k] <- if (exempt[k]) 0 else scales[k]^2
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
.nnPenSpec <- function(aug, profiles, l2, smooth, M = .nnPenM) {
  l2 <- if (is.null(l2) || !is.finite(l2)) 0 else as.numeric(l2)
  smooth <- if (is.null(smooth) || !is.finite(smooth)) 0 else as.numeric(smooth)
  if (l2 <= 0 && smooth <= 0) return(NULL)
  if (is.null(aug$nets) || !length(aug$nets)) return(NULL)
  .nets <- tryCatch(lapply(seq_along(aug$nets), function(.i) {
    .n <- aug$nets[[.i]]
    .p <- if (is.null(profiles)) NULL else profiles[[.i]]
    .sc <- if (is.null(.p)) {
      rep(1, .n$K)
    } else {
      vapply(.p, function(.q) as.numeric(.q$scale), numeric(1))
    }
    .exempt <- if (is.null(.p)) {
      logical(.n$K)
    } else {
      vapply(.p, function(.q) identical(.q$kind, "eta"), logical(1))
    }
    list(id = .n$id, K = .n$K, H = .n$H, nW = .n$nW, gIdx = .n$gIdx,
         act = .nnActCode[[.n$act]],
         mult = .nnL2Mult(.n$K, .n$H, .sc, .exempt),
         grids = if (smooth > 0 && !is.null(.p)) .nnPenGrids(.p, .n$K, M) else list())
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
  ## The normalizer lives in an environment so that freezing it is visible to
  ## every closure already holding the spec -- the weight stepper captures `pen`
  ## once, at construction, long before any objective exists.
  list(nets = .nets, nW = aug$nW, l2 = l2, smooth = smooth,
       env = local({
         .e <- new.env(parent = emptyenv())
         .e$kW <- NA_real_
         .e$kC <- NA_real_
         .e
       }))
}

## The two penalty terms for ONE network, UNWEIGHTED (no lambda, no normalizer).
## `w` is that network's own weight vector, in nnWeightLayout() order.
## Returns each term's value and its gradient wrt w.
.nnPenTerms <- function(spec, w, smooth) {
  .wv <- sum(spec$mult * w^2)
  .wg <- 2 * spec$mult * w
  .cv <- 0
  .cg <- numeric(length(w))
  if (smooth && length(spec$grids)) {
    for (.g in spec$grids) {
      .M <- nrow(.g)
      if (.M < 3L) next
      .f <- .Call(`_nlmixr2nn_nnForwardW`, spec$K, spec$H, spec$act,
                  as.double(w), .g)
      .i <- seq.int(2L, .M - 1L)
      .C <- .f[.i + 1L] - 2 * .f[.i] + .f[.i - 1L]
      .cv <- .cv + sum(.C^2)
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
        .cg <- .cg + 2 * .C[.t] * (.J[, .m + 1L] - 2 * .J[, .m] + .J[, .m - 1L])
      }
    }
  }
  list(l2 = list(value = .wv, grad = .wg), smooth = list(value = .cv, grad = .cg))
}

## Both terms for the whole model, UNWEIGHTED.  `w` is the GLOBAL weight vector
## (every network concatenated in aug$nets order); the gradients match it.
.nnPenaltyRaw <- function(pen, w) {
  .z <- list(l2 = list(value = 0, grad = numeric(length(w))),
             smooth = list(value = 0, grad = numeric(length(w))))
  if (is.null(pen)) return(.z)
  .out <- .z
  for (.s in pen$nets) {
    .t <- .nnPenTerms(.s, w[.s$gIdx], pen$smooth > 0)
    for (.nm in c("l2", "smooth")) {
      .out[[.nm]]$value <- .out[[.nm]]$value + .t[[.nm]]$value
      .out[[.nm]]$grad[.s$gIdx] <- .out[[.nm]]$grad[.s$gIdx] + .t[[.nm]]$grad
    }
  }
  .out
}

## TRUE once the normalizers are known and the penalty is live.
.nnPenActive <- function(pen) {
  !is.null(pen) && !is.na(pen$env$kW)
}

## Freeze the normalizers from the first objective the fit produces, so that each
## term starts at `lambda * abs(obj)`.  Called from wherever an objective and the
## weights are both in hand: the population fit's own objective evaluation
## (R/nnPopFit.R) and the round loop's first inner fit (R/nnEst.R).  Idempotent --
## only the first call sets them.
##
## A term whose raw value is 0 at the starting weights carries no scale
## information (a flat network has no curvature to measure).  It is normalized to
## 0, which switches that term off for this fit rather than dividing by zero --
## and a term that starts at exactly zero has nothing to shrink anyway.
.nnPenFreeze <- function(pen, w, obj) {
  if (is.null(pen) || !is.na(pen$env$kW)) return(invisible(NULL))
  if (is.null(obj) || !is.finite(obj)) return(invisible(NULL))
  .r <- .nnPenaltyRaw(pen, w)
  .k <- function(v) if (is.finite(v) && v > 0) abs(obj) / v else 0
  ## the weight term is what arms the penalty: it is never legitimately zero for
  ## a network with any nonzero weight, so if it is, something is wrong and the
  ## penalty stays inactive rather than guessing a scale
  if (!is.finite(.r$l2$value) || .r$l2$value <= 0) return(invisible(NULL))
  pen$env$kW <- .k(.r$l2$value)
  pen$env$kC <- .k(.r$smooth$value)
  invisible(NULL)
}

## Value + gradient of the penalty as it actually enters the objective, on the
## -2LL scale.  Zero while the normalizers are unknown: a penalty applied at an
## unknown relative scale is the thing they exist to prevent.
.nnPenalty <- function(pen, w) {
  if (!.nnPenActive(pen)) return(list(value = 0, grad = numeric(length(w))))
  .r <- .nnPenaltyRaw(pen, w)
  .a <- pen$l2 * pen$env$kW
  .b <- pen$smooth * pen$env$kC
  list(value = .a * .r$l2$value + .b * .r$smooth$value,
       grad = .a * .r$l2$grad + .b * .r$smooth$grad)
}

## Freeze the normalizers from `obj` if they are not set yet, then report the
## penalty at `w`.  The population branch (R/nnEst.R) needs both in one
## expression, because its parHist row is built in a single data.frame() call.
.nnPenFrozenValue <- function(pen, w, obj) {
  .nnPenFreeze(pen, w, obj)
  .nnPenalty(pen, w)$value
}

## Add the penalty to a gradient expressed on `scale` * (-LL).
##
##   scale = 1L  the torch weight step (R/nnWeightStep.R), objective -LL
##   scale = 2L  the population pre-fit (R/nnPopFit.R),     objective -2LL
##
## The penalty is defined on -2LL, so a -LL-scale gradient takes half of it.
.nnAddPen <- function(g, w, pen, scale) {
  if (!.nnPenActive(pen)) return(g)
  g + .nnPenalty(pen, w)$grad * (scale / 2)
}

## Same, for a single network's local gradient and weights.  Takes the whole
## `pen` rather than one net's spec so that the shared normalizers are in scope.
.nnAddPenNet <- function(g, w, pen, spec, scale) {
  if (!.nnPenActive(pen) || is.null(spec)) return(g)
  .t <- .nnPenTerms(spec, w, pen$smooth > 0)
  g + (pen$l2 * pen$env$kW * .t$l2$grad +
         pen$smooth * pen$env$kC * .t$smooth$grad) * (scale / 2)
}
