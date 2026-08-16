## Input scaling.
##
## A network's inputs are raw model quantities: an amount of 500, a weight of 70,
## a concentration of 0.02.  Every sane weight initialization assumes inputs of
## order 1, so on raw inputs the first-layer pre-activations are enormous and the
## network starts saturated -- `tanh` pins at +/-1 with an input derivative of
## ~1e-8.  That derivative is not cosmetic: it IS the FOCEi sensitivity for a
## latent eta passed as an input, and it IS the weight-training signal.  A
## saturated network therefore has an unidentifiable eta and does not train.
##
## The fix is to scale each input by its typical magnitude.  It is applied to the
## FIRST-LAYER WEIGHTS rather than to the model text, which is exactly
## equivalent -- feeding x_k/s_k with weights W1 gives the same pre-activation as
## feeding x_k with weights W1/s_k -- and avoids three real hazards of rewriting
## the call: operator precedence inside a user's input expression, the
## parenthesis-free form the augmented-model parser requires, and having to carry
## the scale as yet another model parameter.
##
## Scaling only changes where training STARTS.  The weights move freely from
## there, so nothing about the reachable set of functions is constrained.

## A robust magnitude for a vector: the median of its nonzero absolute values.
##
## The target is that a TYPICAL input produce a pre-activation of order 1 -- the
## region where an activation is both responsive and still nonlinear.  Median
## rather than mean because dosing rows and the tail of a decay curve are wildly
## non-representative; nonzero values only, because a compartment sits at exactly
## 0 before its first dose.
##
## A high quantile was tried and is worse, for a reason worth recording: mapping
## the 90th percentile to 1 puts the typical input well inside the activation's
## near-linear region, so the network loses the curvature it exists to provide.
## Saturating in the upper tail is fine -- a bounded activation is legitimately
## flat there.
.nnTypicalScale <- function(v) {
  v <- as.numeric(v)
  v <- v[is.finite(v) & v != 0]
  if (length(v) == 0L) return(1)
  .s <- stats::median(abs(v))
  if (!is.finite(.s) || .s <= 0) 1 else .s
}

## Typical magnitude of every model quantity, from ONE solve of the model at its
## initial estimates over the real data.  This is what makes the scale
## data-derived rather than guessed: a state's magnitude depends on the dosing
## and the time grid, neither of which can be read off the model alone.
## Returns a named numeric vector, or NULL if the trial solve fails (in which
## case the caller falls back to the data columns alone).
.nnTrialScales <- function(ui, data) {
  .s <- tryCatch(
    suppressWarnings(rxode2::rxSolve(ui, data, returnType = "data.frame")),
    error = function(e) NULL)
  if (is.null(.s) || !is.data.frame(.s) || nrow(.s) == 0L) return(NULL)
  .num <- vapply(.s, is.numeric, logical(1))
  .out <- vapply(.s[.num], .nnTypicalScale, numeric(1))
  .out[!(names(.out) %in% c("id", "time"))]
}

## Scale for each input of one network, in the order the inputs appear in nn().
##
## Priority: a quantity the trial solve produced (state or lhs) -> a column of
## the data (covariate) -> an eta (already order 1 by construction, so 1) -> 1.
.nnScalesForNet <- function(inputs, trial, data, etaNames) {
  .dataScale <- function(nm) {
    .hit <- names(data)[toupper(names(data)) == toupper(nm)]
    if (length(.hit) == 0L) return(NA_real_)
    .nnTypicalScale(data[[.hit[1L]]])
  }
  vapply(inputs, function(.in) {
    .nm <- trimws(.in)
    ## an eta is a standardized random effect: it is already order 1, and
    ## dividing it out would change what the model means
    if (.nm %in% etaNames) return(1)
    if (!is.null(trial) && .nm %in% names(trial)) {
      .v <- trial[[.nm]]
      if (is.finite(.v) && .v > 0) return(.v)
    }
    .d <- .dataScale(.nm)
    if (is.finite(.d) && .d > 0) return(.d)
    ## a compound expression (e.g. nn(central/Vc)) has no single source; leave it
    ## unscaled rather than guess, and say so at the call site
    1
  }, numeric(1), USE.NAMES = FALSE)
}

## Divide a network's first-layer weights by the per-input scales.
## Layout (nnWeightLayout): W1 is H*K row-major, so input k of hidden unit j is
## at W1[j*K + k]; b1/W2/b2 follow and are untouched -- only the input side is
## rescaled.
.nnRescaleW1 <- function(w, K, H, scales) {
  if (length(scales) != K) return(w)
  if (all(!is.finite(scales) | scales == 1)) return(w)
  for (j in seq_len(H) - 1L) {
    for (k in seq_len(K)) {
      .i <- j * K + k
      if (is.finite(scales[k]) && scales[k] > 0) w[.i] <- w[.i] / scales[k]
    }
  }
  w
}

## Per-network input scales for a whole model, given the fitting data.
## `nets` is the augmented-model network metadata (id/K/H/weights + inputs).
.nnInputScales <- function(ui, data, nets, inputsById) {
  .trial <- .nnTrialScales(ui, data)
  .etas <- tryCatch(ui$eta, error = function(e) character(0))
  lapply(nets, function(.n) {
    .in <- inputsById[[as.character(.n$id)]]
    if (is.null(.in)) return(rep(1, .n$K))
    .nnScalesForNet(.in, .trial, data, .etas)
  })
}
