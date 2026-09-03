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

## ONE solve of the model at its initial estimates over the real data, reduced to
## its numeric columns.  This is what makes everything derived from it
## data-derived rather than guessed: a state's magnitude depends on the dosing
## and the time grid, neither of which can be read off the model alone.
##
## Returns the data.frame, or NULL if the trial solve fails.  The FRAME rather
## than a summary of it, because two callers want two different summaries: input
## scaling wants one typical magnitude per column, and the curvature penalty
## (R/nnPenalty.R) wants the quantiles that say where a network's inputs actually
## live.  Collapsing here would throw the latter away -- which is exactly what
## this function used to do.
.nnTrialSolve <- function(ui, data) {
  .s <- tryCatch(
    suppressWarnings(rxode2::rxSolve(ui, data, returnType = "data.frame")),
    error = function(e) NULL)
  if (is.null(.s) || !is.data.frame(.s) || nrow(.s) == 0L) return(NULL)
  .num <- vapply(.s, is.numeric, logical(1))
  .s <- .s[.num]
  .s[!(names(.s) %in% c("id", "time"))]
}

## Typical magnitude of every model quantity, as a named numeric vector.
## NULL in, NULL out.
.nnTrialScales <- function(trial) {
  if (is.null(trial) || !is.data.frame(trial) || !ncol(trial)) return(NULL)
  vapply(trial, .nnTypicalScale, numeric(1))
}

## Where an input's values come from.  THE single place the priority order
## lives, so the scaling and the penalty cannot drift apart: an eta beats a
## solved quantity, which beats a data column (matched case-insensitively,
## because data columns are conventionally upper case); anything else -- a
## compound expression such as `nn(central/Vc)` -- has no source at all.
##
## `trialNames` must already be filtered to the columns that yield a USABLE
## value, so that an unusable solved quantity falls through to the data exactly
## as it always has.
.nnInputSource <- function(nm, trialNames, dataNames, etaNames) {
  .nm <- trimws(nm)
  if (.nm %in% etaNames) return(list(kind = "eta", col = NA_character_))
  if (.nm %in% trialNames) return(list(kind = "trial", col = .nm))
  .hit <- dataNames[toupper(dataNames) == toupper(.nm)]
  if (length(.hit) > 0L) return(list(kind = "data", col = .hit[1L]))
  list(kind = "expr", col = NA_character_)
}

## Scale for each input of one network, in the order the inputs appear in nn().
##
## Priority: a quantity the trial solve produced (state or lhs) -> a column of
## the data (covariate) -> an eta (already order 1 by construction, so 1) -> 1.
.nnScalesForNet <- function(inputs, trial, data, etaNames) {
  .ok <- if (is.null(trial)) {
    character(0)
  } else {
    names(trial)[is.finite(trial) & trial > 0]
  }
  vapply(inputs, function(.in) {
    .src <- .nnInputSource(.in, .ok, names(data), etaNames)
    switch(.src$kind,
           ## an eta is a standardized random effect: it is already order 1, and
           ## dividing it out would change what the model means
           eta = 1,
           trial = trial[[.src$col]],
           data = {
             .d <- .nnTypicalScale(data[[.src$col]])
             if (is.finite(.d) && .d > 0) .d else 1
           },
           ## a compound expression (e.g. nn(central/Vc)) has no single source;
           ## leave it unscaled rather than guess
           1)
  }, numeric(1), USE.NAMES = FALSE)
}

## Everything the curvature penalty needs to know about one network's inputs:
## per input, its typical magnitude, a center to hold it at while the OTHER
## inputs are swept, and the range to sweep it over.
##
## `trial` is the data.frame from .nnTrialSolve() (NULL if it failed), so the
## range is where the inputs actually go over the real dosing and time grid,
## not a guess.  An input with no resolvable range -- an eta, whose EBEs move
## every round and which is not a solve column at all, or a compound expression
## -- gets range = NA and is simply held at its center, never gridded.
##
## `etaNames` must carry BOTH spellings of every eta.  The augmented model
## renames `eta.nn` to `eta_nn` (R/nnAugmentUi.R), so inputs parsed from it are
## sanitized while `ui$eta` is not; matching against one spelling alone silently
## fails to recognize an eta, which then falls through to the data lookup, misses,
## and returns 1 -- the right answer today, so the bug would stay invisible.
.nnInputProfile <- function(inputs, trial, data, etaNames) {
  .scales <- .nnTrialScales(trial)
  .ok <- if (is.null(.scales)) {
    character(0)
  } else {
    names(.scales)[is.finite(.scales) & .scales > 0]
  }
  .summarize <- function(v) {
    v <- as.numeric(v)
    v <- v[is.finite(v)]
    if (length(v) < 2L) return(NULL)
    .q <- unname(stats::quantile(v, c(0.1, 0.9), na.rm = TRUE))
    list(center = stats::median(v),
         range = if (all(is.finite(.q)) && .q[2L] > .q[1L]) .q else NULL)
  }
  lapply(inputs, function(.in) {
    .src <- .nnInputSource(.in, .ok, names(data), etaNames)
    .out <- list(name = trimws(.in), kind = .src$kind, scale = 1,
                 center = 0, range = NULL)
    if (identical(.src$kind, "eta") || identical(.src$kind, "expr")) return(.out)
    .v <- if (identical(.src$kind, "trial")) trial[[.src$col]] else data[[.src$col]]
    .s <- .nnTypicalScale(.v)
    .out$scale <- if (is.finite(.s) && .s > 0) .s else 1
    .sum <- .summarize(.v)
    if (!is.null(.sum)) {
      .out$center <- .sum$center
      .out$range <- .sum$range
    }
    .out
  })
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

## Per-network input scales for a whole model, given the fitting data and the
## trial solve.  `nets` is the augmented-model network metadata, which carries
## each network's input expressions (R/nnAugmentUi.R); `trial` is the frame from
## .nnTrialSolve(), or NULL when it failed, in which case the scales come from
## the data columns alone.
.nnInputScales <- function(trial, data, nets, etaNames) {
  .scales <- .nnTrialScales(trial)
  lapply(nets, function(.n) {
    if (is.null(.n$inputs)) return(rep(1, .n$K))
    .nnScalesForNet(.n$inputs, .scales, data, etaNames)
  })
}
