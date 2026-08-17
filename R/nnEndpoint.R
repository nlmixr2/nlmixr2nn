## The endpoint's transformation, and its derivative.
##
## nlmixr2est's likelihood-contribution hook reports d(LL)/d(f) where f is the
## TRANSFORMED prediction -- transform-both-sides is applied in rxode2's
## `rx_pred_` statement, so by the time the hook sees it, f is already on the
## transformed scale.  The augmented solve's `rx_sw` are sensitivities of the
## NATURAL-scale state.
##
## Multiplying those two together skips the transformation's own derivative.
## For `centr ~ lnorm(sd)` the assembled weight gradient was therefore short by
## a factor of 1/f at every observation.  It still trained, because a positive
## per-observation rescaling is usually still an uphill direction -- which is
## precisely why it went unnoticed.  Chaining through this Jacobian is what makes
## the gradient the gradient.

## The transformations rxode2 can put on an endpoint.  Combined levels are a
## bounding transform (logit/probit) followed by a power transform, so their
## derivative is the product of the two.
.nnTransformParts <- function(transform) {
  if (is.na(transform) || !nzchar(transform)) return(character(0))
  trimws(strsplit(transform, "+", fixed = TRUE)[[1L]])
}

## The forward value of a BOUNDING transformation, needed as the inner value of
## a combined level: "logit + yeoJohnson" means yeoJohnson(logit(y)), so the
## outer derivative is evaluated at logit(y), not at y.  Only logit and probit
## ever appear on the inside, so only those are needed here.
.nnTransformFwd <- function(what, y, low = 0, hi = 1) {
  .u <- (y - low) / (hi - low)
  switch(what,
         logit = log(.u / (1 - .u)),
         probit = stats::qnorm(.u),
         y)
}

## d T(y) / dy for a single transformation, elementwise over y.
.nnTransformJac1 <- function(what, y, lambda = 1, low = 0, hi = 1) {
  switch(
    what,
    untransformed = rep(1, length(y)),
    ## log(y)
    lnorm = 1 / y,
    ## Box-Cox: (y^l - 1)/l for l != 0, log(y) at l == 0
    boxCox = if (isTRUE(all.equal(lambda, 0))) 1 / y else y^(lambda - 1),
    ## Yeo-Johnson, which is defined piecewise about 0 so its derivative is too
    yeoJohnson = ifelse(y >= 0, (y + 1)^(lambda - 1), (1 - y)^(1 - lambda)),
    ## logit((y - low)/(hi - low))
    logit = (hi - low) / ((y - low) * (hi - y)),
    ## qnorm((y - low)/(hi - low))
    probit = {
      .u <- (y - low) / (hi - low)
      1 / ((hi - low) * stats::dnorm(stats::qnorm(.u)))
    },
    ## an unknown level is better refused than silently treated as identity,
    ## which is the failure mode this whole function exists to remove
    stop("nlmixr2nn: unsupported endpoint transformation '", what, "'",
         call. = FALSE)
  )
}

#' Derivative of an endpoint's transformation at the natural-scale prediction
#'
#' @param ep endpoint description carrying `transform`, and `lambda`/`trLow`/
#'   `trHi` where the transformation needs them.
#' @param y natural-scale predictions.
#' @return `d T(y) / dy`, elementwise.
#' @keywords internal
#' @noRd
.nnTransformJac <- function(ep, y) {
  .tr <- if (is.null(ep$transform)) "untransformed" else as.character(ep$transform)
  .parts <- .nnTransformParts(.tr)
  if (length(.parts) == 0L) return(rep(1, length(y)))
  ## rxode2 does not evaluate a Box-Cox applied after a bounding transform --
  ## .rxTransform() returns NA for "logit + boxCox" / "probit + boxCox" at every
  ## value and every lambda, so such a model cannot produce predictions in the
  ## first place.  Returning a derivative for it would be inventing a number the
  ## solver never computes.
  if (length(.parts) > 1L && "boxCox" %in% .parts) {
    stop("nlmixr2nn: '", .tr, "' endpoints are not supported -- rxode2 does not ",
         "evaluate a Box-Cox transformation applied after a bounding one",
         call. = FALSE)
  }
  .lambda <- if (is.null(ep$lambda) || is.na(ep$lambda)) 1 else as.numeric(ep$lambda)
  .low <- if (is.null(ep$trLow) || is.na(ep$trLow)) 0 else as.numeric(ep$trLow)
  .hi <- if (is.null(ep$trHi) || is.na(ep$trHi)) 1 else as.numeric(ep$trHi)
  ## A combined level is one transformation applied to the RESULT of another:
  ## "logit + yeoJohnson" is yeoJohnson(logit(y)).  The chain rule therefore
  ## evaluates each derivative at the value flowing into it, not all of them at
  ## the natural scale -- multiplying parts evaluated at y is wrong, and was
  ## caught by finite-differencing rxode2's own transform.
  ##
  ## Only the FIRST part sees the natural scale, which is what `rx_sw` is in, so
  ## the bounds apply there and the power transform sees whatever comes out.
  .j <- rep(1, length(y))
  .cur <- y
  for (.p in .parts) {
    .j <- .j * .nnTransformJac1(.p, .cur, lambda = .lambda, low = .low, hi = .hi)
    .cur <- .nnTransformFwd(.p, .cur, low = .low, hi = .hi)
  }
  .j
}

## Does this endpoint need the Jacobian at all?  An untransformed endpoint is
## the common case and costs nothing to skip.
.nnNeedsTransformJac <- function(ep) {
  .tr <- if (is.null(ep$transform)) "untransformed" else as.character(ep$transform)
  !(is.na(.tr) || !nzchar(.tr) || identical(.tr, "untransformed"))
}
