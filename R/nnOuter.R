## Outer-problem NN training step for a native FOCEI fit.
##
## nlmixr2est's foceiOfv0 hands a registered R function, once per real outer
## objective evaluation, a method-agnostic matrix with one row per observation:
## column 1 is the predicted value f, the remaining columns are the ODE state
## vector (carrying the rx_sw forward-sensitivity states).  This turns that into
## a torch optimizer step on the population NN weights:
##   d(f)/d(w_j) = d(f)/d(state) * rx_sw_state   (columns configured in swCols)
##   d(LL)/d(w_j) = sum_obs (d(LL)/d(f)) * d(f)/d(w_j)
## d(LL)/d(f) is the error-model score at the current fit (additive Gaussian with
## the residual SD profiled by MLE each step; the method's own cotangent can be
## substituted later).

.nnOuterEnv <- new.env(parent = emptyenv())

#' Register the outer-problem NN training step with a FOCEI fit
#'
#' @param nnid network id (torch-held weights).
#' @param swCols 1-based matrix columns giving d(f)/d(w_j) for each weight j --
#'   i.e. the rx_sw column of the observed state (scaled by d(f)/d(state) if the
#'   prediction is not the state itself).
#' @param dv observed values, in the matrix's row order (subject-major).
#' @return invisibly NULL.  Installs the callback via nlmixr2est.
#' @export
nnOuterRegister <- function(nnid, swCols, dv) {
  .nnOuterEnv$nnid <- as.integer(nnid)
  .nnOuterEnv$swCols <- as.integer(swCols)
  .nnOuterEnv$dv <- as.double(dv)
  .nnOuterEnv$llTrace <- numeric(0)
  .nnOuterEnv$dLLdw <- NULL
  fn <- function(mat) nnOuterStep(mat)
  .Call("_nlmixr2est_setNnOuterFn", fn, PACKAGE = "nlmixr2est")
  invisible()
}

#' @rdname nnOuterRegister
#' @export
nnOuterUnregister <- function() {
  .Call("_nlmixr2est_setNnOuterFn", NULL, PACKAGE = "nlmixr2est")
  invisible()
}

## One training step from the outer matrix (called by nlmixr2est's callback).
nnOuterStep <- function(mat) {
  f <- mat[, 1L]
  dv <- .nnOuterEnv$dv
  if (length(dv) != length(f)) return(invisible())   # not aligned -- skip safely
  resid <- dv - f
  sigma <- sqrt(mean(resid^2))
  if (!is.finite(sigma) || sigma <= 0) return(invisible())
  dLLdf <- resid / sigma^2
  dLLdw <- vapply(.nnOuterEnv$swCols,
                  function(cn) sum(dLLdf * mat[, cn]), numeric(1))
  nid <- .nnOuterEnv$nnid
  nnTorchZeroGrad(nid)
  nnTorchSetGrad(nid, -dLLdw)                         # minimize -LL
  nnTorchStep(nid)
  .nnOuterEnv$dLLdw <- dLLdw
  .nnOuterEnv$llTrace <- c(.nnOuterEnv$llTrace, sum(dnorm(dv, f, sigma, log = TRUE)))
  invisible()
}
