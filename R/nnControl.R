## est = "nnIter": ITERATIVE solve/update/solve estimation of an embedded NN.
##
## This is NOT a joint (DeepPumas-style) optimization.  Each round runs a FULL
## inner NLME fit (FOCEi/SAEM/IMP) of the base model with the current network
## weights held fixed, then takes a few torch weight steps from a forward-
## sensitivity solve at the fitted EBEs -- solve, update, solve, ...  It is a
## block-coordinate scheme; hence the "Iter" name.
##
## nnIterControl() wraps an ordinary inner control and rides the NN training
## knobs alongside it.  It *is-a* the inner control (its class extends the inner
## control's class), so the inner estimator is inferred from class(inner) and
## every inner-control accessor keeps working.

#' Control for the iterative neural-network estimation method (`est = "nnIter"`)
#'
#' Wraps an inner NLME control (e.g. [nlmixr2est::foceiControl()]) and adds the
#' neural-network training schedule for the *iterative* method `est = "nnIter"`.
#' The returned object extends the inner control's class, so the inner estimator
#' is inferred from it and the alternating loop drives that estimator each round.
#'
#' **Stopping.**  The loop runs until the network weights stop moving between
#' rounds -- specifically when the relative change in the flattened weight vector,
#' `||w_new - w_old|| / (||w_old|| + eps)`, drops below `tol` -- or until `rounds`
#' (the maximum) is reached, whichever comes first.  The number of rounds actually
#' run and whether it converged are reported on the returned fit (`$nnConverged`,
#' `$nnRounds`) and in `$nnParHist` (the per-round `wChange` column).
#'
#' @param inner an inner estimation control object (default
#'   `nlmixr2est::foceiControl()`).  Its class names the inner estimator.
#' @param rounds MAXIMUM number of alternating rounds (inner fit + weight
#'   updates); the loop stops earlier when `tol` is met.
#' @param tol convergence tolerance on the relative between-round weight change;
#'   the loop stops once the change is below this.  Set to 0 to always run the
#'   full `rounds`.
#' @param warmSteps optional torch weight-optimizer steps taken BEFORE the first
#'   inner fit, as a naive-pooled warm-up at zero random effects (default 0, off).
#'   A small number can move the randomly initialized network off a poor starting
#'   point, but is not required.
#' @param wSteps torch weight-optimizer steps taken per round.
#' @param lr torch optimizer learning rate.
#' @param optimizer torch optimizer, `"adam"` or `"sgd"`.
#' @param seed optional integer seed for the torch weight initialization.
#' @param mode training mode; currently only `"alternating"`.
#' @return an object of class `c("nnIterControl", class(inner))` carrying the NN
#'   schedule in its `"nnIterControl"` attribute.
#' @export
#' @author Matthew L. Fidler
nnIterControl <- function(inner = nlmixr2est::foceiControl(),
                          rounds = 15L, tol = 1e-3, warmSteps = 0L, wSteps = 8L,
                          lr = 0.03, optimizer = c("adam", "sgd"), seed = NULL,
                          mode = c("alternating")) {
  optimizer <- match.arg(optimizer)
  mode <- match.arg(mode)
  if (!inherits(inner, "list") && !is.list(inner)) {
    stop("'inner' must be an nlmixr2 control object (e.g. foceiControl())",
         call. = FALSE)
  }
  checkmate::assertIntegerish(rounds, lower = 1L, len = 1L, .var.name = "rounds")
  checkmate::assertNumeric(tol, lower = 0, len = 1L, .var.name = "tol")
  checkmate::assertIntegerish(warmSteps, lower = 0L, len = 1L, .var.name = "warmSteps")
  checkmate::assertIntegerish(wSteps, lower = 1L, len = 1L, .var.name = "wSteps")
  checkmate::assertNumeric(lr, lower = 0, len = 1L, .var.name = "lr")
  .nn <- list(rounds = as.integer(rounds), tol = as.numeric(tol),
              warmSteps = as.integer(warmSteps), wSteps = as.integer(wSteps),
              lr = as.numeric(lr), optimizer = optimizer,
              seed = if (is.null(seed)) NULL else as.integer(seed),
              mode = mode)
  .ctl <- inner
  attr(.ctl, "nnIterControl") <- .nn
  ## extend, don't replace, the inner control's class so it still is-a foceiControl
  class(.ctl) <- unique(c("nnIterControl", class(inner)))
  .ctl
}

#' Validate the control for `est = "nnIter"`
#' @param control the control passed to `nlmixr2()` (as a length-1 list).
#' @return a valid `nnIterControl` object.
#' @exportS3Method nlmixr2est::getValidNlmixrCtl
getValidNlmixrCtl.nnIter <- function(control) {
  .ctl <- control[[1]]
  if (is.null(.ctl)) .ctl <- nnIterControl()
  if (!inherits(.ctl, "nnIterControl")) {
    stop("est = 'nnIter' needs control = nnIterControl(...)", call. = FALSE)
  }
  .ctl
}

## the inner estimator name inferred from an nnIterControl's inherited class stack
## (the first *Control class that is not nnIterControl), e.g. "focei".
.nnInnerEst <- function(control) {
  .cls <- class(control)
  .cls <- .cls[.cls != "nnIterControl"]
  .w <- grep("Control$", .cls, value = TRUE)
  if (length(.w) == 0L) {
    stop("could not infer the inner estimator from the nnIterControl", call. = FALSE)
  }
  sub("Control$", "", .w[1])
}

## the plain inner control (drop the nnIterControl class + attribute) to hand to
## the inner estimator unchanged.
.nnInnerControl <- function(control) {
  .ctl <- control
  attr(.ctl, "nnIterControl") <- NULL
  class(.ctl) <- setdiff(class(.ctl), "nnIterControl")
  .ctl
}
