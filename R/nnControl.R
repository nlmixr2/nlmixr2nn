## est = "nn": alternating torch-weights + inner-NLME estimation.
##
## nnControl() wraps an ordinary inner control (foceiControl(), saemControl(),
## ...) and rides the NN training knobs alongside it.  It *is-a* the inner
## control (its class extends the inner control's class), so the inner estimator
## is inferred from class(inner) and every inner-control accessor keeps working.

#' Control for the neural-network estimation method (`est = "nn"`)
#'
#' Wraps an inner NLME control (e.g. [nlmixr2est::foceiControl()]) and adds the
#' neural-network training schedule.  The returned object extends the inner
#' control's class, so the inner estimator is inferred from it and the alternating
#' loop drives that estimator each round.
#'
#' @param inner an inner estimation control object (default
#'   `nlmixr2est::foceiControl()`).  Its class names the inner estimator.
#' @param rounds number of alternating rounds (inner fit + weight updates).
#' @param wSteps torch weight-optimizer steps taken per round.
#' @param lr torch optimizer learning rate.
#' @param optimizer torch optimizer, `"adam"` or `"sgd"`.
#' @param seed optional integer seed for the torch weight initialization.
#' @param mode training mode; currently only `"alternating"`.
#' @return an object of class `c("nnControl", class(inner))` carrying the NN
#'   schedule in its `"nnControl"` attribute.
#' @export
#' @author Matthew L. Fidler
nnControl <- function(inner = nlmixr2est::foceiControl(),
                      rounds = 15L, wSteps = 8L, lr = 0.03,
                      optimizer = c("adam", "sgd"), seed = NULL,
                      mode = c("alternating")) {
  optimizer <- match.arg(optimizer)
  mode <- match.arg(mode)
  if (!inherits(inner, "list") && !is.list(inner)) {
    stop("'inner' must be an nlmixr2 control object (e.g. foceiControl())",
         call. = FALSE)
  }
  checkmate::assertIntegerish(rounds, lower = 1L, len = 1L, .var.name = "rounds")
  checkmate::assertIntegerish(wSteps, lower = 1L, len = 1L, .var.name = "wSteps")
  checkmate::assertNumeric(lr, lower = 0, len = 1L, .var.name = "lr")
  .nn <- list(rounds = as.integer(rounds), wSteps = as.integer(wSteps),
              lr = as.numeric(lr), optimizer = optimizer,
              seed = if (is.null(seed)) NULL else as.integer(seed),
              mode = mode)
  .ctl <- inner
  attr(.ctl, "nnControl") <- .nn
  ## extend, don't replace, the inner control's class so it still is-a foceiControl
  class(.ctl) <- unique(c("nnControl", class(inner)))
  .ctl
}

#' Validate the control for `est = "nn"`
#' @param control the control passed to `nlmixr2()` (as a length-1 list).
#' @return a valid `nnControl` object.
#' @exportS3Method nlmixr2est::getValidNlmixrCtl
getValidNlmixrCtl.nn <- function(control) {
  .ctl <- control[[1]]
  if (is.null(.ctl)) .ctl <- nnControl()
  if (!inherits(.ctl, "nnControl")) {
    stop("est = 'nn' needs control = nnControl(...)", call. = FALSE)
  }
  .ctl
}

## the inner estimator name inferred from an nnControl's inherited class stack
## (the first *Control class that is not nnControl), e.g. "focei" for foceiControl.
.nnInnerEst <- function(control) {
  .cls <- class(control)
  .cls <- .cls[.cls != "nnControl"]
  .w <- grep("Control$", .cls, value = TRUE)
  if (length(.w) == 0L) {
    stop("could not infer the inner estimator from the nnControl", call. = FALSE)
  }
  sub("Control$", "", .w[1])
}

## the plain inner control (drop the nnControl class + attribute) to hand to the
## inner estimator unchanged.
.nnInnerControl <- function(control) {
  .ctl <- control
  attr(.ctl, "nnControl") <- NULL
  class(.ctl) <- setdiff(class(.ctl), "nnControl")
  .ctl
}
