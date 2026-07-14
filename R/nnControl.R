## nnControl(): the neural-network TRAINING SCHEDULE for the transparent nlmixr2nn
## workflow.  It is NOT an estimation control and does NOT wrap an inner control.
## You pass it alongside a standard estimation:
##
##   nlmixr2(model, data, "focei", foceiControl(...), nn = nnControl(...))
##
## When the model contains an nn() term, nlmixr2nn's estimation interceptor picks
## up `nn=` (or a default nnControl() if absent) and trains the embedded network
## using the requested estimator (focei/saem/imp/...) as the inner engine.

#' Neural-network training schedule for `nlmixr2nn`
#'
#' Passed as the `nn=` argument to [nlmixr2est::nlmixr2()] to train a model that
#' contains an [nn()] term.  It carries only the network training knobs -- the
#' inner NLME estimator and its control are the ordinary `est`/`control`
#' arguments of `nlmixr2()`.
#'
#' Two modes:
#' * `"joint"` (default) -- co-optimizes the population parameters and the network
#'   weights in one interleaved loop (a few warm-started partial inner iterations
#'   + a torch weight step each round), the DeepPumas-style approach;
#' * `"iter"` -- runs a full inner fit each round with the weights fixed, then a
#'   weight update (simpler solve/update/solve).
#'
#' The loop stops when the weights (and, for `"joint"`, the objective) stop
#' changing between rounds (`tol`), or after `rounds`.
#'
#' @param mode `"joint"` or `"iter"`.
#' @param rounds maximum number of rounds.
#' @param tol convergence tolerance on the relative round-to-round change (weights
#'   for `"iter"`; weights and objective for `"joint"`).  0 always runs `rounds`.
#' @param wSteps torch weight-optimizer steps per round.
#' @param outerPerRound (`"joint"` only) inner outer-iterations per round -- the
#'   partial step that makes it joint rather than a full re-fit.
#' @param lr torch optimizer learning rate.
#' @param warmSteps optional naive-pooled (eta=0) torch warm-up steps before the
#'   loop (default 0).
#' @param warmStart optional population weight pre-fit before the loop: `"none"`
#'   (default) or `"pop"` -- a gradient-based (quasi-Newton `nlminb`) fit of the
#'   weights at eta=0 with an analytic sensitivity gradient, seeding the joint fit
#'   with a robust population weight vector (bridges the fixed-effects and
#'   DeepPumas approaches).  `warmPopIters` caps its iterations.
#' @param warmPopIters iteration cap for `warmStart = "pop"` (default 30).
#' @param optimizer torch optimizer, `"adam"` or `"sgd"`.
#' @param seed optional integer seed for torch weight initialization (ignored when
#'   the model already carries trained weights, which are used as the start).
#' @return an object of class `"nnControl"`.
#' @export
#' @author Matthew L. Fidler
nnControl <- function(mode = c("joint", "iter"),
                      rounds = 200L, tol = 1e-3, wSteps = 2L, outerPerRound = 1L,
                      lr = 0.03, warmSteps = 0L,
                      warmStart = c("none", "pop"), warmPopIters = 30L,
                      optimizer = c("adam", "sgd"), seed = NULL) {
  mode <- match.arg(mode)
  optimizer <- match.arg(optimizer)
  warmStart <- match.arg(warmStart)
  checkmate::assertIntegerish(rounds, lower = 1L, len = 1L, .var.name = "rounds")
  checkmate::assertNumeric(tol, lower = 0, len = 1L, .var.name = "tol")
  checkmate::assertIntegerish(wSteps, lower = 1L, len = 1L, .var.name = "wSteps")
  checkmate::assertIntegerish(outerPerRound, lower = 1L, len = 1L, .var.name = "outerPerRound")
  checkmate::assertNumeric(lr, lower = 0, len = 1L, .var.name = "lr")
  checkmate::assertIntegerish(warmSteps, lower = 0L, len = 1L, .var.name = "warmSteps")
  checkmate::assertIntegerish(warmPopIters, lower = 1L, len = 1L, .var.name = "warmPopIters")
  structure(list(mode = mode, rounds = as.integer(rounds), tol = as.numeric(tol),
                 wSteps = as.integer(wSteps), outerPerRound = as.integer(outerPerRound),
                 lr = as.numeric(lr), warmSteps = as.integer(warmSteps),
                 warmStart = warmStart, warmPopIters = as.integer(warmPopIters),
                 optimizer = optimizer,
                 seed = if (is.null(seed)) NULL else as.integer(seed)),
            class = "nnControl")
}

#' @export
print.nnControl <- function(x, ...) {
  cat(sprintf("nnControl (nlmixr2nn training schedule): mode=%s, rounds<=%d, tol=%.2g, wSteps=%d, lr=%.3g\n",
              x$mode, x$rounds, x$tol, x$wSteps, x$lr))
  invisible(x)
}
