## Progress reporting for the training loop.
##
## A default fit is a multi-round loop, and each round's inner fit is silenced so
## its own output does not scroll past.  Left alone that looks hung for minutes,
## so the loop shows a progress bar while it runs and a one-line summary of what
## happened when it stops.
##
## The bar is rxode2's, so it looks like the rest of the ecosystem, and it is
## suppressed by the inner control's existing `print = 0` convention rather than
## by a second quiet switch of our own.

.nnProgressStart <- function(rounds, quiet) {
  if (isTRUE(quiet) || !interactive()) return(NULL)
  .ok <- tryCatch({ rxode2::rxProgress(as.integer(rounds)); TRUE },
                  error = function(e) FALSE)
  if (.ok) list(rounds = as.integer(rounds)) else NULL
}

.nnProgressTick <- function(prog) {
  if (is.null(prog)) return(invisible())
  tryCatch(rxode2::rxTick(), error = function(e) NULL)
  invisible()
}

## Idempotent: the loop stops the bar when it finishes normally, and an on.exit
## stops it again if the fit errors out, so a failed fit cannot leave a stuck bar
## on the console.
.nnProgressStop <- function(prog) {
  if (is.null(prog)) return(invisible())
  tryCatch(rxode2::rxProgressStop(), error = function(e) NULL)
  invisible()
}

## The closing summary: what the loop did, whether it got there, and where the
## objective and the residual error ended up.  `parHist` carries the full trace
## for anyone who wants the round-by-round detail.
.nnRunSummary <- function(interleave, converged, nRun, maxRounds, wChange,
                          objfChange, objf, parHist, quiet) {
  if (isTRUE(quiet)) return(invisible())
  .how <- if (interleave) "joint" else "iterative"
  .why <- if (converged) {
    "converged"
  } else if (nRun >= maxRounds) {
    "stopped at the round limit"
  } else {
    "stopped"
  }
  .rmse <- tryCatch({
    .h <- do.call(rbind, parHist)
    if (is.null(.h) || !nrow(.h)) NULL else c(.h$rmse[1L], .h$rmse[nrow(.h)])
  }, error = function(e) NULL)
  message(sprintf("nn: %s training %s after %d round%s", .how, .why, nRun,
                  if (nRun == 1L) "" else "s"))
  ## the penalty share, when there is one -- the reported objective is the
  ## UNPENALIZED -2LL, so without this the number the loop converged on is
  ## invisible
  .pen <- tryCatch({
    .h <- do.call(rbind, parHist)
    if (is.null(.h) || !nrow(.h) || is.null(.h$pen)) NULL else .h$pen[nrow(.h)]
  }, error = function(e) NULL)
  message(sprintf("    objective %.4g%s | weight change %.2g%s", objf,
                  if (!is.null(.pen) && is.finite(.pen) && .pen > 0) {
                    sprintf(" (+ %.4g penalty)", .pen)
                  } else "",
                  wChange,
                  if (interleave) sprintf(" | objf change %.2g", objfChange) else ""))
  if (!is.null(.rmse) && all(is.finite(.rmse))) {
    message(sprintf("    rmse %.4g -> %.4g", .rmse[1L], .rmse[2L]))
  }
  if (!converged && nRun >= maxRounds) {
    message("    (raise nnControl(rounds=) if the weights were still moving)")
  }
  invisible()
}
