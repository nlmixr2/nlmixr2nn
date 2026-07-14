## est = "nn": JOINT (DeepPumas-style) NN-in-ODE estimation.
##
## Unlike est = "nnIter" (which runs a FULL inner NLME fit each round), the joint
## method co-optimizes the population parameters and the network weights in ONE
## loop: each round is a warm-started PARTIAL inner step (a few outer iterations,
## resuming from the previous round's estimates) interleaved with torch weight
## steps -- so theta/Omega and the weights descend together, not one-after-the-
## other-to-convergence.  It stops when both stabilise (relative objective change
## and weight change below `tol`).
##
## Shares the augmented-model builder (.nnAugmentFromUi) and the weight step
## (.nnWeightStepper) with est = "nnIter".

#' Control for the joint neural-network estimation method (`est = "nn"`)
#'
#' Wraps an inner NLME control (e.g. [nlmixr2est::foceiControl()]) for the *joint*
#' method `est = "nn"`, which co-optimizes the population parameters and the NN
#' weights in a single interleaved loop (contrast [nnIterControl()], which runs a
#' full inner fit each round).  The returned object extends the inner control's
#' class, so the inner estimator is inferred from it.
#'
#' **Stopping.**  Each round takes `outerPerRound` inner outer-iterations
#' (warm-started from the previous round) plus `wSteps` torch weight steps.  The
#' loop stops when both the relative objective change and the relative weight
#' change fall below `tol`, or after `maxRounds`.  `$nnConverged`, `$nnRounds` and
#' `$nnParHist` (with `objfChange`/`wChange` columns) report the outcome.
#'
#' @param inner an inner estimation control object (default
#'   `nlmixr2est::foceiControl()`).  Its class names the inner estimator.
#' @param maxRounds MAXIMUM number of interleaved rounds.
#' @param outerPerRound inner outer-iterations taken per round (small, e.g. 1--3);
#'   the partial step that makes this joint rather than a full re-fit.
#' @param wSteps torch weight-optimizer steps per round.
#' @param tol convergence tolerance on the relative round-to-round objective and
#'   weight change (both must be below it to stop).
#' @param lr torch optimizer learning rate.
#' @param warmSteps optional naive-pooled warm-up steps before the loop (default 0).
#' @param optimizer torch optimizer, `"adam"` or `"sgd"`.
#' @param seed optional integer seed for the torch weight initialization.
#' @return an object of class `c("nnControl", class(inner))` carrying the joint
#'   schedule in its `"nnControl"` attribute.
#' @export
#' @author Matthew L. Fidler
nnControl <- function(inner = nlmixr2est::foceiControl(),
                      maxRounds = 200L, outerPerRound = 1L, wSteps = 2L,
                      tol = 1e-3, lr = 0.03, warmSteps = 0L,
                      optimizer = c("adam", "sgd"), seed = NULL) {
  optimizer <- match.arg(optimizer)
  if (!inherits(inner, "list") && !is.list(inner)) {
    stop("'inner' must be an nlmixr2 control object (e.g. foceiControl())",
         call. = FALSE)
  }
  checkmate::assertIntegerish(maxRounds, lower = 1L, len = 1L, .var.name = "maxRounds")
  checkmate::assertIntegerish(outerPerRound, lower = 1L, len = 1L, .var.name = "outerPerRound")
  checkmate::assertIntegerish(wSteps, lower = 1L, len = 1L, .var.name = "wSteps")
  checkmate::assertIntegerish(warmSteps, lower = 0L, len = 1L, .var.name = "warmSteps")
  checkmate::assertNumeric(tol, lower = 0, len = 1L, .var.name = "tol")
  checkmate::assertNumeric(lr, lower = 0, len = 1L, .var.name = "lr")
  .nn <- list(maxRounds = as.integer(maxRounds), outerPerRound = as.integer(outerPerRound),
              wSteps = as.integer(wSteps), tol = as.numeric(tol), lr = as.numeric(lr),
              warmSteps = as.integer(warmSteps), optimizer = optimizer,
              seed = if (is.null(seed)) NULL else as.integer(seed))
  .ctl <- inner
  attr(.ctl, "nnControl") <- .nn
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

#' nlmixr2 joint estimation method for embedded neural networks (`est = "nn"`)
#'
#' Not called directly -- dispatched by `nlmixr2(..., est = "nn",
#' control = nnControl(...))`.  Co-optimizes population parameters and NN weights
#' by interleaving warm-started partial inner steps with torch weight steps.
#' @param env nlmixr2 estimation environment.
#' @param ... ignored.
#' @return an `nlmixr2FitData` on the base model carrying the trained NN weights
#'   (as `rxForcedPars`) and the training trace in `$nnParHist`.
#' @exportS3Method nlmixr2est::nlmixr2Est
nlmixr2Est.nn <- function(env, ...) {
  .ui <- env$ui
  .data <- env$data
  .control <- env$control
  if (!inherits(.control, "nnControl")) {
    stop("est = 'nn' needs control = nnControl(...)", call. = FALSE)
  }
  .nn <- attr(.control, "nnControl")
  .innerEst <- .nnInnerEst(.control)
  .innerCtl <- .nnInnerControl(.control)
  ## partial warm-started steps: no per-round covariance/tables (restored for ONE
  ## final fit), and only `outerPerRound` outer iterations per round.
  .origTablesCov <- .nnStashTablesCov(.innerCtl)
  .innerCtl <- .nnDisableTablesCov(.innerCtl)
  if (!is.null(.innerCtl$maxOuterIterations)) .innerCtl$maxOuterIterations <- .nn$outerPerRound

  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) stop("est = 'nn' requires the libtorch backend (nnTorchAvailable() is FALSE)",
                call. = FALSE)

  .aug <- .nnAugmentFromUi(.ui)
  .data <- nnCovData(.data)
  .baseBase <- nnUpdate(.ui)$base[1L]
  nnTorchInit(.aug$id, .aug$K, .aug$H, act = .aug$act, seed = .nn$seed)
  nnTorchOptInit(.aug$id, .nn$optimizer, .nn$lr)
  on.exit(tryCatch(nnTorchFree(.aug$id), silent = TRUE), add = TRUE)

  .idCol <- if ("ID" %in% names(.data)) "ID" else "id"
  .obs <- .data[[if ("EVID" %in% names(.data)) "EVID" else "evid"]]
  .obs <- is.na(.obs) | .obs == 0
  .dv <- .data[[if ("DV" %in% names(.data)) "DV" else "dv"]]
  .wPlaceholder <- stats::setNames(rep(0, .aug$nW), .aug$weights)
  .weightStep <- .nnWeightStepper(.aug, .data, .idCol, .obs, .dv, .wPlaceholder)

  ## optional naive-pooled warm-up at eta=0 from the model initial parameters
  if (.nn$warmSteps > 0L) {
    .iniDf <- .ui$iniDf
    .th0 <- stats::setNames(.iniDf$est[!is.na(.iniDf$ntheta)], .iniDf$name[!is.na(.iniDf$ntheta)])
    .errPar0 <- list(add = if (is.na(.aug$errAdd)) 0 else unname(.th0[.aug$errAdd]),
                     prop = if (is.na(.aug$errProp)) 0 else unname(.th0[.aug$errProp]))
    if (.errPar0$add == 0 && .errPar0$prop == 0) .errPar0$add <- 1
    .ids <- as.character(unique(.data[[.idCol]]))
    .ebes0 <- stats::setNames(rep(0, length(.ids)), .ids)
    for (.ws in seq_len(.nn$warmSteps)) .weightStep(.ebes0, .errPar0, .th0)
  }

  ## interleaved joint loop: warm-started partial inner step + weight step(s),
  ## co-descending theta/Omega and the network weights until both stabilise.
  .parHist <- vector("list", .nn$maxRounds)
  .curUi <- .ui
  .fit <- NULL
  .wPrev <- nnTorchWeights(.aug$id)
  .objfPrev <- NA_real_
  .converged <- FALSE
  .nRun <- 0L
  .latent <- names(.aug$covMap)[1L]
  for (.round in seq_len(.nn$maxRounds)) {
    .nRun <- .round
    nnSetMeta(.aug$id, .baseBase, .aug$K, .aug$H, .aug$act)   # base-model weight base
    .w <- nnTorchWeights(.aug$id)
    nnSetWeights(.aug$id, .w)
    .dw <- .data
    for (.j in seq_along(.aug$weights)) .dw[[.aug$weights[.j]]] <- .w[.j]
    ## partial, warm-started inner step: .curUi carries the previous round's
    ## estimates so this resumes rather than restarts.
    .fit <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(.curUi, .dw, est = .innerEst, control = .innerCtl)))
    .curUi <- .fit$ui                                    # warm-start next round
    .ebes <- stats::setNames(.fit$eta[[.latent]], as.character(.fit$eta[["ID"]]))
    .thetas <- .fit$theta
    .errPar <- list(add = if (is.na(.aug$errAdd)) 0 else .fit$theta[[.aug$errAdd]],
                    prop = if (is.na(.aug$errProp)) 0 else .fit$theta[[.aug$errProp]])
    for (.ws in seq_len(.nn$wSteps)) .rmse <- .weightStep(.ebes, .errPar, .thetas)
    .wNow <- nnTorchWeights(.aug$id)
    .wChange <- sqrt(sum((.wNow - .wPrev)^2)) / (sqrt(sum(.wPrev^2)) + 1e-8)
    .wPrev <- .wNow
    .objfChange <- if (is.na(.objfPrev)) Inf else abs(.fit$objf - .objfPrev) / (abs(.objfPrev) + 1e-8)
    .objfPrev <- .fit$objf
    .parHist[[.round]] <- data.frame(round = .round, objf = .fit$objf,
                                     errAdd = .errPar$add, errProp = .errPar$prop,
                                     rmse = .rmse, wChange = .wChange, objfChange = .objfChange)
    if (.nn$tol > 0 && .round > 1L && .wChange < .nn$tol && .objfChange < .nn$tol) {
      .converged <- TRUE; break
    }
  }
  .parHist <- .parHist[seq_len(.nRun)]
  if (.converged) {
    message(sprintf("nn (joint) converged after %d rounds (weight change %.2g, objf change %.2g < tol %.2g)",
                    .nRun, .wChange, .objfChange, .nn$tol))
  } else {
    message(sprintf("nn (joint) stopped at the maximum %d rounds (weight change %.2g, objf change %.2g)",
                    .nRun, .wChange, .objfChange))
  }

  ## ONE final fit at the trained weights WITH the user's tables + covariance
  ## (skipped during the iteration), warm-started from the converged estimates.
  .trained <- stats::setNames(nnTorchWeights(.aug$id), .aug$weights)
  .finalCtl <- .nnRestoreTablesCov(.innerCtl, .origTablesCov)
  if (!is.null(.finalCtl$maxOuterIterations)) {
    .finalCtl$maxOuterIterations <- .nnInnerControl(.control)$maxOuterIterations
  }
  .fit <- .nnFinalFit(.curUi, .data, nnTorchWeights(.aug$id), .baseBase, .aug,
                      .innerEst, .finalCtl, .idCol)
  ## bake the trained weights into the fit's ui as forcedPars.
  .fitEnv <- .fit$env
  .storedUi <- rxode2::rxUiDecompress(get("ui", envir = .fitEnv))
  rxode2::rxForcedPars(.storedUi) <- .trained
  assign("ui", .storedUi, envir = .fitEnv)
  assign("nnParHist", do.call(rbind, .parHist), envir = .fitEnv)
  assign("nnWeights", .trained, envir = .fitEnv)
  assign("nnConverged", .converged, envir = .fitEnv)
  assign("nnRounds", .nRun, envir = .fitEnv)
  .fit
}
attr(nlmixr2Est.nn, "covariate") <- NULL
