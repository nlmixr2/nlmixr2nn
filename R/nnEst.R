## est = "nn": the DeepPumas-style alternating estimator.
##
## Each round: (a) inner NLME fit of the BASE model with the current NN weights
## injected (population Omega, residual error, per-subject latent EBEs), then
## (b) a transient solve of the augmented (forward-sensitivity) model at those
## EBEs to get d(rx_pred_)/dw, from which dLL/dw = sum_obs dLL/df * d(rx_pred_)/dw
## drives a torch weight step.  The base model never carries sensitivity states;
## the augmented model is built, solved, and discarded inside the loop.
##
## The final fit is the last inner base-model nlmixr2FitData with the trained
## weights attached as rxForcedPars so predict()/simulate() are self-contained.

## Parse the single additive endpoint from the normalized model lines:
## `<state> ~ add(<sdName>)`.  Returns list(state, sd) or NULL.
.nnErrEndpoint <- function(lines) {
  .re <- "^\\s*([A-Za-z._][A-Za-z0-9._]*)\\s*~\\s*add\\(\\s*([A-Za-z._][A-Za-z0-9._]*)\\s*\\)\\s*$"
  .m <- regmatches(lines, regexec(.re, lines))
  .hit <- Filter(function(x) length(x) == 3L, .m)
  if (length(.hit) != 1L) return(NULL)
  list(state = .hit[[1L]][[2L]], sd = .hit[[1L]][[3L]])
}

## d(prediction)/d(state) for each ODE state, so the prediction's forward
## sensitivity wrt a weight can be chained from the state sensitivities rx_sw:
## d(pred)/dw = sum_s d(pred)/d(s) * rx_sw_s.  When the prediction IS a state
## this is the identity (1 for that state); when it is an lhs (e.g. cp = centr/V)
## the chain is nontrivial.  Returns a character vector of derivative expressions
## (rxode2 syntax) named by state.
.nnDpDs <- function(modelText, states, predVar) {
  if (predVar %in% states) {
    return(stats::setNames(as.character(as.integer(states == predVar)), states))
  }
  .model <- rxode2::rxS(rxode2::rxGetModel(modelText), TRUE, promoteLinSens = FALSE)
  .p <- get0(predVar, envir = .model, inherits = FALSE)
  if (is.null(.p)) {
    stop("est = 'nn': cannot resolve the prediction variable '", predVar, "'",
         call. = FALSE)
  }
  vapply(states, function(s) {
    .dd <- symengine::D(.p, symengine::Symbol(s))   # bind before rxFromSE (NSE)
    rxode2::rxFromSE(.dd)
  }, character(1))
}

## Auto-build the ephemeral augmented model from a base nn() ui: drop the weight
## dummy-covariate line and the error line, rename the latent eta(s) to
## covariates, declare param(weights, latent covariates), and hand to
## nnAugmentModel().  Returns the compiled augmented model + the metadata the
## weight step needs.
.nnAugmentFromUi <- function(ui) {
  .reg <- .nnEnv$reg
  if (length(.reg) != 1L) {
    stop("est = 'nn' currently supports exactly one nn() term", call. = FALSE)
  }
  .m <- .reg[[1L]]
  .lines <- ui$lstChr
  .end <- .nnErrEndpoint(.lines)
  if (is.null(.end)) {
    stop("est = 'nn' currently supports a single additive endpoint (var ~ add(sd))",
         call. = FALSE)
  }
  ## drop the weight dummy-covariate declaration and the error line(s)
  .keep <- .lines[!grepl("^\\s*rx_nnw[0-9]+_\\s*<-", .lines) & !grepl("~", .lines)]
  ## latent etas among the nn inputs -> covariate names (dots -> underscores)
  .etas <- ui$eta
  .covMap <- stats::setNames(gsub("[^A-Za-z0-9_]", "_", .etas), .etas)
  for (.e in .etas) {
    .keep <- gsub(paste0("\\b", gsub("\\.", "\\\\.", .e), "\\b"), .covMap[[.e]], .keep)
  }
  ## the model's non-weight covariates are NN inputs too (e.g. WT in
  ## `nn(WT, eta.nn)`) -- declare them so the augmented solve reads them from the
  ## data, alongside the weight block and the latent-eta covariates.
  .realCovs <- setdiff(ui$allCovs, .m$weights)
  .param <- paste0("param(",
                   paste(c(.m$weights, .realCovs, unname(.covMap)), collapse = ", "), ")")
  .augBase <- paste(c(.param, .keep), collapse = "\n")
  .augText <- nnAugmentModel(.augBase, H = .m$H)
  .nW <- .m$H * .m$K + 2L * .m$H + 1L
  ## prediction forward sensitivity wrt each weight, chained through the states:
  ## rx_predsw_<j>_ = sum_s d(pred)/d(s) * rx_sw_<s>_<j>_.  Emitting this as an lhs
  ## makes the endpoint work whether it is a raw state or an lhs (e.g. cp=centr/V)
  ## with no per-endpoint code in the weight step.
  .states <- rxode2::rxStateOde(rxode2::rxS(rxode2::rxGetModel(.augBase), TRUE,
                                            promoteLinSens = FALSE))
  .dpds <- .nnDpDs(.augBase, .states, .end$state)
  if (all(.dpds == "0")) {
    stop("est = 'nn': the prediction '", .end$state,
         "' does not depend on any ODE state -- nothing for the network to fit",
         call. = FALSE)
  }
  .predsw <- vapply(seq_len(.nW) - 1L, function(j) {
    .terms <- character(0)
    for (.si in seq_along(.states)) {
      if (!identical(.dpds[[.si]], "0") && nzchar(.dpds[[.si]])) {
        .terms <- c(.terms, sprintf("(%s)*rx_sw_%s_%d_", .dpds[[.si]], .states[.si], j))
      }
    }
    sprintf("rx_predsw_%d_ = %s", j, paste(.terms, collapse = " + "))
  }, character(1))
  .augText <- paste(c(.augText, .predsw), collapse = "\n")
  list(text = .augText,
       mAug = rxode2::rxode2(.augText),
       base = .nnWeightBase(rxode2::rxode2(.augBase), .m$id, .m$K, .m$H),
       id = .m$id, K = .m$K, H = .m$H, act = .m$act,
       weights = .m$weights, nW = .nW, covMap = .covMap, realCovs = .realCovs,
       endpoint = .end$state, sdName = .end$sd,
       predswCols = sprintf("rx_predsw_%d_", seq_len(.nW) - 1L))
}

#' nlmixr2 estimation method for embedded neural networks (`est = "nn"`)
#'
#' Not called directly -- dispatched by `nlmixr2(..., est = "nn",
#' control = nnControl(...))`.
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

  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) stop("est = 'nn' requires the libtorch backend (nnTorchAvailable() is FALSE)",
                call. = FALSE)

  ## augmented (forward-sensitivity) model + weight-block metadata
  .aug <- .nnAugmentFromUi(.ui)
  .data <- nnCovData(.data)
  nnUpdate(.ui)                                    # register the base network layout
  ## torch module for the network weights
  nnSetMeta(.aug$id, .aug$base, .aug$K, .aug$H, .aug$act)
  nnTorchInit(.aug$id, .aug$K, .aug$H, act = .aug$act, seed = .nn$seed)
  nnTorchOptInit(.aug$id, .nn$optimizer, .nn$lr)
  on.exit(tryCatch(nnTorchFree(.aug$id), silent = TRUE), add = TRUE)

  .idCol <- if ("ID" %in% names(.data)) "ID" else "id"
  .obs <- .data[[if ("EVID" %in% names(.data)) "EVID" else "evid"]]
  .obs <- is.na(.obs) | .obs == 0
  .dv <- .data[[if ("DV" %in% names(.data)) "DV" else "dv"]]
  .wPlaceholder <- stats::setNames(rep(0, .aug$nW), .aug$weights)

  ## one torch weight step from the per-subject EBEs + current residual SD.
  ## `thetas` are the fitted population parameters -- supplied so NN inputs that
  ## are computed parameters (e.g. `nn(cl, eta.nn)` with `cl <- exp(tcl)`) take
  ## their fitted values in the augmented solve; NN input covariates (e.g. WT)
  ## ride in the data.
  .weightStep <- function(ebes, sigma, thetas) {
    .ad <- .data
    for (.e in names(.aug$covMap)) .ad[[.aug$covMap[[.e]]]] <- ebes[as.character(.ad[[.idCol]])]
    nnSetWeights(.aug$id, nnTorchWeights(.aug$id))
    .p <- c(thetas, .wPlaceholder)
    .s <- rxode2::rxSolve(.aug$mAug, .ad, params = .p, returnType = "data.frame")
    .ik <- match(paste(.data[[.idCol]][.obs], .data$time[.obs]),
                 paste(.s$id, .s$time))
    .resid <- .dv[.obs] - .s[[.aug$endpoint]][.ik]
    .dLLdf <- .resid / sigma^2
    if (!all(.aug$predswCols %in% names(.s))) {
      stop("est = 'nn': augmented solve is missing the prediction-sensitivity ",
           "columns (rx_predsw_*)", call. = FALSE)
    }
    .dLLdw <- vapply(.aug$predswCols, function(cn) sum(.dLLdf * .s[[cn]][.ik]), numeric(1))
    nnTorchZeroGrad(.aug$id)
    nnTorchSetGrad(.aug$id, -.dLLdw)
    nnTorchStep(.aug$id)
    sqrt(mean(.resid^2))
  }

  .parHist <- vector("list", .nn$rounds)
  .fit <- NULL
  for (.round in seq_len(.nn$rounds)) {
    nnSetWeights(.aug$id, nnTorchWeights(.aug$id))
    .fit <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(.ui, .data, est = .innerEst, control = .innerCtl)))
    .latent <- names(.aug$covMap)[1L]              # single latent eta name (MVP)
    .ebes <- stats::setNames(.fit$eta[[.latent]], as.character(.fit$eta[["ID"]]))
    .sigma <- .fit$theta[[.aug$sdName]]
    .thetas <- .fit$theta
    for (.ws in seq_len(.nn$wSteps)) .rmse <- .weightStep(.ebes, .sigma, .thetas)
    .parHist[[.round]] <- data.frame(round = .round, objf = .fit$objf,
                                     add.sd = .sigma, rmse = .rmse)
  }

  ## finalize: the last inner fit is the base-model fit; bake the trained weights
  ## into the ui as forcedPars so predict()/simulate() reproduce them with no
  ## torch/loader state, and attach the training trace.  f$ui returns a CLONE, so
  ## the forced weights must be written into the ui STORED in the fit env.
  .trained <- stats::setNames(nnTorchWeights(.aug$id), .aug$weights)
  .fitEnv <- .fit$env
  .storedUi <- rxode2::rxUiDecompress(get("ui", envir = .fitEnv))
  rxode2::rxForcedPars(.storedUi) <- .trained
  assign("ui", .storedUi, envir = .fitEnv)
  .fit$nnParHist <- do.call(rbind, .parHist)
  .fit$nnWeights <- .trained
  .fit
}
attr(nlmixr2Est.nn, "covariate") <- NULL
