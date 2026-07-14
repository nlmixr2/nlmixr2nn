## est = "nnIter": the ITERATIVE (block-coordinate) NN-in-ODE estimator.  This is
## NOT the DeepPumas joint optimization -- it is a solve/update/solve loop.
##
## Each round: (a) a FULL inner NLME fit of the BASE model with the current NN
## weights held fixed (population Omega, residual error, per-subject latent EBEs),
## then (b) a transient solve of the augmented (forward-sensitivity) model at those
## EBEs to get d(rx_pred_)/dw, from which dLL/dw = sum_obs dLL/df * d(rx_pred_)/dw
## drives a torch weight step.  The base model never carries sensitivity states;
## the augmented model is built, solved, and discarded inside the loop.  The loop
## repeats until the between-round weight change falls below `tol` or `rounds`.
##
## The final fit is the last inner base-model nlmixr2FitData with the trained
## weights attached as rxForcedPars so predict()/simulate() are self-contained.

## Parse the single (Gaussian) endpoint from the normalized model lines:
## `<pred> ~ add(<a>)`, `~ prop(<b>)`, or `~ add(<a>) + prop(<b>)`.  Returns
## list(state = pred var, add = additive-sd param or NA, prop = proportional-sd
## param or NA), or NULL when the error model is unsupported / not found.
.nnErrEndpoint <- function(lines) {
  .re <- "^\\s*([A-Za-z._][A-Za-z0-9._]*)\\s*~\\s*(.+?)\\s*$"
  .m <- regmatches(lines, regexec(.re, lines))
  .hit <- Filter(function(x) length(x) == 3L, .m)
  if (length(.hit) != 1L) return(NULL)
  .var <- .hit[[1L]][[2L]]
  .rhs <- .hit[[1L]][[3L]]
  .term <- function(fn) {
    .r <- sprintf("\\b%s\\(\\s*([A-Za-z._][A-Za-z0-9._]*)\\s*\\)", fn)
    if (!grepl(.r, .rhs)) return(NA_character_)
    regmatches(.rhs, regexec(.r, .rhs))[[1L]][[2L]]
  }
  .add <- .term("add"); .prop <- .term("prop")
  if (is.na(.add) && is.na(.prop)) return(NULL)   # only add/prop/combined for now
  list(state = .var, add = .add, prop = .prop)
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
       endpoint = .end$state, errAdd = .end$add, errProp = .end$prop,
       predswCols = sprintf("rx_predsw_%d_", seq_len(.nW) - 1L))
}

## The intermediate inner fits are throwaway, so their covariance/FIM and output
## tables are skipped; the user's original settings are stashed and restored for
## ONE final fit at the trained weights.  Shared by est="nnIter" and est="nn".
.nnStashTablesCov <- function(ctl) {
  list(calcTables = if (!is.null(ctl$calcTables)) ctl$calcTables else NULL,
       covMethod  = if (!is.null(ctl$covMethod))  ctl$covMethod  else NULL)
}
.nnDisableTablesCov <- function(ctl) {
  if (!is.null(ctl$calcTables)) ctl$calcTables <- FALSE
  if (!is.null(ctl$covMethod) &&
        (inherits(ctl, "foceiControl") || inherits(ctl, "saemControl"))) {
    ctl$covMethod <- ""
  }
  ctl
}
.nnRestoreTablesCov <- function(ctl, stash) {
  if (!is.null(stash$calcTables)) ctl$calcTables <- stash$calcTables
  if (!is.null(stash$covMethod))  ctl$covMethod  <- stash$covMethod
  ctl
}

## A final inner fit at the trained weights, with the user's original tables +
## covariance, warm-started from `ui`.  Returns the fit (the returned deliverable).
.nnFinalFit <- function(ui, data, weights, baseBase, aug, innerEst, finalCtl, idCol) {
  nnSetMeta(aug$id, baseBase, aug$K, aug$H, aug$act)
  nnSetWeights(aug$id, weights)
  .dw <- data
  for (.j in seq_along(aug$weights)) .dw[[aug$weights[.j]]] <- weights[.j]
  suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(ui, .dw, est = innerEst, control = finalCtl)))
}

## Factory for one torch weight step, shared by est="nnIter" and est="nn".  Given
## the augmented model + weight metadata and the fixed data pieces, returns a
## closure weightStep(ebes, errPar, thetas) that: sets the latent-eta covariates
## to the EBEs, solves the augmented model at the fitted thetas, forms the Gaussian
## cotangent dLL/df for variance R(f)=add^2+(prop*f)^2, assembles
## dLL/dw = sum_obs dLL/df * rx_predsw, and takes one torch optimizer step.
## Returns the RMSE.
.nnWeightStepper <- function(aug, data, idCol, obs, dv, wPlaceholder) {
  function(ebes, errPar, thetas) {
    .ad <- data
    for (.e in names(aug$covMap)) .ad[[aug$covMap[[.e]]]] <- ebes[as.character(.ad[[idCol]])]
    nnSetMeta(aug$id, aug$base, aug$K, aug$H, aug$act)   # augmented weight base
    nnSetWeights(aug$id, nnTorchWeights(aug$id))
    .p <- c(thetas, wPlaceholder)
    .s <- rxode2::rxSolve(aug$mAug, .ad, params = .p, returnType = "data.frame")
    if (!all(aug$predswCols %in% names(.s))) {
      stop("est = 'nn': augmented solve is missing the prediction-sensitivity ",
           "columns (rx_predsw_*)", call. = FALSE)
    }
    .ik <- match(paste(data[[idCol]][obs], data$time[obs]), paste(.s$id, .s$time))
    .f <- .s[[aug$endpoint]][.ik]
    .resid <- dv[obs] - .f
    .R <- errPar$add^2 + (errPar$prop * .f)^2
    .dRdf <- 2 * errPar$prop^2 * .f
    .dLLdf <- .resid / .R + 0.5 * (.resid^2 / .R^2 - 1 / .R) * .dRdf
    .dLLdw <- vapply(aug$predswCols, function(cn) sum(.dLLdf * .s[[cn]][.ik]), numeric(1))
    nnTorchZeroGrad(aug$id)
    nnTorchSetGrad(aug$id, -.dLLdw)
    nnTorchStep(aug$id)
    sqrt(mean(.resid^2))
  }
}

#' nlmixr2 iterative estimation method for embedded neural networks (`est = "nnIter"`)
#'
#' Not called directly -- dispatched by `nlmixr2(..., est = "nnIter",
#' control = nnIterControl(...))`.  Solve/update/solve: each round a full inner
#' NLME fit with the weights fixed, then torch weight steps; repeated until the
#' weights stop moving (`tol`) or `rounds` is reached.
#' @param env nlmixr2 estimation environment.
#' @param ... ignored.
#' @return an `nlmixr2FitData` on the base model carrying the trained NN weights
#'   (as `rxForcedPars`) and the training trace in `$nnParHist`.
#' @exportS3Method nlmixr2est::nlmixr2Est
nlmixr2Est.nnIter <- function(env, ...) {
  .ui <- env$ui
  .data <- env$data
  .control <- env$control
  if (!inherits(.control, "nnIterControl")) {
    stop("est = 'nnIter' needs control = nnIterControl(...)", call. = FALSE)
  }
  .nn <- attr(.control, "nnIterControl")
  .innerEst <- .nnInnerEst(.control)
  .innerCtl <- .nnInnerControl(.control)
  ## the intermediate fits are throwaway (only their EBEs + error params are
  ## used): skip the per-round covariance/FIM and output tables -- both are
  ## wasteful, and the covariance is often ill-conditioned at the round-1 random
  ## weights.  The user's original table/cov settings are restored for ONE final
  ## fit at the trained weights (see below).
  .origTablesCov <- .nnStashTablesCov(.innerCtl)
  .innerCtl <- .nnDisableTablesCov(.innerCtl)

  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) stop("est = 'nnIter' requires the libtorch backend (nnTorchAvailable() is FALSE)",
                call. = FALSE)

  ## augmented (forward-sensitivity) model + weight-block metadata
  .aug <- .nnAugmentFromUi(.ui)
  .data <- nnCovData(.data)
  ## the weight block sits at DIFFERENT par_ptr positions in the base model vs the
  ## augmented model (the base has leading thetas), so the injection base must be
  ## switched per context: `.baseBase` for the inner fit, `.aug$base` for the
  ## augmented solve.  Using the wrong one feeds the network garbage (FOCEi limps
  ## through it; SAEM's likelihood hits a non-PD matrix).
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

  ## warm-up: a naive-pooled step schedule at zero random effects, from the model
  ## initial parameters, so the first inner fit sees a non-degenerate network.
  if (.nn$warmSteps > 0L) {
    .iniDf <- .ui$iniDf
    .th0 <- stats::setNames(.iniDf$est[!is.na(.iniDf$ntheta)],
                            .iniDf$name[!is.na(.iniDf$ntheta)])
    .errPar0 <- list(add = if (is.na(.aug$errAdd)) 0 else unname(.th0[.aug$errAdd]),
                     prop = if (is.na(.aug$errProp)) 0 else unname(.th0[.aug$errProp]))
    if (.errPar0$add == 0 && .errPar0$prop == 0) .errPar0$add <- 1
    .ids <- as.character(unique(.data[[.idCol]]))
    .ebes0 <- stats::setNames(rep(0, length(.ids)), .ids)
    for (.ws in seq_len(.nn$warmSteps)) .weightStep(.ebes0, .errPar0, .th0)
  }

  ## iterate solve/update/solve until the weights stop moving (relative
  ## between-round change < tol) or `rounds` (the maximum) is reached.
  .parHist <- vector("list", .nn$rounds)
  .fit <- NULL
  .wPrev <- nnTorchWeights(.aug$id)
  .converged <- FALSE
  .nRun <- 0L
  for (.round in seq_len(.nn$rounds)) {
    .nRun <- .round
    ## Inject the current weights BOTH ways so the inner fit sees them regardless
    ## of estimator: (a) the par-loader (FOCEi's solve hits rxCallParLoaders) via
    ## nnSetWeights, and (b) the data covariate columns -- SAEM's estimation kernel
    ## does NOT call the par-loader, so it reads the weights straight from the data.
    nnSetMeta(.aug$id, .baseBase, .aug$K, .aug$H, .aug$act)   # base-model weight base
    .w <- nnTorchWeights(.aug$id)
    nnSetWeights(.aug$id, .w)
    .dw <- .data
    for (.j in seq_along(.aug$weights)) .dw[[.aug$weights[.j]]] <- .w[.j]
    .fit <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(.ui, .dw, est = .innerEst, control = .innerCtl)))
    .latent <- names(.aug$covMap)[1L]              # single latent eta name (MVP)
    .ebes <- stats::setNames(.fit$eta[[.latent]], as.character(.fit$eta[["ID"]]))
    .thetas <- .fit$theta
    .errPar <- list(add = if (is.na(.aug$errAdd)) 0 else .fit$theta[[.aug$errAdd]],
                    prop = if (is.na(.aug$errProp)) 0 else .fit$theta[[.aug$errProp]])
    for (.ws in seq_len(.nn$wSteps)) .rmse <- .weightStep(.ebes, .errPar, .thetas)
    ## relative change in the flattened weight vector since the last round
    .wNow <- nnTorchWeights(.aug$id)
    .wChange <- sqrt(sum((.wNow - .wPrev)^2)) / (sqrt(sum(.wPrev^2)) + 1e-8)
    .wPrev <- .wNow
    .parHist[[.round]] <- data.frame(round = .round, objf = .fit$objf,
                                     errAdd = .errPar$add, errProp = .errPar$prop,
                                     rmse = .rmse, wChange = .wChange)
    if (.nn$tol > 0 && .wChange < .nn$tol) { .converged <- TRUE; break }
  }
  .parHist <- .parHist[seq_len(.nRun)]
  if (.converged) {
    message(sprintf("nnIter converged after %d rounds (weight change %.2g < tol %.2g)",
                    .nRun, .wChange, .nn$tol))
  } else {
    message(sprintf("nnIter stopped at the maximum %d rounds (weight change %.2g, tol %.2g)",
                    .nRun, .wChange, .nn$tol))
  }

  ## ONE final fit at the trained weights WITH the user's tables + covariance
  ## (skipped during the iteration); this is the returned deliverable.
  .trained <- stats::setNames(nnTorchWeights(.aug$id), .aug$weights)
  .finalCtl <- .nnRestoreTablesCov(.innerCtl, .origTablesCov)
  .fit <- .nnFinalFit(.ui, .data, nnTorchWeights(.aug$id), .baseBase, .aug,
                      .innerEst, .finalCtl, .idCol)
  ## bake the trained weights into the fit's ui as forcedPars so predict()/
  ## simulate() reproduce them.  f$ui returns a CLONE, so write the ui STORED in
  ## the fit env.
  .fitEnv <- .fit$env
  .storedUi <- rxode2::rxUiDecompress(get("ui", envir = .fitEnv))
  rxode2::rxForcedPars(.storedUi) <- .trained
  assign("ui", .storedUi, envir = .fitEnv)
  ## store the NN metadata in the fit env (NOT via `$<-`, which would try to add a
  ## data column); `fit$nnParHist` etc. fall back to the env.
  assign("nnParHist", do.call(rbind, .parHist), envir = .fitEnv)
  assign("nnWeights", .trained, envir = .fitEnv)
  assign("nnConverged", .converged, envir = .fitEnv)
  assign("nnRounds", .nRun, envir = .fitEnv)
  .fit
}
attr(nlmixr2Est.nnIter, "covariate") <- NULL
