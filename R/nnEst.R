## NN-in-ODE training engine (.nnRun) for the transparent nlmixr2nn workflow.
## It is driven by nlmixr2nn's estimation interceptor (R/nnInterceptor.R): a model
## with an nn() term fitted with a standard est (focei/saem/...) is claimed here.
##
## Two modes (nnControl(mode=)):
## * "iter" -- block-coordinate solve/update/solve: each round a FULL inner NLME
##   fit of the BASE model with the weights held fixed, then torch weight steps.
## * "joint" -- DeepPumas-style interleave: each round a warm-started PARTIAL inner
##   step (outerPerRound outer iterations) + weight steps, co-descending the
##   population parameters and the weights (only when the inner estimator exposes
##   maxOuterIterations; otherwise it degrades to "iter").
##
## Each round the weight gradient comes from a transient solve of the augmented
## (forward-sensitivity) model at the inner EBEs: dLL/dw = sum_obs dLL/df *
## d(rx_pred_)/dw.  The base model never carries sensitivity states; the augmented
## model is built, solved, and discarded inside the loop.  The loop stops when the
## between-round weight change (and, for "joint", the objective change) falls below
## `tol`, or at `rounds`.
##
## The returned fit is the last inner base-model nlmixr2FitData with the trained
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
    stop("nlmixr2nn: cannot resolve the prediction variable '", predVar, "'",
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
    stop("nlmixr2nn currently supportsexactly one nn() term", call. = FALSE)
  }
  .m <- .reg[[1L]]
  .lines <- ui$lstChr
  .end <- .nnErrEndpoint(.lines)
  if (is.null(.end)) {
    stop("nlmixr2nn currently supportsa single additive endpoint (var ~ add(sd))",
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
    stop("nlmixr2nn: the prediction '", .end$state,
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
## tables are skipped; the user's original settings are stashed and restored when
## the tables + covariance are added post-hoc to the final fit.
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
## Add output tables + covariance to the ALREADY-FITTED object post-hoc, from the
## user's original control settings -- NO re-fitting.  addTable() computes the
## residual/table columns; .setCov() computes the covariance at the converged
## parameters (maxOuterIterations = 0, so no re-estimation).  The trained weights
## must already be injected (data columns + loader) so both solves use them.
.nnAddTablesCov <- function(fit, weights, baseBase, aug, innerEst, orig) {
  nnSetMeta(aug$id, baseBase, aug$K, aug$H, aug$act)
  nnSetWeights(aug$id, weights)
  if (isTRUE(orig$calcTables)) {
    fit <- tryCatch(nlmixr2est:::addTable(fit), error = function(e) fit)
  }
  .cm <- orig$covMethod
  if (!is.null(.cm) && !identical(.cm, "") && grepl("focei?$|^i?focei?", innerEst)) {
    ## post-hoc FOCEi covariance is a finite-difference r/s calc at the converged
    ## estimates; `analytic`/other labels fall back to "r,s".
    .post <- if (.cm %in% c("r,s", "r", "s")) .cm else "r,s"
    tryCatch(nlmixr2est:::.setCov(fit, covMethod = .post), error = function(e) NULL)
  }
  fit
}

## Factory for one torch weight step (shared by both nn training modes).  Given
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
      stop("nlmixr2nn: augmented solve is missing the prediction-sensitivity ",
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

## Warm-start weights from a ui that already carries trained weights (as
## rxForcedPars on the weight covariates -- a previous nn fit, or an nlm
## population warm-start).  Returns the weight vector (aug$weights order) or NULL.
.nnExistingWeights <- function(ui, aug) {
  .fp <- tryCatch(rxode2::rxForcedPars(ui), error = function(e) NULL)
  if (is.null(.fp)) return(NULL)
  .w <- .fp[aug$weights]
  if (length(.w) != aug$nW || anyNA(.w)) return(NULL)
  unname(.w)
}

## Which control field caps the inner outer iterations -- the knob that makes a
## partial, warm-startable step for mode="joint"?  Detected by field presence so
## it is estimator-agnostic: maxOuterIterations (FOCEi family: focei/foce/foi/
## laplace/agq + the mu*/i* variants) or iters (the variational advi/vae).  Both
## genuinely RESUME from a ui carrying the previous round's estimates (a gradient
## optimizer restarts at those estimates; the variational fit re-optimizes from
## that point), so warm-started partial steps co-descend with the weight steps.
## Returns NULL for estimators that do NOT resume this way -- saem/fsaem (the
## stochastic-approximation gain sequence restarts each call, so nEm chunks do not
## resume the SA chain and interleave is no better than a full fit) and imp/qrpem/
## nlm-family (no outer-iteration knob) -- which use the block-coordinate iterate
## loop instead.
.nnInterleaveKnob <- function(ctl) {
  if (!is.null(ctl$maxOuterIterations)) return("maxOuterIterations")
  if (!is.null(ctl$iters))              return("iters")
  NULL
}

## Run the NN training loop.  `env` carries ui/data + the standard inner estimator
## (class(env)[1]) and its control (env$control); `sched` is an nnControl().
## mode="joint" interleaves warm-started PARTIAL inner steps (outerPerRound outer
## iterations, via the estimator's outer-step knob) with weight steps -- co-
## descending parameters and weights -- and falls back to the block-coordinate
## iterate loop for inner estimators without a partial outer step; mode="iter"
## runs a full inner fit each round.  The last inner fit is the returned
## deliverable, with tables/covariance added post-hoc.
.nnRun <- function(env, sched) {
  .ui <- rxode2::rxUiDecompress(env$ui)
  .data <- env$data
  .innerEst <- class(env)[1L]
  ## env$control is already a validated control for .innerEst (nlmixr2() fills it
  ## before dispatch), so it is used directly as the inner NLME control.
  .innerCtl <- env$control
  .origTablesCov <- .nnStashTablesCov(.innerCtl)
  .innerCtl <- .nnDisableTablesCov(.innerCtl)

  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  if (!ok) stop("nlmixr2nn neural-network training requires the libtorch backend",
                call. = FALSE)

  .aug <- .nnAugmentFromUi(.ui)
  .data <- nnCovData(.data)
  ## the weight block sits at DIFFERENT par_ptr positions in the base vs the
  ## augmented model, so the injection base is switched per context (base-model
  ## base for the inner fit, augmented base for the sensitivity solve).
  .baseBase <- nnUpdate(.ui)$base[1L]
  nnTorchInit(.aug$id, .aug$K, .aug$H, act = .aug$act, seed = sched$seed)
  ## warm start from existing built-in weights when the model already carries them
  .existing <- .nnExistingWeights(.ui, .aug)
  if (!is.null(.existing)) nnTorchSetWeights(.aug$id, .existing)
  nnTorchOptInit(.aug$id, sched$optimizer, sched$lr)
  on.exit(tryCatch(nnTorchFree(.aug$id), silent = TRUE), add = TRUE)

  .idCol <- if ("ID" %in% names(.data)) "ID" else "id"
  .obs <- .data[[if ("EVID" %in% names(.data)) "EVID" else "evid"]]
  .obs <- is.na(.obs) | .obs == 0
  .dv <- .data[[if ("DV" %in% names(.data)) "DV" else "dv"]]
  .wPlaceholder <- stats::setNames(rep(0, .aug$nW), .aug$weights)
  .weightStep <- .nnWeightStepper(.aug, .data, .idCol, .obs, .dv, .wPlaceholder)
  .latent <- names(.aug$covMap)[1L]

  ## true interleave only when the inner estimator exposes a partial outer step
  .knob <- .nnInterleaveKnob(.innerCtl)
  .interleave <- sched$mode == "joint" && !is.null(.knob)
  if (sched$mode == "joint" && !.interleave) {
    message(sprintf("est=\"%s\" has no partial outer step; nn joint uses the iterative loop",
                    .innerEst))
  }

  ## optional naive-pooled warm-up at eta=0 from the model initial parameters
  if (sched$warmSteps > 0L) {
    .iniDf <- .ui$iniDf
    .th0 <- stats::setNames(.iniDf$est[!is.na(.iniDf$ntheta)], .iniDf$name[!is.na(.iniDf$ntheta)])
    .errPar0 <- list(add = if (is.na(.aug$errAdd)) 0 else unname(.th0[.aug$errAdd]),
                     prop = if (is.na(.aug$errProp)) 0 else unname(.th0[.aug$errProp]))
    if (.errPar0$add == 0 && .errPar0$prop == 0) .errPar0$add <- 1
    .ids <- as.character(unique(.data[[.idCol]]))
    .ebes0 <- stats::setNames(rep(0, length(.ids)), .ids)
    for (.ws in seq_len(sched$warmSteps)) .weightStep(.ebes0, .errPar0, .th0)
  }

  .parHist <- vector("list", sched$rounds)
  .fit <- NULL
  .curUi <- .ui
  .wPrev <- nnTorchWeights(.aug$id)
  .objfPrev <- NA_real_
  .converged <- FALSE
  .nRun <- 0L
  for (.round in seq_len(sched$rounds)) {
    .nRun <- .round
    ## inject the current weights BOTH ways (par-loader for FOCEi's solve + data
    ## covariate columns for SAEM's kernel, which bypasses the loader).
    nnSetMeta(.aug$id, .baseBase, .aug$K, .aug$H, .aug$act)   # base-model weight base
    .w <- nnTorchWeights(.aug$id)
    nnSetWeights(.aug$id, .w)
    .dw <- .data
    for (.j in seq_along(.aug$weights)) .dw[[.aug$weights[.j]]] <- .w[.j]
    ## interleave: warm-started PARTIAL step from the previous ui; else full fit
    .fitUi <- if (.interleave) .curUi else .ui
    .roundCtl <- .innerCtl
    if (.interleave) .roundCtl[[.knob]] <- sched$outerPerRound
    .fit <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(.fitUi, .dw, est = .innerEst, control = .roundCtl)))
    if (.interleave) .curUi <- .fit$ui                       # warm-start next round
    .ebes <- stats::setNames(.fit$eta[[.latent]], as.character(.fit$eta[["ID"]]))
    .thetas <- .fit$theta
    .errPar <- list(add = if (is.na(.aug$errAdd)) 0 else .fit$theta[[.aug$errAdd]],
                    prop = if (is.na(.aug$errProp)) 0 else .fit$theta[[.aug$errProp]])
    for (.ws in seq_len(sched$wSteps)) .rmse <- .weightStep(.ebes, .errPar, .thetas)
    .wNow <- nnTorchWeights(.aug$id)
    .wChange <- sqrt(sum((.wNow - .wPrev)^2)) / (sqrt(sum(.wPrev^2)) + 1e-8)
    .wPrev <- .wNow
    .objfChange <- if (is.na(.objfPrev)) Inf else abs(.fit$objf - .objfPrev) / (abs(.objfPrev) + 1e-8)
    .objfPrev <- .fit$objf
    .parHist[[.round]] <- data.frame(round = .round, objf = .fit$objf,
                                     errAdd = .errPar$add, errProp = .errPar$prop,
                                     rmse = .rmse, wChange = .wChange, objfChange = .objfChange)
    ## stop when the weights (and, when interleaving, the objective) stabilise
    .stop <- .wChange < sched$tol && (!.interleave || .objfChange < sched$tol)
    if (sched$tol > 0 && .round > 1L && .stop) { .converged <- TRUE; break }
  }
  .parHist <- .parHist[seq_len(.nRun)]
  message(sprintf("nn (%s) %s after %d rounds (weight change %.2g%s)",
                  if (.interleave) "joint" else "iterative",
                  if (.converged) "converged" else "stopped at max rounds", .nRun, .wChange,
                  if (.interleave) sprintf(", objf change %.2g", .objfChange) else ""))

  ## the last inner fit IS the deliverable (no re-fit); add tables + covariance
  ## post-hoc from the user's original control settings.
  .trained <- stats::setNames(nnTorchWeights(.aug$id), .aug$weights)
  .fit <- .nnAddTablesCov(.fit, nnTorchWeights(.aug$id), .baseBase, .aug, .innerEst, .origTablesCov)
  .fitEnv <- .fit$env
  .storedUi <- rxode2::rxUiDecompress(get("ui", envir = .fitEnv))
  rxode2::rxForcedPars(.storedUi) <- .trained
  assign("ui", .storedUi, envir = .fitEnv)
  ## NN metadata in the fit env (NOT $<-, which would add a data column)
  assign("nnParHist", do.call(rbind, .parHist), envir = .fitEnv)
  assign("nnWeights", .trained, envir = .fitEnv)
  assign("nnConverged", .converged, envir = .fitEnv)
  assign("nnRounds", .nRun, envir = .fitEnv)
  .fit
}
