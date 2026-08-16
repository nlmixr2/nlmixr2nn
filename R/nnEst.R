## The nn weight par-loader is registered under the NAME "nlmixr2nn:nnParLoader"
## (see src/nlmixr2nnPtr.cpp), so rxode2 runs it ONLY while this injector is the
## active flag -- otherwise it would clobber an unrelated model's par_ptr.  A model
## carrying nn() is flagged on its ui (nnUpdate() sets rxParLoader()), which the
## rxSolve.rxUi bridge honors; the estimation engine below also sets the flag
## directly around its internal solves (inner fits + augmented solves) because they
## bypass that bridge.  Both no-op gracefully on an older rxode2.
.nnLoaderName <- "nlmixr2nn:nnParLoader"
.nnLoaderOn <- function() {
  tryCatch(.Call("_rxode2_rxSetActiveParLoader", .nnLoaderName, PACKAGE = "rxode2"),
           error = function(e) NULL)
}
.nnLoaderOff <- function() {
  tryCatch(.Call("_rxode2_rxClearActiveParLoader", PACKAGE = "rxode2"),
           error = function(e) NULL)
}

## Per-observation error-model cotangent capture (nlmixr2est likelihood-contribution
## hook).  .nnCapReset(TRUE) clears + arms it before an inner fit; .nnCapGet()
## returns list(id, k, dLLdf) of the fit's converged per-obs cotangents.  No-op on
## an nlmixr2est without the lik-contrib API.
.nnCapReset <- function(on) {
  tryCatch(.Call("_nlmixr2nn_capReset", on, PACKAGE = "nlmixr2nn"), error = function(e) NULL)
}
.nnCapGet <- function() {
  tryCatch(.Call("_nlmixr2nn_capGet", PACKAGE = "nlmixr2nn"), error = function(e) NULL)
}

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
  ## add/prop give the closed-form Gaussian cotangent; any OTHER error model (add
  ## and prop both NA, e.g. lnorm / transform-both-sides) still yields the endpoint
  ## state -- its cotangent then comes from the inner fit (cotangent = "exact").
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
  if (length(.reg) == 0L) {
    stop("nlmixr2nn: no nn() term found in the model", call. = FALSE)
  }
  ## networks ordered by id -> a stable GLOBAL weight layout (net 0's weights, then
  ## net 1's, ...), matching nnAugmentModel's rx_sw_<state>_<globalj>_ indexing.
  .nets <- .reg[order(vapply(.reg, function(m) m$id, integer(1)))]
  .lines <- ui$lstChr
  .end <- .nnErrEndpoint(.lines)
  if (is.null(.end)) {
    stop("nlmixr2nn currently supports a single additive endpoint (var ~ add(sd))",
         call. = FALSE)
  }
  ## drop the weight dummy-covariate declaration(s) and the error line(s)
  .keep <- .lines[!grepl("^\\s*rx_nnw[0-9]+_\\s*<-", .lines) & !grepl("~", .lines)]
  ## latent etas among the nn inputs -> covariate names (dots -> underscores)
  .etas <- ui$eta
  .covMap <- stats::setNames(gsub("[^A-Za-z0-9_]", "_", .etas), .etas)
  for (.e in .etas) {
    .keep <- gsub(paste0("\\b", gsub("\\.", "\\\\.", .e), "\\b"), .covMap[[.e]], .keep)
  }
  ## all networks' weights + the model's non-weight covariates (also NN inputs, e.g.
  ## WT in `nn(WT, eta.nn)`) + the latent-eta covariates, declared so the augmented
  ## solve reads them from the data.
  .allW <- unlist(lapply(.nets, function(m) m$weights), use.names = FALSE)
  .realCovs <- setdiff(ui$allCovs, .allW)
  .param <- paste0("param(",
                   paste(c(.allW, .realCovs, unname(.covMap)), collapse = ", "), ")")
  .augBase <- paste(c(.param, .keep), collapse = "\n")
  .H <- stats::setNames(vapply(.nets, function(m) as.integer(m$H), integer(1)),
                        vapply(.nets, function(m) as.character(m$id), character(1)))
  .augText <- nnAugmentModel(.augBase, H = .H)
  .totW <- sum(vapply(.nets, function(m) as.integer(m$H * m$K + 2L * m$H + 1L), integer(1)))
  ## prediction forward sensitivity wrt each GLOBAL weight, chained through the
  ## states: rx_predsw_<globalj>_ = sum_s d(pred)/d(s) * rx_sw_<s>_<globalj>_.
  .states <- rxode2::rxStateOde(rxode2::rxS(rxode2::rxGetModel(.augBase), TRUE,
                                            promoteLinSens = FALSE))
  .dpds <- .nnDpDs(.augBase, .states, .end$state)
  if (all(.dpds == "0")) {
    stop("nlmixr2nn: the prediction '", .end$state,
         "' does not depend on any ODE state -- nothing for the network to fit",
         call. = FALSE)
  }
  .predsw <- vapply(seq_len(.totW) - 1L, function(j) {
    .terms <- character(0)
    for (.si in seq_along(.states)) {
      if (!identical(.dpds[[.si]], "0") && nzchar(.dpds[[.si]])) {
        .terms <- c(.terms, sprintf("(%s)*rx_sw_%s_%d_", .dpds[[.si]], .states[.si], j))
      }
    }
    sprintf("rx_predsw_%d_ = %s", j, paste(.terms, collapse = " + "))
  }, character(1))
  .augText <- paste(c(.augText, .predsw), collapse = "\n")
  .mAugBase <- rxode2::rxode2(.augBase)
  ## per-network metadata carrying the global-weight offset + the network's weight
  ## base in the augmented base model (each net's block is contiguous there).
  .off <- 0L
  .netMeta <- lapply(.nets, function(m) {
    .nWm <- as.integer(m$H * m$K + 2L * m$H + 1L)
    .meta <- list(id = m$id, K = m$K, H = m$H, act = m$act, weights = m$weights, nW = .nWm,
                  offset = .off, gIdx = .off + seq_len(.nWm),   # 1-based global indices
                  augBase = .nnWeightBase(.mAugBase, m$id, m$K, m$H),
                  predswCols = sprintf("rx_predsw_%d_", .off + seq_len(.nWm) - 1L))
    .off <<- .off + .nWm
    .meta
  })
  list(text = .augText, mAug = rxode2::rxode2(.augText),
       covMap = .covMap, realCovs = .realCovs, weights = .allW, nW = .totW,
       endpoint = .end$state, errAdd = .end$add, errProp = .end$prop,
       predswCols = sprintf("rx_predsw_%d_", seq_len(.totW) - 1L),
       nets = .netMeta)
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
.nnAddTablesCov <- function(fit, weights, baseBases, aug, innerEst, orig) {
  for (.net in aug$nets) {                       # each net at its base-model base
    nnSetMeta(.net$id, baseBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
    nnSetWeights(.net$id, weights[.net$gIdx])
  }
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
  ## weightStep(ebes, errPar, thetas, dLLdfObs = NULL): dLLdfObs, when supplied, is
  ## the per-observation error-model cotangent captured from the inner fit (the
  ## EXACT dLL/df for any residual model), aligned to the observation rows; NULL
  ## uses the closed-form additive/proportional-Gaussian cotangent.
  function(ebes, errPar, thetas, dLLdfObs = NULL) {
    .ad <- data
    for (.e in names(aug$covMap)) .ad[[aug$covMap[[.e]]]] <- ebes[as.character(.ad[[idCol]])]
    for (.net in aug$nets) {                            # each net at its augmented base
      nnSetMeta(.net$id, .net$augBase, .net$K, .net$H, .net$act)
      nnSetWeights(.net$id, nnTorchWeights(.net$id))
    }
    .p <- c(thetas, wPlaceholder)
    .s <- rxode2::rxSolve(aug$mAug, .ad, params = .p, returnType = "data.frame")
    if (!all(aug$predswCols %in% names(.s))) {
      stop("nlmixr2nn: augmented solve is missing the prediction-sensitivity ",
           "columns (rx_predsw_*)", call. = FALSE)
    }
    .ik <- match(paste(data[[idCol]][obs], data$time[obs]), paste(.s$id, .s$time))
    .f <- .s[[aug$endpoint]][.ik]
    .resid <- dv[obs] - .f
    if (!is.null(dLLdfObs)) {
      .dLLdf <- dLLdfObs                               # exact cotangent from the inner fit
    } else {
      .R <- errPar$add^2 + (errPar$prop * .f)^2
      .dRdf <- 2 * errPar$prop^2 * .f
      .dLLdf <- .resid / .R + 0.5 * (.resid^2 / .R^2 - 1 / .R) * .dRdf
    }
    for (.net in aug$nets) {                            # per-network gradient + step
      .dLLdw <- vapply(.net$predswCols, function(cn) sum(.dLLdf * .s[[cn]][.ik]), numeric(1))
      nnTorchZeroGrad(.net$id)
      nnTorchSetGrad(.net$id, -.dLLdw)
      nnTorchStep(.net$id)
    }
    sqrt(mean(.resid^2))
  }
}

## The nlm family: population-only optimizers (no between-subject variability).
## Used both as warmStart choices and, when passed as the est of a model with an
## nn() term, to trigger the no-BSV (QSP) population weight-fit path in .nnRun.
.nnNlmOptimizers <- c("nlm", "nlminb", "optim", "lbfgsb3c", "n1qn1",
                      "bobyqa", "newuoa", "uobyqa")

## The nn weight block's base par_ptr index in the nlm-family SOLVE model.  The
## nlm log-likelihood model declares params(THETA[1..nTheta], DV, allCovs) (see
## nlmixr2est's rxUiGet.nlmParams), so the weights sit after the thetas and the
## inserted DV -- a DIFFERENT base than the standard [thetas, covariates] layout
## nnUpdate() resolves for the base model / FOCEi.  Returns NA if the weights are
## not in allCovs (then the caller falls back to FOCEi materialization).
.nnNlmBase <- function(ui, aug) {
  .nTheta <- length(which(!ui$iniDf$fix))
  .order <- c(paste0("THETA[", seq_len(.nTheta), "]"), "DV", ui$allCovs)
  .b <- vapply(aug$nets, function(.n) match(.n$weights[1L], .order) - 1L, integer(1))
  if (anyNA(.b)) return(NULL)
  stats::setNames(.b, vapply(aug$nets, function(.n) as.character(.n$id), character(1)))
}

## Dispatch the population weight fit to any nlm-family optimizer -- nlm is simply
## an optimizer.  The gradient-based nlminb/nlm/optim(BFGS)/lbfgsb3c/n1qn1 use the
## analytic sensitivity gradient; the derivative-free minqa family bobyqa/newuoa/
## uobyqa use only the objective (rhobeg/rhoend set so an at-optimum start is safe).
## None uses a random effect.  Returns the fitted weight vector (w0 on failure).
.nnPopOptimize <- function(w0, objf, grf, est, iters) {
  ## the non-stats optimizers live in Suggests packages; if one is unavailable
  ## (e.g. the default lbfgsb3c is not installed) fall back to nlminb (always in
  ## stats) rather than skipping the warm start.
  .pkg <- c(lbfgsb3c = "lbfgsb3c", n1qn1 = "n1qn1",
            bobyqa = "minqa", newuoa = "minqa", uobyqa = "minqa")[est]
  if (!is.na(.pkg) && !requireNamespace(.pkg, quietly = TRUE)) est <- "nlminb"
  .rho <- list(rhobeg = 0.2, rhoend = 1e-4, maxfun = 50L * iters)
  .par <- tryCatch(switch(est,
    nlminb   = stats::nlminb(w0, objf, grf,
                             control = list(iter.max = iters, eval.max = 3L * iters))$par,
    nlm      = stats::nlm(function(w) { .r <- objf(w); attr(.r, "gradient") <- grf(w); .r },
                          w0, iterlim = iters)$estimate,
    optim    = stats::optim(w0, objf, grf, method = "BFGS",
                            control = list(maxit = iters))$par,
    lbfgsb3c = lbfgsb3c::lbfgsb3c(w0, objf, grf, control = list(maxit = iters))$par,
    n1qn1    = n1qn1::n1qn1(objf, grf, w0, max_iterations = iters)$par,
    bobyqa   = minqa::bobyqa(w0, objf, control = .rho)$par,
    newuoa   = minqa::newuoa(w0, objf, control = .rho)$par,
    uobyqa   = minqa::uobyqa(w0, objf, control = .rho)$par,
    stop("nlmixr2nn: unknown warmStart optimizer '", est, "'", call. = FALSE)),
    error = function(e) NULL)
  if (is.null(.par) || length(.par) != length(w0) || anyNA(.par)) w0 else unname(.par)
}

## Population (eta=0) weight pre-fit over the weight vector -- the "nlm bridge": a
## robust fixed-effects fit of the network weights used as the warm start for the
## mixed-model joint fit.  Weights sit in the optimized vector (as nlm would place
## them), the eta is fixed at its warm-start location 0 (so any population-only
## optimizer applies, no random effect), and the population error params come from
## the model ini.  `est` names the nlm-family optimizer.  The objective is the
## pooled -2 log-likelihood with the analytic d(-2LL)/dw from the augmented
## rx_predsw sensitivities.  Returns the fitted weight vector (aug$weights order).
## One exact-cotangent evaluation for the nlm population weight fit.  Bakes the
## current weights (w) into the nlm data columns, LOADS + solves the nlm objective
## once via nlmixr2est's population engine -- which fires the lik-contrib hook, so
## the EXACT per-obs error-model score d(LL)/d(f) is captured in C++ (any
## prediction-based error model, censoring included) rather than re-derived from
## the closed-form Gaussian add/prop formula.  Returns list(obj = -2LL, dLLdf
## aligned to the observation rows), or NULL to fall back to the Gaussian score.
## A fresh setup per call is required (and correct): the weights enter the solve
## as data covariates, so they are baked in at setup time.
.nnNlmExactCotangent <- function(ctx, aug, adata, w) {
  for (.net in aug$nets) {
    nnSetMeta(.net$id, ctx$nlmBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
    nnSetWeights(.net$id, w[.net$gIdx])
  }
  .dw <- .nnFillWeightCols(aug, adata)          # torch weights (= w) -> data columns
  .parIni <- tryCatch(suppressWarnings(suppressMessages(
    nlmixr2est::nlmObjectiveSetup(ctx$ui, .dw, ctx$control))), error = function(e) NULL)
  if (is.null(.parIni)) return(NULL)
  on.exit(try(nlmixr2est::.nlmFreeEnv(), silent = TRUE), add = TRUE)
  .nnCapReset(TRUE)
  ## `:::` deliberately: nlmSolveR is internal to nlmixr2est.  It was written as
  ## `::`, which raises "not an exported object" -- and because that error was
  ## swallowed by the tryCatch below, this whole exact-cotangent path silently
  ## fell back to the Gaussian score on every call and never once ran.
  .obj <- tryCatch(nlmixr2est:::nlmSolveR(.parIni), error = function(e) NA_real_)
  .cap <- .nnCapGet(); .nnCapReset(FALSE)
  if (!is.finite(.obj) || is.null(.cap) || !length(.cap$id)) return(NULL)
  .dLLdf <- .cap$dLLdf[match(ctx$obsKey, .cap$id * 1024L + .cap$k)]
  if (anyNA(.dLLdf)) return(NULL)
  list(obj = 2 * .obj, dLLdf = .dLLdf)          # objf = -2LL = 2 * minimum
}

.nnPopWarmStart <- function(aug, data, idCol, obs, dv, wPlaceholder, thetas, errPar,
                            w0, est, iters, exactCtx = NULL) {
  .ad <- data
  for (.e in names(aug$covMap)) .ad[[aug$covMap[[.e]]]] <- 0    # population: eta = 0
  if (errPar$add == 0 && errPar$prop == 0) errPar$add <- 1      # avoid R(f)=0
  .key <- paste(data[[idCol]][obs], data$time[obs])
  .dvObs <- dv[obs]
  ## objective (-2 log-likelihood) + analytic gradient at the GLOBAL weight vector w
  ## (all networks concatenated in aug$nets order)
  .eval <- function(w) {
    ## exact per-obs score from the C++ nlm solve when requested (else NULL)
    .ex <- if (!is.null(exactCtx)) .nnNlmExactCotangent(exactCtx, aug, .ad, w) else NULL
    for (.net in aug$nets) {                                    # split w per net
      nnSetMeta(.net$id, .net$augBase, .net$K, .net$H, .net$act)
      nnSetWeights(.net$id, w[.net$gIdx])
    }
    .s <- rxode2::rxSolve(aug$mAug, .ad, params = c(thetas, wPlaceholder),
                          returnType = "data.frame")
    .ik <- match(.key, paste(.s$id, .s$time))
    .f <- .s[[aug$endpoint]][.ik]
    .resid <- .dvObs - .f
    .R <- errPar$add^2 + (errPar$prop * .f)^2
    if (!is.null(.ex)) {                          # exact C++ cotangent + -2LL
      .dLLdf <- .ex$dLLdf
      .obj <- .ex$obj
    } else {                                       # closed-form Gaussian score
      .dRdf <- 2 * errPar$prop^2 * .f
      .dLLdf <- .resid / .R + 0.5 * (.resid^2 / .R^2 - 1 / .R) * .dRdf
      .obj <- sum(log(2 * pi * .R) + .resid^2 / .R)
    }
    list(obj = .obj,
         grad = -2 * vapply(aug$predswCols, function(cn) sum(.dLLdf * .s[[cn]][.ik]),
                            numeric(1), USE.NAMES = FALSE))
  }
  ## cache the last evaluation so paired objective/gradient calls solve once
  .cache <- new.env(parent = emptyenv())
  .get <- function(w) {
    if (is.null(.cache$w) || !isTRUE(all.equal(w, .cache$w))) {
      .cache$w <- w; .cache$v <- .eval(w)
    }
    .cache$v
  }
  .nnPopOptimize(w0, function(w) .get(w)$obj, function(w) .get(w)$grad, est, iters)
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

## Read the serialized per-network shape metadata (id/weights/K/H/act) carried on
## a ui, or NULL.  This is the transient registry (.nnEnv$reg) snapshotted onto
## the ui at fit time so it survives saveRDS()/reload.
.nnUiMeta <- function(ui) {
  .u <- tryCatch(rxode2::rxUiDecompress(ui), error = function(e) ui)
  if (is.environment(.u) && exists("nnMeta", envir = .u, inherits = FALSE)) {
    return(get("nnMeta", envir = .u, inherits = FALSE))
  }
  NULL
}

## rxode2 ui-prep hook (registered in .onLoad).  When a saved fit/model carrying
## nn() is reloaded in a fresh session, the transient shape registry (.nnEnv$reg)
## is empty, so nnForward() cannot stride the weight block.  Rebuild each of THIS
## ui's networks' shapes in the C registry -- resolving the block base BY NAME
## from the current solve-parameter layout (robust to a rebuilt/reordered model).
## The weight VALUES already arrive via rxForcedPars(); only the shapes are
## transient.  A no-op during training (the training loop owns the registry and
## switches bases between the base and augmented models) and for non-nn models.
## Flag `ui` as owning the nn par-loader, for a model whose weights come from the
## LOADER BUFFER.  Two things it must not do:
##  - claim an unrelated model: the loader would write into its par_ptr, which is
##    exactly what naming the loader prevents;
##  - claim a persisted fit: rxCallParLoaders writes forcedPars first and then
##    lets loaders override, so a ui carrying its trained weights in
##    rxForcedPars() would have them overwritten by whatever the (transient, and
##    after a reload empty) loader buffer holds.
## A no-op on an older rxode2 without rxParLoader().
.nnClaimParLoader <- function(ui) {
  if (!("rxParLoader<-" %in% getNamespaceExports("rxode2"))) return(invisible(FALSE))
  .u <- tryCatch(rxode2::rxUiDecompress(ui), error = function(e) NULL)
  if (!is.environment(.u)) return(invisible(FALSE))   # compressed: cannot set in place
  if (identical(tryCatch(rxode2::rxParLoader(.u), error = function(e) NULL),
                .nnLoaderName)) {
    return(invisible(TRUE))                           # already claimed
  }
  .p <- tryCatch(rxode2::rxModelVars(.u)$params, error = function(e) NULL)
  .w <- grep("^rxnn(W1|B1|W2|B2)_", .p, value = TRUE)
  if (length(.w) == 0L) return(invisible(FALSE))      # not an nn model
  .fp <- tryCatch(rxode2::rxForcedPars(.u), error = function(e) NULL)
  ## Weights riding on the ui normally WIN: a parsed or fitted model describes
  ## its own network, and letting a transient buffer override it would silently
  ## zero the network after a reload (the buffer is empty in a fresh session).
  ## During training that is inverted -- the loop owns the weights and injects
  ## them through the loader every round, so it must outrank the (by then stale)
  ## values the ui was parsed with.
  if (!isTRUE(.nnEnv$training) && any(.w %in% names(.fp))) return(invisible(FALSE))
  tryCatch({
    rxode2::rxParLoader(.u) <- .nnLoaderName
    invisible(TRUE)
  }, error = function(e) invisible(FALSE))
}

.nnRehydrate <- function(ui, solveModel = NULL) {
  ## Claim the par-loader for any model that carries nn weight columns, BEFORE
  ## the training early-return -- during training the loader buffer is exactly
  ## where the weights live, so a training-time ui solve needs the flag most.
  ## This has to happen HERE rather than in nnUpdate(): rxSolve.rxUi calls
  ## .rxApplyParLoader() right after these hooks, and that CLEARS the active
  ## loader for a model with no flag -- including a name nnWithLoader() just set.
  ## Without the flag the weights never reach par_ptr and every nn output
  ## collapses to its all-zero-weight value.  nnUpdate() cannot do it because a
  ## ui often arrives compressed, so its in-place set would not reach the caller;
  ## the ui is decompressed by the time a prep hook sees it.
  .nnClaimParLoader(ui)
  if (isTRUE(.nnEnv$training)) return(invisible())
  .meta <- .nnUiMeta(ui)
  if (is.null(.meta) || length(.meta) == 0L) return(invisible())
  ## restore the R-side registry (used by nnCovData / .nnAugmentFromUi / nnUpdate
  ## / predict) for any of THIS ui's nets missing from the transient registry.
  .have <- if (length(.nnEnv$reg)) {
    vapply(.nnEnv$reg, function(m) m$id, integer(1))
  } else integer(0)
  for (.m in .meta) {
    if (!(.m$id %in% .have)) .nnEnv$reg[[as.character(.m$id)]] <- .m
  }
  ## Rebuild the C shape registry for THIS model, resolving the weight-block base
  ## BY NAME against the model actually being solved.  `solveModel` is what the
  ## gpars layout uses, so it is the authority; the ui is only a fallback for an
  ## older rxode2 whose prep hooks pass one argument.
  .params <- tryCatch(
    if (!is.null(solveModel)) rxode2::rxModelVars(solveModel)$params
    else .nnSolveParams(ui),
    error = function(e) NULL)
  if (is.null(.params)) return(invisible())
  ## CLEAR-THEN-SET.  The C registry is a flat array keyed by a MODEL-LOCAL id,
  ## so every single-network model is network 0.  Binding only this model's
  ## networks -- after clearing -- is what stops two live models from reading
  ## each other's weights; it is also why nnClearMeta() is no longer something a
  ## user (or a test) has to call.
  nnClearMeta()
  for (.m in .meta) {
    .idx <- match(.m$weights, .params)
    if (anyNA(.idx) || any(diff(.idx) != 1L)) next   # not this ui's layout -> skip
    nnSetMeta(.m$id, .idx[1L] - 1L, .m$K, .m$H, .m$act)
  }
  invisible()
}

## The GLOBAL weight vector = every network's torch weights concatenated in
## aug$nets order (matching the rx_sw/rx_predsw global index).
.nnAllTorchWeights <- function(aug) {
  unlist(lapply(aug$nets, function(.n) nnTorchWeights(.n$id)), use.names = FALSE)
}
## Push a GLOBAL weight vector back into each network's torch module (per gIdx).
.nnSetAllTorchWeights <- function(aug, w) {
  for (.net in aug$nets) nnTorchSetWeights(.net$id, w[.net$gIdx])
  invisible()
}
## Inject the current torch weights into `data` covariate columns (all networks),
## for estimators whose kernel reads covariates from the data rather than the
## par-loader (e.g. SAEM).  Returns the updated data.
.nnFillWeightCols <- function(aug, data) {
  for (.net in aug$nets) {
    .w <- nnTorchWeights(.net$id)
    for (.j in seq_along(.net$weights)) data[[.net$weights[.j]]] <- .w[.j]
  }
  data
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

  ## activate the named nn par-loader for every solve done during training (the
  ## inner fits + augmented solves bypass the rxSolve.rxUi flag bridge); cleared on
  ## exit so it never leaks into an unrelated model's solve.
  .nnLoaderOn()
  on.exit(.nnLoaderOff(), add = TRUE)
  ## refit of a reloaded fit (fresh session): the model text already carries
  ## nn<K>() (not nn()), so the UDF will not repopulate the registry -- restore it
  ## from the ui's persisted shapes so augmentation + warm-start-from-stored work.
  if (length(.nnEnv$reg) == 0L) {
    .meta <- .nnUiMeta(.ui)
    if (!is.null(.meta)) for (.m in .meta) .nnEnv$reg[[as.character(.m$id)]] <- .m
  }
  ## the training loop owns the C shape registry (switching bases between the base
  ## and augmented models); suppress the reload rehydrate hook while it runs.
  .nnEnv$training <- TRUE
  on.exit(assign("training", FALSE, envir = .nnEnv), add = TRUE)

  .aug <- .nnAugmentFromUi(.ui)
  .data <- nnCovData(.data)
  ## each network's weight block sits at DIFFERENT par_ptr positions in the base vs
  ## the augmented model, so the injection base is switched per context (base-model
  ## base per network for the inner fit, augmented base for the sensitivity solve).
  .baseInfo <- nnUpdate(.ui)
  .baseBases <- stats::setNames(.baseInfo$base, as.character(.baseInfo$id))
  ## init one torch module per network (seed offset so distinct nets differ)
  for (.i in seq_along(.aug$nets)) {
    .net <- .aug$nets[[.i]]
    .seed <- if (is.null(sched$seed)) NULL else as.integer(sched$seed + .i - 1L)
    nnTorchInit(.net$id, .net$K, .net$H, act = .net$act, seed = .seed)
    nnTorchOptInit(.net$id, sched$optimizer, sched$lr)
  }
  ## Warm start from the weights the model already carries -- the values nn()
  ## drew at parse time, or the trained values of a fit being refitted.
  .existing <- .nnExistingWeights(.ui, .aug)
  if (!is.null(.existing)) .nnSetAllTorchWeights(.aug, .existing)
  ## ...and then take them OFF the working ui.  From here the loop owns the
  ## weights and injects the current values every round; leaving the parse-time
  ## values in rxForcedPars would let them override that injection in any inner
  ## solve that applies forced parameters, silently pinning the network at its
  ## starting point.  `.ui` is a private decompressed copy, so this does not
  ## touch the caller's model; the trained values are written back at the end.
  if (!is.null(.existing)) {
    .fpKeep <- tryCatch(rxode2::rxForcedPars(.ui), error = function(e) NULL)
    if (!is.null(.fpKeep)) {
      .fpKeep <- .fpKeep[setdiff(names(.fpKeep), .aug$weights)]
      rxode2::rxForcedPars(.ui) <- if (length(.fpKeep)) .fpKeep else NULL
    }
  }
  ## INPUT SCALING (R/nnScale.R).  A network fed raw model quantities -- amounts
  ## in the hundreds, concentrations in the hundredths -- starts saturated, and a
  ## saturated network has no input derivative: no FOCEi sensitivity for a latent
  ## eta, and no weight-training signal.  Rescale the first-layer weights by each
  ## input's typical magnitude, measured from one solve over the real data.
  ##
  ## Only for a model that has never been trained.  Trained weights already
  ## embody whatever scaling the previous fit found, so rescaling them would
  ## silently corrupt a refit's warm start.
  .untrained <- !isTRUE(tryCatch(get("nnTrained", envir = .ui, inherits = FALSE),
                                 error = function(e) FALSE))
  if (.untrained) {
    .inputsById <- tryCatch({
      .cl <- .nnParseCallAll(.ui$lstChr)
      stats::setNames(lapply(.cl, function(.c) .c$inputs),
                      vapply(.cl, function(.c) as.character(.c$id), character(1)))
    }, error = function(e) NULL)
    if (!is.null(.inputsById)) {
      ## the trial solve must see the current weights, so hand it a copy of the
      ## data carrying them (the same route the inner fits use)
      .scales <- tryCatch(
        .nnInputScales(.ui, .nnFillWeightCols(.aug, .data), .aug$nets, .inputsById),
        error = function(e) NULL)
      if (!is.null(.scales)) {
        for (.i in seq_along(.aug$nets)) {
          .net <- .aug$nets[[.i]]
          .sc <- .scales[[.i]]
          if (any(is.finite(.sc) & .sc != 1)) {
            nnTorchSetWeights(.net$id,
                              .nnRescaleW1(nnTorchWeights(.net$id), .net$K, .net$H, .sc))
            .nnEnv$scales[[as.character(.net$id)]] <- .sc
          }
        }
      }
    }
    ## The trial solve went through rxSolve.rxUi, which clears the active
    ## par-loader on exit.  Leaving it cleared silently disarms weight injection
    ## for every remaining solve in the fit -- the objective then never moves.
    ## Re-arm unconditionally: the solve clears it whether or not it succeeded.
    .nnLoaderOn()
  }
  on.exit(for (.net in .aug$nets) tryCatch(nnTorchFree(.net$id), silent = TRUE), add = TRUE)

  .idCol <- if ("ID" %in% names(.data)) "ID" else "id"
  .obs <- .data[[if ("EVID" %in% names(.data)) "EVID" else "evid"]]
  .obs <- is.na(.obs) | .obs == 0
  .dv <- .data[[if ("DV" %in% names(.data)) "DV" else "dv"]]
  .wPlaceholder <- stats::setNames(rep(0, .aug$nW), .aug$weights)
  .weightStep <- .nnWeightStepper(.aug, .data, .idCol, .obs, .dv, .wPlaceholder)
  ## eta-free (population UDE) models have no latent input to the network; the
  ## weight step then solves at the pooled population (no per-subject EBEs).
  .hasEta <- length(.aug$covMap) > 0L
  .latent <- if (.hasEta) names(.aug$covMap)[1L] else NA_character_

  ## exact-cotangent path: map each observation row to the inner fit's (internal id,
  ## obs index k) so the captured per-obs cotangent aligns with the augmented solve.
  .exact <- identical(sched$cotangent, "exact")
  ## methods whose inner fit's FINAL likelihood eval is a clean pass at the fitted
  ## etas (FOCEi/Laplace family + ADVI/VAE, which are maxOuter=0 FOCEi evals) fire
  ## the contribution hook cleanly during the fit -> self-capture.  Other methods
  ## (imp/impmap/qrpem: hook fires at importance draws; saem: kernel bypasses the
  ## FOCEi inner) get the cotangent from a dedicated FOCEi posthoc at the fit.
  .exactSelf <- .exact && (.innerEst %in% c("focei", "foce", "foi", "fo",
                                            "laplace", "agq", "emvi", "fbvi", "vae"))
  .exactPosthoc <- .exact && !.exactSelf
  ## a non-add()/prop() error model has no closed-form Gaussian cotangent -- its
  ## score must come from the inner fit.
  if (is.na(.aug$errAdd) && is.na(.aug$errProp) && !.exact) {
    stop("nlmixr2nn: this error model has no add()/prop() term; ",
         "fit with nnControl(cotangent = \"exact\")", call. = FALSE)
  }
  .obsIdVals <- .data[[.idCol]][.obs]
  .obsKey <- (match(.obsIdVals, unique(.obsIdVals)) - 1L) * 1024L +
    (stats::ave(seq_along(.obsIdVals), .obsIdVals, FUN = function(z) seq_along(z)) - 1L)

  ## true interleave only when the inner estimator exposes a partial outer step
  .knob <- .nnInterleaveKnob(.innerCtl)
  .interleave <- sched$mode == "joint" && !is.null(.knob)
  if (sched$mode == "joint" && !.interleave) {
    message(sprintf("est=\"%s\" has no partial outer step; nn joint uses the iterative loop",
                    .innerEst))
  }

  ## model initial population thetas + error params (shared warm-start inputs)
  .iniDf <- .ui$iniDf
  .th0 <- stats::setNames(.iniDf$est[!is.na(.iniDf$ntheta)], .iniDf$name[!is.na(.iniDf$ntheta)])
  .errPar0 <- list(add = if (is.na(.aug$errAdd)) 0 else unname(.th0[.aug$errAdd]),
                   prop = if (is.na(.aug$errProp)) 0 else unname(.th0[.aug$errProp]))

  ## QSP / no between-subject variability: an nlm-family estimator is a pure
  ## population optimizer -- it rejects random-effects models, and it cannot
  ## OPTIMIZE the network weights (they are covariates, not in its parameter
  ## vector).  So the weights are fit directly as a population problem with that
  ## optimizer over the augmented sensitivity solve (which reads the weights), then
  ## the fit is materialized at the fixed trained weights with the SAME nlm-family
  ## estimator -- whose solve now reads the weights natively (nlmixr2est's nlm model
  ## declares them as covariates), at the nlm-model weight base.  If the nlm base is
  ## not resolvable, or the model still carries a random effect (which the nlm family
  ## rejects), materialization falls back to FOCEi.  This is "run an nlm-family
  ## optimizer without between-subject variability".
  if (.innerEst %in% .nnNlmOptimizers) {
    .nlmBases <- if (.hasEta) NULL else .nnNlmBase(.ui, .aug)
    ## The exact per-observation score for this branch would come from
    ## nlmixr2est's nlm C++ solve via .nnNlmExactCotangent().  It is DISABLED.
    ##
    ## That path never actually ran: it called an unexported `nlmSolveR` through
    ## `::`, and the resulting error was swallowed by a tryCatch that returned
    ## NA, so every call fell back to the Gaussian score.  Fixing the call
    ## revealed why that went unnoticed -- setting the objective up and tearing it
    ## down once per optimizer evaluation double-frees the shared capture store
    ## (src/nlmixr2nnContrib.c grows one buffer with realloc), which segfaults.
    ##
    ## Turning it on is gated on making that store safe.  Until then this branch
    ## uses the closed-form Gaussian score, and .nnInferSched() refuses an
    ## endpoint that has no closed form rather than fitting one wrongly.
    .exactCtx <- NULL
    .wFit <- .nnPopWarmStart(.aug, .data, .idCol, .obs, .dv, .wPlaceholder, .th0, .errPar0,
                             .nnAllTorchWeights(.aug), .innerEst, sched$rounds,
                             exactCtx = .exactCtx)
    .nnSetAllTorchWeights(.aug, .wFit)
    .trained <- stats::setNames(.wFit, .aug$weights)
    .dw <- .nnFillWeightCols(.aug, .data)
    if (!is.null(.nlmBases)) {
      ## native nlm materialization: the nlm solve reads each network's weights at
      ## its nlm-model base; nlm estimates the residual error at the fixed weights.
      for (.net in .aug$nets) {
        nnSetMeta(.net$id, .nlmBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
        nnSetWeights(.net$id, .wFit[.net$gIdx])
      }
      .matCtl <- .innerCtl
      if (!is.null(.matCtl$calcTables)) .matCtl$calcTables <- FALSE
      .fit <- suppressWarnings(suppressMessages(
        nlmixr2est::nlmixr2(.ui, .dw, est = .innerEst, control = .matCtl)))
      .matEst <- .innerEst
      message(sprintf("nn: population (no-BSV) fit via %s (native weight-reading solve)",
                      .innerEst))
    } else {
      ## fallback: nlm base unresolved or a random effect is present -> FOCEi reads
      ## the weights and estimates the residual error / any Omega.
      if (.hasEta) {
        message(sprintf(paste0("est=\"%s\" is population-only; the nn() random effect ",
                               "is materialized with focei"), .innerEst))
      }
      for (.net in .aug$nets) {
        nnSetMeta(.net$id, .baseBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
        nnSetWeights(.net$id, .wFit[.net$gIdx])
      }
      .matCtl <- nlmixr2est::foceiControl(print = 0L, calcTables = FALSE,
                                          maxInnerIterations = if (.hasEta) 30L else 1L,
                                          maxOuterIterations = 30L)
      .fit <- suppressWarnings(suppressMessages(
        nlmixr2est::nlmixr2(.ui, .dw, est = "focei", control = .matCtl)))
      .matEst <- "focei"
      message(sprintf("nn: population (no-BSV) weight fit via %s, materialized with focei",
                      .innerEst))
    }
    .fit <- .nnAddTablesCov(.fit, .wFit, .baseBases, .aug, .matEst, .origTablesCov)
    .fitEnv <- .fit$env
    .storedUi <- rxode2::rxUiDecompress(get("ui", envir = .fitEnv))
    rxode2::rxForcedPars(.storedUi) <- .trained
    .nnMarkTrained(.storedUi, .aug)
    assign("ui", .storedUi, envir = .fitEnv)
    assign("nnParHist", data.frame(round = 1L, objf = .fit$objf,
             errAdd = if (is.na(.aug$errAdd)) NA_real_ else .fit$theta[[.aug$errAdd]],
             errProp = if (is.na(.aug$errProp)) NA_real_ else .fit$theta[[.aug$errProp]],
             rmse = NA_real_, wChange = NA_real_, objfChange = NA_real_), envir = .fitEnv)
    assign("nnWeights", .trained, envir = .fitEnv)
    assign("nnConverged", TRUE, envir = .fitEnv)
    assign("nnRounds", 1L, envir = .fitEnv)
    return(.fit)
  }

  ## the nlm bridge: a gradient-based population (eta=0) weight pre-fit seeding the
  ## joint fit with a robust weight vector (unless the model already carries
  ## trained weights, in which case those are the warm start).
  if (!identical(sched$warmStart, "none") && is.null(.existing)) {
    .wPop <- .nnPopWarmStart(.aug, .data, .idCol, .obs, .dv, .wPlaceholder,
                             .th0, .errPar0, .nnAllTorchWeights(.aug),
                             sched$warmStart, sched$warmPopIters)
    .nnSetAllTorchWeights(.aug, .wPop)
    message(sprintf("nn: population (nlm-bridge, %s) warm start applied", sched$warmStart))
  }

  ## optional naive-pooled (eta=0) torch warm-up from the model initial parameters
  if (sched$warmSteps > 0L) {
    .errPar0s <- .errPar0
    if (.errPar0s$add == 0 && .errPar0s$prop == 0) .errPar0s$add <- 1
    .ids <- as.character(unique(.data[[.idCol]]))
    .ebes0 <- stats::setNames(rep(0, length(.ids)), .ids)
    for (.ws in seq_len(sched$warmSteps)) .weightStep(.ebes0, .errPar0s, .th0)
  }

  .parHist <- vector("list", sched$rounds)
  .fit <- NULL
  .curUi <- .ui
  .wPrev <- .nnAllTorchWeights(.aug)
  .objfPrev <- NA_real_
  .converged <- FALSE
  .nRun <- 0L
  ## A default fit is a multi-round loop whose inner fits are silenced, so
  ## without this it looks hung for minutes.  Honour the inner control's own
  ## print=0 convention rather than inventing a second quiet switch.
  .quiet <- isTRUE(tryCatch(.innerCtl$print == 0L, error = function(e) FALSE))
  .prog <- .nnProgressStart(sched$rounds, .quiet)
  on.exit(.nnProgressStop(.prog), add = TRUE)
  for (.round in seq_len(sched$rounds)) {
    .nRun <- .round
    ## inject each network's current weights BOTH ways (par-loader for FOCEi's solve
    ## + data covariate columns for SAEM's kernel, which bypasses the loader).
    for (.net in .aug$nets) {
      nnSetMeta(.net$id, .baseBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
      nnSetWeights(.net$id, nnTorchWeights(.net$id))
    }
    .dw <- .nnFillWeightCols(.aug, .data)
    ## interleave: warm-started PARTIAL step from the previous ui; else full fit
    .fitUi <- if (.interleave) .curUi else .ui
    .roundCtl <- .innerCtl
    if (.interleave) .roundCtl[[.knob]] <- sched$outerPerRound
    if (.exactSelf) .nnCapReset(TRUE)                        # self-capture during fit
    .fit <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(.fitUi, .dw, est = .innerEst, control = .roundCtl)))
    if (.interleave) .curUi <- .fit$ui                       # warm-start next round
    ## exact per-obs cotangent (any residual model): from the inner fit when it
    ## self-captures, else from a dedicated FOCEi posthoc at the fit's estimates
    ## (maxOuter=0 keeps the outer parameters, re-optimizes the EBEs, and fires the
    ## contribution hook cleanly).  The augmented solve then uses THOSE EBEs.
    .dLLdfObs <- NULL; .ebeFit <- .fit
    if (.exact) {
      if (.exactPosthoc) {
        .nnCapReset(TRUE)
        .ebeFit <- suppressWarnings(suppressMessages(
          nlmixr2est::nlmixr2(.fit$finalUi, .dw, est = "focei",
            nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 0L,
                                     maxInnerIterations = 30L, calcTables = FALSE))))
      }
      .cap <- .nnCapGet(); .nnCapReset(FALSE)
      if (!is.null(.cap) && length(.cap$id)) {
        .cand <- .cap$dLLdf[match(.obsKey, .cap$id * 1024L + .cap$k)]
        if (!anyNA(.cand)) .dLLdfObs <- .cand
      }
      if (is.null(.dLLdfObs) && is.na(.aug$errAdd) && is.na(.aug$errProp)) {
        stop(sprintf(paste0("nlmixr2nn: cotangent=\"exact\" captured no per-observation ",
             "cotangents (est=\"%s\"); a non-add()/prop() error model needs the ",
             "FOCEi contribution hook"), .innerEst), call. = FALSE)
      }
    }
    .ebes <- if (.hasEta) {
      stats::setNames(.ebeFit$eta[[.latent]], as.character(.ebeFit$eta[["ID"]]))
    } else stats::setNames(numeric(0), character(0))         # population: no EBEs
    .thetas <- .fit$theta
    .errPar <- list(add = if (is.na(.aug$errAdd)) 0 else .fit$theta[[.aug$errAdd]],
                    prop = if (is.na(.aug$errProp)) 0 else .fit$theta[[.aug$errProp]])
    for (.ws in seq_len(sched$wSteps)) .rmse <- .weightStep(.ebes, .errPar, .thetas, .dLLdfObs)
    .wNow <- .nnAllTorchWeights(.aug)
    .wChange <- sqrt(sum((.wNow - .wPrev)^2)) / (sqrt(sum(.wPrev^2)) + 1e-8)
    .wPrev <- .wNow
    .objfChange <- if (is.na(.objfPrev)) Inf else abs(.fit$objf - .objfPrev) / (abs(.objfPrev) + 1e-8)
    .objfPrev <- .fit$objf
    .parHist[[.round]] <- data.frame(round = .round, objf = .fit$objf,
                                     errAdd = .errPar$add, errProp = .errPar$prop,
                                     rmse = .rmse, wChange = .wChange, objfChange = .objfChange)
    ## stop when the weights (and, when interleaving, the objective) stabilise;
    ## isTRUE guards a NaN change (e.g. an unstable solve) -> keep going, don't crash
    .nnProgressTick(.prog)
    .stop <- isTRUE(.wChange < sched$tol && (!.interleave || .objfChange < sched$tol))
    if (sched$tol > 0 && .round > 1L && .stop) { .converged <- TRUE; break }
  }
  .parHist <- .parHist[seq_len(.nRun)]
  .nnProgressStop(.prog)
  .nnRunSummary(.interleave, .converged, .nRun, sched$rounds, .wChange, .objfChange,
                .fit$objf, .parHist, .quiet)

  ## the last inner fit IS the deliverable (no re-fit); add tables + covariance
  ## post-hoc from the user's original control settings.
  .allW <- .nnAllTorchWeights(.aug)
  .trained <- stats::setNames(.allW, .aug$weights)
  .fit <- .nnAddTablesCov(.fit, .allW, .baseBases, .aug, .innerEst, .origTablesCov)
  .fitEnv <- .fit$env
  .storedUi <- rxode2::rxUiDecompress(get("ui", envir = .fitEnv))
  rxode2::rxForcedPars(.storedUi) <- .trained
  ## snapshot the transient shape registry onto the ui so a reloaded fit can
  ## rebuild it (.nnRehydrate) and stride the weights carried in rxForcedPars().
  .nnMeta <- lapply(.nnEnv$reg, function(m) {
    list(id = m$id, weights = m$weights, K = m$K, H = m$H, act = m$act)
  })
  assign("nnMeta", .nnMeta, envir = .storedUi)
  .sticky <- if (exists("sticky", envir = .storedUi, inherits = FALSE)) {
    get("sticky", envir = .storedUi, inherits = FALSE)
  } else character(0)
  assign("sticky", unique(c(.sticky, "nnMeta")), envir = .storedUi)
  .nnMarkTrained(.storedUi, .aug)
  assign("ui", .storedUi, envir = .fitEnv)
  ## NN metadata in the fit env (NOT $<-, which would add a data column)
  assign("nnParHist", do.call(rbind, .parHist), envir = .fitEnv)
  assign("nnWeights", .trained, envir = .fitEnv)
  assign("nnConverged", .converged, envir = .fitEnv)
  assign("nnRounds", .nRun, envir = .fitEnv)
  .fit
}
