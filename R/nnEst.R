## The NN training loop.
##
## `.nnRun` is the entry point the estimation interceptor calls; it owns only the
## global state (the par-loader flag, the shape registry's training mode, the
## torch modules) and the order of the phases.  Everything else is a phase
## function taking the `ctx` list that `.nnRunCtx()` builds:
##
##   .nnRunCtx        the ui, the augmented model, the data columns the gradient
##                    reads, the per-estimator weight-block bases, the capture
##                    keys, and the resolved schedule -- everything derived once
##   .nnRunPopFit     the no-BSV (QSP) branch: an nlm-family est fits the weights
##                    as a pure population problem and returns
##   .nnRunWarmStarts the nlm-bridge population pre-fit and the naive-pooled
##                    torch warm-up, both pure side effects on the modules
##   .nnRunLoop       the round loop, returning the last inner fit and its history
##   .nnStoreNnFit    (R/nnFinalize.R) attaches weights + history to the fit
##
## These used to be one 400-line function whose 25 locals were all in scope
## everywhere, so the only way to know what the loop actually depended on was to
## read all of it.  The ctx list is that dependency set, written down.

## `env` carries ui/data + the standard inner estimator (class(env)[1]) and its
## control (env$control); `sched` is an nnControl().  The last inner fit is the
## returned deliverable, with tables/covariance added post-hoc.
.nnRun <- function(env, sched) {
  .nnTorchRequire("fitting a model that contains nn()")
  ## activate the named nn par-loader for every solve done during training (the
  ## inner fits + augmented solves bypass the rxSolve.rxUi flag bridge), and put
  ## the shape registry in training mode: the loop owns it, switching bases
  ## between the base and augmented models, so the reload rehydrate hook must not
  ## fire underneath it.
  ##
  ## Both are armed HERE, before any setup that can throw, and cleared on exit --
  ## a failed setup must not leave the loader live for an unrelated model's solve.
  .nnLoaderOn()
  .nnEnv$training <- TRUE
  on.exit({
    .nnLoaderOff()
    assign("training", FALSE, envir = .nnEnv)
  }, add = TRUE)

  ## Clear the compiled shape registry on the way out, whatever happened.
  ##
  ## The registry is a flat array keyed by a MODEL-LOCAL id, so every
  ## single-network model is network 0 -- the entry this fit wrote is exactly the
  ## entry the NEXT fit's network 0 would read.  On the success path the ui-prep
  ## hook rebinds per solve and nothing shows.  On a FAILED fit nothing rebinds,
  ## and the stale shape outlives the model it described: a fit whose network
  ## took two inputs, followed by one whose network takes one, made nnForward
  ## stride a two-input block through a one-input parameter vector.  That is not
  ## a wrong number, it is a segfault -- observed, from an errored fit two cells
  ## earlier in a smoke sweep, and not reproducible on its own.
  on.exit(try(nnClearMeta(), silent = TRUE), add = TRUE)
  ctx <- .nnRunCtx(env, sched)
  ## registered where the single-frame version registered it: after setup, so a
  ## throw inside .nnRunCtx leaks the modules exactly as it did before.  Freeing
  ## them on a failed setup too is a behaviour change, not a move.
  on.exit(for (.net in ctx$aug$nets) try(nnTorchFree(.net$id), silent = TRUE),
          add = TRUE)

  if (ctx$innerEst %in% .nnNlmOptimizers) return(.nnRunPopFit(ctx))
  .nnRunWarmStarts(ctx)
  .res <- .nnRunLoop(ctx)
  ## the last inner fit IS the deliverable (no re-fit); add tables + covariance
  ## post-hoc from the user's original control settings.
  .fit <- .nnAddTablesCov(.res$fit, .res$weights, ctx$baseBases, ctx$aug,
                          ctx$innerEst, ctx$origTablesCov)
  .nnStoreNnFit(ctx, .fit, .res$weights, .res$parHist, .res$converged, .res$nRun)
}

## Everything the phases below need, derived once.  Also the record of what the
## loop depends on: a field here is a real dependency, and nothing outside this
## function may add one.
.nnRunCtx <- function(env, sched) {
  .ui <- rxode2::rxUiDecompress(env$ui)
  .data <- env$data
  .innerEst <- class(env)[1L]
  ## env$control is already a validated control for .innerEst (nlmixr2() fills it
  ## before dispatch), so it is used directly as the inner NLME control.
  .innerCtl <- env$control
  ## Checked FIRST, before anything is compiled or allocated: unusable data is
  ## the cheapest failure to report and the most expensive one to discover in
  ## round three.  Resolved case-insensitively (R/nnBind.R) -- ID/id, TIME/time,
  ## DV/dv, EVID/evid are all legal nlmixr2 spellings, and reading the wrong one
  ## is silent, not an error.  nnCovData() below only ADDS columns, so names
  ## resolved here stay valid.
  .idCol <- .nnDataCol(.data, "ID")
  .timeCol <- .nnDataCol(.data, "TIME")
  .dvCol <- .nnDataCol(.data, "DV")
  .evidCol <- .nnDataCol(.data, "EVID")
  if (anyNA(c(.idCol, .timeCol, .dvCol))) {
    stop("nlmixr2nn: the fitting data needs ID, TIME and DV columns (any case); ",
         "missing ",
         paste(c("ID", "TIME", "DV")[is.na(c(.idCol, .timeCol, .dvCol))],
               collapse = ", "), call. = FALSE)
  }
  .dv <- .data[[.dvCol]]
  ## An OBSERVATION is EVID 0 *with a DV*.  Rows carrying no DV -- a dropped
  ## sample, a BLQ record, a placeholder time -- are not observations to anyone:
  ## nlmixr2est never evaluates them, so its likelihood hook never numbers them,
  ## and they contribute nothing to a likelihood.
  ##
  ## Counting them here did two things, both silent.  The captured cotangent is
  ## keyed by (subject, observation index), so one missing DV shifted every
  ## later index and the exact score could not be matched at all.  Worse, the
  ## closed-form score of a missing DV is NA, and one NA in a sum makes the whole
  ## weight gradient NA -- torch then wrote NaN into every weight and the fit
  ## returned, objective and all, saying nothing.  Measured: a single missing DV
  ## in a 30-row data set took a converging fit (objf 598) to NaN weights.
  .obs <- if (is.na(.evidCol)) rep(TRUE, nrow(.data)) else .data[[.evidCol]]
  .obs <- (is.na(.obs) | .obs == 0) & !is.na(.dv)
  if (!any(.obs)) {
    stop("nlmixr2nn: the fitting data has no observation rows (EVID 0 with a ",
         "non-missing DV)", call. = FALSE)
  }
  .origTablesCov <- .nnStashTablesCov(.innerCtl)
  .innerCtl <- .nnDisableTablesCov(.innerCtl)
  ## refit of a reloaded fit (fresh session): the model text already carries
  ## nn<K>() (not nn()), so the UDF will not repopulate the registry -- restore it
  ## from the ui's persisted shapes so augmentation + warm-start-from-stored work.
  if (length(.nnEnv$reg) == 0L) {
    .meta <- .nnUiMeta(.ui)
    if (!is.null(.meta)) for (.m in .meta) .nnEnv$reg[[as.character(.m$id)]] <- .m
  }

  .aug <- .nnAugmentFromUi(.ui)
  .data <- nnCovData(.data)
  ## Each network's weight block sits at a DIFFERENT par_ptr offset in every
  ## model involved -- the base ui, the model this estimator solves, and the
  ## augmented sensitivity model -- so the registered offset is switched per
  ## context.  The estimator's own model is the authority for the inner fit:
  ## reading at another model's offset does not error, it evaluates a different
  ## network (see .nnEstSolveParams).
  .baseInfo <- nnUpdate(.ui, params = .nnEstSolveParams(.ui, .innerEst))
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
  .adata <- .nnFillWeightCols(.aug, .data)
  ## Every eta under BOTH spellings.  The augmented model renames `eta.nn` to
  ## `eta_nn` (R/nnAugmentUi.R) and the net metadata carries the sanitized form,
  ## while `ui$eta` carries the dotted one; matching against only one of them
  ## silently fails to recognize an eta, which then falls through to the data
  ## lookup, misses, and yields a scale of 1 -- the right answer by accident, so
  ## the bug would not show until a data column happened to be called `eta_nn`.
  .etaAll <- unique(c(tryCatch(.ui$eta, error = function(e) character(0)),
                      unname(.aug$covMap)))

  ## ONE trial solve of the model at its initial estimates, serving two callers:
  ## input scaling below, and the curvature penalty's grid (R/nnPenalty.R).  It
  ## also supplies the per-input magnitudes the L2 term needs, which is why it is
  ## taken on a refit too even though the scaling is not re-applied there --
  ## without it a state input's L2 multiplier would silently fall back to 1 and
  ## that weight column would go all but unpenalized on the second fit.
  ##
  ## Binding the registry and pushing the current weights first is not optional:
  ## the loader injects from its own buffer and `.nnEnv$training` is TRUE inside
  ## the loop, so without this the trial solve runs against whatever the previous
  ## fit left behind -- zeros in a fresh session -- and the derived scale, and
  ## therefore the whole fit, stops being reproducible under a fixed set.seed().
  .trial <- NULL
  if (.untrained || isTRUE(sched$l2 > 0) || isTRUE(sched$smooth > 0)) {
    for (.net in .aug$nets) {
      nnSetMeta(.net$id, .baseBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
      nnSetWeights(.net$id, nnTorchWeights(.net$id))
    }
    .trial <- tryCatch(.nnTrialSolve(.ui, .adata), error = function(e) NULL)
    ## The trial solve went through rxSolve.rxUi, which clears the active
    ## par-loader on exit.  Leaving it cleared silently disarms weight injection
    ## for every remaining solve in the fit -- the objective then never moves.
    ## Re-arm unconditionally: the solve clears it whether or not it succeeded.
    .nnLoaderOn()
  }

  ## INPUT SCALING (R/nnScale.R).  A network fed raw model quantities -- amounts
  ## in the hundreds, concentrations in the hundredths -- starts saturated, and a
  ## saturated network has no input derivative: no FOCEi sensitivity for a latent
  ## eta, and no weight-training signal.  Rescale the first-layer weights by each
  ## input's typical magnitude, measured from the trial solve above.
  ##
  ## Only for a model that has never been trained.  Trained weights already
  ## embody whatever scaling the previous fit found, so rescaling them would
  ## silently corrupt a refit's warm start.
  if (.untrained) {
    .scales <- tryCatch(.nnInputScales(.trial, .adata, .aug$nets, .etaAll),
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

  ## The weight penalty (R/nnPenalty.R).  Built once, outside the untrained gate,
  ## so a refit is regularizable too.  NULL when both lambdas are 0, which is the
  ## strict no-op: nnControl(l2 = 0, smooth = 0) reproduces an unregularized fit.
  ## The grid's ranges come from the pre-rescale trajectory, which is fine -- it
  ## is a region to measure curvature over, not a likelihood.
  .profiles <- tryCatch(lapply(.aug$nets, function(.n) {
    if (is.null(.n$inputs)) return(NULL)
    .nnInputProfile(.n$inputs, .trial, .adata, .etaAll)
  }), error = function(e) NULL)
  .pen <- .nnPenSpec(.aug, .profiles, sched$l2, sched$smooth)

  .wPlaceholder <- stats::setNames(rep(0, .aug$nW), .aug$weights)
  ## `pen` rides on the FACTORY, not the returned closure, so it is captured once
  ## and every caller of the step -- the round loop and the naive-pooled warm-up
  ## alike -- is penalized without having to remember to pass it.
  .weightStep <- .nnWeightStepper(.aug, .data, .idCol, .timeCol, .obs, .dv,
                                  .wPlaceholder, pen = .pen)
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
  ## score comes from the endpoint's distribution (a count endpoint) or from the
  ## inner fit (anything else).
  if (is.na(.aug$errAdd) && is.na(.aug$errProp) && !.exact && is.null(.aug$dist)) {
    stop("nlmixr2nn: this error model has no add()/prop() term; ",
         "fit with nnControl(cotangent = \"exact\")", call. = FALSE)
  }
  .obsIdVals <- .data[[.idCol]][.obs]
  ## The capture store's flat index is id * stride + k, and the SAME stride has
  ## to be used on both sides -- it used to be the literal 1024 in three places
  ## here and a #define in C, which silently dropped any subject with 1024+
  ## observations.  It is now sized from this data set and threaded through.
  .capDims <- .nnCapDims(.data, .idCol, .obs)
  .capStride <- .capDims$kStride
  .obsKey <- (match(.obsIdVals, unique(.obsIdVals)) - 1L) * .capStride +
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

  list(ui = .ui, data = .data, innerEst = .innerEst, control = .innerCtl,
       origTablesCov = .origTablesCov, aug = .aug, baseBases = .baseBases,
       existing = .existing, idCol = .idCol, timeCol = .timeCol,
       obs = .obs, dv = .dv,
       wPlaceholder = .wPlaceholder, weightStep = .weightStep,
       hasEta = .hasEta, latent = .latent,
       exact = .exact, exactSelf = .exactSelf, exactPosthoc = .exactPosthoc,
       capDims = .capDims, capStride = .capStride, obsKey = .obsKey,
       knob = .knob, interleave = .interleave,
       th0 = .th0, errPar0 = .errPar0, sched = sched, pen = .pen)
}

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
.nnRunPopFit <- function(ctx) {
  .ui <- ctx$ui
  .data <- ctx$data
  .aug <- ctx$aug
  .innerEst <- ctx$innerEst
  .innerCtl <- ctx$control
  .baseBases <- ctx$baseBases
  .hasEta <- ctx$hasEta
  .idCol <- ctx$idCol
  .timeCol <- ctx$timeCol
  .obs <- ctx$obs
  .dv <- ctx$dv
  .wPlaceholder <- ctx$wPlaceholder
  .th0 <- ctx$th0
  .errPar0 <- ctx$errPar0
  sched <- ctx$sched
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
  .wFit <- .nnPopWarmStart(.aug, .data, .idCol, .timeCol, .obs, .dv, .wPlaceholder,
                           .th0, .errPar0, .nnAllTorchWeights(.aug), .innerEst,
                           sched$rounds, exactCtx = .exactCtx, pen = ctx$pen)
  .nnSetAllTorchWeights(.aug, .wFit)
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
  .fit <- .nnAddTablesCov(.fit, .wFit, .baseBases, .aug, .matEst, ctx$origTablesCov)
  .nnStoreNnFit(ctx, .fit, .wFit,
                ## same columns as the round loop's parHist (R/nnEst.R): both
                ## branches fill the one `nnParHist` slot, so a column added to
                ## one and not the other ships two schemas under one name
                data.frame(round = 1L, objf = .fit$objf,
                  pen = .nnPenFrozenValue(ctx$pen, .wFit, .fit$objf),
                  errAdd = if (is.na(.aug$errAdd)) NA_real_ else .fit$theta[[.aug$errAdd]],
                  errProp = if (is.na(.aug$errProp)) NA_real_ else .fit$theta[[.aug$errProp]],
                  rmse = NA_real_, wChange = NA_real_, objfChange = NA_real_),
                converged = TRUE, nRun = 1L)
}

## The two population pre-fits that seed the loop.  Both act on the torch modules
## in place and return nothing -- the weights ARE the state being warmed.
.nnRunWarmStarts <- function(ctx) {
  .aug <- ctx$aug
  .data <- ctx$data
  .idCol <- ctx$idCol
  .timeCol <- ctx$timeCol
  .obs <- ctx$obs
  .dv <- ctx$dv
  .wPlaceholder <- ctx$wPlaceholder
  .th0 <- ctx$th0
  .errPar0 <- ctx$errPar0
  .existing <- ctx$existing
  .weightStep <- ctx$weightStep
  sched <- ctx$sched
  ## the nlm bridge: a gradient-based population (eta=0) weight pre-fit seeding the
  ## joint fit with a robust weight vector (unless the model already carries
  ## trained weights, in which case those are the warm start).
  if (!identical(sched$warmStart, "none") && is.null(.existing)) {
    .wPop <- .nnPopWarmStart(.aug, .data, .idCol, .timeCol, .obs, .dv, .wPlaceholder,
                             .th0, .errPar0, .nnAllTorchWeights(.aug),
                             sched$warmStart, sched$warmPopIters, pen = ctx$pen)
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
  invisible()
}

## The round loop.  mode="joint" interleaves warm-started PARTIAL inner steps
## (outerPerRound outer iterations, via the estimator's outer-step knob) with
## weight steps -- co-descending the population parameters and the weights -- and
## falls back to the block-coordinate iterate loop for inner estimators without a
## partial outer step; mode="iter" runs a full inner fit each round.
##
## Returns the last inner fit plus the history needed to finish it -- deliberately
## NOT the finished fit, so the loop can be run and inspected on its own, without
## the table and covariance work.
.nnRunLoop <- function(ctx) {
  .ui <- ctx$ui
  .data <- ctx$data
  .aug <- ctx$aug
  .innerEst <- ctx$innerEst
  .innerCtl <- ctx$control
  .baseBases <- ctx$baseBases
  .weightStep <- ctx$weightStep
  .hasEta <- ctx$hasEta
  .latent <- ctx$latent
  .exact <- ctx$exact
  .exactSelf <- ctx$exactSelf
  .exactPosthoc <- ctx$exactPosthoc
  .capDims <- ctx$capDims
  .capStride <- ctx$capStride
  .obsKey <- ctx$obsKey
  .knob <- ctx$knob
  .interleave <- ctx$interleave
  sched <- ctx$sched
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
    ## self-capture during the fit; the store is sized from this data set
    if (.exactSelf) .nnCapReset(TRUE, .capDims$nId, .capDims$kStride)
    .fit <- suppressWarnings(suppressMessages(
      nlmixr2est::nlmixr2(.fitUi, .dw, est = .innerEst, control = .roundCtl)))
    if (.interleave) .curUi <- .fit$ui                       # warm-start next round
    ## exact per-obs cotangent (any residual model): from the inner fit when it
    ## self-captures, else from a dedicated FOCEi posthoc at the fit's estimates
    ## (maxOuter=0 keeps the outer parameters, re-optimizes the EBEs, and fires the
    ## contribution hook cleanly).  The augmented solve then uses THOSE EBEs.
    .dLLdfObs <- NULL
    .ebeFit <- .fit
    if (.exact) {
      if (.exactPosthoc) {
        ## This capture fit is FOCEi, whatever the round's estimator is -- so the
        ## weight block has to be registered at FOCEi's offset for the duration,
        ## not the outer estimator's.  Without this a SAEM round captured its
        ## cotangents from a network read one slot off, and its eta recovery
        ## collapsed while every assertion but one still passed.
        .phBases <- .nnEstBases(.ui, "focei", .aug)
        if (!is.null(.phBases)) .nnSetNetBases(.aug$nets, .phBases)
        .nnCapReset(TRUE, .capDims$nId, .capDims$kStride)
        .ebeFit <- suppressWarnings(suppressMessages(
          nlmixr2est::nlmixr2(.fit$finalUi, .dw, est = "focei",
            nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 0L,
                                     maxInnerIterations = 30L, calcTables = FALSE))))
        ## back to the round's estimator for everything after
        .nnSetNetBases(.aug$nets, .baseBases)
      }
      .cap <- .nnCapGet()
      .nnCapReset(FALSE)
      if (!is.null(.cap) && length(.cap$id)) {
        .cand <- .cap$dLLdf[match(.obsKey, .cap$id * .capStride + .cap$k)]
        if (!anyNA(.cand)) .dLLdfObs <- .cand
      }
      ## Capture failed.  Falling back to the closed form is only safe where the
      ## closed form is the right score for this endpoint; on a transformed one
      ## it is a different function, and using it would train on the wrong
      ## gradient without saying so.
      if (is.null(.dLLdfObs) &&
            !.nnCanUseClosedForm(.aug$ep, .aug$errAdd, .aug$errProp)) {
        stop(sprintf(paste0(
          "nlmixr2nn: cotangent=\"exact\" captured no per-observation cotangents ",
          "(est=\"%s\", endpoint transformation '%s').  There is no closed-form ",
          "score to fall back to for this endpoint, and using the ",
          "additive/proportional one would silently train on a different ",
          "gradient."), .innerEst,
          if (is.null(.aug$ep$transform)) "untransformed" else .aug$ep$transform),
          call. = FALSE)
      }
    }
    ## population: no EBEs
    .ebes <- if (.hasEta) {
      stats::setNames(.ebeFit$eta[[.latent]], as.character(.ebeFit$eta[["ID"]]))
    } else {
      stats::setNames(numeric(0), character(0))
    }
    .thetas <- .fit$theta
    .errPar <- list(add = if (is.na(.aug$errAdd)) 0 else .fit$theta[[.aug$errAdd]],
                    prop = if (is.na(.aug$errProp)) 0 else .fit$theta[[.aug$errProp]])
    ## The penalty AT THE WEIGHTS THE INNER FIT JUST SAW -- before any weight step
    ## moves them.  Taking it from the step's return instead would report the
    ## value at weights up to `wSteps` updates later than the objf beside it, and
    ## `objf + pen` would then not be a penalized objective at any single point.
    ## Freeze the normalizer here when the population pre-fit did not already
    ## (warmStart = "none", e.g. a refit): these are the weights the inner fit
    ## just saw, so objf and the penalty refer to the same point.
    .nnPenFreeze(ctx$pen, .nnAllTorchWeights(.aug), .fit$objf)
    .pen <- .nnPenalty(ctx$pen, .nnAllTorchWeights(.aug))$value
    for (.ws in seq_len(sched$wSteps)) {
      .rmse <- .weightStep(.ebes, .errPar, .thetas, .dLLdfObs)$rmse
    }
    .wNow <- .nnAllTorchWeights(.aug)
    .wChange <- sqrt(sum((.wNow - .wPrev)^2)) / (sqrt(sum(.wPrev^2)) + 1e-8)
    .wPrev <- .wNow
    ## Convergence is judged on what the weight step actually descends -- the
    ## PENALIZED objective -- while `objf` stays the unpenalized -2LL that gets
    ## reported and that AIC/BIC are formed from.  With the penalty off, `pen` is
    ## exactly 0 and this is bit-identical to comparing objf alone.
    .objfPen <- .fit$objf + .pen
    .objfChange <- if (is.na(.objfPrev)) Inf else abs(.objfPen - .objfPrev) / (abs(.objfPrev) + 1e-8)
    .objfPrev <- .objfPen
    .parHist[[.round]] <- data.frame(round = .round, objf = .fit$objf, pen = .pen,
                                     errAdd = .errPar$add, errProp = .errPar$prop,
                                     rmse = .rmse, wChange = .wChange, objfChange = .objfChange)
    ## stop when the weights (and, when interleaving, the objective) stabilise;
    ## isTRUE guards a NaN change (e.g. an unstable solve) -> keep going, don't crash
    .nnProgressTick(.prog)
    .stop <- isTRUE(.wChange < sched$tol && (!.interleave || .objfChange < sched$tol))
    if (sched$tol > 0 && .round > 1L && .stop) {
      .converged <- TRUE
      break
    }
  }
  .parHist <- .parHist[seq_len(.nRun)]
  .nnProgressStop(.prog)
  .nnRunSummary(.interleave, .converged, .nRun, sched$rounds, .wChange, .objfChange,
                .fit$objf, .parHist, .quiet)

  list(fit = .fit, weights = .nnAllTorchWeights(.aug),
       parHist = do.call(rbind, .parHist), converged = .converged, nRun = .nRun)
}
