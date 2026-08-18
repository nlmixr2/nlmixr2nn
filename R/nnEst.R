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

  .nnTorchRequire("fitting a model that contains nn()")

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
  if (.untrained) {
    .inputsById <- tryCatch({
      .cl <- .nnParseCallAll(.ui$lstChr)
      stats::setNames(lapply(.cl, function(.c) .c$inputs),
                      vapply(.cl, function(.c) as.character(.c$id), character(1)))
    }, error = function(e) NULL)
    if (!is.null(.inputsById)) {
      ## Bind the registry to the BASE model and push the current weights before
      ## the trial solve.
      ##
      ## Filling the data columns is not enough on its own: the loader also
      ## injects from its own buffer, and inside the loop `.nnEnv$training` is
      ## TRUE, so the ui-prep hook does not rebind.  Without this the trial solve
      ## ran against whatever the previous fit left in the buffer -- zeros in a
      ## fresh session -- so the state trajectory, and therefore the derived
      ## scale, differed from run to run.  That made whole fits irreproducible
      ## under a fixed set.seed().
      for (.net in .aug$nets) {
        nnSetMeta(.net$id, .baseBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
        nnSetWeights(.net$id, nnTorchWeights(.net$id))
      }
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
    ## self-capture during the fit; the store is sized from this data set
    if (.exactSelf) .nnCapReset(TRUE, .capDims$nId, .capDims$kStride)
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
      .cap <- .nnCapGet(); .nnCapReset(FALSE)
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
    .ebes <- if (.hasEta) {
      stats::setNames(.ebeFit$eta[[.latent]], as.character(.ebeFit$eta[["ID"]]))
    } else stats::setNames(numeric(0), character(0))         # population: no EBEs
    .thetas <- .fit$theta
    .errPar <- list(add = if (is.na(.aug$errAdd)) 0 else .fit$theta[[.aug$errAdd]],
                    prop = if (is.na(.aug$errProp)) 0 else .fit$theta[[.aug$errProp]])
    for (.ws in seq_len(sched$wSteps)) {
      .rmse <- .weightStep(.ebes, .errPar, .thetas, .dLLdfObs)$rmse
    }
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
