## Turning the last inner fit into the deliverable.
##
## The intermediate fits are throwaway, so their tables and covariance are
## switched off for the loop and computed once, post-hoc, on the fit that is
## actually returned -- from the user's original control settings, with no
## re-estimation.

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
    fit <- tryCatch(nlmixr2est::addTable(fit), error = function(e) fit)
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

## Attach the trained network to the fit, for BOTH branches -- the round loop and
## the no-BSV population fit.
##
## The two used to carry their own copy of this block, and the copies had already
## drifted: the population branch never snapshotted the shape registry onto its
## stored ui.  That is harmless today only because parse-time adoption
## (.nnAdopt) puts the same shapes there first, so a saved QSP fit does reload --
## but it was one edit away from not being true, and the branch that would break
## is the one with no in-session symptom.  One writer now.
##
## `weights` is the GLOBAL weight vector in aug$weights order.  Everything lands
## on the fit ENV, never with `$<-`, which would add a data column.
.nnStoreNnFit <- function(ctx, fit, weights, parHist, converged, nRun) {
  .trained <- stats::setNames(weights, ctx$aug$weights)
  .fitEnv <- fit$env
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
  .nnMarkTrained(.storedUi, ctx$aug)
  assign("ui", .storedUi, envir = .fitEnv)
  ## the schedule the fit actually ran under.  Everything in it is INFERRED --
  ## the user typically writes no nnControl() at all -- so without this there is
  ## no way to see which cotangent source, mode or round count was chosen.
  assign("nnSched", ctx$sched, envir = .fitEnv)
  assign("nnParHist", parHist, envir = .fitEnv)
  assign("nnWeights", .trained, envir = .fitEnv)
  assign("nnConverged", converged, envir = .fitEnv)
  assign("nnRounds", nRun, envir = .fitEnv)
  fit
}
