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
