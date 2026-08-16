## Transparent DeepPumas-style workflow: loading nlmixr2nn makes nn() usable in a
## model, and fitting that model with a STANDARD estimator trains the embedded
## network automatically -- no special est="nn".  This is done through
## nlmixr2est's estimation-interceptor API (registerEstInterceptor): the
## interceptor is consulted before the ordinary method dispatches, claims the fit
## when the model contains an nn() term, and runs the NN training loop (.nnRun)
## using the requested est (focei/saem/imp/...) as the inner engine.
##
##   nlmixr2(model, data, "focei", foceiControl(), nn = nnControl())
##
## The nn= argument (a pure nnControl() training schedule) is optional; without
## it a default nnControl() is used.  Models without nn() are declined (NULL), so
## ordinary fits are unaffected.

## Does this ui model contain an nn() term?  After UI assembly the nn() UDF has
## expanded to nn<K>(...) calls in the model text, so grep the normalized lines.
.nnUiHasNn <- function(ui) {
  .lines <- tryCatch(ui$lstChr, error = function(e) NULL)
  if (is.null(.lines)) return(FALSE)
  any(grepl("\\bnn[0-9]+\\s*\\(", .lines))
}

## The interceptor consulted by nlmixr2est::nlmixr2Est().  Declines (NULL) unless
## the model has an nn() term; otherwise trains the network with the requested
## inner estimator and returns the finalized fit.
.nnEstInterceptor <- function(env) {
  .ui <- tryCatch(rxode2::rxUiDecompress(env$ui), error = function(e) NULL)
  if (is.null(.ui) || !.nnUiHasNn(.ui)) return(NULL)          # not an nn model
  .user <- env$.nlmixr2Dots$nn
  if (!is.null(.user) && !inherits(.user, "nnControl")) {
    stop("the nn= argument to nlmixr2() must be an nnControl()", call. = FALSE)
  }
  ## The schedule is INFERRED from the model, the data and the estimator that was
  ## asked for; `nn=` is an optional set of overrides, not a required argument.
  ## Inference needs the network size, which comes from the shapes the parse
  ## attached to the ui.
  .meta <- .nnUiMeta(.ui)
  if (is.null(.meta) || length(.meta) == 0L) .meta <- .nnEnv$reg
  .nW <- tryCatch(
    sum(vapply(.meta, function(m) as.integer(m$H * m$K + 2L * m$H + 1L), integer(1))),
    error = function(e) NA_integer_)
  .hasTrained <- isTRUE(tryCatch(get("nnTrained", envir = .ui, inherits = FALSE),
                                 error = function(e) FALSE))
  .sched <- .nnResolveSched(.user, .nnInferSched(env, .nW, .hasTrained))
  .nnRun(env, .sched)
}

.nnRegisterInterceptor <- function() {
  if (requireNamespace("nlmixr2est", quietly = TRUE) &&
      exists("registerEstInterceptor", envir = asNamespace("nlmixr2est"))) {
    nlmixr2est::registerEstInterceptor("nlmixr2nn", .nnEstInterceptor)
    TRUE
  } else {
    FALSE
  }
}

.nnUnregisterInterceptor <- function() {
  if (requireNamespace("nlmixr2est", quietly = TRUE) &&
      exists("removeEstInterceptor", envir = asNamespace("nlmixr2est"))) {
    try(nlmixr2est::removeEstInterceptor("nlmixr2nn"), silent = TRUE)
  }
  invisible()
}
