## Binding a model's weights to a solve: where the values come from, which
## par_ptr offset each network sits at, and how they reach an estimator's kernel.
##
## The values live in three places at different moments -- rxForcedPars on the
## ui (persisted), the torch modules (during training), and data covariate
## columns (for kernels that bypass the loader) -- and these move them between
## the three.  .nnRehydrate() is the reload path: shapes are transient, so a
## ui coming back from saveRDS() rebuilds them here.

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


## Resolve a standard data column by name, case-insensitively.
##
## nlmixr2 accepts ID/id, TIME/time, DV/dv, EVID/evid interchangeably, and the
## engine reads several of them by name.  A missed column is NOT an error:
## `data$time` on a data set written with TIME -- the NONMEM convention, and what
## most users write -- is NULL, the observation match then yields all NA, and the
## assembled gradient is quietly NaN.  Every test in this package happened to use
## the lower-case spelling, so that went unnoticed until a count endpoint, whose
## score asserts its input is finite, refused it.
.nnDataCol <- function(data, want) {
  .i <- match(tolower(want), tolower(names(data)))
  if (is.na(.i)) NA_character_ else names(data)[.i]
}
