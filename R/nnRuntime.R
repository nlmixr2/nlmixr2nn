## Runtime plumbing shared by every nn path: the named par-loader flag, the
## per-observation cotangent capture store, and weight-block base registration.
##
## None of it is training-specific -- a plain rxSolve() of a model carrying nn()
## goes through the same loader flag -- so it is kept apart from the engine.

## The nn weight par-loader is registered under the NAME "nlmixr2nn:nnParLoader"
## (see src/nlmixr2nnPtr.cpp), so rxode2 runs it ONLY while this injector is the
## active flag -- otherwise it would clobber an unrelated model's par_ptr.  A model
## carrying nn() is flagged on its ui (nnUpdate() sets rxParLoader()), which the
## rxSolve.rxUi bridge honors; the estimation engine below also sets the flag
## directly around its internal solves (inner fits + augmented solves) because they
## bypass that bridge.  Both no-op gracefully on an older rxode2.
.nnLoaderName <- "nlmixr2nn:nnParLoader"
## rxode2's public wrappers rather than .Call()ing its compiled entry points by
## name: reaching into another package's DLL is not a supported interface and
## R CMD check flags it.  Both are wrapped because an older rxode2 does not have
## them, in which case the loader simply never activates -- which is correct, as
## that rxode2 has no named-loader dispatch either.
.nnLoaderOn <- function() {
  tryCatch(rxode2::rxSetActiveParLoader(.nnLoaderName), error = function(e) NULL)
}
.nnLoaderOff <- function() {
  tryCatch(rxode2::rxClearActiveParLoader(), error = function(e) NULL)
}

## Per-observation error-model cotangent capture (nlmixr2est likelihood-contribution
## hook).  .nnCapReset(TRUE, nId, kStride) sizes + arms it before an inner fit;
## .nnCapGet() returns list(id, k, dLLdf) of the fit's converged per-obs
## cotangents.  No-op on an nlmixr2est without the lik-contrib API.
##
## The store is sized HERE, from the data, because the hook runs inside
## nlmixr2est's OpenMP region and must not allocate: it used to grow itself with
## R_chk_realloc() from a worker thread, which is a data race on the buffer and
## an R API call off the main thread.
.nnCapReset <- function(on, nId = 1L, kStride = 1L) {
  tryCatch(.Call("_nlmixr2nn_capReset", on, as.integer(nId), as.integer(kStride),
                 PACKAGE = "nlmixr2nn"),
           error = function(e) NULL)
}
.nnCapGet <- function() {
  tryCatch(.Call("_nlmixr2nn_capGet", PACKAGE = "nlmixr2nn"), error = function(e) NULL)
}

## Register every network at a given set of per-id bases.  The same three-line
## loop appeared in half a dozen places, each an opportunity to bind the wrong
## offset silently.
.nnSetNetBases <- function(nets, bases) {
  for (.net in nets) {
    .b <- bases[[as.character(.net$id)]]
    if (!is.null(.b) && !is.na(.b)) {
      nnSetMeta(.net$id, .b, .net$K, .net$H, .net$act)
    }
  }
  invisible()
}

## Per-network weight-block bases for the model a given estimator solves, or
## NULL when that model cannot be built.
.nnEstBases <- function(ui, est, aug) {
  .p <- .nnEstSolveParams(ui, est)
  if (is.null(.p)) return(NULL)
  .b <- tryCatch(nnUpdate(ui, params = .p), error = function(e) NULL)
  if (is.null(.b) || !nrow(.b)) return(NULL)
  stats::setNames(.b$base, as.character(.b$id))
}

## Store dimensions for a dataset: subjects, and the largest number of
## observations any one of them has.  These are exactly what the hook's
## (id, k) indices are bounded by.
.nnCapDims <- function(data, idCol, obs) {
  .ids <- data[[idCol]][obs]
  if (length(.ids) == 0L) return(list(nId = 1L, kStride = 1L))
  list(nId = length(unique(.ids)),
       kStride = max(as.integer(table(.ids))))
}
