## Moving parse-time network state onto the model.
##
## `nn()` runs during parsing and draws the network's weights, but `rxUdfUi()`
## can only return code (`replace`/`before`/`after`) and an `iniDf` -- it has no
## field that carries a value out to the ui.  So the values live briefly in
## `.nnEnv$reg` and are adopted by the model here, from rxode2's ui-assembly
## hook.
##
## That hook fires while the ui is still an ENVIRONMENT the caller shares, which
## is the only moment this can work: `rxUiCompress()` turns the ui into a list,
## and `rxUiDecompress()` then hands back a fresh environment on every call, so
## an assignment made any later is invisible to whoever holds the model.
##
## What lands on the ui:
##   nnMeta      - per-network shapes (id/K/H/act/weight names), sticky, so a
##                 saved model can rebuild the transient C-side registry
##   forcedPars  - the weight VALUES, sticky, injected into every solve column
##   nnTrained   - FALSE for a parse-time draw, TRUE once a fit has trained them

## Networks whose weight names all appear in this model's parameters.  A ui is
## adopted only from the parse that produced it, so a stale registry entry from
## an earlier model cannot leak onto an unrelated one.
.nnRegForUi <- function(ui) {
  .reg <- .nnEnv$reg
  if (length(.reg) == 0L) return(list())
  .params <- tryCatch(rxode2::rxModelVars(ui)$params, error = function(e) NULL)
  if (is.null(.params)) return(list())
  Filter(function(m) all(m$weights %in% .params), .reg)
}

#' Attach parse-time network state to a freshly assembled model
#'
#' Registered with `rxode2::rxRegisterUiAssembled()` when the package loads, so
#' it runs automatically; it is never called directly.
#'
#' @param ui a freshly assembled `rxUi` environment.
#' @return invisibly `TRUE` if the ui adopted networks, `FALSE` otherwise.
#' @keywords internal
#' @noRd
.nnAdopt <- function(ui) {
  if (!is.environment(ui)) return(invisible(FALSE))
  ## already carries its networks (a pipe, a refit, a reloaded model): leave the
  ## existing weights alone -- re-adopting would overwrite trained values with a
  ## fresh random draw.
  if (exists("nnMeta", envir = ui, inherits = FALSE)) return(invisible(FALSE))
  .nets <- .nnRegForUi(ui)
  if (length(.nets) == 0L) return(invisible(FALSE))

  .meta <- lapply(.nets, function(m) {
    list(id = m$id, K = m$K, H = m$H, act = m$act, weights = m$weights,
         metaVersion = if (is.null(m$metaVersion)) 1L else m$metaVersion)
  })
  names(.meta) <- vapply(.nets, function(m) as.character(m$id), character(1))
  assign("nnMeta", .meta, envir = ui)
  assign("nnTrained", FALSE, envir = ui)

  ## The drawn values, in the order the compiled layer strides them.  `.nets` is
  ## keyed by network id, so it must be unnamed before unlist() -- otherwise every
  ## weight name comes back prefixed ("0.rxnnW1_0_1_1") and matches nothing.
  .vals <- unlist(lapply(unname(.nets), function(m) m$values), use.names = TRUE)
  if (!is.null(.vals) && !anyNA(.vals)) {
    .fp <- tryCatch(rxode2::rxForcedPars(ui), error = function(e) NULL)
    .keep <- setdiff(names(.fp), names(.vals))       # never clobber other forcing
    rxode2::rxForcedPars(ui) <- c(.fp[.keep], .vals)
  }
  .sticky <- if (exists("sticky", envir = ui, inherits = FALSE)) {
    get("sticky", envir = ui, inherits = FALSE)
  } else character(0)
  assign("sticky", unique(c(.sticky, "nnMeta", "nnTrained")), envir = ui)
  invisible(TRUE)
}

## Mark a fitted model's weights as TRAINED.
##
## This flag is what separates "the random values nn() drew" from "values a fit
## produced", and two behaviours hinge on it: a refit uses trained weights as its
## warm start instead of re-running the population pre-fit, and -- more
## importantly -- input scaling is applied ONLY to untrained weights.  Trained
## weights already embody the scaling of the fit that produced them, so
## rescaling them again silently corrupts the warm start.
.nnMarkTrained <- function(ui, aug = NULL) {
  if (!is.environment(ui)) return(invisible(FALSE))
  assign("nnTrained", TRUE, envir = ui)
  .sticky <- if (exists("sticky", envir = ui, inherits = FALSE)) {
    get("sticky", envir = ui, inherits = FALSE)
  } else character(0)
  assign("sticky", unique(c(.sticky, "nnTrained")), envir = ui)
  invisible(TRUE)
}

.nnRegisterAdopt <- function() {
  if ("rxRegisterUiAssembled" %in% getNamespaceExports("rxode2")) {
    rxode2::rxRegisterUiAssembled("nlmixr2nn:adopt", .nnAdopt)
    return(invisible(TRUE))
  }
  invisible(FALSE)
}

.nnUnregisterAdopt <- function() {
  if ("rxRemoveUiAssembled" %in% getNamespaceExports("rxode2")) {
    try(rxode2::rxRemoveUiAssembled("nlmixr2nn:adopt"), silent = TRUE)
  }
  invisible()
}

#' Network weights carried by a model or fit
#'
#' Returns the weight vector a model currently uses: the values trained by a fit
#' when it has been fitted, otherwise the initial values drawn when the model was
#' parsed.
#'
#' @param x an `rxUi` model or an `nlmixr2` fit built with [nn()].
#' @return a named numeric vector of weights, or `NULL` if the model has none.
#' @export
#' @author Matthew L. Fidler
nnWeights <- function(x) {
  ## a fit carries the model as $finalUi; a model is already the model
  .ui <- if (inherits(x, "rxUi")) x else tryCatch(x$finalUi, error = function(e) NULL)
  if (!inherits(.ui, "rxUi")) .ui <- x
  .ui <- tryCatch(rxode2::rxUiDecompress(.ui), error = function(e) .ui)
  .meta <- tryCatch(get("nnMeta", envir = .ui, inherits = FALSE),
                    error = function(e) NULL)
  if (is.null(.meta) || length(.meta) == 0L) return(NULL)
  .wn <- unlist(lapply(.meta, function(m) m$weights), use.names = FALSE)
  .fp <- tryCatch(rxode2::rxForcedPars(.ui), error = function(e) NULL)
  if (is.null(.fp)) return(NULL)
  .w <- .fp[.wn]
  if (anyNA(.w)) return(NULL)
  .w
}
