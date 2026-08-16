.onLoad <- function(libname, pkgname) {
  ## install rxode2's C entry points into this package's function-pointer
  ## globals, then register the par-loader hook (both need the table populated)
  .Call(`_nlmixr2nn_iniRxodePtrs`, rxode2::.rxode2ptrs(), PACKAGE = "nlmixr2nn")
  .Call(`_nlmixr2nn_registerLoader`, PACKAGE = "nlmixr2nn")
  .registerRxode2()
  ## install nlmixr2est's likelihood-contribution registry (if available) and
  ## register the neural-network contribution bundle so nn weights get their
  ## objective cotangents during estimation.  Soft dependency: nn() still solves
  ## and simulates without nlmixr2est.
  if (requireNamespace("nlmixr2est", quietly = TRUE) &&
      exists(".nlmixr2estLikContribPtrs", envir = asNamespace("nlmixr2est"))) {
    .p <- try(nlmixr2est:::.nlmixr2estLikContribPtrs(), silent = TRUE)
    if (!inherits(.p, "try-error")) {
      .Call(`_nlmixr2nn_iniLikContrib`, .p, PACKAGE = "nlmixr2nn")
      .Call(`_nlmixr2nn_registerContrib`, PACKAGE = "nlmixr2nn")
    }
  }
  ## transparent workflow: claim estimations of models containing nn() so a
  ## standard est (focei/saem/...) trains the embedded network (R/nnInterceptor.R)
  .nnRegisterInterceptor()
  ## rehydrate transient nn shapes from the ui when a saved fit/model is reloaded
  ## in a fresh session (so predict()/simulate() strides the weights carried in
  ## rxForcedPars()).  Guarded: older rxode2 lacks the ui-prep hook API.
  if ("rxRegisterUiPrep" %in% getNamespaceExports("rxode2")) {
    rxode2::rxRegisterUiPrep("nlmixr2nn:rehydrate", .nnRehydrate)
  }
  ## move each freshly parsed model's parse-time network state (shapes + the
  ## weights nn() drew) onto the ui itself, while it is still a mutable
  ## environment -- see R/nnAdopt.R.  Without this a model would leave the
  ## parser with no weights at all.
  .nnRegisterAdopt()
}

.nnTransRows <- function() {
  ## probe helpers + the generated nn<K> family and nnWg<K> weight-gradient family
  ## (.nnGenNames/.nnGenNargs + .nnWgGenNames/.nnWgGenNargs, R/nnGen.R)
  .rxFun <- c("nnprobe", "nnnpars", .nnGenNames, .nnWgGenNames)
  .nargs <- c(2L, 1L, .nnGenNargs, .nnWgGenNargs)
  data.frame(
    rxFun = .rxFun,
    type  = paste0("rxode2_fn", ifelse(.nargs == 1L, "", as.character(.nargs))),
    nargs = .nargs,
    stringsAsFactors = FALSE
  )
}

.registerRxode2 <- function() {
  .rows <- .nnTransRows()
  .cur <- rxode2parseGetTranslation()
  .newRows <- data.frame(
    rxFun      = .rows$rxFun,
    fun        = .rows$rxFun,
    type       = .rows$type,
    package    = "nlmixr2nn",
    packageFun = .rows$rxFun,
    argMin     = .rows$nargs,
    argMax     = .rows$nargs,
    threadSafe = 1L,
    stringsAsFactors = FALSE
  )
  .cur <- .cur[.cur$package != "nlmixr2nn", , drop = FALSE]
  rxode2parseAssignTranslation(rbind(.cur, .newRows))

  ## probe (constant derivatives)
  rxD("nnprobe", list(function(idx, x) "0.0", function(idx, x) "0.0"))
  rxD("nnnpars", list(function(x) "0.0"))

  ## nn<K> derivative chains (id is a constant -> non-differentiable; d/d input_j
  ## -> nn<K>_d<j>, second derivs -> nn<K>_d<j>_d<l>).  Generated in R/nnGen.R.
  .nnGenRegisterD()
  invisible()
}

.onUnload <- function(libpath) {
  .cur <- try(rxode2parseGetTranslation(), silent = TRUE)
  if (!inherits(.cur, "try-error")) {
    try(rxode2parseAssignTranslation(
      .cur[.cur$package != "nlmixr2nn", , drop = FALSE]
    ), silent = TRUE)
  }
  for (.nm in .nnTransRows()$rxFun) {
    suppressWarnings(try(rxRmFun(.nm), silent = TRUE))
  }
  ## drop the estimation interceptor, par-loader hook + likelihood contribution
  .nnUnregisterInterceptor()
  if ("rxRemoveUiPrep" %in% getNamespaceExports("rxode2")) {
    try(rxode2::rxRemoveUiPrep("nlmixr2nn:rehydrate"), silent = TRUE)
  }
  .nnUnregisterAdopt()
  try(.Call(`_nlmixr2nn_nnUnregisterLoader`), silent = TRUE)
  ## every registration handed to nlmixr2est must come back before this DLL goes
  ## away: nlmixr2est cannot tell that we unloaded, so anything left behind is
  ## called on the next objective evaluation.  The contribution bundle is a raw
  ## function pointer into this DLL; the outer-network callback is a closure whose
  ## environment is this (now dead) namespace.
  try(.Call(`_nlmixr2nn_removeContrib`), silent = TRUE)
  try(nnOuterUnregister(), silent = TRUE)
  library.dynam.unload("nlmixr2nn", libpath)
}
