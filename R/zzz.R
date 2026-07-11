.onLoad <- function(libname, pkgname) {
  ## install rxode2's C entry points into this package's function-pointer
  ## globals, then register the par-loader hook (both need the table populated)
  .Call(`_nlmixr2nn_iniRxodePtrs`, rxode2::.rxode2ptrs(), PACKAGE = "nlmixr2nn")
  .Call(`_nlmixr2nn_registerLoader`, PACKAGE = "nlmixr2nn")
  .registerRxode2()
}

.nnTransRows <- function() {
  ## probe helpers + the generated nn<K> family (.nnGenNames/.nnGenNargs, R/nnGen.R)
  .rxFun <- c("nnprobe", "nnnpars", .nnGenNames)
  .nargs <- c(2L, 1L, .nnGenNargs)
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
  ## drop the par-loader hook from rxode2 before unloading our DLL
  try(.Call(`_nlmixr2nn_nnUnregisterLoader`), silent = TRUE)
  library.dynam.unload("nlmixr2nn", libpath)
}
