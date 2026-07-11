.onLoad <- function(libname, pkgname) {
  ## install rxode2's C entry points into this package's function-pointer
  ## globals, then register the par-loader hook (both need the table populated)
  .Call(`_rxode2nn_iniRxodePtrs`, rxode2::.rxode2ptrs(), PACKAGE = "rxode2nn")
  .Call(`_rxode2nn_registerLoader`, PACKAGE = "rxode2nn")
  .registerRxode2()
}

.nnTransRows <- function() {
  data.frame(
    rxFun = c("nnprobe", "nnnpars",
              "nn1", "nn1_d1", "nn1_d1_d1",
              "nn2", "nn2_d1", "nn2_d2",
              "nn2_d1_d1", "nn2_d1_d2", "nn2_d2_d2"),
    type = c("rxode2_fn2", "rxode2_fn",
             "rxode2_fn2", "rxode2_fn2", "rxode2_fn2",
             "rxode2_fn3", "rxode2_fn3", "rxode2_fn3",
             "rxode2_fn3", "rxode2_fn3", "rxode2_fn3"),
    nargs = c(2L, 1L, 2L, 2L, 2L, 3L, 3L, 3L, 3L, 3L, 3L),
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
    package    = "rxode2nn",
    packageFun = .rows$rxFun,
    argMin     = .rows$nargs,
    argMax     = .rows$nargs,
    threadSafe = 1L,
    stringsAsFactors = FALSE
  )
  .cur <- .cur[.cur$package != "rxode2nn", , drop = FALSE]
  rxode2parseAssignTranslation(rbind(.cur, .newRows))

  ## probe (constant derivatives)
  rxD("nnprobe", list(function(idx, x) "0.0", function(idx, x) "0.0"))
  rxD("nnnpars", list(function(x) "0.0"))

  ## nn1: d/dx1 -> nn1_d1 ; d2/dx1^2 -> nn1_d1_d1
  ## (first arg `id` is a constant -> non-differentiable, NULL)
  rxD("nn1", list(NULL,
    function(id, x1) paste0("nn1_d1(", id, ",", x1, ")")))
  rxD("nn1_d1", list(NULL,
    function(id, x1) paste0("nn1_d1_d1(", id, ",", x1, ")")))

  ## nn2: gradient w.r.t. each input
  rxD("nn2", list(NULL,
    function(id, x1, x2) paste0("nn2_d1(", id, ",", x1, ",", x2, ")"),
    function(id, x1, x2) paste0("nn2_d2(", id, ",", x1, ",", x2, ")")))
  ## Hessian rows (symmetry: d(nn2_d1)/dx2 == d(nn2_d2)/dx1 == nn2_d1_d2)
  rxD("nn2_d1", list(NULL,
    function(id, x1, x2) paste0("nn2_d1_d1(", id, ",", x1, ",", x2, ")"),
    function(id, x1, x2) paste0("nn2_d1_d2(", id, ",", x1, ",", x2, ")")))
  rxD("nn2_d2", list(NULL,
    function(id, x1, x2) paste0("nn2_d1_d2(", id, ",", x1, ",", x2, ")"),
    function(id, x1, x2) paste0("nn2_d2_d2(", id, ",", x1, ",", x2, ")")))
  invisible()
}

.onUnload <- function(libpath) {
  .cur <- try(rxode2parseGetTranslation(), silent = TRUE)
  if (!inherits(.cur, "try-error")) {
    try(rxode2parseAssignTranslation(
      .cur[.cur$package != "rxode2nn", , drop = FALSE]
    ), silent = TRUE)
  }
  for (.nm in .nnTransRows()$rxFun) {
    suppressWarnings(try(rxRmFun(.nm), silent = TRUE))
  }
  ## drop the par-loader hook from rxode2 before unloading our DLL
  try(.Call(`_rxode2nn_nnUnregisterLoader`), silent = TRUE)
  library.dynam.unload("rxode2nn", libpath)
}
