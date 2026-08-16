#' nlmixr2nn: Neural-Network ODEs for 'rxode2' and 'nlmixr2'
#'
#' Embed neural networks inside ODE right-hand sides for rxode2/nlmixr2.  A
#' model containing [nn()] is an ordinary rxode2/nlmixr2 model: it is seeded when
#' it is parsed, it solves, [nlmixr2est::nlmixr2()] trains it, and a fitted model
#' is self-contained.  A C++ libtorch module is the training optimizer, loaded
#' from the model's own weights.
#'
#' The public surface is deliberately four functions: [nn()] in the model,
#' [nnControl()] to steer training, and [nnEval()]/[nnWeights()] to inspect what
#' a network learned.
#'
#' The generated `nn<K>()` / `nnWg<K>()` evaluator family is NOT exported.
#' Compiled model code reaches it through `R_RegisterCCallable()` (see
#' `src/init.c`), not through the R namespace, so exporting it would only put
#' three dozen machine-facing names in front of users.
#'
#' @useDynLib nlmixr2nn, .registration = TRUE
#' @importFrom rxode2 rxD rxRmFun rxUdfUi rxUdfUiIniDf rxUdfUiNum
#' @importFrom rxode2 rxode2parseAssignTranslation rxode2parseGetTranslation
#' @importFrom stats rnorm
#' @keywords internal
"_PACKAGE"
