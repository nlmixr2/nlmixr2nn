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
#' The generated `nn<K>()` evaluator family and its derivatives ARE exported,
#' and must be.  Compiled model code reaches them through
#' `R_RegisterCCallable()`, but rxode2's symengine renderer resolves a model
#' function by NAME with `get(fun, ...)` over the search path when it builds
#' derivatives, so an unexported `nn1` fails with "function 'nn1' or its
#' derivatives are not supported in rxode2" the moment a model needs a Jacobian.
#' They are machine-facing rather than user-facing, hence `@keywords internal`.
#' (`nnWg<K>()` is deliberately not exported: it only ever appears in generated
#' model text handed to C, never through the symengine path.)
#'
#' @useDynLib nlmixr2nn, .registration = TRUE
#' @importFrom rxode2 rxD rxRmFun rxUdfUi rxUdfUiIniDf rxUdfUiNum
#' @importFrom rxode2 rxode2parseAssignTranslation rxode2parseGetTranslation
#' @importFrom stats rnorm
#' @rawNamespace exportPattern("^nn[0-9]")
#' @keywords internal
"_PACKAGE"
