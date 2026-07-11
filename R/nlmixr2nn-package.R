#' nlmixr2nn: Neural-Network ODEs for 'rxode2' and 'nlmixr2'
#'
#' Embed neural networks inside ODE right-hand sides for rxode2/nlmixr2, backed
#' by a C++ libtorch module whose weights are injected into the solve parameter
#' vector by an rxode2 par-loader hook.
#'
#' @useDynLib nlmixr2nn, .registration = TRUE
#' @importFrom rxode2 rxD rxRmFun rxUdfUi rxUdfUiIniDf rxUdfUiNum
#' @importFrom rxode2 rxode2parseAssignTranslation rxode2parseGetTranslation
#' @importFrom stats rnorm
#' @rawNamespace exportPattern("^nn[0-9]")
#' @keywords internal
"_PACKAGE"
