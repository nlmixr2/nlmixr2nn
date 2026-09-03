## Documentation for the generated evaluator family (R/nnGen.R).
##
## These are exported because rxode2 resolves a model function by name over the
## search path when it builds derivatives -- see the package-level help.  They
## are machine-facing, so they get one shared internal help page rather than
## thirty-four individual ones.

#' Compiled neural-network evaluators
#'
#' The `nn<K>()` family evaluates a network of `K` inputs, and `nn<K>_d<j>()` /
#' `nn<K>_d<j>_d<l>()` its first and second input derivatives.  They are
#' generated (see `tools/genNn.R`), registered with rxode2 as model functions,
#' and are not intended to be called directly -- use [nn()] in a model and
#' [nnEval()] to inspect a fitted network.
#'
#' @aliases nn1 nn1_d1 nn1_d1_d1 nn2 nn2_d1 nn2_d1_d1 nn2_d1_d2 nn2_d2 nn2_d2_d2 nn3 nn3_d1
#' @aliases nn3_d1_d1 nn3_d1_d2 nn3_d1_d3 nn3_d2 nn3_d2_d2 nn3_d2_d3 nn3_d3 nn3_d3_d3 nn4 nn4_d1
#' @aliases nn4_d1_d1 nn4_d1_d2 nn4_d1_d3 nn4_d1_d4 nn4_d2 nn4_d2_d2 nn4_d2_d3 nn4_d2_d4 nn4_d3
#' @aliases nn4_d3_d3 nn4_d3_d4 nn4_d4 nn4_d4_d4
#' @param id integer network id (0-based), the first argument of the model call.
#' @param ... the network inputs.
#' @return numeric vector of the network value or derivative at each input row.
#' @keywords internal
#' @name nnEvaluators
NULL
