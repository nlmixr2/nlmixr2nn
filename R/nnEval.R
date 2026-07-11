## Activation code mapping shared with src/nnEval.c (0=ReLU, 1=Softplus, 2=tanh)
.nnActCode <- c(relu = 0L, softplus = 1L, tanh = 2L)

## R-level entry points for the compiled MLP functions.  These mirror the
## rxode2 model-code names one-to-one: rxode2's symengine->rxode2 renderer
## resolves a registered function by get()-ing an R function of the same name
## and arity, so every model/derivative name must exist here (as with the mm
## example package).  They also let the functions be called directly from R.

#' Single-hidden-layer neural-network ODE function and its input derivatives
#'
#' `nn<K>(id, x1, ...)` evaluates the registered network `id` (see
#' [nnSetMeta()]) at the `K` inputs, reading its weights from the current solve
#' parameter vector.  The `_d<j>` / `_d<j>_d<l>` variants give the first and
#' second derivatives with respect to the inputs.
#'
#' @param id integer network id (matches [nnSetMeta()]).
#' @param x1,x2 network inputs.
#' @return numeric vector.
#' @export
nn1 <- function(id, x1) {
  d <- data.frame(id = id, x1 = x1)
  .Call(`_rxode2nn_nn1`, as.double(d$id), as.double(d$x1))
}
#' @rdname nn1
#' @export
nn1_d1 <- function(id, x1) {
  d <- data.frame(id = id, x1 = x1)
  .Call(`_rxode2nn_nn1_d1`, as.double(d$id), as.double(d$x1))
}
#' @rdname nn1
#' @export
nn1_d1_d1 <- function(id, x1) {
  d <- data.frame(id = id, x1 = x1)
  .Call(`_rxode2nn_nn1_d1_d1`, as.double(d$id), as.double(d$x1))
}
#' @rdname nn1
#' @export
nn2 <- function(id, x1, x2) {
  d <- data.frame(id = id, x1 = x1, x2 = x2)
  .Call(`_rxode2nn_nn2`, as.double(d$id), as.double(d$x1), as.double(d$x2))
}
#' @rdname nn1
#' @export
nn2_d1 <- function(id, x1, x2) {
  d <- data.frame(id = id, x1 = x1, x2 = x2)
  .Call(`_rxode2nn_nn2_d1`, as.double(d$id), as.double(d$x1), as.double(d$x2))
}
#' @rdname nn1
#' @export
nn2_d2 <- function(id, x1, x2) {
  d <- data.frame(id = id, x1 = x1, x2 = x2)
  .Call(`_rxode2nn_nn2_d2`, as.double(d$id), as.double(d$x1), as.double(d$x2))
}
#' @rdname nn1
#' @export
nn2_d1_d1 <- function(id, x1, x2) {
  d <- data.frame(id = id, x1 = x1, x2 = x2)
  .Call(`_rxode2nn_nn2_d1_d1`, as.double(d$id), as.double(d$x1), as.double(d$x2))
}
#' @rdname nn1
#' @export
nn2_d1_d2 <- function(id, x1, x2) {
  d <- data.frame(id = id, x1 = x1, x2 = x2)
  .Call(`_rxode2nn_nn2_d1_d2`, as.double(d$id), as.double(d$x1), as.double(d$x2))
}
#' @rdname nn1
#' @export
nn2_d2_d2 <- function(id, x1, x2) {
  d <- data.frame(id = id, x1 = x1, x2 = x2)
  .Call(`_rxode2nn_nn2_d2_d2`, as.double(d$id), as.double(d$x1), as.double(d$x2))
}

#' Register a network's weight-block layout for solving
#'
#' The single-hidden-layer MLP `nn<K>(id, ...)` reads its weights from a
#' contiguous block of the solve parameter vector.  This records, for network
#' `id`, where that block starts (`base`, zero-based index into
#' `rxode2::rxModelVars(model)$params`) and the network dimensions.
#'
#' @param id integer network id (0-based, matches the first `nn<K>()` argument).
#' @param base zero-based index of the first weight (`W1[0]`) in the model's
#'   parameter order.
#' @param K input dimension.
#' @param H hidden width.
#' @param act activation: one of "relu", "softplus", "tanh".
#' @return invisibly TRUE.
#' @export
nnSetMeta <- function(id, base, K, H, act = "relu") {
  act <- match.arg(tolower(act), names(.nnActCode))
  invisible(.Call(`_rxode2nn_nnSetMeta`, as.integer(id), as.integer(base),
                  as.integer(K), as.integer(H), .nnActCode[[act]]))
}

#' Clear all registered network layouts
#' @return invisibly TRUE.
#' @export
nnClearMeta <- function() {
  invisible(.Call(`_rxode2nn_nnClearMeta`))
}

#' Set a network's externally-owned weight buffer
#'
#' Stores the weights (in `nnWeightLayout()` order) that the rxode2 par-loader
#' hook injects into the reserved `par_ptr` block on every solve.  This is how
#' torch-trained weights reach the solve without being nlmixr2 parameters.
#'
#' @param id integer network id.
#' @param values numeric weight vector of length `H*K + 2*H + 1`.
#' @return invisibly TRUE.
#' @export
nnSetWeights <- function(id, values) {
  invisible(.Call(`_rxode2nn_nnSetWeights`, as.integer(id), as.double(values)))
}

#' Weight-block layout for a single-hidden-layer MLP
#'
#' Returns the ordered weight names for a `K`-input, `H`-hidden network, matching
#' the contiguous layout `nn<K>()` expects: `W1` (H*K, row-major), `b1` (H),
#' `W2` (H), `b2` (1).
#'
#' @param id network id used to prefix names.
#' @param K input dimension.
#' @param H hidden width.
#' @return character vector of length `H*K + 2*H + 1`.
#' @export
nnWeightLayout <- function(id, K, H) {
  w1 <- as.vector(t(outer(seq_len(H), seq_len(K),
                          function(j, k) sprintf("nnW1_%d_%d_%d", id, j, k))))
  b1 <- sprintf("nnB1_%d_%d", id, seq_len(H))
  w2 <- sprintf("nnW2_%d_%d", id, seq_len(H))
  b2 <- sprintf("nnB2_%d", id)
  c(w1, b1, w2, b2)
}
