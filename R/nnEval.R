## Activation code mapping shared with src/nnEval.c and MLPImpl in nnTorch.cpp
.nnActCode <- c(relu = 0L, softplus = 1L, tanh = 2L, gelu = 3L, silu = 4L)

## Dispatcher for the generated nn<K>* R wrappers (R/nnGen.R).  Vectorizes over
## the input columns; kind 0 = forward, 1 = gradient(j), 2 = Hessian(j,l).  The
## generated wrappers exist so rxode2's symengine->rxode2 renderer can resolve
## each model/derivative name by get()-ing an R function of the right arity (as
## with the mm example package), and to allow direct R evaluation.
.nnEvalR <- function(id, kind, j, l, ...) {
  m <- cbind(...)
  storage.mode(m) <- "double"
  id <- as.integer(id[1]); kind <- as.integer(kind)
  j <- as.integer(j); l <- as.integer(l)
  vapply(seq_len(nrow(m)),
         function(i) .Call(`_nlmixr2nn_nnEval`, id, m[i, ], kind, j, l),
         numeric(1))
}

## Dispatcher for the generated nnWg<K>(id, j, x...) wrappers: element j of the
## weight gradient d(out)/d(w) at each input row, reading the registered weights.
## Returns NA outside an active solve (no par_ptr); the in-solve path uses the C
## nnWg<K> entry points.  Mainly present so rxode2's renderer can resolve nnWg<K>.
.nnWgEvalR <- function(id, j, ...) {
  m <- cbind(...)
  storage.mode(m) <- "double"
  id <- as.integer(id[1]); j <- as.integer(j)
  vapply(seq_len(nrow(m)), function(i) {
    g <- .Call(`_nlmixr2nn_nnWeightGrad`, id, m[i, ])
    if (length(g) > j) g[j + 1L] else NA_real_
  }, numeric(1))
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
  invisible(.Call(`_nlmixr2nn_nnSetMeta`, as.integer(id), as.integer(base),
                  as.integer(K), as.integer(H), .nnActCode[[act]]))
}

#' Mark a network as individual (pop = FALSE): per-subject weights via weight etas
#'
#' The network's weights become individual, `W_i = f(lW, etaW)`, where `lW` are
#' the population (torch) weights and `etaW` are `nW` random effects starting at
#' `etaBase` in the model's eta vector.  FOCEI's inner weight-injection hook
#' applies this each time a subject's etas are set.
#'
#' @param id network id.
#' @param etaBase 0-based index of the first weight-eta in the model's eta vector.
#' @param etaModel "prop" (`W = lW * exp(etaW)`) or "add" (`W = lW + etaW`).
#' @return invisibly TRUE.
#' @export
nnSetIndividual <- function(id, etaBase, etaModel = c("prop", "add")) {
  etaModel <- match.arg(etaModel)
  invisible(.Call(`_nlmixr2nn_nnSetIndividual`, as.integer(id), as.integer(etaBase),
                  as.integer(etaModel == "add")))
}

#' Clear all registered network layouts
#' @return invisibly TRUE.
#' @export
nnClearMeta <- function() {
  invisible(.Call(`_nlmixr2nn_nnClearMeta`))
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
  invisible(.Call(`_nlmixr2nn_nnSetWeights`, as.integer(id), as.double(values)))
}

#' Analytic gradient of a network's output w.r.t. every weight
#'
#' Returns d(output)/d(w) in `nnWeightLayout()` order for the registered network
#' `id` at input `x`, reading the weights from the loader buffer.  Computed in
#' plain C (thread-safe); this is the `d(g)/d(w)` forcing factor for the
#' forward-sensitivity variational states of the NN weights.
#'
#' @param id integer network id (must be registered via [nnSetMeta()]).
#' @param x numeric input vector of length K.
#' @return numeric vector of length `H*K + 2*H + 1`.
#' @export
nnWeightGrad <- function(id, x) {
  .Call(`_nlmixr2nn_nnWeightGrad`, as.integer(id), as.double(x))
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
  ## rx-prefixed so users recognize them as internal (loader/hook-owned) covariates
  w1 <- as.vector(t(outer(seq_len(H), seq_len(K),
                          function(j, k) sprintf("rxnnW1_%d_%d_%d", id, j, k))))
  b1 <- sprintf("rxnnB1_%d_%d", id, seq_len(H))
  w2 <- sprintf("rxnnW2_%d_%d", id, seq_len(H))
  b2 <- sprintf("rxnnB2_%d", id)
  c(w1, b1, w2, b2)
}
