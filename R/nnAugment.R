## Forward-sensitivity augmented model for NN weights (design B, phase B3).
##
## For an NN output g = nn<K>(id, x...) feeding the ODE RHS, the sensitivity of
## each state wrt each network weight w_j is a variational state
##   s_ij = d(state_i)/d(w_j),  d/dt(s_ij) = sum_k F_X[i,k] s_kj + b_ij,
## where F_X = dR_i/dstate_k is the model Jacobian (rxode2 forms it, chaining the
## registered nn<K>_d* input-gradient functions) and the forcing
##   b_ij = (dR_i/dg)(dg/dw_j)
## is added at solve time (dR_i/dg emitted here as rx_drdg_<state>_ outputs;
## dg/dw_j = nnWeightGrad, supplied by the dydt-force hook -- see nnDydtForce.c).
##
## This builds the augmented rxode2 model TEXT: base model + rx_drdg_ outputs +
## the F_X.s variational-state block.  The forcing term is left to the hook.

## Parse ALL distinct `nn<K>(id, ...)` calls from model text, ordered by id (a
## stable global weight layout).  Each entry is list(fn, K, id, inputs).
.nnParseCallAll <- function(modelText) {
  .txt <- paste(modelText, collapse = "\n")
  .re <- "\\bnn([0-9]+)\\s*\\(([^()]*)\\)"
  .hits <- unique(regmatches(.txt, gregexpr(.re, .txt, perl = TRUE))[[1]])
  if (length(.hits) == 0L) stop("no `nn<K>(id, ...)` call found in model", call. = FALSE)
  .calls <- lapply(.hits, function(h) {
    .mm <- regmatches(h, regexec(.re, h, perl = TRUE))[[1]]
    .args <- trimws(unlist(strsplit(.mm[[3L]], ",", fixed = TRUE)))
    list(fn = paste0("nn", .mm[[2L]]), K = as.integer(.mm[[2L]]),
         id = as.integer(.args[[1L]]), inputs = .args[-1L])
  })
  .calls[order(vapply(.calls, function(.c) .c$id, integer(1)))]
}

## dR_i/dg per state: substitute the exact nn call expression with a fresh symbol
## then differentiate.  The id argument renders as a double in symengine, so the
## substituted expression must use `<id>.0` to match.
.nnDrDg <- function(model, states, call) {
  .G <- symengine::Symbol("rx__nnG__")
  .nnStr <- sprintf("%s(%d.0, %s)", call$fn, call$id, paste(call$inputs, collapse = ", "))
  .nnExpr <- symengine::S(.nnStr)
  vapply(states, function(s) {
    .rhs <- get0(paste0("rx__d_dt_", s, "__"), envir = model, inherits = FALSE)
    if (is.null(.rhs)) return("0")
    ## bind to a local before rxFromSE (NSE gotcha: inline `symengine::D(...)` is
    ## otherwise mis-parsed as a model function call)
    .dd <- symengine::D(symengine::subs(.rhs, .nnExpr, .G), .G)
    ## substitute the placeholder back to the actual nn() call: when the nn output
    ## passes through a nonlinear transform, dR/dg depends on g, so the derivative
    ## still contains rx__nnG__ -- replace it with the real call so the emitted
    ## rx_drdg_ expression is self-contained (rx__nnG__ is not a model variable).
    .dd <- symengine::subs(.dd, .G, .nnExpr)
    rxode2::rxFromSE(.dd)
  }, character(1))
}

## hidden width of a network by id, from the augment metadata list
.hOfNet <- function(nets, id) {
  for (.m in nets) if (.m$id == id) return(as.integer(.m$H))
  NA_integer_
}

## d(var)/dg for an arbitrary model variable, by the same substitution as
## .nnDrDg: replace the nn call with a symbol, differentiate, put it back.
##
## This is the DIRECT dependence of a quantity on the network output, as opposed
## to its dependence through the ODE states.  It is zero for the usual model,
## where the network appears only in a d/dt() -- which is why it was missing and
## nothing noticed.  It is NOT zero when the network feeds the prediction (or a
## distribution's parameter) itself, e.g. `y <- nn(centr)`, and there the forward
## sensitivity through the states carries none of the effect: the assembled
## gradient came out exactly zero and such a model trained not at all.
.nnDvarDg <- function(model, varName, call) {
  .G <- symengine::Symbol("rx__nnG__")
  .nnStr <- sprintf("%s(%d.0, %s)", call$fn, call$id, paste(call$inputs, collapse = ", "))
  .nnExpr <- symengine::S(.nnStr)
  .rhs <- get0(varName, envir = model, inherits = FALSE)
  if (is.null(.rhs)) return("0")
  .dd <- symengine::D(symengine::subs(.rhs, .nnExpr, .G), .G)
  .dd <- symengine::subs(.dd, .G, .nnExpr)
  rxode2::rxFromSE(.dd)
}

#' Build the forward-sensitivity augmented NN model text
#'
#' Supports one or more `nn<K>(id, ...)` outputs used in the ODE RHS.  The weight
#' variational states are indexed by a GLOBAL weight index across all networks
#' (network 0's weights first, then network 1's, ...), so a single-network model
#' keeps the original `rx_sw_<state>_<j>_` layout; `rx_drdg_<state>_` gains a
#' `<id>_` suffix only when more than one network is present.
#'
#' @param modelText rxode2 model text containing the `nn<K>(id, ...)` output(s).
#' @param H hidden width: a scalar for a single network, or a vector named by
#'   network id (character) for multiple networks.
#' @return augmented model text (character scalar): the base model, `rx_drdg_*`
#'   outputs (dR/dg per state per network), and the `rx_sw_<state>_<globalj>_`
#'   variational states whose RHS is the F_X.s block plus the per-network forcing.
#' @keywords internal
nnAugmentModel <- function(modelText, H) {
  .calls <- .nnParseCallAll(modelText)
  .multi <- length(.calls) > 1L
  .model <- rxode2::rxS(rxode2::rxGetModel(modelText), TRUE, promoteLinSens = FALSE)
  .st <- rxode2::rxStateOde(.model)
  .ns <- length(.st)
  invisible(rxode2::.rxJacobian(.model, .st))       ## F_X only (shared by all nets)
  .fx <- function(i, k) {
    .d <- get0(paste0("rx__df_", .st[i], "_dy_", .st[k], "__"), envir = .model, inherits = FALSE)
    if (is.null(.d)) "0" else rxode2::rxFromSE(.d)
  }
  .hOf <- function(id) if (length(H) == 1L && is.null(names(H))) H else H[[as.character(id)]]
  .out <- unlist(strsplit(trimws(modelText), "\n", fixed = TRUE))
  .gj <- 0L                                          # running global weight index
  for (.call in .calls) {
    .K <- .call$K
    .Hn <- .hOf(.call$id)
    .nW <- .Hn * .K + 2L * .Hn + 1L
    .drdg <- .nnDrDg(.model, .st, .call)
    .suf <- if (.multi) sprintf("%d_", .call$id) else ""   # rx_drdg per-net suffix
    ## dR/dg outputs (read by the variational-state forcing)
    for (i in seq_len(.ns)) .out <- c(.out, sprintf("rx_drdg_%s_%s= %s", .st[i], .suf, .drdg[[i]]))
    ## variational states: d/dt(s_ij) = sum_k F_X[i,k] s_kj + (dR_i/dg)(dg/dw_j)
    ## where dg/dw_j = nnWg<K>(id, j, inputs) reads the live injected weights.
    .ins <- paste(.call$inputs, collapse = ", ")
    for (localj in seq_len(.nW) - 1L) {
      for (i in seq_len(.ns)) {
        .terms <- character(0)
        for (k in seq_len(.ns)) {
          .f <- .fx(i, k)
          if (!identical(.f, "0") && nzchar(.f))
            .terms <- c(.terms, sprintf("(%s)*rx_sw_%s_%d_", .f, .st[k], .gj + localj))
        }
        .fxs <- if (length(.terms)) paste(.terms, collapse = " + ") else "0"
        .forcing <- sprintf("rx_drdg_%s_%s*nnWg%d(%d, %d, %s)",
                            .st[i], .suf, .K, .call$id, localj, .ins)
        .rhs <- if (identical(.fxs, "0")) .forcing else paste(.fxs, "+", .forcing)
        .out <- c(.out, sprintf("d/dt(rx_sw_%s_%d_) = %s", .st[i], .gj + localj, .rhs))
      }
    }
    .gj <- .gj + .nW
  }
  paste(.out, collapse = "\n")
}
