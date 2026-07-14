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

## Parse a single `<out> = nn<K>(<id>, <in1>, <in2>, ...)` line from model text.
.nnParseCall <- function(modelText) {
  ## find the nn<K>(id, in1, ...) call ANYWHERE (it may be nested inside another
  ## expression, e.g. `cl <- exp(nn2(0, WT, eta.nn))`), not only as a bare
  ## assignment.  Inputs are simple names, so the call args contain no inner
  ## parens.  nnWg<K> is not matched (a letter follows `nn`).
  .txt <- paste(modelText, collapse = "\n")
  .re <- "\\bnn([0-9]+)\\s*\\(([^()]*)\\)"
  .hits <- regmatches(.txt, gregexpr(.re, .txt, perl = TRUE))[[1]]
  if (length(.hits) == 0L) stop("no `nn<K>(id, ...)` call found in model", call. = FALSE)
  if (length(unique(.hits)) > 1L) stop("multiple nn() calls not yet supported", call. = FALSE)
  .mm <- regmatches(.hits[[1L]], regexec(.re, .hits[[1L]], perl = TRUE))[[1]]
  .args <- trimws(unlist(strsplit(.mm[[3L]], ",", fixed = TRUE)))
  list(fn = paste0("nn", .mm[[2L]]), K = as.integer(.mm[[2L]]),
       id = as.integer(.args[[1L]]), inputs = .args[-1L])
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

#' Build the forward-sensitivity augmented NN model text
#'
#' @param modelText rxode2 model text containing one `g = nn<K>(id, ...)` output
#'   used in the ODE RHS.
#' @param H hidden width of network `id` (weight count nW = H*K + 2H + 1).
#' @return augmented model text (character scalar): the base model, `rx_drdg_*`
#'   outputs (dR/dg per state), and the `rx_sw_<state>_<j>_` variational states
#'   whose RHS is the F_X.s block (forcing added by the dydt-force hook).
#' @export
nnAugmentModel <- function(modelText, H) {
  .call <- .nnParseCall(modelText)
  .K <- .call$K; .nW <- H * .K + 2L * H + 1L
  .model <- rxode2::rxS(rxode2::rxGetModel(modelText), TRUE, promoteLinSens = FALSE)
  .st <- rxode2::rxStateOde(.model); .ns <- length(.st)
  invisible(rxode2::.rxJacobian(.model, .st))       ## F_X only
  .fx <- function(i, k) {
    .d <- get0(paste0("rx__df_", .st[i], "_dy_", .st[k], "__"), envir = .model, inherits = FALSE)
    if (is.null(.d)) "0" else rxode2::rxFromSE(.d)
  }
  .drdg <- .nnDrDg(.model, .st, .call)
  .sw <- function(i, j) sprintf("rx_sw_%s_%d_", .st[i], j)
  .out <- unlist(strsplit(trimws(modelText), "\n", fixed = TRUE))
  ## dR/dg outputs (read by the forcing hook)
  for (i in seq_len(.ns)) .out <- c(.out, sprintf("rx_drdg_%s_ = %s", .st[i], .drdg[[i]]))
  ## variational states: d/dt(s_ij) = sum_k F_X[i,k] s_kj + (dR_i/dg)(dg/dw_j)
  ## where the forcing factor dg/dw_j = nnWg<K>(id, j, inputs) reads the live
  ## (injected) weights at the current nn input; rx_drdg_<i>_ = dR_i/dg (above).
  .ins <- paste(.call$inputs, collapse = ", ")
  for (j in seq_len(.nW) - 1L) {
    for (i in seq_len(.ns)) {
      .terms <- character(0)
      for (k in seq_len(.ns)) {
        .f <- .fx(i, k)
        if (!identical(.f, "0") && nzchar(.f))
          .terms <- c(.terms, sprintf("(%s)*%s", .f, .sw(k, j)))
      }
      .fxs <- if (length(.terms)) paste(.terms, collapse = " + ") else "0"
      .forcing <- sprintf("rx_drdg_%s_*nnWg%d(%d, %d, %s)", .st[i], .K, .call$id, j, .ins)
      .rhs <- if (identical(.fxs, "0")) .forcing else paste(.fxs, "+", .forcing)
      .out <- c(.out, sprintf("d/dt(%s) = %s", .sw(i, j), .rhs))
    }
  }
  paste(.out, collapse = "\n")
}
