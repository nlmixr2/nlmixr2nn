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
  .lines <- unlist(strsplit(modelText, "\n", fixed = TRUE))
  .re <- "^\\s*([A-Za-z._][A-Za-z0-9._]*)\\s*(?:=|<-)\\s*(nn([0-9]+))\\s*\\(([^)]*)\\)\\s*$"
  .m <- regmatches(.lines, regexec(.re, .lines, perl = TRUE))
  .hit <- Filter(function(x) length(x) == 5L, .m)
  if (length(.hit) == 0L) stop("no `out = nn<K>(id, ...)` call found in model", call. = FALSE)
  if (length(.hit) > 1L) stop("multiple nn() calls not yet supported", call. = FALSE)
  .x <- .hit[[1L]]
  .args <- trimws(unlist(strsplit(.x[[5L]], ",", fixed = TRUE)))
  list(out = .x[[2L]], fn = .x[[3L]], K = as.integer(.x[[4L]]),
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
  ## variational states: d/dt(s_ij) = sum_k F_X[i,k] s_kj  (+ hook forcing b_ij)
  for (j in seq_len(.nW) - 1L) {
    for (i in seq_len(.ns)) {
      .terms <- character(0)
      for (k in seq_len(.ns)) {
        .f <- .fx(i, k)
        if (!identical(.f, "0") && nzchar(.f))
          .terms <- c(.terms, sprintf("(%s)*%s", .f, .sw(k, j)))
      }
      .rhs <- if (length(.terms)) paste(.terms, collapse = " + ") else "0"
      .out <- c(.out, sprintf("d/dt(%s) = %s", .sw(i, j), .rhs))
    }
  }
  paste(.out, collapse = "\n")
}
