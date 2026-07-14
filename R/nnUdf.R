## Phase 2: user-facing `nn()` code-generating function.
##
## `nn(state, ..., n_hidden=, act=)` inside an rxode2/nlmixr2 model expands, at
## parse time, to a single compiled call `nn<K>(id, in1, ..., inK)` plus a
## `param()` declaration of the network's weights and randomized initial
## estimates appended to the model `iniDf`.  Unlike pmxNODE's `NN()` (which
## expands the network algebra weight-by-weight), the network stays opaque, so
## the model text is O(1) in the network size.  Lower-case `nn` deliberately
## distinguishes this from pmxNODE's upper-case `NN`.
##
## The weight values live in the model parameter vector as population thetas and
## are read from `par_ptr` at solve time; `nnUpdate()` resolves each network's
## weight-block base index from the assembled parameter order and registers it
## with the compiled layer via [nnSetMeta()].

## `.nnEnv` (network + torch-module registry) is defined in aaa.R.

#' @export
rxUdfUi.nn <- function(fun) {
  eval(fun)
}

.nnActs <- c("relu", "softplus", "tanh")

#' Neural-network term for an rxode2/nlmixr2 model
#'
#' Use `nn(state, ..., n_hidden=, act=)` inside a model to insert a
#' single-hidden-layer neural network of the state input(s).  At parse time it
#' is replaced by a compiled `nn<K>()` call and its weights are added to the
#' model as randomly-initialized population parameters.
#'
#' Inter-individual variability (a "deepNLME"/DeepPumas-style latent random
#' effect) is best added by passing a per-subject latent eta as an INPUT to the
#' population network, e.g. `g <- nn(central, eta.latent)` with
#' `eta.latent ~ omega` in the `ini` block.  Because the eta is then an ordinary
#' (visible) nlmixr2 random effect, its inner FOCEi sensitivity `d(g)/d(eta)` is
#' the network's analytic input derivative -- exact, no finite differences -- and
#' it is far more identifiable than a random effect on every weight.
#'
#' @param ... one or more state/covariate inputs to the network (given
#'   positionally, e.g. `nn(central, t)`).
#' @param n_hidden hidden-layer width (default 5).
#' @param act activation, one of "relu", "softplus", "tanh".
#' @param sd standard deviation of the random weight initialization.
#' @param num network occurrence number; queried via `rxUdfUiNum()` if `NULL`.
#' @param iniDf initial-estimate data.frame; queried via `rxUdfUiIniDf()` if
#'   `NULL`.
#' @return a list consumed by [rxode2::rxUdfUi()] (`replace`, `before`, `iniDf`).
#' @export
nn <- function(..., n_hidden = 5L,
               act = c("relu", "softplus", "tanh", "gelu", "silu"),
               sd = 0.1, num = NULL, iniDf = NULL) {
  act <- match.arg(act)
  ## capture positional inputs symbolically (do NOT evaluate them)
  .dots <- as.list(substitute(list(...)))[-1L]
  ## drop any named options accidentally caught in ... (n_hidden/act/sd)
  .nm <- names(.dots)
  if (!is.null(.nm)) .dots <- .dots[.nm == "" | is.na(.nm)]
  .inputs <- vapply(.dots, function(e) deparse1(e), character(1))
  K <- length(.inputs)
  if (K < 1L) stop("nn() needs at least one state input", call. = FALSE)
  if (K > 2L) {
    stop("nn() currently supports 1 or 2 inputs (nn1/nn2); more coming",
         call. = FALSE)
  }
  H <- as.integer(n_hidden)
  checkmate::assertIntegerish(H, lower = 1L, len = 1L, .var.name = "n_hidden")

  if (is.null(num)) num <- rxUdfUiNum()
  id <- num - 1L                       # 0-based id used by the compiled layer
  if (is.null(iniDf)) iniDf <- rxUdfUiIniDf()

  wnames <- nnWeightLayout(id, K, H)

  ## Declare the weight block as COVARIATES (not param()/thetas).  Reason: the
  ## torch/loader owns the weights (they are not nlmixr2 parameters), and -- unlike
  ## param()-declared thetas, which FOCEI theta-expands and mangles at model
  ## assembly -- covariates are contiguous in par_ptr, are not theta-expanded, and
  ## do not enter mu-referencing.  A single dummy reference makes them detected as
  ## covariates; their placeholder values are added to the data by nnCovData() and
  ## are overwritten by the par-loader (population) / inner hook (individual) on
  ## every solve.
  .before <- paste0("rx_nnw", id, "_ <- ", paste(wnames, collapse = " + "))

  ## record layout so nnUpdate() resolves the base index and nnCovData() adds cols
  .nnEnv$reg[[as.character(id)]] <-
    list(id = id, K = K, H = H, act = act, weights = wnames)

  .replace <- paste0("nn", K, "(", id, ",", paste(.inputs, collapse = ","), ")")
  list(replace = .replace, before = .before)
}
attr(rxUdfUi.nn, "nargs") <- NULL   # variadic

#' Add placeholder weight-covariate columns for an nn() model to data
#'
#' The `nn()` UDF declares each network's weights as covariates; this adds a
#' 0-valued column per weight to `data` so the model solves.  The par-loader
#' overwrites these slots with the network weights on every solve, so the
#' placeholder value is irrelevant.
#'
#' @param data a data.frame with the estimation/simulation data.
#' @return `data` with any missing weight-covariate columns added (value 0).
#' @export
nnCovData <- function(data) {
  for (m in .nnEnv$reg) for (w in m$weights)
    if (is.null(data[[w]])) data[[w]] <- 0
  data
}

#' Register generated networks' weight-block layout for solving
#'
#' Resolves each `nn()`-generated network's weight-block base index from an
#' assembled model's solve (`par_ptr`) parameter order, registers it with the
#' compiled layer via [nnSetMeta()], and seeds the loader weight buffer with the
#' model's current weight values via [nnSetWeights()].  The rxode2 par-loader
#' hook then injects those weights into `par_ptr` on every solve.
#'
#' The solve order differs from `rxode2::rxModelVars(ui)$params` for `ui`
#' models: `par_ptr` places population parameters first in theta (`ntheta`)
#' order, then covariates.  This uses the model `iniDf` theta order when
#' available, and otherwise the declared parameter order (raw rxode2 models).
#'
#' @param x an rxode2 model / ui object, or a character vector of parameter
#'   names in solve order.
#' @return invisibly, a data.frame of the registered networks.
#' @export
nnUpdate <- function(x) {
  reg <- .nnEnv$reg
  if (length(reg) == 0L) return(invisible(data.frame()))
  params <- .nnSolveParams(x)
  iniEst <- .nnIniEst(x)
  info <- lapply(reg, function(m) {
    idx <- match(m$weights, params)
    if (anyNA(idx)) {
      stop("nn() weights for network ", m$id,
           " not found in model parameters; was the model assembled with nn()?",
           call. = FALSE)
    }
    if (any(diff(idx) != 1L)) {
      stop("nn() weights for network ", m$id, " are not contiguous in par_ptr",
           call. = FALSE)
    }
    base <- idx[1] - 1L
    nnSetMeta(m$id, base, m$K, m$H, m$act)
    ## weights source: an attached torch module (preferred) else the ini seed
    if (m$id %in% .nnEnv$torchIds) {
      nnTorchSync(m$id)
    } else if (!is.null(iniEst)) {
      vals <- iniEst[m$weights]
      if (!anyNA(vals)) nnSetWeights(m$id, unname(vals))
    }
    data.frame(id = m$id, base = base, K = m$K, H = m$H, act = m$act)
  })
  invisible(do.call(rbind, info))
}

## parameter names in solve (par_ptr) order
.nnSolveParams <- function(x) {
  if (is.character(x)) return(x)
  ini <- .nnIniDf(x)
  if (!is.null(ini)) {
    th <- ini[!is.na(ini$ntheta), , drop = FALSE]
    th <- th[order(th$ntheta), , drop = FALSE]
    mv <- rxode2::rxModelVars(x)
    covs <- setdiff(mv$params, th$name)   # covariates follow the thetas
    return(c(th$name, covs))
  }
  rxode2::rxModelVars(x)$params
}

## current weight values keyed by name (from the model iniDf), or NULL
.nnIniEst <- function(x) {
  ini <- .nnIniDf(x)
  if (is.null(ini)) return(NULL)
  stats::setNames(ini$est, ini$name)
}

.nnIniDf <- function(x) {
  ini <- try(x$iniDf, silent = TRUE)
  if (inherits(ini, "try-error") || !is.data.frame(ini)) return(NULL)
  ini
}
