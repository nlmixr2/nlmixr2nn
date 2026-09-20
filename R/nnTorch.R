## R interface to the C++ libtorch backend (src/nnTorch.cpp).  The torch modules
## live entirely in C++ so they are reachable from compiled solve/objective code
## without calling back into R; these wrappers only drive setup, sync, and
## save/restore from the main thread.

## `.nnEnv$torchIds` (attached module ids) is initialized in aaa.R.

#' Create a C++ torch module for a network
#'
#' @param id network id.
#' @param K input dimension.
#' @param H hidden width.
#' @param act activation ("relu", "softplus", "tanh").
#' @param seed optional integer seed for weight initialization.
#' @return invisibly, the id.
#' @keywords internal
nnTorchInit <- function(id, K, H, act = "relu", seed = NULL) {
  act <- match.arg(tolower(act), names(.nnActCode))
  .Call(`_nlmixr2nn_nnTorchInit`, as.integer(id), as.integer(K), as.integer(H),
        .nnActCode[[act]], if (is.null(seed)) NULL else as.integer(seed))
  .nnEnv$torchIds <- union(.nnEnv$torchIds, as.integer(id))
  invisible(as.integer(id))
}

#' Push a torch module's weights into the loader buffer (C path, no R round-trip)
#' @param id network id.
#' @return invisibly, the number of weights synced.
#' @keywords internal
nnTorchSync <- function(id) invisible(.Call(`_nlmixr2nn_nnTorchSync`, as.integer(id)))

#' Flat weights of a torch module (nnWeightLayout order)
#' @param id network id.
#' @return numeric vector.
#' @keywords internal
nnTorchWeights <- function(id) .Call(`_nlmixr2nn_nnTorchGetWeights`, as.integer(id))

#' Load flat weights into a torch module
#' @param id network id.
#' @param values numeric weight vector.
#' @return invisibly NULL.
#' @keywords internal
nnTorchSetWeights <- function(id, values) {
  invisible(.Call(`_nlmixr2nn_nnTorchSetWeights`, as.integer(id), as.double(values)))
}

#' Forward pass of a torch module at one input (for validation)
#' @param id network id.
#' @param x numeric input vector of length K.
#' @return numeric scalar.
#' @keywords internal
nnTorchForward <- function(id, x) .Call(`_nlmixr2nn_nnTorchForward`, as.integer(id), as.double(x))

#' Free a torch module
#' @param id network id.
#' @return invisibly NULL.
#' @keywords internal
nnTorchFree <- function(id) {
  .Call(`_nlmixr2nn_nnTorchFree`, as.integer(id))
  .nnEnv$torchIds <- setdiff(.nnEnv$torchIds, as.integer(id))
  invisible()
}

#' Save / load a trained torch module to a binary
#'
#' @param id network id.
#' @param file path to the serialized module.
#' @return invisibly NULL.
#' @keywords internal
nnTorchSave <- function(id, file) invisible(.Call(`_nlmixr2nn_nnTorchSave`, as.integer(id), file))

#' @rdname nnTorchSave
#' @keywords internal
nnTorchLoad <- function(id, file) invisible(.Call(`_nlmixr2nn_nnTorchLoad`, as.integer(id), file))

.nnOptCode <- c(sgd = 0L, adam = 1L)

#' Create an optimizer for a torch module
#' @param id network id.
#' @param type "adam" or "sgd".
#' @param lr learning rate.
#' @return invisibly NULL.
#' @keywords internal
nnTorchOptInit <- function(id, type = "adam", lr = 0.05) {
  type <- match.arg(tolower(type), names(.nnOptCode))
  invisible(.Call(`_nlmixr2nn_nnTorchOptInit`, as.integer(id), .nnOptCode[[type]], as.double(lr)))
}

#' @rdname nnTorchOptInit
#' @keywords internal
nnTorchZeroGrad <- function(id) invisible(.Call(`_nlmixr2nn_nnTorchZeroGrad`, as.integer(id)))

#' Batch forward pass of a torch module
#' @param id network id.
#' @param X numeric matrix of inputs, `N` rows by `K` columns.
#' @return numeric vector of length `N`.
#' @keywords internal
nnTorchForwardBatch <- function(id, X) {
  X <- as.matrix(X)
  .Call(`_nlmixr2nn_nnTorchForwardBatch`, as.integer(id),
        as.double(t(X)), nrow(X), ncol(X))
}

#' Accumulate the vector-Jacobian product d(sum G*y)/d(w) into the grads
#' @param id network id.
#' @param X numeric matrix of inputs, `N` rows by `K` columns.
#' @param G numeric cotangent vector of length `N` (d(loss)/d(output)).
#' @return invisibly NULL.
#' @keywords internal
nnTorchBackward <- function(id, X, G) {
  X <- as.matrix(X)
  invisible(.Call(`_nlmixr2nn_nnTorchBackward`, as.integer(id),
                  as.double(t(X)), as.double(G), nrow(X), ncol(X)))
}

#' Flattened parameter gradients (nnWeightLayout order)
#' @param id network id.
#' @return numeric vector.
#' @keywords internal
nnTorchGetGrad <- function(id) .Call(`_nlmixr2nn_nnTorchGetGrad`, as.integer(id))

#' Set the network's parameter gradients from a flat vector (nnWeightLayout order)
#'
#' Injects an externally computed gradient (e.g. the analytic dLoss/dw from the
#' ODE forward sensitivity) into the torch parameters' `.grad`, to be applied by
#' the next [nnTorchStep()].
#'
#' @param id network id.
#' @param grad numeric gradient vector of length `H*K + 2*H + 1`.
#' @return invisibly NULL.
#' @keywords internal
nnTorchSetGrad <- function(id, grad) {
  invisible(.Call(`_nlmixr2nn_nnTorchSetGrad`, as.integer(id), as.double(grad)))
}

#' Optimizer step; also syncs updated weights into the loader buffer
#' @param id network id.
#' @return invisibly NULL.
#' @keywords internal
nnTorchStep <- function(id) invisible(.Call(`_nlmixr2nn_nnTorchStep`, as.integer(id)))

#' Train a torch module to a target with an L2 loss (standalone helper)
#'
#' Runs `steps` optimizer iterations minimizing `0.5*sum((y - target)^2)` over
#' the batch, using the cotangent `G = y - target` -> [nnTorchBackward()] ->
#' [nnTorchStep()].  This exercises the same VJP path the likelihood-contribution
#' gradient bridge will use (there the cotangent comes from the adjoint sweep).
#'
#' @param id network id.
#' @param X input matrix (`N` x `K`).
#' @param target numeric target vector (length `N`).
#' @param steps number of optimizer iterations.
#' @param lr learning rate.
#' @param type "adam" or "sgd".
#' @return numeric vector of the loss at each step.
#' @keywords internal
nnTorchTrain <- function(id, X, target, steps = 200L, lr = 0.05, type = "adam") {
  X <- as.matrix(X)
  target <- as.double(target)
  nnTorchOptInit(id, type, lr)
  loss <- numeric(steps)
  for (s in seq_len(steps)) {
    y <- nnTorchForwardBatch(id, X)
    g <- y - target                  # d(0.5*sum (y-target)^2)/d(y)
    loss[s] <- 0.5 * sum(g^2)
    nnTorchZeroGrad(id)
    nnTorchBackward(id, X, g)
    nnTorchStep(id)
  }
  loss
}

## Which training backend this build linked: "torch" or "builtin".
##
## configure picks one and compiles only that translation unit, so this is a
## property of the INSTALLED package, not of what is installed alongside it.
.nnBackend <- function() {
  tryCatch(.Call(`_nlmixr2nn_nnBackend`), error = function(e) NA_character_)
}

## Is the libtorch backend actually usable?
##
## The C probe alone answers "was this package built against libtorch", which is
## necessarily TRUE in a package that linked -- so on its own it is not a check
## at all.  Asking torch whether its binaries are present is the part that can
## really be false: install.packages("torch") does not fetch libtorch; that is a
## separate torch::install_torch().
.nnTorchAvailable <- function() {
  identical(.nnBackend(), "torch") &&
    isTRUE(tryCatch(requireNamespace("torch", quietly = TRUE) &&
                      torch::torch_is_installed(), error = function(e) FALSE)) &&
    isTRUE(tryCatch(.Call(`_nlmixr2nn_nnTorchAvailable`), error = function(e) FALSE))
}

## Can weights be trained at all?
##
## The builtin backend needs nothing installed -- it IS the package -- so this
## is only ever false for a libtorch build whose binaries have gone missing.
.nnTrainAvailable <- function() {
  identical(.nnBackend(), "builtin") || .nnTorchAvailable()
}

## The message a user gets when training cannot start: what to run, not just
## what is missing.
.nnTorchRequire <- function(what = "training a neural network") {
  if (.nnTrainAvailable()) return(invisible(TRUE))
  stop(what, " needs the libtorch backend this package was built against, ",
       "which is not available.\n",
       "  install.packages(\"torch\")\n",
       "  torch::install_torch()\n",
       "torch::install_torch_sitrep() reports what torch has installed.\n",
       "Re-installing nlmixr2nn without libtorch present builds the builtin ",
       "optimizer instead, which needs nothing.\n",
       "Parsing, solving and simulating a model with nn() need neither -- ",
       "only training does.", call. = FALSE)
}

#' Create C++ torch modules for all networks in a model
#'
#' For each network generated by [nn()] in an assembled model, creates a C++
#' torch module of the recorded shape.  Weights reach the solve via
#' [nnUpdate()], which syncs each attached module into the loader buffer.
#'
#' @param x an assembled rxode2 model / ui object built with [nn()].
#' @param seed optional integer seed for reproducible initialization.
#' @return invisibly, a data.frame of the created modules.
#' @keywords internal
nnTorchModel <- function(x, seed = NULL) {
  reg <- .nnEnv$reg
  if (length(reg) == 0L) {
    stop("no nn() networks registered; build a model with nn() first", call. = FALSE)
  }
  info <- lapply(reg, function(m) {
    nnTorchInit(m$id, m$K, m$H, m$act,
                seed = if (is.null(seed)) NULL else seed + m$id)
    data.frame(id = m$id, K = m$K, H = m$H, act = m$act)
  })
  invisible(do.call(rbind, info))
}
