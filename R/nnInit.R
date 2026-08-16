## Parse-time weight initialization.
##
## The network's initial weights are drawn HERE, in R, when `nn()` is parsed --
## not by libtorch.  Three consequences, all of them the point:
##
##  * `set.seed()` works.  The same script gives the same network in any session,
##    with or without libtorch loaded, because R's RNG is the only source of
##    randomness.  Previously the only generator was libtorch's *global* RNG, so
##    initialization was reproducible only by accident.
##  * A parsed model is immediately solvable.  The values ride on the ui (see
##    `.nnAdopt()`), so `rxSolve()` needs no torch module and no setup call.
##  * torch is *loaded from* these values rather than generating its own, which
##    is what makes a fit reproducible under a single `set.seed()`.

## Activation gain for a fan-in-scaled normal initialization.
##
## He (gain sqrt(2)) for the one-sided, ReLU-like activations, whose expected
## derivative over a symmetric pre-activation is about 1/2; Glorot (gain 1) for
## tanh, whose derivative at 0 is 1.  A flat sd (the package's previous
## documented-but-dead `sd = 0.1`) is wrong at both ends: it saturates tanh and
## explodes ReLU for large K, and starves the hidden layer for small K.
.nnInitGain <- function(act) {
  if (identical(act, "tanh")) 1.0 else sqrt(2.0)
}

## Draw one network's weights in `nnWeightLayout()` order: W1 (H*K, row-major),
## b1 (H), W2 (H), b2 (1).
##
## `H` is accepted as a VECTOR so that adding a second hidden layer later is an
## additive change here rather than a redesign: `fan_in` walks `c(K, H[-last])`
## and only the final (output) layer gets the output scaling.  `nn()` currently
## rejects a vector `H` at its own boundary, so only the single-layer path is
## reachable today.
##
## `init`:
##   "ude"    - fan-in scaled, zero biases, and a SHRUNK output layer.  The
##              untrained network is then a small perturbation of the mechanistic
##              model it corrects, which is the standard UDE/neural-ODE practice
##              and keeps the initial solve from being stiff or wildly off.
##   "torch"  - libtorch's `nn::Linear::reset_parameters()`, i.e. Kaiming-uniform
##              U(+/- 1/sqrt(fan_in)) for weights AND biases.  For parity checks
##              against the torch module.
##   "normal" - flat N(0, initSd) everywhere; reproduces what the `sd=` argument
##              was documented (but never implemented) to do.
.nnInitDraw <- function(K, H, act = "softplus", init = "ude", initSd = 0.1) {
  K <- as.integer(K)
  H <- as.integer(H)
  nL <- length(H)
  gain <- .nnInitGain(act)
  fanIn <- c(K, H[-nL])
  out <- list()
  for (l in seq_len(nL)) {
    nIn <- fanIn[l]
    nOut <- H[l]
    out[[length(out) + 1L]] <- switch(
      init,
      ude = stats::rnorm(nOut * nIn, 0, gain / sqrt(nIn)),
      torch = stats::runif(nOut * nIn, -1 / sqrt(nIn), 1 / sqrt(nIn)),
      normal = stats::rnorm(nOut * nIn, 0, initSd)
    )
    out[[length(out) + 1L]] <- switch(
      init,
      ude = rep(0, nOut),
      torch = stats::runif(nOut, -1 / sqrt(nIn), 1 / sqrt(nIn)),
      normal = stats::rnorm(nOut, 0, initSd)
    )
  }
  ## output layer: 1 x H[nL] weights plus a scalar bias
  nIn <- H[nL]
  out[[length(out) + 1L]] <- switch(
    init,
    ## shrink the output so the untrained network barely moves the ODE
    ude = stats::rnorm(nIn, 0, initSd / sqrt(nIn)),
    torch = stats::runif(nIn, -1 / sqrt(nIn), 1 / sqrt(nIn)),
    normal = stats::rnorm(nIn, 0, initSd)
  )
  out[[length(out) + 1L]] <- switch(
    init,
    ude = 0,
    torch = stats::runif(1, -1 / sqrt(nIn), 1 / sqrt(nIn)),
    normal = stats::rnorm(1, 0, initSd)
  )
  unlist(out, use.names = FALSE)
}

## Draw weights WITHOUT perturbing the caller's random stream.
##
## `set.seed(1)` must fully determine a model's networks, yet building a model
## must not shift the draws in the rest of a user's script -- otherwise adding a
## network silently changes an unrelated simulation.  Both hold by deriving a
## base seed from the current stream, restoring the stream, and then drawing
## inside an isolated stream seeded with `base + id`.  The per-network offset is
## what keeps two networks in one model different despite each starting from the
## same restored state.
##
## An explicit `seed` pins one network regardless of the ambient seed.
.nnDrawSeeded <- function(id, K, H, act, init, initSd, seed = NULL) {
  if (!is.null(seed)) {
    .s <- as.integer(seed)
  } else {
    ## derive a base seed from the caller's stream, then put the stream back
    .s <- rxode2::rxWithPreserveSeed({
      if (!exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
        stats::runif(1)          # materialize a stream if none exists yet
      }
      as.integer(stats::runif(1, 1, .Machine$integer.max / 2))
    })
    .s <- .s + as.integer(id)
  }
  rxode2::rxWithSeed(.s, .nnInitDraw(K, H, act, init, initSd))
}
