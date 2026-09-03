## Parse-time weight initialization.
##
## The network's initial weights are drawn HERE, when `nn()` is parsed -- not by
## libtorch -- from rxode2's own threefry generator (`rxnorm`/`rxunif`) rather
## than R's.  Four consequences, all of them the point:
##
##  * `rxSetSeed()` works, and it is the seed that matters.  Threefry is the
##    generator the rest of the ecosystem draws from, and it is the one whose
##    stream is defined under multi-threading -- so a model initialized here is
##    reproducible in the same places a parallel rxode2 solve is.  A bare
##    `set.seed()` still pins a model (it is the fallback when `rxSetSeed()` has
##    not been called), but it is not the seed to reach for.
##  * Nothing is disturbed.  Every draw happens inside `rxWithSeed()`, which
##    saves and restores BOTH the R stream and the rxode2 seed, so building a
##    model never shifts the draws in the rest of a user's script.
##  * A parsed model is immediately solvable.  The values ride on the ui (see
##    `.nnAdopt()`), so `rxSolve()` needs no torch module and no setup call.
##  * torch is *loaded from* these values rather than generating its own, which
##    is what makes a whole fit reproducible from one seed.  Previously the only
##    generator was libtorch's *global* RNG, so initialization was reproducible
##    only by accident.

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
      ude = rxode2::rxnorm(0, gain / sqrt(nIn), nOut * nIn),
      torch = rxode2::rxunif(-1 / sqrt(nIn), 1 / sqrt(nIn), nOut * nIn),
      normal = rxode2::rxnorm(0, initSd, nOut * nIn)
    )
    out[[length(out) + 1L]] <- switch(
      init,
      ude = rep(0, nOut),
      torch = rxode2::rxunif(-1 / sqrt(nIn), 1 / sqrt(nIn), nOut),
      normal = rxode2::rxnorm(0, initSd, nOut)
    )
  }
  ## output layer: 1 x H[nL] weights plus a scalar bias
  nIn <- H[nL]
  out[[length(out) + 1L]] <- switch(
    init,
    ## shrink the output so the untrained network barely moves the ODE
    ude = rxode2::rxnorm(0, initSd / sqrt(nIn), nIn),
    torch = rxode2::rxunif(-1 / sqrt(nIn), 1 / sqrt(nIn), nIn),
    normal = rxode2::rxnorm(0, initSd, nIn)
  )
  out[[length(out) + 1L]] <- switch(
    init,
    ude = 0,
    torch = rxode2::rxunif(-1 / sqrt(nIn), 1 / sqrt(nIn), 1L),
    normal = rxode2::rxnorm(0, initSd, 1L)
  )
  unlist(out, use.names = FALSE)
}

## Draw weights WITHOUT perturbing any random stream.
##
## Two things must hold at once.  A seed has to fully determine a model's
## networks, and building a model must NOT shift the draws in the rest of a
## user's script -- otherwise adding a network silently changes an unrelated
## simulation.  `rxode2::rxWithSeed()` gives both: it saves the R stream AND the
## rxode2 threefry seed, sets them for the draw, and restores both on exit.  It
## is the same mechanism nlmixr2est uses everywhere else, for the same reason.
##
## The generator is rxode2's, not R's.  `.nnInitDraw()` draws with
## `rxnorm`/`rxunif`, so the stream is threefry -- the one the rest of the
## ecosystem uses and the one whose behaviour under multiple threads is defined.
## A model initialized here is therefore reproducible in the same places a
## parallel solve is, which R's RNG cannot promise.  `rxSetSeed()` is the seed
## that matters.
##
## `set.seed()` is a fallback rather than a dead end: `rxGetSeed()` is -1 until
## `rxSetSeed()` is called, and the base seed is then derived from R's stream
## (without advancing it), so a script that only calls `set.seed()` still pins
## its model.
##
## The per-network `+ id` offset is what keeps two networks in one model
## different despite starting from the same base -- including the augmented
## networks a single `nn(aug=)` creates.  It applies to an explicit `seed` too:
## without that, `nn(seed = 5)` gave every network of the model the identical
## weight vector, which for augmented networks means k copies of one function.
.nnDrawSeeded <- function(id, K, H, act, init, initSd, seed = NULL) {
  .s <- .nnBaseSeed(seed) + as.integer(id)
  rxode2::rxWithSeed(.s, .nnInitDraw(K, H, act, init, initSd), rxseed = .s)
}

## The base seed: an explicit one, else rxode2's, else derived from R's stream
## without advancing it.
.nnBaseSeed <- function(seed = NULL) {
  if (!is.null(seed)) return(as.integer(seed))
  .rs <- tryCatch(as.integer(rxode2::rxGetSeed()), error = function(e) NA_integer_)
  if (!is.na(.rs) && .rs >= 0L) return(.rs)
  rxode2::rxWithPreserveSeed({
    if (!exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      stats::runif(1)          # materialize a stream if none exists yet
    }
    as.integer(stats::runif(1, 1, .Machine$integer.max / 2))
  })
}
