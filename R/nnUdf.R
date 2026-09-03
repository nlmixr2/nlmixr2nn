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
#' The network's initial weights are drawn WHEN THE MODEL IS PARSED, using
#' rxode2's threefry generator, and are carried on the model itself.  So
#' [rxode2::rxSetSeed()] makes a network reproducible across sessions -- and,
#' because threefry is the generator whose stream is defined under multiple
#' threads, reproducible in the same places a parallel solve is.  A freshly
#' parsed model is immediately solvable: no torch module and no setup call.
#'
#' Drawing weights disturbs nothing.  Every draw runs inside
#' [rxode2::rxWithSeed()], which saves and restores both the R stream and the
#' rxode2 seed, so adding a network never shifts the draws elsewhere in a
#' script.  A bare `set.seed()` still pins a model -- it is the fallback used
#' when `rxSetSeed()` has not been called -- but `rxSetSeed()` is the one to
#' reach for.
#'
#' @param ... one or more state/covariate inputs to the network (given
#'   positionally, e.g. `nn(central, t)`).
#' @param n_hidden hidden-layer width (default 5).
#' @param act activation, one of "softplus" (default), "tanh", "relu", "gelu" or
#'   "silu".  The default is smooth with a nonzero second derivative, which
#'   matters because the model's Jacobian and the FOCEi sensitivities are both
#'   chained through it; "relu" has a kink and a zero second derivative.
#' @param init weight initialization: `"ude"` (default) draws fan-in scaled
#'   weights with zero biases and a shrunk output layer, so an untrained network
#'   is a small perturbation of the mechanistic model; `"torch"` reproduces
#'   libtorch's `nn::Linear` default; `"normal"` is a flat `N(0, initSd)`.
#' @param initSd scale of the initialization: the output-layer standard
#'   deviation for `init = "ude"`, and the standard deviation of every weight
#'   for `init = "normal"`.
#' @param aug number of AUGMENTED states to add (0-3, default 0) -- the ANODE
#'   construction of Dupont et al. (2019).  **Not available yet: any value above
#'   0 is an error.**  Augmenting from inside `nn()` would renumber the model's
#'   compartments (the augmented state takes a `d/dt()` slot next to the `nn()`
#'   call, so `cmt` in the data stops meaning what it means in the model), and a
#'   UDF cannot see the model's states to prevent that.  Doing it at model
#'   assembly instead is the fix.  Write the augmentation by hand meanwhile --
#'   see `vignette("nlmixr2nn-node")`.  `aug = k` creates `k` extra
#'   compartments `rx_nnaug<id>_1..k`, starts them at zero, gives each its own
#'   learned derivative, and passes all of them to this network as additional
#'   inputs.  It is exactly what you would get by writing
#'
#'   ```
#'   d/dt(a1) <- nn(centr, a1, a2)
#'   d/dt(a2) <- nn(centr, a1, a2)
#'   g        <- nn(centr, a1, a2)
#'   ```
#'
#'   by hand, and exists because the thing it fixes is invisible: a learned flow
#'   in one dimension cannot cross itself, so `d/dt(centr) <- -f(nn(centr))`
#'   is topologically constrained no matter how well it is fitted or how much
#'   capacity it has.  Extra dimensions remove the constraint.
#'
#'   The cost is real and worth weighing.  The augmented solve carries one
#'   variational state per (state x weight), and `aug = k` multiplies BOTH: it
#'   adds `k` states and `k` more networks.  Start at 1.
#'
#'   With `init = "ude"` (the default) every augmented derivative starts near
#'   zero, so the augmented states stay near zero and contribute nothing until
#'   training finds a use for them -- the zero-augmentation start ANODE
#'   prescribes.  (The network itself is still a fresh draw over the wider input
#'   vector, so this is not the same model as `aug = 0`.)  Total inputs
#'   (`...` plus `aug`) may not exceed 4.
#'
#'   Each augmented state RELAXES rather than integrates:
#'   `d/dt(a) = f(x, a) - a`.  The textbook drift, `d/dt(a) = f(x, a)`, is
#'   self-exciting here -- `a` re-enters its own derivative through the network --
#'   and blows the solver up on a real dosing horizon.  The `- a` gives it a
#'   stable fixed point at `f`, at the price of a fixed memory timescale of one
#'   model time unit.
#' @param seed optional integer pinning THIS `nn()` call's initial weights
#'   regardless of the ambient seed.  Normally unnecessary -- use
#'   [rxode2::rxSetSeed()].  Every network is drawn at `seed + <network id>`, so
#'   the networks of an `aug =` call differ from each other rather than all
#'   drawing the same weights.
#' @param num network occurrence number; queried via `rxUdfUiNum()` if `NULL`.
#' @param iniDf initial-estimate data.frame; queried via `rxUdfUiIniDf()` if
#'   `NULL`.
#' @return a list consumed by [rxode2::rxUdfUi()] (`replace`, `before`).
#'
#' @references
#' Dupont E, Doucet A, Teh YW (2019). "Augmented neural ODEs." *Advances in
#' Neural Information Processing Systems* 32.  See
#' `vignette("nlmixr2nn-node")` for what else from that literature does and
#' does not carry over.
#' @export
nn <- function(..., n_hidden = 5L,
               act = c("softplus", "tanh", "relu", "gelu", "silu"),
               init = c("ude", "torch", "normal"), initSd = 0.1, aug = 0L,
               seed = NULL, num = NULL, iniDf = NULL) {
  act <- match.arg(act)
  init <- match.arg(init)
  checkmate::assertNumeric(initSd, lower = 0, len = 1L, .var.name = "initSd")
  ## upper = 3 is not a policy choice: an augmented network reads its own
  ## augmented states, so K = length(...) + aug, and the compiled family only
  ## goes to nn4.  With at least one real input, aug can never exceed 3.
  checkmate::assertIntegerish(aug, lower = 0L, upper = .nnAugMax, len = 1L,
                              .var.name = "aug")
  aug <- as.integer(aug)
  if (aug > 0L) {
    ## REFUSED, for a reason that cannot be fixed from inside this function.
    ##
    ## `rxUdfUi()` can only emit code around the line it is expanding (`before`
    ## / `after`), so an augmented `d/dt()` lands next to the user's nn() call
    ## rather than after every compartment they declared.  rxode2 numbers
    ## compartments by first `d/dt()` appearance, so the augmentation TAKES a
    ## compartment number from the model -- and `cmt = 1` in the data then doses
    ## the latent instead of the user's first compartment.  Measured: the dose
    ## landed in rx_nnaug0_1, `centr` stayed at 0 for the whole profile, and the
    ## fit "failed to train" because there was nothing for it to train on.
    ##
    ## Emitting the lines via `after` only narrows the window (it still breaks
    ## the common `g <- nn(centr, ...)` form, where the nn() line precedes every
    ## d/dt), and a UDF cannot see the model's states to pin them first --
    ## `rxUdfUiMv()` is NULL while parsing.
    ##
    ## The fix is to append the augmented compartments at MODEL-ASSEMBLY time,
    ## where the whole model is visible, rather than at parse time.  Everything
    ## else this function builds for `aug` is correct and stays: the leaky drift,
    ## the id allocation, and the multi-network forward sensitivity, which is
    ## finite-difference verified in test-nn-augsens.R.
    stop("nn(aug=) is not available yet.  Augmenting from here would renumber ",
         "your compartments -- the augmented state takes a d/dt() slot next to ",
         "the nn() call, so `cmt` in the data would no longer mean what it ",
         "means in your model.  Add the extra compartments by hand for now:\n",
         "    d/dt(a1) <- nn(centr, a1) - a1\n",
         "    g        <- nn(centr, a1)\n",
         "See vignette(\"nlmixr2nn-node\").", call. = FALSE)
  }
  ## capture positional inputs symbolically (do NOT evaluate them)
  .dots <- as.list(substitute(list(...)))[-1L]
  ## drop any named options accidentally caught in ... (n_hidden/act/sd)
  .nm <- names(.dots)
  if (!is.null(.nm)) .dots <- .dots[.nm == "" | is.na(.nm)]
  .inputs <- vapply(.dots, function(e) deparse1(e), character(1))
  if (length(.inputs) < 1L) stop("nn() needs at least one input", call. = FALSE)
  K <- length(.inputs) + aug
  if (K > 4L) {
    stop("nn() supports 1 to 4 inputs (nn1..nn4), and each augmented state is ",
         "an input too: ", length(.inputs), " given plus aug = ", aug,
         " is ", K, ".  Drop an input or lower `aug`.", call. = FALSE)
  }
  ## a vector n_hidden (a deeper network) is rejected HERE, at the user
  ## boundary, rather than deeper down: the initialization and the weight layout
  ## below are already written for a vector, so adding depth later is an
  ## additive change to the compiled evaluators, not a redesign.
  if (length(n_hidden) != 1L) {
    stop("nn() supports a single hidden layer; `n_hidden` must be one integer ",
         "(multi-layer networks are not yet supported)", call. = FALSE)
  }
  H <- as.integer(n_hidden)
  checkmate::assertIntegerish(H, lower = 1L, len = 1L, .var.name = "n_hidden")

  if (is.null(num)) num <- rxUdfUiNum()
  id <- num - 1L                       # 0-based id used by the compiled layer
  if (is.null(iniDf)) iniDf <- rxUdfUiIniDf()

  ## AUGMENTATION (ANODE).  `aug = k` adds k compartments that this network both
  ## reads and drives.  Every network here -- the user's and the augmented ones --
  ## sees the SAME input vector [user inputs, augmented states], which is what
  ## makes it the ANODE construction rather than k unrelated networks: the flow
  ## being learned is the one on the whole augmented state.
  ##
  ## rxode2 starts a compartment at 0 and needs no declaration for it, so the
  ## zero-augmentation initial condition comes for free.
  .augStates <- if (aug > 0L) sprintf("rx_nnaug%d_%d", id, seq_len(aug)) else character(0)
  .allInputs <- c(.inputs, .augStates)
  .argList <- paste(.allInputs, collapse = ",")
  .callOf <- function(i) paste0("nn", K, "(", i, ",", .argList, ")")

  ## record layout so nnUpdate() resolves the base index and nnCovData() adds cols.
  ## The FIRST nn() of a model parse (num == 1) resets the registry, so networks
  ## from a PREVIOUS model do not leak in (otherwise a later single-nn model would
  ## still carry a stale id-1 network from an earlier multi-nn model).
  if (num == 1L) .nnEnv$reg <- list()

  ## Register one network and return its weight-covariate declaration line.
  ##
  ## Declare the weight block as COVARIATES (not param()/thetas).  Reason: the
  ## torch/loader owns the weights (they are not nlmixr2 parameters), and -- unlike
  ## param()-declared thetas, which FOCEI theta-expands and mangles at model
  ## assembly -- covariates are contiguous in par_ptr, are not theta-expanded, and
  ## do not enter mu-referencing.  A single dummy reference makes them detected as
  ## covariates; their placeholder values are added to the data by nnCovData() and
  ## are overwritten by the par-loader (population) / inner hook (individual) on
  ## every solve.
  ##
  ## The weights are DRAWN HERE, at parse time, with R's RNG (R/nnInit.R).  They
  ## are moved onto the ui by .nnAdopt() the moment the model is assembled --
  ## `rxUdfUi()` has no field that could carry a value out of this function.
  .register <- function(netId, netSeed) {
    .wn <- nnWeightLayout(netId, K, H)
    .v <- .nnDrawSeeded(netId, K, H, act, init, initSd, netSeed)
    .nnEnv$reg[[as.character(netId)]] <-
      list(id = netId, K = K, H = H, act = act, weights = .wn,
           values = stats::setNames(.v, .wn), metaVersion = 1L)
    paste0("rx_nnw", netId, "_ <- ", paste(.wn, collapse = " + "))
  }

  .augIds <- vapply(seq_len(aug), function(j) .nnAugId(id, j), integer(1))
  ## `seed` is passed through unchanged: .nnDrawSeeded() offsets every draw by
  ## the network id, so these get distinct weights even when the user pins one
  ## seed for the whole model
  .before <- c(.register(id, seed),
               vapply(seq_len(aug), function(j) .register(.augIds[[j]], seed),
                      character(1)),
               ## Each augmented state's own learned derivative, RELAXING rather
               ## than integrating: d/dt(a) = f(x, a) - a.
               ##
               ## The textbook ANODE drift is d/dt(a) = f(x, a) with nothing
               ## holding it down, which is fine over the short fixed horizons
               ## that literature integrates.  It is unusable here.  `a` feeds
               ## back into its own derivative through the network, and the
               ## default activation (softplus) is unbounded above, so the
               ## augmented equation is self-exciting: measured, the latent
               ## reached 1e154 by t = 5, LSODA gave up ("h too small for machine
               ## precision"), and the objective stopped depending on the weights
               ## at all -- every optimizer then stalls at its starting point,
               ## which is exactly what it looked like.  A bounded activation only
               ## downgrades that from exponential to linear drift, which
               ## saturates the network over a real dosing horizon instead.
               ##
               ## The `- a` term makes it a first-order filter with a stable fixed
               ## point at f(x, a): still a genuine extra dimension -- which is
               ## all the non-crossing argument needs -- but one that cannot run
               ## away.  The cost is a fixed memory timescale of one model time
               ## unit; a learned decay rate would lift that and is the obvious
               ## next step.
               vapply(seq_len(aug),
                      function(j) paste0("d/dt(", .augStates[[j]], ") <- ",
                                         .callOf(.augIds[[j]]), " - ",
                                         .augStates[[j]]),
                      character(1)))

  list(replace = .callOf(id), before = .before)
}

## The most augmented states a network can have: it reads them all, so
## K = length(inputs) + aug <= 4 with at least one real input.
.nnAugMax <- 3L

## Id of the j-th augmented network belonging to network `id`.
##
## Primary ids are handed out by rxode2 as `rxUdfUiNum() - 1`, counting upward
## from 0, so the augmented ones are allocated DOWNWARD from the top of the
## compiled registry.  The two ranges cannot meet in any model that the C layer
## (NN_MAX = 256 networks) would accept in the first place.
##
## Stateless on purpose: a counter would have to survive rxode2 re-parsing the
## same model, and would make a network's weights depend on how many nn() calls
## happened to precede it.
.nnAugId <- function(id, j) {
  .aid <- 255L - as.integer(id) * .nnAugMax - (as.integer(j) - 1L)
  if (.aid <= id) {
    stop("nn(aug=) ran out of network ids; this model has far too many ",
         "networks", call. = FALSE)
  }
  .aid
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
#' @keywords internal
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
#' @param params parameter order of the model that will be solved, when the
#'   caller knows it; reconstructed from `x` when `NULL`.
#' @param x an rxode2 model / ui object, or a character vector of parameter
#'   names in solve order.
#' @return invisibly, a data.frame of the registered networks.
#' @keywords internal
nnUpdate <- function(x, params = NULL) {
  reg <- .nnEnv$reg
  if (length(reg) == 0L) return(invisible(data.frame()))
  ## `params` lets a caller that KNOWS which model will be solved supply its
  ## parameter order; without it the layout is reconstructed, which is only a
  ## fallback (see .nnEstSolveParams).
  if (is.null(params)) params <- .nnSolveParams(x)
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

#' Run an expression with the nn parameter-loader active
#'
#' The nn weight loader is registered under the name `"nlmixr2nn:nnParLoader"` and
#' runs only while it is the active injector, so it never overwrites an unrelated
#' model's parameters.  The transparent `nlmixr2()` workflow activates it
#' automatically; when solving a model that contains `nn()` DIRECTLY (`rxSolve()`
#' after `nnUpdate()`/`nnSetWeights()`), wrap the solve in `nnWithLoader()` so the
#' weights reach `par_ptr`.
#'
#' @param expr expression to evaluate with the nn loader active.
#' @return the value of `expr`.
#' @keywords internal
#' @author Matthew L. Fidler
nnWithLoader <- function(expr) {
  .nnLoaderOn()
  on.exit(.nnLoaderOff(), add = TRUE)
  force(expr)
}

## parameter names in solve (par_ptr) order
.nnSolveParams <- function(x) {
  if (is.character(x)) return(x)
  ini <- .nnIniDf(x)
  if (!is.null(ini)) {
    th <- ini[!is.na(ini$ntheta), , drop = FALSE]
    th <- th[order(th$ntheta), , drop = FALSE]
    ## the etas sit between the thetas and the covariates; leaving them out
    ## shifted the whole weight block down by the number of random effects
    .etas <- tryCatch(x$eta, error = function(e) character(0))
    mv <- rxode2::rxModelVars(x)
    covs <- setdiff(mv$params, c(th$name, .etas))
    return(c(th$name, .etas, covs))
  }
  rxode2::rxModelVars(x)$params
}

## Parameter order of the model a given ESTIMATOR actually solves.
##
## This has to be estimator-aware, and getting it wrong is silent.  The weight
## block sits at a different offset in each of these, for the same model:
##
##   base ui             weights at 1..13   (base 0)
##   saemModel           weights at 2..14   (base 1)
##   foceiModel$inner    weights at 3..15   (base 2)
##
## Registering the wrong one does not error -- the evaluator simply reads a
## different stretch of the parameter vector and computes a different network.
## Returns NULL when the estimator's model cannot be built, and the caller falls
## back to `.nnSolveParams()`.
.nnEstSolveParams <- function(ui, est) {
  .quiet <- function(e) suppressWarnings(suppressMessages(tryCatch(e, error = function(x) NULL)))
  .p <- NULL
  if (est %in% c("saem", "fsaem")) {
    .p <- .quiet(rxode2::rxModelVars(ui$saemModel)$params)
  } else if (!(est %in% .nnNlmOptimizers)) {
    ## the FOCEi family and everything built on it (laplace/agq/emvi/fbvi/vae,
    ## and the importance samplers, which all evaluate the FOCEi inner model)
    .p <- .quiet(rxode2::rxModelVars(ui$foceiModel$inner)$params)
  }
  if (is.null(.p) || length(.p) == 0L) return(NULL)
  .p
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
