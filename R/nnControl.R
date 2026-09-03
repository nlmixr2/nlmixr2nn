## nnControl(): the neural-network TRAINING SCHEDULE for the transparent nlmixr2nn
## workflow.  It is NOT an estimation control and does NOT wrap an inner control.
## You pass it alongside a standard estimation:
##
##   nlmixr2(model, data, "focei", foceiControl(...), nn = nnControl(...))
##
## When the model contains an nn() term, nlmixr2nn's estimation interceptor picks
## up `nn=` (or a default nnControl() if absent) and trains the embedded network
## using the requested estimator (focei/saem/imp/...) as the inner engine.

#' Neural-network training schedule for `nlmixr2nn`
#'
#' Passed as the `nn=` argument to [nlmixr2est::nlmixr2()] to train a model that
#' contains an [nn()] term.  It carries only the network training knobs -- the
#' inner NLME estimator and its control are the ordinary `est`/`control`
#' arguments of `nlmixr2()`.
#'
#' Two modes:
#' * `"joint"` (default) -- co-optimizes the population parameters and the network
#'   weights in one interleaved loop (a few warm-started partial inner iterations
#'   + a torch weight step each round), the DeepPumas-style approach;
#' * `"iter"` -- runs a full inner fit each round with the weights fixed, then a
#'   weight update (simpler solve/update/solve).
#'
#' The loop stops when the weights (and, for `"joint"`, the objective) stop
#' changing between rounds (`tol`), or after `rounds`.
#'
#' @param mode `"joint"` or `"iter"`.
#' @param rounds maximum number of rounds.
#' @param tol convergence tolerance on the relative round-to-round change (weights
#'   for `"iter"`; weights and objective for `"joint"`).  0 always runs `rounds`.
#' @param wSteps torch weight-optimizer steps per round.
#' @param outerPerRound (`"joint"` only) inner outer-iterations per round -- the
#'   partial step that makes it joint rather than a full re-fit.
#' @param lr torch optimizer learning rate.
#' @param warmSteps optional naive-pooled (eta=0) torch warm-up steps before the
#'   loop (default 0).
#' @param warmStart population weight pre-fit before the loop -- the "nlm bridge".
#'   Runs a fixed-effects (eta = 0) fit of the weights with the named nlm-family
#'   optimizer, seeding the joint fit with a robust population weight vector
#'   (bridging the fixed-effects and DeepPumas approaches).  Since nlm is simply an
#'   optimizer, the choice is `"lbfgsb3c"` (default), `"nlminb"`, `"nlm"`, `"optim"`
#'   (BFGS), `"n1qn1"` (gradient-based, using the analytic sensitivity gradient) or
#'   `"bobyqa"`, `"newuoa"`, `"uobyqa"` (derivative-free); `"none"` disables it.
#'   `"pop"` is an alias for `"nlminb"`.  The pre-fit optimizes only the weights and
#'   never uses a random effect, so any of these population-only optimizers applies.
#'   It is SKIPPED automatically when the model already carries trained weights
#'   (e.g. a fit passed back in) -- those are the warm start.  `warmPopIters` caps
#'   its iterations.
#' @param warmPopIters iteration cap for the `warmStart` pre-fit (default 3).  Kept
#'   deliberately light: a few iterations escape a bad random initialization while
#'   leaving the per-subject variation for the random effect.  A longer population
#'   pre-fit can over-fit -- the network then absorbs IIV that should go to the
#'   random effect (weakening eta recovery) and can push an SAEM inner fit into a
#'   poorly conditioned region -- so raise it only when you want a stronger
#'   fixed-effects starting point (e.g. a population/UDE model with no random
#'   effect).
#' @param l2 strength of an L2 penalty on the network's weights -- the same idea
#'   as ridge regression, and what machine learning calls weight decay.  The name
#'   is the L2 norm: the penalty is the SUM OF SQUARED weights, so a weight costs
#'   more the further it is from zero, and the fit prefers the flattest network
#'   consistent with the data.  `l2` is how much that preference is worth relative
#'   to fitting the observations; `0` expresses no preference at all.
#'
#'   Concretely it adds `l2 * sum(Weff^2)` over the weight MATRICES to the
#'   objective the weight step minimizes.  Biases stay free, so the
#'   network can still move its output level without paying for it.  `Weff` is the
#'   effective weight -- the first layer is penalized on `scale * W1`, because
#'   input scaling is folded into `W1` (see `nnScale.R`) and raw `W1` therefore
#'   means something different for a covariate of magnitude 500 than for one of
#'   magnitude 1.  A latent eta's input column is exempt too: the eta reaches the
#'   model only through the network, so shrinking those weights would shrink the
#'   random effect itself.
#'
#'   The value is a FRACTION, not an absolute amount: `l2 = 0.05` means the weight
#'   term starts at 5% of the objective, which makes one value mean roughly the
#'   same thing across endpoints, estimators and data sizes.
#'
#'   Defaults to `0` (off).  This is measured, not cautious: L2 shrinks toward the
#'   zero function, and here the network IS the model, so any value large enough
#'   to suppress structure the data does not support also attenuates structure it
#'   does.  Reach for it when a fit is visibly overfitting -- `0.05` is where
#'   shrinkage becomes substantial (on a test fixture, weight norm 11.5 -> 3.9
#'   with the unpenalized objective improving) -- and check what the network
#'   learned with [nnEval()], because real effects shrink alongside invented ones.
#' @param smooth strength of a curvature penalty -- a roughness penalty in the
#'   spline sense, asking the network to be a SMOOTH function of its inputs rather
#'   than a small one.  Where `l2` pulls the network toward flat, `smooth` pulls it
#'   toward gently curved, which is usually what you want from a learned covariate
#'   relationship: a shape, not a straight line and not an interpolation of noise.
#'
#'   Concretely it adds `smooth * sum(C^2)` where
#'   `C = f(x+h) - 2*f(x) + f(x-h)` is the second difference of the network along
#'   each input's marginal curve, that input swept over its observed range with
#'   the others held at their center.  Penalizing the SECOND derivative rather
#'   than the first leaves monotone slopes free and charges only for the wiggles.
#'   Inputs whose range is not knowable -- an eta, or a compound expression such
#'   as `nn(central/Vc)` -- are held fixed rather than swept.  Like `l2` it is a
#'   fraction of the objective and defaults to `0`; the two are normalized
#'   separately, because at equal lambda the raw curvature sum is orders of
#'   magnitude smaller than the raw weight sum and the mix would otherwise be
#'   arbitrary.
#'
#'   Both penalties shape the OPTIMIZATION only.  The reported `objf` (and hence
#'   AIC/BIC) stays the unpenalized -2 log-likelihood, so a regularized network
#'   model remains directly comparable to an analytic-covariate one.
#' @param optimizer torch optimizer, `"adam"` or `"sgd"`.
#' @param cotangent source of the endpoint score `dLL/df` used to form the weight
#'   gradient.  `"gaussian"` is the closed-form additive/proportional Gaussian
#'   cotangent.  `"dist"` is the endpoint distribution's own derivative, for a
#'   count endpoint such as `pois()` or `binom()`, where `f` is the distribution's
#'   parameter rather than a prediction.  `"exact"` is the per-observation
#'   cotangent captured from the inner fit's likelihood contribution hook, which
#'   is correct for any NORMAL residual model including transform-both-sides
#'   (lognormal, Box-Cox, logit) -- best paired with `wSteps = 1`, since the
#'   captured cotangent is at the round's starting weights.  `"exact"` does NOT
#'   apply to a count endpoint: there the hook reports `dLL/df = 1`, because `f`
#'   is the log-density itself.  Left unset, the source is chosen from the
#'   endpoint, which is what you want.
#' @param seed optional integer seed for torch weight initialization (ignored when
#'   the model already carries trained weights, which are used as the start).
#' @return an object of class `"nnControl"`.
#' @export
#' @author Matthew L. Fidler
nnControl <- function(mode = NULL, rounds = NULL, tol = NULL, wSteps = NULL,
                      outerPerRound = NULL, lr = NULL, warmSteps = NULL,
                      warmStart = NULL, warmPopIters = NULL, cotangent = NULL,
                      optimizer = NULL, seed = NULL, l2 = NULL, smooth = NULL) {
  ## Every argument defaults to NULL, meaning "infer it" (R/nnSchedule.R).  What
  ## matters is not the value but WHICH arguments the caller named: an explicit
  ## `mode = "joint"` must be distinguishable from the inferred one, so that the
  ## schedule can honour it -- or refuse it -- deliberately.
  .supplied <- setdiff(names(as.list(match.call()))[-1L], "")

  .oneOf <- function(v, nm, choices) {
    if (is.null(v)) return(NULL)
    v <- as.character(v)
    if (length(v) != 1L || !(v %in% choices)) {
      stop("nnControl(", nm, "=) must be one of: ",
           paste0("\"", choices, "\"", collapse = ", "), call. = FALSE)
    }
    v
  }
  mode <- .oneOf(mode, "mode", c("joint", "iter"))
  optimizer <- .oneOf(optimizer, "optimizer", c("adam", "sgd"))
  cotangent <- .oneOf(cotangent, "cotangent", c("gaussian", "dist", "exact"))
  warmStart <- .oneOf(warmStart, "warmStart",
                      c("lbfgsb3c", "nlminb", "nlm", "optim", "n1qn1",
                        "bobyqa", "newuoa", "uobyqa", "none", "pop"))
  if (identical(warmStart, "pop")) warmStart <- "nlminb"   # back-compat alias

  ## `lower` is optional: a seed is an arbitrary integer, with no bound at all,
  ## and checkmate rejects `lower = NA` outright
  .int <- function(v, nm, lower = NULL) {
    if (is.null(v)) return(NULL)
    if (is.null(lower)) {
      checkmate::assertIntegerish(v, len = 1L, .var.name = nm)
    } else {
      checkmate::assertIntegerish(v, lower = lower, len = 1L, .var.name = nm)
    }
    as.integer(v)
  }
  .num <- function(v, nm, lower) {
    if (is.null(v)) return(NULL)
    checkmate::assertNumeric(v, lower = lower, len = 1L, .var.name = nm)
    as.numeric(v)
  }
  structure(list(mode = mode,
                 rounds = .int(rounds, "rounds", 1L),
                 tol = .num(tol, "tol", 0),
                 wSteps = .int(wSteps, "wSteps", 1L),
                 outerPerRound = .int(outerPerRound, "outerPerRound", 1L),
                 lr = .num(lr, "lr", 0),
                 warmSteps = .int(warmSteps, "warmSteps", 0L),
                 warmStart = warmStart,
                 warmPopIters = .int(warmPopIters, "warmPopIters", 1L),
                 cotangent = cotangent, optimizer = optimizer,
                 seed = .int(seed, "seed"),
                 l2 = .num(l2, "l2", 0), smooth = .num(smooth, "smooth", 0)),
            supplied = .supplied, class = "nnControl")
}

#' @export
print.nnControl <- function(x, ...) {
  .set <- attr(x, "supplied")
  if (is.null(.set) || length(.set) == 0L) {
    cat("nnControl (nlmixr2nn): everything inferred from the model, data and estimator\n")
    return(invisible(x))
  }
  cat("nnControl (nlmixr2nn): ",
      paste(vapply(.set, function(.n) paste0(.n, "=", format(x[[.n]])), character(1)),
            collapse = ", "),
      "\n  (all other settings inferred)\n", sep = "")
  invisible(x)
}
