## Pooled NN-in-ODE fitting via the forward-sensitivity gradient + torch.
##
## Trains the weights of an embedded network to fit multi-subject data by
## repeatedly (1) solving the augmented model (base states + rx_sw variational
## states) over the data, (2) forming the analytic dLL/dw = sum_obs (dLL/df) *
## rx_sw at each observation, (3) injecting -dLL/dw into the torch optimizer and
## stepping.  Additive-Gaussian error on a single predicted state; the residual
## SD is profiled out by closed-form MLE each iteration when estSigma = TRUE.
## This is a naive-pooled fit (no random effects) -- the FOCEI/SAEM interleave
## with per-subject etas is a separate, deeper integration.

#' Compute the par_ptr base of a network's weight block for a model
#' @keywords internal
.nnWeightBase <- function(model, nnId, K, H) {
  wnm <- nnWeightLayout(nnId, K, H)
  pars <- rxode2::rxModelVars(model)$params
  pos <- match(wnm, pars)
  if (anyNA(pos)) stop("weight params not all declared in the model; add param(",
                       paste(wnm, collapse = ", "), ")", call. = FALSE)
  if (!all(diff(pos) == 1L)) stop("weight params must be contiguous in the model", call. = FALSE)
  min(pos) - 1L
}

#' Fit an embedded neural network in an ODE by forward-sensitivity + torch
#'
#' @param model rxode2 model text with one `g = nn<K>(id, ...)` output and the
#'   weight block declared via `param(nnWeightLayout(...))`.
#' @param data event data frame (columns id, time, evid, dv, amt as needed).
#' @param pred name of the predicted state that `dv` observes.
#' @param nnId,K,H,act network id, input dim, hidden width, activation.
#' @param params named numeric of fixed population parameters (non-weight).
#' @param inits named numeric of base-state initial conditions.
#' @param optimizer "adam" or "sgd"; `lr` learning rate; `iter` iterations.
#' @param sigma initial residual SD; `estSigma` profiles it by MLE each iteration.
#' @param seed optional torch init seed; `verbose` prints the LL trace.
#' @return list(weights, sigma, ll, llTrace).
#' @export
nnTrain <- function(model, data, pred, nnId = 0L, K, H, act = "tanh",
                    params = numeric(0), inits = numeric(0),
                    optimizer = "adam", lr = 0.05, iter = 50,
                    sigma = 1, estSigma = TRUE, seed = NULL, verbose = FALSE) {
  nW <- H * K + 2L * H + 1L
  wnm <- nnWeightLayout(nnId, K, H)
  mBase <- rxode2::rxode2(model)
  base <- .nnWeightBase(mBase, nnId, K, H)
  nnSetMeta(nnId, base = base, K = K, H = H, act = act)
  if (!is.null(seed)) nnTorchInit(nnId, K, H, act = act, seed = seed)
  nnTorchOptInit(nnId, optimizer, lr)
  mAug <- rxode2::rxode2(nnAugmentModel(model, H = H))

  p <- c(params, setNames(rep(0, nW), wnm))          # weight placeholders (loader overwrites)
  obs <- data[is.na(data$evid) | data$evid == 0, , drop = FALSE]
  obs <- obs[!is.na(obs$dv), , drop = FALSE]
  swCols <- sprintf("rx_sw_%s_%d_", pred, seq_len(nW) - 1L)

  solveObs <- function(mod) {
    nnSetWeights(nnId, nnTorchWeights(nnId))
    s <- rxode2::rxSolve(mod, data, params = p, inits = inits, returnType = "data.frame")
    ik <- match(paste(obs$id, obs$time), paste(s$id, s$time))
    s[ik, , drop = FALSE]
  }
  logLik <- function(f, sig) sum(dnorm(obs$dv, f, sig, log = TRUE))

  llTrace <- numeric(iter)
  for (it in seq_len(iter)) {
    s <- solveObs(mAug)
    f <- s[[pred]]
    resid <- obs$dv - f
    if (estSigma) sigma <- sqrt(mean(resid^2))
    dLLdf <- resid / sigma^2
    dLLdw <- vapply(swCols, function(cn) sum(dLLdf * s[[cn]]), numeric(1))
    nnTorchZeroGrad(nnId)
    nnTorchSetGrad(nnId, -dLLdw)                      # minimize -LL
    nnTorchStep(nnId)
    llTrace[it] <- logLik(f, sigma)
    if (verbose) message(sprintf("iter %d: LL = %.4f  sigma = %.4g", it, llTrace[it], sigma))
  }
  list(weights = nnTorchWeights(nnId), sigma = sigma,
       ll = llTrace[iter], llTrace = llTrace)
}
