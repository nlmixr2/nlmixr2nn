## The population (no between-subject variability) weight fit.
##
## Two callers: the nlm-family warm start that seeds a mixed-model fit, and the
## QSP branch where a population fit is the whole answer.  Both optimize the
## weight vector directly with a population-only optimizer at eta = 0.

## The nlm family: population-only optimizers (no between-subject variability).
## Used both as warmStart choices and, when passed as the est of a model with an
## nn() term, to trigger the no-BSV (QSP) population weight-fit path in .nnRun.
.nnNlmOptimizers <- c("nlm", "nlminb", "optim", "lbfgsb3c", "n1qn1",
                      "bobyqa", "newuoa", "uobyqa")

## The nn weight block's base par_ptr index in the nlm-family SOLVE model.  The
## nlm log-likelihood model declares params(THETA[1..nTheta], DV, allCovs) (see
## nlmixr2est's rxUiGet.nlmParams), so the weights sit after the thetas and the
## inserted DV -- a DIFFERENT base than the standard [thetas, covariates] layout
## nnUpdate() resolves for the base model / FOCEi.  Returns NA if the weights are
## not in allCovs (then the caller falls back to FOCEi materialization).
.nnNlmBase <- function(ui, aug) {
  .nTheta <- length(which(!ui$iniDf$fix))
  .order <- c(paste0("THETA[", seq_len(.nTheta), "]"), "DV", ui$allCovs)
  .b <- vapply(aug$nets, function(.n) match(.n$weights[1L], .order) - 1L, integer(1))
  if (anyNA(.b)) return(NULL)
  stats::setNames(.b, vapply(aug$nets, function(.n) as.character(.n$id), character(1)))
}

## Dispatch the population weight fit to any nlm-family optimizer -- nlm is simply
## an optimizer.  The gradient-based nlminb/nlm/optim(BFGS)/lbfgsb3c/n1qn1 use the
## analytic sensitivity gradient; the derivative-free minqa family bobyqa/newuoa/
## uobyqa use only the objective (rhobeg/rhoend set so an at-optimum start is safe).
## None uses a random effect.  Returns the fitted weight vector (w0 on failure).
.nnPopOptimize <- function(w0, objf, grf, est, iters) {
  ## the non-stats optimizers live in Suggests packages; if one is unavailable
  ## (e.g. the default lbfgsb3c is not installed) fall back to nlminb (always in
  ## stats) rather than skipping the warm start.
  .pkg <- c(lbfgsb3c = "lbfgsb3c", n1qn1 = "n1qn1",
            bobyqa = "minqa", newuoa = "minqa", uobyqa = "minqa")[est]
  if (!is.na(.pkg) && !requireNamespace(.pkg, quietly = TRUE)) est <- "nlminb"
  .rho <- list(rhobeg = 0.2, rhoend = 1e-4, maxfun = 50L * iters)
  .par <- tryCatch(switch(est,
    nlminb   = stats::nlminb(w0, objf, grf,
                             control = list(iter.max = iters, eval.max = 3L * iters))$par,
    nlm      = stats::nlm(function(w) { .r <- objf(w); attr(.r, "gradient") <- grf(w); .r },
                          w0, iterlim = iters)$estimate,
    optim    = stats::optim(w0, objf, grf, method = "BFGS",
                            control = list(maxit = iters))$par,
    lbfgsb3c = lbfgsb3c::lbfgsb3c(w0, objf, grf, control = list(maxit = iters))$par,
    n1qn1    = n1qn1::n1qn1(objf, grf, w0, max_iterations = iters)$par,
    bobyqa   = minqa::bobyqa(w0, objf, control = .rho)$par,
    newuoa   = minqa::newuoa(w0, objf, control = .rho)$par,
    uobyqa   = minqa::uobyqa(w0, objf, control = .rho)$par,
    stop("nlmixr2nn: unknown warmStart optimizer '", est, "'", call. = FALSE)),
    error = function(e) NULL)
  if (is.null(.par) || length(.par) != length(w0) || anyNA(.par)) w0 else unname(.par)
}

## Population (eta=0) weight pre-fit over the weight vector -- the "nlm bridge": a
## robust fixed-effects fit of the network weights used as the warm start for the
## mixed-model joint fit.  Weights sit in the optimized vector (as nlm would place
## them), the eta is fixed at its warm-start location 0 (so any population-only
## optimizer applies, no random effect), and the population error params come from
## the model ini.  `est` names the nlm-family optimizer.  The objective is the
## pooled -2 log-likelihood with the analytic d(-2LL)/dw from the augmented
## rx_predsw sensitivities.  Returns the fitted weight vector (aug$weights order).
## One exact-cotangent evaluation for the nlm population weight fit.  Bakes the
## current weights (w) into the nlm data columns, LOADS + solves the nlm objective
## once via nlmixr2est's population engine -- which fires the lik-contrib hook, so
## the EXACT per-obs error-model score d(LL)/d(f) is captured in C++ (any
## prediction-based error model, censoring included) rather than re-derived from
## the closed-form Gaussian add/prop formula.  Returns list(obj = -2LL, dLLdf
## aligned to the observation rows), or NULL to fall back to the Gaussian score.
## A fresh setup per call is required (and correct): the weights enter the solve
## as data covariates, so they are baked in at setup time.
.nnNlmExactCotangent <- function(ctx, aug, adata, w) {
  for (.net in aug$nets) {
    nnSetMeta(.net$id, ctx$nlmBases[[as.character(.net$id)]], .net$K, .net$H, .net$act)
    nnSetWeights(.net$id, w[.net$gIdx])
  }
  .dw <- .nnFillWeightCols(aug, adata)          # torch weights (= w) -> data columns
  .parIni <- tryCatch(suppressWarnings(suppressMessages(
    nlmixr2est::nlmObjectiveSetup(ctx$ui, .dw, ctx$control))), error = function(e) NULL)
  if (is.null(.parIni)) return(NULL)
  on.exit(try(nlmixr2est::.nlmFreeEnv(), silent = TRUE), add = TRUE)
  .nnCapReset(TRUE, ctx$capDims$nId, ctx$capDims$kStride)
  ## `:::` deliberately: nlmSolveR is internal to nlmixr2est.  It was written as
  ## `::`, which raises "not an exported object" -- and because that error was
  ## swallowed by the tryCatch below, this whole exact-cotangent path silently
  ## fell back to the Gaussian score on every call and never once ran.
  .obj <- tryCatch(nlmixr2est:::nlmSolveR(.parIni), error = function(e) NA_real_)
  .cap <- .nnCapGet(); .nnCapReset(FALSE)
  if (!is.finite(.obj) || is.null(.cap) || !length(.cap$id)) return(NULL)
  .dLLdf <- .cap$dLLdf[match(ctx$obsKey, .cap$id * ctx$capStride + .cap$k)]
  if (anyNA(.dLLdf)) return(NULL)
  list(obj = 2 * .obj, dLLdf = .dLLdf)          # objf = -2LL = 2 * minimum
}

.nnPopWarmStart <- function(aug, data, idCol, obs, dv, wPlaceholder, thetas, errPar,
                            w0, est, iters, exactCtx = NULL) {
  .ad <- data
  for (.e in names(aug$covMap)) .ad[[aug$covMap[[.e]]]] <- 0    # population: eta = 0
  if (errPar$add == 0 && errPar$prop == 0) errPar$add <- 1      # avoid R(f)=0
  .key <- paste(data[[idCol]][obs], data$time[obs])
  .dvObs <- dv[obs]
  ## objective (-2 log-likelihood) + analytic gradient at the GLOBAL weight vector w
  ## (all networks concatenated in aug$nets order)
  .eval <- function(w) {
    ## exact per-obs score from the C++ nlm solve when requested (else NULL)
    .ex <- if (!is.null(exactCtx)) .nnNlmExactCotangent(exactCtx, aug, .ad, w) else NULL
    for (.net in aug$nets) {                                    # split w per net
      nnSetMeta(.net$id, .net$augBase, .net$K, .net$H, .net$act)
      nnSetWeights(.net$id, w[.net$gIdx])
    }
    .s <- rxode2::rxSolve(aug$mAug, .ad, params = c(thetas, wPlaceholder),
                          returnType = "data.frame")
    .ik <- match(.key, paste(.s$id, .s$time))
    .f <- .s[[aug$endpoint]][.ik]
    .resid <- .dvObs - .f
    .R <- errPar$add^2 + (errPar$prop * .f)^2
    if (!is.null(.ex)) {                          # exact C++ cotangent + -2LL
      .dLLdf <- .ex$dLLdf
      .obj <- .ex$obj
    } else {                                       # closed-form Gaussian score
      .dRdf <- 2 * errPar$prop^2 * .f
      .dLLdf <- .resid / .R + 0.5 * (.resid^2 / .R^2 - 1 / .R) * .dRdf
      .obj <- sum(log(2 * pi * .R) + .resid^2 / .R)
    }
    list(obj = .obj,
         grad = -2 * vapply(aug$predswCols, function(cn) sum(.dLLdf * .s[[cn]][.ik]),
                            numeric(1), USE.NAMES = FALSE))
  }
  ## cache the last evaluation so paired objective/gradient calls solve once
  .cache <- new.env(parent = emptyenv())
  .get <- function(w) {
    if (is.null(.cache$w) || !isTRUE(all.equal(w, .cache$w))) {
      .cache$w <- w; .cache$v <- .eval(w)
    }
    .cache$v
  }
  .nnPopOptimize(w0, function(w) .get(w)$obj, function(w) .get(w)$grad, est, iters)
}
