## Choosing the training schedule.
##
## `nnControl()` has thirteen knobs, and requiring a user to set them was the
## single biggest reason fitting an nn() model did not feel like fitting a model.
## Every one of them is now inferred from things the package already knows before
## any solve happens: which estimator was asked for, whether that estimator can
## resume from a partial fit, whether the model has a random effect, what the
## residual model is, and how big the network is relative to the data.
##
## `nnControl()` remains available and always wins where the user set something
## explicitly -- except where the explicit value would be WRONG rather than
## merely different, which errors instead.

## Estimators whose partial outer step does not genuinely resume.  SAEM's
## stochastic-approximation gain sequence restarts on every call, so chopping it
## into `nEm` pieces does not continue the chain -- interleaving buys nothing
## over a full fit for these, and they use the block-coordinate loop instead.
.nnNonResuming <- c("saem", "fsaem", "qrpem", "npag", "npb")

## Endpoint facts, read from the ui's own `predDf` rather than re-parsed from the
## model text: the number of endpoints, the residual distribution, the
## transformation, and whether a closed-form Gaussian score applies.
.nnEndpointInfo <- function(ui) {
  .p <- tryCatch(ui$predDf, error = function(e) NULL)
  if (is.null(.p) || !is.data.frame(.p) || nrow(.p) == 0L) {
    return(list(n = NA_integer_, distribution = NA_character_,
                transform = NA_character_, errType = NA_character_,
                closedForm = FALSE, conds = character(0)))
  }
  .dist <- as.character(.p$distribution[1L])
  .tr <- as.character(.p$transform[1L])
  .err <- as.character(.p$errType[1L])
  list(n = nrow(.p), distribution = .dist, transform = .tr, errType = .err,
       ## the closed-form additive/proportional Gaussian score is only valid on
       ## an untransformed normal endpoint; anything else needs the exact
       ## per-observation score from the likelihood hook
       closedForm = identical(.dist, "norm") &&
         identical(.tr, "untransformed") &&
         !is.na(.err) && grepl("add|prop", .err),
       conds = as.character(.p$cond))
}

.nnClamp <- function(x, lo, hi) as.integer(max(lo, min(hi, x)))

#' Infer a training schedule from the model, data and estimator
#'
#' @param env the estimation environment handed to the interceptor.
#' @param nW total number of network weights.
#' @param hasTrained whether the model already carries trained weights.
#' @return a fully-resolved schedule list.
#' @keywords internal
#' @noRd
.nnInferSched <- function(env, nW = NA_integer_, hasTrained = FALSE) {
  .ui <- tryCatch(rxode2::rxUiDecompress(env$ui), error = function(e) env$ui)
  .est <- class(env)[1L]
  .ctl <- env$control
  .ep <- .nnEndpointInfo(.ui)
  .hasEta <- length(tryCatch(.ui$eta, error = function(e) character(0))) > 0L
  .isNlm <- .est %in% .nnNlmOptimizers
  .knob <- .nnInterleaveKnob(.ctl)
  .resumes <- !is.null(.knob) && !(.est %in% .nnNonResuming)
  .nObs <- tryCatch({
    .d <- env$data
    .e <- .d[[if ("EVID" %in% names(.d)) "EVID" else "evid"]]
    sum(is.na(.e) | .e == 0)
  }, error = function(e) NA_integer_)

  ## ---- guards: refuse combinations that cannot mean what they look like -----
  if (!is.na(.ep$n) && .ep$n > 1L) {
    stop("nlmixr2nn supports a single endpoint; this model has ", .ep$n,
         " (", paste(.ep$conds, collapse = ", "), ")", call. = FALSE)
  }
  if (!is.na(.ep$distribution) && !identical(.ep$distribution, "norm")) {
    stop("nlmixr2nn supports normal-distribution endpoints (any transformation); ",
         "this model's endpoint is '", .ep$distribution, "'.  For ll()/pois()/",
         "binom() the likelihood hook reports d(LL)/d(f) with f the log-density, ",
         "which cannot yet be chained through the ODE sensitivity.", call. = FALSE)
  }
  if (.isNlm && .hasEta) {
    ## the nlm family is a population optimizer: it has no random effects at all.
    ## Fitting the weights at eta = 0 and then materializing Omega with focei
    ## produces a fit whose weights and Omega come from different objectives.
    stop("est=\"", .est, "\" is a population-only optimizer, but this model has a ",
         "random effect (", paste(.ui$eta, collapse = ", "), ").  Use est=\"focei\" ",
         "(or another mixed-effects estimator), or remove the random effect.",
         call. = FALSE)
  }

  ## ---- the schedule ---------------------------------------------------------
  .cot <- if (.ep$closedForm) "gaussian" else "exact"
  .s <- list(
    mode = "iter", tol = 1e-3, outerPerRound = 1L,
    cotangent = .cot,
    ## the exact cotangent is captured once per round at that round's weights, so
    ## taking several weight steps against it would reuse a stale score; the
    ## closed-form score is recomputed from each fresh solve, so it can.
    wSteps = if (identical(.cot, "exact")) 1L else 4L,
    ## a refit starts from a good point, so it takes smaller steps
    lr = if (hasTrained) 0.01 else 0.03,
    warmSteps = 0L, optimizer = "adam", seed = NULL,
    ## trained weights ARE the warm start; re-running the population pre-fit
    ## would throw them away
    warmStart = if (hasTrained) "none" else "lbfgsb3c",
    warmPopIters = 3L)

  if (.isNlm) {
    ## population weight fit: `rounds` is the optimizer's iteration cap, so it
    ## scales with the number of parameters being optimized
    .s$rounds <- if (is.na(nW)) 200L else .nnClamp(5L * nW, 50L, 500L)
    .s$warmStart <- .est
    .s$mode <- "iter"
  } else if (.hasEta && .resumes) {
    ## true interleave: a few outer iterations per round, co-descending the
    ## population parameters and the weights
    .s$mode <- "joint"
    .s$outerPerRound <- if (identical(.knob, "iters")) 30L else 1L
    .s$rounds <- 200L
  } else if (.hasEta) {
    ## each round is a FULL mixed-effects fit, so far fewer of them
    .s$mode <- "iter"
    .s$rounds <- 30L
  } else {
    ## population UDE: no Omega to co-descend, and each round is cheap
    .s$mode <- "iter"
    .s$rounds <- 60L
  }

  if (identical(.s$warmStart, "none")) {
    .s$warmPopIters <- 1L
  } else if (!.hasEta && !is.na(.nObs) && !is.na(nW) && nW > 0L) {
    ## with no random effect there is no per-subject variation for the pre-fit to
    ## absorb, so it can run as long as the data can identify the weights
    .s$warmPopIters <- .nnClamp(.nObs %/% nW, 3L, 50L)
  }
  .s$predicates <- list(est = .est, hasEta = .hasEta, isNlm = .isNlm,
                        resumes = .resumes, knob = .knob, closedForm = .ep$closedForm,
                        transform = .ep$transform, hasTrained = hasTrained)
  .s
}

#' Overlay a user's explicit nnControl() on an inferred schedule
#'
#' Correctness conflicts error; quality or efficiency conflicts message once and
#' honour what the user asked for.
#' @keywords internal
#' @noRd
.nnResolveSched <- function(user, inferred) {
  if (is.null(user)) return(inferred)
  .set <- attr(user, "supplied")
  if (is.null(.set)) {
    ## an nnControl() from before the sentinel defaults: treat every non-NULL
    ## field as deliberate
    .set <- names(user)[!vapply(user, is.null, logical(1))]
  }
  .p <- inferred$predicates
  .out <- inferred
  for (.n in .set) .out[[.n]] <- user[[.n]]

  ## --- correctness ---------------------------------------------------------
  if ("cotangent" %in% .set && identical(user$cotangent, "gaussian") &&
        !isTRUE(.p$closedForm)) {
    stop("nnControl(cotangent = \"gaussian\") needs an untransformed additive/",
         "proportional normal endpoint; this model's endpoint is '",
         .p$transform, "'.  Omit `cotangent=` and the exact score is used.",
         call. = FALSE)
  }
  ## --- quality / efficiency ------------------------------------------------
  if ("mode" %in% .set && identical(user$mode, "joint") && !isTRUE(.p$resumes)) {
    message("nn: est=\"", .p$est, "\" has no resumable partial outer step; ",
            "using the iterative loop")
    .out$mode <- "iter"
  }
  if (identical(.out$cotangent, "exact") && isTRUE(.out$wSteps > 1L)) {
    warning("nn: the exact cotangent is captured once per round, so wSteps > 1 ",
            "reuses a score taken at the round's starting weights", call. = FALSE)
  }
  if ("seed" %in% .set && !is.null(user$seed)) {
    message("nn: nnControl(seed=) no longer affects initialization -- the ",
            "weights are drawn when the model is parsed; use set.seed() before ",
            "building the model")
  }
  .out
}
