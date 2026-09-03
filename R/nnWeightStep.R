## One weight step: the gradient assembly and the torch update.
##
## This is where the chain closes -- the augmented solve's parameter
## sensitivities meet the endpoint's cotangent dLL/df, and the product is handed
## to the optimizer.  Shared by both training modes and by the population fit.

## Factory for one torch weight step (shared by both nn training modes).  Given
## the augmented model + weight metadata and the fixed data pieces, returns a
## closure weightStep(ebes, errPar, thetas) that: sets the latent-eta covariates
## to the EBEs, solves the augmented model at the fitted thetas, forms the score
## dLL/df -- the Gaussian cotangent for variance R(f)=add^2+(prop*f)^2, or the
## endpoint distribution's own derivative for a count endpoint -- assembles
## dLL/dw = sum_obs dLL/df * rx_predsw, and takes one torch optimizer step.
## Returns the RMSE.
.nnWeightStepper <- function(aug, data, idCol, timeCol, obs, dv, wPlaceholder,
                             pen = NULL) {
  ## weightStep(ebes, errPar, thetas, dLLdfObs = NULL): dLLdfObs, when supplied, is
  ## the per-observation error-model cotangent captured from the inner fit (the
  ## EXACT dLL/df for any residual model), aligned to the observation rows; NULL
  ## uses the closed-form additive/proportional-Gaussian cotangent.
  ## the penalty spec keyed by network id, so the per-net lookup inside the step
  ## is a name match rather than a scan
  .penOf <- if (is.null(pen)) {
    NULL
  } else {
    stats::setNames(pen$nets, vapply(pen$nets, function(.s) as.character(.s$id),
                                     character(1)))
  }
  function(ebes, errPar, thetas, dLLdfObs = NULL, step = TRUE) {
    .ad <- data
    for (.e in names(aug$covMap)) .ad[[aug$covMap[[.e]]]] <- ebes[as.character(.ad[[idCol]])]
    for (.net in aug$nets) {                            # each net at its augmented base
      nnSetMeta(.net$id, .net$augBase, .net$K, .net$H, .net$act)
      nnSetWeights(.net$id, nnTorchWeights(.net$id))
    }
    .p <- c(thetas, wPlaceholder)
    .s <- rxode2::rxSolve(aug$mAug, .ad, params = .p, returnType = "data.frame")
    if (!all(aug$predswCols %in% names(.s))) {
      stop("nlmixr2nn: augmented solve is missing the prediction-sensitivity ",
           "columns (rx_predsw_*)", call. = FALSE)
    }
    .ik <- match(paste(data[[idCol]][obs], data[[timeCol]][obs]), paste(.s$id, .s$time))
    .f <- .s[[aug$endpoint]][.ik]
    .resid <- dv[obs] - .f
    if (!is.null(dLLdfObs)) {
      ## The captured score is d(LL)/d(TRANSFORMED prediction) -- transform-both-
      ## sides is applied in rxode2's rx_pred_ statement, so the hook never sees
      ## the natural scale.  `rx_sw` below are natural-scale sensitivities, so
      ## the transformation's own derivative has to close the chain; without it
      ## a lnorm/boxCox/logit endpoint trained on a gradient short by that factor
      ## at every observation (R/nnEndpoint.R).
      .dLLdf <- dLLdfObs
      if (.nnNeedsTransformJac(aug$ep)) {
        .dLLdf <- .dLLdf * .nnTransformJac(aug$ep, .f)
      }
    } else if (!is.null(aug$dist)) {
      ## count endpoint: .f is the distribution's parameter (lam, prob), not a
      ## prediction, and the score is that distribution's own derivative
      .sz <- NULL
      if (!is.na(aug$dist$size)) {
        .sz <- if (aug$dist$size %in% names(.s)) {
          .s[[aug$dist$size]][.ik]
        } else if (aug$dist$size %in% names(data)) {
          data[[aug$dist$size]][obs]
        } else {
          stop("nlmixr2nn: '", aug$dist$size, "' is the size of the ",
               aug$dist$dist, " endpoint but is neither a model variable nor a ",
               "data column, so its score cannot be formed", call. = FALSE)
        }
      }
      .dLLdf <- .nnDistScore(aug$dist, .f, dv[obs], .sz)
    } else {
      .R <- errPar$add^2 + (errPar$prop * .f)^2
      .dRdf <- 2 * errPar$prop^2 * .f
      .dLLdf <- .resid / .R + 0.5 * (.resid^2 / .R^2 - 1 / .R) * .dRdf
    }
    ## `step = FALSE` assembles the gradient without moving the weights, which
    ## is what lets the two cotangent sources be compared AT THE GRADIENT rather
    ## than by running two whole fits and hoping the difference shows.
    .grad <- list()
    for (.net in aug$nets) {                            # per-network gradient + step
      .dLLdw <- vapply(.net$predswCols, function(cn) sum(.dLLdf * .s[[cn]][.ik]), numeric(1))
      ## A non-finite gradient must never reach the optimizer.  One torch step
      ## against NaN puts NaN in every weight, and there is no recovering from
      ## that: every later solve, score and objective is NaN too, while the fit
      ## still returns an object with a number in it.  Refusing here turns the
      ## whole class of causes -- a diverged solve, an unusable score, a data row
      ## that should not have been in the sum -- into something the user can see.
      if (!all(is.finite(.dLLdw))) {
        stop(sprintf(paste0(
          "nlmixr2nn: the weight gradient for network %s is not finite ",
          "(%d of %d components; %d of %d observations have a non-finite score, ",
          "%d a non-finite prediction).  The optimizer is not stepped, because a ",
          "single non-finite step makes every weight NaN for the rest of the fit."),
          .net$id, sum(!is.finite(.dLLdw)), length(.dLLdw),
          sum(!is.finite(.dLLdf)), length(.dLLdf), sum(!is.finite(.f))),
          call. = FALSE)
      }
      .grad[[as.character(.net$id)]] <- unname(.dLLdw)
      if (step) {
        ## The likelihood gradient above is on the -LL scale, the penalty is
        ## defined on -2LL, and .nnAddPen() closes that gap -- see the header of
        ## R/nnPenalty.R before changing either side of this.  `pen = NULL`
        ## (lambda 0) returns the vector untouched, so an unregularized fit is
        ## bit-identical to one built without this call.
        .g <- -.dLLdw
        if (!is.null(pen)) {
          .wNet <- nnTorchWeights(.net$id)
          .g <- .nnAddPenNet(.g, .wNet, pen, .penOf[[as.character(.net$id)]], 1L)
          if (!all(is.finite(.g))) {
            stop(sprintf(paste0(
              "nlmixr2nn: the weight penalty made the gradient for network %s ",
              "non-finite (%d of %d components).  The optimizer is not stepped. ",
              "Lower nnControl(l2=)/nnControl(smooth=), or set them to 0."),
              .net$id, sum(!is.finite(.g)), length(.g)), call. = FALSE)
          }
        }
        nnTorchZeroGrad(.net$id)
        nnTorchSetGrad(.net$id, .g)
        nnTorchStep(.net$id)
      }
    }
    list(rmse = sqrt(mean(.resid^2)), f = .f, dLLdf = .dLLdf, dLLdw = .grad)
  }
}
