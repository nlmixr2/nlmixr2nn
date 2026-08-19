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
.nnWeightStepper <- function(aug, data, idCol, timeCol, obs, dv, wPlaceholder) {
  ## weightStep(ebes, errPar, thetas, dLLdfObs = NULL): dLLdfObs, when supplied, is
  ## the per-observation error-model cotangent captured from the inner fit (the
  ## EXACT dLL/df for any residual model), aligned to the observation rows; NULL
  ## uses the closed-form additive/proportional-Gaussian cotangent.
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
      .grad[[as.character(.net$id)]] <- unname(.dLLdw)
      if (step) {
        nnTorchZeroGrad(.net$id)
        nnTorchSetGrad(.net$id, -.dLLdw)
        nnTorchStep(.net$id)
      }
    }
    list(rmse = sqrt(mean(.resid^2)), f = .f, dLLdf = .dLLdf, dLLdw = .grad)
  }
}
