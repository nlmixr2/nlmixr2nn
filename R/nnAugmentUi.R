## Building the ephemeral forward-sensitivity model from a user's nn() ui.
##
## nnAugmentModel() (R/nnAugment.R) works on model TEXT; this is the layer that
## gets that text out of a ui -- resolving the endpoint, the latent etas that
## become covariates, and each network's slice of the global weight vector --
## and compiles the result.  Built, solved and discarded inside the loop; the
## base model never carries sensitivity states.

## d(prediction)/d(state) for each ODE state, so the prediction's forward
## sensitivity wrt a weight can be chained from the state sensitivities rx_sw:
## d(pred)/dw = sum_s d(pred)/d(s) * rx_sw_s.  When the prediction IS a state
## this is the identity (1 for that state); when it is an lhs (e.g. cp = centr/V)
## the chain is nontrivial.  Returns a character vector of derivative expressions
## (rxode2 syntax) named by state.
.nnDpDs <- function(modelText, states, predVar) {
  if (predVar %in% states) {
    return(stats::setNames(as.character(as.integer(states == predVar)), states))
  }
  .model <- rxode2::rxS(rxode2::rxGetModel(modelText), TRUE, promoteLinSens = FALSE)
  .p <- get0(predVar, envir = .model, inherits = FALSE)
  if (is.null(.p)) {
    stop("nlmixr2nn: cannot resolve the prediction variable '", predVar, "'",
         call. = FALSE)
  }
  vapply(states, function(s) {
    .dd <- symengine::D(.p, symengine::Symbol(s))   # bind before rxFromSE (NSE)
    rxode2::rxFromSE(.dd)
  }, character(1))
}

## Auto-build the ephemeral augmented model from a base nn() ui: drop the weight
## dummy-covariate line and the error line, rename the latent eta(s) to
## covariates, declare param(weights, latent covariates), and hand to
## nnAugmentModel().  Returns the compiled augmented model + the metadata the
## weight step needs.
.nnAugmentFromUi <- function(ui) {
  .reg <- .nnEnv$reg
  if (length(.reg) == 0L) {
    stop("nlmixr2nn: no nn() term found in the model", call. = FALSE)
  }
  ## networks ordered by id -> a stable GLOBAL weight layout (net 0's weights, then
  ## net 1's, ...), matching nnAugmentModel's rx_sw_<state>_<globalj>_ indexing.
  .nets <- .reg[order(vapply(.reg, function(m) m$id, integer(1)))]
  .lines <- ui$lstChr
  .end <- .nnErrEndpoint(.lines)
  if (is.null(.end)) {
    stop("nlmixr2nn currently supports a single additive endpoint (var ~ add(sd))",
         call. = FALSE)
  }
  ## For a normal endpoint the network's effect is chained through the
  ## PREDICTION.  For a count endpoint the prediction is the log-density, which
  ## the solve cannot form (it needs DV), so the chain runs through the
  ## distribution's own parameter instead -- `lam` in `y ~ pois(lam)`.  Only the
  ## target variable changes; everything below is the same sensitivity.
  .dist <- .nnDistInfo(tryCatch(ui$predDf, error = function(e) NULL))
  if (!is.null(.dist) && !isTRUE(.dist$supported)) {
    stop("nlmixr2nn cannot form a weight gradient for a '", .dist$dist,
         "' endpoint: it has no closed-form score here, and the likelihood hook ",
         "reports d(LL)/d(f) = 1 for every non-normal endpoint (f is the ",
         "log-density itself), which is not a quantity the augmented solve can ",
         "differentiate.  Supported: ",
         paste(names(.nnDistFuns), collapse = ", "), ", and any normal endpoint.",
         call. = FALSE)
  }
  .target <- if (is.null(.dist)) .end$state else .dist$target
  ## drop the weight dummy-covariate declaration(s) and the error line(s)
  .keep <- .lines[!grepl("^\\s*rx_nnw[0-9]+_\\s*<-", .lines) & !grepl("~", .lines)]
  ## latent etas among the nn inputs -> covariate names (dots -> underscores)
  .etas <- ui$eta
  .covMap <- stats::setNames(gsub("[^A-Za-z0-9_]", "_", .etas), .etas)
  for (.e in .etas) {
    .keep <- gsub(paste0("\\b", gsub("\\.", "\\\\.", .e), "\\b"), .covMap[[.e]], .keep)
  }
  ## all networks' weights + the model's non-weight covariates (also NN inputs, e.g.
  ## WT in `nn(WT, eta.nn)`) + the latent-eta covariates, declared so the augmented
  ## solve reads them from the data.
  .allW <- unlist(lapply(.nets, function(m) m$weights), use.names = FALSE)
  .realCovs <- setdiff(ui$allCovs, .allW)
  .param <- paste0("param(",
                   paste(c(.allW, .realCovs, unname(.covMap)), collapse = ", "), ")")
  .augBase <- paste(c(.param, .keep), collapse = "\n")
  .H <- stats::setNames(vapply(.nets, function(m) as.integer(m$H), integer(1)),
                        vapply(.nets, function(m) as.character(m$id), character(1)))
  .augText <- nnAugmentModel(.augBase, H = .H)
  .totW <- sum(vapply(.nets, function(m) as.integer(m$H * m$K + 2L * m$H + 1L), integer(1)))
  ## prediction forward sensitivity wrt each GLOBAL weight, chained through the
  ## states: rx_predsw_<globalj>_ = sum_s d(pred)/d(s) * rx_sw_<s>_<globalj>_.
  .states <- rxode2::rxStateOde(rxode2::rxS(rxode2::rxGetModel(.augBase), TRUE,
                                            promoteLinSens = FALSE))
  .dpds <- .nnDpDs(.augBase, .states, .target)
  ## The prediction can depend on the network TWO ways, and both must be in the
  ## sensitivity or the gradient is silently wrong:
  ##
  ##   through the states   sum_s d(pred)/d(s) * ds/dw      <- the usual route
  ##   directly             d(pred)/dg * dg/dw              <- e.g. y <- nn(centr)
  ##
  ## The direct term is zero for the common model, where the network appears only
  ## in a d/dt() -- which is why omitting it went unnoticed.  When the network
  ## feeds the prediction itself, the state route carries NONE of the effect and
  ## the assembled gradient came out exactly zero, so such a model did not train
  ## at all and said nothing.
  .symBase <- rxode2::rxS(rxode2::rxGetModel(.augBase), TRUE, promoteLinSens = FALSE)
  .calls <- .nnParseCallAll(.augBase)
  .dpdg <- stats::setNames(
    vapply(.calls, function(.c) .nnDvarDg(.symBase, .target, .c), character(1)),
    vapply(.calls, function(.c) as.character(.c$id), character(1)))
  .anyDirect <- any(.dpdg != "0" & nzchar(.dpdg))
  if (all(.dpds == "0") && !.anyDirect) {
    stop("nlmixr2nn: the prediction '", .target,
         "' does not depend on any ODE state, nor on the network directly -- ",
         "nothing for the network to fit", call. = FALSE)
  }
  ## The weight layout is built from the REGISTRY (.nets) while the sensitivity
  ## states are built from the parsed CALLS, so the two must describe the same
  ## set of networks.  When they disagree the failure is obscure: a registry
  ## entry with no call leaves .ownerOf short, and indexing a named vector past
  ## its end raises "subscript out of bounds"; two calls sharing an id make
  ## nnAugmentModel() emit twice the variational states the layout accounts for,
  ## and the extra ones are silently dropped from the gradient.
  ##
  ## Neither is reachable through nn() -- every nn() gets its own id from
  ## rxUdfUiNum() -- but hand-written model text can produce both, so say so
  ## rather than failing obliquely later.
  .callIds <- vapply(.calls, function(.c) .c$id, integer(1))
  .netIds <- vapply(.nets, function(.m) as.integer(.m$id), integer(1))
  if (anyDuplicated(.callIds)) {
    stop("nlmixr2nn: network id ",
         paste(unique(.callIds[duplicated(.callIds)]), collapse = ", "),
         " is called more than once in the model; a network must appear once, ",
         "because its weight sensitivities are laid out per network, not per call",
         call. = FALSE)
  }
  if (!setequal(.callIds, .netIds)) {
    stop("nlmixr2nn: the registered networks (", paste(sort(.netIds), collapse = ", "),
         ") do not match the ones the model calls (", paste(sort(.callIds), collapse = ", "),
         ")", call. = FALSE)
  }
  ## The input dimension has to agree too, and for the same reason: the weight
  ## layout and the torch module size come from the REGISTRY's K, while the
  ## variational states and the nnWg<K> arity come from the CALL's K (nn2 -> 2).
  ## Disagreement desynchronises the layout -- too few registered inputs and the
  ## compiled nnWg<K> strides past the end of the weight buffer, which is a
  ## silent over-read rather than an error.
  for (.c in .calls) {
    .m <- Filter(function(.x) .x$id == .c$id, .nets)[[1L]]
    if (as.integer(.m$K) != as.integer(.c$K)) {
      stop("nlmixr2nn: network ", .c$id, " is registered with ", .m$K,
           " input(s) but the model calls nn", .c$K, "() with ", .c$K,
           call. = FALSE)
    }
  }
  ## global weight index -> which network it belongs to, and its local index
  .ownerOf <- integer(0); .localOf <- integer(0)
  for (.c in .calls) {
    .nWc <- as.integer(.hOfNet(.nets, .c$id) * .c$K + 2L * .hOfNet(.nets, .c$id) + 1L)
    .ownerOf <- c(.ownerOf, rep(.c$id, .nWc))
    .localOf <- c(.localOf, seq_len(.nWc) - 1L)
  }
  .predsw <- vapply(seq_len(.totW) - 1L, function(j) {
    .terms <- character(0)
    for (.si in seq_along(.states)) {
      if (!identical(.dpds[[.si]], "0") && nzchar(.dpds[[.si]])) {
        .terms <- c(.terms, sprintf("(%s)*rx_sw_%s_%d_", .dpds[[.si]], .states[.si], j))
      }
    }
    ## the direct term, for the network this weight belongs to
    .id <- .ownerOf[j + 1L]
    .d <- .dpdg[[as.character(.id)]]
    if (!is.null(.d) && !identical(.d, "0") && nzchar(.d)) {
      .cj <- Filter(function(.c) .c$id == .id, .calls)[[1L]]
      .terms <- c(.terms, sprintf("(%s)*nnWg%d(%d, %d, %s)", .d, .cj$K, .id,
                                  .localOf[j + 1L], paste(.cj$inputs, collapse = ", ")))
    }
    if (length(.terms) == 0L) .terms <- "0"
    sprintf("rx_predsw_%d_ = %s", j, paste(.terms, collapse = " + "))
  }, character(1))
  .augText <- paste(c(.augText, .predsw), collapse = "\n")
  .mAugBase <- rxode2::rxode2(.augBase)
  ## per-network metadata carrying the global-weight offset + the network's weight
  ## base in the augmented base model (each net's block is contiguous there).
  .off <- 0L
  .netMeta <- lapply(.nets, function(m) {
    .nWm <- as.integer(m$H * m$K + 2L * m$H + 1L)
    .meta <- list(id = m$id, K = m$K, H = m$H, act = m$act, weights = m$weights, nW = .nWm,
                  offset = .off, gIdx = .off + seq_len(.nWm),   # 1-based global indices
                  augBase = .nnWeightBase(.mAugBase, m$id, m$K, m$H),
                  predswCols = sprintf("rx_predsw_%d_", .off + seq_len(.nWm) - 1L))
    .off <<- .off + .nWm
    .meta
  })
  ## the endpoint's transformation, read from the ui's own predDf rather than
  ## re-parsed from the model text -- the weight step needs its derivative to
  ## chain a transformed-scale score onto natural-scale sensitivities
  .pd <- tryCatch(ui$predDf, error = function(e) NULL)
  .ep <- if (is.null(.pd) || !nrow(.pd)) {
    list(transform = "untransformed", lambda = 1, trLow = 0, trHi = 1)
  } else {
    list(transform = as.character(.pd$transform[1L]),
         lambda = .pd$lambda[1L], trLow = .pd$trLow[1L], trHi = .pd$trHi[1L])
  }
  list(text = .augText, mAug = rxode2::rxode2(.augText),
       covMap = .covMap, realCovs = .realCovs, weights = .allW, nW = .totW,
       endpoint = .target, errAdd = .end$add, errProp = .end$prop, ep = .ep,
       dist = .dist,
       predswCols = sprintf("rx_predsw_%d_", seq_len(.totW) - 1L),
       nets = .netMeta)
}
