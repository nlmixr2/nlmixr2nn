## `x`, `value` and `group` are columns of the data.frame built inside
## plot.nlmixr2nnEval() and referenced by ggplot2's aes(), which R CMD check
## cannot see through.  Declaring them is preferable to the .data pronoun here,
## which would pull rlang in for a package that only Suggests ggplot2.
utils::globalVariables(c("x", "value", "group"))

## Looking at what a network learned.
##
## After a fit the first question is always "what shape did it find?", and the
## only way to answer it before was to hand-build a grid, solve the fitted model
## over it, and pull a column out.  `nnEval()` asks the network directly, using
## the same compiled activations the ODE right-hand side uses, so what you plot
## is what the model integrates.

## The network metadata carried by a model or fit, as a named list keyed by id.
.nnMetaOf <- function(x) {
  .ui <- if (inherits(x, "rxUi")) x else tryCatch(x$finalUi, error = function(e) NULL)
  if (!inherits(.ui, "rxUi")) .ui <- x
  .ui <- tryCatch(rxode2::rxUiDecompress(.ui), error = function(e) .ui)
  if (!is.environment(.ui)) return(NULL)
  tryCatch(get("nnMeta", envir = .ui, inherits = FALSE), error = function(e) NULL)
}

#' Evaluate a model's neural network at chosen inputs
#'
#' Evaluates the network embedded by [nn()] directly, at the weights the model
#' currently carries -- the trained weights of a fit, or the initial values drawn
#' when the model was parsed.  Use it to see the function the network represents.
#'
#' The inputs are supplied by name, in the order they appear in the `nn()` call.
#' Vectors of unequal length are recycled against each other, so a one-input
#' network takes a single sequence and a two-input network can be swept over a
#' grid built with [expand.grid()].
#'
#' @param object an `rxUi` model or an `nlmixr2` fit containing an [nn()] term.
#'   Named `object` rather than `x` deliberately: the inputs are supplied by
#'   name through `...`, and a network input called `x` is entirely ordinary --
#'   it would otherwise be captured by the first formal instead.
#' @param ... the network inputs, given as named vectors (e.g.
#'   `central = seq(0, 10, 0.1)`).  Names are for labelling only; arguments are
#'   matched to the network's inputs positionally.
#' @param net which network to evaluate, for a model containing more than one
#'   `nn()` term (the id shown in the model text, counting from 0).
#' @return a `data.frame` of class `"nlmixr2nnEval"` with one column per input
#'   and a `value` column holding the network output.
#' @examples
#' \donttest{
#' mod <- function() {
#'   model({
#'     d/dt(central) <- -nn(central, nHidden = 4L)
#'   })
#' }
#' set.seed(1)
#' ui <- rxode2::rxode2(mod)
#' head(nnEval(ui, central = seq(0, 10, length.out = 5)))
#' }
#' @export
#' @author Matthew L. Fidler
nnEval <- function(object, ..., net = 0L) {
  if (missing(object)) {
    stop("nnEval(): give the model or fit as the first argument", call. = FALSE)
  }
  .meta <- .nnMetaOf(object)
  if (is.null(.meta) || length(.meta) == 0L) {
    stop("nnEval(): this model carries no nn() network", call. = FALSE)
  }
  .key <- as.character(as.integer(net))
  if (!(.key %in% names(.meta))) {
    stop("nnEval(): no network ", .key, " in this model (have: ",
         paste(names(.meta), collapse = ", "), ")", call. = FALSE)
  }
  .m <- .meta[[.key]]
  .w <- nnWeights(object)
  if (is.null(.w)) {
    stop("nnEval(): this model carries no network weights", call. = FALSE)
  }
  .w <- .w[.m$weights]
  if (anyNA(.w)) {
    stop("nnEval(): network ", .key, " is missing some of its weights", call. = FALSE)
  }

  .in <- list(...)
  if (length(.in) != .m$K) {
    stop("nnEval(): network ", .key, " takes ", .m$K, " input",
         if (.m$K > 1L) "s" else "", ", but ", length(.in), " ",
         if (length(.in) == 1L) "was" else "were", " given", call. = FALSE)
  }
  .n <- max(vapply(.in, length, integer(1)))
  .in <- lapply(.in, function(v) rep_len(as.numeric(v), .n))
  .X <- matrix(unlist(.in, use.names = FALSE), nrow = .n, ncol = .m$K)

  .val <- .Call(`_nlmixr2nn_nnForwardW`, as.integer(.m$K), as.integer(.m$H),
                .nnActCode[[.m$act]], as.double(unname(.w)), .X)

  .nm <- names(.in)
  if (is.null(.nm) || any(!nzchar(.nm))) {
    .nm <- paste0("x", seq_len(.m$K))
  }
  .out <- as.data.frame(stats::setNames(.in, .nm))
  .out$value <- .val
  attr(.out, "nnNet") <- .m$id
  attr(.out, "nnAct") <- .m$act
  attr(.out, "nnInputs") <- .nm
  class(.out) <- c("nlmixr2nnEval", "data.frame")
  .out
}

#' @export
print.nlmixr2nnEval <- function(x, ...) {
  cat(sprintf("nn() network %s (%s, %d input%s) evaluated at %d point%s\n",
              attr(x, "nnNet"), attr(x, "nnAct"), length(attr(x, "nnInputs")),
              if (length(attr(x, "nnInputs")) > 1L) "s" else "",
              nrow(x), if (nrow(x) > 1L) "s" else ""))
  print(utils::head(as.data.frame(x), 6L))
  if (nrow(x) > 6L) cat("...\n")
  invisible(x)
}

#' Plot a network's learned function
#'
#' @param x an `nnEval()` result.
#' @param true optional reference to overlay: a function of the first input, or a
#'   numeric vector aligned with the rows of `x`.  Use it to compare a learned
#'   term against the mechanism that generated the data.
#' @param ... passed to the underlying plotting call.
#' @return a `ggplot` object when ggplot2 is available, otherwise `NULL`
#'   invisibly (the plot is drawn).
#' @export
#' @author Matthew L. Fidler
plot.nlmixr2nnEval <- function(x, true = NULL, ...) {
  .inputs <- attr(x, "nnInputs")
  .xv <- x[[.inputs[1L]]]
  .df <- data.frame(x = .xv, value = x$value)
  .lab <- .inputs[1L]
  .tv <- NULL
  if (!is.null(true)) {
    .tv <- if (is.function(true)) true(.xv) else rep_len(as.numeric(true), nrow(x))
  }
  ## a two-input network drawn against its first input is misleading unless the
  ## second is held fixed, so say so rather than silently drawing spaghetti
  if (length(.inputs) > 1L) {
    .grp <- interaction(x[.inputs[-1L]], drop = TRUE)
    .df$group <- .grp
  }
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    ## columns are named literally "x"/"value"/"group" above, so plain aes()
    ## suffices -- no rlang/.data pronoun, keeping ggplot2 a pure Suggests
    .p <- ggplot2::ggplot(.df, ggplot2::aes(x = x, y = value))
    .p <- .p + if (length(.inputs) > 1L) {
      ggplot2::geom_line(ggplot2::aes(colour = group))
    } else {
      ggplot2::geom_line()
    }
    if (!is.null(.tv)) {
      .p <- .p + ggplot2::geom_line(data = data.frame(x = .xv, value = .tv),
                                    linetype = 2)
    }
    .p <- .p + ggplot2::labs(x = .lab, y = "nn() output",
                             colour = paste(.inputs[-1L], collapse = ", ")) +
      ggplot2::theme_bw()
    return(.p)
  }
  .o <- order(.xv)
  graphics::plot(.xv[.o], x$value[.o], type = "l", xlab = .lab,
                 ylab = "nn() output", ...)
  if (!is.null(.tv)) graphics::lines(.xv[.o], .tv[.o], lty = 2)
  invisible(NULL)
}
