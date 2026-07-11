#' Probe the current solve parameter vector (development/validation only)
#'
#' `nnprobe(idx, x)` returns `par_ptr[idx]` of the first subject in the
#' current rxode2 solve; `nnnpars(x)` returns the number of parameters.  These
#' validate that a package-exported custom function can read the solve's
#' parameter vector, which the neural-network functions rely on.
#'
#' @param idx zero-based parameter index.
#' @param x ignored placeholder argument.
#' @return numeric vector.
#' @export
nnprobe <- function(idx, x = 0) {
  df <- data.frame(idx = idx, x = x)
  .Call(`_rxode2nn_nnprobe`, as.double(df$idx), as.double(df$x))
}

#' @rdname nnprobe
#' @export
nnnpars <- function(x = 0) {
  .Call(`_rxode2nn_nnnpars`, as.double(x))
}
