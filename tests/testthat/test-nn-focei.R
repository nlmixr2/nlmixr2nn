## End-to-end: an nn() model fits under FOCEI.  The covariate-declared weight
## block (loader-injected) assembles cleanly through FOCEI's theta expansion --
## the fix that unblocked the native path (param()-declared weights were mangled).

test_that("an nn() model assembles and fits under FOCEI", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  nnClearMeta(); on.exit(nnClearMeta(), add = TRUE)

  set.seed(1)
  d <- do.call(rbind, lapply(1:6, function(id) {
    t <- c(1, 2, 4, 6, 8)
    data.frame(ID = id, TIME = c(0, t), EVID = c(1, rep(0, 5)),
               AMT = c(10, rep(0, 5)), DV = c(NA, 9 * exp(-0.25 * t) + rnorm(5, 0, 0.2)),
               CMT = 1)
  }))

  mod <- function() {
    ini({ tk <- 0.25; add.sd <- 0.5 })
    model({
      g <- nn(centr, n_hidden = 1L, act = "tanh")
      d/dt(centr) <- -(tk + g) * centr
      centr ~ add(add.sd)
    })
  }
  ui <- rxode2::rxode2(mod)
  d <- nnCovData(d)                       # weight covariate placeholder columns
  info <- nnUpdate(ui)                    # resolve base + register the layer
  expect_equal(nrow(info), 1L)
  nnSetWeights(0L, c(0.3, -0.2, 0.5, 0.1))

  f <- suppressWarnings(suppressMessages(
    nlmixr2est::nlmixr2(mod, d, est = "focei",
      control = nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 3L,
                                         maxInnerIterations = 4L, calcTables = FALSE))))
  expect_true(is.finite(f$objf))          # FOCEI assembled + ran the nn() model
})
