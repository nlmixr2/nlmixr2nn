## The nn weight par-loader is registered NAMED ("nlmixr2nn:nnParLoader") and runs
## only while it is the active injector -- so registering an nn network must NOT
## cause its weights to leak into an unrelated model's par_ptr on a later solve.

test_that("the nn par-loader never clobbers an unrelated model's parameters", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  nnClearMeta()

  ## register an nn network with a LARGE weight block + obvious sentinel weights
  mnn <- function() {
    ini({ add.sd <- 0.3 })
    model({ g <- nn(centr, tk, nHidden = 10L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  ui <- rxode2::rxode2(mnn)
  nnUpdate(ui)
  nnSetWeights(0L, rep(999, length(nnWeightLayout(0L, 2L, 10L))))

  ## an UNRELATED model with as many parameters as the weight block -- solving it
  ## must leave every parameter untouched (the loader is NOT the active injector).
  m2 <- rxode2::rxode2(paste0("d/dt(a) = -p1*a\n",
                              paste(sprintf("e%d <- p%d", 1:40, 2:41), collapse = "\n")))
  p <- stats::setNames(seq_len(41) / 10, paste0("p", 1:41))
  s <- rxode2::rxSolve(m2, data.frame(time = c(0, 1), evid = c(1, 0), cmt = 1, amt = c(10, 0)),
                       params = p, returnType = "data.frame")
  elhs <- vapply(1:40, function(i) s[[paste0("e", i)]][1], numeric(1))
  expect_equal(elhs, (2:41) / 10, tolerance = 1e-8)   # NOT the 999 sentinel
  expect_false(any(abs(elhs - 999) < 1e-6))

  ## and the model does not carry a parLoader flag
  expect_null(rxode2::rxParLoader(rxode2::rxode2("d/dt(a) = -k*a")))
})

test_that("nnWithLoader activates the nn injector for a direct nn-model solve", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  skip_if_no_torch()
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  nnClearMeta()

  mod <- function() { model({ y <- nn(x, nHidden = 4L, act = "tanh"); d/dt(A) <- -A }) }
  set.seed(1)
  ui <- suppressMessages(rxode2::rxode2(mod))
  ev <- data.frame(id = 1, time = 0, x = 0.7, amt = 0, evid = 0)
  y0 <- rxode2::rxSolve(ui, ev, returnType = "data.frame")$y[1]

  ## A model that carries its own weights is self-describing, and the injector
  ## must leave it alone even when it is armed: after a reload the loader buffer
  ## is empty, so letting it win would silently zero the network.
  meta <- rxode2::rxUiDecompress(ui)$nnMeta[["0"]]
  nnTorchInit(0L, meta$K, meta$H, act = meta$act)
  nnTorchSetWeights(0L, unname(nnWeights(ui)) + 1)   # deliberately different
  nnSetWeights(0L, nnTorchWeights(0L))
  yOn <- nnWithLoader(rxode2::rxSolve(ui, ev, returnType = "data.frame"))$y[1]
  expect_equal(yOn, y0, tolerance = 1e-10)
  expect_false(isTRUE(all.equal(yOn, nnTorchForward(0L, 0.7))))
})
