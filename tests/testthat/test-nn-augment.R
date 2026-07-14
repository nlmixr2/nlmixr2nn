## B3: forward-sensitivity augmented NN model (nnAugmentModel).
## The augmented model's variational states rx_sw_<state>_<j>_ integrate
##   d/dt(s_ij) = sum_k F_X[i,k] s_kj + (dR_i/dg) nnWg(id, j, inputs)
## and so equal d(state)/d(w_j) at solve end.  Validated against a finite
## difference of the base states wrt each weight.

test_that("nnAugmentModel: variational states equal d(state)/d(weight) (FD check)", {
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  ok <- tryCatch(isTRUE(.Call("_nlmixr2nn_nnTorchAvailable")), error = function(e) FALSE)
  nnClearMeta(); nnSetMeta(0L, base = 0L, K = 2L, H = 1L, act = "tanh")  # nW = 5
  on.exit(nnClearMeta(), add = TRUE)
  ## reserve the weight block (base = 0) via param() so the loader-injected
  ## weights occupy dedicated par_ptr slots (indices 0..4) ahead of k.
  wnm <- nnWeightLayout(0L, K = 2L, H = 1L)
  obj <- paste0(
    sprintf("param(%s)\n", paste(wnm, collapse = ", ")),
    "g = nn2(0, centr, peri)\nd/dt(centr) = -g*centr\nd/dt(peri) = g*centr - k*peri")
  aug <- nnAugmentModel(obj, H = 1L)
  expect_equal(length(gregexpr("d/dt(rx_sw_", aug, fixed = TRUE)[[1]]), 10L)  # 2 states * 5 weights

  w  <- c(0.3, -0.4, 0.5, 0.2, -0.1)      # fixed weights (nW = 5)
  ## weight params need placeholder values at solve setup; the par-loader
  ## overwrites them with the nnSetWeights() buffer before integration.
  p  <- c(k = 0.3, setNames(rep(0, length(wnm)), wnm)); ic <- c(centr = 10, peri = 0)
  ev <- rxode2::et(seq(0, 4, by = 1))
  mAug  <- rxode2::rxode2(aug)
  mBase <- rxode2::rxode2(obj)

  nnSetWeights(0L, w)
  sAug <- rxode2::rxSolve(mAug, ev, params = p, inits = ic)

  ## dR/dg outputs
  expect_equal(sAug$rx_drdg_centr_, -sAug$centr, tolerance = 1e-8)
  expect_equal(sAug$rx_drdg_peri_,   sAug$centr, tolerance = 1e-8)

  ## df/dw vs central finite difference of the base states wrt each weight
  h <- 1e-5
  for (j in c(0L, 2L, 4L)) {
    wp <- w; wp[j + 1L] <- wp[j + 1L] + h
    wm <- w; wm[j + 1L] <- wm[j + 1L] - h
    nnSetWeights(0L, wp); sp <- rxode2::rxSolve(mBase, ev, params = p, inits = ic)
    nnSetWeights(0L, wm); sm <- rxode2::rxSolve(mBase, ev, params = p, inits = ic)
    fdCentr <- (sp$centr - sm$centr) / (2 * h)
    fdPeri  <- (sp$peri  - sm$peri)  / (2 * h)
    expect_equal(sAug[[sprintf("rx_sw_centr_%d_", j)]], fdCentr, tolerance = 1e-4,
                 info = paste("weight", j))
    expect_equal(sAug[[sprintf("rx_sw_peri_%d_", j)]],  fdPeri,  tolerance = 1e-4,
                 info = paste("weight", j))
  }
})
