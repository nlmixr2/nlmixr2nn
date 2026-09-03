## Joint DeepPumas-style fit by ALTERNATING: (a) fast analytic FOCEi on the base
## latent-input model for the residual-error + latent Omega + per-subject EBEs, then
## (b) one augmented (forward-sensitivity) solve at those EBEs for a torch weight
## step on the POPULATION network.  Data is Michaelis-Menten elimination with IIV in
## Vmax.  Over the rounds the torch-trained network learns the (nonlinear) population
## shape (residual -> the true noise) while the latent eta recovers the individual
## variation (EBEs -> perfectly correlated with the true per-subject etas).  All
## gradients are analytic/torch -- no finite differences.

test_that("alternating torch-weights + FOCEi-latent-Omega recovers population + IIV", {
  skip_on_cran()
  skip_if_not_installed("rxode2")
  .nnLoaderOn(); on.exit(.nnLoaderOff(), add = TRUE)  # nn par-loader active for direct nn-model solves
  skip_if_no_torch()
  .old <- rxode2::getRxThreads(); on.exit(rxode2::setRxThreads(.old), add = TRUE)
  on.exit({ nnClearMeta(); try(nnTorchFree(0L), silent = TRUE) }, add = TRUE)
  ## this test drives its OWN manual block-coordinate loop (plain fixed-weight FOCEi
  ## fits), so disable the transparent-training interceptor for the duration.
  nlmixr2est::removeEstInterceptor("nlmixr2nn")
  on.exit(nlmixr2nn:::.nnRegisterInterceptor(), add = TRUE)
  rxode2::setRxThreads(1L)

  set.seed(1)
  Vmax <- 2.0; Km <- 3.0; ns <- 8L
  etaTrue <- rnorm(ns, 0, sqrt(0.15))
  mm <- rxode2::rxode2("d/dt(centr) = -(Vmax*exp(eV))*centr/(Km+centr)")
  data <- do.call(rbind, lapply(1:ns, function(id) {
    s <- rxode2::rxSolve(mm, data.frame(id = id, time = c(0, 0.5, 1, 2, 4, 6, 8, 10),
           evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
           params = c(Vmax = Vmax, Km = Km, eV = etaTrue[id]), returnType = "data.frame")
    s <- s[s$time > 0, ]
    data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
               amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr + rnorm(nrow(s), 0, 0.1)))
  }))

  K <- 2L; H <- 3L; nW <- H * K + 2L * H + 1L; wnm <- nnWeightLayout(0L, K, H)
  nnClearMeta()
  modF <- function() {
    ini({ add.sd <- 0.3; eta.nn ~ 0.2 })
    model({ g <- nn(centr, eta.nn, nHidden = 3L, act = "tanh")
            d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
            centr ~ add(add.sd) })
  }
  uiF <- rxode2::rxode2(modF)
  dataF <- nnCovData(data)
  nnUpdate(uiF)
  augTxt <- paste0("param(", paste(c(wnm, "etann"), collapse = ","), ")\n",
                   "g = nn2(0, centr, etann)\n",
                   "d/dt(centr) = -(1.0/(1.0+exp(-g)))*centr\n")
  mAug <- rxode2::rxode2(nnAugmentModel(augTxt, H = H))
  swCols <- sprintf("rx_sw_centr_%d_", seq_len(nW) - 1L)
  nnSetMeta(0L, base = .nnWeightBase(rxode2::rxode2(augTxt), 0L, K, H), K = K, H = H, act = "tanh")
  nnTorchInit(0L, K, H, act = "tanh", seed = 5); nnTorchOptInit(0L, "adam", 0.03)

  obsIdx <- data$evid == 0
  weightStep <- function(ebes, sigma) {
    ad <- nnCovData(data); ad$etann <- ebes[as.character(ad$id)]
    nnSetWeights(0L, nnTorchWeights(0L))
    s <- rxode2::rxSolve(mAug, ad, params = setNames(rep(0, nW), wnm), returnType = "data.frame")
    ik <- match(paste(data$id[obsIdx], data$time[obsIdx]), paste(s$id, s$time))
    resid <- data$dv[obsIdx] - s$centr[ik]; dLLdf <- resid / sigma^2
    dLLdw <- vapply(swCols, function(cn) sum(dLLdf * s[[cn]][ik]), numeric(1))
    nnTorchZeroGrad(0L); nnTorchSetGrad(0L, -dLLdw); nnTorchStep(0L)
    sqrt(mean(resid^2))
  }

  addsd <- corr <- numeric(0)
  for (round in 1:5) {
    nnSetWeights(0L, nnTorchWeights(0L))
    f <- suppressWarnings(suppressMessages(nlmixr2est::nlmixr2(modF, dataF, "focei",
          nlmixr2est::foceiControl(print = 0L, maxOuterIterations = 8L,
                                   maxInnerIterations = 25L, calcTables = FALSE))))
    ebes <- setNames(f$eta$eta.nn, f$eta$ID)
    for (ws in 1:8) weightStep(ebes, f$theta[["add.sd"]])
    addsd <- c(addsd, f$theta[["add.sd"]])
    corr <- c(corr, cor(ebes[order(as.integer(names(ebes)))], etaTrue))
  }

  ## the torch-trained NN drives the residual down toward the true noise, and the
  ## latent eta recovers the individual Vmax variation (|corr| -> ~1).
  expect_lt(addsd[length(addsd)], addsd[1] / 2)
  expect_gt(abs(corr[length(corr)]), 0.8)
})
