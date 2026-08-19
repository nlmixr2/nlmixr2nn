## The engine's phases, exercised apart from a fit.
##
## The point of threading a ctx instead of a 400-line closure is that a phase can
## be driven on its own.  If that is only true in principle it will stop being
## true, so it is used here.

test_that("the ctx a phase reads is the ctx .nnRunCtx builds", {
  ## A phase reading a field nobody sets gets NULL, and NULL is a plausible value
  ## for most of these -- no error, just a quietly different fit.  The two sides
  ## of the contract are checked against each other rather than against a list
  ## written out by hand, which would just be a third thing to keep in sync.
  ns <- asNamespace("nlmixr2nn")
  .b <- body(get(".nnRunCtx", ns))
  built <- names(as.list(.b[[length(.b)]]))[-1L]
  expect_true(length(built) > 10L)

  .fieldsRead <- function(.p) {
    .txt <- paste(deparse(get(.p, ns)), collapse = " ")
    unique(sub("^ctx\\$", "",
               regmatches(.txt, gregexpr("ctx\\$[A-Za-z0-9._]+", .txt))[[1L]]))
  }
  .phases <- c(".nnRun", ".nnRunPopFit", ".nnRunWarmStarts", ".nnRunLoop")
  .all <- unique(unlist(lapply(.phases, .fieldsRead)))
  ## the regex has to actually find something, or the setdiff below passes for
  ## the wrong reason
  expect_gt(length(.all), 10L)
  ## nothing read that is not built ...
  expect_setequal(setdiff(.all, built), character(0))
  ## ... and nothing built that nobody reads: a dead ctx field is a dependency
  ## someone removed without saying so
  expect_setequal(setdiff(built, .all), character(0))
})

test_that("the finalizer attaches the network, driven from a synthetic ctx", {
  skip_if_not_installed("rxode2")
  local_nn()
  m <- function() {
    ini({ lk <- -1; add.sd <- 0.3 })
    model({
      k <- exp(lk)
      d/dt(centr) <- -k * centr + nn(centr, n_hidden = 2L, act = "tanh")
      centr ~ add(add.sd)
    })
  }
  set.seed(3)
  ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
  aug <- .nnAugmentFromUi(rxode2::rxUiDecompress(ui))
  ## the whole ctx this phase needs -- no fit, no torch, no solve
  ctx <- list(aug = aug, sched = nnControl())
  fit <- list(env = local({ e <- new.env(); assign("ui", ui, envir = e); e }))

  w <- seq_along(aug$weights) / 10
  hist <- data.frame(round = 1L, objf = 12.5)
  out <- .nnStoreNnFit(ctx, fit, w, hist, converged = TRUE, nRun = 1L)

  e <- out$env
  expect_equal(get("nnWeights", envir = e), stats::setNames(w, aug$weights))
  expect_equal(get("nnParHist", envir = e), hist)
  expect_true(get("nnConverged", envir = e))
  expect_equal(get("nnRounds", envir = e), 1L)
  ## the inferred schedule travels with the fit, since the user never wrote it
  expect_equal(get("nnSched", envir = e), ctx$sched)

  ## and the stored ui is self-contained: values in forcedPars, shapes in nnMeta,
  ## both sticky, so a fresh session can stride the weight block
  su <- rxode2::rxUiDecompress(get("ui", envir = e))
  expect_equal(rxode2::rxForcedPars(su)[aug$weights], stats::setNames(w, aug$weights))
  expect_true(exists("nnMeta", envir = su, inherits = FALSE))
  expect_true(get("nnTrained", envir = su, inherits = FALSE))
  expect_true(all(c("nnMeta", "nnTrained") %in% get("sticky", envir = su, inherits = FALSE)))
})

test_that("both branches finalize through the one writer", {
  ## The population branch used to carry its own copy that skipped the nnMeta
  ## snapshot -- invisible in session, fatal on reload.  Neither branch may grow
  ## a private copy again.
  ns <- asNamespace("nlmixr2nn")
  for (.f in c(".nnRun", ".nnRunPopFit")) {
    expect_true(any(grepl(".nnStoreNnFit", deparse(get(.f, ns)), fixed = TRUE)))
  }
  ## and nothing else writes the fit-env keys the finalizer owns
  .writers <- Filter(function(.n) {
    .o <- get(.n, ns)
    is.function(.o) && any(grepl("\"nnWeights\"", deparse(.o), fixed = TRUE))
  }, ls(ns, all.names = TRUE))
  expect_equal(.writers, ".nnStoreNnFit")
})
