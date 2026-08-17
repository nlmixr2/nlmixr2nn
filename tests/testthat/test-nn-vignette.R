## The user vignette must not need a helper call.
##
## The whole claim of this package is that a model containing nn() is an
## ordinary rxode2/nlmixr2 model.  A setup call creeping back into the
## documentation is the first sign that claim has quietly stopped being true, so
## it is checked rather than remembered.
##
## The INTERNALS article is deliberately exempt: describing those functions is
## its purpose.

.vignetteDir <- function() {
  ## works from tests/testthat in a source tree and in an unpacked tarball
  for (p in c("../../vignettes", "../vignettes", "vignettes")) {
    if (dir.exists(p)) return(p)
  }
  NULL
}

test_that("the user vignette calls no setup helper", {
  vd <- .vignetteDir()
  skip_if(is.null(vd), "vignettes not available")
  f <- file.path(vd, "nlmixr2nn.Rmd")
  skip_if_not(file.exists(f), "user vignette not found")

  txt <- readLines(f, warn = FALSE)
  ## the backend probe is a knit-time guard, not a workflow call
  txt <- txt[!grepl("_nlmixr2nn_nnTorchAvailable", txt)]

  banned <- c("nnTorchModel", "nnTorchInit", "nnTorchSetWeights", "nnUpdate",
              "nnCovData", "nnClearMeta", "nnWithLoader", "nnSetMeta",
              "nnSetWeights", "nnAugmentModel", "nnTrain")
  for (b in banned) {
    hit <- grep(paste0("\\b", b, "\\b"), txt, value = TRUE)
    expect_length(hit, 0L)
  }
})

test_that("both documents exist, and the internals one is not the user one", {
  vd <- .vignetteDir()
  skip_if(is.null(vd), "vignettes not available")
  expect_true(file.exists(file.path(vd, "nlmixr2nn.Rmd")))
  expect_true(file.exists(file.path(vd, "nlmixr2nn-internals.Rmd")))

  ## the internals article is exempt from the grep above precisely because it
  ## documents that machinery -- assert it actually does, so the exemption is
  ## earned rather than assumed
  int <- readLines(file.path(vd, "nlmixr2nn-internals.Rmd"), warn = FALSE)
  expect_true(any(grepl("nnSetMeta|par-loader|cotangent", int)))
})
