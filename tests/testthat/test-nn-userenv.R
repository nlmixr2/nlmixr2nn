## Things that only break for a USER.
##
## The rest of the suite runs with `env = new.env(parent = asNamespace())`, so
## package internals are visible to it.  A user has only the package attached.
## That difference hid a real bug: unexporting the generated `nn<K>` family
## looked safe -- compiled model code reaches it through R_RegisterCCallable --
## but rxode2's symengine renderer resolves a model function BY NAME with
## `get(fun, ...)` over the search path when it builds derivatives.  With `nn1`
## internal, every model needing a Jacobian died with "function 'nn1' or its
## derivatives are not supported in rxode2", and nothing in the suite noticed.
##
## These tests therefore run in a child of globalenv() and in a clean
## subprocess, deliberately NOT parented to the namespace.

test_that("the generated evaluator family is reachable from the search path", {
  ## this is the exact lookup rxode2's symengine renderer performs
  for (f in c("nn1", "nn2", "nn3", "nn4", "nn1_d1", "nn2_d1_d2")) {
    expect_true(exists(f, envir = as.environment("package:nlmixr2nn"), inherits = FALSE),
                info = paste(f, "must be exported: rxode2 resolves it by name"))
  }
})

test_that("a model needing derivatives builds with only the package attached", {
  skip_if_not_installed("rxode2")
  skip_if_not_installed("callr")
  skip_on_cran()

  ## a fresh process with nothing but the package attached -- no namespace
  ## parenting, no test scaffolding
  out <- callr::r(function() {
    library(nlmixr2nn)
    m <- function() {
      ini({ add.sd <- 0.3 })
      model({
        g <- nn(centr, nHidden = 3L, act = "tanh")
        d/dt(centr) <- -(1.0 / (1.0 + exp(-g))) * centr
        centr ~ add(add.sd)
      })
    }
    ui <- suppressWarnings(suppressMessages(rxode2::rxode2(m)))
    ## Building the FOCEi models differentiates the right-hand side, which is
    ## what forces the symengine path and therefore the by-name lookup of nn1
    ## and its derivatives.  This is the step that failed when nn1 was internal.
    invisible(suppressWarnings(suppressMessages(ui$foceiModel)))
    "ok"
  }, spinner = FALSE)

  expect_equal(out, "ok")
})

test_that("the documented user surface is exactly what is exported", {
  ## a change here should be deliberate; the machine-facing generated family is
  ## excluded because it is an implementation detail rxode2 happens to need
  exported <- getNamespaceExports("nlmixr2nn")
  userFacing <- setdiff(exported, grep("^nn[0-9]|^nnWg|^nnprobe$|^nnnpars$",
                                       exported, value = TRUE))
  expect_setequal(userFacing, c("nn", "nnControl", "nnEval", "nnWeights"))
})
