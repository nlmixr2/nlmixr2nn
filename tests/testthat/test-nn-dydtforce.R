## B1: rxode2 dydt forcing hook.  A registered hook adds forcing to a state
## derivative from the generated model's RHS -- the mechanism the NN-weight
## forward-sensitivity (variational) states use to receive their b_j term on top
## of the J*s_j part rxode2's sensitivity codegen produces.  Here we validate the
## rxode2 mechanism itself with a constant forcing against the analytic solution
## of a linear ODE.

test_that("dydt forcing hook adds forcing to a state derivative (B1)", {
  skip_if_not_installed("rxode2")
  ## d/dt(x) = -x, x(0) = 1.  Baseline solution x(t) = exp(-t).
  m <- rxode2::rxode2("d/dt(x) = -x")
  ev <- rxode2::et(seq(0, 5, by = 0.5))
  tt <- ev$time

  base <- rxode2::rxSolve(m, ev, inits = c(x = 1))
  expect_equal(base$x, exp(-tt), tolerance = 1e-5)

  ## Force +c on state 0 (x): d/dt(x) = -x + c => x(t) = c + (1 - c) exp(-t).
  cc <- 0.5
  on.exit(.Call("_nlmixr2nn_testDydtForce", 0L, 0.0, FALSE, PACKAGE = "nlmixr2nn"),
          add = TRUE)
  .Call("_nlmixr2nn_testDydtForce", 0L, cc, TRUE, PACKAGE = "nlmixr2nn")
  forced <- rxode2::rxSolve(m, ev, inits = c(x = 1))

  ## Mechanism must actually fire: forced solution differs from baseline and
  ## matches the analytic forced solution.
  expect_false(isTRUE(all.equal(forced$x, base$x)))
  expect_equal(forced$x, cc + (1 - cc) * exp(-tt), tolerance = 1e-5)

  ## Removing the hook restores the baseline (no dangling forcing).
  .Call("_nlmixr2nn_testDydtForce", 0L, 0.0, FALSE, PACKAGE = "nlmixr2nn")
  restored <- rxode2::rxSolve(m, ev, inits = c(x = 1))
  expect_equal(restored$x, exp(-tt), tolerance = 1e-5)
})
