# Neural networks in nlmixr2 models

## What `nn()` is

[`nn()`](https://nlmixr2.github.io/nlmixr2nn/reference/nn.md) puts a
neural network inside a model. You write it where you would write any
other term, and everything else is an ordinary `rxode2`/`nlmixr2`
workflow: the model solves, `nlmixr2()` fits it,
[`predict()`](https://rdrr.io/r/stats/predict.html) works,
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) works.

``` r

library(nlmixr2nn)

mod <- function() {
  ini({
    lka <- 0.5
    lVc <- 1
  })
  model({
    ka <- exp(lka)
    Vc <- exp(lVc)
    d/dt(depot)   <- -ka * depot
    ## a 6-unit network of the central amount enters the elimination
    d/dt(central) <-  ka * depot - nn(central, nHidden = 6)
    cp <- central / Vc
  })
}

rxode2::rxSetSeed(42)
ui <- rxode2::rxode2(mod)
#> ℹ parameter labels from comments are typically ignored in non-interactive mode
#> ℹ Need to run with the source intact to parse comments
```

The network is **opaque**: it becomes a single compiled call, so the
model text stays the same size no matter how big the network is. Its
value and its exact input derivatives are computed in fast thread-safe
C, and those derivatives are what make a random effect passed *into* a
network identifiable.

``` r

cat(paste(ui$lstChr, collapse = "\n"))
#> ka <- exp(lka)
#> Vc <- exp(lVc)
#> d/dt(depot) <- -ka * depot
#> rx_nnw0_ <- rxnnW1_0_1_1 + rxnnW1_0_2_1 + rxnnW1_0_3_1 + rxnnW1_0_4_1 + rxnnW1_0_5_1 + rxnnW1_0_6_1 + rxnnB1_0_1 + rxnnB1_0_2 + rxnnB1_0_3 + rxnnB1_0_4 + rxnnB1_0_5 + rxnnB1_0_6 + rxnnW2_0_1 + rxnnW2_0_2 + rxnnW2_0_3 + rxnnW2_0_4 + rxnnW2_0_5 + rxnnW2_0_6 + rxnnB2_0
#> d/dt(central) <- ka * depot - nn1(0, central)
#> cp <- central/Vc
```

The weights were drawn when the model was parsed, using rxode2’s
threefry generator — so `rxSetSeed()` reproduces a network exactly, in
this session or any other, and the model is ready to solve immediately.
Threefry is also the generator whose stream is defined across threads,
so a network is reproducible in the same places a parallel solve is.
Drawing weights disturbs neither the R stream nor the rxode2 seed.

``` r

head(nnWeights(ui), 4)
#> rxnnW1_0_1_1 rxnnW1_0_2_1 rxnnW1_0_3_1 rxnnW1_0_4_1 
#>   -0.9058632    1.5106134    3.5083403   -2.5716819
```

Everything below is a single `nlmixr2()` call. There is no setup step.

## Simulating a model you have not fitted

Because the model carries its own weights, it solves straight away. This
is the quickest way to check that a model does what you think before
fitting anything.

``` r

ev <- rxode2::et(amt = 100, cmt = "depot") |> rxode2::et(0, 24, by = 4)
head(rxode2::rxSolve(ui, ev), 4)
#>   time       ka       Vc rx_nnw0_       cp        depot  central
#> 1    0 1.648721 2.718282 1.118299  0.00000 1.000000e+02  0.00000
#> 2    4 1.648721 2.718282 1.118299 27.78800 1.367344e-01 75.53562
#> 3    8 1.648721 2.718282 1.118299 20.01552 1.869613e-04 54.40783
#> 4   12 1.648721 2.718282 1.118299 14.38974 2.561300e-07 39.11537
```

## Learning a term you do not know (UDE)

Keep the mechanism you trust and let a small network absorb the part you
do not. Here the truth is a saturating (Michaelis-Menten) elimination
that varies between subjects; the model knows only that elimination is
*some* bounded function of the amount, and a latent random effect
`eta.nn` enters the network as an input.

``` r

rxode2::rxSetSeed(1)   # pins the network
set.seed(1)            # pins the simulated data below
truth <- rxode2::rxode2("d/dt(centr) = -(2*exp(eV))*centr/(3+centr)")
etaTrue <- rnorm(8, 0, sqrt(0.15))
d <- do.call(rbind, lapply(1:8, function(id) {
  s <- rxode2::rxSolve(truth,
         data.frame(id = id, time = c(0, .5, 1, 2, 4, 6, 8, 10),
                    evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
         params = c(eV = etaTrue[id]), returnType = "data.frame")
  s <- s[s$time > 0, ]
  data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
             amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr + rnorm(nrow(s), 0, .1)))
}))

ude <- function() {
  ini({ add.sd <- 0.3; eta.nn ~ 0.2 })
  model({
    g <- nn(centr, eta.nn, nHidden = 3, act = "tanh")
    d/dt(centr) <- -(1 / (1 + exp(-g))) * centr     # a bounded learned rate
    centr ~ add(add.sd)
  })
}

fit <- nlmixr2est::nlmixr2(ude, d, "focei")
#> ℹ parameter labels from comments are typically ignored in non-interactive mode
#> ℹ Need to run with the source intact to parse comments
#> Warning: some etas defaulted to non-mu referenced, possible parsing error: eta.nn
#> as a work-around try putting the mu-referenced expression on a simple line
#> Registered S3 method overwritten by 'minqa':
#>   method      from     
#>   print.minqa RcppTrust
#> Warning: some etas defaulted to non-mu referenced, possible parsing error: eta.nn
#> as a work-around try putting the mu-referenced expression on a simple line
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|     2945.8976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|     1606.6904 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|     550.85688 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|     550.85668 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|     68.574875 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|     68.573963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -39.042524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -39.042457 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -39.042472 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -39.042469 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -39.042469 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -39.042469 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -39.042469 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -39.042469 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -39.042469 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -53.378372 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -53.378464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -57.969352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -57.969363 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -57.969373 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -57.969373 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -57.969373 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -57.969373 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -57.969373 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -57.969373 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -57.969373 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -58.799056 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -58.799069 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -60.209250 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -60.209252 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -60.547078 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -60.547084 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -60.823523 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -60.823412 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -61.075595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -61.075604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -61.341950 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -61.341944 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -61.622112 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -61.932306 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -61.932454 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -62.308920 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -62.308923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -62.758617 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -62.758620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -63.303482 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -63.303483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -63.834073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -63.834081 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.179736 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.179704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.264533 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.264490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.207604 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.207572 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.167766 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.167734 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.188511 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.188459 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.266024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.265991 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.266011 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.266011 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.266011 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.266011 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.266011 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.266011 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.266011 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.382067 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.382068 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.512387 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.512386 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.668094 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -64.880006 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -64.879951 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -65.155810 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -65.155815 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -65.479294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -65.479247 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -65.796582 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.031649 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.122550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.122553 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.107211 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.107358 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.057932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.057936 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.041544 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.041599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.053593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.053595 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.073725 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.073790 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.073826 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.073826 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.073826 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.073826 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.073826 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.073826 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.073826 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.117699 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.117705 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.186700 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.186701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.287272 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.287274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.434276 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.625701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.625704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -66.849842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -66.849844 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.088585 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.088591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.310074 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.310107 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.471567 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.471570 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.555875 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.555896 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.555897 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.555897 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.555897 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.555897 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.555897 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.555897 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.555897 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.579273 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.579298 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.568534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.568689 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.568816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.568816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.568816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.568816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.568816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.568816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.568816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.549564 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.549594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.547932 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.546735 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.546743 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.559488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.587044 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.587045 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.636591 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.636593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.708187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.804913 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -67.926004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -67.926008 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.069159 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.069125 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.224902 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.224901 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.380986 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.380990 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.524208 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.524230 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.524229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.524229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.524229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.524229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.524229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.524229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.524229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.635473 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.635489 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.635488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.635488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.635488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.635488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.635488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.635488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.635488 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.705559 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.705573 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.741344 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.741355 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.741354 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.741354 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.741354 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.741354 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.741354 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.741354 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.741354 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.750437 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.750451 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.750450 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.750450 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.750450 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.750450 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.750450 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.750450 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.750450 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.750426 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.750442 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.744717 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.744720 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.742605 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.742622 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.743483 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.743506 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.743505 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.743505 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.743505 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.743505 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.743505 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.743505 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.743505 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.748598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.748600 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.765790 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.765791 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.793087 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.793089 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.832961 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.832962 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.889643 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.889644 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.889645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.889645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.889645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.889645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.889645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.889645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.889645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -68.961318 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -68.961319 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.051533 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.051534 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.149911 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.149935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.261366 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.261382 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.390261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.390262 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.504583 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.504590 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.504589 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.504589 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.504589 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.504589 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.504589 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.504589 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.504589 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.609710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.609716 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.697593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.697594 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.758921 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.758914 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.800113 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.800114 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.819535 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.819537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.829387 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.829388 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.835784 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.835774 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.838986 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.838987 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.843350 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.843352 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.848980 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.848982 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.857783 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.857785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.869817 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.869823 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.890777 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.890778 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.920645 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.920648 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -69.961576 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -69.961566 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.009885 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.009889 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.074821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.074822 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.150168 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.150170 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.234261 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.325795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.325796 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.418911 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.418908 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.505239 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.505237 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.593557 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.662690 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.662701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.725018 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.725040 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.725041 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.725041 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.725041 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.725041 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.725041 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.725041 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.725041 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.768510 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.768521 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.803963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.803973 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.803974 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.803974 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.803974 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.803974 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.803974 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.803974 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.803974 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.824935 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.824945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.824946 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.824946 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.824946 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.824946 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.824946 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.824946 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.824946 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.835064 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.835075 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.844608 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.844618 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.858198 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.858197 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.858184 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.858171 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.858171 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.858171 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.858171 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.858171 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.858171 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.868831 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.868839 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.868840 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.868840 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.868840 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.868840 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.868840 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.868840 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.868840 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.871234 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.871246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.869517 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.869524 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.880258 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.880269 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.880270 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.880270 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.880270 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.880270 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.880270 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.880270 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.880270 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.895809 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.895821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.915890 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.942482 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.942490 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -70.976180 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -70.976187 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -70.976188 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -70.976188 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -70.976188 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -70.976188 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -70.976188 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -70.976188 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -70.976188 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.018287 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.018294 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.064680 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.064681 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.127404 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.127413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.192123 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.192131 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.261085 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.261090 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.337742 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.337750 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.411031 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.411014 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.411015 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.411015 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.411015 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.411015 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.411015 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.411015 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.411015 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.485617 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.485620 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.550407 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.550413 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.594905 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.594903 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.638564 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.638543 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.638528 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.638502 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.638502 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.638502 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.638502 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.638502 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.638502 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.670229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.670227 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.670228 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.670228 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.670228 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.670228 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.670228 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.670228 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.670228 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.696701 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.706410 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.706415 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.711545 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.711550 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.725077 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.725083 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.723717 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.723722 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.731704 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.731710 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.734931 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.734937 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.734938 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.734938 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.734938 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.734938 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.734938 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.734938 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.734938 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.740842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.740847 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.740848 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.740848 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.740848 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.740848 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.740848 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.740848 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.740848 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.745592 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.745598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.752118 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.752124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.762928 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.762933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.762934 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.762934 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.762934 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.762934 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.762934 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.762934 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.762934 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.777721 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.777708 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.777689 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.777670 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.777670 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.777670 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.777670 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.777670 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.777670 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.803708 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.803692 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.835120 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.835124 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.867355 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.867336 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.901821 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.901818 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.947929 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.947933 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -71.997956 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -71.997959 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.052816 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.052820 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.108384 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.162976 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.162975 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.218290 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.218274 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.218275 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.218275 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.218275 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.218275 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.218275 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.218275 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.218275 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.271862 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.271865 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.316807 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.316812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.355540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.355542 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.384397 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.384402 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.406537 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.406540 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.422111 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.433753 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.441203 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.441206 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.441205 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.441205 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.441205 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.441205 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.441205 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.441205 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.441205 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.444915 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.444917 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.447640 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.447641 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.450225 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.450229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.447020 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.447025 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.447024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.447024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.447024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.447024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.447024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.447024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.447024 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.451925 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.451922 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.451923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.451923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.451923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.451923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.451923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.451923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.451923 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.454535 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.454538 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.450366 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.450376 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.450377 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.450377 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.450377 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.450377 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.450377 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.450377 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.450377 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.438759 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.438761 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.446856 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.446858 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.466593 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.466598 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.466599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.466599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.466599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.466599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.466599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.466599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.466599 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.484605 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.484602 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.505726 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.505642 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.530050 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.530053 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.560278 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.560280 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.593425 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.593432 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.593431 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.593431 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.593431 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.593431 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.593431 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.593431 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.593431 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.623836 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.623842 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.660245 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.660248 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.696782 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.696794 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.696795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.696795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.696795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.696795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.696795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.696795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.696795 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.735957 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.735963 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.771517 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.771525 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.807973 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.807996 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.807996 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.807997 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.807997 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.807997 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.807997 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.807997 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.807997 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.844321 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.877003 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.877004 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.902064 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.902073 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.924805 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.924812 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.945100 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.945102 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.958399 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.958397 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.958398 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.958398 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.958398 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.958398 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.958398 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.958398 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.958398 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.961766 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.961771 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.968229 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.968231 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.973246 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.973244 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.974134 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.974146 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.975943 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.975945 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.975464 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.975465 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.975419 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.975428 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.975427 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.975427 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.975427 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.975427 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.975427 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.975427 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.975427 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.976149 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.976162 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.976161 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.976161 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.976161 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.976161 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.976161 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.976161 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.976161 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; G: Gill difference gradient approximation
#> F: Forward difference gradient approximation
#> C: Central difference gradient approximation
#> M: Mixed forward and central difference gradient approximation
#> A: Analytic (forward sensitivity) gradient (fast=TRUE)
#> Unscaled parameters for Omegas=chol(solve(omega));
#> Diagonals are transformed, as specified by foceiControl(diagXform=)
#> 
#> |    #| Function Val. |    add.sd |        o1 |
#> |-----+---------------+-----------+-----------|
#> |    1|    -72.976785 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    2|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    3|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    4|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    5|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    6|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    7|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    8|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> |    9|    -72.976798 |    -1.000 |     1.000 |
#> |    U|               |    0.3000 |     1.495 |
#> |    X|               |    0.3000 |     1.495 |
#> done
#> nn: joint training stopped at the round limit after 200 rounds
#>     objective -72.98 | weight change 0.0012 | objf change 8.7e-06
#>     rmse 1.819 -> 0.09472
#>     (raise nnControl(rounds=) if the weights were still moving)
#> → Calculating residuals/tables
#> ✔ done
#> → loading into symengine environment...
#> → pruning branches (`if`/`else`) of full model...
#> ✔ done
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00
#> → calculate sensitivities
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00
#> → calculate ∂(f)/∂(η)
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00
#> → finding duplicate expressions in inner model...
#> → finding duplicate expressions in EBE model...
#> → compiling inner model...
#> ✔ done
#> → finding duplicate expressions in FD model...
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00
#> → compiling EBE model...
#> ✔ done
#> → compiling events FD model...
#> ✔ done
#> calculating covariance matrix
#> Updated original fit object
```

That is the whole call: no `nn=` argument, and the data was not touched.
The result is an ordinary `nlmixr2` fit, so the random effect recovers
the between-subject variation the way any eta would:

``` r

cor(fit$eta$eta.nn[order(fit$eta$ID)], etaTrue)
#> [1] -0.9956862
```

To see what the network *learned*, evaluate it directly:

``` r

e <- nnEval(fit, centr = seq(0.5, 10, length.out = 25), eta.nn = 0)
head(e, 3)
#> nn() network 0 (tanh, 2 inputs) evaluated at 3 points
#>       centr eta.nn     value
#> 1 0.5000000      0 -1.047115
#> 2 0.8958333      0 -1.150657
#> 3 1.2916667      0 -1.246881
```

`plot(e)` draws that function, and `plot(e, true = f)` overlays a
reference curve when you have one.

## Letting the network find a covariate relationship

The same idea with covariates as the inputs: the network learns how they
drive a parameter, and the latent eta carries what they do not explain.

``` r

covmod <- function() {
  ini({ tCL <- 1; tV <- 3.5; add.sd <- 0.5; eta.nn ~ 0.1 })
  model({
    CL <- exp(tCL + nn(WT, EGFR, eta.nn))        # learned covariate map
    V  <- exp(tV)
    d/dt(central) <- -(CL / V) * central
    cp <- central / V
    cp ~ add(add.sd)
  })
}
fit <- nlmixr2est::nlmixr2(covmod, data, "focei")
```

## A model with no between-subject variability

Systems-pharmacology models are often fitted with no random effects at
all: one mechanistic system, with a network standing in for an unknown
term. Fit those with a population optimizer — `"bobyqa"`, `"nlminb"`,
`"lbfgsb3c"` and friends — and there is no eta to estimate.

``` r

rxode2::rxSetSeed(1)
set.seed(1)
qspTruth <- rxode2::rxode2("d/dt(centr) = -(2)*centr/(3+centr)")
qd <- do.call(rbind, lapply(1:6, function(id) {
  s <- rxode2::rxSolve(qspTruth,
         data.frame(id = id, time = c(0, .5, 1, 2, 4, 6, 8, 10),
                    evid = c(1, rep(0, 7)), cmt = 1, amt = c(10, rep(0, 7))),
         returnType = "data.frame")
  s <- s[s$time > 0, ]
  data.frame(id = id, time = c(0, s$time), evid = c(1, rep(0, nrow(s))), cmt = 1,
             amt = c(10, rep(0, nrow(s))), dv = c(NA, s$centr + rnorm(nrow(s), 0, .1)))
}))

qsp <- function() {
  ini({ add.sd <- 0.3 })                     # residual error only -- no eta
  model({
    g <- nn(centr, nHidden = 3, act = "tanh")
    d/dt(centr) <- -(1 / (1 + exp(-g))) * centr
    centr ~ add(add.sd)
  })
}

qspFit <- nlmixr2est::nlmixr2(qsp, qd, "bobyqa")
#> ℹ parameter labels from comments are typically ignored in non-interactive mode
#> ℹ Need to run with the source intact to parse comments
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00 
#> 
#> Key: U: Unscaled Parameters; X: Back-transformed parameters; 
#> 
#> |    #| Function Val. |    add.sd |
#> |-----+---------------+-----------|
#> |    1|    -10.528246 |     1.000 |
#> |    U|               |    0.3000 |
#> |    X|               |    0.3000 |
#> |    2|    -10.528246 |     1.000 |
#> |    U|               |    0.3000 |
#> |    X|               |    0.3000 |
#> |    3|    -6.7756900 |     1.200 |
#> |    U|               |    0.3300 |
#> |    X|               |    0.3300 |
#> |    4|    -14.614861 |    0.8000 |
#> |    U|               |    0.2700 |
#> |    X|               |    0.2700 |
#> |    5|    -19.088479 |    0.6000 |
#> |    U|               |    0.2400 |
#> |    X|               |    0.2400 |
#> |    6|    -29.417245 |    0.2000 |
#> |    U|               |    0.1800 |
#> |    X|               |    0.1800 |
#> |    7|    -43.487997 |   -0.6000 |
#> |    U|               |   0.06000 |
#> |    X|               |   0.06000 |
#> |    8|     38.725306 |    -1.000 |
#> |    U|               |     0.000 |
#> |    X|               |     0.000 |
#> |    9|    -42.826049 |   -0.2473 |
#> |    U|               |    0.1129 |
#> |    X|               |    0.1129 |
#> |   10|    -46.890505 |   -0.4271 |
#> |    U|               |   0.08594 |
#> |    X|               |   0.08594 |
#> 
#> |    #| Function Val. |    add.sd |
#> |-----+---------------+-----------|
#> |   11|    -46.703420 |   -0.4129 |
#> |    U|               |   0.08806 |
#> |    X|               |   0.08806 |
#> |   12|    -47.038836 |   -0.4412 |
#> |    U|               |   0.08382 |
#> |    X|               |   0.08382 |
#> |   13|    -47.164160 |   -0.4595 |
#> |    U|               |   0.08108 |
#> |    X|               |   0.08108 |
#> |   14|    -47.174400 |   -0.4619 |
#> |    U|               |   0.08072 |
#> |    X|               |   0.08072 |
#> |   15|    -47.190240 |   -0.4667 |
#> |    U|               |   0.08000 |
#> |    X|               |   0.08000 |
#> |   16|    -47.199329 |   -0.4712 |
#> |    U|               |   0.07932 |
#> |    X|               |   0.07932 |
#> |   17|    -47.200278 |   -0.4720 |
#> |    U|               |   0.07921 |
#> |    X|               |   0.07921 |
#> |   18|    -47.201252 |   -0.4730 |
#> |    U|               |   0.07906 |
#> |    X|               |   0.07906 |
#> |   19|    -47.201928 |   -0.4740 |
#> |    U|               |   0.07891 |
#> |    X|               |   0.07891 |
#> |   20|    -47.202383 |   -0.4759 |
#> |    U|               |   0.07862 |
#> |    X|               |   0.07862 |
#> 
#> |    #| Function Val. |    add.sd |
#> |-----+---------------+-----------|
#> |   21|    -47.202182 |   -0.4769 |
#> |    U|               |   0.07847 |
#> |    X|               |   0.07847 |
#> |   22|    -47.202386 |   -0.4757 |
#> |    U|               |   0.07865 |
#> |    X|               |   0.07865 |
#> |   23|    -47.202386 |   -0.4757 |
#> |    U|               |   0.07865 |
#> |    X|               |   0.07865 |
#> |-----+---------------+-----------|
#> [====|====|====|====|====|====|====|====|====|====] 0:00:00
#> nn: population (no-BSV) fit via bobyqa (native weight-reading solve)
#> → Calculating residuals/tables
#> ✔ done
```

The fitted model reproduces the dynamics it was never told about:

``` r

nd <- data.frame(time = c(0, .5, 1, 2, 4, 6, 8, 10), evid = c(1, rep(0, 7)),
                 cmt = 1, amt = c(10, rep(0, 7)))
a <- rxode2::rxSolve(qspFit$finalUi, nd, returnType = "data.frame")
b <- rxode2::rxSolve(qspTruth, nd, returnType = "data.frame")
data.frame(time = a$time, fitted = round(a$centr, 2), true = round(b$centr, 2))
#>   time fitted true
#> 1  0.5   9.21 9.24
#> 2  1.0   8.46 8.49
#> 3  2.0   7.06 7.05
#> 4  4.0   4.54 4.44
#> 5  6.0   2.38 2.35
#> 6  8.0   0.95 0.98
#> 7 10.0   0.30 0.32
```

Passing a population-only optimizer for a model that *does* have a
random effect is an error rather than a silent compromise — the weights
and the Omega would otherwise come from two different objectives.

## Saving, reloading and refitting

The trained weights are carried on the fitted model, so a saved fit is
self-contained: it reloads and re-solves in a fresh session with no
external state. Feeding a fit back in resumes from its trained weights.

``` r

f <- tempfile(fileext = ".rds")
saveRDS(qspFit, f)
reloaded <- readRDS(f)
identical(round(rxode2::rxSolve(reloaded$finalUi, nd, returnType = "data.frame")$centr, 8),
          round(a$centr, 8))
#> [1] TRUE
unlink(f)
```

``` r

refit <- nlmixr2est::nlmixr2(qspFit, moreData, "bobyqa")   # warm start
```

## When you want to steer the training

The training schedule is inferred from the model, the data and the
estimator you chose: whether the estimator can resume from a partial
fit, whether there is a random effect, and what the residual model is.
[`nnControl()`](https://nlmixr2.github.io/nlmixr2nn/reference/nnControl.md)
overrides any part of it and leaves the rest inferred.

``` r

nlmixr2est::nlmixr2(ude, d, "focei", nn = nnControl(rounds = 400, lr = 0.01))
```

| setting | inferred as | override when |
|----|----|----|
| `mode` | `"joint"` when the estimator resumes from a partial fit, else `"iter"` | you want fully converged inner fits each round |
| `rounds` | 200 joint, 30 iterative, 60 population | training was still improving when it stopped |
| `cotangent` | the Gaussian closed form for an untransformed `add()`/`prop()` endpoint, the distribution’s own score for a count endpoint, the captured exact score otherwise | rarely — a source the endpoint cannot support is an error, not a silent fallback |
| `lr`, `optimizer` | 0.03 (0.01 on a refit), Adam | the objective oscillates or moves too slowly |
| `warmStart` | a population pre-fit, skipped when the model already carries trained weights | you want a longer or shorter pre-fit |
| `l2`, `smooth` | `0` – no penalty on the weights or on curvature | a fitted network looks wigglier than the data can justify (see below) |
| `kinetic` | `0` | the learned term is stiff, or the augmented solve is slow |

### Regularization, when a network learns too much

A network with more capacity than the data supports will invent
structure — a covariate curve with sharp wiggles in it, or per-subject
variation that should have gone to the random effect. It does this
silently: the fit converges, the objective looks fine, and only the
learned shape gives it away.

Two penalties are available for that, and they express different
preferences. `l2` is an L2-norm penalty on the weights — ridge
regression, or weight decay in machine-learning terms — which charges a
weight for being far from zero and so prefers the flattest network the
data will allow. `smooth` is a roughness penalty in the spline sense: it
charges for the *second* derivative of the network along each input, so
a monotone slope is free and only the wiggles are paid for. Put plainly,
`l2` asks the network to be small and `smooth` asks it to be smooth,
which is usually what you want from a learned covariate relationship — a
shape, not a straight line and not an interpolation of the noise. Biases
are never penalized, and neither is a latent eta’s input column — the
eta reaches the model only through the network, so shrinking those
weights would shrink the random effect itself.

Both are a **fraction of the objective**, not an absolute amount:
`l2 = 0.05` means the weight penalty starts at 5% of the objective, so
one value means roughly the same thing whatever the endpoint, estimator
or data size.

``` r

fit <- nlmixr2est::nlmixr2(ude, d, "focei", nn = nnControl(l2 = 0.05))
plot(nnEval(fit, centr = seq(0, 10, length.out = 101)))
```

**Both default to `0`.** That is a measured choice, not a cautious one.
`l2` shrinks the network toward the zero function, and in these models
the network *is* the model — so any value large enough to suppress
structure the data does not support is also large enough to attenuate
structure it does. On the package’s own recovery tests, every value from
`1e-3` up cost a real covariate effect, and every value that preserved
them moved the weight norm by less than half a percent.

So reach for it deliberately, when a fitted curve looks wigglier than
the data can justify. `l2 = 0.05` is where shrinkage becomes substantial
— on a test fixture with no true covariate effect it took the weight
norm from 11.5 to 3.9 while the unpenalized objective *improved*. Always
look at what the network learned with
[`nnEval()`](https://nlmixr2.github.io/nlmixr2nn/reference/nnEval.md)
afterwards rather than trusting the objective alone: real effects shrink
alongside invented ones.

A third penalty, `kinetic`, asks a different question. `l2` and `smooth`
are about the *shape* of the learned function; `kinetic` is about the
*dynamics* it produces. It charges for the network’s output along the
trajectory the model actually solves — how hard the network is pushing
where the solution really goes, rather than on a grid of input
combinations that may never occur — and so asks the model to reach the
same data by pushing less hard.

``` r

fit <- nlmixr2est::nlmixr2(ude, d, "focei", nn = nnControl(kinetic = 0.05))
```

That makes the learned dynamics easier to integrate, so it can make a
fit *faster* as well as smoother — which matters here because the
training solve carries a sensitivity state for every state-weight pair,
so every solver step saved is multiplied. It defaults to `0` for a
reason of its own on top of the one above: it is the only one of the
three that changes the dynamics being solved rather than only steering
the optimizer.

`kinetic` is a fraction of the objective too, though in a weaker sense
than `l2` and `smooth`: `kinetic = 0.05` charges 5% of the objective per
unit of the term, rather than starting at exactly 5% of it. The
difference is deliberate and
[`vignette("nlmixr2nn-node")`](https://nlmixr2.github.io/nlmixr2nn/articles/nlmixr2nn-node.md)
explains it — briefly, the default initialization starts the network
near the zero function on purpose, so normalizing this term against its
own starting value would divide by something four orders of magnitude
too small.

All three shape the optimization only. The reported `objf` — and so AIC
and BIC — stays the unpenalized -2 log-likelihood, so a regularized
network model is still directly comparable to one with an analytic
covariate relationship.
[`vignette("nlmixr2nn-node")`](https://nlmixr2.github.io/nlmixr2nn/articles/nlmixr2nn-node.md)
covers where `kinetic` comes from and which other ideas from the neural
ODE literature do and do not carry over to a model like this one.

## What is supported

- **Inputs**: 1 to 4, given positionally — states, covariates, or a
  latent eta.

- **Activations**: `"softplus"` (default), `"tanh"`, `"relu"`, `"gelu"`,
  `"silu"`. The default is smooth with a nonzero second derivative,
  which matters because the model Jacobian and the FOCEi sensitivities
  are chained through it.

- **Hidden layers**: one. `nHidden` is a single width.

- **Endpoints**: one. Any normal-distribution residual model — `add()`,
  `prop()`, both, or a transformation such as `lnorm()` — and the count
  endpoints `pois()` and `binom()`.

  A count endpoint differs in one way worth knowing. For `y ~ pois(lam)`
  there is no “prediction” to differentiate: the model’s prediction *is*
  the log-density, so the network’s effect is chained through the
  distribution’s own parameter (`lam`) instead, and the score is that
  distribution’s derivative. Two consequences:
  `nnControl(cotangent = "exact")` does not apply and is refused, and
  **the parameter must be positive by construction** — write
  `lam <- exp(nn(...))` rather than letting a rate wander negative,
  which aborts the likelihood evaluation. For `binom(n, p)` the network
  drives `p`; the size `n` must reach the solve, so bind it to a data
  column (`n <- ntrials`) rather than writing a bare constant, which the
  model compiler folds away.

  Other distributions (`nbinom()`, ordinal, and the rest) are refused
  with a message naming the reason, not fitted wrongly.

- **Networks per model**: several; they are trained together, and
  `nnEval(net=)` selects between them.

- **Estimators**: any standard one. `"focei"`, the Laplace/AGQ methods,
  the importance-sampling methods (`"imp"`, `"impmap"`) and the
  variational methods interleave network training with the population
  fit, because their outer iteration genuinely resumes from where it
  left off. SAEM (`"saem"`, `"fsaem"`), `"qrpem"` and the nonparametric
  methods (`"npag"`, `"npb"`) alternate instead: SAEM’s
  stochastic-approximation gain sequence restarts on every call, so
  chopping it into pieces would not continue the chain.

Network initialization is scaled to each input’s magnitude, measured
from the data, so a network reading a concentration of 0.02 and one
reading an amount of 500 both start in a usable range.
