# Neural-network term for an rxode2/nlmixr2 model

Use \`nn(state, ..., nHidden=, act=)\` inside a model to insert a
single-hidden-layer neural network of the state input(s). At parse time
it is replaced by a compiled \`nn\<K\>()\` call and its weights are
added to the model as randomly-initialized population parameters.

## Usage

``` r
nn(
  ...,
  nHidden = 5L,
  act = c("softplus", "tanh", "relu", "gelu", "silu"),
  init = c("ude", "torch", "normal"),
  initSd = 0.1,
  aug = 0L,
  seed = NULL,
  num = NULL,
  iniDf = NULL
)
```

## Arguments

- ...:

  one or more state/covariate inputs to the network (given positionally,
  e.g. \`nn(central, t)\`).

- nHidden:

  hidden-layer width (default 5).

- act:

  activation, one of "softplus" (default), "tanh", "relu", "gelu" or
  "silu". The default is smooth with a nonzero second derivative, which
  matters because the model's Jacobian and the FOCEi sensitivities are
  both chained through it; "relu" has a kink and a zero second
  derivative.

- init:

  weight initialization: \`"ude"\` (default) draws fan-in scaled weights
  with zero biases and a shrunk output layer, so an untrained network is
  a small perturbation of the mechanistic model; \`"torch"\` reproduces
  libtorch's \`nn::Linear\` default; \`"normal"\` is a flat \`N(0,
  initSd)\`.

- initSd:

  scale of the initialization: the output-layer standard deviation for
  \`init = "ude"\`, and the standard deviation of every weight for
  \`init = "normal"\`.

- aug:

  number of AUGMENTED states to add (0-3, default 0) – the ANODE
  construction of Dupont et al. (2019). \*\*Not available yet: any value
  above 0 is an error.\*\* Augmenting from inside \`nn()\` would
  renumber the model's compartments (the augmented state takes a
  \`d/dt()\` slot next to the \`nn()\` call, so \`cmt\` in the data
  stops meaning what it means in the model), and a UDF cannot see the
  model's states to prevent that. Doing it at model assembly instead is
  the fix. Write the augmentation by hand meanwhile – see
  \`vignette("nlmixr2nn-node")\`. \`aug = k\` creates \`k\` extra
  compartments \`rx_nnaug\<id\>\_1..k\`, starts them at zero, gives each
  its own learned derivative, and passes all of them to this network as
  additional inputs. It is exactly what you would get by writing

  “\` d/dt(a1) \<- nn(centr, a1, a2) d/dt(a2) \<- nn(centr, a1, a2) g
  \<- nn(centr, a1, a2) “\`

  by hand, and exists because the thing it fixes is invisible: a learned
  flow in one dimension cannot cross itself, so \`d/dt(centr) \<-
  -f(nn(centr))\` is topologically constrained no matter how well it is
  fitted or how much capacity it has. Extra dimensions remove the
  constraint.

  The cost is real and worth weighing. The augmented solve carries one
  variational state per (state x weight), and \`aug = k\` multiplies
  BOTH: it adds \`k\` states and \`k\` more networks. Start at 1.

  With \`init = "ude"\` (the default) every augmented derivative starts
  near zero, so the augmented states stay near zero and contribute
  nothing until training finds a use for them – the zero-augmentation
  start ANODE prescribes. (The network itself is still a fresh draw over
  the wider input vector, so this is not the same model as \`aug = 0\`.)
  Total inputs (\`...\` plus \`aug\`) may not exceed 4.

  Each augmented state RELAXES rather than integrates: \`d/dt(a) =
  f(x, a) - a\`. The textbook drift, \`d/dt(a) = f(x, a)\`, is
  self-exciting here – \`a\` re-enters its own derivative through the
  network – and blows the solver up on a real dosing horizon. The \`-
  a\` gives it a stable fixed point at \`f\`, at the price of a fixed
  memory timescale of one model time unit.

- seed:

  optional integer pinning THIS \`nn()\` call's initial weights
  regardless of the ambient seed. Normally unnecessary – use
  \[rxode2::rxSetSeed()\]. Every network is drawn at \`seed + \<network
  id\>\`, so the networks of an \`aug =\` call differ from each other
  rather than all drawing the same weights.

- num:

  network occurrence number; queried via \`rxUdfUiNum()\` if \`NULL\`.

- iniDf:

  initial-estimate data.frame; queried via \`rxUdfUiIniDf()\` if
  \`NULL\`.

## Value

a list consumed by \[rxode2::rxUdfUi()\] (\`replace\`, \`before\`).

## Details

Inter-individual variability (a "deepNLME"/DeepPumas-style latent random
effect) is best added by passing a per-subject latent eta as an INPUT to
the population network, e.g. \`g \<- nn(central, eta.latent)\` with
\`eta.latent ~ omega\` in the \`ini\` block. Because the eta is then an
ordinary (visible) nlmixr2 random effect, its inner FOCEi sensitivity
\`d(g)/d(eta)\` is the network's analytic input derivative – exact, no
finite differences – and it is far more identifiable than a random
effect on every weight.

The network's initial weights are drawn WHEN THE MODEL IS PARSED, using
rxode2's threefry generator, and are carried on the model itself. So
\[rxode2::rxSetSeed()\] makes a network reproducible across sessions –
and, because threefry is the generator whose stream is defined under
multiple threads, reproducible in the same places a parallel solve is. A
freshly parsed model is immediately solvable: no torch module and no
setup call.

Drawing weights disturbs nothing. Every draw runs inside
\[rxode2::rxWithSeed()\], which saves and restores both the R stream and
the rxode2 seed, so adding a network never shifts the draws elsewhere in
a script. A bare \`set.seed()\` still pins a model – it is the fallback
used when \`rxSetSeed()\` has not been called – but \`rxSetSeed()\` is
the one to reach for.

## References

Dupont E, Doucet A, Teh YW (2019). "Augmented neural ODEs." \*Advances
in Neural Information Processing Systems\* 32. See
\`vignette("nlmixr2nn-node")\` for what else from that literature does
and does not carry over.
