## Package-internal state, defined first (files are sourced alphabetically).
## Registry of networks generated during parsing and attached torch modules.
.nnEnv <- new.env(parent = emptyenv())
.nnEnv$reg <- list()          # id -> {id, K, H, act, weights (names)}
.nnEnv$torchIds <- integer(0) # ids with an attached C++ torch module
.nnEnv$scales <- list()       # id -> per-input scales applied to W1 (reporting)
