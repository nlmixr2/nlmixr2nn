// C++ libtorch backend for nlmixr2nn.  Isolated in its own translation unit so
// only this file pulls in the (heavy) torch headers; the hot-path evaluation
// (nnEval.c) and registration (init.c) stay plain C.
//
// Owns the neural-network modules (weights + autograd) entirely in C++ so they
// are reachable from compiled solve/objective code without ever calling back
// into R (which is not thread-safe).  Module weights are pushed into the
// nnEval weight buffer via the C setter nnSetWeightsC(), and the rxode2
// par-loader hook injects them into par_ptr each solve.
#include <torch/torch.h>
#include <map>
#include <memory>

#define STRICT_R_HEADERS
#include <R.h>
#include <Rinternals.h>

// C setter in nnEval.c -- fills a network's weight buffer without an R round-trip.
extern "C" void nnSetWeightsC(int id, const double *w, int n);

// Single-hidden-layer MLP whose weight layout matches nnEval.c:
//   W1 (H,K) row-major, b1 (H), W2 (1,H), b2 (1).
struct MLPImpl : torch::nn::Module {
  torch::nn::Linear l1{nullptr}, l2{nullptr};
  int K, H, act;
  MLPImpl(int K_, int H_, int act_) : K(K_), H(H_), act(act_) {
    l1 = register_module("l1", torch::nn::Linear(torch::nn::LinearOptions(K, H)));
    l2 = register_module("l2", torch::nn::Linear(torch::nn::LinearOptions(H, 1)));
    this->to(torch::kFloat64);
  }
  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor h = l1->forward(x);
    switch (act) {
    case 0: h = torch::relu(h); break;
    case 1: h = torch::softplus(h); break;
    case 2: h = torch::tanh(h); break;
    default: break;
    }
    return l2->forward(h);
  }
  // flat weights in nnEval layout order
  torch::Tensor flatten() {
    return torch::cat({l1->weight.reshape({-1}), l1->bias,
                       l2->weight.reshape({-1}), l2->bias})
      .contiguous().to(torch::kFloat64);
  }
  // load flat weights (inverse of flatten), e.g. to restore a saved network
  void unflatten(const double *w, int n) {
    torch::NoGradGuard ng;
    int need = H * K + 2 * H + 1;
    if (n != need) Rf_error("nn weight length %d != expected %d", n, need);
    torch::Tensor t = torch::from_blob((void *) w, {n}, torch::kFloat64).clone();
    int o = 0;
    l1->weight.data().copy_(t.slice(0, o, o + H * K).reshape({H, K})); o += H * K;
    l1->bias.data().copy_(t.slice(0, o, o + H));                      o += H;
    l2->weight.data().copy_(t.slice(0, o, o + H).reshape({1, H}));    o += H;
    l2->bias.data().copy_(t.slice(0, o, o + 1));
  }
};
TORCH_MODULE(MLP);

static std::map<int, MLP> g_modules;

static MLP &getModule(int id) {
  auto it = g_modules.find(id);
  if (it == g_modules.end()) Rf_error("no torch module registered for nn id %d", id);
  return it->second;
}

extern "C" {

SEXP _nlmixr2nn_nnTorchAvailable(void) { return Rf_ScalarLogical(TRUE); }

SEXP _nlmixr2nn_nnTorchProbe(SEXP n) {
  torch::NoGradGuard ng;
  return Rf_ScalarReal(torch::ones({Rf_asInteger(n)}, torch::kFloat64).sum().item<double>());
}

// create (or replace) a module for network `id`
SEXP _nlmixr2nn_nnTorchInit(SEXP id, SEXP K, SEXP H, SEXP act, SEXP seed) {
  int i = Rf_asInteger(id);
  if (!Rf_isNull(seed)) torch::manual_seed((uint64_t) Rf_asInteger(seed));
  // operator[] would default-construct MLP (which has no default ctor)
  g_modules.insert_or_assign(i, MLP(Rf_asInteger(K), Rf_asInteger(H), Rf_asInteger(act)));
  return Rf_ScalarInteger(i);
}

SEXP _nlmixr2nn_nnTorchFree(SEXP id) {
  g_modules.erase(Rf_asInteger(id));
  return R_NilValue;
}

// push the module's weights into the nnEval buffer (C path, no R round-trip)
SEXP _nlmixr2nn_nnTorchSync(SEXP id) {
  int i = Rf_asInteger(id);
  torch::NoGradGuard ng;
  torch::Tensor flat = getModule(i)->flatten();
  int n = (int) flat.numel();
  nnSetWeightsC(i, flat.data_ptr<double>(), n);
  return Rf_ScalarInteger(n);
}

// flat weights as an R numeric vector (nnEval layout order)
SEXP _nlmixr2nn_nnTorchGetWeights(SEXP id) {
  torch::NoGradGuard ng;
  torch::Tensor flat = getModule(Rf_asInteger(id))->flatten();
  int n = (int) flat.numel();
  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  std::memcpy(REAL(out), flat.data_ptr<double>(), (size_t) n * sizeof(double));
  UNPROTECT(1);
  return out;
}

// load flat weights into the module (restore)
SEXP _nlmixr2nn_nnTorchSetWeights(SEXP id, SEXP vals) {
  getModule(Rf_asInteger(id))->unflatten(REAL(vals), Rf_length(vals));
  return R_NilValue;
}

// forward pass of the module at one input row (length K) -- for validation
SEXP _nlmixr2nn_nnTorchForward(SEXP id, SEXP x) {
  torch::NoGradGuard ng;
  MLP m = getModule(Rf_asInteger(id));
  int k = Rf_length(x);
  torch::Tensor xt = torch::from_blob(REAL(x), {1, k}, torch::kFloat64).clone();
  torch::Tensor y = m->forward(xt);
  return Rf_ScalarReal(y.item<double>());
}

SEXP _nlmixr2nn_nnTorchSave(SEXP id, SEXP path) {
  torch::save(getModule(Rf_asInteger(id)), std::string(CHAR(STRING_ELT(path, 0))));
  return R_NilValue;
}

SEXP _nlmixr2nn_nnTorchLoad(SEXP id, SEXP path) {
  torch::load(getModule(Rf_asInteger(id)), std::string(CHAR(STRING_ELT(path, 0))));
  return R_NilValue;
}

} // extern "C"
