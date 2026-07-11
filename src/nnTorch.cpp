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
    case 3: h = torch::gelu(h); break;          // exact (erf) GELU
    case 4: h = torch::silu(h); break;          // SiLU / Swish
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
static std::map<int, std::shared_ptr<torch::optim::Optimizer>> g_opt;

static MLP &getModule(int id) {
  auto it = g_modules.find(id);
  if (it == g_modules.end()) Rf_error("no torch module registered for nn id %d", id);
  return it->second;
}

static torch::optim::Optimizer *getOpt(int id) {
  auto it = g_opt.find(id);
  if (it == g_opt.end()) Rf_error("no optimizer for nn id %d (call nnTorchOptInit)", id);
  return it->second.get();
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

// ---- training (torch owns/updates the weights) ---------------------------
// type: 0 = SGD, 1 = Adam.  Creates an optimizer over the module parameters.
SEXP _nlmixr2nn_nnTorchOptInit(SEXP id, SEXP type, SEXP lr) {
  int i = Rf_asInteger(id);
  MLP m = getModule(i);
  double l = Rf_asReal(lr);
  if (Rf_asInteger(type) == 1) {
    g_opt[i] = std::make_shared<torch::optim::Adam>(m->parameters(),
                 torch::optim::AdamOptions(l));
  } else {
    g_opt[i] = std::make_shared<torch::optim::SGD>(m->parameters(),
                 torch::optim::SGDOptions(l));
  }
  return R_NilValue;
}

SEXP _nlmixr2nn_nnTorchZeroGrad(SEXP id) {
  getOpt(Rf_asInteger(id))->zero_grad();
  return R_NilValue;
}

// forward a batch: X is row-major [N x K]; returns y [N]
SEXP _nlmixr2nn_nnTorchForwardBatch(SEXP id, SEXP X, SEXP Nn, SEXP Kk) {
  torch::NoGradGuard ng;
  int N = Rf_asInteger(Nn), K = Rf_asInteger(Kk);
  torch::Tensor xt = torch::from_blob(REAL(X), {N, K}, torch::kFloat64).clone();
  torch::Tensor y = getModule(Rf_asInteger(id))->forward(xt).reshape({N});
  SEXP out = PROTECT(Rf_allocVector(REALSXP, N));
  std::memcpy(REAL(out), y.contiguous().data_ptr<double>(), (size_t) N * sizeof(double));
  UNPROTECT(1);
  return out;
}

// vector-Jacobian product: accumulate d(sum G*y)/d(w) into the parameter grads.
// X row-major [N x K]; G [N] is the upstream cotangent d(loss)/d(output).
SEXP _nlmixr2nn_nnTorchBackward(SEXP id, SEXP X, SEXP G, SEXP Nn, SEXP Kk) {
  int N = Rf_asInteger(Nn), K = Rf_asInteger(Kk);
  MLP m = getModule(Rf_asInteger(id));
  torch::Tensor xt = torch::from_blob(REAL(X), {N, K}, torch::kFloat64).clone();
  torch::Tensor gt = torch::from_blob(REAL(G), {N, 1}, torch::kFloat64).clone();
  torch::Tensor y = m->forward(xt);          // [N,1], graph over params
  y.backward(gt);                            // param.grad += sum_n G_n * dy_n/dw
  return R_NilValue;
}

// flattened parameter gradients in nnEval layout order (for gradient checks)
SEXP _nlmixr2nn_nnTorchGetGrad(SEXP id) {
  torch::NoGradGuard ng;
  MLP m = getModule(Rf_asInteger(id));
  torch::Tensor flat = torch::cat({m->l1->weight.grad().reshape({-1}),
                                   m->l1->bias.grad(),
                                   m->l2->weight.grad().reshape({-1}),
                                   m->l2->bias.grad()}).contiguous().to(torch::kFloat64);
  int n = (int) flat.numel();
  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  std::memcpy(REAL(out), flat.data_ptr<double>(), (size_t) n * sizeof(double));
  UNPROTECT(1);
  return out;
}

// optimizer step, then push the updated weights into the loader buffer so the
// next solve uses them.
SEXP _nlmixr2nn_nnTorchStep(SEXP id) {
  int i = Rf_asInteger(id);
  getOpt(i)->step();
  torch::NoGradGuard ng;
  torch::Tensor flat = getModule(i)->flatten();
  nnSetWeightsC(i, flat.data_ptr<double>(), (int) flat.numel());
  return R_NilValue;
}

} // extern "C"
