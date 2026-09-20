// Self-contained training backend for nlmixr2nn -- the fallback compiled when
// libtorch is not available (see ./configure), and the default on Windows (see
// ./configure.win).
//
// It implements exactly the same extern "C" entry points as nnTorch.cpp, so
// nothing above it -- R wrappers, the fit loop, init.c -- knows which backend
// it is talking to.  Only `_nlmixr2nn_nnBackend()` distinguishes them.
//
// Why this can be a straight substitute in a real fit: the production path
// never asks the backend to differentiate anything.  It injects the analytic
// dLoss/dw that came from the ODE forward sensitivity (nnTorchSetGrad) and asks
// for an optimizer step (nnTorchStep) -- see R/nnTrain.R and R/nnWeightStep.R.
// So what libtorch actually supplies there is Adam/SGD plus a float64
// parameter store, which is what this file is.
//
// The autograd entry points (ForwardBatch/Backward/GetGrad) are used by the
// gradient-check tests rather than by a fit.  They are implemented here too,
// from the SAME analytic weight-gradient routine the solve uses
// (nnWeightGradWC in nnEval.c), so the checks stay meaningful: they compare the
// backend against finite differences, not against a second copy of itself.
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <new>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>

#define STRICT_R_HEADERS
#include <R.h>
#include <Rinternals.h>

// From nnEval.c.  Declared here rather than via nlmixr2nn.h, which needs
// rxode2's rx_solve struct in scope; this TU has no business with the solve.
extern "C" {
void nnSetWeightsC(int id, const double *w, int n);
double nnForwardWC(int K, int H, int act, const double *w, const double *x);
void nnWeightGradWC(int K, int H, int act, const double *w, const double *x,
                    double *g);
}

// A C++ exception escaping .Call() is not an R error -- it unwinds past R's
// context and hits std::terminate, killing the session.  Every entry point that
// can allocate is wrapped, so a bad_alloc becomes an ordinary R error.  Rf_error
// is called only after the try block has exited, so nothing live is skipped.
#define NN_BEGIN try {
// NB: the parameter is NOT called `what`.  It would be substituted inside
// `e_.what()` as well, yielding `e_."init"()` -- the preprocessor does not know
// that one is a member function and the other a macro argument.
#define NN_END(nn_what_)                                                     \
  } catch (const std::exception &e_) {                                       \
    Rf_error("nlmixr2nn builtin backend (%s): %s", nn_what_, e_.what());     \
  } catch (...) {                                                            \
    Rf_error("nlmixr2nn builtin backend (%s): unknown C++ exception",        \
             nn_what_);                                                      \
  }                                                                          \
  return R_NilValue;

namespace {

// Guard against a weight count that overflows int before it is ever allocated.
// K comes from the model (1..4) and H from the user via nn(nHidden=), so both
// are reachable from R and neither can be trusted here.
int nWeightsChecked(int K, int H) {
  if (K < 1 || H < 1) {
    Rf_error("nn dimensions must be positive (K=%d, H=%d)", K, H);
  }
  // H*K + 2H + 1 in double first: no wraparound while checking for wraparound.
  double n = (double) H * (double) K + 2.0 * (double) H + 1.0;
  if (n > 1e8) {
    Rf_error("nn network too large (K=%d, H=%d would need %.0f weights)",
             K, H, n);
  }
  return H * K + 2 * H + 1;
}

// splitmix64: a small, self-contained, reproducible generator.  Only used to
// initialize a fresh module -- in a real model the weights are drawn by nn() at
// parse time (rxode2's threefry) and pushed in with SetWeights, so this never
// decides anything a user sees.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double unif() {                       // [0,1)
    return (double) (next() >> 11) * (1.0 / 9007199254740992.0);
  }
};

struct Net {
  int K, H, act;
  std::vector<double> w, g;
  Net(int K_, int H_, int act_, uint64_t seed)
    : K(K_), H(H_), act(act_) {
    int n = nWeightsChecked(K, H);
    w.assign((size_t) n, 0.0);
    g.assign((size_t) n, 0.0);
    // torch's nn::Linear default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for both
    // weight and bias, per layer.
    Rng r(seed);
    double b1 = 1.0 / std::sqrt((double) K);
    double b2 = 1.0 / std::sqrt((double) H);
    int o = 0;
    for (int i = 0; i < H * K; i++) w[o++] = (2.0 * r.unif() - 1.0) * b1;
    for (int i = 0; i < H; i++)     w[o++] = (2.0 * r.unif() - 1.0) * b1;
    for (int i = 0; i < H; i++)     w[o++] = (2.0 * r.unif() - 1.0) * b2;
    w[o] = (2.0 * r.unif() - 1.0) * b2;
  }
};

// Adam exactly as torch formulates it (betas 0.9/0.999, eps 1e-8, no weight
// decay, no amsgrad): denom = sqrt(v)/sqrt(bc2) + eps, step = lr/bc1.  Written
// this way rather than the textbook mhat/vhat form so a fit started under one
// backend and continued under the other takes the same steps.
struct Opt {
  int type;            // 0 = SGD, 1 = Adam
  double lr;
  int64_t t;           // NOT long: that is 32-bit on Windows, and an overflow
                       // here flips the bias correction negative -- the
                       // optimizer would silently start ascending.
  std::vector<double> m, v;
  Opt(int type_, double lr_, int n)
    : type(type_), lr(lr_), t(0),
      m((size_t) n, 0.0), v((size_t) n, 0.0) {}

  void step(std::vector<double> &w, const std::vector<double> &g) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    if (type == 1) {
      t++;
      double bc1 = 1.0 - std::pow(b1, (double) t);
      double bc2 = 1.0 - std::pow(b2, (double) t);
      double stepSize = lr / bc1;
      for (size_t i = 0; i < w.size(); i++) {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        double denom = std::sqrt(v[i]) / std::sqrt(bc2) + eps;
        w[i] -= stepSize * m[i] / denom;
      }
    } else {
      for (size_t i = 0; i < w.size(); i++) w[i] -= lr * g[i];
    }
  }
};

std::map<int, Net> g_nets;
std::map<int, Opt> g_opt;

Net &getNet(int id) {
  std::map<int, Net>::iterator it = g_nets.find(id);
  if (it == g_nets.end()) Rf_error("no module registered for nn id %d", id);
  return it->second;
}

Opt &getOpt(int id) {
  std::map<int, Opt>::iterator it = g_opt.find(id);
  if (it == g_opt.end()) {
    Rf_error("no optimizer for nn id %d (call nnTorchOptInit)", id);
  }
  return it->second;
}

// Serialized format.  Fixed-width fields and an explicit byte-order sentinel:
// a file written by one build must not be misread by another, and "silently
// wrong weights" is a far worse failure than "refuses to load".
const int32_t kMagic = 0x324E4E42;          // "BNN2"
const uint64_t kOrder = 0x0102030405060708ULL;

}  // namespace

extern "C" {

SEXP _nlmixr2nn_nnBackend(void) { return Rf_mkString("builtin"); }

// FALSE: this asks whether libtorch is linked, which it is not.  Training still
// works -- R and the vignette check the BACKEND, not this, before deciding
// whether a fit can run.
SEXP _nlmixr2nn_nnTorchAvailable(void) { return Rf_ScalarLogical(FALSE); }

SEXP _nlmixr2nn_nnTorchProbe(SEXP n) {
  return Rf_ScalarReal((double) Rf_asInteger(n));
}

SEXP _nlmixr2nn_nnTorchInit(SEXP id, SEXP K, SEXP H, SEXP act, SEXP seed) {
  int i = Rf_asInteger(id);
  int k = Rf_asInteger(K), h = Rf_asInteger(H);
  nWeightsChecked(k, h);                 // reject before allocating anything
  uint64_t s;
  if (Rf_isNull(seed)) {
    // torch's module constructor draws from torch's global RNG when it is not
    // seeded, so an unseeded module is random there.  Match that by drawing
    // from R's stream rather than silently using a fixed seed, which would make
    // the two backends disagree about what "unseeded" means.
    GetRNGstate();
    s = (uint64_t) (unif_rand() * 9007199254740992.0);
    PutRNGstate();
  } else {
    s = (uint64_t) Rf_asInteger(seed);
  }
  NN_BEGIN
    g_nets.erase(i);
    g_nets.insert(std::make_pair(i, Net(k, h, Rf_asInteger(act), s)));
    g_opt.erase(i);        // a new module invalidates the optimizer state
    return Rf_ScalarInteger(i);
  NN_END("init")
}

SEXP _nlmixr2nn_nnTorchFree(SEXP id) {
  int i = Rf_asInteger(id);
  g_nets.erase(i);
  g_opt.erase(i);
  return R_NilValue;
}

SEXP _nlmixr2nn_nnTorchSync(SEXP id) {
  int i = Rf_asInteger(id);
  Net &n = getNet(i);
  nnSetWeightsC(i, &n.w[0], (int) n.w.size());
  return Rf_ScalarInteger((int) n.w.size());
}

SEXP _nlmixr2nn_nnTorchGetWeights(SEXP id) {
  Net &n = getNet(Rf_asInteger(id));
  SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t) n.w.size()));
  std::memcpy(REAL(out), &n.w[0], n.w.size() * sizeof(double));
  UNPROTECT(1);
  return out;
}

SEXP _nlmixr2nn_nnTorchSetWeights(SEXP id, SEXP vals) {
  Net &n = getNet(Rf_asInteger(id));
  int len = Rf_length(vals), need = (int) n.w.size();
  if (len != need) Rf_error("nn weight length %d != expected %d", len, need);
  std::memcpy(&n.w[0], REAL(vals), (size_t) need * sizeof(double));
  return R_NilValue;
}

SEXP _nlmixr2nn_nnTorchForward(SEXP id, SEXP x) {
  Net &n = getNet(Rf_asInteger(id));
  if (Rf_length(x) != n.K) {
    Rf_error("nn input length %d != expected %d", Rf_length(x), n.K);
  }
  return Rf_ScalarReal(nnForwardWC(n.K, n.H, n.act, &n.w[0], REAL(x)));
}

// Weights only, in nnWeightLayout order.  Deliberately NOT torch's archive
// format: a module saved by one backend is not loadable by the other, and the
// header makes that a clear error rather than a silent misread.
SEXP _nlmixr2nn_nnTorchSave(SEXP id, SEXP path) {
  Net &n = getNet(Rf_asInteger(id));
  const char *p = CHAR(STRING_ELT(path, 0));
  FILE *f = fopen(p, "wb");
  if (f == NULL) Rf_error("cannot open '%s' for writing", p);
  int32_t hdr[4];
  hdr[0] = kMagic; hdr[1] = n.K; hdr[2] = n.H; hdr[3] = n.act;
  uint64_t order = kOrder;
  uint64_t nw = (uint64_t) n.w.size();
  if (fwrite(hdr, sizeof(int32_t), 4, f) != 4 ||
      fwrite(&order, sizeof(uint64_t), 1, f) != 1 ||
      fwrite(&nw, sizeof(uint64_t), 1, f) != 1 ||
      fwrite(&n.w[0], sizeof(double), (size_t) nw, f) != (size_t) nw) {
    fclose(f);
    Rf_error("short write saving nn module to '%s'", p);
  }
  fclose(f);
  return R_NilValue;
}

SEXP _nlmixr2nn_nnTorchLoad(SEXP id, SEXP path) {
  Net &n = getNet(Rf_asInteger(id));
  const char *p = CHAR(STRING_ELT(path, 0));
  FILE *f = fopen(p, "rb");
  if (f == NULL) Rf_error("cannot open '%s' for reading", p);
  int32_t hdr[4];
  uint64_t order = 0, nw = 0;
  if (fread(hdr, sizeof(int32_t), 4, f) != 4 ||
      fread(&order, sizeof(uint64_t), 1, f) != 1 ||
      fread(&nw, sizeof(uint64_t), 1, f) != 1) {
    fclose(f);
    Rf_error("'%s' is too short to be an nn module", p);
  }
  if (hdr[0] != kMagic) {
    fclose(f);
    Rf_error("'%s' was not written by the builtin nn backend "
             "(a libtorch-saved module cannot be read by it)", p);
  }
  if (order != kOrder) {
    fclose(f);
    Rf_error("'%s' was written on a machine with a different byte order", p);
  }
  if (hdr[1] != n.K || hdr[2] != n.H || hdr[3] != n.act ||
      nw != (uint64_t) n.w.size()) {
    fclose(f);
    Rf_error("'%s' holds a K=%d H=%d act=%d network, but nn id %d is "
             "K=%d H=%d act=%d", p, hdr[1], hdr[2], hdr[3],
             Rf_asInteger(id), n.K, n.H, n.act);
  }
  if (fread(&n.w[0], sizeof(double), (size_t) nw, f) != (size_t) nw) {
    fclose(f);
    Rf_error("short read loading nn module from '%s'", p);
  }
  fclose(f);
  return R_NilValue;
}

SEXP _nlmixr2nn_nnTorchOptInit(SEXP id, SEXP type, SEXP lr) {
  int i = Rf_asInteger(id);
  Net &n = getNet(i);
  int nw = (int) n.w.size();
  int ty = Rf_asInteger(type);
  double l = Rf_asReal(lr);
  NN_BEGIN
    g_opt.erase(i);
    g_opt.insert(std::make_pair(i, Opt(ty, l, nw)));
    return R_NilValue;
  NN_END("optInit")
}

SEXP _nlmixr2nn_nnTorchZeroGrad(SEXP id) {
  Net &n = getNet(Rf_asInteger(id));
  std::fill(n.g.begin(), n.g.end(), 0.0);
  return R_NilValue;
}

// X is row-major [N x K] (R passes t(X)); returns y [N].
SEXP _nlmixr2nn_nnTorchForwardBatch(SEXP id, SEXP X, SEXP Nn, SEXP Kk) {
  Net &n = getNet(Rf_asInteger(id));
  int N = Rf_asInteger(Nn), K = Rf_asInteger(Kk);
  if (K != n.K) Rf_error("nn input dim %d != expected %d", K, n.K);
  if (N < 0) Rf_error("nn batch size %d is negative", N);
  if ((double) N * K > (double) Rf_xlength(X)) {
    Rf_error("nn input matrix too short");
  }
  const double *x = REAL(X);
  SEXP out = PROTECT(Rf_allocVector(REALSXP, N));
  double *o = REAL(out);
  for (int i = 0; i < N; i++) {
    o[i] = nnForwardWC(n.K, n.H, n.act, &n.w[0], x + (size_t) i * K);
  }
  UNPROTECT(1);
  return out;
}

// Vector-Jacobian product: g += sum_i G_i * d(y_i)/d(w), accumulated into the
// stored gradient exactly as torch's backward() accumulates into .grad.
SEXP _nlmixr2nn_nnTorchBackward(SEXP id, SEXP X, SEXP G, SEXP Nn, SEXP Kk) {
  Net &n = getNet(Rf_asInteger(id));
  int N = Rf_asInteger(Nn), K = Rf_asInteger(Kk);
  if (K != n.K) Rf_error("nn input dim %d != expected %d", K, n.K);
  if (N < 0) Rf_error("nn batch size %d is negative", N);
  if ((double) N * K > (double) Rf_xlength(X)) {
    Rf_error("nn input matrix too short");
  }
  if (Rf_xlength(G) < N) Rf_error("nn cotangent shorter than the batch");
  const double *x = REAL(X), *gt = REAL(G);
  size_t nw = n.w.size();
  NN_BEGIN
    std::vector<double> row(nw, 0.0);
    for (int i = 0; i < N; i++) {
      nnWeightGradWC(n.K, n.H, n.act, &n.w[0], x + (size_t) i * K, &row[0]);
      double c = gt[i];
      for (size_t j = 0; j < nw; j++) n.g[j] += c * row[j];
    }
    return R_NilValue;
  NN_END("backward")
}

SEXP _nlmixr2nn_nnTorchGetGrad(SEXP id) {
  Net &n = getNet(Rf_asInteger(id));
  SEXP out = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t) n.g.size()));
  std::memcpy(REAL(out), &n.g[0], n.g.size() * sizeof(double));
  UNPROTECT(1);
  return out;
}

SEXP _nlmixr2nn_nnTorchSetGrad(SEXP id, SEXP grad) {
  Net &n = getNet(Rf_asInteger(id));
  int len = Rf_length(grad), need = (int) n.g.size();
  if (len != need) Rf_error("nn grad length %d != expected %d", len, need);
  std::memcpy(&n.g[0], REAL(grad), (size_t) need * sizeof(double));
  return R_NilValue;
}

SEXP _nlmixr2nn_nnTorchStep(SEXP id) {
  int i = Rf_asInteger(id);
  Net &n = getNet(i);
  Opt &o = getOpt(i);
  if (o.m.size() != n.w.size()) {
    Rf_error("optimizer state for nn id %d does not match the module "
             "(call nnTorchOptInit after re-initializing it)", i);
  }
  o.step(n.w, n.g);
  nnSetWeightsC(i, &n.w[0], (int) n.w.size());
  return R_NilValue;
}

}  // extern "C"
