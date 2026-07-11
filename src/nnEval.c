#define STRICT_R_HEADERS
#include <stdint.h>
#include <math.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <rxode2parseStruct.h>

/* Single-hidden-layer MLP evaluated inside an rxode2 ODE right-hand side.

   A network with input dimension K and hidden width H has weights laid out
   contiguously in the solve parameter vector (par_ptr) starting at index
   `base`, in this order (all population thetas, no IIV):

     W1 : H*K   row-major, hidden row j, input col k -> W1[j*K + k]
     b1 : H
     W2 : H     output weights
     b2 : 1

   so the block length is H*K + 2*H + 1.  Weights are read live from par_ptr on
   every call (read-only, population-invariant across subjects -> thread-safe).

   Per-network layout metadata (base, K, H, activation) is model-static and is
   pushed into a small registry by nnSetMeta() before solving.  The network id
   is the first argument to nn<K>(). */

#define NN_MAX 256

typedef struct {
  int set;
  int base;   /* par_ptr index of W1[0] */
  int K;      /* input dimension */
  int H;      /* hidden width */
  int act;    /* 0 = ReLU, 1 = Softplus, 2 = tanh */
} nn_meta;

static nn_meta nnReg[NN_MAX];

static rx_solve *(*getRxSolve_fn)(void) = NULL;
static rx_solve *nnGetRx(void) {
  if (getRxSolve_fn == NULL) {
    getRxSolve_fn = (rx_solve *(*)(void)) R_GetCCallable("rxode2", "getRxSolve_");
  }
  return getRxSolve_fn();
}

/* activation and its first two derivatives w.r.t. the pre-activation z */
static inline double nnAct(int act, double z) {
  switch (act) {
  case 0: return z > 0 ? z : 0.0;                 /* ReLU */
  case 1: return z > 0 ? z + log1p(exp(-z)) : log1p(exp(z)); /* Softplus, stable */
  case 2: return tanh(z);
  default: return z;
  }
}
static inline double nnActD(int act, double z) {
  switch (act) {
  case 0: return z > 0 ? 1.0 : 0.0;
  case 1: return 1.0 / (1.0 + exp(-z));           /* logistic */
  case 2: { double t = tanh(z); return 1.0 - t * t; }
  default: return 1.0;
  }
}
static inline double nnActD2(int act, double z) {
  switch (act) {
  case 0: return 0.0;
  case 1: { double s = 1.0 / (1.0 + exp(-z)); return s * (1.0 - s); }
  case 2: { double t = tanh(z); return -2.0 * t * (1.0 - t * t); }
  default: return 0.0;
  }
}

/* fetch the network's weight block base pointer and dims */
static const double *nnWeights(int id, int *K, int *H, int *act) {
  if (id < 0 || id >= NN_MAX || !nnReg[id].set) return NULL;
  rx_solve *rx = nnGetRx();
  if (rx == NULL) return NULL;
  const double *pp = rx->subjects[0].par_ptr;
  *K = nnReg[id].K; *H = nnReg[id].H; *act = nnReg[id].act;
  return pp + nnReg[id].base;
}

/* forward value; x has length K */
static double nnForward(int id, const double *x) {
  int K, H, act;
  const double *w = nnWeights(id, &K, &H, &act);
  if (w == NULL) return NA_REAL;
  const double *W1 = w;
  const double *b1 = W1 + H * K;
  const double *W2 = b1 + H;
  double b2 = W2[H];
  double out = b2;
  for (int j = 0; j < H; j++) {
    double z = b1[j];
    for (int k = 0; k < K; k++) z += W1[j * K + k] * x[k];
    out += W2[j] * nnAct(act, z);
  }
  return out;
}

/* d out / d x_m */
static double nnGrad(int id, const double *x, int m) {
  int K, H, act;
  const double *w = nnWeights(id, &K, &H, &act);
  if (w == NULL) return NA_REAL;
  const double *W1 = w;
  const double *b1 = W1 + H * K;
  const double *W2 = b1 + H;
  double g = 0.0;
  for (int j = 0; j < H; j++) {
    double z = b1[j];
    for (int k = 0; k < K; k++) z += W1[j * K + k] * x[k];
    g += W2[j] * nnActD(act, z) * W1[j * K + m];
  }
  return g;
}

/* d2 out / d x_m d x_l */
static double nnHess(int id, const double *x, int m, int l) {
  int K, H, act;
  const double *w = nnWeights(id, &K, &H, &act);
  if (w == NULL) return NA_REAL;
  const double *W1 = w;
  const double *b1 = W1 + H * K;
  const double *W2 = b1 + H;
  double h = 0.0;
  for (int j = 0; j < H; j++) {
    double z = b1[j];
    for (int k = 0; k < K; k++) z += W1[j * K + k] * x[k];
    h += W2[j] * nnActD2(act, z) * W1[j * K + m] * W1[j * K + l];
  }
  return h;
}

/* ---- fixed-arity rxode2 entry points (id is the first argument) ---------- */
/* K = 1 */
double nn1(double id, double x1) {
  double x[1] = {x1};
  return nnForward((int) id, x);
}
double nn1_d1(double id, double x1) {
  double x[1] = {x1};
  return nnGrad((int) id, x, 0);
}
double nn1_d1_d1(double id, double x1) {
  double x[1] = {x1};
  return nnHess((int) id, x, 0, 0);
}
/* K = 2 */
double nn2(double id, double x1, double x2) {
  double x[2] = {x1, x2};
  return nnForward((int) id, x);
}
double nn2_d1(double id, double x1, double x2) {
  double x[2] = {x1, x2};
  return nnGrad((int) id, x, 0);
}
double nn2_d2(double id, double x1, double x2) {
  double x[2] = {x1, x2};
  return nnGrad((int) id, x, 1);
}
double nn2_d1_d1(double id, double x1, double x2) {
  double x[2] = {x1, x2};
  return nnHess((int) id, x, 0, 0);
}
double nn2_d1_d2(double id, double x1, double x2) {
  double x[2] = {x1, x2};
  return nnHess((int) id, x, 0, 1);
}
double nn2_d2_d2(double id, double x1, double x2) {
  double x[2] = {x1, x2};
  return nnHess((int) id, x, 1, 1);
}

/* ---- registry management (called from R) --------------------------------- */
SEXP _rxode2nn_nnSetMeta(SEXP id, SEXP base, SEXP K, SEXP H, SEXP act) {
  int i = asInteger(id);
  if (i < 0 || i >= NN_MAX) error("nn id out of range");
  nnReg[i].set  = 1;
  nnReg[i].base = asInteger(base);
  nnReg[i].K    = asInteger(K);
  nnReg[i].H    = asInteger(H);
  nnReg[i].act  = asInteger(act);
  return ScalarLogical(1);
}
SEXP _rxode2nn_nnClearMeta(void) {
  for (int i = 0; i < NN_MAX; i++) nnReg[i].set = 0;
  return ScalarLogical(1);
}

/* SEXP wrappers so the functions are also callable directly from R for tests */
#define NN_WRAP2(nm)                                            \
  SEXP _rxode2nn_##nm(SEXP id, SEXP x1) {                       \
    int n = LENGTH(x1);                                         \
    SEXP out = PROTECT(allocVector(REALSXP, n));                \
    double *pid = REAL(id), *p1 = REAL(x1), *r = REAL(out);     \
    for (int i = 0; i < n; i++) r[i] = nm(pid[i], p1[i]);       \
    UNPROTECT(1); return out;                                   \
  }
#define NN_WRAP3(nm)                                                    \
  SEXP _rxode2nn_##nm(SEXP id, SEXP x1, SEXP x2) {                      \
    int n = LENGTH(x1);                                                 \
    SEXP out = PROTECT(allocVector(REALSXP, n));                        \
    double *pid = REAL(id), *p1 = REAL(x1), *p2 = REAL(x2), *r = REAL(out); \
    for (int i = 0; i < n; i++) r[i] = nm(pid[i], p1[i], p2[i]);        \
    UNPROTECT(1); return out;                                           \
  }
NN_WRAP2(nn1)
NN_WRAP2(nn1_d1)
NN_WRAP2(nn1_d1_d1)
NN_WRAP3(nn2)
NN_WRAP3(nn2_d1)
NN_WRAP3(nn2_d2)
NN_WRAP3(nn2_d1_d1)
NN_WRAP3(nn2_d1_d2)
NN_WRAP3(nn2_d2_d2)
