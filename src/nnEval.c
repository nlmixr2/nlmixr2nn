#define STRICT_R_HEADERS
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <R.h>
#include <Rinternals.h>
#include <rxode2parseStruct.h>   /* full rx_solve struct (C-safe, no Rcpp) */
#include "nlmixr2nn.h"            /* C bridge to rxode2's function-pointer table */

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
  int nW;     /* block length = H*K + 2*H + 1 */
  int hasW;   /* whether an external weight buffer is populated */
  double *weights; /* externally-owned weights (from torch / nnSetWeights) */
} nn_meta;

static nn_meta nnReg[NN_MAX];

/* Loader hook invoked by rxode2 once per solve (single-threaded, before the
   parallel integration).  Overwrites each registered network's reserved
   par_ptr block, in every column, with its externally-owned weights so that
   torch -- not nlmixr2's optimizer -- controls the network parameters. */
void nnParLoader(rx_solve *rx, double *gpars, int npars, int ncols) {
  (void) rx;
  for (int id = 0; id < NN_MAX; id++) {
    if (!nnReg[id].set || !nnReg[id].hasW || nnReg[id].weights == NULL) continue;
    int base = nnReg[id].base, nW = nnReg[id].nW;
    if (base < 0 || base + nW > npars) continue;
    const double *w = nnReg[id].weights;
    for (int c = 0; c < ncols; c++) {
      double *col = gpars + (size_t) c * npars;
      for (int k = 0; k < nW; k++) col[base + k] = w[k];
    }
  }
}

static rx_solve *nnGetRx(void) {
  return nlmixr2nnGetRxSolve();   /* NULL until the table is installed */
}

/* activation codes (keep in sync with .nnActCode in R and MLPImpl in nnTorch.cpp):
   0 ReLU, 1 Softplus, 2 tanh, 3 GELU (exact/erf), 4 SiLU (Swish) */
#define NN_INV_SQRT_2PI 0.3989422804014327  /* 1/sqrt(2*pi) */

/* activation and its first two derivatives w.r.t. the pre-activation z */
static inline double nnAct(int act, double z) {
  switch (act) {
  case 0: return z > 0 ? z : 0.0;                 /* ReLU */
  case 1: return z > 0 ? z + log1p(exp(-z)) : log1p(exp(z)); /* Softplus, stable */
  case 2: return tanh(z);
  case 3: return 0.5 * z * (1.0 + erf(z * M_SQRT1_2));       /* GELU */
  case 4: return z / (1.0 + exp(-z));                        /* SiLU / Swish */
  default: return z;
  }
}
static inline double nnActD(int act, double z) {
  switch (act) {
  case 0: return z > 0 ? 1.0 : 0.0;
  case 1: return 1.0 / (1.0 + exp(-z));           /* logistic */
  case 2: { double t = tanh(z); return 1.0 - t * t; }
  case 3: { double Phi = 0.5 * (1.0 + erf(z * M_SQRT1_2));
            double phi = exp(-0.5 * z * z) * NN_INV_SQRT_2PI;
            return Phi + z * phi; }
  case 4: { double s = 1.0 / (1.0 + exp(-z)); return s * (1.0 + z * (1.0 - s)); }
  default: return 1.0;
  }
}
static inline double nnActD2(int act, double z) {
  switch (act) {
  case 0: return 0.0;
  case 1: { double s = 1.0 / (1.0 + exp(-z)); return s * (1.0 - s); }
  case 2: { double t = tanh(z); return -2.0 * t * (1.0 - t * t); }
  case 3: { double phi = exp(-0.5 * z * z) * NN_INV_SQRT_2PI;
            return phi * (2.0 - z * z); }
  case 4: { double s = 1.0 / (1.0 + exp(-z)); double s1 = s * (1.0 - s);
            return 2.0 * s1 + z * s1 * (1.0 - 2.0 * s); }
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
double nnForward(int id, const double *x) {
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
double nnGrad(int id, const double *x, int m) {
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
double nnHess(int id, const double *x, int m, int l) {
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

/* The fixed-arity rxode2 entry points nn<K> / nn<K>_d<j> / nn<K>_d<j>_d<l>
   (for K = 1..NN_KMAX) are code-generated in nnEvalGen.c and call nnForward /
   nnGrad / nnHess above. */

/* ---- registry management (called from R) --------------------------------- */
SEXP _nlmixr2nn_nnSetMeta(SEXP id, SEXP base, SEXP K, SEXP H, SEXP act) {
  int i = asInteger(id);
  if (i < 0 || i >= NN_MAX) error("nn id out of range");
  nnReg[i].set  = 1;
  nnReg[i].base = asInteger(base);
  nnReg[i].K    = asInteger(K);
  nnReg[i].H    = asInteger(H);
  nnReg[i].act  = asInteger(act);
  nnReg[i].nW   = nnReg[i].H * nnReg[i].K + 2 * nnReg[i].H + 1;
  return ScalarLogical(1);
}

/* C-linkage setter so the torch backend (nnTorch.cpp) can fill a network's
   weight buffer directly, without an R round-trip.  Used at single-threaded
   solve setup / training steps. */
void nnSetWeightsC(int id, const double *w, int n) {
  if (id < 0 || id >= NN_MAX || n <= 0) return;
  double *buf = (double *) malloc((size_t) n * sizeof(double));
  if (buf == NULL) return;
  memcpy(buf, w, (size_t) n * sizeof(double));
  if (nnReg[id].weights != NULL) free(nnReg[id].weights);
  nnReg[id].weights = buf;
  nnReg[id].nW = n;
  nnReg[id].hasW = 1;
}

/* set (or update) a network's externally-owned weight buffer */
SEXP _nlmixr2nn_nnSetWeights(SEXP id, SEXP vals) {
  int i = asInteger(id);
  if (i < 0 || i >= NN_MAX) error("nn id out of range");
  int n = LENGTH(vals);
  double *buf = (double *) malloc((size_t) n * sizeof(double));
  if (buf == NULL) error("could not allocate nn weight buffer");
  memcpy(buf, REAL(vals), (size_t) n * sizeof(double));
  if (nnReg[i].weights != NULL) free(nnReg[i].weights);
  nnReg[i].weights = buf;
  nnReg[i].nW = n;
  nnReg[i].hasW = 1;
  return ScalarLogical(1);
}

SEXP _nlmixr2nn_nnClearMeta(void) {
  for (int i = 0; i < NN_MAX; i++) {
    nnReg[i].set = 0;
    nnReg[i].hasW = 0;
    if (nnReg[i].weights != NULL) { free(nnReg[i].weights); nnReg[i].weights = NULL; }
  }
  return ScalarLogical(1);
}

/* Single dispatcher for direct R evaluation (the R nn<K>* functions call this):
   kind 0 = forward, 1 = gradient w.r.t. input j, 2 = Hessian w.r.t. inputs j,l.
   x is the length-K input vector; id/kind/j/l are scalars. */
SEXP _nlmixr2nn_nnEval(SEXP id, SEXP x, SEXP kind, SEXP j, SEXP l) {
  int i = asInteger(id), K = LENGTH(x), knd = asInteger(kind);
  int jj = asInteger(j), ll = asInteger(l);
  double *px = REAL(x);
  double val;
  if (knd == 1) val = nnGrad(i, px, jj);
  else if (knd == 2) val = nnHess(i, px, jj, ll);
  else val = nnForward(i, px);
  (void) K;
  return ScalarReal(val);
}
