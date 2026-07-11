#define STRICT_R_HEADERS
#include <stdint.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <rxode2parseStruct.h>

/* Probe functions used to validate that a package-exported rxode2 custom
   function can read the current solve's parameter vector (par_ptr) during a
   solve.  NN weights will be population thetas (identical across subjects), so
   reading subjects[0] is representative and thread-safe (read-only). */

static rx_solve *(*getRxSolve_fn)(void) = NULL;

static rx_solve *nnGetRx(void) {
  if (getRxSolve_fn == NULL) {
    getRxSolve_fn = (rx_solve *(*)(void)) R_GetCCallable("rxode2", "getRxSolve_");
  }
  return getRxSolve_fn();
}

/* nnprobe(idx, x): return par_ptr[idx] of subject 0 in the current solve.
   x is ignored (keeps the function arity at 2). */
double nnprobe(double idx, double x) {
  (void) x;
  rx_solve *rx = nnGetRx();
  if (rx == NULL) return -999.0;
  int i = (int) idx;
  if (i < 0 || i >= rx->npars) return -888.0;
  return rx->subjects[0].par_ptr[i];
}

/* nnnpars(x): return the number of parameters (length of par_ptr). */
double nnnpars(double x) {
  (void) x;
  rx_solve *rx = nnGetRx();
  if (rx == NULL) return -999.0;
  return (double) rx->npars;
}

SEXP _rxode2nn_nnprobe(SEXP idx, SEXP x) {
  int n = LENGTH(idx);
  SEXP out = PROTECT(allocVector(REALSXP, n));
  double *pidx = REAL(idx), *px = REAL(x), *res = REAL(out);
  for (int i = 0; i < n; i++) res[i] = nnprobe(pidx[i], px[i]);
  UNPROTECT(1);
  return out;
}

SEXP _rxode2nn_nnnpars(SEXP x) {
  int n = LENGTH(x);
  SEXP out = PROTECT(allocVector(REALSXP, n));
  double *px = REAL(x), *res = REAL(out);
  for (int i = 0; i < n; i++) res[i] = nnnpars(px[i]);
  UNPROTECT(1);
  return out;
}
