// nlmixr2nn's likelihood-contribution bundle.  Installs nlmixr2est's registry
// entry points as function pointers (via nlmixr2estLikContribPtr.h) and
// registers a bundle whose per-observation hook records the cotangent d(LL)/d(f)
// -- the boundary condition an adjoint sweep + torch VJP will turn into the
// neural-network weight gradient d(LL)/d(w).  Registration is single-threaded
// (from .onLoad); the obs hook runs inside nlmixr2est's parallel region.
#include <R.h>
#include <Rinternals.h>
#include <nlmixr2estLikContribPtr.h>

// define the pointer globals + iniNlmixr2estLikContrib(p)
iniNlmixr2estLikContribGlobals

// --- contribution bundle -----------------------------------------------------
// First version: record the per-observation cotangents so the wiring can be
// validated end-to-end.  The adjoint -> torch weight gradient is layered on top.
static double _nnSumDLLdf;
static int _nnNObs, _nnNBegin, _nnNEnd;

// d(LL)/d(w) accumulation: the augmented model emits d(f)/d(w_j) as extra lhs
// (rx_dfdw_<j>_); their lhs indices are configured from R (setDfdwLhs).  Each
// observation contributes d(LL)/d(f) * d(f)/d(w_j) to d(LL)/d(w_j) -- the
// forward-sensitivity weight gradient assembled during the fit.
#define NN_MAX_DFDW 4096
static int _nnDfdwIdx[NN_MAX_DFDW];
static int _nnNDfdw = 0;
static double _nnDLLdw[NN_MAX_DFDW];

static void _nnBegin(const nlmixrLikSubj *s) { (void) s; _nnNBegin++; }
static void _nnEnd(const nlmixrLikSubj *s) { (void) s; _nnNEnd++; }
static void _nnObs(nlmixrLikObs *o) {
  _nnNObs++;
  _nnSumDLLdf += o->dLL_df;
  if (o->lhs != NULL && _nnNDfdw > 0) {
    for (int j = 0; j < _nnNDfdw; ++j) {
      int ix = _nnDfdwIdx[j];
      if (ix >= 0 && ix < o->nlhs) _nnDLLdw[j] += o->dLL_df * o->lhs[ix];
    }
  }
}
static const nlmixrLikContrib _nnContribBundle = { _nnBegin, _nnObs, _nnEnd };

SEXP _nlmixr2nn_iniLikContrib(SEXP p) { return iniNlmixr2estLikContrib(p); }

SEXP _nlmixr2nn_registerContrib(void) {
  _nnSumDLLdf = 0.0; _nnNObs = _nnNBegin = _nnNEnd = 0;
  for (int j = 0; j < _nnNDfdw; ++j) _nnDLLdw[j] = 0.0;
  if (nlmixrRegisterLikContribP != NULL) nlmixrRegisterLikContribP(&_nnContribBundle);
  return R_NilValue;
}

// configure the lhs indices (0-based) of the d(f)/d(w_j) outputs, and reset the
// d(LL)/d(w) accumulator.
SEXP _nlmixr2nn_setDfdwLhs(SEXP idx) {
  int n = Rf_length(idx);
  if (n > NN_MAX_DFDW) n = NN_MAX_DFDW;
  _nnNDfdw = n;
  int *p = INTEGER(idx);
  for (int j = 0; j < n; ++j) { _nnDfdwIdx[j] = p[j]; _nnDLLdw[j] = 0.0; }
  return R_NilValue;
}

// accumulated d(LL)/d(w) since the last register/setDfdwLhs.
SEXP _nlmixr2nn_getDLLdw(void) {
  SEXP r = PROTECT(Rf_allocVector(REALSXP, _nnNDfdw));
  for (int j = 0; j < _nnNDfdw; ++j) REAL(r)[j] = _nnDLLdw[j];
  UNPROTECT(1);
  return r;
}
SEXP _nlmixr2nn_removeContrib(void) {
  if (nlmixrRemoveLikContribP != NULL) nlmixrRemoveLikContribP(&_nnContribBundle);
  return R_NilValue;
}
SEXP _nlmixr2nn_getContrib(void) {
  SEXP r = PROTECT(Rf_allocVector(REALSXP, 4));
  REAL(r)[0] = (double) _nnNObs;   REAL(r)[1] = _nnSumDLLdf;
  REAL(r)[2] = (double) _nnNBegin; REAL(r)[3] = (double) _nnNEnd;
  UNPROTECT(1);
  return r;
}
