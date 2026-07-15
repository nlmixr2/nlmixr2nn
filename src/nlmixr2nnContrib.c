// nlmixr2nn's likelihood-contribution bundle.  Installs nlmixr2est's registry
// entry points as function pointers (via nlmixr2estLikContribPtr.h) and
// registers a bundle whose per-observation hook records the cotangent d(LL)/d(f)
// -- the boundary condition an adjoint sweep + torch VJP will turn into the
// neural-network weight gradient d(LL)/d(w).  Registration is single-threaded
// (from .onLoad); the obs hook runs inside nlmixr2est's parallel region.
#include <R.h>
#include <Rinternals.h>
#include <string.h>
#include <nlmixr2estLikContribPtr.h>

// define the pointer globals + iniNlmixr2estLikContrib(p)
iniNlmixr2estLikContribGlobals

// --- contribution bundle -----------------------------------------------------
// The per-observation hook records the EXACT error-model cotangent d(LL)/d(f)
// computed by the inner fit (correct for ANY residual/likelihood model, not just
// Gaussian).  For the NN weight step, dLL/dw = sum_obs dLL/df * d(f)/d(w), where
// dLL/df is captured here (at the fitted etas) and d(f)/d(w) comes from the
// augmented sensitivity solve.
static double _nnSumDLLdf;
static int _nnNObs, _nnNBegin, _nnNEnd;

// Per-(subject id, obs index k) capture, last-write-wins, so the final (converged)
// objective eval's cotangents survive the many optimization/FD evals.  Flat store
// indexed by id*_NN_KSTRIDE + k.
#define _NN_KSTRIDE 1024
static int    _nnCapOn = 0;
static double *_nnCapVal = NULL;    // dLL/df at (id,k)
static char   *_nnCapHas = NULL;    // 1 if (id,k) seen
static int     _nnCapCap = 0;       // capacity (entries)
static int     _nnCapMax = -1;      // highest index written

static void _nnCapEnsure(int idx) {
  if (idx < _nnCapCap) return;
  int newCap = (idx + _NN_KSTRIDE);          // grow with headroom
  double *v = (double *) R_chk_realloc(_nnCapVal, (size_t) newCap * sizeof(double));
  char   *h = (char *)   R_chk_realloc(_nnCapHas, (size_t) newCap * sizeof(char));
  memset(h + _nnCapCap, 0, (size_t)(newCap - _nnCapCap) * sizeof(char));
  _nnCapVal = v; _nnCapHas = h; _nnCapCap = newCap;
}

static void _nnBegin(const nlmixrLikSubj *s) { (void) s; _nnNBegin++; }
static void _nnEnd(const nlmixrLikSubj *s) { (void) s; _nnNEnd++; }
static void _nnObs(nlmixrLikObs *o) {
  _nnNObs++;
  _nnSumDLLdf += o->dLL_df;
  if (_nnCapOn && o->id >= 0 && o->k >= 0 && o->k < _NN_KSTRIDE) {
    int idx = o->id * _NN_KSTRIDE + o->k;
    _nnCapEnsure(idx);
    _nnCapVal[idx] = o->dLL_df;
    _nnCapHas[idx] = 1;
    if (idx > _nnCapMax) _nnCapMax = idx;
  }
}
static const nlmixrLikContrib _nnContribBundle = { _nnBegin, _nnObs, _nnEnd };

// Enable/disable capture and clear the store (call before an inner fit whose
// converged cotangents are wanted).
SEXP _nlmixr2nn_capReset(SEXP onSxp) {
  _nnCapOn = (Rf_length(onSxp) >= 1 && Rf_asLogical(onSxp) == TRUE) ? 1 : 0;
  if (_nnCapHas != NULL && _nnCapCap > 0) memset(_nnCapHas, 0, (size_t) _nnCapCap * sizeof(char));
  _nnCapMax = -1;
  return R_NilValue;
}

// Return the captured cotangents as a list(id=, k=, dLLdf=) for present entries.
SEXP _nlmixr2nn_capGet(void) {
  int n = 0;
  for (int i = 0; i <= _nnCapMax; ++i) if (_nnCapHas[i]) n++;
  SEXP id = PROTECT(Rf_allocVector(INTSXP, n));
  SEXP kk = PROTECT(Rf_allocVector(INTSXP, n));
  SEXP dv = PROTECT(Rf_allocVector(REALSXP, n));
  int j = 0;
  for (int i = 0; i <= _nnCapMax; ++i) {
    if (!_nnCapHas[i]) continue;
    INTEGER(id)[j] = i / _NN_KSTRIDE;
    INTEGER(kk)[j] = i % _NN_KSTRIDE;
    REAL(dv)[j] = _nnCapVal[i];
    j++;
  }
  SEXP ret = PROTECT(Rf_allocVector(VECSXP, 3));
  SET_VECTOR_ELT(ret, 0, id); SET_VECTOR_ELT(ret, 1, kk); SET_VECTOR_ELT(ret, 2, dv);
  SEXP nm = PROTECT(Rf_allocVector(STRSXP, 3));
  SET_STRING_ELT(nm, 0, Rf_mkChar("id")); SET_STRING_ELT(nm, 1, Rf_mkChar("k"));
  SET_STRING_ELT(nm, 2, Rf_mkChar("dLLdf"));
  Rf_setAttrib(ret, R_NamesSymbol, nm);
  UNPROTECT(5);
  return ret;
}

SEXP _nlmixr2nn_iniLikContrib(SEXP p) { return iniNlmixr2estLikContrib(p); }

SEXP _nlmixr2nn_registerContrib(void) {
  _nnSumDLLdf = 0.0; _nnNObs = _nnNBegin = _nnNEnd = 0;
  if (nlmixrRegisterLikContribP != NULL) nlmixrRegisterLikContribP(&_nnContribBundle);
  return R_NilValue;
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
