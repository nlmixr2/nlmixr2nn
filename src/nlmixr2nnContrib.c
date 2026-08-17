// nlmixr2nn's likelihood-contribution bundle.  Installs nlmixr2est's registry
// entry points as function pointers (via nlmixr2estLikContribPtr.h) and
// registers a bundle whose per-observation hook records the cotangent d(LL)/d(f)
// -- the boundary condition an adjoint sweep + torch VJP will turn into the
// neural-network weight gradient d(LL)/d(w).  Registration is single-threaded
// (from .onLoad); the obs hook runs inside nlmixr2est's parallel region.
#include <R.h>
#include <Rinternals.h>
#include <string.h>
#include <limits.h>
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

// Per-(subject id, obs index k) capture, last-write-wins, so the final
// (converged) objective eval's cotangents survive the many optimization/FD
// evals that precede it.  Flat store indexed by id*_nnCapStride + k.
//
// THE HOOK RUNS INSIDE nlmixr2est's OpenMP PARALLEL REGION.  It therefore must
// not allocate, must not call into R, and must not resize anything: this store
// used to grow itself with R_chk_realloc() from whichever worker thread saw a
// new index first, which is a data race on the buffer pointer, a use-after-free
// for every other thread mid-write, and an R API call off the main thread.  It
// crashed with "double free or corruption" the moment the path was actually
// exercised.
//
// So the store is sized ONCE from R, on the main thread, before the fit, and
// the hook only ever writes into it.  Distinct (id,k) land in distinct slots,
// so concurrent writes from different threads do not overlap.
static int     _nnCapOn = 0;
static double *_nnCapVal = NULL;    // dLL/df at (id,k)
static char   *_nnCapHas = NULL;    // 1 if (id,k) seen
static int     _nnCapCap = 0;       // capacity (entries)
static int     _nnCapStride = 0;    // k-stride == max obs per subject
static int     _nnCapNId = 0;       // subjects the store was sized for
// Set when the hook sees an index the store was not sized for.  Reported to R
// rather than silently dropped: a missing observation would misalign the whole
// cotangent vector against the augmented solve, which is worse than an error.
static int     _nnCapOverflow = 0;

static void _nnCapFree(void) {
  if (_nnCapVal != NULL) { R_Free(_nnCapVal); _nnCapVal = NULL; }
  if (_nnCapHas != NULL) { R_Free(_nnCapHas); _nnCapHas = NULL; }
  _nnCapCap = _nnCapStride = _nnCapNId = 0;
}

static void _nnBegin(const nlmixrLikSubj *s) { (void) s; _nnNBegin++; }
static void _nnEnd(const nlmixrLikSubj *s) { (void) s; _nnNEnd++; }

// Hot path, parallel region: bounds-check and write.  No allocation, no R API.
static void _nnObs(nlmixrLikObs *o) {
  if (!_nnCapOn) return;
  _nnNObs++;
  _nnSumDLLdf += o->dLL_df;
  if (o->id < 0 || o->k < 0) return;
  if (o->id >= _nnCapNId || o->k >= _nnCapStride) { _nnCapOverflow = 1; return; }
  int idx = o->id * _nnCapStride + o->k;
  if (idx >= _nnCapCap) { _nnCapOverflow = 1; return; }
  _nnCapVal[idx] = o->dLL_df;
  _nnCapHas[idx] = 1;
}
static const nlmixrLikContrib _nnContribBundle = { _nnBegin, _nnObs, _nnEnd };

// Enable/disable capture, and size the store.
//
// `nId` and `kStride` come from the data (subjects, and max observations per
// subject) and are what the hook bounds-check against.  Allocating here -- on
// the main thread, before the fit -- is what lets the hook be allocation-free.
SEXP _nlmixr2nn_capReset(SEXP onSxp, SEXP nIdSxp, SEXP kStrideSxp) {
  int on = (Rf_length(onSxp) >= 1 && Rf_asLogical(onSxp) == TRUE) ? 1 : 0;
  _nnCapOverflow = 0;
  if (!on) {
    _nnCapOn = 0;
    _nnCapFree();
    return R_NilValue;
  }
  int nId = Rf_asInteger(nIdSxp);
  int kStride = Rf_asInteger(kStrideSxp);
  if (nId == NA_INTEGER || kStride == NA_INTEGER || nId <= 0 || kStride <= 0) {
    Rf_error("nlmixr2nn: cotangent capture needs a positive subject count and "
             "observations-per-subject");
  }
  double need = (double) nId * (double) kStride;
  if (need > (double) INT_MAX) {
    Rf_error("nlmixr2nn: cotangent capture store too large (%d subjects x %d "
             "observations)", nId, kStride);
  }
  int cap = nId * kStride;
  if (cap != _nnCapCap) {
    _nnCapFree();
    _nnCapVal = (double *) R_Calloc((size_t) cap, double);
    _nnCapHas = (char *)   R_Calloc((size_t) cap, char);
    _nnCapCap = cap;
  } else {
    memset(_nnCapHas, 0, (size_t) cap * sizeof(char));
  }
  _nnCapNId = nId;
  _nnCapStride = kStride;
  _nnCapOn = 1;
  return R_NilValue;
}

// Return the captured cotangents as a list(id=, k=, dLLdf=) for present entries.
SEXP _nlmixr2nn_capGet(void) {
  if (_nnCapOverflow) {
    Rf_error("nlmixr2nn: the likelihood hook reported an observation outside the "
             "captured range (%d subjects x %d observations).  The cotangent "
             "vector would not line up with the sensitivity solve.",
             _nnCapNId, _nnCapStride);
  }
  int n = 0;
  for (int i = 0; i < _nnCapCap; ++i) if (_nnCapHas[i]) n++;
  SEXP id = PROTECT(Rf_allocVector(INTSXP, n));
  SEXP kk = PROTECT(Rf_allocVector(INTSXP, n));
  SEXP dv = PROTECT(Rf_allocVector(REALSXP, n));
  int j = 0;
  for (int i = 0; i < _nnCapCap; ++i) {
    if (!_nnCapHas[i]) continue;
    INTEGER(id)[j] = i / _nnCapStride;
    INTEGER(kk)[j] = i % _nnCapStride;
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
  _nnCapOn = 0;
  _nnCapFree();
  return R_NilValue;
}

SEXP _nlmixr2nn_getContrib(void) {
  SEXP r = PROTECT(Rf_allocVector(REALSXP, 4));
  REAL(r)[0] = (double) _nnNObs;   REAL(r)[1] = _nnSumDLLdf;
  REAL(r)[2] = (double) _nnNBegin; REAL(r)[3] = (double) _nnNEnd;
  UNPROTECT(1);
  return r;
}
