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

static void _nnBegin(const nlmixrLikSubj *s) { (void) s; _nnNBegin++; }
static void _nnEnd(const nlmixrLikSubj *s) { (void) s; _nnNEnd++; }
static void _nnObs(nlmixrLikObs *o) {
  _nnNObs++;
  _nnSumDLLdf += o->dLL_df;
}
static const nlmixrLikContrib _nnContribBundle = { _nnBegin, _nnObs, _nnEnd };

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
/* register/remove the inner-block per-subject weight injection (nnInnerWeight in
   nnEval.c) with nlmixr2est via the contribution pointer table (element 5).  A
   no-op when built against an older nlmixr2est (pointer stays NULL). */
extern void nnInnerWeight(int cid, const double *eta, int neta);
SEXP _nlmixr2nn_registerInnerWt(void) {
  if (nlmixrSetInnerWeightFnP != NULL) nlmixrSetInnerWeightFnP(nnInnerWeight);
  return R_NilValue;
}
SEXP _nlmixr2nn_removeInnerWt(void) {
  if (nlmixrSetInnerWeightFnP != NULL) nlmixrSetInnerWeightFnP(NULL);
  return R_NilValue;
}

SEXP _nlmixr2nn_getContrib(void) {
  SEXP r = PROTECT(Rf_allocVector(REALSXP, 4));
  REAL(r)[0] = (double) _nnNObs;   REAL(r)[1] = _nnSumDLLdf;
  REAL(r)[2] = (double) _nnNBegin; REAL(r)[3] = (double) _nnNEnd;
  UNPROTECT(1);
  return r;
}
