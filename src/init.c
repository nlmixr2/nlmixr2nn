#include <stdint.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>
#include <rxode2parseStruct.h>   /* rx_solve (C-safe) */
#include "nlmixr2nn.h"            /* nnParLoader + C bridge wrappers */

/* rxode2 function-pointer-table installer (defined in nlmixr2nnPtr.cpp) */
extern SEXP _nlmixr2nn_iniRxodePtrs(SEXP);
SEXP _nlmixr2nn_registerLoader(void);

/* nlmixr2est likelihood-contribution bundle (nlmixr2nnContrib.c) */
extern SEXP _nlmixr2nn_iniLikContrib(SEXP);
extern SEXP _nlmixr2nn_registerContrib(void);
extern SEXP _nlmixr2nn_removeContrib(void);
extern SEXP _nlmixr2nn_getContrib(void);
extern SEXP _nlmixr2nn_setDfdwLhs(SEXP);
extern SEXP _nlmixr2nn_getDLLdw(void);

/* probe (validation) */
extern double nnprobe(double, double);
extern double nnnpars(double);
extern SEXP _nlmixr2nn_nnprobe(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnnpars(SEXP);

/* nn<K> scalar entry points are code-generated (nnEvalGen.c) and registered via
   nnRegisterCallables(); direct R evaluation goes through one dispatcher. */
extern SEXP _nlmixr2nn_nnEval(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nlmixr2nn_nnWeightGrad(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnWeightGradW(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nlmixr2nn_nnSetMeta(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nlmixr2nn_nnSetWeights(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnClearMeta(void);
extern SEXP _nlmixr2nn_nnUnregisterLoader(void);

/* dydt forcing hook test entry (nnDydtForce.c) */
extern SEXP _nlmixr2nn_testDydtForce(SEXP, SEXP, SEXP);

/* torch backend (nnTorch.cpp) */
extern SEXP _nlmixr2nn_nnTorchProbe(SEXP);
extern SEXP _nlmixr2nn_nnTorchAvailable(void);
extern SEXP _nlmixr2nn_nnTorchInit(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchFree(SEXP);
extern SEXP _nlmixr2nn_nnTorchSync(SEXP);
extern SEXP _nlmixr2nn_nnTorchGetWeights(SEXP);
extern SEXP _nlmixr2nn_nnTorchSetWeights(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchForward(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchSave(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchLoad(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchOptInit(SEXP, SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchZeroGrad(SEXP);
extern SEXP _nlmixr2nn_nnTorchForwardBatch(SEXP, SEXP, SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchBackward(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchGetGrad(SEXP);
extern SEXP _nlmixr2nn_nnTorchSetGrad(SEXP, SEXP);
extern SEXP _nlmixr2nn_nnTorchStep(SEXP);

void R_init_nlmixr2nn(DllInfo *dll) {
  static const R_CallMethodDef callMethods[] = {
    {"_nlmixr2nn_nnprobe",    (DL_FUNC) &_nlmixr2nn_nnprobe,    2},
    {"_nlmixr2nn_nnnpars",    (DL_FUNC) &_nlmixr2nn_nnnpars,    1},
    {"_nlmixr2nn_nnEval",     (DL_FUNC) &_nlmixr2nn_nnEval,     5},
    {"_nlmixr2nn_nnWeightGrad",(DL_FUNC) &_nlmixr2nn_nnWeightGrad, 2},
    {"_nlmixr2nn_nnWeightGradW",(DL_FUNC) &_nlmixr2nn_nnWeightGradW, 5},
    {"_nlmixr2nn_nnSetMeta",  (DL_FUNC) &_nlmixr2nn_nnSetMeta,  5},
    {"_nlmixr2nn_nnSetWeights",(DL_FUNC) &_nlmixr2nn_nnSetWeights, 2},
    {"_nlmixr2nn_nnClearMeta",(DL_FUNC) &_nlmixr2nn_nnClearMeta,0},
    {"_nlmixr2nn_nnUnregisterLoader",(DL_FUNC) &_nlmixr2nn_nnUnregisterLoader,0},
    {"_nlmixr2nn_testDydtForce",(DL_FUNC) &_nlmixr2nn_testDydtForce,3},
    {"_nlmixr2nn_iniRxodePtrs",(DL_FUNC) &_nlmixr2nn_iniRxodePtrs,1},
    {"_nlmixr2nn_registerLoader",(DL_FUNC) &_nlmixr2nn_registerLoader,0},
    {"_nlmixr2nn_iniLikContrib",(DL_FUNC) &_nlmixr2nn_iniLikContrib,1},
    {"_nlmixr2nn_registerContrib",(DL_FUNC) &_nlmixr2nn_registerContrib,0},
    {"_nlmixr2nn_removeContrib",(DL_FUNC) &_nlmixr2nn_removeContrib,0},
    {"_nlmixr2nn_getContrib",(DL_FUNC) &_nlmixr2nn_getContrib,0},
    {"_nlmixr2nn_setDfdwLhs",(DL_FUNC) &_nlmixr2nn_setDfdwLhs,1},
    {"_nlmixr2nn_getDLLdw",(DL_FUNC) &_nlmixr2nn_getDLLdw,0},
    {"_nlmixr2nn_nnTorchProbe",(DL_FUNC) &_nlmixr2nn_nnTorchProbe,1},
    {"_nlmixr2nn_nnTorchAvailable",(DL_FUNC) &_nlmixr2nn_nnTorchAvailable,0},
    {"_nlmixr2nn_nnTorchInit",(DL_FUNC) &_nlmixr2nn_nnTorchInit,5},
    {"_nlmixr2nn_nnTorchFree",(DL_FUNC) &_nlmixr2nn_nnTorchFree,1},
    {"_nlmixr2nn_nnTorchSync",(DL_FUNC) &_nlmixr2nn_nnTorchSync,1},
    {"_nlmixr2nn_nnTorchGetWeights",(DL_FUNC) &_nlmixr2nn_nnTorchGetWeights,1},
    {"_nlmixr2nn_nnTorchSetWeights",(DL_FUNC) &_nlmixr2nn_nnTorchSetWeights,2},
    {"_nlmixr2nn_nnTorchForward",(DL_FUNC) &_nlmixr2nn_nnTorchForward,2},
    {"_nlmixr2nn_nnTorchSave",(DL_FUNC) &_nlmixr2nn_nnTorchSave,2},
    {"_nlmixr2nn_nnTorchLoad",(DL_FUNC) &_nlmixr2nn_nnTorchLoad,2},
    {"_nlmixr2nn_nnTorchOptInit",(DL_FUNC) &_nlmixr2nn_nnTorchOptInit,3},
    {"_nlmixr2nn_nnTorchZeroGrad",(DL_FUNC) &_nlmixr2nn_nnTorchZeroGrad,1},
    {"_nlmixr2nn_nnTorchForwardBatch",(DL_FUNC) &_nlmixr2nn_nnTorchForwardBatch,4},
    {"_nlmixr2nn_nnTorchBackward",(DL_FUNC) &_nlmixr2nn_nnTorchBackward,5},
    {"_nlmixr2nn_nnTorchGetGrad",(DL_FUNC) &_nlmixr2nn_nnTorchGetGrad,1},
    {"_nlmixr2nn_nnTorchSetGrad",(DL_FUNC) &_nlmixr2nn_nnTorchSetGrad,2},
    {"_nlmixr2nn_nnTorchStep",(DL_FUNC) &_nlmixr2nn_nnTorchStep,1},
    {NULL, NULL, 0}
  };
  R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

  /* NB: the par-loader hook is registered from .onLoad (_nlmixr2nn_registerLoader),
     after the rxode2 pointer table is installed -- the rxRegisterParLoader
     pointer is NULL at R_init time.  The nn<K> functions below stay on
     R_RegisterCCallable: that is rxode2's custom-function resolution path used by
     generated model code (see the mm vignette). */
  R_RegisterCCallable("nlmixr2nn", "nnprobe",   (DL_FUNC) &nnprobe);
  R_RegisterCCallable("nlmixr2nn", "nnnpars",   (DL_FUNC) &nnnpars);
  nnRegisterCallables();   /* nn<K> family, generated in nnEvalGen.c */
}

/* Register the par-loader hook with rxode2; called from .onLoad after the
   rxode2 pointer table is installed (so rxRegisterParLoader is non-NULL). */
SEXP _nlmixr2nn_registerLoader(void) {
  nlmixr2nnRegisterLoader(nnParLoader);
  return R_NilValue;
}

/* Called from .onUnload before the DLL is removed, so rxode2 does not retain a
   dangling pointer to nnParLoader. */
SEXP _nlmixr2nn_nnUnregisterLoader(void) {
  nlmixr2nnRemoveLoader(nnParLoader);
  return R_NilValue;
}
