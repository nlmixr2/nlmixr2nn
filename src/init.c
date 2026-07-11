#include <stdint.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>
#include <rxode2parseStruct.h>   /* rx_solve (C-safe) */
#include "rxode2nn.h"            /* nnParLoader + C bridge wrappers */

/* rxode2 function-pointer-table installer (defined in rxode2nnPtr.cpp) */
extern SEXP _rxode2nn_iniRxodePtrs(SEXP);
SEXP _rxode2nn_registerLoader(void);

/* probe (validation) */
extern double nnprobe(double, double);
extern double nnnpars(double);
extern SEXP _rxode2nn_nnprobe(SEXP, SEXP);
extern SEXP _rxode2nn_nnnpars(SEXP);

/* MLP scalar entry points */
extern double nn1(double, double);
extern double nn1_d1(double, double);
extern double nn1_d1_d1(double, double);
extern double nn2(double, double, double);
extern double nn2_d1(double, double, double);
extern double nn2_d2(double, double, double);
extern double nn2_d1_d1(double, double, double);
extern double nn2_d1_d2(double, double, double);
extern double nn2_d2_d2(double, double, double);

/* SEXP wrappers */
extern SEXP _rxode2nn_nn1(SEXP, SEXP);
extern SEXP _rxode2nn_nn1_d1(SEXP, SEXP);
extern SEXP _rxode2nn_nn1_d1_d1(SEXP, SEXP);
extern SEXP _rxode2nn_nn2(SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nn2_d1(SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nn2_d2(SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nn2_d1_d1(SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nn2_d1_d2(SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nn2_d2_d2(SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nnSetMeta(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nnSetWeights(SEXP, SEXP);
extern SEXP _rxode2nn_nnClearMeta(void);
extern SEXP _rxode2nn_nnUnregisterLoader(void);

/* torch backend (nnTorch.cpp) */
extern SEXP _rxode2nn_nnTorchProbe(SEXP);
extern SEXP _rxode2nn_nnTorchAvailable(void);
extern SEXP _rxode2nn_nnTorchInit(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _rxode2nn_nnTorchFree(SEXP);
extern SEXP _rxode2nn_nnTorchSync(SEXP);
extern SEXP _rxode2nn_nnTorchGetWeights(SEXP);
extern SEXP _rxode2nn_nnTorchSetWeights(SEXP, SEXP);
extern SEXP _rxode2nn_nnTorchForward(SEXP, SEXP);
extern SEXP _rxode2nn_nnTorchSave(SEXP, SEXP);
extern SEXP _rxode2nn_nnTorchLoad(SEXP, SEXP);

void R_init_rxode2nn(DllInfo *dll) {
  static const R_CallMethodDef callMethods[] = {
    {"_rxode2nn_nnprobe",    (DL_FUNC) &_rxode2nn_nnprobe,    2},
    {"_rxode2nn_nnnpars",    (DL_FUNC) &_rxode2nn_nnnpars,    1},
    {"_rxode2nn_nn1",        (DL_FUNC) &_rxode2nn_nn1,        2},
    {"_rxode2nn_nn1_d1",     (DL_FUNC) &_rxode2nn_nn1_d1,     2},
    {"_rxode2nn_nn1_d1_d1",  (DL_FUNC) &_rxode2nn_nn1_d1_d1,  2},
    {"_rxode2nn_nn2",        (DL_FUNC) &_rxode2nn_nn2,        3},
    {"_rxode2nn_nn2_d1",     (DL_FUNC) &_rxode2nn_nn2_d1,     3},
    {"_rxode2nn_nn2_d2",     (DL_FUNC) &_rxode2nn_nn2_d2,     3},
    {"_rxode2nn_nn2_d1_d1",  (DL_FUNC) &_rxode2nn_nn2_d1_d1,  3},
    {"_rxode2nn_nn2_d1_d2",  (DL_FUNC) &_rxode2nn_nn2_d1_d2,  3},
    {"_rxode2nn_nn2_d2_d2",  (DL_FUNC) &_rxode2nn_nn2_d2_d2,  3},
    {"_rxode2nn_nnSetMeta",  (DL_FUNC) &_rxode2nn_nnSetMeta,  5},
    {"_rxode2nn_nnSetWeights",(DL_FUNC) &_rxode2nn_nnSetWeights, 2},
    {"_rxode2nn_nnClearMeta",(DL_FUNC) &_rxode2nn_nnClearMeta,0},
    {"_rxode2nn_nnUnregisterLoader",(DL_FUNC) &_rxode2nn_nnUnregisterLoader,0},
    {"_rxode2nn_iniRxodePtrs",(DL_FUNC) &_rxode2nn_iniRxodePtrs,1},
    {"_rxode2nn_registerLoader",(DL_FUNC) &_rxode2nn_registerLoader,0},
    {"_rxode2nn_nnTorchProbe",(DL_FUNC) &_rxode2nn_nnTorchProbe,1},
    {"_rxode2nn_nnTorchAvailable",(DL_FUNC) &_rxode2nn_nnTorchAvailable,0},
    {"_rxode2nn_nnTorchInit",(DL_FUNC) &_rxode2nn_nnTorchInit,5},
    {"_rxode2nn_nnTorchFree",(DL_FUNC) &_rxode2nn_nnTorchFree,1},
    {"_rxode2nn_nnTorchSync",(DL_FUNC) &_rxode2nn_nnTorchSync,1},
    {"_rxode2nn_nnTorchGetWeights",(DL_FUNC) &_rxode2nn_nnTorchGetWeights,1},
    {"_rxode2nn_nnTorchSetWeights",(DL_FUNC) &_rxode2nn_nnTorchSetWeights,2},
    {"_rxode2nn_nnTorchForward",(DL_FUNC) &_rxode2nn_nnTorchForward,2},
    {"_rxode2nn_nnTorchSave",(DL_FUNC) &_rxode2nn_nnTorchSave,2},
    {"_rxode2nn_nnTorchLoad",(DL_FUNC) &_rxode2nn_nnTorchLoad,2},
    {NULL, NULL, 0}
  };
  R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

  /* NB: the par-loader hook is registered from .onLoad (_rxode2nn_registerLoader),
     after the rxode2 pointer table is installed -- the rxRegisterParLoader
     pointer is NULL at R_init time.  The nn<K> functions below stay on
     R_RegisterCCallable: that is rxode2's custom-function resolution path used by
     generated model code (see the mm vignette). */
  R_RegisterCCallable("rxode2nn", "nnprobe",   (DL_FUNC) &nnprobe);
  R_RegisterCCallable("rxode2nn", "nnnpars",   (DL_FUNC) &nnnpars);
  R_RegisterCCallable("rxode2nn", "nn1",       (DL_FUNC) &nn1);
  R_RegisterCCallable("rxode2nn", "nn1_d1",    (DL_FUNC) &nn1_d1);
  R_RegisterCCallable("rxode2nn", "nn1_d1_d1", (DL_FUNC) &nn1_d1_d1);
  R_RegisterCCallable("rxode2nn", "nn2",       (DL_FUNC) &nn2);
  R_RegisterCCallable("rxode2nn", "nn2_d1",    (DL_FUNC) &nn2_d1);
  R_RegisterCCallable("rxode2nn", "nn2_d2",    (DL_FUNC) &nn2_d2);
  R_RegisterCCallable("rxode2nn", "nn2_d1_d1", (DL_FUNC) &nn2_d1_d1);
  R_RegisterCCallable("rxode2nn", "nn2_d1_d2", (DL_FUNC) &nn2_d1_d2);
  R_RegisterCCallable("rxode2nn", "nn2_d2_d2", (DL_FUNC) &nn2_d2_d2);
}

/* Register the par-loader hook with rxode2; called from .onLoad after the
   rxode2 pointer table is installed (so rxRegisterParLoader is non-NULL). */
SEXP _rxode2nn_registerLoader(void) {
  rxode2nnRegisterLoader(nnParLoader);
  return R_NilValue;
}

/* Called from .onUnload before the DLL is removed, so rxode2 does not retain a
   dangling pointer to nnParLoader. */
SEXP _rxode2nn_nnUnregisterLoader(void) {
  rxode2nnRemoveLoader(nnParLoader);
  return R_NilValue;
}
