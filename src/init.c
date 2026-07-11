#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

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
extern SEXP _rxode2nn_nnClearMeta(void);

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
    {"_rxode2nn_nnClearMeta",(DL_FUNC) &_rxode2nn_nnClearMeta,0},
    {NULL, NULL, 0}
  };
  R_registerRoutines(dll, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

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
