#ifndef RXODE2NN_H
#define RXODE2NN_H
/* Thin C-linkage bridge so the plain-C files (nnEval.c, init.c) can use rxode2's
   function-pointer table without including rxode2ptr.h -> rxode2.h, which pulls
   in Rcpp.h (C++ only).  The wrappers are defined in the C++ TU nlmixr2nnPtr.cpp.
   Requires rx_solve to be defined already (include rxode2parseStruct.h first). */
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nlmixr2nn_parLoader_t)(rx_solve *rx, double *gpars, int npars, int ncols);
typedef void (*nlmixr2nn_dydtForce_t)(int *neq, double t, double *y, double *dydt);

rx_solve *nlmixr2nnGetRxSolve(void);
void nlmixr2nnRegisterLoader(nlmixr2nn_parLoader_t cb);
void nlmixr2nnRemoveLoader(nlmixr2nn_parLoader_t cb);
void nlmixr2nnRegisterDydtForce(nlmixr2nn_dydtForce_t cb);
void nlmixr2nnRemoveDydtForce(nlmixr2nn_dydtForce_t cb);

/* implemented in nnEval.c */
void nnParLoader(rx_solve *rx, double *gpars, int npars, int ncols);
void nnSetWeightsC(int id, const double *w, int n);
double nnForward(int id, const double *x);
double nnGrad(int id, const double *x, int m);
double nnHess(int id, const double *x, int m, int l);
void nnWeightGrad(int id, const double *x, double *g);   /* d(out)/d(each weight) */
double nnWeightGradJ(int id, const double *x, int j);    /* d(out)/d(w_j) scalar */

/* Same math as above, but from an EXPLICIT weight vector rather than the
   registry + par_ptr.  The builtin training backend (nnBuiltin.cpp) owns its
   weights outside the solve, and uses these so there is ONE copy of the
   activation switch and the weight-gradient formula in the package -- a second
   copy in the backend would be free to drift from the one the solve uses.
   Layout is nnWeightLayout order: W1 (H,K) row-major, b1 (H), W2 (1,H), b2. */
double nnForwardWC(int K, int H, int act, const double *w, const double *x);
void nnWeightGradWC(int K, int H, int act, const double *w, const double *x,
                    double *g);

/* generated in nnEvalGen.c: registers the nn<K> scalar entry points with rxode2 */
void nnRegisterCallables(void);

#ifdef __cplusplus
}
#endif
#endif
