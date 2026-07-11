#ifndef RXODE2NN_H
#define RXODE2NN_H
/* Thin C-linkage bridge so the plain-C files (nnEval.c, init.c) can use rxode2's
   function-pointer table without including rxode2ptr.h -> rxode2.h, which pulls
   in Rcpp.h (C++ only).  The wrappers are defined in the C++ TU rxode2nnPtr.cpp.
   Requires rx_solve to be defined already (include rxode2parseStruct.h first). */
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*rxode2nn_parLoader_t)(rx_solve *rx, double *gpars, int npars, int ncols);

rx_solve *rxode2nnGetRxSolve(void);
void rxode2nnRegisterLoader(rxode2nn_parLoader_t cb);
void rxode2nnRemoveLoader(rxode2nn_parLoader_t cb);

/* implemented in nnEval.c */
void nnParLoader(rx_solve *rx, double *gpars, int npars, int ncols);
void nnSetWeightsC(int id, const double *w, int n);

#ifdef __cplusplus
}
#endif
#endif
