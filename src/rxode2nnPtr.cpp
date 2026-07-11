// Consume rxode2's C entry points through its function-pointer table (see
// rxode2/CLAUDE.md and inst/include/rxode2ptr.h) instead of R_RegisterCCallable.
// This translation unit defines this package's copies of the rxode2 pointers
// (getRxSolve_, rxRegisterParLoader, rxRemoveParLoader, ...) and the installer
// _rxode2nn_iniRxodePtrs(), which .onLoad calls with rxode2::.rxode2ptrs().
// Rcpp must precede rxode2ptr.h -> rxode2.h (which pulls rxode2_RcppExports.h);
// including R.h first (as rxode2.h would) breaks Rcpp -- match nlmixr2est's order.
#include <Rcpp.h>

#define iniRxodePtrs0 _rxode2nn_iniRxodePtrs0
#include <rxode2ptr.h>
#include "rxode2nn.h"

extern "C" {
#define iniRxodePtrs _rxode2nn_iniRxodePtrs
  iniRxode2ptr

  // C wrappers over the table so the plain-C files avoid rxode2ptr.h/Rcpp.
  rx_solve *rxode2nnGetRxSolve(void) {
    return (getRxSolve_ != NULL) ? getRxSolve_() : (rx_solve *) NULL;
  }
  void rxode2nnRegisterLoader(rxode2nn_parLoader_t cb) {
    if (rxRegisterParLoader != NULL) rxRegisterParLoader(cb);
  }
  void rxode2nnRemoveLoader(rxode2nn_parLoader_t cb) {
    if (rxRemoveParLoader != NULL) rxRemoveParLoader(cb);
  }
}
