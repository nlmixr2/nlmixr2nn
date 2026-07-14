// Consume rxode2's C entry points through its function-pointer table (see
// rxode2/CLAUDE.md and inst/include/rxode2ptr.h) instead of R_RegisterCCallable.
// This translation unit defines this package's copies of the rxode2 pointers
// (getRxSolve_, rxRegisterParLoader, rxRemoveParLoader, ...) and the installer
// _nlmixr2nn_iniRxodePtrs(), which .onLoad calls with rxode2::.rxode2ptrs().
// Rcpp must precede rxode2ptr.h -> rxode2.h (which pulls rxode2_RcppExports.h);
// including R.h first (as rxode2.h would) breaks Rcpp -- match nlmixr2est's order.
#include <Rcpp.h>

#define iniRxodePtrs0 _nlmixr2nn_iniRxodePtrs0
#include <rxode2ptr.h>
#include "nlmixr2nn.h"

extern "C" {
#define iniRxodePtrs _nlmixr2nn_iniRxodePtrs
  iniRxode2ptr

  // C wrappers over the table so the plain-C files avoid rxode2ptr.h/Rcpp.
  rx_solve *nlmixr2nnGetRxSolve(void) {
    return (getRxSolve_ != NULL) ? getRxSolve_() : (rx_solve *) NULL;
  }
  void nlmixr2nnRegisterLoader(nlmixr2nn_parLoader_t cb) {
    // register NAMED so the nn weight loader runs ONLY for models flagged
    // "nlmixr2nn:nnParLoader" (via rxParLoader()), never clobbering an unrelated
    // model's par_ptr.  Fall back to the unnamed (always-run) registration on an
    // older rxode2 that lacks the named entry.
    if (rxRegisterParLoaderNamed != NULL) {
      rxRegisterParLoaderNamed("nlmixr2nn:nnParLoader", cb);
    } else if (rxRegisterParLoader != NULL) {
      rxRegisterParLoader(cb);
    }
  }
  void nlmixr2nnRemoveLoader(nlmixr2nn_parLoader_t cb) {
    if (rxRemoveParLoader != NULL) rxRemoveParLoader(cb);
  }
  void nlmixr2nnRegisterDydtForce(nlmixr2nn_dydtForce_t cb) {
    if (rxRegisterDydtForce != NULL) rxRegisterDydtForce(cb);
  }
  void nlmixr2nnRemoveDydtForce(nlmixr2nn_dydtForce_t cb) {
    if (rxRemoveDydtForce != NULL) rxRemoveDydtForce(cb);
  }
}
