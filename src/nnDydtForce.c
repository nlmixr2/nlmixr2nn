#include <stdint.h>
#include <R.h>
#include <Rinternals.h>
#include <rxode2parseStruct.h>   /* rx_solve (C-safe) */
#include "nlmixr2nn.h"

/* dydt forcing hook: rxode2's generated model calls every registered hook at the
   end of its RHS so a plugin can ADD forcing to state derivatives -- the b_j term
   for NN-weight forward-sensitivity (variational) states.  This file currently
   holds only a configurable TEST hook that validates the rxode2 mechanism (B1);
   the real nn() forcing assembly (b_j = dR/dg * nnWeightGrad) lands in a later
   phase (B3). */

static int    _nnTestForceState = -1;   /* state index to force (-1 = disabled) */
static double _nnTestForceVal   = 0.0;   /* constant added to that derivative */

/* Adds a constant to one state's derivative each RHS evaluation.  Thread-safe:
   reads only its own file-static config, writes only its own dydt slot. */
static void nnTestDydtForce(int *neq, double t, double *y, double *dydt) {
  (void) t; (void) y;
  int s = _nnTestForceState;
  if (s >= 0 && s < neq[0]) dydt[s] += _nnTestForceVal;
}

/* Enable (on != 0) or disable the test hook, setting the forced state + constant.
   Registers/removes nnTestDydtForce with rxode2 via the function-pointer table. */
SEXP _nlmixr2nn_testDydtForce(SEXP stateIdx, SEXP val, SEXP on) {
  if (Rf_asLogical(on)) {
    _nnTestForceState = Rf_asInteger(stateIdx);
    _nnTestForceVal   = Rf_asReal(val);
    nlmixr2nnRegisterDydtForce(nnTestDydtForce);
  } else {
    nlmixr2nnRemoveDydtForce(nnTestDydtForce);
    _nnTestForceState = -1;
    _nnTestForceVal   = 0.0;
  }
  return R_NilValue;
}
