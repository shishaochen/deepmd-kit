#include "prod_nb_grad.h"

namespace deepmd {

template<typename FPTYPE>
void prod_force_grad_nb_cpu(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const int *nlist, 
    const int nloc, 
    const int nnei) {
  const int ndescrpt = nnei * 3;
  
  // reset the frame to 0
  for (int ii = 0; ii < nloc; ++ii) {
    for (int aa = 0; aa < ndescrpt; ++aa) {
      grad_net[ii * ndescrpt + aa] = 0;
    }
  }

  // compute grad of one frame
  for (int i_idx = 0; i_idx < nloc; ++i_idx) {
    // deriv wrt center atom
    for (int aa = 0; aa < nnei; ++aa) {
      for (int dd = 0; dd < 3; ++dd) {
        grad_net[i_idx * ndescrpt + aa * 3 + dd] += grad[i_idx * 3 + dd];
      }
    }

    // loop over neighbors
    for (int jj = 0; jj < nnei; ++jj) {
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
	    for (int dd = 0; dd < 3; ++dd) {
	      grad_net[i_idx * ndescrpt + jj * 3 + dd] -= grad[j_idx * 3 + dd];
	    }
    }
  }
}

template<typename FPTYPE>
void prod_virial_grad_nb_cpu(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nnei) {
  const int ndescrpt = nnei * 3;

  // reset the frame to 0
  for (int ii = 0; ii < nloc; ++ii) {
    for (int aa = 0; aa < ndescrpt; ++aa) {
      grad_net[ii * ndescrpt + aa] = 0;
    }
  }      

  // compute grad of one frame
  for (int i_idx = 0; i_idx < nloc; ++i_idx) {
    // deriv wrt neighbors
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      for (int dd0 = 0; dd0 < 3; ++dd0) {
        for (int dd1 = 0; dd1 < 3; ++dd1) {
          grad_net[i_idx * ndescrpt + jj * 3 + dd0] -= grad[dd0 * 3 + dd1] * rij[i_idx * ndescrpt + jj * 3 + dd1];
        }
      }
    }
  }
}


template
void prod_force_grad_nb_cpu<double>(
    double *grad_net,
    const double *grad,
    const int *nlist,
    const int nloc,
    const int nnei);

template
void prod_force_grad_nb_cpu<float>(
    float *grad_net,
    const float *grad,
    const int *nlist,
    const int nloc,
    const int nnei);

template
void prod_virial_grad_nb_cpu(
    double *grad_net,
    const double *grad,
    const double *rij,
    const int *nlist,
    const int nloc,
    const int nnei);

template
void prod_virial_grad_nb_cpu(
    float *grad_net,
    const float *grad,
    const float *rij,
    const int *nlist,
    const int nloc,
    const int nnei);

}