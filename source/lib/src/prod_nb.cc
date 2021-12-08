#include <cassert>
#include <iostream>
#include <string.h>

#include "errors.h"
#include "fmt_nlist.h"
#include "prod_nb.h"

namespace deepmd {

template<typename FPTYPE>
void prod_nb_mat_cpu(
    FPTYPE *rij,
    int *nlist,
    const FPTYPE *coord,
    const int *type,
    const InputNlist &inlist,
    const int max_nbor_size,
    const int nloc,
    const int nall,
    const float rcut,
    const std::vector<int> sec) {
  const int nnei = sec.back();

  // set & normalize coord
  std::vector<FPTYPE> d_coord3(nall * 3);
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      d_coord3[ii * 3 + dd] = coord[ii * 3 + dd];
    }
  }

  // set type
  std::vector<int> d_type (nall);
  for (int ii = 0; ii < nall; ++ii) {
    d_type[ii] = type[ii];
  }

  // build nlist
  std::vector<std::vector<int > > d_nlist(nloc);

  assert(nloc == inlist.inum);
  for (unsigned ii = 0; ii < nloc; ++ii) {
    d_nlist[ii].reserve(max_nbor_size);
  }
  for (unsigned ii = 0; ii < nloc; ++ii) {
    int i_idx = inlist.ilist[ii];
    for(unsigned jj = 0; jj < inlist.numneigh[ii]; ++jj){
      int j_idx = inlist.firstneigh[ii][jj];
      d_nlist[i_idx].push_back(j_idx);
    }
  }

#pragma omp parallel for 
  for (int ii = 0; ii < nloc; ++ii) {
    std::vector<int> fmt_nlist;
    int ret = format_nlist_i_cpu(fmt_nlist, d_coord3, d_type, ii, d_nlist[ii], rcut, sec);
    std::vector<FPTYPE> d_rij;

    // compute the diff of the neighbors
    d_rij.resize(nnei * 3);
    fill(d_rij.begin(), d_rij.end(), 0.0);
    for (int kk = 0; kk < int(sec.size()) - 1; ++kk) {
        for (int jj = sec[kk]; jj < sec[kk + 1]; ++jj) {
            if (fmt_nlist[jj] < 0) break;
            const int & j_idx = fmt_nlist[jj];
            for (int dd = 0; dd < 3; ++dd) {
                d_rij[jj * 3 + dd] = d_coord3[j_idx * 3 + dd] - d_coord3[ii * 3 + dd];
            }
        }
    }

    // check sizes
    assert (d_rij.size() == nnei * 3);
    assert (fmt_nlist.size() == nnei);
    // record outputs
    for (int jj = 0; jj < nnei * 3; ++jj) {
      rij[ii * nnei * 3 + jj] = d_rij[jj];
    }
    for (int jj = 0; jj < nnei; ++jj) {
      nlist[ii * nnei + jj] = fmt_nlist[jj];
    }
  }
}

template<typename FPTYPE>
void prod_force_nb_cpu(
    FPTYPE *force,
    const FPTYPE *net_deriv,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei) {
  const int ndescrpt = 3 * nnei;
  memset(force, 0.0, sizeof(FPTYPE) * nall * 3);

  // compute force of a frame
  for (int i_idx = 0; i_idx < nloc; ++i_idx) {
    // deriv wrt center atom
    for (int aa = 0; aa < nnei; ++aa) {
      force[i_idx * 3 + 0] += net_deriv[i_idx * ndescrpt + aa * 3 + 0];
      force[i_idx * 3 + 1] += net_deriv[i_idx * ndescrpt + aa * 3 + 1];
      force[i_idx * 3 + 2] += net_deriv[i_idx * ndescrpt + aa * 3 + 2];
    }
    // deriv wrt neighbors
    for (int jj = 0; jj < nnei; ++jj) {
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      force[j_idx * 3 + 0] -= net_deriv[i_idx * ndescrpt + jj * 3 + 0];
      force[j_idx * 3 + 1] -= net_deriv[i_idx * ndescrpt + jj * 3 + 1];
      force[j_idx * 3 + 2] -= net_deriv[i_idx * ndescrpt + jj * 3 + 2];
    }
  }
}

template<typename FPTYPE>
void prod_virial_nb_cpu(
    FPTYPE *virial,
    FPTYPE *atom_virial,
    const FPTYPE *net_deriv,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei) {
  const int ndescrpt = 3 * nnei;

  for (int ii = 0; ii < 9; ++ ii) {
    virial[ii] = 0.;
  }
  for (int ii = 0; ii < 9 * nall; ++ ii) {
    atom_virial[ii] = 0.;
  }

  // compute virial of a frame
  for (int i_idx = 0; i_idx < nloc; ++i_idx) {
    // deriv wrt neighbors
    for (int jj = 0; jj < nnei; ++jj) {
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      for (int dd0 = 0; dd0 < 3; ++dd0) {
        FPTYPE force = net_deriv[i_idx * ndescrpt + jj * 3 + dd0];
        for (int dd1 = 0; dd1 < 3; ++dd1) {
          FPTYPE tmp_v = force * rij[i_idx * ndescrpt + jj * 3 + dd1];
          virial[dd0 * 3 + dd1] -= tmp_v;
          atom_virial[j_idx * 9 + dd0 * 3 + dd1] -= tmp_v;
        }
      }
    }
  }  
}

template
void prod_nb_mat_cpu<double>(
    double *rij,
    int *nlist,
    const double *coord,
    const int *type,
    const InputNlist &inlist,
    const int max_nbor_size,
    const int nloc,
    const int nall,
    const float rcut,
    const std::vector<int> sec);

template
void prod_nb_mat_cpu<float>(
    float *rij,
    int *nlist,
    const float *coord,
    const int *type,
    const InputNlist & inlist,
    const int max_nbor_size,
    const int nloc,
    const int nall,
    const float rcut,
    const std::vector<int> sec);

template
void prod_force_nb_cpu<double>(
    double *force,
    const double *net_deriv,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

template
void prod_force_nb_cpu<float>(
    float *force,
    const float *net_deriv,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

template
void prod_virial_nb_cpu<double>(
    double *virial,
    double *atom_virial,
    const double *net_deriv,
    const double *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

template
void prod_virial_nb_cpu<float>(
    float *virial,
    float *atom_virial,
    const float *net_deriv,
    const float *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

}
