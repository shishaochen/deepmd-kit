#pragma once

#include <vector>
#include "device.h"
#include "neighbor_list.h"

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
    const std::vector<int> sec);

template<typename FPTYPE>
void prod_force_nb_cpu(
    FPTYPE *force,
    const FPTYPE * net_deriv,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

template<typename FPTYPE>
void prod_virial_nb_cpu(
    FPTYPE *virial,
    FPTYPE *atom_virial,
    const FPTYPE *net_deriv,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

#if GOOGLE_CUDA

template<typename FPTYPE>
void prod_nb_mat_gpu_cuda(
    FPTYPE *rij,
    int *nlist,
    const FPTYPE *coord,
    const int *type,
    const InputNlist &gpu_inlist,
    int *array_int,
    unsigned long long *array_longlong,
    const int max_nbor_size,
    const int nloc,
    const int nall,
    const float rcut,
    const std::vector<int> sec);

template<typename FPTYPE> 
void prod_force_nb_gpu_cuda(
    FPTYPE * force,
    const FPTYPE * net_deriv,
    const int * nlist,
    const int nloc,
    const int nall,
    const int nnei);

template<typename FPTYPE>
void prod_virial_nb_gpu_cuda(
    FPTYPE *virial,
    FPTYPE *atom_virial,
    const FPTYPE *net_deriv,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

#endif
}