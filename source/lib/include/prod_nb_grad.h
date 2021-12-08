#pragma once

namespace deepmd {
  
template<typename FPTYPE>
void prod_force_grad_nb_cpu(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const int *nlist,
    const int nloc,
    const int nnei);

template<typename FPTYPE>
void prod_virial_grad_nb_cpu(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nnei);

#if GOOGLE_CUDA

template<typename FPTYPE>
void prod_force_grad_nb_gpu_cuda(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const int *nlist,
    const int nloc,
    const int nnei);

template<typename FPTYPE>
void prod_virial_grad_nb_gpu_cuda(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nnei);

#endif

}