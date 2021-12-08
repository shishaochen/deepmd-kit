#include "device.h"
#include "prod_nb_grad.h"

template<typename FPTYPE>
__global__ void force_grad_wrt_center_atom(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const int ndescrpt) {
  __shared__ FPTYPE grad_one[3];
  unsigned int center_idx = blockIdx.x;
  unsigned int tid = threadIdx.x;
  if (tid < 3) {
      grad_one[tid] = grad[center_idx * 3 + tid];
  }
  __syncthreads();
  unsigned int descrpt_idx = blockIdx.y * blockDim.x + tid;
  if (descrpt_idx < ndescrpt) {
    const int dd = descrpt_idx % 3;
    grad_net[center_idx * ndescrpt + descrpt_idx] += grad_one[dd];
  }
}

template<typename FPTYPE>
__global__ void force_grad_wrt_neighbors(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const int *nlist,
    const int nloc,
    const int nnei) {
    // idy -> nnei
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idw = threadIdx.y;
    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    grad_net[idx * nnei * 3 + idy * 3 + idw] -= grad[j_idx * 3 + idw];
}

template<typename FPTYPE>
__global__ void virial_grad_wrt_neighbors(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nnei) {
  // idy -> nnei
  const unsigned int tid = threadIdx.x;
  const unsigned int idx = blockIdx.x * blockDim.x + tid;
  const unsigned int idy = blockIdx.y;
  const unsigned int dd0 = threadIdx.y;
  const int ndescrpt = nnei * 3;
  __shared__ FPTYPE grad_one[9];
  if (dd0 == 0 && tid < 9) {
      grad_one[tid] = grad[tid];
  }
  __syncthreads(); 
  if (idx >= nloc) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  FPTYPE tmp = 0.;
  for (int dd1 = 0; dd1 < 3; ++dd1) {
    tmp += grad_one[dd0 * 3 + dd1] * rij[idx * ndescrpt + idy * 3 + dd1];
  }
  grad_net[idx * ndescrpt + idy * 3 + dd0] -= tmp;
}

namespace deepmd {

template<typename FPTYPE>
void prod_force_grad_nb_gpu_cuda(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const int *nlist,
    const int nloc,
    const int nnei) {
  const int ndescrpt = nnei * 3;
  DPErrcheck(cudaMemset(grad_net, 0.0, sizeof(FPTYPE) * nloc * ndescrpt));
  const int nblock = (ndescrpt + TPB - 1) / TPB;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(TPB, 1);
  force_grad_wrt_center_atom<<<block_grid, thread_grid>>>(grad_net, grad, ndescrpt);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());

  const int LEN = 128;
  const int nblock_ = (nloc + LEN -1) / LEN;
  dim3 block_grid_(nblock_, nnei);
  dim3 thread_grid_(LEN, 3);
  force_grad_wrt_neighbors<<<block_grid_, thread_grid_>>>(grad_net, grad, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template<typename FPTYPE>
void prod_virial_grad_nb_gpu_cuda(
    FPTYPE *grad_net,
    const FPTYPE *grad,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nnei) {
    const int ndescrpt = nnei * 3;
  DPErrcheck(cudaMemset(grad_net, 0.0, sizeof(FPTYPE) * nloc * ndescrpt));
  const int LEN = 128;
  const int nblock = (nloc + LEN -1) / LEN;
  dim3 block_grid(nblock, nnei);
  dim3 thread_grid(LEN, 3);
  virial_grad_wrt_neighbors<<<block_grid, thread_grid>>>(
      grad_net, grad, rij, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template
void prod_force_grad_nb_gpu_cuda<float>(
    float *grad_net,
    const float *grad,
    const int *nlist,
    const int nloc,
    const int nnei);

template
void prod_force_grad_nb_gpu_cuda<double>(
    double *grad_net,
    const double *grad,
    const int *nlist,
    const int nloc,
    const int nnei);

template
void prod_virial_grad_nb_gpu_cuda<float>(
    float *grad_net,
    const float *grad,
    const float *rij,
    const int *nlist,
    const int nloc,
    const int nnei);

template
void prod_virial_grad_nb_gpu_cuda<double>(
    double *grad_net,
    const double *grad,
    const double *rij,
    const int *nlist,
    const int nloc,
    const int nnei);

}