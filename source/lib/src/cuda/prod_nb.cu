#include "device.h"
#include "fmt_nlist.h"
#include "prod_nb.h"

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void compute_nb_mat(
    FPTYPE* rij,
    const FPTYPE* coord,
    const int* type,
    const int* nlist,
    const int nnei,
    const float rcut) {   
  // <<<nloc, TPB>>>
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (tid >= nnei) {
    return;
  }
  const int *row_nlist = nlist + bid * nnei;
  FPTYPE *row_rij = rij + bid * nnei * 3;
  for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
    if (row_nlist[ii] >= 0) {
      FPTYPE rr[3]  = {0};
      const int j_idx = row_nlist[ii];
      for (int kk = 0; kk < 3; kk++) {
        rr[kk] = coord[j_idx * 3 + kk] - coord[bid * 3 + kk];
        row_rij[ii * 3 + kk] = rr[kk];
      }
    }
  }
}

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void force_deriv_wrt_center_atom(
    FPTYPE * force,
    const FPTYPE * net_deriv,
    const int nnei) {
  __shared__ FPTYPE data[THREADS_PER_BLOCK * 3];
  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  for (int ii = tid; ii < THREADS_PER_BLOCK * 3; ii += THREADS_PER_BLOCK) {
    data[ii] = 0.f;
  }
  const int ndescrpt = nnei * 3;
  for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
    for (int jj = 0; jj < 3; jj++) {
      data[jj * THREADS_PER_BLOCK + tid] += net_deriv[bid * ndescrpt + ii * 3 + jj];
    }
  }
  __syncthreads(); 
  // do reduction in shared memory
  for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
    if (tid < ii) {
      for (int jj = 0; jj < 3; jj++) {
        data[jj * THREADS_PER_BLOCK + tid] += data[jj * THREADS_PER_BLOCK + tid + ii];
      }
    }
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) {
    force[bid * 3 + 0] += data[THREADS_PER_BLOCK * 0];
    force[bid * 3 + 1] += data[THREADS_PER_BLOCK * 1];
    force[bid * 3 + 2] += data[THREADS_PER_BLOCK * 2];
  }
}

template<typename FPTYPE>
__global__ void force_deriv_wrt_neighbors(
    FPTYPE *force,
    const FPTYPE *net_deriv,
    const int *nlist,
    const int nloc,
    const int nnei) {
  // idy -> nnei
  const unsigned int idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
  const unsigned int idz = threadIdx.y;
  if (idy >= nnei) {
    return;
  }
  // deriv wrt neighbors
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  FPTYPE force_tmp = -1.0 * net_deriv[idx * nnei * 3 + idy * 3 + idz];
  atomicAdd(force + j_idx * 3 + idz, force_tmp);
}

template<typename FPTYPE>
__global__ void virial_deriv_wrt_neighbors(
    FPTYPE *atom_virial,
    const FPTYPE *net_deriv,
    const FPTYPE *rij,
    const int *nlist,
    const int nnei) {
  const unsigned int idx = blockIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
  const unsigned int idz = threadIdx.y;
  const int ndescrpt = nnei * 3;
  if (idy >= nnei) {
    return;
  }
  int j_idx = nlist[idx * nnei + idy];
  if (j_idx < 0) {
    return;
  }
  const int fid = idz / 3; // 力的分量
  const int rid = idz % 3; // 径的分量
  FPTYPE virial_tmp = -1.0 * net_deriv[idx * ndescrpt + idy * 3 + fid] * rij[idx * ndescrpt + idy * 3 + rid];
  atomicAdd(atom_virial + j_idx * 9 + idz, virial_tmp);
}

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void atom_virial_reduction(
    FPTYPE *virial, 
    const FPTYPE *atom_virial,
    const int nall) {
  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  __shared__ FPTYPE data[THREADS_PER_BLOCK];
  data[tid] = 0.f;
  for (int ii = tid; ii < nall; ii += THREADS_PER_BLOCK) {
    data[tid] += atom_virial[ii * 9 + bid];
  }
  __syncthreads(); 
  // do reduction in shared memory
  for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
    if (tid < ii) {
      data[tid] += data[tid + ii];
    }
    __syncthreads();
  }
  // write result for this block to global memory
  if (tid == 0) virial[bid] = data[0];
}

namespace deepmd {

template <typename FPTYPE>
void prod_nb_mat_gpu_cuda(
    FPTYPE * rij,
    int * nlist,
    const FPTYPE * coord,
    const int * type,
    const InputNlist & gpu_inlist,
    int * array_int,
    uint_64 * array_longlong,
    const int max_nbor_size,
    const int nloc,
    const int nall,
    const float rcut,
    const std::vector<int> sec) {
  const int nnei = sec.back();
  DPErrcheck(cudaMemset(rij, 0., sizeof(FPTYPE) * nloc * nnei * 3));

  format_nbor_list_gpu_cuda(
      nlist, coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, nloc, nall, rcut, sec);
  nborErrcheck(cudaGetLastError());
  nborErrcheck(cudaDeviceSynchronize());

  compute_nb_mat<FPTYPE, TPB> <<<nloc, TPB>>> (rij, coord, type, nlist, nnei, rcut);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template<typename FPTYPE>
void prod_force_nb_gpu_cuda(
    FPTYPE *force,
    const FPTYPE *net_deriv,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei) {
  DPErrcheck(cudaMemset(force, 0.0, sizeof(FPTYPE) * nall * 3));

  force_deriv_wrt_center_atom<FPTYPE, TPB> <<<nloc, TPB>>>(force, net_deriv, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());

  const int LEN = 64;
  const int nblock = (nnei + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(LEN, 3);
  force_deriv_wrt_neighbors<<<block_grid, thread_grid>>>(force, net_deriv, nlist, nloc, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template<typename FPTYPE>
void prod_virial_nb_gpu_cuda(
    FPTYPE *virial,
    FPTYPE *atom_virial,
    const FPTYPE *net_deriv,
    const FPTYPE *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei) {
  DPErrcheck(cudaMemset(virial, 0.0, sizeof(FPTYPE) * 9));
  DPErrcheck(cudaMemset(atom_virial, 0.0, sizeof(FPTYPE) * 9 * nall));
    
  const int LEN = 16;
  int nblock = (nnei + LEN - 1) / LEN;
  dim3 block_grid(nloc, nblock);
  dim3 thread_grid(LEN, 9);
  // compute virial of a frame
  virial_deriv_wrt_neighbors<<<block_grid, thread_grid>>>(
      atom_virial, net_deriv, rij, nlist, nnei);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
  // reduction atom_virial to virial
  atom_virial_reduction<FPTYPE, TPB> <<<9, TPB>>>(virial, atom_virial, nall);
  DPErrcheck(cudaGetLastError());
  DPErrcheck(cudaDeviceSynchronize());
}

template void prod_nb_mat_gpu_cuda<float>(
    float *rij,
    int *nlist,
    const float *coord,
    const int *type,
    const InputNlist &gpu_inlist,
    int *array_int,
    unsigned long long *array_longlong,
    const int max_nbor_size,
    const int nloc,
    const int nall,
    const float rcut,
    const std::vector<int> sec);

template void prod_nb_mat_gpu_cuda<double>(
    double *rij,
    int *nlist,
    const double *coord,
    const int *type,
    const InputNlist &gpu_inlist,
    int *array_int,
    unsigned long long *array_longlong,
    const int max_nbor_size,
    const int nloc,
    const int nall,
    const float rcut,
    const std::vector<int> sec);

template void prod_force_nb_gpu_cuda<float>(
    float *force,
    const float *net_deriv,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

template void prod_force_nb_gpu_cuda<double>(
    double *force,
    const double *net_deriv,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

template void prod_virial_nb_gpu_cuda<float>(
    float *virial,
    float *atom_virial,
    const float *net_deriv,
    const float *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

template void prod_virial_nb_gpu_cuda<double>(
    double *virial,
    double *atom_virial,
    const double *net_deriv,
    const double *rij,
    const int *nlist,
    const int nloc,
    const int nall,
    const int nnei);

}
