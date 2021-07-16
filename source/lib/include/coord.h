#pragma once

#include "region.h"

namespace deepmd{

// normalize coords
// 对于 natom 个原子的坐标，对于周期性边界的轴向，调节为实际距离和镜像原子距离的较小者。
// 输入 & 输出：
// - coord 原子坐标。
// 输入：
// - natom 原子数量。
// - region 模拟区域。
template <typename FPTYPE>
void
normalize_coord_cpu(
    FPTYPE * coord,
    const int natom,
    const deepmd::Region<FPTYPE> & region);

// copy coordinates
// outputs:
//	out_c, out_t, mapping, nall
// inputs:
//	in_c, in_t, nloc, mem_nall, rc, region
//	mem_nall is the size of allocated memory for out_c, out_t, mapping
// returns
//	0: succssful
//	1: the memory is not large enough to hold all copied coords and types.
//	   i.e. nall > mem_nall
// 将模拟区域边界外、且在 rcut 距离内的原子镜像到对边和对角。
// 输出：
// - out_c 存储所有原子的 3 维坐标，镜像后的原子坐标补在 in_c 后面。
// - out_t 存储所有原子的类别，镜像后的原子坐标补在 in_t 后面。
// - mapping 存储所有原子的编号映射，被镜像的原子编号补在原集合的后面。
// - nall 所有原子（包括镜像原子）的数量。
// 输入：
// - in_c 存储所有原子的 3 维坐标。
// - in_t 存储所有原子的类别。
// - nloc 原子的数量。
// - mem_nall_ 为 out_c、out_t、mapping 三块内存 buffer 预留的容量。
// - rcut 截断距离。
// - region 模拟区域。
// 返回值：
// - 0 成功。
// - 1 容量 mem_nall_ 不够。
template <typename FPTYPE>
int
copy_coord_cpu(
    FPTYPE * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    const FPTYPE * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall,
    const float & rcut,
    const deepmd::Region<FPTYPE> & region);

// compute cell information
// output:
// cell_info: nat_stt,ncell,ext_stt,ext_end,ngcell,cell_shift,cell_iter,total_cellnum,loc_cellnum
// input:
// boxt
template <typename FPTYPE>
void
compute_cell_info(
    int * cell_info,
    const float & rcut,
    const deepmd::Region<FPTYPE> & region);

#if GOOGLE_CUDA
// normalize coords
// output:
// coord
// input:
// natom, box_info: boxt, rec_boxt
template <typename FPTYPE>
void
normalize_coord_gpu(
    FPTYPE * coord,
    const int natom,
    const deepmd::Region<FPTYPE> & region);

// copy coordinates
// outputs:
//	out_c, out_t, mapping, nall, 
//  int_data(temp cuda memory):idx_map,idx_map_noshift,temp_idx_order,loc_cellnum_map,total_cellnum_map,mask_cellnum_map,
//                             cell_map,cell_shift_map,sec_loc_cellnum_map,sec_total_cellnum_map,loc_clist
// inputs:
//	in_c, in_t, nloc, mem_nall, loc_cellnum, total_cellnum, cell_info, box_info
//	mem_nall is the size of allocated memory for out_c, out_t, mapping
// returns
//	0: succssful
//	1: the memory is not large enough to hold all copied coords and types.
//	   i.e. nall > mem_nall
template <typename FPTYPE>
int
copy_coord_gpu(
    FPTYPE * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    int * int_data,
    const FPTYPE * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall,
    const int & loc_cellnum,
    const int & total_cellnum,
    const int * cell_info,
    const deepmd::Region<FPTYPE> & region);
#endif // GOOGLE_CUDA


#if TENSORFLOW_USE_ROCM
// normalize coords
// output:
// coord
// input:
// natom, box_info: boxt, rec_boxt
template <typename FPTYPE>
void
normalize_coord_gpu_rocm(
    FPTYPE * coord,
    const int natom,
    const deepmd::Region<FPTYPE> & region);

// copy coordinates
// outputs:
//	out_c, out_t, mapping, nall, 
//  int_data(temp cuda memory):idx_map,idx_map_noshift,temp_idx_order,loc_cellnum_map,total_cellnum_map,mask_cellnum_map,
//                             cell_map,cell_shift_map,sec_loc_cellnum_map,sec_total_cellnum_map,loc_clist
// inputs:
//	in_c, in_t, nloc, mem_nall, loc_cellnum, total_cellnum, cell_info, box_info
//	mem_nall is the size of allocated memory for out_c, out_t, mapping
// returns
//	0: succssful
//	1: the memory is not large enough to hold all copied coords and types.
//	   i.e. nall > mem_nall
template <typename FPTYPE>
int
copy_coord_gpu_rocm(
    FPTYPE * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    int * int_data,
    const FPTYPE * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall,
    const int & loc_cellnum,
    const int & total_cellnum,
    const int * cell_info,
    const deepmd::Region<FPTYPE> & region);
#endif // TENSORFLOW_USE_ROCM

}
