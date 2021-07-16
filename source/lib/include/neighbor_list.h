#pragma once

#include <algorithm>
#include <iterator>
#include <cassert>
#include <vector>

#include "region.h"
#include "utilities.h"
#include "SimulationRegion.h"

namespace deepmd{

// format of the input neighbor list
struct InputNlist
{
  int inum;  // 原子数量
  int * ilist; // 所有原子的编号
  int * numneigh;  // 每个原子的邻居数量
  int ** firstneigh;  // 每个原子的邻居列表
  InputNlist () 
      : inum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL)
      {};
  InputNlist (
      int inum_, 
      int * ilist_,
      int * numneigh_, 
      int ** firstneigh_
      ) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {};
  ~InputNlist(){};
};

void convert_nlist(
    InputNlist & to_nlist,
    std::vector<std::vector<int> > & from_nlist
    );

int max_numneigh(
    const InputNlist & to_nlist
    );

// build neighbor list.
// outputs
//	nlist, max_list_size
//	max_list_size is the maximal size of jlist.
// inputs
//	c_cpy, nloc, nall, mem_size, rcut, region
//	mem_size is the size of allocated memory for jlist.
// returns
//	0: succssful
//	1: the memory is not large enough to hold all neighbors.
//	   i.e. max_list_size > mem_nall
template <typename FPTYPE>
int
build_nlist_cpu(
    InputNlist & nlist,
    int * max_list_size,
    const FPTYPE * c_cpy,
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
void convert_nlist_gpu_device(
    InputNlist & gpu_nlist,
    InputNlist & cpu_nlist,
    int* & gpu_memory,
    const int & max_nbor_size);

void free_nlist_gpu_device(
    InputNlist & gpu_nlist);

void use_nlist_map(
    int * nlist, 
    const int * nlist_map, 
    const int nloc, 
    const int nnei);

#endif //GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
// build neighbor list.
// outputs
//	nlist, max_list_size
//	max_list_size is the maximal size of jlist.
// inputs
//	c_cpy, nloc, nall, mem_size, rcut, region
//	mem_size is the size of allocated memory for jlist.
// returns
//	0: succssful
//	1: the memory is not large enough to hold all neighbors.
//	   i.e. max_list_size > mem_nall
template <typename FPTYPE>
int
build_nlist_gpu(
    InputNlist & nlist,
    int * max_list_size,
    int * nlist_data,
    const FPTYPE * c_cpy, 
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);

#endif // GOOGLE_CUDA


#if TENSORFLOW_USE_ROCM
// build neighbor list.
// outputs
//	nlist, max_list_size
//	max_list_size is the maximal size of jlist.
// inputs
//	c_cpy, nloc, nall, mem_size, rcut, region
//	mem_size is the size of allocated memory for jlist.
// returns
//	0: succssful
//	1: the memory is not large enough to hold all neighbors.
//	   i.e. max_list_size > mem_nall
template <typename FPTYPE>
int
build_nlist_gpu_rocm(
    InputNlist & nlist,
    int * max_list_size,
    int * nlist_data,
    const FPTYPE * c_cpy, 
    const int & nloc, 
    const int & nall, 
    const int & mem_size,
    const float & rcut);
	
#endif // TENSORFLOW_USE_ROCM

} // namespace deepmd


////////////////////////////////////////////////////////
// legacy code
////////////////////////////////////////////////////////

// build nlist by an extended grid
// 构建原子邻居列表，截断半径按 rc0 和 rc1 各 1 个，分别存储在 nlist0 和 nlist1 中。
// 输出：
// - nlist0 按截断半径 rc0 得到的邻居列表。
// - nlist1 按截断半径 rc1 得到的邻居列表，不包含 nlist0 的内容。
// 输入：
// - coord 中是所有原子铺开的 3 维坐标。
// - nloc 目标原子数量。
// - rc02 较小的截断半径。
// - rc12 较大的截断半径。
// - nat_stt 模拟区域的起始格点编号。
// - nat_end 模拟区域的终止格点编号。
// - ext_stt 扩展区域（包含边界外的 Cell）的起始格点编号。
// - ext_end 扩展区域的终止格点编号。
// - region 是模拟区域的工具类。
// - global_grid 是模拟区域在 X Y Z 三个维度上的 Cell 数量。
void
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const int &			nloc,
	     const double &			rc0,
	     const double &			rc1,
	     const std::vector<int > &		nat_stt_,
	     const std::vector<int > &		nat_end_,
	     const std::vector<int > &		ext_stt_,
	     const std::vector<int > &		ext_end_,
	     const SimulationRegion<double> &	region,
	     const std::vector<int > &		global_grid);

// build nlist by a grid for a periodic region
void
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const double &			rc0,
	     const double &			rc1,
	     const std::vector<int > &		grid,
	     const SimulationRegion<double> &	region);

// build nlist by a grid for a periodic region, atoms selected by sel0 and sel1
void
build_nlist (std::vector<std::vector<int > > &	nlist0,
	     std::vector<std::vector<int > > &	nlist1,
	     const std::vector<double > &	coord,
	     const std::vector<int> &		sel0,
	     const std::vector<int> &		sel1,
	     const double &			rc0,
	     const double &			rc1,
	     const std::vector<int > &		grid,
	     const SimulationRegion<double> &	region);

// brute force (all-to-all distance computation) neighbor list building
// if region is NULL, open boundary is assumed,
// otherwise, periodic boundary condition is defined by region
// 构建原子邻居列表，截断半径按 rc0 和 rc1 各 1 个，分别存储在 nlist0 和 nlist1 中。
// 输出：
// - nlist0 按截断半径 rc0 得到的邻居列表。
// - nlist1 按截断半径 rc1 得到的邻居列表，不包含 nlist0 的内容。
// 输入：
// - coord 中是所有原子铺开的 3 维坐标。
// - rc02 较小的截断半径。
// - rc12 较大的截断半径。
// - region 是模拟区域的工具类。
// - global_grid 是模拟区域在 X Y Z 三个维度上的 Cell 数量。
void
build_nlist (std::vector<std::vector<int > > & nlist0,
	     std::vector<std::vector<int > > & nlist1,
	     const std::vector<double > &	coord,
	     const double &			rc0_,
	     const double &			rc1_,
	     const SimulationRegion<double > * region = NULL);

// copy periodic images for the system
// 将模拟区域边界外的原子镜像到对边和对角，并据此构建 Cell 列表。
// 输出：
// - out_c 存储所有原子的 3 维坐标，镜像后的原子坐标补在 in_c 后面。
// - out_t 存储所有原子的类别，镜像后的原子坐标补在 in_t 后面。
// - mapping 存储所有原子的编号映射，被镜像的原子编号补在原集合的后面。
// - ncell 模拟区域内的 Cell 数量。
// - ngcell 模拟区域外的 Cell 数量，里面存储的是 Ghost 原子。
// 输入：
// - in_c 存储所有原子的 3 维坐标。
// - in_t 存储所有原子的类别。
// - rc 统计每个原子邻居的截断距离，影响 ncell 和 ngcell 的值。
// - region 模拟区域。
void 
copy_coord (std::vector<double > &		out_c, 
	    std::vector<int > &			out_t, 
	    std::vector<int > &			mapping,
	    std::vector<int> &			ncell,
	    std::vector<int> &			ngcell,
	    const std::vector<double > &	in_c,
	    const std::vector<int > &		in_t,
	    const double &			rc,
	    const SimulationRegion<double > &	region);
