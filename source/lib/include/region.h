#pragma once

namespace deepmd{

// 模拟区域。
template<typename FPTYPE>
struct Region
{
  FPTYPE * boxt; // 用于内部坐标转世界坐标
  FPTYPE * rec_boxt; // 用于世界坐标转内部坐标
  Region();
  ~Region();
};

// 从 3*3 数组 boxt 构建模拟区域。
// 输出：
// - region 模拟区域，包括世界坐标基和内部坐标基。
// 输入：
// - boxt 存储 XYZ 三个轴向量。
template<typename FPTYPE>
void
init_region_cpu(
    Region<FPTYPE> & region,
    const FPTYPE * boxt);

template<typename FPTYPE>
FPTYPE
volume_cpu(
    const Region<FPTYPE> & region);

// 将世界坐标转为内部坐标。
// 输出：
// - ri 内部坐标。
// 输入：
// - region 模拟区域。
// - rp 世界坐标。
template<typename FPTYPE>
void
convert_to_inter_cpu(
    FPTYPE * ri, 
    const Region<FPTYPE> & region,
    const FPTYPE * rp);

// 将内部坐标转为世界坐标。
// 输出：
// - rp 内部坐标。
// 输入：
// - region 模拟区域。
// - ri 世界坐标。
template<typename FPTYPE>
void
convert_to_phys_cpu(
    FPTYPE * rp, 
    const Region<FPTYPE> & region,
    const FPTYPE * ri);

#if GOOGLE_CUDA
//only for unittest
template<typename FPTYPE>
void
convert_to_inter_gpu(
    FPTYPE * ri, 
    const Region<FPTYPE> & region,
    const FPTYPE * rp);

template<typename FPTYPE>
void
convert_to_phys_gpu(
    FPTYPE * rp, 
    const Region<FPTYPE> & region,
    const FPTYPE * ri);

template<typename FPTYPE>
void
volume_gpu(
    FPTYPE * volume, 
    const Region<FPTYPE> & region);
#endif // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
//only for unittest
template<typename FPTYPE>
void
convert_to_inter_gpu_rocm(
    FPTYPE * ri, 
    const Region<FPTYPE> & region,
    const FPTYPE * rp);

template<typename FPTYPE>
void
convert_to_phys_gpu_rocm(
    FPTYPE * rp, 
    const Region<FPTYPE> & region,
    const FPTYPE * ri);

template<typename FPTYPE>
void
volume_gpu_rocm(
    FPTYPE * volume, 
    const Region<FPTYPE> & region);
#endif // TENSORFLOW_USE_ROCM
}


