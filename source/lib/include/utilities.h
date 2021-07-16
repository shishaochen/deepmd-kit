#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

namespace deepmd{

// 计算 n_sel 的累积和，存入 sec
// 例如 n_sel 是 [1,2]，则 sec 是 [0,1,3]
void cum_sum(
    std::vector<int> & sec, 
    const std::vector<int> & n_sel);

// 计算 r0 和 r1 两个 1 维向量的点积，并返回结果。
template <typename TYPE>
inline TYPE
dot1 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0];
}

// 计算 r0 和 r1 两个 2 维向量的点积，并返回结果。
template <typename TYPE>
inline TYPE
dot2 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0] + r0[1] * r1[1];
}

// 计算 r0 和 r1 两个 3 维向量的点积，并返回结果。
template <typename TYPE>
inline TYPE
dot3 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2];
}

// 计算 r0 和 r1 两个 4 维向量的点积，并返回结果。
template <typename TYPE>
inline TYPE
dot4 (const TYPE* r0, const TYPE* r1)
{
  return r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2] + r0[3] * r1[3];
}

// 计算 3*3 矩阵 tensor 和 3 维列向量 vec_i 的乘积，存储到 vec_o。
template <typename TYPE>
inline void 
dotmv3 (TYPE * vec_o, const TYPE * tensor, const TYPE * vec_i)
{
  vec_o[0] = dot3(tensor+0, vec_i);
  vec_o[1] = dot3(tensor+3, vec_i);
  vec_o[2] = dot3(tensor+6, vec_i);
}

// 计算 r0 和 r1 两个三维向量的叉积，存储到 r2。
template <typename TYPE>
inline void
cprod (const TYPE * r0,
       const TYPE * r1,
       TYPE* r2)
{
  r2[0] = r0[1] * r1[2] - r0[2] * r1[1];
  r2[1] = r0[2] * r1[0] - r0[0] * r1[2];
  r2[2] = r0[0] * r1[1] - r0[1] * r1[0];
}

// 求浮点数 x 的平方根倒数。
template <typename TYPE>
inline TYPE invsqrt (const TYPE x);

// 计算 x 平方根的倒数，并返回结果。
template <>
inline double
invsqrt<double> (const double x) 
{
  return 1./sqrt (x);
}

// 计算 x 平方根的倒数，并返回结果。
template <>
inline float
invsqrt<float> (const float x) 
{
  return 1./sqrtf (x);
}

}
