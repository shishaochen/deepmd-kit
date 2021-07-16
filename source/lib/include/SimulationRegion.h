#ifndef __SimulationRegion_h_wanghan__
#define __SimulationRegion_h_wanghan__

#define MOASPNDIM 3
#include "utilities.h"
#include <fstream>

  template<typename VALUETYPE>
  class SimulationRegion
  {
protected:
    // 空间坐标的维数 3。
    const static int SPACENDIM = MOASPNDIM;
public:
    // 从 3*3 数组 boxv_ 构建盒子。
    void reinitBox (const double * boxv);
    void affineTransform (const double * affine_map);
    void reinitOrigin (const double * orig);
    void reinitOrigin (const std::vector<double> & orig);
    // 把 3*3 数组 boxt 的数值复制到 boxt_bk。
    void backup  ();
    // 从 3*3 数组 boxt_bk 恢复盒子。
    void recover ();
public:
    SimulationRegion ();
    ~SimulationRegion ();
    double *		getBoxTensor	()		{return boxt;};
    const double *	getBoxTensor	() const	{return boxt;};
    double *		getRecBoxTensor ()		{return rec_boxt;}
    const double *	getRecBoxTensor () const	{return rec_boxt;}
    double *		getBoxOrigin	()		{return origin;}
    const double *	getBoxOrigin	() const	{return origin;}
    double		getVolume	() const	{return volume;}
public:
    // 对 boxt 数组内 3 个轴向量组成的平行六面体，依次求出垂直于 Y-Z、Z-X、X-Y 三个平面的高度，存储在 dd。
    void		toFaceDistance	(double * dd) const;
public:
    // 从 3 维世界坐标，转换为内部坐标（以 boxt 三个轴向量为基）。
    void phys2Inter (double * i_v, const VALUETYPE * p_v) const;
    // 从内部坐标，转换为 3 维世界坐标。
    void inter2Phys (VALUETYPE * p_v, const double * i_v) const;
public:
    bool		isPeriodic	(const int dim) const {return is_periodic[dim];}
    static int		compactIndex	(const int * idx) ;
    // 给定格点编号，返回在世界格点数组中偏移位置。
    double *		getShiftVec	(const int index = 0) ;
    // 给定格点编号，返回在世界格点数组中偏移位置。
    const double *	getShiftVec	(const int index = 0) const;
    // 给定 3 维世界格点坐标，计算全局（27 个点）唯一的编号 ID。
    // 格点坐标 X==idx[0] Y==idx[1] Z==idx[2] 三个轴的值域是 [-1,0,1]。
    int			getShiftIndex	(const int * idx) const;
    int			getNullShiftIndex() const;
    void		shiftCoord	(const int * idx,
					 VALUETYPE &x,
					 VALUETYPE &y,
					 VALUETYPE &z) const;
    static int		getNumbShiftVec ()	 {return shift_info_size;}
    static int		getShiftVecTotalSize ()  {return shift_vec_size;}
public:
    // 对于给定 2 个原子的坐标差，对于周期性边界的轴向，调节为实际距离和镜像原子距离的较小者。
    void 
    diffNearestNeighbor (const VALUETYPE * r0,
			 const VALUETYPE * r1,
			 VALUETYPE * phys) const;
    virtual void 
    diffNearestNeighbor (const VALUETYPE x0,
			 const VALUETYPE y0,
			 const VALUETYPE z0,
			 const VALUETYPE x1,
			 const VALUETYPE y1,
			 const VALUETYPE z1,
			 VALUETYPE & dx,
			 VALUETYPE & dy,
			 VALUETYPE & dz) const ;
    virtual void diffNearestNeighbor (const VALUETYPE x0,
				      const VALUETYPE y0,
				      const VALUETYPE z0,
				      const VALUETYPE x1,
				      const VALUETYPE y1,
				      const VALUETYPE z1,
				      VALUETYPE & dx,
				      VALUETYPE & dy,
				      VALUETYPE & dz,
				      int & shift_x,
				      int & shift_y,
				      int & shift_z) const ;
    virtual void diffNearestNeighbor (const VALUETYPE x0,
				      const VALUETYPE y0,
				      const VALUETYPE z0,
				      const VALUETYPE x1,
				      const VALUETYPE y1,
				      const VALUETYPE z1,
				      VALUETYPE & dx,
				      VALUETYPE & dy,
				      VALUETYPE & dz,
				      VALUETYPE & shift_x,
				      VALUETYPE & shift_y,
				      VALUETYPE & shift_z) const ;
private:
    // 计算由 boxt 内 3 组轴向量组成的平行六面体的体积（也是行列式），存储在 volume。
    // 其倒数存在 volumei。
    void computeVolume ();
    // 计算 boxt 作为 3*3 矩阵的逆矩阵，存储在 rec_boxt。
    void computeRecBox ();
    double		volume;
    double		volumei;
    double		boxt		[SPACENDIM*SPACENDIM];  // 用于内部坐标转世界坐标
    double		boxt_bk		[SPACENDIM*SPACENDIM];
    double		rec_boxt	[SPACENDIM*SPACENDIM];  // 用于世界坐标转内部坐标
    double		origin		[SPACENDIM];
    bool		is_periodic	[SPACENDIM];
    std::string		class_name;
    bool		enable_restart;
protected:
    void computeShiftVec ();
    const static int			DBOX_XX = 1;  // X 轴上的格点间距，1。
    const static int			DBOX_YY = 1;  // Y 轴上的格点间距，1。
    const static int			DBOX_ZZ = 1;  // Z 轴上的格点间距，1。
    const static int			NBOX_XX = DBOX_XX*2+1;  // X 轴上的格点数量，3。
    const static int			NBOX_YY = DBOX_YY*2+1;  // Y 轴上的格点数量，3。
    const static int			NBOX_ZZ = DBOX_ZZ*2+1;  // Z 轴上的格点数量，3。
    const static int			shift_info_size = NBOX_XX * NBOX_YY * NBOX_ZZ;  // 格点的数量，3*3*3=27。
    const static int			shift_vec_size = SPACENDIM * shift_info_size;  // 格点的坐标数量，27*3=81。
    double				shift_vec	[shift_vec_size];  // 按序存储 27 个格点的世界坐标。
    double				inter_shift_vec [shift_vec_size];  // 按序存储 27 个标准格点的内部坐标。
    // 给定格点坐标，计算全局唯一的编号 ID。
    // 格点坐标 X Y Z 三个轴的值域是 [-1,0,1]。
    static int index3to1 (const int tx, const int ty, const int tz) 
	{
	  return (NBOX_ZZ * (NBOX_YY * (tx+DBOX_XX) + ty+DBOX_YY)+ tz+DBOX_ZZ);
	}
    // 给定格点编号，返回在内部坐标的格点数组中偏移位置。
    double *		getInterShiftVec	(const int index = 0) ;
    // 给定格点编号，返回在世界坐标格点数组中偏移位置。
    const double *	getInterShiftVec	(const int index = 0) const;
private:
    // 复制 3 维向量 i_v 的值到 o_v。
    void copy	    (double * o_v, const double * i_v) const;
    // 3*3 方阵 i_t 点乘 3 维列向量 i_v。
    void naiveTensorDotVector (double * out,
			       const double * i_t,
			       const double * i_v) const;
    // 3 维行向量 i_v 点乘 3*3 方阵 i_t。
    void naiveTensorTransDotVector (double * out,
				    const double * i_t,
				    const double * i_v) const;
    // 3*3 方阵 i_t 点乘 3 维列向量 i_v。
    void tensorDotVector (double * out,
			  const double * i_t,
			  const double * i_v) const;
     // 3 维行向量 i_v 点乘 3*3 方阵 i_t。
    void tensorTransDotVector (double * out,
			       const double * i_t,
			       const double * i_v) const;
    void getFromRestart (double * my_boxv, double * my_orig, bool * period) const;
    void defaultInitBox (double * my_boxv, double * my_orig, bool * period) const;
    // 对于给定轴方向的距离分量 dd[dim]，如果是周期性边界条件，调节为实际距离和镜像原子距离的较小者。
    void apply_periodic (int dim, double * dd) const;
    void apply_periodic (int dim, double * dd, int & shift) const;
private:
    std::fstream fp;
  };

#ifdef MOASP_INLINE_IMPLEMENTATION
#include "SimulationRegion_Impl.h"
#endif

#endif


