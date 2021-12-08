#include "custom_op.h"
#include "utilities.h"
#include "coord.h"
#include "region.h"
#include "neighbor_list.h"
#include "prod_env_mat.h"
#include "prod_nb.h"
#include "errors.h"

REGISTER_OP("ProdNbMat")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("coord: T")
    .Input("type: int32")
    .Input("natoms: int32")
    .Input("box: T")
    .Input("mesh: int32")
    .Attr("rcut: float")
    .Attr("sel: list(int)")
    .Output("rij: T")
    .Output("nlist: int32");

REGISTER_OP("ProdForceNb")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("net_deriv: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("force: T");

REGISTER_OP("ProdVirialNb")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("net_deriv: T")
    .Input("rij: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("virial: T")
    .Output("atom_virial: T");

template<typename FPTYPE>
static int
_norm_copy_coord_cpu(
    std::vector<FPTYPE> & coord_cpy,
    std::vector<int> & type_cpy,
    std::vector<int> & mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r);

template<typename FPTYPE>
static int
_build_nlist_cpu(
    std::vector<int> &ilist, 
    std::vector<int> &numneigh,
    std::vector<int*> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r);

static void
_map_nlist_cpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei);

template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    OpKernelContext* context,
    FPTYPE const ** coord,
    std::vector<FPTYPE> & coord_cpy,
    int const** type,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    deepmd::InputNlist & inlist,
    std::vector<int> & ilist,
    std::vector<int> & numneigh,
    std::vector<int*> & firstneigh,
    std::vector<std::vector<int>> & jlist,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial);

#if GOOGLE_CUDA

template<typename FPTYPE>
static int
_norm_copy_coord_gpu(
    OpKernelContext* context,
    Tensor * tensor_list,
    FPTYPE * & coord_cpy,
    int * & type_cpy,
    int * & idx_mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r);

template<typename FPTYPE>
static int
_build_nlist_gpu(
    OpKernelContext* context,
    Tensor * tensor_list,
    int * &ilist, 
    int * &numneigh,
    int ** &firstneigh,
    int * &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r);

static void
_map_nlist_gpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei);

template <typename FPTYPE>
static void
_prepare_coord_nlist_gpu(
    OpKernelContext* context,
    Tensor * tensor_list,
    FPTYPE const ** coord,
    FPTYPE * & coord_cpy,
    int const** type,
    int * & type_cpy,
    int * & idx_mapping,
    deepmd::InputNlist & inlist,
    int * & ilist,
    int * & numneigh,
    int ** & firstneigh,
    int * & jlist,
    int * & nbor_list_dev,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int mesh_tensor_size,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial);
    
#endif //GOOGLE_CUDA

template<typename FPTYPE>
static int
_norm_copy_coord_cpu(
    std::vector<FPTYPE> & coord_cpy,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r)
{
  std::vector<FPTYPE> tmp_coord(nall*3);
  std::copy(coord, coord+nall*3, tmp_coord.begin());
  deepmd::Region<FPTYPE> region;
  init_region_cpu(region, box);
  normalize_coord_cpu(&tmp_coord[0], nall, region);
  int tt;
  for(tt = 0; tt < max_cpy_trial; ++tt){
    coord_cpy.resize(mem_cpy*3);
    type_cpy.resize(mem_cpy);
    idx_mapping.resize(mem_cpy);
    int ret = copy_coord_cpu(
	&coord_cpy[0], &type_cpy[0], &idx_mapping[0], &nall, 
	&tmp_coord[0], type, nloc, mem_cpy, rcut_r, region);
    if(ret == 0){
      break;
    }
    else{
      mem_cpy *= 2;
    }
  }
  return (tt != max_cpy_trial);
}

template<typename FPTYPE>
static int
_build_nlist_cpu(
    std::vector<int> &ilist, 
    std::vector<int> &numneigh,
    std::vector<int*> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r)
{
  int tt;
  for(tt = 0; tt < max_nnei_trial; ++tt){
    for(int ii = 0; ii < nloc; ++ii){
      jlist[ii].resize(mem_nnei);
      firstneigh[ii] = &jlist[ii][0];
    }
    deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
    int ret = build_nlist_cpu(
	inlist, &max_nnei, 
	coord, nloc, new_nall, mem_nnei, rcut_r);
    if(ret == 0){
      break;
    }
    else{
      mem_nnei *= 2;
    }
  }
  return (tt != max_nnei_trial);
}
    
static void
_map_nlist_cpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei)
{
  for (int ii = 0; ii < nloc; ++ii){
    for (int jj = 0; jj < nnei; ++jj){
      int record = nlist[ii*nnei+jj];
      if (record >= 0) {		
	nlist[ii*nnei+jj] = idx_mapping[record];	      
      }
    }
  }  
}

template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    OpKernelContext* context,
    FPTYPE const ** coord,
    std::vector<FPTYPE> & coord_cpy,
    int const** type,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    deepmd::InputNlist & inlist,
    std::vector<int> & ilist,
    std::vector<int> & numneigh,
    std::vector<int*> & firstneigh,
    std::vector<std::vector<int>> & jlist,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial)
{
  inlist.inum = nloc;
  if(nei_mode != 3){
    // build nlist by myself
    // normalize and copy coord
    if(nei_mode == 1){
      int copy_ok = _norm_copy_coord_cpu(
	  coord_cpy, type_cpy, idx_mapping, new_nall, mem_cpy,
	  *coord, box, *type, nloc, max_cpy_trial, rcut_r);
      OP_REQUIRES (context, copy_ok, errors::Aborted("cannot allocate mem for copied coords"));
      *coord = &coord_cpy[0];
      *type = &type_cpy[0];
    }
    // build nlist
    int build_ok = _build_nlist_cpu(
	ilist, numneigh, firstneigh, jlist, max_nbor_size, mem_nnei,
	*coord, nloc, new_nall, max_nnei_trial, rcut_r);
    OP_REQUIRES (context, build_ok, errors::Aborted("cannot allocate mem for nlist"));
    inlist.ilist = &ilist[0];
    inlist.numneigh = &numneigh[0];
    inlist.firstneigh = &firstneigh[0];
  }
  else{
    // copy pointers to nlist data
    memcpy(&inlist.ilist, 4 + mesh_tensor_data, sizeof(int *));
    memcpy(&inlist.numneigh, 8 + mesh_tensor_data, sizeof(int *));
    memcpy(&inlist.firstneigh, 12 + mesh_tensor_data, sizeof(int **));
    max_nbor_size = max_numneigh(inlist);
  }
}

#if GOOGLE_CUDA

template<typename FPTYPE>
static int
_norm_copy_coord_gpu(
    OpKernelContext* context,
    Tensor * tensor_list,
    FPTYPE * & coord_cpy,
    int * & type_cpy,
    int * & idx_mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r)
{
  // Tensor FPTYPE_temp;
  TensorShape FPTYPE_shape;
  FPTYPE_shape.AddDim(nall*3);
  context->allocate_temp(DataTypeToEnum<FPTYPE>::value, FPTYPE_shape, tensor_list);
  FPTYPE * tmp_coord = (*tensor_list).flat<FPTYPE>().data();
  DPErrcheck(cudaMemcpy(tmp_coord, coord, sizeof(FPTYPE) * nall * 3, cudaMemcpyDeviceToDevice));
  
  deepmd::Region<FPTYPE> region;
  init_region_cpu(region, box);
  FPTYPE box_info[18];
  std::copy(region.boxt, region.boxt+9, box_info);
  std::copy(region.rec_boxt, region.rec_boxt+9, box_info+9);
  int cell_info[23];
  deepmd::compute_cell_info(cell_info, rcut_r, region);
  const int loc_cellnum=cell_info[21];
  const int total_cellnum=cell_info[22];
  //Tensor double_temp;
  TensorShape double_shape;
  double_shape.AddDim(18);
  context->allocate_temp(DataTypeToEnum<FPTYPE>::value, double_shape, tensor_list+1);
  //Tensor int_temp;
  TensorShape int_shape;
  int_shape.AddDim(23+nloc*3+loc_cellnum+total_cellnum*3+total_cellnum*3+loc_cellnum+1+total_cellnum+1+nloc);
  context, context->allocate_temp(DT_INT32, int_shape, tensor_list+2);
  FPTYPE * box_info_dev = (*(tensor_list+1)).flat<FPTYPE>().data();
  int * cell_info_dev = (*(tensor_list+2)).flat<int>().data();
  int * int_data_dev = cell_info_dev + 23;
  deepmd::memcpy_host_to_device(box_info_dev, box_info, 18);
  deepmd::memcpy_host_to_device(cell_info_dev, cell_info, 23);
  deepmd::Region<FPTYPE> region_dev;
  FPTYPE * new_boxt = region_dev.boxt;
  FPTYPE * new_rec_boxt = region_dev.rec_boxt;
  region_dev.boxt = box_info_dev;
  region_dev.rec_boxt = box_info_dev + 9;
  deepmd::normalize_coord_gpu(tmp_coord, nall, region_dev);
  int tt;
  for(tt = 0; tt < max_cpy_trial; ++tt){
    //Tensor cpy_temp;
    TensorShape cpy_shape;
    cpy_shape.AddDim(mem_cpy*3);
    context->allocate_temp(DataTypeToEnum<FPTYPE>::value, cpy_shape, tensor_list+3);
    //Tensor t_temp;
    TensorShape t_shape;
    t_shape.AddDim(mem_cpy*2);
    context, context->allocate_temp(DT_INT32, t_shape, tensor_list+4);
    coord_cpy = (*(tensor_list+3)).flat<FPTYPE>().data();
    type_cpy = (*(tensor_list+4)).flat<int>().data();
    idx_mapping = type_cpy + mem_cpy;
    int ret = deepmd::copy_coord_gpu(
        coord_cpy, type_cpy, idx_mapping, &nall, int_data_dev,
        tmp_coord, type, nloc, mem_cpy, loc_cellnum, total_cellnum, cell_info_dev, region_dev);
    if(ret == 0){
      break;
    }
    else{
      mem_cpy *= 2;
    }
  }
  region_dev.boxt = new_boxt;
  region_dev.rec_boxt = new_rec_boxt;
  return (tt != max_cpy_trial);
}

template<typename FPTYPE>
static int
_build_nlist_gpu(
    OpKernelContext* context,
    Tensor * tensor_list,
    int * &ilist, 
    int * &numneigh,
    int ** &firstneigh,
    int * &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r)
{
  //Tensor nlist_temp;
  TensorShape nlist_shape;
  nlist_shape.AddDim(nloc*2);
  context->allocate_temp(DT_INT32, nlist_shape, tensor_list);
  ilist = (*tensor_list).flat<int>().data();
  numneigh = ilist + nloc;
  //Tensor jlist_temp;
  int * ind_data = NULL;
  
  std::vector<int*> firstneigh_host(nloc);
  int tt;
  for(tt = 0; tt < max_nnei_trial; ++tt){
    TensorShape jlist_shape;
    jlist_shape.AddDim(3*nloc*mem_nnei);
    context->allocate_temp(DT_INT32, jlist_shape, tensor_list+1);
    jlist = (*(tensor_list+1)).flat<int>().data();
    ind_data = jlist + nloc * mem_nnei;
    for(int ii = 0; ii < nloc; ++ii){
      firstneigh_host[ii] = jlist + ii * mem_nnei;
    }
    deepmd::memcpy_host_to_device(firstneigh, firstneigh_host);
    deepmd::InputNlist inlist(nloc, ilist, numneigh, firstneigh);
    int ret = deepmd::build_nlist_gpu(
        inlist, &max_nnei, ind_data, 
        coord, nloc, new_nall, mem_nnei, rcut_r);
    if(ret == 0){
      break;
    }
    else{
      mem_nnei *= 2;
    }
  }
  return (tt != max_nnei_trial);
}

static void
_map_nlist_gpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei)
{
  deepmd::use_nlist_map(nlist, idx_mapping, nloc, nnei);
}

template <typename FPTYPE>
static void
_prepare_coord_nlist_gpu(
    OpKernelContext* context,
    Tensor * tensor_list,
    FPTYPE const ** coord,
    FPTYPE * & coord_cpy,
    int const** type,
    int * & type_cpy,
    int * & idx_mapping,
    deepmd::InputNlist & inlist,
    int * & ilist,
    int * & numneigh,
    int ** & firstneigh,
    int * & jlist,
    int * & nbor_list_dev,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int mesh_tensor_size,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial)
{    
  if(nei_mode != 3){
    inlist.inum = nloc;
    // build nlist by myself
    // normalize and copy coord
    if(nei_mode == 1){
      int copy_ok = _norm_copy_coord_gpu(
        context, tensor_list, coord_cpy, type_cpy, idx_mapping, new_nall, mem_cpy,
        *coord, box, *type, nloc, max_cpy_trial, rcut_r);
      OP_REQUIRES (context, copy_ok, errors::Aborted("cannot allocate mem for copied coords"));
      *coord = coord_cpy;
      *type = type_cpy;
    }
    //build nlist
    int build_ok = _build_nlist_gpu(
      context, tensor_list + 5, ilist, numneigh, firstneigh, jlist, max_nbor_size, mem_nnei,
      *coord, nloc, new_nall, max_nnei_trial, rcut_r);
    OP_REQUIRES (context, build_ok, errors::Aborted("cannot allocate mem for nlist"));
    if (max_nbor_size <= 1024) {
      max_nbor_size = 1024;
    }
    else if (max_nbor_size <= 2048) {
      max_nbor_size = 2048;
    }
    else {
      max_nbor_size = 4096;
    }
    inlist.ilist = ilist;
    inlist.numneigh = numneigh;
    inlist.firstneigh = firstneigh;
  }
  else{
    // update nbor list
    deepmd::InputNlist inlist_temp;
    inlist_temp.inum = nloc;
    deepmd::env_mat_nbor_update(
        inlist_temp, inlist, max_nbor_size, nbor_list_dev,
        mesh_tensor_data, mesh_tensor_size);
    OP_REQUIRES (context, (max_numneigh(inlist_temp) <= GPU_MAX_NBOR_SIZE), errors::InvalidArgument ("Assert failed, max neighbor size of atom(lammps) " + std::to_string(max_numneigh(inlist_temp)) + " is larger than " + std::to_string(GPU_MAX_NBOR_SIZE) + ", which currently is not supported by deepmd-kit."));
  }
}

#endif  // GOOGLE_CUDA

template <typename Device, typename FPTYPE>
class ProdNbMatOp : public OpKernel {
public:
  explicit ProdNbMatOp(OpKernelConstruction* context) : OpKernel(context) {
    float nloc_f, nall_f;
    OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
    OP_REQUIRES_OK(context, context->GetAttr("sel", &sel));
    deepmd::cum_sum(sec, sel);
    nnei = sec.back();
    max_nbor_size = 1024;
    max_cpy_trial = 100;
    mem_cpy = 256;
    max_nnei_trial = 100;
    mem_nnei = 256;
  }

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(context, [this](OpKernelContext* context) {
      this->_Compute(context);
    });
  }

  void _Compute(OpKernelContext* context) {
    const Tensor& coord_tensor= context->input(0);
    const Tensor& type_tensor = context->input(1);
    const Tensor& natoms_tensor	= context->input(2);
    const Tensor& box_tensor = context->input(3);
    const Tensor& mesh_tensor = context->input(4);
    OP_REQUIRES(context, (coord_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of coord should be 2"));
    OP_REQUIRES(context, (type_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of type should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1), errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (box_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of box should be 2"));
    OP_REQUIRES(context, (mesh_tensor.shape().dims() == 1), errors::InvalidArgument("Dim of mesh should be 1"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument("number of atoms should be larger than (or equal to) 3"));
    DeviceFunctor()(
        device,
        context->eigen_device<Device>()
    );
    const int *natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int ntypes = natoms_tensor.shape().dim_size(0) - 2; //nloc and nall mean something.
    int nsamples = coord_tensor.shape().dim_size(0);

    // check the sizes
    OP_REQUIRES(context, (nsamples == type_tensor.shape().dim_size(0)), errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nsamples == box_tensor.shape().dim_size(0)), errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nall*3 == coord_tensor.shape().dim_size(1)), errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (nall == type_tensor.shape().dim_size(1)), errors::InvalidArgument("number of atoms should match"));
    OP_REQUIRES(context, (9 == box_tensor.shape().dim_size(1)), errors::InvalidArgument("number of box should be 9"));
    OP_REQUIRES(context, (ntypes == int(sel.size())), errors::InvalidArgument("number of types should match the length of sel array"));

    int nei_mode = 0;
    bool b_nlist_map = false;
    if (mesh_tensor.shape().dim_size(0) == 16) {
      // lammps neighbor list
      nei_mode = 3;
    }
    else if (mesh_tensor.shape().dim_size(0) == 6) {
      // manual copied pbc
      assert (nloc == nall);
      nei_mode = 1;
      b_nlist_map = true;
    }
    else if (mesh_tensor.shape().dim_size(0) == 0) {
      // no pbc
      assert (nloc == nall);
      nei_mode = -1;
    }
    else {
      throw deepmd::deepmd_exception("invalid mesh tensor");
    }

    // Create output tensors
    TensorShape rij_shape;
    rij_shape.AddDim(nsamples);
    rij_shape.AddDim(nloc * nnei * 3);
    TensorShape nlist_shape;
    nlist_shape.AddDim(nsamples);
    nlist_shape.AddDim(nloc * nnei);
    int context_output_index = 0;
    Tensor *rij_tensor = NULL;
    Tensor *nlist_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        0,
        rij_shape,
        &rij_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(
        1,
        nlist_shape,
        &nlist_tensor));

    FPTYPE *p_rij = rij_tensor->flat<FPTYPE>().data();
    int *p_nlist = nlist_tensor->flat<int>().data();
    const FPTYPE *p_coord = coord_tensor.flat<FPTYPE>().data();
    const FPTYPE *p_box = box_tensor.flat<FPTYPE>().data();
    const int *p_type = type_tensor.flat<int>().data();

    // loop over samples
    for (int ff = 0; ff < nsamples; ++ff) {
      FPTYPE *rij = p_rij + ff*nloc*nnei*3;
      int *nlist = p_nlist + ff*nloc*nnei;
      const FPTYPE *coord = p_coord + ff*nall*3;
      const FPTYPE *box = p_box + ff*9;
      const int *type = p_type + ff*nall;

    if (device == "GPU") {
    #if GOOGLE_CUDA
      int *idx_mapping = NULL;
      int *ilist = NULL, *numneigh = NULL;
      int **firstneigh = NULL;
      deepmd::malloc_device_memory(firstneigh, nloc);
      int *jlist = NULL;
      FPTYPE *coord_cpy;
      int *type_cpy;
      int frame_nall = nall;
      int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
      std::vector<Tensor> tensor_list(7);
      // prepare coord and nlist
      _prepare_coord_nlist_gpu<FPTYPE>(
          context, &tensor_list[0], &coord, coord_cpy, &type, type_cpy, idx_mapping, 
          gpu_inlist, ilist, numneigh, firstneigh, jlist, nbor_list_dev,
          frame_nall, mem_cpy, mem_nnei, max_nbor_size,
          box, mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc, nei_mode, rcut, max_cpy_trial, max_nnei_trial);

      // allocate temp memory, temp memory must not be used after this operation!
      Tensor int_temp;
      TensorShape int_shape;
      int_shape.AddDim(sec.size() + nloc * sec.size() + nloc);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, int_shape, &int_temp));
      Tensor uint64_temp;
      TensorShape uint64_shape;
      uint64_shape.AddDim(nloc * GPU_MAX_NBOR_SIZE * 2);
      OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape, &uint64_temp));
      array_int = int_temp.flat<int>().data(); 
      array_longlong = uint64_temp.flat<unsigned long long>().data();

      // launch the gpu(nv) compute function
      deepmd::prod_nb_mat_gpu_cuda(
          rij, nlist, coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, nloc, frame_nall, rcut, sec);
      if(b_nlist_map) _map_nlist_gpu(nlist, idx_mapping, nloc, nnei);
      deepmd::delete_device_memory(firstneigh);
    #endif //GOOGLE_CUDA
    } else if (device == "CPU") {
      deepmd::InputNlist inlist;
      // some buffers, be freed after the evaluation of this frame
      std::vector<int> idx_mapping;
      std::vector<int> ilist(nloc), numneigh(nloc);
      std::vector<int*> firstneigh(nloc);
      std::vector<std::vector<int>> jlist(nloc);
      std::vector<FPTYPE> coord_cpy;
      std::vector<int> type_cpy;
      int frame_nall = nall;
      // prepare coord and nlist
      _prepare_coord_nlist_cpu<FPTYPE>(
	  context, &coord, coord_cpy, &type, type_cpy, idx_mapping, 
	  inlist, ilist, numneigh, firstneigh, jlist,
	  frame_nall, mem_cpy, mem_nnei, max_nbor_size,
	  box, mesh_tensor.flat<int>().data(), nloc, nei_mode, rcut, max_cpy_trial, max_nnei_trial);
      // launch the cpu compute function
      deepmd::prod_nb_mat_cpu(rij, nlist, coord, type, inlist, max_nbor_size, nloc, frame_nall, rcut, sec);
      // do nlist mapping if coords were copied
      if(b_nlist_map) _map_nlist_cpu(nlist, &idx_mapping[0], nloc, nnei);
    }
    }
  }

/////////////////////////////////////////////////////////////////////////////////////////////
private:
  float rcut;
  std::vector<int32> sel;
  std::vector<int> sec;
  int nnei, nloc, nall, max_nbor_size;
  int mem_cpy, max_cpy_trial;
  int mem_nnei, max_nnei_trial;
  std::string device;  
#if GOOGLE_CUDA
  int *array_int = NULL;
  unsigned long long *array_longlong = NULL;
  deepmd::InputNlist gpu_inlist;
  int *nbor_list_dev = NULL;
#endif
};

template<typename Device, typename FPTYPE>
class ProdForceNbOp : public OpKernel {
public:
  explicit ProdForceNbOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    deepmd::safe_compute(context, [this](OpKernelContext* context) {
      this->_Compute(context);
    });
  }

  void _Compute(OpKernelContext* context) {
    const Tensor& net_deriv_tensor = context->input(0);
    const Tensor& nlist_tensor = context->input(1);
    const Tensor& natoms_tensor = context->input(2);
    OP_REQUIRES(context, (net_deriv_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of net deriv should be 2"));
    OP_REQUIRES(context, (nlist_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of nlist should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1), errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument("number of atoms should be larger than (or equal to) 3"));

    const int *natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nframes = net_deriv_tensor.shape().dim_size(0);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;
    OP_REQUIRES(context, (nframes == nlist_tensor.shape().dim_size(0)), errors::InvalidArgument("number of samples should match"));

    // Create an output tensor
    TensorShape force_shape;
    force_shape.AddDim(nframes);
    force_shape.AddDim(3 * nall);
    Tensor* force_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, force_shape, &force_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    assert (nframes == force_shape.dim_size(0));
    assert (nframes == net_deriv_tensor.shape().dim_size(0));
    assert (nframes == nlist_tensor.shape().dim_size(0));
    assert (nall * 3 == force_shape.dim_size(1));
    assert (nloc * ndescrpt == net_deriv_tensor.shape().dim_size(1));
    assert (nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert (nnei * 3 == ndescrpt);

    // flat the tensors
    FPTYPE *p_force = force_tensor->flat<FPTYPE>().data();
    const FPTYPE *p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const int *p_nlist = nlist_tensor.flat<int>().data();

    for(int kk = 0; kk < nframes; ++kk){
      FPTYPE *force = p_force + kk * nall * 3;
      const FPTYPE *net_deriv = p_net_deriv + kk * nloc * ndescrpt;
      const int *nlist = p_nlist + kk * nloc * nnei;      
    if (device == "GPU") {
    #if GOOGLE_CUDA
      deepmd::prod_force_nb_gpu_cuda(force, net_deriv, nlist, nloc, nall, nnei);
    #endif // GOOGLE_CUDA
    }
    else if (device == "CPU") {
      deepmd::prod_force_nb_cpu(force, net_deriv, nlist, nloc, nall, nnei);
    }
    }
  }

 private:
  std::string device;
};

template<typename Device, typename FPTYPE>
class ProdVirialNbOp : public OpKernel {
 public:
  explicit ProdVirialNbOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(context, [this](OpKernelContext* context) {
      this->_Compute(context);
    });
  }

  void _Compute(OpKernelContext* context) {
    const Tensor& net_deriv_tensor = context->input(0);
    const Tensor& rij_tensor = context->input(1);
    const Tensor& nlist_tensor = context->input(2);
    const Tensor& natoms_tensor = context->input(3);
    OP_REQUIRES(context, (net_deriv_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of net deriv should be 2"));
    OP_REQUIRES(context, (rij_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of rij should be 2"));
    OP_REQUIRES(context, (nlist_tensor.shape().dims() == 2), errors::InvalidArgument("Dim of nlist should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1), errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument("number of atoms should be larger than (or equal to) 3"));

    const int *natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;
    int nframes = net_deriv_tensor.shape().dim_size(0);
    int ndescrpt = net_deriv_tensor.shape().dim_size(1) / nloc;
    OP_REQUIRES(context, (nframes == rij_tensor.shape().dim_size(0)), errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nframes == nlist_tensor.shape().dim_size(0)), errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nloc * nnei * 3 == rij_tensor.shape().dim_size(1)), errors::InvalidArgument("dim of rij should be nnei * 3"));

    // Create an output tensor
    TensorShape virial_shape;
    virial_shape.AddDim(nframes);
    virial_shape.AddDim(9);
    TensorShape atom_virial_shape;
    atom_virial_shape.AddDim(nframes);
    atom_virial_shape.AddDim(9 * nall);
    Tensor* virial_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, virial_shape, &virial_tensor));
    Tensor* atom_virial_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, atom_virial_shape, &atom_virial_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    // flat the tensors
    FPTYPE *p_virial = virial_tensor->flat<FPTYPE>().data();
    FPTYPE *p_atom_virial = atom_virial_tensor->flat<FPTYPE>().data();
    const FPTYPE *p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE *p_rij = rij_tensor.flat<FPTYPE>().data();
    const int *p_nlist = nlist_tensor.flat<int>().data();
    
    for(int kk = 0; kk < nframes; ++kk){
      FPTYPE *virial = p_virial + kk * 9;
      FPTYPE *atom_virial = p_atom_virial + kk * nall * 9;
      const FPTYPE *net_deriv = p_net_deriv + kk * nloc * ndescrpt;
      const FPTYPE *rij = p_rij + kk * nloc * nnei * 3;
      const int *nlist = p_nlist + kk * nloc * nnei;      
      if (device == "GPU") {
      #if GOOGLE_CUDA
        deepmd::prod_virial_nb_gpu_cuda(
            virial, atom_virial, net_deriv, rij, nlist, nloc, nall, nnei);
      #endif // GOOGLE_CUDA
      }
      else if (device == "CPU") {
        deepmd::prod_virial_nb_cpu(
            virial, atom_virial, net_deriv, rij, nlist, nloc, nall, nnei);
      }
    }
  }
 private:
  std::string device;
};

// Register the CPU kernels
#define REGISTER_CPU(T)                                                                                   \
REGISTER_KERNEL_BUILDER(                                                                                  \
    Name("ProdNbMat").Device(DEVICE_CPU).TypeConstraint<T>("T"),                                          \
    ProdNbMatOp<CPUDevice, T>);                                                                           \
REGISTER_KERNEL_BUILDER(                                                                                  \
    Name("ProdForceNb").Device(DEVICE_CPU).TypeConstraint<T>("T"),                                        \
    ProdForceNbOp<CPUDevice, T>);                                                                         \
REGISTER_KERNEL_BUILDER(                                                                                  \
    Name("ProdVirialNb").Device(DEVICE_CPU).TypeConstraint<T>("T"),                                       \
    ProdVirialNbOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels
#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                                   \
REGISTER_KERNEL_BUILDER(                                                                                  \
    Name("ProdNbMat").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms").HostMemory("box"),   \
    ProdNbMatOp<GPUDevice, T>);                                                                           \
REGISTER_KERNEL_BUILDER(                                                                                  \
    Name("ProdForceNb").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"),                   \
    ProdForceNbOp<GPUDevice, T>);                                                                         \
REGISTER_KERNEL_BUILDER(                                                                                  \
    Name("ProdVirialNb").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"),                  \
    ProdVirialNbOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
