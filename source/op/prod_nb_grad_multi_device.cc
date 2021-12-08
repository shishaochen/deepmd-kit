#include "custom_op.h"
#include "prod_nb_grad.h"

REGISTER_OP("ProdForceNbGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("grad: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("grad_net: T");

REGISTER_OP("ProdVirialNbGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("grad: T")
    .Input("rij: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("grad_net: T");

template<typename Device, typename FPTYPE>
class ProdForceNbGradOp : public OpKernel {
public:
  explicit ProdForceNbGradOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
      deepmd::safe_compute(context, [this](OpKernelContext* context) {this->_Compute(context);});
  }

  void _Compute(OpKernelContext* context) {
    const Tensor& grad_tensor = context->input(0);
    const Tensor& nlist_tensor = context->input(1);
    const Tensor& natoms_tensor = context->input(2);
    TensorShape grad_shape = grad_tensor.shape();
    TensorShape nlist_shape = nlist_tensor.shape();
    OP_REQUIRES(context, (grad_shape.dims() == 2), errors::InvalidArgument ("Dim of grad should be 2"));
    OP_REQUIRES(context, (nlist_shape.dims() == 2), errors::InvalidArgument ("Dim of nlist should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1), errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));

    auto natoms = natoms_tensor.flat<int>();
    int nframes = nlist_shape.dim_size(0);
    int nloc = natoms(0);
    int nnei = nlist_shape.dim_size(1) / nloc;
    int ndescrpt = nnei * 3;
    OP_REQUIRES(context, (nframes == grad_shape.dim_size(0)),  errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES(context, (nframes == nlist_shape.dim_size(0)),  errors::InvalidArgument ("number of frames should match"));
    OP_REQUIRES(context, (nloc * 3 == grad_shape.dim_size(1)),  errors::InvalidArgument ("input grad shape should be 3 x natoms"));

    // Create an output tensor
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim(nframes);
    grad_net_shape.AddDim(nloc * ndescrpt);
    Tensor* grad_net_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape, &grad_net_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    assert(nframes == grad_net_shape.dim_size(0));
    assert(nframes == grad_shape.dim_size(0));
    assert(nframes == nlist_tensor.shape().dim_size(0));
    assert(nloc * ndescrpt == grad_net_shape.dim_size(1));
    assert(nloc * 3 == grad_shape.dim_size(1));
    assert(nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert(nnei * 3 == ndescrpt);

    // Flat the tensors
    FPTYPE *p_grad_net = grad_net_tensor->flat<FPTYPE>().data();
    const FPTYPE *p_grad = grad_tensor.flat<FPTYPE>().data();
    const int *p_nlist = nlist_tensor.flat<int>().data();
    for (int kk = 0; kk < nframes; ++kk){
      FPTYPE *grad_net = p_grad_net + kk * nloc * ndescrpt;
      const FPTYPE *grad = p_grad + kk * nloc * 3;
      const int *nlist = p_nlist + kk * nloc * nnei; 
      if (device == "GPU") {
    #if GOOGLE_CUDA
        deepmd::prod_force_grad_nb_gpu_cuda(grad_net, grad, nlist, nloc, nnei);
    #endif
      } else if (device == "CPU") {
        deepmd::prod_force_grad_nb_cpu(grad_net, grad, nlist, nloc, nnei);
      }
    }
  }

private:
  std::string device;
};


template<typename Device, typename FPTYPE>
class ProdVirialNbGradOp : public OpKernel {
public:
  explicit ProdVirialNbGradOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(context, [this](OpKernelContext* context) {this->_Compute(context);});
  }

  void _Compute(OpKernelContext* context) {
    const Tensor& grad_tensor = context->input(0);
    const Tensor& rij_tensor = context->input(1);
    const Tensor& nlist_tensor = context->input(2);
    const Tensor& natoms_tensor = context->input(3);

    TensorShape grad_shape = grad_tensor.shape();
    TensorShape rij_shape = rij_tensor.shape();
    TensorShape nlist_shape = nlist_tensor.shape();
    OP_REQUIRES(context, (grad_shape.dims() == 2), errors::InvalidArgument("Dim of grad should be 2"));
    OP_REQUIRES(context, (rij_shape.dims() == 2), errors::InvalidArgument("Dim of rij should be 2"));
    OP_REQUIRES(context, (nlist_shape.dims() == 2), errors::InvalidArgument("Dim of nlist should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1), errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3), errors::InvalidArgument("number of atoms should be larger than (or equal to) 3"));

    auto natoms = natoms_tensor.flat<int>();
    int nframes = nlist_shape.dim_size(0);
    int nloc = natoms(0);
    int nnei = nlist_tensor.shape().dim_size(1) / nloc;
    int ndescrpt = nnei * 3;
    OP_REQUIRES(context, (nframes == grad_shape.dim_size(0)), errors::InvalidArgument("number of frames should match"));
    OP_REQUIRES(context, (nframes == rij_shape.dim_size(0)), errors::InvalidArgument("number of frames should match"));
    OP_REQUIRES(context, (nframes == nlist_shape.dim_size(0)), errors::InvalidArgument("number of frames should match"));
    OP_REQUIRES(context, (9 == grad_shape.dim_size(1)), errors::InvalidArgument("input grad shape should be 3 x natoms"));
    OP_REQUIRES(context, (nloc * nnei * 3 == rij_shape.dim_size(1)), errors::InvalidArgument("dim of rij should be  nnei * 3"));

    // Create an output tensor
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim(nframes);
    grad_net_shape.AddDim(nloc * ndescrpt);
    Tensor* grad_net_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape, &grad_net_tensor));
    DeviceFunctor() (
        device,
        context->eigen_device<Device>()
    );
    assert(nframes == grad_net_shape.dim_size(0));
    assert(nframes == grad_shape.dim_size(0));
    assert(nframes == rij_tensor.shape().dim_size(0));
    assert(nframes == nlist_tensor.shape().dim_size(0));
    assert(nloc * ndescrpt == grad_net_shape.dim_size(1));
    assert(9 == grad_shape.dim_size(1));
    assert(nloc * nnei * 3 == rij_tensor.shape().dim_size(1));
    assert(nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert(nnei * 3 == ndescrpt);

    // Flat the tensors
    FPTYPE *p_grad_net = grad_net_tensor->flat<FPTYPE>().data();
    const FPTYPE *p_grad = grad_tensor.flat<FPTYPE>().data();
    const FPTYPE *p_rij = rij_tensor.flat<FPTYPE>().data();
    const int *p_nlist = nlist_tensor.flat<int>().data();
    for (int kk = 0; kk < nframes; ++kk){
      FPTYPE *grad_net = p_grad_net + kk * nloc * ndescrpt;
      const FPTYPE *grad = p_grad + kk * 9;
      const FPTYPE *rij = p_rij + kk * nloc * nnei * 3;
      const int *nlist = p_nlist + kk * nloc * nnei; 
      if (device == "GPU") {
      #if GOOGLE_CUDA
        deepmd::prod_virial_grad_nb_gpu_cuda(grad_net, grad, rij, nlist, nloc, nnei);
      #endif
      }
      else if (device == "CPU") {
        deepmd::prod_virial_grad_nb_cpu(grad_net, grad, rij, nlist, nloc, nnei);
      }
    }
  }

private:
  std::string device;
};

#define REGISTER_CPU(T)                                                                      \
REGISTER_KERNEL_BUILDER(                                                                     \
    Name("ProdForceNbGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    ProdForceNbGradOp<CPUDevice, T>);                                                        \
REGISTER_KERNEL_BUILDER(                                                                     \
    Name("ProdVirialNbGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),                      \
    ProdVirialNbGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);

#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                      \
REGISTER_KERNEL_BUILDER(                                                                     \
    Name("ProdForceNbGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"),  \
    ProdForceNbGradOp<GPUDevice, T>);                                                        \
REGISTER_KERNEL_BUILDER(                                                                     \
    Name("ProdVirialNbGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"), \
    ProdVirialNbGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA