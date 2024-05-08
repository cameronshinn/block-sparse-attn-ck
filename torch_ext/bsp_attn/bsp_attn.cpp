#include <hip/hip_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>

// #include <sparse-mha/utils/hip_utils.hpp>
// #include <ck/ck.hpp>

#define CHECK_HIP(expr) do {                    \
    hipError_t result = (expr);                 \
    if (result != hipSuccess) {                 \
        fprintf(stderr, "%s:%d: %s (%d)\n",     \
            __FILE__, __LINE__,                 \
            hipGetErrorString(result), result); \
        exit(EXIT_FAILURE);                     \
    }                                           \
} while(0)

template <typename T>
__global__ void double_array_kernel(const T *inp, T *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T val = inp[tid];
    out[tid] = val * 2;
}

torch::Tensor double_array(torch::Tensor in) {
    torch::Tensor out = torch::empty({in.sizes()}, torch::TensorOptions().device(in.device()));

    AT_DISPATCH_ALL_TYPES(
        in.scalar_type(), "double_array_kernel", [&](){
            int num_blocks = (in.numel() + warpSize - 1) / warpSize;
            int block_size = warpSize;
            double_array_kernel<<<num_blocks, block_size>>>(
                in.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>()
            );
            CHECK_HIP(hipGetLastError());
        }
    );

    CHECK_HIP(hipDeviceSynchronize());

    return out;
}

// https://pytorch.org/tutorials/advanced/dispatcher.html
// TORCH_LIBRARY(bsp_attn_ext, m) {
//     m.def("double_array(Tensor inp) -> Tensor");
//     m.impl("double_array", c10::DispatchKey::CUDA, TORCH_FN(double_array));  // c10::DispatchKey::HIP doesn't work for some reason
// }

PYBIND11_MODULE(bsp_attn_ext, m) {
    m.def("double_array", &double_array, "double array");
}
