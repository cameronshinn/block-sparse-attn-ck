#include <cstring>
#include <ostream>
#include <random>

#include <ck/utility/common_header.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <ck/tensor_description/cluster_descriptor.hpp>
// #include <ck/tensor/tensor_view.hpp>
#include <ck_tile/core/tensor/tensor_view.hpp>
#include <ck/host_utility/device_prop.hpp>
#include <ck/host_utility/kernel_launch.hpp>

#include <ck/library/utility/check_err.hpp>
#include <ck/library/utility/device_memory.hpp>
#include <ck/library/utility/fill.hpp>
#include <ck/library/utility/host_tensor.hpp>
#include <ck/library/utility/host_tensor_generator.hpp>

#include <ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp>
#include <ck_tile/ops/fmha/pipeline/sparse_block_fmha_pipeline_qr_ks_vs.hpp>  // TODO: Need to add this back in
#include <ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp>

#include "tilesparse_fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "fmha_fwd_epilogue.hpp"
#include "sddmm_tile_mask.hpp"

#include <torch/torch.h>
#include <torch/extension.h>

#if 1
using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;
#else
using QDataType           = ck::bhalf_t;
using KDataType           = ck::bhalf_t;
using VDataType           = ck::bhalf_t;
using SaccDataType        = float;       // data type for first gemm accumulation
using SMPLComputeDataType = float;       // data type for reduction, softmax
using PDataType           = ck::bhalf_t; // data type for A matrix of second gemm
using OaccDataType        = float;       // data type for second gemm accumulation
using ODataType           = ck::bhalf_t;
#endif

//                                                 M0   N0  K0   N1  K1  K0L
// using FmhaShape = ck::tile_program::TileFmhaShape<128,  64, 64, 128, 64>;
// using FmhaShape = ck::tile_program::TileFmhaShape<128, 256, 32, 128, 32>;
using VLayout = ck_tile::tensor_layout::gemm::RowMajor; // (bs, nhead) seqlen * hdim
// using VLayout = ck::tensor_layout::gemm::ColumnMajor; // (bs, nhead) hdim * seqlen

static constexpr ck::index_t M0_h128  = 128;
static constexpr ck::index_t N0_h128  = 128;
static constexpr ck::index_t K0_h128  = 32;
static constexpr ck::index_t N1_h128  = 128;
static constexpr ck::index_t K1_h128  = 32;
static constexpr ck::index_t K0L_h128 = 128;

// static constexpr ck::index_t M0_h128  = 64;
// static constexpr ck::index_t N0_h128  = 64;
// static constexpr ck::index_t K0_h128  = 32;
// static constexpr ck::index_t N1_h128  = 128;
// static constexpr ck::index_t K1_h128  = 32;
// static constexpr ck::index_t K0L_h128 = 128;

using FmhaBlockTileHdim128 = ck::Sequence<M0_h128, N0_h128, K0_h128, N1_h128, K1_h128, K0L_h128>;
using FmhaBlockWarps       = ck::Sequence<4, 1, 1>;
// using FmhaBlockWarps       = ck::Sequence<2, 1, 1>;
using FmhaWarpTile         = ck::Sequence<32, 32, 16>;
static constexpr ck::index_t BlockSize = FmhaBlockWarps::At(0) *
                                         FmhaBlockWarps::At(1) *
                                         FmhaBlockWarps::At(2) *
                                         ck::get_warp_size();

using FmhaShapeHDim128     = ck_tile::TileFmhaShape<FmhaBlockTileHdim128,
                                                         FmhaBlockWarps,
                                                         FmhaWarpTile,
                                                         FmhaBlockWarps,
                                                         FmhaWarpTile,
                                                         ck::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>>;

using SddmmTileMaskHdim128 = SddmmTileMask<M0_h128, N0_h128>;

using FmhaTilePartitionerHDim128 = FmhaFwdTilePartitioner<FmhaShapeHDim128>;
using FmhaPipelineProblemHDim128 =
    ck_tile::BlockFmhaPipelineProblem<QDataType,
                                                      KDataType,
                                                      VDataType,
                                                      SaccDataType,
                                                      SMPLComputeDataType,
                                                      PDataType,
                                                      OaccDataType,
                                                      ODataType,
                                                      BlockSize,
                                                      FmhaShapeHDim128>;

using FmhaPipelineHDim128 =
    ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblemHDim128>;

using FmhaEpilogue     = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;
using FmhaKernelHDim128 =
    FmhaFwdKernel<FmhaTilePartitionerHDim128, FmhaPipelineHDim128, FmhaEpilogue>;

template <typename FmhaKernel>
float invoker_fmha_kernel(const void* q_ptr,
                          const void* k_ptr,
                          const void* v_ptr,
                          void* o_ptr,
                          const void *mask_offsets_ptr,
                          const void *mask_indices_ptr,
                          ck::index_t batch,
                          ck::index_t nhead,
                          ck::index_t seqlen_q,
                          ck::index_t seqlen_k,
                          ck::index_t hdim_q,
                          ck::index_t hdim_v,
                          ck::index_t mask_cols,
                          float scale,
                          bool i_perm,
                          bool o_perm)
{
    dim3 kGridSize            = FmhaKernel::GridSize(batch, nhead, seqlen_q, hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernel::VLayout, ck_tile::tensor_layout::gemm::RowMajor>;

    // batch * nhead * seqlen * hdim or batch * seqlen * nhead * hdim
    auto kargs = FmhaKernel::MakeKargs(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        mask_offsets_ptr,
        mask_indices_ptr,
        seqlen_q, // seqlen_q
        seqlen_k, // seqlen_k
        hdim_q,   // hdim_q
        hdim_v,   // hdim_v
        scale,
        i_perm ? hdim_q : nhead * hdim_q, // stride_q
        i_perm ? hdim_q : nhead * hdim_q, // stride_k
        [&]() {
            if constexpr(is_v_rowmajor)
                return i_perm ? hdim_v : nhead * hdim_v;
            else
                return i_perm ? seqlen_k : nhead * seqlen_k;
        }(),                                 // stride_v
        o_perm ? hdim_v : nhead * hdim_v,    // stride_o
        i_perm ? seqlen_q * hdim_q : hdim_q, // nhead_stride_q
        i_perm ? seqlen_k * hdim_q : hdim_q, // nhead_stride_k
        [&]() {
            if constexpr(is_v_rowmajor)
                return i_perm ? seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_k : seqlen_k;
        }(),                                 // nhead_stride_v
        o_perm ? seqlen_q * hdim_v : hdim_v, // nhead_stride_o
        nhead * seqlen_q * hdim_q,           // batch_stride_q
        nhead * seqlen_k * hdim_q,           // batch_stride_k
        nhead * hdim_v * seqlen_k,           // batch_stride_v
        nhead * seqlen_q * hdim_v);          // batch_stride_o

    float ave_time = launch_kernel<kBlockSize.x, kBlockPerCu>(StreamConfig{nullptr, true},
                                                              FmhaKernel{},
                                                              kGridSize,
                                                              kBlockSize,
                                                              mask_cols * sizeof(mask_cols),
                                                              kargs); // BatchStrideO
    return ave_time;
}


torch::Tensor fmha_kernel_dispatch(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor attn_blk_mask_indices,
    torch::Tensor attn_blk_mask_offests,
    float dropout_p,
    bool is_causal,
    float scale
) {
    // TODO: Handle data type dispatch (currently has data type hardcoded I believe)

    // asume 4d input
    int n = query.size(0);
    int s = key.size(2);
    int l = query.size(2);
    int e_qk = query.size(3);
    int e_v = value.size(3);

    auto other_batch_dim = query.sizes().slice(1, query.dim() - 3);  // https://pytorch.org/cppdocs/api/classc10_1_1_array_ref.html#_CPPv4NK3c108ArrayRef5sliceE6size_t6size_t
    // torch::Tensor output = torch::empty({n, other_batch_dim..., l, e_v}, options);  // TODO

    auto options =torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(query.device().type());
    torch::Tensor output = torch::empty({n, query.size(1), l, e_v}, options);

    // if (e_qk == e_qk && e_qk == 128) {
        // There's a macro "CK_TIME_KERNEL" that I need to diable for e2e performance measurements
        invoker_fmha_kernel<FmhaKernelHDim128>(
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            output.data_ptr(),
            attn_blk_mask_indices.data_ptr(),  // TODO
            attn_blk_mask_offests.data_ptr(),  // TODO
            n,
            query.size(1),  // head dimension
            l,
            s,
            e_qk,
            e_qk,
            SddmmTileMaskHdim128::TileN / s,  // TODO: verify it's K and not Q
            scale,
            true,  // Hardcode to true for now because that will match the dimension ordering from the torch API
            true
        );
    // } else {
    //     // std::cout << "not support hdim, will not run" << std::endl;
    //     // return -1;
    //     // TODO: Need to look up a torch exception to throw here
    // }

    return output;
}

// CURRENT GOAL: End to end function correctness, IGNORE PERFORMANCE
torch::Tensor scaled_dot_product_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor attn_blk_mask_offsets,  // FIXME: Will probably need support for static and dynamic attention masking
    torch::Tensor attn_blk_mask_indices,
    float dropout_p,
    bool is_causal,
    float scale
) {
    return fmha_kernel_dispatch(
        query,
        key,
        value,
        attn_blk_mask_offsets,
        attn_blk_mask_indices,
        dropout_p,
        is_causal,
        scale
    );
}


PYBIND11_MODULE(bsp_attn_ext, m) {
    m.def("scaled_dot_product_attention", &scaled_dot_product_attention, "scaled dot product attention");
}
