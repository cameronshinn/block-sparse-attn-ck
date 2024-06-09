// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck/utility/common_header.hpp>
#include <ck_tile/core/tensor/store_tile.hpp>
#include <ck_tile/core/tensor/tile_elementwise.hpp>

template <typename OaccDataType_, typename ODataType_>
struct FmhaFwdEpilogueProblem
{
    using OaccDataType = ck::remove_cvref_t<OaccDataType_>;
    using ODataType    = ck::remove_cvref_t<ODataType_>;
};

template <typename Problem_, typename Policy_ = void>
struct FmhaFwdEpilogue
{
    using Problem      = ck::remove_cvref_t<Problem_>;
    using OaccDataType = ck::remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType    = ck::remove_cvref_t<typename Problem::ODataType>;

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    template <typename ODramWindowTmp, typename OAccTile>
    __device__ auto operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile)
    {
        using namespace ck;
        using namespace ck::tile_program;

        const auto o = tile_elementwise_in(type_convert<ODataType, OaccDataType>, o_acc_tile);
        store_tile(o_dram_window_tmp, o);
    }
};
