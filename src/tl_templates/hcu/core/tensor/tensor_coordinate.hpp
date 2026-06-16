// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include <tl_templates/hcu/core/config.hpp>
#include <tl_templates/hcu/core/numeric/integer.hpp>
#include <tl_templates/hcu/core/numeric/integral_constant.hpp>
#include <tl_templates/hcu/core/algorithm/coordinate_transform.hpp>
#include <tl_templates/hcu/core/tensor/tensor_adaptor.hpp>
#include <tl_templates/hcu/core/tensor/tensor_adaptor_coordinate.hpp>
#include <tl_templates/hcu/core/container/container_helper.hpp>
#include <tl_templates/hcu/core/container/multi_index.hpp>
#include <tl_templates/hcu/core/numeric/math.hpp>
#include <tl_templates/hcu/core/utility/type_traits.hpp>

namespace tl {

template <index_t NDimHidden, typename TopDimensionHiddenIds>
struct tensor_coordinate
    : public tensor_adaptor_coordinate<NDimHidden, sequence<0>, TopDimensionHiddenIds>
{
    using Base = tensor_adaptor_coordinate<NDimHidden, sequence<0>, TopDimensionHiddenIds>;

    // TODO make these private
    static constexpr index_t ndim_top_ = TopDimensionHiddenIds::size();

    using HiddenIndex = multi_index<NDimHidden>;
    using TopIndex    = multi_index<ndim_top_>;

    public:
    TL_HOST_DEVICE constexpr tensor_coordinate() = default;

    TL_HOST_DEVICE constexpr tensor_coordinate(const HiddenIndex& idx_hidden)
        : Base{idx_hidden}
    {
    }

    // construct from TensorAdaptorCoordinte base class
    TL_HOST_DEVICE constexpr tensor_coordinate(const Base& adaptor_coord) : Base{adaptor_coord}
    {
    }

    TL_HOST_DEVICE constexpr auto get_index() const { return Base::get_top_index(); }

    TL_HOST_DEVICE constexpr index_t get_offset() const
    {
        return Base::get_bottom_index()[number<0>{}];
    }

    TL_HOST_DEVICE constexpr const auto& get_hidden_index() const
    {
        return Base::get_hidden_index();
    }

    TL_HOST_DEVICE auto& get_hidden_index() { return Base::get_hidden_index(); }
};

template <typename TensorDesc, typename TopIndex>
TL_HOST_DEVICE constexpr auto make_tensor_coordinate(const TensorDesc& tensor_desc,
                                                          const TopIndex& idx_top)
{
    const auto adaptor_coord = make_tensor_adaptor_coordinate(tensor_desc, idx_top);

    return tensor_coordinate<TensorDesc::get_num_of_hidden_dimension(),
                             remove_cvref_t<decltype(TensorDesc::get_top_dimension_hidden_ids())>>{
        adaptor_coord};
}

template <bool JudgeDoTransforms = true, typename TensorDesc, typename TensorCoord, typename Index>
TL_HOST_DEVICE constexpr void
move_tensor_coordinate(const TensorDesc& tensor_desc, TensorCoord& coord, const Index& coord_step)
{
    move_tensor_adaptor_coordinate(tensor_desc, coord, coord_step);
}

template <typename TensorDesc, typename TensorCoord>
TL_HOST_DEVICE constexpr bool
coordinate_has_valid_offset_assuming_top_index_is_valid(const TensorDesc& tensor_desc,
                                                        const TensorCoord& coord)
{
    return adaptor_coordinate_is_valid_assuming_top_index_is_valid(tensor_desc, coord);
}

template <typename TensorDesc, typename TensorCoord>
TL_HOST_DEVICE constexpr bool coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord)
{
    return adaptor_coordinate_is_valid(tensor_desc, coord);
}

} // namespace tl
