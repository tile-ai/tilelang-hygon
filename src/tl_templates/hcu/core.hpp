// TileLang HCU device header umbrella (trimmed from composable_kernel
// descriptor stack). Third-party MIT parts: see vendor/NOTICE.md. HCU
// MLS/matrix paths live under mls/.

#pragma once

#include <tl_templates/hcu/core/config.hpp>

// algorithm (CK MIT)
#include <tl_templates/hcu/core/algorithm/coordinate_transform.hpp>
#include <tl_templates/hcu/core/algorithm/indexing_adaptor.hpp>
#include <tl_templates/hcu/core/algorithm/space_filling_curve.hpp>

// arch: buffer addressing + HCU target helpers (MLS-specific arch is under
// mls/)
#include <tl_templates/hcu/core/arch/amd_buffer_addressing.hpp>
#include <tl_templates/hcu/core/arch/arch.hpp>
#include <tl_templates/hcu/core/arch/utility.hpp>

// container + numeric
#include <tl_templates/hcu/core/container/array.hpp>
#include <tl_templates/hcu/core/container/container_helper.hpp>
#include <tl_templates/hcu/core/container/map.hpp>
#include <tl_templates/hcu/core/container/multi_index.hpp>
#include <tl_templates/hcu/core/container/sequence.hpp>
#include <tl_templates/hcu/core/container/statically_indexed_array.hpp>
#include <tl_templates/hcu/core/container/thread_buffer.hpp>
#include <tl_templates/hcu/core/container/tuple.hpp>
#include <tl_templates/hcu/core/numeric/bfloat16.hpp>
#include <tl_templates/hcu/core/numeric/float8.hpp>
#include <tl_templates/hcu/core/numeric/half.hpp>
#include <tl_templates/hcu/core/numeric/int8.hpp>
#include <tl_templates/hcu/core/numeric/integer.hpp>
#include <tl_templates/hcu/core/numeric/integral_constant.hpp>
#include <tl_templates/hcu/core/numeric/math.hpp>
#include <tl_templates/hcu/core/numeric/numeric.hpp>
#include <tl_templates/hcu/core/numeric/type_convert.hpp>
#include <tl_templates/hcu/core/numeric/vector_type.hpp>

// tensor (CK MIT descriptor stack)
#include <tl_templates/hcu/core/tensor/tensor_adaptor.hpp>
#include <tl_templates/hcu/core/tensor/tensor_adaptor_coordinate.hpp>
#include <tl_templates/hcu/core/tensor/tensor_coordinate.hpp>
#include <tl_templates/hcu/core/tensor/tensor_descriptor.hpp>
#include <tl_templates/hcu/core/tensor/tile_distribution_encoding.hpp>

// utility
#include <tl_templates/hcu/core/utility/bit_cast.hpp>
#include <tl_templates/hcu/core/utility/functional.hpp>
#include <tl_templates/hcu/core/utility/functional_with_tuple.hpp>
#include <tl_templates/hcu/core/utility/magic_div.hpp>
#include <tl_templates/hcu/core/utility/to_sequence.hpp>
#include <tl_templates/hcu/core/utility/type_traits.hpp>
