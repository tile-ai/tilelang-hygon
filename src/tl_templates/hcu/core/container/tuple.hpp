// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <tl_templates/hcu/core/config.hpp>
#include <tl_templates/hcu/core/container/sequence.hpp>
#include <tl_templates/hcu/core/numeric/integer.hpp>
#include <tl_templates/hcu/core/numeric/integral_constant.hpp>
#include <tl_templates/hcu/core/numeric/math.hpp>
#include <tl_templates/hcu/core/utility/functional.hpp>
#include <tl_templates/hcu/core/utility/type_traits.hpp>
#include <utility>
#include <initializer_list>

#ifndef TL_TUPLE_IMPL
#define TL_TUPLE_IMPL 1
#endif

namespace tl {

namespace impl {
template <typename T, index_t N>
struct tuple_array_impl;
}

template <typename T, index_t N>
using tuple_array = typename impl::tuple_array_impl<T, N>::type;

namespace impl {

// the place where content is stored
template <index_t idx, typename T, bool is_empty = std::is_empty_v<T>>
struct tuple_object
{
};

template <index_t idx, typename T>
struct tuple_object<idx, T, true>
{
    TL_HOST_DEVICE constexpr tuple_object() {}
#if TL_TUPLE_IMPL == 0
    template <typename U>
    TL_HOST_DEVICE constexpr tuple_object(U&&)
    {
    }
    template <typename U>
    TL_HOST_DEVICE constexpr tuple_object(const U&)
    {
    }
    template <typename U>
    TL_HOST_DEVICE constexpr tuple_object(U&)
    {
    }
#elif TL_TUPLE_IMPL == 1
    template <typename U,
              typename std::enable_if<!std::is_same<remove_cvref_t<U>, tuple_object>::value,
                                      bool>::type = false>
    TL_HOST_DEVICE constexpr tuple_object(U&&)
    {
    }
#endif
};

template <index_t idx, typename T>
struct tuple_object<idx, T, false>
{
    TL_HOST_DEVICE constexpr tuple_object() : element{} {}
#if TL_TUPLE_IMPL == 0
    template <typename U>
    TL_HOST_DEVICE constexpr tuple_object(U&& e) : element(std::forward<U>(e))
    {
    }
    template <typename U>
    TL_HOST_DEVICE constexpr tuple_object(const U& e) : element(e)
    {
    }
    template <typename U>
    TL_HOST_DEVICE constexpr tuple_object(U& e) : element(e)
    {
    }
#elif TL_TUPLE_IMPL == 1
    template <typename U,
              typename std::enable_if<!std::is_same<remove_cvref_t<U>, tuple_object>::value,
                                      bool>::type = false>
    TL_HOST_DEVICE constexpr tuple_object(U&& e) : element(std::forward<U>(e))
    {
    }
#endif
    T element;
};

// NOTE: we return a instance(not a reference) if content is empty
template <index_t I, class T>
TL_HOST_DEVICE constexpr T getv(const tuple_object<I, T, true>&)
{
    return {};
}

template <index_t I, class T>
TL_HOST_DEVICE constexpr const T& getv(const tuple_object<I, T, false>& x)
{
    return x.element;
}

template <index_t I, class T>
TL_HOST_DEVICE constexpr T& getv(tuple_object<I, T, false>& x)
{
    return x.element;
}

template <index_t I, class T>
TL_HOST_DEVICE constexpr T&& getv(tuple_object<I, T, false>&& x)
{
    return static_cast<T&&>(x.element);
}

template <typename index_seq, typename... T>
struct tuple_base;

template <index_t... I, typename... T>
struct tuple_base<sequence<I...>, T...> : tuple_object<I, T>...
{
    TL_HOST_DEVICE constexpr tuple_base() = default;

#if TL_TUPLE_CTOR_WITH_INITIALIZER_LIST
#define _ILE() (std::initializer_list<U>{}.size() - 1)
    template <typename U>
    TL_HOST_DEVICE constexpr tuple_base(std::initializer_list<U> us)
        : tuple_object<I, T>(static_cast<T>(*(us.begin() + (I >= _ILE() ? _ILE() : I))))...
    {
    }
#undef _ILE
#endif

#if TL_TUPLE_IMPL == 0
    template <class... U>
    TL_HOST_DEVICE constexpr explicit tuple_base(U&&... u)
        : tuple_object<I, T>(std::forward<U>(u))...
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr explicit tuple_base(const U&... u) : tuple_object<I, T>(u)...
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr explicit tuple_base(U&... u) : tuple_object<I, T>(u)...
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple_base(tuple_base<sequence<I...>, U...>&& u)
        : tuple_object<I, T>(getv(static_cast<tuple_object<I, U>&&>(u)))...
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple_base(const tuple_base<sequence<I...>, U...>& u)
        : tuple_object<I, T>(getv(static_cast<const tuple_object<I, U>&>(u)))...
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple_base(tuple_base<sequence<I...>, U...>& u)
        : tuple_object<I, T>(getv(static_cast<tuple_object<I, U>&>(u)))...
    {
    }
#elif TL_TUPLE_IMPL == 1
    template <class U,
              typename std::enable_if<sizeof...(I) == 1 && sizeof...(T) == 1 &&
                                          !std::is_same<remove_cvref_t<U>, tuple_base>::value,
                                      bool>::type = false>
    TL_HOST_DEVICE constexpr tuple_base(U&& u) : tuple_object<I, T>(std::forward<U>(u))...
    {
    }

    template <typename... U, typename std::enable_if<sizeof...(U) >= 2, bool>::type = false>
    TL_HOST_DEVICE constexpr tuple_base(U&&... u) : tuple_object<I, T>(std::forward<U>(u))...
    {
        static_assert(sizeof...(I) == sizeof...(T) && sizeof...(I) == sizeof...(U),
                      "wrong! inconsistent size");
    }

#endif
};
} // namespace impl

template <class... T>
struct tuple : impl::tuple_base<make_index_sequence<sizeof...(T)>, T...>
{
    TL_HOST_DEVICE
    static constexpr auto size() { return sizeof...(T); }
    using base = impl::tuple_base<make_index_sequence<sizeof...(T)>, T...>;
    TL_HOST_DEVICE constexpr tuple() = default;

#if TL_TUPLE_CTOR_WITH_INITIALIZER_LIST
    template <typename U>
    TL_HOST_DEVICE constexpr tuple(std::initializer_list<U> us) : base(us)
    {
    }
#endif

#if TL_TUPLE_IMPL == 0
    template <class... U>
    TL_HOST_DEVICE constexpr tuple(U&&... u) : base(std::forward<U>(u)...)
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple(const U&... u) : base(u...)
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple(U&... u) : base(u...)
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple(tuple<U...>&& u)
        : base(static_cast<impl::tuple_base<make_index_sequence<sizeof...(U)>, U...>&&>(u))
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple(const tuple<U...>& u)
        : base(static_cast<const impl::tuple_base<make_index_sequence<sizeof...(U)>, U...>&>(u))
    {
    }

    template <class... U>
    TL_HOST_DEVICE constexpr tuple(tuple<U...>& u)
        : base(static_cast<impl::tuple_base<make_index_sequence<sizeof...(U)>, U...>&>(u))
    {
    }
#elif TL_TUPLE_IMPL == 1
    template <
        typename U,
        typename std::enable_if<sizeof...(T) == 1 && !std::is_same<remove_cvref_t<U>, tuple>::value,
                                bool>::type = false>
    TL_HOST_DEVICE constexpr tuple(U&& u) : base(std::forward<U>(u))
    {
    }

    template <typename... U,
              typename std::enable_if<sizeof...(U) == sizeof...(T) && sizeof...(U) >= 2,
                                      bool>::type = false>
    TL_HOST_DEVICE constexpr tuple(U&&... u) : base(std::forward<U>(u)...)
    {
    }
#endif
    TL_HOST_DEVICE static constexpr bool is_static()
    {
        bool flag = true;

        static_for<0, sizeof...(T), 1>{}([&flag](auto i) {
            flag &= is_static_v<remove_cvref_t<__type_pack_element<i.value, T...>>>;
        });

        return flag;
    }

    TL_HOST_DEVICE static constexpr bool IsTuple() { return true; }

#define TP_COM_() static_assert(I < size(), "wrong! out of range")
    // clang-format off
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get() const &          { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get(number<I>) const & { TP_COM_(); return get<I>(); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get() &                { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get(number<I>) &       { TP_COM_(); return get<I>(); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get() &&               { TP_COM_(); return impl::getv<I>(std::move(*this)); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get(number<I>) &&      { TP_COM_(); return std::move(*this).template get<I>(); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get() const &&         { TP_COM_(); return impl::getv<I>(std::move(*this)); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) get(number<I>) const &&{ TP_COM_(); return std::move(*this).template get<I>(); }

    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) at() const          { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) at(number<I>) const { TP_COM_(); return get<I>(); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) at()                      { TP_COM_(); return impl::getv<I>(*this); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) at(number<I>)             { TP_COM_(); return get<I>(); }

    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) operator[](number<I>)             { TP_COM_(); return get<I>(); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) operator[](number<I>) const { TP_COM_(); return get<I>(); }
    template<index_t I> TL_HOST_DEVICE constexpr decltype(auto) operator()(number<I>)             { TP_COM_(); return get<I>(); }  // TODO: compatible

    // below function should be used under tuple_array<> type, no extra check will perform here
    template <typename Tx> TL_HOST_DEVICE constexpr decltype(auto) get_as()                            { return reinterpret_cast<tuple_array<Tx, size()>&>(*this); }
    template <typename Tx> TL_HOST_DEVICE constexpr decltype(auto) get_as() const                      { return reinterpret_cast<const tuple_array<Tx, size()>&>(*this); }
    // below index is for index *AFTER* type convert, not before
    //template <typename Tx> TL_HOST_DEVICE constexpr decltype(auto) get_as(index_t i)                   { TP_COM_(); return reinterpret_cast<tuple_array<Tx, size()>&>(*this).at(i); }
    //template <typename Tx> TL_HOST_DEVICE constexpr decltype(auto) get_as(index_t i) const             { TP_COM_(); return reinterpret_cast<const tuple_array<Tx, size()>&>(*this).at(i); }
    template <typename Tx, index_t I> TL_HOST_DEVICE constexpr decltype(auto) get_as(number<I>)        { TP_COM_(); return reinterpret_cast<tuple_array<Tx, size()>&>(*this).at(number<I>{}); }
    template <typename Tx, index_t I> TL_HOST_DEVICE constexpr decltype(auto) get_as(number<I>) const  { TP_COM_(); return reinterpret_cast<const tuple_array<Tx, size()>&>(*this).at(number<I>{}); }
    
    // template <typename Tx> TL_HOST_DEVICE constexpr void set_as(index_t i, const Tx & x)               { TP_COM_(); reinterpret_cast<tuple_array<Tx, size()>&>(*this).at(i) = x; }
    template <typename Tx, index_t I> TL_HOST_DEVICE constexpr void set_as(number<I>, const Tx & x)    { TP_COM_(); reinterpret_cast<tuple_array<Tx, size()>&>(*this).at(number<I>{}) = x; }

    // clang-format on
#undef TP_COM_
};

template <typename... T>
TL_HOST_DEVICE void print(const tuple<T...>& t)
{
    printf("tuple<");
    if constexpr(sizeof...(T) > 0)
    {
        bool first = true;
        static_for<0, sizeof...(T), 1>{}([&t, &first](auto i) {
            if(!first)
                printf(", ");
            print(t.get(i));
            first = false;
        });
    }
    printf(">");
}

template <typename, typename>
struct vector_traits;

// specialization for array
template <typename... T>
struct vector_traits<tuple<T...>, void>
{
    using scalar_type                    = __type_pack_element<0, T...>;
    static constexpr index_t vector_size = sizeof...(T);
};

// template <class... T>
// TL_HOST_DEVICE constexpr
// tuple<T...>
// make_tuple(T const&... t)
// {
//     return {t...};
// }
template <typename... Xs>
TL_HOST_DEVICE constexpr bool operator==(const tuple<Xs...>& a, const tuple<Xs...>& b)
{
    bool same = true;

    static_for<0, sizeof...(Xs), 1>{}([&](auto i) {
        if(a[i] != b[i])
        {
            same = false;
        }
    });

    return same;
}

template <typename... Xs>
TL_HOST_DEVICE constexpr bool operator!=(const tuple<Xs...>& a, const tuple<Xs...>& b)
{
    return !(a == b);
}

template <typename... Xs>
TL_HOST_DEVICE constexpr auto make_tuple(Xs&&... xs)
{
    // here xs is always a lvalue as function arg
    // Xs may deduced as (e.g try to pass in a integer in following cases)
    //  1). if pass in a rvalue (like function return or int{}) -> Xs is "int"
    //  2). if pass in a const lvalue -> Xs is "const int &"
    //  3). if pass in a non-const lvalue -> Xs is "int &"
    // so the return type of std::forward will dependes on Xs
    //  1). std::forward -> int&&
    //  2). std::forward -> const int&
    //  3). std::forward -> int&
    return tuple<remove_cvref_t<Xs>...>(std::forward<Xs>(xs)...);
}

// https://en.cppreference.com/w/cpp/utility/tuple/tie
template <typename... Args>
constexpr tuple<Args&...> tie(Args&... args) noexcept
{
    return {args...};
}

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<tuple<Xs...>, tuple<Ys...>>
{
    using type = tuple<Xs..., Ys...>;
};

namespace impl {
// be very careful using this type (because we want the internal type)
// template deduction will fail if infering the inner type
// e.g.
// template<typename T, index_t N> using some_wrapper = typename tuple_array_impl<T, N>::type;
// template<typename T, index_t N> void foo(const some_wrapper<T, N>&) {}
//   -> compiler will fail to deduce this type, because this is under non-deduced context
//   (https://en.cppreference.com/w/cpp/language/template_argument_deduction, "Non-deduced
//   contexts")
//
// -> use this instead
// template<typename Tup> void foo(const Tup&) {}
template <typename T, index_t N>
struct tuple_array_impl
{
    using type = typename tuple_concat<typename tuple_array_impl<T, N / 2>::type,
                                       typename tuple_array_impl<T, N - N / 2>::type>::type;
};

template <typename T>
struct tuple_array_impl<T, 0>
{
    using type = tuple<>;
};

template <typename T>
struct tuple_array_impl<T, 1>
{
    using type = tuple<T>;
};
} // namespace impl

template <typename F, index_t... ids>
TL_HOST_DEVICE constexpr auto generate_tuple_for(F&& f, sequence<ids...>)
{
    return make_tuple(f(number<ids>{})...);
}

template <typename F, index_t N>
TL_HOST_DEVICE constexpr auto generate_tuple(F&& f, number<N>)
{
    return generate_tuple_for(f, make_index_sequence<N>{});
}

template <typename F, index_t N>
TL_HOST_DEVICE constexpr auto generate_tie(F&& f, number<N>)
{
    return unpack([&f](auto&&... is) { return tie(f(is)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

// tx and ty are tuple of references, return type of will tuple of referennce (not rvalue)
template <typename... X, typename... Y>
TL_HOST_DEVICE constexpr auto concat_tuple_of_reference(const tuple<X&...>& tx,
                                                             const tuple<Y&...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

template <typename... X, typename... Y>
TL_HOST_DEVICE constexpr auto concat_tuple(const tuple<X...>& tx, const tuple<Y...>& ty)
{
    return unpack2(
        [&](auto... zs) { return tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

// Support any number of tuples to concat (also 1)
template <typename... X>
TL_HOST_DEVICE constexpr auto concat_tuple(const tuple<X...>& tx)
{
    return tx;
}

template <typename... X, typename... Tuples>
TL_HOST_DEVICE constexpr auto concat_tuple(const tuple<X...>& tx, const Tuples&... tuples)
{
    return concat_tuple(tx, concat_tuple(tuples...));
}

namespace detail {

template <typename F, typename X, index_t... Is>
TL_HOST_DEVICE constexpr auto transform_tuples_impl(F f, const X& x, sequence<Is...>)
{
    return make_tuple(f(x.at(number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
TL_HOST_DEVICE constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, sequence<Is...>)
{
    return make_tuple(f(x.at(number<Is>{}), y.at(number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
TL_HOST_DEVICE constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, sequence<Is...>)
{
    return make_tuple(f(x.at(number<Is>{}), y.at(number<Is>{}), z.at(number<Is>{}))...);
}

template <typename F, typename Tuple, index_t... Is>
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, sequence<Is...>)
{
    return std::forward<F>(f)(std::forward<Tuple>(t).get(number<Is>{})...);
}

} // namespace detail

template <typename F, typename X>
TL_HOST_DEVICE constexpr auto transform_tuples(F f, const X& x)
{
    return detail::transform_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::size(), 1>::type{});
}

template <typename F, typename X, typename Y>
TL_HOST_DEVICE constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return detail::transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::size(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
TL_HOST_DEVICE constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return detail::transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::size(), 1>::type{});
}

template <typename F, typename Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t)
{
    constexpr index_t N = std::decay_t<Tuple>::size();
    return detail::apply_impl(std::forward<F>(f), std::forward<Tuple>(t), make_index_sequence<N>{});
}

namespace detail {

template <typename F, typename X, index_t... Is>
TL_HOST_DEVICE constexpr auto embed_tuples_impl(F f, const X& x, sequence<Is...>)
{
    return concat_tuple(f(x.at(number<Is>{}))...);
}

} // namespace detail

// make sure F return at least a tuple
// e.g. x : tuple<X, Y>, f will return tuple<Z, W>
// this function will return
template <typename F, typename X>
TL_HOST_DEVICE constexpr auto embed_tuples(F f, const X& x)
{
    return detail::embed_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::size(), 1>::type{});
}

// By default unroll to the flatten
template <index_t Depth = 0, index_t MaxDepth = -1>
TL_HOST_DEVICE constexpr auto unroll_nested_tuple(const tuple<>& t)
{
    return t;
}

template <index_t Depth = 0, index_t MaxDepth = -1, typename T>
TL_HOST_DEVICE constexpr auto unroll_nested_tuple(const T& t)
{
    return make_tuple(t);
}

template <index_t Depth = 0, index_t MaxDepth = -1, typename... Ts>
TL_HOST_DEVICE constexpr auto unroll_nested_tuple(const tuple<Ts...>& t)
{
    if constexpr(Depth == MaxDepth)
    {
        return t;
    }
    else
    {
        return unpack(
            [&](auto&&... ts) {
                return concat_tuple(unroll_nested_tuple<Depth + 1, MaxDepth>(ts)...);
            },
            t);
    }
}

template <typename... Ts>
TL_HOST_DEVICE constexpr auto tuple_reverse(const tuple<Ts...>& t)
{
    return generate_tuple(
        [&](auto i) {
            using Idx = number<tuple<Ts...>::size() - i - 1>;
            return t.at(Idx{});
        },
        number<tuple<Ts...>::size()>{});
}

// Reduce tuple values in specific range using Function
template <index_t Idx, index_t End, typename F, typename... Ts>
TL_HOST_DEVICE constexpr auto tuple_reduce(F&& f, const tuple<Ts...>& t)
{
    static_assert(Idx < End, "Wrong parameters for tuple_reduce");
    if constexpr(Idx + 1 == End)
    {
        return t.at(number<Idx>{});
    }
    else
    {
        return f(t.at(number<Idx>{}), tuple_reduce<Idx + 1, End>(f, t));
    }
}

template <typename T>
using is_tuple = decltype(std::declval<T&>().IsTuple());

template <typename... Ts>
TL_HOST_DEVICE constexpr auto is_nested_tuple(const tuple<Ts...>&)
{
    return (is_detected<is_tuple, Ts>::value || ...);
}

template <index_t depth = 0, typename T>
TL_HOST_DEVICE constexpr auto tuple_depth(const T&)
{
    return depth;
}

template <index_t depth = 0, typename... Ts>
TL_HOST_DEVICE constexpr auto tuple_depth(const tuple<Ts...>&)
{
    return max(tuple_depth<depth + 1>(Ts{})...);
}

template <typename... Seqs>
TL_HOST_DEVICE constexpr auto to_array_of_array(tuple<Seqs...> t_of_s)
{
    constexpr index_t n0 = sizeof...(Seqs);

    constexpr index_t max_n1 = [&] {
        index_t max_n1_ = 0;

        static_for<0, n0, 1>{}([&](auto i0) {
            constexpr index_t n1 = t_of_s[i0].size();

            max_n1_ = max_n1_ < n1 ? n1 : max_n1_;
        });

        return max_n1_;
    }();

    array<array<index_t, max_n1>, n0> a_of_a{{-1}};

    static_for<0, n0, 1>{}([&](auto i0) {
        constexpr index_t n1 = t_of_s[i0].size();

        static_for<0, n1, 1>{}([&](auto i1) { a_of_a(i0)(i1) = t_of_s[i0][i1]; });
    });

    return a_of_a;
}

// Here should use MultiIndex<NSize>, instead of tuple<Ys...>, although the former
// is the alias of the latter. This is because compiler cannot infer the NSize if
// using MultiIndex<NSize>
// TODO: how to fix this?
template <typename... Ys,
          typename X,
          std::enable_if_t<!std::is_integral<X>::value && !std::is_floating_point<X>::value, bool> =
              false>
TL_HOST_DEVICE constexpr auto operator+=(tuple<Ys...>& y, const X& x)
{
    static_assert(X::size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y[i] += x[i]; });
    return y;
}

template <typename... Ys,
          typename X,
          std::enable_if_t<!std::is_integral<X>::value && !std::is_floating_point<X>::value, bool> =
              false>
TL_HOST_DEVICE constexpr auto operator-=(tuple<Ys...>& y, const X& x)
{
    static_assert(X::size() == sizeof...(Ys), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Ys);
    static_for<0, NSize, 1>{}([&](auto i) { y[i] -= x[i]; });
    return y;
}

template <typename... Xs,
          typename Y,
          std::enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> =
              false>
TL_HOST_DEVICE constexpr auto operator+(const tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] + y[i]; });
    return r;
}

template <typename... Xs, typename... Ys>
TL_HOST_DEVICE constexpr auto operator+(const tuple<Xs...>& x, const tuple<Ys...>& y)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong!");
    constexpr index_t NSize = sizeof...(Xs);
    return generate_tuple([&](auto i) { return x[i] + y[i]; }, number<NSize>{});
}

template <typename... Xs,
          typename Y,
          std::enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> =
              false>
TL_HOST_DEVICE constexpr auto operator-(const tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] - y[i]; });
    return r;
}

template <typename... Xs, typename... Ys>
TL_HOST_DEVICE constexpr auto operator-(const tuple<Xs...>& x, const tuple<Ys...>& y)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong!");
    constexpr index_t NSize = sizeof...(Xs);
    return generate_tuple([&](auto i) { return x[i] - y[i]; }, number<NSize>{});
}

template <typename... Xs,
          typename Y,
          std::enable_if_t<!std::is_integral<Y>::value && !std::is_floating_point<Y>::value, bool> =
              false>
TL_HOST_DEVICE constexpr auto operator*(const tuple<Xs...>& x, const Y& y)
{
    static_assert(Y::size() == sizeof...(Xs), "wrong! size not the same");
    constexpr index_t NSize = sizeof...(Xs);

    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = x[i] * y[i]; });
    return r;
}

// MultiIndex = scalar * MultiIndex
template <
    typename... Xs,
    typename Y,
    std::enable_if_t<std::is_integral<Y>::value || std::is_floating_point<Y>::value, bool> = false>
TL_HOST_DEVICE constexpr auto operator*(Y a, const tuple<Xs...>& x)
{
    constexpr index_t NSize = sizeof...(Xs);
    tuple<Xs...> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = a * x[i]; });
    return r;
}

// MultiIndex = MultiIndex * scalar
template <
    typename... Xs,
    typename Y,
    std::enable_if_t<std::is_integral<Y>::value || std::is_floating_point<Y>::value, bool> = false>
TL_HOST_DEVICE constexpr auto operator*(const tuple<Xs...>& x, Y a)
{
    return a * x;
}

template <typename... Xs, typename... Ys>
TL_HOST_DEVICE constexpr auto operator*(const tuple<Xs...>& x, const tuple<Ys...>& y)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong!");
    constexpr index_t NSize = sizeof...(Xs);
    return generate_tuple([&](auto i) { return x[i] * y[i]; }, number<NSize>{});
}

template <typename... Xs, typename... Ys>
TL_HOST_DEVICE constexpr auto operator/(const tuple<Xs...>& x, const tuple<Ys...>& y)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong!");
    constexpr index_t NSize = sizeof...(Xs);
    return generate_tuple([&](auto i) { return x[i] / y[i]; }, number<NSize>{});
}

} // namespace tl

#include <tuple>
// WARNING: needed by compiler for C++ structured binding support only, don't use this
namespace std {

template <typename... Ts>
struct tuple_size<tl::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)>
{
};

template <std::size_t I, typename... Ts>
struct tuple_element<I, tl::tuple<Ts...>> : std::tuple_element<I, std::tuple<Ts...>>
{
};

template <typename... Ts>
struct tuple_size<const tl::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)>
{
};

template <std::size_t I, typename... Ts>
struct tuple_element<I, const tl::tuple<Ts...>>
    : std::tuple_element<I, const std::tuple<Ts...>>
{
};

} // namespace std

#if 1
#define TO_TUPLE_OF_NUMBER(a, n)                                                             \
    _Pragma("clang diagnostic push") _Pragma(                                                \
        "clang diagnostic ignored \"-Wc++20-extensions\"")[a]<tl::index_t... IDX_IDX_>( \
        tl::sequence<IDX_IDX_...>)                                                      \
    {                                                                                        \
        return tl::tuple<tl::number<a[tl::number<IDX_IDX_>{}]>...>{};         \
    }                                                                                        \
    (tl::make_index_sequence<n>{}) _Pragma("clang diagnostic pop")
#else
#define TO_TUPLE_OF_NUMBER(arr, n_)                                                      \
    [&arr, n_] {                                                                         \
        static_assert(arr.size() >= n_, "wrong! out of bound");                          \
                                                                                         \
        static_assert(n_ < 7, "not implemented");                                        \
                                                                                         \
        if constexpr(n_ == 0)                                                            \
        {                                                                                \
            return tl::tuple<>{};                                                   \
        }                                                                                \
        else if constexpr(n_ == 1)                                                       \
        {                                                                                \
            return tl::tuple<number<arr[0]>>{};                                     \
        }                                                                                \
        else if constexpr(n_ == 2)                                                       \
        {                                                                                \
            return tl::tuple<number<arr[0]>, number<arr[1]>>{};                     \
        }                                                                                \
        else if constexpr(n_ == 3)                                                       \
        {                                                                                \
            return tl::tuple<number<arr[0]>, number<arr[1]>, number<arr[2]>>{};     \
        }                                                                                \
        else if constexpr(n_ == 4)                                                       \
        {                                                                                \
            return tl::                                                             \
                tuple<number<arr[0]>, number<arr[1]>, number<arr[2]>, number<arr[3]>>{}; \
        }                                                                                \
        else if constexpr(n_ == 5)                                                       \
        {                                                                                \
            return tl::tuple<number<arr[0]>,                                        \
                                  number<arr[1]>,                                        \
                                  number<arr[2]>,                                        \
                                  number<arr[3]>,                                        \
                                  number<arr[4]>>{};                                     \
        }                                                                                \
        else if constexpr(n_ == 6)                                                       \
        {                                                                                \
            return tl::tuple<number<arr[0]>,                                        \
                                  number<arr[1]>,                                        \
                                  number<arr[2]>,                                        \
                                  number<arr[3]>,                                        \
                                  number<arr[4]>,                                        \
                                  number<arr[5]>>{};                                     \
        }                                                                                \
    }()
#endif
