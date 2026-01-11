#pragma once
#include <cmath>
#include <type_traits>

// fma_auto(a,b,c) will pick float FMA (fmaf) iff the common type is float.
// Otherwise it uses double/long double fma.

#if defined(__CUDACC__)
#	define HD __host__ __device__ __forceinline__
#else
#	define HD inline
#endif

template<class T>
HD T fma_typed(T a, T b, T c)
{
#if defined(__CUDA_ARCH__)
	// Device (GPU) path
	if constexpr (std::is_same_v<T, float>) {
		return __fmaf_rn(a, b, c); // fused float FMA
	} else if constexpr (std::is_same_v<T, double>) {
		return __fma_rn(a, b, c); // fused double FMA
	} else {
		// long double not really supported on device; fall back to fma on host compilation
		static_assert(false); // slef-check. shouldn't used by now
		return T(a * b + c);
	}
#else
	// Host (CPU) path
	throw "Not Expected to be used!";
	if constexpr (std::is_same_v<T, float>) {
		return std::fmaf(a, b, c);
	} else {
		return std::fma(a, b, c); // works for double and long double
	}
#endif
}

template<class A, class B, class C>
HD auto fma_auto(A a, B b, C c)
{
	using T = std::common_type_t<A, B, C>;
	return fma_typed<T>(static_cast<T>(a), static_cast<T>(b), static_cast<T>(c));
}

// Usage:
// float  rf = fma_auto(af, bf, cf);   // uses fmaf / __fmaf_rn
// double rd = fma_auto(ad, bd, cd);   // uses fma  / __fma_rn
