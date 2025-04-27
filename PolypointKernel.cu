#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "common.h"

namespace gpu {

__device__ double signDistance(const geom::Plane &p, const double3 &point)
{
	return p.a * point.x + p.b * point.y + p.c * point.z + p.d;
}

__device__ void solve4x4(const double A[16], const double B[4], double X[4])
{
	// Simple Gauss elimination for small 4x4 system

	double M[4][5]; // 4x4 matrix + RHS
#pragma unroll
	for (int i = 0; i < 4; ++i) {
#pragma unroll
		for (int j = 0; j < 4; ++j) {
			M[i][j] = A[i * 4 + j];
		}
		M[i][4] = B[i];
	}

	// Forward elimination
#pragma unroll
	for (int i = 0; i < 4; ++i) {
		double pivot = M[i][i];
#pragma unroll
		for (int j = i; j <= 4; ++j) {
			M[i][j] /= pivot;
		}
#pragma unroll
		for (int k = i + 1; k < 4; ++k) {
			double factor = M[k][i];
#pragma unroll
			for (int j = i; j <= 4; ++j) {
				M[k][j] -= factor * M[i][j];
			}
		}
	}

	// Back substitution
#pragma unroll
	for (int i = 3; i >= 0; --i) {
		X[i] = M[i][4];
#pragma unroll
		for (int j = i + 1; j < 4; ++j) {
			X[i] -= M[i][j] * X[j];
		}
	}
}

// clang-format off
__device__ geom::Plane getPolypointPlaneCUDA(
    const geom::Plane &plane, const double3 *origBasises, const double3 *resBasises, int basisCount)
{
	double acc[14] = {0.0}; // instead a1, b1, c1, ..., d4
	double rhs[4] = {0.0}; // instead r1, r2, r3, r4

	for (int i = 0; i < basisCount; ++i) {
		const double3 &orig_basis_p = origBasises[i];
		const double3 &res_basis_p = resBasises[i];

		const double gamma = signDistance(plane, orig_basis_p);
		const double gamma_inv = __drcp_rn(gamma); // device func: fast approximate "1 / gamma"
		const double gamma_squared_inv = gamma_inv * gamma_inv; // device func: "1 / gamma^2"

		const double x = res_basis_p.x;
		const double y = res_basis_p.y;
		const double z = res_basis_p.z;

		acc[0] += x * x * gamma_squared_inv; // a1 = x * x / gamma_squared;
		acc[1] += x * y * gamma_squared_inv; // b1 = x * y / gamma_squared;
		acc[2] += x * z * gamma_squared_inv; // c1 = x * z / gamma_squared;
		acc[3] += x * gamma_squared_inv;     // d1 = x / gamma_squared;

		acc[4] += y * y * gamma_squared_inv; // b2 = y * y / gamma_squared;
		acc[5] += y * z * gamma_squared_inv; // c2 = y * z / gamma_squared;
		acc[6] += y * gamma_squared_inv;     // d2 = y / gamma_squared;

		acc[7] += z * z * gamma_squared_inv; // c3 = z * z / gamma_squared;
		acc[8] += z * gamma_squared_inv;     // d3 = z / gamma_squared;

		acc[9]  += gamma_squared_inv;        // d4 = 1 / gamma_squared;

		rhs[0] += x * gamma_inv;             // r1 = x / gamma;
		rhs[1] += y * gamma_inv;             // r2 = y / gamma;
		rhs[2] += z * gamma_inv;             // r3 = z / gamma;
		rhs[3] += gamma_inv;                 // r4 = 1 / gamma;
	}

	const double A[16] = {
	    acc[0] + consts::reg_term, acc[1], acc[2], acc[3],
	    acc[1], acc[4] + consts::reg_term, acc[5], acc[6],
	    acc[2], acc[5], acc[7] + consts::reg_term, acc[8],
	    acc[3], acc[6], acc[8], acc[9] + consts::reg_term,
	};

	double X[4];
	solve4x4(A, rhs, X);

	return geom::Plane{plane.id, X[0], X[1], X[2], X[3]};
}

// clang-format on

// Deform each plain on CUDA thread
__global__ void deformPlanesPolypointKernel(
    const geom::Plane *inPlanes,
    geom::Plane *outPlanes,
    const double3 *origBasises,
    const double3 *resBasises,
    int basisCount,
    int planeCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= planeCount) {
		return;
	}

	outPlanes[idx] = getPolypointPlaneCUDA(inPlanes[idx], origBasises, resBasises, basisCount);
}

void deformPlanesPolypoint(
    __out geom::PlaneList &outPlanes,
    const geom::PlaneList &inPlanes,
    const geom::BasisList &origBasises,
    const geom::BasisList &resBasises)
{
	// Must be the same size for Polypoint transformation!
	assert(origBasises.size() == resBasises.size());

	int planeCount = inPlanes.size();
	int basisCount = origBasises.size();

	// Upload data to GPU
	thrust::device_vector<geom::Plane> d_inPlanes = inPlanes;
	thrust::device_vector<geom::Plane> d_outPlanes(planeCount);
	thrust::device_vector<double3> d_origBasises = origBasises;
	thrust::device_vector<double3> d_resBasises = resBasises;

	// Launch kernel
	int blockSize = 256;
	int gridSize = (planeCount + blockSize - 1) / blockSize;

	deformPlanesPolypointKernel<<<gridSize, blockSize>>>(
	    thrust::raw_pointer_cast(d_inPlanes.data()),
	    thrust::raw_pointer_cast(d_outPlanes.data()),
	    thrust::raw_pointer_cast(d_origBasises.data()),
	    thrust::raw_pointer_cast(d_resBasises.data()),
	    basisCount,
	    planeCount);
	cudaDeviceSynchronize();

	// Download result from GPU
	outPlanes.resize(planeCount);
	thrust::copy(d_outPlanes.begin(), d_outPlanes.end(), outPlanes.begin());
}

} // namespace gpu
