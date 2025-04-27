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

__device__ geom::Plane getPolypointPlaneCUDA(
    const geom::Plane &plane, const double3 *origBasises, const double3 *resBasises, int basisCount)
{
	double a1 = 0, b1 = 0, c1 = 0, d1 = 0, r1 = 0;
	double b2 = 0, c2 = 0, d2 = 0, r2 = 0;
	double c3 = 0, d3 = 0, r3 = 0;
	double d4 = 0, r4 = 0;

	for (int i = 0; i < basisCount; ++i) {
		const double3 &orig_basis_p = origBasises[i];
		const double3 &res_basis_p = resBasises[i];

		double gamma = signDistance(plane, orig_basis_p);
		double gamma_squared = gamma * gamma;

		double x = res_basis_p.x;
		double y = res_basis_p.y;
		double z = res_basis_p.z;

		a1 += x * x / gamma_squared;
		b1 += x * y / gamma_squared;
		c1 += x * z / gamma_squared;
		d1 += x / gamma_squared;

		b2 += y * y / gamma_squared;
		c2 += y * z / gamma_squared;
		d2 += y / gamma_squared;

		c3 += z * z / gamma_squared;
		d3 += z / gamma_squared;

		d4 += 1.0 / gamma_squared;

		r1 += x / gamma;
		r2 += y / gamma;
		r3 += z / gamma;
		r4 += 1.0 / gamma;
	}

	// clang-format off
    double A[16] = {
        a1 + consts::reg_term, b1, c1, d1,
        b1, b2 + consts::reg_term, c2, d2,
        c1, c2, c3 + consts::reg_term, d3,
        d1, d2, d3, d4 + consts::reg_term
    };
	// clang-format on

	double B[4] = {r1, r2, r3, r4};
	double X[4];
	solve4x4(A, B, X);

	return geom::Plane{plane.id, X[0], X[1], X[2], X[3]};
}

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
