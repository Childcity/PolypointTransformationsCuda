#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "common.h"
#include "fma_auto.h"

namespace gpu {

__device__ geom::real signDistance(const geom::Plane &p, const geom::real3 &point)
{
	// p.a * point.x + p.b * point.y + p.c * point.z + p.d;
	geom::real t = fma_auto(p.c, point.z, p.d); // p.c*point.z + p.d
	t = fma_auto(p.b, point.y, t); // p.b*point.y + t
	return fma_auto(p.a, point.x, t); // p.a*point.x + t
}

__device__ void solve4x4(const geom::real A[16], const geom::real B[4], geom::real X[4])
{
	// Simple Gauss elimination for small 4x4 system

	geom::real M[4][5]; // 4x4 matrix + RHS
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
		geom::real pivot = M[i][i];
#pragma unroll
		for (int j = i; j <= 4; ++j) {
			M[i][j] /= pivot;
		}
#pragma unroll
		for (int k = i + 1; k < 4; ++k) {
			geom::real factor = M[k][i];
#pragma unroll
			for (int j = i; j <= 4; ++j) {
				M[k][j] = fma_auto(-factor, M[i][j], M[k][j]); // M[k][j] -= factor * M[i][j];
			}
		}
	}

	// Back substitution
#pragma unroll
	for (int i = 3; i >= 0; --i) {
		X[i] = M[i][4];
#pragma unroll
		for (int j = i + 1; j < 4; ++j) {
			X[i] = fma_auto(-M[i][j], X[j], X[i]); // X[i] -= M[i][j] * X[j];
		}
	}
}

// clang-format off
__device__ geom::Plane getPolypointPlaneCUDA(
	const geom::Plane &plane, const geom::real3 *origBasises, const geom::real3 *resBasises, int basisCount)
{
	geom::real acc[14] = {0.0};
	geom::real rhs[4]  = {0.0};

	for (int i = 0; i < basisCount; ++i) {
		const geom::real3 &orig_basis_p = origBasises[i];
		const geom::real3 &res_basis_p  = resBasises[i];

		const geom::real gamma		   = signDistance(plane, orig_basis_p);
		const geom::real gamma_inv	   = __drcp_rn(gamma);
		const geom::real gamma_sq_inv	= gamma_inv * gamma_inv;

		const geom::real x = res_basis_p.x;
		const geom::real y = res_basis_p.y;
		const geom::real z = res_basis_p.z;

		// Accumulate with FMA
		acc[0] = fma_auto(x * x, gamma_sq_inv, acc[0]);		// a1 = x * x / gamma_squared;
		acc[1] = fma_auto(x * y, gamma_sq_inv, acc[1]);		// b1 = x * y / gamma_squared;
		acc[2] = fma_auto(x * z, gamma_sq_inv, acc[2]);		// c1 = x * z / gamma_squared;
		acc[3] = fma_auto(x,	 gamma_sq_inv, acc[3]);		// d1 = x / gamma_squared;

		acc[4] = fma_auto(y * y, gamma_sq_inv, acc[4]);		// b2 = y * y / gamma_squared;
		acc[5] = fma_auto(y * z, gamma_sq_inv, acc[5]);		// c2 = y * z / gamma_squared;
		acc[6] = fma_auto(y,	 gamma_sq_inv, acc[6]);		// d2 = y / gamma_squared;

		acc[7] = fma_auto(z * z, gamma_sq_inv, acc[7]);		// c3 = z * z / gamma_squared;
		acc[8] = fma_auto(z,	 gamma_sq_inv, acc[8]);		// d3 = z / gamma_squared;

		acc[9] = fma_auto(1,   gamma_sq_inv, acc[9]);		// d4 = 1 / gamma_squared;

		// RHS
		rhs[0] = fma_auto(x, gamma_inv, rhs[0]);			// r1 = x / gamma;
		rhs[1] = fma_auto(y, gamma_inv, rhs[1]);			// r2 = y / gamma;
		rhs[2] = fma_auto(z, gamma_inv, rhs[2]);			// r3 = z / gamma;
		rhs[3] = fma_auto(1, gamma_inv, rhs[3]);			// r4 = 1 / gamma;
	}

	const geom::real A[16] = {
	    acc[0] + consts::reg_term, acc[1], acc[2], acc[3],
	    acc[1], acc[4] + consts::reg_term, acc[5], acc[6],
	    acc[2], acc[5], acc[7] + consts::reg_term, acc[8],
	    acc[3], acc[6], acc[8], acc[9] + consts::reg_term,
	};

	geom::real X[4];
	solve4x4(A, rhs, X);

	return geom::Plane{plane.id, X[0], X[1], X[2], X[3]};
}

// clang-format on

// Deform each plain on CUDA thread
__global__ void deformPlanesPolypointKernel(
    const geom::Plane *inPlanes,
    geom::Plane *outPlanes,
    const geom::real3 *origBasises,
    const geom::real3 *resBasises,
    int basisCount,
    int planeCount)
{
	extern __shared__ geom::real3 sharedMemory[]; // dynamic shared memory
	geom::real3 *sharedOrigBasises = sharedMemory;
	geom::real3 *sharedResBasises = sharedMemory + basisCount;

	// Only one thread per block loads origBasises and resBasises into shared memory
	for (int i = threadIdx.x; i < basisCount; i += blockDim.x) {
		sharedOrigBasises[i] = origBasises[i];
		sharedResBasises[i] = resBasises[i];
	}
	__syncthreads(); // wait for all threads to finish copying

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= planeCount) {
		return;
	}

	outPlanes[idx] = getPolypointPlaneCUDA(
	    inPlanes[idx], sharedOrigBasises, sharedResBasises, basisCount);
}

void checkErrorCUDA(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
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
	thrust::device_vector<geom::real3> d_origBasises = origBasises;
	thrust::device_vector<geom::real3> d_resBasises = resBasises;

	// Launch kernel
	int blockSize = 512;
	int gridSize = (planeCount + blockSize - 1) / blockSize;
	size_t sharedMemSize = 2 * basisCount * sizeof(geom::real3);
	std::cout << "CUDA SM config: blockSize: " << blockSize << ", gridSize: " << gridSize
	          << ", sharedMemSize: " << sharedMemSize << std::endl;

	deformPlanesPolypointKernel<<<gridSize, blockSize, sharedMemSize>>>(
	    thrust::raw_pointer_cast(d_inPlanes.data()),
	    thrust::raw_pointer_cast(d_outPlanes.data()),
	    thrust::raw_pointer_cast(d_origBasises.data()),
	    thrust::raw_pointer_cast(d_resBasises.data()),
	    basisCount,
	    planeCount);

	checkErrorCUDA("Kernel launch failed");

	cudaDeviceSynchronize();

	checkErrorCUDA("Kernel execution failed");

	// Download result from GPU
	outPlanes.resize(planeCount);
	thrust::copy(d_outPlanes.begin(), d_outPlanes.end(), outPlanes.begin());
}

} // namespace gpu
