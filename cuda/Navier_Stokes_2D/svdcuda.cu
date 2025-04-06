// svdcuda.cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <math.h>
#include <algorithm>  // for std::min

#include <cusolverDn.h>
#include <cuda_runtime_api.h>

#include "Utilities.cuh"  // Provides gpuErrchk, cusolveSafeCall, and iDivUp

/**
 * Compute the Singular Value Decomposition (SVD) of a matrix using cuSOLVER.
 *
 * The input matrix h_A is assumed to be stored in column-major order with dimensions:
 *     Nrows x Ncols   (with Nrows >= Ncols)
 *
 * The SVD computed is: h_A = U * S * V^T, where:
 *   - U is an (Nrows x Nrows) matrix (left singular vectors)
 *   - S is a vector of length min(Nrows, Ncols) (singular values)
 *   - V is an (Ncols x Ncols) matrix (right singular vectors)
 *
 * @param h_A   Pointer to the input host matrix (size: Nrows * Ncols)
 * @param Nrows Number of rows in the matrix.
 * @param Ncols Number of columns in the matrix.
 * @param h_S   Pointer to the output array for singular values (size: min(Nrows, Ncols)).
 * @param h_U   Pointer to the output array for U (size: Nrows * Nrows).
 * @param h_V   Pointer to the output array for V (size: Ncols * Ncols).
 */
void svdCudas(const double* h_A, int Nrows, int Ncols,
             double* h_S, double* h_U, double* h_V)
{
    // Ensure Nrows >= Ncols as required by gesvd.
    int minDim = std::min(Nrows, Ncols);

    // --- Create cuSOLVER handle.
    cusolverDnHandle_t solver_handle;
    cusolveSafeCall(cusolverDnCreate(&solver_handle));

    // --- Allocate device memory for matrix A and copy h_A into device memory.
    double* d_A = nullptr;
    gpuErrchk(cudaMalloc(&d_A, Nrows * Ncols * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));

    // --- Allocate device memory for S, U, and V.
    double *d_S = nullptr, *d_U = nullptr, *d_V = nullptr;
    gpuErrchk(cudaMalloc(&d_S, Nrows*Ncols * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_U, Nrows * Nrows * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_V, Ncols * Ncols * sizeof(double)));

    // --- Allocate device memory for info output.
    int* devInfo = nullptr;
    gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));

    // --- Query working space for SVD.
    int work_size = 0;
    cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handle, Nrows, Ncols, &work_size));
    double* d_work = nullptr;
    gpuErrchk(cudaMalloc(&d_work, work_size * sizeof(double)));

    // --- Execute SVD.
    // 'A' for jobu and jobvt indicates that all columns of U and V^T are computed.
    cusolveSafeCall(cusolverDnDgesvd(solver_handle, 'A', 'A',
                                     Nrows, Ncols, d_A, Nrows,
                                     d_S, d_U, Nrows, d_V, Ncols,
                                     d_work, work_size, NULL, devInfo));

    // --- Check execution status.
    int devInfo_h = 0;
    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0) {
        std::cerr << "SVD failed, devInfo = " << devInfo_h << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- Copy results from device to host.
    gpuErrchk(cudaMemcpy(h_S, d_S, minDim * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_U, d_U, Nrows * Nrows * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_V, d_V, Ncols * Ncols * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Free device memory and destroy the solver handle.
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_S));
    gpuErrchk(cudaFree(d_U));
    gpuErrchk(cudaFree(d_V));
    gpuErrchk(cudaFree(devInfo));
    gpuErrchk(cudaFree(d_work));
    cusolverDnDestroy(solver_handle);
}
