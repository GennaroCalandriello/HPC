#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <math.h>
#define CHECK_CUDA(call) {                                  \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}

// This function assumes that the following pointers are already allocated on device memory:
//   d_snapshots: pointer to snapshots (size: d x Ns)
//   d_Q: pointer to eigenvector matrix Q (size: Ns x r)
//   d_lambda: pointer to eigenvalues (array of length r)
// It will allocate output memory for POD modes (size: d x r), launch the kernel,
// and copy the result back into a host std::vector.

#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute POD modes.
// Inputs:
//   snapshots: device pointer to snapshots matrix (size: d x Ns) in column-major order.
//              That is, snapshots[ l + j*d ] is the l-th coordinate of snapshot j.
//   Q:         device pointer to the eigenvector matrix (size: Ns x r) in column-major order.
//              So, Q[ j + i*Ns ] is the weight for snapshot j in POD mode i.
//   lambda:    device pointer to the eigenvalues (array of length r).
//   d:         full spatial dimension (length of each snapshot).
//   Ns:        number of snapshots (columns in U_s).
//   r:         number of POD modes to compute.
// Output:
//   pod_modes: device pointer to output POD modes, size: d x r, stored in column-major order.
__global__ void computePODModesKernel(const double* snapshots,
                                      const double* Q,
                                      const double* lambda,
                                      double* pod_modes,
                                      int d, int Ns, int r)
{
    // Each thread computes one element of a POD mode.
    // mode: index of the POD mode (0 <= mode < r)
    // spatial: index of the spatial coordinate (0 <= spatial < d)
    int mode = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = blockIdx.y * blockDim.y + threadIdx.y;

    if (mode < r && spatial < d) {
        double sum = 0.0;
        // Sum over all snapshots.
        for (int j = 0; j < Ns; j++) {
            // u_j(l) is the element at index: spatial + j*d
            double u_val = snapshots[ spatial + j*d ];
            // Q_{ij} is at index: j + mode*Ns.
            double weight = Q[ j + mode * Ns ];
            sum += u_val * weight;
        }
        // Normalize by sqrt(lambda_i), provided lambda[mode] is not zero.
        double norm_factor = (lambda[mode] > 1e-12) ? (1.0 / sqrt(lambda[mode])) : 0.0;
        // Output: element of POD mode i at spatial location
        pod_modes[ spatial + mode * d ] = sum * norm_factor;
    }
}

std::vector<double> computePODModes(const double* d_snapshots,
                                    const double* d_Q,
                                    const double* d_lambda,
                                    int d, int Ns, int r)
{
    // Allocate memory for output on device.
    double* d_pod_modes = nullptr;
    size_t size_pod = d * r * sizeof(double);
    CHECK_CUDA(cudaMalloc((void**)&d_pod_modes, size_pod));
    CHECK_CUDA(cudaMemset(d_pod_modes, 0, size_pod));

    // Configure grid/block dimensions.
    dim3 block(16, 16);
    dim3 grid((r + block.x - 1) / block.x, (d + block.y - 1) / block.y);

    // Launch the kernel.
    computePODModesKernel<<<grid, block>>>(d_snapshots, d_Q, d_lambda, d_pod_modes, d, Ns, r);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host.
    std::vector<double> h_pod_modes(d * r);
    CHECK_CUDA(cudaMemcpy(h_pod_modes.data(), d_pod_modes, size_pod, cudaMemcpyDeviceToHost));

    // Free device memory.
    CHECK_CUDA(cudaFree(d_pod_modes));

    return h_pod_modes;
}
