// fluids.cu

#include "const.h"
#include <fstream> // Add this line

// Global variables
Vector2f C;
Vector2f F;
float global_decay_rate = DECAY_RATE;

// First step: apply the external force field to the data
__device__ void force(Vector2f x, Vector2f* field, Vector2f C, Vector2f F, float timestep, float r, unsigned dim) {
    float xC[2] = { x.x - C.x, x.y - C.y };
    float exp_val = (xC[0] * xC[0] + xC[1] * xC[1]) / r;
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);
    float factor = timestep * expf(-exp_val) * 0.001f;
    Vector2f temp = F * factor;
    if (i >= 0 && i < dim && j >= 0 && j < dim) {
        field[IND(i, j, dim)] += temp;
    }
}

// Bilinear interpolation
__device__ Vector2f bilerp(Vector2f pos, Vector2f* field, unsigned dim) {
    int i = static_cast<int>(pos.x);
    int j = static_cast<int>(pos.y);
    float dx = pos.x - i;
    float dy = pos.y - j;

    if (i < 0 || i >= dim - 1 || j < 0 || j >= dim - 1) {
        // Out of bounds
        return Vector2f::Zero();
    }
    else {
        // Perform bilinear interpolation
        Vector2f f00 = field[IND(i, j, dim)];
        Vector2f f10 = field[IND(i + 1, j, dim)];
        Vector2f f01 = field[IND(i, j + 1, dim)];
        Vector2f f11 = field[IND(i + 1, j + 1, dim)];

        Vector2f f0 = f00 * (1.0f - dx) + f10 * dx;
        Vector2f f1 = f01 * (1.0f - dx) + f11 * dx;

        return f0 * (1.0f - dy) + f1 * dy;
    }
}

// Second step: advect the data through method of characteristics
__device__ void advect(Vector2f x, Vector2f* field, Vector2f* velfield, float timestep, float rdx, unsigned dim) {
    int idx = IND(static_cast<int>(x.x), static_cast<int>(x.y), dim);
    Vector2f velocity = velfield[idx];
    Vector2f pos = x - velocity * (timestep * rdx);
    field[idx] = bilerp(pos, field, dim);
}

// Third step: diffuse the data
template <typename T>
__device__ void jacobi(Vector2f x, T* field, float alpha, float beta, T b, T zero, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    T f_left = (i > 0) ? field[IND(i - 1, j, dim)] : zero;
    T f_right = (i < dim - 1) ? field[IND(i + 1, j, dim)] : zero;
    T f_down = (j > 0) ? field[IND(i, j - 1, dim)] : zero;
    T f_up = (j < dim - 1) ? field[IND(i, j + 1, dim)] : zero;
    T ab = alpha * b;

    field[IND(i, j, dim)] = (f_left + f_right + f_down + f_up + ab) / beta;
}

// Compute divergence
__device__ float divergence(Vector2f x, Vector2f* from, float halfrdx, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return 0.0f;

    Vector2f wL = (i > 0) ? from[IND(i - 1, j, dim)] : Vector2f::Zero();
    Vector2f wR = (i < dim - 1) ? from[IND(i + 1, j, dim)] : Vector2f::Zero();
    Vector2f wB = (j > 0) ? from[IND(i, j - 1, dim)] : Vector2f::Zero();
    Vector2f wT = (j < dim - 1) ? from[IND(i, j + 1, dim)] : Vector2f::Zero();

    float div = halfrdx * ((wR.x - wL.x) + (wT.y - wB.y));
    return div;
}

// Obtain the approximate gradient of a scalar field
__device__ Vector2f gradient(Vector2f x, float* p, float halfrdx, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return Vector2f::Zero();

    float pL = (i > 0) ? p[IND(i - 1, j, dim)] : 0.0f;
    float pR = (i < dim - 1) ? p[IND(i + 1, j, dim)] : 0.0f;
    float pB = (j > 0) ? p[IND(i, j - 1, dim)] : 0.0f;
    float pT = (j < dim - 1) ? p[IND(i, j + 1, dim)] : 0.0f;

    Vector2f grad;
    grad.x = halfrdx * (pR - pL);
    grad.y = halfrdx * (pT - pB);
    return grad;
}

// Navier-Stokes kernel
__global__ void NSkernel(Vector2f* u, float* p, float rdx, float viscosity, Vector2f C, Vector2f F, float timestep, float r, unsigned dim) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dim || j >= dim)
        return;

    Vector2f x(static_cast<float>(i), static_cast<float>(j));

    // Advection
    advect(x, u, u, timestep, rdx, dim);
    __syncthreads();

    // Diffusion
    float alpha = rdx * rdx / (viscosity * timestep);
    float beta = 4.0f + alpha;
    jacobi<Vector2f>(x, u, alpha, beta, u[IND(i, j, dim)], Vector2f::Zero(), dim);
    __syncthreads();

    // Force application
    force(x, u, C, F, timestep, r, dim);
    __syncthreads();

    // Pressure calculation
    alpha = -rdx * rdx;
    beta = 4.0f;
    float div = divergence(x, u, 0.5f * rdx, dim);
    jacobi<float>(x, p, alpha, beta, div, 0.0f, dim);
    __syncthreads();

    // Pressure gradient subtraction
    Vector2f grad_p = gradient(x, p, 0.5f * rdx, dim);
    u[IND(i, j, dim)] -= grad_p;
    __syncthreads();

    // Optional: Print state for debugging
    /*
    if (i == dim / 2 && j == dim / 2) {
        Vector2f vel = u[IND(i, j, dim)];
        printf("u[%d, %d] = (%f, %f)\n", i, j, vel.x, vel.y);
    }
    */
}

// Decay the convection force F
void decayForce() {
    float nx = F.x - global_decay_rate;
    float ny = F.y - global_decay_rate;
    nx = (nx > 0) ? nx : 0.0f;
    ny = (ny > 0) ? ny : 0.0f;
    F = Vector2f(nx, ny);
}

// Function to write simulation data to a file
void writeSimulationData(Vector2f* velocityField, unsigned dim, int timestep) {
    std::ofstream outfile("fluid_simulation_output.txt", std::ios::app);
    outfile << "Timestep: " << timestep << "\n";
    for (unsigned j = 0; j < dim; ++j) {
        for (unsigned i = 0; i < dim; ++i) {
            Vector2f vel = velocityField[IND(i, j, dim)];
            outfile << "(" << vel.x << ", " << vel.y << ") ";
        }
        outfile << "\n";
    }
    outfile << "\n";
    outfile.close();
}

int main(int argc, char** argv) {
    // Simulation parameters
    float timestep = TIMESTEP;
    unsigned dim = DIM;
    float rdx = static_cast<float>(RES) / dim;
    float viscosity = VISCOSITY;
    global_decay_rate = DECAY_RATE;
    float r = RADIUS;

    // Force parameters
    C = Vector2f(static_cast<float>(dim) / 2.0f, static_cast<float>(dim) / 2.0f); // Center of the domain
    F = Vector2f(10.0f, 10.0f); // Initial force

    // Velocity vector field and pressure scalar field
    Vector2f* u = (Vector2f*)malloc(dim * dim * sizeof(Vector2f));
    float* p = (float*)malloc(dim * dim * sizeof(float));

    // Initialize host fields
    for (unsigned i = 0; i < dim * dim; i++) {
        u[i] = Vector2f::Zero();
        p[i] = 0.0f;
    }
    // initialize host gaussian field
    for (unsigned j = 0; j < dim; ++j) {
        for (unsigned i = 0; i < dim; ++i) {
            float x = static_cast<float>(i) - dim / 2.0f;
            float y = static_cast<float>(j) - dim / 2.0f;
            float exp_val = (x * x + y * y) / (r * r);
            u[IND(i, j, dim)] = Vector2f(10.0f * expf(-exp_val), 0.0f);
        }
    }
    // Device memory allocation
    Vector2f* dev_u;
    float* dev_p;
    cudaMalloc((void**)&dev_u, dim * dim * sizeof(Vector2f));
    cudaMalloc((void**)&dev_p, dim * dim * sizeof(float));

    // Copy initial values to device
    cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, dim * dim * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA grid and block dimensions
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX, (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);

    // Run simulation for a certain number of timesteps
    int num_timesteps = NUM_TIMESTEPS;
    for (int t = 0; t < num_timesteps; t++) {
        if (t % 10 == 0) {
            printf("Timestep %d\n", t);
        }
        // Update force parameters over time to simulate interaction
        // C = Vector2f(dim / 2.0f + 5.0f * sinf(timestep * 0.8f), dim / 2.0f + 5.0f * cosf(timestep * 0.8f));
        F = Vector2f(10.0f * sinf(t), 10.0f * cosf(t));

        // Launch the Navier-Stokes kernel
        NSkernel<<<blocks, threads>>>(dev_u, dev_p, rdx, viscosity, C, F, timestep, r, dim);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
            return 1;
        }

        cudaDeviceSynchronize();

        // Copy the velocity field back to host
        cudaMemcpy(u, dev_u, dim * dim * sizeof(Vector2f), cudaMemcpyDeviceToHost);

        // Write simulation data to file
        writeSimulationData(u, dim, t);

        // Decay the force
        decayForce();
    }

    // Free memory
    free(u);
    free(p);
    cudaFree(dev_u);
    cudaFree(dev_p);

    return 0;
}
