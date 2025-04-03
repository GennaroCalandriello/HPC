// mainfluids_visualization_snapshots.cu
#include "functions.h"
#include "obstacles.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

// Viridis colormap etc. skipped for brevity

// Global variables
Vector2f C;
Vector2f F;
float global_decay_rate = DECAY_RATE;

// Snapshot storage
std::vector<std::vector<float>> snapshots;

void saveSnapshot(Vector2f* dev_u, int dim) {
    std::vector<float> snapshot(2 * dim * dim);
    cudaMemcpy(snapshot.data(), dev_u, 2 * dim * dim * sizeof(float), cudaMemcpyDeviceToHost);
    snapshots.push_back(snapshot);
}

int main(int argc, char** argv) {
    float betabouyancy = BETA_BOUYANCY;
    float gravity = -9.81f;
    int framecount = 0;

    // Allocate and initialize obstacle field
    int* obstacleField = (int*)malloc(dim * dim * sizeof(int));
    initializeObstacle(obstacleField, dim, obstacleCenterX, obstacleCenterY, obstacleRadius);

    int* dev_obstacleField;
    cudaMalloc((void**)&dev_obstacleField, dim * dim * sizeof(int));
    cudaMemcpy(dev_obstacleField, obstacleField, dim * dim * sizeof(int), cudaMemcpyHostToDevice);

    Vector2f* u = (Vector2f*)malloc(dim * dim * sizeof(Vector2f));
    float* p = (float*)malloc(dim * dim * sizeof(float));
    float* c = (float*)malloc(dim * dim * sizeof(float));

    for (unsigned i = 0; i < dim * dim; i++) {
        u[i] = Vector2f::Zero(); p[i] = 0.0f; c[i] = 0.0f;
    }

    Vector2f* dev_u; float* dev_p; float* dev_c;
    cudaMalloc((void**)&dev_c, dim * dim * sizeof(float));
    cudaMalloc((void**)&dev_u, dim * dim * sizeof(Vector2f));
    cudaMalloc((void**)&dev_p, dim * dim * sizeof(float));
    cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, dim * dim * sizeof(float), cudaMemcpyHostToDevice);

    Vector3f* colorField = (Vector3f*)malloc(dim * dim * sizeof(Vector3f));
    Vector3f* dev_colorField;
    cudaMalloc((void**)&dev_colorField, dim * dim * sizeof(Vector3f));

    // GLFW initialization and rendering setup skipped for brevity

    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX, (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);

    while (!glfwWindowShouldClose(window)) {
        float time = glfwGetTime();
        float c_ambient = C_AMBIENT;

        if (PERIODIC_FORCE == 1) F = Vector2f(magnitude * sin(time), 0.0f);
        C = Vector2f(dim / 2.0f + 50.0f * sinf(time), dim / 2.0f);

        NSkernel<<<blocks, threads>>>(dev_u, dev_p, dev_c, dev_obstacleField, c_ambient, gravity, betabouyancy,
                                      rdx, viscosity, C, F, timestep, r, dim);
        cudaDeviceSynchronize();

        // Save snapshots every N frames
        if (framecount % SNAPSHOT_INTERVAL == 0 && snapshots.size() < MAX_SNAPSHOTS) {
            saveSnapshot(dev_u, dim);
        }

        framecount++;

        // Rendering (same as before, skipped here)
    }

    // === POD COMPUTATION ===
    using namespace Eigen;
    if (!snapshots.empty()) {
        const int N = 2 * dim * dim;
        const int M = snapshots.size();
        MatrixXd S(N, M);

        for (int i = 0; i < M; i++)
            S.col(i) = Map<VectorXd>(snapshots[i].data(), N);

        VectorXd mean = S.rowwise().mean();
        MatrixXd S_centered = S.colwise() - mean;

        JacobiSVD<MatrixXd> svd(S_centered, ComputeThinU);
        int r = 10;
        MatrixXd POD_basis = svd.matrixU().leftCols(r);

        std::cout << "POD basis shape: " << POD_basis.rows() << " x " << POD_basis.cols() << "\n";
        std::cout << "Energy captured: " << svd.singularValues().head(r).array().square().sum()
                  / svd.singularValues().array().square().sum() << "\n";
    }

    // Cleanup (same as before)
    free(u); free(p); free(c); free(obstacleField); free(colorField);
    cudaFree(dev_u); cudaFree(dev_p); cudaFree(dev_c);
    cudaFree(dev_obstacleField); cudaFree(dev_colorField);
    glfwTerminate();

    return 0;
}
