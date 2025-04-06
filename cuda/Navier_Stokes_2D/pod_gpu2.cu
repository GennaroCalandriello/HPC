#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cusolverDn.h>
#include "scalar.h"  // Defines DIM and MAX_FRAMES

using namespace std;

// Global flags to control saving behavior
bool savesigma   = true;  // Save singular values if true
int  pod_to_save = 10;    // Number of POD modes (columns) to save

// Macros to check CUDA and cuSOLVER errors
#define CHECK_CUDA(call) {                                  \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s at line %d: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}

#define CHECK_CUSOLVER(call) {                                \
    cusolverStatus_t err = (call);                            \
    if (err != CUSOLVER_STATUS_SUCCESS) {                     \
        fprintf(stderr, "cuSolver error %s at line %d: %d\n",  \
                __FILE__, __LINE__, err);                     \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}

//------------------------------------------------------------------------------
// Kernel: Matrix multiplication in column-major order.
// Multiplies A (dimensions A_rows x A_cols) and B (A_cols x B_cols) to produce C (A_rows x B_cols).
// Element (i, j) is stored at index (i + j*A_rows).
//------------------------------------------------------------------------------
__global__ void matMulKernel(const double* A, const double* B, double* C,
                             int A_rows, int A_cols, int B_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index in C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in C

    if (row < A_rows && col < B_cols) {
        double sum = 0.0;
        for (int k = 0; k < A_cols; k++) {
            double a = A[row + k * A_rows];   // A(row,k)
            double b = B[k + col * A_cols];     // B(k,col)
            sum += a * b;
        }
        C[row + col * A_rows] = sum;
    }
}

//------------------------------------------------------------------------------
// Function: loadCsv
// Reads a CSV file (assumed to be stored in row-major order, with each line representing a snapshot)
// and returns its transpose as a vector<double> in column-major order.
// If the CSV has R rows and C columns, then we set:
//   N = C  (spatial dimension, expected to be 2*DIM*DIM)
//   M = R  (number of snapshots, expected to be MAX_FRAMES)
//------------------------------------------------------------------------------
std::vector<double> loadCsv(const std::string& filename, int& N, int& M)
{
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    // Read the CSV file into a 2D vector (each inner vector is a row)
    vector<vector<double>> rows;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        vector<double> row;
        while (getline(ss, token, ',')) {
            row.push_back(stod(token));
        }
        rows.push_back(row);
    }
    file.close();
    
    // Original CSV dimensions:
    int R = rows.size();          // snapshots
    if (R == 0) {
        cerr << "No data in file: " << filename << endl;
        exit(1);
    }
    int C = rows[0].size();       // spatial data per snapshot
    for (int i = 1; i < R; i++) {
        if (rows[i].size() != C) {
            cerr << "Inconsistent row lengths in file: " << filename << endl;
            exit(1);
        }
    }
    
    // We want the transpose of the CSV.
    // That is, the output matrix will have dimensions (C x R) stored in column-major order.
    N = C; // spatial dimension = 2*DIM*DIM (expected)
    M = R; // number of snapshots = MAX_FRAMES (expected)
    vector<double> matrix(N * M, 0.0);
    // For each element in the original CSV at (i, j) (i: row, j: col),
    // place it at (j, i) in the output. In column-major order,
    // element (row, col) is stored at index (row + col*N).
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            matrix[j + i * N] = rows[i][j];
        }
    }
    return matrix;
}

//------------------------------------------------------------------------------
// Function: centerData
// Centers the data row-wise. For each spatial coordinate (each row of the (N x M) matrix),
// subtracts the mean across all snapshots.
//------------------------------------------------------------------------------
void centerData(std::vector<double>& data, int N, int M)
{
    for (int r = 0; r < N; r++) {
        double sum = 0.0;
        for (int c = 0; c < M; c++) {
            sum += data[c * N + r];
        }
        double mean = sum / M;
        for (int c = 0; c < M; c++) {
            data[c * N + r] -= mean;
        }
    }
}

//------------------------------------------------------------------------------
// Function: svdCudas
// Performs full SVD on the input matrix h_A (of dimensions Nrows x Ncols, stored in column-major order)
// using cuSOLVER. It computes: h_A = U * S * V^T.
// h_S receives the singular values (length = min(Nrows, Ncols)).
// h_U receives the left singular vectors (Nrows x Nrows).
// h_V receives the right singular vectors (Ncols x Ncols). Note: V returned here is V (not V^T).
//------------------------------------------------------------------------------
void svdCudas(const double* h_A, int Nrows, int Ncols,
              double* h_S, double* h_U, double* h_V)
{
    int mn = (Nrows < Ncols ? Nrows : Ncols);
    
    double* d_A = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, Nrows * Ncols * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));
    
    double *d_S = nullptr, *d_U = nullptr, *d_V = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_S, mn * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_U, Nrows * Nrows * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_V, Ncols * Ncols * sizeof(double)));
    
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(handle, Nrows, Ncols, &lwork));
    double* d_work = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_work, lwork * sizeof(double)));
    
    int* devInfo = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));
    
    char jobu  = 'A';  // Compute full U (Nrows x Nrows)
    char jobvt = 'A';  // Compute full V^T (Ncols x Ncols)
    
    CHECK_CUSOLVER(cusolverDnDgesvd(handle, jobu, jobvt, Nrows, Ncols, d_A, Nrows,
                                    d_S, d_U, Nrows, d_V, Ncols, d_work, lwork, nullptr, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    int info_cpu = 0;
    CHECK_CUDA(cudaMemcpy(&info_cpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_cpu != 0) {
        cerr << "SVD failed, info = " << info_cpu << endl;
        exit(1);
    }
    
    memcpy(h_S, d_S, mn * sizeof(double));
    memcpy(h_U, d_U, Nrows * Nrows * sizeof(double));
    memcpy(h_V, d_V, Ncols * Ncols * sizeof(double));
    
    cudaFree(d_A);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);
}

//------------------------------------------------------------------------------
// Function: printMatrix
// Prints up to 10 rows and 10 columns of a matrix stored in column-major order.
//------------------------------------------------------------------------------
void printMatrix(const std::vector<double>& mat, int N, int M)
{
    for (int r = 0; r < min(N, 10); r++) {
        for (int c = 0; c < min(M, 10); c++) {
            double val = mat[c * N + r];
            cout << val << " ";
        }
        cout << "\n";
    }
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Expected input: "snapshots.csv"
    // The CSV file is assumed to have MAX_FRAMES rows (snapshots) and
    // 2*DIM*DIM columns (spatial data) in row-major order.
    // We load it and then transpose it to get a matrix in column-major order with dimensions:
    //    N = 2*DIM*DIM   (spatial dimension)
    //    M = MAX_FRAMES  (number of snapshots)
    string filename = "snapshots.csv";
    
    // Set expected dimensions for the original CSV:
    int orig_rows = MAX_FRAMES;         // snapshots
    int orig_cols = 2 * DIM * DIM;        // spatial data per snapshot
    
    int N, M;
    // After transposition: N = orig_cols, M = orig_rows.
    vector<double> data = loadCsv(filename, N, M);
    // cout << "Loaded data from " << filename << ": (N = " << N << ", M = " << M << ")\n";
    
    // (Optional) Center the data row-wise.
    // centerData(data, N, M);
    
    // Preallocate SVD output vectors.
    // For full SVD: S has length mn = min(N, M), U is (N x N), Vt is (M x M).
    int mn = min(N, M);  // With expected dimensions: N = 2*DIM*DIM, M = MAX_FRAMES, so mn = MAX_FRAMES.
    vector<double> S(mn, 0.0);
    vector<double> U(N * N, 0.0);
    vector<double> Vt(M * M, 0.0);
    
    // Perform SVD on the data matrix.
    svdCudas(data.data(), N, M, S.data(), U.data(), Vt.data());
    cout << "SVD done. First 5 singular values:\n";
    for (int i = 0; i < min((int)S.size(), 5); i++){
        cout << S[i] << " ";
    }
    cout << "\n";
    
    // Build the Σ (Sigma) matrix as an (N x M) diagonal matrix.
    // In column-major order, the diagonal element (i,i) is stored at index: i + i * N.
    vector<double> SigmaMat(N * M, 0.0);
    for (int i = 0; i < mn; i++){
        SigmaMat[i + i * N] = S[i];
    }
    
    // Compute two products:
    // 1. C = SigmaMat * Vt, where SigmaMat is (N x M) and Vt is (M x M) => C is (N x M)
    // 2. Snp = U * C, where U is (N x N) and C is (N x M) => Snp is (N x M)
    vector<double> C(N * M, 0.0), Snp(N * M, 0.0);
    
    double *d_Sigma = nullptr, *d_Vt = nullptr, *d_C = nullptr, *d_Snp = nullptr, *d_U = nullptr;
    
    CHECK_CUDA(cudaMalloc((void**)&d_U,     N * N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Sigma, N * M * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Vt,    M * M * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_C,     N * M * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Snp,   N * M * sizeof(double)));
    
    // Copy SigmaMat, Vt, and U to device memory.
    CHECK_CUDA(cudaMemcpy(d_Sigma, SigmaMat.data(), N * M * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Vt,    Vt.data(),       M * M * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_U,     U.data(),        N * N * sizeof(double), cudaMemcpyHostToDevice));
    // Zero output matrices.
    CHECK_CUDA(cudaMemset(d_C, 0, N * M * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_Snp, 0, N * M * sizeof(double)));
    
    // Define CUDA grid and block dimensions.
    dim3 block(16, 16);
    dim3 grid1((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    // Multiply: C = SigmaMat * Vt.
    // SigmaMat: (N x M), Vt: (M x M) => C: (N x M)
    matMulKernel<<<grid1, block>>>(d_Sigma, d_Vt, d_C, N, M, M);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Multiply: Snp = U * C.
    // U: (N x N), C: (N x M) => Snp: (N x M)
    dim3 grid2((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matMulKernel<<<grid2, block>>>(d_U, d_C, d_Snp, N, N, M);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back to host.
    CHECK_CUDA(cudaMemcpy(C.data(), d_C, N * M * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(Snp.data(), d_Snp, N * M * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Free device memory.
    CHECK_CUDA(cudaFree(d_Sigma));
    CHECK_CUDA(cudaFree(d_Vt));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_Snp));
    
    // Optionally, save singular values to "output.txt"
    if (savesigma) {
        ofstream outFile("output.txt");
        for (const auto &val : S) {
            outFile << val << "\n";
        }
        outFile.close();
    }
    
    // Save the first 'pod_to_save' modes (i.e., the first pod_to_save columns of Snp) to "pod_modes.txt".
    // Snp is stored in column-major order, so column j is stored at indices [j*N, (j+1)*N).
    ofstream modesOut("pod_modes.txt");
    for (int i = 0; i < N; i++){
        for (int j = 0; j < pod_to_save; j++){
            double val = Snp[i + j * N];
            modesOut << val;
            if (j < pod_to_save - 1)
                modesOut << ",";
        }
        modesOut << "\n";
    }
    modesOut.close();
    
    cout << "Done. Check 'pod_modes.txt' and 'output.txt'.\n";
    return 0;
}
