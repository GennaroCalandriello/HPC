#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "scalar.h"  // Defines DIM and MAX_FRAMES

using namespace std;
enum class Layout { RowMajor, ColMajor };

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



void loadCsv(const std::string& filename,
             std::vector<double>& matrix,
             int& rows,
             int& cols,
             Layout layout = Layout::RowMajor) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::vector<double>> tempData;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;

        while (std::getline(ss, val, ',')) {
            row.push_back(std::stod(val));
        }

        if (!tempData.empty() && row.size() != tempData[0].size())
            throw std::runtime_error("Inconsistent number of columns in CSV");

        tempData.push_back(row);
    }

    rows = tempData.size();
    cols = tempData[0].size();
    matrix.resize(rows * cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (layout == Layout::RowMajor)
                matrix[i * cols + j] = tempData[i][j];
            else // ColMajor
                matrix[j * rows + i] = tempData[i][j];
        }
    }
}

void printMatrix(const std::vector<double>& matrix, int rows, int cols, Layout layout = Layout::RowMajor) {
    int nrow = 100;
    int ncol =5;
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            double val;
            if (layout == Layout::RowMajor)
                val = matrix[i * cols + j];
            else // ColMajor
                val = matrix[j * rows + i];

            std::cout << val << "\t";
        }
        std::cout << "\n";
    }
}

void saveMatrix(const std::string &filename,
                   const std::vector<double> &mat,
                   int N, int M,
                   Layout order =Layout::RowMajor)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Write matrix elements to file.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double value = 0.0;
            if (order == Layout::ColMajor) {
                // In column-major order, element (i, j) is at index i + j*N.
                value = mat[i + j * N];
            } else {
                // In row-major order, element (i, j) is at index i*M + j.
                value = mat[i * M + j];
            }
            outFile << value;
            if (j < M - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }
    outFile.close();
}


void svdCudas(double* h_A, const int N, const int M,
              double* h_S, double* h_U, double* h_V)
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;

    double *d_A = NULL,
           *d_S = NULL,
           *d_U = NULL,
           *d_VT = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    double *d_rwork = NULL;
    double *d_W = NULL;  // W = S*VT

    int lwork = 0;
    int info_gpu = 0;
    const double h_one = 1;
    const double h_minus_one = -1;
    // step 1: create cusolverDn/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // step 2: copy A and B to device
    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * N * M);
    cudaStat2 = cudaMalloc((void**)&d_S, sizeof(double) * M);
    cudaStat3 = cudaMalloc((void**)&d_U, sizeof(double) * N * N);
    cudaStat4 = cudaMalloc((void**)&d_VT, sizeof(double) * M * M);
    cudaStat5 = cudaMalloc((void**)&devInfo, sizeof(int));
    cudaStat6 = cudaMalloc((void**)&d_W, sizeof(double) * N * M);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat6);

    cudaStat1 = cudaMemcpy(d_A, h_A, sizeof(double) * N * M, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    //step 3 query working space of SVD
    cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, N, M, &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1=cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    //step 4 compute svd
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    //ATTENZIONE QUA, cusolver_status = cusolverDnDgesvd (
        // cusolverH,
        // jobu,
        // jobvt,
        // N,
        // M,
        // d_A,
        // N,
        // d_S,
        // d_U,
        // N,  // ldu
        // d_VT,
        // N, // ldvt, GNOGNOGNOGNOGNOGNOGNO!!!
        // d_work,
        // lwork,
        // d_rwork,
        // devInfo);
    cusolver_status = cusolverDnDgesvd(cusolverH, jobu, jobvt, N, M,
                                       d_A, N, d_S, d_U, N, d_VT, M,
                                       d_work, lwork, d_rwork,
                                       devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    assert (cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(h_S, d_S, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(h_U, d_U, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(h_V, d_VT, sizeof(double) * M * M, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    //info GPU
    printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    printf("=====\n");

    printf("S = (matlab base-1)\n");
    printf("=====\n");

    printf("U = (matlab base-1)\n");
    printf("=====\n");

    printf("VT = (matlab base-1)\n");
    printf("=====\n");

    //free resources
    // free resources
    if (d_A    ) cudaFree(d_A);
    if (d_S    ) cudaFree(d_S);
    if (d_U    ) cudaFree(d_U);
    if (d_VT   ) cudaFree(d_VT);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
    if (d_W    ) cudaFree(d_W);

    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();

}


int main(){
    int N = 2*DIM*DIM;
    int M = MAX_FRAMES; // Number of frames (columns)
    string filename = "snapshots.csv";
    
    //    std::cout << "Row-major matrix:\n";
    // printMatrix(mat, rows, cols, Layout::RowMajor);
    vector<double> data;
    loadCsv(filename,data, N, M, Layout::RowMajor);
    // printMatrix(data, N, M, Layout::RowMajor);
    // cout << "------------------------\n";
    // printMatrix(data, N, M, Layout::ColMajor);
    int mn = std::min(N, M);
    vector<double> S(mn, 0.0);
    vector<double> U(N * N, 0.0);
    vector<double> Vt(M * M, 0.0);
    cout <<"Qua ngi arrivi mannaggiacristo e la madonna\n";
    //reconstruct the Sigma matrix
    


    // Perform SVD
    svdCudas(data.data(), N, M, S.data(), U.data(), Vt.data());
    vector<double> Sigma(N*M, 0.0);
    for (int i = 0; i < mn; ++i) {
        Sigma[i * N + i] = S[i];
        cout << S[i] << "\t";
    }
    //print each matrix
    // printMatrix(Sigma, N, M, Layout::RowMajor);
    // cout << "------------------------\n";
    // printMatrix(U, N, N, Layout::RowMajor);
    // cout << "------------------------\n";
    // printMatrix(Vt, M, M, Layout::RowMajor);
    // cout << "------------------------\n";
    //save
    saveMatrix("Sigma.txt", Sigma, N, M, Layout::RowMajor);
    saveMatrix("Vt.txt", Vt, M, M, Layout::RowMajor);
    saveMatrix("U.txt", U, N, N, Layout::RowMajor);

    return 0;
}