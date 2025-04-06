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

enum class Layout { RowMajor, ColMajor };

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

void svdCudas(const double* h_A, int Nrows, int Ncols,
              double* h_S, double* h_U, double* h_V)
{
    
    double* d_A = nullptr;
    cout<< "Numero1"<< endl;
    CHECK_CUDA(cudaMalloc((void**)&d_A, Nrows * Ncols * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice));
    
    double *d_S = nullptr, *d_U = nullptr, *d_V = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_S, Nrows*Ncols * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_U, Nrows * Nrows * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_V, Ncols * Ncols * sizeof(double)));
    cout<< "Numero2"<< endl;
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    cout<< "Numero3"<< endl;
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(handle, Ncols, Nrows, &lwork));
    cout<< "Numero1"<< endl;
    double* d_work = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_work, lwork * sizeof(double)));
    
    int* devInfo = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));
    
    char jobu  = 'A';  // Compute full U (Nrows x Nrows)
    char jobvt = 'A';  // Compute full V^T (Ncols x Ncols)
    
    CHECK_CUSOLVER(cusolverDnDgesvd(handle, jobu, jobvt, Nrows, Ncols, d_A, Nrows,
                                    d_S, d_U, Nrows, d_V, Ncols, d_work, lwork, nullptr, devInfo));
    cout<< "Numero7"<< endl;
    CHECK_CUDA(cudaDeviceSynchronize());
    
    int info_cpu = 0;
    CHECK_CUDA(cudaMemcpy(&info_cpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_cpu != 0) {
        cerr << "SVD failed, info = " << info_cpu << endl;
        exit(1);
    }
    CHECK_CUDA(cudaMemcpy(h_S, d_S, Nrows*Ncols * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_U, d_U, Nrows * Nrows * sizeof(double), cudaMemcpyDeviceToHost));   
    CHECK_CUDA(cudaMemcpy(h_V, d_V, Ncols * Ncols * sizeof(double), cudaMemcpyDeviceToHost));
    cout<< "Numero9"<< endl;
    cudaFree(d_A);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);
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
    vector<double> S(N*M, 0.0);
    vector<double> U(N * N, 0.0);
    vector<double> Vt(M * M, 0.0);
    cout <<"Qua ngi arrivi mannaggiacristo e la madonna\n";


    // Perform SVD
    svdCudas(data.data(), N, M, S.data(), U.data(), Vt.data());
    //print each matrix
    printMatrix(S, N, M, Layout::RowMajor);
    cout << "------------------------\n";
    printMatrix(U, N, N, Layout::RowMajor);
    cout << "------------------------\n";
    printMatrix(Vt, M, M, Layout::RowMajor);
    cout << "------------------------\n";


    return 0;
}