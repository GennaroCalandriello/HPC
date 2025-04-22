#include "scalar.h" // Defines DIM and MAX_FRAMES
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>

using namespace std;
enum class Layout { RowMajor, ColMajor };
bool diag =
    false; // Flag to control whether to diagonalize the covariance matrix

// Global flags to control saving behavior
bool savesigma = true; // Save singular values if true
int pod_to_save = 10;  // Number of POD modes (columns) to save

// Macros to check CUDA and cuSOLVER errors
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at line %d: %s\n", __FILE__, __LINE__,    \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_CUSOLVER(call)                                                   \
  {                                                                            \
    cusolverStatus_t err = (call);                                             \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "cuSolver error %s at line %d: %d\n", __FILE__,          \
              __LINE__, err);                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

//==================================================
// Kernel functions
//==================================================
__global__ void matMulKernel(const double *A, const double *B, double *C, int N,
                             int K, int M, Layout layout) {
  // Determine the row and column for this thread.
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < M) {
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
      double a, b;
      if (layout == Layout::RowMajor) {
        // For row-major: element (i,k) at index i*K + k; element (k,j) at index
        // k*M + j.
        a = A[row * K + k];
        b = B[k * M + col];
      } else {
        // For column-major: element (i,k) at index i + k*N; element (k,j) at
        // index k + j*K.
        a = A[row + k * N];
        b = B[k + col * K];
      }
      sum += a * b;
    }
    if (layout == Layout::RowMajor) {
      // For row-major: C(i,j) is at index i*M + j.
      C[row * M + col] = sum;
    } else {
      // For column-major: C(i,j) is at index i + j*N.
      C[row + col * N] = sum;
    }
  }
}

__global__ void computePODModesKernel(double *phi, double *lambda, int N,
                                      int M) {

  // phi = 1/(lambda*M)*phi;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      phi[i * N + j] = phi[i * N + j] / (lambda[i] * M);
    }
  }
}

//===================================================
// Host functions
//===================================================

vector<double> matmul(const std::vector<double> &A,
                      const std::vector<double> &B, int N, int K, int M,
                      Layout layout) {
  // Allocate device memory for A, B, and C.
  cout << "Allocating device memory\n";
  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_A, N * K * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K * M * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, N * M * sizeof(double)));
  cout << "Device memory allocated\n";
  // Copy host data to device.
  CHECK_CUDA(cudaMemcpy(d_A, A.data(), N * K * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * M * sizeof(double),
                        cudaMemcpyHostToDevice));
  cout << "Data copied to device\n";
  // Define grid and block dimensions.
  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  // Launch the kernel.
  cout << "pre kernel\n";
  matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N, K, M, layout);
  cout << "post kernel\n";
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result from device back to host.
  std::vector<double> C(N * M);
  CHECK_CUDA(cudaMemcpy(C.data(), d_C, N * M * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // Free device memory.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}

void computePODModes(const vector<double> &snap, const vector<double> &Q,
                     const vector<double> lambda, vector<double> &phi, int N,
                     int M) {
  // Allocate memory for output on device.
  double *d_s = nullptr, *d_Q = nullptr, *d_lambda = nullptr, *d_phi = nullptr;

  // Allocation
  CHECK_CUDA(cudaMalloc((void **)&d_s, sizeof(double) * N * M));
  CHECK_CUDA(cudaMalloc((void **)&d_Q, sizeof(double) * M * M));
  CHECK_CUDA(cudaMalloc((void **)&d_lambda, sizeof(double) * M));
  CHECK_CUDA(cudaMalloc((void **)&d_phi, sizeof(double) * N * M));

  // Memcpy
  CHECK_CUDA(cudaMemcpy(d_s, snap.data(), sizeof(double) * N * M,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_Q, Q.data(), sizeof(double) * M * M,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * M,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_phi, phi.data(), sizeof(double) * N * M,
                        cudaMemcpyHostToDevice));

  // Configure grid/block dimensions.
  dim3 block(BLOCKSIZEX, BLOCKSIZEY);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  // Launch the kernel.
  // computePODModesKernel<<<grid, block>>>(d_s, d_Q, d_lambda, d_phi, N, M);
  phi = matmul(snap, Q, N, M, N, Layout::ColMajor);
  cout << "Test-debuggggg\n";
  computePODModesKernel<<<grid, block>>>(d_phi, d_lambda, N, M);
  // vector<double> new_phi(N*M, 0.0);
  // CHECK_CUDA(cudaMemcpy(new_phi.data(), d_phi, sizeof(double)*N*M,
  // cudaMemcpyDeviceToHost)); Check for errors in kernel launch. Copy result
  // back to host.
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(phi.data(), d_phi, sizeof(double) * N * M,
                        cudaMemcpyDeviceToHost));
  // Free device memory.
  CHECK_CUDA(cudaFree(d_s));
  CHECK_CUDA(cudaFree(d_Q));
  CHECK_CUDA(cudaFree(d_lambda));
  CHECK_CUDA(cudaFree(d_phi));
}
void loadCsv(const std::string &filename, std::vector<double> &matrix,
             int &rows, int &cols, Layout layout = Layout::RowMajor) {
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

void printMatrix(const std::vector<double> &matrix, int rows, int cols,
                 Layout layout = Layout::RowMajor) {
  int nrow = 100;
  int ncol = 5;
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

void saveMatrix(const std::string &filename, const std::vector<double> &mat,
                int N, int M, Layout order = Layout::RowMajor) {
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

void svdCudas(double *h_A, const int N, const int M, double *h_S, double *h_U,
              double *h_V) {
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

  double *d_A = NULL, *d_S = NULL, *d_U = NULL, *d_VT = NULL;
  int *devInfo = NULL;
  double *d_work = NULL;
  double *d_rwork = NULL;
  double *d_W = NULL; // W = S*VT

  int lwork = 0;
  int info_gpu = 0;

  // step 1: create cusolverDn/cublas handle
  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status);

  // step 2: copy A and B to device
  cudaStat1 = cudaMalloc((void **)&d_A, sizeof(double) * N * M);
  cudaStat2 = cudaMalloc((void **)&d_S, sizeof(double) * M);
  cudaStat3 = cudaMalloc((void **)&d_U, sizeof(double) * N * N);
  cudaStat4 = cudaMalloc((void **)&d_VT, sizeof(double) * M * M);
  cudaStat5 = cudaMalloc((void **)&devInfo, sizeof(int));
  cudaStat6 = cudaMalloc((void **)&d_W, sizeof(double) * N * M);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);
  assert(cudaSuccess == cudaStat5);
  assert(cudaSuccess == cudaStat6);

  cudaStat1 =
      cudaMemcpy(d_A, h_A, sizeof(double) * N * M, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);

  // step 3 query working space of SVD
  cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, N, M, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  cudaStat1 = cudaMalloc((void **)&d_work, sizeof(double) * lwork);
  assert(cudaSuccess == cudaStat1);

  // step 4 compute svd
  signed char jobu = 'A';  // all m columns of U
  signed char jobvt = 'A'; // all n columns of VT
  // ATTENZIONE QUA, cusolver_status = cusolverDnDgesvd (
  //  cusolverH,
  //  jobu,
  //  jobvt,
  //  N,
  //  M,
  //  d_A,
  //  N,
  //  d_S,
  //  d_U,
  //  N,  // ldu
  //  d_VT,
  //  N, // ldvt, GNOGNOGNOGNOGNOGNOGNO!!!
  //  d_work,
  //  lwork,
  //  d_rwork,
  //  devInfo);
  cusolver_status =
      cusolverDnDgesvd(cusolverH, jobu, jobvt, N, M, d_A, N, d_S, d_U, N, d_VT,
                       M, d_work, lwork, d_rwork, devInfo);
  cudaStat1 = cudaDeviceSynchronize();
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
  assert(cudaSuccess == cudaStat1);

  cudaStat1 = cudaMemcpy(h_S, d_S, sizeof(double) * M, cudaMemcpyDeviceToHost);
  cudaStat2 =
      cudaMemcpy(h_U, d_U, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
  cudaStat3 =
      cudaMemcpy(h_V, d_VT, sizeof(double) * M * M, cudaMemcpyDeviceToHost);
  cudaStat4 =
      cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);

  // info GPU
  printf("after gesvd: info_gpu = %d\n", info_gpu);
  assert(0 == info_gpu);
  printf("=====\n");

  printf("S = (matlab base-1)\n");
  printf("=====\n");

  printf("U = (matlab base-1)\n");
  printf("=====\n");

  printf("VT = (matlab base-1)\n");
  printf("=====\n");

  // free resources
  //  free resources
  if (d_A)
    cudaFree(d_A);
  if (d_S)
    cudaFree(d_S);
  if (d_U)
    cudaFree(d_U);
  if (d_VT)
    cudaFree(d_VT);
  if (devInfo)
    cudaFree(devInfo);
  if (d_work)
    cudaFree(d_work);
  if (d_rwork)
    cudaFree(d_rwork);
  if (d_W)
    cudaFree(d_W);

  if (cublasH)
    cublasDestroy(cublasH);
  if (cusolverH)
    cusolverDnDestroy(cusolverH);

  cudaDeviceReset();
}
//-------------------------------------------------------
// cuDiagonalization: Diagonalize a symmetric matrix (n x n) in column-major
// order. The computed eigenvectors are stored in h_Q and eigenvalues in
// h_lambda.
void cuDiagonalization(const double *h_C, double *h_Q, double *h_lambda,
                       const int n) {
  double *d_C = NULL, *d_lambda = NULL;
  // d_Q is not used because the eigenvectors overwrite d_C.
  CHECK_CUDA(cudaMalloc((void **)&d_C, sizeof(double) * n * n));
  CHECK_CUDA(cudaMalloc((void **)&d_lambda, sizeof(double) * n));
  CHECK_CUDA(
      cudaMemcpy(d_C, h_C, sizeof(double) * n * n, cudaMemcpyHostToDevice));
  cout << "Matrix copied to device\n";
  cusolverDnHandle_t cusolverH = NULL;
  CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
  int lwork = 0;
  cout << "Querying buffer size\n";
  CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
      cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_C, n,
      d_lambda, &lwork));
  double *d_work = nullptr;
  cout << "Allocating work space\n";
  CHECK_CUDA(cudaMalloc((void **)&d_work, lwork * sizeof(double)));
  int *devInfo = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&devInfo, sizeof(int)));
  // Compute the eigenvalue decomposition.
  CHECK_CUSOLVER(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_UPPER, n, d_C, n, d_lambda,
                                  d_work, lwork, devInfo));
  CHECK_CUDA(cudaDeviceSynchronize());
  cout << "Eigenvalue decomposition completed\n";
  int info_gpu = 0;
  CHECK_CUDA(
      cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_gpu != 0) {
    fprintf(stderr,
            "Error: eigenvalue decomposition did not converge. Info = %d\n",
            info_gpu);
    exit(EXIT_FAILURE);
  }
  cout << "Copying eigenvalues to host\n";
  CHECK_CUDA(cudaMemcpy(h_lambda, d_lambda, sizeof(double) * n,
                        cudaMemcpyDeviceToHost));
  // The eigenvectors are stored in d_C (column-major order).
  CHECK_CUDA(
      cudaMemcpy(h_Q, d_C, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
  cout << "Eigenvectors copied to host\n";
  cudaFree(d_C);
  cudaFree(d_lambda);
  cudaFree(d_work);
  cudaFree(devInfo);
  cusolverDnDestroy(cusolverH);
}

int pod_main() {
  int N = 2 * DIM * DIM;
  int M = MAX_FRAMES % SNAPSHOT_INTERVAL; // Number of frames (columns)
  // integer part of a division
  string filename = "snapshots.csv";

  //    std::cout << "Row-major matrix:\n";
  // printMatrix(mat, rows, cols, Layout::RowMajor);
  vector<double> data;
  loadCsv(filename, data, N, M, Layout::ColMajor);
  // printMatrix(data, N, M, Layout::RowMajor);
  // cout << "------------------------\n";
  // printMatrix(data, N, M, Layout::ColMajor);
  vector<double> S(M, 0.0);
  vector<double> U(M * M, 0.0);
  vector<double> Vt(M * M, 0.0);
  vector<double> data_T(M * N, 0.0);
  vector<double> Corr;
  // make the transpose of the data matrix
  for (int i = 0; i < M;
       ++i) { // i iterates over the rows of Aᵀ (which are the columns of A)
    for (int j = 0; j < N;
         ++j) { // j iterates over the columns of Aᵀ (which are the rows of A)
      data_T[j * M + i] = data[i * N + j];
    }
  }

  // reconstruct the Sigma matrix

  //==========================================
  // Perform SVD
  // svdCudas(data.data(), N, M, S.data(), U.data(), Vt.data());

  // vector<double> Sigma(N*M, 0.0);
  // for (int i = 0; i < mn; ++i) {
  //     Sigma[i * N + i] = S[i];
  //     cout << S[i] << "\t";
  // }
  cout << "Checking for correlations\n";
  Corr = matmul(data_T, data, M, N, M, Layout::ColMajor);
  cout << "Covariance Matrix Computed\n";
  cuDiagonalization(Corr.data(), U.data(), S.data(), M);
  cout << "Diagonalization Successful\n";
  cout << "------------------------\n";
  // printMatrix(Corr, N, N, Layout::RowMajor);
  //  printMatrix(U, N, N, Layout::RowMajor);
  // printMatrix(S, N, M, Layout::RowMajor);
  // save
  //  saveMatrix("Sigma.txt", Sigma, N, M, Layout::RowMajor);
  //  saveMatrix("Vt.txt", Vt, M, M, Layout::RowMajor);
  // reconstruct diagonal matrix
  //  vector<double> diag(N*N, 0.0);
  //  for (int i = 0; i < mn; ++i) {
  //      diag[i * N + i] = S[i];
  //  }

  // ==========================================
  // compute pod modes
  vector<double> phi(N * M, 0.0);
  phi = matmul(data, U, N, M, M, Layout::ColMajor);
  // print eigenvalues:

  cout << "phivaluess:\n";
  for (int i = 0; i < M; ++i) {
    cout << phi[i] << "\t";
  }
  // print the size of phi

  // for (int i =0; i<M; i++){
  //     for (int j=0; j<N; j++){
  //         phi[i*N+j] = phi[i*N+j]/(S[i]*M);
  //         cout << phi[i*N+j] << "\t";

  //     }
  //     cout << "\n";
  // }
  cout << "POD Modes Computed\n";
  cout << "------------------------\n";
  // save pod modes
  saveMatrix("resultsdata/pod_modes.txt", phi, N, M, Layout::ColMajor);
  cout << "POD Modes Saved\n";
  // save eigenvalues
  if (savesigma) {
    saveMatrix("sigma.txt", S, M, 1, Layout::RowMajor);
  }
  cout << "Eigenvalues Saved\n";
  cout << "------------------------\n";
  // save covariance matrix}
  return 0;
}