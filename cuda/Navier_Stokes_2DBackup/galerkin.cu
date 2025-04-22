#include "snaptest.cu"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cusolverDn.h>
enum class Layout { RowMajor, ColMajor };
bool wantSnapshot = true;
bool wantGalerkin = true;
bool wantPodGeneration = true;
bool loadSnapFromFile = false;
bool savePOD = true;
bool loadPODFromFile = false;
Layout layout = Layout::ColMajor;
int M = static_cast<int>(MAX_FRAMES / SNAPSHOT_INTERVAL);
int N = 2 * DIM * DIM; // Number of rows in the matrix.
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at "          \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  }
// Central differences in x for one component vf
__device__ float ddx_(const Vector2f *field, int i, int j, unsigned dim,
                      float dx, int comp) {
  // indices
  int center = IND(i, j, dim);
  int left = (i > 0) ? IND(i - 1, j, dim) : center;
  int right = (i < int(dim) - 1) ? IND(i + 1, j, dim) : center;

  // fields
  float left_f = (comp == 0) ? field[left].x : field[left].y;
  float right_f = (comp == 0) ? field[right].x : field[right].y;
  return (right_f - left_f) / (2.0f * dx);
}

__device__ float pressureGradient(const Vector2f *field, int i, int j,
                                  unsigned dim, float dx) {
  // indices
  int center = IND(i, j, dim);
  int left = (i > 0) ? IND(i - 1, j, dim) : center;
  int right = (i < int(dim) - 1) ? IND(i + 1, j, dim) : center;
  int down = (j > 0) ? IND(i, j - 1, dim) : center;
  int up = (j < int(dim) - 1) ? IND(i, j + 1, dim) : center;

  // fields
  float left_f = field[left].x;
  float right_f = field[right].x;
  float down_f = field[down].y;
  float up_f = field[up].y;

  return (right_f - left_f) / (2.0f * dx) + (up_f - down_f) / (2.0f * dx);
}

__device__ float ddy_(const Vector2f *field, int i, int j, unsigned dim,
                      float dx, int comp) {
  // indices
  int center = IND(i, j, dim);
  int down = (j > 0) ? IND(i, j - 1, dim) : center;
  int up = (j < int(dim) - 1) ? IND(i, j + 1, dim) : center;

  // fields
  float down_f = (comp == 0) ? field[down].x : field[down].y;
  float up_f = (comp == 0) ? field[up].x : field[up].y;
  return (up_f - down_f) / (2.0f * dx);
}
// Symm. gradient tensor (returns S11, S22, S12= S21)
__device__ void grad_s(const Vector2f *phi, int i, int j, unsigned dim,
                       float dx, float &S11, float &S22, float &S12) {
  S11 = ddx_(phi, i, j, dim, dx, 0);             // ∂(φ.x)/∂x
  S22 = ddy_(phi, i, j, dim, dx, 1);             // ∂(φ.y)/∂y
  float dphi_x_dy = ddy_(phi, i, j, dim, dx, 0); // ∂(φ.x)/∂y
  float dphi_y_dx = ddx_(phi, i, j, dim, dx, 1); // ∂(φ.y)/∂x
  S12 = 0.5f * (dphi_x_dy + dphi_y_dx);          // (∂(φ.x)/∂y+ ∂(φ.y)/∂x) / 2
}
// Divergence of 2∇_s φ at grid point (i,j).
__device__ Vector2f div_g(const Vector2f *phi, int i, int j, int dim,
                          float dx) {
  // First, compute the symmetric gradient components at (i,j)
  float S11, S22, S12;
  grad_s(phi, i, j, dim, dx, S11, S22, S12);

  // We want to compute the divergence:
  Vector2f div;
  float S11_u, S22_u,
      S12_u; // S11_u = ∂(S11)/∂x, S22_u = ∂(S22)/∂y, S12_u = ∂(S12)/∂y
  float S11_d, S22_d, S12_d;
  float S11_r, S22_r, S12_r;
  float S11_l, S22_l, S12_l;
  float div_x, div_y;
  float d_S12_dy, d_S12_dx, d_S11_dx, d_S22_dy;

  // For ∂(2S12)/∂x: compute S12 at (i+1, j) and (i-1, j)
  {
    float uff1, uff2, S12temp;
    grad_s(phi, min(i + 1, dim - 1), j, dim, dx, uff1, uff2, S12temp);
    S12_r = 2.0f * S12temp;
    grad_s(phi, max(i - 1, 0), j, dim, dx, uff1, uff2, S12temp);
    S12_l = 2.0f * S12temp;
  }
  //
  // For ∂(2S12)/∂y: compute S12 at (i, j+1) and (i, j-1)
  {
    float uff1, uff2, S12temp;
    grad_s(phi, i, min(j + 1, dim - 1), dim, dx, uff1, S12temp, uff2);
    S12_u = 2.0f * S12temp;
    grad_s(phi, i, max(j - 1, 0), dim, dx, uff1, S12temp, uff2);
    S12_d = 2.0f * S12temp;
  }
  //   ∂(2S22) / ∂y : compute S22 at(i, j + 1) and (i, j - 1)
  {
    float uff1, uff2, S22temp;
    grad_s(phi, i, min(j + 1, (int)dim - 1), dim, dx, uff1, S22temp, uff2);
    S22_u = 2.0f * S22temp;
    grad_s(phi, i, max(j - 1, 0), (int)dim, dx, uff1, S22temp, uff2);
    S22_d = 2.0f * S22temp;
  }
  // For ∂(2S11)/∂x: compute S11 at (i+1, j) and (i-1, j)
  {
    float uff1, uff2, S11temp;
    grad_s(phi, min(i + 1, (int)dim - 1), j, dim, dx, S11temp, uff1, uff2);
    S11_r = 2.0f * S11temp;
    grad_s(phi, max(i - 1, 0), j, dim, dx, S11temp, uff1, uff2);
    S11_l = 2.0f * S11temp;
  }

  d_S11_dx = (S11_r - S11_l) / (2.0f * dx); // ∂(S11)/∂x
  d_S22_dy = (S22_u - S22_d) / (2.0f * dx); // ∂(S22)/∂y
  d_S12_dx = (S12_r - S12_l) / (2.0f * dx); // ∂(S12)/∂x
  d_S12_dy = (S12_u - S12_d) / (2.0f * dx); // ∂(S12)/∂y

  // Compute the divergence:
  div.x = d_S11_dx + d_S12_dy; // ∂/∂x(2S11) + ∂/∂y(2S12)
  div.y = d_S12_dx + d_S22_dy; // ∂/∂x(2S12) + ∂/∂y(2S22)

  return div;
}
__device__ Vector2f divOutProd(const Vector2f *u, int i, int j, int dim,
                               float dx) {
  int idx = IND(i, j, dim);

  // (div T)_x = dT11/dx + dT12/dy,
  // (div T)_y = dT21/dx + dT22/dy.
  // central diff.

  float T11_r, T11_l, T12_u, T12_d, T12_r, T12_l, T22_u, T22_d;

  // tutti i nearest neighbours
  T11_r = u[IND(min(i + 1, dim - 1), j, dim)].x *
          u[IND(min(i + 1, dim - 1), j, dim)].x;
  T11_l = u[IND(max(i - 1, 0), j, dim)].x * u[IND(max(i - 1, 0), j, dim)].x;
  T12_u = u[IND(i, min(j + 1, (int)dim - 1), dim)].x *
          u[IND(i, min(j + 1, (int)dim - 1), dim)].y;
  T12_d = u[IND(i, max(j - 1, 0), dim)].x * u[IND(i, max(j - 1, 0), dim)].y;
  T12_r = u[IND(min(i + 1, dim - 1), j, dim)].x *
          u[IND(min(i + 1, dim - 1), j, dim)].y;
  T12_l = u[IND(max(i - 1, 0), j, dim)].x * u[IND(max(i - 1, 0), j, dim)].y;
  T22_u = u[IND(i, min(j + 1, (int)dim - 1), dim)].y *
          u[IND(i, min(j + 1, (int)dim - 1), dim)].y;
  T22_d = u[IND(i, max(j - 1, 0), dim)].y * u[IND(i, max(j - 1, 0), dim)].y;

  float dT11_dx = (T11_r - T11_l) / (2.0f * dx);
  float dT12_dy = (T12_u - T12_d) / (2.0f * dx);
  float dT12_dx = (T12_r - T12_l) / (2.0f * dx);
  float dT22_dy = (T22_u - T22_d) / (2.0f * dx);
  Vector2f div;
  div.x = dT11_dx + dT12_dy; // ∂/∂x(2S11) + ∂/∂y(2S12)
  div.y = dT12_dx + dT22_dy; // ∂/∂x(2S12) + ∂/∂y(2S22)
  // if (i % 400 == 0 && j % 400 == 0) {
  //   printf("divOutProd: i=%d, j=%d, div.x=%f, div.y=%f\n", i, j, div.x,
  //   div.y);
  //   // div.print();
  // }
  return div;
}

// Galerkin kernel
__global__ void galerkin_kernel(Vector2f *phi, Vector2f *div_phi,
                                Vector2f *grad_chi, Vector2f *div_grad_phi,
                                Vector2f *div_outProd_phi, float *p, float *c,
                                float rdx, unsigned dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= dim || j >= dim)
    return;

  int idx = IND(i, j, dim);
  // compute divgradphi
  // Vector2f div = div_g(phi, i, j, dim, rdx);
  // div_phi = divOutProd(phi, i, j, dim, rdx);
  // float gradP = pressureGradient(p, i, j, dim, rdx);
  // inner product with phi

  // Optionally check obstacleField[idx] here.
  // Vector2f div = divOutProd(phi, i, j, dim, rdx);

  // // **write it into the output array at position idx**:
  // div_phi = div;
}

__global__ void innerProd_kernel(Vector2f *phi, Vector2f *phi2, float *result,
                                 unsigned dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= dim || j >= dim)
    return;

  int idx = IND(i, j, dim);
  result[idx] = phi[idx].x * phi2[idx].x + phi[idx].y * phi2[idx].y;
}

void loadfile(const std::string &filename, std::vector<double> &matrix,
              int &rows, int &cols, Layout layout = Layout::ColMajor) {
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
  file.close();
}

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Loads an N×M CSV so that matrix[i][j] = row i, col j
// matrix.size()==N  and  matrix[i].size()==M
void loadMatrix(const std::string &filename,
                std::vector<std::vector<double>> &matrix) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Cannot open file: " + filename);

  std::string line;
  matrix.clear();

  // Read each CSV line into a row
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell));
    }
    // Check for consistent column count
    if (!matrix.empty() && row.size() != matrix[0].size())
      throw std::runtime_error("Inconsistent number of columns in CSV");
    matrix.push_back(std::move(row));
  }
  file.close();
}
// Example main() to test the function:
bool diag =
    false; // Flag to control whether to diagonalize the covariance matrix

// Global flags to control saving behavior
bool savesigma = true; // Save singular values if true
int pod_to_save = 10;  // Number of POD modes (columns) to save

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

void pod_main(vector<double> data) {

  // integer part of a division
  if (loadSnapFromFile) {
    string filename = "resultsdata/snapshots.csv";

    //    std::cout << "Row-major matrix:\n";
    // printMatrix(mat, rows, cols, Layout::RowMajor);
    // vector<double> data;
    loadCsv(filename, data, N, M, Layout::ColMajor);
  }

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
  cout << "Covariance dimensions: " << Corr.size() << "\n";
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
  if (savePOD) {
    cout << "Saving POD modes\n";
    saveMatrix("resultsdata/pod_modes.txt", phi, N, M, Layout::ColMajor);
    cout << "POD Modes Saved\n";
    // save eigenvalues
  }

  if (savesigma) {
    saveMatrix("sigma.txt", S, M, 1, Layout::RowMajor);
  }
  cout << "Eigenvalues Saved\n";
  cout << "------------------------\n";
  // save covariance matrix}
}

int main() {

  vector<vector<double>> new_matrix;
  vector<double> data;
  data.resize(N * M, 0.0);
  std::vector<std::vector<float>> snapshot_matrix;

  cout << "Number of pods: " << M << endl;

  if (wantSnapshot) {
    cout << "Snapshot requested, function call snap_main()." << endl;

    snap_main(snapshot_matrix);
  } else {
    cout << "No snapshot requested." << endl;
  }
  cout << "Snapshot matrix size: " << snapshot_matrix.size() << "--"
       << snapshot_matrix[0].size() << endl;
  if (wantPodGeneration) {
    cout << "POD requested, function call pod_main()." << endl;

    // ERRORE QUI!!! snapshot_matrix is rowmajor, soltanto quando salva, la
    // salva in colmajor
    if (!loadSnapFromFile) {

      for (int j = 0; j < M; ++j) {
        for (int i = 0; i < N; ++i) {
          double v = snapshot_matrix[j][i];
          if (layout == Layout::ColMajor) {
            // store (i,j) at index = i + j*N
            data[i + j * N] = v;
          } else {
            // store (i,j) in row‑major at index = i*M + j
            data[i * M + j] = v;
          }
        }
      }
    }
    cout << "Snapshot matrix passed into data." << endl;

    cout << "Data matrix size: " << data.size() << endl;
    // save data array
    // saveMatrix("resultsdata/data.txt", data, N, M, layout);

    pod_main(data);

    new_matrix.reserve(N);

    for (int i = 0; i < N; ++i) {
      std::vector<double> row;
      row.reserve(M);
      for (int j = 0; j < M; ++j) {
        double v;
        if (layout == Layout::ColMajor) {
          // element (i,j) lives at data[i + j*N]
          v = data[i + j * N];
        } else {
          // row‐major: data[i*M + j]
          v = data[i * M + j];
        }
        row.push_back(v);
      }
      new_matrix.push_back(std::move(row));
    }

  } else {
    cout << "No POD requested." << endl;
  }

  if (wantGalerkin) {
    cout << "Galerkin requested, function call galerkin_main()." << endl;
    if (loadPODFromFile) {
      cout << "Loading POD modes from file." << endl;
      string filename = "resultsdata/pod_modes.txt";
      loadMatrix(filename, new_matrix);
      cout << "Loaded POD modes from file." << endl;
    }

    std::vector<Vector2f> u(N * M);

    cout << "Loaded " << N << " rows and " << M << " columns from file."
         << endl;

    // Convert in Vector2f struct

    int dim = DIM;
    int gridPoints = DIM * DIM; // Number of spatial points per component.
    int gridSize = static_cast<int>(std::sqrt(static_cast<double>(gridPoints)));
    int framecount = 0;

    // fields
    float *p = (float *)malloc(DIM * DIM * sizeof(float));
    int *c = (int *)malloc(DIM * DIM * sizeof(int));
    int *obstacle = (int *)malloc(DIM * DIM * sizeof(int));
    // Matrix for galerkin projection
    // 3) Pre‐allocazione host‐side per i risultati
    //    Tutte e M le modalità
    // 1) Dichiara un array di puntatori
    Vector2f **all_phi = new Vector2f *[M];
    Vector2f **all_divphi = new Vector2f *[M];
    Vector2f **all_gradchi = new Vector2f *[M];
    Vector2f **all_divgradphi = new Vector2f *[M];
    Vector2f **all_divoutprodphi = new Vector2f *[M];

    // 4) Allocazione device‐side (una volta sola)
    Vector2f *d_mode, *d_divphi, *d_gradchi, *d_divgradphi, *d_divoutprodphi;
    CHECK_CUDA(cudaMalloc(&d_mode, gridPoints * sizeof(Vector2f)));
    CHECK_CUDA(cudaMalloc(&d_divphi, gridPoints * sizeof(Vector2f)));
    CHECK_CUDA(cudaMalloc(&d_gradchi, gridPoints * sizeof(Vector2f)));
    CHECK_CUDA(cudaMalloc(&d_divgradphi, gridPoints * sizeof(Vector2f)));
    CHECK_CUDA(cudaMalloc(&d_divoutprodphi, gridPoints * sizeof(Vector2f)));

    // float *Mass = (float *)malloc(M * M, sizeof(float));
    // float *A = (float *)malloc(M * M, sizeof(float));
    // float *B = (float *)malloc(M * M, sizeof(float));
    // float *P = (float *)malloc(M * M, sizeof(float));

    // I primi gridPoints elementi sono le componenti x, i successivi
    // gridPoints quelli y. La matrice dei dati è in column-major.
    //===================================================
    //======WARNING: CONTROLLARE BENE L'OUTPUT================
    int *d_obstacle;
    float *d_p;
    float *d_c;
    CHECK_CUDA(cudaMalloc(&d_obstacle, gridPoints * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_p, gridPoints * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, gridPoints * sizeof(int)));
    //===================================================

    cudaMemcpy(d_obstacle, obstacle, gridPoints * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, gridPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, gridPoints * sizeof(int), cudaMemcpyHostToDevice);

    //===================================================
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX,
                (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);

    for (int m = 0; m < M; m++) {
      // qui calcolo tutti i termini necessari e li salvo in matrici
      // M*gridpoints per usarli nei loop successivi

      // alloco i vettori per ciascun termine
      Vector2f *mode = (Vector2f *)malloc(DIM * DIM * sizeof(Vector2f));

      // define the mode for each frame

      cout << "check time step " << m << endl;
      cout << new_matrix.size() << " " << new_matrix[0].size() << endl;
      for (int i = 0; i < gridPoints; i++) {
        float xt =
            new_matrix[i][m]; // x component (dalla prima metà delle righe)
        float yt = new_matrix[i + gridPoints][m]; // y component (dalla seconda
        mode[i].x = xt;                           // metà delle righe)
        mode[i].y = yt;
      }
      cout << "check6" << endl;
      cudaMemcpy(d_mode, mode, gridPoints * sizeof(Vector2f),
                 cudaMemcpyHostToDevice);
      cout << "Time slice: " << m << endl;
      galerkin_kernel<<<blocks, threads>>>(d_mode, d_divphi, d_gradchi,
                                           d_divgradphi, d_divoutprodphi, d_p,
                                           d_c, rdx, dim);
      // ricopia sull'host tutti i valori
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      // 5d) Copio indietro su host i risultati in vettori temporanei
      Vector2f *h_mode = (Vector2f *)malloc(gridPoints * sizeof(Vector2f));
      Vector2f *h_divphi = (Vector2f *)malloc(gridPoints * sizeof(Vector2f));
      Vector2f *h_gradchi = (Vector2f *)malloc(gridPoints * sizeof(Vector2f));
      Vector2f *h_divgradphi =
          (Vector2f *)malloc(gridPoints * sizeof(Vector2f));
      Vector2f *h_divoutprodphi =
          (Vector2f *)malloc(gridPoints * sizeof(Vector2f));
      CHECK_CUDA(cudaMemcpy(h_divphi, d_divphi, gridPoints * sizeof(Vector2f),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(h_gradchi, d_gradchi, gridPoints * sizeof(Vector2f),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(h_divgradphi, d_divgradphi,
                            gridPoints * sizeof(Vector2f),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(h_divoutprodphi, d_divoutprodphi,
                            gridPoints * sizeof(Vector2f),
                            cudaMemcpyDeviceToHost));

      // 5e) Spingo nei vettori all_ con push_back
      all_phi[m] = new Vector2f[gridPoints];
      all_divphi[m] = new Vector2f[gridPoints];
      all_gradchi[m] = new Vector2f[gridPoints];
      all_divgradphi[m] = new Vector2f[gridPoints];
      all_divoutprodphi[m] = new Vector2f[gridPoints];
      // free host
    }
  }
}