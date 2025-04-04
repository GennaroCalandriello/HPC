#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cusolverDn.h>
#include "scalar.h"

using namespace std;
bool savesigma = true;
int pod_to_save = 10;

//======================================================
//definisco CHECK_CUDA e CHECK_CUSOLVER per controllare gli errori
//======================================================
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSolver error in %s at line %d: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
}

//matmul kernel
__global__ void matMulKer( double* A, double* B, double* C, int N, int M) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row<N && col <N){
        double val = 0.0;
        for (int k =0; k<M; k++){
            val+= A[row*M+k]*B[col*M+k];
        }
        C[row*N+col] = val;
    }
}

void CovarianceMatrix(vector<double>& A, int N, int M, vector<double>& cov){
    //alloco la memoria device
    double *d_A = nullptr, *d_cov = nullptr, *d_AT = nullptr;
    vector<double> AT(N*M, 0.0);
    //transpose A in AT

    for (int i =0; i<N; i++){
        for (int j =0; j<M; j++){
            AT[j*N+i] = A[i*M+j];
        }
    }

    CHECK_CUDA(cudaMalloc(&d_AT, M*N*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_A, N*M*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cov, N*N*sizeof(double)));

    //copia da host a device

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), N*M*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_AT, AT.data(), M*N*sizeof(double), cudaMemcpyHostToDevice));

    //Lancio il kernel

    dim3 block(BLOCKSIZEX, BLOCKSIZEY);
    dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);
    matMulKer<<<grid, block>>>(d_AT, d_A, d_cov, N, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(cov.data(), d_cov, N*N*sizeof(double), cudaMemcpyDeviceToHost));

    //clean
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_cov));
}

vector<double> loadCsv(const string& filename, int& N, int& M) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    vector<vector<double>> rows;
    string line;
    while(getline(file, line)){
        stringstream ss(line);
        string token;
        vector<double> row;
        while(getline(ss, token, ',')){
            row.push_back(stod(token));
        }
        rows.push_back(row);
    } 
    file.close();

    //check data
    N = (int)rows.size();
    if (N == 0) {
        cerr << "Error: No data found in file: " << filename << endl;
        exit(1);
    }
    M = (int)rows[0].size();
    //convertiamo a 1D
    vector<double> matrix(N*M);
    for (int i =0; i<N; i++){
        if ((int)rows[i].size() != M) {
            cerr << "Error: Inconsistent number of columns in file: " << filename << endl;
            exit(1);
        }
        for (int j =0; j<M; j++){
            matrix[i*M+j] = rows[i][j];
        }
    }
    return matrix;

}

//center the data
void centerData(vector<double>& data, int N, int M){
    for (int i =0; i<N; i++){
        //media della riga i
        double sum =0.0;
        for (int j =0; j<M; j++){
            sum+= data[i*M+j];
        }
        double mean = sum/M;
        //sottraggo la media
        for (int j =0; j<M; j++){
            data[i*M+j] -= mean;
        }

    }
}

//=====================================================
//GPU SVD using cuSolver
//=====================================================
void svdGPU(vector<double>& cov, int N, vector<double>& S, vector<double>& U){
    //cov in device
    double* d_cov = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cov, N*N*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_cov, cov.data(), N*N*sizeof(double), cudaMemcpyHostToDevice));
    
    //SVD: cov = U*S*V^T
    double* d_U = nullptr;
    double* d_S = nullptr;
    double* d_VT = nullptr;
    CHECK_CUDA(cudaMalloc(&d_U, N*N*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_S, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_VT, N*N*sizeof(double)));

    //cuSolver
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(handle, N, N, &lwork));
    double* d_work = nullptr;
    CHECK_CUDA(cudaMalloc(&d_work, lwork*sizeof(double)));
    int* devInfo = nullptr;
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    char jobu = 'A'; // all U
    char jobvt = 'A'; // all V^T

    //SVD
    CHECK_CUSOLVER(cusolverDnDgesvd(handle, jobu, jobvt, N, N, d_cov, N, d_S, d_U, N, d_VT, N, d_work, lwork, nullptr, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());

    //check devInfo
    int infoCpu=0;
    CHECK_CUDA(cudaMemcpy(&infoCpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if(infoCpu != 0) {
        cerr << "Error: SVD failed with info = " << infoCpu << endl;
        exit(1);
    }
    //copy results to host
    S.resize(N);
    U.resize(N*N);
    vector<double> VT(N*N);

    CHECK_CUDA(cudaMemcpy(S.data(), d_S, N*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(U.data(), d_U, N*N*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(VT.data(), d_VT, N*N*sizeof(double), cudaMemcpyDeviceToHost));

    //free memory
    cudaFree(d_cov);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_VT);
    cudaFree(d_work);
    cudaFree(devInfo);
    cudaFree(handle);
    cusolverDnDestroy(handle);
}

void printMatrix(const vector<double>& matrix) {
    int rows = (int)sqrt(matrix.size());
    int cols = (int)sqrt(matrix.size());
    cout << "Matrix:" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char** argv){

    // if(argc<2){
    //     cerr << "Usage: " << argv[0] << " <input_file.csv>" << endl;
    //     return 1;
    // }
    // string filename = argv[1];
    string filename = "snapshots.csv";

    //1. Load data from CSV file
    int N =DIM, M = DIM; //DIM è definito in scalar.h
    vector<double> data = loadCsv(filename, N, M);
    cout << "Data loaded from " << filename << endl;

    //2. Center the data
    // centerData(data, N, M);
    // cout << "Data centered" << endl;

    //3. Compute covariance matrix
    vector<double> cov(N*N, 0.0);
    CovarianceMatrix(data, N, M, cov);
    cout << "Covariance matrix computed" << endl;
    cout << "Covariance matrix" << endl;
    printMatrix(cov); // Stampa la matrice di covarianza

    //4. Compute SVD on GPU
    vector<double> Sigma, U;
    svdGPU(cov, N, Sigma, U);

    //5. show some results
    cout <<"Singular values (first 5): ";
    for (int i =0; i<5 && i<(int)Sigma.size(); i++){
        cout << Sigma[i] << " ";
    }

    //6. Salva su file la sigma se la vuoi
    if (savesigma){ofstream outFile("output.txt");
    for(auto &val : Sigma){
        outFile << val << endl;

    }
    outFile.close();}
    

    //7. Save the first pod_to_save vectors in U
    ofstream modesOut("pod_modes.txt");
    //scriviamo riga per riga i primi pod_to_save vettori di U
    for (int i =0; i<N; i++){
        for (int j =0; j<pod_to_save; j++){
            double val = U[i+j*N]; //U è in colonna => (i, j) -> i+j*N
            modesOut << val << " ";
            if(j< pod_to_save-1) modesOut <<",";
        }
        modesOut << "\n";
    }
    modesOut.close();
    cout <<"Vediamo se funziona" << endl;

    return 0;
}