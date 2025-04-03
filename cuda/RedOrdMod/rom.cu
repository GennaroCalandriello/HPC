// rom.cu
#include <Eigen/Dense>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Parameters
const int Nx = 512;
const int Ns = 50;
const int r  = 5;

// CUDA kernel: Evaluate ROM
__global__ void romEval(double* basis, double* coeffs, double* output, double t, int Nx, int r){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < Nx){
        double val = 0.0;
        for(int k=0;k<r;k++)
            val += basis[idx*r+k]*coeffs[k]*sin(0.05*t*(k+1));
        output[idx] = val;
    }
}

// Generate snapshots (CPU)
void generateSnapshots(Eigen::MatrixXd &snapshots){
    for(int i=0; i<Ns; i++)
        for(int j=0; j<Nx; j++)
            snapshots(j,i)=sin(0.01*j*(i+1));
}

int main(){
    // GLFW Initialization
    if(!glfwInit()){ std::cerr<<"GLFW Init error\n"; return -1; }
    GLFWwindow* window=glfwCreateWindow(800,600,"ROM CUDA+GLEW",NULL,NULL);
    glfwMakeContextCurrent(window);

    // GLEW Initialization
    glewExperimental=true;
    if(glewInit()!=GLEW_OK){ std::cerr<<"GLEW Init error\n"; return -1; }

    // --- POD (Eigen) ---
    Eigen::MatrixXd snapshots(Nx,Ns);
    generateSnapshots(snapshots);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(snapshots, Eigen::ComputeThinU);
    Eigen::MatrixXd basis=svd.matrixU().leftCols(r);

    // Transfer POD basis to GPU
    double *d_basis,*d_coeffs,*d_output;
    std::vector<double> h_basis(Nx*r);
    for(int i=0;i<Nx;i++)
        for(int k=0;k<r;k++)
            h_basis[i*r+k]=basis(i,k);

    cudaMalloc(&d_basis,Nx*r*sizeof(double));
    cudaMalloc(&d_coeffs,r*sizeof(double));
    cudaMalloc(&d_output,Nx*sizeof(double));
    cudaMemcpy(d_basis,h_basis.data(),Nx*r*sizeof(double),cudaMemcpyHostToDevice);

    // Coefficients setup
    double h_coeffs[r]; for(int k=0;k<r;k++)h_coeffs[k]=1.0/(k+1);
    cudaMemcpy(d_coeffs,h_coeffs,r*sizeof(double),cudaMemcpyHostToDevice);

    // OpenGL VBO setup
    GLuint vbo;
    glGenBuffers(1,&vbo);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,Nx*2*sizeof(float),nullptr,GL_DYNAMIC_DRAW);

    double t=0.0;
    std::vector<double> h_output(Nx);
    float dx=2.0f/Nx;

    // Main render loop clearly visualizing animation
    while(!glfwWindowShouldClose(window)){
        // CUDA kernel execution
        int blocks=(Nx+255)/256;
        romEval<<<blocks,256>>>(d_basis,d_coeffs,d_output,t,Nx,r);
        cudaMemcpy(h_output.data(),d_output,Nx*sizeof(double),cudaMemcpyDeviceToHost);

        // Update vertices
        std::vector<float> vertices(Nx*2);
        for(int i=0;i<Nx;i++){
            vertices[2*i]=-1.0f+i*dx;
            vertices[2*i+1]=(float)h_output[i]*0.5f; // scaled visualization
        }
        glBufferSubData(GL_ARRAY_BUFFER,0,vertices.size()*sizeof(float),vertices.data());

        // Clear and Draw
        glClearColor(0.1f,0.1f,0.1f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2,GL_FLOAT,0,0);
        glDrawArrays(GL_LINE_STRIP,0,Nx);
        glDisableClientState(GL_VERTEX_ARRAY);

        // Update window
        glfwSwapBuffers(window);
        glfwPollEvents();

        t+=0.2; // time stepping speed
    }

    // Cleanup
    cudaFree(d_basis); cudaFree(d_coeffs); cudaFree(d_output);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
