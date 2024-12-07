#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

// Include GLFW
#include "glfw-3.4.bin.WIN64/include/GLFW//glfw3.h"

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16
#define THREADS_PER_BLOCK 256
#define RES 512       // Resolution for the OpenGL window
#define dim 256       // Simulation grid dimension
#define timestep 0.01f
#define hbar 1.0f     // Planck's constant divided by 2π
#define pi 3.14159265358979323846f

#define IND(i, j, k, dim) ((i)+(j)*dim +(k)*dim*dim)

//wavefunction components
cuDoubleComplex *psi1;
cuDoubleComplex *psi2;

//device components
cuDoubleComplex *dev_psi1;
cuDoubleComplex* dev_psi2;

//Schrodinger mask for evolution in momentum space
cuDoubleComplex *mask;
cuDoubleComplex *dev_mask;

//CUDA FFT
cufftHandle plan;

//Visualizations
float* probdensity;
float* dev_probdensity;