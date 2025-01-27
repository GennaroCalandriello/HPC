#include "torus.cu"

struct ISflow {
    double hbar; //control the vortex quantization
    double dt;
    cuDoubleComplex* mask; //mask is a complex array
};

struct parameters {

    //pagina 3 Schrodinger's Smoke
    double velocity[3];

    char* isJet;

    cuDoubleComplex* psi1;
    cuDoubleComplex psi2;

    double kvector[3];
    double omega;
    double* phase;
}

__constant__ parameters d_params;
d_params d_params_cpu;

__constant__ ISFlow isflow;
ISFlow isflow_cpu;

__global__ void ISFlow_normalize_ker()
{
    int ind = check_limit(torus.plen);
    if (ind<0) return;

    cuDoubleComplex* psi1 = d_params.psi1;
    cuDoubleComplex* psi2 = d_params.psi2;

    double norm_psi = sqrt(psi1[ind].x*psi1[ind].x+psi1[ind].y*psi1[ind].y+
                           psi2[ind].x*psi2[ind].x+psi2[ind].y*psi2[ind].y);

    complexDiv(&psi1[ind], norm_psi);
    complexDiv(&psi2[ind], norm_psi);
}

void ISFlow_normalize()
{
    int nb = numblock(torus.plen);
    ISFlow_normalize_ker<<<nb, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
}

__global__ void ISFlow_Schrodinger_ker()
//This kernel constructs a time evolution operator
//ψ(k,t+dt)= exp(-iħΔt|k|/2)ψ(k,t)

{
    int ind = check_limit(torus.plen);
    if(ind<0) return;
    double nx = torus.resx, ny = torus.resy; nz = torus.resz;
    double factor = -4.0f*pi*pi*isflow.hbar;

    int i, j, k;
    getCoords(ind, &i, &j, &k);
    double kx = (i-nx/2)/torus.sizex; //for zero frequencies at the center
    double ky = (j-ny/2)/torus.sizey;
    double kz = (k-nz/2)/torus.sizez;
    double lambda = factor*(kx*kx+ky*ky+kz*kz); //laplacian operator in fourier space

    cuDoubleComplex temp;
    temp.x = 0;
    temp.y = lambda*isflow.dt/2;

    isflow.mask[index3d(i, j, k)] = exp_complex(temp);
}

void ISFlow_kernel_exe()
{
    int nb = numblock(torus.plen);
    ISFlow_Schrodinger_ker<<<nb, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
}

__global__ void ISFlow_Prod(cuDoubleComplex* a, cuDoubleComplex* b)
{
    int i = check_limit(torus.plen);
    if(i<0) return; 
    a[i] = cuCmul(a[i], b[i]);
}

void ISF_Flow()
//Solve Schroedinger equation for dt time
{
    fftcuda(para_cpu.psi1);
    fftcuda(para_cpu.psi2);
    cudaDeviceSynchronize();

    int nb = numblock(torus_cpu.plen/2);
    fftshift<<<nb, THREADS_PER_BLOCK>>>(para_cpu.psi1);
    fftshift<<<nb, THREADS_PER_BLOCK>>>(para_cpu.psi2);
    cudaDeviceSynchronize();
    
    int nb2 = numblock(torus_cpu.plen);
    ISFlow_prod<<<nb2, THREADS_PER_BLOCK>>>(para_cpu.psi1, isflow_cpu.mask);
    ISFlow_prod<<<nb2, THREADS_PER_BLOCK>>>(para_cpu.psi2, isflow_cpu.mask);
    
    cudaDeviceSynchronize();

    fftshift<<<nb, THREADS_PER_BLOCK>>>(para_cpu.psi1);
    fftshift<<<nb, THREADS_PER_BLOCK>>>(para_cpu.psi2);
    cudaDeviceSynchronize();
    ifftcuda(para_cpu.psi1);
    ifftcuda(para_cpu.psi2);
    cudaDeviceSynchronize();
}

__global__ void ISFlow_velocity_1form()
{
    int ind = check_limit(torus.plen);
    if (ind<0) return;
    cuDoubleComplex* psi1 = d_params.psi1;
    cuDoubleComplex* psi2 = d_params.psi2;

    int i, j, k;
    getCoords(ind, &i, &j, &k);
    
    int ixp = (i+1)%torus.resx;
    int iyp = (j+1)%torus.resy;
    int izp = (k+1)%torus.resz;

    //vettore velocità
    int ivx = index3d(ixp, j, k);
    int ivy = index3d(i, iyp, k);
    int ivz = index3d(i, j, izp);
    cuDoubleComplex vxraw = cuCadd(cuCmul(cuConj(psi1[ind]), psi1[ivx]), cuCmul(cuConj(psi2[ind]), psi2[ivx]));
    cuDoubleComplex vyraw = cuCadd(cuCmul(cuConj(psi1[ind]), psi1[ivy]), cuCmul(cuConj(psi2[ind]), psi2[ivy]));
    cuDoubleComplex vzraw = cuCadd(cuCmul(cuConj(psi1[ind]), psi1[ivz]), cuCmul(cuConj(psi2[ind]), psi2[inz]));

    torus.vx[ind] = arctan(vxraw)*hbar;
    torus.vy[ind] = arctan(vyraw)*hbar;
    torus.vz[ind] = arctan(vzraw)*hbar;
    
}

void ISFlow_velocity()
{
    int nb = numblock(torus.plen);
    ISFlow_velocity_1form<<<nb, THREADS_PER_BLOCK>>>(hbar);
    cudaDeviceSynchronize();
}

__global__ void GaugeTransform()
{ //perform a gauge transformation on the wavefunction --> ψ(k,t) = exp(iħθ(k,t))ψ(k,t)
    int ind = check_limit(torus.plen);
    if(ind<0) return;
    cuDoubleComplex phase = make_cuDoubleComplex(0, -1.0/torus.plen);
    cuDoubleComplex* psi1 = d_params.psi1;
    cuDoubleComplex* psi2 = d_params.psi2;
    cuDoubleComplex* q = torus.fftbuf;

    cuDoubleComplex eiq = exp_complex(cuCmul(phase, q[ind]));
    psi1[ind] = cuCmul(psi1[ind], eiq);
    psi2[ind] = cuCmul(psi2[ind], eiq);
}

void ISFlow_pressure_projection()
{
    ISFlow_velocity(hbar);
    int nb = numblock(torus.plen);
    torus_div<<<nb, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();

    Torus_Poisson_solver();

    GaugeTransform<<<nb, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
}