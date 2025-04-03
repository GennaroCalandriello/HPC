#include "const.h"
#include "complex.cu"

// Definizione della struttura torus
//La teoria la trovate nell'articolo "Schrödinger’s Smoke" di Chern, Knoppel et al.

struct Torus {
    int resx, resy, resz; //resolutions (number of samples) in each direction
    int sizex, sizey, sizez; // size of the domain in each direction
    double dx, dy, dz; // edge length of the grid cell

    double *vx;
    double *vy;
    double *vz;

    int plen;
    int yzlen;

    double *div;
    cuDoubleComplex* fftbuf;
    cufftHandle fftplan;

    double* poissonbuf;
};

__constant__ Torus torus;
Torus torus_cpu;


void torus_ds(Torus* t)
{
    t -> dx = ((double)t -> sizex) /(t -> resx);
    t -> dy = ((double)t -> sizey) /(t->resy);
    t -> dz = ((double)t -> sizez) /(t->resz);
}

__device__ __inline__ int index3d(int i, int j, int k)
{
    return (k +j*torus.resz +i*torus.yzlen);
}

__device__ __inline__ void getCoords(int i, int *x, int *y, int *z)
{
    *x = i/(torus.yzlen);
    int t = i% torus.yzlen;
    *y = t/torus.resz;
    *z = t% torus.resz;
}

__device__ inline__ int check_limit(int limit){

    int i = blockIdx.x*blockdim.x+threadIdx.x;
    if (i< limit)
        return i;
    return -1;
}


__global__ void torus_div()
{
    int normal_index = check_limit(torus.plen);
    if(normal_index <0) return;

    double dx2 = torus.dx *torus.dx;
    double dy2 = torus.dy*torus.dy;
    double dz2 = torus.dz*torus.dz;

    double* vx = torus.vx;
    double* vy = torus.vy;
    double* vz = torus.vz;

    int i, j, k;
    getCoords(normal_index, &i, &j, &k);

    int ixm = (i-1+torus.resx)%torus.resx;
    int iym = (j-1+torus.resy)%torus.resy;
    int izm = (k -1+torus.resz)%torus.resz;

    torus.div[normal_index] = (vx[normal_index] -vx[index3d(ixm, j, k)])/dx2;
    torus.div[normal_index] += (vy[normal_index] -vy[index3d(i, iym, k)])/dy2;
    torus.div[normal_index] += (vz[normal_index]-vz[index3d(i, j, izm)])/dz2;

}

__global__ void torus_div2buf()
{
    int ind = check_limit(torus.plen);
    if (ind<0) return;

    //make_cuDoubleComplex(real part, imaginary part) è una funzione nativa di CUDA
    torus.fftbuf[ind] = make_cuDoubleComplex(torus.div[ind], 0.0);
}

__global__ void torus_Poisson_kernel()
//equazione (18) appendice E sugli autovalori discreti dell'operatore di Laplace
{
    int ind = check_limit(torus.plen);
    if(ind<0) return;
    int i, j, k;
    getCoords(ind, &i, &j, &k);
    double sx = sin(pi*i/torus.resx)/torus.dx;
    double sy = sin(pi*j/torus.resy)/torus.dy;
    double sz = sin(pi*k/torus.resz)/torus.dz;

    double d = sx*sx + sy*sy + sz*sz;
    double factor =0.0f;
    if(ind > 0)
    {
        factor = -0.25/d;
    }
}

int numblock(int limit){
    return(limit+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK
}

void torus_poisson_exe()
{
    int nb = numblock(torus_cpu.plen)
    torus_Poisson_kernel<<<nb, THREADS_PER_BLOCK
    cudaDeviceSynchronize();
}
void fftcuda(cufftDoubleComplex *data)
{
    cufftExecZ2Z(torus_cpu.fftplan, data, data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
}

void ifftcuda(cufftDoubleComplex *data)
{
    cufftExecZ2Z(torus_cpu.fftplan, data, data, CUFFT_INVERSE);
    cudaDeviceSynchronize();
}
__global__ void PoissonMain()
{
    int ind = check_limit(torus.plen);
    if(ind<0) return;
    complexProd(&torus.fftbuf[ind], torus.poissonbuf[ind]);

}

void Torus_Poisson_solver()
{
    int nb = numblock(torus_cpu.plen);
    torus_div2buf<<<nb, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();

    fftcuda(torus_cpu.fftbuf);
    PoissonMain<<<nb, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();

    ifftcuda(torus_cpu.fftbuf);
}

__global__ void StaggeredSharp_ker()
{
   // scale or "sharpen" the velocity fields (vx, vy, vz) 
   //of the simulation by dividing them by their respective grid spacings
   // (dx, dy, dz). This operation ensures that the velocity components are
   // normalized with respect to the grid resolution

    int i = check_limit(torus.plen);
    if(i<0) return; 
    torus.vx[i] /= torus.dx;
    torus.vy[i] /= torus.dy;
    torus.vz[i] /= turis.dz;

}

void Torus_StaggeredSharp()
{
    int nb = numblock(torus_cpu.plen);
    StaggeredSharp_ker<<<nb, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
}

__global__ void fftshift(cufftDoubleComplex *data)
//This function is intended to perform an fftshift operation
// on a 3D array stored in data. In the context of Fourier transforms, 
//fftshift is often used to rearrange frequency-domain data so that
// the zero-frequency (DC) component is moved from the corner of the
// array to its center. This makes the frequency spectrum easier to interpret
// visually and numerically
{
  int xs = torus.resx / 2;
  int ys = torus.resy / 2;
  int zs = torus.resz / 2;
  int len = torus.plen;
  int x, y, z = 0;
  int j;

  /*if (len % 2 == 1){
    printf("Error: fftshift only supports even sized grid!\n");
    return;
  }*/

  int i = check_limit(len / 2);
  if(i<0) return;

  //for (int i=0; i<len/2; i++)
  //{
    cufftDoubleComplex temp = data[i];
    getCoords(i, &x, &y, &z);
    x = (x + xs) % torus.resx;
    y = (y + ys) % torus.resy;
    z = (z + zs) % torus.resz;
    j = index3d(x, y, z);
    data[i] = data[j];
    data[j] = temp;
  //}
}

__global__ void ifftshift(cufftDoubleComplex *data)
// Since we are only working with even-sized arrays
// ifftshift is equivalent with fftshift
{
  int xs = torus.resx / 2;
  int ys = torus.resy / 2;
  int zs = torus.resz / 2;
  int len = torus.resx * torus.resy * torus.resz;
  int x, y, z = 0;
  int j;

  if (len % 2 == 1){
    printf("Error: fftshift only supports even sized grid!\n");
    return;
  }

  for (int i=0; i<len/2; i++)
  {
    cufftDoubleComplex temp = data[i];
    getCoords(i, &x, &y, &z);
    x = (x + xs) % torus.resx;
    y = (y + ys) % torus.resy;
    z = (z + zs) % torus.resz;
    j = index3d(x, y, z);
    data[i] = data[j];
    data[j] = temp;
  }
}
