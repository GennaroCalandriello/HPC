#include "const.h"
#include <fstream>

//mouse click location
Vector2f C;
//direction and length of mouse drag
Vector2f F;
//decayrate
float global_decay_rate = DECAY_RATE;

//steps of integration: 
//1. apply external force field to the data
//2. advect the data through method of characteristics (w_1(x) =w_0(p(x, -dt))) + bilinear interpolation
//3. diffuse the data, it is: \partial_t w = \nu \nabla^2 w
// //3.1implicit method: (I-\nu *dt \nabla^2)w_1 = w_0
//4. projection step: (require a good Poisson solver)
// //4.1 \nabla^2 p = div u_1
// //4.2 u_2 = u_1 - \nabla p


/*** 
    *parameters
    @param rdx: approximation constant (?) controlla meglio

*/

// First step: apply the external force field to the data
__device__ void force(Vector2f x, Vector2f *field, Vector2f C, Vector2f F, float timestep, float r, unsigned dim) {
    float xC[2] = {x[0] - C[0], x[1] - C[1]};
    float exp = (xC[0] * xC[0] + xC[1] * xC[1]) / r;
    int i = x(0);
    int j = x(1);
    Vector2f temp = F*timestep*pow(2.718, exp)*0.001;
    field[IND(i, j, dim)] +=F*timestep*pow(2.718, exp)*0.001;
    if ((temp(0) != 0 || temp(1) !=0) && x(0) == DIM/2 && x(1) == DIM/2) {
        printf("Force: %f %f\n", temp(0), temp(1));
    }
}
//-----------------------------------------------------------------------------------------------
__device__ Vector2f bilerp(Vector2f pos, Vector2f *field, unsigned dim) {
    int i = pos(0);
    int j = pos(1);
    double dx = pos(0) - i;
    double dy = pos(1) - j;

    if (i<0 || i>=dim || j<0 || j>= dim){
        //out of bounds
        return Vector2f::Zero();
    }
    else {
        //perform bilinear interpolation
        Vector2f f00 = (i<0 || i>=dim || j<0 || j>=dim) ? Vector2f::Zero() : field[IND(i, j, dim)];
        Vector2f f01 = (i+1<0 || i+1>=dim || j < 0 || j>=dim) ? Vector2f::Zero() : field[IND(i+1, j, dim)];
        Vector2f f10 = (i<0 || i>=dim || j+1<0 || j+1>=dim) ? Vector2f::Zero() : field[IND(i, j+1, dim)];
        Vector2f f11 = (i+1<0 || i+1>=dim || j+1<0 || j+1>=dim) ? Vector2f::Zero() : field[IND(i+1, j+1, dim)];

        Vector2f f0 = (1-dx)*f00+dx*f10;
        Vector2f f1 = (1-dx)*f01+dx*f11;

        return (1-dy)*f0+dy*f1;
    }
}

//Second step: advect the data through method of characteristics
__device__ void advect(Vector2f x, Vector2f *field, Vector2f *velfield, float timestep, float rdx, unsigned dim) {
    Vector2f pos = x-timestep*rdx*velfield[IND(x(0), x(1), dim)];
    field[IND(x(0), x(1), dim)] = bilerp(pos, field, dim);
}



//------------------------------------------------------------------------------------------------------------------------
//Third step: diffuse the data
template <typename T>
__device__ void jacobi(Vector2f x, T *field, float alpha, float beta, T b, T zero, unsigned dim)
{
    int i  = (int)x(0);
    int j = (int)x(1);

    T f00 = (i-1<0 || i-1>=dim || j<0 || j>=dim) ? zero : field[IND(i-1, j, dim)];
    T f01 = (i+1<0 || i+1>=dim || j<0 || j>=dim) ? zero : field[IND(i+1, j, dim)];
    T f10 = (i<0 || i>=dim || j-1<0 || j-1>=dim) ? zero : field[IND(i, j-1, dim)];
    T f11 = (i<0 || i>=dim || j+1<0 || j+1>=dim) ? zero : field[IND(i, j+1, dim)];
    T ab = (i<0 || i>=dim || j<0 || j>=dim) ? zero : alpha*b;

    field[IND(i-1, j, dim)] = (f00+f01+f10+f11+ab)/(beta);
}

//for the pressure term
__device__ float divergence(Vector2f x, Vector2f *from, float halfrdx, unsigned dim)
{
    int i = x(0);
    int j = x(1);
    if (i<0 || i>= dim || j <0 || j>= dim)
        return 0;
    
        Vector2f wL = (i-1<0) ? Vector2f::Zero() : from[IND(i-1, j, dim)];
        Vector2f wR = (i+1>=dim) ? Vector2f::Zero() : from[IND(i+1, j, dim)];
        Vector2f wB = (j-1<0) ? Vector2f::Zero() : from[IND(i, j-1, dim)];
        Vector2f wT = (j+1>=dim) ? Vector2f::Zero() : from[IND(i, j+1, dim)];

        return halfrdx * (wR(0)-wL(0), wT(1)-wB(1));
}

//Obtain the approximate gradient of a scalar field [in this case, p].
// The gradient is calculated using the immediate neighboring value only.
__device__ Vector2f gradient(Vector2f x, float *p, float halfrdx, unsigned dim)
{
    int i = x(0);
    int j = x(1);

    if (i<0||i>= dim || j<0 || j>=dim)
        return Vector2f::Zero();
    
    float pL = (i-1<0) ? 0 : p[IND(i-1, j, dim)];
    float pR = (i+1>=dim) ? 0 : p[IND(i+1, j, dim)];
    float pB = (j-1<0) ? 0 : p[IND(i, j-1, dim)];
    float pT = (j+1>=dim) ? 0 : p[IND(i, j+1, dim)];

    return halfrdx * Vector2f(pR-pL, pT-pB);
}

//Navier Stokes kernel
__global__ void NSkernel(Vector2f *u, float *p, float rdx, float viscosity, Vector2f C, Vector2f F, float timestep, float r, unsigned dim)
{
    Vector2f x(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    //advection
    advect(x, u, u, timestep, rdx, dim);
    __syncthreads();

    //diffusion
    float alpha = rdx*rdx/(viscosity*timestep), beta = 4+alpha;
    jacobi<Vector2f>(x, u, alpha, beta, u[IND(x(0), x(1), dim)], Vector2f::Zero(), dim);
    __syncthreads();

    //force application
    force(x, u, C, F, timestep, r, dim);
    __syncthreads();

    //pressure
    alpha = -1*rdx*rdx, beta =4;
    jacobi<float>(x, p, alpha, beta, divergence(x, u, (float)(rdx/2), dim), 0, dim);
    __syncthreads();

    //u = w - grad(p)
    u[IND(x(0), x(1), dim)] -= gradient(x, p, (float)(rdx/2), dim);
    __syncthreads();

     // print state
    if (x(0) >= DIM / 2 && x(1) >= DIM / 2)
        printf("u[%.1f, %.1f] = (%f, %f)\n", x(0), x(1), u[IND(x(0), x(1), dim)](0), u[IND(x(0), x(1), dim)](1));
}

/**
 * Maps velocity vectors to a color
 * @param uc Array of RGB values for every pixel
 * @param u The velocity vector at that location
 * @param dim The maximum dimension of the field [for bound checking]
 * @authors Patrick Yevych
 */
// __global__ void clrkernel(Vector3f *uc, Vector2f *u, unsigned dim)
// {
//     Vector2f x(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
//     uc[IND(x(0), x(1), dim)] = getColor(
//         (double)u[IND(x(0), x(1), dim)].norm());
// }
// /**
//  * Decays the convection force F.
//  * @authors Patrick Yevych
//  */
void decayForce()
{
    float nx = F(0) - global_decay_rate;
    float ny = F(1) - global_decay_rate;
    nx = (nx > 0) ? nx : 0;
    ny = (ny > 0) ? ny : 0;
    F = Vector2f(nx, ny);
}

// Function to write simulation data to a file
void writeSimulationData(Vector2f *velocityField, unsigned dim, int timestep) {
    std::ofstream outfile("fluid_simulation_output.txt", std::ios::app);
    outfile << "Timestep: " << timestep << "\n";
    for (unsigned i = 0; i < dim; ++i) {
        for (unsigned j = 0; j < dim; ++j) {
            Vector2f vel = velocityField[IND(i, j, dim)];
            outfile << "(" << vel(0) << ", " << vel(1) << ") ";
        }
        outfile << "\n";
    }
    outfile << "\n";
    outfile.close();
}

int main(int argc, char **argv)
{
    //quarter of second timestep
    float timestep = TIMESTEP;
    //dimension of velocity field
    unsigned dim = DIM;
    // how may pixels a cell of the vector field represents
    float rdx = RES/dim;
    //fluid viscosity
    float viscosity = VISCOSITY;
    //decay rate
    global_decay_rate = DECAY_RATE;
    //force radius
    float r = RADIUS;

    //user provided simulation parameters
    // if (argc ==5){
    //     timestep = atof(argv[1]);
    //     viscosity = atof(argv[2]);
    //     global_decay_rate = atof(argv[3]);
    //     r = atof(argv[4]);
    // } else if (argc != 1){
    //     printf("Usage: %s [timestep] [viscosity] [decay rate] [force radius]\n", argv[0]);
    //     return 1;
    // }

    //force parameters
    // C = Vector2f::Zero(); F = Vector2f::Zero();
    // Initial force parameters
    C = Vector2f(dim / 2, dim / 2); // Set the initial force center to the middle of the domain
    F = Vector2f(10.0, 10.0); // Set a non-zero initial force to induce movement

    //velocity vector field and pressure scalar field p
    Vector2f *u, *dev_u;
    float *p, *dev_p;

    //Allocate memory for host
     u = (Vector2f *)malloc(dim*dim*sizeof(Vector2f));
     p = (float *)malloc(dim*dim*sizeof(float));

     //Initialize host fields
     for (unsigned i =0; i<dim*dim; i++){
         u[i] = Vector2f::Zero();
         p[i] = 0;
     }

     // Initialize host fields with Gaussian distribution
    // float sigma = dim / 10.0;  // Standard deviation for Gaussian
    // float center = dim / 2.0;
    // for (unsigned j = 0; j < dim; ++j) {
    //     for (unsigned i = 0; i < dim; ++i) {
    //         float x = i - center;
    //         float y = j - center;
    //         float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
    //         u[IND(i, j, dim)] = Vector2f(value, value);  // Set initial velocity with Gaussian
    //         p[IND(i, j, dim)] = 0.3;
    //     }
    // }
    // writeSimulationData(u, dim, 0);

     //allocate memory for device
     cudaMalloc((void**)&dev_u, dim*dim*sizeof(Vector2f));
     cudaMalloc((void**)&dev_p, dim * dim*sizeof(float));

     //copy initial value to device
     cudaMemcpy(dev_u, u, dim*dim*sizeof(Vector2f), cudaMemcpyHostToDevice);
     cudaMemcpy(dev_p, p, dim*dim*sizeof(float), cudaMemcpyHostToDevice);

     dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
     dim3 blocks(dim/BLOCKSIZEX, dim/BLOCKSIZEY);

     //run simulation for a certain # of timesteps
     int num_timesteps = 20;
     for (int t =0; t<num_timesteps; t++){
        // Forza variabile nel tempo per simulare l'interazione
        C = Vector2f(dim / 2 + 5 * sin(timestep * 0.1), dim / 2 + 5 * cos(timestep * 0.1));
        F = Vector2f(10.0 * sin(timestep * 0.05), 10.0 * cos(timestep * 0.05));

        NSkernel<<<blocks, threads>>>(dev_u, dev_p, rdx, viscosity, C, F, timestep, r, dim);
        cudaDeviceSynchronize();
        // clrkernel<<<blocks, threads>>>(dev_uc, dev_u, dim);
        // cudaDeviceSynchronize();
        cudaMemcpy(u, dev_u, dim*dim*sizeof(Vector2f), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        writeSimulationData(u, dim, t);
        decayForce();
     }

    free(u);
    free(p);
    cudaFree(dev_u);
    cudaFree(dev_p);
}