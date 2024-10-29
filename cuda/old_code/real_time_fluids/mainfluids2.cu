
// mainfluids_visualization.cu
#include "functions.h"

// Viridis colormap data (You need to include all 256 entries)
// For brevity, only a few entries are shown here. Include all in your code.

// Global variables
Vector2f C;
Vector2f F;
// Vector2f C1;
// Vector2f F1;
float global_decay_rate = DECAY_RATE;

// Function to decay the force
void decayForce() {
    float nx = F.x - global_decay_rate;
    float ny = F.y - global_decay_rate;
    nx = (nx > 0) ? nx : 0.0f;
    ny = (ny > 0) ? ny : 0.0f;
    F = Vector2f(nx, ny);
}

// First step: apply the external force field to the data
__device__ void force(Vector2f x, Vector2f* field, Vector2f C, Vector2f F, float timestep, float r, unsigned dim) {
    if (periodic == 1){
        // Apply periodic wrapping to the position
        // Apply periodic wrapping to the center position
        Vector2f xC = x - C;
        xC.x = fmodf(xC.x + dim / 2.0f, dim) - dim / 2.0f;
        xC.y = fmodf(xC.y + dim / 2.0f, dim) - dim / 2.0f;

        float exp_val = (xC.x * xC.x + xC.y * xC.y) / r;
        float factor = timestep * expf(-exp_val) * 0.001f;
        Vector2f temp = F * factor;

        int idx = IND(static_cast<int>(x.x), static_cast<int>(x.y), dim);
        field[idx] += temp;
    }

    else {
        float xC[2] = { x.x - C.x, x.y - C.y };
        float exp_val = (xC[0] * xC[0] + xC[1] * xC[1]) / r;
        int i = static_cast<int>(x.x);
        int j = static_cast<int>(x.y);
        float factor = timestep * expf(-exp_val) * 0.001f;
        Vector2f temp = F * factor;
        if (i >= 0 && i < dim && j >= 0 && j < dim) {
            field[IND(i, j, dim)] += temp;
        }
    }
}

// Bilinear interpolation
/***
If the interpolated position falls outside the simulation grid, the function returns a zero vector, 
imposing a zero velocity at the boundaries.
*/

// Second step: advect the data through method of characteristics
__device__ void advect(Vector2f x, Vector2f* field, Vector2f* velfield, float timestep, float rdx, unsigned dim) {
    float dt0 = timestep * rdx;
    // HEART OF SIM ---> It solves the nonlinearity in the NS equations. Two steps:
    // 1. Backtrace the particle to its original position
    // 2. Interpolate the field at the backtraced position
    //The backtracing is computed using a fourth-order Runge-Kutta integration scheme.
    // Compute k1
    Vector2f k1 = velocityAt(x, velfield, dim);
    Vector2f x1 = x - 0.5f * dt0 * k1;

    // Compute k2
    Vector2f k2 = velocityAt(x1, velfield, dim);
    Vector2f x2 = x - 0.5f * dt0 * k2;

    // Compute k3
    Vector2f k3 = velocityAt(x2, velfield, dim);
    Vector2f x3 = x - dt0 * k3;

    // Compute k4
    Vector2f k4 = velocityAt(x3, velfield, dim);

    // Combine to get final position
    Vector2f pos = x - (dt0 / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    if (periodic ==1){
        // Apply periodic wrapping to the position
    pos.x = fmodf(pos.x + dim, dim);
    pos.y = fmodf(pos.y + dim, dim);

    // Interpolate the field at the backtraced position
    int idx = IND(static_cast<int>(x.x), static_cast<int>(x.y), dim);
    field[idx] = bilinearInterpolation(pos, field, dim);
    }

    else {
    // Interpolate the field at the backtraced position
    field[IND(static_cast<int>(x.x), static_cast<int>(x.y), dim)] = bilinearInterpolation(pos, field, dim);
    }
}

// Third step: diffuse the data
template <typename T>
__device__ void jacobi(Vector2f x, T* field, float alpha, float beta, T b, T zero, unsigned dim)
{ 
    int i = (int)x.x; // or x(0) if x is accessed that way
    int j = (int)x.y; // or x(1)
    if (periodic ==1) {
        // Use periodic indexing
    int iL = periodicIndex(i - 1, dim);
    int iR = periodicIndex(i + 1, dim);
    int jB = periodicIndex(j - 1, dim);
    int jT = periodicIndex(j + 1, dim);

    // Neighbor values
    T fL = field[IND(iL, j, dim)];
    T fR = field[IND(iR, j, dim)];
    T fB = field[IND(i, jB, dim)];
    T fT = field[IND(i, jT, dim)];

    // Update the current grid point
    field[IND(i, j, dim)] = (fL + fR + fB + fT + alpha * b) / beta;
    }

    else {


    // Left neighbor
    T f00 = (i - 1 < 0 || i - 1 >= dim || j < 0 || j >= dim) ? zero : field[IND(i - 1, j, dim)];
    // Right neighbor
    T f01 = (i + 1 < 0 || i + 1 >= dim || j < 0 || j >= dim) ? zero : field[IND(i + 1, j, dim)];
    // Bottom neighbor
    T f10 = (i < 0 || i >= dim || j - 1 < 0 || j - 1 >= dim) ? zero : field[IND(i, j - 1, dim)];
    // Top neighbor
    T f11 = (i < 0 || i >= dim || j + 1 < 0 || j + 1 >= dim) ? zero : field[IND(i, j + 1, dim)];

    // Source term with boundary check
    T ab = (i < 0 || i >= dim || j < 0 || j >= dim) ? zero : alpha * b;

    // Update the current grid point
    field[IND(i, j, dim)] = (f00 + f01 + f10 + f11 + ab) / beta;}
}

// Compute divergence
__device__ float divergence(Vector2f x, Vector2f* from, float halfrdx, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return 0.0f;

    if (periodic ==1){
        // Use periodic indexing
        int iL = periodicIndex(i - 1, dim);
        int iR = periodicIndex(i + 1, dim);
        int jB = periodicIndex(j - 1, dim);
        int jT = periodicIndex(j + 1, dim);

        Vector2f wL = from[IND(iL, j, dim)];
        Vector2f wR = from[IND(iR, j, dim)];
        Vector2f wB = from[IND(i, jB, dim)];
        Vector2f wT = from[IND(i, jT, dim)];

        float div = halfrdx * ((wR.x - wL.x) + (wT.y - wB.y));
        return div;
        }

    else {

        Vector2f wL = (i > 0) ? from[IND(i - 1, j, dim)] : Vector2f::Zero();
        Vector2f wR = (i < dim - 1) ? from[IND(i + 1, j, dim)] : Vector2f::Zero();
        Vector2f wB = (j > 0) ? from[IND(i, j - 1, dim)] : Vector2f::Zero();
        Vector2f wT = (j < dim - 1) ? from[IND(i, j + 1, dim)] : Vector2f::Zero();

        float div = halfrdx * ((wR.x - wL.x) + (wT.y - wB.y));
        return div;
    } 
}

// Obtain the approximate gradient of a scalar field
__device__ Vector2f gradient(Vector2f x, float* p, float halfrdx, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return Vector2f::Zero();

    if (periodic == 1){
        int iL = periodicIndex(i - 1, dim);
        int iR = periodicIndex(i + 1, dim);
        int jB = periodicIndex(j - 1, dim);
        int jT = periodicIndex(j + 1, dim);

        float pL = p[IND(iL, j, dim)];
        float pR = p[IND(iR, j, dim)];
        float pB = p[IND(i, jB, dim)];
        float pT = p[IND(i, jT, dim)];

        Vector2f grad;
        grad.x = halfrdx * (pR - pL);
        grad.y = halfrdx * (pT - pB);
        return grad;
    }
    else {
    
    float pL = (i > 0) ? p[IND(i - 1, j, dim)] : 0.0f;
    float pR = (i < dim - 1) ? p[IND(i + 1, j, dim)] : 0.0f;
    float pB = (j > 0) ? p[IND(i, j - 1, dim)] : 0.0f;
    float pT = (j < dim - 1) ? p[IND(i, j + 1, dim)] : 0.0f;

    Vector2f grad;
    grad.x = halfrdx * (pR - pL);
    grad.y = halfrdx * (pT - pB);
    return grad; 
    }
}

// Navier-Stokes kernel
__global__ void NSkernel(Vector2f* u, float* p, float* c, float c_ambient, float gravity, float betabouyancy, float rdx, float viscosity, Vector2f C, Vector2f F, float timestep, float r, unsigned dim) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int boolvortex = VORTEX;
    int booljet = FLUID_INJ;

    if (i >= dim || j >= dim)
        return;

    Vector2f x(static_cast<float>(i), static_cast<float>(j));

    
    // Force application
    force(x, u, C, F, timestep, r, dim);
    __syncthreads();

    // Advection
    advect(x, u, u, timestep, rdx, dim);
    __syncthreads();

    if (boolvortex == 1)
        applyVortex(u, F, dim);
        __syncthreads();

    if (booljet == 1)
        injectFluid(u, dim);
        __syncthreads();
 
   
    // Diffusion
    float alpha = rdx * rdx / (viscosity * timestep);
    float beta = 4.0f + alpha;
    for (int iter =0; iter<NUM_OF_DIFFUSION_STEPS; iter++){
        jacobi<Vector2f>(x, u, alpha, beta, u[IND(i, j, dim)], Vector2f::Zero(), dim);
        __syncthreads(); }

    // __syncthreads();

    // Pressure calculation
    // // u = w - nabla p
    alpha = -rdx * rdx;
    beta = 4.0f;
    float div = divergence(x, u, 0.5f * rdx, dim);
    jacobi<float>(x, p, alpha, beta, div, 0.0f, dim);
    __syncthreads();

    // Pressure gradient subtraction
    Vector2f grad_p = gradient(x, p, 0.5f * rdx, dim);
    u[IND(i, j, dim)] -= grad_p;
    __syncthreads();

    if (advect_scalar_bool==1 ) {
    //     // Advection of scalar field c
    advectScalar(x, c, u, timestep, rdx, dim);
    __syncthreads();

    // Diffusion of scalar field c
    diffuseScalar(x, c, diffusion_rate, timestep, rdx, dim);
    __syncthreads();

        // **Applica la forza di galleggiamento basata su c**
    applyBuoyancy(x, u, c, c_ambient, betabouyancy, gravity, dim);
    __syncthreads(); 
    }
}

int main(int argc, char** argv) {
    float c_ambient = 0.0f;    // Valore ambientale di c
    float betabouyancy = BETA_BOUYANCY;         // Coefficiente di espansione (regola l'influenza di c su u)
    float gravity = -9.81f;  
    int framecount = 0;

    // Define wind direction and magnitude
    Vector2f windDirection(1.0f, 2.0f); // Wind blowing to the right
    float windMagnitude = 0.1f;         // Adjust the magnitude as needed
    Vector2f windForce = windDirection * windMagnitude;

    // Force parameters
    // C = Vector2f(static_cast<float>(dim) / 2.0f, static_cast<float>(dim) / 2.0f); // Center of the domain
    // C = Vector2f(0, 0);
    // F = Vector2f(0.0f, 0.8f); // Initial force

    // Velocity vector field and pressure scalar field
    Vector2f* u = (Vector2f*)malloc(dim * dim * sizeof(Vector2f));
    float* p = (float*)malloc(dim * dim * sizeof(float));
    float* c = (float*)malloc(dim*dim*sizeof(float));

    // // Initialize host fields
    for (unsigned i = 0; i < dim * dim; i++) {
        u[i] = Vector2f::Zero();
        p[i] = 0.0f;
        c[i] = 0.0001f;
    }
   // Gaussian initial pressure
    // for (unsigned i = 0; i < dim; i++) {
    //     for (unsigned j = 0; j < dim; j++) {
    //         float x = static_cast<float>(i) - dim / 2.0f;
    //         float y = static_cast<float>(j) - dim / 2.0f;
    //         float r = sqrtf(x * x + y * y);
    //         float stdev = 5.0f;
    //         p[IND(i, j, dim)] =4* expf(-r * r / (2.0f * stdev * stdev));
    //     }}

    // Device memory allocation
    Vector2f* dev_u;
    float* dev_p;
    float* dev_c;

    cudaMalloc((void**)&dev_c, dim*dim*sizeof(float));
    cudaMalloc((void**)&dev_u, dim * dim * sizeof(Vector2f));
    cudaMalloc((void**)&dev_p, dim * dim * sizeof(float));

    // Copy initial values to device
    cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, dim*dim*sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for color field
    Vector3f* colorField = (Vector3f*)malloc(dim * dim * sizeof(Vector3f));
    Vector3f* dev_colorField;
    cudaMalloc((void**)&dev_colorField, dim * dim * sizeof(Vector3f));

    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(RES, RES, "Navier-Stokes Simulation", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Disable VSync
    glfwSwapInterval(0);

    // Set up the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, RES, 0, RES, -1, 1);

    // Set up the modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Create a texture
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // CUDA grid and block dimensions
    dim3 threads(BLOCKSIZEX, BLOCKSIZEY);
    dim3 blocks((dim + BLOCKSIZEX - 1) / BLOCKSIZEX, (dim + BLOCKSIZEY - 1) / BLOCKSIZEY);

    // Simulation loop
    while (!glfwWindowShouldClose(window)) {
        // Time step
        // Execute the Navier-Stokes kernel
        // Force parameters
        
        //add F as a periodic func
        float time = glfwGetTime();

        if (PERIODIC_FORCE == 1) {
            F = Vector2f(magnitude * sin(time), 0.0f); // Initial force
        }
        // F1 = Vector2f(magnitude * sin(time), magnitude * cos(time)); // Initial force
        C = Vector2f(dim / 2.0f + 50.0f * sinf(glfwGetTime()), dim / 2.0f);

        // C = Vector2f(static_cast<float>(dim)/3, static_cast<float>(dim) / 2.0f) ; // Center of the domain
        NSkernel<<<blocks, threads>>>(dev_u, dev_p, dev_c,c_ambient, gravity, betabouyancy, rdx, viscosity, C, F, timestep, r, dim);
     
        // injectFluid<<<blocks, threads>>>(dev_u, dim);
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error after NSkernel: %s\n", cudaGetErrorString(err));
            return 1;
        }

        cudaDeviceSynchronize();

        // Map the velocity field to colors
        
        framecount++;
        if (framecount % RENDERING == 0) {
            colorKernel<<<blocks, threads>>>(dev_colorField, dev_u, dim);
        

            // Check for CUDA errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error after colorKernel: %s\n", cudaGetErrorString(err));
                return 1;
            }

            cudaDeviceSynchronize();

            // Copy color data from device to host
            cudaMemcpy(colorField, dev_colorField, dim * dim * sizeof(Vector3f), cudaMemcpyDeviceToHost);

            // Update the texture
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, dim, dim, 0, GL_RGB, GL_FLOAT, colorField);

            // Render the texture
            glClear(GL_COLOR_BUFFER_BIT);

            glEnable(GL_TEXTURE_2D);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(0, 0);
            glTexCoord2f(1, 0); glVertex2f(RES, 0);
            glTexCoord2f(1, 1); glVertex2f(RES, RES);
            glTexCoord2f(0, 1); glVertex2f(0, RES);
            glEnd();
            glDisable(GL_TEXTURE_2D);

            // Swap buffers
            glfwSwapBuffers(window);

            // Poll for and process events
            glfwPollEvents();
        }
        // Decay the force
        // decayForce();

        // Optionally update force parameters
        // For example, to move the force center or change its magnitude
        // C = Vector2f(dim / 2.0f + 50.0f * sinf(glfwGetTime()), dim / 2.0f);
        // F = Vector2f(10.0f, 10.0f);
    }

    // Free memory
    free(u);
    free(p);
    free(colorField);
    cudaFree(dev_u);
    cudaFree(dev_p);
    cudaFree(dev_colorField);

    // Terminate GLFW
    glfwTerminate();

    return 0;
}

