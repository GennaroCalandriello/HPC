// mainfluids_visualization.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "const.h"

// Include constants and vector structures

// Include GLFW
#include "glfw-3.4.bin.WIN64/include/GLFW//glfw3.h"

// Simulation parameters
#define TIMESTEP 0.3f
#define DIM 1000
#define RES 1000
#define VISCOSITY 0.1f
#define RADIUS (DIM * DIM)
#define DECAY_RATE 1.0f
#define NUM_TIMESTEPS 300
#define MAX_VELOCITY 3.0f  // Adjust as needed for normalization

// CUDA kernel parameters
#define BLOCKSIZEY 16
#define BLOCKSIZEX 16

#define IND(x, y, d) int((y) * (d) + (x))
#define CLAMP(x) ((x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x))



// Viridis colormap data (You need to include all 256 entries)
// For brevity, only a few entries are shown here. Include all in your code.
__device__ __constant__ float viridis_colormap[256][3] = {
    {0.267004, 0.004874, 0.329415}, {0.268510, 0.009605, 0.335427}, {0.269944, 0.014625, 0.341379}, {0.271305, 0.019942, 0.347269}, {0.272594, 0.025563, 0.353093}, {0.273809, 0.031497, 0.358853}, {0.274952, 0.037752, 0.364543}, {0.276022, 0.044167, 0.370164}, {0.277018, 0.050344, 0.375715}, {0.277941, 0.056324, 0.381191}, {0.278791, 0.062145, 0.386592}, {0.279566, 0.067836, 0.391917}, {0.280267, 0.073417, 0.397163}, {0.280894, 0.078907, 0.402329}, {0.281446, 0.084320, 0.407414}, {0.281924, 0.089666, 0.412415}, {0.282327, 0.094955, 0.417331}, {0.282656, 0.100196, 0.422160}, {0.282910, 0.105393, 0.426902}, {0.283091, 0.110553, 0.431554}, {0.283197, 0.115680, 0.436115}, {0.283229, 0.120777, 0.440584}, {0.283187, 0.125848, 0.444960}, {0.283072, 0.130895, 0.449241}, {0.282884, 0.135920, 0.453427}, {0.282623, 0.140926, 0.457517}, {0.282290, 0.145912, 0.461510}, {0.281887, 0.150881, 0.465405}, {0.281412, 0.155834, 0.469201}, {0.280868, 0.160771, 0.472899}, {0.280255, 0.165693, 0.476498}, {0.279574, 0.170599, 0.479997}, {0.278826, 0.175490, 0.483397}, {0.278012, 0.180367, 0.486697}, {0.277134, 0.185228, 0.489898}, {0.276194, 0.190074, 0.493001}, {0.275191, 0.194905, 0.496005}, {0.274128, 0.199721, 0.498911}, {0.273006, 0.204520, 0.501721}, {0.271828, 0.209303, 0.504434}, {0.270595, 0.214069, 0.507052}, {0.269308, 0.218818, 0.509577}, {0.267968, 0.223549, 0.512008}, {0.266580, 0.228262, 0.514349}, {0.265145, 0.232956, 0.516599}, {0.263663, 0.237631, 0.518762}, {0.262138, 0.242286, 0.520837}, {0.260571, 0.246922, 0.522828}, {0.258965, 0.251537, 0.524736}, {0.257322, 0.256130, 0.526563}, {0.255645, 0.260703, 0.528312}, {0.253935, 0.265254, 0.529983}, {0.252194, 0.269783, 0.531579}, {0.250425, 0.274290, 0.533103}, {0.248629, 0.278775, 0.534556}, {0.246811, 0.283237, 0.535941}, {0.244972, 0.287675, 0.537260}, {0.243113, 0.292092, 0.538516}, {0.241237, 0.296485, 0.539709}, {0.239346, 0.300855, 0.540844}, {0.237441, 0.305202, 0.541921}, {0.235526, 0.309527, 0.542944}, {0.233603, 0.313828, 0.543914}, {0.231674, 0.318106, 0.544834}, {0.229739, 0.322361, 0.545706}, {0.227802, 0.326594, 0.546532}, {0.225863, 0.330805, 0.547314}, {0.223925, 0.334994, 0.548053}, {0.221989, 0.339161, 0.548752}, {0.220057, 0.343307, 0.549413}, {0.218130, 0.347432, 0.550038}, {0.216210, 0.351535, 0.550627}, {0.214298, 0.355619, 0.551184}, {0.212395, 0.359683, 0.551710}, {0.210503, 0.363727, 0.552206}, {0.208623, 0.367752, 0.552675}, {0.206756, 0.371758, 0.553117}, {0.204903, 0.375746, 0.553533}, {0.203063, 0.379716, 0.553925}, {0.201239, 0.383670, 0.554294}, {0.199430, 0.387607, 0.554642}, {0.197636, 0.391528, 0.554969}, {0.195860, 0.395433, 0.555276}, {0.194100, 0.399323, 0.555565}, {0.192357, 0.403199, 0.555836}, {0.190631, 0.407061, 0.556089}, {0.188923, 0.410910, 0.556326}, {0.187231, 0.414746, 0.556547}, {0.185556, 0.418570, 0.556753}, {0.183898, 0.422383, 0.556944}, {0.182256, 0.426184, 0.557120}, {0.180629, 0.429975, 0.557282}, {0.179019, 0.433756, 0.557430}, {0.177423, 0.437527, 0.557565}, {0.175841, 0.441290, 0.557685}, {0.174274, 0.445044, 0.557792}, {0.172719, 0.448791, 0.557885}, {0.171176, 0.452530, 0.557965}, {0.169646, 0.456262, 0.558030}, {0.168126, 0.459988, 0.558082}, {0.166617, 0.463708, 0.558119}, {0.165117, 0.467423, 0.558141}, {0.163625, 0.471133, 0.558148}, {0.162142, 0.474838, 0.558140}, {0.160665, 0.478540, 0.558115}, {0.159194, 0.482237, 0.558073}, {0.157729, 0.485932, 0.558013}, {0.156270, 0.489624, 0.557936}, {0.154815, 0.493313, 0.557840}, {0.153364, 0.497000, 0.557724}, {0.151918, 0.500685, 0.557587}, {0.150476, 0.504369, 0.557430}, {0.149039, 0.508051, 0.557250}, {0.147607, 0.511733, 0.557049}, {0.146180, 0.515413, 0.556823}, {0.144759, 0.519093, 0.556572}, {0.143343, 0.522773, 0.556295}, {0.141935, 0.526453, 0.555991}, {0.140536, 0.530132, 0.555659}, {0.139147, 0.533812, 0.555298}, {0.137770, 0.537492, 0.554906}, {0.136408, 0.541173, 0.554483}, {0.135066, 0.544853, 0.554029}, {0.133743, 0.548535, 0.553541}, {0.132444, 0.552216, 0.553018}, {0.131172, 0.555899, 0.552459}, {0.129933, 0.559582, 0.551864}, {0.128729, 0.563265, 0.551229}, {0.127568, 0.566949, 0.550556}, {0.126453, 0.570633, 0.549841}, {0.125394, 0.574318, 0.549086}, {0.124395, 0.578002, 0.548287}, {0.123463, 0.581687, 0.547445}, {0.122606, 0.585371, 0.546557}, {0.121831, 0.589055, 0.545623}, {0.121148, 0.592739, 0.544641}, {0.120565, 0.596422, 0.543611}, {0.120092, 0.600104, 0.542530}, {0.119738, 0.603785, 0.541400}, {0.119512, 0.607464, 0.540218}, {0.119423, 0.611141, 0.538982}, {0.119483, 0.614817, 0.537692}, {0.119699, 0.618490, 0.536347}, {0.120081, 0.622161, 0.534946}, {0.120638, 0.625828, 0.533488}, {0.121380, 0.629492, 0.531973}, {0.122312, 0.633153, 0.530398}, {0.123444, 0.636809, 0.528763}, {0.124780, 0.640461, 0.527068}, {0.126326, 0.644107, 0.525311}, {0.128087, 0.647749, 0.523491}, {0.130067, 0.651384, 0.521608}, {0.132268, 0.655014, 0.519661}, {0.134692, 0.658636, 0.517649}, {0.137339, 0.662252, 0.515571}, {0.140210, 0.665859, 0.513427}, {0.143303, 0.669459, 0.511215}, {0.146616, 0.673050, 0.508936}, {0.150148, 0.676631, 0.506589}, {0.153894, 0.680203, 0.504172}, {0.157851, 0.683765, 0.501686}, {0.162016, 0.687316, 0.499129}, {0.166383, 0.690856, 0.496502}, {0.170948, 0.694384, 0.493803}, {0.175707, 0.697900, 0.491033}, {0.180653, 0.701402, 0.488189}, {0.185783, 0.704891, 0.485273}, {0.191090, 0.708366, 0.482284}, {0.196571, 0.711827, 0.479221}, {0.202219, 0.715272, 0.476084}, {0.208030, 0.718701, 0.472873}, {0.214000, 0.722114, 0.469588}, {0.220124, 0.725509, 0.466226}, {0.226397, 0.728888, 0.462789}, {0.232815, 0.732247, 0.459277}, {0.239374, 0.735588, 0.455688}, {0.246070, 0.738910, 0.452024}, {0.252899, 0.742211, 0.448284}, {0.259857, 0.745492, 0.444467}, {0.266941, 0.748751, 0.440573}, {0.274149, 0.751988, 0.436601}, {0.281477, 0.755203, 0.432552}, {0.288921, 0.758394, 0.428426}, {0.296479, 0.761561, 0.424223}, {0.304148, 0.764704, 0.419943}, {0.311925, 0.767822, 0.415586}, {0.319809, 0.770914, 0.411152}, {0.327796, 0.773980, 0.406640}, {0.335885, 0.777018, 0.402049}, {0.344074, 0.780029, 0.397381}, {0.352360, 0.783011, 0.392636}, {0.360741, 0.785964, 0.387814}, {0.369214, 0.788888, 0.382914}, {0.377779, 0.791781, 0.377939}, {0.386433, 0.794644, 0.372886}, {0.395174, 0.797475, 0.367757}, {0.404001, 0.800275, 0.362552}, {0.412913, 0.803041, 0.357269}, {0.421908, 0.805774, 0.351910}, {0.430983, 0.808473, 0.346476}, {0.440137, 0.811138, 0.340967}, {0.449368, 0.813768, 0.335384}, {0.458674, 0.816363, 0.329727}, {0.468053, 0.818921, 0.323998}, {0.477504, 0.821444, 0.318195}, {0.487026, 0.823929, 0.312321}, {0.496615, 0.826376, 0.306377}, {0.506271, 0.828786, 0.300362}, {0.515992, 0.831158, 0.294279}, {0.525776, 0.833491, 0.288127}, {0.535621, 0.835785, 0.281908}, {0.545524, 0.838039, 0.275626}, {0.555484, 0.840254, 0.269281}, {0.565498, 0.842430, 0.262877}, {0.575563, 0.844566, 0.256415}, {0.585678, 0.846661, 0.249897}, {0.595839, 0.848717, 0.243329}, {0.606045, 0.850733, 0.236712}, {0.616293, 0.852709, 0.230052}, {0.626579, 0.854645, 0.223353}, {0.636902, 0.856542, 0.216620}, {0.647257, 0.858400, 0.209861}, {0.657642, 0.860219, 0.203082}, {0.668054, 0.861999, 0.196293}, {0.678489, 0.863742, 0.189503}, {0.688944, 0.865448, 0.182725}, {0.699415, 0.867117, 0.175971}, {0.709898, 0.868751, 0.169257}, {0.720391, 0.870350, 0.162603}, {0.730889, 0.871916, 0.156029}, {0.741388, 0.873449, 0.149561}, {0.751884, 0.874951, 0.143228}, {0.762373, 0.876424, 0.137064}, {0.772852, 0.877868, 0.131109}, {0.783315, 0.879285, 0.125405}, {0.793760, 0.880678, 0.120005}, {0.804182, 0.882046, 0.114965}, {0.814576, 0.883393, 0.110347}, {0.824940, 0.884720, 0.106217}, {0.835270, 0.886029, 0.102646}, {0.845561, 0.887322, 0.099702}, {0.855810, 0.888601, 0.097452}, {0.866013, 0.889868, 0.095953}, {0.876168, 0.891125, 0.095250}, {0.886271, 0.892374, 0.095374}, {0.896320, 0.893616, 0.096335}, {0.906311, 0.894855, 0.098125}, {0.916242, 0.896091, 0.100717}, {0.926106, 0.897330, 0.104071}, {0.935904, 0.898570, 0.108131}, {0.945636, 0.899815, 0.112838}, {0.955300, 0.901065, 0.118128}, {0.964894, 0.902323, 0.123941}, {0.974417, 0.903590, 0.130215}, {0.983868, 0.904867, 0.136897}, {0.993248, 0.906157, 0.143936}
};

// Global variables
Vector2f C;
Vector2f F;
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

// Bilinear interpolation
/***
If the interpolated position falls outside the simulation grid, the function returns a zero vector, 
imposing a zero velocity at the boundaries.
*/
__device__ Vector2f bilinearInterpolation(Vector2f pos, Vector2f* field, unsigned dim) {
    int i = static_cast<int>(pos.x);
    int j = static_cast<int>(pos.y);
    float dx = pos.x - i;
    float dy = pos.y - j;

    if (i < 0 || i >= dim - 1 || j < 0 || j >= dim - 1) {
        // Out of bounds
        return Vector2f::Zero();
    }
    else {
        // Perform bilinear interpolation
        Vector2f f00 = field[IND(i, j, dim)];
        Vector2f f10 = field[IND(i + 1, j, dim)];
        Vector2f f01 = field[IND(i, j + 1, dim)];
        Vector2f f11 = field[IND(i + 1, j + 1, dim)];

        Vector2f f0 = f00 * (1.0f - dx) + f10 * dx;
        Vector2f f1 = f01 * (1.0f - dx) + f11 * dx;

        return f0 * (1.0f - dy) + f1 * dy;
    }
}

// Second step: advect the data through method of characteristics
__device__ void advect(Vector2f x, Vector2f* field, Vector2f* velfield, float timestep, float rdx, unsigned dim) {
    int idx = IND(static_cast<int>(x.x), static_cast<int>(x.y), dim);
    Vector2f velocity = velfield[idx];
    Vector2f pos = x - velocity * (timestep * rdx);
    field[idx] = bilinearInterpolation(pos, field, dim);
}

// Third step: diffuse the data
template <typename T>
__device__ void jacobi(Vector2f x, T* field, float alpha, float beta, T b, T zero, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    T f_left = (i > 0) ? field[IND(i - 1, j, dim)] : zero;
    T f_right = (i < dim - 1) ? field[IND(i + 1, j, dim)] : zero;
    T f_down = (j > 0) ? field[IND(i, j - 1, dim)] : zero;
    T f_up = (j < dim - 1) ? field[IND(i, j + 1, dim)] : zero;
    T ab = alpha * b;

    field[IND(i, j, dim)] = (f_left + f_right + f_down + f_up + ab) / beta;
}

// Compute divergence
__device__ float divergence(Vector2f x, Vector2f* from, float halfrdx, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return 0.0f;

    Vector2f wL = (i > 0) ? from[IND(i - 1, j, dim)] : Vector2f::Zero();
    Vector2f wR = (i < dim - 1) ? from[IND(i + 1, j, dim)] : Vector2f::Zero();
    Vector2f wB = (j > 0) ? from[IND(i, j - 1, dim)] : Vector2f::Zero();
    Vector2f wT = (j < dim - 1) ? from[IND(i, j + 1, dim)] : Vector2f::Zero();

    float div = halfrdx * ((wR.x - wL.x) + (wT.y - wB.y));
    return div;
}

// Obtain the approximate gradient of a scalar field
__device__ Vector2f gradient(Vector2f x, float* p, float halfrdx, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    if (i < 0 || i >= dim || j < 0 || j >= dim)
        return Vector2f::Zero();

    float pL = (i > 0) ? p[IND(i - 1, j, dim)] : 0.0f;
    float pR = (i < dim - 1) ? p[IND(i + 1, j, dim)] : 0.0f;
    float pB = (j > 0) ? p[IND(i, j - 1, dim)] : 0.0f;
    float pT = (j < dim - 1) ? p[IND(i, j + 1, dim)] : 0.0f;

    Vector2f grad;
    grad.x = halfrdx * (pR - pL);
    grad.y = halfrdx * (pT - pB);
    return grad;
}

// Navier-Stokes kernel
__global__ void NSkernel(Vector2f* u, float* p, float rdx, float viscosity, Vector2f C, Vector2f F, float timestep, float r, unsigned dim) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dim || j >= dim)
        return;

    Vector2f x(static_cast<float>(i), static_cast<float>(j));

    // Advection
    advect(x, u, u, timestep, rdx, dim);
    __syncthreads();

    // Diffusion
    float alpha = rdx * rdx / (viscosity * timestep);
    float beta = 4.0f + alpha;
    jacobi<Vector2f>(x, u, alpha, beta, u[IND(i, j, dim)], Vector2f::Zero(), dim);
    __syncthreads();

    // Force application
    force(x, u, C, F, timestep, r, dim);
    __syncthreads();

    // Pressure calculation
    alpha = -rdx * rdx;
    beta = 4.0f;
    float div = divergence(x, u, 0.5f * rdx, dim);
    jacobi<float>(x, p, alpha, beta, div, 0.0f, dim);
    __syncthreads();

    // Pressure gradient subtraction
    Vector2f grad_p = gradient(x, p, 0.5f * rdx, dim);
    u[IND(i, j, dim)] -= grad_p;
    __syncthreads();
}

// Map velocity magnitude to color using the colormap
__global__ void colorKernel(Vector3f* colorField, Vector2f* velocityField, unsigned dim) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dim || j >= dim)
        return;

    int idx = IND(i, j, dim);
    float velocityMagnitude = velocityField[idx].norm();

    // Normalize the magnitude
    float normalizedMagnitude = velocityMagnitude / MAX_VELOCITY;

    // Clamp the value between 0 and 1
    normalizedMagnitude = CLAMP(normalizedMagnitude);

    // Map to color using the colormap
    int colorIndex = static_cast<int>(normalizedMagnitude * 255.0f);
    colorIndex = min(max(colorIndex, 0), 255);

    // Access the colormap
    float r = viridis_colormap[colorIndex][0];
    float g = viridis_colormap[colorIndex][1];
    float b = viridis_colormap[colorIndex][2];

    colorField[idx] = Vector3f(r, g, b);
}

int main(int argc, char** argv) {
    // Simulation parameters
    float timestep = TIMESTEP;
    unsigned dim = DIM;
    float rdx = static_cast<float>(RES) / dim;
    float viscosity = VISCOSITY;
    global_decay_rate = DECAY_RATE;
    float r = RADIUS;

    // Force parameters
    C = Vector2f(static_cast<float>(dim) / 2.0f, static_cast<float>(dim) / 2.0f); // Center of the domain
    F = Vector2f(10.0f, 10.0f); // Initial force

    // Velocity vector field and pressure scalar field
    Vector2f* u = (Vector2f*)malloc(dim * dim * sizeof(Vector2f));
    float* p = (float*)malloc(dim * dim * sizeof(float));

    // Initialize host fields
    for (unsigned i = 0; i < dim * dim; i++) {
        u[i] = Vector2f::Zero();
        p[i] = 0.0f;
    }

    // Device memory allocation
    Vector2f* dev_u;
    float* dev_p;
    cudaMalloc((void**)&dev_u, dim * dim * sizeof(Vector2f));
    cudaMalloc((void**)&dev_p, dim * dim * sizeof(float));

    // Copy initial values to device
    cudaMemcpy(dev_u, u, dim * dim * sizeof(Vector2f), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, dim * dim * sizeof(float), cudaMemcpyHostToDevice);

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
        // Execute the Navier-Stokes kernel
        // Force parameters
        // C = Vector2f(static_cast<float>(dim) / 2.0f, static_cast<float>(dim) / 2.0f); // Center of the domain
        //add F as a periodic func
        F=Vector2f(10.0f*sin(glfwGetTime()), 0.0f);
        // F = Vector2f(0.0f, 40.0f); // Initial force
        NSkernel<<<blocks, threads>>>(dev_u, dev_p, rdx, viscosity, C, F, timestep, r, dim);

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error after NSkernel: %s\n", cudaGetErrorString(err));
            return 1;
        }

        cudaDeviceSynchronize();

        // Map the velocity field to colors
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

        // Decay the force
        decayForce();

        // Optionally update force parameters
        // For example, to move the force center or change its magnitude
        C = Vector2f(dim / 2.0f + 50.0f * sinf(glfwGetTime()), dim / 2.0f);
        F = Vector2f(10.0f, 10.0f);
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
