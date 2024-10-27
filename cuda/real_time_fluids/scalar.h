#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Include GLFW
#include "glfw-3.4.bin.WIN64/include/GLFW//glfw3.h"

#define IND(x, y, d) int((y) * (d) + (x))
#define CLAMP(x) ((x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x))

#define VEL 1250
#define TIMESTEP 0.0001f //Noi cosa vogliamo, delta t più grandi o piu piccoli?
#define DIM 1100
#define RES 1100
#define VISCOSITY 0.0001f
#define RADIUS (DIM * DIM)
#define DECAY_RATE 0.3f
#define NUM_TIMESTEPS 300
#define MAX_VELOCITY VEL  // Adjust as needed for normalization (used in colorKernel--graphic parameter)
#define JETX DIM / 2
#define JETY 322
#define JETRADIUS DIM / 20
#define JETSPEED VEL
#define VORTEX_CENTER_X DIM/2
#define VORTEX_CENTER_Y DIM / 2
#define VORTEX_STRENGTH 2.0f
#define VORTEX_RADIUS DIM / 40
#define NUM_OF_DIFFUSION_STEPS 1
#define RENDERING 10 //Graphic parameter
#define BETA_BOUYANCY 0.9f

//Bool variables
#define FLUID_INJ 1
#define PERIODIC_FORCE 0
#define VORTEX 0

// CUDA kernel parameters
#define BLOCKSIZEY 16
#define BLOCKSIZEX 16

// Buondary parameter
#define periodic 1 // 1 if periodic boundary conditions are used, 0 otherwise
//Insert an advection external scalar field
#define advect_scalar_bool 1 // 1 if the scalar field is advected, 0 otherwise
#define diffusion_rate 0.001f // Diffusion rate of the scalar field  

// Simulation parameters
float timestep = TIMESTEP;
unsigned dim = DIM;
float rdx = static_cast<float>(RES) / dim;
float viscosity = VISCOSITY;
float r = 4000;
float magnitude = 5.0f;

struct Vector2f {
    float x, y;

    __host__ __device__ Vector2f() : x(0.0f), y(0.0f) {}

    __host__ __device__ Vector2f(float _x, float _y) : x(_x), y(_y) {}

    // Access operators
    __host__ __device__ float& operator()(int index) {
        return (index == 0) ? x : y;
    }

    __host__ __device__ const float& operator()(int index) const {
        return (index == 0) ? x : y;
    }

    // Addition
    __host__ __device__ Vector2f operator+(const Vector2f& other) const {
        return Vector2f(x + other.x, y + other.y);
    }

    __host__ __device__ Vector2f& operator+=(const Vector2f& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    // Subtraction
    __host__ __device__ Vector2f operator-(const Vector2f& other) const {
        return Vector2f(x - other.x, y - other.y);
    }

    __host__ __device__ Vector2f& operator-=(const Vector2f& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    // Multiplication by scalar
    __host__ __device__ Vector2f operator*(float scalar) const {
        return Vector2f(x * scalar, y * scalar);
    }

    __host__ __device__ Vector2f& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    // Division by scalar
    __host__ __device__ Vector2f operator/(float scalar) const {
        return Vector2f(x / scalar, y / scalar);
    }

    __host__ __device__ Vector2f& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    // Zero vector
    __host__ __device__ static Vector2f Zero() {
        return Vector2f(0.0f, 0.0f);
    }

    // Norm (magnitude)
    __host__ __device__ float norm() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ friend Vector2f operator*(float scalar, const Vector2f& v) {
        return Vector2f(scalar * v.x, scalar * v.y);
    }
};

struct Vector3f {
    float r, g, b;

    __host__ __device__ Vector3f() : r(0.0f), g(0.0f), b(0.0f) {}

    __host__ __device__ Vector3f(float _r, float _g, float _b) : r(_r), g(_g), b(_b) {}

    // Access operators
    __host__ __device__ float& operator()(int index) {
        if (index == 0) return r;
        else if (index == 1) return g;
        else return b;
    }

    __host__ __device__ const float& operator()(int index) const {
        if (index == 0) return r;
        else if (index == 1) return g;
        else return b;
    }

    // Addition
    __host__ __device__ Vector3f operator+(const Vector3f& other) const {
        return Vector3f(r + other.r, g + other.g, b + other.b);
    }

    __host__ __device__ Vector3f& operator+=(const Vector3f& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    // Multiplication by scalar
    __host__ __device__ Vector3f operator*(float scalar) const {
        return Vector3f(r * scalar, g * scalar, b * scalar);
    }

    __host__ __device__ Vector3f& operator*=(float scalar) {
        r *= scalar;
        g *= scalar;
        b *= scalar;
        return *this;
    }

    // Zero vector
    __host__ __device__ static Vector3f Zero() {
        return Vector3f(0.0f, 0.0f, 0.0f);
    }
};
// Include constants and vector structures


__device__ Vector2f bilinearInterpolation(Vector2f pos, Vector2f* field, unsigned dim) {
    if (periodic ==1 ){
            // Apply periodic wrapping
        pos.x = fmodf(pos.x + dim, dim);
        pos.y = fmodf(pos.y + dim, dim);

        int i0 = static_cast<int>(floorf(pos.x)) % dim;
        int j0 = static_cast<int>(floorf(pos.y)) % dim;
        int i1 = (i0 + 1) % dim;
        int j1 = (j0 + 1) % dim;

        float s1 = pos.x - floorf(pos.x);
        float s0 = 1.0f - s1;
        float t1 = pos.y - floorf(pos.y);
        float t0 = 1.0f - t1;

        Vector2f f00 = field[IND(i0, j0, dim)];
        Vector2f f10 = field[IND(i1, j0, dim)];
        Vector2f f01 = field[IND(i0, j1, dim)];
        Vector2f f11 = field[IND(i1, j1, dim)];

        return s0 * (t0 * f00 + t1 * f01) + s1 * (t0 * f10 + t1 * f11);
        }

    else 
        {
        pos.x = fmaxf(0.0f, fminf(pos.x, dim - 1.001f));
        pos.y = fmaxf(0.0f, fminf(pos.y, dim - 1.001f));

        int i = static_cast<int>(pos.x);
        int j = static_cast<int>(pos.y);
        float dx = pos.x - i;
        float dy = pos.y - j;

        // Adjust indices for safety
        int i1 = min(i + 1, dim - 1);
        int j1 = min(j + 1, dim - 1);

        // // Perform bilinear interpolation
        Vector2f f00 = field[IND(i, j, dim)];
        Vector2f f10 = field[IND(i1, j, dim)];
        Vector2f f01 = field[IND(i, j1, dim)];
        Vector2f f11 = field[IND(i1, j1, dim)];

        Vector2f f0 = f00 * (1.0f - dx) + f10 * dx;
        Vector2f f1 = f01 * (1.0f - dx) + f11 * dx;

        return f0 * (1.0f - dy) + f1 * dy; }
        // Vector2f f00 = (i < 0 || i >= dim || j < 0 || j >= dim) ? Vector2f::Zero() : field[IND(i , j , dim)];
        // Vector2f f01 = (i + 1 < 0 || i + 1 >= dim || j  < 0 || j  >= dim) ? Vector2f::Zero() : field[IND(i + 1, j , dim)];
        // Vector2f f10 = (i  < 0 || i  >= dim || j + 1 < 0 || j + 1 >= dim) ? Vector2f::Zero() : field[IND(i , j + 1, dim)];
        // Vector2f f11 = (i + 1 < 0 || i + 1 >= dim || j + 1 < 0 || j + 1 >= dim) ? Vector2f::Zero() : field[IND(i + 1, j + 1, dim)];

        // Vector2f f0 = (1 - dx) * f00 + dx * f10;
        // Vector2f f1 = (1 - dx) * f01 + dx * f11;

        // return (1 - dy) * f0 + dy * f1;
}

__device__ Vector2f velocityAt(Vector2f pos, Vector2f* velfield, unsigned dim) {
    // Clamp positions to grid boundaries
    if (periodic ==1) {
            // Apply periodic wrapping
        pos.x = fmodf(pos.x + dim, dim);
        pos.y = fmodf(pos.y + dim, dim);

        // Perform bilinear interpolation on the velocity field
        return bilinearInterpolation(pos, velfield, dim);
        }

    else {
            pos.x = fmaxf(0.0f, fminf(pos.x, dim - 1.001f));
            pos.y = fmaxf(0.0f, fminf(pos.y, dim - 1.001f));

            // Perform bilinear interpolation on the velocity field
            return bilinearInterpolation(pos, velfield, dim);

        }
    }  

__device__ float interpolateScalar(Vector2f pos, float* field, unsigned dim) {
    if (periodic == 1) {
        // Apply periodic wrapping
        pos.x = fmodf(pos.x + dim, dim);
        pos.y = fmodf(pos.y + dim, dim);

        int i0 = static_cast<int>(floorf(pos.x)) % dim;
        int j0 = static_cast<int>(floorf(pos.y)) % dim;
        int i1 = (i0 + 1) % dim;
        int j1 = (j0 + 1) % dim;

        float s1 = pos.x - floorf(pos.x);
        float s0 = 1.0f - s1;
        float t1 = pos.y - floorf(pos.y);
        float t0 = 1.0f - t1;

        float f00 = field[IND(i0, j0, dim)];
        float f10 = field[IND(i1, j0, dim)];
        float f01 = field[IND(i0, j1, dim)];
        float f11 = field[IND(i1, j1, dim)];

        return s0 * (t0 * f00 + t1 * f01) + s1 * (t0 * f10 + t1 * f11);
    } else {
        // Non-periodic boundary conditions (clamping)
        pos.x = fmaxf(0.0f, fminf(pos.x, dim - 1.001f));
        pos.y = fmaxf(0.0f, fminf(pos.y, dim - 1.001f));

        int i0 = static_cast<int>(pos.x);
        int j0 = static_cast<int>(pos.y);
        int i1 = min(i0 + 1, dim - 1);
        int j1 = min(j0 + 1, dim - 1);

        float s1 = pos.x - i0;
        float s0 = 1.0f - s1;
        float t1 = pos.y - j0;
        float t0 = 1.0f - t1;

        float f00 = field[IND(i0, j0, dim)];
        float f10 = field[IND(i1, j0, dim)];
        float f01 = field[IND(i0, j1, dim)];
        float f11 = field[IND(i1, j1, dim)];

        return s0 * (t0 * f00 + t1 * f01) + s1 * (t0 * f10 + t1 * f11);
    }
}



__device__ void advectScalar(Vector2f x, float* field, Vector2f* velfield, float timestep, float rdx, unsigned dim) {
    float dt0 = timestep * rdx;
    // RK4 integration
    
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

    // Interpolate the scalar field at the backtraced position
    int idx = IND(static_cast<int>(x.x), static_cast<int>(x.y), dim);
    field[idx] = interpolateScalar(pos, field, dim);
}

__device__ int periodicIndex(int idx, int max) {
    if (idx < 0)
        return idx + max;
    else if (idx >= max)
        return idx - max;
    else
        return idx;
}

__device__ void diffuseScalar(Vector2f x, float* field, float diffRate, float timestep, float rdx, unsigned dim) {
    int i = static_cast<int>(x.x);
    int j = static_cast<int>(x.y);

    float alpha = rdx * rdx / (diffRate * timestep);
    float beta = 4.0f + alpha;

    if (periodic == 0) {
        // apply Periodic BC via periodicIndex function
        float f_left = field[IND(periodicIndex(i - 1, dim), j, dim)];
        float f_right = field[IND(periodicIndex(i + 1, dim), j, dim)];
        float f_down = field[IND(i, periodicIndex(j - 1, dim), dim)];
        float f_up = field[IND(i, periodicIndex(j + 1, dim), dim)];
        float f_center = field[IND(i, j, dim)];

        float b = f_center;

        // Jacobi iteration
        field[IND(i, j, dim)] = (f_left + f_right + f_down + f_up + alpha * b) / beta;

        
    }
    else {
        float f_left = (i > 0) ? field[IND(i - 1, j, dim)] : 0.0f;
        float f_right = (i < dim - 1) ? field[IND(i + 1, j, dim)] : 0.0f;
        float f_down = (j > 0) ? field[IND(i, j - 1, dim)] : 0.0f;
        float f_up = (j < dim - 1) ? field[IND(i, j + 1, dim)] : 0.0f;
        float f_center = field[IND(i, j, dim)];

        float b = f_center;

        // Jacobi iteration
        field[IND(i, j, dim)] = (f_left + f_right + f_down + f_up + alpha * b) / beta; 
        }
}

__device__ void applyBuoyancy(Vector2f x, Vector2f* u, float* c, float c_ambient, float beta, float gravity, unsigned dim) {
    int idx = IND(static_cast<int>(x.x), static_cast<int>(x.y), dim);
    float c_value = c[idx];

    // Calcola la forza di galleggiamento
    float buoyancyForce = beta * (c_value - c_ambient);

    // Applica la forza al componente verticale della velocità
    u[idx].y += buoyancyForce * gravity;
}


