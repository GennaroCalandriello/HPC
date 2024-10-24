// Structures

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


// Include constants and vector structures

// Include GLFW
#include "glfw-3.4.bin.WIN64/include/GLFW//glfw3.h"

// Simulation parameters
#define TIMESTEP 0.3f
#define DIM 1000
#define RES 1000
#define VISCOSITY 2f
#define RADIUS (DIM * DIM)
#define DECAY_RATE 3.0f
#define NUM_TIMESTEPS 300
#define MAX_VELOCITY 1.0f  // Adjust as needed for normalization

// CUDA kernel parameters
#define BLOCKSIZEY 16
#define BLOCKSIZEX 16

#define IND(x, y, d) int((y) * (d) + (x))
#define CLAMP(x) ((x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x))

// Simulation parameters
float timestep = TIMESTEP;
unsigned dim = DIM;
float rdx = static_cast<float>(RES) / dim;
float viscosity = VISCOSITY;
float r = 3000;
float magnitude = 70.0f;

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

__device__ void applyWind(Vector2f field, float windX, float windY, unsigned dim) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dim || j >= dim)
        return;

    int idx = IND(i, j, dim);

    // Apply wind uniformly
    field.x += windX;
    field.y += windY;

    // If you want to apply wind in a specific region, add conditions:
    // if (i >= startX && i <= endX && j >= startY && j <= endY) {
    //     field[idx].x += windX;
    //     field[idx].y += windY;
    // }
}
