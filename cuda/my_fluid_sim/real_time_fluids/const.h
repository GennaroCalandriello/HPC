// Structures
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