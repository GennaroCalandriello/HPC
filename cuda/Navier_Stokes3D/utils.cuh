// 3D indexing macro

#define IND3D(x, y, z, d) (((z) * (d) + (y)) * (d) + (x))

// 3-component velocity vector (x, y, z)
struct Vector3fVel {
  float x, y, z;

  __host__ __device__ Vector3fVel() : x(0.0f), y(0.0f), z(0.0f) {}
  __host__ __device__ Vector3fVel(float _x, float _y, float _z)
      : x(_x), y(_y), z(_z) {}

  // element access: 0→x, 1→y, 2→z
  __host__ __device__ float &operator()(int i) {
    return (i == 0 ? x : (i == 1 ? y : z));
  }
  __host__ __device__ const float &operator()(int i) const {
    return (i == 0 ? x : (i == 1 ? y : z));
  }

  // addition
  __host__ __device__ Vector3fVel operator+(const Vector3fVel &b) const {
    return Vector3fVel(x + b.x, y + b.y, z + b.z);
  }
  __host__ __device__ Vector3fVel &operator+=(const Vector3fVel &b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  // subtraction
  __host__ __device__ Vector3fVel operator-(const Vector3fVel &b) const {
    return Vector3fVel(x - b.x, y - b.y, z - b.z);
  }
  __host__ __device__ Vector3fVel &operator-=(const Vector3fVel &b) {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }

  // scalar multiplication / division
  __host__ __device__ Vector3fVel operator*(float s) const {
    return Vector3fVel(x * s, y * s, z * s);
  }
  __host__ __device__ Vector3fVel &operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  __host__ __device__ Vector3fVel operator/(float s) const {
    return Vector3fVel(x / s, y / s, z / s);
  }
  __host__ __device__ Vector3fVel &operator/=(float s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }

  // dot & norm
  __host__ __device__ float dot(const Vector3fVel &b) const {
    return x * b.x + y * b.y + z * b.z;
  }
  __host__ __device__ float norm() const { return sqrtf(dot(*this)); }

  // zero vector
  __host__ __device__ static Vector3fVel Zero() {
    return Vector3fVel(0.0f, 0.0f, 0.0f);
  }
};

// Reflect an index into [0..max] by mirroring
__device__ int reflectIndex(int i, int maxI) {
  if (i < 0)
    return -i;
  if (i > maxI)
    return 2 * maxI - i;
  return i;
}

// Clamp an index into [0..maxI]
__device__ int clampIndex(int i, int maxI) {
  return i < 0 ? 0 : (i > maxI ? maxI : i);
}

// Continuous coord reflection into [0..maxC]
__device__ float reflectCoord(float x, float maxC) {
  if (x < 0.0f)
    x = -x;
  else if (x > maxC)
    x = 2.0f * maxC - x;
  return fminf(fmaxf(x, 0.0f), maxC);
}

// Continuous coord clamping into [0..maxC]
__device__ float clampCoord(float x, float maxC) {
  return fminf(fmaxf(x, 0.0f), maxC);
}

// The interpolator
__device__ Vector3fVel
trilinearInterpolation3D(Vector3fVel pos,          // continuous (i,j,k)
                         const Vector3fVel *field, // dim³ velocity array
                         const int *geometryField, // dim³: 0=fluid,1=solid
                         unsigned dim, BoundaryCondition bcType) {
  float maxC = float(dim - 1);

  // 1) Handle domain‐boundary BC on the *position* (for REFLECT/ZERO_GRADIENT)
  if (bcType == BC_REFLECT) {
    pos.x = reflectCoord(pos.x, maxC);
    pos.y = reflectCoord(pos.y, maxC);
    pos.z = reflectCoord(pos.z, maxC);
  } else if (bcType == BC_ZERO_GRADIENT) {
    pos.x = clampCoord(pos.x, maxC);
    pos.y = clampCoord(pos.y, maxC);
    pos.z = clampCoord(pos.z, maxC);
  }
  // if BC_NO_SLIP_OUTLET, we leave pos unchanged so out‐of‐range samples get
  // zeroed below

  // 2) Base‐index and fractional
  int i0 = int(floorf(pos.x)), j0 = int(floorf(pos.y)), k0 = int(floorf(pos.z));
  int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
  float sx = pos.x - i0, sy = pos.y - j0, sz = pos.z - k0;
  float rx = 1.0f - sx, ry = 1.0f - sy, rz = 1.0f - sz;

  // 3) Corner‐sampling lambda
  auto sample = [&](int ii, int jj, int kk) {
    // domain‐boundary handling *per‐corner*
    if (ii < 0 || ii >= int(dim) || jj < 0 || jj >= int(dim) || kk < 0 ||
        kk >= int(dim)) {
      switch (bcType) {
      case BC_REFLECT:
        ii = reflectIndex(ii, dim - 1);
        jj = reflectIndex(jj, dim - 1);
        kk = reflectIndex(kk, dim - 1);
        break;
      case BC_ZERO_GRADIENT:
        ii = clampIndex(ii, dim - 1);
        jj = clampIndex(jj, dim - 1);
        kk = clampIndex(kk, dim - 1);
        break;
      case BC_NO_SLIP_OUTLET:
        return Vector3fVel::Zero();
      }
    }

    int idx = IND3D(ii, jj, kk, dim);
    // interior solids are always no-slip
    if (geometryField[idx] == 1)
      return Vector3fVel::Zero();
    return field[idx];
  };

  // 4) Fetch corners (8 samples)
  Vector3fVel c000 = sample(i0, j0, k0);
  Vector3fVel c100 = sample(i1, j0, k0);
  Vector3fVel c010 = sample(i0, j1, k0);
  Vector3fVel c110 = sample(i1, j1, k0);
  Vector3fVel c001 = sample(i0, j0, k1);
  Vector3fVel c101 = sample(i1, j0, k1);
  Vector3fVel c011 = sample(i0, j1, k1);
  Vector3fVel c111 = sample(i1, j1, k1);

  // 5) Interpolate: X → Y → Z
  Vector3fVel c00 = c000 * rx + c100 * sx;
  Vector3fVel c10 = c010 * rx + c110 * sx;
  Vector3fVel c01 = c001 * rx + c101 * sx;
  Vector3fVel c11 = c011 * rx + c111 * sx;

  Vector3fVel c0 = c00 * ry + c10 * sy;
  Vector3fVel c1 = c01 * ry + c11 * sy;

  return c0 * rz + c1 * sz;
}
