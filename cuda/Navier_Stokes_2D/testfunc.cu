#define IND(i, j, dim) ((i) + (j) * (dim))

// Compute central difference in x for one component of a vector field.
__device__ float ddx_component(const Vector2f *field, int i, int j,
                               unsigned dim, float dx, int comp) {
  int idx_center = IND(i, j, dim);
  int idx_left = (i > 0) ? IND(i - 1, j, dim) : idx_center;
  int idx_right = (i < (int)dim - 1) ? IND(i + 1, j, dim) : idx_center;
  float left_val = (comp == 0) ? field[idx_left].x : field[idx_left].y;
  float right_val = (comp == 0) ? field[idx_right].x : field[idx_right].y;
  return (right_val - left_val) / (2.0f * dx);
}

// Compute central difference in y for one component of a vector field.
__device__ float ddy_component(const Vector2f *field, int i, int j,
                               unsigned dim, float dx, int comp) {
  int idx_center = IND(i, j, dim);
  int idx_down = (j > 0) ? IND(i, j - 1, dim) : idx_center;
  int idx_up = (j < (int)dim - 1) ? IND(i, j + 1, dim) : idx_center;
  float down_val = (comp == 0) ? field[idx_down].x : field[idx_down].y;
  float up_val = (comp == 0) ? field[idx_up].x : field[idx_up].y;
  return (up_val - down_val) / (2.0f * dx);
}

// Compute the symmetric gradient tensor components at (i,j):
// Returns S11, S22, and S12 (with S12 = S21)
__device__ void symmetric_gradient(const Vector2f *phi, int i, int j,
                                   unsigned dim, float dx, float &S11,
                                   float &S22, float &S12) {
  S11 = ddx_component(phi, i, j, dim, dx, 0);             // ∂(φ.x)/∂x
  S22 = ddy_component(phi, i, j, dim, dx, 1);             // ∂(φ.y)/∂y
  float dphi_x_dy = ddy_component(phi, i, j, dim, dx, 0); // ∂(φ.x)/∂y
  float dphi_y_dx = ddx_component(phi, i, j, dim, dx, 1); // ∂(φ.y)/∂x
  S12 = 0.5f * (dphi_x_dy + dphi_y_dx); // (∂(φ.x)/∂y+ ∂(φ.y)/∂x) / 2
}
// Compute central difference in x for a scalar field.
__device__ float ddx_scalar(const float *field, int i, int j, unsigned dim,
                            float dx) {
  int idx = IND(i, j, dim);
  int idx_left = (i > 0) ? IND(i - 1, j, dim) : idx;
  int idx_right = (i < (int)dim - 1) ? IND(i + 1, j, dim) : idx;
  return (field[idx_right] - field[idx_left]) / (2.0f * dx);
}

// Compute central difference in y for a scalar field.
__device__ float ddy_scalar(const float *field, int i, int j, unsigned dim,
                            float dx) {
  int idx = IND(i, j, dim);
  int idx_down = (j > 0) ? IND(i, j - 1, dim) : idx;
  int idx_up = (j < (int)dim - 1) ? IND(i, j + 1, dim) : idx;
  return (field[idx_up] - field[idx_down]) / (2.0f * dx);
}
// Compute divergence of 2∇_s φ at grid point (i,j) given the field phi.
__device__ Vector2f div_symGrad(const Vector2f *phi, int i, int j, unsigned dim,
                                float dx) {
  // First, compute the symmetric gradient components at (i,j)
  float S11, S22, S12;
  symmetric_gradient(phi, i, j, dim, dx, S11, S22, S12);

  // We want to compute the divergence:
  // (div)_1 = ∂/∂x(2S11) + ∂/∂y(2S12)
  // (div)_2 = ∂/∂x(2S12) + ∂/∂y(2S22)
  // To approximate these, we need to evaluate S11, S12, S22 at neighboring
  // points.

  // For simplicity, we compute the following at (i,j) using central
  // differences. Note: In a production code you would add more robust boundary
  // handling.

  // For 2S11, form a temporary scalar field value at each grid point:
  float valS11_center = 2.0f * S11;
  float valS12_center = 2.0f * S12;
  float valS22_center = 2.0f * S22;
  //   __device__ void symmetric_gradient(const Vector2f *phi, int i, int j,
  //                                      unsigned dim, float dx, float &S11,
  //                                      float &S22, float &S12)
  // For ∂(2S11)/∂x: compute S11 at (i+1, j) and (i-1, j)
  float S11_right, S11_left;
  {
    float t1, t2, dummy;
    symmetric_gradient(phi, min(i + 1, (int)dim - 1), j, dim, dx, t1, dummy,
                       t2);
    S11_right = 2.0f * t1;
    symmetric_gradient(phi, max(i - 1, 0), j, dim, dx, t1, dummy, t2);
    S11_left = 2.0f * t1;
  }
  float d_twoS11_dx = (S11_right - S11_left) / (2.0f * dx);

  // For ∂(2S12)/∂y: compute S12 at (i, j+1) and (i, j-1)
  float S12_up, S12_down;
  {
    float dummy, S12_up_val;
    symmetric_gradient(phi, i, min(j + 1, (int)dim - 1), dim, dx, dummy, dummy,
                       S12_up_val);
    S12_up = 2.0f * S12_up_val;
    symmetric_gradient(phi, i, max(j - 1, 0), dim, dx, dummy, dummy,
                       S12_up_val);
    S12_down = 2.0f * S12_up_val;
  }
  float d_twoS12_dy = (S12_up - S12_down) / (2.0f * dx);

  float div_x = d_twoS11_dx + d_twoS12_dy;

  // For ∂(2S12)/∂x: compute S12 at (i+1, j) and (i-1, j)
  float S12_right, S12_left;
  {
    float dummy, S12_temp;
    symmetric_gradient(phi, min(i + 1, (int)dim - 1), j, dim, dx, dummy, dummy,
                       S12_temp);
    S12_right = 2.0f * S12_temp;
    symmetric_gradient(phi, max(i - 1, 0), j, dim, dx, dummy, dummy, S12_temp);
    S12_left = 2.0f * S12_temp;
  }
  float d_twoS12_dx = (S12_right - S12_left) / (2.0f * dx);

  // For ∂(2S22)/∂y: compute S22 at (i, j+1) and (i, j-1)
  float S22_up, S22_down;
  {
    float dummy, dummy2, S22_temp;
    symmetric_gradient(phi, i, min(j + 1, (int)dim - 1), dim, dx, dummy,
                       S22_temp, dummy2);
    S22_up = 2.0f * S22_temp;
    symmetric_gradient(phi, i, max(j - 1, 0), dim, dx, dummy, S22_temp, dummy2);
    S22_down = 2.0f * S22_temp;
  }
  float d_twoS22_dy = (S22_up - S22_down) / (2.0f * dx);

  float div_y = d_twoS12_dx + d_twoS22_dy;

  return Vector2f(div_x, div_y);
}
// Suppose phi_i and divField are arrays of Vector2f of length gridPoints.
float innerProduct = 0.0f;
for (int k = 0; k < gridPoints; k++) {
  // Define dot for Vector2f, e.g.:
  float dot_val = phi_i[k].x * divField[k].x + phi_i[k].y * divField[k].y;
  innerProduct += dot_val * (dx * dx); // multiply by cell area
}
