# Navier-Stokes Fluid Simulation on GPU

This project implements a 2D Navier-Stokes fluid simulation using CUDA for GPU acceleration. The simulation visualizes fluid flow using color-mapped velocity magnitudes and includes interactive elements such as central forces that influence the fluid dynamics. The simulation leverages numerical techniques to solve the Navier-Stokes equations, which describe the motion of fluid substances.

## Navier-Stokes Equations

The 2D Navier-Stokes equations for incompressible fluid flow are given by:

1. **Momentum Equation**:

∂u/∂t + (u · ∇)u = - (1/ρ) ∇p + ν ∇²u + f
   - \( \mathbf{u} \): Velocity vector field
   - \( p \): Pressure
   - \( \rho \): Fluid density (assumed constant)
   - \( \nu \): Kinematic viscosity
   - \( \mathbf{f} \): External forces (e.g., a central force applied to the fluid)

2. **Incompressibility Condition**:

   \[
   \nabla \cdot \mathbf{u} = 0
   \]

## Numerical Method

The simulation uses a **grid-based finite difference method** to discretize the equations:

1. **Advection**: Solved using a semi-Lagrangian method with bilinear interpolation, transporting the fluid quantities along the velocity field.
2. **Diffusion**: Approximated with a **Jacobi iteration**, applying diffusion to the velocity field using iterative relaxation.
3. **Pressure Projection**: A Poisson equation is solved iteratively to ensure the velocity field remains divergence-free, maintaining incompressibility.
4. **Force Application**: External forces, like a central force, are applied to simulate interactions with the fluid. The forces are introduced as a source term in the momentum equation.
5. **Boundary Conditions**: The simulation enforces zero velocity (no-slip) at the domain boundaries and handles interactions with obstacles using explicit boundary conditions.

## Key Features

- **GPU Acceleration**: Leveraging CUDA for parallel computation, significantly speeding up the simulation.
- **Real-Time Visualization**: Visualization of the fluid dynamics using color maps representing the velocity magnitude.
- **Interactive Central Force**: Apply customizable forces at the center of the grid to observe fluid behavior.
- **Obstacle Integration**: The code supports adding static obstacles to the simulation, modifying fluid flow around them.

## Installation

To compile and run the simulation, ensure you have the necessary dependencies installed:

- **CUDA** (for GPU computation)
- **GLFW** (for visualization)
- **OpenGL** (for rendering, using GLFW for the context)

Compile the code using the following command:

```bash
nvcc -o fluidsimulation mainfluids.cu -lglfw