# Navier-Stokes Fluid Simulation on GPU

This project implements a 2D Navier-Stokes fluid simulation using CUDA for GPU acceleration. The simulation visualizes fluid flow using color-mapped velocity magnitudes and includes interactive elements such as central forces that influence the fluid dynamics. The simulation leverages numerical techniques to solve the Navier-Stokes equations, which describe the motion of fluid substances.

## Navier-Stokes Equations

The 2D Navier-Stokes equations for incompressible fluid flow are given by:

1. **Momentum Equation**:

∂u/∂t + (u · ∇)u = - (1/ρ) ∇p + ν ∇²u + f

- `u`: Velocity vector field
- `p`: Pressure
- `ρ`: Fluid density (assumed constant)
- `ν`: Kinematic viscosity
- `f`: External forces (e.g., a central force applied to the fluid)

2. **Incompressibility Condition**:

∇ · u = 0

## Numerical Method

The simulation uses a **grid-based finite difference method** to discretize the equations:

1. **Advection**: Solved using a semi-Lagrangian method with bilinear interpolation, transporting the fluid quantities along the velocity field. HEART OF SIM ---> It solves the nonlinearity in the NS equations. Two steps:
     1. Backtrace the particle (RK4) to its original position
     2. Interpolate the field at the backtraced position
  The backtracing is computed using a fourth-order Runge-Kutta integration scheme.
2. **Diffusion**: Approximated with a **Jacobi iteration**, applying diffusion to the velocity field using iterative relaxation.
3. **Pressure Projection**: A Poisson equation is solved iteratively to ensure the velocity field remains divergence-free, maintaining incompressibility.
4. **Force Application**: External forces, like a central force, are applied to simulate interactions with the fluid. The forces are introduced as a source term in the momentum equation.
5. **Boundary Conditions**: The simulation enforces zero velocity (no-slip), or PBC at the domain boundaries and handles interactions with obstacles using explicit boundary conditions (questo ancora no).

## Key Features

- **GPU Acceleration**
- **Real-Time Visualization**
- **Interactive Central Force**
- **Obstacle Integration**

## Installation
All needed libraries are included in the directory. Change paths in .vscode/task.json. Execute Run Build Task, Ctrl Shift B. The simulation parameters are in scalar.h

