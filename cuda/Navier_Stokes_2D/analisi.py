import numpy as np
import matplotlib.pyplot as plt

# Load the modes: shape (Ns*Ns*2, Nmodes)
modes = np.loadtxt("pod_modes.txt", delimiter=",")  # shape: (Ns² * 2, Nmodes)

# Check shape
print("Loaded modes shape:", modes)

# Recover spatial resolution
Ns2 = modes.shape[0] // 2
dim = int(np.sqrt(Ns2))  # assuming square grid
print(f"Grid: {dim}x{dim}, Num modes: {modes.shape[1]}")

mode_id = 5  # Select mode index

# Split x and y components
mode_x_flat = modes[:Ns2, mode_id]
mode_y_flat = modes[Ns2:, mode_id]

# Reshape for visualization
mode_x = mode_x_flat.reshape((dim, dim))
mode_y = mode_y_flat.reshape((dim, dim))

# Plot the mode
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(mode_x, cmap='seismic', origin='lower')
plt.title(f"POD Mode {mode_id+1} (x)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(mode_y, cmap='seismic', origin='lower')
plt.title(f"POD Mode {mode_id+1} (y)")
plt.colorbar()

plt.tight_layout()
plt.show()
