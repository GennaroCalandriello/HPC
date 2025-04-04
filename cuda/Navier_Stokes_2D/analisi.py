import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plotModes(mode_id=5):
    # Load the modes: shape (dim² * 2, Nmodes)
    modes = np.loadtxt("pod_modes.txt", delimiter=",")  # shape: (2*dim², Nmodes)

    # Check shape
    print("Loaded modes shape:", modes.shape)

    if modes.ndim != 2:
        raise ValueError("Expected a 2D array from pod_modes.txt")

    Ns2 = modes.shape[0] // 2
    dim = int(np.sqrt(Ns2))  # assuming a square grid

    # if dim * dim != Ns2:
    #     raise ValueError("Grid is not square or inconsistent dimensions in data")

    print(f"Grid: {dim}x{dim}, Num modes: {modes.shape[1]}")

    if mode_id >= modes.shape[1]:
        raise IndexError(f"Requested mode_id {mode_id} exceeds number of available modes ({modes.shape[1]})")

    # Extract x and y components
    mode_x_flat = modes[:Ns2, mode_id]
    mode_y_flat = modes[Ns2:, mode_id]

    # Reshape into 2D fields
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


def checkCSV():
    # load the data snapshots.csv
    data = np.loadtxt("snapshots.csv", delimiter=",")  # shape: (Ns² * 2, Nsnapshots)
    print("Loaded data shape:", data.shape)
    #covariance matrix
    covariance_matrix = data @ data.T/ data.shape[1]  # shape: (Ns² * 2, Ns² * 2)
    print("Covariance matrix shape:", covariance_matrix.shape)
    print("Covariance values:", covariance_matrix)
    
if __name__ == "__main__":
    plotModes()
    # checkCSV()