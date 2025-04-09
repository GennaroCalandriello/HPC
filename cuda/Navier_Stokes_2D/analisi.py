import numpy as np
import matplotlib.pyplot as plt
import math

def plot_mode_colormap(mode: np.ndarray, mode_index: int = 4):
    """
    Plot an eigenmode as a colormap by computing the magnitude of its (x, y) components.
    
    The mode is assumed to be a 1D array with length 2*Ns^2, where:
      - The first half are the x-components,
      - The second half are the y-components.
    
    The function reshapes these halves into (Ns x Ns) arrays, computes the magnitude
    sqrt(x^2 + y^2) at each grid point, and plots a colormap of the resulting scalar field.
    
    Parameters:
        mode (np.ndarray): 1D array representing the eigenmode.
        mode_index (int): The index of the mode (for labeling purposes).
    """
    total = mode.size
    if total % 2 != 0:
        raise ValueError("Mode length must be even.")
    
    half = total // 2
    Ns = int(math.sqrt(half))
    # if Ns * Ns != half:
    #     raise ValueError("Mode length does not allow a square grid.")
    
    # Reshape to get x and y components in a square grid
    mode_x = mode[:half].reshape((Ns, Ns))
    mode_y = mode[half:].reshape((Ns, Ns))
    
    # Compute the magnitude at each grid point
    magnitude = np.sqrt(mode_x**2 + mode_y**2)
    
    # Plot the magnitude as a colormap
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap='viridis', origin='lower')
    plt.title(f"Eigenmode {mode_index+1} Magnitude")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()
    
def plot_eigenvalues(eigenvalues: np.ndarray):
    """
    Plot eigenvalues as a bar chart.
    
    Parameters:
        eigenvalues (np.ndarray): 1D array of eigenvalues.
    """
    #normalize
    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)
    eigenvalues = eigenvalues/max_eigenvalue
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(eigenvalues)), eigenvalues, color='blue')
    plt.title("Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.tight_layout()
    plt.show()
    
# // Write matrix elements to file.
#     for (int i = 0; i < N; i++) {
#         for (int j = 0; j < M; j++) {
#             double value = 0.0;
#             if (order == Layout::ColMajor) {
#                 // In column-major order, element (i, j) is at index i + j*N.
#                 value = mat[i + j * N];
#             } else {
#                 // In row-major order, element (i, j) is at index i*M + j.
#                 value = mat[i * M + j];
#             }
#             outFile << value;
#             if (j < M - 1) {
#                 outFile << ",";
#             }
#         }
#         outFile << "\n";
#     }
# Example usage:
if __name__ == "__main__":
    # Example: load a specific mode from a large CSV file.
    # Here we assume the file 'pod_modes.txt' is very large and each row is a mode.
    # We use np.loadtxt with skiprows and max_rows to load only one row.
    
    file_path = "pod_modes.txt"
    chosen_mode = 3  # zero-indexed: this will load the 6th mode
    
    # Load only the chosen row from the file
    mode = np.loadtxt(file_path, delimiter=",")
    eigenvalues = np.loadtxt("sigma.txt", delimiter=",")
    modecol =np.copy(mode)
    N, M = mode.shape
    # for i in range(N):
    #     for j in range(M):
    #         value = 0.0
    #             # In column-major order, element (i, j) is at index i + j*N.
    #         value = mode[i + j * N]
    #         modecol[i,j] = value
            
    print("Loaded mode with shape:", mode.shape)
    print("Loaded mode with shape:", mode.shape)
    mode = mode[:, chosen_mode]
    # Plot the mode as a colormap of the magnitude
    plot_mode_colormap(mode, mode_index=chosen_mode)
    # Plot the eigenvalues
    plot_eigenvalues(eigenvalues)
