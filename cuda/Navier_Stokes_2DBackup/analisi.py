import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.animation as animation

def animate_pod_modes(all_modes: np.ndarray, interval: int = 200):
    """
    Animate all POD (or eigen-) modes in a 2D colormap.
    
    Parameters:
    -----------
    all_modes : np.ndarray, shape (2*N_s^2, n_modes)
        Each column represents one mode. The first N_s^2 entries are the x-components,
        and the next N_s^2 entries are the y-components of that mode.
    interval  : int
        Delay between frames in milliseconds for the animation.

    Returns:
    --------
    ani : matplotlib.animation.FuncAnimation
        The animation object. You can display it in a Jupyter notebook or 
        save it to a file using ani.save('modes.mp4').
    """

    # Determine grid size and number of modes
    n_rows, n_modes = all_modes.shape
    if n_rows % 2 != 0:
        raise ValueError("The number of rows (2*N_s^2) must be even.")
        
    half = n_rows // 2
    Ns = int(math.isqrt(half))  # integer square root in Python 3.8+
    if Ns * Ns != half:
        raise ValueError("The first half of the mode does not form a perfect square (x-components).")
    
    # Precompute the magnitude fields for each mode
    magnitude_fields = []
    for mode_idx in range(n_modes):
        mode_x = all_modes[:half, mode_idx].reshape((Ns, Ns))
        mode_y = all_modes[half:, mode_idx].reshape((Ns, Ns))
        mag = np.sqrt(mode_x**2 + mode_y**2)
        magnitude_fields.append(mag)
    
    # Determine a global color scale so the animation doesn't rescale each frame
    vmin, vmax = 0.0, max(np.max(mag) for mag in magnitude_fields)

    # Set up the figure and initial image
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(magnitude_fields[0],
                   cmap='viridis',
                   origin='lower',
                   vmin=vmin,
                   vmax=vmax,
                   animated=True)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Magnitude")
    ax.set_title("POD Mode 1")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Update function for FuncAnimation
    def update(frame):
        im.set_data(magnitude_fields[frame])
        ax.set_title(f"POD Mode {frame+1}")  # frame is zero-based
        return [im]

    # Create the animation
    ani = animation.FuncAnimation(
        fig,        # Figure to plot into
        update,     # Update function
        frames=range(n_modes),  # Sequence of modes
        interval=interval,      # Delay between frames in ms
        blit=True
    )

    # Display the animation
    plt.tight_layout()
    plt.show()
    
    return ani

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
    
    file_path = "resultsdata/pod_modes.txt"
    chosen_mode = 198
    # zero-indexed: this will load the 6th mode
    
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
    mode1 = mode[:, chosen_mode]    
    plot_mode_colormap(mode1, mode_index=chosen_mode)
    # plot_eigenvalues(eigenvalues[:-4])
    ani = animate_pod_modes(mode[:, 160:], interval=30)
