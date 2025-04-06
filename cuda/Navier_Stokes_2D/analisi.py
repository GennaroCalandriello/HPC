import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

import polars as pl
import numpy as np

def load_matrix_polars(file_path: str, delimiter: str = ",") -> np.ndarray:
    """
    Load a CSV file using Polars and convert it to a NumPy array.

    Parameters:
        file_path (str): Path to the CSV file.
        delimiter (str): Field delimiter in the CSV file.

    Returns:
        np.ndarray: 2D NumPy array with the loaded data.
    """
    # Read the CSV file using Polars
    df = pl.read_csv(file_path, separator=delimiter)
    # Convert the Polars DataFrame to a NumPy array.
    # Note: The resulting array will be in row-major order.
    matrix = df.to_numpy()
    return matrix

def plot_eigenmode(modes, mode_index=6):
    """
    Plot a single eigenmode's x and y components.
    
    Parameters:
       modes: numpy array of shape (num_modes, 2*Ns^2)
       mode_index: index (row) of the mode to visualize.
    
    The first half of the row is reshaped to an (Ns x Ns) array for the x–component,
    and the second half to an (Ns x Ns) array for the y–component.
    """
    mode = modes[:, mode_index]
    print("ode values:", mode)  # Debugging line to check the mode values
    total = mode.size
    if total % 2 != 0:
        raise ValueError("Mode length must be even.")
    half = total // 2
    Ns  = int(math.sqrt(half))
    # if Ns * Ns != half:
    #     raise ValueError("Cannot reshape mode into a square grid.")
    
    # Split the mode into x and y components and reshape to (Ns, Ns)
    mode_x = mode[:half].reshape((Ns, Ns))
    mode_y = mode[half:].reshape((Ns, Ns))
    
    # Plot using two subplots with colormaps
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(mode_x, cmap='viridis', origin='lower')
    axs[0].set_title(f"Mode {mode_index+1} - x component")
    plt.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(mode_y, cmap='viridis', origin='lower')
    axs[1].set_title(f"Mode {mode_index+1} - y component")
    plt.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.show()

def animate_eigenmodes(modes, interval=1000):
    """
    Animate the eigenmodes sequentially.
    
    Parameters:
       modes: numpy array of shape (num_modes, 2*Ns^2)
       interval: delay (in ms) between frames.
    """
    num_modes, total = modes.shape
    if total % 2 != 0:
        raise ValueError("Mode length must be even.")
    half = total // 2
    Ns = int(math.sqrt(half))
    if Ns * Ns != half:
        raise ValueError("Cannot reshape mode into a square grid.")
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    im0 = ax0.imshow(np.zeros((Ns, Ns)), cmap='viridis', origin='lower')
    ax0.set_title("x component")
    im1 = ax1.imshow(np.zeros((Ns, Ns)), cmap='viridis', origin='lower')
    ax1.set_title("y component")
    
    def update(frame):
        mode = modes[frame, :]
        mode_x = mode[:half].reshape((Ns, Ns))
        mode_y = mode[half:].reshape((Ns, Ns))
        im0.set_data(mode_x)
        im1.set_data(mode_y)
        fig.suptitle(f"Mode {frame+1}")
        return im0, im1

    ani = animation.FuncAnimation(fig, update, frames=num_modes, interval=interval, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = "U.txt"
    modes = load_matrix_polars(filename, delimiter=",")
    print("Loaded modes with shape:", modes.shape)  # Expect (num_modes, 2*Ns^2)
    
    # Visualize the first mode
    # plot_eigenmode(modes, mode_index=0)
    
    # Uncomment the following line to animate all eigenmodes:
    # animate_eigenmodes(modes, interval=1000)
