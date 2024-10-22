import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load simulation data from the output text file
def load_simulation_data(filename):
    data = []
    with open(filename, 'r') as file:
        timestep_data = []
        for line in file:
            line = line.strip()
            if line.startswith("Timestep"):
                if timestep_data:
                    data.append(np.array(timestep_data))
                    timestep_data = []
            elif line:
                velocities = []
                components = line.replace('(', '').replace(')', '').replace(',', '').split()
                for i in range(0, len(components), 2):
                    vx = float(components[i])
                    vy = float(components[i + 1])
                    velocities.append([vx, vy])
                timestep_data.append(velocities)
        if timestep_data:
            data.append(np.array(timestep_data))

    # Debugging: Print information about the loaded data
    # print(f"Loaded {len(data)} timesteps")
    # for idx, timestep in enumerate(data):
    #     print(f"Timestep {idx}: shape {timestep.shape}")
    #     print(f"Sample data (first row): {timestep[0][:5]}")  # Print first 5 vectors of the first row
    return data

# Visualize the velocity field using quiver plot with colors
def visualize_velocity_field_with_colors(data, interval=100, save=True, filename='velocity_field_animation.mp4'):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import colors

    fig, ax = plt.subplots()
    
    # Downsample factor
    downsample_factor = 2
    dim_y, dim_x = data[0].shape[:2]
    
    X, Y = np.meshgrid(
        np.arange(0, dim_x, downsample_factor),
        np.arange(0, dim_y, downsample_factor)
    )
    
    # Initialize the quiver plot with the first frame
    velocities = data[0][::downsample_factor, ::downsample_factor]
    U = velocities[:, :, 0]
    V = velocities[:, :, 1]
    magnitude = np.sqrt(U**2 + V**2)
    
    # Set consistent color limits
    max_magnitude = np.max([np.sqrt(frame[:, :, 0]**2 + frame[:, :, 1]**2).max() for frame in data])
    norm = colors.Normalize(vmin=0, vmax=max_magnitude)
    
    Q = ax.quiver(X, Y, U, V, magnitude, cmap='jet', norm=norm, scale=50)
    fig.colorbar(Q, ax=ax, label='Velocity Magnitude')
    ax.set_title('Timestep 0')
    ax.set_xlim(0, dim_x)
    ax.set_ylim(0, dim_y)
    ax.set_aspect('equal')

    def update_quiver(frame):
        velocities = data[frame][::downsample_factor, ::downsample_factor]
        U = velocities[:, :, 0]
        V = velocities[:, :, 1]
        magnitude = np.sqrt(U**2 + V**2)
        Q.set_UVC(U, V, magnitude)
        ax.set_title(f'Timestep {frame}')
        return Q,
    
    ani = animation.FuncAnimation(fig, update_quiver, frames=len(data), interval=interval, blit=False)

    if save:
        # Set up formatting for the movie files
        print(f"Saving animation as {filename}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1000/interval, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename, writer=writer)
        print(f"Animation saved as {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    # Load data from the file
    filename = 'fluid_simulation_output.txt'
    simulation_data = load_simulation_data(filename)
    
    # Reshape data if necessary
    for idx, timestep in enumerate(simulation_data):
        simulation_data[idx] = timestep.reshape((timestep.shape[0], timestep.shape[1], 2))
    
    # Visualize the velocity field with colors
    visualize_velocity_field_with_colors(simulation_data)
