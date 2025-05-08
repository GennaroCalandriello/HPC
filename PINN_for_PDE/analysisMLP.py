import torch
import numpy as np
import matplotlib.pyplot as plt
from modelMLP import PINN
import hyperpar as hp
from matplotlib.animation import FuncAnimation

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
torch.set_num_threads(1)

#Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the trained model
model = PINN(hp.LAYERS).to(device)


# State dict is a Python dictionary object that maps each layer to its parameter tensor.
# It contains all the weights and biases of the model.
model.load_state_dict(torch.load("model.pth"))

# Set the model to evaluation mode
model.eval()

#create meshgrid
nx, nt = 200, 200
x = np.linspace(0, 1, nx)
t = np.linspace(0, 0.5, nt)
X, T = np.meshgrid(x, t)

#convert to tensors and flatten, it moves on the GPU
#Flattens the array into a 1D array and then adds a new dimension to it. [:, None] adds a new axis to the flattened array. Now 
#they are 2D arrays with shape (nx*nt, 1) instead of (nx*nt,)
X_t = torch.tensor(X.flatten()[:, None], dtype = torch.float32, device = device)
T_t = torch.tensor(T.flatten()[:, None], dtype = torch.float32, device = device)

def staticPlot():
#evaluation of the model on the grid
    with torch.no_grad():
        U_t = model(X_t, T_t).cpu().numpy() #move to CPU and convert to numpy array
    U = U_t.reshape(X.shape)

    #plot the velocity field
    plt.figure(figsize=(10, 6))
    pcm = plt.pcolormesh(X, T, U, shading='auto', cmap='viridis')
    plt.colorbar(pcm, label='Velocity (u)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Velocity Field')
    plt.tight_layout()
    plt.show()

def animatePlot():

    with torch.no_grad():
        U_flat = model(X_t, T_t).cpu().numpy().flatten()

    U = U_flat.reshape((nt, nx))

    fig, ax = plt.subplots()
    line, = ax.plot(x, U[0, :], 'r-', lw =2)
    ax.set_xlim(0, 1)
    ax.set_ylim(U.min(), U.max())
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    title = ax.text(0.5, 1.05, "", ha ='center')

    #animation function
    def update(frame):
        yf = U[frame,:]
        line.set_data(x, yf)
        title.set_text(f"t = {frame*0.5/(nt-1):.2f}")
        return line, title

    ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=100)
    plt.show()

def plot_loss_functions():
    # Load the loss data
    history = np.load("loss.npy", allow_pickle=True).item()

    # Plot the loss functions
    plt.figure(figsize=(10, 6))
    plt.plot(history["total_loss"], label="Total Loss")
    plt.plot(history["pde_loss"], label="PDE Loss")
    plt.plot(history["bc_loss"], label="BC Loss")
    plt.plot(history["ic_loss"], label="IC Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Functions")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    staticPlot()
    animatePlot()
    plot_loss_functions()



