import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import hyperpar as hp
# Allow duplicate OpenMP libs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 1) Hyperparameters & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 50
num_layers  = hp.NUM_LAYERS
x_lb, x_ub  = -1.0, 1.0
t_lb, t_ub  =  0.0, 1.0
Nx, Nt      = 200, 200   # grid resolution

# 2) Define your LSTM‐PINN architecture (must match training)
class LSTMPINN(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x, t):
        # x, t are [B,1] → concatenate → [B,2] → add seq‐dim → [B,1,2]
        seq = torch.cat([x, t], dim=-1).unsqueeze(1)
        out, _ = self.lstm(seq)            # → [B,1,hidden_size]
        return self.fc(out[:, -1, :])      # → [B,1]

# 3) Load the trained model
model = LSTMPINN(input_size=2, hidden_size=hidden_size, num_layers=num_layers).to(device)
model.load_state_dict(torch.load("model/lstm_pinn.pth", map_location=device))
model.eval()

# 4) Build the (x,t) grid and run the model
x = np.linspace(x_lb, x_ub, Nx)
t = np.linspace(t_lb, t_ub, Nt)
Xg, Tg = np.meshgrid(x, t)

# Flatten and convert to torch
x_flat = torch.tensor(Xg.ravel(), dtype=torch.float32).view(-1,1).to(device)
t_flat = torch.tensor(Tg.ravel(), dtype=torch.float32).view(-1,1).to(device)

with torch.no_grad():
    u_flat = model(x_flat, t_flat).cpu().numpy()

# Reshape back to [Nt, Nx]
U = u_flat.reshape(Nt, Nx)

# 5) Static pcolormap
def plot_pcolormesh():
    plt.figure(figsize=(6,4))
    plt.pcolormesh(x, t, U, shading='auto', cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Burgers solution heatmap')
    plt.tight_layout()
    # plt.savefig("burgers_heatmap.png", dpi=300)
    plt.show()

#load and plot losses for data and PDE from a dictionary

def plot_losses():
    # load the dict you saved with np.save(...)
    loss_dict = np.load("loss/lossLSTM.npy", allow_pickle=True).item()
    
    plt.figure(figsize=(6,4))
    plt.plot(loss_dict['data'], label='Data Loss')
    plt.plot(loss_dict['pde'],  label='PDE  Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses during training')
    plt.legend()
    plt.tight_layout()
    plt.show()

def saveAnimation():
    # 6) Animation of u(x,t) as a line over x
    fig, ax = plt.subplots(figsize=(6,4))
    line, = ax.plot(x, U[0,:], lw=2)
    ax.set_xlim(x_lb, x_ub)
    ax.set_ylim(U.min(), U.max())
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'Burgers at t = {t[0]:.3f}')

    def update(frame):
        line.set_ydata(U[frame, :])
        ax.set_title(f'Burgers at t = {t[frame]:.3f}')
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=Nt, interval=100, blit=True
    )

    # Save as GIF (requires ffmpeg) or use other writers
    ani.save("burgers_evolution.gif", writer="ffmpeg", dpi=150)
    plt.show()
    plt.close(fig)  # avoid duplicate static plot in some environments

if __name__ == "__main__":
    plot_pcolormesh()
    plot_losses()
    saveAnimation()