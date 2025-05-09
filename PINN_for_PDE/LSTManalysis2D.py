# burgers2d_analysis.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import hyperpar as hp

# 1) Hyperparameters & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 50
num_layers  = hp.NUM_LAYERS
x_lb, x_ub  = hp.X_LB, hp.X_UB
y_lb, y_ub  = hp.Y_LB, hp.Y_UB
t_lb, t_ub  = hp.T_LB, hp.T_UB
Nx, Ny, Nt  = 100, 100, 60   # grid resolution

# 2) Model definition (must match training exactly)
class LSTMPINN(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x, y, t):
        # x,y,t each [B,1] → concat → [B,3] → add seq‐dim → [B,1,3]
        seq = torch.cat([x, y, t], dim=-1).unsqueeze(1)
        out, _ = self.lstm(seq)           # → [B,1,hidden_size]
        return self.fc(out[:, -1, :])     # → [B,1]

# 3) Load trained model
model = LSTMPINN(input_size=3, hidden_size=hidden_size, num_layers=num_layers).to(device)
model.load_state_dict(torch.load("model/lstm_pinn2D.pth", map_location=device))
model.eval()

# 4) Build (x,y,t) grid
x = np.linspace(x_lb, x_ub, Nx)
y = np.linspace(y_lb, y_ub, Ny)
t = np.linspace(t_lb, t_ub, Nt)

# meshgrid with 'ij' so Xg[i,j,k]=x[i], Yg[i,j,k]=y[j], Tg[i,j,k]=t[k]
Xg, Yg, Tg = np.meshgrid(x, y, t, indexing='ij')   # shapes (Nx,Ny,Nt)

# flatten into list of points
pts = Xg.ravel(), Yg.ravel(), Tg.ravel()
x_flat = torch.tensor(pts[0], dtype=torch.float32).view(-1,1).to(device)
y_flat = torch.tensor(pts[1], dtype=torch.float32).view(-1,1).to(device)
t_flat = torch.tensor(pts[2], dtype=torch.float32).view(-1,1).to(device)

# evaluate
with torch.no_grad():
    u_flat = model(x_flat, y_flat, t_flat).cpu().numpy()

# reshape back → (Nx,Ny,Nt), then transpose to (Nt,Nx,Ny)
U_xyz = u_flat.reshape(Nx, Ny, Nt)
U = np.transpose(U_xyz, (2,0,1))  # U[t_idx, i_x, i_y]

# --- Plotting functions ---

def plot_colormap(t_idx=0):
    """Static 2D heatmap at time index t_idx."""
    plt.figure(figsize=(6,4))
    plt.pcolormesh(x, y, U[t_idx].T, shading='auto', cmap='viridis')
    plt.colorbar(label='u(x,y)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'u(x,y) at t = {t[t_idx]:.3f}')
    plt.tight_layout()
    plt.show()

def plot_surface(t_idx=0):
    """Static 3D surface at time index t_idx."""
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    X2d, Y2d = np.meshgrid(x, y, indexing='xy')  # (Ny,Nx)
    Z = U[t_idx].T                               # (Ny,Nx)
    surf = ax.plot_surface(X2d, Y2d, Z, cmap='viridis')
    fig.colorbar(surf, ax=ax, label='u')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
    ax.set_title(f'u(x,y) at t = {t[t_idx]:.3f}')
    plt.tight_layout()
    plt.show()

def animate_colormap(save_path="burgers2d_colormap.gif"):
    """Animate 2D heatmap over time, save as GIF."""
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.pcolormesh(x, y, U[0].T, shading='auto', cmap='viridis')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'u(x,y) at t={t[0]:.3f}')
    cbar = fig.colorbar(im, ax=ax, label='u')

    def update(frame):
        im.set_array(U[frame].T.ravel())
        ax.set_title(f'u(x,y) at t={t[frame]:.3f}')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=100, blit=True)
    ani.save(save_path, writer='imagemagick')  # or 'ffmpeg'
    plt.close(fig)

def animate_surface(save_path="burgers2d_surface.mp4"):
    """Animate 3D surface over time, save as MP4."""
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    X2d, Y2d = np.meshgrid(x, y, indexing='xy')
    surf = [ax.plot_surface(X2d, Y2d, U[0].T, cmap='viridis')]
    ax.set_zlim(U.min(), U.max())
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')

    def update(frame):
        ax.collections.clear()  # remove old surface
        surf[0] = ax.plot_surface(X2d, Y2d, U[frame].T, cmap='viridis')
        ax.set_title(f'u(x,y) at t={t[frame]:.3f}')
        return surf

    ani = animation.FuncAnimation(fig, update, frames=Nt, interval=100, blit=False)
    ani.save(save_path, writer='ffmpeg', dpi=150)
    plt.close(fig)

# --- Loss plotting (as before) ---
import numpy as _np
def plot_losses():
    loss_dict = _np.load("loss/lossLSTM.npy", allow_pickle=True).item()
    plt.figure(figsize=(6,4))
    plt.plot(loss_dict['data'], label='Data Loss')
    plt.plot(loss_dict['pde'],  label='PDE Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend(); plt.tight_layout(); plt.show()

# --- Main ---
if __name__ == "__main__":
    # static
    plot_colormap(t_idx=0)
    plot_surface(t_idx=0)
    # animations
    animate_colormap("burgers2d_colormap.gif")
    animate_surface("burgers2d_surface.mp4")
    # loss curves
    plot_losses()
