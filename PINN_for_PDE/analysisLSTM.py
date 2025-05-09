import torch
import numpy as np
import matplotlib.pyplot as plt
from modelLSTM import LSTM_PINN
import hyperpar as hp
from matplotlib.animation import FuncAnimation
import os

# Allow duplicate OpenMP libs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Limit threads
torch.set_num_threads(1)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load the LSTM-PINN model
model = LSTM_PINN(hp.NX, hp.SEQ_LEN, hp.HIDDEN_SIZE, hp.NUM_LAYERS).to(device)
model.load_state_dict(torch.load("models/modelLSTM.pth", map_location=device))
model.eval()

# Create spatial-temporal grid
nx, nt = hp.NX, hp.NT*2
x = np.linspace(0, 1, nx)
t = np.linspace(0, 0.5, nt)
X, T = np.meshgrid(x, t)

# Prepare input sequences for static prediction:
# We need for each time step t_i a preceding sequence of length SEQ_LEN.
# For simplicity, use zero initial history (or repeat boundary IC) 
# and roll forward via model.
U_pred = np.zeros((nt, nx), dtype=np.float32)
# initial condition for t[0] is sin(pi x)
U_pred[0, :] = np.sin(np.pi * x)

# Roll forward using LSTM-PINN
for i in range(1, nt):
    # build sequence of last SEQ_LEN snapshots
    start = max(0, i - hp.SEQ_LEN)
    hist = U_pred[start:i]
    # if not enough history, pad with initial
    if hist.shape[0] < hp.SEQ_LEN:
        pad = np.tile(U_pred[0], (hp.SEQ_LEN - hist.shape[0], 1))
        hist = np.vstack([pad, hist])
    seq_u = torch.tensor(hist[np.newaxis,...], device=device)  # [1, seq_len, nx]
    with torch.no_grad():
        U_pred[i] = model(seq_u).cpu().numpy()

def staticPlot():
    plt.figure(figsize=(8,6))
    pcm = plt.pcolormesh(X, T, U_pred, shading='auto', cmap='viridis')
    plt.colorbar(pcm, label='u')
    plt.xlabel('x'); plt.ylabel('t'); plt.title('LSTM-DisPINN Prediction')
    plt.tight_layout()
    plt.show()

def animatePlot():
    fig, ax = plt.subplots()
    line, = ax.plot(x, U_pred[0,:], 'r-')
    ax.set_xlim(0,1); ax.set_ylim(U_pred.min(), U_pred.max())
    title = ax.text(0.5,1.05,'', ha='center')

    def update(i):
        line.set_ydata(U_pred[i,:])
        title.set_text(f't = {t[i]:.3f}')
        print(f"Frame {i+1}/{nt}", end='\r')
        return line, title

    ani = FuncAnimation(fig, update, frames=nt, interval=100, blit=True)
    plt.show()

def plot_loss():
    history = np.load("lossLSTM_DisPINN.npy", allow_pickle=True).item()
    plt.figure(figsize=(8,6))
    plt.plot(history["total"], label="Total Loss")
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss')
    plt.legend(); plt.grid()
    plt.show()

if __name__ == "__main__":
    staticPlot()
    animatePlot()
    # plot_loss()
