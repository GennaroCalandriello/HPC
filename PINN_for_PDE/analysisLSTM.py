import torch
import numpy as np
import matplotlib.pyplot as plt
from modelLSTM import LSTM_PINN
import hyperpar as hp
from matplotlib.animation import FuncAnimation

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

#device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the trained model
model = LSTM_PINN(hp.NX, hp.SEQ_LEN, hp.HIDDEN_SIZE, hp.NUM_LAYERS).to(device)
model.load_state_dict(torch.load("modelLSTM.pth", map_location = device))
model.eval()

#Spacetime grid
nx, nt = hp.NX, hp.NT
x_lin = np.linspace(0, 1, nx)
t_lin = np.linspace(0, 0.5, nt)

def analytic_snap(t):
    return np.sin(np.pi*x_lin)*np.exp(-np.pi**2*hp.NU*t)

ground_truth = np.stack([analytic_snap(t) for t in t_lin], axis = 0)

#predicted field by recurrence
pred = np.zeros((nt, nx), dtype = np.float32)
#seeding with the first SEQ_LEN snapshots
pred[:hp.SEQ_LEN] = ground_truth[:hp.SEQ_LEN]

with torch.no_grad():
    for i in range(hp.SEQ_LEN, nt):
        #grab the last SEQ_LEN frames, shape (SEQ_LEN,nx)
        #to tensor of shape (1,SEQ_LEN,nx)
        seq = pred[i-hp.SEQ_LEN : i] #last SEQ_LEN frames
        seq_t = torch.tensor(seq[None, :, :], device = device, dtype = torch.float32)
        #prediction
        next_u = model(seq_t).cpu().numpy().ravel()
        pred[i] = next_u

def staticPlot():
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(x_lin, t_lin, pred, shading='auto', cmap='viridis')
    fig.colorbar(pcm, ax=ax, label='u(x,t)')
    ax.set_xlabel('x'); ax.set_ylabel('t')
    ax.set_title('LSTM‐PINN predicted field')
    plt.tight_layout()
    plt.show()

def animatePlot():
   fig, ax = plt.subplots()
   line, = ax.plot(x_lin, pred[0], 'r-', lw=2)
   ax.set_xlim(0, 1)
   ax.set_ylim(pred.min(), pred.max())
   ax.set_xlabel('x')
   ax.set_ylabel('u(x,t)')
   title = ax.text(0.5, 1.05, "", ha = "center")
   
   def update(k):
       line.set_ydata(pred[k])
       title.set_text(f"t = {k*hp.DT:.2f}")
       return line, title
   
   ani = FuncAnimation(fig, update, frames = nt, interval = 50, blit = True)
   plt.show()
   
if __name__ == "__main__":
    staticPlot()
    animatePlot()

        
