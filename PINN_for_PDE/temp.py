import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Disable CuDNN for RNN double-backward support
torch.backends.cudnn.enabled = False
# Allow duplicate OpenMP libs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 1. Hyperparameters and device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu          = 0.01
hidden_size = 50
num_layers  = 1
lr          = 1e-3
epochs      = 2000
N_data      = 200
N_colloc    = 10000
x_lb, x_ub  = -1.0, 1.0
t_lb, t_ub  =  0.0, 1.0

# 2. Data & collocation sampling
x_data = torch.linspace(x_lb, x_ub, N_data).view(-1,1).to(device)
t_data = torch.zeros_like(x_data).to(device)
u_data = torch.sin(np.pi * x_data).to(device)  # placeholder IC

x_coll = (x_ub - x_lb) * torch.rand(N_colloc,1) + x_lb
t_coll = (t_ub - t_lb) * torch.rand(N_colloc,1) + t_lb
x_coll, t_coll = x_coll.to(device), t_coll.to(device)

# 3. LSTM-PINN definition
class LSTMPINN(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x, t):
        seq = torch.cat([x, t], dim=-1).unsqueeze(1)  # [B,1,2]
        out, _ = self.lstm(seq)                      # → [B,1,hidden_size]
        u_pred = self.fc(out[:, -1, :])              # → [B,1]
        return u_pred

model = LSTMPINN(input_size=2, hidden_size=hidden_size, num_layers=num_layers).to(device)

# 4. PDE residual via autograd
def pde_residual(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(x, t)
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx

# 5. Loss function
mse = nn.MSELoss()
def compute_loss():
    L_data = mse(model(x_data, t_data), u_data)
    r      = pde_residual(model, x_coll, t_coll)
    L_phys = mse(r, torch.zeros_like(r))
    return L_data + L_phys

def train():
    # 6. Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch:04d} | Loss = {loss.item():.3e}")

    # Save model
    if not os.path.exists("model"):
        os.makedirs("model")
    torch.save(model.state_dict(), "model/lstm_pinn.pth")
    if not os.path.exists("loss"):
        os.makedirs("loss")
    np.save("loss/lossLSTM_DisPINN.npy", loss.item())
    # 7. Plot prediction at t=0.5


if __name__ == "__main__":
    train()