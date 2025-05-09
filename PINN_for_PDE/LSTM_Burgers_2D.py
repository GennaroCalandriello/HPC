import os
import torch
import torch.nn as nn
import numpy as np
import hyperpar as hp

# 0) Disable CuDNN for double-backward through LSTM
torch.backends.cudnn.enabled = False

# 1) Hyperparameters & device
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nu          = hp.NU
hidden_size = hp.HIDDEN_SIZE
num_layers  = hp.NUM_LAYERS
lr          = hp.LR
epochs      = hp.EPOCHS
N_data      = hp.N_SAMPLES
N_colloc    = hp.N_COLLOCATION
λ_data      = hp.LAMBDA_DATA
λ_pde       = hp.LAMBDA_PDE

# Domain bounds
x_lb, x_ub = -1.0, 1.0
y_lb, y_ub = -1.0, 1.0
t_lb, t_ub =  0.0, 1.0

# 2) Sample initial‐condition points (t=0) uniformly in 2D
x_data = (x_ub - x_lb) * torch.rand(N_data,1) + x_lb
y_data = (y_ub - y_lb) * torch.rand(N_data,1) + y_lb
t_data = torch.zeros_like(x_data)
u_data = torch.sin(np.pi * x_data) * torch.sin(np.pi * y_data)

x_data, y_data, t_data, u_data = [t.to(device) for t in (x_data,y_data,t_data,u_data)]

# 3) Sample collocation points in the full domain
x_coll = (x_ub - x_lb) * torch.rand(N_colloc,1) + x_lb
y_coll = (y_ub - y_lb) * torch.rand(N_colloc,1) + y_lb
t_coll = (t_ub - t_lb) * torch.rand(N_colloc,1) + t_lb
x_coll, y_coll, t_coll = [t.to(device) for t in (x_coll,y_coll,t_coll)]

# 4) LSTM-PINN model for (x,y,t)→u
class LSTM_PINN(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x, y, t):
        # cat → [B,3] → add seq-dim → [B,1,3]
        seq = torch.cat([x,y,t], dim=-1).unsqueeze(1)
        out,_ = self.lstm(seq)               # → [B,1,hidden]
        return self.fc(out[:, -1, :])        # → [B,1]

model = LSTM_PINN(3, hidden_size, num_layers).to(device)

# 5) PDE residual for 2D Burgers: u_t + u*(u_x+u_y) - ν*(u_xx+u_yy)
def pde_res(model, x, y, t):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(x,y,t)
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_y  = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                create_graph=True)[0]
    return u_t + u*(u_x + u_y) - nu*(u_xx + u_yy)

# 6) Loss function
mse = nn.MSELoss()
def compute_losses():
    # data loss @ t=0
    u_pred_data = model(x_data, y_data, t_data)
    Ld = mse(u_pred_data, u_data)
    # physics loss over collocation
    r  = pde_res(model, x_coll, y_coll, t_coll)
    Lp = mse(r, torch.zeros_like(r))
    # weighted
    return λ_data * Ld + λ_pde * Lp, Ld, Lp

# 7) Training loop
def train():
    history = {"total":[], "data":[], "pde":[]}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(1, epochs+1):
        optimizer.zero_grad()
        loss, Ld, Lp = compute_losses()
        loss.backward()
        optimizer.step()

        history["total"].append(loss.item())
        history["data"].append(Ld.item())
        history["pde"].append(Lp.item())

        if e % 20 == 0:
            print(f"Epoch {e:4d}/{epochs}  "
                  f"Loss={loss.item():.3e}  "
                  f"Data={Ld.item():.3e}  "
                  f"PDE ={Lp.item():.3e}")

    # ensure dirs
    os.makedirs("model", exist_ok=True)
    os.makedirs("loss",  exist_ok=True)
    # save
    torch.save(model.state_dict(), "model/lstm_pinn2D.pth")
    np.save("loss/lossLSTM_2D.npy", history)
    print("✔️  Training complete, model + history saved.")

if __name__ == "__main__":
    train()
