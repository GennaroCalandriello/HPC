import torch
import torch.nn as nn
import numpy as np
import os
import hyperpar as hp

"""Sto usando un modello LSTM per risolvere l'eq. di Burgers in 1D"""
#Disabilito cose:
torch.backends.cudnn.enabled = False # Disable CuDNN for RNN double-backward support


#Hyperparameters & device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nu = hp.NU
hidden_size = hp.HIDDEN_SIZE
num_layers =  hp.NUM_LAYERS
lr = hp.LR
epochs = hp.EPOCHS
N_data = hp.N_SAMPLES
N_colloc = hp.N_COLLOCATION
x_lb, x_ub = -1.0, 1.0
t_lb, t_ub = 0.0, 1.0

#Data & collocation sampling
x_data = torch.linspace(x_lb, x_ub, N_data).view(-1, 1).to(device)
t_data = torch.zeros_like(x_data).to(device)
u_data = torch.sin(np.pi*x_data).to(device)

x_coll = (x_ub-x_lb) * torch.rand(N_colloc, 1) +x_lb
t_coll = (t_ub-t_lb) * torch.rand(N_colloc,1) +t_lb
x_coll, t_coll = x_coll.to(device), t_coll.to(device)

#LSTM-PINN
class LSTM_PINN(nn.Module):
    def __init__(self, input_size = 2, hidden_size =50, num_layers = num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)
        
    
    def forward(self, x, t):
        seq = torch.cat([x, t], dim = -1).unsqueeze(1) # [B,1,2]
        out, _ = self.lstm(seq)
        u_pred = self.fc(out[:, -1, :])
        return u_pred

model = LSTM_PINN(input_size = 2, hidden_size = hidden_size, num_layers = num_layers).to(device)

#PDE residual via autograd for BURGERS
def pde_res(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs = torch.ones_like(u), create_graph = True)[0] #create_graph=True retains the results, and allows to compute higher order derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph = True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones_like(u_x), create_graph=True)[0]
    
    return u_t +u*u_x -nu*u_xx

#loss function
mse = nn.MSELoss()
def compute_loss():
    #PDE
    L_data = mse(model(x_data, t_data), u_data)
    r = pde_res(model, x_coll, t_coll)
    L_pde = mse(r, torch.zeros_like(r))
    
    return L_data, L_pde
    
def train():
    history = {"total": [], "data": [], "pde": []}
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    for e in range(1, epochs+1):
        optimizer.zero_grad()
        loss_data, loss_pde = compute_loss()
        loss = hp.LAMBDA_DATA * loss_data + hp.LAMBDA_PDE * loss_pde
        loss.backward()
        optimizer.step()
        
        if e % 20 == 0:
            print(f"Epoch {e}/{epochs}, Loss data: {loss.item():.4e}")
        history["total"].append(loss.item())
        history["data"].append(loss_data.item())
        history["pde"].append(loss_pde.item())
    #Salviamo
    if not os.path.exists("model"):
        os.makedirs("model")
    torch.save(model.state_dict(), "model/lstm_pinn.pth")
    if not os.path.exists("loss"):
        os.makedirs("loss")
    np.save("loss/lossLSTM_DisPINN.npy", history)

if __name__ == "__main__":
    train()
    print("Training complete.")
    
    

        