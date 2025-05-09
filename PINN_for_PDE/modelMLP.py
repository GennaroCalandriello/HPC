import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hyperpar as hp
import os
#Device seting
device = torch.device("cuda" if torch.cuda .is_available() else "cpu")

"""This is a simple implementation of a Physics Informed Neural Network (PINN) for solving the 1D Burgers' equation.
It is based on a feedforward Multilayer Perceptron (MLP) architecture. The PINN is trained to minimize the residual of the PDE, as well as the boundary and initial conditions."""
#PINN class

    
#============================================================= PINN with MLP ========================================================= 
class PINN(nn.Module):

    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        #Xavier initialization = inizializza i pesi della rete
        for x in self.layers:
            nn.init.xavier_normal_(x.weight)
            nn.init.zeros_(x.bias)
    
    def forward(self, x, t):
        """Forward pass through the network"""
        xt = torch.cat([x,t], dim =1) #shape [batch, 2]
        b = xt
        for lyr in self.layers[:-1]:
            b = torch.tanh(lyr(b))
        u = self.layers[-1](b)
        return u

#Calcoliamo i residui per equazione di Burgers: u_t + u u_x - nu u_xx = 0
def pde_res(model, x, t, nu):
    """Compute the PDE residual with automatic differentiation. Here some informations about network
    ♪ create_graph = True allows to compute the gradient of the residual w.r.t. the input
    ♪
    ♪
    ♪"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs = torch.ones_like(u), create_graph = True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs = torch.ones_like(u_x), create_graph = True)[0]

    return u_t-u*u_x+nu*u_xx

#Generate collocation and BC/IC points
def generate_training_data(N_c, N_bc, N_ic):
    """Generate collocation and boundary/initial points"""

    #Domain: x in [0, 1], t in [0, 0.5]
    x_c = np.random.rand(N_c, 1)
    t_c = np.random.rand(N_c, 1)*0.5

    #Boundary points: x = 0 and x = 1, random t
    t_bc = np.random.rand(N_bc, 1)*0.5
    x_bc = np.vstack([np.zeros((N_bc//2, 1)), np.ones((N_bc//2, 1))])

    #Initial condition t =0, random x
    x_ic = np.random.rand(N_ic, 1)
    t_ic = np.zeros((N_ic, 1))
    u_ic = np.sin(np.pi*x_ic) #example: u(x,0) = sin(pi x)

    #convert to tensors
    x_c = torch.tensor(x_c, dtype = torch.float32, device = device)
    t_c  = torch.tensor(t_c, dtype = torch.float32, device = device)
    x_bc = torch.tensor(x_bc, dtype = torch.float32, device = device)
    t_bc = torch.tensor(t_bc, dtype = torch.float32, device = device)
    x_ic = torch.tensor(x_ic, dtype = torch.float32, device = device)
    t_ic = torch.tensor(t_ic, dtype = torch.float32, device = device)
    u_ic = torch.tensor(u_ic, dtype = torch.float32, device = device)

    return x_c, t_c, x_bc, t_bc, x_ic, t_ic, u_ic

def trainMLP():
    """Train the PINN"""
    print("Training the PINN...")
    history = {"total_loss": [], "pde_loss": [], "bc_loss":[], "ic_loss":[]}
    #Hyperparameters from hyperpar.py
    layers = hp.LAYERS
    nu = hp.NU
    lr = hp.LR
    epochs = hp.EPOCHS
    N_c, N_bc, N_ic = hp.N_INT, hp.N_BC, hp.N_IC
    
    #Model, optimizer, loss
    model = PINN(layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr) #lr = learning rate smaller lr = slower but more precise
    mse = nn.MSELoss() #Mean Squared Error loss function

    #prepare training data 
    x_c, t_c, x_bc, t_bc, x_ic, t_ic, u_ic = generate_training_data(N_c, N_bc, N_ic)

    #training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        #Here I write all the losses associated to the PDE, BC and IC

        #PDE residual loss
        res = pde_res(model, x_c, t_c, nu)
        loss_pde = mse(res, torch.zeros_like(res))

        #boundary loss (u(0,t) = u(1,t) = 0)
        u_bc = model(x_bc, t_bc)
        loss_bc = mse(u_bc, torch.zeros_like(u_bc))

        #initial condition loss
        u0 = model(x_ic, t_ic)
        loss_ic = mse(u0, u_ic)

        #total loss
        loss = loss_pde + loss_bc +loss_ic
        loss.backward()
        optimizer.step()

        #recording the losses
        history["total_loss"].append(loss.item())
        history["pde_loss"].append(loss_pde.item())
        history["bc_loss"].append(loss_bc.item())
        history["ic_loss"].append(loss_ic.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}, PDE Loss: {loss_pde.item():.4e}, BC Loss: {loss_bc.item():.4e}, IC Loss: {loss_ic.item():.4e}")

    #write the model to a file
    if (hp.SAVE_MODEL):
        #elimina il file se esiste
        try:
            os.remove("model.pth")
        except OSError:
            pass
        torch.save(model.state_dict(), "model.pth")
        print("Model saved to model.pth")

    #Save the velocity field
    if(hp.SAVE_U):
        #elimina il file se esiste
        try:
            os.remove("u.npy")
        except OSError:
            pass
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        t = np.linspace(0, 0.5, 100).reshape(-1, 1)
        x = torch.tensor(x, dtype = torch.float32, device = device)
        t = torch.tensor(t, dtype = torch.float32, device = device)
        u = model(x, t).cpu().detach().numpy()
        np.save("u.npy", u)
        print("Velocity field saved to u.npy")

    #Save the losses
    if (hp.SAVE_LOSS):
        #elimina il file se esiste
        try:
            os.remove("loss.npy")
        except OSError:
            pass
        np.save("loss.npy", history)
        print("Losses saved to loss.npy")
    
if __name__ == "__main__":

    trainMLP()




