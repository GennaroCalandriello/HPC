# before training
history = {"total": [], "pde": [], "bc": [], "ic": []}

for epoch in range(epochs):
    optimizer.zero_grad()

    # compute your losses
    loss_pde = ...
    loss_bc  = ...
    loss_ic  = ...
    loss = loss_pde + loss_bc + loss_ic

    loss.backward()
    optimizer.step()

    # record them
    history["total"].append(loss.item())
    history["pde"].append(  loss_pde.item())
    history["bc"].append(   loss_bc.item() )
    history["ic"].append(   loss_ic.item() )

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: total={loss.item():.3e}")
