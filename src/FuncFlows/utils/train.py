import torch

# literally just the standard training loop except you feed in the objective
    # currently just ReverseKL which works via a class method i.e. you just call the thing
def train(objective, parameters, num_steps=5000, learning_rate=0.01, decay=0.8, decay_every=500):

    optimiser = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, decay_every, decay)


    losses = []
    for _ in range(num_steps):
        optimiser.zero_grad()
        loss = objective()
        loss.backward()
        optimiser.step(); scheduler.step()
        losses.append(loss.item())

    
    return losses